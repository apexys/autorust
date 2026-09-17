//! Port of `uop/divandmod.py` — `fold_divmod_general` + the fast inline rules.
//!
//! NOTE: Python caches `fold_divmod_general` on index UOps (`functools.cache`);
//! we recompute per call (it is a pure function of the node).


use super::ctx::{sval_cmp, UOpCtx};
use super::graph::UopId;
use super::ops::{group, Ops};
use super::alu::{floordiv, py_mod};
use super::sym::{cvar, cvar_arg, r, sget, var, weak_int_dts};
use super::sint::SVal;
use super::upat::{simplify, CtxVal, PatternMatcher, RuleFn, Store, UPat, UPatOrConst};
use crate::dtype::ConstVal as CV;

/// Python `sum(items)` over UOps: starts at int 0, so the first `+` is
/// `0 + t1` → `ADD(CONST(0), t1)` (via `__radd__`/reverse binop), then the
/// normal left fold. Returns None only for the empty sum (Python int 0).
fn py_sum(ctx: &mut UOpCtx, items: &[UopId]) -> Option<UopId> {
    let mut it = items.iter().cloned();
    let first = it.next()?;
    let zero = ctx.const_py(crate::dtype::ConstVal::Int(0), None);
    // 0 + t1 with reverse=True: (y, self) = (CONST(0), t1), broadcasted
    let (a, b) = ctx.broadcasted(zero, first);
    let mut acc = ctx.alu2(Ops::Add, a, b);
    for t in it {
        acc = ctx.add(acc, t);
    }
    Some(acc)
}

/// `sum(items) + c` where the Python int 0 start also participates.
fn py_sum_plus(ctx: &mut UOpCtx, items: &[UopId], c: UopId) -> UopId {
    match py_sum(ctx, items) {
        Some(s) => ctx.add(s, c),
        None => {
            // sum([]) == int 0; `0 + c` via __radd__ → ADD(CONST(0), c)
            let zero = ctx.const_py(crate::dtype::ConstVal::Int(0), None);
            let (a, b) = ctx.broadcasted(zero, c);
            ctx.alu2(Ops::Add, a, b)
        }
    }
}

/// `fold_divmod_general` (divandmod.py:8-96).
pub fn fold_divmod_general(ctx: &mut UOpCtx, d: UopId) -> Option<UopId> {
    let dop = ctx.op_of(&d);
    let srcs = ctx.srcs(&d).to_vec();
    let (x, y) = ((srcs[0]).clone(), (srcs[1]).clone());

    let (ymin, ymax) = ctx.min_max(&y).ok()?;
    if ymin.clone().as_i128_or_panic() == 0 && ymax.as_i128_or_panic() == 0 {
        panic!("TGERR: division by zero trying to rewrite");
    }

    // x//y is constant
    let xdiv = ctx.div_floor(x.clone(), y.clone());
    if let Ok((xdmin, xdmax)) = ctx.min_max(&xdiv)
        && xdmin == xdmax {
            let c = ctx.const_like(x.clone(), CV::Int(xdmin.as_i128_or_panic()));
            if dop == Ops::Floormod {
                let m = ctx.mul(c, y);
                return Some(ctx.sub(x, m));
            }
            return Some(c);
        }

    // PARAM // c is irreducible
    if ctx.op_of(&x) == Ops::Param && ctx.op_of(&y) == Ops::Const
        && let Some(p) = ctx.has_param(&x) {
            let m = p.multiple_of.unwrap_or(1) as i128;
            let yv = ctx.val(&y).as_int_exact()?;
            if m % yv == 0 && dop == Ops::Floormod {
                return Some(ctx.const_like(d, CV::Int(0)));
            }
            if m % yv == 0 {
                return None;
            }
        }

    // split uops
    let (x_peeled, const_v) = ctx.pop_const(x.clone(), Ops::Add);
    let uops_no_const = ctx.split_uop_vec(x_peeled.clone(), Ops::Add);
    let const_i = const_v.as_int_exact().unwrap_or(0);

    // ** Constant Denominator Rules ** (y a scalar const > 0)
    if ctx.op_of(&y) == Ops::Const {
        let c = ctx.val(&y).as_int_exact()?;
        if c > 0 {
            // nested_div: (x%(k*c))//c -> (x//c)%k
            if dop == Ops::Floordiv && ctx.op_of(&x) == Ops::Floormod {
                let k = ctx.divides(ctx.srcs(&x)[1].clone(), c);
                if let Some(kv) = k.as_ref().and_then(|k| {
                    if ctx.op_of(k) == Ops::Const {
                        ctx.val(k).as_int_exact()
                    } else {
                        None
                    }
                })
                    && kv > 0 {
                        let first = ctx.div_floor(ctx.srcs(&x)[0].clone(), y);
                        let kconst = ctx.const_like(x, CV::Int(kv));
                        return Some(ctx.mod_(first, kconst));
                    }
            }

            // remove_nested_mod in sum: (a%4 + b)%2 -> (a+b)%2
            if dop == Ops::Floormod {
                let mut new_xs: Vec<UopId> = Vec::new();
                let mut changed = false;
                for u in &uops_no_const {
                    let mut u2 = u.clone();
                    if ctx.op_of(u) == Ops::Floormod && ctx.divides(ctx.srcs(u)[1].clone(), c).is_some() {
                        u2 = ctx.srcs(u)[0].clone();
                        changed = true;
                    }
                    new_xs.push(u2);
                }
                if changed {
                    // UOp.usum(*new_xs): reduce add over [x0, rest...] — NO leading zero
                    let sum = match new_xs.len() {
                        0 => return None,
                        1 => new_xs[0].clone(),
                        _ => {
                            let first = new_xs[0].clone();
                            ctx.usum(first, &new_xs[1..])
                        }
                    };
                    let with_const = if const_i == 0 { sum } else { let kc = ctx.const_like(x, const_v); ctx.add(sum, kc) };
                    return Some(ctx.mod_(with_const, y));
                }
            }

            // shared decomposition
            let mut terms: Vec<UopId> = Vec::new();
            let mut factors: Vec<i128> = Vec::new();
            for u in &uops_no_const {
                let f = ctx.const_factor(u.clone());
                factors.push(f);
                terms.push(ctx.divides(u.clone(), f).unwrap_or(u.clone()));
            }

            // fold_divmod_congruence: try both signs of the remainder for a lone
            // term or an exact tie; else the smaller (by abs)
            let rem_choices: Vec<Vec<i128>> = factors
                .iter()
                .map(|&f| {
                    let rr = py_mod(f, c);
                    if rr * 2 == c || terms.len() == 1 {
                        vec![rr, rr - c]
                    } else if rr.abs() <= (rr - c).abs() {
                        vec![rr]
                    } else {
                        vec![rr - c]
                    }
                })
                .collect();

            // iterate the cartesian product
            let mut idx = vec![0usize; rem_choices.len()];
            'product: loop {
                let rems: Vec<i128> = idx.iter().enumerate().map(|(i, &j)| rem_choices[i][j]).collect();
                // rem = sum(r*v for r,v in zip(rems, terms)) + const%c  (as a node)
                let mut parts: Vec<UopId> = Vec::new();
                let mut sum_i = py_mod(const_i, c);
                let mut all_int = true;
                for (i, &r_) in rems.iter().enumerate() {
                    let v = terms[i].clone();
                    if ctx.op_of(&v) == Ops::Const {
                        sum_i = sum_i.wrapping_add(r_.wrapping_mul(ctx.val(&v).as_int_exact().unwrap_or(0)));
                    } else {
                        all_int = false;
                        // `r*v`: int * UOp → MUL(const, v)
                        let rv = ctx.const_like(v.clone(), CV::Int(r_));
                        parts.push(ctx.mul(rv, v));
                    }
                }
                let rem_expr: UopId = if all_int {
                    ctx.const_like(x.clone(), CV::Int(sum_i))
                } else {
                    // sum(parts) + const%c  (builtin sum: leading CONST(0))
                    let cst = ctx.const_like(x.clone(), CV::Int(py_mod(const_i, c)));
                    py_sum_plus(ctx, &parts, cst)
                };
                if let Ok((rmin, rmax)) = ctx.min_max(&rem_expr) {
                    let (rmin_i, rmax_i) = (rmin.as_i128_or_panic(), rmax.as_i128_or_panic());
                    if floordiv(rmin_i, c) == floordiv(rmax_i, c) {
                        if dop == Ops::Floormod {
                            let off = ctx.const_like(x, CV::Int(floordiv(rmin_i, c).wrapping_mul(c)));
                            return Some(ctx.sub(rem_expr, off));
                        } else {
                            // sum((f-r)//c * v) + const//c + rmin//c — zero terms KEPT
                            let mut qparts: Vec<UopId> = Vec::new();
                            for (i, &r_) in rems.iter().enumerate() {
                                let f = factors[i];
                                let q = floordiv(f - r_, c);
                                let v = terms[i].clone();
                                let qn = ctx.const_like(v.clone(), CV::Int(q));
                                qparts.push(ctx.mul(qn, v));
                            }
                            let tail1 = ctx.const_like(x.clone(), CV::Int(floordiv(const_i, c)));
                            let tail2 = ctx.const_like(x, CV::Int(floordiv(rmin_i, c)));
                            let with_tail1 = py_sum_plus(ctx, &qparts, tail1);
                            return Some(ctx.add(with_tail1, tail2));
                        }
                    }
                }
                // advance the product index
                let mut i = rem_choices.len();
                loop {
                    if i == 0 {
                        break 'product;
                    }
                    i -= 1;
                    idx[i] += 1;
                    if idx[i] < rem_choices[i].len() {
                        break;
                    }
                    idx[i] = 0;
                }
            }

            // gcd_with_remainder: factor out common gcd from numerator
            let mut gvals = factors.clone();
            gvals.push(c);
            let g = gcd_all(&gvals);
            if g > 1 {
                let xg = ctx.divides(x_peeled, g)?;
                let new_x = simplify(ctx, xg);
                let cg = ctx.const_like(x.clone(), CV::Int(floordiv(const_i, g)));
                let cg2 = ctx.const_like(x.clone(), CV::Int(c / g));
                let cg = ctx.mod_(cg, cg2);
                let new_x = ctx.add(new_x, cg);
                let (nmin, _) = ctx.min_max(&new_x).ok()?;
                if nmin.as_i128_or_panic() >= 0 {
                    if dop == Ops::Floormod {
                        let cdiv_g = ctx.const_like(x.clone(), CV::Int(c / g));
                        let m1 = ctx.mod_(new_x, cdiv_g);
                        let gk = ctx.const_like(x.clone(), CV::Int(g));
                        let m2 = ctx.mul(m1, gk);
                        let m3 = ctx.const_like(x, CV::Int(py_mod(const_i, g)));
                        return Some(ctx.add(m2, m3));
                    } else {
                        let cdiv_g2 = ctx.const_like(x.clone(), CV::Int(c / g));
                        let q1 = ctx.div_floor(new_x, cdiv_g2);
                        let q2 = ctx.const_like(x, CV::Int(floordiv(const_i, c)));
                        return Some(ctx.add(q1, q2));
                    }
                }
            }

            // nest_by_factor: x//c -> (x//f)//(c//f); x%c -> (x//f%(c//f))*f + b
            let mut results: Vec<(usize, UopId)> = Vec::new();
            let mut divs_seen: Vec<i128> = Vec::new();
            for (u, f) in uops_no_const.iter().zip(factors.iter()) {
                let af = f.abs();
                if ctx.op_of(u) != Ops::Const && 1 < af && af < c && c % af == 0 && !divs_seen.contains(&af) {
                    divs_seen.push(af);
                }
            }
            for &div in &divs_seen {
                // (x//div) — build then recurse
                let dv = ctx.const_like(x.clone(), CV::Int(div));
                let xdiv2 = ctx.div_floor(x.clone(), dv);
                if let Some(newxs) = fold_divmod_general(ctx, xdiv2) {
                    if dop == Ops::Floordiv {
                        let bsn = ctx.toposort(&newxs).len();
                        let cdd = ctx.const_like(x.clone(), CV::Int(c / div));
                        let q = ctx.div_floor(newxs, cdd);
                        results.push((bsn, q));
                    } else {
                        let (xmin, _) = ctx.min_max(&x).ok()?;
                        let (nxmin, _) = ctx.min_max(&newxs).ok()?;
                        if xmin.as_i128_or_panic() >= 0 && nxmin.as_i128_or_panic() >= 0 {
                            // b = sum(f%div * t for f,t if f%div) + const%div
                            let mut b_parts: Vec<UopId> = Vec::new();
                            for (i, &f) in factors.iter().enumerate() {
                                if py_mod(f, div) != 0 {
                                    let t = terms[i].clone();
                                    let fk = ctx.const_like(t.clone(), CV::Int(py_mod(f, div)));
                                    let m = ctx.mul(fk, t);
                                    b_parts.push(m);
                                }
                            }
                            if py_mod(const_i, div) != 0 {
                                b_parts.push(ctx.const_like(x.clone(), CV::Int(py_mod(const_i, div))));
                            }
                            // UOp.usum(*b_parts): reduce with first as start (no leading zero)
                            let b = match b_parts.len() {
                                0 => ctx.const_like(x.clone(), CV::Int(0)),
                                1 => b_parts[0].clone(),
                                _ => {
                                    let first = b_parts[0].clone();
                                    ctx.usum(first, &b_parts[1..])
                                }
                            };
                            let (bmin, bmax) = ctx.min_max(&b).ok()?;
                            if bmin.as_i128_or_panic() >= 0 && bmax.as_i128_or_panic() < div {
                                let cdd2 = ctx.const_like(x.clone(), CV::Int(c / div));
                                let inner = ctx.mod_(newxs, cdd2);
                                let divk = ctx.const_like(x.clone(), CV::Int(div));
                                let inner = ctx.mul(inner, divk);
                                let r = ctx.add(inner, b);
                                let bsn = ctx.toposort(&r).len();
                                results.push((bsn, r));
                            }
                        }
                    }
                }
            }
            if !results.is_empty() {
                let best = results.iter().min_by_key(|(n, _)| *n).unwrap();
                return Some(best.1.clone());
            }
        }
    }

    // ** Variable Denominator / Fallback Rules **
    let all_uops = ctx.split_uop_vec(x.clone(), Ops::Add);
    // divide_by_gcd: x//y -> (x//gcd)//(y//gcd)
    let mut gu = all_uops.clone();
    gu.push(y.clone());
    let gcd = ctx.uop_gcd(&gu);
    let gcd_s = simplify(ctx, gcd);
    let gcd_is_one = ctx.op_of(&gcd_s) == Ops::Const && ctx.val(&gcd_s) == CV::Int(1);
    if !gcd_is_one {
        let xq = ctx.divide_exact(x, gcd_s.clone())?;
        let yq = ctx.divide_exact(y, gcd_s.clone())?;

        let ret = ctx.alu2(dop, xq, yq);
        return Some(if dop == Ops::Floormod { ctx.mul(ret, gcd_s) } else { ret });
    }

    // factor_remainder: (d*x+y)//d -> x+y//d
    let (xmin, _) = ctx.min_max(&x).ok()?;
    if ymin.as_i128_or_panic() < 0 || xmin.as_i128_or_panic() < 0 {
        return None;
    }
    let mut quo: Vec<UopId> = Vec::new();
    let mut rem: Vec<UopId> = Vec::new();
    for u in &all_uops {
        match ctx.divide_exact(u.clone(), y.clone()) {
            Some(q) => quo.push(q),
            None => {
                if ctx.op_of(&y) == Ops::Const {
                    let cy = ctx.val(&y).as_int_exact()?;
                    let cf = ctx.const_factor(u.clone());
                    if py_mod(cf, cy) != cf {
                        let uc = ctx.divides(u.clone(), cf)?;
                        { let rk = ctx.const_like(u.clone(), CV::Int(py_mod(cf, cy))); rem.push(ctx.mul(uc, rk)); }
                        if dop == Ops::Floordiv {
                            let qc = ctx.divides(u.clone(), cf)?;
                            { let qk = ctx.const_like(u.clone(), CV::Int(floordiv(cf, cy))); quo.push(ctx.mul(qc, qk)); }
                        } else {
                            quo.push(ctx.const_like(u.clone(), CV::Int(0)));
                        }
                    } else {
                        rem.push(u.clone());
                    }
                } else {
                    rem.push(u.clone());
                }
            }
        }
    }
    if quo.is_empty() {
        return None;
    }
    // new_x = sum(rem) + x.const_like(0)
    let zero = ctx.const_like(x.clone(), CV::Int(0));
    let rem_sum = py_sum_plus(ctx, &rem, zero);
    let (newmin, _) = ctx.min_max(&rem_sum).ok()?;
    if newmin.as_i128_or_panic() < 0 {
        return None;
    }
    if dop == Ops::Floormod {
        Some(ctx.mod_(rem_sum, y))
    } else {
        let q = ctx.div_floor(rem_sum, y);
        let zero2 = ctx.const_like(x, CV::Int(0));
        let qsum = py_sum_plus(ctx, &quo, zero2);
        Some(ctx.add(q, qsum))
    }
}

fn gcd_all(vals: &[i128]) -> i128 {
    let g = |a: i128, b: i128| {
        let (mut a, mut b) = (a.abs(), b.abs());
        while b != 0 {
            let t = a % b;
            a = b;
            b = t;
        }
        a
    };
    vals.iter().cloned().fold(0, g)
}

pub fn div_and_mod_symbolic() -> &'static PatternMatcher {
    PM.get_or_init(|| {
        let rules: Vec<(UPat, RuleFn)> = vec![
            // (x//c+a)//d -> (x+a*c)//(c*d) for c>0, d>0
            (
                var("x").floordiv(UPatOrConst::Pat(cvar("c"))).add(UPatOrConst::Pat(cvar_arg("a", CV::Int(0)))).floordiv(UPatOrConst::Pat(cvar("d"))),
                r(|ctx, st, _| {
                    let (x, c, a, d) = (sget(st, "x"), sget(st, "c"), sget(st, "a"), sget(st, "d"));
                    let (dmin, _) = ctx.min_max(&d).ok()?;
                    if dmin.as_i128_or_panic() > 0 {
                        let cv = ctx.val(&c).as_int_exact()?;
                        let av = ctx.val(&a).as_int_exact()?;
                        let dv = ctx.val(&d).as_int_exact()?;
                        let ik = ctx.const_like(x.clone(), CV::Int(av * cv));
                        let inner = ctx.add(x.clone(), ik);
                        let dd = ctx.const_like(x, CV::Int(cv * dv));
                        return Some(ctx.div_floor(inner, dd));
                    }
                    None
                }),
            ),
            // (x+c)//d -> (x+c%d)//d + c//d ; (x+c)%d -> (x+c%d)%d
            (
                UPat::op_var(vec![Ops::Floordiv, Ops::Floormod])
                    .with_src_tup(vec![
                        var("x").with_dtype(Some(weak_int_dts())).add(UPatOrConst::Pat(cvar("c"))),
                        cvar("d"),
                    ])
                    .named("n"),
                r(|ctx, st, _| {
                    let (n, x, c, d) = (sget(st, "n"), sget(st, "x"), sget(st, "c"), sget(st, "d"));
                    let dv = ctx.val(&d).as_int_exact()?;
                    let cv = ctx.val(&c).as_int_exact()?;
                    if dv == 0 || py_mod(cv, dv) == cv {
                        return None;
                    }
                    let ck = ctx.const_like(x.clone(), CV::Int(py_mod(cv, dv)));
                    let x2 = ctx.add(x.clone(), ck);
                    if ctx.op_of(&n) == Ops::Floordiv {
                        let q = ctx.div_floor(x2, d);
                        let k = ctx.const_like(x, CV::Int(floordiv(cv, dv)));
                        Some(ctx.add(q, k))
                    } else {
                        Some(ctx.mod_(x2, d))
                    }
                }),
            ),
            // ** slow rules **
            (
                UPat::op_var(vec![Ops::Floordiv, Ops::Floormod]).with_dtype(Some(weak_int_dts())).named("d"),
                r(|ctx, st, _| fold_divmod_general(ctx, sget(st, "d"))),
            ),
        ];
        PatternMatcher::new(rules)
    })
}

#[allow(unused)]
fn type_keepers(_: &SVal, _: &CtxVal, _: &Store, _: fn(&UOpCtx, UopId, UopId) -> std::cmp::Ordering) {
    let _ = sval_cmp;
    let _ = group::UNARY;
}

static PM: crate::uop::upat::PmSlot = crate::uop::upat::PmSlot::new();
