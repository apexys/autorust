//! `codegen/simplify.py` port: range flattening and reduce simplification
//! used by the rangeify pipeline (M4).

use crate::dtype::{self, ConstVal};
use crate::uop::ops::{group, Ops};
use crate::uop::sym::{r, sget, sget_opt, var, CtxVal};
use crate::uop::sint::SVal;
use crate::uop::upat::{PatternMatcher, RuleFn, Store, UPat, UPatOrConst};
use crate::uop::{arg::Arg, ctx::UOpCtx, UopId};
use std::collections::HashMap;
use std::sync::Arc;

fn red_add_arg() -> Arg {
    Arg::Seq { tuple: true, items: vec![Arg::Enum(crate::uop::arg::EnumVal { class: crate::uop::arg::EnumClass::Ops, value: Ops::Add as u16 }), Arg::Int(0)] }
}

/// `no_range(u)` — no RANGE in the backward slice (with self).
fn no_range(ctx: &mut UOpCtx, u: UopId) -> bool {
    !ctx.op_in_backward_slice_with_self(u, &[Ops::Range])
}

/// `flatten_range` (simplify.py:8).
pub fn flatten_range(ctx: &mut UOpCtx, rr: UopId) -> Option<UopId> {
    let (op, dt, srcs, arg, _tag) = ctx.snapshot(rr);
    let off: usize = match op {
        Ops::Reduce | Ops::End | Ops::Stage => 1,
        _ => return None,
    };
    let rngs = srcs[off..].to_vec();
    if rngs.is_empty() {
        return None;
    }
    // ranges in the cond should not be ended
    let mut backedge = Vec::new();
    let mut rest = Vec::new();
    for &x in &rngs {
        let d = ctx.dt_of(x);
        if d == dtype::VOID || d == dtype::BOOL {
            backedge.push(x);
        } else {
            rest.push(x);
        }
    }
    let sink = crate::tensor::sink(ctx, rest.clone());
    let flat = ctx.ranges(sink).to_vec();
    let mut new_srcs = srcs[..off].to_vec();
    new_srcs.extend(flat);
    new_srcs.extend(backedge);
    Some(ctx.uop(op, Some(dt), new_srcs, arg))
}

/// `pm_flatten_range` — real ranges only.
pub fn pm_flatten_range() -> PatternMatcher {
    PatternMatcher::new(vec![(
        UPat::op_var(vec![Ops::Reduce, Ops::End]).named("r"),
        r(|ctx, st, _| flatten_range(ctx, sget(st, "r"))),
    )])
}

/// `reduce_unparented` (simplify.py:118).
pub fn reduce_unparented(ctx: &mut UOpCtx, red: UopId) -> Option<UopId> {
    let (op, dt, srcs, arg, _tag) = ctx.snapshot(red);
    // arg[0] in {ADD, MAX, MUL}
    let rop: Ops = match &arg {
        Arg::Seq { tuple: true, items } => match &items[0] {
            Arg::Enum(e) if e.class == crate::uop::arg::EnumClass::Ops => match Ops::from_u16(e.value) {
                Some(o) => o,
                None => return None,
            },
            _ => return None,
        },
        _ => return None,
    };
    if !matches!(rop, Ops::Add | Ops::Max | Ops::Mul) {
        return None;
    }
    let s0 = srcs[0];
    assert!(srcs[1..].iter().all(|&x| ctx.op_of(x) == Ops::Range), "TGERR: some reduce srcs aren't ranges");
    let in_ranges = ctx.ranges(s0).to_vec();
    let mut parented = Vec::new();
    let mut unparented = Vec::new();
    for &x in &srcs[1..] {
        if in_ranges.contains(&x) {
            parented.push(x);
        } else {
            unparented.push(x);
        }
    }
    if unparented.is_empty() {
        return None;
    }
    let mut ret: UopId = if !parented.is_empty() || dt != ctx.dt_of(s0) {
        let mut new_srcs = vec![s0];
        new_srcs.extend(parented.iter().copied());
        ctx.uop(op, Some(dt), new_srcs, arg)
    } else {
        s0
    };
    match rop {
        Ops::Add => {
            for r in &unparented {
                let end = ctx.srcs(*r)[0];
                ret = ctx.mul(ret, end);
            }
        }
        Ops::Mul => {
            for r in &unparented {
                let end = ctx.srcs(*r)[0];
                ret = ctx.pow(ret, end);
            }
        }
        _ => {}
    }
    Some(ret)
}

/// `pm_reduce_unparented`.
pub fn pm_reduce_unparented() -> PatternMatcher {
    PatternMatcher::new(vec![(
        UPat::op_var(vec![Ops::Reduce]).named("red"),
        r(|ctx, st, _| reduce_unparented(ctx, sget(st, "red"))),
    )])
}

/// `reduce_collapse` (simplify.py:154) — for REDUCE(ADD, 0) with srcs, try to
/// lift the reduce out of each range by substituting input VALUES with fresh
/// variables, rewriting the collapse, and splicing back.
pub fn reduce_collapse(ctx: &mut UOpCtx, red: UopId, pm: &PatternMatcher) -> Option<UopId> {
    let rngs = ctx.srcs(red)[1..].to_vec();
    let mut u = ctx.srcs(red)[0];
    for &r in &rngs {
        // included = u.toposort(gate=lambda x: r in x.ranges)
        let ranges_of_r = ctx.ranges(r).to_vec();
        let included = ctx.toposort_gated(u, Some(&|x| ranges_of_r.contains(&x)), true);
        if included.iter().any(|&x| matches!(ctx.op_of(x), Ops::Store | Ops::Reduce)) {
            return None;
        }
        let mut replaces: HashMap<UopId, UopId> = HashMap::new();
        let mut rev: Vec<(UopId, UopId)> = Vec::new();
        for &inc in &included {
            let srcs = ctx.srcs(inc).to_vec();
            for s in srcs {
                if included.contains(&s) || replaces.contains_key(&s) {
                    continue;
                }
                if matches!(ctx.op_of(s), Ops::Const | Ops::Param | Ops::Buffer) {
                    continue;
                }
                let (vmin, vmax) = ctx.min_max(s).ok()?;
                let name = format!("in{}", rev.len());
                let vmin_c = sval_to_constval(&vmin);
                let vmax_c = sval_to_constval(&vmax);
                let v = ctx.variable(&name, vmin_c, vmax_c, ctx.dt_of(s), true);
                replaces.insert(s, v);
                rev.push((v, s));
            }
        }
        // collapse_fxn = u.substitute(replaces).reduce(r, arg=Ops.ADD)
        let subbed = crate::uop::upat::substitute(ctx, u, replaces, true, true);
        let collapse = ctx.uop(Ops::Reduce, None, vec![subbed, r], red_add_arg());
        let sink = match crate::uop::upat::graph_rewrite(ctx, collapse, pm, false, None, false, false, CtxVal::None) {
            Ok(s) => s,
            Err(_) => return None,
        };
        if !no_range(ctx, sink) {
            return None;
        }
        // u = sink.substitute({v: k for k, v in replaces.items()})
        let mut back: HashMap<UopId, UopId> = HashMap::new();
        for (v, s) in &rev {
            back.insert(*v, *s);
        }
        u = crate::uop::upat::substitute(ctx, sink, back, true, true);
    }
    Some(u)
}

/// `pm_reduce_collapse` = unparented + the lift rules (+symbolic).
pub fn pm_reduce_collapse() -> PatternMatcher {
    let mut rules: Vec<(UPat, RuleFn)> = vec![];
    // lift x+y out of reduce on lt: (x+y).or_casted() < c  ->  x < (c-y)
    rules.push((
        var("x").add(UPatOrConst::Pat(var("y"))).or_casted(None).lt(UPatOrConst::Pat(var("c"))),
        r(|ctx, st, _| {
            let (x, y, c) = (sget(st, "x"), sget(st, "y"), sget(st, "c"));
            if !no_range(ctx, y) || !no_range(ctx, c) {
                return None;
            }
            let cy = ctx.sub(c, y);
            Some(ctx.lt(x, cy))
        }),
    ));
    // lift x*y out of reduce: (x*y) < c  ->  x < (c+y-1)//y  (int y, vmin>0)
    rules.push((
        var("x").mul(UPatOrConst::Pat(var("y"))).lt(UPatOrConst::Pat(var("c"))),
        r(|ctx, st, _| {
            let (x, y, c) = (sget(st, "x"), sget(st, "y"), sget(st, "c"));
            if !no_range(ctx, y) || !no_range(ctx, c) {
                return None;
            }
            let ydt = ctx.dt_of(y);
            if !crate::dtype::is_int(ydt) {
                return None;
            }
            let (vmin, _) = ctx.min_max(y).ok()?;
            if vmin.as_i128_or_panic() <= 0 {
                return None;
            }
            let cy = ctx.add(c, y);
            let one = ctx.const_like(c, CV_num(1));
            let cy1 = ctx.add(cy, one);
            let div = ctx.alu2(Ops::Floordiv, cy1, y);
            Some(ctx.lt(x, div))
        }),
    ));
    // sum over r in [0,N) of gated val -> clamp * val
    let r_rng = UPat::op_var(vec![Ops::Range]).named("r");
    let alt1 = r_rng.clone().lt(UPatOrConst::Pat(var("upper"))).where_(UPatOrConst::Pat(var("val")), UPatOrConst::Val(ConstVal::Int(0)));
    let alt2 = r_rng.clone().lt(UPatOrConst::Pat(var("lower"))).where_(UPatOrConst::Val(ConstVal::Int(0)), UPatOrConst::Pat(var("val")));
    let alt3 = var("r")
        .lt(UPatOrConst::Pat(var("lower")))
        .logical_not()
        .and_(UPatOrConst::Pat(r_rng.clone().lt(UPatOrConst::Pat(var("upper")))))
        .where_(UPatOrConst::Pat(var("val")), UPatOrConst::Val(ConstVal::Int(0)));
    rules.push((
        UPat::any(vec![alt1, alt2, alt3]).reduce(vec![var("r")], Some(red_add_arg()), false),
        r(|ctx, st, _| {
            let (r, val) = (sget(st, "r"), sget(st, "val"));
            let lower = sget_opt(st, "lower");
            let upper = sget_opt(st, "upper");
            if !no_range(ctx, val) {
                return None;
            }
            let rend = ctx.srcs(r)[0];
            let hi = match upper {
                Some(u) => ctx.minimum(u, rend),
                None => rend,
            };
            let lo = match lower {
                Some(l) => {
                    let zero = ctx.const_like(l, CV_num(0));
                    ctx.maximum(l, zero)
                }
                None => ctx.const_like(rend, CV_num(0)),
            };
            let width = ctx.sub(hi, lo);
            let zero = ctx.const_like(lo, CV_num(0));
            let wmax = ctx.maximum(width, zero);
            let clamped = ctx.minimum(wmax, rend);
            Some(ctx.mul(clamped, val))
        }),
    ));
    // invalid_gate.reduce(ADD) -> cond.where(x.reduce(...), i)
    rules.push((
        crate::uop::sym::invalid_gate().reduce(vec![], Some(red_add_arg()), true).named("r"),
        r(|ctx, st, _| {
            let (cond, x, i, rr) = (sget(st, "cond"), sget(st, "x"), sget(st, "i"), sget(st, "r"));
            if !no_range(ctx, cond) {
                return None;
            }
            let rngs = ctx.srcs(rr)[1..].to_vec();
            let mut v = vec![x];
            v.extend(rngs.iter().copied());
            let xr = ctx.uop(Ops::Reduce, None, v, red_add_arg());
            Some(ctx.where_(cond, xr, i))
        }),
    ));
    // (x+y).reduce(ADD) -> x.reduce(...) + y.reduce(...)
    rules.push((
        var("x").add(UPatOrConst::Pat(var("y"))).reduce(vec![], Some(red_add_arg()), true).named("r"),
        r(|ctx, st, _| {
            let (x, y, rr) = (sget(st, "x"), sget(st, "y"), sget(st, "r"));
            let rngs = ctx.srcs(rr)[1..].to_vec();
            let rx = ctx.uop(Ops::Reduce, None, {
                let mut v = vec![x];
                v.extend(rngs.iter().copied());
                v
            }, red_add_arg());
            let ry = ctx.uop(Ops::Reduce, None, {
                let mut v = vec![y];
                v.extend(rngs.iter().copied());
                v
            }, red_add_arg());
            Some(ctx.add(rx, ry))
        }),
    ));
    // (PARAM(x) & y).where(c, 0).reduce(ADD) -> y.where(c, 0).reduce(...) * x
    rules.push((
        UPat::ops1(Ops::Param)
            .named("x")
            .and_(UPatOrConst::Pat(var("y")))
            .where_(UPatOrConst::Pat(var("c")), UPatOrConst::Val(ConstVal::Int(0)))
            .reduce(vec![], Some(red_add_arg()), true)
            .named("r"),
        r(|ctx, st, _| {
            let (x, y, c, rr) = (sget(st, "x"), sget(st, "y"), sget(st, "c"), sget(st, "r"));
            let rngs = ctx.srcs(rr)[1..].to_vec();
            let zero = ctx.const_like(y, CV_num(0));
            // python: y.where(c, 0).reduce(*r.src[1:], arg=Ops.ADD)*x
            let yw = ctx.where_(y, c, zero);
            let mut v2 = vec![yw];
            v2.extend(rngs.iter().copied());
            let yr2 = ctx.uop(Ops::Reduce, None, v2, red_add_arg());
            let _ = x;
            Some(ctx.mul(yr2, x))
        }),
    ));
    // x * gate(bool).cast() -> gate.where(x, 0)
    rules.push((
        var("x").mul(UPatOrConst::Pat(crate::uop::sym::var_dt("gate", vec![dtype::BOOL]).cast(None))),
        r(|ctx, st, _| {
            let (x, gate) = (sget(st, "x"), sget(st, "gate"));
            let zero = ctx.const_like(x, CV_num(0));
            Some(ctx.where_(gate, x, zero))
        }),
    ));
    let base = PatternMatcher::new(rules);
    pm_reduce_unparented().add(&base).add(crate::uop::sym::symbolic())
}

fn CV_num(v: i64) -> ConstVal {
    ConstVal::Int(v as i128)
}

/// `pm_reduce_simplify` (simplify.py:170) — remove REDUCE without loads.
pub fn pm_reduce_simplify() -> PatternMatcher {
    let reduce_collapse_rule: (UPat, RuleFn) = (
        UPat::ops1(Ops::Reduce)
            .with_arg(Some(red_add_arg()))
            .with_src_list(vec![var("u")])
            .named("red"),
        r(|ctx, st, _| {
            let red = sget(st, "red");
            let pm = pm_reduce_collapse();
            let u = ctx.srcs(red)[0];
            reduce_collapse(ctx, red, &pm).map(|ret| {
                let _ = u;
                ret
            })
        }),
    );
    pm_reduce_unparented().add(&PatternMatcher::new(vec![reduce_collapse_rule]))
}


fn sval_to_constval(s: &SVal) -> ConstVal {
    match s {
        SVal::I(i) => ConstVal::Int(*i),
        _ => panic!("TGERR: symbolic bound on substitute variable"),
    }
}
