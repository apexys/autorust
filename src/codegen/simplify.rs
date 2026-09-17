//! `codegen/simplify.py` port: range flattening and reduce simplification
//! used by the rangeify pipeline (M4).

use crate::dtype::{self, ConstVal};
use crate::uop::ops::Ops;
use crate::uop::sym::{r, sget, sget_opt, symbolic, var, CtxVal};
use crate::uop::sint::SVal;
use crate::uop::range::AxisType;
use crate::uop::upat::{graph_rewrite, substitute, PatternMatcher, RuleFn, Store, UPat, UPatOrConst};
use crate::uop::{arg::Arg, ctx::UOpCtx, UopId};
use std::collections::HashMap;
use std::sync::Arc;

pub fn red_add_arg() -> Arg {
    Arg::Seq { tuple: true, items: vec![Arg::Enum(crate::uop::arg::EnumVal { class: crate::uop::arg::EnumClass::Ops, value: Ops::Add as u16 }), Arg::Int(0)] }
}

/// `no_range(u)` — no RANGE in the backward slice (with self).
fn no_range(ctx: &mut UOpCtx, u: UopId) -> bool {
    !ctx.op_in_backward_slice_with_self(&u, &[Ops::Range])
}

/// `flatten_range` (simplify.py:8).
pub fn flatten_range(ctx: &mut UOpCtx, rr: UopId) -> Option<UopId> {
    let (op, dt, srcs, arg, _tag) = ctx.snapshot(&rr);
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
    for x in &rngs {
        let d = ctx.dt_of(x);
        if d == dtype::VOID || d == dtype::BOOL {
            backedge.push(x.clone());
        } else {
            rest.push(x.clone());
        }
    }
    let sink = crate::tensor::sink(ctx, rest.clone());
    let flat = ctx.ranges(&sink).to_vec();
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

fn const_i(ctx: &UOpCtx, u: UopId) -> i128 {
    match ctx.val(&u) {
        ConstVal::Int(i) => i,
        other => panic!("TGERR: expected int const, got {other:?}"),
    }
}

/// `count_divmod` (simplify.py:22).
fn count_divmod(ctx: &UOpCtx, u: UopId) -> usize {
    ctx.backward_slice(&u)
        .iter()
        .filter(|&x| matches!(ctx.op_of(x), Ops::Floordiv | Ops::Floormod))
        .count()
}

/// `simplify_merge_adjacent` (simplify.py:23).
pub fn simplify_merge_adjacent(ctx: &mut UOpCtx, u0: UopId) -> Option<UopId> {
    let er = ctx.ended_ranges(&u0.clone());
    if er.is_empty() || !er.iter().all(|r| ctx.op_of(r) == Ops::Range) {
        return None;
    }
    let bs = ctx.backward_slice_with_self(&u0);
    let reds: Vec<UopId> = bs.iter().filter(|&x| ctx.op_of(x) == Ops::Reduce).cloned().collect();
    let reduce_ranges: Vec<std::rc::Rc<Vec<UopId>>> = reds.into_iter().map(|x| ctx.ranges(&x)).collect();
    // on END we only want to merge adjacent ranges, on REDUCE we want to try all combinations
    let pairs: Vec<(UopId, UopId)> = if ctx.op_of(&u0) == Ops::End {
        er.iter().zip(er[1..].iter()).map(|(a, b)| (a.clone(), b.clone())).collect()
    } else {
        let mut p = Vec::new();
        for i in 0..er.len() {
            for j in 0..er.len() {
                if i != j {
                    p.push(((er[i]).clone(), (er[j]).clone()));
                }
            }
        }
        p
    };
    let mut cur = u0.clone();
    for (r0, r1) in pairs {
        if AxisType::range_axis_type(ctx, r0.clone()) != AxisType::range_axis_type(ctx, r1.clone()) {
            continue;
        }
        // check if the ranges to merge are in the same reduces
        if !reduce_ranges.iter().all(|rr| rr.contains(&r0) == rr.contains(&r1)) {
            continue;
        }
        let s0 = &ctx.srcs(&r0)[0].clone();
        let s1 = &ctx.srcs(&r1)[0].clone();
        let prod = ctx.mul(s0.clone(), s1.clone());
        let new_range =
            ctx.replace(&r0, Ops::Range, ctx.dt_of(&r0), vec![prod], ctx.arg_of(&r0).clone(), ctx.tag_of(&r0).clone());
        let d = ctx.div_floor(new_range.clone(), s1.clone());
        let m = ctx.mod_(new_range, s1.clone());
        let mut map = HashMap::new();
        map.insert(r0, d);
        map.insert(r1, m);
        let nidx = substitute(ctx, cur.clone(), map, false, false);
        let pmf = symbolic().add(&pm_flatten_range());
        let nidx = graph_rewrite(ctx, nidx, &pmf, false, None, false, false, CtxVal::None)
            .unwrap_or_else(|e| panic!("TGERR: merge rewrite failed: {e}"));
        // check if it simplifies
        if count_divmod(ctx, nidx.clone()) <= count_divmod(ctx, cur.clone()) {
            cur = nidx;
        }
    }
    if cur == u0 {
        None
    } else {
        Some(cur)
    }
}

/// `pm_simplify_ranges` merge rule (simplify.py:54; gating done by `range_gates`).
pub fn pm_simplify_ranges() -> PatternMatcher {
    PatternMatcher::new(vec![(
        UPat::op_var(vec![Ops::End, Ops::Reduce]).named("u"),
        r(|ctx, st, _| simplify_merge_adjacent(ctx, sget(st, "u"))),
    )])
}

/// `mark_gated` (simplify.py:43) — collected over the graph by `range_gates`.
fn mark_gated(ctx: &mut UOpCtx, idx: UopId, g: &mut HashMap<UopId, UopId>) {
    let srcs = ctx.srcs(&idx).to_vec();
    let mut guards: HashMap<UopId, UopId> = HashMap::new();
    let x;
    if srcs.len() > 1 && ctx.op_of(&srcs[1]) == Ops::Where {
        let ws = ctx.srcs(&srcs[1]).to_vec();
        let cond = ws[0].clone();
        let stmts = ctx.split_uop_vec(cond, Ops::And);
        for v in &stmts {
            if ctx.op_of(v) == Ops::Cmplt {
                let vs = ctx.srcs(v);
                if ctx.op_of(&vs[0]) == Ops::Range && ctx.op_of(&vs[1]) == Ops::Const {
                    guards.insert(vs[0].clone(), vs[1].clone());
                }
            }
        }
        x = (ws[1]).clone();
    } else {
        x = idx;
    }
    for (r, c) in &guards {
        match g.get(r) {
            Some(cur) => {
                if const_i(ctx, cur.clone()) < const_i(ctx, c.clone()) {
                    g.insert(r.clone(), c.clone());
                }
            }
            None => {
                g.insert(r.clone(), c.clone());
            }
        }
    }
    for r in ctx.ranges(&x).iter() {
        if !guards.contains_key(r) && ctx.op_of(r) == Ops::Range {
            let s0 = ctx.srcs(r)[0].clone();
            g.insert(r.clone(), s0);
        }
    }
}

/// `pm_simplify_ranges` gating half (mark_gated + REDUCE pin + SINK substitute).
pub fn range_gates(ctx: &mut UOpCtx, sink: UopId) -> UopId {
    let mut g: HashMap<UopId, UopId> = HashMap::new();
    let order = ctx.toposort(&sink);
    for x in order.iter().rev() {
        match ctx.op_of(x) {
            Ops::Index => mark_gated(ctx, x.clone(), &mut g),
            Ops::Reduce => {
                let srcs = ctx.srcs(x).to_vec();
                for r in &srcs[1..] {
                    if ctx.op_of(r) == Ops::Range {
                        let s0 = ctx.srcs(r)[0].clone();
                        g.insert(r.clone(), s0);
                    }
                }
            }
            _ => {}
        }
    }
    let mut map: HashMap<UopId, UopId> = HashMap::new();
    let entries: Vec<(UopId, UopId)> = g.into_iter().collect();
    for (r, c) in entries {
        if ctx.srcs(&r)[0] != c {
            let nr = ctx.replace(&r, Ops::Range, ctx.dt_of(&r), vec![c], ctx.arg_of(&r).clone(), ctx.tag_of(&r).clone());
            map.insert(r, nr);
        }
    }
    if map.is_empty() {
        return sink;
    }
    let ret = substitute(ctx, sink, map, false, false);
    crate::uop::upat::simplify(ctx, ret)
}

/// `mark_range_mod` (simplify.py:62).
fn mark_range_mod(ctx: &UOpCtx, r: UopId, c: UopId, g: &mut HashMap<UopId, UopId>) {
    if g.contains_key(&r) {
        return;
    }
    // ranges that aren't looped over can't be split
    let at = AxisType::range_axis_type(ctx, r.clone());
    if at == AxisType::Warp || at == AxisType::Device {
        return;
    }
    let s0 = ctx.srcs(&r)[0].clone();
    if ctx.op_of(&s0) != Ops::Const {
        return;
    }
    let (n, cv) = match (ctx.val(&s0), ctx.val(&c)) {
        (ConstVal::Int(n), ConstVal::Int(v)) => (n, v),
        _ => return,
    };
    // `r.src[0].divides(c.val) is not None`
    if cv != 0 && n % cv == 0 {
        g.insert(r, c);
    }
}

/// `pm_split_ranges` split fn (simplify.py:75): replace a marked range `k` with
/// `range(len//v, axis, 0, ty)*v + range(v, axis, 1, ty)`.
fn split_range(ctx: &mut UOpCtx, r: UopId, c: UopId) -> UopId {
    let arg = ctx.arg_of(&r).clone();
    let Arg::Seq { tuple: true, items } = arg else { panic!("TGERR: bad range arg") };
    let head = items[..items.len() - 1].to_vec();
    let tail = items.last().unwrap().clone();
    let n = ctx.srcs(&r)[0].clone();
    let dt = ctx.dt_of(&r);
    let mk = |ctx: &mut UOpCtx, end: UopId, split: i64| {
        let mut it = head.clone();
        it.push(Arg::Int(split));
        it.push(tail.clone());
        ctx.uop(Ops::Range, Some(dt), vec![end], Arg::Seq { tuple: true, items: it })
    };
    let outer_end = ctx.div_floor(n, c.clone());
    let outer = mk(ctx, outer_end, 0);
    let inner = mk(ctx, c.clone(), 1);
    let oc = ctx.mul(outer, c);
    ctx.add(oc, inner)
}

/// `do_substitute` (simplify.py:67) for `pm_split_ranges`.
fn do_split_substitute(ctx: &mut UOpCtx, sink: UopId, g: &mut HashMap<UopId, UopId>) -> Option<UopId> {
    if g.is_empty() {
        return None;
    }
    let entries: Vec<(UopId, UopId)> = g.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    g.clear();
    let mut map: HashMap<UopId, UopId> = HashMap::new();
    for (r, c) in entries {
        map.insert(r.clone(), split_range(ctx, r, c));
    }
    let ret = substitute(ctx, sink.clone(), map, false, false);
    if ret == sink {
        return None;
    }
    Some(crate::uop::upat::simplify(ctx, ret))
}

/// `pm_split_ranges` (simplify.py:72) — split a range that is used modulo a
/// constant whose multiple it is (`range % c` with `len % c == 0`).
pub fn pm_split_ranges() -> PatternMatcher {
    PatternMatcher::new(vec![
        (
            UPat::ops1(Ops::Range).named("r").mod_(UPat::cvar(Some("c"), None, None)),
            crate::uop::upat::r_dict(|g, ctx, st| {
                mark_range_mod(ctx, sget(st, "r"), sget(st, "c"), g);
                None
            }),
        ),
        (
            UPat::ops1(Ops::Sink).named("x"),
            crate::uop::upat::r_dict(|g, ctx, st| do_split_substitute(ctx, sget(st, "x"), g)),
        ),
    ])
}

/// `reduce_unparented` (simplify.py:118).
pub fn reduce_unparented(ctx: &mut UOpCtx, red: UopId) -> Option<UopId> {
    let (op, dt, srcs, arg, _tag) = ctx.snapshot(&red);
    // arg[0] in {ADD, MAX, MUL}
    let rop: Ops = match &arg {
        Arg::Seq { tuple: true, items } => match &items[0] {
            Arg::Enum(e) if e.class == crate::uop::arg::EnumClass::Ops => Ops::from_u16(e.value)?,
            _ => return None,
        },
        _ => return None,
    };
    if !matches!(rop, Ops::Add | Ops::Max | Ops::Mul) {
        return None;
    }
    let s0 = &srcs[0].clone();
    assert!(srcs[1..].iter().all(|x| ctx.op_of(x) == Ops::Range), "TGERR: some reduce srcs aren't ranges");
    let in_ranges = ctx.ranges(&s0.clone()).to_vec();
    let mut parented = Vec::new();
    let mut unparented = Vec::new();
    for x in &srcs[1..] {
        if in_ranges.contains(x) {
            parented.push(x);
        } else {
            unparented.push(x);
        }
    }
    if unparented.is_empty() {
        return None;
    }
    let mut ret: UopId = if !parented.is_empty() || dt != ctx.dt_of(s0) {
        let mut new_srcs = vec![s0.clone()];
        new_srcs.extend(parented.iter().map(|x| (*x).clone()));
        ctx.uop(op, Some(dt), new_srcs, arg)
    } else {
        s0.clone()
    };
    match rop {
        Ops::Add => {
            for r in &unparented {
                let end = &ctx.srcs(r)[0].clone();
                ret = ctx.mul(ret, end.clone());
            }
        }
        Ops::Mul => {
            for r in &unparented {
                let end = &ctx.srcs(r)[0].clone();
                ret = ctx.pow(ret, end.clone());
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
    let rngs = ctx.srcs(&red)[1..].to_vec();
    let mut u = ctx.srcs(&red)[0].clone();
    for r in &rngs {
        // included = u.toposort(gate=lambda x: r in x.ranges)
        // precompute ranges for every node below u (the gate needs an immutable ctx)
        let mut rng_map: std::collections::HashMap<UopId, std::rc::Rc<Vec<UopId>>> = std::collections::HashMap::new();
        for nd in ctx.toposort(&u).iter() {
            rng_map.insert(nd.clone(), ctx.ranges(&nd.clone()));
        }
        let included = ctx.toposort_gated(u.clone(), Some(&|x| rng_map.get(&x).is_some_and(|v| v.contains(r))), true);
        if included.iter().any(|x| matches!(ctx.op_of(x), Ops::Store | Ops::Reduce)) {
            return None;
        }
        let mut replaces: HashMap<UopId, UopId> = HashMap::new();
        let mut rev: Vec<(UopId, UopId)> = Vec::new();
        for inc in &included {
            let srcs = ctx.srcs(inc).to_vec();
            for s in srcs {
                if included.contains(&s) || replaces.contains_key(&s) {
                    continue;
                }
                if matches!(ctx.op_of(&s), Ops::Const | Ops::Param | Ops::Buffer) {
                    continue;
                }
                let (vmin, vmax) = ctx.min_max(&s).ok()?;
                let name = format!("in{}", rev.len());
                let vmin_c = sval_to_constval(&vmin);
                let vmax_c = sval_to_constval(&vmax);
                let v = ctx.variable(&name, vmin_c, vmax_c, ctx.dt_of(&s), true);
                replaces.insert(s.clone(), v.clone());
                rev.push((v, s));
            }
        }
        // collapse_fxn = u.substitute(replaces).reduce(r, arg=Ops.ADD)
        let subbed = crate::uop::upat::substitute(ctx, u, replaces, true, true);
        let collapse = ctx.uop(Ops::Reduce, None, vec![subbed, r.clone()], red_add_arg());
        let sink = match crate::uop::upat::graph_rewrite(ctx, collapse, pm, false, None, false, false, CtxVal::None) {
            Ok(s) => s,
            Err(_) => return None,
        };
        if !no_range(ctx, sink.clone()) {
            return None;
        }
        // u = sink.substitute({v: k for k, v in replaces.items()})
        let mut back: HashMap<UopId, UopId> = HashMap::new();
        for (v, s) in &rev {
            back.insert(v.clone(), s.clone());
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
            if !no_range(ctx, y.clone()) || !no_range(ctx, c.clone()) {
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
            if !no_range(ctx, y.clone()) || !no_range(ctx, c.clone()) {
                return None;
            }
            let ydt = ctx.dt_of(&y);
            if !crate::dtype::is_int(ydt) {
                return None;
            }
            let (vmin, _) = ctx.min_max(&y).ok()?;
            if vmin.as_i128_or_panic() <= 0 {
                return None;
            }
            let cy = ctx.add(c.clone(), y.clone());
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
            if !no_range(ctx, val.clone()) {
                return None;
            }
            let rend = &ctx.srcs(&r)[0].clone();
            let hi = match upper {
                Some(u) => ctx.minimum(u, rend.clone()),
                None => rend.clone(),
            };
            let lo = match lower {
                Some(l) => {
                    let zero = ctx.const_like(l.clone(), CV_num(0));
                    ctx.maximum(l, zero)
                }
                None => ctx.const_like(rend.clone(), CV_num(0)),
            };
            let width = ctx.sub(hi, lo.clone());
            let zero = ctx.const_like(lo, CV_num(0));
            let wmax = ctx.maximum(width, zero);
            let clamped = ctx.minimum(wmax, rend.clone());
            Some(ctx.mul(clamped, val))
        }),
    ));
    // invalid_gate.reduce(ADD) -> cond.where(x.reduce(...), i)
    rules.push((
        crate::uop::sym::invalid_gate().reduce(vec![], Some(red_add_arg()), true).named("r"),
        r(|ctx, st, _| {
            let (cond, x, i, rr) = (sget(st, "cond"), sget(st, "x"), sget(st, "i"), sget(st, "r"));
            if !no_range(ctx, cond.clone()) {
                return None;
            }
            let rngs = ctx.srcs(&rr)[1..].to_vec();
            let mut v = vec![x];
            v.extend(rngs.iter().cloned());
            let xr = ctx.uop(Ops::Reduce, None, v, red_add_arg());
            Some(ctx.where_(cond, xr, i))
        }),
    ));
    // (x+y).reduce(ADD) -> x.reduce(...) + y.reduce(...)
    rules.push((
        var("x").add(UPatOrConst::Pat(var("y"))).reduce(vec![], Some(red_add_arg()), true).named("r"),
        r(|ctx, st, _| {
            let (x, y, rr) = (sget(st, "x"), sget(st, "y"), sget(st, "r"));
            let rngs = ctx.srcs(&rr)[1..].to_vec();
            let rx = ctx.uop(Ops::Reduce, None, {
                let mut v = vec![x];
                v.extend(rngs.iter().cloned());
                v
            }, red_add_arg());
            let ry = ctx.uop(Ops::Reduce, None, {
                let mut v = vec![y];
                v.extend(rngs.iter().cloned());
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
            let rngs = ctx.srcs(&rr)[1..].to_vec();
            let zero = ctx.const_like(y.clone(), CV_num(0));
            // python: y.where(c, 0).reduce(*r.src[1:], arg=Ops.ADD)*x
            let yw = ctx.where_(y, c, zero);
            let mut v2 = vec![yw];
            v2.extend(rngs.iter().cloned());
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
            let zero = ctx.const_like(x.clone(), CV_num(0));
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
            .allow_all_srcs()
            .named("red"),
        r(|ctx, st, _| {
            let red = sget(st, "red");
            let pm = pm_reduce_collapse();
            let u = ctx.srcs(&red)[0].clone();
            reduce_collapse(ctx, red, &pm).inspect(|_ret| {
                let _ = u;
            })
        }),
    );
    pm_reduce_unparented().add(&PatternMatcher::new(vec![reduce_collapse_rule]))
}

/// `no_load(u)` (simplify.py:154) — no INDEX in the backward slice (with self).
fn no_load(ctx: &UOpCtx, u: UopId) -> bool {
    !ctx.op_in_backward_slice_with_self(&u, &[Ops::Index])
}

/// `reduce_load_collapse` (simplify.py:147).
pub fn reduce_load_collapse(ctx: &mut UOpCtx, red: UopId) -> Option<UopId> {
    let pm = pm_reduce_load_collapse();
    reduce_collapse(ctx, red, &pm)
}

/// `pm_reduce_load_collapse` (simplify.py:124) = `pm_reduce_collapse` + the two
/// load-collapse lift rules (ne-lift + gated-load gather).
pub fn pm_reduce_load_collapse() -> PatternMatcher {
    let mut rules: Vec<(UPat, RuleFn)> = vec![];
    // lift x+y out of reduce on ne: (x+y).or_casted() != c  ->  x != (c.cast(y.dtype)-y)
    rules.push((
        var("x").add(UPatOrConst::Pat(var("y"))).or_casted(None).ne(var("c")),
        r(|ctx, st, _| {
            let (x, y, c) = (sget(st, "x"), sget(st, "y"), sget(st, "c"));
            if !no_range(ctx, y.clone()) || !no_range(ctx, c.clone()) {
                return None;
            }
            let ccast = ctx.cast(c, ctx.dt_of(&y));
            let cmy = ctx.sub(ccast, y);
            Some(ctx.ne(x, cmy))
        }),
    ));
    // reduce on a gated load becomes a valid-gated gather (substitute the range):
    //   (idx != r.or_casted()).where(0, expr).reduce(r, ADD)
    //     -> ((idx.cast(r.dtype) >= 0) & (idx.cast(r.dtype) < r.src[0]))
    //          .where(expr.substitute({r: idx.cast(r.dtype).valid(v)}), 0)
    let r_rng = UPat::op_var(vec![Ops::Range]).named("r");
    rules.push((
        var("idx")
            .ne(r_rng.or_casted(Some("r")))
            .where_(UPatOrConst::Val(ConstVal::Int(0)), UPatOrConst::Pat(var("expr")))
            .reduce(vec![var("r")], Some(red_add_arg()), false),
        r(|ctx, st, _| {
            let (rr, idx, expr) = (sget(st, "r"), sget(st, "idx"), sget(st, "expr"));
            let rdt = ctx.dt_of(&rr);
            let icast = ctx.cast(idx, rdt);
            let zero = ctx.const_like(icast.clone(), CV_num(0));
            // python's `>=` is `(a < b).logical_not()`
            let lt0 = ctx.lt(icast.clone(), zero.clone());
            let ge = ctx.logical_not(lt0);
            let rend = ctx.srcs(&rr)[0].clone();
            let lt_end = ctx.lt(icast.clone(), rend);
            let v = ctx.and_(ge, lt_end);
            let valid = ctx.valid(icast, v.clone());
            let mut m: HashMap<UopId, UopId> = HashMap::new();
            m.insert(rr, valid);
            let subbed = substitute(ctx, expr, m, true, true);
            Some(ctx.where_(v, subbed, zero))
        }),
    ));
    pm_reduce_collapse().add(&PatternMatcher::new(rules))
}

/// `pm_load_collapse` (simplify.py:155) — remove REDUCE on a load (indexing a
/// tensor with another tensor); also undoes the weak-index arithmetic that the
/// collapse enables so we don't do math on a loaded index (overflow).
pub fn pm_load_collapse() -> PatternMatcher {
    let mut rules: Vec<(UPat, RuleFn)> = vec![];
    rules.push((
        var("u").reduce(vec![UPat::var(None, None)], Some(red_add_arg()), false).named("red"),
        r(|ctx, st, _| reduce_load_collapse(ctx, sget(st, "red"))),
    ));
    // (weakint x + y) < c  ->  x < c - y  when y,c have no load and x does
    rules.push((
        crate::uop::sym::var_dt("x", vec![dtype::WEAKINT])
            .add(UPatOrConst::Pat(var("y")))
            .lt(UPatOrConst::Pat(var("c"))),
        r(|ctx, st, _| {
            let (x, y, c) = (sget(st, "x"), sget(st, "y"), sget(st, "c"));
            if no_load(ctx, y.clone()) && no_load(ctx, c.clone()) && !no_load(ctx, x.clone()) {
                let cmy = ctx.sub(c, y);
                Some(ctx.lt(x, cmy))
            } else {
                None
            }
        }),
    ));
    PatternMatcher::new(rules)
}


fn sval_to_constval(s: &SVal) -> ConstVal {
    match s {
        SVal::I(i) => ConstVal::Int(*i),
        _ => panic!("TGERR: symbolic bound on substitute variable"),
    }
}

// **** pm_reduce_local (codegen/__init__.py:227-234, run in "remove reduces") ****

/// `ReduceContext` — the acc placeholder slot counter (codegen/__init__.py:209).
pub type ReduceCtx = std::rc::Rc<std::cell::Cell<i64>>;

/// `UOp.placeholder(ProdArg, slot, AddrSpace.REG)` (ops.py:1138-1150): LOCAL/REG
/// placeholders are Ops.BUFFER with a REG-addrspace ParamArg and no device.
fn placeholder_reg(ctx: &mut UOpCtx, like: UopId, slot: i64) -> UopId {
    let dt = ctx.dt_of(&like);
    let shape = ctx.shape(&like).unwrap().unwrap_or_default();
    let numel = crate::tensor::movement::prod_sval(ctx, &shape);
    let arg = crate::uop::arg::ParamArg {
        slot,
        dtype: dt,
        vmin_vmax: None,
        multiple_of: None,
        name: None,
        addrspace: Some(dtype::AddrSpace::Reg),
        axis: None,
        device: None,
        volatile: false,
    };
    let src = ctx.shape_arg(&[numel]);
    ctx.uop(Ops::Buffer, None, vec![src], Arg::Param(arg))
}

/// `x.ended_ranges` (ops.py:481-488).
fn ended_ranges_of(ctx: &UOpCtx, u: UopId, out: &mut Vec<UopId>) {
    let op = ctx.op_of(&u);
    if op == Ops::End || op == Ops::Reduce || op == Ops::Stage {
        out.extend(ctx.srcs(&u)[1..].iter().cloned());
    } else if op == Ops::After {
        for s in &ctx.srcs(&u)[1..] {
            ended_ranges_of(ctx, s.clone(), out);
        }
    }
}

fn reduce_op_and_axes(ctx: &UOpCtx, r: UopId) -> (Ops, i64) {
    match ctx.arg_of(&r) {
        Arg::Seq { tuple: true, items } => {
            let op = match &items[0] {
                Arg::Enum(e) => Ops::from_u16(e.value).unwrap_or_else(|| panic!("TGERR: bad reduce op")),
                _ => panic!("TGERR: bad reduce arg"),
            };
            let n = match &items[1] {
                Arg::Int(i) => *i,
                _ => panic!("TGERR: bad reduce arg"),
            };
            (op, n)
        }
        _ => panic!("TGERR: bad reduce arg"),
    }
}

/// `fix_group_for_reduce` (codegen/__init__.py:170-184) — corpus subset: GROUP
/// (LOCAL shared-memory) reduces are not exercised; only the early-return path.
pub fn fix_group_for_reduce(ctx: &UOpCtx, x: UopId) -> Option<UopId> {
    let has_gfr = ctx.srcs(&x)[1..].iter().any(|u| ctx.op_of(u) == Ops::Range && AxisType::range_axis_type(ctx, u.clone()) == AxisType::GroupReduce);
    if has_gfr {
        panic!("TGERR: GROUP_REDUCE reduces are not supported in the M6 corpus");
    }
    None
}

/// `reduce_ranges_to_acc` (codegen/__init__.py:210-222).
pub fn reduce_ranges_to_acc(ctx: &mut UOpCtx, r: UopId, acc_num: &std::cell::Cell<i64>) -> Option<UopId> {
    let (_op, _dt, srcs, arg, _tag) = ctx.snapshot(&r);
    if srcs.len() < 2 {
        return None; // pattern requires input + at least one range
    }
    let (rop, _nax) = reduce_op_and_axes(ctx, r.clone());
    let acc = placeholder_reg(ctx, r.clone(), acc_num.get());
    acc_num.set(acc_num.get() + 1);
    let topo = ctx.toposort(&srcs[0]).to_vec();
    let mut ended: Vec<UopId> = Vec::new();
    for x in &topo {
        ended_ranges_of(ctx, x.clone(), &mut ended);
    }
    let input_ranges: Vec<UopId> = topo
        .iter().filter(|&x| ctx.op_of(x) == Ops::Range && !srcs[1..].contains(x) && !ended.contains(x)).cloned()
        .collect();
    // acc_init = acc.after(*input_ranges).store(identity_element(r.arg[0], r.dtype))
    let ident = ctx.identity_element(rop, ctx.dt_of(&r));
    let acc_in = crate::tensor::after(ctx, acc.clone(), input_ranges.clone());
    let init_store = crate::tensor::store(ctx, acc_in, ident, None);
    // acc_initted = acc.after(acc_init, *r.src[1:])
    let acc_initted = crate::tensor::after(ctx, acc.clone(), [vec![init_store], srcs[1..].to_vec()].concat());
    // inp = r.src[0].reduce(arg=r.arg) if r.arg[1] else r.src[0]
    let nax = match &arg {
        Arg::Seq { tuple: true, items } => match &items[1] {
            Arg::Int(i) => *i,
            _ => 0,
        },
        _ => 0,
    };
    let inp = if nax > 0 {
        ctx.uop(Ops::Reduce, Some(ctx.dt_of(&srcs[0])), vec![(srcs[0]).clone()], arg.clone())
    } else {
        srcs[0].clone()
    };
    // acc_out = acc_initted.store(acc_initted.alu(r.arg[0], inp)).end(*r.src[1:]).rtag("mergeable")
    let acc_val = ctx.uop(rop, Some(ctx.dt_of(&acc_initted)), vec![acc_initted.clone(), inp], Arg::None);
    let store_out = crate::tensor::store(ctx, acc_initted, acc_val, None);
    let acc_out = ctx.uop_tag(Ops::End, None, [vec![store_out], srcs[1..].to_vec()].concat(), Arg::None, Arg::Str("mergeable".into()));
    // return acc.after(acc_out)
    Some(crate::tensor::after(ctx, acc, vec![acc_out]))
}

/// `expand_horizontal_reduce` (codegen/__init__.py:216-221) — REDUCE with a
/// single src: index over the leading arg[1] axes and chain the alu.
pub fn expand_horizontal_reduce(ctx: &mut UOpCtx, r: UopId) -> Option<UopId> {
    let (_op, _dt, srcs, _arg, _tag) = ctx.snapshot(&r);
    if srcs.len() != 1 {
        return None;
    }
    let inp = srcs[0].clone();
    let (rop, nax) = reduce_op_and_axes(ctx, r.clone());
    let shape = ctx.shape(&inp).unwrap_or_else(|_| panic!("TGERR: no shape on horizontal reduce input"));
    let shape: &Vec<SVal> = &shape.clone().unwrap_or_default();
    let dims: Vec<i64> = (0..nax as usize)
        .map(|a| match shape.get(a) {
            Some(SVal::I(n)) => *n as i64,
            _ => panic!("TGERR: horizontal reduce on symbolic axis"),
        })
        .collect();
    let dt = ctx.dt_of(&r);
    let mut vals: Vec<UopId> = Vec::new();
    let mut ctr = vec![0i64; dims.len()];
    loop {
        let idxs: Vec<UopId> = ctr
            .iter()
            .map(|&i| ctx.const_py(ConstVal::Int(i as i128), Some(dtype::WEAKINT)))
            .collect();
        let mut srcs2 = vec![inp.clone()];
        srcs2.extend(idxs);
        vals.push(ctx.uop(Ops::Index, Some(dt), srcs2, Arg::None));
        // next product element (row-major)
        let mut i = dims.len();
        loop {
            if i == 0 {
                i = usize::MAX;
                break;
            }
            i -= 1;
            ctr[i] += 1;
            if ctr[i] < dims[i] {
                break;
            }
            ctr[i] = 0;
        }
        if i == usize::MAX {
            break;
        }
    }
    let mut ret = vals[0].clone();
    for v in &vals[1..] {
        ret = ctx.uop(rop, Some(dt), vec![ret, v.clone()], Arg::None);
    }
    Some(ret)
}

/// `merge_reduce_ends` (codegen/__init__.py:190-208) — merge ENDs that share
/// the same range list and were tagged "mergeable".
pub fn merge_reduce_ends(ctx: &mut UOpCtx, sink: UopId) -> Option<UopId> {
    // range_to_ends: key = the END's range srcs (srcs[1..]), value = ends
    let mut range_to_ends: HashMap<Vec<UopId>, Vec<UopId>> = HashMap::new();
    for u in ctx.toposort(&sink).iter() {
        if ctx.op_of(u) == Ops::End && matches!(ctx.tag_of(u), Arg::Str(s) if s == "mergeable") {
            range_to_ends.entry(ctx.srcs(u)[1..].to_vec()).or_default().push(u.clone());
        }
    }
    let next_axis_start = ctx
        .toposort(&sink)
        .iter()
        .filter(|&u| ctx.op_of(u) == Ops::Range)
        .filter_map(|u| match ctx.arg_of(u) {
            Arg::Seq { tuple: true, items } => match &items[0] {
                Arg::Int(i) => Some(*i),
                _ => None,
            },
            _ => None,
        })
        .max()
        .unwrap_or(-1)
        + 1;
    let mut subs: Vec<(UopId, UopId)> = Vec::new();
    let mut next_axis = next_axis_start;
    for (r, ends) in range_to_ends.iter() {
        if ends.len() <= 1 {
            continue;
        }
        // group by the END's e.ranges (the ranges of e.src[0])
        let mut by_ctx: Vec<(Vec<UopId>, Vec<UopId>)> = Vec::new(); // (sorted ranges, ends)
        for e in ends {
            let mut rs: Vec<UopId> = ctx.ranges(&ctx.srcs(e)[0].clone()).to_vec();
            rs.sort_unstable();
            match by_ctx.iter_mut().find(|(k, _)| *k == rs) {
                Some((_, v)) => v.push(e.clone()),
                None => by_ctx.push((rs, vec![e.clone()])),
            }
        }
        for (i, (_, group)) in by_ctx.into_iter().enumerate() {
            let tr: Vec<UopId> = if i == 0 {
                r.clone()
            } else {
                let renamed: Vec<UopId> = r
                    .iter()
                    .enumerate()
                    .map(|(j, rr)| {
                        let (_op, dt, srcs, arg, _tag) = ctx.snapshot(rr);
                        let Arg::Seq { tuple: true, items } = &arg else { panic!("TGERR: bad range") };
                        // `rr.replace(arg=(next_axis + j, *rr.arg[1:]))`
                        let mut items2 = vec![Arg::Int(next_axis + j as i64)];
                        items2.extend(items[1..].iter().cloned());
                        ctx.uop(Ops::Range, Some(dt), srcs.to_vec(), Arg::Seq { tuple: true, items: items2 })
                    })
                    .collect();
                next_axis += r.len() as i64;
                renamed
            };
            // mapped = e.substitute(dict(zip(r, tr))) when i > 0
            let mapped: Vec<UopId> = group
                .iter()
                .map(|e| {
                    if i > 0 {
                        let m: HashMap<UopId, UopId> = r.iter().cloned().zip(tr.iter().cloned()).collect();
                        crate::uop::upat::substitute(ctx, e.clone(), m, false, false)
                    } else {
                        e.clone()
                    }
                })
                .collect();
            let merged = if mapped.len() == 1 {
                &mapped[0].clone()
            } else {
                let g = ctx.uop(Ops::Group, None, mapped, Arg::None);
                &ctx.uop_tag(Ops::End, None, [vec![g], tr.clone()].concat(), Arg::None, Arg::Str("mergeable".into()))
            };
            for e in &group {
                subs.push((e.clone(), merged.clone()));
            }
        }
    }
    if subs.is_empty() {
        return None;
    }
    let m: HashMap<UopId, UopId> = subs.into_iter().collect();
    Some(crate::uop::upat::substitute(ctx, sink, m, false, false))
}

/// `pm_reduce_local` (codegen/__init__.py:227-234): fix groups, build acc
/// patterns for parented reduces, expand horizontal reduces, merge ENDs.
pub fn pm_reduce_local() -> PatternMatcher {
    PatternMatcher::new(vec![
        (
            UPat::ops1(Ops::Reduce).named("x"),
            Arc::new(move |ctx: &mut UOpCtx, st: &Store, _: &CtxVal| fix_group_for_reduce(ctx, sget(st, "x"))),
        ),
        (
            UPat::ops1(Ops::Reduce)
                .with_src_tup(vec![UPat::var(None, None), UPat::var(None, None)])
                .allow_all_srcs()
                .named("r"),
            crate::uop::upat::r_cnt(|c: &std::cell::Cell<i64>, ctx: &mut UOpCtx, st: &Store| {
                reduce_ranges_to_acc(ctx, sget(st, "r"), c)
            }),
        ),
        (
            UPat::ops1(Ops::Reduce).named("r"),
            Arc::new(move |ctx: &mut UOpCtx, st: &Store, _: &CtxVal| expand_horizontal_reduce(ctx, sget(st, "r"))),
        ),
        (
            UPat::ops1(Ops::Sink).named("sink"),
            Arc::new(move |ctx: &mut UOpCtx, st: &Store, _: &CtxVal| merge_reduce_ends(ctx, sget(st, "sink"))),
        ),
    ])
    .add(crate::uop::sym::pm_clean_up_group_sink())
}
