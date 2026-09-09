//! `uop/weak.py` port (M5): weak-dtype commitment and index-dtype lowering.

use crate::dtype::{self, ConstVal, DTypeId};
use crate::uop::ops::{group, Ops};
use crate::uop::arg::Arg;
use crate::uop::sym::{commit_weak, r, sget};
use crate::uop::upat::{graph_rewrite, CtxVal, PatternMatcher, UPat};
use crate::uop::ctx::UOpCtx;
use crate::uop::UopId;
use std::collections::HashMap;

/// `default_dtype` (weak.py:8).
pub fn default_dtype(ctx: &mut UOpCtx, u: UopId) -> DTypeId {
    if ctx.dt_of(u) == dtype::WEAKFLOAT {
        return dtype::default_float();
    }
    if overflows_int32(ctx, u) {
        dtype::INT64
    } else {
        dtype::default_int()
    }
}

fn overflows_int32(ctx: &mut UOpCtx, u: UopId) -> bool {
    let (vmin, vmax) = match ctx.min_max(u) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let (dmin, dmax) = ctx.dtype_bounds(dtype::INT32).unwrap();
    let lo = vmin.as_i128_or_panic();
    let hi = vmax.as_i128_or_panic();
    let dl = dmin.as_i128_or_panic();
    let dh = dmax.as_i128_or_panic();
    lo < dl || dh < hi
}

/// `commit_srcs_at` (weak.py:16) — commit weak srcs at dt, root re-derives.
fn commit_srcs_at(ctx: &mut UOpCtx, u: UopId, dt: DTypeId) -> UopId {
    let (op, old_dt, srcs, arg, tag) = ctx.snapshot(u);
    let mut new_srcs = Vec::with_capacity(srcs.len());
    let mut changed = false;
    for &s in &srcs {
        if dtype::is_weak(ctx.dt_of(s)) {
            new_srcs.push(commit_weak(ctx, s, dt));
            changed = true;
        } else {
            new_srcs.push(s);
        }
    }
    if !changed {
        return u;
    }
    // python's replace re-derives the dtype when src dtypes change
    let dt2 = if new_srcs.iter().zip(srcs.iter()).all(|(a, b)| ctx.dt_of(*a) == ctx.dt_of(*b)) {
        old_dt
    } else {
        ctx.dtype_from_uop(op, &new_srcs, &arg).unwrap_or(old_dt)
    };
    ctx.uop_tag(op, Some(dt2), new_srcs, arg, tag)
}

/// `commit_weak_srcs` (weak.py:20).
fn commit_weak_srcs(ctx: &mut UOpCtx, u: UopId) -> Option<UopId> {
    let srcs = ctx.srcs(u).to_vec();
    let any_weak = srcs.iter().any(|&s| dtype::is_weak(ctx.dt_of(s)));
    if !any_weak {
        return None;
    }
    let dt = dtype::least_upper_dtype(&srcs.iter().map(|&s| ctx.dt_of(s)).collect::<Vec<_>>());
    if dtype::is_weak(dt) {
        return None;
    }
    Some(commit_srcs_at(ctx, u, dt))
}

/// `cast_weak_srcs` (weak.py:27).
fn cast_weak_srcs(ctx: &mut UOpCtx, c: UopId, u: UopId) -> Option<UopId> {
    let cdt = ctx.dt_of(c);
    if dtype::is_weak(cdt) || dtype::weak_dtype(cdt) != ctx.dt_of(u) {
        return None;
    }
    let d = default_dtype(ctx, u);
    let dt = dtype::least_upper_dtype2(cdt, d);
    let committed = commit_srcs_at(ctx, u, dt);
    Some(ctx.cast(committed, cdt))
}

/// `pm_cast_weak` (weak.py:31).
pub fn pm_cast_weak() -> PatternMatcher {
    PatternMatcher::new(vec![
        (
            UPat::ops1(Ops::Cast)
                .with_src_tup(vec![UPat::op_var(group::ALU.to_vec()).with_dtype(Some(dtype::WEAKS.to_vec())).named("u")])
                .named("c"),
            r(|ctx, st, _| {
                let c = sget(st, "c");
                let u = ctx.srcs(c)[0];
                cast_weak_srcs(ctx, c, u)
            }),
        ),
        (
            UPat::ops1(Ops::Cast)
                .with_src_tup(vec![UPat::ops1(Ops::Const).with_dtype(Some(dtype::WEAKS.to_vec())).named("u")])
                .named("c"),
            r(|ctx, st, _| {
                let c = sget(st, "c");
                let u = ctx.srcs(c)[0];
                Some(commit_weak(ctx, u, ctx.dt_of(c)))
            }),
        ),
    ])
}

/// `pm_commit_weak` (weak.py:38).
pub fn pm_commit_weak() -> PatternMatcher {
    PatternMatcher::new(vec![
        (
            UPat::op_var(group::BROADCASTABLE.to_vec()).named("u"),
            r(|ctx, st, _| commit_weak_srcs(ctx, sget(st, "u"))),
        ),
        (
            UPat::ops1(Ops::Store).with_src_any_len(true).named("u"),
            r(|ctx, st, _| {
                let u = sget(st, "u");
                let srcs = ctx.srcs(u).to_vec();
                if !dtype::is_weak(ctx.dt_of(srcs[1])) {
                    return None;
                }
                let committed = commit_weak(ctx, srcs[1], ctx.dt_of(srcs[0]));
                let mut new_srcs = vec![srcs[0], committed];
                new_srcs.extend(srcs[2..].iter().copied());
                let (_, d, _, arg, tag) = ctx.snapshot(u);
                Some(ctx.replace(u, Ops::Store, d, new_srcs, arg, tag))
            }),
        ),
    ])
}

/// `_lower_weak_ops` (weak.py:45).
fn in_lower_weak_ops(op: Ops) -> bool {
    (group::in_(group::BINARY, op) && !group::in_(group::COMPARISON, op)) || group::in_(group::UNARY, op) || matches!(op, Ops::Where | Ops::Range | Ops::Stack | Ops::Special)
}

/// `lower_weak_node` (weak.py:47).
fn lower_weak_node(ctx: &mut UOpCtx, u: UopId) -> Option<UopId> {
    let (op, udt, usrcs, arg, tag) = ctx.snapshot(u);
    let mut src: Vec<UopId> = Vec::with_capacity(usrcs.len());
    for &s in &usrcs {
        if ctx.op_of(s) == Ops::Cast && dtype::is_weak(ctx.dt_of(s)) {
            src.push(ctx.srcs(s)[0]);
        } else {
            src.push(s);
        }
    }
    let start = if op == Ops::Where { 1 } else { 0 };
    let unchanged = src == usrcs;
    let any_weak = src[start.min(src.len())..].iter().any(|&s| dtype::is_weak(ctx.dt_of(s)));
    if unchanged || any_weak {
        return None;
    }
    let dd = default_dtype(ctx, u);
    let dt = if group::in_(group::BINARY, op) && !group::in_(group::COMPARISON, op) {
        let mut dts = vec![dd];
        dts.extend(src.iter().map(|&s| ctx.dt_of(s)));
        dtype::strong_dtype(dtype::least_upper_dtype(&dts))
    } else {
        ctx.dtype_from_uop(op, &src, &arg).unwrap_or(udt)
    };
    let mut new_srcs: Vec<UopId> = src[..start.min(src.len())].to_vec();
    for &s in &src[start.min(src.len())..] {
        if ctx.is_invalid(ctx.base(s)) {
            new_srcs.push(s);
        } else {
            new_srcs.push(commit_weak(ctx, s, dt));
        }
    }
    let rebuilt = ctx.uop_tag(op, Some(dt), new_srcs, arg, tag);
    Some(ctx.cast(rebuilt, udt))
}

/// `pm_lower_weak` (weak.py:57).
pub fn pm_lower_weak() -> PatternMatcher {
    PatternMatcher::new(vec![
        (
            UPat::ops1(Ops::Const).with_dtype(Some(dtype::WEAKS.to_vec())).named("u"),
            r(|ctx, st, _| {
                let u = sget(st, "u");
                let v = ctx.val(u);
                let d = default_dtype(ctx, u);
                let c = ctx.const_py(v, Some(d));
                Some(ctx.cast(c, ctx.dt_of(u)))
            }),
        ),
        // two stacked weak casts resolve at each kind's default
        (
            UPat::ops1(Ops::Cast)
                .with_dtype(Some(dtype::WEAKS.to_vec()))
                .with_src_tup(vec![UPat::ops1(Ops::Cast)
                    .with_dtype(Some(dtype::WEAKS.to_vec()))
                    .with_src_tup(vec![UPat::var(Some("x"), None)])
                    .named("_inner")])
                .named("u"),
            r(|ctx, st, _| {
                let u = sget(st, "u");
                let x = ctx.srcs(ctx.srcs(u)[0])[0];
                if dtype::is_weak(ctx.dt_of(x)) {
                    return None;
                }
                let inner = ctx.srcs(u)[0];
                let d1 = default_dtype(ctx, inner);
                let a = ctx.cast(x, d1);
                let d2 = default_dtype(ctx, u);
                let b = ctx.cast(a, d2);
                Some(ctx.cast(b, ctx.dt_of(u)))
            }),
        ),
        // ALU PARAM/BUFFER weakint → commit at the default
        (
            UPat::op_var(vec![Ops::Param, Ops::Buffer]).with_dtype(Some(vec![dtype::WEAKINT])).named("u"),
            r(|ctx, st, _| {
                let u = sget(st, "u");
                if ctx.addrspace_of(u) != Some(dtype::AddrSpace::Alu) {
                    return None;
                }
                let d = default_dtype(ctx, u);
                let (op, _dt, srcs, arg, tag) = ctx.snapshot(u);
                let mut new_arg = arg.clone();
                if let Arg::Param(p) = &mut new_arg {
                    p.dtype = d;
                }
                let rebuilt = ctx.uop_tag(op, Some(d), srcs, new_arg, tag);
                Some(ctx.cast(rebuilt, dtype::WEAKINT))
            }),
        ),
        (
            UPat::op_var((0..=79u16).filter_map(Ops::from_u16).collect()).named("u"),
            r(|ctx, st, _| {
                let u = sget(st, "u");
                if !in_lower_weak_ops(ctx.op_of(u)) {
                    return None;
                }
                lower_weak_node(ctx, u)
            }),
        ),
    ])
}

/// `lower_weak_srcs` (weak.py:76).
fn lower_weak_srcs(ctx: &mut UOpCtx, memo: &mut HashMap<UopId, UopId>, u: UopId) -> UopId {
    fn lower(ctx: &mut UOpCtx, memo: &mut HashMap<UopId, UopId>, s: UopId) -> UopId {
        if let Some(&r) = memo.get(&s) {
            return r;
        }
        let rw = graph_rewrite(ctx, s, &pm_lower_weak(), false, None, false, false, CtxVal::None)
            .unwrap_or_else(|e| panic!("TGERR: lower weak: {e}"));
        // the consumer absorbs the cast on its own edge
        let rw = if ctx.op_of(rw) == Ops::Cast && dtype::is_weak(ctx.dt_of(rw)) {
            ctx.srcs(rw)[0]
        } else {
            rw
        };
        memo.insert(s, rw);
        rw
    }
    if group::in_(group::COMPARISON, ctx.op_of(u)) {
        lower(ctx, memo, u)
    } else {
        let (op, dt, srcs, arg, tag) = ctx.snapshot(u);
        let mut new_srcs = Vec::with_capacity(srcs.len());
        let mut changed = false;
        for &s in &srcs {
            if dtype::is_weak(ctx.dt_of(s)) {
                new_srcs.push(lower(ctx, memo, s));
                changed = true;
            } else {
                new_srcs.push(s);
            }
        }
        if !changed {
            u
        } else {
            let dt2 = if new_srcs.iter().zip(srcs.iter()).all(|(a, b)| ctx.dt_of(*a) == ctx.dt_of(*b)) {
                dt
            } else {
                ctx.dtype_from_uop(op, &new_srcs, &arg).unwrap_or(dt)
            };
            ctx.uop_tag(op, Some(dt2), new_srcs, arg, tag)
        }
    }
}

/// `pm_lower_index_dtype` (weak.py:81) — commit_weak+cast_weak+lower driver.
pub fn pm_lower_index_dtype() -> PatternMatcher {
    let all_pat = UPat::op_var((0..=79u16).filter_map(Ops::from_u16).collect()).named("u");
    PatternMatcher::new(vec![(
        all_pat,
        r(|ctx, st, _| {
            let u = sget(st, "u");
            if dtype::is_weak(ctx.dt_of(u)) {
                return None;
            }
            let srcs = ctx.srcs(u).to_vec();
            if !srcs.iter().any(|&s| dtype::is_weak(ctx.dt_of(s))) {
                return None;
            }
            // python threads a ctx dict through the rewrite; our memo lives per-rule-invocation
            let mut m = HashMap::new();
            let ret = lower_weak_srcs(ctx, &mut m, u);
            if ret == u {
                None
            } else {
                Some(ret)
            }
        }),
    )])
    .add(&pm_commit_weak())
    .add(&pm_cast_weak())
}
