//! `schedule/rangeify.py` port — kernel graph construction: movement-on-INDEX
//! rewrites (`pm_mops`), earliest cleanup rewrites, copy→store conversion,
//! bufferize removal, buffer insertion, and kernel splitting.

use std::collections::{HashMap, HashSet};

use crate::dtype::{AddrSpace, ConstVal};
use crate::uop::arg::{kernel_info, Arg, DeviceSpec, ParamArg};
use crate::uop::ctx::UOpCtx;
use crate::uop::ops::{group, Ops};
use crate::uop::range::AxisType;
use crate::uop::sint::{SInt, SVal};
use crate::uop::sym::{self, pm_mop_cleanup, symbolic, r, CtxVal};
use crate::uop::upat::{graph_rewrite, r_cnt, r_ctx, r_la, substitute_extra, PatternMatcher, RuleFn, SrcPat, UPat};
use crate::uop::UopId;

use super::indexing::apply_movement_op;

// ---------- pm_mops (rangeify.py:39-57) ----------

/// `_mop_index` (rangeify.py:39) — movement op on INDEX.
fn mop_index(ctx: &mut UOpCtx, r: UopId, idx: UopId) -> Option<UopId> {
    let idxs = ctx.srcs(&idx)[1..].to_vec();
    let r_shape = ctx.shape(&r).ok().flatten()?;
    let r_op = ctx.op_of(&r);
    let r_src0 = ctx.srcs(&r)[0].clone();
    let src0_shape = ctx.shape(&r_src0).ok().flatten()?;
    let idx_arg = ctx.arg_of(&idx).clone();
    if idxs.len() == r_shape.len() {
        let marg = marg_args(ctx, r);
        let rngs = apply_movement_op(ctx, r_op, &src0_shape, &marg, &idxs);
        let ret = ctx.index(r_src0, rngs);
        if matches!(idx_arg, Arg::None) {
            return Some(ret);
        }
        let ret_srcs = ctx.srcs(&ret).to_vec();
        return Some(ctx.uop_tag(Ops::Index, None, ret_srcs, idx_arg, Arg::None));
    }
    if r_op == Ops::Reshape && idxs.len() < r_shape.len() {
        let src_prefix = src0_shape.len() as i64 - r_shape[idxs.len()..].len() as i64;
        if src_prefix >= 0 && src0_shape[src_prefix as usize..] == r_shape[idxs.len()..] {
            if src_prefix == 0 {
                return if ctx.dt_of(&r_src0) == ctx.dt_of(&idx) { Some(r_src0) } else { None };
            }
            let marg = marg_args(ctx, r);
            let rngs = apply_movement_op(ctx, r_op, &src0_shape[..src_prefix as usize], &marg, &idxs);
            let ret = ctx.index(r_src0, rngs);
            let ret_shape = ctx.shape(&ret).ok().flatten()?;
            return if ret_shape == ctx.shape(&idx).ok().flatten()? { Some(ret) } else { None };
        }
    }
    None
}

/// movement `marg` as flat args: RESHAPE/EXPAND shapes, PAD/SHRINK pairs,
/// PERMUTE/FLIP node arg tuples.
pub fn marg_args(ctx: &mut UOpCtx, x: UopId) -> Vec<Arg> {
    let op = ctx.op_of(&x);
    let srcs = ctx.srcs(&x).to_vec();
    match op {
        Ops::Reshape | Ops::Expand => shape_src_args(ctx, (srcs[1]).clone()),
        Ops::Pad | Ops::Shrink => {
            let a = shape_src_args(ctx, (srcs[1]).clone());
            let b = shape_src_args(ctx, (srcs[2]).clone());
            a.into_iter().zip(b).map(|(x, y)| Arg::Seq { tuple: true, items: vec![x, y] }).collect()
        }
        Ops::Permute | Ops::Flip => match ctx.arg_of(&x) {
            Arg::Seq { tuple: _, items } => items.clone(),
            _ => vec![],
        },
        other => panic!("TGERR: {other:?} is not a MovementOp"),
    }
}

fn shape_arg_to_arg(v: ConstVal) -> Arg {
    match v {
        ConstVal::Int(i) => Arg::Int(if i >= i64::MIN as i128 && i <= i64::MAX as i128 { i as i64 } else { panic!("TGERR: shape const out of i64") }),
        other => panic!("TGERR: bad shape src const {other:?}"),
    }
}

fn shape_src_args(ctx: &mut UOpCtx, stack: UopId) -> Vec<Arg> {
    // the shape STACK srcs: each a CONST (int) — symbolic shapes panic (concrete corpus)
    // NOTE: `shape_arg` collapses a single-element shape STACK to the bare const, so a 1-D
    // shape arrives here as a CONST (no srcs) rather than a STACK. tinygrad's `u.arg` is
    // always a tuple (e.g. EXPAND arg `(2,)` has len 1), so a bare const must still report
    // one shape dim — omitting it made `apply_movement_op(EXPAND)` drop 0 ranges instead of 1.
    let srcs = ctx.srcs(&stack).to_vec();
    if srcs.is_empty() && ctx.op_of(&stack) == Ops::Const {
        return vec![shape_arg_to_arg(ctx.val(&stack))];
    }
    srcs.iter().map(|c| shape_arg_to_arg(ctx.val(c))).collect()
}

/// SVal shape list -> Arg list (for apply_movement_op RESHAPE).
pub fn svals_to_args(sv: &[SVal]) -> Vec<Arg> {
    sv.iter()
        .map(|s| match s {
            SVal::I(i) => Arg::Int(if *i >= i64::MIN as i128 && *i <= i64::MAX as i128 { *i as i64 } else { panic!("TGERR: shape const out of i64") }),
            SVal::U(_) => panic!("TGERR: symbolic shape in movement arg"),
        })
        .collect()
}

/// `pm_mops` (rangeify.py:50).
pub fn pm_mops() -> PatternMatcher {
    static PM: crate::uop::upat::PmSlot = crate::uop::upat::PmSlot::new();
    PM.get_or_init(build_pm_mops).clone()
}

fn build_pm_mops() -> PatternMatcher {
    let rules: Vec<(UPat, RuleFn)> = vec![
        // handle movement ops on INDEX: INDEX(MOVEMENT(v, marg), idxs...)
        // (python: `UPat(GroupOp.Movement, name="r").f(Ops.INDEX, allow_any_len=True, name="idx")`)
        (
            UPat::ops1(Ops::Index)
                .with_src_tup(vec![UPat::op_var(group::MOVEMENT.to_vec()).named("r")])
                .allow_all_srcs()
                .named("idx"),
            r(|ctx, st, _| {
                let idx = sym::sget(st, "idx");
                let r = sym::sget(st, "r");
                mop_index(ctx, r, idx)
            }),
        ),
        // move movement ops and INDEX after AFTER: AFTER(MOVEMENT(v, extra), deps) -> MOVEMENT(AFTER(v, deps), extra)
        (
            {
                let inner = UPat::op_var(group::MOVEMENT.iter().cloned().chain([Ops::Index]).collect::<Vec<Ops>>()).named("r");
                let dt = inner.dtype.clone();
                let mut p = UPat::new(Some(vec![Ops::After]), dt, SrcPat::Tup(vec![inner]), None, Some("a"), true, None);
                p.recompute_early_reject();
                p
            },
            r_ctx(|ctx, st| {
                let r = sym::sget(st, "r");
                let a = sym::sget(st, "a");
                let rsrcs = ctx.srcs(&r).to_vec();
                let asrcs = ctx.srcs(&a).to_vec();
                let mut new_asrcs = vec![(rsrcs[0]).clone()];
                new_asrcs.extend(asrcs[1..].iter().cloned());
                let new_a = ctx.uop_tag(ctx.op_of(&a), Some(ctx.dt_of(&a)), new_asrcs, ctx.arg_of(&a).clone(), ctx.tag_of(&a).clone());
                let mut ret_srcs = vec![new_a];
                ret_srcs.extend(rsrcs[1..].iter().cloned());
                Some(ctx.uop_tag(ctx.op_of(&r), Some(ctx.dt_of(&r)), ret_srcs, ctx.arg_of(&r).clone(), ctx.tag_of(&r).clone()))
            }),
        ),
        // move movement ops after END: END(MOVEMENT(v, extra), deps) -> END(v, deps)
        (
            {
                let inner = UPat::op_var(group::MOVEMENT.to_vec()).named("r");
                let dt = inner.dtype.clone();
                let mut p = UPat::new(Some(vec![Ops::End]), dt, SrcPat::Tup(vec![inner]), None, Some("a"), false, None);
                p.recompute_early_reject();
                p
            },
            r_ctx(|ctx, st| {
                let r = sym::sget(st, "r");
                let a = sym::sget(st, "a");
                let rsrcs = ctx.srcs(&r).to_vec();
                let asrcs = ctx.srcs(&a).to_vec();
                let mut new_asrcs = vec![(rsrcs[0]).clone()];
                new_asrcs.extend(asrcs[1..].iter().cloned());
                Some(ctx.uop_tag(ctx.op_of(&a), Some(ctx.dt_of(&a)), new_asrcs, ctx.arg_of(&a).clone(), ctx.tag_of(&a).clone()))
            }),
        ),
    ];
    PatternMatcher::new(rules)
}

// ---------- earliest rewrites (rangeify.py:127-187) ----------

/// `split_reduceop` (rangeify.py:71) — split big reduces into two kernels.
fn split_reduceop(ctx: &mut UOpCtx, reduce: UopId, x: UopId) -> Option<UopId> {
    let red_shape = ctx.shape(&reduce).ok().flatten()?;
    let x_shape = ctx.shape(&x).ok().flatten()?;
    if red_shape.iter().any(|s| s.clone().as_i64() == Some(0)) {
        return None;
    }
    if !x_shape.iter().all(|s| matches!(s, SVal::I(_))) {
        return None;
    }
    let numel_in: i128 = x_shape.iter().map(|s| s.clone().as_i128_or_panic()).product();
    let numel_out: i128 = red_shape.iter().map(|s| s.clone().as_i128_or_panic()).product();
    const THRESHOLD: i128 = 32768;
    if numel_in.checked_div(numel_out).unwrap_or(0) < THRESHOLD {
        return None;
    }
    // get expanded-by-rangeifying view of x
    let mut rng_srcs = Vec::new();
    for (i, s) in x_shape.iter().enumerate() {
        let r = if ctx.resolve_ord(s.clone(), crate::uop::ctx::CmpOrd::Gt, SVal::I(1), true) {
            ctx.range(s.clone(), i as i64, AxisType::Weak)
        } else {
            ctx.const_py(ConstVal::Int(0), Some(crate::dtype::WEAKINT))
        };
        rng_srcs.push(r);
    }
    let indexed = ctx.index(x.clone(), rng_srcs.clone());
    // range_nums after substituting x -> NOOP with pm_mops applied
    let mut dvars = HashMap::new();
    let x_base = ctx.base(&x);
    let noop = ctx.uop(Ops::Noop, None, vec![], Arg::None);
    dvars.insert(x_base, noop);
    let sub = substitute_extra(ctx, indexed, dvars, Some(&pm_mops()), false, false);
    let range_nums: HashSet<i64> = ctx.ranges(&sub).iter().map(|r| AxisType::range_axis_id(ctx, r.clone())).collect();
    let is_expanded: Vec<bool> = (0..x_shape.len()).map(|i| !range_nums.contains(&(i as i64))).collect();

    let (_op, _dt, _srcs, arg, _tag) = ctx.snapshot(&reduce);
    let n_reduce = match &arg {
        Arg::Seq { tuple: true, items } => match &items[1] {
            Arg::Int(n) => *n,
            _ => 0,
        },
        _ => 0,
    };
    let red_numel: i128 = red_shape.iter().map(|s| s.clone().as_i128_or_panic()).product();
    let max_d = ((1i128 << 22).checked_div(red_numel.max(1)).unwrap_or(0)).min(256);
    let mut split_candidates: Vec<(usize, i128)> = Vec::new();
    for i in 0..n_reduce as usize {
        for d in (8..=max_d).rev() {
            let xi = <SInt as Clone>::clone(&x_shape[i]).as_i128_or_panic();
            if xi % d == 0 && !is_expanded[i] {
                split_candidates.push((i, d));
            }
        }
    }
    let (dim_to_split, divisor) = *split_candidates.first()?;
    let xi = <SInt as Clone>::clone(&x_shape[dim_to_split]).as_i128_or_panic() as i64;
    let mut splitted_shape: Vec<SVal> = x_shape.to_vec();
    splitted_shape[dim_to_split] = SVal::I(divisor);
    splitted_shape.insert(dim_to_split + 1, SVal::I(xi as i128 / divisor));
    let splitted = crate::tensor::movement::reshape(ctx, x, &splitted_shape);
    let perm: Vec<usize> =
        (0..splitted_shape.len()).filter(|&d| d != dim_to_split).chain([dim_to_split]).collect();
    let splitted = crate::tensor::movement::permute_internal(ctx, splitted, &perm);
    // reduce original axes, then split
    let reduced = crate::tensor::reduce::rop(ctx, reduce_op_of(&arg), splitted, &(0..n_reduce as usize).collect::<Vec<_>>());
    let contig = ctx.uop(Ops::Contiguous, None, vec![reduced], Arg::None);
    let reduced2 = crate::tensor::reduce::rop(ctx, reduce_op_of(&arg), contig, &[red_shape.len()]);
    let final_shape: Vec<SVal> = red_shape.to_vec();
    Some(crate::tensor::movement::reshape(ctx, reduced2, &final_shape))
}

fn reduce_op_of(arg: &Arg) -> Ops {
    match arg {
        Arg::Seq { tuple: true, items } => match &items[0] {
            Arg::Enum(e) => Ops::from_u16(e.value).expect("TGERR: bad reduce op"),
            other => panic!("TGERR: bad reduce arg {other:?}"),
        },
        other => panic!("TGERR: bad reduce arg {other:?}"),
    }
}

/// `expand_bitcast` (rangeify.py:116) — shape-changing bitcast (rare; corpus has none).
fn expand_bitcast(_ctx: &mut UOpCtx, _bc: UopId) -> Option<UopId> {
    // TODO(M5): port when BITCAST corpus coverage arrives
    None
}

/// `fix_store_hazard` (rangeify.py:62).
fn fix_store_hazard(ctx: &mut UOpCtx, target: UopId, src: UopId) -> Option<UopId> {
    let base = ctx.base(&target);
    let ts = ctx.toposort_gated(src.clone(), None, false);
    if !ts.contains(&base) {
        return None;
    }
    let mut unsafe_ops = vec![Ops::Permute, Ops::Flip];
    if ctx.op_in_backward_slice_with_self(&target, &[Ops::Shrink]) {
        unsafe_ops.push(Ops::Shrink);
    }
    // reaches_base over src toposort, not entering CONTIGUOUS
    let ts_gated = ctx.toposort_gated(src.clone(), Some(&|s| ctx.op_of(&s) != Ops::Contiguous), true);
    let mut reaches_base: HashMap<UopId, bool> = HashMap::new();
    for s in ts_gated {
        let srcs = ctx.srcs(&s).to_vec();
        let rb = s == base || srcs.iter().any(|c| reaches_base.get(c).cloned().unwrap_or(false));
        reaches_base.insert(s.clone(), rb);
        if rb && unsafe_ops.contains(&ctx.op_of(&s)) && !(s == target && ctx.op_of(&s) == Ops::Shrink) {
            let contig = ctx.uop(Ops::Contiguous, None, vec![src], Arg::None);
            return Some(crate::tensor::store(ctx, target, contig, None));
        }
    }
    None
}

/// `earliest_rewrites` (rangeify.py:127).
pub fn earliest_rewrites() -> PatternMatcher {
    static PM: crate::uop::upat::PmSlot = crate::uop::upat::PmSlot::new();
    PM.get_or_init(build_earliest_rewrites).clone()
}

fn build_earliest_rewrites() -> PatternMatcher {
    let rules: Vec<(UPat, RuleFn)> = vec![
        // split_reduceop
        (
            UPat::ops1(Ops::Reduce).with_src_tup(vec![UPat::var(Some("x"), None)]).named("reduce"),
            r_ctx(|ctx, st| split_reduceop(ctx, sym::sget(st, "reduce"), sym::sget(st, "x"))),
        ),
        // remove DETACH/CONTIGUOUS_BACKWARD
        (
            UPat::op_var(vec![Ops::Detach, Ops::ContiguousBackward]).named("x"),
            r_ctx(|ctx, st| {
                let x = sym::sget(st, "x");
                Some(ctx.srcs(&x)[0].clone())
            }),
        ),
        // SINK only ever references the base
        (
            UPat::ops1(Ops::Sink).allow_all_srcs().named("x"),
            r_ctx(|ctx, st| {
                let x = sym::sget(st, "x");
                let srcs = ctx.srcs(&x).to_vec();
                let new_srcs: Vec<UopId> = srcs.iter().map(|y| ctx.unsharded_base(y.clone())).collect();
                if new_srcs == ctx.srcs(&x).to_vec() {
                    None
                } else {
                    Some(ctx.uop_tag(Ops::Sink, Some(ctx.dt_of(&x)), new_srcs, ctx.arg_of(&x).clone(), ctx.tag_of(&x).clone()))
                }
            }),
        ),
        // ** copy rules **
        // COPY transfers a contiguous range: materialize moved srcs
        (
            UPat::ops1(Ops::Copy).with_src_tup(vec![UPat::op_var(group::MOVEMENT.to_vec()).named("r")]).named("c"),
            r_ctx(|ctx, st| {
                let (c, r) = (sym::sget(st, "c"), sym::sget(st, "r"));
                let numel_r = ctx.shape(&r).ok().flatten()?.iter().map(|s| s.clone().as_i128_or_panic()).product::<i128>();
                let numel_base = {
                    let b = ctx.base(&r);
                    ctx.shape(&b).ok().flatten()?.iter().map(|s| s.clone().as_i128_or_panic()).product::<i128>()
                };
                if numel_r != numel_base || ctx.contiguous_view_offset(r.clone()).is_none() {
                    let contig = ctx.uop(Ops::Contiguous, None, vec![r], Arg::None);
                    return Some(ctx.uop_tag(Ops::Copy, Some(ctx.dt_of(&c)), vec![contig], ctx.arg_of(&c).clone(), ctx.tag_of(&c).clone()));
                }
                None
            }),
        ),
        // copy to same device is a no-op
        (
            UPat::ops1(Ops::Copy).with_src_tup(vec![UPat::var(Some("x"), None)]).named("copy"),
            r_ctx(|ctx, st| {
                let (x, copy) = (sym::sget(st, "x"), sym::sget(st, "copy"));
                let xdev = super::indexing::device_of(ctx, x.clone());
                let cdev = match ctx.arg_of(&copy) {
                    Arg::Str(s) => Some(DeviceSpec::Single(s.clone())),
                    _ => None,
                };
                if xdev == cdev {
                    Some(x)
                } else {
                    None
                }
            }),
        ),
        // copy on reshape is reshape on copy
        (
            UPat::ops1(Ops::Copy)
                .with_src_tup(vec![UPat::ops1(Ops::Reshape).with_src_tup(vec![UPat::var(Some("shp0"), None)]).named("shp")])
                .named("cpy"),
            r_ctx(|ctx, st| {
                let (shp, cpy) = (sym::sget(st, "shp"), sym::sget(st, "cpy"));
                let inner = ctx.srcs(&shp)[0].clone();
                let dev = match ctx.arg_of(&cpy) {
                    Arg::Str(s) => s.clone(),
                    other => panic!("TGERR: bad copy device {other:?}"),
                };
                let cp = crate::tensor::creation::copy_to_device(ctx, inner, &dev);
                let shp_shape = ctx.shape(&shp).ok().flatten()?;
                Some(crate::tensor::movement::reshape(ctx, cp, &shp_shape))
            }),
        ),
        // reshaping on STORE can be a NOOP
        (
            UPat::ops1(Ops::Store)
                .with_src_tup(vec![
                    UPat::ops1(Ops::Reshape).with_src_tup(vec![UPat::var(Some("dst"), None)]).allow_all_srcs(),
                    UPat::ops1(Ops::Reshape).with_src_tup(vec![UPat::var(Some("src"), None)]).allow_all_srcs(),
                ])
                .named("st"),
            r_ctx(|ctx, st| {
                let dst = sym::sget(st, "dst");
                let src = sym::sget(st, "src");
                if ctx.shape(&dst).ok().flatten()? == ctx.shape(&src).ok().flatten()? {
                    Some(crate::tensor::store(ctx, dst, src, None))
                } else {
                    None
                }
            }),
        ),
        // fix store hazard
        (
            UPat::ops1(Ops::Store).with_src_tup(vec![UPat::var(Some("target"), None), UPat::var(Some("src"), None)]),
            r_ctx(|ctx, st| fix_store_hazard(ctx, sym::sget(st, "target"), sym::sget(st, "src"))),
        ),
        // move bitcast from store dest to source
        (
            UPat::ops1(Ops::Store)
                .with_src_tup(vec![UPat::ops1(Ops::Bitcast).with_src_tup(vec![UPat::var(Some("target"), None)]), UPat::var(Some("src"), None)]),
            r_ctx(|ctx, st| {
                let (target, src) = (sym::sget(st, "target"), sym::sget(st, "src"));
                let b = ctx.bitcast(src, ctx.dt_of(&target));
                Some(crate::tensor::store(ctx, target, b, None))
            }),
        ),
        (
            UPat::ops1(Ops::Bitcast).named("bc"),
            r_ctx(|ctx, st| expand_bitcast(ctx, sym::sget(st, "bc"))),
        ),
    ];
    PatternMatcher::new(rules).add(pm_mop_cleanup())
}

/// `pm_copy_to_store` (rangeify.py:579).
pub fn pm_copy_to_store() -> PatternMatcher {
    static PM: crate::uop::upat::PmSlot = crate::uop::upat::PmSlot::new();
    PM.get_or_init(build_pm_copy_to_store).clone()
}

fn build_pm_copy_to_store() -> PatternMatcher {
    let rules: Vec<(UPat, RuleFn)> = vec![
        // existing buffer stores a copy
        (
            UPat::var(Some("existing_buf"), None)
                .store(vec![UPat::ops1(Ops::Copy).named("copy")])
                .named("eb"),
            r_cnt(|cnt, ctx, st| {
                let existing = sym::sget(st, "existing_buf");
                let copy = sym::sget(st, "copy");
                convert_copy_to_store(cnt, ctx, copy, Some(existing))
            }),
        ),
        // plain copy
        (
            UPat::ops1(Ops::Copy).named("copy"),
            r_cnt(|cnt, ctx, st| convert_copy_to_store(cnt, ctx, sym::sget(st, "copy"), None)),
        ),
    ];
    PatternMatcher::new(rules)
}

/// `convert_copy_to_store` (rangeify.py:565).
fn convert_copy_to_store(cnt: &std::cell::Cell<i64>, ctx: &mut UOpCtx, copy: UopId, existing_buf: Option<UopId>) -> Option<UopId> {
    let input_src0 = ctx.srcs(&copy)[0].clone();
    let input_src = if !ctx.has_buffer_identity(input_src0.clone(), true) {
        ctx.uop(Ops::Contiguous, None, vec![input_src0], Arg::None)
    } else {
        input_src0
    };
    let input_src = crate::tensor::movement::flatten(ctx, input_src, -1, -1);
    let copy_dev = match ctx.arg_of(&copy) {
        Arg::Str(s) => s.clone(),
        other => panic!("TGERR: bad copy device {other:?}"),
    };
    if let Some(existing_buf) = existing_buf {
        if !ctx.has_buffer_identity(existing_buf.clone(), true) {
            return None;
        }
        let flat = crate::tensor::movement::flatten(ctx, existing_buf, -1, -1);
        return Some(crate::tensor::store(ctx, flat, input_src, None));
    }
    // create the output buffer
    let max_shape = ctx.max_shape(input_src.clone());
    let size: i64 = max_shape.iter().product();
    let slot = cnt.get();
    cnt.set(slot + 1);
    let pa = ParamArg {
        slot,
        dtype: ctx.dt_of(&copy),
        vmin_vmax: None,
        multiple_of: None,
        name: None,
        addrspace: Some(AddrSpace::Global),
        axis: None,
        device: Some(DeviceSpec::Single(copy_dev)),
        volatile: false,
    };
    let shape_src = ctx.shape_arg(&[SVal::I(size as i128)]);
    let buf = ctx.uop(Ops::Buffer, Some(ctx.dt_of(&copy)), vec![shape_src], Arg::Param(pa));
    let st = crate::tensor::store(ctx, buf.clone(), input_src, None);
    let aft = crate::tensor::after(ctx, buf, vec![st]);
    let copy_shape = ctx.shape(&copy).ok().flatten()?;
    Some(crate::tensor::movement::reshape(ctx, aft, &copy_shape))
}

// ---------- 3.5 cleanups: pm_const_buffer_folding + pm_remove_bufferize ----------

const ALWAYS_RUN_OPS: &[Ops] = &[Ops::Contiguous, Ops::Noop];

/// `cleanup_dead_axes` (rangeify.py:195).
fn cleanup_dead_axes(ctx: &mut UOpCtx, b: UopId) -> Option<UopId> {
    let Arg::Data(d) = ctx.arg_of(&b) else { return None };
    if !d.as_bool("removable") {
        return None;
    }
    let s0 = ctx.srcs(&b)[0].clone();
    if ALWAYS_RUN_OPS.contains(&ctx.op_of(&s0)) || ctx.op_of(&s0) == Ops::After {
        return None;
    }
    let b_shape = ctx.shape(&b).ok().flatten()?;
    let rngs = ctx.srcs(&b)[1..].to_vec();
    let mut new_rng: Vec<UopId> = Vec::new();
    let mut hit = false;
    let mut reshape: Vec<SVal> = Vec::new();
    for (i, (s, rng)) in b_shape.iter().zip(rngs.iter()).enumerate() {
        let _ = i;
        if ctx.op_of(rng) == Ops::Range {
            let end = ctx.srcs(rng)[0].clone();
            if ctx.op_of(&end) != Ops::Const {
                return None; // skip for symbolic
            }
        }
        let rng_in_body = ctx.ranges(&s0.clone()).contains(rng);
        if ctx.op_of(rng) == Ops::Const || (ctx.op_of(rng) == Ops::Range && !rng_in_body) {
            reshape.push(SVal::I(1));
            hit = true;
        } else {
            reshape.push(s.clone());
            new_rng.push(rng.clone());
        }
    }
    if hit {
        let (op, dt, _srcs, arg, tag) = ctx.snapshot(&b);
        let mut new_srcs = vec![s0];
        new_srcs.extend(new_rng);
        let nb = ctx.uop_tag(op, Some(dt), new_srcs, arg, tag);
        let reshaped = crate::tensor::movement::reshape(ctx, nb, &reshape);
        return Some(crate::tensor::movement::expand(ctx, reshaped, &b_shape));
    }
    None
}

/// `remove_noop_bufferize` (rangeify.py:290).
fn remove_noop_bufferize(ctx: &mut UOpCtx, idx: UopId, b2: UopId) -> Option<UopId> {
    if ctx.srcs(&idx)[1..] != ctx.srcs(&b2)[1..] {
        return None;
    }
    let b2_shape = ctx.shape(&b2).ok().flatten()?;
    if b2_shape.is_empty() {
        return Some(ctx.srcs(&idx)[0].clone());
    }
    let bounds: Vec<Arg> = b2_shape.iter().map(|s| sval_pair(SVal::I(0), s.clone())).collect();
    Some(shrink_args(ctx, ctx.srcs(&idx)[0].clone(), &bounds))
}

/// SHRINK with concrete (start, size)... python bounds are (start, stop) via
/// pairs of ints — build through movement::shrink's (start, size) pairs.
fn shrink_args(ctx: &mut UOpCtx, x: UopId, bounds: &[Arg]) -> UopId {
    let pairs: Vec<(SVal, SVal)> = bounds
        .iter()
        .map(|a| match a {
            Arg::Seq { tuple: true, items } => match (&items[0], &items[1]) {
                (Arg::Int(st), Arg::Int(sz)) => (SVal::I(*st as i128), SVal::I(*sz as i128)),
                other => panic!("TGERR: bad shrink bound {other:?}"),
            },
            other => panic!("TGERR: bad shrink bound {other:?}"),
        })
        .collect();
    crate::tensor::movement::shrink(ctx, x, &pairs)
}

fn sval_pair(a: SVal, b: SVal) -> Arg {
    Arg::Seq {
        tuple: true,
        items: vec![
            Arg::Int(a.as_i128_or_panic() as i64),
            Arg::Int(b.as_i128_or_panic() as i64),
        ],
    }
}

/// `after_all_invalid` (rangeify.py:294).
fn after_all_invalid(ctx: &mut UOpCtx, after: UopId) -> bool {
    let buf = ctx.buf_uop(ctx.srcs(&after)[0].clone());
    let srcs = ctx.srcs(&after).to_vec();
    if srcs.is_empty() {
        return false;
    }
    let mut all_stores_valid = true;
    for s in &srcs[1..] {
        if ctx.op_of(s) != Ops::End {
            all_stores_valid = false;
            break;
        }
        let ssrcs = ctx.srcs(s).to_vec();
        if ssrcs.is_empty() {
            all_stores_valid = false;
            break;
        }
        let st = ssrcs[0].clone();
        if ctx.op_of(&st) != Ops::Store {
            all_stores_valid = false;
            break;
        }
        let stsrcs = ctx.srcs(&st).to_vec();
        if !ctx.is_invalid(&ctx.base(&stsrcs[1])) || ctx.buf_uop((stsrcs[0]).clone()) != buf {
            all_stores_valid = false;
            break;
        }
        // all ended ranges used
        let ended = ctx.ended_ranges(&s.clone());
        let _ = ended;
    }
    if !all_stores_valid {
        return false;
    }
    // resolve(prod(r.src[0] for r in s.ended_ranges).eq(buf.numel())) — concrete only
    let numel = ctx.max_numel(buf);
    for s in &srcs[1..] {
        let ended = ctx.ended_ranges(&s.clone());
        let mut prod_ends: i128 = 1;
        for r in &ended {
            let e = ctx.srcs(r)[0].clone();
            let mn = ctx.min_max(&e).map(|(mn, _)| mn).unwrap_or(SVal::I(0));
            prod_ends *= mn.as_i128_or_panic();
        }
        if prod_ends as i64 != numel {
            return false;
        }
    }
    true
}

/// `pm_const_buffer_folding` (rangeify.py:301).
pub fn pm_const_buffer_folding() -> PatternMatcher {
    static PM: crate::uop::upat::PmSlot = crate::uop::upat::PmSlot::new();
    PM.get_or_init(build_pm_const_buffer_folding).clone()
}

fn build_pm_const_buffer_folding() -> PatternMatcher {
    let rules: Vec<(UPat, RuleFn)> = vec![
        (UPat::ops1(Ops::Stage).allow_all_srcs().named("b"), r_ctx(|ctx, st| cleanup_dead_axes(ctx, sym::sget(st, "b")))),
        // remove noop buffers (INDEX of STAGE with same rngs)
        (
            UPat::ops1(Ops::Index).allow_all_srcs().named("idx").f(Ops::Stage).with_src_any_len(true).named("b2"),
            r_ctx(|ctx, st| remove_noop_bufferize(ctx, sym::sget(st, "idx"), sym::sget(st, "b2"))),
        ),
        (
            UPat::ops1(Ops::Index)
                .with_src_tup(vec![UPat::ops1(Ops::Stage).allow_all_srcs()])
                .allow_all_srcs()
                .named("idx")
                .f(Ops::Noop)
                .f(Ops::Stage)
                .with_src_any_len(true)
                .named("b2"),
            r_ctx(|ctx, st| remove_noop_bufferize(ctx, sym::sget(st, "idx"), sym::sget(st, "b2"))),
        ),
        // no buffers for const
        (
            UPat::ops1(Ops::Const).named("c").f(Ops::Stage).with_src_any_len(true).named("b"),
            r_ctx(|ctx, st| {
                let (c, b) = (sym::sget(st, "c"), sym::sget(st, "b"));
                let v = ctx.val(&c);
                Some(ctx.const_like_dt(b, v, None))
            }),
        ),
        // indexing a const is a const
        (
            UPat::ops1(Ops::Index).with_src_tup(vec![UPat::ops1(Ops::Const).named("c")]).allow_all_srcs(),
            r_ctx(|_ctx, st| Some(sym::sget(st, "c"))),
        ),
        // indexing an after with all fully invalid stores is invalid
        (
            UPat::ops1(Ops::Index)
                .with_src_tup(vec![UPat::ops1(Ops::After).allow_all_srcs().named("after")])
                .allow_all_srcs()
                .named("idx"),
            r_ctx(|ctx, st| {
                let (idx, after) = (sym::sget(st, "idx"), sym::sget(st, "after"));
                if after_all_invalid(ctx, after) {
                    Some(ctx.const_like_dt(idx, ConstVal::Invalid, None))
                } else {
                    None
                }
            }),
        ),
        // hack if a noop turned to a const
        (UPat::ops1(Ops::Noop).with_src_tup(vec![sym::cvar("c")]), r_ctx(|_ctx, st| Some(sym::sget(st, "c")))),
        // a deviceless MSTACK src is the same value on every device
        (
            UPat::ops1(Ops::Mstack)
                .with_src_tup(vec![UPat::var(Some("s"), None)])
                .allow_all_srcs()
                .f(Ops::Index)
                .with_src_any_len(true)
                .named("idx"),
            r_ctx(|ctx, st| {
                let (s, idx) = (sym::sget(st, "s"), sym::sget(st, "idx"));
                if super::indexing::device_of(ctx, s.clone()).is_none() {
                    let mut srcs = vec![s];
                    srcs.extend(ctx.srcs(&idx)[1..].iter().cloned());
                    Some(ctx.uop_tag(Ops::Index, Some(ctx.dt_of(&idx)), srcs, ctx.arg_of(&idx).clone(), ctx.tag_of(&idx).clone()))
                } else {
                    None
                }
            }),
        ),
    ];
    pm_mops().add(&PatternMatcher::new(rules))
}

/// `pm_remove_bufferize` (rangeify.py:321).
pub fn pm_remove_bufferize() -> PatternMatcher {
    static PM: crate::uop::upat::PmSlot = crate::uop::upat::PmSlot::new();
    PM.get_or_init(build_pm_remove_bufferize).clone()
}

fn build_pm_remove_bufferize() -> PatternMatcher {
    let rules: Vec<(UPat, RuleFn)> = vec![
        // remove reindexing with cost function
        (
            UPat::var(Some("src"), None)
                .f(Ops::Stage)
                .with_src_any_len(true)
                .named("buf")
                .f(Ops::Index)
                .with_src_any_len(true)
                .named("idx"),
            r_ctx(|ctx, st| remove_bufferize(ctx, sym::sget(st, "src"), sym::sget(st, "buf"), sym::sget(st, "idx"))),
        ),
        // STORE to self is NOOP
        (
            UPat::ops1(Ops::Store).with_src_tup(vec![UPat::var(Some("x"), None), UPat::var(Some("x"), None)]),
            r_ctx(|ctx, _st| {
                let noop = ctx.uop(Ops::Noop, None, vec![], Arg::None);
                Some(noop)
            }),
        ),
        // END on NOOP is NOOP
        (
            UPat::ops1(Ops::End).with_src_tup(vec![UPat::ops1(Ops::Noop).named("x")]).allow_all_srcs(),
            r_ctx(|_ctx, st| Some(sym::sget(st, "x"))),
        ),
    ];
    PatternMatcher::new(rules)
}

/// `remove_bufferize` (rangeify.py:221).
fn remove_bufferize(ctx: &mut UOpCtx, src: UopId, buf: UopId, idx: UopId) -> Option<UopId> {
    let (_buf_op, _bdt, buf_srcs, buf_arg, _btag) = ctx.snapshot(&buf);
    let idx_srcs = ctx.srcs(&idx).to_vec();
    let Arg::Data(d) = &buf_arg else { return None };
    assert_eq!(buf_srcs.len(), idx_srcs.len(), "TGERR: index on wrong bufferize");
    assert!(
        buf_srcs[1..].iter().all(|x| matches!(ctx.op_of(x), Ops::Range | Ops::Const)),
        "TGERR: bad bufferize rngs"
    );
    // if it's user contiguous, we never remove it
    if ALWAYS_RUN_OPS.contains(&ctx.op_of(&src)) || !d.as_bool("removable") {
        return None;
    }
    // accessed buffers via gated toposort (red_gate)
    let accessed: std::cell::RefCell<(Vec<UopId>, Vec<UopId>, Vec<UopId>)> =
        std::cell::RefCell::new((Vec::new(), Vec::new(), Vec::new()));
    {
        let gate2 = |x: UopId| -> bool {
            if ctx.op_of(&x) == Ops::After {
                accessed.borrow_mut().0.push(ctx.srcs(&x)[0].clone()); // AFTER marker: mapped via buf_uop after the walk
                return false;
            }
            if ctx.op_of(&x) == Ops::Stage
                && let Arg::Data(dd) = ctx.arg_of(&x)
                    && matches!(dd.get("addrspace"), Arg::Enum(e) if e.value == AddrSpace::Global as u16) {
                        accessed.borrow_mut().0.push(x);
                        return false;
                    }
            if ctx.op_of(&x) == Ops::Mstack {
                accessed.borrow_mut().0.push(x);
                return false;
            }
            if ctx.op_of(&x) == Ops::Store {
                return false;
            }
            if ctx.op_of(&x) == Ops::Param {
                accessed.borrow_mut().0.push(x.clone());
            }
            if ctx.op_of(&x) == Ops::Index {
                accessed.borrow_mut().1.push(x.clone());
            }
            if ctx.op_of(&x) == Ops::Reduce {
                accessed.borrow_mut().2.push(x);
            }
            true
        };
        ctx.toposort_gated(src.clone(), Some(&gate2), true);
    }
    let (mut accessed_buffers, mut indexes, mut reduces) = accessed.into_inner();
    // AFTER markers were raw src nodes; map through buf_uop now
    for a in accessed_buffers.iter_mut() {
        if ctx.op_of(a) == Ops::After {
            *a = ctx.buf_uop(a.clone());
        }
    }
    let _ = (&mut indexes, &mut reduces);
    accessed_buffers.dedup();
    // if this is generated from multiple buffers, don't remove this buffer
    if accessed_buffers.len() > 3 {
        return None;
    }
    // if any reduces access a buffer, don't remove this buffer
    let mut buffer_in_reduce = false;
    {
        let reduce_srcs: Vec<UopId> = reduces.iter().map(|r| ctx.srcs(r)[0].clone()).collect();
        let sink = crate::tensor::sink(ctx, reduce_srcs);
        let buf_gate = |_x: UopId| -> bool {
            if buffer_in_reduce {
                return false;
            }
            true
        };
        // visit and set flag: any PARAM/STAGE/AFTER inside => buffer_in_reduce
        let ts = ctx.toposort_gated(sink, Some(&buf_gate), true);
        for x in ts {
            if matches!(ctx.op_of(&x), Ops::Param | Ops::Stage | Ops::After) {
                buffer_in_reduce = true;
                break;
            }
        }
    }
    if buffer_in_reduce {
        return None; // PCONTIG=0
    }
    // if it makes it here, the bufferize is removed
    let mut replaced: HashMap<UopId, UopId> = HashMap::new();
    for (k, v) in buf_srcs[1..].iter().zip(idx_srcs[1..].iter()) {
        if ctx.op_of(k) == Ops::Const {
            continue;
        }
        if ctx.op_of(v) == Ops::Const && ctx.is_invalid(v) {
            continue;
        }
        replaced.insert(k.clone(), v.clone());
    }
    Some(crate::uop::upat::substitute(ctx, src, replaced, false, false))
}

// ---------- 4. put in buffers for bufferize (rangeify.py:392-463) ----------

/// `bufferize_to_store` (rangeify.py:392).
fn bufferize_to_store(cnt: &std::cell::Cell<i64>, ctx: &mut UOpCtx, x: UopId, idx: UopId, allow_locals: bool) -> Option<UopId> {
    let x_shape = ctx.shape(&x).ok().flatten()?;
    let size: i128 = x_shape.iter().map(|s| s.clone().as_i128_or_panic()).product();
    let x_dt = ctx.dt_of(&x);
    let dtype = crate::dtype::strong_dtype(x_dt);
    let mut rngs = ctx.ranges(&idx.clone()).to_vec();
    rngs.sort_by_key(|r| AxisType::range_axis_id(ctx, r.clone()));
    assert!(size > 0, "TGERR: no zero sized or symbolic sized buffers");

    let x_srcs = ctx.srcs(&x).to_vec();
    // AFTER: add END to the existing STORE, return buffer with kernel dependency
    if ctx.op_of(&x_srcs[0]) == Ops::After {
        let after = &x_srcs[0].clone();
        let a0 = &ctx.srcs(after)[0].clone();
        let bu = ctx.buf_uop(a0.clone());
        let buf = ctx.base(&bu);
        let stores: Vec<UopId> = ctx
            .srcs(after)[1..]
            .iter().filter(|&s| ctx.op_of(s) == Ops::Store && ctx.op_of(&ctx.srcs(s)[0]) == Ops::Index).cloned()
            .collect();
        if stores.is_empty() {
            return Some(buf);
        }
        let mut ended_stores = Vec::new();
        for store in &stores {
            let ssrcs = ctx.srcs(store).to_vec();
            let mut store_target = ssrcs[0].clone();
            {
                let tsrcs = ctx.srcs(&store_target).to_vec();
                if ctx.op_of(&tsrcs[0]) == Ops::Stage {
                    let inner = ctx.srcs(&tsrcs[0]).to_vec();
                    if ctx.op_of(&inner[0]) == Ops::Index {
                        store_target = inner[0].clone();
                    }
                }
            }
            if ssrcs[1] == store_target {
                continue; // skip self-assign
            }
            let mut end_set: Vec<UopId> = ctx.ranges(&store_target.clone()).to_vec();
            for r in &rngs {
                if !end_set.contains(r) {
                    end_set.push(r.clone());
                }
            }
            end_set.sort_by_key(|r| AxisType::range_axis_id(ctx, r.clone()));
            let st = crate::tensor::store(ctx, store_target.clone(), (ssrcs[1]).clone(), None);
            ended_stores.push(ctx.end(st, end_set));
        }
        return Some(crate::tensor::after(ctx, buf, ended_stores));
    }

    let Arg::Data(d) = ctx.arg_of(&x) else { panic!("TGERR: STAGE without BufferizeOpts") };
    let addrspace = match d.get("addrspace") {
        Arg::Enum(e) => AddrSpace::from_u8(e.value as u8).expect("TGERR: bad addrspace"),
        _ => panic!("TGERR: STAGE without addrspace"),
    };
    let dev: Option<DeviceSpec> = d.get("device").clone().try_into().ok().flatten();

    if addrspace == AddrSpace::Global {
        let slot = cnt.get();
        cnt.set(slot + 1);
        let pa = ParamArg {
            slot,
            dtype,
            vmin_vmax: None,
            multiple_of: None,
            name: None,
            addrspace: Some(AddrSpace::Global),
            axis: None,
            device: dev,
            volatile: false,
        };
        let shape_src = ctx.shape_arg(&[SVal::I(size)]);
        let buf = ctx.uop(Ops::Buffer, Some(dtype), vec![shape_src], Arg::Param(pa));
        let indexed = ctx.index(buf.clone(), vec![idx]);
        let val = ctx.cast((x_srcs[0]).clone(), dtype);
        let st = crate::tensor::store(ctx, indexed, val, None);
        let ended = ctx.end(st, rngs);
        let aft = crate::tensor::after(ctx, buf, vec![ended]);
        return Some(ctx.cast(aft, x_dt));
    }

    if allow_locals {
        let slot = cnt.get();
        cnt.set(slot + 1);
        let pa = ParamArg {
            slot,
            dtype,
            vmin_vmax: None,
            multiple_of: None,
            name: None,
            addrspace: Some(AddrSpace::Local),
            axis: None,
            device: None,
            volatile: false,
        };
        let shape_src = ctx.shape_arg(&[SVal::I(size)]);
        let buf = ctx.uop(Ops::Buffer, Some(dtype), vec![shape_src], Arg::Param(pa));
        let indexed = ctx.index(buf.clone(), vec![idx]);
        let val = ctx.cast((x_srcs[0]).clone(), dtype);
        let st = crate::tensor::store(ctx, indexed, val, None);
        let ended = ctx.end(st, rngs);
        let aft = crate::tensor::after(ctx, buf, vec![ended]);
        return Some(ctx.cast(aft, x_dt));
    }
    None
}

/// `add_local_buffer` (codegen/__init__.py:245): turn a remaining (local)
/// STAGE into a placeholder local BUFFER + `index/store/end`, wrapped in an
/// AFTER. Runs at the codegen "add local buffers" stage (after scheduling), so
/// GLOBAL STAGEs were already consumed by `bufferize_to_store`.
fn add_local_buffer(cnt: &std::cell::Cell<i64>, ctx: &mut UOpCtx, x: UopId) -> Option<UopId> {
    let x_shape = ctx.shape(&x).ok().flatten()?;
    let size: i128 = x_shape.iter().map(|s| s.clone().as_i128_or_panic()).product();
    assert!(size > 0, "TGERR: no zero sized or symbolic sized buffers");
    let x_dt = ctx.dt_of(&x);
    let dtype = crate::dtype::strong_dtype(x_dt);
    let x_srcs = ctx.srcs(&x).to_vec();
    let idxs = x_srcs[1..].to_vec();
    let idx = idxs[0].clone();
    let mut rngs = ctx.ranges(&idx.clone()).to_vec();
    rngs.sort_by_key(|r| AxisType::range_axis_id(ctx, r.clone()));

    let slot = cnt.get();
    cnt.set(slot + 1);
    let pa = ParamArg {
        slot,
        dtype,
        vmin_vmax: None,
        multiple_of: None,
        name: None,
        addrspace: Some(AddrSpace::Local),
        axis: None,
        device: None,
        volatile: false,
    };
    let shape_src = ctx.shape_arg(&[SVal::I(size)]);
    let buf = ctx.uop(Ops::Buffer, Some(dtype), vec![shape_src], Arg::Param(pa));
    let indexed = ctx.index(buf.clone(), vec![idx]);
    let val = ctx.cast((x_srcs[0]).clone(), dtype);
    let st = crate::tensor::store(ctx, indexed, val, None);
    let ended = ctx.end(st, rngs);
    let aft = crate::tensor::after(ctx, buf, vec![ended]);
    Some(ctx.cast(aft, x_dt))
}

/// `pm_add_local_buffers` (codegen/__init__.py:252).
pub fn pm_add_local_buffers() -> PatternMatcher {
    static PM: crate::uop::upat::PmSlot = crate::uop::upat::PmSlot::new();
    PM.get_or_init(build_pm_add_local_buffers).clone()
}

fn build_pm_add_local_buffers() -> PatternMatcher {
    PatternMatcher::new(vec![(
        UPat::ops1(Ops::Stage).allow_all_srcs().named("x"),
        r_cnt(|cnt, ctx, st| add_local_buffer(cnt, ctx, sym::sget(st, "x"))),
    )])
}

/// `flatten_bufferize` (rangeify.py:426).
fn flatten_bufferize(ctx: &mut UOpCtx, x: UopId) -> Option<UopId> {
    let x_srcs = ctx.srcs(&x).to_vec();
    if x_srcs.len() == 2 {
        return None;
    }
    let x_shape = ctx.shape(&x).ok().flatten()?;
    let prod: SVal = crate::tensor::movement::prod_sval(ctx, &x_shape);
    let in_shape: Vec<SVal> = prod_shape_1(&prod);
    let rngs = x_srcs[1..].to_vec();
    let flat = apply_movement_op(ctx, Ops::Reshape, &in_shape, &svals_to_args(&x_shape), &rngs);
    assert_eq!(flat.len(), 1, "TGERR: flatten_bufferize multi index");
    let (op, dt, _old, arg, tag) = ctx.snapshot(&x);
    let ret = ctx.uop_tag(op, Some(dt), vec![(x_srcs[0]).clone(), (flat[0]).clone()], arg, tag);
    let ret = crate::tensor::movement::reshape(ctx, ret, &x_shape);
    if rngs.iter().any(|r| ctx.op_of(r) == Ops::Range && ctx.op_of(&ctx.srcs(r)[0]) != Ops::Const) {
        let sym_shape: Vec<SVal> = rngs
            .iter()
            .map(|r| if ctx.op_of(r) != Ops::Const { SVal::U(ctx.srcs(r)[0].clone()) } else { SVal::I(1) })
            .collect();
        let bounds: Vec<Arg> = sym_shape.iter().map(|s| sval_pair(SVal::I(0), s.clone())).collect();
        return Some(shrink_args(ctx, ret, &bounds));
    }
    Some(ret)
}

fn prod_shape_1(p: &SVal) -> Vec<SVal> {
    vec![p.clone()]
}

/// `is_noop_after_dep` / `remove_noop_afters` (rangeify.py:437).
fn remove_noop_afters(ctx: &mut UOpCtx, x: UopId) -> Option<UopId> {
    fn is_noop_after_dep(ctx: &UOpCtx, u: UopId) -> bool {
        if ctx.op_of(&u) == Ops::Noop && ctx.srcs(&u).is_empty() {
            return true;
        }
        ctx.op_of(&u) == Ops::End && !ctx.srcs(&u).is_empty() && is_noop_after_dep(ctx, ctx.srcs(&u)[0].clone())
    }
    let x_srcs = ctx.srcs(&x).to_vec();
    let kept: Vec<UopId> = std::iter::once((x_srcs[0]).clone())
        .chain(x_srcs[1..].iter().filter(|&s| !is_noop_after_dep(ctx, s.clone())).cloned())
        .collect();
    if kept.len() != x_srcs.len() {
        if kept.len() == 1 {
            return Some((kept[0]).clone());
        }
        return Some(ctx.uop_tag(ctx.op_of(&x), Some(ctx.dt_of(&x)), kept, ctx.arg_of(&x).clone(), ctx.tag_of(&x).clone()));
    }
    None
}

/// `pm_add_buffers` (rangeify.py:445).
pub fn pm_add_buffers() -> PatternMatcher {
    static PM: crate::uop::upat::PmSlot = crate::uop::upat::PmSlot::new();
    PM.get_or_init(build_pm_add_buffers).clone()
}

fn build_pm_add_buffers() -> PatternMatcher {
    let rules: Vec<(UPat, RuleFn)> = vec![
        (
            UPat::ops1(Ops::Stage)
                .with_src_tup(vec![UPat::var(None, None), UPat::var(Some("idx"), None)])
                .named("x"),
            r_cnt(|cnt, ctx, st| bufferize_to_store(cnt, ctx, sym::sget(st, "x"), sym::sget(st, "idx"), false)),
        ),
        // INDEX of a buffer through the weak cast: index the buffer directly
        (
            UPat::ops1(Ops::Index)
                .with_src_tup(vec![UPat::ops1(Ops::Cast).with_dtype(Some(crate::dtype::WEAKS.to_vec())).with_src_tup(vec![UPat::var(Some("buf"), None)])])
                .allow_all_srcs()
                .named("u"),
            r_ctx(|ctx, st| {
                let (u, buf) = (sym::sget(st, "u"), sym::sget(st, "buf"));
                let mut srcs = vec![buf];
                srcs.extend(ctx.srcs(&u)[1..].iter().cloned());
                let udt = ctx.dt_of(&u);
                // u.replace(dtype=None, src=...).cast(u.dtype) — dtype None means keep src dtype
                let replaced = ctx.uop_tag(Ops::Index, None, srcs, ctx.arg_of(&u).clone(), ctx.tag_of(&u).clone());
                Some(ctx.cast(replaced, udt))
            }),
        ),
        // move RESHAPEs through MSELECT/MSTACK
        (
            UPat::op_var(vec![Ops::Mselect, Ops::Mstack])
                .with_src_all_reshape()
                .named("m"),
            r_ctx(|ctx, st| {
                let m = sym::sget(st, "m");
                let srcs = ctx.srcs(&m).to_vec();
                let new_srcs: Vec<UopId> = srcs.iter().map(|x| ctx.base(&ctx.srcs(x)[0])).collect();
                let m_shape = ctx.shape(&m).ok().flatten()?;
                let replaced = ctx.uop_tag(ctx.op_of(&m), Some(ctx.dt_of(&m)), new_srcs, ctx.arg_of(&m).clone(), ctx.tag_of(&m).clone());
                Some(crate::tensor::movement::reshape(ctx, replaced, &m_shape))
            }),
        ),
        // remove any RESHAPEs on CALL
        (
            UPat::ops1(Ops::Call).allow_all_srcs().named("k"),
            r_ctx(|ctx, st| {
                let k = sym::sget(st, "k");
                let srcs = ctx.srcs(&k).to_vec();
                let new_srcs: Vec<UopId> = srcs
                    .iter()
                    .map(|x| if ctx.op_of(x) == Ops::Reshape { ctx.srcs(x)[0].clone() } else { x.clone() })
                    .collect();
                if new_srcs == srcs {
                    None
                } else {
                    Some(ctx.uop_tag(Ops::Call, Some(ctx.dt_of(&k)), new_srcs, ctx.arg_of(&k).clone(), ctx.tag_of(&k).clone()))
                }
            }),
        ),
        // remove invalid writes
        (
            UPat::ops1(Ops::Store)
                .with_src_tup(vec![UPat::var(None, None), UPat::ops1(Ops::Contiguous).with_src_tup(vec![UPat::ops1(Ops::Const).with_arg(Some(Arg::Const(ConstVal::Invalid)))]).allow_all_srcs()]),
            r_ctx(|ctx, _st| {
                let noop = ctx.uop(Ops::Noop, None, vec![], Arg::None);
                Some(noop)
            }),
        ),
        (
            UPat::ops1(Ops::Store)
                .with_src_tup(vec![UPat::var(None, None), UPat::ops1(Ops::Const).with_arg(Some(Arg::Const(ConstVal::Invalid)))]),
            r_ctx(|ctx, _st| {
                let noop = ctx.uop(Ops::Noop, None, vec![], Arg::None);
                Some(noop)
            }),
        ),
        (UPat::ops1(Ops::After).allow_all_srcs().named("x"), r_ctx(|ctx, st| remove_noop_afters(ctx, sym::sget(st, "x")))),
    ];
    pm_mops()
        .add(&PatternMatcher::new(vec![(UPat::ops1(Ops::Stage).allow_all_srcs().named("x"), r_ctx(|ctx, st| flatten_bufferize(ctx, sym::sget(st, "x"))))]))
        .add(&PatternMatcher::new(rules))
}

/// `pm_add_param_range_tags` (rangeify.py:543).
pub fn pm_add_param_range_tags() -> PatternMatcher {
    static PM: crate::uop::upat::PmSlot = crate::uop::upat::PmSlot::new();
    PM.get_or_init(build_pm_add_param_range_tags).clone()
}

fn build_pm_add_param_range_tags() -> PatternMatcher {
    PatternMatcher::new(vec![(
        UPat::op_var(vec![Ops::Param, Ops::Range]).named("x"),
        r_ctx(|ctx, st| {
            let x = sym::sget(st, "x");
            let empty = Arg::Seq { tuple: true, items: vec![] };
            if matches!(ctx.tag_of(&x), Arg::Seq { tuple: true, items } if items.is_empty()) {
                return None;
            }
            Some(ctx.uop_tag(ctx.op_of(&x), Some(ctx.dt_of(&x)), ctx.srcs(&x).to_vec(), ctx.arg_of(&x).clone(), empty))
        }),
    )])
}

/// `pm_no_indexing_calls` (rangeify.py:350).
pub fn pm_no_indexing_calls() -> PatternMatcher {
    static PM: crate::uop::upat::PmSlot = crate::uop::upat::PmSlot::new();
    PM.get_or_init(build_pm_no_indexing_calls).clone()
}

fn build_pm_no_indexing_calls() -> PatternMatcher {
    PatternMatcher::new(vec![(
        UPat::ops1(Ops::Call).allow_all_srcs().named("u"),
        r_ctx(|ctx, st| {
            let u = sym::sget(st, "u");
            let srcs = ctx.srcs(&u).to_vec();
            let mut new_srcs = Vec::new();
            for x in srcs {
                if ctx.op_of(&x) == Ops::Index {
                    new_srcs.push(ctx.srcs(&x)[0].clone());
                } else if ctx.op_of(&x) == Ops::Shrink {
                    new_srcs.push(strip_zero_offset_shrink(ctx, x));
                } else if ctx.op_of(&x) == Ops::Mstack {
                    let msrcs = ctx.srcs(&x).to_vec();
                    let stripped: Vec<UopId> = msrcs.iter().map(|s| strip_zero_offset_shrink(ctx, s.clone())).collect();
                    new_srcs.push(ctx.uop_tag(Ops::Mstack, Some(ctx.dt_of(&x)), stripped, ctx.arg_of(&x).clone(), ctx.tag_of(&x).clone()));
                } else {
                    new_srcs.push(x);
                }
            }
            if new_srcs == ctx.srcs(&u).to_vec() {
                None
            } else {
                Some(ctx.uop_tag(Ops::Call, Some(ctx.dt_of(&u)), new_srcs, ctx.arg_of(&u).clone(), ctx.tag_of(&u).clone()))
            }
        }),
    )])
}

/// `strip_zero_offset_shrink` (rangeify.py:330).
fn strip_zero_offset_shrink(ctx: &mut UOpCtx, x: UopId) -> UopId {
    if ctx.op_of(&x) == Ops::Shrink {
        let marg = ctx.marg_pairs(x.clone()).unwrap_or_default();
        if marg.iter().all(|(start, _)| *start == 0) {
            return ctx.srcs(&x)[0].clone();
        }
    }
    x
}

// ---------- 5. split into kernels (rangeify.py:466-563) ----------

/// `debuf` (rangeify.py:476).
fn debuf(lctx: &mut crate::uop::upat::LocalAddBufferContext, ctx: &mut UOpCtx, buf: UopId) -> UopId {
    // Variables (ALU buffers with a value range) become ALU params
    if ctx.is_variable(buf.clone()) {
        let (_op, dt, srcs, arg, tag) = ctx.snapshot(&buf);
        return ctx.uop_tag(Ops::Param, Some(dt), srcs, arg, tag);
    }
    let buf_dt = ctx.dt_of(&buf);
    let max_shape = ctx.max_shape(buf.clone());
    let size: i64 = max_shape.iter().product();
    let slot = lctx.dg;
    // python: if buf not in ctx.map: ctx.map[buf] = buf — every touched buffer
    // becomes a call arg (first-touch order); slots are per-encounter
    if !lctx.map_keys.contains(&buf) {
        lctx.map_keys.push(buf.clone());
        lctx.map_vals.push(buf.clone());
    }
    let pa = ParamArg {
        slot,
        dtype: buf_dt,
        vmin_vmax: None,
        multiple_of: None,
        name: None,
        addrspace: ctx.addrspace_of(buf.clone()),
        axis: None,
        device: ctx.device_of(&buf.clone()).cloned(),
        volatile: false,
    };
    lctx.dg += 1;
    // python: UOp.const(prod) -> weakint
    let size_src = ctx.uop(Ops::Const, Some(crate::dtype::weak_dtype(crate::dtype::default_int())), vec![], Arg::Const(ConstVal::Int(size as i128)));
    let param = ctx.uop(Ops::Param, Some(buf_dt), vec![size_src], Arg::Param(pa));
    let ret = crate::tensor::movement::reshape(ctx, param, &max_shape.iter().map(|&s| SVal::I(s as i128)).collect::<Vec<_>>());
    let buf_shape = ctx.shape(&buf).ok().flatten().unwrap_or_default();
    if max_shape.iter().map(|&s| SVal::I(s as i128)).collect::<Vec<_>>() != buf_shape.to_vec() {
        let bounds: Vec<Arg> = buf_shape.iter().map(|s| sval_pair(SVal::I(0), s.clone())).collect();
        return shrink_args(ctx, ret, &bounds);
    }
    // NOTE: python also registers ctx.map[buf] = buf if absent — handled by caller
    ret
}

/// `handle_after` (rangeify.py:488).
fn handle_after(lctx: &mut crate::uop::upat::LocalAddBufferContext, ctx: &mut UOpCtx, after: UopId) -> Option<UopId> {
    // python `handle_after`: `if after.addrspace == AddrSpace.LOCAL` — the derived
    // property, so an AFTER wrapping a LOCAL buffer is skipped (addrspace_of alone
    // would return None and wrongly add it).
    if ctx.addrspace_derived(after.clone()) == Some(AddrSpace::Local) {
        return None;
    }
    let buf = ctx.buf_uop(after.clone());
    // bottom-up: only add it once
    if lctx.map_get(buf.clone()).is_none() {
        lctx.map_insert(buf.clone(), after);
    }
    Some(buf)
}

/// `renumber_range` (rangeify.py:495).
fn renumber_range(lctx: &mut crate::uop::upat::LocalAddBufferContext, ctx: &mut UOpCtx, r: UopId) -> Option<UopId> {
    // only renumber tags set to the empty tuple by pm_add_param_range_tags
    let tag = ctx.tag_of(&r).clone();
    let is_empty_tuple = matches!(&tag, Arg::Seq { tuple: true, items } if items.is_empty());
    if !is_empty_tuple {
        return None;
    }
    let (_op, dt, srcs, arg, _tag) = ctx.snapshot(&r);
    let Arg::Seq { tuple: true, items } = &arg else { panic!("TGERR: bad RANGE arg") };
    let new_arg = Arg::Seq { tuple: true, items: vec![Arg::Int(lctx.range), items[1].clone()] };
    lctx.range += 1;
    Some(ctx.uop_tag(Ops::Range, Some(dt), srcs, new_arg, Arg::None))
}

/// `find_bufs` (rangeify.py:501) — cycle detection over indexed buffers.
fn find_bufs(ctx: &mut UOpCtx, x: UopId) {
    let ts = ctx.toposort_gated(x, Some(&|s| ctx.op_of(&s) != Ops::After), true);
    let idxs: Vec<UopId> = ts.into_iter().filter(|s| ctx.op_of(s) == Ops::Index).collect();
    let mut read_from: HashMap<UopId, Ops> = HashMap::new();
    for idx in &idxs {
        let buf = ctx.buf_uop(idx.clone());
        if matches!(ctx.op_of(&buf), Ops::Buffer | Ops::Param) {
            let op = ctx.op_of(&ctx.srcs(idx)[0]);
            match read_from.get(&buf) {
                None => {
                    read_from.insert(buf, op);
                }
                Some(prev) if *prev != op => panic!("TGERR: cycle detected while indexing {buf:?}"),
                _ => {}
            }
        }
    }
}

/// `to_define_global` (rangeify.py:507).
pub fn to_define_global() -> PatternMatcher {
    let rules: Vec<(UPat, RuleFn)> = vec![
        (UPat::ops1(Ops::Store).named("x"), r_la(|_lctx, ctx, st| {
            find_bufs(ctx, sym::sget(st, "x"));
            None
        })),
        (
            UPat::op_var(vec![Ops::Buffer, Ops::Mstack, Ops::Mselect]).named("buf"),
            r_la(|lctx, ctx, st| {
                let buf = sym::sget(st, "buf");
                let ret = debuf(lctx, ctx, buf.clone());
                if ret != buf {
                    Some(ret)
                } else {
                    None
                }
            }),
        ),
        // named variables become ALU params with slot -1
        (
            UPat::ops1(Ops::Param).named("v"),
            r_la(|_lctx, ctx, st| {
                let v = sym::sget(st, "v");
                if let Arg::Param(p) = ctx.arg_of(&v)
                    && p.name.is_some() && p.vmin_vmax.is_some() && p.slot != -1 {
                        let mut np = p.clone();
                        np.slot = -1;
                        return Some(ctx.uop_tag(Ops::Param, Some(ctx.dt_of(&v)), ctx.srcs(&v).to_vec(), Arg::Param(np), ctx.tag_of(&v).clone()));
                    }
                None
            }),
        ),
        // renumber the params
        (
            UPat::ops1(Ops::Param).named("buf"),
            r_la(|lctx, ctx, st| {
                let buf = sym::sget(st, "buf");
                let tag = ctx.tag_of(&buf).clone();
                let is_empty_tuple = matches!(&tag, Arg::Seq { tuple: true, items } if items.is_empty());
                if !is_empty_tuple {
                    return None;
                }
                if let Arg::Param(p) = ctx.arg_of(&buf) {
                    if p.name.is_some() {
                        return None;
                    }
                } else {
                    return None;
                }
                match ctx.shape(&buf) {
                    Ok(Some(_)) => {}
                    _ => return None,
                }
                let ret = debuf(lctx, ctx, buf.clone());
                if ret != buf {
                    Some(ret)
                } else {
                    None
                }
            }),
        ),
        // ALU params are scalar symbolic values, not buffers
        (
            UPat::ops1(Ops::Index).with_src_tup(vec![UPat::ops1(Ops::Param).named("v")]).allow_all_srcs(),
            r_la(|_lctx, ctx, st| {
                let v = sym::sget(st, "v");
                if ctx.addrspace_of(v.clone()) == Some(AddrSpace::Alu) {
                    Some(v) // no-op: stops matching like python returning the node
                } else {
                    None
                }
            }),
        ),
        // bound Variables: strip the store
        (
            UPat::ops1(Ops::After).allow_all_srcs().named("b"),
            r_la(|_lctx, ctx, st| {
                let b = sym::sget(st, "b");
                if ctx.is_bound_var(b.clone()) {
                    return Some(ctx.srcs(&b)[0].clone());
                }
                None
            }),
        ),
        (UPat::ops1(Ops::After).allow_all_srcs().named("after"), r_la(|lctx, ctx, st| handle_after(lctx, ctx, sym::sget(st, "after")))),
        // remove device from local BUFFERIZE
        (
            UPat::ops1(Ops::Stage).allow_all_srcs().named("b"),
            r_la(|_lctx, ctx, st| {
                let b = sym::sget(st, "b");
                let Arg::Data(d) = ctx.arg_of(&b) else { return None };
                if matches!(d.get("device"), Arg::None) {
                    return None;
                }
                let mut nd = d.clone();
                let i = nd.fields.iter().position(|(n, _)| n == "device").expect("TGERR: device field");
                nd.fields[i].1 = Arg::None;
                Some(ctx.uop_tag(Ops::Stage, Some(ctx.dt_of(&b)), ctx.srcs(&b).to_vec(), Arg::Data(nd), ctx.tag_of(&b).clone()))
            }),
        ),
        // renumber the ranges starting with 0
        (UPat::ops1(Ops::Range).named("r"), r_la(|lctx, ctx, st| renumber_range(lctx, ctx, sym::sget(st, "r")))),
    ];
    PatternMatcher::new(rules)
}

/// `rangeify_codegen` (rangeify.py:535).
pub fn rangeify_codegen() -> PatternMatcher {
    PatternMatcher::new(vec![
        (
            UPat::ops1(Ops::Contiguous).named("x"),
            r_la(|lctx, ctx, st| {
                let x = sym::sget(st, "x");
                // CONTIGUOUS arg carries Opt tuples; corpus leaves it None
                if let Arg::Seq { tuple: true, items } = ctx.arg_of(&x) {
                    lctx.opts = Some(items.clone());
                }
                Some(ctx.srcs(&x)[0].clone())
            }),
        ),
        (
            UPat::ops1(Ops::Noop).named("x"),
            r_la(|_lctx, ctx, st| {
                let x = sym::sget(st, "x");
                if !ctx.srcs(&x).is_empty() {
                    Some(ctx.srcs(&x)[0].clone())
                } else {
                    None
                }
            }),
        ),
    ])
}

/// `split_store` (rangeify.py:547).
fn split_store(ctx: &mut UOpCtx, x: UopId) -> Option<UopId> {
    // if we have any open ranges here, we don't split. open DEVICE ranges are fine
    for r in ctx.ranges(&x.clone()).iter() {
        if AxisType::range_axis_type(ctx, r.clone()) != AxisType::Device {
            return None;
        }
    }
    // the store of a bound Variable is an input value, not a kernel
    let st = if ctx.op_of(&x) == Ops::End { ctx.srcs(&x)[0].clone() } else { x.clone() };
    if ctx.op_of(&st) == Ops::Store && ctx.is_variable(ctx.srcs(&st)[0].clone()) {
        return None;
    }

    let lctx = std::rc::Rc::new(std::cell::RefCell::new(crate::uop::upat::LocalAddBufferContext::default()));
    let pm = to_define_global()
        .add(&crate::codegen::simplify::pm_flatten_range())
        .add(&rangeify_codegen());
    let ret = graph_rewrite(ctx, x, &pm, true, None, false, false, CtxVal::La(lctx.clone()))
        .unwrap_or_else(|e| panic!("TGERR: kernel split: {e}"));

    // create the Kernel
    let ki = kernel_info("test", None);
    let sink = ctx.sink_arg(vec![ret], Arg::Data(ki));
    let lc = lctx.borrow();
    let args: Vec<UopId> = lc.map_vals.clone();
    Some(ctx.call(sink, args, None))
}

/// `split_kernels` (rangeify.py:561).
pub fn split_kernels() -> PatternMatcher {
    static PM: crate::uop::upat::PmSlot = crate::uop::upat::PmSlot::new();
    PM.get_or_init(build_split_kernels).clone()
}

fn build_split_kernels() -> PatternMatcher {
    PatternMatcher::new(vec![(
        UPat::op_var(vec![Ops::Store, Ops::End]).named("x"),
        r_ctx(|ctx, st| split_store(ctx, sym::sget(st, "x"))),
    )])
}

// ---------- get_kernel_graph (rangeify.py:584) ----------

pub fn get_kernel_graph(ctx: &mut UOpCtx, sink: UopId) -> UopId {
    // (single-device: multi_pm skipped; OPENPILOT_HACKS off)

    // 0. earliest rewrites (python: pm_mops+earliest_rewrites, bottom_up)
    let pm_earliest = earliest_rewrites().add(&pm_mops());
    let tsink = graph_rewrite(ctx, sink, &pm_earliest, true, None, false, false, CtxVal::None)
        .unwrap_or_else(|e| panic!("TGERR: earliest rewrites: {e}"));

    // convert copies to stores
    let tsink = graph_rewrite(ctx, tsink, &pm_copy_to_store(), true, None, false, false, CtxVal::Cnt(std::rc::Rc::new(std::cell::Cell::new(0))))
        .unwrap_or_else(|e| panic!("TGERR: convert copy to store: {e}"));

    // convert movement ops to ranges
    let tsink = super::indexing::run_rangeify(ctx, tsink, std::env::var("TG_DEBUG_RANGEIFY").is_ok());

    // cleanups for speed and runability
    let pm = symbolic()
        .add(&crate::codegen::simplify::pm_reduce_simplify())
        .add(&pm_const_buffer_folding())
        .add(&pm_remove_bufferize());
    let tsink = graph_rewrite(ctx, tsink, &pm, false, None, false, false, CtxVal::None)
        .unwrap_or_else(|e| panic!("TGERR: symbolic+reduce_collapse+debuf: {e}"));

    // limit buffers (NV: MAX_KERNEL_BUFFERS=0 and DEVICE_MAX_BUFS has no NV entry — no-op)

    // bufferize -> store
    let mut max_slot = -1i64;
    for n in ctx.toposort(&tsink).iter() {
        if ctx.op_of(n) == Ops::Buffer
            && let Arg::Param(p) = ctx.arg_of(n)
                && p.addrspace == Some(AddrSpace::Global) && p.slot > max_slot {
                    max_slot = p.slot;
                }
    }
    let paramarg_start = max_slot + 1;
    let pm = pm_add_buffers().add(&pm_add_param_range_tags());
    let tsink = graph_rewrite(ctx, tsink, &pm, true, None, false, false, CtxVal::Cnt(std::rc::Rc::new(std::cell::Cell::new(paramarg_start))))
        .unwrap_or_else(|e| panic!("TGERR: stage to store: {e}"));

    // split into kernels
    let tsink = graph_rewrite(ctx, tsink, &split_kernels(), true, None, false, false, CtxVal::None)
        .unwrap_or_else(|e| panic!("TGERR: split kernels: {e}"));

    // remove indexing from call args
    graph_rewrite(ctx, tsink, &pm_no_indexing_calls(), false, None, false, false, CtxVal::None)
        .unwrap_or_else(|e| panic!("TGERR: remove indexing from call args: {e}"))
}
