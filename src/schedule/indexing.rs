//! `schedule/indexing.py` port — rangeify: converts a tensor graph SINK into
//! ranged form (INDEX/STAGE/RANGE machinery) via `run_rangeify`.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::dtype::AddrSpace;
use crate::dtype::ConstVal;
use crate::uop::arg::{bufferize_opts, Arg, DataArg, DeviceSpec};
use crate::uop::ctx::UOpCtx;
use crate::uop::ops::{group, Ops};
use crate::uop::range::AxisType;
use crate::uop::sint::SVal;
use crate::uop::sym::symbolic;
use crate::uop::upat::{graph_rewrite, r_dev, r_idx, CtxVal, PatternMatcher, RuleFn, UPat};
use crate::uop::UopId;

/// `ALWAYS_CONTIGUOUS` (indexing.py:24).
pub fn is_always_contiguous(op: Ops) -> bool {
    matches!(
        op,
        Ops::Contiguous | Ops::After | Ops::Buffer | Ops::Const | Ops::Mselect | Ops::Mstack | Ops::Param | Ops::Load | Ops::Call | Ops::Function
    )
}

/// `IndexingContext` (indexing.py:10).
pub struct IndexingContext {
    pub realize_map: HashMap<UopId, Option<Vec<i64>>>,
    pub non_removable: HashSet<UopId>,
    /// input ranges / output ranges
    pub range_map: HashMap<UopId, (Vec<UopId>, Vec<UopId>)>,
    pub range_idx: i64,
}

impl IndexingContext {
    pub fn new() -> Self {
        IndexingContext { realize_map: HashMap::new(), non_removable: HashSet::new(), range_map: HashMap::new(), range_idx: 0 }
    }

    /// `new_range(s, axistype)` — 1-sized ranges become CONST 0.
    pub fn new_range(&mut self, ctx: &mut UOpCtx, s: SVal, axistype: AxisType) -> UopId {
        if let SVal::U(u) = s {
            if ctx.op_of(u) == Ops::Range {
                return u;
            }
        }
        if ctx.resolve_ord(s, crate::uop::ctx::CmpOrd::Ne, SVal::I(1), true) {
            let r = ctx.range(s, self.range_idx, axistype);
            self.range_idx += 1;
            r
        } else {
            ctx.const_py(ConstVal::Int(0), Some(crate::dtype::WEAKINT))
        }
    }
}

/// `data_srcs` (indexing.py:69).
pub fn data_srcs(ctx: &UOpCtx, op: Ops, srcs: &[UopId]) -> Vec<UopId> {
    if matches!(op, Ops::Param | Ops::Buffer | Ops::Range | Ops::Special) {
        return vec![];
    }
    // the store of a bound Variable only carries the input value
    if op == Ops::Store && !srcs.is_empty() && ctx.is_variable(srcs[0]) {
        return vec![];
    }
    if group::in_(group::MOVEMENT, op) || matches!(op, Ops::Index | Ops::Stage | Ops::Reduce | Ops::After | Ops::End) {
        return srcs[..1].to_vec();
    }
    srcs.to_vec()
}

/// `broadcast_rngs` (indexing.py:63).
fn broadcast_rngs(ctx: &mut UOpCtx, x: UopId, src: UopId, rngs: &[UopId]) -> Vec<UopId> {
    if !group::in_(group::BROADCASTABLE, ctx.op_of(x)) {
        return rngs.to_vec();
    }
    let src_shape = ctx.shape(src).ok().flatten().unwrap_or_default();
    let x_shape = ctx.shape(x).ok().flatten().unwrap_or_default();
    let baxes = crate::tensor::movement::broadcast_axes(ctx, &src_shape, &x_shape);
    let nleft = x_shape.len() as i64 - src_shape.len() as i64;
    rngs
        .iter()
        .enumerate()
        .filter(|(j, _)| *j as i64 >= nleft)
        .map(|(j, &r)| {
            if baxes.contains(&j) {
                ctx.const_like(r, ConstVal::Int(0))
            } else {
                r
            }
        })
        .collect()
}

/// `device` property (ops.py:847) — recursive, owned result.
pub fn device_of(ctx: &mut UOpCtx, id: UopId) -> Option<DeviceSpec> {
    fn rec(ctx: &mut UOpCtx, id: UopId, seen: &mut HashSet<UopId>) -> Option<DeviceSpec> {
        if !seen.insert(id) {
            return None;
        }
        let (op, _dt, srcs, arg, _tag) = ctx.snapshot(id);
        match op {
            Ops::Param | Ops::Buffer | Ops::Stage => match &arg {
                Arg::Param(p) => p.device.clone(),
                Arg::Data(d) => d.get("device").clone().into(),
                _ => None,
            },
            Ops::After => srcs.first().and_then(|&s| rec(ctx, s, seen)),
            Ops::Copy => arg_dev(&arg),
            Ops::Mstack => {
                let mut devs = Vec::new();
                for s in srcs {
                    if let Some(d) = rec(ctx, s, seen) {
                        match d {
                            DeviceSpec::Single(s) => devs.push(s),
                            DeviceSpec::Multi(v) => devs.extend(v),
                        }
                    }
                }
                Some(DeviceSpec::Multi(devs))
            }
            _ => {
                for s in srcs {
                    if let Some(d) = rec(ctx, s, seen) {
                        return Some(d);
                    }
                }
                None
            }
        }
    }
    let mut seen = HashSet::new();
    rec(ctx, id, &mut seen)
}

fn arg_dev(arg: &Arg) -> Option<DeviceSpec> {
    match arg {
        Arg::Str(s) => Some(DeviceSpec::Single(s.clone())),
        Arg::None => None,
        other => panic!("TGERR: bad COPY device arg {other:?}"),
    }
}

impl From<Arg> for Option<DeviceSpec> {
    fn from(a: Arg) -> Option<DeviceSpec> {
        match a {
            Arg::Str(s) => Some(DeviceSpec::Single(s)),
            Arg::Seq { tuple: true, items } => {
                let mut v = Vec::new();
                for i in items {
                    match i {
                        Arg::Str(s) => v.push(s),
                        _ => panic!("TGERR: bad device tuple"),
                    }
                }
                Some(DeviceSpec::Multi(v))
            }
            Arg::None => None,
            other => panic!("TGERR: bad device arg {other:?}"),
        }
    }
}

// ---------- pm_generate_realize_map ----------

fn realize_custom_kernel_srcs(ictx: &mut IndexingContext, ctx: &mut UOpCtx, c: UopId) {
    let srcs = ctx.srcs(c).to_vec();
    for s in srcs.into_iter().skip(1) {
        let mut s = s;
        while ctx.op_of(s) == Ops::Reshape {
            s = ctx.srcs(s)[0];
        }
        if !is_always_contiguous(ctx.op_of(s)) {
            ictx.realize_map.insert(s, None);
            ictx.non_removable.insert(s);
        }
    }
}

fn realize_store_after_src(ictx: &mut IndexingContext, ctx: &mut UOpCtx, dest: UopId, src: UopId) {
    // WAR hazard: dest reachable from src
    let dest_base = ctx.base(dest);
    let ts = ctx.toposort_gated(src, None, false);
    if ts.contains(&dest_base) {
        ictx.realize_map.insert(src, None);
    }
}

/// `pm_generate_realize_map` — populates realize_map over the whole graph.
pub fn pm_generate_realize_map() -> PatternMatcher {
    let rules: Vec<(UPat, RuleFn)> = vec![
        // realize the inputs of custom kernel calls
        (
            UPat::ops1(Ops::Call).with_src_any(vec![UPat::op_var(vec![Ops::Sink, Ops::Program])]).with_src_any_len(true).named("c"),
            r_idx(|ictx, ctx, st| {
                let c = crate::uop::sym::sget(st, "c");
                realize_custom_kernel_srcs(ictx, ctx, c);
                None
            }),
        ),
        // always realize
        (
            UPat::op_var(vec![Ops::Contiguous, Ops::Store]).named("tr"),
            r_idx(|ictx, _ctx, st| {
                let tr = crate::uop::sym::sget(st, "tr");
                ictx.realize_map.insert(tr, None);
                None
            }),
        ),
        // realize srcs of MSELECT/MSTACK
        (
            UPat::op_var(vec![Ops::Mselect, Ops::Mstack]).named("root").with_src_any_len(true),
            r_idx(|ictx, ctx, st| {
                let rb = crate::uop::sym::sget(st, "root");
                let srcs = ctx.srcs(rb).to_vec();
                for s in srcs {
                    if !is_always_contiguous(ctx.op_of(ctx.base(s))) {
                        ictx.realize_map.insert(s, None);
                    }
                }
                None
            }),
        ),
        // sometimes we need to realize the src of STORE if there's a self-access
        (
            UPat::ops1(Ops::Store).with_src_tup(vec![UPat::var(Some("dest"), None), UPat::var(Some("src"), None)]),
            r_idx(|ictx, ctx, st| {
                let (dest, src) = (crate::uop::sym::sget(st, "dest"), crate::uop::sym::sget(st, "src"));
                realize_store_after_src(ictx, ctx, dest, src);
                None
            }),
        ),
    ];
    PatternMatcher::new(rules)
}

// ---------- pm_apply_rangeify rules ----------

/// `create_bufferize_and_index_srcs` (indexing.py:76).
fn create_bufferize_and_index_srcs(ictx: &mut IndexingContext, ctx: &mut UOpCtx, x: UopId) -> Vec<UopId> {
    let srcs = ctx.srcs(x).to_vec();
    let data_src_count = data_srcs(ctx, ctx.op_of(x), &srcs).len();
    let x_in_range_map = ictx.range_map.contains_key(&x);
    let x_rngs: Vec<UopId> = if x_in_range_map { ictx.range_map[&x].0.clone() } else { vec![] };
    let mut new_srcs = Vec::with_capacity(srcs.len());
    for (i, &s) in srcs.iter().enumerate() {
        let mut new_src = s;
        let src_rngs: Vec<UopId> = if x_in_range_map { broadcast_rngs(ctx, x, s, &x_rngs) } else { vec![] };
        if matches!(ctx.op_of(s), Ops::Param | Ops::Buffer | Ops::Mstack | Ops::Mselect | Ops::After) {
            if x_in_range_map && i < data_src_count {
                new_src = ctx.index(new_src, src_rngs.clone());
            }
        } else if ictx.realize_map.contains_key(&s) {
            let realized_ranges = ictx
                .realize_map
                .get(&s)
                .cloned()
                .flatten()
                .expect("TGERR: realize map must contain range list");
            let s_out = ictx.range_map[&s].1.clone();
            let closed_ranges: Vec<UopId> = s_out
                .iter()
                .enumerate()
                .filter(|(i, _)| realized_ranges.contains(&(*i as i64)))
                .map(|(_, &r)| r)
                .collect();
            if ctx.op_of(s) == Ops::Store {
                // add the ends if this is a store
                let rng_ends: Vec<UopId> = closed_ranges.iter().copied().filter(|&r| ctx.op_of(r) == Ops::Range).collect();
                new_src = ctx.end(s, rng_ends);
                ictx.realize_map.remove(&s);
            } else {
                let removable = !is_always_contiguous(ctx.op_of(s)) && !ictx.non_removable.contains(&s);
                let s_dev = device_of(ctx, s);
                let opts = if s_out.len() == realized_ranges.len() {
                    bufferize_opts(dev_to_arg(s_dev), Some(AddrSpace::Global), removable)
                } else {
                    bufferize_opts(dev_to_arg(s_dev), Some(AddrSpace::Local), removable)
                };
                let mut stage_srcs = vec![new_src];
                stage_srcs.extend(closed_ranges.iter().copied());
                new_src = ctx.uop(
                    Ops::Stage,
                    Some(ctx.dt_of(s)),
                    stage_srcs,
                    Arg::Data(DataArg { ..opts_data(&opts) }),
                );
                if x_in_range_map {
                    let idx_rngs: Vec<UopId> = src_rngs
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| realized_ranges.contains(&(*i as i64)))
                        .map(|(_, &r)| r)
                        .collect();
                    new_src = ctx.index(new_src, idx_rngs);
                }
            }
        }
        new_srcs.push(new_src);
    }
    new_srcs
}

fn opts_data(d: &DataArg) -> DataArg {
    d.clone()
}

fn dev_to_arg(d: Option<DeviceSpec>) -> Arg {
    match d {
        Some(DeviceSpec::Single(s)) => Arg::Str(s),
        Some(DeviceSpec::Multi(v)) => Arg::Seq { tuple: false, items: v.into_iter().map(Arg::Str).collect() },
        None => Arg::None,
    }
}

/// `create_bufferize_and_index_based_on_ranges` (indexing.py:103).
fn create_bufferize_and_index_based_on_ranges(ictx: &mut IndexingContext, ctx: &mut UOpCtx, x: UopId) -> Option<UopId> {
    if matches!(ctx.op_of(x), Ops::Stage | Ops::Index) {
        return None;
    }
    let new_srcs = create_bufferize_and_index_srcs(ictx, ctx, x);
    let (op, dt, _old, arg, tag) = ctx.snapshot(x);
    Some(ctx.uop_tag(op, Some(dt), new_srcs, arg, tag))
}

/// `convert_pad_to_where_to_keep_behavior_local` (indexing.py:107).
fn convert_pad_to_where(ictx: &mut IndexingContext, ctx: &mut UOpCtx, x: UopId) -> Option<UopId> {
    if !ictx.range_map.contains_key(&x) {
        return None;
    }
    let bx = create_bufferize_and_index_based_on_ranges(ictx, ctx, x)?;
    let (in_rngs, _out) = ictx.range_map[&x].clone();
    let mut valids: Vec<UopId> = in_rngs.iter().map(|&r| ctx.get_valid(r)).collect();
    let t = ctx.const_py(ConstVal::Bool(true), Some(crate::dtype::BOOL));
    let valid = if valids.is_empty() {
        t
    } else {
        let first = valids.remove(0);
        if valids.is_empty() {
            first
        } else {
            ctx.uprod(first, &valids)
        }
    };
    let bx_srcs = ctx.srcs(bx).to_vec();
    // python: UOp.const(x.dtype.const(0)) — a SCALAR const (no expand;
    // const_like would expand it to x's shape and break the sum-of-gate fold)
    let zero = ctx.const_py(ConstVal::Int(0), Some(ctx.dt_of(bx_srcs[0])));
    let w = ctx.where_(valid, bx_srcs[0], zero);
    Some(w)
}

/// `convert_reduce_to_reduce_with_ranges` (indexing.py:113).
fn convert_reduce_to_ranges(ictx: &mut IndexingContext, ctx: &mut UOpCtx, x: UopId) -> Option<UopId> {
    // arg[1] == 0 -> no axis reduce
    let (op, dt, _srcs, arg, _tag) = ctx.snapshot(x);
    let n_axes = match &arg {
        Arg::Seq { tuple: true, items } => match &items[1] {
            Arg::Int(n) => *n,
            _ => panic!("TGERR: bad REDUCE arg"),
        },
        _ => panic!("TGERR: bad REDUCE arg"),
    };
    if n_axes == 0 {
        return None;
    }
    if !ictx.range_map.contains_key(&x) {
        panic!("TGERR: REDUCE has no ranges in rangeify, UOp verification failed");
    }
    let bx = create_bufferize_and_index_based_on_ranges(ictx, ctx, x)?;
    let (in_rngs, _) = ictx.range_map[&x].clone();
    let new_ranges: Vec<UopId> = in_rngs[..n_axes as usize].to_vec();
    let bx_srcs = ctx.srcs(bx).to_vec();
    let mut srcs = vec![bx_srcs[0]];
    srcs.extend(new_ranges);
    let new_arg = Arg::Seq {
        tuple: true,
        items: vec![arg_item(&arg, 0), Arg::Int(0)],
    };
    Some(ctx.uop(Ops::Reduce, Some(dt), srcs, new_arg))
}

fn arg_item(arg: &Arg, i: usize) -> Arg {
    match arg {
        Arg::Seq { tuple: true, items } => items[i].clone(),
        other => panic!("TGERR: bad seq arg {other:?}"),
    }
}

/// `convert_stack_to_where` (indexing.py:121).
fn convert_stack_to_where(ictx: &mut IndexingContext, ctx: &mut UOpCtx, x: UopId) -> Option<UopId> {
    if !ictx.range_map.contains_key(&x) || ctx.dt_of(x) == crate::dtype::VOID {
        return None;
    }
    let srcs = create_bufferize_and_index_srcs(ictx, ctx, x);
    let r0 = ictx.range_map[&x].1[0];
    let mut ret = srcs[srcs.len() - 1];
    for k in (0..srcs.len() - 1).rev() {
        let kk = ctx.const_like(r0, ConstVal::Int(k as i128));
        let cond = ctx.eq(r0, kk);
        ret = ctx.where_(cond, srcs[k], ret);
    }
    Some(ret)
}

/// `remove_movement_op_after_rangeify` (indexing.py:131).
fn remove_movement_op_after_rangeify(ictx: &mut IndexingContext, ctx: &mut UOpCtx, x: UopId) -> Option<UopId> {
    if ictx.range_map.contains_key(&x) || ctx.op_of(ctx.srcs(x)[0]) == Ops::Index {
        return Some(ctx.srcs(x)[0]);
    }
    None
}

/// `pm_apply_rangeify` (indexing.py:134).
pub fn pm_apply_rangeify() -> PatternMatcher {
    let rules: Vec<(UPat, RuleFn)> = vec![
        // REDUCE(op, axis) -> REDUCE(op) with ranges
        (UPat::ops1(Ops::Reduce).named("x"), r_idx(|ictx, ctx, st| convert_reduce_to_ranges(ictx, ctx, crate::uop::sym::sget(st, "x")))),
        // PAD -> WHERE
        (UPat::ops1(Ops::Pad).named("x"), r_idx(|ictx, ctx, st| convert_pad_to_where(ictx, ctx, crate::uop::sym::sget(st, "x")))),
        // STACK -> WHERE select on the leading range
        (UPat::ops1(Ops::Stack).with_src_any_len(true).named("x"), r_idx(|ictx, ctx, st| convert_stack_to_where(ictx, ctx, crate::uop::sym::sget(st, "x")))),
        // finally, apply_rangeify
        (
            UPat::op_var(crate::uop::ops::OPS_ALL.to_vec()).named("x"),
            r_idx(|ictx, ctx, st| create_bufferize_and_index_based_on_ranges(ictx, ctx, crate::uop::sym::sget(st, "x"))),
        ),
        // remove movement op
        (
            UPat::op_var(group::MOVEMENT.to_vec()).named("x"),
            r_idx(|ictx, ctx, st| remove_movement_op_after_rangeify(ictx, ctx, crate::uop::sym::sget(st, "x"))),
        ),
    ];
    PatternMatcher::new(rules)
}

/// `pm_fix_deviceless` (indexing.py:147).
pub fn pm_fix_deviceless() -> PatternMatcher {
    PatternMatcher::new(vec![(
        UPat::ops1(Ops::Stage).named("b"),
        r_dev(|dev, ctx, st| {
            let b = crate::uop::sym::sget(st, "b");
            let (_op, _dt, _srcs, arg, _tag) = ctx.snapshot(b);
            let Arg::Data(d) = &arg else { return None };
            let addrspace = match d.get("addrspace") {
                Arg::Enum(e) => AddrSpace::from_u8(e.value as u8).expect("TGERR: bad addrspace"),
                _ => return None,
            };
            let is_none = matches!(d.get("device"), Arg::None);
            if addrspace == AddrSpace::Global && is_none {
                let mut nd = d.clone();
                let i = nd.fields.iter().position(|(n, _)| n == "device").expect("TGERR: device field");
                nd.fields[i].1 = dev_to_arg(dev.clone());
                Some(ctx.uop_tag(Ops::Stage, Some(ctx.dt_of(b)), ctx.srcs(b).to_vec(), Arg::Data(nd), ctx.tag_of(b).clone()))
            } else {
                None
            }
        }),
    )])
}

// ---------- movement application (indexing.py:152-185) ----------

/// `_apply_reshape` (indexing.py:152).
fn apply_reshape(ctx: &mut UOpCtx, in_shape: &[SVal], out_shape: &[SVal], urngs: UopId) -> Vec<UopId> {
    let mut acc = SVal::I(1);
    let mut axes_in: Vec<UopId> = Vec::new();
    let urngs_srcs = ctx.srcs(urngs).to_vec();
    for (s, &src) in out_shape.iter().zip(urngs_srcs.iter()).rev() {
        // python: acc*src with the int promoted via from_py -> weakint (the
        // commutative flip only applies to weakint, so the dtype matters here)
        let acc_u = match acc {
            SVal::I(v) => ctx.const_py(crate::dtype::ConstVal::Int(v as i128), Some(crate::dtype::WEAKINT)),
            SVal::U(u) => u,
        };
        let m = ctx.uop(Ops::Mul, Some(crate::dtype::WEAKINT), vec![acc_u, src], Arg::None);
        axes_in.push(m);
        acc = ctx.sval_mul(acc, *s);
    }
    // combined_axes = UOp.const(0).usum(axes_in)
    let zero = ctx.const_py(ConstVal::Int(0), Some(crate::dtype::WEAKINT));
    let combined = if axes_in.is_empty() {
        zero
    } else {
        ctx.usum(zero, &axes_in)
    };
    let mut axes_out: Vec<UopId> = Vec::new();
    let mut comb = combined;
    for s in in_shape.iter().rev() {
        let su = ctx.sint_to_uop_dt(*s, ctx.dt_of(comb));
        axes_out.push(ctx.mod_(comb, su));
        comb = ctx.div_floor(comb, su);
    }
    axes_out.reverse();
    let sink = crate::tensor::sink(ctx, axes_out);
    let pm = symbolic().add(&crate::uop::sym::pm_simplify_valid()).add(&crate::uop::sym::pm_drop_and_clauses());
    let out = graph_rewrite(ctx, sink, &pm, false, None, false, false, CtxVal::None)
        .unwrap_or_else(|e| panic!("TGERR: reshape rewrite: {e}"));
    ctx.srcs(out).to_vec()
}

fn comb_dt(ctx: &UOpCtx, u: UopId) -> crate::dtype::DTypeId {
    ctx.dt_of(u)
}

/// `apply_movement_op` (indexing.py:168) — rngs in, rngs out. Recomputed
/// (python's functools.cache is an identity/interning hack; structural
/// interning in the Rust graph makes recomputation produce identical nodes).
pub fn apply_movement_op(ctx: &mut UOpCtx, op: Ops, in_shape: &[SVal], arg: &[Arg], rngs: &[UopId]) -> Vec<UopId> {
    match op {
        Ops::Shrink => {
            let pairs: Vec<(i64, i64)> = arg
                .iter()
                .map(|a| match a {
                    Arg::Seq { tuple: true, items } => match (&items[0], &items[1]) {
                        (Arg::Int(o), Arg::Int(_)) => (*o, 0),
                        other => panic!("TGERR: bad shrink arg {other:?}"),
                    },
                    other => panic!("TGERR: bad shrink arg {other:?}"),
                })
                .collect();
            rngs
                .iter()
                .zip(pairs.iter())
                .map(|(&a, (off, _))| {
                    if *off == 0 {
                        a
                    } else {
                        let offu = ctx.const_like(a, ConstVal::Int(*off as i128));
                        ctx.add(a, offu)
                    }
                })
                .collect()
        }
        Ops::Permute => {
            let perm: Vec<i64> = arg
                .iter()
                .map(|a| match a {
                    Arg::Int(i) => *i,
                    other => panic!("TGERR: bad permute arg {other:?}"),
                })
                .collect();
            // argsort
            let mut idx: Vec<usize> = (0..perm.len()).collect();
            idx.sort_by_key(|&i| perm[i]);
            idx.iter().map(|&i| rngs[i]).collect()
        }
        Ops::Flip => {
            let flips: Vec<bool> = arg
                .iter()
                .map(|a| match a {
                    Arg::Bool(b) => *b,
                    other => panic!("TGERR: bad flip arg {other:?}"),
                })
                .collect();
            rngs
                .iter()
                .zip(in_shape.iter())
                .zip(flips.iter())
                .map(|((&a, s), f)| {
                    if *f {
                        let sm1 = ctx.sval_sub(*s, SVal::I(1));
                        let sm1u = ctx.sval_to_uop(sm1);
                        ctx.sub(sm1u, a)
                    } else {
                        a
                    }
                })
                .collect()
        }
        // python slices clamp: expanding a 0-dim (scalar) yields no new ranges
        Ops::Expand => rngs[arg.len().min(rngs.len())..].to_vec(),
        Ops::Pad => {
            let pairs: Vec<(i64, i64)> = arg
                .iter()
                .map(|a| match a {
                    Arg::Seq { tuple: true, items } => match (&items[0], &items[1]) {
                        (Arg::Int(o), Arg::Int(sz)) => (*o, *sz),
                        other => panic!("TGERR: bad pad arg {other:?}"),
                    },
                    other => panic!("TGERR: bad pad arg {other:?}"),
                })
                .collect();
            let pm = symbolic().add(&crate::uop::sym::pm_simplify_valid());
            rngs
                .iter()
                .zip(in_shape.iter())
                .zip(pairs.iter())
                .map(|((&r, sh), (off, sz))| {
                    let same = *sz == sh.as_i128_or_panic() as i64 && *off == 0;
                    if same {
                        r
                    } else {
                        // (r >= off) & (r < (sh+off))
                        let offu = ctx.const_like(r, ConstVal::Int(*off as i128));
                        let ge = crate::tensor::elementwise::ge(ctx, r, offu);
                        let hi = ctx.sval_add(*sh, SVal::I(*off as i128));
                        let hiu = ctx.sint_to_uop_dt(hi, ctx.dt_of(r));
                        let lt = ctx.lt(r, hiu);
                        let cond = ctx.and_(ge, lt);
                        let simplified = graph_rewrite(ctx, cond, &pm, false, None, false, false, CtxVal::None)
                            .unwrap_or_else(|e| panic!("TGERR: pad rewrite: {e}"));
                        let sub = ctx.sub(r, offu);
                        ctx.valid(sub, simplified)
                    }
                })
                .collect()
        }
        Ops::Reshape => {
            // sink + simplify (applies any commutative flips to the rngs early)
            let sink = crate::tensor::sink(ctx, rngs.to_vec());
            let sink = crate::uop::upat::simplify(ctx, sink);
            let mut sub_array: HashMap<UopId, UopId> = HashMap::new();
            let mut back: HashMap<UopId, UopId> = HashMap::new();
            for (i, &r) in ctx.ranges(sink).iter().enumerate() {
                let (rop, rdt, rsrcs, _rarg, _rtag) = ctx.snapshot(r);
                let ph = ctx.uop_tag(rop, Some(rdt), rsrcs[..1].to_vec(), Arg::Seq { tuple: true, items: vec![Arg::Int(i as i64), Arg::Enum(AxisType::Placeholder.to_enum_val())] }, Arg::None);
                sub_array.insert(r, ph);
                back.insert(ph, r);
            }
            let substituted = crate::uop::upat::substitute(ctx, sink, sub_array, true, true);
            let out_shape: Vec<Arg> = arg.to_vec();
            let out_shape_sval: Vec<SVal> = out_shape
                .iter()
                .map(|a| match a {
                    Arg::Int(i) => SVal::I(*i as i128),
                    Arg::Seq { tuple: true, items } if items.is_empty() => SVal::I(1),
                    other => panic!("TGERR: bad reshape shape {other:?}"),
                })
                .collect();
            let axes = apply_reshape(ctx, in_shape, &out_shape_sval, substituted);
            let reshaped = crate::tensor::sink(ctx, axes);
            let restored = crate::uop::upat::substitute(ctx, reshaped, back, true, true);
            ctx.srcs(restored).to_vec()
        }
        _ => panic!("TGERR: {op:?} is not a MovementOp"),
    }
}


// ---------- run_rangeify (indexing.py:187) ----------

fn all_same_ids(v: &[UopId]) -> bool {
    v.iter().all(|&x| x == v[0])
}

/// `gate_kernel_sink` (ops.py:1760).
pub fn gate_kernel_sink(ctx: &UOpCtx, x: UopId) -> bool {
    if ctx.op_of(x) == Ops::Linear {
        return false;
    }
    if ctx.op_of(x) == Ops::Sink {
        if let Arg::Data(d) = ctx.arg_of(x) {
            if d.tag == "KernelInfo" {
                return false;
            }
        }
    }
    true
}

/// elementwise/reduce ending-ranges realization check
/// (indexing.py:270-279), mutating out_rngs / realize_map.
fn maybe_realize_axes(
    ictx: &mut IndexingContext,
    ctx: &mut UOpCtx,
    x: UopId,
    x_shape: &[SVal],
    out_rngs: &mut Vec<UopId>,
    ending_ranges: &mut HashMap<UopId, Vec<UopId>>,
) {
    let has_ending = ending_ranges.get(&x).map(|v| !v.is_empty()).unwrap_or(false);
    if !has_ending || !(group::in_(group::ELEMENTWISE, ctx.op_of(x)) || ctx.op_of(x) == Ops::Reduce) {
        return;
    }
    let mut ra = ictx.realize_map.get(&x).cloned().flatten().unwrap_or_default();
    let er = ending_ranges.get(&x).cloned().unwrap_or_default();
    for (i, &r) in out_rngs.iter().enumerate() {
        if ra.contains(&(i as i64)) {
            continue;
        }
        // PCONTIG=0: always check that r's ranges precede all ending ranges
        let rranges = ctx.ranges(r).to_vec();
        if rranges.iter().any(|&rr| er.iter().any(|&e| AxisType::range_axis_id(ctx, rr) > AxisType::range_axis_id(ctx, e))) {
            ra.push(i as i64);
        }
    }
    ending_ranges.insert(x, vec![]);
    if !ra.is_empty() {
        ictx.realize_map.insert(x, Some(ra.clone()));
        *out_rngs = out_rngs
            .iter()
            .enumerate()
            .map(|(i, &r)| if ra.contains(&(i as i64)) { ictx.new_range(ctx, x_shape[i], AxisType::Weak) } else { r })
            .collect();
    }
}

/// shared tail of the rangeify loop: movement ops, STACK, EXPAND, REDUCE rngs,
/// and the range_map assignment (indexing.py:291-316).
fn finish_node(
    ictx: &mut IndexingContext,
    ctx: &mut UOpCtx,
    x: UopId,
    x_shape: &[SVal],
    out_rngs: &[UopId],
    ending_ranges: &mut HashMap<UopId, Vec<UopId>>,
) {
    let mut rngs = out_rngs.to_vec();

    // apply movement ops
    if group::in_(group::MOVEMENT, ctx.op_of(x)) {
        let (op, _dt, srcs, arg, _tag) = ctx.snapshot(x);
        let in_shape = ctx.shape(srcs[0]).expect("TGERR: movement src shape").unwrap_or_default();
        // RESHAPE/EXPAND store their shape in shape-srcs (our M2 convention),
        // python carries it in the movement arg — normalize here. PAD/SHRINK
        // pairs likewise live in shape-srcs ((start, stop) lists), python's
        // marg (ops.py:779) reads them from src[1]/src[2] — without this the
        // pad/shrink arms zip over nothing and the ranges pass through
        // unwrapped (the arange cumsum mask bug).
        let arg_refs: Vec<Arg> = if matches!(op, Ops::Reshape | Ops::Expand) {
            match ctx.marg_shape(x) {
                Ok(shape) => shape.iter().map(|s| match s {
                    SVal::I(v) => Arg::Int(*v as i64),
                    SVal::U(_) => panic!("TGERR: symbolic reshape dim unsupported in rangeify"),
                }).collect(),
                Err(_) => vec![],
            }
        } else if matches!(op, Ops::Pad | Ops::Shrink) {
            match ctx.marg_pairs(x) {
                Ok(pairs) => pairs
                    .iter()
                    .map(|(a, b)| Arg::Seq { tuple: true, items: vec![Arg::Int(*a), Arg::Int(*b)] })
                    .collect(),
                // symbolic pairs (kernel-LOCAL views): fall back to the
                // arg-carried path — python handles sint symbolically here.
                Err(_) => match &arg {
                    Arg::Seq { tuple: true, items } => items.clone(),
                    _ => vec![],
                },
            }
        } else {
            match &arg {
                Arg::Seq { tuple: true, items } => items.clone(),
                Arg::None => vec![],
                other => vec![other.clone()],
            }
        };
        rngs = apply_movement_op(ctx, op, &in_shape, &arg_refs, out_rngs);
    }
    // STACK: the leading range selects the src, srcs get the trailing ranges
    if ctx.op_of(x) == Ops::Stack {
        rngs = out_rngs[1..].to_vec();
    }
    // if the EXPAND is used to inject a range, we don't mark it as ending_ranges. otherwise we do.
    if ctx.op_of(x) == Ops::Expand && x_shape.iter().all(|y| matches!(y, SVal::I(_))) {
        let n = {
            match ctx.marg_pairs(x) {
                Ok(v) => v.len(),
                Err(_) => match ctx.marg_ints(x) {
                    Ok(v) => v.len(),
                    Err(_) => ctx.marg_shape(x).map(|s| s.len()).unwrap_or(0),
                },
            }
        };
        // python slice clamps: a scalar expand has no out ranges
        let sink = crate::tensor::sink(ctx, out_rngs[..n.min(out_rngs.len())].to_vec());
        let rr = ctx.ranges(sink).to_vec();
        ending_ranges.entry(x).or_default().extend(rr);
    }

    // REDUCE creates ranges for the axes it is reducing
    if ctx.op_of(x) == Ops::Reduce {
        let (_op, _dt, srcs, arg, _tag) = ctx.snapshot(x);
        let n_axes = match &arg {
            Arg::Seq { tuple: true, items } => match &items[1] {
                Arg::Int(n) => *n,
                _ => 0,
            },
            _ => 0,
        };
        if n_axes > 0 {
            let in_shape = ctx.shape(srcs[0]).expect("TGERR: reduce src shape").unwrap_or_default();
            let mut new_rngs: Vec<UopId> =
                in_shape[..n_axes as usize].iter().map(|&s| ictx.new_range(ctx, s, AxisType::Reduce)).collect();
            new_rngs.extend(out_rngs.iter().copied());
            rngs = new_rngs;
        }
    }

    // assign to the range map. rngs are the input ranges, out_rngs are the output ranges.
    ictx.range_map.insert(x, (rngs, out_rngs.to_vec()));
}

pub fn run_rangeify(ctx: &mut UOpCtx, tsink: UopId, _debug: bool) -> UopId {
    let shared = Rc::new(RefCell::new(IndexingContext::new()));

    // get ops to realize
    {
        let pm_realize = pm_generate_realize_map();
        graph_rewrite(ctx, tsink, &pm_realize, false, None, false, false, CtxVal::Idx(shared.clone()))
            .unwrap_or_else(|e| panic!("TGERR: get realize: {e}"));
    }

    // get the consumer map
    let tsink_toposort = {
        let ictx = shared.borrow();
        let _ = ictx;
        ctx.toposort_gated(tsink, Some(&|x| gate_kernel_sink(ctx, x)), true)
    };
    let mut consumer_map: HashMap<UopId, Vec<UopId>> = HashMap::new();
    for &n in &tsink_toposort {
        consumer_map.entry(n).or_default();
    }
    for &c in &tsink_toposort {
        let (c_op, _dt, c_srcs, _arg, _tag) = ctx.snapshot(c);
        for x in data_srcs(ctx, c_op, &c_srcs) {
            if let Some(v) = consumer_map.get_mut(&x) {
                if !v.contains(&c) {
                    v.push(c);
                }
            }
        }
    }

    // explicit rangeify
    let mut ending_ranges: HashMap<UopId, Vec<UopId>> = HashMap::new();
    for &x in tsink_toposort.iter().rev() {
        // no ranges on kernels, they are internal
        if matches!(ctx.op_of(x), Ops::Call | Ops::Function | Ops::Linear) {
            continue;
        }
        // AFTER doesn't have range
        if ctx.op_of(x) == Ops::After {
            continue;
        }
        // treat MSTACK/MSELECT like SINK
        if matches!(ctx.op_of(x), Ops::Mstack | Ops::Mselect) {
            continue;
        }

        let mut ictx = shared.borrow_mut();
        let consumers = consumer_map.get(&x).cloned().unwrap_or_default();
        let mut er: Vec<UopId> = Vec::new();
        for u in &consumers {
            if let Some(v) = ending_ranges.get(u) {
                er.extend(v.iter().copied());
            }
        }
        ending_ranges.insert(x, er);

        // ranges the consumers iterate that this node broadcasts over
        let x_shape = ctx.shape(x).expect("TGERR: rangeify node shape").unwrap_or_default();
        let mut ended: Vec<UopId> = Vec::new();
        for c in &consumers {
            if !ictx.range_map.contains_key(c) || !group::in_(group::BROADCASTABLE, ctx.op_of(*c)) {
                continue;
            }
            let c_shape = ctx.shape(*c).expect("TGERR: consumer shape").unwrap_or_default();
            for i in crate::tensor::movement::broadcast_axes(ctx, &x_shape, &c_shape) {
                ended.push(ictx.range_map[c].0[i]);
            }
        }
        let ended_sink = crate::tensor::sink(ctx, ended);
        let broadcast_ending_ranges: Vec<UopId> = ctx.ranges(ended_sink).to_vec();
        if ctx.op_of(x) == Ops::Reduce {
            ending_ranges.get_mut(&x).unwrap().extend(broadcast_ending_ranges.iter().copied());
        }

        // *** the ranges on the output are
        //  1. new if this op is realized
        //  2. from the single consumer if this op only has one consumer
        //  3. potentially new if this op has 2+ consumers

        let mut consumer_rngs: Vec<Vec<UopId>> = Vec::new();
        for c in &consumers {
            if ictx.range_map.contains_key(c) {
                let crngs = ictx.range_map[c].0.clone();
                consumer_rngs.push(broadcast_rngs(ctx, *c, x, &crngs));
            }
        }

        let mut out_rngs: Vec<UopId>;
        if ictx.realize_map.contains_key(&x) {
            // if this is in the realize_map, we create new ranges (at the output)
            out_rngs = x_shape.iter().map(|&s| ictx.new_range(ctx, s, AxisType::Weak)).collect();
            // all ranges are ended now
            ending_ranges.insert(x, vec![]);
            // mark all ranges as ended
            assert!(ictx.realize_map.get(&x).unwrap().is_none(), "TGERR: realize_map[x] must be None");
            ictx.realize_map.insert(x, Some((0..x_shape.len() as i64).collect()));
        } else if consumer_rngs.is_empty() {
            // if no consumers have ranges and this isn't realized, this doesn't have ranges either.
            continue;
        } else if consumer_rngs.len() == 1 {
            // if this has one consumer, it inherits the ranges from it
            out_rngs = consumer_rngs.remove(0);
        } else {
            // if this has two consumers, we have to merge the ranges and might create new ones
            let naxes = consumer_rngs[0].len();
            let mut rngs_valids: Vec<(Vec<UopId>, Vec<UopId>)> = Vec::new();
            for i in 0..naxes {
                let mut local_rngs = Vec::new();
                let mut valids = Vec::new();
                for cr in &consumer_rngs {
                    let r = cr[i];
                    local_rngs.push(ctx.get_idx(r));
                    valids.push(ctx.get_valid(r));
                }
                rngs_valids.push((local_rngs, valids));
            }
            let all_all_same = rngs_valids.iter().all(|(lr, _)| all_same_ids(lr));
            let mut _out_rngs: Vec<UopId> = Vec::new();
            let mut _realize_axis: Vec<i64> = Vec::new();
            for (i, (local_rngs, valids)) in rngs_valids.iter().enumerate() {
                // we compare the ranges without their valids
                if all_all_same {
                    // the new valid is the OR of all the children valids
                    let f = ctx.const_py(ConstVal::Bool(false), Some(crate::dtype::BOOL));
                    let minimum_valid = if valids.is_empty() { f } else { ctx.usum(f, valids) };
                    let base = ctx.valid(local_rngs[0], minimum_valid);
                    let merged = graph_rewrite(ctx, base, symbolic(), false, None, false, false, CtxVal::None)
                        .unwrap_or_else(|e| panic!("TGERR: minimum_valid: {e}"));
                    _out_rngs.push(merged);
                } else {
                    _out_rngs.push(ictx.new_range(ctx, x_shape[i], AxisType::Weak));
                    _realize_axis.push(i as i64);
                }
            }
            out_rngs = _out_rngs;
            // we have to (partially) realize here if there's new ranges
            if !_realize_axis.is_empty() {
                ictx.realize_map.insert(x, Some(_realize_axis));
            }
        }

        // if this element is a reduce and there's ended ranges, we might have to end some other ranges
        maybe_realize_axes(&mut ictx, ctx, x, &x_shape, &mut out_rngs, &mut ending_ranges);
        ending_ranges.get_mut(&x).unwrap().extend(broadcast_ending_ranges.iter().copied());

        finish_node(&mut ictx, ctx, x, &x_shape, &out_rngs, &mut ending_ranges);

        // python: DEBUG_RANGEIFY trace (indexing.py:306)
        if _debug {
            let rdesc = |ctx: &UOpCtx, rs: &[UopId]| -> String {
                rs.iter()
                    .map(|&r| match ctx.arg_of(r) {
                        Arg::Seq { tuple: true, items } if matches!(items.as_slice(), [Arg::Int(_), _]) => {
                            format!(
                                "r{}({:?})",
                                crate::uop::range::AxisType::range_axis_id(ctx, r),
                                crate::uop::range::AxisType::from_arg(&items[1])
                            )
                        }
                        _ => format!("#{}", r.0),
                    })
                    .collect::<Vec<_>>()
                    .join(",")
            };
            let shape_s: Vec<String> = x_shape
                .iter()
                .map(|s| match s {
                    crate::uop::sint::SVal::I(i) => i.to_string(),
                    crate::uop::sint::SVal::U(u) => format!("#{}", u.0),
                    _ => "?".into(),
                })
                .collect();
            eprintln!(
                "{} {:2} {:?} ({}) {} [{}] -> [{}]",
                if ictx.realize_map.contains_key(&x) { "***" } else { "   " },
                consumers.len(),
                ctx.op_of(x),
                shape_s.join(","),
                ending_ranges.get(&x).map(|v| v.len()).unwrap_or(0),
                rdesc(ctx, &ictx.range_map[&x].0),
                rdesc(ctx, &ictx.range_map[&x].1)
            );
        }
    }

    // apply rangeify (bottom up)
    let pm = pm_apply_rangeify();
    let tsink = graph_rewrite(ctx, tsink, &pm, true, None, false, false, CtxVal::Idx(shared.clone()))
        .unwrap_or_else(|e| panic!("TGERR: apply rangeify: {e}"));

    // if a deviceless value must materialize, place it on the sink device
    let dev = device_of(ctx, tsink);
    let pm_dev = pm_fix_deviceless();
    graph_rewrite(ctx, tsink, &pm_dev, false, None, false, false, CtxVal::Dev(dev))
        .unwrap_or_else(|e| panic!("TGERR: add device to deviceless: {e}"))
}
