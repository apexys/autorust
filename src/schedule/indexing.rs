//! `schedule/indexing.py` port — rangeify: converts a tensor graph SINK into
//! ranged form (INDEX/STAGE/RANGE machinery) via `run_rangeify`.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::dtype::AddrSpace;
use crate::dtype::ConstVal;
use crate::uop::UopId;
use crate::uop::arg::{Arg, DataArg, DeviceSpec, bufferize_opts};
use crate::uop::ctx::UOpCtx;
use crate::uop::ops::{Ops, group};
use crate::uop::range::AxisType;
use crate::uop::sint::{SInt, SVal};
use crate::uop::sym::symbolic;
use crate::uop::upat::{CtxVal, PatternMatcher, RuleFn, UPat, graph_rewrite, r_ctx, r_dev, r_idx};

/// `ALWAYS_CONTIGUOUS` (indexing.py:24).
pub fn is_always_contiguous(op: Ops) -> bool {
    matches!(
        op,
        Ops::Contiguous
            | Ops::After
            | Ops::Buffer
            | Ops::Const
            | Ops::Mselect
            | Ops::Mstack
            | Ops::Param
            | Ops::Load
            | Ops::Call
            | Ops::Function
    )
}

/// `IndexingContext` (indexing.py:10).
pub struct IndexingContext {
    /// Ops that must be realized, mapped to the output axis indices to realize (`None` = all).
    pub realize_map: HashMap<UopId, Option<Vec<i64>>>,
    /// Ops that must survive rangeify (side effects and barriers).
    pub non_removable: HashSet<UopId>,
    /// input ranges / output ranges
    pub range_map: HashMap<UopId, (Vec<UopId>, Vec<UopId>)>,
    /// Monotonic counter for fresh RANGE ids.
    pub range_idx: i64,
}

impl Default for IndexingContext {
    fn default() -> Self {
        Self::new()
    }
}

impl IndexingContext {
    /// Create an empty indexing context.
    pub fn new() -> Self {
        IndexingContext {
            realize_map: HashMap::new(),
            non_removable: HashSet::new(),
            range_map: HashMap::new(),
            range_idx: 0,
        }
    }

    /// `new_range(s, axistype)` — 1-sized ranges become CONST 0.
    pub fn new_range(&mut self, ctx: &mut UOpCtx, s: SVal, axistype: AxisType) -> UopId {
        if let SVal::U(ref u) = s
            && ctx.op_of(u) == Ops::Range
        {
            return u.clone();
        }
        if ctx.resolve_ord(s.clone(), crate::uop::ctx::CmpOrd::Ne, SVal::I(1), true) {
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
    if op == Ops::Store && !srcs.is_empty() && ctx.is_variable(srcs[0].clone()) {
        return vec![];
    }
    if group::in_(group::MOVEMENT, op)
        || matches!(
            op,
            Ops::Index | Ops::Stage | Ops::Reduce | Ops::After | Ops::End
        )
    {
        return srcs[..1].to_vec();
    }
    srcs.to_vec()
}

/// `broadcast_rngs` (indexing.py:63).
fn broadcast_rngs(ctx: &mut UOpCtx, x: UopId, src: UopId, rngs: &[UopId]) -> Vec<UopId> {
    if !group::in_(group::BROADCASTABLE, ctx.op_of(&x)) {
        return rngs.to_vec();
    }
    let src_shape = ctx.shape(&src).ok().flatten().unwrap_or_default();
    let x_shape = ctx.shape(&x).ok().flatten().unwrap_or_default();
    let baxes = crate::tensor::movement::broadcast_axes(ctx, &src_shape, &x_shape);
    let nleft = x_shape.len() as i64 - src_shape.len() as i64;
    rngs.iter()
        .enumerate()
        .filter(|(j, _)| *j as i64 >= nleft)
        .map(|(j, r)| {
            if baxes.contains(&j) {
                ctx.const_like(r.clone(), ConstVal::Int(0))
            } else {
                r.clone()
            }
        })
        .collect()
}

/// `device` property (ops.py:847) — recursive, owned result.
pub fn device_of(ctx: &mut UOpCtx, id: UopId) -> Option<DeviceSpec> {
    fn rec(ctx: &mut UOpCtx, id: UopId, seen: &mut HashSet<UopId>) -> Option<DeviceSpec> {
        if !seen.insert(id.clone()) {
            return None;
        }
        let (op, _dt, srcs, arg, _tag) = ctx.snapshot(&id);
        match op {
            Ops::Param | Ops::Buffer | Ops::Stage => match &arg {
                Arg::Param(p) => p.device.clone(),
                Arg::Data(d) => d.get("device").clone().into(),
                _ => None,
            },
            Ops::After => srcs.first().and_then(|s| rec(ctx, s.clone(), seen)),
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
    let srcs = ctx.srcs(&c).to_vec();
    for s in srcs.into_iter().skip(1) {
        let mut s = s;
        while ctx.op_of(&s) == Ops::Reshape {
            s = ctx.srcs(&s)[0].clone();
        }
        if !is_always_contiguous(ctx.op_of(&s)) {
            ictx.realize_map.insert(s.clone(), None);
            ictx.non_removable.insert(s);
        }
    }
}

fn realize_store_after_src(ictx: &mut IndexingContext, ctx: &mut UOpCtx, dest: UopId, src: UopId) {
    // WAR hazard: dest reachable from src
    let dest_base = ctx.base(&dest);
    let ts = ctx.toposort_gated(src.clone(), None, false);
    if ts.contains(&dest_base) {
        ictx.realize_map.insert(src, None);
    }
}

/// `pm_generate_realize_map` — populates realize_map over the whole graph.
pub fn pm_generate_realize_map() -> PatternMatcher {
    static PM: crate::uop::upat::PmSlot = crate::uop::upat::PmSlot::new();
    PM.get_or_init(build_pm_generate_realize_map).clone()
}

fn build_pm_generate_realize_map() -> PatternMatcher {
    let rules: Vec<(UPat, RuleFn)> = vec![
        // realize the inputs of custom kernel calls
        (
            UPat::ops1(Ops::Call)
                .with_src_any(vec![UPat::op_var(vec![Ops::Sink, Ops::Program])])
                .with_src_any_len(true)
                .named("c"),
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
            UPat::op_var(vec![Ops::Mselect, Ops::Mstack])
                .named("root")
                .with_src_any_len(true),
            r_idx(|ictx, ctx, st| {
                let rb = crate::uop::sym::sget(st, "root");
                let srcs = ctx.srcs(&rb).to_vec();
                for s in srcs {
                    if !is_always_contiguous(ctx.op_of(&ctx.base(&s))) {
                        ictx.realize_map.insert(s, None);
                    }
                }
                None
            }),
        ),
        // sometimes we need to realize the src of STORE if there's a self-access
        (
            UPat::ops1(Ops::Store).with_src_tup(vec![
                UPat::var(Some("dest"), None),
                UPat::var(Some("src"), None),
            ]),
            r_idx(|ictx, ctx, st| {
                let (dest, src) = (
                    crate::uop::sym::sget(st, "dest"),
                    crate::uop::sym::sget(st, "src"),
                );
                realize_store_after_src(ictx, ctx, dest, src);
                None
            }),
        ),
        // A sum of two contractions (`ADD(REDUCE1, REDUCE2)`) is one kernel
        // with multiple reductions, which `matmul_ranges` declines and the
        // generic heuristic leaves scalar. The ViT attention input gradient
        // (`dQ@Wq + dK@Wk + dV@Wv`) is exactly this: 5.7 ms/layer as one fused
        // kernel. Materialize each contraction first; every one becomes a
        // single-reduce store that the matmul tiler / host GEMM can claim, and
        // the sum is a cheap elementwise kernel.
        (
            UPat::ops1(Ops::Add).with_src_any_len(true).named("a"),
            r_idx(|ictx, ctx, st| {
                let a = crate::uop::sym::sget(st, "a");
                let srcs = ctx.srcs(&a).to_vec();
                if srcs.len() != 2 {
                    return None;
                }
                let (r0, r1) = (
                    contraction_root(ctx, srcs[0].clone()),
                    contraction_root(ctx, srcs[1].clone()),
                );
                if let (Some(r0), Some(r1)) = (r0, r1)
                    && r0 != r1
                {
                    ictx.realize_map.insert(r0, None);
                    ictx.realize_map.insert(r1, None);
                }
                None
            }),
        ),
        // A contraction with a pointwise prologue on one operand
        // (`where(mask, dY, 0) @ W`, `(dY ∘ gelu'(x)) @ W`) cannot use the
        // library GEMM: the operand is not an affine buffer and the generated
        // kernel recomputes the prologue per output column. Record it here and
        // `convert_reduce_to_ranges` stages it once.
        (
            UPat::op_var(vec![Ops::Reduce]).with_src_any_len(true).named("r"),
            r_idx(|ictx, ctx, st| {
                let r = crate::uop::sym::sget(st, "r");
                let (x, y) = contraction_operands(ctx, r.clone())?;
                let (sx, sy) = (is_simple_load(ctx, x.clone()), is_simple_load(ctx, y.clone()));
                if sx == sy {
                    return None;
                }
                let p = if sx { y } else { x };
                // Skip operands that are already materialized (a boundary from
                // another rule): re-staging them would put two stores in the
                // producer kernel.
                if matches!(
                    ctx.op_of(&p),
                    Ops::Param | Ops::Buffer | Ops::After | Ops::Stage | Ops::Store | Ops::Index
                ) {
                    return None;
                }
                ictx.realize_map.insert(p, None);
                None
            }),
        ),
    ];
    PatternMatcher::new(rules)
}

/// The `REDUCE(ADD, MUL(a, b))` node of a contraction, looking through casts,
/// movement ops and the reduce body's view. `None` for anything else (so
/// residual adds and other reductions are not split).
fn contraction_root(ctx: &UOpCtx, u: UopId) -> Option<UopId> {
    let mut cur = u;
    for _ in 0..6 {
        match ctx.op_of(&cur) {
            Ops::Cast | Ops::Contiguous | Ops::Reshape | Ops::Bitcast => {
                cur = ctx.srcs(&cur)[0].clone();
            }
            Ops::Reduce => {
                let mut mul = ctx.srcs(&cur)[0].clone();
                for _ in 0..8 {
                    match ctx.op_of(&mul) {
                        Ops::Cast
                        | Ops::Contiguous
                        | Ops::Reshape
                        | Ops::Permute
                        | Ops::Pad
                        | Ops::Expand
                        | Ops::Shrink
                        | Ops::Flip
                        | Ops::Bitcast
                        | Ops::Mselect => mul = ctx.srcs(&mul)[0].clone(),
                        _ => break,
                    }
                }
                return (ctx.op_of(&mul) == Ops::Mul && ctx.srcs(&mul).len() == 2)
                    .then_some(cur);
            }
            _ => return None,
        }
    }
    None
}

/// The `(a, b)` operands of a `REDUCE(ADD, view(MUL(a, b)))`.
fn contraction_operands(ctx: &UOpCtx, r: UopId) -> Option<(UopId, UopId)> {
    if ctx.op_of(&r) != Ops::Reduce {
        return None;
    }
    let mut mul = ctx.srcs(&r)[0].clone();
    for _ in 0..8 {
        match ctx.op_of(&mul) {
            Ops::Cast
            | Ops::Contiguous
            | Ops::Reshape
            | Ops::Permute
            | Ops::Pad
            | Ops::Expand
            | Ops::Shrink
            | Ops::Flip
            | Ops::Bitcast
            | Ops::Mselect => mul = ctx.srcs(&mul)[0].clone(),
            _ => break,
        }
    }
    if ctx.op_of(&mul) == Ops::Mul && ctx.srcs(&mul).len() == 2 {
        let srcs = ctx.srcs(&mul).to_vec();
        Some((srcs[0].clone(), srcs[1].clone()))
    } else {
        None
    }
}

/// A buffer/param view the library GEMM can read directly (no ALU).
fn is_simple_load(ctx: &UOpCtx, u: UopId) -> bool {
    let mut cur = u;
    for _ in 0..8 {
        match ctx.op_of(&cur) {
            Ops::Cast | Ops::Contiguous | Ops::Reshape | Ops::Permute | Ops::Pad | Ops::Expand
            | Ops::Shrink | Ops::Flip | Ops::Bitcast | Ops::Mselect => {
                cur = ctx.srcs(&cur)[0].clone();
            }
            Ops::Param | Ops::Buffer | Ops::After => return true,
            _ => return false,
        }
    }
    false
}

// ---------- pm_apply_rangeify rules ----------

/// `create_bufferize_and_index_srcs` (indexing.py:76).
fn create_bufferize_and_index_srcs(
    ictx: &mut IndexingContext,
    ctx: &mut UOpCtx,
    x: UopId,
) -> Vec<UopId> {
    let srcs = ctx.srcs(&x).to_vec();
    let data_src_count = data_srcs(ctx, ctx.op_of(&x), &srcs).len();
    let x_in_range_map = ictx.range_map.contains_key(&x);
    let x_rngs: Vec<UopId> = if x_in_range_map {
        ictx.range_map[&x].0.clone()
    } else {
        vec![]
    };
    let mut new_srcs = Vec::with_capacity(srcs.len());
    for (i, s) in srcs.iter().enumerate() {
        let mut new_src = s.clone();
        let src_rngs: Vec<UopId> = if x_in_range_map {
            broadcast_rngs(ctx, x.clone(), s.clone(), &x_rngs)
        } else {
            vec![]
        };
        if matches!(
            ctx.op_of(s),
            Ops::Param | Ops::Buffer | Ops::Mstack | Ops::Mselect | Ops::After
        ) {
            if x_in_range_map && i < data_src_count {
                new_src = ctx.index(new_src, src_rngs.clone());
            }
        } else if ictx.realize_map.contains_key(s) {
            let realized_ranges = ictx
                .realize_map
                .get(s)
                .cloned()
                .flatten()
                .expect("TGERR: realize map must contain range list");
            let s_out = ictx.range_map[s].1.clone();
            let closed_ranges: Vec<UopId> = s_out
                .iter()
                .enumerate()
                .filter(|(i, _)| realized_ranges.contains(&(*i as i64)))
                .map(|(_, r)| r.clone())
                .collect();
            if ctx.op_of(s) == Ops::Store {
                // add the ends if this is a store
                let rng_ends: Vec<UopId> = closed_ranges
                    .iter()
                    .filter(|&r| ctx.op_of(r) == Ops::Range)
                    .cloned()
                    .collect();
                new_src = ctx.end(s.clone(), rng_ends);
                ictx.realize_map.remove(s);
            } else {
                let removable =
                    !is_always_contiguous(ctx.op_of(s)) && !ictx.non_removable.contains(s);
                let s_dev = device_of(ctx, s.clone());
                let opts = if s_out.len() == realized_ranges.len() {
                    bufferize_opts(dev_to_arg(s_dev), Some(AddrSpace::Global), removable)
                } else {
                    bufferize_opts(dev_to_arg(s_dev), Some(AddrSpace::Local), removable)
                };
                let mut stage_srcs = vec![new_src];
                stage_srcs.extend(closed_ranges.iter().cloned());
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
                        .map(|(_, r)| r.clone())
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
        Some(DeviceSpec::Multi(v)) => Arg::Seq {
            tuple: false,
            items: v.into_iter().map(Arg::Str).collect(),
        },
        None => Arg::None,
    }
}

/// `create_bufferize_and_index_based_on_ranges` (indexing.py:103).
fn create_bufferize_and_index_based_on_ranges(
    ictx: &mut IndexingContext,
    ctx: &mut UOpCtx,
    x: UopId,
) -> Option<UopId> {
    if matches!(ctx.op_of(&x), Ops::Stage | Ops::Index) {
        return None;
    }
    let new_srcs = create_bufferize_and_index_srcs(ictx, ctx, x.clone());
    let (op, dt, _old, arg, tag) = ctx.snapshot(&x);
    Some(ctx.uop_tag(op, Some(dt), new_srcs, arg, tag))
}

/// `convert_pad_to_where_to_keep_behavior_local` (indexing.py:107).
fn convert_pad_to_where(ictx: &mut IndexingContext, ctx: &mut UOpCtx, x: UopId) -> Option<UopId> {
    if !ictx.range_map.contains_key(&x) {
        return None;
    }
    let bx = create_bufferize_and_index_based_on_ranges(ictx, ctx, x.clone())?;
    let (in_rngs, _out) = ictx.range_map[&x].clone();
    let mut valids: Vec<UopId> = in_rngs.iter().map(|r| ctx.get_valid(r.clone())).collect();
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
    let bx_srcs = ctx.srcs(&bx).to_vec();
    // python: UOp.const(x.dtype.const(0)) — a SCALAR const (no expand;
    // const_like would expand it to x's shape and break the sum-of-gate fold)
    let zero = ctx.const_py(ConstVal::Int(0), Some(ctx.dt_of(&bx_srcs[0])));
    let w = ctx.where_(valid, (bx_srcs[0]).clone(), zero);
    Some(w)
}

/// Stage the pointwise prologue recorded by `pm_generate_realize_map` inside a
/// reduce body: the reduce-body equivalent of
/// `create_bufferize_and_index_srcs`, run before the body is movement-folded.
fn stage_reduce_prologue(ictx: &mut IndexingContext, ctx: &mut UOpCtx, x: UopId) -> UopId {
    if ctx.op_of(&x) != Ops::Reduce {
        return x;
    }
    let body = ctx.srcs(&x)[0].clone();
    let nb = stage_prologue_in_body(ictx, ctx, body.clone());
    if nb == body {
        return x;
    }
    let (op, dt, mut srcs, arg, tag) = ctx.snapshot(&x);
    srcs[0] = nb.clone();
    let nx = ctx.uop_tag(op, Some(dt), srcs, arg, tag);
    if let Some(rm) = ictx.range_map.get(&x).cloned() {
        ictx.range_map.insert(nx.clone(), rm);
    }
    // The rebuild changes the node id. Carry the realization decision and the
    // "keep this boundary" mark across, or `pm_remove_bufferize` inlines the
    // STAGE and the contraction is fused again (it sees only `non_removable`).
    if let Some(rm) = ictx.realize_map.get(&x).cloned() {
        ictx.realize_map.insert(nx.clone(), rm);
    }
    if ictx.non_removable.contains(&x) {
        ictx.non_removable.insert(nx.clone());
    }
    if let Some(rm) = ictx.range_map.get(&body).cloned() {
        ictx.range_map.insert(nb, rm);
    }
    nx
}

/// Descend the body's casts/movement chain to the `MUL` and replace a recorded
/// pointwise operand with `INDEX(STAGE(operand))` (all of its axes realized).
fn stage_prologue_in_body(ictx: &mut IndexingContext, ctx: &mut UOpCtx, u: UopId) -> UopId {
    let op = ctx.op_of(&u);
    if op != Ops::Mul {
        let is_wrapper = matches!(
            op,
            Ops::Cast
                | Ops::Contiguous
                | Ops::Reshape
                | Ops::Permute
                | Ops::Pad
                | Ops::Expand
                | Ops::Shrink
                | Ops::Flip
                | Ops::Bitcast
                | Ops::Mselect
        );
        if !is_wrapper || ctx.srcs(&u).is_empty() {
            return u;
        }
        let inner = ctx.srcs(&u)[0].clone();
        let ni = stage_prologue_in_body(ictx, ctx, inner.clone());
        if ni == inner {
            return u;
        }
        let (op, dt, mut srcs, arg, tag) = ctx.snapshot(&u);
        srcs[0] = ni.clone();
        let nu = ctx.uop_tag(op, Some(dt), srcs, arg, tag);
        if let Some(rm) = ictx.range_map.get(&u).cloned() {
            ictx.range_map.insert(nu.clone(), rm);
        }
        if let Some(rm) = ictx.range_map.get(&inner).cloned() {
            ictx.range_map.insert(ni, rm);
        }
        return nu;
    }
    let srcs = ctx.srcs(&u).to_vec();
    let Some(pos) = srcs.iter().position(|s| {
        ictx.realize_map.contains_key(s)
            && ictx.range_map.contains_key(s)
            && !matches!(
                ctx.op_of(s),
                Ops::Param | Ops::Buffer | Ops::After | Ops::Stage | Ops::Store | Ops::Index
            )
    }) else {
        return u;
    };
    let p = srcs[pos].clone();
    // Only a real contraction: `A[m,k] * B[k,n]` has differently-shaped
    // operands, while `sum(x * mask)` (same shape) does not. Ranges are not
    // attached yet at this stage, so shapes are the available discriminator.
    // Only a real contraction: `A[m,k] * B[k,n]` has differently-shaped
    // operands, while `x*x` (LRN/Lp) and masked sums do not.
    let other_pos = 1 - pos;
    let other = srcs[other_pos].clone();
    // A movement-view operand is only worth staging while training: the
    // backward re-reads it, while an eval forward would just pay the copy.
    let is_view = matches!(
        ctx.op_of(&p),
        Ops::Permute
            | Ops::Reshape
            | Ops::Shrink
            | Ops::Pad
            | Ops::Expand
            | Ops::Flip
            | Ops::Mselect
            | Ops::Contiguous
    );
    let (sp, so) = (
        ctx.shape(&p).ok().flatten().map(|x| x.to_vec()),
        ctx.shape(&other).ok().flatten().map(|x| x.to_vec()),
    );
    if sp.is_none() || sp == so {
        return u;
    }
    // A movement-view operand is only worth staging while training (the
    // backward re-reads it; an eval forward would just pay the copy), and only
    // for small tensors: staging large conv/pool views increases peak memory
    // enough to OOM MNIST at bs512.
    if is_view {
        let numel = sp
            .as_ref()
            .map(|s| s.iter().map(|d| d.clone().as_i128_or_panic()).product::<i128>())
            .unwrap_or(0);
        if !crate::helpers::context::training() || numel > 4_000_000 {
            return u;
        }
    }
    let (Some((body_in, _)), Some((_, p_out))) = (
        ictx.range_map.get(&u).cloned(),
        ictx.range_map.get(&p).cloned(),
    ) else {
        return u;
    };
    let src_rngs = broadcast_rngs(ctx, u.clone(), p.clone(), &body_in);
    if p_out.is_empty() || src_rngs.is_empty() {
        return u;
    }
    let realized: Vec<i64> = (0..p_out.len() as i64).collect();
    ictx.realize_map.insert(p.clone(), Some(realized.clone()));
    ictx.non_removable.insert(p.clone());
    let opts = bufferize_opts(
        dev_to_arg(device_of(ctx, p.clone())),
        Some(crate::dtype::AddrSpace::Global),
        false,
    );
    let mut stage_srcs = vec![p.clone()];
    stage_srcs.extend(p_out.iter().cloned());
    let stage = ctx.uop(
        Ops::Stage,
        Some(ctx.dt_of(&p)),
        stage_srcs,
        Arg::Data(opts_data(&opts)),
    );
    let idx_rngs: Vec<UopId> = src_rngs
        .iter()
        .enumerate()
        .filter(|(j, _)| realized.contains(&(*j as i64)))
        .map(|(_, r)| r.clone())
        .collect();
    let new_p = ctx.index(stage, idx_rngs);
    let (op, dt, mut new_srcs, arg, tag) = ctx.snapshot(&u);
    new_srcs[pos] = new_p;
    let nu = ctx.uop_tag(op, Some(dt), new_srcs, arg, tag);
    if let Some(rm) = ictx.range_map.get(&u).cloned() {
        ictx.range_map.insert(nu.clone(), rm);
    }
    nu
}

/// `convert_reduce_to_reduce_with_ranges` (indexing.py:113).
fn convert_reduce_to_ranges(
    ictx: &mut IndexingContext,
    ctx: &mut UOpCtx,
    x: UopId,
) -> Option<UopId> {
    // arg[1] == 0 -> no axis reduce
    let (_op, dt, _srcs, arg, _tag) = ctx.snapshot(&x);
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
    // Stage a recorded pointwise prologue before the body is folded into
    // movement ops by `create_bufferize_and_index_based_on_ranges`; after that
    // the generic realize pass can no longer see the operand.
    let x = stage_reduce_prologue(ictx, ctx, x);
    let bx = create_bufferize_and_index_based_on_ranges(ictx, ctx, x.clone())?;
    let (in_rngs, _) = ictx.range_map[&x].clone();
    let new_ranges: Vec<UopId> = in_rngs[..n_axes as usize].to_vec();
    let bx_srcs = ctx.srcs(&bx).to_vec();
    let mut srcs = vec![(bx_srcs[0]).clone()];
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
    if !ictx.range_map.contains_key(&x) || ctx.dt_of(&x) == crate::dtype::VOID {
        return None;
    }
    let srcs = create_bufferize_and_index_srcs(ictx, ctx, x.clone());
    let r0 = ictx.range_map[&x].1[0].clone();
    let mut ret = srcs[srcs.len() - 1].clone();
    for k in (0..srcs.len() - 1).rev() {
        let kk = ctx.const_like(r0.clone(), ConstVal::Int(k as i128));
        let cond = ctx.eq(r0.clone(), kk);
        ret = ctx.where_(cond, (srcs[k]).clone(), ret);
    }
    Some(ret)
}

/// `remove_movement_op_after_rangeify` (indexing.py:131).
fn remove_movement_op_after_rangeify(
    ictx: &mut IndexingContext,
    ctx: &mut UOpCtx,
    x: UopId,
) -> Option<UopId> {
    if ictx.range_map.contains_key(&x) || ctx.op_of(&ctx.srcs(&x)[0]) == Ops::Index {
        return Some(ctx.srcs(&x)[0].clone());
    }
    None
}

/// Add to every kernel-root `END(STORE)` the loop ranges of its store that it
/// does not end yet.
///
/// `create_bufferize_and_index_srcs` ends the store's own out ranges before the
/// store's sources are rangeified, so a realized interior node (a contraction
/// staged under an elementwise epilogue) can contribute loop ranges that the
/// END never sees. `split_store` rejects such an END ("open ranges"), leaving
/// the kernel unformed; closing them makes the store a proper kernel that the
/// splitter can callify.
pub fn close_store_ends() -> PatternMatcher {
    static PM: crate::uop::upat::PmSlot = crate::uop::upat::PmSlot::new();
    PM.get_or_init(|| {
        PatternMatcher::new(vec![(
            UPat::ops1(Ops::End).with_src_any_len(true).named("e"),
            r_ctx(|ctx, st| {
                let e = crate::uop::sym::sget(st, "e");
                let srcs = ctx.srcs(&e).to_vec();
                if ctx.op_of(&srcs[0]) != Ops::Store {
                    return None;
                }
                let mut ended: Vec<UopId> = srcs[1..].to_vec();
                let mut changed = false;
                for r in ctx.ranges(&srcs[0]).to_vec() {
                    if ctx.op_of(&r) == Ops::Range && !ended.contains(&r) {
                        ended.push(r);
                        changed = true;
                    }
                }
                if !changed {
                    return None;
                }
                ended.sort_by_key(|r| AxisType::range_axis_id(ctx, r.clone()));
                let mut new_srcs = vec![srcs[0].clone()];
                new_srcs.extend(ended);
                Some(ctx.uop_tag(
                    Ops::End,
                    Some(crate::dtype::VOID),
                    new_srcs,
                    Arg::None,
                    ctx.tag_of(&e).clone(),
                ))
            }),
        )])
    })
    .clone()
}

/// `pm_apply_rangeify` (indexing.py:134).
pub fn pm_apply_rangeify() -> PatternMatcher {
    static PM: crate::uop::upat::PmSlot = crate::uop::upat::PmSlot::new();
    PM.get_or_init(build_pm_apply_rangeify).clone()
}

fn build_pm_apply_rangeify() -> PatternMatcher {
    let rules: Vec<(UPat, RuleFn)> = vec![
        // REDUCE(op, axis) -> REDUCE(op) with ranges
        (
            UPat::ops1(Ops::Reduce).named("x"),
            r_idx(|ictx, ctx, st| {
                convert_reduce_to_ranges(ictx, ctx, crate::uop::sym::sget(st, "x"))
            }),
        ),
        // PAD -> WHERE
        (
            UPat::ops1(Ops::Pad).named("x"),
            r_idx(|ictx, ctx, st| convert_pad_to_where(ictx, ctx, crate::uop::sym::sget(st, "x"))),
        ),
        // STACK -> WHERE select on the leading range
        (
            UPat::ops1(Ops::Stack).with_src_any_len(true).named("x"),
            r_idx(|ictx, ctx, st| {
                convert_stack_to_where(ictx, ctx, crate::uop::sym::sget(st, "x"))
            }),
        ),
        // finally, apply_rangeify
        (
            UPat::op_var(crate::uop::ops::OPS_ALL.to_vec()).named("x"),
            r_idx(|ictx, ctx, st| {
                create_bufferize_and_index_based_on_ranges(
                    ictx,
                    ctx,
                    crate::uop::sym::sget(st, "x"),
                )
            }),
        ),
        // remove movement op
        (
            UPat::op_var(group::MOVEMENT.to_vec()).named("x"),
            r_idx(|ictx, ctx, st| {
                remove_movement_op_after_rangeify(ictx, ctx, crate::uop::sym::sget(st, "x"))
            }),
        ),
    ];
    PatternMatcher::new(rules)
}

/// `pm_fix_deviceless` (indexing.py:147).
pub fn pm_fix_deviceless() -> PatternMatcher {
    static PM: crate::uop::upat::PmSlot = crate::uop::upat::PmSlot::new();
    PM.get_or_init(build_pm_fix_deviceless).clone()
}

fn build_pm_fix_deviceless() -> PatternMatcher {
    PatternMatcher::new(vec![(
        UPat::ops1(Ops::Stage).named("b"),
        r_dev(|dev, ctx, st| {
            let b = crate::uop::sym::sget(st, "b");
            let (_op, _dt, _srcs, arg, _tag) = ctx.snapshot(&b);
            let Arg::Data(d) = &arg else { return None };
            let addrspace = match d.get("addrspace") {
                Arg::Enum(e) => AddrSpace::from_u8(e.value as u8).expect("TGERR: bad addrspace"),
                _ => return None,
            };
            let is_none = matches!(d.get("device"), Arg::None);
            if addrspace == AddrSpace::Global && is_none {
                let mut nd = d.clone();
                let i = nd
                    .fields
                    .iter()
                    .position(|(n, _)| n == "device")
                    .expect("TGERR: device field");
                nd.fields[i].1 = dev_to_arg(dev.clone());
                Some(ctx.uop_tag(
                    Ops::Stage,
                    Some(ctx.dt_of(&b)),
                    ctx.srcs(&b).to_vec(),
                    Arg::Data(nd),
                    ctx.tag_of(&b).clone(),
                ))
            } else {
                None
            }
        }),
    )])
}

// ---------- movement application (indexing.py:152-185) ----------

/// `_apply_reshape` (indexing.py:152).
fn apply_reshape(
    ctx: &mut UOpCtx,
    in_shape: &[SVal],
    out_shape: &[SVal],
    urngs: UopId,
) -> Vec<UopId> {
    let mut acc = SVal::I(1);
    let mut axes_in: Vec<UopId> = Vec::new();
    let urngs_srcs = ctx.srcs(&urngs).to_vec();
    for (s, src) in out_shape.iter().zip(urngs_srcs.iter()).rev() {
        // python: acc*src with the int promoted via from_py -> weakint (the
        // commutative flip only applies to weakint, so the dtype matters here)
        let acc_u = match acc {
            SVal::I(v) => ctx.const_py(crate::dtype::ConstVal::Int(v), Some(crate::dtype::WEAKINT)),
            SVal::U(ref u) => u.clone(),
        };
        let m = ctx.uop(
            Ops::Mul,
            Some(crate::dtype::WEAKINT),
            vec![acc_u, src.clone()],
            Arg::None,
        );
        axes_in.push(m);
        acc = ctx.sval_mul(acc, s.clone());
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
        let su = ctx.sint_to_uop_dt(s.clone(), ctx.dt_of(&comb));
        axes_out.push(ctx.mod_(comb.clone(), su.clone()));
        comb = ctx.div_floor(comb, su);
    }
    axes_out.reverse();
    let sink = crate::tensor::sink(ctx, axes_out);
    // `indexing.py:152` is `@functools.cache`d; the interned SINK is the key
    // (identical reshapes across blocks/optimizer steps hit this).
    if let Some(cached) = ctx.reshape_cache.get(&sink) {
        return cached.clone();
    }
    let pm = symbolic()
        .add(&crate::uop::sym::pm_simplify_valid())
        .add(&crate::uop::sym::pm_drop_and_clauses());
    let out = graph_rewrite(
        ctx,
        sink.clone(),
        &pm,
        false,
        None,
        false,
        false,
        CtxVal::None,
    )
    .unwrap_or_else(|e| panic!("TGERR: reshape rewrite: {e}"));
    let ret = ctx.srcs(&out).to_vec();
    ctx.reshape_cache.insert(sink, ret.clone());
    ret
}

/// `apply_movement_op` (indexing.py:168) — rngs in, rngs out. Recomputed
/// (python's functools.cache is an identity/interning hack; structural
/// interning in the Rust graph makes recomputation produce identical nodes).
pub fn apply_movement_op(
    ctx: &mut UOpCtx,
    op: Ops,
    in_shape: &[SVal],
    arg: &[Arg],
    rngs: &[UopId],
) -> Vec<UopId> {
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
            rngs.iter()
                .zip(pairs.iter())
                .map(|(a, (off, _))| {
                    if *off == 0 {
                        a.clone()
                    } else {
                        let offu = ctx.const_like(a.clone(), ConstVal::Int(*off as i128));
                        ctx.add(a.clone(), offu)
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
            idx.iter().map(|&i| rngs[i].clone()).collect()
        }
        Ops::Flip => {
            let flips: Vec<bool> = arg
                .iter()
                .map(|a| match a {
                    Arg::Bool(b) => *b,
                    other => panic!("TGERR: bad flip arg {other:?}"),
                })
                .collect();
            rngs.iter()
                .zip(in_shape.iter())
                .zip(flips.iter())
                .map(|((a, s), f)| {
                    if *f {
                        let sm1 = ctx.sval_sub(s.clone(), SVal::I(1));
                        let sm1u = ctx.sval_to_uop(sm1);
                        ctx.sub(sm1u, a.clone())
                    } else {
                        a.clone()
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
            rngs.iter()
                .zip(in_shape.iter())
                .zip(pairs.iter())
                .map(|((r, sh), (off, sz))| {
                    let same = *sz == sh.clone().as_i128_or_panic() as i64 && *off == 0;
                    if same {
                        r.clone()
                    } else {
                        // (r >= off) & (r < (sh+off))
                        let offu = ctx.const_like(r.clone(), ConstVal::Int(*off as i128));
                        let ge = crate::tensor::elementwise::ge(ctx, r.clone(), offu.clone());
                        let hi = ctx.sval_add(sh.clone(), SVal::I(*off as i128));
                        let hiu = ctx.sint_to_uop_dt(hi, ctx.dt_of(r));
                        let lt = ctx.lt(r.clone(), hiu);
                        let cond = ctx.and_(ge, lt);
                        let simplified =
                            graph_rewrite(ctx, cond, &pm, false, None, false, false, CtxVal::None)
                                .unwrap_or_else(|e| panic!("TGERR: pad rewrite: {e}"));
                        let sub = ctx.sub(r.clone(), offu);
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
            for (i, r) in ctx.ranges(&sink.clone()).iter().enumerate() {
                let (rop, rdt, rsrcs, _rarg, _rtag) = ctx.snapshot(r);
                let ph = ctx.uop_tag(
                    rop,
                    Some(rdt),
                    rsrcs[..1].to_vec(),
                    Arg::Seq {
                        tuple: true,
                        items: vec![
                            Arg::Int(i as i64),
                            Arg::Enum(AxisType::Placeholder.to_enum_val()),
                        ],
                    },
                    Arg::None,
                );
                sub_array.insert(r.clone(), ph.clone());
                back.insert(ph, r.clone());
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
            ctx.srcs(&restored).to_vec()
        }
        _ => panic!("TGERR: {op:?} is not a MovementOp"),
    }
}

// ---------- run_rangeify (indexing.py:187) ----------

fn all_same_ids(v: &[UopId]) -> bool {
    v.iter().all(|x| *x == v[0])
}

/// `gate_kernel_sink` (ops.py:1760).
pub fn gate_kernel_sink(ctx: &UOpCtx, x: UopId) -> bool {
    if ctx.op_of(&x) == Ops::Linear {
        return false;
    }
    if ctx.op_of(&x) == Ops::Sink
        && let Arg::Data(d) = ctx.arg_of(&x)
        && d.tag == "KernelInfo"
    {
        return false;
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
    let has_ending = ending_ranges
        .get(&x)
        .map(|v| !v.is_empty())
        .unwrap_or(false);
    if !has_ending
        || !(group::in_(group::ELEMENTWISE, ctx.op_of(&x)) || ctx.op_of(&x) == Ops::Reduce)
    {
        return;
    }
    let mut ra = ictx
        .realize_map
        .get(&x)
        .cloned()
        .flatten()
        .unwrap_or_default();
    let er = ending_ranges.get(&x).cloned().unwrap_or_default();
    for (i, r) in out_rngs.iter().enumerate() {
        if ra.contains(&(i as i64)) {
            continue;
        }
        // `if not (PCONTIG > 1) or any(rr.arg > e.arg ...)` (indexing.py:274).
        // PCONTIG defaults to 0, so the `not (PCONTIG > 1)` arm is **always true**:
        // with ending ranges present, *every* axis is realized. The per-axis
        // comparison is dead under the default context.
        let pcontig_gt_1 = false;
        let rranges = ctx.ranges(&r.clone()).to_vec();
        if !pcontig_gt_1
            || rranges.iter().any(|rr| {
                er.iter().any(|e| {
                    AxisType::range_axis_id(ctx, rr.clone())
                        > AxisType::range_axis_id(ctx, e.clone())
                })
            })
        {
            ra.push(i as i64);
        }
    }
    ending_ranges.insert(x.clone(), vec![]);
    if !ra.is_empty() {
        ictx.realize_map.insert(x, Some(ra.clone()));
        *out_rngs = out_rngs
            .iter()
            .enumerate()
            .map(|(i, r)| {
                if ra.contains(&(i as i64)) {
                    ictx.new_range(ctx, x_shape[i].clone(), AxisType::Weak)
                } else {
                    r.clone()
                }
            })
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
    if group::in_(group::MOVEMENT, ctx.op_of(&x)) {
        let (op, _dt, srcs, arg, _tag) = ctx.snapshot(&x);
        let in_shape = ctx
            .shape(&srcs[0])
            .expect("TGERR: movement src shape")
            .unwrap_or_default();
        // RESHAPE/EXPAND store their shape in shape-srcs (our M2 convention),
        // python carries it in the movement arg — normalize here. PAD/SHRINK
        // pairs likewise live in shape-srcs ((start, stop) lists), python's
        // marg (ops.py:779) reads them from src[1]/src[2] — without this the
        // pad/shrink arms zip over nothing and the ranges pass through
        // unwrapped (the arange cumsum mask bug).
        let arg_refs: Vec<Arg> = if matches!(op, Ops::Reshape | Ops::Expand) {
            match ctx.marg_shape(x.clone()) {
                Ok(shape) => shape
                    .iter()
                    .map(|s| match s {
                        SVal::I(v) => Arg::Int(*v as i64),
                        SVal::U(_) => panic!("TGERR: symbolic reshape dim unsupported in rangeify"),
                    })
                    .collect(),
                Err(_) => vec![],
            }
        } else if matches!(op, Ops::Pad | Ops::Shrink) {
            match ctx.marg_pairs(x.clone()) {
                Ok(pairs) => pairs
                    .iter()
                    .map(|(a, b)| Arg::Seq {
                        tuple: true,
                        items: vec![Arg::Int(*a), Arg::Int(*b)],
                    })
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
    if ctx.op_of(&x) == Ops::Stack {
        rngs = out_rngs[1..].to_vec();
    }
    // if the EXPAND is used to inject a range, we don't mark it as ending_ranges. otherwise we do.
    if ctx.op_of(&x) == Ops::Expand && x_shape.iter().all(|y| matches!(y, SVal::I(_))) {
        let n = {
            match ctx.marg_pairs(x.clone()) {
                Ok(v) => v.len(),
                Err(_) => match ctx.marg_ints(x.clone()) {
                    Ok(v) => v.len(),
                    Err(_) => ctx.marg_shape(x.clone()).map(|s| s.len()).unwrap_or(0),
                },
            }
        };
        // python slice clamps: a scalar expand has no out ranges
        let sink = crate::tensor::sink(ctx, out_rngs[..n.min(out_rngs.len())].to_vec());
        let rr = ctx.ranges(&sink).to_vec();
        ending_ranges.entry(x.clone()).or_default().extend(rr);
    }

    // REDUCE creates ranges for the axes it is reducing
    if ctx.op_of(&x) == Ops::Reduce {
        let (_op, _dt, srcs, arg, _tag) = ctx.snapshot(&x);
        let n_axes = match &arg {
            Arg::Seq { tuple: true, items } => match &items[1] {
                Arg::Int(n) => *n,
                _ => 0,
            },
            _ => 0,
        };
        if n_axes > 0 {
            let in_shape = ctx
                .shape(&srcs[0])
                .expect("TGERR: reduce src shape")
                .unwrap_or_default();
            let mut new_rngs: Vec<UopId> = in_shape[..n_axes as usize]
                .iter()
                .map(|s| ictx.new_range(ctx, s.clone(), AxisType::Reduce))
                .collect();
            new_rngs.extend(out_rngs.iter().cloned());
            rngs = new_rngs;
        }
    }

    // assign to the range map. rngs are the input ranges, out_rngs are the output ranges.
    ictx.range_map.insert(x, (rngs, out_rngs.to_vec()));
}

/// Add realize-map entries so no fused kernel needs more than `max_buffers`
/// buffer arguments (`limit` includes the kernel's output buffer).
///
/// Walks the graph producer-first, carrying each node's set of distinct input
/// buffers (capped just over the budget). When a node's inputs exceed the
/// budget it greedily realizes the largest non-leaf source, replacing that
/// source's set with a single buffer in the consumer. This bounds every
/// prospective kernel while keeping the cuts as late (and as few) as possible.
fn budget_realizations(
    ctx: &mut UOpCtx,
    sink: UopId,
    ictx: &mut IndexingContext,
    max_buffers: usize,
) {
    // Leave ~1/3 headroom for scheduler-created output/stage buffers and
    // multi-output kernels: the leaf-buffer union is a lower bound on the final
    // kernel's argument count.
    let budget = (max_buffers * 2 / 3).max(1);
    let order = ctx.toposort(&sink);
    let mut downstream: HashMap<UopId, Vec<UopId>> = HashMap::new();
    let mut realized: std::collections::HashSet<UopId> = ictx.realize_map.keys().cloned().collect();
    let mut extra: Vec<UopId> = Vec::new();

    for x in order {
        let op = ctx.op_of(&x);
        if matches!(op, Ops::Buffer | Ops::Param) {
            downstream.insert(x.clone(), vec![x.clone()]);
            continue;
        }
        let (_op, _dt, raw_srcs, _arg, _tag) = ctx.snapshot(&x);
        let srcs = data_srcs(ctx, op, &raw_srcs);
        // (source, its downstream set)
        let mut src_sets: Vec<(UopId, Vec<UopId>)> = srcs
            .iter()
            .map(|s| {
                let set = downstream
                    .get(s)
                    .cloned()
                    .unwrap_or_else(|| vec![s.clone()]);
                (s.clone(), set)
            })
            .collect();

        let mut union = union_capped(
            src_sets.iter().flat_map(|(_, set)| set.iter().cloned()),
            budget + 1,
        );
        while union.len() > budget {
            // Realize the largest non-leaf source that is not a boundary yet.
            let mut best: Option<usize> = None;
            for (i, (s, set)) in src_sets.iter().enumerate() {
                if set.len() <= 1
                    || matches!(
                        ctx.op_of(s),
                        Ops::Buffer
                            | Ops::Param
                            | Ops::After
                            | Ops::Call
                            | Ops::Function
                            | Ops::Linear
                            | Ops::Mstack
                            | Ops::Mselect
                    )
                {
                    continue;
                }
                if best
                    .map(|b| set.len() > src_sets[b].1.len())
                    .unwrap_or(true)
                {
                    best = Some(i);
                }
            }
            let Some(i) = best else { break };
            let s = src_sets[i].0.clone();
            if realized.contains(&s) {
                // Already a boundary; its set should be a singleton, but guard
                // against duplicate cut attempts.
                src_sets[i].1 = vec![s.clone()];
                union = union_capped(
                    src_sets.iter().flat_map(|(_, set)| set.iter().cloned()),
                    budget + 1,
                );
                continue;
            }
            if ctx.shape(&s).ok().flatten().is_none() {
                break; // cannot materialize a shape-less node
            }
            realized.insert(s.clone());
            extra.push(s.clone());
            src_sets[i].1 = vec![s.clone()];
            union = union_capped(
                src_sets.iter().flat_map(|(_, set)| set.iter().cloned()),
                budget + 1,
            );
        }

        // The kernel producing x needs `union`; downstream consumers see x as
        // one buffer only if x is already/now a realized boundary.
        if realized.contains(&x) {
            downstream.insert(x.clone(), vec![x.clone()]);
        } else {
            downstream.insert(x.clone(), union);
        }
    }

    for x in extra {
        // `None` mirrors `pm_generate_realize_map`'s "realize all axes" marker;
        // the main rangeify loop fills in the concrete axis list.
        ictx.realize_map.insert(x, None);
    }
}

/// Union of a stream of buffer ids, deduplicated and capped at `cap` entries.
fn union_capped<I: Iterator<Item = UopId>>(it: I, cap: usize) -> Vec<UopId> {
    let mut out: Vec<UopId> = Vec::new();
    for u in it {
        if out.contains(&u) {
            continue;
        }
        out.push(u);
        if out.len() >= cap {
            break;
        }
    }
    out
}

fn on_host_device(ctx: &mut UOpCtx, u: UopId) -> bool {
    matches!(
        device_of(ctx, u),
        Some(DeviceSpec::Single(d))
            if d.starts_with("RUST")
                || d.starts_with("CC")
                || d.starts_with("RAYON")
                || d.starts_with("INTERP")
                || d.starts_with("CPU")
    )
}

/// Realize each *interior* contraction on a host backend when its reduction is
/// large enough to pay for the extra buffer.
///
/// A contraction consumed directly by a store is already its own kernel, and
/// the host GEMM claims it (including an affine bias/residual/scale epilogue).
/// An interior one (the FFN expansion under GELU, for instance) is fused into
/// a larger kernel whose elementwise epilogue blocks the library match, and
/// the generated matmul runs at roughly a tenth of the packed kernel's rate.
/// Materializing it makes the contraction a plain GEMM plus a small
/// elementwise kernel.
///
/// Independently of the split, a *big* contraction (>= 16M MACs) whose operand
/// is a pointwise expression gets that operand staged: attention's `Q`/`K` add
/// their bias inside the reduction, so the score kernel is neither a library
/// GEMM nor a reusable buffer. Staging costs one small copy and turns the same
/// kernel into a plain batched matmul.
fn realize_interior_contractions(
    ctx: &mut UOpCtx,
    order: &[UopId],
    consumer_map: &HashMap<UopId, Vec<UopId>>,
    ictx: &mut IndexingContext,
) {
    for x in order {
        if ctx.op_of(x) != Ops::Reduce || ictx.realize_map.contains_key(x) {
            continue;
        }
        let Some(cons) = consumer_map.get(x) else {
            continue;
        };
        // A kernel root (consumed directly by a store/boundary) is left alone.
        if cons.is_empty()
            || cons.iter().any(|c| {
                matches!(
                    ctx.op_of(c),
                    Ops::Store | Ops::After | Ops::Call | Ops::Function | Ops::Sink
                )
            })
        {
            continue;
        }
        let (_op, _dt, srcs, arg, _tag) = ctx.snapshot(x);
        let n_axes = match &arg {
            Arg::Seq { tuple: true, items } => match items.get(1) {
                Some(Arg::Int(n)) => *n,
                _ => 0,
            },
            _ => 0,
        };
        if n_axes <= 0 || srcs.is_empty() {
            continue;
        }
        let src_shape = ctx
            .shape(&srcs[0])
            .ok()
            .flatten()
            .map(|s| s.to_vec())
            .unwrap_or_default();
        let k: i128 = src_shape
            .iter()
            .take(n_axes as usize)
            .map(|d| d.clone().as_i128_or_panic())
            .product();
        if !on_host_device(ctx, x.clone()) {
            continue;
        }
        let out_numel: i128 = ctx
            .shape(x)
            .ok()
            .flatten()
            .map(|s| {
                s.iter()
                    .map(|d| d.clone().as_i128_or_panic())
                    .product::<i128>()
            })
            .unwrap_or(i128::MAX);
        let macs = out_numel.saturating_mul(k);
        let Some((a, b)) = contraction_operands(ctx, x.clone()) else {
            continue;
        };
        let simple = [
            is_simple_load(ctx, a.clone()),
            is_simple_load(ctx, b.clone()),
        ];
        // Stage a pointwise operand of a big contraction (attention's biased
        // Q/K) so the kernel becomes a plain matmul. Small enough to be cheap
        // and only when the GEMM is large.
        if macs >= 16_000_000 {
            for (operand, is_simple) in [(a.clone(), simple[0]), (b.clone(), simple[1])] {
                if is_simple || operand == *x {
                    continue;
                }
                if ctx
                    .shape(&operand)
                    .ok()
                    .flatten()
                    .map(|s| {
                        s.iter()
                            .map(|d| d.clone().as_i128_or_panic())
                            .product::<i128>()
                    })
                    .unwrap_or(i128::MAX)
                    <= 4_000_000
                {
                    ictx.realize_map.insert(operand.clone(), None);
                    ictx.non_removable.insert(operand);
                }
            }
        }
        let numel_of = |ctx: &mut UOpCtx, u: &UopId| -> i128 {
            ctx.shape(u)
                .ok()
                .flatten()
                .map(|s| {
                    s.iter()
                        .map(|d| d.clone().as_i128_or_panic())
                        .product::<i128>()
                })
                .unwrap_or(i128::MAX)
        };
        if a == b {
            // `x * x` (a normalisation variance): cheap to stage, and the split
            // materialises the normalised activation the projections read.
            if macs < 1_000_000 || numel_of(ctx, &a) > 512_000 {
                continue;
            }
        } else if simple[0] == simple[1] {
            // Both operands are already buffers (attention's Q/K/P/V views) or
            // both are expressions (a weight-gradient contraction). Only the
            // former is cheap; require a big GEMM so the extra kernel pays.
            if !simple[0] || macs < 16_000_000 {
                continue;
            }
        } else {
            // Exactly one operand is a buffer: the forward projection shape
            // (`activation @ weight`). Stage the activation; a small K cannot
            // amortise the extra kernel (attention's head contractions), and a
            // tiny GEMM is not worth a buffer either.
            if k < 512 || macs < 1_000_000 {
                continue;
            }
        }
        let mut stage: Vec<UopId> = Vec::new();
        let mut ok = true;
        for (operand, is_simple) in [a, b].into_iter().zip(simple) {
            if is_simple {
                continue;
            }
            let numel = ctx
                .shape(&operand)
                .ok()
                .flatten()
                .map(|s| {
                    s.iter()
                        .map(|d| d.clone().as_i128_or_panic())
                        .product::<i128>()
                })
                .unwrap_or(i128::MAX);
            if numel > 8_000_000 {
                ok = false;
                break;
            }
            stage.push(operand);
        }
        if !ok {
            continue;
        }
        ictx.realize_map.insert(x.clone(), None);
        ictx.non_removable.insert(x.clone());
        for operand in stage {
            ictx.realize_map.insert(operand.clone(), None);
            ictx.non_removable.insert(operand);
        }
    }
}

/// `run_rangeify` — rewrite the SINK into ranged INDEX/STAGE/RANGE form.
pub fn run_rangeify(ctx: &mut UOpCtx, tsink: UopId, _debug: bool, max_buffers: usize) -> UopId {
    let shared = Rc::new(RefCell::new(IndexingContext::new()));

    // get ops to realize
    {
        let pm_realize = pm_generate_realize_map();
        graph_rewrite(
            ctx,
            tsink.clone(),
            &pm_realize,
            false,
            None,
            false,
            false,
            CtxVal::Idx(shared.clone()),
        )
        .unwrap_or_else(|e| panic!("TGERR: get realize: {e}"));
    }

    // Fusion budget: a device with a hard kernel-buffer limit (Metal exposes 31
    // `[[buffer(n)]]` slots) needs intermediate realizations so no one kernel
    // reads more buffers than the limit. This is a port addition: tinygrad has
    // no equivalent pass. `usize::MAX` (all non-Metal devices) is a no-op.
    if max_buffers != usize::MAX {
        let mut ictx = shared.borrow_mut();
        budget_realizations(ctx, tsink.clone(), &mut ictx, max_buffers);
    }

    // get the consumer map
    let tsink_toposort = {
        let ictx = shared.borrow();
        let _ = ictx;
        ctx.toposort_gated(tsink.clone(), Some(&|x| gate_kernel_sink(ctx, x)), true)
    };
    let mut consumer_map: HashMap<UopId, Vec<UopId>> = HashMap::new();
    for n in &tsink_toposort {
        consumer_map.entry(n.clone()).or_default();
    }
    for c in &tsink_toposort {
        let (c_op, _dt, c_srcs, _arg, _tag) = ctx.snapshot(c);
        for x in data_srcs(ctx, c_op, &c_srcs) {
            if let Some(v) = consumer_map.get_mut(&x)
                && !v.contains(c)
            {
                v.push(c.clone());
            }
        }
    }

    // Host backends: the packed CPU GEMM is ~10x the generated matmul, so an
    // interior contraction with a big reduction is worth its own buffer.
    realize_interior_contractions(ctx, &tsink_toposort, &consumer_map, &mut shared.borrow_mut());

    // explicit rangeify
    let mut ending_ranges: HashMap<UopId, Vec<UopId>> = HashMap::new();
    for x in tsink_toposort.iter().rev() {
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
        let consumers = consumer_map.get(x).cloned().unwrap_or_default();
        let mut er: Vec<UopId> = Vec::new();
        for u in &consumers {
            if let Some(v) = ending_ranges.get(u) {
                er.extend(v.iter().cloned());
            }
        }
        ending_ranges.insert(x.clone(), er);

        // ranges the consumers iterate that this node broadcasts over
        let x_shape = ctx
            .shape(x)
            .expect("TGERR: rangeify node shape")
            .unwrap_or_default();
        let mut ended: Vec<UopId> = Vec::new();
        for c in &consumers {
            if !ictx.range_map.contains_key(c) || !group::in_(group::BROADCASTABLE, ctx.op_of(c)) {
                continue;
            }
            let c_shape = ctx
                .shape(c)
                .expect("TGERR: consumer shape")
                .unwrap_or_default();
            for i in crate::tensor::movement::broadcast_axes(ctx, &x_shape, &c_shape) {
                ended.push((ictx.range_map[c].0[i]).clone());
            }
        }
        let ended_sink = crate::tensor::sink(ctx, ended);
        let broadcast_ending_ranges: Vec<UopId> = ctx.ranges(&ended_sink).to_vec();
        if ctx.op_of(x) == Ops::Reduce {
            ending_ranges
                .get_mut(x)
                .unwrap()
                .extend(broadcast_ending_ranges.iter().cloned());
        }

        // *** the ranges on the output are
        //  1. new if this op is realized
        //  2. from the single consumer if this op only has one consumer
        //  3. potentially new if this op has 2+ consumers

        let mut consumer_rngs: Vec<Vec<UopId>> = Vec::new();
        for c in &consumers {
            if ictx.range_map.contains_key(c) {
                let crngs = ictx.range_map[c].0.clone();
                consumer_rngs.push(broadcast_rngs(ctx, c.clone(), x.clone(), &crngs));
            }
        }

        let mut out_rngs: Vec<UopId>;
        if ictx.realize_map.contains_key(x) {
            // if this is in the realize_map, we create new ranges (at the output)
            out_rngs = x_shape
                .iter()
                .map(|s| ictx.new_range(ctx, s.clone(), AxisType::Weak))
                .collect();
            // all ranges are ended now
            ending_ranges.insert(x.clone(), vec![]);
            // mark all ranges as ended
            assert!(
                ictx.realize_map.get(x).unwrap().is_none(),
                "TGERR: realize_map[x] must be None"
            );
            ictx.realize_map
                .insert(x.clone(), Some((0..x_shape.len() as i64).collect()));
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
                    let r = &cr[i].clone();
                    local_rngs.push(ctx.get_idx(r.clone()));
                    valids.push(ctx.get_valid(r.clone()));
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
                    let minimum_valid = if valids.is_empty() {
                        f
                    } else {
                        ctx.usum(f, valids)
                    };
                    let base = ctx.valid((local_rngs[0]).clone(), minimum_valid);
                    let merged = graph_rewrite(
                        ctx,
                        base,
                        symbolic(),
                        false,
                        None,
                        false,
                        false,
                        CtxVal::None,
                    )
                    .unwrap_or_else(|e| panic!("TGERR: minimum_valid: {e}"));
                    _out_rngs.push(merged);
                } else {
                    _out_rngs.push(ictx.new_range(
                        ctx,
                        <SInt as Clone>::clone(&x_shape[i]),
                        AxisType::Weak,
                    ));
                    _realize_axis.push(i as i64);
                }
            }
            out_rngs = _out_rngs;
            // we have to (partially) realize here if there's new ranges
            if !_realize_axis.is_empty() {
                ictx.realize_map.insert(x.clone(), Some(_realize_axis));
            }
        }

        // if this element is a reduce and there's ended ranges, we might have to end some other ranges
        maybe_realize_axes(
            &mut ictx,
            ctx,
            x.clone(),
            &x_shape,
            &mut out_rngs,
            &mut ending_ranges,
        );
        ending_ranges
            .get_mut(x)
            .unwrap()
            .extend(broadcast_ending_ranges.iter().cloned());

        finish_node(
            &mut ictx,
            ctx,
            x.clone(),
            &x_shape,
            &out_rngs,
            &mut ending_ranges,
        );

        // python: DEBUG_RANGEIFY trace (indexing.py:306)
        if _debug {
            let rdesc = |ctx: &UOpCtx, rs: &[UopId]| -> String {
                rs.iter()
                    .map(|r| match ctx.arg_of(r) {
                        Arg::Seq { tuple: true, items }
                            if matches!(items.as_slice(), [Arg::Int(_), _]) =>
                        {
                            format!(
                                "r{}({:?})",
                                crate::uop::range::AxisType::range_axis_id(ctx, r.clone()),
                                crate::uop::range::AxisType::from_arg(&items[1])
                            )
                        }
                        _ => format!("#{}", r.id()),
                    })
                    .collect::<Vec<_>>()
                    .join(",")
            };
            let shape_s: Vec<String> = x_shape
                .iter()
                .map(|s| match s {
                    crate::uop::sint::SVal::I(i) => i.to_string(),
                    crate::uop::sint::SVal::U(u) => format!("#{}", u.id()),
                })
                .collect();
            eprintln!(
                "{} {:2} {:?} ({}) {} [{}] -> [{}]",
                if ictx.realize_map.contains_key(x) {
                    "***"
                } else {
                    "   "
                },
                consumers.len(),
                ctx.op_of(x),
                shape_s.join(","),
                ending_ranges.get(x).map(|v| v.len()).unwrap_or(0),
                rdesc(ctx, &ictx.range_map[x].0),
                rdesc(ctx, &ictx.range_map[x].1)
            );
        }
    }

    // apply rangeify (bottom up)
    let pm = pm_apply_rangeify();
    let tsink = graph_rewrite(
        ctx,
        tsink,
        &pm,
        true,
        None,
        false,
        false,
        CtxVal::Idx(shared.clone()),
    )
    .unwrap_or_else(|e| panic!("TGERR: apply rangeify: {e}"));

    // if a deviceless value must materialize, place it on the sink device
    let dev = device_of(ctx, tsink.clone());
    let pm_dev = pm_fix_deviceless();
    graph_rewrite(
        ctx,
        tsink,
        &pm_dev,
        false,
        None,
        false,
        false,
        CtxVal::Dev(dev),
    )
    .unwrap_or_else(|e| panic!("TGERR: add device to deviceless: {e}"))
}
