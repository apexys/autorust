//! M3 movement mixin (mixin/movement.py): all sugar is UOp-level; `_mop` emits
//! shape-args as SRC consts (RESHAPE/EXPAND: 1 shape src; PAD/SHRINK: lo/end list
//! srcs; PERMUTE/FLIP: int-list arg; STACK: promoted srcs).

use crate::dtype::DTypeId;
use crate::uop::arg::Arg;
use crate::uop::ctx::{Shape, UOpCtx};
use crate::uop::graph::UopId;
use crate::uop::ops::Ops;
use crate::uop::sint::SVal;

use super::ix::Ix;
use super::{sink, stack};

/// Python `ceildiv(num, amt) = (num + amt - 1) // amt` (helpers.py:63; amt > 0).
pub fn ceildiv(a: i128, b: i128) -> i128 {
    assert!(b > 0, "TGERR: ceildiv amount must be positive");
    (a + b - 1).div_euclid(b)
}

pub fn round_up(num: i128, amt: i128) -> i128 {
    (num + amt - 1) / amt * amt
}

/// `prod` over sint shapes, as SVal (ints fold; UOp dims build MUL chains via ctx).
pub fn prod_sval(ctx: &mut UOpCtx, shape: &[SVal]) -> SVal {
    if shape.iter().all(|s| matches!(s, SVal::I(_))) {
        let mut p: i128 = 1;
        for s in shape {
            p *= s.as_i128_or_panic();
        }
        return SVal::I(p);
    }
    let mut acc = ctx.cint(1);
    for &s in shape {
        let u = match s {
            SVal::I(v) => ctx.cint_dt(v as i64, crate::dtype::default_int()),
            SVal::U(u) => u,
        };
        acc = ctx.alu2(Ops::Mul, acc, u);
    }
    SVal::U(acc)
}

pub fn ndim(shape: &[SVal]) -> usize {
    shape.len()
}

/// `MovementMixin._resolve_dim(dim, extra=False)` (movement.py:81-86).
pub fn resolve_dim(n: usize, dim: i64, extra: bool) -> usize {
    let total = n as i64 + extra as i64;
    let lo = -max_i64(1, total);
    let hi = max_i64(1, total) - 1;
    assert!(lo <= dim && dim <= hi, "TGERR: IndexError dim {dim} out of range [{lo}, {hi}]");
    (if dim < 0 { dim + total } else { dim }) as usize
}

fn max_i64(a: i64, b: i64) -> i64 {
    if a > b {
        a
    } else {
        b
    }
}

/// `_align_left` (ops.py:67-69): left-pad the shorter shape with 1s.
pub fn align_left(a: &[SVal], b: &[SVal]) -> (Shape, Shape) {
    let max_dim = a.len().max(b.len());
    let pad = |s: &[SVal]| {
        let mut v = vec![SVal::I(1); max_dim - s.len()];
        v.extend_from_slice(s);
        v
    };
    (pad(a), pad(b))
}

/// `broadcast_axes(src_shape, out_shape)` (ops.py:79-82).
pub fn broadcast_axes(ctx: &mut UOpCtx, src_shape: &[SVal], out_shape: &[SVal]) -> Vec<usize> {
    let nleft = out_shape.len() as i64 - src_shape.len() as i64;
    assert!(nleft >= 0, "TGERR: cannot broadcast {src_shape:?} into {out_shape:?}");
    let nleft = nleft as usize;
    let mut ret: Vec<usize> = (0..nleft).collect();
    for (i, s) in src_shape.iter().enumerate() {
        if ctx.sval_is_one(s).unwrap_or(false)
            && !ctx.sval_is_one(&out_shape[nleft + i]).unwrap_or(true)
        {
            ret.push(nleft + i);
        }
    }
    ret
}

/// `MovementMixin._mop` for RESHAPE/EXPAND (shape list arg).
pub fn mop_shape(ctx: &mut UOpCtx, op: Ops, x: UopId, shape: &[SVal]) -> UopId {
    if op == Ops::Expand && shape.is_empty() {
        return x;
    }
    let usrcs = vec![ctx.shape_arg(shape)];
    emit_with_shape_srcs(ctx, op, x, usrcs, Arg::None)
}

/// `MovementMixin._mop` for PAD/SHRINK (`(start, stop)` pairs).
pub fn mop_pairs(ctx: &mut UOpCtx, op: Ops, x: UopId, arg: &[(SVal, SVal)]) -> UopId {
    if arg.is_empty() {
        assert!(ctx.shape(x).unwrap().unwrap_or_default().is_empty(), "TGERR: 0 len arg only valid on zero length shape");
        return x;
    }
    let starts: Vec<SVal> = arg.iter().map(|(a, _)| *a).collect();
    let ends: Vec<SVal> = arg.iter().map(|(_, b)| *b).collect();
    let usrcs = vec![ctx.shape_arg(&starts), ctx.shape_arg(&ends)];
    emit_with_shape_srcs(ctx, op, x, usrcs, Arg::None)
}

/// shared tail of `_mop`: `UOp(op, src=(self,)+UOp.sink(*usrcs).simplify().src)`.
fn emit_with_shape_srcs(ctx: &mut UOpCtx, op: Ops, x: UopId, usrcs: Vec<UopId>, arg: Arg) -> UopId {
    let mut srcs = vec![x];
    if !usrcs.is_empty() {
        let s0 = sink(ctx, usrcs);
        let sid = crate::uop::upat::simplify(ctx, s0);
        srcs.extend(ctx.srcs(sid).to_vec());
    }
    ctx.uop(op, None, srcs, arg)
}

/// `MovementMixin.reshape` (movement.py:165-184). `shape` has Nones pre-resolved.
pub fn reshape(ctx: &mut UOpCtx, x: UopId, shape: &[SVal]) -> UopId {
    reshape_internal(ctx, x, shape)
}

pub(crate) fn reshape_internal(ctx: &mut UOpCtx, x: UopId, shape: &[SVal]) -> UopId {
    let self_shape = ctx.shape(x).unwrap().unwrap_or_default();
    let mut new_shape: Vec<SVal> = shape.to_vec();
    // resolve -1
    let c = new_shape.iter().filter(|s| **s == SVal::I(-1)).count();
    assert!(c <= 1, "TGERR: only one dimension can be inferred using -1");
    if c == 1 {
        let n = new_shape.len();
        let prod_self = prod_sval(ctx, &self_shape);
        let prod_rest = prod_sval(ctx, &new_shape.iter().filter(|s| **s != SVal::I(-1)).cloned().collect::<Vec<_>>());
        // -prod(self.shape) // prod(rest)
        let neg_prod = match prod_self {
            SVal::I(v) => SVal::I(-v),
            SVal::U(u) => SVal::U(ctx.alu1(Ops::Neg, u)),
        };
        let dim = ctx.sval_div_floor(neg_prod, prod_rest);
        new_shape = new_shape.iter().map(|s| if *s == SVal::I(-1) { dim } else { *s }).collect();
        let _ = n;
    }
    // prod check (only when all-int; symbolic prods compare structurally in Python)
    if self_shape.iter().all(|s| matches!(s, SVal::I(_))) && new_shape.iter().all(|s| matches!(s, SVal::I(_))) {
        let a: i128 = self_shape.iter().map(|s| s.as_i128_or_panic()).product();
        let b: i128 = new_shape.iter().map(|s| s.as_i128_or_panic()).product();
        assert_eq!(a, b, "TGERR: size mismatch, can't reshape {self_shape:?} -> {new_shape:?}");
    }
    let ret = mop_shape(ctx, Ops::Reshape, x, &new_shape);
    let ret_shape = ctx.shape(ret).unwrap().unwrap_or_default();
    if ret_shape == self_shape {
        x
    } else {
        ret
    }
}

/// `MovementMixin.expand` (movement.py:150-162). `shape` = target with `-1`/None pre-resolved.
pub fn expand(ctx: &mut UOpCtx, x: UopId, shape: &[SVal]) -> UopId {
    let self_shape = ctx.shape(x).unwrap().unwrap_or_default();
    let (mine, theirs) = align_left(&self_shape, shape);
    let new_shape: Vec<SVal> = mine.iter().zip(theirs.iter()).map(|(f, t)| if *t == SVal::I(-1) { *f } else { *t }).collect();
    broadcast_to(ctx, x, &new_shape)
}

/// `MovementMixin._broadcast_to` (movement.py:129-149).
pub fn broadcast_to(ctx: &mut UOpCtx, x: UopId, new_shape: &[SVal]) -> UopId {
    let self_shape = ctx.shape(x).unwrap().unwrap_or_default();
    if self_shape.iter().copied().eq(new_shape.iter().copied()) {
        return x;
    }
    assert!(self_shape.len() <= new_shape.len(), "TGERR: cannot broadcast tensor to fewer dimensions");
    // first unsqueeze left with 1s
    let (shape, _) = align_left(&self_shape, new_shape);
    // for each dimension, either dim is 1, or it does not change
    for (s, ns) in shape.iter().zip(new_shape.iter()) {
        let same = s == ns;
        let is_one = matches!(s, SVal::I(1));
        assert!(same || is_one, "TGERR: cannot broadcast {self_shape:?} to {new_shape:?}");
    }
    let n_left = new_shape.len() - self_shape.len();
    let axes = broadcast_axes(ctx, &shape, new_shape);
    let expand_at: Vec<usize> = axes.into_iter().filter(|&i| i >= n_left).collect();
    let kept: Vec<usize> = (0..self_shape.len()).filter(|i| !expand_at.contains(i)).collect();
    let squeezed = reshape_internal(ctx, x, &kept.iter().map(|&i| self_shape[i]).collect::<Vec<_>>());
    let mut expand_shape: Vec<SVal> = new_shape[..n_left].to_vec();
    expand_shape.extend(expand_at.iter().map(|&i| new_shape[n_left + i]));
    let expanded = mop_shape(ctx, Ops::Expand, squeezed, &expand_shape);
    let perm: Vec<usize> = (0..n_left)
        .chain(expand_at.iter().map(|&i| {
            n_left
                + expand_at
                    .iter()
                    .position(|&j| j == i)
                    .unwrap()
        }))
        .chain(kept.iter().map(|&i| n_left + expand_at.len() + kept.iter().position(|&j| j == i).unwrap()))
        .collect();
    permute_internal(ctx, expanded, &perm)
}

/// `MovementMixin.pad` (movement.py:187-192): `arg` = (lo, hi) per dim or None → caller
/// resolves None to (0, 0).
pub fn pad(ctx: &mut UOpCtx, x: UopId, arg: &[(SVal, SVal)]) -> UopId {
    let self_shape = ctx.shape(x).unwrap().unwrap_or_default();
    assert_eq!(self_shape.len(), arg.len(), "TGERR: ndim != len(arg)");
    let pairs: Vec<(SVal, SVal)> = arg
        .iter()
        .zip(self_shape.iter())
        .map(|((lo, hi), s)| {
            let lhi = ctx.sval_add(*lo, *hi);
            let end = ctx.sval_add(*s, lhi);
            (*lo, end)
        })
        .collect();
    let ret = mop_pairs(ctx, Ops::Pad, x, &pairs);
    let ret_shape = ctx.shape(ret).unwrap().unwrap_or_default();
    if ret_shape == self_shape {
        x
    } else {
        ret
    }
}

/// `MovementMixin.shrink` (movement.py:193-214): `arg` = (start, end) per dim.
pub fn shrink(ctx: &mut UOpCtx, x: UopId, arg: &[(SVal, SVal)]) -> UopId {
    let self_shape = ctx.shape(x).unwrap().unwrap_or_default();
    assert_eq!(self_shape.len(), arg.len(), "TGERR: ndim != len(arg)");
    let pairs: Vec<(SVal, SVal)> = arg
        .iter()
        .zip(self_shape.iter())
        .map(|((a, b), s)| (*a, ctx.sval_sub(*b, *a)))
        .collect();
    let ret = mop_pairs(ctx, Ops::Shrink, x, &pairs);
    let ret_shape = ctx.shape(ret).unwrap().unwrap_or_default();
    if ret_shape == self_shape {
        x
    } else {
        ret
    }
}

/// `MovementMixin.permute` (movement.py:215-233).
pub fn permute(ctx: &mut UOpCtx, x: UopId, order: &[i64]) -> UopId {
    let n = ctx.shape(x).unwrap().unwrap_or_default().len();
    let order_arg: Vec<i64> = order.iter().map(|&d| resolve_dim(n, d, false) as i64).collect();
    permute_internal(ctx, x, &order_arg.iter().map(|&d| d as usize).collect::<Vec<_>>())
}

pub(crate) fn permute_internal(ctx: &mut UOpCtx, x: UopId, order: &[usize]) -> UopId {
    let n = ctx.shape(x).unwrap().unwrap_or_default().len();
    let mut sorted_order = order.to_vec();
    sorted_order.sort_unstable();
    assert!(
        sorted_order == (0..n).collect::<Vec<_>>(),
        "TGERR: order is not a valid permutation, getting {order:?}"
    );
    if order.iter().copied().eq(0..n) {
        return x;
    }
    let arg = Arg::Seq { tuple: true, items: order.iter().map(|&i| Arg::Int(i as i64)).collect() };
    ctx.uop(Ops::Permute, None, vec![x], arg)
}

/// `MovementMixin.flip` (movement.py:234-256): `axis` must be pre-resolved to non-negative dims.
pub fn flip(ctx: &mut UOpCtx, x: UopId, axis: &[usize]) -> UopId {
    let shape = ctx.shape(x).unwrap().unwrap_or_default();
    assert!(axis.iter().all(|&a| a < shape.len()), "TGERR: flip args must be axis ints");
    assert!(axis.len() == {
        let mut v = axis.to_vec();
        v.sort_unstable();
        v.dedup();
        v.len()
    }, "TGERR: dim can appear at most once");
    let flip_arg: Vec<bool> = (0..shape.len()).map(|i| axis.contains(&i)).collect();
    if !flip_arg.iter().any(|&b| b) {
        return x;
    }
    let arg = Arg::Seq { tuple: true, items: flip_arg.into_iter().map(Arg::Bool).collect() };
    ctx.uop(Ops::Flip, None, vec![x], arg)
}

/// `MovementMixin.stack(*args, dim=0)` (movement.py:257-276).
pub fn stack_tensors(ctx: &mut UOpCtx, tensors: &[UopId], dim: i64) -> UopId {
    let n = ctx.shape(tensors[0]).unwrap().unwrap_or_default().len();
    let dim = resolve_dim(n, dim, true);
    let first = ctx.shape(tensors[0]).unwrap().unwrap_or_default();
    for t in tensors {
        assert_eq!(ctx.shape(*t).unwrap().unwrap_or_default(), first, "TGERR: all shapes must match for stack");
    }
    let dt = ctx.dtype_from_uop(Ops::Stack, tensors, &Arg::None).unwrap();
    let srcs = tensors
        .iter()
        .map(|&u| {
            if ctx.is_invalid(ctx.base(u)) {
                u
            } else if ctx.op_of(u) == Ops::Const {
                ctx.const_like_dt(u, ctx.val(u), Some(dt))
            } else {
                ctx.cast(u, dt)
            }
        })
        .collect();
    let ret = ctx.uop(Ops::Stack, None, srcs, Arg::None);
    if dim == 0 {
        ret
    } else {
        let nd = ctx.shape(ret).unwrap().unwrap_or_default().len();
        let perm: Vec<usize> = (1..=dim).chain(std::iter::once(0)).chain((dim + 1)..nd).collect();
        permute_internal(ctx, ret, &perm)
    }
}

/// `MovementMixin.shrink_to` (movement.py:277-279).
pub fn shrink_to(ctx: &mut UOpCtx, x: UopId, shape: &[SVal]) -> UopId {
    let self_shape = ctx.shape(x).unwrap().unwrap_or_default();
    let arg: Vec<(SVal, SVal)> = shape
        .iter()
        .zip(self_shape.iter())
        .map(|(ns, s)| match ns {
            SVal::U(_) => (SVal::I(0), *s),
            SVal::I(v) if *v < 0 => (SVal::I(0), *s),
            SVal::I(v) => (SVal::I(0), SVal::I(*v)),
        })
        .collect();
    shrink(ctx, x, &arg)
}

/// `MovementMixin.pad_to` (movement.py:280-283).
pub fn pad_to(ctx: &mut UOpCtx, x: UopId, shape: &[SVal]) -> UopId {
    let self_shape = ctx.shape(x).unwrap().unwrap_or_default();
    let arg: Vec<(SVal, SVal)> = shape
        .iter()
        .zip(self_shape.iter())
        .map(|(ns, s)| match ns {
            SVal::U(_) => (SVal::I(0), *s),
            SVal::I(v) if *v < 0 => (SVal::I(0), *s),
            SVal::I(v) => (SVal::I(0), SVal::I(*v)),
        })
        .collect();
    let ret = mop_pairs(ctx, Ops::Pad, x, &arg);
    let ret_shape = ctx.shape(ret).unwrap().unwrap_or_default();
    if ret_shape == self_shape {
        x
    } else {
        ret
    }
}

/// `MovementMixin.squeeze(dim=None)` (movement.py:288-308).
pub fn squeeze(ctx: &mut UOpCtx, x: UopId, dim: Option<i64>) -> UopId {
    let shape = ctx.shape(x).unwrap().unwrap_or_default();
    match dim {
        None => {
            let kept: Vec<SVal> = shape.iter().filter(|s| **s != SVal::I(1)).copied().collect();
            reshape_internal(ctx, x, &kept)
        }
        Some(d) => {
            let dim = resolve_dim(shape.len(), d, false);
            if shape.is_empty() || shape[dim] != SVal::I(1) {
                x
            } else {
                let kept: Vec<SVal> = shape.iter().enumerate().filter(|(i, _)| *i != dim).map(|(_, s)| *s).collect();
                reshape_internal(ctx, x, &kept)
            }
        }
    }
}

/// `MovementMixin.unsqueeze(dim)` (movement.py:309-324).
pub fn unsqueeze(ctx: &mut UOpCtx, x: UopId, dim: i64) -> UopId {
    let shape = ctx.shape(x).unwrap().unwrap_or_default();
    let dim = resolve_dim(shape.len(), dim, true);
    let mut new_shape: Vec<SVal> = shape[..dim].to_vec();
    new_shape.push(SVal::I(1));
    new_shape.extend_from_slice(&shape[dim..]);
    reshape_internal(ctx, x, &new_shape)
}

/// `MovementMixin.transpose(dim0=1, dim1=0)` (movement.py:329-345).
pub fn transpose(ctx: &mut UOpCtx, x: UopId, dim0: i64, dim1: i64) -> UopId {
    let n = ctx.shape(x).unwrap().unwrap_or_default().len();
    let mut order: Vec<usize> = (0..n).collect();
    let d0 = resolve_dim(n, dim0, false);
    let d1 = resolve_dim(n, dim1, false);
    order.swap(d0, d1);
    permute_internal(ctx, x, &order)
}

/// `MovementMixin.flatten(start_dim=0, end_dim=-1)` (movement.py:346-361).
pub fn flatten(ctx: &mut UOpCtx, x: UopId, start_dim: i64, end_dim: i64) -> UopId {
    let shape = ctx.shape(x).unwrap().unwrap_or_default();
    let n = shape.len() as i64;
    let start = if start_dim < 0 { start_dim + n } else { start_dim };
    let end = if end_dim < 0 { end_dim + n } else { end_dim };
    let mid = prod_sval(ctx, &shape[start as usize..=(end as usize)]);
    let mut new_shape: Vec<SVal> = shape[..start as usize].to_vec();
    new_shape.push(mid);
    new_shape.extend_from_slice(&shape[end as usize + 1..]);
    reshape_internal(ctx, x, &new_shape)
}

/// `MovementMixin.unflatten(dim, sizes)` (movement.py:362-378).
pub fn unflatten(ctx: &mut UOpCtx, x: UopId, dim: i64, sizes: &[i64]) -> UopId {
    let shape = ctx.shape(x).unwrap().unwrap_or_default();
    let n = shape.len() as i64;
    let dim = if dim < 0 { dim + n } else { dim };
    let itemsize = sizes.iter().filter(|&&s| s != -1).map(|s| *s as i128).product::<i128>();
    let mut new_shape: Vec<SVal> = shape[..dim as usize].to_vec();
    for &s in sizes {
        if s == -1 {
            new_shape.push(ctx.sval_div_floor(
                shape[dim as usize],
                SVal::I(if itemsize == 0 { 1 } else { itemsize }),
            ));
        } else {
            new_shape.push(SVal::I(s as i128));
        }
    }
    new_shape.extend_from_slice(&shape[dim as usize + 1..]);
    reshape_internal(ctx, x, &new_shape)
}

// **** basic getitem (slicing): `_normalize_indices` + `_parse_view_index` + `_apply_view_ops` ****

/// One parsed view index: Python's `{"size", "boundary", "stride", "collapse_dim"}`.
#[derive(Clone, Debug)]
pub struct ViewIndex {
    pub size: SVal,
    pub boundary: (SVal, SVal),
    pub stride: i64,
    pub collapse_dim: bool,
}

/// Python `slice(start, stop, step)` with int bounds (`Ix::Range` pre-resolved).
#[derive(Clone, Copy, Debug)]
pub struct SliceSpec {
    pub start: Option<i64>,
    pub stop: Option<i64>,
    pub step: i64,
}

// `resolve(x, default)` lives in the engine: `UOpCtx::resolve_ord` (CmpOrd).
/// CPython `slice.indices(length)` with raw Option bounds (None resolved per
/// CPython: start -> lower/upper by sign of step, stop -> upper/lower).
pub fn slice_indices(length: i64, start: Option<i64>, stop: Option<i64>, step: i64) -> (i64, i64, i64) {
    let (lower, upper) = if step > 0 { (0, length) } else { (-1, length - 1) };
    // CPython: the `+= length` adjustment applies ONLY to user-provided
    // negatives; None-resolved bounds go straight to the clamp.
    let mut start = match start {
        None => {
            if step > 0 {
                lower
            } else {
                upper
            }
        }
        Some(s) => {
            let s = if s < 0 { s + length } else { s };
            if s < lower {
                lower
            } else if s > upper {
                upper
            } else {
                s
            }
        }
    };
    let _ = &mut start;
    let mut stop = match stop {
        None => {
            if step > 0 {
                upper
            } else {
                lower
            }
        }
        Some(s) => {
            let s = if s < 0 { s + length } else { s };
            if s < lower {
                lower
            } else if s > upper {
                upper
            } else {
                s
            }
        }
    };
    (start, stop, step)
}

/// `MovementMixin._parse_view_index` (movement.py:87-115) for the slice case.
/// `vmax` = `int(size.vmax)` when `size` is symbolic (CPython indices target).
pub fn parse_view_index(ctx: &mut UOpCtx, size: SVal, vmax: Option<i64>, spec: &SliceSpec) -> ViewIndex {
    assert!(spec.step != 0, "TGERR: slice cannot have 0 as step");
    // start, stop = 0 if None else start, size if None else stop
    let start0 = spec.start.unwrap_or(0);
    let size_sym = !matches!(size, SVal::I(_));
    // negative int bounds resolve against the (possibly symbolic) size, like
    // slice.indices: `if isinstance(start, int) and start < 0: start = start + size`
    let start_v: SVal = if start0 < 0 && size_sym {
        let sum = ctx.sval_add(size, SVal::I(start0 as i128));
        sum
    } else {
        SVal::I(start0 as i128)
    };
    let stop_v: SVal = match spec.stop {
        Some(s) if s < 0 && size_sym => ctx.sval_add(size, SVal::I(s as i128)),
        Some(s) => SVal::I(s as i128),
        None => size,
    };
    // all-int bounds -> clamp/stride via Python slice.indices (against vmax when symbolic)
    if matches!(start_v, SVal::I(_)) && matches!(stop_v, SVal::I(_)) {
        let n = vmax.unwrap_or_else(|| size.as_i128_or_panic() as i64);
        // the all-int path calls the RAW slice's indices() — None bounds are
        // resolved inside CPython's clamp, not defaulted to the size first
        let (b0, b1, stride) = slice_indices(n, spec.start, spec.stop, spec.step);
        let (b0, b1) = if stride * (b1 - b0) < 0 {
            (0, 0)
        } else if stride < 0 {
            (b1 + 1, b0 + 1)
        } else {
            (b0, b1)
        };
        ViewIndex {
            size: SVal::I(ceildiv(b1 as i128 - b0 as i128, stride.abs() as i128)),
            boundary: (SVal::I(b0 as i128), SVal::I(b1 as i128)),
            stride,
            collapse_dim: false,
        }
    } else {
        // symbolic: only step == 1 with stop-start >= 0 is supported
        assert!(spec.step == 1, "TGERR: symbolic slice {spec:?} is not supported");
        assert!(
            (|| {
                let diff = ctx.sval_sub(stop_v, start_v);
                ctx.resolve_ord(diff, crate::uop::ctx::CmpOrd::Ge, SVal::I(0), false)
            })(),
            "TGERR: symbolic slice {spec:?} is not supported"
        );
        let size_v = ctx.sval_sub(stop_v, start_v);
        ViewIndex { size: size_v, boundary: (start_v, stop_v), stride: 1, collapse_dim: false }
    }
}

/// `MovementMixin._apply_view_ops` (movement.py:116-128): shrink + flip + stride.
pub fn apply_view_ops(ctx: &mut UOpCtx, x: UopId, mops: &[ViewIndex]) -> UopId {
    // flip negative strides
    let flip_axes: Vec<usize> = mops.iter().enumerate().filter(|(_, m)| m.stride < 0).map(|(i, _)| i).collect();
    let boundaries: Vec<(SVal, SVal)> = mops.iter().map(|m| m.boundary).collect();
    let mut x = shrink(ctx, x, &boundaries);
    x = flip(ctx, x, &flip_axes);
    let strides: Vec<i64> = mops.iter().map(|m| m.stride.abs()).collect();
    // apply stride
    if strides.iter().any(|&st| st != 1) {
        let shape = ctx.shape(x).unwrap().unwrap_or_default();
        assert!(shape.iter().all(|s| matches!(s, SVal::I(_))), "TGERR: symbolic shape not supported");
        let ints: Vec<i64> = shape.iter().map(|s| s.as_i128_or_panic() as i64).collect();
        let padded: Vec<SVal> = ints.iter().zip(strides.iter()).map(|(&s, &st)| SVal::I(round_up(s as i128, st as i128))).collect();
        x = pad_to(ctx, x, &padded);
        let shape2 = ctx.shape(x).unwrap().unwrap_or_default();
        // reshape(flatten((s // st, st) for s, st in zip(shape, strides)))
        let mut new_shape: Vec<SVal> = Vec::with_capacity(shape2.len() * 2);
        for (s, st) in shape2.iter().zip(strides.iter()) {
            new_shape.push(ctx.sval_div_floor(*s, SVal::I(*st as i128)));
            new_shape.push(SVal::I(*st as i128));
        }
        x = reshape_internal(ctx, x, &new_shape);
        let shape3 = ctx.shape(x).unwrap().unwrap_or_default();
        // shrink_to(flatten((s, 1) for s in shape[::2])).reshape(shape[::2])
        let shrink_shape: Vec<SVal> = (0..shape3.len()).step_by(2).flat_map(|i| [shape3[i], SVal::I(1)]).collect();
        x = shrink_to(ctx, x, &shrink_shape);
        let kept: Vec<SVal> = shape3.iter().step_by(2).copied().collect();
        x = reshape_internal(ctx, x, &kept);
    }
    x
}

/// `Tensor._getitem` (op.py:74-163), basic path: int/None/slice/Ellipsis indices.
/// Advanced (tensor) indexing is a follow-up.
pub fn getitem(ctx: &mut UOpCtx, x: UopId, ixs: &[Ix]) -> UopId {
    use super::ix::Ix;
    let shape = ctx.shape(x).unwrap().unwrap_or_default();
    let ndim = shape.len();
    // `_normalize_indices`: single ellipsis, fill, too-many check
    let ell_count = ixs.iter().filter(|i| matches!(i, Ix::Ellipsis)).count();
    assert!(ell_count <= 1, "TGERR: IndexError indices can only have a single ellipsis");
    let none_count = ixs.iter().filter(|i| matches!(i, Ix::NewAxis)).count();
    let num_real = ixs.len() - ell_count - none_count;
    assert!(num_real <= ndim, "TGERR: IndexError too many indices ({num_real}) for {ndim}D");
    let fill_at = ixs.iter().position(|i| matches!(i, Ix::Ellipsis));
    let fills = ndim - num_real;
    let mut norm: Vec<Ix> = Vec::with_capacity(ixs.len() + fills);
    for (i, ix) in ixs.iter().enumerate() {
        if Some(i) == fill_at {
            norm.extend(std::iter::repeat(Ix::All).take(fills));
            continue;
        }
        norm.push(*ix);
    }
    if fill_at.is_none() {
        norm.extend(std::iter::repeat(Ix::All).take(fills));
    }

    // parse loop (op.py:80-97)
    let mut parsed: Vec<ViewIndex> = Vec::with_capacity(norm.len());
    let mut mops_parsed: Vec<ViewIndex> = Vec::with_capacity(norm.len());
    let mut dim = 0usize;
    for ix in &norm {
        let size = match ix {
            Ix::NewAxis => SVal::I(1),
            _ => shape[dim],
        };
        let vi = match ix {
            Ix::NewAxis => ViewIndex { size: SVal::I(1), boundary: (SVal::I(0), SVal::I(1)), stride: 1, collapse_dim: false },
            Ix::Index(i) => {
                // bounds check: resolve(index >= size) or resolve(index < -size) -> IndexError
                let size_i = if size_sym_or(&size) { None } else { Some(size.as_i128_or_panic() as i64) };
                let neg_size = ctx.sval_neg(size);
                let iv = SVal::I(*i as i128);
                let ge = ctx.resolve_ord(iv, crate::uop::ctx::CmpOrd::Ge, size, false);
                let lt = ctx.resolve_ord(iv, crate::uop::ctx::CmpOrd::Lt, neg_size, false);
                assert!(!(ge || lt), "TGERR: IndexError index {i} is out of bounds with size {size_i:?}");
                let b = if *i >= 0 { *i } else { *i + size_i.unwrap_or(0) };
                ViewIndex { size, boundary: (SVal::I(b as i128), SVal::I((b + 1) as i128)), stride: 1, collapse_dim: true }
            }
            Ix::All => {
                let vmax = sval_vmax_i(ctx, size);
                parse_view_index(ctx, size, vmax, &SliceSpec { start: None, stop: None, step: 1 })
            }
            Ix::Range { start, stop, step } => {
                let vmax = sval_vmax_i(ctx, size);
                parse_view_index(ctx, size, vmax, &SliceSpec { start: *start, stop: *stop, step: *step })
            }
            Ix::Ellipsis => unreachable!(),
        };
        if !matches!(ix, Ix::NewAxis) {
            mops_parsed.push(vi.clone());
            dim += 1;
        }
        parsed.push(vi);
    }

    // apply view ops on the non-None dims (`p["index"] is not None`), then dim
    // injection (NewAxis) + collapse (int)
    let mops: Vec<ViewIndex> = std::mem::take(&mut mops_parsed);
    let x = apply_view_ops(ctx, x, &mops);
    let sizes: Vec<SVal> = parsed.iter().filter(|p| !p.collapse_dim).map(|p| p.size).collect();
    reshape_internal(ctx, x, &sizes)
}

fn size_sym_or(s: &SVal) -> bool {
    !matches!(s, SVal::I(_))
}

fn sval_vmax_i(ctx: &mut UOpCtx, s: SVal) -> Option<i64> {
    match s {
        SVal::I(v) => Some(v as i64),
        SVal::U(u) => ctx.vmax(u).ok().and_then(|v| v.as_i64()),
    }
}
