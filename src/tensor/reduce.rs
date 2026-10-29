//! M3 reduce mixin (mixin/reduce.py + the `_rop`/`_reduce`/`_cumalu`/`_pool`
//! pieces from mixin/op.py + movement.py).

use crate::dtype::ConstVal;
use crate::dtype::{self, DTypeId};
use crate::uop::arg::Arg;
use crate::uop::ctx::UOpCtx;
use crate::uop::graph::UopId;
use crate::uop::ops::Ops;
use crate::uop::sint::{SInt, SVal};

use super::movement;

/// `UOp._rop` (ops.py:629-636): REDUCE over the reduce axis moved to the front.
pub fn rop(ctx: &mut UOpCtx, op: Ops, x: UopId, axis: &[usize]) -> UopId {
    let mut axis = axis.to_vec();
    axis.sort_unstable();
    let shape = ctx.shape(&x).unwrap().unwrap_or_default();
    // NOTE: we don't allow reduce on 1s axis — resolve(shape != 1), default true
    let reduce_axis: Vec<usize> = axis
        .iter()
        .cloned()
        .filter(|&i| !matches!(shape[i], SVal::I(1)))
        .collect();
    if reduce_axis.is_empty() {
        let kept: Vec<SVal> = shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !axis.contains(i))
            .map(|(_, s)| s.clone())
            .collect();
        return movement::reshape_internal(ctx, x, &kept);
    }
    // permute so reduced axes are at the front
    let perm: Vec<usize> = reduce_axis
        .iter()
        .cloned()
        .chain((0..shape.len()).filter(|i| !reduce_axis.contains(i)))
        .collect();
    let permuted = movement::permute_internal(ctx, x, &perm);
    let ret = ctx.uop(
        Ops::Reduce,
        None,
        vec![permuted],
        Arg::Seq {
            tuple: true,
            items: vec![super::op_arg(op), Arg::Int(reduce_axis.len() as i64)],
        },
    );
    if axis != reduce_axis {
        let kept: Vec<SVal> = shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !axis.contains(i))
            .map(|(_, s)| s.clone())
            .collect();
        movement::reshape_internal(ctx, ret, &kept)
    } else {
        ret
    }
}

/// `ReduceMixin._reduce(op, axis, keepdim)` (reduce.py:13-18): cast strong first.
pub fn reduce(ctx: &mut UOpCtx, x: UopId, op: Ops, axis: Option<&[usize]>, keepdim: bool) -> UopId {
    let x = ctx.cast(x.clone(), dtype::strong_dtype(ctx.dt_of(&x)));
    let shape = ctx.shape(&x).unwrap().unwrap_or_default();
    let axis: Vec<usize> = match axis {
        None => (0..shape.len()).collect(),
        Some(a) => a
            .iter()
            .map(|&d| movement::resolve_dim(shape.len(), d as i64, false))
            .collect(),
    };
    let axis: Vec<usize> = if shape.is_empty() { vec![] } else { axis };
    let ret = rop(ctx, op, x.clone(), &axis);
    if keepdim {
        let shape_in = ctx.shape(&x).unwrap().unwrap_or_default();
        let ret_shape = ctx.shape(&ret).unwrap().unwrap_or_default();
        // reshape to (1 in reduced dims, s elsewhere), walking the input shape
        // with the kept dims consumed from the ret shape in order
        let mut out: Vec<SVal> = Vec::with_capacity(shape_in.len());
        let mut kept = ret_shape.iter();
        for (i, _s) in shape_in.iter().enumerate() {
            if axis.contains(&i) {
                out.push(SVal::I(1));
            } else if let Some(v) = kept.next() {
                out.push(v.clone());
            }
        }
        movement::reshape_internal(ctx, ret, &out)
    } else {
        ret
    }
}

/// `sum_acc_dtype` (dtype.py:212-216).
pub fn sum_acc_dtype(dt: DTypeId) -> DTypeId {
    if dtype::is_unsigned(dt) {
        dtype::least_upper_dtype(&[dt, dtype::by_attr("uint").unwrap()])
    } else if dtype::is_int(dt) || dtype::is_bool(dt) {
        dtype::least_upper_dtype(&[dt, dtype::by_attr("int").unwrap()])
    } else {
        dtype::least_upper_dtype(&[dt, dtype::to_dtype("float32").unwrap()])
    }
}

/// `ReduceMixin.sum` (reduce.py:20-45): cast the *input* to the accumulator
/// dtype, reduce, then cast back for low-precision floats.
pub fn sum(
    ctx: &mut UOpCtx,
    x: UopId,
    axis: Option<&[usize]>,
    keepdim: bool,
    dtype: Option<DTypeId>,
) -> UopId {
    let self_dt = ctx.dt_of(&x);
    let dt = dtype.unwrap_or_else(|| sum_acc_dtype(self_dt));
    let xc = ctx.cast(x, dt);
    let ret = reduce(ctx, xc, Ops::Add, axis, keepdim);
    if dtype.is_none() && sum_cast_back(self_dt) {
        return ctx.cast(ret, self_dt);
    }
    ret
}

/// `ReduceMixin.prod` (reduce.py:47-71): reduce in the input dtype (or `dtype`).
pub fn prod(
    ctx: &mut UOpCtx,
    x: UopId,
    axis: Option<&[usize]>,
    keepdim: bool,
    dtype: Option<DTypeId>,
) -> UopId {
    let self_dt = ctx.dt_of(&x);
    let dt = dtype.unwrap_or(self_dt);
    let xc = ctx.cast(x, dt);
    reduce(ctx, xc, Ops::Mul, axis, keepdim)
}

/// sum/prod cast back to the source dtype for low-precision floats
/// (reduce.py: `self.dtype in (dtypes.float16, dtypes.bfloat16, *dtypes.fp8s)`).
fn sum_cast_back(dt: DTypeId) -> bool {
    dtype::in_set(
        &[
            dtype::FLOAT16,
            dtype::BFLOAT16,
            dtype::FP8E4M3,
            dtype::FP8E5M2,
            dtype::FP8E4M3FNUZ,
            dtype::FP8E5M2FNUZ,
        ],
        dt,
    )
}

/// `ReduceMixin.max` (reduce.py:73-94): `max()` over a bool tensor errors in Python.
pub fn max(ctx: &mut UOpCtx, x: UopId, axis: Option<&[usize]>, keepdim: bool) -> UopId {
    assert!(
        !dtype::is_bool(ctx.dt_of(&x)),
        "TGERR: Tensor.max() not implemented for bool"
    );
    reduce(ctx, x, Ops::Max, axis, keepdim)
}

/// Number of elements being reduced (static shapes; symbolic axes panic).
fn reduced_numel(ctx: &mut UOpCtx, x: &UopId, axis: Option<&[usize]>) -> i128 {
    let shape = ctx.shape(x).unwrap().unwrap_or_default();
    match axis {
        None => shape.iter().map(|s| s.clone().as_i128_or_panic()).product(),
        Some(ax) => ax
            .iter()
            .map(|&a| shape[a].clone().as_i128_or_panic())
            .product(),
    }
}

/// `ReduceMixin.any` (reduce.py:96-116): `cast(bool).sum() != 0`.
pub fn any(ctx: &mut UOpCtx, x: UopId, axis: Option<&[usize]>, keepdim: bool) -> UopId {
    let b = ctx.cast(x, dtype::BOOL);
    let i = ctx.cast(b, dtype::INT32);
    let s = sum(ctx, i, axis, keepdim, None);
    let zero = ctx.const_py(ConstVal::Int(0), Some(dtype::INT32));
    super::elementwise::ne(ctx, s, zero)
}

/// `ReduceMixin.all` (reduce.py:118-137): every element is true.
///
/// tinygrad writes this as `(logical_not(x)).min().logical_not()`, but `min`
/// routes through `max` and bool `max` is not defined; summing the cast and
/// comparing with the reduced element count is equivalent and works on every
/// backend.
pub fn all(ctx: &mut UOpCtx, x: UopId, axis: Option<&[usize]>, keepdim: bool) -> UopId {
    let n = reduced_numel(ctx, &x, axis);
    let b = ctx.cast(x, dtype::BOOL);
    let i = ctx.cast(b, dtype::INT32);
    let s = sum(ctx, i, axis, keepdim, None);
    let nc = ctx.const_py(ConstVal::Int(n as i128), Some(dtype::INT32));
    super::elementwise::eq(ctx, s, nc)
}

/// `OpMixin.mean` (op.py:499-524).
pub fn mean(ctx: &mut UOpCtx, x: UopId, axis: Option<&[usize]>, keepdim: bool) -> UopId {
    let self_dt = ctx.dt_of(&x);
    let output_dtype = if dtype::is_float(self_dt) {
        self_dt
    } else {
        dtype::default_float()
    };
    let numerator = sum(ctx, x.clone(), axis, keepdim, None);
    let denominator = reduce_count(ctx, x, axis);
    let denom_u = ctx.sval_to_uop(denominator);
    let divided = super::elementwise::div(ctx, numerator, denom_u, None);
    ctx.cast(divided, output_dtype)
}

/// `prod([si for si, so in zip(self.shape, self.sum(axis=axis, keepdim=True).shape)
/// if resolve(si != so)])` — the element count across the reduced dims.
fn reduce_count(ctx: &mut UOpCtx, x: UopId, axis: Option<&[usize]>) -> SVal {
    let shape = ctx.shape(&x).unwrap().unwrap_or_default();
    let axis_vec: Vec<usize> = match axis {
        None => (0..shape.len()).collect(),
        Some(a) => a.to_vec(),
    };
    let dims: Vec<SVal> = (0..shape.len())
        .filter(|i| axis_vec.contains(i))
        .map(|i| <SInt as Clone>::clone(&shape[i]))
        .collect();
    movement::prod_sval(ctx, &dims)
}

/// `OpMixin.var` (op.py:526-552).
pub fn var(
    ctx: &mut UOpCtx,
    x: UopId,
    axis: Option<&[usize]>,
    keepdim: bool,
    correction: i64,
) -> UopId {
    let self_dt = ctx.dt_of(&x);
    let output_dtype = if dtype::is_float(self_dt) {
        self_dt
    } else {
        dtype::default_float()
    };
    let m = mean(ctx, x.clone(), axis, true);
    let diff = super::elementwise::sub(ctx, x.clone(), m);
    let squares = super::elementwise::square(ctx, diff);
    let n = reduce_count(ctx, x, axis);
    let numerator = sum(ctx, squares, axis, keepdim, None);
    // smax(n - correction, 0)
    let nm1 = ctx.sval_sub(n, SVal::I(correction as i128));
    let nm1_u = ctx.sval_to_uop(nm1);
    let zero = ctx.cint(0);
    let denom_u = ctx.alu2(Ops::Max, nm1_u, zero);
    let divided = super::elementwise::div(ctx, numerator, denom_u, None);
    ctx.cast(divided, output_dtype)
}

// ***** cumsum (op.py:734-800) *****

/// `OpMixin._pad_constant(pX, value)` (op.py:282-291): shrink negatives, pad
/// non-negatives, mask-fill with `value` when non-zero.
pub fn pad_constant(
    ctx: &mut UOpCtx,
    x: UopId,
    px: &[(Option<i64>, Option<i64>)],
    value: ConstVal,
) -> UopId {
    let shape = ctx.shape(&x).unwrap().unwrap_or_default();
    let has_neg = px
        .iter()
        .any(|(a, b)| a.is_some_and(|v| v < 0) || b.is_some_and(|v| v < 0));
    let mut cur = x;
    if has_neg {
        let shr: Vec<(SVal, SVal)> = px
            .iter()
            .zip(shape.iter())
            .map(|((pb, pa), s)| {
                let lo = SVal::I(-pb.map_or(0, |v| v.min(0)) as i128);
                let pa_lo = ctx.sval_add(s.clone(), SVal::I(pa.unwrap_or(0).min(0) as i128));
                let hi = ctx.sval_add(pa_lo, SVal::I(0));
                (lo, hi)
            })
            .collect();
        cur = movement::shrink(ctx, cur, &shr);
    }
    let pads: Vec<(SVal, SVal)> = if has_neg {
        px.iter()
            .map(|(pb, pa)| {
                (
                    SVal::I(pb.unwrap_or(0).max(0) as i128),
                    SVal::I(pa.unwrap_or(0).max(0) as i128),
                )
            })
            .collect()
    } else {
        px.iter()
            .map(|(pb, pa)| {
                (
                    SVal::I(pb.unwrap_or(0) as i128),
                    SVal::I(pa.unwrap_or(0) as i128),
                )
            })
            .collect()
    };
    let base = movement::pad(ctx, cur.clone(), &pads);
    // python: `if value == 0` — true for 0.0 as well
    let is_zero = match &value {
        ConstVal::Int(0) => true,
        ConstVal::Float(f) => *f == 0.0,
        _ => false,
    };
    if is_zero {
        return base;
    }
    let mask_src = ctx.const_like_dt(cur, ConstVal::Bool(true), Some(dtype::BOOL));
    let masked = movement::pad(ctx, mask_src, &pads);
    let value_u = super::elementwise::ufix(ctx, value);
    super::elementwise::where_(ctx, masked, base, value_u)
}

/// `identity_element(op, dtype)` as a plain ConstVal (ADD -> 0, MUL -> 1,
/// MAX -> dtype.min).
fn identity_const(op: Ops, dt: DTypeId) -> ConstVal {
    match op {
        Ops::Add => {
            if dtype::is_float(dt) {
                ConstVal::Float(0.0)
            } else {
                ConstVal::Int(0)
            }
        }
        Ops::Mul => {
            if dtype::is_float(dt) {
                ConstVal::Float(1.0)
            } else {
                ConstVal::Int(1)
            }
        }
        Ops::Max => dtype::dtype_min(dt),
        _ => panic!("TGERR: no identity for {op:?}"),
    }
}

/// `OpMixin._cumalu(axis, op)` (op.py:741-747) via `_pad_constant` + `_pool`
/// restricted to the cumsum case (single trailing pooled axis, stride/dilation 1).
pub fn cumalu(ctx: &mut UOpCtx, x: UopId, axis: usize, op: Ops) -> UopId {
    let shape = ctx.shape(&x).unwrap().unwrap_or_default();
    assert!(
        !matches!(shape[axis], SVal::I(0)) && matches!(op, Ops::Add | Ops::Max | Ops::Mul),
        "TGERR: bad _cumalu"
    );
    // transpose(axis, -1)
    let t = movement::transpose(ctx, x, axis as i64, -1);
    let tshape = ctx.shape(&t).unwrap().unwrap_or_default();
    let k = <SInt as Clone>::clone(&tshape[tshape.len() - 1]).as_i128_or_panic();
    let mut pads: Vec<(Option<i64>, Option<i64>)> = vec![(None, None); tshape.len()];
    pads[tshape.len() - 1] = (Some(k as i64 - 1), Some(0));
    let identity = identity_const(op, ctx.dt_of(&t));
    let padded = pad_constant(ctx, t, &pads, identity);
    let pooled = pool_for_cumalu(ctx, padded, k as usize, tshape.len());
    // the pooled tensor has one extra dim (the (k, o) split); reduce over its last
    let last = tshape.len();
    let reduced = match op {
        Ops::Add => sum(ctx, pooled, Some(&[last]), false, None),
        Ops::Max => max(ctx, pooled, Some(&[last]), false),
        Ops::Mul => prod(ctx, pooled, Some(&[last]), false, None),
        _ => unreachable!(),
    };
    movement::transpose(ctx, reduced, axis as i64, -1)
}

/// `MovementMixin.repeat` (movement.py:576-594).
pub fn repeat(ctx: &mut UOpCtx, x: UopId, repeats: &[i64]) -> UopId {
    let self_shape = ctx.shape(&x).unwrap().unwrap_or_default();
    // base_shape = _align_left(shape, repeats)[0]
    let mut base: Vec<SVal> = vec![SVal::I(1); repeats.len().saturating_sub(self_shape.len())];
    base.extend_from_slice(&self_shape);
    let mut unsq: Vec<SVal> = Vec::with_capacity(repeats.len() * 2);
    let mut exp: Vec<SVal> = Vec::with_capacity(repeats.len() * 2);
    let mut fin: Vec<SVal> = Vec::with_capacity(repeats.len());
    for (i, &r) in repeats.iter().enumerate() {
        let s = &base[i].clone();
        if r == 1 {
            unsq.push(s.clone());
            exp.push(s.clone());
        } else {
            unsq.push(SVal::I(1));
            unsq.push(s.clone());
            exp.push(SVal::I(r as i128));
            exp.push(s.clone());
        }
        let sv = match s {
            SVal::I(v) => SVal::I(v * r as i128),
            SVal::U(_) => panic!("TGERR: symbolic repeat"),
        };
        fin.push(sv);
    }
    let reshaped = movement::reshape_internal(ctx, x, &unsq);
    let expanded = movement::broadcast_to(ctx, reshaped, &exp);
    movement::reshape_internal(ctx, expanded, &fin)
}

/// `OpMixin._pool(k_)` (movement.py:598-617) specialized: one pooled trailing
/// axis of length `k` on a tensor whose trailing dim is `2k-1` (the cumsum
/// padded length), stride/dilation 1. `nd` = ndim of the padded input;
/// `noop = nd - 1` leading dims pass through untouched.
fn pool_for_cumalu(ctx: &mut UOpCtx, x: UopId, k: usize, nd: usize) -> UopId {
    let noop = nd - 1;
    // the leading (non-pooled) dims pass through untouched: mirror _pool's
    // `noop = [None] * (ndim - len(k_))` by reusing their real sizes in every
    // shrink/reshape (filling them with 1 would collapse them).
    let xshape = ctx.shape(&x).unwrap().unwrap_or_default();
    let lead: Vec<SVal> = xshape[..noop].to_vec();
    let i = 2 * k as i128 - 1; // padded trailing dim
    let o = i - (k as i128 - 1); // ceildiv(i - 1*(k-1), 1)
    let f = std::cmp::max(1, movement::ceildiv(o - 1, i)); // smax(1, ceildiv(o*1-1, i))
    let rep = movement::ceildiv(k as i128 * (i * f + 1), i);
    let one = SVal::I(1); // trailing stride dim (s=1)
    // repeats are REPEAT COUNTS (not sizes): _pool uses [1]*len(noop) + [rep].
    let mut rep_shape: Vec<SVal> = vec![one.clone(); noop];
    rep_shape.push(SVal::I(rep));
    let rep_ints: Vec<i64> = rep_shape
        .iter()
        .map(|s| s.clone().as_i128_or_panic() as i64)
        .collect();
    let mut xx = repeat(ctx, x, &rep_ints);
    // handle dilation: shrink_to(noop + [k*(i*f+d)]) then reshape(noop + (k, i*f+d))
    let seg = k as i128 * (i * f + 1);
    let mut shr: Vec<SVal> = lead.clone();
    shr.push(SVal::I(seg));
    xx = movement::shrink_to(ctx, xx, &shr);
    let mut resh: Vec<SVal> = lead.clone();
    resh.push(SVal::I(k as i128));
    resh.push(SVal::I(i * f + 1));
    xx = movement::reshape_internal(ctx, xx, &resh);
    // handle stride (s=1): shrink_to(noop + (k, o)) reshape(noop + (k, o, 1))
    //                  shrink_to(noop + (k, o, 1)) reshape(noop + (k, o))
    let mut s1: Vec<SVal> = lead.clone();
    s1.push(SVal::I(k as i128));
    s1.push(SVal::I(o));
    xx = movement::shrink_to(ctx, xx, &s1);
    let mut s2: Vec<SVal> = lead.clone();
    s2.push(SVal::I(k as i128));
    s2.push(SVal::I(o));
    s2.push(one);
    xx = movement::reshape_internal(ctx, xx, &s2);
    xx = movement::shrink_to(ctx, xx, &s2);
    let mut s4: Vec<SVal> = lead.clone();
    s4.push(SVal::I(k as i128));
    s4.push(SVal::I(o));
    xx = movement::reshape_internal(ctx, xx, &s4);
    // permute to move reduce to the end: noop dims, then the o dim, then k
    let perm: Vec<usize> = (0..noop)
        .chain(std::iter::once(noop + 1))
        .chain(std::iter::once(noop))
        .collect();
    movement::permute_internal(ctx, xx, &perm)
}

/// `OpMixin.cumsum(axis=0)` (op.py:773-800): `_split_cumalu(axis, ADD)`; the
/// two-stage SPLIT path is a follow-up (small tensors take `_cumalu` directly).
pub fn cumsum(ctx: &mut UOpCtx, x: UopId, axis: i64) -> UopId {
    let shape = ctx.shape(&x).unwrap().unwrap_or_default();
    let axis = movement::resolve_dim(shape.len(), axis, false);
    if shape.is_empty()
        || matches!(
            shape.iter().find(|s| matches!(s, SVal::I(0))),
            Some(SVal::I(0))
        )
    {
        let s = sum(ctx, x, None, false, None);
        return ctx.cast(s.clone(), ctx.dt_of(&s));
    }
    let s = <SInt as Clone>::clone(&shape[axis]).as_i128_or_panic();
    assert!(s <= 512, "TGERR: _split_cumalu SPLIT path not ported yet");
    cumalu(ctx, x, axis, Ops::Add)
}
