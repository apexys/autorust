//! Fast gather / scatter / segment paths for advanced indexing.
//!
//! The general `getitem` path expresses an advanced index as a one-hot mask
//! followed by `where` + `sum`. That is the semantic reference, and the codegen
//! gated-load lift already collapses the common 1-D case to one indirect
//! `LOAD`, but the graph still carries the mask. The APIs here keep the same
//! fallback semantics while giving autograd an explicit, deterministic
//! scatter-add VJP and adding an interpolating row gather for DORSCH.
//!
//! `Ops::Gather` is an internal marker: it is lowered by [`lower_gathers`] to
//! the existing advanced-index graph before scheduling, so the scheduler,
//! renderers and interpreter only see already-ported ops. The marker exists
//! only long enough to attach the gather VJP.

use crate::dtype::{self, ConstVal};
use crate::uop::UopId;
use crate::uop::arg::Arg;
use crate::uop::ctx::UOpCtx;
use crate::uop::ops::Ops;
use crate::uop::sint::SVal;
use crate::uop::sym::{CtxVal, r};
use crate::uop::upat::{PatternMatcher, UPat, graph_rewrite};

use super::ix::Ix;
use super::movement;
use super::{Tensor, creation};

/// Build the internal `GATHER(src, index)` marker (flat source, non-negative
/// flat index).
fn gather_marker(ctx: &mut UOpCtx, src: UopId, index: UopId) -> UopId {
    let dt = ctx.dt_of(&src);
    ctx.uop(Ops::Gather, Some(dt), vec![src, index], Arg::None)
}

fn as_tensor(ctx: &UOpCtx, u: UopId) -> Tensor {
    Tensor::from_uop(ctx, u)
}

/// Flatten a tensor to 1-D without materializing (reshape view).
fn flat(ctx: &mut UOpCtx, x: UopId) -> UopId {
    let shape = ctx.shape(&x).ok().flatten().unwrap_or_default();
    if shape.len() <= 1 {
        return x;
    }
    let n: i128 = shape.iter().map(|s| s.clone().as_i128_or_panic()).product();
    movement::reshape(ctx, x, &[SVal::I(n)])
}

fn numel(ctx: &mut UOpCtx, u: &UopId) -> i64 {
    ctx.shape(u)
        .ok()
        .flatten()
        .map(|s| {
            s.iter()
                .map(|d| d.clone().as_i128_or_panic())
                .product::<i128>()
        })
        .unwrap_or(0) as i64
}

/// Zero scalar of `like`'s dtype (broadcast by `where`, not expanded).
fn zero_of(ctx: &mut UOpCtx, like: &UopId) -> UopId {
    let dt = ctx.dt_of(like);
    let v = if dtype::is_float(dt) {
        ConstVal::Float(0.0)
    } else {
        ConstVal::Int(0)
    };
    ctx.const_py(v, Some(dt))
}

/// Normalize a possibly-negative index over `size` and clamp it to
/// `[0, size - 1]`, returning `(safe, valid)`.
fn normalize_clamp(ctx: &mut UOpCtx, index: UopId, size: i64) -> (Tensor, Tensor) {
    let idx = as_tensor(ctx, index);
    let idt = idx.dtype_ctx(ctx);
    let size_c = idx.const_like_ctx(ctx, ConstVal::Int(size as i128), Some(idt));
    let neg = idx.lt_ctx(ctx, 0i64);
    let wrapped = idx.add_ctx(ctx, &size_c);
    let norm = neg.where__ctx(ctx, &wrapped, &idx);
    let ge = norm.ge_ctx(ctx, 0i64);
    let lt = norm.lt_ctx(ctx, size);
    let valid = ge.bitwise_and_ctx(ctx, &lt);
    let safe = norm.minimum_ctx(ctx, size - 1).maximum_ctx(ctx, 0i64);
    (safe, valid)
}

/// Like [`normalize_clamp`] but without Python-style negative wrapping; used
/// by `gather_interp`, where a negative row is simply out of range.
fn clamp_no_wrap(ctx: &mut UOpCtx, index: UopId, size: i64) -> (Tensor, Tensor) {
    let idx = as_tensor(ctx, index);
    let ge = idx.ge_ctx(ctx, 0i64);
    let lt = idx.lt_ctx(ctx, size);
    let valid = ge.bitwise_and_ctx(ctx, &lt);
    let safe = idx.minimum_ctx(ctx, size - 1).maximum_ctx(ctx, 0i64);
    (safe, valid)
}

/// Zero a gathered value where `mask` is false without a data-dependent branch.
fn mask_zero(ctx: &mut UOpCtx, x: &Tensor, mask: &Tensor) -> Tensor {
    let dt = x.dtype_ctx(ctx);
    let m = mask.cast_ctx(ctx, dt);
    x.mul_ctx(ctx, &m)
}

/// `out = src[index]` for a flat source. Negative indices wrap once against
/// `src.numel()`; out-of-range indices read zero.
fn gather_flat(ctx: &mut UOpCtx, src: UopId, index: UopId) -> UopId {
    let n = numel(ctx, &src);
    let (safe, valid) = normalize_clamp(ctx, index, n.max(1));
    let marker_u = gather_marker(ctx, src.clone(), safe.uop());
    let marker = as_tensor(ctx, marker_u);
    mask_zero(ctx, &marker, &valid).uop()
}

fn arange_c(ctx: &mut UOpCtx, n: i64, dt: dtype::DTypeId) -> Tensor {
    Tensor::arange(&mut *ctx, 0.0, Some(n as f64), 1.0, Some(dt))
}

impl Tensor {
    /// Direct integer gather along one axis.
    ///
    /// Supported cases:
    /// - 1-D `self`, any index shape, `dim = 0`;
    /// - 2-D `self`, `dim = 0`: out shape is `index.shape + [S]`;
    /// - 2-D `self`, `dim = 1`: `index.shape` is the output shape (leading
    ///   dimensions are flattened to rows).
    ///
    /// Negative indices wrap Python-style. Out-of-range indices produce `0`.
    /// Forward is one indirect load per output element; the VJP scatters the
    /// upstream gradient into the source (see [`Tensor::scatter_add`]).
    pub fn gather_direct(&self, dim: i64, index: &Tensor) -> Tensor {
        self.gather_direct_ctx(&mut self.ctx.borrow_mut(), dim, index)
    }
    #[doc(hidden)]
    pub fn gather_direct_ctx(&self, ctx: &mut UOpCtx, dim: i64, index: &Tensor) -> Tensor {
        let sshape: Vec<SVal> = self.shape_ctx(ctx).into_vec();
        let ishape: Vec<SVal> = index.shape_ctx(ctx).into_vec();
        assert!(
            index.dtype_ctx(ctx) == dtype::INT32 || index.dtype_ctx(ctx) == dtype::INT64,
            "TGERR: gather_direct index must be int32/int64"
        );
        if sshape.len() == 1 {
            assert_eq!(
                dim, 0,
                "TGERR: gather_direct 1-D source only supports dim=0"
            );
            let src = flat(ctx, self.uop());
            let out = gather_flat(ctx, src, index.uop());
            return as_tensor(ctx, out);
        }
        assert_eq!(
            sshape.len(),
            2,
            "TGERR: gather_direct supports 1-D or 2-D sources"
        );
        let d = movement::resolve_dim(2, dim, false);
        let n = sshape[0]
            .clone()
            .as_i64()
            .expect("TGERR: gather_direct static source rows");
        let s = sshape[1]
            .clone()
            .as_i64()
            .expect("TGERR: gather_direct static source cols");
        let src = flat(ctx, self.uop());
        let (safe, valid) = normalize_clamp(ctx, index.uop(), if d == 0 { n } else { s });
        if d == 0 {
            // [idx..., S] = safe_row * S + arange(S)
            let safe_i = safe.mul_ctx(ctx, s);
            let ar = arange_c(ctx, s, dtype::INT32);
            let mut ar_shape = vec![SVal::I(1); ishape.len()];
            ar_shape.push(SVal::I(s as i128));
            let ar = ar.reshape_ctx(ctx, ar_shape);
            let flat_idx = safe_i.unsqueeze_ctx(ctx, -1).add_ctx(ctx, &ar);
            let mut exp_shape = ishape.clone();
            exp_shape.push(SVal::I(s as i128));
            let flat_idx = flat_idx.expand_ctx(ctx, exp_shape);
            let marker_u = gather_marker(ctx, src, flat_idx.uop());
            let out = as_tensor(ctx, marker_u);
            let mut mask_shape = ishape.clone();
            mask_shape.push(SVal::I(s as i128));
            let mask = valid.unsqueeze_ctx(ctx, -1).expand_ctx(ctx, mask_shape);
            let masked = mask_zero(ctx, &out, &mask);
            as_tensor(ctx, masked.uop())
        } else {
            // Flatten the leading index dims to rows; the last dim are columns.
            let nd = ishape.len();
            assert!(
                nd >= 1,
                "TGERR: gather_direct 2-D dim=1 needs a non-scalar index"
            );
            let rows: i128 = ishape[..nd - 1]
                .iter()
                .map(|v| v.clone().as_i128_or_panic())
                .product::<i128>()
                .max(1);
            let cols = ishape[nd - 1].clone().as_i128_or_panic();
            let flat_shape = vec![SVal::I(rows), SVal::I(cols)];
            let safe2 = safe.reshape_ctx(ctx, flat_shape.clone());
            let row = arange_c(ctx, rows as i64, dtype::INT32).unsqueeze_ctx(ctx, -1);
            let row = row.expand_ctx(ctx, flat_shape.clone());
            let col_valid = valid.reshape_ctx(ctx, flat_shape.clone());
            let col_valid = if rows <= n as i128 {
                col_valid
            } else {
                let rvalid = arange_c(ctx, rows as i64, dtype::INT32).lt_ctx(ctx, n);
                let rvalid = rvalid
                    .unsqueeze_ctx(ctx, -1)
                    .expand_ctx(ctx, flat_shape.clone());
                col_valid.bitwise_and_ctx(ctx, &rvalid)
            };
            let flat_idx = row.mul_ctx(ctx, s).add_ctx(ctx, &safe2);
            let marker_u = gather_marker(ctx, src, flat_idx.uop());
            let out = as_tensor(ctx, marker_u).reshape_ctx(ctx, ishape.clone());
            let mask = col_valid.reshape_ctx(ctx, ishape);
            let masked = mask_zero(ctx, &out, &mask);
            as_tensor(ctx, masked.uop())
        }
    }

    /// Fused 2-D interpolating row gather.
    ///
    /// `self` is an atlas `[N, S]`, `rows` and `q` have equal shape `[..]`.
    /// Returns `(value, slope)` with the same shape as `q`:
    ///
    /// ```text
    /// i = floor(q), clamped to [0, S-2]
    /// f = clamp(q - i, 0, 1)
    /// value = atlas[row, i] * (1 - f) + atlas[row, i+1] * f
    /// slope = atlas[row, i+1] - atlas[row, i]
    /// ```
    ///
    /// Out-of-range (including negative) rows give `value = slope = 0`.
    /// `q <= 0` gives `i = 0, f = 0`; `q >= S-1` gives `i = S-2, f = 1`.
    /// The forward performs two adjacent indirect loads per output element.
    pub fn gather_interp(&self, rows: &Tensor, q: &Tensor) -> (Tensor, Tensor) {
        self.gather_interp_ctx(&mut self.ctx.borrow_mut(), rows, q)
    }
    #[doc(hidden)]
    pub fn gather_interp_ctx(
        &self,
        ctx: &mut UOpCtx,
        rows: &Tensor,
        q: &Tensor,
    ) -> (Tensor, Tensor) {
        let sshape: Vec<SVal> = self.shape_ctx(ctx).into_vec();
        assert_eq!(
            sshape.len(),
            2,
            "TGERR: gather_interp atlas must be 2-D [N, S]"
        );
        assert_eq!(
            rows.shape_ctx(ctx).into_vec(),
            q.shape_ctx(ctx).into_vec(),
            "TGERR: gather_interp rows and q must have equal shape"
        );
        let n = sshape[0]
            .clone()
            .as_i64()
            .expect("TGERR: gather_interp static rows");
        let s = sshape[1]
            .clone()
            .as_i64()
            .expect("TGERR: gather_interp static cols");
        assert!(s >= 2, "TGERR: gather_interp needs at least two columns");
        let qdt = q.dtype_ctx(ctx);
        assert!(dtype::is_float(qdt), "TGERR: gather_interp q must be float");

        // i = floor(q), clamped to [0, S-2]
        let i_int = q.floor_ctx(ctx).cast_ctx(ctx, dtype::INT32);
        let i_int = i_int.minimum_ctx(ctx, s - 2).maximum_ctx(ctx, 0i64);
        // f = q - i inside the valid range, 0 at/below 0, 1 at/above S-1.
        // A range test (not a clamp) keeps the q derivative exactly 1 at an
        // interior integer, where q - i is 0.
        let i_f = i_int.cast_ctx(ctx, qdt);
        let f0 = q.sub_ctx(ctx, &i_f);
        let f = q.gt_ctx(ctx, 0.0).where__ctx(ctx, &f0, 0.0);
        let f = q.lt_ctx(ctx, (s - 1) as f64).where__ctx(ctx, &f, 1.0);

        let (safe_row, row_valid) = clamp_no_wrap(ctx, rows.uop(), n);
        let base = safe_row.mul_ctx(ctx, s).add_ctx(ctx, &i_int);
        let idx1 = base.add_ctx(ctx, 1i64);

        // One marker gather over [.., 2] so the two taps stay in one kernel.
        let b0 = base.unsqueeze_ctx(ctx, -1);
        let b1 = idx1.unsqueeze_ctx(ctx, -1);
        let idx_pair = b0.cat_ctx(ctx, &[b1], -1);
        let src = flat(ctx, self.uop());
        let marker_u = gather_marker(ctx, src, idx_pair.uop());
        let taps = as_tensor(ctx, marker_u);
        let qshape: Vec<SVal> = q.shape_ctx(ctx).into_vec();
        let nd = qshape.len();
        let mut ixs0: Vec<Ix> = vec![Ix::All; nd];
        ixs0.push(Ix::Index(0));
        let mut ixs1: Vec<Ix> = vec![Ix::All; nd];
        ixs1.push(Ix::Index(1));
        let v0 = taps.getitem_ctx(ctx, &ixs0);
        let v1 = taps.getitem_ctx(ctx, &ixs1);

        let one_minus_f = f.neg_ctx(ctx).add_ctx(ctx, 1.0);
        let v1f = v1.mul_ctx(ctx, &f);
        let value = v0.mul_ctx(ctx, &one_minus_f).add_ctx(ctx, &v1f);
        let slope = v1.sub_ctx(ctx, &v0);
        let value = mask_zero(ctx, &value, &row_valid);
        let slope_ = mask_zero(ctx, &slope, &row_valid);
        (value, slope_)
    }

    /// `out[c] = sum(values[row_ptr[c] .. row_ptr[c+1]])`.
    ///
    /// `row_ptr` is an int32 `[C+1]` tensor that must be nondecreasing with
    /// `row_ptr[C] == values.numel()`; a static `row_ptr` is validated at graph
    /// construction (a dynamic `row_ptr` is trusted). Empty segments are
    /// allowed. Forward is `O(E + C)` via a prefix sum plus two gathers.
    pub fn segment_sum(row_ptr: &Tensor, values: &Tensor) -> Tensor {
        Tensor::segment_sum_ctx(&mut row_ptr.ctx.borrow_mut(), row_ptr, values)
    }
    #[doc(hidden)]
    pub fn segment_sum_ctx(ctx: &mut UOpCtx, row_ptr: &Tensor, values: &Tensor) -> Tensor {
        assert_eq!(
            row_ptr.dtype_ctx(ctx),
            dtype::INT32,
            "TGERR: segment_sum row_ptr must be int32"
        );
        let cshape: Vec<SVal> = row_ptr.shape_ctx(ctx).into_vec();
        assert_eq!(
            cshape.len(),
            1,
            "TGERR: segment_sum row_ptr must be 1-D [C+1]"
        );
        let c1 = cshape[0]
            .clone()
            .as_i64()
            .expect("TGERR: segment_sum static row_ptr");
        assert!(
            c1 >= 1,
            "TGERR: segment_sum row_ptr must have C+1 >= 1 entries"
        );
        let vals = flat(ctx, values.uop());
        let e = numel(ctx, &vals);
        if let Some(ptr) = static_ints(ctx, row_ptr) {
            assert_eq!(ptr.len() as i64, c1, "TGERR: segment_sum row_ptr length");
            assert!(
                ptr.windows(2).all(|w| w[0] <= w[1]),
                "TGERR: segment_sum row_ptr must be nondecreasing"
            );
            assert_eq!(
                ptr[c1 as usize - 1],
                e,
                "TGERR: segment_sum last row_ptr must equal values.numel()"
            );
        }
        let vdt = ctx.dt_of(&vals);
        let dev = device_opt(ctx, values);
        if e == 0 || c1 == 1 {
            return creation::zeros(ctx, &[SVal::I((c1 - 1) as i128)], Some(vdt), dev.as_deref());
        }
        // out[c] = cumsum[hi-1] - cumsum[lo-1], with the `lo == 0` term
        // zeroed. This avoids a [0, cumsum] PAD view whose padding can alias
        // under vectorization; `cs` is materialized once and read twice.
        let cs = as_tensor(ctx, vals).cumsum_ctx(ctx, 0).contiguous_ctx(ctx);
        let lo = row_ptr.getitem_ctx(
            ctx,
            &[Ix::Range {
                start: Some(0),
                stop: Some(c1 - 1),
                step: 1,
            }],
        );
        let hi = row_ptr.getitem_ctx(
            ctx,
            &[Ix::Range {
                start: Some(1),
                stop: Some(c1),
                step: 1,
            }],
        );
        let hi_valid = hi.gt_ctx(ctx, 0i64);
        let lo_valid = lo.gt_ctx(ctx, 0i64);
        let hi_idx = hi
            .sub_ctx(ctx, 1i64)
            .minimum_ctx(ctx, e - 1)
            .maximum_ctx(ctx, 0i64);
        let lo_idx = lo
            .sub_ctx(ctx, 1i64)
            .minimum_ctx(ctx, e - 1)
            .maximum_ctx(ctx, 0i64);
        let a = cs.gather_direct_ctx(ctx, 0, &hi_idx);
        let b = cs.gather_direct_ctx(ctx, 0, &lo_idx);
        let a = mask_zero(ctx, &a, &hi_valid);
        let b = mask_zero(ctx, &b, &lo_valid);
        a.sub_ctx(ctx, &b)
    }

    /// `out = zeros(len); out[idx[e]] += values[e]`.
    ///
    /// Deterministic and free of float atomics. Out-of-range indices (negative
    /// or `>= len`) are ignored. A static `idx` is turned into a stable CSR
    /// (counting sort + `segment_sum`); a dynamic index uses a one-hot
    /// reduction, which is still deterministic and atomic-free. The VJP w.r.t.
    /// `values` is a gather of the upstream gradient at `idx`.
    pub fn scatter_add(len: i64, idx: &Tensor, values: &Tensor) -> Tensor {
        Tensor::scatter_add_ctx(&mut idx.ctx.borrow_mut(), len, idx, values)
    }
    #[doc(hidden)]
    pub fn scatter_add_ctx(ctx: &mut UOpCtx, len: i64, idx: &Tensor, values: &Tensor) -> Tensor {
        assert!(len >= 0, "TGERR: scatter_add len must be >= 0");
        assert_eq!(
            idx.dtype_ctx(ctx),
            dtype::INT32,
            "TGERR: scatter_add idx must be int32"
        );
        assert_eq!(
            idx.shape_ctx(ctx).numel_int(),
            values.shape_ctx(ctx).numel_int(),
            "TGERR: scatter_add idx and values must have the same numel"
        );
        let idx_flat = flat(ctx, idx.uop());
        let vals_flat = flat(ctx, values.uop());
        let e = numel(ctx, &idx_flat).max(0) as usize;
        let dt = ctx.dt_of(&vals_flat);
        let dev = device_opt(ctx, values);
        if len == 0 || e == 0 {
            return creation::zeros(ctx, &[SVal::I(len as i128)], Some(dt), dev.as_deref());
        }
        if let Some(raw) = static_ints(ctx, idx) {
            return scatter_add_csr(ctx, len, &raw, vals_flat, dt, dev);
        }
        // Dynamic fallback: out[c] = sum_e (idx[e] == c) ? values[e] : 0.
        let idx_t = as_tensor(ctx, idx_flat).reshape_ctx(ctx, [1, e as i64]);
        let vals_t = as_tensor(ctx, vals_flat.clone()).reshape_ctx(ctx, [1, e as i64]);
        let ar = arange_c(ctx, len, dtype::INT32).reshape_ctx(ctx, [len, 1]);
        let idx_e = idx_t.expand_ctx(ctx, [len, e as i64]);
        let ar_e = ar.expand_ctx(ctx, [len, e as i64]);
        let cond = idx_e.eq_ctx(ctx, &ar_e);
        let zero_u = zero_of(ctx, &vals_flat);
        let zero = as_tensor(ctx, zero_u);
        let vals_e = vals_t.expand_ctx(ctx, [len, e as i64]);
        let masked = cond.where__ctx(ctx, &vals_e, zero);
        masked.sum_ctx(ctx, Some(&[1]), false, Some(dt))
    }
}

/// Device of a tensor as a name, for host-built index constants.
fn device_opt(ctx: &mut UOpCtx, t: &Tensor) -> Option<String> {
    match crate::schedule::indexing::device_of(ctx, t.uop()) {
        Some(crate::uop::arg::DeviceSpec::Single(s)) => Some(s),
        Some(crate::uop::arg::DeviceSpec::Multi(v)) => v.first().cloned(),
        None => None,
    }
}

/// Read a constant integer tensor's values without scheduling. Only contiguous
/// `from_slice` leaves (possibly reshaped) qualify; anything else returns `None`.
pub(crate) fn static_ints(ctx: &mut UOpCtx, t: &Tensor) -> Option<Vec<i64>> {
    let mut u = t.uop();
    while matches!(
        ctx.op_of(&u),
        Ops::Reshape | Ops::Detach | Ops::Contiguous | Ops::Copy | Ops::After
    ) {
        u = ctx.srcs(&u)[0].clone();
    }
    let data = u.node().initial_data.borrow().clone()?;
    let dt = ctx.dt_of(&u);
    let shape = ctx.shape(&u).ok().flatten()?;
    let n: i128 = shape.iter().map(|d| d.clone().as_i128_or_panic()).product();
    let bytes = data.borrow();
    let mut out = Vec::with_capacity(n.max(0) as usize);
    if dt == dtype::INT32 {
        for c in bytes.chunks_exact(4).take(n.max(0) as usize) {
            out.push(i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64);
        }
    } else if dt == dtype::INT64 {
        for c in bytes.chunks_exact(8).take(n.max(0) as usize) {
            out.push(i64::from_le_bytes([
                c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
            ]));
        }
    } else {
        return None;
    }
    (out.len() as i128 == n).then_some(out)
}

/// Stable counting sort + segmented sum for a static index.
fn scatter_add_csr(
    ctx: &mut UOpCtx,
    len: i64,
    raw: &[i64],
    vals: UopId,
    dt: dtype::DTypeId,
    device: Option<String>,
) -> Tensor {
    let c = len as usize;
    let mut counts = vec![0i64; c];
    let mut valid = Vec::with_capacity(raw.len());
    for &v in raw {
        if v >= 0 && (v as usize) < c {
            counts[v as usize] += 1;
            valid.push(v as usize);
        }
    }
    // row_ptr: exclusive prefix sum of counts.
    let mut row_ptr = vec![0i32; c + 1];
    for i in 0..c {
        row_ptr[i + 1] = row_ptr[i] + counts[i] as i32;
    }
    let nvalid = valid.len();
    let mut offset = row_ptr.clone();
    let mut perm = vec![0i32; nvalid];
    for (e, &v) in raw.iter().enumerate() {
        if v >= 0 && (v as usize) < c {
            let p = offset[v as usize] as usize;
            perm[p] = e as i32;
            offset[v as usize] += 1;
        }
    }
    let dev = device.as_deref();
    if nvalid == 0 {
        return creation::zeros(ctx, &[SVal::I(len as i128)], Some(dt), dev);
    }
    let perm_t = Tensor::from_slice(&mut *ctx, &perm, Some(dtype::INT32), dev);
    let sorted = as_tensor(ctx, vals).gather_direct_ctx(ctx, 0, &perm_t);
    let ptr_t = Tensor::from_slice(&mut *ctx, &row_ptr, Some(dtype::INT32), dev);
    // `segment_sum` is already dense over the original `len` slots (empty
    // segments are zeros), so no second scatter/mapping pass is needed.
    Tensor::segment_sum_ctx(ctx, &ptr_t, &sorted).contiguous_ctx(ctx)
}

/// Replace every internal `GATHER` marker with the already-ported advanced
/// index graph. Called before scheduling; the codegen gated-load lift then
/// turns that graph into one indirect load per output element.
pub fn lower_gathers(ctx: &mut UOpCtx, root: UopId) -> UopId {
    if !ctx
        .toposort(&root)
        .iter()
        .any(|u| ctx.op_of(u) == Ops::Gather)
    {
        return root;
    }
    let pm = PatternMatcher::new(vec![(
        UPat::ops1(Ops::Gather).allow_all_srcs().named("g"),
        r(|ctx, st, _| {
            let node = crate::uop::sym::sget(st, "g");
            let srcs = ctx.srcs(&node).to_vec();
            let (src, idx) = (srcs[0].clone(), srcs[1].clone());
            let n = numel(ctx, &src);
            let idx_t = as_tensor(ctx, idx.clone());
            let idt = ctx.dt_of(&idx);
            let zero_i = ctx.const_py(ConstVal::Int(0), Some(idt));
            let n_c = ctx.const_py(ConstVal::Int(n as i128), Some(idt));
            let neg = ctx.lt(idx.clone(), zero_i);
            let lt = ctx.lt(idx.clone(), n_c);
            let not_neg = ctx.logical_not(neg);
            let valid = ctx.and_(not_neg, lt);
            let zero_i2 = ctx.const_py(ConstVal::Int(0), Some(idt));
            let safe_i = ctx.where_(valid.clone(), idx.clone(), zero_i2);
            let _ = idx_t;
            let safe_t = as_tensor(ctx, safe_i);
            let gathered = movement::getitem(ctx, src.clone(), &[Ix::Tensor(safe_t)]);
            let zero_u = zero_of(ctx, &src);
            let out = ctx.where_(valid, gathered, zero_u);
            Some(out)
        }),
    )]);
    graph_rewrite(ctx, root, &pm, false, None, false, false, CtxVal::None)
        .unwrap_or_else(|e| panic!("TGERR: lower gather: {e}"))
}
