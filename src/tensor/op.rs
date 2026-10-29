//! `mixin/op.py` op-level tensor operations: conv2d, pooling, cross-entropy,
//! one-hot / argmax sugar, and the `_pad_constant` helper the conv/pool paths
//! share. Ported one-for-one from tinygrad v0.14.

use crate::dtype::{self, ConstVal, DTypeId};
use crate::uop::ctx::UOpCtx;
use crate::uop::sint::SVal;

use super::Tensor;

/// `flat_to_grouped(padding)` (helpers.py:55): `(l, r, t, b, ...)` →
/// `((t, b), (l, r), ...)`.
pub fn flat_to_grouped(padding: &[i64]) -> Vec<(i64, i64)> {
    let n = padding.len();
    let first: Vec<i64> = (0..n / 2).map(|k| padding[n - 2 - 2 * k]).collect();
    let second: Vec<i64> = (0..n / 2).map(|k| padding[n - 1 - 2 * k]).collect();
    first.into_iter().zip(second).collect()
}

/// `resolve_pool_pads(padding, dims)` (helpers.py:56-59): normalize an int /
/// length-`dims` / length-`2*dims` padding to the flat `2*dims` form.
pub fn resolve_pool_pads(padding: &[i64], dims: usize) -> Vec<i64> {
    if padding.len() == 1 {
        vec![padding[0]; 2 * dims]
    } else if padding.len() == 2 * dims {
        padding.to_vec()
    } else if padding.len() == dims {
        // `[p for p in padding for _ in range(2)][::-1]`
        let mut v = Vec::with_capacity(2 * dims);
        for &p in padding {
            v.push(p);
            v.push(p);
        }
        v.reverse();
        v
    } else {
        panic!(
            "TGERR: padding must be an int or a sequence of length {dims} or {}",
            2 * dims
        );
    }
}

/// integer `ceildiv(num, amt)` (helpers.py) — Python floor-division semantics.
fn ceildiv(num: i64, amt: i64) -> i64 {
    -((-num).div_euclid(amt))
}

/// `Tensor._apply_ceil_mode(pads, k_, s_, d_)` (op.py:1276-1287).
fn apply_ceil_mode(
    x_shape: &[SVal],
    pads: &[i64],
    k: &[i64],
    stride: &[i64],
    dilation: &[i64],
) -> Vec<i64> {
    let n = k.len();
    let i_: Vec<i64> = x_shape[x_shape.len() - n..]
        .iter()
        .map(|s| s.clone().as_i64().unwrap_or(0))
        .collect();
    let mut grouped = flat_to_grouped(pads);
    let o_: Vec<i64> = (0..n)
        .map(|d| {
            ceildiv(
                i_[d] + grouped[d].0 + grouped[d].1 - (dilation[d] * (k[d] - 1) + 1),
                stride[d],
            ) + 1
        })
        .collect();
    for d in 0..n {
        let (pb, pa) = grouped[d];
        // s*(o-1) + (d*(k-1)+1) - (i+pb+pa)  minus  max(s*(o-1) - (pb+i-1), 0)
        let tail = stride[d] * (o_[d] - 1) + (dilation[d] * (k[d] - 1) + 1) - (i_[d] + pb + pa);
        let cut = std::cmp::max(stride[d] * (o_[d] - 1) - (pb + i_[d] - 1), 0);
        grouped[d] = (pb, pa + tail - cut);
    }
    // `flatten(reversed(grouped_pads))`
    let mut out = Vec::with_capacity(2 * n);
    for (a, b) in grouped.iter().rev() {
        out.push(*a);
        out.push(*b);
    }
    out
}

fn is_zero(v: &ConstVal) -> bool {
    matches!(v, ConstVal::Int(0)) || matches!(v, ConstVal::Float(f) if *f == 0.0)
}

impl Tensor {
    /// `Tensor.eq(other)` — elementwise equality (tinygrad `__eq__`).
    pub fn eq(&self, other: impl Into<super::TensorAddArg>) -> Tensor {
        self.eq_ctx(&mut self.ctx.borrow_mut(), other)
    }
    #[doc(hidden)]
    pub fn eq_ctx(&self, ctx: &mut UOpCtx, other: impl Into<super::TensorAddArg>) -> Tensor {
        let o = self.cmp_operand(ctx, other);
        {
            let __u = super::elementwise::eq(ctx, self.uop().clone(), o);
            Tensor::from_uop(ctx, __u)
        }
    }

    /// `Tensor.argmax(axis, keepdim)` (op.py:866-891): index of the max along
    /// `axis`. `m = (x == x.max(axis, keepdim=True))`; `idx = m * arange(C, 0, -1)`;
    /// result `C - idx.max(axis, keepdim)` (the descending arange makes the
    /// smallest index win ties, matching tinygrad).
    pub fn argmax(&self, axis: Option<i64>, keepdim: bool) -> Tensor {
        self.argmax_ctx(&mut self.ctx.borrow_mut(), axis, keepdim)
    }
    #[doc(hidden)]
    pub fn argmax_ctx(&self, ctx: &mut UOpCtx, axis: Option<i64>, keepdim: bool) -> Tensor {
        let axis = match axis {
            None => return self.flatten_ctx(ctx, 0, -1).argmax_ctx(ctx, Some(0), keepdim),
            Some(a) => super::movement::resolve_dim(self.shape_ctx(ctx).len(), a, false),
        };
        let ndim = self.shape_ctx(ctx).len();
        let c = self.shape_ctx(ctx)[axis].clone().as_i64().unwrap_or(0);
        let mx_axis = self.max_ctx(ctx, Some(&[axis]), true);
        let m = self.eq_ctx(ctx, mx_axis);
        let ar = Tensor::arange(
            &mut *ctx,
            c as f64,
            Some(0.0),
            -1.0,
            Some(dtype::default_int()),
        );
        let mut sh: Vec<SVal> = vec![SVal::I(c as i128)];
        sh.extend(std::iter::repeat_n(SVal::I(1), ndim - axis - 1));
        let arr = ar.reshape_ctx(ctx, sh);
        let idx = m.mul_ctx(ctx, &arr);
        let mx = idx.max_ctx(ctx, Some(&[axis]), keepdim);
        let c_u = ctx.const_like(mx.uop().clone(), ConstVal::Int(c as i128));
        let c_t = Tensor::from_uop(ctx, c_u);
        c_t.sub_ctx(ctx, &mx).cast_ctx(ctx, dtype::default_int())
    }

    /// `Tensor._pad_constant(p_x, value)` (op.py:282-290): shrink first for
    /// negative pads, then pad with the non-negative remainder. `p_x` is a
    /// per-dim `(before, after)` list (`None` → 0).
    pub fn pad_constant(&self, p_x: &[(Option<i64>, Option<i64>)], value: ConstVal) -> Tensor {
        self.pad_constant_ctx(&mut self.ctx.borrow_mut(), p_x, value)
    }
    #[doc(hidden)]
    pub fn pad_constant_ctx(
        &self,
        ctx: &mut UOpCtx,
        p_x: &[(Option<i64>, Option<i64>)],
        value: ConstVal,
    ) -> Tensor {
        let shape: Vec<SVal> = self.shape_ctx(ctx).into_vec();
        assert_eq!(
            shape.len(),
            p_x.len(),
            "TGERR: pad_constant ndim != len(p_x)"
        );
        let pairs: Vec<(i64, i64)> = p_x
            .iter()
            .map(|(a, b)| (a.unwrap_or(0), b.unwrap_or(0)))
            .collect();
        let has_neg = pairs.iter().any(|(a, b)| *a < 0 || *b < 0);
        let (x, pads): (Tensor, Vec<(i64, i64)>) = if has_neg {
            // shrink to `(-min(p_b,0), min(p_a+s, s))` per dim
            let shr: Vec<(i64, i64)> = pairs
                .iter()
                .zip(shape.iter())
                .map(|((p_b, p_a), s)| {
                    let si = s.clone().as_i64().unwrap_or(0);
                    ((-p_b).max(0), (p_a + si).min(si))
                })
                .collect();
            let x = self.shrink_ctx(ctx, &shr);
            let pads = pairs
                .iter()
                .map(|(p_b, p_a)| ((*p_b).max(0), (*p_a).max(0)))
                .collect();
            (x, pads)
        } else {
            (self.clone(), pairs)
        };
        let base = x.pad_ctx(ctx, &pads);
        if is_zero(&value) {
            return base;
        }
        // `pad(x.const_like(True, bool), pads).where(base, value)`
        let mask_u = ctx.const_like_dt(x.uop().clone(), ConstVal::Bool(true), Some(dtype::BOOL));
        let mask = Tensor::from_uop(ctx, mask_u).pad_ctx(ctx, &pads);
        mask.where__ctx(ctx, base, value)
    }

    /// `Tensor.one_hot(num_classes)` (op.py:1013-1024):
    /// `self[..., None]._one_hot_along_dim(num_classes).where(1, 0)`.
    pub fn one_hot(&self, num_classes: i64) -> Tensor {
        self.one_hot_ctx(&mut self.ctx.borrow_mut(), num_classes)
    }
    #[doc(hidden)]
    pub fn one_hot_ctx(&self, ctx: &mut UOpCtx, num_classes: i64) -> Tensor {
        assert!(
            num_classes >= 0,
            "TGERR: num_classes must be non-negative, got {num_classes}"
        );
        let x = self.unsqueeze_ctx(ctx, -1);
        let oh = x.one_hot_along_dim_ctx(ctx, num_classes, -1);
        // `where(1, 0)` gives the concrete int class ids (tinygrad's `one_hot`).
        oh.where__ctx(ctx, 1, 0).cast_ctx(ctx, dtype::default_int())
    }

    /// `Tensor._one_hot_along_dim(num_classes, dim)` (op.py:1008-1011).
    pub fn one_hot_along_dim(&self, num_classes: i64, dim: i64) -> Tensor {
        self.one_hot_along_dim_ctx(&mut self.ctx.borrow_mut(), num_classes, dim)
    }
    #[doc(hidden)]
    pub fn one_hot_along_dim_ctx(&self, ctx: &mut UOpCtx, num_classes: i64, dim: i64) -> Tensor {
        let dt = self.dtype_ctx(ctx);
        assert!(
            dtype::is_int(dt),
            "TGERR: _one_hot_along_dim expects int index tensor, got {dt:?}"
        );
        let ndim = self.shape_ctx(ctx).len();
        let d = super::movement::resolve_dim(ndim, dim, false);
        let offset = ndim - d - 1;
        let ar = Tensor::arange(
            &mut *ctx,
            0.0,
            Some(num_classes as f64),
            1.0,
            Some(dtype::default_int()),
        );
        let mut sh: Vec<SVal> = vec![SVal::I(num_classes as i128)];
        sh.extend(std::iter::repeat_n(SVal::I(1), offset));
        let arr = ar.reshape_ctx(ctx, sh);
        self.eq_ctx(ctx, arr)
    }

    /// `Tensor.conv2d(weight, bias, groups, stride, dilation, padding, dtype)`
    /// (op.py:1566-1607). `padding` is the int / length-`dims` / length-`2*dims`
    /// form. Works for any number of spatial dims (like tinygrad).
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        groups: i64,
        stride: i64,
        dilation: i64,
        padding: &[i64],
        dtype: Option<DTypeId>,
    ) -> Tensor {
        self.conv2d_ctx(
            &mut self.ctx.borrow_mut(),
            weight,
            bias,
            groups,
            stride,
            dilation,
            padding,
            dtype,
        )
    }
    #[doc(hidden)]
    pub fn conv2d_ctx(
        &self,
        ctx: &mut UOpCtx,
        weight: &Tensor,
        bias: Option<&Tensor>,
        groups: i64,
        stride: i64,
        dilation: i64,
        padding: &[i64],
        dtype: Option<DTypeId>,
    ) -> Tensor {
        let x_shape: Vec<SVal> = self.shape_ctx(ctx).into_vec();
        let w_shape: Vec<SVal> = weight.shape_ctx(ctx).into_vec();
        let hw_len = w_shape.len() - 2;
        let padding_ = resolve_pool_pads(padding, hw_len);
        let (bs, cin_) = (x_shape[0].clone(), x_shape[1].clone());
        let (cout, cin) = (w_shape[0].clone(), w_shape[1].clone());
        let cin_i = cin.clone().as_i64().unwrap_or(0);
        let cin_i_ = cin_.clone().as_i64().unwrap_or(0);
        let cout_i = cout.clone().as_i64().unwrap_or(0);
        assert_eq!(
            groups * cin_i,
            cin_i_,
            "TGERR: input shape {:?} does not match weight {:?}",
            x_shape,
            w_shape
        );
        assert_eq!(
            x_shape.len(),
            w_shape.len(),
            "TGERR: input/weight ndims differ"
        );

        // conv2d is a pooling op (with padding, possibly negative — _pad_constant handles the shrink)
        let grouped = flat_to_grouped(&padding_);
        let mut p_x: Vec<(Option<i64>, Option<i64>)> =
            vec![(Some(0), Some(0)); x_shape.len() - hw_len];
        p_x.extend(grouped.iter().map(|(a, b)| (Some(*a), Some(*b))));
        let x = self.pad_constant_ctx(ctx, &p_x, ConstVal::Float(0.0));

        let hw: Vec<i64> = w_shape[2..]
            .iter()
            .map(|s| s.clone().as_i64().unwrap_or(0))
            .collect();
        // `_pool(HW, stride, dilation)` then `oyx = x.shape[2:-len(HW)]`
        let x = x.pool_ctx(ctx, &hw, stride, dilation);
        let x_shape2: Vec<SVal> = x.shape_ctx(ctx).into_vec();
        let oyx: Vec<SVal> = x_shape2[2..2 + hw_len].to_vec();
        let rcout = cout_i / groups;

        // x.reshape(bs, groups, cin, 1, *oyx, *HW).expand(bs, groups, cin, rcout, *oyx, *HW)
        let mut sh1: Vec<SVal> = vec![bs.clone(), SVal::I(groups as i128), cin.clone(), SVal::I(1)];
        sh1.extend(oyx.iter().cloned());
        sh1.extend(hw.iter().map(|&h| SVal::I(h as i128)));
        let x = x.reshape_ctx(ctx, sh1);
        let mut sh2: Vec<SVal> = vec![
            bs.clone(),
            SVal::I(groups as i128),
            cin.clone(),
            SVal::I(rcout as i128),
        ];
        sh2.extend(oyx.iter().cloned());
        sh2.extend(hw.iter().map(|&h| SVal::I(h as i128)));
        let x = x.expand_ctx(ctx, sh2);

        // .permute(0, 1, 3, 4..4+len(oyx), 2, 4+len(oyx)..)
        let mut perm: Vec<i64> = vec![0, 1, 3];
        perm.extend((0..hw_len).map(|i| (4 + i) as i64));
        perm.push(2);
        perm.extend((0..hw_len).map(|i| (4 + hw_len + i) as i64));
        let x = x.permute_ctx(ctx, &perm);

        // weight.reshape(1, groups, rcout, *[1]*len(oyx), cin, *HW)
        let mut wsh: Vec<SVal> = vec![SVal::I(1), SVal::I(groups as i128), SVal::I(rcout as i128)];
        wsh.extend(std::iter::repeat_n(SVal::I(1), hw_len));
        wsh.push(cin);
        wsh.extend(hw.iter().map(|&h| SVal::I(h as i128)));
        let w = weight.reshape_ctx(ctx, wsh);

        // (x*w).sum([-1-i for i in range(1+len(oyx))], keepdim=True, dtype).reshape(bs, cout, *oyx)
        let prod = x.mul_ctx(ctx, &w);
        let prod_ndim = prod.shape_ctx(ctx).len();
        let axes: Vec<usize> = (0..1 + hw_len).map(|i| prod_ndim - 1 - i).collect();
        let ret = prod.sum_ctx(ctx, Some(&axes), true, dtype);
        let mut osh: Vec<SVal> = vec![bs, cout];
        osh.extend(oyx);
        let ret = ret.reshape_ctx(ctx, osh);

        match bias {
            Some(b) => {
                let mut bsh: Vec<SVal> = vec![SVal::I(1), SVal::I(cout_i as i128)];
                bsh.extend(std::iter::repeat_n(SVal::I(1), hw_len));
                let b_r = b.reshape_ctx(ctx, bsh);
                ret.add_ctx(ctx, &b_r)
            }
            None => ret,
        }
    }

    /// `Tensor.conv_transpose2d(weight, bias, groups, stride, dilation, padding,
    /// output_padding)` (op.py:1607-1650). Works for any number of spatial dims.
    #[allow(clippy::too_many_arguments)]
    pub fn conv_transpose2d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        groups: i64,
        stride: i64,
        dilation: i64,
        padding: &[i64],
        output_padding: i64,
    ) -> Tensor {
        self.conv_transpose2d_ctx(
            &mut self.ctx.borrow_mut(),
            weight,
            bias,
            groups,
            stride,
            dilation,
            padding,
            output_padding,
        )
    }
    #[doc(hidden)]
    pub fn conv_transpose2d_ctx(
        &self,
        ctx: &mut UOpCtx,
        weight: &Tensor,
        bias: Option<&Tensor>,
        groups: i64,
        stride: i64,
        dilation: i64,
        padding: &[i64],
        output_padding: i64,
    ) -> Tensor {
        // x, w = self, weight.unflatten(0, (groups, -1)).transpose(1, 2).flip(*range(3, len+1))
        let w = weight
            .unflatten_ctx(ctx, 0, &[groups, -1])
            .transpose_ctx(ctx, 1, 2);
        let w_ndim = w.shape_ctx(ctx).len();
        let w = if w_ndim > 3 {
            w.flip_ctx(ctx, &(3..w_ndim).collect::<Vec<_>>())
        } else {
            w
        };

        // HW = weight.shape[2:]
        let w_shape: Vec<SVal> = weight.shape_ctx(ctx).into_vec();
        let hw: Vec<i64> = w_shape[2..]
            .iter()
            .map(|s| s.clone().as_i64().unwrap_or(0))
            .collect();
        let nhw = hw.len();

        // padding = flat_to_grouped(resolve_pool_pads(padding, len(HW)))
        let pads = flat_to_grouped(&resolve_pool_pads(padding, nhw));

        // stride, dilation, output_padding = [make_tuple(x, len(HW)) for x in (...)]
        let stride_v = vec![stride; nhw];
        let dilation_v = vec![dilation; nhw];
        let opad_v = vec![output_padding; nhw];

        // if any(s>1 for s in stride): insert zeros between input positions, then shrink
        let mut x = self.clone();
        if stride_v.iter().any(|&s| s > 1) {
            // x.reshape(None, None, *flatten((k,1) for k in x.shape[2:]))
            let xs: Vec<SVal> = x.shape_ctx(ctx).into_vec();
            let mut sh: Vec<SVal> = vec![xs[0].clone(), xs[1].clone()];
            for k in &xs[2..] {
                sh.push(k.clone());
                sh.push(SVal::I(1));
            }
            x = x.reshape_ctx(ctx, sh);
            // x.pad((None, None, *flatten((None,(0,s-1)) for s in stride)))
            let mut parg: Vec<(i64, i64)> = vec![(0, 0), (0, 0)];
            for &s in &stride_v {
                parg.push((0, 0));
                parg.push((0, s - 1));
            }
            x = x.pad_ctx(ctx, &parg);
            // x.reshape(None, None, *[k*s for k,s in zip(x.shape[2::2], stride)])
            let xs2: Vec<SVal> = x.shape_ctx(ctx).into_vec();
            let mut sh2: Vec<SVal> = vec![xs2[0].clone(), xs2[1].clone()];
            for (i, &s) in stride_v.iter().enumerate() {
                let k = xs2[2 + 2 * i].clone();
                sh2.push(ctx.sval_mul(k, SVal::I(s as i128)));
            }
            x = x.reshape_ctx(ctx, sh2);
            // x.shrink_to(None, None, *[k-(s-1) for k,s in zip(x.shape[2:], stride)])
            let xs3: Vec<SVal> = x.shape_ctx(ctx).into_vec();
            let mut sht: Vec<SVal> = vec![SVal::I(-1), SVal::I(-1)];
            for (i, &s) in stride_v.iter().enumerate() {
                let k = xs3[2 + i].clone();
                sht.push(ctx.sval_sub(k, SVal::I((s - 1) as i128)));
            }
            x = x.shrink_to_ctx(ctx, &sht);
        }

        // padding = flatten((((k-1)*d-pB,(k-1)*d-pA+op) for ... in reversed(...)))
        let mut padding_flat: Vec<i64> = Vec::with_capacity(2 * nhw);
        for i in (0..nhw).rev() {
            let k = hw[i];
            let d = dilation_v[i];
            let (pb, pa) = pads[i];
            let op = opad_v[i];
            padding_flat.push((k - 1) * d - pb);
            padding_flat.push((k - 1) * d - pa + op);
        }

        // x.conv2d(w.flatten(end_dim=1), groups=groups, bias=bias, dilation=dilation, padding=padding)
        let wflat = w.flatten_ctx(ctx, 0, 1);
        x.conv2d_ctx(ctx, &wflat, bias, groups, 1, dilation, &padding_flat, None)
    }

    /// `Tensor.max_pool2d(kernel_size, stride, dilation, padding, ceil_mode,
    /// return_indices)` (op.py:1338-1381). Returns `(pooled, indices?)`.
    #[allow(clippy::too_many_arguments)]
    pub fn max_pool2d(
        &self,
        kernel_size: &[i64],
        stride: Option<&[i64]>,
        dilation: &[i64],
        padding: &[i64],
        ceil_mode: bool,
        return_indices: bool,
    ) -> (Tensor, Option<Tensor>) {
        self.max_pool2d_ctx(
            &mut self.ctx.borrow_mut(),
            kernel_size,
            stride,
            dilation,
            padding,
            ceil_mode,
            return_indices,
        )
    }
    #[doc(hidden)]
    pub fn max_pool2d_ctx(
        &self,
        ctx: &mut UOpCtx,
        kernel_size: &[i64],
        stride: Option<&[i64]>,
        dilation: &[i64],
        padding: &[i64],
        ceil_mode: bool,
        return_indices: bool,
    ) -> (Tensor, Option<Tensor>) {
        let k = kernel_size;
        let s_: Vec<i64> = stride.map(|s| s.to_vec()).unwrap_or_else(|| k.to_vec());
        let d_ = dilation.to_vec();
        let shape: Vec<SVal> = self.shape_ctx(ctx).into_vec();
        let mut pads = resolve_pool_pads(padding, k.len());
        if ceil_mode {
            pads = apply_ceil_mode(&shape, &pads, k, &s_, &d_);
        }
        let dt = self.dtype_ctx(ctx);
        let grouped = flat_to_grouped(&pads);
        let mut p_x: Vec<(Option<i64>, Option<i64>)> =
            vec![(Some(0), Some(0)); shape.len() - k.len()];
        p_x.extend(grouped.iter().map(|(a, b)| (Some(*a), Some(*b))));
        let pooled = self
            .pad_constant_ctx(ctx, &p_x, dtype::dtype_min(dt))
            .pool_vec_ctx(ctx, k, &s_, &d_);

        let ndim = pooled.shape_ctx(ctx).len();
        let axis: Vec<usize> = (ndim - k.len()..ndim).collect();
        if !return_indices {
            return (pooled.max_ctx(ctx, Some(&axis), false), None);
        }

        // return_indices path (op.py:1377-1381)
        let spatial: Vec<i64> = shape[shape.len() - k.len()..]
            .iter()
            .map(|s| s.clone().as_i64().unwrap_or(0))
            .collect();
        let spatial_sz: i64 = spatial.iter().product();
        let idx = Tensor::arange(
            &mut *ctx,
            spatial_sz as f64,
            Some(0.0),
            -1.0,
            Some(dtype::default_int()),
        )
        .reshape_ctx(ctx, spatial.clone());
        let pooled_max = pooled.max_ctx(ctx, Some(&axis), true);
        let m = pooled.eq_ctx(ctx, pooled_max);
        let idx_ndim = idx.shape_ctx(ctx).len();
        let mut idx_p: Vec<(Option<i64>, Option<i64>)> =
            vec![(Some(0), Some(0)); idx_ndim - k.len()];
        idx_p.extend(grouped.iter().map(|(a, b)| (Some(*a), Some(*b))));
        let idx_pad = idx.pad_constant_ctx(ctx, &idx_p, dtype::dtype_min(idx.dtype_ctx(ctx)));
        let idx_pooled = idx_pad.pool_vec_ctx(ctx, k, &s_, &d_);
        let idx2 = m.mul_ctx(ctx, &idx_pooled);
        let max = pooled.max_ctx(ctx, Some(&axis), false);
        let idx_out = idx2
            .max_ctx(ctx, Some(&axis), false)
            .neg_ctx(ctx)
            .add_ctx(ctx, spatial_sz);
        (max, Some(idx_out))
    }

    /// `Tensor.sparse_categorical_crossentropy(Y, ignore_index, label_smoothing,
    /// reduction)` (op.py:1725-1748). `self` is logits with the class axis last.
    pub fn sparse_categorical_crossentropy(
        &self,
        y: &Tensor,
        ignore_index: i64,
        label_smoothing: f64,
        reduction: &str,
    ) -> Tensor {
        self.sparse_categorical_crossentropy_ctx(
            &mut self.ctx.borrow_mut(),
            y,
            ignore_index,
            label_smoothing,
            reduction,
        )
    }
    #[doc(hidden)]
    pub fn sparse_categorical_crossentropy_ctx(
        &self,
        ctx: &mut UOpCtx,
        y: &Tensor,
        ignore_index: i64,
        label_smoothing: f64,
        reduction: &str,
    ) -> Tensor {
        assert!(
            (0.0..=1.0).contains(&label_smoothing),
            "TGERR: label_smoothing must be in [0.0, 1.0]"
        );
        let self_shape: Vec<SVal> = self.shape_ctx(ctx).into_vec();
        let ndim = self_shape.len();
        let class_axis = ndim - 1;
        let log_probs = super::nn::log_softmax_ctx(self, ctx, Some(class_axis));

        let loss_mask = if ignore_index != -1 {
            y.ne_ctx(ctx, ignore_index)
        } else {
            let u = ctx.const_like_dt(y.uop().clone(), ConstVal::Bool(true), Some(dtype::BOOL));
            Tensor::from_uop(ctx, u)
        };
        let num_classes = self_shape[class_axis].clone().as_i64().unwrap_or(0);
        let one_hot = y
            .unsqueeze_ctx(ctx, -1)
            .one_hot_along_dim_ctx(ctx, num_classes, -1);
        let mask_unsq = loss_mask.unsqueeze_ctx(ctx, -1);
        let y_mask = one_hot.mul_ctx(ctx, &mask_unsq);

        let lp_mean = log_probs.mean_ctx(ctx, Some(&[class_axis]), false);
        let smoothing = lp_mean
            .mul_ctx(ctx, &loss_mask)
            .mul_ctx(ctx, label_smoothing);
        let lp_y = log_probs
            .mul_ctx(ctx, &y_mask)
            .sum_ctx(ctx, Some(&[class_axis]), false, None);
        let unreduced = lp_y
            .mul_ctx(ctx, 1.0 - label_smoothing)
            .add_ctx(ctx, &smoothing);
        match reduction {
            "mean" => {
                let num = unreduced.sum_ctx(ctx, None, false, None).neg_ctx(ctx);
                let den = loss_mask.sum_ctx(ctx, None, false, None);
                num.div_ctx(ctx, &den)
            }
            "sum" => unreduced.sum_ctx(ctx, None, false, None).neg_ctx(ctx),
            "none" => unreduced.neg_ctx(ctx),
            other => panic!("TGERR: unsupported reduction {other:?}"),
        }
    }
}
