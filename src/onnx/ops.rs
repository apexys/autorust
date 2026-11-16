//! ONNX operator implementations. Mirrors `tinygrad/nn/onnx.py`'s op table for
//! the T0/T1 surface; every handler takes already-resolved inputs.

use super::pb::{Attribute, Node, OnnxError};
use super::{OnnxModel, Result};
use crate::dtype::{self, ConstVal};
use crate::tensor::{Ix, Tensor};
use crate::uop::ctx::UOpCtx;
use crate::uop::graph::UopId;

fn inp(ins: &[Option<Tensor>], i: usize) -> Result<&Tensor> {
    ins.get(i)
        .and_then(|x| x.as_ref())
        .ok_or_else(|| OnnxError::Runtime(format!("missing input #{i}")))
}

fn opt(ins: &[Option<Tensor>], i: usize) -> Option<&Tensor> {
    ins.get(i).and_then(|x| x.as_ref())
}


fn gather_nd_0(x: &Tensor, idx: &Tensor) -> Result<Tensor> {
    let rank = idx.shape().as_slice().len();
    if rank == 0 {
        return Err(OnnxError::Unsupported("GatherND with scalar indices".into()));
    }
    let k = idx.shape().as_slice()[rank - 1]
        .clone()
        .as_i64()
        .unwrap_or(0);
    if k == 0 {
        // `indices[..., :0]` selects the whole `x` once per index prefix:
        // output shape is `indices.shape[:-1] + x.shape` (broadcast).
        let ishape = idx.shape();
        let prefix = &ishape.as_slice()[..rank - 1];
        let xshape: Vec<i64> = x
            .shape()
            .as_slice()
            .iter()
            .map(|s| s.clone().as_i64().unwrap_or(0))
            .collect();
        let mut out_shape: Vec<i64> = prefix
            .iter()
            .map(|s| s.clone().as_i64().unwrap_or(0))
            .collect();
        out_shape.extend_from_slice(&xshape);
        let mut target = vec![1i64; prefix.len()];
        target.extend_from_slice(&xshape);
        return Ok(x.reshape(target).expand(out_shape));
    }
    let (idx, squeeze_out) = if rank == 1 {
        (idx.unsqueeze(0), true)
    } else {
        (idx.clone(), false)
    };
    let parts = idx.split(&vec![1i64; k as usize], -1);
    let mut ixs: Vec<Ix> = Vec::with_capacity(k as usize);
    for p in parts {
        ixs.push(Ix::Tensor(p.squeeze(Some(-1))));
    }
    let out = x.getitem(&ixs);
    Ok(if squeeze_out {
        out.squeeze(Some(0))
    } else {
        out
    })
}


/// Insert `d-1` zeros between the taps of each spatial kernel axis (dilation
/// expansion). The resulting kernel has size `(k-1)*d+1`; pair it with a
/// dilation-1 convolution.
fn expand_kernel_dilation(w: &Tensor, dilations: &[i64]) -> Tensor {
    let mut cur = w.clone();
    for (i, &d) in dilations.iter().enumerate() {
        if d <= 1 {
            continue;
        }
        let ax = 2 + i;
        let shp: Vec<i64> = cur
            .shape()
            .as_slice()
            .iter()
            .map(|s| s.clone().as_i64().unwrap_or(0))
            .collect();
        let k = shp[ax];
        // split the axis into (k, 1)
        let mut split: Vec<i64> = Vec::with_capacity(shp.len() + 1);
        for (j, &v) in shp.iter().enumerate() {
            split.push(v);
            if j == ax {
                split.push(1);
            }
        }
        cur = cur.reshape(split.clone());
        // zero-pad the singleton axis to d
        let mut parg = vec![(0i64, 0i64); split.len()];
        parg[split.len() - 1] = (0, d - 1);
        cur = cur.pad(&parg);
        // merge (k, d) back into one axis of length k*d
        let mut merged = split.clone();
        merged[ax] = k * d;
        merged.remove(ax + 1);
        cur = cur.reshape(merged);
        // keep only the taps up to (k-1)*d
        let mut target: Vec<crate::uop::sint::SVal> = shp
            .iter()
            .map(|v| crate::uop::sint::SVal::I(*v as i128))
            .collect();
        target[ax] = crate::uop::sint::SVal::I(((k - 1) * d + 1) as i128);
        cur = cur.shrink_to(&target);
    }
    cur
}

/// Zero-insert the input along each spatial axis with stride > 1, so a
/// transposed convolution of stride 1 sees the same samples at `i*s`.
fn zero_insert_strides(x: &Tensor, strides: &[i64]) -> Tensor {
    if strides.iter().all(|&s| s == 1) {
        return x.clone();
    }
    let xs: Vec<i64> = x
        .shape()
        .as_slice()
        .iter()
        .map(|s| s.clone().as_i64().unwrap_or(0))
        .collect();
    // split every spatial axis into (k, 1)
    let mut split: Vec<i64> = vec![xs[0], xs[1]];
    for k in &xs[2..] {
        split.push(*k);
        split.push(1);
    }
    let mut y = x.reshape(split.clone());
    // pad each singleton block to `stride`
    let mut parg: Vec<(i64, i64)> = vec![(0, 0), (0, 0)];
    for &s in strides {
        parg.push((0, 0));
        parg.push((0, s - 1));
    }
    y = y.pad(&parg);
    // collapse each (k, stride) block into k*stride
    let ys: Vec<i64> = y
        .shape()
        .as_slice()
        .iter()
        .map(|s| s.clone().as_i64().unwrap_or(0))
        .collect();
    let mut merged: Vec<i64> = vec![ys[0], ys[1]];
    for (i, &s) in strides.iter().enumerate() {
        merged.push(ys[2 + 2 * i] * s);
    }
    y = y.reshape(merged);
    // keep (k-1)*s+1 samples: the interleaved positions with the final pad cut
    let mut target: Vec<crate::uop::sint::SVal> =
        vec![crate::uop::sint::SVal::I(-1), crate::uop::sint::SVal::I(-1)];
    for (i, &s) in strides.iter().enumerate() {
        let k = xs[2 + i];
        target.push(crate::uop::sint::SVal::I(((k - 1) * s + 1) as i128));
    }
    y.shrink_to(&target)
}

impl OnnxModel {
    pub(crate) fn eval(&mut self, node: &Node, ins: &[Option<Tensor>]) -> Result<Vec<Tensor>> {
        let op = node.op_type.as_str();
        let one = |t: Tensor| Ok(vec![t]);
        match op {
            // ---------------- constants / identity ----------------
            "Identity" => one(inp(ins, 0)?.clone()),
            "Constant" => {
                if let Some(v) = node.attrs.get("value").and_then(Attribute::as_tensor) {
                    return one(super::tensor_from_proto(
                        &self.ctx,
                        v,
                        self.origin(),
                        self.device(),
                    )?);
                }
                if let Some(v) = node.attrs.get("value_float").and_then(Attribute::as_f64) {
                    return one(Tensor::full_const(
                        &self.ctx,
                        &[] as &[i64],
                        ConstVal::Float(v),
                        Some(dtype::FLOAT32),
                    ));
                }
                if let Some(v) = node.attrs.get("value_floats").and_then(Attribute::as_f64s) {
                    if v.is_empty() {
                        return Err(OnnxError::Unsupported(
                            "Constant with empty value_floats".into(),
                        ));
                    }
                    return one(Tensor::from_slice(
                        &self.ctx,
                        v.as_slice(),
                        Some(dtype::FLOAT32),
                        self.device(),
                    ));
                }
                if let Some(v) = node.attrs.get("value_int").and_then(Attribute::as_i64) {
                    return one(Tensor::full_const(
                        &self.ctx,
                        &[] as &[i64],
                        ConstVal::Int(v as i128),
                        Some(dtype::INT64),
                    ));
                }
                if let Some(v) = node.attrs.get("value_ints").and_then(Attribute::as_i64s) {
                    if v.is_empty() {
                        return Err(OnnxError::Unsupported(
                            "Constant with empty value_ints".into(),
                        ));
                    }
                    return one(Tensor::from_slice(
                        &self.ctx,
                        v,
                        Some(dtype::INT64),
                        self.device(),
                    ));
                }
                Err(OnnxError::Unsupported(
                    "Constant with value_string/sparse_value".into(),
                ))
            }
            "ConstantOfShape" => {
                let shape = self.i64s(inp(ins, 0)?)?;
                let value = if let Some(v) = opt(ins, 1) {
                    v.clone()
                } else if let Some(v) = node.attrs.get("value").and_then(Attribute::as_tensor) {
                    super::tensor_from_proto(&self.ctx, v, self.origin(), self.device())?
                } else {
                    Tensor::full_const(
                        &self.ctx,
                        &[] as &[i64],
                        ConstVal::Float(0.0),
                        Some(dtype::FLOAT32),
                    )
                };
                if shape == [0] {
                    let dt = value.dtype();
                    return one(Tensor::from_slice(
                        &self.ctx,
                        &[] as &[u8],
                        Some(dt),
                        self.device(),
                    ));
                }
                one(value.reshape(&[] as &[i64]).expand(shape))
            }
            "Shape" => {
                let x = inp(ins, 0)?;
                let n = x.shape().as_slice().len() as i64;
                let start = resolve_index(node.attr_i64("start", 0), n);
                let end_raw = node.attr_i64("end", i64::MAX);
                let end = if end_raw == i64::MAX {
                    n
                } else {
                    resolve_index(end_raw, n)
                }
                .min(n);
                let dims: Vec<i64> = x
                    .shape()
                    .as_slice()
                    .iter()
                    .map(|s| s.clone().as_i64().unwrap_or(0))
                    .collect();
                let start = start.clamp(0, n) as usize;
                let end = end.clamp(start as i64, n) as usize;
                one(Tensor::from_slice(
                    &self.ctx,
                    &dims[start..end],
                    Some(dtype::INT64),
                    self.device(),
                ))
            }
            "Size" => {
                let x = inp(ins, 0)?;
                one(Tensor::full_const(
                    &self.ctx,
                    &[] as &[i64],
                    ConstVal::Int(x.numel() as i128),
                    Some(dtype::INT64),
                ))
            }

            // ---------------- unary ----------------
            "Abs" => one(inp(ins, 0)?.abs()),
            "Neg" => one(inp(ins, 0)?.neg()),
            "Exp" => one(inp(ins, 0)?.exp()),
            "Log" => one(inp(ins, 0)?.log()),
            "Sqrt" => one(inp(ins, 0)?.sqrt()),
            "Reciprocal" => one(inp(ins, 0)?.reciprocal()),
            "Ceil" => one(inp(ins, 0)?.ceil()),
            "Floor" => one(inp(ins, 0)?.floor()),
            "Round" => one(inp(ins, 0)?.round()),
            "Sign" => one(inp(ins, 0)?.sign()),
            "Erf" => one(inp(ins, 0)?.erf()),
            "Sin" => one(crate::tensor::nn::sin(inp(ins, 0)?)),
            "Cos" => one(crate::tensor::nn::cos(inp(ins, 0)?)),
            "Tan" => one(inp(ins, 0)?.tan()),
            "Tanh" => one(crate::tensor::nn::tanh(inp(ins, 0)?)),
            "Sigmoid" => one(crate::tensor::nn::sigmoid(inp(ins, 0)?)),
            "Relu" => one(crate::tensor::nn::relu(inp(ins, 0)?)),
            "LeakyRelu" => {
                let alpha = node.attr_f64("alpha", 0.01);
                one(inp(ins, 0)?.leaky_relu(alpha))
            }
            "Elu" => {
                let alpha = node.attr_f64("alpha", 1.0);
                one(inp(ins, 0)?.elu(alpha))
            }
            "Celu" => {
                let alpha = node.attr_f64("alpha", 1.0);
                one(inp(ins, 0)?.celu(alpha))
            }
            "Selu" => {
                let alpha = node.attr_f64("alpha", 1.673_263_242_354_377_2);
                let gamma = node.attr_f64("gamma", 1.050_700_987_355_480_5);
                one(inp(ins, 0)?.selu(alpha, gamma))
            }
            "Softplus" => one(inp(ins, 0)?.softplus(1.0)),
            "Softsign" => one(inp(ins, 0)?.softsign()),
            "Mish" => one(inp(ins, 0)?.mish()),
            "HardSwish" => one(inp(ins, 0)?.hardswish()),
            "HardSigmoid" => {
                let alpha = node.attr_f64("alpha", 0.2);
                let beta = node.attr_f64("beta", 0.5);
                let x = inp(ins, 0)?;
                let y = x.mul(alpha).add(beta);
                one(y.clip(Some(0.0), Some(1.0)))
            }
            "Swish" => {
                let alpha = node.attr_f64("alpha", 1.0);
                let x = inp(ins, 0)?;
                one(x.mul(crate::tensor::nn::sigmoid(&x.mul(alpha))))
            }
            "FastGelu" => {
                let x = inp(ins, 0)?;
                let z = match opt(ins, 1) {
                    Some(bias) => x.add(bias),
                    None => x.clone(),
                };
                one(crate::tensor::nn::gelu(&z))
            }
            "BiasGelu" => {
                let x = inp(ins, 0)?;
                let bias = inp(ins, 1)?;
                let z = x.add(bias);
                let approximate = node.attr_str("approximate", "none");
                if approximate == "tanh" {
                    one(crate::tensor::nn::gelu(&z))
                } else {
                    let half = z.const_like(ConstVal::Float(0.5), None);
                    let one_ = z.const_like(ConstVal::Float(1.0), None);
                    let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
                    one(z.mul(&half).mul(one_.add(z.mul(inv_sqrt2).erf())))
                }
            }
            "PRelu" => {
                let x = inp(ins, 0)?;
                let slope = inp(ins, 1)?;
                one(x.gt(0).where_(x, x.mul(slope)))
            }
            "ThresholdedRelu" => {
                let alpha = node.attr_f64("alpha", 1.0);
                let x = inp(ins, 0)?;
                one(x.gt(alpha).where_(x, 0.0))
            }
            "Gelu" => {
                let x = inp(ins, 0)?;
                let approximate = node.attr_str("approximate", "none");
                if approximate == "tanh" {
                    one(crate::tensor::nn::gelu(x))
                } else {
                    let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
                    let half = x.const_like(ConstVal::Float(0.5), None);
                    let one_ = x.const_like(ConstVal::Float(1.0), None);
                    one(x.mul(&half).mul(one_.add(x.mul(inv_sqrt2).erf())))
                }
            }
            "Softmax" => {
                let axis = self.axis(
                    node,
                    "axis",
                    default_softmax_axis(self.opset(node)),
                    inp(ins, 0)?.shape().as_slice().len() as i64,
                )?;
                let t = crate::tensor::nn::softmax(inp(ins, 0)?, Some(axis));
                one(t)
            }
            "LogSoftmax" => {
                let axis = self.axis(
                    node,
                    "axis",
                    default_softmax_axis(self.opset(node)),
                    inp(ins, 0)?.shape().as_slice().len() as i64,
                )?;
                one(crate::tensor::nn::log_softmax(inp(ins, 0)?, Some(axis)))
            }
            "NegativeLogLikelihoodLoss" => self.nll_loss_op(node, ins, false),
            "SoftmaxCrossEntropyLoss" => self.nll_loss_op(node, ins, true),
            "Clip" => {
                let x = inp(ins, 0)?;
                let lo = match opt(ins, 1) {
                    Some(t) => Some(self.f64_scalar(t)?),
                    None => node.attrs.get("min").and_then(Attribute::as_f64),
                };
                let hi = match opt(ins, 2) {
                    Some(t) => Some(self.f64_scalar(t)?),
                    None => node.attrs.get("max").and_then(Attribute::as_f64),
                };
                if lo.is_none() && hi.is_none() {
                    one(x.clone())
                } else {
                    one(x.clip(lo, hi))
                }
            }
            "IsNaN" => one(inp(ins, 0)?.isnan()),
            "IsInf" => {
                let x = inp(ins, 0)?;
                let neg = node.attr_i64("detect_negative", 1) != 0;
                let pos = node.attr_i64("detect_positive", 1) != 0;
                let out = match (neg, pos) {
                    (true, true) => x.isinf(),
                    (true, false) => x.eq(f64::NEG_INFINITY),
                    (false, true) => x.eq(f64::INFINITY),
                    (false, false) => x.isnan().logical_not().logical_and(&x.isnan()),
                };
                one(out)
            }
            "Not" => one(inp(ins, 0)?.logical_not()),
            "Shrink" => {
                let x = inp(ins, 0)?;
                let bias = node.attr_f64("bias", 0.0);
                let lambd = node.attr_f64("lambd", 0.5);
                let pos = x.gt(lambd).where_(x.sub(bias), 0.0);
                let neg = x.lt(-lambd).where_(x.add(bias), 0.0);
                one(pos.add(&neg))
            }
            "BitwiseNot" => {
                let x = inp(ins, 0)?;
                one(x.neg().sub(1.0))
            }

            // ---------------- binary ----------------
            "Add" => one(inp(ins, 0)?.add(inp(ins, 1)?)),
            "Sub" => one(inp(ins, 0)?.sub(inp(ins, 1)?)),
            "Mul" => one(inp(ins, 0)?.mul(inp(ins, 1)?)),
            "Div" => {
                let a = inp(ins, 0)?;
                let b = inp(ins, 1)?;
                if dtype::is_int(a.dtype()) {
                    one(self.binop_raw(a, b, |ctx, x, y| {
                        crate::tensor::elementwise::div(ctx, x, y, Some("trunc"))
                    }))
                } else {
                    one(a.div(b))
                }
            }
            "Pow" => {
                let a = inp(ins, 0)?;
                let b = inp(ins, 1)?;
                // Small integer exponents are folded to multiplications: the
                // CUDA/C renderers have no ALU Pow, and shape math in exported
                // models is full of `x**2`. A scalar base also folds to a host
                // constant (ONNX shape subgraphs are scalar arithmetic).
                if a.numel() == 1 && b.numel() == 1 {
                    let av = self.f64_scalar(a)?;
                    let bv = self.f64_scalar(b)?;
                    let dt = a.dtype();
                    let shape: Vec<i64> = a
                        .shape()
                        .as_slice()
                        .iter()
                        .map(|s| s.clone().as_i64().unwrap_or(0))
                        .collect();
                    let c = if dtype::is_int(dt) || dtype::is_bool(dt) {
                        Tensor::full_const(
                            &self.ctx,
                            &[] as &[i64],
                            ConstVal::Int(av.powf(bv) as i128),
                            Some(dt),
                        )
                    } else {
                        Tensor::full_const(
                            &self.ctx,
                            &[] as &[i64],
                            ConstVal::Float(av.powf(bv)),
                            Some(dt),
                        )
                    };
                    one(if shape.is_empty() {
                        c
                    } else {
                        c.reshape(shape)
                    })
                } else if b.numel() == 1 {
                    let e = self.f64_scalar(b)?;
                    if e.fract() == 0.0 && (0.0..=16.0).contains(&e) {
                        one(pow_positive(a, e as u32))
                    } else if e.fract() == 0.0 && (-16.0..0.0).contains(&e) {
                        one(pow_positive(a, (-e) as u32).reciprocal())
                    } else {
                        one(xpow(a, b))
                    }
                } else {
                    one(xpow(a, b))
                }
            }
            "Mod" => {
                let a = inp(ins, 0)?;
                let b = inp(ins, 1)?;
                if node.attr_i64("fmod", 0) != 0 {
                    one(self.binop_raw(a, b, crate::tensor::elementwise::fmod))
                } else {
                    one(a.mod_(b))
                }
            }
            "Min" => {
                let mut acc = inp(ins, 0)?.clone();
                for i in 1..ins.len() {
                    acc = acc.minimum(inp(ins, i)?);
                }
                one(acc)
            }
            "Max" => {
                let mut acc = inp(ins, 0)?.clone();
                for i in 1..ins.len() {
                    acc = acc.maximum(inp(ins, i)?);
                }
                one(acc)
            }
            "Sum" => {
                let mut acc = inp(ins, 0)?.clone();
                for i in 1..ins.len() {
                    acc = acc.add(inp(ins, i)?);
                }
                one(acc)
            }
            "Mean" => {
                let n = ins.len() as f64;
                let mut acc = inp(ins, 0)?.clone();
                for i in 1..ins.len() {
                    acc = acc.add(inp(ins, i)?);
                }
                one(acc.div(n))
            }
            "Equal" => one(inp(ins, 0)?.eq(inp(ins, 1)?)),
            "Less" => one(inp(ins, 0)?.lt(inp(ins, 1)?)),
            "LessOrEqual" => one(inp(ins, 0)?.le(inp(ins, 1)?)),
            "Greater" => one(inp(ins, 0)?.gt(inp(ins, 1)?)),
            "GreaterOrEqual" => one(inp(ins, 0)?.ge(inp(ins, 1)?)),
            "And" => one(inp(ins, 0)?.logical_and(inp(ins, 1)?)),
            "Or" => one(inp(ins, 0)?.logical_or(inp(ins, 1)?)),
            "Xor" => one(inp(ins, 0)?.logical_xor(inp(ins, 1)?)),
            "BitwiseAnd" => one(inp(ins, 0)?.logical_and(inp(ins, 1)?)),
            "BitwiseOr" => one(inp(ins, 0)?.logical_or(inp(ins, 1)?)),
            "BitwiseXor" => one(inp(ins, 0)?.logical_xor(inp(ins, 1)?)),
            "Where" => one(inp(ins, 0)?.where_(inp(ins, 1)?, inp(ins, 2)?)),

            // ---------------- cast ----------------
            "Cast" => {
                let to = node.attr_i64("to", -1);
                let dt = super::pb::onnx_dtype(to as i32)?;
                one(inp(ins, 0)?.cast(dt))
            }
            "CastLike" => {
                let dt = inp(ins, 1)?.dtype();
                one(inp(ins, 0)?.cast(dt))
            }

            // ---------------- movement ----------------
            "Reshape" => {
                let x = inp(ins, 0)?;
                let mut shape = self.i64s(inp(ins, 1)?)?;
                let allowzero = node.attr_i64("allowzero", 0) != 0;
                if !allowzero {
                    let src = x.shape();
                    let src = src.as_slice();
                    for (i, d) in shape.iter_mut().enumerate() {
                        if *d == 0 {
                            *d = src.get(i).and_then(|s| s.clone().as_i64()).unwrap_or(0);
                        }
                    }
                }
                one(x.reshape(shape))
            }
            "Flatten" => {
                let x = inp(ins, 0)?;
                let ndim = x.shape().as_slice().len() as i64;
                let axis = resolve_index(node.attr_i64("axis", 1), ndim);
                let outer: i64 = x
                    .shape()
                    .as_slice()
                    .iter()
                    .take(axis as usize)
                    .map(|s| s.clone().as_i64().unwrap_or(1))
                    .product();
                one(x.reshape([outer.max(1), -1]))
            }
            "Squeeze" => {
                let x = inp(ins, 0)?;
                let axes = match opt(ins, 1) {
                    Some(t) => Some(self.i64s(t)?),
                    None => node.attr_i64s("axes"),
                };
                match axes {
                    None => one(x.squeeze(None)),
                    Some(axes) => {
                        let mut out = x.clone();
                        let mut ax = axes;
                        ax.sort_unstable_by(|a, b| b.cmp(a));
                        for d in ax {
                            out = out.squeeze(Some(d));
                        }
                        one(out)
                    }
                }
            }
            "Unsqueeze" => {
                let x = inp(ins, 0)?;
                let axes = match opt(ins, 1) {
                    Some(t) => self.i64s(t)?,
                    None => node
                        .attr_i64s("axes")
                        .ok_or_else(|| OnnxError::Runtime("Unsqueeze needs axes".into()))?,
                };
                let mut out = x.clone();
                let mut ax = axes;
                ax.sort_unstable();
                for d in ax {
                    out = out.unsqueeze(d);
                }
                one(out)
            }
            "Transpose" => {
                let x = inp(ins, 0)?;
                let ndim = x.shape().as_slice().len() as i64;
                let perm = node
                    .attr_i64s("perm")
                    .unwrap_or_else(|| (0..ndim).rev().collect());
                one(x.permute(&perm))
            }
            "Concat" => {
                let axis = node.attr_i64("axis", 0);
                let first = inp(ins, 0)?;
                let rest: Vec<Tensor> = ins[1..]
                    .iter()
                    .map(|t| {
                        t.clone()
                            .ok_or_else(|| OnnxError::Runtime("Concat empty input".into()))
                    })
                    .collect::<Result<_>>()?;
                one(first.cat(&rest, axis))
            }
            "Split" => {
                let x = inp(ins, 0)?;
                let axis = node.attr_i64("axis", 0);
                let axis_nd = resolve_axis(x, axis);
                let total = x.shape().as_slice()[axis_nd].clone().as_i64().unwrap_or(0);
                let split = match opt(ins, 1) {
                    Some(t) => self.i64s(t)?,
                    None => match node.attr_i64s("split") {
                        Some(v) if !v.is_empty() => v,
                        _ => {
                            let n = node.outputs.len().max(1) as i64;
                            (0..n)
                                .map(|i| total / n + i64::from(i < total % n))
                                .collect()
                        }
                    },
                };
                Ok(x.split(&split, axis))
            }
            "Expand" => {
                let x = inp(ins, 0)?;
                let target = self.i64s(inp(ins, 1)?)?;
                let src: Vec<i64> = x
                    .shape()
                    .as_slice()
                    .iter()
                    .map(|s| s.clone().as_i64().unwrap_or(1))
                    .collect();
                one(x.expand(broadcast_shape(&src, &target)?))
            }
            "Tile" => {
                let x = inp(ins, 0)?;
                let reps = self.i64s(inp(ins, 1)?)?;
                one(x.repeat(&reps))
            }
            "Slice" => {
                let x = inp(ins, 0)?;
                let starts = self.i64s(inp(ins, 1)?)?;
                let ends = self.i64s(inp(ins, 2)?)?;
                let axes = match opt(ins, 3) {
                    Some(t) => self.i64s(t)?,
                    None => (0..starts.len() as i64).collect(),
                };
                let steps = match opt(ins, 4) {
                    Some(t) => self.i64s(t)?,
                    None => vec![1; starts.len()],
                };
                let ndim = x.shape().as_slice().len();
                let mut ixs: Vec<Ix> = vec![Ix::All; ndim];
                for i in 0..starts.len() {
                    let a = resolve_index(axes[i], ndim as i64) as usize;
                    if a >= ndim {
                        return Err(OnnxError::Shape(format!("Slice axis {a} out of range")));
                    }
                    let step = *steps.get(i).unwrap_or(&1);
                    if step == 0 {
                        return Err(OnnxError::Runtime("Slice step 0".into()));
                    }
                    ixs[a] = Ix::slice(Some(starts[i]), Some(ends[i]), step);
                }
                one(x.getitem(&ixs))
            }
            "Gather" => {
                let x = inp(ins, 0)?;
                let idx = inp(ins, 1)?;
                let axis = resolve_axis(x, node.attr_i64("axis", 0));
                let scalar = idx.shape().as_slice().is_empty();
                let idx = if scalar {
                    idx.unsqueeze(0)
                } else {
                    idx.clone()
                };
                let ndim = x.shape().as_slice().len();
                let mut ixs: Vec<Ix> = vec![Ix::All; ndim];
                ixs[axis] = Ix::Tensor(idx);
                let out = x.getitem(&ixs);
                one(if scalar {
                    out.squeeze(Some(axis as i64))
                } else {
                    out
                })
            }
            "GatherND" => {
                let x = inp(ins, 0)?;
                let idx = inp(ins, 1)?;
                let batch = node.attr_i64("batch_dims", 0);
                one(self.gather_nd(x, idx, batch)?)
            }
            "GatherElements" => {
                let x = inp(ins, 0)?;
                let idx = inp(ins, 1)?;
                let axis = resolve_axis(x, node.attr_i64("axis", 0));
                one(self.gather_elements(x, idx, axis)?)
            }
            "DepthToSpace" => {
                let x = inp(ins, 0)?;
                let bs = node.attr_i64("blocksize", 1);
                let crd = node.attr_str("mode", "DCR") == "CRD";
                one(depth_to_space(x, bs, crd)?)
            }
            "SpaceToDepth" => {
                let x = inp(ins, 0)?;
                let bs = node.attr_i64("blocksize", 1);
                one(space_to_depth(x, bs)?)
            }
            "Pad" => {
                let x = inp(ins, 0)?;
                let pads = self.i64s(inp(ins, 1)?)?;
                let mode = node.attr_str("mode", "constant");
                let axes = match opt(ins, 3) {
                    Some(t) => Some(self.i64s(t)?),
                    None => node.attr_i64s("axes"),
                };
                let value = match opt(ins, 2) {
                    Some(t) => self.f64_scalar(t)?,
                    None => node.attr_f64("value", 0.0),
                };
                let full = expand_pads(&pads, axes.as_deref(), x.shape().as_slice().len())?;
                one(self.pad_op(x, &full, mode, value)?)
            }

            // ---------------- reductions / sorting ----------------
            "ReduceSum" | "ReduceMean" | "ReduceMax" | "ReduceMin" | "ReduceProd" | "ReduceL1"
            | "ReduceL2" | "ReduceSumSquare" | "ReduceLogSum" | "ReduceLogSumExp" => {
                let x = inp(ins, 0)?;
                let axes = self.reduce_axes(node, ins, x.shape().as_slice().len())?;
                let keep = node.attr_i64("keepdims", 1) != 0;
                one(self.reduce_op(op, x, axes.as_deref(), keep)?)
            }
            "ArgMax" | "ArgMin" => {
                let x = inp(ins, 0)?;
                let axis = self.axis(node, "axis", 0, x.shape().as_slice().len() as i64)?;
                let keep = node.attr_i64("keepdims", 1) != 0;
                let select_last = node.attr_i64("select_last_index", 0) != 0;
                let mut t = if op == "ArgMax" {
                    x.argmax(Some(axis as i64), keep)
                } else {
                    x.argmin(Some(axis as i64), keep)
                };
                if select_last {
                    let n = x.shape().as_slice()[axis].clone().as_i64().unwrap_or(1);
                    let flipped = x.flip(&[axis]);
                    let f = if op == "ArgMax" {
                        flipped.argmax(Some(axis as i64), keep)
                    } else {
                        flipped.argmin(Some(axis as i64), keep)
                    };
                    t = f.neg().add(n - 1);
                }
                one(t.cast(dtype::INT64))
            }
            "CumSum" => {
                let x = inp(ins, 0)?;
                let axis = self.i64_scalar(inp(ins, 1)?)?;
                let exclusive = node.attr_i64("exclusive", 0) != 0;
                let reverse = node.attr_i64("reverse", 0) != 0;
                one(self.cumsum_op(x, axis, exclusive, reverse)?)
            }
            "TopK" => {
                let x = inp(ins, 0)?;
                let k = self.i64_scalar(inp(ins, 1)?)?;
                let axis = node.attr_i64("axis", -1);
                let largest = node.attr_i64("largest", 1) != 0;
                let (v, i) = x.topk(k, axis, largest);
                Ok(vec![v, i.cast(dtype::INT64)])
            }
            "NonZero" => {
                let x = inp(ins, 0)?;
                let shape: Vec<i64> = x
                    .shape()
                    .as_slice()
                    .iter()
                    .map(|s| s.clone().as_i64().unwrap_or(0))
                    .collect();
                let rank = shape.len();
                if rank == 0 {
                    return Err(OnnxError::Unsupported("NonZero on a scalar".into()));
                }
                let vals = x.to_list::<f64>(&mut self.sess);
                let mut coords: Vec<i64> = Vec::new();
                let mut nnz = 0i64;
                let mut idx = vec![0i64; rank];
                for &v in &vals {
                    if v != 0.0 {
                        coords.extend_from_slice(&idx);
                        nnz += 1;
                    }
                    for d in (0..rank).rev() {
                        idx[d] += 1;
                        if idx[d] < shape[d] {
                            break;
                        }
                        idx[d] = 0;
                    }
                }
                if nnz == 0 {
                    return one(
                        Tensor::from_slice(
                            &self.ctx,
                            &[] as &[i64],
                            Some(dtype::INT64),
                            self.device.as_deref(),
                        )
                        .reshape([rank as i64, 0]),
                    );
                }
                // `coords` is [nnz][rank]; the ONNX output is [rank][nnz].
                let mut out = vec![0i64; rank * nnz as usize];
                for i in 0..nnz as usize {
                    for d in 0..rank {
                        out[d * nnz as usize + i] = coords[i * rank + d];
                    }
                }
                let t = Tensor::from_slice(
                    &self.ctx,
                    &out,
                    Some(dtype::INT64),
                    self.device.as_deref(),
                )
                .reshape([rank as i64, nnz]);
                one(t)
            }
            "Compress" => {
                let data = inp(ins, 0)?;
                let cond = inp(ins, 1)?;
                let cond_vals = cond.to_list::<i64>(&mut self.sess);
                let axis_attr = node.attr_i64("axis", i64::MIN);
                let ndim = data.shape().as_slice().len();
                let (flat, axis, dim) = if axis_attr == i64::MIN {
                    let n = data
                        .shape()
                        .as_slice()
                        .iter()
                        .map(|s| s.clone().as_i64().unwrap_or(0))
                        .product::<i64>();
                    (true, 0usize, n)
                } else {
                    let axis = resolve_axis(data, axis_attr);
                    let n = data.shape().as_slice()[axis].clone().as_i64().unwrap_or(0);
                    (false, axis, n)
                };
                let keep: Vec<i64> = cond_vals
                    .iter()
                    .enumerate()
                    .filter(|(i, v)| **v != 0 && (*i as i64) < dim)
                    .map(|(i, _)| i as i64)
                    .collect();
                let idx = Tensor::from_slice(
                    &self.ctx,
                    &keep,
                    Some(dtype::INT64),
                    self.device.as_deref(),
                );
                if flat {
                    let flat_data = data.reshape([-1]);
                    one(flat_data.getitem(&[Ix::Tensor(idx)]))
                } else {
                    let mut ixs: Vec<Ix> = vec![Ix::All; ndim];
                    ixs[axis] = Ix::Tensor(idx);
                    one(data.getitem(&ixs))
                }
            }
            "CenterCropPad" => {
                let data = inp(ins, 0)?;
                let shape_t = inp(ins, 1)?;
                let target = shape_t.to_list::<i64>(&mut self.sess);
                let rank = data.shape().as_slice().len();
                let cur: Vec<i64> = data
                    .shape()
                    .as_slice()
                    .iter()
                    .map(|s| s.clone().as_i64().unwrap_or(0))
                    .collect();
                let mut tgt = cur.clone();
                match node.attr_i64s("axes") {
                    Some(axes) if !axes.is_empty() => {
                        if target.len() != axes.len() {
                            return Err(OnnxError::Shape(
                                "CenterCropPad shape/axes length".into(),
                            ));
                        }
                        for (i, &ax) in axes.iter().enumerate() {
                            let ax = resolve_axis(data, ax);
                            tgt[ax] = target[i];
                        }
                    }
                    _ => {
                        if target.len() != rank {
                            return Err(OnnxError::Shape(
                                "CenterCropPad shape length".into(),
                            ));
                        }
                        tgt = target.clone();
                    }
                }
                let mut out = data.clone();
                if (0..rank).any(|d| tgt[d] < cur[d]) {
                    let mut ixs: Vec<Ix> = Vec::with_capacity(rank);
                    for d in 0..rank {
                        if tgt[d] < cur[d] {
                            let start = (cur[d] - tgt[d]) / 2;
                            ixs.push(Ix::Range {
                                start: Some(start),
                                stop: Some(start + tgt[d]),
                                step: 1,
                            });
                        } else {
                            ixs.push(Ix::All);
                        }
                    }
                    out = out.getitem(&ixs);
                }
                let mut pads = vec![0i64; 2 * rank];
                for d in 0..rank {
                    if tgt[d] > cur[d] {
                        let diff = tgt[d] - cur[d];
                        pads[d] = diff / 2;
                        pads[rank + d] = diff - diff / 2;
                    }
                }
                if pads.iter().any(|&v| v != 0) {
                    out = self.pad_op(&out, &pads, "constant", 0.0)?;
                }
                one(out)
            }
            "Scatter" | "ScatterElements" => {
                let data = inp(ins, 0)?;
                let indices = inp(ins, 1)?;
                let updates = inp(ins, 2)?;
                let axis = resolve_axis(data, node.attr_i64("axis", 0));
                let reduction = node.attr_str("reduction", "none");
                one(self.scatter_elements(data, indices, updates, axis, reduction)?)
            }
            "ScatterND" => {
                let data = inp(ins, 0)?;
                let indices = inp(ins, 1)?;
                let updates = inp(ins, 2)?;
                let reduction = node.attr_str("reduction", "none");
                one(self.scatter_nd(data, indices, updates, reduction)?)
            }
            "TensorScatter" => {
                let past = inp(ins, 0)?;
                let update = inp(ins, 1)?;
                let write = opt(ins, 2);
                let axis = node.attr_i64("axis", -2);
                let mode = node.attr_str("mode", "linear");
                one(self.tensor_scatter_op(past, update, write, axis, mode)?)
            }

            // ---------------- creation ----------------
            "Range" => {
                let start = self.f64_scalar(inp(ins, 0)?)?;
                let limit = self.f64_scalar(inp(ins, 1)?)?;
                let delta = self.f64_scalar(inp(ins, 2)?)?;
                let dt = inp(ins, 0)?.dtype();
                one(Tensor::arange(
                    &self.ctx,
                    start,
                    Some(limit),
                    delta,
                    Some(dt),
                ))
            }
            "EyeLike" => {
                let x = inp(ins, 0)?;
                let dt = match node.attr_i64("dtype", -1) {
                    -1 => x.dtype(),
                    v => super::pb::onnx_dtype(v as i32)?,
                };
                let n = x.shape().as_slice()[0].clone().as_i64().unwrap_or(0);
                let m = x
                    .shape()
                    .as_slice()
                    .get(1)
                    .map(|s| s.clone().as_i64().unwrap_or(n))
                    .unwrap_or(n);
                one(Tensor::eye(&self.ctx, n, Some(m), Some(dt)))
            }
            "Trilu" => {
                let x = inp(ins, 0)?;
                let k = match opt(ins, 1) {
                    Some(t) => self.i64_scalar(t)?,
                    None => node.attr_i64("k", 0),
                };
                let upper = node.attr_i64("upper", 1) != 0;
                one(if upper { x.triu(k) } else { x.tril(k) })
            }
            "OneHot" => {
                let indices = inp(ins, 0)?.cast(dtype::INT32);
                let depth = self.i64_scalar(inp(ins, 1)?)?;
                let values = inp(ins, 2)?;
                let axis = node.attr_i64("axis", -1);
                let idx = indices.lt(0).where_(indices.add(depth), indices);
                let ndim = idx.shape().as_slice().len() as i64;
                let axis_nd = resolve_index(axis, ndim + 1) as i64;
                let oh = idx.unsqueeze(axis_nd).one_hot_along_dim(depth, axis_nd);
                let on = values.getitem(&[Ix::Index(1)]).reshape(&[] as &[i64]);
                let off = values.getitem(&[Ix::Index(0)]).reshape(&[] as &[i64]);
                one(oh.where_(on, off))
            }

            // ---------------- nn ----------------
            "Conv" => one(self.conv_op(node, ins)?),
            "ConvTranspose" => one(self.conv_transpose_op(node, ins)?),
            "MatMul" => one(inp(ins, 0)?.matmul(inp(ins, 1)?)),
            "Gemm" => {
                let a = inp(ins, 0)?;
                let b = inp(ins, 1)?;
                let alpha = node.attr_f64("alpha", 1.0);
                let beta = node.attr_f64("beta", 1.0);
                let ta = node.attr_i64("transA", 0) != 0;
                let tb = node.attr_i64("transB", 0) != 0;
                let a = if ta { a.transpose(0, 1) } else { a.clone() };
                let b = if tb { b.transpose(0, 1) } else { b.clone() };
                let mut out = a.matmul(&b);
                if alpha != 1.0 {
                    out = out.mul(alpha);
                }
                if let Some(c) = opt(ins, 2) {
                    let c = if beta != 1.0 { c.mul(beta) } else { c.clone() };
                    out = out.add(&c);
                }
                one(out)
            }
            "Einsum" => {
                let eq = node.attr_str("equation", "");
                let operands: Result<Vec<&Tensor>> =
                    (0..ins.len()).map(|i| inp(ins, i)).collect();
                one(Tensor::einsum_many(&operands?, eq))
            }
            "MaxUnpool" => {
                let x = inp(ins, 0)?;
                let indices = inp(ins, 1)?;
                let out_shape = opt(ins, 2);
                one(self.max_unpool(x, indices, out_shape, node)?)
            }
            "MaxPool" => Ok(self.max_pool_op(node, ins)?),
            "AveragePool" => one(self.avg_pool_op(node, ins)?),
            "GlobalAveragePool" => {
                let x = inp(ins, 0)?;
                let n = x.shape().as_slice().len();
                let axes: Vec<usize> = (2..n).collect();
                one(x.mean(Some(&axes), true))
            }
            "GlobalMaxPool" => {
                let x = inp(ins, 0)?;
                let n = x.shape().as_slice().len();
                let axes: Vec<usize> = (2..n).collect();
                one(x.max(Some(&axes), true))
            }
            "BatchNormalization" => {
                let x = inp(ins, 0)?;
                let scale = inp(ins, 1)?;
                let bias = inp(ins, 2)?;
                let mean = inp(ins, 3)?;
                let var = inp(ins, 4)?;
                let eps = node.attr_f64("epsilon", 1e-5);
                if node.attr_i64("training_mode", 0) != 0 {
                    return Err(OnnxError::Unsupported(
                        "BatchNormalization training_mode".into(),
                    ));
                }
                let invstd = var.add(eps).rsqrt();
                one(x.batchnorm(Some(scale), Some(bias), mean, &invstd, &[1]))
            }
            "LayerNormalization" => {
                let x = inp(ins, 0)?;
                let scale = inp(ins, 1)?;
                let bias = opt(ins, 2);
                let axis = self.axis(node, "axis", -1, x.shape().as_slice().len() as i64)? as i64;
                let eps = node.attr_f64("epsilon", 1e-5);
                if node.attr_i64("stash_type", 1) != 1 {
                    return Err(OnnxError::Unsupported(
                        "LayerNormalization stash_type != 1".into(),
                    ));
                }
                let ndim = x.shape().as_slice().len() as i64;
                let start = resolve_index(axis, ndim) as usize;
                let axes: Vec<usize> = (start..ndim as usize).collect();
                let x32 = x.cast(dtype::FLOAT32);
                let mean = x32.mean(Some(&axes), true);
                let inv = x32
                    .sub(&mean)
                    .square()
                    .mean(Some(&axes), true)
                    .add(eps)
                    .rsqrt();
                let mut out = x32.sub(&mean).mul(&inv).cast(x.dtype()).mul(scale);
                if let Some(b) = bias {
                    out = out.add(b);
                }
                Ok(vec![out, mean, inv])
            }
            "SkipLayerNormalization" => self.skip_layer_norm_op(node, ins),
            "EmbedLayerNormalization" => self.embed_layer_norm_op(node, ins),
            "RotaryEmbedding" => self.rotary_embedding_op(node, ins),
            "AffineGrid" => self.affine_grid_op(node, ins),
            "InstanceNormalization" => {
                let x = inp(ins, 0)?;
                let scale = inp(ins, 1)?;
                let bias = inp(ins, 2)?;
                let eps = node.attr_f64("epsilon", 1e-5);
                one(self.group_norm(
                    x,
                    scale,
                    bias,
                    x.shape().as_slice()[1].clone().as_i64().unwrap_or(1),
                    eps,
                )?)
            }
            "GroupNormalization" => {
                let x = inp(ins, 0)?;
                let scale = inp(ins, 1)?;
                let bias = inp(ins, 2)?;
                let groups = node.attr_i64("num_groups", 1);
                let eps = node.attr_f64("epsilon", 1e-5);
                one(self.group_norm(x, scale, bias, groups, eps)?)
            }
            "Dropout" => {
                let x = inp(ins, 0)?;
                let training = match opt(ins, 2) {
                    Some(t) => self.i64_scalar(t)? != 0,
                    None => false,
                };
                if training {
                    return Err(OnnxError::Unsupported(
                        "Dropout with training_mode=true".into(),
                    ));
                }
                if node.outputs.len() > 1 {
                    let mask = x.ones_like().cast(dtype::BOOL);
                    Ok(vec![x.clone(), mask])
                } else {
                    one(x.clone())
                }
            }
            "LpNormalization" => {
                let x = inp(ins, 0)?;
                let p = node.attr_f64("p", 2.0);
                let axis = self.axis(node, "axis", -1, x.shape().as_slice().len() as i64)?;
                let axes = [axis];
                let abs = x.abs();
                let mut den = if p == 1.0 {
                    abs.sum(Some(&axes), true, None)
                } else if p == 2.0 {
                    abs.square().sum(Some(&axes), true, None)
                } else {
                    self.pow_scalar(&abs, p).sum(Some(&axes), true, None)
                };
                if p == 2.0 {
                    den = den.sqrt();
                } else if p != 1.0 {
                    den = self.pow_scalar(&den, 1.0 / p);
                }
                one(x.div(den.maximum(1e-12)))
            }
            "MeanVarianceNormalization" => {
                let x = inp(ins, 0)?;
                let axes: Vec<usize> = node
                    .attr_i64s("axes")
                    .unwrap_or_else(|| vec![0, 2, 3])
                    .into_iter()
                    .map(|a| resolve_index(a, x.shape().as_slice().len() as i64) as usize)
                    .collect();
                let mean = x.mean(Some(&axes), true);
                let var = x.sub(&mean).square().mean(Some(&axes), true);
                one(x.sub(&mean).div(var.add(1e-9).sqrt()))
            }
            "LRN" => {
                let x = inp(ins, 0)?;
                let size = node.attr_i64("size", 1);
                let alpha = node.attr_f64("alpha", 1e-4);
                let beta = node.attr_f64("beta", 0.75);
                let bias = node.attr_f64("bias", 1.0);
                one(self.lrn(x, size, alpha, beta, bias)?)
            }

            // ---------------- resize ----------------
            "Resize" => one(self.resize_op(node, ins)?),
            "Upsample" => one(self.resize_op(node, ins)?),

            other => Err(OnnxError::Unsupported(format!(
                "op `{other}` (domain `{}`) is not implemented",
                if node.domain.is_empty() {
                    "ai.onnx"
                } else {
                    &node.domain
                }
            ))),
        }
    }

    fn binop_raw(
        &self,
        a: &Tensor,
        b: &Tensor,
        f: fn(&mut UOpCtx, UopId, UopId) -> UopId,
    ) -> Tensor {
        a.ctx().ensure_same(b.ctx());
        let mut ctx = a.ctx().borrow_mut();
        let u = f(&mut ctx, a.uop(), b.uop());
        Tensor::from_uop(&ctx, u)
    }

    fn axis(&self, node: &Node, name: &str, default: i64, rank: i64) -> Result<usize> {
        Ok(resolve_index(node.attr_i64(name, default), rank) as usize)
    }

    fn reduce_axes(
        &mut self,
        node: &Node,
        ins: &[Option<Tensor>],
        rank: usize,
    ) -> Result<Option<Vec<usize>>> {
        let noop = node.attr_i64("noop_with_empty_axes", 0) != 0;
        let axes = match opt(ins, 1) {
            Some(t) => self.i64s(t)?,
            None => node.attr_i64s("axes").unwrap_or_default(),
        };
        if axes.is_empty() {
            if noop { Ok(Some(Vec::new())) } else { Ok(None) }
        } else {
            Ok(Some(
                axes.into_iter()
                    .map(|a| resolve_index(a, rank as i64) as usize)
                    .collect(),
            ))
        }
    }

    fn reduce_op(
        &mut self,
        op: &str,
        x: &Tensor,
        axes: Option<&[usize]>,
        keep: bool,
    ) -> Result<Tensor> {
        if axes.is_some_and(|a| a.is_empty()) {
            return Ok(x.clone());
        }
        Ok(match op {
            "ReduceSum" => x.sum(axes, keep, None),
            "ReduceMean" => x.mean(axes, keep),
            "ReduceMax" => x.max(axes, keep),
            "ReduceMin" => x.min(axes, keep),
            "ReduceProd" => x.prod(axes, keep, None),
            "ReduceL1" => x.abs().sum(axes, keep, None),
            "ReduceL2" => x.square().sum(axes, keep, None).sqrt(),
            "ReduceSumSquare" => x.square().sum(axes, keep, None),
            "ReduceLogSum" => x.sum(axes, keep, None).log(),
            "ReduceLogSumExp" => x.exp().sum(axes, keep, None).log(),
            _ => unreachable!(),
        })
    }

    fn cumsum_op(
        &mut self,
        x: &Tensor,
        axis: i64,
        exclusive: bool,
        reverse: bool,
    ) -> Result<Tensor> {
        let ndim = x.shape().as_slice().len();
        let ax = resolve_axis(x, axis);
        let mut t = if reverse {
            self.reverse_axis(x, ax)?
        } else {
            x.clone()
        };
        if exclusive {
            let mut pads = vec![(0i64, 0i64); ndim];
            pads[ax] = (1, 0);
            let padded = t.pad(&pads);
            let mut ends: Vec<(i64, i64)> = Vec::with_capacity(ndim);
            for (i, s) in padded.shape().as_slice().iter().enumerate() {
                let n = s.clone().as_i64().unwrap_or(0);
                ends.push((0, if i == ax { n - 1 } else { n }));
            }
            t = padded.shrink(&ends);
        }
        t = t.cumsum(ax as i64);
        if reverse {
            t = self.reverse_axis(&t, ax)?;
        }
        Ok(t)
    }

    fn pow_scalar(&self, x: &Tensor, p: f64) -> Tensor {
        let e = Tensor::full_const(
            &self.ctx,
            &[] as &[i64],
            ConstVal::Float(p),
            Some(x.dtype()),
        );
        xpow(x, &e)
    }

    fn reverse_axis(&mut self, x: &Tensor, ax: usize) -> Result<Tensor> {
        let n = x.shape().as_slice()[ax].clone().as_i64().unwrap_or(0);
        let idx: Vec<i64> = (0..n).rev().collect();
        let idx = Tensor::from_slice(&self.ctx, &idx, Some(dtype::INT32), self.device());
        let mut ixs: Vec<Ix> = vec![Ix::All; x.shape().as_slice().len()];
        ixs[ax] = Ix::Tensor(idx);
        Ok(x.getitem(&ixs))
    }

    /// Rebuild a host-computed result with `template`'s dtype and `shape`.
    fn build_like(&mut self, template: &Tensor, vals: &[f64], shape: &[i64]) -> Result<Tensor> {
        let dt = template.dtype();
        let t = if dt == dtype::FLOAT32 {
            let v: Vec<f32> = vals.iter().map(|x| *x as f32).collect();
            Tensor::from_slice(&self.ctx, &v, Some(dt), self.device.as_deref())
        } else if dt == dtype::FLOAT64 {
            Tensor::from_slice(&self.ctx, vals, Some(dt), self.device.as_deref())
        } else if dt == dtype::INT64 {
            let v: Vec<i64> = vals.iter().map(|x| *x as i64).collect();
            Tensor::from_slice(&self.ctx, &v, Some(dt), self.device.as_deref())
        } else if dt == dtype::INT32 {
            let v: Vec<i32> = vals.iter().map(|x| *x as i32).collect();
            Tensor::from_slice(&self.ctx, &v, Some(dt), self.device.as_deref())
        } else if dt == dtype::BOOL {
            let v: Vec<bool> = vals.iter().map(|x| *x != 0.0).collect();
            Tensor::from_slice(&self.ctx, &v, Some(dt), self.device.as_deref())
        } else if dtype::is_float(dt) {
            let v: Vec<f32> = vals.iter().map(|x| *x as f32).collect();
            Tensor::from_slice(&self.ctx, &v, Some(dtype::FLOAT32), self.device.as_deref())
                .cast(dt)
        } else {
            return Err(OnnxError::Unsupported(format!("host build on dtype {dt:?}")));
        };
        Ok(t.reshape(shape))
    }

    fn scatter_elements(
        &mut self,
        data: &Tensor,
        indices: &Tensor,
        updates: &Tensor,
        axis: usize,
        reduction: &str,
    ) -> Result<Tensor> {
        let shape: Vec<i64> = data
            .shape()
            .as_slice()
            .iter()
            .map(|s| s.clone().as_i64().unwrap_or(0))
            .collect();
        let rank = shape.len();
        let ishape: Vec<i64> = indices
            .shape()
            .as_slice()
            .iter()
            .map(|s| s.clone().as_i64().unwrap_or(0))
            .collect();
        if ishape.len() != rank {
            return Err(OnnxError::Shape(
                "ScatterElements indices/rank mismatch".into(),
            ));
        }
        let dv = data.to_list::<f64>(&mut self.sess);
        let iv = indices.to_list::<i64>(&mut self.sess);
        let uv = updates.to_list::<f64>(&mut self.sess);
        let mut strides = vec![1i64; rank];
        for d in (0..rank.saturating_sub(1)).rev() {
            strides[d] = strides[d + 1] * shape[d + 1];
        }
        let total: i64 = ishape.iter().product();
        let mut out = dv.clone();
        let mut coords = vec![0i64; rank];
        for p in 0..total {
            let mut rem = p;
            for d in (0..rank).rev() {
                coords[d] = rem % ishape[d];
                rem /= ishape[d];
            }
            let mut idx = iv[p as usize];
            if idx < 0 {
                idx += shape[axis];
            }
            if idx < 0 || idx >= shape[axis] {
                return Err(OnnxError::Runtime(
                    "ScatterElements index out of range".into(),
                ));
            }
            coords[axis] = idx;
            let di: i64 = coords.iter().zip(&strides).map(|(c, s)| c * s).sum();
            let u = uv[p as usize];
            match reduction {
                "" | "none" => out[di as usize] = u,
                "add" => out[di as usize] += u,
                "mul" => out[di as usize] *= u,
                "max" => out[di as usize] = out[di as usize].max(u),
                "min" => out[di as usize] = out[di as usize].min(u),
                other => {
                    return Err(OnnxError::Unsupported(format!(
                        "Scatter reduction `{other}`"
                    )));
                }
            }
        }
        self.build_like(data, &out, &shape)
    }

    fn scatter_nd(
        &mut self,
        data: &Tensor,
        indices: &Tensor,
        updates: &Tensor,
        reduction: &str,
    ) -> Result<Tensor> {
        let shape: Vec<i64> = data
            .shape()
            .as_slice()
            .iter()
            .map(|s| s.clone().as_i64().unwrap_or(0))
            .collect();
        let rank = shape.len();
        let ishape: Vec<i64> = indices
            .shape()
            .as_slice()
            .iter()
            .map(|s| s.clone().as_i64().unwrap_or(0))
            .collect();
        let Some(&k64) = ishape.last() else {
            return Err(OnnxError::Shape("ScatterND scalar indices".into()));
        };
        let k = k64 as usize;
        if k > rank {
            return Err(OnnxError::Shape("ScatterND index depth".into()));
        }
        let num: i64 = ishape[..ishape.len() - 1].iter().product();
        let per: i64 = shape[k..].iter().product();
        let dv = data.to_list::<f64>(&mut self.sess);
        let iv = indices.to_list::<i64>(&mut self.sess);
        let uv = updates.to_list::<f64>(&mut self.sess);
        let mut strides = vec![1i64; rank];
        for d in (0..rank.saturating_sub(1)).rev() {
            strides[d] = strides[d + 1] * shape[d + 1];
        }
        let mut out = dv.clone();
        let mut idx = vec![0i64; k];
        for n in 0..num {
            for d in 0..k {
                let mut v = iv[n as usize * k + d];
                if v < 0 {
                    v += shape[d];
                }
                if v < 0 || v >= shape[d] {
                    return Err(OnnxError::Runtime("ScatterND index out of range".into()));
                }
                idx[d] = v;
            }
            let base: i64 = idx.iter().zip(&strides[..k]).map(|(c, s)| c * s).sum();
            for t in 0..per {
                let di = (base + t) as usize;
                let u = uv[(n * per + t) as usize];
                match reduction {
                    "" | "none" => out[di] = u,
                    "add" => out[di] += u,
                    "mul" => out[di] *= u,
                    "max" => out[di] = out[di].max(u),
                    "min" => out[di] = out[di].min(u),
                    other => {
                        return Err(OnnxError::Unsupported(format!(
                            "ScatterND reduction `{other}`"
                        )));
                    }
                }
            }
        }
        self.build_like(data, &out, &shape)
    }

    fn pad_op(&mut self, x: &Tensor, pads: &[i64], mode: &str, value: f64) -> Result<Tensor> {
        let n = x.shape().as_slice().len();
        if pads.len() != n * 2 {
            return Err(OnnxError::Shape(format!(
                "Pad expects {} pads, got {}",
                n * 2,
                pads.len()
            )));
        }
        let pairs: Vec<(i64, i64)> = (0..n).map(|i| (pads[i], pads[i + n])).collect();
        match mode {
            "constant" => {
                let cv = match (x.dtype(), value.fract() == 0.0) {
                    (dt, _) if dtype::is_int(dt) || dtype::is_bool(dt) => {
                        ConstVal::Int(value as i128)
                    }
                    _ => ConstVal::Float(value),
                };
                let p: Vec<(Option<i64>, Option<i64>)> =
                    pairs.iter().map(|(a, b)| (Some(*a), Some(*b))).collect();
                Ok(x.pad_constant(&p, cv))
            }
            "edge" | "reflect" | "wrap" => self.pad_modes(x, &pairs, mode),
            other => Err(OnnxError::Unsupported(format!("Pad mode `{other}`"))),
        }
    }

    fn pad_modes(&mut self, x: &Tensor, pads: &[(i64, i64)], mode: &str) -> Result<Tensor> {
        let mut out = x.clone();
        let ndim = x.shape().as_slice().len();
        for axis in 0..ndim {
            let (lo, hi) = pads[axis];
            if lo == 0 && hi == 0 {
                continue;
            }
            let n = out.shape().as_slice()[axis].clone().as_i64().unwrap_or(0);
            if n == 0 {
                continue;
            }
            let mut idx: Vec<i64> = Vec::with_capacity((n + lo + hi) as usize);
            for p in -lo..(n + hi) {
                let v = match mode {
                    "edge" => p.clamp(0, n - 1),
                    "reflect" => {
                        if n == 1 {
                            0
                        } else {
                            let period = 2 * (n - 1);
                            let mut q = p.rem_euclid(period);
                            if q >= n {
                                q = period - q;
                            }
                            q
                        }
                    }
                    "wrap" => p.rem_euclid(n),
                    _ => unreachable!(),
                };
                idx.push(v);
            }
            let idx_t = Tensor::from_slice(&self.ctx, &idx, Some(dtype::INT32), self.device());
            let mut ixs: Vec<Ix> = vec![Ix::All; ndim];
            ixs[axis] = Ix::Tensor(idx_t);
            out = out.getitem(&ixs);
        }
        Ok(out)
    }

    fn gather_nd(&mut self, x: &Tensor, idx: &Tensor, batch_dims: i64) -> Result<Tensor> {
        let xshape: Vec<i64> = x
            .shape()
            .as_slice()
            .iter()
            .map(|s| s.clone().as_i64().unwrap_or(0))
            .collect();
        let ishape: Vec<i64> = idx
            .shape()
            .as_slice()
            .iter()
            .map(|s| s.clone().as_i64().unwrap_or(0))
            .collect();
        let r = xshape.len();
        let q = ishape.len();
        if q == 0 {
            return Err(OnnxError::Unsupported("GatherND with scalar indices".into()));
        }
        let b = batch_dims.max(0) as usize;
        if b > q - 1 || b > r {
            return Err(OnnxError::Shape("GatherND batch_dims".into()));
        }
        for d in 0..b {
            if xshape[d] != ishape[d] {
                return Err(OnnxError::Shape(
                    "GatherND batch dimensions do not match".into(),
                ));
            }
        }
        let k = ishape[q - 1].max(0) as usize;
        if b + k > r {
            return Err(OnnxError::Shape("GatherND index depth".into()));
        }
        if b == 0 {
            return gather_nd_0(x, idx);
        }
        // Flatten the batch axes, gather each slice, then stack and reshape.
        let batch_vol: i64 = xshape[..b].iter().product();
        let mut xf = xshape[b..].to_vec();
        xf.insert(0, batch_vol);
        let mut inf = ishape[b..].to_vec();
        inf.insert(0, batch_vol);
        let x_flat = x.reshape(xf);
        let idx_flat = idx.reshape(inf);
        let mut parts: Vec<Tensor> = Vec::with_capacity(batch_vol as usize);
        for bi in 0..batch_vol {
            let xb = x_flat.getitem(&[Ix::Index(bi)]);
            let ib = idx_flat.getitem(&[Ix::Index(bi)]);
            parts.push(gather_nd_0(&xb, &ib)?);
        }
        let first = parts.remove(0).unsqueeze(0);
        let rest: Vec<Tensor> = parts.into_iter().map(|t| t.unsqueeze(0)).collect();
        let stacked = if rest.is_empty() {
            first
        } else {
            first.cat(&rest, 0)
        };
        let mut out_shape: Vec<i64> = ishape[..q - 1].to_vec();
        out_shape.extend_from_slice(&xshape[b + k..]);
        Ok(stacked.reshape(out_shape))
    }

    fn max_unpool(
        &mut self,
        x: &Tensor,
        indices: &Tensor,
        out_shape: Option<&Tensor>,
        node: &Node,
    ) -> Result<Tensor> {
        let xshape: Vec<i64> = x
            .shape()
            .as_slice()
            .iter()
            .map(|s| s.clone().as_i64().unwrap_or(0))
            .collect();
        let rank = xshape.len();
        if rank < 2 {
            return Err(OnnxError::Shape("MaxUnpool needs N,C,spatial".into()));
        }
        let spatial = rank - 2;
        let out_shape: Vec<i64> = if let Some(t) = out_shape {
            self.i64s(t)?
        } else {
            let kernel = node
                .attr_i64s("kernel_shape")
                .ok_or_else(|| OnnxError::Shape("MaxUnpool kernel_shape".into()))?;
            let strides = node
                .attr_i64s("strides")
                .unwrap_or_else(|| vec![1; spatial]);
            let pads = node
                .attr_i64s("pads")
                .unwrap_or_else(|| vec![0; 2 * spatial]);
            if kernel.len() != spatial || strides.len() != spatial || pads.len() != 2 * spatial {
                return Err(OnnxError::Shape("MaxUnpool attributes".into()));
            }
            let mut s = vec![xshape[0], xshape[1]];
            for i in 0..spatial {
                s.push((xshape[2 + i] - 1) * strides[i] - pads[i] - pads[spatial + i] + kernel[i]);
            }
            s
        };
        if out_shape.len() != rank {
            return Err(OnnxError::Shape("MaxUnpool output shape rank".into()));
        }
        let planes: i64 = xshape[0].saturating_mul(xshape[1]);
        let in_vol: i64 = xshape[2..].iter().product();
        let out_vol: i64 = out_shape[2..].iter().product();
        let xv = x.to_list::<f64>(&mut self.sess);
        let iv = indices.to_list::<i64>(&mut self.sess);
        let mut out = vec![0.0f64; (planes * out_vol) as usize];
        for plane in 0..planes {
            for p in 0..in_vol {
                let o = iv[(plane * in_vol + p) as usize];
                if o < 0 || o >= out_vol {
                    return Err(OnnxError::Runtime("MaxUnpool index out of range".into()));
                }
                out[(plane * out_vol + o) as usize] = xv[(plane * in_vol + p) as usize];
            }
        }
        self.build_like(x, &out, &out_shape)
    }

    /// `NegativeLogLikelihoodLoss` / `SoftmaxCrossEntropyLoss`: thin wrappers
    /// over [`Tensor::nll_loss_axis`] (class axis 1), returning the optional
    /// log-probabilities for the second `SoftmaxCrossEntropyLoss` output.
    fn nll_loss_op(
        &mut self,
        node: &Node,
        ins: &[Option<Tensor>],
        softmax: bool,
    ) -> Result<Vec<Tensor>> {
        let scores = inp(ins, 0)?;
        let labels = inp(ins, 1)?;
        let weight = opt(ins, 2);
        let reduction = node.attr_str("reduction", "mean");
        if !matches!(reduction, "none" | "mean" | "sum") {
            return Err(OnnxError::Unsupported(format!(
                "loss reduction `{reduction}`"
            )));
        }
        let ignore = node.attr_i64("ignore_index", i64::MIN);
        let ignore = if ignore == i64::MIN {
            None
        } else {
            Some(ignore)
        };
        let logp = if softmax {
            crate::tensor::nn::log_softmax(scores, Some(1))
        } else {
            scores.clone()
        };
        let loss = logp.nll_loss_axis(labels, weight, ignore, reduction, 1);
        let mut outs = vec![loss];
        if softmax && node.outputs.len() > 1 {
            outs.push(logp);
        }
        Ok(outs)
    }

    fn skip_layer_norm_op(&mut self, node: &Node, ins: &[Option<Tensor>]) -> Result<Vec<Tensor>> {
        let x = inp(ins, 0)?;
        let skip = inp(ins, 1)?;
        let gamma = inp(ins, 2)?;
        let beta = opt(ins, 3);
        let bias = opt(ins, 4);
        let eps = node.attr_f64("epsilon", 1e-12);
        let mut sum = x.add(skip);
        if let Some(b) = bias {
            sum = sum.add(b);
        }
        let ndim = sum.shape().as_slice().len();
        let axis = vec![ndim - 1];
        let mean = sum.mean(Some(&axis), true);
        let inv = sum
            .sub(&mean)
            .square()
            .mean(Some(&axis), true)
            .add(eps)
            .rsqrt();
        let normed = sum.sub(&mean).mul(&inv).mul(gamma);
        let out = match beta {
            Some(b) => normed.add(b),
            None => normed,
        };
        let n = node.outputs.len();
        let mut outs = Vec::with_capacity(n);
        outs.push(out);
        if n > 1 {
            outs.push(mean);
        }
        if n > 2 {
            outs.push(inv);
        }
        if n > 3 {
            outs.push(sum);
        }
        Ok(outs)
    }

    fn embed_layer_norm_op(&mut self, node: &Node, ins: &[Option<Tensor>]) -> Result<Vec<Tensor>> {
        let input_ids = inp(ins, 0)?;
        let segment_ids = opt(ins, 1);
        let word = inp(ins, 2)?;
        let pos = inp(ins, 3)?;
        let segment = opt(ins, 4);
        let gamma = opt(ins, 5);
        let beta = opt(ins, 6);
        let mask = opt(ins, 7);
        let position_ids = opt(ins, 8);
        let eps = node.attr_f64("epsilon", 1e-12);
        if mask.is_some() {
            return Err(OnnxError::Unsupported(
                "EmbedLayerNormalization mask".into(),
            ));
        }
        if segment_ids.is_some() != segment.is_some() {
            return Err(OnnxError::Shape(
                "EmbedLayerNormalization segment inputs".into(),
            ));
        }
        let shape = input_ids.shape().into_vec();
        let seq = shape[1].clone().as_i64().unwrap_or(0);
        let pos_ids = match position_ids {
            Some(p) => p.clone(),
            None => Tensor::arange(&self.ctx, 0.0, Some(seq as f64), 1.0, Some(dtype::INT64))
                .reshape([1i64, seq])
                .expand(shape.clone()),
        };
        let embed = |ids: &Tensor, w: &Tensor| w.getitem(&[Ix::Tensor(ids.clone())]);
        let mut sum = embed(input_ids, word).add(embed(&pos_ids, pos));
        if let (Some(ids), Some(w)) = (segment_ids, segment) {
            sum = sum.add(embed(ids, w));
        }
        let ndim = sum.shape().as_slice().len();
        let axis = vec![ndim - 1];
        let mean = sum.mean(Some(&axis), true);
        let inv = sum
            .sub(&mean)
            .square()
            .mean(Some(&axis), true)
            .add(eps)
            .rsqrt();
        let mut out = sum.sub(&mean).mul(&inv);
        if let Some(g) = gamma {
            out = out.mul(g);
        }
        if let Some(b) = beta {
            out = out.add(b);
        }
        let n = node.outputs.len();
        let mut outs = Vec::with_capacity(n);
        outs.push(out);
        if n > 1 {
            if !node.outputs[1].is_empty() {
                return Err(OnnxError::Unsupported(
                    "EmbedLayerNormalization mask_index output".into(),
                ));
            }
            outs.push(sum.clone());
        }
        if n > 2 {
            outs.push(sum);
        }
        Ok(outs)
    }

    fn rotary_embedding_op(&mut self, node: &Node, ins: &[Option<Tensor>]) -> Result<Vec<Tensor>> {
        let x = inp(ins, 0)?;
        let cos = inp(ins, 1)?;
        let sin = inp(ins, 2)?;
        let pos = opt(ins, 3);
        let interleaved = node.attr_i64("interleaved", 0) != 0;
        let num_heads = node.attr_i64("num_heads", 0);
        let rot_dim = node.attr_i64("rotary_embedding_dim", 0);
        let orig_rank = x.shape().as_slice().len();
        let x4 = match orig_rank {
            4 => x.permute(&[0, 2, 1, 3]),
            3 => {
                if num_heads <= 0 {
                    return Err(OnnxError::Unsupported(
                        "RotaryEmbedding 3-D input needs num_heads".into(),
                    ));
                }
                let sh = x.shape().into_vec();
                let b = sh[0].clone().as_i64().unwrap_or(0);
                let s = sh[1].clone().as_i64().unwrap_or(0);
                let h = sh[2].clone().as_i64().unwrap_or(0);
                if h % num_heads != 0 {
                    return Err(OnnxError::Shape("RotaryEmbedding hidden/heads".into()));
                }
                x.reshape([b, s, num_heads, h / num_heads])
            }
            _ => {
                return Err(OnnxError::Unsupported(format!(
                    "RotaryEmbedding rank {orig_rank}"
                )));
            }
        };
        let head = x4.shape().as_slice()[3].clone().as_i64().unwrap_or(0);
        let rd = if rot_dim > 0 { rot_dim } else { head };
        if rd % 2 != 0 || rd > head {
            return Err(OnnxError::Shape("RotaryEmbedding rot_dim".into()));
        }
        let half = (rd / 2) as i64;
        let mut xr_ix = vec![Ix::All; 4];
        xr_ix[3] = Ix::Range {
            start: Some(0),
            stop: Some(rd),
            step: 1,
        };
        let x_rot = x4.getitem(&xr_ix);
        let mut xp_ix = vec![Ix::All; 4];
        xp_ix[3] = Ix::Range {
            start: Some(rd),
            stop: None,
            step: 1,
        };
        let x_pass = x4.getitem(&xp_ix);
        let prep = |cache: &Tensor| -> Tensor {
            let mut t = match pos {
                Some(p) => cache.getitem(&[Ix::Tensor(p.clone())]),
                None => {
                    let seq = x4.shape().as_slice()[2].clone().as_i64().unwrap_or(0);
                    cache.getitem(&[
                        Ix::Range {
                            start: Some(0),
                            stop: Some(seq),
                            step: 1,
                        },
                        Ix::All,
                    ])
                }
            };
            let rank = t.shape().as_slice().len();
            let mut cix = vec![Ix::All; rank];
            cix[rank - 1] = Ix::Range {
                start: Some(0),
                stop: Some(half),
                step: 1,
            };
            t = t.getitem(&cix);
            if t.shape().as_slice().len() == 2 {
                t = t.unsqueeze(0);
            }
            t.unsqueeze(2)
        };
        let cos = prep(cos);
        let sin = prep(sin);
        let (x1, x2) = if interleaved {
            let mut ex = vec![Ix::All; 4];
            ex[3] = Ix::Range {
                start: Some(0),
                stop: None,
                step: 2,
            };
            let mut ox = vec![Ix::All; 4];
            ox[3] = Ix::Range {
                start: Some(1),
                stop: None,
                step: 2,
            };
            (x_rot.getitem(&ex), x_rot.getitem(&ox))
        } else {
            let mut a = vec![Ix::All; 4];
            a[3] = Ix::Range {
                start: Some(0),
                stop: Some(half),
                step: 1,
            };
            let mut b = vec![Ix::All; 4];
            b[3] = Ix::Range {
                start: Some(half),
                stop: None,
                step: 1,
            };
            (x_rot.getitem(&a), x_rot.getitem(&b))
        };
        let real = x1.mul(&cos).sub(x2.mul(&sin));
        let imag = x1.mul(&sin).add(x2.mul(&cos));
        let rotated = if interleaved {
            real.unsqueeze(-1)
                .cat(&[imag.unsqueeze(-1)], -1)
                .flatten(-2, -1)
        } else {
            real.cat(&[imag], -1)
        };
        let output = rotated.cat(&[x_pass], -1);
        let output = if orig_rank == 3 {
            output.flatten(2, 3)
        } else {
            output.permute(&[0, 2, 1, 3])
        };
        Ok(vec![output])
    }

    fn affine_grid_op(&mut self, node: &Node, ins: &[Option<Tensor>]) -> Result<Vec<Tensor>> {
        let theta = inp(ins, 0)?;
        let size = self.i64s(inp(ins, 1)?)?;
        let align = node.attr_i64("align_corners", 0) != 0;
        let tshape: Vec<i64> = theta
            .shape()
            .as_slice()
            .iter()
            .map(|s| s.clone().as_i64().unwrap_or(0))
            .collect();
        if tshape.len() != 3 {
            return Err(OnnxError::Shape("AffineGrid theta rank".into()));
        }
        let n = tshape[0];
        let ndim = tshape[1] as usize;
        if size.len() != ndim + 2 {
            return Err(OnnxError::Shape("AffineGrid size length".into()));
        }
        let spatial: Vec<i64> = size[2..].to_vec();
        let grid = |steps: i64| -> Tensor {
            if align {
                Tensor::linspace(&self.ctx, -1.0, 1.0, steps, None)
            } else {
                let a = -1.0 + 1.0 / steps as f64;
                Tensor::linspace(&self.ctx, a, 1.0 - 1.0 / steps as f64, steps, None)
            }
        };
        let mut coords: Vec<Tensor> = Vec::with_capacity(ndim);
        for d in 0..ndim {
            let steps = spatial[ndim - 1 - d];
            let mut rshape = vec![1i64; ndim];
            rshape[ndim - 1 - d] = steps;
            coords.push(grid(steps).reshape(rshape).expand(spatial.clone()));
        }
        let ones = coords[0].mul(0.0).add(1.0);
        let base = coords[0].unsqueeze(-1);
        let mut rest: Vec<Tensor> = coords[1..].iter().map(|t| t.unsqueeze(-1)).collect();
        rest.push(ones.unsqueeze(-1));
        let base = base.cat(&rest, -1);
        let prod: i64 = spatial.iter().product();
        let flat = base
            .reshape([1i64, prod, (ndim + 1) as i64])
            .expand([n, prod, (ndim + 1) as i64]);
        let out = flat.matmul(theta.transpose(1, 2));
        let mut oshape = vec![n];
        oshape.extend_from_slice(&spatial);
        oshape.push(ndim as i64);
        Ok(vec![out.reshape(oshape)])
    }

    fn tensor_scatter_op(
        &mut self,
        past: &Tensor,
        update: &Tensor,
        write: Option<&Tensor>,
        axis: i64,
        mode: &str,
    ) -> Result<Tensor> {
        let rank = past.shape().as_slice().len();
        let ax = resolve_index(axis, rank as i64) as usize;
        if ax == 0 {
            return Err(OnnxError::Unsupported("TensorScatter axis 0".into()));
        }
        if !matches!(mode, "linear" | "circular") {
            return Err(OnnxError::Unsupported(format!("TensorScatter mode `{mode}`")));
        }
        if ax != rank - 2 {
            // The KV layout (-2) uses the fast one-hot/matmul graph. Other axes
            // are rare and hit a pre-existing symbolic-load codegen path when
            // permuted, so scatter them on the host.
            return self.tensor_scatter_host(past, update, write, ax, mode);
        }
        let shape: Vec<i64> = past
            .shape()
            .as_slice()
            .iter()
            .map(|s| s.clone().as_i64().unwrap_or(0))
            .collect();
        let b = shape[0];
        let d = shape[rank - 2];
        let last = shape[rank - 1];
        let pfx: i64 = shape[1..rank - 2].iter().product();
        let p = b.saturating_mul(pfx);
        let ushape: Vec<i64> = update
            .shape()
            .as_slice()
            .iter()
            .map(|s| s.clone().as_i64().unwrap_or(0))
            .collect();
        let seq = ushape[rank - 2];
        if seq <= 0 || p <= 0 || d <= 0 {
            return Err(OnnxError::Shape("TensorScatter empty update".into()));
        }
        let past_f = past.reshape([p, d, last]).cast(dtype::FLOAT32);
        let upd_f = update.reshape([p, seq, last]).cast(dtype::FLOAT32);
        let write = match write {
            Some(w) => w.cast(dtype::INT64),
            None => Tensor::full_const(
                &mut self.ctx.borrow_mut(),
                [b],
                ConstVal::Int(0),
                Some(dtype::INT64),
            ),
        };
        let w = write.reshape([b, 1]).expand([b, pfx]).reshape([p, 1]);
        let sidx = Tensor::arange(
            &self.ctx,
            0.0,
            Some(seq as f64),
            1.0,
            Some(dtype::INT64),
        )
        .reshape([1, seq]);
        let mut cidx = w.add(&sidx);
        if mode == "circular" {
            cidx = cidx.mod_(d);
        } else {
            let mx = cidx
                .max(None, false)
                .to_list::<i64>(&mut self.sess)
                .first()
                .copied()
                .unwrap_or(0);
            if mx >= d {
                return Err(OnnxError::Runtime(
                    "TensorScatter write exceeds max_sequence_length".into(),
                ));
            }
        }
        let mask = cidx
            .unsqueeze(-1)
            .one_hot_along_dim(d, -1)
            .cast(dtype::FLOAT32);
        let scattered = mask.transpose(1, 2).matmul(&upd_f);
        let has = mask.max(Some(&[1]), false);
        let keep = has.mul(-1.0).add(1.0).unsqueeze(-1);
        let present = past_f.mul(&keep).add(&scattered);
        Ok(present.cast(past.dtype()).reshape(shape))
    }

    fn tensor_scatter_host(
        &mut self,
        past: &Tensor,
        update: &Tensor,
        write: Option<&Tensor>,
        ax: usize,
        mode: &str,
    ) -> Result<Tensor> {
        let shape: Vec<i64> = past
            .shape()
            .as_slice()
            .iter()
            .map(|s| s.clone().as_i64().unwrap_or(0))
            .collect();
        let rank = shape.len();
        let ushape: Vec<i64> = update
            .shape()
            .as_slice()
            .iter()
            .map(|s| s.clone().as_i64().unwrap_or(0))
            .collect();
        let b = shape[0];
        let max_seq = shape[ax];
        let seq = ushape[ax];
        if seq <= 0 || max_seq <= 0 {
            return Err(OnnxError::Shape("TensorScatter empty update".into()));
        }
        let pv = past.to_list::<f64>(&mut self.sess);
        let uv = update.to_list::<f64>(&mut self.sess);
        let wv: Vec<i64> = match write {
            Some(w) => w.to_list::<i64>(&mut self.sess),
            None => vec![0; b as usize],
        };
        if wv.len() != b as usize {
            return Err(OnnxError::Shape("TensorScatter write_indices length".into()));
        }
        let mut strides = vec![1i64; rank];
        for d in (0..rank.saturating_sub(1)).rev() {
            strides[d] = strides[d + 1] * shape[d + 1];
        }
        let total: i64 = ushape.iter().product();
        let mut out = pv.clone();
        let mut coords = vec![0i64; rank];
        for flat in 0..total {
            let mut rem = flat;
            for d in (0..rank).rev() {
                coords[d] = rem % ushape[d];
                rem /= ushape[d];
            }
            let bi = coords[0] as usize;
            let mut cs = wv[bi] + coords[ax];
            if mode == "circular" {
                cs = cs.rem_euclid(max_seq);
            } else if cs < 0 || cs >= max_seq {
                return Err(OnnxError::Runtime(
                    "TensorScatter write exceeds max_sequence_length".into(),
                ));
            }
            let mut ci = coords.clone();
            ci[ax] = cs;
            let di: i64 = ci.iter().zip(&strides).map(|(c, s)| c * s).sum();
            out[di as usize] = uv[flat as usize];
        }
        self.build_like(past, &out, &shape)
    }

    fn gather_elements(&mut self, x: &Tensor, idx: &Tensor, axis: usize) -> Result<Tensor> {
        // Build the gather as one advanced-index `getitem`: axis coordinates
        // come from `indices`, every other axis from an `arange` broadcast to
        // the output shape. No index values are read on the host, so the op
        // stays inside the replay plan.
        let xshape: Vec<i64> = x
            .shape()
            .as_slice()
            .iter()
            .map(|s| s.clone().as_i64().unwrap_or(0))
            .collect();
        let ishape: Vec<i64> = idx
            .shape()
            .as_slice()
            .iter()
            .map(|s| s.clone().as_i64().unwrap_or(0))
            .collect();
        let rank = ishape.len();
        if xshape.len() != rank {
            return Err(OnnxError::Shape("GatherElements rank mismatch".into()));
        }
        if axis >= rank {
            return Err(OnnxError::Shape("GatherElements axis".into()));
        }
        // Negative indices wrap like ONNX (`i + dim`).
        let dim = xshape[axis];
        let idx = idx.cast(dtype::INT64);
        let idx = idx.add(idx.lt(0).cast(dtype::INT64).mul(dim));
        let mut ixs: Vec<Ix> = Vec::with_capacity(rank);
        for d in 0..rank {
            if d == axis {
                ixs.push(Ix::Tensor(idx.clone()));
            } else {
                let ar = Tensor::arange(
                    &self.ctx,
                    0.0,
                    Some(ishape[d] as f64),
                    1.0,
                    Some(dtype::INT32),
                );
                let mut sh = vec![1i64; rank];
                sh[d] = ishape[d];
                ixs.push(Ix::Tensor(ar.reshape(sh)));
            }
        }
        Ok(x.getitem(&ixs).reshape(ishape))
    }
}

/// tinygrad's `xpow`: `exp2(e * log2(|b|))` with negative-base, odd-exponent,
/// NaN and `x**0` fixups (`codegen/decomp/transcendental.py:257`). Used for the
/// general ONNX `Pow` case so no backend needs a native ALU `POW`.
fn xpow(base: &Tensor, exponent: &Tensor) -> Tensor {
    let abs = base.lt(0).where_(base.neg(), base);
    let ret = abs.log2().mul(exponent).exp2();
    let exp_int = exponent.cast(dtype::INT32).cast(exponent.dtype());
    let non_int = exponent.ne(&exp_int);
    let is_odd = exponent.abs().cast(dtype::INT32).mod_(2).ne(0);
    let nan = base.const_like(ConstVal::Float(f64::NAN), None);
    let neg_base = non_int.where_(
        base.ne(f64::NEG_INFINITY).where_(nan, &ret),
        is_odd.where_(ret.neg(), &ret),
    );
    let out = exponent.eq(0).where_(
        ret.const_like(ConstVal::Float(1.0), None),
        base.lt(0).where_(neg_base, &ret),
    );
    out.cast(base.dtype())
}

fn pow_positive(x: &Tensor, e: u32) -> Tensor {
    let mut r = x.ones_like();
    for _ in 0..e {
        r = r.mul(x);
    }
    r
}

fn default_softmax_axis(opset: i64) -> i64 {
    if opset < 13 { 1 } else { -1 }
}

fn resolve_index(v: i64, n: i64) -> i64 {
    if v < 0 { (v + n).max(0) } else { v }
}

fn resolve_axis(x: &Tensor, axis: i64) -> usize {
    let n = x.shape().as_slice().len() as i64;
    resolve_index(axis, n) as usize
}

fn broadcast_shape(a: &[i64], b: &[i64]) -> Result<Vec<i64>> {
    let n = a.len().max(b.len());
    let mut out = vec![1i64; n];
    for i in 0..n {
        let av = if i + a.len() >= n {
            a[i + a.len() - n]
        } else {
            1
        };
        let bv = if i + b.len() >= n {
            b[i + b.len() - n]
        } else {
            1
        };
        out[i] = if av == bv || bv == 1 {
            av
        } else if av == 1 || av == -1 {
            bv
        } else if bv == -1 {
            av
        } else {
            return Err(OnnxError::Shape(format!(
                "cannot broadcast {a:?} with {b:?}"
            )));
        };
    }
    Ok(out)
}

fn expand_pads(pads: &[i64], axes: Option<&[i64]>, ndim: usize) -> Result<Vec<i64>> {
    if axes.is_none() {
        if pads.len() != ndim * 2 {
            return Err(OnnxError::Shape(format!(
                "Pad expects {} values, got {}",
                ndim * 2,
                pads.len()
            )));
        }
        return Ok(pads.to_vec());
    }
    let axes = axes.unwrap();
    let mut full = vec![0i64; ndim * 2];
    let n = axes.len();
    for (i, &a) in axes.iter().enumerate() {
        let a = a.rem_euclid(ndim as i64) as usize;
        full[a] = pads[i];
        full[a + ndim] = pads[i + n];
    }
    Ok(full)
}

// ---------------------------------------------------------------------------
// Convolution / pooling / normalization / resize
// ---------------------------------------------------------------------------

fn ceildiv(a: i64, b: i64) -> i64 {
    -((-a).div_euclid(b))
}

/// ONNX pads `[x1_begin, x2_begin, ..., x1_end, x2_end]` ->
/// tinygrad flat `[x1_begin, x1_end, x2_begin, x2_end, ...]` in *spatial*
/// order (the caller passes this through `resolve_pool_pads`).
fn onnx_pads_to_tiny(pads: &[i64]) -> Vec<i64> {
    let n = pads.len() / 2;
    let mut out = Vec::with_capacity(pads.len());
    for i in (0..n).rev() {
        out.push(pads[i]);
        out.push(pads[i + n]);
    }
    out
}

fn attr_list(node: &Node, name: &str, n: usize, default: i64) -> Vec<i64> {
    match node.attr_i64s(name) {
        Some(v) if v.len() == n => v,
        Some(v) if v.len() == 1 => vec![v[0]; n],
        _ => vec![default; n],
    }
}

fn resolve_auto_pads(
    input: &[i64],
    kernel: &[i64],
    strides: &[i64],
    dilations: &[i64],
    pads: &[i64],
    auto_pad: &str,
) -> Result<Vec<i64>> {
    let n = kernel.len();
    if auto_pad == "VALID" {
        return Ok(vec![0; 2 * n]);
    }
    if auto_pad == "NOTSET" || auto_pad.is_empty() {
        if pads.is_empty() {
            return Ok(vec![0; 2 * n]);
        }
        if pads.len() != 2 * n {
            return Err(OnnxError::Shape(format!(
                "expected {} pads, got {}",
                2 * n,
                pads.len()
            )));
        }
        return Ok(onnx_pads_to_tiny(pads));
    }
    if auto_pad != "SAME_UPPER" && auto_pad != "SAME_LOWER" {
        return Err(OnnxError::Unsupported(format!("auto_pad `{auto_pad}`")));
    }
    let mut begins = Vec::with_capacity(n);
    let mut ends = Vec::with_capacity(n);
    for i in 0..n {
        let inp = input[input.len() - n + i];
        let out = ceildiv(inp, strides[i]);
        let total = ((out - 1) * strides[i] + (dilations[i] * (kernel[i] - 1) + 1) - inp).max(0);
        let (b, e) = if auto_pad == "SAME_UPPER" {
            (total / 2, total - total / 2)
        } else {
            (total - total / 2, total / 2)
        };
        begins.push(b);
        ends.push(e);
    }
    let onnx: Vec<i64> = begins.into_iter().chain(ends).collect();
    Ok(onnx_pads_to_tiny(&onnx))
}

impl OnnxModel {
    fn conv_op(&mut self, node: &Node, ins: &[Option<Tensor>]) -> Result<Tensor> {
        let x = inp(ins, 0)?;
        let w = inp(ins, 1)?;
        let b = opt(ins, 2);
        let xshape: Vec<i64> = x
            .shape()
            .as_slice()
            .iter()
            .map(|s| s.clone().as_i64().unwrap_or(0))
            .collect();
        let wshape: Vec<i64> = w
            .shape()
            .as_slice()
            .iter()
            .map(|s| s.clone().as_i64().unwrap_or(0))
            .collect();
        let n = wshape.len() - 2;
        let kernel = node
            .attr_i64s("kernel_shape")
            .filter(|v| v.len() == n)
            .unwrap_or_else(|| wshape[2..].to_vec());
        let strides = attr_list(node, "strides", n, 1);
        let dilations = attr_list(node, "dilations", n, 1);
        let pads = node.attr_i64s("pads").unwrap_or_default();
        let auto_pad = node.attr_str("auto_pad", "NOTSET");
        let padding = resolve_auto_pads(&xshape, &kernel, &strides, &dilations, &pads, auto_pad)?;
        let groups = node.attr_i64("group", 1);
        let s = strides[0];
        let d = dilations[0];
        if strides.iter().any(|&v| v != s) || dilations.iter().any(|&v| v != d) {
            // Per-dimension stride/dilation: expand the dilation into zeros in
            // the kernel, run the stride-1 convolution, then subsample. This
            // is exact for 2-D and keeps the group/dilation/pad semantics.
            if n != 2 {
                return Err(OnnxError::Unsupported(
                    "Conv with per-dimension strides/dilations".into(),
                ));
            }
            let (sh, sw) = (strides[0], strides[1]);
            let (dh, dw) = (dilations[0], dilations[1]);
            let (kh, kw) = (wshape[2], wshape[3]);
            let (eh, ew) = ((kh - 1) * dh + 1, (kw - 1) * dw + 1);
            let w2 = if eh == kh && ew == kw {
                w.clone()
            } else {
                let vals = w.to_list::<f64>(&mut self.sess);
                let mut exp = vec![0.0f64; (wshape[0] * wshape[1] * eh * ew) as usize];
                for co in 0..wshape[0] {
                    for ci in 0..wshape[1] {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let src = (((co * wshape[1] + ci) * kh + ky) * kw + kx) as usize;
                                let dst = (((co * wshape[1] + ci) * eh + ky * dh) * ew + kx * dw)
                                    as usize;
                                exp[dst] = vals[src];
                            }
                        }
                    }
                }
                self.build_like(w, &exp, &[wshape[0], wshape[1], eh, ew])?
            };
            let out = x.conv2d(&w2, b, groups, 1, 1, &padding, None);
            let mut ixs = vec![Ix::All; 4];
            ixs[2] = Ix::Range {
                start: Some(0),
                stop: None,
                step: sh,
            };
            ixs[3] = Ix::Range {
                start: Some(0),
                stop: None,
                step: sw,
            };
            return Ok(out.getitem(&ixs));
        }
        // Use the tensor-core kxk kernel when the input is already realized
        // (notably the stem); `conv2d_with` falls back without a forced realize,
        // so later layers keep the fused/captured graph.
        Ok(x.conv2d_with(w, b, groups, s, d, &padding, None, &mut self.sess))
    }

    fn conv_transpose_op(&mut self, node: &Node, ins: &[Option<Tensor>]) -> Result<Tensor> {
        let x = inp(ins, 0)?;
        let w = inp(ins, 1)?;
        let b = opt(ins, 2);
        let xshape: Vec<i64> = x
            .shape()
            .as_slice()
            .iter()
            .map(|s| s.clone().as_i64().unwrap_or(0))
            .collect();
        let wshape: Vec<i64> = w
            .shape()
            .as_slice()
            .iter()
            .map(|s| s.clone().as_i64().unwrap_or(0))
            .collect();
        let n = wshape.len() - 2;
        let kernel = node
            .attr_i64s("kernel_shape")
            .filter(|v| v.len() == n)
            .unwrap_or_else(|| wshape[2..].to_vec());
        let strides = attr_list(node, "strides", n, 1);
        let dilations = attr_list(node, "dilations", n, 1);
        let output_padding = attr_list(node, "output_padding", n, 0);
        let auto_pad = node.attr_str("auto_pad", "NOTSET");
        let mut pads = node.attr_i64s("pads");
        let output_shape = node.attr_i64s("output_shape");
        let input_spatial = &xshape[2..];
        if let Some(os) = &output_shape {
            if os.len() != n {
                return Err(OnnxError::Shape("ConvTranspose output_shape length".into()));
            }
            let totals: Vec<i64> = (0..n)
                .map(|i| {
                    strides[i] * (input_spatial[i] - 1)
                        + output_padding[i]
                        + (dilations[i] * (kernel[i] - 1) + 1)
                        - os[i]
                })
                .collect();
            pads = Some(split_conv_transpose_pads(&totals, auto_pad));
        }
        let pads = match pads {
            Some(p) => onnx_pads_to_tiny(&p),
            None => {
                let out_spatial: Vec<i64> = (0..n).map(|i| input_spatial[i] * strides[i]).collect();
                let totals: Vec<i64> = (0..n)
                    .map(|i| {
                        strides[i] * (input_spatial[i] - 1)
                            + output_padding[i]
                            + (dilations[i] * (kernel[i] - 1) + 1)
                            - out_spatial[i]
                    })
                    .collect();
                if auto_pad == "NOTSET" || auto_pad.is_empty() {
                    vec![0; 2 * n]
                } else {
                    onnx_pads_to_tiny(&split_conv_transpose_pads(&totals, auto_pad))
                }
            }
        };
        let s = strides[0];
        let d = dilations[0];
        let op = output_padding[0];
        let groups = node.attr_i64("group", 1);
        if strides.iter().all(|&v| v == s)
            && dilations.iter().all(|&v| v == d)
            && output_padding.iter().all(|&v| v == op)
        {
            return Ok(x.conv_transpose2d(w, b, groups, s, d, &pads, op));
        }
        // Per-dimension attributes: expand the dilation into the kernel, insert
        // zeros between input samples for the strides, run a stride-1/dilation-1
        // transposed convolution, then append the per-axis output padding.
        let w2 = expand_kernel_dilation(w, &dilations);
        let x2 = zero_insert_strides(x, &strides);
        let mut out = x2.conv_transpose2d(&w2, b, groups, 1, 1, &pads, 0);
        if output_padding.iter().any(|&v| v > 0) {
            let mut parg = vec![(0i64, 0i64); out.shape().as_slice().len()];
            for (i, &v) in output_padding.iter().enumerate() {
                parg[2 + i] = (0, v);
            }
            out = out.pad(&parg);
        }
        Ok(out)
    }

    fn max_pool_op(&mut self, node: &Node, ins: &[Option<Tensor>]) -> Result<Vec<Tensor>> {
        let x = inp(ins, 0)?;
        let xshape: Vec<i64> = x
            .shape()
            .as_slice()
            .iter()
            .map(|s| s.clone().as_i64().unwrap_or(0))
            .collect();
        let kernel = node
            .attr_i64s("kernel_shape")
            .ok_or_else(|| OnnxError::Runtime("MaxPool needs kernel_shape".into()))?;
        let n = kernel.len();
        let strides = attr_list(node, "strides", n, 1);
        let dilations = attr_list(node, "dilations", n, 1);
        let pads = node.attr_i64s("pads").unwrap_or_default();
        let auto_pad = node.attr_str("auto_pad", "NOTSET");
        let padding = resolve_auto_pads(&xshape, &kernel, &strides, &dilations, &pads, auto_pad)?;
        let ceil = node.attr_i64("ceil_mode", 0) != 0;
        let want_idx = node.outputs.len() > 1;
        let (out, idx) = x.max_pool2d(
            &kernel,
            Some(&strides),
            &dilations,
            &padding,
            ceil,
            want_idx,
        );
        if want_idx {
            let mut idx =
                idx.ok_or_else(|| OnnxError::Runtime("MaxPool indices missing".into()))?;
            if node.attr_i64("storage_order", 0) != 0 {
                idx = idx.transpose(-2, -1);
            }
            Ok(vec![out, idx.cast(dtype::INT64)])
        } else {
            Ok(vec![out])
        }
    }

    fn avg_pool_op(&mut self, node: &Node, ins: &[Option<Tensor>]) -> Result<Tensor> {
        let x = inp(ins, 0)?;
        let xshape: Vec<i64> = x
            .shape()
            .as_slice()
            .iter()
            .map(|s| s.clone().as_i64().unwrap_or(0))
            .collect();
        let kernel = node
            .attr_i64s("kernel_shape")
            .ok_or_else(|| OnnxError::Runtime("AveragePool needs kernel_shape".into()))?;
        let n = kernel.len();
        let strides = attr_list(node, "strides", n, 1);
        let dilations = vec![1; n];
        let pads = node.attr_i64s("pads").unwrap_or_default();
        let auto_pad = node.attr_str("auto_pad", "NOTSET");
        let padding = resolve_auto_pads(&xshape, &kernel, &strides, &dilations, &pads, auto_pad)?;
        let ceil = node.attr_i64("ceil_mode", 0) != 0;
        let count_include_pad = node.attr_i64("count_include_pad", 1) != 0;
        Ok(x.avg_pool2d(
            &kernel,
            Some(&strides),
            1,
            &padding,
            ceil,
            count_include_pad,
        ))
    }

    fn group_norm(
        &mut self,
        x: &Tensor,
        scale: &Tensor,
        bias: &Tensor,
        groups: i64,
        eps: f64,
    ) -> Result<Tensor> {
        let shape: Vec<i64> = x
            .shape()
            .as_slice()
            .iter()
            .map(|s| s.clone().as_i64().unwrap_or(0))
            .collect();
        let (n, c) = (shape[0], shape[1]);
        let xr = x.reshape([n, groups, -1]);
        let normed = xr
            .cast(dtype::FLOAT32)
            .layernorm(-1, eps)
            .cast(x.dtype())
            .reshape(shape.clone());
        let mut pshape = vec![1i64, c];
        pshape.extend(std::iter::repeat_n(1i64, shape.len() - 2));
        Ok(normed
            .mul(scale.reshape(pshape.clone()))
            .add(bias.reshape(pshape)))
    }

    fn lrn(&mut self, x: &Tensor, size: i64, alpha: f64, beta: f64, bias: f64) -> Result<Tensor> {
        let shape: Vec<i64> = x
            .shape()
            .as_slice()
            .iter()
            .map(|s| s.clone().as_i64().unwrap_or(0))
            .collect();
        if shape.len() < 3 {
            return Err(OnnxError::Shape("LRN expects NCHW".into()));
        }
        let (b, c) = (shape[0], shape[1]);
        let hw: i64 = shape[2..].iter().product();
        let sq = x.square().reshape([b, 1, c, hw]);
        let mut pads = [(0i64, 0i64); 4];
        pads[2] = ((size - 1) / 2, size / 2);
        let p: Vec<(Option<i64>, Option<i64>)> =
            pads.iter().map(|(a, b)| (Some(*a), Some(*b))).collect();
        let pooled = sq
            .pad_constant(&p, ConstVal::Float(0.0))
            .avg_pool2d(&[size, 1], Some(&[1, 1]), 1, &[], false, true)
            .reshape(shape.clone());
        let denom = self.pow_scalar(&pooled.mul(alpha).add(bias), beta);
        Ok(x.div(&denom))
    }

    fn resize_op(&mut self, node: &Node, ins: &[Option<Tensor>]) -> Result<Tensor> {
        let x = inp(ins, 0)?;
        let rank = x.shape().as_slice().len();
        let sizes = match opt(ins, 3) {
            Some(t) if t.numel() > 0 => Some(self.i64s(t)?),
            _ => None,
        };
        let scales = match opt(ins, 2) {
            Some(t) if t.numel() > 0 => Some(self.f64s(t)?),
            _ => None,
        };
        let in_shape: Vec<i64> = x
            .shape()
            .as_slice()
            .iter()
            .map(|s| s.clone().as_i64().unwrap_or(0))
            .collect();
        // Like tinygrad, resize the spatial dims: batch/channel are never
        // resized. ONNX `axes` overrides the default (trailing dims).
        let axes: Vec<usize> = match node.attr_i64s("axes") {
            Some(a) => a
                .into_iter()
                .map(|v| resolve_index(v, rank as i64) as usize)
                .collect(),
            None => {
                let k = sizes
                    .as_ref()
                    .map(Vec::len)
                    .or_else(|| scales.as_ref().map(Vec::len))
                    .ok_or_else(|| OnnxError::Runtime("Resize needs sizes or scales".into()))?;
                if k > rank {
                    return Err(OnnxError::Shape(
                        "Resize target rank exceeds input rank".into(),
                    ));
                }
                let start = rank.saturating_sub(k).max(2.min(rank));
                (start..rank).collect()
            }
        };
        let mut targets: Vec<i64> = if let Some(sizes) = &sizes {
            if sizes.len() == rank {
                axes.iter().map(|&a| sizes[a]).collect()
            } else if sizes.len() == axes.len() {
                sizes.clone()
            } else {
                return Err(OnnxError::Shape("Resize sizes length".into()));
            }
        } else if let Some(scales) = &scales {
            if scales.len() == rank {
                axes.iter()
                    .map(|&a| ((in_shape[a] as f64) * scales[a]).floor() as i64)
                    .collect()
            } else if scales.len() == axes.len() {
                axes.iter()
                    .zip(scales.iter())
                    .map(|(&a, s)| ((in_shape[a] as f64) * *s).floor() as i64)
                    .collect()
            } else {
                return Err(OnnxError::Shape("Resize scales length".into()));
            }
        } else {
            return Err(OnnxError::Runtime("Resize needs sizes or scales".into()));
        };
        let policy = node.attr_str("keep_aspect_ratio_policy", "stretch");
        if policy != "stretch" {
            if sizes.is_none() {
                return Err(OnnxError::Unsupported(
                    "Resize keep_aspect_ratio_policy with scales".into(),
                ));
            }
            let ratios: Vec<f64> = targets
                .iter()
                .zip(axes.iter())
                .map(|(&t, &a)| t as f64 / in_shape[a] as f64)
                .collect();
            let scale = match policy {
                "not_larger" => ratios.iter().cloned().fold(f64::INFINITY, f64::min),
                "not_smaller" => ratios.iter().cloned().fold(0.0, f64::max),
                other => {
                    return Err(OnnxError::Unsupported(format!(
                        "Resize keep_aspect_ratio_policy `{other}`"
                    )));
                }
            };
            for (i, &a) in axes.iter().enumerate() {
                targets[i] = if policy == "not_larger" {
                    (in_shape[a] as f64 * scale).floor() as i64
                } else {
                    (in_shape[a] as f64 * scale).ceil() as i64
                };
            }
        }
        // Drop axes whose size is unchanged (avoids a 0/0 in the linear
        // coordinate transform for size-1 dimensions such as N/C).
        let mut axes = axes;
        let mut i = 0;
        while i < axes.len() {
            if targets[i] == in_shape[axes[i]] {
                axes.remove(i);
                targets.remove(i);
            } else {
                i += 1;
            }
        }
        if axes.is_empty() {
            return Ok(x.clone());
        }
        let antialias = node.attr_i64("antialias", 0) != 0;
        let mode = node.attr_str("mode", "nearest");
        let mode = match mode {
            "linear" | "bilinear" | "trilinear" => "linear",
            "nearest" => "nearest",
            "cubic" => "cubic",
            other => return Err(OnnxError::Unsupported(format!("Resize mode `{other}`"))),
        };
        let coord = node.attr_str("coordinate_transformation_mode", "half_pixel");
        if !matches!(
            coord,
            "half_pixel"
                | "align_corners"
                | "asymmetric"
                | "pytorch_half_pixel"
                | "half_pixel_symmetric"
                | "tf_half_pixel_for_nn"
        ) {
            return Err(OnnxError::Unsupported(format!(
                "Resize coordinate_transformation_mode `{coord}`"
            )));
        }
        if mode == "nearest" {
            let nearest_mode = node.attr_str("nearest_mode", "round_prefer_floor");
            return self.resize_nearest(x, &axes, &targets, &in_shape, coord, nearest_mode);
        }
        // Permute the resized axes to the end, resize, then invert.
        let others: Vec<i64> = (0..rank)
            .filter(|i| !axes.contains(i))
            .map(|i| i as i64)
            .collect();
        let mut perm: Vec<i64> = others.clone();
        perm.extend(axes.iter().map(|&a| a as i64));
        let xt = if perm.iter().enumerate().all(|(i, &p)| i as i64 == p) {
            x.clone()
        } else {
            x.permute(&perm)
        };
        let resized = if mode == "cubic" {
            let a = node.attr_f64("cubic_coeff_a", -0.75);
            let exclude = node.attr_i64("exclude_outside", 0) != 0;
            xt.interpolate_cubic_coord(&targets, coord, a, exclude, antialias)
        } else {
            xt.interpolate_linear_coord(&targets, coord, antialias)
        };
        if perm.iter().enumerate().all(|(i, &p)| i as i64 == p) {
            Ok(resized)
        } else {
            let mut inv = vec![0i64; rank];
            for (i, &p) in perm.iter().enumerate() {
                inv[p as usize] = i as i64;
            }
            Ok(resized.permute(&inv))
        }
    }
}

impl OnnxModel {
    fn resize_nearest(
        &mut self,
        x: &Tensor,
        axes: &[usize],
        targets: &[i64],
        in_shape: &[i64],
        coord: &str,
        nearest_mode: &str,
    ) -> Result<Tensor> {
        let rank = x.shape().as_slice().len();
        let mut out = x.clone();
        for (&ax, &out_sz) in axes.iter().zip(targets.iter()) {
            let in_sz = in_shape[ax];
            if in_sz <= 0 || out_sz <= 0 {
                return Err(OnnxError::Shape("Resize zero-sized dimension".into()));
            }
            let scale = out_sz as f64 / in_sz as f64;
            let mut idx = Vec::with_capacity(out_sz as usize);
            for i in 0..out_sz {
                let i = i as f64;
                let p = match coord {
                    "half_pixel" => (i + 0.5) / scale - 0.5,
                    "align_corners" => {
                        if out_sz > 1 {
                            i * (in_sz - 1) as f64 / (out_sz - 1) as f64
                        } else {
                            0.0
                        }
                    }
                    "asymmetric" => i / scale,
                    "pytorch_half_pixel" => {
                        if out_sz > 1 {
                            (i + 0.5) / scale - 0.5
                        } else {
                            0.0
                        }
                    }
                    "half_pixel_symmetric" => {
                        let output_dim_scaled = in_sz as f64 * scale;
                        (in_sz as f64 / 2.0) * (1.0 - (out_sz as f64 / output_dim_scaled))
                            + (i + 0.5) / scale
                            - 0.5
                    }
                    other => {
                        return Err(OnnxError::Unsupported(format!(
                            "Resize coordinate_transformation_mode `{other}`"
                        )));
                    }
                };
                let p = p.clamp(0.0, (in_sz - 1) as f64);
                let q = match nearest_mode {
                    "round_prefer_floor" => (p - 0.5).ceil(),
                    "round_prefer_ceil" => (p + 0.5).floor(),
                    "floor" => p.floor(),
                    "ceil" => p.ceil(),
                    other => {
                        return Err(OnnxError::Unsupported(format!(
                            "Resize nearest_mode `{other}`"
                        )));
                    }
                };
                idx.push(q as i64);
            }
            let idx = Tensor::from_slice(&self.ctx, &idx, Some(dtype::INT32), self.device());
            let mut ixs: Vec<Ix> = vec![Ix::All; rank];
            ixs[ax] = Ix::Tensor(idx);
            out = out.getitem(&ixs);
        }
        Ok(out)
    }
}

fn split_conv_transpose_pads(totals: &[i64], auto_pad: &str) -> Vec<i64> {
    let mut begins = Vec::with_capacity(totals.len());
    let mut ends = Vec::with_capacity(totals.len());
    for &t in totals {
        let (b, e) = if auto_pad == "SAME_UPPER" {
            (t / 2, t - t / 2)
        } else {
            (t - t / 2, t / 2)
        };
        begins.push(b);
        ends.push(e);
    }
    begins.into_iter().chain(ends).collect()
}

fn depth_to_space(x: &Tensor, bs: i64, crd: bool) -> Result<Tensor> {
    let s: Vec<i64> = x
        .shape()
        .as_slice()
        .iter()
        .map(|v| v.clone().as_i64().unwrap_or(0))
        .collect();
    if s.len() != 4 {
        return Err(OnnxError::Shape("DepthToSpace expects 4-D input".into()));
    }
    let (b, c, h, w) = (s[0], s[1], s[2], s[3]);
    let c_out = c / (bs * bs);
    if crd {
        let r = x.reshape([b, c_out, bs, bs, h, w]);
        let p = r.permute(&[0, 1, 4, 2, 5, 3]);
        Ok(p.reshape([b, c_out, h * bs, w * bs]))
    } else {
        let r = x.reshape([b, bs, bs, c_out, h, w]);
        let p = r.permute(&[0, 3, 4, 1, 5, 2]);
        Ok(p.reshape([b, c_out, h * bs, w * bs]))
    }
}

fn space_to_depth(x: &Tensor, bs: i64) -> Result<Tensor> {
    let s: Vec<i64> = x
        .shape()
        .as_slice()
        .iter()
        .map(|v| v.clone().as_i64().unwrap_or(0))
        .collect();
    if s.len() != 4 {
        return Err(OnnxError::Shape("SpaceToDepth expects 4-D input".into()));
    }
    let (b, c, h, w) = (s[0], s[1], s[2], s[3]);
    let r = x.reshape([b, c, h / bs, bs, w / bs, bs]);
    let p = r.permute(&[0, 3, 5, 1, 2, 4]);
    Ok(p.reshape([b, bs * bs * c, h / bs, w / bs]))
}
