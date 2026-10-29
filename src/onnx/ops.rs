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
                        &v,
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
                if batch != 0 {
                    return Err(OnnxError::Unsupported("GatherND batch_dims != 0".into()));
                }
                let idx_shape = idx.shape().into_vec();
                let rank = idx_shape.len();
                if rank == 0 {
                    return Err(OnnxError::Unsupported(
                        "GatherND with scalar indices".into(),
                    ));
                }
                let k = idx_shape[rank - 1].clone().as_i64().unwrap_or(0);
                if k == 0 {
                    return one(x.clone());
                }
                let (idx, squeeze_out) = if rank == 1 {
                    (idx.unsqueeze(0), true)
                } else {
                    (idx.clone(), false)
                };
                let parts = idx.split(&vec![1; k as usize], -1);
                let mut ixs: Vec<Ix> = Vec::with_capacity(k as usize);
                for p in parts {
                    ixs.push(Ix::Tensor(p.squeeze(Some(-1))));
                }
                let out = x.getitem(&ixs);
                one(if squeeze_out {
                    out.squeeze(Some(0))
                } else {
                    out
                })
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
            "NonZero" => Err(OnnxError::Unsupported(
                "NonZero (needs masked_select)".into(),
            )),

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
                if ins.len() == 2 {
                    one(inp(ins, 0)?.einsum(inp(ins, 1)?, eq))
                } else {
                    Err(OnnxError::Unsupported(
                        "Einsum with more than two operands".into(),
                    ))
                }
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
                one(x.normalize(p, axis as i64, 1e-12))
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

    fn reverse_axis(&mut self, x: &Tensor, ax: usize) -> Result<Tensor> {
        let n = x.shape().as_slice()[ax].clone().as_i64().unwrap_or(0);
        let idx: Vec<i64> = (0..n).rev().collect();
        let idx = Tensor::from_slice(&self.ctx, &idx, Some(dtype::INT32), self.device());
        let mut ixs: Vec<Ix> = vec![Ix::All; x.shape().as_slice().len()];
        ixs[ax] = Ix::Tensor(idx);
        Ok(x.getitem(&ixs))
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

    fn gather_elements(&mut self, x: &Tensor, idx: &Tensor, axis: usize) -> Result<Tensor> {
        // GatherElements is Gather with full-shape indices: flatten both along
        // the axis and gather per position, then restore the index shape.
        let xshape = x.shape().as_slice().to_vec();
        let ishape = idx.shape().as_slice().to_vec();
        if xshape.len() != ishape.len() {
            return Err(OnnxError::Shape("GatherElements rank mismatch".into()));
        }
        let mut out = Vec::with_capacity(idx.numel() as usize);
        // Read the index tensor once (host-side); it is small in practice.
        let indices = self.i64s(idx)?;
        for (flat, &raw) in indices.iter().enumerate() {
            let mut rem = flat;
            let mut coords = vec![0i64; xshape.len()];
            for d in (0..xshape.len()).rev() {
                let s = ishape[d].clone().as_i64().unwrap_or(1) as usize;
                coords[d] = (rem % s) as i64;
                rem /= s;
            }
            let n = xshape[axis].clone().as_i64().unwrap_or(0);
            let mut i = raw;
            if i < 0 {
                i += n;
            }
            coords[axis] = i;
            out.push(coords);
        }
        let mut ixs: Vec<Ix> = Vec::with_capacity(xshape.len());
        for d in 0..xshape.len() {
            let col: Vec<i64> = out.iter().map(|c| c[d]).collect();
            ixs.push(Ix::Tensor(Tensor::from_slice(
                &self.ctx,
                &col,
                Some(dtype::INT32),
                self.device(),
            )));
        }
        let flat = x.getitem(&ixs);
        Ok(flat.reshape(
            ishape
                .into_iter()
                .map(|s| s.clone().as_i64().unwrap_or(0))
                .collect::<Vec<_>>(),
        ))
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
            return Err(OnnxError::Unsupported(
                "Conv with per-dimension strides/dilations".into(),
            ));
        }
        Ok(x.conv2d(w, b, groups, s, d, &padding, None))
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
        if strides.iter().any(|&v| v != s)
            || dilations.iter().any(|&v| v != d)
            || output_padding.iter().any(|&v| v != op)
        {
            return Err(OnnxError::Unsupported(
                "ConvTranspose with per-dimension strides/dilations/output_padding".into(),
            ));
        }
        let groups = node.attr_i64("group", 1);
        Ok(x.conv_transpose2d(w, b, groups, s, d, &pads, op))
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
        pads[3] = ((size - 1) / 2, size / 2);
        let p: Vec<(Option<i64>, Option<i64>)> =
            pads.iter().map(|(a, b)| (Some(*a), Some(*b))).collect();
        let pooled = sq
            .pad_constant(&p, ConstVal::Float(0.0))
            .avg_pool2d(&[size, 1], Some(&[1, 1]), 1, &[], false, true)
            .reshape(shape.clone());
        let betac = x.const_like(ConstVal::Float(beta), None);
        let denom = pooled.mul(alpha).add(bias).pow(&betac);
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
        if node.attr_i64("antialias", 0) != 0 {
            return Err(OnnxError::Unsupported("Resize antialias".into()));
        }
        let mode = node.attr_str("mode", "nearest");
        let mode = match mode {
            "linear" | "bilinear" | "trilinear" => "linear",
            "nearest" => "nearest",
            "cubic" => {
                return Err(OnnxError::Unsupported("Resize cubic".into()));
            }
            other => return Err(OnnxError::Unsupported(format!("Resize mode `{other}`"))),
        };
        if node.attr_str("keep_aspect_ratio_policy", "stretch") != "stretch" {
            return Err(OnnxError::Unsupported(
                "Resize keep_aspect_ratio_policy != stretch".into(),
            ));
        }
        let coord = node.attr_str("coordinate_transformation_mode", "half_pixel");
        if mode == "nearest" {
            let nearest_mode = node.attr_str("nearest_mode", "round_prefer_floor");
            return self.resize_nearest(x, &axes, &targets, &in_shape, coord, nearest_mode);
        }
        let align_corners = match coord {
            "align_corners" => true,
            "half_pixel" => false,
            "pytorch_half_pixel" if !targets.contains(&1) => false,
            other => {
                return Err(OnnxError::Unsupported(format!(
                    "Resize linear coordinate_transformation_mode `{other}`"
                )));
            }
        };
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
        let resized = xt.interpolate(&targets, mode, align_corners);
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
