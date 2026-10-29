//! Long-tail `Tensor` operations that are compositions over the existing UOp
//! primitives (the tinygrad `elementwise.py`/`op.py` helpers not needed by the
//! core milestones). Keeping them here avoids bloating `tensor/mod.rs`; they
//! use only the public `Tensor` API.

use crate::dtype::{ConstVal, DTypeId};
use crate::tensor::{Shape, Tensor};
use crate::uop::ctx::UOpCtx;

impl Tensor {
    /// `silu()` / `swish()`: `x * sigmoid(x)`.
    pub fn silu(&self) -> Tensor {
        self.mul(crate::tensor::nn::sigmoid(self))
    }

    /// `swish()` alias.
    pub fn swish(&self) -> Tensor {
        self.silu()
    }

    /// `quick_gelu()`: `x * sigmoid(1.702*x)`.
    pub fn quick_gelu(&self) -> Tensor {
        self.mul(crate::tensor::nn::sigmoid(&self.mul(1.702)))
    }

    /// `leaky_relu(neg_slope=0.01)`.
    pub fn leaky_relu(&self, neg_slope: f64) -> Tensor {
        self.lt(0).where_(self.mul(neg_slope), self)
    }

    /// `relu6()`: `relu(x) - relu(x - 6)`.
    pub fn relu6(&self) -> Tensor {
        crate::tensor::nn::relu(self).sub(self.sub(6.0).relu())
    }

    /// `hardswish()`: `x * relu6(x + 3) / 6`.
    pub fn hardswish(&self) -> Tensor {
        self.mul(self.add(3.0).relu6()).mul(1.0 / 6.0)
    }

    /// `hardsigmoid(alpha=1/6, beta=0.5)`.
    pub fn hardsigmoid(&self) -> Tensor {
        let a = self.mul(1.0 / 6.0).add(0.5);
        a.relu().sub(a.sub(1.0).relu())
    }

    /// `hardtanh(min_val=-1, max_val=1)`.
    pub fn hardtanh(&self, min_val: f64, max_val: f64) -> Tensor {
        self.clip(Some(min_val), Some(max_val))
    }

    /// `elu(alpha=1.0)`: `relu(x) - alpha*relu(1 - exp(x))`.
    pub fn elu(&self, alpha: f64) -> Tensor {
        crate::tensor::nn::relu(self).sub(
            self.const_like(ConstVal::Float(1.0), None)
                .sub(self.exp())
                .relu()
                .mul(alpha),
        )
    }

    /// `celu(alpha=1.0)`.
    pub fn celu(&self, alpha: f64) -> Tensor {
        let neg = self.div(alpha).exp().sub(1.0).mul(alpha).minimum(0.0);
        self.maximum(0.0).add(neg)
    }

    /// `selu(alpha=1.67326, gamma=1.0507)`.
    pub fn selu(&self, alpha: f64, gamma: f64) -> Tensor {
        self.ge(0)
            .where_(self, self.exp().sub(1.0).mul(alpha))
            .mul(gamma)
    }

    /// `softplus(beta=1.0)`: `logaddexp(beta*x, 0) / beta`.
    pub fn softplus(&self, beta: f64) -> Tensor {
        self.mul(beta)
            .logaddexp(&self.const_like(ConstVal::Float(0.0), None))
            .div(beta)
    }

    /// `softsign()`: `x / (1 + |x|)`.
    pub fn softsign(&self) -> Tensor {
        self.div(self.abs().add(1.0))
    }

    /// `mish()`: `x * tanh(softplus(x))`.
    pub fn mish(&self) -> Tensor {
        self.mul(self.softplus(1.0).tanh())
    }

    /// `logsigmoid()`: `-softplus(-x)`.
    pub fn logsigmoid(&self) -> Tensor {
        self.neg().softplus(1.0).neg()
    }

    /// `sign()`: `-1 / 0 / 1`.
    pub fn sign(&self) -> Tensor {
        let inner = self.lt(0).where_(-1.0, 1.0);
        self.ne(0)
            .where_(inner, self.const_like(ConstVal::Int(0), None))
    }

    /// `round()`: round half to even (the tinygrad expression).
    pub fn round(&self) -> Tensor {
        let b = self.trunc().div(2.0);
        let bb = b.trunc().eq(&b);
        let cond = self.gt(0).eq(bb);
        cond.where_(self.sub(0.5).ceil(), self.add(0.5).floor())
    }

    /// `tan()`: `sin(x) / cos(x)`.
    pub fn tan(&self) -> Tensor {
        self.sin().div(crate::tensor::nn::cos(self))
    }

    /// `log10()`: `log2(x) * log10(2)`.
    pub fn log10(&self) -> Tensor {
        self.log2().mul(std::f64::consts::LOG10_2)
    }

    /// `logaddexp(other)`: `log(exp(a) + exp(b))`, numerically shifted.
    pub fn logaddexp(&self, other: &Tensor) -> Tensor {
        let m = self.maximum(other);
        self.sub(&m).exp().add(other.sub(&m).exp()).log().add(&m)
    }

    /// `copysign(other)`: magnitude of `self`, sign of `other`.
    pub fn copysign(&self, other: &Tensor) -> Tensor {
        let mag = self.abs();
        other.lt(0).where_(mag.neg(), mag)
    }

    /// `min(axis, keepdim)`: `-max(-x)`.
    pub fn min(&self, axis: Option<&[usize]>, keepdim: bool) -> Tensor {
        self.neg().max(axis, keepdim).neg()
    }

    /// `argmin(axis, keepdim)`: `argmax(-x)`.
    pub fn argmin(&self, axis: Option<i64>, keepdim: bool) -> Tensor {
        self.neg().argmax(axis, keepdim)
    }

    /// `std(axis, keepdim, correction)`: `sqrt(var(...))`.
    pub fn std(&self, axis: Option<&[usize]>, keepdim: bool, correction: i64) -> Tensor {
        self.var(axis, keepdim, correction).sqrt()
    }

    /// `std_mean(axis, keepdim, correction)`.
    pub fn std_mean(
        &self,
        axis: Option<&[usize]>,
        keepdim: bool,
        correction: i64,
    ) -> (Tensor, Tensor) {
        (
            self.std(axis, keepdim, correction),
            self.mean(axis, keepdim),
        )
    }

    /// `normalize(p=2, dim=1, eps=1e-12)`.
    pub fn normalize(&self, p: f64, dim: i64, eps: f64) -> Tensor {
        let axes = [resolve_dim(self, dim)];
        let den = if p == 0.0 {
            self.ne(0).sum(Some(&axes), true, None)
        } else {
            self.abs()
                .pow_const(p)
                .sum(Some(&axes), true, None)
                .pow_const(1.0 / p)
        };
        self.div(den.maximum(eps))
    }

    /// `logsumexp(axis, keepdim)` (max-shifted).
    pub fn logsumexp(&self, axis: Option<&[usize]>, keepdim: bool) -> Tensor {
        let m = self.max(axis, true).detach();
        let sum = self.sub(&m).exp().sum(axis, keepdim, None).log();
        let m2 = if keepdim { m } else { squeeze_axes(&m, axis) };
        sum.add(&m2)
    }

    /// `cumprod(axis)`.
    pub fn cumprod(&self, axis: i64) -> Tensor {
        let ax = resolve_dim(self, axis);
        let mut ctx: std::cell::RefMut<'_, UOpCtx> = self.ctx().borrow_mut();
        let u = crate::tensor::reduce::cumalu(&mut ctx, self.uop(), ax, crate::uop::ops::Ops::Mul);
        Tensor::from_uop(&ctx, u)
    }

    /// `isnan()`: `x != x`.
    pub fn isnan(&self) -> Tensor {
        self.ne(self)
    }

    /// `isinf()` (either sign).
    pub fn isinf(&self) -> Tensor {
        self.abs().eq(f64::INFINITY)
    }

    /// `isfinite()`.
    pub fn isfinite(&self) -> Tensor {
        self.isnan()
            .where_(ConstVal::Bool(false), self.isinf().logical_not())
    }

    /// `isclose(other, rtol=1e-5, atol=1e-8)`.
    pub fn isclose(&self, other: &Tensor, rtol: f64, atol: f64) -> Tensor {
        let tol = other.abs().mul(rtol).add(atol);
        self.sub(other).abs().gt(tol).logical_not()
    }

    /// `allclose(other, rtol=1e-5, atol=1e-8)`.
    pub fn allclose(&self, other: &Tensor, rtol: f64, atol: f64) -> bool {
        self.isclose(other, rtol, atol).all().item_fresh::<bool>()
    }

    /// `numel()`.
    pub fn numel(&self) -> i64 {
        self.shape().numel_int().unwrap_or(0) as i64
    }

    /// `size(dim)` (negative dims supported, `None` for the full shape? here int).
    pub fn size(&self, dim: i64) -> i64 {
        let shape = self.shape();
        let s = shape.as_slice();
        let d = if dim < 0 { s.len() as i64 + dim } else { dim };
        s[d as usize].clone().as_i64().unwrap_or(0)
    }

    /// `nbytes()`.
    pub fn nbytes(&self) -> usize {
        self.numel() as usize * self.dtype().itemsize()
    }

    /// `zeros_like()`.
    pub fn zeros_like(&self) -> Tensor {
        Tensor::zeros(self.ctx(), self.shape())
    }

    /// `ones_like()`.
    pub fn ones_like(&self) -> Tensor {
        Tensor::ones(self.ctx(), self.shape())
    }

    /// `full_like(fill_value, dtype)`.
    pub fn full_like(&self, fill: ConstVal, dtype: Option<DTypeId>) -> Tensor {
        Tensor::full(
            self.ctx(),
            self.shape(),
            fill,
            dtype.or(Some(self.dtype())),
            self.device().as_deref(),
        )
    }

    /// `Tensor.erf()` (Abramowitz–Stegun 7.1.26, matching tinygrad).
    pub fn erf(&self) -> Tensor {
        let t = self.abs().mul(0.3275911).add(1.0).reciprocal();
        let poly = t
            .mul(1.061405429)
            .sub(1.453152027)
            .mul(&t)
            .add(1.421413741)
            .mul(&t)
            .sub(0.284496736)
            .mul(&t)
            .add(0.254829592);
        let e = self.mul(self).neg().exp();
        let one = self.const_like(ConstVal::Float(1.0), None);
        self.sign().mul(one.sub(t.mul(poly).mul(e)))
    }

    ///  method form (the  helper is context-free).
    pub fn sigmoid(&self) -> Tensor {
        crate::tensor::nn::sigmoid(self)
    }

    ///  method form.
    pub fn relu(&self) -> Tensor {
        crate::tensor::nn::relu(self)
    }

    ///  method form.
    pub fn cos(&self) -> Tensor {
        crate::tensor::nn::cos(self)
    }

    /// .
    pub fn logical_not(&self) -> Tensor {
        let mut ctx = self.ctx().borrow_mut();
        let u = crate::tensor::elementwise::logical_not(&mut ctx, self.uop());
        Tensor::from_uop(&ctx, u)
    }

    ///  (reduce over every axis).
    pub fn all(&self) -> Tensor {
        let mut ctx = self.ctx().borrow_mut();
        let u = crate::tensor::reduce::all(&mut ctx, self.uop(), None, false);
        Tensor::from_uop(&ctx, u)
    }
}

fn resolve_dim(t: &Tensor, dim: i64) -> usize {
    let n = t.shape().as_slice().len() as i64;
    let d = if dim < 0 { n + dim } else { dim };
    d as usize
}

fn squeeze_axes(t: &Tensor, axes: Option<&[usize]>) -> Tensor {
    let mut out = t.clone();
    let n = t.shape().as_slice().len();
    let mut ax: Vec<usize> = axes.map(|a| a.to_vec()).unwrap_or_else(|| (0..n).collect());
    ax.sort_unstable();
    for &a in ax.iter().rev() {
        out = out.squeeze(Some(a as i64));
    }
    out
}

trait PowConst {
    fn pow_const(&self, p: f64) -> Tensor;
}
impl PowConst for Tensor {
    fn pow_const(&self, p: f64) -> Tensor {
        let mut ctx = self.ctx().borrow_mut();
        let u = crate::tensor::elementwise::pow_c(&mut ctx, self.uop(), ConstVal::Float(p));
        Tensor::from_uop(&ctx, u)
    }
}

#[allow(dead_code)]
fn _shape_len(s: &Shape) -> usize {
    s.as_slice().len()
}

impl Tensor {
    /// `sort(dim=-1, descending=False)` (bitonic, matching tinygrad).
    pub fn sort(&self, dim: i64, descending: bool) -> (Tensor, Tensor) {
        let ndim = self.shape().as_slice().len();
        let dim = resolve_dim(self, dim);
        let orig_len = self.shape().as_slice()[dim].clone().as_i64().unwrap_or(0);
        if orig_len <= 1 {
            return (
                self.clone(),
                self.const_like(ConstVal::Int(0), Some(crate::dtype::INT32)),
            );
        }
        let n_stages = (orig_len - 1).ilog2() as usize + 1;
        let mut pads = vec![(None, None); ndim];
        pads[dim] = (Some(0), Some((1i64 << n_stages) - orig_len));
        let fill = if descending {
            ConstVal::Float(f64::NEG_INFINITY)
        } else {
            ConstVal::Float(f64::INFINITY)
        };
        let mut x = self
            .pad_constant(&pads, fill)
            .unflatten(dim as i64, &vec![2i64; n_stages]);
        for stage in 1..=n_stages {
            let crossover_dim = if stage != n_stages {
                Some(dim + n_stages - stage - 1)
            } else {
                None
            };
            let flip_dims: Vec<usize> = if stage != n_stages {
                let xn = x.shape().as_slice().len();
                (1..=stage + (ndim - dim)).map(|i| xn - i).collect()
            } else {
                Vec::new()
            };
            if let Some(cd) = crossover_dim {
                let parts = x.split(&[1, 1], cd as i64);
                x = parts[0].cat(&[parts[1].flip(&flip_dims)], cd as i64);
            }
            for substage in (0..stage).rev() {
                let partner_dim = dim + n_stages - substage - 1;
                let parts = x.split(&[1, 1], partner_dim as i64);
                let (top, bottom) = (&parts[0], &parts[1]);
                let larger = top.maximum(bottom);
                let smaller = top.minimum(bottom);
                x = if descending {
                    larger.cat(&[smaller], partner_dim as i64)
                } else {
                    smaller.cat(&[larger], partner_dim as i64)
                };
            }
            if let Some(cd) = crossover_dim {
                let parts = x.split(&[1, 1], cd as i64);
                x = parts[0].cat(&[parts[1].flip(&flip_dims)], cd as i64);
            }
        }
        let shape = self.shape();
        let x = x
            .flatten(dim as i64, (dim + n_stages - 1) as i64)
            .shrink_to(shape.as_slice());

        // Stable indices: count equal values before each position.
        let n = orig_len as i64;
        let mut mask_shape = vec![n, n];
        mask_shape.extend(std::iter::repeat(1).take(ndim - dim - 1));
        let mask = Tensor::full_const(
            self.ctx(),
            [n, n],
            ConstVal::Bool(true),
            Some(crate::dtype::BOOL),
        )
        .tril(0)
        .reshape(mask_shape);
        let counts = |t: &Tensor| {
            let eq = t.unsqueeze((dim + 1) as i64).eq(t.unsqueeze(dim as i64));
            mask.mul(eq).sum(Some(&[dim + 1]), false, None)
        };
        let count_orig = counts(self);
        let count_sorted = counts(&x);
        let cond = self
            .unsqueeze((dim + 1) as i64)
            .eq(x.unsqueeze(dim as i64))
            .mul(
                count_orig
                    .unsqueeze((dim + 1) as i64)
                    .eq(count_sorted.unsqueeze(dim as i64)),
            );
        let mut idx_shape = vec![1i64; ndim];
        idx_shape[dim] = n;
        let idx = Tensor::arange(
            self.ctx(),
            0.0,
            Some(n as f64),
            1.0,
            Some(crate::dtype::INT32),
        )
        .reshape(idx_shape)
        .unsqueeze((dim + 1) as i64);
        let idx = cond.mul(idx).sum(Some(&[dim]), false, None);
        (x, idx)
    }

    /// `argsort(dim=-1, descending=False)`.
    pub fn argsort(&self, dim: i64, descending: bool) -> Tensor {
        self.sort(dim, descending).1
    }

    /// `topk(k, dim=-1, largest=True)` (sorted_=False is unsupported upstream).
    pub fn topk(&self, k: i64, dim: i64, largest: bool) -> (Tensor, Tensor) {
        let d = resolve_dim(self, dim);
        let (v, i) = self.sort(dim, largest);
        let mut shape = self.shape().as_slice().to_vec();
        shape[d] = crate::uop::sint::SVal::I(k as i128);
        (v.shrink_to(&shape), i.shrink_to(&shape))
    }
}

impl Tensor {
    /// `repeat(*repeats)` (movement.py).
    pub fn repeat(&self, repeats: &[i64]) -> Tensor {
        let mut ctx = self.ctx().borrow_mut();
        let u = crate::tensor::reduce::repeat(&mut ctx, self.uop(), repeats);
        Tensor::from_uop(&ctx, u)
    }

    /// `roll(shifts, dims)` (movement.py:540-556).
    pub fn roll(&self, shifts: i64, dims: Option<i64>) -> Tensor {
        let Some(dim) = dims else {
            return self
                .flatten(0, -1)
                .roll(shifts, Some(0))
                .reshape(self.shape());
        };
        let d = resolve_dim(self, dim);
        let size = self.size(d as i64);
        if size == 0 {
            return self.clone();
        }
        let delta = (size - shifts.rem_euclid(size)) % size;
        let ndim = self.shape().as_slice().len();
        let mut reps = vec![1i64; ndim];
        reps[d] = 2;
        let mut shrink = vec![(0i64, size); ndim];
        shrink[d] = (delta, delta + size);
        self.repeat(&reps).shrink(&shrink)
    }

    /// `masked_fill(mask, value)`: `mask ? value : self`.
    pub fn masked_fill(
        &self,
        mask: &Tensor,
        value: impl Into<crate::tensor::TensorAddArg>,
    ) -> Tensor {
        mask.where_(value, self)
    }

    /// `diag()`: a 1-D tensor as the main diagonal of a square matrix.
    pub fn diag(&self) -> Tensor {
        assert_eq!(
            self.shape().as_slice().len(),
            1,
            "TGERR: diag expects a 1-D tensor"
        );
        let n = self.size(0);
        self.unsqueeze(-1)
            .pad(&[(0, 0), (0, n)])
            .flatten(0, -1)
            .shrink(&[(0, n * n)])
            .reshape([n, n])
    }

    /// `meshgrid(*others, indexing="ij"|"xy")`.
    pub fn meshgrid(&self, others: &[Tensor], indexing: &str) -> Vec<Tensor> {
        assert!(
            indexing == "ij" || indexing == "xy",
            "TGERR: indexing must be ij or xy"
        );
        let mut all = Vec::with_capacity(others.len() + 1);
        all.push(self.clone());
        all.extend(others.iter().cloned());
        if all.len() == 1 {
            return all;
        }
        let n = all.len();
        let basis: Vec<usize> = if indexing == "ij" {
            (0..n).collect()
        } else {
            let mut b = vec![1, 0];
            b.extend(2..n);
            b
        };
        let mut shaped = Vec::with_capacity(n);
        for (t, &i) in all.iter().zip(basis.iter()) {
            let mut shape = vec![-1i64];
            shape.extend(std::iter::repeat(1).take(others.len() - i));
            shaped.push(t.reshape(shape));
        }
        let mut out_shape = vec![1i64; n];
        for t in &shaped {
            let s = t.shape();
            let sl = s.as_slice();
            let off = n - sl.len();
            for (k, v) in sl.iter().enumerate() {
                let v = v.clone().as_i64().unwrap_or(1);
                if v != 1 {
                    out_shape[off + k] = v;
                }
            }
        }
        shaped.iter().map(|t| t.expand(out_shape.clone())).collect()
    }
}

impl Tensor {
    /// Two-operand `einsum` (e.g. `"ij,jk->ik"`). Covers the common attention /
    /// ONNX cases; ellipsis and repeated labels within one operand are rejected.
    pub fn einsum(&self, other: &Tensor, subscripts: &str) -> Tensor {
        let (lhs, rhs, out) = parse_einsum(subscripts);
        let a_labels: Vec<char> = lhs.chars().collect();
        let b_labels: Vec<char> = rhs.chars().collect();
        assert_eq!(
            a_labels.len(),
            self.shape().as_slice().len(),
            "einsum: lhs rank mismatch"
        );
        assert_eq!(
            b_labels.len(),
            other.shape().as_slice().len(),
            "einsum: rhs rank mismatch"
        );
        assert_eq!(
            a_labels
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            a_labels.len(),
            "einsum: repeated lhs labels are unsupported"
        );
        assert_eq!(
            b_labels
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            b_labels.len(),
            "einsum: repeated rhs labels are unsupported"
        );

        let mut sizes = std::collections::HashMap::new();
        for (lab, sz) in a_labels.iter().zip(self.shape().as_slice()) {
            sizes.insert(*lab, sz.clone().as_i64().unwrap_or(1));
        }
        for (lab, sz) in b_labels.iter().zip(other.shape().as_slice()) {
            let v = sz.clone().as_i64().unwrap_or(1);
            if let Some(old) = sizes.get(lab) {
                assert_eq!(*old, v, "einsum: size mismatch for label {lab}");
            }
            sizes.insert(*lab, v);
        }
        let out_labels: Vec<char> = match out {
            Some(s) => s.chars().collect(),
            None => {
                let mut counts = std::collections::HashMap::new();
                for c in a_labels.iter().chain(b_labels.iter()) {
                    *counts.entry(*c).or_insert(0) += 1;
                }
                let mut v: Vec<char> = counts
                    .iter()
                    .filter(|(_, n)| **n == 1)
                    .map(|(c, _)| *c)
                    .collect();
                v.sort_unstable();
                v
            }
        };
        let mut union: Vec<char> = out_labels.clone();
        for c in a_labels.iter().chain(b_labels.iter()) {
            if !union.contains(c) {
                union.push(*c);
            }
        }
        let union_shape: Vec<i64> = union.iter().map(|c| sizes[c]).collect();

        let prepare = |t: &Tensor, labels: &[char]| -> Tensor {
            let present: Vec<usize> = (0..union.len())
                .filter(|&ui| labels.contains(&union[ui]))
                .collect();
            let axes: Vec<i64> = present
                .iter()
                .map(|&ui| labels.iter().position(|x| *x == union[ui]).unwrap() as i64)
                .collect();
            let mut out = if axes.len() == labels.len() {
                t.permute(&axes)
            } else {
                t.clone()
            };
            let mut shape = vec![1i64; union.len()];
            for &ui in &present {
                shape[ui] = sizes[&union[ui]];
            }
            out = out.reshape(shape);
            out.expand(union_shape.clone())
        };
        let prod = prepare(self, &a_labels).mul(prepare(other, &b_labels));
        let red: Vec<usize> = union
            .iter()
            .enumerate()
            .filter(|(_, c)| !out_labels.contains(c))
            .map(|(i, _)| i)
            .collect();
        let summed = if red.is_empty() {
            prod
        } else {
            prod.sum(Some(&red), false, None)
        };
        let rem: Vec<char> = union
            .iter()
            .filter(|c| out_labels.contains(c))
            .cloned()
            .collect();
        let perm: Vec<i64> = out_labels
            .iter()
            .map(|c| rem.iter().position(|x| x == c).unwrap() as i64)
            .collect();
        if perm.is_empty() {
            summed
        } else {
            summed.permute(&perm)
        }
    }
}

fn parse_einsum(s: &str) -> (String, String, Option<String>) {
    let s = s.replace(' ', "");
    let (inputs, out) = match s.split_once("->") {
        Some((a, b)) => (a, Some(b.to_string())),
        None => (s.as_str(), None),
    };
    let mut parts = inputs.split(',');
    let lhs = parts.next().unwrap_or("").to_string();
    let rhs = parts.next().unwrap_or("").to_string();
    assert!(
        parts.next().is_none(),
        "einsum: expected exactly two operands"
    );
    (lhs, rhs, out)
}

impl Tensor {
    /// `asin()` (Abramowitz–Stegun 4.4.46, matching tinygrad).
    pub fn asin(&self) -> Tensor {
        const C: [f64; 8] = [
            -0.0012624911,
            0.0066700901,
            -0.0170881256,
            0.0308918810,
            -0.0501743046,
            0.0889789874,
            -0.2145988016,
            1.5707963050,
        ];
        let abs = self.abs();
        let poly = C
            .iter()
            .fold(abs.const_like(ConstVal::Float(0.0), None), |acc, c| {
                acc.mul(&abs).add(*c)
            });
        let x = abs
            .const_like(ConstVal::Float(std::f64::consts::FRAC_PI_2), None)
            .sub(
                abs.const_like(ConstVal::Float(1.0), None)
                    .sub(&abs)
                    .sqrt()
                    .mul(poly),
            );
        self.sign().mul(x)
    }

    /// `acos()`: `pi/2 - asin(x)`.
    pub fn acos(&self) -> Tensor {
        self.const_like(ConstVal::Float(std::f64::consts::FRAC_PI_2), None)
            .sub(self.asin())
    }

    /// `atan()`: `asin(x / sqrt(1 + x^2))`.
    pub fn atan(&self) -> Tensor {
        self.div(self.mul(self).add(1.0).sqrt()).asin()
    }

    /// `asinh()`: `log(x + sqrt(x^2 + 1))`.
    pub fn asinh(&self) -> Tensor {
        self.add(self.mul(self).add(1.0).sqrt()).log()
    }

    /// `acosh()`: `log(x + sqrt(x^2 - 1))`.
    pub fn acosh(&self) -> Tensor {
        self.add(self.mul(self).sub(1.0).sqrt()).log()
    }

    /// `atanh()`: `log((1 + x) / (1 - x)) / 2`.
    pub fn atanh(&self) -> Tensor {
        self.add(1.0).div(self.neg().add(1.0)).log().div(2.0)
    }

    /// `sinh()`: `(exp(x) - exp(-x)) / 2`.
    pub fn sinh(&self) -> Tensor {
        self.exp().sub(self.neg().exp()).div(2.0)
    }

    /// `cosh()`: `(exp(x) + exp(-x)) / 2`.
    pub fn cosh(&self) -> Tensor {
        self.exp().add(self.neg().exp()).div(2.0)
    }
}

impl Tensor {
    fn cum_max(&self, axis: i64) -> Tensor {
        let ax = resolve_dim(self, axis);
        let mut ctx = self.ctx().borrow_mut();
        let u = crate::tensor::reduce::cumalu(&mut ctx, self.uop(), ax, crate::uop::ops::Ops::Max);
        Tensor::from_uop(&ctx, u)
    }

    /// `cummax(axis=0)` -> `(values, indices)` (movement.py/op.py:801-818).
    pub fn cummax(&self, axis: i64) -> (Tensor, Tensor) {
        let ndim = self.shape().as_slice().len();
        let ax = resolve_dim(self, axis);
        let values = self.cum_max(axis);
        if ndim == 0 {
            let z = self.zeros_like().cast(crate::dtype::INT32);
            return (values, z);
        }
        let n = self.size(ax as i64);
        let x = self.transpose(axis, -1);
        let vt = values.transpose(axis, -1);
        let mask = Tensor::full_const(
            self.ctx(),
            [n, n],
            ConstVal::Bool(true),
            Some(crate::dtype::BOOL),
        )
        .triu(0);
        let mask_i = mask.cast(crate::dtype::INT32);
        let match_i = x
            .unsqueeze(-1)
            .eq(vt.unsqueeze(-2))
            .cast(crate::dtype::INT32)
            .mul(mask_i);
        let weights = Tensor::arange(
            self.ctx(),
            n as f64,
            Some(0.0),
            -1.0,
            Some(crate::dtype::INT32),
        )
        .reshape([n, 1]);
        let red = ndim - 1;
        let idx = match_i
            .mul(weights)
            .max(Some(&[red]), false)
            .neg()
            .add(n)
            .cast(crate::dtype::INT32);
        (values, idx.transpose(-1, axis))
    }

    /// `cummin(axis=0)` -> `(values, indices)`.
    pub fn cummin(&self, axis: i64) -> (Tensor, Tensor) {
        let (v, i) = self.neg().cummax(axis);
        (v.neg(), i)
    }

    /// `logcumsumexp(axis=0)` (op.py:833-864).
    pub fn logcumsumexp(&self, axis: i64) -> Tensor {
        let ndim = self.shape().as_slice().len();
        if ndim == 0 {
            return self.clone();
        }
        let x = self.transpose(axis, -1);
        let n = x.size(-1);
        let cm = x.cum_max(-1).detach();
        let mask = Tensor::full_const(
            self.ctx(),
            [n, n],
            ConstVal::Bool(true),
            Some(crate::dtype::BOOL),
        )
        .tril(0);
        let ninf = self.const_like(ConstVal::Float(f64::NEG_INFINITY), None);
        let ret = mask
            .where_(x.unsqueeze(-2).sub(cm.unsqueeze(-1)), ninf)
            .exp()
            .sum(Some(&[ndim]), false, None)
            .log()
            .add(&cm);
        ret.transpose(-1, axis)
    }
}
