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
        let n = orig_len;
        let mut mask_shape = vec![n, n];
        mask_shape.extend(std::iter::repeat_n(1, ndim - dim - 1));
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
            shape.extend(std::iter::repeat_n(1, others.len() - i));
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
    /// Two-operand `einsum` (e.g. `"ij,jk->ik"`). Thin wrapper over
    /// [`Tensor::einsum_many`].
    pub fn einsum(&self, other: &Tensor, subscripts: &str) -> Tensor {
        Tensor::einsum_many(&[self, other], subscripts)
    }

    /// N-operand `einsum`, with ellipsis, implicit output and repeated labels.
    ///
    /// Examples: `"ij,jk,kl->il"`, `"...ij,...jk->...ik"`, `"ii->i"`.
    /// Shared named labels must have equal sizes; `...` dims broadcast
    /// right-aligned (numpy/ONNX rules). A label repeated within one operand
    /// is extracted as a diagonal before the contraction.
    pub fn einsum_many(operands: &[&Tensor], subscripts: &str) -> Tensor {
        assert!(!operands.is_empty(), "einsum: no operands");
        let eq = subscripts.replace(' ', "");
        let (inputs, out_spec) = match eq.split_once("->") {
            Some((a, b)) => (a, Some(b)),
            None => (eq.as_str(), None),
        };
        let specs: Vec<&str> = inputs.split(',').collect();
        assert_eq!(
            specs.len(),
            operands.len(),
            "einsum: operand count mismatch"
        );

        // Parse each spec into its explicit labels and whether it has `...`.
        let mut explicit: Vec<Vec<char>> = Vec::new();
        let mut has_ell: Vec<bool> = Vec::new();
        for spec in &specs {
            let mut labels = Vec::new();
            let mut ell = false;
            let bytes = spec.as_bytes();
            let mut i = 0;
            while i < bytes.len() {
                if bytes[i] == b'.' {
                    assert!(
                        bytes.get(i + 1) == Some(&b'.') && bytes.get(i + 2) == Some(&b'.'),
                        "einsum: bad ellipsis in {spec:?}"
                    );
                    assert!(!ell, "einsum: more than one ellipsis in {spec:?}");
                    ell = true;
                    i += 3;
                } else {
                    let c = bytes[i] as char;
                    assert!(c.is_ascii_alphabetic(), "einsum: bad label {c:?}");
                    labels.push(c);
                    i += 1;
                }
            }
            explicit.push(labels);
            has_ell.push(ell);
        }

        let ranks: Vec<usize> = operands.iter().map(|t| t.shape().as_slice().len()).collect();
        let ell_ranks: Vec<usize> = (0..operands.len())
            .map(|i| {
                if has_ell[i] {
                    assert!(
                        ranks[i] >= explicit[i].len(),
                        "einsum: operand {i} has more labels than dims"
                    );
                    ranks[i] - explicit[i].len()
                } else {
                    assert_eq!(
                        ranks[i],
                        explicit[i].len(),
                        "einsum: operand {i} rank/label mismatch"
                    );
                    0
                }
            })
            .collect();
        let ell_rank = *ell_ranks.iter().max().unwrap_or(&0);
        let ell_label = |j: usize| char::from_u32(0xE000 + j as u32).unwrap();
        let is_ell = |c: char| (0xE000..0xE000 + ell_rank as u32).contains(&(c as u32));

        // Full label list per operand (ellipsis expanded, right-aligned) + sizes.
        let mut full: Vec<Vec<char>> = Vec::new();
        let mut named: std::collections::HashMap<char, i64> = std::collections::HashMap::new();
        let mut ell_sizes: Vec<i64> = vec![1; ell_rank];
        for i in 0..operands.len() {
            let mut labels = Vec::new();
            if has_ell[i] {
                labels.extend((ell_rank - ell_ranks[i]..ell_rank).map(ell_label));
            }
            labels.extend(explicit[i].iter().copied());
            assert_eq!(labels.len(), ranks[i], "einsum: operand {i} rank mismatch");
            for (lab, dim) in labels.iter().zip(operands[i].shape().as_slice()) {
                let d = dim.clone().as_i64().unwrap_or(1);
                if is_ell(*lab) {
                    let j = (*lab as u32 - 0xE000) as usize;
                    if ell_sizes[j] == 1 {
                        ell_sizes[j] = d;
                    } else {
                        assert!(
                            d == 1 || d == ell_sizes[j],
                            "einsum: ellipsis dim {j} mismatch"
                        );
                    }
                } else if let Some(old) = named.get(lab) {
                    assert_eq!(*old, d, "einsum: size mismatch for label {lab:?}");
                } else {
                    named.insert(*lab, d);
                }
            }
            full.push(labels);
        }
        let size_of = |lab: char| -> i64 {
            if is_ell(lab) {
                ell_sizes[(lab as u32 - 0xE000) as usize]
            } else {
                named[&lab]
            }
        };

        // Implicit output: ellipsis dims, then labels that occur exactly once in
        // the input (original occurrence counts, like numpy).
        let mut counts: std::collections::HashMap<char, usize> = std::collections::HashMap::new();
        for labels in &full {
            for &lab in labels {
                *counts.entry(lab).or_insert(0) += 1;
            }
        }
        let out_labels: Vec<char> = match out_spec {
            Some(spec) => {
                let mut labels = Vec::new();
                let bytes = spec.as_bytes();
                let mut i = 0;
                while i < bytes.len() {
                    if bytes[i] == b'.' {
                        assert!(
                            bytes.get(i + 1) == Some(&b'.') && bytes.get(i + 2) == Some(&b'.'),
                            "einsum: bad output ellipsis"
                        );
                        labels.extend((0..ell_rank).map(ell_label));
                        i += 3;
                    } else {
                        let c = bytes[i] as char;
                        assert!(c.is_ascii_alphabetic(), "einsum: bad output label {c:?}");
                        labels.push(c);
                        i += 1;
                    }
                }
                labels
            }
            None => {
                let mut v: Vec<char> = (0..ell_rank).map(ell_label).collect();
                let mut named_once: Vec<char> = full
                    .iter()
                    .flatten()
                    .copied()
                    .filter(|c| !is_ell(*c) && counts.get(c) == Some(&1))
                    .collect::<std::collections::HashSet<_>>()
                    .into_iter()
                    .collect();
                named_once.sort_unstable();
                v.extend(named_once);
                v
            }
        };
        let mut out_seen = std::collections::HashSet::new();
        for &c in &out_labels {
            assert!(out_seen.insert(c), "einsum: repeated output label {c:?}");
            assert!(counts.contains_key(&c), "einsum: output label {c:?} not in input");
        }

        // Repeated labels within one operand become diagonals.
        let mut prepared_inputs: Vec<Tensor> = Vec::new();
        let mut prepared_labels: Vec<Vec<char>> = Vec::new();
        for (i, t) in operands.iter().enumerate() {
            let (t, labels) = dedup_diagonals(t, &full[i]);
            prepared_inputs.push(t);
            prepared_labels.push(labels);
        }

        // Union: output labels first, then every other label in first-seen order.
        let mut union = out_labels.clone();
        for labels in &prepared_labels {
            for &c in labels {
                if !union.contains(&c) {
                    union.push(c);
                }
            }
        }
        let union_shape: Vec<i64> = union.iter().map(|&c| size_of(c)).collect();
        let mut prod: Option<Tensor> = None;
        for (t, labels) in prepared_inputs.iter().zip(&prepared_labels) {
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
                shape[ui] = union_shape[ui];
            }
            out = out.reshape(shape).expand(union_shape.clone());
            prod = Some(match prod {
                Some(p) => p.mul(&out),
                None => out,
            });
        }
        let mut summed = prod.expect("einsum: no product");
        let red: Vec<usize> = (out_labels.len()..union.len()).collect();
        if !red.is_empty() {
            summed = summed.sum(Some(&red), false, None);
        }
        summed
    }
}

fn dedup_diagonals(t: &Tensor, labels: &[char]) -> (Tensor, Vec<char>) {
    let mut t = t.clone();
    let mut labels = labels.to_vec();
    loop {
        let mut counts = std::collections::HashMap::new();
        for &c in &labels {
            *counts.entry(c).or_insert(0usize) += 1;
        }
        let Some((&dup, _)) = counts.iter().find(|&(_, &n)| n > 1) else {
            break;
        };
        let pos: Vec<usize> = labels
            .iter()
            .enumerate()
            .filter(|(_, c)| **c == dup)
            .map(|(i, _)| i)
            .collect();
        let (p0, p1) = (pos[0], pos[1]);
        let shape = t.shape();
        let n = shape.as_slice()[p0].clone().as_i64().unwrap_or(0);
        let m = shape.as_slice()[p1].clone().as_i64().unwrap_or(0);
        assert_eq!(n, m, "einsum: diagonal size mismatch for {dup:?}");
        assert!(n > 0, "einsum: zero-length diagonal is unsupported");
        let eye = Tensor::eye(t.ctx(), n, Some(n), Some(t.dtype()));
        let mut mshape = vec![1i64; labels.len()];
        mshape[p0] = n;
        mshape[p1] = n;
        t = t.mul(eye.reshape(mshape)).sum(Some(&[p1]), false, None);
        labels.remove(p1);
    }
    (t, labels)
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

impl Tensor {
    /// `a <= b` (the port predates the `le` method; ONNX needs it).
    pub fn le(&self, other: impl Into<crate::tensor::TensorAddArg>) -> Tensor {
        let other = other.into();
        let mut ctx = self.ctx().borrow_mut();
        let ou = match other {
            crate::tensor::TensorAddArg::T(o) => {
                self.ctx().ensure_same(o.ctx());
                o.uop()
            }
            crate::tensor::TensorAddArg::C(c) => self.const_like_ctx(&mut ctx, c, None).uop(),
        };
        let u = crate::tensor::elementwise::le(&mut ctx, self.uop(), ou);
        Tensor::from_uop(&ctx, u)
    }

    /// `a & b` for bools (ONNX `And`).
    pub fn logical_and(&self, other: &Tensor) -> Tensor {
        self.ctx().ensure_same(other.ctx());
        let mut ctx = self.ctx().borrow_mut();
        let u = crate::tensor::elementwise::and(&mut ctx, self.uop(), other.uop());
        Tensor::from_uop(&ctx, u)
    }

    /// `a | b` for bools (ONNX `Or`).
    pub fn logical_or(&self, other: &Tensor) -> Tensor {
        self.ctx().ensure_same(other.ctx());
        let mut ctx = self.ctx().borrow_mut();
        let u = crate::tensor::elementwise::or(&mut ctx, self.uop(), other.uop());
        Tensor::from_uop(&ctx, u)
    }

    /// `a ^ b` for bools (ONNX `Xor`).
    pub fn logical_xor(&self, other: &Tensor) -> Tensor {
        self.ctx().ensure_same(other.ctx());
        let mut ctx = self.ctx().borrow_mut();
        let u = crate::tensor::elementwise::xor(&mut ctx, self.uop(), other.uop());
        Tensor::from_uop(&ctx, u)
    }

    /// `a ** b` (ONNX `Pow`); the output keeps the base dtype, as in ONNX.
    pub fn pow(&self, other: &Tensor) -> Tensor {
        self.ctx().ensure_same(other.ctx());
        let base_dt = self.dtype();
        let out = {
            let mut ctx = self.ctx().borrow_mut();
            let u = crate::tensor::elementwise::pow(&mut ctx, self.uop(), other.uop());
            Tensor::from_uop(&ctx, u)
        };
        if crate::dtype::is_int(base_dt) {
            out.round().cast(base_dt)
        } else {
            out.cast(base_dt)
        }
    }
}
