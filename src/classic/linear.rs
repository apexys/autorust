//! Linear and logistic regression.

use super::linalg::least_squares;
use super::{
    dtype_from_debug, float_dtype, meta_f64, meta_i64, meta_str_optional, put_meta, read_labels,
    read_matrix, read_target, require_fitted, state_f64, tensor_from_f64, ClassicError, Persist,
    Result,
};
use crate::dtype::DTypeId;
use crate::tensor::state::State;
use crate::tensor::{Ctx, Tensor};

/// Ordinary least-squares linear regression, with optional ridge regularization.
///
/// `alpha = 0` is ordinary least squares; `alpha > 0` is ridge regression.
/// A fitted single-output model predicts `[n]`; a multi-output model predicts
/// `[n, n_targets]`.
pub struct LinearRegression {
    ctx: Ctx,
    coef: Option<Tensor>,
    intercept: Option<Tensor>,
    alpha: f64,
    fit_intercept: bool,
    n_features: usize,
    n_targets: usize,
    single_output: bool,
    dtype: DTypeId,
}

impl LinearRegression {
    /// Create an unfitted ordinary least-squares model.
    pub fn new(ctx: &Ctx) -> Self {
        LinearRegression {
            ctx: ctx.clone(),
            coef: None,
            intercept: None,
            alpha: 0.0,
            fit_intercept: true,
            n_features: 0,
            n_targets: 0,
            single_output: true,
            dtype: crate::dtype::FLOAT32,
        }
    }

    /// Create a ridge regression model (`alpha > 0`).
    pub fn ridge(ctx: &Ctx, alpha: f64) -> Self {
        let mut m = Self::new(ctx);
        m.alpha = alpha;
        m
    }

    /// Set the L2 regularization strength (`0` = OLS).
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha.max(0.0);
        self
    }

    /// Enable/disable the intercept term.
    pub fn with_intercept(mut self, yes: bool) -> Self {
        self.fit_intercept = yes;
        self
    }

    /// Fit the model.
    pub fn fit(&mut self, x: &Tensor, y: &Tensor) -> Result<()> {
        let (xv, n, d) = read_matrix(x, "LinearRegression.x")?;
        let (yv, m, single) = read_target(y, n, "LinearRegression.y")?;
        if self.alpha < 0.0 {
            return Err(ClassicError::Invalid("alpha must be non-negative".into()));
        }
        let dtype = float_dtype(x)?;
        let device = x.device();
        let mut xm = vec![0.0; d];
        let mut ym = vec![0.0; m];
        if self.fit_intercept {
            for i in 0..n {
                for j in 0..d {
                    xm[j] += xv[i * d + j];
                }
                for k in 0..m {
                    ym[k] += yv[i * m + k];
                }
            }
            for v in &mut xm {
                *v /= n as f64;
            }
            for v in &mut ym {
                *v /= n as f64;
            }
        }
        let mut xc = xv;
        let mut yc = yv;
        if self.fit_intercept {
            for i in 0..n {
                for j in 0..d {
                    xc[i * d + j] -= xm[j];
                }
                for k in 0..m {
                    yc[i * m + k] -= ym[k];
                }
            }
        }
        let w = least_squares(&xc, &yc, n, d, m, self.alpha)?;
        let mut b = vec![0.0; m];
        if self.fit_intercept {
            for k in 0..m {
                let mut acc = ym[k];
                for j in 0..d {
                    acc -= xm[j] * w[j * m + k];
                }
                b[k] = acc;
            }
        }
        self.coef = Some(tensor_from_f64(
            &self.ctx,
            &w,
            &[d as i64, m as i64],
            dtype,
            device.as_deref(),
        ));
        self.intercept = Some(tensor_from_f64(
            &self.ctx,
            &b,
            &[m as i64],
            dtype,
            device.as_deref(),
        ));
        self.n_features = d;
        self.n_targets = m;
        self.single_output = single;
        self.dtype = dtype;
        Ok(())
    }

    /// Predict targets for `x` (`[n, n_features]`).
    pub fn predict(&self, x: &Tensor) -> Result<Tensor> {
        let coef = require_fitted(&self.coef, "LinearRegression")?;
        let intercept = require_fitted(&self.intercept, "LinearRegression")?;
        let sh = super::shape_i64(x);
        if sh.len() != 2 || sh[1] as usize != self.n_features {
            return Err(ClassicError::Shape(format!(
                "LinearRegression expects [n, {}], got {sh:?}",
                self.n_features
            )));
        }
        let out = x.matmul(coef).add(intercept);
        Ok(if self.single_output {
            out.squeeze(Some(-1))
        } else {
            out
        })
    }

    /// Fitted coefficients, shape `[n_features, n_targets]`.
    pub fn coef(&self) -> Option<&Tensor> {
        self.coef.as_ref()
    }

    /// Fitted intercept, shape `[n_targets]`.
    pub fn intercept(&self) -> Option<&Tensor> {
        self.intercept.as_ref()
    }
}

impl Persist for LinearRegression {
    fn state(&self) -> Result<State> {
        let coef = require_fitted(&self.coef, "LinearRegression")?;
        let intercept = require_fitted(&self.intercept, "LinearRegression")?;
        let mut s = State::new();
        s.insert("coef", super::detach_tensor(coef)?);
        s.insert("intercept", super::detach_tensor(intercept)?);
        put_meta(&mut s, "classic.kind", "linear_regression");
        put_meta(&mut s, "classic.version", 1);
        put_meta(&mut s, "classic.alpha", self.alpha);
        put_meta(&mut s, "classic.fit_intercept", self.fit_intercept as u8);
        put_meta(&mut s, "classic.n_features", self.n_features);
        put_meta(&mut s, "classic.n_targets", self.n_targets);
        put_meta(&mut s, "classic.single_output", self.single_output as u8);
        put_meta(&mut s, "classic.dtype", super::dtype_label(self.dtype));
        Ok(s)
    }

    fn load_state(&mut self, state: &State) -> Result<()> {
        if meta_str_optional(state, "classic.kind") != Some("linear_regression") {
            return Err(ClassicError::State(
                "state is not a LinearRegression".into(),
            ));
        }
        let d = meta_i64(state, "classic.n_features")? as usize;
        let m = meta_i64(state, "classic.n_targets")? as usize;
        let dtype =
            dtype_from_debug(meta_str_optional(state, "classic.dtype").unwrap_or("FLOAT32"))?;
        let w = state_f64(state, "coef")?;
        let b = state_f64(state, "intercept")?;
        if w.len() != d * m || b.len() != m {
            return Err(ClassicError::State(
                "LinearRegression state has inconsistent dimensions".into(),
            ));
        }
        self.coef = Some(tensor_from_f64(
            &self.ctx,
            &w,
            &[d as i64, m as i64],
            dtype,
            None,
        ));
        self.intercept = Some(tensor_from_f64(&self.ctx, &b, &[m as i64], dtype, None));
        self.alpha = meta_f64(state, "classic.alpha").unwrap_or(0.0);
        self.fit_intercept = meta_i64(state, "classic.fit_intercept").unwrap_or(1) != 0;
        self.n_features = d;
        self.n_targets = m;
        self.single_output = meta_i64(state, "classic.single_output").unwrap_or(1) != 0;
        self.dtype = dtype;
        Ok(())
    }
}

/// Multinomial logistic regression trained with full-batch gradient descent.
///
/// Labels must be contiguous class indices `0..n_classes`. Use
/// [`StandardScaler`](super::StandardScaler) first if features have very
/// different scales.
pub struct LogisticRegression {
    ctx: Ctx,
    coef: Option<Tensor>,
    intercept: Option<Tensor>,
    n_features: usize,
    n_classes: usize,
    max_iter: usize,
    tol: f64,
    l2: f64,
    dtype: DTypeId,
}

impl LogisticRegression {
    /// Create an unfitted classifier.
    pub fn new(ctx: &Ctx) -> Self {
        LogisticRegression {
            ctx: ctx.clone(),
            coef: None,
            intercept: None,
            n_features: 0,
            n_classes: 0,
            max_iter: 500,
            tol: 1e-8,
            l2: 1e-4,
            dtype: crate::dtype::FLOAT32,
        }
    }

    /// Set the maximum number of full-batch iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter.max(1);
        self
    }

    /// Set the L2 regularization strength.
    pub fn with_l2(mut self, l2: f64) -> Self {
        self.l2 = l2.max(0.0);
        self
    }

    /// Set the gradient-norm convergence tolerance.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol.max(0.0);
        self
    }

    /// Fit the classifier.
    #[allow(clippy::needless_range_loop)]
    pub fn fit(&mut self, x: &Tensor, y: &Tensor) -> Result<()> {
        let (xv, n, d) = read_matrix(x, "LogisticRegression.x")?;
        let labels = read_labels(y, n, "LogisticRegression.y")?;
        let n_classes = labels.iter().copied().max().unwrap_or(-1) + 1;
        if n_classes < 2 {
            return Err(ClassicError::Data(
                "LogisticRegression needs at least two classes".into(),
            ));
        }
        if labels.iter().any(|&v| v < 0 || v >= n_classes) {
            return Err(ClassicError::Data(
                "LogisticRegression labels must be contiguous 0..n_classes-1".into(),
            ));
        }
        let k = n_classes as usize;
        let dtype = float_dtype(x)?;
        let device = x.device();
        let mut w = vec![0.0; d * k];
        let mut b = vec![0.0; k];
        let mut p = vec![0.0; k];
        let mut prev_loss = f64::INFINITY;
        for _ in 0..self.max_iter {
            // Loss and gradients.
            let mut gw = vec![0.0; d * k];
            let mut gb = vec![0.0; k];
            let mut loss = 0.0;
            for i in 0..n {
                let mut max_logit = f64::NEG_INFINITY;
                for c in 0..k {
                    let mut z = b[c];
                    for j in 0..d {
                        z += xv[i * d + j] * w[j * k + c];
                    }
                    p[c] = z;
                    max_logit = max_logit.max(z);
                }
                let mut zsum = 0.0;
                for c in 0..k {
                    p[c] = (p[c] - max_logit).exp();
                    zsum += p[c];
                }
                for c in 0..k {
                    p[c] /= zsum;
                }
                let yi = labels[i] as usize;
                loss -= p[yi].max(1e-300).ln();
                for c in 0..k {
                    let diff = p[c] - if c == yi { 1.0 } else { 0.0 };
                    gb[c] += diff;
                    for j in 0..d {
                        gw[j * k + c] += xv[i * d + j] * diff;
                    }
                }
            }
            loss /= n as f64;
            loss += 0.5 * self.l2 * w.iter().map(|v| v * v).sum::<f64>();
            for v in &mut gw {
                *v /= n as f64;
            }
            for c in 0..k {
                gb[c] /= n as f64;
            }
            for j in 0..d {
                for c in 0..k {
                    gw[j * k + c] += self.l2 * w[j * k + c];
                }
            }
            let gnorm = gw.iter().map(|v| v * v).sum::<f64>().sqrt()
                + gb.iter().map(|v| v * v).sum::<f64>().sqrt();
            if gnorm <= self.tol {
                break;
            }
            // Backtracking line search on the full-batch loss.
            let mut step = 1.0;
            let mut accepted = false;
            let mut new_loss = loss;
            for _ in 0..30 {
                let mut cand_w = w.clone();
                let mut cand_b = b.clone();
                for j in 0..d {
                    for c in 0..k {
                        cand_w[j * k + c] -= step * gw[j * k + c];
                    }
                }
                for c in 0..k {
                    cand_b[c] -= step * gb[c];
                }
                let cand_loss = softmax_loss(&xv, &labels, n, d, k, &cand_w, &cand_b, self.l2);
                if cand_loss <= loss + 1e-12 || step < 1e-10 {
                    w = cand_w;
                    b = cand_b;
                    new_loss = cand_loss;
                    accepted = true;
                    break;
                }
                step *= 0.5;
            }
            if !accepted {
                break;
            }
            if prev_loss.is_finite() && (prev_loss - new_loss).abs() < self.tol {
                break;
            }
            prev_loss = new_loss;
        }
        self.coef = Some(tensor_from_f64(
            &self.ctx,
            &w,
            &[d as i64, k as i64],
            dtype,
            device.as_deref(),
        ));
        self.intercept = Some(tensor_from_f64(
            &self.ctx,
            &b,
            &[k as i64],
            dtype,
            device.as_deref(),
        ));
        self.n_features = d;
        self.n_classes = k;
        self.dtype = dtype;
        Ok(())
    }

    /// Class probabilities, shape `[n, n_classes]`.
    pub fn predict_proba(&self, x: &Tensor) -> Result<Tensor> {
        let coef = require_fitted(&self.coef, "LogisticRegression")?;
        let intercept = require_fitted(&self.intercept, "LogisticRegression")?;
        let sh = super::shape_i64(x);
        if sh.len() != 2 || sh[1] as usize != self.n_features {
            return Err(ClassicError::Shape(format!(
                "LogisticRegression expects [n, {}], got {sh:?}",
                self.n_features
            )));
        }
        let logits = x.matmul(coef).add(intercept);
        Ok(crate::tensor::nn::softmax(&logits, Some(1)))
    }

    /// Predicted class indices `[n]` (int64).
    ///
    /// The class is selected on the host from the probability matrix; this
    /// avoids relying on exact floating-point equality in a graph `argmax`
    /// when the probabilities come from the fused softmax path.
    pub fn predict(&self, x: &Tensor) -> Result<Tensor> {
        let proba = self.predict_proba(x)?;
        let vals = proba.to_list_fresh::<f64>();
        let k = self.n_classes;
        let n = vals.len() / k;
        let mut labels = Vec::with_capacity(n);
        for row in 0..n {
            let mut best = 0usize;
            let mut best_v = f64::NEG_INFINITY;
            for c in 0..k {
                let v = vals[row * k + c];
                if v > best_v {
                    best_v = v;
                    best = c;
                }
            }
            labels.push(best as i64);
        }
        let device = x.device();
        Ok(super::tensor_from_i64(
            &self.ctx,
            &labels,
            &[n as i64],
            device.as_deref(),
        ))
    }

    /// Fitted coefficient matrix `[n_features, n_classes]`.
    pub fn coef(&self) -> Option<&Tensor> {
        self.coef.as_ref()
    }

    /// Number of classes seen during fitting.
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }
}

#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
fn softmax_loss(
    x: &[f64],
    y: &[i64],
    n: usize,
    d: usize,
    k: usize,
    w: &[f64],
    b: &[f64],
    l2: f64,
) -> f64 {
    let mut loss = 0.0;
    for i in 0..n {
        let mut max_logit = f64::NEG_INFINITY;
        let mut logits = vec![0.0; k];
        for c in 0..k {
            let mut z = b[c];
            for j in 0..d {
                z += x[i * d + j] * w[j * k + c];
            }
            logits[c] = z;
            max_logit = max_logit.max(z);
        }
        let mut zsum = 0.0;
        for c in 0..k {
            logits[c] = (logits[c] - max_logit).exp();
            zsum += logits[c];
        }
        loss -= (logits[y[i] as usize] / zsum).max(1e-300).ln();
    }
    loss / n as f64 + 0.5 * l2 * w.iter().map(|v| v * v).sum::<f64>()
}

impl Persist for LogisticRegression {
    fn state(&self) -> Result<State> {
        let coef = require_fitted(&self.coef, "LogisticRegression")?;
        let intercept = require_fitted(&self.intercept, "LogisticRegression")?;
        let mut s = State::new();
        s.insert("coef", super::detach_tensor(coef)?);
        s.insert("intercept", super::detach_tensor(intercept)?);
        put_meta(&mut s, "classic.kind", "logistic_regression");
        put_meta(&mut s, "classic.version", 1);
        put_meta(&mut s, "classic.n_features", self.n_features);
        put_meta(&mut s, "classic.n_classes", self.n_classes);
        put_meta(&mut s, "classic.max_iter", self.max_iter);
        put_meta(&mut s, "classic.tol", self.tol);
        put_meta(&mut s, "classic.l2", self.l2);
        put_meta(&mut s, "classic.dtype", super::dtype_label(self.dtype));
        Ok(s)
    }

    fn load_state(&mut self, state: &State) -> Result<()> {
        if meta_str_optional(state, "classic.kind") != Some("logistic_regression") {
            return Err(ClassicError::State(
                "state is not a LogisticRegression".into(),
            ));
        }
        let d = meta_i64(state, "classic.n_features")? as usize;
        let k = meta_i64(state, "classic.n_classes")? as usize;
        let dtype =
            dtype_from_debug(meta_str_optional(state, "classic.dtype").unwrap_or("FLOAT32"))?;
        let w = state_f64(state, "coef")?;
        let b = state_f64(state, "intercept")?;
        if w.len() != d * k || b.len() != k {
            return Err(ClassicError::State(
                "LogisticRegression state has inconsistent dimensions".into(),
            ));
        }
        self.coef = Some(tensor_from_f64(
            &self.ctx,
            &w,
            &[d as i64, k as i64],
            dtype,
            None,
        ));
        self.intercept = Some(tensor_from_f64(&self.ctx, &b, &[k as i64], dtype, None));
        self.n_features = d;
        self.n_classes = k;
        self.max_iter = meta_i64(state, "classic.max_iter").unwrap_or(500) as usize;
        self.tol = meta_f64(state, "classic.tol").unwrap_or(1e-8);
        self.l2 = meta_f64(state, "classic.l2").unwrap_or(1e-4);
        self.dtype = dtype;
        Ok(())
    }
}
