//! Linear support-vector machines.

use super::{
    dtype_from_debug, float_dtype, meta_f64, meta_i64, meta_str_optional, put_meta, read_labels,
    read_matrix, read_target, require_fitted, shape_i64, state_f64, tensor_from_f64, ClassicError,
    Persist, Result,
};
use crate::dtype::DTypeId;
use crate::tensor::state::State;
use crate::tensor::{Ctx, Tensor};

fn dot_row(x: &[f64], w: &[f64], d: usize) -> f64 {
    (0..d).map(|j| x[j] * w[j]).sum()
}

/// Linear support-vector classifier (one-vs-rest, squared-free hinge loss).
pub struct LinearSVC {
    ctx: Ctx,
    coef: Option<Tensor>,
    intercept: Option<Tensor>,
    n_features: usize,
    n_classes: usize,
    l2: f64,
    max_iter: usize,
    tol: f64,
    dtype: DTypeId,
}

impl LinearSVC {
    /// Create an unfitted linear SVC.
    pub fn new(ctx: &Ctx) -> Self {
        LinearSVC {
            ctx: ctx.clone(),
            coef: None,
            intercept: None,
            n_features: 0,
            n_classes: 0,
            l2: 1e-4,
            max_iter: 1000,
            tol: 1e-8,
            dtype: crate::dtype::FLOAT32,
        }
    }

    /// Set the L2 regularization strength.
    pub fn with_l2(mut self, l2: f64) -> Self {
        self.l2 = l2.max(0.0);
        self
    }

    /// Set the maximum number of full-batch iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter.max(1);
        self
    }

    /// Set the convergence tolerance.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol.max(0.0);
        self
    }

    /// Fit one-vs-rest linear classifiers.
    pub fn fit(&mut self, x: &Tensor, y: &Tensor) -> Result<()> {
        let (xv, n, d) = read_matrix(x, "LinearSVC.x")?;
        let labels = read_labels(y, n, "LinearSVC.y")?;
        let k = labels.iter().copied().max().unwrap_or(-1) + 1;
        if k < 2 {
            return Err(ClassicError::Data(
                "LinearSVC needs at least two classes".into(),
            ));
        }
        if labels.iter().any(|&v| v < 0 || v >= k) {
            return Err(ClassicError::Data(
                "LinearSVC labels must be contiguous 0..n_classes-1".into(),
            ));
        }
        let k = k as usize;
        let dtype = float_dtype(x)?;
        let device = x.device();
        let mut coef = vec![0.0; k * d];
        let mut intercept = vec![0.0; k];
        for c in 0..k {
            let mut w = vec![0.0; d];
            let mut b = 0.0;
            let mut prev = f64::INFINITY;
            for _ in 0..self.max_iter {
                let loss = hinge_loss(&xv, &labels, n, d, &w, b, c as i64, self.l2);
                let mut gw = vec![0.0; d];
                let mut gb = 0.0;
                for i in 0..n {
                    let yt = if labels[i] == c as i64 { 1.0 } else { -1.0 };
                    let margin = yt * (dot_row(&xv[i * d..i * d + d], &w, d) + b);
                    if margin < 1.0 {
                        for j in 0..d {
                            gw[j] -= yt * xv[i * d + j] / n as f64;
                        }
                        gb -= yt / n as f64;
                    }
                }
                for j in 0..d {
                    gw[j] += self.l2 * w[j];
                }
                let gnorm = gw.iter().map(|v| v * v).sum::<f64>().sqrt() + gb.abs();
                if gnorm <= self.tol {
                    break;
                }
                let mut step = 1.0;
                let mut accepted = false;
                let mut new_loss = loss;
                for _ in 0..30 {
                    let mut cw = w.clone();
                    for j in 0..d {
                        cw[j] -= step * gw[j];
                    }
                    let cb = b - step * gb;
                    let cand = hinge_loss(&xv, &labels, n, d, &cw, cb, c as i64, self.l2);
                    if cand <= loss + 1e-12 || step < 1e-10 {
                        w = cw;
                        b = cb;
                        new_loss = cand;
                        accepted = true;
                        break;
                    }
                    step *= 0.5;
                }
                if !accepted || (prev.is_finite() && (prev - new_loss).abs() < self.tol) {
                    break;
                }
                prev = new_loss;
            }
            for j in 0..d {
                coef[c * d + j] = w[j];
            }
            intercept[c] = b;
        }
        self.coef = Some(tensor_from_f64(
            &self.ctx,
            &coef,
            &[k as i64, d as i64],
            dtype,
            device.as_deref(),
        ));
        self.intercept = Some(tensor_from_f64(
            &self.ctx,
            &intercept,
            &[k as i64],
            dtype,
            device.as_deref(),
        ));
        self.n_features = d;
        self.n_classes = k;
        self.dtype = dtype;
        Ok(())
    }

    /// One-vs-rest decision scores `[n, n_classes]`.
    pub fn decision_function(&self, x: &Tensor) -> Result<Tensor> {
        let coef = require_fitted(&self.coef, "LinearSVC")?;
        let intercept = require_fitted(&self.intercept, "LinearSVC")?;
        let sh = shape_i64(x);
        if sh.len() != 2 || sh[1] as usize != self.n_features {
            return Err(ClassicError::Shape(format!(
                "LinearSVC expects [n, {}], got {sh:?}",
                self.n_features
            )));
        }
        let x = if x.dtype() != coef.dtype() {
            x.cast(coef.dtype())
        } else {
            x.clone()
        };
        Ok(x.matmul(coef.transpose(0, 1)).add(intercept))
    }

    /// Predicted class indices `[n]` (int64).
    pub fn predict(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self
            .decision_function(x)?
            .argmax(Some(1), false)
            .cast(crate::dtype::INT64))
    }

    /// Fitted coefficients `[n_classes, n_features]`.
    pub fn coef(&self) -> Option<&Tensor> {
        self.coef.as_ref()
    }
}

#[allow(clippy::too_many_arguments)]
fn hinge_loss(
    x: &[f64],
    y: &[i64],
    n: usize,
    d: usize,
    w: &[f64],
    b: f64,
    class: i64,
    l2: f64,
) -> f64 {
    let mut loss = 0.0;
    for i in 0..n {
        let yt = if y[i] == class { 1.0 } else { -1.0 };
        let margin = yt * (dot_row(&x[i * d..i * d + d], w, d) + b);
        loss += (1.0 - margin).max(0.0);
    }
    loss / n as f64 + 0.5 * l2 * w.iter().map(|v| v * v).sum::<f64>()
}

/// Linear epsilon-insensitive support-vector regression.
pub struct LinearSVR {
    ctx: Ctx,
    coef: Option<Tensor>,
    intercept: Option<Tensor>,
    n_features: usize,
    epsilon: f64,
    l2: f64,
    max_iter: usize,
    tol: f64,
    dtype: DTypeId,
}

impl LinearSVR {
    /// Create an unfitted linear SVR with `epsilon`.
    pub fn new(ctx: &Ctx, epsilon: f64) -> Self {
        LinearSVR {
            ctx: ctx.clone(),
            coef: None,
            intercept: None,
            n_features: 0,
            epsilon: epsilon.max(0.0),
            l2: 1e-4,
            max_iter: 1000,
            tol: 1e-8,
            dtype: crate::dtype::FLOAT32,
        }
    }

    /// Set the L2 regularization strength.
    pub fn with_l2(mut self, l2: f64) -> Self {
        self.l2 = l2.max(0.0);
        self
    }

    /// Set the maximum number of full-batch iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter.max(1);
        self
    }

    /// Fit the regression model.
    pub fn fit(&mut self, x: &Tensor, y: &Tensor) -> Result<()> {
        let (xv, n, d) = read_matrix(x, "LinearSVR.x")?;
        let (yv, m, single) = read_target(y, n, "LinearSVR.y")?;
        if m != 1 {
            return Err(ClassicError::Shape(
                "LinearSVR supports a single target".into(),
            ));
        }
        let dtype = float_dtype(x)?;
        let device = x.device();
        let mut w = vec![0.0; d];
        let mut b = 0.0;
        let mut prev = f64::INFINITY;
        for _ in 0..self.max_iter {
            let loss = epsilon_loss(&xv, &yv, n, d, &w, b, self.epsilon, self.l2);
            let mut gw = vec![0.0; d];
            let mut gb = 0.0;
            for i in 0..n {
                let r = dot_row(&xv[i * d..i * d + d], &w, d) + b - yv[i];
                if r > self.epsilon {
                    for j in 0..d {
                        gw[j] += xv[i * d + j] / n as f64;
                    }
                    gb += 1.0 / n as f64;
                } else if r < -self.epsilon {
                    for j in 0..d {
                        gw[j] -= xv[i * d + j] / n as f64;
                    }
                    gb -= 1.0 / n as f64;
                }
            }
            for j in 0..d {
                gw[j] += self.l2 * w[j];
            }
            let gnorm = gw.iter().map(|v| v * v).sum::<f64>().sqrt() + gb.abs();
            if gnorm <= self.tol {
                break;
            }
            let mut step = 1.0;
            let mut accepted = false;
            let mut new_loss = loss;
            for _ in 0..30 {
                let mut cw = w.clone();
                for j in 0..d {
                    cw[j] -= step * gw[j];
                }
                let cb = b - step * gb;
                let cand = epsilon_loss(&xv, &yv, n, d, &cw, cb, self.epsilon, self.l2);
                if cand <= loss + 1e-12 || step < 1e-10 {
                    w = cw;
                    b = cb;
                    new_loss = cand;
                    accepted = true;
                    break;
                }
                step *= 0.5;
            }
            if !accepted || (prev.is_finite() && (prev - new_loss).abs() < self.tol) {
                break;
            }
            prev = new_loss;
        }
        self.coef = Some(tensor_from_f64(
            &self.ctx,
            &w,
            &[d as i64, 1],
            dtype,
            device.as_deref(),
        ));
        self.intercept = Some(tensor_from_f64(
            &self.ctx,
            &[b],
            &[1],
            dtype,
            device.as_deref(),
        ));
        self.n_features = d;
        let _ = single;
        self.dtype = dtype;
        Ok(())
    }

    /// Predict a single target `[n]`.
    pub fn predict(&self, x: &Tensor) -> Result<Tensor> {
        let coef = require_fitted(&self.coef, "LinearSVR")?;
        let intercept = require_fitted(&self.intercept, "LinearSVR")?;
        let sh = shape_i64(x);
        if sh.len() != 2 || sh[1] as usize != self.n_features {
            return Err(ClassicError::Shape(format!(
                "LinearSVR expects [n, {}], got {sh:?}",
                self.n_features
            )));
        }
        let x = if x.dtype() != coef.dtype() {
            x.cast(coef.dtype())
        } else {
            x.clone()
        };
        Ok(x.matmul(coef).add(intercept).squeeze(Some(-1)))
    }

    /// Fitted coefficients `[n_features, 1]`.
    pub fn coef(&self) -> Option<&Tensor> {
        self.coef.as_ref()
    }
}

#[allow(clippy::too_many_arguments)]
fn epsilon_loss(
    x: &[f64],
    y: &[f64],
    n: usize,
    d: usize,
    w: &[f64],
    b: f64,
    epsilon: f64,
    l2: f64,
) -> f64 {
    let mut loss = 0.0;
    for i in 0..n {
        let r = (dot_row(&x[i * d..i * d + d], w, d) + b - y[i]).abs();
        loss += (r - epsilon).max(0.0);
    }
    loss / n as f64 + 0.5 * l2 * w.iter().map(|v| v * v).sum::<f64>()
}

impl Persist for LinearSVC {
    fn state(&self) -> Result<State> {
        let coef = require_fitted(&self.coef, "LinearSVC")?;
        let intercept = require_fitted(&self.intercept, "LinearSVC")?;
        let mut s = State::new();
        s.insert("coef", super::detach_tensor(coef)?);
        s.insert("intercept", super::detach_tensor(intercept)?);
        put_meta(&mut s, "classic.kind", "linear_svc");
        put_meta(&mut s, "classic.version", 1);
        put_meta(&mut s, "classic.n_features", self.n_features);
        put_meta(&mut s, "classic.n_classes", self.n_classes);
        put_meta(&mut s, "classic.l2", self.l2);
        put_meta(&mut s, "classic.max_iter", self.max_iter);
        put_meta(&mut s, "classic.tol", self.tol);
        put_meta(&mut s, "classic.dtype", super::dtype_label(self.dtype));
        Ok(s)
    }

    fn load_state(&mut self, state: &State) -> Result<()> {
        if meta_str_optional(state, "classic.kind") != Some("linear_svc") {
            return Err(ClassicError::State("state is not a LinearSVC".into()));
        }
        let d = meta_i64(state, "classic.n_features")? as usize;
        let k = meta_i64(state, "classic.n_classes")? as usize;
        let dtype =
            dtype_from_debug(meta_str_optional(state, "classic.dtype").unwrap_or("FLOAT32"))?;
        let coef = state_f64(state, "coef")?;
        let b = state_f64(state, "intercept")?;
        if coef.len() != k * d || b.len() != k {
            return Err(ClassicError::State(
                "LinearSVC state has inconsistent dimensions".into(),
            ));
        }
        self.coef = Some(tensor_from_f64(
            &self.ctx,
            &coef,
            &[k as i64, d as i64],
            dtype,
            None,
        ));
        self.intercept = Some(tensor_from_f64(&self.ctx, &b, &[k as i64], dtype, None));
        self.n_features = d;
        self.n_classes = k;
        self.l2 = meta_f64(state, "classic.l2").unwrap_or(1e-4);
        self.max_iter = meta_i64(state, "classic.max_iter").unwrap_or(1000) as usize;
        self.tol = meta_f64(state, "classic.tol").unwrap_or(1e-8);
        self.dtype = dtype;
        Ok(())
    }
}

impl Persist for LinearSVR {
    fn state(&self) -> Result<State> {
        let coef = require_fitted(&self.coef, "LinearSVR")?;
        let intercept = require_fitted(&self.intercept, "LinearSVR")?;
        let mut s = State::new();
        s.insert("coef", super::detach_tensor(coef)?);
        s.insert("intercept", super::detach_tensor(intercept)?);
        put_meta(&mut s, "classic.kind", "linear_svr");
        put_meta(&mut s, "classic.version", 1);
        put_meta(&mut s, "classic.n_features", self.n_features);
        put_meta(&mut s, "classic.epsilon", self.epsilon);
        put_meta(&mut s, "classic.l2", self.l2);
        put_meta(&mut s, "classic.max_iter", self.max_iter);
        put_meta(&mut s, "classic.tol", self.tol);
        put_meta(&mut s, "classic.dtype", super::dtype_label(self.dtype));
        Ok(s)
    }

    fn load_state(&mut self, state: &State) -> Result<()> {
        if meta_str_optional(state, "classic.kind") != Some("linear_svr") {
            return Err(ClassicError::State("state is not a LinearSVR".into()));
        }
        let d = meta_i64(state, "classic.n_features")? as usize;
        let dtype =
            dtype_from_debug(meta_str_optional(state, "classic.dtype").unwrap_or("FLOAT32"))?;
        let coef = state_f64(state, "coef")?;
        let b = state_f64(state, "intercept")?;
        if coef.len() != d || b.len() != 1 {
            return Err(ClassicError::State(
                "LinearSVR state has inconsistent dimensions".into(),
            ));
        }
        self.coef = Some(tensor_from_f64(
            &self.ctx,
            &coef,
            &[d as i64, 1],
            dtype,
            None,
        ));
        self.intercept = Some(tensor_from_f64(&self.ctx, &b, &[1], dtype, None));
        self.n_features = d;
        self.epsilon = meta_f64(state, "classic.epsilon").unwrap_or(0.0);
        self.l2 = meta_f64(state, "classic.l2").unwrap_or(1e-4);
        self.max_iter = meta_i64(state, "classic.max_iter").unwrap_or(1000) as usize;
        self.tol = meta_f64(state, "classic.tol").unwrap_or(1e-8);
        self.dtype = dtype;
        Ok(())
    }
}
