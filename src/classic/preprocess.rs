//! Feature preprocessing.

use super::{
    float_dtype, meta_i64, put_meta, read_matrix, require_fitted, state_f64, tensor_from_f64,
    ClassicError, Persist, Result,
};
use crate::dtype::DTypeId;
use crate::tensor::state::State;
use crate::tensor::{Ctx, Tensor};

/// Standardize features to zero mean and unit variance.
///
/// The variance uses the population convention (`ddof = 0`), matching
/// scikit-learn's `StandardScaler`. A feature with zero variance gets a scale
/// of one.
pub struct StandardScaler {
    ctx: Ctx,
    with_mean: bool,
    with_std: bool,
    mean: Option<Tensor>,
    scale: Option<Tensor>,
    n_features: usize,
    dtype: DTypeId,
}

impl StandardScaler {
    /// Create an unfitted scaler in `ctx`.
    pub fn new(ctx: &Ctx) -> Self {
        StandardScaler {
            ctx: ctx.clone(),
            with_mean: true,
            with_std: true,
            mean: None,
            scale: None,
            n_features: 0,
            dtype: crate::dtype::FLOAT32,
        }
    }

    /// Disable mean centering.
    pub fn without_mean(mut self) -> Self {
        self.with_mean = false;
        self
    }

    /// Disable variance scaling.
    pub fn without_std(mut self) -> Self {
        self.with_std = false;
        self
    }

    /// Fit the per-feature mean and scale.
    pub fn fit(&mut self, x: &Tensor) -> Result<()> {
        let (vals, n, d) = read_matrix(x, "StandardScaler.x")?;
        let dtype = float_dtype(x)?;
        let device = x.device();
        let mut mean = vec![0.0; d];
        for row in 0..n {
            for col in 0..d {
                mean[col] += vals[row * d + col];
            }
        }
        for v in &mut mean {
            *v /= n as f64;
        }
        let mut scale = vec![1.0; d];
        if self.with_std {
            for col in 0..d {
                let mut var = 0.0;
                for row in 0..n {
                    let dx = vals[row * d + col] - mean[col];
                    var += dx * dx;
                }
                var /= n as f64;
                let s = var.sqrt();
                scale[col] = if s > 1e-15 { s } else { 1.0 };
            }
        }
        if !self.with_mean {
            mean.fill(0.0);
        }
        self.mean = Some(tensor_from_f64(
            &self.ctx,
            &mean,
            &[d as i64],
            dtype,
            device.as_deref(),
        ));
        self.scale = Some(tensor_from_f64(
            &self.ctx,
            &scale,
            &[d as i64],
            dtype,
            device.as_deref(),
        ));
        self.n_features = d;
        self.dtype = dtype;
        Ok(())
    }

    /// Whether the scaler has been fitted.
    pub fn is_fitted(&self) -> bool {
        self.mean.is_some() && self.scale.is_some()
    }

    /// Standardize `x`.
    pub fn transform(&self, x: &Tensor) -> Result<Tensor> {
        let mean = require_fitted(&self.mean, "StandardScaler")?;
        let scale = require_fitted(&self.scale, "StandardScaler")?;
        let sh = super::shape_i64(x);
        if sh.len() != 2 || sh[1] as usize != self.n_features {
            return Err(ClassicError::Shape(format!(
                "StandardScaler expects [n, {}], got {:?}",
                self.n_features,
                super::shape_i64(x)
            )));
        }
        Ok(x.sub(mean).div(scale))
    }

    /// Undo [`StandardScaler::transform`].
    pub fn inverse_transform(&self, x: &Tensor) -> Result<Tensor> {
        let mean = require_fitted(&self.mean, "StandardScaler")?;
        let scale = require_fitted(&self.scale, "StandardScaler")?;
        Ok(x.mul(scale).add(mean))
    }

    /// Fit and transform in one call.
    pub fn fit_transform(&mut self, x: &Tensor) -> Result<Tensor> {
        self.fit(x)?;
        self.transform(x)
    }

    /// Number of features seen during fitting (0 before `fit`).
    pub fn n_features(&self) -> usize {
        self.n_features
    }
}

impl Persist for StandardScaler {
    fn state(&self) -> Result<State> {
        let mean = require_fitted(&self.mean, "StandardScaler")?;
        let scale = require_fitted(&self.scale, "StandardScaler")?;
        let mut s = State::new();
        s.insert("mean", super::detach_tensor(mean)?);
        s.insert("scale", super::detach_tensor(scale)?);
        put_meta(&mut s, "classic.kind", "standard_scaler");
        put_meta(&mut s, "classic.version", 1);
        put_meta(&mut s, "classic.n_features", self.n_features);
        put_meta(&mut s, "classic.with_mean", self.with_mean as u8);
        put_meta(&mut s, "classic.with_std", self.with_std as u8);
        put_meta(&mut s, "classic.dtype", super::dtype_label(self.dtype));
        Ok(s)
    }

    fn load_state(&mut self, state: &State) -> Result<()> {
        if super::meta_str_optional(state, "classic.kind") != Some("standard_scaler") {
            return Err(ClassicError::State("state is not a StandardScaler".into()));
        }
        let mean = state_f64(state, "mean")?;
        let scale = state_f64(state, "scale")?;
        if mean.len() != scale.len() || mean.is_empty() {
            return Err(ClassicError::State(
                "StandardScaler state has inconsistent mean/scale".into(),
            ));
        }
        let dtype = super::dtype_from_debug(
            super::meta_str_optional(state, "classic.dtype").unwrap_or("FLOAT32"),
        )?;
        let d = mean.len() as i64;
        self.mean = Some(tensor_from_f64(&self.ctx, &mean, &[d], dtype, None));
        self.scale = Some(tensor_from_f64(&self.ctx, &scale, &[d], dtype, None));
        self.n_features = mean.len();
        self.dtype = dtype;
        self.with_mean = meta_i64(state, "classic.with_mean").unwrap_or(1) != 0;
        self.with_std = meta_i64(state, "classic.with_std").unwrap_or(1) != 0;
        Ok(())
    }
}
