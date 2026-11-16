//! PCA / linear dimensionality reduction.

use super::linalg::symmetric_eigh;
use super::{
    dtype_from_debug, float_dtype, meta_f64, meta_i64, meta_str_optional, put_meta, read_matrix,
    require_fitted, shape_i64, state_f64, tensor_from_f64, ClassicError, Persist, Result,
};
use crate::dtype::DTypeId;
use crate::tensor::state::State;
use crate::tensor::{Ctx, Tensor};

/// Principal component analysis using a host eigendecomposition of the
/// covariance matrix.
///
/// The fitted components are `[n_components, n_features]`; `transform` projects
/// centered data and `inverse_transform` reconstructs it. Component signs are
/// chosen deterministically (largest absolute loading positive).
pub struct PCA {
    ctx: Ctx,
    components: Option<Tensor>,
    mean: Option<Tensor>,
    explained_variance: Option<Tensor>,
    n_components: usize,
    n_features: usize,
    total_variance: f64,
    dtype: DTypeId,
}

impl PCA {
    /// Create a PCA with `n_components` output dimensions.
    pub fn new(ctx: &Ctx, n_components: usize) -> Self {
        PCA {
            ctx: ctx.clone(),
            components: None,
            mean: None,
            explained_variance: None,
            n_components,
            n_features: 0,
            total_variance: 0.0,
            dtype: crate::dtype::FLOAT32,
        }
    }

    /// Fit the principal components of `x`.
    pub fn fit(&mut self, x: &Tensor) -> Result<()> {
        let (xv, n, d) = read_matrix(x, "PCA.x")?;
        if self.n_components == 0 || self.n_components > d {
            return Err(ClassicError::Invalid(format!(
                "PCA n_components must be in 1..={d}, got {}",
                self.n_components
            )));
        }
        let dtype = float_dtype(x)?;
        let device = x.device();
        let mut mean = vec![0.0; d];
        for i in 0..n {
            for j in 0..d {
                mean[j] += xv[i * d + j];
            }
        }
        for v in &mut mean {
            *v /= n as f64;
        }
        let mut cov = vec![0.0; d * d];
        for i in 0..n {
            for p in 0..d {
                let a = xv[i * d + p] - mean[p];
                for q in p..d {
                    cov[p * d + q] += a * (xv[i * d + q] - mean[q]);
                }
            }
        }
        let denom = (n.saturating_sub(1)).max(1) as f64;
        for p in 0..d {
            for q in 0..p {
                cov[p * d + q] = cov[q * d + p];
            }
        }
        for v in &mut cov {
            *v /= denom;
        }
        let (evals, evecs) = symmetric_eigh(&cov, d)?;
        let k = self.n_components;
        let mut comps = vec![0.0; k * d];
        for c in 0..k {
            // Deterministic sign: the largest absolute loading is positive.
            let mut max_at = 0usize;
            let mut max_abs = 0.0;
            for j in 0..d {
                let v = evecs[j * d + c].abs();
                if v > max_abs {
                    max_abs = v;
                    max_at = j;
                }
            }
            let sign = if evecs[max_at * d + c] < 0.0 {
                -1.0
            } else {
                1.0
            };
            for j in 0..d {
                comps[c * d + j] = sign * evecs[j * d + c];
            }
        }
        self.components = Some(tensor_from_f64(
            &self.ctx,
            &comps,
            &[k as i64, d as i64],
            dtype,
            device.as_deref(),
        ));
        self.mean = Some(tensor_from_f64(
            &self.ctx,
            &mean,
            &[d as i64],
            dtype,
            device.as_deref(),
        ));
        self.explained_variance = Some(tensor_from_f64(
            &self.ctx,
            &evals[..k],
            &[k as i64],
            dtype,
            device.as_deref(),
        ));
        self.n_features = d;
        self.total_variance = evals.iter().map(|v| v.max(0.0)).sum();
        self.dtype = dtype;
        Ok(())
    }

    /// Project `x` (`[n, n_features]`) to `[n, n_components]`.
    pub fn transform(&self, x: &Tensor) -> Result<Tensor> {
        let comps = require_fitted(&self.components, "PCA")?;
        let mean = require_fitted(&self.mean, "PCA")?;
        let sh = shape_i64(x);
        if sh.len() != 2 || sh[1] as usize != self.n_features {
            return Err(ClassicError::Shape(format!(
                "PCA expects [n, {}], got {sh:?}",
                self.n_features
            )));
        }
        Ok(x.sub(mean).matmul(comps.transpose(0, 1)))
    }

    /// Reconstruct `[n, n_features]` from a projection.
    pub fn inverse_transform(&self, z: &Tensor) -> Result<Tensor> {
        let comps = require_fitted(&self.components, "PCA")?;
        let mean = require_fitted(&self.mean, "PCA")?;
        let sh = shape_i64(z);
        if sh.len() != 2 || sh[1] as usize != self.n_components {
            return Err(ClassicError::Shape(format!(
                "PCA projection must be [n, {}], got {sh:?}",
                self.n_components
            )));
        }
        Ok(z.matmul(comps).add(mean))
    }

    /// Fit and project in one call.
    pub fn fit_transform(&mut self, x: &Tensor) -> Result<Tensor> {
        self.fit(x)?;
        self.transform(x)
    }

    /// Fitted components `[n_components, n_features]`.
    pub fn components(&self) -> Option<&Tensor> {
        self.components.as_ref()
    }

    /// Per-feature mean used for centering.
    pub fn mean(&self) -> Option<&Tensor> {
        self.mean.as_ref()
    }

    /// Explained variance of each retained component.
    pub fn explained_variance(&self) -> Option<&Tensor> {
        self.explained_variance.as_ref()
    }

    /// Explained variance ratios, as a host vector.
    pub fn explained_variance_ratio(&self) -> Result<Vec<f64>> {
        let ev = require_fitted(&self.explained_variance, "PCA")?.to_list_fresh::<f64>();
        if self.total_variance <= 0.0 {
            return Ok(vec![0.0; ev.len()]);
        }
        Ok(ev.iter().map(|v| v / self.total_variance).collect())
    }
}

impl Persist for PCA {
    fn state(&self) -> Result<State> {
        let comps = require_fitted(&self.components, "PCA")?;
        let mean = require_fitted(&self.mean, "PCA")?;
        let ev = require_fitted(&self.explained_variance, "PCA")?;
        let mut s = State::new();
        s.insert("components", super::detach_tensor(comps)?);
        s.insert("mean", super::detach_tensor(mean)?);
        s.insert("explained_variance", super::detach_tensor(ev)?);
        put_meta(&mut s, "classic.kind", "pca");
        put_meta(&mut s, "classic.version", 1);
        put_meta(&mut s, "classic.n_components", self.n_components);
        put_meta(&mut s, "classic.n_features", self.n_features);
        put_meta(&mut s, "classic.total_variance", self.total_variance);
        put_meta(&mut s, "classic.dtype", super::dtype_label(self.dtype));
        Ok(s)
    }

    fn load_state(&mut self, state: &State) -> Result<()> {
        if meta_str_optional(state, "classic.kind") != Some("pca") {
            return Err(ClassicError::State("state is not a PCA".into()));
        }
        let k = meta_i64(state, "classic.n_components")? as usize;
        let d = meta_i64(state, "classic.n_features")? as usize;
        let dtype =
            dtype_from_debug(meta_str_optional(state, "classic.dtype").unwrap_or("FLOAT32"))?;
        let comps = state_f64(state, "components")?;
        let mean = state_f64(state, "mean")?;
        let ev = state_f64(state, "explained_variance")?;
        if comps.len() != k * d || mean.len() != d || ev.len() != k {
            return Err(ClassicError::State(
                "PCA state has inconsistent dimensions".into(),
            ));
        }
        self.components = Some(tensor_from_f64(
            &self.ctx,
            &comps,
            &[k as i64, d as i64],
            dtype,
            None,
        ));
        self.mean = Some(tensor_from_f64(&self.ctx, &mean, &[d as i64], dtype, None));
        self.explained_variance = Some(tensor_from_f64(&self.ctx, &ev, &[k as i64], dtype, None));
        self.n_components = k;
        self.n_features = d;
        self.total_variance = meta_f64(state, "classic.total_variance").unwrap_or(0.0);
        self.dtype = dtype;
        Ok(())
    }
}
