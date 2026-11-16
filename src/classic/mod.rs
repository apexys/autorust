//! Classical machine learning (C0/C1).
//!
//! This module provides dense tabular estimators that reuse the [`Tensor`]
//! linear-algebra stack for prediction while keeping training algorithms in
//! host `f64` where that is the straightforward, stable choice.
//!
//! The C0 foundation (`linalg`, `metrics`, `preprocess`, persistence) and the
//! C1 estimators (linear/ridge, logistic, PCA, k-means, exact kNN and linear
//! SVM/SVR) are implemented. Trees, forests, GBT, kernel SVM, sparse inputs and
//! pipelines remain future work; see `CLASSIC-ML.md`.

use std::path::Path;

use crate::dtype::DTypeId;
use crate::tensor::state::{self, State};
use crate::tensor::{Ctx, Tensor};

pub mod cluster;
pub mod decomposition;
pub mod linalg;
pub mod linear;
pub mod metrics;
pub mod neighbors;
pub mod preprocess;
pub mod svm;

pub use cluster::KMeans;
pub use decomposition::PCA;
pub use linear::{LinearRegression, LogisticRegression};
pub use neighbors::KNNClassifier;
pub use preprocess::StandardScaler;
pub use svm::{LinearSVC, LinearSVR};

/// A classical-ML error.
#[derive(Clone, Debug)]
pub enum ClassicError {
    /// A tensor/shape or hyperparameter validation failure.
    Shape(String),
    /// Invalid data (empty, non-finite, wrong label range, ...).
    Data(String),
    /// The estimator has not been fitted.
    NotFitted(String),
    /// A singular or ill-conditioned linear system.
    Singular(String),
    /// Missing or malformed persisted state.
    State(String),
    /// I/O failure while saving/loading.
    Io(String),
    /// Unsupported configuration.
    Invalid(String),
}

impl std::fmt::Display for ClassicError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClassicError::Shape(s) => write!(f, "classic shape: {s}"),
            ClassicError::Data(s) => write!(f, "classic data: {s}"),
            ClassicError::NotFitted(s) => write!(f, "classic not fitted: {s}"),
            ClassicError::Singular(s) => write!(f, "classic singular: {s}"),
            ClassicError::State(s) => write!(f, "classic state: {s}"),
            ClassicError::Io(s) => write!(f, "classic io: {s}"),
            ClassicError::Invalid(s) => write!(f, "classic invalid: {s}"),
        }
    }
}

impl std::error::Error for ClassicError {}

/// Result alias for classical-ML operations.
pub type Result<T> = std::result::Result<T, ClassicError>;

/// Persist/load an estimator through the safetensors state format.
pub trait Persist {
    /// The estimator's tensor parameters and metadata.
    fn state(&self) -> Result<State>;

    /// Load parameters/metadata produced by [`Persist::state`].
    fn load_state(&mut self, state: &State) -> Result<()>;

    /// Save the estimator to a safetensors file.
    fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let state = self.state()?;
        state::safe_save(&state, path, Some(&state.metadata)).map_err(ClassicError::Io)
    }
}

/// Load a safetensors state file for a classical estimator.
pub fn load_state_file(path: impl AsRef<Path>) -> Result<State> {
    state::safe_load(path).map_err(ClassicError::Io)
}

pub(crate) fn shape_i64(t: &Tensor) -> Vec<i64> {
    t.shape()
        .as_slice()
        .iter()
        .map(|s| s.clone().as_i64().unwrap_or(0))
        .collect()
}

pub(crate) fn float_dtype(t: &Tensor) -> Result<DTypeId> {
    let dt = t.dtype();
    if dt == crate::dtype::FLOAT32 || dt == crate::dtype::FLOAT64 {
        Ok(dt)
    } else {
        Err(ClassicError::Data(format!(
            "expected a float32/float64 tensor, got {dt:?}"
        )))
    }
}

/// Read a 2-D feature matrix into host f64.
pub(crate) fn read_matrix(t: &Tensor, name: &str) -> Result<(Vec<f64>, usize, usize)> {
    let sh = shape_i64(t);
    if sh.len() != 2 {
        return Err(ClassicError::Shape(format!(
            "{name} must be 2-D [n_samples, n_features], got {sh:?}"
        )));
    }
    let n = sh[0] as usize;
    let d = sh[1] as usize;
    if n == 0 || d == 0 {
        return Err(ClassicError::Data(format!("{name} is empty ({n} x {d})")));
    }
    let vals = detach_tensor(t)?.to_list_fresh::<f64>();
    if vals.len() != n * d {
        return Err(ClassicError::Shape(format!(
            "{name} has {} values, expected {}",
            vals.len(),
            n * d
        )));
    }
    if !vals.iter().all(|v| v.is_finite()) {
        return Err(ClassicError::Data(format!(
            "{name} contains NaN or infinity"
        )));
    }
    Ok((vals, n, d))
}

/// Read a regression target as a host f64 vector/matrix. A 2-D target with
/// one column is accepted as a single-output target.
pub(crate) fn read_target(t: &Tensor, n: usize, name: &str) -> Result<(Vec<f64>, usize, bool)> {
    let sh = shape_i64(t);
    let (vals, single, m) = match sh.len() {
        1 if sh[0] as usize == n => (detach_tensor(t)?.to_list_fresh::<f64>(), true, 1usize),
        2 if sh[0] as usize == n => {
            let m = sh[1] as usize;
            if m == 0 {
                return Err(ClassicError::Data(format!("{name} has no targets")));
            }
            (detach_tensor(t)?.to_list_fresh::<f64>(), m == 1, m)
        }
        _ => {
            return Err(ClassicError::Shape(format!(
                "{name} must have shape [{n}] or [{n}, m], got {sh:?}"
            )));
        }
    };
    if vals.len() != n * m {
        return Err(ClassicError::Shape(format!(
            "{name} has {} values, expected {}",
            vals.len(),
            n * m
        )));
    }
    if !vals.iter().all(|v| v.is_finite()) {
        return Err(ClassicError::Data(format!(
            "{name} contains NaN or infinity"
        )));
    }
    Ok((vals, m, single))
}

/// Read classification labels as i64.
pub(crate) fn read_labels(t: &Tensor, n: usize, name: &str) -> Result<Vec<i64>> {
    if !crate::dtype::is_int(t.dtype()) {
        return Err(ClassicError::Data(format!(
            "{name} must have an integer dtype, got {:?}",
            t.dtype()
        )));
    }
    let sh = shape_i64(t);
    if sh.len() != 1 || sh[0] as usize != n {
        return Err(ClassicError::Shape(format!(
            "{name} must have shape [{n}], got {sh:?}"
        )));
    }
    let vals = detach_tensor(t)?.to_list_fresh::<i64>();
    if vals.len() != n {
        return Err(ClassicError::Shape(format!(
            "{name} has {} values, expected {n}",
            vals.len()
        )));
    }
    Ok(vals)
}

pub(crate) fn tensor_from_f64(
    ctx: &Ctx,
    vals: &[f64],
    shape: &[i64],
    dtype: DTypeId,
    device: Option<&str>,
) -> Tensor {
    Tensor::from_slice(ctx, vals, Some(dtype), device).reshape(shape.to_vec())
}

pub(crate) fn tensor_from_i64(
    ctx: &Ctx,
    vals: &[i64],
    shape: &[i64],
    device: Option<&str>,
) -> Tensor {
    Tensor::from_slice(ctx, vals, Some(crate::dtype::INT64), device).reshape(shape.to_vec())
}

/// Copy a tensor into a fresh `Ctx` without realizing (and therefore without
/// tying) the source tensor to a particular `Session`.
///
/// Fitted parameters are `from_slice` buffers, so their bytes can be copied
/// directly from the node's initial data. A computed tensor falls back to a
/// host read, which is fine for training data but should be avoided for model
/// parameters.
pub(crate) fn detach_tensor(t: &Tensor) -> Result<Tensor> {
    let dtype = t.dtype();
    let shape = shape_i64(t);
    let mut u = t.uop();
    loop {
        match u.node.op {
            crate::uop::ops::Ops::Buffer => {
                if let Some(data) = u.node.initial_data.borrow().as_ref() {
                    let bytes = data.borrow().clone();
                    let fresh = Ctx::new();
                    return Ok(Tensor::from_bytes(&fresh, &bytes, dtype, None).reshape(shape));
                }
                break;
            }
            crate::uop::ops::Ops::Reshape
            | crate::uop::ops::Ops::Contiguous
            | crate::uop::ops::Ops::Shrink
            | crate::uop::ops::Ops::Copy => {
                u = u.node.srcs[0].clone();
            }
            _ => break,
        }
    }
    let fresh = Ctx::new();
    if dtype == crate::dtype::INT64 {
        let vals = t.to_list_fresh::<i64>();
        Ok(Tensor::from_slice(&fresh, &vals, Some(dtype), None).reshape(shape))
    } else {
        let vals = t.to_list_fresh::<f64>();
        Ok(Tensor::from_slice(&fresh, &vals, Some(dtype), None).reshape(shape))
    }
}

pub(crate) fn state_tensor(state: &State, name: &str) -> Result<Tensor> {
    state
        .get(name)
        .cloned()
        .ok_or_else(|| ClassicError::State(format!("missing tensor `{name}`")))
}

pub(crate) fn state_f64(state: &State, name: &str) -> Result<Vec<f64>> {
    Ok(state_tensor(state, name)?.to_list_fresh::<f64>())
}

pub(crate) fn state_i64(state: &State, name: &str) -> Result<Vec<i64>> {
    Ok(state_tensor(state, name)?.to_list_fresh::<i64>())
}

pub(crate) fn meta_i64(state: &State, name: &str) -> Result<i64> {
    state
        .metadata
        .get(name)
        .ok_or_else(|| ClassicError::State(format!("missing metadata `{name}`")))?
        .parse()
        .map_err(|_| ClassicError::State(format!("bad metadata `{name}`")))
}

pub(crate) fn meta_f64(state: &State, name: &str) -> Result<f64> {
    state
        .metadata
        .get(name)
        .ok_or_else(|| ClassicError::State(format!("missing metadata `{name}`")))?
        .parse()
        .map_err(|_| ClassicError::State(format!("bad metadata `{name}`")))
}

pub(crate) fn meta_str_optional<'a>(state: &'a State, name: &str) -> Option<&'a str> {
    state.metadata.get(name).map(|s| s.as_str())
}

pub(crate) fn dtype_label(dt: DTypeId) -> &'static str {
    if dt == crate::dtype::FLOAT64 {
        "FLOAT64"
    } else {
        "FLOAT32"
    }
}

pub(crate) fn dtype_from_debug(name: &str) -> Result<DTypeId> {
    match name {
        "FLOAT32" => Ok(crate::dtype::FLOAT32),
        "FLOAT64" => Ok(crate::dtype::FLOAT64),
        other => Err(ClassicError::State(format!("unsupported dtype {other}"))),
    }
}

pub(crate) fn put_meta(state: &mut State, name: &str, value: impl ToString) {
    state.metadata.insert(name.to_string(), value.to_string());
}

pub(crate) fn require_fitted<'a, T>(value: &'a Option<T>, name: &str) -> Result<&'a T> {
    value
        .as_ref()
        .ok_or_else(|| ClassicError::NotFitted(name.to_string()))
}
