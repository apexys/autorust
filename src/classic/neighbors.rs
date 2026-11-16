//! Exact k-nearest-neighbour classification.

use super::{
    dtype_from_debug, float_dtype, meta_i64, meta_str_optional, put_meta, read_labels, read_matrix,
    require_fitted, shape_i64, state_f64, state_i64, tensor_from_f64, tensor_from_i64,
    ClassicError, Persist, Result,
};
use crate::dtype::DTypeId;
use crate::tensor::ix::Ix;
use crate::tensor::state::State;
use crate::tensor::{Ctx, Tensor};

/// Exact brute-force k-nearest-neighbour classifier.
///
/// Prediction builds the squared-distance matrix with Tensor ops, takes the
/// `k` smallest distances, and majority-votes the stored training labels. Labels
/// must be contiguous `0..n_classes-1`.
pub struct KNNClassifier {
    ctx: Ctx,
    x_train: Option<Tensor>,
    y_train: Option<Tensor>,
    k: usize,
    n_classes: usize,
    n_features: usize,
    dtype: DTypeId,
}

impl KNNClassifier {
    /// Create an unfitted classifier with `k` neighbours.
    pub fn new(ctx: &Ctx, k: usize) -> Self {
        KNNClassifier {
            ctx: ctx.clone(),
            x_train: None,
            y_train: None,
            k,
            n_classes: 0,
            n_features: 0,
            dtype: crate::dtype::FLOAT32,
        }
    }

    /// Store the training set.
    pub fn fit(&mut self, x: &Tensor, y: &Tensor) -> Result<()> {
        let (_, n, d) = read_matrix(x, "KNNClassifier.x")?;
        let labels = read_labels(y, n, "KNNClassifier.y")?;
        if self.k == 0 || self.k > n {
            return Err(ClassicError::Invalid(format!(
                "KNNClassifier k must be in 1..={n}, got {}",
                self.k
            )));
        }
        let n_classes = labels.iter().copied().max().unwrap_or(-1) + 1;
        if n_classes < 1 {
            return Err(ClassicError::Data(
                "KNNClassifier needs at least one class".into(),
            ));
        }
        if labels.iter().any(|&v| v < 0 || v >= n_classes) {
            return Err(ClassicError::Data(
                "KNNClassifier labels must be contiguous 0..n_classes-1".into(),
            ));
        }
        let dtype = float_dtype(x)?;
        let device = x.device();
        self.x_train = Some(super::detach_tensor(x)?);
        self.y_train = Some(tensor_from_i64(
            &self.ctx,
            &labels,
            &[n as i64],
            device.as_deref(),
        ));
        self.n_features = d;
        self.n_classes = n_classes as usize;
        self.dtype = dtype;
        Ok(())
    }

    /// Per-row class votes as `[n, n_classes]` normalized by `k`.
    pub fn predict_proba(&self, x: &Tensor) -> Result<Tensor> {
        let xt = require_fitted(&self.x_train, "KNNClassifier")?;
        let yt = require_fitted(&self.y_train, "KNNClassifier")?;
        let sh = shape_i64(x);
        if sh.len() != 2 || sh[1] as usize != self.n_features {
            return Err(ClassicError::Shape(format!(
                "KNNClassifier expects [n, {}], got {sh:?}",
                self.n_features
            )));
        }
        let x = if x.dtype() != xt.dtype() {
            x.cast(xt.dtype())
        } else {
            x.clone()
        };
        let x2 = x.square().sum(Some(&[1]), true, None);
        let t2 = xt.square().sum(Some(&[1]), false, None).unsqueeze(0);
        let dot = x.matmul(xt.transpose(0, 1));
        let dist = x2.add(t2).sub(dot.mul(2.0));
        let (_, idx) = dist.topk(self.k as i64, 1, false);
        let votes = yt.getitem(&[Ix::Tensor(idx)]);
        let onehot = votes
            .unsqueeze(-1)
            .one_hot_along_dim(self.n_classes as i64, -1);
        let counts = onehot
            .cast(crate::dtype::FLOAT32)
            .sum(Some(&[1]), false, None);
        Ok(counts.mul(1.0 / self.k as f64))
    }

    /// Predicted class indices `[n]` (int64).
    pub fn predict(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self
            .predict_proba(x)?
            .argmax(Some(1), false)
            .cast(crate::dtype::INT64))
    }

    /// Number of classes seen during fitting.
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }

    /// Number of neighbours.
    pub fn k(&self) -> usize {
        self.k
    }
}

impl Persist for KNNClassifier {
    fn state(&self) -> Result<State> {
        let xt = require_fitted(&self.x_train, "KNNClassifier")?;
        let yt = require_fitted(&self.y_train, "KNNClassifier")?;
        let mut s = State::new();
        s.insert("x_train", super::detach_tensor(xt)?);
        s.insert("y_train", super::detach_tensor(yt)?);
        put_meta(&mut s, "classic.kind", "knn_classifier");
        put_meta(&mut s, "classic.version", 1);
        put_meta(&mut s, "classic.k", self.k);
        put_meta(&mut s, "classic.n_classes", self.n_classes);
        put_meta(&mut s, "classic.n_features", self.n_features);
        put_meta(&mut s, "classic.dtype", super::dtype_label(self.dtype));
        Ok(s)
    }

    fn load_state(&mut self, state: &State) -> Result<()> {
        if meta_str_optional(state, "classic.kind") != Some("knn_classifier") {
            return Err(ClassicError::State("state is not a KNNClassifier".into()));
        }
        let k = meta_i64(state, "classic.k")? as usize;
        let c = meta_i64(state, "classic.n_classes")? as usize;
        let d = meta_i64(state, "classic.n_features")? as usize;
        let dtype =
            dtype_from_debug(meta_str_optional(state, "classic.dtype").unwrap_or("FLOAT32"))?;
        let xtv = state_f64(state, "x_train")?;
        let ytv = state_i64(state, "y_train")?;
        if xtv.len() % d != 0 || xtv.len() / d != ytv.len() {
            return Err(ClassicError::State(
                "KNNClassifier state has inconsistent dimensions".into(),
            ));
        }
        let n = ytv.len() as i64;
        self.x_train = Some(tensor_from_f64(
            &self.ctx,
            &xtv,
            &[n, d as i64],
            dtype,
            None,
        ));
        self.y_train = Some(tensor_from_i64(&self.ctx, &ytv, &[n], None));
        self.k = k;
        self.n_classes = c;
        self.n_features = d;
        self.dtype = dtype;
        Ok(())
    }
}
