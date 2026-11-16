# Classic ML

This document records the design and roadmap for a classical-machine-learning
module alongside the deep-learning `Tensor`/`nn` stack. It records the scope,
current implementation status and roadmap.

**Status (2026-11-16):** **C0, C1 and C2a are implemented** in `src/classic/`
with `tests/classic.rs`; C2b (GBT) and C3 remain planned. The tiers below are
the scope contract, not a promise of additional work.

## Goals

- Offer dense tabular estimators with a consistent `fit` / `predict` /
  `transform` lifecycle.
- Reuse the existing `Tensor` linear algebra and device stack where it genuinely
  helps (GEMM, reductions, distance matrices), with CPU/GPU selectable per model.
- Persist trained models through the existing tensor/safetensors state path
  where possible, and through a compact, documented node-array format for trees.
- Provide CPU-first implementations for algorithms that are not tensor/GEMM
  problems, without pretending they are GPU kernels.
- Keep the core std-only; use `rayon` only where the backend feature is enabled.

## Non-goals (initially)

- Sparse matrices, categorical/missing-value encoders, and feature pipelines.
  Start with dense numeric `Tensor` inputs; add a `Table` abstraction later.
- Approximate nearest-neighbour indexes (HNSW/IVF) and online/streaming learning.
- GPU tree/forest/GBT training. These are host/CPU algorithms first; a GPU
  histogram path is a later, measured optimization.
- Distributed training, model serving, and automatic cross-validation.
- A second deep-learning framework: `nn` remains the neural-network API.

## Placement

```
src/classic/
  mod.rs            # public API, errors, prelude
  linalg.rs         # host f64 solve/QR/SVD/eigh helpers
  linear.rs         # LinearRegression, Ridge, LogisticRegression
  decomposition.rs  # PCA, randomized/truncated SVD
  cluster.rs        # KMeans
  neighbors.rs      # exact KNN
  svm.rs            # LinearSVC/SVR (kernel SVM later)
  tree.rs           # DecisionTree, RandomForest, GBT
  preprocess.rs     # StandardScaler, maybe one-hot
  metrics.rs        # accuracy, MSE, R², silhouette, confusion matrix
  state.rs          # model save/load helpers
```

The module belongs in the core crate: it uses `Ctx`/`Tensor` and the existing
device matrix, and classic models should be usable from the same binary as a
neural model. No new mandatory dependency. If the code becomes large, a
default-on `classic` feature can isolate it later; start without one.

The API is **not** a replacement for `nn`. Linear/logistic/PCA models can
implement `Module`/`named_params` so they work with `state` and `freeze`, but
trees/kNN are host-loop models and should not be forced through the neural
module trait.

## Data model and API

Start with dense, row-major tensors:

- `x`: `[n_samples, n_features]`, f32 or f64.
- `y` regression: `[n_samples]` or `[n_samples, n_targets]`.
- `y` classification: integer class indices `[n_samples]` or one-hot/probability
  matrices where the estimator documents it.

Validation returns `Result` (shape, finite values, label range, sample count);
tensor misuse continues to panic with `TGERR:` per the project convention.

The C1 estimators expose inherent `fit`/`predict`/`transform` methods. A
common trait layer is a C3 refinement; the intended shape is:

```rust
pub trait Fit<X, Y> {
    fn fit(&mut self, x: &Tensor, y: &Y) -> Result<()>;
}

pub trait Predict {
    fn predict(&self, x: &Tensor) -> Tensor;
}

pub trait Transform {
    fn transform(&self, x: &Tensor) -> Tensor;
}
```

A common `Result` error type should cover shape mismatch, empty data,
non-finite values, insufficient samples, and unsupported hyperparameters.
Pipelines (`StandardScaler -> PCA -> LogisticRegression`), train/test split and
cross-validation are C3 conveniences, not prerequisites.

## Algorithm tiers

The split is by **algorithmic fit to the tensor/GPU stack**, not by popularity.

| Tier | Contents | Why |
|---|---|---|
| **C0** | `linalg`, `metrics`, `StandardScaler`, model state | Shared numerical foundation and persistence |
| **C1** | Linear/Ridge, Logistic, PCA, KMeans, exact KNN, LinearSVC/SVR | GEMM/reduction/distances; strong GPU candidates |
| **C2** | DecisionTree, RandomForest, GBT | Host/CPU-first; branchy split search and histograms |
| **C3** | Kernel SVM/SMO, randomized PCA/SVD, approximate NN, sparse/categorical input, pipelines/CV | Larger numerical/algorithmic projects |

### Implementation status

| Tier | Status |
|---|---|
| **C0** | ✅ `linalg` (`solve`/Cholesky/QR/SVD/symmetric eigh/least squares), metrics, `StandardScaler`, safetensors `Persist` state |
| **C1** | ✅ `LinearRegression` (OLS/ridge), `LogisticRegression`, `PCA`, `KMeans`, exact `KNNClassifier`, `LinearSVC`, `LinearSVR` |
| **C2a** | ✅ `DecisionTree` and `RandomForest` (classification/regression, CART, seeds, safetensors node arrays) |
| **C2b** | ⬜ GBT |
| **C3** | ⬜ kernel SVM, randomized SVD, approximate NN, sparse/categorical input, pipelines/CV |

C1 training runs on the host in f64 for numerical stability; predictions build
ordinary `Tensor` graphs (matmul/distance/top-k) and therefore use the selected
backend. Persistence uses `tensor::state`/safetensors; parameters are detached
from the model's live tensors before serialization so saving does not bind them
to a temporary `Session`.

### C0 — foundation

- Dense host f64 linear algebra: Cholesky/QR/SVD/symmetric eigendecomposition,
  linear solve and least squares. These do **not** exist in the `Tensor` core
  today and are required for stable closed-form linear regression, PCA and some
  SVM paths.
- Metrics: accuracy, precision/recall/F1, confusion matrix, MSE/RMSE/MAE, R²,
  silhouette (for k-means diagnostics).
- `StandardScaler` and simple preprocessing.
- Model state helpers: tensor parameters plus metadata.

Host linalg should be used for small dense matrices (`d` up to a few thousand).
For large `d`, prefer GPU matmul plus an iterative/randomized solver rather than
forming an enormous decomposition on the host.

### C1 — tensor/GEMM-friendly estimators

- **Linear regression / Ridge:** closed-form normal equations with a stable
  solve (prefer QR/SVD over explicit inverse), or iterative methods for large
  `n`/`d`. GPU contributes `XᵀX`, `Xᵀy`, residual norms.
- **Logistic regression:** IRLS/Newton for small `d`, SGD/Adam from `optim` for
  large data; binary and multinomial. GPU contributes `XW`, gradients and
  reductions.
- **PCA:** covariance `XᵀX` on the device, eigendecomposition on the host for
  small `d`, randomized/truncated SVD later for large `d`. Projection is a
  matmul.
- **KMeans:** distance/matrix assignment, one-hot centroid update, inertia.
  k-means++ initialization (seeded local RNG) and empty-cluster handling are
  required. Assignment can stay on-device; convergence needs a scalar readback
  per iteration.
- **Exact KNN:** brute-force squared distances
  `x² + y² - 2xy` with a clamp, then `topk`/`argmin`. No training; persistence
  stores the training matrix and metadata. Chunk large pairwise matrices.
- **Linear SVM/SVR:** hinge/epsilon-insensitive loss with SGD or coordinate
  descent; parameters are a weight/bias tensor. Kernel SVM is C3.

### C2a — CPU-first trees (implemented)

- **DecisionTree:** classification (Gini/entropy) and regression (MSE) CART via
  exact sorted split search; `max_depth`, `min_samples_split`,
  `min_samples_leaf`, `max_features` and a deterministic feature-subsampling
  seed. Exposure: `DecisionTree::classifier`/`regressor`.
- **RandomForest:** bootstrap aggregation with per-tree seeds and averaged
  probabilities/regression values; exact host inference. Exposure:
  `RandomForest::classifier`/`regressor`.
- Both persist node arrays (`feature`, `threshold`, `left`, `right`, `value`)
  as safetensors tensors. `freeze` is not applicable to host traversal.

### C2b — GBT (planned)

- Sequential boosting with histogram-based split search; gradients can use
  `Tensor`/GPU, but tree construction is host/CPU.
- Inference is branchy. Batch it over samples on the host; a GPU traversal is a
  later, measured optimization. Store nodes as column arrays
  (`feature`, `threshold`, `left`, `right`, `value`, `leaf`) plus metadata.
- Reproducibility requires per-estimator seeds, not the global tensor RNG.

### C3 — advanced

- Kernel SVM (RBF/poly) with SMO or another QP solver; the kernel matrix can be
  built on the GPU, but the solver is host-side and O(n²) memory.
- Randomized/truncated PCA/SVD for large `d`.
- Approximate nearest-neighbour indexes.
- Sparse/categorical input and a `Table` abstraction.
- Pipelines, train/test split, cross-validation, grid search.
- Online/streaming estimators if a concrete need appears.

## Where GPU actually helps

| Workload | GPU fit | Notes |
|---|---|---|
| `XW`, `XᵀX`, `Xᵀy`, gradients | High | Large `n`/`d`; small tabular batches lose to transfer overhead |
| Pairwise distances (KNN/k-means) | High | O(n²) memory; chunk |
| PCA covariance + projection | High for products; host eigen | d small/medium |
| Linear SVM SGD | High | Weight updates and data passes |
| Kernel matrix construction | High | Solver remains host |
| Tree split search / histograms | Low-medium | CPU/rayon first |
| Kernel QP/SMO | Low | Host solver; GPU only for kernel evaluation |
| Tree/forest inference | Low | Branchy; host batch traversal |

Practical rules:

- Default to f32 on GPU; f64 on CPU for accuracy-sensitive solves. Consumer GPUs
  are slow at f64.
- Let the caller choose the device (or use `DEFAULT_DEVICE`); provide a CPU
  fallback and benchmark both. Do not force GPU for small tabular data.
- Avoid per-iteration readbacks where possible; when an algorithm needs a scalar
  (k-means inertia/convergence, kNN indices), document the sync.
- Cap or chunk pairwise matrices; expose a memory estimate/limit instead of
  silently OOMing.
- Seed k-means/RF/GBT locally for reproducibility; do not disturb the global
  `Tensor` RNG.

## Persistence and freeze

`freeze` is **not** model serialization. It captures a realized `Jit` plan into
CPU Rust kernels; learned weights stay external. For classic models:

- Linear, logistic, PCA, k-means and linear SVM parameters are tensors → use the
  existing `state`/safetensors path with metadata.
- Trees/forests/GBT store node arrays as safetensors tensors plus metadata
  (feature count, class labels, tree offsets, hyperparameters).
- KNN stores the training matrix and distance metric metadata.
- `freeze` can be an optional deployment path for the *inference graph* of
  linear/logistic/PCA/k-means on CPU, but it is not a persistence format and
  does not apply to tree/kNN host loops.
- Model metadata should include a format version, estimator kind, dtype,
  feature count and class labels where relevant.

## Testing and benchmarks

- Small dense problems with an independent host/f64 reference implementation;
  compare parameters and predictions within explicit tolerances.
- Use scikit-learn as a **dev-only oracle** (not a dependency) for meaningful
  algorithm checks in the existing bench/Python environment.
- Gate convergence and numerical stability explicitly (ill-conditioned linear
  systems, singular covariance, empty k-means clusters, duplicate feature
  values in trees).
- Persistence round-trip: fit → save → load → identical predictions.
- CPU vs GPU benchmarks on named hardware and stated `(n_samples, n_features,
  classes, k)`; only keep a GPU path if it wins at the intended sizes.
- Determinism tests with per-estimator seeds.

## Risks and open questions

- **Input model:** raw `Tensor` versus a `Table`/`Dataset` with column names and
  categorical metadata. Recommendation: start with tensors, add `Table` in C3.
- **Trait shape:** `Fit<X, Y>` generics can become awkward for classification
  versus regression. Recommendation: start with per-estimator methods and add
  traits once the common surface is clear.
- **Linear algebra:** how much to implement in-tree (Cholesky/QR/Jacobi) versus
  iterative GPU methods. Recommendation: small f64 host solvers first.
- **f32 versus f64 defaults:** GPU speed versus numerical stability.
- **Sparse/categorical data:** a major usability need for tabular ML, but a
  separate data model and encoding project.
- **Tree persistence format:** safetensors arrays + metadata is the leading
  option; a custom binary format is a fallback.
- **Feature gate:** keep `classic` default-on until compile size is measured; a
  `classic` feature is the fallback.
- **GPU tree training:** defer until a measured workload shows the CPU/rayon
  path is insufficient.
- **Model selection:** cross-validation, scoring and pipelines are C3; users can
  write their own loops initially.

## Security and robustness

- Validate all shapes and finite values; reject NaN/inf before fitting.
- Bound pairwise matrix sizes and decomposition dimensions; report a clear error
  instead of OOMing or spending unbounded time.
- No `unsafe`; no new dependencies.
- Treat persisted model files as untrusted input: validate dimensions, node
  indices, class counts and metadata before loading.
