//! CART decision trees and random forests (C2a).
//!
//! Training is host-side and deterministic for a fixed seed. Trees store their
//! nodes as parallel column arrays for persistence; prediction is a host
//! traversal over dense numeric rows.

use super::{
    dtype_from_debug, dtype_label, float_dtype, meta_i64, meta_str_optional, put_meta, read_labels,
    read_matrix, read_target, shape_i64, state_f64, state_i64, tensor_from_f64, tensor_from_i64,
    ClassicError, Persist, Result,
};
use crate::dtype::DTypeId;
use crate::tensor::state::State;
use crate::tensor::{Ctx, Tensor};

/// Whether a tree predicts class labels or a continuous target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TreeTask {
    /// Classification: targets are contiguous integer class indices.
    Classification,
    /// Regression: targets are continuous.
    Regression,
}

impl TreeTask {
    fn as_str(self) -> &'static str {
        match self {
            TreeTask::Classification => "classification",
            TreeTask::Regression => "regression",
        }
    }
    fn parse(s: &str) -> Result<Self> {
        match s {
            "classification" => Ok(TreeTask::Classification),
            "regression" => Ok(TreeTask::Regression),
            other => Err(ClassicError::State(format!("unknown tree task `{other}`"))),
        }
    }
}

/// Split criterion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TreeCriterion {
    /// Gini impurity (classification).
    Gini,
    /// Shannon entropy (classification).
    Entropy,
    /// Mean squared error / variance (regression).
    Mse,
}

impl TreeCriterion {
    fn as_str(self) -> &'static str {
        match self {
            TreeCriterion::Gini => "gini",
            TreeCriterion::Entropy => "entropy",
            TreeCriterion::Mse => "mse",
        }
    }
    fn parse(s: &str) -> Result<Self> {
        match s {
            "gini" => Ok(TreeCriterion::Gini),
            "entropy" => Ok(TreeCriterion::Entropy),
            "mse" => Ok(TreeCriterion::Mse),
            other => Err(ClassicError::State(format!(
                "unknown tree criterion `{other}`"
            ))),
        }
    }
}

#[derive(Clone, Debug)]
struct Node {
    /// Split feature; `-1` for a leaf.
    feature: i64,
    threshold: f64,
    left: i64,
    right: i64,
    /// Class probabilities (classification) or the mean (regression).
    value: Vec<f64>,
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.max(1))
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }
}

#[derive(Clone)]
struct TreeConfig {
    task: TreeTask,
    criterion: TreeCriterion,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features: Option<usize>,
    n_features: usize,
    n_classes: usize,
}

fn classify_impurity(counts: &[f64], n: f64, criterion: TreeCriterion) -> f64 {
    if n <= 0.0 {
        return 0.0;
    }
    match criterion {
        TreeCriterion::Gini => {
            let mut sum = 0.0;
            for &c in counts {
                let p = c / n;
                sum += p * p;
            }
            1.0 - sum
        }
        TreeCriterion::Entropy => {
            let mut h = 0.0;
            for &c in counts {
                let p = c / n;
                if p > 0.0 {
                    h -= p * p.log2();
                }
            }
            h
        }
        TreeCriterion::Mse => f64::INFINITY,
    }
}

fn leaf_value(task: TreeTask, counts: &[f64], sum: f64, n: usize) -> Vec<f64> {
    match task {
        TreeTask::Classification => counts.iter().map(|c| c / n as f64).collect(),
        TreeTask::Regression => vec![sum / n as f64],
    }
}

fn choose_features(cfg: &TreeConfig, rng: &mut Rng) -> Vec<usize> {
    let d = cfg.n_features;
    let k = cfg.max_features.unwrap_or(d).min(d).max(1);
    if k >= d {
        return (0..d).collect();
    }
    let mut all: Vec<usize> = (0..d).collect();
    for i in 0..k {
        let j = i + rng.below(d - i);
        all.swap(i, j);
    }
    let mut out = all[..k].to_vec();
    out.sort_unstable();
    out
}

/// Grow a node and return its index in `nodes`.
fn build_node(
    cfg: &TreeConfig,
    x: &[f64],
    y: &[f64],
    idx: &[usize],
    depth: usize,
    rng: &mut Rng,
    nodes: &mut Vec<Node>,
) -> i64 {
    let n = idx.len();
    let d = cfg.n_features;
    let k = cfg.n_classes;
    let mut counts = vec![0.0; k];
    let mut sum = 0.0;
    let mut sumsq = 0.0;
    for &i in idx {
        match cfg.task {
            TreeTask::Classification => {
                let c = y[i] as usize;
                counts[c] += 1.0;
            }
            TreeTask::Regression => {
                sum += y[i];
                sumsq += y[i] * y[i];
            }
        }
    }
    let node_id = nodes.len() as i64;
    let leaf = Node {
        feature: -1,
        threshold: 0.0,
        left: -1,
        right: -1,
        value: leaf_value(cfg.task, &counts, sum, n),
    };
    nodes.push(leaf);
    let stop = n < 2
        || n < cfg.min_samples_split
        || n < 2 * cfg.min_samples_leaf
        || cfg.max_depth.is_some_and(|md| depth >= md);
    if stop {
        return node_id;
    }
    let parent_imp = match cfg.task {
        TreeTask::Classification => classify_impurity(&counts, n as f64, cfg.criterion),
        TreeTask::Regression => {
            let var = sumsq / n as f64 - (sum / n as f64) * (sum / n as f64);
            var.max(0.0)
        }
    };
    if parent_imp <= 1e-12 {
        return node_id;
    }
    let mut best_gain = 1e-12;
    let mut best: Option<(usize, f64, usize)> = None;
    for feature in choose_features(cfg, rng) {
        let mut order = idx.to_vec();
        order.sort_by(|&a, &b| {
            x[a * d + feature]
                .partial_cmp(&x[b * d + feature])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut left_counts = vec![0.0; k];
        let mut right_counts = counts.clone();
        let mut left_sum = 0.0;
        let mut left_sumsq = 0.0;
        let mut right_sum = sum;
        let mut right_sumsq = sumsq;
        for p in 0..n.saturating_sub(1) {
            let i = order[p];
            match cfg.task {
                TreeTask::Classification => {
                    let c = y[i] as usize;
                    left_counts[c] += 1.0;
                    right_counts[c] -= 1.0;
                }
                TreeTask::Regression => {
                    left_sum += y[i];
                    left_sumsq += y[i] * y[i];
                    right_sum -= y[i];
                    right_sumsq -= y[i] * y[i];
                }
            }
            let left_n = p + 1;
            let right_n = n - left_n;
            if left_n < cfg.min_samples_leaf || right_n < cfg.min_samples_leaf {
                continue;
            }
            let a = x[order[p] * d + feature];
            let b = x[order[p + 1] * d + feature];
            if a.partial_cmp(&b) != Some(std::cmp::Ordering::Less) {
                continue;
            }
            let gain = match cfg.task {
                TreeTask::Classification => {
                    let li = classify_impurity(&left_counts, left_n as f64, cfg.criterion);
                    let ri = classify_impurity(&right_counts, right_n as f64, cfg.criterion);
                    parent_imp * n as f64 - li * left_n as f64 - ri * right_n as f64
                }
                TreeTask::Regression => {
                    let lsse = left_sumsq - left_sum * left_sum / left_n as f64;
                    let rsse = right_sumsq - right_sum * right_sum / right_n as f64;
                    let psse = sumsq - sum * sum / n as f64;
                    psse - lsse - rsse
                }
            };
            if gain > best_gain {
                best_gain = gain;
                best = Some((feature, 0.5 * (a + b), left_n));
            }
        }
    }
    let Some((feature, threshold, left_n)) = best else {
        return node_id;
    };
    // Recompute the split ordering for the chosen feature (deterministic).
    let mut order = idx.to_vec();
    order.sort_by(|&a, &b| {
        x[a * d + feature]
            .partial_cmp(&x[b * d + feature])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let left_idx = &order[..left_n];
    let right_idx = &order[left_n..];
    let left = build_node(cfg, x, y, left_idx, depth + 1, rng, nodes);
    let right = build_node(cfg, x, y, right_idx, depth + 1, rng, nodes);
    nodes[node_id as usize].feature = feature as i64;
    nodes[node_id as usize].threshold = threshold;
    nodes[node_id as usize].left = left;
    nodes[node_id as usize].right = right;
    node_id
}

fn leaf_index(nodes: &[Node], x: &[f64], row: usize, d: usize) -> usize {
    let mut i = 0usize;
    loop {
        let node = &nodes[i];
        if node.feature < 0 {
            return i;
        }
        let f = node.feature as usize;
        i = if x[row * d + f] <= node.threshold {
            node.left as usize
        } else {
            node.right as usize
        };
    }
}

fn validate_criterion(task: TreeTask, criterion: TreeCriterion) -> Result<()> {
    match (task, criterion) {
        (TreeTask::Classification, TreeCriterion::Mse) => Err(ClassicError::Invalid(
            "MSE is not a classification criterion".into(),
        )),
        (TreeTask::Regression, TreeCriterion::Gini | TreeCriterion::Entropy) => Err(
            ClassicError::Invalid("Gini/entropy are not regression criteria".into()),
        ),
        _ => Ok(()),
    }
}

/// A CART decision tree for classification or regression.
pub struct DecisionTree {
    ctx: Ctx,
    task: TreeTask,
    criterion: TreeCriterion,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features: Option<usize>,
    seed: u64,
    nodes: Vec<Node>,
    n_features: usize,
    n_classes: usize,
    dtype: DTypeId,
}

impl DecisionTree {
    /// Create an unfitted classification tree.
    pub fn classifier(ctx: &Ctx) -> Self {
        Self::new(ctx, TreeTask::Classification)
    }

    /// Create an unfitted regression tree.
    pub fn regressor(ctx: &Ctx) -> Self {
        Self::new(ctx, TreeTask::Regression)
    }

    fn new(ctx: &Ctx, task: TreeTask) -> Self {
        DecisionTree {
            ctx: ctx.clone(),
            task,
            criterion: match task {
                TreeTask::Classification => TreeCriterion::Gini,
                TreeTask::Regression => TreeCriterion::Mse,
            },
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            seed: 0,
            nodes: Vec::new(),
            n_features: 0,
            n_classes: 0,
            dtype: crate::dtype::FLOAT32,
        }
    }

    /// Set the split criterion.
    pub fn with_criterion(mut self, criterion: TreeCriterion) -> Result<Self> {
        validate_criterion(self.task, criterion)?;
        self.criterion = criterion;
        Ok(self)
    }

    /// Limit the tree depth (`None` = unlimited).
    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Minimum samples required to consider a split.
    pub fn with_min_samples_split(mut self, n: usize) -> Self {
        self.min_samples_split = n.max(1);
        self
    }

    /// Minimum samples required in each child.
    pub fn with_min_samples_leaf(mut self, n: usize) -> Self {
        self.min_samples_leaf = n.max(1);
        self
    }

    /// Number of features considered at each split (`None` = all).
    pub fn with_max_features(mut self, max_features: Option<usize>) -> Self {
        self.max_features = max_features;
        self
    }

    /// Seed for feature subsampling.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Fit the tree on dense `x` and integer labels or a continuous target.
    pub fn fit(&mut self, x: &Tensor, y: &Tensor) -> Result<()> {
        let (xv, n, d) = read_matrix(x, "DecisionTree.x")?;
        let dtype = float_dtype(x)?;
        let device = x.device();
        let (yv, n_classes) = match self.task {
            TreeTask::Classification => {
                let labels = read_labels(y, n, "DecisionTree.y")?;
                let c = labels.iter().copied().max().unwrap_or(-1) + 1;
                if c < 1 {
                    return Err(ClassicError::Data(
                        "DecisionTree needs at least one class".into(),
                    ));
                }
                if labels.iter().any(|&v| v < 0 || v >= c) {
                    return Err(ClassicError::Data(
                        "DecisionTree labels must be contiguous 0..n_classes-1".into(),
                    ));
                }
                let yv: Vec<f64> = labels.iter().map(|&v| v as f64).collect();
                (yv, c as usize)
            }
            TreeTask::Regression => {
                let (yv, m, _) = read_target(y, n, "DecisionTree.y")?;
                if m != 1 {
                    return Err(ClassicError::Shape(
                        "DecisionTree regression supports one target".into(),
                    ));
                }
                (yv, 0)
            }
        };
        let idx: Vec<usize> = (0..n).collect();
        self.fit_indices(&xv, &yv, &idx, d, n_classes, dtype, device.as_deref())?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn fit_indices(
        &mut self,
        xv: &[f64],
        yv: &[f64],
        idx: &[usize],
        d: usize,
        n_classes: usize,
        dtype: DTypeId,
        _device: Option<&str>,
    ) -> Result<()> {
        if idx.is_empty() || d == 0 {
            return Err(ClassicError::Data("DecisionTree has no samples".into()));
        }
        if let Some(k) = self.max_features.filter(|&k| k == 0 || k > d) {
            return Err(ClassicError::Invalid(format!(
                "max_features must be in 1..={d}, got {k}"
            )));
        }
        if self.task == TreeTask::Classification && n_classes < 1 {
            return Err(ClassicError::Data(
                "DecisionTree classification needs classes".into(),
            ));
        }
        let cfg = TreeConfig {
            task: self.task,
            criterion: self.criterion,
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            max_features: self.max_features,
            n_features: d,
            n_classes,
        };
        let mut rng = Rng::new(self.seed.wrapping_add(0x9E37_79B9_7F4A_7C15));
        let mut nodes = Vec::new();
        build_node(&cfg, xv, yv, idx, 0, &mut rng, &mut nodes);
        self.nodes = nodes;
        self.n_features = d;
        self.n_classes = n_classes;
        self.dtype = dtype;
        Ok(())
    }

    fn proba_host(&self, xv: &[f64], n: usize) -> Vec<f64> {
        let k = self.n_classes;
        let mut out = vec![0.0; n * k];
        for row in 0..n {
            let leaf = leaf_index(&self.nodes, xv, row, self.n_features);
            out[row * k..(row + 1) * k].copy_from_slice(&self.nodes[leaf].value);
        }
        out
    }

    fn regression_host(&self, xv: &[f64], n: usize) -> Vec<f64> {
        (0..n)
            .map(|row| {
                let leaf = leaf_index(&self.nodes, xv, row, self.n_features);
                self.nodes[leaf].value[0]
            })
            .collect()
    }

    /// Predicted class indices or regression values.
    pub fn predict(&self, x: &Tensor) -> Result<Tensor> {
        self.check_fitted()?;
        let (xv, n, d) = read_matrix(x, "DecisionTree.x")?;
        if d != self.n_features {
            return Err(ClassicError::Shape(format!(
                "DecisionTree expects {} features, got {d}",
                self.n_features
            )));
        }
        let device = x.device();
        match self.task {
            TreeTask::Classification => {
                let p = self.proba_host(&xv, n);
                let k = self.n_classes;
                let labels: Vec<i64> = (0..n)
                    .map(|row| {
                        let mut best = 0usize;
                        let mut bv = f64::NEG_INFINITY;
                        for c in 0..k {
                            if p[row * k + c] > bv {
                                bv = p[row * k + c];
                                best = c;
                            }
                        }
                        best as i64
                    })
                    .collect();
                Ok(tensor_from_i64(
                    &self.ctx,
                    &labels,
                    &[n as i64],
                    device.as_deref(),
                ))
            }
            TreeTask::Regression => Ok(tensor_from_f64(
                &self.ctx,
                &self.regression_host(&xv, n),
                &[n as i64],
                self.dtype,
                device.as_deref(),
            )),
        }
    }

    /// Class probabilities `[n, n_classes]` for a classification tree.
    pub fn predict_proba(&self, x: &Tensor) -> Result<Tensor> {
        self.check_fitted()?;
        if self.task != TreeTask::Classification {
            return Err(ClassicError::Invalid(
                "predict_proba is only available for classification trees".into(),
            ));
        }
        let (xv, n, d) = read_matrix(x, "DecisionTree.x")?;
        if d != self.n_features {
            return Err(ClassicError::Shape(format!(
                "DecisionTree expects {} features, got {d}",
                self.n_features
            )));
        }
        let device = x.device();
        Ok(tensor_from_f64(
            &self.ctx,
            &self.proba_host(&xv, n),
            &[n as i64, self.n_classes as i64],
            self.dtype,
            device.as_deref(),
        ))
    }

    fn check_fitted(&self) -> Result<()> {
        if self.nodes.is_empty() {
            Err(ClassicError::NotFitted("DecisionTree".into()))
        } else {
            Ok(())
        }
    }

    /// Number of nodes in the fitted tree.
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Task of this tree.
    pub fn task(&self) -> TreeTask {
        self.task
    }

    #[allow(clippy::type_complexity)]
    fn node_arrays(&self) -> (Vec<i64>, Vec<f64>, Vec<i64>, Vec<i64>, Vec<f64>, usize) {
        let n = self.nodes.len();
        let n_outputs = if self.task == TreeTask::Classification {
            self.n_classes
        } else {
            1
        };
        let mut feature = Vec::with_capacity(n);
        let mut threshold = Vec::with_capacity(n);
        let mut left = Vec::with_capacity(n);
        let mut right = Vec::with_capacity(n);
        let mut value = Vec::with_capacity(n * n_outputs);
        for node in &self.nodes {
            feature.push(node.feature);
            threshold.push(node.threshold);
            left.push(node.left);
            right.push(node.right);
            value.extend_from_slice(&node.value);
        }
        (feature, threshold, left, right, value, n_outputs)
    }

    fn load_nodes_from_state(&mut self, state: &State, prefix: &str) -> Result<()> {
        let feature = state_i64(state, &format!("{prefix}feature"))?;
        let threshold = state_f64(state, &format!("{prefix}threshold"))?;
        let left = state_i64(state, &format!("{prefix}left"))?;
        let right = state_i64(state, &format!("{prefix}right"))?;
        let value = state_f64(state, &format!("{prefix}value"))?;
        let value_shape = state
            .get(&format!("{prefix}value"))
            .map(shape_i64)
            .unwrap_or_default();
        if value_shape.len() != 2 {
            return Err(ClassicError::State(
                "tree value tensor must be [n_nodes, n_outputs]".into(),
            ));
        }
        let n = feature.len();
        if n == 0
            || threshold.len() != n
            || left.len() != n
            || right.len() != n
            || value_shape[0] as usize != n
            || value.len() != n * value_shape[1] as usize
        {
            return Err(ClassicError::State(
                "tree node arrays have inconsistent lengths".into(),
            ));
        }
        let n_outputs = value_shape[1] as usize;
        if self.task == TreeTask::Classification && n_outputs != self.n_classes {
            return Err(ClassicError::State(
                "tree value width does not match n_classes".into(),
            ));
        }
        let mut nodes = Vec::with_capacity(n);
        for i in 0..n {
            if feature[i] < -1 || feature[i] >= self.n_features as i64 {
                return Err(ClassicError::State(format!(
                    "tree node {i} has invalid feature {}",
                    feature[i]
                )));
            }
            if feature[i] >= 0
                && (left[i] < 0 || right[i] < 0 || left[i] as usize >= n || right[i] as usize >= n)
            {
                return Err(ClassicError::State(format!(
                    "tree node {i} has invalid children"
                )));
            }
            if !threshold[i].is_finite() {
                return Err(ClassicError::State(format!(
                    "tree node {i} has a non-finite threshold"
                )));
            }
            nodes.push(Node {
                feature: feature[i],
                threshold: threshold[i],
                left: left[i],
                right: right[i],
                value: value[i * n_outputs..(i + 1) * n_outputs].to_vec(),
            });
        }
        self.nodes = nodes;
        Ok(())
    }

    fn write_meta(&self, state: &mut State, kind: &str) {
        put_meta(state, "classic.kind", kind);
        put_meta(state, "classic.version", 1);
        put_meta(state, "classic.task", self.task.as_str());
        put_meta(state, "classic.criterion", self.criterion.as_str());
        put_meta(state, "classic.n_features", self.n_features);
        put_meta(state, "classic.n_classes", self.n_classes);
        put_meta(
            state,
            "classic.max_depth",
            self.max_depth.map(|v| v as i64).unwrap_or(-1),
        );
        put_meta(state, "classic.min_samples_split", self.min_samples_split);
        put_meta(state, "classic.min_samples_leaf", self.min_samples_leaf);
        put_meta(
            state,
            "classic.max_features",
            self.max_features.map(|v| v as i64).unwrap_or(-1),
        );
        put_meta(state, "classic.seed", self.seed);
        put_meta(state, "classic.dtype", dtype_label(self.dtype));
    }

    fn read_meta(&mut self, state: &State) -> Result<()> {
        self.task =
            TreeTask::parse(meta_str_optional(state, "classic.task").unwrap_or("classification"))?;
        self.criterion =
            TreeCriterion::parse(meta_str_optional(state, "classic.criterion").unwrap_or("gini"))?;
        self.n_features = meta_i64(state, "classic.n_features")? as usize;
        self.n_classes = meta_i64(state, "classic.n_classes")? as usize;
        self.max_depth = match meta_i64(state, "classic.max_depth").unwrap_or(-1) {
            -1 => None,
            v => Some(v as usize),
        };
        self.min_samples_split = meta_i64(state, "classic.min_samples_split").unwrap_or(2) as usize;
        self.min_samples_leaf = meta_i64(state, "classic.min_samples_leaf").unwrap_or(1) as usize;
        self.max_features = match meta_i64(state, "classic.max_features").unwrap_or(-1) {
            -1 => None,
            v => Some(v as usize),
        };
        self.seed = meta_i64(state, "classic.seed").unwrap_or(0) as u64;
        self.dtype =
            dtype_from_debug(meta_str_optional(state, "classic.dtype").unwrap_or("FLOAT32"))?;
        validate_criterion(self.task, self.criterion)?;
        Ok(())
    }
}

impl Persist for DecisionTree {
    fn state(&self) -> Result<State> {
        self.check_fitted()?;
        let (feature, threshold, left, right, value, n_outputs) = self.node_arrays();
        let n = self.nodes.len() as i64;
        let mut s = State::new();
        s.insert("feature", tensor_from_i64(&self.ctx, &feature, &[n], None));
        s.insert(
            "threshold",
            tensor_from_f64(&self.ctx, &threshold, &[n], crate::dtype::FLOAT64, None),
        );
        s.insert("left", tensor_from_i64(&self.ctx, &left, &[n], None));
        s.insert("right", tensor_from_i64(&self.ctx, &right, &[n], None));
        s.insert(
            "value",
            tensor_from_f64(
                &self.ctx,
                &value,
                &[n, n_outputs as i64],
                crate::dtype::FLOAT64,
                None,
            ),
        );
        self.write_meta(&mut s, "decision_tree");
        Ok(s)
    }

    fn load_state(&mut self, state: &State) -> Result<()> {
        if meta_str_optional(state, "classic.kind") != Some("decision_tree") {
            return Err(ClassicError::State("state is not a DecisionTree".into()));
        }
        self.read_meta(state)?;
        self.load_nodes_from_state(state, "")
    }
}

/// A bootstrap-aggregated forest of CART trees.
pub struct RandomForest {
    ctx: Ctx,
    task: TreeTask,
    criterion: TreeCriterion,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features: Option<usize>,
    seed: u64,
    n_trees: usize,
    trees: Vec<DecisionTree>,
    n_features: usize,
    n_classes: usize,
    dtype: DTypeId,
}

impl RandomForest {
    /// Create an unfitted classification forest with `n_trees` trees.
    pub fn classifier(ctx: &Ctx, n_trees: usize) -> Self {
        Self::new(ctx, TreeTask::Classification, n_trees)
    }

    /// Create an unfitted regression forest with `n_trees` trees.
    pub fn regressor(ctx: &Ctx, n_trees: usize) -> Self {
        Self::new(ctx, TreeTask::Regression, n_trees)
    }

    fn new(ctx: &Ctx, task: TreeTask, n_trees: usize) -> Self {
        RandomForest {
            ctx: ctx.clone(),
            task,
            criterion: match task {
                TreeTask::Classification => TreeCriterion::Gini,
                TreeTask::Regression => TreeCriterion::Mse,
            },
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            seed: 0,
            n_trees: n_trees.max(1),
            trees: Vec::new(),
            n_features: 0,
            n_classes: 0,
            dtype: crate::dtype::FLOAT32,
        }
    }

    /// Set the split criterion.
    pub fn with_criterion(mut self, criterion: TreeCriterion) -> Result<Self> {
        validate_criterion(self.task, criterion)?;
        self.criterion = criterion;
        Ok(self)
    }

    /// Limit each tree's depth.
    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Minimum samples required to split a node.
    pub fn with_min_samples_split(mut self, n: usize) -> Self {
        self.min_samples_split = n.max(1);
        self
    }

    /// Minimum samples required in each child.
    pub fn with_min_samples_leaf(mut self, n: usize) -> Self {
        self.min_samples_leaf = n.max(1);
        self
    }

    /// Features considered at each split (`None` = all).
    pub fn with_max_features(mut self, max_features: Option<usize>) -> Self {
        self.max_features = max_features;
        self
    }

    /// Set the bootstrap/feature-sampling seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Fit the forest.
    pub fn fit(&mut self, x: &Tensor, y: &Tensor) -> Result<()> {
        let (xv, n, d) = read_matrix(x, "RandomForest.x")?;
        let dtype = float_dtype(x)?;
        let device = x.device();
        let (yv, n_classes) = match self.task {
            TreeTask::Classification => {
                let labels = read_labels(y, n, "RandomForest.y")?;
                let c = labels.iter().copied().max().unwrap_or(-1) + 1;
                if c < 1 {
                    return Err(ClassicError::Data(
                        "RandomForest needs at least one class".into(),
                    ));
                }
                if labels.iter().any(|&v| v < 0 || v >= c) {
                    return Err(ClassicError::Data(
                        "RandomForest labels must be contiguous 0..n_classes-1".into(),
                    ));
                }
                (
                    labels.iter().map(|&v| v as f64).collect::<Vec<_>>(),
                    c as usize,
                )
            }
            TreeTask::Regression => {
                let (yv, m, _) = read_target(y, n, "RandomForest.y")?;
                if m != 1 {
                    return Err(ClassicError::Shape(
                        "RandomForest regression supports one target".into(),
                    ));
                }
                (yv, 0)
            }
        };
        self.trees.clear();
        let mut rng = Rng::new(self.seed.wrapping_add(0x9E37_79B9_7F4A_7C15));
        for t in 0..self.n_trees {
            let sample: Vec<usize> = (0..n).map(|_| rng.below(n)).collect();
            let mut tree = DecisionTree::new(&self.ctx, self.task);
            tree.criterion = self.criterion;
            tree.max_depth = self.max_depth;
            tree.min_samples_split = self.min_samples_split;
            tree.min_samples_leaf = self.min_samples_leaf;
            tree.max_features = self.max_features;
            tree.seed = self
                .seed
                .wrapping_add((t as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                .wrapping_add(0xD1B5_4A32_D192_ED03);
            tree.fit_indices(&xv, &yv, &sample, d, n_classes, dtype, device.as_deref())?;
            self.trees.push(tree);
        }
        self.n_features = d;
        self.n_classes = n_classes;
        self.dtype = dtype;
        Ok(())
    }

    fn check_fitted(&self) -> Result<()> {
        if self.trees.is_empty() {
            Err(ClassicError::NotFitted("RandomForest".into()))
        } else {
            Ok(())
        }
    }

    /// Predicted class indices or regression values.
    pub fn predict(&self, x: &Tensor) -> Result<Tensor> {
        self.check_fitted()?;
        let (xv, n, d) = read_matrix(x, "RandomForest.x")?;
        if d != self.n_features {
            return Err(ClassicError::Shape(format!(
                "RandomForest expects {} features, got {d}",
                self.n_features
            )));
        }
        let device = x.device();
        match self.task {
            TreeTask::Classification => {
                let k = self.n_classes;
                let mut labels = Vec::with_capacity(n);
                for row in 0..n {
                    let mut votes = vec![0usize; k];
                    for tree in &self.trees {
                        let leaf = leaf_index(&tree.nodes, &xv, row, d);
                        let mut best = 0usize;
                        let mut bv = f64::NEG_INFINITY;
                        for c in 0..k {
                            if tree.nodes[leaf].value[c] > bv {
                                bv = tree.nodes[leaf].value[c];
                                best = c;
                            }
                        }
                        votes[best] += 1;
                    }
                    let mut best = 0usize;
                    for c in 1..k {
                        if votes[c] > votes[best] {
                            best = c;
                        }
                    }
                    labels.push(best as i64);
                }
                Ok(tensor_from_i64(
                    &self.ctx,
                    &labels,
                    &[n as i64],
                    device.as_deref(),
                ))
            }
            TreeTask::Regression => {
                let mut vals = vec![0.0; n];
                for tree in &self.trees {
                    for (row, v) in vals.iter_mut().enumerate() {
                        let leaf = leaf_index(&tree.nodes, &xv, row, d);
                        *v += tree.nodes[leaf].value[0] / self.trees.len() as f64;
                    }
                }
                Ok(tensor_from_f64(
                    &self.ctx,
                    &vals,
                    &[n as i64],
                    self.dtype,
                    device.as_deref(),
                ))
            }
        }
    }

    /// Averaged class probabilities for a classification forest.
    pub fn predict_proba(&self, x: &Tensor) -> Result<Tensor> {
        self.check_fitted()?;
        if self.task != TreeTask::Classification {
            return Err(ClassicError::Invalid(
                "predict_proba is only available for classification forests".into(),
            ));
        }
        let (xv, n, d) = read_matrix(x, "RandomForest.x")?;
        if d != self.n_features {
            return Err(ClassicError::Shape(format!(
                "RandomForest expects {} features, got {d}",
                self.n_features
            )));
        }
        let k = self.n_classes;
        let mut out = vec![0.0; n * k];
        for tree in &self.trees {
            let p = tree.proba_host(&xv, n);
            for i in 0..n * k {
                out[i] += p[i] / self.trees.len() as f64;
            }
        }
        let device = x.device();
        Ok(tensor_from_f64(
            &self.ctx,
            &out,
            &[n as i64, k as i64],
            self.dtype,
            device.as_deref(),
        ))
    }

    /// Number of trees.
    pub fn n_trees(&self) -> usize {
        self.n_trees
    }

    /// Task of this forest.
    pub fn task(&self) -> TreeTask {
        self.task
    }
}

impl Persist for RandomForest {
    fn state(&self) -> Result<State> {
        self.check_fitted()?;
        let mut s = State::new();
        for (i, tree) in self.trees.iter().enumerate() {
            let ts = tree.state()?;
            for (k, v) in ts.tensors {
                s.insert(format!("tree_{i}.{k}"), v);
            }
        }
        put_meta(&mut s, "classic.kind", "random_forest");
        put_meta(&mut s, "classic.version", 1);
        put_meta(&mut s, "classic.task", self.task.as_str());
        put_meta(&mut s, "classic.criterion", self.criterion.as_str());
        put_meta(&mut s, "classic.n_features", self.n_features);
        put_meta(&mut s, "classic.n_classes", self.n_classes);
        put_meta(&mut s, "classic.n_trees", self.n_trees);
        put_meta(
            &mut s,
            "classic.max_depth",
            self.max_depth.map(|v| v as i64).unwrap_or(-1),
        );
        put_meta(&mut s, "classic.min_samples_split", self.min_samples_split);
        put_meta(&mut s, "classic.min_samples_leaf", self.min_samples_leaf);
        put_meta(
            &mut s,
            "classic.max_features",
            self.max_features.map(|v| v as i64).unwrap_or(-1),
        );
        put_meta(&mut s, "classic.seed", self.seed);
        put_meta(&mut s, "classic.dtype", dtype_label(self.dtype));
        Ok(s)
    }

    fn load_state(&mut self, state: &State) -> Result<()> {
        if meta_str_optional(state, "classic.kind") != Some("random_forest") {
            return Err(ClassicError::State("state is not a RandomForest".into()));
        }
        self.task =
            TreeTask::parse(meta_str_optional(state, "classic.task").unwrap_or("classification"))?;
        self.criterion =
            TreeCriterion::parse(meta_str_optional(state, "classic.criterion").unwrap_or("gini"))?;
        self.n_features = meta_i64(state, "classic.n_features")? as usize;
        self.n_classes = meta_i64(state, "classic.n_classes")? as usize;
        self.n_trees = meta_i64(state, "classic.n_trees")? as usize;
        self.max_depth = match meta_i64(state, "classic.max_depth").unwrap_or(-1) {
            -1 => None,
            v => Some(v as usize),
        };
        self.min_samples_split = meta_i64(state, "classic.min_samples_split").unwrap_or(2) as usize;
        self.min_samples_leaf = meta_i64(state, "classic.min_samples_leaf").unwrap_or(1) as usize;
        self.max_features = match meta_i64(state, "classic.max_features").unwrap_or(-1) {
            -1 => None,
            v => Some(v as usize),
        };
        self.seed = meta_i64(state, "classic.seed").unwrap_or(0) as u64;
        self.dtype =
            dtype_from_debug(meta_str_optional(state, "classic.dtype").unwrap_or("FLOAT32"))?;
        validate_criterion(self.task, self.criterion)?;
        self.trees.clear();
        for i in 0..self.n_trees {
            let mut tree = DecisionTree::new(&self.ctx, self.task);
            tree.criterion = self.criterion;
            tree.max_depth = self.max_depth;
            tree.min_samples_split = self.min_samples_split;
            tree.min_samples_leaf = self.min_samples_leaf;
            tree.max_features = self.max_features;
            tree.seed = self.seed;
            tree.n_features = self.n_features;
            tree.n_classes = self.n_classes;
            tree.dtype = self.dtype;
            tree.load_nodes_from_state(state, &format!("tree_{i}."))?;
            self.trees.push(tree);
        }
        Ok(())
    }
}
