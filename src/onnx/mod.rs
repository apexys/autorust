//! ONNX importer (`tinygrad.nn.onnx` equivalent).
//!
//! Parses an ONNX model with a std-only protobuf reader and executes it with the
//! regular [`Tensor`] front-end. The implementation targets the **T0/T1**
//! surface in `ONNX-Compat.md`: static shapes, transformer/CNN inference graphs,
//! external data, and the common elementwise/movement/reduction/nn ops.
//!
//! ```no_run
//! use autorust::onnx::OnnxModel;
//!
//! let mut model = OnnxModel::load("model.onnx")?;
//! let image = autorust::tensor::Tensor::zeros(model.ctx(), [1, 3, 518, 518]);
//! let out = model.run(&[("image", image)])?;
//! # Ok::<(), autorust::onnx::OnnxError>(())
//! ```
//!
//! The runner owns a [`Ctx`] and a [`Session`]. Inputs must be tensors built
//! from [`OnnxModel::ctx`]. The returned tensors stay valid after the model is
//! dropped because the arena is reference-counted.

pub mod pb;

mod ops;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::engine::session::Session;
use crate::tensor::{Ctx, Tensor};
use crate::uop::graph::UopId;

pub use pb::{Attribute, Dim, GraphProto, ModelProto, Node, OnnxError, TensorProto, ValueInfo};

pub(crate) type Result<T> = std::result::Result<T, OnnxError>;

/// An ONNX model ready to run.
pub struct OnnxModel {
    ctx: Ctx,
    sess: Session,
    device: Option<String>,
    origin: Option<PathBuf>,
    /// Initializers, materialized at load time.
    values: HashMap<String, Tensor>,
    nodes: Vec<Node>,
    inputs: Vec<ValueInfo>,
    outputs: Vec<String>,
    opsets: HashMap<String, i64>,
    /// Host values of small integer shape tensors, keyed by interned UOp id.
    host_i64: HashMap<UopId, Vec<i64>>,
    host_f64: HashMap<UopId, Vec<f64>>,
}

impl OnnxModel {
    /// Load a model from a file on the default device.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::load_with_device(path, None)
    }

    /// Load a model, placing initializers on `device` (`None` = default).
    pub fn load_with_device(path: impl AsRef<Path>, device: Option<&str>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .map_err(|e| OnnxError::Io(format!("read {}: {e}", path.display())))?;
        let model = pb::parse_model(&bytes)?;
        drop(bytes);
        Self::from_model(model, Some(path.to_path_buf()), device)
    }

    /// Build a runner from an already-parsed model (used by tests and subgraphs).
    pub fn from_bytes(bytes: &[u8], device: Option<&str>) -> Result<Self> {
        let model = pb::parse_model(bytes)?;
        Self::from_model(model, None, device)
    }

    pub(crate) fn from_model(
        model: ModelProto,
        origin: Option<PathBuf>,
        device: Option<&str>,
    ) -> Result<Self> {
        let ctx = Ctx::new();
        let graph = model.graph;
        let mut values = HashMap::new();
        for tp in &graph.initializers {
            if tp.name.is_empty() {
                return Err(OnnxError::Parse("initializer without a name".into()));
            }
            let t = tensor_from_proto(&ctx, tp, origin.as_deref(), device)?;
            values.insert(tp.name.clone(), t);
        }
        let inputs: Vec<ValueInfo> = graph
            .inputs
            .into_iter()
            .filter(|i| !values.contains_key(&i.name))
            .collect();
        let mut opsets = HashMap::new();
        let mut saw_onnx = false;
        for (domain, version) in &model.opsets {
            let d = normalize_domain(domain);
            if d == "ai.onnx" {
                saw_onnx = true;
            }
            opsets.insert(d, *version);
        }
        if !saw_onnx {
            opsets.insert("ai.onnx".to_string(), 1);
        }
        Ok(OnnxModel {
            ctx,
            sess: Session::new(),
            device: device.map(str::to_string),
            origin,
            values,
            nodes: graph.nodes,
            inputs,
            outputs: graph.outputs.into_iter().map(|o| o.name).collect(),
            opsets,
            host_i64: HashMap::new(),
            host_f64: HashMap::new(),
        })
    }

    /// The arena that owns this model's tensors.
    pub fn ctx(&self) -> &Ctx {
        &self.ctx
    }

    /// The session used for realize/read. Exposed for reading returned tensors.
    pub fn session(&self) -> &Session {
        &self.sess
    }

    /// Mutable access to the session (needed by [`Tensor::to_list`]).
    pub fn session_mut(&mut self) -> &mut Session {
        &mut self.sess
    }

    /// Declared graph inputs (initializers excluded).
    pub fn inputs(&self) -> &[ValueInfo] {
        &self.inputs
    }

    /// Declared graph output names.
    pub fn outputs(&self) -> &[String] {
        &self.outputs
    }

    /// Number of graph nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Run the graph. Inputs must be built from [`OnnxModel::ctx`].
    ///
    /// Returns the realized values of the declared graph outputs. The caller can
    /// read them with `model.session_mut()`:
    ///
    /// ```ignore
    /// let out = model.run(&[("image", image)])?;
    /// let depth = out["depth"].to_list::<f32>(model.session_mut());
    /// ```
    pub fn run(&mut self, inputs: &[(&str, Tensor)]) -> Result<HashMap<String, Tensor>> {
        let mut env = self.values.clone();
        for spec in &self.inputs {
            let Some((_, t)) = inputs.iter().find(|(n, _)| *n == spec.name) else {
                return Err(OnnxError::Runtime(format!(
                    "missing input `{}` (expected shape {:?})",
                    spec.name, spec.dims
                )));
            };
            let mut t = t.clone();
            if !self.ctx.same(t.ctx()) {
                return Err(OnnxError::Runtime(format!(
                    "input `{}` belongs to a different Ctx; build it with model.ctx()",
                    spec.name
                )));
            }
            let shape = t.shape().into_vec();
            if !spec.dims.is_empty() && spec.dims.len() != shape.len() {
                return Err(OnnxError::Shape(format!(
                    "input `{}` has rank {}, expected {}",
                    spec.name,
                    shape.len(),
                    spec.dims.len()
                )));
            }
            for (i, (d, s)) in spec.dims.iter().zip(shape.iter()).enumerate() {
                if let Dim::Value(v) = d {
                    let got = s.clone().as_i64().unwrap_or(-1);
                    if got != *v {
                        return Err(OnnxError::Shape(format!(
                            "input `{}` dim {i} is {got}, expected {v}",
                            spec.name
                        )));
                    }
                }
            }
            if let Some(want) = spec.dtype
                && t.dtype() != want
            {
                t = t.cast(want);
            }
            // Pin the input storage for the duration of the run: evaluating an
            // unrelated shape/index constant realizes a schedule, and `prune`
            // would otherwise reclaim an input buffer that later nodes still
            // read (visible as zeros / out-of-bounds loads).
            {
                let mut c = self.ctx.borrow_mut();
                t.pin_buffer_ctx(&mut c, &mut self.sess);
            }
            env.insert(spec.name.clone(), t);
        }

        for i in 0..self.nodes.len() {
            let node = self.nodes[i].clone();
            let ins: Vec<Option<Tensor>> = node
                .inputs
                .iter()
                .map(|n| {
                    if n.is_empty() {
                        Ok(None)
                    } else {
                        env.get(n).cloned().map(Some).ok_or_else(|| {
                            OnnxError::Runtime(format!(
                                "node {} (`{}`) references undefined `{n}`",
                                node.name, node.op_type
                            ))
                        })
                    }
                })
                .collect::<Result<_>>()?;
            let outs = self.eval(&node, &ins)?;
            for (name, t) in node.outputs.iter().zip(outs) {
                if !name.is_empty() {
                    env.insert(name.clone(), t);
                }
            }
        }

        let mut result = HashMap::new();
        for name in &self.outputs {
            let t = env
                .get(name)
                .cloned()
                .ok_or_else(|| OnnxError::Runtime(format!("output `{name}` was never produced")))?;
            result.insert(name.clone(), t);
        }
        Ok(result)
    }

    /// Convenience: run and read every declared output as `Vec<f32>`.
    pub fn run_f32(&mut self, inputs: &[(&str, Tensor)]) -> Result<HashMap<String, Vec<f32>>> {
        let out = self.run(inputs)?;
        let mut result = HashMap::new();
        for (name, t) in out {
            let vals = t.to_list::<f32>(&mut self.sess);
            result.insert(name, vals);
        }
        Ok(result)
    }

    // ---- helpers shared with the op dispatcher ----

    pub(crate) fn device(&self) -> Option<&str> {
        self.device.as_deref()
    }

    pub(crate) fn origin(&self) -> Option<&Path> {
        self.origin.as_deref()
    }

    pub(crate) fn opset(&self, node: &Node) -> i64 {
        self.opsets
            .get(&normalize_domain(&node.domain))
            .copied()
            .unwrap_or(1)
    }

    /// Host `i64` values of a shape/index tensor, cached by interned UOp.
    pub(crate) fn i64s(&mut self, t: &Tensor) -> Result<Vec<i64>> {
        let key = t.uop();
        if let Some(v) = self.host_i64.get(&key) {
            return Ok(v.clone());
        }
        let v: Vec<i64> = t.to_list::<i64>(&mut self.sess);
        self.host_i64.insert(key, v.clone());
        Ok(v)
    }

    /// Host `f64` values of a small tensor, cached by interned UOp.
    pub(crate) fn f64s(&mut self, t: &Tensor) -> Result<Vec<f64>> {
        let key = t.uop();
        if let Some(v) = self.host_f64.get(&key) {
            return Ok(v.clone());
        }
        let v: Vec<f64> = t.to_list::<f64>(&mut self.sess);
        self.host_f64.insert(key, v.clone());
        Ok(v)
    }

    pub(crate) fn i64_scalar(&mut self, t: &Tensor) -> Result<i64> {
        self.i64s(t)?
            .into_iter()
            .next()
            .ok_or_else(|| OnnxError::Shape("expected a scalar constant".into()))
    }

    pub(crate) fn f64_scalar(&mut self, t: &Tensor) -> Result<f64> {
        self.f64s(t)?
            .into_iter()
            .next()
            .ok_or_else(|| OnnxError::Shape("expected a scalar constant".into()))
    }
}

fn normalize_domain(d: &str) -> String {
    if d.is_empty() {
        "ai.onnx".to_string()
    } else {
        d.to_string()
    }
}

/// Materialize a `TensorProto` in `ctx`.
pub(crate) fn tensor_from_proto(
    ctx: &Ctx,
    tp: &TensorProto,
    origin: Option<&Path>,
    device: Option<&str>,
) -> Result<Tensor> {
    let dtype = tp.dtype()?;
    if tp.numel()? == 0 {
        return Err(OnnxError::Unsupported(format!(
            "zero-element tensor `{}` (empty storage is not supported yet)",
            tp.name
        )));
    }
    let bytes = tp.to_bytes(origin)?;
    let t = Tensor::from_bytes(ctx, &bytes, dtype, device);
    let dims = tp.dims.clone();
    Ok(t.reshape(dims))
}
