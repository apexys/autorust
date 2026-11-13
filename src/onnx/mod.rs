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
//! from [`OnnxModel::ctx`]. The first run for an input shape captures a replay
//! plan and later runs replay it; the returned tensors alias the plan's buffers,
//! so they are valid until the next [`OnnxModel::run`].
//! [`OnnxModel::set_plan_cache`] disables the plan for debugging.

pub mod pb;

mod ops;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::engine::jit::Jit;
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
    /// Captured replay plan for the current input signature. `None` until the
    /// first run (and after a shape change, which recaptures).
    plan: Option<OnnxPlan>,
    /// Whether to capture a replay plan at all (tests flip this to compare the
    /// eager path).
    plan_enabled: bool,
}

/// One captured replay plan: the whole graph's LINEARS, the dedicated input
/// buffers they read, and the captured outputs. Replay skips parsing/eval and
/// every `create_linear_with_vars`, which is ~90% of a small ONNX step.
struct OnnxPlan {
    jit: Jit,
    /// Input buffers owned by the plan. Each run copies the caller's input in.
    inputs: Vec<(String, Tensor)>,
    /// `(name, shape, dtype)` the plan was captured for.
    sig: Vec<(String, Vec<i64>, crate::dtype::DTypeId)>,
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
            plan: None,
            plan_enabled: true,
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
    /// read them with [`OnnxModel::session_mut`]:
    ///
    /// ```ignore
    /// let out = model.run(&[("image", image)])?;
    /// let depth = out["depth"].to_list::<f32>(model.session_mut());
    /// ```
    ///
    /// The first run for a given input shape captures a replay plan; later runs
    /// copy the inputs into the plan's buffers and execute the captured LINEARS.
    /// Outputs alias the plan buffers, so they are valid until the next `run`
    /// (same contract as [`crate::engine::jit::Jit`]). A changed input shape
    /// recaptures. [`OnnxModel::set_plan_cache`] disables it.
    pub fn run(&mut self, inputs: &[(&str, Tensor)]) -> Result<HashMap<String, Tensor>> {
        let env_in = self.prepare_inputs(inputs)?;
        let sig = self.signature(&env_in);
        if self.plan_enabled
            && let Some(plan) = self.plan.as_mut()
            && plan.sig == sig
        {
            for (name, dst) in &plan.inputs {
                dst.assign(&mut self.sess, &env_in[name]);
            }
            let outs = {
                let mut c = self.ctx.borrow_mut();
                plan.jit.replay(&mut c, &mut self.sess)
            };
            return Ok(self.outputs.iter().cloned().zip(outs).collect());
        }
        self.plan = None;
        if self.plan_enabled
            && let Some(out) = self.run_captured(env_in.clone(), sig)?
        {
            return Ok(out);
        }
        self.run_eager(env_in)
    }

    /// Enable or disable the captured replay plan (default enabled). Disabling
    /// drops the current plan; the next `run` is eager.
    pub fn set_plan_cache(&mut self, on: bool) {
        self.plan_enabled = on;
        if !on {
            self.plan = None;
        }
    }

    /// Whether a replay plan is currently captured.
    pub fn has_plan(&self) -> bool {
        self.plan.is_some()
    }

    /// Validate the caller's inputs (shape/dtype/ctx) and apply the model's
    /// declared input dtype.
    fn prepare_inputs(&self, inputs: &[(&str, Tensor)]) -> Result<HashMap<String, Tensor>> {
        let mut env = HashMap::new();
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
            env.insert(spec.name.clone(), t);
        }
        Ok(env)
    }

    /// `(name, shape, dtype)` of the prepared inputs, compared on every run.
    fn signature(
        &self,
        env: &HashMap<String, Tensor>,
    ) -> Vec<(String, Vec<i64>, crate::dtype::DTypeId)> {
        self.inputs
            .iter()
            .map(|spec| {
                let t = &env[&spec.name];
                let shape: Vec<i64> = t
                    .shape()
                    .into_vec()
                    .iter()
                    .map(|d| d.clone().as_i64().unwrap_or(-1))
                    .collect();
                (spec.name.clone(), shape, t.dtype())
            })
            .collect()
    }

    /// Build + evaluate the graph over `env` (which already contains the
    /// initializers and any prepared inputs).
    fn run_graph(&mut self, mut env: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
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

    /// One eager pass: no capture, fresh output tensors.
    fn run_eager(&mut self, env_in: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
        // Pin the input storage for the duration of the run: evaluating an
        // unrelated shape/index constant realizes a schedule, and `prune`
        // would otherwise reclaim an input buffer that later nodes still
        // read (visible as zeros / out-of-bounds loads).
        {
            let mut c = self.ctx.borrow_mut();
            for t in env_in.values() {
                t.pin_buffer_ctx(&mut c, &mut self.sess);
            }
        }
        let mut env = self.values.clone();
        env.extend(env_in);
        let out = self.run_graph(env)?;
        let rets: Vec<Tensor> = self.outputs.iter().map(|n| out[n].clone()).collect();
        {
            let mut c = self.ctx.borrow_mut();
            Tensor::realize_many(&mut c, &mut self.sess, rets.iter());
        }
        Ok(self.outputs.iter().cloned().zip(rets).collect())
    }

    /// Run once under capture and adopt the plan. Returns `None` when the graph
    /// realized nothing (a constant-only graph cannot be replayed) so the
    /// caller falls back to the eager path.
    fn run_captured(
        &mut self,
        env_in: HashMap<String, Tensor>,
        sig: Vec<(String, Vec<i64>, crate::dtype::DTypeId)>,
    ) -> Result<Option<HashMap<String, Tensor>>> {
        // Bind each input to a dedicated buffer the plan owns: replay reads
        // these, and `assign` copies a new input in before each replay.
        let mut env = self.values.clone();
        let mut plan_inputs = Vec::new();
        for spec in &self.inputs {
            let src = &env_in[&spec.name];
            let dst = Tensor::zeros_param(
                &self.ctx,
                src.shape().into_vec(),
                Some(src.dtype()),
                self.device.as_deref(),
            );
            dst.assign(&mut self.sess, src);
            plan_inputs.push((spec.name.clone(), dst.clone()));
            env.insert(spec.name.clone(), dst);
        }
        self.sess.begin_capture();
        let out = self.run_graph(env)?;
        let rets: Vec<Tensor> = self.outputs.iter().map(|n| out[n].clone()).collect();
        {
            let mut c = self.ctx.borrow_mut();
            Tensor::realize_many(&mut c, &mut self.sess, rets.iter());
        }
        let planned = self.sess.end_capture();
        if planned.is_empty() {
            return Ok(None);
        }
        let mut jit = Jit::new();
        {
            let mut c = self.ctx.borrow_mut();
            jit.lower_capture(&mut c, &mut self.sess, planned, rets.clone());
        }
        self.plan = Some(OnnxPlan {
            jit,
            inputs: plan_inputs,
            sig,
        });
        Ok(Some(self.outputs.iter().cloned().zip(rets).collect()))
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
