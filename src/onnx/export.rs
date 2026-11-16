//! ONNX exporter (E0-E2).
//!
//! The exporter traces the lazy `Tensor`/UOp graph produced by a closure or an
//! `nn::Module` forward. It emits a standard ONNX graph with inputs,
//! initializers and a focused set of elementwise/movement/reduction/nn
//! primitives. Unsupported UOps fail with the node id and op rather than
//! producing a silently wrong model.

use std::collections::HashMap;
use std::path::Path;

use super::pb::{Attribute, Dim, GraphProto, ModelProto, Node, TensorProto, ValueInfo};
use super::write::{dtype_code, encode_model_with_ir, tensor_proto};
use crate::dtype::{self, ConstVal, DTypeId};
use crate::tensor::nn::Module;
use crate::tensor::{Ctx, Tensor};
use crate::uop::arg::Arg;
use crate::uop::ctx::{ConvMeta, UOpCtx};
use crate::uop::graph::UopId;
use crate::uop::ops::Ops;
use crate::uop::sint::SVal;

/// Export failure with an explicit op/path.
#[derive(Clone, Debug)]
pub enum ExportError {
    /// File write failure.
    Io(String),
    /// A UOp has no ONNX lowering.
    UnsupportedOp {
        /// The op name.
        op: String,
        /// UOp path/id for diagnostics.
        path: String,
    },
    /// A tensor/value cannot be represented (symbolic shape, dtype, ...).
    UnsupportedTensor(String),
    /// Shape/attribute inconsistency.
    Shape(String),
    /// Invalid API use.
    Invalid(String),
}

impl std::fmt::Display for ExportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExportError::Io(s) => write!(f, "onnx export io: {s}"),
            ExportError::UnsupportedOp { op, path } => {
                write!(f, "onnx export unsupported op `{op}` at {path}")
            }
            ExportError::UnsupportedTensor(s) => write!(f, "onnx export tensor: {s}"),
            ExportError::Shape(s) => write!(f, "onnx export shape: {s}"),
            ExportError::Invalid(s) => write!(f, "onnx export invalid: {s}"),
        }
    }
}

impl std::error::Error for ExportError {}

/// Export result alias.
pub type Result<T> = std::result::Result<T, ExportError>;

/// A fixed-shape graph input.
#[derive(Clone, Debug)]
pub struct InputSpec {
    /// Input name.
    pub name: String,
    /// Input element dtype.
    pub dtype: DTypeId,
    /// Concrete placeholder dimensions. A dynamic axis uses `1` as the
    /// tracing placeholder and [`InputSpec::dynamic_dims`] to name it.
    pub shape: Vec<i64>,
    /// `(axis, name)` pairs declared as symbolic ONNX `dim_param`s.
    pub dynamic_dims: Vec<(usize, String)>,
}

impl InputSpec {
    /// Construct a fixed-shape input.
    pub fn new(name: impl Into<String>, dtype: DTypeId, shape: &[i64]) -> Self {
        InputSpec {
            name: name.into(),
            dtype,
            shape: shape.to_vec(),
            dynamic_dims: Vec::new(),
        }
    }

    /// Mark `axis` as a symbolic dimension (the tracing placeholder is 1).
    pub fn with_dynamic_dim(mut self, axis: usize, name: impl Into<String>) -> Self {
        self.dynamic_dims.push((axis, name.into()));
        self
    }
}

/// A graph output name.
#[derive(Clone, Debug)]
pub struct OutputSpec {
    /// Output name.
    pub name: String,
    /// `(axis, name)` pairs declared as symbolic output dimensions.
    pub dynamic_dims: Vec<(usize, String)>,
}

impl OutputSpec {
    /// Construct an output with `name`.
    pub fn new(name: impl Into<String>) -> Self {
        OutputSpec {
            name: name.into(),
            dynamic_dims: Vec::new(),
        }
    }

    /// Mark `axis` as a symbolic output dimension.
    pub fn with_dynamic_dim(mut self, axis: usize, name: impl Into<String>) -> Self {
        self.dynamic_dims.push((axis, name.into()));
        self
    }
}

/// Export options.
#[derive(Clone, Debug)]
pub struct ExportOptions {
    /// Target ONNX opset.
    pub opset: i64,
    /// ONNX IR version.
    pub ir_version: i64,
    /// Producer name.
    pub producer: String,
    /// External-data threshold in bytes; `None` embeds raw initializers.
    pub external_data: Option<u64>,
}

impl Default for ExportOptions {
    fn default() -> Self {
        ExportOptions {
            opset: 17,
            ir_version: 8,
            producer: "autorust".to_string(),
            external_data: None,
        }
    }
}

/// Manual ONNX graph builder (E0).
///
/// This is the low-level escape hatch: callers add inputs, initializers, nodes
/// and outputs directly. `export_fn`/`export_module` are the traced paths.
pub struct GraphBuilder {
    name: String,
    nodes: Vec<Node>,
    initializers: Vec<TensorProto>,
    inputs: Vec<ValueInfo>,
    outputs: Vec<ValueInfo>,
}

impl GraphBuilder {
    /// Create an empty graph.
    pub fn new(name: impl Into<String>) -> Self {
        GraphBuilder {
            name: name.into(),
            nodes: Vec::new(),
            initializers: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// Declare a graph input.
    pub fn input(&mut self, name: impl Into<String>, dtype: DTypeId, dims: &[i64]) -> &mut Self {
        self.inputs.push(ValueInfo {
            name: name.into(),
            dtype: Some(dtype),
            dims: dims.iter().copied().map(Dim::Value).collect(),
            is_optional: false,
            is_sequence: false,
        });
        self
    }

    /// Add an initializer with raw little-endian bytes.
    pub fn initializer(
        &mut self,
        name: impl Into<String>,
        dtype: DTypeId,
        dims: &[i64],
        raw: Vec<u8>,
    ) -> Result<&mut Self> {
        let name = name.into();
        let code = dtype_code_for(dtype)?;
        self.initializers
            .push(tensor_proto(&name, dims.to_vec(), code, raw));
        Ok(self)
    }

    /// Add a node with optional attributes.
    pub fn node(
        &mut self,
        op_type: impl Into<String>,
        inputs: &[&str],
        outputs: &[&str],
        attrs: &[(&str, Attribute)],
    ) -> &mut Self {
        self.nodes.push(Node {
            op_type: op_type.into(),
            domain: String::new(),
            name: format!("node_{}", self.nodes.len()),
            inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
            outputs: outputs.iter().map(|s| (*s).to_string()).collect(),
            attrs: attrs
                .iter()
                .map(|(k, v)| ((*k).to_string(), v.clone()))
                .collect(),
        });
        self
    }

    /// Declare a graph output.
    pub fn output(&mut self, name: impl Into<String>, dtype: DTypeId, dims: &[i64]) -> &mut Self {
        self.outputs.push(ValueInfo {
            name: name.into(),
            dtype: Some(dtype),
            dims: dims.iter().copied().map(Dim::Value).collect(),
            is_optional: false,
            is_sequence: false,
        });
        self
    }

    /// Serialize and write the model.
    pub fn save(&self, path: impl AsRef<Path>, opts: &ExportOptions) -> Result<()> {
        let graph = GraphProto {
            name: self.name.clone(),
            nodes: self.nodes.clone(),
            initializers: self.initializers.clone(),
            inputs: self.inputs.clone(),
            outputs: self.outputs.clone(),
        };
        let model = ModelProto {
            graph,
            opsets: HashMap::from([(String::new(), opts.opset)]),
            producer: opts.producer.clone(),
        };
        let bytes = encode_model_with_ir(&model, opts.ir_version);
        std::fs::write(path, bytes).map_err(|e| ExportError::Io(e.to_string()))?;
        Ok(())
    }
}

/// Export a closure with fixed input shapes.
pub fn export_fn<F>(
    path: impl AsRef<Path>,
    inputs: &[InputSpec],
    outputs: &[OutputSpec],
    f: F,
    opts: &ExportOptions,
) -> Result<()>
where
    F: FnOnce(&[Tensor]) -> Vec<Tensor>,
{
    let ctx = Ctx::new();
    let mut input_map = HashMap::new();
    let mut tensors = Vec::with_capacity(inputs.len());
    for spec in inputs {
        if spec.shape.iter().any(|&d| d < 0) {
            return Err(ExportError::UnsupportedTensor(format!(
                "input `{}` has a negative dimension",
                spec.name
            )));
        }
        let shape: Vec<SVal> = spec
            .shape
            .iter()
            .enumerate()
            .map(|(i, &d)| {
                let d = if spec.dynamic_dims.iter().any(|(a, _)| *a == i) {
                    1
                } else {
                    d
                };
                SVal::I(d as i128)
            })
            .collect();
        let fill = if dtype::is_int(spec.dtype) {
            ConstVal::Int(0)
        } else if dtype::is_float(spec.dtype) {
            ConstVal::Float(0.0)
        } else {
            return Err(ExportError::UnsupportedTensor(format!(
                "input `{}` dtype {:?} is not supported",
                spec.name, spec.dtype
            )));
        };
        let t = Tensor::full_const(&ctx, shape, fill, Some(spec.dtype));
        input_map.insert(t.uop(), spec.name.clone());
        tensors.push(t);
    }
    let outs = f(&tensors);
    write_model(
        path,
        &ctx,
        inputs,
        input_map,
        HashMap::new(),
        &outs,
        outputs,
        opts,
    )
}

/// Export an `nn::Module` with fixed input shapes.
///
/// Parameters and buffers from `named_state()` are materialized once and
/// written as initializers. `forward` receives the module and the placeholder
/// inputs and returns the output tensors.
pub fn export_module<M, F>(
    path: impl AsRef<Path>,
    model: &M,
    inputs: &[InputSpec],
    outputs: &[OutputSpec],
    forward: F,
    opts: &ExportOptions,
) -> Result<()>
where
    M: Module,
    F: FnOnce(&M, &[Tensor]) -> Vec<Tensor>,
{
    // Inputs must share the module's arena; use the first parameter's Ctx.
    let ctx = model
        .named_state()
        .next()
        .map(|(_, t)| t.ctx().clone())
        .unwrap_or_default();
    let mut input_map = HashMap::new();
    let mut tensors = Vec::with_capacity(inputs.len());
    for spec in inputs {
        let shape: Vec<SVal> = spec
            .shape
            .iter()
            .enumerate()
            .map(|(i, &d)| {
                let d = if spec.dynamic_dims.iter().any(|(a, _)| *a == i) {
                    1
                } else {
                    d
                };
                SVal::I(d as i128)
            })
            .collect();
        let fill = if dtype::is_int(spec.dtype) {
            ConstVal::Int(0)
        } else if dtype::is_float(spec.dtype) {
            ConstVal::Float(0.0)
        } else {
            return Err(ExportError::UnsupportedTensor(format!(
                "input `{}` dtype {:?} is not supported",
                spec.name, spec.dtype
            )));
        };
        let t = Tensor::full_const(&ctx, shape, fill, Some(spec.dtype));
        input_map.insert(t.uop(), spec.name.clone());
        tensors.push(t);
    }

    // Materialize parameters/buffers before the forward so the graph sees
    // stable leaves.
    let mut param_map: HashMap<UopId, (String, TensorProto)> = HashMap::new();
    for (name, tensor) in model.named_state() {
        if param_map.values().any(|(v, _)| v == &name) {
            return Err(ExportError::Invalid(format!(
                "duplicate parameter name `{name}`"
            )));
        }
        let bytes = tensor.to_bytes_fresh();
        let dims = tensor_dims(&tensor)?;
        let dt = tensor.dtype();
        let code = dtype_code(dt).ok_or_else(|| {
            ExportError::UnsupportedTensor(format!("parameter `{name}` dtype {dt:?}"))
        })?;
        let init_name = format!("param/{name}");
        param_map.insert(
            tensor.uop(),
            (
                init_name.clone(),
                tensor_proto(&init_name, dims, code, bytes),
            ),
        );
    }
    let outs = forward(model, &tensors);
    write_model(
        path, &ctx, inputs, input_map, param_map, &outs, outputs, opts,
    )
}

fn tensor_dims(t: &Tensor) -> Result<Vec<i64>> {
    t.shape()
        .as_slice()
        .iter()
        .map(|s| {
            s.clone()
                .as_i64()
                .ok_or_else(|| ExportError::UnsupportedTensor("symbolic tensor dimension".into()))
        })
        .collect()
}

fn sval_i64(s: &SVal) -> Result<i64> {
    s.clone()
        .as_i64()
        .ok_or_else(|| ExportError::UnsupportedTensor("symbolic dimension".into()))
}

fn shape_of(ctx: &mut UOpCtx, u: &UopId) -> Result<Vec<i64>> {
    let shape = ctx
        .shape(u)
        .map_err(|e| ExportError::Shape(e.to_string()))?
        .ok_or_else(|| ExportError::Shape(format!("node {} has no shape", u.id())))?;
    shape.iter().map(sval_i64).collect()
}

fn dtype_code_for(dt: DTypeId) -> Result<i32> {
    dtype_code(dt)
        .ok_or_else(|| ExportError::UnsupportedTensor(format!("ONNX has no dtype code for {dt:?}")))
}

fn const_bytes(c: &ConstVal, dt: DTypeId) -> Result<Vec<u8>> {
    Ok(match c {
        ConstVal::Bool(v) => vec![*v as u8],
        ConstVal::Int(v) => {
            let v = *v as i64;
            match dt {
                d if d == dtype::INT8 => vec![v as i8 as u8],
                d if d == dtype::UINT8 => vec![v as u8],
                d if d == dtype::INT16 => (v as i16).to_le_bytes().to_vec(),
                d if d == dtype::UINT16 => (v as u16).to_le_bytes().to_vec(),
                d if d == dtype::INT32 => (v as i32).to_le_bytes().to_vec(),
                d if d == dtype::UINT32 => (v as u32).to_le_bytes().to_vec(),
                d if d == dtype::INT64 => v.to_le_bytes().to_vec(),
                d if d == dtype::UINT64 => (v as u64).to_le_bytes().to_vec(),
                d if d == dtype::FLOAT32 => (v as f32).to_le_bytes().to_vec(),
                d if d == dtype::FLOAT64 => (v as f64).to_le_bytes().to_vec(),
                _ => {
                    return Err(ExportError::UnsupportedTensor(format!(
                        "constant int for dtype {dt:?}"
                    )));
                }
            }
        }
        ConstVal::Float(v) => {
            if dt == dtype::FLOAT64 {
                v.to_le_bytes().to_vec()
            } else if dt == dtype::FLOAT32 {
                (*v as f32).to_le_bytes().to_vec()
            } else {
                return Err(ExportError::UnsupportedTensor(format!(
                    "float constant for dtype {dt:?}"
                )));
            }
        }
        ConstVal::Big(_) => {
            return Err(ExportError::UnsupportedTensor(
                "large integer constant".into(),
            ));
        }
        ConstVal::Invalid => {
            return Err(ExportError::UnsupportedTensor("invalid constant".into()));
        }
    })
}

struct Tracer<'a> {
    ctx: &'a mut UOpCtx,
    names: HashMap<UopId, String>,
    input_names: HashMap<UopId, String>,
    param_inits: HashMap<UopId, (String, TensorProto)>,
    nodes: Vec<Node>,
    initializers: Vec<TensorProto>,
    next_init: usize,
    next_node: usize,
}

impl<'a> Tracer<'a> {
    fn new(
        ctx: &'a mut UOpCtx,
        input_names: HashMap<UopId, String>,
        param_inits: HashMap<UopId, (String, TensorProto)>,
    ) -> Self {
        let mut initializers = Vec::with_capacity(param_inits.len());
        for (_, proto) in param_inits.values() {
            initializers.push(proto.clone());
        }
        Tracer {
            ctx,
            names: HashMap::new(),
            input_names,
            param_inits,
            nodes: Vec::new(),
            initializers,
            next_init: 0,
            next_node: 0,
        }
    }

    fn init_name(&mut self, prefix: &str) -> String {
        let n = self.next_init;
        self.next_init += 1;
        format!("{prefix}_{n}")
    }

    fn node_name(&mut self) -> String {
        let n = self.next_node;
        self.next_node += 1;
        format!("node_{n}")
    }

    fn value(&mut self, id: &UopId) -> String {
        self.names
            .entry(id.clone())
            .or_insert_with(|| format!("t{}", id.id()))
            .clone()
    }

    fn add_initializer(&mut self, proto: TensorProto) -> String {
        let name = proto.name.clone();
        self.initializers.push(proto);
        name
    }

    fn scalar_init(&mut self, dt: DTypeId, value: ConstVal, prefix: &str) -> Result<String> {
        let code = dtype_code_for(dt)?;
        let name = self.init_name(prefix);
        let bytes = const_bytes(&value, dt)?;
        let p = tensor_proto(&name, vec![], code, bytes);
        Ok(self.add_initializer(p))
    }

    fn shape_init(&mut self, shape: &[i64]) -> String {
        let name = self.init_name("shape");
        let mut bytes = Vec::with_capacity(shape.len() * 8);
        for &v in shape {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let p = tensor_proto(&name, vec![shape.len() as i64], 7, bytes);
        self.add_initializer(p)
    }

    fn emit(
        &mut self,
        id: &UopId,
        op_type: &str,
        inputs: Vec<String>,
        attrs: Vec<(&str, Attribute)>,
    ) -> String {
        let out = self.value(id);
        let mut map = HashMap::new();
        for (k, v) in attrs {
            map.insert(k.to_string(), v);
        }
        let node_name = self.node_name();
        self.nodes.push(Node {
            op_type: op_type.to_string(),
            domain: String::new(),
            name: node_name,
            inputs,
            outputs: vec![out.clone()],
            attrs: map,
        });
        out
    }

    fn trace(&mut self, id: &UopId) -> Result<String> {
        if let Some(n) = self.names.get(id) {
            return Ok(n.clone());
        }
        if let Some(n) = self.input_names.get(id) {
            self.names.insert(id.clone(), n.clone());
            return Ok(n.clone());
        }
        if let Some((n, _)) = self.param_inits.get(id) {
            let n = n.clone();
            self.names.insert(id.clone(), n.clone());
            return Ok(n);
        }
        // `conv2d` records its high-level parameters; emit one `Conv` instead
        // of walking the im2col expansion.
        if let Some(meta) = self.ctx.conv_meta(id) {
            return self.emit_conv(id, meta);
        }
        let op = self.ctx.op_of(id);
        let dt = self.ctx.dt_of(id);
        let path = format!("uop#{}", id.id());
        let srcs = self.ctx.srcs(id).to_vec();

        // Pass-through nodes.
        if matches!(
            op,
            Ops::After | Ops::Group | Ops::Noop | Ops::Contiguous | Ops::Copy | Ops::Detach
        ) {
            let n = self.trace(&srcs[0])?;
            self.names.insert(id.clone(), n.clone());
            return Ok(n);
        }

        // Leaf constants and buffers.
        match op {
            Ops::Const => {
                let Arg::Const(c) = self.ctx.arg_of(id).clone() else {
                    return Err(ExportError::UnsupportedOp {
                        op: "CONST without Const arg".into(),
                        path,
                    });
                };
                let sdt = dtype::strong_dtype(dt);
                let code = dtype_code_for(sdt)?;
                let dims = shape_of(self.ctx, id)?;
                let bytes = const_bytes(&c, sdt)?;
                let name = self.init_name("const");
                let p = tensor_proto(&name, dims, code, bytes);
                self.initializers.push(p);
                self.names.insert(id.clone(), name.clone());
                return Ok(name);
            }
            Ops::Buffer | Ops::Param => {
                let data = id
                    .node
                    .initial_data
                    .borrow()
                    .as_ref()
                    .map(|d| d.borrow().clone());
                let Some(bytes) = data else {
                    return Err(ExportError::UnsupportedTensor(format!(
                        "leaf {path} has no exportable data (is it a runtime input or parameter?)"
                    )));
                };
                let code = dtype_code_for(dt)?;
                let dims = shape_of(self.ctx, id)?;
                let name = self.init_name("buffer");
                self.initializers
                    .push(tensor_proto(&name, dims, code, bytes));
                self.names.insert(id.clone(), name.clone());
                return Ok(name);
            }
            _ => {}
        }

        // Shape/movement ops use only src[0]; their target shape is the result
        // shape, so the shape-argument subgraph need not be exported.
        match op {
            Ops::Reshape | Ops::Expand => {
                let x = self.trace(&srcs[0])?;
                let shape = shape_of(self.ctx, id)?;
                let sh = self.shape_init(&shape);
                let name = if op == Ops::Reshape {
                    "Reshape"
                } else {
                    "Expand"
                };
                Ok(self.emit(id, name, vec![x, sh], vec![]))
            }
            Ops::Permute => {
                let x = self.trace(&srcs[0])?;
                let perm = match self.ctx.arg_of(id).clone() {
                    Arg::Seq { items, .. } => items
                        .iter()
                        .map(|a| match a {
                            Arg::Int(v) => Ok(*v),
                            _ => Err(ExportError::UnsupportedOp {
                                op: "PERMUTE arg".into(),
                                path: path.clone(),
                            }),
                        })
                        .collect::<Result<Vec<_>>>()?,
                    _ => {
                        return Err(ExportError::UnsupportedOp {
                            op: "PERMUTE without perm".into(),
                            path,
                        });
                    }
                };
                Ok(self.emit(
                    id,
                    "Transpose",
                    vec![x],
                    vec![("perm", Attribute::Ints(perm))],
                ))
            }
            Ops::Reduce => {
                // `Tensor::dot` lowers to a broadcast multiply plus a
                // permutation and reduction. Recognize that exact shape when
                // the graph is a plain matmul and emit one ONNX `MatMul`;
                // anything that does not match keeps the expanded arithmetic
                // form below (both are valid, the fallback is the reference).
                if let Some(matmul) = self.try_matmul(id, &path)? {
                    return Ok(matmul);
                }
                let x = self.trace(&srcs[0])?;
                let (rop, count) = reduce_arg(self.ctx.arg_of(id), &path)?;
                let axes: Vec<i64> = (0..count as i64).collect();
                let onnx = match rop {
                    Ops::Add => "ReduceSum",
                    Ops::Mul => "ReduceProd",
                    Ops::Max => "ReduceMax",
                    other => {
                        return Err(ExportError::UnsupportedOp {
                            op: format!("REDUCE {other:?}"),
                            path,
                        });
                    }
                };
                if onnx == "ReduceSum" {
                    let ax = self.init_i64("axes", &axes);
                    Ok(self.emit(id, onnx, vec![x, ax], vec![("keepdims", Attribute::Int(0))]))
                } else {
                    Ok(self.emit(
                        id,
                        onnx,
                        vec![x],
                        vec![
                            ("axes", Attribute::Ints(axes)),
                            ("keepdims", Attribute::Int(0)),
                        ],
                    ))
                }
            }
            Ops::Shrink => {
                // `shrink` stores `(start, size)` per dim. ONNX `Slice` wants
                // `[start, start + size)` with unit steps.
                let x = self.trace(&srcs[0])?;
                let rank = shape_of(self.ctx, &srcs[0])?.len();
                let (Some(starts), Some(sizes)) =
                    (self.const_ints(&srcs[1]), self.const_ints(&srcs[2]))
                else {
                    return Err(ExportError::UnsupportedOp {
                        op: "SHRINK with non-constant bounds".into(),
                        path,
                    });
                };
                let (Some(starts), Some(sizes)) = (fit_rank(starts, rank), fit_rank(sizes, rank))
                else {
                    return Err(ExportError::UnsupportedOp {
                        op: "SHRINK bounds do not match the rank".into(),
                        path,
                    });
                };
                let ends: Vec<i64> = starts.iter().zip(&sizes).map(|(s, n)| s + n).collect();
                let axes: Vec<i64> = (0..rank as i64).collect();
                let steps = vec![1i64; rank];
                let st = self.init_i64("starts", &starts);
                let en = self.init_i64("ends", &ends);
                let ax = self.init_i64("axes", &axes);
                let sp = self.init_i64("steps", &steps);
                Ok(self.emit(id, "Slice", vec![x, st, en, ax, sp], vec![]))
            }
            Ops::Pad => {
                // `pad` stores `(start, stop)` where `stop` is the new
                // dimension size (`start + input + after`). ONNX wants
                // `[before dims..., after dims...]`.
                let x = self.trace(&srcs[0])?;
                let in_shape = shape_of(self.ctx, &srcs[0])?;
                let rank = in_shape.len();
                let (Some(lo), Some(stop)) = (self.const_ints(&srcs[1]), self.const_ints(&srcs[2]))
                else {
                    return Err(ExportError::UnsupportedOp {
                        op: "PAD with non-constant bounds".into(),
                        path,
                    });
                };
                let (Some(lo), Some(stop)) = (fit_rank(lo, rank), fit_rank(stop, rank)) else {
                    return Err(ExportError::UnsupportedOp {
                        op: "PAD bounds do not match the rank".into(),
                        path,
                    });
                };
                let mut pads = Vec::with_capacity(rank * 2);
                pads.extend_from_slice(&lo);
                for ((s, l), d) in stop.iter().zip(&lo).zip(&in_shape) {
                    let after = s - d - l;
                    if after < 0 || *l < 0 {
                        return Err(ExportError::UnsupportedOp {
                            op: "PAD with negative padding".into(),
                            path,
                        });
                    }
                    pads.push(after);
                }
                let p = self.init_i64("pads", &pads);
                Ok(self.emit(id, "Pad", vec![x, p], vec![]))
            }
            Ops::Cast => {
                let x = self.trace(&srcs[0])?;
                let to = dtype_code_for(dt)?;
                Ok(self.emit(id, "Cast", vec![x], vec![("to", Attribute::Int(to as i64))]))
            }
            _ => self.trace_elementwise(id, op, dt, &srcs, &path),
        }
    }

    /// Emit one ONNX `Conv` from the parameters recorded by `Tensor::conv2d`.
    fn emit_conv(&mut self, id: &UopId, meta: ConvMeta) -> Result<String> {
        let x = self.trace(&meta.x)?;
        let w = self.trace(&meta.weight)?;
        let mut inputs = vec![x, w];
        if let Some(b) = meta.bias {
            inputs.push(self.trace(&b)?);
        }
        let mut pads = Vec::with_capacity(meta.pads.len() * 2);
        for (before, _) in &meta.pads {
            pads.push(*before);
        }
        for (_, after) in &meta.pads {
            pads.push(*after);
        }
        Ok(self.emit(
            id,
            "Conv",
            inputs,
            vec![
                ("group", Attribute::Int(meta.groups)),
                ("strides", Attribute::Ints(meta.strides)),
                ("dilations", Attribute::Ints(meta.dilations)),
                ("pads", Attribute::Ints(pads)),
            ],
        ))
    }

    fn init_i64(&mut self, prefix: &str, vals: &[i64]) -> String {
        let name = self.init_name(prefix);
        let mut bytes = Vec::with_capacity(vals.len() * 8);
        for &v in vals {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        self.initializers
            .push(tensor_proto(&name, vec![vals.len() as i64], 7, bytes));
        name
    }

    /// Evaluate a small integer subgraph used as a movement bound. `shrink`
    /// and `pad` build their bounds from `Stack`/`Const` nodes; anything more
    /// complex is rejected rather than guessed.
    fn const_ints(&mut self, id: &UopId) -> Option<Vec<i64>> {
        match self.ctx.op_of(id) {
            Ops::Const => match self.ctx.arg_of(id) {
                Arg::Const(crate::dtype::ConstVal::Int(v)) => {
                    i64::try_from(*v).ok().map(|v| vec![v])
                }
                _ => None,
            },
            Ops::Stack => {
                let srcs = self.ctx.srcs(id).to_vec();
                let mut out = Vec::new();
                for s in &srcs {
                    out.extend(self.const_ints(s)?);
                }
                Some(out)
            }
            Ops::Reshape | Ops::Cast => {
                let src = self.ctx.srcs(id)[0].clone();
                self.const_ints(&src)
            }
            Ops::Expand => {
                let src = self.ctx.srcs(id)[0].clone();
                let vals = self.const_ints(&src)?;
                let shape = shape_of(self.ctx, id).ok()?;
                let n: usize = shape.iter().map(|&d| d as usize).product();
                if vals.len() == n {
                    Some(vals)
                } else if vals.len() == 1 {
                    Some(vec![vals[0]; n])
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Match the expanded `dot` graph (`permute(mul(reshape(x), permute(reshape(w))))
    /// -> reduce(add)`) and emit a single `MatMul`.
    ///
    /// The pattern is validated against the operand shapes, so a partial or
    /// unrelated reduction falls back to the expanded form. `dot` always
    /// inserts one broadcast dimension (`x` becomes `[.., N, 1, K]`, `w`
    /// becomes `[.., 1, K, M]`), which is what lets this distinguish an actual
    /// contraction from an elementwise multiply followed by an unrelated sum.
    fn try_matmul(&mut self, id: &UopId, path: &str) -> Result<Option<String>> {
        let (rop, count) = match reduce_arg(self.ctx.arg_of(id), path) {
            Ok(v) => v,
            Err(_) => return Ok(None),
        };
        if rop != Ops::Add || count != 1 {
            return Ok(None);
        }
        let srcs = self.ctx.srcs(id).to_vec();
        let Some(perm_node) = srcs.first() else {
            return Ok(None);
        };
        if self.ctx.op_of(perm_node) != Ops::Permute {
            return Ok(None);
        }
        let Ok(pd) = shape_of(self.ctx, perm_node).map(|s| s.len()) else {
            return Ok(None);
        };
        let Some(outer_perm) = perm_ints(self.ctx.arg_of(perm_node)) else {
            return Ok(None);
        };
        if preduce_permute(&outer_perm, pd).is_none() {
            return Ok(None);
        }
        let mul = self.ctx.srcs(perm_node)[0].clone();
        if self.ctx.op_of(&mul) != Ops::Mul {
            return Ok(None);
        }
        let mul_srcs = self.ctx.srcs(&mul).to_vec();
        if mul_srcs.len() != 2 {
            return Ok(None);
        }
        let (xr, wt) = (&mul_srcs[0], &mul_srcs[1]);
        if self.ctx.op_of(xr) != Ops::Reshape || self.ctx.op_of(wt) != Ops::Permute {
            return Ok(None);
        }
        let wr = self.ctx.srcs(wt)[0].clone();
        if self.ctx.op_of(&wr) != Ops::Reshape {
            return Ok(None);
        }
        let Some(wperm) = perm_ints(self.ctx.arg_of(wt)) else {
            return Ok(None);
        };
        let Ok(xs) = shape_of(self.ctx, xr) else {
            return Ok(None);
        };
        let Ok(ws) = shape_of(self.ctx, &wr) else {
            return Ok(None);
        };
        let nd = xs.len();
        if nd < 3 || ws.len() != nd {
            return Ok(None);
        }
        if preduce_swap_last(&wperm, nd).is_none() {
            return Ok(None);
        }
        // `dot` inserts the broadcast axis at `-2` in x and `-3` in w.
        if xs[nd - 2] != 1 || ws[nd - 3] != 1 {
            return Ok(None);
        }
        let k = xs[nd - 1];
        if ws[nd - 2] != k {
            return Ok(None);
        }
        let m = ws[nd - 1];
        let x = self.ctx.srcs(xr)[0].clone();
        let w = self.ctx.srcs(&wr)[0].clone();
        let mut want_x = xs[..nd - 2].to_vec();
        want_x.push(k);
        if shape_of(self.ctx, &x).ok().as_deref() != Some(want_x.as_slice()) {
            return Ok(None);
        }
        let mut want_w = ws[..nd - 3].to_vec();
        want_w.push(k);
        want_w.push(m);
        if shape_of(self.ctx, &w).ok().as_deref() != Some(want_w.as_slice()) {
            return Ok(None);
        }
        let mut want_out = xs[..nd - 2].to_vec();
        want_out.push(m);
        if shape_of(self.ctx, id).ok().as_deref() != Some(want_out.as_slice()) {
            return Ok(None);
        }
        let a = self.trace(&x)?;
        let b = self.trace(&w)?;
        Ok(Some(self.emit(id, "MatMul", vec![a, b], vec![])))
    }

    fn trace_elementwise(
        &mut self,
        id: &UopId,
        op: Ops,
        dt: DTypeId,
        srcs: &[UopId],
        path: &str,
    ) -> Result<String> {
        let bin = |t: &mut Self, name: &str| -> Result<String> {
            let a = t.trace(&srcs[0])?;
            let b = t.trace(&srcs[1])?;
            Ok(t.emit(id, name, vec![a, b], vec![]))
        };
        let un = |t: &mut Self, name: &str| -> Result<String> {
            let a = t.trace(&srcs[0])?;
            Ok(t.emit(id, name, vec![a], vec![]))
        };
        match op {
            Ops::Add => bin(self, "Add"),
            Ops::Sub => bin(self, "Sub"),
            Ops::Mul => bin(self, "Mul"),
            Ops::Fdiv | Ops::Cdiv => bin(self, "Div"),
            Ops::Pow => bin(self, "Pow"),
            Ops::Max => bin(self, "Max"),
            Ops::Cmod | Ops::Floormod => bin(self, "Mod"),
            Ops::And => bin(self, "And"),
            Ops::Or => bin(self, "Or"),
            Ops::Xor => bin(self, "Xor"),
            Ops::Cmplt => bin(self, "Less"),
            Ops::Cmpne => {
                let a = self.trace(&srcs[0])?;
                let b = self.trace(&srcs[1])?;
                let eq_name = format!("{}_eq", self.value(id));
                let node_name = self.node_name();
                self.nodes.push(Node {
                    op_type: "Equal".into(),
                    domain: String::new(),
                    name: node_name,
                    inputs: vec![a, b],
                    outputs: vec![eq_name.clone()],
                    attrs: HashMap::new(),
                });
                Ok(self.emit(id, "Not", vec![eq_name], vec![]))
            }
            Ops::Cmpeq => bin(self, "Equal"),
            Ops::Neg => un(self, "Neg"),
            Ops::Sqrt => un(self, "Sqrt"),
            Ops::Sin => un(self, "Sin"),
            Ops::Reciprocal => {
                let a = self.trace(&srcs[0])?;
                let one = self.scalar_init(dt, ConstVal::Float(1.0), "one")?;
                Ok(self.emit(id, "Div", vec![one, a], vec![]))
            }
            Ops::Exp2 => {
                let a = self.trace(&srcs[0])?;
                let two = self.scalar_init(dt, ConstVal::Float(2.0), "two")?;
                Ok(self.emit(id, "Pow", vec![two, a], vec![]))
            }
            Ops::Log2 => {
                let a = self.trace(&srcs[0])?;
                let log = format!("{}_log", self.value(id));
                let node_name = self.node_name();
                self.nodes.push(Node {
                    op_type: "Log".into(),
                    domain: String::new(),
                    name: node_name,
                    inputs: vec![a],
                    outputs: vec![log.clone()],
                    attrs: HashMap::new(),
                });
                let ln2 = self.scalar_init(dt, ConstVal::Float(std::f64::consts::LN_2), "ln2")?;
                Ok(self.emit(id, "Div", vec![log, ln2], vec![]))
            }
            Ops::Trunc => un(self, "Trunc"),
            Ops::Where => {
                let c = self.trace(&srcs[0])?;
                let a = self.trace(&srcs[1])?;
                let b = self.trace(&srcs[2])?;
                Ok(self.emit(id, "Where", vec![c, a, b], vec![]))
            }
            Ops::Mulacc => {
                let a = self.trace(&srcs[0])?;
                let b = self.trace(&srcs[1])?;
                let c = self.trace(&srcs[2])?;
                let m = format!("{}_mul", self.value(id));
                let node_name = self.node_name();
                self.nodes.push(Node {
                    op_type: "Mul".into(),
                    domain: String::new(),
                    name: node_name,
                    inputs: vec![a, b],
                    outputs: vec![m.clone()],
                    attrs: HashMap::new(),
                });
                Ok(self.emit(id, "Add", vec![m, c], vec![]))
            }
            _ => Err(ExportError::UnsupportedOp {
                op: format!("{op:?}"),
                path: path.to_string(),
            }),
        }
    }
}

/// Broadcast a one-element bound list to `rank`, as tinygrad's shape args do
/// for scalars.
fn fit_rank(v: Vec<i64>, rank: usize) -> Option<Vec<i64>> {
    if v.len() == rank {
        Some(v)
    } else if v.len() == 1 {
        Some(vec![v[0]; rank])
    } else {
        None
    }
}

/// Integer list inside a `PERMUTE`-style arg.
fn perm_ints(arg: &Arg) -> Option<Vec<usize>> {
    let Arg::Seq { items, .. } = arg else {
        return None;
    };
    items
        .iter()
        .map(|a| match a {
            Arg::Int(v) if *v >= 0 => usize::try_from(*v).ok(),
            _ => None,
        })
        .collect()
}

/// `[nd-1, 0, 1, .., nd-2]`: move the reduced contraction axis to the front.
fn preduce_permute(perm: &[usize], nd: usize) -> Option<()> {
    if perm.len() != nd || nd < 2 || perm[0] != nd - 1 {
        return None;
    }
    if perm[1..] != (0..nd - 1).collect::<Vec<_>>()[..] {
        return None;
    }
    Some(())
}

/// `[0, .., nd-3, nd-1, nd-2]`: `transpose(-1, -2)`.
fn preduce_swap_last(perm: &[usize], nd: usize) -> Option<()> {
    if perm.len() != nd || nd < 2 {
        return None;
    }
    let mut want: Vec<usize> = (0..nd).collect();
    want.swap(nd - 1, nd - 2);
    (perm == want.as_slice()).then_some(())
}

fn reduce_arg(arg: &Arg, path: &str) -> Result<(Ops, usize)> {
    let Arg::Seq { items, .. } = arg else {
        return Err(ExportError::UnsupportedOp {
            op: "REDUCE arg".into(),
            path: path.to_string(),
        });
    };
    if items.len() < 2 {
        return Err(ExportError::UnsupportedOp {
            op: "REDUCE arg length".into(),
            path: path.to_string(),
        });
    }
    let op = match &items[0] {
        Arg::Enum(e) => Ops::from_u16(e.value).ok_or_else(|| ExportError::UnsupportedOp {
            op: format!("REDUCE enum {}", e.value),
            path: path.to_string(),
        })?,
        _ => {
            return Err(ExportError::UnsupportedOp {
                op: "REDUCE op".into(),
                path: path.to_string(),
            });
        }
    };
    let count = match items[1] {
        Arg::Int(v) if v >= 0 => v as usize,
        _ => {
            return Err(ExportError::UnsupportedOp {
                op: "REDUCE axis count".into(),
                path: path.to_string(),
            });
        }
    };
    Ok((op, count))
}

fn out_dims(dims: &[i64], dynamic: &[(usize, String)]) -> Vec<Dim> {
    dims.iter()
        .enumerate()
        .map(|(i, &d)| {
            dynamic
                .iter()
                .find(|(a, _)| *a == i)
                .map(|(_, n)| Dim::Param(n.clone()))
                .unwrap_or(Dim::Value(d))
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn write_model(
    path: impl AsRef<Path>,
    ctx: &Ctx,
    inputs: &[InputSpec],
    input_names: HashMap<UopId, String>,
    params: HashMap<UopId, (String, TensorProto)>,
    outs: &[Tensor],
    outputs: &[OutputSpec],
    opts: &ExportOptions,
) -> Result<()> {
    if outs.len() != outputs.len() {
        return Err(ExportError::Invalid(format!(
            "{} outputs returned but {} output specs",
            outs.len(),
            outputs.len()
        )));
    }
    // Read tensor metadata before taking the mutable arena borrow.
    let out_meta: Vec<(Vec<i64>, DTypeId)> = outs
        .iter()
        .map(|t| Ok((tensor_dims(t)?, t.dtype())))
        .collect::<Result<Vec<_>>>()?;
    let mut ctx_ref = ctx.borrow_mut();
    let mut tracer = Tracer::new(&mut ctx_ref, input_names, params);
    let mut out_infos = Vec::with_capacity(outs.len());
    for (idx, (t, o)) in outs.iter().zip(outputs).enumerate() {
        let name = tracer.trace(&t.uop())?;
        let (dims, out_dtype) = &out_meta[idx];
        if !tracer.names.values().any(|v| v == &o.name) {
            let out_name = o.name.clone();
            let node_name = tracer.node_name();
            tracer.nodes.push(Node {
                op_type: "Identity".into(),
                domain: String::new(),
                name: node_name,
                inputs: vec![name],
                outputs: vec![out_name.clone()],
                attrs: HashMap::new(),
            });
            out_infos.push(ValueInfo {
                name: out_name,
                dtype: Some(*out_dtype),
                dims: out_dims(dims, &o.dynamic_dims),
                is_optional: false,
                is_sequence: false,
            });
        } else if name == o.name {
            out_infos.push(ValueInfo {
                name,
                dtype: Some(*out_dtype),
                dims: out_dims(dims, &o.dynamic_dims),
                is_optional: false,
                is_sequence: false,
            });
        } else {
            return Err(ExportError::Invalid(format!(
                "duplicate output name `{}`",
                o.name
            )));
        }
    }
    let graph_inputs = inputs
        .iter()
        .map(|spec| ValueInfo {
            name: spec.name.clone(),
            dtype: Some(spec.dtype),
            dims: spec
                .shape
                .iter()
                .enumerate()
                .map(|(i, &d)| {
                    spec.dynamic_dims
                        .iter()
                        .find(|(a, _)| *a == i)
                        .map(|(_, n)| Dim::Param(n.clone()))
                        .unwrap_or(Dim::Value(d))
                })
                .collect(),
            is_optional: false,
            is_sequence: false,
        })
        .collect::<Vec<_>>();
    let graph = GraphProto {
        name: "autorust".into(),
        nodes: tracer.nodes,
        initializers: tracer.initializers,
        inputs: graph_inputs,
        outputs: out_infos,
    };
    let mut graph = graph;
    if let Some(limit) = opts.external_data {
        externalize(&mut graph, path.as_ref(), limit)?;
    }
    let model = ModelProto {
        graph,
        opsets: HashMap::from([(String::new(), opts.opset)]),
        producer: opts.producer.clone(),
    };
    let bytes = encode_model_with_ir(&model, opts.ir_version);
    std::fs::write(path, bytes).map_err(|e| ExportError::Io(e.to_string()))?;
    Ok(())
}

fn externalize(graph: &mut GraphProto, model_path: &Path, limit: u64) -> Result<()> {
    let total: u64 = graph
        .initializers
        .iter()
        .filter_map(|t| t.data.raw.as_ref().map(|r| r.len() as u64))
        .sum();
    if total <= limit {
        return Ok(());
    }
    let sidecar = format!("{}.data", model_path.to_string_lossy());
    let location = Path::new(&sidecar)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("model.onnx.data")
        .to_string();
    let mut bytes = Vec::with_capacity(total as usize);
    let mut offset = 0u64;
    for t in &mut graph.initializers {
        // Keep integer shape/axis initializers inline: ONNX shape inference
        // (onnxruntime and onnx.checker) needs their values while loading.
        let float_tensor = matches!(t.data_type, 1 | 10 | 11 | 16);
        if !float_tensor {
            continue;
        }
        if let Some(raw) = t.data.raw.take() {
            let len = raw.len() as u64;
            t.external = vec![
                ("location".to_string(), location.clone()),
                ("offset".to_string(), offset.to_string()),
                ("length".to_string(), len.to_string()),
            ];
            t.data_location = 1;
            offset += len;
            bytes.extend_from_slice(&raw);
        }
    }
    std::fs::write(&sidecar, bytes).map_err(|e| ExportError::Io(e.to_string()))?;
    Ok(())
}
