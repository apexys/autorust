//! Minimal, allocation-conscious protobuf reader for the ONNX wire format.
//!
//! ONNX models are protobufs; the core dependency policy has no protobuf crate,
//! so this module implements only the wire types ONNX uses (varint, fixed32,
//! fixed64, length-delimited) and the message shapes we need. It is deliberately
//! strict: malformed lengths and unsupported wire types are errors, because ONNX
//! is untrusted input.
//!
//! Reference: <https://protobuf.dev/programming-guides/encoding/>

use std::collections::HashMap;
use std::path::Path;

use crate::dtype::{self, DTypeId};

pub(crate) const WIRE_VARINT: u8 = 0;
pub(crate) const WIRE_FIXED64: u8 = 1;
pub(crate) const WIRE_LENGTH: u8 = 2;
pub(crate) const WIRE_START_GROUP: u8 = 3;
pub(crate) const WIRE_END_GROUP: u8 = 4;
pub(crate) const WIRE_FIXED32: u8 = 5;

/// Upper bound on any single message/tensor we are willing to allocate.
pub(crate) const MAX_TENSOR_BYTES: usize = 16 << 30;

/// Error type shared by the parser and runner.
#[derive(Debug, Clone)]
pub enum OnnxError {
    Io(String),
    Parse(String),
    Unsupported(String),
    Shape(String),
    Runtime(String),
}

impl std::fmt::Display for OnnxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OnnxError::Io(s) => write!(f, "onnx io: {s}"),
            OnnxError::Parse(s) => write!(f, "onnx parse: {s}"),
            OnnxError::Unsupported(s) => write!(f, "onnx unsupported: {s}"),
            OnnxError::Shape(s) => write!(f, "onnx shape: {s}"),
            OnnxError::Runtime(s) => write!(f, "onnx runtime: {s}"),
        }
    }
}

impl std::error::Error for OnnxError {}

impl From<String> for OnnxError {
    fn from(s: String) -> Self {
        OnnxError::Parse(s)
    }
}

pub(crate) type Result<T> = std::result::Result<T, OnnxError>;

/// A cursor over a protobuf byte range.
pub(crate) struct Reader<'a> {
    bytes: &'a [u8],
    pub pos: usize,
    end: usize,
}

impl<'a> Reader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Reader {
            bytes,
            pos: 0,
            end: bytes.len(),
        }
    }

    pub fn at_end(&self) -> bool {
        self.pos >= self.end
    }

    fn remaining(&self) -> usize {
        self.end.saturating_sub(self.pos)
    }

    pub fn varint(&mut self) -> Result<u64> {
        let mut value = 0u64;
        let mut shift = 0u32;
        loop {
            if self.pos >= self.end {
                return Err(OnnxError::Parse("truncated varint".into()));
            }
            let b = self.bytes[self.pos];
            self.pos += 1;
            value |= u64::from(b & 0x7f) << shift;
            if b & 0x80 == 0 {
                return Ok(value);
            }
            shift += 7;
            if shift >= 70 {
                return Err(OnnxError::Parse("varint longer than 10 bytes".into()));
            }
        }
    }

    pub fn tag(&mut self) -> Result<(u32, u8)> {
        let t = self.varint()?;
        let field = (t >> 3) as u32;
        let wire = (t & 0x07) as u8;
        if field == 0 {
            return Err(OnnxError::Parse("field number 0 is invalid".into()));
        }
        Ok((field, wire))
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if n > self.remaining() {
            return Err(OnnxError::Parse(format!(
                "truncated message: need {n} bytes, have {}",
                self.remaining()
            )));
        }
        let out = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }

    pub fn fixed32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn fixed64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    pub fn len_delimited(&mut self) -> Result<&'a [u8]> {
        let n = self.varint()? as usize;
        if n > MAX_TENSOR_BYTES {
            return Err(OnnxError::Parse(format!(
                "length-delimited field of {n} bytes exceeds the cap"
            )));
        }
        self.take(n)
    }

    pub fn string(&mut self) -> Result<String> {
        let b = self.len_delimited()?;
        String::from_utf8(b.to_vec()).map_err(|_| OnnxError::Parse("invalid UTF-8 string".into()))
    }

    pub fn sub(&mut self) -> Result<Reader<'a>> {
        let b = self.len_delimited()?;
        Ok(Reader::new(b))
    }

    pub fn skip(&mut self, wire: u8) -> Result<()> {
        match wire {
            WIRE_VARINT => {
                self.varint()?;
            }
            WIRE_FIXED64 => {
                self.take(8)?;
            }
            WIRE_LENGTH => {
                self.len_delimited()?;
            }
            WIRE_FIXED32 => {
                self.take(4)?;
            }
            WIRE_START_GROUP | WIRE_END_GROUP => {
                return Err(OnnxError::Parse("protobuf groups are not supported".into()));
            }
            other => {
                return Err(OnnxError::Parse(format!("unknown wire type {other}")));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// TensorProto
// ---------------------------------------------------------------------------

/// Storage of a `TensorProto`, exactly one of which is present in a valid file.
#[derive(Debug, Default, Clone)]
pub struct TensorData {
    pub raw: Option<Vec<u8>>,
    pub floats: Vec<f32>,
    pub doubles: Vec<f64>,
    pub int32s: Vec<u64>,
    pub int64s: Vec<i64>,
    pub uint64s: Vec<u64>,
    pub has_floats: bool,
    pub has_doubles: bool,
    pub has_int32s: bool,
    pub has_int64s: bool,
    pub has_uint64s: bool,
}

#[derive(Debug, Default, Clone)]
pub struct TensorProto {
    pub name: String,
    pub dims: Vec<i64>,
    pub data_type: i32,
    pub data: TensorData,
    pub external: Vec<(String, String)>,
    pub data_location: i64,
}

impl TensorProto {
    pub fn dtype(&self) -> Result<DTypeId> {
        onnx_dtype(self.data_type)
    }

    pub fn numel(&self) -> Result<usize> {
        let mut n: usize = 1;
        for &d in &self.dims {
            if d < 0 {
                return Err(OnnxError::Parse(format!(
                    "tensor `{}` has a negative dimension {d}",
                    self.name
                )));
            }
            n = n
                .checked_mul(d as usize)
                .ok_or_else(|| OnnxError::Parse(format!("tensor `{}` is too large", self.name)))?;
        }
        Ok(n)
    }

    /// Materialize the little-endian storage bytes for this tensor.
    ///
    /// `origin` is the `.onnx` file path, used to resolve `data_location=1`
    /// external tensors relative to the model directory.
    pub fn to_bytes(&self, origin: Option<&Path>) -> Result<Vec<u8>> {
        let dt = self.dtype()?;
        let numel = self.numel()?;
        let want = numel
            .checked_mul(dt.itemsize())
            .ok_or_else(|| OnnxError::Parse(format!("tensor `{}` is too large", self.name)))?;
        if want > MAX_TENSOR_BYTES {
            return Err(OnnxError::Parse(format!(
                "tensor `{}` needs {want} bytes, above the {MAX_TENSOR_BYTES}-byte cap",
                self.name
            )));
        }

        if self.data_location == 1 {
            let bytes = self.read_external(origin, want)?;
            if bytes.len() < want {
                return Err(OnnxError::Parse(format!(
                    "external tensor `{}` has {} bytes, expected {want}",
                    self.name,
                    bytes.len()
                )));
            }
            return Ok(bytes[..want].to_vec());
        }

        if let Some(raw) = &self.data.raw {
            if raw.len() < want {
                return Err(OnnxError::Parse(format!(
                    "tensor `{}` raw_data has {} bytes, expected {want}",
                    self.name,
                    raw.len()
                )));
            }
            return Ok(raw[..want].to_vec());
        }

        // Typed data fields.
        let d = &self.data;
        let mut out = Vec::with_capacity(want);
        match self.data_type {
            // FLOAT
            1 => {
                if !d.has_floats {
                    return Err(missing(&self.name, "float_data"));
                }
                for &v in &d.floats {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
            // DOUBLE
            11 => {
                if !d.has_doubles {
                    return Err(missing(&self.name, "double_data"));
                }
                for &v in &d.doubles {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
            // INT64
            7 => {
                if !d.has_int64s {
                    return Err(missing(&self.name, "int64_data"));
                }
                for &v in &d.int64s {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
            // INT32 / UINT32
            6 | 12 => {
                if !d.has_int32s && !d.has_uint64s {
                    return Err(missing(&self.name, "int32_data"));
                }
                if d.has_uint64s {
                    for &v in &d.uint64s {
                        out.extend_from_slice(&(v as u32).to_le_bytes());
                    }
                } else {
                    for &v in &d.int32s {
                        out.extend_from_slice(&(v as u32).to_le_bytes());
                    }
                }
            }
            // UINT64
            13 => {
                if !d.has_uint64s && !d.has_int64s {
                    return Err(missing(&self.name, "uint64_data"));
                }
                if d.has_uint64s {
                    for &v in &d.uint64s {
                        out.extend_from_slice(&v.to_le_bytes());
                    }
                } else {
                    for &v in &d.int64s {
                        out.extend_from_slice(&(v as u64).to_le_bytes());
                    }
                }
            }
            // UINT8 / INT8 / UINT16 / INT16 / BOOL
            2 | 3 | 4 | 5 | 9 => {
                if !d.has_int32s {
                    return Err(missing(&self.name, "int32_data"));
                }
                let n = dt.itemsize();
                for &v in &d.int32s {
                    let b = (v as u32).to_le_bytes();
                    out.extend_from_slice(&b[..n]);
                }
            }
            // FLOAT16 / BFLOAT16
            10 | 16 => {
                if !d.has_int32s {
                    return Err(missing(&self.name, "int32_data"));
                }
                for &v in &d.int32s {
                    out.extend_from_slice(&((v as u16).to_le_bytes()));
                }
            }
            other => {
                return Err(OnnxError::Unsupported(format!(
                    "ONNX tensor data type {other} for `{}`",
                    self.name
                )));
            }
        }
        if out.len() < want {
            return Err(OnnxError::Parse(format!(
                "tensor `{}` has {} typed bytes, expected {want}",
                self.name,
                out.len()
            )));
        }
        out.truncate(want);
        Ok(out)
    }

    fn read_external(&self, origin: Option<&Path>, want: usize) -> Result<Vec<u8>> {
        use std::io::{Read, Seek, SeekFrom};
        let mut location = None;
        let mut offset = 0u64;
        let mut length: Option<u64> = None;
        for (k, v) in &self.external {
            match k.as_str() {
                "location" => location = Some(v.clone()),
                "offset" => {
                    offset = v.parse().map_err(|_| {
                        OnnxError::Parse(format!("bad external offset `{v}` for `{}`", self.name))
                    })?
                }
                "length" => {
                    length = Some(v.parse().map_err(|_| {
                        OnnxError::Parse(format!("bad external length `{v}` for `{}`", self.name))
                    })?)
                }
                _ => {}
            }
        }
        let location = location.ok_or_else(|| {
            OnnxError::Parse(format!("external tensor `{}` has no location", self.name))
        })?;
        let origin = origin.ok_or_else(|| {
            OnnxError::Parse(format!(
                "external tensor `{}` needs the model path to resolve `{location}`",
                self.name
            ))
        })?;
        let dir = origin.parent().unwrap_or_else(|| Path::new("."));
        let path = dir.join(location);
        let mut f = std::fs::File::open(&path)
            .map_err(|e| OnnxError::Io(format!("open external data {}: {e}", path.display())))?;
        let n = length.unwrap_or(want as u64);
        if n > MAX_TENSOR_BYTES as u64 {
            return Err(OnnxError::Parse(format!(
                "external tensor `{}` length {n} exceeds the cap",
                self.name
            )));
        }
        if n < want as u64 {
            return Err(OnnxError::Parse(format!(
                "external tensor `{}` length {n} is below the expected {want} bytes",
                self.name
            )));
        }
        let mut out = vec![0u8; n as usize];
        f.seek(SeekFrom::Start(offset))
            .map_err(|e| OnnxError::Io(format!("seek external data: {e}")))?;
        f.read_exact(&mut out)
            .map_err(|e| OnnxError::Io(format!("read external data: {e}")))?;
        Ok(out)
    }
}

fn missing(name: &str, field: &str) -> OnnxError {
    OnnxError::Parse(format!("tensor `{name}` has no raw_data and no {field}"))
}

/// ONNX `TensorProto.DataType` -> tinygrad dtype.
pub(crate) fn onnx_dtype(t: i32) -> Result<DTypeId> {
    Ok(match t {
        1 => dtype::FLOAT32,
        2 => dtype::UINT8,
        3 => dtype::INT8,
        4 => dtype::UINT16,
        5 => dtype::INT16,
        6 => dtype::INT32,
        7 => dtype::INT64,
        9 => dtype::BOOL,
        10 => dtype::FLOAT16,
        11 => dtype::FLOAT64,
        12 => dtype::UINT32,
        13 => dtype::UINT64,
        16 => dtype::BFLOAT16,
        other => {
            return Err(OnnxError::Unsupported(format!(
                "ONNX dtype {other} (only the T0/T1 scalar types are supported)"
            )));
        }
    })
}

// ---------------------------------------------------------------------------
// Attributes, values, nodes, graph, model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum Attribute {
    Float(f32),
    Int(i64),
    String(String),
    Tensor(TensorProto),
    Graph(GraphProto),
    Floats(Vec<f32>),
    Ints(Vec<i64>),
    Strings(Vec<String>),
}

impl Attribute {
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Attribute::Int(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Attribute::Float(v) => Some(*v as f64),
            Attribute::Int(v) => Some(*v as f64),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Attribute::String(v) => Some(v),
            _ => None,
        }
    }
    pub fn as_i64s(&self) -> Option<&[i64]> {
        match self {
            Attribute::Ints(v) => Some(v),
            _ => None,
        }
    }
    pub fn as_f64s(&self) -> Option<Vec<f64>> {
        match self {
            Attribute::Floats(v) => Some(v.iter().map(|x| *x as f64).collect()),
            Attribute::Ints(v) => Some(v.iter().map(|x| *x as f64).collect()),
            _ => None,
        }
    }
    pub fn as_tensor(&self) -> Option<&TensorProto> {
        match self {
            Attribute::Tensor(t) => Some(t),
            _ => None,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct Node {
    pub op_type: String,
    pub domain: String,
    pub name: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attrs: HashMap<String, Attribute>,
}

impl Node {
    pub fn attr_i64(&self, name: &str, default: i64) -> i64 {
        self.attrs
            .get(name)
            .and_then(Attribute::as_i64)
            .unwrap_or(default)
    }
    pub fn attr_i64s(&self, name: &str) -> Option<Vec<i64>> {
        self.attrs
            .get(name)
            .and_then(Attribute::as_i64s)
            .map(|v| v.to_vec())
    }
    pub fn attr_f64(&self, name: &str, default: f64) -> f64 {
        self.attrs
            .get(name)
            .and_then(Attribute::as_f64)
            .unwrap_or(default)
    }
    pub fn attr_str<'a>(&'a self, name: &str, default: &'a str) -> &'a str {
        self.attrs
            .get(name)
            .and_then(Attribute::as_str)
            .unwrap_or(default)
    }
}

/// One dimension of a declared ONNX value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dim {
    Value(i64),
    Param(String),
    Unknown,
}

#[derive(Debug, Clone)]
pub struct ValueInfo {
    pub name: String,
    pub dtype: Option<DTypeId>,
    pub dims: Vec<Dim>,
    pub is_optional: bool,
    pub is_sequence: bool,
}

impl ValueInfo {
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Static dims as concrete integers, or `None` if any dim is symbolic.
    pub fn static_shape(&self) -> Option<Vec<i64>> {
        self.dims
            .iter()
            .map(|d| match d {
                Dim::Value(v) => Some(*v),
                _ => None,
            })
            .collect()
    }
}

#[derive(Debug, Default, Clone)]
pub struct GraphProto {
    pub name: String,
    pub nodes: Vec<Node>,
    pub initializers: Vec<TensorProto>,
    pub inputs: Vec<ValueInfo>,
    pub outputs: Vec<ValueInfo>,
}

#[derive(Debug, Default, Clone)]
pub struct ModelProto {
    pub graph: GraphProto,
    pub opsets: HashMap<String, i64>,
    pub producer: String,
}

/// Parse an ONNX `ModelProto` from a byte slice.
pub fn parse_model(bytes: &[u8]) -> Result<ModelProto> {
    let mut r = Reader::new(bytes);
    let mut model = ModelProto::default();
    while !r.at_end() {
        let (field, wire) = r.tag()?;
        match field {
            2 => model.producer = r.string()?,
            7 => model.graph = parse_graph(&mut r.sub()?)?,
            8 => {
                let (domain, version) = parse_opset(&mut r.sub()?)?;
                model.opsets.insert(domain, version);
            }
            _ => r.skip(wire)?,
        }
    }
    Ok(model)
}

fn parse_opset(r: &mut Reader<'_>) -> Result<(String, i64)> {
    let mut domain = String::new();
    let mut version = 1i64;
    while !r.at_end() {
        let (field, wire) = r.tag()?;
        match field {
            1 => domain = r.string()?,
            2 => version = r.varint()? as i64,
            _ => r.skip(wire)?,
        }
    }
    Ok((domain, version))
}

fn parse_graph(r: &mut Reader<'_>) -> Result<GraphProto> {
    let mut g = GraphProto::default();
    while !r.at_end() {
        let (field, wire) = r.tag()?;
        match field {
            1 => g.nodes.push(parse_node(&mut r.sub()?)?),
            2 => g.name = r.string()?,
            5 => g.initializers.push(parse_tensor(&mut r.sub()?)?),
            11 => g.inputs.push(parse_value_info(&mut r.sub()?)?),
            12 => g.outputs.push(parse_value_info(&mut r.sub()?)?),
            _ => r.skip(wire)?,
        }
    }
    Ok(g)
}

fn parse_node(r: &mut Reader<'_>) -> Result<Node> {
    let mut n = Node::default();
    while !r.at_end() {
        let (field, wire) = r.tag()?;
        match field {
            1 => n.inputs.push(r.string()?),
            2 => n.outputs.push(r.string()?),
            3 => n.name = r.string()?,
            4 => n.op_type = r.string()?,
            5 => {
                let (name, attr) = parse_attribute(&mut r.sub()?)?;
                n.attrs.insert(name, attr);
            }
            7 => n.domain = r.string()?,
            _ => r.skip(wire)?,
        }
    }
    Ok(n)
}

fn parse_attribute(r: &mut Reader<'_>) -> Result<(String, Attribute)> {
    let mut name = String::new();
    let mut f: Option<f32> = None;
    let mut i: Option<i64> = None;
    let mut s: Option<String> = None;
    let mut t: Option<TensorProto> = None;
    let mut g: Option<GraphProto> = None;
    let mut floats = Vec::new();
    let mut has_floats = false;
    let mut ints = Vec::new();
    let mut has_ints = false;
    let mut strings = Vec::new();
    let mut has_strings = false;
    let mut atype: Option<i64> = None;

    while !r.at_end() {
        let (field, wire) = r.tag()?;
        match field {
            1 => name = r.string()?,
            2 => f = Some(f32::from_bits(r.fixed32()?)),
            3 => i = Some(r.varint()? as i64),
            4 => s = Some(r.string()?),
            5 => t = Some(parse_tensor(&mut r.sub()?)?),
            6 => g = Some(parse_graph(&mut r.sub()?)?),
            7 => {
                has_floats = true;
                read_floats(r, wire, &mut floats)?;
            }
            8 => {
                has_ints = true;
                read_i64s(r, wire, &mut ints)?;
            }
            9 => {
                has_strings = true;
                let b = r.len_delimited()?;
                strings.push(
                    String::from_utf8(b.to_vec())
                        .map_err(|_| OnnxError::Parse("attribute string is not UTF-8".into()))?,
                );
            }
            20 => atype = Some(r.varint()? as i64),
            _ => r.skip(wire)?,
        }
    }

    let attr = match atype {
        Some(1) => Attribute::Float(f.ok_or_else(|| missing_attr(&name, "f"))?),
        Some(2) => Attribute::Int(i.ok_or_else(|| missing_attr(&name, "i"))?),
        Some(3) => Attribute::String(s.ok_or_else(|| missing_attr(&name, "s"))?),
        Some(4) => Attribute::Tensor(t.ok_or_else(|| missing_attr(&name, "t"))?),
        Some(5) => Attribute::Graph(g.ok_or_else(|| missing_attr(&name, "g"))?),
        Some(6) => Attribute::Floats(floats),
        Some(7) => Attribute::Ints(ints),
        Some(8) => Attribute::Strings(strings),
        Some(other) => {
            return Err(OnnxError::Unsupported(format!(
                "attribute `{name}` has type {other}"
            )));
        }
        None => {
            // Older files omit `type`; infer from the populated slot.
            if let Some(v) = f {
                Attribute::Float(v)
            } else if let Some(v) = i {
                Attribute::Int(v)
            } else if let Some(v) = s {
                Attribute::String(v)
            } else if let Some(v) = t {
                Attribute::Tensor(v)
            } else if let Some(v) = g {
                Attribute::Graph(v)
            } else if has_floats {
                Attribute::Floats(floats)
            } else if has_ints {
                Attribute::Ints(ints)
            } else if has_strings {
                Attribute::Strings(strings)
            } else {
                return Err(OnnxError::Parse(format!(
                    "attribute `{name}` has no recognizable value"
                )));
            }
        }
    };
    Ok((name, attr))
}

fn missing_attr(name: &str, field: &str) -> OnnxError {
    OnnxError::Parse(format!(
        "attribute `{name}` declares a type but has no {field}"
    ))
}

fn parse_tensor(r: &mut Reader<'_>) -> Result<TensorProto> {
    let mut t = TensorProto::default();
    while !r.at_end() {
        let (field, wire) = r.tag()?;
        match field {
            1 => read_i64s(r, wire, &mut t.dims)?,
            2 => t.data_type = r.varint()? as i32,
            4 => {
                t.data.has_floats = true;
                read_floats(r, wire, &mut t.data.floats)?;
            }
            5 => {
                t.data.has_int32s = true;
                read_varints(r, wire, &mut t.data.int32s)?;
            }
            7 => {
                t.data.has_int64s = true;
                read_i64s(r, wire, &mut t.data.int64s)?;
            }
            8 => t.name = r.string()?,
            9 => t.data.raw = Some(r.len_delimited()?.to_vec()),
            10 => {
                t.data.has_doubles = true;
                read_f64s(r, wire, &mut t.data.doubles)?;
            }
            11 => {
                t.data.has_uint64s = true;
                read_varints(r, wire, &mut t.data.uint64s)?;
            }
            13 => {
                let (k, v) = parse_string_pair(&mut r.sub()?)?;
                t.external.push((k, v));
            }
            14 => t.data_location = r.varint()? as i64,
            _ => r.skip(wire)?,
        }
    }
    Ok(t)
}

fn parse_string_pair(r: &mut Reader<'_>) -> Result<(String, String)> {
    let mut k = String::new();
    let mut v = String::new();
    while !r.at_end() {
        let (field, wire) = r.tag()?;
        match field {
            1 => k = r.string()?,
            2 => v = r.string()?,
            _ => r.skip(wire)?,
        }
    }
    Ok((k, v))
}

fn parse_value_info(r: &mut Reader<'_>) -> Result<ValueInfo> {
    let mut name = String::new();
    let mut ty: Option<Reader<'_>> = None;
    // The type message borrows the same input buffer; keep the raw slice instead.
    let mut ty_bytes: Option<&[u8]> = None;
    while !r.at_end() {
        let (field, wire) = r.tag()?;
        match field {
            1 => name = r.string()?,
            2 => ty_bytes = Some(r.len_delimited()?),
            _ => r.skip(wire)?,
        }
    }
    let mut vi = ValueInfo {
        name,
        dtype: None,
        dims: Vec::new(),
        is_optional: false,
        is_sequence: false,
    };
    if let Some(b) = ty_bytes {
        ty = Some(Reader::new(b));
    }
    if let Some(t) = ty.as_mut() {
        parse_type(t, &mut vi)?;
    }
    Ok(vi)
}

fn parse_type(r: &mut Reader<'_>, vi: &mut ValueInfo) -> Result<()> {
    while !r.at_end() {
        let (field, wire) = r.tag()?;
        match field {
            1 => {
                let mut sub = r.sub()?;
                parse_tensor_type(&mut sub, vi)?;
            }
            4 | 9 => {
                if field == 9 {
                    vi.is_optional = true;
                } else {
                    vi.is_sequence = true;
                }
                // wrapper: field 1 contains the element TypeProto
                let mut w = r.sub()?;
                while !w.at_end() {
                    let (f2, w2) = w.tag()?;
                    match f2 {
                        1 => {
                            let mut inner = w.sub()?;
                            parse_type(&mut inner, vi)?;
                        }
                        _ => w.skip(w2)?,
                    }
                }
            }
            _ => r.skip(wire)?,
        }
    }
    Ok(())
}

fn parse_tensor_type(r: &mut Reader<'_>, vi: &mut ValueInfo) -> Result<()> {
    while !r.at_end() {
        let (field, wire) = r.tag()?;
        match field {
            1 => vi.dtype = Some(onnx_dtype(r.varint()? as i32)?),
            2 => {
                let mut shape = r.sub()?;
                parse_shape(&mut shape, vi)?;
            }
            _ => r.skip(wire)?,
        }
    }
    Ok(())
}

fn parse_shape(r: &mut Reader<'_>, vi: &mut ValueInfo) -> Result<()> {
    while !r.at_end() {
        let (field, wire) = r.tag()?;
        match field {
            1 => {
                let mut d = r.sub()?;
                vi.dims.push(parse_dim(&mut d)?);
            }
            _ => r.skip(wire)?,
        }
    }
    Ok(())
}

fn parse_dim(r: &mut Reader<'_>) -> Result<Dim> {
    let mut value: Option<i64> = None;
    let mut param: Option<String> = None;
    while !r.at_end() {
        let (field, wire) = r.tag()?;
        match field {
            1 => value = Some(r.varint()? as i64),
            2 => param = Some(r.string()?),
            _ => r.skip(wire)?,
        }
    }
    Ok(match (value, param) {
        (Some(v), _) => Dim::Value(v),
        (None, Some(p)) => Dim::Param(p),
        (None, None) => Dim::Unknown,
    })
}

fn read_floats(r: &mut Reader<'_>, wire: u8, out: &mut Vec<f32>) -> Result<()> {
    match wire {
        WIRE_FIXED32 => out.push(f32::from_bits(r.fixed32()?)),
        WIRE_LENGTH => {
            let b = r.len_delimited()?;
            if b.len() % 4 != 0 {
                return Err(OnnxError::Parse(
                    "packed float field has an odd length".into(),
                ));
            }
            for chunk in b.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
        }
        other => {
            return Err(OnnxError::Parse(format!(
                "bad wire type {other} for float field"
            )));
        }
    }
    Ok(())
}

fn read_f64s(r: &mut Reader<'_>, wire: u8, out: &mut Vec<f64>) -> Result<()> {
    match wire {
        WIRE_FIXED64 => out.push(f64::from_bits(r.fixed64()?)),
        WIRE_LENGTH => {
            let b = r.len_delimited()?;
            if b.len() % 8 != 0 {
                return Err(OnnxError::Parse(
                    "packed double field has an odd length".into(),
                ));
            }
            for chunk in b.chunks_exact(8) {
                out.push(f64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]));
            }
        }
        other => {
            return Err(OnnxError::Parse(format!(
                "bad wire type {other} for double field"
            )));
        }
    }
    Ok(())
}

fn read_i64s(r: &mut Reader<'_>, wire: u8, out: &mut Vec<i64>) -> Result<()> {
    match wire {
        WIRE_VARINT => out.push(r.varint()? as i64),
        WIRE_LENGTH => {
            let mut sub = r.sub()?;
            while !sub.at_end() {
                out.push(sub.varint()? as i64);
            }
        }
        other => {
            return Err(OnnxError::Parse(format!(
                "bad wire type {other} for int64 field"
            )));
        }
    }
    Ok(())
}

fn read_varints(r: &mut Reader<'_>, wire: u8, out: &mut Vec<u64>) -> Result<()> {
    match wire {
        WIRE_VARINT => out.push(r.varint()?),
        WIRE_LENGTH => {
            let mut sub = r.sub()?;
            while !sub.at_end() {
                out.push(sub.varint()?);
            }
        }
        other => {
            return Err(OnnxError::Parse(format!(
                "bad wire type {other} for varint field"
            )));
        }
    }
    Ok(())
}
