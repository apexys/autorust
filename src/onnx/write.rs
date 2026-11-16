//! Minimal ONNX protobuf writer.
//!
//! Mirrors the field numbers consumed by [`super::pb`]. Only the encodings the
//! exporter needs are emitted; unsupported attribute kinds are skipped by the
//! caller before reaching this module.

use super::pb::{Attribute, Dim, GraphProto, ModelProto, Node, TensorData, TensorProto, ValueInfo};

fn varint(mut v: u64, out: &mut Vec<u8>) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn key(field: u32, wire: u32, out: &mut Vec<u8>) {
    varint(((field as u64) << 3) | wire as u64, out);
}

fn field_varint(field: u32, v: u64, out: &mut Vec<u8>) {
    key(field, 0, out);
    varint(v, out);
}

fn field_i64(field: u32, v: i64, out: &mut Vec<u8>) {
    field_varint(field, v as u64, out);
}

fn field_bytes(field: u32, bytes: &[u8], out: &mut Vec<u8>) {
    key(field, 2, out);
    varint(bytes.len() as u64, out);
    out.extend_from_slice(bytes);
}

fn field_str(field: u32, s: &str, out: &mut Vec<u8>) {
    field_bytes(field, s.as_bytes(), out);
}

fn field_msg(field: u32, bytes: &[u8], out: &mut Vec<u8>) {
    field_bytes(field, bytes, out);
}

fn packed_i64(field: u32, values: &[i64], out: &mut Vec<u8>) {
    if values.is_empty() {
        return;
    }
    let mut body = Vec::new();
    for &v in values {
        varint(v as u64, &mut body);
    }
    field_bytes(field, &body, out);
}

fn packed_f32(field: u32, values: &[f32], out: &mut Vec<u8>) {
    if values.is_empty() {
        return;
    }
    let mut body = Vec::with_capacity(values.len() * 4);
    for &v in values {
        body.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    field_bytes(field, &body, out);
}

/// ONNX `TensorProto.DataType` code for a dtype.
pub(crate) fn dtype_code(dt: crate::dtype::DTypeId) -> Option<i32> {
    use crate::dtype;
    Some(match dt {
        dtype::FLOAT32 => 1,
        dtype::UINT8 => 2,
        dtype::INT8 => 3,
        dtype::UINT16 => 4,
        dtype::INT16 => 5,
        dtype::INT32 => 6,
        dtype::INT64 => 7,
        dtype::BOOL => 9,
        dtype::FLOAT16 => 10,
        dtype::FLOAT64 => 11,
        dtype::UINT32 => 12,
        dtype::UINT64 => 13,
        dtype::BFLOAT16 => 16,
        _ => return None,
    })
}

fn dim_msg(d: &Dim) -> Vec<u8> {
    let mut b = Vec::new();
    match d {
        Dim::Value(v) => field_i64(1, *v, &mut b),
        Dim::Param(s) => field_str(2, s, &mut b),
        Dim::Unknown => {}
    }
    b
}

fn value_info_msg(vi: &ValueInfo) -> Vec<u8> {
    let mut b = Vec::new();
    field_str(1, &vi.name, &mut b);
    // type -> tensor_type
    let mut tensor_type = Vec::new();
    if let Some(code) = vi.dtype.and_then(dtype_code) {
        field_i64(1, code as i64, &mut tensor_type);
    }
    let mut shape = Vec::new();
    for d in &vi.dims {
        field_msg(1, &dim_msg(d), &mut shape);
    }
    field_msg(2, &shape, &mut tensor_type);
    let mut ty = Vec::new();
    field_msg(1, &tensor_type, &mut ty);
    field_msg(2, &ty, &mut b);
    b
}

fn tensor_msg(t: &TensorProto) -> Vec<u8> {
    let mut b = Vec::new();
    packed_i64(1, &t.dims, &mut b);
    field_i64(2, t.data_type as i64, &mut b);
    field_str(8, &t.name, &mut b);
    if !t.external.is_empty() {
        for (k, v) in &t.external {
            let mut m = Vec::new();
            field_str(1, k, &mut m);
            field_str(2, v, &mut m);
            field_msg(13, &m, &mut b);
        }
        field_i64(14, t.data_location, &mut b);
    } else if let Some(raw) = &t.data.raw {
        field_bytes(9, raw, &mut b);
    } else if t.data.has_floats || !t.data.floats.is_empty() {
        packed_f32(4, &t.data.floats, &mut b);
    } else if t.data.has_doubles || !t.data.doubles.is_empty() {
        let mut body = Vec::with_capacity(t.data.doubles.len() * 8);
        for v in &t.data.doubles {
            body.extend_from_slice(&v.to_le_bytes());
        }
        field_bytes(10, &body, &mut b);
    } else if t.data.has_int64s || !t.data.int64s.is_empty() {
        packed_i64(7, &t.data.int64s, &mut b);
    } else if t.data.has_int32s || !t.data.int32s.is_empty() {
        let mut vals = Vec::with_capacity(t.data.int32s.len());
        for v in &t.data.int32s {
            vals.push(*v as i64);
        }
        packed_i64(5, &vals, &mut b);
    }
    b
}

fn attr_msg(name: &str, attr: &Attribute) -> Option<Vec<u8>> {
    let mut b = Vec::new();
    field_str(1, name, &mut b);
    let ty = match attr {
        Attribute::Float(v) => {
            key(2, 5, &mut b);
            b.extend_from_slice(&v.to_bits().to_le_bytes());
            1
        }
        Attribute::Int(v) => {
            field_i64(3, *v, &mut b);
            2
        }
        Attribute::String(v) => {
            field_str(4, v, &mut b);
            3
        }
        Attribute::Tensor(t) => {
            field_msg(5, &tensor_msg(t), &mut b);
            4
        }
        Attribute::Graph(_) => return None,
        Attribute::Floats(v) => {
            packed_f32(7, v, &mut b);
            6
        }
        Attribute::Ints(v) => {
            packed_i64(8, v, &mut b);
            7
        }
        Attribute::Strings(v) => {
            for s in v {
                field_str(9, s, &mut b);
            }
            8
        }
    };
    field_i64(20, ty, &mut b);
    Some(b)
}

fn node_msg(n: &Node) -> Vec<u8> {
    let mut b = Vec::new();
    for s in &n.inputs {
        field_str(1, s, &mut b);
    }
    for s in &n.outputs {
        field_str(2, s, &mut b);
    }
    if !n.name.is_empty() {
        field_str(3, &n.name, &mut b);
    }
    field_str(4, &n.op_type, &mut b);
    let mut attrs: Vec<(&String, &Attribute)> = n.attrs.iter().collect();
    attrs.sort_by(|a, b| a.0.cmp(b.0));
    for (name, attr) in attrs {
        if let Some(m) = attr_msg(name, attr) {
            field_msg(5, &m, &mut b);
        }
    }
    if !n.domain.is_empty() {
        field_str(7, &n.domain, &mut b);
    }
    b
}

fn graph_msg(g: &GraphProto) -> Vec<u8> {
    let mut b = Vec::new();
    for n in &g.nodes {
        field_msg(1, &node_msg(n), &mut b);
    }
    if !g.name.is_empty() {
        field_str(2, &g.name, &mut b);
    }
    for t in &g.initializers {
        field_msg(5, &tensor_msg(t), &mut b);
    }
    for v in &g.inputs {
        field_msg(11, &value_info_msg(v), &mut b);
    }
    for v in &g.outputs {
        field_msg(12, &value_info_msg(v), &mut b);
    }
    b
}

/// Encode a model as ONNX protobuf bytes with the default IR version.
pub fn encode_model(model: &ModelProto) -> Vec<u8> {
    encode_model_with_ir(model, 8)
}

/// Encode a model with an explicit IR version.
pub fn encode_model_with_ir(model: &ModelProto, ir_version: i64) -> Vec<u8> {
    let mut b = Vec::new();
    field_i64(1, ir_version, &mut b);
    if !model.producer.is_empty() {
        field_str(2, &model.producer, &mut b);
    }
    field_msg(7, &graph_msg(&model.graph), &mut b);
    let mut opsets: Vec<(&String, &i64)> = model.opsets.iter().collect();
    opsets.sort_by(|a, b| a.0.cmp(b.0));
    for (domain, version) in opsets {
        let mut m = Vec::new();
        field_str(1, domain, &mut m);
        field_i64(2, *version, &mut m);
        field_msg(8, &m, &mut b);
    }
    b
}

/// Convenience constructor for an initializer tensor.
pub(crate) fn tensor_proto(
    name: &str,
    dims: Vec<i64>,
    data_type: i32,
    raw: Vec<u8>,
) -> TensorProto {
    TensorProto {
        name: name.to_string(),
        dims,
        data_type,
        data: TensorData {
            raw: Some(raw),
            ..Default::default()
        },
        external: Vec::new(),
        data_location: 0,
    }
}
