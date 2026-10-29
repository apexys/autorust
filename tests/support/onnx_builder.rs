#![allow(dead_code)]
//! Tiny ONNX protobuf writer used by `tests/onnx.rs` (test-only).
//!
//! It emits just enough of `ModelProto`/`GraphProto`/`NodeProto`/`TensorProto`
//! to build small fixtures without the `onnx` crate or Python.

pub const FLOAT: i64 = 1;
pub const INT64: i64 = 7;
pub const BOOL: i64 = 9;

#[derive(Default, Clone)]
pub struct Pb(pub Vec<u8>);

impl Pb {
    pub fn new() -> Self {
        Pb(Vec::new())
    }
    fn raw_varint(&mut self, mut v: u64) {
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                self.0.push(b);
                return;
            }
            self.0.push(b | 0x80);
        }
    }
    fn tag(&mut self, field: u32, wire: u8) {
        self.raw_varint(((field as u64) << 3) | wire as u64);
    }
    pub fn uint(&mut self, field: u32, v: u64) -> &mut Self {
        self.tag(field, 0);
        self.raw_varint(v);
        self
    }
    pub fn int(&mut self, field: u32, v: i64) -> &mut Self {
        self.uint(field, v as u64)
    }
    pub fn bytes(&mut self, field: u32, b: &[u8]) -> &mut Self {
        self.tag(field, 2);
        self.raw_varint(b.len() as u64);
        self.0.extend_from_slice(b);
        self
    }
    pub fn string(&mut self, field: u32, s: &str) -> &mut Self {
        self.bytes(field, s.as_bytes())
    }
    pub fn f32(&mut self, field: u32, v: f32) -> &mut Self {
        self.tag(field, 5);
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    pub fn message(&mut self, field: u32, m: &Pb) -> &mut Self {
        self.bytes(field, &m.0)
    }
    pub fn packed_i64(&mut self, field: u32, vals: &[i64]) -> &mut Self {
        let mut inner = Pb::new();
        for &v in vals {
            inner.raw_varint(v as u64);
        }
        self.bytes(field, &inner.0)
    }
    pub fn packed_f32(&mut self, field: u32, vals: &[f32]) -> &mut Self {
        let mut inner = Pb::new();
        for &v in vals {
            inner.0.extend_from_slice(&v.to_le_bytes());
        }
        self.bytes(field, &inner.0)
    }
}

fn shape_proto(dims: &[Dim]) -> Pb {
    let mut shape = Pb::new();
    for d in dims {
        let mut dim = Pb::new();
        match d {
            Dim::Val(v) => {
                dim.int(1, *v);
            }
            Dim::Param(s) => {
                dim.string(2, s);
            }
        }
        shape.message(1, &dim);
    }
    shape
}

#[derive(Clone)]
pub enum Dim {
    Val(i64),
    Param(&'static str),
}

pub fn value_info(name: &str, elem_type: i64, dims: &[Dim]) -> Pb {
    let mut tensor_type = Pb::new();
    tensor_type.int(1, elem_type);
    let shape = shape_proto(dims);
    tensor_type.message(2, &shape);
    let mut ty = Pb::new();
    ty.message(1, &tensor_type);
    let mut vi = Pb::new();
    vi.string(1, name);
    vi.message(2, &ty);
    vi
}

fn tensor_field(name: &str, dims: &[i64], data_type: i64, raw: &[u8], external: bool) -> Pb {
    let mut t = Pb::new();
    t.packed_i64(1, dims);
    t.int(2, data_type);
    t.string(8, name);
    if external {
        // data_location = 1; raw_data is omitted.
        t.int(14, 1);
    } else {
        t.bytes(9, raw);
    }
    t
}

pub fn f32_tensor(name: &str, dims: &[i64], data: &[f32]) -> Pb {
    let mut raw = Vec::with_capacity(data.len() * 4);
    for v in data {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    tensor_field(name, dims, FLOAT, &raw, false)
}

/// f32 tensor encoded with the typed `float_data` field instead of `raw_data`.
pub fn f32_tensor_typed(name: &str, dims: &[i64], data: &[f32]) -> Pb {
    let mut t = Pb::new();
    t.packed_i64(1, dims);
    t.int(2, FLOAT);
    t.string(8, name);
    t.packed_f32(4, data);
    t
}

pub fn i64_tensor(name: &str, dims: &[i64], data: &[i64]) -> Pb {
    let mut raw = Vec::with_capacity(data.len() * 8);
    for v in data {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    tensor_field(name, dims, INT64, &raw, false)
}

/// `TensorProto` whose bytes live in an external file at `location`.
pub fn external_f32_tensor(
    name: &str,
    dims: &[i64],
    location: &str,
    offset: u64,
    length: u64,
) -> Pb {
    let mut t = tensor_field(name, dims, FLOAT, &[], true);
    for (k, v) in [
        ("location", location.to_string()),
        ("offset", offset.to_string()),
        ("length", length.to_string()),
    ] {
        let mut kv = Pb::new();
        kv.string(1, k);
        kv.string(2, &v);
        t.message(13, &kv);
    }
    t
}

pub enum Attr<'a> {
    Int(&'a str, i64),
    Float(&'a str, f32),
    Ints(&'a str, Vec<i64>),
    Floats(&'a str, Vec<f32>),
    Str(&'a str, &'a str),
    Tensor(&'a str, Pb),
}

fn attribute(a: &Attr<'_>) -> Pb {
    let mut p = Pb::new();
    match a {
        Attr::Int(name, v) => {
            p.string(1, name);
            p.int(3, *v);
            p.int(20, 2);
        }
        Attr::Float(name, v) => {
            p.string(1, name);
            p.f32(2, *v);
            p.int(20, 1);
        }
        Attr::Ints(name, v) => {
            p.string(1, name);
            p.packed_i64(8, v);
            p.int(20, 7);
        }
        Attr::Floats(name, v) => {
            p.string(1, name);
            p.packed_f32(7, v);
            p.int(20, 6);
        }
        Attr::Str(name, v) => {
            p.string(1, name);
            p.string(4, v);
            p.int(20, 3);
        }
        Attr::Tensor(name, t) => {
            p.string(1, name);
            p.message(5, t);
            p.int(20, 4);
        }
    }
    p
}

pub fn node(op: &str, inputs: &[&str], outputs: &[&str], attrs: &[Attr<'_>]) -> Pb {
    let mut n = Pb::new();
    for i in inputs {
        n.string(1, i);
    }
    for o in outputs {
        n.string(2, o);
    }
    n.string(3, op);
    n.string(4, op);
    for a in attrs {
        n.message(5, &attribute(a));
    }
    n
}

pub fn graph(nodes: &[Pb], inits: &[Pb], inputs: &[Pb], outputs: &[Pb], name: &str) -> Pb {
    let mut g = Pb::new();
    for n in nodes {
        g.message(1, n);
    }
    g.string(2, name);
    for i in inits {
        g.message(5, i);
    }
    for i in inputs {
        g.message(11, i);
    }
    for o in outputs {
        g.message(12, o);
    }
    g
}

pub fn model(graph: &Pb, opset: i64) -> Vec<u8> {
    let mut m = Pb::new();
    m.string(2, "autorust-test");
    m.message(7, graph);
    let mut op = Pb::new();
    op.int(2, opset);
    m.message(8, &op);
    m.0
}
