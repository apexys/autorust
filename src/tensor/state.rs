//! `nn/state.py`: safetensors + state-dict loading/saving.
//!
//! The reader is streaming: safetensors stores every tensor as a byte range, so
//! we seek to that range and hand the reader to [`Tensor::from_read`]. Only one
//! tensor's bytes are live at a time, which is what makes multi-GB checkpoints
//! usable.
//!
//! The header is JSON. The core dependency policy has no JSON DOM (nanoserde is
//! derive-only), so this module carries a small, strict JSON reader for the
//! safetensors header. It is not a general-purpose JSON library.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};

use crate::dtype::{self, DTypeId};
use crate::tensor::{Ctx, Tensor};

fn safe_name(dt: DTypeId) -> Option<&'static str> {
    Some(match dt {
        dtype::BOOL => "BOOL",
        dtype::INT8 => "I8",
        dtype::UINT8 => "U8",
        dtype::INT16 => "I16",
        dtype::UINT16 => "U16",
        dtype::INT32 => "I32",
        dtype::UINT32 => "U32",
        dtype::INT64 => "I64",
        dtype::UINT64 => "U64",
        dtype::FLOAT16 => "F16",
        dtype::BFLOAT16 => "BF16",
        dtype::FLOAT32 => "F32",
        dtype::FLOAT64 => "F64",
        dtype::FP8E4M3 => "F8_E4M3",
        dtype::FP8E5M2 => "F8_E5M2",
        _ => return None,
    })
}

fn dtype_from_safe(name: &str) -> Option<DTypeId> {
    Some(match name {
        "BOOL" => dtype::BOOL,
        "I8" => dtype::INT8,
        "U8" => dtype::UINT8,
        "I16" => dtype::INT16,
        "U16" => dtype::UINT16,
        "I32" => dtype::INT32,
        "U32" => dtype::UINT32,
        "I64" => dtype::INT64,
        "U64" => dtype::UINT64,
        "F16" => dtype::FLOAT16,
        "BF16" => dtype::BFLOAT16,
        "F32" => dtype::FLOAT32,
        "F64" => dtype::FLOAT64,
        "F8_E4M3" => dtype::FP8E4M3,
        "F8_E5M2" => dtype::FP8E5M2,
        _ => return None,
    })
}

/// Minimal JSON value used for the safetensors header.
#[derive(Clone, Debug)]
enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
    fn as_u64(&self) -> Option<u64> {
        match self {
            Json::Num(n) if *n >= 0.0 => Some(*n as u64),
            _ => None,
        }
    }
    fn as_arr(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(v) => Some(v),
            _ => None,
        }
    }
}

struct JsonParser<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> JsonParser<'a> {
    fn new(s: &'a str) -> Self {
        JsonParser {
            b: s.as_bytes(),
            i: 0,
        }
    }
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\n' | b'\r' | b'\t') {
            self.i += 1;
        }
    }
    fn eat(&mut self, c: u8) -> Result<(), String> {
        self.ws();
        if self.b.get(self.i) == Some(&c) {
            self.i += 1;
            Ok(())
        } else {
            Err(format!("expected '{}' at {}", c as char, self.i))
        }
    }
    fn parse(&mut self) -> Result<Json, String> {
        self.ws();
        let v = match self.b.get(self.i).copied() {
            Some(b'{') => self.object()?,
            Some(b'[') => self.array()?,
            Some(b'"') => Json::Str(self.string()?),
            Some(b't') => {
                self.lit("true")?;
                Json::Bool(true)
            }
            Some(b'f') => {
                self.lit("false")?;
                Json::Bool(false)
            }
            Some(b'n') => {
                self.lit("null")?;
                Json::Null
            }
            Some(_) => self.number()?,
            None => return Err("unexpected end of JSON".into()),
        };
        self.ws();
        Ok(v)
    }
    fn lit(&mut self, s: &str) -> Result<(), String> {
        if self.b[self.i..].starts_with(s.as_bytes()) {
            self.i += s.len();
            Ok(())
        } else {
            Err(format!("bad literal at {}", self.i))
        }
    }
    fn string(&mut self) -> Result<String, String> {
        self.eat(b'"')?;
        let mut out = String::new();
        while let Some(&c) = self.b.get(self.i) {
            self.i += 1;
            match c {
                b'"' => return Ok(out),
                b'\\' => {
                    let e = *self.b.get(self.i).ok_or("truncated escape")?;
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{0008}'),
                        b'f' => out.push('\u{000c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hex = self.b.get(self.i..self.i + 4).ok_or("truncated \\u")?;
                            self.i += 4;
                            let h = std::str::from_utf8(hex).map_err(|_| "bad \\u")?;
                            let n = u16::from_str_radix(h, 16).map_err(|_| "bad \\u hex")?;
                            out.push(char::from_u32(n as u32).unwrap_or('\u{fffd}'));
                        }
                        _ => return Err("bad escape".into()),
                    }
                }
                _ => out.push(c as char),
            }
        }
        Err("unterminated string".into())
    }
    fn number(&mut self) -> Result<Json, String> {
        let start = self.i;
        while self.i < self.b.len()
            && matches!(
                self.b[self.i],
                b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9'
            )
        {
            self.i += 1;
        }
        let s = std::str::from_utf8(&self.b[start..self.i]).map_err(|_| "bad number")?;
        s.parse::<f64>()
            .map(Json::Num)
            .map_err(|_| "bad number".into())
    }
    fn array(&mut self) -> Result<Json, String> {
        self.eat(b'[')?;
        let mut v = Vec::new();
        self.ws();
        if self.b.get(self.i) == Some(&b']') {
            self.i += 1;
            return Ok(Json::Arr(v));
        }
        loop {
            v.push(self.parse()?);
            self.ws();
            match self.b.get(self.i) {
                Some(b',') => {
                    self.i += 1;
                }
                Some(b']') => {
                    self.i += 1;
                    return Ok(Json::Arr(v));
                }
                _ => return Err("bad array".into()),
            }
        }
    }
    fn object(&mut self) -> Result<Json, String> {
        self.eat(b'{')?;
        let mut kv = Vec::new();
        self.ws();
        if self.b.get(self.i) == Some(&b'}') {
            self.i += 1;
            return Ok(Json::Obj(kv));
        }
        loop {
            self.ws();
            let k = self.string()?;
            self.eat(b':')?;
            let v = self.parse()?;
            kv.push((k, v));
            self.ws();
            match self.b.get(self.i) {
                Some(b',') => {
                    self.i += 1;
                }
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Json::Obj(kv));
                }
                _ => return Err("bad object".into()),
            }
        }
    }
}

fn parse_json(s: &str) -> Result<Json, String> {
    JsonParser::new(s).parse()
}

/// A named collection of tensors (tinygrad `state_dict`).
pub struct State {
    pub tensors: HashMap<String, Tensor>,
    pub metadata: HashMap<String, String>,
}

impl State {
    pub fn new() -> Self {
        State {
            tensors: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    pub fn insert(&mut self, name: impl Into<String>, t: Tensor) {
        self.tensors.insert(name.into(), t);
    }

    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    pub fn remove(&mut self, name: &str) -> Option<Tensor> {
        self.tensors.remove(name)
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &Tensor)> {
        self.tensors.iter()
    }

    /// Names under `prefix` with the prefix removed (`"model.da3."` → relative
    /// names). Useful for checkpoints with a wrapper prefix.
    pub fn subset(&self, prefix: &str) -> State {
        let mut out = State::new();
        for (k, v) in &self.tensors {
            if let Some(rest) = k.strip_prefix(prefix) {
                out.tensors
                    .insert(rest.trim_start_matches('.').to_string(), v.clone());
            }
        }
        out.metadata = self.metadata.clone();
        out
    }

    /// Copy every tensor to `device` (tinygrad `state_dict` `.to(device)`).
    pub fn to(&self, device: &str) -> State {
        let mut out = State::new();
        for (k, v) in &self.tensors {
            out.tensors.insert(k.clone(), v.to(device));
        }
        out.metadata = self.metadata.clone();
        out
    }
}

impl Default for State {
    fn default() -> Self {
        Self::new()
    }
}

struct SafeInfo {
    dtype: DTypeId,
    shape: Vec<i64>,
    start: u64,
    end: u64,
}

fn parse_info(v: &Json) -> Result<SafeInfo, String> {
    let dtype_s = v
        .get("dtype")
        .and_then(Json::as_str)
        .ok_or("safetensors: missing dtype")?;
    let dtype = dtype_from_safe(dtype_s)
        .ok_or_else(|| format!("safetensors: unsupported dtype {dtype_s}"))?;
    let shape = v
        .get("shape")
        .and_then(Json::as_arr)
        .ok_or("safetensors: missing shape")?
        .iter()
        .map(|x| x.as_u64().map(|n| n as i64).ok_or("safetensors: bad shape"))
        .collect::<Result<Vec<_>, _>>()?;
    let offs = v
        .get("data_offsets")
        .and_then(Json::as_arr)
        .ok_or("safetensors: missing data_offsets")?;
    if offs.len() != 2 {
        return Err("safetensors: data_offsets must have two entries".into());
    }
    let start = offs[0].as_u64().ok_or("safetensors: bad offset")?;
    let end = offs[1].as_u64().ok_or("safetensors: bad offset")?;
    if end < start {
        return Err("safetensors: end < start".into());
    }
    Ok(SafeInfo {
        dtype,
        shape,
        start,
        end,
    })
}

/// Load a safetensors stream. `device` is forwarded to every tensor (`None` =
/// the default device).
pub fn safe_load_from<R: Read + Seek>(
    reader: &mut R,
    device: Option<&str>,
) -> Result<State, String> {
    let mut len_buf = [0u8; 8];
    reader
        .read_exact(&mut len_buf)
        .map_err(|e| format!("safetensors: header length: {e}"))?;
    let header_len = u64::from_le_bytes(len_buf);
    if header_len > 256 * 1024 * 1024 {
        return Err("safetensors: unreasonably large header".into());
    }
    let mut header = vec![0u8; header_len as usize];
    reader
        .read_exact(&mut header)
        .map_err(|e| format!("safetensors: header: {e}"))?;
    let summary = std::str::from_utf8(&header).map_err(|_| "safetensors: header is not UTF-8")?;
    let root = parse_json(summary)?;
    let Json::Obj(entries) = root else {
        return Err("safetensors: header is not an object".into());
    };

    let data_start = 8 + header_len;
    let ctx = Ctx::new();
    let mut state = State::new();
    for (name, value) in entries {
        if name == "__metadata__" {
            if let Json::Obj(kv) = value {
                for (k, v) in kv {
                    if let Some(s) = v.as_str() {
                        state.metadata.insert(k, s.to_string());
                    }
                }
            }
            continue;
        }
        let info = parse_info(&value)?;
        let nbytes = (info.end - info.start) as usize;
        let want = info.shape.iter().product::<i64>() as usize * info.dtype.itemsize();
        if nbytes != want {
            return Err(format!(
                "safetensors: {name}: byte range {nbytes} != shape*itemsize {want}"
            ));
        }
        reader
            .seek(SeekFrom::Start(data_start + info.start))
            .map_err(|e| format!("safetensors: seek {name}: {e}"))?;
        let mut take = reader.take(nbytes as u64);
        let t = Tensor::from_read(&ctx, &mut take, info.dtype, info.shape.clone(), device);
        state.tensors.insert(name, t);
    }
    Ok(state)
}

/// Load a `.safetensors` file.
pub fn safe_load(path: impl AsRef<std::path::Path>) -> Result<State, String> {
    let mut f = File::open(path.as_ref()).map_err(|e| format!("safetensors: open: {e}"))?;
    safe_load_from(&mut f, None)
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Write a safetensors file. Tensor order is sorted by name for reproducibility;
/// each tensor is realized and written one at a time.
pub fn safe_save(
    state: &State,
    path: impl AsRef<std::path::Path>,
    metadata: Option<&HashMap<String, String>>,
) -> Result<(), String> {
    let mut names: Vec<&String> = state.tensors.keys().collect();
    names.sort();
    let mut header = String::from("{");
    if let Some(md) = metadata {
        header.push_str("\"__metadata__\":{");
        let mut first = true;
        let mut md_names: Vec<&String> = md.keys().collect();
        md_names.sort();
        for k in md_names {
            if !first {
                header.push(',');
            }
            first = false;
            header.push_str(&format!(
                "\"{}\":\"{}\"",
                json_escape(k),
                json_escape(&md[k])
            ));
        }
        header.push_str("},");
    }
    let mut offset: u64 = 0;
    let mut bodies: Vec<Vec<u8>> = Vec::new();
    for (i, name) in names.iter().enumerate() {
        let t = &state.tensors[*name];
        let dt = t.dtype();
        let dtype_name = safe_name(dt)
            .ok_or_else(|| format!("safe_save: unsupported dtype {dt:?} for {name}"))?;
        let shape: Vec<i64> = t
            .shape()
            .into_vec()
            .iter()
            .map(|s| s.clone().as_i64().unwrap_or(0))
            .collect();
        let bytes = t.to_bytes_fresh();
        let n = bytes.len() as u64;
        if i > 0 {
            header.push(',');
        }
        header.push_str(&format!(
            "\"{}\":{{\"dtype\":\"{}\",\"shape\":{:?},\"data_offsets\":[{},{}]}}",
            json_escape(name),
            dtype_name,
            shape,
            offset,
            offset + n
        ));
        offset += n;
        bodies.push(bytes);
    }
    header.push('}');
    // The safetensors header length must be a multiple of 8.
    while header.len() % 8 != 0 {
        header.push(' ');
    }
    let mut f = File::create(path.as_ref()).map_err(|e| format!("safe_save: create: {e}"))?;
    f.write_all(&(header.len() as u64).to_le_bytes())
        .map_err(|e| e.to_string())?;
    f.write_all(header.as_bytes()).map_err(|e| e.to_string())?;
    for b in bodies {
        f.write_all(&b).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// tinygrad `get_state_dict`: parameters **and** buffers, name-flattened.
pub fn get_state_dict<M: crate::tensor::nn::Module>(module: &M) -> State {
    let mut state = State::new();
    for (name, t) in module.named_state() {
        state.tensors.insert(name, t);
    }
    state
}

/// tinygrad `load_state_dict(model, state_dict, strict)`: install every tensor
/// into the model in place (clones of a parameter see the new storage).
///
/// Tensors are copied through bytes when the state was loaded in a different
/// [`Ctx`]; loading a checkpoint must not move a node into a foreign arena.
pub fn load_state_dict<M: crate::tensor::nn::Module>(
    module: &M,
    state: &State,
    strict: bool,
) -> Result<Vec<Tensor>, String> {
    let model: HashMap<String, Tensor> = module.named_state().collect();
    let mut loaded = Vec::new();
    for (name, param) in &model {
        let Some(src0) = state.get(name) else {
            if strict {
                return Err(format!("load_state_dict: missing weight `{name}`"));
            }
            continue;
        };
        let mut src = src0.clone();
        let dst_shape = param.shape().into_vec();
        let src_shape = src.shape().into_vec();
        if src_shape != dst_shape {
            let unit_swap = (src_shape.len() == 0
                && dst_shape.len() == 1
                && dst_shape[0] == crate::uop::sint::SVal::I(1))
                || (dst_shape.len() == 0
                    && src_shape.len() == 1
                    && src_shape[0] == crate::uop::sint::SVal::I(1));
            if !unit_swap {
                return Err(format!(
                    "load_state_dict: shape mismatch for `{name}`: model {dst_shape:?}, state {src_shape:?}"
                ));
            }
            src = src.reshape(dst_shape.clone());
        }
        let pctx = param.ctx().clone();
        if pctx.same(src.ctx()) {
            param.replace(&src);
        } else {
            // Cross-arena: materialize the checkpoint tensor and rebuild it in
            // the model's arena on the parameter's device.
            let bytes = src.to_bytes_fresh();
            let dev = param.device();
            let fresh = Tensor::from_bytes(&pctx, &bytes, src.dtype(), dev.as_deref())
                .reshape(dst_shape.clone());
            param.replace(&fresh);
        }
        loaded.push(param.clone());
    }
    if strict {
        for name in state.keys() {
            if !model.contains_key(name) {
                return Err(format!("load_state_dict: unexpected weight `{name}`"));
            }
        }
    }
    Ok(loaded)
}
