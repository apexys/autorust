//! The uops interpreter (`runtime/ops_python.py` model) — corpus subset.
//!
//! Executes linearized uops: global/local launch loops from SPECIAL extents,
//! per-thread value lists, byte-backed buffer views for memory, exec masks, ALU.

use crate::dtype::{self, AddrSpace, ConstVal, DTypeId};
use crate::uop::arg::Arg;
use crate::uop::ctx::UOpCtx;
use crate::uop::ops::{group, Ops};
use crate::uop::UopId;
use std::rc::Rc;
use std::cell::RefCell;

/// A byte-backed allocation (one per BUFFER uop).

/// A memoryview: (buffer, byte offset, element size in bytes, element count).
#[derive(Clone, Debug)]
pub struct MemView(pub Rc<RefCell<Vec<u8>>>, pub usize, pub usize, pub usize);

pub fn from_storage_scalar(bytes: &[u8], dt: DTypeId) -> f64 {
    let b = |i: usize| *bytes.get(i).unwrap_or(&0);
    match dt {
        d if d == dtype::BOOL => (b(0) != 0) as i64 as f64,
        d if d == dtype::UINT8 => b(0) as f64,
        d if d == dtype::INT8 => b(0) as i8 as f64,
        d if d == dtype::INT32 => i32::from_le_bytes([b(0), b(1), b(2), b(3)]) as f64,
        d if d == dtype::UINT32 => u32::from_le_bytes([b(0), b(1), b(2), b(3)]) as f64,
        d if d == dtype::FLOAT32 => f32::from_le_bytes([b(0), b(1), b(2), b(3)]) as f64,
        d if d == dtype::INT64 || d == dtype::UINT64 => {
            let mut a = [0u8; 8];
            a.copy_from_slice(&bytes[..8]);
            if d == dtype::INT64 { i64::from_le_bytes(a) as f64 } else { u64::from_le_bytes(a) as f64 }
        }
        d if d == dtype::FLOAT64 => {
            let mut a = [0u8; 8];
            a.copy_from_slice(&bytes[..8]);
            f64::from_le_bytes(a)
        }
        d if d == dtype::INT16 => i16::from_le_bytes([b(0), b(1)]) as f64,
        d if d == dtype::UINT16 => u16::from_le_bytes([b(0), b(1)]) as f64,
        d if d == dtype::FLOAT16 => {
            let raw = u16::from_le_bytes([b(0), b(1)]);
            dtype::f16_bits_to_f64(raw)
        }
        d if d == dtype::BFLOAT16 => {
            let raw = u16::from_le_bytes([b(0), b(1)]) as u32;
            f32::from_bits(raw << 16) as f64
        }
        d if d == dtype::FP8E4M3 || d == dtype::FP8E5M2 || d == dtype::FP8E4M3FNUZ || d == dtype::FP8E5M2FNUZ => {
            dtype::fp8_to_float(b(0), d)
        }
        _ => panic!("TGERR: from_storage_scalar {dt:?}"),
    }
}

pub fn to_storage_scalar(v: f64, dt: DTypeId) -> Vec<u8> {
    match dt {
        d if d == dtype::BOOL => vec![(v != 0.0) as u8],
        d if d == dtype::UINT8 => vec![v as u8],
        d if d == dtype::INT8 => vec![v as i8 as u8],
        d if d == dtype::INT32 => (v as i32).to_le_bytes().to_vec(),
        d if d == dtype::UINT32 => (v as u32).to_le_bytes().to_vec(),
        d if d == dtype::FLOAT32 => (v as f32).to_le_bytes().to_vec(),
        d if d == dtype::INT64 => (v as i64).to_le_bytes().to_vec(),
        d if d == dtype::UINT64 => (v as u64).to_le_bytes().to_vec(),
        d if d == dtype::FLOAT64 => v.to_le_bytes().to_vec(),
        d if d == dtype::INT16 => (v as i16).to_le_bytes().to_vec(),
        d if d == dtype::UINT16 => (v as u16).to_le_bytes().to_vec(),
        d if d == dtype::FLOAT16 => dtype::f64_to_f16_bits(v).to_le_bytes().to_vec(),
        d if d == dtype::BFLOAT16 => {
            let f32_bits = (dtype::float_to_bf16(v) as f32).to_bits();
            ((f32_bits >> 16) as u16).to_le_bytes().to_vec()
        }
        d if d == dtype::FP8E4M3 || d == dtype::FP8E5M2 || d == dtype::FP8E4M3FNUZ || d == dtype::FP8E5M2FNUZ => {
            vec![dtype::float_to_fp8(v, d)]
        }
        _ => panic!("TGERR: to_storage_scalar {dt:?}"),
    }
}

#[derive(Clone)]
enum Val {
    /// per-thread scalar
    S(Vec<f64>),
    /// per-(lane, thread) vector (lane-major)
    V(Vec<Vec<f64>>),
    /// per-thread (view, element offset) pairs
    P(Vec<(MemView, usize)>),
    /// per-thread buffer view
    M(Vec<MemView>),
}

/// `exec_alu` (corpus subset).
/// `python_alu` (ops.py:1287) over f64 lanes; int ops go through i64 bitops.
pub(crate) fn exec_alu(op: Ops, a: f64, b: f64) -> f64 {
    match op {
        Ops::Add => a + b,
        Ops::Mul => a * b,
        Ops::Sub => a - b,
        Ops::Fdiv => a / b,
        Ops::Max => a.max(b),
        Ops::Neg => -a,
        Ops::Exp2 => {
            let r = (2.0f64).powf(a);
            if r.is_infinite() { f64::INFINITY } else { r }
        }
        Ops::Log2 => {
            if a > 0.0 { a.log2() } else if a == 0.0 { f64::NEG_INFINITY } else { f64::NAN }
        }
        Ops::Sqrt => {
            if a >= 0.0 { a.sqrt() } else { f64::NAN }
        }
        Ops::Reciprocal => {
            if a != 0.0 { 1.0 / a } else { f64::copysign(f64::INFINITY, a) }
        }
        Ops::Sin => {
            if a.is_infinite() { f64::NAN } else { a.sin() }
        }
        Ops::Pow => {
            let p = a.powf(b);
            if p.is_nan() && !(a.is_nan() || b.is_nan()) {
                // python pow(neg, non-integer) -> complex -> nan; 0^neg -> inf
                if a == 0.0 { f64::INFINITY } else { f64::NAN }
            } else if p.is_infinite() && a == 0.0 && b < 0.0 {
                f64::INFINITY
            } else {
                p
            }
        }
        Ops::Trunc => {
            if a.is_finite() { a.trunc() } else { a }
        }
        Ops::Cmpne => (a != b) as i64 as f64,
        Ops::Cmpeq => (a == b) as i64 as f64,
        Ops::Cmplt => (a < b) as i64 as f64,
        Ops::Xor => ((a as i64) ^ (b as i64)) as f64,
        Ops::Or => ((a as i64) | (b as i64)) as f64,
        Ops::And => ((a as i64) & (b as i64)) as f64,
        Ops::Shr => ((a as i64) >> (b as i64)) as f64,
        Ops::Shl => ((a as i64) << (b as i64)) as f64,
        Ops::Cdiv => {
            // helpers.py:71 — abs(x)//abs(y)*sign, 0 on y==0
            if b as i64 == 0 {
                0.0
            } else {
                let (ai, bi) = (a as i64, b as i64);
                (ai.abs() / bi.abs() * if ai ^ bi < 0 { -1 } else { 1 }) as f64
            }
        }
        Ops::Floordiv => {
            // helpers.py:74 — python floor semantics, 0 on y==0
            if b as i64 == 0 {
                0.0
            } else {
                let (ai, bi) = (a as i64, b as i64);
                ai.div_euclid(bi) as f64
            }
        }
        Ops::Cmod => {
            // helpers.py:72 — cmod(x,y) = x - cdiv(x,y)*y
            let d = exec_alu(Ops::Cdiv, a, b);
            a - d * b
        }
        Ops::Floormod => {
            // helpers.py:75 — floormod(x,y) = x - floordiv(x,y)*y
            let d = exec_alu(Ops::Floordiv, a, b);
            a - d * b
        }
        other => panic!("TGERR: interp ALU {other:?} not ported"),
    }
}

/// ops.py:1295 — MULACC/WHERE are the ternary ALU ops.
pub(crate) fn exec_alu_ternary(op: Ops, a: f64, b: f64, c: f64) -> f64 {
    match op {
        Ops::Mulacc => (a * b) + c,
        Ops::Where => {
            if a != 0.0 {
                b
            } else {
                c
            }
        }
        other => panic!("TGERR: interp ternary ALU {other:?} not ported"),
    }
}

/// `truncate` (ops.py:1278) — round/clamp to the dtype's representable values.
pub(crate) fn truncate(v: f64, dt: DTypeId) -> f64 {
    if dt == dtype::BOOL { (v != 0.0) as i64 as f64 }
    else if dtype::is_float(dt) { dtype::trunc_float(dt, v) }
    else {
        // clang C-cast semantics (what tinygrad realize produces): unsigned
        // dests clamp negatives to 0 at the float level, then the truncated
        // integer wraps to the dtype width (two's complement low bits).
        let i = if dtype::in_set(dtype::UINTS, dt) {
            (if v < 0.0 { 0.0 } else { v }) as i64
        } else {
            v as i64
        };
        let unsigned = dtype::in_set(dtype::UINTS, dt);
        match dt.bitsize() {
            8 if unsigned => (i as u8) as f64,
            8 => (i as i8) as f64,
            16 if unsigned => (i as u16) as f64,
            16 => (i as i16) as f64,
            32 if unsigned => (i as u32) as f64,
            32 => (i as i32) as f64,
            _ => i as f64,
        }
    }
}

fn _load(m: &MemView, i: usize, dt: DTypeId) -> f64 {
    if i >= m.3 { panic!("TGERR: load out of bounds, size is {} and access is {i}", m.3); }
    let boff = m.1 + i * m.2;
    let bytes: Vec<u8> = (0..m.2).map(|k| m.0.borrow()[boff + k]).collect();
    from_storage_scalar(&bytes, dt)
}

fn _store(m: &MemView, i: usize, v: f64, dt: DTypeId) {
    if i >= m.3 { panic!("TGERR: store out of bounds, size is {}, access is {i}", m.3); }
    // python packs sub-dword views; the corpus stores at matching itemsize
    assert!(m.2 >= dt.itemsize(), "TGERR: sub-dword store not ported (elemsize={} dt={:?} itemsize={} off={} size={})", m.2, dt, dt.itemsize(), m.1, m.3);
    let boff = m.1 + i * m.2;
    let sval = to_storage_scalar(v, dt);
    let mut b = m.0.borrow_mut();
    for (k, x) in sval.iter().enumerate() { b[boff + k] = *x; }
}

/// one (global_idx, local_size) launch invocation.
fn exec_launch(
    ctx: &UOpCtx,
    numels: &std::collections::HashMap<UopId, usize>,
    uops: &[UopId],
    gidx: [usize; 3],
    lsize: [usize; 3],
    views: &std::collections::HashMap<i64, MemView>,
    vals: &[f64],
) {
    // warp threads: last local dim varies fastest
    let mut warp: Vec<[usize; 3]> = Vec::new();
    for c in 0..lsize[2] {
        for b in 0..lsize[1] {
            for a in 0..lsize[0] {
                warp.push([a, b, c]);
            }
        }
    }
    let wsize = warp.len().max(1);
    let mut values: std::collections::HashMap<UopId, Val> = std::collections::HashMap::new();
    let _pbufs: std::collections::VecDeque<MemView> = std::collections::VecDeque::new();
    let mut pvals: std::collections::VecDeque<f64> = vals.iter().cloned().collect();
    let mut exec_masks: Vec<Vec<bool>> = vec![vec![true; wsize]];

    let sv = |values: &std::collections::HashMap<UopId, Val>, u: &UopId| -> Vec<f64> {
        match &values[u] {
            Val::S(v) => v.clone(),
            Val::V(l) => l.iter().map(|t| t.first().cloned().unwrap_or(0.0)).collect(),
            _ => panic!("TGERR: interp src not scalar"),
        }
    };

    let mut i = 0usize;
    while i < uops.len() {
        let u = &uops[i].clone();
        let op = ctx.op_of(u);
        let srcs: Vec<UopId> = ctx.srcs(u).to_vec();
        let src_vals: Vec<Val> = srcs.iter().filter(|s| ctx.dt_of(s) != dtype::VOID).map(|s| values.get(s).cloned().unwrap_or_else(|| panic!("TGERR: interp missing value: src {} op {:?} of u {} op {:?}", s.id(), ctx.op_of(s), u.id(), op))).collect();
        match op {
            Ops::End => {
                if srcs.len() == 3 {
                    let cond = sv(&values, &srcs[2]);
                    if cond.first().cloned().unwrap_or(0.0) != 0.0 {
                        // conditional backedge: jump back to the loop (python: uop_to_index[u.src[1]])
                        i = loop_start_index(uops, &srcs[1]);
                        continue;
                    }
                } else {
                    i = loop_start_index(uops, &srcs[1]);
                    continue;
                }
            }
            Ops::Range => {
                if ctx.dt_of(u) == dtype::VOID { i += 1; continue; }
                let ext = numels.get(u).cloned().unwrap_or(1);
                match values.get_mut(u) {
                    None => { values.insert(u.clone(), Val::S(vec![0.0; wsize])); }
                    Some(Val::S(vs)) => {
                        for v in vs.iter_mut() { *v += 1.0; }
                        if vs[0] == ext as f64 {
                            values.remove(u);
                            i = loop_end_index(ctx, uops, u) + 1;
                            continue;
                        }
                    }
                    _ => panic!("TGERR: interp RANGE value"),
                }
            }
            Ops::If => {
                let cond = sv(&values, &srcs[0]);
                let last = exec_masks.last().unwrap().clone();
                exec_masks.push(last.iter().zip(cond.iter().chain(std::iter::repeat(&cond[0]))).map(|(a, b)| *a && *b != 0.0).collect());
            }
            Ops::Endif => {
                exec_masks.pop();
            }
            Ops::After => {
                // python: values[u] = src_values[0] — the After node carries
                // its first (non-void) src's value forward
                let v = srcs
                    .iter()
                    .find(|s| ctx.dt_of(s) != dtype::VOID)
                    .and_then(|s| values.get(s).cloned());                match v {
                    Some(v) => { values.insert(u.clone(), v); }
                    None => panic!("TGERR: interp AFTER src missing value"),
                }
            }
            Ops::Barrier | Ops::Sink | Ops::Noop | Ops::Group => {}
            Ops::Store => {
                let gate = exec_masks.last().unwrap().clone();
                let store_n = numels.get(u).cloned().unwrap_or(1);
                let vals_lane: Vec<Vec<f64>> = match &src_vals[1] {
                    Val::S(vs) if store_n == 1 => vec![vs.clone()],
                    Val::V(lanes) => lanes.clone(),
                    _ => panic!("TGERR: interp store value"),
                };
                let pairs_owned: Vec<(MemView, usize)>;
                let pairs: &Vec<(MemView, usize)> = match &src_vals[0] {
                    Val::P(p) => p,
                    // a stacked DEST view (devectorized store: one dest view
                    // per lane) — treat each element as its own pair list
                    Val::V(_) | Val::S(_) => {
                        pairs_owned = Vec::new();
                        &pairs_owned
                    }
                    _ => panic!("TGERR: interp store src node {} val={:?}", u.id(), ctx.op_of(&srcs[0])),
                };
                // the store writes the DEST buffer's dtype (python's kernel
                // invariant: value dtype == buffer dtype; the codegen may
                // widen ALU to f32 for sub-dword types — cast back here)
                let store_dt = ctx.dt_of(&ctx.srcs(&srcs[0])[0]);
                if ctx.op_of(&srcs[0]) == Ops::Stack && matches!(src_vals[0], Val::S(_)) {
                    // devectorized store: Stack(dest view per lane) — each
                    // lane writes through its own view at its own offset
                    let dest_srcs = ctx.srcs(&srcs[0]).to_vec();
                    for (j, val) in vals_lane.iter().enumerate() {
                        let dj = dest_srcs.get(j).cloned().unwrap_or((dest_srcs[0]).clone());
                        let dpairs = match values.get(&dj) {
                            Some(Val::P(p)) => p.clone(),
                            _ => continue,
                        };
                        for (t, ((m, o), v)) in dpairs.iter().zip(val.iter()).enumerate() {
                            if gate[t] { _store(m, *o, *v, store_dt); }
                        }
                    }
                    i += 1;
                    continue;
                }
                for (j, val) in vals_lane.iter().enumerate() {
                    for (t, ((m, o), v)) in pairs.iter().zip(val.iter()).enumerate() {
                        if gate[t] { _store(m, *o + j, *v, store_dt); }
                    }
                }
            }
            Ops::Param | Ops::Buffer => {
                if ctx.addrspace_of(u.clone()) == Some(AddrSpace::Alu) && op == Ops::Param {
                    let v = pvals.pop_front().unwrap_or(0.0);
                    values.insert(u.clone(), Val::S(vec![v; wsize]));
                } else if op == Ops::Param {
                    let slot = match ctx.arg_of(u) {
                        Arg::Param(p) => p.slot,
                        _ => panic!("TGERR: interp PARAM arg"),
                    };
                    let mv = views.get(&slot).unwrap_or_else(|| panic!("TGERR: interp missing view for slot {slot}")).clone();
                    values.insert(u.clone(), Val::M(vec![mv; wsize]));
                } else if ctx.addrspace_of(u.clone()) == Some(AddrSpace::Reg) {
                    // REGs are per thread
                    let nel = numels.get(u).cloned().unwrap_or(1);
                    let isz = ctx.dt_of(u).itemsize();
                    values.insert(u.clone(), Val::M((0..wsize).map(|_| MemView(Rc::new(RefCell::new(vec![0u8; nel * isz])), 0, isz, nel)).collect()));
                } else {
                    let nel = numels.get(u).cloned().unwrap_or(1);
                    let isz = ctx.dt_of(u).itemsize();
                    let mv = MemView(Rc::new(RefCell::new(vec![0u8; nel * isz])), 0, isz, nel);
                    values.insert(u.clone(), Val::M(vec![mv; wsize]));
                }
            }
            Ops::Special => {
                let name = match ctx.arg_of(u) {
                    Arg::Str(s) => s.clone(),
                    _ => panic!("TGERR: interp SPECIAL arg"),
                };
                if name.starts_with('g') {
                    let dim: usize = name[name.len() - 1..].parse().unwrap();
                    values.insert(u.clone(), Val::S(vec![gidx[2 - dim] as f64; wsize]));
                } else {
                    let dim: usize = name[name.len() - 1..].parse().unwrap();
                    let vv: Vec<f64> = warp.iter().map(|w| w[2 - dim] as f64).collect();
                    #[cfg(feature = "dump")]
                    if std::env::var("TG_DEBUG_LOAD").is_ok() {
                        eprintln!("TGDBG lidx name={} dim={} vals={:?} warp={:?}", name, dim, vv, warp);
                    }
                    values.insert(u.clone(), Val::S(vv));
                }
            }
            Ops::Const => {
                let v = match ctx.val(u) {
                    ConstVal::Int(x) => x as f64,
                    ConstVal::Float(x) => x,
                    ConstVal::Bool(b) => b as i64 as f64,
                    other => panic!("TGERR: interp const {other:?}"),
                };
                values.insert(u.clone(), Val::S(vec![v; wsize]));
            }
            Ops::Index | Ops::Shrink => {
                // (view, element offset) pairs
                // ALU storage, a devectorized multi-lane load, or a scalar value
                // (indexing a scalar is identity — `INDEX(scalar, i) -> scalar`)
                if ctx.addrspace_derived((srcs[0]).clone()) == Some(AddrSpace::Alu)
                    || matches!(src_vals[0], Val::V(_) | Val::S(_))
                {
                    // index over a vector value: [lane][thread] pick (ALU
                    // storage or a devectorized multi-lane load)
                    // python: ret = [src_values[0][i][t] for t, i in enumerate(src_values[1])]
                    let ret: Vec<f64> = match (&src_vals[0], &src_vals[1]) {
                        (Val::V(a), Val::S(b)) => (0..wsize).map(|t| a[b[t.min(b.len() - 1)] as usize][t.min(a[0].len() - 1)]).collect(),
                        (Val::V(a), Val::V(b)) => (0..wsize).map(|t| a[b[t.min(b.len() - 1)][0] as usize][t.min(a[0].len() - 1)]).collect(),
                        (Val::S(a), _) => a.clone(),
                        _ => panic!("TGERR: interp ALU index srcs"),
                    };
                    values.insert(u.clone(), Val::S(ret));
                    i += 1;
                    continue;
                }
                let mut ret: Vec<(MemView, usize)> = Vec::new();
                match (&src_vals[0], &src_vals[1]) {
                    (Val::M(ms), Val::S(os)) => {
                        for (m, o) in ms.iter().zip(os.iter()) { ret.push((m.clone(), *o as usize)); }
                    }
                    (Val::P(ps), Val::S(os)) => {
                        for ((m, o), o2) in ps.iter().zip(os.iter()) { ret.push((m.clone(), o + *o2 as usize)); }
                    }
                    (Val::M(_), Val::V(_)) | (Val::P(_), Val::V(_)) => {
                        let os = match &src_vals[1] {
                            Val::V(l) => l.iter().map(|t| t.first().cloned().unwrap_or(0.0) as usize).collect::<Vec<usize>>(),
                            _ => unreachable!(),
                        };
                        let ms: Vec<MemView> = match &src_vals[0] {
                            Val::M(ms) => ms.clone(),
                            Val::P(ps) => ps.iter().map(|(m, _)| m.clone()).collect(),
                            _ => unreachable!(),
                        };
                        for (t, m) in ms.iter().enumerate() { ret.push((m.clone(), os[t.min(os.len() - 1)])); }
                    }
                    _ => panic!(
                        "TGERR: interp index srcs u={} op={:?} src0_op={:?} src0={} v0={} v1={} nsrc={}",
                        u.id(),
                        ctx.op_of(u),
                        ctx.op_of(&srcs[0]),
                        srcs[0].id(),
                        match &src_vals[0] { Val::S(_) => "S", Val::V(_) => "V", Val::M(_) => "M", Val::P(_) => "P" },
                        match &src_vals[1] { Val::S(_) => "S", Val::V(_) => "V", Val::M(_) => "M", Val::P(_) => "P" },
                        srcs.len(),
                    ),
                }
                values.insert(u.clone(), Val::P(ret));
            }
            Ops::Load => {
                let load_sz = numels.get(u).cloned().unwrap_or(1);
                let pairs: Vec<(MemView, usize)> = match &src_vals[0] {
                    Val::P(p) => p.clone(),
                    other => panic!(
                        "TGERR: interp load src u={} src0={} op={:?} dt={:?} addrspace={:?} variant_tag={} nsrc={}",
                        u.id(),
                        srcs[0].id(),
                        ctx.op_of(&srcs[0]),
                        ctx.dt_of(&srcs[0]),
                        ctx.addrspace_of((srcs[0]).clone()),
                        match other { Val::S(_) => "S", Val::V(_) => "V", Val::M(_) => "M", Val::P(_) => "P" },
                        srcs.len(),
                    ),
                };
                // gated load: srcs = [index, alt(fill), gate(valid)].
                // python eval: buf[i] if valid else alt — the gate combines with
                // the active control-flow mask; out-of-range reads yield `alt`
                // (e.g. PAD fill), never a clamped base index.
                let gated = src_vals.len() >= 3;
                let tovec = |v: &Val| -> Vec<f64> {
                    match v {
                        Val::S(s) => s.clone(),
                        Val::V(l) => l.iter().map(|t| *t.first().unwrap_or(&0.0)).collect(),
                        _ => panic!("TGERR: interp load gate/alt src"),
                    }
                };
                let (alt, gate) = if gated { (tovec(&src_vals[1]), tovec(&src_vals[2])) } else { (Vec::new(), Vec::new()) };
                let mask_at = |t: usize| -> bool {
                    let em = exec_masks.last().unwrap();
                    em[t.min(em.len() - 1)] && (!gated || gate[t.min(gate.len() - 1)] != 0.0)
                };
                let alt_at = |t: usize| -> f64 { if gated { alt[t.min(alt.len() - 1)] } else { 0.0 } };
                if load_sz > 1 {
                    // lane-major vector load
                    let lanes: Vec<Vec<f64>> = (0..load_sz)
                        .map(|j| pairs.iter().enumerate().map(|(t, (m, o))| if mask_at(t) { _load(m, o + j, ctx.dt_of(u)) } else { alt_at(t) }).collect())
                        .collect();
                    values.insert(u.clone(), Val::V(lanes));
                } else {
                    #[cfg(feature = "dump")]
                    if std::env::var("TG_DEBUG_LOAD").is_ok() {
                        let offs: Vec<usize> = pairs.iter().map(|(_, o)| *o).collect();
                        eprintln!("TGDBG load uop={} size={} offs={:?}", u.id(), pairs[0].0.3, offs);
                    }
                    let tv: Vec<f64> = pairs.iter().enumerate().map(|(t, (m, o))| {
                        if mask_at(t) { _load(m, *o, ctx.dt_of(u)) } else { alt_at(t) }
                    }).collect();
                    values.insert(u.clone(), Val::S(tv));
                }
            }
            Ops::Cast => {
                match &src_vals[0] {
                    Val::S(vs) => {
                        let r: Vec<f64> = vs.iter().map(|&v| truncate(v, ctx.dt_of(u))).collect();
                        values.insert(u.clone(), Val::S(r));
                    }
                    Val::V(ls) => {
                        let r: Vec<Vec<f64>> = ls.iter().map(|tv| tv.iter().map(|&v| truncate(v, ctx.dt_of(u))).collect()).collect();
                        values.insert(u.clone(), Val::V(r));
                    }
                    _ => panic!("TGERR: interp cast src"),
                };
            }
            Ops::Bitcast => {
                // reinterpret the storage bits: f64 lane -> src_dtype bytes ->
                // dst_dtype value. Covers int<->float (threefry mantissa
                // assembly) on the f64-valued interpreter.
                let src_dt = ctx.dt_of(&srcs[0]);
                let dst_dt = ctx.dt_of(u);
                let bc = |v: f64| -> f64 {
                    let bytes = to_storage_scalar(v, src_dt);
                    from_storage_scalar(&bytes, dst_dt)
                };
                match &src_vals[0] {
                    Val::S(vs) => {
                        let r: Vec<f64> = vs.iter().map(|&v| bc(v)).collect();
                        values.insert(u.clone(), Val::S(r));
                    }
                    Val::V(ls) => {
                        let r: Vec<Vec<f64>> = ls.iter().map(|tv| tv.iter().map(|&v| bc(v)).collect()).collect();
                        values.insert(u.clone(), Val::V(r));
                    }
                    _ => panic!("TGERR: interp bitcast src"),
                };
            }
            Ops::Stack => {
                // values[u] = src_values (list of src values)
                let mut lanes: Vec<Vec<f64>> = Vec::new();
                for (k, sv_) in src_vals.iter().enumerate() {
                    match sv_ {
                        Val::S(vs) => lanes.push(vs.clone()),
                        Val::V(l) => {
                            // a stacked lane value: each src contributes its
                            // per-thread lane list flattened at the same width
                            let width = l.first().map(|t| t.len()).unwrap_or(1);
                            for j in 0..width {
                                lanes.push(l.iter().map(|t| t.get(j).cloned().unwrap_or(0.0)).collect());
                            }
                        }
                        // a stacked VIEW (coalesce folds the load into its
                        // index): the lanes are the memory values it addresses
                        Val::P(ps) => {
                            let dt = ctx.dt_of(&srcs[k]);
                            lanes.push(ps.iter().map(|(m, o)| _load(m, *o, dt)).collect());
                        }
                        _ => panic!("TGERR: interp STACK src node {} src_ops={:?}", u.id(), srcs.iter().map(|x| (x.id(), ctx.op_of(x), ctx.dt_of(x).attr())).collect::<Vec<_>>()),
                    }
                }
                values.insert(u.clone(), Val::V(lanes));
            }
            _ if group::in_(group::ALU, op) => {
                if src_vals.len() == 1 {
                    match &src_vals[0] {
                        Val::S(a) => {
                            let r: Vec<f64> = a.iter().map(|&x| truncate(exec_alu(op, x, 0.0), ctx.dt_of(u))).collect();
                            values.insert(u.clone(), Val::S(r));
                        }
                        Val::V(a) => {
                            let r: Vec<Vec<f64>> = a.iter().map(|tv| tv.iter().map(|&x| truncate(exec_alu(op, x, 0.0), ctx.dt_of(u))).collect()).collect();
                            values.insert(u.clone(), Val::V(r));
                        }
                        _ => panic!("TGERR: interp unary alu src"),
                    }
                    i += 1;
                    continue;
                }
                if src_vals.len() == 3 {
                    let sv = |v: &Val| match v {
                        Val::S(vs) => vs.clone(),
                        Val::V(l) => l.iter().map(|t| t.first().cloned().unwrap_or(0.0)).collect(),
                        _ => panic!("TGERR: interp ternary alu src"),
                    };
                    let (a, b, c) = (sv(&src_vals[0]), sv(&src_vals[1]), sv(&src_vals[2]));
                    let n = a.len().max(b.len()).max(c.len());
                    let r: Vec<f64> = (0..n)
                        .map(|k| truncate(exec_alu_ternary(op, a[k.min(a.len() - 1)], b[k.min(b.len() - 1)], c[k.min(c.len() - 1)]), ctx.dt_of(u)))
                        .collect();
                    values.insert(u.clone(), Val::S(r));
                    i += 1;
                    continue;
                }
                match (&src_vals[0], &src_vals[1]) {
                    (Val::S(a), Val::S(b)) => {
                        let r: Vec<f64> = a.iter().zip(b.iter()).map(|(&x, &y)| truncate(exec_alu(op, x, y), ctx.dt_of(u))).collect();
                        values.insert(u.clone(), Val::S(r));
                    }
                    (Val::V(a), Val::S(b)) | (Val::S(b), Val::V(a)) => {
                        let r: Vec<Vec<f64>> = a.iter().map(|tv| tv.iter().map(|&x| truncate(exec_alu(op, x, b[0]), ctx.dt_of(u))).collect()).collect();
                        values.insert(u.clone(), Val::V(r));
                    }
                    (Val::V(a), Val::V(b)) => {
                        let r: Vec<Vec<f64>> = a.iter().zip(b.iter()).map(|(ta, tb)| ta.iter().zip(tb.iter()).map(|(&x, &y)| truncate(exec_alu(op, x, y), ctx.dt_of(u))).collect()).collect();
                        values.insert(u.clone(), Val::V(r));
                    }
                    _ => panic!("TGERR: interp alu srcs"),
                };
            }
            other => panic!("TGERR: interp op {other:?} not ported"),
        }
        i += 1;
    }
}

fn loop_end_index(ctx: &UOpCtx, uops: &[UopId], range: &UopId) -> usize {
    uops.iter()
        .position(|u| ctx.op_of(u) == Ops::End && ctx.srcs(u).len() >= 2 && ctx.srcs(u)[1] == *range)
        .unwrap_or_else(|| panic!("TGERR: interp missing END for RANGE"))
}

fn loop_start_index(uops: &[UopId], range: &UopId) -> usize {
    uops.iter().position(|u| u == range).unwrap_or_else(|| panic!("TGERR: interp missing RANGE start"))
}

/// `ProgramInfo.from_sink` SPECIAL scan: derive (global, local) extents.
pub fn derive_dims(ctx: &mut UOpCtx, uops: &[UopId]) -> ([usize; 3], [usize; 3]) {
    let mut global = [1usize; 3];
    let mut local = [1usize; 3];
    for u in uops {
        if ctx.op_of(u) == Ops::Special {
            let name = match ctx.arg_of(u) {
                Arg::Str(s) => s.clone(),
                _ => continue,
            };
            let ext = match ctx.ssimplify(ctx.srcs(u)[0].clone()) {
                Ok(crate::uop::sint::SVal::I(v)) => v as usize,
                _ => panic!("TGERR: interp SPECIAL extent"),
            };
            let dim: usize = name[name.len() - 1..].parse().unwrap_or(0);
            if name.starts_with('g') {
                global[2 - dim] = ext;
            } else {
                local[2 - dim] = ext;
            }
        }
    }
    (global, local)
}

/// `PythonProgram.__call__` — derive launch dims from the SPECIAL extents.
pub fn exec_kernel(
    ctx: &mut UOpCtx,
    uops: &[UopId],
    views: &std::collections::HashMap<i64, MemView>,
    vals: &[f64],
) {
    let (global, local) = derive_dims(ctx, uops);
    exec_kernel_with_dims(ctx, uops, views, vals, global, Some(local));
}

/// `PythonProgram.__call__` with explicit launch dims (the Program seam).
pub fn exec_kernel_with_dims(
    ctx: &mut UOpCtx,
    uops: &[UopId],
    views: &std::collections::HashMap<i64, MemView>,
    vals: &[f64],
    global: [usize; 3],
    local: Option<[usize; 3]>,
) {
    let local = local.unwrap_or([1usize; 3]);
    #[cfg(feature = "dump")]
    if std::env::var("TG_DUMP_INTERP").is_ok() {
        let pname = crate::codegen::postrange::kernel_name(ctx, uops);
        eprintln!("INT### BEGIN name={}", pname);
        for u in uops {
            let srcs: Vec<String> = ctx.srcs(u).iter().map(|s| s.id().to_string()).collect();
            eprintln!("INT {} {:?}({}) as={:?}", u.id(), ctx.op_of(u), srcs.join(","), ctx.addrspace_of(u.clone()));
        }
        eprintln!("INT### END name={}", pname);
    }
    let mut numels: std::collections::HashMap<UopId, usize> = std::collections::HashMap::new();
    for u in uops {
        match ctx.op_of(u) {
            Ops::Load | Ops::Store | Ops::Buffer | Ops::Param => { numels.insert(u.clone(), ctx.max_numel(u.clone()) as usize); }
            Ops::Range => {
                let ext = match ctx.ssimplify(ctx.srcs(u)[0].clone()) {
                    Ok(crate::uop::sint::SVal::I(v)) => v.max(1) as usize,
                    _ => panic!("TGERR: interp RANGE extent"),
                };
                numels.insert(u.clone(), ext);
            }
            Ops::Special => {
                let ext = match ctx.ssimplify(ctx.srcs(u)[0].clone()) {
                    Ok(crate::uop::sint::SVal::I(v)) => v.max(1) as usize,
                    _ => 1,
                };
                numels.insert(u.clone(), ext);
            }
            _ => {}
        }
    }
    // python launches global reversed-product; corpus global=(1,1,1) except gidx kernels
    let mut order = Vec::new();
    for c in 0..global[2] {
        for b in 0..global[1] {
            for a in 0..global[0] {
                order.push([a, b, c]);
            }
        }
    }
    if order.is_empty() {
        order.push([0, 0, 0]);
    }
    for g in order {
        exec_launch(ctx, &numels, uops, g, local, views, vals);
    }
}
