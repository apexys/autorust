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
pub struct Buf { pub bytes: Vec<u8> }

/// A memoryview: (buffer, byte offset, element size in bytes, element count).
#[derive(Clone)]
pub struct MemView(pub Rc<RefCell<Buf>>, pub usize, pub usize, pub usize);

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
        d if d == dtype::FLOAT16 => panic!("TGERR: f16 storage scalar not ported"),
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
        d if d == dtype::FLOAT16 => panic!("TGERR: f16 storage scalar not ported"),
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
fn exec_alu(op: Ops, a: f64, b: f64) -> f64 {
    match op {
        Ops::Add => a + b,
        Ops::Mul => a * b,
        Ops::Sub => a - b,
        Ops::Fdiv => a / b,
        Ops::Max => a.max(b),
        _ => panic!("TGERR: interp ALU {op:?} not ported"),
    }
}

/// `truncate` (ops.py:1278) — round/clamp to the dtype's representable values.
fn truncate(v: f64, dt: DTypeId) -> f64 {
    if dt == dtype::FLOAT64 { v }
    else if dt == dtype::FLOAT32 { (v as f32) as f64 }
    else if dt == dtype::FLOAT16 { panic!("TGERR: f16 truncate not ported") }
    else if dt == dtype::BOOL { (v != 0.0) as i64 as f64 }
    else {
        let i = v as i64;
        let (lo, hi) = match (dtype::dtype_min(dt), dtype::dtype_max(dt)) {
            (ConstVal::Int(l), ConstVal::Int(h)) => (l as i64, h as i64),
            _ => (i64::MIN, i64::MAX),
        };
        i.clamp(lo, hi) as f64
    }
}

fn _load(m: &MemView, i: usize, dt: DTypeId) -> f64 {
    if i >= m.3 { panic!("TGERR: load out of bounds, size is {} and access is {i}", m.3); }
    let boff = m.1 + i * m.2;
    let bytes: Vec<u8> = (0..m.2).map(|k| m.0.borrow().bytes[boff + k]).collect();
    from_storage_scalar(&bytes, dt)
}

fn _store(m: &MemView, i: usize, v: f64, dt: DTypeId) {
    if i >= m.3 { panic!("TGERR: store out of bounds, size is {}, access is {i}", m.3); }
    // python packs sub-dword views; the corpus stores at matching itemsize
    assert!(m.2 >= dt.itemsize(), "TGERR: sub-dword store not ported");
    let boff = m.1 + i * m.2;
    let sval = to_storage_scalar(v, dt);
    let mut b = m.0.borrow_mut();
    for (k, x) in sval.iter().enumerate() { b.bytes[boff + k] = *x; }
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
    let mut pbufs: std::collections::VecDeque<MemView> = std::collections::VecDeque::new();
    let mut pvals: std::collections::VecDeque<f64> = vals.iter().copied().collect();
    let mut exec_masks: Vec<Vec<bool>> = vec![vec![true; wsize]];

    let sv = |values: &std::collections::HashMap<UopId, Val>, u: &UopId| -> Vec<f64> {
        match &values[u] {
            Val::S(v) => v.clone(),
            Val::V(l) => l.iter().map(|t| t.first().copied().unwrap_or(0.0)).collect(),
            _ => panic!("TGERR: interp src not scalar"),
        }
    };

    let mut i = 0usize;
    while i < uops.len() {
        let u = uops[i];
        let op = ctx.op_of(u);
        let srcs: Vec<UopId> = ctx.srcs(u).to_vec();
        let src_vals: Vec<Val> = srcs.iter().filter(|s| ctx.dt_of(**s) != dtype::VOID).map(|s| values.get(s).cloned().expect("TGERR: interp missing value")).collect();
        match op {
            Ops::End => {
                if srcs.len() == 3 {
                    let cond = sv(&values, &srcs[2]);
                    if cond.first().copied().unwrap_or(0.0) != 0.0 {
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
                let ext = numels.get(&u).copied().unwrap_or(1);
                match values.get_mut(&u) {
                    None => { values.insert(u, Val::S(vec![0.0; wsize])); }
                    Some(Val::S(vs)) => {
                        for v in vs.iter_mut() { *v += 1.0; }
                        if vs[0] == ext as f64 {
                            values.remove(&u);
                            i = loop_end_index(ctx, uops, &u) + 1;
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
            Ops::Barrier | Ops::Sink | Ops::Noop | Ops::Group | Ops::After => {}
            Ops::Store => {
                let gate = exec_masks.last().unwrap().clone();
                let store_n = numels.get(&u).copied().unwrap_or(1);
                let vals_lane: Vec<Vec<f64>> = match &src_vals[1] {
                    Val::S(vs) if store_n == 1 => vec![vs.clone()],
                    Val::V(lanes) => lanes.clone(),
                    _ => panic!("TGERR: interp store value"),
                };
                let pairs: &Vec<(MemView, usize)> = match &src_vals[0] {
                    Val::P(p) => p,
                    _ => panic!("TGERR: interp store src"),
                };
                for (j, val) in vals_lane.iter().enumerate() {
                    for (t, ((m, o), v)) in pairs.iter().zip(val.iter()).enumerate() {
                        if gate[t] { _store(m, *o + j, *v, ctx.dt_of(srcs[1])); }
                    }
                }
            }
            Ops::Param | Ops::Buffer => {
                if ctx.addrspace_of(u) == Some(AddrSpace::Alu) && op == Ops::Param {
                    let v = pvals.pop_front().unwrap_or(0.0);
                    values.insert(u, Val::S(vec![v; wsize]));
                } else if op == Ops::Param {
                    let slot = match ctx.arg_of(u) {
                        Arg::Param(p) => p.slot,
                        _ => panic!("TGERR: interp PARAM arg"),
                    };
                    let mv = views.get(&slot).unwrap_or_else(|| panic!("TGERR: interp missing view for slot {slot}")).clone();
                    values.insert(u, Val::M(vec![mv; wsize]));
                } else if ctx.addrspace_of(u) == Some(AddrSpace::Reg) {
                    // REGs are per thread
                    let nel = numels.get(&u).copied().unwrap_or(1);
                    let isz = ctx.dt_of(u).itemsize();
                    values.insert(u, Val::M((0..wsize).map(|_| MemView(Rc::new(RefCell::new(Buf { bytes: vec![0u8; nel * isz] })), 0, isz, nel)).collect()));
                } else {
                    let nel = numels.get(&u).copied().unwrap_or(1);
                    let isz = ctx.dt_of(u).itemsize();
                    let mv = MemView(Rc::new(RefCell::new(Buf { bytes: vec![0u8; nel * isz] })), 0, isz, nel);
                    values.insert(u, Val::M(vec![mv; wsize]));
                }
            }
            Ops::Special => {
                let name = match ctx.arg_of(u) {
                    Arg::Str(s) => s.clone(),
                    _ => panic!("TGERR: interp SPECIAL arg"),
                };
                if name.starts_with('g') {
                    let dim: usize = name[name.len() - 1..].parse().unwrap();
                    values.insert(u, Val::S(vec![gidx[2 - dim] as f64; wsize]));
                } else {
                    let dim: usize = name[name.len() - 1..].parse().unwrap();
                    values.insert(u, Val::S(warp.iter().map(|w| w[2 - dim] as f64).collect()));
                }
            }
            Ops::Const => {
                let v = match ctx.val(u) {
                    ConstVal::Int(x) => x as f64,
                    ConstVal::Float(x) => x,
                    ConstVal::Bool(b) => b as i64 as f64,
                    other => panic!("TGERR: interp const {other:?}"),
                };
                values.insert(u, Val::S(vec![v; wsize]));
            }
            Ops::Index | Ops::Shrink => {
                // (view, element offset) pairs
                if ctx.addrspace_derived(srcs[0]) == Some(AddrSpace::Alu) {
                    // ALU-storage index over a vector value: [lane][thread] pick
                    // python: ret = [src_values[0][i][t] for t, i in enumerate(src_values[1])]
                    let ret: Vec<f64> = match (&src_vals[0], &src_vals[1]) {
                        (Val::V(a), Val::S(b)) => (0..wsize).map(|t| a[b[t.min(b.len() - 1)] as usize][t.min(a[0].len() - 1)]).collect(),
                        (Val::V(a), Val::V(b)) => (0..wsize).map(|t| a[b[t.min(b.len() - 1)][0] as usize][t.min(a[0].len() - 1)]).collect(),
                        (Val::S(a), _) => a.clone(),
                        _ => panic!("TGERR: interp ALU index srcs"),
                    };
                    values.insert(u, Val::S(ret));
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
                            Val::V(l) => l.iter().map(|t| t.first().copied().unwrap_or(0.0) as usize).collect::<Vec<usize>>(),
                            _ => unreachable!(),
                        };
                        let ms: Vec<MemView> = match &src_vals[0] {
                            Val::M(ms) => ms.clone(),
                            Val::P(ps) => ps.iter().map(|(m, _)| m.clone()).collect(),
                            _ => unreachable!(),
                        };
                        for (t, m) in ms.iter().enumerate() { ret.push((m.clone(), os[t.min(os.len() - 1)])); }
                    }
                    _ => panic!("TGERR: interp index srcs"),
                }
                values.insert(u, Val::P(ret));
            }
            Ops::Load => {
                let load_sz = numels.get(&u).copied().unwrap_or(1);
                let pairs: Vec<(MemView, usize)> = match &src_vals[0] {
                    Val::P(p) => p.clone(),
                    _ => panic!("TGERR: interp load src"),
                };
                if load_sz > 1 {
                    // lane-major vector load
                    let lanes: Vec<Vec<f64>> = (0..load_sz).map(|j| pairs.iter().map(|(m, o)| _load(m, o + j, ctx.dt_of(u))).collect()).collect();
                    values.insert(u, Val::V(lanes));
                } else {
                    let tv: Vec<f64> = pairs.iter().enumerate().map(|(t, (m, o))| {
                        if exec_masks.last().unwrap()[t] { _load(m, *o, ctx.dt_of(u)) } else { 0.0 }
                    }).collect();
                    values.insert(u, Val::S(tv));
                }
            }
            Ops::Cast => {
                match &src_vals[0] {
                    Val::S(vs) => {
                        let r: Vec<f64> = vs.iter().map(|&v| truncate(v, ctx.dt_of(u))).collect();
                        values.insert(u, Val::S(r));
                    }
                    Val::V(ls) => {
                        let r: Vec<Vec<f64>> = ls.iter().map(|tv| tv.iter().map(|&v| truncate(v, ctx.dt_of(u))).collect()).collect();
                        values.insert(u, Val::V(r));
                    }
                    _ => panic!("TGERR: interp cast src"),
                };
            }
            Ops::Stack => {
                // values[u] = src_values (list of src values)
                let mut lanes: Vec<Vec<f64>> = Vec::new();
                for sv_ in &src_vals {
                    match sv_ {
                        Val::S(vs) => lanes.push(vs.clone()),
                        _ => panic!("TGERR: interp STACK src"),
                    }
                }
                values.insert(u, Val::V(lanes));
            }
            _ if group::in_(group::ALU, op) => {
                match (&src_vals[0], &src_vals[1]) {
                    (Val::S(a), Val::S(b)) => {
                        let r: Vec<f64> = a.iter().zip(b.iter()).map(|(&x, &y)| truncate(exec_alu(op, x, y), ctx.dt_of(u))).collect();
                        values.insert(u, Val::S(r));
                    }
                    (Val::V(a), Val::S(b)) | (Val::S(b), Val::V(a)) => {
                        let r: Vec<Vec<f64>> = a.iter().map(|tv| tv.iter().map(|&x| truncate(exec_alu(op, x, b[0]), ctx.dt_of(u))).collect()).collect();
                        values.insert(u, Val::V(r));
                    }
                    (Val::V(a), Val::V(b)) => {
                        let r: Vec<Vec<f64>> = a.iter().zip(b.iter()).map(|(ta, tb)| ta.iter().zip(tb.iter()).map(|(&x, &y)| truncate(exec_alu(op, x, y), ctx.dt_of(u))).collect()).collect();
                        values.insert(u, Val::V(r));
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
        .position(|u| ctx.op_of(*u) == Ops::End && ctx.srcs(*u).len() >= 2 && ctx.srcs(*u)[1] == *range)
        .unwrap_or_else(|| panic!("TGERR: interp missing END for RANGE"))
}

fn loop_start_index(uops: &[UopId], range: &UopId) -> usize {
    uops.iter().position(|u| u == range).unwrap_or_else(|| panic!("TGERR: interp missing RANGE start"))
}

/// `PythonProgram.__call__` — derive launch dims from the SPECIAL extents.
pub fn exec_kernel(
    ctx: &mut UOpCtx,
    uops: &[UopId],
    views: &std::collections::HashMap<i64, MemView>,
    vals: &[f64],
) {
    let mut global = [1usize; 3];
    let mut local = [1usize; 3];
    for &u in uops {
        if ctx.op_of(u) == Ops::Special {
            let name = match ctx.arg_of(u) {
                Arg::Str(s) => s.clone(),
                _ => continue,
            };
            let ext = match ctx.ssimplify(ctx.srcs(u)[0]) {
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
    let mut numels: std::collections::HashMap<UopId, usize> = std::collections::HashMap::new();
    for &u in uops {
        match ctx.op_of(u) {
            Ops::Load | Ops::Store | Ops::Buffer | Ops::Param => { numels.insert(u, ctx.max_numel(u) as usize); }
            Ops::Range => {
                let ext = match ctx.ssimplify(ctx.srcs(u)[0]) {
                    Ok(crate::uop::sint::SVal::I(v)) => v.max(1) as usize,
                    _ => panic!("TGERR: interp RANGE extent"),
                };
                numels.insert(u, ext);
            }
            Ops::Special => {
                let ext = match ctx.ssimplify(ctx.srcs(u)[0]) {
                    Ok(crate::uop::sint::SVal::I(v)) => v.max(1) as usize,
                    _ => 1,
                };
                numels.insert(u, ext);
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
