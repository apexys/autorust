//! The uops interpreter (`runtime/ops_python.py` model) — corpus subset.
//!
//! Executes linearized uops: global/local launch loops from SPECIAL extents,
//! per-thread value lists, (mem, offset) pairs for memory, exec masks, ALU.

use crate::dtype::{self, AddrSpace, ConstVal, DTypeId};
use crate::uop::ctx::UOpCtx as _C;
use crate::uop::arg::Arg;
use crate::uop::ctx::UOpCtx;
use crate::uop::ops::{group, Ops};
use crate::uop::UopId;

/// A memory cell: an f64-backed flat store (corpus dtypes are f32/i32-sized).
pub struct Mem {
    pub data: Vec<f64>,
    pub nbytes: usize,
    pub itemsize: usize,
}

impl Mem {
    fn load(&self, i: usize, dt: DTypeId) -> f64 {
        self.data.get(i).copied().unwrap_or_else(|| panic!("TGERR: load out of bounds"))
    }
    fn store(&mut self, i: usize, v: f64, dt: DTypeId) {
        if i >= self.data.len() {
            panic!("TGERR: store out of bounds");
        }
        self.data[i] = v;
    }
}

#[derive(Clone)]
enum Val {
    /// per-thread scalar
    S(Vec<f64>),
    /// per-(lane, thread) vector (lane-major)
    V(Vec<Vec<f64>>),
    /// (mem, offset) pairs per thread
    P(Vec<(RcMem, usize)>),
    /// buffer backing store per thread
    M(Vec<RcMem>),
}

type RcMem = std::rc::Rc<std::cell::RefCell<Mem>>;

fn mk_mem(nbytes: usize, itemsize: usize) -> RcMem {
    std::rc::Rc::new(std::cell::RefCell::new(Mem { data: vec![0.0; nbytes / itemsize], nbytes, itemsize }))
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

fn truncate(v: f64, dt: DTypeId) -> f64 {
    if dtype::is_float(dt) {
        v
    } else if dt == dtype::BOOL {
        (v != 0.0) as i64 as f64
    } else {
        let i = v as i64;
        let (lo, hi) = match (dtype::dtype_min(dt), dtype::dtype_max(dt)) {
            (ConstVal::Int(l), ConstVal::Int(h)) => (l as i64, h as i64),
            _ => (i64::MIN, i64::MAX),
        };
        i.clamp(lo, hi) as f64
    }
}

/// one (global_idx, local_size) launch invocation.
fn exec_launch(ctx: &UOpCtx, numels: &std::collections::HashMap<UopId, usize>, uops: &[UopId], gidx: [usize; 3], lsize: [usize; 3], bufs: &[RcMem], vals: &[f64], loop_ends: &std::collections::HashMap<UopId, usize>) {
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
    let mut pbufs: std::collections::VecDeque<RcMem> = bufs.iter().cloned().collect();
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
                        i = loop_ends[&srcs[1]];
                        continue;
                    }
                } else {
                    i = loop_ends[&srcs[1]];
                    continue;
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
            Ops::Barrier | Ops::Sink | Ops::Noop | Ops::Group => {}
            Ops::Store => {
                let gate = exec_masks.last().unwrap().clone();
                match (&values[&srcs[0]], &values[&srcs[1]]) {
                    (Val::P(pairs), Val::S(vs)) => {
                        for (t, (m, o)) in pairs.iter().enumerate() {
                            if gate[t] {
                                m.borrow_mut().store(*o, vs[t.min(vs.len() - 1)], ctx.dt_of(srcs[1]));
                            }
                        }
                    }
                    (Val::P(pairs), Val::V(lanes)) => {
                        for (j, lane) in lanes.iter().enumerate() {
                            for (t, (m, o)) in pairs.iter().enumerate() {
                                if gate[t] {
                                    m.borrow_mut().store(o + j, lane[t.min(lane.len() - 1)], ctx.dt_of(srcs[1]));
                                }
                            }
                        }
                    }
                    _ => panic!("TGERR: interp store srcs"),
                }
            }
            Ops::Param | Ops::Buffer => {
                if ctx.addrspace_of(u) == Some(AddrSpace::Alu) {
                    let v = pvals.pop_front().unwrap_or(0.0);
                    values.insert(u, Val::S(vec![v; wsize]));
                } else if ctx.addrspace_of(u) == Some(AddrSpace::Reg) {
                    let m = mk_mem(numels[&u] * ctx.dt_of(u).itemsize(), ctx.dt_of(u).itemsize());
                    values.insert(u, Val::M(vec![m; wsize]));
                } else if op == Ops::Param {
                    let m = pbufs.pop_front().expect("TGERR: interp missing buffer");
                    values.insert(u, Val::M(vec![m; wsize]));
                } else {
                    let m = mk_mem(numels[&u] * ctx.dt_of(u).itemsize(), ctx.dt_of(u).itemsize());
                    values.insert(u, Val::M(vec![m; wsize]));
                }
            }
            Ops::Special => {
                let name = match ctx.arg_of(u) {
                    Arg::Str(s) => s.clone(),
                    _ => panic!("TGERR: interp SPECIAL arg"),
                };
                let ext = numels.get(&u).copied().unwrap_or(1);
                if name.starts_with('g') {
                    let dim: usize = name[name.len() - 1..].parse().unwrap();
                    values.insert(u, Val::S(vec![gidx[2 - dim] as f64; wsize]));
                } else {
                    let dim: usize = name[name.len() - 1..].parse().unwrap();
                    values.insert(u, Val::S(warp.iter().map(|w| w[2 - dim] as f64 % ext.max(1) as f64).collect()));
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
                // (mem, offset) pairs; scale 1 for the corpus (no bitcast views)
                let (sm, so) = match (&src_vals[0], &src_vals[1]) {
                    (Val::M(ms), Val::S(os)) => (ms.iter().map(|m| m.clone()).collect::<Vec<RcMem>>(), os.clone()),
                    (Val::P(ps), Val::S(os)) => (ps.iter().map(|(m, _)| m.clone()).collect(), os.clone()),
                    (Val::V(_), _) | (_, Val::V(_)) => {
                        // vector offset (the SHRINK fan-out under a vec store): keep pairs
                        let os = match &src_vals[1] {
                            Val::V(l) => l.iter().map(|t| t.first().copied().unwrap_or(0.0) as usize).collect::<Vec<usize>>(),
                            _ => unreachable!(),
                        };
                        let ms: Vec<RcMem> = match &src_vals[0] {
                            Val::M(ms) => ms.clone(),
                            Val::P(ps) => ps.iter().map(|(m, _)| m.clone()).collect(),
                            _ => unreachable!(),
                        };
                        let mut ret = Vec::new();
                        for (t, m) in ms.iter().enumerate() {
                            ret.push((m.clone(), os[t.min(os.len() - 1)]));
                        }
                        values.insert(u, Val::P(ret));
                        i += 1;
                        continue;
                    }
                    _ => panic!("TGERR: interp index srcs"),
                };
                let mut ret: Vec<(RcMem, usize)> = Vec::new();
                for (t, m) in sm.iter().enumerate() {
                    ret.push((m.clone(), so[t] as usize));
                }
                values.insert(u, Val::P(ret));
            }
            Ops::Load => {
                let load_sz = numels[&u];
                let (pairs, gate): (Vec<(RcMem, usize)>, Option<Vec<bool>>) = match &src_vals[0] {
                    Val::P(p) => (p.clone(), None),
                    _ => panic!("TGERR: interp load src"),
                };
                if load_sz > 1 {
                    let mut lanes = Vec::with_capacity(load_sz);
                    for j in 0..load_sz {
                        let mut tv = Vec::with_capacity(wsize);
                        for (t, (m, o)) in pairs.iter().enumerate() {
                            tv.push(m.borrow().load(o + j, ctx.dt_of(u)));
                        }
                        let _ = gate;
                        lanes.push(tv);
                    }
                    values.insert(u, Val::V(lanes));
                } else {
                    let mut tv = Vec::with_capacity(wsize);
                    for (t, (m, o)) in pairs.iter().enumerate() {
                        if exec_masks.last().unwrap()[t] {
                            tv.push(m.borrow().load(*o, ctx.dt_of(u)));
                        } else {
                            tv.push(0.0);
                        }
                    }
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

/// `PythonProgram.__call__` — derive launch dims from the SPECIAL extents.
pub fn exec_kernel(ctx: &mut UOpCtx, uops: &[UopId], bufs: &[RcMem], vals: &[f64]) {
    let mut global = [1usize; 3];
    let mut local = [1usize; 3];
    for &u in uops {
        if ctx.op_of(u) == Ops::Special {
            let name = match ctx.arg_of(u) {
                Arg::Str(s) => s.clone(),
                _ => continue,
            };
            let ext = match ctx.ssimplify(ctx.srcs(u)[0]) {
                Ok(crate::uop::sint::SVal::I(v)) => v.max(1) as usize,
                _ => 1,
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
        if matches!(ctx.op_of(u), Ops::Load | Ops::Buffer | Ops::Param | Ops::Special) {
            let ext = match ctx.ssimplify(ctx.srcs(u)[0]) {
                Ok(crate::uop::sint::SVal::I(v)) => v.max(1) as usize,
                _ => 1,
            };
            numels.insert(u, if ctx.op_of(u) == Ops::Special { ext } else { ctx.max_numel(u) as usize });
        }
    }
    let loop_ends: std::collections::HashMap<UopId, usize> = uops
        .iter()
        .enumerate()
        .filter(|(_, u)| ctx.op_of(**u) == Ops::End && ctx.srcs(**u).len() >= 2)
        .map(|(i, u)| (ctx.srcs(*u)[1], i))
        .collect();
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
        exec_launch(ctx, &numels, uops, g, local, bufs, vals, &loop_ends);
    }
}
