//! Typed (dtype-native) executor for `ops_rayon` — R2 of `CURRENT_STATE.md` §9.8.
//!
//! `interp::exec_launch` re-expressed over [`ray_value::Bytes`] instead of `f64`
//! lanes: register values are exact bytes at the node dtype, ALU is
//! monomorphized per dtype, loads/stores are byte copies (no per-element
//! conversion), and `uint64` is exact. Control flow (RANGE/END/IF, SPECIAL,
//! STACK, devectorized stores) mirrors the `f64` executor in `ray_kernel.rs`
//! one-for-one, so numerics match `INTERP`.
//!
//! `ops_rayon` layers typed → `f64` → `INTERP`, so unsupported dtypes (sub-word
//! floats, size-changing bitcasts) fall back without losing correctness.

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};

use crate::dtype::{self, AddrSpace, ConstVal, DTypeId};
use crate::runtime::interp::{from_storage_scalar, to_storage_scalar};
use crate::runtime::ray_kernel::{RayKernel, SendMemView};
use crate::runtime::ray_value::{self as rv, Bytes};
use crate::uop::arg::Arg;
use crate::uop::ops::{group, Ops};
use crate::uop::UopId;

type Pair = (SendMemView, usize);

/// A per-instruction register value.
enum Slot {
    Empty,
    /// lane-major typed elements: `lanes * wsize`, element `(l,t)` at
    /// `(l*wsize + t) * dt.itemsize()`.
    Data { buf: Bytes, dt: DTypeId, lanes: usize },
    P(Vec<Pair>),
    M(Vec<SendMemView>),
}

/// Reusable per-thread scratch: one slot per instruction plus a pool of value
/// buffers. Recycled per launch so a rayon worker running many launches
/// allocates only once.
#[derive(Default)]
struct Scratch {
    slots: Vec<Slot>,
    pool: Vec<Bytes>,
    owned: Vec<Box<[u8]>>,
}

impl Scratch {
    fn recycle(&mut self, n: usize) {
        let Scratch { slots, pool, owned } = self;
        for slot in slots.iter_mut() {
            if let Slot::Data { buf, .. } = slot {
                pool.push(std::mem::take(buf));
            }
            *slot = Slot::Empty;
        }
        owned.clear();
        if slots.len() < n {
            slots.resize_with(n, || Slot::Empty);
        }
    }

    fn put_data(&mut self, i: usize, dt: DTypeId, lanes: usize, wsize: usize) {
        let nbytes = lanes * wsize * dt.itemsize();
        if let Slot::Data { buf, dt: ddt, lanes: dlanes } = &mut self.slots[i] {
            if buf.nbytes() != nbytes {
                buf.reset(nbytes);
                buf.as_bytes_mut().fill(0);
            }
            *ddt = dt;
            *dlanes = lanes;
            return;
        }
        let mut b = self.pool.pop().unwrap_or_default();
        b.reset(nbytes);
        b.as_bytes_mut().fill(0);
        self.slots[i] = Slot::Data { buf: b, dt, lanes };
    }

    #[inline]
    fn data(&self, i: usize) -> &Bytes {
        match &self.slots[i] {
            Slot::Data { buf, .. } => buf,
            _ => panic!("TGERR: typed expected data at instr {i}"),
        }
    }

    #[inline]
    fn meta(&self, i: usize) -> (DTypeId, usize) {
        match &self.slots[i] {
            Slot::Data { dt, lanes, .. } => (*dt, *lanes),
            _ => panic!("TGERR: typed expected data at instr {i}"),
        }
    }
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
}

/// The typed executor.
pub struct TypedKernel {
    kernel: RayKernel,
}

impl TypedKernel {
    pub fn compile(ctx: &mut crate::uop::ctx::UOpCtx, uops: &[UopId]) -> Option<TypedKernel> {
        let kernel = RayKernel::compile(ctx, uops)?;
        // Commit weak dtypes to concrete storage before the support scan.
        let kernel = kernel.remap_dtypes(rv::storage_dtype);
        for ins in kernel.instrs() {
            if ins.dt != dtype::VOID && !rv::is_native_dtype(ins.dt) {
                return None;
            }
            for &s in &ins.srcs {
                let d = kernel.instrs()[s].dt;
                if d != dtype::VOID && !rv::is_native_dtype(d) {
                    return None;
                }
            }
            if ins.op == Ops::Bitcast {
                let sd = kernel.instrs()[ins.srcs[0]].dt;
                if sd.itemsize() != ins.dt.itemsize() {
                    return None;
                }
            }
        }
        Some(TypedKernel { kernel })
    }

    pub fn globals(&self) -> &[i64] {
        &self.kernel.globals
    }
    pub fn sig_dtypes(&self) -> &HashMap<i64, DTypeId> {
        &self.kernel.sig_dtypes
    }

    pub fn exec(&self, views: &HashMap<i64, SendMemView>, global: [usize; 3], local: Option<[usize; 3]>, vals: &[f64], parallel: bool) {
        let local = local.unwrap_or([1usize; 3]);
        let mut order: Vec<[usize; 3]> = Vec::new();
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
        let n = self.kernel.instrs().len();
        if parallel && order.len() > 1 {
            use rayon::prelude::*;
            order.par_iter().for_each(|g| {
                SCRATCH.with(|s| {
                    let mut s = s.borrow_mut();
                    s.recycle(n);
                    self.exec_launch(*g, local, views, vals, &mut s);
                });
            });
        } else {
            SCRATCH.with(|s| {
                let mut s = s.borrow_mut();
                for g in order {
                    s.recycle(n);
                    self.exec_launch(g, local, views, vals, &mut s);
                }
            });
        }
    }

    fn exec_launch(&self, gidx: [usize; 3], lsize: [usize; 3], views: &HashMap<i64, SendMemView>, vals: &[f64], sc: &mut Scratch) {
        let instrs = self.kernel.instrs();
        let mut warp: Vec<[usize; 3]> = Vec::new();
        for c in 0..lsize[2] {
            for b in 0..lsize[1] {
                for a in 0..lsize[0] {
                    warp.push([a, b, c]);
                }
            }
        }
        let wsize = warp.len().max(1);
        let mut pvals: VecDeque<f64> = vals.iter().cloned().collect();
        let mut masks: Vec<Vec<bool>> = vec![vec![true; wsize]];

        let isz = |dt: DTypeId| dt.itemsize();
        let op = |i: usize| instrs[i].op;
        let dt = |i: usize| instrs[i].dt;
        let srcs_of = |i: usize| instrs[i].srcs.clone();

        let mut i = 0usize;
        while i < instrs.len() {
            let o = op(i);
            let srcs = srcs_of(i);
            let real: Vec<usize> = srcs.iter().cloned().filter(|&s| dt(s) != dtype::VOID).collect();
            match o {
                Ops::End => {
                    if srcs.len() == 3 {
                        let (cdt, _) = sc.meta(srcs[2]);
                        if rv::truthy(cdt, sc.data(srcs[2]), 0) {
                            i = srcs[1];
                            continue;
                        }
                    } else {
                        i = srcs[1];
                        continue;
                    }
                }
                Ops::Range => {
                    if dt(i) == dtype::VOID {
                        i += 1;
                        continue;
                    }
                    let ext = instrs[i].numel;
                    if !matches!(sc.slots[i], Slot::Data { .. }) {
                        // first encounter: initialize to 0, do not increment
                        sc.put_data(i, dt(i), 1, wsize);
                    } else {
                        rv::add_i128(dt(i), sc.data_mut(i), 1);
                        if rv::get_i128(dt(i), sc.data(i), 0) == ext as i128 {
                            sc.slots[i] = Slot::Empty;
                            i = self.loop_end_index(&srcs, i) + 1;
                            continue;
                        }
                    }
                }
                Ops::If => {
                    let (cdt, cl) = sc.meta(srcs[0]);
                    let n = cl * wsize;
                    let last = masks.last().unwrap().clone();
                    let condv: Vec<bool> = (0..wsize)
                        .map(|t| rv::truthy(cdt, sc.data(srcs[0]), lane_elem(cl, wsize, t).min(n.saturating_sub(1))))
                        .collect();
                    masks.push(last.iter().zip(condv.iter().chain(std::iter::repeat(&condv[0]))).map(|(a, b)| *a && *b).collect());
                }
                Ops::Endif => {
                    masks.pop();
                }
                Ops::After => {
                    let s = *real.first().expect("TGERR: typed AFTER");
                    copy_slot(sc, i, s);
                }
                Ops::Barrier | Ops::Sink | Ops::Noop | Ops::Group => {}
                Ops::Store => self.store(sc, i, &srcs, &real, wsize, &masks),
                Ops::Param | Ops::Buffer => {
                    let is_alu = instrs[i].addrspace == Some(AddrSpace::Alu);
                    if is_alu && o == Ops::Param {
                        let v = pvals.pop_front().unwrap_or(0.0);
                        sc.put_data(i, dt(i), 1, wsize);
                        rv::fill_f64(dt(i), sc.data_mut(i), v);
                    } else if o == Ops::Param {
                        let slot = match &instrs[i].arg {
                            Arg::Param(p) => p.slot,
                            _ => panic!("TGERR: typed PARAM arg"),
                        };
                        let mv = *views.get(&slot).unwrap_or_else(|| panic!("TGERR: typed missing view slot {slot}"));
                        sc.slots[i] = Slot::M(vec![mv; wsize]);
                    } else if instrs[i].addrspace == Some(AddrSpace::Reg) {
                        let (nel, s) = (instrs[i].numel, isz(dt(i)));
                        let vs: Vec<SendMemView> =
                            (0..wsize).map(|_| SendMemView::from_owned(vec![0u8; nel * s], s, nel, &mut sc.owned)).collect();
                        sc.slots[i] = Slot::M(vs);
                    } else {
                        let (nel, s) = (instrs[i].numel, isz(dt(i)));
                        let mv = SendMemView::from_owned(vec![0u8; nel * s], s, nel, &mut sc.owned);
                        sc.slots[i] = Slot::M(vec![mv; wsize]);
                    }
                }
                Ops::Special => {
                    let name = match &instrs[i].arg {
                        Arg::Str(s) => s.clone(),
                        _ => panic!("TGERR: typed SPECIAL arg"),
                    };
                    sc.put_data(i, dt(i), 1, wsize);
                    let dim: usize = name[name.len() - 1..].parse().unwrap();
                    if name.starts_with('g') {
                        rv::fill_i128(dt(i), sc.data_mut(i), gidx[2 - dim] as i128);
                    } else {
                        for (t, w) in warp.iter().enumerate() {
                            rv::set_i128(dt(i), sc.data_mut(i), t, w[2 - dim] as i128);
                        }
                    }
                }
                Ops::Const => {
                    sc.put_data(i, dt(i), 1, wsize);
                    match instrs[i].const_val.as_ref().expect("TGERR: typed const") {
                        ConstVal::Int(x) => rv::fill_i128(dt(i), sc.data_mut(i), *x),
                        ConstVal::Float(x) => rv::fill_f64(dt(i), sc.data_mut(i), *x),
                        ConstVal::Bool(b) => rv::fill_i128(dt(i), sc.data_mut(i), *b as i128),
                        other => panic!("TGERR: typed const {other:?}"),
                    }
                }
                Ops::Index | Ops::Shrink => {
                    if instrs[srcs[0]].addrspace_derived == Some(AddrSpace::Alu) || matches!(sc.slots[srcs[0]], Slot::Data { .. }) {
                        self.index_data(sc, i, &srcs, wsize);
                    } else {
                        self.index_pairs(sc, i, &srcs, wsize);
                    }
                }
                Ops::Load => self.load(sc, i, &srcs, &real, wsize, &masks),
                Ops::Cast => {
                    let (sdt, lanes) = sc.meta(srcs[0]);
                    sc.put_data(i, dt(i), lanes, wsize);
                    let src = sc.data(srcs[0]).clone();
                    assert!(rv::cast(sdt, dt(i), sc.data_mut(i), &src), "TGERR: typed CAST failed");
                }
                Ops::Bitcast => {
                    let (sdt, lanes) = sc.meta(srcs[0]);
                    debug_assert_eq!(sdt.itemsize(), dt(i).itemsize());
                    sc.put_data(i, dt(i), lanes, wsize);
                    let src = sc.data(srcs[0]).as_bytes().to_vec();
                    sc.data_mut(i).copy_from(&src);
                }
                Ops::Stack => self.stack(sc, i, &srcs, &real, wsize),
                _ if group::in_(group::ALU, o) => self.alu(sc, i, o, &srcs, &real, wsize),
                other => panic!("TGERR: typed op {other:?} not ported"),
            }
            i += 1;
        }
    }

    fn store(&self, sc: &mut Scratch, i: usize, srcs: &[usize], real: &[usize], wsize: usize, masks: &[Vec<bool>]) {
        let instrs = self.kernel.instrs();
        let gate = masks.last().unwrap();
        let store_n = instrs[i].numel;
        let (val_dt, val_lanes) = sc.meta(srcs[1]);
        let store_dt = instrs[instrs[srcs[0]].srcs[0]].dt;
        let val_isz = val_dt.itemsize();
        let store_isz = store_dt.itemsize();
        assert_eq!(val_isz, store_isz, "TGERR: typed store dtype mismatch");
        let lanes = val_lanes.max(1);
        let val = sc.data(srcs[1]).as_bytes().to_vec();
        let gate_at = |t: usize| gate[t.min(gate.len() - 1)];

        if instrs[srcs[0]].op == Ops::Stack && val_lanes == 1 {
            let dest_srcs = instrs[srcs[0]].srcs.clone();
            for j in 0..lanes {
                let dj = dest_srcs.get(j).copied().unwrap_or(dest_srcs[0]);
                let dpairs: Vec<Pair> = match &sc.slots[dj] {
                    Slot::P(p) => p.clone(),
                    _ => continue,
                };
                for (t, (m, oo)) in dpairs.iter().enumerate() {
                    if gate_at(t) {
                        write_mem(m, *oo, val_elem(&val, wsize, val_isz, j, t), val_isz);
                    }
                }
            }
            return;
        }
        let _ = real;
        let _ = store_n;
        let pairs: Vec<Pair> = match &sc.slots[srcs[0]] {
            Slot::P(p) => p.clone(),
            _ => panic!("TGERR: typed store src not pairs"),
        };
        for j in 0..lanes {
            for (t, (m, oo)) in pairs.iter().enumerate() {
                if gate_at(t) {
                    write_mem(m, *oo + j, val_elem(&val, wsize, val_isz, j, t), val_isz);
                }
            }
        }
    }

    fn index_data(&self, sc: &mut Scratch, i: usize, srcs: &[usize], wsize: usize) {
        let (a_dt, a_lanes) = sc.meta(srcs[0]);
        let (b_dt, b_lanes) = sc.meta(srcs[1]);
        let isz = a_dt.itemsize();
        let src = sc.data(srcs[0]).as_bytes().to_vec();
        let idx: Vec<usize> = (0..wsize)
            .map(|t| rv::get_i128(b_dt, sc.data(srcs[1]), lane_elem(b_lanes, wsize, t)).max(0) as usize)
            .collect();
        sc.put_data(i, a_dt, 1, wsize);
        for (t, &ix) in idx.iter().enumerate() {
            let alane = if a_lanes > 1 { ix.min(a_lanes - 1) } else { 0 };
            let so = (alane * wsize + t) * isz;
            let d = t * isz;
            sc.data_mut(i).as_bytes_mut()[d..d + isz].copy_from_slice(&src[so..so + isz]);
        }
    }

    fn index_pairs(&self, sc: &mut Scratch, i: usize, srcs: &[usize], wsize: usize) {
        let instrs = self.kernel.instrs();
        // (M|P) indexed by a scalar or per-thread offset value.
        let (ms, base): (Vec<SendMemView>, Vec<usize>) = match &sc.slots[srcs[0]] {
            Slot::M(m) => (m.clone(), vec![0; m.len()]),
            Slot::P(p) => (p.iter().map(|(m, _)| *m).collect(), p.iter().map(|(_, o)| *o).collect()),
            _ => panic!("TGERR: typed index_pairs src0"),
        };
        let b_dt = instrs[srcs[1]].dt;
        let b_lanes = match &sc.slots[srcs[1]] {
            Slot::Data { lanes, .. } => *lanes,
            _ => 1,
        };
        let n = sc.data(srcs[1]).nbytes() / b_dt.itemsize();
        let mut ret = Vec::with_capacity(ms.len());
        for (t, m) in ms.iter().enumerate() {
            let be = lane_elem(b_lanes, wsize, t);
            let o = rv::get_i128(b_dt, sc.data(srcs[1]), be.min(n.saturating_sub(1))).max(0) as usize;
            ret.push((*m, base[t.min(base.len() - 1)] + o));
        }
        sc.slots[i] = Slot::P(ret);
    }

    fn load(&self, sc: &mut Scratch, i: usize, srcs: &[usize], real: &[usize], wsize: usize, masks: &[Vec<bool>]) {
        let instrs = self.kernel.instrs();
        let load_n = instrs[i].numel.max(1);
        let dtv = instrs[i].dt;
        let isz = dtv.itemsize();
        let pairs: Vec<Pair> = match &sc.slots[srcs[0]] {
            Slot::P(p) => p.clone(),
            _ => panic!("TGERR: typed load src not pairs"),
        };
        let gated = real.len() >= 3;
        let alt = if gated { sc.data(srcs[1]).as_bytes().to_vec() } else { Vec::new() };
        let alt_dt = if gated { instrs[srcs[1]].dt } else { dtv };
        let gate_dt = if gated { instrs[srcs[2]].dt } else { dtype::BOOL };
        let gate_lanes = if gated {
            match &sc.slots[srcs[2]] {
                Slot::Data { lanes, .. } => *lanes,
                _ => 1,
            }
        } else {
            1
        };
        let gate_n = if gated { sc.data(srcs[2]).nbytes() / gate_dt.itemsize() } else { 0 };
        let gatev: Vec<bool> = if gated {
            (0..wsize)
                .map(|t| rv::truthy(gate_dt, sc.data(srcs[2]), lane_elem(gate_lanes, wsize, t).min(gate_n.saturating_sub(1))))
                .collect()
        } else {
            vec![true; wsize]
        };
        let em = masks.last().unwrap().clone();
        let alt_lanes = if gated {
            match &sc.slots[srcs[1]] {
                Slot::Data { lanes, .. } => *lanes,
                _ => 1,
            }
        } else {
            1
        };
        let alt_n = if gated { alt.len() / alt_dt.itemsize() } else { 0 };

        sc.put_data(i, dtv, load_n, wsize);
        for j in 0..load_n {
            for (t, (m, oo)) in pairs.iter().enumerate() {
                let valid = em[t.min(em.len() - 1)] && gatev[t.min(gatev.len() - 1)];
                let d = (j * wsize + t) * isz;
                if valid {
                    let mut tmp = vec![0u8; isz];
                    read_mem(m, *oo + j, dtv, &mut tmp);
                    sc.data_mut(i).as_bytes_mut()[d..d + isz].copy_from_slice(&tmp);
                } else if gated {
                    // alt is read with the f64 `tovec` convention (lane t, thread 0).
                    let mut tmp = vec![0u8; isz];
                    read_elem_conv(&alt, alt_dt, lane_elem(alt_lanes, wsize, t).min(alt_n.saturating_sub(1)), dtv, &mut tmp);
                    sc.data_mut(i).as_bytes_mut()[d..d + isz].copy_from_slice(&tmp);
                } else {
                    sc.data_mut(i).as_bytes_mut()[d..d + isz].fill(0);
                }
            }
        }
    }

    fn stack(&self, sc: &mut Scratch, i: usize, srcs: &[usize], real: &[usize], wsize: usize) {
        let instrs = self.kernel.instrs();
        let dtv = instrs[i].dt;
        let isz = dtv.itemsize();
        let mut payload: Vec<Vec<u8>> = Vec::new();
        for &s in srcs {
            if instrs[s].dt == dtype::VOID {
                continue;
            }
            let sdt = instrs[s].dt;
            let s_isz = sdt.itemsize();
            match &sc.slots[s] {
                Slot::Data { buf, lanes, .. } => {
                    let b = buf.as_bytes();
                    let stride = wsize * s_isz;
                    for lane in 0..(*lanes).max(1) {
                        let start = lane * stride;
                        let mut lane_bytes = vec![0u8; wsize * isz];
                        for t in 0..wsize {
                            let so = start + t * s_isz;
                            let d = t * isz;
                            read_slice_conv(&b[so..so + s_isz], sdt, dtv, &mut lane_bytes[d..d + isz]);
                        }
                        payload.push(lane_bytes);
                    }
                }
                Slot::P(p) => {
                    let mut lane_bytes = vec![0u8; wsize * isz];
                    for (t, (m, oo)) in p.iter().enumerate() {
                        read_mem(m, *oo, dtv, &mut lane_bytes[t * isz..t * isz + isz]);
                    }
                    payload.push(lane_bytes);
                }
                _ => panic!("TGERR: typed STACK src"),
            }
        }
        let _ = real;
        sc.put_data(i, dtv, payload.len().max(1), wsize);
        let d = sc.data_mut(i).as_bytes_mut();
        for (l, p) in payload.iter().enumerate() {
            let start = l * wsize * isz;
            d[start..start + p.len()].copy_from_slice(p);
        }
    }

    fn alu(&self, sc: &mut Scratch, i: usize, opv: Ops, srcs: &[usize], real: &[usize], wsize: usize) {
        let instrs = self.kernel.instrs();
        let n = real.len();
        if n == 3 {
            let dtv = instrs[i].dt;
            let (a_dt, al) = sc.meta(srcs[0]);
            let (b_dt, bl) = sc.meta(srcs[1]);
            let (c_dt, cl) = sc.meta(srcs[2]);
            let lanes = al.max(bl).max(cl);
            sc.put_data(i, dtv, lanes, wsize);
            let parts = sc.slots.get_disjoint_mut([i, srcs[0], srcs[1], srcs[2]]);
            if let Ok([d, s0, s1, s2]) = parts {
                let d = slot_buf_mut(d);
                let s0b = slot_buf(s0);
                let s1b = slot_buf(s1);
                let s2b = slot_buf(s2);
                if opv == Ops::Where {
                    let fast = a_dt == dtype::BOOL && b_dt == dtv && c_dt == dtv && al == lanes && bl == lanes && cl == lanes;
                    if fast {
                        assert!(rv::alu_where(dtv, d, s0b, s1b, s2b), "TGERR: typed WHERE failed");
                    } else {
                        let cond = broadcast(&coerce(s0b, a_dt, dtype::BOOL), al, lanes, wsize);
                        let b = broadcast(&coerce(s1b, b_dt, dtv), bl, lanes, wsize);
                        let c = broadcast(&coerce(s2b, c_dt, dtv), cl, lanes, wsize);
                        assert!(rv::alu_where(dtv, d, &cond, &b, &c), "TGERR: typed WHERE failed");
                    }
                } else {
                    let fast = a_dt == dtv && b_dt == dtv && c_dt == dtv && al == lanes && bl == lanes && cl == lanes;
                    if fast {
                        assert!(rv::alu_mulacc(dtv, d, s0b, s1b, s2b), "TGERR: typed MULACC failed");
                    } else {
                        let a = broadcast(&coerce(s0b, a_dt, dtv), al, lanes, wsize);
                        let b = broadcast(&coerce(s1b, b_dt, dtv), bl, lanes, wsize);
                        let c = broadcast(&coerce(s2b, c_dt, dtv), cl, lanes, wsize);
                        assert!(rv::alu_mulacc(dtv, d, &a, &b, &c), "TGERR: typed MULACC failed");
                    }
                }
            } else {
                // duplicate source operands (e.g. `a*a`): clone them.
                let s0b = sc.data(srcs[0]).clone();
                let s1b = sc.data(srcs[1]).clone();
                let s2b = sc.data(srcs[2]).clone();
                let d = sc.data_mut(i);
                if opv == Ops::Where {
                    let cond = broadcast(&coerce(&s0b, a_dt, dtype::BOOL), al, lanes, wsize);
                    let b = broadcast(&coerce(&s1b, b_dt, dtv), bl, lanes, wsize);
                    let c = broadcast(&coerce(&s2b, c_dt, dtv), cl, lanes, wsize);
                    assert!(rv::alu_where(dtv, d, &cond, &b, &c), "TGERR: typed WHERE failed");
                } else {
                    let a = broadcast(&coerce(&s0b, a_dt, dtv), al, lanes, wsize);
                    let b = broadcast(&coerce(&s1b, b_dt, dtv), bl, lanes, wsize);
                    let c = broadcast(&coerce(&s2b, c_dt, dtv), cl, lanes, wsize);
                    assert!(rv::alu_mulacc(dtv, d, &a, &b, &c), "TGERR: typed MULACC failed");
                }
            }
            return;
        }
        if n == 1 {
            let dtv = instrs[i].dt;
            let (s_dt, sl) = sc.meta(srcs[0]);
            sc.put_data(i, dtv, sl, wsize);
            let [d, s0] = sc.slots.get_disjoint_mut([i, srcs[0]]).unwrap();
            let d = slot_buf_mut(d);
            let s0b = slot_buf(s0);
            if s_dt == dtv {
                assert!(rv::alu_unary(opv, dtv, d, s0b), "TGERR: typed unary ALU {opv:?} failed");
            } else {
                let a = coerce(s0b, s_dt, dtv);
                assert!(rv::alu_unary(opv, dtv, d, &a), "TGERR: typed unary ALU {opv:?} failed");
            }
            return;
        }
        let out_dt = instrs[i].dt;
        let op_dt = instrs[srcs[0]].dt;
        let (a_dt, al) = sc.meta(srcs[0]);
        let (b_dt, bl) = sc.meta(srcs[1]);
        let lanes = al.max(bl);
        sc.put_data(i, out_dt, lanes, wsize);
        let parts = sc.slots.get_disjoint_mut([i, srcs[0], srcs[1]]);
        if let Ok([d, s0, s1]) = parts {
            let d = slot_buf_mut(d);
            let s0b = slot_buf(s0);
            let s1b = slot_buf(s1);
            if a_dt == op_dt && b_dt == op_dt && al == lanes && bl == lanes {
                assert!(rv::alu_binary(opv, op_dt, d, s0b, s1b), "TGERR: typed binary ALU {opv:?} failed");
            } else {
                let a = broadcast(&coerce(s0b, a_dt, op_dt), al, lanes, wsize);
                let b = broadcast(&coerce(s1b, b_dt, op_dt), bl, lanes, wsize);
                assert!(rv::alu_binary(opv, op_dt, d, &a, &b), "TGERR: typed binary ALU {opv:?} failed");
            }
        } else {
            // duplicate source operands (e.g. `a*a`): clone them.
            let s0b = sc.data(srcs[0]).clone();
            let s1b = sc.data(srcs[1]).clone();
            let d = sc.data_mut(i);
            let a = broadcast(&coerce(&s0b, a_dt, op_dt), al, lanes, wsize);
            let b = broadcast(&coerce(&s1b, b_dt, op_dt), bl, lanes, wsize);
            assert!(rv::alu_binary(opv, op_dt, d, &a, &b), "TGERR: typed binary ALU {opv:?} failed");
        }
    }

    fn loop_end_index(&self, srcs: &[usize], range: usize) -> usize {
        self.kernel
            .instrs()
            .iter()
            .position(|x| x.op == Ops::End && x.srcs.len() >= 2 && x.srcs[1] == range)
            .unwrap_or_else(|| panic!("TGERR: typed missing END for RANGE (range {range}, srcs {srcs:?})"))
    }
}

impl Scratch {
    fn data_mut(&mut self, i: usize) -> &mut Bytes {
        match &mut self.slots[i] {
            Slot::Data { buf, .. } => buf,
            _ => panic!("TGERR: typed expected data at instr {i}"),
        }
    }
}

fn copy_slot(sc: &mut Scratch, dst: usize, src: usize) {
    match &sc.slots[src] {
        Slot::Data { dt, lanes, buf } => {
            let (dt, lanes) = (*dt, *lanes);
            let bytes = buf.as_bytes().to_vec();
            let items = lanes * dt.itemsize();
            let wsize = (bytes.len() / items).max(1);
            sc.put_data(dst, dt, lanes, wsize);
            sc.data_mut(dst).copy_from(&bytes);
        }
        Slot::P(p) => {
            let p = p.clone();
            sc.slots[dst] = Slot::P(p);
        }
        Slot::M(m) => {
            let m = m.clone();
            sc.slots[dst] = Slot::M(m);
        }
        Slot::Empty => panic!("TGERR: typed copy from empty"),
    }
}

fn slot_buf(s: &Slot) -> &Bytes {
    match s {
        Slot::Data { buf, .. } => buf,
        _ => panic!("TGERR: typed expected data slot"),
    }
}

fn slot_buf_mut(s: &mut Slot) -> &mut Bytes {
    match s {
        Slot::Data { buf, .. } => buf,
        _ => panic!("TGERR: typed expected data slot"),
    }
}

/// Convert a value buffer to `to` dtype (no-op when already `to`).
fn coerce(buf: &Bytes, from: DTypeId, to: DTypeId) -> Bytes {
    if from == to || buf.is_empty() {
        return buf.clone();
    }
    let n = (buf.nbytes() / from.itemsize()).max(1);
    let mut out = Bytes::zeroed(n * to.itemsize());
    if rv::cast(from, to, &mut out, buf) {
        out
    } else {
        buf.clone()
    }
}

/// Widen/repeat a scalar (1-lane) value to `target` lanes.
fn broadcast(src: &Bytes, src_lanes: usize, target: usize, wsize: usize) -> Bytes {
    if src_lanes == target || src_lanes == 0 {
        return src.clone();
    }
    assert_eq!(src_lanes, 1, "TGERR: typed broadcast expects a scalar source");
    let per_thread = src.nbytes() / wsize.max(1);
    let mut out = Bytes::zeroed(target * wsize * per_thread);
    let sb = src.as_bytes();
    let db = out.as_bytes_mut();
    for l in 0..target {
        db[l * wsize * per_thread..(l + 1) * wsize * per_thread].copy_from_slice(&sb[0..wsize * per_thread]);
    }
    out
}

/// Convert a single element's bytes from `src_dt` to `dst_dt` in `out`.
fn read_slice_conv(src: &[u8], src_dt: DTypeId, dst_dt: DTypeId, out: &mut [u8]) {
    if src_dt == dst_dt {
        out[..dst_dt.itemsize()].copy_from_slice(&src[..dst_dt.itemsize()]);
    } else {
        let v = from_storage_scalar(&src[..src_dt.itemsize()], src_dt);
        out[..dst_dt.itemsize()].copy_from_slice(&to_storage_scalar(v, dst_dt)[..dst_dt.itemsize()]);
    }
}

/// Flat element index for "thread `t`" of a per-thread vector view. The f64
/// executor's `sv`/`tovec` helpers read **lane `t`, thread 0** for a `Val::V`
/// (a transpose of the lane-major layout); scalar (`lanes == 1`) values are
/// per-thread directly. Mirror that here.
fn lane_elem(lanes: usize, wsize: usize, t: usize) -> usize {
    if lanes > 1 { t.min(lanes - 1) * wsize } else { t }
}

/// Read element `(lane,t)` from a `Data`-style byte buffer, converting dtypes.
fn read_elem_conv(buf: &[u8], dt: DTypeId, elem: usize, out_dt: DTypeId, out: &mut [u8]) {
    let isz = dt.itemsize();
    let off = elem * isz;
    read_slice_conv(&buf[off..off + isz], dt, out_dt, out);
}

fn read_mem(m: &SendMemView, idx: usize, dt: DTypeId, out: &mut [u8]) {
    let isz = dt.itemsize();
    if m.elemsize() == isz {
        m.read_bytes(idx, &mut out[..isz]);
    } else {
        let mut tmp = vec![0u8; m.elemsize()];
        m.read_bytes(idx, &mut tmp);
        let v = from_storage_scalar(&tmp, dt);
        out[..isz].copy_from_slice(&to_storage_scalar(v, dt)[..isz]);
    }
}

fn write_mem(m: &SendMemView, idx: usize, src: &[u8], isz: usize) {
    if m.elemsize() == isz {
        m.write_bytes(idx, &src[..isz]);
    } else {
        // sub-dword: logical value is `src` at its dtype; the f64 interpreter
        // writes `itemsize` bytes at the window's stride.
        m.write_bytes(idx, &src[..isz.min(m.elemsize())]);
    }
}

/// Element `(lane j, thread t)` of a value buffer.
fn val_elem(bytes: &[u8], wsize: usize, isz: usize, j: usize, t: usize) -> &[u8] {
    let off = (j * wsize + t) * isz;
    &bytes[off..off + isz]
}
