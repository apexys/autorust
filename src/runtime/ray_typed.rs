//! Typed (dtype-native) executor for `ops_rayon` — R2 of `CURRENT_STATE.md` §9.8.
//!
//! `interp::exec_launch` re-expressed over `ray_value::Bytes` instead of `f64`
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
use crate::uop::UopId;
use crate::uop::arg::Arg;
use crate::uop::ops::{Ops, group};

type Pair = (SendMemView, usize);

/// A per-instruction register value.
enum Slot {
    Empty,
    /// lane-major typed elements: `lanes * wsize`, element `(l,t)` at
    /// `(l*wsize + t) * dt.itemsize()`.
    Data {
        buf: Bytes,
        dt: DTypeId,
        lanes: usize,
    },
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
    /// recycled `Slot::P` / `Slot::M` payloads (one per launch per slot).
    pool_p: Vec<Vec<Pair>>,
    pool_m: Vec<Vec<SendMemView>>,
    owned: Vec<Box<[u8]>>,
}

impl Scratch {
    fn recycle(&mut self, n: usize) {
        let Scratch {
            slots,
            pool,
            pool_p,
            pool_m,
            owned,
        } = self;
        for slot in slots.iter_mut() {
            match slot {
                Slot::Data { buf, .. } => pool.push(std::mem::take(buf)),
                Slot::P(v) => {
                    v.clear();
                    pool_p.push(std::mem::take(v));
                }
                Slot::M(v) => {
                    v.clear();
                    pool_m.push(std::mem::take(v));
                }
                Slot::Empty => {}
            }
            *slot = Slot::Empty;
        }
        owned.clear();
        if slots.len() < n {
            slots.resize_with(n, || Slot::Empty);
        }
    }

    #[inline]
    fn take_p(&mut self, i: usize) -> Vec<Pair> {
        match std::mem::replace(&mut self.slots[i], Slot::Empty) {
            Slot::P(v) => v,
            _ => self.pool_p.pop().unwrap_or_default(),
        }
    }

    #[inline]
    fn take_m(&mut self, i: usize) -> Vec<SendMemView> {
        match std::mem::replace(&mut self.slots[i], Slot::Empty) {
            Slot::M(v) => v,
            _ => self.pool_m.pop().unwrap_or_default(),
        }
    }

    fn put_data(&mut self, i: usize, dt: DTypeId, lanes: usize, wsize: usize) {
        let nbytes = lanes * wsize * dt.itemsize();
        if let Slot::Data {
            buf,
            dt: ddt,
            lanes: dlanes,
        } = &mut self.slots[i]
        {
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

    /// Detach the buffer at slot `i` (reusing its allocation) and size it for a
    /// `lanes * wsize` value at `dt`. Leaves the slot `Empty` so the caller can
    /// read other slots immutably while filling the returned buffer, then hand
    /// it back with [`Scratch::set_data`]. This is the allocation-free path used
    /// by the executor's hot ops; those ops fully overwrite the buffer, so it is
    /// deliberately *not* zeroed.
    fn take_sized(&mut self, i: usize, dt: DTypeId, lanes: usize, wsize: usize) -> Bytes {
        let nbytes = lanes * wsize * dt.itemsize();
        let mut b = match std::mem::replace(&mut self.slots[i], Slot::Empty) {
            Slot::Data { buf, .. } => buf,
            _ => self.pool.pop().unwrap_or_default(),
        };
        if b.nbytes() != nbytes {
            b.reset(nbytes);
        }
        b
    }

    #[inline]
    fn set_data(&mut self, i: usize, buf: Bytes, dt: DTypeId, lanes: usize) {
        self.slots[i] = Slot::Data { buf, dt, lanes };
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

/// Dense per-instruction dispatch tag, resolved once at compile time so the
/// interpreter loop matches a small enum instead of walking `Ops` and group
/// membership on every iteration.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    End,
    Range,
    If,
    Endif,
    After,
    Noop,
    Store,
    ParamBuffer,
    Special,
    Const,
    IndexData,
    IndexPairs,
    Load,
    Cast,
    Bitcast,
    Stack,
    AluUnary,
    AluBinary,
    AluTernary,
    Unsupported,
}

/// Compile-time per-instruction metadata. Hoisting this out of the interpreter
/// loop removes two `Vec` allocations *per instruction per iteration* (the
/// `srcs` clone and the non-`VOID` filter) and the O(n) `END` scan that the
/// RANGE branch used to do on every iteration.
struct Plan {
    /// Non-`VOID` source indices, per instruction.
    real: Vec<Vec<usize>>,
    /// For a RANGE instruction, the instruction to resume at once the loop
    /// ends (its matching `END` index + 1). `usize::MAX` when not a RANGE.
    range_after: Vec<usize>,
    /// Precomputed dispatch tag.
    kind: Vec<Kind>,
    /// The kernel contains `IF` (so `masks` is needed).
    has_if: bool,
    /// The kernel reads ALU parameters (`pvals`).
    has_alu_param: bool,
    /// The kernel has a `SPECIAL lidxN` (so the `warp` table is needed).
    needs_warp: bool,
}

impl Plan {
    fn build(instrs: &[crate::runtime::ray_kernel::Instr]) -> Plan {
        let n = instrs.len();
        let real: Vec<Vec<usize>> = instrs
            .iter()
            .map(|ins| {
                ins.srcs
                    .iter()
                    .copied()
                    .filter(|&s| instrs[s].dt != dtype::VOID)
                    .collect()
            })
            .collect();
        let mut range_after = vec![usize::MAX; n];
        for (i, ins) in instrs.iter().enumerate() {
            if ins.op == Ops::End
                && ins.srcs.len() >= 2
                && let r = ins.srcs[1]
                && r < n
                && instrs[r].op == Ops::Range
            {
                range_after[r] = i + 1;
            }
        }
        // Static value kind: 0 = none, 1 = data, 2 = pairs, 3 = views. Determines
        // whether an `INDEX` gathers from a register (`index_data`) or forms
        // address pairs (`index_pairs`).
        let mut vk = vec![0u8; n];
        for i in 0..n {
            let ins = &instrs[i];
            vk[i] = match ins.op {
                Ops::Param => {
                    if ins.addrspace == Some(AddrSpace::Alu) {
                        1
                    } else {
                        3
                    }
                }
                Ops::Buffer => 3,
                Ops::Const
                | Ops::Special
                | Ops::Range
                | Ops::Load
                | Ops::Cast
                | Ops::Bitcast
                | Ops::Stack => 1,
                Ops::After => ins
                    .srcs
                    .iter()
                    .copied()
                    .find(|&s| instrs[s].dt != dtype::VOID)
                    .map(|s| vk[s])
                    .unwrap_or(0),
                Ops::Index | Ops::Shrink => {
                    let s0 = ins.srcs[0];
                    if ins.addrspace_derived == Some(AddrSpace::Alu) || vk[s0] == 1 {
                        1
                    } else {
                        2
                    }
                }
                _ if group::in_(group::ALU, ins.op) => 1,
                _ => 0,
            };
        }
        let kind: Vec<Kind> = instrs
            .iter()
            .enumerate()
            .map(|(i, ins)| match ins.op {
                Ops::End => Kind::End,
                Ops::Range => Kind::Range,
                Ops::If => Kind::If,
                Ops::Endif => Kind::Endif,
                Ops::After => Kind::After,
                Ops::Barrier | Ops::Sink | Ops::Noop | Ops::Group => Kind::Noop,
                Ops::Store => Kind::Store,
                Ops::Param | Ops::Buffer => Kind::ParamBuffer,
                Ops::Special => Kind::Special,
                Ops::Const => Kind::Const,
                Ops::Index | Ops::Shrink => {
                    if vk[i] == 1 {
                        Kind::IndexData
                    } else {
                        Kind::IndexPairs
                    }
                }
                Ops::Load => Kind::Load,
                Ops::Cast => Kind::Cast,
                Ops::Bitcast => Kind::Bitcast,
                Ops::Stack => Kind::Stack,
                _ if group::in_(group::ALU, ins.op) => match real[i].len() {
                    1 => Kind::AluUnary,
                    2 => Kind::AluBinary,
                    3 => Kind::AluTernary,
                    _ => Kind::Unsupported,
                },
                _ => Kind::Unsupported,
            })
            .collect();
        let has_if = instrs.iter().any(|ins| ins.op == Ops::If);
        let has_alu_param = instrs
            .iter()
            .any(|ins| ins.op == Ops::Param && ins.addrspace == Some(AddrSpace::Alu));
        let needs_warp = instrs.iter().any(|ins| {
            ins.op == Ops::Special && matches!(&ins.arg, Arg::Str(s) if s.starts_with('l'))
        });
        Plan {
            real,
            range_after,
            kind,
            has_if,
            has_alu_param,
            needs_warp,
        }
    }

    fn is_supported(&self) -> bool {
        self.kind.iter().all(|&k| k != Kind::Unsupported)
    }
}

/// The typed executor.
pub struct TypedKernel {
    kernel: RayKernel,
    plan: Plan,
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
        let plan = Plan::build(kernel.instrs());
        if !plan.is_supported() {
            return None;
        }
        Some(TypedKernel { kernel, plan })
    }

    pub fn globals(&self) -> &[i64] {
        &self.kernel.globals
    }
    pub fn sig_dtypes(&self) -> &HashMap<i64, DTypeId> {
        &self.kernel.sig_dtypes
    }

    pub fn exec(
        &self,
        views: &HashMap<i64, SendMemView>,
        global: [usize; 3],
        local: Option<[usize; 3]>,
        vals: &[f64],
        parallel: bool,
    ) {
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

    fn exec_launch(
        &self,
        gidx: [usize; 3],
        lsize: [usize; 3],
        views: &HashMap<i64, SendMemView>,
        vals: &[f64],
        sc: &mut Scratch,
    ) {
        let instrs = self.kernel.instrs();
        let wsize = (lsize[0] * lsize[1] * lsize[2]).max(1);
        let mut warp: Vec<[usize; 3]> = Vec::new();
        if self.plan.needs_warp {
            for c in 0..lsize[2] {
                for b in 0..lsize[1] {
                    for a in 0..lsize[0] {
                        warp.push([a, b, c]);
                    }
                }
            }
        }
        let mut pvals: VecDeque<f64> = if self.plan.has_alu_param {
            vals.iter().cloned().collect()
        } else {
            VecDeque::new()
        };
        let mut masks: Vec<Vec<bool>> = if self.plan.has_if {
            vec![vec![true; wsize]]
        } else {
            Vec::new()
        };

        let isz = |dt: DTypeId| dt.itemsize();
        let op = |i: usize| instrs[i].op;
        let dt = |i: usize| instrs[i].dt;

        let mut i = 0usize;
        while i < instrs.len() {
            let o = op(i);
            let srcs: &[usize] = &instrs[i].srcs;
            let real: &[usize] = &self.plan.real[i];
            match self.plan.kind[i] {
                Kind::End => {
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
                Kind::Range => {
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
                            i = self.plan.range_after[i];
                            continue;
                        }
                    }
                }
                Kind::If => {
                    let (cdt, cl) = sc.meta(srcs[0]);
                    let n = cl * wsize;
                    let last = masks.last().unwrap().clone();
                    let condv: Vec<bool> = (0..wsize)
                        .map(|t| {
                            rv::truthy(
                                cdt,
                                sc.data(srcs[0]),
                                lane_elem(cl, wsize, t).min(n.saturating_sub(1)),
                            )
                        })
                        .collect();
                    masks.push(
                        last.iter()
                            .zip(condv.iter().chain(std::iter::repeat(&condv[0])))
                            .map(|(a, b)| *a && *b)
                            .collect(),
                    );
                }
                Kind::Endif => {
                    masks.pop();
                }
                Kind::After => {
                    copy_slot(sc, i, real[0]);
                }
                Kind::Noop => {}
                Kind::Store => self.store(sc, i, srcs, real, wsize, &masks),
                Kind::ParamBuffer => {
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
                        let mv = *views
                            .get(&slot)
                            .unwrap_or_else(|| panic!("TGERR: typed missing view slot {slot}"));
                        let mut vs = sc.take_m(i);
                        vs.clear();
                        vs.resize(wsize, mv);
                        sc.slots[i] = Slot::M(vs);
                    } else if instrs[i].addrspace == Some(AddrSpace::Reg) {
                        let (nel, s) = (instrs[i].numel, isz(dt(i)));
                        let vs: Vec<SendMemView> = (0..wsize)
                            .map(|_| {
                                SendMemView::from_owned(vec![0u8; nel * s], s, nel, &mut sc.owned)
                            })
                            .collect();
                        sc.slots[i] = Slot::M(vs);
                    } else {
                        let (nel, s) = (instrs[i].numel, isz(dt(i)));
                        let mv = SendMemView::from_owned(vec![0u8; nel * s], s, nel, &mut sc.owned);
                        let mut vs = sc.take_m(i);
                        vs.clear();
                        vs.resize(wsize, mv);
                        sc.slots[i] = Slot::M(vs);
                    }
                }
                Kind::Special => {
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
                Kind::Const => {
                    sc.put_data(i, dt(i), 1, wsize);
                    match instrs[i].const_val.as_ref().expect("TGERR: typed const") {
                        ConstVal::Int(x) => rv::fill_i128(dt(i), sc.data_mut(i), *x),
                        ConstVal::Float(x) => rv::fill_f64(dt(i), sc.data_mut(i), *x),
                        ConstVal::Bool(b) => rv::fill_i128(dt(i), sc.data_mut(i), *b as i128),
                        other => panic!("TGERR: typed const {other:?}"),
                    }
                }
                Kind::IndexData => self.index_data(sc, i, srcs, wsize),
                Kind::IndexPairs => self.index_pairs(sc, i, srcs, wsize),
                Kind::Load => self.load(sc, i, srcs, real, wsize, &masks),
                Kind::Cast => {
                    let (sdt, lanes) = sc.meta(srcs[0]);
                    sc.put_data(i, dt(i), lanes, wsize);
                    let src = sc.data(srcs[0]).clone();
                    assert!(
                        rv::cast(sdt, dt(i), sc.data_mut(i), &src),
                        "TGERR: typed CAST failed"
                    );
                }
                Kind::Bitcast => {
                    let (sdt, lanes) = sc.meta(srcs[0]);
                    debug_assert_eq!(sdt.itemsize(), dt(i).itemsize());
                    sc.put_data(i, dt(i), lanes, wsize);
                    let src = sc.data(srcs[0]).as_bytes().to_vec();
                    sc.data_mut(i).copy_from(&src);
                }
                Kind::Stack => self.stack(sc, i, srcs, real, wsize),
                Kind::AluUnary | Kind::AluBinary | Kind::AluTernary => {
                    self.alu(sc, i, o, srcs, real, wsize)
                }
                Kind::Unsupported => {
                    unreachable!("TGERR: typed unsupported op rejected at compile")
                }
            }
            i += 1;
        }
    }

    fn store(
        &self,
        sc: &mut Scratch,
        _i: usize,
        srcs: &[usize],
        _real: &[usize],
        wsize: usize,
        masks: &[Vec<bool>],
    ) {
        let instrs = self.kernel.instrs();
        let gate = masks.last();
        let (val_dt, val_lanes) = sc.meta(srcs[1]);
        let store_dt = instrs[instrs[srcs[0]].srcs[0]].dt;
        let val_isz = val_dt.itemsize();
        assert_eq!(
            val_isz,
            store_dt.itemsize(),
            "TGERR: typed store dtype mismatch"
        );
        let lanes = val_lanes.max(1);
        let vb = sc.data(srcs[1]).as_bytes();
        let gate_at = |t: usize| gate.is_none_or(|g| g[t.min(g.len() - 1)]);

        if instrs[srcs[0]].op == Ops::Stack && val_lanes == 1 {
            let dest_srcs = &instrs[srcs[0]].srcs;
            for j in 0..lanes {
                let dj = dest_srcs.get(j).copied().unwrap_or(dest_srcs[0]);
                if let Slot::P(dp) = &sc.slots[dj] {
                    for (t, (m, oo)) in dp.iter().enumerate() {
                        if gate_at(t) {
                            write_mem(m, *oo, val_elem(vb, wsize, val_isz, j, t), val_isz);
                        }
                    }
                }
            }
            return;
        }
        let pairs: &[Pair] = match &sc.slots[srcs[0]] {
            Slot::P(p) => p,
            _ => panic!("TGERR: typed store src not pairs"),
        };
        for j in 0..lanes {
            for (t, (m, oo)) in pairs.iter().enumerate() {
                if gate_at(t) {
                    write_mem(m, *oo + j, val_elem(vb, wsize, val_isz, j, t), val_isz);
                }
            }
        }
    }

    fn index_data(&self, sc: &mut Scratch, i: usize, srcs: &[usize], wsize: usize) {
        let (a_dt, a_lanes) = sc.meta(srcs[0]);
        let (b_dt, b_lanes) = sc.meta(srcs[1]);
        let isz = a_dt.itemsize();
        let n_b = sc.data(srcs[1]).nbytes() / b_dt.itemsize();
        let mut dest = sc.take_sized(i, a_dt, 1, wsize);
        {
            let sb = sc.data(srcs[0]).as_bytes();
            let ib: &Bytes = sc.data(srcs[1]);
            let ix32: Option<&[i32]> = if b_dt == dtype::INT32 {
                Some(ib.as_slice::<i32>())
            } else {
                None
            };
            let db = dest.as_bytes_mut();
            for t in 0..wsize {
                let be = lane_elem(b_lanes, wsize, t).min(n_b.saturating_sub(1));
                let ix = match ix32 {
                    Some(xs) => xs[be] as i128,
                    None => rv::get_i128(b_dt, ib, be),
                };
                let alane = if a_lanes > 1 {
                    (ix.max(0) as usize).min(a_lanes - 1)
                } else {
                    0
                };
                let so = (alane * wsize + t) * isz;
                db[t * isz..t * isz + isz].copy_from_slice(&sb[so..so + isz]);
            }
        }
        sc.set_data(i, dest, a_dt, 1);
    }

    fn index_pairs(&self, sc: &mut Scratch, i: usize, srcs: &[usize], wsize: usize) {
        let instrs = self.kernel.instrs();
        // (M|P) indexed by a scalar or per-thread offset value. Detach the
        // destination so its `Vec` capacity is reused and the sources can be
        // borrowed immutably without any cloning.
        let b_dt = instrs[srcs[1]].dt;
        let (b_lanes, n_b) = match &sc.slots[srcs[1]] {
            Slot::Data { lanes, buf, .. } => (*lanes, buf.nbytes() / b_dt.itemsize()),
            _ => (1, 0),
        };
        let mut ret = sc.take_p(i);
        ret.clear();
        {
            let ib: &Bytes = sc.data(srcs[1]);
            // Fast path for the dominant 32-bit signed index dtype.
            let ix32: Option<&[i32]> = if b_dt == dtype::INT32 {
                Some(ib.as_slice::<i32>())
            } else {
                None
            };
            let read_idx = |be: usize| -> i128 {
                match ix32 {
                    Some(xs) => xs[be] as i128,
                    None => rv::get_i128(b_dt, ib, be),
                }
            };
            match &sc.slots[srcs[0]] {
                Slot::M(ms) => {
                    for (t, m) in ms.iter().enumerate() {
                        let be = lane_elem(b_lanes, wsize, t).min(n_b.saturating_sub(1));
                        ret.push((*m, read_idx(be).max(0) as usize));
                    }
                }
                Slot::P(ps) => {
                    for (t, (m, base)) in ps.iter().enumerate() {
                        let be = lane_elem(b_lanes, wsize, t).min(n_b.saturating_sub(1));
                        ret.push((*m, base + read_idx(be).max(0) as usize));
                    }
                }
                _ => panic!("TGERR: typed index_pairs src0"),
            }
        }
        sc.slots[i] = Slot::P(ret);
    }

    fn load(
        &self,
        sc: &mut Scratch,
        i: usize,
        srcs: &[usize],
        real: &[usize],
        wsize: usize,
        masks: &[Vec<bool>],
    ) {
        let instrs = self.kernel.instrs();
        let load_n = instrs[i].numel.max(1);
        let dtv = instrs[i].dt;
        let isz = dtv.itemsize();
        let gated = real.len() >= 3;
        let em = masks.last();
        let mut dest = sc.take_sized(i, dtv, load_n, wsize);
        let pairs: &[Pair] = match &sc.slots[srcs[0]] {
            Slot::P(p) => p,
            _ => panic!("TGERR: typed load src not pairs"),
        };
        if !gated {
            let db = dest.as_bytes_mut();
            for j in 0..load_n {
                for (t, (m, oo)) in pairs.iter().enumerate() {
                    let d = (j * wsize + t) * isz;
                    if em.is_none_or(|e| e[t.min(e.len() - 1)]) {
                        read_mem(m, *oo + j, dtv, &mut db[d..d + isz]);
                    } else {
                        db[d..d + isz].fill(0);
                    }
                }
            }
        } else {
            let alt: &Bytes = sc.data(srcs[1]);
            let (alt_dt, alt_lanes) = sc.meta(srcs[1]);
            let alt_n = alt.nbytes() / alt_dt.itemsize();
            let gate: &Bytes = sc.data(srcs[2]);
            let (gate_dt, gate_lanes) = sc.meta(srcs[2]);
            let gate_n = gate.nbytes() / gate_dt.itemsize();
            let db = dest.as_bytes_mut();
            for j in 0..load_n {
                for (t, (m, oo)) in pairs.iter().enumerate() {
                    let d = (j * wsize + t) * isz;
                    let valid = em.is_none_or(|e| e[t.min(e.len() - 1)])
                        && rv::truthy(
                            gate_dt,
                            gate,
                            lane_elem(gate_lanes, wsize, t).min(gate_n.saturating_sub(1)),
                        );
                    if valid {
                        read_mem(m, *oo + j, dtv, &mut db[d..d + isz]);
                    } else {
                        let be = lane_elem(alt_lanes, wsize, t).min(alt_n.saturating_sub(1));
                        read_elem_conv(alt.as_bytes(), alt_dt, be, dtv, &mut db[d..d + isz]);
                    }
                }
            }
        }
        sc.set_data(i, dest, dtv, load_n);
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
                            read_slice_conv(
                                &b[so..so + s_isz],
                                sdt,
                                dtv,
                                &mut lane_bytes[d..d + isz],
                            );
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

    fn alu(
        &self,
        sc: &mut Scratch,
        i: usize,
        opv: Ops,
        srcs: &[usize],
        real: &[usize],
        wsize: usize,
    ) {
        let instrs = self.kernel.instrs();
        let n = real.len();
        if n == 3 {
            let dtv = instrs[i].dt;
            let (a_dt, al) = sc.meta(srcs[0]);
            let (b_dt, bl) = sc.meta(srcs[1]);
            let (c_dt, cl) = sc.meta(srcs[2]);
            let lanes = al.max(bl).max(cl);
            let mut dest = sc.take_sized(i, dtv, lanes, wsize);
            let s0b = sc.data(srcs[0]);
            let s1b = sc.data(srcs[1]);
            let s2b = sc.data(srcs[2]);
            if opv == Ops::Where {
                let fast = a_dt == dtype::BOOL
                    && b_dt == dtv
                    && c_dt == dtv
                    && al == lanes
                    && bl == lanes
                    && cl == lanes;
                if fast {
                    assert!(
                        rv::alu_where(dtv, &mut dest, s0b, s1b, s2b),
                        "TGERR: typed WHERE failed"
                    );
                } else {
                    let cond = broadcast(&coerce(s0b, a_dt, dtype::BOOL), al, lanes, wsize);
                    let b = broadcast(&coerce(s1b, b_dt, dtv), bl, lanes, wsize);
                    let c = broadcast(&coerce(s2b, c_dt, dtv), cl, lanes, wsize);
                    assert!(
                        rv::alu_where(dtv, &mut dest, &cond, &b, &c),
                        "TGERR: typed WHERE failed"
                    );
                }
            } else {
                let fast = a_dt == dtv
                    && b_dt == dtv
                    && c_dt == dtv
                    && al == lanes
                    && bl == lanes
                    && cl == lanes;
                if fast {
                    assert!(
                        rv::alu_mulacc(dtv, &mut dest, s0b, s1b, s2b),
                        "TGERR: typed MULACC failed"
                    );
                } else {
                    let a = broadcast(&coerce(s0b, a_dt, dtv), al, lanes, wsize);
                    let b = broadcast(&coerce(s1b, b_dt, dtv), bl, lanes, wsize);
                    let c = broadcast(&coerce(s2b, c_dt, dtv), cl, lanes, wsize);
                    assert!(
                        rv::alu_mulacc(dtv, &mut dest, &a, &b, &c),
                        "TGERR: typed MULACC failed"
                    );
                }
            }
            sc.set_data(i, dest, dtv, lanes);
            return;
        }
        if n == 1 {
            let dtv = instrs[i].dt;
            let (s_dt, sl) = sc.meta(srcs[0]);
            let mut dest = sc.take_sized(i, dtv, sl, wsize);
            let s0b = sc.data(srcs[0]);
            if s_dt == dtv {
                assert!(
                    rv::alu_unary(opv, dtv, &mut dest, s0b),
                    "TGERR: typed unary ALU {opv:?} failed"
                );
            } else {
                let a = coerce(s0b, s_dt, dtv);
                assert!(
                    rv::alu_unary(opv, dtv, &mut dest, &a),
                    "TGERR: typed unary ALU {opv:?} failed"
                );
            }
            sc.set_data(i, dest, dtv, sl);
            return;
        }
        let out_dt = instrs[i].dt;
        let op_dt = instrs[srcs[0]].dt;
        let (a_dt, al) = sc.meta(srcs[0]);
        let (b_dt, bl) = sc.meta(srcs[1]);
        let lanes = al.max(bl);
        let mut dest = sc.take_sized(i, out_dt, lanes, wsize);
        let s0b = sc.data(srcs[0]);
        let s1b = sc.data(srcs[1]);
        if a_dt == op_dt && b_dt == op_dt && al == lanes && bl == lanes {
            assert!(
                rv::alu_binary(opv, op_dt, &mut dest, s0b, s1b),
                "TGERR: typed binary ALU {opv:?} failed"
            );
        } else {
            let a = broadcast(&coerce(s0b, a_dt, op_dt), al, lanes, wsize);
            let b = broadcast(&coerce(s1b, b_dt, op_dt), bl, lanes, wsize);
            assert!(
                rv::alu_binary(opv, op_dt, &mut dest, &a, &b),
                "TGERR: typed binary ALU {opv:?} failed"
            );
        }
        sc.set_data(i, dest, out_dt, lanes);
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
    if matches!(sc.slots[src], Slot::P(_)) {
        let mut v = sc.take_p(dst);
        v.clear();
        if let Slot::P(p) = &sc.slots[src] {
            v.extend_from_slice(p);
        }
        sc.slots[dst] = Slot::P(v);
    } else if matches!(sc.slots[src], Slot::M(_)) {
        let mut v = sc.take_m(dst);
        v.clear();
        if let Slot::M(m) = &sc.slots[src] {
            v.extend_from_slice(m);
        }
        sc.slots[dst] = Slot::M(v);
    } else {
        let (dt, lanes, nbytes) = match &sc.slots[src] {
            Slot::Data { dt, lanes, buf } => (*dt, *lanes, buf.nbytes()),
            Slot::Empty => panic!("TGERR: typed copy from empty"),
            _ => unreachable!(),
        };
        let mut b = match std::mem::replace(&mut sc.slots[dst], Slot::Empty) {
            Slot::Data { buf, .. } => buf,
            _ => Bytes::default(),
        };
        if b.nbytes() != nbytes {
            b.reset(nbytes);
        }
        b.copy_from(sc.data(src).as_bytes());
        sc.slots[dst] = Slot::Data { buf: b, dt, lanes };
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
    assert_eq!(
        src_lanes, 1,
        "TGERR: typed broadcast expects a scalar source"
    );
    let per_thread = src.nbytes() / wsize.max(1);
    let mut out = Bytes::zeroed(target * wsize * per_thread);
    let sb = src.as_bytes();
    let db = out.as_bytes_mut();
    for l in 0..target {
        db[l * wsize * per_thread..(l + 1) * wsize * per_thread]
            .copy_from_slice(&sb[0..wsize * per_thread]);
    }
    out
}

/// Convert a single element's bytes from `src_dt` to `dst_dt` in `out`.
fn read_slice_conv(src: &[u8], src_dt: DTypeId, dst_dt: DTypeId, out: &mut [u8]) {
    if src_dt == dst_dt {
        out[..dst_dt.itemsize()].copy_from_slice(&src[..dst_dt.itemsize()]);
    } else {
        let v = from_storage_scalar(&src[..src_dt.itemsize()], src_dt);
        out[..dst_dt.itemsize()]
            .copy_from_slice(&to_storage_scalar(v, dst_dt)[..dst_dt.itemsize()]);
    }
}

/// Flat element index for "thread `t`" of a per-thread vector view. The f64
/// executor's `sv`/`tovec` helpers read **lane `t`, thread 0** for a `Val::V`
/// (a transpose of the lane-major layout); scalar (`lanes == 1`) values are
/// per-thread directly. Mirror that here.
fn lane_elem(lanes: usize, wsize: usize, t: usize) -> usize {
    if lanes > 1 {
        t.min(lanes - 1) * wsize
    } else {
        t
    }
}

/// Read element `(lane,t)` from a `Data`-style byte buffer, converting dtypes.
fn read_elem_conv(buf: &[u8], dt: DTypeId, elem: usize, out_dt: DTypeId, out: &mut [u8]) {
    let isz = dt.itemsize();
    let off = elem * isz;
    read_slice_conv(&buf[off..off + isz], dt, out_dt, out);
}

#[inline(always)]
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

#[inline(always)]
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
