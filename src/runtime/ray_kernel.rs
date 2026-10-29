//! Owned, `Send` scalar kernel IR for the `ops_rayon` backend (R1 of §9.8).
//!
//! The interpreter (`runtime/interp.rs`) reads every kernel fact straight out of
//! the `UOpCtx` (`Rc<Node>` + `RefCell` caches) and holds host memory as
//! `Rc<RefCell<Vec<u8>>>`. Neither is `Send`/`Sync`, so the kernel cannot be run
//! on `rayon` workers as-is. This module **compiles the linearized uop list once**
//! into plain owned data (op, dtype, src indices, arg, addrspace, numel, const)
//! plus raw-pointer memory windows, then executes that owned IR — serially or
//! across the global launch grid with `rayon`.
//!
//! The executor is a faithful port of `interp::exec_launch` (same `Val` model,
//! same `exec_alu`/`truncate` semantics), so its numerics match `INTERP` exactly;
//! that equality is the R1 gate. Anything it cannot compile (`compile` returns
//! `None`) falls back to the interpreter in `ops_rayon`, keeping `RAYON`
//! correct-by-construction while fast-path coverage grows.

use std::collections::HashMap;

use crate::dtype::{self, AddrSpace, ConstVal, DTypeId};
use crate::runtime::interp::{
    exec_alu, exec_alu_ternary, from_storage_scalar, to_storage_scalar, truncate,
};
use crate::uop::UopId;
use crate::uop::arg::Arg;
use crate::uop::ctx::UOpCtx;
use crate::uop::ops::{Ops, group};

/// A byte window into host memory, as a raw pointer. `Send`/`Sync` are asserted
/// manually: the pointer stays valid for the duration of a `call` (the backing
/// `Rc<dyn OpaqueT>` outlives the rayon scope — see `RayKernel::exec`), and
/// parallel tasks touch **disjoint output** regions / read-only inputs.
#[derive(Clone, Copy)]
pub struct SendMemView {
    ptr: *mut u8,
    offset: usize,
    elemsize: usize,
    count: usize,
}

// SAFETY: see the type doc — pointer lifetime is bounded by the enclosing
// `RayKernel::exec`, and tasks write disjoint output slices.
unsafe impl Send for SendMemView {}
unsafe impl Sync for SendMemView {}

impl SendMemView {
    /// Build a window from a host opaque (borrows `mem` only to read the base
    /// pointer; execution afterwards uses raw pointer arithmetic, never a
    /// `RefCell` borrow, so it is safe to share across workers).
    pub fn from_host(
        h: &crate::runtime::buffer::HostOpaque,
        elemsize: usize,
        count: usize,
    ) -> SendMemView {
        let ptr = h.mem.borrow().as_ptr() as *mut u8;
        SendMemView {
            ptr,
            offset: h.offset,
            elemsize,
            count,
        }
    }

    #[inline]
    pub(crate) fn elemsize(&self) -> usize {
        self.elemsize
    }

    /// Byte offset of element `i`.
    #[inline(always)]
    fn byte_at(&self, i: usize) -> usize {
        self.offset + i * self.elemsize
    }

    /// Copy `n` bytes out of element `i` (typed loads). Specializes the common
    /// element widths to a single unaligned load, avoiding a `memcpy` call.
    #[inline(always)]
    pub(crate) fn read_bytes(&self, i: usize, dst: &mut [u8]) {
        debug_assert!(i < self.count);
        let src = unsafe { self.ptr.add(self.byte_at(i)) };
        let n = dst.len();
        unsafe {
            match n {
                1 => *dst.get_unchecked_mut(0) = *src,
                2 => (dst.as_mut_ptr() as *mut u16)
                    .write_unaligned((src as *const u16).read_unaligned()),
                4 => (dst.as_mut_ptr() as *mut u32)
                    .write_unaligned((src as *const u32).read_unaligned()),
                8 => (dst.as_mut_ptr() as *mut u64)
                    .write_unaligned((src as *const u64).read_unaligned()),
                _ => std::ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), n),
            }
        }
    }

    /// Copy `n` bytes into element `i` (typed stores); see [`Self::read_bytes`].
    #[inline(always)]
    pub(crate) fn write_bytes(&self, i: usize, src: &[u8]) {
        debug_assert!(i < self.count);
        let dst = unsafe { self.ptr.add(self.byte_at(i)) };
        let n = src.len();
        unsafe {
            match n {
                1 => *dst = src[0],
                2 => (dst as *mut u16)
                    .write_unaligned(u16::from_le_bytes(src[..2].try_into().unwrap())),
                4 => (dst as *mut u32)
                    .write_unaligned(u32::from_le_bytes(src[..4].try_into().unwrap())),
                8 => (dst as *mut u64)
                    .write_unaligned(u64::from_le_bytes(src[..8].try_into().unwrap())),
                _ => std::ptr::copy_nonoverlapping(src.as_ptr(), dst, n),
            }
        }
    }

    fn load(&self, i: usize, dt: DTypeId) -> f64 {
        if i >= self.count {
            panic!(
                "TGERR: load out of bounds, size is {} and access is {i}",
                self.count
            );
        }
        let boff = self.offset + i * self.elemsize;
        let bytes: Vec<u8> = (0..self.elemsize)
            .map(|k| unsafe { *self.ptr.add(boff + k) })
            .collect();
        from_storage_scalar(&bytes, dt)
    }

    fn store(&self, i: usize, v: f64, dt: DTypeId) {
        if i >= self.count {
            panic!(
                "TGERR: store out of bounds, size is {} and access is {i}",
                self.count
            );
        }
        assert!(
            self.elemsize >= dt.itemsize(),
            "TGERR: sub-dword store not ported (elemsize={} dt={dt:?})",
            self.elemsize
        );
        let boff = self.offset + i * self.elemsize;
        let sval = to_storage_scalar(v, dt);
        for (k, x) in sval.iter().enumerate() {
            unsafe {
                *self.ptr.add(boff + k) = *x;
            }
        }
    }
}

/// One pre-resolved uop of the linearized kernel.
#[derive(Clone)]
pub(crate) struct Instr {
    pub(crate) op: Ops,
    pub(crate) dt: DTypeId,
    pub(crate) srcs: Vec<usize>,
    pub(crate) arg: Arg,
    pub(crate) addrspace: Option<AddrSpace>,
    pub(crate) addrspace_derived: Option<AddrSpace>,
    /// `numels[u]` (`exec_kernel_with_dims`): max_numel for
    /// Load/Store/Buffer/Param, the extent for Range, the SPECIAL extent else.
    pub(crate) numel: usize,
    /// Decoded `ConstVal` for `CONST`.
    pub(crate) const_val: Option<ConstVal>,
}

/// A compiled, `Send` kernel: the linearized uops as owned data.
pub struct RayKernel {
    instrs: Vec<Instr>,
    /// Signature slots of the global params (used to bind buffer windows).
    pub globals: Vec<i64>,
    pub sig_dtypes: HashMap<i64, DTypeId>,
}

impl RayKernel {
    /// The compiled instruction list (shared with the typed executor).
    pub(crate) fn instrs(&self) -> &[Instr] {
        &self.instrs
    }

    /// Clone with every instruction dtype mapped through `f` (the typed
    /// executor uses this to commit weak dtypes to concrete storage).
    pub(crate) fn remap_dtypes(&self, f: impl Fn(DTypeId) -> DTypeId) -> RayKernel {
        RayKernel {
            instrs: self
                .instrs
                .iter()
                .map(|ins| Instr {
                    dt: f(ins.dt),
                    ..ins.clone()
                })
                .collect(),
            globals: self.globals.clone(),
            sig_dtypes: self.sig_dtypes.clone(),
        }
    }

    /// Compile a decoded linear uop list. Returns `None` if any op is outside
    /// the supported set (the caller falls back to the interpreter).
    pub fn compile(ctx: &mut UOpCtx, uops: &[UopId]) -> Option<RayKernel> {
        // index of each uop in the list (srcs must be present).
        let index: HashMap<UopId, usize> = uops
            .iter()
            .enumerate()
            .map(|(i, u)| (u.clone(), i))
            .collect();

        // `numels` (mirrors `exec_kernel_with_dims`).
        let mut numels: HashMap<usize, usize> = HashMap::new();
        for (i, u) in uops.iter().enumerate() {
            match ctx.op_of(u) {
                Ops::Load | Ops::Store | Ops::Buffer | Ops::Param => {
                    numels.insert(i, ctx.max_numel(u.clone()) as usize);
                }
                Ops::Range => {
                    let ext = match ctx.ssimplify(ctx.srcs(u)[0].clone()) {
                        Ok(crate::uop::sint::SVal::I(v)) => v.max(1) as usize,
                        _ => panic!("TGERR: rayon RANGE extent"),
                    };
                    numels.insert(i, ext);
                }
                Ops::Special => {
                    let ext = match ctx.ssimplify(ctx.srcs(u)[0].clone()) {
                        Ok(crate::uop::sint::SVal::I(v)) => v.max(1) as usize,
                        _ => 1,
                    };
                    numels.insert(i, ext);
                }
                _ => {}
            }
        }

        let mut instrs: Vec<Instr> = Vec::with_capacity(uops.len());
        for (i, u) in uops.iter().enumerate() {
            let op = ctx.op_of(u);
            if !supported_op(op) {
                return None;
            }
            let srcs: Option<Vec<usize>> =
                ctx.srcs(u).iter().map(|s| index.get(s).copied()).collect();
            let srcs = srcs?;
            let const_val = match op {
                Ops::Const => Some(ctx.val(u)),
                _ => None,
            };
            instrs.push(Instr {
                op,
                dt: ctx.dt_of(u),
                srcs,
                arg: ctx.arg_of(u).clone(),
                addrspace: ctx.addrspace_of(u.clone()),
                addrspace_derived: ctx.addrspace_derived(u.clone()),
                numel: numels.get(&i).copied().unwrap_or(0),
                const_val,
            });
        }

        // Signature: non-ALU params in linear order (same as `to_program`).
        let mut globals: Vec<i64> = Vec::new();
        let mut sig_dtypes: HashMap<i64, DTypeId> = HashMap::new();
        for (i, u) in uops.iter().enumerate() {
            let _ = i;
            if ctx.op_of(u) == Ops::Param
                && let Arg::Param(p) = ctx.arg_of(u)
                && p.addrspace != Some(AddrSpace::Alu)
            {
                globals.push(p.slot);
                sig_dtypes.insert(p.slot, ctx.dt_of(u));
            }
        }
        if globals.is_empty() {
            return None;
        }
        Some(RayKernel {
            instrs,
            globals,
            sig_dtypes,
        })
    }

    /// Execute the kernel over the global launch grid. `views` maps buffer slot
    /// → window; `vals` are the ALU param bindings. `parallel` fans the global
    /// launches out over rayon (safe because tinygrad's work items write disjoint
    /// output slices and only read shared inputs).
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
        if parallel && order.len() > 1 {
            use rayon::prelude::*;
            order
                .par_iter()
                .for_each(|g| self.exec_launch(*g, local, views, vals));
        } else {
            for g in order {
                self.exec_launch(g, local, views, vals);
            }
        }
    }

    /// one (global_idx, local_size) launch invocation — faithful port of
    /// `interp::exec_launch`.
    fn exec_launch(
        &self,
        gidx: [usize; 3],
        lsize: [usize; 3],
        views: &HashMap<i64, SendMemView>,
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
        let mut values: HashMap<usize, Val> = HashMap::new();
        let mut pvals: std::collections::VecDeque<f64> = vals.iter().cloned().collect();
        let mut exec_masks: Vec<Vec<bool>> = vec![vec![true; wsize]];
        // Backing allocations for per-launch REG / non-shared-local scratch.
        let mut scratch: Vec<Box<[u8]>> = Vec::new();

        let op_of = |i: usize| self.instrs[i].op;
        let dt_of = |i: usize| self.instrs[i].dt;
        let srcs_of = |i: usize| self.instrs[i].srcs.clone();

        let sv = |values: &HashMap<usize, Val>, i: usize| -> Vec<f64> {
            match &values[&i] {
                Val::S(v) => v.clone(),
                Val::V(l) => l
                    .iter()
                    .map(|t| t.first().cloned().unwrap_or(0.0))
                    .collect(),
                _ => panic!("TGERR: interp src not scalar"),
            }
        };

        let mut i = 0usize;
        while i < self.instrs.len() {
            let op = op_of(i);
            let srcs = srcs_of(i);
            let src_vals: Vec<Val> = srcs
                .iter()
                .filter(|&&s| dt_of(s) != dtype::VOID)
                .map(|&s| {
                    values.get(&s).cloned().unwrap_or_else(|| {
                        panic!(
                            "TGERR: rayon missing value: src {} op {:?} of u {} op {:?}",
                            s,
                            op_of(s),
                            i,
                            op
                        )
                    })
                })
                .collect();
            match op {
                Ops::End => {
                    if srcs.len() == 3 {
                        let cond = sv(&values, srcs[2]);
                        if cond.first().cloned().unwrap_or(0.0) != 0.0 {
                            i = srcs[1];
                            continue;
                        }
                    } else {
                        i = srcs[1];
                        continue;
                    }
                }
                Ops::Range => {
                    if dt_of(i) == dtype::VOID {
                        i += 1;
                        continue;
                    }
                    let ext = self.instrs[i].numel;
                    match values.get_mut(&i) {
                        None => {
                            values.insert(i, Val::S(vec![0.0; wsize]));
                        }
                        Some(Val::S(vs)) => {
                            for v in vs.iter_mut() {
                                *v += 1.0;
                            }
                            if vs[0] == ext as f64 {
                                values.remove(&i);
                                i = self.loop_end_index(&srcs, i) + 1;
                                continue;
                            }
                        }
                        _ => panic!("TGERR: interp RANGE value"),
                    }
                }
                Ops::If => {
                    let cond = sv(&values, srcs[0]);
                    let last = exec_masks.last().unwrap().clone();
                    exec_masks.push(
                        last.iter()
                            .zip(cond.iter().chain(std::iter::repeat(&cond[0])))
                            .map(|(a, b)| *a && *b != 0.0)
                            .collect(),
                    );
                }
                Ops::Endif => {
                    exec_masks.pop();
                }
                Ops::After => {
                    let v = srcs
                        .iter()
                        .find(|&&s| dt_of(s) != dtype::VOID)
                        .and_then(|&s| values.get(&s).cloned());
                    match v {
                        Some(v) => {
                            values.insert(i, v);
                        }
                        None => panic!("TGERR: interp AFTER src missing value"),
                    }
                }
                Ops::Barrier | Ops::Sink | Ops::Noop | Ops::Group => {}
                Ops::Store => {
                    let gate = exec_masks.last().unwrap().clone();
                    let store_n = self.instrs[i].numel;
                    let vals_lane: Vec<Vec<f64>> = match &src_vals[1] {
                        Val::S(vs) if store_n == 1 => vec![vs.clone()],
                        Val::V(lanes) => lanes.clone(),
                        _ => panic!("TGERR: interp store value"),
                    };
                    let pairs_owned: Vec<(SendMemView, usize)>;
                    let pairs: &Vec<(SendMemView, usize)> = match &src_vals[0] {
                        Val::P(p) => p,
                        Val::V(_) | Val::S(_) => {
                            pairs_owned = Vec::new();
                            &pairs_owned
                        }
                        _ => panic!("TGERR: interp store src"),
                    };
                    let store_dt = dt_of(self.instrs[srcs[0]].srcs[0]);
                    if op_of(srcs[0]) == Ops::Stack && matches!(src_vals[0], Val::S(_)) {
                        let dest_srcs = self.instrs[srcs[0]].srcs.clone();
                        for (j, val) in vals_lane.iter().enumerate() {
                            let dj = dest_srcs.get(j).copied().unwrap_or(dest_srcs[0]);
                            let dpairs = match values.get(&dj) {
                                Some(Val::P(p)) => p.clone(),
                                _ => continue,
                            };
                            for (t, ((m, o), v)) in dpairs.iter().zip(val.iter()).enumerate() {
                                if gate[t] {
                                    m.store(*o, *v, store_dt);
                                }
                            }
                        }
                        i += 1;
                        continue;
                    }
                    for (j, val) in vals_lane.iter().enumerate() {
                        for (t, ((m, o), v)) in pairs.iter().zip(val.iter()).enumerate() {
                            if gate[t] {
                                m.store(*o + j, *v, store_dt);
                            }
                        }
                    }
                }
                Ops::Param | Ops::Buffer => {
                    let is_alu = self.instrs[i].addrspace == Some(AddrSpace::Alu);
                    if is_alu && op == Ops::Param {
                        let v = pvals.pop_front().unwrap_or(0.0);
                        values.insert(i, Val::S(vec![v; wsize]));
                    } else if op == Ops::Param {
                        let slot = match &self.instrs[i].arg {
                            Arg::Param(p) => p.slot,
                            _ => panic!("TGERR: interp PARAM arg"),
                        };
                        let mv = views
                            .get(&slot)
                            .unwrap_or_else(|| panic!("TGERR: interp missing view for slot {slot}"))
                            .to_owned();
                        values.insert(i, Val::M(vec![mv; wsize]));
                    } else if self.instrs[i].addrspace == Some(AddrSpace::Reg) {
                        let nel = self.instrs[i].numel;
                        let isz = dt_of(i).itemsize();
                        values.insert(
                            i,
                            Val::M(
                                (0..wsize)
                                    .map(|_| {
                                        SendMemView::from_owned(
                                            vec![0u8; nel * isz],
                                            isz,
                                            nel,
                                            &mut scratch,
                                        )
                                    })
                                    .collect(),
                            ),
                        );
                    } else {
                        let nel = self.instrs[i].numel;
                        let isz = dt_of(i).itemsize();
                        let mv =
                            SendMemView::from_owned(vec![0u8; nel * isz], isz, nel, &mut scratch);
                        values.insert(i, Val::M(vec![mv; wsize]));
                    }
                }
                Ops::Special => {
                    let name = match &self.instrs[i].arg {
                        Arg::Str(s) => s.clone(),
                        _ => panic!("TGERR: interp SPECIAL arg"),
                    };
                    if name.starts_with('g') {
                        let dim: usize = name[name.len() - 1..].parse().unwrap();
                        values.insert(i, Val::S(vec![gidx[2 - dim] as f64; wsize]));
                    } else {
                        let dim: usize = name[name.len() - 1..].parse().unwrap();
                        let vv: Vec<f64> = warp.iter().map(|w| w[2 - dim] as f64).collect();
                        values.insert(i, Val::S(vv));
                    }
                }
                Ops::Const => {
                    let v = match self.instrs[i]
                        .const_val
                        .as_ref()
                        .expect("TGERR: rayon const")
                    {
                        ConstVal::Int(x) => *x as f64,
                        ConstVal::Float(x) => *x,
                        ConstVal::Bool(b) => *b as i64 as f64,
                        other => panic!("TGERR: interp const {other:?}"),
                    };
                    values.insert(i, Val::S(vec![v; wsize]));
                }
                Ops::Index | Ops::Shrink => {
                    if self.instrs[srcs[0]].addrspace_derived == Some(AddrSpace::Alu)
                        || matches!(src_vals[0], Val::V(_) | Val::S(_))
                    {
                        let ret: Vec<f64> = match (&src_vals[0], &src_vals[1]) {
                            (Val::V(a), Val::S(b)) => (0..wsize)
                                .map(|t| a[b[t.min(b.len() - 1)] as usize][t.min(a[0].len() - 1)])
                                .collect(),
                            (Val::V(a), Val::V(b)) => (0..wsize)
                                .map(|t| {
                                    a[b[t.min(b.len() - 1)][0] as usize][t.min(a[0].len() - 1)]
                                })
                                .collect(),
                            (Val::S(a), _) => a.clone(),
                            _ => panic!("TGERR: interp ALU index srcs"),
                        };
                        values.insert(i, Val::S(ret));
                        i += 1;
                        continue;
                    }
                    let mut ret: Vec<(SendMemView, usize)> = Vec::new();
                    match (&src_vals[0], &src_vals[1]) {
                        (Val::M(ms), Val::S(os)) => {
                            for (m, o) in ms.iter().zip(os.iter()) {
                                ret.push((*m, *o as usize));
                            }
                        }
                        (Val::P(ps), Val::S(os)) => {
                            for ((m, o), o2) in ps.iter().zip(os.iter()) {
                                ret.push((*m, o + *o2 as usize));
                            }
                        }
                        (Val::M(_), Val::V(_)) | (Val::P(_), Val::V(_)) => {
                            let os: Vec<usize> = match &src_vals[1] {
                                Val::V(l) => l
                                    .iter()
                                    .map(|t| t.first().cloned().unwrap_or(0.0) as usize)
                                    .collect(),
                                _ => unreachable!(),
                            };
                            let ms: Vec<SendMemView> = match &src_vals[0] {
                                Val::M(ms) => ms.clone(),
                                Val::P(ps) => ps.iter().map(|(m, _)| *m).collect(),
                                _ => unreachable!(),
                            };
                            for (t, m) in ms.iter().enumerate() {
                                ret.push((*m, os[t.min(os.len() - 1)]));
                            }
                        }
                        _ => panic!("TGERR: interp index srcs u={i} op={op:?}"),
                    }
                    values.insert(i, Val::P(ret));
                }
                Ops::Load => {
                    let load_sz = self.instrs[i].numel;
                    let pairs: Vec<(SendMemView, usize)> = match &src_vals[0] {
                        Val::P(p) => p.clone(),
                        other => panic!(
                            "TGERR: interp load src u={} op={:?} variant={}",
                            i,
                            op_of(srcs[0]),
                            match other {
                                Val::S(_) => "S",
                                Val::V(_) => "V",
                                Val::M(_) => "M",
                                Val::P(_) => "P",
                            }
                        ),
                    };
                    let gated = src_vals.len() >= 3;
                    let tovec = |v: &Val| -> Vec<f64> {
                        match v {
                            Val::S(s) => s.clone(),
                            Val::V(l) => l.iter().map(|t| *t.first().unwrap_or(&0.0)).collect(),
                            _ => panic!("TGERR: interp load gate/alt src"),
                        }
                    };
                    let (alt, gate) = if gated {
                        (tovec(&src_vals[1]), tovec(&src_vals[2]))
                    } else {
                        (Vec::new(), Vec::new())
                    };
                    let mask_at = |t: usize| -> bool {
                        let em = exec_masks.last().unwrap();
                        em[t.min(em.len() - 1)] && (!gated || gate[t.min(gate.len() - 1)] != 0.0)
                    };
                    let alt_at = |t: usize| -> f64 {
                        if gated {
                            alt[t.min(alt.len() - 1)]
                        } else {
                            0.0
                        }
                    };
                    if load_sz > 1 {
                        let lanes: Vec<Vec<f64>> = (0..load_sz)
                            .map(|j| {
                                pairs
                                    .iter()
                                    .enumerate()
                                    .map(|(t, (m, o))| {
                                        if mask_at(t) {
                                            m.load(o + j, dt_of(i))
                                        } else {
                                            alt_at(t)
                                        }
                                    })
                                    .collect()
                            })
                            .collect();
                        values.insert(i, Val::V(lanes));
                    } else {
                        let tv: Vec<f64> = pairs
                            .iter()
                            .enumerate()
                            .map(|(t, (m, o))| {
                                if mask_at(t) {
                                    m.load(*o, dt_of(i))
                                } else {
                                    alt_at(t)
                                }
                            })
                            .collect();
                        values.insert(i, Val::S(tv));
                    }
                }
                Ops::Cast => {
                    match &src_vals[0] {
                        Val::S(vs) => {
                            let r: Vec<f64> = vs.iter().map(|&v| truncate(v, dt_of(i))).collect();
                            values.insert(i, Val::S(r));
                        }
                        Val::V(ls) => {
                            let r: Vec<Vec<f64>> = ls
                                .iter()
                                .map(|tv| tv.iter().map(|&v| truncate(v, dt_of(i))).collect())
                                .collect();
                            values.insert(i, Val::V(r));
                        }
                        _ => panic!("TGERR: interp cast src"),
                    };
                }
                Ops::Bitcast => {
                    let src_dt = dt_of(srcs[0]);
                    let dst_dt = dt_of(i);
                    let bc = |v: f64| -> f64 {
                        let bytes = to_storage_scalar(v, src_dt);
                        from_storage_scalar(&bytes, dst_dt)
                    };
                    match &src_vals[0] {
                        Val::S(vs) => {
                            let r: Vec<f64> = vs.iter().map(|&v| bc(v)).collect();
                            values.insert(i, Val::S(r));
                        }
                        Val::V(ls) => {
                            let r: Vec<Vec<f64>> = ls
                                .iter()
                                .map(|tv| tv.iter().map(|&v| bc(v)).collect())
                                .collect();
                            values.insert(i, Val::V(r));
                        }
                        _ => panic!("TGERR: interp bitcast src"),
                    }
                }
                Ops::Stack => {
                    let mut lanes: Vec<Vec<f64>> = Vec::new();
                    for (k, sv_) in src_vals.iter().enumerate() {
                        match sv_ {
                            Val::S(vs) => lanes.push(vs.clone()),
                            Val::V(l) => {
                                let width = l.first().map(|t| t.len()).unwrap_or(1);
                                for j in 0..width {
                                    lanes.push(
                                        l.iter()
                                            .map(|t| t.get(j).cloned().unwrap_or(0.0))
                                            .collect(),
                                    );
                                }
                            }
                            Val::P(ps) => {
                                let dt = dt_of(srcs[k]);
                                lanes.push(ps.iter().map(|(m, o)| m.load(*o, dt)).collect());
                            }
                            _ => panic!("TGERR: interp STACK src u={i}"),
                        }
                    }
                    values.insert(i, Val::V(lanes));
                }
                _ if group::in_(group::ALU, op) => {
                    if src_vals.len() == 1 {
                        match &src_vals[0] {
                            Val::S(a) => {
                                let r: Vec<f64> = a
                                    .iter()
                                    .map(|&x| truncate(exec_alu(op, x, 0.0), dt_of(i)))
                                    .collect();
                                values.insert(i, Val::S(r));
                            }
                            Val::V(a) => {
                                let r: Vec<Vec<f64>> = a
                                    .iter()
                                    .map(|tv| {
                                        tv.iter()
                                            .map(|&x| truncate(exec_alu(op, x, 0.0), dt_of(i)))
                                            .collect()
                                    })
                                    .collect();
                                values.insert(i, Val::V(r));
                            }
                            _ => panic!("TGERR: interp unary alu src"),
                        }
                        i += 1;
                        continue;
                    }
                    if src_vals.len() == 3 {
                        let svt = |v: &Val| match v {
                            Val::S(vs) => vs.clone(),
                            Val::V(l) => l
                                .iter()
                                .map(|t| t.first().cloned().unwrap_or(0.0))
                                .collect(),
                            _ => panic!("TGERR: interp ternary alu src"),
                        };
                        let (a, b, c) = (svt(&src_vals[0]), svt(&src_vals[1]), svt(&src_vals[2]));
                        let n = a.len().max(b.len()).max(c.len());
                        let r: Vec<f64> = (0..n)
                            .map(|k| {
                                truncate(
                                    exec_alu_ternary(
                                        op,
                                        a[k.min(a.len() - 1)],
                                        b[k.min(b.len() - 1)],
                                        c[k.min(c.len() - 1)],
                                    ),
                                    dt_of(i),
                                )
                            })
                            .collect();
                        values.insert(i, Val::S(r));
                        i += 1;
                        continue;
                    }
                    match (&src_vals[0], &src_vals[1]) {
                        (Val::S(a), Val::S(b)) => {
                            let r: Vec<f64> = a
                                .iter()
                                .zip(b.iter())
                                .map(|(&x, &y)| truncate(exec_alu(op, x, y), dt_of(i)))
                                .collect();
                            values.insert(i, Val::S(r));
                        }
                        (Val::V(a), Val::S(b)) | (Val::S(b), Val::V(a)) => {
                            let r: Vec<Vec<f64>> = a
                                .iter()
                                .map(|tv| {
                                    tv.iter()
                                        .map(|&x| truncate(exec_alu(op, x, b[0]), dt_of(i)))
                                        .collect()
                                })
                                .collect();
                            values.insert(i, Val::V(r));
                        }
                        (Val::V(a), Val::V(b)) => {
                            let r: Vec<Vec<f64>> = a
                                .iter()
                                .zip(b.iter())
                                .map(|(ta, tb)| {
                                    ta.iter()
                                        .zip(tb.iter())
                                        .map(|(&x, &y)| truncate(exec_alu(op, x, y), dt_of(i)))
                                        .collect()
                                })
                                .collect();
                            values.insert(i, Val::V(r));
                        }
                        _ => panic!("TGERR: interp alu srcs"),
                    };
                }
                other => panic!("TGERR: rayon op {other:?} not ported"),
            }
            i += 1;
        }
    }

    fn loop_end_index(&self, srcs: &[usize], range: usize) -> usize {
        self.instrs
            .iter()
            .position(|x| x.op == Ops::End && x.srcs.len() >= 2 && x.srcs[1] == range)
            .unwrap_or_else(|| {
                panic!("TGERR: interp missing END for RANGE (range instr {range}, srcs {srcs:?})")
            })
    }
}

/// Ops the owned executor implements. Anything else makes `compile` fall back.
fn supported_op(op: Ops) -> bool {
    matches!(
        op,
        Ops::End
            | Ops::Range
            | Ops::If
            | Ops::Endif
            | Ops::After
            | Ops::Barrier
            | Ops::Sink
            | Ops::Noop
            | Ops::Group
            | Ops::Store
            | Ops::Param
            | Ops::Buffer
            | Ops::Special
            | Ops::Const
            | Ops::Index
            | Ops::Shrink
            | Ops::Load
            | Ops::Cast
            | Ops::Bitcast
            | Ops::Stack
    ) || group::in_(group::ALU, op)
}

#[derive(Clone)]
enum Val {
    S(Vec<f64>),
    V(Vec<Vec<f64>>),
    P(Vec<(SendMemView, usize)>),
    M(Vec<SendMemView>),
}

impl SendMemView {
    /// An owned scratch window (REG buffers / non-shared locals). The backing
    /// `Box<[u8]>` is moved into `scratch`, which the caller keeps alive for the
    /// duration of the launch and drops afterwards.
    pub(crate) fn from_owned(
        bytes: Vec<u8>,
        elemsize: usize,
        count: usize,
        scratch: &mut Vec<Box<[u8]>>,
    ) -> SendMemView {
        let boxed = bytes.into_boxed_slice();
        let ptr = boxed.as_ptr() as *mut u8;
        scratch.push(boxed);
        SendMemView {
            ptr,
            offset: 0,
            elemsize,
            count,
        }
    }
}
