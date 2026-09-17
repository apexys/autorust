//! Range machinery for the scheduler (M4): `AxisType`, range construction,
//! and the `ranges` / `ended_ranges` / `get_idx` / `get_valid` properties
//! from uop/ops.py.

use super::arg::{Arg, EnumClass, EnumVal};
use crate::dtype::ConstVal;
use super::ctx::UOpCtx;
use super::ops::Ops;
use super::sint::SVal;
use super::UopId;
use crate::dtype::{self, BOOL};
use std::rc::Rc;

/// `AxisType` enum values, mirroring Python's `auto()` numbering.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AxisType {
    Device = 1,
    Global = 2,
    Warp = 3,
    Local = 4,
    Weak = 5,
    GroupReduce = 6,
    Reduce = 7,
    Upcast = 8,
    Unroll = 9,
    Thread = 10,
    Placeholder = 11,
    Loop = 12,
}

impl AxisType {
    pub fn from_val(v: u16) -> AxisType {
        match v {
            1 => AxisType::Device,
            2 => AxisType::Global,
            3 => AxisType::Warp,
            4 => AxisType::Local,
            5 => AxisType::Weak,
            6 => AxisType::GroupReduce,
            7 => AxisType::Reduce,
            8 => AxisType::Upcast,
            9 => AxisType::Unroll,
            10 => AxisType::Thread,
            11 => AxisType::Placeholder,
            12 => AxisType::Loop,
            _ => panic!("TGERR: bad AxisType {v}"),
        }
    }

    pub fn to_enum_val(self) -> EnumVal {
        EnumVal { class: EnumClass::AxisType, value: self as u16 }
    }

    pub fn from_arg(a: &Arg) -> AxisType {
        match a {
            Arg::Enum(e) if e.class == EnumClass::AxisType => AxisType::from_val(e.value),
            other => panic!("TGERR: expected AxisType arg, got {other:?}"),
        }
    }

    /// RANGE arg tuple item at position 1.
    pub fn range_axis_type(ctx: &UOpCtx, r: UopId) -> AxisType {
        match ctx.arg_of(r) {
            Arg::Seq { tuple: true, items } => AxisType::from_arg(&items[1]),
            other => panic!("TGERR: bad RANGE arg {other:?}"),
        }
    }

    /// RANGE arg axis id (position 0).
    pub fn range_axis_id(ctx: &UOpCtx, r: UopId) -> i64 {
        match ctx.arg_of(r) {
            Arg::Seq { tuple: true, items } => match &items[0] {
                Arg::Int(i) => *i,
                other => panic!("TGERR: bad RANGE axis id {other:?}"),
            },
            other => panic!("TGERR: bad RANGE arg {other:?}"),
        }
    }
}

impl UOpCtx {
    /// `UOp.range(end, axis_id, axis_type)` (dtype=weakint, no extra srcs).
    pub fn range(&mut self, end: SVal, axis_id: i64, axis_type: AxisType) -> UopId {
        let end_u = self.sint_to_uop_dt(end, dtype::WEAKINT);
        let arg = Arg::Seq {
            tuple: true,
            items: vec![Arg::Int(axis_id), Arg::Enum(axis_type.to_enum_val())],
        };
        self.uop(Ops::Range, Some(dtype::WEAKINT), vec![end_u], arg)
    }

    /// `sint_to_uop(end, dtype)`.
    pub fn sint_to_uop_dt(&mut self, s: SVal, dt: crate::dtype::DTypeId) -> UopId {
        match s {
            SVal::U(u) => u,
            v => {
                let cv = match v {
                    SVal::I(i) => ConstVal::Int(i),
                    _ => panic!("TGERR: bad sint value"),
                };
                self.const_py(cv, Some(dt))
            }
        }
    }

    /// `u.end(*src)` — END with extra srcs, or self when none.
    pub fn end(&mut self, id: UopId, srcs: Vec<UopId>) -> UopId {
        if srcs.is_empty() {
            return id;
        }
        let mut all = vec![id];
        all.extend(srcs);
        self.uop(Ops::End, Some(crate::dtype::VOID), all, Arg::None)
    }

    /// `u.valid(cond)` — `cond.where(self, Invalid)` (no broadcast on scalars).
    pub fn valid(&mut self, id: UopId, cond: UopId) -> UopId {
        let inv = self.const_invalid();
        self.where_(cond, id, inv)
    }

    pub fn const_invalid(&mut self) -> UopId {
        self.uop(Ops::Const, Some(BOOL), vec![], Arg::Const(ConstVal::Invalid))
    }

    /// `u.index(*srcs)` (ops.py:559): plain INDEX ctor, but fold
    /// `STACK.index(single_const)` to the indexed child (ops.py:561). tinygrad
    /// relies on this shortcut (e.g. `do_devectorize`'s `x.index(*idx_c)`);
    /// without it a `STACK` can leak into `LOAD`/`SHRINK`.
    pub fn index(&mut self, id: UopId, srcs: Vec<UopId>) -> UopId {
        // ops.py:561 — `len(srcs) == 1 and src[0].op is CONST and self.op is STACK`
        if srcs.len() == 1 && self.op_of(srcs[0]) == Ops::Const && self.op_of(id) == Ops::Stack {
            let i = match self.val(srcs[0]) {
                ConstVal::Int(v) => v as usize,
                other => panic!("TGERR: non-int STACK index {other:?}"),
            };
            return self.srcs(id)[i];
        }
        let mut all = vec![id];
        all.extend(srcs);
        self.uop(Ops::Index, None, all, Arg::None)
    }

    /// `u.get_idx()` — the idx part of a WHERE-wrapped range.
    pub fn get_idx(&mut self, id: UopId) -> UopId {
        if self.op_of(id) == Ops::Stack {
            let srcs = self.srcs(id).to_vec();
            let idxs: Vec<UopId> = srcs.iter().map(|&s| self.get_idx(s)).collect();
            return crate::tensor::stack(self, idxs);
        }
        if self.op_of(id) == Ops::Where {
            let srcs = self.srcs(id);
            if srcs.len() >= 3 && self.is_invalid(srcs[2]) {
                return srcs[1];
            }
        }
        id
    }

    /// `u.get_valid()` — the valid part of a WHERE-wrapped range.
    pub fn get_valid(&mut self, id: UopId) -> UopId {
        if self.op_of(id) == Ops::Stack {
            let srcs = self.srcs(id).to_vec();
            let vs: Vec<UopId> = srcs.iter().map(|&s| self.get_valid(s)).collect();
            return crate::tensor::stack(self, vs);
        }
        if self.op_of(id) == Ops::Where {
            let srcs = self.srcs(id);
            if srcs.len() >= 3 && self.is_invalid(srcs[2]) {
                return srcs[0];
            }
        }
        self.const_bool(!self.is_invalid(id))
    }

    /// `ended_ranges` property (ops.py:480).
    pub fn ended_ranges(&mut self, id: UopId) -> Vec<UopId> {
        let (op, _dt, srcs, _arg, _tag) = self.snapshot(id);
        match op {
            Ops::End | Ops::Call | Ops::Function => {
                // python: self.src[range_start[op]:] — raw srcs, NOT recursive
                srcs[1..].to_vec()
            }
            Ops::Wmma => srcs[3..].to_vec(),
            Ops::Linear => srcs.to_vec(),
            Ops::Unshard => srcs[1..].to_vec(),
            Ops::After => {
                let mut ret = vec![];
                for &s in &srcs[1..] {
                    ret.extend(self.ended_ranges(s));
                }
                ret
            }
            Ops::Reduce => srcs[1..].to_vec(),
            Ops::Stage => srcs[1..].to_vec(),
            _ => vec![],
        }
    }

    /// `_ranges`+`ranges` recursive property (ops.py:489/504), memoized in
    /// `range_c`. Insertion-ordered like the Python dict.
    pub fn ranges(&mut self, id: UopId) -> Rc<Vec<UopId>> {
        if let Some(Some(cached)) = self.range_c[id.0 as usize].clone() {
            return cached;
        }
        thread_local! {
            static IN_PROGRESS: std::cell::RefCell<Vec<UopId>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        IN_PROGRESS.with(|ip| {
            let mut ip = ip.borrow_mut();
            if ip.contains(&id) {
                panic!("TGERR: range cycle detected: node ids {:?} -> {} (op {:?}) — chain ops: {:?}", ip.iter().map(|u| u.0).collect::<Vec<_>>(), id.0, self.op_of(id), ip.iter().map(|u| (u.0, format!("{:?}", self.op_of(*u)))).collect::<Vec<_>>());
            }
            ip.push(id);
        });
        let out = self.ranges_inner(id);
        IN_PROGRESS.with(|ip| {
            let mut ip = ip.borrow_mut();
            ip.retain(|&x| x != id);
        });
        out
    }

    fn ranges_inner(&mut self, id: UopId) -> Rc<Vec<UopId>> {
        // compute over the toposort bottom-up, then cache each node
        let ts = self.toposort(id);
        for &n in ts.iter().rev() {
            if self.range_c[n.0 as usize].is_none() {
                let r = Rc::new(self.ranges_uncached(n));
                self.range_c[n.0 as usize] = Some(Some(r));
            }
        }
        match &self.range_c[id.0 as usize] {
            Some(Some(r)) => r.clone(),
            _ => {
                let r = Rc::new(self.ranges_uncached(id));
                self.range_c[id.0 as usize] = Some(Some(r.clone()));
                r
            }
        }
    }

    fn ranges_uncached(&mut self, id: UopId) -> Vec<UopId> {
        let mut ret: Vec<UopId> = vec![];
        let srcs = self.srcs(id).to_vec();
        for s in srcs {
            for &u in self.ranges(s).iter() {
                if !ret.contains(&u) {
                    ret.push(u);
                }
            }
        }
        for er in self.ended_ranges(id) {
            if self.op_of(er) == Ops::Range {
                ret.retain(|&u| u != er);
            } else {
                let er_ranges = self.ranges(er);
                ret.retain(|u| !er_ranges.contains(u));
            }
        }
        if self.op_of(id) == Ops::Range {
            if !ret.contains(&id) {
                ret.insert(0, id);
            }
        }
        ret
    }

    /// `u.bufferize(*rngs, arg=opts)` — STAGE node.
    pub fn stage(&mut self, id: UopId, rngs: Vec<UopId>, arg: Arg) -> UopId {
        let dt = self.dt_of(id);
        let mut srcs = vec![id];
        srcs.extend(rngs);
        self.uop(Ops::Stage, Some(dt), srcs, arg)
    }
}

/// `new_range(s, axistype)` from IndexingContext: 1-sized ranges become CONST 0
/// (python `resolve(s != 1)` — raises on symbolic ambiguity).
pub fn new_range(ctx: &mut UOpCtx, s: SVal, axistype: AxisType) -> UopId {
    if let SVal::U(u) = s {
        if ctx.op_of(u) == Ops::Range {
            return u;
        }
    }
    // lenient like resolve_ord's other callers: ambiguous symbolic sizes make a
    // range (the CONST-0 path only matters for concrete 1-sized axes)
    if ctx.resolve_ord(s, super::ctx::CmpOrd::Ne, SVal::I(1), true) {
        ctx.range(s, ctx.range_idx, axistype)
    } else {
        ctx.const_py(ConstVal::Int(0), Some(dtype::WEAKINT))
    }
}
