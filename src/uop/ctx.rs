//! The UOp context: arena + hash-consing + recursive properties, mirroring
//! Python's `UOpMetaClass.__call__` (ops.py:186) plus the `UOp` methods the
//! M2 rewrite rules use.
//!
//! Built in steps:
//!   1. construction (`uop`, `dtype_from_uop`, const sugar, `val`)
//!   2. shape (movement ops, broadcasting)
//!   3. min_max (symbolic ranges)
//!   4. algebraic sugar (binop/where/cast/usum/...) + symbolic helpers
//!
//! Property computations may fail exactly where Python raises; they `panic!`
//! with a `TGERR:` prefix. The oracle binaries catch these and report the
//! graph as errored (Python raises → rewrite aborts → differential skips it).

use super::alu::{cdiv, floordiv, py_mod};
use super::arg::{Arg, DeviceSpec, ParamArg};
use super::graph::{Graph, UopId};
use super::ops::{Ops, group};
use super::sint::SVal;
use crate::dtype::{
    self, AddrSpace, ConstVal, DTypeId, SINTS, WEAKFLOAT, WEAKINT, is_float, is_int, is_unsigned,
    least_upper_dtype, weak_dtype,
};
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::{Rc, Weak};

pub type Shape = Vec<SVal>;

/// weak dtype bounds saturate to ±(i128::MAX/4); anything at or beyond this
/// magnitude is treated as "unbounded" by the min/max arithmetic (see the
/// `poisoned` guard — Python's exact bignums never produce tight results for
/// unbounded operands).
pub const MM_SENTINEL: i128 = i128::MAX / 4;

pub struct UOpCtx {
    pub g: Graph,
    /// IndexingContext-style range counter (used by new_range)
    pub range_idx: i64,
    /// Set when the context is owned by a shared [`crate::tensor::Ctx`]; lets
    /// `Tensor::from_uop` capture the same handle without threading it through
    /// every internal function.
    pub(crate) self_ref: Weak<RefCell<UOpCtx>>,
}

impl Default for UOpCtx {
    fn default() -> Self {
        Self::new()
    }
}

impl UOpCtx {
    pub fn new() -> Self {
        UOpCtx {
            g: Graph::new(),
            range_idx: 0,
            self_ref: Weak::new(),
        }
    }

    /// Create a context owned by a shared handle. Used by
    /// [`crate::tensor::Ctx::new`]; a plain [`UOpCtx::new`] cannot back
    /// `Tensor`s because it has no stable address to hand out.
    pub fn shared() -> Rc<RefCell<UOpCtx>> {
        Rc::new_cyclic(|weak| {
            RefCell::new(UOpCtx {
                g: Graph::new(),
                range_idx: 0,
                self_ref: weak.clone(),
            })
        })
    }

    /// The shared handle, if this context was created with [`UOpCtx::shared`].
    pub(crate) fn handle(&self) -> Option<Rc<RefCell<UOpCtx>>> {
        self.self_ref.upgrade()
    }

    /// No-op retained for the oracle/bridge API (property caches now live on
    /// nodes, so no arena-sized slot growth is needed).
    pub fn add_cache_slots(&mut self) {}

    /// No-op retained for the oracle/bridge API.
    pub fn grow_to(&mut self, _n: usize) {}

    /// Attach (or overwrite) the autograd gradient for tensor uop `target`
    /// (stored on the node, so it is freed with the tensor's subgraph).
    pub fn set_grad(&self, target: &UopId, grad: UopId) {
        target.node.grad.replace(Some(grad));
    }

    /// Read the autograd gradient attached to tensor uop `target`, if any.
    pub fn grad(&self, target: &UopId) -> Option<UopId> {
        target.node.grad.borrow().clone()
    }

    /// Drop the gradient attached to tensor uop `target` (tinygrad `Optimizer.zero_grad`).
    pub fn clear_grad(&self, target: &UopId) {
        target.node.grad.replace(None);
    }

    /// `UOp.const(bool)` in python's bool dtype.
    pub fn const_bool(&mut self, b: bool) -> UopId {
        self.const_py(crate::dtype::ConstVal::Bool(b), Some(crate::dtype::BOOL))
    }

    // ---------- construction ----------

    /// Raw intern with cache-slot growth (engine rebuild path).
    pub fn intern_raw(
        &mut self,
        op: Ops,
        dtype: DTypeId,
        srcs: Vec<UopId>,
        arg: Arg,
        tag: Arg,
    ) -> UopId {
        self.g.intern(op, dtype, srcs, arg, tag)
    }

    /// Python `UOp(op, dtype=None, src, arg)` constructor: derives dtype when
    /// omitted (`dtype_from_uop(op, src, arg) or dtypes.void`).
    pub fn uop(&mut self, op: Ops, dtype: Option<DTypeId>, srcs: Vec<UopId>, arg: Arg) -> UopId {
        let dt = dtype.unwrap_or_else(|| {
            self.dtype_from_uop(op, &srcs, &arg)
                .unwrap_or(crate::dtype::VOID)
        });
        self.g.intern(op, dt, srcs, arg, Arg::None)
    }

    pub fn uop_tag(
        &mut self,
        op: Ops,
        dtype: Option<DTypeId>,
        srcs: Vec<UopId>,
        arg: Arg,
        tag: Arg,
    ) -> UopId {
        let dt = dtype.unwrap_or_else(|| {
            self.dtype_from_uop(op, &srcs, &arg)
                .unwrap_or(crate::dtype::VOID)
        });
        self.g.intern(op, dt, srcs, arg, tag)
    }

    /// Python `UOp.const(b, dtype=None)` for scalar values.
    pub fn const_py(&mut self, v: ConstVal, dtype: Option<DTypeId>) -> UopId {
        let dt = match (&v, dtype) {
            (_, Some(d)) => d,
            (ConstVal::Bool(_), None) | (ConstVal::Invalid, None) => crate::dtype::BOOL,
            (ConstVal::Float(_), None) => WEAKFLOAT,
            (_, None) => WEAKINT,
        };
        let arg = Arg::Const(dtype_const_canon(dt, v));
        self.uop(Ops::Const, Some(dt), vec![], arg)
    }

    /// `UOp.const` from an i64 (common in rules).
    pub fn cint(&mut self, v: i64) -> UopId {
        self.const_py(ConstVal::Int(v as i128), None)
    }

    pub fn cint_dt(&mut self, v: i64, dt: DTypeId) -> UopId {
        self.const_py(ConstVal::Int(v as i128), Some(dt))
    }

    pub fn cfloat(&mut self, v: f64) -> UopId {
        self.const_py(ConstVal::Float(v), None)
    }

    /// Python `UOp.variable(name, min, max, dtype=weakint, param=False)`.
    pub fn variable(
        &mut self,
        name: &str,
        vmin: ConstVal,
        vmax: ConstVal,
        dtype: DTypeId,
        param: bool,
    ) -> UopId {
        let arg = ParamArg {
            slot: -1,
            dtype,
            vmin_vmax: Some((vmin, vmax)),
            multiple_of: Some(1),
            name: Some(name.to_string()),
            addrspace: Some(AddrSpace::Alu),
            axis: None,
            device: None,
            volatile: false,
        };
        let shape0 = self.shape_arg(&[]); // shape_to_shape_arg(())
        let op = if param { Ops::Param } else { Ops::Buffer };
        self.uop(op, None, vec![shape0], Arg::Param(arg))
    }

    // ---------- field access ----------

    #[inline]
    pub fn op_of(&self, id: &UopId) -> Ops {
        id.op
    }

    #[inline]
    pub fn dt_of(&self, id: &UopId) -> DTypeId {
        id.dtype
    }

    #[inline]
    pub fn srcs<'a>(&self, id: &'a UopId) -> &'a [UopId] {
        &id.srcs
    }

    #[inline]
    pub fn arg_of<'a>(&self, id: &'a UopId) -> &'a Arg {
        &id.arg
    }

    #[inline]
    pub fn tag_of<'a>(&self, id: &'a UopId) -> &'a Arg {
        &id.tag
    }

    /// Node snapshot (clone) so mutations of the arena don't alias.
    pub fn snapshot(&self, id: &UopId) -> (Ops, DTypeId, Vec<UopId>, Arg, Arg) {
        (
            id.op,
            id.dtype,
            id.srcs.clone(),
            id.arg.clone(),
            id.tag.clone(),
        )
    }

    /// CONST value (`uop.val`).
    pub fn val(&self, id: &UopId) -> ConstVal {
        assert!(id.op == Ops::Const, "val is only for CONST");
        match &id.arg {
            Arg::Const(c) => c.clone(),
            Arg::Int(i) => ConstVal::Int(*i as i128),
            Arg::UInt(u) => ConstVal::Int(*u as i128),
            Arg::Bool(b) => ConstVal::Bool(*b),
            Arg::Float(f) => ConstVal::Float(*f),
            Arg::NonFinite(super::arg::NonFinite::Nan) => ConstVal::Float(f64::NAN),
            Arg::NonFinite(super::arg::NonFinite::PosInf) => ConstVal::Float(f64::INFINITY),
            Arg::NonFinite(super::arg::NonFinite::NegInf) => ConstVal::Float(f64::NEG_INFINITY),
            other => panic!("TGERR: bad CONST arg {other:?}"),
        }
    }

    /// CONST arg as the raw stored arg (for pattern matching, `cvar(arg=..)`).
    pub fn const_arg(&self, id: &UopId) -> Option<ConstVal> {
        if id.op != Ops::Const {
            return None;
        }
        Some(self.val(id))
    }

    pub fn is_invalid(&self, id: &UopId) -> bool {
        id.op == Ops::Const && matches!(&id.arg, Arg::Const(ConstVal::Invalid))
    }

    pub fn has_param<'a>(&self, id: &'a UopId) -> Option<&'a ParamArg> {
        match &id.arg {
            Arg::Param(p) => Some(p),
            _ => None,
        }
    }

    /// `uop.base` — strips movement ops (and DETACH).
    pub fn base(&self, id: &UopId) -> UopId {
        let mut u = id.clone();
        loop {
            let op = u.op;
            if group::in_(group::MOVEMENT, op) || op == Ops::Detach {
                u = u.srcs[0].clone();
            } else {
                return u;
            }
        }
    }

    /// `x.replace(...)` — returns id or None if nothing changed (Python returns
    /// self when fields are equal).
    pub fn replace(
        &mut self,
        id: &UopId,
        op: Ops,
        dtype: DTypeId,
        srcs: Vec<UopId>,
        arg: Arg,
        tag: Arg,
    ) -> UopId {
        if id.op == op && id.dtype == dtype && id.srcs == srcs && id.arg == arg && id.tag == tag {
            return id.clone();
        }
        self.g.intern(op, dtype, srcs, arg, tag)
    }

    /// `rtag(tag=True)`.
    pub fn rtag(&mut self, id: &UopId) -> UopId {
        let (o, d, s, a, _t) = self.snapshot(id);
        self.replace(id, o, d, s, a, Arg::Bool(true))
    }

    // ---------- dtype production rules (ops.py:114-184) ----------

    pub fn promo_dtype(&mut self, srcs: &[UopId]) -> DTypeId {
        let mut dts = srcs.iter().map(|s| self.dt_of(s)).collect::<Vec<_>>();
        dts.dedup();
        if dts.len() == 1 {
            dts[0]
        } else {
            least_upper_dtype(&dts)
        }
    }

    pub fn dtype_from_uop(&mut self, op: Ops, srcs: &[UopId], arg: &Arg) -> Option<DTypeId> {
        use Ops::*;
        match op {
            Store | Linear | Sink | Program | Source | End | Barrier | Group | If | Endif
            | Tuple | Function | CustomFunction | RewriteError | Pyliteral => Some(dtype::VOID),
            Call => {
                // CALL of an opaque body is void; a CALL of an address can return a value
                if srcs.is_empty() {
                    return Some(dtype::VOID);
                }
                if self.dt_of(&srcs[0]) == dtype::VOID {
                    Some(dtype::VOID)
                } else {
                    None
                }
            }
            Custom | Customi | Ins | Noop => None,
            Index => {
                let b = srcs[0].clone();
                if self.op_of(&b) == Param && self.is_image_shape_of(&b) {
                    Some(crate::dtype::FLOAT32)
                } else {
                    Some(self.dt_of(&b))
                }
            }
            Load | Unshard | Reduce | After | Range | Contiguous | ContiguousBackward | Copy
            | Stage | Detach | Mstack | Mselect | Allreduce | Special => Some(self.dt_of(&srcs[0])),
            Cmplt | Cmpne | Cmpeq => Some(crate::dtype::BOOL),
            Sin | Log2 | Exp2 | Sqrt | Reciprocal => {
                if self.is_invalid(&self.base(&srcs[0])) {
                    Some(crate::dtype::BOOL)
                } else {
                    Some(self.least_upper_float(self.dt_of(&srcs[0])))
                }
            }
            Where => {
                if self.dt_of(&srcs[0]) != crate::dtype::BOOL {
                    panic!(
                        "TGERR: where cond must be bool, got {}",
                        self.dt_of(&srcs[0]).attr()
                    );
                }
                Some(self.promo_dtype(&srcs[1..]))
            }
            Stack => {
                if srcs.is_empty() {
                    Some(dtype::VOID)
                } else {
                    Some(self.promo_dtype(srcs))
                }
            }
            Wmma => Some(self.dt_of(&srcs[2])),
            Gettuple => {
                let t0 = srcs[0].clone();
                let in_tuple = if self.op_of(&t0) == Function {
                    self.srcs(&t0)[0].clone()
                } else {
                    t0
                };
                let idx = arg_as_int(arg).expect("GETTUPLE arg int");
                Some(self.dt_of(&self.srcs(&in_tuple)[idx as usize]))
            }
            Getaddr => Some(crate::dtype::UINT64),
            Shl | Shr => {
                for s in srcs {
                    let d = self.dt_of(s);
                    if !(is_int(d) || self.is_invalid(&self.base(s))) {
                        panic!("TGERR: shift operands must be int");
                    }
                }
                Some(self.dt_of(&srcs[0]))
            }
            Buffer | Param => match arg {
                Arg::Param(p) => Some(p.dtype),
                _ => panic!("TGERR: BUFFER/PARAM must have ParamArg"),
            },
            Binary => Some(crate::dtype::UINT8),
            Cast | Bitcast => match arg {
                Arg::Dtype(d) => Some(*d),
                _ => panic!("TGERR: CAST/BITCAST arg must be DType"),
            },
            Const => match arg {
                // order matters: bool is an int subclass in Python
                Arg::Const(c) => match c {
                    ConstVal::Invalid => Some(crate::dtype::BOOL),
                    ConstVal::Bool(_) => Some(crate::dtype::BOOL),
                    ConstVal::Int(_) | ConstVal::Big(_) => Some(WEAKINT),
                    ConstVal::Float(_) => Some(WEAKFLOAT),
                },
                Arg::Bool(_) => Some(crate::dtype::BOOL),
                Arg::Int(_) | Arg::UInt(_) => Some(WEAKINT),
                Arg::Float(_) | Arg::NonFinite(_) => Some(WEAKFLOAT),
                _ => panic!("TGERR: no dtype for CONST with arg {arg:?}"),
            },
            _ => {
                if group::in_(group::UNARY, op) {
                    Some(self.dt_of(&srcs[0]))
                } else if group::in_(group::BROADCASTABLE, op) {
                    Some(self.promo_dtype(srcs))
                } else if group::in_(group::MOVEMENT, op) {
                    Some(self.dt_of(&srcs[0]))
                } else {
                    panic!("TGERR: no dtype for {op:?} with arg {arg:?}")
                }
            }
        }
    }

    pub fn least_upper_float(&self, d: DTypeId) -> DTypeId {
        if d == WEAKINT {
            WEAKFLOAT
        } else if is_float(d) {
            d
        } else {
            least_upper_dtype(&[d, crate::dtype::FLOAT32])
        }
    }

    pub fn is_image_shape_of(&mut self, u: &UopId) -> bool {
        match self.shape(u) {
            Ok(Some(s)) => s.len() == 3 && matches!(s[2], SVal::I(4)),
            _ => false,
        }
    }

    /// `_shape` (ops.py:326-464), cached. `Ok(None)` = no shape (late ops).
    pub fn shape(&mut self, id: &UopId) -> Result<Option<Rc<Shape>>, String> {
        if let Some(cached) = id.shape_c.borrow().clone() {
            return Ok(cached);
        }
        let ret = self.shape_uncached(id)?;
        *id.shape_c.borrow_mut() = Some(ret.clone());
        Ok(ret)
    }

    /// `shape` property — errors when the node has no shape (Python raises).
    pub fn shape_of(&mut self, id: &UopId) -> Result<Rc<Shape>, String> {
        match self.shape(id)? {
            Some(s) => Ok(s),
            None => Err(format!(
                "shape requested, but {:?} doesn't have a shape",
                id.op
            )),
        }
    }

    fn shape_uncached(&mut self, id: &UopId) -> Result<Option<Rc<Shape>>, String> {
        use Ops::*;
        let (op, _dt, srcs, arg, _tag) = self.snapshot(id);
        let shp: Option<Shape> = match op {
            // late ops don't have shape
            If | Barrier | Sink | RewriteError | Endif | Group | Linear | Program | Source
            | Tuple | Function => None,
            // a void CALL has no shape; the return value of a CALL has shape ()
            Call => {
                if self.dt_of(id) == dtype::VOID {
                    None
                } else {
                    Some(vec![])
                }
            }
            Ins => {
                if self.dt_of(id) == dtype::VOID {
                    None
                } else {
                    Some(vec![])
                }
            }
            // special (terrible) case for RESHAPE on NOOP
            Reshape if self.op_of(&srcs[0]) == Noop => Some(self.marg_shape(id.clone())?),
            // hacks for NOOP
            Noop => {
                if !srcs.is_empty() {
                    self.shape(&srcs[0])?.map(|s| (*s).clone())
                } else {
                    None
                }
            }
            Gettuple => {
                let t0 = srcs[0].clone();
                let (in_tuple, through_fn) = if self.op_of(&t0) == Function {
                    (self.srcs(&t0)[0].clone(), true)
                } else {
                    (t0.clone(), false)
                };
                let idx = arg_as_int(&arg).ok_or("GETTUPLE arg int")? as usize;
                let inner = self.shape_of(&self.srcs(&in_tuple)[idx])?;
                if through_fn {
                    // substitute internal PARAMs with the call args
                    let args = self.srcs(&t0)[1..].to_vec();
                    let mut out = Vec::with_capacity(inner.len());
                    for s in inner.iter() {
                        match s {
                            SVal::U(u) => out.push(SVal::U(self.substitute_one(u.clone(), &args)?)),
                            i => out.push(i.clone()),
                        }
                    }
                    Some(out)
                } else {
                    Some((*inner).clone())
                }
            }
            Index => {
                let mut shp: Shape = Vec::new();
                for s in &srcs[1..] {
                    shp.extend(self.shape_of(s)?.iter().cloned());
                }
                let bshape = self.shape_of(&srcs[0])?;
                let skip = srcs.len() - 1;
                shp.extend(bshape.iter().skip(skip).cloned());
                Some(shp)
            }
            Stack => {
                if srcs.is_empty() {
                    Some(vec![])
                } else {
                    let mut shp = vec![SVal::I(srcs.len() as i128)];
                    shp.extend(self.shape_of(&srcs[0])?.iter().cloned());
                    Some(shp)
                }
            }
            Const => Some(vec![]),
            Getaddr => Some(vec![]),
            Range | Special => Some(vec![]),
            Binary => {
                let n = arg_as_int(&arg).ok_or("BINARY arg len")?;
                Some(vec![SVal::I(n as i128)])
            }
            Buffer => Some(self.as_shape((srcs[0]).clone())?),
            Custom | Customi => {
                if self.dt_of(id) == dtype::VOID {
                    None
                } else {
                    let mut shapes: Vec<Shape> = Vec::new();
                    for s in &srcs {
                        if let Some(sh) = self.shape(s)? {
                            shapes.push((*sh).clone());
                        }
                    }
                    if shapes.is_empty() {
                        None
                    } else {
                        Some(self.broadcast_shape(&shapes)?)
                    }
                }
            }
            CustomFunction | Pyliteral => None,
            Stage => {
                // STAGE adds the existing shape to the front
                let mut shp: Shape = Vec::new();
                for r in &srcs[1..] {
                    let vmax = self.vmax(r)?;
                    shp.push(SVal::I(vmax.as_i64_or_panic() as i128 + 1));
                }
                shp.extend(self.shape_of(&srcs[0])?.iter().cloned());
                Some(shp)
            }
            Param => Some(self.as_shape((srcs[0]).clone())?),
            Wmma => {
                let a = self.shape_of(&srcs[0])?;
                let b = self.shape_of(&srcs[1])?;
                let c = self.shape_of(&srcs[2])?;
                // python slicing handles the empty (scalar) shape: `()[:-1] == ()`
                let head = |v: &[SVal]| -> Vec<SVal> {
                    if v.is_empty() {
                        Vec::new()
                    } else {
                        v[..v.len() - 1].to_vec()
                    }
                };
                let mut front = self.broadcast_shape(&[head(&a), head(&b), head(&c)])?;
                front.extend(c.last().cloned());
                Some(front)
            }
            Mstack | Mselect | Detach | Contiguous | ContiguousBackward | After | Load | Copy
            | Allreduce | Store | End => self.shape(&srcs[0])?.map(|s| (*s).clone()),
            Bitcast => {
                let ps = self.shape(&srcs[0])?;
                match ps {
                    None => None,
                    Some(ps) => {
                        let out_sz = self.dt_of(id).itemsize();
                        let in_sz = self.dt_of(&srcs[0]).itemsize();
                        if out_sz != in_sz && !ps.is_empty() {
                            let last = ps.last().unwrap().clone();
                            match last {
                                SVal::I(v) => {
                                    if (v * in_sz as i128) % out_sz as i128 != 0 {
                                        return Err("unsupported size in bitcast".into());
                                    }
                                    let mut sh = ps[..ps.len() - 1].to_vec();
                                    sh.push(SVal::I((v * in_sz as i128) / out_sz as i128));
                                    Some(sh)
                                }
                                SVal::U(u) => {
                                    let mut sh = ps[..ps.len() - 1].to_vec();
                                    sh.push(self.ssimplify_shrink_last(u, in_sz, out_sz)?);
                                    Some(sh)
                                }
                            }
                        } else {
                            Some((*ps).clone())
                        }
                    }
                }
            }
            Unshard if srcs.is_empty() => None,
            _ => {
                // movement ops change the shape
                if group::in_(group::MOVEMENT, op) || op == Unshard || op == Reduce {
                    let ps = match self.shape(&srcs[0])? {
                        Some(p) => p,
                        None => {
                            return Err(format!(
                                "movement op {op:?} requires shape, {:?} doesn't have one",
                                self.op_of(&srcs[0])
                            ));
                        }
                    };
                    return Ok(Some(Rc::new(self.shape_movement(
                        id.clone(),
                        op,
                        &arg,
                        &ps,
                    )?)));
                }
                if group::in_(group::UNARY, op) || op == Cast {
                    return Ok(Some(self.shape_of(&srcs[0])?));
                }
                if group::in_(group::BROADCASTABLE, op) {
                    let mut input_shapes = Vec::with_capacity(srcs.len());
                    for s in &srcs {
                        match self.shape(s)? {
                            Some(sh) => input_shapes.push((*sh).clone()),
                            None => {
                                return Err(format!("None input shape not supported for {op:?}"));
                            }
                        }
                    }
                    if input_shapes.is_empty() {
                        return Err(format!("None input shape not supported for {op:?}"));
                    }
                    return Ok(Some(Rc::new(self.broadcast_shape(&input_shapes)?)));
                }
                return Err(format!("no shape handling for {op:?}"));
            }
        };
        Ok(shp.map(Rc::new))
    }

    /// movement op shapes (ops.py:415-448)
    fn shape_movement(
        &mut self,
        id: UopId,
        op: Ops,
        _arg: &Arg,
        ps: &[SVal],
    ) -> Result<Shape, String> {
        use Ops::*;
        match op {
            Reshape => {
                let marg = self.marg_shape(id)?;
                for x in &marg {
                    if let SVal::I(v) = x
                        && *v < 0
                    {
                        return Err("shape can't contain negative numbers".into());
                    }
                }
                // tinygrad asserts prod(in) == prod(out) with symbolic algebra.
                // Our port only proves concrete products; when either side has a
                // symbolic dim (e.g. a SPECIAL local-range index reshape) we trust
                // graph construction and skip the check rather than hard-fail.
                if let (Some(prod_in), Some(prod_out)) =
                    (super::sint::prod_int(ps), super::sint::prod_int(&marg))
                    && prod_in != prod_out
                {
                    return Err(format!("bad reshape: {ps:?} -> {marg:?}"));
                }
                Ok(marg)
            }
            Expand => {
                let mut out = self.marg_shape(id)?;
                out.extend(ps.iter().cloned());
                Ok(out)
            }
            Permute => {
                let perm: Vec<usize> = self.marg_ints(id)?.iter().map(|&i| i as usize).collect();
                let mut sorted = perm.clone();
                sorted.sort();
                if sorted != (0..ps.len()).collect::<Vec<_>>() {
                    return Err(format!("invalid permutation {perm:?} of len {}", ps.len()));
                }
                Ok(perm.iter().map(|&i| ps[i].clone()).collect())
            }
            Pad | Shrink => {
                let marg = self.marg_pairs_svals(id)?;
                if marg.len() != ps.len() {
                    return Err(format!("invalid pad/shrink {marg:?} for {ps:?}"));
                }
                let mut out = Vec::with_capacity(ps.len());
                for (i, (o, sz)) in marg.iter().enumerate() {
                    match (o, sz.clone()) {
                        (SVal::I(o), SVal::I(sz)) => {
                            // python indexing.py conventions: PAD pairs are
                            // (start, stop) — new dim = stop; SHRINK pairs are
                            // (start, size) — new dim = size (mixin converts
                            // the slice bounds to sizes).
                            let ok = match op {
                                Pad => {
                                    let start_dim = self.sval_add(SVal::I(*o), ps[i].clone());
                                    self.resolve_ord(SVal::I(sz), CmpOrd::Ge, SVal::I(0), true)
                                        && self.resolve_ord(
                                            SVal::I(0),
                                            CmpOrd::Le,
                                            SVal::I(*o),
                                            true,
                                        )
                                        && self.resolve_ord(
                                            start_dim,
                                            CmpOrd::Le,
                                            SVal::I(sz),
                                            true,
                                        )
                                }
                                _ => {
                                    let start_dim = self.sval_add(SVal::I(*o), SVal::I(sz));
                                    self.resolve_ord(SVal::I(0), CmpOrd::Le, SVal::I(*o), true)
                                        && self.resolve_ord(
                                            start_dim,
                                            CmpOrd::Le,
                                            ps[i].clone(),
                                            true,
                                        )
                                }
                            };
                            if !ok {
                                return Err(format!(
                                    "invalid pad/shrink for {ps:?} pair=({o},{sz}) op={op:?}"
                                ));
                            }
                            out.push(SVal::I(sz));
                        }
                        _ => {
                            // symbolic view (e.g. a gid-derived LOCAL window):
                            // the new dim is the pair's size/stop half.
                            match self.ssimplify_const_val(sz.clone()) {
                                Some(v) => out.push(SVal::I(v)),
                                None => out.push(sz.clone()),
                            }
                        }
                    }
                }
                Ok(out)
            }
            Flip => {
                // FLIP's arg is a tuple of bool flags (one per dim)
                let flags: Option<Vec<bool>> = match self.arg_of(&id) {
                    Arg::Seq { tuple: true, items } => items
                        .iter()
                        .map(|x| match x {
                            Arg::Bool(b) => Some(*b),
                            _ => None,
                        })
                        .collect(),
                    _ => None,
                };
                match flags {
                    Some(f) if f.len() == ps.len() => Ok(ps.to_vec()),
                    _ => Err(format!("bad flip on {ps:?}")),
                }
            }
            Unshard => {
                let axes = self.unshard_axes(id.clone())?;
                let srcs = self.srcs(&id).to_vec();
                let mut out = Vec::with_capacity(ps.len());
                for (i, s) in ps.iter().enumerate() {
                    if let Some(pos) = axes.iter().position(|&a| a as usize == i) {
                        let vmax = self.vmax(&srcs[1 + pos])?;
                        out.push(SVal::I(
                            s.clone().as_i64_or_panic() as i128
                                * (vmax.as_i64_or_panic() as i128 + 1),
                        ));
                    } else {
                        out.push(s.clone());
                    }
                }
                Ok(out)
            }
            Reduce => {
                let num_axes = match self.arg_of(&id) {
                    Arg::Seq { tuple: true, items } => arg_as_int(&items[1]).ok_or("REDUCE arg")?,
                    _ => return Err("invalid type for axis".into()),
                };
                if !(0..=ps.len() as i64).contains(&num_axes) {
                    return Err(format!("invalid type for axis: {num_axes}"));
                }
                Ok(ps[num_axes as usize..].to_vec())
            }
            _ => Err(format!("{op:?} is not a MovementOp")),
        }
    }

    /// marg for RESHAPE/EXPAND: the new shape (may be symbolic).
    pub fn marg_shape(&mut self, id: UopId) -> Result<Shape, String> {
        let srcs = self.srcs(&id).to_vec();
        if srcs.len() < 2 {
            panic!(
                "TGERR: marg_shape on {} with {} srcs",
                DebugOp(self.op_of(&id)),
                srcs.len()
            );
        }
        self.as_shape((srcs[1]).clone())
    }

    /// marg for PAD/SHRINK: (start, stop) pairs.
    pub fn marg_pairs(&mut self, id: UopId) -> Result<Vec<(i64, i64)>, String> {
        Ok(self
            .marg_pairs_svals(id)?
            .iter()
            .map(|(a, b)| (a.clone().as_i64_or_panic(), b.clone().as_i64_or_panic()))
            .collect())
    }

    /// A symbolic SVal that simplifies to a concrete int (e.g. the size half
    /// of a SHRINK pair is usually a const).
    fn ssimplify_const_val(&mut self, v: SVal) -> Option<i128> {
        match v {
            SVal::I(i) => Some(i),
            SVal::U(u) => self.ssimplify(u).ok().and_then(|s| match s {
                SVal::I(i) => Some(i),
                _ => None,
            }),
        }
    }

    /// Pad/Shrink pairs as raw SVals — symbolic starts/stops (e.g. a
    /// gid-derived view inside a LOCAL kernel) stay symbolic here and are
    /// resolved by ssimplify in the shape arm.
    pub fn marg_pairs_svals(&mut self, id: UopId) -> Result<Vec<(SVal, SVal)>, String> {
        let srcs = self.srcs(&id).to_vec();
        if srcs.len() >= 3 {
            let starts = self.as_shape((srcs[1]).clone())?;
            let stops = self.as_shape((srcs[2]).clone())?;
            return Ok(starts
                .iter()
                .zip(stops.iter())
                .map(|(a, b)| (a.clone(), b.clone()))
                .collect());
        }
        // arg-carried pairs (schedule-created SHRINK/PAD: ((start, stop), ...));
        // other movement ops (e.g. Expand) arrive via probing callers -> Err.
        let pairs: Vec<(SVal, SVal)> = match self.arg_of(&id) {
            Arg::Seq { tuple: true, items } => items
                .iter()
                .map(|pair| match pair {
                    Arg::Seq { tuple: true, items } if items.len() == 2 => {
                        match (&items[0], &items[1]) {
                            (Arg::Int(a), Arg::Int(b)) => {
                                Ok((SVal::I(*a as i128), SVal::I(*b as i128)))
                            }
                            _ => Err(format!("TGERR: bad pair arg {pair:?}")),
                        }
                    }
                    other => Err(format!("TGERR: bad pair arg {other:?}")),
                })
                .collect::<Result<Vec<_>, String>>()?,
            _ => {
                return Err(format!(
                    "TGERR: node {} ({:?}) has no pair arg",
                    id.id(),
                    self.op_of(&id)
                ));
            }
        };
        Ok(pairs)
    }

    /// marg for PERMUTE/FLIP: the int list arg.
    pub fn marg_ints(&mut self, id: UopId) -> Result<Vec<i64>, String> {
        match self.arg_of(&id) {
            Arg::Seq { tuple: true, items } => items
                .iter()
                .map(arg_as_int)
                .collect::<Option<Vec<_>>>()
                .ok_or("bad movement arg".to_string()),
            _ => Err("bad movement arg".into()),
        }
    }

    fn unshard_axes(&mut self, id: UopId) -> Result<Vec<i64>, String> {
        match self.arg_of(&id) {
            Arg::Seq { tuple: true, items } => items
                .iter()
                .map(arg_as_int)
                .collect::<Option<Vec<_>>>()
                .ok_or("UNSHARD arg".to_string()),
            _ => Err("UNSHARD arg".into()),
        }
    }

    /// `marg` (ops.py:779-785): the movement op's arg, from srcs for shape-carrying ops.
    pub fn marg_of(&mut self, id: UopId) -> Result<Vec<(i64, i64)>, String> {
        use Ops::*;
        let (op, _, srcs, arg, _) = self.snapshot(&id);
        match op {
            Reshape | Expand => Ok(self
                .as_shape((srcs[1]).clone())?
                .iter()
                .map(|x| (x.clone().as_i64_or_panic(), 0))
                .collect()),
            Pad | Shrink => {
                let starts = self.as_shape((srcs[1]).clone())?;
                let stops = self.as_shape((srcs[2]).clone())?;
                match starts
                    .iter()
                    .zip(stops.iter())
                    .map(|(a, b)| a.clone().as_i64().zip(b.clone().as_i64()))
                    .collect::<Option<Vec<_>>>()
                {
                    Some(pairs) => Ok(pairs),
                    None => {
                        let dbg = |sv: &SVal| match sv {
                            SVal::I(i) => format!("I({i})"),
                            SVal::U(u) => format!(
                                "U({}: {:?} arg={:?} srcs={:?})",
                                u.id(),
                                self.op_of(u),
                                self.arg_of(u),
                                self.srcs(u)
                                    .iter()
                                    .map(|s| (s.id(), self.op_of(s)))
                                    .collect::<Vec<_>>()
                            ),
                        };
                        panic!(
                            "TGERR: symbolic movement pair on node {} (op {:?}): starts=[{}] stops=[{}]",
                            id.id(),
                            self.op_of(&id),
                            starts.iter().map(dbg).collect::<Vec<_>>().join(", "),
                            stops.iter().map(dbg).collect::<Vec<_>>().join(", ")
                        )
                    }
                }
            }
            Permute | Flip => match &arg {
                Arg::Seq { tuple: true, items } => items
                    .iter()
                    .map(|x| {
                        arg_as_int(x)
                            .map(|i| (i, 0))
                            .ok_or("bad movement arg".to_string())
                    })
                    .collect(),
                _ => Err(format!("{op:?} is not a MovementOp")),
            },
            _ => Err(format!("{op:?} is not a MovementOp")),
        }
    }

    /// `as_shape` (ops.py:773-777)
    pub fn as_shape(&mut self, u: UopId) -> Result<Shape, String> {
        let (op, _, srcs, _arg, _) = self.snapshot(&u);
        if op == Ops::Const {
            let v = self.val(&u);
            return Ok(vec![self.constval_to_sval(&v, "as_shape")?]);
        }
        if op != Ops::Stack {
            return Ok(vec![self.ssimplify(u)?]);
        }
        let mut out = Vec::with_capacity(srcs.len());
        for s in &srcs {
            if self.op_of(s) == Ops::Const {
                out.push(self.constval_to_sval(&self.val(s), "as_shape")?);
            } else {
                out.push(self.ssimplify(s.clone())?);
            }
        }
        Ok(out)
    }

    /// `shape_to_shape_arg` (ops.py:96-100)
    pub fn shape_arg(&mut self, shape: &[SVal]) -> UopId {
        let mut src = Vec::with_capacity(shape.len());
        for x in shape {
            match x {
                SVal::U(u) => {
                    if !is_int(self.dt_of(u)) {
                        panic!("TGERR: shape must be int");
                    }
                    src.push(u.clone());
                }
                // Python: UOp.const(x) -> from_py(int) -> weakint
                SVal::I(v) => src.push(self.const_py(crate::dtype::ConstVal::Int(*v), None)),
            }
        }
        if src.len() == 1 {
            (src[0]).clone()
        } else {
            self.uop(Ops::Stack, None, src, Arg::None)
        }
    }

    /// `_broadcast_shape` (ops.py:70-78)
    pub fn broadcast_shape(&mut self, shapes: &[Shape]) -> Result<Shape, String> {
        if shapes.iter().all(|s| s == &shapes[0]) {
            return Ok(shapes[0].clone());
        }
        let max_dim = shapes.iter().map(|s| s.len()).max().unwrap_or(0);
        let mut ret = Vec::with_capacity(max_dim);
        for d in 0..max_dim {
            let mut rest: Vec<SVal> = Vec::new();
            for s in shapes {
                let pad = max_dim - s.len();
                let v = if d < pad {
                    SVal::I(1)
                } else {
                    s[d - pad].clone()
                };
                let non_one = match v {
                    SVal::U(_) => true,
                    SVal::I(x) => x != 1,
                };
                if non_one && !rest.contains(&v) {
                    rest.push(v);
                }
            }
            if rest.len() > 1 {
                return Err(format!(
                    "shape mismatch: objects cannot be broadcast to a single shape {shapes:?}"
                ));
            }
            ret.push(rest.first().cloned().unwrap_or(SVal::I(1)));
        }
        Ok(ret)
    }

    /// `_min_max` (ops.py:1050-1105), cached.
    pub fn min_max(&mut self, id: &UopId) -> Result<(SVal, SVal), String> {
        if let Some(cached) = &*id.mm_c.borrow() {
            return cached.clone().ok_or_else(|| "no min/max".to_string());
        }
        let ret = self.min_max_uncached(id);
        let ok = match &ret {
            Ok(v) => Some(Some(v.clone())),
            Err(_) => Some(None),
        };
        *id.mm_c.borrow_mut() = ok;
        ret
    }

    pub fn vmin(&mut self, id: &UopId) -> Result<SVal, String> {
        Ok(self.min_max(id)?.0)
    }

    pub fn vmax(&mut self, id: &UopId) -> Result<SVal, String> {
        Ok(self.min_max(id)?.1)
    }

    fn min_max_uncached(&mut self, id: &UopId) -> Result<(SVal, SVal), String> {
        use Ops::*;
        let (op, dt, srcs, _arg, _) = self.snapshot(id);
        // int-typed binary ops (Python: GroupOp.Binary and not float)
        if group::in_(group::BINARY, op) && !is_float(dt) && !srcs.is_empty() {
            let (s0min, s0max) = self.min_max(&srcs[0])?;
            let (s1min, s1max) = self.min_max(&srcs[1])?;
            let a0 = s0min.as_i128_or_panic();
            let a1 = s0max.as_i128_or_panic();
            let b0 = s1min.as_i128_or_panic();
            let b1 = s1max.as_i128_or_panic();
            // weak-bound sentinels (±i128::MAX/4-ish) poison the arithmetic arms:
            // Python computes exact huge values there, which are never tight, so
            // the dtype-bounds fallback matches its conclusions.
            let poisoned = a0.abs() >= MM_SENTINEL
                || a1.abs() >= MM_SENTINEL
                || b0.abs() >= MM_SENTINEL
                || b1.abs() >= MM_SENTINEL;
            let ret: (i128, i128) = match op {
                Add if !poisoned => (a0.wrapping_add(b0), a1.wrapping_add(b1)),
                Sub if !poisoned => (a0.wrapping_sub(b1), a1.wrapping_sub(b0)),
                And if is_int(dt) && b0 == b1 && b0 >= 0 => {
                    if a0 < 0 {
                        (0, b1)
                    } else {
                        (0, a1.min(b1))
                    }
                }
                Mul if !poisoned || a0 == 0 && a1 == 0 || b0 == 0 && b1 == 0 => {
                    // 0 * anything == 0 exactly, even for unbounded operands
                    if poisoned {
                        (0, 0)
                    } else {
                        let vals = [
                            a0.wrapping_mul(b0),
                            a0.wrapping_mul(b1),
                            a1.wrapping_mul(b0),
                            a1.wrapping_mul(b1),
                        ];
                        (*vals.iter().min().unwrap(), *vals.iter().max().unwrap())
                    }
                }
                // SHL/SHR on consts only
                Shl if b0 == b1 => {
                    if b0 < 0 {
                        // Python: int << negative raises even for huge lhs
                        panic!("TGERR: negative shift count");
                    }
                    if !poisoned {
                        return Ok((
                            SVal::I(a0.wrapping_shl(b0.min(126) as u32)),
                            SVal::I(a1.wrapping_shl(b0.min(126) as u32)),
                        ));
                    }
                    return self.mm_fallback(id.clone(), dt);
                }
                Shr if b0 == b1 => {
                    if b0 < 0 {
                        panic!("TGERR: negative shift count");
                    }
                    if !poisoned {
                        return Ok((
                            SVal::I(a0.wrapping_shr(b0.min(126) as u32)),
                            SVal::I(a1.wrapping_shr(b0.min(126) as u32)),
                        ));
                    }
                    return self.mm_fallback(id.clone(), dt);
                }
                Cmod => {
                    if b0 == b1 && b0 > 0 {
                        // (0 if s0_vmin > 0 else s0_vmin if 0 >= s0_vmin > -c else -(c-1),
                        //  0 if s0_vmax < 0 else s0_vmax if 0 <= s0_vmax < c else c-1)
                        let lo = if a0 > 0 {
                            0
                        } else if 0 >= a0 && a0 > -b0 {
                            a0
                        } else {
                            -(b1 - 1)
                        };
                        let hi = if a1 < 0 {
                            0
                        } else if 0 <= a1 && a1 < b1 {
                            a1
                        } else {
                            b1 - 1
                        };
                        (lo, hi)
                    } else if b0 > 0 {
                        if a0 >= 0 {
                            (0, b1 - 1)
                        } else if a1 <= 0 {
                            (-(b1 - 1), 0)
                        } else {
                            (-(b1 - 1), b1 - 1)
                        }
                    } else if b1 < 0 {
                        // NOTE: the bound comes from s1_vmin (b0), per Python
                        if a0 >= 0 {
                            (0, -b0 - 1)
                        } else if a1 <= 0 {
                            (b0 + 1, 0)
                        } else {
                            (b0 + 1, -b0 - 1)
                        }
                    } else {
                        return self.mm_fallback(id.clone(), dt);
                    }
                }
                Cdiv => {
                    if (b0 > 0 && b1 > 0) || (b0 < 0 && b1 < 0) {
                        let vals = [cdiv(a0, b0), cdiv(a0, b1), cdiv(a1, b0), cdiv(a1, b1)];
                        (*vals.iter().min().unwrap(), *vals.iter().max().unwrap())
                    } else {
                        return self.mm_fallback(id.clone(), dt);
                    }
                }
                Floordiv => {
                    if a0 > a1 {
                        (0, 0)
                    } else if (b0 > 0 && b1 > 0) || (b0 < 0 && b1 < 0) {
                        let vals = [
                            floordiv(a0, b0),
                            floordiv(a0, b1),
                            floordiv(a1, b0),
                            floordiv(a1, b1),
                        ];
                        (*vals.iter().min().unwrap(), *vals.iter().max().unwrap())
                    } else {
                        return self.mm_fallback(id.clone(), dt);
                    }
                }
                Floormod => {
                    if a0 > a1 {
                        (0, 0)
                    } else if b0 == b1 && b0 > 0 {
                        if floordiv(a0, b0) == floordiv(a1, b0) {
                            (py_mod(a0, b0), py_mod(a1, b0))
                        } else {
                            (0, b1 - 1)
                        }
                    } else if b0 == b1 && b0 < 0 {
                        if floordiv(a0, b0) == floordiv(a1, b0) {
                            (py_mod(a0, b0), py_mod(a1, b0))
                        } else {
                            (b1 + 1, 0)
                        }
                    } else if b0 > 0 {
                        (0, b1 - 1)
                    } else if b1 < 0 {
                        // NOTE: bound from s1_vmin (b0), per Python
                        (b0 + 1, 0)
                    } else {
                        return self.mm_fallback(id.clone(), dt);
                    }
                }
                Xor if b0 == b1 && b1 == -1 => (!a1, !a0),
                Max => (a0.max(b0), a1.max(b1)),
                Cmplt => (if a1 < b0 { 1 } else { 0 }, if a0 < b1 { 1 } else { 0 }),
                Cmpne => (
                    if a1 < b0 || b1 < a0 { 1 } else { 0 },
                    if !(a0 == a1 && a0 == b0 && b0 == b1) {
                        1
                    } else {
                        0
                    },
                ),
                Or if dt == crate::dtype::BOOL => (a0 | b0, a1 | b1),
                And if dt == crate::dtype::BOOL => (a0 & b0, a1 & b1),
                _ => return self.mm_fallback(id.clone(), dt),
            };
            return Ok((SVal::I(ret.0), SVal::I(ret.1)));
        }
        self.mm_fallback(id.clone(), dt)
    }

    /// the non-Binary tail of `_min_max`
    fn mm_fallback(&mut self, id: UopId, dt: DTypeId) -> Result<(SVal, SVal), String> {
        use Ops::*;
        let (op, _, srcs, _arg, _) = self.snapshot(&id);
        // WHERE with int dtype
        if op == Where && is_int(dt) && srcs.len() == 3 {
            let (t0, t1) = self.min_max(&srcs[1])?;
            let (f0, f1) = self.min_max(&srcs[2])?;
            let lo = super::ctx::sval_min(t0, f0);
            let hi = super::ctx::sval_max(t1, f1);
            return Ok((lo, hi));
        }
        // PARAM/BUFFER with vmin_vmax
        if (op == Param || op == Buffer)
            && srcs.len() == 1
            && let Some(p) = self.has_param(&id)
            && let Some((vmin, vmax)) = &p.vmin_vmax
        {
            let a = self.constval_to_sval(vmin, "vmin_vmax")?;
            let b = self.constval_to_sval(vmax, "vmin_vmax")?;
            return Ok((a, b));
        }
        if (op == Range || op == Special) && dt != dtype::VOID && !srcs.is_empty() {
            // (self.src[0]-1).vmax — SUB with const 1; an unbounded src0 falls
            // back to dtype bounds like python's _min_max (never panics)
            let hi = match self.vmax(&srcs[0])? {
                SVal::I(v) => SVal::I(v - 1),
                SVal::U(_) => match self.dtype_bounds(dt)?.1 {
                    SVal::I(v) => SVal::I(v - 1),
                    other => other,
                },
            };
            return Ok((SVal::I(0), hi));
        }
        if op == After {
            return self.min_max(&srcs[0]);
        }
        if op == Stack {
            if srcs.is_empty() {
                // Python: min() over an empty sequence raises
                return Err("empty STACK has no min/max".to_string());
            }
            let mut lo = SVal::I(i128::MAX);
            let mut hi = SVal::I(i128::MIN);
            for s in &srcs {
                let (a, b) = self.min_max(s)?;
                lo = sval_min(lo, a);
                hi = sval_max(hi, b);
            }
            return Ok((lo, hi));
        }
        if op == Const && !self.is_invalid(&id) {
            let v = self.constval_to_sval(&self.val(&id), "min_max")?;
            return Ok((v.clone(), v));
        }
        if op == Index {
            return self.min_max(&srcs[0]);
        }
        if op == Cast {
            let (smin, smax) = self.min_max(&srcs[0])?;
            // rounding is monotone
            let mut smin = smin;
            let mut smax = smax;
            if is_float(dt) {
                smin = self.trunc_sval_to_dtype(smin, dt)?;
                smax = self.trunc_sval_to_dtype(smax, dt)?;
            } else if is_int(dt) {
                smin = smin.trunc_toward_zero();
                smax = smax.trunc_toward_zero();
            }
            let (dmin, dmax) = self.dtype_bounds(dt)?;
            if is_unsigned(dt) {
                let zero = SVal::I(0);
                if sval_cmp(smin.clone(), zero) != std::cmp::Ordering::Less
                    && sval_cmp(smax.clone(), dmax) != std::cmp::Ordering::Greater
                {
                    return Ok((smin, smax));
                }
            } else if dt == WEAKINT
                || crate::dtype::in_set(crate::dtype::FLOATS, dt)
                || SINTS.contains(&dt)
            {
                let lo = if sval_cmp(smin.clone(), dmin.clone()) == std::cmp::Ordering::Less {
                    dmin
                } else {
                    smin
                };
                let hi = if sval_cmp(smax.clone(), dmax.clone()) == std::cmp::Ordering::Greater {
                    dmax
                } else {
                    smax
                };
                return Ok((lo, hi));
            }
        }
        let (dmin, dmax) = self.dtype_bounds(dt)?;
        Ok((dmin, dmax))
    }

    pub fn dtype_bounds(&mut self, d: DTypeId) -> Result<(SVal, SVal), String> {
        Ok((
            self.constval_to_sval(&dtype::dtype_min(d), "dtype bounds")?,
            self.constval_to_sval(&dtype::dtype_max(d), "dtype bounds")?,
        ))
    }

    /// ConstVal -> SVal (Big values saturate; documented divergence from exact
    /// Python bignums — the M2 corpus keeps values in range).
    pub fn constval_to_sval(&self, v: &ConstVal, _ctx: &str) -> Result<SVal, String> {
        Ok(match v {
            ConstVal::Bool(b) => SVal::I(if *b { 1 } else { 0 }),
            ConstVal::Int(i) => SVal::I(*i),
            ConstVal::Big(b) => match b.as_i64() {
                Some(i) => SVal::I(i as i128),
                None => {
                    if b.is_negative() {
                        SVal::I(i128::MIN / 2)
                    } else {
                        SVal::I(i128::MAX / 2)
                    }
                }
            },
            ConstVal::Float(f) => {
                // shapes/vmin are integral; infinities clamp into the sentinel band
                let v = *f as i128;
                SVal::I(v.clamp(-MM_SENTINEL, MM_SENTINEL))
            }
            ConstVal::Invalid => SVal::I(0),
        })
    }

    fn trunc_sval_to_dtype(&mut self, s: SVal, _d: DTypeId) -> Result<SVal, String> {
        // float dtype truncation of an int-valued sval: to the dtype's grid
        match s {
            SVal::I(_) => Ok(s.trunc_toward_zero()),
            SVal::U(_) => Ok(s), // TODO: complete when the rewrite engine lands
        }
    }

    // ---------- placeholders completed when the rewrite engine lands ----------

    /// `ssimplify` → SVal. Placeholder: folds only CONST / CAST(CONST).
    pub fn ssimplify(&mut self, u: UopId) -> Result<SVal, String> {
        let (op, dt, srcs, _, _) = self.snapshot(&u);
        if op == Ops::Const {
            return self.constval_to_sval(&self.val(&u), "ssimplify");
        }
        if op == Ops::Cast && self.op_of(&srcs[0]) == Ops::Const {
            let c = self.val(&srcs[0]);
            let v = dtype::dtype_const(dt, c);
            return self.constval_to_sval(&v, "ssimplify");
        }
        Ok(SVal::U(u))
    }

    fn ssimplify_shrink_last(
        &mut self,
        u: UopId,
        in_sz: usize,
        out_sz: usize,
    ) -> Result<SVal, String> {
        // (last * in_sz) // out_sz — placeholder keeps the uop symbolic
        let _ = (in_sz, out_sz);
        Ok(SVal::U(u))
    }

    /// `resolve`-style comparison: True iff the relation certainly holds.
    /// Placeholder: concrete ints only; symbolic → lenient true (completed in
    /// the rewrite-engine step).
    /// `resolve(x, default)` (ops.py:54) for a comparison `a ORD b`: fold it to a
    /// bool via the same UOps Python builds (`ge` = NOT(lt)); `default` only when
    /// the folded result is not provably constant (vmin != vmax).
    pub fn resolve_ord(&mut self, a: SVal, ord: CmpOrd, b: SVal, default: bool) -> bool {
        match (a.clone(), b.clone()) {
            (SVal::I(x), SVal::I(y)) => match ord {
                CmpOrd::Lt => x < y,
                CmpOrd::Le => x <= y,
                CmpOrd::Eq => x == y,
                CmpOrd::Ne => x != y,
                CmpOrd::Ge => x >= y,
                CmpOrd::Gt => x > y,
            },
            _ => {
                let (ua, ub) = (self.sval_to_uop(a), self.sval_to_uop(b));
                match ord {
                    CmpOrd::Lt => {
                        let u = self.alu2(Ops::Cmplt, ua, ub);
                        resolve_uop_bool(self, u, default)
                    }
                    CmpOrd::Gt => {
                        let u = self.alu2(Ops::Cmplt, ub, ua);
                        resolve_uop_bool(self, u, default)
                    }
                    CmpOrd::Eq => {
                        let u = self.alu2(Ops::Cmpeq, ua, ub);
                        resolve_uop_bool(self, u, default)
                    }
                    CmpOrd::Ne => {
                        let u = self.alu2(Ops::Cmpeq, ua, ub);
                        !resolve_uop_bool(self, u, !default)
                    }
                    CmpOrd::Le => {
                        let u = self.alu2(Ops::Cmplt, ub, ua);
                        !resolve_uop_bool(self, u, !default)
                    }
                    CmpOrd::Ge => {
                        let u = self.alu2(Ops::Cmplt, ua, ub);
                        !resolve_uop_bool(self, u, !default)
                    }
                }
            }
        }
    }

    pub fn resolve_cmp(
        &mut self,
        a: SVal,
        ord: std::cmp::Ordering,
        b: SVal,
    ) -> Result<bool, String> {
        match (a, b) {
            (SVal::I(x), SVal::I(y)) => Ok(x.cmp(&y) == ord),
            _ => Ok(true),
        }
    }

    /// GETTUPLE-through-FUNCTION shape substitution placeholder.
    fn substitute_one(&mut self, u: UopId, args: &[UopId]) -> Result<UopId, String> {
        let _ = args;
        Ok(u)
    }

    pub fn toposort(&self, root: &UopId) -> Vec<UopId> {
        Graph::toposort_root(&self.g, root)
    }

    /// `backward_slice` — toposort without self (set semantics used by rules).
    pub fn backward_slice(&self, root: &UopId) -> HashSet<UopId> {
        let ts = self.toposort(root);
        ts.into_iter().filter(|x| x != root).collect()
    }

    pub fn backward_slice_with_self(&self, root: &UopId) -> HashSet<UopId> {
        self.toposort(root).into_iter().collect()
    }

    pub fn op_in_backward_slice_with_self(&self, root: &UopId, ops: &[Ops]) -> bool {
        self.toposort(root)
            .iter()
            .any(|x| ops.contains(&self.op_of(x)))
    }
}

// `dtype.const` for CONST args: canonicalize to the dtype's const value space.
fn dtype_const_canon(d: DTypeId, v: ConstVal) -> ConstVal {
    // Weak dtypes keep exact values; concrete dtypes canonicalize through
    // dtype.const (floats truncate to width; ints stay exact).
    dtype::dtype_const(d, v)
}

struct DebugOp(Ops);
impl std::fmt::Display for DebugOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

pub fn arg_as_int(a: &Arg) -> Option<i64> {
    match a {
        Arg::Int(i) => Some(*i),
        Arg::UInt(u) => i64::try_from(*u).ok(),
        _ => None,
    }
}

// ---- free helpers ----

fn resolve_uop_bool(ctx: &mut UOpCtx, u: UopId, default: bool) -> bool {
    let s = crate::uop::upat::simplify(ctx, u);
    match ctx.min_max(&s) {
        Ok((vmin, vmax)) if vmin == vmax => vmin != SVal::I(0),
        _ => default,
    }
}

pub fn sval_min(a: SVal, b: SVal) -> SVal {
    if sval_cmp(a.clone(), b.clone()) == std::cmp::Ordering::Less {
        a
    } else {
        b
    }
}

pub fn sval_max(a: SVal, b: SVal) -> SVal {
    if sval_cmp(a.clone(), b.clone()) == std::cmp::Ordering::Greater {
        a
    } else {
        b
    }
}

pub fn sval_cmp(a: SVal, b: SVal) -> std::cmp::Ordering {
    match (a, b) {
        (SVal::I(x), SVal::I(y)) => x.cmp(&y),
        (SVal::U(x), SVal::U(y)) => x.cmp(&y),
        _ => std::cmp::Ordering::Equal, // incomparable; treated as equal (never in M2 corpus)
    }
}

impl SVal {
    pub fn trunc_toward_zero(self) -> SVal {
        match self {
            SVal::I(v) => SVal::I(v),
            u => u,
        }
    }
}

// ---------- Step 4: algebraic sugar (elementwise mixin, UOp level) ----------

/// Comparison operator set (Python `<= < == != >= >`); std `Ordering` has only
/// 3 variants and extra match arms would silently bind as variables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmpOrd {
    Lt,
    Le,
    Eq,
    Ne,
    Ge,
    Gt,
}
impl UOpCtx {
    /// `x.alu(op, *src)` — raw construction, dtype derived.
    pub fn alu(&mut self, op: Ops, srcs: Vec<UopId>) -> UopId {
        self.uop(op, None, srcs, Arg::None)
    }

    pub fn alu1(&mut self, op: Ops, a: UopId) -> UopId {
        self.alu(op, vec![a])
    }

    pub fn alu2(&mut self, op: Ops, a: UopId, b: UopId) -> UopId {
        self.alu(op, vec![a, b])
    }

    pub fn alu3(&mut self, op: Ops, a: UopId, b: UopId, c: UopId) -> UopId {
        self.alu(op, vec![a, b, c])
    }

    /// elementwise `_broadcasted` promotion (mixin/elementwise.py:21-30).
    pub fn broadcasted(&mut self, x: UopId, y: UopId) -> (UopId, UopId) {
        let out = least_upper_dtype(&[self.dt_of(&x), self.dt_of(&y)]);
        (self.promote(x, out), self.promote(y, out))
    }

    fn promote(&mut self, t: UopId, out: DTypeId) -> UopId {
        let base = self.base(&t);
        if self.is_invalid(&base) {
            return t;
        }
        if is_weak_dt(self.dt_of(&t)) && self.op_of(&base) == Ops::Const {
            let v = self.val(&base);
            return self.const_py(v, Some(weak_dtype(out)));
        }
        self.cast(t, out)
    }

    /// `UOp.cast` (mixin/dtype.py:16-33): identity when dtypes match.
    pub fn cast(&mut self, t: UopId, dt: DTypeId) -> UopId {
        if self.dt_of(&t) == dt {
            return t;
        }
        self.uop(Ops::Cast, None, vec![t], Arg::Dtype(dt))
    }

    pub fn bitcast(&mut self, t: UopId, dt: DTypeId) -> UopId {
        assert!(
            !is_weak_dt(self.dt_of(&t)) && !is_weak_dt(dt),
            "bitcast requires concrete dtypes"
        );
        if self.dt_of(&t) == dt {
            return t;
        }
        self.uop(Ops::Bitcast, None, vec![t], Arg::Dtype(dt))
    }

    /// elementwise `_binop`: broadcasted promotion then alu.
    pub fn binop(&mut self, op: Ops, a: UopId, b: UopId) -> UopId {
        let (x, y) = self.broadcasted(a, b);
        self.alu2(op, x, y)
    }

    pub fn add(&mut self, a: UopId, b: UopId) -> UopId {
        self.binop(Ops::Add, a, b)
    }

    pub fn sub(&mut self, a: UopId, b: UopId) -> UopId {
        // a + (-b): _broadcasted already promoted these
        let (x, y) = self.broadcasted(a, b);
        let neg_y = self.neg(y);
        self.alu2(Ops::Add, x, neg_y)
    }

    pub fn mul(&mut self, a: UopId, b: UopId) -> UopId {
        self.binop(Ops::Mul, a, b)
    }

    pub fn neg(&mut self, a: UopId) -> UopId {
        if self.dt_of(&a) == crate::dtype::BOOL {
            return self.logical_not(a);
        }
        {
            let c = self.const_py(ConstVal::Int(-1), None);
            self.mul(a, c)
        }
    }

    pub fn logical_not(&mut self, a: UopId) -> UopId {
        let b = self.cast(a, crate::dtype::BOOL);
        let t = self.const_py(ConstVal::Bool(true), Some(crate::dtype::BOOL));
        self.binop(Ops::Cmpne, b, t)
    }

    /// `x % y` (mixin `mod`): int/int -> FLOORMOD, else floor-div path.
    pub fn mod_(&mut self, a: UopId, b: UopId) -> UopId {
        let (x, y) = self.broadcasted(a, b);
        if is_int(self.dt_of(&x)) && is_int(self.dt_of(&y)) {
            return self.alu2(Ops::Floormod, x, y);
        }
        let d = self.div_floor(x.clone(), y.clone());
        {
            let m = self.mul(d, y);
            self.sub(x, m)
        }
    }

    /// `x / y` (mixin `div`): int/int -> CDIV/FLOORDIV, else reciprocal mul.
    pub fn div_floor(&mut self, a: UopId, b: UopId) -> UopId {
        self.div_mode(a, b, true)
    }

    pub fn div_trunc(&mut self, a: UopId, b: UopId) -> UopId {
        self.div_mode(a, b, false)
    }

    fn div_mode(&mut self, a: UopId, b: UopId, floor: bool) -> UopId {
        let (x, y) = self.broadcasted(a, b);
        if is_int(self.dt_of(&x)) && is_int(self.dt_of(&y)) {
            return self.alu2(if floor { Ops::Floordiv } else { Ops::Cdiv }, x, y);
        }
        let x = if is_int(self.dt_of(&x)) || self.dt_of(&x) == crate::dtype::BOOL {
            self.cast(x, crate::dtype::default_float())
        } else {
            x
        };
        let r = self.alu1(Ops::Reciprocal, y);
        self.alu2(Ops::Mul, x, r)
    }

    pub fn lt(&mut self, a: UopId, b: UopId) -> UopId {
        self.binop(Ops::Cmplt, a, b)
    }

    pub fn gt(&mut self, a: UopId, b: UopId) -> UopId {
        self.binop(Ops::Cmplt, b, a)
    }

    pub fn ne(&mut self, a: UopId, b: UopId) -> UopId {
        self.binop(Ops::Cmpne, a, b)
    }

    pub fn eq(&mut self, a: UopId, b: UopId) -> UopId {
        {
            let n = self.ne(a, b);
            self.logical_not(n)
        }
    }

    pub fn and_(&mut self, a: UopId, b: UopId) -> UopId {
        self.binop(Ops::And, a, b)
    }

    pub fn or_(&mut self, a: UopId, b: UopId) -> UopId {
        self.binop(Ops::Or, a, b)
    }

    pub fn xor(&mut self, a: UopId, b: UopId) -> UopId {
        self.binop(Ops::Xor, a, b)
    }

    pub fn shl(&mut self, a: UopId, b: UopId) -> UopId {
        self.binop(Ops::Shl, a, b)
    }

    pub fn shr(&mut self, a: UopId, b: UopId) -> UopId {
        self.binop(Ops::Shr, a, b)
    }

    pub fn maximum(&mut self, a: UopId, b: UopId) -> UopId {
        self.binop(Ops::Max, a, b)
    }

    pub fn minimum(&mut self, a: UopId, b: UopId) -> UopId {
        let (t, x) = self.broadcasted(a, b);
        let dt = least_upper_dtype(&[self.dt_of(&t), self.dt_of(&x)]);
        if is_float(dt) {
            let nt = self.neg(t);
            let nx = self.neg(x);
            let m = self.alu2(Ops::Max, nt, nx);
            return self.neg(m);
        }
        // python: k = dt.const(dt.min + dt.max) — for every two's-complement
        // int dtype (weakint included) min+max == -1
        let k = self.const_py(ConstVal::Int(-1), Some(dt));
        let a1 = self.alu2(Ops::Xor, t, k.clone());
        let b1 = self.alu2(Ops::Xor, x, k.clone());
        let m = self.alu2(Ops::Max, a1, b1);
        self.alu2(Ops::Xor, m, k)
    }

    /// `cond.where(x, y)` (mixin where: both UOp → broadcasted promotion).
    pub fn where_(&mut self, cond: UopId, x: UopId, y: UopId) -> UopId {
        let (x, y) = self.broadcasted(x, y);
        self.alu3(Ops::Where, cond, x, y)
    }

    pub fn reciprocal(&mut self, a: UopId) -> UopId {
        self.alu1(Ops::Reciprocal, a)
    }

    pub fn sqrt(&mut self, a: UopId) -> UopId {
        self.alu1(Ops::Sqrt, a)
    }

    pub fn exp2(&mut self, a: UopId) -> UopId {
        self.alu1(Ops::Exp2, a)
    }

    pub fn pow(&mut self, a: UopId, b: UopId) -> UopId {
        let (x, y) = self.broadcasted(a, b);
        self.alu2(Ops::Pow, x, y)
    }

    pub fn trunc(&mut self, a: UopId) -> UopId {
        self.alu1(Ops::Trunc, a)
    }

    /// `usum(*uops)` — reduce add (or) over [self] + others.
    pub fn usum(&mut self, first: UopId, rest: &[UopId]) -> UopId {
        let is_bool = self.dt_of(&first) == crate::dtype::BOOL;
        let mut acc = first;
        for n in rest {
            acc = if is_bool {
                self.or_(acc, n.clone())
            } else {
                self.add(acc, n.clone())
            };
        }
        acc
    }

    /// `uprod(*uops)` — reduce mul (and) over [self] + others.
    pub fn uprod(&mut self, first: UopId, rest: &[UopId]) -> UopId {
        let is_bool = self.dt_of(&first) == crate::dtype::BOOL;
        let mut acc = first;
        for n in rest {
            acc = if is_bool {
                self.and_(acc, n.clone())
            } else {
                self.mul(acc, n.clone())
            };
        }
        acc
    }

    /// `identity_element(op, dt)` (ops.py:51).
    pub fn identity_element(&mut self, op: Ops, dt: DTypeId) -> UopId {
        let v = match op {
            Ops::Add => ConstVal::Int(0),
            Ops::Mul => ConstVal::Int(1),
            Ops::Max => dtype::dtype_min(dt),
            _ => panic!("TGERR: identity_element for {op:?}"),
        };
        self.const_py(dtype::dtype_const(dt, v), Some(dt))
    }

    /// `x.const_like(b)` — CONST (or STACK of CONSTs) with x's dtype,
    /// EXPANDed to x's shape when it differs (ops.py:581-583).
    pub fn const_like(&mut self, x: UopId, v: ConstVal) -> UopId {
        self.const_like_dt(x.clone(), v, Some(self.dt_of(&x)))
    }

    pub fn const_like_dt(&mut self, x: UopId, v: ConstVal, dtype: Option<DTypeId>) -> UopId {
        let ret = self.const_py(v, dtype);
        // expand to x's shape when it differs
        let xshape = match self.shape(&x) {
            Ok(s) => s,
            Err(_) => return ret,
        };
        let rshape = match self.shape(&ret) {
            Ok(s) => s,
            Err(_) => return ret,
        };
        match (xshape, rshape) {
            (Some(xs), Some(rs)) if !xs.is_empty() && xs != rs => {
                // python const_like: ret._mop(EXPAND, arg=self._shape) — shape as src
                let shape_src = self.shape_arg(&xs);
                let s0 = crate::tensor::sink(self, vec![shape_src]);
                let sid = crate::uop::upat::simplify(self, s0);
                let mut srcs = vec![ret];
                srcs.extend(self.srcs(&sid).to_vec());
                self.uop(Ops::Expand, None, srcs, Arg::None)
            }
            _ => ret,
        }
    }

    /// SVal -> UopId (const wrap for ints), the `ufix` of shape math.
    pub fn sval_to_uop(&mut self, s: SVal) -> UopId {
        match s {
            SVal::I(v) => self.cint_dt(v as i64, crate::dtype::default_int()),
            SVal::U(u) => u,
        }
    }

    /// Python sint binop: int-int folds, UOp sides keep receiver order with the
    /// int side wrapped as a CONST in place (`3 + u` -> ADD(CONST(3), u)).
    pub fn sval_binop(&mut self, op: Ops, a: SVal, b: SVal) -> SVal {
        if let (SVal::I(x), SVal::I(y)) = (&a, &b) {
            let (x, y) = (*x, *y);
            return match op {
                Ops::Add => SVal::I(x.wrapping_add(y)),
                Ops::Sub => SVal::I(x.wrapping_sub(y)),
                Ops::Mul => SVal::I(x.wrapping_mul(y)),
                Ops::Floordiv => SVal::I(crate::uop::alu::floordiv(x, y)),
                Ops::Cdiv => SVal::I(crate::uop::alu::cdiv(x, y)),
                Ops::Cmod => SVal::I(crate::uop::alu::cmod(x, y)),
                Ops::Floormod => SVal::I(crate::uop::alu::floormod(x, y)),
                _ => panic!("TGERR: sval_binop unsupported op {op:?}"),
            };
        }
        let (ua, ub) = (self.sval_to_uop(a), self.sval_to_uop(b));
        SVal::U(self.alu2(op, ua, ub))
    }

    pub fn sval_add(&mut self, a: SVal, b: SVal) -> SVal {
        self.sval_binop(Ops::Add, a, b)
    }

    pub fn sval_sub(&mut self, a: SVal, b: SVal) -> SVal {
        self.sval_binop(Ops::Sub, a, b)
    }

    pub fn sval_mul(&mut self, a: SVal, b: SVal) -> SVal {
        self.sval_binop(Ops::Mul, a, b)
    }

    pub fn sval_div_floor(&mut self, a: SVal, b: SVal) -> SVal {
        self.sval_binop(Ops::Floordiv, a, b)
    }

    pub fn sval_neg(&mut self, a: SVal) -> SVal {
        match a {
            SVal::I(v) => SVal::I(-v),
            SVal::U(u) => SVal::U(self.alu1(Ops::Neg, u)),
        }
    }

    pub fn sval_is_one(&mut self, s: &SVal) -> Option<bool> {
        match s {
            SVal::I(v) => Some(*v == 1),
            SVal::U(_) => None,
        }
    }

    pub fn sval_to_arg(&mut self, s: &SVal) -> Arg {
        match s {
            SVal::I(v) => arg_from_i128(*v),
            SVal::U(u) => {
                // shape uops are carried by index in the STACK arg of movement ops —
                // but Python stores them as UOp objects in the arg, which we cannot
                // represent. These only appear for symbolic movement args; panic
                // until M3 needs them through args (they travel as srcs instead).
                let _ = u;
                panic!("TGERR: symbolic shape element in arg")
            }
        }
    }

    // ---------- symbolic helpers (ops.py:1005-1045) ----------

    /// `x.split_uop(sep)` — flatten the tree on `sep`.
    pub fn split_uop(&self, root: UopId, sep: Ops, out: &mut Vec<UopId>) {
        if self.op_of(&root) == sep {
            for s in self.srcs(&root) {
                self.split_uop(s.clone(), sep, out);
            }
        } else {
            out.push(root);
        }
    }

    pub fn split_uop_vec(&self, root: UopId, sep: Ops) -> Vec<UopId> {
        let mut v = Vec::new();
        self.split_uop(root, sep, &mut v);
        v
    }

    /// `x.pop_const(op)` (ops.py:1029-1030).
    pub fn pop_const(&mut self, x: UopId, op: Ops) -> (UopId, ConstVal) {
        let (xo, _, xsrcs, _, _) = self.snapshot(&x);
        if xo == op && xsrcs.len() >= 2 && self.op_of(&xsrcs[1]) == Ops::Const {
            ((xsrcs[0]).clone(), self.val(&xsrcs[1]))
        } else {
            {
                let ie = self.identity_element(op, self.dt_of(&x));
                let v = self.val(&ie);
                (x, v)
            }
        }
    }

    /// `x.const_factor()` (ops.py:1007-1015).
    pub fn const_factor(&mut self, x: UopId) -> i128 {
        let (op, _, srcs, _arg, _) = self.snapshot(&x);
        match op {
            Ops::Const => self.val(&x).as_int_exact().unwrap_or(1),
            Ops::Stack => {
                let mut g: i128 = 0;
                for s in &srcs {
                    g = gcd_i128(g, self.const_factor(s.clone()));
                }
                g
            }
            Ops::Add => gcd_i128(
                self.const_factor((srcs[0]).clone()),
                self.const_factor((srcs[1]).clone()),
            ),
            Ops::Mul => {
                if self.op_of(&srcs[0]) == Ops::Const {
                    self.val(&srcs[0]).as_int_exact().unwrap_or(1)
                } else if self.op_of(&srcs[1]) == Ops::Const {
                    self.val(&srcs[1]).as_int_exact().unwrap_or(1)
                } else {
                    1
                }
            }
            Ops::Param | Ops::Buffer => {
                if let Some(p) = self.has_param(&x) {
                    p.multiple_of.unwrap_or(0).max(0) as i128
                } else {
                    1
                }
            }
            _ => 1,
        }
    }

    /// `x.divides(v)` (ops.py:1016-1028) — `None` when unsure.
    pub fn divides(&mut self, x: UopId, v: i128) -> Option<UopId> {
        if v == 1 {
            return Some(x);
        }
        let (op, _, srcs, _, _) = self.snapshot(&x);
        match op {
            Ops::Const => {
                let val = self.val(&x);
                let iv = val.as_int_exact()?;
                if iv % v == 0 {
                    Some(self.const_like(x, ConstVal::Int(iv / v)))
                } else {
                    None
                }
            }
            Ops::Stack => {
                let mut srcs2 = Vec::with_capacity(srcs.len());
                for s in &srcs {
                    srcs2.push(self.divides(s.clone(), v)?);
                }
                Some(self.uop(Ops::Stack, None, srcs2, Arg::None))
            }
            Ops::Add => {
                let d0 = self.divides((srcs[0]).clone(), v)?;
                let d1 = self.divides((srcs[1]).clone(), v)?;
                Some(self.add(d0, d1))
            }
            Ops::Mul => {
                if let Some(d0) = self.divides((srcs[0]).clone(), v) {
                    return Some(self.mul(d0, (srcs[1]).clone()));
                }
                if let Some(d1) = self.divides((srcs[1]).clone(), v) {
                    return Some(self.mul((srcs[0]).clone(), d1));
                }
                None
            }
            Ops::Param | Ops::Buffer => {
                let p = self.has_param(&x)?;
                let m = p.multiple_of? as i128;
                if m % v == 0 {
                    {
                        let d = self.cint(v as i64);
                        Some(self.div_trunc(x, d))
                    }
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// `UOp.gcd(*uops)` (ops.py:1031-1035).
    pub fn uop_gcd(&mut self, uops: &[UopId]) -> UopId {
        let mut terms = Vec::with_capacity(uops.len());
        let mut factors = Vec::with_capacity(uops.len());
        for u in uops {
            let f = self.const_factor(u.clone());
            factors.push(f);
            terms.push((self.divides(u.clone(), f), f));
        }
        // Counter intersection over MUL factors, first term's key order
        let mut count: Vec<(UopId, usize)> = Vec::new();
        for t in self.split_uop_vec(terms[0].0.clone().unwrap(), Ops::Mul) {
            bump_count(&mut count, t);
        }
        for (term, _) in terms.iter().skip(1) {
            let tcount = {
                let mut tc: Vec<(UopId, usize)> = Vec::new();
                for t in self.split_uop_vec(term.clone().unwrap(), Ops::Mul) {
                    bump_count(&mut tc, t);
                }
                tc
            };
            for (k, c) in count.iter_mut() {
                let other = tcount
                    .iter()
                    .find(|(k2, _)| k2 == k)
                    .map(|(_, c2)| *c2)
                    .unwrap_or(0);
                *c = (*c).min(other);
            }
        }
        let elements: Vec<UopId> = count
            .iter()
            .filter(|(_, c)| *c > 0)
            .flat_map(|(k, c)| vec![k.clone(); *c])
            .collect();
        let g = gcd_i128_all(&factors);
        let const_node = self.const_like(uops[0].clone(), ConstVal::Int(g));
        // math.prod([*elements, const]) — default start=1: 1*e1 → MUL(CONST(1), e1);
        // callers simplify immediately, so the plain fold converges to the same graph
        let mut all = elements;
        all.push(const_node);
        let first = all.remove(0);
        self.uprod(first, &all)
    }

    /// `x.divide_exact(v)` (ops.py:1036-1045).
    pub fn divide_exact(&mut self, x: UopId, v: UopId) -> Option<UopId> {
        if x == v {
            return Some(self.const_like(x, ConstVal::Int(1)));
        }
        if self.op_of(&v) == Ops::Const {
            let iv = self.val(&v).as_int_exact()?;
            return self.divides(x, iv);
        }
        if self.op_of(&x) == Ops::Add {
            let s0 = self.divide_exact(self.srcs(&x)[0].clone(), v.clone())?;
            let s1 = self.divide_exact(self.srcs(&x)[1].clone(), v)?;
            return Some(self.add(s0, s1));
        }
        if self.op_of(&x) == Ops::Mul {
            let (fac, c) = self.pop_const(x.clone(), Ops::Mul);
            let (div_fac, div_c) = self.pop_const(v, Ops::Mul);
            // Counter subtraction
            let mut count: Vec<(UopId, i64)> = Vec::new();
            for t in self.split_uop_vec(fac, Ops::Mul) {
                bump_count(&mut count, t);
            }
            // Python Counter.subtract creates negative entries for missing keys
            for t in self.split_uop_vec(div_fac, Ops::Mul) {
                match count.iter_mut().find(|(k, _)| *k == t) {
                    Some((_, c)) => *c -= 1,
                    None => count.push((t, -1)),
                }
            }
            let cv = c.as_int_exact()?;
            let dv = div_c.as_int_exact()?;
            if cv % dv == 0 && count.iter().all(|(_, c)| *c >= 0) {
                let elements: Vec<UopId> = count
                    .iter()
                    .filter(|(_, c)| *c > 0)
                    .flat_map(|(k, c)| vec![k.clone(); *c as usize])
                    .collect();
                // math.prod(elements, start=const_like(const//div_const)) — empty elements → start alone
                let start = self.const_like(x, ConstVal::Int(cv / dv));
                if elements.is_empty() {
                    return Some(start);
                }
                let first = elements[0].clone();
                let mut acc = self.mul(start, first);
                for e in &elements[1..] {
                    acc = self.mul(acc, e.clone());
                }
                return Some(acc);
            }
        }
        None
    }

    /// `x.tuplize()` — an Ord key mirroring Python's tuple for commutative flips.
    pub fn tuplize(&self, id: &UopId) -> TupKey {
        TupKey {
            op: id.op as u16,
            arg: arg_key(&id.arg),
            dtype: DtKey::of(id.dtype),
            srcs: id.srcs.iter().map(|s| self.tuplize(s)).collect(),
        }
    }
}

/// Serializable ordering key for `tuplize` comparisons, mirroring Python's
/// tuple semantics: `Ops` values, then `arg` (Python `Any` compared with
/// `<`: ParamArg is `order=True`, DType is `(priority,bitsize,name,fmt)`,
/// floats compare numerically, bools are ints), then dtype, then srcs.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct TupKey {
    op: u16,
    arg: ArgKey,
    dtype: DtKey,
    srcs: Vec<TupKey>,
}

/// Numeric key consistent with Python's cross-type numeric `<` over
/// PyConsts. Floats keep their f64; comparisons against ints go through f64
/// (exact for corpus magnitudes; documented divergence beyond 2^53).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NumKey {
    I(i128),
    F(f64),
}

impl Eq for NumKey {}

fn cmp_i_f64(i: i128, f: f64) -> std::cmp::Ordering {
    if f.is_nan() {
        return std::cmp::Ordering::Greater; // NaN sorts above everything, both ways (Python: unordered)
    }
    let fi = i as f64;
    if fi == f {
        return std::cmp::Ordering::Equal;
    }
    fi.partial_cmp(&f).unwrap_or(std::cmp::Ordering::Equal)
}

impl PartialOrd for NumKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for NumKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (NumKey::I(a), NumKey::I(b)) => a.cmp(b),
            (NumKey::I(a), NumKey::F(b)) => cmp_i_f64(*a, *b),
            (NumKey::F(a), NumKey::I(b)) => cmp_i_f64(*b, *a).reverse(),
            (NumKey::F(a), NumKey::F(b)) => {
                if a.is_nan() && b.is_nan() {
                    std::cmp::Ordering::Equal
                } else if a.is_nan() {
                    std::cmp::Ordering::Greater
                } else if b.is_nan() {
                    std::cmp::Ordering::Greater
                } else {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                }
            }
        }
    }
}

fn float_numkey(f: f64) -> NumKey {
    NumKey::F(f)
}

fn pyconst_numkey(v: &ConstVal) -> NumKey {
    match v {
        ConstVal::Bool(b) => NumKey::I(*b as i128),
        ConstVal::Int(i) => NumKey::I(*i),
        ConstVal::Big(b) => {
            NumKey::I(b.as_i64().map(|x| x as i128).unwrap_or(if b.is_negative() {
                i128::MIN / 4
            } else {
                i128::MAX / 4
            }))
        }
        ConstVal::Float(f) => float_numkey(*f),
        ConstVal::Invalid => NumKey::I(0),
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ArgKey {
    None,
    Bool(bool),
    Int(i128),
    Float(NumKey),
    Str(String),
    Bytes(Vec<u8>),
    Enum { class: u8, value: u16 },
    Dtype(DtKey),
    Seq { tuple: bool, items: Vec<ArgKey> },
    Param(ParamKey),
    Pickle(Vec<u8>),
    Data(String, Vec<(String, ArgKey)>),
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DtKey {
    priority: i32,
    bitsize: u32,
    name: &'static str,
    fmt: i32, // -1 = None
}

impl DtKey {
    pub fn of(d: DTypeId) -> DtKey {
        let dt = d.dt();
        DtKey {
            priority: dt.priority,
            bitsize: dt.bitsize,
            name: dt.name,
            fmt: match dt.fmt {
                None => -1,
                Some(c) => c as i32,
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ParamKey {
    slot: i64,
    dtype: DtKey,
    vmin: Option<NumKey>,
    vmax: Option<NumKey>,
    multiple_of: Option<i64>,
    name: Option<String>,
    addrspace: Option<u8>,
    axis: Option<i64>,
    device: Option<DeviceKey>,
    volatile: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum DeviceKey {
    Single(String),
    Multi(Vec<String>),
}

fn arg_key(a: &Arg) -> ArgKey {
    match a {
        Arg::None => ArgKey::None,
        Arg::Bool(b) => ArgKey::Bool(*b),
        Arg::Int(i) => ArgKey::Int(*i as i128),
        Arg::UInt(u) => ArgKey::Int(*u as i128),
        Arg::Float(f) => ArgKey::Float(float_numkey(*f)),
        Arg::NonFinite(n) => {
            let f = match n {
                super::arg::NonFinite::Nan => f64::NAN,
                super::arg::NonFinite::PosInf => f64::INFINITY,
                super::arg::NonFinite::NegInf => f64::NEG_INFINITY,
            };
            ArgKey::Float(float_numkey(f))
        }
        Arg::Str(s) => ArgKey::Str(s.clone()),
        Arg::Bytes(b) => ArgKey::Bytes(b.clone()),
        Arg::Enum(e) => ArgKey::Enum {
            class: e.class as u8,
            value: e.value,
        },
        Arg::Dtype(d) => ArgKey::Dtype(DtKey::of(*d)),
        Arg::Seq { tuple, items } => ArgKey::Seq {
            tuple: *tuple,
            items: items.iter().map(arg_key).collect(),
        },
        Arg::Param(p) => ArgKey::Param(ParamKey {
            slot: p.slot,
            dtype: DtKey::of(p.dtype),
            vmin: p.vmin_vmax.as_ref().map(|(a, _)| pyconst_numkey(a)),
            vmax: p.vmin_vmax.as_ref().map(|(_, b)| pyconst_numkey(b)),
            multiple_of: p.multiple_of,
            name: p.name.clone(),
            addrspace: p.addrspace.map(|a| a as u8),
            axis: p.axis,
            device: p.device.as_ref().map(|d| match d {
                super::arg::DeviceSpec::Single(s) => DeviceKey::Single(s.clone()),
                super::arg::DeviceSpec::Multi(v) => DeviceKey::Multi(v.clone()),
            }),
            volatile: p.volatile,
        }),
        Arg::Pickle(b) => ArgKey::Pickle(b.clone()),
        Arg::Data(d) => ArgKey::Data(
            d.tag.clone(),
            d.fields
                .iter()
                .map(|(n, a)| (n.clone(), arg_key(a)))
                .collect(),
        ),
        Arg::Const(c) => match c {
            ConstVal::Bool(b) => ArgKey::Bool(*b),
            ConstVal::Int(i) => ArgKey::Int(*i),
            ConstVal::Big(_) => ArgKey::Int(pyconst_numkey(c).as_i128_hint()),
            ConstVal::Float(f) => ArgKey::Float(float_numkey(*f)),
            ConstVal::Invalid => ArgKey::None,
        },
    }
}

impl NumKey {
    fn as_i128_hint(self) -> i128 {
        match self {
            NumKey::I(i) => i,
            NumKey::F(f) => f as i128,
        }
    }
}

fn bump_count<N>(count: &mut Vec<(UopId, N)>, t: UopId)
where
    N: std::ops::AddAssign + From<u8>,
{
    match count.iter_mut().find(|(k, _)| *k == t) {
        Some((_, c)) => *c += 1.into(),
        None => count.push((t, 1.into())),
    }
}

fn is_weak_dt(d: DTypeId) -> bool {
    d == WEAKINT || d == WEAKFLOAT
}

fn gcd_i128(a: i128, b: i128) -> i128 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

fn gcd_i128_all(vals: &[i128]) -> i128 {
    vals.iter().cloned().fold(0, gcd_i128)
}

/// i128 → Arg for shape arg lists (Big values not representable in args).
pub fn arg_from_i128(v: i128) -> Arg {
    if v >= i64::MIN as i128 && v <= i64::MAX as i128 {
        Arg::Int(v as i64)
    } else {
        Arg::Const(ConstVal::Int(v))
    }
}

// ---------- M4 scheduler graph utilities (uop/ops.py) ----------

impl UOpCtx {
    /// `u.toposort(gate=, enter_calls=)` (ops.py:294) — insertion-ordered list
    /// (python dict semantics), gate returning false prunes the subtree.
    pub fn toposort_gated(
        &self,
        root: UopId,
        gate: Option<&dyn Fn(UopId) -> bool>,
        enter_calls: bool,
    ) -> Vec<UopId> {
        enum Step {
            Enter(UopId),
            Emit(UopId),
        }
        let mut out: Vec<UopId> = vec![];
        let mut seen: HashSet<u32> = HashSet::new();
        let mut stack = vec![Step::Enter(root)];
        while let Some(s) = stack.pop() {
            match s {
                Step::Enter(id) => {
                    if seen.contains(&id.id()) {
                        continue;
                    }
                    let go = match gate {
                        None => true,
                        Some(g) => g(id.clone()),
                    };
                    if go {
                        stack.push(Step::Emit(id.clone()));
                        let op = id.op;
                        let skip_body = !enter_calls && matches!(op, Ops::Call | Ops::Function);
                        for (i, src) in id.srcs.iter().enumerate().rev() {
                            if skip_body && i == 0 {
                                continue;
                            }
                            stack.push(Step::Enter(src.clone()));
                        }
                    }
                }
                Step::Emit(id) => {
                    if !seen.contains(&id.id()) {
                        seen.insert(id.id());
                        out.push(id);
                    }
                }
            }
        }
        out
    }

    /// ParamArg of a BUFFER/PARAM node, if any.
    pub fn param_arg<'a>(&self, id: &'a UopId) -> Option<&'a ParamArg> {
        match self.arg_of(id) {
            Arg::Param(p) => Some(p),
            _ => None,
        }
    }

    /// `u.device` — ParamArg.device of a BUFFER/PARAM (None otherwise).
    pub fn device_of<'a>(&self, id: &'a UopId) -> Option<&'a DeviceSpec> {
        self.param_arg(id).and_then(|p| p.device.as_ref())
    }

    /// `u.addrspace` — ParamArg.addrspace (None when absent).
    pub fn addrspace_of(&self, id: UopId) -> Option<AddrSpace> {
        self.param_arg(&id).and_then(|p| p.addrspace)
    }

    /// `u.addrspace` derived property (ops.py:867) — python's full logic.
    ///
    /// tinygrad marks this `@recursive_property`: the walk is memoized per node
    /// (via a gated toposort) so it is linear in the DAG. Without the cache the
    /// naive recursion is exponential on shared subgraphs — which bf16 ALU
    /// emulation produces in abundance (an inner value repeated N times per op).
    pub fn addrspace_derived(&self, id: UopId) -> Option<AddrSpace> {
        if let Some(cached) = id.node.addrspace_c.borrow().clone() {
            return cached;
        }
        let ret = self.addrspace_derived_uncached(id.clone());
        *id.node.addrspace_c.borrow_mut() = Some(ret);
        ret
    }

    fn addrspace_derived_uncached(&self, id: UopId) -> Option<AddrSpace> {
        let op = self.op_of(&id);
        if op == Ops::Param || op == Ops::Buffer {
            return self.addrspace_of(id);
        }
        if op == Ops::Special || op == Ops::Range || op == Ops::Load {
            return Some(AddrSpace::Alu);
        }
        if matches!(
            op,
            Ops::Index
                | Ops::Cast
                | Ops::After
                | Ops::Reduce
                | Ops::Store
                | Ops::Mstack
                | Ops::Mselect
                | Ops::End
                | Ops::Unshard
        ) || group::in_(group::MOVEMENT, op)
        {
            let srcs = self.srcs(&id);
            if srcs.is_empty() {
                return None;
            }
            return self.addrspace_derived(srcs[0].clone());
        }
        if op == Ops::Stack
            || op == Ops::Wmma
            || op == Ops::Group
            || group::in_(group::ELEMENTWISE, op)
        {
            let mut ad: Vec<AddrSpace> = Vec::new();
            for s in self.srcs(&id) {
                if let Some(a) = self.addrspace_derived(s.clone()) {
                    ad.push(a);
                }
            }
            if ad.is_empty() || !ad.iter().all(|&a| a == ad[0]) {
                return None;
            }
            return Some(ad[0]);
        }
        None
    }

    /// `u.is_variable` — 0-d ALU BUFFER with a value range (shape arg is the
    /// empty shape, checked structurally without the shape cache).
    pub fn is_variable(&self, id: UopId) -> bool {
        if self.op_of(&id) != Ops::Buffer {
            return false;
        }
        match self.param_arg(&id) {
            Some(p) => p.vmin_vmax.is_some() && p.addrspace == Some(AddrSpace::Alu),
            None => false,
        }
    }

    /// `u.is_bound_var` — AFTER wrapping a bound variable store.
    pub fn is_bound_var(&self, id: UopId) -> bool {
        if self.op_of(&id) != Ops::After {
            return false;
        }
        // AFTER(var_buffer.store(value)): src[0].src[0] is the variable
        let srcs = self.srcs(&id);
        if srcs.is_empty() || self.op_of(&srcs[0]) != Ops::Store {
            return false;
        }
        let st = self.srcs(&srcs[0]);
        !st.is_empty() && self.is_variable(st[0].clone())
    }

    /// `u.buf_uop` (ops.py:881).
    pub fn buf_uop(&mut self, id: UopId) -> UopId {
        let op = self.op_of(&id);
        if op == Ops::Buffer || op == Ops::Param {
            return id;
        }
        if op == Ops::Mselect {
            return id; // per-device selection stays (corpus is single-device)
        }
        if op == Ops::Mstack {
            let srcs = self.srcs(&id).to_vec();
            let bufs: Vec<UopId> = srcs.iter().map(|s| self.buf_uop(s.clone())).collect();
            return self.uop(Ops::Mstack, None, bufs, Arg::None);
        }
        if self.op_of(&self.base(&id)) == Ops::After {
            let b = self.base(&id);
            let bsrc = self.srcs(&b)[0].clone();
            let bu = self.buf_uop(bsrc);
            return self.base(&bu);
        }
        let mut s = id;
        loop {
            let srcs = self.srcs(&s);
            let op = self.op_of(&s);
            if srcs.is_empty() || matches!(op, Ops::Buffer | Ops::Param | Ops::Stage | Ops::Mstack)
            {
                return s;
            }
            s = srcs[0].clone();
        }
    }

    /// `u.unsharded_base` (ops.py:765) — movement/base chain to the storage base.
    pub fn unsharded_base(&mut self, id: UopId) -> UopId {
        let op = self.op_of(&id);
        if group::in_(group::MOVEMENT, op) || op == Ops::Detach || op == Ops::Unshard {
            let s0 = self.srcs(&id)[0].clone();
            return self.base(&s0);
        }
        id
    }

    /// `u.has_buffer_identity(after_ok)` (ops.py:908).
    pub fn has_buffer_identity(&mut self, id: UopId, after_ok: bool) -> bool {
        let op = self.op_of(&id);
        if matches!(op, Ops::Reshape | Ops::Unshard | Ops::Mselect) {
            let s0 = self.srcs(&id)[0].clone();
            return self.has_buffer_identity(s0, after_ok);
        }
        if after_ok && op == Ops::After {
            let s0 = self.srcs(&id)[0].clone();
            return self.has_buffer_identity(s0, after_ok);
        }
        matches!(op, Ops::Buffer | Ops::Param)
    }

    /// `u.max_numel()` — int max of shape product.
    pub fn max_numel(&mut self, id: UopId) -> i64 {
        let shp = self.max_shape(id);
        shp.iter().product::<i64>()
    }

    /// `u.max_shape` — per-axis vmax (concrete ints in the corpus).
    pub fn max_shape(&mut self, id: UopId) -> Vec<i64> {
        let shp = self
            .shape_of(&id)
            .expect("TGERR: max_shape on shapeless node");
        shp.iter()
            .map(|s| match s {
                SVal::I(i) => *i as i64,
                SVal::U(u) => {
                    let (_, vmax) = self.min_max(u).expect("TGERR: unbounded symbolic shape");
                    vmax.as_i64_or_panic()
                }
            })
            .collect()
    }

    /// `contiguous_view_offset` (ops.py:890-906) — flat offset when the
    /// movement chain is contiguous, None otherwise.
    pub fn contiguous_view_offset(&mut self, x: UopId) -> Option<i64> {
        self.contiguous_view(x).map(|(_, off)| off)
    }

    /// `contiguous_view` (ops.py:890) — (tagged base buffer, offset).
    pub fn contiguous_view(&mut self, x: UopId) -> Option<(UopId, i64)> {
        // WEBGPU and CL do not support views (ops.py:890): the CL runtime has no
        // reliable sub-buffer alignment, so contiguous-view collapse is disabled
        // and movement stays as kernels. Without this guard `contiguous_mops_to_view`
        // fails to reach a fixed point on CL and loops.
        if let Some(crate::uop::arg::DeviceSpec::Single(s)) =
            crate::schedule::indexing::device_of(self, x.clone())
        {
            if s.starts_with("WEBGPU") || s.starts_with("CL") {
                return None;
            }
        }
        let numel = self.max_numel(x.clone());
        let flat = crate::tensor::movement::flatten(self, x, -1, -1);
        let r = self.range(SVal::I(numel as i128), 0, crate::uop::range::AxisType::Weak);
        let idx = self.index(flat, vec![r]);
        let pm = crate::uop::sym::pm_mop_cleanup();
        let cvo = crate::uop::ops::pm_contiguous_view_offset();
        let full = pm.add(crate::uop::sym::symbolic()).add(&cvo);
        let out = crate::uop::upat::graph_rewrite(
            self,
            idx,
            &full,
            false,
            None,
            false,
            false,
            super::upat::CtxVal::None,
        )
        .ok()?;
        if self.op_of(&out) != Ops::Index {
            return None;
        }
        let osrcs = self.srcs(&out).to_vec();
        if osrcs.len() != 2 {
            return None;
        }
        // (b:=out.src[0]).tag must be truthy
        let tagged = !matches!(self.tag_of(&osrcs[0]), Arg::None);
        if !tagged {
            return None;
        }
        if self.op_of(&osrcs[1]) != Ops::Const {
            return None;
        }
        match self.val(&osrcs[1]) {
            ConstVal::Int(i) if i >= i64::MIN as i128 && i <= i64::MAX as i128 => {
                // `b.rtag(None)` (ops.py:895): strip the tag, so the returned buffer
                // is structurally stable and `contiguous_mops_to_view` reaches a
                // fixed point (a retained tag would make each rewrite a new uop).
                let b = osrcs[0].clone();
                let (op, dt, srcs, arg, _tag) = self.snapshot(&b);
                let untagged = self.uop_tag(op, Some(dt), srcs, arg, Arg::None);
                Some((untagged, i as i64))
            }
            _ => None,
        }
    }

    /// `ret.sink(arg=...)` — SINK node with an arg.
    pub fn sink_arg(&mut self, srcs: Vec<UopId>, arg: Arg) -> UopId {
        self.uop(Ops::Sink, None, srcs, arg)
    }

    /// `body.call(*srcs)` for OPAQUE bodies (SINK kernel) — `Ops.CALL` with CallInfo.
    pub fn call(&mut self, body: UopId, srcs: Vec<UopId>, name: Option<&str>) -> UopId {
        let arg = Arg::Data(super::arg::call_info(name));
        let mut all = vec![body];
        all.extend(srcs);
        self.uop(Ops::Call, None, all, arg)
    }
}
