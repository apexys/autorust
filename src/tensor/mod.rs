//! M3: the `Tensor` front-end. v0.14's `Tensor` is a thin wrapper over `uop: UOp`
//! (tensor.py:274); all sugar (movement/elementwise/reduce/creation) is UOp-level
//! via the mixin chain and returns UOps. Submodules mirror the mixin files.

mod callify;
pub mod ix;
pub use ix::Ix;
mod shape;
mod shapearg;

pub use shape::Shape;
pub use shapearg::ShapeArg;
mod creation;
mod dtype;
mod elementwise;
pub mod movement;
mod reduce;

use crate::dtype::{AddrSpace, ConstVal};
use crate::uop::arg::{Arg, EnumClass, EnumVal};
use crate::uop::ctx::UOpCtx;
use crate::uop::graph::UopId;
use crate::uop::ops::Ops;
use crate::dtype::DTypeId;
use crate::uop::sint::{SInt, SVal};

/// Python `Tensor.__slots__ = ("uop", "is_param", "grad")`.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub uop: UopId,
    pub grad: Option<Box<Tensor>>,
    pub is_param: bool,
}

impl Tensor {
    pub fn from_uop(uop: UopId) -> Tensor {
        Tensor { uop, grad: None, is_param: true }
    }

    /// `Tensor._apply_uop`: map a UOp-level fxn over `(self, *x).uop` and re-wrap.
    pub fn apply_uop(self, ctx: &mut UOpCtx, srcs: &[Tensor], f: impl FnOnce(&mut UOpCtx, Vec<UopId>) -> UopId) -> Tensor {
        let uops: Vec<UopId> = std::iter::once(self.uop).chain(srcs.iter().map(|t| t.uop)).collect();
        Tensor { uop: f(ctx, uops), grad: None, is_param: true }
    }

    /// `Tensor.const`: `Tensor(UOp.const(b, dtype))`.
    pub fn const_(ctx: &mut UOpCtx, b: ConstVal, dtype: Option<DTypeId>) -> Tensor {
        Tensor::from_uop(ctx.const_py(b, dtype))
    }

    /// `Tensor(Variable(...))`-style: 0-d BUFFER in the ALU addrspace with a value range.
    pub fn variable(ctx: &mut UOpCtx, name: &str, vmin: ConstVal, vmax: ConstVal, dtype: DTypeId, param: bool) -> Tensor {
        Tensor::from_uop(ctx.variable(name, vmin, vmax, dtype, param))
    }

    /// `Tensor.shape` -> the public `Shape` wrapper (list-like ergonomics).
    pub fn shape(self, ctx: &mut UOpCtx) -> Shape {
        Shape::new((*ctx.shape(self.uop).unwrap().unwrap_or_default()).clone())
    }

    pub fn dtype(&self, ctx: &UOpCtx) -> DTypeId {
        ctx.dt_of(self.uop)
    }

    /// `Tensor.reshape`: public wrapper taking any `ShapeArg`.
    pub fn reshape(self, ctx: &mut UOpCtx, shape: impl ShapeArg) -> Tensor {
        Tensor::from_uop(movement::reshape(ctx, self.uop, &shape.into_shape_vec()))
    }

    /// `Tensor.expand`.
    pub fn expand(self, ctx: &mut UOpCtx, shape: impl ShapeArg) -> Tensor {
        Tensor::from_uop(movement::expand(ctx, self.uop, &shape.into_shape_vec()))
    }

    /// `Tensor.unsqueeze(dim)`.
    pub fn unsqueeze(self, ctx: &mut UOpCtx, dim: i64) -> Tensor {
        Tensor::from_uop(movement::unsqueeze(ctx, self.uop, dim))
    }

    /// `Tensor.pad(arg)`: (lo, hi) per dim.
    pub fn pad(self, ctx: &mut UOpCtx, arg: &[(i64, i64)]) -> Tensor {
        let pairs: Vec<(SVal, SVal)> = arg.iter().map(|(a, b)| (SVal::I(*a as i128), SVal::I(*b as i128))).collect();
        Tensor::from_uop(movement::pad(ctx, self.uop, &pairs))
    }

    /// `Tensor.permute(order)`.
    pub fn permute(self, ctx: &mut UOpCtx, order: &[i64]) -> Tensor {
        Tensor::from_uop(movement::permute(ctx, self.uop, order))
    }

    /// `Tensor.flip(axis)`.
    pub fn flip(self, ctx: &mut UOpCtx, axis: &[usize]) -> Tensor {
        Tensor::from_uop(movement::flip(ctx, self.uop, axis))
    }

    /// `Tensor.transpose(dim0, dim1)`.
    pub fn transpose(self, ctx: &mut UOpCtx, dim0: i64, dim1: i64) -> Tensor {
        Tensor::from_uop(movement::transpose(ctx, self.uop, dim0, dim1))
    }

    /// `Tensor.squeeze(dim=None)`.
    pub fn squeeze(self, ctx: &mut UOpCtx, dim: Option<i64>) -> Tensor {
        Tensor::from_uop(movement::squeeze(ctx, self.uop, dim))
    }

    /// `Tensor.flatten(start_dim, end_dim)`.
    pub fn flatten(self, ctx: &mut UOpCtx, start_dim: i64, end_dim: i64) -> Tensor {
        Tensor::from_uop(movement::flatten(ctx, self.uop, start_dim, end_dim))
    }

    /// `Tensor[t1, t2, ...]` — the `s![...]` index spec.
    pub fn getitem(self, ctx: &mut UOpCtx, ixs: &[Ix]) -> Tensor {
        Tensor::from_uop(movement::getitem(ctx, self.uop, ixs))
    }

    /// `Tensor.is_param_`: sets the flag in place.
    pub fn is_param_(&mut self, is_param: bool) {
        self.is_param = is_param;
    }
}

// *** UOp-level builders used by the mixins (ops.py) ***

/// `UOp.sink`: `Ops.SINK` over srcs (ops.py:548).
pub fn sink(ctx: &mut UOpCtx, srcs: Vec<UopId>) -> UopId {
    ctx.uop(Ops::Sink, None, srcs, Arg::None)
}

/// `UOp.mstack`: `MSTACK(self, *srcs)` or self when empty (ops.py:751).
pub fn mstack(ctx: &mut UOpCtx, x: UopId, srcs: Vec<UopId>) -> UopId {
    if srcs.is_empty() {
        return x;
    }
    ctx.uop(Ops::Mstack, None, std::iter::once(x).chain(srcs).collect(), Arg::None)
}

/// `UOp.after`: `AFTER(self, *src)` or self when empty (ops.py).
pub fn after(ctx: &mut UOpCtx, x: UopId, srcs: Vec<UopId>) -> UopId {
    if srcs.is_empty() {
        return x;
    }
    ctx.uop(Ops::After, None, std::iter::once(x).chain(srcs).collect(), Arg::None)
}

/// `UOp.store`: `STORE(dest, value[, gate])` (ops.py).
pub fn store(ctx: &mut UOpCtx, dest: UopId, src: UopId, gate: Option<UopId>) -> UopId {
    let mut srcs = vec![dest, src];
    if let Some(g) = gate {
        srcs.push(g);
    }
    ctx.uop(Ops::Store, None, srcs, Arg::None)
}

/// `UOp.load`: `LOAD(self, *src)` (ops.py).
pub fn load(ctx: &mut UOpCtx, x: UopId, srcs: Vec<UopId>) -> UopId {
    ctx.uop(Ops::Load, None, std::iter::once(x).chain(srcs).collect(), Arg::None)
}

/// `UOp.without_after` (ops.py): unwrap one AFTER level.
pub fn without_after(ctx: &UOpCtx, id: UopId) -> UopId {
    if ctx.op_of(id) == Ops::After {
        return ctx.srcs(id)[0];
    }
    id
}

/// `UOp.stack` (ops.py): STACK over promoted srcs (constants recast to the common dtype).
pub fn stack(ctx: &mut UOpCtx, srcs: Vec<UopId>) -> UopId {
    let dt = ctx.promo_dtype(&srcs);
    let srcs = srcs
        .iter()
        .map(|&u| {
            if ctx.is_invalid(ctx.base(u)) {
                u
            } else if ctx.op_of(u) == Ops::Const {
                ctx.const_like_dt(u, ctx.val(u), Some(dt))
            } else {
                ctx.cast(u, dt)
            }
        })
        .collect();
    ctx.uop(Ops::Stack, None, srcs, Arg::None)
}

/// `UOp.invalid()` / `UOp.valid(cond)` (ops.py).
pub fn invalid(ctx: &mut UOpCtx) -> UopId {
    ctx.const_py(ConstVal::Invalid, None)
}

pub fn valid(ctx: &mut UOpCtx, x: UopId, cond: UopId) -> UopId {
    let inv = ctx.const_like_dt(x, ConstVal::Invalid, None);
    ctx.where_(cond, x, inv)
}

/// ops enum value as an Arg (for REDUCE `(op, n)` args and similar).
pub fn op_arg(op: Ops) -> Arg {
    Arg::Enum(EnumVal { class: EnumClass::Ops, value: op as u16 })
}

impl Tensor {
    /// `Tensor(data=list, ...)` — host-data constructor (`&[T]`, T: SliceElem).
    pub fn from_slice<T: creation::SliceElem>(ctx: &mut UOpCtx, data: &[T], dtype: Option<DTypeId>, device: Option<&str>) -> Tensor {
        creation::from_slice(ctx, data, dtype, device)
    }

    /// `Tensor(data=bytes, dtype, device=?)` — raw bytes + dtype.
    pub fn from_bytes(ctx: &mut UOpCtx, data: &[u8], dtype: DTypeId, device: Option<&str>) -> Tensor {
        creation::from_bytes(ctx, data, dtype, device)
    }

    /// `Tensor.full(shape, fill_value, ...)`.
    pub fn full(ctx: &mut UOpCtx, shape: impl ShapeArg, fill_value: ConstVal, dtype: Option<DTypeId>, device: Option<&str>) -> Tensor {
        creation::full(ctx, &shape.into_shape_vec(), fill_value, dtype, device, true)
    }

    /// `Tensor.full(shape, fill_value, buffer=False)` — the broadcast const.
    pub fn full_const(ctx: &mut UOpCtx, shape: impl ShapeArg, fill_value: ConstVal, dtype: Option<DTypeId>) -> Tensor {
        creation::full(ctx, &shape.into_shape_vec(), fill_value, dtype, None, false)
    }

    /// `Tensor.zeros(*shape)`.
    pub fn zeros(ctx: &mut UOpCtx, shape: impl ShapeArg) -> Tensor {
        creation::zeros(ctx, &shape.into_shape_vec(), None, None)
    }

    /// `Tensor.ones(*shape)`.
    pub fn ones(ctx: &mut UOpCtx, shape: impl ShapeArg) -> Tensor {
        creation::ones(ctx, &shape.into_shape_vec(), None, None)
    }

    /// `Tensor.arange(start, stop, step, dtype)`.
    pub fn arange(ctx: &mut UOpCtx, start: f64, stop: Option<f64>, step: f64, dtype: Option<DTypeId>) -> Tensor {
        creation::arange(ctx, start, stop, step, dtype)
    }

    /// `Tensor.eye(n, m, dtype)`.
    pub fn eye(ctx: &mut UOpCtx, n: i64, m: Option<i64>, dtype: Option<DTypeId>) -> Tensor {
        creation::eye(ctx, n, m, dtype)
    }

    /// `Tensor.linspace(start, stop, steps, dtype)`.
    pub fn linspace(ctx: &mut UOpCtx, start: f64, stop: f64, steps: i64, dtype: Option<DTypeId>) -> Tensor {
        creation::linspace(ctx, start, stop, steps, dtype)
    }

    /// Tensor-level reduce sugar.
    pub fn sum(self, ctx: &mut UOpCtx, axis: Option<&[usize]>, keepdim: bool, dtype: Option<DTypeId>) -> Tensor {
        Tensor::from_uop(reduce::sum(ctx, self.uop, axis, keepdim, dtype))
    }

    pub fn prod(self, ctx: &mut UOpCtx, axis: Option<&[usize]>, keepdim: bool, dtype: Option<DTypeId>) -> Tensor {
        Tensor::from_uop(reduce::prod(ctx, self.uop, axis, keepdim, dtype))
    }

    pub fn max(self, ctx: &mut UOpCtx, axis: Option<&[usize]>, keepdim: bool) -> Tensor {
        Tensor::from_uop(reduce::max(ctx, self.uop, axis, keepdim))
    }

    pub fn mean(self, ctx: &mut UOpCtx, axis: Option<&[usize]>, keepdim: bool) -> Tensor {
        Tensor::from_uop(reduce::mean(ctx, self.uop, axis, keepdim))
    }

    pub fn var(self, ctx: &mut UOpCtx, axis: Option<&[usize]>, keepdim: bool, correction: i64) -> Tensor {
        Tensor::from_uop(reduce::var(ctx, self.uop, axis, keepdim, correction))
    }

    pub fn cumsum(self, ctx: &mut UOpCtx, axis: i64) -> Tensor {
        Tensor::from_uop(reduce::cumsum(ctx, self.uop, axis))
    }

    /// `Tensor.alu`: `u[0].alu(op, u[1:])` — a bare op ctor; promotion/broadcast happen
    /// in `dtype_from_uop` during construction.
    pub fn alu(self, ctx: &mut UOpCtx, op: Ops, srcs: &[Tensor]) -> Tensor {
        self.apply_uop(ctx, srcs, |c, u| c.alu(op, u))
    }
}
