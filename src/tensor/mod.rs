//! M3: the `Tensor` front-end. v0.14's `Tensor` is a thin wrapper over `uop: UOp`
//! (tensor.py:274); all sugar (movement/elementwise/reduce/creation) is UOp-level
//! via the mixin chain and returns UOps. Submodules mirror the mixin files.


pub mod callify;
pub mod ix;
pub use ix::Ix;
mod shape;
mod shapearg;

pub use shape::Shape;
pub use shapearg::ShapeArg;
pub mod creation;
mod dtype;
pub mod elementwise;
pub mod grad;
pub mod movement;
pub mod nn;
pub mod optim;
pub mod reduce;
pub mod rand;

use crate::dtype::{AddrSpace, ConstVal};

/// arg for Tensor.add/mul: another Tensor or a python scalar.
pub enum TensorAddArg {
    T(Tensor),
    C(ConstVal),
}
impl From<Tensor> for TensorAddArg {
    fn from(t: Tensor) -> Self {
        TensorAddArg::T(t)
    }
}
impl From<&Tensor> for TensorAddArg {
    fn from(t: &Tensor) -> Self {
        TensorAddArg::T(t.clone())
    }
}
impl From<f64> for TensorAddArg {
    fn from(v: f64) -> Self {
        TensorAddArg::C(ConstVal::Float(v))
    }
}
impl From<i64> for TensorAddArg {
    fn from(v: i64) -> Self {
        TensorAddArg::C(ConstVal::Int(v as i128))
    }
}
impl From<ConstVal> for TensorAddArg {
    fn from(v: ConstVal) -> Self {
        TensorAddArg::C(v)
    }
}
use crate::uop::arg::{Arg, EnumClass, EnumVal};
use crate::uop::ctx::UOpCtx;
use crate::uop::graph::UopId;
use crate::uop::ops::Ops;
use crate::dtype::DTypeId;
use crate::dtype::ReadDtype;
use crate::runtime::buffer::BufTable;
use crate::engine::realize::ProgCache;
use crate::uop::sint::{SInt, SVal};

/// Python `Tensor.__slots__ = ("uop",)`. A `Tensor` is a plain `Copy` handle
/// onto an interned `UopId`. `grad` is tracked in `UOpCtx` keyed by `UopId`
/// (a gradient is itself just a `UopId`), so `backward` can populate grads
/// without `&mut Tensor`. Whether a tensor is a *trainable parameter* is a
/// property of the `Module` tree (see [`nn::Module`]), not of the tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Tensor {
    pub uop: UopId,
}

impl Tensor {
    pub fn from_uop(uop: UopId) -> Tensor {
        Tensor { uop }
    }

    /// `Tensor._apply_uop`: map a UOp-level fxn over `(self, *x).uop` and re-wrap.
    pub fn apply_uop(&self, ctx: &mut UOpCtx, srcs: &[Tensor], f: impl FnOnce(&mut UOpCtx, Vec<UopId>) -> UopId) -> Tensor {
        let uops: Vec<UopId> = std::iter::once(self.uop).chain(srcs.iter().map(|t| t.uop)).collect();
        Tensor::from_uop(f(ctx, uops))
    }

    /// The autograd gradient attached to this tensor's `uop` in `ctx` (see
    /// [`UOpCtx::grad`]). `None` until `backward`/`gradient` populates it.
    pub fn grad(&self, ctx: &UOpCtx) -> Option<Tensor> {
        ctx.grad(self.uop).map(Tensor::from_uop)
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
    pub fn shape(&self, ctx: &mut UOpCtx) -> Shape {
        Shape::new((*ctx.shape(self.uop).unwrap().unwrap_or_default()).clone())
    }

    pub fn dtype(&self, ctx: &UOpCtx) -> DTypeId {
        ctx.dt_of(self.uop)
    }

    /// `Tensor.cast(dtype)`: in-graph cast to another dtype (a no-op when equal).
    pub fn cast(&self, ctx: &mut UOpCtx, dtype: DTypeId) -> Tensor {
        Tensor::from_uop(ctx.cast(self.uop, dtype))
    }

    /// Trigger the compute needed to read this tensor and return its raw storage
    /// bytes at its own dtype (python `_buffer()` copyout). For typed,
    /// cast-aware read-out use [`Tensor::to_list`] / [`Tensor::item`].
    pub fn to_bytes(&self, ctx: &mut UOpCtx) -> Vec<u8> {
        self.to_bytes_with(ctx, &mut BufTable::default(), &mut ProgCache::default())
    }

    /// [`Tensor::to_bytes`] with a caller-owned buffer/program cache, so realized
    /// buffers can be reused across many reads (e.g. the test harness).
    pub fn to_bytes_with(&self, ctx: &mut UOpCtx, t: &mut BufTable, cache: &mut ProgCache) -> Vec<u8> {
        crate::engine::realize::tensor_bytes(ctx, self.uop, t, cache)
    }

    /// `Tensor.realize`: trigger the computation needed to materialize this tensor
    /// and return `&self`. Buffers are not persisted across calls in this revision
    /// (see CURRENT_STATE.md open question #1), so a later read recomputes.
    pub fn realize(&self, ctx: &mut UOpCtx) -> &Tensor {
        let _ = self.to_bytes(ctx);
        self
    }

    /// Read the tensor out as `Vec<T>`: cast to `T::DTYPE` in the graph (a no-op
    /// when already that dtype), realize, and reinterpret the little-endian bytes
    /// — no `f64` round-trip. Panics if the tensor cannot be cast to `T::DTYPE`.
    pub fn to_list<T: ReadDtype>(&self, ctx: &mut UOpCtx) -> Vec<T> {
        self.to_list_with(ctx, &mut BufTable::default(), &mut ProgCache::default())
    }

    /// [`Tensor::to_list`] with a caller-owned buffer/program cache.
    pub fn to_list_with<T: ReadDtype>(&self, ctx: &mut UOpCtx, t: &mut BufTable, cache: &mut ProgCache) -> Vec<T> {
        let target = <T as ReadDtype>::DTYPE;
        let src = self.dtype(ctx);
        assert!(!crate::dtype::is_weak(src), "TGERR: cannot read a weak-dtype tensor ({src:?}) into {target:?}; realize it first");
        assert!(!crate::dtype::is_weak(target), "TGERR: cannot read tensor into weak dtype {target:?}");
        let uop = ctx.cast(self.uop, target);
        let bytes = crate::engine::realize::tensor_bytes(ctx, uop, t, cache);
        let isz = target.itemsize();
        assert_eq!(bytes.len() % isz, 0, "TGERR: byte length {} not a multiple of itemsize {isz} ({target:?})", bytes.len());
        bytes.chunks_exact(isz).map(T::from_bytes_le).collect()
    }

    /// Read a single-element tensor out as `T` (python `item()`); asserts `numel()==1`.
    pub fn item<T: ReadDtype>(&self, ctx: &mut UOpCtx) -> T {
        let n = self.shape(ctx).numel_int().unwrap_or(0);
        assert_eq!(n, 1, "TGERR: item() requires exactly one element, tensor has {n}");
        self.to_list::<T>(ctx).into_iter().next().expect("TGERR: item() on empty tensor")
    }

    /// `Tensor.reshape`: public wrapper taking any `ShapeArg`.
    pub fn reshape(&self, ctx: &mut UOpCtx, shape: impl ShapeArg) -> Tensor {
        Tensor::from_uop(movement::reshape(ctx, self.uop, &shape.into_shape_vec()))
    }

    /// `Tensor.expand`.
    pub fn expand(&self, ctx: &mut UOpCtx, shape: impl ShapeArg) -> Tensor {
        Tensor::from_uop(movement::expand(ctx, self.uop, &shape.into_shape_vec()))
    }

    /// `Tensor.unsqueeze(dim)`.
    pub fn unsqueeze(&self, ctx: &mut UOpCtx, dim: i64) -> Tensor {
        Tensor::from_uop(movement::unsqueeze(ctx, self.uop, dim))
    }

    /// `Tensor.pad(arg)`: (lo, hi) per dim.
    pub fn pad(&self, ctx: &mut UOpCtx, arg: &[(i64, i64)]) -> Tensor {
        let pairs: Vec<(SVal, SVal)> = arg.iter().map(|(a, b)| (SVal::I(*a as i128), SVal::I(*b as i128))).collect();
        Tensor::from_uop(movement::pad(ctx, self.uop, &pairs))
    }

    /// `Tensor.permute(order)`.
    pub fn permute(&self, ctx: &mut UOpCtx, order: &[i64]) -> Tensor {
        Tensor::from_uop(movement::permute(ctx, self.uop, order))
    }

    /// `Tensor.flip(axis)`.
    pub fn flip(&self, ctx: &mut UOpCtx, axis: &[usize]) -> Tensor {
        Tensor::from_uop(movement::flip(ctx, self.uop, axis))
    }

    /// `Tensor.transpose(dim0, dim1)`.
    pub fn transpose(&self, ctx: &mut UOpCtx, dim0: i64, dim1: i64) -> Tensor {
        Tensor::from_uop(movement::transpose(ctx, self.uop, dim0, dim1))
    }

    /// `Tensor.squeeze(dim=None)`.
    pub fn squeeze(&self, ctx: &mut UOpCtx, dim: Option<i64>) -> Tensor {
        Tensor::from_uop(movement::squeeze(ctx, self.uop, dim))
    }

    /// `Tensor.flatten(start_dim, end_dim)`.
    pub fn flatten(&self, ctx: &mut UOpCtx, start_dim: i64, end_dim: i64) -> Tensor {
        Tensor::from_uop(movement::flatten(ctx, self.uop, start_dim, end_dim))
    }

    /// `Tensor[t1, t2, ...]` — the `s![...]` index spec.
    pub fn getitem(&self, ctx: &mut UOpCtx, ixs: &[Ix]) -> Tensor {
        Tensor::from_uop(movement::getitem(ctx, self.uop, ixs))
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

    /// `Tensor.rand(*shape)`: random uniform [0, 1).
    pub fn rand(ctx: &mut UOpCtx, shape: &[i64], dtype: Option<DTypeId>, device: Option<&str>) -> Tensor {
        let s: Vec<SVal> = shape.iter().map(|&x| SVal::I(x as i128)).collect();
        creation::rand(ctx, &s, dtype, device)
    }

    /// `Tensor.uniform(*shape, low=0.0, high=1.0)`: random uniform [low, high).
    pub fn uniform(ctx: &mut UOpCtx, shape: &[i64], low: Option<f64>, high: Option<f64>, dtype: Option<DTypeId>, device: Option<&str>) -> Tensor {
        let s: Vec<SVal> = shape.iter().map(|&x| SVal::I(x as i128)).collect();
        let low = low.unwrap_or(0.0);
        let high = high.unwrap_or(1.0);
        creation::uniform(ctx, &s, low, high, dtype, device)
    }

    /// `Tensor.randn(*shape)`: random values from a standard normal distribution.
    pub fn randn(ctx: &mut UOpCtx, shape: &[i64], dtype: Option<DTypeId>, device: Option<&str>) -> Tensor {
        let s: Vec<SVal> = shape.iter().map(|&x| SVal::I(x as i128)).collect();
        rand::randn(ctx, &s, dtype, device)
    }

    /// `Tensor.randint(*shape, low, high)`: random integers in `[low, high)`.
    pub fn randint(ctx: &mut UOpCtx, shape: &[i64], low: i32, high: i32, dtype: Option<DTypeId>, device: Option<&str>) -> Tensor {
        let s: Vec<SVal> = shape.iter().map(|&x| SVal::I(x as i128)).collect();
        rand::randint(ctx, &s, low, high, dtype, device)
    }

    /// `Tensor.manual_seed(seed)` (tensor.py:624): reset the Threefry RNG state.
    pub fn manual_seed(seed: i64) {
        rand::manual_seed(seed)
    }

    fn prod_tail(shape: &[i64]) -> f64 {
        shape[1..].iter().map(|&x| x as f64).product()
    }

    /// `Tensor.kaiming_uniform(*shape, a=0.01)` (rand.py:208):
    /// `uniform(-bound, bound)` with `bound = sqrt(6/(1+a^2)/prod(shape[1:]))`.
    pub fn kaiming_uniform(ctx: &mut UOpCtx, shape: &[i64], a: f64, device: Option<&str>) -> Tensor {
        let bound = (6.0 / (1.0 + a * a) / Tensor::prod_tail(shape)).sqrt();
        let s: Vec<SVal> = shape.iter().map(|&x| SVal::I(x as i128)).collect();
        creation::uniform(ctx, &s, -bound, bound, None, device)
    }

    /// `Tensor.glorot_uniform(*shape)` (rand.py:192):
    /// `bound = sqrt(6/(shape[0]+prod(shape[1:])))`.
    pub fn glorot_uniform(ctx: &mut UOpCtx, shape: &[i64], device: Option<&str>) -> Tensor {
        let head = shape[0] as f64;
        let bound = (6.0 / (head + Tensor::prod_tail(shape))).sqrt();
        let s: Vec<SVal> = shape.iter().map(|&x| SVal::I(x as i128)).collect();
        creation::uniform(ctx, &s, -bound, bound, None, device)
    }

    /// `Tensor.scaled_uniform(*shape)` (rand.py:176):
    /// `uniform(-1, 1) * prod(shape)**-0.5`.
    pub fn scaled_uniform(ctx: &mut UOpCtx, shape: &[i64], device: Option<&str>) -> Tensor {
        let s: Vec<SVal> = shape.iter().map(|&x| SVal::I(x as i128)).collect();
        let t = creation::uniform(ctx, &s, -1.0, 1.0, None, device);
        let prod: f64 = shape.iter().map(|&x| x as f64).product();
        t.mul(ctx, prod.powf(-0.5))
    }

    /// Tensor-level reduce sugar.
    pub fn sum(&self, ctx: &mut UOpCtx, axis: Option<&[usize]>, keepdim: bool, dtype: Option<DTypeId>) -> Tensor {
        Tensor::from_uop(reduce::sum(ctx, self.uop, axis, keepdim, dtype))
    }

    pub fn prod(&self, ctx: &mut UOpCtx, axis: Option<&[usize]>, keepdim: bool, dtype: Option<DTypeId>) -> Tensor {
        Tensor::from_uop(reduce::prod(ctx, self.uop, axis, keepdim, dtype))
    }

    pub fn max(&self, ctx: &mut UOpCtx, axis: Option<&[usize]>, keepdim: bool) -> Tensor {
        Tensor::from_uop(reduce::max(ctx, self.uop, axis, keepdim))
    }

    pub fn mean(&self, ctx: &mut UOpCtx, axis: Option<&[usize]>, keepdim: bool) -> Tensor {
        Tensor::from_uop(reduce::mean(ctx, self.uop, axis, keepdim))
    }

    pub fn var(&self, ctx: &mut UOpCtx, axis: Option<&[usize]>, keepdim: bool, correction: i64) -> Tensor {
        Tensor::from_uop(reduce::var(ctx, self.uop, axis, keepdim, correction))
    }

    pub fn cumsum(&self, ctx: &mut UOpCtx, axis: i64) -> Tensor {
        Tensor::from_uop(reduce::cumsum(ctx, self.uop, axis))
    }

    // *** nn-layer operations ***

    /// `Tensor.sub(other)`: `self - other` (composed via add/neg).
    pub fn sub(&self, ctx: &mut UOpCtx, other: impl Into<TensorAddArg>) -> Tensor {
        match other.into() {
            TensorAddArg::T(o) => Tensor::from_uop(elementwise::sub(ctx, self.uop, o.uop)),
            TensorAddArg::C(c) => Tensor::from_uop(elementwise::sub_c(ctx, self.uop, c)),
        }
    }

    /// `Tensor.matmul(other)`: matrix multiply — last-1 of self × last-1 of other, sum over that dim.
    /// Handles batch dims: self (N, K) @ other (K, M) → (N, M).
    /// For general batch shapes: self (.., N, K), other (.., K, M) → (.., N, M).
    pub fn matmul(&self, ctx: &mut UOpCtx, other: impl Into<TensorAddArg>) -> Tensor {
        // tinygrad `matmul` is just `dot` (mixin/op.py:407) — `dot` handles the
        // batch/contraction reshaping itself, so there is no extra batch logic.
        let other = match other.into() {
            TensorAddArg::T(o) => o,
            TensorAddArg::C(_) => panic!("matmul: RHS must be a Tensor, not a scalar"),
        };
        self.dot(ctx, &other)
    }

    /// `Tensor.square()`: `self * self`.
    pub fn square(&self, ctx: &mut UOpCtx) -> Tensor {
        Tensor::from_uop(elementwise::square(ctx, self.uop))
    }

    /// `Tensor.sqrt()`: square root.
    pub fn sqrt(&self, ctx: &mut UOpCtx) -> Tensor {
        Tensor::from_uop(elementwise::sqrt(ctx, self.uop))
    }

    /// `Tensor.mean()`: mean over all dims.
    pub fn mean_all(&self, ctx: &mut UOpCtx) -> Tensor {
        self.mean(ctx, None, false)
    }

    /// `Tensor.abs()`: max(self, -self).
    pub fn abs(&self, ctx: &mut UOpCtx) -> Tensor {
        let neg = self.neg(ctx);
        self.maximum(ctx, &neg)
    }

    /// `Tensor.maximum(other)`: elementwise max.
    pub fn maximum(&self, ctx: &mut UOpCtx, other: impl Into<TensorAddArg>) -> Tensor {
        match other.into() {
            TensorAddArg::T(o) => Tensor::from_uop(elementwise::maximum(ctx, self.uop, o.uop)),
            TensorAddArg::C(c) => Tensor::from_uop(elementwise::binop_c(ctx, Ops::Max, self.uop, c)),
        }
    }

    /// `Tensor.minimum(other)`: elementwise min.
    pub fn minimum(&self, ctx: &mut UOpCtx, other: impl Into<TensorAddArg>) -> Tensor {
        match other.into() {
            TensorAddArg::T(o) => Tensor::from_uop(elementwise::minimum(ctx, self.uop, o.uop)),
            // No Ops::MIN — use min(a, b) == a + b - max(a, b).
            TensorAddArg::C(c) => {
                let sum = self.add(ctx, TensorAddArg::C(c.clone()));
                let mx = self.maximum(ctx, TensorAddArg::C(c));
                sum.sub(ctx, mx)
            }
        }
    }

    /// `Tensor.reciprocal()`: 1 / self.
    pub fn reciprocal(&self, ctx: &mut UOpCtx) -> Tensor {
        Tensor::from_uop(elementwise::reciprocal(ctx, self.uop))
    }

    /// `Tensor.log()`: natural log.
    pub fn log(&self, ctx: &mut UOpCtx) -> Tensor {
        Tensor::from_uop(elementwise::log(ctx, self.uop))
    }

    /// `Tensor.log2()`: log base 2.
    pub fn log2(self, ctx: &mut UOpCtx) -> Tensor {
        Tensor::from_uop(elementwise::log2(ctx, self.uop))
    }

    /// `Tensor.exp()`: exponential.
    pub fn exp(&self, ctx: &mut UOpCtx) -> Tensor {
        Tensor::from_uop(elementwise::exp(ctx, self.uop))
    }

    /// `Tensor.exp2()`: 2^self.
    pub fn exp2(self, ctx: &mut UOpCtx) -> Tensor {
        Tensor::from_uop(elementwise::exp2(ctx, self.uop))
    }

    /// `Tensor.trunc()`: truncate toward zero.
    pub fn trunc(&self, ctx: &mut UOpCtx) -> Tensor {
        Tensor::from_uop(elementwise::trunc(ctx, self.uop))
    }

    /// `Tensor.ceil()`: ceiling.
    pub fn ceil(&self, ctx: &mut UOpCtx) -> Tensor {
        Tensor::from_uop(elementwise::ceil(ctx, self.uop))
    }

    /// `Tensor.floor()`: floor.
    pub fn floor(&self, ctx: &mut UOpCtx) -> Tensor {
        Tensor::from_uop(elementwise::floor(ctx, self.uop))
    }

    /// `Tensor.sin()`: sine.
    pub fn sin(&self, ctx: &mut UOpCtx) -> Tensor {
        Tensor::from_uop(elementwise::sin(ctx, self.uop))
    }

    /// `Tensor.tanh()`: hyperbolic tangent.
    pub fn tanh(&self, ctx: &mut UOpCtx) -> Tensor {
        Tensor::from_uop(elementwise::tanh(ctx, self.uop))
    }

    /// `Tensor.dot(w)`: sum product over last axis of self and second-to-last of w.
    /// self (.., N) @ w (.., K, N) → (.., K) — mirrors tinygrad's dot.
    pub fn dot(&self, ctx: &mut UOpCtx, w: &Tensor) -> Tensor {
        let x_shape = self.shape(ctx);
        let w_shape = w.shape(ctx);
        let dx = x_shape.len();
        let dw = w_shape.len();
        assert!(dx > 0 && dw > 0, "Tensor::dot: both tensors must be at least 1D");
        // contraction axis of w: index dw - min(dw, 2) (last axis for 1-D w, second-to-last for N-D w)
        let axis_w = (dw - std::cmp::min(dw, 2)) as usize;
        let x_last = x_shape[dx - 1].as_i64().unwrap_or(0);
        let w_contract = w_shape[axis_w].as_i64().unwrap_or(0);
        assert_eq!(x_last, w_contract, "Tensor::dot: shape mismatch {} vs {}", x_shape, w_shape);

        let rep = std::cmp::min(std::cmp::min(dx - 1, dw - 1), 1);
        // x -> (*x.shape[:-1], [1]*rep, x.shape[-1])
        let mut xr: Vec<SVal> = x_shape[..dx - 1].to_vec();
        xr.extend(std::iter::repeat(SVal::I(1)).take(rep));
        xr.push(x_shape[dx - 1].clone());
        let x_r = self.reshape(ctx, xr);

        // w -> (*w.shape[:-2], [1]*rep, *w.shape[axis_w:]) then transpose(-1, axis_w - dw)
        let mut wr: Vec<SVal> = w_shape[..dw.saturating_sub(2)].to_vec();
        wr.extend(std::iter::repeat(SVal::I(1)).take(rep));
        wr.extend_from_slice(&w_shape[axis_w..]);
        let w_r = w.reshape(ctx, wr);
        let axis_w_neg = -(std::cmp::min(dw, 2) as i64);
        let w_t = w_r.transpose(ctx, -1, axis_w_neg);

        // sum-reduce the broadcasted product over the last axis
        let product = x_r.mul(ctx, &w_t);
        let last_axis = product.shape(ctx).len() - 1;
        product.sum(ctx, Some(&[last_axis]), false, None)
    }

    /// `Tensor.linear(weight, bias)`: dot/mul + bias add.
    /// If weight is 1D: self * weight + bias (elementwise).
    /// Otherwise: self.dot(weight) + bias.
    pub fn linear(&self, ctx: &mut UOpCtx, weight: &Tensor, bias: Option<&Tensor>) -> Tensor {
        let w_shape = weight.shape(ctx);
        if w_shape.len() == 1 {
            // For 1D weight, broadcast: expand self with a singleton at the end, then add bias.
            let mut expand_shape = w_shape.to_vec();
            expand_shape.push(SVal::I(1));
            let expanded = self.expand(ctx, expand_shape);
            return match bias {
                Some(b) => expanded.add(ctx, b),
                None => expanded,
            };
        }
        let result = self.dot(ctx, weight);
        match bias {
            Some(b) => result.add(ctx, b),
            None => result,
        }
    }

    /// `Tensor.batchnorm(weight, bias, mean, invstd)`: batch normalization.
    pub fn batchnorm(
        &self,
        ctx: &mut UOpCtx,
        weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        mean: &Tensor,
        invstd: &Tensor,
        axis: &[usize],
    ) -> Tensor {
        let shape = self.shape(ctx);
        // Build reduction shape: dim i is 1 if i in axis, else shape[i]
        let mut norm_shape = Vec::with_capacity(shape.len());
        for (i, s) in shape.iter().enumerate() {
            if axis.contains(&i) {
                norm_shape.push(SVal::I(1));
            } else {
                norm_shape.push(s.clone());
            }
        }
        let mean_reshaped = mean.reshape(ctx, &norm_shape);
        let mut x = self.sub(ctx, &mean_reshaped);
        if let Some(w) = weight {
            let w_reshaped = w.reshape(ctx, &norm_shape);
            x = x.mul(ctx, &w_reshaped);
        }
        let invstd_reshaped = if invstd.shape(ctx).len() == axis.len() {
            invstd.reshape(ctx, &norm_shape)
        } else {
            invstd.clone()
        };
        let ret = x.mul(ctx, &invstd_reshaped);
        match bias {
            Some(b) => {
                let b_reshaped = b.reshape(ctx, &norm_shape);
                ret.add(ctx, &b_reshaped)
            }
            None => ret,
        }
    }

    /// `Tensor.layernorm(axis, eps)`: layer normalization.
    pub fn layernorm(&self, ctx: &mut UOpCtx, axis: i64, eps: f64) -> Tensor {
        // Normalize over the axis dimension(s)
        let ndim = self.shape(ctx).len();
        let actual_axis = if axis < 0 { ndim as i64 + axis } else { axis };
        let axis_vec = vec![actual_axis as usize];
        let mean = self.mean(ctx, Some(&axis_vec), true);
        let diff = self.sub(ctx, TensorAddArg::T(mean));
        let var = diff.square(ctx).mean(ctx, Some(&axis_vec), true);
        let eps_tensor = Tensor::full_const(ctx, &[] as &[i64], ConstVal::Float(eps), None);
        let invstd = var.add(ctx, TensorAddArg::T(eps_tensor)).sqrt(ctx).reciprocal(ctx);
        diff.mul(ctx, TensorAddArg::T(invstd))
    }

    /// `Tensor.pool(kernel_size, stride, dilation)`: sliding window pool (for conv2d internals).
    pub fn pool(&self, ctx: &mut UOpCtx, kernel_size: &[i64], stride: i64, dilation: i64) -> Tensor {
        Tensor::from_uop(movement::pool(ctx, self.uop, kernel_size, stride, dilation))
    }

    /// `Tensor.div(other)`: elementwise division (a / b).
    pub fn div(&self, ctx: &mut UOpCtx, other: impl Into<TensorAddArg>) -> Tensor {
        match other.into() {
            TensorAddArg::T(o) => Tensor::from_uop(elementwise::div(ctx, self.uop, o.uop, None)),
            TensorAddArg::C(c) => Tensor::from_uop(elementwise::div_c(ctx, self.uop, c, None)),
        }
    }

    /// Tensor-level elementwise sugar for the oracle/schedule corpus.
    pub fn add(&self, ctx: &mut UOpCtx, other: impl Into<TensorAddArg>) -> Tensor {
        match other.into() {
            TensorAddArg::T(o) => Tensor::from_uop(elementwise::binop(ctx, Ops::Add, self.uop, o.uop)),
            TensorAddArg::C(c) => Tensor::from_uop(elementwise::binop_c(ctx, Ops::Add, self.uop, c)),
        }
    }

    pub fn mul(&self, ctx: &mut UOpCtx, other: impl Into<TensorAddArg>) -> Tensor {
        match other.into() {
            TensorAddArg::T(o) => Tensor::from_uop(elementwise::binop(ctx, Ops::Mul, self.uop, o.uop)),
            TensorAddArg::C(c) => Tensor::from_uop(elementwise::binop_c(ctx, Ops::Mul, self.uop, c)),
        }
    }

    pub fn neg(&self, ctx: &mut UOpCtx) -> Tensor {
        Tensor::from_uop(elementwise::neg(ctx, self.uop))
    }

    /// `Tensor.contiguous()` — CONTIGUOUS node when not virtual.
    pub fn contiguous(&self, ctx: &mut UOpCtx) -> Tensor {
        let dt = self.dtype(ctx);
        Tensor::from_uop(ctx.uop(Ops::Contiguous, Some(dt), vec![self.uop], Arg::None))
    }

    /// `Tensor.alu`: `u[0].alu(op, u[1:])` — a bare op ctor; promotion/broadcast happen
    /// in `dtype_from_uop` during construction.
    pub fn alu(&self, ctx: &mut UOpCtx, op: Ops, srcs: &[Tensor]) -> Tensor {
        self.apply_uop(ctx, srcs, |c, u| c.alu(op, u))
    }

    /// `Tensor.bitcast(dtype)` (`mixin/dtype.py:35`) — reinterpret the bits.
    pub fn bitcast(&self, ctx: &mut UOpCtx, dtype: DTypeId) -> Tensor {
        Tensor::from_uop(ctx.bitcast(self.uop, dtype))
    }

    /// `bitwise_or` / `bitwise_and` / `bitwise_xor` — bare binops (`mixin/elementwise.py`).
    pub fn bitwise_or(&self, ctx: &mut UOpCtx, other: impl Into<TensorAddArg>) -> Tensor {
        match other.into() {
            TensorAddArg::T(o) => Tensor::from_uop(elementwise::binop(ctx, Ops::Or, self.uop, o.uop)),
            TensorAddArg::C(c) => Tensor::from_uop(elementwise::binop_c(ctx, Ops::Or, self.uop, c)),
        }
    }
    pub fn bitwise_and(&self, ctx: &mut UOpCtx, other: impl Into<TensorAddArg>) -> Tensor {
        match other.into() {
            TensorAddArg::T(o) => Tensor::from_uop(elementwise::binop(ctx, Ops::And, self.uop, o.uop)),
            TensorAddArg::C(c) => Tensor::from_uop(elementwise::binop_c(ctx, Ops::And, self.uop, c)),
        }
    }
    pub fn bitwise_xor(&self, ctx: &mut UOpCtx, other: impl Into<TensorAddArg>) -> Tensor {
        match other.into() {
            TensorAddArg::T(o) => Tensor::from_uop(elementwise::binop(ctx, Ops::Xor, self.uop, o.uop)),
            TensorAddArg::C(c) => Tensor::from_uop(elementwise::binop_c(ctx, Ops::Xor, self.uop, c)),
        }
    }

    /// `Tensor.shl(x)` / `Tensor.rshift(x)` — `self._binop(Ops.SHL/SHR, x)`.
    pub fn shl(&self, ctx: &mut UOpCtx, bits: i64) -> Tensor {
        let b = elementwise::ufix(ctx, ConstVal::Int(bits as i128));
        Tensor::from_uop(ctx.shl(self.uop, b))
    }
    pub fn rshift(&self, ctx: &mut UOpCtx, bits: i64) -> Tensor {
        let b = elementwise::ufix(ctx, ConstVal::Int(bits as i128));
        Tensor::from_uop(ctx.shr(self.uop, b))
    }

    /// `Tensor.cat(*others, dim=0)` (`mixin/op.py:734`), equal-shape fast path:
    /// `self.stack(*others, dim).flatten(dim, dim+1)`. The RNG path always
    /// concatenates same-shape operands along dim 0, so this covers it.
    pub fn cat(&self, ctx: &mut UOpCtx, others: &[Tensor]) -> Tensor {
        let uops: Vec<UopId> = std::iter::once(self.uop).chain(others.iter().map(|t| t.uop)).collect();
        let st = Tensor::from_uop(stack(ctx, uops));
        st.flatten(ctx, 0, 2)
    }

    /// `Tensor.threefry(seed)` (`mixin/elementwise.py:454`): `self.alu(Ops.THREEFRY, seed)`.
    pub fn threefry(&self, ctx: &mut UOpCtx, seed: &Tensor) -> Tensor {
        Tensor::from_uop(ctx.uop(Ops::Threefry, Some(crate::dtype::UINT64), vec![self.uop, seed.uop], Arg::None))
    }
}
