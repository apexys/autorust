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
pub mod op;
pub mod optim;
pub mod reduce;
pub mod rand;

use crate::dtype::ConstVal;

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
use crate::uop::arg::{Arg, DeviceSpec, EnumClass, EnumVal};
use crate::uop::ctx::UOpCtx;
use crate::uop::graph::UopId;
use crate::uop::ops::Ops;
use crate::dtype::DTypeId;
use crate::dtype::ReadDtype;
use crate::runtime::buffer::Buffer;
use crate::engine::session::Session;
use std::cell::RefCell;
use std::rc::Rc;
use crate::uop::sint::{SInt, SVal};

/// Python `Tensor.__slots__ = ("uop",)`. In v0.14 a `Tensor` is a Python object
/// whose identity keeps its `UOp` (and, once realized, its buffer) alive. The Rust
/// port models that explicitly: a `Tensor` is a **`Clone`** handle (not `Copy`)
/// holding a strong [`UopId`] (`Rc<Node>`) plus an `Rc` realized-buffer cell
/// shared by all clones. `grad` is tracked in `UOpCtx` keyed by `UopId`. Whether a
/// tensor is a *trainable parameter* is a property of the `Module` tree
/// (see [`nn::Module`]), not of the tensor.
pub struct Tensor {
    pub uop: UopId,
    /// Realized storage, shared by clones and reused across reads.
    buffer: Rc<RefCell<Option<Buffer>>>,
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        Tensor { uop: self.uop.clone(), buffer: self.buffer.clone() }
    }
}

// Tensor identity is its uop (the buffer is bookkeeping).
impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.uop == other.uop
    }
}
impl Eq for Tensor {}
impl std::hash::Hash for Tensor {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.uop.hash(state);
    }
}
impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor({:?})", self.uop)
    }
}

impl Tensor {
    /// Wrap an engine uop as a `Tensor` handle. The `UopId` is a strong handle, so
    /// no extra pinning is needed; `ctx` is accepted to keep the construction
    /// uniform with the other builders.
    pub fn from_uop(_ctx: &UOpCtx, uop: UopId) -> Tensor {
        Tensor { uop, buffer: Rc::new(RefCell::new(None)) }
    }

    /// The realized storage, if any (shared with clones).
    pub fn realized_buffer(&self) -> Option<Buffer> {
        self.buffer.borrow().clone()
    }

    /// Record realized storage for this tensor (and its clones).
    pub fn set_realized_buffer(&self, b: Buffer) {
        *self.buffer.borrow_mut() = Some(b);
    }

    /// `Tensor._apply_uop`: map a UOp-level fxn over `(self, *x).uop` and re-wrap.
    pub fn apply_uop(&self, ctx: &mut UOpCtx, srcs: &[Tensor], f: impl FnOnce(&mut UOpCtx, Vec<UopId>) -> UopId) -> Tensor {
        let uops: Vec<UopId> = std::iter::once(self.uop.clone()).chain(srcs.iter().map(|t| t.uop.clone())).collect();
        let u = f(ctx, uops);
        Tensor::from_uop(ctx, u)
    }

    /// The autograd gradient attached to this tensor's `uop` in `ctx` (see
    /// [`UOpCtx::grad`]). `None` until `backward`/`gradient` populates it.
    pub fn grad(&self, ctx: &UOpCtx) -> Option<Tensor> {
        ctx.grad(&self.uop).map(|g| Tensor::from_uop(ctx, g))
    }

    /// `Tensor.const`: `Tensor(UOp.const(b, dtype))`.
    pub fn const_(ctx: &mut UOpCtx, b: ConstVal, dtype: Option<DTypeId>) -> Tensor {
        let u = ctx.const_py(b, dtype);
        Tensor::from_uop(ctx, u)
    }

    /// `Tensor(Variable(...))`-style: 0-d BUFFER in the ALU addrspace with a value range.
    pub fn variable(ctx: &mut UOpCtx, name: &str, vmin: ConstVal, vmax: ConstVal, dtype: DTypeId, param: bool) -> Tensor {
        let u = ctx.variable(name, vmin, vmax, dtype, param);
        Tensor::from_uop(ctx, u)
    }

    /// `Tensor.shape` -> the public `Shape` wrapper (list-like ergonomics).
    pub fn shape(&self, ctx: &mut UOpCtx) -> Shape {
        Shape::new((*ctx.shape(&self.uop).unwrap().unwrap_or_default()).clone())
    }

    pub fn dtype(&self, ctx: &UOpCtx) -> DTypeId {
        ctx.dt_of(&self.uop)
    }

    /// `Tensor.device` / `UOp.device` (tensor.py / ops.py:847): the device string
    /// backing this tensor's storage, or `None` when it has no placement yet
    /// (e.g. a computed expression). Walks movement/COPY/multi-device ops like
    /// the Python property.
    pub fn device(&self, ctx: &UOpCtx) -> Option<String> {
        uop_device(ctx, &self.uop)
    }

    /// `Tensor.cast(dtype)`: in-graph cast to another dtype (a no-op when equal).
    pub fn cast(&self, ctx: &mut UOpCtx, dtype: DTypeId) -> Tensor {
        { let __u = ctx.cast(self.uop.clone(), dtype); Tensor::from_uop(ctx, __u) }
    }

    /// Trigger the compute needed to read this tensor and return its raw storage
    /// bytes at its own dtype (python `_buffer()` copyout). For typed,
    /// cast-aware read-out use [`Tensor::to_list`] / [`Tensor::item`]. The
    /// realized buffer is cached on the tensor (shared by clones), so a repeat
    /// read does not re-realize.
    pub fn to_bytes(&self, ctx: &mut UOpCtx, sess: &mut Session) -> Vec<u8> {
        if let Some(b) = self.realized_buffer() {
            return crate::engine::realize::buffer_bytes(&b);
        }
        let b = sess.realize_tensor(ctx, self.uop.clone());
        let bytes = crate::engine::realize::buffer_bytes(&b);
        self.set_realized_buffer(b);
        bytes
    }

    /// [`Tensor::to_bytes`] with a throwaway [`Session`] (one-shot reads).
    pub fn to_bytes_fresh(&self, ctx: &mut UOpCtx) -> Vec<u8> {
        self.to_bytes(ctx, &mut Session::new())
    }

    /// `Tensor.assign` (tensor.py:436): realize `value` and copy its bytes into
    /// *this* tensor's storage in place, leaving `self.uop` unchanged.
    ///
    /// This is what keeps a persistent parameter a stable leaf across optimizer
    /// steps: `w - lr*grad` is built as a fresh subgraph, written back into `w`'s
    /// buffer, then dropped. Without it, rebinding `w = w - lr*grad` chains each
    /// step into the previous one and the graph grows without bound.
    pub fn assign(&self, ctx: &mut UOpCtx, sess: &mut Session, value: &Tensor) {
        assert_eq!(self.dtype(ctx), value.dtype(ctx), "TGERR: assign dtype mismatch");
        // Realize the value first: its graph reads the parameter's current bytes.
        // If the value was already co-realized (e.g. via `Tensor::realize_many`
        // in `Optimizer.step`), reuse that buffer instead of scheduling again.
        let src = match value.realized_buffer() {
            Some(b) => b,
            None => sess.realize_tensor(ctx, value.uop.clone()),
        };
        // Destination is this tensor's storage *by buffer identity* — the same
        // base the forward pass resolves through `buffer_of`, so the write is
        // visible to subsequent realizes.
        let dst = crate::runtime::buffer::buffer_of(ctx, &mut sess.buffers, self.uop.clone());
        // Same device: direct buffer-to-buffer copy (no host round-trip).
        // Different devices: host round-trip inside `buffer_copy_into`.
        crate::engine::realize::buffer_copy_into(&dst, &src);
        self.set_realized_buffer(dst);
    }

    /// Turn a computed tensor (e.g. a random-initialized parameter) into a
    /// buffer-backed leaf with the same shape/dtype, realizing the current value
    /// into its storage (`Node.initial_data`). tinygrad parameters are realized
    /// buffers, which is what lets [`Tensor::assign`] update them in place;
    /// a raw init *expression* would be recomputed (and the update lost) on every
    /// realize.
    pub fn to_param(&self, ctx: &mut UOpCtx) -> Tensor {
        let shape = self.shape(ctx);
        let dt = self.dtype(ctx);
        // Realize the init in a throwaway session (it may be on a different
        // device than the training buffers) and stage the bytes on the leaf.
        let bytes = self.to_bytes_fresh(ctx);
        let dev = self.realized_buffer().map(|b| b.device().to_string()).unwrap_or_else(|| "PYTHON".to_string());
        let u = crate::tensor::creation::empty(ctx, &shape, dt, &dev);
        crate::tensor::creation::register_initial(ctx, u.clone(), bytes);
        Tensor::from_uop(ctx, u)
    }

    /// `Tensor.realize`: trigger the computation needed to materialize this tensor
    /// and return `&self`.
    pub fn realize(&self, ctx: &mut UOpCtx, sess: &mut Session) -> &Tensor {
        let _ = self.to_bytes(ctx, sess);
        self
    }

    /// [`Tensor::realize`] with a throwaway [`Session`].
    pub fn realize_fresh(&self, ctx: &mut UOpCtx) -> &Tensor {
        let _ = self.to_bytes_fresh(ctx);
        self
    }

    /// `Tensor.realize(*tensors)` (tensor.py) — co-realize all listed tensors in
    /// a **single** schedule, caching each one's realized buffer. This is the
    /// tinygrad co-realization the optimizer relies on to fold the loss and its
    /// updates into one schedule (`Optimizer.schedule_step`).
    pub fn realize_many<'a>(ctx: &mut UOpCtx, sess: &mut Session, tensors: impl IntoIterator<Item = &'a Tensor>) -> Vec<Buffer> {
        let ts: Vec<&Tensor> = tensors.into_iter().collect();
        let uops: Vec<UopId> = ts.iter().map(|t| t.uop.clone()).collect();
        let bufs = sess.realize_many(ctx, &uops);
        for (t, b) in ts.iter().zip(bufs.iter()) {
            t.set_realized_buffer(b.clone());
        }
        bufs
    }

    /// Reserve the graph cast that reads this tensor as `T::DTYPE`, asserting
    /// weak dtypes are realized first.
    fn read_target<T: ReadDtype>(&self, ctx: &mut UOpCtx) -> UopId {
        let target = <T as ReadDtype>::DTYPE;
        let src = self.dtype(ctx);
        assert!(!crate::dtype::is_weak(src), "TGERR: cannot read a weak-dtype tensor ({src:?}) into {target:?}; realize it first");
        assert!(!crate::dtype::is_weak(target), "TGERR: cannot read tensor into weak dtype {target:?}");
        ctx.cast(self.uop.clone(), target)
    }

    /// Reinterpret little-endian bytes as `[T]` (no `f64` round-trip).
    fn decode<T: ReadDtype>(bytes: Vec<u8>) -> Vec<T> {
        let isz = <T as ReadDtype>::DTYPE.itemsize();
        assert_eq!(bytes.len() % isz, 0, "TGERR: byte length {} not a multiple of itemsize {isz}", bytes.len());
        bytes.chunks_exact(isz).map(T::from_bytes_le).collect()
    }

    /// Read the tensor out as `Vec<T>`: cast to `T::DTYPE` in the graph (a no-op
    /// when already that dtype), realize, and reinterpret the little-endian bytes
    /// — no `f64` round-trip. Panics if the tensor cannot be cast to `T::DTYPE`.
    pub fn to_list<T: ReadDtype>(&self, ctx: &mut UOpCtx, sess: &mut Session) -> Vec<T> {
        if <T as ReadDtype>::DTYPE == self.dtype(ctx) {
            return Self::decode::<T>(self.to_bytes(ctx, sess));
        }
        let uop = self.read_target::<T>(ctx);
        Self::decode::<T>(sess.tensor_bytes(ctx, uop))
    }

    /// [`Tensor::to_list`] with a throwaway [`Session`] (one-shot reads).
    pub fn to_list_fresh<T: ReadDtype>(&self, ctx: &mut UOpCtx) -> Vec<T> {
        self.to_list(ctx, &mut Session::new())
    }

    /// Read a single-element tensor out as `T` (python `item()`); asserts `numel()==1`.
    pub fn item<T: ReadDtype>(&self, ctx: &mut UOpCtx, sess: &mut Session) -> T {
        let n = self.shape(ctx).numel_int().unwrap_or(0);
        assert_eq!(n, 1, "TGERR: item() requires exactly one element, tensor has {n}");
        self.to_list::<T>(ctx, sess).into_iter().next().expect("TGERR: item() on empty tensor")
    }

    /// [`Tensor::item`] with a throwaway [`Session`].
    pub fn item_fresh<T: ReadDtype>(&self, ctx: &mut UOpCtx) -> T {
        self.item(ctx, &mut Session::new())
    }

    /// `Tensor.reshape`: public wrapper taking any `ShapeArg`.
    pub fn reshape(&self, ctx: &mut UOpCtx, shape: impl ShapeArg) -> Tensor {
        { let __u = movement::reshape(ctx, self.uop.clone(), &shape.into_shape_vec()); Tensor::from_uop(ctx, __u) }
    }

    /// `Tensor.expand`.
    pub fn expand(&self, ctx: &mut UOpCtx, shape: impl ShapeArg) -> Tensor {
        { let __u = movement::expand(ctx, self.uop.clone(), &shape.into_shape_vec()); Tensor::from_uop(ctx, __u) }
    }

    /// `Tensor.unsqueeze(dim)`.
    pub fn unsqueeze(&self, ctx: &mut UOpCtx, dim: i64) -> Tensor {
        { let __u = movement::unsqueeze(ctx, self.uop.clone(), dim); Tensor::from_uop(ctx, __u) }
    }

    /// `Tensor.pad(arg)`: (lo, hi) per dim.
    pub fn pad(&self, ctx: &mut UOpCtx, arg: &[(i64, i64)]) -> Tensor {
        let pairs: Vec<(SVal, SVal)> = arg.iter().map(|(a, b)| (SVal::I(*a as i128), SVal::I(*b as i128))).collect();
        { let __u = movement::pad(ctx, self.uop.clone(), &pairs); Tensor::from_uop(ctx, __u) }
    }

    /// `Tensor.shrink(arg)`: explicit `(start, end)` bounds per dim.
    pub fn shrink(&self, ctx: &mut UOpCtx, arg: &[(i64, i64)]) -> Tensor {
        let pairs: Vec<(SVal, SVal)> = arg.iter().map(|(a, b)| (SVal::I(*a as i128), SVal::I(*b as i128))).collect();
        { let __u = movement::shrink(ctx, self.uop.clone(), &pairs); Tensor::from_uop(ctx, __u) }
    }

    /// `Tensor.permute(order)`.
    pub fn permute(&self, ctx: &mut UOpCtx, order: &[i64]) -> Tensor {
        { let __u = movement::permute(ctx, self.uop.clone(), order); Tensor::from_uop(ctx, __u) }
    }

    /// `Tensor.flip(axis)`.
    pub fn flip(&self, ctx: &mut UOpCtx, axis: &[usize]) -> Tensor {
        { let __u = movement::flip(ctx, self.uop.clone(), axis); Tensor::from_uop(ctx, __u) }
    }

    /// `Tensor.transpose(dim0, dim1)`.
    pub fn transpose(&self, ctx: &mut UOpCtx, dim0: i64, dim1: i64) -> Tensor {
        { let __u = movement::transpose(ctx, self.uop.clone(), dim0, dim1); Tensor::from_uop(ctx, __u) }
    }

    /// `Tensor.squeeze(dim=None)`.
    pub fn squeeze(&self, ctx: &mut UOpCtx, dim: Option<i64>) -> Tensor {
        { let __u = movement::squeeze(ctx, self.uop.clone(), dim); Tensor::from_uop(ctx, __u) }
    }

    /// `Tensor.flatten(start_dim, end_dim)`.
    pub fn flatten(&self, ctx: &mut UOpCtx, start_dim: i64, end_dim: i64) -> Tensor {
        { let __u = movement::flatten(ctx, self.uop.clone(), start_dim, end_dim); Tensor::from_uop(ctx, __u) }
    }

    /// `Tensor[t1, t2, ...]` — the `s![...]` index spec.
    pub fn getitem(&self, ctx: &mut UOpCtx, ixs: &[Ix]) -> Tensor {
        { let __u = movement::getitem(ctx, self.uop.clone(), ixs); Tensor::from_uop(ctx, __u) }
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
    if ctx.op_of(&id) == Ops::After {
        return ctx.srcs(&id)[0].clone();
    }
    id
}

/// `UOp.stack` (ops.py): STACK over promoted srcs (constants recast to the common dtype).
pub fn stack(ctx: &mut UOpCtx, srcs: Vec<UopId>) -> UopId {
    let dt = ctx.promo_dtype(&srcs);
    let srcs = srcs
        .iter()
        .map(|u| {
            if ctx.is_invalid(&ctx.base(u)) {
                u.clone()
            } else if ctx.op_of(u) == Ops::Const {
                ctx.const_like_dt(u.clone(), ctx.val(u), Some(dt))
            } else {
                ctx.cast(u.clone(), dt)
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
    let inv = ctx.const_like_dt(x.clone(), ConstVal::Invalid, None);
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

    /// `Tensor.zeros(*shape)` as a persistent optimizer-state leaf: a pure
    /// BUFFER with `initial_data` and **no in-graph STORE**, so [`Tensor::assign`]
    /// can update it in place without the zero-store re-running on a later
    /// realize (see CURRENT_STATE §9.5). Use this (not [`Tensor::zeros`]) for
    /// optimizer momentum / Adam moments.
    pub fn zeros_param(ctx: &mut UOpCtx, shape: impl ShapeArg, dtype: Option<DTypeId>, device: Option<&str>) -> Tensor {
        let t = creation::zeros(ctx, &shape.into_shape_vec(), dtype, device);
        t.to_param(ctx)
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
        { let __u = reduce::sum(ctx, self.uop.clone(), axis, keepdim, dtype); Tensor::from_uop(ctx, __u) }
    }

    pub fn prod(&self, ctx: &mut UOpCtx, axis: Option<&[usize]>, keepdim: bool, dtype: Option<DTypeId>) -> Tensor {
        { let __u = reduce::prod(ctx, self.uop.clone(), axis, keepdim, dtype); Tensor::from_uop(ctx, __u) }
    }

    pub fn max(&self, ctx: &mut UOpCtx, axis: Option<&[usize]>, keepdim: bool) -> Tensor {
        { let __u = reduce::max(ctx, self.uop.clone(), axis, keepdim); Tensor::from_uop(ctx, __u) }
    }

    pub fn mean(&self, ctx: &mut UOpCtx, axis: Option<&[usize]>, keepdim: bool) -> Tensor {
        { let __u = reduce::mean(ctx, self.uop.clone(), axis, keepdim); Tensor::from_uop(ctx, __u) }
    }

    pub fn var(&self, ctx: &mut UOpCtx, axis: Option<&[usize]>, keepdim: bool, correction: i64) -> Tensor {
        { let __u = reduce::var(ctx, self.uop.clone(), axis, keepdim, correction); Tensor::from_uop(ctx, __u) }
    }

    pub fn cumsum(&self, ctx: &mut UOpCtx, axis: i64) -> Tensor {
        { let __u = reduce::cumsum(ctx, self.uop.clone(), axis); Tensor::from_uop(ctx, __u) }
    }

    // *** nn-layer operations ***

    /// `Tensor.sub(other)`: `self - other` (composed via add/neg).
    pub fn sub(&self, ctx: &mut UOpCtx, other: impl Into<TensorAddArg>) -> Tensor {
        match other.into() {
            TensorAddArg::T(o) => { let __u = elementwise::sub(ctx, self.uop.clone(), o.uop); Tensor::from_uop(ctx, __u) },
            TensorAddArg::C(c) => { let __u = elementwise::sub_c(ctx, self.uop.clone(), c); Tensor::from_uop(ctx, __u) },
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
        { let __u = elementwise::square(ctx, self.uop.clone()); Tensor::from_uop(ctx, __u) }
    }

    /// `Tensor.sqrt()`: square root.
    pub fn sqrt(&self, ctx: &mut UOpCtx) -> Tensor {
        { let __u = elementwise::sqrt(ctx, self.uop.clone()); Tensor::from_uop(ctx, __u) }
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
            TensorAddArg::T(o) => { let __u = elementwise::maximum(ctx, self.uop.clone(), o.uop); Tensor::from_uop(ctx, __u) },
            TensorAddArg::C(c) => { let __u = elementwise::binop_c(ctx, Ops::Max, self.uop.clone(), c); Tensor::from_uop(ctx, __u) },
        }
    }

    /// `Tensor.minimum(other)`: elementwise min.
    pub fn minimum(&self, ctx: &mut UOpCtx, other: impl Into<TensorAddArg>) -> Tensor {
        match other.into() {
            TensorAddArg::T(o) => { let __u = elementwise::minimum(ctx, self.uop.clone(), o.uop); Tensor::from_uop(ctx, __u) },
            // No Ops::MIN — use min(a, b) == a + b - max(a, b).
            TensorAddArg::C(c) => {
                let sum = self.add(ctx, TensorAddArg::C(c.clone()));
                let mx = self.maximum(ctx, TensorAddArg::C(c));
                sum.sub(ctx, mx)
            }
        }
    }

    /// `Tensor.__gt__` — elementwise `self > other` (tinygrad `gt`).
    pub fn gt(&self, ctx: &mut UOpCtx, other: impl Into<TensorAddArg>) -> Tensor {
        let o = self.cmp_operand(ctx, other);
        { let __u = elementwise::gt(ctx, self.uop.clone(), o); Tensor::from_uop(ctx, __u) }
    }

    /// `Tensor.__lt__` — elementwise `self < other`.
    pub fn lt(&self, ctx: &mut UOpCtx, other: impl Into<TensorAddArg>) -> Tensor {
        let o = self.cmp_operand(ctx, other);
        { let __u = elementwise::lt(ctx, self.uop.clone(), o); Tensor::from_uop(ctx, __u) }
    }

    /// `Tensor.__ne__` — elementwise `self != other`.
    pub fn ne(&self, ctx: &mut UOpCtx, other: impl Into<TensorAddArg>) -> Tensor {
        let o = self.cmp_operand(ctx, other);
        { let __u = elementwise::ne(ctx, self.uop.clone(), o); Tensor::from_uop(ctx, __u) }
    }

    /// `cond.where(x, y)` — elementwise select using `self` as the condition
    /// (tinygrad `Tensor.where`). Scalars are promoted to `self`'s shape.
    pub fn where_(&self, ctx: &mut UOpCtx, x: impl Into<TensorAddArg>, y: impl Into<TensorAddArg>) -> Tensor {
        let xu = self.cmp_operand(ctx, x);
        let yu = self.cmp_operand(ctx, y);
        { let __u = elementwise::where_(ctx, self.uop.clone(), xu, yu); Tensor::from_uop(ctx, __u) }
    }

    /// UOp for a binop operand: tensor passes through, python scalar becomes a
    /// same-shaped const so the binop broadcasts.
    fn cmp_operand(&self, ctx: &mut UOpCtx, other: impl Into<TensorAddArg>) -> UopId {
        match other.into() {
            TensorAddArg::T(o) => o.uop,
            TensorAddArg::C(c) => ctx.const_like_dt(self.uop.clone(), c, None),
        }
    }

    /// `Tensor.reciprocal()`: 1 / self.
    pub fn reciprocal(&self, ctx: &mut UOpCtx) -> Tensor {
        { let __u = elementwise::reciprocal(ctx, self.uop.clone()); Tensor::from_uop(ctx, __u) }
    }

    /// `Tensor.log()`: natural log.
    pub fn log(&self, ctx: &mut UOpCtx) -> Tensor {
        { let __u = elementwise::log(ctx, self.uop.clone()); Tensor::from_uop(ctx, __u) }
    }

    /// `Tensor.log2()`: log base 2.
    pub fn log2(self, ctx: &mut UOpCtx) -> Tensor {
        { let __u = elementwise::log2(ctx, self.uop); Tensor::from_uop(ctx, __u) }
    }

    /// `Tensor.exp()`: exponential.
    pub fn exp(&self, ctx: &mut UOpCtx) -> Tensor {
        { let __u = elementwise::exp(ctx, self.uop.clone()); Tensor::from_uop(ctx, __u) }
    }

    /// `Tensor.exp2()`: 2^self.
    pub fn exp2(self, ctx: &mut UOpCtx) -> Tensor {
        { let __u = elementwise::exp2(ctx, self.uop); Tensor::from_uop(ctx, __u) }
    }

    /// `Tensor.trunc()`: truncate toward zero.
    pub fn trunc(&self, ctx: &mut UOpCtx) -> Tensor {
        { let __u = elementwise::trunc(ctx, self.uop.clone()); Tensor::from_uop(ctx, __u) }
    }

    /// `Tensor.ceil()`: ceiling.
    pub fn ceil(&self, ctx: &mut UOpCtx) -> Tensor {
        { let __u = elementwise::ceil(ctx, self.uop.clone()); Tensor::from_uop(ctx, __u) }
    }

    /// `Tensor.floor()`: floor.
    pub fn floor(&self, ctx: &mut UOpCtx) -> Tensor {
        { let __u = elementwise::floor(ctx, self.uop.clone()); Tensor::from_uop(ctx, __u) }
    }

    /// `Tensor.sin()`: sine.
    pub fn sin(&self, ctx: &mut UOpCtx) -> Tensor {
        { let __u = elementwise::sin(ctx, self.uop.clone()); Tensor::from_uop(ctx, __u) }
    }

    /// `Tensor.tanh()`: hyperbolic tangent.
    pub fn tanh(&self, ctx: &mut UOpCtx) -> Tensor {
        { let __u = elementwise::tanh(ctx, self.uop.clone()); Tensor::from_uop(ctx, __u) }
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
        let x_last = <SInt as Clone>::clone(&x_shape[dx - 1]).as_i64().unwrap_or(0);
        let w_contract = <SInt as Clone>::clone(&w_shape[axis_w]).as_i64().unwrap_or(0);
        assert_eq!(x_last, w_contract, "Tensor::dot: shape mismatch {} vs {}", x_shape, w_shape);

        let rep = std::cmp::min(std::cmp::min(dx - 1, dw - 1), 1);
        // x -> (*x.shape[:-1], [1]*rep, x.shape[-1])
        let mut xr: Vec<SVal> = x_shape[..dx - 1].to_vec();
        xr.extend(std::iter::repeat_n(SVal::I(1), rep));
        xr.push(x_shape[dx - 1].clone());
        let x_r = self.reshape(ctx, xr);

        // w -> (*w.shape[:-2], [1]*rep, *w.shape[axis_w:]) then transpose(-1, axis_w - dw)
        let mut wr: Vec<SVal> = w_shape[..dw.saturating_sub(2)].to_vec();
        wr.extend(std::iter::repeat_n(SVal::I(1), rep));
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
    /// Scalar `stride`/`dilation` broadcast to every spatial dim.
    pub fn pool(&self, ctx: &mut UOpCtx, kernel_size: &[i64], stride: i64, dilation: i64) -> Tensor {
        let s: Vec<i64> = vec![stride; kernel_size.len()];
        let d: Vec<i64> = vec![dilation; kernel_size.len()];
        self.pool_vec(ctx, kernel_size, &s, &d)
    }

    /// `Tensor._pool(k, stride, dilation)` with per-dim `stride`/`dilation`.
    pub fn pool_vec(&self, ctx: &mut UOpCtx, kernel_size: &[i64], stride: &[i64], dilation: &[i64]) -> Tensor {
        { let __u = movement::pool(ctx, self.uop.clone(), kernel_size, stride, dilation); Tensor::from_uop(ctx, __u) }
    }

    /// `Tensor.div(other)`: elementwise division (a / b).
    pub fn div(&self, ctx: &mut UOpCtx, other: impl Into<TensorAddArg>) -> Tensor {
        match other.into() {
            TensorAddArg::T(o) => { let __u = elementwise::div(ctx, self.uop.clone(), o.uop, None); Tensor::from_uop(ctx, __u) },
            TensorAddArg::C(c) => { let __u = elementwise::div_c(ctx, self.uop.clone(), c, None); Tensor::from_uop(ctx, __u) },
        }
    }

    /// Tensor-level elementwise sugar for the oracle/schedule corpus.
    pub fn add(&self, ctx: &mut UOpCtx, other: impl Into<TensorAddArg>) -> Tensor {
        match other.into() {
            TensorAddArg::T(o) => { let __u = elementwise::binop(ctx, Ops::Add, self.uop.clone(), o.uop); Tensor::from_uop(ctx, __u) },
            TensorAddArg::C(c) => { let __u = elementwise::binop_c(ctx, Ops::Add, self.uop.clone(), c); Tensor::from_uop(ctx, __u) },
        }
    }

    pub fn mul(&self, ctx: &mut UOpCtx, other: impl Into<TensorAddArg>) -> Tensor {
        match other.into() {
            TensorAddArg::T(o) => { let __u = elementwise::binop(ctx, Ops::Mul, self.uop.clone(), o.uop); Tensor::from_uop(ctx, __u) },
            TensorAddArg::C(c) => { let __u = elementwise::binop_c(ctx, Ops::Mul, self.uop.clone(), c); Tensor::from_uop(ctx, __u) },
        }
    }

    pub fn neg(&self, ctx: &mut UOpCtx) -> Tensor {
        { let __u = elementwise::neg(ctx, self.uop.clone()); Tensor::from_uop(ctx, __u) }
    }

    /// `Tensor.contiguous()` — CONTIGUOUS node when not virtual.
    pub fn contiguous(&self, ctx: &mut UOpCtx) -> Tensor {
        let dt = self.dtype(ctx);
        { let __u = ctx.uop(Ops::Contiguous, Some(dt), vec![self.uop.clone()], Arg::None); Tensor::from_uop(ctx, __u) }
    }

    /// `Tensor.alu`: `u[0].alu(op, u[1:])` — a bare op ctor; promotion/broadcast happen
    /// in `dtype_from_uop` during construction.
    pub fn alu(&self, ctx: &mut UOpCtx, op: Ops, srcs: &[Tensor]) -> Tensor {
        self.apply_uop(ctx, srcs, |c, u| c.alu(op, u))
    }

    /// `Tensor.bitcast(dtype)` (`mixin/dtype.py:35`) — reinterpret the bits.
    pub fn bitcast(&self, ctx: &mut UOpCtx, dtype: DTypeId) -> Tensor {
        { let __u = ctx.bitcast(self.uop.clone(), dtype); Tensor::from_uop(ctx, __u) }
    }

    /// `bitwise_or` / `bitwise_and` / `bitwise_xor` — bare binops (`mixin/elementwise.py`).
    pub fn bitwise_or(&self, ctx: &mut UOpCtx, other: impl Into<TensorAddArg>) -> Tensor {
        match other.into() {
            TensorAddArg::T(o) => { let __u = elementwise::binop(ctx, Ops::Or, self.uop.clone(), o.uop); Tensor::from_uop(ctx, __u) },
            TensorAddArg::C(c) => { let __u = elementwise::binop_c(ctx, Ops::Or, self.uop.clone(), c); Tensor::from_uop(ctx, __u) },
        }
    }
    pub fn bitwise_and(&self, ctx: &mut UOpCtx, other: impl Into<TensorAddArg>) -> Tensor {
        match other.into() {
            TensorAddArg::T(o) => { let __u = elementwise::binop(ctx, Ops::And, self.uop.clone(), o.uop); Tensor::from_uop(ctx, __u) },
            TensorAddArg::C(c) => { let __u = elementwise::binop_c(ctx, Ops::And, self.uop.clone(), c); Tensor::from_uop(ctx, __u) },
        }
    }
    pub fn bitwise_xor(&self, ctx: &mut UOpCtx, other: impl Into<TensorAddArg>) -> Tensor {
        match other.into() {
            TensorAddArg::T(o) => { let __u = elementwise::binop(ctx, Ops::Xor, self.uop.clone(), o.uop); Tensor::from_uop(ctx, __u) },
            TensorAddArg::C(c) => { let __u = elementwise::binop_c(ctx, Ops::Xor, self.uop.clone(), c); Tensor::from_uop(ctx, __u) },
        }
    }

    /// `Tensor.shl(x)` / `Tensor.rshift(x)` — `self._binop(Ops.SHL/SHR, x)`.
    pub fn shl(&self, ctx: &mut UOpCtx, bits: i64) -> Tensor {
        let b = elementwise::ufix(ctx, ConstVal::Int(bits as i128));
        { let __u = ctx.shl(self.uop.clone(), b); Tensor::from_uop(ctx, __u) }
    }
    pub fn rshift(&self, ctx: &mut UOpCtx, bits: i64) -> Tensor {
        let b = elementwise::ufix(ctx, ConstVal::Int(bits as i128));
        { let __u = ctx.shr(self.uop.clone(), b); Tensor::from_uop(ctx, __u) }
    }

    /// `Tensor.cat(*others, dim=0)` (`mixin/op.py:734`), equal-shape fast path:
    /// `self.stack(*others, dim).flatten(dim, dim+1)`. The RNG path always
    /// concatenates same-shape operands along dim 0, so this covers it.
    pub fn cat(&self, ctx: &mut UOpCtx, others: &[Tensor]) -> Tensor {
        let uops: Vec<UopId> = std::iter::once(self.uop.clone()).chain(others.iter().map(|t| t.uop.clone())).collect();
        let st = { let __u = stack(ctx, uops); Tensor::from_uop(ctx, __u) };
        st.flatten(ctx, 0, 2)
    }

    /// `Tensor.threefry(seed)` (`mixin/elementwise.py:454`): `self.alu(Ops.THREEFRY, seed)`.
    pub fn threefry(&self, ctx: &mut UOpCtx, seed: &Tensor) -> Tensor {
        { let __u = ctx.uop(Ops::Threefry, Some(crate::dtype::UINT64), vec![self.uop.clone(), seed.uop.clone()], Arg::None); Tensor::from_uop(ctx, __u) }
    }
}

fn device_spec(spec: Option<&DeviceSpec>) -> Option<String> {
    match spec {
        Some(DeviceSpec::Single(s)) => Some(s.clone()),
        Some(DeviceSpec::Multi(v)) => v.first().cloned(),
        None => None,
    }
}

/// `UOp.device` (ops.py:847): placement of a node, walking movement/COPY and
/// falling through to the first src that has one. Multi-device ops (`MSELECT`/
/// `MSTACK`) collapse to their first device on this single-device port.
fn uop_device(ctx: &UOpCtx, u: &UopId) -> Option<String> {
    use Ops::*;
    match ctx.op_of(u) {
        Param | Buffer => match ctx.arg_of(u) {
            Arg::Param(p) => device_spec(p.device.as_ref()),
            _ => None,
        },
        After => ctx.srcs(u).first().and_then(|s| uop_device(ctx, s)),
        Copy => match ctx.arg_of(u) {
            Arg::Str(s) => Some(s.clone()),
            _ => ctx.srcs(u).first().and_then(|s| uop_device(ctx, s)),
        },
        Mselect | Mstack | Allreduce => ctx.srcs(u).first().and_then(|s| uop_device(ctx, s)),
        _ => ctx.srcs(u).iter().find_map(|s| uop_device(ctx, s)),
    }
}
