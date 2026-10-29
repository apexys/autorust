//! `tensor/` — the user-facing `Tensor` front-end (tinygrad `tensor.py` +
//! `mixin/`), plus `nn`, optimizers, autograd, RNG, indexing and state loading.
//!
//! A [`Tensor`] is a cheap `Clone` handle to an interned UOp plus a shared
//! realized-buffer cell. Graph-building methods take `&mut UOpCtx` (the
//! interning arena); reads and realize take a
//! [`Session`] as well. The `Tensor` API
//! mirrors tinygrad's method names and argument order.
//!
//! Module map:
//!
//! - [`mod@creation`], [`movement`], [`elementwise`], [`reduce`], [`op`] — the
//!   mixin ports (factories, movement, ALU, reductions, conv/pool).
//! - [`nn`] — layers (`Linear`, `Conv*`, norms, `Embedding`, `LSTMCell`) and the
//!   [`nn::Module`] parameter tree.
//! - [`optim`] — SGD/LARS/Adam/AdamW/LAMB.
//! - [`grad`] — reverse-mode autograd.
//! - [`rand`] — Threefry RNG (host and device-graph paths).
//! - [`ix`] — the `s![...]` indexing DSL.
//! - [`state`] — safetensors/state-dict loading and saving.
//! - [`callify`] — the v0.14 `CALL(FUNCTION(...))` transform.
//!
//! This is the stable, user-facing part of the crate; see the crate docs for
//! the full stability statement.

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
pub mod extra;
pub mod grad;
pub mod movement;
pub mod nn;
pub mod op;
pub mod optim;
pub mod rand;
pub mod state;
pub use state::State;
pub mod reduce;

use crate::dtype::ConstVal;

/// arg for Tensor.add/mul: another Tensor or a python scalar.
pub enum TensorAddArg {
    /// A tensor operand.
    T(Tensor),
    /// A Python scalar operand.
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
use crate::dtype::DTypeId;
use crate::dtype::ReadDtype;
use crate::engine::session::Session;
use crate::runtime::buffer::Buffer;
use crate::uop::arg::{Arg, DeviceSpec, EnumClass, EnumVal};
use crate::uop::ctx::UOpCtx;
use crate::uop::graph::UopId;
use crate::uop::ops::Ops;
use crate::uop::sint::{SInt, SVal};
use std::cell::{Ref, RefCell, RefMut};
use std::rc::Rc;

/// Shared owner of a [`UOpCtx`]. `Tensor`s created from the same `Ctx` can be
/// combined with operator syntax and without passing `&mut UOpCtx` at every
/// call site.
///
/// A `Ctx` is a cheap, cloneable handle (`Rc<RefCell<UOpCtx>>`); it is not
/// `Send`/`Sync`, matching the single-threaded front-end.
#[derive(Clone)]
pub struct Ctx {
    inner: Rc<RefCell<UOpCtx>>,
}

impl Ctx {
    /// Create a fresh arena.
    pub fn new() -> Self {
        Ctx {
            inner: UOpCtx::shared(),
        }
    }

    /// Borrow the arena for a read-only operation.
    pub fn borrow(&self) -> Ref<'_, UOpCtx> {
        self.inner.borrow()
    }

    /// Borrow the arena mutably for an operation.
    pub fn borrow_mut(&self) -> RefMut<'_, UOpCtx> {
        self.inner.borrow_mut()
    }

    /// Alias for [`Ctx::borrow_mut`] used by raw-context helper calls.
    pub fn raw(&self) -> RefMut<'_, UOpCtx> {
        self.inner.borrow_mut()
    }

    /// The underlying shared handle (used by engine/oracle glue).
    pub fn handle(&self) -> Rc<RefCell<UOpCtx>> {
        self.inner.clone()
    }

    /// Whether two handles refer to the same arena.
    pub fn same(&self, other: &Ctx) -> bool {
        Rc::ptr_eq(&self.inner, &other.inner)
    }

    /// Panic with a clear message when two tensors come from different arenas.
    pub fn ensure_same(&self, other: &Ctx) {
        assert!(
            self.same(other),
            "TGERR: tensors belong to different Ctx instances"
        );
    }
}

impl Default for Ctx {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for Ctx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Ctx({:p})", Rc::as_ptr(&self.inner))
    }
}

/// Anything that can lend a mutable [`UOpCtx`] to a constructor.
///
/// Implemented for [`Ctx`], `&Ctx`, `&mut Ctx`, and `&mut UOpCtx`. The last is
/// for internal/oracle paths; a bare `UOpCtx::new()` cannot back a `Tensor`
/// (use [`Ctx::new`]).
pub trait AsCtx {
    /// Run `f` with mutable access to the arena.
    fn with_ctx<T>(self, f: impl FnOnce(&mut UOpCtx) -> T) -> T;
}

impl AsCtx for Ctx {
    fn with_ctx<T>(self, f: impl FnOnce(&mut UOpCtx) -> T) -> T {
        f(&mut self.inner.borrow_mut())
    }
}

impl AsCtx for &Ctx {
    fn with_ctx<T>(self, f: impl FnOnce(&mut UOpCtx) -> T) -> T {
        f(&mut self.inner.borrow_mut())
    }
}

impl AsCtx for &mut Ctx {
    fn with_ctx<T>(self, f: impl FnOnce(&mut UOpCtx) -> T) -> T {
        f(&mut self.inner.borrow_mut())
    }
}

impl AsCtx for &mut UOpCtx {
    fn with_ctx<T>(self, f: impl FnOnce(&mut UOpCtx) -> T) -> T {
        f(self)
    }
}

impl AsCtx for &mut &mut UOpCtx {
    fn with_ctx<T>(self, f: impl FnOnce(&mut UOpCtx) -> T) -> T {
        f(self)
    }
}

impl AsCtx for &mut RefMut<'_, UOpCtx> {
    fn with_ctx<T>(self, f: impl FnOnce(&mut UOpCtx) -> T) -> T {
        f(self)
    }
}

/// Python `Tensor.__slots__ = ("uop",)`. In v0.14 a `Tensor` is a Python object
/// whose identity keeps its `UOp` (and, once realized, its buffer) alive. The Rust
/// port models that explicitly: a `Tensor` is a **`Clone`** handle (not `Copy`)
/// holding a strong [`UopId`] (`Rc<Node>`) plus an `Rc` realized-buffer cell
/// shared by all clones. `grad` is tracked in `UOpCtx` keyed by `UopId`. Whether a
/// tensor is a *trainable parameter* is a property of the `Module` tree
/// (see [`nn::Module`]), not of the tensor.
pub struct Tensor {
    /// The tensor's UOp, **shared by all clones**. tinygrad's `Tensor.uop()` is a
    /// mutable attribute: `_apply_map_to_tensors` (tensor.py:248) replaces it in
    /// place with the tensor's realized-storage UOp after a schedule, so every
    /// handle to the same tensor sees the realized form. Storing it in a shared
    /// cell mirrors that; a plain copied field would not.
    uop_cell: Rc<RefCell<UopId>>,
    /// Realized storage, shared by clones and reused across reads.
    buffer: Rc<RefCell<Option<Buffer>>>,
    /// The arena this tensor belongs to.
    ctx: Ctx,
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        Tensor {
            uop_cell: self.uop_cell.clone(),
            buffer: self.buffer.clone(),
            ctx: self.ctx.clone(),
        }
    }
}

// Tensor identity is its uop (the buffer is bookkeeping).
impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.uop() == other.uop()
    }
}
impl Eq for Tensor {}
impl std::hash::Hash for Tensor {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.uop().hash(state);
    }
}
impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor({:?})", self.uop())
    }
}

impl Tensor {
    /// Wrap an engine uop as a `Tensor` handle. The `UopId` is a strong handle, so
    /// no extra pinning is needed; `ctx` is accepted to keep the construction
    /// uniform with the other builders.
    pub fn from_uop(ctx: &UOpCtx, uop: UopId) -> Tensor {
        let handle = ctx
            .handle()
            .expect("TGERR: Tensor requires a shared context; construct it with Ctx::new()/UOpCtx::shared()");
        Tensor {
            uop_cell: Rc::new(RefCell::new(uop)),
            buffer: Rc::new(RefCell::new(None)),
            ctx: Ctx { inner: handle },
        }
    }

    /// Wrap a uop with an explicit [`Ctx`] handle (used by constructors that
    /// already own one).
    pub fn from_uop_ctx(ctx: &Ctx, uop: UopId) -> Tensor {
        Tensor {
            uop_cell: Rc::new(RefCell::new(uop)),
            buffer: Rc::new(RefCell::new(None)),
            ctx: ctx.clone(),
        }
    }

    /// The arena this tensor belongs to.
    #[inline]
    pub fn ctx(&self) -> &Ctx {
        &self.ctx
    }

    /// tinygrad `Tensor.uop()` — the current UOp, which after a realize is the
    /// tensor's realized-storage UOp rather than the original lazy expression.
    #[inline]
    pub fn uop(&self) -> UopId {
        self.uop_cell.borrow().clone()
    }

    /// tinygrad `_apply_map_to_tensors` (tensor.py:248): replace this tensor's UOp
    /// in place (with, e.g., its realized storage form).
    pub fn set_uop(&self, u: UopId) {
        *self.uop_cell.borrow_mut() = u;
    }

    /// The realized storage, if any (shared with clones).
    pub fn realized_buffer(&self) -> Option<Buffer> {
        self.buffer.borrow().clone()
    }

    /// Record realized storage for this tensor (and its clones).
    pub fn set_realized_buffer(&self, b: Buffer) {
        *self.buffer.borrow_mut() = Some(b);
    }

    /// `Tensor._apply_uop`: map a UOp-level fxn over `(self, *x).uop()` and re-wrap.
    pub fn apply_uop(
        &self,
        srcs: &[Tensor],
        f: impl FnOnce(&mut UOpCtx, Vec<UopId>) -> UopId,
    ) -> Tensor {
        self.apply_uop_ctx(&mut self.ctx.borrow_mut(), srcs, f)
    }
    #[doc(hidden)]
    pub fn apply_uop_ctx(
        &self,
        ctx: &mut UOpCtx,
        srcs: &[Tensor],
        f: impl FnOnce(&mut UOpCtx, Vec<UopId>) -> UopId,
    ) -> Tensor {
        let uops: Vec<UopId> = std::iter::once(self.uop().clone())
            .chain(srcs.iter().map(|t| t.uop().clone()))
            .collect();
        let u = f(ctx, uops);
        Tensor::from_uop(ctx, u)
    }

    /// The autograd gradient attached to this tensor's `uop` in `ctx` (see
    /// [`UOpCtx::grad`]). `None` until `backward`/`gradient` populates it.
    pub fn grad(&self) -> Option<Tensor> {
        self.grad_ctx(&self.ctx.borrow())
    }
    #[doc(hidden)]
    pub fn grad_ctx(&self, ctx: &UOpCtx) -> Option<Tensor> {
        ctx.grad(&self.uop()).map(|g| Tensor::from_uop(ctx, g))
    }

    /// `Tensor.const`: `Tensor(UOp.const(b, dtype))`.
    pub fn const_(ctx: impl AsCtx, b: ConstVal, dtype: Option<DTypeId>) -> Tensor {
        ctx.with_ctx(|ctx| {
            let u = ctx.const_py(b, dtype);
            Tensor::from_uop(ctx, u)
        })
    }

    /// `Tensor(Variable(...))`-style: 0-d BUFFER in the ALU addrspace with a value range.
    pub fn variable(
        ctx: impl AsCtx,
        name: &str,
        vmin: ConstVal,
        vmax: ConstVal,
        dtype: DTypeId,
        param: bool,
    ) -> Tensor {
        ctx.with_ctx(|ctx| {
            let u = ctx.variable(name, vmin, vmax, dtype, param);
            Tensor::from_uop(ctx, u)
        })
    }

    /// `Tensor.shape` -> the public `Shape` wrapper (list-like ergonomics).
    pub fn shape(&self) -> Shape {
        self.shape_ctx(&mut self.ctx.borrow_mut())
    }
    #[doc(hidden)]
    pub fn shape_ctx(&self, ctx: &mut UOpCtx) -> Shape {
        Shape::new((*ctx.shape(&self.uop()).unwrap().unwrap_or_default()).clone())
    }
    /// `Tensor.dtype` — this tensor's dtype.
    pub fn dtype(&self) -> DTypeId {
        self.dtype_ctx(&self.ctx.borrow())
    }
    #[doc(hidden)]
    pub fn dtype_ctx(&self, ctx: &UOpCtx) -> DTypeId {
        ctx.dt_of(&self.uop())
    }

    /// `Tensor.device` / `UOp.device` (tensor.py / ops.py:847): the device string
    /// backing this tensor's storage, or `None` when it has no placement yet
    /// (e.g. a computed expression). Walks movement/COPY/multi-device ops like
    /// the Python property.
    pub fn device(&self) -> Option<String> {
        self.device_ctx(&self.ctx.borrow())
    }
    #[doc(hidden)]
    pub fn device_ctx(&self, ctx: &UOpCtx) -> Option<String> {
        uop_device(ctx, &self.uop())
    }

    /// `Tensor.cast(dtype)`: in-graph cast to another dtype (a no-op when equal).
    pub fn cast(&self, dtype: DTypeId) -> Tensor {
        self.cast_ctx(&mut self.ctx.borrow_mut(), dtype)
    }
    #[doc(hidden)]
    pub fn cast_ctx(&self, ctx: &mut UOpCtx, dtype: DTypeId) -> Tensor {
        {
            let __u = ctx.cast(self.uop().clone(), dtype);
            Tensor::from_uop(ctx, __u)
        }
    }

    /// Trigger the compute needed to read this tensor and return its raw storage
    /// bytes at its own dtype (python `_buffer()` copyout). For typed,
    /// cast-aware read-out use [`Tensor::to_list`] / [`Tensor::item`]. The
    /// realized buffer is cached on the tensor (shared by clones), so a repeat
    /// read does not re-realize.
    pub fn to_bytes(&self, sess: &mut Session) -> Vec<u8> {
        self.to_bytes_ctx(&mut self.ctx.borrow_mut(), sess)
    }
    #[doc(hidden)]
    pub fn to_bytes_ctx(&self, ctx: &mut UOpCtx, sess: &mut Session) -> Vec<u8> {
        if let Some(b) = self.realized_buffer() {
            return crate::engine::realize::buffer_bytes(&b);
        }
        let (b, leaf) = sess.realize_tensor(ctx, self.uop());
        let bytes = crate::engine::realize::buffer_bytes(&b);
        // tinygrad `_apply_map_to_tensors`: the tensor's uop becomes its realized
        // storage form, so later graphs read the buffer instead of re-running it.
        self.set_uop(leaf);
        self.set_realized_buffer(b);
        bytes
    }

    /// [`Tensor::to_bytes`] with a throwaway [`Session`] (one-shot reads).
    pub fn to_bytes_fresh(&self) -> Vec<u8> {
        self.to_bytes_fresh_ctx(&mut self.ctx.borrow_mut())
    }
    #[doc(hidden)]
    pub fn to_bytes_fresh_ctx(&self, ctx: &mut UOpCtx) -> Vec<u8> {
        self.to_bytes_ctx(ctx, &mut Session::new())
    }

    /// The write effect for `self = value` (tinygrad tensor.py:455):
    /// `self.uop().after(self.uop().store(value.uop()))`. STORE is the write effect
    /// (void), AFTER wraps the view for correct shape/ranging and orders any
    /// read of `self` in the same schedule *before* the store.
    ///
    /// Returned as a `Tensor` so a caller can co-realize it with other roots
    /// (`Optim::schedule_step`), folding the parameter write into the same
    /// schedule as the loss instead of a separate copy pass.
    pub fn assign_effect(&self, value: &Tensor) -> Tensor {
        self.assign_effect_ctx(&mut self.ctx.borrow_mut(), value)
    }
    #[doc(hidden)]
    pub fn assign_effect_ctx(&self, ctx: &mut UOpCtx, value: &Tensor) -> Tensor {
        assert_eq!(
            self.dtype_ctx(ctx),
            value.dtype_ctx(ctx),
            "TGERR: assign dtype mismatch"
        );
        let st = crate::tensor::store(ctx, self.uop().clone(), value.uop().clone(), None);
        let aft = crate::tensor::after(ctx, self.uop().clone(), vec![st]);
        Tensor::from_uop(ctx, aft)
    }

    /// tinygrad's Tensor object owns its realized buffer, so a parameter's
    /// storage is not garbage-collected between steps. Here the shared buffer
    /// cell is that owner: pin *this* tensor's own storage by buffer identity
    /// (`buffer_of(self.uop())`) so `BufTable::prune` cannot drop it before the
    /// next read — without this, a schedule that writes a parameter through a
    /// fused STORE would see its buffer freed and re-zeroed on the next realize.
    pub fn pin_buffer(&self, sess: &mut Session) {
        self.pin_buffer_ctx(&mut self.ctx.borrow_mut(), sess)
    }
    #[doc(hidden)]
    pub fn pin_buffer_ctx(&self, ctx: &mut UOpCtx, sess: &mut Session) {
        let dst = crate::runtime::buffer::buffer_of(ctx, &mut sess.buffers, self.uop().clone());
        self.set_realized_buffer(dst);
    }

    /// `Tensor.assign` (tensor.py:436): write `value` into *this* tensor's
    /// storage in place (a fused STORE+AFTER, not a copy), leaving `self.uop()`
    /// unchanged.    ///
    /// This is what keeps a persistent parameter a stable leaf across optimizer
    /// steps: `w - lr*grad` is built as a fresh subgraph, stored back into `w`'s
    /// buffer by the scheduler, then dropped. Without it, rebinding
    /// `w = w - lr*grad` chains each step into the previous one and the graph
    /// grows without bound.
    pub fn assign(&self, sess: &mut Session, value: &Tensor) {
        self.assign_ctx(&mut self.ctx.borrow_mut(), sess, value)
    }
    #[doc(hidden)]
    pub fn assign_ctx(&self, ctx: &mut UOpCtx, sess: &mut Session, value: &Tensor) {
        let effect = self.assign_effect_ctx(ctx, value);
        // Realize the STORE. If the caller is co-realizing a larger schedule,
        // `schedule_step` uses `assign_effect` directly so the write fuses with
        // the loss; this standalone path is for one-off writes (e.g. running
        // stats, optimizer moments that are read again immediately).
        let _ = sess.realize_tensor(ctx, effect.uop().clone());
        self.pin_buffer_ctx(ctx, sess);
    }

    /// Replace this tensor's graph node and realized storage with `other`
    /// (tinygrad `Tensor.replace`, used by `load_state_dict`). Clones of this
    /// tensor see the new storage immediately; the optimizer's references stay
    /// valid.
    pub fn replace(&self, other: &Tensor) {
        self.ctx().ensure_same(other.ctx());
        self.set_uop(other.uop());
        *self.buffer.borrow_mut() = other.realized_buffer();
    }

    /// `Tensor.to(device)`: copy this tensor's storage to `device` (a no-op view
    /// when it is already there). Tinygrad's `.to` is the user-facing device move.
    pub fn to(&self, device: &str) -> Tensor {
        let dev = creation::canonicalize_device(Some(device));
        let mut ctx = self.ctx.borrow_mut();
        let u = creation::copy_to_device(&mut ctx, self.uop(), &dev);
        Tensor::from_uop(&ctx, u)
    }

    /// Turn a computed tensor (e.g. a random-initialized parameter) into a
    /// buffer-backed leaf with the same shape/dtype, realizing the current value
    /// into its storage (`Node.initial_data`). tinygrad parameters are realized
    /// buffers, which is what lets [`Tensor::assign`] update them in place;
    /// a raw init *expression* would be recomputed (and the update lost) on every
    /// realize.
    pub fn to_param(&self) -> Tensor {
        self.to_param_ctx(&mut self.ctx.borrow_mut())
    }
    #[doc(hidden)]
    pub fn to_param_ctx(&self, ctx: &mut UOpCtx) -> Tensor {
        let shape = self.shape_ctx(ctx);
        let dt = self.dtype_ctx(ctx);
        // Realize the init in a throwaway session (it may be on a different
        // device than the training buffers) and stage the bytes on the leaf.
        let bytes = self.to_bytes_fresh_ctx(ctx);
        let dev = self
            .realized_buffer()
            .map(|b| b.device().to_string())
            .unwrap_or_else(|| "PYTHON".to_string());
        let u = crate::tensor::creation::empty(ctx, &shape, dt, &dev);
        crate::tensor::creation::register_initial(ctx, u.clone(), bytes);
        Tensor::from_uop(ctx, u)
    }

    /// Freeze an already-realized tensor into a buffer-backed leaf that *shares*
    /// its computed bytes (no recomputation). Used by the optimizer to realize a
    /// batch of gradient subgraphs once — as a single shared schedule — and then
    /// let the per-parameter state updates read a plain buffer instead of
    /// re-realizing the whole backward graph.
    pub fn realize_to_leaf(&self) -> Tensor {
        self.realize_to_leaf_ctx(&mut self.ctx.borrow_mut())
    }
    #[doc(hidden)]
    pub fn realize_to_leaf_ctx(&self, ctx: &mut UOpCtx) -> Tensor {
        let b = self
            .realized_buffer()
            .unwrap_or_else(|| panic!("TGERR: realize_to_leaf requires a realized tensor"));
        let bytes = crate::engine::realize::buffer_bytes(&b);
        let shape = self.shape_ctx(ctx);
        let dt = self.dtype_ctx(ctx);
        let dev = b.device().to_string();
        let u = crate::tensor::creation::empty(ctx, &shape, dt, &dev);
        crate::tensor::creation::register_initial(ctx, u.clone(), bytes);
        Tensor::from_uop(ctx, u)
    }

    /// `Tensor.realize`: trigger the computation needed to materialize this tensor
    /// and return `&self`.
    pub fn realize(&self, sess: &mut Session) -> &Tensor {
        self.realize_ctx(&mut self.ctx.borrow_mut(), sess)
    }
    #[doc(hidden)]
    pub fn realize_ctx(&self, ctx: &mut UOpCtx, sess: &mut Session) -> &Tensor {
        let _ = self.to_bytes_ctx(ctx, sess);
        self
    }

    /// [`Tensor::realize`] with a throwaway [`Session`].
    pub fn realize_fresh(&self) -> &Tensor {
        self.realize_fresh_ctx(&mut self.ctx.borrow_mut())
    }
    #[doc(hidden)]
    pub fn realize_fresh_ctx(&self, ctx: &mut UOpCtx) -> &Tensor {
        let _ = self.to_bytes_fresh_ctx(ctx);
        self
    }

    /// `Tensor.realize(*tensors)` (tensor.py) — co-realize all listed tensors in
    /// a **single** schedule, caching each one's realized buffer. This is the
    /// tinygrad co-realization the optimizer relies on to fold the loss and its
    /// updates into one schedule (`Optimizer.schedule_step`).
    pub fn realize_many<'a>(
        ctx: &mut UOpCtx,
        sess: &mut Session,
        tensors: impl IntoIterator<Item = &'a Tensor>,
    ) -> Vec<Buffer> {
        let ts: Vec<&Tensor> = tensors.into_iter().collect();
        let uops: Vec<UopId> = ts.iter().map(|t| t.uop()).collect();
        let (bufs, leaves) = sess.realize_many(ctx, &uops);
        for ((t, b), leaf) in ts.iter().zip(bufs.iter()).zip(leaves.iter()) {
            t.set_uop(leaf.clone());
            t.set_realized_buffer(b.clone());
        }
        bufs
    }

    /// Reserve the graph cast that reads this tensor as `T::DTYPE`, asserting
    /// weak dtypes are realized first.
    fn read_target<T: ReadDtype>(&self, ctx: &mut UOpCtx) -> UopId {
        let target = <T as ReadDtype>::DTYPE;
        let src = self.dtype_ctx(ctx);
        assert!(
            !crate::dtype::is_weak(src),
            "TGERR: cannot read a weak-dtype tensor ({src:?}) into {target:?}; realize it first"
        );
        assert!(
            !crate::dtype::is_weak(target),
            "TGERR: cannot read tensor into weak dtype {target:?}"
        );
        ctx.cast(self.uop().clone(), target)
    }

    /// Reinterpret little-endian bytes as `[T]` (no `f64` round-trip).
    fn decode<T: ReadDtype>(bytes: Vec<u8>) -> Vec<T> {
        let isz = <T as ReadDtype>::DTYPE.itemsize();
        assert_eq!(
            bytes.len() % isz,
            0,
            "TGERR: byte length {} not a multiple of itemsize {isz}",
            bytes.len()
        );
        bytes.chunks_exact(isz).map(T::from_bytes_le).collect()
    }

    /// Read the tensor out as `Vec<T>`: cast to `T::DTYPE` in the graph (a no-op
    /// when already that dtype), realize, and reinterpret the little-endian bytes
    /// — no `f64` round-trip. Panics if the tensor cannot be cast to `T::DTYPE`.
    pub fn to_list<T: ReadDtype>(&self, sess: &mut Session) -> Vec<T> {
        self.to_list_ctx(&mut self.ctx.borrow_mut(), sess)
    }
    #[doc(hidden)]
    pub fn to_list_ctx<T: ReadDtype>(&self, ctx: &mut UOpCtx, sess: &mut Session) -> Vec<T> {
        if <T as ReadDtype>::DTYPE == self.dtype_ctx(ctx) {
            return Self::decode::<T>(self.to_bytes_ctx(ctx, sess));
        }
        let uop = self.read_target::<T>(ctx);
        Self::decode::<T>(sess.tensor_bytes(ctx, uop))
    }

    /// [`Tensor::to_list`] with a throwaway [`Session`] (one-shot reads).
    pub fn to_list_fresh<T: ReadDtype>(&self) -> Vec<T> {
        self.to_list_fresh_ctx(&mut self.ctx.borrow_mut())
    }
    #[doc(hidden)]
    pub fn to_list_fresh_ctx<T: ReadDtype>(&self, ctx: &mut UOpCtx) -> Vec<T> {
        self.to_list_ctx(ctx, &mut Session::new())
    }

    /// Read a single-element tensor out as `T` (python `item()`); asserts `numel()==1`.
    pub fn item<T: ReadDtype>(&self, sess: &mut Session) -> T {
        self.item_ctx(&mut self.ctx.borrow_mut(), sess)
    }
    #[doc(hidden)]
    pub fn item_ctx<T: ReadDtype>(&self, ctx: &mut UOpCtx, sess: &mut Session) -> T {
        let n = self.shape_ctx(ctx).numel_int().unwrap_or(0);
        assert_eq!(
            n, 1,
            "TGERR: item() requires exactly one element, tensor has {n}"
        );
        self.to_list_ctx::<T>(ctx, sess)
            .into_iter()
            .next()
            .expect("TGERR: item() on empty tensor")
    }

    /// [`Tensor::item`] with a throwaway [`Session`].
    pub fn item_fresh<T: ReadDtype>(&self) -> T {
        self.item_fresh_ctx(&mut self.ctx.borrow_mut())
    }
    #[doc(hidden)]
    pub fn item_fresh_ctx<T: ReadDtype>(&self, ctx: &mut UOpCtx) -> T {
        self.item_ctx(ctx, &mut Session::new())
    }

    /// `Tensor.reshape`: public wrapper taking any `ShapeArg`.
    pub fn reshape(&self, shape: impl ShapeArg) -> Tensor {
        self.reshape_ctx(&mut self.ctx.borrow_mut(), shape)
    }
    #[doc(hidden)]
    pub fn reshape_ctx(&self, ctx: &mut UOpCtx, shape: impl ShapeArg) -> Tensor {
        {
            let __u = movement::reshape(ctx, self.uop().clone(), &shape.into_shape_vec());
            Tensor::from_uop(ctx, __u)
        }
    }

    /// `Tensor.expand`.
    pub fn expand(&self, shape: impl ShapeArg) -> Tensor {
        self.expand_ctx(&mut self.ctx.borrow_mut(), shape)
    }
    #[doc(hidden)]
    pub fn expand_ctx(&self, ctx: &mut UOpCtx, shape: impl ShapeArg) -> Tensor {
        {
            let __u = movement::expand(ctx, self.uop().clone(), &shape.into_shape_vec());
            Tensor::from_uop(ctx, __u)
        }
    }

    /// `Tensor.unsqueeze(dim)`.
    pub fn unsqueeze(&self, dim: i64) -> Tensor {
        self.unsqueeze_ctx(&mut self.ctx.borrow_mut(), dim)
    }
    #[doc(hidden)]
    pub fn unsqueeze_ctx(&self, ctx: &mut UOpCtx, dim: i64) -> Tensor {
        {
            let __u = movement::unsqueeze(ctx, self.uop().clone(), dim);
            Tensor::from_uop(ctx, __u)
        }
    }

    /// `Tensor.pad(arg)`: (lo, hi) per dim.
    pub fn pad(&self, arg: &[(i64, i64)]) -> Tensor {
        self.pad_ctx(&mut self.ctx.borrow_mut(), arg)
    }
    #[doc(hidden)]
    pub fn pad_ctx(&self, ctx: &mut UOpCtx, arg: &[(i64, i64)]) -> Tensor {
        let pairs: Vec<(SVal, SVal)> = arg
            .iter()
            .map(|(a, b)| (SVal::I(*a as i128), SVal::I(*b as i128)))
            .collect();
        {
            let __u = movement::pad(ctx, self.uop().clone(), &pairs);
            Tensor::from_uop(ctx, __u)
        }
    }

    /// `Tensor.shrink(arg)`: explicit `(start, end)` bounds per dim.
    pub fn shrink(&self, arg: &[(i64, i64)]) -> Tensor {
        self.shrink_ctx(&mut self.ctx.borrow_mut(), arg)
    }
    #[doc(hidden)]
    pub fn shrink_ctx(&self, ctx: &mut UOpCtx, arg: &[(i64, i64)]) -> Tensor {
        let pairs: Vec<(SVal, SVal)> = arg
            .iter()
            .map(|(a, b)| (SVal::I(*a as i128), SVal::I(*b as i128)))
            .collect();
        {
            let __u = movement::shrink(ctx, self.uop().clone(), &pairs);
            Tensor::from_uop(ctx, __u)
        }
    }

    /// `Tensor.permute(order)`.
    pub fn permute(&self, order: &[i64]) -> Tensor {
        self.permute_ctx(&mut self.ctx.borrow_mut(), order)
    }
    #[doc(hidden)]
    pub fn permute_ctx(&self, ctx: &mut UOpCtx, order: &[i64]) -> Tensor {
        {
            let __u = movement::permute(ctx, self.uop().clone(), order);
            Tensor::from_uop(ctx, __u)
        }
    }

    /// `Tensor.flip(axis)`.
    pub fn flip(&self, axis: &[usize]) -> Tensor {
        self.flip_ctx(&mut self.ctx.borrow_mut(), axis)
    }
    #[doc(hidden)]
    pub fn flip_ctx(&self, ctx: &mut UOpCtx, axis: &[usize]) -> Tensor {
        {
            let __u = movement::flip(ctx, self.uop().clone(), axis);
            Tensor::from_uop(ctx, __u)
        }
    }

    /// `Tensor.transpose(dim0, dim1)`.
    pub fn transpose(&self, dim0: i64, dim1: i64) -> Tensor {
        self.transpose_ctx(&mut self.ctx.borrow_mut(), dim0, dim1)
    }
    #[doc(hidden)]
    pub fn transpose_ctx(&self, ctx: &mut UOpCtx, dim0: i64, dim1: i64) -> Tensor {
        {
            let __u = movement::transpose(ctx, self.uop().clone(), dim0, dim1);
            Tensor::from_uop(ctx, __u)
        }
    }

    /// `Tensor.squeeze(dim=None)`.
    pub fn squeeze(&self, dim: Option<i64>) -> Tensor {
        self.squeeze_ctx(&mut self.ctx.borrow_mut(), dim)
    }
    #[doc(hidden)]
    pub fn squeeze_ctx(&self, ctx: &mut UOpCtx, dim: Option<i64>) -> Tensor {
        {
            let __u = movement::squeeze(ctx, self.uop().clone(), dim);
            Tensor::from_uop(ctx, __u)
        }
    }

    /// `Tensor.flatten(start_dim, end_dim)`.
    pub fn flatten(&self, start_dim: i64, end_dim: i64) -> Tensor {
        self.flatten_ctx(&mut self.ctx.borrow_mut(), start_dim, end_dim)
    }
    #[doc(hidden)]
    pub fn flatten_ctx(&self, ctx: &mut UOpCtx, start_dim: i64, end_dim: i64) -> Tensor {
        {
            let __u = movement::flatten(ctx, self.uop().clone(), start_dim, end_dim);
            Tensor::from_uop(ctx, __u)
        }
    }

    /// `MovementMixin.split` (movement.py:424-448): split along `dim` into pieces
    /// of the given sizes. `dim` is resolved Python-style; a size list must sum
    /// exactly to the resolved dimension length.
    pub fn split(&self, sizes: &[i64], dim: i64) -> Vec<Tensor> {
        self.split_ctx(&mut self.ctx.borrow_mut(), sizes, dim)
    }
    #[doc(hidden)]
    pub fn split_ctx(&self, ctx: &mut UOpCtx, sizes: &[i64], dim: i64) -> Vec<Tensor> {
        let self_shape: Vec<SVal> = self.shape_ctx(ctx).into_vec();
        let ndim = self_shape.len();
        let d = movement::resolve_dim(ndim, dim, false);
        let dim_sz = self_shape[d].clone().as_i64().unwrap_or(0);
        assert_eq!(
            sizes.iter().sum::<i64>(),
            dim_sz,
            "TGERR: expect sizes to sum exactly to {dim_sz}"
        );
        let mut out = Vec::with_capacity(sizes.len());
        let mut acc = 0i64;
        for &sz in sizes {
            let mut arg: Vec<(i64, i64)> = Vec::with_capacity(ndim);
            for (j, s) in self_shape.iter().enumerate() {
                if j == d {
                    arg.push((acc, acc + sz));
                } else {
                    arg.push((0, s.clone().as_i64().unwrap_or(0)));
                }
            }
            out.push(self.shrink_ctx(ctx, &arg));
            acc += sz;
        }
        out
    }

    /// `MovementMixin.chunk` (movement.py:450-476): split into `chunks` pieces
    /// along `dim`. All pieces are the same size except the last (which may be
    /// smaller); fewer than `chunks` pieces are returned when `dim` is short.
    pub fn chunk(&self, chunks: i64, dim: i64) -> Vec<Tensor> {
        self.chunk_ctx(&mut self.ctx.borrow_mut(), chunks, dim)
    }
    #[doc(hidden)]
    pub fn chunk_ctx(&self, ctx: &mut UOpCtx, chunks: i64, dim: i64) -> Vec<Tensor> {
        let self_shape: Vec<SVal> = self.shape_ctx(ctx).into_vec();
        let ndim = self_shape.len();
        let d = movement::resolve_dim(ndim, dim, false);
        let dim_sz = self_shape[d].clone().as_i64().unwrap_or(0);
        assert!(
            chunks > 0,
            "TGERR: expect chunks to be greater than 0, got: {chunks}"
        );
        let sizes: Vec<i64> = if dim_sz == 0 {
            vec![0; chunks as usize]
        } else {
            let c = (dim_sz + chunks - 1) / chunks;
            (0..dim_sz)
                .step_by(c.max(1) as usize)
                .map(|i| c.min(dim_sz - i))
                .collect()
        };
        self.split_ctx(ctx, &sizes, dim)
    }

    /// `Tensor.unflatten(dim, sizes)` (movement.py:362-378).
    pub fn unflatten(&self, dim: i64, sizes: &[i64]) -> Tensor {
        self.unflatten_ctx(&mut self.ctx.borrow_mut(), dim, sizes)
    }
    #[doc(hidden)]
    pub fn unflatten_ctx(&self, ctx: &mut UOpCtx, dim: i64, sizes: &[i64]) -> Tensor {
        {
            let __u = movement::unflatten(ctx, self.uop().clone(), dim, sizes);
            Tensor::from_uop(ctx, __u)
        }
    }

    /// `Tensor.shrink_to(shape)` (movement.py:277-279): shrink each dim to the
    /// given size (a `None`/negative entry leaves the dim untouched).
    pub fn shrink_to(&self, shape: &[SVal]) -> Tensor {
        self.shrink_to_ctx(&mut self.ctx.borrow_mut(), shape)
    }
    #[doc(hidden)]
    pub fn shrink_to_ctx(&self, ctx: &mut UOpCtx, shape: &[SVal]) -> Tensor {
        {
            let __u = movement::shrink_to(ctx, self.uop().clone(), shape);
            Tensor::from_uop(ctx, __u)
        }
    }

    /// `Tensor[t1, t2, ...]` — the `s![...]` index spec.
    pub fn getitem(&self, ixs: &[Ix]) -> Tensor {
        self.getitem_ctx(&mut self.ctx.borrow_mut(), ixs)
    }
    #[doc(hidden)]
    pub fn getitem_ctx(&self, ctx: &mut UOpCtx, ixs: &[Ix]) -> Tensor {
        {
            let __u = movement::getitem(ctx, self.uop().clone(), ixs);
            Tensor::from_uop(ctx, __u)
        }
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
    ctx.uop(
        Ops::Mstack,
        None,
        std::iter::once(x).chain(srcs).collect(),
        Arg::None,
    )
}

/// `UOp.after`: `AFTER(self, *src)` or self when empty (ops.py).
pub fn after(ctx: &mut UOpCtx, x: UopId, srcs: Vec<UopId>) -> UopId {
    if srcs.is_empty() {
        return x;
    }
    ctx.uop(
        Ops::After,
        None,
        std::iter::once(x).chain(srcs).collect(),
        Arg::None,
    )
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
    ctx.uop(
        Ops::Load,
        None,
        std::iter::once(x).chain(srcs).collect(),
        Arg::None,
    )
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

/// `UOp.valid(cond)` — `x` where `cond`, else an INVALID const.
pub fn valid(ctx: &mut UOpCtx, x: UopId, cond: UopId) -> UopId {
    let inv = ctx.const_like_dt(x.clone(), ConstVal::Invalid, None);
    ctx.where_(cond, x, inv)
}

/// ops enum value as an Arg (for REDUCE `(op, n)` args and similar).
pub fn op_arg(op: Ops) -> Arg {
    Arg::Enum(EnumVal {
        class: EnumClass::Ops,
        value: op as u16,
    })
}

impl Tensor {
    /// `Tensor(data=list, ...)` — host-data constructor (`&[T]`, T: SliceElem).
    pub fn from_slice<T: creation::SliceElem>(
        ctx: impl AsCtx,
        data: &[T],
        dtype: Option<DTypeId>,
        device: Option<&str>,
    ) -> Tensor {
        ctx.with_ctx(|ctx| creation::from_slice(ctx, data, dtype, device))
    }

    /// `Tensor(data=bytes, dtype, device=?)` — raw bytes + dtype.
    pub fn from_bytes(
        ctx: impl AsCtx,
        data: &[u8],
        dtype: DTypeId,
        device: Option<&str>,
    ) -> Tensor {
        ctx.with_ctx(|ctx| creation::from_bytes(ctx, data, dtype, device))
    }

    /// Read exactly `numel(shape) * dtype.itemsize()` bytes from `reader` and
    /// build a tensor of `shape` on `device`.
    ///
    /// This is the weight-loading primitive: `reader` is usually a
    /// `std::fs::File` window into a safetensors tensor (or any `Read`). Only
    /// one tensor's bytes are held at a time, so a multi-GB checkpoint does not
    /// need a multi-GB host buffer.
    pub fn from_read<R: std::io::Read>(
        ctx: impl AsCtx,
        reader: &mut R,
        dtype: DTypeId,
        shape: impl ShapeArg,
        device: Option<&str>,
    ) -> Tensor {
        let shape = shape.into_shape_vec();
        let numel: i128 = shape.iter().map(|s| s.clone().as_i128_or_panic()).product();
        assert!(
            numel >= 0,
            "TGERR: from_read shape has a negative dimension"
        );
        let mut bytes = vec![0u8; numel as usize * dtype.itemsize()];
        reader
            .read_exact(&mut bytes)
            .expect("TGERR: from_read: reader ended before the tensor was complete");
        ctx.with_ctx(|ctx| {
            let t = creation::from_bytes(ctx, &bytes, dtype, device);
            let u = movement::reshape(ctx, t.uop(), &shape);
            Tensor::from_uop(ctx, u)
        })
    }

    /// `Tensor.full(shape, fill_value, ...)`.
    pub fn full(
        ctx: impl AsCtx,
        shape: impl ShapeArg,
        fill_value: ConstVal,
        dtype: Option<DTypeId>,
        device: Option<&str>,
    ) -> Tensor {
        ctx.with_ctx(|ctx| {
            creation::full(
                ctx,
                &shape.into_shape_vec(),
                fill_value,
                dtype,
                device,
                true,
            )
        })
    }

    /// `Tensor.full(shape, fill_value, buffer=False)` — the broadcast const.
    pub fn full_const(
        ctx: impl AsCtx,
        shape: impl ShapeArg,
        fill_value: ConstVal,
        dtype: Option<DTypeId>,
    ) -> Tensor {
        ctx.with_ctx(|ctx| {
            creation::full(ctx, &shape.into_shape_vec(), fill_value, dtype, None, false)
        })
    }

    /// `Tensor.zeros(*shape)`.
    pub fn zeros(ctx: impl AsCtx, shape: impl ShapeArg) -> Tensor {
        ctx.with_ctx(|ctx| creation::zeros(ctx, &shape.into_shape_vec(), None, None))
    }

    /// `Tensor.ones(*shape)`.
    pub fn ones(ctx: impl AsCtx, shape: impl ShapeArg) -> Tensor {
        ctx.with_ctx(|ctx| creation::ones(ctx, &shape.into_shape_vec(), None, None))
    }

    /// `Tensor.zeros(*shape)` as a persistent optimizer-state leaf: a pure
    /// BUFFER with `initial_data` and **no in-graph STORE**, so [`Tensor::assign`]
    /// can update it in place without the zero-store re-running on a later
    /// realize (see CURRENT_STATE §9.5). Use this (not [`Tensor::zeros`]) for
    /// optimizer momentum / Adam moments.
    pub fn zeros_param(
        ctx: impl AsCtx,
        shape: impl ShapeArg,
        dtype: Option<DTypeId>,
        device: Option<&str>,
    ) -> Tensor {
        ctx.with_ctx(|ctx| {
            let t = creation::zeros(ctx, &shape.into_shape_vec(), dtype, device);
            t.to_param_ctx(ctx)
        })
    }

    /// `Tensor.arange(start, stop, step, dtype)`.
    pub fn arange(
        ctx: impl AsCtx,
        start: f64,
        stop: Option<f64>,
        step: f64,
        dtype: Option<DTypeId>,
    ) -> Tensor {
        ctx.with_ctx(|ctx| creation::arange(ctx, start, stop, step, dtype))
    }

    /// `Tensor.eye(n, m, dtype)`.
    pub fn eye(ctx: impl AsCtx, n: i64, m: Option<i64>, dtype: Option<DTypeId>) -> Tensor {
        ctx.with_ctx(|ctx| creation::eye(ctx, n, m, dtype))
    }

    /// `Tensor.linspace(start, stop, steps, dtype)`.
    pub fn linspace(
        ctx: impl AsCtx,
        start: f64,
        stop: f64,
        steps: i64,
        dtype: Option<DTypeId>,
    ) -> Tensor {
        ctx.with_ctx(|ctx| creation::linspace(ctx, start, stop, steps, dtype))
    }

    /// `Tensor.rand(*shape)`: random uniform [0, 1).
    pub fn rand(
        ctx: impl AsCtx,
        shape: &[i64],
        dtype: Option<DTypeId>,
        device: Option<&str>,
    ) -> Tensor {
        ctx.with_ctx(|ctx| {
            let s: Vec<SVal> = shape.iter().map(|&x| SVal::I(x as i128)).collect();
            creation::rand(ctx, &s, dtype, device)
        })
    }

    /// `Tensor.uniform(*shape, low=0.0, high=1.0)`: random uniform [low, high).
    pub fn uniform(
        ctx: impl AsCtx,
        shape: &[i64],
        low: Option<f64>,
        high: Option<f64>,
        dtype: Option<DTypeId>,
        device: Option<&str>,
    ) -> Tensor {
        ctx.with_ctx(|ctx| {
            let s: Vec<SVal> = shape.iter().map(|&x| SVal::I(x as i128)).collect();
            let low = low.unwrap_or(0.0);
            let high = high.unwrap_or(1.0);
            creation::uniform(ctx, &s, low, high, dtype, device)
        })
    }

    /// `Tensor.randn(*shape)`: random values from a standard normal distribution.
    pub fn randn(
        ctx: impl AsCtx,
        shape: &[i64],
        dtype: Option<DTypeId>,
        device: Option<&str>,
    ) -> Tensor {
        ctx.with_ctx(|ctx| {
            let s: Vec<SVal> = shape.iter().map(|&x| SVal::I(x as i128)).collect();
            rand::randn(ctx, &s, dtype, device)
        })
    }

    /// `Tensor.randint(*shape, low, high)`: random integers in `[low, high)`.
    pub fn randint(
        ctx: impl AsCtx,
        shape: &[i64],
        low: i32,
        high: i32,
        dtype: Option<DTypeId>,
        device: Option<&str>,
    ) -> Tensor {
        ctx.with_ctx(|ctx| {
            let s: Vec<SVal> = shape.iter().map(|&x| SVal::I(x as i128)).collect();
            rand::randint(ctx, &s, low, high, dtype, device)
        })
    }

    /// `Tensor.manual_seed(seed)` (tensor.py:624): reset the Threefry RNG state.
    pub fn manual_seed(seed: i64) {
        rand::manual_seed(seed)
    }

    /// `Tensor.rand_like(dtype=None, device=None)` (rand.py:70):
    /// `rand(*self.shape, dtype=dtype or self.dtype, device=device or self.device)`.
    /// (`contiguous` is a layout flag in tinygrad and does not change the values;
    /// the port's `rand` already yields a contiguous result.)
    pub fn rand_like(&self, dtype: Option<DTypeId>) -> Tensor {
        self.rand_like_ctx(&mut self.ctx.borrow_mut(), dtype)
    }
    #[doc(hidden)]
    pub fn rand_like_ctx(&self, ctx: &mut UOpCtx, dtype: Option<DTypeId>) -> Tensor {
        let shape = self.shape_ctx(ctx).into_vec();
        let dev = self.device_ctx(ctx);
        let dt = dtype.unwrap_or_else(|| self.dtype_ctx(ctx));
        rand::rand(ctx, &shape, Some(dt), dev.as_deref())
    }

    /// `Tensor.const_like(b, dtype=None)` (creation.py:14): a tensor of `b` with
    /// `self`'s shape (and dtype unless overridden).
    pub fn const_like(&self, b: ConstVal, dtype: Option<DTypeId>) -> Tensor {
        self.const_like_ctx(&mut self.ctx.borrow_mut(), b, dtype)
    }
    #[doc(hidden)]
    pub fn const_like_ctx(&self, ctx: &mut UOpCtx, b: ConstVal, dtype: Option<DTypeId>) -> Tensor {
        let shape = self.shape_ctx(ctx).into_vec();
        let dt = dtype.unwrap_or_else(|| self.dtype_ctx(ctx));
        Tensor::full_const(&mut *ctx, shape, b, Some(dt))
    }

    /// `Tensor.dropout(p=0.5)` (rand.py:274-296): inverted dropout, active only
    /// under `TRAINING`. Kept elements are scaled by `1/(1-p)` so the expected
    /// value is unchanged. Set `TRAINING` via
    /// [`Session::training`](crate::engine::session::Session::training) or
    /// [`crate::helpers::context::with_training`].
    pub fn dropout(&self, p: f64) -> Tensor {
        self.dropout_ctx(&mut self.ctx.borrow_mut(), p)
    }
    #[doc(hidden)]
    pub fn dropout_ctx(&self, ctx: &mut UOpCtx, p: f64) -> Tensor {
        assert!(
            (0.0..=1.0).contains(&p),
            "TGERR: dropout p is out of range [0, 1]"
        );
        if !crate::helpers::context::training() || p == 0.0 {
            return self.clone();
        }
        if p == 1.0 {
            return self.const_like_ctx(ctx, ConstVal::Float(0.0), None);
        }
        // (self.rand_like(default_float) >= p).contiguous().where(self, 0) / (1 - p)
        let r = self.rand_like_ctx(ctx, Some(crate::dtype::default_float()));
        let thr = Tensor::full_const(
            &mut *ctx,
            &[] as &[i64],
            ConstVal::Float(p),
            Some(crate::dtype::default_float()),
        );
        let mask = r.ge_ctx(ctx, &thr).contiguous_ctx(ctx);
        let zero = Tensor::full_const(&mut *ctx, &[] as &[i64], ConstVal::Float(0.0), None);
        let kept = mask.where__ctx(ctx, self, zero);
        // oracle divides by `(1 - p)` (inverted dropout)
        let scale = Tensor::full_const(&mut *ctx, &[] as &[i64], ConstVal::Float(1.0 - p), None);
        kept.div_ctx(ctx, &scale)
    }

    fn prod_tail(shape: &[i64]) -> f64 {
        shape[1..].iter().map(|&x| x as f64).product()
    }

    /// `Tensor.kaiming_uniform(*shape, a=0.01)` (rand.py:208):
    /// `uniform(-bound, bound)` with `bound = sqrt(6/(1+a^2)/prod(shape[1:]))`.
    pub fn kaiming_uniform(ctx: impl AsCtx, shape: &[i64], a: f64, device: Option<&str>) -> Tensor {
        ctx.with_ctx(|ctx| {
            let bound = (6.0 / (1.0 + a * a) / Tensor::prod_tail(shape)).sqrt();
            let s: Vec<SVal> = shape.iter().map(|&x| SVal::I(x as i128)).collect();
            creation::uniform(ctx, &s, -bound, bound, None, device)
        })
    }

    /// `Tensor.glorot_uniform(*shape)` (rand.py:192):
    /// `bound = sqrt(6/(shape[0]+prod(shape[1:])))`.
    pub fn glorot_uniform(ctx: impl AsCtx, shape: &[i64], device: Option<&str>) -> Tensor {
        ctx.with_ctx(|ctx| {
            let head = shape[0] as f64;
            let bound = (6.0 / (head + Tensor::prod_tail(shape))).sqrt();
            let s: Vec<SVal> = shape.iter().map(|&x| SVal::I(x as i128)).collect();
            creation::uniform(ctx, &s, -bound, bound, None, device)
        })
    }

    /// `Tensor.scaled_uniform(*shape)` (rand.py:176):
    /// `uniform(-1, 1) * prod(shape)**-0.5`.
    pub fn scaled_uniform(ctx: impl AsCtx, shape: &[i64], device: Option<&str>) -> Tensor {
        ctx.with_ctx(|ctx| {
            let s: Vec<SVal> = shape.iter().map(|&x| SVal::I(x as i128)).collect();
            let t = creation::uniform(ctx, &s, -1.0, 1.0, None, device);
            let prod: f64 = shape.iter().map(|&x| x as f64).product();
            t.mul_ctx(ctx, prod.powf(-0.5))
        })
    }

    /// Tensor-level reduce sugar.
    pub fn sum(&self, axis: Option<&[usize]>, keepdim: bool, dtype: Option<DTypeId>) -> Tensor {
        self.sum_ctx(&mut self.ctx.borrow_mut(), axis, keepdim, dtype)
    }
    #[doc(hidden)]
    pub fn sum_ctx(
        &self,
        ctx: &mut UOpCtx,
        axis: Option<&[usize]>,
        keepdim: bool,
        dtype: Option<DTypeId>,
    ) -> Tensor {
        {
            let __u = reduce::sum(ctx, self.uop().clone(), axis, keepdim, dtype);
            Tensor::from_uop(ctx, __u)
        }
    }
    /// `Tensor.prod(axis, keepdim)` — product along `axis`.
    pub fn prod(&self, axis: Option<&[usize]>, keepdim: bool, dtype: Option<DTypeId>) -> Tensor {
        self.prod_ctx(&mut self.ctx.borrow_mut(), axis, keepdim, dtype)
    }
    #[doc(hidden)]
    pub fn prod_ctx(
        &self,
        ctx: &mut UOpCtx,
        axis: Option<&[usize]>,
        keepdim: bool,
        dtype: Option<DTypeId>,
    ) -> Tensor {
        {
            let __u = reduce::prod(ctx, self.uop().clone(), axis, keepdim, dtype);
            Tensor::from_uop(ctx, __u)
        }
    }
    /// `Tensor.max(axis, keepdim)` — maximum along `axis`.
    pub fn max(&self, axis: Option<&[usize]>, keepdim: bool) -> Tensor {
        self.max_ctx(&mut self.ctx.borrow_mut(), axis, keepdim)
    }
    #[doc(hidden)]
    pub fn max_ctx(&self, ctx: &mut UOpCtx, axis: Option<&[usize]>, keepdim: bool) -> Tensor {
        {
            let __u = reduce::max(ctx, self.uop().clone(), axis, keepdim);
            Tensor::from_uop(ctx, __u)
        }
    }
    /// `Tensor.mean(axis, keepdim)` — arithmetic mean along `axis`.
    pub fn mean(&self, axis: Option<&[usize]>, keepdim: bool) -> Tensor {
        self.mean_ctx(&mut self.ctx.borrow_mut(), axis, keepdim)
    }
    #[doc(hidden)]
    pub fn mean_ctx(&self, ctx: &mut UOpCtx, axis: Option<&[usize]>, keepdim: bool) -> Tensor {
        {
            let __u = reduce::mean(ctx, self.uop().clone(), axis, keepdim);
            Tensor::from_uop(ctx, __u)
        }
    }
    /// `Tensor.var(axis, keepdim, correction)` — variance with Bessel correction.
    pub fn var(&self, axis: Option<&[usize]>, keepdim: bool, correction: i64) -> Tensor {
        self.var_ctx(&mut self.ctx.borrow_mut(), axis, keepdim, correction)
    }
    #[doc(hidden)]
    pub fn var_ctx(
        &self,
        ctx: &mut UOpCtx,
        axis: Option<&[usize]>,
        keepdim: bool,
        correction: i64,
    ) -> Tensor {
        {
            let __u = reduce::var(ctx, self.uop().clone(), axis, keepdim, correction);
            Tensor::from_uop(ctx, __u)
        }
    }
    /// `Tensor.cumsum(axis)` — cumulative sum along `axis`.
    pub fn cumsum(&self, axis: i64) -> Tensor {
        self.cumsum_ctx(&mut self.ctx.borrow_mut(), axis)
    }
    #[doc(hidden)]
    pub fn cumsum_ctx(&self, ctx: &mut UOpCtx, axis: i64) -> Tensor {
        {
            let __u = reduce::cumsum(ctx, self.uop().clone(), axis);
            Tensor::from_uop(ctx, __u)
        }
    }

    // *** nn-layer operations ***

    /// `Tensor.sub(other)`: `self - other` (composed via add/neg).
    pub fn sub(&self, other: impl Into<TensorAddArg>) -> Tensor {
        self.sub_ctx(&mut self.ctx.borrow_mut(), other)
    }
    #[doc(hidden)]
    pub fn sub_ctx(&self, ctx: &mut UOpCtx, other: impl Into<TensorAddArg>) -> Tensor {
        match other.into() {
            TensorAddArg::T(o) => {
                let __u = elementwise::sub(ctx, self.uop().clone(), o.uop());
                Tensor::from_uop(ctx, __u)
            }
            TensorAddArg::C(c) => {
                let __u = elementwise::sub_c(ctx, self.uop().clone(), c);
                Tensor::from_uop(ctx, __u)
            }
        }
    }

    /// `Tensor.matmul(other)`: matrix multiply — last-1 of self × last-1 of other, sum over that dim.
    /// Handles batch dims: self (N, K) @ other (K, M) → (N, M).
    /// For general batch shapes: self (.., N, K), other (.., K, M) → (.., N, M).
    pub fn matmul(&self, other: impl Into<TensorAddArg>) -> Tensor {
        self.matmul_ctx(&mut self.ctx.borrow_mut(), other)
    }
    #[doc(hidden)]
    pub fn matmul_ctx(&self, ctx: &mut UOpCtx, other: impl Into<TensorAddArg>) -> Tensor {
        // tinygrad `matmul` is just `dot` (mixin/op.py:407) — `dot` handles the
        // batch/contraction reshaping itself, so there is no extra batch logic.
        let other = match other.into() {
            TensorAddArg::T(o) => o,
            TensorAddArg::C(_) => panic!("matmul: RHS must be a Tensor, not a scalar"),
        };
        self.dot_ctx(ctx, &other)
    }

    /// `Tensor.square()`: `self * self`.
    pub fn square(&self) -> Tensor {
        self.square_ctx(&mut self.ctx.borrow_mut())
    }
    #[doc(hidden)]
    pub fn square_ctx(&self, ctx: &mut UOpCtx) -> Tensor {
        {
            let __u = elementwise::square(ctx, self.uop().clone());
            Tensor::from_uop(ctx, __u)
        }
    }

    /// `Tensor.sqrt()`: square root.
    pub fn sqrt(&self) -> Tensor {
        self.sqrt_ctx(&mut self.ctx.borrow_mut())
    }
    #[doc(hidden)]
    pub fn sqrt_ctx(&self, ctx: &mut UOpCtx) -> Tensor {
        {
            let __u = elementwise::sqrt(ctx, self.uop().clone());
            Tensor::from_uop(ctx, __u)
        }
    }

    /// `Tensor.rsqrt()`: reciprocal square root (`1 / sqrt(x)`).
    pub fn rsqrt(&self) -> Tensor {
        self.rsqrt_ctx(&mut self.ctx.borrow_mut())
    }
    #[doc(hidden)]
    pub fn rsqrt_ctx(&self, ctx: &mut UOpCtx) -> Tensor {
        self.sqrt_ctx(ctx).reciprocal_ctx(ctx)
    }

    /// `Tensor.mean()`: mean over all dims.
    pub fn mean_all(&self) -> Tensor {
        self.mean_all_ctx(&mut self.ctx.borrow_mut())
    }
    #[doc(hidden)]
    pub fn mean_all_ctx(&self, ctx: &mut UOpCtx) -> Tensor {
        self.mean_ctx(ctx, None, false)
    }

    /// `Tensor.abs()`: max(self, -self).
    pub fn abs(&self) -> Tensor {
        self.abs_ctx(&mut self.ctx.borrow_mut())
    }
    #[doc(hidden)]
    pub fn abs_ctx(&self, ctx: &mut UOpCtx) -> Tensor {
        let neg = self.neg_ctx(ctx);
        self.maximum_ctx(ctx, &neg)
    }

    /// `Tensor.maximum(other)`: elementwise max.
    pub fn maximum(&self, other: impl Into<TensorAddArg>) -> Tensor {
        self.maximum_ctx(&mut self.ctx.borrow_mut(), other)
    }
    #[doc(hidden)]
    pub fn maximum_ctx(&self, ctx: &mut UOpCtx, other: impl Into<TensorAddArg>) -> Tensor {
        match other.into() {
            TensorAddArg::T(o) => {
                let __u = elementwise::maximum(ctx, self.uop().clone(), o.uop());
                Tensor::from_uop(ctx, __u)
            }
            TensorAddArg::C(c) => {
                let __u = elementwise::binop_c(ctx, Ops::Max, self.uop().clone(), c);
                Tensor::from_uop(ctx, __u)
            }
        }
    }

    /// `Tensor.minimum(other)`: elementwise min.
    pub fn minimum(&self, other: impl Into<TensorAddArg>) -> Tensor {
        self.minimum_ctx(&mut self.ctx.borrow_mut(), other)
    }
    #[doc(hidden)]
    pub fn minimum_ctx(&self, ctx: &mut UOpCtx, other: impl Into<TensorAddArg>) -> Tensor {
        match other.into() {
            TensorAddArg::T(o) => {
                let __u = elementwise::minimum(ctx, self.uop().clone(), o.uop());
                Tensor::from_uop(ctx, __u)
            }
            // No Ops::MIN — use min(a, b) == a + b - max(a, b).
            TensorAddArg::C(c) => {
                let sum = self.add_ctx(ctx, TensorAddArg::C(c.clone()));
                let mx = self.maximum_ctx(ctx, TensorAddArg::C(c));
                sum.sub_ctx(ctx, mx)
            }
        }
    }

    /// `Tensor.__gt__` — elementwise `self > other` (tinygrad `gt`).
    pub fn gt(&self, other: impl Into<TensorAddArg>) -> Tensor {
        self.gt_ctx(&mut self.ctx.borrow_mut(), other)
    }
    #[doc(hidden)]
    pub fn gt_ctx(&self, ctx: &mut UOpCtx, other: impl Into<TensorAddArg>) -> Tensor {
        let o = self.cmp_operand(ctx, other);
        {
            let __u = elementwise::gt(ctx, self.uop().clone(), o);
            Tensor::from_uop(ctx, __u)
        }
    }

    /// `Tensor.__ge__` — elementwise `self >= other` (tinygrad `ge`).
    pub fn ge(&self, other: impl Into<TensorAddArg>) -> Tensor {
        self.ge_ctx(&mut self.ctx.borrow_mut(), other)
    }
    #[doc(hidden)]
    pub fn ge_ctx(&self, ctx: &mut UOpCtx, other: impl Into<TensorAddArg>) -> Tensor {
        let o = self.cmp_operand(ctx, other);
        {
            let __u = elementwise::ge(ctx, self.uop().clone(), o);
            Tensor::from_uop(ctx, __u)
        }
    }

    /// `Tensor.__lt__` — elementwise `self < other`.
    pub fn lt(&self, other: impl Into<TensorAddArg>) -> Tensor {
        self.lt_ctx(&mut self.ctx.borrow_mut(), other)
    }
    #[doc(hidden)]
    pub fn lt_ctx(&self, ctx: &mut UOpCtx, other: impl Into<TensorAddArg>) -> Tensor {
        let o = self.cmp_operand(ctx, other);
        {
            let __u = elementwise::lt(ctx, self.uop().clone(), o);
            Tensor::from_uop(ctx, __u)
        }
    }

    /// `Tensor.__ne__` — elementwise `self != other`.
    pub fn ne(&self, other: impl Into<TensorAddArg>) -> Tensor {
        self.ne_ctx(&mut self.ctx.borrow_mut(), other)
    }
    #[doc(hidden)]
    pub fn ne_ctx(&self, ctx: &mut UOpCtx, other: impl Into<TensorAddArg>) -> Tensor {
        let o = self.cmp_operand(ctx, other);
        {
            let __u = elementwise::ne(ctx, self.uop().clone(), o);
            Tensor::from_uop(ctx, __u)
        }
    }

    /// `cond.where(x, y)` — elementwise select using `self` as the condition
    /// (tinygrad `Tensor.where`). Scalars are promoted to `self`'s shape.
    pub fn where_(&self, x: impl Into<TensorAddArg>, y: impl Into<TensorAddArg>) -> Tensor {
        self.where__ctx(&mut self.ctx.borrow_mut(), x, y)
    }
    #[doc(hidden)]
    #[allow(non_snake_case)]
    pub fn where__ctx(
        &self,
        ctx: &mut UOpCtx,
        x: impl Into<TensorAddArg>,
        y: impl Into<TensorAddArg>,
    ) -> Tensor {
        let xu = self.cmp_operand(ctx, x);
        let yu = self.cmp_operand(ctx, y);
        {
            let __u = elementwise::where_(ctx, self.uop().clone(), xu, yu);
            Tensor::from_uop(ctx, __u)
        }
    }

    /// UOp for a binop operand: tensor passes through, python scalar becomes a
    /// same-shaped const so the binop broadcasts.
    fn cmp_operand(&self, ctx: &mut UOpCtx, other: impl Into<TensorAddArg>) -> UopId {
        match other.into() {
            TensorAddArg::T(o) => o.uop(),
            TensorAddArg::C(c) => ctx.const_like_dt(self.uop().clone(), c, None),
        }
    }

    /// `Tensor.reciprocal()`: 1 / self.
    pub fn reciprocal(&self) -> Tensor {
        self.reciprocal_ctx(&mut self.ctx.borrow_mut())
    }
    #[doc(hidden)]
    pub fn reciprocal_ctx(&self, ctx: &mut UOpCtx) -> Tensor {
        {
            let __u = elementwise::reciprocal(ctx, self.uop().clone());
            Tensor::from_uop(ctx, __u)
        }
    }

    /// `Tensor.log()`: natural log.
    pub fn log(&self) -> Tensor {
        self.log_ctx(&mut self.ctx.borrow_mut())
    }
    #[doc(hidden)]
    pub fn log_ctx(&self, ctx: &mut UOpCtx) -> Tensor {
        {
            let __u = elementwise::log(ctx, self.uop().clone());
            Tensor::from_uop(ctx, __u)
        }
    }

    /// `Tensor.log2()`: log base 2.
    pub fn log2(&self) -> Tensor {
        self.log2_ctx(&mut self.ctx.borrow_mut())
    }
    #[doc(hidden)]
    pub fn log2_ctx(&self, ctx: &mut UOpCtx) -> Tensor {
        {
            let __u = elementwise::log2(ctx, self.uop());
            Tensor::from_uop(ctx, __u)
        }
    }

    /// `Tensor.exp()`: exponential.
    pub fn exp(&self) -> Tensor {
        self.exp_ctx(&mut self.ctx.borrow_mut())
    }
    #[doc(hidden)]
    pub fn exp_ctx(&self, ctx: &mut UOpCtx) -> Tensor {
        {
            let __u = elementwise::exp(ctx, self.uop().clone());
            Tensor::from_uop(ctx, __u)
        }
    }

    /// `Tensor.exp2()`: 2^self.
    pub fn exp2(&self) -> Tensor {
        self.exp2_ctx(&mut self.ctx.borrow_mut())
    }
    #[doc(hidden)]
    pub fn exp2_ctx(&self, ctx: &mut UOpCtx) -> Tensor {
        {
            let __u = elementwise::exp2(ctx, self.uop());
            Tensor::from_uop(ctx, __u)
        }
    }

    /// `Tensor.trunc()`: truncate toward zero.
    pub fn trunc(&self) -> Tensor {
        self.trunc_ctx(&mut self.ctx.borrow_mut())
    }
    #[doc(hidden)]
    pub fn trunc_ctx(&self, ctx: &mut UOpCtx) -> Tensor {
        {
            let __u = elementwise::trunc(ctx, self.uop().clone());
            Tensor::from_uop(ctx, __u)
        }
    }

    /// `Tensor.ceil()`: ceiling.
    pub fn ceil(&self) -> Tensor {
        self.ceil_ctx(&mut self.ctx.borrow_mut())
    }
    #[doc(hidden)]
    pub fn ceil_ctx(&self, ctx: &mut UOpCtx) -> Tensor {
        {
            let __u = elementwise::ceil(ctx, self.uop().clone());
            Tensor::from_uop(ctx, __u)
        }
    }

    /// `Tensor.floor()`: floor.
    pub fn floor(&self) -> Tensor {
        self.floor_ctx(&mut self.ctx.borrow_mut())
    }
    #[doc(hidden)]
    pub fn floor_ctx(&self, ctx: &mut UOpCtx) -> Tensor {
        {
            let __u = elementwise::floor(ctx, self.uop().clone());
            Tensor::from_uop(ctx, __u)
        }
    }

    /// `Tensor.sin()`: sine.
    pub fn sin(&self) -> Tensor {
        self.sin_ctx(&mut self.ctx.borrow_mut())
    }
    #[doc(hidden)]
    pub fn sin_ctx(&self, ctx: &mut UOpCtx) -> Tensor {
        {
            let __u = elementwise::sin(ctx, self.uop().clone());
            Tensor::from_uop(ctx, __u)
        }
    }

    /// `Tensor.tanh()`: hyperbolic tangent.
    pub fn tanh(&self) -> Tensor {
        self.tanh_ctx(&mut self.ctx.borrow_mut())
    }
    #[doc(hidden)]
    pub fn tanh_ctx(&self, ctx: &mut UOpCtx) -> Tensor {
        {
            let __u = elementwise::tanh(ctx, self.uop().clone());
            Tensor::from_uop(ctx, __u)
        }
    }

    /// `Tensor.dot(w)`: sum product over last axis of self and second-to-last of w.
    /// self (.., N) @ w (.., K, N) → (.., K) — mirrors tinygrad's dot.
    pub fn dot(&self, w: &Tensor) -> Tensor {
        self.dot_ctx(&mut self.ctx.borrow_mut(), w)
    }
    #[doc(hidden)]
    pub fn dot_ctx(&self, ctx: &mut UOpCtx, w: &Tensor) -> Tensor {
        self.dot_impl(ctx, w, None)
    }

    /// `Tensor.dot(w, dtype=None)` (op.py:367): like `dot` but accumulate
    /// the contraction in `dtype` (result cast to `dtype`). `None` keeps the
    /// input dtype.
    pub fn dot_dtype(&self, w: &Tensor, dtype: Option<DTypeId>) -> Tensor {
        self.dot_dtype_ctx(&mut self.ctx.borrow_mut(), w, dtype)
    }
    #[doc(hidden)]
    pub fn dot_dtype_ctx(&self, ctx: &mut UOpCtx, w: &Tensor, dtype: Option<DTypeId>) -> Tensor {
        self.dot_impl(ctx, w, dtype)
    }

    fn dot_impl(&self, ctx: &mut UOpCtx, w: &Tensor, dtype: Option<DTypeId>) -> Tensor {
        let x_shape = self.shape_ctx(ctx);
        let w_shape = w.shape_ctx(ctx);
        let dx = x_shape.len();
        let dw = w_shape.len();
        assert!(
            dx > 0 && dw > 0,
            "Tensor::dot: both tensors must be at least 1D"
        );
        // contraction axis of w: index dw - min(dw, 2) (last axis for 1-D w, second-to-last for N-D w)
        let axis_w = (dw - std::cmp::min(dw, 2)) as usize;
        let x_last = <SInt as Clone>::clone(&x_shape[dx - 1])
            .as_i64()
            .unwrap_or(0);
        let w_contract = <SInt as Clone>::clone(&w_shape[axis_w])
            .as_i64()
            .unwrap_or(0);
        assert_eq!(
            x_last, w_contract,
            "Tensor::dot: shape mismatch {} vs {}",
            x_shape, w_shape
        );

        let rep = std::cmp::min(std::cmp::min(dx - 1, dw - 1), 1);
        // x -> (*x.shape[:-1], [1]*rep, x.shape[-1])
        let mut xr: Vec<SVal> = x_shape[..dx - 1].to_vec();
        xr.extend(std::iter::repeat_n(SVal::I(1), rep));
        xr.push(x_shape[dx - 1].clone());
        let x_r = self.reshape_ctx(ctx, xr);

        // w -> (*w.shape[:-2], [1]*rep, *w.shape[axis_w:]) then transpose(-1, axis_w - dw)
        let mut wr: Vec<SVal> = w_shape[..dw.saturating_sub(2)].to_vec();
        wr.extend(std::iter::repeat_n(SVal::I(1), rep));
        wr.extend_from_slice(&w_shape[axis_w..]);
        let w_r = w.reshape_ctx(ctx, wr);
        let axis_w_neg = -(std::cmp::min(dw, 2) as i64);
        let w_t = w_r.transpose_ctx(ctx, -1, axis_w_neg);

        // sum-reduce the broadcasted product over the last axis
        let product = x_r.mul_ctx(ctx, &w_t);
        let last_axis = product.shape_ctx(ctx).len() - 1;
        let summed = product.sum_ctx(ctx, Some(&[last_axis]), false, dtype);
        match dtype {
            Some(dt) => summed.cast_ctx(ctx, dt),
            None => summed,
        }
    }

    /// `Tensor.linear(weight, bias)`: dot/mul + bias add.
    /// If weight is 1D: self * weight + bias (elementwise).
    /// Otherwise: self.dot(weight) + bias.
    pub fn linear(&self, weight: &Tensor, bias: Option<&Tensor>) -> Tensor {
        self.linear_ctx(&mut self.ctx.borrow_mut(), weight, bias)
    }
    #[doc(hidden)]
    pub fn linear_ctx(&self, ctx: &mut UOpCtx, weight: &Tensor, bias: Option<&Tensor>) -> Tensor {
        let w_shape = weight.shape_ctx(ctx);
        if w_shape.len() == 1 {
            // 1-D weight (`nn/__init__` LayerNorm affine): `self.mul(weight) + bias`.
            let last = self
                .shape_ctx(ctx)
                .into_vec()
                .last()
                .cloned()
                .unwrap_or(SVal::I(0));
            assert_eq!(
                w_shape[0], last,
                "TGERR: linear 1-D weight length must match self.shape[-1]"
            );
            let result = self.mul_ctx(ctx, weight);
            return match bias {
                Some(b) => result.add_ctx(ctx, b),
                None => result,
            };
        }
        let result = self.dot_ctx(ctx, weight);
        match bias {
            Some(b) => result.add_ctx(ctx, b),
            None => result,
        }
    }

    /// `Tensor.batchnorm(weight, bias, mean, invstd)`: batch normalization.
    pub fn batchnorm(
        &self,
        weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        mean: &Tensor,
        invstd: &Tensor,
        axis: &[usize],
    ) -> Tensor {
        self.batchnorm_ctx(&mut self.ctx.borrow_mut(), weight, bias, mean, invstd, axis)
    }
    #[doc(hidden)]
    pub fn batchnorm_ctx(
        &self,
        ctx: &mut UOpCtx,
        weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        mean: &Tensor,
        invstd: &Tensor,
        axis: &[usize],
    ) -> Tensor {
        let shape = self.shape_ctx(ctx);
        // tinygrad `batchnorm`: `shape = tuple(s if ax in axis_ else 1 ...)` —
        // the reduced axes keep their size (e.g. (1, C, 1, 1) for axis=1).
        let mut norm_shape = Vec::with_capacity(shape.len());
        for (i, s) in shape.iter().enumerate() {
            if axis.contains(&i) {
                norm_shape.push(s.clone());
            } else {
                norm_shape.push(SVal::I(1));
            }
        }
        let mean_reshaped = mean.reshape_ctx(ctx, &norm_shape);
        let mut x = self.sub_ctx(ctx, &mean_reshaped);
        if let Some(w) = weight {
            let w_reshaped = w.reshape_ctx(ctx, &norm_shape);
            x = x.mul_ctx(ctx, &w_reshaped);
        }
        let invstd_reshaped = if invstd.shape_ctx(ctx).len() == axis.len() {
            invstd.reshape_ctx(ctx, &norm_shape)
        } else {
            invstd.clone()
        };
        let ret = x.mul_ctx(ctx, &invstd_reshaped);
        match bias {
            Some(b) => {
                let b_reshaped = b.reshape_ctx(ctx, &norm_shape);
                ret.add_ctx(ctx, &b_reshaped)
            }
            None => ret,
        }
    }

    /// `Tensor.scaled_dot_product_attention(key, value, attn_mask=None,
    /// dropout_p=0.0, is_causal=False, enable_gqa=False)` (rand.py:298-327).
    /// `self` is the query: `softmax(q·kᵀ/√d + mask)·v` with f32 accumulation.
    pub fn scaled_dot_product_attention(
        &self,
        key: &Tensor,
        value: &Tensor,
        attn_mask: Option<&Tensor>,
        dropout_p: f64,
        is_causal: bool,
        enable_gqa: bool,
    ) -> Tensor {
        self.scaled_dot_product_attention_ctx(
            &mut self.ctx.borrow_mut(),
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            enable_gqa,
        )
    }
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)] // signature mirrors the tinygrad oracle
    pub fn scaled_dot_product_attention_ctx(
        &self,
        ctx: &mut UOpCtx,
        key: &Tensor,
        value: &Tensor,
        attn_mask: Option<&Tensor>,
        dropout_p: f64,
        is_causal: bool,
        enable_gqa: bool,
    ) -> Tensor {
        let ndim = self.shape_ctx(ctx).len();
        let (key, value) = if enable_gqa {
            let qh = self.shape_ctx(ctx).into_vec()[ndim - 3]
                .clone()
                .as_i64()
                .unwrap_or(0);
            let kh = key.shape_ctx(ctx).into_vec()[key.shape_ctx(ctx).len() - 3]
                .clone()
                .as_i64()
                .unwrap_or(0);
            let vh = value.shape_ctx(ctx).into_vec()[value.shape_ctx(ctx).len() - 3]
                .clone()
                .as_i64()
                .unwrap_or(0);
            (
                key.repeat_interleave_ctx(ctx, qh / kh, Some(-3)),
                value.repeat_interleave_ctx(ctx, qh / vh, Some(-3)),
            )
        } else {
            (key.clone(), value.clone())
        };
        let q = self.clone();
        let q_last = q.shape_ctx(ctx).into_vec()[ndim - 1]
            .clone()
            .as_i64()
            .unwrap_or(0);
        let acc = crate::dtype::least_upper_dtype(&[
            q.dtype_ctx(ctx),
            key.dtype_ctx(ctx),
            crate::dtype::FLOAT32,
        ]);
        let kt = key.transpose_ctx(ctx, -2, -1);
        let qk = q.dot_dtype_ctx(ctx, &kt, Some(acc)).div_ctx(
            ctx,
            TensorAddArg::C(ConstVal::Float((q_last as f64).sqrt())),
        );
        let mask: Option<Tensor> = if is_causal {
            assert!(
                attn_mask.is_none(),
                "TGERR: cannot set attn_mask when is_causal=True"
            );
            Some(
                qk.const_like_ctx(ctx, ConstVal::Bool(true), Some(crate::dtype::BOOL))
                    .tril_ctx(ctx, 0),
            )
        } else {
            attn_mask.cloned()
        };
        let qk = if let Some(m) = &mask {
            let m2 = if m.dtype_ctx(ctx) == crate::dtype::BOOL {
                let zero = Tensor::full_const(&mut *ctx, &[] as &[i64], ConstVal::Float(0.0), None);
                let neg_inf = Tensor::full_const(
                    &mut *ctx,
                    &[] as &[i64],
                    ConstVal::Float(f64::NEG_INFINITY),
                    None,
                );
                m.where__ctx(ctx, zero, neg_inf)
            } else {
                m.clone()
            };
            qk.add_ctx(ctx, &m2)
        } else {
            qk
        };
        let sm = crate::tensor::nn::softmax_ctx(
            &qk.cast_ctx(ctx, self.dtype_ctx(ctx)),
            ctx,
            Some(ndim - 1),
        );
        let sm = sm.dropout_ctx(ctx, dropout_p);
        sm.matmul_ctx(ctx, &value)
    }

    /// `Tensor.layernorm(axis, eps)`: layer normalization.
    pub fn layernorm(&self, axis: i64, eps: f64) -> Tensor {
        self.layernorm_ctx(&mut self.ctx.borrow_mut(), axis, eps)
    }
    #[doc(hidden)]
    pub fn layernorm_ctx(&self, ctx: &mut UOpCtx, axis: i64, eps: f64) -> Tensor {
        // Normalize over the axis dimension(s)
        let ndim = self.shape_ctx(ctx).len();
        let actual_axis = if axis < 0 { ndim as i64 + axis } else { axis };
        let axis_vec = vec![actual_axis as usize];
        let mean = self.mean_ctx(ctx, Some(&axis_vec), true);
        let diff = self.sub_ctx(ctx, TensorAddArg::T(mean));
        let var = diff.square_ctx(ctx).mean_ctx(ctx, Some(&axis_vec), true);
        let eps_tensor = Tensor::full_const(&mut *ctx, &[] as &[i64], ConstVal::Float(eps), None);
        let invstd = var
            .add_ctx(ctx, TensorAddArg::T(eps_tensor))
            .sqrt_ctx(ctx)
            .reciprocal_ctx(ctx);
        diff.mul_ctx(ctx, TensorAddArg::T(invstd))
    }

    /// `Tensor.pool(kernel_size, stride, dilation)`: sliding window pool (for conv2d internals).
    /// Scalar `stride`/`dilation` broadcast to every spatial dim.
    pub fn pool(&self, kernel_size: &[i64], stride: i64, dilation: i64) -> Tensor {
        self.pool_ctx(&mut self.ctx.borrow_mut(), kernel_size, stride, dilation)
    }
    #[doc(hidden)]
    pub fn pool_ctx(
        &self,
        ctx: &mut UOpCtx,
        kernel_size: &[i64],
        stride: i64,
        dilation: i64,
    ) -> Tensor {
        let s: Vec<i64> = vec![stride; kernel_size.len()];
        let d: Vec<i64> = vec![dilation; kernel_size.len()];
        self.pool_vec_ctx(ctx, kernel_size, &s, &d)
    }

    /// `Tensor._pool(k, stride, dilation)` with per-dim `stride`/`dilation`.
    pub fn pool_vec(&self, kernel_size: &[i64], stride: &[i64], dilation: &[i64]) -> Tensor {
        self.pool_vec_ctx(&mut self.ctx.borrow_mut(), kernel_size, stride, dilation)
    }
    #[doc(hidden)]
    pub fn pool_vec_ctx(
        &self,
        ctx: &mut UOpCtx,
        kernel_size: &[i64],
        stride: &[i64],
        dilation: &[i64],
    ) -> Tensor {
        {
            let __u = movement::pool(ctx, self.uop().clone(), kernel_size, stride, dilation);
            Tensor::from_uop(ctx, __u)
        }
    }

    /// `Tensor.div(other)`: elementwise division (a / b).
    pub fn div(&self, other: impl Into<TensorAddArg>) -> Tensor {
        self.div_ctx(&mut self.ctx.borrow_mut(), other)
    }
    #[doc(hidden)]
    pub fn div_ctx(&self, ctx: &mut UOpCtx, other: impl Into<TensorAddArg>) -> Tensor {
        match other.into() {
            TensorAddArg::T(o) => {
                let __u = elementwise::div(ctx, self.uop().clone(), o.uop(), None);
                Tensor::from_uop(ctx, __u)
            }
            TensorAddArg::C(c) => {
                let __u = elementwise::div_c(ctx, self.uop().clone(), c, None);
                Tensor::from_uop(ctx, __u)
            }
        }
    }

    /// `Tensor // other` (elementwise.py:275): floor division
    /// (`div(rounding_mode="floor")`). Int/int uses `Ops.FLOORDIV`.
    pub fn floordiv(&self, other: impl Into<TensorAddArg>) -> Tensor {
        self.floordiv_ctx(&mut self.ctx.borrow_mut(), other)
    }
    #[doc(hidden)]
    pub fn floordiv_ctx(&self, ctx: &mut UOpCtx, other: impl Into<TensorAddArg>) -> Tensor {
        match other.into() {
            TensorAddArg::T(o) => {
                let __u = elementwise::div(ctx, self.uop().clone(), o.uop(), Some("floor"));
                Tensor::from_uop(ctx, __u)
            }
            TensorAddArg::C(c) => {
                let __u = elementwise::div_c(ctx, self.uop().clone(), c, Some("floor"));
                Tensor::from_uop(ctx, __u)
            }
        }
    }

    /// `Tensor.mod(other)` / `%` (elementwise.py:198): Python floor remainder.
    pub fn mod_(&self, other: impl Into<TensorAddArg>) -> Tensor {
        self.mod__ctx(&mut self.ctx.borrow_mut(), other)
    }
    #[doc(hidden)]
    #[allow(non_snake_case)]
    pub fn mod__ctx(&self, ctx: &mut UOpCtx, other: impl Into<TensorAddArg>) -> Tensor {
        match other.into() {
            TensorAddArg::T(o) => {
                let __u = elementwise::mod_(ctx, self.uop().clone(), o.uop());
                Tensor::from_uop(ctx, __u)
            }
            TensorAddArg::C(c) => {
                let b = self.const_like_ctx(ctx, c, None);
                let b = b.uop().clone();
                let __u = elementwise::mod_(ctx, self.uop().clone(), b);
                Tensor::from_uop(ctx, __u)
            }
        }
    }

    /// `Tensor.interpolate(size, mode="linear", align_corners=False)`
    /// (op.py:1046-1078): resize the last `size.len()` dims. Supports `linear`
    /// (bilinear/trilinear/...) and `nearest`/`nearest-exact`.
    pub fn interpolate(&self, size: &[i64], mode: &str, align_corners: bool) -> Tensor {
        self.interpolate_ctx(&mut self.ctx.borrow_mut(), size, mode, align_corners)
    }
    #[doc(hidden)]
    pub fn interpolate_ctx(
        &self,
        ctx: &mut UOpCtx,
        size: &[i64],
        mode: &str,
        align_corners: bool,
    ) -> Tensor {
        let ndim = self.shape_ctx(ctx).len();
        assert!(
            !size.is_empty() && size.len() <= ndim,
            "TGERR: invalid interpolate size"
        );
        assert!(
            matches!(mode, "linear" | "nearest" | "nearest-exact"),
            "TGERR: only linear/nearest/nearest-exact"
        );
        assert!(
            !(align_corners && mode != "linear"),
            "TGERR: align_corners only with linear"
        );
        let self_shape: Vec<SVal> = self.shape_ctx(ctx).into_vec();
        let int_dt = Some(crate::dtype::default_int());
        let mut x = self.clone();
        for k in 0..size.len() {
            let dim = ndim - size.len() + k;
            let os = size[k];
            let in_sz = self_shape[dim].clone().as_i64().unwrap_or(0);
            let mut reshape: Vec<SVal> = vec![SVal::I(1); ndim];
            reshape[dim] = SVal::I(os as i128);
            let mut expand: Vec<SVal> = x.shape_ctx(ctx).into_vec();
            expand[dim] = SVal::I(os as i128);
            if mode == "linear" {
                let arr = Tensor::arange(&mut *ctx, 0.0, Some(os as f64), 1.0, int_dt);
                let (num, den_val): (Tensor, i64) = if align_corners {
                    (
                        arr.mul_ctx(ctx, TensorAddArg::C(ConstVal::Int((in_sz - 1) as i128))),
                        os - 1,
                    )
                } else {
                    let t = arr
                        .mul_ctx(ctx, TensorAddArg::C(ConstVal::Int(2)))
                        .add_ctx(ctx, TensorAddArg::C(ConstVal::Int(1)));
                    let num = t
                        .mul_ctx(ctx, TensorAddArg::C(ConstVal::Int(in_sz as i128)))
                        .sub_ctx(ctx, TensorAddArg::C(ConstVal::Int(os as i128)));
                    (num, os * 2)
                };
                let den = Tensor::full_const(
                    &mut *ctx,
                    &[] as &[i64],
                    ConstVal::Int(den_val as i128),
                    int_dt,
                );
                let num = num.clip_ctx(ctx, Some(0.0), Some(((in_sz - 1) * den_val) as f64));
                let low = num
                    .floordiv_ctx(ctx, &den)
                    .reshape_ctx(ctx, reshape.clone())
                    .expand_ctx(ctx, expand.clone());
                let high = num
                    .add_ctx(ctx, &den)
                    .sub_ctx(ctx, TensorAddArg::C(ConstVal::Int(1)))
                    .floordiv_ctx(ctx, &den)
                    .reshape_ctx(ctx, reshape.clone())
                    .expand_ctx(ctx, expand.clone());
                let numer = num.mod__ctx(ctx, &den).cast_ctx(ctx, crate::dtype::FLOAT32);
                let denf = den.cast_ctx(ctx, crate::dtype::FLOAT32);
                let perc = numer
                    .div_ctx(ctx, &denf)
                    .reshape_ctx(ctx, reshape.clone())
                    .expand_ctx(ctx, expand.clone());
                let gl = x.gather_ctx(ctx, dim as i64, &low);
                let gh = x.gather_ctx(ctx, dim as i64, &high);
                x = gl.lerp_ctx(ctx, &gh, &perc);
            } else {
                let scale = in_sz as f64 / os as f64;
                let arr = Tensor::arange(
                    &mut *ctx,
                    0.0,
                    Some(os as f64),
                    1.0,
                    Some(crate::dtype::FLOAT32),
                );
                let scaled = if mode == "nearest-exact" {
                    arr.add_ctx(ctx, TensorAddArg::C(ConstVal::Float(0.5)))
                        .mul_ctx(ctx, TensorAddArg::C(ConstVal::Float(scale)))
                } else {
                    arr.mul_ctx(ctx, TensorAddArg::C(ConstVal::Float(scale)))
                };
                let index = scaled
                    .cast_ctx(ctx, crate::dtype::default_int())
                    .reshape_ctx(ctx, reshape.clone())
                    .expand_ctx(ctx, expand.clone());
                x = x.gather_ctx(ctx, dim as i64, &index);
            }
        }
        x.cast_ctx(ctx, self.dtype_ctx(ctx))
    }

    /// Tensor-level elementwise sugar for the oracle/schedule corpus.
    pub fn add(&self, other: impl Into<TensorAddArg>) -> Tensor {
        self.add_ctx(&mut self.ctx.borrow_mut(), other)
    }
    #[doc(hidden)]
    pub fn add_ctx(&self, ctx: &mut UOpCtx, other: impl Into<TensorAddArg>) -> Tensor {
        match other.into() {
            TensorAddArg::T(o) => {
                let __u = elementwise::binop(ctx, Ops::Add, self.uop().clone(), o.uop());
                Tensor::from_uop(ctx, __u)
            }
            TensorAddArg::C(c) => {
                let __u = elementwise::binop_c(ctx, Ops::Add, self.uop().clone(), c);
                Tensor::from_uop(ctx, __u)
            }
        }
    }
    /// `Tensor.mul(other)` — elementwise multiply by a tensor or scalar.
    pub fn mul(&self, other: impl Into<TensorAddArg>) -> Tensor {
        self.mul_ctx(&mut self.ctx.borrow_mut(), other)
    }
    #[doc(hidden)]
    pub fn mul_ctx(&self, ctx: &mut UOpCtx, other: impl Into<TensorAddArg>) -> Tensor {
        match other.into() {
            TensorAddArg::T(o) => {
                let __u = elementwise::binop(ctx, Ops::Mul, self.uop().clone(), o.uop());
                Tensor::from_uop(ctx, __u)
            }
            TensorAddArg::C(c) => {
                let __u = elementwise::binop_c(ctx, Ops::Mul, self.uop().clone(), c);
                Tensor::from_uop(ctx, __u)
            }
        }
    }
    /// `Tensor.neg()` — arithmetic negation (logical not for bools).
    pub fn neg(&self) -> Tensor {
        self.neg_ctx(&mut self.ctx.borrow_mut())
    }
    #[doc(hidden)]
    pub fn neg_ctx(&self, ctx: &mut UOpCtx) -> Tensor {
        {
            let __u = elementwise::neg(ctx, self.uop().clone());
            Tensor::from_uop(ctx, __u)
        }
    }

    /// `Tensor.contiguous()` — CONTIGUOUS node when not virtual.
    pub fn contiguous(&self) -> Tensor {
        self.contiguous_ctx(&mut self.ctx.borrow_mut())
    }
    #[doc(hidden)]
    pub fn contiguous_ctx(&self, ctx: &mut UOpCtx) -> Tensor {
        let dt = self.dtype_ctx(ctx);
        {
            let __u = ctx.uop(
                Ops::Contiguous,
                Some(dt),
                vec![self.uop().clone()],
                Arg::None,
            );
            Tensor::from_uop(ctx, __u)
        }
    }

    /// `Tensor.alu`: `u[0].alu(op, u[1:])` — a bare op ctor; promotion/broadcast happen
    /// in `dtype_from_uop` during construction.
    pub fn alu(&self, op: Ops, srcs: &[Tensor]) -> Tensor {
        self.alu_ctx(&mut self.ctx.borrow_mut(), op, srcs)
    }
    #[doc(hidden)]
    pub fn alu_ctx(&self, ctx: &mut UOpCtx, op: Ops, srcs: &[Tensor]) -> Tensor {
        self.apply_uop_ctx(ctx, srcs, |c, u| c.alu(op, u))
    }

    /// `Tensor.bitcast(dtype)` (`mixin/dtype.py:35`) — reinterpret the bits.
    pub fn bitcast(&self, dtype: DTypeId) -> Tensor {
        self.bitcast_ctx(&mut self.ctx.borrow_mut(), dtype)
    }
    #[doc(hidden)]
    pub fn bitcast_ctx(&self, ctx: &mut UOpCtx, dtype: DTypeId) -> Tensor {
        {
            let __u = ctx.bitcast(self.uop().clone(), dtype);
            Tensor::from_uop(ctx, __u)
        }
    }

    /// `bitwise_or` / `bitwise_and` / `bitwise_xor` — bare binops (`mixin/elementwise.py`).
    pub fn bitwise_or(&self, other: impl Into<TensorAddArg>) -> Tensor {
        self.bitwise_or_ctx(&mut self.ctx.borrow_mut(), other)
    }
    #[doc(hidden)]
    pub fn bitwise_or_ctx(&self, ctx: &mut UOpCtx, other: impl Into<TensorAddArg>) -> Tensor {
        match other.into() {
            TensorAddArg::T(o) => {
                let __u = elementwise::binop(ctx, Ops::Or, self.uop().clone(), o.uop());
                Tensor::from_uop(ctx, __u)
            }
            TensorAddArg::C(c) => {
                let __u = elementwise::binop_c(ctx, Ops::Or, self.uop().clone(), c);
                Tensor::from_uop(ctx, __u)
            }
        }
    }
    /// `Tensor.bitwise_and(other)` — elementwise and with a tensor or scalar.
    pub fn bitwise_and(&self, other: impl Into<TensorAddArg>) -> Tensor {
        self.bitwise_and_ctx(&mut self.ctx.borrow_mut(), other)
    }
    #[doc(hidden)]
    pub fn bitwise_and_ctx(&self, ctx: &mut UOpCtx, other: impl Into<TensorAddArg>) -> Tensor {
        match other.into() {
            TensorAddArg::T(o) => {
                let __u = elementwise::binop(ctx, Ops::And, self.uop().clone(), o.uop());
                Tensor::from_uop(ctx, __u)
            }
            TensorAddArg::C(c) => {
                let __u = elementwise::binop_c(ctx, Ops::And, self.uop().clone(), c);
                Tensor::from_uop(ctx, __u)
            }
        }
    }
    /// `Tensor.bitwise_xor(other)` — elementwise xor with a tensor or scalar.
    pub fn bitwise_xor(&self, other: impl Into<TensorAddArg>) -> Tensor {
        self.bitwise_xor_ctx(&mut self.ctx.borrow_mut(), other)
    }
    #[doc(hidden)]
    pub fn bitwise_xor_ctx(&self, ctx: &mut UOpCtx, other: impl Into<TensorAddArg>) -> Tensor {
        match other.into() {
            TensorAddArg::T(o) => {
                let __u = elementwise::binop(ctx, Ops::Xor, self.uop().clone(), o.uop());
                Tensor::from_uop(ctx, __u)
            }
            TensorAddArg::C(c) => {
                let __u = elementwise::binop_c(ctx, Ops::Xor, self.uop().clone(), c);
                Tensor::from_uop(ctx, __u)
            }
        }
    }

    /// `Tensor.shl(x)` / `Tensor.rshift(x)` — `self._binop(Ops.SHL/SHR, x)`.
    pub fn shl(&self, bits: i64) -> Tensor {
        self.shl_ctx(&mut self.ctx.borrow_mut(), bits)
    }
    #[doc(hidden)]
    pub fn shl_ctx(&self, ctx: &mut UOpCtx, bits: i64) -> Tensor {
        let b = elementwise::ufix(ctx, ConstVal::Int(bits as i128));
        {
            let __u = ctx.shl(self.uop().clone(), b);
            Tensor::from_uop(ctx, __u)
        }
    }
    /// `Tensor.rshift(bits)` — logical right shift by `bits`.
    pub fn rshift(&self, bits: i64) -> Tensor {
        self.rshift_ctx(&mut self.ctx.borrow_mut(), bits)
    }
    #[doc(hidden)]
    pub fn rshift_ctx(&self, ctx: &mut UOpCtx, bits: i64) -> Tensor {
        let b = elementwise::ufix(ctx, ConstVal::Int(bits as i128));
        {
            let __u = ctx.shr(self.uop().clone(), b);
            Tensor::from_uop(ctx, __u)
        }
    }

    /// `Tensor.cat(*others, dim=0)` (`mixin/op.py:734`): concatenate along `dim`.
    /// When every operand is the same size along `dim`, this is
    /// `stack(*args, dim).flatten(dim, dim+1)`; otherwise each is zero-padded to
    /// the total length and summed (the oracle's `padded[0].usum(*padded[1:])`,
    /// valid because the padded regions are disjoint).
    pub fn cat(&self, others: &[Tensor], dim: i64) -> Tensor {
        self.cat_ctx(&mut self.ctx.borrow_mut(), others, dim)
    }
    #[doc(hidden)]
    pub fn cat_ctx(&self, ctx: &mut UOpCtx, others: &[Tensor], dim: i64) -> Tensor {
        let mut tensors: Vec<Tensor> = Vec::with_capacity(others.len() + 1);
        tensors.push(self.clone());
        tensors.extend(others.iter().cloned());
        let self_shape: Vec<SVal> = self.shape_ctx(ctx).into_vec();
        let ndim = self_shape.len();
        let d = movement::resolve_dim(ndim, dim, false);
        for t in &tensors {
            let ts: Vec<SVal> = t.shape_ctx(ctx).into_vec();
            assert_eq!(ts.len(), ndim, "TGERR: cat: ndim mismatch");
            for i in 0..ndim {
                if i != d {
                    assert_eq!(
                        ts[i], self_shape[i],
                        "TGERR: cat: shapes differ on non-cat dim {i}"
                    );
                }
            }
        }
        let sizes: Vec<i64> = tensors
            .iter()
            .map(|t| t.shape_ctx(ctx).into_vec()[d].clone().as_i64().unwrap_or(0))
            .collect();
        if sizes.iter().all(|&s| s == sizes[0]) {
            let uops: Vec<UopId> = tensors.iter().map(|t| t.uop().clone()).collect();
            let st = {
                let __u = movement::stack_tensors(ctx, &uops, d as i64);
                Tensor::from_uop(ctx, __u)
            };
            return st.flatten_ctx(ctx, d as i64, d as i64 + 1);
        }
        let mut cum: Vec<i64> = vec![0];
        for &s in &sizes {
            cum.push(cum.last().unwrap() + s);
        }
        let total = *cum.last().unwrap();
        let mut out: Option<Tensor> = None;
        for (i, t) in tensors.iter().enumerate() {
            let mut arg: Vec<(i64, i64)> = Vec::with_capacity(ndim);
            for j in 0..ndim {
                arg.push(if j == d {
                    (cum[i], total - cum[i + 1])
                } else {
                    (0, 0)
                });
            }
            let p = t.pad_ctx(ctx, &arg);
            out = Some(match out {
                None => p,
                Some(acc) => acc.add_ctx(ctx, &p),
            });
        }
        out.unwrap()
    }

    /// `Tensor.tril(diagonal=0)` (op.py:259): lower-triangular part, else 0.
    pub fn tril(&self, diagonal: i64) -> Tensor {
        self.tril_ctx(&mut self.ctx.borrow_mut(), diagonal)
    }
    #[doc(hidden)]
    pub fn tril_ctx(&self, ctx: &mut UOpCtx, diagonal: i64) -> Tensor {
        creation::tril(ctx, self.uop().clone(), diagonal)
    }

    /// `Tensor.triu(diagonal=0)` (op.py:236): upper-triangular part, else 0.
    pub fn triu(&self, diagonal: i64) -> Tensor {
        self.triu_ctx(&mut self.ctx.borrow_mut(), diagonal)
    }
    #[doc(hidden)]
    pub fn triu_ctx(&self, ctx: &mut UOpCtx, diagonal: i64) -> Tensor {
        creation::triu(ctx, self.uop().clone(), diagonal)
    }

    /// `Tensor.clamp(min_=None, max_=None)` (elementwise.py:583): min/max bound.
    pub fn clamp(&self, min_: Option<f64>, max_: Option<f64>) -> Tensor {
        self.clamp_ctx(&mut self.ctx.borrow_mut(), min_, max_)
    }
    #[doc(hidden)]
    pub fn clamp_ctx(&self, ctx: &mut UOpCtx, min_: Option<f64>, max_: Option<f64>) -> Tensor {
        assert!(
            min_.is_some() || max_.is_some(),
            "TGERR: at least one of min_/max_ must be set"
        );
        let dt = self.dtype_ctx(ctx);
        let mk = |v: f64| {
            if crate::dtype::is_float(dt) {
                ConstVal::Float(v)
            } else {
                ConstVal::Int(v as i128)
            }
        };
        let mut ret = self.clone();
        if let Some(mn) = min_ {
            let mn_t = ret.const_like_ctx(ctx, mk(mn), None);
            ret = ret
                .lt_ctx(ctx, &mn_t)
                .contiguous_ctx(ctx)
                .where__ctx(ctx, &mn_t, &ret);
        }
        if let Some(mx) = max_ {
            let mx_t = ret.const_like_ctx(ctx, mk(mx), None);
            ret = ret
                .gt_ctx(ctx, &mx_t)
                .contiguous_ctx(ctx)
                .where__ctx(ctx, &mx_t, &ret);
        }
        ret
    }

    /// `Tensor.clip(min_=None, max_=None)` (elementwise.py:596): alias for `clamp`.
    pub fn clip(&self, min_: Option<f64>, max_: Option<f64>) -> Tensor {
        self.clip_ctx(&mut self.ctx.borrow_mut(), min_, max_)
    }
    #[doc(hidden)]
    pub fn clip_ctx(&self, ctx: &mut UOpCtx, min_: Option<f64>, max_: Option<f64>) -> Tensor {
        self.clamp_ctx(ctx, min_, max_)
    }

    /// `Tensor.lerp(end, weight)` (elementwise.py:1078): `self + (end - self)*weight`.
    /// (The uint8 integer path is not ported; only the float form is used by
    /// `interpolate`.)
    pub fn lerp(&self, end: &Tensor, weight: impl Into<TensorAddArg>) -> Tensor {
        self.lerp_ctx(&mut self.ctx.borrow_mut(), end, weight)
    }
    #[doc(hidden)]
    pub fn lerp_ctx(
        &self,
        ctx: &mut UOpCtx,
        end: &Tensor,
        weight: impl Into<TensorAddArg>,
    ) -> Tensor {
        let diff = end.sub_ctx(ctx, self);
        let scaled = diff.mul_ctx(ctx, weight);
        self.add_ctx(ctx, &scaled)
    }

    /// `Tensor.gather(dim, index)` (op.py:1026): select along `dim` by index.
    pub fn gather(&self, dim: i64, index: &Tensor) -> Tensor {
        self.gather_ctx(&mut self.ctx.borrow_mut(), dim, index)
    }
    #[doc(hidden)]
    pub fn gather_ctx(&self, ctx: &mut UOpCtx, dim: i64, index: &Tensor) -> Tensor {
        let self_shape: Vec<SVal> = self.shape_ctx(ctx).into_vec();
        let idx_shape: Vec<SVal> = index.shape_ctx(ctx).into_vec();
        let ndim = self_shape.len();
        assert_eq!(
            idx_shape.len(),
            ndim,
            "TGERR: gather: self.ndim must equal index.ndim"
        );
        let d = movement::resolve_dim(ndim, dim, false);
        let target: Vec<SVal> = idx_shape
            .iter()
            .enumerate()
            .map(|(i, s)| if i == d { SVal::I(-1) } else { s.clone() })
            .collect();
        let x = self
            .shrink_to_ctx(ctx, &target)
            .unsqueeze_ctx(ctx, -1)
            .transpose_ctx(ctx, -1, d as i64);
        let ncl = self_shape[d].clone().as_i64().unwrap_or(0);
        let oh = index
            .unsqueeze_ctx(ctx, -1)
            .one_hot_along_dim_ctx(ctx, ncl, -1);
        let zero = Tensor::full_const(&mut *ctx, &[] as &[i64], ConstVal::Float(0.0), None);
        let picked = oh.where__ctx(ctx, &x, zero);
        let last = picked.shape_ctx(ctx).len() - 1;
        picked.sum_ctx(ctx, Some(&[last]), false, Some(self.dtype_ctx(ctx)))
    }

    /// `Tensor.repeat_interleave(repeats, dim=None)` (movement.py:560).
    pub fn repeat_interleave(&self, repeats: i64, dim: Option<i64>) -> Tensor {
        self.repeat_interleave_ctx(&mut self.ctx.borrow_mut(), repeats, dim)
    }
    #[doc(hidden)]
    pub fn repeat_interleave_ctx(
        &self,
        ctx: &mut UOpCtx,
        repeats: i64,
        dim: Option<i64>,
    ) -> Tensor {
        let (x, d) = match dim {
            None => (self.flatten_ctx(ctx, 0, -1), 0usize),
            Some(dd) => {
                let ndim = self.shape_ctx(ctx).len();
                (self.clone(), movement::resolve_dim(ndim, dd, false))
            }
        };
        let shp: Vec<SVal> = x.shape_ctx(ctx).into_vec();
        let mut r1: Vec<SVal> = shp[..d + 1].to_vec();
        r1.push(SVal::I(1));
        r1.extend_from_slice(&shp[d + 1..]);
        let x = x.reshape_ctx(ctx, r1);
        let mut e: Vec<SVal> = shp[..d + 1].to_vec();
        e.push(SVal::I(repeats as i128));
        e.extend_from_slice(&shp[d + 1..]);
        let x = x.expand_ctx(ctx, e);
        let mut r2: Vec<SVal> = shp[..d].to_vec();
        r2.push(ctx.sval_mul(shp[d].clone(), SVal::I(repeats as i128)));
        r2.extend_from_slice(&shp[d + 1..]);
        x.reshape_ctx(ctx, r2)
    }

    /// `Tensor.threefry(seed)` (`mixin/elementwise.py:454`): `self.alu(Ops.THREEFRY, seed)`.
    pub fn threefry(&self, seed: &Tensor) -> Tensor {
        self.threefry_ctx(&mut self.ctx.borrow_mut(), seed)
    }
    #[doc(hidden)]
    pub fn threefry_ctx(&self, ctx: &mut UOpCtx, seed: &Tensor) -> Tensor {
        {
            let __u = ctx.uop(
                Ops::Threefry,
                Some(crate::dtype::UINT64),
                vec![self.uop().clone(), seed.uop().clone()],
                Arg::None,
            );
            Tensor::from_uop(ctx, __u)
        }
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

// ---------------------------------------------------------------------------
// std::ops sugar
// ---------------------------------------------------------------------------
//
// `Tensor` stores its `Ctx`, so arithmetic no longer needs a `&mut UOpCtx`.
// Both operands must come from the same arena; mixing contexts is a hard error
// (a silent cross-arena graph would intern nodes in the wrong graph).

macro_rules! impl_binop {
    ($trait:ident, $method:ident) => {
        impl std::ops::$trait<&Tensor> for &Tensor {
            type Output = Tensor;
            fn $method(self, rhs: &Tensor) -> Tensor {
                self.ctx().ensure_same(rhs.ctx());
                self.$method(rhs)
            }
        }
        impl std::ops::$trait<Tensor> for &Tensor {
            type Output = Tensor;
            fn $method(self, rhs: Tensor) -> Tensor {
                self.ctx().ensure_same(rhs.ctx());
                self.$method(&rhs)
            }
        }
        impl std::ops::$trait<&Tensor> for Tensor {
            type Output = Tensor;
            fn $method(self, rhs: &Tensor) -> Tensor {
                self.ctx().ensure_same(rhs.ctx());
                Tensor::$method(&self, rhs)
            }
        }
        impl std::ops::$trait<Tensor> for Tensor {
            type Output = Tensor;
            fn $method(self, rhs: Tensor) -> Tensor {
                self.ctx().ensure_same(rhs.ctx());
                Tensor::$method(&self, &rhs)
            }
        }
        impl std::ops::$trait<f64> for &Tensor {
            type Output = Tensor;
            fn $method(self, rhs: f64) -> Tensor {
                self.$method(rhs)
            }
        }
        impl std::ops::$trait<f64> for Tensor {
            type Output = Tensor;
            fn $method(self, rhs: f64) -> Tensor {
                Tensor::$method(&self, rhs)
            }
        }
    };
}

impl_binop!(Add, add);
impl_binop!(Sub, sub);
impl_binop!(Mul, mul);
impl_binop!(Div, div);

impl std::ops::Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        self.neg()
    }
}

impl std::ops::Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        Tensor::neg(&self)
    }
}
