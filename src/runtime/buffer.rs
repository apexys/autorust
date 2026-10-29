//! `device.py` Buffer + allocator seams (M6, backend-trait split M8).
//!
//! `Buffer` mirrors python device.py:102 — a base allocation plus views
//! sharing it with a byte offset. `OpaqueT` is the backend-owned opaque
//! handle (python's untyped `Any`): each backend defines its own concrete
//! type and the core only sees `Rc<dyn OpaqueT>`. `HostOpaque` is the
//! built-in byte storage. Staged host bytes for opaquely-constructed buffers
//! (Tensor(data) → `Buffer(opaque=...)`) live on the buffer's node
//! (`Node.initial_data`, python's `buffers` WeakKeyDictionary analogue); they
//! are staged in the base and copied in through the backend allocator on
//! allocation, so Tensor(data) works on any device.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::{Rc, Weak};

use crate::dtype::DTypeId;
use crate::uop::arg::{Arg, DeviceSpec};
use crate::uop::ctx::UOpCtx;
use crate::uop::graph::UopId;
use crate::uop::ops::Ops;

use super::device::Compiled;

/// `BufferSpec` — trivial defaults for the interpreter devices (M8 adds
/// nolru/zero/external_ptr for real backends).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct BufferSpec {}

/// The backend-owned opaque handle (python `Allocator._alloc` return type —
/// an untyped `Any` passed from the allocator to `Program.__call__`).
pub trait OpaqueT: 'static {
    /// Window size in bytes.
    fn size(&self) -> usize;
    /// Identity of the underlying storage (same allocation => same id).
    fn storage_id(&self) -> usize;
    /// Upcast to `Any` for backend-specific downcasting.
    fn as_any(&self) -> &dyn std::any::Any;
}

/// The built-in host byte storage plus a (offset, size) window — python's
/// memoryview slicing model (`_offset` returns a narrower window).
#[derive(Clone)]
pub struct HostOpaque {
    /// Shared host byte storage backing the base allocation.
    pub mem: Rc<RefCell<Vec<u8>>>,
    /// Window start in bytes within `mem`.
    pub offset: usize,
    /// Window length in bytes.
    pub size: usize,
}

impl HostOpaque {
    /// Wrap `mem` as host storage with a full-size window.
    pub fn new(mem: Vec<u8>) -> HostOpaque {
        let size = mem.len();
        HostOpaque {
            mem: Rc::new(RefCell::new(mem)),
            offset: 0,
            size,
        }
    }
    /// Copy the current window out as a fresh byte vector.
    pub fn read(&self) -> Vec<u8> {
        self.mem.borrow()[self.offset..self.offset + self.size].to_vec()
    }
    /// Overwrite the current window with `data`, which must match its size.
    pub fn write(&self, data: &[u8]) {
        assert!(data.len() == self.size, "TGERR: opaque write size mismatch");
        self.mem.borrow_mut()[self.offset..self.offset + self.size].copy_from_slice(data);
    }
}

impl OpaqueT for HostOpaque {
    fn size(&self) -> usize {
        self.size
    }
    fn storage_id(&self) -> usize {
        Rc::as_ptr(&self.mem) as *const u8 as usize
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Downcast to the host storage (the interp backend only runs on host bytes).
pub fn host_of(o: &dyn OpaqueT) -> &HostOpaque {
    o.as_any()
        .downcast_ref::<HostOpaque>()
        .expect("TGERR: expected host opaque")
}

/// An owned opaque view (`Rc<dyn OpaqueT>`) — `Buffer._buf`.
pub type OpaqueView = Rc<dyn OpaqueT>;

/// The base `Buffer` (python device.py:102): one allocation per base.
pub struct BufferBase {
    /// Backend device string this buffer lives on.
    pub device: String,
    /// Number of elements in the base allocation.
    pub size: usize,
    /// Element dtype of the base allocation.
    pub dtype: DTypeId,
    /// Allocated opaque storage, `None` until first allocation.
    pub buf: RefCell<Option<OpaqueView>>,
    /// Tensor(data) staged bytes, copied in through the backend allocator on
    /// first allocation (python: opaque= host path generalized to devices).
    pub pending_copyin: RefCell<Option<Vec<u8>>>,
    /// Scheduler refcount, python's `Buffer.uop_refcount`.
    pub uop_refcount: Cell<i64>,
}

impl BufferBase {
    /// Size of the base allocation in bytes.
    pub fn nbytes(&self) -> usize {
        self.size * self.dtype.itemsize()
    }
    /// Whether the base has been allocated by the backend.
    pub fn is_allocated(&self) -> bool {
        self.buf.borrow().is_some()
    }
    /// Install `o` as the backing storage, applying any pending Tensor(data) copy-in.
    pub fn allocate_with(&self, o: OpaqueView) {
        assert!(
            !self.is_allocated(),
            "TGERR: can't allocate already allocated buffer"
        );
        if let Some(bytes) = self.pending_copyin.borrow_mut().take() {
            assert_eq!(bytes.len(), o.size(), "TGERR: initial data size mismatch");
            // host fast path keeps byte storage shared with Node.initial_data
            if let Some(h) = o.as_any().downcast_ref::<HostOpaque>() {
                if h.offset == 0 && h.size == bytes.len() {
                    *h.mem.borrow_mut() = bytes;
                } else {
                    h.write(&bytes);
                }
            } else {
                panic!(
                    "TGERR: pending copyin requires a device — allocate through ensure_allocated"
                );
            }
        }
        *self.buf.borrow_mut() = Some(o);
    }
    /// Drop the backing storage; the caller ensures there are no live users.
    pub fn deallocate(&self) {
        assert!(
            self.is_allocated(),
            "TGERR: buffer must be allocated to deallocate"
        );
        *self.buf.borrow_mut() = None;
    }
}

/// A (possibly view) `Buffer`: shares the base, carries a byte offset.
#[derive(Clone)]
pub struct Buffer {
    /// Shared base allocation.
    pub base: Rc<BufferBase>,
    /// Number of elements in this view.
    pub size: usize,
    /// Element dtype of this view.
    pub dtype: DTypeId,
    /// Byte offset of this view within the base allocation.
    pub offset: usize,
}

impl Buffer {
    /// `Buffer(device, size, dtype)` — a fresh base.
    pub fn new_base(device: &str, size: usize, dtype: DTypeId) -> Buffer {
        let base = Rc::new(BufferBase {
            device: device.to_string(),
            size,
            dtype,
            buf: RefCell::new(None),
            pending_copyin: RefCell::new(None),
            uop_refcount: Cell::new(0),
        });
        Buffer {
            base,
            size,
            dtype,
            offset: 0,
        }
    }

    /// Size of this view in bytes.
    pub fn nbytes(&self) -> usize {
        self.size * self.dtype.itemsize()
    }
    /// Whether the shared base has been allocated.
    pub fn is_allocated(&self) -> bool {
        self.base.is_allocated()
    }
    /// Device string of the shared base.
    pub fn device(&self) -> &str {
        &self.base.device
    }

    /// `Buffer.view(size, dtype, offset)` — offset in bytes.
    pub fn view(&self, size: usize, dtype: DTypeId, offset: usize) -> Buffer {
        assert!(
            offset < self.nbytes(),
            "TGERR: offset must be less than nbytes"
        );
        Buffer {
            base: self.base.clone(),
            size,
            dtype,
            offset: self.offset + offset,
        }
    }

    /// `Buffer.ensure_allocated` — allocate on the device if needed.
    pub fn ensure_allocated(&self, dev: &Compiled) {
        if self.is_allocated() {
            return;
        }
        let o = dev.allocator.alloc(self.base.nbytes());
        if let Some(bytes) = self.base.pending_copyin.borrow_mut().take() {
            assert_eq!(bytes.len(), o.size(), "TGERR: initial data size mismatch");
            dev.allocator._copyin(&*o, &bytes);
        }
        *self.base.buf.borrow_mut() = Some(o);
    }

    /// `Buffer.allocate` — through the device LRU allocator (or an explicit
    /// opaque, the `opaque=` ctor path).
    pub fn allocate(&self, dev: &Compiled, opaque: Option<OpaqueView>) {
        assert!(
            !self.is_allocated(),
            "TGERR: can't allocate already allocated buffer"
        );
        match opaque {
            Some(o) => self.base.allocate_with(o),
            None => {
                let o = dev.allocator.alloc(self.base.nbytes());
                if let Some(bytes) = self.base.pending_copyin.borrow_mut().take() {
                    dev.allocator._copyin(&*o, &bytes);
                }
                *self.base.buf.borrow_mut() = Some(o);
            }
        }
    }

    /// `Buffer.get_buf(device)` — the opaque window for kernel args.
    /// Single-device corpus: device == self.device; views go through `_offset`.
    pub fn get_buf(&self, dev: &Compiled) -> OpaqueView {
        self.ensure_allocated(dev);
        let b = self.base.buf.borrow().as_ref().unwrap().clone();
        let nb = self.nbytes();
        if b.size() == nb && self.offset == 0 {
            b
        } else {
            dev.buf_view(&*b, nb, self.offset)
        }
    }
}

/// The engine's `buffers` dict: BUFFER uops (and views) → engine Buffers.
///
/// `bases` holds the strong reference that keeps a realized allocation alive for
/// the duration of a realize; `views` are weak handles rebuilt on access.
/// [`BufTable::prune`] (called at the start of each realize) drops any base no
/// live caller (e.g. a `Tensor`-owned buffer) still references, keeping the table
/// bounded across repeated computation.
#[derive(Default)]
pub struct BufTable {
    bases: HashMap<u32, Rc<BufferBase>>,
    views: HashMap<u32, WeakBuffer>,
}

/// A non-owning `Buffer` handle (base + window metadata).
#[derive(Clone)]
struct WeakBuffer {
    base: Weak<BufferBase>,
    size: usize,
    dtype: DTypeId,
    offset: usize,
}

impl WeakBuffer {
    fn of(b: &Buffer) -> WeakBuffer {
        WeakBuffer {
            base: Rc::downgrade(&b.base),
            size: b.size,
            dtype: b.dtype,
            offset: b.offset,
        }
    }
    fn upgrade(&self) -> Option<Buffer> {
        self.base.upgrade().map(|base| Buffer {
            base,
            size: self.size,
            dtype: self.dtype,
            offset: self.offset,
        })
    }
}

impl BufTable {
    /// Number of base allocations still referenced by a live caller (leak
    /// observability: a repeated realize that reuses storage must not grow).
    pub fn n_bases(&self) -> usize {
        self.bases
            .values()
            .filter(|b| Rc::strong_count(b) > 1)
            .count()
    }
    /// Number of cached view entries (uop id → weak `Buffer`).
    pub fn n_views(&self) -> usize {
        self.views.len()
    }
    /// Drop bases (and their views) with no reference outside the table.
    pub fn prune(&mut self) {
        self.bases.retain(|_, b| Rc::strong_count(b) > 1);
        self.views.retain(|_, v| v.base.strong_count() > 0);
    }

    /// Register `b` as the storage for the buffer/view uop `uop_id`, so a realized
    /// tensor's `uop` can be a plain leaf that reads this allocation. No data moves.
    ///
    /// Only the (weak) view entry is inserted: the allocation is kept alive by the
    /// tensor's own buffer cell (and the producing uop's table entry), and `prune`
    /// can then reclaim both once the tensor dies. A strong `bases` entry here would
    /// keep `strong_count > 1` forever and leak one base per realize.
    pub fn alias(&mut self, uop_id: u32, b: &Buffer) {
        self.views.insert(uop_id, WeakBuffer::of(b));
    }
}

/// `UOp.buffer` (ops.py:922): resolve a (post-schedule) buffer-ish uop to its
/// engine Buffer. Base BUFFER uops create the Buffer (picking up
/// `Node.initial_data` for opaque-constructed storage); movement/bitcast views
/// resolve to `base.view(size, dtype, byte_offset)`.
pub fn buffer_of(ctx: &mut UOpCtx, t: &mut BufTable, u: UopId) -> Buffer {
    match ctx.op_of(&u) {
        Ops::Contiguous
        | Ops::Reshape
        | Ops::After
        | Ops::Cast
        | Ops::Index
        | Ops::Mselect
        | Ops::Permute
        | Ops::Pad
        | Ops::Flip
        | Ops::Expand => {
            // view/movement ops share the base buffer; the reshape/permute/pad/
            // expand/flip mapping is applied through the kernel's index math, so
            // storage resolution just walks to src[0]'s buffer.
            let s0 = &ctx.srcs(&u)[0].clone();
            buffer_of(ctx, t, s0.clone())
        }
        Ops::Buffer => {
            if let Some(b) = t.views.get(&u.id()).and_then(|v| v.upgrade()) {
                return b;
            }
            let pa: crate::uop::arg::ParamArg = match ctx.arg_of(&u) {
                Arg::Param(p) => p.clone(),
                other => panic!("TGERR: BUFFER arg {other:?}"),
            };
            let dev = match &pa.device {
                Some(DeviceSpec::Single(s)) => s.clone(),
                Some(DeviceSpec::Multi(v)) => v[0].clone(),
                None => "CPU".to_string(),
            };
            let size = ctx.max_numel(u.clone()) as usize;
            let dt = pa.dtype;
            let base = Rc::new(BufferBase {
                device: dev,
                size,
                dtype: dt,
                buf: RefCell::new(None),
                pending_copyin: RefCell::new(None),
                uop_refcount: Cell::new(1),
            });
            // Tensor(data) path: initial bytes staged, copied in on allocation
            // (through the backend allocator, so any device works).
            if let Some(bytes) = u.node.initial_data.borrow().as_ref() {
                let nb = size * dt.itemsize();
                assert!(
                    bytes.borrow().len() == nb,
                    "TGERR: initial data size mismatch"
                );
                *base.pending_copyin.borrow_mut() = Some(bytes.borrow().clone());
            }
            let b = Buffer {
                base: base.clone(),
                size,
                dtype: dt,
                offset: 0,
            };
            t.bases.insert(u.id(), base);
            t.views.insert(u.id(), WeakBuffer::of(&b));
            b
        }
        Ops::Bitcast => {
            if let Some(b) = t.views.get(&u.id()).and_then(|v| v.upgrade()) {
                return b;
            }
            let s0 = &ctx.srcs(&u)[0].clone();
            let inner = buffer_of(ctx, t, s0.clone());
            let (n, dt) = (ctx.max_numel(u.clone()) as usize, ctx.dt_of(&u));
            let b = inner.view(n, dt, 0);
            t.views.insert(u.id(), WeakBuffer::of(&b));
            b
        }
        Ops::Copy => {
            // single-device corpus: a COPY aliases its source's storage (the
            // copy-in/copy-out through the allocator keeps bytes in sync).
            let s0 = &ctx.srcs(&u)[0].clone();
            buffer_of(ctx, t, s0.clone())
        }
        Ops::Shrink => {
            if let Some(b) = t.views.get(&u.id()).and_then(|v| v.upgrade()) {
                return b;
            }
            let src = &ctx.srcs(&u)[0].clone();
            let inner = buffer_of(ctx, t, src.clone());
            let start = match ctx.val(&ctx.srcs(&u)[1]) {
                crate::dtype::ConstVal::Int(v) => v as usize,
                other => panic!("TGERR: view SHRINK start {other:?}"),
            };
            // SHRINK args are elements of the inner dtype
            let byte_off = start * ctx.dt_of(src).itemsize();
            let (n, dt) = (ctx.max_numel(u.clone()) as usize, ctx.dt_of(&u));
            let b = inner.view(n, dt, byte_off);
            t.views.insert(u.id(), WeakBuffer::of(&b));
            b
        }
        other => panic!("TGERR: buffer_of op {other:?}"),
    }
}
