//! `device.py` Buffer + allocator seams (M6, backend-trait split M8).
//!
//! `Buffer` mirrors python device.py:102 — a base allocation plus views
//! sharing it with a byte offset. `OpaqueT` is the backend-owned opaque
//! handle (python's untyped `Any`): each backend defines its own concrete
//! type and the core only sees `Rc<dyn OpaqueT>`. `HostOpaque` is the
//! built-in byte storage. The `initial_data` side table on UOpCtx plays the
//! role of python's `buffers` WeakKeyDictionary for opaquely-constructed
//! buffers (Tensor(data) → `Buffer(opaque=...)`); initial bytes are staged
//! in the base and copied in through the backend allocator on allocation,
//! so Tensor(data) works on any device.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

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
    fn as_any(&self) -> &dyn std::any::Any;
}

/// The built-in host byte storage plus a (offset, size) window — python's
/// memoryview slicing model (`_offset` returns a narrower window).
#[derive(Clone)]
pub struct HostOpaque {
    pub mem: Rc<RefCell<Vec<u8>>>,
    pub offset: usize,
    pub size: usize,
}

impl HostOpaque {
    pub fn new(mem: Vec<u8>) -> HostOpaque {
        let size = mem.len();
        HostOpaque { mem: Rc::new(RefCell::new(mem)), offset: 0, size }
    }
    pub fn read(&self) -> Vec<u8> {
        self.mem.borrow()[self.offset..self.offset + self.size].to_vec()
    }
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
    o.as_any().downcast_ref::<HostOpaque>().expect("TGERR: expected host opaque")
}

/// An owned opaque view (`Rc<dyn OpaqueT>`) — `Buffer._buf`.
pub type OpaqueView = Rc<dyn OpaqueT>;

/// The base `Buffer` (python device.py:102): one allocation per base.
pub struct BufferBase {
    pub device: String,
    pub size: usize,
    pub dtype: DTypeId,
    pub buf: RefCell<Option<OpaqueView>>,
    /// Tensor(data) staged bytes, copied in through the backend allocator on
    /// first allocation (python: opaque= host path generalized to devices).
    pub pending_copyin: RefCell<Option<Vec<u8>>>,
    pub uop_refcount: Cell<i64>,
}

impl BufferBase {
    pub fn nbytes(&self) -> usize {
        self.size * self.dtype.itemsize()
    }
    pub fn is_allocated(&self) -> bool {
        self.buf.borrow().is_some()
    }
    pub fn allocate_with(&self, o: OpaqueView) {
        assert!(!self.is_allocated(), "TGERR: can't allocate already allocated buffer");
        if let Some(bytes) = self.pending_copyin.borrow_mut().take() {
            assert_eq!(bytes.len(), o.size(), "TGERR: initial data size mismatch");
            // host fast path keeps byte storage shared with ctx.initial_data
            if let Some(h) = o.as_any().downcast_ref::<HostOpaque>() {
                if h.offset == 0 && h.size == bytes.len() {
                    *h.mem.borrow_mut() = bytes;
                } else {
                    h.write(&bytes);
                }
            } else {
                panic!("TGERR: pending copyin requires a device — allocate through ensure_allocated");
            }
        }
        *self.buf.borrow_mut() = Some(o);
    }
    pub fn deallocate(&self) {
        assert!(self.is_allocated(), "TGERR: buffer must be allocated to deallocate");
        *self.buf.borrow_mut() = None;
    }
}

/// A (possibly view) `Buffer`: shares the base, carries a byte offset.
#[derive(Clone)]
pub struct Buffer {
    pub base: Rc<BufferBase>,
    pub size: usize,
    pub dtype: DTypeId,
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
        Buffer { base, size, dtype, offset: 0 }
    }

    pub fn nbytes(&self) -> usize {
        self.size * self.dtype.itemsize()
    }
    pub fn is_allocated(&self) -> bool {
        self.base.is_allocated()
    }
    pub fn device(&self) -> &str {
        &self.base.device
    }

    /// `Buffer.view(size, dtype, offset)` — offset in bytes.
    pub fn view(&self, size: usize, dtype: DTypeId, offset: usize) -> Buffer {
        assert!(offset < self.nbytes(), "TGERR: offset must be less than nbytes");
        Buffer { base: self.base.clone(), size, dtype, offset: self.offset + offset }
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
        assert!(!self.is_allocated(), "TGERR: can't allocate already allocated buffer");
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
#[derive(Default)]
pub struct BufTable {
    bases: HashMap<u32, Rc<BufferBase>>,
    views: HashMap<u32, Buffer>,
}

/// `UOp.buffer` (ops.py:922): resolve a (post-schedule) buffer-ish uop to its
/// engine Buffer. Base BUFFER uops create the Buffer (picking up
/// `ctx.initial_data` for opaque-constructed storage); movement/bitcast views
/// resolve to `base.view(size, dtype, byte_offset)`.
pub fn buffer_of(ctx: &mut UOpCtx, t: &mut BufTable, u: UopId) -> Buffer {
    match ctx.op_of(u) {
        Ops::Contiguous | Ops::Reshape | Ops::After | Ops::Cast | Ops::Index | Ops::Mselect => {
            let s0 = ctx.srcs(u)[0];
            buffer_of(ctx, t, s0)
        }
        Ops::Buffer => {
            if let Some(b) = t.views.get(&u.0) {
                return b.clone();
            }
            let pa: crate::uop::arg::ParamArg = match ctx.arg_of(u) {
                Arg::Param(p) => p.clone(),
                other => panic!("TGERR: BUFFER arg {other:?}"),
            };
            let dev = match &pa.device {
                Some(DeviceSpec::Single(s)) => s.clone(),
                Some(DeviceSpec::Multi(v)) => v[0].clone(),
                None => "CPU".to_string(),
            };
            let size = ctx.max_numel(u) as usize;
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
            if let Some(bytes) = ctx.initial_data.get(&u.0) {
                let nb = size * dt.itemsize();
                assert!(bytes.borrow().len() == nb, "TGERR: initial data size mismatch");
                *base.pending_copyin.borrow_mut() = Some(bytes.borrow().clone());
            }
            let b = Buffer { base: base.clone(), size, dtype: dt, offset: 0 };
            t.bases.insert(u.0, base);
            t.views.insert(u.0, b.clone());
            b
        }
        Ops::Bitcast => {
            if let Some(b) = t.views.get(&u.0) {
                return b.clone();
            }
            let s0 = ctx.srcs(u)[0];
            let inner = buffer_of(ctx, t, s0);
            let (n, dt) = (ctx.max_numel(u) as usize, ctx.dt_of(u));
            let b = inner.view(n, dt, 0);
            t.views.insert(u.0, b.clone());
            b
        }
        Ops::Shrink => {
            if let Some(b) = t.views.get(&u.0) {
                return b.clone();
            }
            let src = ctx.srcs(u)[0];
            let inner = buffer_of(ctx, t, src);
            let start = match ctx.val(ctx.srcs(u)[1]) {
                crate::dtype::ConstVal::Int(v) => v as usize,
                other => panic!("TGERR: view SHRINK start {other:?}"),
            };
            // SHRINK args are elements of the inner dtype
            let byte_off = start * ctx.dt_of(src).itemsize();
            let (n, dt) = (ctx.max_numel(u) as usize, ctx.dt_of(u));
            let b = inner.view(n, dt, byte_off);
            t.views.insert(u.0, b.clone());
            b
        }
        other => panic!("TGERR: buffer_of op {other:?}"),
    }
}
