//! `device.py` Buffer + allocator seams (M6).
//!
//! `Buffer` mirrors python device.py:102 — a base allocation plus views
//! sharing it with a byte offset. `Opaque` is the device-side handle (a byte
//! storage window); M8 adds the OpenCL variant. The `initial_data` side table
//! on UOpCtx plays the role of python's `buffers` WeakKeyDictionary for
//! opaquely-constructed buffers (Tensor(data) → `Buffer(opaque=...)`).

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

/// Opaque device handle: a shared byte storage plus a (offset, size) window —
/// python's memoryview slicing model (`_offset` returns a narrower window).
#[derive(Clone)]
pub struct Opaque {
    pub mem: Rc<RefCell<Vec<u8>>>,
    pub offset: usize,
    pub size: usize,
}

impl Opaque {
    pub fn read(&self) -> Vec<u8> {
        self.mem.borrow()[self.offset..self.offset + self.size].to_vec()
    }
    pub fn write(&self, data: &[u8]) {
        assert!(data.len() == self.size, "TGERR: opaque write size mismatch");
        self.mem.borrow_mut()[self.offset..self.offset + self.size].copy_from_slice(data);
    }
}

/// The base `Buffer` (python device.py:102): one allocation per base.
pub struct BufferBase {
    pub device: String,
    pub size: usize,
    pub dtype: DTypeId,
    pub buf: RefCell<Option<Opaque>>,
    pub uop_refcount: Cell<i64>,
}

impl BufferBase {
    pub fn nbytes(&self) -> usize {
        self.size * self.dtype.itemsize()
    }
    pub fn is_allocated(&self) -> bool {
        self.buf.borrow().is_some()
    }
    pub fn allocate_with(&self, o: Opaque) {
        assert!(!self.is_allocated(), "TGERR: can't allocate already allocated buffer");
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
        let base = Rc::new(BufferBase { device: device.to_string(), size, dtype, buf: RefCell::new(None), uop_refcount: Cell::new(0) });
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
        if !self.is_allocated() {
            self.allocate(dev, None);
        }
    }

    /// `Buffer.allocate` — through the device LRU allocator (or an explicit
    /// opaque, the `opaque=` ctor path used by Tensor(data)).
    pub fn allocate(&self, dev: &Compiled, opaque: Option<Opaque>) {
        assert!(!self.is_allocated(), "TGERR: can't allocate already allocated buffer");
        match opaque {
            Some(o) => self.base.allocate_with(o),
            None => self.base.allocate_with(dev.allocator.alloc(self.base.nbytes())),
        }
    }

    /// `Buffer.get_buf(device)` — the opaque window for kernel args.
    /// Single-device corpus: device == self.device; views go through `_offset`.
    pub fn get_buf(&self, dev: &Compiled) -> Opaque {
        self.ensure_allocated(dev);
        let b = self.base.buf.borrow().as_ref().unwrap().clone();
        let nb = self.nbytes();
        if self.offset == 0 && nb == b.size {
            b
        } else {
            dev.allocator.offset(&b, nb, self.offset)
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
                uop_refcount: Cell::new(1),
            });
            // Tensor(data) opaque path: initial bytes come with the buffer.
            if let Some(bytes) = ctx.initial_data.get(&u.0) {
                let nb = size * dt.itemsize();
                assert!(bytes.borrow().len() == nb, "TGERR: initial data size mismatch");
                base.allocate_with(Opaque { mem: bytes.clone(), offset: 0, size: nb });
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
