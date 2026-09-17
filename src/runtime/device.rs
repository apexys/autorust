//! `device.py` device seams: `Allocator` + `LRUAllocator` (over the public
//! `BackendAllocator` trait), `Compiler`, `TinyELF`, `Program`, `Compiled`,
//! and the `Device` registry.
//!
//! A user backend implements `OpaqueT`, `BackendAllocator`, `UopRenderer`,
//! `Comp` and `Prog`, builds a `Compiled` via `Compiled::new`, and registers
//! it with `register_device` — mirroring python's `Device._DeviceMap[name]`
//! assignment. The built-in interp backend (`ops_interp`) registers itself
//! through the same path.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

use crate::renderer::UopRenderer;

use super::buffer::OpaqueT;
// ---------------- Allocator ----------------

/// The runtime-implemented allocator methods (python `Allocator._alloc` etc).
/// Opaques are backend-owned (`OpaqueT` trait objects); the default host
/// implementation is `MemAllocator`.
pub trait BackendAllocator {
    fn _alloc(&self, size: usize) -> Rc<dyn OpaqueT>;
    fn _free(&self, o: Rc<dyn OpaqueT>);
    fn _copyin(&self, dest: &dyn OpaqueT, src: &[u8]);
    fn _copyout(&self, dest: &mut [u8], src: &dyn OpaqueT);
    /// `Allocator._offset` — slice the window (memoryview semantics).
    fn _offset(&self, buf: &dyn OpaqueT, size: usize, offset: usize) -> Rc<dyn OpaqueT>;
    /// Direct device-to-device copy `src` → `dest` (same device; python
    /// `clEnqueueCopyBuffer`). The default falls back to a host round-trip
    /// (`_copyout` + `_copyin`), so a backend need not implement it.
    fn _copy_into(&self, dest: &dyn OpaqueT, src: &dyn OpaqueT) {
        let mut tmp = vec![0u8; src.size()];
        self._copyout(&mut tmp, src);
        self._copyin(dest, &tmp);
    }
}

/// `PythonAllocator` / the interp allocator: plain host memory.
pub struct MemAllocator;
impl BackendAllocator for MemAllocator {
    fn _alloc(&self, size: usize) -> Rc<dyn OpaqueT> {
        Rc::new(super::buffer::HostOpaque::new(vec![0u8; size]))
    }
    fn _free(&self, _o: Rc<dyn OpaqueT>) {}
    fn _copyin(&self, dest: &dyn OpaqueT, src: &[u8]) {
        super::buffer::host_of(dest).write(src);
    }
    fn _copyout(&self, dest: &mut [u8], src: &dyn OpaqueT) {
        dest.copy_from_slice(&super::buffer::host_of(src).read());
    }
    fn _offset(&self, buf: &dyn OpaqueT, size: usize, offset: usize) -> Rc<dyn OpaqueT> {
        let b = super::buffer::host_of(buf);
        assert!(offset + size <= b.size, "TGERR: offset out of range");
        Rc::new(super::buffer::HostOpaque { mem: b.mem.clone(), offset: b.offset + offset, size })
    }
    /// Host memory is already the "device" memory: copy the windows directly.
    fn _copy_into(&self, dest: &dyn OpaqueT, src: &dyn OpaqueT) {
        let d = super::buffer::host_of(dest);
        let s = super::buffer::host_of(src);
        assert_eq!(d.size, s.size, "TGERR: copy size mismatch");
        let bytes = s.mem.borrow()[s.offset..s.offset + s.size].to_vec();
        d.mem.borrow_mut()[d.offset..d.offset + d.size].copy_from_slice(&bytes);
    }
}

/// `LRUAllocator` (device.py:276) — the (size, options)-keyed free-list cache
/// with OOM-triggered `free_cache()` retry. Perf-critical for real backends
/// (every uncached alloc is a clCreateBuffer in M8).
pub struct LruAllocator {
    inner: Box<dyn BackendAllocator>,
    cache: RefCell<HashMap<(usize, super::buffer::BufferSpec), Vec<Rc<dyn OpaqueT>>>>,
}

impl LruAllocator {
    /// The host default (`MemAllocator`-backed).
    pub fn new() -> Self {
        LruAllocator { inner: Box::new(MemAllocator), cache: RefCell::new(HashMap::new()) }
    }

    /// Wrap any backend allocator (`CLAllocator` etc).
    pub fn with_backend(inner: Box<dyn BackendAllocator>) -> Self {
        LruAllocator { inner, cache: RefCell::new(HashMap::new()) }
    }

    pub fn alloc(&self, size: usize) -> Rc<dyn OpaqueT> {
        assert!(size > 0, "TGERR: alloc size must be positive");
        if let Some(o) = self.cache.borrow_mut().get_mut(&(size, Default::default())).and_then(|v| v.pop()) {
            return o;
        }
        // OOM retry path: our host allocator only fails on OOM, same retry
        self.inner._alloc(size)
    }

    /// `LRUAllocator.free` — push to the cache (LRU on by default).
    pub fn free(&self, opaque: Rc<dyn OpaqueT>, size: usize) {
        self.cache.borrow_mut().entry((size, Default::default())).or_default().push(opaque);
    }

    /// `LRUAllocator.free_cache` — drop every cached opaque.
    pub fn free_cache(&self) {
        let mut cache = self.cache.borrow_mut();
        for (_, opaques) in cache.drain() {
            for o in opaques {
                self.inner._free(o);
            }
        }
    }

    pub fn _copyin(&self, dest: &dyn OpaqueT, src: &[u8]) {
        self.inner._copyin(dest, src)
    }
    pub fn _copyout(&self, dest: &mut [u8], src: &dyn OpaqueT) {
        self.inner._copyout(dest, src)
    }
    pub fn _offset(&self, buf: &dyn OpaqueT, size: usize, offset: usize) -> Rc<dyn OpaqueT> {
        self.inner._offset(buf, size, offset)
    }
    pub fn _copy_into(&self, dest: &dyn OpaqueT, src: &dyn OpaqueT) {
        self.inner._copy_into(dest, src)
    }
}

impl Default for LruAllocator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------- Compiler ----------------

/// `Compiler` — the per-backend compiler (`compile(src) -> lib bytes`).
/// The empty default is `src.encode()` (device.py:306). Disk caching is out
/// of scope; `compile_cached` keeps an in-memory map.
pub trait Comp {
    fn compile(&self, src: &str) -> Vec<u8> {
        src.as_bytes().to_vec()
    }
    fn compile_cached(&self, src: &str) -> Vec<u8>;
}

/// The empty compiler default.
#[derive(Default)]
pub struct IdentityCompiler {
    cache: RefCell<HashMap<String, Vec<u8>>>,
}
impl Comp for IdentityCompiler {
    fn compile_cached(&self, src: &str) -> Vec<u8> {
        if let Some(lib) = self.cache.borrow().get(src) {
            return lib.clone();
        }
        let lib = self.compile(src);
        self.cache.borrow_mut().insert(src.to_string(), lib.clone());
        lib
    }
}

// ---------------- Program (runtime) ----------------

/// `TinyELF` (device.py) — the compiled artifact seam: lib bytes, function
/// name, and the buffer signature (slot, dtype) in linear order. lib holds
/// whatever `Compiler.compile` returned (pickle bytes for interp, source
/// text for NVIDIA CL).
pub struct TinyElf {
    pub lib: Vec<u8>,
    pub name: String,
    pub signature: Vec<(i64, crate::dtype::DTypeId)>,
}

/// `Program.__call__(*bufs, global_size, local_size, vals, wait)` — the
/// runtime launch seam. Returns elapsed seconds when `wait`.
pub trait Prog {
    fn call(&mut self, bufs: &[Rc<dyn OpaqueT>], global_size: [usize; 3], local_size: Option<[usize; 3]>, vals: &[i64], wait: bool) -> Option<f64>;
}

// ---------------- Compiled device ----------------

/// `Compiled` (device.py:343) — renderer + allocator + compiler + runtime
/// factory, the four public seams every backend plugs into.
pub struct Compiled {
    pub device: String,
    pub allocator: LruAllocator,
    pub renderer: Box<dyn UopRenderer>,
    pub compiler: Rc<dyn Comp>,
    runtime_ctor: Rc<dyn Fn(&TinyElf) -> Rc<RefCell<dyn Prog>>>,
    compiles: Cell<usize>,
    runtime_cache: RefCell<HashMap<u64, Rc<RefCell<dyn Prog>>>>,
}

impl Compiled {
    /// Compose a device from its public seams (python `Compiled.__init__`).
    pub fn new(
        device: &str,
        renderer: Box<dyn UopRenderer>,
        allocator: LruAllocator,
        compiler: Rc<dyn Comp>,
        runtime_ctor: Rc<dyn Fn(&TinyElf) -> Rc<RefCell<dyn Prog>>>,
    ) -> Rc<Compiled> {
        Rc::new(Compiled {
            device: device.to_string(),
            allocator,
            renderer,
            compiler,
            runtime_ctor,
            compiles: Cell::new(0),
            runtime_cache: RefCell::new(HashMap::new()),
        })
    }

    /// `Device[device].runtime(elf)` — construct (or fetch cached) the runtime
    /// for a compiled artifact. Cache key = lib hash (python: (ast.key, device)).
    pub fn runtime(self: &Rc<Self>, elf: TinyElf) -> Rc<RefCell<dyn Prog>> {
        let key = elf.lib.iter().fold(0xcbf29ce484222325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x100000001b3));
        if let Some(rt) = self.runtime_cache.borrow().get(&key) {
            return rt.clone();
        }
        self.compiles.set(self.compiles.get() + 1);
        let rt = (self.runtime_ctor)(&elf);
        self.runtime_cache.borrow_mut().insert(key, rt.clone());
        rt
    }

    pub fn compile_count(&self) -> usize {
        self.compiles.get()
    }

    /// `Compiled.synchronize` — overridden per backend (the interp is
    /// synchronous; CL flushes its queue in the allocator).
    pub fn synchronize(&self) {}

    /// `as_memoryview` + copyout: read a buffer window's bytes.
    pub fn copyout(&self, o: &dyn OpaqueT) -> Vec<u8> {
        let mut out = vec![0u8; o.size()];
        self.allocator._copyout(&mut out, o);
        out
    }

    /// `_copyin`: write bytes into a buffer window.
    pub fn copyin_to(&self, o: &dyn OpaqueT, data: &[u8]) {
        self.allocator._copyin(o, data);
    }

    /// Direct device-to-device copy between two opaque windows on *this* device
    /// (`clEnqueueCopyBuffer`; host backends copy the windows directly).
    pub fn copy_into(&self, dest: &dyn OpaqueT, src: &dyn OpaqueT) {
        self.allocator._copy_into(dest, src);
    }

    /// The opaque window for kernel args (`Buffer._buf` with `_offset` views).
    pub fn buf_view(&self, buf: &dyn OpaqueT, size: usize, offset: usize) -> Rc<dyn OpaqueT> {
        self.allocator._offset(buf, size, offset)
    }
}

// ---------------- Device registry ----------------

/// `Device._canonicalize` (device.py:19): upper-case the prefix, strip a
/// trailing `:0` index.
pub fn canonicalize_device_name(device: &str) -> String {
    match device.split_once(':') {
        Some((prefix, rest)) => {
            let joined = format!("{}{rest}", prefix.to_uppercase());
            joined.strip_suffix(":0").map(str::to_string).unwrap_or(joined)
        }
        None => device.to_uppercase(),
    }
}

pub type DeviceFactory = Rc<dyn Fn() -> Rc<Compiled>>;

thread_local! {
    static REGISTRY: RefCell<HashMap<String, (DeviceFactory, Option<Rc<Compiled>>)>> = RefCell::new(HashMap::new());
    static BUILTINS: Cell<bool> = const { Cell::new(false) };
}

/// `Device._DeviceMap[name] = ...` — register a backend under its canonical
/// device name ("CL", "MYDEV", ...). Later registration wins.
pub fn register_device(name: &str, factory: DeviceFactory) {
    REGISTRY.with(|reg| {
        reg.borrow_mut().insert(canonicalize_device_name(name), (factory, None));
    });
}

fn ensure_builtins() {
    BUILTINS.with(|b| {
        if !b.get() {
            b.set(true);
            super::ops_interp::register();
            #[cfg(feature = "backend_cl")]
            super::ops_cl::register();
        }
    });
}

/// Total runtime constructions across all devices (the no-O(n)-recompiles
/// gate: identical ASTs must hit the to_program/runtime caches).
pub fn total_compiles() -> usize {
    REGISTRY.with(|reg| reg.borrow().values().filter_map(|(_, d)| d.as_ref()).map(|d| d.compile_count()).sum())
}

/// The registered device names (sorted; built-ins included). A device is
/// only "available" if constructing it succeeds — callers should probe by
/// calling `device(name)` inside `catch_unwind` for availability checks.
pub fn registered_devices() -> Vec<String> {
    ensure_builtins();
    let mut names: Vec<String> = REGISTRY.with(|reg| reg.borrow().keys().cloned().collect());
    names.sort();
    names
}

/// `Device[name]` — the registered device singletons (interp "CPU"/"INTERP";
/// "CL" when built with `backend_cl`). Unknown names fall back to an
/// interp-backed device (the M6 stub — concrete GPU runtimes are not ported).
/// Single-threaded like python's runtime.
pub fn device(name: &str) -> Rc<Compiled> {
    ensure_builtins();
    let canon = canonicalize_device_name(name);
    REGISTRY.with(|reg| {
        let mut reg = reg.borrow_mut();
        let fallback_name = canon.clone();
        reg.entry(canon.clone()).or_insert_with(|| {
            (Rc::new(move || super::ops_interp::fallback(&fallback_name)) as DeviceFactory, None)
        });
    });
    loop {
        let action = REGISTRY.with(|reg| {
            let reg = reg.borrow();
            let e = reg.get(&canon)?;
            match &e.1 {
                Some(d) => Some((None, Some(d.clone()))),
                None => Some((Some(e.0.clone()), None)),
            }
        });
        match action {
            Some((_, Some(d))) => return d,
            Some((Some(f), None)) => {
                let d = f();
                REGISTRY.with(|reg| {
                    let mut reg = reg.borrow_mut();
                    let e = reg.get_mut(&canon).unwrap();
                    if e.1.is_none() {
                        e.1 = Some(d);
                    }
                });
            }
            _ => unreachable!(),
        }
    }
}
