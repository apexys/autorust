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
use crate::uop::arg::DataArg;

use super::buffer::{OpaqueT, OpaqueView};
// ---------------- Allocator ----------------

/// Device memory capacity and current availability.
///
/// `total` is the backend's capacity in bytes. `free` is the amount currently
/// available when the driver exposes it; some backends (notably OpenCL without
/// a vendor extension) can only report the total. Use a safety margin: the
/// value is a snapshot, and allocations from other processes or the driver's
/// own bookkeeping are not under this allocator's control.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryInfo {
    /// Total device memory in bytes.
    pub total: usize,
    /// Currently available device memory in bytes, when known.
    pub free: Option<usize>,
}

/// The runtime-implemented allocator methods (python `Allocator._alloc` etc).
/// Opaques are backend-owned (`OpaqueT` trait objects); the default host
/// implementation is `MemAllocator`.
pub trait BackendAllocator {
    /// Allocate `size` bytes, returning a backend-owned opaque handle.
    fn _alloc(&self, size: usize) -> Rc<dyn OpaqueT>;
    /// Release an opaque previously returned by `_alloc`.
    fn _free(&self, o: Rc<dyn OpaqueT>);
    /// Copy host bytes into an opaque window (`Allocator._copyin`).
    fn _copyin(&self, dest: &dyn OpaqueT, src: &[u8]);
    /// Copy an opaque window out into host bytes (`Allocator._copyout`).
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
    /// Device memory capacity/availability, when the backend can report it.
    fn memory_info(&self) -> Option<MemoryInfo> {
        None
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
        Rc::new(super::buffer::HostOpaque {
            mem: b.mem.clone(),
            offset: b.offset + offset,
            size,
        })
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

/// Free-list keyed by `(size, buffer options)`.
type OpaqueCache = HashMap<(usize, super::buffer::BufferSpec), Vec<Rc<dyn OpaqueT>>>;

/// Constructs a backend runtime for a compiled artifact.
pub type RuntimeCtor = Rc<dyn Fn(&TinyElf) -> Rc<RefCell<dyn Prog>>>;

/// Programs compiled for one device, keyed by ELF hash.
type RuntimeCache = HashMap<u64, Rc<RefCell<dyn Prog>>>;

/// `LRUAllocator` (device.py:276) — the (size, options)-keyed free-list cache
/// with OOM-triggered `free_cache()` retry. Perf-critical for real backends
/// (every uncached alloc is a clCreateBuffer in M8).
pub struct LruAllocator {
    inner: Box<dyn BackendAllocator>,
    cache: RefCell<OpaqueCache>,
}

impl LruAllocator {
    /// The host default (`MemAllocator`-backed).
    pub fn new() -> Self {
        LruAllocator {
            inner: Box::new(MemAllocator),
            cache: RefCell::new(HashMap::new()),
        }
    }

    /// Wrap any backend allocator (`CLAllocator` etc).
    pub fn with_backend(inner: Box<dyn BackendAllocator>) -> Self {
        LruAllocator {
            inner,
            cache: RefCell::new(HashMap::new()),
        }
    }

    /// Allocate `size` bytes, reusing a cached free-list entry when possible.
    pub fn alloc(&self, size: usize) -> Rc<dyn OpaqueT> {
        assert!(size > 0, "TGERR: alloc size must be positive");
        if let Some(o) = self
            .cache
            .borrow_mut()
            .get_mut(&(size, Default::default()))
            .and_then(|v| v.pop())
        {
            return o;
        }
        // OOM retry path: our host allocator only fails on OOM, same retry
        self.inner._alloc(size)
    }

    /// `LRUAllocator.free` — push to the cache (LRU on by default).
    pub fn free(&self, opaque: Rc<dyn OpaqueT>, size: usize) {
        self.cache
            .borrow_mut()
            .entry((size, Default::default()))
            .or_default()
            .push(opaque);
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

    /// Device memory capacity/availability, when the backend can report it.
    pub fn memory_info(&self) -> Option<MemoryInfo> {
        self.inner.memory_info()
    }

    /// Copy host bytes into `dest` through the backend allocator.
    pub fn _copyin(&self, dest: &dyn OpaqueT, src: &[u8]) {
        self.inner._copyin(dest, src)
    }
    /// Copy `src` out into host bytes through the backend allocator.
    pub fn _copyout(&self, dest: &mut [u8], src: &dyn OpaqueT) {
        self.inner._copyout(dest, src)
    }
    /// Return a `(size, offset)` sub-window of `buf`.
    pub fn _offset(&self, buf: &dyn OpaqueT, size: usize, offset: usize) -> Rc<dyn OpaqueT> {
        self.inner._offset(buf, size, offset)
    }
    /// Copy device-to-device from `src` into `dest`.
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
    /// Compile `src` to library bytes; the identity default returns the source bytes.
    fn compile(&self, src: &str) -> Vec<u8> {
        src.as_bytes().to_vec()
    }
    /// `Compiler.compile` with an in-memory cache keyed by source.
    fn compile_cached(&self, src: &str) -> Vec<u8>;
    /// Compile a batch of sources, preserving order. Backends that can compile
    /// concurrently (CUDA/NVRTC) override this; the default stays serial.
    fn compile_batch(&self, srcs: &[String]) -> Vec<Vec<u8>> {
        srcs.iter().map(|s| self.compile_cached(s)).collect()
    }
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
    /// Compiler output (pickle bytes for interp, source text for NVIDIA CL).
    pub lib: Vec<u8>,
    /// Entry-point function name.
    pub name: String,
    /// Buffer signature as `(slot, dtype)` pairs in linear order.
    pub signature: Vec<(i64, crate::dtype::DTypeId)>,
}

/// Content key for a compiled artifact (FNV-1a over `lib`). Computed once at
/// compile time and stored on `CompiledProg`; hashing the artifact on every
/// launch was 97% of the per-kernel launch cost on the DA3 replay path.
pub fn elf_key(elf: &TinyElf) -> u64 {
    elf.lib.iter().fold(0xcbf29ce484222325u64, |h, &b| {
        (h ^ b as u64).wrapping_mul(0x100000001b3)
    })
}

/// `Program.__call__(*bufs, global_size, local_size, vals, wait)` — the
/// runtime launch seam. Returns elapsed seconds when `wait`.
pub trait Prog {
    /// Launch with the given buffers, grid sizes and integer vals; returns elapsed seconds when `wait`.
    fn call(
        &mut self,
        bufs: &[Rc<dyn OpaqueT>],
        global_size: [usize; 3],
        local_size: Option<[usize; 3]>,
        vals: &[i64],
        wait: bool,
    ) -> Option<f64>;
}

// ---------------- Compiled device ----------------

/// A host-side operation executed instead of a generated kernel (currently the
/// optional cuBLAS GEMM). The descriptor is the `HostGemm` dataclass carried by
/// the kernel's `KernelInfo`; `bufs` is indexed by buffer slot.
pub type HostExec = dyn Fn(&[OpaqueView], &DataArg);

/// `Compiled` (device.py:343) — renderer + allocator + compiler + runtime
/// factory, the four public seams every backend plugs into.
pub struct Compiled {
    /// Device name in canonical form.
    pub device: String,
    /// Free-list allocator for this device.
    pub allocator: LruAllocator,
    /// UOp renderer producing device source.
    pub renderer: Box<dyn UopRenderer>,
    /// Compiler turning rendered source into library bytes.
    pub compiler: Rc<dyn Comp>,
    /// Optional host-op dispatcher (library GEMM).
    pub host_exec: RefCell<Option<Box<HostExec>>>,
    runtime_ctor: RuntimeCtor,
    compiles: Cell<usize>,
    runtime_cache: RefCell<RuntimeCache>,
}

impl Compiled {
    /// Compose a device from its public seams (python `Compiled.__init__`).
    pub fn new(
        device: &str,
        renderer: Box<dyn UopRenderer>,
        allocator: LruAllocator,
        compiler: Rc<dyn Comp>,
        runtime_ctor: RuntimeCtor,
    ) -> Rc<Compiled> {
        Rc::new(Compiled {
            device: device.to_string(),
            allocator,
            renderer,
            compiler,
            host_exec: RefCell::new(None),
            runtime_ctor,
            compiles: Cell::new(0),
            runtime_cache: RefCell::new(HashMap::new()),
        })
    }

    /// `Device[device].runtime(elf)` — construct (or fetch cached) the runtime
    /// for a compiled artifact. Cache key = lib hash (python: (ast.key, device)).
    pub fn runtime(self: &Rc<Self>, elf: TinyElf) -> Rc<RefCell<dyn Prog>> {
        let key = elf_key(&elf);
        self.runtime_keyed(key, &elf)
    }

    /// Device memory capacity/availability, when the backend can report it.
    pub fn memory_info(&self) -> Option<MemoryInfo> {
        self.allocator.memory_info()
    }

    /// Compile and launch a **hand-written kernel** for this backend.
    ///
    /// `src` is the backend's kernel language (CUDA C for `CUDA`, OpenCL C for
    /// `CL`, the generated-Rust ABI for `RUST`). This is the seam for fused
    /// kernels that cannot be expressed as generated uops — flash attention is
    /// the motivating case: online softmax needs running per-thread state, which
    /// the uop IR cannot carry.
    ///
    /// `signature` is `(slot, dtype)` in argument order. A `slot` below
    /// `bufs.len()` is a buffer pointer; beyond it the value comes from `vals`
    /// at `slot - bufs.len()`, exactly like a generated kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn run_host_kernel(
        self: &Rc<Self>,
        src: &str,
        name: &str,
        signature: &[(i64, crate::dtype::DTypeId)],
        bufs: &[OpaqueView],
        global: [usize; 3],
        local: Option<[usize; 3]>,
        vals: &[i64],
        wait: bool,
    ) -> Option<f64> {
        let lib = self.compiler.compile_cached(src);
        let elf = TinyElf {
            lib,
            name: name.to_string(),
            signature: signature.to_vec(),
        };
        let rt = self.runtime(elf);
        rt.borrow_mut().call(bufs, global, local, vals, wait)
    }

    /// [`Compiled::runtime`] with a caller-supplied content key (computed once
    /// at compile time and stored on
    /// [`CompiledProg`](crate::engine::realize::CompiledProg)), so a launch does
    /// not re-hash the compiled artifact.
    pub fn runtime_keyed(&self, key: u64, elf: &TinyElf) -> Rc<RefCell<dyn Prog>> {
        // Fold the buffer signature into the runtime identity. `key` is a hash of
        // the compiled artifact; a cached runtime built from an artifact with a
        // different signature would index `bufs` with slots the launching program
        // does not provide (observed: signature slot 7 against six buffers).
        let mut key = key;
        for (slot, dt) in &elf.signature {
            key = key
                .wrapping_mul(1099511628211)
                .wrapping_add(*slot as u64)
                .wrapping_add((dt.0 as u64) << 32);
        }
        if let Some(rt) = self.runtime_cache.borrow().get(&key) {
            return rt.clone();
        }
        self.compiles.set(self.compiles.get() + 1);
        let rt = (self.runtime_ctor)(elf);
        self.runtime_cache.borrow_mut().insert(key, rt.clone());
        rt
    }

    /// Construct a runtime **without** caching it. BEAM search builds thousands
    /// of throwaway programs; caching each would leak backends (CL programs) and
    /// exhaust the driver. The caller drops the returned runtime to release it.
    pub fn runtime_nocache(&self, elf: &TinyElf) -> Rc<RefCell<dyn Prog>> {
        (self.runtime_ctor)(elf)
    }

    /// Number of runtimes constructed through the caching path.
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
/// trailing `:0` index (`cuda:1` stays `CUDA:1`).
pub fn canonicalize_device_name(device: &str) -> String {
    let name = match device.split_once(':') {
        Some((prefix, rest)) => format!("{}:{}", prefix.to_uppercase(), rest),
        None => device.to_uppercase(),
    };
    name.strip_suffix(":0").map(str::to_string).unwrap_or(name)
}

/// Whether the named device executes on the `f64`-lane scalar interpreter
/// (`ops_interp`) rather than a dtype-native backend. The interpreter cannot
/// represent the `uint64` Threefry pack exactly, so `rand`/`randn` use the host
/// evaluator on these devices (§8.23).
///
/// `RAYON` is interpreter-backed until its dtype-native range IR lands
/// (`CURRENT_STATE.md` §9.8, R2); flip it here when that happens.
pub fn is_interp_engine(device: &str) -> bool {
    match canonicalize_device_name(device).as_str() {
        "INTERP" | "PYTHON" | "CPU" => true,
        #[cfg(feature = "backend_rayon")]
        "RAYON" => true,
        _ => false,
    }
}

/// Lazily builds a `Compiled` device for the registry.
pub type DeviceFactory = Rc<dyn Fn() -> Rc<Compiled>>;

/// Builds a [`DeviceFactory`] for an indexed device name (`"CL:1"`) from the
/// backend family registered under its base name (`"CL"`).
pub type DeviceFamily = Rc<dyn Fn(&str) -> DeviceFactory>;

/// A registered backend and its lazily-constructed device.
struct DeviceEntry {
    factory: DeviceFactory,
    device: Option<Rc<Compiled>>,
    /// Kernel-buffer limit from the backend's renderer facts (available without
    /// constructing the device; the scheduler's fusion budget uses it).
    max_buffers: usize,
}

thread_local! {
    static REGISTRY: RefCell<HashMap<String, DeviceEntry>> = RefCell::new(HashMap::new());
    static FAMILIES: RefCell<HashMap<String, DeviceFamily>> = RefCell::new(HashMap::new());
    static BUILTINS: Cell<bool> = const { Cell::new(false) };
}

/// `Device._DeviceMap[name] = ...` — register a backend under its canonical
/// device name ("CL", "MYDEV", ...). Later registration wins. The scheduler's
/// fusion budget is unbounded unless the backend registers a renderer limit.
pub fn register_device(name: &str, factory: DeviceFactory) {
    register_device_with_limit(name, usize::MAX, factory);
}

/// Register a backend that can open several physical devices under indexed
/// names (`CL`, `CL:1`, `CL:2`, ...). `device("CL:1")` then constructs the
/// second device lazily; the bare name is index 0 (tinygrad strips `:0`).
pub fn register_device_family(base: &str, family: DeviceFamily) {
    FAMILIES.with(|f| f.borrow_mut().insert(base.to_uppercase(), family));
}

/// [`register_device`] with the renderer's kernel-buffer limit. Bounded
/// backends (Metal, a future OpenGL compute backend) use this so the scheduler
/// can consult the limit without constructing the device.
pub fn register_device_with_limit(name: &str, max_buffers: usize, factory: DeviceFactory) {
    REGISTRY.with(|reg| {
        reg.borrow_mut().insert(
            canonicalize_device_name(name),
            DeviceEntry {
                factory,
                device: None,
                max_buffers,
            },
        );
    });
}

/// The kernel-buffer limit registered for `name` (unbounded when unknown).
/// Does not construct the device.
pub fn registered_max_buffers(name: &str) -> usize {
    ensure_builtins();
    let canon = canonicalize_device_name(name);
    REGISTRY.with(|reg| {
        reg.borrow()
            .get(&canon)
            .map(|e| e.max_buffers)
            .unwrap_or(usize::MAX)
    })
}

fn ensure_builtins() {
    BUILTINS.with(|b| {
        if !b.get() {
            b.set(true);
            super::ops_interp::register();
            #[cfg(feature = "backend_rayon")]
            super::ops_rayon::register();
            #[cfg(feature = "backend_cl")]
            super::ops_cl::register();
            #[cfg(feature = "backend_vk")]
            super::ops_vk::register();
            #[cfg(all(feature = "backend_metal", target_os = "macos"))]
            super::ops_metal::register();
            #[cfg(feature = "backend_clang")]
            super::ops_clang::register();
            #[cfg(all(
                feature = "backend_cuda",
                any(target_os = "linux", target_os = "windows")
            ))]
            super::ops_cuda::register();
            #[cfg(feature = "backend_rust")]
            super::ops_rust::register();
        }
    });
}

/// Total runtime constructions across all devices (the no-O(n)-recompiles
/// gate: identical ASTs must hit the to_program/runtime caches).
pub fn total_compiles() -> usize {
    REGISTRY.with(|reg| {
        reg.borrow()
            .values()
            .filter_map(|e| e.device.as_ref())
            .map(|d| d.compile_count())
            .sum()
    })
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
        if reg.contains_key(&canon) {
            return;
        }
        // An indexed name (`CL:1`) uses the base backend's family factory.
        let family = canon
            .split_once(':')
            .and_then(|(base, _)| FAMILIES.with(|f| f.borrow().get(base).cloned()));
        let entry = match family {
            Some(make) => DeviceEntry {
                factory: make(&canon),
                device: None,
                max_buffers: usize::MAX,
            },
            None => {
                let fallback_name = canon.clone();
                DeviceEntry {
                    factory: Rc::new(move || super::ops_interp::fallback(&fallback_name))
                        as DeviceFactory,
                    device: None,
                    max_buffers: usize::MAX,
                }
            }
        };
        reg.insert(canon.clone(), entry);
    });
    loop {
        let action = REGISTRY.with(|reg| {
            let reg = reg.borrow();
            let e = reg.get(&canon)?;
            match &e.device {
                Some(d) => Some((None, Some(d.clone()))),
                None => Some((Some(e.factory.clone()), None)),
            }
        });
        match action {
            Some((_, Some(d))) => return d,
            Some((Some(f), None)) => {
                let d = f();
                REGISTRY.with(|reg| {
                    let mut reg = reg.borrow_mut();
                    let e = reg.get_mut(&canon).unwrap();
                    if e.device.is_none() {
                        e.device = Some(d);
                    }
                });
            }
            _ => unreachable!(),
        }
    }
}
