//! `device.py` device seams (M6): Allocator + LRUAllocator default impl,
//! Compiler, Program (runtime) trait, Compiled device, and the Device registry.
//!
//! The interpreter backend registers as the first device: "CPU" and "PYTHON"
//! both run `InterpProgram` over byte-storage opaques. Nothing here assumes
//! CPU beyond the opaque being bytes — M8 (OpenCL) adds a second `Compiled`.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::codegen::interp::{exec_kernel_with_dims, MemView};
use crate::codegen::postrange::Renderer;
use crate::dtype::DTypeId;
use crate::uop::ctx::UOpCtx;
use crate::uop::flat::{parse_flat, FlatGraph};
use crate::uop::graph::Graph;
use crate::uop::UopId;

use super::buffer::Opaque;

// ---------------- Allocator ----------------

/// The runtime-implemented allocator methods (python `Allocator._alloc` etc).
/// M6 opaques are byte storages; `_copyin`/`_copyout` are memcpy windows.
pub trait BackendAllocator {
    fn _alloc(&self, size: usize) -> Opaque {
        Opaque { mem: Rc::new(RefCell::new(vec![0u8; size])), offset: 0, size }
    }
    fn _free(&self, _o: Opaque) {}
    fn _copyin(&self, dest: &Opaque, src: &[u8]) {
        dest.write(src);
    }
    fn _copyout(&self, dest: &mut [u8], src: &Opaque) {
        dest.copy_from_slice(&src.read());
    }
    /// `Allocator._offset` — slice the window (memoryview semantics).
    fn offset(&self, buf: &Opaque, size: usize, offset: usize) -> Opaque {
        assert!(offset + size <= buf.size, "TGERR: offset out of range");
        Opaque { mem: buf.mem.clone(), offset: buf.offset + offset, size }
    }
}

/// `PythonAllocator` / the M6 interpreter allocator: plain host memory.
pub struct MemAllocator;
impl BackendAllocator for MemAllocator {}

/// `LRUAllocator` (device.py:276) — the (size, options)-keyed free-list cache
/// with OOM-triggered `free_cache()` retry. Perf-critical for real backends
/// (every uncached alloc is a clCreateBuffer in M8).
pub struct LruAllocator {
    inner: MemAllocator,
    cache: RefCell<HashMap<(usize, crate::engine::buffer::BufferSpec), Vec<Opaque>>>,
}

impl LruAllocator {
    pub fn new() -> Self {
        LruAllocator { inner: MemAllocator, cache: RefCell::new(HashMap::new()) }
    }

    pub fn alloc(&self, size: usize) -> Opaque {
        assert!(size > 0, "TGERR: alloc size must be positive");
        if let Some(o) = self.cache.borrow_mut().get_mut(&(size, Default::default())).and_then(|v| v.pop()) {
            return o;
        }
        // OOM retry path: our host allocator only fails on OOM, same retry
        self.inner._alloc(size)
    }

    /// `LRUAllocator.free` — push to the cache (LRU on by default).
    pub fn free(&self, opaque: Opaque, size: usize) {
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
}

impl Default for LruAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl LruAllocator {
    pub fn _copyin(&self, dest: &Opaque, src: &[u8]) {
        self.inner._copyin(dest, src)
    }
    pub fn _copyout(&self, dest: &mut [u8], src: &Opaque) {
        self.inner._copyout(dest, src)
    }
    pub fn offset(&self, buf: &Opaque, size: usize, offset: usize) -> Opaque {
        self.inner.offset(buf, size, offset)
    }
}

// ---------------- Compiler ----------------

/// `Compiler` — the empty compiler default: `compile(src) -> src.bytes`.
/// compile_cached disk caching is skipped (helpers diskcache is out of scope).
#[derive(Default)]
pub struct Compiler;

impl Compiler {
    pub fn compile(&self, src: &str) -> Vec<u8> {
        src.as_bytes().to_vec()
    }
    pub fn compile_cached(&self, src: &str) -> Vec<u8> {
        self.compile(src)
    }
}

// ---------------- Program (runtime) ----------------

/// `TinyELF` (device.py) — the compiled artifact seam: lib bytes, function
/// name, and the buffer signature (slot, dtype) in linear order.
pub struct TinyElf {
    pub lib: Vec<u8>,
    pub name: String,
    pub signature: Vec<(i64, DTypeId)>,
}

/// `Program.__call__(*bufs, global_size, local_size, vals, wait)` — the
/// runtime launch seam. Returns elapsed seconds when `wait`.
pub trait Prog {
    fn call(&mut self, bufs: &[Opaque], global_size: [usize; 3], local_size: Option<[usize; 3]>, vals: &[i64], wait: bool) -> Option<f64>;
}

/// The interpreter `PythonProgram`: un-pickles (flat-decodes) the linear uops
/// from the elf lib and executes them with the M5 uops interpreter.
pub struct InterpProgram {
    ctx: UOpCtx,
    uops: Vec<UopId>,
    globals: Vec<i64>,
    sig_dtypes: HashMap<i64, DTypeId>,
}

impl InterpProgram {
    pub fn new(elf: &TinyElf) -> InterpProgram {
        let src = std::str::from_utf8(&elf.lib).expect("TGERR: interp elf lib not utf8");
        let fg: FlatGraph = parse_flat(src).expect("TGERR: interp elf lib parse");
        let g = Graph::from_flat(&fg).expect("TGERR: interp elf lib graph");
        let uops = g.roots.clone();
        let n = fg.nodes.len();
        let mut ctx = UOpCtx::new();
        ctx.grow_to(n);
        ctx.g = g;
        let globals: Vec<i64> = elf.signature.iter().map(|(s, _)| *s).collect();
        let sig_dtypes: HashMap<i64, DTypeId> = elf.signature.iter().copied().collect();
        InterpProgram { ctx, uops, globals, sig_dtypes }
    }
}

impl Prog for InterpProgram {
    fn call(&mut self, bufs: &[Opaque], global_size: [usize; 3], local_size: Option<[usize; 3]>, vals: &[i64], wait: bool) -> Option<f64> {
        let st = std::time::Instant::now();
        // bind the signature slots to the passed buffer windows
        let mut views: HashMap<i64, MemView> = HashMap::new();
        assert!(bufs.len() == self.globals.len(), "TGERR: kernel arg count mismatch");
        for (i, (&slot, o)) in self.globals.iter().zip(bufs.iter()).enumerate() {
            let dt = self.sig_dtypes[&slot];
            let isz = dt.itemsize();
            let _ = i;
            views.insert(slot, MemView(o.mem.clone(), o.offset, isz, o.size / isz));
        }
        let vals_f: Vec<f64> = vals.iter().map(|&v| v as f64).collect();
        // python's PythonProgram iterates `global_size[::-1]` / `local_size[::-1]`;
        // the M5 interpreter shares that little-endian dim convention.
        let gs_rev = [global_size[2], global_size[1], global_size[0]];
        let ls_rev = local_size.map(|l| [l[2], l[1], l[0]]);
        exec_kernel_with_dims(&mut self.ctx, &self.uops, &views, &vals_f, gs_rev, ls_rev);
        if wait {
            Some(st.elapsed().as_secs_f64())
        } else {
            None
        }
    }
}

// ---------------- Compiled device ----------------

/// `Compiled` (device.py:343) — renderer + allocator + runtime factory.
pub struct Compiled {
    pub device: String,
    pub allocator: LruAllocator,
    pub renderer: Renderer,
    compiles: std::cell::Cell<usize>,
    runtime_cache: RefCell<HashMap<u64, Rc<RefCell<dyn Prog>>>>,
}

impl Compiled {
    fn new(device: &str) -> Compiled {
        let mut renderer = Renderer::cpu_interp();
        renderer.device = device.to_string();
        Compiled {
            device: device.to_string(),
            allocator: LruAllocator::new(),
            renderer,
            compiles: std::cell::Cell::new(0),
            runtime_cache: RefCell::new(HashMap::new()),
        }
    }

    /// `Device[device].runtime(elf)` — construct (or fetch cached) the runtime
    /// for a compiled artifact. Cache key = lib hash (python: (ast.key, device)).
    pub fn runtime(self: &Rc<Self>, elf: TinyElf) -> Rc<RefCell<dyn Prog>> {
        let key = elf.lib.iter().fold(0xcbf29ce484222325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x100000001b3));
        if let Some(rt) = self.runtime_cache.borrow().get(&key) {
            return rt.clone();
        }
        self.compiles.set(self.compiles.get() + 1);
        let rt: Rc<RefCell<dyn Prog>> = Rc::new(RefCell::new(InterpProgram::new(&elf)));
        self.runtime_cache.borrow_mut().insert(key, rt.clone());
        rt
    }

    pub fn compile_count(&self) -> usize {
        self.compiles.get()
    }

    /// `Compiled.synchronize` — the interpreter is synchronous.
    pub fn synchronize(&self) {}

    /// `as_memoryview` + copyout: read a buffer window's bytes.
    pub fn copyout(&self, o: &Opaque) -> Vec<u8> {
        o.read()
    }

    /// `_copyin`: write bytes into a buffer window.
    pub fn copyin_to(&self, o: &Opaque, data: &[u8]) {
        self.allocator._copyin(o, data);
    }
}

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

thread_local! {
    static DEVICES: RefCell<HashMap<String, Rc<Compiled>>> = RefCell::new(HashMap::new());
}

/// Total runtime constructions across all devices (the no-O(n)-recompiles
/// gate: identical ASTs must hit the to_program/runtime caches).
pub fn total_compiles() -> usize {
    DEVICES.with(|reg| reg.borrow().values().map(|d| d.compile_count()).sum())
}

/// `Device[name]` — the registered device singletons ("CPU" and "PYTHON" both
/// run the interpreter backend for M6). Single-threaded like python's runtime.
pub fn device(name: &str) -> Rc<Compiled> {
    let canon = canonicalize_device_name(name);
    DEVICES.with(|reg| reg.borrow_mut().entry(canon.clone()).or_insert_with(|| Rc::new(Compiled::new(&canon))).clone())
}
