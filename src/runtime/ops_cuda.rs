//! `runtime/ops_cuda.py` port — the CUDA driver-API backend (`cudarc` behind the
//! `backend_cuda` feature).
//!
//! `CUDADevice` = CudaCompiler (NVRTC source → PTX) + CudaAllocator
//! (`cuMemAlloc` under the LRU) + CudaProgram (`cuModuleLoadData` +
//! `cuModuleGetFunction` + `cuLaunchKernel`) over a `cudarc` context/stream.
//! Registers as "CUDA" through the same public seams
//! (`OpaqueT`/`BackendAllocator`/`Comp`/`Prog`/`UopRenderer` + `register_device`)
//! as the CL and interp backends. Mirrors tinygrad's `CUDAProgram` arg encoding:
//! global buffers are passed as `CUdeviceptr` (u64) values, image/uniform vars as
//! `c_int<bitsize>`.

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
    result,
};
use cudarc::nvrtc::{CompileOptions, Ptx, compile_ptx_with_opts};

use crate::dtype::DTypeId;
use crate::renderer::{Renderer, UopRenderer};
use crate::runtime::buffer::OpaqueT;
use crate::runtime::device::{register_device, BackendAllocator, Comp, Compiled, LruAllocator, Prog, TinyElf};
use crate::uop::ctx::UOpCtx;
use crate::uop::UopId;

// ---------------- device inner ----------------

/// Bounded source→PTX cache (`CUDADevice.compiler.compile_cached`, CCACHE-based
/// upstream). FIFO-capped so a BEAM search can't grow it without bound.
struct CompileCache {
    map: HashMap<String, Vec<u8>>,
    order: VecDeque<String>,
    cap: usize,
}

impl CompileCache {
    fn new() -> CompileCache {
        let cap = std::env::var("CUDA_COMPILE_CACHE").ok().and_then(|s| s.parse().ok()).unwrap_or(1024);
        CompileCache { map: HashMap::new(), order: VecDeque::new(), cap }
    }
    fn get(&self, src: &str) -> Option<Vec<u8>> {
        self.map.get(src).cloned()
    }
    fn put(&mut self, src: &str, lib: Vec<u8>) {
        if self.map.insert(src.to_string(), lib).is_none() {
            self.order.push_back(src.to_string());
        }
        while self.order.len() > self.cap {
            if let Some(k) = self.order.pop_front() {
                self.map.remove(&k);
            }
        }
    }
}

/// The CUDA context/stream pair shared by allocator, compiler and programs.
struct CudaInner {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    arch: String,
    device_name: String,
    compile_cache: RefCell<CompileCache>,
}

impl CudaInner {
    /// python `CUDADevice.__init__`: open `device` ordinal, compute the SM arch.
    fn open(device: &str) -> CudaInner {
        let ordinal = device.split(':').nth(1).and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
        let ctx = CudaContext::new(ordinal).unwrap_or_else(|e| panic!("TGERR: cuInit/cuCtxCreate: {e}"));
        let (major, minor) = ctx.compute_capability().unwrap_or((0, 0));
        let arch = format!("sm_{major}{minor}");
        let device_name = ctx.name().unwrap_or_else(|_| "CUDA".to_string());
        eprintln!("CUDADevice: opening {device_name} ({arch})");
        let stream = ctx.default_stream();
        CudaInner { ctx, stream, arch, device_name, compile_cache: RefCell::new(CompileCache::new()) }
    }

    fn synchronize(&self) {
        self.stream.synchronize().unwrap_or_else(|e| panic!("TGERR: cuStreamSynchronize: {e}"));
    }
}

// ---------------- opaque ----------------

/// A `CUdeviceptr` window over a `cudarc` allocation (python: `CUdeviceptr_v2`).
struct CudaOpaque {
    /// The owning allocation. Kept alive for the window's lifetime.
    slice: Rc<CudaSlice<u8>>,
    /// Base device pointer of the allocation.
    base: u64,
    /// Byte offset of this window within the allocation (`Allocator._offset`).
    offset: usize,
    size: usize,
}

impl CudaOpaque {
    fn ptr(&self) -> u64 {
        self.base + self.offset as u64
    }
}

impl OpaqueT for CudaOpaque {
    fn size(&self) -> usize {
        self.size
    }
    fn storage_id(&self) -> usize {
        self.base as usize
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn cuda_of(o: &dyn OpaqueT) -> &CudaOpaque {
    o.as_any().downcast_ref::<CudaOpaque>().expect("TGERR: expected CUDA opaque")
}

// ---------------- allocator ----------------

struct CudaAllocator {
    inner: Rc<CudaInner>,
}

impl BackendAllocator for CudaAllocator {
    fn _alloc(&self, size: usize) -> Rc<dyn OpaqueT> {
        // `cuMemAlloc_v2`; uninitialized (tinygrad does not zero either).
        let slice = unsafe { self.inner.stream.alloc::<u8>(size) }
            .unwrap_or_else(|e| panic!("TGERR: cuMemAlloc_v2: {e}"));
        let base = {
            let (b, guard) = slice.device_ptr(&self.inner.stream);
            drop(guard);
            b
        };
        Rc::new(CudaOpaque { slice: Rc::new(slice), base, offset: 0, size })
    }
    fn _free(&self, o: Rc<dyn OpaqueT>) {
        drop(o); // CudaSlice drop => cuMemFree_v2
    }
    fn _copyin(&self, dest: &dyn OpaqueT, src: &[u8]) {
        // Blocking copy (`cuMemcpyHtoD_v2`); python posts async through a retained
        // staging buffer (`pending_copyin`), we keep it synchronous.
        self.inner.ctx.bind_to_thread().ok();
        unsafe { result::memcpy_htod_sync::<u8>(cuda_of(dest).ptr(), src) }
            .unwrap_or_else(|e| panic!("TGERR: cuMemcpyHtoD_v2: {e}"));
    }
    fn _copyout(&self, dest: &mut [u8], src: &dyn OpaqueT) {
        self.inner.ctx.bind_to_thread().ok();
        self.inner.synchronize();
        unsafe { result::memcpy_dtoh_sync::<u8>(dest, cuda_of(src).ptr()) }
            .unwrap_or_else(|e| panic!("TGERR: cuMemcpyDtoH_v2: {e}"));
    }
    fn _copy_into(&self, dest: &dyn OpaqueT, src: &dyn OpaqueT) {
        assert_eq!(dest.size(), src.size(), "TGERR: copy size mismatch");
        self.inner.ctx.bind_to_thread().ok();
        unsafe {
            result::memcpy_dtod_async(cuda_of(dest).ptr(), cuda_of(src).ptr(), src.size(), self.inner.stream.cu_stream())
        }
        .unwrap_or_else(|e| panic!("TGERR: cuMemcpyDtoDAsync_v2: {e}"));
        self.inner.synchronize();
    }
    fn _offset(&self, buf: &dyn OpaqueT, size: usize, offset: usize) -> Rc<dyn OpaqueT> {
        // `CUDAAllocator._offset`: CUdeviceptr_v2(buf.value + offset)
        let o = cuda_of(buf);
        assert!(offset + size <= o.size, "TGERR: offset out of range");
        Rc::new(CudaOpaque { slice: o.slice.clone(), base: o.base, offset: o.offset + offset, size })
    }
}

// ---------------- compiler ----------------

struct CudaCompiler {
    inner: Rc<CudaInner>,
}

impl Comp for CudaCompiler {
    /// `NVRTCCompiler.compile` — CUDA C source → PTX bytes.
    fn compile(&self, src: &str) -> Vec<u8> {
        // python: `--gpu-architecture={arch}` (arch = "sm_86") plus the CUDA
        // include search paths for cuda_bf16.h/cuda_fp16.h/cuda_fp8.h.
        let mut options = vec![format!("--gpu-architecture={}", self.inner.arch)];
        match std::env::var("CUDA_PATH") {
            Ok(p) if !p.is_empty() => options.push(format!("-I{p}/include")),
            _ => {
                options.push("-I/usr/local/cuda/include".to_string());
                options.push("-I/usr/include".to_string());
                options.push("-I/opt/cuda/include".to_string());
            }
        }
        let opts = CompileOptions { options, ..Default::default() };
        let ptx = compile_ptx_with_opts(src, opts)
            .unwrap_or_else(|e| panic!("TGERR: NVRTC Compile Error ({})\n--- src ---\n{src}\n--- e ---\n{e}", self.inner.arch));
        ptx.to_src().into_bytes()
    }

    fn compile_cached(&self, src: &str) -> Vec<u8> {
        if let Some(v) = self.inner.compile_cache.borrow().get(src) {
            return v;
        }
        let lib = self.compile(src);
        self.inner.compile_cache.borrow_mut().put(src, lib.clone());
        lib
    }
}

// ---------------- program ----------------

/// `CUDAProgram` — load the PTX module, fetch the kernel, launch it.
pub struct CudaProgram {
    inner: Rc<CudaInner>,
    /// `module` is kept alive for `func` (python `self.module`/`self.prg`).
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    func: CudaFunction,
    signature: Vec<(i64, DTypeId)>,
}

impl CudaProgram {
    fn new(inner: Rc<CudaInner>, obj: &TinyElf) -> CudaProgram {
        // `obj.lib` holds the PTX text produced by `CudaCompiler.compile`.
        let ptx_src = String::from_utf8(obj.lib.clone()).expect("TGERR: CUDA elf lib not utf8");
        let module = inner.ctx.load_module(Ptx::from_src(ptx_src))
            .unwrap_or_else(|e| panic!("TGERR: cuModuleLoadData: {e}"));
        let func = module.load_function(&obj.name)
            .unwrap_or_else(|e| panic!("TGERR: cuModuleGetFunction({}): {e}", obj.name));
        CudaProgram { inner, module, func, signature: obj.signature.clone() }
    }
}

impl Prog for CudaProgram {
    fn call(
        &mut self,
        bufs: &[Rc<dyn OpaqueT>],
        global_size: [usize; 3],
        local_size: Option<[usize; 3]>,
        vals: &[i64],
        wait: bool,
    ) -> Option<f64> {
        self.inner.ctx.bind_to_thread().ok();
        // python `encode_args`: buffers are CUdeviceptr (u64) values, vars are
        // c_int<bitsize>. Keep every value alive until `launch` (the builder
        // stores raw pointers).
        let mut buf_ptrs: Vec<u64> = Vec::new();
        let mut v8: Vec<i8> = Vec::new();
        let mut v16: Vec<i16> = Vec::new();
        let mut v32: Vec<i32> = Vec::new();
        let mut v64: Vec<i64> = Vec::new();
        for (slot, dt) in self.signature.iter() {
            if (*slot as usize) < bufs.len() {
                buf_ptrs.push(cuda_of(&*bufs[*slot as usize]).ptr());
            } else {
                let v = vals[(*slot as usize) - bufs.len()];
                match dt.bitsize() {
                    8 => v8.push(v as i8),
                    16 => v16.push(v as i16),
                    32 => v32.push(v as i32),
                    64 => v64.push(v),
                    b => panic!("TGERR: CUDA val arg bitsize {b}"),
                }
            }
        }
        let mut args = self.inner.stream.launch_builder(&self.func);
        let (mut bi, mut i8i, mut i16i, mut i32i, mut i64i) = (0usize, 0usize, 0usize, 0usize, 0usize);
        for (slot, dt) in self.signature.iter() {
            if (*slot as usize) < bufs.len() {
                args.arg(&buf_ptrs[bi]);
                bi += 1;
            } else {
                match dt.bitsize() {
                    8 => { args.arg(&v8[i8i]); i8i += 1; }
                    16 => { args.arg(&v16[i16i]); i16i += 1; }
                    32 => { args.arg(&v32[i32i]); i32i += 1; }
                    64 => { args.arg(&v64[i64i]); i64i += 1; }
                    b => panic!("TGERR: CUDA val arg bitsize {b}"),
                }
            }
        }
        // python passes global_size as gridDim and local_size as blockDim.
        let ls = local_size.unwrap_or([1, 1, 1]);
        let cfg = LaunchConfig {
            grid_dim: (global_size[0] as u32, global_size[1] as u32, global_size[2] as u32),
            block_dim: (ls[0] as u32, ls[1] as u32, ls[2] as u32),
            shared_mem_bytes: 0,
        };
        let start = std::time::Instant::now();
        unsafe { args.launch(cfg) }.unwrap_or_else(|e| panic!("TGERR: cuLaunchKernel: {e}"));
        if wait {
            self.inner.synchronize();
            return Some(start.elapsed().as_secs_f64());
        }
        None
    }
}

// ---------------- renderer ----------------

/// `CUDARenderer` — cstyle with the CUDA flavor.
pub struct CudaUopRenderer {
    facts: Renderer,
}

impl UopRenderer for CudaUopRenderer {
    fn facts(&self) -> &Renderer {
        &self.facts
    }
    fn render(&self, ctx: &mut UOpCtx, lst: &[UopId]) -> String {
        crate::renderer::cstyle::render_cuda(ctx, lst)
    }
}

// ---------------- device + registration ----------------

/// The `CUDADevice` composition + `Device._DeviceMap["CUDA"]` assignment.
pub fn register() {
    let factory: crate::runtime::device::DeviceFactory = Rc::new(|| {
        let inner = Rc::new(CudaInner::open("CUDA"));
        let facts = Renderer::cuda(&inner.arch);
        let renderer: Box<dyn UopRenderer> = Box::new(CudaUopRenderer { facts });
        // `CUDARenderer.compiler` = NVRTCCompiler; the elf lib is the PTX text.
        let compiler: Rc<dyn Comp> = Rc::new(CudaCompiler { inner: inner.clone() });
        let allocator = LruAllocator::with_backend(Box::new(CudaAllocator { inner: inner.clone() }));
        let ctor: Rc<dyn Fn(&TinyElf) -> Rc<RefCell<dyn Prog>>> = {
            let inner = inner.clone();
            Rc::new(move |elf| Rc::new(RefCell::new(CudaProgram::new(inner.clone(), elf))))
        };
        Compiled::new("CUDA", renderer, allocator, compiler, ctor)
    });
    register_device("CUDA", factory);
}
