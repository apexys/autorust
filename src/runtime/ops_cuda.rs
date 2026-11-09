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
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, LaunchConfig,
    PushKernelArg, result,
};
use cudarc::nvrtc::{CompileOptions, Ptx, compile_ptx_with_opts};

use crate::dtype::DTypeId;
use crate::renderer::{Renderer, UopRenderer};
use crate::runtime::buffer::OpaqueT;
use crate::runtime::device::{
    BackendAllocator, Comp, Compiled, LruAllocator, Prog, TinyElf, register_device,
};
use crate::uop::UopId;
use crate::uop::ctx::UOpCtx;

// ---------------- device inner ----------------

thread_local! {
    /// SM version of the open device, as `major * 10 + minor` (86 for sm_86).
    /// `0` = not opened on this thread yet. Set by [`CudaInner::open`]; read by
    /// kernels that need to gate on an instruction-set generation.
    static ARCH_SM: std::cell::Cell<i64> = const { std::cell::Cell::new(0) };
    /// The device stream, for the benchmark-only synchronize hook below.
    static STREAM: std::cell::RefCell<Option<Arc<CudaStream>>> =
        const { std::cell::RefCell::new(None) };
}

/// Block until all queued work on the open CUDA stream finishes.
///
/// Benchmark seam only: launches are asynchronous, so timing a loop of them
/// without a host readback needs an explicit stream sync. No-op when CUDA has
/// not been opened on this thread.
pub fn synchronize() {
    if let Some(s) = STREAM.with(|s| s.borrow().clone()) {
        let _ = s.synchronize();
    }
}

/// SM version of the open CUDA device (`86` for `sm_86`), or `None` when the
/// backend is compiled out / no device is present.
///
/// Opens the device on first use, so callers can ask before any tensor exists.
/// Thread-local like the device registry itself.
pub fn arch_sm() -> Option<i64> {
    let v = ARCH_SM.with(|a| a.get());
    if v > 0 {
        return Some(v);
    }
    if !available() {
        return None;
    }
    let _ = crate::runtime::device::device("CUDA");
    let v = ARCH_SM.with(|a| a.get());
    (v > 0).then_some(v)
}

/// Whether the open device has warp-level tensor-core instructions usable by
/// our hand-written kernels (TF32 `mma` needs `sm_80`).
pub fn has_tensor_cores() -> bool {
    arch_sm().is_some_and(|sm| sm >= 80)
}

/// Bounded source→PTX cache (`CUDADevice.compiler.compile_cached`, CCACHE-based
/// upstream). FIFO-capped so a BEAM search can't grow it without bound.
struct CompileCache {
    map: HashMap<String, Vec<u8>>,
    order: VecDeque<String>,
    cap: usize,
}

impl CompileCache {
    fn new() -> CompileCache {
        let cap = crate::helpers::env::get("CUDA_COMPILE_CACHE")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1024);
        CompileCache {
            map: HashMap::new(),
            order: VecDeque::new(),
            cap,
        }
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
    #[allow(dead_code)]
    device_name: String,
    compile_cache: RefCell<CompileCache>,
    /// Lazily created cuBLAS handle bound to `stream` (host GEMM dispatch).
    cublas: RefCell<Option<crate::runtime::cublas::Handle>>,
    /// Cached bias-epilogue kernels keyed by dtype id.
    bias_kernels: RefCell<HashMap<u16, (Arc<CudaModule>, CudaFunction)>>,
}

impl CudaInner {
    /// python `CUDADevice.__init__`: open `device` ordinal, compute the SM arch.
    fn open(device: &str) -> CudaInner {
        let ordinal = device
            .split(':')
            .nth(1)
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        let ctx =
            CudaContext::new(ordinal).unwrap_or_else(|e| panic!("TGERR: cuInit/cuCtxCreate: {e}"));
        let (major, minor) = ctx.compute_capability().unwrap_or((0, 0));
        let arch = format!("sm_{major}{minor}");
        ARCH_SM.with(|a| a.set(((major * 10 + minor) as i64).max(0)));
        let device_name = ctx.name().unwrap_or_else(|_| "CUDA".to_string());
        eprintln!("CUDADevice: opening {device_name} ({arch})");
        let stream = ctx.default_stream();
        STREAM.with(|s| *s.borrow_mut() = Some(stream.clone()));
        CudaInner {
            ctx,
            stream,
            arch,
            device_name,
            compile_cache: RefCell::new(CompileCache::new()),
            cublas: RefCell::new(None),
            bias_kernels: RefCell::new(HashMap::new()),
        }
    }

    /// Compile (and cache) a one-off epilogue kernel.
    fn epilogue(&self, dtype: DTypeId) -> (Arc<CudaModule>, CudaFunction) {
        if let Some(k) = self.bias_kernels.borrow().get(&dtype.0) {
            return (k.0.clone(), k.1.clone());
        }
        // `C = (C + bias[col]) * scale[col] + residual[row, col]`, in place.
        // Any subset of the three can be disabled; the corresponding pointer is
        // then unused (we still pass `c` so the argument is never null).
        let (src, name) = match dtype {
            crate::dtype::FLOAT32 => (
                r#"
extern "C" __global__ void autorust_epi_f32(float* c, const float* b, const float* s, const float* r,
    long long total, long long n, long long m, long long stride_c,
    long long c_off, long long b_off, long long s_off, long long r_off,
    int has_bias, int has_scale, int has_res) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  long long col = i % n;
  long long row = (i / n) % m;
  long long bat = i / (n * m);
  long long rel = bat * stride_c + row * n + col;
  long long ci = c_off + rel;
  float v = c[ci];
  if (has_bias) v += b[b_off + col];
  if (has_scale) v *= s[s_off + col];
  if (has_res) v += r[r_off + rel];
  c[ci] = v;
}
"#,
                "autorust_epi_f32",
            ),
            crate::dtype::FLOAT16 => (
                r#"
#include <cuda_fp16.h>
extern "C" __global__ void autorust_epi_f16(__half* c, const __half* b, const __half* s, const __half* r,
    long long total, long long n, long long m, long long stride_c,
    long long c_off, long long b_off, long long s_off, long long r_off,
    int has_bias, int has_scale, int has_res) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  long long col = i % n;
  long long row = (i / n) % m;
  long long bat = i / (n * m);
  long long rel = bat * stride_c + row * n + col;
  long long ci = c_off + rel;
  float v = __half2float(c[ci]);
  if (has_bias) v += __half2float(b[b_off + col]);
  if (has_scale) v *= __half2float(s[s_off + col]);
  if (has_res) v += __half2float(r[r_off + rel]);
  c[ci] = __float2half(v);
}
"#,
                "autorust_epi_f16",
            ),
            crate::dtype::BFLOAT16 => (
                r#"
#include <cuda_bf16.h>
extern "C" __global__ void autorust_epi_bf16(__nv_bfloat16* c, const __nv_bfloat16* b, const __nv_bfloat16* s, const __nv_bfloat16* r,
    long long total, long long n, long long m, long long stride_c,
    long long c_off, long long b_off, long long s_off, long long r_off,
    int has_bias, int has_scale, int has_res) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  long long col = i % n;
  long long row = (i / n) % m;
  long long bat = i / (n * m);
  long long rel = bat * stride_c + row * n + col;
  long long ci = c_off + rel;
  float v = __bfloat162float(c[ci]);
  if (has_bias) v += __bfloat162float(b[b_off + col]);
  if (has_scale) v *= __bfloat162float(s[s_off + col]);
  if (has_res) v += __bfloat162float(r[r_off + rel]);
  c[ci] = __float2bfloat16(v);
}
"#,
                "autorust_epi_bf16",
            ),
            other => panic!("TGERR: unsupported epilogue dtype {other:?}"),
        };
        let mut options = vec![format!("--gpu-architecture={}", self.arch)];
        match crate::helpers::env::get("CUDA_PATH") {
            Some(p) if !p.is_empty() => options.push(format!("-I{p}/include")),
            _ => {
                options.push("-I/usr/local/cuda/include".to_string());
                options.push("-I/usr/include".to_string());
                options.push("-I/opt/cuda/include".to_string());
            }
        }
        let ptx = compile_ptx_with_opts(
            src,
            CompileOptions {
                options,
                ..Default::default()
            },
        )
        .unwrap_or_else(|e| panic!("TGERR: NVRTC epilogue compile: {e}"));
        let module = self
            .ctx
            .load_module(Ptx::from_src(
                String::from_utf8_lossy(ptx.to_src().as_bytes()).to_string(),
            ))
            .unwrap_or_else(|e| panic!("TGERR: cuModuleLoadData epilogue: {e}"));
        let func = module
            .load_function(name)
            .unwrap_or_else(|e| panic!("TGERR: epilogue function {name}: {e}"));
        self.bias_kernels
            .borrow_mut()
            .insert(dtype.0, (module.clone(), func.clone()));
        (module, func)
    }

    /// Apply the fused epilogue in place over `C[batch, m, n]`.
    #[allow(clippy::too_many_arguments)]
    fn epilogue_apply(
        &self,
        dtype: DTypeId,
        c: u64,
        bias: Option<(u64, i64)>,
        scale: Option<(u64, i64)>,
        residual: Option<(u64, i64)>,
        batch: i64,
        m: i64,
        n: i64,
        stride_c: i64,
        c_off: i64,
    ) {
        if bias.is_none() && scale.is_none() && residual.is_none() {
            return;
        }
        let total = batch.saturating_mul(m).saturating_mul(n);
        if total <= 0 {
            return;
        }
        let (_module, func) = self.epilogue(dtype);
        let b_ptr = bias.map(|x| x.0).unwrap_or(c);
        let s_ptr = scale.map(|x| x.0).unwrap_or(c);
        let r_ptr = residual.map(|x| x.0).unwrap_or(c);
        let (b_off, s_off, r_off) = (
            bias.map(|x| x.1).unwrap_or(0),
            scale.map(|x| x.1).unwrap_or(0),
            residual.map(|x| x.1).unwrap_or(0),
        );
        let (has_bias, has_scale, has_res) = (
            bias.is_some() as i32,
            scale.is_some() as i32,
            residual.is_some() as i32,
        );
        let cfg = LaunchConfig {
            grid_dim: (((total + 255) / 256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut args = self.stream.launch_builder(&func);
        args.arg(&c)
            .arg(&b_ptr)
            .arg(&s_ptr)
            .arg(&r_ptr)
            .arg(&total)
            .arg(&n)
            .arg(&m)
            .arg(&stride_c)
            .arg(&c_off)
            .arg(&b_off)
            .arg(&s_off)
            .arg(&r_off)
            .arg(&has_bias)
            .arg(&has_scale)
            .arg(&has_res);
        unsafe { args.launch(cfg) }.unwrap_or_else(|e| panic!("TGERR: epilogue launch: {e}"));
    }

    /// Run a host GEMM on this device's stream. The descriptor is the
    /// `HostGemm` dataclass attached by the scheduler.
    fn host_gemm(
        &self,
        bufs: &[crate::runtime::buffer::OpaqueView],
        desc: &crate::uop::arg::DataArg,
    ) {
        use crate::schedule::host_gemm::HostGemm;
        let g = HostGemm::from_arg(&crate::uop::arg::Arg::Data(desc.clone()))
            .unwrap_or_else(|| panic!("TGERR: bad HostGemm descriptor"));
        let elem = match g.dtype {
            crate::dtype::FLOAT32 => crate::runtime::cublas::Elem::F32,
            crate::dtype::FLOAT16 => crate::runtime::cublas::Elem::F16,
            crate::dtype::BFLOAT16 => crate::runtime::cublas::Elem::Bf16,
            other => panic!("TGERR: cublas unsupported dtype {other:?}"),
        };
        let isz = match elem {
            crate::runtime::cublas::Elem::F32 => 4i64,
            _ => 2,
        };
        let ptr = |slot: i64, off: i64| -> u64 {
            let o = cuda_of(&*bufs[slot as usize]);
            o.ptr() + (off * isz) as u64
        };
        let (a, b, c) = (
            ptr(g.a_slot, g.a_off),
            ptr(g.b_slot, g.b_off),
            ptr(g.c_slot, g.c_off),
        );
        // cuBLAS is column-major: compute C^T(n,m) = B^T (n,k) @ A^T (k,m), so
        // the cublas "A" operand is our B and the "B" operand is our A.
        let trans_a = g.b_k == 1;
        let trans_b = g.a_m == 1;
        let gemm = crate::runtime::cublas::GemmDesc {
            elem,
            trans_a,
            trans_b,
            m: g.n,
            n: g.m,
            k: g.k,
            lda: if trans_a { g.b_n } else { g.b_k },
            ldb: if trans_b { g.a_k } else { g.a_m },
            ldc: g.c_m,
            alpha: g.alpha,
        };
        self.ctx.bind_to_thread().ok();
        let mut handle = self.cublas.borrow_mut();
        if handle.is_none() {
            *handle = unsafe {
                crate::runtime::cublas::Handle::new(self.stream.cu_stream() as *mut std::ffi::c_void)
            };
        }
        let handle = handle
            .as_ref()
            .unwrap_or_else(|| panic!("TGERR: cuBLAS handle creation failed"));
        let rc = unsafe {
            crate::runtime::cublas::gemm(
                handle,
                &gemm,
                b as *const std::ffi::c_void,
                a as *const std::ffi::c_void,
                c as *mut std::ffi::c_void,
                g.batch,
                g.b_b,
                g.a_b,
                g.c_b,
            )
        };
        if let Err(e) = rc {
            panic!(
                "TGERR: {e} (m={} n={} k={} batch={})",
                g.m, g.n, g.k, g.batch
            );
        }
        let opt = |slot: i64, off: i64| -> Option<(u64, i64)> {
            if slot >= 0 {
                Some((ptr(slot, off), off))
            } else {
                None
            }
        };
        self.epilogue_apply(
            g.dtype,
            c,
            opt(g.bias_slot, g.bias_off),
            opt(g.scale_slot, g.scale_off),
            opt(g.res_slot, g.res_off),
            g.batch,
            g.m,
            g.n,
            if g.batch > 1 { g.c_b } else { 0 },
            g.c_off,
        );
    }

    fn synchronize(&self) {
        self.stream
            .synchronize()
            .unwrap_or_else(|e| panic!("TGERR: cuStreamSynchronize: {e}"));
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
    o.as_any()
        .downcast_ref::<CudaOpaque>()
        .expect("TGERR: expected CUDA opaque")
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
        Rc::new(CudaOpaque {
            slice: Rc::new(slice),
            base,
            offset: 0,
            size,
        })
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
        // No `cuStreamSynchronize`: the copy is enqueued on the same stream as
        // the surrounding kernels, so program order already serializes it
        // (tinygrad's `CUDAAllocator._copy` likewise just posts the async D2D).
        // Syncing here cost ~1.2 ms per copy, and a DA3 forward emits ~290 of
        // them, i.e. the entire steady-state frame time.
        unsafe {
            result::memcpy_dtod_async(
                cuda_of(dest).ptr(),
                cuda_of(src).ptr(),
                src.size(),
                self.inner.stream.cu_stream(),
            )
        }
        .unwrap_or_else(|e| panic!("TGERR: cuMemcpyDtoDAsync_v2: {e}"));
    }
    fn _offset(&self, buf: &dyn OpaqueT, size: usize, offset: usize) -> Rc<dyn OpaqueT> {
        // `CUDAAllocator._offset`: CUdeviceptr_v2(buf.value + offset)
        let o = cuda_of(buf);
        assert!(offset + size <= o.size, "TGERR: offset out of range");
        Rc::new(CudaOpaque {
            slice: o.slice.clone(),
            base: o.base,
            offset: o.offset + offset,
            size,
        })
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
        match crate::helpers::env::get("CUDA_PATH") {
            Some(p) if !p.is_empty() => options.push(format!("-I{p}/include")),
            _ => {
                options.push("-I/usr/local/cuda/include".to_string());
                options.push("-I/usr/include".to_string());
                options.push("-I/opt/cuda/include".to_string());
            }
        }
        let opts = CompileOptions {
            options,
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(src, opts).unwrap_or_else(|e| {
            panic!(
                "TGERR: NVRTC Compile Error ({})\n--- src ---\n{src}\n--- e ---\n{e}",
                self.inner.arch
            )
        });
        ptx.to_src().into_bytes()
    }

    fn compile_cached(&self, src: &str) -> Vec<u8> {
        if let Some(v) = self.inner.compile_cache.borrow().get(src) {
            return v;
        }
        // NVRTC is slow; cache the PTX on disk like tinygrad's CCACHE so a
        // second process doesn't recompile all ~100 MNIST kernels (minutes).
        let mut h = std::collections::hash_map::DefaultHasher::new();
        src.hash(&mut h);
        let key = format!("{}:{:016x}:{}", self.inner.arch, h.finish(), src.len());
        if let Some(s) = crate::engine::disk_cache::get("cuda_ptx", &key) {
            let lib = s.into_bytes();
            self.inner.compile_cache.borrow_mut().put(src, lib.clone());
            return lib;
        }
        let lib = self.compile(src);
        if let Ok(text) = std::str::from_utf8(&lib) {
            crate::engine::disk_cache::put("cuda_ptx", &key, text);
        }
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
    name: String,
    signature: Vec<(i64, DTypeId)>,
}

/// `AUTORUST_KPROF=1` synchronizes after each kernel and prints its name/time.
/// This serializes launches (so it is a profiling aid, not a benchmark mode),
/// but it works without GPU performance-counter permissions.
fn kprof_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::helpers::env::get("AUTORUST_KPROF").is_some())
}

impl CudaProgram {
    fn new(inner: Rc<CudaInner>, obj: &TinyElf) -> CudaProgram {
        // `obj.lib` holds the PTX text produced by `CudaCompiler.compile`.
        let ptx_src = String::from_utf8(obj.lib.clone()).expect("TGERR: CUDA elf lib not utf8");
        let module = inner
            .ctx
            .load_module(Ptx::from_src(ptx_src))
            .unwrap_or_else(|e| panic!("TGERR: cuModuleLoadData: {e}"));
        let func = module
            .load_function(&obj.name)
            .unwrap_or_else(|e| panic!("TGERR: cuModuleGetFunction({}): {e}", obj.name));
        if crate::helpers::env::get_os("AUTORUST_KERNEL_INFO").is_some() {
            use cudarc::driver::sys;
            let f = func.cu_function();
            let get = |a: sys::CUfunction_attribute| -> i32 {
                let mut v: std::ffi::c_int = 0;
                unsafe { sys::cuFuncGetAttribute(&mut v, a, f) };
                v
            };
            eprintln!(
                "[cuda] {}: {} regs, {} max threads, {} B shared, {} B local",
                obj.name,
                get(sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_NUM_REGS),
                get(sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK),
                get(sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES),
                get(sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES),
            );
        }
        CudaProgram {
            inner,
            module,
            func,
            name: obj.name.clone(),
            signature: obj.signature.clone(),
        }
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
        let (mut bi, mut i8i, mut i16i, mut i32i, mut i64i) =
            (0usize, 0usize, 0usize, 0usize, 0usize);
        for (slot, dt) in self.signature.iter() {
            if (*slot as usize) < bufs.len() {
                args.arg(&buf_ptrs[bi]);
                bi += 1;
            } else {
                match dt.bitsize() {
                    8 => {
                        args.arg(&v8[i8i]);
                        i8i += 1;
                    }
                    16 => {
                        args.arg(&v16[i16i]);
                        i16i += 1;
                    }
                    32 => {
                        args.arg(&v32[i32i]);
                        i32i += 1;
                    }
                    64 => {
                        args.arg(&v64[i64i]);
                        i64i += 1;
                    }
                    b => panic!("TGERR: CUDA val arg bitsize {b}"),
                }
            }
        }
        // python passes global_size as gridDim and local_size as blockDim.
        let ls = local_size.unwrap_or([1, 1, 1]);
        let cfg = LaunchConfig {
            grid_dim: (
                global_size[0] as u32,
                global_size[1] as u32,
                global_size[2] as u32,
            ),
            block_dim: (ls[0] as u32, ls[1] as u32, ls[2] as u32),
            shared_mem_bytes: 0,
        };
        let start = std::time::Instant::now();
        unsafe { args.launch(cfg) }.unwrap_or_else(|e| panic!("TGERR: cuLaunchKernel: {e}"));
        if kprof_enabled() && !wait {
            self.inner.synchronize();
            let secs = start.elapsed().as_secs_f64();
            eprintln!("KPROF\t{}\t{secs:.6}", self.name);
            return Some(secs);
        }
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

/// Whether a CUDA device is present, without opening a device singleton and
/// without panicking (the default-device probe). `CudaContext::new` dynamically
/// loads `libcuda`/`libcudart` and calls `cuInit`; failure (no driver/device or
/// no lib) returns `false` rather than panicking.
pub fn available() -> bool {
    // `CudaContext::new` ultimately calls cudarc's `culib()`, which *panics* if
    // the driver library cannot be loaded. That would abort `default_device()`
    // on a machine without CUDA even though `dynamic-loading` lets the backend
    // be compiled in. Probe the library without panicking first; only then is
    // a `cuInit` failure a normal `Err`.
    (unsafe { cudarc::driver::sys::is_culib_present() })
        && cudarc::driver::CudaContext::new(0).is_ok()
}

/// The `CUDADevice` composition + `Device._DeviceMap["CUDA"]` assignment.
pub fn register() {
    let factory: crate::runtime::device::DeviceFactory = Rc::new(|| {
        let inner = Rc::new(CudaInner::open("CUDA"));
        let facts = Renderer::cuda(&inner.arch);
        let renderer: Box<dyn UopRenderer> = Box::new(CudaUopRenderer { facts });
        // `CUDARenderer.compiler` = NVRTCCompiler; the elf lib is the PTX text.
        let compiler: Rc<dyn Comp> = Rc::new(CudaCompiler {
            inner: inner.clone(),
        });
        let allocator = LruAllocator::with_backend(Box::new(CudaAllocator {
            inner: inner.clone(),
        }));
        let ctor: crate::runtime::device::RuntimeCtor = {
            let inner = inner.clone();
            Rc::new(move |elf| Rc::new(RefCell::new(CudaProgram::new(inner.clone(), elf))))
        };
        let dev = Compiled::new("CUDA", renderer, allocator, compiler, ctor);
        // Install the dispatcher whenever the library loads: whether a host
        // GEMM is *emitted* is the scheduler's decision (`AUTORUST_CUBLAS`),
        // and an already-open device must still be able to run one.
        if crate::runtime::cublas::library_present() {
            let inner2 = inner.clone();
            *dev.host_exec.borrow_mut() =
                Some(Box::new(move |bufs, desc| inner2.host_gemm(bufs, desc)));
        }
        dev
    });
    register_device("CUDA", factory);
}
