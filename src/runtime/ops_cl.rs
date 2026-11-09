//! `runtime/ops_cl.py` port — the OpenCL backend (dynamically loaded `cl_sys`
//! behind the `backend_cl` feature).
//!
//! `CLDevice` = CLCompiler (source → CL binary) + CLAllocator (clCreateBuffer
//! under the LRU) + CLProgram (binary → clCreateKernel, NDRange launches) over
//! an `Rc` context/queue pair. Registers as "CL" through the same public
//! seams (`OpaqueT`/`BackendAllocator`/`Comp`/`Prog`/`UopRenderer` +
//! `register_device`) as the interp backend. Corpus constraints mirrored from
//! python: no images, no `_offset` (python's CLAllocator has none), blocking
//! copyin (python retains pending_copyin instead).
//!
//! The OpenCL library is loaded at run time (`cl_sys`), so a binary built with
//! `backend_cl` starts and falls back cleanly on machines without an ICD
//! loader.

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

use crate::dtype::DTypeId;
use crate::renderer::{Renderer, UopRenderer};
use crate::runtime::buffer::OpaqueT;
use crate::runtime::cl_sys::{
    CL_DEVICE_TYPE_DEFAULT, CL_DEVICE_TYPE_GPU, CL_MEM_READ_WRITE, CL_QUEUE_PROFILING_ENABLE,
    CommandQueue, Context, Device, Kernel, Mem, Program, cl_device_id, cl_mem, get_all_devices,
};
use crate::runtime::device::{
    BackendAllocator, Comp, Compiled, LruAllocator, Prog, TinyElf, register_device,
};
use crate::uop::UopId;
use crate::uop::ctx::UOpCtx;

// ---------------- device inner ----------------

/// Bounded source→binary cache (`CLCompiler.compile_cached`). tinygrad backs
/// this with a *disk* cache (`CCACHE`); the port keeps it in memory but caps it
/// with FIFO eviction so compiling many distinct kernels (BEAM search) can't
/// grow it without bound. Repeats still hit until evicted.
struct CompileCache {
    map: HashMap<String, Vec<u8>>,
    order: VecDeque<String>,
    cap: usize,
}

impl CompileCache {
    fn new() -> CompileCache {
        let cap = crate::helpers::env::get("CL_COMPILE_CACHE")
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

/// The CL device/context/queue pair shared by allocator, compiler and
/// programs (python: CLDevice fields).
#[allow(dead_code)]
struct ClInner {
    // Drop order: the queue must be released before the context, and both
    // before the device handle is dropped (the device is not refcounted).
    queue: CommandQueue,
    context: Context,
    dev: Device,
    device_name: String,
    driver_version: String,
    exts: String,
    compile_cache: RefCell<CompileCache>,
    /// Cached f32 epilogue kernel `(kernel, program)` for library GEMMs.
    epilogue: RefCell<Option<(Kernel, Program)>>,
}

impl ClInner {
    /// python `CLDevice.__init__`: GPU else DEFAULT device.
    fn open() -> ClInner {
        let ids: Vec<cl_device_id> = match get_all_devices(CL_DEVICE_TYPE_GPU) {
            Ok(ids) if !ids.is_empty() => ids,
            _ => get_all_devices(CL_DEVICE_TYPE_DEFAULT).expect("TGERR: no OpenCL devices"),
        };
        let dev = Device::new(*ids.first().expect("TGERR: no OpenCL devices"));
        let device_name = check(dev.name(), "CL device name");
        let driver_version = check(dev.driver_version(), "CL driver version");
        let exts = check(dev.extensions(), "CL extensions");
        eprintln!("CLDevice: opening {device_name} with version {driver_version}");
        let context = check(Context::from_device(&dev), "clCreateContext");
        let queue = check(
            CommandQueue::create(&context, dev.id(), CL_QUEUE_PROFILING_ENABLE),
            "clCreateCommandQueue",
        );
        ClInner {
            queue,
            context,
            dev,
            device_name,
            driver_version,
            exts,
            compile_cache: RefCell::new(CompileCache::new()),
            epilogue: RefCell::new(None),
        }
    }

    fn synchronize(&self) {
        check(self.queue.finish(), "clFinish");
    }

    /// `C = (C + bias[col]) * scale[col] + residual[row, col]`, in place.
    fn epilogue_kernel(&self) -> std::cell::RefMut<'_, Option<(Kernel, Program)>> {
        let mut slot = self.epilogue.borrow_mut();
        if slot.is_none() {
            let src = r#"
__kernel void autorust_epi_f32(__global float* c, __global const float* b,
    __global const float* s, __global const float* r,
    long total, long n, long m, long stride_c,
    long c_off, long b_off, long s_off, long r_off,
    int has_bias, int has_scale, int has_res) {
  long i = (long)get_global_id(0);
  if (i >= total) return;
  long col = i % n;
  long row = (i / n) % m;
  long bat = i / (n * m);
  long rel = bat * stride_c + row * n + col;
  long ci = c_off + rel;
  float v = c[ci];
  if (has_bias) v += b[b_off + col];
  if (has_scale) v *= s[s_off + col];
  if (has_res) v += r[r_off + rel];
  c[ci] = v;
}
"#;
            let program = check(
                Program::create_from_source(&self.context, src),
                "clCreateProgramWithSource(epilogue)",
            );
            let devices = [self.dev.id()];
            if let Err(e) = program.build(&devices, "") {
                let log = program.get_build_log(devices[0]).unwrap_or_default();
                panic!("TGERR: OpenCL epilogue build error: {e}\n{log}");
            }
            let kernel = check(
                Kernel::create(&program, "autorust_epi_f32"),
                "clCreateKernel(epilogue)",
            );
            *slot = Some((kernel, program));
        }
        slot
    }

    #[allow(clippy::too_many_arguments)]
    fn epilogue_apply(
        &self,
        c: crate::runtime::cl_sys::cl_mem,
        bias: Option<(crate::runtime::cl_sys::cl_mem, i64)>,
        scale: Option<(crate::runtime::cl_sys::cl_mem, i64)>,
        residual: Option<(crate::runtime::cl_sys::cl_mem, i64)>,
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
        let guard = self.epilogue_kernel();
        let (kernel, _program) = guard.as_ref().expect("epilogue kernel");
        let b_ptr = bias.map(|x| x.0).unwrap_or(c);
        let s_ptr = scale.map(|x| x.0).unwrap_or(c);
        let r_ptr = residual.map(|x| x.0).unwrap_or(c);
        let b_off = bias.map(|x| x.1).unwrap_or(0);
        let s_off = scale.map(|x| x.1).unwrap_or(0);
        let r_off = residual.map(|x| x.1).unwrap_or(0);
        let (has_bias, has_scale, has_res) = (
            bias.is_some() as i32,
            scale.is_some() as i32,
            residual.is_some() as i32,
        );
        check(kernel.set_arg(0, &c), "epilogue arg c");
        check(kernel.set_arg(1, &b_ptr), "epilogue arg b");
        check(kernel.set_arg(2, &s_ptr), "epilogue arg s");
        check(kernel.set_arg(3, &r_ptr), "epilogue arg r");
        check(kernel.set_arg(4, &total), "epilogue arg total");
        check(kernel.set_arg(5, &n), "epilogue arg n");
        check(kernel.set_arg(6, &m), "epilogue arg m");
        check(kernel.set_arg(7, &stride_c), "epilogue arg stride_c");
        check(kernel.set_arg(8, &c_off), "epilogue arg c_off");
        check(kernel.set_arg(9, &b_off), "epilogue arg b_off");
        check(kernel.set_arg(10, &s_off), "epilogue arg s_off");
        check(kernel.set_arg(11, &r_off), "epilogue arg r_off");
        check(kernel.set_arg(12, &has_bias), "epilogue arg has_bias");
        check(kernel.set_arg(13, &has_scale), "epilogue arg has_scale");
        check(kernel.set_arg(14, &has_res), "epilogue arg has_res");
        let gs = [total as usize, 1, 1];
        check(
            self.queue.enqueue_nd_range_kernel(kernel.get(), &gs, None),
            "clEnqueueNDRangeKernel(epilogue)",
        );
    }

    /// Run a host GEMM (CLBlast) described by the scheduler's `HostGemm`.
    fn host_gemm(
        &self,
        bufs: &[crate::runtime::buffer::OpaqueView],
        desc: &crate::uop::arg::DataArg,
    ) {
        use crate::runtime::clblast;
        use crate::schedule::host_gemm::HostGemm;
        let g = HostGemm::from_arg(&crate::uop::arg::Arg::Data(desc.clone()))
            .unwrap_or_else(|| panic!("TGERR: bad HostGemm descriptor"));
        assert_eq!(
            g.dtype,
            crate::dtype::FLOAT32,
            "CLBlast path only supports f32"
        );
        let mem = |slot: i64| -> crate::runtime::cl_sys::cl_mem {
            cl_of(&*bufs[slot as usize]).mem.get()
        };
        // CLBlast takes an explicit layout, so pass row-major and use the
        // transpose flags only when an operand is stored the other way round.
        let (trans_a, a_ld) = if g.a_k == 1 {
            (clblast::TRANSPOSE_NO, g.a_m)
        } else {
            (clblast::TRANSPOSE_YES, g.a_k)
        };
        let (trans_b, b_ld) = if g.b_n == 1 {
            (clblast::TRANSPOSE_NO, g.b_k)
        } else {
            (clblast::TRANSPOSE_YES, g.b_n)
        };
        let c_ld = g.c_m;
        let (a, b, c) = (mem(g.a_slot), mem(g.b_slot), mem(g.c_slot));
        if let Err(e) = unsafe {
            clblast::gemm(
                self.queue.raw(),
                trans_a,
                trans_b,
                g.m as usize,
                g.n as usize,
                g.k as usize,
                g.alpha,
                a,
                g.a_off as usize,
                a_ld as usize,
                b,
                g.b_off as usize,
                b_ld as usize,
                c,
                g.c_off as usize,
                c_ld as usize,
                g.batch as usize,
                g.a_b.max(0) as usize,
                g.b_b.max(0) as usize,
                g.c_b.max(0) as usize,
            )
        } {
            panic!(
                "TGERR: {e} (m={} n={} k={} batch={})",
                g.m, g.n, g.k, g.batch
            );
        }
        if g.bias_slot >= 0 || g.scale_slot >= 0 || g.res_slot >= 0 {
            let opt = |slot: i64, off: i64| -> Option<(crate::runtime::cl_sys::cl_mem, i64)> {
                if slot >= 0 {
                    Some((mem(slot), off))
                } else {
                    None
                }
            };
            self.epilogue_apply(
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
    }
}

fn check<T, E: std::fmt::Display>(r: Result<T, E>, what: &str) -> T {
    r.unwrap_or_else(|e| panic!("TGERR: OpenCL error in {what}: {e}"))
}

// ---------------- opaque ----------------

/// `cl_mem` handle wrapped as the backend opaque (python `cl.cl_mem`).
struct ClOpaque {
    mem: Mem,
    size: usize,
}

impl OpaqueT for ClOpaque {
    fn size(&self) -> usize {
        self.size
    }
    fn storage_id(&self) -> usize {
        self.mem.get() as usize
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn cl_of(o: &dyn OpaqueT) -> &ClOpaque {
    o.as_any()
        .downcast_ref::<ClOpaque>()
        .expect("TGERR: expected CL opaque")
}

// ---------------- allocator ----------------

struct ClAllocator {
    inner: Rc<ClInner>,
}

impl BackendAllocator for ClAllocator {
    fn _alloc(&self, size: usize) -> Rc<dyn OpaqueT> {
        let mem = check(
            Mem::create(
                &self.inner.context,
                CL_MEM_READ_WRITE,
                size,
                std::ptr::null_mut(),
            ),
            "clCreateBuffer",
        );
        Rc::new(ClOpaque { mem, size })
    }
    fn _free(&self, o: Rc<dyn OpaqueT>) {
        drop(o); // ClOpaque drop => Mem drop => clReleaseMemObject
    }
    fn _copyin(&self, dest: &dyn OpaqueT, src: &[u8]) {
        check(
            self.inner
                .queue
                .enqueue_write_buffer(cl_of(dest).mem.get(), true, 0, src),
            "clEnqueueWriteBuffer",
        );
    }
    fn _copyout(&self, dest: &mut [u8], src: &dyn OpaqueT) {
        check(
            self.inner
                .queue
                .enqueue_read_buffer(cl_of(src).mem.get(), true, 0, dest),
            "clEnqueueReadBuffer",
        );
        self.inner.synchronize();
    }
    fn _copy_into(&self, dest: &dyn OpaqueT, src: &dyn OpaqueT) {
        // Direct device-to-device copy (`clEnqueueCopyBuffer`); no host transfer.
        assert_eq!(dest.size(), src.size(), "TGERR: copy size mismatch");
        check(
            self.inner.queue.enqueue_copy_buffer(
                cl_of(src).mem.get(),
                cl_of(dest).mem.get(),
                0,
                0,
                src.size(),
            ),
            "clEnqueueCopyBuffer",
        );
        // `_copyout` is blocking; keep the same visibility guarantee.
        self.inner.synchronize();
    }
    fn _offset(&self, _buf: &dyn OpaqueT, _size: usize, _offset: usize) -> Rc<dyn OpaqueT> {
        panic!("TGERR: need offset"); // python's CLAllocator has no _offset either
    }
}

// ---------------- compiler ----------------

struct ClCompiler {
    inner: Rc<ClInner>,
}

impl Comp for ClCompiler {
    /// `CLCompiler.compile` — build from source, fetch the device binary.
    fn compile(&self, src: &str) -> Vec<u8> {
        let program = check(
            Program::create_from_source(&self.inner.context, src),
            "clCreateProgramWithSource",
        );
        let devices = [self.inner.dev.id()];
        if let Err(e) = program.build(&devices, "") {
            let log = program.get_build_log(devices[0]).unwrap_or_default();
            panic!("TGERR: OpenCL Compile Error\n--- src ---\n{src}\n--- log ---\n{log}\n{e}");
        }
        check(program.get_binaries(), "CL_PROGRAM_BINARIES")
            .into_iter()
            .next()
            .unwrap_or_default()
    }

    fn compile_cached(&self, src: &str) -> Vec<u8> {
        // in-memory source cache (python: diskcache under CCACHE), FIFO-capped
        // so a BEAM search can't grow it without bound.
        if let Some(v) = self.inner.compile_cache.borrow().get(src) {
            return v;
        }
        let lib = self.compile(src);
        self.inner.compile_cache.borrow_mut().put(src, lib.clone());
        lib
    }
}

// ---------------- program ----------------

/// `CLProgram` — create the kernel from the compiled binary and launch it.
#[allow(dead_code)]
pub struct ClProgram {
    /// Released before `program`; field order is drop order.
    kernel: Kernel,
    program: Program,
    inner: Rc<ClInner>,
    signature: Vec<(i64, DTypeId)>,
}

impl ClProgram {
    fn new(cl_compiler: &ClCompiler, inner: Rc<ClInner>, obj: &TinyElf) -> ClProgram {
        // python CLProgram.__init__: re-run the CL compiler on the elf lib
        // (the TinyELF holds the *rendered source* — the empty-compiler seam —
        // and CLCompiler turns it into the device binary here)
        let src = std::str::from_utf8(&obj.lib).expect("TGERR: CL elf lib not utf8");
        let compiled = cl_compiler.compile_cached(src);
        let devices = [inner.dev.id()];
        let binaries: Vec<&[u8]> = vec![&compiled];
        let program = check(
            Program::create_from_binary(&inner.context, &devices, &binaries),
            "clCreateProgramWithBinary",
        );
        check(program.build(&devices, ""), "clBuildProgram");
        let kernel = check(Kernel::create(&program, &obj.name), "clCreateKernel");
        ClProgram {
            inner,
            kernel,
            program,
            signature: obj.signature.clone(),
        }
    }
}

impl Prog for ClProgram {
    fn call(
        &mut self,
        bufs: &[Rc<dyn OpaqueT>],
        global_size: [usize; 3],
        local_size: Option<[usize; 3]>,
        vals: &[i64],
        wait: bool,
    ) -> Option<f64> {
        // python: set each signature slot from bufs[slot] or vals[slot-len(bufs)]
        for (i, (slot, dt)) in self.signature.iter().enumerate() {
            if (*slot as usize) < bufs.len() {
                let mem: cl_mem = cl_of(&*bufs[*slot as usize]).mem.get();
                check(self.kernel.set_arg(i as u32, &mem), "clSetKernelArg");
            } else {
                let v = vals[(*slot as usize) - bufs.len()];
                match dt.bitsize() {
                    8 => set_arg_u8(&self.kernel, i as u32, v as u8),
                    16 => set_arg_u16(&self.kernel, i as u32, v as u16),
                    32 => set_arg_i32(&self.kernel, i as u32, v as i32),
                    64 => set_arg_i64(&self.kernel, i as u32, v),
                    b => panic!("TGERR: CL val arg bitsize {b}"),
                }
            }
        }
        // python: local_size scales global_size
        let (gs, ls) = match local_size {
            Some(l) => (
                [
                    global_size[0] * l[0],
                    global_size[1] * l[1],
                    global_size[2] * l[2],
                ],
                Some(l),
            ),
            None => (global_size, None),
        };
        let st = std::time::Instant::now();
        let ev = check(
            self.inner
                .queue
                .enqueue_nd_range_kernel(self.kernel.get(), &gs, ls.as_ref()),
            "clEnqueueNDRangeKernel",
        );
        if wait {
            // tinygrad waits on the event (`clWaitForEvents`, ops_cl.py:72), not
            // `clFinish` on the shared queue — the latter is prone to being
            // invalidated after many throwaway BEAM programs.
            check(ev.wait(), "clWaitForEvents");
            let start = check(ev.profiling_command_start(), "event start");
            let end = check(ev.profiling_command_end(), "event end");
            return Some(end.saturating_sub(start) as f64 * 1e-9);
        }
        let _ = st;
        None
    }
}

fn set_arg_u8(k: &Kernel, i: u32, v: u8) {
    check(k.set_arg(i, &v), "clSetKernelArg");
}
fn set_arg_u16(k: &Kernel, i: u32, v: u16) {
    check(k.set_arg(i, &v), "clSetKernelArg");
}
fn set_arg_i32(k: &Kernel, i: u32, v: i32) {
    check(k.set_arg(i, &v), "clSetKernelArg");
}
fn set_arg_i64(k: &Kernel, i: u32, v: i64) {
    check(k.set_arg(i, &v), "clSetKernelArg");
}

// ---------------- renderer ----------------

/// `OpenCLRenderer` — cstyle with the OpenCL C flavor.
pub struct OclUopRenderer {
    facts: Renderer,
}

impl UopRenderer for OclUopRenderer {
    fn facts(&self) -> &Renderer {
        &self.facts
    }
    fn render(&self, ctx: &mut UOpCtx, lst: &[UopId]) -> String {
        crate::renderer::cstyle::render_opencl(ctx, lst)
    }
}

// ---------------- device + registration ----------------

/// Whether an OpenCL device is present, without opening a context/queue and
/// without panicking (the default-device probe). The ICD loader is loaded
/// dynamically; a missing library simply returns `false`.
pub fn available() -> bool {
    if crate::runtime::cl_sys::ClLib::get().is_none() {
        return false;
    }
    match get_all_devices(CL_DEVICE_TYPE_GPU) {
        Ok(ids) if !ids.is_empty() => true,
        _ => get_all_devices(CL_DEVICE_TYPE_DEFAULT)
            .map(|ids| !ids.is_empty())
            .unwrap_or(false),
    }
}

/// The `CLDevice` composition + `Device._DeviceMap["CL"]` assignment.
pub fn register() {
    // A binary built with `backend_cl` must start on machines without OpenCL;
    // only register the device when the loader and a device are actually there.
    if !available() {
        return;
    }
    let factory: crate::runtime::device::DeviceFactory = Rc::new(|| {
        let inner = Rc::new(ClInner::open());
        let facts = {
            // python: arch = ",".join(device_exts) — half/double gated on the
            // real device's cl_khr_* extensions (cstyle.py:339).
            Renderer::opencl_arch(&inner.exts)
        };
        let renderer: Box<dyn UopRenderer> = Box::new(OclUopRenderer { facts });
        // python: OpenCLRenderer.compiler is the base empty Compiler — the elf
        // lib is the rendered source; the real CL compile happens in CLProgram
        let compiler: Rc<dyn Comp> = Rc::new(crate::runtime::device::IdentityCompiler::default());
        let allocator = LruAllocator::with_backend(Box::new(ClAllocator {
            inner: inner.clone(),
        }));
        let ctor: crate::runtime::device::RuntimeCtor = {
            let inner = inner.clone();
            Rc::new(move |elf| {
                let cl_compiler = ClCompiler {
                    inner: inner.clone(),
                };
                Rc::new(RefCell::new(ClProgram::new(
                    &cl_compiler,
                    inner.clone(),
                    elf,
                )))
            })
        };
        let dev = Compiled::new("CL", renderer, allocator, compiler, ctor);
        if crate::runtime::clblast::library_present() {
            let inner2 = inner.clone();
            *dev.host_exec.borrow_mut() =
                Some(Box::new(move |bufs, desc| inner2.host_gemm(bufs, desc)));
        }
        dev
    });
    register_device("CL", factory);
}
