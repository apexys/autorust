//! `runtime/ops_metal.py` port — the Apple Metal backend (`objc2` behind the
//! `backend_metal` feature; macOS only).
//!
//! `MetalDevice` is a MetalAllocator (`newBufferWithLength:options:` under the
//! LRU) plus a MetalProgram (`newLibraryWithSource:` → `newFunctionWithName:` →
//! `newComputePipelineState...` → `dispatchThreadgroups:`) over an `Rc`
//! device/command-queue pair. Registers as "METAL" through the same public seams
//! (`OpaqueT`/`BackendAllocator`/`Comp`/`Prog`/`UopRenderer` +
//! `register_device`) as the other backends.
//!
//! The crate links the OS-provided Objective-C runtime and Metal framework (no
//! Xcode or Metal toolchain is required): shaders are compiled at run time by
//! `newLibraryWithSource:options:error:`, exactly like tinygrad's non-Xcode
//! path. Buffers use `MTLResourceStorageModeShared`, so host upload/download is
//! a CPU copy over the buffer's `contents` pointer.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::rc::Rc;

use objc2::AnyThread;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineDescriptor,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLFunction, MTLLibrary,
    MTLMathMode, MTLPipelineOption, MTLResourceOptions, MTLSize,
};
use objc2_metal_performance_shaders::{
    MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication,
};

use crate::dtype::DTypeId;
use crate::renderer::{Renderer, UopRenderer};
use crate::runtime::buffer::OpaqueT;
use crate::runtime::device::{
    BackendAllocator, Comp, Compiled, IdentityCompiler, LruAllocator, MemoryInfo, Prog, TinyElf,
    register_device_with_limit,
};
use crate::uop::UopId;
use crate::uop::ctx::UOpCtx;

// `MTLCreateSystemDefaultDevice` requires CoreGraphics to be loaded (Metal
// framework docs); link it explicitly so a stripped binary still works.
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

fn check_error(cb: &ProtocolObject<dyn MTLCommandBuffer>) {
    if let Some(err) = cb.error() {
        panic!(
            "TGERR: Metal command buffer error: {}",
            err.localizedDescription()
        );
    }
}

/// Number of host GEMMs encoded through MetalPerformanceShaders (diagnostics).
pub static MPS_GEMMS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

// ---------------- device inner ----------------

/// MPS GEMM kernel cache key: `(m, n, k, trans_a, trans_b, alpha_bits, dtype)`.
type MpsOpKey = (i64, i64, i64, bool, bool, u32, u16);
/// MPS matrix descriptor cache key: `(rows, cols, leading_dim, dtype)`.
type MpsDescKey = (i64, i64, i64, u16);

/// The Metal device/queue pair shared by allocator, programs, and the
/// synchronize path (python: `MetalDevice` fields).
struct MetalInner {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// Command buffers committed without `wait` (python `mtl_buffers_in_flight`).
    in_flight: RefCell<VecDeque<Retained<ProtocolObject<dyn MTLCommandBuffer>>>>,
    /// Cached MPS GEMM kernels keyed by `(m,n,k,trans_a,trans_b,alpha,dtype)`.
    mps_ops: RefCell<std::collections::HashMap<MpsOpKey, Retained<MPSMatrixMultiplication>>>,
    /// Cached MPS matrix descriptors keyed by `(rows,cols,ld,dtype)`.
    mps_descs: RefCell<std::collections::HashMap<MpsDescKey, Retained<MPSMatrixDescriptor>>>,
}

impl MetalInner {
    fn open() -> MetalInner {
        let device = MTLCreateSystemDefaultDevice()
            .expect("TGERR: no Metal device (MTLCreateSystemDefaultDevice returned nil)");
        let device_name = device.name().to_string();
        let queue = device
            .newCommandQueue()
            .expect("TGERR: newCommandQueueWithMaxCommandBufferCount");
        eprintln!("MetalDevice: opening {device_name}");
        MetalInner {
            device,
            queue,
            in_flight: RefCell::new(VecDeque::new()),
            mps_ops: RefCell::new(std::collections::HashMap::new()),
            mps_descs: RefCell::new(std::collections::HashMap::new()),
        }
    }

    /// Wait for every committed command buffer (python `MetalDevice.synchronize`).
    fn synchronize(&self) {
        let pending: Vec<_> = self.in_flight.borrow_mut().drain(..).collect();
        for cb in pending {
            cb.waitUntilCompleted();
            check_error(&cb);
        }
    }

    /// Run a host GEMM (MetalPerformanceShaders `MPSMatrixMultiplication`).
    ///
    /// Mirrors the cuBLAS/CLBlast paths: the scheduler has already extracted a
    /// plain strided matrix product; this only has to bind the three buffers
    /// and encode the library kernel. Batches are encoded as separate GEMMs so
    /// a shared operand stride (`*_b == 0`) is naturally supported. The caller
    /// (`host_exec`) leaves the command buffer queued; subsequent Metal work is
    /// ordered after it on the same queue.
    fn host_gemm(
        &self,
        bufs: &[crate::runtime::buffer::OpaqueView],
        desc: &crate::uop::arg::DataArg,
    ) {
        use crate::schedule::host_gemm::HostGemm;
        let g = HostGemm::from_arg(&crate::uop::arg::Arg::Data(desc.clone()))
            .unwrap_or_else(|| panic!("TGERR: bad HostGemm descriptor"));
        MPS_GEMMS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let isz = g.dtype.itemsize();
        let c_isz = g.c_dtype.itemsize();
        let mps_dtype = |dt: DTypeId| match dt {
            crate::dtype::FLOAT32 => MPSDataType::Float32,
            crate::dtype::FLOAT16 => MPSDataType::Float16,
            other => panic!("TGERR: MPS GEMM unsupported dtype {other:?}"),
        };
        let a = metal_of(&*bufs[g.a_slot as usize]);
        let b = metal_of(&*bufs[g.b_slot as usize]);
        let c = metal_of(&*bufs[g.c_slot as usize]);
        // An MPS matrix is row-major. A transposed operand is described by its
        // stored shape; a column-major result is not representable without a
        // second transpose, so `extract` keeps that on the generated kernel.
        assert_eq!(g.c_n, 1, "TGERR: MPS GEMM requires a row-major C");
        let (trans_a, a_rows, a_cols, a_ld) = if g.a_k == 1 {
            (false, g.m, g.k, g.a_m)
        } else {
            (true, g.k, g.m, g.a_k)
        };
        let (trans_b, b_rows, b_cols, b_ld) = if g.b_n == 1 {
            (false, g.k, g.n, g.b_k)
        } else {
            (true, g.n, g.k, g.b_n)
        };
        let desc_for = |rows: i64,
                        cols: i64,
                        ld: i64,
                        dt: DTypeId,
                        esz: usize|
         -> Retained<MPSMatrixDescriptor> {
            let key = (rows, cols, ld, dt.0);
            if let Some(d) = self.mps_descs.borrow().get(&key) {
                return d.clone();
            }
            let d = unsafe {
                MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                    rows as usize,
                    cols as usize,
                    (ld * esz as i64) as usize,
                    mps_dtype(dt),
                )
            };
            self.mps_descs.borrow_mut().insert(key, d.clone());
            d
        };
        let a_desc = desc_for(a_rows, a_cols, a_ld, g.dtype, isz);
        let b_desc = desc_for(b_rows, b_cols, b_ld, g.dtype, isz);
        let c_desc = desc_for(g.m, g.n, g.c_m, g.c_dtype, c_isz);
        let op_key = (
            g.m,
            g.n,
            g.k,
            trans_a,
            trans_b,
            g.alpha.to_bits(),
            g.dtype.0,
        );
        let mm = if let Some(mm) = self.mps_ops.borrow().get(&op_key) {
            mm.clone()
        } else {
            let mm = unsafe {
                MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
                    MPSMatrixMultiplication::alloc(),
                    &self.device,
                    trans_a,
                    trans_b,
                    g.m as usize,
                    g.n as usize,
                    g.k as usize,
                    g.alpha as f64,
                    0.0,
                )
            };
            self.mps_ops.borrow_mut().insert(op_key, mm.clone());
            mm
        };
        let cb = self
            .queue
            .commandBuffer()
            .expect("TGERR: Metal commandBuffer (MPS)");
        for batch in 0..g.batch {
            let elem = |frame: i64, stride: i64, esz: usize| -> usize {
                ((frame + batch * stride) * esz as i64) as usize
            };
            let a_off = elem(g.a_off, g.a_b, isz);
            let b_off = elem(g.b_off, g.b_b, isz);
            let c_off = elem(g.c_off, g.c_b, c_isz);
            let a_mat = unsafe {
                MPSMatrix::initWithBuffer_offset_descriptor(
                    MPSMatrix::alloc(),
                    &a.buf,
                    a_off,
                    &a_desc,
                )
            };
            let b_mat = unsafe {
                MPSMatrix::initWithBuffer_offset_descriptor(
                    MPSMatrix::alloc(),
                    &b.buf,
                    b_off,
                    &b_desc,
                )
            };
            let c_mat = unsafe {
                MPSMatrix::initWithBuffer_offset_descriptor(
                    MPSMatrix::alloc(),
                    &c.buf,
                    c_off,
                    &c_desc,
                )
            };
            unsafe {
                mm.setBatchStart(0);
                mm.setBatchSize(1);
                mm.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
                    &cb, &a_mat, &b_mat, &c_mat,
                );
            }
        }
        cb.commit();
        self.in_flight.borrow_mut().push_back(cb);
    }
}

// ---------------- opaque ----------------

/// A Metal buffer plus the (offset, size) window this opaque represents.
/// `newBufferWithLength` allocations are shared storage, so `contents` is a
/// valid host pointer (python `MetalBuffer`).
struct MetalOpaque {
    buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    size: usize,
    offset: usize,
}

impl OpaqueT for MetalOpaque {
    fn size(&self) -> usize {
        self.size
    }
    fn storage_id(&self) -> usize {
        Retained::as_ptr(&self.buf) as usize
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn metal_of(o: &dyn OpaqueT) -> &MetalOpaque {
    o.as_any()
        .downcast_ref::<MetalOpaque>()
        .expect("TGERR: expected Metal opaque")
}

// ---------------- allocator ----------------

struct MetalAllocator {
    inner: Rc<MetalInner>,
}

impl BackendAllocator for MetalAllocator {
    fn _alloc(&self, size: usize) -> Rc<dyn OpaqueT> {
        let buf = self
            .inner
            .device
            .newBufferWithLength_options(size, MTLResourceOptions::StorageModeShared)
            .unwrap_or_else(|| panic!("TGERR: Metal OOM while allocating {size} bytes"));
        Rc::new(MetalOpaque {
            buf,
            size,
            offset: 0,
        })
    }
    fn _free(&self, o: Rc<dyn OpaqueT>) {
        drop(o); // Retained<MTLBuffer> release
    }
    fn _copyin(&self, dest: &dyn OpaqueT, src: &[u8]) {
        self.inner.synchronize();
        let dest = metal_of(dest);
        assert_eq!(
            src.len(),
            dest.size,
            "TGERR: Metal copyin size mismatch ({} != {})",
            src.len(),
            dest.size
        );
        let ptr = dest.buf.contents().as_ptr() as *mut u8;
        let ptr = ptr.wrapping_add(dest.offset);
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), ptr, src.len()) };
    }
    fn _copyout(&self, dest: &mut [u8], src: &dyn OpaqueT) {
        self.inner.synchronize();
        let src = metal_of(src);
        assert_eq!(
            dest.len(),
            src.size,
            "TGERR: Metal copyout size mismatch ({} != {})",
            dest.len(),
            src.size
        );
        let ptr = src.buf.contents().as_ptr() as *const u8;
        let ptr = ptr.wrapping_add(src.offset);
        unsafe { std::ptr::copy_nonoverlapping(ptr, dest.as_mut_ptr(), dest.len()) };
    }
    fn _copy_into(&self, dest: &dyn OpaqueT, src: &dyn OpaqueT) {
        // Direct device-to-device copy with a blit encoder (python
        // `MetalAllocator.transfer`); no host round-trip.
        let (d, s) = (metal_of(dest), metal_of(src));
        assert_eq!(d.size, s.size, "TGERR: copy size mismatch");
        self.inner.synchronize();
        let cb = self
            .inner
            .queue
            .commandBuffer()
            .expect("TGERR: Metal commandBuffer");
        let enc = cb
            .blitCommandEncoder()
            .expect("TGERR: Metal blitCommandEncoder");
        unsafe {
            enc.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &s.buf, s.offset, &d.buf, d.offset, s.size,
            );
        }
        enc.endEncoding();
        cb.commit();
        self.inner.in_flight.borrow_mut().push_back(cb);
    }
    fn _offset(&self, buf: &dyn OpaqueT, size: usize, offset: usize) -> Rc<dyn OpaqueT> {
        // Unlike OpenCL, Metal has real offset views (`MetalBuffer` carries a
        // (buf, size, offset) triple); share the parent allocation.
        let b = metal_of(buf);
        assert!(
            offset + size <= b.size,
            "TGERR: Metal offset view out of range"
        );
        Rc::new(MetalOpaque {
            buf: b.buf.clone(),
            size,
            offset: b.offset + offset,
        })
    }
    fn memory_info(&self) -> Option<MemoryInfo> {
        // Metal's recommended working set is the budget the OS wants us to stay
        // under; `currentAllocatedSize` is what this process has allocated.
        let total = self.inner.device.recommendedMaxWorkingSetSize() as usize;
        if total == 0 {
            return None;
        }
        let used = self.inner.device.currentAllocatedSize();
        Some(MemoryInfo {
            total,
            free: Some(total.saturating_sub(used)),
        })
    }
}

// ---------------- program ----------------

/// `MetalProgram` — the compiled source's library/function/pipeline; launches
/// with `dispatchThreadgroups:threadsPerThreadgroup:`.
pub struct MetalProgram {
    /// Field order is drop order; the library must outlive the function and the
    /// function must outlive the pipeline state.
    _library: Retained<ProtocolObject<dyn MTLLibrary>>,
    _function: Retained<ProtocolObject<dyn MTLFunction>>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    inner: Rc<MetalInner>,
    signature: Vec<(i64, DTypeId)>,
}

impl MetalProgram {
    fn new(inner: Rc<MetalInner>, obj: &TinyElf) -> MetalProgram {
        // The compiler seam is the empty identity compiler (like CL): the ELF
        // lib is the rendered Metal source, compiled here at runtime.
        assert!(
            !obj.signature
                .iter()
                .any(|(_, dt)| *dt == crate::dtype::FLOAT64),
            "TGERR: Metal has no hardware double; FLOAT64 is lowered to FLOAT32 by the Tensor API (raw f64 UOp graphs are unsupported)"
        );
        let src = std::str::from_utf8(&obj.lib).expect("TGERR: Metal elf lib not utf8");
        // Apple GPUs expose at most 31 `[[buffer(n)]]` slots. A fused kernel
        // that needs more cannot be compiled; fail with an actionable message
        // instead of the driver's "no buffer resource location" error.
        assert!(
            obj.signature.len() <= 31,
            "TGERR: Metal kernel {} needs {} buffers; Apple GPUs support at most 31",
            obj.name,
            obj.signature.len()
        );
        let ns_src = NSString::from_str(src);
        // tinygrad sets `fastMathEnabled` from `METAL_FAST_MATH` (default 0).
        // An explicit non-fast mode matters for correctness: under fast math
        // Metal folds `x + -inf` to `x`, which silently drops attention masks.
        let options = MTLCompileOptions::new();
        let fast_math = crate::helpers::env::get("METAL_FAST_MATH")
            .map(|v| v != "0")
            .unwrap_or(false);
        options.setMathMode(if fast_math {
            MTLMathMode::Fast
        } else {
            MTLMathMode::Safe
        });
        let library = match inner
            .device
            .newLibraryWithSource_options_error(&ns_src, Some(&options))
        {
            Ok(lib) => lib,
            Err(e) => panic!(
                "TGERR: Metal Compile Error\n--- src ---\n{src}\n--- log ---\n{}",
                e.localizedDescription()
            ),
        };
        let name = NSString::from_str(&obj.name);
        let function = library
            .newFunctionWithName(&name)
            .unwrap_or_else(|| panic!("TGERR: Metal function {} not found", obj.name));
        let desc = MTLComputePipelineDescriptor::new();
        desc.setComputeFunction(Some(&function));
        desc.setSupportIndirectCommandBuffers(true);
        let pipeline = match inner
            .device
            .newComputePipelineStateWithDescriptor_options_reflection_error(
                &desc,
                MTLPipelineOption::None,
                None,
            ) {
            Ok(p) => p,
            Err(e) => panic!(
                "TGERR: Metal pipeline creation failed: {}",
                e.localizedDescription()
            ),
        };
        MetalProgram {
            _library: library,
            _function: function,
            pipeline,
            inner,
            signature: obj.signature.clone(),
        }
    }
}

impl Prog for MetalProgram {
    fn call(
        &mut self,
        bufs: &[Rc<dyn OpaqueT>],
        global_size: [usize; 3],
        local_size: Option<[usize; 3]>,
        vals: &[i64],
        wait: bool,
    ) -> Option<f64> {
        let cb = self
            .inner
            .queue
            .commandBuffer()
            .expect("TGERR: Metal commandBuffer");
        let enc = cb
            .computeCommandEncoder()
            .expect("TGERR: Metal computeCommandEncoder");
        enc.setComputePipelineState(&self.pipeline);
        // Kernel argument order is the ELF signature order. A slot below
        // `bufs.len()` is a buffer; beyond it the value comes from `vals` at
        // `slot - bufs.len()` (the same contract as CL/CUDA). Hand-written
        // kernels include their scalar arguments in the signature.
        for (i, (slot, dt)) in self.signature.iter().enumerate() {
            if (*slot as usize) < bufs.len() {
                let o = &bufs[*slot as usize];
                let m = metal_of(&**o);
                unsafe { enc.setBuffer_offset_atIndex(Some(&m.buf), m.offset, i) };
            } else {
                let v = vals
                    .get((*slot as usize) - bufs.len())
                    .unwrap_or_else(|| panic!("TGERR: Metal missing value slot {slot}"));
                let bits = dt.bitsize();
                let bytes = (bits / 8) as usize;
                // Keep the value in a local that outlives the `setBytes` call;
                // taking a reference to a temporary here would dangle.
                let mut raw = [0u8; 8];
                match bits {
                    8 => raw[0] = *v as u8,
                    16 => raw[..2].copy_from_slice(&(*v as u16).to_ne_bytes()),
                    32 => raw[..4].copy_from_slice(&(*v as i32).to_ne_bytes()),
                    64 => raw.copy_from_slice(&v.to_ne_bytes()),
                    b => panic!("TGERR: Metal val arg bitsize {b}"),
                }
                let ptr = NonNull::from(&raw).cast::<c_void>();
                unsafe { enc.setBytes_length_atIndex(ptr, bytes, i) };
            }
        }
        let gs = MTLSize {
            width: global_size[0],
            height: global_size[1],
            depth: global_size[2],
        };
        let threads = local_size.unwrap_or([1, 1, 1]);
        let ls = MTLSize {
            width: threads[0],
            height: threads[1],
            depth: threads[2],
        };
        enc.dispatchThreadgroups_threadsPerThreadgroup(gs, ls);
        enc.endEncoding();
        let st = std::time::Instant::now();
        cb.commit();
        if wait {
            cb.waitUntilCompleted();
            check_error(&cb);
            return Some(st.elapsed().as_secs_f64());
        }
        self.inner.in_flight.borrow_mut().push_back(cb);
        None
    }
}

// ---------------- renderer ----------------

/// `MetalRenderer` — cstyle with the Metal Shading Language flavor.
pub struct MetalUopRenderer {
    facts: Renderer,
}

impl UopRenderer for MetalUopRenderer {
    fn facts(&self) -> &Renderer {
        &self.facts
    }
    fn render(&self, ctx: &mut UOpCtx, lst: &[UopId]) -> String {
        crate::renderer::cstyle::render_metal(ctx, lst)
    }
}

// ---------------- device + registration ----------------

/// Whether a Metal device is present (the default-device probe). Never panics.
pub fn available() -> bool {
    MTLCreateSystemDefaultDevice().is_some()
}

/// Whether the host GEMM path (MetalPerformanceShaders) can be used.
pub fn mps_available() -> bool {
    available() && objc2::runtime::AnyClass::get(c"MPSMatrixMultiplication").is_some()
}

/// The `MetalDevice` composition + `Device._DeviceMap["METAL"]` assignment.
pub fn register() {
    if !available() {
        return;
    }
    let factory: crate::runtime::device::DeviceFactory = Rc::new(|| {
        let inner = Rc::new(MetalInner::open());
        let renderer: Box<dyn UopRenderer> = Box::new(MetalUopRenderer {
            facts: Renderer::metal(),
        });
        // Same composition as CL: the compiler seam is the base empty compiler;
        // `MetalProgram` compiles the rendered source with Metal at runtime.
        let compiler: Rc<dyn Comp> = Rc::new(IdentityCompiler::default());
        let allocator = LruAllocator::with_backend(Box::new(MetalAllocator {
            inner: inner.clone(),
        }));
        let ctor_inner = inner.clone();
        let ctor: crate::runtime::device::RuntimeCtor =
            Rc::new(move |elf| Rc::new(RefCell::new(MetalProgram::new(ctor_inner.clone(), elf))));
        let dev = Compiled::new("METAL", renderer, allocator, compiler, ctor);
        if mps_available() {
            let inner2 = inner.clone();
            *dev.host_exec.borrow_mut() =
                Some(Box::new(move |bufs, desc| inner2.host_gemm(bufs, desc)));
        }
        dev
    });
    register_device_with_limit("METAL", Renderer::metal().max_buffers, factory);
}
