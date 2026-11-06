//! Optional dynamically loaded cuBLAS.
//!
//! The CUDA backend renders its own kernels, but a library GEMM is much faster
//! for large f32 contractions (and is the only easy way to use tensor cores on
//! prime shapes such as ViT's 197-token sequence). This module loads
//! `libcublas` at run time with the same discipline as `ops_cl.rs`: no CUDA
//! toolkit is needed at build time, and a missing library simply disables the
//! fast path (`available()` returns `false`).
//!
//! The scheduler tags a pure `MUL -> REDUCE(ADD)` kernel with a [`HostGemm`]
//! descriptor (see `schedule/host_gemm.rs`); `engine/realize.rs` then calls
//! [`gemm`] instead of launching generated code.
//!
//! [`HostGemm`]: crate::schedule::host_gemm::HostGemm

use std::ffi::c_void;
use std::sync::OnceLock;

use libloading::Library;

/// cuBLAS operation selector (`CUBLAS_OP_N`).
pub const OP_N: i32 = 0;
/// cuBLAS operation selector (`CUBLAS_OP_T`).
pub const OP_T: i32 = 1;

/// `cudaDataType_t` for f32.
pub const CUDA_R_32F: i32 = 0;
/// `cudaDataType_t` for f16.
pub const CUDA_R_16F: i32 = 2;
/// `cudaDataType_t` for bf16.
pub const CUDA_R_16BF: i32 = 14;
/// `cublasComputeType_t` for f32 accumulation.
pub const COMPUTE_32F: i32 = 68;
/// `cublasComputeType_t` for fast TF32 accumulation.
pub const COMPUTE_32F_FAST_TF32: i32 = 77;
/// `cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT`.
pub const GEMM_DEFAULT: i32 = -1;

type CublasCreate = unsafe extern "C" fn(*mut *mut c_void) -> i32;
type CublasDestroy = unsafe extern "C" fn(*mut c_void) -> i32;
type CublasSetStream = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32;
type CublasSetMathMode = unsafe extern "C" fn(*mut c_void, u32) -> i32;
type CublasSgemm = unsafe extern "C" fn(
    *mut c_void,
    i32,
    i32,
    i32,
    i32,
    i32,
    *const f32,
    *const f32,
    i32,
    *const f32,
    i32,
    *const f32,
    *mut f32,
    i32,
) -> i32;
#[allow(clippy::too_many_arguments)]
type CublasSgemmStridedBatched = unsafe extern "C" fn(
    *mut c_void,
    i32,
    i32,
    i32,
    i32,
    i32,
    *const f32,
    *const f32,
    i32,
    i64,
    *const f32,
    i32,
    i64,
    *const f32,
    *mut f32,
    i32,
    i64,
    i32,
) -> i32;
#[allow(clippy::too_many_arguments)]
type CublasGemmEx = unsafe extern "C" fn(
    *mut c_void,
    i32,
    i32,
    i32,
    i32,
    i32,
    *const c_void,
    *const c_void,
    i32,
    i32,
    *const c_void,
    i32,
    i32,
    *const c_void,
    *mut c_void,
    i32,
    i32,
    i32,
    i32,
) -> i32;
#[allow(clippy::too_many_arguments)]
type CublasGemmStridedBatchedEx = unsafe extern "C" fn(
    *mut c_void,
    i32,
    i32,
    i32,
    i32,
    i32,
    *const c_void,
    *const c_void,
    i32,
    i32,
    i64,
    *const c_void,
    i32,
    i32,
    i64,
    *const c_void,
    *mut c_void,
    i32,
    i32,
    i64,
    i32,
    i32,
    i32,
) -> i32;

struct CublasLib {
    _lib: Library,
    create: CublasCreate,
    destroy: CublasDestroy,
    set_stream: CublasSetStream,
    set_math_mode: Option<CublasSetMathMode>,
    sgemm: CublasSgemm,
    sgemm_strided_batched: CublasSgemmStridedBatched,
    gemm_ex: CublasGemmEx,
    gemm_strided_batched_ex: CublasGemmStridedBatchedEx,
}

// Function pointers and the (thread-safe) dlopen handle are Send/Sync.
unsafe impl Send for CublasLib {}
unsafe impl Sync for CublasLib {}

/// Candidate shared-library names, most specific first.
fn candidates() -> Vec<String> {
    let mut v = Vec::new();
    if let Ok(path) = std::env::var("AUTORUST_CUBLAS_LIB")
        && !path.is_empty()
    {
        v.push(path);
    }
    for name in [
        "libcublas.so.13",
        "libcublas.so.12",
        "libcublas.so.11",
        "libcublas.so",
        "libcublas.12.dylib",
        "libcublas.dylib",
        "cublas64_13.dll",
        "cublas64_12.dll",
    ] {
        v.push(name.to_string());
    }
    v
}

fn load() -> Option<CublasLib> {
    for name in candidates() {
        // SAFETY: loading an arbitrary shared library is inherently unsafe; a
        // failed load is handled, and every symbol is resolved before use.
        let lib = match unsafe { Library::new(&name) } {
            Ok(lib) => lib,
            Err(_) => continue,
        };
        unsafe {
            macro_rules! sym {
                ($name:literal) => {
                    match lib.get::<*const c_void>($name.as_bytes()) {
                        Ok(s) => std::mem::transmute_copy::<*const c_void, _>(&*s),
                        Err(_) => continue,
                    }
                };
            }
            let create: CublasCreate = sym!("cublasCreate_v2");
            let destroy: CublasDestroy = sym!("cublasDestroy_v2");
            let set_stream: CublasSetStream = sym!("cublasSetStream_v2");
            let sgemm: CublasSgemm = sym!("cublasSgemm_v2");
            let sgemm_strided_batched: CublasSgemmStridedBatched =
                sym!("cublasSgemmStridedBatched");
            let gemm_ex: CublasGemmEx = sym!("cublasGemmEx");
            let gemm_strided_batched_ex: CublasGemmStridedBatchedEx =
                sym!("cublasGemmStridedBatchedEx");
            let set_math_mode: Option<CublasSetMathMode> = lib
                .get::<*const c_void>(b"cublasSetMathMode")
                .ok()
                .map(|s| std::mem::transmute_copy::<*const c_void, CublasSetMathMode>(&*s));
            return Some(CublasLib {
                _lib: lib,
                create,
                destroy,
                set_stream,
                set_math_mode,
                sgemm,
                sgemm_strided_batched,
                gemm_ex,
                gemm_strided_batched_ex,
            });
        }
    }
    None
}

fn lib() -> Option<&'static CublasLib> {
    static LIB: OnceLock<Option<CublasLib>> = OnceLock::new();
    LIB.get_or_init(load).as_ref()
}

/// Whether the cuBLAS library loaded (ignoring the user toggle).
pub fn library_present() -> bool {
    lib().is_some()
}

/// Whether cuBLAS is loaded and the fast path should be used.
///
/// `AUTORUST_CUBLAS=0` disables it; any other value keeps the default (on when
/// the library is present).
pub fn available() -> bool {
    if matches!(
        std::env::var("AUTORUST_GEMM")
            .or_else(|_| std::env::var("AUTORUST_CUBLAS"))
            .as_deref(),
        Ok("0")
    ) {
        return false;
    }
    lib().is_some()
}

/// Math mode: `AUTORUST_CUBLAS_TF32=1` allows TF32 on f32 GEMMs (faster,
/// slightly less precise). Default off to match PyTorch's default.
fn tf32_enabled() -> bool {
    matches!(
        std::env::var("AUTORUST_CUBLAS_TF32").as_deref(),
        Ok("1") | Ok("true") | Ok("on")
    )
}

/// An owned cuBLAS handle bound to one CUDA stream.
pub struct Handle {
    raw: *mut c_void,
    lib: &'static CublasLib,
}

impl Handle {
    /// Create a handle for `stream` (a `CUstream`).
    ///
    /// # Safety
    /// `stream` must be a valid CUDA stream on the current device.
    pub unsafe fn new(stream: *mut c_void) -> Option<Handle> {
        let lib = lib()?;
        let mut raw: *mut c_void = std::ptr::null_mut();
        let rc = unsafe { (lib.create)(&mut raw) };
        if rc != 0 || raw.is_null() {
            return None;
        }
        let rc = unsafe { (lib.set_stream)(raw, stream) };
        if rc != 0 {
            unsafe { (lib.destroy)(raw) };
            return None;
        }
        if let Some(set_math) = lib.set_math_mode {
            // CUBLAS_TF32_TENSOR_OP_MATH = 3, CUBLAS_DEFAULT_MATH = 0.
            let mode = if tf32_enabled() { 3 } else { 0 };
            let _ = unsafe { set_math(raw, mode) };
        }
        Some(Handle { raw, lib })
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        unsafe { (self.lib.destroy)(self.raw) };
    }
}

/// Element type of the matrices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Elem {
    /// IEEE binary32.
    F32,
    /// IEEE binary16.
    F16,
    /// bfloat16.
    Bf16,
}

impl Elem {
    fn cuda_type(self) -> i32 {
        match self {
            Elem::F32 => CUDA_R_32F,
            Elem::F16 => CUDA_R_16F,
            Elem::Bf16 => CUDA_R_16BF,
        }
    }
    fn itemsize(self) -> usize {
        match self {
            Elem::F32 => 4,
            Elem::F16 | Elem::Bf16 => 2,
        }
    }
}

/// Column-major GEMM descriptor (the natural cuBLAS layout).
///
/// Computes `C = alpha * op(A) * op(B)` with `A` of shape
/// `(m, k)` when [`OP_N`] or `(k, m)` when [`OP_T`] (transposed), likewise `B`
/// of `(k, n)` / `(n, k)`, and `C` of `(m, n)`.
#[derive(Clone, Copy, Debug)]
pub struct GemmDesc {
    /// Element type of all three matrices.
    pub elem: Elem,
    /// Whether `A` is stored transposed.
    pub trans_a: bool,
    /// Whether `B` is stored transposed.
    pub trans_b: bool,
    /// Rows of `C`.
    pub m: i64,
    /// Columns of `C`.
    pub n: i64,
    /// Contraction length.
    pub k: i64,
    /// Leading dimension of `A`.
    pub lda: i64,
    /// Leading dimension of `B`.
    pub ldb: i64,
    /// Leading dimension of `C`.
    pub ldc: i64,
    /// Scalar multiplier.
    pub alpha: f32,
}

/// Run one GEMM (or a strided batch of them) with raw device pointers.
///
/// `a`, `b`, `c` are the first element of each matrix *including any batch
/// element offset*. `batch == 1` uses the plain GEMM path.
///
/// # Safety
/// All pointers must be valid device memory on the handle's stream, and the
/// strides/shapes must describe in-bounds reads and writes.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm(
    handle: &Handle,
    desc: &GemmDesc,
    a: *const c_void,
    b: *const c_void,
    c: *mut c_void,
    batch: i64,
    stride_a: i64,
    stride_b: i64,
    stride_c: i64,
) -> Result<(), String> {
    let lib = handle.lib;
    let ta = if desc.trans_a { OP_T } else { OP_N };
    let tb = if desc.trans_b { OP_T } else { OP_N };
    let (m, n, k) = (
        i32::try_from(desc.m).map_err(|_| "cublas: m too large")?,
        i32::try_from(desc.n).map_err(|_| "cublas: n too large")?,
        i32::try_from(desc.k).map_err(|_| "cublas: k too large")?,
    );
    let (lda, ldb, ldc) = (
        i32::try_from(desc.lda).map_err(|_| "cublas: lda too large")?,
        i32::try_from(desc.ldb).map_err(|_| "cublas: ldb too large")?,
        i32::try_from(desc.ldc).map_err(|_| "cublas: ldc too large")?,
    );
    let beta = 0.0f32;
    let rc = unsafe {
        match desc.elem {
            Elem::F32 => {
                let alpha = desc.alpha;
                if batch <= 1 {
                    (lib.sgemm)(
                        handle.raw,
                        ta,
                        tb,
                        m,
                        n,
                        k,
                        &alpha,
                        a as *const f32,
                        lda,
                        b as *const f32,
                        ldb,
                        &beta,
                        c as *mut f32,
                        ldc,
                    )
                } else {
                    (lib.sgemm_strided_batched)(
                        handle.raw,
                        ta,
                        tb,
                        m,
                        n,
                        k,
                        &alpha,
                        a as *const f32,
                        lda,
                        stride_a,
                        b as *const f32,
                        ldb,
                        stride_b,
                        &beta,
                        c as *mut f32,
                        ldc,
                        stride_c,
                        i32::try_from(batch).map_err(|_| "cublas: batch too large")?,
                    )
                }
            }
            _ => {
                let alpha = desc.alpha;
                let compute = if tf32_enabled() && desc.elem == Elem::F32 {
                    COMPUTE_32F_FAST_TF32
                } else {
                    COMPUTE_32F
                };
                let ty = desc.elem.cuda_type();
                if batch <= 1 {
                    (lib.gemm_ex)(
                        handle.raw,
                        ta,
                        tb,
                        m,
                        n,
                        k,
                        &alpha as *const f32 as *const c_void,
                        a,
                        ty,
                        lda,
                        b,
                        ty,
                        ldb,
                        &beta as *const f32 as *const c_void,
                        c,
                        ty,
                        ldc,
                        compute,
                        GEMM_DEFAULT,
                    )
                } else {
                    (lib.gemm_strided_batched_ex)(
                        handle.raw,
                        ta,
                        tb,
                        m,
                        n,
                        k,
                        &alpha as *const f32 as *const c_void,
                        a,
                        ty,
                        lda,
                        stride_a * desc.elem.itemsize() as i64,
                        b,
                        ty,
                        ldb,
                        stride_b * desc.elem.itemsize() as i64,
                        &beta as *const f32 as *const c_void,
                        c,
                        ty,
                        ldc,
                        stride_c * desc.elem.itemsize() as i64,
                        i32::try_from(batch).map_err(|_| "cublas: batch too large")?,
                        compute,
                        GEMM_DEFAULT,
                    )
                }
            }
        }
    };
    if rc != 0 {
        return Err(format!("cublas gemm failed with status {rc}"));
    }
    Ok(())
}
