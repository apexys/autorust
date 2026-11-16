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

// ---- cuBLASLt (fused bias/ReLU epilogue) ----
type LtCreate = unsafe extern "C" fn(*mut *mut c_void) -> i32;
type LtDestroy = unsafe extern "C" fn(*mut c_void) -> i32;
type LtMatmulDescCreate = unsafe extern "C" fn(*mut *mut c_void, i32, i32) -> i32;
type LtMatmulDescDestroy = unsafe extern "C" fn(*mut c_void) -> i32;
type LtMatmulDescSetAttr = unsafe extern "C" fn(*mut c_void, i32, *const c_void, usize) -> i32;
type LtMatrixLayoutCreate = unsafe extern "C" fn(*mut *mut c_void, i32, u64, u64, i64) -> i32;
type LtMatrixLayoutDestroy = unsafe extern "C" fn(*mut c_void) -> i32;
type LtMatrixLayoutSetAttr = unsafe extern "C" fn(*mut c_void, i32, *const c_void, usize) -> i32;
#[allow(clippy::too_many_arguments)]
type LtMatmul = unsafe extern "C" fn(
    *mut c_void,
    *mut c_void,
    *const c_void,
    *const c_void,
    *mut c_void,
    *const c_void,
    *mut c_void,
    *const c_void,
    *const c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *const c_void,
    *mut c_void,
    usize,
    *mut c_void,
) -> i32;

/// `cublasLtMatmulDescAttributes_t` values used here.
const LT_TRANSA: i32 = 3;
const LT_TRANSB: i32 = 4;
const LT_EPILOGUE: i32 = 7;
const LT_BIAS_POINTER: i32 = 8;
/// `cublasLtEpilogue_t`: bias plus optional ReLU.
const LT_EPILOGUE_BIAS: u32 = 4;
const LT_EPILOGUE_BIAS_RELU: u32 = 6;
/// `cublasLtMatrixLayoutAttributes_t` values used here.
const LT_LAYOUT_BATCH_COUNT: i32 = 5;
const LT_LAYOUT_STRIDED_OFFSET: i32 = 6;

struct LtFns {
    create: LtCreate,
    destroy: LtDestroy,
    desc_create: LtMatmulDescCreate,
    desc_destroy: LtMatmulDescDestroy,
    desc_set_attr: LtMatmulDescSetAttr,
    layout_create: LtMatrixLayoutCreate,
    layout_destroy: LtMatrixLayoutDestroy,
    layout_set_attr: LtMatrixLayoutSetAttr,
    matmul: LtMatmul,
}

struct CublasLib {
    _lib: Library,
    _lt_lib: Option<Library>,
    lt: Option<LtFns>,
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
///
/// Discovery is shared with NVRTC ([`crate::runtime::libfind`]): the soname
/// encodes the toolkit major version, so trying only the build-time version
/// would break against a different installed toolkit.
fn candidates() -> Vec<String> {
    let mut out = crate::runtime::libfind::env_overrides(&["AUTORUST_CUBLAS_LIB"]);
    if !out.is_empty() {
        return out;
    }
    out.extend(crate::runtime::libfind::scanned_libraries("cublas"));
    out.push(crate::runtime::libfind::unversioned_name("cublas"));
    for major in (9..=20).rev() {
        out.push(crate::runtime::libfind::versioned_name("cublas", major));
    }
    out.dedup();
    out
}

/// Load `libcublasLt` (the epilogue-fusion API lives in its own library).
fn load_lt() -> Option<(Library, LtFns)> {
    let mut names = crate::runtime::libfind::env_overrides(&["AUTORUST_CUBLASLT_LIB"]);
    names.extend(crate::runtime::libfind::scanned_libraries("cublasLt"));
    names.push(crate::runtime::libfind::unversioned_name("cublasLt"));
    for major in (9..=20).rev() {
        names.push(crate::runtime::libfind::versioned_name("cublasLt", major));
    }
    names.dedup();
    for name in names {
        let lib = match unsafe { Library::new(&name) } {
            Ok(lib) => lib,
            Err(_) => continue,
        };
        unsafe {
            macro_rules! sym {
                ($name:literal, $ty:ty) => {
                    match lib.get::<*const c_void>($name.as_bytes()) {
                        Ok(s) => std::mem::transmute_copy::<*const c_void, $ty>(&*s),
                        Err(_) => continue,
                    }
                };
            }
            let fns = LtFns {
                create: sym!("cublasLtCreate", LtCreate),
                destroy: sym!("cublasLtDestroy", LtDestroy),
                desc_create: sym!("cublasLtMatmulDescCreate", LtMatmulDescCreate),
                desc_destroy: sym!("cublasLtMatmulDescDestroy", LtMatmulDescDestroy),
                desc_set_attr: sym!("cublasLtMatmulDescSetAttribute", LtMatmulDescSetAttr),
                layout_create: sym!("cublasLtMatrixLayoutCreate", LtMatrixLayoutCreate),
                layout_destroy: sym!("cublasLtMatrixLayoutDestroy", LtMatrixLayoutDestroy),
                layout_set_attr: sym!("cublasLtMatrixLayoutSetAttribute", LtMatrixLayoutSetAttr),
                matmul: sym!("cublasLtMatmul", LtMatmul),
            };
            return Some((lib, fns));
        }
    }
    None
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
            let loaded_lt = load_lt();
            let (lt_lib, lt) = match loaded_lt {
                Some((l, f)) => (Some(l), Some(f)),
                None => (None, None),
            };
            return Some(CublasLib {
                _lib: lib,
                _lt_lib: lt_lib,
                lt,
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

/// Whether the fused bias/ReLU epilogue is available and enabled.
///
/// `AUTORUST_CUBLASLT=0` forces the separate-epilogue path (A/B testing and an
/// escape hatch for a driver-specific issue).
pub fn lt_available() -> bool {
    if crate::helpers::env::get("AUTORUST_CUBLASLT").is_some_and(|v| v == "0") {
        return false;
    }
    lib().is_some_and(|l| l.lt.is_some())
}

fn lt_check(rc: i32) -> Result<(), String> {
    if rc == 0 {
        Ok(())
    } else {
        Err(format!("cublasLt status {rc}"))
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn lt_layout(
    lt: &LtFns,
    rows: i64,
    cols: i64,
    ld: i64,
    batch: i64,
    stride: i64,
) -> Result<*mut c_void, String> {
    let mut l: *mut c_void = std::ptr::null_mut();
    unsafe {
        lt_check((lt.layout_create)(
            &mut l,
            CUDA_R_32F,
            rows as u64,
            cols as u64,
            ld,
        ))?
    };
    if batch > 1 {
        let bc: i32 = i32::try_from(batch).map_err(|_| "cublasLt: batch too large")?;
        let so: i64 = stride;
        unsafe {
            lt_check((lt.layout_set_attr)(
                l,
                LT_LAYOUT_BATCH_COUNT,
                &bc as *const i32 as *const c_void,
                4,
            ))?;
            lt_check((lt.layout_set_attr)(
                l,
                LT_LAYOUT_STRIDED_OFFSET,
                &so as *const i64 as *const c_void,
                8,
            ))?;
        }
    }
    Ok(l)
}

/// Run `C = alpha*op(A)*op(B) + bias[col]` with an optional ReLU, fused into
/// the cuBLASLt epilogue.
///
/// The separate epilogue kernel (`autorust_epi_f32`) is skipped by the caller
/// when this succeeds. Only f32 and the plain bias / bias+ReLU shapes are
/// handled; the per-column multiplicative scale and the elementwise residual
/// are not expressible in a cuBLASLt epilogue and stay on the kernel.
///
/// # Safety
/// The pointers and layouts must describe in-bounds device memory on
/// `handle`'s stream, with the bias vector holding `desc.n` elements.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_bias_lt(
    handle: &Handle,
    desc: &GemmDesc,
    a: *const c_void,
    b: *const c_void,
    c: *mut c_void,
    bias: *const c_void,
    batch: i64,
    stride_a: i64,
    stride_b: i64,
    stride_c: i64,
    relu: bool,
) -> Result<(), String> {
    let lib = handle.lib;
    let Some(lt) = lib.lt.as_ref() else {
        return Err("cublasLt not loaded".into());
    };
    let Some(lt_raw) = handle.lt_raw else {
        return Err("cublasLt handle missing".into());
    };
    if desc.elem != Elem::F32 {
        return Err("cublasLt epilogue: f32 only".into());
    }
    let mut md: *mut c_void = std::ptr::null_mut();
    unsafe { lt_check((lt.desc_create)(&mut md, COMPUTE_32F, CUDA_R_32F))? };
    let res = (|| -> Result<(), String> {
        let ta: i32 = if desc.trans_a { OP_T } else { OP_N };
        let tb: i32 = if desc.trans_b { OP_T } else { OP_N };
        let ep: u32 = if relu {
            LT_EPILOGUE_BIAS_RELU
        } else {
            LT_EPILOGUE_BIAS
        };
        unsafe {
            lt_check((lt.desc_set_attr)(
                md,
                LT_TRANSA,
                &ta as *const i32 as *const c_void,
                4,
            ))?;
            lt_check((lt.desc_set_attr)(
                md,
                LT_TRANSB,
                &tb as *const i32 as *const c_void,
                4,
            ))?;
            lt_check((lt.desc_set_attr)(
                md,
                LT_EPILOGUE,
                &ep as *const u32 as *const c_void,
                4,
            ))?;
            lt_check((lt.desc_set_attr)(
                md,
                LT_BIAS_POINTER,
                &bias as *const *const c_void as *const c_void,
                std::mem::size_of::<*const c_void>(),
            ))?;
        }
        // Stored dimensions of each operand (column-major layouts).
        let (ar, ac) = if desc.trans_a {
            (desc.k, desc.m)
        } else {
            (desc.m, desc.k)
        };
        let (br, bc) = if desc.trans_b {
            (desc.n, desc.k)
        } else {
            (desc.k, desc.n)
        };
        let ad = unsafe { lt_layout(lt, ar, ac, desc.lda, batch, stride_a)? };
        let bd = unsafe { lt_layout(lt, br, bc, desc.ldb, batch, stride_b)? };
        let cd = unsafe { lt_layout(lt, desc.m, desc.n, desc.ldc, batch, stride_c)? };
        let alpha = desc.alpha;
        let beta = desc.beta;
        let rc = unsafe {
            (lt.matmul)(
                lt_raw,
                md,
                &alpha as *const f32 as *const c_void,
                a,
                ad,
                b,
                bd,
                &beta as *const f32 as *const c_void,
                c as *const c_void,
                cd,
                c,
                cd,
                std::ptr::null(),
                std::ptr::null_mut(),
                0,
                handle.stream,
            )
        };
        unsafe {
            (lt.layout_destroy)(ad);
            (lt.layout_destroy)(bd);
            (lt.layout_destroy)(cd);
        }
        lt_check(rc)
    })();
    unsafe { (lt.desc_destroy)(md) };
    res
}

/// Cached result of [`probe_current_context`]: `None` until a CUDA context has
/// been current on some thread, then the probe outcome.
static PROBE: OnceLock<bool> = OnceLock::new();

/// Whether the loaded cuBLAS can actually initialise on the CUDA context bound
/// to the calling thread.
///
/// A soname match is not enough: cuBLAS from a newer toolkit can reject an old
/// GPU (`CUBLAS_STATUS_ARCH_MISMATCH`) or an old driver. The CUDA device probe
/// calls this while the context is current, so a broken library disables the
/// host-GEMM fast path instead of panicking on the first scheduled GEMM.
///
/// Cached for the process (the handle is created once and destroyed again).
pub fn probe_current_context() -> bool {
    *PROBE.get_or_init(|| {
        let Some(lib) = lib() else {
            return false;
        };
        let mut raw: *mut c_void = std::ptr::null_mut();
        let rc = unsafe { (lib.create)(&mut raw) };
        if rc != 0 || raw.is_null() {
            eprintln!(
                "autorust: cuBLAS cannot create a handle on this CUDA driver/GPU; using generated GEMM kernels"
            );
            return false;
        }
        unsafe { (lib.destroy)(raw) };
        true
    })
}

/// Whether cuBLAS is loadable and (once probed) actually usable.
///
/// `AUTORUST_CUBLAS=0` disables it; any other value keeps the default (on when
/// the library is present). Before the device probe runs this is the load-only
/// check, so scheduling that happens before the CUDA context exists still sees
/// the library.
pub fn available() -> bool {
    match PROBE.get() {
        Some(&ok) => ok,
        None => lib().is_some(),
    }
}

/// Math mode: TF32 on f32 GEMMs unless `AUTORUST_CUBLAS_TF32=0` (faster,
/// slightly less precise). Default off to match PyTorch's default.
fn tf32_enabled() -> bool {
    crate::settings::global().allow_tf32
}

/// An owned cuBLAS handle bound to one CUDA stream.
pub struct Handle {
    raw: *mut c_void,
    /// The stream bound with `cublasSetStream`; `cublasLtMatmul` takes it
    /// explicitly.
    stream: *mut c_void,
    /// `cublasLtHandle_t`, created once when the Lt library is present.
    lt_raw: Option<*mut c_void>,
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
            // 3 = CUBLAS_TF32_TENSOR_OP_MATH, 2 = CUBLAS_PEDANTIC_MATH.
            // 0 (the old default) also allows TF32 on CUDA >= 11, so `false`
            // has to be pedantic to actually mean full-precision f32.
            let mode = if tf32_enabled() { 3 } else { 2 };
            let _ = unsafe { set_math(raw, mode) };
        }
        let lt_raw = lib.lt.as_ref().and_then(|lt| {
            let mut l: *mut c_void = std::ptr::null_mut();
            let rc = unsafe { (lt.create)(&mut l) };
            if rc == 0 && !l.is_null() {
                Some(l)
            } else {
                None
            }
        });
        Some(Handle {
            raw,
            stream,
            lt_raw,
            lib,
        })
    }
}

impl Handle {
    /// Switch the handle between pedantic f32 and TF32 tensor-core math.
    ///
    /// Used by the opt-in conv-only TF32 policy; the normal handle setup still
    /// follows [`tf32_enabled`].
    pub fn set_math_mode(&self, tf32: bool) {
        let Some(set_math) = self.lib.set_math_mode else {
            return;
        };
        let mode = if tf32 { 3 } else { 2 };
        let _ = unsafe { set_math(self.raw, mode) };
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        if let (Some(lt), Some(raw)) = (&self.lib.lt, self.lt_raw) {
            unsafe { (lt.destroy)(raw) };
        }
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
    /// `C = alpha*op(A)*op(B) + beta*C` (0 to overwrite, 1 to accumulate).
    pub beta: f32,
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
    let beta = desc.beta;
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
