//! Optional dynamically loaded CLBlast — the OpenCL analogue of `cublas.rs`.
//!
//! CLBlast provides tuned OpenCL GEMM kernels. Like the cuBLAS path it is
//! loaded at run time (`libloading`), so a missing library simply disables the
//! fast path. The scheduler's [`HostGemm`] descriptor is device-agnostic; the
//! OpenCL backend's host dispatcher feeds it here.
//!
//! Unlike cuBLAS, CLBlast has an explicit `layout` argument, so the tensors are
//! passed row-major and the column-major transpose dance is unnecessary.
//!
//! [`HostGemm`]: crate::schedule::host_gemm::HostGemm

use std::ffi::c_void;
use std::sync::OnceLock;

use libloading::Library;

/// `CLBlastLayoutRowMajor` (the C enum starts at 101).
pub const LAYOUT_ROW_MAJOR: i32 = 101;
/// `CLBlastTransposeNo` (the C enum starts at 111).
pub const TRANSPOSE_NO: i32 = 111;
/// `CLBlastTransposeYes`.
pub const TRANSPOSE_YES: i32 = 112;

type Sgemm = unsafe extern "C" fn(
    i32, // layout
    i32, // a_transpose
    i32, // b_transpose
    usize,
    usize,
    usize, // m, n, k
    f32,   // alpha
    *const c_void,
    usize,
    usize, // a, a_offset, a_ld
    *const c_void,
    usize,
    usize, // b, b_offset, b_ld
    f32,   // beta
    *mut c_void,
    usize,
    usize,            // c, c_offset, c_ld
    *mut *mut c_void, // queue
    *mut *mut c_void, // event
) -> i32;

type SgemmStridedBatched = unsafe extern "C" fn(
    i32,
    i32,
    i32,
    usize,
    usize,
    usize,
    f32,
    *const c_void,
    usize,
    usize,
    usize, // a, a_offset, a_ld, a_stride
    *const c_void,
    usize,
    usize,
    usize, // b, b_offset, b_ld, b_stride
    f32,
    *mut c_void,
    usize,
    usize,
    usize, // c, c_offset, c_ld, c_stride
    usize, // batch_count
    *mut *mut c_void,
    *mut *mut c_void,
) -> i32;

struct ClblastLib {
    _lib: Library,
    sgemm: Sgemm,
    sgemm_strided_batched: SgemmStridedBatched,
}

// Function pointers and the `dlopen` handle are thread-safe.
unsafe impl Send for ClblastLib {}
unsafe impl Sync for ClblastLib {}

fn candidates() -> Vec<String> {
    let mut v = Vec::new();
    if let Ok(path) = std::env::var("AUTORUST_CLBLAST_LIB")
        && !path.is_empty()
    {
        v.push(path);
    }
    for name in [
        "libclblast.so.1",
        "libclblast.so",
        "libclblast.dylib",
        "clblast.dll",
    ] {
        v.push(name.to_string());
    }
    v
}

fn load() -> Option<ClblastLib> {
    for name in candidates() {
        // SAFETY: loading a shared library; a failed load is handled and every
        // symbol is resolved before use.
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
            let sgemm: Sgemm = sym!("CLBlastSgemm");
            let sgemm_strided_batched: SgemmStridedBatched = sym!("CLBlastSgemmStridedBatched");
            return Some(ClblastLib {
                _lib: lib,
                sgemm,
                sgemm_strided_batched,
            });
        }
    }
    None
}

fn lib() -> Option<&'static ClblastLib> {
    static LIB: OnceLock<Option<ClblastLib>> = OnceLock::new();
    LIB.get_or_init(load).as_ref()
}

/// Whether the CLBlast library loaded (ignoring the user toggle).
pub fn library_present() -> bool {
    lib().is_some()
}

/// Whether CLBlast is loaded and the fast path should be used.
///
/// `AUTORUST_CLBLAST=0` disables it; any other value keeps the default.
pub fn available() -> bool {
    lib().is_some()
}

/// Run one f32 GEMM (or a strided batch) on `queue`.
///
/// Shapes are row-major: `C[m,n] = alpha * op(A) * op(B)`, `A` is `(m,k)` when
/// `trans_a == TRANSPOSE_NO` (leading dimension `lda >= k`) and a row-major
/// `(k,m)` otherwise, likewise for `B`. Offsets and strides are in **elements**.
///
/// # Safety
/// The `cl_mem` handles must be valid on `queue`, and the offsets/strides must
/// describe in-bounds regions.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm(
    queue: *mut c_void,
    trans_a: i32,
    trans_b: i32,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const c_void,
    a_off: usize,
    a_ld: usize,
    b: *const c_void,
    b_off: usize,
    b_ld: usize,
    c: *mut c_void,
    c_off: usize,
    c_ld: usize,
    batch: usize,
    a_stride: usize,
    b_stride: usize,
    c_stride: usize,
) -> Result<(), String> {
    let lib = lib().ok_or_else(|| "clblast: library not loaded".to_string())?;
    let beta = 0.0f32;
    // CLBlast may replace the queue pointer if it has to create its own; keep a
    // local copy so ours is not clobbered.
    let mut q = queue;
    let rc = unsafe {
        if batch <= 1 {
            (lib.sgemm)(
                LAYOUT_ROW_MAJOR,
                trans_a,
                trans_b,
                m,
                n,
                k,
                alpha,
                a,
                a_off,
                a_ld,
                b,
                b_off,
                b_ld,
                beta,
                c,
                c_off,
                c_ld,
                &mut q,
                std::ptr::null_mut(),
            )
        } else {
            (lib.sgemm_strided_batched)(
                LAYOUT_ROW_MAJOR,
                trans_a,
                trans_b,
                m,
                n,
                k,
                alpha,
                a,
                a_off,
                a_ld,
                a_stride,
                b,
                b_off,
                b_ld,
                b_stride,
                beta,
                c,
                c_off,
                c_ld,
                c_stride,
                batch,
                &mut q,
                std::ptr::null_mut(),
            )
        }
    };
    if rc != 0 {
        return Err(format!("clblast gemm failed with status {rc}"));
    }
    Ok(())
}
