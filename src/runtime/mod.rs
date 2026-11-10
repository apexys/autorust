//! `runtime/` — the device-facing half of the crate (`tinygrad/runtime/` +
//! `device.py`): buffer/opaque seams, the `Compiled` device composed from
//! public backend traits, and the built-in backends. A user backend is added
//! by implementing `OpaqueT`, `BackendAllocator`, `Comp`, `Prog` and
//! `UopRenderer` and calling `register_device` — nothing else.
pub mod autotune;
pub mod buffer;
#[cfg(feature = "backend_cl")]
pub mod cl_flash;
#[cfg(feature = "backend_cl")]
pub(crate) mod cl_sys;
#[cfg(feature = "backend_cl")]
pub mod clblast;
// The packed CPU GEMM serves every host-memory backend, not only `RUST`.
#[cfg(any(
    feature = "backend_rust",
    feature = "backend_clang",
    feature = "backend_rayon"
))]
pub mod cpu_gemm;
#[cfg(all(
    feature = "backend_cuda",
    any(target_os = "linux", target_os = "windows")
))]
pub mod cublas;
#[cfg(all(
    feature = "backend_cuda",
    any(target_os = "linux", target_os = "windows")
))]
pub mod cuda_flash;
#[cfg(all(
    feature = "backend_cuda",
    any(target_os = "linux", target_os = "windows")
))]
pub mod cuda_flash_bf16;
#[cfg(all(
    feature = "backend_cuda",
    any(target_os = "linux", target_os = "windows")
))]
pub mod cuda_flash_sage;
#[cfg(all(
    feature = "backend_cuda",
    any(target_os = "linux", target_os = "windows")
))]
pub mod cuda_flash_tc;
pub mod device;
pub mod interp;
#[cfg(all(feature = "backend_metal", target_os = "macos"))]
pub mod metal_flash;
#[cfg(feature = "backend_cl")]
pub mod ops_cl;
#[cfg(feature = "backend_clang")]
pub mod ops_clang;
#[cfg(all(
    feature = "backend_cuda",
    any(target_os = "linux", target_os = "windows")
))]
pub mod ops_cuda;
pub mod ops_interp;
#[cfg(all(feature = "backend_metal", target_os = "macos"))]
pub mod ops_metal;
#[cfg(feature = "backend_rayon")]
pub mod ops_rayon;
#[cfg(feature = "backend_rust")]
pub mod ops_rust;
#[cfg(feature = "backend_rayon")]
pub mod ray_kernel;
#[cfg(feature = "backend_rayon")]
pub mod ray_typed;
#[cfg(feature = "backend_rayon")]
pub mod ray_value;
#[cfg(all(
    feature = "backend_cuda",
    any(target_os = "linux", target_os = "windows")
))]
pub mod sage_flash;
