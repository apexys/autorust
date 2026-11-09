//! `runtime/` — the device-facing half of the crate (`tinygrad/runtime/` +
//! `device.py`): buffer/opaque seams, the `Compiled` device composed from
//! public backend traits, and the built-in backends. A user backend is added
//! by implementing `OpaqueT`, `BackendAllocator`, `Comp`, `Prog` and
//! `UopRenderer` and calling `register_device` — nothing else.
pub mod buffer;
#[cfg(feature = "backend_cl")]
pub mod cl_flash;
#[cfg(feature = "backend_cl")]
pub(crate) mod cl_sys;
#[cfg(feature = "backend_cl")]
pub mod clblast;
#[cfg(feature = "backend_rust")]
pub mod cpu_gemm;
#[cfg(feature = "backend_cuda")]
pub mod cublas;
#[cfg(feature = "backend_cuda")]
pub mod cuda_flash;
#[cfg(feature = "backend_cuda")]
pub mod cuda_flash_tc;
pub mod device;
pub mod interp;
#[cfg(feature = "backend_cl")]
pub mod ops_cl;
#[cfg(feature = "backend_clang")]
pub mod ops_clang;
#[cfg(feature = "backend_cuda")]
pub mod ops_cuda;
pub mod ops_interp;
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
#[cfg(feature = "backend_cuda")]
pub mod sage_flash;
