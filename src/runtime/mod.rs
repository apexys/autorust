//! `runtime/` — the device-facing half of the crate (`tinygrad/runtime/` +
//! `device.py`): buffer/opaque seams, the `Compiled` device composed from
//! public backend traits, and the built-in backends. A user backend is added
//! by implementing `OpaqueT`, `BackendAllocator`, `Comp`, `Prog` and
//! `UopRenderer` and calling `register_device` — nothing else.
pub mod buffer;
pub mod device;
pub mod interp;
#[cfg(feature = "backend_cl")]
pub mod ops_cl;
#[cfg(feature = "backend_cuda")]
pub mod ops_cuda;
pub mod ops_interp;
#[cfg(feature = "backend_rayon")]
pub mod ops_rayon;
#[cfg(feature = "backend_rayon")]
pub mod ray_kernel;
#[cfg(feature = "backend_rayon")]
pub mod ray_typed;
#[cfg(feature = "backend_rayon")]
pub mod ray_value;
