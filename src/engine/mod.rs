//! `engine/` — realize (schedule → kernels → launches), sessions, and JIT.
//!
//! - [`realize`] — `ProgramInfo::from_sink`, `to_program`, `run_linear`.
//! - [`session`] — [`Session`](session::Session): buffer table + content-keyed
//!   program cache + symbolic values; reuse it across training steps.
//! - [`jit`] — opt-in TinyJit-style capture/replay of planned LINEARs.
//! - [`disk_cache`] — the BEAM/PTX disk cache.
//!
//! A `Tensor` owns its realized buffer; the `Session` keeps views/bases alive
//! for the duration of a realize and prunes unreferenced bases at the start of
//! the next one.
pub mod disk_cache;
pub mod jit;
pub mod realize;
pub mod session;
