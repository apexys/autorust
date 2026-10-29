//! `schedule/` — Seam B: turn a tensor graph into a schedule of kernel SINKs.
//!
//! - [`rangeify`] — `get_kernel_graph`, movement-op lowering, bufferization.
//! - [`indexing`] — `run_rangeify`/`IndexingContext`, realize decisions.
//! - [`sched`] — `create_schedule`/`create_linear_with_vars`, optimizer hooks.
//! - [`memory`] — graph-level lifetime analysis and arena suballocation.
//!
//! Multi-device (`UNSHARD`/`MSELECT`/`MSTACK`) is out of scope; the scheduler
//! asserts a single device.
pub mod indexing;
pub mod rangeify;
pub mod sched;
pub mod memory;
