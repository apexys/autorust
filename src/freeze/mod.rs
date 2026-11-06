//! Freeze a realized/compiled schedule into a portable plan.
//!
//! The plan is a backend-independent list of storage buffers, typed views, and
//! kernel/copy steps. `FrozenRunner` executes it with a kernel table (the JIT
//! path loads the `RUST` cdylibs; the AOT emitter will provide the same table
//! as statically linked Rust functions).

pub mod emit;
pub mod engine;
pub mod plan;
pub mod runner;

pub use emit::{emit_rust, emit_to};
pub use engine::{Engine, Frozen, ModuleForward};
pub use plan::{FrozenPlan, Step, StorageSpec, ViewSpec};
pub use runner::FrozenRunner;
