//! `uop/` — the interned UOp graph and its rewrite machinery (M2).
//!
//! - [`ops`] — the frozen v0.14 `Ops` enum and group sets.
//! - [`graph`]/[`ctx`] — hash-consed nodes, [`UOpCtx`](ctx::UOpCtx), cached
//!   recursive properties (`shape`, dtype, vmin/vmax, address space).
//! - [`upat`] — the `UPat` DSL and `PatternMatcher` (`graph_rewrite`).
//! - [`sym`]/[`divandmod`] — symbolic simplification and floor-div/mod rules.
//! - [`alu`] — `exec_alu`/`python_alu` constant evaluation.
//! - [`flat`] — the JSON interchange used by the Python bridge.
//!
//! This module is public for backend authors and differential tooling, but its
//! API is **unstable** in 0.1.x.
pub mod alu;
pub mod arg;
pub mod ctx;
pub mod divandmod;
pub mod flat;
pub mod graph;
pub mod ops;
pub mod range;
pub mod sint;
pub mod sym;
pub mod upat;

pub use arg::*;
pub use flat::*;
pub use graph::*;
pub use ops::*;
pub mod weak;
