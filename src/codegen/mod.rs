//! `codegen/` — the kernel codegen pipeline (`tinygrad/codegen/`): dtype
//! decomposition, late rewrites, rangeify/mops, linearization, register
//! allocation, coalescing, control flow, and the optional BEAM search driver.
//!
//! The authoritative stage list is [`pipeline::full_rewrite_to_sink`]
//! (mirroring `codegen/__init__.py:294-393`). The renderer-specific pieces
//! (`extra_matcher`, `code_for_op`, tensor-core layouts) live in
//! [`crate::renderer`].
//!
//! Known gaps in 0.1.x: the late/transcendental rewrite sets
//! (`get_late_rewrite_patterns`, `get_transcendental_patterns`), parameter
//! renumbering (`pm_number_params`), the image path, and the BEAM worker pool.
//! These are documented in `CURRENT_STATE.md` §5 and §9.6.
pub mod pipeline;
pub mod opt_search;
pub mod simplify;
pub mod postrange;
pub mod gpudims;
pub mod coalesce;
pub mod linearize;
pub mod gater;
pub mod decomp;
pub mod decomp_dtype;
