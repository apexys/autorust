//! `codegen/__init__.py` port: the `to_program` pipeline (M5) — IN PROGRESS.
//!
//! `full_rewrite_to_sink` lowers a kernel AST (SINK with KernelInfo) into the
//! program SINK the renderer/interpreter consumes. This file implements the
//! first block of stages; the remaining python stages (postrange opts,
//! expander, reduce-local accs, gpudims, devectorize, decomps, control flow)
//! are listed in PLAN.md §M5 and append to `stages()` as they land.
//!
//! Differential loop: `py/dump_codegen_stages.py` emits per-stage UOp key
//! multisets from python; `TG_M5_STAGES` truncates this pipeline so the test
//! compares stage-for-stage.

use crate::codegen::simplify::{pm_flatten_range, pm_reduce_unparented};
use crate::uop::ops::Ops;
use crate::uop::sym::{pm_mop_cleanup, symbolic};
use crate::uop::upat::{graph_rewrite, CtxVal, PatternMatcher};
use crate::uop::ctx::UOpCtx;
use crate::uop::UopId;

/// Truncation for stage-by-stage differentials (default: run everything).
fn stage_limit() -> usize {
    std::env::var("TG_M5_STAGES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(usize::MAX)
}

/// One named graph_rewrite stage (name matches python for the key diff).
struct Stage {
    name: &'static str,
    pm: PatternMatcher,
    bottom_up: bool,
}

/// The implemented python stages, in order.
fn stages() -> Vec<Stage> {
    vec![
        Stage {
            name: "early movement ops",
            pm: PatternMatcher::new(vec![]).add(&crate::schedule::rangeify::pm_mops()).add(&pm_mop_cleanup()),
            bottom_up: true,
        },
        Stage { name: "initial symbolic", pm: symbolic().add(&pm_flatten_range()), bottom_up: false },
        Stage { name: "postopt symbolic", pm: symbolic().add(&pm_flatten_range()).add(&pm_reduce_unparented()), bottom_up: false },
    ]
}

/// `full_rewrite_to_sink` (codegen/__init__.py:288) — partial: the opening
/// mops/symbolic block. Returns the rewritten SINK.
pub fn full_rewrite_to_sink(ctx: &mut UOpCtx, ast: UopId) -> UopId {
    let limit = stage_limit();
    let mut cur = ast;
    for (i, st) in stages().into_iter().enumerate() {
        if i + 1 > limit {
            break;
        }
        cur = graph_rewrite(ctx, cur, &st.pm, st.bottom_up, None, false, false, CtxVal::None)
            .unwrap_or_else(|e| panic!("TGERR: {}: {e}", st.name));
    }
    cur
}
