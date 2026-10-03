//! `engine/jit.py` (subset) — a session-driven TinyJit.
//!
//! tinygrad's `@TinyJit` runs the decorated function once to *capture* the
//! LINEAR schedules it produces (via the `capturing` global), lowers them once,
//! and then replays those schedules with fresh buffers instead of rebuilding the
//! graph. That removes the per-step `create_linear_with_vars` (rangeify /
//! kernelize) cost, which dominates the port's training step (CURRENT_STATE
//! §8.21).
//!
//! This port keeps the same shape but is opt-in and session-driven, matching the
//! user-facing [`Session::jit`](crate::engine::session::Session::jit) knob:
//!
//! ```ignore
//! let mut jit = Jit::new();
//! loop {
//!     let loss = jit.call(&mut ctx, &mut sess, |ctx, sess| {
//!         // build + realize one step; return the realized loss
//!         train_step(ctx, sess, /* … */)
//!     });
//!     // `loss` reads the buffer the (re)played schedule wrote
//! }
//! ```
//!
//! The first call captures and executes; later calls only re-run the captured
//! planned LINEARS. Because the RNG is a device graph (§8.23) and parameters/data
//! are persistent buffers, replay advances the sampler and optimizer correctly.
//!
//! As in tinygrad, not every function is JIT-able: inputs must be real buffers,
//! and the step must not read tensor data to host during capture. The port does
//! not (yet) reject such graphs; it captures what it sees.

use std::collections::HashMap;

use crate::engine::realize::{compile_linear_cached, exec_linear, CompiledLinear};
use crate::engine::session::Session;
use crate::runtime::buffer::{buffer_of, Buffer};
use crate::tensor::Tensor;
use crate::uop::ctx::UOpCtx;

/// `_TinyJit` (jit.py:220) — captures the compiled LINEARS of one `call` and
/// replays them. Owns the captured result tensor and keeps every captured buffer
/// alive (so the session's `prune` cannot reclaim a replay's intermediates).
pub struct Jit {
    linears: Vec<CompiledLinear>,
    ret: Option<Tensor>,
    /// Strong refs to every buffer the captured linears touch.
    held: Vec<Buffer>,
}

impl Jit {
    pub fn new() -> Self {
        Jit { linears: Vec::new(), ret: None, held: Vec::new() }
    }

    /// `TinyJit.reset` (jit.py:230): drop the capture so the next call captures.
    pub fn reset(&mut self) {
        self.linears.clear();
        self.ret = None;
        self.held.clear();
    }

    pub fn is_captured(&self) -> bool {
        self.ret.is_some()
    }

    /// The number of captured LINEARS (kernels/copies), for diagnostics.
    pub fn n_linears(&self) -> usize {
        self.linears.len()
    }

    /// Run `f` once to capture and execute one step; thereafter replay the
    /// captured schedules and return the captured result tensor.
    pub fn call<F>(&mut self, ctx: &mut UOpCtx, sess: &mut Session, f: F) -> Tensor
    where
        F: FnOnce(&mut UOpCtx, &mut Session) -> Tensor,
    {
        if !sess.jit_enabled() {
            // `if not JIT:` (jit.py:244) — run the step directly, no capture.
            return f(ctx, sess);
        }
        if self.ret.is_none() {
            // capture: run the step, collecting every planned LINEAR it realizes.
            sess.begin_capture();
            let ret = f(ctx, sess);
            // `if len(params:=get_parameters(ret)): Tensor.realize(*params)`
            // (jit.py:258): force the return to realize inside the capture (a
            // lazy eval graph would otherwise capture nothing). Idempotent.
            ret.realize(ctx, sess);
            let planned = sess.end_capture();
            assert!(!planned.is_empty(), "TGERR: JIT captured no linears (did the step realize anything?)");
            // `jit_lower`: lower + compile each linear once; replay executes these.
            self.linears = planned
                .into_iter()
                .map(|l| compile_linear_cached(ctx, l, &mut sess.cache))
                .collect();
            // hold all buffers the captured linears reference so `prune` keeps them.
            self.held = collect_buffers(ctx, sess, &self.linears);
            self.ret = Some(ret.clone());
            ret
        } else {
            // replay: execute the compiled schedules against the live buffers.
            for c in &self.linears {
                exec_linear(ctx, &mut sess.buffers, c, &HashMap::new());
            }
            self.ret.clone().expect("TGERR: JIT replay missing captured result")
        }
    }
}

impl Default for Jit {
    fn default() -> Self {
        Self::new()
    }
}

/// Resolve every BUFFER referenced by the compiled `linears` to its session
/// allocation, so the caller can hold them across future realizes (whose `prune`
/// would otherwise drop an intermediate that only the table references). Each
/// CALL's args (PARAM-globals plus arena buffers) are walked.
fn collect_buffers(ctx: &mut UOpCtx, sess: &mut Session, linears: &[CompiledLinear]) -> Vec<Buffer> {
    let mut out: Vec<Buffer> = Vec::new();
    let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for c in linears {
        for u in ctx.toposort(&c.line) {
            if ctx.op_of(&u) == crate::uop::Ops::Buffer && seen.insert(u.id()) {
                out.push(buffer_of(ctx, &mut sess.buffers, u));
            }
        }
    }
    out
}
