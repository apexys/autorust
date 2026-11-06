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

use crate::engine::realize::{CompiledLinear, compile_linear_cached, exec_linear};
use crate::engine::session::Session;
use crate::runtime::buffer::{Buffer, buffer_of};
use crate::tensor::{Ctx, Tensor};
use crate::uop::ctx::UOpCtx;

/// `_TinyJit` (jit.py:220) — captures the compiled LINEARS of one `call` and
/// replays them. Owns the captured result tensor and keeps every captured buffer
/// alive (so the session's `prune` cannot reclaim a replay's intermediates).
/// One replayed step: a compiled linear schedule or a hand-written launch.
pub enum JitStep {
    /// A compiled generated linear.
    Linear(CompiledLinear),
    /// A hand-written kernel launch (fused attention).
    Host(Box<crate::engine::session::HostCall>),
}

pub struct Jit {
    steps: Vec<JitStep>,
    rets: Vec<Tensor>,
    /// Strong refs to every buffer the captured linears touch.
    held: Vec<Buffer>,
}

impl Jit {
    /// Create an empty (uncaptured) JIT.
    pub fn new() -> Self {
        Jit {
            steps: Vec::new(),
            rets: Vec::new(),
            held: Vec::new(),
        }
    }

    /// `TinyJit.reset` (jit.py:230): drop the capture so the next call captures.
    pub fn reset(&mut self) {
        self.steps.clear();
        self.rets.clear();
        self.held.clear();
    }

    /// Whether a capture has completed and can be replayed.
    pub fn is_captured(&self) -> bool {
        !self.rets.is_empty()
    }

    /// The number of captured LINEARS (kernels/copies), for diagnostics.
    pub fn n_linears(&self) -> usize {
        self.steps
            .iter()
            .filter(|s| matches!(s, JitStep::Linear(_)))
            .count()
    }

    /// The captured compiled LINEARS (freeze/plan lowering).
    pub fn steps(&self) -> &[JitStep] {
        &self.steps
    }

    /// The captured output tensors, in the order returned by `call_many`.
    pub fn outputs(&self) -> &[Tensor] {
        &self.rets
    }

    /// Run `f` once to capture and execute one step; thereafter replay the
    /// captured schedules and return the captured result tensor.
    pub fn call<F>(&mut self, ctx: &Ctx, sess: &mut Session, f: F) -> Tensor
    where
        F: FnOnce(&Ctx, &mut Session) -> Tensor,
    {
        if !sess.jit_enabled() {
            // `if not JIT:` (jit.py:244) — run the step directly, no capture.
            return f(ctx, sess);
        }
        if self.rets.is_empty() {
            // Single-root capture keeps the original `realize_ctx` path (the
            // CL device-RNG/JIT tests depend on its exact capture semantics).
            sess.begin_capture();
            let ret = f(ctx, sess);
            let mut c = ctx.borrow_mut();
            ret.realize_ctx(&mut c, sess);
            let planned = sess.end_capture();
            assert!(
                !planned.is_empty(),
                "TGERR: JIT captured no linears (did the step realize anything?)"
            );
            self.steps = planned
                .into_iter()
                .map(|l| match l {
                    crate::engine::session::CaptureEntry::Linear(u) => {
                        JitStep::Linear(compile_linear_cached(&mut c, u, &mut sess.cache))
                    }
                    crate::engine::session::CaptureEntry::Host(h) => JitStep::Host(h),
                })
                .collect();
            self.held = collect_buffers(&mut c, sess, &self.steps);
            self.rets = vec![ret.clone()];
            ret
        } else {
            let mut c = ctx.borrow_mut();
            for st in &self.steps {
                match st {
                    JitStep::Linear(cl) => {
                        exec_linear(&mut c, &mut sess.buffers, cl, &HashMap::new())
                    }
                    JitStep::Host(h) => h.exec(),
                }
            }
            self.rets
                .first()
                .cloned()
                .expect("TGERR: JIT replay missing captured result")
        }
    }

    /// Multi-root variant of [`Jit::call`]: all returned tensors are realized
    /// in **one** corealized schedule and replayed together.
    pub fn call_many<F>(&mut self, ctx: &Ctx, sess: &mut Session, f: F) -> Vec<Tensor>
    where
        F: FnOnce(&Ctx, &mut Session) -> Vec<Tensor>,
    {
        if !sess.jit_enabled() {
            // `if not JIT:` (jit.py:244) — run the step directly, no capture.
            return f(ctx, sess);
        }
        if self.rets.is_empty() {
            // capture: run the step, collecting every planned LINEAR it realizes.
            sess.begin_capture();
            let rets = f(ctx, sess);
            let mut c = ctx.borrow_mut();
            // `if len(params:=get_parameters(ret)): Tensor.realize(*params)`
            // (jit.py:258): force every root to realize inside the capture.
            Tensor::realize_many(&mut c, sess, rets.iter());
            let planned = sess.end_capture();
            assert!(
                !planned.is_empty(),
                "TGERR: JIT captured no linears (did the step realize anything?)"
            );
            // `jit_lower`: lower + compile each linear once; replay executes these.
            self.steps = planned
                .into_iter()
                .map(|l| match l {
                    crate::engine::session::CaptureEntry::Linear(u) => {
                        JitStep::Linear(compile_linear_cached(&mut c, u, &mut sess.cache))
                    }
                    crate::engine::session::CaptureEntry::Host(h) => JitStep::Host(h),
                })
                .collect();
            // hold all buffers the captured steps reference so `prune` keeps them.
            self.held = collect_buffers(&mut c, sess, &self.steps);
            self.rets = rets.clone();
            rets
        } else {
            // replay: execute the compiled schedules against the live buffers.
            let mut c = ctx.borrow_mut();
            for st in &self.steps {
                match st {
                    JitStep::Linear(cl) => {
                        exec_linear(&mut c, &mut sess.buffers, cl, &HashMap::new())
                    }
                    JitStep::Host(h) => h.exec(),
                }
            }
            self.rets.clone()
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
fn collect_buffers(ctx: &mut UOpCtx, sess: &mut Session, steps: &[JitStep]) -> Vec<Buffer> {
    let mut out: Vec<Buffer> = Vec::new();
    let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
    let mut held_host: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for st in steps {
        let c = match st {
            JitStep::Linear(c) => c,
            JitStep::Host(h) => {
                // Dedup by base pointer in a separate set: `seen` holds uop ids,
                // and a truncated pointer must not collide with one of those.
                for b in &h.bufs {
                    if held_host.insert(std::rc::Rc::as_ptr(&b.base) as usize) {
                        out.push(b.clone());
                    }
                }
                continue;
            }
        };
        for u in ctx.toposort(&c.line) {
            if ctx.op_of(&u) == crate::uop::Ops::Buffer && seen.insert(u.id()) {
                out.push(buffer_of(ctx, &mut sess.buffers, u));
            }
        }
    }
    out
}
