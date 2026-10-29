//! `Session` — persistent realize state.
//!
//! tinygrad keeps realized storage in a global `buffers` side-table (a
//! `WeakKeyDictionary` keyed by the buffer uop) plus the `to_program_cache`, so a
//! tensor realized once is not recompiled or reallocated on the next read. The
//! Rust port had been rebuilding a fresh [`BufTable`] + [`ProgCache`] on every
//! `to_bytes`/`to_list`/`item`, which meant every training step recompiled and
//! reallocated all of its kernels.
//!
//! A `Session` is that realize state, carried as the second argument to every
//! read/optimizer call (tinygrad reaches its globals implicitly; we pass one
//! handle instead of the old `(BufTable, ProgCache)` pair):
//!
//! ```ignore
//! let mut sess = Session::new();
//! let a = Tensor::from_slice(&mut ctx, &[1.0f32, 2.0], None, None);
//! let b = a.add_ctx(ctx, &a);
//! let v = b.to_list::<f32>(&mut ctx, &mut sess);
//! // second realize: no recompile, no reallocation
//! let v2 = b.to_list::<f32>(&mut ctx, &mut sess);
//! ```
//!
//! Scope/correctness: `buffers` is keyed by uop id, so a session's `buffers` is
//! only valid for the `UOpCtx` whose ids it was populated from (reuse a session
//! with its own ctx, or reset `buffers` when starting a new arena). `cache` is
//! keyed by a **content hash of the AST**, so it is safe to share across arenas /
//! graph rebuilds (that is what makes compiled kernels survive a fresh `UOpCtx`).
//!
//! There is deliberately no realized-tensor memo here: that lives on the
//! [`Tensor`](crate::tensor::Tensor) itself (its shared buffer cell), matching
//! tinygrad's per-tensor `Tensor.realized`. Keeping a session-wide map keyed by
//! uop would pin every step's graph in a training loop (each step builds fresh
//! derived uops), which is exactly the unbounded growth the in-place loop fixes.

use std::collections::HashMap;

use crate::engine::realize::{self, ProgCache};
use crate::runtime::buffer::{BufTable, Buffer};
use crate::uop::ctx::UOpCtx;
use crate::uop::graph::UopId;

/// Realize state: buffers + compiled-program cache + variable values.
#[derive(Default)]
pub struct Session {
    /// The engine `buffers` dict (see module docs for the id-scoping caveat).
    pub buffers: BufTable,
    /// Compiled-program cache, keyed by AST content (safe across arenas).
    pub cache: ProgCache,
    /// Values for symbolic `Variable`s (empty on the concrete corpus).
    pub var_vals: HashMap<String, i64>,
    /// When `Some`, each realize appends its planned LINEAR here (JIT capture,
    /// `engine/jit.py:capturing`).
    pub capturing: Option<Vec<UopId>>,
    /// `JIT`: whether [`Jit`](crate::engine::jit::Jit) capture/replay is active.
    pub jit: bool,
    /// `TRAINING` (helpers.py:237): training vs eval mode. Set through
    /// [`Session::training`], which also mirrors the value into the global flag
    /// that `Tensor.dropout` reads at graph-build time.
    pub training: bool,
}

impl Session {
    /// Create a session with the default (JIT off, eval mode) settings.
    pub fn new() -> Self {
        Session::default()
    }

    /// `sess.jit(true)` / `sess.jit(false)`: enable/disable JIT capture+replay.
    pub fn jit(&mut self, on: bool) {
        self.jit = on;
        crate::tensor::rand::set_jit(on);
    }

    /// Alias for `jit(true)`.
    pub fn enable_jit(&mut self) {
        self.jit(true);
    }

    /// Whether JIT capture+replay is enabled.
    pub fn jit_enabled(&self) -> bool {
        self.jit
    }

    /// `sess.training(true)` / `sess.training(false)`: set the global `TRAINING`
    /// flag that gates training-only ops (`Tensor.dropout`). Mirrors
    /// [`Session::jit`].
    pub fn training(&mut self, on: bool) {
        self.training = on;
        crate::helpers::context::set_training(on);
    }

    /// Whether training mode is enabled.
    pub fn training_enabled(&self) -> bool {
        self.training
    }

    /// `capturing.append(jit)` / `capturing.clear()` (jit.py:255) — start
    /// collecting the planned LINEARS of every realize in a step.
    pub fn begin_capture(&mut self) {
        self.capturing = Some(Vec::new());
    }

    /// Stop collecting and return the captured planned linears (in order).
    pub fn end_capture(&mut self) -> Vec<UopId> {
        self.capturing.take().unwrap_or_default()
    }

    /// Schedule + execute `tensor_uop`, returning its realized buffer, reusing
    /// every buffer/compiled program already in the session. A tensor that has
    /// already realized caches its buffer on itself, so repeat reads hit that
    /// (see [`Tensor::to_list`](crate::tensor::Tensor::to_list)).
    pub fn realize_tensor(&mut self, ctx: &mut UOpCtx, tensor_uop: UopId) -> (Buffer, UopId) {
        let mut cap = self.capturing.take();
        let r = realize::realize_tensor(
            ctx,
            tensor_uop,
            &mut self.buffers,
            &mut self.cache,
            cap.as_mut(),
        );
        self.capturing = cap;
        r
    }

    /// `Tensor.realize(*tensors)`: co-realize every root in one schedule and
    /// return their realized buffers (optimizer co-realization, `schedule_step`).
    pub fn realize_many(
        &mut self,
        ctx: &mut UOpCtx,
        tensor_uops: &[UopId],
    ) -> (Vec<Buffer>, Vec<UopId>) {
        let mut cap = self.capturing.take();
        let r = realize::realize_many(
            ctx,
            tensor_uops,
            &mut self.buffers,
            &mut self.cache,
            cap.as_mut(),
        );
        self.capturing = cap;
        r
    }

    /// `Tensor.to_bytes` against this session (realize + copyout).
    pub fn tensor_bytes(&mut self, ctx: &mut UOpCtx, tensor_uop: UopId) -> Vec<u8> {
        realize::buffer_bytes(&self.realize_tensor(ctx, tensor_uop).0)
    }

    /// `Tensor.to_list` (f64 decode) against this session.
    pub fn tensor_values(&mut self, ctx: &mut UOpCtx, tensor_uop: UopId) -> Vec<f64> {
        realize::buffer_values(&self.realize_tensor(ctx, tensor_uop).0)
    }
}
