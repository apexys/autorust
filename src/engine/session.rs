//! `Session` — persistent realize state.
//!
//! tinygrad keeps realized storage in a global `buffers` side-table (a
//! `WeakKeyDictionary` keyed by the buffer uop) plus the `to_program_cache`, so a
//! tensor realized once is not recompiled or reallocated on the next read. The
//! Rust port had been rebuilding a fresh [`BufTable`] + [`ProgCache`] on every
//! `to_bytes`/`to_list`/`item`, which meant every training step recompiled and
//! reallocated all of its kernels.
//!
//! A `Session` bundles that state so callers can carry it across many realizes:
//!
//! ```ignore
//! let mut sess = Session::new();
//! let a = Tensor::from_slice(&mut ctx, &[1.0f32, 2.0], None, None);
//! let b = a.add(&mut ctx, &a);
//! let v = b.to_list_in::<f32>(&mut ctx, &mut sess);
//! // second realize: no recompile, no reallocation
//! let v2 = b.to_list_in::<f32>(&mut ctx, &mut sess);
//! ```
//!
//! Scope/correctness: `buffers` is keyed by uop id, so a session's `buffers` is
//! only valid for the `UOpCtx` whose ids it was populated from (reuse a session
//! with its own ctx). `cache` is keyed by a **content hash of the AST**, so it is
//! safe to share across arenas / graph rebuilds (that is what makes compiled
//! kernels survive the per-step `UOpCtx` in the current training loop).

use std::collections::HashMap;

use crate::engine::realize::{self, ProgCache};
use crate::runtime::buffer::{Buffer, BufTable};
use crate::uop::ctx::UOpCtx;
use crate::uop::graph::UopId;

/// Persistent realize state: buffers + compiled-program cache + variable values.
#[derive(Default)]
pub struct Session {
    /// The engine `buffers` dict (see module docs for the id-scoping caveat).
    pub buffers: BufTable,
    /// Compiled-program cache, keyed by AST content (safe across arenas).
    pub cache: ProgCache,
    /// Values for symbolic `Variable`s (empty on the concrete corpus).
    pub var_vals: HashMap<String, i64>,
    /// Realized root uops → their storage buffer. Mirrors tinygrad
    /// `Tensor.realized`: a second realize of the same tensor is a no-op. UOps
    /// are immutable, so a root's realized value cannot go stale (removing this
    /// guarantee is what tinygrad uses buffer invalidation for).
    realized: HashMap<UopId, Buffer>,
}

impl Session {
    pub fn new() -> Self {
        Session::default()
    }

    /// Schedule + execute `tensor_uop`, returning its realized buffer, reusing
    /// every buffer/compiled program already in the session. A tensor already
    /// realized in this session returns its buffer immediately.
    pub fn realize_tensor(&mut self, ctx: &mut UOpCtx, tensor_uop: UopId) -> Buffer {
        if let Some(b) = self.realized.get(&tensor_uop) {
            return b.clone();
        }
        let b = realize::realize_tensor(ctx, tensor_uop.clone(), &mut self.buffers, &mut self.cache);
        self.realized.insert(tensor_uop, b.clone());
        b
    }

    /// Number of tensors realized through this session.
    pub fn n_realized(&self) -> usize {
        self.realized.len()
    }

    /// `Tensor.to_bytes` against this session (memoized realize + copyout).
    pub fn tensor_bytes(&mut self, ctx: &mut UOpCtx, tensor_uop: UopId) -> Vec<u8> {
        realize::buffer_bytes(&self.realize_tensor(ctx, tensor_uop))
    }

    /// `Tensor.to_list` (f64 decode) against this session.
    pub fn tensor_values(&mut self, ctx: &mut UOpCtx, tensor_uop: UopId) -> Vec<f64> {
        realize::buffer_values(&self.realize_tensor(ctx, tensor_uop))
    }
}
