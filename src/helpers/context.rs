//! Scoped global flags, mirroring tinygrad's `helpers.ContextVar`
//! (`tinygrad/helpers.py`) — currently just `TRAINING`, which gates training-only
//! behavior such as `Tensor.dropout` and BatchNorm's running-stat updates.
//!
//! tinygrad uses a `ContextVar` so `with Context(TRAINING=1):` scopes the flag
//! to a block. The port keeps a thread-local cell plus explicit setters, matching
//! the existing `rand::set_jit` pattern (there is no `Context` object here); the
//! [`TrainingGuard`] RAII type gives the scoped `with Context(TRAINING=1)` feel.

use std::cell::Cell;

thread_local! {
    /// `TRAINING` (helpers.py:237): 1 inside a training step, 0 for inference.
    static TRAINING: Cell<bool> = const { Cell::new(false) };
}

/// `Tensor.training` / `TRAINING.value` (helpers.py:237).
pub fn training() -> bool {
    TRAINING.with(|c| c.get())
}

/// Set the global training flag directly (used by
/// [`Session::training`](crate::engine::session::Session::training)).
pub fn set_training(on: bool) {
    TRAINING.with(|c| c.set(on));
}

/// RAII guard for `with Context(TRAINING=...):` — restores the previous value on
/// drop, so nesting is safe.
pub struct TrainingGuard(bool);

impl TrainingGuard {
    /// Enter `TRAINING = on`, remembering the previous value.
    pub fn new(on: bool) -> Self {
        let prev = training();
        set_training(on);
        TrainingGuard(prev)
    }
}

impl Drop for TrainingGuard {
    fn drop(&mut self) {
        set_training(self.0);
    }
}

/// `with Context(TRAINING=on): f()` — set `TRAINING` for the duration of `f`.
pub fn with_training<T>(on: bool, f: impl FnOnce() -> T) -> T {
    let _g = TrainingGuard::new(on);
    f()
}
