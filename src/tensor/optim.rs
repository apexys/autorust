//! `nn/optim.py` — optimizers. They own their parameter list (tinygrad
//! `optim.SGD([w, b], lr)`) and read each parameter's gradient off its graph node
//! (written by [`Tensor::backward`]).
//!
//! `step` is the tinygrad operation: realize the update, then `assign` it back
//! into the parameter's existing buffer (`Optimizer.step`, optim.py:35-56). That
//! keeps the parameters stable leaves, so a training loop does not chain its
//! graph across steps. `step_graph` is the pure-graph variant (returns updated
//! tensors, never mutates a buffer) kept for the numeric-parity tests that
//! rebuild the graph each epoch.

use crate::dtype::ConstVal;
use crate::engine::session::Session;
use crate::tensor::Tensor;
use crate::uop::ctx::UOpCtx;

/// Common surface for an optimizer step.
pub trait Optim {
    /// The parameters this optimizer updates, in order.
    fn params(&self) -> &[Tensor];

    /// Pure-graph update: build `w - lr*grad` for every parameter and return the
    /// updated tensors (missing gradients leave the parameter unchanged).
    fn step_graph(&self, ctx: &mut UOpCtx) -> Vec<Tensor>;

    /// `Optimizer.step` (optim.py:35-56): realize the updates and `assign` each
    /// back into its parameter's buffer, leaving the parameter `uop` unchanged.
    fn step(&self, ctx: &mut UOpCtx, sess: &mut Session) {
        let updates = self.step_graph(ctx);
        for (p, u) in self.params().iter().zip(updates.iter()) {
            p.assign(ctx, sess, u);
        }
    }

    /// `Optimizer.zero_grad`: clear each parameter's attached gradient so the
    /// next `backward` starts from an empty accumulator.
    fn zero_grad(&self, ctx: &UOpCtx) {
        for p in self.params() {
            ctx.clear_grad(&p.uop);
        }
    }
}

/// Stochastic gradient descent (tinygrad `optim.SGD`, momentum = 0):
/// `w -= lr * grad`. Missing gradients are treated as zero (tinygrad skips a
/// param whose `.grad is None`; here a missing grad leaves the param untouched).
pub struct SGD {
    params: Vec<Tensor>,
    pub lr: f32,
}

impl SGD {
    /// tinygrad `optim.SGD(params, lr=...)`.
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        SGD { params, lr }
    }
}

impl Optim for SGD {
    fn params(&self) -> &[Tensor] {
        &self.params
    }

    fn step_graph(&self, ctx: &mut UOpCtx) -> Vec<Tensor> {
        self.params
            .iter()
            .map(|p| match ctx.grad(&p.uop) {
                Some(g) => {
                    let lr_c = ctx.const_like(p.uop.clone(), ConstVal::Float(self.lr as f64));
                    let g_lr = ctx.mul(g, lr_c);
                    let w_new = ctx.sub(p.uop.clone(), g_lr);
                    Tensor::from_uop(ctx, w_new)
                }
                None => p.clone(),
            })
            .collect()
    }
}
