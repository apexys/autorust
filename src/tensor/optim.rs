//! `nn/optim.py` — optimizers. They read each parameter's gradient off its
//! graph node (written by [`Tensor::backward`]) and produce *new* parameter
//! tensors. This is a pure-graph optimizer: like the rest of the port it never
//! mutates a buffer in place, so `step` returns the updated weight handles which
//! the caller carries into the next iteration.

use crate::dtype::ConstVal;
use crate::engine::realize::ProgCache;
use crate::runtime::buffer::BufTable;
use crate::tensor::Tensor;
use crate::uop::ctx::UOpCtx;

/// Common surface for an optimizer step. `step` reads the gradients currently
/// attached to `params` and returns the updated tensors (in the same order).
/// `step_inplace` instead writes those updates back into the parameters'
/// existing buffers (`Tensor.assign`, tinygrad `Optimizer.step`/optim.py:35),
/// which keeps parameters a stable leaf across steps. `zero_grad` clears each
/// parameter's attached gradient so the next `backward` starts from an empty
/// accumulator (tinygrad `Optimizer.zero_grad`).
pub trait Optim {
    fn step(&self, ctx: &mut UOpCtx, params: &[Tensor]) -> Vec<Tensor>;

    /// `Optimizer.step` (optim.py:35-56): realize the updates and `assign` each
    /// back into its parameter's buffer, leaving the parameter `uop` unchanged.
    fn step_inplace(&self, ctx: &mut UOpCtx, bufs: &mut BufTable, cache: &mut ProgCache, params: &[Tensor]) {
        let updates = self.step(ctx, params);
        for (p, u) in params.iter().zip(updates.iter()) {
            p.assign(ctx, bufs, cache, u);
        }
    }

    fn zero_grad(&self, ctx: &UOpCtx, params: &[Tensor]) {
        for p in params {
            ctx.clear_grad(&p.uop);
        }
    }
}

/// Stochastic gradient descent (tinygrad `optim.SGD`, momentum = 0):
/// `w -= lr * grad`. Missing gradients are treated as zero (tinygrad skips a
/// param whose `.grad is None`; here a missing grad leaves the param untouched).
pub struct SGD {
    pub lr: f32,
}

impl SGD {
    pub fn new(lr: f32) -> Self {
        SGD { lr }
    }
}

impl Optim for SGD {
    fn step(&self, ctx: &mut UOpCtx, params: &[Tensor]) -> Vec<Tensor> {
        params
            .iter()
            .map(|p| match ctx.grad(&p.uop) {
                Some(g) => {
                    let lr_c = ctx.const_like(p.uop.clone(), ConstVal::Float(self.lr as f64));
                    let g_lr = ctx.mul(g, lr_c);
                    let w_new = ctx.sub(p.uop.clone(), g_lr);
                    { let __u = w_new; Tensor::from_uop(ctx, __u) }
                }
                None => p.clone(),
            })
            .collect()
    }
}
