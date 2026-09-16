//! `nn/optim.py` — optimizers. They read each parameter's gradient out of the
//! `UOpCtx` side-table (written by [`Tensor::backward`]) and produce *new*
//! parameter tensors. This is a pure-graph optimizer: like the rest of the port
//! it never mutates a buffer in place, so `step` returns the updated weight
//! handles which the caller carries into the next iteration.

use crate::dtype::ConstVal;
use crate::tensor::Tensor;
use crate::uop::ctx::UOpCtx;

/// Common surface for an optimizer step. `step` consumes the gradients currently
/// sitting in `ctx.grads` for `params` and returns the updated tensors (in the
/// same order). `zero_grad` wipes the side-table so the next `backward` starts
/// from an empty accumulator (tinygrad `Optimizer.zero_grad`).
pub trait Optim {
    fn step(&self, ctx: &mut UOpCtx, params: &[Tensor]) -> Vec<Tensor>;
    fn zero_grad(&self, ctx: &mut UOpCtx) {
        ctx.clear_grads();
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
            .map(|p| match ctx.grad(p.uop) {
                Some(g) => {
                    let lr_c = ctx.const_like(p.uop, ConstVal::Float(self.lr as f64));
                    let g_lr = ctx.mul(g, lr_c);
                    let w_new = ctx.sub(p.uop, g_lr);
                    Tensor::from_uop(w_new)
                }
                None => *p,
            })
            .collect()
    }
}
