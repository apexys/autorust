//! Optimizer-only microbench: one AdamW step over ~100M parameters, with the
//! gradients set once, so the measurement is the pure elementwise update
//! (the store-free moment buffers, the decay, and the parameter write).
//!
//! Traffic per parameter: read grad, read m, read v, write m, write v, read p,
//! write p = 28 bytes.

use std::time::Instant;

use autorust::engine::jit::Jit;
use autorust::engine::session::Session;
use autorust::tensor::ix::Ix;
use autorust::tensor::optim::{AdamW, Optim};
use autorust::tensor::{Ctx, Tensor};

/// One measured step.
pub struct Case {
    /// Description of the parameter set.
    pub name: String,
    /// Median wall time in milliseconds.
    pub ms: f64,
    /// Effective bandwidth in GB/s (28 bytes per parameter).
    pub gbs: f64,
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(f64::total_cmp);
    let n = xs.len();
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        0.5 * (xs[n / 2 - 1] + xs[n / 2])
    }
}

/// Run one AdamW step `warmup + iters` times over `total` f32 parameters split
/// into `tensors` buffers (a real model has many).
pub fn run(
    ctx: &Ctx,
    sess: &mut Session,
    total: i64,
    tensors: i64,
    warmup: usize,
    iters: usize,
) -> Case {
    let per = total / tensors;
    let params: Vec<Tensor> = (0..tensors)
        .map(|_| Tensor::zeros(ctx, [per]).to_param())
        .collect();
    // Attach a constant gradient once; `schedule_step` reads it from the graph.
    let grads: Vec<Tensor> = (0..tensors).map(|_| Tensor::ones(ctx, [per])).collect();
    let param_uops: Vec<_> = params.iter().map(|p| p.uop().clone()).collect();
    let grad_uops: Vec<_> = grads.iter().map(|g| g.uop().clone()).collect();
    {
        let c = ctx.borrow_mut();
        for (p, g) in param_uops.iter().zip(grad_uops.iter()) {
            c.set_grad(p, g.clone());
        }
    }
    let opt = AdamW::new(params.clone(), 1e-3);
    let roots: Vec<Tensor> = params.clone();
    let mut jit = Jit::new();
    let step = |jit: &mut Jit, sess: &mut Session| {
        let y = jit.call(ctx, sess, |_c, s| {
            opt.schedule_step(s, &[]);
            roots[0].clone()
        });
        let _ = y
            .getitem(&[Ix::slice(None, Some(1), 1)])
            .to_list::<f32>(sess);
    };
    step(&mut jit, sess);
    for _ in 0..warmup {
        step(&mut jit, sess);
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        step(&mut jit, sess);
        samples.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let ms = median(samples);
    Case {
        name: format!("adamw {total} params, {tensors} tensors"),
        ms,
        gbs: 28.0 * total as f64 / (ms * 1e6),
    }
}
