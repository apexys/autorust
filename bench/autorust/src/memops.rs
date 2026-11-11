//! Memory-bound op microbench: one pass over a 16M-element tensor per op, so
//! the result is bandwidth (GB/s), not FLOP/s.
//!
//! The byte counts are the minimum traffic of each op (reads + writes); a
//! fused implementation can only reduce them, so a low GB/s means the kernel
//! is not saturating memory.

use std::time::Instant;

use autorust::dtype;
use autorust::engine::jit::Jit;
use autorust::engine::session::Session;
use autorust::tensor::ix::Ix;
use autorust::tensor::nn::LayerNorm;
use autorust::tensor::{Ctx, Tensor};

/// One measured op.
pub struct Case {
    /// Op name (with the element count).
    pub name: String,
    /// Median wall time in milliseconds.
    pub ms: f64,
    /// Effective bandwidth in GB/s.
    pub gbs: f64,
}

const N: i64 = 1 << 24;
const SIDE: i64 = 4096;

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(f64::total_cmp);
    let n = xs.len();
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        0.5 * (xs[n / 2 - 1] + xs[n / 2])
    }
}

fn read_one(sess: &mut Session, y: &Tensor) {
    let rank = y.shape().as_slice().len();
    let ixs = vec![Ix::slice(None, Some(1), 1); rank];
    let _ = y
        .getitem(&ixs)
        .cast(dtype::FLOAT32)
        .to_list::<f32>(sess);
}

/// Run every op once (JIT-captured) and return the medians.
pub fn run(
    ctx: &Ctx,
    sess: &mut Session,
    warmup: usize,
    iters: usize,
    only: Option<&str>,
) -> Vec<Case> {
    let a = Tensor::rand(ctx, &[N], None, None).to_param();
    let b = Tensor::rand(ctx, &[N], None, None).to_param();
    let m2 = a.reshape([SIDE, SIDE]);
    let ln = LayerNorm::new(ctx, vec![SIDE], 1e-5, true);
    let nbytes = (N * 4) as f64;
    // (name, bytes moved, output)
    let ops: Vec<(&str, f64, Tensor)> = vec![
        ("add", 3.0 * nbytes, a.add(&b)),
        ("copy", 2.0 * nbytes, a.mul(1.0)),
        ("gelu", 2.0 * nbytes, autorust::tensor::nn::gelu(&a)),
        ("softmax", 2.0 * nbytes, autorust::tensor::nn::softmax(&m2, Some(1))),
        ("layernorm", 2.0 * nbytes, ln.forward(&m2)),
        ("sum", nbytes, a.sum(None, false, None)),
        ("amax", nbytes, a.max(None, false)),
        (
            "transpose",
            2.0 * nbytes,
            m2.transpose(0, 1).contiguous(),
        ),
    ];
    let mut out = Vec::new();
    for (name, bytes, y) in ops {
        if let Some(pat) = only
            && !name.contains(pat)
        {
            continue;
        }
        let mut jit = Jit::new();
        let step = |jit: &mut Jit, sess: &mut Session| {
            let y = jit.call(ctx, sess, |_c, _s| y.clone());
            read_one(sess, &y);
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
        out.push(Case {
            name: format!("{name} {N}"),
            ms,
            gbs: bytes / (ms * 1e6),
        });
    }
    out
}
