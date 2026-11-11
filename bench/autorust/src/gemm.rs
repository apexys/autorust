//! GEMM microbench: a fixed `matmul` shape sweep that isolates the tiled
//! matrix-multiply path from the rest of a model.
//!
//! Each case is measured through the same `Jit` capture/replay path as the
//! model workloads, with a one-element readback to force completion (the
//! result is never copied to the host). `tflops` is `2*M*N*K*B / time`.

use std::time::Instant;

use autorust::dtype;
use autorust::engine::jit::Jit;
use autorust::engine::session::Session;
use autorust::tensor::ix::Ix;
use autorust::tensor::{Ctx, Tensor};

/// One measured case (shape + median time + achieved throughput).
pub struct Case {
    /// Human-readable shape, e.g. `f32 2048x2048x2048` or `f32 TN 2048³`.
    pub name: String,
    /// Median wall time in milliseconds.
    pub ms: f64,
    /// Achieved TFLOP/s.
    pub tflops: f64,
}

struct Shape {
    name: &'static str,
    batch: i64,
    m: i64,
    n: i64,
    k: i64,
    transpose_a: bool,
    transpose_b: bool,
}

const SHAPES: &[Shape] = &[
    Shape { name: "1024³", batch: 1, m: 1024, n: 1024, k: 1024, transpose_a: false, transpose_b: false },
    Shape { name: "2048³", batch: 1, m: 2048, n: 2048, k: 2048, transpose_a: false, transpose_b: false },
    Shape { name: "4096³", batch: 1, m: 4096, n: 4096, k: 4096, transpose_a: false, transpose_b: false },
    Shape { name: "1024x1024x4096", batch: 1, m: 1024, n: 1024, k: 4096, transpose_a: false, transpose_b: false },
    Shape { name: "4096x1024x1024", batch: 1, m: 4096, n: 1024, k: 1024, transpose_a: false, transpose_b: false },
    Shape { name: "TN 2048³", batch: 1, m: 2048, n: 2048, k: 2048, transpose_a: true, transpose_b: false },
    Shape { name: "NT 2048³", batch: 1, m: 2048, n: 2048, k: 2048, transpose_a: false, transpose_b: true },
    Shape { name: "batched 8x512³", batch: 8, m: 512, n: 512, k: 512, transpose_a: false, transpose_b: false },
];

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(f64::total_cmp);
    let n = xs.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        0.5 * (xs[n / 2 - 1] + xs[n / 2])
    }
}

/// Run the sweep at one dtype (f32 when `bf16` is false).
///
/// Returns one [`Case`] per shape. The cold time of the first case (capture +
/// codegen) is not included in the medians.
pub fn run(
    ctx: &Ctx,
    sess: &mut Session,
    bf16: bool,
    warmup: usize,
    iters: usize,
    only: Option<&str>,
) -> Vec<Case> {
    let mut out = Vec::new();
    for s in SHAPES {
        let label = format!("{} {}", if bf16 { "bf16" } else { "f32" }, s.name);
        if let Some(pat) = only
            && !label.contains(pat)
        {
            continue;
        }
        let dt = if bf16 { Some(dtype::BFLOAT16) } else { None };
        let a = Tensor::rand(ctx, &[s.batch, s.m, s.k], dt, None).to_param();
        let b_shape = if s.transpose_b {
            [s.batch, s.n, s.k]
        } else {
            [s.batch, s.k, s.n]
        };
        let b = Tensor::rand(ctx, &b_shape, dt, None).to_param();
        let lhs = if s.transpose_a {
            a.transpose(-2, -1)
        } else {
            a.clone()
        };
        let rhs = if s.transpose_b {
            b.transpose(-2, -1)
        } else {
            b.clone()
        };
        let mut jit = Jit::new();
        let step = |jit: &mut Jit, sess: &mut Session| {
            let y = jit.call(ctx, sess, |_c, _s| lhs.dot(&rhs));
            // Force completion without copying the (large) result back.
            let rank = y.shape().as_slice().len();
            let ixs = vec![Ix::slice(None, Some(1), 1); rank];
            let _ = y
                .getitem(&ixs)
                .cast(dtype::FLOAT32)
                .to_list::<f32>(sess);
        };
        // First call compiles the kernels; it is excluded from the medians.
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
        let flops = 2.0 * (s.batch * s.m * s.n * s.k) as f64;
        out.push(Case {
            name: label,
            ms,
            tflops: flops / (ms * 1e9),
        });
    }
    out
}
