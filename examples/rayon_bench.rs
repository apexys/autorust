//! `rayon_bench` — a small CPU-backend benchmark: `RAYON` (multithreaded) vs
//! `INTERP` (scalar reference). Not a tinygrad example; a port convenience for
//! measuring the `ops_rayon` fast path (CURRENT_STATE.md §9.8).
//!
//! ```text
//! cargo run --release --example rayon_bench             # default N=256, iters=5
//! N=512 ITERS=10 cargo run --release --example rayon_bench
//! RAYON_SERIAL=1 cargo run --release --example rayon_bench   # single-threaded owned IR
//! BENCH_SKIP_INTERP=1 cargo run --release --example rayon_bench   # skip the slow oracle
//! ```
//!
//! Times a square matmul and an elementwise chain end-to-end (schedule+exec),
//! per device, and prints the speedup. `RAYON_NUM_THREADS` controls the pool.

use autorust::tensor::Tensor;
use autorust::uop::ctx::UOpCtx;
use std::time::Instant;

fn time_device(dev: &str, n: usize, iters: usize) -> (f64, f64) {
    let mut ctx = UOpCtx::new();
    let a_data: Vec<f32> = (0..n * n).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
    let w_data: Vec<f32> = (0..n * n).map(|i| ((i % 89) as f32 - 44.0) * 0.01).collect();
    let x_data: Vec<f32> = (0..n * n).map(|i| (i % 13) as f32 * 0.1).collect();

    let a = Tensor::from_slice(&mut ctx, &a_data, None, Some(dev)).reshape(&mut ctx, [n, n]);
    let w = Tensor::from_slice(&mut ctx, &w_data, None, Some(dev)).reshape(&mut ctx, [n, n]);
    let x = Tensor::from_slice(&mut ctx, &x_data, None, Some(dev)).reshape(&mut ctx, [n, n]);

    // warm compile
    let _ = a.dot(&mut ctx, &w).realize_fresh(&mut ctx);
    let _ = x.add(&mut ctx, 1.0).sqrt(&mut ctx).mul(&mut ctx, 2.0).realize_fresh(&mut ctx);

    let t = Instant::now();
    for _ in 0..iters {
        let _ = a.dot(&mut ctx, &w).realize_fresh(&mut ctx);
    }
    let mm = t.elapsed().as_secs_f64() / iters as f64;

    let t = Instant::now();
    for _ in 0..iters {
        let _ = x.add(&mut ctx, 1.0).sqrt(&mut ctx).mul(&mut ctx, 2.0).realize_fresh(&mut ctx);
    }
    let ew = t.elapsed().as_secs_f64() / iters as f64;
    (mm, ew)
}

fn main() {
    let n: usize = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(256);
    let iters: usize = std::env::var("ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(5);

    let threads = autorust::runtime::ops_rayon::num_threads();
    println!("matmul {n}x{n}, elementwise {n}x{n}, iters={iters}, rayon threads={threads}");

    let t0 = Instant::now();
    let (r_mm, r_ew) = time_device("RAYON", n, iters);
    let rayon_total = t0.elapsed().as_secs_f64();
    if std::env::var("BENCH_SKIP_INTERP").is_ok() {
        println!("                 matmul      elementwise");
        println!("RAYON   {:>10.4} ms {:>10.4} ms", r_mm * 1e3, r_ew * 1e3);
        println!("(skip INTERP; RAYON total incl. warmup/compile: {rayon_total:.2}s)");
        return;
    }
    let t0 = Instant::now();
    let (i_mm, i_ew) = time_device("INTERP", n, iters);
    let interp_total = t0.elapsed().as_secs_f64();

    println!("                 matmul      elementwise");
    println!("RAYON   {:>10.4} ms {:>10.4} ms", r_mm * 1e3, r_ew * 1e3);
    println!("INTERP  {:>10.4} ms {:>10.4} ms", i_mm * 1e3, i_ew * 1e3);
    println!("speedup {:>10.2}x  {:>10.2}x", i_mm / r_mm, i_ew / r_ew);
    println!("(total incl. warmup/compile: RAYON {rayon_total:.2}s, INTERP {interp_total:.2}s)");
}
