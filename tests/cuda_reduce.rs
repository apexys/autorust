//! Fused CUDA full reductions (`Tensor::sum_with` / `max_with`) vs the graph.
//!
//! The host kernels are forward-only and limited to a contiguous f32 scalar
//! reduction; every other case must fall back to the graph. These cases cover
//! the kernel, the fallback, and the direct launch cost.

#![cfg(all(
    feature = "backend_cuda",
    any(target_os = "linux", target_os = "windows")
))]

use autorust::engine::session::Session;
use autorust::runtime::buffer::Buffer;
use autorust::tensor::{Ctx, Tensor};

fn cuda_available() -> bool {
    autorust::runtime::ops_cuda::available()
}

fn values(seed: usize, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i + seed) % 31) as f32) * 0.07 - 0.9)
        .collect()
}

fn check(n: usize) {
    let ctx = Ctx::new();
    let data = values(5, n);
    let x = Tensor::from_slice(&ctx, &data, None, Some("CUDA")).to_param();
    let mut s1 = Session::new();
    let gsum = x.sum(None, false, None).to_list::<f32>(&mut s1);
    let gmax = x.max(None, false).to_list::<f32>(&mut s1);
    let mut s2 = Session::new();
    let fsum = x
        .sum_with(None, false, None, &mut s2)
        .to_list::<f32>(&mut s2);
    let fmax = x.max_with(None, false, &mut s2).to_list::<f32>(&mut s2);
    let rel = |a: f32, b: f32| (a - b).abs() / a.abs().max(1.0);
    assert!(rel(gsum[0], fsum[0]) < 2e-4, "sum {gsum:?} vs {fsum:?}");
    assert!((gmax[0] - fmax[0]).abs() < 1e-3, "max {gmax:?} vs {fmax:?}");
}

#[test]
fn fused_reductions_match_graph() {
    if !cuda_available() {
        return;
    }
    check(1 << 20);
    check((1 << 20) + 12345);
}

/// A non-scalar reduction is outside the kernel's contract and uses the graph.
#[test]
fn fused_reductions_fall_back_for_axis() {
    if !cuda_available() {
        return;
    }
    let ctx = Ctx::new();
    let data = values(2, 1024 * 1024);
    let x = Tensor::from_slice(&ctx, &data, None, Some("CUDA"))
        .reshape([1024, 1024])
        .to_param();
    let mut s1 = Session::new();
    let g = x.sum(Some(&[1]), false, None).to_list::<f32>(&mut s1);
    let mut s2 = Session::new();
    let f = x
        .sum_with(Some(&[1]), false, None, &mut s2)
        .to_list::<f32>(&mut s2);
    assert_eq!(g, f);
}

/// Diagnostic: time the two direct host launches, without JIT/readback.
#[test]
#[ignore]
fn host_reduce_kernel_timing() {
    if !cuda_available() {
        return;
    }
    let n = 1usize << 24;
    let ctx = Ctx::new();
    let data = vec![0.01f32; n];
    let x = Tensor::from_slice(&ctx, &data, None, Some("CUDA")).to_param();
    let mut sess = Session::new();
    x.realize(&mut sess);
    let inb = x.realized_buffer().unwrap();
    let temp = Buffer::new_base("CUDA", 256, autorust::dtype::FLOAT32);
    temp.ensure_allocated(&autorust::runtime::device::device("CUDA"));
    let out = Buffer::new_base("CUDA", 1, autorust::dtype::FLOAT32);
    out.ensure_allocated(&autorust::runtime::device::device("CUDA"));
    let c1 =
        autorust::runtime::cuda_reduce::sum_partials(vec![inb.clone(), temp.clone()], n as i64);
    let c2 = autorust::runtime::cuda_reduce::sum_finish(vec![temp.clone(), out.clone()], 256);
    let a1 = autorust::runtime::cuda_reduce::amax_partials(vec![inb, temp.clone()], n as i64);
    let a2 = autorust::runtime::cuda_reduce::amax_finish(vec![temp, out], 256);
    for name in ["sum", "amax"] {
        let (c1, c2) = if name == "sum" {
            (&c1, &c2)
        } else {
            (&a1, &a2)
        };
        c1.exec();
        c2.exec();
        let iters = 20;
        let t = std::time::Instant::now();
        for _ in 0..iters {
            c1.exec();
            c2.exec();
        }
        let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
        eprintln!(
            "host {name}: {ms:.3} ms ({} GB/s)",
            (n * 4) as f64 / (ms * 1e6)
        );
    }
}
