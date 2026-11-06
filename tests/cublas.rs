//! Library GEMM (`cuBLAS`) vs the generated CUDA kernels.
//!
//! The scheduler describes a pure `MUL -> REDUCE(ADD)` kernel as a `HostGemm`
//! and the CUDA backend runs it with cuBLAS. These tests force the library path
//! (`AUTORUST_CUBLAS=all`) and compare against the generated kernels on the same
//! tensors, covering the layouts ViT/DINOv3 produce: prime M, transposed weight
//! views, a fused bias, and strided batches.

#![cfg(feature = "backend_cuda")]

use autorust::dtype;
use autorust::tensor::nn::Linear;
use autorust::tensor::{Ctx, Tensor};

fn have_cuda_and_cublas() -> bool {
    if !autorust::runtime::ops_cuda::available() || !autorust::runtime::cublas::available() {
        return false;
    }
    unsafe { std::env::set_var("DEFAULT_DEVICE", "CUDA") };
    true
}

/// The library toggle is a process-global env var; serialize the cases.
static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn set_lib(on: bool) {
    unsafe { std::env::set_var("AUTORUST_CUBLAS", if on { "all" } else { "0" }) };
}

fn values(seed: usize, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i + seed) % 23) as f32) * 0.01 - 0.11)
        .collect()
}

fn run<F>(f: F) -> Vec<f32>
where
    F: Fn(&Ctx) -> Vec<Tensor>,
{
    let ctx = Ctx::new();
    let outs = f(&ctx);
    let mut out = Vec::new();
    for t in outs {
        out.extend(t.to_list_fresh::<f32>());
    }
    out
}

fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn cublas_matches_generated_gemm() {
    let _guard = LOCK.lock().unwrap();
    if !have_cuda_and_cublas() {
        return;
    }
    for (m, n, k) in [(197usize, 768, 768), (64, 64, 64), (33, 17, 65)] {
        let ad = values(0, m * k);
        let bd = values(7, k * n);
        let run_case = |lib: bool| {
            set_lib(lib);
            run(|ctx| {
                let a =
                    Tensor::from_slice(ctx, &ad, None, Some("CUDA")).reshape([m as i64, k as i64]);
                let b =
                    Tensor::from_slice(ctx, &bd, None, Some("CUDA")).reshape([k as i64, n as i64]);
                vec![a.dot(&b)]
            })
        };
        let want = run_case(false);
        let got = run_case(true);
        let d = maxdiff(&want, &got);
        assert!(d < 1e-3, "{m}x{n}x{k}: cuBLAS diff {d:e}");
    }
}

#[test]
fn cublas_matches_transposed_weight_with_bias() {
    let _guard = LOCK.lock().unwrap();
    if !have_cuda_and_cublas() {
        return;
    }
    let (m, k, n) = (197usize, 768usize, 384usize);
    let ad = values(1, m * k);
    let wd = values(3, n * k);
    let bd = values(5, n);
    let run_case = |lib: bool| {
        set_lib(lib);
        run(|ctx| {
            let a = Tensor::from_slice(ctx, &ad, None, Some("CUDA")).reshape([m as i64, k as i64]);
            let lin = Linear {
                weight: Tensor::from_slice(ctx, &wd, None, Some("CUDA"))
                    .reshape([n as i64, k as i64]),
                bias: Some(Tensor::from_slice(ctx, &bd, None, Some("CUDA"))),
            };
            vec![lin.forward(&a)]
        })
    };
    let d = maxdiff(&run_case(false), &run_case(true));
    assert!(d < 1e-3, "transposed weight: cuBLAS diff {d:e}");
}

#[test]
fn cublas_matches_strided_batched_attention() {
    let _guard = LOCK.lock().unwrap();
    if !have_cuda_and_cublas() {
        return;
    }
    let (b, h, m, k, n) = (1usize, 3usize, 37usize, 16usize, 37usize);
    let qd = values(2, b * h * m * k);
    let kd = values(9, b * h * n * k);
    let run_case = |lib: bool| {
        set_lib(lib);
        run(|ctx| {
            let q = Tensor::from_slice(ctx, &qd, None, Some("CUDA"))
                .reshape([b as i64, m as i64, h as i64, k as i64]);
            let kk = Tensor::from_slice(ctx, &kd, None, Some("CUDA"))
                .reshape([b as i64, n as i64, h as i64, k as i64]);
            // (b, h, m, k) and (b, h, n, k): the same layout the transformer
            // attention block builds (`reshape` then `transpose(1, 2)`).
            let q = q.transpose(1, 2);
            let kt = kk.transpose(1, 2).transpose(2, 3);
            vec![q.dot(&kt)]
        })
    };
    let d = maxdiff(&run_case(false), &run_case(true));
    assert!(d < 1e-3, "batched attention: cuBLAS diff {d:e}");
}

#[test]
fn cublas_matches_bf16() {
    let _guard = LOCK.lock().unwrap();
    if !have_cuda_and_cublas() {
        return;
    }
    let (m, n, k) = (197usize, 769usize, 768usize);
    let ad = values(4, m * k);
    let bd = values(6, k * n);
    let run_case = |lib: bool| {
        set_lib(lib);
        run(|ctx| {
            let a = Tensor::from_slice(ctx, &ad, Some(dtype::BFLOAT16), Some("CUDA"))
                .reshape([m as i64, k as i64]);
            let b = Tensor::from_slice(ctx, &bd, Some(dtype::BFLOAT16), Some("CUDA"))
                .reshape([k as i64, n as i64]);
            vec![a.dot(&b).cast(dtype::FLOAT32)]
        })
    };
    let d = maxdiff(&run_case(false), &run_case(true));
    assert!(d < 5e-3, "bf16: cuBLAS diff {d:e}");
}
