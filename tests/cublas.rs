//! Library GEMM (`cuBLAS`) vs the generated CUDA kernels.
//!
//! The scheduler describes a pure `MUL -> REDUCE(ADD)` kernel as a `HostGemm`
//! and the CUDA backend runs it with cuBLAS. These tests force the library path
//! (`AUTORUST_CUBLAS=all`) and compare against the generated kernels on the same
//! tensors, covering the layouts ViT/DINOv3 produce: prime M, transposed weight
//! views, a fused bias, and strided batches.

#![cfg(all(
    feature = "backend_cuda",
    any(target_os = "linux", target_os = "windows")
))]

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

fn values(seed: usize, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i + seed) % 23) as f32) * 0.01 - 0.11)
        .collect()
}

fn run<F>(policy: autorust::settings::LibraryGemm, f: F) -> Vec<f32>
where
    F: Fn(&Ctx) -> Vec<Tensor>,
{
    let ctx = Ctx::new();
    // Policy is per-context; the process defaults are read once from env.
    ctx.borrow_mut().settings.library_gemm = autorust::settings::LibraryGemmPolicy::uniform(policy);
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
            run(
                if lib {
                    autorust::settings::LibraryGemm::Always
                } else {
                    autorust::settings::LibraryGemm::Never
                },
                |ctx| {
                    let a = Tensor::from_slice(ctx, &ad, None, Some("CUDA"))
                        .reshape([m as i64, k as i64]);
                    let b = Tensor::from_slice(ctx, &bd, None, Some("CUDA"))
                        .reshape([k as i64, n as i64]);
                    vec![a.dot(&b)]
                },
            )
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
        run(
            if lib {
                autorust::settings::LibraryGemm::Always
            } else {
                autorust::settings::LibraryGemm::Never
            },
            |ctx| {
                let a =
                    Tensor::from_slice(ctx, &ad, None, Some("CUDA")).reshape([m as i64, k as i64]);
                let lin = Linear {
                    weight: Tensor::from_slice(ctx, &wd, None, Some("CUDA"))
                        .reshape([n as i64, k as i64]),
                    bias: Some(Tensor::from_slice(ctx, &bd, None, Some("CUDA"))),
                };
                vec![lin.forward(&a)]
            },
        )
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
        run(
            if lib {
                autorust::settings::LibraryGemm::Always
            } else {
                autorust::settings::LibraryGemm::Never
            },
            |ctx| {
                let q = Tensor::from_slice(ctx, &qd, None, Some("CUDA"))
                    .reshape([b as i64, m as i64, h as i64, k as i64]);
                let kk = Tensor::from_slice(ctx, &kd, None, Some("CUDA"))
                    .reshape([b as i64, n as i64, h as i64, k as i64]);
                // (b, h, m, k) and (b, h, n, k): the same layout the transformer
                // attention block builds (`reshape` then `transpose(1, 2)`).
                let q = q.transpose(1, 2);
                let kt = kk.transpose(1, 2).transpose(2, 3);
                vec![q.dot(&kt)]
            },
        )
    };
    let d = maxdiff(&run_case(false), &run_case(true));
    assert!(d < 1e-3, "batched attention: cuBLAS diff {d:e}");
}

#[test]
fn cublas_handles_two_axis_batch_layout() {
    let _guard = LOCK.lock().unwrap();
    if !have_cuda_and_cublas() {
        return;
    }
    use std::sync::atomic::Ordering;
    // `b > 1` makes `(batch, head)` two non-collapsible axes: the head stride is
    // `k` while the batch stride is `m*h*k`, so `head_stride * h != batch_stride`.
    // This is the ViT training-batch layout the generated kernel used to keep.
    let (b, h, m, k, n) = (2usize, 3usize, 37usize, 16usize, 37usize);
    let qd = values(2, b * h * m * k);
    let kd = values(9, b * h * n * k);
    let run_case = |lib: bool| {
        autorust::runtime::ops_cuda::CUBLAS_GEMMS.store(0, Ordering::Relaxed);
        let out = run(
            if lib {
                autorust::settings::LibraryGemm::Always
            } else {
                autorust::settings::LibraryGemm::Never
            },
            |ctx| {
                let q = Tensor::from_slice(ctx, &qd, None, Some("CUDA"))
                    .reshape([b as i64, m as i64, h as i64, k as i64])
                    .transpose(1, 2);
                let kk = Tensor::from_slice(ctx, &kd, None, Some("CUDA"))
                    .reshape([b as i64, n as i64, h as i64, k as i64])
                    .transpose(1, 2)
                    .transpose(2, 3);
                vec![q.dot(&kk)]
            },
        );
        (
            out,
            autorust::runtime::ops_cuda::CUBLAS_GEMMS.load(Ordering::Relaxed),
        )
    };
    let (want, calls_never) = run_case(false);
    let (got, calls_always) = run_case(true);
    assert_eq!(calls_never, 0, "Never policy must stay generated");
    assert!(
        calls_always > 0,
        "two-axis (batch, head) layout must dispatch to cuBLAS"
    );
    let d = maxdiff(&want, &got);
    assert!(d < 1e-3, "two-axis batch: cuBLAS diff {d:e}");
}

#[test]
fn cublas_collapses_two_reduce_axes() {
    let _guard = LOCK.lock().unwrap();
    if !have_cuda_and_cublas() {
        return;
    }
    use std::sync::atomic::Ordering;
    // `dW = sum_{batch, tokens} x[b,t,:] (x) dy[b,t,:]`: the scheduler keeps two
    // reduction axes, and both operands are contiguous over the pair, so it is
    // one `K = batch*tokens` contraction rather than two nested reductions.
    let (b, t, k, n) = (2usize, 37usize, 64usize, 48usize);
    let xd = values(1, b * t * k);
    let dd = values(7, b * t * n);
    let run_case = |lib: bool| {
        autorust::runtime::ops_cuda::CUBLAS_GEMMS.store(0, Ordering::Relaxed);
        let out = run(
            if lib {
                autorust::settings::LibraryGemm::Always
            } else {
                autorust::settings::LibraryGemm::Never
            },
            |ctx| {
                let x = Tensor::from_slice(ctx, &xd, None, Some("CUDA"))
                    .reshape([b as i64, t as i64, k as i64]);
                let dy = Tensor::from_slice(ctx, &dd, None, Some("CUDA"))
                    .reshape([b as i64, t as i64, n as i64]);
                let prod = x.unsqueeze(3).mul(dy.unsqueeze(2));
                vec![prod.sum(Some(&[0, 1]), false, None)]
            },
        );
        (
            out,
            autorust::runtime::ops_cuda::CUBLAS_GEMMS.load(Ordering::Relaxed),
        )
    };
    let (want, calls_never) = run_case(false);
    let (got, calls_always) = run_case(true);
    assert_eq!(calls_never, 0, "Never policy must stay generated");
    assert!(
        calls_always > 0,
        "a contiguous two-axis reduction must collapse to one GEMM"
    );
    let d = maxdiff(&want, &got);
    assert!(d < 1e-3, "collapsed reduction: cuBLAS diff {d:e}");
}

#[test]
fn cublas_handles_column_major_output() {
    let _guard = LOCK.lock().unwrap();
    if !have_cuda_and_cublas() {
        return;
    }
    use std::sync::atomic::Ordering;
    // `assign` into a transposed view stores the product through a column-major
    // view of the destination: the scheduler's C index then has stride 1 on the
    // *m* axis, which a row-major descriptor cannot describe. cuBLAS computes it
    // natively, so `extract` swaps the operand roles (`C^T = B^T A^T`).
    let (m, n, k) = (37usize, 48usize, 64usize);
    let ad = values(1, m * k);
    let bd = values(7, k * n);
    let run_case = |lib: bool| {
        autorust::runtime::ops_cuda::CUBLAS_GEMMS.store(0, Ordering::Relaxed);
        let ctx = Ctx::new();
        ctx.borrow_mut().settings.library_gemm =
            autorust::settings::LibraryGemmPolicy::uniform(if lib {
                autorust::settings::LibraryGemm::Always
            } else {
                autorust::settings::LibraryGemm::Never
            });
        let mut sess = autorust::engine::session::Session::new();
        let a = Tensor::from_slice(&ctx, &ad, None, Some("CUDA")).reshape([m as i64, k as i64]);
        let b = Tensor::from_slice(&ctx, &bd, None, Some("CUDA")).reshape([k as i64, n as i64]);
        let c = Tensor::zeros_param(&ctx, [n as i64, m as i64], None, Some("CUDA"));
        let ct = c.transpose(0, 1);
        ct.assign(&mut sess, &a.dot(&b));
        let out = ct.to_list::<f32>(&mut sess);
        (
            out,
            autorust::runtime::ops_cuda::CUBLAS_GEMMS.load(Ordering::Relaxed),
        )
    };
    let (want, calls_never) = run_case(false);
    let (got, calls_always) = run_case(true);
    assert_eq!(calls_never, 0, "Never policy must stay generated");
    assert!(
        calls_always > 0,
        "a column-major output must dispatch to cuBLAS"
    );
    assert!(
        got.iter().any(|v| *v != 0.0),
        "the transposed store must land (got all zeros)"
    );
    let d = maxdiff(&want, &got);
    assert!(d < 1e-4, "column-major output: cuBLAS diff {d:e}");
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
        run(
            if lib {
                autorust::settings::LibraryGemm::Always
            } else {
                autorust::settings::LibraryGemm::Never
            },
            |ctx| {
                let a = Tensor::from_slice(ctx, &ad, Some(dtype::BFLOAT16), Some("CUDA"))
                    .reshape([m as i64, k as i64]);
                let b = Tensor::from_slice(ctx, &bd, Some(dtype::BFLOAT16), Some("CUDA"))
                    .reshape([k as i64, n as i64]);
                vec![a.dot(&b).cast(dtype::FLOAT32)]
            },
        )
    };
    let d = maxdiff(&run_case(false), &run_case(true));
    assert!(d < 5e-3, "bf16: cuBLAS diff {d:e}");
}

#[test]
fn cublas_matches_scale_and_residual_epilogue() {
    let _guard = LOCK.lock().unwrap();
    if !have_cuda_and_cublas() {
        return;
    }
    // `x + (linear(h) + bias) * layer_scale`, the transformer block epilogue the
    // store-level GEMM detection has to peel to keep the contraction dispatchable.
    let (m, k, n) = (197usize, 384usize, 384usize);
    let hd = values(2, m * k);
    let wd = values(4, n * k);
    let bd = values(6, n);
    let sd = values(8, n);
    let xd = values(11, m * n);
    let run_case = |lib: bool| {
        run(
            if lib {
                autorust::settings::LibraryGemm::Always
            } else {
                autorust::settings::LibraryGemm::Never
            },
            |ctx| {
                let h =
                    Tensor::from_slice(ctx, &hd, None, Some("CUDA")).reshape([m as i64, k as i64]);
                let x =
                    Tensor::from_slice(ctx, &xd, None, Some("CUDA")).reshape([m as i64, n as i64]);
                let lin = Linear {
                    weight: Tensor::from_slice(ctx, &wd, None, Some("CUDA"))
                        .reshape([n as i64, k as i64]),
                    bias: Some(Tensor::from_slice(ctx, &bd, None, Some("CUDA"))),
                };
                let scale = Tensor::from_slice(ctx, &sd, None, Some("CUDA"));
                vec![x.add(lin.forward(&h).mul(&scale))]
            },
        )
    };
    let d = maxdiff(&run_case(false), &run_case(true));
    assert!(d < 1e-3, "scale+residual: cuBLAS diff {d:e}");
}
