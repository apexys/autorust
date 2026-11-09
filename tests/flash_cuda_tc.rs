//! Warp-tiled TF32 tensor-core FA2 kernel vs a scalar reference.
//!
//! Covers the shapes the DINOv3 attention actually uses (`h=6`, `n=789`,
//! `d=64`) plus small and non-multiple-of-tile lengths, checks the setting
//! gate, and reports the kernel time next to the scalar f32 kernel.

#![cfg(feature = "backend_cuda")]

use std::time::Instant;

use autorust::runtime::device::device;
use autorust::settings::Settings;
use autorust::tensor::{Ctx, Tensor};

fn fill(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i + seed) % 37) as f32) * 0.01 - 0.18)
        .collect()
}

/// Straightforward per-head reference in f32 (no tensor cores, no online
/// softmax), so the comparison does not depend on another backend's path.
fn sdpa_ref(q: &[f32], k: &[f32], v: &[f32], b: i64, h: i64, n: i64, d: i64) -> Vec<f32> {
    let (b, h, n, d) = (b as usize, h as usize, n as usize, d as usize);
    let scale = 1.0 / (d as f32).sqrt();
    let mut out = vec![0.0f32; b * h * n * d];
    for bh in 0..b * h {
        let off = bh * n * d;
        let mut s = vec![0.0f32; n];
        for i in 0..n {
            let mut mx = f32::NEG_INFINITY;
            for j in 0..n {
                let mut dot = 0.0f32;
                for t in 0..d {
                    dot += q[off + i * d + t] * k[off + j * d + t];
                }
                s[j] = dot * scale;
                mx = mx.max(s[j]);
            }
            let mut sum = 0.0f32;
            for x in s.iter_mut() {
                *x = (*x - mx).exp();
                sum += *x;
            }
            for t in 0..d {
                let mut acc = 0.0f32;
                for j in 0..n {
                    acc += s[j] * v[off + j * d + t];
                }
                out[off + i * d + t] = acc / sum;
            }
        }
    }
    out
}

/// Run the tensor-core kernel on `q/k/v` and read the output back.
fn run_tc(
    ctx: &Ctx,
    sess: &mut autorust::engine::session::Session,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    b: i64,
    h: i64,
    n: i64,
    d: i64,
) -> Vec<f32> {
    let out = Tensor::from_slice(
        ctx,
        &vec![0.0f32; (b * h * n * d) as usize],
        None,
        Some("CUDA"),
    )
    .reshape([b, h, n, d])
    .to_param();
    for t in [q, k, v, &out] {
        let _ = t.to_list::<f32>(sess);
    }
    let dev = device("CUDA");
    let ov = |t: &Tensor| t.realized_buffer().expect("realized").get_buf(&dev);
    autorust::runtime::cuda_flash_tc::run(&dev, &ov(q), &ov(k), &ov(v), &ov(&out), b, h, n);
    out.to_list::<f32>(sess)
}

/// Run the scalar f32 kernel on `q/k/v` and read the output back.
fn run_scalar(
    ctx: &Ctx,
    sess: &mut autorust::engine::session::Session,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    b: i64,
    h: i64,
    n: i64,
    d: i64,
) -> Vec<f32> {
    let out = Tensor::from_slice(
        ctx,
        &vec![0.0f32; (b * h * n * d) as usize],
        None,
        Some("CUDA"),
    )
    .reshape([b, h, n, d])
    .to_param();
    for t in [q, k, v, &out] {
        let _ = t.to_list::<f32>(sess);
    }
    let dev = device("CUDA");
    let ov = |t: &Tensor| t.realized_buffer().expect("realized").get_buf(&dev);
    autorust::runtime::cuda_flash::run(&dev, &ov(q), &ov(k), &ov(v), &ov(&out), b, h, n);
    out.to_list::<f32>(sess)
}

#[test]
fn tc_matches_reference() {
    if !autorust::runtime::ops_cuda::available() {
        return;
    }
    if !autorust::runtime::ops_cuda::has_tensor_cores() {
        eprintln!("skip: device has no TF32 tensor cores (sm < 80)");
        return;
    }
    let s = Settings::default();
    let ctx = Ctx::new();
    let mut sess = autorust::engine::session::Session::new();
    let (b, h, d) = (1i64, 6i64, 64i64);
    for n in [1i64, 7, 8, 31, 32, 33, 63, 64, 65, 100, 224, 789] {
        if !autorust::runtime::cuda_flash_tc::supported(n, d, &s) {
            panic!("expected tensor-core kernel to support n={n}, d={d}");
        }
        let qd = fill((b * h * n * d) as usize, 3);
        let kd = fill((b * h * n * d) as usize, 11);
        let vd = fill((b * h * n * d) as usize, 19);
        let q = Tensor::from_slice(&ctx, &qd, None, Some("CUDA")).reshape([b, h, n, d]);
        let k = Tensor::from_slice(&ctx, &kd, None, Some("CUDA")).reshape([b, h, n, d]);
        let v = Tensor::from_slice(&ctx, &vd, None, Some("CUDA")).reshape([b, h, n, d]);
        let want = sdpa_ref(&qd, &kd, &vd, b, h, n, d);
        let exact = run_scalar(&ctx, &mut sess, &q, &k, &v, b, h, n, d);
        let got = run_tc(&ctx, &mut sess, &q, &k, &v, b, h, n, d);
        let maxd = |a: &[f32], c: &[f32]| {
            a.iter()
                .zip(c)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };
        eprintln!(
            "n={n}: scalar-vs-ref {:e}, tc-vs-ref {:e}, tc-vs-scalar {:e}",
            maxd(&exact, &want),
            maxd(&got, &want),
            maxd(&got, &exact)
        );
        assert!(
            maxd(&got, &want) < 5e-3,
            "n={n}: tc vs scalar-ref maxabs {:e}",
            maxd(&got, &want)
        );
    }
}

#[test]
fn tc_gate_respects_settings() {
    if !autorust::runtime::ops_cuda::available() {
        return;
    }
    if !autorust::runtime::ops_cuda::has_tensor_cores() {
        return;
    }
    let mut s = Settings::default();
    assert!(autorust::runtime::cuda_flash_tc::supported(64, 64, &s));
    // `allow_tf32` is cuBLAS-only; TF32 is inherent to the tensor-core kernel.
    s.allow_tf32 = false;
    assert!(autorust::runtime::cuda_flash_tc::supported(64, 64, &s));
    s.tensor_cores = false;
    assert!(!autorust::runtime::cuda_flash_tc::supported(64, 64, &s));
    assert!(!autorust::runtime::cuda_flash_tc::supported(
        64,
        128,
        &Settings::default()
    ));
}

#[test]
fn tc_timing_vs_scalar() {
    if !autorust::runtime::ops_cuda::available() {
        return;
    }
    if !autorust::runtime::ops_cuda::has_tensor_cores() {
        eprintln!("skip: device has no TF32 tensor cores (sm < 80)");
        return;
    }
    let (b, h, n, d) = (8i64, 6i64, 789i64, 64i64);
    let ctx = Ctx::new();
    let mut sess = autorust::engine::session::Session::new();
    let q = Tensor::from_slice(&ctx, &fill((b * h * n * d) as usize, 5), None, Some("CUDA"))
        .reshape([b, h, n, d]);
    let k = Tensor::from_slice(
        &ctx,
        &fill((b * h * n * d) as usize, 13),
        None,
        Some("CUDA"),
    )
    .reshape([b, h, n, d]);
    let v = Tensor::from_slice(
        &ctx,
        &fill((b * h * n * d) as usize, 23),
        None,
        Some("CUDA"),
    )
    .reshape([b, h, n, d]);
    let o = Tensor::from_slice(
        &ctx,
        &vec![0.0f32; (b * h * n * d) as usize],
        None,
        Some("CUDA"),
    )
    .reshape([b, h, n, d])
    .to_param();
    for t in [&q, &k, &v, &o] {
        let _ = t.to_list::<f32>(&mut sess);
    }
    let dev = device("CUDA");
    let ov = |t: &Tensor| t.realized_buffer().expect("realized").get_buf(&dev);
    let (qv, kv, vv, ovv) = (ov(&q), ov(&k), ov(&v), ov(&o));

    autorust::runtime::cuda_flash::run(&dev, &qv, &kv, &vv, &ovv, b, h, n);
    let run_tc = |async_pipeline: bool| {
        autorust::runtime::cuda_flash_tc::run_ex(
            &dev,
            &qv,
            &kv,
            &vv,
            &ovv,
            None,
            b,
            h,
            n,
            autorust::runtime::cuda_flash_tc::Options {
                causal: false,
                has_mask: false,
                h_kv: h,
                async_pipeline,
            },
        );
    };
    run_tc(true);
    let async_out = o.to_list::<f32>(&mut sess);
    run_tc(false);
    let sync_out = o.to_list::<f32>(&mut sess);
    assert_eq!(
        async_out, sync_out,
        "async and synchronous staging must produce the same values"
    );

    let time = |iters: usize, f: &dyn Fn()| {
        let t = Instant::now();
        for _ in 0..iters {
            f();
        }
        t.elapsed().as_secs_f64() * 1e3 / iters as f64
    };
    let scalar_ms = time(10, &|| {
        autorust::runtime::cuda_flash::run(&dev, &qv, &kv, &vv, &ovv, b, h, n)
    });
    let tc_ms = time(10, &|| run_tc(true));
    let sync_ms = time(10, &|| run_tc(false));
    eprintln!(
        "batch {b}, h {h}, n {n}: scalar {scalar_ms:.2} ms, tf32 {tc_ms:.2} ms, tf32 sync {sync_ms:.2} ms ({:.2}x, async/sync {:.2}x)",
        scalar_ms / tc_ms,
        sync_ms / tc_ms
    );
}

/// The graph-level dispatch should pick the tensor-core kernel when it can.
#[test]
fn fused_dispatch_selects_tc() {
    if !autorust::runtime::ops_cuda::available() {
        return;
    }
    if !autorust::runtime::ops_cuda::has_tensor_cores() {
        return;
    }
    let (b, h, n, d) = (1i64, 2i64, 64i64, 64i64);
    let ctx = Ctx::new();
    let mut sess = autorust::engine::session::Session::new();
    let qd = fill((b * h * n * d) as usize, 3);
    let kd = fill((b * h * n * d) as usize, 11);
    let vd = fill((b * h * n * d) as usize, 19);
    let q = Tensor::from_slice(&ctx, &qd, None, Some("CUDA")).reshape([b, h, n, d]);
    let k = Tensor::from_slice(&ctx, &kd, None, Some("CUDA")).reshape([b, h, n, d]);
    let v = Tensor::from_slice(&ctx, &vd, None, Some("CUDA")).reshape([b, h, n, d]);

    let before = autorust::runtime::cuda_flash_tc::selections();
    let out = q
        .attention_with(&k, &v, None, 0.0, false, &mut sess)
        .to_list::<f32>(&mut sess);
    let after = autorust::runtime::cuda_flash_tc::selections();
    assert!(
        after > before,
        "expected the tensor-core kernel to be selected"
    );

    let want = sdpa_ref(&qd, &kd, &vd, b, h, n, d);
    let maxd = out
        .iter()
        .zip(&want)
        .map(|(a, c)| (a - c).abs())
        .fold(0.0f32, f32::max);
    assert!(maxd < 5e-3, "fused vs scalar-ref maxabs {maxd:e}");
}
