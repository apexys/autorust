//! bf16 warp-tiled tensor-core FA2 kernel vs a bf16-emulating reference.
//!
//! The reference rounds the operands and the probabilities to bf16 (as the
//! hardware does), keeps the accumulations in f32, and stores the result in
//! bf16, so it matches the kernel to within accumulation order.

#![cfg(feature = "backend_cuda")]

use std::time::Instant;

use autorust::dtype;
use autorust::runtime::device::device;
use autorust::settings::Settings;
use autorust::tensor::{Ctx, Tensor};

fn fill(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i + seed) % 37) as f32) * 0.01 - 0.18)
        .collect()
}

/// Round-to-nearest-even f32 -> bf16 -> f32 (what `cvt.rn.bf16.f32` does).
fn bf16(x: f32) -> f32 {
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    f32::from_bits(bits.wrapping_add(0x7fff + lsb) & 0xffff_0000)
}

/// Emulates the kernel: bf16 operands, f32 score/softmax, bf16 P, f32 acc.
fn bf16_ref(q: &[f32], k: &[f32], v: &[f32], b: i64, h: i64, n: i64, d: i64) -> Vec<f32> {
    let (b, h, n, d) = (b as usize, h as usize, n as usize, d as usize);
    let scale = 1.0 / (d as f32).sqrt();
    let mut out = vec![0.0f32; b * h * n * d];
    let qb: Vec<f32> = q.iter().map(|&x| bf16(x)).collect();
    let kb: Vec<f32> = k.iter().map(|&x| bf16(x)).collect();
    let vb: Vec<f32> = v.iter().map(|&x| bf16(x)).collect();
    for bh in 0..b * h {
        let off = bh * n * d;
        for i in 0..n {
            let mut s = vec![0.0f32; n];
            let mut mx = f32::NEG_INFINITY;
            for j in 0..n {
                let mut dot = 0.0f32;
                for t in 0..d {
                    dot += qb[off + i * d + t] * kb[off + j * d + t];
                }
                s[j] = dot * scale;
                mx = mx.max(s[j]);
            }
            let p: Vec<f32> = s.iter().map(|x| (*x - mx).exp()).collect();
            let sum: f32 = p.iter().sum();
            for t in 0..d {
                let mut acc = 0.0f32;
                for j in 0..n {
                    acc += bf16(p[j]) * vb[off + j * d + t];
                }
                out[off + i * d + t] = bf16(acc / sum);
            }
        }
    }
    out
}

fn run_bf16(
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
    let qb = q.cast(dtype::BFLOAT16);
    let kb = k.cast(dtype::BFLOAT16);
    let vb = v.cast(dtype::BFLOAT16);
    let out = Tensor::from_slice(
        ctx,
        &vec![0.0f32; (b * h * n * d) as usize],
        Some(dtype::BFLOAT16),
        Some("CUDA"),
    )
    .reshape([b, h, n, d])
    .to_param();
    {
        let mut c = ctx.borrow_mut();
        for t in [&qb, &kb, &vb, &out] {
            t.realize_device_ctx(&mut c, sess);
        }
    }
    let dev = device("CUDA");
    let ov = |t: &Tensor| t.realized_buffer().expect("realized").get_buf(&dev);
    autorust::runtime::cuda_flash_bf16::run(&dev, &ov(&qb), &ov(&kb), &ov(&vb), &ov(&out), b, h, n);
    out.cast(dtype::FLOAT32).to_list::<f32>(sess)
}

#[test]
fn bf16_matches_reference() {
    if !autorust::runtime::ops_cuda::available() {
        return;
    }
    if !autorust::runtime::ops_cuda::has_tensor_cores() {
        eprintln!("skip: device has no bf16 tensor cores (sm < 80)");
        return;
    }
    let s = Settings::default();
    let ctx = Ctx::new();
    let mut sess = autorust::engine::session::Session::new();
    let (b, h, d) = (1i64, 6i64, 64i64);
    for n in [1i64, 7, 8, 31, 32, 33, 63, 64, 65, 100, 224, 789] {
        assert!(autorust::runtime::cuda_flash_bf16::supported(n, d, &s));
        let qd = fill((b * h * n * d) as usize, 3);
        let kd = fill((b * h * n * d) as usize, 11);
        let vd = fill((b * h * n * d) as usize, 19);
        let q = Tensor::from_slice(&ctx, &qd, None, Some("CUDA")).reshape([b, h, n, d]);
        let k = Tensor::from_slice(&ctx, &kd, None, Some("CUDA")).reshape([b, h, n, d]);
        let v = Tensor::from_slice(&ctx, &vd, None, Some("CUDA")).reshape([b, h, n, d]);
        let want = bf16_ref(&qd, &kd, &vd, b, h, n, d);
        let got = run_bf16(&ctx, &mut sess, &q, &k, &v, b, h, n, d);
        let maxd = got
            .iter()
            .zip(&want)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        eprintln!("n={n}: bf16 kernel vs bf16 ref maxabs {maxd:e}");
        assert!(maxd < 5e-3, "n={n}: bf16 maxabs {maxd:e}");
    }
}

#[test]
fn bf16_gate_and_dispatch() {
    if !autorust::runtime::ops_cuda::has_tensor_cores() {
        return;
    }
    let mut s = Settings::default();
    assert!(autorust::runtime::cuda_flash_bf16::supported(64, 64, &s));
    s.tensor_cores = false;
    assert!(!autorust::runtime::cuda_flash_bf16::supported(64, 64, &s));
    assert!(!autorust::runtime::cuda_flash_bf16::supported(64, 128, &s));

    // The graph-level dispatch should pick the bf16 kernel for bf16 inputs.
    let (b, h, n, d) = (1i64, 2i64, 64i64, 64i64);
    let ctx = Ctx::new();
    let mut sess = autorust::engine::session::Session::new();
    let qd = fill((b * h * n * d) as usize, 3);
    let kd = fill((b * h * n * d) as usize, 11);
    let vd = fill((b * h * n * d) as usize, 19);
    let q = Tensor::from_slice(&ctx, &qd, None, Some("CUDA"))
        .reshape([b, h, n, d])
        .cast(dtype::BFLOAT16);
    let k = Tensor::from_slice(&ctx, &kd, None, Some("CUDA"))
        .reshape([b, h, n, d])
        .cast(dtype::BFLOAT16);
    let v = Tensor::from_slice(&ctx, &vd, None, Some("CUDA"))
        .reshape([b, h, n, d])
        .cast(dtype::BFLOAT16);

    let before = autorust::runtime::cuda_flash_bf16::selections();
    let out = q
        .attention_with(&k, &v, None, 0.0, false, &mut sess)
        .cast(dtype::FLOAT32)
        .to_list::<f32>(&mut sess);
    assert!(
        autorust::runtime::cuda_flash_bf16::selections() > before,
        "expected the bf16 kernel to be selected"
    );
    let want = bf16_ref(&qd, &kd, &vd, b, h, n, d);
    let maxd = out
        .iter()
        .zip(&want)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(maxd < 5e-3, "dispatching bf16 maxabs {maxd:e}");
}

#[test]
fn bf16_timing_vs_f32_tc() {
    if !autorust::runtime::ops_cuda::has_tensor_cores() {
        return;
    }
    let (b, h, n, d) = (8i64, 6i64, 789i64, 64i64);
    let ctx = Ctx::new();
    let mut sess = autorust::engine::session::Session::new();
    let qd = fill((b * h * n * d) as usize, 5);
    let kd = fill((b * h * n * d) as usize, 13);
    let vd = fill((b * h * n * d) as usize, 23);
    let q = Tensor::from_slice(&ctx, &qd, None, Some("CUDA")).reshape([b, h, n, d]);
    let k = Tensor::from_slice(&ctx, &kd, None, Some("CUDA")).reshape([b, h, n, d]);
    let v = Tensor::from_slice(&ctx, &vd, None, Some("CUDA")).reshape([b, h, n, d]);
    let qb = q.cast(dtype::BFLOAT16);
    let kb = k.cast(dtype::BFLOAT16);
    let vb = v.cast(dtype::BFLOAT16);
    let of = Tensor::from_slice(
        &ctx,
        &vec![0.0f32; (b * h * n * d) as usize],
        None,
        Some("CUDA"),
    )
    .reshape([b, h, n, d])
    .to_param();
    let ob = Tensor::from_slice(
        &ctx,
        &vec![0.0f32; (b * h * n * d) as usize],
        Some(dtype::BFLOAT16),
        Some("CUDA"),
    )
    .reshape([b, h, n, d])
    .to_param();
    {
        let mut c = ctx.borrow_mut();
        for t in [&q, &k, &v, &qb, &kb, &vb, &of, &ob] {
            t.realize_device_ctx(&mut c, &mut sess);
        }
    }
    let dev = device("CUDA");
    let ov = |t: &Tensor| t.realized_buffer().expect("realized").get_buf(&dev);
    autorust::runtime::cuda_flash_tc::run(&dev, &ov(&q), &ov(&k), &ov(&v), &ov(&of), b, h, n);
    autorust::runtime::cuda_flash_bf16::run(&dev, &ov(&qb), &ov(&kb), &ov(&vb), &ov(&ob), b, h, n);

    let t0 = Instant::now();
    for _ in 0..5 {
        autorust::runtime::cuda_flash_tc::run(&dev, &ov(&q), &ov(&k), &ov(&v), &ov(&of), b, h, n);
    }
    let tf32_ms = t0.elapsed().as_secs_f64() * 1e3 / 5.0;
    let t1 = Instant::now();
    for _ in 0..5 {
        autorust::runtime::cuda_flash_bf16::run(
            &dev,
            &ov(&qb),
            &ov(&kb),
            &ov(&vb),
            &ov(&ob),
            b,
            h,
            n,
        );
    }
    let bf16_ms = t1.elapsed().as_secs_f64() * 1e3 / 5.0;
    eprintln!(
        "batch {b}, h {h}, n {n}: tf32 {tf32_ms:.2} ms, bf16 {bf16_ms:.2} ms ({:.2}x)",
        tf32_ms / bf16_ms
    );
}
