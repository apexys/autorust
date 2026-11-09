//! Causal masking, additive masks and GQA on the three CUDA FA2 kernels.
//!
//! Each case is compared against a straightforward f32 reference (optionally
//! with bf16 operand/probability rounding, which is what the bf16 kernel does).

#![cfg(feature = "backend_cuda")]

use autorust::dtype;
use autorust::runtime::device::device;
use autorust::tensor::{Ctx, Tensor};

#[derive(Clone, Copy, Debug)]
enum Kernel {
    Scalar,
    Tf32,
    Bf16,
}

fn fill(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i + seed) % 37) as f32) * 0.01 - 0.18)
        .collect()
}

fn bf16(x: f32) -> f32 {
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    f32::from_bits(bits.wrapping_add(0x7fff + lsb) & 0xffff_0000)
}

/// f32 reference with optional bf16 rounding and GQA.
#[allow(clippy::too_many_arguments)]
fn attention_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: Option<&[f32]>,
    b: i64,
    hq: i64,
    hkv: i64,
    n: i64,
    d: i64,
    causal: bool,
    round: bool,
) -> Vec<f32> {
    let (b, hq, hkv, n, d) = (
        b as usize,
        hq as usize,
        hkv as usize,
        n as usize,
        d as usize,
    );
    let scale = 1.0 / (d as f32).sqrt();
    let group = hq / hkv;
    let rnd = |x: f32| if round { bf16(x) } else { x };
    let qb: Vec<f32> = q.iter().map(|&x| rnd(x)).collect();
    let kb: Vec<f32> = k.iter().map(|&x| rnd(x)).collect();
    let vb: Vec<f32> = v.iter().map(|&x| rnd(x)).collect();
    let mut out = vec![0.0f32; b * hq * n * d];
    for bi in 0..b {
        for hh in 0..hq {
            let kh = hh / group;
            let qo = (bi * hq + hh) * n * d;
            let ko = (bi * hkv + kh) * n * d;
            for i in 0..n {
                let mut s = vec![f32::NEG_INFINITY; n];
                let mut mx = f32::NEG_INFINITY;
                for j in 0..n {
                    if causal && j > i {
                        continue;
                    }
                    let mut dot = 0.0f32;
                    for t in 0..d {
                        dot += qb[qo + i * d + t] * kb[ko + j * d + t];
                    }
                    dot *= scale;
                    if let Some(m) = mask {
                        dot += m[i * n + j];
                    }
                    s[j] = dot;
                    mx = mx.max(dot);
                }
                let p: Vec<f32> = s.iter().map(|x| (*x - mx).exp()).collect();
                let sum: f32 = p.iter().sum();
                for t in 0..d {
                    let mut acc = 0.0f32;
                    for j in 0..n {
                        acc += rnd(p[j]) * vb[ko + j * d + t];
                    }
                    out[qo + i * d + t] = rnd(acc / sum);
                }
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn run_kernel(
    kind: Kernel,
    ctx: &Ctx,
    sess: &mut autorust::engine::session::Session,
    qd: &[f32],
    kd: &[f32],
    vd: &[f32],
    mask: Option<&[f32]>,
    b: i64,
    hq: i64,
    hkv: i64,
    n: i64,
    d: i64,
    causal: bool,
) -> Vec<f32> {
    let q = Tensor::from_slice(ctx, qd, None, Some("CUDA")).reshape([b, hq, n, d]);
    let k = Tensor::from_slice(ctx, kd, None, Some("CUDA")).reshape([b, hkv, n, d]);
    let v = Tensor::from_slice(ctx, vd, None, Some("CUDA")).reshape([b, hkv, n, d]);
    let m = mask.map(|m| Tensor::from_slice(ctx, m, None, Some("CUDA")).reshape([n, n]));
    let (qb, kb, vb) = if matches!(kind, Kernel::Bf16) {
        (
            q.cast(dtype::BFLOAT16),
            k.cast(dtype::BFLOAT16),
            v.cast(dtype::BFLOAT16),
        )
    } else {
        (q, k, v)
    };
    let out = if matches!(kind, Kernel::Bf16) {
        Tensor::from_slice(
            ctx,
            &vec![0.0f32; (b * hq * n * d) as usize],
            Some(dtype::BFLOAT16),
            Some("CUDA"),
        )
    } else {
        Tensor::from_slice(
            ctx,
            &vec![0.0f32; (b * hq * n * d) as usize],
            None,
            Some("CUDA"),
        )
    }
    .reshape([b, hq, n, d])
    .to_param();
    {
        let mut c = ctx.borrow_mut();
        for t in [&qb, &kb, &vb, &out] {
            t.realize_device_ctx(&mut c, sess);
        }
        if let Some(m) = &m {
            m.realize_device_ctx(&mut c, sess);
        }
    }
    let dev = device("CUDA");
    let ov = |t: &Tensor| t.realized_buffer().expect("realized").get_buf(&dev);
    let mv = m
        .as_ref()
        .map(|m| m.realized_buffer().expect("mask realized").get_buf(&dev));
    let opts_common = (causal, mask.is_some(), hkv);
    match kind {
        Kernel::Scalar => autorust::runtime::cuda_flash::run_ex(
            &dev,
            &ov(&qb),
            &ov(&kb),
            &ov(&vb),
            &ov(&out),
            mv.as_ref(),
            b,
            hq,
            n,
            autorust::runtime::cuda_flash::Options {
                causal: opts_common.0,
                has_mask: opts_common.1,
                h_kv: opts_common.2,
            },
        ),
        Kernel::Tf32 => autorust::runtime::cuda_flash_tc::run_ex(
            &dev,
            &ov(&qb),
            &ov(&kb),
            &ov(&vb),
            &ov(&out),
            mv.as_ref(),
            b,
            hq,
            n,
            autorust::runtime::cuda_flash_tc::Options {
                causal: opts_common.0,
                has_mask: opts_common.1,
                h_kv: opts_common.2,
            },
        ),
        Kernel::Bf16 => autorust::runtime::cuda_flash_bf16::run_ex(
            &dev,
            &ov(&qb),
            &ov(&kb),
            &ov(&vb),
            &ov(&out),
            mv.as_ref(),
            b,
            hq,
            n,
            autorust::runtime::cuda_flash_bf16::Options {
                causal: opts_common.0,
                has_mask: opts_common.1,
                h_kv: opts_common.2,
            },
        ),
    }
    out.cast(dtype::FLOAT32).to_list::<f32>(sess)
}

#[test]
fn causal_mask_gqa_all_kernels() {
    if !autorust::runtime::ops_cuda::available() {
        return;
    }
    if !autorust::runtime::ops_cuda::has_tensor_cores() {
        eprintln!("skip: no tensor cores");
        return;
    }
    let ctx = Ctx::new();
    let mut sess = autorust::engine::session::Session::new();
    let d = 64i64;

    // (label, b, hq, hkv, n, causal, has_mask)
    let cases = [
        ("causal", 1, 2, 2, 100, true, false),
        ("mask", 1, 2, 2, 64, false, true),
        ("causal+mask", 1, 2, 2, 100, true, true),
        ("gqa", 2, 4, 2, 64, false, false),
        ("gqa+causal", 2, 4, 2, 100, true, false),
        ("gqa+mask", 2, 4, 1, 64, false, true),
    ];
    for (label, b, hq, hkv, n, causal, has_mask) in cases {
        let qd = fill((b * hq * n * d) as usize, 3);
        let kd = fill((b * hkv * n * d) as usize, 11);
        let vd = fill((b * hkv * n * d) as usize, 19);
        let mask: Option<Vec<f32>> = has_mask.then(|| {
            (0..(n * n) as usize)
                .map(|i| (((i * 7) % 23) as f32 - 11.0) * 0.05)
                .collect()
        });
        for kind in [Kernel::Scalar, Kernel::Tf32, Kernel::Bf16] {
            let round = matches!(kind, Kernel::Bf16);
            let want = attention_ref(
                &qd,
                &kd,
                &vd,
                mask.as_deref(),
                b,
                hq,
                hkv,
                n,
                d,
                causal,
                round,
            );
            let got = run_kernel(
                kind,
                &ctx,
                &mut sess,
                &qd,
                &kd,
                &vd,
                mask.as_deref(),
                b,
                hq,
                hkv,
                n,
                d,
                causal,
            );
            let maxd = got
                .iter()
                .zip(&want)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            let tol = if round { 5e-3 } else { 2e-4 };
            eprintln!("{label} {kind:?}: maxabs {maxd:e}");
            assert!(maxd < tol, "{label} {kind:?}: maxabs {maxd:e}");
        }
    }
}

/// The graph-level `attention_with` path must take the fused kernel for a mask
/// and for causal, and match SDPA.
#[test]
fn dispatch_handles_mask_and_causal() {
    if !autorust::runtime::ops_cuda::has_tensor_cores() {
        return;
    }
    let (b, h, n, d) = (1i64, 2i64, 64i64, 64i64);
    let ctx = Ctx::new();
    let mut sess = autorust::engine::session::Session::new();
    let qd = fill((b * h * n * d) as usize, 3);
    let kd = fill((b * h * n * d) as usize, 11);
    let vd = fill((b * h * n * d) as usize, 19);
    let md: Vec<f32> = (0..(n * n) as usize)
        .map(|i| (((i * 5) % 17) as f32 - 8.0) * 0.02)
        .collect();
    let q = Tensor::from_slice(&ctx, &qd, None, Some("CUDA")).reshape([b, h, n, d]);
    let k = Tensor::from_slice(&ctx, &kd, None, Some("CUDA")).reshape([b, h, n, d]);
    let v = Tensor::from_slice(&ctx, &vd, None, Some("CUDA")).reshape([b, h, n, d]);
    let m = Tensor::from_slice(&ctx, &md, None, Some("CUDA")).reshape([n, n]);

    let want = attention_ref(&qd, &kd, &vd, Some(&md), b, h, h, n, d, true, false);
    let before = autorust::runtime::cuda_flash_tc::selections();
    let got = q
        .attention_with(&k, &v, Some(&m), 0.0, true, &mut sess)
        .to_list::<f32>(&mut sess);
    assert!(
        autorust::runtime::cuda_flash_tc::selections() > before,
        "expected the fused kernel for causal+mask"
    );
    let maxd = got
        .iter()
        .zip(&want)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(maxd < 2e-4, "dispatch causal+mask maxabs {maxd:e}");
}

/// The automatic path must handle GQA (fewer K/V heads) and match SDPA.
#[test]
fn dispatch_handles_gqa() {
    if !autorust::runtime::ops_cuda::has_tensor_cores() {
        return;
    }
    let (b, hq, hkv, n, d) = (1i64, 4i64, 2i64, 64i64, 64i64);
    let ctx = Ctx::new();
    let mut sess = autorust::engine::session::Session::new();
    let qd = fill((b * hq * n * d) as usize, 3);
    let kd = fill((b * hkv * n * d) as usize, 11);
    let vd = fill((b * hkv * n * d) as usize, 19);
    let q = Tensor::from_slice(&ctx, &qd, None, Some("CUDA")).reshape([b, hq, n, d]);
    let k = Tensor::from_slice(&ctx, &kd, None, Some("CUDA")).reshape([b, hkv, n, d]);
    let v = Tensor::from_slice(&ctx, &vd, None, Some("CUDA")).reshape([b, hkv, n, d]);

    let want = attention_ref(&qd, &kd, &vd, None, b, hq, hkv, n, d, false, false);
    let got = q
        .attention_with(&k, &v, None, 0.0, false, &mut sess)
        .to_list::<f32>(&mut sess);
    let maxd = got
        .iter()
        .zip(&want)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(maxd < 2e-4, "dispatch gqa maxabs {maxd:e}");
}

/// Causal should nearly halve the key tiles at the app shape.
#[test]
fn causal_skips_half_the_work() {
    if !autorust::runtime::ops_cuda::has_tensor_cores() {
        return;
    }
    use std::time::Instant;
    let (b, h, n, d) = (8i64, 6i64, 789i64, 64i64);
    let ctx = Ctx::new();
    let mut sess = autorust::engine::session::Session::new();
    let qd = fill((b * h * n * d) as usize, 5);
    let kd = fill((b * h * n * d) as usize, 13);
    let vd = fill((b * h * n * d) as usize, 23);
    let q = Tensor::from_slice(&ctx, &qd, None, Some("CUDA")).reshape([b, h, n, d]);
    let k = Tensor::from_slice(&ctx, &kd, None, Some("CUDA")).reshape([b, h, n, d]);
    let v = Tensor::from_slice(&ctx, &vd, None, Some("CUDA")).reshape([b, h, n, d]);
    let o = Tensor::from_slice(
        &ctx,
        &vec![0.0f32; (b * h * n * d) as usize],
        None,
        Some("CUDA"),
    )
    .reshape([b, h, n, d])
    .to_param();
    {
        let mut c = ctx.borrow_mut();
        for t in [&q, &k, &v, &o] {
            t.realize_device_ctx(&mut c, &mut sess);
        }
    }
    let dev = device("CUDA");
    let ov = |t: &Tensor| t.realized_buffer().expect("realized").get_buf(&dev);
    let run = |causal: bool| {
        let opts = autorust::runtime::cuda_flash_tc::Options {
            causal,
            has_mask: false,
            h_kv: h,
        };
        autorust::runtime::cuda_flash_tc::run_ex(
            &dev,
            &ov(&q),
            &ov(&k),
            &ov(&v),
            &ov(&o),
            None,
            b,
            h,
            n,
            opts,
        );
    };
    run(false);
    run(true);
    let t0 = Instant::now();
    for _ in 0..5 {
        run(false);
    }
    let full = t0.elapsed().as_secs_f64() * 1e3 / 5.0;
    let t1 = Instant::now();
    for _ in 0..5 {
        run(true);
    }
    let causal = t1.elapsed().as_secs_f64() * 1e3 / 5.0;
    eprintln!(
        "tf32 n={n}: full {full:.2} ms, causal {causal:.2} ms ({:.2}x)",
        full / causal
    );
    assert!(
        causal < full * 0.8,
        "causal should skip key tiles: {causal} vs {full}"
    );
}
