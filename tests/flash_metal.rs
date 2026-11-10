//! Fused FA2 kernel on Metal vs a reference / `scaled_dot_product_attention`.

#![cfg(all(feature = "backend_metal", target_os = "macos"))]

use autorust::runtime::device::device;
use autorust::tensor::{Ctx, Tensor};
use std::time::Instant;

fn fill(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i + seed) % 37) as f32) * 0.01 - 0.18)
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn reference(
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
                        dot += q[qo + i * d + t] * k[ko + j * d + t];
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
                        acc += p[j] * v[ko + j * d + t];
                    }
                    out[qo + i * d + t] = acc / sum;
                }
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn check(label: &str, b: i64, hq: i64, hkv: i64, n: i64, d: i64, causal: bool, has_mask: bool) {
    if !autorust::runtime::ops_metal::available() {
        return;
    }
    let ctx = Ctx::new();
    let mut sess = autorust::engine::session::Session::new();
    let qd = fill((b * hq * n * d) as usize, 3);
    let kd = fill((b * hkv * n * d) as usize, 11);
    let vd = fill((b * hkv * n * d) as usize, 19);
    let md: Vec<f32> = (0..(n * n) as usize)
        .map(|i| (((i * 7) % 23) as f32 - 11.0) * 0.05)
        .collect();
    let q = Tensor::from_slice(&ctx, &qd, None, Some("METAL")).reshape([b, hq, n, d]);
    let k = Tensor::from_slice(&ctx, &kd, None, Some("METAL")).reshape([b, hkv, n, d]);
    let v = Tensor::from_slice(&ctx, &vd, None, Some("METAL")).reshape([b, hkv, n, d]);
    let mask = has_mask.then(|| Tensor::from_slice(&ctx, &md, None, Some("METAL")).reshape([n, n]));
    let out = Tensor::from_slice(
        &ctx,
        &vec![0.0f32; (b * hq * n * d) as usize],
        None,
        Some("METAL"),
    )
    .reshape([b, hq, n, d])
    .to_param();
    {
        let mut c = ctx.borrow_mut();
        for t in [&q, &k, &v, &out] {
            t.realize_device_ctx(&mut c, &mut sess);
        }
        if let Some(m) = &mask {
            m.realize_device_ctx(&mut c, &mut sess);
        }
    }
    let dev = device("METAL");
    let ov = |t: &Tensor| t.realized_buffer().expect("realized").get_buf(&dev);
    let mv = mask
        .as_ref()
        .map(|m| m.realized_buffer().expect("realized").get_buf(&dev));
    autorust::runtime::metal_flash::run_ex(
        &dev,
        &ov(&q),
        &ov(&k),
        &ov(&v),
        &ov(&out),
        mv.as_ref(),
        b,
        hq,
        n,
        autorust::runtime::metal_flash::Options {
            causal,
            has_mask,
            h_kv: hkv,
        },
    );
    let got = out.to_list::<f32>(&mut sess);
    let want = reference(
        &qd,
        &kd,
        &vd,
        if has_mask { Some(&md) } else { None },
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
        .map(|(a, c)| (a - c).abs())
        .fold(0.0f32, f32::max);
    eprintln!("metal {label}: maxabs {maxd:e}");
    assert!(maxd < 2e-5, "metal {label}: maxabs {maxd:e}");
}

#[test]
fn metal_flash_causal_mask_gqa() {
    check("causal", 1, 2, 2, 100, 64, true, false);
    check("mask", 1, 2, 2, 64, 64, false, true);
    check("causal+mask", 1, 2, 2, 100, 64, true, true);
    check("gqa", 2, 4, 2, 64, 64, false, false);
    check("odd-n", 1, 1, 1, 101, 64, false, false);
    check("nocausal16", 1, 1, 1, 16, 64, false, false);
    check("nocausal32", 1, 1, 1, 32, 64, false, false);
    check("nocausal", 1, 1, 1, 100, 64, false, false);
    check("nocausal-large", 1, 2, 2, 789, 64, false, false);
}

#[test]
fn metal_flash_dispatcher_matches_sdpa() {
    if !autorust::runtime::ops_metal::available() {
        return;
    }
    let (b, h, n, d) = (2i64, 2i64, 67i64, 64i64);
    let ctx = Ctx::new();
    let mut sess = autorust::engine::session::Session::new();
    let q = Tensor::from_slice(
        &ctx,
        &fill((b * h * n * d) as usize, 3),
        None,
        Some("METAL"),
    )
    .reshape([b, h, n, d]);
    let k = Tensor::from_slice(
        &ctx,
        &fill((b * h * n * d) as usize, 11),
        None,
        Some("METAL"),
    )
    .reshape([b, h, n, d]);
    let v = Tensor::from_slice(
        &ctx,
        &fill((b * h * n * d) as usize, 19),
        None,
        Some("METAL"),
    )
    .reshape([b, h, n, d]);
    let want = q
        .scaled_dot_product_attention(&k, &v, None, 0.0, true, false)
        .to_list::<f32>(&mut sess);
    let got = q
        .flash_attention_with(&k, &v, None, 0.0, true, &mut sess)
        .to_list::<f32>(&mut sess);
    let maxd = got
        .iter()
        .zip(&want)
        .map(|(a, c)| (a - c).abs())
        .fold(0.0f32, f32::max);
    assert!(maxd < 2e-5, "dispatcher flash vs sdpa maxabs {maxd:e}");
}

#[test]
fn metal_flash_matches_sdpa_and_timing() {
    if !autorust::runtime::ops_metal::available() {
        return;
    }
    let (b, h, n, d) = (1i64, 6i64, 789i64, 64i64);
    let ctx = Ctx::new();
    let mut sess = autorust::engine::session::Session::new();
    let q = Tensor::from_slice(
        &ctx,
        &fill((b * h * n * d) as usize, 3),
        None,
        Some("METAL"),
    )
    .reshape([b, h, n, d]);
    let k = Tensor::from_slice(
        &ctx,
        &fill((b * h * n * d) as usize, 11),
        None,
        Some("METAL"),
    )
    .reshape([b, h, n, d]);
    let v = Tensor::from_slice(
        &ctx,
        &fill((b * h * n * d) as usize, 19),
        None,
        Some("METAL"),
    )
    .reshape([b, h, n, d]);
    for t in [&q, &k, &v] {
        let _ = t.to_list::<f32>(&mut sess);
    }

    let t0 = Instant::now();
    let want = q
        .scaled_dot_product_attention(&k, &v, None, 0.0, false, false)
        .to_list::<f32>(&mut sess);
    let sdp_ms = t0.elapsed().as_secs_f64() * 1e3;

    let o = Tensor::from_slice(&ctx, &vec![0.0f32; want.len()], None, Some("METAL"))
        .reshape([b, h, n, d])
        .to_param();
    let _ = o.to_list::<f32>(&mut sess);

    let dev = device("METAL");
    let ov = |t: &Tensor| t.realized_buffer().expect("realized").get_buf(&dev);
    let (qv, kv, vv, outv) = (ov(&q), ov(&k), ov(&v), ov(&o));
    autorust::runtime::metal_flash::run(&dev, &qv, &kv, &vv, &outv, b, h, n);
    let t1 = Instant::now();
    for _ in 0..5 {
        autorust::runtime::metal_flash::run(&dev, &qv, &kv, &vv, &outv, b, h, n);
    }
    let flash_ms = t1.elapsed().as_secs_f64() * 1e3 / 5.0;

    let got = o.to_list::<f32>(&mut sess);
    let maxd = got
        .iter()
        .zip(&want)
        .map(|(a, c)| (a - c).abs())
        .fold(0.0f32, f32::max);
    eprintln!("metal flash: {flash_ms:.2} ms   sdpa: {sdp_ms:.2} ms   maxabs {maxd:e}");
    assert!(maxd < 2e-5, "metal flash vs sdpa maxabs {maxd:e}");
}
