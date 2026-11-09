//! Causal masking, additive masks and GQA on the OpenCL FA2 kernel.
//!
//! Compares `cl_flash::run_ex` against a straightforward f32 reference (the
//! backend's `scaled_dot_product_attention` rejects mask+causal together).

#![cfg(feature = "backend_cl")]

use autorust::runtime::device::device;
use autorust::tensor::{Ctx, Tensor};

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
    let ctx = Ctx::new();
    let mut sess = autorust::engine::session::Session::new();
    let qd = fill((b * hq * n * d) as usize, 3);
    let kd = fill((b * hkv * n * d) as usize, 11);
    let vd = fill((b * hkv * n * d) as usize, 19);
    let md: Vec<f32> = (0..(n * n) as usize)
        .map(|i| (((i * 7) % 23) as f32 - 11.0) * 0.05)
        .collect();
    let q = Tensor::from_slice(&ctx, &qd, None, Some("CL")).reshape([b, hq, n, d]);
    let k = Tensor::from_slice(&ctx, &kd, None, Some("CL")).reshape([b, hkv, n, d]);
    let v = Tensor::from_slice(&ctx, &vd, None, Some("CL")).reshape([b, hkv, n, d]);
    let mask = has_mask.then(|| Tensor::from_slice(&ctx, &md, None, Some("CL")).reshape([n, n]));
    let out = Tensor::from_slice(
        &ctx,
        &vec![0.0f32; (b * hq * n * d) as usize],
        None,
        Some("CL"),
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
    let dev = device("CL");
    let ov = |t: &Tensor| t.realized_buffer().expect("realized").get_buf(&dev);
    let mv = mask
        .as_ref()
        .map(|m| m.realized_buffer().expect("realized").get_buf(&dev));
    autorust::runtime::cl_flash::run_ex(
        &dev,
        &ov(&q),
        &ov(&k),
        &ov(&v),
        &ov(&out),
        mv.as_ref(),
        b,
        hq,
        n,
        autorust::runtime::cl_flash::Options {
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
    eprintln!("cl {label}: maxabs {maxd:e}");
    assert!(maxd < 2e-5, "cl {label}: maxabs {maxd:e}");
}

#[test]
fn cl_flash_causal_mask_gqa() {
    if !autorust::runtime::ops_cl::available() {
        return;
    }
    check("causal", 1, 2, 2, 100, 64, true, false);
    check("mask", 1, 2, 2, 64, 64, false, true);
    check("causal+mask", 1, 2, 2, 100, 64, true, true);
    check("gqa", 2, 4, 2, 64, 64, false, false);
    check("gqa+causal", 2, 4, 2, 100, 64, true, false);
    check("gqa+mask", 2, 4, 1, 64, 64, false, true);
}
