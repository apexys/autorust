//! Fused FA2 kernel on OpenCL vs `scaled_dot_product_attention`.

#![cfg(feature = "backend_cl")]

use std::time::Instant;

use autorust::runtime::device::device;
use autorust::tensor::{Ctx, Tensor};

fn fill(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i + seed) % 37) as f32) * 0.01 - 0.18)
        .collect()
}

#[test]
fn flash_cl_matches_sdpa_and_timing() {
    if !autorust::runtime::ops_cl::available() {
        return;
    }
    let (b, h, n, d) = (1i64, 6i64, 789i64, 64i64);
    let ctx = Ctx::new();
    let mut sess = autorust::engine::session::Session::new();
    let q = Tensor::from_slice(&ctx, &fill((b * h * n * d) as usize, 3), None, Some("CL"))
        .reshape([b, h, n, d]);
    let k = Tensor::from_slice(&ctx, &fill((b * h * n * d) as usize, 11), None, Some("CL"))
        .reshape([b, h, n, d]);
    let v = Tensor::from_slice(&ctx, &fill((b * h * n * d) as usize, 19), None, Some("CL"))
        .reshape([b, h, n, d]);
    for t in [&q, &k, &v] {
        let _ = t.to_list::<f32>(&mut sess);
    }

    let t0 = Instant::now();
    let want = q
        .scaled_dot_product_attention(&k, &v, None, 0.0, false, false)
        .to_list::<f32>(&mut sess);
    let sdp_ms = t0.elapsed().as_secs_f64() * 1e3;

    let o = Tensor::from_slice(&ctx, &vec![0.0f32; want.len()], None, Some("CL"))
        .reshape([b, h, n, d])
        .to_param();
    let _ = o.to_list::<f32>(&mut sess);

    let dev = device("CL");
    let ov = |t: &Tensor| t.realized_buffer().expect("realized").get_buf(&dev);
    let (qv, kv, vv, outv) = (ov(&q), ov(&k), ov(&v), ov(&o));

    autorust::runtime::cl_flash::run(&dev, &qv, &kv, &vv, &outv, b, h, n);
    let t1 = Instant::now();
    for _ in 0..5 {
        autorust::runtime::cl_flash::run(&dev, &qv, &kv, &vv, &outv, b, h, n);
    }
    let flash_ms = t1.elapsed().as_secs_f64() * 1e3 / 5.0;

    let got = o.to_list::<f32>(&mut sess);
    let maxd = got
        .iter()
        .zip(&want)
        .map(|(a, c)| (a - c).abs())
        .fold(0.0f32, f32::max);
    eprintln!("cl flash: {flash_ms:.2} ms   sdpa: {sdp_ms:.2} ms   maxabs {maxd:e}");
    assert!(maxd < 2e-5, "cl flash vs sdpa maxabs {maxd:e}");

    let b8 = 8i64;
    let q8 = Tensor::from_slice(&ctx, &fill((b8 * h * n * d) as usize, 5), None, Some("CL"))
        .reshape([b8, h, n, d])
        .to_param();
    let k8 = Tensor::from_slice(&ctx, &fill((b8 * h * n * d) as usize, 13), None, Some("CL"))
        .reshape([b8, h, n, d])
        .to_param();
    let v8 = Tensor::from_slice(&ctx, &fill((b8 * h * n * d) as usize, 23), None, Some("CL"))
        .reshape([b8, h, n, d])
        .to_param();
    let o8 = Tensor::from_slice(
        &ctx,
        &vec![0.0f32; (b8 * h * n * d) as usize],
        None,
        Some("CL"),
    )
    .reshape([b8, h, n, d])
    .to_param();
    for t in [&q8, &k8, &v8, &o8] {
        let _ = t.to_list::<f32>(&mut sess);
    }
    let (q8v, k8v, v8v, o8v) = (ov(&q8), ov(&k8), ov(&v8), ov(&o8));
    autorust::runtime::cl_flash::run(&dev, &q8v, &k8v, &v8v, &o8v, b8, h, n);
    let t2 = Instant::now();
    for _ in 0..5 {
        autorust::runtime::cl_flash::run(&dev, &q8v, &k8v, &v8v, &o8v, b8, h, n);
    }
    let b8_ms = t2.elapsed().as_secs_f64() * 1e3 / 5.0;
    eprintln!(
        "cl batch 8: {b8_ms:.2} ms total, {:.2} ms/image/layer",
        b8_ms / 8.0
    );
}
