//! INT8 SageAttention-style QKᵀ vs a reference that performs the same
//! per-row/per-key quantization (testable on sm_80..88, including the 3060).

#![cfg(all(
    feature = "backend_cuda",
    any(target_os = "linux", target_os = "windows")
))]

use std::time::Instant;

use autorust::runtime::device::device;
use autorust::tensor::{Ctx, Tensor};

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

fn q8(x: f32) -> i32 {
    let v = x.round_ties_even() as i32;
    v.clamp(-127, 127)
}

/// Emulates the kernel: per-row Q and per-key K symmetric int8, `s32` QKᵀ,
/// f32 softmax, bf16 P/V.
fn sage_ref(q: &[f32], k: &[f32], v: &[f32], b: i64, h: i64, n: i64, d: i64) -> Vec<f32> {
    let (b, h, n, d) = (b as usize, h as usize, n as usize, d as usize);
    let mut out = vec![0.0f32; b * h * n * d];
    for bh in 0..b * h {
        let off = bh * n * d;
        let mut sq = vec![1.0f32; n];
        let mut sk = vec![1.0f32; n];
        for i in 0..n {
            let mq = (0..d).fold(0.0f32, |m, t| m.max(q[off + i * d + t].abs()));
            let mk = (0..d).fold(0.0f32, |m, t| m.max(k[off + i * d + t].abs()));
            if mq > 0.0 {
                sq[i] = mq / 127.0;
            }
            if mk > 0.0 {
                sk[i] = mk / 127.0;
            }
        }
        for i in 0..n {
            let mut s = vec![0.0f32; n];
            let mut mx = f32::NEG_INFINITY;
            for j in 0..n {
                let mut acc = 0i32;
                for t in 0..d {
                    let qi = q8(q[off + i * d + t] / sq[i]);
                    let kj = q8(k[off + j * d + t] / sk[j]);
                    acc += qi * kj;
                }
                s[j] = acc as f32 * (sq[i] * sk[j] * 0.125);
                mx = mx.max(s[j]);
            }
            let p: Vec<f32> = s.iter().map(|x| (*x - mx).exp()).collect();
            let sum: f32 = p.iter().sum();
            for t in 0..d {
                let mut acc = 0.0f32;
                for j in 0..n {
                    acc += bf16(p[j]) * bf16(v[off + j * d + t]);
                }
                out[off + i * d + t] = acc / sum;
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn run_sage(
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
    {
        let mut c = ctx.borrow_mut();
        for t in [q, k, v, &out] {
            t.realize_device_ctx(&mut c, sess);
        }
    }
    let dev = device("CUDA");
    let ov = |t: &Tensor| t.realized_buffer().expect("realized").get_buf(&dev);
    autorust::runtime::cuda_flash_sage::run_ex(
        &dev,
        &ov(q),
        &ov(k),
        &ov(v),
        &ov(&out),
        None,
        b,
        h,
        n,
        autorust::runtime::cuda_flash_sage::Options {
            causal: false,
            has_mask: false,
            h_kv: h,
            fp8: false,
        },
    );
    out.to_list::<f32>(sess)
}

#[test]
fn int8_sage_matches_reference() {
    if !autorust::runtime::ops_cuda::has_tensor_cores() {
        return;
    }
    let ctx = Ctx::new();
    let mut sess = autorust::engine::session::Session::new();
    let (b, h, d) = (1i64, 6i64, 64i64);
    for n in [1i64, 8, 33, 100, 789] {
        let qd = fill((b * h * n * d) as usize, 3);
        let kd = fill((b * h * n * d) as usize, 11);
        let vd = fill((b * h * n * d) as usize, 19);
        let q = Tensor::from_slice(&ctx, &qd, None, Some("CUDA")).reshape([b, h, n, d]);
        let k = Tensor::from_slice(&ctx, &kd, None, Some("CUDA")).reshape([b, h, n, d]);
        let v = Tensor::from_slice(&ctx, &vd, None, Some("CUDA")).reshape([b, h, n, d]);
        let got = run_sage(&ctx, &mut sess, &q, &k, &v, b, h, n, d);
        let want = sage_ref(&qd, &kd, &vd, b, h, n, d);
        let maxd = got
            .iter()
            .zip(&want)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        eprintln!("n={n}: int8 sage vs reference maxabs {maxd:e}");
        assert!(maxd < 2e-3, "n={n}: int8 sage maxabs {maxd:e}");
    }
}

#[test]
fn int8_sage_dispatch_and_timing() {
    if !autorust::runtime::ops_cuda::has_tensor_cores() {
        return;
    }
    // The explicit `sage_attention_with` path must select the quantized kernel.
    let (b, h, n, d) = (1i64, 2i64, 64i64, 64i64);
    let ctx = Ctx::new();
    let mut sess = autorust::engine::session::Session::new();
    let qd = fill((b * h * n * d) as usize, 3);
    let kd = fill((b * h * n * d) as usize, 11);
    let vd = fill((b * h * n * d) as usize, 19);
    let q = Tensor::from_slice(&ctx, &qd, None, Some("CUDA")).reshape([b, h, n, d]);
    let k = Tensor::from_slice(&ctx, &kd, None, Some("CUDA")).reshape([b, h, n, d]);
    let v = Tensor::from_slice(&ctx, &vd, None, Some("CUDA")).reshape([b, h, n, d]);
    let before = autorust::runtime::cuda_flash_sage::selections();
    let got = q
        .sage_attention_with(&k, &v, None, 0.0, false, &mut sess)
        .to_list::<f32>(&mut sess);
    assert!(
        autorust::runtime::cuda_flash_sage::selections() > before,
        "expected the quantized kernel to be selected"
    );
    let want = sage_ref(&qd, &kd, &vd, b, h, n, d);
    let maxd = got
        .iter()
        .zip(&want)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(maxd < 2e-3, "sage dispatch maxabs {maxd:e}");

    // Timing at the app shape.
    let (b, h, n) = (8i64, 6i64, 789i64);
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
    let run = |fp8: bool| {
        autorust::runtime::cuda_flash_sage::run_ex(
            &dev,
            &ov(&q),
            &ov(&k),
            &ov(&v),
            &ov(&o),
            None,
            b,
            h,
            n,
            autorust::runtime::cuda_flash_sage::Options {
                causal: false,
                has_mask: false,
                h_kv: h,
                fp8,
            },
        )
    };
    run(false);
    let t0 = Instant::now();
    for _ in 0..10 {
        run(false);
    }
    let int8_ms = t0.elapsed().as_secs_f64() * 1e3 / 10.0;
    let t1 = Instant::now();
    for _ in 0..10 {
        autorust::runtime::cuda_flash_tc::run(&dev, &ov(&q), &ov(&k), &ov(&v), &ov(&o), b, h, n);
    }
    let tf32_ms = t1.elapsed().as_secs_f64() * 1e3 / 10.0;
    eprintln!(
        "batch {b}, h {h}, n {n}: int8 sage {int8_ms:.2} ms, tf32 {tf32_ms:.2} ms ({:.2}x)",
        tf32_ms / int8_ms
    );
}

/// E4M3 (positive) representable values: subnormals, normals, and the finite
/// top row (`448` max; `m=7` at `e=15` is NaN and excluded).
fn e4m3_values() -> Vec<f32> {
    let mut v = Vec::new();
    for m in 0..8 {
        v.push((m as f32 / 8.0) * 2f32.powi(-6)); // e=0 subnormal
    }
    for e in 1..15 {
        for m in 0..8 {
            v.push((1.0 + m as f32 / 8.0) * 2f32.powi(e - 7));
        }
    }
    for m in 0..7 {
        v.push((1.0 + m as f32 / 8.0) * 2f32.powi(8));
    }
    v
}

fn e4m3(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let a = x.abs();
    let vals = e4m3_values();
    let mut best = 0.0;
    let mut bd = f32::INFINITY;
    for &v in &vals {
        let d = (v - a).abs();
        if d < bd {
            bd = d;
            best = v;
        }
    }
    sign * best
}

/// Same structure as [`sage_ref`], with E4M3 operands and `max/448` scales.
fn sage_ref_fp8(q: &[f32], k: &[f32], v: &[f32], b: i64, h: i64, n: i64, d: i64) -> Vec<f32> {
    let (b, h, n, d) = (b as usize, h as usize, n as usize, d as usize);
    let mut out = vec![0.0f32; b * h * n * d];
    for bh in 0..b * h {
        let off = bh * n * d;
        let mut sq = vec![1.0f32; n];
        let mut sk = vec![1.0f32; n];
        for i in 0..n {
            let mq = (0..d).fold(0.0f32, |m, t| m.max(q[off + i * d + t].abs()));
            let mk = (0..d).fold(0.0f32, |m, t| m.max(k[off + i * d + t].abs()));
            if mq > 0.0 {
                sq[i] = mq / 448.0;
            }
            if mk > 0.0 {
                sk[i] = mk / 448.0;
            }
        }
        for i in 0..n {
            let mut s = vec![0.0f32; n];
            let mut mx = f32::NEG_INFINITY;
            for j in 0..n {
                let mut acc = 0.0f32;
                for t in 0..d {
                    let qi = e4m3(q[off + i * d + t] / sq[i]);
                    let kj = e4m3(k[off + j * d + t] / sk[j]);
                    acc += qi * kj;
                }
                s[j] = acc * (sq[i] * sk[j] * 0.125);
                mx = mx.max(s[j]);
            }
            let p: Vec<f32> = s.iter().map(|x| (*x - mx).exp()).collect();
            let sum: f32 = p.iter().sum();
            for t in 0..d {
                let mut acc = 0.0f32;
                for j in 0..n {
                    acc += bf16(p[j]) * bf16(v[off + j * d + t]);
                }
                out[off + i * d + t] = acc / sum;
            }
        }
    }
    out
}

/// Runs only on Ada/Hopper (`sm_89+`), where the FP8 `mma` exists.
#[test]
fn fp8_sage_matches_reference() {
    if !autorust::runtime::ops_cuda::has_tensor_cores()
        || !autorust::runtime::cuda_flash_sage::fp8_available()
    {
        eprintln!("skip: no FP8 tensor cores (sm < 89)");
        return;
    }
    let ctx = Ctx::new();
    let mut sess = autorust::engine::session::Session::new();
    let (b, h, d) = (1i64, 6i64, 64i64);
    for n in [8i64, 33, 100, 789] {
        let qd = fill((b * h * n * d) as usize, 3);
        let kd = fill((b * h * n * d) as usize, 11);
        let vd = fill((b * h * n * d) as usize, 19);
        let q = Tensor::from_slice(&ctx, &qd, None, Some("CUDA")).reshape([b, h, n, d]);
        let k = Tensor::from_slice(&ctx, &kd, None, Some("CUDA")).reshape([b, h, n, d]);
        let v = Tensor::from_slice(&ctx, &vd, None, Some("CUDA")).reshape([b, h, n, d]);
        let out = Tensor::from_slice(
            &ctx,
            &vec![0.0f32; (b * h * n * d) as usize],
            None,
            Some("CUDA"),
        )
        .reshape([b, h, n, d])
        .to_param();
        {
            let mut c = ctx.borrow_mut();
            for t in [&q, &k, &v, &out] {
                t.realize_device_ctx(&mut c, &mut sess);
            }
        }
        let dev = device("CUDA");
        let ov = |t: &Tensor| t.realized_buffer().expect("realized").get_buf(&dev);
        autorust::runtime::cuda_flash_sage::run_ex(
            &dev,
            &ov(&q),
            &ov(&k),
            &ov(&v),
            &ov(&out),
            None,
            b,
            h,
            n,
            autorust::runtime::cuda_flash_sage::Options {
                causal: false,
                has_mask: false,
                h_kv: h,
                fp8: true,
            },
        );
        let got = out.to_list::<f32>(&mut sess);
        let want = sage_ref_fp8(&qd, &kd, &vd, b, h, n, d);
        let maxd = got
            .iter()
            .zip(&want)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        eprintln!("n={n}: fp8 sage vs reference maxabs {maxd:e}");
        assert!(maxd < 5e-3, "n={n}: fp8 sage maxabs {maxd:e}");
    }
}
