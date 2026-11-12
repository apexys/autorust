//! `VK` backend gate: run the same graphs on `VK` and `INTERP`, compare.
//!
//! Skipped when no usable Vulkan device is present (like the CUDA gates), so a
//! machine without a driver still passes `cargo test`.

use autorust::dtype;
use autorust::engine::realize::{ProgCache, tensor_values};
use autorust::runtime::buffer::BufTable;
use autorust::tensor::{Ctx, Tensor};

fn have_vk() -> bool {
    autorust::runtime::ops_vk::available()
}

/// Evaluate `f` on `dev` and on INTERP, returning both host vectors.
fn both(ctx: &Ctx, dev: Option<&str>, f: impl Fn(&Ctx, Option<&str>) -> Tensor) -> (Vec<f64>, Vec<f64>) {
    let mut cache = ProgCache::default();
    let t = f(ctx, dev);
    let got = tensor_values(&mut ctx.raw(), t.uop(), &mut BufTable::default(), &mut cache);
    let r = f(ctx, None);
    let want = tensor_values(&mut ctx.raw(), r.uop(), &mut BufTable::default(), &mut cache);
    (got, want)
}

fn check(name: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let tol = 1e-4 * w.abs().max(1.0);
        assert!(
            (g - w).abs() <= tol,
            "{name}[{i}]: VK {g} vs INTERP {w}"
        );
    }
}

fn fx(ctx: &Ctx, dev: Option<&str>) -> Tensor {
    Tensor::from_slice(
        ctx,
        &[-2.0f32, -1.0, 0.5, 1.0, 2.0, 3.5, 4.0, 10.0],
        None,
        dev,
    )
}

#[test]
fn vk_matches_interp_elementwise() {
    if !have_vk() {
        return;
    }
    let ctx = Ctx::new();
    type Case = (&'static str, fn(&Ctx, Option<&str>) -> Tensor);
    let cases: Vec<Case> = vec![
        ("add", |c, d| fx(c, d).add(2.0)),
        ("mul_sub", |c, d| fx(c, d).mul(3.0).sub(0.5)),
        ("sqrt", |c, d| fx(c, d).add(10.0).sqrt()),
        ("exp2", |c, d| fx(c, d).exp2()),
        ("log2", |c, d| fx(c, d).add(10.0).log2()),
        ("recip", |c, d| fx(c, d).reciprocal()),
        ("neg", |c, d| fx(c, d).neg()),
        ("cmp_where", |c, d| fx(c, d).lt(0.0).where_(fx(c, d), fx(c, d).neg())),
        ("max", |c, d| fx(c, d).maximum(1.0)),
        ("sum", |c, d| fx(c, d).sum(None, false, None)),
        ("amax", |c, d| fx(c, d).max(None, false)),
        (
            "softmax",
            |c, d| autorust::tensor::nn::softmax(&fx(c, d).reshape([2, 4]), Some(1)),
        ),
        ("gelu", |c, d| autorust::tensor::nn::gelu(&fx(c, d))),
        (
            "normalize",
            |c, d| {
                // (x - mean) / sqrt(var + eps) with plain tensor ops (no params,
                // so this stays device-agnostic).
                let x = fx(c, d).reshape([2, 4]);
                let m = x.mean(Some(&[1usize]), true);
                let d2 = x.sub(&m);
                d2.mul(&d2).mean(Some(&[1usize]), true).add(1e-5).sqrt()
            },
        ),
    ];
    for (name, f) in cases {
        let (got, want) = both(&ctx, Some("VK"), f);
        check(name, &got, &want);
    }
}

#[test]
fn vk_matches_interp_casts() {
    if !have_vk() {
        return;
    }
    let ctx = Ctx::new();
    for dt in [dtype::UINT8, dtype::INT8, dtype::INT32, dtype::UINT32] {
        let mut cache = ProgCache::default();
        let t = {
            let x = fx(&ctx, Some("VK"));
            let u = ctx.raw().cast(x.uop(), dt);
            Tensor::from_uop_ctx(&ctx, u)
        };
        let got = tensor_values(&mut ctx.raw(), t.uop(), &mut BufTable::default(), &mut cache);
        let r = {
            let x = fx(&ctx, None);
            let u = ctx.raw().cast(x.uop(), dt);
            Tensor::from_uop_ctx(&ctx, u)
        };
        let want = tensor_values(&mut ctx.raw(), r.uop(), &mut BufTable::default(), &mut cache);
        check(&format!("cast_{}", dt.attr()), &got, &want);
    }
}

#[test]
fn vk_matches_interp_f64_and_matmul() {
    if !have_vk() {
        return;
    }
    if !autorust::runtime::ops_vk::supported_dtype(dtype::FLOAT64) {
        return;
    }
    let ctx = Ctx::new();
    // f64 storage + arithmetic.
    let (got, want) = both(&ctx, Some("VK"), |c, d| {
        Tensor::from_slice(
            c,
            &[-2.0f64, -1.0, 0.5, 1.0, 2.0, 3.5, 4.0, 10.0],
            Some(dtype::FLOAT64),
            d,
        )
        .mul(2.0)
        .add(1.0)
    });
    check("f64_ew", &got, &want);

    // f32 -> f64 -> f32 round trip.
    let (got, want) =
        both(&ctx, Some("VK"), |c, d| fx(c, d).cast(dtype::FLOAT64).cast(dtype::FLOAT32));
    check("f64_roundtrip", &got, &want);

    let n = 256usize;
    let a: Vec<f32> = (0..n * n).map(|i| (i % 7) as f32 * 0.5).collect();
    let a2 = a.clone();
    let (got, want) = both(&ctx, Some("VK"), move |c, d| {
        let at = Tensor::from_slice(c, &a2, None, d).reshape([n as i64, n as i64]);
        at.dot(&at.transpose(0, 1))
    });
    check("matmul", &got, &want);
}

/// Device selection: VK sits after the vendor GPU API, and an explicit name
/// still resolves; a machine without Vulkan keeps the old priority.
#[test]
fn vk_is_a_fallback_after_cl() {
    use autorust::tensor::creation::{pick_device_with_gpus, pick_device_with_metal};
    assert_eq!(pick_device_with_gpus(false, false, true, false, true, true), "VK");
    assert_eq!(pick_device_with_gpus(false, false, true, true, true, true), "CL");
    assert_eq!(pick_device_with_gpus(true, false, true, true, true, true), "METAL");
    assert_eq!(pick_device_with_gpus(false, true, true, true, true, true), "CUDA");
    // The old entry point keeps its behaviour (no Vulkan candidate).
    assert_eq!(pick_device_with_metal(false, false, false, true, false), "RUST");
    if have_vk() {
        assert!(
            autorust::runtime::device::registered_devices()
                .iter()
                .any(|d| d == "VK"),
            "VK should be registered when a device is usable"
        );
    }
}

/// Native `f16` storage + arithmetic, when the device advertises it.
#[test]
fn vk_f16_round_trip() {
    if !have_vk() || !autorust::runtime::ops_vk::supported_dtype(dtype::FLOAT16) {
        return;
    }
    let ctx = Ctx::new();
    let (got, want) = both(&ctx, Some("VK"), |c, d| {
        Tensor::from_slice(c, &[1.5f32, -2.25, 0.5, 8.0], None, d)
            .cast(dtype::FLOAT16)
            .mul(2.0)
            .add(1.0)
            .cast(dtype::FLOAT32)
    });
    check("f16", &got, &want);
}

/// A bf16 matmul with an f32 accumulator: registers/casts/widening must keep the
/// value correct (the first cut of the register-array promotion dropped the
/// loop-carried accumulator, which a render-only test cannot see).
///
/// A bf16 `dot` may run on the cooperative matrix, which accumulates in f32 while
/// the interpreter rounds every partial sum to bf16 — so the tolerance is a few
/// 1e-3 relative, not exact. The bug this pins was a dropped K step, which is
/// orders of magnitude larger.
#[test]
fn vk_bf16_matmul_matches_interp() {
    if !have_vk() {
        return;
    }
    let ctx = Ctx::new();
    let n = 64i64;
    let data: Vec<f32> = (0..(n * n)).map(|i| ((i % 17) as f32 - 8.0) * 0.125).collect();
    let d2 = data.clone();
    let (got, want) = both(&ctx, Some("VK"), move |c, d| {
        let a = Tensor::from_slice(c, &d2, None, d).reshape([n, n]).cast(dtype::BFLOAT16);
        let b = Tensor::from_slice(c, &data, None, d).reshape([n, n]).cast(dtype::BFLOAT16);
        a.dot(&b).cast(dtype::FLOAT32)
    });
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-2 * w.abs().max(1.0),
            "bf16_matmul[{i}]: VK {g} vs INTERP {w}"
        );
    }
}

/// A loop-carried accumulator must survive the *last* iteration. The Vulkan
/// emitter promotes register-array accumulators to `OpPhi`s, and the loop exit
/// branch is taken from the continue block, so reading the header phi after the
/// loop silently dropped the final chunk (a 64-element sum returned 24).
#[test]
fn vk_reduce_covers_the_last_chunk() {
    if !have_vk() {
        return;
    }
    let ctx = Ctx::new();
    // Extents that are *not* a multiple of the unroll factor matter as much as
    // the divisible ones: the emitter runs `n / 8` full chunks and then emits the
    // remainder as straight-line copies, and reading a partial chunk instead
    // would silently pick up neighbouring elements.
    for n in [16i64, 17, 64, 100, 128, 197, 1000, 1024] {
        let data: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
        let d2 = data.clone();
        let (got, want) = both(&ctx, Some("VK"), move |c, d| {
            Tensor::from_slice(c, &d2, None, d).sum(Some(&[0usize]), false, None)
        });
        check(&format!("sum{n}"), &got, &want);
    }
}
