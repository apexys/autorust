//! Metal `simdgroup` tensor-core GEMM vs the FFMA generated path.
//!
//! Mirrors `tests/tc_gemm.rs` (CUDA). The table is the v0.14.0 `tc.metal`
//! definition: 8x8x8, f32/f32, f16/f32, f16/f16, bf16/f32, bf16/bf16.

#![cfg(all(feature = "backend_metal", target_os = "macos"))]

use autorust::dtype;
use autorust::engine::session::Session;
use autorust::runtime::device::device;
use autorust::tensor::{Ctx, Tensor};

fn rnd(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn gemm(m: i64, n: i64, k: i64, dt: dtype::DTypeId, tensor_cores: bool) -> Vec<f32> {
    let ctx = Ctx::new();
    {
        let mut c = ctx.borrow_mut();
        c.settings.tensor_cores = tensor_cores;
        c.settings.library_gemm =
            autorust::settings::LibraryGemmPolicy::uniform(autorust::settings::LibraryGemm::Never);
    }
    let mut sess = Session::new();
    let a = Tensor::from_slice(&ctx, &rnd((m * k) as usize, 7), Some(dt), Some("METAL"))
        .reshape([m, k]);
    let b = Tensor::from_slice(&ctx, &rnd((k * n) as usize, 11), Some(dt), Some("METAL"))
        .reshape([k, n]);
    let y = if dt == dtype::FLOAT32 {
        a.dot(&b)
    } else {
        a.dot_dtype(&b, Some(dtype::FLOAT32))
    };
    y.to_list::<f32>(&mut sess)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn metal_has_simdgroup_tensor_cores() {
    let facts = device("METAL").renderer.facts().clone();
    assert!(
        !facts.tensor_cores.is_empty(),
        "Metal renderer should expose simdgroup tensor cores on arm64"
    );
}

#[test]
fn metal_tc_gemm_matches_ffma() {
    if device("METAL").renderer.facts().tensor_cores.is_empty() {
        return;
    }
    for (dt, tol) in [
        (dtype::FLOAT32, 2e-2f32),
        (dtype::FLOAT16, 5e-2),
        (dtype::BFLOAT16, 8e-2),
    ] {
        for &(m, n, k) in &[(64i64, 64, 64), (128, 128, 128), (96, 64, 32)] {
            let tc = gemm(m, n, k, dt, true);
            let ff = gemm(m, n, k, dt, false);
            let scale = ff.iter().fold(1e-6f32, |a, x| a.max(x.abs()));
            let diff = max_abs_diff(&tc, &ff);
            eprintln!("{dt:?} {m}x{n}x{k}: tc vs ffma maxabs {diff:e} (|y|max {scale:e})");
            assert!(
                diff <= tol * scale,
                "{dt:?} {m}x{n}x{k}: maxabs {diff:e} (|y|max {scale:e})"
            );
        }
    }
}
