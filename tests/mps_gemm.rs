//! MetalPerformanceShaders host GEMM vs the generated Metal kernel.
//!
//! The scheduler attaches a `HostGemm` to a plain strided contraction; with
//! `library_gemm = Always` that runs through MPS, with `Never` through the
//! generated register-tiled kernel. Both must agree. This also pins that the
//! MPS path is actually taken (the diagnostic counter).

#![cfg(all(feature = "backend_metal", target_os = "macos"))]

use autorust::dtype;
use autorust::engine::session::Session;
use autorust::runtime::ops_metal::{MPS_GEMMS, mps_available};
use autorust::settings::{LibraryGemm, LibraryGemmPolicy};
use autorust::tensor::{Ctx, Tensor};
use std::sync::atomic::Ordering;

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

fn gemm(m: i64, n: i64, k: i64, dt: dtype::DTypeId, policy: LibraryGemm, batch: i64) -> Vec<f32> {
    let ctx = Ctx::new();
    ctx.borrow_mut().settings.library_gemm = LibraryGemmPolicy::uniform(policy);
    let mut sess = Session::new();
    let a_shape: Vec<i64> = if batch > 1 {
        vec![batch, m, k]
    } else {
        vec![m, k]
    };
    let b_shape: Vec<i64> = if batch > 1 {
        vec![batch, k, n]
    } else {
        vec![k, n]
    };
    let a = Tensor::from_slice(
        &ctx,
        &rnd((batch * m * k) as usize, 7),
        Some(dt),
        Some("METAL"),
    )
    .reshape(a_shape);
    let b = Tensor::from_slice(
        &ctx,
        &rnd((batch * k * n) as usize, 11),
        Some(dt),
        Some("METAL"),
    )
    .reshape(b_shape);
    // `from_slice` stages f32 and casts to f16/bf16; materialize the cast so
    // the scheduler sees a native typed buffer (a library GEMM cannot read the
    // casted staging buffer).
    let (a, b) = if dt == dtype::FLOAT32 {
        (a, b)
    } else {
        (a.to_param(), b.to_param())
    };
    let y = a.matmul(b);
    y.to_list::<f32>(&mut sess)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn mps_is_available_and_used() {
    assert!(
        mps_available(),
        "MetalPerformanceShaders should be available"
    );
    let before = MPS_GEMMS.load(Ordering::Relaxed);
    let _ = gemm(64, 64, 64, dtype::FLOAT32, LibraryGemm::Always, 1);
    let after = MPS_GEMMS.load(Ordering::Relaxed);
    assert!(
        after > before,
        "expected the MPS host GEMM path to run (before={before}, after={after})"
    );
}

#[test]
fn mps_matches_generated_kernel() {
    if !mps_available() {
        return;
    }
    // f16 inputs with an f32 accumulator are supported by MPS (mixed
    // descriptor types); bf16 is rejected and stays on the generated kernel.
    for (dt, tol) in [(dtype::FLOAT32, 2e-3f32), (dtype::FLOAT16, 1e-2)] {
        for &(m, n, k) in &[(64i64, 64, 64), (128, 96, 64), (32, 32, 32)] {
            let before = MPS_GEMMS.load(Ordering::Relaxed);
            let mps = gemm(m, n, k, dt, LibraryGemm::Always, 1);
            let after = MPS_GEMMS.load(Ordering::Relaxed);
            assert!(after > before, "{dt:?} {m}x{n}x{k}: MPS path not used");
            let native_before = after;
            let native = gemm(m, n, k, dt, LibraryGemm::Never, 1);
            assert_eq!(
                MPS_GEMMS.load(Ordering::Relaxed),
                native_before,
                "LibraryGemm::Never must not call MPS"
            );
            let scale = native.iter().fold(1e-6f32, |a, x| a.max(x.abs()));
            let diff = max_abs_diff(&mps, &native);
            eprintln!("{dt:?} {m}x{n}x{k}: mps vs generated maxabs {diff:e}");
            assert!(
                diff <= tol * scale,
                "{dt:?} {m}x{n}x{k}: maxabs {diff:e} (|y|max {scale:e})"
            );
        }
    }
}

#[test]
fn mps_handles_batched_matmul() {
    if !mps_available() {
        return;
    }
    let before = MPS_GEMMS.load(Ordering::Relaxed);
    let mps = gemm(32, 48, 64, dtype::FLOAT32, LibraryGemm::Always, 2);
    let after = MPS_GEMMS.load(Ordering::Relaxed);
    assert!(
        after - before >= 2,
        "expected one MPS GEMM per batch (delta {})",
        after - before
    );
    let native = gemm(32, 48, 64, dtype::FLOAT32, LibraryGemm::Never, 2);
    let scale = native.iter().fold(1e-6f32, |a, x| a.max(x.abs()));
    let diff = max_abs_diff(&mps, &native);
    eprintln!("batched mps vs generated maxabs {diff:e}");
    assert!(diff <= 2e-3 * scale, "batched maxabs {diff:e}");
}
