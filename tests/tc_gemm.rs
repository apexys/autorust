//! Generated tensor-core GEMM vs the FFMA generated GEMM on random data.
//!
//! Verifies the WMMA lowering (`tensor_cores = true`) against the register-tiled
//! FFMA path (`tensor_cores = false`). f32 uses TF32 (10-bit mantissa), so the
//! tolerance is relative to the output magnitude; bf16 is looser.

#![cfg(all(
    feature = "backend_cuda",
    any(target_os = "linux", target_os = "windows")
))]

use autorust::dtype;
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

fn gemm(m: i64, n: i64, k: i64, dt: dtype::DTypeId, tensor_cores: bool) -> (Vec<f32>, Vec<f32>) {
    let ctx = Ctx::new();
    {
        let mut c = ctx.borrow_mut();
        c.settings.tensor_cores = tensor_cores;
        // Force the generated kernel (Auto would pick cuBLAS for f32).
        c.settings.library_gemm =
            autorust::settings::LibraryGemmPolicy::uniform(autorust::settings::LibraryGemm::Never);
    }
    let mut sess = autorust::engine::session::Session::new();
    let a =
        Tensor::from_slice(&ctx, &rnd((m * k) as usize, 7), Some(dt), Some("CUDA")).reshape([m, k]);
    let b = Tensor::from_slice(&ctx, &rnd((k * n) as usize, 11), Some(dt), Some("CUDA"))
        .reshape([k, n]);
    let av = a.to_list::<f32>(&mut sess);
    let bv = b.to_list::<f32>(&mut sess);
    let y = if dt == dtype::FLOAT32 {
        a.dot(&b)
    } else {
        a.dot_dtype(&b, Some(dtype::FLOAT32))
    };
    (
        y.to_list::<f32>(&mut sess),
        av.iter().chain(&bv).copied().collect(),
    )
}

#[test]
fn generated_tc_gemm_matches_ffma() {
    if !autorust::runtime::ops_cuda::has_tensor_cores() {
        return;
    }
    for &(m, n, k) in &[
        (64i64, 64, 64),
        (128, 128, 128),
        (512, 128, 128),
        (100, 100, 100),
    ] {
        let (tc, _) = gemm(m, n, k, dtype::FLOAT32, true);
        let (ff, _) = gemm(m, n, k, dtype::FLOAT32, false);
        let maxa = ff.iter().fold(0.0f32, |a, x| a.max(x.abs())).max(1e-6);
        let maxd = tc
            .iter()
            .zip(&ff)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("f32 {m}x{n}x{k}: tc vs ffma maxabs {maxd:e} (|y|max {maxa:e})");
        assert!(maxd <= 2e-2 * maxa, "f32 {m}x{n}x{k}: maxabs {maxd:e}");
    }
    for &(m, n, k) in &[(64i64, 64, 64), (128, 128, 128)] {
        let (tc, _) = gemm(m, n, k, dtype::BFLOAT16, true);
        let (ff, _) = gemm(m, n, k, dtype::BFLOAT16, false);
        let maxa = ff.iter().fold(0.0f32, |a, x| a.max(x.abs())).max(1e-6);
        let maxd = tc
            .iter()
            .zip(&ff)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("bf16 {m}x{n}x{k}: tc vs ffma maxabs {maxd:e} (|y|max {maxa:e})");
        assert!(maxd <= 5e-2 * maxa, "bf16 {m}x{n}x{k}: maxabs {maxd:e}");
    }
}
