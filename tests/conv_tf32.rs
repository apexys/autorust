//! Opt-in conv-only TF32: the cuBLAS im2col path uses tensor-core math for
//! batched f32 conv GEMMs. Strict f32 remains the default; this checks the
//! opt-in forward against the INTERP reference with a TF32-appropriate
//! tolerance.

#![cfg(all(
    feature = "backend_cuda",
    any(target_os = "linux", target_os = "windows")
))]

use autorust::engine::session::Session;
use autorust::tensor::{Ctx, Tensor};

static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn values(seed: usize, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i + seed) % 31) as f32) * 0.03 - 0.4)
        .collect()
}

#[test]
fn conv_tf32_matches_interp_with_tf32_tolerance() {
    let _guard = LOCK.lock().unwrap();
    if !autorust::runtime::ops_cuda::available() || !autorust::runtime::cublas::available() {
        return;
    }
    // 32x32 spatial keeps the im2col host GEMM's n at 32, so the conv-only
    // TF32 gate (batch > 1, n >= 28) applies.
    let (b, cin, h, w, cout) = (1i64, 32i64, 32i64, 32i64, 16i64);
    let xv = values(1, (b * cin * h * w) as usize);
    let wv = values(2, (cout * cin * 3 * 3) as usize);
    let bv = values(3, cout as usize);

    let run = |dev: &str, sess: &mut Session| {
        let ctx = Ctx::new();
        let x = Tensor::from_slice(&ctx, &xv, None, Some(dev)).reshape([b, cin, h, w]);
        let weight = Tensor::from_slice(&ctx, &wv, None, Some(dev)).reshape([cout, cin, 3, 3]);
        let bias = Tensor::from_slice(&ctx, &bv, None, Some(dev));
        x.conv2d(&weight, Some(&bias), 1, 1, 1, &[1, 1], None)
            .to_list::<f32>(sess)
    };

    let mut s1 = Session::new();
    let reference = run("INTERP", &mut s1);
    let mut s2 = Session::new();
    let strict = run("CUDA", &mut s2);
    let max = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };
    assert!(
        max(&reference, &strict) < 2e-5,
        "strict f32 {:e}",
        max(&reference, &strict)
    );

    // The flag is read at host-GEMM dispatch.
    unsafe { std::env::set_var("AUTORUST_CONV_TF32", "1") };
    let mut s3 = Session::new();
    let tf32 = run("CUDA", &mut s3);
    unsafe { std::env::remove_var("AUTORUST_CONV_TF32") };

    let err = max(&reference, &tf32);
    assert!(err < 2e-2, "TF32 conv differs from INTERP: max abs {err:e}");
    // TF32 is still a useful approximation, not garbage.
    assert!(err > 0.0 || max(&reference, &strict) == 0.0);
}
