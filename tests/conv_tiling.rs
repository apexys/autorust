//! Differential: the GPU direct-convolution tiling (LOCAL on `cout`, UPCAST on
//! the two spatial axes) must match the reference interpreter.
//!
//! The tinygrad goldens in `tests/nn_conv.rs` only cover small convolutions
//! (`cout < 8`), which `try_conv_tiling` declines, so this pins the tiled path.

use autorust::tensor::{Ctx, Tensor};

fn run(dev: &str) -> Vec<f32> {
    let ctx = Ctx::new();
    let mut sess = autorust::engine::session::Session::new();
    let (b, cin, cout, h, w, k) = (2i64, 8i64, 32i64, 20i64, 20i64, 5i64);
    let xd: Vec<f32> = (0..b * cin * h * w)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.01)
        .collect();
    let wd: Vec<f32> = (0..cout * cin * k * k)
        .map(|i| ((i * 53 % 97) as f32 - 48.0) * 0.005)
        .collect();
    let bd: Vec<f32> = (0..cout).map(|i| i as f32 * 0.01).collect();
    let x = Tensor::from_slice(&ctx, &xd, None, Some(dev)).reshape([b, cin, h, w]);
    let wt = Tensor::from_slice(&ctx, &wd, None, Some(dev)).reshape([cout, cin, k, k]);
    let bi = Tensor::from_slice(&ctx, &bd, None, Some(dev)).reshape([cout]);
    let out = x.conv2d(&wt, Some(&bi), 1, 2, 1, &[2], None);
    let mut got = out.to_list::<f32>(&mut sess);
    got.truncate(4096);
    got
}

#[test]
fn conv_gpu_tiling_matches_interp() {
    if !autorust::runtime::ops_cuda::available() {
        return;
    }
    let want = run("INTERP");
    let got = run("CUDA");
    assert_eq!(got.len(), want.len());
    let maxd = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(maxd < 1e-4, "conv CUDA vs INTERP maxabs {maxd:e}");
}
