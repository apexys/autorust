//! CUDA's low-K im2col floor: a 32-channel 3x3 conv (K = 288) uses im2col +
//! cuBLAS on CUDA while the interpreter runs the direct convolution. Compare
//! forward and gradients on the same data.

#![cfg(all(
    feature = "backend_cuda",
    any(target_os = "linux", target_os = "windows")
))]

use autorust::engine::session::Session;
use autorust::tensor::{Ctx, Tensor};

fn cuda_available() -> bool {
    autorust::runtime::ops_cuda::available()
}

fn values(seed: usize, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i + seed) % 29) as f32) * 0.05 - 0.7)
        .collect()
}

fn shapes() -> (i64, i64, i64, i64, i64) {
    // cin=32 -> K=288: below the old 576 floor, above the new CUDA 256.
    (1, 32, 8, 8, 16)
}

fn build(
    dev: &str,
    xv: &[f32],
    wv: &[f32],
    bv: &[f32],
    gv: &[f32],
) -> (Tensor, Tensor, Tensor, Tensor) {
    let (b, cin, h, w, cout) = shapes();
    let ctx = Ctx::new();
    let x = Tensor::from_slice(&ctx, xv, None, Some(dev)).reshape([b, cin, h, w]);
    let weight = Tensor::from_slice(&ctx, wv, None, Some(dev)).reshape([cout, cin, 3, 3]);
    let bias = Tensor::from_slice(&ctx, bv, None, Some(dev));
    let g = Tensor::from_slice(&ctx, gv, None, Some(dev)).reshape([b, cout, h, w]);
    (x, weight, bias, g)
}

#[test]
fn cuda_low_k_conv_matches_interp_forward_and_grad() {
    if !cuda_available() {
        return;
    }
    let (b, cin, h, w, cout) = shapes();
    let xv = values(1, (b * cin * h * w) as usize);
    let wv = values(2, (cout * cin * 3 * 3) as usize);
    let bv = values(3, cout as usize);
    let gv = values(4, (b * cout * h * w) as usize);

    let run = |dev: &str| {
        let (x, weight, bias, g) = build(dev, &xv, &wv, &bv, &gv);
        let y = x.conv2d(&weight, Some(&bias), 1, 1, 1, &[1, 1], None);
        let loss = y.mul(&g).sum(None, false, None);
        let grads = loss.gradient(&[x, weight, bias], None);
        let mut sess = Session::new();
        let yv = y.to_list::<f32>(&mut sess);
        let dx = grads[0].to_list::<f32>(&mut sess);
        let dw = grads[1].to_list::<f32>(&mut sess);
        let db = grads[2].to_list::<f32>(&mut sess);
        (yv, dx, dw, db)
    };

    let cpu = run("INTERP");
    let cuda = run("CUDA");
    let max = |a: &[f32], b: &[f32]| {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };
    assert!(
        max(&cpu.0, &cuda.0) < 2e-5,
        "forward {:e}",
        max(&cpu.0, &cuda.0)
    );
    assert!(max(&cpu.1, &cuda.1) < 2e-5, "dx {:e}", max(&cpu.1, &cuda.1));
    assert!(max(&cpu.2, &cuda.2) < 2e-5, "dw {:e}", max(&cpu.2, &cuda.2));
    assert!(max(&cpu.3, &cuda.3) < 2e-5, "db {:e}", max(&cpu.3, &cuda.3));
}
