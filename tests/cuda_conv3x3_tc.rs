//! Bounded tensor-core 3x3 conv (`Tensor::conv2d_with`) vs INTERP, the
//! stride-2 fallback, and the graph gradient for the training path.

#![cfg(all(
    feature = "backend_cuda",
    any(target_os = "linux", target_os = "windows")
))]

use autorust::engine::session::Session;
use autorust::tensor::{Ctx, Tensor};

static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn values(seed: usize, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i + seed) % 31) as f32) * 0.04 - 0.6)
        .collect()
}

fn max(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn run(
    dev: &str,
    stride: i64,
    (b, cin, h, w, cout): (i64, i64, i64, i64, i64),
    xv: &[f32],
    wv: &[f32],
    bv: &[f32],
    fused: bool,
) -> Vec<f32> {
    let ctx = Ctx::new();
    let x = Tensor::from_slice(&ctx, xv, None, Some(dev)).reshape([b, cin, h, w]);
    let weight = Tensor::from_slice(&ctx, wv, None, Some(dev)).reshape([cout, cin, 3, 3]);
    let bias = Tensor::from_slice(&ctx, bv, None, Some(dev));
    let mut sess = Session::new();
    if fused {
        x.realize(&mut sess);
        x.conv2d_with(&weight, Some(&bias), 1, stride, 1, &[1, 1], None, &mut sess)
            .to_list::<f32>(&mut sess)
    } else {
        x.conv2d(&weight, Some(&bias), 1, stride, 1, &[1, 1], None)
            .to_list::<f32>(&mut sess)
    }
}

#[test]
fn conv3x3_tc_matches_interp() {
    let _guard = LOCK.lock().unwrap();
    if !autorust::runtime::ops_cuda::available() {
        return;
    }
    // rows = 1*128*128 = 16384, at the kernel's minimum.
    let shape = (1, 8, 128, 128, 16);
    let (b, cin, h, w, cout) = shape;
    let xv = values(1, (b * cin * h * w) as usize);
    let wv = values(2, (cout * cin * 9) as usize);
    let bv = values(3, cout as usize);

    let reference = run("INTERP", 1, shape, &xv, &wv, &bv, false);
    unsafe { std::env::set_var("AUTORUST_CONV3X3_TC", "1") };
    let fused = run("CUDA", 1, shape, &xv, &wv, &bv, true);
    unsafe { std::env::remove_var("AUTORUST_CONV3X3_TC") };
    let err = max(&reference, &fused);
    assert!(err < 3e-2, "TC 3x3 differs from INTERP: max abs {err:e}");
}

#[test]
fn conv3x3_stride_falls_back() {
    let _guard = LOCK.lock().unwrap();
    if !autorust::runtime::ops_cuda::available() {
        return;
    }
    let shape = (1, 4, 16, 16, 6);
    let (b, cin, h, w, cout) = shape;
    let xv = values(4, (b * cin * h * w) as usize);
    let wv = values(5, (cout * cin * 9) as usize);
    let bv = values(6, cout as usize);
    unsafe { std::env::set_var("AUTORUST_CONV3X3_TC", "1") };
    let graph = run("CUDA", 2, shape, &xv, &wv, &bv, false);
    let via_with = run("CUDA", 2, shape, &xv, &wv, &bv, true);
    unsafe { std::env::remove_var("AUTORUST_CONV3X3_TC") };
    assert!(max(&graph, &via_with) < 2e-5);
}

#[test]
fn conv3x3_graph_gradient_matches_interp() {
    let _guard = LOCK.lock().unwrap();
    if !autorust::runtime::ops_cuda::available() {
        return;
    }
    let (b, cin, h, w, cout) = (1i64, 4i64, 8i64, 8i64, 6i64);
    let xv = values(7, (b * cin * h * w) as usize);
    let wv = values(8, (cout * cin * 9) as usize);
    let bv = values(9, cout as usize);
    let gv = values(10, (b * cout * h * w) as usize);

    let grads = |dev: &str| {
        let ctx = Ctx::new();
        let x = Tensor::from_slice(&ctx, &xv, None, Some(dev)).reshape([b, cin, h, w]);
        let weight = Tensor::from_slice(&ctx, &wv, None, Some(dev)).reshape([cout, cin, 3, 3]);
        let bias = Tensor::from_slice(&ctx, &bv, None, Some(dev));
        let g = Tensor::from_slice(&ctx, &gv, None, Some(dev)).reshape([b, cout, h, w]);
        let y = x.conv2d(&weight, Some(&bias), 1, 1, 1, &[1, 1], None);
        let loss = y.mul(&g).sum(None, false, None);
        let grads = loss.gradient(&[x, weight, bias], None);
        let mut sess = Session::new();
        (
            grads[0].to_list::<f32>(&mut sess),
            grads[1].to_list::<f32>(&mut sess),
            grads[2].to_list::<f32>(&mut sess),
        )
    };
    let cpu = grads("INTERP");
    let cuda = grads("CUDA");
    assert!(max(&cpu.0, &cuda.0) < 2e-5, "dx {:e}", max(&cpu.0, &cuda.0));
    assert!(max(&cpu.1, &cuda.1) < 2e-5, "dw {:e}", max(&cpu.1, &cuda.1));
    assert!(max(&cpu.2, &cuda.2) < 2e-5, "db {:e}", max(&cpu.2, &cuda.2));
}
