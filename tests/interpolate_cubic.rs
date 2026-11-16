//! `Tensor::interpolate_cubic`: Keys/Catmull-Rom resize against explicit host
//! values (computed separately with the same kernel formula).

use autorust::engine::session::Session;
use autorust::tensor::{Ctx, Tensor};

fn vals(x: &Tensor) -> Vec<f32> {
    let mut s = Session::new();
    x.to_list::<f32>(&mut s)
}

fn close(got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "len {got:?} vs {want:?}");
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert!((a - b).abs() <= tol, "[{i}] {a} != {b}");
    }
}

#[test]
fn constant_stays_constant() {
    let ctx = Ctx::new();
    let x = Tensor::from_slice(&ctx, &[2.0f32; 16], None, Some("INTERP")).reshape([1i64, 1, 4, 4]);
    close(&vals(&x.interpolate_cubic(&[8, 8], false, -0.75)), &[2.0; 64], 1e-5);
}

#[test]
fn one_dimensional_half_pixel() {
    let ctx = Ctx::new();
    let x = Tensor::from_slice(&ctx, &[0.0f32, 1.0, 2.0, 3.0], None, Some("INTERP"))
        .reshape([1i64, 4]);
    let want = [
        -0.1054688, 0.1914062, 0.6679688, 1.296875, 1.703125, 2.3320312, 2.8085938, 3.1054688,
    ];
    close(&vals(&x.interpolate_cubic(&[8], false, -0.75)), &want, 1e-5);
}

#[test]
fn two_dimensional_half_pixel() {
    let ctx = Ctx::new();
    let x = Tensor::from_slice(&ctx, &[1.0f32, 2.0, 3.0, 4.0], None, Some("INTERP"))
        .reshape([1i64, 1, 2, 2]);
    let want = [
        0.6835938, 1.015625, 1.5625, 1.8945312, 1.3476562, 1.6796875, 2.2265625, 2.5585938,
        2.4414062, 2.7734375, 3.3203125, 3.6523438, 3.1054688, 3.4375, 3.984375, 4.3164062,
    ];
    close(&vals(&x.interpolate_cubic(&[4, 4], false, -0.75)), &want, 1e-5);
}

#[test]
fn align_corners_keeps_the_edges() {
    let ctx = Ctx::new();
    let x = Tensor::from_slice(&ctx, &[0.0f32, 1.0, 2.0, 3.0], None, Some("INTERP"))
        .reshape([1i64, 4]);
    let got = vals(&x.interpolate_cubic(&[8], true, -0.75));
    assert!((got[0] - 0.0).abs() < 1e-5, "left edge {}", got[0]);
    assert!((got[7] - 3.0).abs() < 1e-5, "right edge {}", got[7]);
}
