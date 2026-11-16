//! Linear/cubic resize with explicit coordinate transforms and antialiasing,
//! checked against an independent host implementation of the same kernels.

use autorust::engine::session::Session;
use autorust::tensor::{Ctx, Tensor};

fn coord(mode: &str, i: f64, in_sz: f64, out_sz: f64) -> f64 {
    let scale = out_sz / in_sz;
    match mode {
        "half_pixel" => (i + 0.5) / scale - 0.5,
        "align_corners" => {
            if out_sz > 1.0 {
                i * (in_sz - 1.0) / (out_sz - 1.0)
            } else {
                0.0
            }
        }
        "asymmetric" => i / scale,
        "pytorch_half_pixel" => {
            if out_sz > 1.0 {
                (i + 0.5) / scale - 0.5
            } else {
                0.0
            }
        }
        "half_pixel_symmetric" => {
            let o = in_sz * scale;
            (in_sz / 2.0) * (1.0 - out_sz / o) + (i + 0.5) / scale - 0.5
        }
        "tf_half_pixel_for_nn" => (i + 0.5) / scale,
        other => panic!("bad coord {other}"),
    }
}

fn cubic_w(t: f64, a: f64) -> f64 {
    let t = t.abs();
    if t <= 1.0 {
        (a + 2.0) * t.powi(3) - (a + 3.0) * t.powi(2) + 1.0
    } else if t < 2.0 {
        a * t.powi(3) - 5.0 * a * t.powi(2) + 8.0 * a * t - 4.0 * a
    } else {
        0.0
    }
}

fn reference(input: &[f32], out: usize, mode: &str, cubic: bool, antialias: bool) -> Vec<f32> {
    let in_sz = input.len() as f64;
    let os = out as f64;
    let scale = os / in_sz;
    let aa = antialias && scale < 1.0;
    let radius = if cubic { 2.0 } else { 1.0 };
    let support = if aa {
        (radius / scale).ceil() as i64
    } else {
        radius as i64
    };
    let ntaps = 2 * support + 1;
    let mut res = vec![0.0f32; out];
    for (i, r) in res.iter_mut().enumerate() {
        let xc = coord(mode, i as f64, in_sz, os);
        let fl = xc.floor();
        let (mut acc, mut wsum) = (0.0f64, 0.0f64);
        for j in 0..ntaps {
            let tap = fl as i64 - support + j;
            let d = if aa {
                (tap as f64 - xc) * scale
            } else {
                tap as f64 - xc
            };
            let w = if cubic { cubic_w(d, -0.75) } else { (1.0 - d.abs()).max(0.0) };
            let ti = tap.clamp(0, input.len() as i64 - 1) as usize;
            acc += w * input[ti] as f64;
            wsum += w;
        }
        *r = (acc / wsum) as f32;
    }
    res
}

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
fn linear_asymmetric() {
    let ctx = Ctx::new();
    let data = [0.0f32, 1.0, 2.0, 3.0];
    let x = Tensor::from_slice(&ctx, &data, None, None).reshape([1i64, 4]);
    close(
        &vals(&x.interpolate_linear_coord(&[8], "asymmetric", false)),
        &reference(&data, 8, "asymmetric", false, false),
        1e-5,
    );
}

#[test]
fn linear_half_pixel_symmetric() {
    let ctx = Ctx::new();
    let data = [0.0f32, 1.0, 2.0, 3.0];
    let x = Tensor::from_slice(&ctx, &data, None, None).reshape([1i64, 4]);
    close(
        &vals(&x.interpolate_linear_coord(&[8], "half_pixel_symmetric", false)),
        &reference(&data, 8, "half_pixel_symmetric", false, false),
        1e-5,
    );
}

#[test]
fn linear_antialias_downscale() {
    let ctx = Ctx::new();
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x = Tensor::from_slice(&ctx, &data, None, None).reshape([1i64, 8]);
    close(
        &vals(&x.interpolate_linear_coord(&[4], "half_pixel", true)),
        &reference(&data, 4, "half_pixel", false, true),
        1e-5,
    );
}

#[test]
fn cubic_antialias_downscale() {
    let ctx = Ctx::new();
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x = Tensor::from_slice(&ctx, &data, None, None).reshape([1i64, 8]);
    close(
        &vals(&x.interpolate_cubic_coord(&[4], "half_pixel", -0.75, false, true)),
        &reference(&data, 4, "half_pixel", true, true),
        1e-5,
    );
}

#[test]
fn linear_tf_half_pixel_for_nn() {
    let ctx = Ctx::new();
    let data = [0.0f32, 1.0, 2.0, 3.0];
    let x = Tensor::from_slice(&ctx, &data, None, None).reshape([1i64, 4]);
    close(
        &vals(&x.interpolate_linear_coord(&[8], "tf_half_pixel_for_nn", false)),
        &reference(&data, 8, "tf_half_pixel_for_nn", false, false),
        1e-5,
    );
}
