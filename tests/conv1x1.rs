//! The 1x1-conv fast path (plain `[B*H*W, Cin] @ [Cin, Cout]` matmul) vs a
//! direct host sum. The stride-2 case is outside the fast path and exercises
//! the general convolution fallback.

use autorust::engine::session::Session;
use autorust::tensor::{Ctx, Tensor};

fn values(seed: usize, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i + seed) % 19) as f32) * 0.07 - 0.6)
        .collect()
}

fn reference(
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    (b, cin, h, wd, cout, stride): (usize, usize, usize, usize, usize, usize),
) -> Vec<f32> {
    let oh = (h - 1) / stride + 1;
    let ow = (wd - 1) / stride + 1;
    let mut out = vec![0.0f32; b * cout * oh * ow];
    for bi in 0..b {
        for co in 0..cout {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut acc = bias[co];
                    for ci in 0..cin {
                        let xv = x[((bi * cin + ci) * h + oy * stride) * wd + ox * stride];
                        let wv = w[co * cin + ci];
                        acc += xv * wv;
                    }
                    out[((bi * cout + co) * oh + oy) * ow + ox] = acc;
                }
            }
        }
    }
    out
}

fn check(stride: i64, device: Option<&str>) {
    let (b, cin, h, wd, cout) = (2usize, 3usize, 6usize, 6usize, 4usize);
    let xv = values(1, b * cin * h * wd);
    let wv = values(2, cout * cin);
    let bv = values(3, cout);
    let ctx = Ctx::new();
    let x = Tensor::from_slice(&ctx, &xv, None, device)
        .reshape([b as i64, cin as i64, h as i64, wd as i64]);
    let w = Tensor::from_slice(&ctx, &wv, None, device).reshape([cout as i64, cin as i64, 1, 1]);
    let bias = Tensor::from_slice(&ctx, &bv, None, device);
    let y = x.conv2d(&w, Some(&bias), 1, stride, 1, &[0, 0], None);
    let mut sess = Session::new();
    let got = y.to_list::<f32>(&mut sess);
    let want = reference(&xv, &wv, &bv, (b, cin, h, wd, cout, stride as usize));
    assert_eq!(got.len(), want.len());
    let max = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max < 2e-5, "1x1 conv stride {stride}: max abs {max:e}");
}

#[test]
fn conv1x1_matches_direct_sum() {
    check(1, None);
}

#[test]
fn conv1x1_stride_falls_back() {
    check(2, None);
}

#[cfg(all(
    feature = "backend_cuda",
    any(target_os = "linux", target_os = "windows")
))]
#[test]
fn conv1x1_cuda_fast_path() {
    if !autorust::runtime::ops_cuda::available() {
        return;
    }
    check(1, Some("CUDA"));
    check(2, Some("CUDA"));
}

#[cfg(all(
    feature = "backend_cuda",
    any(target_os = "linux", target_os = "windows")
))]
#[test]
fn conv1x1_cuda_gradient_matches_direct_sum() {
    if !autorust::runtime::ops_cuda::available() {
        return;
    }
    let (b, cin, h, wd, cout) = (1usize, 2usize, 4usize, 4usize, 3usize);
    let xv = values(11, b * cin * h * wd);
    let wv = values(12, cout * cin);
    let bv = values(13, cout);
    let gv = values(14, b * cout * h * wd);
    let ctx = Ctx::new();
    let x = Tensor::from_slice(&ctx, &xv, None, Some("CUDA"))
        .reshape([b as i64, cin as i64, h as i64, wd as i64]);
    let w =
        Tensor::from_slice(&ctx, &wv, None, Some("CUDA")).reshape([cout as i64, cin as i64, 1, 1]);
    let bias = Tensor::from_slice(&ctx, &bv, None, Some("CUDA"));
    let g = Tensor::from_slice(&ctx, &gv, None, Some("CUDA")).reshape([
        b as i64,
        cout as i64,
        h as i64,
        wd as i64,
    ]);
    let y = x.conv2d(&w, Some(&bias), 1, 1, 1, &[0, 0], None);
    let loss = y.mul(&g).sum(None, false, None);
    let grads = loss.gradient(&[x, w, bias], None);
    let mut sess = Session::new();
    let dx = grads[0].to_list::<f32>(&mut sess);
    let dw = grads[1].to_list::<f32>(&mut sess);
    let db = grads[2].to_list::<f32>(&mut sess);

    let g = |bi: usize, co: usize, yy: usize, xx: usize| gv[((bi * cout + co) * h + yy) * wd + xx];
    let xv_at =
        |bi: usize, ci: usize, yy: usize, xx: usize| xv[((bi * cin + ci) * h + yy) * wd + xx];
    let mut want_dx = vec![0.0f32; xv.len()];
    let mut want_dw = vec![0.0f32; wv.len()];
    let mut want_db = vec![0.0f32; bv.len()];
    for bi in 0..b {
        for ci in 0..cin {
            for yy in 0..h {
                for xx in 0..wd {
                    let mut acc = 0.0f32;
                    for co in 0..cout {
                        acc += wv[co * cin + ci] * g(bi, co, yy, xx);
                    }
                    want_dx[((bi * cin + ci) * h + yy) * wd + xx] = acc;
                }
            }
        }
    }
    for co in 0..cout {
        for ci in 0..cin {
            let mut acc = 0.0f32;
            for bi in 0..b {
                for yy in 0..h {
                    for xx in 0..wd {
                        acc += g(bi, co, yy, xx) * xv_at(bi, ci, yy, xx);
                    }
                }
            }
            want_dw[co * cin + ci] = acc;
        }
        let mut acc = 0.0f32;
        for bi in 0..b {
            for yy in 0..h {
                for xx in 0..wd {
                    acc += g(bi, co, yy, xx);
                }
            }
        }
        want_db[co] = acc;
    }
    let max = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };
    assert!(max(&dx, &want_dx) < 2e-5, "dx {:e}", max(&dx, &want_dx));
    assert!(max(&dw, &want_dw) < 2e-5, "dw {:e}", max(&dw, &want_dw));
    assert!(max(&db, &want_db) < 2e-5, "db {:e}", max(&db, &want_db));
}
