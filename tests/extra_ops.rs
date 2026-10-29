//! Long-tail Tensor ops implemented as compositions (silu/erf/normalize/…).

use autorust::engine::session::Session;
use autorust::tensor::{Ctx, Tensor};

fn f32s(ctx: &Ctx, data: &[f32], shape: &[i64]) -> Tensor {
    Tensor::from_slice(ctx, data, None, None).reshape(shape)
}

fn close(got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len());
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert!((a - b).abs() <= tol, "[{i}] {a} != {b}");
    }
}

#[test]
fn activations_match_reference_formulas() {
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let x = f32s(&ctx, &[-3.0, -1.0, 0.0, 1.0, 3.0, 9.0], &[6]);
    close(
        &x.silu().to_list::<f32>(&mut sess),
        &[-0.142277, -0.268941, 0.0, 0.731059, 2.857722, 8.99889],
        1e-4,
    );
    close(
        &x.leaky_relu(0.1).to_list::<f32>(&mut sess),
        &[-0.3, -0.1, 0.0, 1.0, 3.0, 9.0],
        1e-6,
    );
    close(
        &x.relu6().to_list::<f32>(&mut sess),
        &[0.0, 0.0, 0.0, 1.0, 3.0, 6.0],
        1e-6,
    );
    close(
        &x.hardtanh(-1.0, 1.0).to_list::<f32>(&mut sess),
        &[-1.0, -1.0, 0.0, 1.0, 1.0, 1.0],
        1e-6,
    );
    close(
        &x.softsign().to_list::<f32>(&mut sess),
        &[-0.75, -0.5, 0.0, 0.5, 0.75, 0.9],
        1e-6,
    );
}

#[test]
fn erf_and_round_match_tinygrad() {
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let x = f32s(&ctx, &[-1.0, 0.0, 1.0, 2.0], &[4]);
    close(
        &x.erf().to_list::<f32>(&mut sess),
        &[-0.842701, 0.0, 0.842701, 0.995322],
        2e-4,
    );

    let r = f32s(&ctx, &[-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5], &[8]);
    close(
        &r.round().to_list::<f32>(&mut sess),
        &[-4.0, -2.0, -2.0, 0.0, 0.0, 2.0, 2.0, 4.0],
        1e-6,
    );
    close(
        &r.sign().to_list::<f32>(&mut sess),
        &[-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
        1e-6,
    );
}

#[test]
fn reductions_and_normalization() {
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let x = f32s(&ctx, &[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    close(&x.min(None, false).to_list::<f32>(&mut sess), &[1.0], 1e-6);
    close(
        &x.max(Some(&[0]), false).to_list::<f32>(&mut sess),
        &[3.0, 4.0],
        1e-6,
    );
    assert_eq!(x.argmin(None, false).to_list::<i32>(&mut sess), vec![0]);
    close(
        &x.std(None, false, 1).to_list::<f32>(&mut sess),
        &[1.290994],
        1e-5,
    );
    close(
        &x.logsumexp(None, false).to_list::<f32>(&mut sess),
        &[4.440189],
        1e-5,
    );
    let n = x.normalize(2.0, 1, 1e-12);
    let norm: f32 = n
        .to_list::<f32>(&mut sess)
        .iter()
        .map(|v| v * v)
        .sum::<f32>();
    // Two rows, each unit length -> total squared norm 2.
    assert!((norm - 2.0).abs() < 1e-4, "squared norm {norm}");
    assert_eq!(x.numel(), 4);
    assert_eq!(x.size(1), 2);
    assert_eq!(x.nbytes(), 16);
    assert_eq!(x.zeros_like().to_list::<f32>(&mut sess), vec![0.0; 4]);
}

#[test]
fn cumprod_logaddexp_and_predicates() {
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let x = f32s(&ctx, &[1.0, 2.0, 3.0], &[3]);
    close(
        &x.cumprod(0).to_list::<f32>(&mut sess),
        &[1.0, 2.0, 6.0],
        1e-6,
    );
    let a = f32s(&ctx, &[0.0, 1.0], &[2]);
    let b = f32s(&ctx, &[0.0, 2.0], &[2]);
    close(
        &a.logaddexp(&b).to_list::<f32>(&mut sess),
        &[0.693147, 2.313262],
        1e-5,
    );
    assert_eq!(a.isnan().to_list::<bool>(&mut sess), vec![false, false]);
    assert_eq!(a.isinf().to_list::<bool>(&mut sess), vec![false, false]);
    assert_eq!(a.isfinite().to_list::<bool>(&mut sess), vec![true, true]);
    assert!(a.allclose(&a, 1e-5, 1e-8));
}

#[test]
fn multivariate_ops_respect_axes() {
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let x = f32s(&ctx, &[1.0, -2.0, 3.0, -4.0], &[2, 2]);
    close(
        &x.min(Some(&[1]), true).to_list::<f32>(&mut sess),
        &[-2.0, -4.0],
        1e-6,
    );
    close(
        &x.abs()
            .sum(Some(&[1]), false, None)
            .to_list::<f32>(&mut sess),
        &[3.0, 7.0],
        1e-6,
    );
    close(
        &x.tan().to_list::<f32>(&mut sess),
        &[1.557408, 2.185040, -0.142547, -1.157821],
        1e-5,
    );
}

#[test]
fn sort_argsort_and_topk_match_tinygrad() {
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let x = f32s(&ctx, &[0.1, 0.5, 1.2, 3.4, 2.1], &[5]);
    let (v, i) = x.sort(-1, false);
    close(
        &v.to_list::<f32>(&mut sess),
        &[0.1, 0.5, 1.2, 2.1, 3.4],
        1e-6,
    );
    assert_eq!(i.to_list::<i32>(&mut sess), vec![0, 1, 2, 4, 3]);
    let (d, di) = x.sort(-1, true);
    close(
        &d.to_list::<f32>(&mut sess),
        &[3.4, 2.1, 1.2, 0.5, 0.1],
        1e-6,
    );
    assert_eq!(di.to_list::<i32>(&mut sess), vec![3, 4, 2, 1, 0]);
    assert_eq!(
        x.argsort(-1, false).to_list::<i32>(&mut sess),
        vec![0, 1, 2, 4, 3]
    );
    let (tv, ti) = x.topk(2, -1, true);
    close(&tv.to_list::<f32>(&mut sess), &[3.4, 2.1], 1e-6);
    assert_eq!(ti.to_list::<i32>(&mut sess), vec![3, 4]);
}

#[test]
fn avg_pool2d_matches_tinygrad() {
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let x = Tensor::arange(&ctx, 0.0, Some(25.0), 1.0, Some(autorust::dtype::FLOAT32))
        .reshape([1, 1, 5, 5]);
    let y = x.avg_pool2d(&[2, 2], None, 1, &[], false, true);
    assert_eq!(y.shape().as_slice().len(), 4);
    assert_eq!(y.to_list::<f32>(&mut sess), vec![3.0, 5.0, 13.0, 15.0]);
    let yp = x.avg_pool2d(&[2, 2], None, 1, &[1], false, true);
    assert_eq!(
        yp.to_list::<f32>(&mut sess),
        vec![0.0, 0.75, 1.75, 3.75, 9.0, 11.0, 8.75, 19.0, 21.0]
    );
    let yc = x.avg_pool2d(&[2, 2], None, 1, &[1], false, false);
    assert_eq!(
        yc.to_list::<f32>(&mut sess),
        vec![0.0, 1.5, 3.5, 7.5, 9.0, 11.0, 17.5, 19.0, 21.0]
    );
}

#[test]
fn roll_diag_meshgrid_and_masked_fill() {
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let x = f32s(&ctx, &[1.0, 2.0, 3.0, 4.0, 5.0], &[5]);
    close(
        &x.roll(2, Some(0)).to_list::<f32>(&mut sess),
        &[4.0, 5.0, 1.0, 2.0, 3.0],
        1e-6,
    );
    let d = f32s(&ctx, &[1.0, 2.0, 3.0], &[3]).diag();
    assert_eq!(d.shape().as_slice().len(), 2);
    close(
        &d.to_list::<f32>(&mut sess),
        &[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0],
        1e-6,
    );

    let xs = f32s(&ctx, &[1.0, 2.0, 3.0], &[3]);
    let ys = f32s(&ctx, &[4.0, 5.0], &[2]);
    let g = xs.meshgrid(&[ys], "ij");
    close(
        &g[0].to_list::<f32>(&mut sess),
        &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        1e-6,
    );
    close(
        &g[1].to_list::<f32>(&mut sess),
        &[4.0, 5.0, 4.0, 5.0, 4.0, 5.0],
        1e-6,
    );

    let m = Tensor::from_slice(&ctx, &[true, false, true], None, None);
    let v = f32s(&ctx, &[1.0, 2.0, 3.0], &[3]).masked_fill(&m, -1.0);
    close(&v.to_list::<f32>(&mut sess), &[-1.0, 2.0, -1.0], 1e-6);
}

#[test]
fn einsum_common_patterns() {
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let a = f32s(&ctx, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = f32s(&ctx, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);
    let mm = a.einsum(&b, "ij,jk->ik");
    close(
        &mm.to_list::<f32>(&mut sess),
        &[58.0, 64.0, 139.0, 154.0],
        1e-5,
    );
    let bias = f32s(&ctx, &[10.0, 20.0, 30.0], &[3]);
    let ab = a.einsum(&bias, "ij,j->i");
    close(&ab.to_list::<f32>(&mut sess), &[140.0, 320.0], 1e-5);
    let x = f32s(&ctx, &[1.0, 2.0, 3.0], &[3]);
    let y = f32s(&ctx, &[4.0, 5.0, 6.0], &[3]);
    let dot = x.einsum(&y, "i,i->");
    close(&dot.to_list::<f32>(&mut sess), &[32.0], 1e-5);
}
