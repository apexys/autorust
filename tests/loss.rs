//! `Tensor::nll_loss` / `Tensor::softmax_cross_entropy`: the framework-level
//! equivalents of the ONNX loss wrappers.

use autorust::engine::session::Session;
use autorust::tensor::{Ctx, Tensor};

fn t(ctx: &Ctx, data: &[f32], shape: &[i64]) -> Tensor {
    Tensor::from_slice(ctx, data, None, None).reshape(shape.to_vec())
}

fn labels(ctx: &Ctx, data: &[i64], shape: &[i64]) -> Tensor {
    Tensor::from_slice(ctx, data, Some(autorust::dtype::INT64), None).reshape(shape.to_vec())
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

fn log_probs(ctx: &Ctx) -> Tensor {
    t(ctx, &[-1.0, -2.0, -3.0, -0.5, -1.5, -2.5], &[2, 3])
}

#[test]
fn nll_loss_reductions() {
    let ctx = Ctx::new();
    let x = log_probs(&ctx);
    let y = labels(&ctx, &[0, 1], &[2]);
    close(&vals(&x.nll_loss(&y, None, None, "none")), &[1.0, 1.5], 1e-6);
    close(&vals(&x.nll_loss(&y, None, None, "mean")), &[1.25], 1e-6);
    close(&vals(&x.nll_loss(&y, None, None, "sum")), &[2.5], 1e-6);
}

#[test]
fn nll_loss_ignore_index() {
    let ctx = Ctx::new();
    let x = log_probs(&ctx);
    let y = labels(&ctx, &[0, 1], &[2]);
    close(
        &vals(&x.nll_loss(&y, None, Some(1), "none")),
        &[1.0, 0.0],
        1e-6,
    );
    close(&vals(&x.nll_loss(&y, None, Some(1), "mean")), &[1.0], 1e-6);
    close(&vals(&x.nll_loss(&y, None, Some(1), "sum")), &[1.0], 1e-6);
}

#[test]
fn nll_loss_class_weights() {
    let ctx = Ctx::new();
    let x = log_probs(&ctx);
    let y = labels(&ctx, &[0, 1], &[2]);
    let w = t(&ctx, &[2.0, 3.0, 4.0], &[3]);
    close(
        &vals(&x.nll_loss(&y, Some(&w), None, "none")),
        &[2.0, 4.5],
        1e-6,
    );
    // Weighted mean divides by the sum of the selected class weights.
    close(&vals(&x.nll_loss(&y, Some(&w), None, "mean")), &[1.3], 1e-6);
    close(&vals(&x.nll_loss(&y, Some(&w), None, "sum")), &[6.5], 1e-6);
}

#[test]
fn softmax_cross_entropy_from_logits() {
    let ctx = Ctx::new();
    let scores = t(&ctx, &[0.0; 6], &[2, 3]);
    let y = labels(&ctx, &[0, 1], &[2]);
    let ln3 = 3.0f32.ln();
    close(
        &vals(&scores.softmax_cross_entropy(&y, None, None, "none")),
        &[ln3, ln3],
        1e-5,
    );
    close(
        &vals(&scores.softmax_cross_entropy(&y, None, None, "mean")),
        &[ln3],
        1e-5,
    );
    close(
        &vals(&scores.softmax_cross_entropy(&y, None, Some(1), "sum")),
        &[ln3],
        1e-5,
    );
}

#[test]
fn class_axis_is_one_for_higher_rank_inputs() {
    let ctx = Ctx::new();
    // [N=1, C=3, D=2]: log-softmax over axis 1 is -ln(3) everywhere.
    let scores = t(&ctx, &[0.0; 6], &[1, 3, 2]);
    let y = labels(&ctx, &[0, 1], &[1, 2]);
    let ln3 = 3.0f32.ln();
    let got = vals(&scores.softmax_cross_entropy(&y, None, None, "none"));
    close(&got, &[ln3, ln3], 1e-5);
}
