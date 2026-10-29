//! Autograd gradchecks: build a forward graph, run `Tensor::gradient`, and
//! compare the returned gradients against hand-computed analytic values.

use autorust::tensor::nn;
use autorust::tensor::{Ctx, Tensor};

fn almost(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "len mismatch: {a:?} vs {b:?}");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let tol = 1e-3f32.max(x.abs() * 1e-3 + y.abs() * 1e-3);
        assert!((x - y).abs() <= tol, "value mismatch at {i}: {x} vs {y}");
    }
}

fn f32s(ctx: &Ctx, data: &[f32]) -> Tensor {
    Tensor::from_slice(ctx, data, None, None)
}

#[test]
fn grad_sum_square() {
    let ctx = Ctx::new();
    let x = f32s(&ctx, &[1.0, 2.0, 3.0, 4.0]);
    let sq = x.mul(&x);
    let loss = sq.sum(None, false, None);
    // Compute the gradient before any realize: `realize` replaces the tensor's uop
    // with its realized storage (`_apply_map_to_tensors`), after which the graph no
    // longer carries the forward expression (tinygrad's `loss.backward()` after
    // `loss.realize()` likewise yields no grad).
    let gs = loss.gradient(&[x], None);
    assert_eq!(loss.to_list_fresh::<f32>(), vec![30.0]);
    almost(&gs[0].to_list_fresh::<f32>(), &[2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn grad_linear_affine() {
    let ctx = Ctx::new();
    let x = f32s(&ctx, &[1.0, 2.0, 3.0, 4.0]);
    let lin = x.mul(2.0).add(3.0);
    let loss = lin.sum(None, false, None);
    let gs = loss.gradient(&[x], None);
    almost(&gs[0].to_list_fresh::<f32>(), &[2.0, 2.0, 2.0, 2.0]);
}

#[test]
fn grad_bias_broadcast() {
    let ctx = Ctx::new();
    let x = f32s(
        &ctx,
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    )
    .reshape([3, 4]);
    let b = f32s(&ctx, &[10.0, 20.0, 30.0, 40.0]).reshape([1, 4]);
    let out = x.add(&b);
    let loss = out.sum(None, false, None);
    let gs = loss.gradient(&[x, b], None);
    almost(&gs[0].to_list_fresh::<f32>(), &[1.0; 12]);
    almost(&gs[1].to_list_fresh::<f32>(), &[3.0, 3.0, 3.0, 3.0]);
}

#[test]
fn grad_sigmoid() {
    let ctx = Ctx::new();
    let x = f32s(&ctx, &[0.0, 1.0, -1.0, 2.0]);
    let s = nn::sigmoid(&x);
    let loss = s.sum(None, false, None);
    let gs = loss.gradient(&[x], None);
    let sig = |z: f32| 1.0 / (1.0 + (-z).exp());
    let exp: Vec<f32> = [0.0f32, 1.0, -1.0, 2.0]
        .iter()
        .map(|&z| sig(z) * (1.0 - sig(z)))
        .collect();
    almost(&gs[0].to_list_fresh::<f32>(), &exp);
}

#[test]
fn grad_matmul() {
    let ctx = Ctx::new();
    let a = f32s(&ctx, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape([2, 3]);
    let w = f32s(
        &ctx,
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
    )
    .reshape([3, 4]);
    let y = a.dot(&w);
    let loss = y.sum(None, false, None);
    let gs = loss.gradient(&[a, w], None);
    // grad_w[i][j] = sum of column i of a (same across j)
    almost(
        &gs[1].to_list_fresh::<f32>(),
        &[5.0, 5.0, 5.0, 5.0, 7.0, 7.0, 7.0, 7.0, 9.0, 9.0, 9.0, 9.0],
    );
    // grad_a[p][i] = row sum of w[i]
    almost(
        &gs[0].to_list_fresh::<f32>(),
        &[6.0, 22.0, 38.0, 6.0, 22.0, 38.0],
    );
}

#[test]
fn grad_mse_like() {
    let ctx = Ctx::new();
    let x = f32s(&ctx, &[1.0, 2.0, 3.0]);
    let t = f32s(&ctx, &[2.0, 2.0, 2.0]);
    // loss = mean((x - t)^2) = mean(x^2 - 4x + 4); d/dx = 2(x - t)/3
    let diff = x.sub(&t);
    let sq = diff.mul(&diff);
    let loss = sq.mean(None, false);
    let gs = loss.gradient(&[x, t], None);
    almost(
        &gs[0].to_list_fresh::<f32>(),
        &[2.0 * (1.0 - 2.0) / 3.0, 0.0, 2.0 * (3.0 - 2.0) / 3.0],
    );
    almost(
        &gs[1].to_list_fresh::<f32>(),
        &[-2.0 * (1.0 - 2.0) / 3.0, 0.0, -2.0 * (3.0 - 2.0) / 3.0],
    );
}
