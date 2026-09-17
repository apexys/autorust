//! Autograd gradchecks: build a forward graph, run `Tensor::gradient`, and
//! compare the returned gradients against hand-computed analytic values.

use autorust::tensor::nn;
use autorust::tensor::Tensor;
use autorust::uop::ctx::UOpCtx;

fn almost(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "len mismatch: {a:?} vs {b:?}");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let tol = 1e-3f32.max(x.abs() * 1e-3 + y.abs() * 1e-3);
        assert!((x - y).abs() <= tol, "value mismatch at {i}: {x} vs {y}");
    }
}

fn f32s(ctx: &mut UOpCtx, data: &[f32]) -> Tensor {
    Tensor::from_slice(ctx, data, None, None)
}

#[test]
fn grad_sum_square() {
    let mut ctx = UOpCtx::new();
    let x = f32s(&mut ctx, &[1.0, 2.0, 3.0, 4.0]);
    let sq = x.mul(&mut ctx, &x);
    let loss = sq.sum(&mut ctx, None, false, None);
    assert_eq!(loss.to_list_fresh::<f32>(&mut ctx), vec![30.0]);
    let gs = loss.gradient(&mut ctx, &[x], None);
    almost(&gs[0].to_list_fresh::<f32>(&mut ctx), &[2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn grad_linear_affine() {
    let mut ctx = UOpCtx::new();
    let x = f32s(&mut ctx, &[1.0, 2.0, 3.0, 4.0]);
    let lin = x.mul(&mut ctx, 2.0).add(&mut ctx, 3.0);
    let loss = lin.sum(&mut ctx, None, false, None);
    let gs = loss.gradient(&mut ctx, &[x], None);
    almost(&gs[0].to_list_fresh::<f32>(&mut ctx), &[2.0, 2.0, 2.0, 2.0]);
}

#[test]
fn grad_bias_broadcast() {
    let mut ctx = UOpCtx::new();
    let x = f32s(&mut ctx, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).reshape(&mut ctx, [3, 4]);
    let b = f32s(&mut ctx, &[10.0, 20.0, 30.0, 40.0]).reshape(&mut ctx, [1, 4]);
    let out = x.add(&mut ctx, &b);
    let loss = out.sum(&mut ctx, None, false, None);
    let gs = loss.gradient(&mut ctx, &[x, b], None);
    almost(&gs[0].to_list_fresh::<f32>(&mut ctx), &[1.0; 12]);
    almost(&gs[1].to_list_fresh::<f32>(&mut ctx), &[3.0, 3.0, 3.0, 3.0]);
}

#[test]
fn grad_sigmoid() {
    let mut ctx = UOpCtx::new();
    let x = f32s(&mut ctx, &[0.0, 1.0, -1.0, 2.0]);
    let s = nn::sigmoid(&x, &mut ctx);
    let loss = s.sum(&mut ctx, None, false, None);
    let gs = loss.gradient(&mut ctx, &[x], None);
    let sig = |z: f32| 1.0 / (1.0 + (-z).exp());
    let exp: Vec<f32> = [0.0f32, 1.0, -1.0, 2.0].iter().map(|&z| sig(z) * (1.0 - sig(z))).collect();
    almost(&gs[0].to_list_fresh::<f32>(&mut ctx), &exp);
}

#[test]
fn grad_matmul() {
    let mut ctx = UOpCtx::new();
    let a = f32s(&mut ctx, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&mut ctx, [2, 3]);
    let w = f32s(&mut ctx, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]).reshape(&mut ctx, [3, 4]);
    let y = a.dot(&mut ctx, &w);
    let loss = y.sum(&mut ctx, None, false, None);
    let gs = loss.gradient(&mut ctx, &[a, w], None);
    // grad_w[i][j] = sum of column i of a (same across j)
    almost(&gs[1].to_list_fresh::<f32>(&mut ctx), &[5.0, 5.0, 5.0, 5.0, 7.0, 7.0, 7.0, 7.0, 9.0, 9.0, 9.0, 9.0]);
    // grad_a[p][i] = row sum of w[i]
    almost(&gs[0].to_list_fresh::<f32>(&mut ctx), &[6.0, 22.0, 38.0, 6.0, 22.0, 38.0]);
}

#[test]
fn grad_mse_like() {
    let mut ctx = UOpCtx::new();
    let x = f32s(&mut ctx, &[1.0, 2.0, 3.0]);
    let t = f32s(&mut ctx, &[2.0, 2.0, 2.0]);
    // loss = mean((x - t)^2) = mean(x^2 - 4x + 4); d/dx = 2(x - t)/3
    let diff = x.sub(&mut ctx, &t);
    let sq = diff.mul(&mut ctx, &diff);
    let loss = sq.mean(&mut ctx, None, false);
    let gs = loss.gradient(&mut ctx, &[x, t], None);
    almost(&gs[0].to_list_fresh::<f32>(&mut ctx), &[2.0 * (1.0 - 2.0) / 3.0, 0.0, 2.0 * (3.0 - 2.0) / 3.0]);
    almost(&gs[1].to_list_fresh::<f32>(&mut ctx), &[-2.0 * (1.0 - 2.0) / 3.0, 0.0, -2.0 * (3.0 - 2.0) / 3.0]);
}
