//! Tensor-level op tests driven through `to_list`/`item`.
//!
//! These exercise the full path: build op graph -> realize -> typed read-out,
//! mirroring the op families in `test/backend/test_ops.py`. Expected values are
//! the tinygrad semantics on small, exactly-representable inputs (transcendentals
//! compared with a relative epsilon).

use autorust::dtype;
use autorust::tensor::nn;
use autorust::tensor::{Ctx, Tensor};
use autorust::uop::ctx::UOpCtx;

fn almost(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "len mismatch: {a:?} vs {b:?}");
    for (x, y) in a.iter().zip(b) {
        let tol = 1e-4f32.max(x.abs() * 1e-4 + y.abs() * 1e-4);
        assert!((x - y).abs() <= tol, "value mismatch: {x} vs {y}");
    }
}

fn f32s(ctx: &Ctx, data: &[f32]) -> Tensor {
    Tensor::from_slice(ctx, data, None, None)
}

#[test]
fn binary_arithmetic() {
    let mut ctx = Ctx::new();
    let a = f32s(&ctx, &[1.0, 2.0, 3.0, 4.0]);
    let b = f32s(&ctx, &[10.0, 20.0, 30.0, 40.0]);
    almost(&a.add(&b).to_list_fresh::<f32>(), &[11.0, 22.0, 33.0, 44.0]);
    almost(&b.sub(&a).to_list_fresh::<f32>(), &[9.0, 18.0, 27.0, 36.0]);
    almost(&a.mul(&b).to_list_fresh::<f32>(), &[10.0, 40.0, 90.0, 160.0]);
    almost(&b.div(&a).to_list_fresh::<f32>(), &[10.0, 10.0, 10.0, 10.0]);
    // scalar (f64) operand broadcasts
    almost(&a.mul(2.0).to_list_fresh::<f32>(), &[2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn unary_pow_log() {
    let mut ctx = Ctx::new();
    let x = f32s(&ctx, &[1.0, 4.0, 9.0, 16.0]);
    almost(&x.sqrt().to_list_fresh::<f32>(), &[1.0, 2.0, 3.0, 4.0]);
    almost(&x.square().to_list_fresh::<f32>(), &[1.0, 16.0, 81.0, 256.0]);
    almost(&x.reciprocal().to_list_fresh::<f32>(), &[1.0, 0.25, 1.0 / 9.0, 1.0 / 16.0]);

    let r = f32s(&ctx, &[0.0, 1.0, 2.0]);
    almost(&r.exp().to_list_fresh::<f32>(), &[1.0, std::f32::consts::E, 7.389056]);
    almost(&r.exp2().to_list_fresh::<f32>(), &[1.0, 2.0, 4.0]);

    let p = f32s(&ctx, &[1.0, 2.0, 4.0, 8.0]);
    almost(
        &p.log().to_list_fresh::<f32>(),
        &[0.0, std::f32::consts::LN_2, 2.0 * std::f32::consts::LN_2, 3.0 * std::f32::consts::LN_2],
    );
    almost(&p.log2().to_list_fresh::<f32>(), &[0.0, 1.0, 2.0, 3.0]);
}

#[test]
fn unary_misc() {
    let mut ctx = Ctx::new();
    let s = f32s(&ctx, &[0.0, 1.0, -1.0]);
    almost(&s.sin().to_list_fresh::<f32>(), &[0.0, 0.841471, -0.841471]);
    almost(&s.tanh().to_list_fresh::<f32>(), &[0.0, 0.7615942, -0.7615942]);

    let n = f32s(&ctx, &[-2.5, -1.1, 0.5, 2.7]);
    almost(&n.trunc().to_list_fresh::<f32>(), &[-2.0, -1.0, 0.0, 2.0]);
    almost(&n.ceil().to_list_fresh::<f32>(), &[-2.0, -1.0, 1.0, 3.0]);
    almost(&n.floor().to_list_fresh::<f32>(), &[-3.0, -2.0, 0.0, 2.0]);
}

#[test]
fn neg_abs_and_bounds() {
    let mut ctx = Ctx::new();
    let x = f32s(&ctx, &[-3.0, 2.0, 0.0]);
    almost(&x.neg().to_list_fresh::<f32>(), &[3.0, -2.0, 0.0]);
    almost(&x.abs().to_list_fresh::<f32>(), &[3.0, 2.0, 0.0]);
    let a = f32s(&ctx, &[1.0, 5.0, 3.0]);
    let b = f32s(&ctx, &[4.0, 2.0, 3.0]);
    almost(&a.maximum(&b).to_list_fresh::<f32>(), &[4.0, 5.0, 3.0]);
    almost(&a.minimum(&b).to_list_fresh::<f32>(), &[1.0, 2.0, 3.0]);
    almost(&a.maximum(2.0).to_list_fresh::<f32>(), &[2.0, 5.0, 3.0]);
}

#[test]
fn reductions() {
    let mut ctx = Ctx::new();
    let x = f32s(&ctx, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    almost(&x.sum(None, false, None).to_list_fresh::<f32>(), &[36.0]);
    almost(&x.mean(None, false).to_list_fresh::<f32>(), &[4.5]);
    almost(&x.prod(None, false, None).to_list_fresh::<f32>(), &[40320.0]);

    let m = x.reshape([2, 4]);
    almost(&m.sum(Some(&[0]), false, None).to_list_fresh::<f32>(), &[6.0, 8.0, 10.0, 12.0]);
    almost(&m.sum(Some(&[1]), false, None).to_list_fresh::<f32>(), &[10.0, 26.0]);

    let i = Tensor::from_slice(&ctx, &[3i32, -7, 0, 12, 5, -2, 8, 1], None, None);
    assert_eq!(i.max(None, false).item_fresh::<i32>(), 12i32);

    almost(&x.cumsum(0).to_list_fresh::<f32>(), &[1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0, 36.0]);
}

#[test]
fn movement_ops() {
    let mut ctx = Ctx::new();
    let x = f32s(&ctx, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    // permute a 2x4 into 4x2
    let p = x.reshape([2, 4]).permute(&[1, 0]);
    almost(&p.to_list_fresh::<f32>(), &[1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0]);
    // transpose of 2x4 == permute(1,0)
    let t = x.reshape([2, 4]).transpose(0, 1);
    almost(&t.to_list_fresh::<f32>(), &[1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0]);
    // flip 1-D
    almost(&x.flip(&[0]).to_list_fresh::<f32>(), &[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
    // flatten round-trips identity
    almost(&x.reshape([2, 4]).flatten(0, 1).to_list_fresh::<f32>(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn matmul_ops() {
    let mut ctx = Ctx::new();
    // (2x3) @ (3x2)
    let a = f32s(&ctx, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape([2, 3]);
    let w = f32s(&ctx, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).reshape([3, 2]);
    // w = [[7,8],[9,10],[11,12]]
    // row0 = [1,2,3] -> [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
    // row1 = [4,5,6] -> [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
    almost(&a.dot(&w).to_list_fresh::<f32>(), &[58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn casts_across_dtypes() {
    let mut ctx = Ctx::new();
    let i = Tensor::from_slice(&ctx, &[1i32, 2, 3], None, None);
    almost(&i.cast(dtype::FLOAT32).to_list_fresh::<f32>(), &[1.0, 2.0, 3.0]);
    let f = f32s(&ctx, &[1.7, -2.9, 3.2]);
    assert_eq!(f.cast(dtype::INT32).to_list_fresh::<i32>(), vec![1i32, -2, 3]);
}

#[test]
fn activations() {
    let mut ctx = Ctx::new();
    let x = f32s(&ctx, &[-2.0, -1.0, 0.0, 1.0, 2.0]);
    almost(&nn::relu(&x).to_list_fresh::<f32>(), &[0.0, 0.0, 0.0, 1.0, 2.0]);
    // gelu approx ~ x * sigmoid(2x)? tinygrad gelu default approximate; compare to std tanh gelu
    let g = nn::gelu(&x).to_list_fresh::<f32>();
    assert_eq!(g.len(), 5);
    assert!(g[4] > 1.9 && g[4] <= 2.0001);
    assert!(g[0] > -0.06 && g[0] < 0.0);

    let sm = nn::softmax(&x, Some(0)).to_list_fresh::<f32>();
    let sum: f32 = sm.iter().sum();
    almost(&[sum], &[1.0]);
}
