//! Regression: a reduction whose result is broadcast and then reduced again
//! (`r = a*b; s = r.sum(2); y = r * s.reshape(2,4,1); o = y.sum(1)`).
//!
//! tinygrad realizes ALL axes of an intermediate `REDUCE` whenever it has ending
//! ranges (`indexing.py:274`: `not (PCONTIG > 1) or ...`, and `PCONTIG` defaults
//! to 0 so the first arm is always true), which materializes the reduce as its
//! own kernel/buffer. The port previously evaluated only the per-axis
//! comparison, realizing a single axis, so the two reductions were fused into
//! one (incorrect) kernel and the result — and its gradient — were wrong.
use autorust::engine::session::Session;
use autorust::tensor::Tensor;
use autorust::uop::ctx::UOpCtx;

fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "length {} != {}", a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() < 1e-4, "[{i}] {x} != {y} (got {a:?}, want {b:?})");
    }
}

fn ab(ctx: &mut UOpCtx) -> (Tensor, Tensor) {
    let a = Tensor::from_slice(ctx, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], None, None).reshape(ctx, [2, 1, 3]);
    let b = Tensor::from_slice(ctx, &[1.0f32, 0.5, -1.0, 2.0, 1.0, 0.5, -0.5, 3.0, 1.5, 0.25, -2.0, 1.0], None, None).reshape(ctx, [1, 4, 3]);
    (a, b)
}

#[test]
fn reduce_broadcast_reduce_values() {
    let mut ctx = UOpCtx::new();
    let mut sess = Session::new();
    let (a, b) = ab(&mut ctx);
    let r = a.mul(&mut ctx, &b);
    let s = r.sum(&mut ctx, Some(&[2]), false, None);
    let sr = s.reshape(&mut ctx, [2, 4, 1]);
    let y = r.mul(&mut ctx, &sr);
    let o = y.sum(&mut ctx, Some(&[1]), false, None);
    close(&o.to_list::<f32>(&mut ctx, &mut sess), &[4.8125, 73.0, 54.0, 83.0, 441.25, 225.0]);
}

/// Two *independent* full reductions in one realize where one input is itself a
/// reduce. tinygrad fuses them into a single kernel that uses one REG accumulator
/// per reduce; that lowering must keep the accumulator in the reduce's real shape.
///
/// Two port bugs made this wrong/crash:
/// - `placeholder_reg` built the flat `(prod(shape),)` buffer but never reshaped it
///   to `shape` like `UOp.placeholder` does (`ops.py:1148`), so `acc.alu(ADD, inp)`
///   broadcast a 1-D acc against a 2-D input (`[2] + [2,1] -> [2,2]`) and the
///   reduction silently summed wrong.
/// - `memory_coalescing` tested `addrspace_of` (ParamArg only) instead of the
///   derived `buf.addrspace`, so the `AFTER`-wrapped REG buffer wasn't skipped and
///   its (legitimately repeated) serial-reduce stores tripped the coalesce assert
///   ("attempting multiple stores").
#[test]
fn sibling_reductions_same_realize() {
    let mut ctx = UOpCtx::new();
    let mut sess = Session::new();
    let a = Tensor::from_slice(&mut ctx, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], None, None).reshape(&mut ctx, [2, 3]);
    let b = Tensor::from_slice(&mut ctx, &[1.0f32, 1.0], None, None);
    let bs = b.sum(&mut ctx, None, false, None);
    let out = a.sum(&mut ctx, Some(&[1]), false, None).sum(&mut ctx, None, false, None).add(&mut ctx, &bs);
    close(&out.to_list::<f32>(&mut ctx, &mut sess), &[23.0]);
}

/// Same shape of graph but with a multi-element non-reduced axis, so the REG
/// accumulator's shape matters for broadcasting (`a.sum(2).sum(1) + b.sum()`).
#[test]
fn sibling_reductions_batched() {
    let mut ctx = UOpCtx::new();
    let mut sess = Session::new();
    // a:(2,4,3), reduce twice -> (2); b:(4) reduce -> scalar; sum them.
    let a = Tensor::arange(&mut ctx, 1.0, Some(25.0), 1.0, None).reshape(&mut ctx, [2, 4, 3]);
    let b = Tensor::from_slice(&mut ctx, &[1.0f32, 2.0, 3.0, 4.0], None, None);
    let bs = b.sum(&mut ctx, None, false, None);
    let out = a.sum(&mut ctx, Some(&[2]), false, None).sum(&mut ctx, Some(&[1]), false, None).add(&mut ctx, &bs);
    // batch0 = sum(1..12) = 78, batch1 = sum(13..24) = 222; + b.sum() = 10.
    close(&out.to_list::<f32>(&mut ctx, &mut sess), &[88.0, 232.0]);
}

#[test]
fn reduce_broadcast_reduce_grad() {
    let mut ctx = UOpCtx::new();
    let mut sess = Session::new();
    let p = Tensor::from_slice(&mut ctx, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], None, None).reshape(&mut ctx, [2, 1, 3]);
    let q = Tensor::from_slice(&mut ctx, &[1.0f32, 0.5, -1.0, 2.0, 1.0, 0.5, -0.5, 3.0, 1.5, 0.25, -2.0, 1.0], None, None).reshape(&mut ctx, [1, 4, 3]);
    let r = p.mul(&mut ctx, &q);
    let s = r.sum(&mut ctx, Some(&[2]), false, None);
    let loss = s.mul(&mut ctx, &s).sum(&mut ctx, None, false, None);
    loss.backward(&mut ctx, &[p.clone(), q.clone()], None);
    // tinygrad: p.grad = [9.625, 73, 36, 41.5, 176.5, 75]
    close(&p.grad(&ctx).unwrap().to_list::<f32>(&mut ctx, &mut sess), &[9.625, 73.0, 36.0, 41.5, 176.5, 75.0]);
}
