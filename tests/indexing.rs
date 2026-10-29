//! Advanced (tensor) indexing: `Tensor.getitem` with `Ix::Tensor` indices,
//! ported from `_getitem` (mixin/op.py:74-146). Expected values are the tinygrad
//! oracle for the same inputs (int32 index tensors, `arange` data).
use autorust::engine::session::Session;
use autorust::tensor::ix::Ix;
use autorust::tensor::{Ctx, Tensor};
use autorust::uop::ctx::UOpCtx;

fn idx(ctx: &Ctx, v: &[i32]) -> Tensor {
    Tensor::from_slice(ctx, v, None, None)
}

fn at(t: &Tensor, ixs: &[Ix]) -> Vec<f32> {
    let mut sess = Session::new();
    t.getitem(ixs).to_list::<f32>(&mut sess)
}

fn arr2x3(ctx: &Ctx) -> Tensor {
    Tensor::arange(ctx, 0.0, Some(12.0), 1.0, None).reshape([3, 4])
}

fn arr3(ctx: &Ctx) -> Tensor {
    Tensor::arange(ctx, 0.0, Some(24.0), 1.0, None).reshape([2, 3, 4])
}

fn check(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "len {} != {} (got {got:?}, want {want:?})", got.len(), want.len());
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert!((a - b).abs() < 1e-5, "[{i}] {a} != {b} (got {got:?}, want {want:?})");
    }
}

#[test]
fn tensor_index_on_first_dim() {
    let mut ctx = Ctx::new();
    let t = arr2x3(&ctx);
    let i = idx(&ctx, &[0, 2]);
    check(&at(&t, &[Ix::Tensor(i), Ix::All]), &[0., 1., 2., 3., 8., 9., 10., 11.]);
}

#[test]
fn tensor_index_on_second_dim() {
    let mut ctx = Ctx::new();
    let t = arr2x3(&ctx);
    let i = idx(&ctx, &[1, 3]);
    check(&at(&t, &[Ix::All, Ix::Tensor(i)]), &[1., 3., 5., 7., 9., 11.]);
}

/// Consecutive tensor dims with int shapes take the linear-index fast path.
#[test]
fn two_consecutive_tensor_indices() {
    let mut ctx = Ctx::new();
    let t = arr2x3(&ctx);
    let a = idx(&ctx, &[0, 2]);
    let b = idx(&ctx, &[1, 3]);
    check(&at(&t, &[Ix::Tensor(a.clone()), Ix::Tensor(b)]), &[1., 11.]);
    let c = idx(&ctx, &[0, 2]);
    check(&at(&t, &[Ix::Tensor(a), Ix::Tensor(c)]), &[0., 10.]);
}

#[test]
fn negative_and_out_of_bounds() {
    let mut ctx = Ctx::new();
    let t = arr2x3(&ctx);
    let neg = idx(&ctx, &[-1, 0, -2]);
    check(&at(&t, &[Ix::Tensor(neg), Ix::All]), &[8., 9., 10., 11., 0., 1., 2., 3., 4., 5., 6., 7.]);
    // out-of-bounds reads zero (tinygrad `NOTE: Out-of-bounds indexing results in 0`)
    let oob = idx(&ctx, &[5, 0]);
    check(&at(&t, &[Ix::Tensor(oob), Ix::All]), &[0., 0., 0., 0., 0., 1., 2., 3.]);
}

#[test]
fn mixed_basic_and_advanced() {
    let mut ctx = Ctx::new();
    let t = arr2x3(&ctx);
    let i = idx(&ctx, &[0, 2]);
    check(&at(&t, &[Ix::slice(Some(1), None, 1), Ix::Tensor(i.clone())]), &[4., 6., 8., 10.]);
    check(&at(&t, &[Ix::Tensor(i), Ix::slice(Some(1), None, 1)]), &[1., 2., 3., 9., 10., 11.]);
}

#[test]
fn broadcast_tensor_indices() {
    let mut ctx = Ctx::new();
    let t = arr2x3(&ctx);
    // (2,1) broadcast against (2,)
    let col = Tensor::from_slice(&ctx, &[0i32, 2], None, None).reshape([2, 1]);
    let i = idx(&ctx, &[1, 3]);
    check(&at(&t, &[Ix::Tensor(col), Ix::Tensor(i)]), &[1., 3., 9., 11.]);
}

#[test]
#[should_panic(expected = "shape mismatch")]
fn non_broadcastable_tensor_indices_panic() {
    let mut ctx = Ctx::new();
    let t = arr2x3(&ctx);
    let row = Tensor::from_slice(&ctx, &[0i32, 1, 2], None, None).reshape([1, 3]);
    let i = idx(&ctx, &[0, 2]);
    // (1,3) vs (2,) cannot broadcast (3 != 2)
    let _ = at(&t, &[Ix::Tensor(row), Ix::Tensor(i)]);
}

#[test]
fn three_d_non_consecutive_and_consecutive() {
    let mut ctx = Ctx::new();
    let t = arr3(&ctx);
    let i01 = idx(&ctx, &[0, 1]);
    let i02 = idx(&ctx, &[0, 2]);
    let i12 = idx(&ctx, &[1, 2]);
    // advanced on dims 0 and 2 (non-consecutive -> mask + permute path)
    check(&at(&t, &[Ix::Tensor(i01.clone()), Ix::All, Ix::Tensor(i12.clone())]), &[1., 5., 9., 14., 18., 22.]);
    // advanced on dims 1 and 2 with leading All (consecutive, linear path)
    check(&at(&t, &[Ix::All, Ix::Tensor(i02.clone()), Ix::Tensor(i12)]), &[1., 10., 13., 22.]);
    // advanced on dims 0 and 1 -> trailing All (consecutive, linear path)
    check(&at(&t, &[Ix::Tensor(i01), Ix::Tensor(i02)]), &[0., 1., 2., 3., 20., 21., 22., 23.]);
}

#[test]
fn single_tensor_on_1d() {
    let mut ctx = Ctx::new();
    let t = Tensor::arange(&ctx, 0.0, Some(5.0), 1.0, None);
    let i = idx(&ctx, &[0, 3, 3]);
    check(&at(&t, &[Ix::Tensor(i)]), &[0., 3., 3.]);
}
