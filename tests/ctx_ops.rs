//! `Ctx`-in-`Tensor` ergonomics: operator syntax, same-arena enforcement, and
//! the context-free graph API.

use autorust::engine::session::Session;
use autorust::tensor::{Ctx, Tensor};

fn f32s(ctx: &Ctx, data: &[f32]) -> Tensor {
    Tensor::from_slice(ctx, data, None, None)
}

#[test]
fn operator_syntax_uses_the_stored_ctx() {
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let a = f32s(&ctx, &[1.0, 2.0, 3.0]);
    let b = f32s(&ctx, &[10.0, 20.0, 30.0]);
    let c = &a + &b;
    assert_eq!(c.to_list::<f32>(&mut sess), vec![11.0, 22.0, 33.0]);
    let d = &a - &b;
    assert_eq!(d.to_list::<f32>(&mut sess), vec![-9.0, -18.0, -27.0]);
    let e = &a * 2.0;
    assert_eq!(e.to_list::<f32>(&mut sess), vec![2.0, 4.0, 6.0]);
    let f = (&a / 2.0).neg();
    assert_eq!(f.to_list::<f32>(&mut sess), vec![-0.5, -1.0, -1.5]);
    let g = a + b;
    assert_eq!(g.to_list::<f32>(&mut sess), vec![11.0, 22.0, 33.0]);
}

#[test]
fn methods_do_not_need_the_context_argument() {
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let a = f32s(&ctx, &[1.0, 2.0, 3.0, 4.0]).reshape([2, 2]);
    assert_eq!(a.shape().as_slice().len(), 2);
    let s = a.sum(None, false, None);
    assert_eq!(s.to_list::<f32>(&mut sess), vec![10.0]);
    let t = a.transpose(0, 1);
    assert_eq!(t.to_list::<f32>(&mut sess), vec![1.0, 3.0, 2.0, 4.0]);
}

#[test]
#[should_panic(expected = "different Ctx")]
fn mixing_contexts_is_rejected() {
    let a_ctx = Ctx::new();
    let b_ctx = Ctx::new();
    let a = f32s(&a_ctx, &[1.0]);
    let b = f32s(&b_ctx, &[2.0]);
    let _ = &a + &b;
}

#[test]
fn tensors_expose_their_arena() {
    let ctx = Ctx::new();
    let a = f32s(&ctx, &[1.0]);
    assert!(a.ctx().same(&ctx));
    assert!(a.ctx().same(&a.ctx().clone()));
}
