//! Weak-dtype lowering (`uop/weak.py`): a weak comparison must be committed to a
//! concrete width before rendering, otherwise the C-style/OpenCL backends emit
//! the non-existent `weakint` type (INTERP tolerates it, so this only shows up on
//! CL — which is why it needs its own gate).

use autorust::dtype::{self, ConstVal};
use autorust::uop::ctx::UOpCtx;
use autorust::uop::upat::{graph_rewrite, CtxVal};
use autorust::uop::weak::pm_lower_index_dtype;
use autorust::uop::UopId;

fn lower(ctx: &mut UOpCtx, root: UopId) -> UopId {
    graph_rewrite(ctx, root, &pm_lower_index_dtype(), false, None, false, false, CtxVal::None).expect("lower weak")
}

fn assert_no_weak(ctx: &mut UOpCtx, root: &UopId) {
    let weak = ctx.backward_slice_with_self(root).iter().any(|u| dtype::is_weak(ctx.dt_of(u)));
    assert!(!weak, "a weak dtype survived pm_lower_index_dtype");
}

/// tinygrad oracle: `(x.cast(weakint) < const(10, weakint))` →
/// `Cmplt(Cast(int), Const(int))` (the cast commits to the unified width).
#[test]
fn weak_comparison_is_lowered() {
    let mut ctx = UOpCtx::new();
    let x = ctx.variable("x", ConstVal::Int(0), ConstVal::Int(255), dtype::UINT8, false);
    let a = ctx.cast(x, dtype::WEAKINT);
    let b = ctx.const_py(ConstVal::Int(10), Some(dtype::WEAKINT));
    let c = ctx.lt(a, b);
    let r = lower(&mut ctx, c);
    assert_no_weak(&mut ctx, &r);
    assert_eq!(ctx.op_of(&r), autorust::uop::ops::Ops::Cmplt);
    let srcs = ctx.srcs(&r).to_vec();
    assert_eq!(srcs.len(), 2);
    assert_eq!(ctx.dt_of(&srcs[0]), dtype::INT32);
    assert_eq!(ctx.dt_of(&srcs[1]), dtype::INT32);
}

/// A weak binary consumed by a comparison is committed along the comparison edge.
#[test]
fn weak_binary_fed_to_comparison_is_lowered() {
    let mut ctx = UOpCtx::new();
    let x = ctx.variable("x", ConstVal::Int(0), ConstVal::Int(255), dtype::UINT8, false);
    let a = ctx.cast(x, dtype::WEAKINT);
    let b = ctx.const_py(ConstVal::Int(10), Some(dtype::WEAKINT));
    let sum = ctx.add(a, b);
    let c20 = ctx.const_py(ConstVal::Int(20), Some(dtype::WEAKINT));
    let c = ctx.lt(sum, c20);
    let r = lower(&mut ctx, c);
    assert_no_weak(&mut ctx, &r);
}
