//! M2 smoke tests: construction, shape, min/max, and the rewrite engine.

use autorust::dtype::{ConstVal, DTypeId};
use autorust::uop::ctx::UOpCtx;
use autorust::uop::graph::UopId;
use autorust::uop::ops::Ops;
use autorust::uop::sint::SVal;
use autorust::uop::upat::{graph_rewrite, CtxVal, PatternMatcher, UPat};

fn var(c: &mut UOpCtx, name: &str, lo: i64, hi: i64) -> UopId {
    c.variable(name, ConstVal::Int(lo as i128), ConstVal::Int(hi as i128), autorust::dtype::WEAKINT, true)
}

#[test]
fn test_intern_dedup() {
    let mut c = UOpCtx::new();
    let a = var(&mut c, "a", 0, 8);
    let a2 = var(&mut c, "a", 0, 8);
    assert_eq!(a, a2);
    let k = c.cint(2);
    let x = c.mul(a.clone(), k.clone());
    let x2 = c.mul(a, k);
    assert_eq!(x, x2);
}

#[test]
fn test_shape_basic() {
    let mut c = UOpCtx::new();
    // shape of a CONST is ()
    let k = c.cint(3);
    assert_eq!(*c.shape_of(&k).unwrap(), vec![]);
    // STACK of 4 consts has shape (4,), reshape to (2,2)
    let items: Vec<UopId> = (0..4).map(|i| c.cint_dt(i, crate_int())).collect();
    let base = c.uop(Ops::Stack, None, items, autorust::uop::arg::Arg::None);
    assert_eq!(&*c.shape_of(&base).unwrap(), &vec![SVal::I(4)]);
    let shp_node = c.shape_arg(&[SVal::I(2), SVal::I(2)]);
    let r = c.uop(Ops::Reshape, None, vec![base, shp_node], autorust::uop::arg::Arg::None);
    let sh = c.shape_of(&r).unwrap();
    assert_eq!(&*sh, &vec![SVal::I(2), SVal::I(2)]);
}

fn crate_int() -> DTypeId {
    autorust::dtype::default_int()
}

#[test]
fn test_min_max_range() {
    let mut c = UOpCtx::new();
    let a = var(&mut c, "a", 3, 8);
    let (lo, hi) = c.min_max(&a).unwrap();
    assert_eq!(lo, SVal::I(3));
    assert_eq!(hi, SVal::I(8));
    // a + 1 -> [4, 9]
    let one = c.cint(1);
    let s = c.add(a.clone(), one);
    let (lo, hi) = c.min_max(&s).unwrap();
    assert_eq!(lo, SVal::I(4));
    assert_eq!(hi, SVal::I(9));
    // CMPLT(a, 4) -> [0, 1]
    let four = c.cint(4);
    let lt = c.lt(a, four);
    let (lo, hi) = c.min_max(&lt).unwrap();
    assert_eq!(lo, SVal::I(0));
    assert_eq!(hi, SVal::I(1));
}

#[test]
fn test_rewrite_const_add() {
    let mut c = UOpCtx::new();
    let c1 = c.cfloat(1.0);
    let c2 = c.cfloat(2.0);
    let sum = c.add(c1, c2);
    // simple PM: (cvar x + cvar y) -> const(x+y)
    let pat = UPat::ops1(Ops::Add).with_src_tup(vec![
        UPat::cvar(Some("x"), None, None),
        UPat::cvar(Some("y"), None, None),
    ]);
    let pm = PatternMatcher::new(vec![(
        pat,
        std::sync::Arc::new(|ctx: &mut UOpCtx, st, _| {
            let x = autorust::uop::upat::sget(st, "x");
            let y = autorust::uop::upat::sget(st, "y");
            let a = ctx.val(&x).as_f64();
            let b = ctx.val(&y).as_f64();
            Some(ctx.cfloat(a + b))
        }),
    )]);
    let out = graph_rewrite(&mut c, sum, &pm, false, None, false, false, CtxVal::None).unwrap();
    assert_eq!(c.op_of(&out), Ops::Const);
    assert_eq!(c.val(&out).as_f64(), 3.0);
}

#[test]
fn test_rewrite_bottom_up_chain() {
    // (1+2)+(3+4) with bottom_up const folding -> 10
    let mut c = UOpCtx::new();
    let (one, two, three, four) = (c.cint(1), c.cint(2), c.cint(3), c.cint(4));
    let a = c.add(one, two);
    let b = c.add(three, four);
    let root = c.add(a, b);
    let pat = UPat::ops1(Ops::Add).with_src_tup(vec![
        UPat::cvar(Some("x"), None, None),
        UPat::cvar(Some("y"), None, None),
    ]);
    let pm = PatternMatcher::new(vec![(
        pat,
        std::sync::Arc::new(|ctx: &mut UOpCtx, st, _| {
            let x = autorust::uop::upat::sget(st, "x");
            let y = autorust::uop::upat::sget(st, "y");
            let (a, b) = (ctx.val(&x).as_int_exact().unwrap(), ctx.val(&y).as_int_exact().unwrap());
            Some(ctx.cint((a + b) as i64))
        }),
    )]);
    let out = graph_rewrite(&mut c, root, &pm, true, None, false, false, CtxVal::None).unwrap();
    assert_eq!(c.op_of(&out), Ops::Const);
    assert_eq!(c.val(&out), ConstVal::Int(10));
}

#[test]
fn test_substitute() {
    let mut c = UOpCtx::new();
    let a = var(&mut c, "a", 0, 8);
    let b = var(&mut c, "b", 0, 8);
    let two = c.cint(2);
    let x = c.mul(a.clone(), two.clone());
    let y = c.add(b.clone(), x);
    let mut dvars = std::collections::HashMap::new();
    dvars.insert(a, b.clone());
    let out = autorust::uop::upat::substitute(&mut c, y, dvars, false, false);
    // expected: b + b*2
    let bm = c.mul(b.clone(), two);
    let expected = c.add(b, bm);
    assert_eq!(out, expected);
}



#[test]
fn test_getitem_slices() {
    use autorust::s;
use autorust::tensor::{Ctx, ix::Ix, movement, Tensor};
    use autorust::uop::ctx::UOpCtx;
    use autorust::uop::ops::Ops;
    let mut ctx = Ctx::new();
    // x = STACK of 12 consts reshaped to (3, 4): simple way to get a shaped tensor
    let vals: Vec<_> = (0..12).map(|i| ctx.raw().cint(i)).collect();
    let x = ctx.raw().uop(Ops::Stack, None, vals, autorust::uop::arg::Arg::None);
    let x = movement::reshape(
        &mut ctx.raw(),
        x,
        &[autorust::uop::sint::SVal::I(3), autorust::uop::sint::SVal::I(4)],
    );
    let t = Tensor::from_uop(&ctx.raw(), x);
    // t[1] -> shape (4,)
    let t1 = t.clone().getitem(&[Ix::Index(1)]);
    assert_eq!(t1.shape().as_slice(), &[autorust::uop::sint::SVal::I(4)]);
    // t[:, 1] -> shape (3,)
    let tc = t.clone().getitem(&s![.., 1]);
    assert_eq!(tc.shape().as_slice(), &[autorust::uop::sint::SVal::I(3)]);
    // t[1:3, ::2] -> shape (2, 2)
    let ts = t.clone().getitem(&s![1..3, ..;2]);
    let sh = ts.shape();
    assert_eq!(sh.as_slice(), &[autorust::uop::sint::SVal::I(2), autorust::uop::sint::SVal::I(2)]);
    // t[NewAxis, 2] -> shape (1, 4)
    let tn = t.getitem(&s![NewAxis, 2]);
    let sh = tn.shape();
    assert_eq!(sh.as_slice(), &[autorust::uop::sint::SVal::I(1), autorust::uop::sint::SVal::I(4)]);
}
