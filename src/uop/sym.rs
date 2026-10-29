//! The symbolic rewrite rule sets (uop/symbolic.py + uop/movement.py +
//! uop/divandmod.py), ported rule-by-rule in Python order.
use crate::uop::upat::PmSlot;
use std::sync::Arc;

use super::alu::exec_alu;
use super::arg::Arg;
use super::ctx::UOpCtx;
use std::collections::HashMap;

use super::ctx::sval_cmp;
use super::graph::UopId;
use super::ops::{Ops, group};
use super::sint::SVal;
pub use super::upat::sget;
use super::upat::substitute;
pub use super::upat::{
    ConstArgVal, CtxVal, PatternMatcher, RuleFn, Store, UPat, UPatOrConst, const_arg, opset_diff,
    opset_union, permutations, sget_opt,
};
use crate::dtype::{self, ConstVal, DTypeId, WEAKINT, can_lossless_cast};

use ConstVal as CV;

/// `commit_weak` (weak.py:10-12).
pub fn commit_weak(ctx: &mut UOpCtx, s: UopId, dt: DTypeId) -> UopId {
    if ctx.op_of(&s) == Ops::Const {
        let v = ctx.val(&s);
        ctx.const_py(v, Some(dt))
    } else {
        ctx.cast(s, dt)
    }
}

/// `overflows` (ops.py:658).
fn overflows(ctx: &mut UOpCtx, u: UopId, d: DTypeId) -> bool {
    let (vmin, vmax) = match ctx.min_max(&u) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let (dmin, dmax) = ctx.dtype_bounds(d).unwrap();
    let lo = vmin.as_i128_or_panic();
    let hi = vmax.as_i128_or_panic();
    let (dl, dh) = (dmin.as_i128_or_panic(), dmax.as_i128_or_panic());
    lo < dl || dh < hi
}

/// `exec_alu` with tuple (STACK) support — `fold_const_alu`'s body.
fn fold_const_alu(ctx: &mut UOpCtx, a: UopId) -> Option<UopId> {
    let (op, dt, srcs, _, _) = ctx.snapshot(&a);
    let mut vals: Vec<Option<ConstArgVal>> = Vec::with_capacity(srcs.len());
    for s in &srcs {
        {
            let c = const_arg(ctx, s.clone())?;
            vals.push(Some(c))
        }
    }
    // vector expansion like Python's exec_alu tuple branch
    let veclen = vals
        .iter()
        .flatten()
        .filter_map(|v| match v {
            ConstArgVal::Vec(vs) => Some(vs.len()),
            _ => None,
        })
        .max();
    let result: Result<ConstArgVal, String> = match veclen {
        None => {
            let scalars: Vec<ConstVal> = vals
                .iter()
                .map(|v| v.as_ref().unwrap().clone().one().unwrap())
                .collect();
            match exec_alu(op, dt, &scalars, false) {
                Ok(v) => Ok(ConstArgVal::One(v)),
                // Python: an exception inside a rule aborts the rewrite
                Err(e) => panic!("TGERR: exec_alu: {e}"),
            }
        }
        Some(n) => {
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let scalars: Vec<ConstVal> = vals
                    .iter()
                    .map(|v| match v.as_ref().unwrap() {
                        ConstArgVal::One(c) => c.clone(),
                        ConstArgVal::Vec(vs) => vs[i].clone(),
                    })
                    .collect();
                match exec_alu(op, dt, &scalars, false) {
                    Ok(v) => out.push(v),
                    Err(e) => panic!("TGERR: exec_alu: {e}"),
                }
            }
            Ok(ConstArgVal::Vec(out))
        }
    };
    match result {
        Err(_) => None, // Python raises (e.g. KeyError for FDIV) -> rewrite error; treat as no-match (divergence documented)
        Ok(v) => Some(const_like_cav(ctx, a, v)),
    }
}

/// `const_like` for a ConstArgVal (scalar or tuple → STACK).
fn const_like_cav(ctx: &mut UOpCtx, a: UopId, v: ConstArgVal) -> UopId {
    match v {
        ConstArgVal::One(c) => ctx.const_like(a, c),
        ConstArgVal::Vec(items) => {
            let dt = ctx.dt_of(&a);
            let srcs: Vec<UopId> = items
                .into_iter()
                .map(|c| ctx.const_py(c, Some(dt)))
                .collect();
            ctx.uop(Ops::Stack, None, srcs, Arg::None)
        }
    }
}

/// `simplify_pow` (symbolic.py:16-21).
fn simplify_pow(ctx: &mut UOpCtx, x: UopId, c: UopId) -> Option<UopId> {
    let cv = ctx.val(&c).as_f64();
    if cv < 0.0 {
        let neg = ctx.const_py(CV::Float(-cv), None);
        {
            let rx = ctx.reciprocal(x);
            return Some(ctx.pow(rx, neg));
        }
    }
    if cv == 0.0 {
        return Some(ctx.const_like(x, CV::Int(1)));
    }
    let iv = (cv - 0.5).trunc() + 0.5;
    if iv == cv {
        let half = ctx.const_py(CV::Float(cv - 0.5), None);
        let p = ctx.pow(x.clone(), half);
        let s = ctx.sqrt(x);
        return Some(ctx.mul(p, s));
    }
    if cv.fract() == 0.0 {
        let ci = cv as i128;
        let half_exp = ctx.const_py(CV::Float((ci / 2) as f64), None);
        let y = ctx.pow(x.clone(), half_exp);
        let y2 = ctx.mul(y.clone(), y);
        if ci % 2 == 1 {
            Some(ctx.mul(y2, x))
        } else {
            {
                let one = ctx.const_like(x, CV::Int(1));
                Some(ctx.mul(y2, one))
            }
        }
    } else {
        None
    }
}

/// `fold_bitcast` (symbolic.py:23-25).
fn fold_bitcast(ctx: &mut UOpCtx, root: UopId, c: UopId) -> Option<UopId> {
    let cdt = ctx.dt_of(&c);
    let rdt = ctx.dt_of(&root);
    if cdt.fmt().is_none() || rdt.fmt().is_none() || cdt.itemsize() != rdt.itemsize() {
        return None;
    }
    let v = ctx.val(&c);
    let out = dtype::bitcast_val(v, cdt, rdt);
    Some(ctx.const_like(root, out))
}

/// `fold_const_where` (symbolic.py:104-107).
fn fold_const_where(
    ctx: &mut UOpCtx,
    _gate: UopId,
    c0: UopId,
    c1: UopId,
    w: UopId,
) -> Option<UopId> {
    let gate_val = match ctx.val(&_gate) {
        CV::Bool(b) => b,
        CV::Int(i) => i != 0,
        _ => return None,
    };
    let ret = if gate_val { c0 } else { c1 };
    let wdt = ctx.dt_of(&w);
    if ctx.op_of(&ret) == Ops::Const && dtype::is_weak(ctx.dt_of(&ret)) && !dtype::is_weak(wdt) {
        Some(commit_weak(ctx, ret, wdt))
    } else {
        Some(ret)
    }
}

/// canonical boolean/dtype sets used by the rules
pub fn weak_dts() -> Vec<DTypeId> {
    vec![dtype::WEAKINT, dtype::WEAKFLOAT]
}

/// `dtypes.ints + (weakint,)` — int dtypes accepted by the symbolic rules.
pub fn ints_plus_weak() -> Vec<DTypeId> {
    let mut v: Vec<DTypeId> = dtype::INTS.to_vec();
    v.push(WEAKINT);
    v
}

/// `dtypes.sints + (weakint,)` — signed int dtypes accepted by the symbolic rules.
pub fn weak_int_dts() -> Vec<DTypeId> {
    let mut v: Vec<DTypeId> = crate::dtype::SINTS.to_vec();
    v.push(crate::dtype::WEAKINT);
    v
}

/// `dtypes.sints + (weakint,)` — signed int dtypes for constant-folding rules.
pub fn sints_plus_weakint() -> Vec<DTypeId> {
    let mut v: Vec<DTypeId> = dtype::SINTS.to_vec();
    v.push(WEAKINT);
    v
}

/// CONST-or-STACK folding pattern.
fn const_folding_pat() -> UPat {
    UPat::op_var(vec![Ops::Const, Ops::Stack])
}

/// `UPat.var(name)` — a named pattern variable matching any op.
pub fn var(name: &'static str) -> UPat {
    UPat::var(Some(name), None)
}

/// `UPat.var(name, dtype)` — a named pattern variable restricted to dtypes.
pub fn var_dt(name: &'static str, dts: Vec<DTypeId>) -> UPat {
    UPat::var(Some(name), Some(dts))
}

/// `UPat.cvar(name)` — a named pattern matching a const node.
pub fn cvar(name: &'static str) -> UPat {
    UPat::cvar(Some(name), None, None)
}

/// `UPat.cvar(name, arg)` — a const pattern that must carry the given arg.
pub fn cvar_arg(name: &'static str, a: CV) -> UPat {
    UPat::cvar(Some(name), None, Some(Arg::Const(a)))
}

// ---------- pm_data_invalid (symbolic.py:74-96) ----------

/// Pattern matching the `Invalid` const.
pub fn invalid_pat() -> UPat {
    UPat::cvar(Some("i"), None, Some(Arg::Const(CV::Invalid)))
}

/// `cond.where(x, Invalid)` — the data-invalid gate pattern.
pub fn invalid_gate() -> UPat {
    var("cond").where_(UPatOrConst::Pat(var("x")), UPatOrConst::Pat(invalid_pat()))
}

fn mop_base(name: &'static str) -> UPat {
    UPat::var(Some(name), None)
}

fn is_invalid_node(ctx: &UOpCtx, u: UopId) -> bool {
    ctx.op_of(&u) == Ops::Const && matches!(ctx.arg_of(&u), Arg::Const(CV::Invalid))
}

fn pm_data_invalid() -> PatternMatcher {
    let unary_cast = opset_union(group::UNARY, &[Ops::Cast, Ops::Bitcast]);
    let bin_no_cmp = opset_diff(group::BINARY, group::COMPARISON);
    let rules: Vec<(UPat, RuleFn)> = vec![
        // STACK(any number of invalid) -> invalid
        (
            invalid_pat().broadcast(false),
            r(|_, st, _| Some(sget(st, "i"))),
        ),
        // unary/cast of invalid -> invalid
        (
            UPat::op_var(unary_cast.clone()).with_src_tup(vec![invalid_pat()]),
            r(|_, st, _| Some(sget(st, "i"))),
        ),
        // gated unary lifts the gate out
        (
            UPat::op_var(unary_cast.clone())
                .with_src_tup(vec![invalid_gate()])
                .named("op"),
            r(|ctx, st, _| {
                let (cond, x, op, i) = (
                    sget(st, "cond"),
                    sget(st, "x"),
                    sget(st, "op"),
                    sget(st, "i"),
                );
                let (o, d, _, a, t) = ctx.snapshot(&op);
                let replaced = ctx.replace(&op, o, d, vec![x], a, t);
                Some(ctx.where_(cond, replaced, i))
            }),
        ),
        // binary ops move inside the gate
        (
            UPat::op_var(group::BINARY.to_vec())
                .with_src_tup(vec![invalid_gate(), var("y")])
                .named("alu"),
            r(|ctx, st, _| {
                let (cond, x, y, alu, i) = (
                    sget(st, "cond"),
                    sget(st, "x"),
                    sget(st, "y"),
                    sget(st, "alu"),
                    sget(st, "i"),
                );
                let inner = ctx.alu2(ctx.op_of(&alu), x, y);
                Some(ctx.where_(cond, inner, i))
            }),
        ),
        (
            UPat::op_var(group::BINARY.to_vec())
                .with_src_tup(vec![var("y"), invalid_gate()])
                .named("alu"),
            r(|ctx, st, _| {
                let (cond, x, y, alu, i) = (
                    sget(st, "cond"),
                    sget(st, "x"),
                    sget(st, "y"),
                    sget(st, "alu"),
                    sget(st, "i"),
                );
                let inner = ctx.alu2(ctx.op_of(&alu), y, x);
                Some(ctx.where_(cond, inner, i))
            }),
        ),
        // binary (non-comparison) with an invalid operand is invalid
        (
            UPat::op_var(bin_no_cmp).with_src_list(vec![invalid_pat(), UPat::var(None, None)]),
            r(|_, st, _| Some(sget(st, "i"))),
        ),
        // invalid condition poisons the whole where
        (
            invalid_pat().where_(
                UPatOrConst::Pat(UPat::var(None, None)),
                UPatOrConst::Pat(UPat::var(None, None)),
            ),
            r(|_, st, _| Some(sget(st, "i"))),
        ),
        // gated invalid condition lifts the gate out
        (
            invalid_gate().where_(UPatOrConst::Pat(var("a")), UPatOrConst::Pat(var("b"))),
            r(|ctx, st, _| {
                let (cond, x, i, a, b) = (
                    sget(st, "cond"),
                    sget(st, "x"),
                    sget(st, "i"),
                    sget(st, "a"),
                    sget(st, "b"),
                );
                let inner = ctx.where_(x, a, b);
                Some(ctx.where_(cond, inner, i))
            }),
        ),
        // where(cond, Invalid, val) -> where(~cond, val, Invalid)
        (
            var("cond").where_(
                UPatOrConst::Pat(invalid_pat()),
                UPatOrConst::Pat(var("val")),
            ),
            r(|ctx, st, _| {
                let (cond, i, val) = (sget(st, "cond"), sget(st, "i"), sget(st, "val"));
                if !is_invalid_node(ctx, val.clone()) {
                    let nc = ctx.logical_not(cond);
                    Some(ctx.where_(nc, val, i))
                } else {
                    Some(i)
                }
            }),
        ),
        // a.where(cond.where(x, Invalid), c) -> (~a|cond).where(a.where(x, c), Invalid)
        (
            var("a").where_(UPatOrConst::Pat(invalid_gate()), UPatOrConst::Pat(var("c"))),
            r(|ctx, st, _| {
                let (cond, i, x, a, c) = (
                    sget(st, "cond"),
                    sget(st, "i"),
                    sget(st, "x"),
                    sget(st, "a"),
                    sget(st, "c"),
                );
                if !is_invalid_node(ctx, c.clone()) {
                    let na = ctx.logical_not(a.clone());
                    let g = ctx.or_(na, cond);
                    let inner = ctx.where_(a, x, c);
                    Some(ctx.where_(g, inner, i))
                } else {
                    None
                }
            }),
        ),
        (
            var("a").where_(UPatOrConst::Pat(var("b")), UPatOrConst::Pat(invalid_gate())),
            r(|ctx, st, _| {
                let (cond, i, x, a, b) = (
                    sget(st, "cond"),
                    sget(st, "i"),
                    sget(st, "x"),
                    sget(st, "a"),
                    sget(st, "b"),
                );
                if !is_invalid_node(ctx, b.clone()) {
                    let g = ctx.or_(a.clone(), cond);
                    let inner = ctx.where_(a, b, x);
                    Some(ctx.where_(g, inner, i))
                } else {
                    None
                }
            }),
        ),
        // fold gated STORE/LOAD
        (
            UPat::ops1(Ops::Store).with_src_tup(vec![
                UPat::ops1(Ops::Index)
                    .with_src_tup(vec![UPat::var(None, None), invalid_pat()])
                    .with_src_any_len(true)
                    .or_casted(None),
                UPat::var(None, None),
            ]),
            r(|ctx, _, _| Some(ctx.uop(Ops::Noop, None, vec![], Arg::None))),
        ),
        (
            UPat::ops1(Ops::Load)
                .with_src_tup(vec![
                    UPat::ops1(Ops::Index)
                        .with_src_tup(vec![UPat::var(None, None), invalid_pat()])
                        .with_src_any_len(true)
                        .or_casted(None),
                ])
                .named("x"),
            r(|ctx, st, _| {
                let (x, _) = (sget(st, "x"), sget(st, "i"));
                let srcs = ctx.srcs(&x).to_vec();
                if srcs.len() > 1 {
                    Some((srcs[1]).clone())
                } else {
                    Some(ctx.const_like(x, CV::Int(0)))
                }
            }),
        ),
    ];
    PatternMatcher::new(rules)
}

// shorthand for rule closures
/// Wrap a rewrite closure as a `RuleFn`.
pub fn r<F>(f: F) -> RuleFn
where
    F: Fn(&mut UOpCtx, &Store, &CtxVal) -> Option<UopId> + Send + Sync + 'static,
{
    Arc::new(f)
}

impl UPat {
    /// `UPat(..., src=UPat.any(*alts))` — match any one of `alts` as the src pattern.
    pub fn with_src_any(self, alts: Vec<UPat>) -> UPat {
        let mut p = self.clone();
        p.src = super::upat::SrcPat::Any(alts);
        p.strict_length = false;
        p.required_len = 0;
        p.early_reject = Vec::new(); // Any alternatives have no direct srcs
        p
    }

    /// `UPat(..., src=(...))` — match a fixed-length list of srcs in any order.
    pub fn with_src_list(self, srcs: Vec<UPat>) -> UPat {
        let n = srcs.len();
        let first = srcs.clone();
        let mut p = self.clone();
        p.src = super::upat::SrcPat::List(permutations(srcs));
        p.strict_length = true;
        p.required_len = n;
        p.early_reject = {
            let mut er = Vec::new();
            for q in &first {
                if let Some(ops) = &q.op
                    && ops.len() == 1
                    && !er.contains(&ops[0])
                {
                    er.push(ops[0]);
                }
            }
            er
        };
        p
    }
}

/// `mop_cleanup` (movement.py).
fn pm_mop_cleanup_inner() -> PatternMatcher {
    let rules: Vec<(UPat, RuleFn)> = vec![
        // merge adjacent RESHAPES
        (
            UPat::ops1(Ops::Reshape)
                .with_src_tup(vec![
                    UPat::ops1(Ops::Reshape).named("x2"),
                    UPat::var(None, None),
                ])
                .named("x"),
            r(|ctx, st, _| {
                let (x, x2) = (sget(st, "x"), sget(st, "x2"));
                let (_, _, _, arg, _) = ctx.snapshot(&x);
                let inner = &ctx.srcs(&x2)[0].clone();
                let (_, d, _, _, _) = ctx.snapshot(&x);
                let dt = if ctx.dt_of(inner) == ctx.dt_of(&ctx.srcs(&x)[0]) {
                    d
                } else {
                    ctx.dt_of(inner)
                };
                Some(ctx.replace(
                    &x,
                    Ops::Reshape,
                    dt,
                    vec![inner.clone(), ctx.srcs(&x)[1].clone()],
                    arg,
                    Arg::None,
                ))
            }),
        ),
        // remove noop RESHAPES
        (
            UPat::ops1(Ops::Reshape)
                .with_src_tup(vec![mop_base("x2"), UPat::var(None, None)])
                .named("x"),
            r(|ctx, st, _| {
                let (x, x2) = (sget(st, "x"), sget(st, "x2"));
                if ctx.shape(&x2).ok().flatten().is_some() {
                    let s2 = ctx.shape_of(&x2).ok()?;
                    let sx = ctx.shape_of(&x).ok()?;
                    if *s2 == *sx {
                        return Some(x2);
                    }
                }
                None
            }),
        ),
        // merge PERMUTEs
        (
            UPat::ops1(Ops::Permute)
                .with_src_tup(vec![UPat::ops1(Ops::Permute).named("x2")])
                .named("x"),
            r(|ctx, st, _| {
                let (x, x2) = (sget(st, "x"), sget(st, "x2"));
                let perm = ctx.marg_ints(x).ok()?;
                let arg2: Vec<i64> = ctx.marg_ints(x2.clone()).ok()?;
                let new: Vec<Arg> = perm.iter().map(|&i| Arg::Int(arg2[i as usize])).collect();
                let (_, d, _, _, _) = ctx.snapshot(&x2);
                Some(ctx.replace(
                    &x2,
                    Ops::Permute,
                    d,
                    ctx.srcs(&x2).to_vec(),
                    Arg::Seq {
                        tuple: true,
                        items: new,
                    },
                    Arg::None,
                ))
            }),
        ),
        // remove noop PERMUTEs
        (
            UPat::ops1(Ops::Permute).named("x"),
            r(|ctx, st, _| {
                let x = sget(st, "x");
                let perm = ctx.marg_ints(x.clone()).ok()?;
                if perm == (0..perm.len() as i64).collect::<Vec<_>>() {
                    Some(ctx.srcs(&x)[0].clone())
                } else {
                    None
                }
            }),
        ),
        // STACK on INDEX CONST
        (
            UPat::ops1(Ops::Stack)
                .with_src_rep(
                    UPat::ops1(Ops::Index).with_src_tup(vec![var("src"), UPat::ops1(Ops::Const)]),
                )
                .named("stk"),
            r(|ctx, st, _| {
                let (src, stk) = (sget(st, "src"), sget(st, "stk"));
                let sshape = ctx.shape_of(&stk).ok()?;
                let srcshape = ctx.shape_of(&src).ok()?;
                if *sshape != *srcshape {
                    return None;
                }
                let srcs = ctx.srcs(&stk).to_vec();
                let want: Vec<i64> = (0..srcs.len() as i64).collect();
                let got: Vec<i64> = srcs
                    .iter()
                    .filter_map(|s| {
                        let idx = &ctx.srcs(s)[1].clone();
                        ctx.val(idx).as_int_exact().map(|v| v as i64)
                    })
                    .collect();
                if got == want { Some(src) } else { None }
            }),
        ),
        // const INDEX into STACK is src
        (
            UPat::ops1(Ops::Index)
                .with_src_tup(vec![UPat::ops1(Ops::Stack).named("a"), cvar("i")])
                // python: `allow_any_len=True` — the index may carry extra args
                // (`INDEX(STACK, i, *rest)`), which must not fall through to the
                // "INDEX on shaped INDEX" rule (that creates `INDEX(STACK, ..)`)
                .with_src_any_len(true)
                .named("idx"),
            r(|ctx, st, _| {
                let (a, i, idx) = (sget(st, "a"), sget(st, "i"), sget(st, "idx"));
                let iv = ctx.val(&i).as_int_exact()? as usize;
                let srcs = ctx.srcs(&a).to_vec();
                if ctx.srcs(&idx).len() <= 2 {
                    srcs.get(iv).cloned()
                } else {
                    // a.src[i].index(*idx.src[2:])
                    let base = srcs.get(iv)?.clone();
                    let extra = ctx.srcs(&idx)[2..].to_vec();
                    Some(ctx.index(base, extra))
                }
            }),
        ),
        // INDEX on INDEX is INDEX (all scalar index args)
        (
            UPat::ops1(Ops::Index)
                .with_src_tup(vec![
                    UPat::ops1(Ops::Index).with_src_any_len(true).named("idx1"),
                ])
                .with_src_any_len(true)
                .named("idx2"),
            r(|ctx, st, _| {
                let (idx1, idx2) = (sget(st, "idx1"), sget(st, "idx2"));
                let s1 = ctx.srcs(&idx1).to_vec();
                let s2 = ctx.srcs(&idx2).to_vec();
                let scalar_ok = s1[1..]
                    .iter()
                    .chain(s2[1..].iter())
                    .all(|u| *ctx.shape_of(u).unwrap_or_default() == vec![]);
                if !scalar_ok {
                    return None;
                }
                // python: `idx1.src[0].index(*idx1.src[1:], *idx2.src[1:])`
                let mut ns: Vec<UopId> = s1[1..].to_vec();
                ns.extend(s2[1..].iter().cloned());
                Some(ctx.index((s1[0]).clone(), ns))
            }),
        ),
        // INDEX on shaped INDEX
        (
            UPat::ops1(Ops::Index)
                .with_src_tup(vec![
                    UPat::ops1(Ops::Index).with_src_tup(vec![var("buf"), var("idx1_arg")]),
                ])
                .with_src_any_len(true)
                .named("idx2"),
            r(|ctx, st, _| {
                let (buf, idx1_arg, idx2) =
                    (sget(st, "buf"), sget(st, "idx1_arg"), sget(st, "idx2"));
                let s2 = ctx.srcs(&idx2).to_vec();
                let arg_shape = ctx.shape_of(&idx1_arg).ok()?;
                if arg_shape.len() == s2.len() - 1 {
                    // python: `buf.index(idx1_arg.index(*idx2.src[1:]))` — the
                    // inner `.index()` goes through the UOp.index fold
                    let inner = ctx.index(idx1_arg, s2[1..].to_vec());
                    Some(ctx.uop(Ops::Index, None, vec![buf, inner], Arg::None))
                } else {
                    None
                }
            }),
        ),
    ];
    PatternMatcher::new(rules)
}

impl UPat {
    fn with_src_rep(self, p: UPat) -> UPat {
        let mut q = self.clone();
        q.recompute_early_reject_rep(&p);
        q.src = super::upat::SrcPat::Repeat(Box::new(p));
        q.strict_length = false;
        q.required_len = 0;
        q
    }

    /// Mirror `UPat(allow_any_len=any)`: relax the strict src-length check.
    pub fn with_src_any_len(self, any: bool) -> UPat {
        let mut q = self;
        if any {
            q.strict_length = false;
        }
        q
    }
}

/// `commutative` (symbolic.py:214-219).
fn pm_commutative_inner() -> PatternMatcher {
    let rules: Vec<(UPat, RuleFn)> = vec![(
        UPat::op_var(group::COMMUTATIVE.to_vec())
            .with_dtype(Some(vec![WEAKINT]))
            .named("x"),
        r(|ctx, st, _| {
            let x = sget(st, "x");
            let srcs = ctx.srcs(&x).to_vec();
            if srcs.len() != 2 {
                return None;
            }
            let (a, b) = (ctx.tuplize(&srcs[0]), ctx.tuplize(&srcs[1]));
            if b < a && !(a < b) {
                let (_, d, _, _, _) = ctx.snapshot(&x);
                Some(ctx.replace(
                    &x,
                    ctx.op_of(&x),
                    d,
                    vec![(srcs[1]).clone(), (srcs[0]).clone()],
                    ctx.arg_of(&x).clone(),
                    ctx.tag_of(&x).clone(),
                ))
            } else {
                None
            }
        }),
    )];
    PatternMatcher::new(rules)
}

// ---------- PM assembly (symbolic_simple first, phase 2 next) ----------

static SYMBOLIC_SIMPLE: PmSlot = PmSlot::new();
static SYMBOLIC_PM: PmSlot = PmSlot::new();

/// `symbolic_simple` = pm_data_invalid + phase-1 folding + mop_cleanup
/// (symbolic.py:109-191).
pub fn symbolic_simple() -> &'static PatternMatcher {
    SYMBOLIC_SIMPLE.get_or_init(build_symbolic_simple)
}

/// `symbolic` = symbolic_simple + commutative + phase-2 (symbolic.py:228-309).
pub fn symbolic() -> &'static PatternMatcher {
    SYMBOLIC_PM.get_or_init(|| {
        symbolic_simple()
            .add(pm_commutative())
            .add(&build_symbolic_phase2())
    })
}

fn build_symbolic_simple() -> PatternMatcher {
    let mut rules: Vec<(UPat, RuleFn)> = Vec::new();
    // ** self folding **
    rules.push((
        UPat::op_var(vec![Ops::Add, Ops::Xor, Ops::Or])
            .with_src_list(vec![var("x"), cvar_arg("c", CV::Int(0))]),
        r(|_, st, _| Some(sget(st, "x"))),
    ));
    rules.push((
        UPat::op_var(vec![Ops::Shl, Ops::Shr])
            .with_src_tup(vec![var("x"), cvar_arg("c", CV::Int(0))]),
        r(|_, st, _| Some(sget(st, "x"))),
    ));
    rules.push((
        var("x").mul(UPatOrConst::Val(CV::Int(1))),
        r(|_, st, _| Some(sget(st, "x"))),
    ));
    rules.push((var("x").floordiv(UPatOrConst::Pat(var("x"))), {
        r(|ctx, st, _| Some(ctx.const_like(sget(st, "x"), CV::Int(1))))
    }));
    rules.push((
        var("x").floordiv(UPatOrConst::Val(CV::Int(1))),
        r(|_, st, _| Some(sget(st, "x"))),
    ));
    rules.push((
        var("x").floordiv(UPatOrConst::Val(CV::Int(-1))),
        r(|ctx, st, _| Some(ctx.neg(sget(st, "x")))),
    ));
    rules.push((
        var("x")
            .xor(UPatOrConst::Pat(var("y")))
            .xor(UPatOrConst::Pat(var("y"))),
        r(|_, st, _| Some(sget(st, "x"))),
    ));
    rules.push((
        UPat::var(None, None)
            .mod_(UPatOrConst::Pat(var("y")))
            .named("base")
            .mod_(UPatOrConst::Pat(var("y"))),
        r(|_, st, _| Some(sget(st, "base"))),
    ));
    // variations of (x%c)+(x//c)*c = x — fold_add_divmod_recombine (deferred to phase 2 below)
    rules.push((
        UPat::ops1(Ops::Add)
            .with_dtype(Some(vec![WEAKINT]))
            .named("x"),
        r(|ctx, st, _| fold_add_divmod_recombine(ctx, sget(st, "x"))),
    ));
    rules.push((
        var_dt("x", vec![crate::dtype::BOOL]).and_(UPatOrConst::Pat(cvar("c"))),
        r(|ctx, st, _| {
            let (x, c) = (sget(st, "x"), sget(st, "c"));
            if ctx.val(&c).py_cmp(&CV::Int(0)) == Some(std::cmp::Ordering::Greater) {
                Some(x)
            } else {
                Some(c)
            }
        }),
    ));
    rules.push((
        var_dt("x", vec![crate::dtype::BOOL]).or_(UPatOrConst::Pat(cvar("c"))),
        r(|ctx, st, _| {
            let (x, c) = (sget(st, "x"), sget(st, "c"));
            if ctx.val(&c).py_cmp(&CV::Int(0)) == Some(std::cmp::Ordering::Greater) {
                Some(c)
            } else {
                Some(x)
            }
        }),
    ));
    rules.push((
        var_dt("x", vec![crate::dtype::BOOL]).ne(UPatOrConst::Val(CV::Bool(false))),
        r(|_, st, _| Some(sget(st, "x"))),
    ));
    rules.push((
        UPat::op_var(group::IDEMPOTENT.to_vec()).with_src_list(vec![var("x"), var("x")]),
        r(|_, st, _| Some(sget(st, "x"))),
    ));
    rules.push((
        var_dt("x", vec![crate::dtype::BOOL])
            .logical_not()
            .logical_not(),
        r(|_, st, _| Some(sget(st, "x"))),
    ));
    rules.push((
        var_dt("x", vec![crate::dtype::BOOL]).where_(
            UPatOrConst::Pat(UPat::cvar(
                None,
                Some(vec![crate::dtype::BOOL]),
                Some(Arg::Const(CV::Bool(true))),
            )),
            UPatOrConst::Pat(UPat::cvar(
                None,
                Some(vec![crate::dtype::BOOL]),
                Some(Arg::Const(CV::Bool(false))),
            )),
        ),
        r(|_, st, _| Some(sget(st, "x"))),
    ));
    rules.push((
        var_dt("x", vec![crate::dtype::BOOL]).where_(
            UPatOrConst::Pat(UPat::cvar(
                None,
                Some(vec![crate::dtype::BOOL]),
                Some(Arg::Const(CV::Bool(false))),
            )),
            UPatOrConst::Pat(UPat::cvar(
                None,
                Some(vec![crate::dtype::BOOL]),
                Some(Arg::Const(CV::Bool(true))),
            )),
        ),
        r(|ctx, st, _| Some(ctx.logical_not(sget(st, "x")))),
    ));
    // CAST(bool -> int) != const
    rules.push((
        var_dt("x", vec![crate::dtype::BOOL])
            .cast_dts(ints_plus_weak())
            .ne(UPatOrConst::Pat(cvar("c"))),
        r(|ctx, st, _| {
            let (x, c) = (sget(st, "x"), sget(st, "c"));
            let cv = ctx.val(&c).as_int_exact().unwrap_or(0);
            if cv == 0 {
                Some(x)
            } else if cv == 1 {
                Some(ctx.logical_not(x))
            } else {
                Some(ctx.const_like(x, CV::Bool(true)))
            }
        }),
    ));
    rules.push((
        var_dt("x", ints_plus_weak()).trunc(),
        r(|_, st, _| Some(sget(st, "x"))),
    ));
    // ** zero folding **
    rules.push((var("x").lt(UPatOrConst::Pat(var("x"))), {
        r(|ctx, st, _| {
            Some(ctx.const_like_dt(sget(st, "x"), CV::Bool(false), Some(crate::dtype::BOOL)))
        })
    }));
    rules.push((var("x").mod_(UPatOrConst::Pat(var("x"))), {
        r(|ctx, st, _| Some(ctx.const_like(sget(st, "x"), CV::Int(0))))
    }));
    rules.push((var("x").xor(UPatOrConst::Pat(var("x"))), {
        r(|ctx, st, _| Some(ctx.const_like(sget(st, "x"), CV::Int(0))))
    }));
    rules.push((var("x").and_(UPatOrConst::Val(CV::Int(0))), {
        r(|ctx, st, _| Some(ctx.const_like(sget(st, "x"), CV::Int(0))))
    }));
    // (x&mask)>>k -> x>>k when mask only clears bits below k
    rules.push((
        var("x")
            .and_(UPatOrConst::Pat(cvar("mask")))
            .shr(UPatOrConst::Pat(cvar("k"))),
        r(|ctx, st, _| {
            let (x, mask, k) = (sget(st, "x"), sget(st, "mask"), sget(st, "k"));
            let m = ctx.val(&mask).as_int_exact()?;
            let kk = ctx.val(&k).as_int_exact()?;
            if m | ((1i128 << kk.min(126)) - 1) == -1 {
                let kn = ctx.const_py(CV::Int(kk), None);
                Some(ctx.shr(x, kn))
            } else {
                None
            }
        }),
    ));
    // (x&mask)//c -> x//c when c is a power of 2 covering mask
    rules.push((
        var("x")
            .and_(UPatOrConst::Pat(cvar("mask")))
            .floordiv(UPatOrConst::Pat(cvar("c"))),
        r(|ctx, st, _| {
            let (x, mask, c) = (sget(st, "x"), sget(st, "mask"), sget(st, "c"));
            let m = ctx.val(&mask).as_int_exact()?;
            let cv = ctx.val(&c).as_int_exact()?;
            if cv > 0 && cv & (cv - 1) == 0 && m | (cv - 1) == -1 {
                let cn = ctx.const_py(CV::Int(cv), None);
                Some(ctx.div_floor(x, cn))
            } else {
                None
            }
        }),
    ));
    rules.push((
        var_dt("x", ints_plus_weak()).ne(UPatOrConst::Pat(var("x"))),
        r(|ctx, st, _| {
            Some(ctx.const_like_dt(sget(st, "x"), CV::Bool(false), Some(crate::dtype::BOOL)))
        }),
    ));
    // ** constant folding **
    rules.push((
        UPat::ops1(Ops::Cast)
            .with_dtype(Some(dtype::ALL.to_vec()))
            .named("root")
            .with_src_tup(vec![cvar("c")]),
        r(|ctx, st, _| {
            let (root, c) = (sget(st, "root"), sget(st, "c"));
            let dt = ctx.dt_of(&root);
            Some(ctx.const_like(root, dtype::dtype_const(dt, ctx.val(&c))))
        }),
    ));
    rules.push((
        UPat::op_var(group::UNARY.to_vec())
            .with_src_tup(vec![const_folding_pat()])
            .named("a"),
        r(|ctx, st, _| fold_const_alu(ctx, sget(st, "a"))),
    ));
    rules.push((
        UPat::op_var(opset_diff(group::BINARY, &[Ops::Threefry]))
            .with_src_list(vec![const_folding_pat(), const_folding_pat()])
            .named("a"),
        r(|ctx, st, _| fold_const_alu(ctx, sget(st, "a"))),
    ));
    rules.push((
        UPat::op_var(group::TERNARY.to_vec())
            .with_src_tup(vec![
                const_folding_pat(),
                const_folding_pat(),
                const_folding_pat(),
            ])
            .named("a"),
        r(|ctx, st, _| fold_const_alu(ctx, sget(st, "a"))),
    ));
    // bool MUL is AND, ADD/MAX is OR
    rules.push((
        var_dt("x", vec![crate::dtype::BOOL])
            .mul(UPatOrConst::Pat(var_dt("y", vec![crate::dtype::BOOL]))),
        r(|ctx, st, _| Some(ctx.and_(sget(st, "x"), sget(st, "y")))),
    ));
    rules.push((
        var_dt("x", vec![crate::dtype::BOOL])
            .add(UPatOrConst::Pat(var_dt("y", vec![crate::dtype::BOOL]))),
        r(|ctx, st, _| Some(ctx.or_(sget(st, "x"), sget(st, "y")))),
    ));
    rules.push((
        var_dt("x", vec![crate::dtype::BOOL])
            .maximum(UPatOrConst::Pat(var_dt("y", vec![crate::dtype::BOOL]))),
        r(|ctx, st, _| Some(ctx.or_(sget(st, "x"), sget(st, "y")))),
    ));
    // *** div rules ***
    rules.push((
        cvar_arg("x", CV::Int(0)).truediv(UPatOrConst::Val(CV::Int(0))),
        r(|ctx, st, _| Some(ctx.const_like(sget(st, "x"), CV::Float(f64::NAN)))),
    ));
    rules.push((
        var("x")
            .mul(UPatOrConst::Val(CV::Int(0)))
            .truediv(UPatOrConst::Val(CV::Int(0))),
        r(|ctx, st, _| Some(ctx.const_like(sget(st, "x"), CV::Float(f64::NAN)))),
    ));
    rules.push((var("x").truediv(UPatOrConst::Pat(var("x"))), {
        r(|ctx, st, _| Some(ctx.const_like(sget(st, "x"), CV::Int(1))))
    }));
    rules.push((
        var("x")
            .mul(UPatOrConst::Pat(var("x2")))
            .truediv(UPatOrConst::Pat(var("x2"))),
        r(|_, st, _| Some(sget(st, "x"))),
    ));
    // x*0 -> 0 (nan-aware)
    rules.push((
        var("x").mul(UPatOrConst::Val(CV::Int(0))),
        r(|ctx, st, _| {
            let x = sget(st, "x");
            if ctx.op_of(&x) == Ops::Const
                && let CV::Float(f) = ctx.val(&x)
                && (f.is_nan() || f.is_infinite())
            {
                return Some(ctx.const_like(x, CV::Float(f64::NAN)));
            }
            Some(ctx.const_like(x, CV::Int(0)))
        }),
    ));
    // *** cast/bitcast ***
    rules.push((
        UPat::op_var(vec![Ops::Cast, Ops::Bitcast]).named("root"),
        r(|ctx, st, _| {
            let root = sget(st, "root");
            let srcs = ctx.srcs(&root).to_vec();
            if ctx.dt_of(&root) == ctx.dt_of(&srcs[0]) {
                Some((srcs[0]).clone())
            } else {
                None
            }
        }),
    ));
    rules.push((
        UPat::ops1(Ops::Bitcast)
            .named("root")
            .with_src_tup(vec![cvar("c")]),
        r(|ctx, st, _| fold_bitcast(ctx, sget(st, "root"), sget(st, "c"))),
    ));
    // b.cast(a).cast(b) -> b if a preserves all values in b
    rules.push((
        var("x")
            .cast(Option::<DTypeId>::None)
            .named("a")
            .cast(Option::<DTypeId>::None)
            .named("b"),
        r(|ctx, st, _| {
            let (x, a, b) = (sget(st, "x"), sget(st, "a"), sget(st, "b"));
            if ctx.dt_of(&x) == ctx.dt_of(&b) && can_lossless_cast(ctx.dt_of(&b), ctx.dt_of(&a)) {
                Some(x)
            } else {
                None
            }
        }),
    ));
    // bitcast twice
    rules.push((
        UPat::ops1(Ops::Bitcast)
            .named("b")
            .with_src_tup(vec![var("x").bitcast(Option::<DTypeId>::None)]),
        r(|ctx, st, _| {
            let (x, b) = (sget(st, "x"), sget(st, "b"));
            let dt = ctx.dt_of(&b);
            Some(ctx.bitcast(x, dt))
        }),
    ));
    rules.push((
        var("x").cast(Some(crate::dtype::BOOL)),
        r(|ctx, st, _| {
            let x = sget(st, "x");
            let zero = ctx.const_py(CV::Int(0), None);
            Some(ctx.ne(x, zero))
        }),
    ));
    // ** pow **
    rules.push((
        var("x").alu(Ops::Pow, vec![cvar("c")]),
        r(|ctx, st, _| simplify_pow(ctx, sget(st, "x"), sget(st, "c"))),
    ));
    rules.push((
        cvar("c").alu(Ops::Pow, vec![var("x")]),
        r(|ctx, st, _| {
            let (c, x) = (sget(st, "c"), sget(st, "x"));
            let cv = ctx.val(&c).as_f64();
            if cv == 1.0 {
                return Some(c);
            }
            if cv > 0.0 {
                let k = ctx.const_py(CV::Float(cv.log2()), None);
                let m = ctx.mul(x, k);
                return Some(ctx.exp2(m));
            }
            None
        }),
    ));
    // unpack a uint64 packed from two uint32 (threefry):
    // `((x:uint64 << 32) | y:uint32.cast(uint64)).cast(uint32) -> y`.
    // Mirror symbolic.py:179 exactly: the named uint32 operand is the *second*
    // (low) word, so the u32 truncation returns it.
    rules.push((
        UPat::var(None, Some(vec![crate::dtype::UINT64]))
            .shl(UPatOrConst::Val(CV::Int(32)))
            .or_(UPatOrConst::Pat(
                var_dt("y", vec![crate::dtype::UINT32]).cast(Some(crate::dtype::UINT64)),
            ))
            .cast(Some(crate::dtype::UINT32)),
        r(|_, st, _| Some(sget(st, "y"))),
    ));
    rules.push((
        var_dt("x", vec![crate::dtype::UINT32])
            .cast(Some(crate::dtype::UINT64))
            .shl(UPatOrConst::Val(CV::Int(32)))
            .or_(UPatOrConst::Pat(UPat::var(
                None,
                Some(vec![crate::dtype::UINT64]),
            )))
            .shr(UPatOrConst::Val(CV::Int(32))),
        r(|ctx, st, _| {
            let x = sget(st, "x");
            Some(ctx.cast(x, crate::dtype::UINT64))
        }),
    ));
    // ** simple where folding **
    rules.push((
        UPat::var(None, None).where_(UPatOrConst::Pat(var("val")), UPatOrConst::Pat(var("val"))),
        r(|_, st, _| Some(sget(st, "val"))),
    ));
    rules.push((
        cvar("gate")
            .where_(UPatOrConst::Pat(var("c0")), UPatOrConst::Pat(var("c1")))
            .named("w"),
        r(|ctx, st, _| {
            fold_const_where(
                ctx,
                sget(st, "gate"),
                sget(st, "c0"),
                sget(st, "c1"),
                sget(st, "w"),
            )
        }),
    ));
    rules.push((
        var("a").where_(
            UPatOrConst::Pat(
                var("b").where_(UPatOrConst::Pat(var("c")), UPatOrConst::Pat(var("d"))),
            ),
            UPatOrConst::Pat(var("d")),
        ),
        r(|ctx, st, _| {
            let (a, b, c, d) = (sget(st, "a"), sget(st, "b"), sget(st, "c"), sget(st, "d"));
            let g = ctx.and_(a, b);
            Some(ctx.where_(g, c, d))
        }),
    ));
    rules.push((
        var("a").where_(
            UPatOrConst::Pat(var("c")),
            UPatOrConst::Pat(
                var("b").where_(UPatOrConst::Pat(var("c")), UPatOrConst::Pat(var("d"))),
            ),
        ),
        r(|ctx, st, _| {
            let (a, b, c, d) = (sget(st, "a"), sget(st, "b"), sget(st, "c"), sget(st, "d"));
            let g = ctx.or_(a, b);
            Some(ctx.where_(g, c, d))
        }),
    ));
    let mut pm = pm_data_invalid();
    pm = pm.add(&PatternMatcher::new(rules));
    pm = pm.add(pm_mop_cleanup());
    pm
}

// ---------- phase 2 (symbolic.py:193-309) ----------

/// `lt_folding` (symbolic.py:195-199).
fn lt_folding(ctx: &mut UOpCtx, x: UopId, c: i128) -> Option<UopId> {
    let terms = ctx.split_uop_vec(x, Ops::Add);
    let mut p: Vec<UopId> = Vec::new();
    let mut np: Vec<UopId> = Vec::new();
    for u in terms {
        if ctx.const_factor(u.clone()) == 1 {
            p.push(u);
        } else {
            np.push(u);
        }
    }
    if np.is_empty() {
        return None;
    }
    let factors: Vec<i128> = np
        .iter()
        .map(|u| ctx.const_factor(u.clone()))
        .chain(std::iter::once(c))
        .collect();
    let d = gcd_all_i128(&factors);
    if d > 1 {
        // 0 <= sum(u.vmin for u in p) and sum(u.vmax for u in p) < d
        let mut smin = 0i128;
        let mut smax = 0i128;
        for u in &p {
            let (a, b) = ctx.min_max(u).ok()?;
            smin += a.as_i128_or_panic();
            smax += b.as_i128_or_panic();
        }
        if 0 <= smin && smax < d {
            let first = np.first().unwrap().clone();
            let rest = &np[1..];
            let mut acc = if rest.is_empty() {
                first
            } else {
                ctx.usum(first, rest)
            };
            acc = ctx.divides(acc, d)?;
            let bound = ctx.const_py(CV::Int(c / d), None);
            return Some(ctx.lt(acc, bound));
        }
    }
    None
}

/// `canonicalize_simplex` (symbolic.py:201-212).
fn canonicalize_simplex(ctx: &mut UOpCtx, x: UopId) -> Option<UopId> {
    let mut changed = false;
    let mut ret: Vec<UopId> = Vec::new();
    for u in ctx.split_uop_vec(x, Ops::Add) {
        let mut u = u;
        let (_, _, srcs, _, _) = ctx.snapshot(&u);
        if ctx.op_of(&u) == Ops::Mul && srcs.len() == 2 && ctx.op_of(&srcs[1]) == Ops::Const {
            let cv = ctx.val(&srcs[1]).as_int_exact().unwrap_or(0);
            if cv > 0 {
                changed = true;
                u = (srcs[0]).clone();
            }
        }
        if !group::in_(group::IRREDUCIBLE, ctx.op_of(&u)) {
            return None;
        }
        let vmin = ctx.vmin(&u).ok()?;
        if vmin.as_i128_or_panic() < 0 {
            return None;
        }
        ret.push(u);
    }
    if changed {
        let first = ret.first().unwrap().clone();
        let rest = &ret[1..];
        Some(if rest.is_empty() {
            first
        } else {
            ctx.usum(first, rest)
        })
    } else {
        None
    }
}

/// `_quotient_base` (symbolic.py:40-50).
fn quotient_base(ctx: &mut UOpCtx, q: UopId, base: UopId, div: i128) -> Option<UopId> {
    let (qs, s) = ctx.pop_const(q, Ops::Add);
    let (num, a) = ctx.pop_const(base.clone(), Ops::Add);
    let (qop, _, qsrcs, _, _) = ctx.snapshot(&qs);
    if qop != Ops::Floordiv || ctx.op_of(&qsrcs[1]) != Ops::Const {
        return None;
    }
    let (n_op, _, nsrcs, _, _) = ctx.snapshot(&num);
    let mut num_s = num;
    let mut a_v = a.as_int_exact()?;
    let d_big;
    let qdiv = ctx.val(&qsrcs[1]).as_int_exact()?;
    if div > 0 && nsrcs.len() == 2 && n_op == Ops::Floordiv {
        let c = ctx.val(&nsrcs[1]).as_int_exact()?;
        if qdiv == c * div {
            num_s = (nsrcs[0]).clone();
            a_v *= c;
            d_big = c * div;
        } else if qdiv == div {
            d_big = div;
        } else {
            return None;
        }
    } else if qdiv == div {
        d_big = div;
    } else {
        return None;
    }
    let (xs, xa) = ctx.pop_const(num_s, Ops::Add);
    let (p, pa) = ctx.pop_const((qsrcs[0]).clone(), Ops::Add);
    if p != xs {
        return None;
    }
    let t = xa.as_int_exact()? + a_v - pa.as_int_exact()?;
    if t % d_big != 0 {
        return None;
    }
    // python: `k = t//D - s` (floor division; t may be a nonzero multiple of D)
    let k = t.div_euclid(d_big) - s.as_int_exact()?;
    if k == 0 {
        Some(base)
    } else {
        // base - k*div
        let kd = ctx.const_py(CV::Int(k * div), None);
        Some(ctx.sub(base, kd))
    }
}

/// `fold_add_divmod_recombine` (symbolic.py:52-68) — full port.
fn fold_add_divmod_recombine(ctx: &mut UOpCtx, x: UopId) -> Option<UopId> {
    let terms = ctx.split_uop_vec(x, Ops::Add);
    for (i, u) in terms.iter().enumerate() {
        let (mod_, mul) = ctx.pop_const(u.clone(), Ops::Mul);
        let (mop, _, msrcs, _, _) = ctx.snapshot(&mod_);
        if mop != Ops::Floormod || ctx.op_of(&msrcs[1]) != Ops::Const {
            continue;
        }
        let (base, div) = ((msrcs[0]).clone(), ctx.val(&msrcs[1]).as_int_exact()?);
        for (j, v) in terms.iter().enumerate() {
            let (q, scale) = ctx.pop_const(v.clone(), Ops::Mul);
            if i == j {
                continue;
            }
            let scale_v = scale.as_int_exact()?;
            if scale_v != div * mul.as_int_exact().unwrap_or(1) {
                continue;
            }
            let mut rest: Vec<UopId> = Vec::new();
            for (k, t) in terms.iter().enumerate() {
                if k != i && k != j {
                    rest.push(t.clone());
                }
            }
            if let Some(b) = quotient_base(ctx, q.clone(), base.clone(), div) {
                // (b*mul).usum(*rest)
                let mk = ctx.const_like(v.clone(), mul);
                let bm = ctx.mul(b, mk);
                return Some(if rest.is_empty() {
                    bm
                } else {
                    ctx.usum(bm, &rest)
                });
            }
            let (qop, _, qsrcs, _, _) = ctx.snapshot(&q);
            if qop == Ops::Floormod && ctx.op_of(&qsrcs[1]) == Ops::Const {
                let d = ctx.val(&qsrcs[1]).as_int_exact()?;
                if d > 0
                    && let Some(b) = quotient_base(ctx, (qsrcs[0]).clone(), base.clone(), div)
                {
                    // python: `(b % (div*d))*mul`
                    let dv = ctx.const_py(CV::Int(d * div), None);
                    let bm = ctx.mod_(b, dv);
                    let mk2 = ctx.const_like(v.clone(), mul);
                    let bm = ctx.mul(bm, mk2);
                    return Some(if rest.is_empty() {
                        bm
                    } else {
                        ctx.usum(bm, &rest)
                    });
                }
            }
        }
    }
    None
}

/// `fold_where_closure` (symbolic.py:221-226).
fn fold_where_closure(ctx: &mut UOpCtx, cond: UopId, t: UopId, f: UopId) -> Option<UopId> {
    let t_slice = ctx.backward_slice_with_self(&t);
    let f_slice = ctx.backward_slice_with_self(&f);
    if !t_slice.contains(&cond) && !f_slice.contains(&cond) {
        return None;
    }
    // INDEX gates are owned by the valid/store-coalescing machinery
    let index_ops = [Ops::Index];
    if ctx.op_in_backward_slice_with_self(&cond, &index_ops)
        || ctx.op_in_backward_slice_with_self(&t, &index_ops)
        || ctx.op_in_backward_slice_with_self(&f, &index_ops)
    {
        return None;
    }
    let mut d1 = std::collections::HashMap::new();
    d1.insert(cond.clone(), ctx.const_like(cond.clone(), CV::Bool(true)));
    let mut d2 = std::collections::HashMap::new();
    d2.insert(cond.clone(), ctx.const_like(cond.clone(), CV::Bool(false)));
    let t2 = substitute(ctx, t, d1, false, false);
    let f2 = substitute(ctx, f, d2, false, false);
    Some(ctx.where_(cond, t2, f2))
}

fn gcd_all_i128(vals: &[i128]) -> i128 {
    let g = |a: i128, b: i128| {
        let (mut a, mut b) = (a.abs(), b.abs());
        while b != 0 {
            let t = a % b;
            a = b;
            b = t;
        }
        a
    };
    vals.iter().cloned().fold(0, g)
}

fn build_symbolic_phase2() -> PatternMatcher {
    let weak_int = || vec![WEAKINT];
    let ints_weak = || ints_plus_weak();
    let mut rules: Vec<(UPat, RuleFn)> = Vec::new();
    // ** boolean algebra **
    rules.push((
        var_dt("x", vec![crate::dtype::BOOL]).or_(UPatOrConst::Pat(
            var_dt("x", vec![crate::dtype::BOOL]).logical_not(),
        )),
        r(|ctx, st, _| Some(ctx.const_like(sget(st, "x"), CV::Bool(true)))),
    ));
    // ** combine terms **
    rules.push((
        var("x")
            .mul(UPatOrConst::Pat(cvar("c0")))
            .add(var("x").mul(UPatOrConst::Pat(cvar("c1")))),
        r(|ctx, st, _| {
            let (x, c0, c1) = (sget(st, "x"), sget(st, "c0"), sget(st, "c1"));
            let s = const_add(ctx, ctx.val(&c0), ctx.val(&c1));
            {
                let k = ctx.const_like(x.clone(), s);
                Some(ctx.mul(x, k))
            }
        }),
    ));
    rules.push((
        var("y")
            .add(UPatOrConst::Pat(var("x").mul(UPatOrConst::Pat(cvar("c0")))))
            .add(UPatOrConst::Pat(var("x").mul(UPatOrConst::Pat(cvar("c1"))))),
        r(|ctx, st, _| {
            let (x, y, c0, c1) = (sget(st, "x"), sget(st, "y"), sget(st, "c0"), sget(st, "c1"));
            let s = const_add(ctx, ctx.val(&c0), ctx.val(&c1));
            let k = ctx.const_like(x.clone(), s);
            let xm = ctx.mul(x, k);
            Some(ctx.add(y, xm))
        }),
    ));
    rules.push((
        var("x").add(var("x").mul(UPatOrConst::Pat(cvar("c")))),
        r(|ctx, st, _| {
            let (x, c) = (sget(st, "x"), sget(st, "c"));
            let s = const_add(ctx, ctx.val(&c), CV::Int(1));
            {
                let k = ctx.const_like(x.clone(), s);
                Some(ctx.mul(x, k))
            }
        }),
    ));
    rules.push((
        var("y")
            .add(UPatOrConst::Pat(var("x")))
            .add(UPatOrConst::Pat(var("x").mul(UPatOrConst::Pat(cvar("c"))))),
        r(|ctx, st, _| {
            let (x, y, c) = (sget(st, "x"), sget(st, "y"), sget(st, "c"));
            let s = const_add(ctx, ctx.val(&c), CV::Int(1));
            let k = ctx.const_like(x.clone(), s);
            let xm = ctx.mul(x, k);
            Some(ctx.add(y, xm))
        }),
    ));
    rules.push((
        var("y")
            .add(UPatOrConst::Pat(var("x").mul(UPatOrConst::Pat(cvar("c")))))
            .add(UPatOrConst::Pat(var("x"))),
        r(|ctx, st, _| {
            let (x, y, c) = (sget(st, "x"), sget(st, "y"), sget(st, "c"));
            let s = const_add(ctx, ctx.val(&c), CV::Int(1));
            let k = ctx.const_like(x.clone(), s);
            let xm = ctx.mul(x, k);
            Some(ctx.add(y, xm))
        }),
    ));
    rules.push((
        var("x").add(var("x")),
        r(|ctx, st, _| {
            let x = sget(st, "x");
            let two = ctx.const_like(x.clone(), CV::Int(2));
            Some(ctx.mul(x, two))
        }),
    ));
    rules.push((
        var("y").add(var("x")).add(var("x")),
        r(|ctx, st, _| {
            let (x, y) = (sget(st, "x"), sget(st, "y"));
            let two = ctx.const_like(x.clone(), CV::Int(2));
            let xm = ctx.mul(x, two);
            Some(ctx.add(y, xm))
        }),
    ));
    rules.push((
        var("x").truediv(var("x2")).truediv(var("x3")),
        r(|ctx, st, _| {
            let (x, x2, x3) = (sget(st, "x"), sget(st, "x2"), sget(st, "x3"));
            if x2 != x3 {
                let d = ctx.mul(x2, x3);
                Some(ctx.div_floor(x, d))
            } else {
                None
            }
        }),
    ));
    rules.push((
        cvar_arg("neg1", CV::Int(-1))
            .mul(UPatOrConst::Pat(var("x").add(UPatOrConst::Pat(cvar("c"))))),
        r(|ctx, st, _| {
            let (x, c) = (sget(st, "x"), sget(st, "c"));
            let nx = ctx.neg(x.clone());
            let ck = ctx.const_like(x, ctx.val(&c));
            let nc = ctx.neg(ck);
            Some(ctx.add(nx, nc))
        }),
    ));
    rules.push((
        cvar("y").mul(UPatOrConst::Pat(
            var_dt("x", weak_int()).add(UPatOrConst::Pat(cvar("c"))),
        )),
        r(|ctx, st, _| {
            let (x, y, c) = (sget(st, "x"), sget(st, "y"), sget(st, "c"));
            let yx = ctx.mul(y.clone(), x.clone());
            let ck = ctx.const_like(x, ctx.val(&c));
            let yc = ctx.mul(y, ck);
            Some(ctx.add(yx, yc))
        }),
    ));
    // ** where folding **
    rules.push((
        var_dt("cond", vec![crate::dtype::BOOL])
            .logical_not()
            .where_(UPatOrConst::Pat(var("t")), UPatOrConst::Pat(var("f"))),
        r(|ctx, st, _| {
            let (cond, t, f) = (sget(st, "cond"), sget(st, "t"), sget(st, "f"));
            if !is_invalid_node(ctx, f.clone()) {
                // python: cond.where(f, t) — the branches swap
                Some(ctx.where_(cond, f, t))
            } else {
                None
            }
        }),
    ));
    rules.push((
        var_dt("cond", vec![crate::dtype::BOOL])
            .where_(UPatOrConst::Pat(var("t")), UPatOrConst::Pat(var("f"))),
        r(|ctx, st, _| fold_where_closure(ctx, sget(st, "cond"), sget(st, "t"), sget(st, "f"))),
    ));
    // alu of two where with same conds
    rules.push((
        UPat::op_var(group::BINARY.to_vec())
            .named("alu")
            .with_src_tup(vec![
                var("c").where_(UPatOrConst::Pat(var("t")), UPatOrConst::Pat(var("f"))),
                var("c").where_(UPatOrConst::Pat(var("tt")), UPatOrConst::Pat(var("ff"))),
            ]),
        r(|ctx, st, _| {
            let (alu, c, t, tt, f, ff) = (
                sget(st, "alu"),
                sget(st, "c"),
                sget(st, "t"),
                sget(st, "tt"),
                sget(st, "f"),
                sget(st, "ff"),
            );
            let tt_const = ctx.op_of(&tt) == Ops::Const;
            let t_const = ctx.op_of(&t) == Ops::Const;
            let f_const = ctx.op_of(&f) == Ops::Const;
            let ff_const = ctx.op_of(&ff) == Ops::Const;
            if (t_const && tt_const) || (f_const && ff_const) {
                let lhs = ctx.alu2(ctx.op_of(&alu), t, tt);
                let rhs = ctx.alu2(ctx.op_of(&alu), f, ff);
                return Some(ctx.where_(c, lhs, rhs));
            }
            None
        }),
    ));
    rules.push((
        var("y")
            .add(UPatOrConst::Pat(var("c").where_(
                UPatOrConst::Pat(var("t")),
                UPatOrConst::Pat(var("f")),
            )))
            .add(UPatOrConst::Pat(var("c").where_(
                UPatOrConst::Pat(var("tt")),
                UPatOrConst::Pat(var("ff")),
            ))),
        r(|ctx, st, _| {
            let (y, c, t, tt, f, ff) = (
                sget(st, "y"),
                sget(st, "c"),
                sget(st, "t"),
                sget(st, "tt"),
                sget(st, "f"),
                sget(st, "ff"),
            );
            let tt_const = ctx.op_of(&tt) == Ops::Const;
            let t_const = ctx.op_of(&t) == Ops::Const;
            let f_const = ctx.op_of(&f) == Ops::Const;
            let ff_const = ctx.op_of(&ff) == Ops::Const;
            if (t_const && tt_const) || (f_const && ff_const) {
                let lhs = ctx.add(t, tt);
                let rhs = ctx.add(f, ff);
                let w = ctx.where_(c, lhs, rhs);
                return Some(ctx.add(y, w));
            }
            None
        }),
    ));
    rules.push((
        var("c")
            .where_(UPatOrConst::Pat(var("t")), UPatOrConst::Val(CV::Int(0)))
            .add(UPatOrConst::Pat(var("c").where_(
                UPatOrConst::Val(CV::Int(0)),
                UPatOrConst::Pat(var("f")),
            ))),
        r(|ctx, st, _| {
            let (c, t, f) = (sget(st, "c"), sget(st, "t"), sget(st, "f"));
            Some(ctx.where_(c, t, f))
        }),
    ));
    // ALU/variable min==max -> CONST
    rules.push((
        UPat::op_var(vec![
            Ops::Cmplt,
            Ops::Cmpne,
            Ops::Floordiv,
            Ops::Floormod,
            Ops::Param,
            Ops::After,
            Ops::Special,
        ])
        .named("x"),
        r(|ctx, st, _| {
            let x = sget(st, "x");
            let (vmin, vmax) = ctx.min_max(&x).ok()?;
            if vmin == vmax {
                Some(ctx.const_like(x, sval_to_const(vmin)))
            } else {
                None
            }
        }),
    ));
    rules.push((
        UPat::ops1(Ops::Range)
            .with_src_tup(vec![UPat::ops1(Ops::Const)])
            .named("x"),
        r(|ctx, st, _| {
            let x = sget(st, "x");
            let (vmin, vmax) = ctx.min_max(&x).ok()?;
            if vmin == vmax {
                Some(ctx.const_like(x, sval_to_const(vmin)))
            } else {
                None
            }
        }),
    ));
    // max folding
    rules.push((
        cvar("a")
            .lt(UPatOrConst::Pat(var("b")))
            .where_(UPatOrConst::Pat(var("b")), UPatOrConst::Pat(cvar("c"))),
        r(|ctx, st, _| {
            let (a, b, c) = (sget(st, "a"), sget(st, "b"), sget(st, "c"));
            if ctx.val(&a) == ctx.val(&c) {
                Some(ctx.maximum(a, b))
            } else {
                None
            }
        }),
    ));
    rules.push((
        var("a")
            .lt(UPatOrConst::Pat(cvar("b")))
            .where_(UPatOrConst::Pat(cvar("c")), UPatOrConst::Pat(var("a"))),
        r(|ctx, st, _| {
            let (a, b, c) = (sget(st, "a"), sget(st, "b"), sget(st, "c"));
            if ctx.val(&b) == ctx.val(&c) {
                Some(ctx.maximum(a, b))
            } else {
                None
            }
        }),
    ));
    rules.push((
        var("x").maximum(UPatOrConst::Pat(var("y"))),
        r(|ctx, st, _| {
            let (x, y) = (sget(st, "x"), sget(st, "y"));
            let (xmin, xmax) = ctx.min_max(&x).ok()?;
            let (ymin, ymax) = ctx.min_max(&y).ok()?;
            if sval_cmp(xmin, ymax) != std::cmp::Ordering::Less {
                Some(x)
            } else if sval_cmp(xmax, ymin) != std::cmp::Ordering::Greater {
                Some(y)
            } else {
                None
            }
        }),
    ));
    // ** two stage ALU folding ** (Associative ops)
    for op in group::ASSOCIATIVE {
        rules.push((
            var("x")
                .alu(*op, vec![cvar("c1")])
                .alu(*op, vec![cvar("c2")])
                .named("f"),
            r(|ctx, st, _| {
                let (f, x, c1, c2) = (sget(st, "f"), sget(st, "x"), sget(st, "c1"), sget(st, "c2"));
                let inner = ctx.alu2(ctx.op_of(&f), c1, c2);
                Some(ctx.alu2(ctx.op_of(&f), x, inner))
            }),
        ));
    }
    // (x//c1)//c2 -> x//(c1*c2) for c2>0
    rules.push((
        var("x").floordiv(cvar("c1")).floordiv(cvar("c2")),
        r(|ctx, st, _| {
            let (x, c1, c2) = (sget(st, "x"), sget(st, "c1"), sget(st, "c2"));
            let (c2min, _) = ctx.min_max(&c2).ok()?;
            if c2min.as_i128_or_panic() > 0 {
                let k1 = ctx.const_like(x.clone(), ctx.val(&c1));
                let k2 = ctx.const_like(x.clone(), ctx.val(&c2));
                let prod = ctx.mul(k1, k2);
                return Some(ctx.div_floor(x, prod));
            }
            None
        }),
    ));
    // ** lt **
    rules.push((
        cvar("c0").add(var_dt("x", ints_weak())).lt(cvar("c1")),
        r(|ctx, st, _| {
            let (x, c0, c1) = (sget(st, "x"), sget(st, "c0"), sget(st, "c1"));
            let a = ctx.val(&c0).as_int_exact()?;
            let b = ctx.val(&c1).as_int_exact()?;
            let k = ctx.const_py(CV::Int(b - a), None);
            Some(ctx.lt(x, k))
        }),
    ));
    rules.push((
        cvar("c0").mul(var_dt("x", weak_int())).lt(cvar("c1")),
        r(|ctx, st, _| {
            let (x, c0, c1) = (sget(st, "x"), sget(st, "c0"), sget(st, "c1"));
            let a = ctx.val(&c0).as_int_exact()?;
            let b = ctx.val(&c1).as_int_exact()?;
            if a.abs() > 1 {
                let newx = if a > 0 { x } else { ctx.neg(x) };
                let bound = -(-b / a.abs());
                let k = ctx.const_py(CV::Int(bound), None);
                return Some(ctx.lt(newx, k));
            }
            None
        }),
    ));
    rules.push((
        var_dt("x", weak_int()).floordiv(cvar("d")).lt(cvar("c")),
        r(|ctx, st, _| {
            let (x, d, c) = (sget(st, "x"), sget(st, "d"), sget(st, "c"));
            let dv = ctx.val(&d).as_int_exact()?;
            let cv = ctx.val(&c).as_int_exact()?;
            if dv > 0 {
                let k = ctx.const_py(CV::Int(cv * dv), None);
                Some(ctx.lt(x, k))
            } else if dv < 0 {
                let k = ctx.const_py(CV::Int(cv * dv), None);
                Some(ctx.gt(x, k))
            } else {
                None
            }
        }),
    ));
    // ** move add/mul consts to end **
    rules.push((
        var("x")
            .add(UPatOrConst::Pat(cvar("c1")))
            .add(UPatOrConst::Pat(var("y"))),
        r(|ctx, st, _| {
            let (x, c1, y) = (sget(st, "x"), sget(st, "c1"), sget(st, "y"));
            if ctx.op_of(&y) != Ops::Const {
                let xy = ctx.add(x.clone(), y);
                {
                    let ck = ctx.const_like(x, ctx.val(&c1));
                    return Some(ctx.add(xy, ck));
                }
            }
            None
        }),
    ));
    rules.push((
        var("x")
            .mul(UPatOrConst::Pat(cvar("c1")))
            .mul(UPatOrConst::Pat(var("y"))),
        r(|ctx, st, _| {
            let (x, c1, y) = (sget(st, "x"), sget(st, "c1"), sget(st, "y"));
            if ctx.op_of(&y) != Ops::Const {
                let xy = ctx.mul(x.clone(), y);
                {
                    let ck = ctx.const_like(x, ctx.val(&c1));
                    return Some(ctx.mul(xy, ck));
                }
            }
            None
        }),
    ));
    // *** rules from symbolic ***
    rules.push((
        var_dt("x", weak_int()).lt(cvar("c")),
        r(|ctx, st, _| {
            let (x, c) = (sget(st, "x"), sget(st, "c"));
            let cv = ctx.val(&c).as_int_exact()?;
            if cv > 0 { lt_folding(ctx, x, cv) } else { None }
        }),
    ));
    rules.push((
        var_dt("x", weak_int())
            .mul(UPatOrConst::Val(CV::Int(-1)))
            .lt(UPatOrConst::Pat(
                var("y").mul(UPatOrConst::Val(CV::Int(-1))),
            )),
        r(|ctx, st, _| {
            let (x, y) = (sget(st, "x"), sget(st, "y"));
            Some(ctx.lt(y, x))
        }),
    ));
    // canonicalize a simplex
    rules.push((
        var_dt("x", weak_int())
            .lt(UPatOrConst::Val(CV::Int(1)))
            .ne(UPatOrConst::Val(CV::Bool(true))),
        r(|ctx, st, _| {
            let x = sget(st, "x");
            if let Some(newx) = canonicalize_simplex(ctx, x) {
                let one = ctx.const_like(newx.clone(), CV::Int(1));
                let lt = ctx.lt(newx, one);
                {
                    let tr = ctx.const_like(lt.clone(), CV::Bool(true));
                    return Some(ctx.ne(lt, tr));
                }
            }
            None
        }),
    ));
    // a range mod its own upper bound
    rules.push((
        UPat::ops1(Ops::Range)
            .with_src_tup(vec![var("end")])
            .named("r")
            .mod_(UPatOrConst::Pat(var("end"))),
        r(|_, st, _| Some(sget(st, "r"))),
    ));
    rules.push((
        UPat::ops1(Ops::Range)
            .with_src_tup(vec![var("end")])
            .named("r")
            .floordiv(UPatOrConst::Pat(var("end"))),
        r(|ctx, st, _| Some(ctx.const_like(sget(st, "r"), CV::Int(0)))),
    ));
    // cast/long folding
    rules.push((
        var("x")
            .cast(Option::<DTypeId>::None)
            .named("a")
            .cast(Option::<DTypeId>::None)
            .named("b"),
        r(|ctx, st, _| {
            let (x, b) = (sget(st, "x"), sget(st, "b"));
            if can_lossless_cast(ctx.dt_of(&x), ctx.dt_of(&sget(st, "a"))) {
                Some(ctx.cast(x, ctx.dt_of(&b)))
            } else {
                None
            }
        }),
    ));
    rules.push((
        var_dt("x", ints_weak())
            .cast_dts(ints_weak())
            .named("a")
            .cast(Option::<DTypeId>::None)
            .named("b"),
        r(|ctx, st, _| {
            let (x, a, b) = (sget(st, "x"), sget(st, "a"), sget(st, "b"));
            let (xmin, xmax) = ctx.min_max(&x).ok()?;
            let (amin, amax) = ctx.dtype_bounds(ctx.dt_of(&a)).ok()?;
            if sval_cmp(xmin, amin) != std::cmp::Ordering::Less
                && sval_cmp(xmax, amax) != std::cmp::Ordering::Greater
            {
                return Some(ctx.cast(x, ctx.dt_of(&b)));
            }
            None
        }),
    ));
    // try to do math in int instead of long
    rules.push((
        UPat::op_var(group::BINARY.to_vec())
            .named("u")
            .with_src_tup(vec![
                var("x").with_dtype(Some(vec![crate::dtype::INT64, WEAKINT])),
                var("y").with_dtype(Some(vec![crate::dtype::INT64, WEAKINT])),
            ]),
        r(|ctx, st, _| {
            let (u, x, y) = (sget(st, "u"), sget(st, "x"), sget(st, "y"));
            let (xd, yd) = (ctx.dt_of(&x), ctx.dt_of(&y));
            if xd != crate::dtype::INT64 && yd != crate::dtype::INT64 {
                return None;
            }
            if overflows(ctx, x.clone(), crate::dtype::INT32)
                || overflows(ctx, y.clone(), crate::dtype::INT32)
                || overflows(ctx, u.clone(), crate::dtype::INT32)
            {
                return None;
            }
            let xi = if ctx.op_of(&x) == Ops::Const {
                ctx.const_py(ctx.val(&x), None)
            } else {
                ctx.cast(x, crate::dtype::INT32)
            };
            let yi = if ctx.op_of(&y) == Ops::Const {
                ctx.const_py(ctx.val(&y), None)
            } else {
                ctx.cast(y, crate::dtype::INT32)
            };
            let inner = ctx.alu2(ctx.op_of(&u), xi, yi);
            Some(ctx.cast(inner, ctx.dt_of(&u)))
        }),
    ));
    rules.push((
        var_dt("x", weak_int())
            .add(UPatOrConst::Pat(cvar("c")))
            .cast_dts(dtype::SINTS.to_vec())
            .named("cast"),
        r(|ctx, st, _| {
            let (x, c, cast) = (sget(st, "x"), sget(st, "c"), sget(st, "cast"));
            let cdt = ctx.dt_of(&cast);
            let x1 = ctx.cast(x, cdt);
            let k = ctx.const_like(cast, ctx.val(&c));
            Some(ctx.add(x1, k))
        }),
    ));
    // only RANGE/IF/STORE/KERNEL have side effects — AFTER passthrough
    rules.push((
        UPat::ops1(Ops::After).named("x"),
        r(|ctx, st, _| {
            let x = sget(st, "x");
            let srcs = ctx.srcs(&x).to_vec();
            let side_effect_ops = [
                Ops::Range,
                Ops::Store,
                Ops::Call,
                Ops::Function,
                Ops::Barrier,
                Ops::End,
                Ops::Linear,
                Ops::Stage,
            ];
            let mut new_srcs = vec![(srcs[0]).clone()];
            for y in &srcs[1..] {
                if group::in_(&side_effect_ops, ctx.op_of(y)) {
                    new_srcs.push(y.clone());
                } else {
                    new_srcs.extend(ctx.srcs(y).iter().cloned());
                }
            }
            // dedup
            let mut deduped: Vec<UopId> = Vec::new();
            for s in new_srcs {
                if !deduped.contains(&s) {
                    deduped.push(s);
                }
            }
            let (_, d, _, _, _) = ctx.snapshot(&x);
            Some(ctx.replace(
                &x,
                Ops::After,
                d,
                deduped,
                ctx.arg_of(&x).clone(),
                ctx.tag_of(&x).clone(),
            ))
        }),
    ));
    // after/end with 1 src is just src[0]
    rules.push((
        UPat::op_var(vec![Ops::After, Ops::End]).with_src_tup(vec![var("s")]),
        r(|_, st, _| Some(sget(st, "s"))),
    ));
    let mut pm = PatternMatcher::new(rules);
    pm = pm.add(super::divandmod::div_and_mod_symbolic());
    pm
}

/// Python `+` on const values.
fn const_add(_ctx: &mut UOpCtx, a: CV, b: CV) -> CV {
    let (x, y) = (a, b);
    match (x, y) {
        (CV::Int(p), CV::Int(q)) => CV::Int(p.wrapping_add(q)),
        (CV::Float(p), q) | (q, CV::Float(p)) => CV::Float(p + q.as_f64()),
        (p, q) => CV::Float(p.as_f64() + q.as_f64()),
    }
}

fn sval_to_const(v: SVal) -> CV {
    match v {
        SVal::I(i) => CV::Int(i),
        SVal::U(_) => CV::Invalid,
    }
}

static COMMUTATIVE_PM: PmSlot = PmSlot::new();
/// `commutative` (symbolic.py:214) — flip weakint commutative ops into canonical order.
pub fn pm_commutative() -> &'static PatternMatcher {
    COMMUTATIVE_PM.get_or_init(pm_commutative_inner)
}

static MOP_CLEANUP_PM: PmSlot = PmSlot::new();
/// `mop_cleanup` (uop/movement.py) — merge and drop redundant movement ops.
pub fn pm_mop_cleanup() -> &'static PatternMatcher {
    MOP_CLEANUP_PM.get_or_init(pm_mop_cleanup_inner)
}

// ---------- pm_clean_up_group_sink (symbolic.py:432-438) ----------
static CLEAN_UP_GROUP_SINK: PmSlot = PmSlot::new();

/// `pm_clean_up_group_sink` — inline GROUP/NOOP/STACK/SINK children into sink-like roots.
pub fn pm_clean_up_group_sink() -> &'static PatternMatcher {
    CLEAN_UP_GROUP_SINK.get_or_init(|| {
        let sink_like = [Ops::Noop, Ops::Stack, Ops::Sink, Ops::Group];
        let rules: Vec<(UPat, RuleFn)> = vec![
            // GROUP with a single src is that src
            (
                UPat::ops1(Ops::Group).with_src_tup(vec![var("x")]),
                r(|_ctx, st, _| Some(sget(st, "x"))),
            ),
            // SINK/GROUP roots: inline sink-like children
            (
                UPat::op_var(vec![Ops::Sink, Ops::Group])
                    .allow_all_srcs()
                    .named("root"),
                r(move |ctx, st, _| {
                    let root = sget(st, "root");
                    let srcs = ctx.srcs(&root).to_vec();
                    if !srcs.iter().any(|s| sink_like.contains(&ctx.op_of(s))) {
                        return None;
                    }
                    let mut new_srcs: Vec<UopId> = Vec::new();
                    for x in &srcs {
                        if sink_like.contains(&ctx.op_of(x)) {
                            new_srcs.extend(ctx.srcs(x).iter().cloned());
                        } else {
                            new_srcs.push(x.clone());
                        }
                    }
                    let (_, d, _, arg, tag) = ctx.snapshot(&root);
                    Some(ctx.replace(&root, ctx.op_of(&root), d, new_srcs, arg, tag))
                }),
            ),
        ];
        PatternMatcher::new(rules)
    })
}

// ---------- pm_simplify_valid / pm_drop_and_clauses (symbolic.py:316-429) ----------

/// `parse_valid` (symbolic.py:316) — (expr, is_upper, c).
fn parse_valid(ctx: &mut UOpCtx, v: UopId) -> Option<(UopId, bool, i64)> {
    let op = ctx.op_of(&v);
    let srcs = ctx.srcs(&v).to_vec();
    if op == Ops::Cmpne && srcs.len() == 2 {
        // (X < c).ne(True) -> X >= c
        let s1 = srcs[1].clone();
        let s0 = srcs[0].clone();
        if ctx.op_of(&s1) == Ops::Const
            && ctx.val(&s1) == ConstVal::Int(1)
            && ctx.op_of(&s0) == Ops::Cmplt
            && crate::dtype::is_int(ctx.dt_of(&ctx.srcs(&s0)[0]))
        {
            let s0s = ctx.srcs(&s0).to_vec();
            let vmin = ctx.min_max(&s0s[1]).ok()?.0;
            return Some(((s0s[0]).clone(), false, vmin.as_i64_or_panic()));
        }
    }
    if op == Ops::Cmplt && srcs.len() == 2 && crate::dtype::is_int(ctx.dt_of(&srcs[0])) {
        // c < X -> X >= c+1
        if ctx.op_of(&srcs[0]) == Ops::Const {
            let c = match ctx.val(&srcs[0]) {
                ConstVal::Int(i) => i as i64,
                _ => return None,
            };
            return Some(((srcs[1]).clone(), false, c + 1));
        }
        // X < c -> X <= c-1
        let vmax = ctx.min_max(&srcs[1]).ok()?.1;
        return Some(((srcs[0]).clone(), true, vmax.as_i64_or_panic() - 1));
    }
    None
}

/// `uop_given_valid` (symbolic.py:335), including the python loop-variable
/// leak (`expr`/`v0` from the last bounds iteration feeding try_simplex).
pub fn uop_given_valid(ctx: &mut UOpCtx, valid: UopId, mut uop: UopId, try_simplex: bool) -> UopId {
    // bounds: expr -> [lower, upper]
    let mut bound_exprs: Vec<UopId> = Vec::new();
    let mut bounds: Vec<(UopId, [Option<ConstVal>; 2])> = Vec::new();
    let mut stmts: Vec<UopId> = Vec::new();
    ctx.split_uop(valid, Ops::And, &mut stmts);
    for stmt in stmts {
        if let Some((expr, is_upper, c)) = parse_valid(ctx, stmt) {
            match bounds.iter_mut().find(|(e, _)| *e == expr) {
                Some((_, slot)) => slot[is_upper as usize] = Some(ConstVal::Int(c as i128)),
                None => {
                    bound_exprs.push(expr.clone());
                    bounds.push((expr, {
                        let mut s: [Option<ConstVal>; 2] = [None, None];
                        s[is_upper as usize] = Some(ConstVal::Int(c as i128));
                        s
                    }));
                }
            }
        }
    }

    // all_candidates
    let mut all_candidates: Vec<(UopId, UopId)> = Vec::new();
    for (i, (expr, v)) in bounds.iter().enumerate() {
        let (vmn, vmx) = ctx
            .min_max(expr)
            .unwrap_or_else(|e| panic!("TGERR: bounds vmin/vmax: {e}"));
        let v0 = match &v[0] {
            Some(c) => match *c {
                ConstVal::Int(i) => SVal::I(i),
                ConstVal::Float(f) => SVal::I(f as i128),
                _ => panic!("TGERR: bad lower bound const"),
            },
            None => vmn,
        };
        let v1 = match &v[1] {
            Some(c) => match *c {
                ConstVal::Int(i) => SVal::I(i),
                ConstVal::Float(f) => SVal::I(f as i128),
                _ => panic!("TGERR: bad upper bound const"),
            },
            None => vmx,
        };
        let fake = ctx.variable(
            &format!("fake{i}"),
            sval_cv(v0.clone()),
            sval_cv(v1),
            ctx.dt_of(expr),
            true,
        );
        all_candidates.push((expr.clone(), fake));

        if try_simplex {
            // every candidate is a set of constrained UOp based on valid, and if every item
            // in a set simplifies the uop into a same output, we rewrite uop. This runs per
            // bound (inside the loop), matching symbolic.py:344.
            let mut candidates: Vec<Vec<(UopId, UopId)>> = vec![vec![
                all_candidates.last().expect("TGERR: no bounds").clone(),
            ]];
            let irreducible = |x: UopId| group::in_(group::IRREDUCIBLE, ctx.op_of(&x));
            if ctx.op_of(expr) == Ops::Add && v0 == SVal::I(1) {
                let mut parts: Vec<UopId> = Vec::new();
                ctx.split_uop(expr.clone(), Ops::Add, &mut parts);
                if parts.iter().all(|u| irreducible(u.clone())) {
                    let cand: Vec<(UopId, UopId)> = parts
                        .iter()
                        .enumerate()
                        .map(|(j, xi)| {
                            let (_, vmx) = ctx.min_max(xi).unwrap_or((SVal::I(0), SVal::I(0)));
                            (
                                xi.clone(),
                                ctx.variable(
                                    &format!("fake{j}"),
                                    ConstVal::Int(1),
                                    sval_cv(vmx),
                                    ctx.dt_of(xi),
                                    true,
                                ),
                            )
                        })
                        .collect();
                    candidates.push(cand);
                }
            }
            for candidate in candidates {
                let back = ctx.backward_slice_with_self(&uop);
                if candidate.iter().any(|(x, _)| !back.contains(x)) {
                    continue;
                }
                let mut newuops: Vec<UopId> = Vec::new();
                for (x, newx) in &candidate {
                    let mut d = HashMap::new();
                    d.insert(x.clone(), newx.clone());
                    let sub = crate::uop::upat::substitute(ctx, uop.clone(), d.clone(), true, true);
                    let simp = crate::uop::upat::simplify(ctx, sub);
                    let mut back2 = HashMap::new();
                    back2.insert(newx.clone(), x.clone());
                    let res = crate::uop::upat::substitute(ctx, simp, back2, true, true);
                    newuops.push(crate::uop::upat::simplify(ctx, res));
                }
                let all_same = newuops.iter().all(|n| *n == newuops[0]);
                if all_same {
                    uop = (newuops[0]).clone();
                } else if ctx.op_of(&uop) == Ops::Stack && ctx.srcs(&uop).len() == 2 {
                    let s0s: Vec<UopId> = newuops.iter().map(|n| ctx.srcs(n)[0].clone()).collect();
                    let s1s: Vec<UopId> = newuops.iter().map(|n| ctx.srcs(n)[1].clone()).collect();
                    if s0s.iter().all(|n| *n == s0s[0]) {
                        let srcs = ctx.srcs(&uop).to_vec();
                        uop = ctx.uop(
                            Ops::Stack,
                            None,
                            vec![(s0s[0]).clone(), (srcs[1]).clone()],
                            Arg::None,
                        );
                    }
                    if s1s.iter().all(|n| *n == s1s[0]) {
                        let srcs = ctx.srcs(&uop).to_vec();
                        uop = ctx.uop(
                            Ops::Stack,
                            None,
                            vec![(srcs[0]).clone(), (s1s[0]).clone()],
                            Arg::None,
                        );
                    }
                }
            }
        }
    }

    // try all the valids together
    let mut sub_dict: HashMap<UopId, UopId> = HashMap::new();
    for (x, newx) in &all_candidates {
        sub_dict.insert(x.clone(), newx.clone());
    }
    let s_uop = crate::uop::upat::substitute(ctx, uop.clone(), sub_dict.clone(), true, true);
    if s_uop != uop {
        let simp = crate::uop::upat::simplify(ctx, s_uop);
        let mut back: HashMap<UopId, UopId> = HashMap::new();
        for (x, newx) in &sub_dict {
            back.insert(newx.clone(), x.clone());
        }
        let res = crate::uop::upat::substitute(ctx, simp, back, true, true);
        uop = crate::uop::upat::simplify(ctx, res);
    }
    uop
}

fn sval_cv(s: SVal) -> ConstVal {
    match s {
        SVal::I(i) => ConstVal::Int(i),
        _ => panic!("TGERR: symbolic bound in variable"),
    }
}

fn valid_priority(ctx: &mut UOpCtx, v: UopId, valids: &[UopId]) -> i64 {
    match parse_valid(ctx, v) {
        None => 0,
        Some((expr, _, _)) => {
            let mut n = 0i64;
            for other in valids {
                if ctx.op_in_backward_slice_with_self(other, &[Ops::Range]) {
                    let _ = other;
                }
                // python: sum(-1 for other in valids if res[0] in other.backward_slice_with_self)
                if ctx.backward_slice_with_self(other).contains(&expr) {
                    n -= 1;
                }
            }
            n
        }
    }
}

/// `simplify_valid` (symbolic.py:369).
pub fn simplify_valid(ctx: &mut UOpCtx, valid: UopId) -> Option<UopId> {
    if ctx.op_in_backward_slice_with_self(&valid, &[Ops::Index]) {
        return None;
    }
    let mut valids: Vec<UopId> = Vec::new();
    ctx.split_uop(valid, Ops::And, &mut valids);
    // sort by priority (stable, like python's sorted)
    let mut prios: Vec<(i64, usize)> = valids
        .iter()
        .enumerate()
        .map(|(i, v)| (valid_priority(ctx, v.clone(), &valids), i))
        .collect();
    prios.sort_by_key(|&(p, _)| p);
    let sorted: Vec<UopId> = prios
        .into_iter()
        .map(|(_, i)| (valids[i]).clone())
        .collect();
    let valids = sorted;
    let mut deduped: Vec<UopId> = Vec::new();
    for v in &valids {
        if !deduped.contains(v) {
            deduped.push(v.clone());
        }
    }
    let mut ret: Vec<UopId> = Vec::new();
    for stmt in deduped {
        if !ret.is_empty() {
            let acc = if ret.len() == 1 {
                ret[0].clone()
            } else {
                ctx.uprod((ret[0]).clone(), &ret[1..])
            };
            let s = uop_given_valid(ctx, acc, stmt, true);
            ret.push(s);
        } else {
            ret.push(stmt);
        }
    }
    if ret != valids {
        Some(if ret.len() == 1 {
            (ret[0]).clone()
        } else {
            ctx.uprod((ret[0]).clone(), &ret[1..])
        })
    } else {
        None
    }
}

/// `gated_given_valid` (symbolic.py:418).
fn gated_given_valid(ctx: &mut UOpCtx, cond: UopId, x: UopId, i: UopId) -> Option<UopId> {
    if ctx.dt_of(&x) != crate::dtype::WEAKINT
        || ctx.op_in_backward_slice_with_self(&x, &[Ops::Index])
    {
        return None;
    }
    let gx = uop_given_valid(ctx, cond.clone(), x, false);
    Some(ctx.where_(cond, gx, i))
}

/// `drop_and_clauses` (symbolic.py:389).
fn drop_and_clauses(ctx: &mut UOpCtx, cond: UopId, x: UopId, i: UopId) -> Option<UopId> {
    let mut parts: Vec<UopId> = Vec::new();
    ctx.split_uop(cond, Ops::And, &mut parts);
    let xr = ctx.ranges(&x.clone()).to_vec();
    let mut keep = Vec::new();
    let mut drop = false;
    for c in parts {
        let has = ctx.ranges(&c.clone()).iter().any(|r| xr.contains(r));
        if has {
            keep.push(c);
        } else {
            drop = true;
        }
    }
    if !drop {
        return None;
    }
    let t = ctx.const_py(ConstVal::Bool(true), Some(crate::dtype::BOOL));
    let kept = if keep.is_empty() {
        t
    } else if keep.len() == 1 {
        keep[0].clone()
    } else {
        ctx.uprod((keep[0]).clone(), &keep[1..])
    };
    Some(ctx.where_(kept, x, i))
}

/// `pm_drop_and_clauses` (symbolic.py:395).
pub fn pm_drop_and_clauses() -> PatternMatcher {
    static PM: crate::uop::upat::PmSlot = crate::uop::upat::PmSlot::new();
    PM.get_or_init(build_pm_drop_and_clauses).clone()
}

fn build_pm_drop_and_clauses() -> PatternMatcher {
    PatternMatcher::new(vec![(
        invalid_gate(),
        r(|ctx, st, _| {
            let (cond, x, i) = (sget(st, "cond"), sget(st, "x"), sget(st, "i"));
            drop_and_clauses(ctx, cond, x, i)
        }),
    )])
}

/// `pm_simplify_valid` (symbolic.py:425).
pub fn pm_simplify_valid() -> PatternMatcher {
    static PM: crate::uop::upat::PmSlot = crate::uop::upat::PmSlot::new();
    PM.get_or_init(build_pm_simplify_valid).clone()
}

fn build_pm_simplify_valid() -> PatternMatcher {
    PatternMatcher::new(vec![
        (
            UPat::op_var(vec![Ops::And]).named("valid"),
            r(|ctx, st, _| simplify_valid(ctx, sget(st, "valid"))),
        ),
        (
            invalid_gate(),
            r(|ctx, st, _| {
                let (cond, x, i) = (sget(st, "cond"), sget(st, "x"), sget(st, "i"));
                gated_given_valid(ctx, cond.clone(), x.clone(), i.clone())
                    .or_else(|| drop_and_clauses(ctx, cond, x, i))
            }),
        ),
    ])
}

// ---------- pm_move_where_on_load (symbolic.py:398-423) ----------

/// `where_on_load` (symbolic.py:398).
pub fn where_on_load(
    ctx: &mut UOpCtx,
    cond: UopId,
    buf: UopId,
    idx: UopId,
    or_cast: UopId,
) -> Option<UopId> {
    let where_clauses = ctx.split_uop_vec(cond, Ops::And);
    let load_valid = ctx.get_valid(idx.clone());
    let in_load: std::collections::HashSet<UopId> = ctx
        .split_uop_vec(load_valid.clone(), Ops::And)
        .into_iter()
        .collect();
    let idx_index: std::collections::HashSet<UopId> = ctx
        .backward_slice_with_self(&idx)
        .iter()
        .filter(|&u| ctx.op_of(u) == Ops::Index)
        .cloned()
        .collect();
    let idx_ranges: std::collections::HashSet<UopId> =
        ctx.ranges(&idx.clone()).iter().cloned().collect();
    // can move if: condition's ranges are subset of idx's ranges, and no data dependent INDEX
    let mut moved: Vec<UopId> = Vec::new();
    let mut keep: Vec<UopId> = Vec::new();
    let cands: Vec<UopId> = where_clauses
        .iter()
        .filter(|c| !in_load.contains(c))
        .cloned()
        .collect();
    for c in &cands {
        let cbs = ctx.backward_slice_with_self(c);
        let sub = ctx
            .ranges(&c.clone())
            .iter()
            .all(|r| idx_ranges.contains(r));
        let no_idx = cbs
            .iter()
            .filter(|&u| ctx.op_of(u) == Ops::Index)
            .all(|u| idx_index.contains(u));
        if sub && no_idx {
            moved.push(c.clone());
        } else {
            keep.push(c.clone());
        }
    }
    if keep.len() == where_clauses.len() {
        return None;
    }
    let new_valid = ctx.uprod(load_valid, &moved);
    let gi = ctx.get_idx(idx);
    let gv = ctx.valid(gi, new_valid);
    let new_idx = ctx.index(buf, vec![gv]);
    let ret_idx = if ctx.op_of(&or_cast) == Ops::Cast {
        ctx.cast(new_idx, ctx.dt_of(&or_cast))
    } else {
        new_idx
    };
    let ct = ctx.const_bool(true);
    let k = ctx.uprod(ct, &keep);
    let zero = ctx.const_like(ret_idx.clone(), CV::Int(0));
    Some(ctx.where_(k, ret_idx, zero))
}

/// `pm_move_where_on_load` (symbolic.py:412).
pub fn pm_move_where_on_load() -> PatternMatcher {
    static PM: crate::uop::upat::PmSlot = crate::uop::upat::PmSlot::new();
    PM.get_or_init(build_pm_move_where_on_load).clone()
}

fn build_pm_move_where_on_load() -> PatternMatcher {
    PatternMatcher::new(vec![
        (
            var("cond").where_(
                var("buf")
                    .index(vec![var("idx")])
                    .or_casted(Some("or_cast")),
                UPatOrConst::Val(CV::Int(0)),
            ),
            r(|ctx, st, _| {
                let (cond, buf, idx, or_cast) = (
                    sget(st, "cond"),
                    sget(st, "buf"),
                    sget(st, "idx"),
                    sget(st, "or_cast"),
                );
                where_on_load(ctx, cond, buf, idx, or_cast)
            }),
        ),
        (
            var("cond").where_(
                UPatOrConst::Val(CV::Int(0)),
                var("buf")
                    .index(vec![var("idx")])
                    .or_casted(Some("or_cast")),
            ),
            r(|ctx, st, _| {
                let (cond, buf, idx, or_cast) = (
                    sget(st, "cond"),
                    sget(st, "buf"),
                    sget(st, "idx"),
                    sget(st, "or_cast"),
                );
                let nc = ctx.logical_not(cond);
                where_on_load(ctx, nc, buf, idx, or_cast)
            }),
        ),
    ])
}

/// `pm_remove_invalid` (symbolic.py:98).
pub fn pm_remove_invalid() -> PatternMatcher {
    static PM: crate::uop::upat::PmSlot = crate::uop::upat::PmSlot::new();
    PM.get_or_init(build_pm_remove_invalid).clone()
}

fn build_pm_remove_invalid() -> PatternMatcher {
    PatternMatcher::new(vec![
        (
            invalid_gate().named("w"),
            r(|ctx, st, _| {
                let (cond, x, w) = (sget(st, "cond"), sget(st, "x"), sget(st, "w"));
                let zero = ctx.const_like(w.clone(), CV::Int(0));
                let srcs = ctx.srcs(&w).to_vec();
                if srcs[0] == cond && srcs[1] == x && srcs[2] == zero {
                    return None;
                }
                Some(ctx.uop(
                    Ops::Where,
                    Some(ctx.dt_of(&w)),
                    vec![cond, x, zero],
                    Arg::None,
                ))
            }),
        ),
        (
            UPat::op_var(vec![Ops::Stack]).named("s"),
            r(|ctx, st, _| {
                let s = sget(st, "s");
                let srcs = ctx.srcs(&s).to_vec();
                if !srcs.iter().any(|x| ctx.is_invalid(x)) {
                    return None;
                }
                let dt = ctx.dt_of(&s);
                let new_srcs: Vec<UopId> = srcs
                    .iter()
                    .map(|x| {
                        if ctx.is_invalid(x) {
                            ctx.const_py(CV::Int(0), Some(dt))
                        } else {
                            x.clone()
                        }
                    })
                    .collect();
                Some(ctx.uop(Ops::Stack, Some(dt), new_srcs, Arg::None))
            }),
        ),
    ])
}

/// `pm_cast_float_alu` (codegen/__init__.py:254).
pub fn pm_cast_float_alu() -> PatternMatcher {
    static PM: crate::uop::upat::PmSlot = crate::uop::upat::PmSlot::new();
    PM.get_or_init(build_pm_cast_float_alu).clone()
}

fn build_pm_cast_float_alu() -> PatternMatcher {
    PatternMatcher::new(vec![(
        UPat::op_var(vec![
            Ops::Sin,
            Ops::Log2,
            Ops::Exp2,
            Ops::Sqrt,
            Ops::Reciprocal,
        ])
        .with_src_tup(vec![UPat::var(None, None)])
        .named("u"),
        r(|ctx, st, _| {
            let u = sget(st, "u");
            let x = ctx.srcs(&u)[0].clone();
            if ctx.dt_of(&x) == ctx.dt_of(&u) {
                return None;
            }
            let (op, dt, _, arg, tag) = ctx.snapshot(&u);
            let cx = ctx.cast(x, dt);
            Some(ctx.replace(&u, op, dt, vec![cx], arg, tag))
        }),
    )])
}

// ---------- sym = symbolic+pm_simplify_valid+extra (symbolic.py:441-466) ----------

fn noop(ctx: &mut UOpCtx) -> UopId {
    ctx.uop(Ops::Noop, Some(crate::dtype::VOID), vec![], Arg::None)
}

fn store2(ctx: &mut UOpCtx, idx: UopId, val: UopId) -> UopId {
    ctx.uop(
        Ops::Store,
        Some(crate::dtype::VOID),
        vec![idx, val],
        Arg::None,
    )
}

/// `reduce_mul_chain` (symbolic.py:381).
fn reduce_mul_chain(ctx: &mut UOpCtx, r: UopId) -> Option<UopId> {
    let (rop, rdt, srcs, arg, _tag) = ctx.snapshot(&r);
    let top: Ops = match &arg {
        Arg::Seq { tuple: true, items } => match &items[0] {
            Arg::Enum(e) if e.class == crate::uop::arg::EnumClass::Ops => Ops::from_u16(e.value)?,
            _ => return None,
        },
        _ => return None,
    };
    if !matches!(top, Ops::Add | Ops::Max) || rdt != ctx.dt_of(&srcs[0]) {
        return None;
    }
    let reds = &srcs[1..];
    let mut inside: Vec<UopId> = Vec::new();
    let mut outside: Vec<UopId> = Vec::new();
    for m in ctx.split_uop_vec((srcs[0]).clone(), Ops::Mul) {
        let mp = ctx.backward_slice(&m);
        if !reds.contains(&m)
            && reds.iter().all(|rr| !mp.contains(rr))
            && (top != Ops::Max || matches!(ctx.vmin(&m), Ok(SVal::I(i)) if i >= 0))
        {
            outside.push(m);
        } else {
            inside.push(m);
        }
    }
    if outside.is_empty() {
        return None;
    }
    let one = ctx.const_like((srcs[0]).clone(), CV::Int(1));
    let inner = if inside.is_empty() {
        one
    } else {
        ctx.uprod((inside[0]).clone(), &inside[1..])
    };
    let new_red = ctx.replace(
        &r,
        rop,
        rdt,
        std::iter::once(inner).chain(reds.iter().cloned()).collect(),
        arg,
        Arg::None,
    );
    let mut prod = outside[0].clone();
    for m in &outside[1..] {
        prod = ctx.mul(prod.clone(), m.clone());
    }
    Some(ctx.mul(new_red, prod.clone()))
}

/// The extra rules of python `sym` (symbolic.py:441-466).
pub fn pm_sym_extra() -> PatternMatcher {
    static PM: crate::uop::upat::PmSlot = crate::uop::upat::PmSlot::new();
    PM.get_or_init(build_pm_sym_extra).clone()
}

fn build_pm_sym_extra() -> PatternMatcher {
    let idx_pat = || {
        UPat::op_var(vec![Ops::Index])
            .allow_all_srcs()
            .named("index")
    };
    let recip1p = || {
        UPat::op_var(vec![Ops::Add])
            .with_src_tup(vec![UPat::cvar(None, None, Some(Arg::Int(1))), var("x")])
            .reciprocal()
            .named("d")
    };
    PatternMatcher::new(vec![
        // reorder ALU/STACK
        (
            UPat::op_var(group::ALU.to_vec())
                .with_src_tup(vec![
                    UPat::op_var(vec![Ops::Stack]).with_src_tup(vec![var("x")]),
                    UPat::op_var(vec![Ops::Stack]).with_src_tup(vec![var("y")]),
                ])
                .named("alu"),
            r(|ctx, st, _| {
                let (alu, x, y) = (sget(st, "alu"), sget(st, "x"), sget(st, "y"));
                let inner = ctx.uop(
                    ctx.op_of(&alu),
                    Some(ctx.dt_of(&alu)),
                    vec![x, y],
                    ctx.arg_of(&alu).clone(),
                );
                Some(ctx.uop(Ops::Stack, Some(ctx.dt_of(&alu)), vec![inner], Arg::None))
            }),
        ),
        // push cast to branches: WHERE(a,b).cast(dt)
        (
            var("s")
                .where_(UPatOrConst::Pat(var("a")), UPatOrConst::Pat(var("b")))
                .cast(None)
                .named("cast"),
            r(|ctx, st, _| {
                let (s, a, b, cast) = (
                    sget(st, "s"),
                    sget(st, "a"),
                    sget(st, "b"),
                    sget(st, "cast"),
                );
                let dt = ctx.dt_of(&cast);
                let ca = commit_weak(ctx, a, dt);
                let cb = commit_weak(ctx, b, dt);
                Some(ctx.where_(s, ca, cb))
            }),
        ),
        // load/store folding: STORE(idx, LOAD(idx)) -> NOOP
        (
            UPat::op_var(vec![Ops::Store]).with_src_tup(vec![
                idx_pat(),
                UPat::op_var(vec![Ops::Load]).with_src_tup(vec![idx_pat()]),
            ]),
            r(|ctx, st, _| {
                let _ = st;
                Some(noop(ctx))
            }),
        ),
        // STORE(idx, WHERE(gate, alt, LOAD(idx))) -> STORE(idx', alt)
        (
            UPat::op_var(vec![Ops::Store]).with_src_tup(vec![
                idx_pat(),
                var("gate").where_(
                    UPatOrConst::Pat(var("alt")),
                    UPatOrConst::Pat(UPat::op_var(vec![Ops::Load]).with_src_tup(vec![idx_pat()])),
                ),
            ]),
            r(|ctx, st, _| {
                let (index, gate, alt) = (sget(st, "index"), sget(st, "gate"), sget(st, "alt"));
                let isrc = ctx.srcs(&index).to_vec();
                let gv = ctx.valid((isrc[1]).clone(), gate);
                let ni = ctx.index((isrc[0]).clone(), vec![gv]);
                Some(store2(ctx, ni, alt))
            }),
        ),
        // fold invalid stores: STORE(_, invalid) -> NOOP
        (
            UPat::op_var(vec![Ops::Store]).with_src_tup(vec![UPat::var(None, None), invalid_pat()]),
            r(|ctx, st, _| {
                let _ = st;
                Some(noop(ctx))
            }),
        ),
        // store of where with invalid -> gated store
        (
            UPat::op_var(vec![Ops::Store]).with_src_tup(vec![
                idx_pat(),
                var("cond").where_(
                    UPatOrConst::Pat(var("val")),
                    UPatOrConst::Pat(invalid_pat()),
                ),
            ]),
            r(|ctx, st, _| {
                let (index, cond, val) = (sget(st, "index"), sget(st, "cond"), sget(st, "val"));
                let isrc = ctx.srcs(&index).to_vec();
                let gv = ctx.valid((isrc[1]).clone(), cond);
                let ni = ctx.index((isrc[0]).clone(), vec![gv]);
                Some(store2(ctx, ni, val))
            }),
        ),
        // 1/(x*x) -> (1/x)*(1/x)
        (
            var("x").mul(var("x")).reciprocal(),
            r(|ctx, st, _| {
                let x = sget(st, "x");
                let rx = ctx.reciprocal(x);
                Some(ctx.mul(rx.clone(), rx))
            }),
        ),
        // 1/(x*x*x) -> (1/x)*(1/x)*(1/x)
        (
            var("x").mul(var("x")).mul(var("x")).reciprocal(),
            r(|ctx, st, _| {
                let x = sget(st, "x");
                let rx = ctx.reciprocal(x);
                let a = ctx.mul(rx.clone(), rx.clone());
                Some(ctx.mul(a, rx))
            }),
        ),
        // 1/(x*c) -> (1/x)*(1/c)
        (
            var("x").mul(UPat::cvar(Some("c"), None, None)).reciprocal(),
            r(|ctx, st, _| {
                let (x, c) = (sget(st, "x"), sget(st, "c"));
                let rx = ctx.reciprocal(x);
                let rc = ctx.reciprocal(c);
                Some(ctx.mul(rx, rc))
            }),
        ),
        // x*(1/(1+x)) -> 1-1/(1+x)
        (
            var("x").mul(recip1p()),
            r(|ctx, st, _| {
                let d = sget(st, "d");
                let one = ctx.const_py(CV::Int(1), Some(ctx.dt_of(&d)));
                Some(ctx.sub(one, d))
            }),
        ),
        // x*(1/(1+x)*y) -> y*(1-1/(1+x))
        (
            UPat::op_var(vec![Ops::Mul]).with_src_tup(vec![
                var("x"),
                UPat::op_var(vec![Ops::Mul]).with_src_tup(vec![recip1p(), var("y")]),
            ]),
            r(|ctx, st, _| {
                let (d, y) = (sget(st, "d"), sget(st, "y"));
                let one = ctx.const_py(CV::Int(1), Some(ctx.dt_of(&d)));
                let t = ctx.sub(one, d);
                Some(ctx.mul(y, t))
            }),
        ),
        // x*(1/(1+x)+y) -> (1-1/(1+x))+x*y
        (
            UPat::op_var(vec![Ops::Mul]).with_src_tup(vec![
                var("x"),
                UPat::op_var(vec![Ops::Add]).with_src_tup(vec![recip1p(), var("y")]),
            ]),
            r(|ctx, st, _| {
                let (x, d, y) = (sget(st, "x"), sget(st, "d"), sget(st, "y"));
                let one = ctx.const_py(CV::Int(1), Some(ctx.dt_of(&d)));
                let t = ctx.sub(one, d);
                let xy = ctx.mul(x, y);
                Some(ctx.add(t, xy))
            }),
        ),
        // move const multiply after the reduce (ADD)
        (
            var("x")
                .mul(UPat::cvar(Some("c"), None, None))
                .reduce(vec![], Some(crate::codegen::simplify::red_add_arg()), true)
                .named("r"),
            r(|ctx, st, _| {
                let (r, x, c) = (sget(st, "r"), sget(st, "x"), sget(st, "c"));
                let (op, dt, srcs, arg, tag) = ctx.snapshot(&r);
                if srcs.len() < 2 {
                    return None;
                }
                let cv = match ctx.const_arg(&c) {
                    Some(CV::Int(i)) => CV::Int(i),
                    Some(CV::Float(f)) => CV::Float(f),
                    _ => return None,
                };
                let inner = ctx.replace(
                    &r,
                    op,
                    dt,
                    std::iter::once(x)
                        .chain(srcs[1..].iter().cloned())
                        .collect(),
                    arg,
                    tag,
                );
                let kc = ctx.const_py(cv, Some(dt));
                Some(ctx.mul(inner, kc))
            }),
        ),
        // reduce mul chain — `UPat(Ops.MUL).reduce(allow_any_len=True)`:
        // the REDUCE's src[0] must be a MUL.
        (
            UPat::ops1(Ops::Reduce)
                .with_src_tup(vec![UPat::op_var(vec![Ops::Mul])])
                .allow_all_srcs()
                .named("r"),
            r(|ctx, st, _| reduce_mul_chain(ctx, sget(st, "r"))),
        ),
    ])
}

/// python `sym` (symbolic.py:441) = symbolic + pm_simplify_valid + extra.
pub fn sym() -> &'static PatternMatcher {
    static SYM: PmSlot = PmSlot::new();
    SYM.get_or_init(|| {
        symbolic()
            .add(&pm_simplify_valid())
            .add(&pm_sym_extra())
            // python: `sym = symbolic+pm_simplify_valid+PatternMatcher([...])+pm_clean_up_group_sink`
            .add(pm_clean_up_group_sink())
    })
}
