//! `codegen/decomp/dtype.py` port: `pm_dtype_decomps` — emulate unsupported /
//! configured dtypes. `pm_long_decomp` splits long/ulong into two 32-bit words;
//! `pm_float_decomp` emulates fp8/bf16/fp16 by widening to a supported float.

use std::collections::HashMap;
use std::collections::HashSet;
use std::rc::Rc;

use crate::dtype::{self, ConstVal, DTypeId};
use crate::uop::arg::Arg;
use crate::uop::ctx::UOpCtx;
use crate::uop::ops::Ops;
use crate::uop::sym::{r, sget};
use crate::uop::upat::{graph_rewrite, r_l2i, r_pair, CtxVal, PatternMatcher, RuleFn, UPat};
use crate::uop::UopId;

type L2iMemo = Rc<std::cell::RefCell<HashMap<(Ops, DTypeId, Vec<UopId>), Vec<UopId>>>>;
type L2iMap = HashMap<(Ops, DTypeId, Vec<UopId>), Vec<UopId>>;

fn new_l2i() -> L2iMemo {
    Rc::new(std::cell::RefCell::new(HashMap::new()))
}

// ---------- dtype tables ----------

fn is_fp8_fnuz(dt: DTypeId) -> bool {
    dtype::FP8_FNUZ.contains(&dt)
}

/// `dtypes.finfo` (dtype.py:107) -> (exponent bits, mantissa bits).
fn finfo(dt: DTypeId) -> (i64, i64) {
    match dt {
        x if x == dtype::FLOAT16 => (5, 10),
        x if x == dtype::BFLOAT16 => (8, 7),
        x if x == dtype::FLOAT32 => (8, 23),
        x if x == dtype::FLOAT64 => (11, 52),
        x if x == dtype::FP8E4M3 || x == dtype::FP8E4M3FNUZ => (4, 3),
        x if x == dtype::FP8E5M2 || x == dtype::FP8E5M2FNUZ => (5, 2),
        other => panic!("TGERR: no finfo for {:?}", other.attr()),
    }
}

/// `exponent_bias` (transcendental.py:11).
fn exponent_bias(dt: DTypeId) -> i64 {
    (1i64 << (finfo(dt).0 - 1)) - if is_fp8_fnuz(dt) { 0 } else { 1 }
}

/// `f2f_dt` (dtype.py:89): the uint type of the same width as a float.
fn f2f_dt(dt: DTypeId) -> DTypeId {
    match dt.bitsize() {
        8 => dtype::UINT8,
        16 => dtype::UINT16,
        32 => dtype::UINT32,
        64 => dtype::UINT64,
        other => panic!("TGERR: no f2f_dt for bitsize {other}"),
    }
}

/// `l2i_dt` (dtype.py:11).
fn l2i_dt(dt: DTypeId) -> DTypeId {
    if dt == dtype::INT64 {
        dtype::INT32
    } else {
        dtype::UINT32
    }
}

/// `2**y`.
fn pow2(y: i64) -> i128 {
    1i128 << y
}

/// `shl(x, y)` (transcendental.py:15): `x * 2**y`.
fn shl_u(ctx: &mut UOpCtx, x: UopId, y: i64) -> UopId {
    let c = ctx.const_py(ConstVal::Int(pow2(y)), None);
    ctx.mul(x, c)
}

/// `shr(x, y)` (transcendental.py:14): `x // 2**y`.
fn shr_u(ctx: &mut UOpCtx, x: UopId, y: i64) -> UopId {
    let c = ctx.const_py(ConstVal::Int(pow2(y)), None);
    ctx.div_floor(x, c)
}

/// `a >= c` (`(a < c).logical_not()`).
fn ge_c(ctx: &mut UOpCtx, a: UopId, v: i64) -> UopId {
    let c = ctx.const_like(a, ConstVal::Int(v as i128));
    let lt = ctx.lt(a, c);
    ctx.logical_not(lt)
}

// ---------- tag helpers ----------

/// read a `(word, dtype)` tag.
fn tag_pair(ctx: &UOpCtx, x: UopId) -> Option<(i64, DTypeId)> {
    match ctx.tag_of(x) {
        Arg::Seq { tuple: true, items } if items.len() == 2 => {
            let w = match &items[0] {
                Arg::Int(i) => *i,
                _ => return None,
            };
            let d = match &items[1] {
                Arg::Dtype(d) => *d,
                _ => return None,
            };
            Some((w, d))
        }
        _ => None,
    }
}

fn has_tag(ctx: &UOpCtx, x: UopId) -> bool {
    !matches!(ctx.tag_of(x), Arg::None)
}

/// `u.rtag((w, dt))`.
fn set_rtag(ctx: &mut UOpCtx, x: UopId, w: i64, dt: DTypeId) -> UopId {
    let (op, d, srcs, arg, _tag) = ctx.snapshot(x);
    let tag = Arg::Seq { tuple: true, items: vec![Arg::Int(w), Arg::Dtype(dt)] };
    ctx.uop_tag(op, Some(d), srcs, arg, tag)
}

fn clear_tag(ctx: &mut UOpCtx, x: UopId) -> UopId {
    let (op, d, srcs, arg, _tag) = ctx.snapshot(x);
    ctx.uop_tag(op, Some(d), srcs, arg, Arg::None)
}

// ---------- long-as-2xint32 ----------

/// `unpack32` (dtype.py:12).
fn unpack32(ctx: &mut UOpCtx, v: UopId) -> (UopId, UopId) {
    let b = ctx.bitcast(v, dtype::UINT32);
    let mask = ctx.const_py(ConstVal::Int(0xFFFF), None);
    let lo = ctx.binop(Ops::And, b, mask);
    let hi = shr_u(ctx, b, 16);
    (lo, hi)
}

/// `reindex` (dtype.py:13).
fn reindex(ctx: &mut UOpCtx, idx: UopId, off: i64, mul: i64) -> UopId {
    let srcs = ctx.srcs(idx).to_vec();
    if ctx.op_of(idx) == Ops::Shrink {
        assert_eq!(mul, 1, "TGERR: can't reindex SHRINK with mul != 1");
        let o = ctx.const_py(ConstVal::Int(off as i128), None);
        let new_off = ctx.add(srcs[1], o);
        let tag = ctx.tag_of(idx).clone();
        let arg = ctx.arg_of(idx).clone();
        ctx.replace(idx, Ops::Index, ctx.dt_of(idx), vec![srcs[0], new_off], arg, tag)
    } else {
        let m = ctx.const_py(ConstVal::Int(mul as i128), None);
        let m2 = ctx.mul(srcs[1], m);
        let o = ctx.const_py(ConstVal::Int(off as i128), None);
        let new1 = ctx.add(m2, o);
        let mut new_srcs = vec![srcs[0], new1];
        new_srcs.extend(srcs[2..].iter().copied());
        let (op, d, _s, arg, tag) = ctx.snapshot(idx);
        ctx.replace(idx, op, d, new_srcs, arg, tag)
    }
}

/// `l2i` (dtype.py:20). `uops` are already word-split; returns 1 or 2 nodes.
fn l2i(ctx: &mut UOpCtx, op: Ops, dt: DTypeId, uops: &[UopId]) -> Vec<UopId> {
    let zero = ctx.const_py(ConstVal::Int(0), Some(dt));
    match op {
        Ops::Neg => l2i(ctx, Ops::Sub, dt, &[zero, zero, uops[0]]),
        Ops::Cast if (dt == dtype::INT64 || dt == dtype::UINT64) && !dtype::FLOATS.contains(&ctx.dt_of(uops[0])) => {
            let x = uops[0];
            let lo = ctx.cast(x, l2i_dt(dt));
            if ctx.dt_of(x) == dtype::BOOL || dtype::UINTS.contains(&ctx.dt_of(x)) {
                let z = ctx.const_like(lo, ConstVal::Int(0));
                vec![lo, z]
            } else {
                let z = ctx.const_like(lo, ConstVal::Int(0));
                let neg1 = ctx.const_like(lo, ConstVal::Int(-1));
                let c = ctx.lt(x, z);
                vec![lo, ctx.where_(c, neg1, z)]
            }
        }
        Ops::Cast if dt == dtype::INT64 || dt == dtype::UINT64 => {
            // from float
            let x = uops[0];
            let lo = ctx.cast(x, l2i_dt(dt));
            let two32 = ctx.const_py(ConstVal::Int(pow2(32)), None);
            let d = ctx.div_floor(x, two32);
            let d = ctx.cast(d, l2i_dt(dt));
            let z = ctx.const_like(x, ConstVal::Int(0));
            let lt = ctx.lt(x, z);
            let zlo = ctx.const_like(lo, ConstVal::Int(0));
            let lone = ctx.ne(lo, zlo);
            let and = ctx.and_(lt, lone);
            let hi = ctx.sub(d, and);
            vec![lo, hi]
        }
        Ops::Cast if dtype::FLOATS.contains(&dt) => {
            let (a0, a1) = (uops[0], uops[1]);
            let cdt = if dt == dtype::FLOAT64 { dtype::FLOAT64 } else { dtype::FLOAT32 };
            let z = ctx.const_py(ConstVal::Int(0), None);
            let a1_eq0 = ctx.eq(a1, z);
            let m1 = ctx.const_py(ConstVal::Int(-1), None);
            let a1_eqm1 = ctx.eq(a1, m1);
            let a0_lt0 = ctx.lt(a0, z);
            let a0_ge0 = ctx.logical_not(a0_lt0);
            let c0 = ctx.and_(a1_eq0, a0_ge0);
            let c1 = ctx.and_(a1_eqm1, a0_lt0);
            let small = ctx.or_(c0, c1);
            let a1c = ctx.cast(a1, cdt);
            let big32 = ctx.const_py(ConstVal::Int(pow2(32)), None);
            let big = ctx.mul(a1c, big32);
            let a0u = ctx.bitcast(a0, dtype::UINT32);
            let a0uc = ctx.cast(a0u, cdt);
            let big = ctx.add(big, a0uc);
            let big = ctx.cast(big, dt);
            let a0c = ctx.cast(a0, dt);
            vec![ctx.where_(small, a0c, big)]
        }
        Ops::Cast => {
            let b = ctx.bitcast(uops[0], dtype::UINT32);
            vec![ctx.cast(b, dt)]
        }
        Ops::Bitcast => vec![ctx.bitcast(uops[0], dt), ctx.bitcast(uops[1], dt)],
        Ops::Shl => {
            let a0u = ctx.bitcast(uops[0], dtype::UINT32);
            let a1u = ctx.bitcast(uops[1], dtype::UINT32);
            let mask = ctx.const_py(ConstVal::Int(31), None);
            let and = ctx.binop(Ops::And, uops[2], mask);
            let n = ctx.cast(and, dtype::UINT32);
            let sh = ctx.shl(a0u, n);
            let lo = ctx.bitcast(sh, dt);
            let one = ctx.const_py(ConstVal::Int(1), None);
            let a0s = ctx.shr(a0u, one);
            let c31 = ctx.const_py(ConstVal::Int(31), None);
            let sub = ctx.sub(c31, n);
            let a0s = ctx.shr(a0s, sub);
            let a1sh = ctx.shl(a1u, n);
            let or1 = ctx.or_(a1sh, a0s);
            let hi = ctx.bitcast(or1, dt);
            let cond = ge_c(ctx, uops[2], 32);
            let z = ctx.const_py(ConstVal::Int(0), Some(dt));
            vec![ctx.where_(cond, z, lo), ctx.where_(cond, lo, hi)]
        }
        Ops::Shr => {
            let a0u = ctx.bitcast(uops[0], dtype::UINT32);
            let a1u = ctx.bitcast(uops[1], dtype::UINT32);
            let mask = ctx.const_py(ConstVal::Int(31), None);
            let and = ctx.binop(Ops::And, uops[2], mask);
            let n = ctx.cast(and, dtype::UINT32);
            let a0s = ctx.shr(a0u, n);
            let one = ctx.const_py(ConstVal::Int(1), None);
            let a1l = ctx.shl(a1u, one);
            let c31 = ctx.const_py(ConstVal::Int(31), None);
            let sub = ctx.sub(c31, n);
            let a1l = ctx.shl(a1l, sub);
            let or2 = ctx.or_(a0s, a1l);
            let lo = ctx.bitcast(or2, dt);
            let and2 = ctx.binop(Ops::And, uops[2], c31);
            let hi = ctx.shr(uops[1], and2);
            let fill = if dt == dtype::INT32 {
                let c31b = ctx.const_py(ConstVal::Int(31), None);
                ctx.shr(uops[1], c31b)
            } else {
                zero
            };
            let cond = ge_c(ctx, uops[2], 32);
            vec![ctx.where_(cond, hi, lo), ctx.where_(cond, fill, hi)]
        }
        Ops::Add => {
            let low = ctx.add(uops[0], uops[2]);
            let l0u = ctx.bitcast(low, dtype::UINT32);
            let au = ctx.bitcast(uops[0], dtype::UINT32);
            let carry = ctx.lt(l0u, au);
            let s = ctx.add(uops[1], uops[3]);
            let hi = ctx.add(s, carry);
            vec![low, hi]
        }
        Ops::Sub => {
            let low = ctx.sub(uops[0], uops[2]);
            let a0u = ctx.bitcast(uops[0], dtype::UINT32);
            let b0u = ctx.bitcast(uops[2], dtype::UINT32);
            let borrow = ctx.lt(a0u, b0u);
            let s = ctx.sub(uops[1], uops[3]);
            let hi = ctx.sub(s, borrow);
            vec![low, hi]
        }
        Ops::Mul => {
            let (a0, a1, b0, b1) = (uops[0], uops[1], uops[2], uops[3]);
            let (a00, a01) = unpack32(ctx, a0);
            let (b00, b01) = unpack32(ctx, b0);
            let p1 = ctx.mul(a00, b01);
            let s1 = shl_u(ctx, p1, 16);
            let t0 = ctx.bitcast(s1, dt);
            let p1b = ctx.mul(a00, b01);
            let s2 = shr_u(ctx, p1b, 16);
            let t1 = ctx.bitcast(s2, dt);
            let p2 = ctx.mul(a01, b00);
            let s3 = shl_u(ctx, p2, 16);
            let t2 = ctx.bitcast(s3, dt);
            let p2b = ctx.mul(a01, b00);
            let s4 = shr_u(ctx, p2b, 16);
            let t3 = ctx.bitcast(s4, dt);
            let mid = l2i(ctx, Ops::Add, dt, &[t0, t1, t2, t3]);
            let mm = ctx.mul(a00, b00);
            let term0 = ctx.bitcast(mm, dt);
            let nn = ctx.mul(a01, b01);
            let base = ctx.bitcast(nn, dt);
            let ab = ctx.mul(a0, b1);
            let cd = ctx.mul(a1, b0);
            let t = ctx.add(base, ab);
            let term1 = ctx.add(t, cd);
            l2i(ctx, Ops::Add, dt, &[mid[0], mid[1], term0, term1])
        }
        Ops::Cdiv | Ops::Cmod => {
            let signed = dt == dtype::INT32;
            let (mut a0, mut a1, mut b0, mut b1) = (uops[0], uops[1], uops[2], uops[3]);
            let mut a_neg = ctx.const_py(ConstVal::Int(0), None);
            let mut b_neg = ctx.const_py(ConstVal::Int(0), None);
            if signed {
                let ua0 = ctx.bitcast(a0, dtype::UINT32);
                let ua1 = ctx.bitcast(a1, dtype::UINT32);
                let ub0 = ctx.bitcast(b0, dtype::UINT32);
                let ub1 = ctx.bitcast(b1, dtype::UINT32);
                let za = ctx.const_py(ConstVal::Int(0), Some(dtype::UINT32));
                let aneg = ctx.lt(ua1, za);
                let nb = l2i(ctx, Ops::Neg, dtype::UINT32, &[ua0, ua1]);
                let na0 = ctx.where_(aneg, nb[0], ua0);
                let na1 = ctx.where_(aneg, nb[1], ua1);
                let zb = ctx.const_py(ConstVal::Int(0), Some(dtype::UINT32));
                let bneg = ctx.lt(ub1, zb);
                let nb2 = l2i(ctx, Ops::Neg, dtype::UINT32, &[ub0, ub1]);
                let nb0 = ctx.where_(bneg, nb2[0], ub0);
                let nb1 = ctx.where_(bneg, nb2[1], ub1);
                a0 = na0;
                a1 = na1;
                b0 = nb0;
                b1 = nb1;
                a_neg = aneg;
                b_neg = bneg;
            }
            let zu = ctx.const_py(ConstVal::Int(0), Some(dtype::UINT32));
            let mut q: (UopId, UopId) = (zu, zu);
            let mut rr: (UopId, UopId) = (zu, zu);
            for i in (0..64).rev() {
                let one = ctx.const_py(ConstVal::Int(1), Some(dtype::UINT32));
                let z = ctx.const_py(ConstVal::Int(0), Some(dtype::UINT32));
                let r2 = l2i(ctx, Ops::Shl, dtype::UINT32, &[rr.0, rr.1, one, z]);
                let iu = ctx.const_py(ConstVal::Int(i as i128), Some(dtype::UINT32));
                let sha = l2i(ctx, Ops::Shr, dtype::UINT32, &[a0, a1, iu, z]);
                let one2 = ctx.const_py(ConstVal::Int(1), Some(dtype::UINT32));
                let amask = ctx.binop(Ops::And, sha[0], one2);
                let lo = ctx.or_(r2[0], amask);
                rr = (lo, r2[1]);
                let cmplt = l2i(ctx, Ops::Cmplt, dtype::UINT32, &[rr.0, rr.1, b0, b1]);
                let cond = ctx.logical_not(cmplt[0]);
                let diff = l2i(ctx, Ops::Sub, dtype::UINT32, &[rr.0, rr.1, b0, b1]);
                let ccond = ctx.cast(cond, dtype::UINT32);
                let shifted = shl_u(ctx, ccond, (i % 32) as i64);
                if i < 32 {
                    q = (ctx.or_(q.0, shifted), q.1);
                } else {
                    q = (q.0, ctx.or_(q.1, shifted));
                }
                let nr0 = ctx.where_(cond, diff[0], rr.0);
                let nr1 = ctx.where_(cond, diff[1], rr.1);
                rr = (nr0, nr1);
            }
            if signed {
                let nq = l2i(ctx, Ops::Neg, dtype::UINT32, &[q.0, q.1]);
                let nr = l2i(ctx, Ops::Neg, dtype::UINT32, &[rr.0, rr.1]);
                let nq0 = ctx.bitcast(nq[0], dt);
                let nq1 = ctx.bitcast(nq[1], dt);
                let nr0 = ctx.bitcast(nr[0], dt);
                let nr1 = ctx.bitcast(nr[1], dt);
                let q0 = ctx.bitcast(q.0, dt);
                let q1 = ctx.bitcast(q.1, dt);
                let r0 = ctx.bitcast(rr.0, dt);
                let r1 = ctx.bitcast(rr.1, dt);
                if op == Ops::Cmod {
                    return vec![ctx.where_(a_neg, nr0, r0), ctx.where_(a_neg, nr1, r1)];
                }
                let sgn = ctx.xor(a_neg, b_neg);
                return vec![ctx.where_(sgn, nq0, q0), ctx.where_(sgn, nq1, q1)];
            }
            if op == Ops::Cmod {
                vec![rr.0, rr.1]
            } else {
                vec![q.0, q.1]
            }
        }
        Ops::Cmplt => {
            let (a0, a1, b0, b1) = (uops[0], uops[1], uops[2], uops[3]);
            let hi_lt = ctx.lt(a1, b1);
            let hi_eq = ctx.eq(a1, b1);
            let a0u = ctx.bitcast(a0, dtype::UINT32);
            let b0u = ctx.bitcast(b0, dtype::UINT32);
            let lo_lt = ctx.lt(a0u, b0u);
            let and = ctx.and_(hi_eq, lo_lt);
            vec![ctx.or_(hi_lt, and)]
        }
        Ops::Cmpeq => {
            let e0 = ctx.eq(uops[0], uops[2]);
            let e1 = ctx.eq(uops[1], uops[3]);
            vec![ctx.and_(e0, e1)]
        }
        Ops::Cmpne => {
            let e0 = ctx.ne(uops[0], uops[2]);
            let e1 = ctx.ne(uops[1], uops[3]);
            vec![ctx.or_(e0, e1)]
        }
        Ops::Xor | Ops::Or | Ops::And => {
            let lo = ctx.binop(op, uops[0], uops[2]);
            let hi = ctx.binop(op, uops[1], uops[3]);
            vec![lo, hi]
        }
        Ops::Where => {
            let c = uops[0];
            vec![ctx.where_(c, uops[1], uops[3]), ctx.where_(c, uops[2], uops[4])]
        }
        Ops::Max => {
            let cmplt = l2i(ctx, Ops::Cmplt, dt, uops);
            l2i(ctx, Ops::Where, dt, &[cmplt[0], uops[2], uops[3], uops[0], uops[1]])
        }
        other => panic!("TGERR: long decomposition of {other:?} unsupported"),
    }
}

fn split_l2i(ctx: &mut UOpCtx, memo: &L2iMemo, op: Ops, dt: DTypeId, uops: Vec<UopId>) -> Vec<UopId> {
    let key = (op, dt, uops.clone());
    if let Some(v) = memo.borrow().get(&key).cloned() {
        return v;
    }
    let split = split_inputs(ctx, &uops, memo);
    let out = l2i(ctx, op, dt, &split);
    memo.borrow_mut().insert(key, out.clone());
    out
}

fn split_inputs(ctx: &mut UOpCtx, uops: &[UopId], memo: &L2iMemo) -> Vec<UopId> {
    if uops.is_empty() {
        return vec![];
    }
    let sink = crate::tensor::sink(ctx, uops.to_vec());
    let pm = pm_long_decomp();
    let out = graph_rewrite(ctx, sink, &pm, true, None, false, false, CtxVal::L2i(memo.clone()))
        .unwrap_or_else(|e| panic!("TGERR: split_l2i rewrite failed: {e}"));
    ctx.srcs(out).to_vec()
}

/// STORE with explicit srcs (reuse the node's existing tag/arg).
fn store_with(ctx: &mut UOpCtx, st_: UopId, idx: UopId, val: UopId) -> UopId {
    let (op, d, srcs, arg, tag) = ctx.snapshot(st_);
    let mut new_srcs = vec![idx, val];
    new_srcs.extend(srcs[2..].iter().copied());
    ctx.replace(st_, op, d, new_srcs, arg, tag)
}

/// `pm_long_decomp` (dtype.py:139).
pub fn pm_long_decomp() -> PatternMatcher {
    let long_dt: Vec<DTypeId> = vec![dtype::INT64, dtype::UINT64];
    let mut rules: Vec<(UPat, RuleFn)> = vec![];

    // Defines (PARAM/BUFFER) with long/ulong dtype
    rules.push((
        UPat::op_var(vec![Ops::Param, Ops::Buffer])
            .with_dtype(Some(long_dt.clone()))
            .with_src_tup(vec![UPat::var(Some("sz"), None)])
            .named("x"),
        r(|ctx, st, _| {
            let (sz, x) = (sget(st, "sz"), sget(st, "x"));
            let (op, _d, _srcs, arg, tag) = ctx.snapshot(x);
            let two = ctx.const_py(ConstVal::Int(2), None);
            let sz2 = ctx.mul(sz, two);
            let new_arg = match arg {
                Arg::Param(mut p) => {
                    p.dtype = l2i_dt(ctx.dt_of(x));
                    Arg::Param(p)
                }
                other => other,
            };
            Some(ctx.uop_tag(op, None, vec![sz2], new_arg, tag))
        }),
    ));
    // INDEX long
    rules.push((
        UPat::ops1(Ops::Index).with_dtype(Some(long_dt.clone())).named("x"),
        r(|ctx, st, _| {
            let x = sget(st, "x");
            let (w, _dt) = tag_pair(ctx, x)?;
            let r = reindex(ctx, x, w, 2);
            Some(clear_tag(ctx, r))
        }),
    ));
    // STORE with long idx
    rules.push((
        UPat::ops1(Ops::Store)
            .with_src_tup(vec![UPat::var(Some("idx"), Some(long_dt.clone())), UPat::var(Some("val"), None)])
            .allow_all_srcs()
            .named("st"),
        r(|ctx, st, _| {
            let (st_, idx, val) = (sget(st, "st"), sget(st, "idx"), sget(st, "val"));
            if has_tag(ctx, val) {
                return None;
            }
            let dt = l2i_dt(ctx.dt_of(idx));
            let i0 = set_rtag(ctx, idx, 0, dt);
            let v0 = set_rtag(ctx, val, 0, dt);
            let s0 = store_with(ctx, st_, i0, v0);
            let i1 = set_rtag(ctx, idx, 1, dt);
            let v1 = set_rtag(ctx, val, 1, dt);
            let s1 = store_with(ctx, st_, i1, v1);
            Some(ctx.uop(Ops::Group, None, vec![s0, s1], Arg::None))
        }),
    ));
    // Comparison with long src
    rules.push((
        UPat::op_var(vec![Ops::Cmplt, Ops::Cmpne, Ops::Cmpeq])
            .with_src_tup(vec![UPat::var(Some("a"), Some(long_dt.clone())), UPat::var(Some("b"), None)])
            .named("x"),
        r_l2i(|memo, ctx, st| {
            let (a, x) = (sget(st, "a"), sget(st, "x"));
            let dt = l2i_dt(ctx.dt_of(a));
            let srcs = ctx.srcs(x).to_vec();
            let mut v = vec![];
            for s in srcs {
                v.push(set_rtag(ctx, s, 0, dt));
                v.push(set_rtag(ctx, s, 1, dt));
            }
            let out = split_l2i(ctx, memo, ctx.op_of(x), dt, v);
            Some(out[0])
        }),
    ));
    // CAST long->long bitcast
    rules.push((
        UPat::ops1(Ops::Cast)
            .with_dtype(Some(long_dt.clone()))
            .with_src_tup(vec![UPat::var(Some("a"), Some(long_dt.clone()))])
            .named("x"),
        r_l2i(|memo, ctx, st| {
            let (a, x) = (sget(st, "a"), sget(st, "x"));
            let (w, _dt) = tag_pair(ctx, x)?;
            let dt = l2i_dt(ctx.dt_of(a));
            let a0 = set_rtag(ctx, a, 0, dt);
            let a1 = set_rtag(ctx, a, 1, dt);
            let out = split_l2i(ctx, memo, Ops::Bitcast, l2i_dt(ctx.dt_of(x)), vec![a0, a1]);
            Some(out[w as usize])
        }),
    ));
    // CAST to long from non-long (tag present)
    rules.push((
        UPat::ops1(Ops::Cast)
            .with_dtype(Some(long_dt.clone()))
            .with_src_tup(vec![UPat::var(Some("a"), None)])
            .named("x"),
        r_l2i(|memo, ctx, st| {
            let (a, x) = (sget(st, "a"), sget(st, "x"));
            let (w, _dt) = tag_pair(ctx, x)?;
            let out = split_l2i(ctx, memo, ctx.op_of(x), ctx.dt_of(x), vec![a]);
            Some(out[w as usize])
        }),
    ));
    // CAST from long to non-long (a has no tag)
    rules.push((
        UPat::ops1(Ops::Cast).with_src_tup(vec![UPat::var(Some("a"), Some(long_dt.clone()))]).named("x"),
        r_l2i(|memo, ctx, st| {
            let (a, x) = (sget(st, "a"), sget(st, "x"));
            let xdt = ctx.dt_of(x);
            if xdt == dtype::INT64 || xdt == dtype::UINT64 || has_tag(ctx, a) {
                return None;
            }
            let dt = l2i_dt(ctx.dt_of(a));
            let a0 = set_rtag(ctx, a, 0, dt);
            let a1 = set_rtag(ctx, a, 1, dt);
            let out = split_l2i(ctx, memo, ctx.op_of(x), xdt, vec![a0, a1]);
            Some(out[0])
        }),
    ));
    // SHL/SHR long
    rules.push((
        UPat::op_var(vec![Ops::Shl, Ops::Shr])
            .with_dtype(Some(long_dt.clone()))
            .with_src_tup(vec![UPat::var(Some("a"), None), UPat::var(Some("b"), None)])
            .named("x"),
        r_l2i(|memo, ctx, st| {
            let (a, b, x) = (sget(st, "a"), sget(st, "b"), sget(st, "x"));
            let (w, _dt) = tag_pair(ctx, x)?;
            let dt = l2i_dt(ctx.dt_of(x));
            let a0 = set_rtag(ctx, a, 0, dt);
            let a1 = set_rtag(ctx, a, 1, dt);
            let b0 = set_rtag(ctx, b, 0, dt);
            let out = split_l2i(ctx, memo, ctx.op_of(x), dt, vec![a0, a1, b0]);
            Some(out[w as usize])
        }),
    ));
    // WHERE long
    rules.push((
        UPat::ops1(Ops::Where)
            .with_dtype(Some(long_dt.clone()))
            .with_src_tup(vec![UPat::var(Some("c"), None), UPat::var(Some("a"), None), UPat::var(Some("b"), None)])
            .named("x"),
        r_l2i(|memo, ctx, st| {
            let (c, a, b, x) = (sget(st, "c"), sget(st, "a"), sget(st, "b"), sget(st, "x"));
            let (w, _dt) = tag_pair(ctx, x)?;
            let dt = l2i_dt(ctx.dt_of(x));
            let a0 = set_rtag(ctx, a, 0, dt);
            let a1 = set_rtag(ctx, a, 1, dt);
            let b0 = set_rtag(ctx, b, 0, dt);
            let b1 = set_rtag(ctx, b, 1, dt);
            let out = split_l2i(ctx, memo, Ops::Where, dt, vec![c, a0, a1, b0, b1]);
            Some(out[w as usize])
        }),
    ));
    // other ALU + BITCAST with long dtype
    let alu_set: Vec<Ops> = crate::uop::ops::group::ALU
        .iter()
        .copied()
        .filter(|o| {
            !crate::uop::ops::group::COMPARISON.contains(o)
                && !matches!(o, Ops::Shl | Ops::Shr | Ops::Where)
        })
        .chain(std::iter::once(Ops::Bitcast))
        .collect();
    rules.push((
        UPat::op_var(alu_set).with_dtype(Some(long_dt.clone())).allow_all_srcs().named("x"),
        r_l2i(|memo, ctx, st| {
            let x = sget(st, "x");
            let (w, _dt) = tag_pair(ctx, x)?;
            let dt = l2i_dt(ctx.dt_of(x));
            let srcs = ctx.srcs(x).to_vec();
            let mut v = vec![];
            for a in srcs {
                v.push(set_rtag(ctx, a, 0, dt));
                v.push(set_rtag(ctx, a, 1, dt));
            }
            let out = split_l2i(ctx, memo, ctx.op_of(x), dt, v);
            Some(out[w as usize])
        }),
    ));
    // LOAD long
    rules.push((
        UPat::ops1(Ops::Load)
            .with_dtype(Some(long_dt.clone()))
            .with_src_tup(vec![UPat::var(Some("idx"), None)])
            .allow_all_srcs()
            .named("x"),
        r_l2i(|memo, ctx, st| {
            let (idx, x) = (sget(st, "idx"), sget(st, "x"));
            let (w, _dt) = tag_pair(ctx, x)?;
            let pm = pm_long_decomp();
            let ridx = graph_rewrite(ctx, idx, &pm, true, None, false, false, CtxVal::L2i(memo.clone()))
                .unwrap_or_else(|e| panic!("TGERR: long load idx rewrite failed: {e}"));
            let ridx = reindex(ctx, ridx, w, 2);
            let rt = clear_tag(ctx, ridx);
            let srcs = ctx.srcs(x).to_vec();
            Some(crate::tensor::load(ctx, rt, srcs[1..].to_vec()))
        }),
    ));
    // tagged CONST
    rules.push((
        UPat::ops1(Ops::Const).named("x"),
        r(|ctx, st, _| {
            let x = sget(st, "x");
            let (w, dt) = tag_pair(ctx, x)?;
            if !(dt == dtype::INT32 || dt == dtype::UINT32) {
                return None;
            }
            let v = match ctx.val(x) {
                ConstVal::Int(i) => i,
                _ => return None,
            };
            let word = if w == 1 { v >> 32 } else { v & 0xFFFF_FFFF };
            Some(ctx.const_py(ConstVal::Int(word), Some(dt)))
        }),
    ));
    PatternMatcher::new(rules)
}

// ---------- floats ----------

/// `rne` (dtype.py:91): round to nearest even.
fn rne(ctx: &mut UOpCtx, v: UopId, s: i64) -> UopId {
    let hi = shr_u(ctx, v, s);
    let one = ctx.const_py(ConstVal::Int(1), None);
    let bit0 = shr_u(ctx, v, s - 1);
    let bit = ctx.binop(Ops::And, bit0, one);
    let mask = ctx.const_py(ConstVal::Int(pow2(s - 1) - 1), None);
    let low = ctx.binop(Ops::And, v, mask);
    let z = ctx.const_py(ConstVal::Int(0), None);
    let low_ne = ctx.ne(low, z);
    let one2 = ctx.const_py(ConstVal::Int(1), None);
    let hibit = ctx.binop(Ops::And, hi, one2);
    let or = ctx.or_(low_ne, hibit);
    let inc = ctx.binop(Ops::And, bit, or);
    ctx.add(hi, inc)
}

/// `f2f_clamp` (dtype.py:120).
fn f2f_clamp(ctx: &mut UOpCtx, val: UopId, dt: DTypeId, sat: bool) -> UopId {
    let (e, m) = finfo(dt);
    let (max_exp, max_man) = if is_fp8_fnuz(dt) {
        ((1i64 << e) - 1, (1i64 << m) - 1)
    } else if dt == dtype::FP8E4M3 {
        ((1i64 << e) - 1, (1i64 << m) - 2)
    } else {
        ((1i64 << e) - 2, (1i64 << m) - 1)
    };
    let mx_f = 2f64.powi((max_exp - exponent_bias(dt)) as i32) * (1.0 + max_man as f64 / (1i64 << m) as f64);
    let mx = ctx.const_like(val, ConstVal::Float(mx_f));
    let sat_val = if dtype::FP8S.contains(&dt) && sat {
        mx
    } else {
        ctx.const_like(val, ConstVal::Float(f64::INFINITY))
    };
    let neg_mx = ctx.neg(mx);
    let neg_sat = ctx.neg(sat_val);
    let nan_cond = ctx.ne(val, val);
    let lt = ctx.lt(val, neg_mx);
    let gt = ctx.lt(mx, val);
    let inner = ctx.where_(gt, sat_val, val);
    let outer = ctx.where_(lt, neg_sat, inner);
    ctx.where_(nan_cond, val, outer)
}

/// `f2f` (dtype.py:93).
fn f2f(ctx: &mut UOpCtx, v: UopId, fr: DTypeId, to: DTypeId, sat: bool) -> UopId {
    let fs = fr.bitsize() as i64;
    let (fe, fm) = finfo(fr);
    let fb = exponent_bias(fr);
    let ts = to.bitsize() as i64;
    let (te, tm) = finfo(to);
    let tb = exponent_bias(to);
    if fe <= te && fm < tm {
        let sign_bit = ctx.const_py(ConstVal::Int(pow2(fs - 1)), None);
        let vand = ctx.binop(Ops::And, v, sign_bit);
        let vand_c = ctx.cast(vand, f2f_dt(to));
        let sign = shl_u(ctx, vand_c, ts - fs);
        let nosign_mask = ctx.const_py(ConstVal::Int(pow2(fs - 1) - 1), None);
        let vand2 = ctx.binop(Ops::And, v, nosign_mask);
        let nosign = ctx.cast(vand2, f2f_dt(to));
        let exp = shr_u(ctx, nosign, fm);
        let norm0 = shl_u(ctx, nosign, tm - fm);
        let nconst = ctx.const_py(ConstVal::Int((tb - fb) as i128 * pow2(tm)), None);
        let norm = ctx.add(norm0, nconst);
        let nan0 = shl_u(ctx, nosign, tm - fm);
        let nanconst = ctx.const_py(ConstVal::Int((pow2(te) - 1) * pow2(tm)), None);
        let nan = ctx.binop(Ops::Or, nan0, nanconst);
        if is_fp8_fnuz(fr) {
            let z = ctx.const_py(ConstVal::Int(0), None);
            let n_sign = ctx.ne(sign, z);
            let n_nosign = ctx.eq(nosign, z);
            let fnuz_nan = ctx.and_(n_sign, n_nosign);
            let qnan_v = (pow2(te) - 1) * pow2(tm) + pow2(tm - 1);
            let qnan = ctx.const_py(ConstVal::Int(qnan_v), None);
            let thresh = std::cmp::max(fb - tb, 0) + 1;
            let tc = ctx.const_py(ConstVal::Int(thresh as i128), None);
            let exp_lt = ctx.lt(exp, tc);
            let z2 = ctx.const_py(ConstVal::Int(0), None);
            let inner = ctx.where_(exp_lt, z2, norm);
            let res = ctx.binop(Ops::Or, sign, inner);
            let bc = ctx.where_(fnuz_nan, qnan, res);
            return ctx.bitcast(bc, to);
        }
        let is_nan = if fr == dtype::FP8E4M3 {
            let c = ctx.const_py(ConstVal::Int(pow2(fm + fe) - 1), None);
            ctx.eq(nosign, c)
        } else {
            let c = ctx.const_py(ConstVal::Int(pow2(fe) - 1), None);
            ctx.eq(exp, c)
        };
        let inner = ctx.where_(is_nan, nan, norm);
        let z = ctx.const_py(ConstVal::Int(0), None);
        let exp_eq0 = ctx.eq(exp, z);
        let inner = ctx.where_(exp_eq0, z, inner);
        let or3 = ctx.binop(Ops::Or, sign, inner);
        ctx.bitcast(or3, to)
    } else if fe >= te && fm > tm {
        let raw = ctx.bitcast(v, fr);
        let clamped = f2f_clamp(ctx, raw, to, sat);
        let v = ctx.bitcast(clamped, f2f_dt(fr));
        let sign_shift = shr_u(ctx, v, fs - ts);
        let sign_mask = ctx.const_py(ConstVal::Int(pow2(ts - 1)), None);
        let sign = ctx.binop(Ops::And, sign_shift, sign_mask);
        let nosign_mask = ctx.const_py(ConstVal::Int(pow2(fs - 1) - 1), None);
        let nosign = ctx.binop(Ops::And, v, nosign_mask);
        let r = rne(ctx, nosign, fm - tm);
        let sub = ctx.const_py(ConstVal::Int((fb - tb) as i128 * pow2(tm)), None);
        let subv = ctx.sub(r, sub);
        let norm = ctx.cast(subv, f2f_dt(to));
        let sh = shr_u(ctx, v, fm);
        let emask = ctx.const_py(ConstVal::Int(pow2(fe) - 1), None);
        let and = ctx.binop(Ops::And, sh, emask);
        let uf_c = ctx.const_py(ConstVal::Int((1 + fb - tb) as i128), None);
        let underflow = ctx.lt(and, uf_c);
        let nan_mantissa = if to == dtype::FP8E4M3 {
            ctx.const_py(ConstVal::Int(pow2(tm) - 1), None)
        } else {
            let a = shr_u(ctx, nosign, fm - tm);
            let m = ctx.const_py(ConstVal::Int(pow2(tm) - 1), None);
            ctx.binop(Ops::And, a, m)
        };
        let sign_mant = ctx.binop(Ops::Or, sign, nan_mantissa);
        let exp_bits = ctx.const_py(ConstVal::Int((pow2(te) - 1) * pow2(tm)), None);
        let nan_raw = ctx.binop(Ops::Or, sign_mant, exp_bits);
        let nan = ctx.cast(nan_raw, f2f_dt(to));
        let sh2 = shr_u(ctx, v, fm);
        let and2 = ctx.binop(Ops::And, sh2, emask);
        let fexp = ctx.const_py(ConstVal::Int(pow2(fe) - 1), None);
        let is_nan = ctx.eq(and2, fexp);
        if dtype::FP8_FNUZ.contains(&to) {
            let one = ctx.const_py(ConstVal::Int(1), None);
            let z = ctx.const_py(ConstVal::Int(0), None);
            let sc = ctx.cast(sign, f2f_dt(to));
            let or = ctx.binop(Ops::Or, sc, norm);
            let res = ctx.where_(underflow, z, or);
            let shifted = shl_u(ctx, one, ts - 1);
            return ctx.where_(is_nan, shifted, res);
        }
        let sc = ctx.cast(sign, f2f_dt(to));
        let z = ctx.const_py(ConstVal::Int(0), None);
        let inner = ctx.where_(underflow, z, norm);
        let res = ctx.binop(Ops::Or, sc, inner);
        ctx.where_(is_nan, nan, res)
    } else {
        panic!("TGERR: unsupported f2f decomp {fr:?} -> {to:?}")
    }
}

/// `f2f_load` (dtype.py:129).
fn f2f_load(ctx: &mut UOpCtx, x: UopId, fr: DTypeId, to: DTypeId) -> UopId {
    let xsrcs = ctx.srcs(x).to_vec();
    let pm = pm_float_decomp();
    let storage_idx = graph_rewrite(ctx, xsrcs[0], &pm, true, None, false, false, CtxVal::Pair(fr, to))
        .unwrap_or_else(|e| panic!("TGERR: f2f_load idx rewrite failed: {e}"));
    let n = ctx.max_numel(x);
    if n == 1 {
        let ld = crate::tensor::load(ctx, storage_idx, xsrcs[1..].to_vec());
        return f2f(ctx, ld, fr, to, true);
    }
    let mut vals = vec![];
    for i in 0..n {
        let ridx = reindex(ctx, storage_idx, i, 1);
        let ld = crate::tensor::load(ctx, ridx, xsrcs[1..].to_vec());
        vals.push(f2f(ctx, ld, fr, to, true));
    }
    crate::tensor::stack(ctx, vals)
}

/// `f2f_store` (dtype.py:134).
fn f2f_store(ctx: &mut UOpCtx, st_: UopId, idx: UopId, val: UopId, fr: DTypeId, to: DTypeId) -> UopId {
    let n = ctx.max_numel(val);
    if n == 1 {
        let bc = ctx.bitcast(val, f2f_dt(to));
        let v = f2f(ctx, bc, to, fr, true);
        return store_with(ctx, st_, idx, v);
    }
    let mut groups = vec![];
    for i in 0..n {
        let ridx = reindex(ctx, idx, i, 1);
        let ic = ctx.const_py(ConstVal::Int(i as i128), None);
        let vi = ctx.index(val, vec![ic]);
        let bc = ctx.bitcast(vi, f2f_dt(to));
        let v = f2f(ctx, bc, to, fr, true);
        groups.push(store_with(ctx, st_, ridx, v));
    }
    ctx.uop(Ops::Group, None, groups, Arg::None)
}

/// `pm_float_decomp` (dtype.py:170).
pub fn pm_float_decomp() -> PatternMatcher {
    let all = all_ops();
    let mut rules: Vec<(UPat, RuleFn)> = vec![];
    // Defines
    rules.push((
        UPat::op_var(vec![Ops::Param, Ops::Buffer]).named("x"),
        r_pair(|fr, _to, ctx, st| {
            let x = sget(st, "x");
            if ctx.dt_of(x) != fr {
                return None;
            }
            let (op, _d, srcs, arg, _tag) = ctx.snapshot(x);
            let new_arg = match arg {
                Arg::Param(mut p) => {
                    p.dtype = f2f_dt(fr);
                    Arg::Param(p)
                }
                other => other,
            };
            Some(ctx.uop_tag(op, None, srcs, new_arg, Arg::Dtype(fr)))
        }),
    ));
    // INDEX / SHRINK
    let not_load_stack: Vec<Ops> = all.iter().copied().filter(|o| !matches!(o, Ops::Load | Ops::Stack)).collect();
    rules.push((
        UPat::op_var(vec![Ops::Index, Ops::Shrink])
            .with_src_tup(vec![UPat::op_var(not_load_stack)])
            .allow_all_srcs()
            .named("x"),
        r_pair(|fr, to, ctx, st| {
            let x = sget(st, "x");
            if ctx.dt_of(x) != fr {
                return None;
            }
            let srcs = ctx.srcs(x).to_vec();
            let pm = pm_float_decomp();
            let s0 = graph_rewrite(ctx, srcs[0], &pm, true, None, false, false, CtxVal::Pair(fr, to))
                .unwrap_or_else(|e| panic!("TGERR: float idx rewrite failed: {e}"));
            let mut new_srcs = vec![s0];
            new_srcs.extend(srcs[1..].iter().copied());
            let arg = ctx.arg_of(x).clone();
            Some(ctx.uop_tag(ctx.op_of(x), None, new_srcs, arg, Arg::Dtype(fr)))
        }),
    ));
    // LOAD floats
    rules.push((
        UPat::ops1(Ops::Load).with_dtype(Some(dtype::FLOATS.to_vec())).named("x"),
        r_pair(|fr, to, ctx, st| {
            let x = sget(st, "x");
            if ctx.dt_of(x) != fr {
                return None;
            }
            Some(f2f_load(ctx, x, fr, to))
        }),
    ));
    // BITCAST(LOAD)
    rules.push((
        UPat::ops1(Ops::Bitcast)
            .with_src_tup(vec![UPat::ops1(Ops::Load).named("ld")])
            .named("bc"),
        r_pair(|fr, to, ctx, st| {
            let (bc, ld) = (sget(st, "bc"), sget(st, "ld"));
            if ctx.dt_of(ld) != fr {
                return None;
            }
            let srcs = ctx.srcs(ld).to_vec();
            let pm = pm_float_decomp();
            let base = graph_rewrite(ctx, srcs[0], &pm, true, None, false, false, CtxVal::Pair(fr, to))
                .unwrap_or_else(|e| panic!("TGERR: float bitcast-load rewrite failed: {e}"));
            let ld2 = crate::tensor::load(ctx, base, srcs[1..].to_vec());
            Some(ctx.bitcast(ld2, ctx.dt_of(bc)))
        }),
    ));
    // BITCAST from float
    rules.push((
        UPat::ops1(Ops::Bitcast)
            .with_src_tup(vec![UPat::var(Some("x"), Some(dtype::FLOATS.to_vec()))])
            .named("bc"),
        r_pair(|fr, to, ctx, st| {
            let (bc, x) = (sget(st, "bc"), sget(st, "x"));
            if ctx.dt_of(x) != to || ctx.dt_of(bc).bitsize() != fr.bitsize() {
                return None;
            }
            let bc2 = ctx.bitcast(x, f2f_dt(to));
            let v = f2f(ctx, bc2, to, fr, true);
            let (op, _d, _s, arg, _t) = ctx.snapshot(bc);
            Some(ctx.uop(op, None, vec![v], arg))
        }),
    ));
    // BITCAST to
    rules.push((
        UPat::ops1(Ops::Bitcast).with_src_tup(vec![UPat::var(Some("x"), None)]).named("bc"),
        r_pair(|fr, to, ctx, st| {
            let (bc, x) = (sget(st, "bc"), sget(st, "x"));
            if ctx.dt_of(bc) != fr {
                return None;
            }
            let b = ctx.bitcast(x, f2f_dt(fr));
            Some(f2f(ctx, b, fr, to, true))
        }),
    ));
    // CAST to float
    rules.push((
        UPat::ops1(Ops::Cast)
            .with_dtype(Some(dtype::FLOATS.to_vec()))
            .with_src_tup(vec![UPat::var(Some("val"), None)])
            .named("x"),
        r_pair(|fr, to, ctx, st| {
            let (x, val) = (sget(st, "x"), sget(st, "val"));
            if ctx.dt_of(x) != fr {
                return None;
            }
            let v = ctx.cast(val, to);
            Some(f2f_clamp(ctx, v, fr, true))
        }),
    ));
    // CONST float
    rules.push((
        UPat::ops1(Ops::Const).with_dtype(Some(dtype::FLOATS.to_vec())).named("x"),
        r_pair(|fr, to, ctx, st| {
            let x = sget(st, "x");
            if ctx.dt_of(x) != fr {
                return None;
            }
            let v = ctx.val(x);
            Some(ctx.const_py(v, Some(to)))
        }),
    ));
    // generic float op
    let generic: Vec<Ops> = all
        .iter()
        .copied()
        .filter(|o| !matches!(o, Ops::Param | Ops::Buffer | Ops::Cast | Ops::Bitcast | Ops::Const))
        .collect();
    rules.push((
        UPat::op_var(generic).with_dtype(Some(dtype::FLOATS.to_vec())).allow_all_srcs().named("x"),
        r_pair(|fr, to, ctx, st| {
            let x = sget(st, "x");
            if ctx.dt_of(x) != fr {
                return None;
            }
            let srcs = ctx.srcs(x).to_vec();
            let new_srcs: Vec<UopId> = srcs
                .into_iter()
                .map(|s| if ctx.dt_of(s) == fr { ctx.cast(s, to) } else { s })
                .collect();
            let (op, _d, _s, arg, tag) = ctx.snapshot(x);
            Some(ctx.uop_tag(op, None, new_srcs, arg, tag))
        }),
    ));
    // STORE with bitcast float val
    rules.push((
        UPat::ops1(Ops::Store)
            .with_src_tup(vec![
                UPat::var(Some("idx"), None),
                UPat::ops1(Ops::Bitcast).with_dtype(Some(dtype::FLOATS.to_vec())).named("val"),
            ])
            .allow_all_srcs()
            .named("st"),
        r_pair(|fr, _to, ctx, st| {
            let (st_, idx, val) = (sget(st, "st"), sget(st, "idx"), sget(st, "val"));
            if ctx.dt_of(val) != fr || !matches!(ctx.tag_of(idx), Arg::Dtype(d) if *d == fr) {
                return None;
            }
            let vs = ctx.srcs(val).to_vec();
            let b = ctx.bitcast(vs[0], f2f_dt(fr));
            Some(store_with(ctx, st_, idx, b))
        }),
    ));
    // STORE with float val
    rules.push((
        UPat::ops1(Ops::Store)
            .with_src_tup(vec![
                UPat::var(Some("idx"), None),
                UPat::var(Some("val"), Some(dtype::FLOATS.to_vec())),
            ])
            .allow_all_srcs()
            .named("st"),
        r_pair(|fr, to, ctx, st| {
            let (st_, idx, val) = (sget(st, "st"), sget(st, "idx"), sget(st, "val"));
            if ctx.dt_of(val) != to || !matches!(ctx.tag_of(idx), Arg::Dtype(d) if *d == fr) {
                return None;
            }
            Some(f2f_store(ctx, st_, idx, val, fr, to))
        }),
    ));
    PatternMatcher::new(rules)
}

// ---------- pm_dtype_decomps ----------

/// Context for `pm_dtype_decomps` (`ctx=(set(), ren)` in python).
#[derive(Default)]
pub struct DtypeDecompCtx {
    pub dtypes: HashSet<DTypeId>,
    pub supported: Vec<DTypeId>,
    pub emulated: Vec<DTypeId>,
}

fn all_ops() -> Vec<Ops> {
    (0..=79u16).filter_map(Ops::from_u16).collect()
}

/// `EMULATED_DTYPES` env (helpers.py:249): comma-separated dtype attrs.
pub fn emulated_dtypes() -> Vec<DTypeId> {
    std::env::var("EMULATED_DTYPES")
        .unwrap_or_default()
        .split(',')
        .filter(|s| !s.is_empty())
        .filter_map(dtype::to_dtype)
        .collect()
}

fn should_emulate(dt: DTypeId, supported: &HashSet<DTypeId>, emulated: &[DTypeId]) -> bool {
    emulated.contains(&dt) || !supported.contains(&dt)
}

/// `do_dtype_decomps` (dtype.py:199).
pub fn do_dtype_decomps(ctx: &mut UOpCtx, sink: UopId, dctx: &mut DtypeDecompCtx) -> Option<UopId> {
    let supported: HashSet<DTypeId> = dctx.supported.iter().copied().collect();
    let emulated: Vec<DTypeId> = dctx.emulated.clone();
    let mut dts: Vec<DTypeId> = dctx
        .dtypes
        .iter()
        .copied()
        .filter(|&dt| should_emulate(dt, &supported, &emulated))
        .collect();
    dts.sort_by_key(|d| (d.priority(), d.0));
    dts.dedup();
    let mut cur = sink;
    let mut changed = false;
    for fr in dts {
        let to = if fr == dtype::INT64 {
            dtype::INT32
        } else if !should_emulate(dtype::FLOAT16, &supported, &emulated) && dtype::FP8S.contains(&fr) {
            dtype::FLOAT16
        } else {
            dtype::FLOAT32
        };
        let is_float = dtype::FLOATS.contains(&fr);
        let pm = if is_float { pm_float_decomp() } else { pm_long_decomp() };
        let ctxv = if is_float { CtxVal::Pair(fr, to) } else { CtxVal::L2i(new_l2i()) };
        cur = graph_rewrite(ctx, cur, &pm, true, None, false, false, ctxv)
            .unwrap_or_else(|e| panic!("TGERR: decomp {fr:?} -> {to:?} failed: {e}"));
        changed = true;
    }
    dctx.dtypes.clear();
    if changed {
        Some(cur)
    } else {
        None
    }
}

/// `pm_dtype_decomps` (dtype.py:209).
pub fn pm_dtype_decomps() -> PatternMatcher {
    let detect: Vec<DTypeId> = dtype::FP8S
        .iter()
        .copied()
        .chain([dtype::BFLOAT16, dtype::FLOAT16, dtype::INT64, dtype::UINT64])
        .collect();
    PatternMatcher::new(vec![
        (
            UPat::op_var(all_ops()).with_dtype(Some(detect)).named("x"),
            crate::uop::upat::r_decomp(|dctx, ctx, st| {
                let x = sget(st, "x");
                let dt = ctx.dt_of(x);
                let dt = if dt == dtype::UINT64 { dtype::INT64 } else { dt };
                dctx.dtypes.insert(dt);
                None
            }),
        ),
        (
            UPat::ops1(Ops::Sink).named("sink"),
            crate::uop::upat::r_decomp(|dctx, ctx, st| do_dtype_decomps(ctx, sget(st, "sink"), dctx)),
        ),
    ])
}
