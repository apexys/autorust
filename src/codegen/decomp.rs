//! `codegen/decomp/op.py` port: decomposition patterns applied in the codegen
//! "late decompositions" stage. Currently only `threefry2x32`, which lowers
//! `Ops::THREEFRY(uint64, x, key)` into primitive uint32 ALU that the
//! interpreter/renderer execute (no hardware supports THREEFRY directly).

use crate::dtype::ConstVal;
use crate::dtype::{self, DTypeId};
use crate::uop::UopId;
use crate::uop::ctx::UOpCtx;
use crate::uop::ops::Ops;
use crate::uop::sym::{r, sget};
use crate::uop::upat::{PatternMatcher, RuleFn, UPat};

fn c_int(ctx: &mut UOpCtx, v: i64) -> UopId {
    ctx.const_py(ConstVal::Int(v as i128), None)
}

/// `threefry2x32(x, key)` (decomp/op.py:48): split uint64 `x`,`key` into two
/// uint32 lanes, run 5 rounds of the two-cycle Threefry, recombine to uint64.
fn threefry2x32(ctx: &mut UOpCtx, x: UopId, key: UopId) -> UopId {
    let u32: DTypeId = dtype::UINT32;
    let u64: DTypeId = dtype::UINT64;

    let c32 = c_int(ctx, 32);
    let x1s = ctx.binop(Ops::Shr, x.clone(), c32.clone());
    let x1 = ctx.cast(x1s, u32);
    let x0 = ctx.cast(x, u32);

    let key1s = ctx.binop(Ops::Shr, key.clone(), c32);
    let key1 = ctx.cast(key1s, u32);
    let key0 = ctx.cast(key, u32);

    let rotations: [[i64; 4]; 2] = [[13, 15, 26, 6], [17, 29, 16, 24]];
    let kx1 = ctx.binop(Ops::Xor, key0.clone(), key1.clone());
    let cx = c_int(ctx, 0x1BD11BDA);
    let kx = ctx.binop(Ops::Xor, kx1, cx);
    let ks: [UopId; 3] = [key1, kx, key0];

    let mut xr0 = ctx.binop(Ops::Add, x0, ks[2].clone());
    let mut xr1 = ctx.binop(Ops::Add, x1, ks[0].clone());

    for i in 0..5 {
        for &rot in rotations[i % 2].iter() {
            // new0 = xr0 + xr1 ; new1 = new0 ^ ((xr1<<r) + (xr1>>(32-r)))
            let new0 = ctx.binop(Ops::Add, xr0, xr1.clone());
            let cr = c_int(ctx, rot);
            let c32r = c_int(ctx, 32 - rot);
            let shl = ctx.binop(Ops::Shl, xr1.clone(), cr);
            let shr = ctx.binop(Ops::Shr, xr1, c32r);
            let rotsum = ctx.binop(Ops::Add, shl, shr);
            let new1 = ctx.binop(Ops::Xor, new0.clone(), rotsum);
            xr0 = new0;
            xr1 = new1;
        }
        // xr = [xr0 + ks[i%3], xr1 + ks[(i+1)%3] + i + 1]  (left-assoc, exactly
        // as python: ((xr1 + ks) + i) + 1)
        let t0 = ctx.binop(Ops::Add, xr0, ks[i % 3].clone());
        let t1 = ctx.binop(Ops::Add, xr1, ks[(i + 1) % 3].clone());
        let ci = c_int(ctx, i as i64);
        let t1 = ctx.binop(Ops::Add, t1, ci);
        let one = c_int(ctx, 1);
        let t1 = ctx.binop(Ops::Add, t1, one);
        xr0 = t0;
        xr1 = t1;
    }

    let c32b = c_int(ctx, 32);
    let xr1_64 = ctx.cast(xr1, u64);
    let hi = ctx.binop(Ops::Shl, xr1_64, c32b);
    let xr0_64 = ctx.cast(xr0, u64);

    ctx.binop(Ops::Or, hi, xr0_64)
}

/// `powers_of_two` (decomp/op.py:74) — `2**i -> i` for `i in range(64)`; the
/// python rule uses `if (v := powers_of_two.get(c.val, 0))`, so `1` and
/// non-powers yield nothing.
fn pow2_exp(v: i128) -> Option<i64> {
    if v > 1 && v <= (1i128 << 63) && (v & (v - 1)) == 0 {
        Some(v.trailing_zeros() as i64)
    } else {
        None
    }
}

fn const_int(ctx: &UOpCtx, u: UopId) -> Option<i128> {
    match ctx.val(&u) {
        ConstVal::Int(i) => Some(i),
        _ => None,
    }
}

/// `floordiv_to_idiv` (decomp/op.py:64).
fn floordiv_to_idiv(ctx: &mut UOpCtx, a: UopId, b: UopId) -> UopId {
    let (avmin, avmax) = ctx.min_max(&a).expect("TGERR: floordiv min_max");
    let (bvmin, bvmax) = ctx.min_max(&b).expect("TGERR: floordiv min_max");
    let (avmin, avmax) = (avmin.as_i128_or_panic(), avmax.as_i128_or_panic());
    let (bvmin, bvmax) = (bvmin.as_i128_or_panic(), bvmax.as_i128_or_panic());
    let cdiv = ctx.alu2(Ops::Cdiv, a.clone(), b.clone());
    if (avmin >= 0 && bvmin > 0) || (avmax <= 0 && bvmax < 0) {
        return cdiv;
    }
    // a.alu(CMOD, b).ne(0) & (a<0).ne(b<0)
    let cmod = ctx.alu2(Ops::Cmod, a.clone(), b.clone());
    let z0 = ctx.const_py(ConstVal::Int(0), None);
    let rnz = ctx.ne(cmod, z0);
    let za = ctx.const_py(ConstVal::Int(0), None);
    let a_lt = ctx.lt(a, za);
    let zb = ctx.const_py(ConstVal::Int(0), None);
    let b_lt = ctx.lt(b, zb);
    let sign_ne = ctx.ne(a_lt, b_lt);
    let cond = ctx.and_(rnz, sign_ne);
    ctx.sub(cdiv, cond)
}

/// `floormod_to_mod` (decomp/op.py:68).
fn floormod_to_mod(ctx: &mut UOpCtx, a: UopId, b: UopId) -> UopId {
    let (avmin, avmax) = ctx.min_max(&a).expect("TGERR: floormod min_max");
    let (bvmin, bvmax) = ctx.min_max(&b).expect("TGERR: floormod min_max");
    let (avmin, avmax) = (avmin.as_i128_or_panic(), avmax.as_i128_or_panic());
    let (bvmin, bvmax) = (bvmin.as_i128_or_panic(), bvmax.as_i128_or_panic());
    let r = ctx.alu2(Ops::Cmod, a.clone(), b.clone());
    if (avmin >= 0 && bvmin > 0) || (avmax <= 0 && bvmax < 0) {
        return r;
    }
    // r + (r.ne(0) & (a<0).ne(b<0)).where(b, b.const_like(0))
    let z0 = ctx.const_py(ConstVal::Int(0), None);
    let rnz = ctx.ne(r.clone(), z0);
    let za = ctx.const_py(ConstVal::Int(0), None);
    let a_lt = ctx.lt(a, za);
    let zb = ctx.const_py(ConstVal::Int(0), None);
    let b_lt = ctx.lt(b.clone(), zb);
    let sign_ne = ctx.ne(a_lt, b_lt);
    let cond = ctx.and_(rnz, sign_ne);
    let zero_b = ctx.const_like(b.clone(), ConstVal::Int(0));
    let w = ctx.where_(cond, b, zero_b);
    ctx.add(r, w)
}

/// `get_simplifying_rewrite_patterns` (decomp/op.py:76) for this port's renderer
/// op sets. `SHR`/`AND`/`CMPLT` are always present; `THREEFRY` is absent from
/// every ported renderer so it always decomposes. `MAX` is present only in the
/// interpreter set (`python_alu`), so `supports_max` selects its rewrite
/// (decomp/op.py:89: cstyle/PTX omit MAX from `code_for_op`).
pub fn get_simplifying_rewrite_patterns(supports_max: bool) -> PatternMatcher {
    let mut rules: Vec<(UPat, RuleFn)> = vec![];
    // FLOORDIV by 2**y -> x >> y (arithmetic shift is floor division for any sign)
    rules.push((
        UPat::var(Some("x"), Some(dtype::INTS.to_vec())).floordiv(UPat::cvar(
            Some("c"),
            None,
            None,
        )),
        r(|ctx, st, _| {
            let (x, c) = (sget(st, "x"), sget(st, "c"));
            let v = pow2_exp(const_int(ctx, c)?)?;
            let k = ctx.const_py(ConstVal::Int(v as i128), None);
            Some(ctx.shr(x, k))
        }),
    ));
    // a // b -> floordiv_to_idiv
    rules.push((
        UPat::var(Some("a"), None).floordiv(UPat::var(Some("b"), None)),
        r(|ctx, st, _| Some(floordiv_to_idiv(ctx, sget(st, "a"), sget(st, "b")))),
    ));
    // FLOORMOD by 2**y -> x & (2**y - 1)
    rules.push((
        UPat::var(Some("x"), Some(dtype::INTS.to_vec())).mod_(UPat::cvar(Some("c"), None, None)),
        r(|ctx, st, _| {
            let (x, c) = (sget(st, "x"), sget(st, "c"));
            let cv = const_int(ctx, c)?;
            pow2_exp(cv)?;
            let k = ctx.const_py(ConstVal::Int(cv - 1), None);
            Some(ctx.binop(Ops::And, x, k))
        }),
    ));
    // a % b -> floormod_to_mod
    rules.push((
        UPat::var(Some("a"), None).mod_(UPat::var(Some("b"), None)),
        r(|ctx, st, _| Some(floormod_to_mod(ctx, sget(st, "a"), sget(st, "b")))),
    ));
    // THREEFRY(uint64, x, key) -> threefry2x32
    let key_pat = UPat::var(Some("key"), None);
    let x_pat = UPat::var(Some("x"), None);
    let pat = UPat::op_var(vec![Ops::Threefry])
        .with_dtype(Some(vec![dtype::UINT64]))
        .with_src_tup(vec![x_pat, key_pat]);
    rules.push((
        pat,
        r(|ctx, st, _| {
            let x = sget(st, "x");
            let key = sget(st, "key");
            Some(threefry2x32(ctx, x, key))
        }),
    ));
    // MAX(a, b) -> (a < b).where(b, a) (decomp/op.py:89): only when the renderer
    // does not support MAX (cstyle/PTX omit it; the interpreter's python_alu has it).
    if !supports_max {
        rules.push((
            UPat::op_var(vec![Ops::Max]).named("m"),
            r(|ctx, st, _| {
                let m = sget(st, "m");
                let a = ctx.srcs(&m)[0].clone();
                let b = ctx.srcs(&m)[1].clone();
                let cond = ctx.lt(a.clone(), b.clone());
                Some(ctx.where_(cond, b, a))
            }),
        ));
    }
    PatternMatcher::new(rules)
}

/// `pm_decomp` (codegen/__init__.py:365) = `symbolic_simple + get_simplifying_rewrite_patterns`.
/// `supports_max` is the renderer's (`code_for_op` contains `Ops::MAX`); the
/// interpreter has it, the C-style/OpenCL renderers do not.
pub fn pm_decomp(supports_max: bool) -> PatternMatcher {
    crate::uop::sym::symbolic_simple().add(&get_simplifying_rewrite_patterns(supports_max))
}
