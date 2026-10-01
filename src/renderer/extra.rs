//! `renderer/cstyle.py:76-102` — non-native float emulation.
//!
//! cstyle targets like OpenCL store bf16/fp8/fp16 as integers and have no native
//! arithmetic for them. `create_non_native_float_pats` computes their ALU (and
//! their casts) through f32, casting back to the storage dtype;
//! `pm_manual_bf16_cast` does the f32<->bf16 conversion by bit manipulation.
//! A renderer exposes these as its `extra_matcher`, which the codegen pipeline
//! runs in the "final rewrite" stage (`codegen/__init__.py:377-379`).

use crate::dtype::{self, ConstVal, DTypeId};
use crate::uop::ctx::UOpCtx;
use crate::uop::ops::{group, Ops};
use crate::uop::sym::{r, sget};
use crate::uop::upat::{opset_diff, PatternMatcher, RuleFn, UPat};
use crate::uop::UopId;

fn f32() -> DTypeId {
    dtype::FLOAT32
}

fn cint(ctx: &mut UOpCtx, v: i128) -> UopId {
    ctx.const_py(ConstVal::Int(v), None)
}

/// `create_non_native_float_pats` (cstyle.py:76) — compute `dts` through f32.
pub fn create_non_native_float_pats(dts: &[DTypeId], casting: bool) -> PatternMatcher {
    let dts_v = dts.to_vec();
    let mut rules: Vec<(UPat, RuleFn)> = vec![
        // WHERE on a non-native dtype: both arms in f32, casts around it.
        (
            UPat::op_var(vec![Ops::Where])
                .with_dtype(Some(dts_v.clone()))
                .with_src_tup(vec![
                    UPat::var(Some("b"), None),
                    UPat::var(Some("x"), None),
                    UPat::var(Some("y"), None),
                ])
                .named("w"),
            r(|ctx, st, _| {
                let w = sget(st, "w");
                let b = sget(st, "b");
                let x = sget(st, "x");
                let y = sget(st, "y");
                let dt = ctx.dt_of(&w);
                let cx = ctx.cast(x, f32());
                let cy = ctx.cast(y, f32());
                let wh = ctx.where_(b, cx, cy);
                Some(ctx.cast(wh, dt))
            }),
        ),
        // every other ALU op on a non-native dtype: srcs to f32, rebuild, cast back.
        (
            UPat::op_var(opset_diff(group::ALU, &[Ops::Where]))
                .with_dtype(Some(dts_v.clone()))
                .named("x"),
            r(|ctx, st, _| {
                let x = sget(st, "x");
                let (op, dt, srcs, arg, _tag) = ctx.snapshot(&x);
                let nsrcs: Vec<UopId> = srcs.iter().map(|s| ctx.cast(s.clone(), f32())).collect();
                // python rebuilds with arg and lets the dtype re-derive (f32)
                let mid = ctx.uop(op, None, nsrcs, arg);
                Some(ctx.cast(mid, dt))
            }),
        ),
        // comparison (bool result) whose two operands are non-native: compare in f32.
        (
            UPat::op_var(group::ALU.to_vec())
                .with_dtype(Some(vec![dtype::BOOL]))
                .with_src_tup(vec![
                    UPat::var(Some("x"), Some(dts_v.clone())),
                    UPat::var(Some("y"), Some(dts_v.clone())),
                ])
                .named("alu"),
            r(|ctx, st, _| {
                let alu = sget(st, "alu");
                let x = sget(st, "x");
                let y = sget(st, "y");
                let (op, _dt, _srcs, arg, _tag) = ctx.snapshot(&alu);
                let cx = ctx.cast(x, f32());
                let cy = ctx.cast(y, f32());
                Some(ctx.uop(op, None, vec![cx, cy], arg))
            }),
        ),
    ];
    if casting {
        // CAST(to non-native, x) => x.cast(f32).cast(to) when x isn't already f32
        rules.push((
            UPat::ops1(Ops::Cast)
                .with_dtype(Some(dts_v.clone()))
                .with_src_tup(vec![UPat::var(Some("x"), None)])
                .named("y"),
            r(|ctx, st, _| {
                let y = sget(st, "y");
                let x = sget(st, "x");
                if ctx.dt_of(&x) == f32() {
                    return None;
                }
                let dt = ctx.dt_of(&y);
                let cx = ctx.cast(x, f32());
                Some(ctx.cast(cx, dt))
            }),
        ));
        // CAST(any, y) where y is non-native => y.cast(f32).cast(to)
        rules.push((
            UPat::ops1(Ops::Cast)
                .with_src_tup(vec![UPat::var(Some("y"), Some(dts_v.clone()))])
                .named("x"),
            r(|ctx, st, _| {
                let x = sget(st, "x");
                let y = sget(st, "y");
                if ctx.dt_of(&y) == f32() {
                    return None;
                }
                let dt = ctx.dt_of(&x);
                let cy = ctx.cast(y, f32());
                Some(ctx.cast(cy, dt))
            }),
        ));
    }
    PatternMatcher::new(rules)
}

/// `cast_float_to_bf16` (cstyle.py:91) — round-to-nearest-even f32 -> bf16 bits.
fn cast_float_to_bf16(ctx: &mut UOpCtx, x: UopId) -> UopId {
    let xu = ctx.bitcast(x, dtype::UINT32);
    // (-x & 0x7f800000) != 0 : the value is normal
    let m_sel = cint(ctx, 0x7f800000);
    let zero = cint(ctx, 0);
    let neg = ctx.neg(xu.clone());
    let sel = ctx.and_(neg, m_sel);
    let is_normal = ctx.ne(sel, zero);
    // normal: x + ((x >> 16) & 1) + 0x7fff   (round to nearest even)
    let c16 = cint(ctx, 16);
    let one = cint(ctx, 1);
    let sh = ctx.shr(xu.clone(), c16);
    let lo = ctx.and_(sh, one);
    let sum = ctx.add(xu.clone(), lo);
    let c7fff = cint(ctx, 0x7fff);
    let round = ctx.add(sum, c7fff);
    // subnormal: (x & 0xffff) != 0 ? (x | 0x10000) : x
    let m_frac = cint(ctx, 0xffff);
    let zero2 = cint(ctx, 0);
    let frac = ctx.and_(xu.clone(), m_frac);
    let cond = ctx.ne(frac, zero2);
    let c10000 = cint(ctx, 0x10000);
    let or = ctx.or_(xu.clone(), c10000);
    let sticky = ctx.where_(cond, or, xu.clone());
    let rounded = ctx.where_(is_normal, round, sticky);
    let c16b = cint(ctx, 16);
    let hi = ctx.shr(rounded, c16b);
    let us = ctx.cast(hi, dtype::UINT16);
    ctx.bitcast(us, dtype::BFLOAT16)
}

/// `pm_manual_bf16_cast` (cstyle.py:98) — avoid compiler bf16 intrinsics.
pub fn pm_manual_bf16_cast() -> PatternMatcher {
    PatternMatcher::new(vec![
        // CAST(float, bf16 x) => (x.bitcast(ushort).cast(uint) << 16).bitcast(float)
        (
            UPat::ops1(Ops::Cast)
                .with_dtype(Some(vec![dtype::FLOAT32]))
                .with_src_tup(vec![UPat::var(Some("x"), Some(vec![dtype::BFLOAT16]))]),
            r(|ctx, st, _| {
                let x = sget(st, "x");
                let u16_ = ctx.bitcast(x, dtype::UINT16);
                let u32_ = ctx.cast(u16_, dtype::UINT32);
                let c16 = cint(ctx, 16);
                let sh = ctx.shl(u32_, c16);
                Some(ctx.bitcast(sh, dtype::FLOAT32))
            }),
        ),
        // CAST(bf16, float x) => cast_float_to_bf16(x)
        (
            UPat::ops1(Ops::Cast)
                .with_dtype(Some(vec![dtype::BFLOAT16]))
                .with_src_tup(vec![UPat::var(Some("x"), Some(vec![dtype::FLOAT32]))]),
            r(|ctx, st, _| {
                let x = sget(st, "x");
                Some(cast_float_to_bf16(ctx, x))
            }),
        ),
    ])
}

/// `ClangRenderer.extra_matcher` (cstyle.py:278-281) — the f64/f16/bf16
/// conversion fixups LLVM needs on systems without native support, plus bf16.
pub fn clang_extra_matcher() -> PatternMatcher {
    let d = |a: DTypeId, b: DTypeId| {
        (
            UPat::ops1(Ops::Cast)
                .with_dtype(Some(vec![b]))
                .with_src_tup(vec![UPat::var(Some("x"), Some(vec![a]))]),
            r(move |ctx, st, _| {
                let x = sget(st, "x");
                let to_f32 = ctx.cast(x, f32());
                Some(ctx.cast(to_f32, b))
            }) as RuleFn,
        )
    };
    PatternMatcher::new(vec![
        d(dtype::FLOAT64, dtype::FLOAT16),
        d(dtype::FLOAT64, dtype::BFLOAT16),
        d(dtype::BFLOAT16, dtype::FLOAT16),
    ])
    .add(&create_non_native_float_pats(&[dtype::BFLOAT16], true))
    .add(&pm_manual_bf16_cast())
}
