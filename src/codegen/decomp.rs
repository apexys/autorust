//! `codegen/decomp/op.py` port: decomposition patterns applied in the codegen
//! "late decompositions" stage. Currently only `threefry2x32`, which lowers
//! `Ops::THREEFRY(uint64, x, key)` into primitive uint32 ALU that the
//! interpreter/renderer execute (no hardware supports THREEFRY directly).

use crate::dtype::{self, DTypeId};
use crate::dtype::ConstVal;
use crate::uop::ctx::UOpCtx;
use crate::uop::ops::Ops;
use crate::uop::sym::{r, sget};
use crate::uop::upat::{PatternMatcher, UPat};
use crate::uop::UopId;

fn c_int(ctx: &mut UOpCtx, v: i64) -> UopId {
    ctx.const_py(ConstVal::Int(v as i128), None)
}

/// `threefry2x32(x, key)` (decomp/op.py:48): split uint64 `x`,`key` into two
/// uint32 lanes, run 5 rounds of the two-cycle Threefry, recombine to uint64.
fn threefry2x32(ctx: &mut UOpCtx, x: UopId, key: UopId) -> UopId {
    let u32: DTypeId = dtype::UINT32;
    let u64: DTypeId = dtype::UINT64;

    let c32 = c_int(ctx, 32);
    let x1s = ctx.binop(Ops::Shr, x, c32);
    let x1 = ctx.cast(x1s, u32);
    let x0 = ctx.cast(x, u32);

    let key1s = ctx.binop(Ops::Shr, key, c32);
    let key1 = ctx.cast(key1s, u32);
    let key0 = ctx.cast(key, u32);

    let rotations: [[i64; 4]; 2] = [[13, 15, 26, 6], [17, 29, 16, 24]];
    let kx1 = ctx.binop(Ops::Xor, key0, key1);
    let cx = c_int(ctx, 0x1BD11BDA);
    let kx = ctx.binop(Ops::Xor, kx1, cx);
    let ks: [UopId; 3] = [key1, kx, key0];

    let mut xr0 = ctx.binop(Ops::Add, x0, ks[2]);
    let mut xr1 = ctx.binop(Ops::Add, x1, ks[0]);

    for i in 0..5 {
        for &rot in rotations[i % 2].iter() {
            // new0 = xr0 + xr1 ; new1 = new0 ^ ((xr1<<r) + (xr1>>(32-r)))
            let new0 = ctx.binop(Ops::Add, xr0, xr1);
            let cr = c_int(ctx, rot);
            let c32r = c_int(ctx, 32 - rot);
            let shl = ctx.binop(Ops::Shl, xr1, cr);
            let shr = ctx.binop(Ops::Shr, xr1, c32r);
            let rotsum = ctx.binop(Ops::Add, shl, shr);
            let new1 = ctx.binop(Ops::Xor, new0, rotsum);
            xr0 = new0;
            xr1 = new1;
        }
        xr0 = ctx.binop(Ops::Add, xr0, ks[i % 3]);
        let ci = c_int(ctx, i as i64 + 1);
        let k = ctx.binop(Ops::Add, ks[(i + 1) % 3], ci);
        xr1 = ctx.binop(Ops::Add, xr1, k);
    }

    let c32b = c_int(ctx, 32);
    let xr1_64 = ctx.cast(xr1, u64);
    let hi = ctx.binop(Ops::Shl, xr1_64, c32b);
    let xr0_64 = ctx.cast(xr0, u64);
    let res = ctx.binop(Ops::Or, hi, xr0_64);
    res
}

/// `get_simplifying_rewrite_patterns` subset needed for RNG: the THREEFRY
/// decomposition (tinygrad registers it only when the renderer lacks THREEFRY,
/// which is always the case for the interp backend).
pub fn pm_decomp() -> PatternMatcher {
    let key_pat = UPat::var(Some("key"), None);
    let x_pat = UPat::var(Some("x"), None);
    let pat = UPat::op_var(vec![Ops::Threefry])
        .with_dtype(Some(vec![dtype::UINT64]))
        .with_src_tup(vec![x_pat, key_pat]);
    PatternMatcher::new(vec![(
        pat,
        r(|ctx, st, _| {
            let x = sget(st, "x");
            let key = sget(st, "key");
            Some(threefry2x32(ctx, x, key))
        }),
    )])
}
