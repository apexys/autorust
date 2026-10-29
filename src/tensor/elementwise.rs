//! M3 elementwise mixin (mixin/elementwise.py), UOp-level. Promotion happens
//! through `dtype_from_uop` at construction; the explicit pieces here mirror
//! Python's `_broadcasted` (weak-CONST-keep) / `_binop` and the ops with
//! special bodies (sub/div/mod/fmod/minimum/where/pow/exp/ceil/floor/...).

use crate::dtype::ConstVal;
use crate::dtype::{self};
use crate::uop::ctx::UOpCtx;
use crate::uop::graph::UopId;
use crate::uop::ops::Ops;

/// Python `ufix`: a const operand becomes a CONST uop with its `from_py` dtype
/// (weakint/weakfloat/bool); tensor operands pass through.
pub fn ufix(ctx: &mut UOpCtx, c: ConstVal) -> UopId {
    ctx.const_py(c, None)
}

/// `Tensor._binop(op, x, reverse)`: `_broadcasted` promotion then a bare op
/// ctor (dtype/broadcast resolved by `dtype_from_uop`).
pub fn binop(ctx: &mut UOpCtx, op: Ops, a: UopId, b: UopId) -> UopId {
    ctx.binop(op, a, b)
}

/// `_binop` with a const RHS.
pub fn binop_c(ctx: &mut UOpCtx, op: Ops, a: UopId, c: ConstVal) -> UopId {
    let b = ufix(ctx, c);
    ctx.binop(op, a, b)
}

/// `ElementwiseMixin.neg()`: bool → logical_not, else `self * (-1)`.
pub fn neg(ctx: &mut UOpCtx, x: UopId) -> UopId {
    if dtype::is_bool(ctx.dt_of(&x)) {
        return logical_not(ctx, x);
    }
    binop_c(ctx, Ops::Mul, x, ConstVal::Int(-1))
}

/// `ElementwiseMixin.logical_not()`: `self.cast(bool).ne(True)`.
pub fn logical_not(ctx: &mut UOpCtx, x: UopId) -> UopId {
    let b = ctx.cast(x, dtype::BOOL);
    binop_c(ctx, Ops::Cmpne, b, ConstVal::Bool(true))
}

/// `ElementwiseMixin.sub`: `a.alu(ADD, -b)` (no second promote — `_broadcasted`
/// already ran; `-b` is `neg`, which for bools is a logical_not).
pub fn sub(ctx: &mut UOpCtx, a: UopId, b: UopId) -> UopId {
    let (a, b) = ctx.broadcasted(a, b);
    let nb = neg(ctx, b);
    ctx.alu2(Ops::Add, a, nb)
}

/// `sub` with a const RHS.
pub fn sub_c(ctx: &mut UOpCtx, a: UopId, c: ConstVal) -> UopId {
    let b = ufix(ctx, c);
    sub(ctx, a, b)
}

/// `ElementwiseMixin.div(x, reverse, rounding_mode)` (elementwise.py:225-252).
pub fn div(ctx: &mut UOpCtx, a: UopId, b: UopId, rounding_mode: Option<&str>) -> UopId {
    let (a, b) = ctx.broadcasted(a, b);
    if dtype::is_int(ctx.dt_of(&a)) && dtype::is_int(ctx.dt_of(&b)) {
        match rounding_mode {
            Some("trunc") => return ctx.alu2(Ops::Cdiv, a, b),
            Some("floor") => return ctx.alu2(Ops::Floordiv, a, b),
            _ => {}
        }
    }
    let a = if dtype::is_int(ctx.dt_of(&a)) || dtype::is_bool(ctx.dt_of(&a)) {
        ctx.cast(a, dtype::default_float())
    } else {
        a
    };
    let recip = ctx.alu1(Ops::Reciprocal, b);
    let d = ctx.alu2(Ops::Mul, a, recip);
    match rounding_mode {
        None => d,
        Some("trunc") => ctx.alu1(Ops::Trunc, d),
        Some("floor") => floor(ctx, d),
        _ => panic!("TGERR: rounding_mode {:?} is not supported", rounding_mode),
    }
}

/// `div` with a const divisor.
pub fn div_c(ctx: &mut UOpCtx, a: UopId, c: ConstVal, rounding_mode: Option<&str>) -> UopId {
    let b = ufix(ctx, c);
    div(ctx, a, b, rounding_mode)
}

/// `ElementwiseMixin.mod`: int&int → FLOORMOD, else `a - a.div(b, floor) * b`.
pub fn mod_(ctx: &mut UOpCtx, a: UopId, b: UopId) -> UopId {
    let (a, b) = ctx.broadcasted(a, b);
    if dtype::is_int(ctx.dt_of(&a)) && dtype::is_int(ctx.dt_of(&b)) {
        return ctx.alu2(Ops::Floormod, a, b);
    }
    let d = div(ctx, a.clone(), b.clone(), Some("floor"));
    let db = ctx.binop(Ops::Mul, d, b);
    sub(ctx, a, db)
}

/// `ElementwiseMixin.fmod`: int&int → CMOD, else `a - a.div(b, trunc) * b`.
pub fn fmod(ctx: &mut UOpCtx, a: UopId, b: UopId) -> UopId {
    let (a, b) = ctx.broadcasted(a, b);
    if dtype::is_int(ctx.dt_of(&a)) && dtype::is_int(ctx.dt_of(&b)) {
        return ctx.alu2(Ops::Cmod, a, b);
    }
    let d = div(ctx, a.clone(), b.clone(), Some("trunc"));
    let db = ctx.binop(Ops::Mul, d, b);
    sub(ctx, a, db)
}

/// `ElementwiseMixin.maximum`: `_binop(Ops.MAX, x)`.
pub fn maximum(ctx: &mut UOpCtx, a: UopId, b: UopId) -> UopId {
    binop(ctx, Ops::Max, a, b)
}

/// `ElementwiseMixin.minimum` (elementwise.py:389-404): the XOR-max identity —
/// ints: `(t ^ k).max(x ^ k) ^ k` with `k = dt.min + dt.max`; floats: `-(-t).max(-x)`.
pub fn minimum(ctx: &mut UOpCtx, a: UopId, b: UopId) -> UopId {
    let (t, x) = ctx.broadcasted(a, b);
    let dt = dtype::least_upper_dtype(&[ctx.dt_of(&t), ctx.dt_of(&x)]);
    if dtype::is_float(dt) {
        let (nt, nx) = (neg(ctx, t), neg(ctx, x));
        let mx = ctx.alu2(Ops::Max, nt, nx);
        return neg(ctx, mx);
    }
    let (min, max) = (dtype::dtype_min(dt), dtype::dtype_max(dt));
    let k = crate::uop::alu::py_add_const(&min, &max);
    let k_u = ctx.const_py(k, Some(dt));
    let tx = ctx.alu2(Ops::Xor, t, k_u.clone());
    let xk = ctx.alu2(Ops::Xor, x, k_u.clone());
    let mx = ctx.alu2(Ops::Max, tx, xk);
    ctx.alu2(Ops::Xor, mx, k_u)
}

/// `ElementwiseMixin.where(cond, x, y)` (elementwise.py:422-436): promote `x`/`y`
/// against each other, then `cond.alu(WHERE, x, y)`.
pub fn where_(ctx: &mut UOpCtx, cond: UopId, x: UopId, y: UopId) -> UopId {
    let (x, y) = ctx.broadcasted(x, y);
    ctx.alu3(Ops::Where, cond, x, y)
}

/// `ElementwiseMixin.pow(x, reverse)` (elementwise.py:545-550).
pub fn pow(ctx: &mut UOpCtx, base: UopId, exponent: UopId) -> UopId {
    let (base, exponent) = ctx.broadcasted(base, exponent);
    ctx.alu2(Ops::Pow, base, exponent)
}

/// `pow` with a const exponent.
pub fn pow_c(ctx: &mut UOpCtx, base: UopId, c: ConstVal) -> UopId {
    let e = ufix(ctx, c);
    pow(ctx, base, e)
}

// ***** unary ops *****

/// `Tensor.reciprocal` — `1/self`.
pub fn reciprocal(ctx: &mut UOpCtx, x: UopId) -> UopId {
    ctx.alu1(Ops::Reciprocal, x)
}

/// `Tensor.sqrt` — square root.
pub fn sqrt(ctx: &mut UOpCtx, x: UopId) -> UopId {
    ctx.alu1(Ops::Sqrt, x)
}

/// `Tensor.sin` — sine.
pub fn sin(ctx: &mut UOpCtx, x: UopId) -> UopId {
    ctx.alu1(Ops::Sin, x)
}

/// `tanh()`: `(exp(2x) - 1) / (exp(2x) + 1)`.
pub fn tanh(ctx: &mut UOpCtx, x: UopId) -> UopId {
    // exp(2x) == exp2(2x/ln2)
    let two = binop_c(ctx, Ops::Mul, x, ConstVal::Float(2.0));
    let scaled = binop_c(
        ctx,
        Ops::Mul,
        two,
        ConstVal::Float(1.0 / std::f64::consts::LN_2),
    );
    let exp_2x = ctx.alu1(Ops::Exp2, scaled);
    let one = ConstVal::Float(1.0);
    let one_const = ctx.const_py(one, None);
    let num = ctx.alu2(Ops::Sub, exp_2x.clone(), one_const.clone());
    let den = ctx.alu2(Ops::Add, exp_2x, one_const);
    let recip = ctx.alu1(Ops::Reciprocal, den);
    ctx.alu2(Ops::Mul, num, recip)
}

/// `Tensor.log2` — base-2 logarithm.
pub fn log2(ctx: &mut UOpCtx, x: UopId) -> UopId {
    ctx.alu1(Ops::Log2, x)
}

/// `Tensor.exp2` — base-2 exponential.
pub fn exp2(ctx: &mut UOpCtx, x: UopId) -> UopId {
    ctx.alu1(Ops::Exp2, x)
}

/// `Tensor.trunc` — round toward zero.
pub fn trunc(ctx: &mut UOpCtx, x: UopId) -> UopId {
    ctx.alu1(Ops::Trunc, x)
}

/// `exp()`: `cast(lub(self, float32)).mul(1/log(2)).exp2().cast(self.dtype)`.
pub fn exp(ctx: &mut UOpCtx, x: UopId) -> UopId {
    let dt = ctx.dt_of(&x);
    let up = dtype::least_upper_dtype(&[dt, dtype::default_float()]);
    let casted = ctx.cast(x, up);
    let scaled = binop_c(
        ctx,
        Ops::Mul,
        casted,
        ConstVal::Float(1.0 / std::f64::consts::LN_2),
    );
    let e = ctx.alu1(Ops::Exp2, scaled);
    ctx.cast(e, dt)
}

/// `log()`: natural log = `self.cast(...).log2() * ln2` — cast to lub(float32)
/// first, multiply by ln(2), cast back (matches tinygrad `tensor.log`).
pub fn log(ctx: &mut UOpCtx, x: UopId) -> UopId {
    let dt = ctx.dt_of(&x);
    let up = dtype::least_upper_dtype(&[dt, dtype::default_float()]);
    let casted = ctx.cast(x, up);
    let l2 = ctx.alu1(Ops::Log2, casted);
    let scaled = binop_c(ctx, Ops::Mul, l2, ConstVal::Float(std::f64::consts::LN_2));
    ctx.cast(scaled, dt)
}

/// `ceil()`: `(self > (b := self.trunc())).where(b+1, b)`.
pub fn ceil(ctx: &mut UOpCtx, x: UopId) -> UopId {
    let b = trunc(ctx, x.clone());
    let gt = binop(ctx, Ops::Cmplt, b.clone(), x); // x > b  ≡  b < x
    let b1 = binop_c(ctx, Ops::Add, b.clone(), ConstVal::Int(1));
    where_(ctx, gt, b1, b)
}

/// `floor()`: `(self < (b := self.trunc())).where(b-1, b)`.
pub fn floor(ctx: &mut UOpCtx, x: UopId) -> UopId {
    let b = trunc(ctx, x.clone());
    let lt = binop(ctx, Ops::Cmplt, x, b.clone());
    let b1 = binop_c(ctx, Ops::Add, b.clone(), ConstVal::Int(-1));
    where_(ctx, lt, b1, b)
}

/// `square()`: `self * self`.
pub fn square(ctx: &mut UOpCtx, x: UopId) -> UopId {
    binop(ctx, Ops::Mul, x.clone(), x)
}

/// comparison sugar — python spellings: `lt`/`ne` are bare binops, the rest
/// compose (`eq = ne.logical_not()`, `le = !gt`, `ge = !lt`, `gt` reverse-lt).
/// `Tensor.lt` — `a < b`.
pub fn lt(ctx: &mut UOpCtx, a: UopId, b: UopId) -> UopId {
    binop(ctx, Ops::Cmplt, a, b)
}
/// `Tensor.gt` — `a > b`, built as `b < a`.
pub fn gt(ctx: &mut UOpCtx, a: UopId, b: UopId) -> UopId {
    binop(ctx, Ops::Cmplt, b, a)
}
/// `Tensor.ne` — `a != b`.
pub fn ne(ctx: &mut UOpCtx, a: UopId, b: UopId) -> UopId {
    binop(ctx, Ops::Cmpne, a, b)
}
/// `Tensor.eq` — `a == b`, built as `!(a != b)`.
pub fn eq(ctx: &mut UOpCtx, a: UopId, b: UopId) -> UopId {
    let n = ne(ctx, a, b);
    logical_not(ctx, n)
}
/// `Tensor.__le__` — `a <= b`, built as `!(a > b)`.
pub fn le(ctx: &mut UOpCtx, a: UopId, b: UopId) -> UopId {
    let g = gt(ctx, a, b);
    logical_not(ctx, g)
}
/// `Tensor.__ge__` — `a >= b`, built as `!(a < b)`.
pub fn ge(ctx: &mut UOpCtx, a: UopId, b: UopId) -> UopId {
    let l = lt(ctx, a, b);
    logical_not(ctx, l)
}
/// `Tensor.__and__` — bitwise/logical and.
pub fn and(ctx: &mut UOpCtx, a: UopId, b: UopId) -> UopId {
    binop(ctx, Ops::And, a, b)
}
/// `Tensor.__or__` — bitwise/logical or.
pub fn or(ctx: &mut UOpCtx, a: UopId, b: UopId) -> UopId {
    binop(ctx, Ops::Or, a, b)
}
/// `Tensor.__xor__` — bitwise/logical xor.
pub fn xor(ctx: &mut UOpCtx, a: UopId, b: UopId) -> UopId {
    binop(ctx, Ops::Xor, a, b)
}
