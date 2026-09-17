//! `python_alu` + `exec_alu` (uop/ops.py:1276-1304) — const folding arithmetic.
//!
//! Operands are `ConstVal` (Python `ConstType = bool|int|float|Invalid`, ints
//! exact via i128/BigInt). Vectors (STACK consts) are handled by the caller
//! like Python's `exec_alu` tuple branch. `FDIV` has no `python_alu` entry in
//! Python (KeyError → rewrite error); mirrored here as an error.

use crate::dtype::{ConstVal, DTypeId};

use super::ops::Ops;

/// Helpers from `tinygrad/helpers.py` (int semantics; Python floor division).
/// Python `+` on ConstVals (int+int -> Int/Big, float wins, bool coerces).
pub fn py_add_const(x: &crate::dtype::ConstVal, y: &crate::dtype::ConstVal) -> crate::dtype::ConstVal {
    use crate::dtype::ConstVal::*;
    match (x, y) {
        (Float(a), b) | (b, Float(a)) => Float(a + b.as_f64()),
        (Int(a), Int(b)) => Int(a.wrapping_add(*b)),
        (Big(a), Big(b)) => Big(a.add(b)),
        (Big(a), Int(b)) | (Int(b), Big(a)) => {
            let other = crate::helpers::bigint::BigInt::from_i128(*b);
            Big(a.add(&other))
        }
        (Bool(a), other) | (other, Bool(a)) => {
            let av = if *a { 1i128 } else { 0 };
            let bv = match other {
                Bool(v) => {
                    if *v {
                        1
                    } else {
                        0
                    }
                }
                Int(v) => *v,
                _ => panic!("TGERR: py_add_const unsupported bool mix"),
            };
            Int(av.wrapping_add(bv))
        }
        _ => panic!("TGERR: py_add_const unsupported"),
    }
}

pub fn cdiv(x: i128, y: i128) -> i128 {
    if y == 0 {
        return 0;
    }
    let q = (x.abs()).wrapping_div(y.abs());
    if (x < 0) ^ (y < 0) {
        -q
    } else {
        q
    }
}

pub fn cmod(x: i128, y: i128) -> i128 {
    // x - cdiv(x,y)*y (wrapping mul to avoid spurious overflow on sentinels)
    x.wrapping_sub(cdiv(x, y).wrapping_mul(y))
}

pub fn floordiv(x: i128, y: i128) -> i128 {
    if y == 0 {
        return 0;
    }
    let q = x / y;
    let r = x % y;
    if r != 0 && ((r < 0) ^ (y < 0)) {
        q - 1
    } else {
        q
    }
}

pub fn floormod(x: i128, y: i128) -> i128 {
    x.wrapping_sub(floordiv(x, y).wrapping_mul(y))
}

/// Python `%` on ints (sign follows the divisor).
pub fn py_mod(x: i128, y: i128) -> i128 {
    floormod(x, y)
}

fn gcd_i128(mut a: i128, mut b: i128) -> i128 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// Python `math.prod` semantics over an i128 accumulator (checked).
pub fn prod_checked(vals: &[i128]) -> Option<i128> {
    let mut acc: i128 = 1;
    for v in vals {
        acc = acc.checked_mul(*v)?;
    }
    Some(acc)
}

/// Python `math.gcd(*xs)`.
pub fn gcd_all(vals: &[i128]) -> i128 {
    vals.iter().cloned().fold(0, gcd_i128)
}

// ---- numeric tower: Python semantics over ConstVal ----

fn is_float_v(v: &ConstVal) -> bool {
    matches!(v, ConstVal::Float(_))
}

fn as_f(v: &ConstVal) -> f64 {
    match v {
        ConstVal::Bool(b) => {
            if *b {
                1.0
            } else {
                0.0
            }
        }
        ConstVal::Int(i) => *i as f64,
        ConstVal::Big(b) => b.to_f64(),
        ConstVal::Float(f) => *f,
        ConstVal::Invalid => f64::NAN,
    }
}

/// Python `int(x)` on a const (exact; NaN raises in Python → None).
fn py_int(v: &ConstVal) -> Option<ConstVal> {
    match v {
        ConstVal::Bool(b) => Some(ConstVal::Int(if *b { 1 } else { 0 })),
        ConstVal::Int(_) => Some(v.clone()),
        ConstVal::Big(_) => Some(v.clone()),
        ConstVal::Float(f) => {
            if f.is_nan() {
                None
            } else {
                Some(crate::dtype::float_to_bigint_trunc(*f))
            }
        }
        ConstVal::Invalid => None,
    }
}

fn num_bin<F>(a: &ConstVal, b: &ConstVal, iop: F, fop: impl Fn(f64, f64) -> f64) -> ConstVal
where
    F: Fn(i128, i128) -> Option<i128>,
{
    if is_float_v(a) || is_float_v(b) {
        return ConstVal::Float(fop(as_f(a), as_f(b)));
    }
    match (py_int(a), py_int(b)) {
        (Some(ConstVal::Int(x)), Some(ConstVal::Int(y))) => match iop(x, y) {
            Some(v) => ConstVal::Int(v),
            None => ConstVal::Float(f64::NAN), // overflow outside contract
        },
        (Some(x), Some(y)) => big_bin(&x, &y, iop, fop).unwrap_or(ConstVal::Float(f64::NAN)),
        _ => ConstVal::Float(f64::NAN),
    }
}

fn big_bin(a: &ConstVal, b: &ConstVal, iop: impl Fn(i128, i128) -> Option<i128>, fop: impl Fn(f64, f64) -> f64) -> Option<ConstVal> {
    // try to go through i128 when both fit
    match (const_to_i128_opt(a), const_to_i128_opt(b)) {
        (Some(x), Some(y)) => iop(x, y).map(ConstVal::Int),
        _ => {
            // fall back to floats (matches Python's int->float promotion only when
            // both fit in f64 exactly enough; documented divergence for huge ints)
            let _ = (a, b, fop);
            None
        }
    }
}

fn const_to_i128_opt(v: &ConstVal) -> Option<i128> {
    match v {
        ConstVal::Int(i) => Some(*i),
        ConstVal::Bool(b) => Some(if *b { 1 } else { 0 }),
        ConstVal::Big(b) => b.as_i64().map(|i| i as i128),
        _ => None,
    }
}

fn safe_exp2(x: f64) -> f64 {
    let r = x.exp2();
    if r.is_infinite() && x.is_finite() {
        f64::INFINITY
    } else {
        r
    }
}

fn safe_pow(x: f64, y: f64) -> f64 {
    // math.pow semantics with Python's safe_pow fallbacks
    if x == 0.0 && y < 0.0 {
        return f64::INFINITY; // ZeroDivisionError
    }
    if x < 0.0 && !y.is_infinite() && y.fract() != 0.0 {
        return f64::NAN; // complex result
    }
    if y.is_infinite() {
        let ay = x.abs();
        if ay == 1.0 {
            return f64::NAN;
        }
        return if (ay > 1.0) == (y > 0.0) { f64::INFINITY } else { 0.0 };
    }
    x.powf(y)
}

/// Python `math.trunc`.
fn math_trunc(x: f64) -> f64 {
    x.trunc()
}

/// One `python_alu[op]` application on scalar values.
pub fn python_alu(op: Ops, operands: &[ConstVal]) -> Result<ConstVal, String> {
    if operands.is_empty() {
        return Err("python_alu: no operands".into());
    }
    Ok(match op {
        Ops::Log2 => {
            let x = as_f(&operands[0]);
            if x > 0.0 {
                ConstVal::Float(x.log2())
            } else if x == 0.0 {
                ConstVal::Float(f64::NEG_INFINITY)
            } else {
                ConstVal::Float(f64::NAN)
            }
        }
        Ops::Exp2 => ConstVal::Float(safe_exp2(as_f(&operands[0]))),
        Ops::Sqrt => {
            let x = as_f(&operands[0]);
            if x >= 0.0 {
                ConstVal::Float(x.sqrt())
            } else {
                ConstVal::Float(f64::NAN)
            }
        }
        Ops::Reciprocal => {
            let x = as_f(&operands[0]);
            if x == 0.0 {
                ConstVal::Float(if x.is_sign_negative() { f64::NEG_INFINITY } else { f64::INFINITY })
            } else {
                ConstVal::Float(1.0 / x)
            }
        }
        Ops::Sin => {
            let x = as_f(&operands[0]);
            if x.is_infinite() {
                ConstVal::Float(f64::NAN)
            } else {
                ConstVal::Float(x.sin())
            }
        }
        Ops::Trunc => {
            let x = &operands[0];
            if is_float_v(x) {
                let f = as_f(x);
                if f.is_finite() {
                    ConstVal::Float(math_trunc(f))
                } else {
                    x.clone()
                }
            } else {
                x.clone()
            }
        }
        Ops::Neg => match operands[0] {
            ConstVal::Bool(b) => ConstVal::Int(if b { -1 } else { 0 }),
            ConstVal::Int(i) => ConstVal::Int(-i),
            ConstVal::Big(ref b) => ConstVal::Big(b.negate()),
            ConstVal::Float(f) => ConstVal::Float(-f),
            ConstVal::Invalid => ConstVal::Invalid,
        },
        Ops::Add => num_bin(&operands[0], &operands[1], |x, y| x.checked_add(y), |a, b| a + b),
        Ops::Sub => num_bin(&operands[0], &operands[1], |x, y| x.checked_sub(y), |a, b| a - b),
        Ops::Mul => num_bin(&operands[0], &operands[1], |x, y| x.checked_mul(y), |a, b| a * b),
        Ops::Cmpne => {
            let a = &operands[0];
            let b = &operands[1];
            if is_float_v(a) || is_float_v(b) {
                ConstVal::Bool(as_f(a) != as_f(b))
            } else {
                ConstVal::Bool(a.py_cmp(b) != Some(std::cmp::Ordering::Equal))
            }
        }
        Ops::Cmplt => {
            let a = &operands[0];
            let b = &operands[1];
            if is_float_v(a) || is_float_v(b) {
                ConstVal::Bool(as_f(a) < as_f(b))
            } else {
                ConstVal::Bool(a.py_cmp(b) == Some(std::cmp::Ordering::Less))
            }
        }
        Ops::Cmpeq => {
            let a = &operands[0];
            let b = &operands[1];
            if is_float_v(a) || is_float_v(b) {
                ConstVal::Bool(as_f(a) == as_f(b))
            } else {
                ConstVal::Bool(a.py_cmp(b) == Some(std::cmp::Ordering::Equal))
            }
        }
        Ops::Xor | Ops::Or | Ops::And => {
            let (a, b) = (&operands[0], &operands[1]);
            if let (ConstVal::Bool(x), ConstVal::Bool(y)) = (a, b) {
                ConstVal::Bool(match op {
                    Ops::Xor => x ^ y,
                    Ops::Or => x | y,
                    _ => x & y,
                })
            } else {
                let x = const_to_i128_opt(a);
                let y = const_to_i128_opt(b);
                match (x, y) {
                    (Some(x), Some(y)) => ConstVal::Int(match op {
                        Ops::Xor => x ^ y,
                        Ops::Or => x | y,
                        _ => x & y,
                    }),
                    _ => return Err(format!("python_alu {op:?} on big ints")),
                }
            }
        }
        Ops::Shr => {
            let (x, y) = (const_to_i128_opt(&operands[0]), const_to_i128_opt(&operands[1]));
            match (x, y) {
                (Some(x), Some(y)) if (0..512).contains(&y) => ConstVal::Int(x >> y),
                (Some(x), Some(y)) if y >= 512 => ConstVal::Int(if x < 0 { -1 } else { 0 }),
                _ => return Err("python_alu SHR".into()),
            }
        }
        Ops::Shl => {
            let (x, y) = (const_to_i128_opt(&operands[0]), const_to_i128_opt(&operands[1]));
            match (x, y) {
                (Some(x), Some(y)) if (0..512).contains(&y) => match x.checked_shl(y.min(126) as u32) {
                    Some(v) if y <= 126 => ConstVal::Int(v),
                    _ => {
                        // saturate like huge Python ints would (documented)
                        ConstVal::Int(if x < 0 { i128::MIN } else { i128::MAX })
                    }
                },
                (Some(_), Some(y)) if y >= 512 => {
                    return Err("python_alu SHL overflow".into());
                }
                _ => return Err("python_alu SHL".into()),
            }
        }
        Ops::Max => {
            let (a, b) = (&operands[0], &operands[1]);
            if is_float_v(a) || is_float_v(b) {
                let (fa, fb) = (as_f(a), as_f(b));
                // Python max: returns the first argument unless the second is strictly greater
                if fb > fa {
                    ConstVal::Float(fb)
                } else {
                    ConstVal::Float(fa)
                }
            } else {
                match a.py_cmp(b) {
                    Some(std::cmp::Ordering::Less) => b.clone(),
                    _ => a.clone(),
                }
            }
        }
        Ops::Cmod => {
            let (x, y) = (const_to_i128_opt(&operands[0]), const_to_i128_opt(&operands[1]));
            match (x, y) {
                (Some(x), Some(y)) => ConstVal::Int(cmod(x, y)),
                _ => return Err("python_alu CMOD".into()),
            }
        }
        Ops::Cdiv => {
            let (x, y) = (const_to_i128_opt(&operands[0]), const_to_i128_opt(&operands[1]));
            match (x, y) {
                (Some(x), Some(y)) => ConstVal::Int(cdiv(x, y)),
                _ => return Err("python_alu CDIV".into()),
            }
        }
        Ops::Floordiv => {
            let (x, y) = (const_to_i128_opt(&operands[0]), const_to_i128_opt(&operands[1]));
            match (x, y) {
                (Some(x), Some(y)) => ConstVal::Int(floordiv(x, y)),
                _ => return Err("python_alu FLOORDIV".into()),
            }
        }
        Ops::Floormod => {
            let (x, y) = (const_to_i128_opt(&operands[0]), const_to_i128_opt(&operands[1]));
            match (x, y) {
                (Some(x), Some(y)) => ConstVal::Int(floormod(x, y)),
                _ => return Err("python_alu FLOORMOD".into()),
            }
        }
        Ops::Mulacc => {
            let xy = python_alu(Ops::Mul, &operands[0..2])?;
            python_alu(Ops::Add, &[xy, operands[2].clone()])?
        }
        Ops::Where => {
            let cond = match &operands[0] {
                ConstVal::Bool(b) => *b,
                ConstVal::Int(i) => *i != 0,
                ConstVal::Big(b) => !b.is_zero(),
                ConstVal::Float(f) => *f != 0.0,
                ConstVal::Invalid => false,
            };
            if cond {
                operands[1].clone()
            } else {
                operands[2].clone()
            }
        }
        Ops::Pow => {
            // safe_pow: nan for complex, inf for ZeroDivision, sign-aware ValueError
            let (x, y) = (&operands[0], &operands[1]);
            if is_float_v(x) || is_float_v(y) {
                return Ok(ConstVal::Float(safe_pow(as_f(x), as_f(y))));
            }
            match (const_to_i128_opt(x), const_to_i128_opt(y)) {
                (Some(xi), Some(yi)) => {
                    if yi < 0 {
                        // int ** negative int → float in Python
                        return Ok(ConstVal::Float(safe_pow(xi as f64, yi as f64)));
                    }
                    // exact integer pow with checked overflow → f64 fallback
                    let mut acc: i128 = 1;
                    let mut base = xi;
                    let mut e = yi;
                    let mut ok = true;
                    while e > 0 {
                        if e & 1 == 1 {
                            acc = match acc.checked_mul(base) {
                                Some(v) => v,
                                None => {
                                    ok = false;
                                    break;
                                }
                            };
                        }
                        e >>= 1;
                        if e > 0 {
                            base = match base.checked_mul(base) {
                                Some(v) => v,
                                None => {
                                    ok = false;
                                    break;
                                }
                            };
                        }
                    }
                    if ok {
                        ConstVal::Int(acc)
                    } else {
                        ConstVal::Float(safe_pow(xi as f64, yi as f64))
                    }
                }
                _ => return Err("no python_alu for Pow".into()),
            }
        }
        _ => return Err(format!("no python_alu for {op:?}")),
    })
}

/// `exec_alu` (scalar operands; vector/tuple expansion handled by callers over STACK consts).
/// `truncate_output=True` applies `truncate[dtype]` like Python.
pub fn exec_alu(op: Ops, dtype: DTypeId, operands: &[ConstVal], truncate_output: bool) -> Result<ConstVal, String> {
    if operands.iter().any(|v| matches!(v, ConstVal::Invalid)) && crate::uop::ops::group::is_binary(op) {
        return Ok(ConstVal::Invalid);
    }
    let alu = python_alu(op, operands)?;
    if truncate_output && !matches!(alu, ConstVal::Invalid) {
        Ok(crate::dtype::truncate_val(dtype, alu))
    } else {
        Ok(alu)
    }
}
