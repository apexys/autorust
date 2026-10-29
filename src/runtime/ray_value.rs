//! dtype-native value storage + primitive ops for the `ops_rayon` typed
//! evaluator (R2, `CURRENT_STATE.md` §9.8).
//!
//! The interpreter (`runtime/interp.rs`) carries every lane as `f64`, which
//! doubles/quadruples memory traffic, forces a byte↔f64 conversion on every
//! load/store, routes integer ops through `i64`/`f64`, and cannot represent
//! `uint64` exactly. This module stores register values as **exact bytes at the
//! node dtype** in an 8-byte-aligned buffer and reinterprets whole slices with
//! `bytemuck` (no per-element conversion, no unsafe `from_raw_parts`). ALU is
//! monomorphized per dtype so the inner loops autovectorize.
//!
//! Supported dtypes: bool, (u/i)8/16/32/64, f32, f64. Sub-word floats
//! (f16/bf16/fp8) are emulated elsewhere and are handled by the `f64` fallback
//! executor; `is_native_dtype` gates that.

use bytemuck::{Pod, cast_slice, cast_slice_mut};

use crate::dtype::{self, DTypeId};
use crate::uop::ops::Ops;

/// An 8-byte-aligned, heap-backed byte buffer. Values are always `count *
/// itemsize` bytes; the backing `Vec<u64>` keeps the base pointer aligned to 8,
/// so `bytemuck::cast_slice` to any native scalar (alignment ≤ 8) is sound.
#[derive(Clone, Default, Debug)]
pub struct Bytes {
    words: Vec<u64>,
    nbytes: usize,
}

impl Bytes {
    /// Allocate a zeroed byte buffer of `nbytes` bytes.
    pub fn zeroed(nbytes: usize) -> Bytes {
        Bytes {
            words: vec![0u64; nbytes.div_ceil(8)],
            nbytes,
        }
    }

    /// Resize to `nbytes`, zero-extending (reuses capacity).
    pub fn reset(&mut self, nbytes: usize) {
        self.words.resize(nbytes.div_ceil(8), 0);
        self.nbytes = nbytes;
    }

    /// Return the buffer length in bytes.
    pub fn nbytes(&self) -> usize {
        self.nbytes
    }

    /// Return whether the buffer holds no bytes.
    pub fn is_empty(&self) -> bool {
        self.nbytes == 0
    }

    /// View the buffer as a byte slice of exactly `nbytes` bytes.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &cast_slice::<u64, u8>(&self.words)[..self.nbytes]
    }

    /// View the buffer as a mutable byte slice of exactly `nbytes` bytes.
    #[inline]
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        let n = self.nbytes;
        &mut cast_slice_mut::<u64, u8>(&mut self.words)[..n]
    }

    /// Reinterpret as a native slice. Panics if `nbytes` isn't a multiple of
    /// `size_of::<T>()` (callers size the buffer from the dtype).
    #[inline]
    pub fn as_slice<T: Pod>(&self) -> &[T] {
        cast_slice(self.as_bytes())
    }

    /// Reinterpret as a mutable native slice of `T`, with the same size rules as [`Bytes::as_slice`].
    #[inline]
    pub fn as_slice_mut<T: Pod>(&mut self) -> &mut [T] {
        cast_slice_mut(self.as_bytes_mut())
    }

    /// Overwrite with `src` (same byte length expected).
    pub fn copy_from(&mut self, src: &[u8]) {
        debug_assert_eq!(src.len(), self.nbytes);
        self.as_bytes_mut().copy_from_slice(src);
    }
}

/// Native scalar types the typed evaluator computes in.
pub trait Prim: Pod + Copy + PartialEq + std::fmt::Debug + Send + Sync {
    /// Whether the scalar is a floating-point type.
    const IS_FLOAT: bool;
    /// Whether the scalar is an unsigned integer type.
    const IS_UNSIGNED: bool;
    /// Width of the scalar in bits.
    const BITS: u32;
    /// Cast an `f64` to this scalar.
    fn from_f64(x: f64) -> Self;
    /// Cast this scalar to `f64`.
    fn to_f64(self) -> f64;
    /// Cast an `i128` to this scalar, truncating to the target width.
    fn from_i128(x: i128) -> Self;
    /// Cast this scalar to `i128`.
    fn to_i128(self) -> i128;
    /// Whether this scalar equals zero.
    fn is_zero(self) -> bool;

    /// Wrapping add (`ADD` for integers, `+` for floats).
    fn add(self, o: Self) -> Self;
    /// Wrapping subtract (`SUB` for integers, `-` for floats).
    fn sub(self, o: Self) -> Self;
    /// Wrapping multiply (`MUL` for integers, `*` for floats).
    fn mul(self, o: Self) -> Self;
    /// Wrapping negate (`NEG` for integers, unary `-` for floats).
    fn neg(self) -> Self;
    /// Float divide (`FDIV`); implemented only for float dtypes.
    fn fdiv(self, o: Self) -> Self;
    /// Elementwise maximum, matching tinygrad's `maximum`.
    fn maxv(self, o: Self) -> Self;
    /// Bitwise and (`AND`); unimplemented for float dtypes.
    fn bitand(self, o: Self) -> Self;
    /// Bitwise or (`OR`); unimplemented for float dtypes.
    fn bitor(self, o: Self) -> Self;
    /// Bitwise xor (`XOR`); unimplemented for float dtypes.
    fn bitxor(self, o: Self) -> Self;
    /// Shift left (`SHL`), yielding 0 for out-of-range shift counts.
    fn shl(self, o: Self) -> Self;
    /// Shift right (`SHR`), sign-extending negatives and saturating out-of-range counts.
    fn shr(self, o: Self) -> Self;
    /// Less-than comparison (`CMPLT`).
    fn lt(self, o: Self) -> bool;
    /// Equality comparison (`CMPEQ`).
    fn eqv(self, o: Self) -> bool;

    /// `exp2` transcendental (`2**x`); unimplemented for integers.
    fn exp2(self) -> Self;
    /// `log2` transcendental with tinygrad's zero/negative edge cases.
    fn log2v(self) -> Self;
    /// `sqrt` transcendental returning NaN for negative inputs.
    fn sqrtv(self) -> Self;
    /// Reciprocal (`1/x`) with a signed infinity at zero.
    fn recip(self) -> Self;
    /// `sin` transcendental returning NaN for infinities.
    fn sinv(self) -> Self;
    /// `pow` transcendental with tinygrad's zero/infinity edge cases.
    fn powv(self, o: Self) -> Self;
    /// Truncate toward zero; non-finite values pass through.
    fn truncv(self) -> Self;
}

macro_rules! impl_int_prim {
    ($t:ty, $bits:expr, $unsigned:expr) => {
        impl Prim for $t {
            const IS_FLOAT: bool = false;
            const IS_UNSIGNED: bool = $unsigned;
            const BITS: u32 = $bits;
            #[inline]
            fn from_f64(x: f64) -> Self {
                x as $t
            }
            #[inline]
            fn to_f64(self) -> f64 {
                self as f64
            }
            #[inline]
            fn from_i128(x: i128) -> Self {
                x as $t
            }
            #[inline]
            fn to_i128(self) -> i128 {
                self as i128
            }
            #[inline]
            fn is_zero(self) -> bool {
                self == 0
            }
            #[inline]
            fn add(self, o: Self) -> Self {
                self.wrapping_add(o)
            }
            #[inline]
            fn sub(self, o: Self) -> Self {
                self.wrapping_sub(o)
            }
            #[inline]
            fn mul(self, o: Self) -> Self {
                self.wrapping_mul(o)
            }
            #[inline]
            fn neg(self) -> Self {
                self.wrapping_neg()
            }
            fn fdiv(self, _o: Self) -> Self {
                unimplemented!("fdiv on integer dtype")
            }
            #[inline]
            fn maxv(self, o: Self) -> Self {
                self.max(o)
            }
            #[inline]
            fn bitand(self, o: Self) -> Self {
                self & o
            }
            #[inline]
            fn bitor(self, o: Self) -> Self {
                self | o
            }
            #[inline]
            fn bitxor(self, o: Self) -> Self {
                self ^ o
            }
            #[inline]
            fn shl(self, o: Self) -> Self {
                let s = o.to_i128();
                if s < 0 || s >= Self::BITS as i128 {
                    0
                } else {
                    self.wrapping_shl(s as u32)
                }
            }
            #[inline]
            #[allow(unused_comparisons)]
            fn shr(self, o: Self) -> Self {
                let s = o.to_i128();
                if s < 0 || s >= Self::BITS as i128 {
                    if self < 0 { !0 as $t } else { 0 }
                } else {
                    self.wrapping_shr(s as u32)
                }
            }
            #[inline]
            fn lt(self, o: Self) -> bool {
                self < o
            }
            #[inline]
            fn eqv(self, o: Self) -> bool {
                self == o
            }
            fn exp2(self) -> Self {
                unimplemented!("exp2 on integer dtype")
            }
            fn log2v(self) -> Self {
                unimplemented!("log2 on integer dtype")
            }
            fn sqrtv(self) -> Self {
                unimplemented!("sqrt on integer dtype")
            }
            fn recip(self) -> Self {
                unimplemented!("reciprocal on integer dtype")
            }
            fn sinv(self) -> Self {
                unimplemented!("sin on integer dtype")
            }
            fn powv(self, _o: Self) -> Self {
                unimplemented!("pow on integer dtype")
            }
            fn truncv(self) -> Self {
                self
            }
        }
    };
}

macro_rules! impl_float_prim {
    ($t:ty, $bits:expr) => {
        impl Prim for $t {
            const IS_FLOAT: bool = true;
            const IS_UNSIGNED: bool = false;
            const BITS: u32 = $bits;
            #[inline]
            fn from_f64(x: f64) -> Self {
                x as $t
            }
            #[inline]
            fn to_f64(self) -> f64 {
                self as f64
            }
            #[inline]
            fn from_i128(x: i128) -> Self {
                x as $t
            }
            #[inline]
            fn to_i128(self) -> i128 {
                self as i128
            }
            #[inline]
            fn is_zero(self) -> bool {
                self == 0.0
            }
            #[inline]
            fn add(self, o: Self) -> Self {
                self + o
            }
            #[inline]
            fn sub(self, o: Self) -> Self {
                self - o
            }
            #[inline]
            fn mul(self, o: Self) -> Self {
                self * o
            }
            #[inline]
            fn neg(self) -> Self {
                -self
            }
            #[inline]
            fn fdiv(self, o: Self) -> Self {
                self / o
            }
            #[inline]
            fn maxv(self, o: Self) -> Self {
                self.max(o)
            }
            fn bitand(self, _o: Self) -> Self {
                unimplemented!("bitand on float dtype")
            }
            fn bitor(self, _o: Self) -> Self {
                unimplemented!("bitor on float dtype")
            }
            fn bitxor(self, _o: Self) -> Self {
                unimplemented!("bitxor on float dtype")
            }
            fn shl(self, _o: Self) -> Self {
                unimplemented!("shl on float dtype")
            }
            fn shr(self, _o: Self) -> Self {
                unimplemented!("shr on float dtype")
            }
            #[inline]
            fn lt(self, o: Self) -> bool {
                self < o
            }
            #[inline]
            fn eqv(self, o: Self) -> bool {
                self == o
            }
            #[inline]
            fn exp2(self) -> Self {
                let r = (2.0 as $t).powf(self);
                if r.is_infinite() { <$t>::INFINITY } else { r }
            }
            #[inline]
            fn log2v(self) -> Self {
                if self > 0.0 {
                    self.log2()
                } else if self == 0.0 {
                    <$t>::NEG_INFINITY
                } else {
                    <$t>::NAN
                }
            }
            #[inline]
            fn sqrtv(self) -> Self {
                if self >= 0.0 { self.sqrt() } else { <$t>::NAN }
            }
            #[inline]
            fn recip(self) -> Self {
                if self != 0.0 {
                    1.0 / self
                } else {
                    <$t>::INFINITY.copysign(self)
                }
            }
            #[inline]
            fn sinv(self) -> Self {
                if self.is_infinite() {
                    <$t>::NAN
                } else {
                    self.sin()
                }
            }
            #[inline]
            fn powv(self, o: Self) -> Self {
                let p = self.powf(o);
                if p.is_nan() && !(self.is_nan() || o.is_nan()) {
                    if self == 0.0 {
                        <$t>::INFINITY
                    } else {
                        <$t>::NAN
                    }
                } else if p.is_infinite() && self == 0.0 && o < 0.0 {
                    <$t>::INFINITY
                } else {
                    p
                }
            }
            #[inline]
            fn truncv(self) -> Self {
                if self.is_finite() { self.trunc() } else { self }
            }
        }
    };
}

impl_int_prim!(i8, 8, false);
impl_int_prim!(i16, 16, false);
impl_int_prim!(i32, 32, false);
impl_int_prim!(i64, 64, false);
impl_int_prim!(u8, 8, true);
impl_int_prim!(u16, 16, true);
impl_int_prim!(u32, 32, true);
impl_int_prim!(u64, 64, true);
impl_float_prim!(f32, 32);
impl_float_prim!(f64, 64);

/// Whether a dtype is handled natively by the typed evaluator. `bool` shares
/// `u8` storage; sub-word floats are emulated and handled by the f64 fallback.
pub fn is_native_dtype(dt: DTypeId) -> bool {
    dt == dtype::BOOL
        || dt == dtype::INT8
        || dt == dtype::INT16
        || dt == dtype::INT32
        || dt == dtype::INT64
        || dt == dtype::UINT8
        || dt == dtype::UINT16
        || dt == dtype::UINT32
        || dt == dtype::UINT64
        || dt == dtype::FLOAT32
        || dt == dtype::FLOAT64
}

/// Concrete storage dtype for the typed executor. Weak dtypes (`weakint`/
/// `weakfloat`, which survive into the linear list as uncommitted constants)
/// are committed to the engine's default `int`/`float`, matching
/// `dtype::default_int`/`default_float` and the concrete operand types their
/// consumers use.
pub fn storage_dtype(dt: DTypeId) -> DTypeId {
    if dt == dtype::WEAKINT {
        dtype::default_int()
    } else if dt == dtype::WEAKFLOAT {
        dtype::default_float()
    } else {
        dt
    }
}

/// Dispatch a generic `fn<T>` over the native scalar type for `dt`; `None` when
/// `dt` is unsupported. The body is monomorphized per type.
macro_rules! dispatch {
    ($dt:expr, $f:ident $(, $arg:expr)*) => {{
        if $dt == dtype::FLOAT32 {
            Some($f::<f32>($($arg),*))
        } else if $dt == dtype::FLOAT64 {
            Some($f::<f64>($($arg),*))
        } else if $dt == dtype::INT8 {
            Some($f::<i8>($($arg),*))
        } else if $dt == dtype::INT16 {
            Some($f::<i16>($($arg),*))
        } else if $dt == dtype::INT32 {
            Some($f::<i32>($($arg),*))
        } else if $dt == dtype::INT64 {
            Some($f::<i64>($($arg),*))
        } else if $dt == dtype::UINT8 || $dt == dtype::BOOL {
            Some($f::<u8>($($arg),*))
        } else if $dt == dtype::UINT16 {
            Some($f::<u16>($($arg),*))
        } else if $dt == dtype::UINT32 {
            Some($f::<u32>($($arg),*))
        } else if $dt == dtype::UINT64 {
            Some($f::<u64>($($arg),*))
        } else {
            None
        }
    }};
}

/// cdiv (helpers.py:71): abs(x)//abs(y)*sign, 0 on y==0.
#[inline]
fn cdiv<T: Prim>(a: T, b: T) -> T {
    if b.is_zero() {
        return T::from_i128(0);
    }
    if T::IS_FLOAT {
        return T::from_f64((a.to_f64() / b.to_f64()).trunc());
    }
    let (ai, bi) = (a.to_i128(), b.to_i128());
    let q = (ai.abs() / bi.abs()) * if (ai ^ bi) < 0 { -1 } else { 1 };
    T::from_i128(q)
}

#[inline]
fn cmod<T: Prim>(a: T, b: T) -> T {
    let d = cdiv(a, b);
    a.sub(d.mul(b))
}

#[inline]
fn floordiv<T: Prim>(a: T, b: T) -> T {
    if b.is_zero() {
        return T::from_i128(0);
    }
    if T::IS_FLOAT {
        return T::from_f64((a.to_f64() / b.to_f64()).floor());
    }
    T::from_i128(a.to_i128().div_euclid(b.to_i128()))
}

#[inline]
fn floormod<T: Prim>(a: T, b: T) -> T {
    let d = floordiv(a, b);
    a.sub(d.mul(b))
}

fn alu_binary_impl<T: Prim>(op: Ops, out: &mut Bytes, a: &Bytes, b: &Bytes) {
    let a = a.as_slice::<T>();
    let b = b.as_slice::<T>();
    match op {
        Ops::Cmplt | Ops::Cmpeq | Ops::Cmpne => {
            let o = out.as_slice_mut::<u8>();
            for k in 0..o.len() {
                o[k] = match op {
                    Ops::Cmplt => a[k].lt(b[k]),
                    Ops::Cmpeq => a[k].eqv(b[k]),
                    Ops::Cmpne => !a[k].eqv(b[k]),
                    _ => unreachable!(),
                } as u8;
            }
        }
        _ => {
            let o = out.as_slice_mut::<T>();
            for k in 0..o.len() {
                o[k] = match op {
                    Ops::Add => a[k].add(b[k]),
                    Ops::Sub => a[k].sub(b[k]),
                    Ops::Mul => a[k].mul(b[k]),
                    Ops::Fdiv => a[k].fdiv(b[k]),
                    Ops::Max => a[k].maxv(b[k]),
                    Ops::And => a[k].bitand(b[k]),
                    Ops::Or => a[k].bitor(b[k]),
                    Ops::Xor => a[k].bitxor(b[k]),
                    Ops::Shl => a[k].shl(b[k]),
                    Ops::Shr => a[k].shr(b[k]),
                    Ops::Cdiv => cdiv(a[k], b[k]),
                    Ops::Cmod => cmod(a[k], b[k]),
                    Ops::Floordiv => floordiv(a[k], b[k]),
                    Ops::Floormod => floormod(a[k], b[k]),
                    Ops::Neg => a[k].neg(),
                    Ops::Exp2 => a[k].exp2(),
                    Ops::Log2 => a[k].log2v(),
                    Ops::Sqrt => a[k].sqrtv(),
                    Ops::Reciprocal => a[k].recip(),
                    Ops::Sin => a[k].sinv(),
                    Ops::Pow => a[k].powv(b[k]),
                    Ops::Trunc => a[k].truncv(),
                    other => panic!("TGERR: typed alu_binary {other:?}"),
                };
            }
        }
    }
}

/// Binary ALU into `out`. Comparisons write `u8` 0/1 (`bool`); other ops compute
/// in `dt`. Returns `false` for unsupported dtypes. Ternary ops (`WHERE`/
/// `MULACC`) are handled by the executor via [`alu_where`]/[`alu_mulacc`].
pub fn alu_binary(op: Ops, dt: DTypeId, out: &mut Bytes, a: &Bytes, b: &Bytes) -> bool {
    debug_assert!(a.nbytes() == b.nbytes());
    match dispatch!(dt, alu_binary_impl, op, out, a, b) {
        Some(()) => true,
        None => false,
    }
}

/// Unary ALU (`exec_alu(op, x, 0)`); only the unary ops that ignore `b`.
pub fn alu_unary(op: Ops, dt: DTypeId, out: &mut Bytes, a: &Bytes) -> bool {
    let zero = Bytes::zeroed(a.nbytes());
    match op {
        Ops::Neg | Ops::Exp2 | Ops::Log2 | Ops::Sqrt | Ops::Reciprocal | Ops::Sin | Ops::Trunc => {
            match dispatch!(dt, alu_binary_impl, op, out, a, &zero) {
                Some(()) => true,
                None => false,
            }
        }
        other => panic!("TGERR: typed alu_unary {other:?}"),
    }
}

fn alu_where_impl<T: Prim>(out: &mut Bytes, cond: &Bytes, a: &Bytes, b: &Bytes) {
    let c = cond.as_bytes();
    let a = a.as_slice::<T>();
    let b = b.as_slice::<T>();
    let o = out.as_slice_mut::<T>();
    for k in 0..o.len() {
        o[k] = if c[k] != 0 { a[k] } else { b[k] };
    }
}

/// `WHERE(cond, a, b)` — cond is `bool` (u8), a/b/out share dtype.
pub fn alu_where(dt: DTypeId, out: &mut Bytes, cond: &Bytes, a: &Bytes, b: &Bytes) -> bool {
    match dispatch!(dt, alu_where_impl, out, cond, a, b) {
        Some(()) => true,
        None => false,
    }
}

fn alu_mulacc_impl<T: Prim>(out: &mut Bytes, a: &Bytes, b: &Bytes, c: &Bytes) {
    let a = a.as_slice::<T>();
    let b = b.as_slice::<T>();
    let c = c.as_slice::<T>();
    let o = out.as_slice_mut::<T>();
    for k in 0..o.len() {
        o[k] = a[k].mul(b[k]).add(c[k]);
    }
}

/// `MULACC(a, b, c)` = a*b + c, in `dt`.
pub fn alu_mulacc(dt: DTypeId, out: &mut Bytes, a: &Bytes, b: &Bytes, c: &Bytes) -> bool {
    match dispatch!(dt, alu_mulacc_impl, out, a, b, c) {
        Some(()) => true,
        None => false,
    }
}

fn per_src<D: Prim, S: Prim>(out: &mut Bytes, a: &Bytes) {
    let av = a.as_slice::<S>();
    let o = out.as_slice_mut::<D>();
    for k in 0..o.len() {
        if D::IS_FLOAT {
            o[k] = D::from_f64(av[k].to_f64());
        } else if S::IS_FLOAT {
            // float -> int mirrors tinygrad `truncate`: clamp negatives for an
            // unsigned destination, truncate toward zero, then wrap to width.
            let v = av[k].to_f64();
            let v = if D::IS_UNSIGNED && v < 0.0 { 0.0 } else { v };
            o[k] = D::from_i128(v as i128);
        } else {
            o[k] = D::from_i128(av[k].to_i128());
        }
    }
}

fn cast_from_src<D: Prim>(src: DTypeId, out: &mut Bytes, a: &Bytes) -> bool {
    if src == dtype::FLOAT32 {
        per_src::<D, f32>(out, a);
    } else if src == dtype::FLOAT64 {
        per_src::<D, f64>(out, a);
    } else if src == dtype::INT8 {
        per_src::<D, i8>(out, a);
    } else if src == dtype::INT16 {
        per_src::<D, i16>(out, a);
    } else if src == dtype::INT32 {
        per_src::<D, i32>(out, a);
    } else if src == dtype::INT64 {
        per_src::<D, i64>(out, a);
    } else if src == dtype::UINT8 || src == dtype::BOOL {
        per_src::<D, u8>(out, a);
    } else if src == dtype::UINT16 {
        per_src::<D, u16>(out, a);
    } else if src == dtype::UINT32 {
        per_src::<D, u32>(out, a);
    } else if src == dtype::UINT64 {
        per_src::<D, u64>(out, a);
    } else {
        return false;
    }
    true
}

/// `CAST` from `src` dtype to `dst` dtype. Int↔int is exact (via `i128`);
/// anything involving a float goes through `f64` (same precision the f64
/// interpreter has). Returns `false` for unsupported dtype pairs.
pub fn cast(src: DTypeId, dst: DTypeId, out: &mut Bytes, a: &Bytes) -> bool {
    if !is_native_dtype(src) || !is_native_dtype(dst) {
        return false;
    }
    matches!(dispatch!(dst, cast_from_src, src, out, a), Some(true))
}

// ---- scalar/index access (control flow: RANGE/SPECIAL/IF, const fill) ----

fn get_i128_impl<T: Prim>(b: &Bytes, i: usize) -> i128 {
    b.as_slice::<T>()[i].to_i128()
}
fn set_i128_impl<T: Prim>(b: &mut Bytes, i: usize, v: i128) {
    b.as_slice_mut::<T>()[i] = T::from_i128(v);
}

/// Element `i` as `i128` (index/loop values). The 32-bit signed case (the
/// dominant index dtype) is fast-pathed.
#[inline]
pub fn get_i128(dt: DTypeId, b: &Bytes, i: usize) -> i128 {
    if dt == dtype::INT32 {
        return b.as_slice::<i32>()[i] as i128;
    }
    dispatch!(dt, get_i128_impl, b, i).expect("TGERR: typed get_i128 dtype")
}

/// Write `v` into element `i` truncating to `dt`.
pub fn set_i128(dt: DTypeId, b: &mut Bytes, i: usize, v: i128) {
    dispatch!(dt, set_i128_impl, b, i, v).expect("TGERR: typed set_i128 dtype");
}

fn truthy_impl<T: Prim>(b: &Bytes, i: usize) -> bool {
    !b.as_slice::<T>()[i].is_zero()
}

/// Whether element `i` is non-zero (the `IF`/`END` condition).
pub fn truthy(dt: DTypeId, b: &Bytes, i: usize) -> bool {
    dispatch!(dt, truthy_impl, b, i).expect("TGERR: typed truthy dtype")
}

fn fill_impl<T: Prim>(b: &mut Bytes, v: i128) {
    let o = b.as_slice_mut::<T>();
    let x = T::from_i128(v);
    for e in o.iter_mut() {
        *e = x;
    }
}

/// Fill every element with `v` (CONST/SPECIAL broadcast).
pub fn fill_i128(dt: DTypeId, b: &mut Bytes, v: i128) {
    dispatch!(dt, fill_impl, b, v).expect("TGERR: typed fill dtype");
}

fn add_impl<T: Prim>(b: &mut Bytes, v: i128) {
    let o = b.as_slice_mut::<T>();
    let x = T::from_i128(v);
    for e in o.iter_mut() {
        *e = e.add(x);
    }
}

/// Add `v` to every element in place (RANGE increment).
pub fn add_i128(dt: DTypeId, b: &mut Bytes, v: i128) {
    dispatch!(dt, add_impl, b, v).expect("TGERR: typed add dtype");
}

fn fill_f64_impl<T: Prim>(b: &mut Bytes, v: f64) {
    let o = b.as_slice_mut::<T>();
    let x = T::from_f64(v);
    for e in o.iter_mut() {
        *e = x;
    }
}

/// Fill every element with `v` (float CONST/ALU-param broadcast).
pub fn fill_f64(dt: DTypeId, b: &mut Bytes, v: f64) {
    dispatch!(dt, fill_f64_impl, b, v).expect("TGERR: typed fill_f64 dtype");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32b(v: &[f32]) -> Bytes {
        let mut b = Bytes::zeroed(v.len() * 4);
        b.as_slice_mut::<f32>().copy_from_slice(v);
        b
    }

    #[test]
    fn binary_add_mul() {
        let a = f32b(&[1.0, 2.0, 3.0]);
        let b = f32b(&[10.0, 20.0, 30.0]);
        let mut o = Bytes::zeroed(12);
        assert!(alu_binary(Ops::Add, dtype::FLOAT32, &mut o, &a, &b));
        assert_eq!(o.as_slice::<f32>(), &[11.0, 22.0, 33.0]);
        assert!(alu_binary(Ops::Mul, dtype::FLOAT32, &mut o, &a, &b));
        assert_eq!(o.as_slice::<f32>(), &[10.0, 40.0, 90.0]);
    }

    #[test]
    fn comparison_writes_bool() {
        let a = f32b(&[1.0, 5.0, 3.0]);
        let b = f32b(&[4.0, 2.0, 3.0]);
        let mut o = Bytes::zeroed(3);
        assert!(alu_binary(Ops::Cmplt, dtype::FLOAT32, &mut o, &a, &b));
        assert_eq!(o.as_slice::<u8>(), &[1, 0, 0]);
    }

    #[test]
    fn uint64_exact_add() {
        let mut a = Bytes::zeroed(16);
        a.as_slice_mut::<u64>()
            .copy_from_slice(&[u64::MAX - 1, 1u64 << 60]);
        let mut b = Bytes::zeroed(16);
        b.as_slice_mut::<u64>().copy_from_slice(&[1, 0]);
        let mut o = Bytes::zeroed(16);
        assert!(alu_binary(Ops::Add, dtype::UINT64, &mut o, &a, &b));
        assert_eq!(o.as_slice::<u64>(), &[u64::MAX, 1u64 << 60]);
    }

    #[test]
    fn int_wrapping_matches_width() {
        let mut a = Bytes::zeroed(4);
        a.as_slice_mut::<i32>().copy_from_slice(&[i32::MAX]);
        let mut b = Bytes::zeroed(4);
        b.as_slice_mut::<i32>().copy_from_slice(&[1]);
        let mut o = Bytes::zeroed(4);
        assert!(alu_binary(Ops::Add, dtype::INT32, &mut o, &a, &b));
        assert_eq!(o.as_slice::<i32>(), &[i32::MIN]);
    }

    #[test]
    fn where_and_mulacc() {
        let mut cond = Bytes::zeroed(3);
        cond.as_slice_mut::<u8>().copy_from_slice(&[1, 0, 1]);
        let a = f32b(&[1.0, 2.0, 3.0]);
        let b = f32b(&[9.0, 9.0, 9.0]);
        let mut o = Bytes::zeroed(12);
        assert!(alu_where(dtype::FLOAT32, &mut o, &cond, &a, &b));
        assert_eq!(o.as_slice::<f32>(), &[1.0, 9.0, 3.0]);
        let c = f32b(&[1.0, 1.0, 1.0]);
        assert!(alu_mulacc(dtype::FLOAT32, &mut o, &a, &b, &c));
        assert_eq!(o.as_slice::<f32>(), &[10.0, 19.0, 28.0]);
    }

    #[test]
    fn cast_int_and_float() {
        let mut a = Bytes::zeroed(12);
        a.as_slice_mut::<i32>().copy_from_slice(&[1, -2, 300]);
        let mut o = Bytes::zeroed(12);
        assert!(cast(dtype::INT32, dtype::FLOAT32, &mut o, &a));
        assert_eq!(o.as_slice::<f32>(), &[1.0, -2.0, 300.0]);
        let mut b = Bytes::zeroed(8);
        b.as_slice_mut::<i64>().copy_from_slice(&[1 << 33]);
        let mut o2 = Bytes::zeroed(4);
        assert!(cast(dtype::INT64, dtype::INT32, &mut o2, &b));
        assert_eq!(o2.as_slice::<i32>(), &[0]);
    }

    #[test]
    fn unsupported_dtype_is_rejected() {
        assert!(!is_native_dtype(dtype::FLOAT16));
        assert!(!is_native_dtype(dtype::BFLOAT16));
    }
}
