//! Port of tinygrad `dtype.py` (v0.14.0).
//!
//! Identity: `DType` is interned in Python via `DTypeMetaClass` on
//! `(priority, bitsize, name, fmt)`. Here the closed set lives in a fixed table
//! (`DTYPES`) and dtypes are referenced by `DTypeId` (index). The table order
//! mirrors the declaration order of the `DTypes` class attributes.
//!
//! The *canonical identity* used by the JSON interchange (and by Python `repr`,
//! via `INVERSE_DTYPES_DICT`) is the attribute name — note tinygrad's class
//! aliases win the inverse mapping, so the canonical attrs are
//! `char/uchar/short/ushort/int/uint/long/ulong` and `half/float/double`
//! (NOT int8/int32/float32). `_uint128`/`_uint256` have no inverse mapping and
//! cannot travel through the interchange (they are outside `dtypes.all`).

use crate::helpers::bigint::BigInt;
use crate::helpers::{DEFAULT_FLOAT, DEFAULT_INT, SUM_DTYPE, getenv_str};
use std::collections::HashMap;
use std::sync::LazyLock;
use std::sync::OnceLock;

/// Address spaces (`AddrSpace` IntEnum, auto() starting at 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum AddrSpace {
    Global = 1,
    Local = 2,
    Reg = 3,
    Alu = 4,
}

impl AddrSpace {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(AddrSpace::Global),
            2 => Some(AddrSpace::Local),
            3 => Some(AddrSpace::Reg),
            4 => Some(AddrSpace::Alu),
            _ => None,
        }
    }
    pub fn name(&self) -> &'static str {
        match self {
            AddrSpace::Global => "AddrSpace.GLOBAL",
            AddrSpace::Local => "AddrSpace.LOCAL",
            AddrSpace::Reg => "AddrSpace.REG",
            AddrSpace::Alu => "AddrSpace.ALU",
        }
    }
}

/// A dtype: the four dataclass fields of Python `DType`.
/// `attr` is the Python attribute name (identity for interchange/repr),
/// `name` is the C name (e.g. "signed char"), `fmt` is the struct format char.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DType {
    pub priority: i32,
    pub bitsize: u32,
    pub attr: &'static str,
    pub name: &'static str,
    pub fmt: Option<u8>,
}

/// Interned dtype reference (index into `DTYPES`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DTypeId(pub u16);

// Table order == declaration order of DTypes class attributes.
pub const VOID: DTypeId = DTypeId(0);
pub const WEAKINT: DTypeId = DTypeId(1);
pub const BOOL: DTypeId = DTypeId(2);
pub const INT8: DTypeId = DTypeId(3);
pub const UINT8: DTypeId = DTypeId(4);
pub const INT16: DTypeId = DTypeId(5);
pub const UINT16: DTypeId = DTypeId(6);
pub const INT32: DTypeId = DTypeId(7);
pub const UINT32: DTypeId = DTypeId(8);
pub const INT64: DTypeId = DTypeId(9);
pub const UINT64: DTypeId = DTypeId(10);
pub const UINT128: DTypeId = DTypeId(11);
pub const UINT256: DTypeId = DTypeId(12);
pub const WEAKFLOAT: DTypeId = DTypeId(13);
pub const FP8E4M3: DTypeId = DTypeId(14);
pub const FP8E5M2: DTypeId = DTypeId(15);
pub const FP8E4M3FNUZ: DTypeId = DTypeId(16);
pub const FP8E5M2FNUZ: DTypeId = DTypeId(17);
pub const FLOAT16: DTypeId = DTypeId(18);
pub const BFLOAT16: DTypeId = DTypeId(19);
pub const FLOAT32: DTypeId = DTypeId(20);
pub const FLOAT64: DTypeId = DTypeId(21);

pub const DTYPES: &[DType] = &[
    DType {
        priority: -1,
        bitsize: 0,
        attr: "void",
        name: "void",
        fmt: None,
    },
    DType {
        priority: 0,
        bitsize: 800,
        attr: "weakint",
        name: "weakint",
        fmt: None,
    },
    DType {
        priority: 0,
        bitsize: 1,
        attr: "bool",
        name: "bool",
        fmt: Some(b'?'),
    },
    DType {
        priority: 1,
        bitsize: 8,
        attr: "char",
        name: "signed char",
        fmt: Some(b'b'),
    },
    DType {
        priority: 2,
        bitsize: 8,
        attr: "uchar",
        name: "unsigned char",
        fmt: Some(b'B'),
    },
    DType {
        priority: 3,
        bitsize: 16,
        attr: "short",
        name: "short",
        fmt: Some(b'h'),
    },
    DType {
        priority: 4,
        bitsize: 16,
        attr: "ushort",
        name: "unsigned short",
        fmt: Some(b'H'),
    },
    DType {
        priority: 5,
        bitsize: 32,
        attr: "int",
        name: "int",
        fmt: Some(b'i'),
    },
    DType {
        priority: 6,
        bitsize: 32,
        attr: "uint",
        name: "unsigned int",
        fmt: Some(b'I'),
    },
    DType {
        priority: 7,
        bitsize: 64,
        attr: "long",
        name: "long",
        fmt: Some(b'q'),
    },
    DType {
        priority: 8,
        bitsize: 64,
        attr: "ulong",
        name: "unsigned long",
        fmt: Some(b'Q'),
    },
    DType {
        priority: 8,
        bitsize: 128,
        attr: "_uint128",
        name: "uint128",
        fmt: None,
    },
    DType {
        priority: 8,
        bitsize: 256,
        attr: "_uint256",
        name: "uint256",
        fmt: None,
    },
    DType {
        priority: 9,
        bitsize: 800,
        attr: "weakfloat",
        name: "weakfloat",
        fmt: None,
    },
    DType {
        priority: 10,
        bitsize: 8,
        attr: "fp8e4m3",
        name: "float8_e4m3",
        fmt: None,
    },
    DType {
        priority: 11,
        bitsize: 8,
        attr: "fp8e5m2",
        name: "float8_e5m2",
        fmt: None,
    },
    DType {
        priority: 10,
        bitsize: 8,
        attr: "fp8e4m3fnuz",
        name: "float8_e4m3fnuz",
        fmt: None,
    },
    DType {
        priority: 11,
        bitsize: 8,
        attr: "fp8e5m2fnuz",
        name: "float8_e5m2fnuz",
        fmt: None,
    },
    DType {
        priority: 12,
        bitsize: 16,
        attr: "half",
        name: "half",
        fmt: Some(b'e'),
    },
    DType {
        priority: 13,
        bitsize: 16,
        attr: "bfloat16",
        name: "__bf16",
        fmt: None,
    },
    DType {
        priority: 14,
        bitsize: 32,
        attr: "float",
        name: "float",
        fmt: Some(b'f'),
    },
    DType {
        priority: 15,
        bitsize: 64,
        attr: "double",
        name: "double",
        fmt: Some(b'd'),
    },
];

impl DTypeId {
    pub fn dt(&self) -> &'static DType {
        &DTYPES[self.0 as usize]
    }
    pub fn priority(&self) -> i32 {
        self.dt().priority
    }
    pub fn bitsize(&self) -> u32 {
        self.dt().bitsize
    }
    pub fn attr(&self) -> &'static str {
        self.dt().attr
    }
    pub fn name(&self) -> &'static str {
        self.dt().name
    }
    pub fn fmt(&self) -> Option<u8> {
        self.dt().fmt
    }
    pub fn itemsize(&self) -> usize {
        self.bitsize().div_ceil(8) as usize
    }
}

/// A Rust scalar a `Tensor` can be read back as (`Tensor::to_list`/`item`).
/// `DTYPE` is the tinygrad dtype the tensor is cast to in-graph before its
/// little-endian storage bytes are reinterpreted as `[Self]`, so the read never
/// round-trips through `f64`. `from_bytes_le` reads exactly `DTYPE.itemsize()`
/// bytes. `f16`/`bf16`/`fp8` tensors are read via a graph cast to a wider type.
pub trait ReadDtype: Sized + Clone + PartialEq + core::fmt::Debug {
    const DTYPE: DTypeId;
    fn from_bytes_le(bytes: &[u8]) -> Self;
}

macro_rules! impl_read_dtype {
    ($($t:ty => $dt:expr, $n:expr);* $(;)?) => {$(
        impl ReadDtype for $t {
            const DTYPE: DTypeId = $dt;
            fn from_bytes_le(bytes: &[u8]) -> Self {
                let mut a = [0u8; $n];
                a.copy_from_slice(&bytes[..$n]);
                <$t>::from_le_bytes(a)
            }
        }
    )*};
}
impl_read_dtype!(
    i8 => INT8, 1; i16 => INT16, 2; i32 => INT32, 4; i64 => INT64, 8;
    u8 => UINT8, 1; u16 => UINT16, 2; u32 => UINT32, 4; u64 => UINT64, 8;
    f32 => FLOAT32, 4; f64 => FLOAT64, 8;
);
impl ReadDtype for bool {
    const DTYPE: DTypeId = BOOL;
    fn from_bytes_le(bytes: &[u8]) -> Self {
        bytes[0] != 0
    }
}

pub static BY_ATTR: LazyLock<HashMap<&'static str, DTypeId>> = LazyLock::new(|| {
    DTYPES
        .iter()
        .enumerate()
        .map(|(i, d)| (d.attr, DTypeId(i as u16)))
        .collect()
});

pub fn by_attr(attr: &str) -> Option<DTypeId> {
    BY_ATTR.get(attr).cloned()
}

/// `to_dtype(DTypeLike)` for the str case.
pub fn to_dtype(attr: &str) -> Option<DTypeId> {
    // accept both the python attr ("uchar") and the friendly alias ("uint8")
    by_attr(attr).or_else(|| match attr {
        "uint8" => by_attr("uchar"),
        "uint16" => by_attr("ushort"),
        "uint32" => by_attr("uint"),
        "uint64" => by_attr("ulong"),
        "int8" => by_attr("char"),
        "int16" => by_attr("short"),
        "int32" => by_attr("int"),
        "int64" => by_attr("long"),
        "float32" => by_attr("float"),
        "float64" => by_attr("double"),
        "float16" => by_attr("half"),
        _ => None,
    })
}

// ---- sets (port of DTypes class-level tuples) ----

pub const FP8_OCP: &[DTypeId] = &[FP8E4M3, FP8E5M2];
pub const FP8_FNUZ: &[DTypeId] = &[FP8E4M3FNUZ, FP8E5M2FNUZ];
pub const FP8S: &[DTypeId] = &[FP8E4M3, FP8E5M2, FP8E4M3FNUZ, FP8E5M2FNUZ];
pub const FLOATS: &[DTypeId] = &[
    FP8E4M3,
    FP8E5M2,
    FP8E4M3FNUZ,
    FP8E5M2FNUZ,
    FLOAT16,
    BFLOAT16,
    FLOAT32,
    FLOAT64,
];
pub const UINTS: &[DTypeId] = &[UINT8, UINT16, UINT32, UINT64];
pub const SINTS: &[DTypeId] = &[INT8, INT16, INT32, INT64];
pub const INTS: &[DTypeId] = &[UINT8, UINT16, UINT32, UINT64, INT8, INT16, INT32, INT64];
pub const WEAKS: &[DTypeId] = &[WEAKINT, WEAKFLOAT];

/// `dtypes.all` — floats + ints + bool (note: `_uint128`/`_uint256` are NOT in `all`).
pub const ALL: &[DTypeId] = &[
    FP8E4M3,
    FP8E5M2,
    FP8E4M3FNUZ,
    FP8E5M2FNUZ,
    FLOAT16,
    BFLOAT16,
    FLOAT32,
    FLOAT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    INT8,
    INT16,
    INT32,
    INT64,
    BOOL,
];

pub fn in_set(set: &[DTypeId], d: DTypeId) -> bool {
    set.contains(&d)
}

pub fn is_float(d: DTypeId) -> bool {
    in_set(FLOATS, d) || d == WEAKFLOAT
}
pub fn is_int(d: DTypeId) -> bool {
    in_set(INTS, d) || d == WEAKINT
}
pub fn is_unsigned(d: DTypeId) -> bool {
    in_set(UINTS, d)
}
pub fn is_bool(d: DTypeId) -> bool {
    d == BOOL
}
pub fn is_weak(d: DTypeId) -> bool {
    d == WEAKINT || d == WEAKFLOAT
}

pub fn default_float() -> DTypeId {
    static V: OnceLock<DTypeId> = OnceLock::new();
    *V.get_or_init(|| {
        to_dtype(&getenv_str(DEFAULT_FLOAT.env, DEFAULT_FLOAT.default)).expect("bad DEFAULT_FLOAT")
    })
}
pub fn default_int() -> DTypeId {
    static V: OnceLock<DTypeId> = OnceLock::new();
    *V.get_or_init(|| {
        to_dtype(&getenv_str(DEFAULT_INT.env, DEFAULT_INT.default)).expect("bad DEFAULT_INT")
    })
}
/// `SUM_DTYPE` (the accumulation dtype for float reductions), read once.
pub fn sum_dtype() -> DTypeId {
    static V: OnceLock<DTypeId> = OnceLock::new();
    *V.get_or_init(|| {
        to_dtype(&getenv_str(SUM_DTYPE.env, SUM_DTYPE.default)).expect("bad SUM_DTYPE")
    })
}

/// Python value space for const args / truncation results.
/// `Int` covers int8..int64 and uint8..uint64 ranges (fits i128);
/// `Big` covers weak dtype bounds (2^800 etc.).
/// Float identity follows Python `arg` equality: bit patterns (`-0.0 != 0.0`),
/// with NaN == NaN (canonical NaN only).
#[derive(Clone, Debug)]
pub enum ConstVal {
    Bool(bool),
    Int(i128),
    Big(BigInt),
    Float(f64),
    Invalid,
}

impl PartialEq for ConstVal {
    fn eq(&self, other: &Self) -> bool {
        use ConstVal::*;
        match (self, other) {
            (Bool(a), Bool(b)) => a == b,
            (Int(a), Int(b)) => a == b,
            (Big(a), Big(b)) => a == b,
            (Float(a), Float(b)) => a.to_bits() == b.to_bits(),
            (Invalid, Invalid) => true,
            _ => false,
        }
    }
}
impl Eq for ConstVal {}

impl std::hash::Hash for ConstVal {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        use ConstVal::*;
        std::mem::discriminant(self).hash(state);
        match self {
            Bool(b) => b.hash(state),
            Int(i) => i.hash(state),
            Big(b) => b.hash(state),
            Float(f) => f.to_bits().hash(state),
            Invalid => {}
        }
    }
}

impl ConstVal {
    pub fn as_f64(&self) -> f64 {
        match self {
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

    /// Python ordering (`<`/`==`); `None` if either side is NaN.
    pub fn py_cmp(&self, other: &ConstVal) -> Option<std::cmp::Ordering> {
        if matches!(self, ConstVal::Float(f) if f.is_nan())
            || matches!(other, ConstVal::Float(f) if f.is_nan())
        {
            return None;
        }
        match (self, other) {
            (ConstVal::Int(a), ConstVal::Int(b)) => Some(a.cmp(b)),
            (ConstVal::Big(a), ConstVal::Big(b)) => Some(a.cmp(b)),
            (ConstVal::Big(a), ConstVal::Int(b)) => Some(a.cmp_i128(*b)),
            (ConstVal::Int(a), ConstVal::Big(b)) => Some(b.cmp_i128(*a).reverse()),
            _ => self.as_f64().partial_cmp(&other.as_f64()),
        }
    }

    /// Exact int view for int consts (used by `const_factor`/`divides`/etc.);
    /// Bigs beyond i64 act as magnitude sentinels.
    pub fn as_int_exact(&self) -> Option<i128> {
        match self {
            ConstVal::Int(i) => Some(*i),
            ConstVal::Big(b) => b.as_i64().map(|i| i as i128).or(Some(i128::MAX)),
            _ => None,
        }
    }
}

/// `DType.min`
pub fn dtype_min(d: DTypeId) -> ConstVal {
    if is_int(d) {
        if is_unsigned(d) {
            ConstVal::Int(0)
        } else if d.bitsize() > 127 {
            ConstVal::Big(BigInt::pow2((d.bitsize() - 1) as u64).negate())
        } else {
            ConstVal::Int(-((1i128) << (d.bitsize() - 1)))
        }
    } else if is_float(d) {
        ConstVal::Float(f64::NEG_INFINITY)
    } else {
        ConstVal::Bool(false)
    }
}

/// `DType.max`
pub fn dtype_max(d: DTypeId) -> ConstVal {
    if is_int(d) {
        if is_unsigned(d) {
            if d.bitsize() >= 128 {
                ConstVal::Big(BigInt::pow2_minus_one(d.bitsize() as u64))
            } else {
                ConstVal::Int((1i128 << d.bitsize()) - 1)
            }
        } else if d.bitsize() > 127 {
            ConstVal::Big(BigInt::pow2_minus_one((d.bitsize() - 1) as u64))
        } else {
            ConstVal::Int((1i128 << (d.bitsize() - 1)) - 1)
        }
    } else if is_float(d) {
        ConstVal::Float(f64::INFINITY)
    } else {
        ConstVal::Bool(true)
    }
}

/// `strong_dtype`
pub fn strong_dtype(d: DTypeId) -> DTypeId {
    match d {
        WEAKINT => default_int(),
        WEAKFLOAT => default_float(),
        _ => d,
    }
}

/// `weak_dtype`
pub fn weak_dtype(d: DTypeId) -> DTypeId {
    if is_float(d) {
        WEAKFLOAT
    } else if is_int(d) {
        WEAKINT
    } else {
        d
    }
}

/// `DType.const` — canonicalize + truncate a python const to the dtype.
/// NOTE: for int dtypes the result is the *exact* Python int (no wrapping) —
/// `int(val)`; truncation to the dtype width only happens later (CAST, ops).
pub fn dtype_const(d: DTypeId, val: ConstVal) -> ConstVal {
    match val {
        ConstVal::Invalid => ConstVal::Invalid,
        ConstVal::Float(f) if f.is_nan() => {
            if is_float(d) {
                ConstVal::Float(f64::NAN)
            } else {
                truncate(d, f64::NAN)
            }
        }
        ConstVal::Float(f) => {
            if is_float(d) {
                ConstVal::Float(trunc_float(d, f))
            } else if is_bool(d) {
                ConstVal::Bool(f != 0.0)
            } else {
                // int(val): exact trunc toward zero
                match BigInt::from_f64_trunc(f) {
                    None => ConstVal::Float(f64::NAN), // unreachable: int(nan) raises in Python
                    Some(b) => b
                        .as_i64()
                        .map(|i| ConstVal::Int(i as i128))
                        .unwrap_or(ConstVal::Big(b)),
                }
            }
        }
        ConstVal::Int(i) => {
            if is_float(d) {
                ConstVal::Float(trunc_float(d, i as f64))
            } else if is_bool(d) {
                ConstVal::Bool(i != 0)
            } else {
                ConstVal::Int(i)
            }
        }
        ConstVal::Big(_) => {
            if is_float(d) {
                ConstVal::Float(trunc_float(d, val.as_f64()))
            } else if is_bool(d) {
                ConstVal::Bool(val.as_f64() != 0.0)
            } else {
                val
            }
        }
        ConstVal::Bool(b) => {
            if is_float(d) {
                ConstVal::Float(trunc_float(d, if b { 1.0 } else { 0.0 }))
            } else if is_bool(d) {
                ConstVal::Bool(b)
            } else {
                ConstVal::Int(if b { 1 } else { 0 })
            }
        }
    }
}

// ---- type promotion (jep 9407 lattice) ----

static PROMO_LATTICE: &[(DTypeId, &[DTypeId])] = &[
    (BOOL, &[WEAKINT]),
    (WEAKINT, &[INT8, UINT8]),
    (INT8, &[INT16]),
    (INT16, &[INT32]),
    (INT32, &[INT64]),
    (INT64, &[WEAKFLOAT]),
    (UINT8, &[INT16, UINT16]),
    (UINT16, &[INT32, UINT32]),
    (UINT32, &[INT64, UINT64]),
    (UINT64, &[WEAKFLOAT]),
    (WEAKFLOAT, &[FP8E4M3, FP8E5M2, FP8E4M3FNUZ, FP8E5M2FNUZ]),
    (FP8E4M3, &[FLOAT16, BFLOAT16]),
    (FP8E5M2, &[FLOAT16, BFLOAT16]),
    (FP8E4M3FNUZ, &[FLOAT16, BFLOAT16]),
    (FP8E5M2FNUZ, &[FLOAT16, BFLOAT16]),
    (FLOAT16, &[FLOAT32]),
    (BFLOAT16, &[FLOAT32]),
    (FLOAT32, &[FLOAT64]),
];

fn lattice_parents(d: DTypeId) -> &'static [DTypeId] {
    for (k, v) in PROMO_LATTICE {
        if *k == d {
            return v;
        }
    }
    &[]
}

/// `_get_recursive_parents` — transitive closure (eagerly computed, 22 nodes).
static RECURSIVE_PARENTS: LazyLock<Vec<u32>> = LazyLock::new(|| {
    let n = DTYPES.len();
    let mut bits = vec![0u32; n];
    // iterate to fixpoint (lattice is a DAG; simple relaxation works)
    loop {
        let mut changed = false;
        for i in 0..n {
            let d = DTypeId(i as u16);
            let mut b = if d == FLOAT64 {
                1u32 << FLOAT64.0
            } else {
                1u32 << i
            };
            for &p in lattice_parents(d) {
                b |= bits[p.0 as usize];
            }
            if b != bits[i] {
                bits[i] = b;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    bits
});

fn dtype_order_key(d: DTypeId) -> (i32, u32, &'static str, Option<u8>) {
    let dt = d.dt();
    (dt.priority, dt.bitsize, dt.attr, dt.fmt)
}

/// `least_upper_dtype(*ds)` — `min` over the intersection of recursive parents.
pub fn least_upper_dtype(ds: &[DTypeId]) -> DTypeId {
    assert!(!ds.is_empty(), "least_upper_dtype needs at least one dtype");
    let mut inter = RECURSIVE_PARENTS[ds[0].0 as usize];
    for d in &ds[1..] {
        inter &= RECURSIVE_PARENTS[d.0 as usize];
    }
    assert!(inter != 0, "no common dtype for {ds:?}");
    let mut best: Option<DTypeId> = None;
    for i in 0..DTYPES.len() {
        if inter & (1u32 << i) != 0 {
            let cand = DTypeId(i as u16);
            best = Some(match best {
                None => cand,
                Some(b) => {
                    if dtype_order_key(cand) < dtype_order_key(b) {
                        cand
                    } else {
                        b
                    }
                }
            });
        }
    }
    best.unwrap()
}

pub fn least_upper_dtype2(a: DTypeId, b: DTypeId) -> DTypeId {
    least_upper_dtype(&[a, b])
}

/// `least_upper_float`
pub fn least_upper_float(d: DTypeId) -> DTypeId {
    if d == WEAKINT {
        WEAKFLOAT
    } else if is_float(d) {
        d
    } else {
        least_upper_dtype2(d, default_float())
    }
}

/// `can_lossless_cast(dt0, dt1)` — true if dt1 preserves values of dt0.
pub fn can_lossless_cast(dt0: DTypeId, dt1: DTypeId) -> bool {
    if dt0 == dt1 || dt0 == BOOL {
        return true;
    }
    match dt1 {
        WEAKINT => in_set(INTS, dt0),
        FLOAT64 => in_set(
            &[
                FLOAT32,
                FLOAT16,
                BFLOAT16,
                FP8E4M3,
                FP8E5M2,
                FP8E4M3FNUZ,
                FP8E5M2FNUZ,
                UINT32,
                UINT16,
                UINT8,
                INT32,
                INT16,
                INT8,
            ],
            dt0,
        ),
        FLOAT32 => in_set(
            &[
                FLOAT16,
                BFLOAT16,
                FP8E4M3,
                FP8E5M2,
                FP8E4M3FNUZ,
                FP8E5M2FNUZ,
                UINT16,
                UINT8,
                INT16,
                INT8,
            ],
            dt0,
        ),
        FLOAT16 => in_set(
            &[FP8E4M3, FP8E5M2, FP8E4M3FNUZ, FP8E5M2FNUZ, UINT8, INT8],
            dt0,
        ),
        UINT64 => in_set(&[UINT32, UINT16, UINT8], dt0),
        UINT32 => in_set(&[UINT16, UINT8], dt0),
        UINT16 => dt0 == UINT8,
        INT64 => in_set(&[UINT32, UINT16, UINT8, INT32, INT16, INT8], dt0),
        INT32 => in_set(&[UINT16, UINT8, INT16, INT8], dt0),
        INT16 => dt0 == UINT8 || dt0 == INT8,
        _ => false,
    }
}

/// `sum_acc_dtype`
pub fn sum_acc_dtype(d: DTypeId) -> DTypeId {
    if is_unsigned(d) {
        least_upper_dtype2(d, UINT32)
    } else if is_int(d) || d == BOOL {
        least_upper_dtype2(d, INT32)
    } else {
        least_upper_dtype2(d, sum_dtype())
    }
}

// ---- truncation (port of `truncate` table + float conversion helpers) ----

/// `float_to_fp16` — f64 -> f16 value (as f64), RNE, overflow -> ±inf.
pub fn float_to_fp16(x: f64) -> f64 {
    let b = f64_to_f16_bits(x);
    f16_bits_to_f64(b)
}

/// raw f16 bits from f64 (RNE), ±inf/NaN preserved.
pub fn f64_to_f16_bits(x: f64) -> u16 {
    let xb = x.to_bits();
    let sign = ((xb >> 48) & 0x8000) as u16;
    let abs = xb & 0x7FFFFFFFFFFFFFFF;
    if x.is_nan() {
        return sign | 0x7E00;
    }
    if abs >= 0x7FF0000000000000 {
        return sign | 0x7C00; // ±inf
    }
    let exp = ((abs >> 52) & 0x7FF) as i32; // biased f64 exponent
    let mant = abs & 0x000FFFFFFFFFFFFF;
    // unbiased f64 exponent -15 = f16 exponent
    let f16exp = exp - 1023 + 15;
    if (1..=30).contains(&f16exp) {
        // normal: mantissa top 10 bits + RNE using the rest
        let shift = 42; // 52 - 10
        let mut m = mant >> shift;
        let rem = mant & ((1u64 << shift) - 1);
        let half = 1u64 << (shift - 1);
        if rem > half || (rem == half && (m & 1) == 1) {
            m += 1;
        }
        let mut e = f16exp as u32;
        if m == 0x400 {
            // mantissa overflowed into exponent
            m = 0;
            e += 1;
        }
        if e > 30 {
            return sign | 0x7C00; // overflow to inf (matches Python OverflowError path)
        }
        return sign | ((e << 10) as u16) | (m as u16);
    }
    if f16exp < 1 {
        if f16exp < -10 {
            // underflow to zero (or inf if huge) — matches Python: huge -> inf handled above
            return sign;
        }
        // subnormal: shift mantissa right; RNE
        let shift = (42 - f16exp) as u32; // extra shift since implicit 1 disappears
        let full = mant | (1u64 << 52); // implicit 1
        let m = full >> shift;
        let rem = full & ((1u64 << shift) - 1);
        let half = 1u64 << (shift - 1);
        let mut m = m;
        if rem > half || (rem == half && (m & 1) == 1) {
            m += 1;
        }
        if m == 0x400 {
            return sign | 0x400; // smallest normal
        }
        return sign | (m as u16);
    }
    // f16exp in 31.. -> overflow to inf
    sign | 0x7C00
}

/// f64 value of raw f16 bits.
pub fn f16_bits_to_f64(b: u16) -> f64 {
    let sign = ((b >> 15) as u64) << 63;
    let exp = ((b >> 10) & 0x1F) as u64;
    let mant = (b & 0x3FF) as u64;
    if exp == 0x1F {
        // inf / nan
        if mant == 0 {
            return f64::from_bits(sign | 0x7FF0000000000000);
        }
        return f64::from_bits(sign | 0x7FF8000000000000);
    }
    if exp == 0 {
        if mant == 0 {
            return f64::from_bits(sign);
        }
        // subnormal: normalize
        let mut e: i32 = -14;
        let mut m = mant;
        while m & 0x400 == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        let fexp = (e + 1023) as u64;
        return f64::from_bits(sign | (fexp << 52) | (m << 42));
    }
    let fexp = exp + 1023 - 15;
    f64::from_bits(sign | (fexp << 52) | (mant << 42))
}

/// `float_to_bf16` — f64 -> bf16 value (as f64), RNE on the f32 mantissa top bits.
pub fn float_to_bf16(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    // truncate[dtypes.float](x): round to nearest f32 first
    let mut u = (x as f32).to_bits();
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000;
    f32::from_bits(u) as f64
}

struct Fp8Cfg {
    bias: i32,
    sig_bits: u32,
    mant_mask: u64,
    min_denorm_half: u64,
    ovf_threshold: u64,
    max_norm: u64,
    min_norm: u64,
}

fn fp8_cfg(d: DTypeId) -> Fp8Cfg {
    match d {
        FP8E4M3 => Fp8Cfg {
            bias: 7,
            sig_bits: 4,
            mant_mask: 0x7,
            min_denorm_half: 0x3F50000000000000,
            ovf_threshold: 0x407D000000000000,
            max_norm: 0x7E,
            min_norm: 0x3F90000000000000,
        },
        FP8E5M2 => Fp8Cfg {
            bias: 15,
            sig_bits: 3,
            mant_mask: 0x3,
            min_denorm_half: 0x3EE0000000000000,
            ovf_threshold: 0x40EE000000000000 - 1,
            max_norm: 0x7B,
            min_norm: 0x3F10000000000000,
        },
        FP8E4M3FNUZ => Fp8Cfg {
            bias: 8,
            sig_bits: 4,
            mant_mask: 0x7,
            min_denorm_half: 0x3F40000000000000,
            ovf_threshold: 0x406F000000000000 - 1,
            max_norm: 0x7F,
            min_norm: 0x3F80000000000000,
        },
        FP8E5M2FNUZ => Fp8Cfg {
            bias: 16,
            sig_bits: 3,
            mant_mask: 0x3,
            min_denorm_half: 0x3ED0000000000000,
            ovf_threshold: 0x40EE000000000000 - 1,
            max_norm: 0x7F,
            min_norm: 0x3F00000000000000,
        },
        _ => panic!("not an fp8 dtype"),
    }
}

fn is_fnuz(d: DTypeId) -> bool {
    d == FP8E4M3FNUZ || d == FP8E5M2FNUZ
}

/// `float_to_fp8` — returns the raw fp8 byte.
pub fn float_to_fp8(x: f64, d: DTypeId) -> u8 {
    let c = fp8_cfg(d);
    if is_fnuz(d) && !x.is_finite() {
        return 0x80;
    }
    if is_fnuz(d) && x == 0.0 {
        return 0x00;
    }
    if d == FP8E4M3 && !x.is_finite() {
        return if x.is_sign_negative() { 0xff } else { 0x7f };
    }
    if d == FP8E5M2 && !x.is_finite() {
        return (if x.is_sign_negative() { 0x80 } else { 0 })
            | if x.is_infinite() { 0x7c } else { 0x7f };
    }
    let xbits = x.to_bits();
    let half_ulp = 1u64 << (52 - c.sig_bits);
    let sign = ((xbits >> 63) & 1) << 7;
    let exp = (((xbits >> 52) & 0x7FF) as i64) - 1023 + c.bias as i64;
    let mantissa = (xbits >> (53 - c.sig_bits)) & c.mant_mask;
    let absx = xbits & 0x7FFFFFFFFFFFFFFF;
    let res: u64 = if absx <= c.min_denorm_half {
        0
    } else if absx > c.ovf_threshold {
        c.max_norm
    } else if absx >= c.min_norm {
        let (mut r, round_bits) = (
            ((exp as u64) << (c.sig_bits - 1)) | mantissa,
            xbits & ((half_ulp << 1) - 1),
        );
        if round_bits > half_ulp || (round_bits == half_ulp && (mantissa & 1) == 1) {
            r += 1;
        }
        r
    } else {
        let shift = (1 - exp) as u32;
        let m = mantissa | (1 << (c.sig_bits - 1));
        let mut r = m >> shift;
        let half = half_ulp << shift;
        let round_bits = (xbits | (1 << 52)) & ((half << 1) - 1);
        if round_bits > half || (round_bits == half && (r & 1) == 1) {
            r += 1;
        }
        r
    };
    if is_fnuz(d) && res == 0 {
        return 0;
    }
    (res | sign) as u8
}

/// `fp8_to_float`
pub fn fp8_to_float(x: u8, d: DTypeId) -> f64 {
    if is_fnuz(d) && x == 0x80 {
        return f64::NAN;
    }
    if (x & 0x7F) == 0 {
        return if x & 0x80 != 0 { -0.0 } else { 0.0 };
    }
    let c = fp8_cfg(d);
    let mant_bits = c.sig_bits - 1;
    let exp_bits = 8 - c.sig_bits;
    let exp_max = (1u32 << exp_bits) - 1;
    let mant_max = (1u32 << mant_bits) - 1;
    let sign = (x >> 7) & 1;
    let exp = ((x >> mant_bits) as u32) & exp_max;
    let mantissa = (x & mant_max as u8) as u32;
    if !is_fnuz(d) && exp == exp_max {
        if d == FP8E5M2 {
            let v = if mantissa != 0 {
                f64::NAN
            } else {
                f64::INFINITY
            };
            return if sign == 1 { -v } else { v };
        }
        if mantissa == mant_max {
            return f64::NAN;
        }
    }
    let val = if exp == 0 {
        (mantissa as f64 / (mant_max as f64 + 1.0)) * 2f64.powi(1 - c.bias)
    } else {
        (1.0 + mantissa as f64 / (mant_max as f64 + 1.0)) * 2f64.powi(exp as i32 - c.bias)
    };
    if sign == 1 { -val } else { val }
}

/// CPython/ctypes int-conversion semantics (probed empirically):
///   v mod 2^64 -> reinterpret per the ctypes type's signedness -> wrap to width.
/// tinygrad only feeds already-const values (Python ints) into the int arms of
/// the truncate table, so `x` arrives from `int(x)` (exact) upstream.
fn trunc_int_ctypes(x: f64, d: DTypeId) -> ConstVal {
    let b = match BigInt::from_f64_trunc(x) {
        Some(b) => b,
        None => return ConstVal::Int(0), // nan/inf: CPython raises; unreachable in tinygrad
    };
    let low = b.low64();
    let m64: u64 = if b.is_negative() && low != 0 {
        (!low).wrapping_add(1)
    } else {
        low
    };
    if is_unsigned(d) {
        // c_ubyte/c_ushort/c_uint/c_ulonglong: unsigned wrap
        ConstVal::Int(match d.bitsize() {
            8 => (m64 as u8) as i128,
            16 => (m64 as u16) as i128,
            32 => (m64 as u32) as i128,
            64 => m64 as i128,
            128 | 256 => m64 as i128,
            _ => unreachable!("bad int bitsize"),
        })
    } else {
        ConstVal::Int(wrap_int(m64 as i64 as i128, d))
    }
}

fn wrap_int(v: i128, d: DTypeId) -> i128 {
    match d.bitsize() {
        8 => v as i8 as i128,
        16 => v as i16 as i128,
        32 => v as i32 as i128,
        64 => v as i64 as i128,
        1 => ((v & 1) as u8 as i8) as i128,
        128 | 256 => v, // not reached from `all`
        _ => unreachable!("bad int bitsize"),
    }
}

pub fn trunc_float(d: DTypeId, x: f64) -> f64 {
    if d == crate::dtype::WEAKFLOAT {
        return x;
    }
    match d {
        FLOAT16 => float_to_fp16(x),
        BFLOAT16 => float_to_bf16(x),
        FP8E4M3 => fp8_to_float(float_to_fp8(x, d), d),
        FP8E5M2 => fp8_to_float(float_to_fp8(x, d), d),
        FP8E4M3FNUZ => fp8_to_float(float_to_fp8(x, d), d),
        FP8E5M2FNUZ => fp8_to_float(float_to_fp8(x, d), d),
        FLOAT32 => (x as f32) as f64,
        FLOAT64 => x,
        _ => panic!("not a float dtype: {}", d.attr()),
    }
}

/// `truncate[dt]` for a python const.
pub fn truncate(d: DTypeId, x: f64) -> ConstVal {
    if is_bool(d) {
        ConstVal::Bool(x != 0.0)
    } else if is_float(d) {
        ConstVal::Float(trunc_float(d, x))
    } else if is_int(d) {
        trunc_int_ctypes(x, d)
    } else {
        panic!("cannot truncate to {}", d.attr())
    }
}

/// `truncate[dt]` for an already-const value (int inputs pass through `int(x)`
/// semantics; only the float path does real truncation).
pub fn truncate_val(d: DTypeId, v: ConstVal) -> ConstVal {
    if is_weak(d) {
        // weak dtypes have no truncate entry (Python falls back to float()/int())
        return v;
    }
    match v {
        ConstVal::Float(f) => truncate(d, f),
        ConstVal::Invalid => ConstVal::Invalid,
        other => {
            if is_float(d) {
                ConstVal::Float(trunc_float(d, other.as_f64()))
            } else if is_bool(d) {
                ConstVal::Bool(!matches!(other, ConstVal::Int(0) | ConstVal::Bool(false)))
            } else {
                other
            }
        }
    }
}

/// Python `int(x)` on a finite float: exact trunc toward zero as BigInt.
pub fn float_to_bigint_trunc(f: f64) -> ConstVal {
    match BigInt::from_f64_trunc(f) {
        Some(b) => b
            .as_i64()
            .map(|i| ConstVal::Int(i as i128))
            .unwrap_or(ConstVal::Big(b)),
        None => ConstVal::Float(f64::NAN),
    }
}

// ---- storage interop ----

/// `storage_fmt_for_dtype` — the char used for raw storage views.
pub fn storage_fmt(d: DTypeId) -> u8 {
    match d {
        BFLOAT16 => b'H',
        FP8E4M3 | FP8E5M2 | FP8E4M3FNUZ | FP8E5M2FNUZ => b'B',
        _ => d.fmt().expect("no storage fmt"),
    }
}

/// Scalar stored in a buffer view: f64 for half (packed back by fmt),
/// u16 for bf16, u8 for fp8, raw value otherwise.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StorageScalar {
    F64(f64),
    U64(u64),
}

/// `to_storage_scalar`
pub fn to_storage_scalar(v: ConstVal, d: DTypeId) -> StorageScalar {
    match d {
        FLOAT16 => StorageScalar::F64(float_to_fp16(match v {
            ConstVal::Float(f) => f,
            other => other.as_f64(),
        })),
        BFLOAT16 => {
            let f = match v {
                ConstVal::Float(f) => f,
                other => other.as_f64(),
            };
            let u = float_to_bf16(f);
            let x = (u as f32).to_bits();
            StorageScalar::U64(((x >> 16) & 0xFFFF) as u64)
        }
        FP8E4M3 | FP8E5M2 | FP8E4M3FNUZ | FP8E5M2FNUZ => StorageScalar::U64(float_to_fp8(
            match v {
                ConstVal::Float(f) => f,
                other => other.as_f64(),
            },
            d,
        ) as u64),
        _ => match v {
            ConstVal::Bool(b) => StorageScalar::U64(b as u64),
            ConstVal::Int(i) => StorageScalar::U64(i as u64),
            ConstVal::Float(f) => StorageScalar::F64(f),
            ConstVal::Big(_) => StorageScalar::U64(0),
            ConstVal::Invalid => StorageScalar::F64(f64::NAN),
        },
    }
}

/// `from_storage_scalar`
pub fn from_storage_scalar(x: StorageScalar, d: DTypeId) -> ConstVal {
    match d {
        BFLOAT16 => {
            let u = match x {
                StorageScalar::U64(u) => u as u32,
                StorageScalar::F64(f) => f as u32,
            };
            ConstVal::Float(f32::from_bits((u & 0xFFFF) << 16) as f64)
        }
        FP8E4M3 | FP8E5M2 | FP8E4M3FNUZ | FP8E5M2FNUZ => {
            let u = match x {
                StorageScalar::U64(u) => u as u8,
                StorageScalar::F64(f) => f as u8,
            };
            ConstVal::Float(fp8_to_float(u, d))
        }
        _ => match x {
            StorageScalar::F64(f) => ConstVal::Float(f),
            StorageScalar::U64(u) => ConstVal::Int(u as i128),
        },
    }
}

/// `bitcast(x, in_dtype, out_dtype)` (dtype.py:292-295) — same-width reinterp.
pub fn bitcast_val(x: ConstVal, in_dt: DTypeId, out_dt: DTypeId) -> ConstVal {
    assert_eq!(
        in_dt.itemsize(),
        out_dt.itemsize(),
        "bitcast itemsize mismatch"
    );
    let w = in_dt.itemsize();
    // to raw little-endian bytes via the storage scalar
    let bits: u128 = match x {
        ConstVal::Bool(b) => b as u128,
        ConstVal::Int(i) => (i as u128) & mask_for(w),
        ConstVal::Big(b) => {
            // 128-bit view of the magnitude with sign applied
            let mut v = b.as_i64().map(|i| i as i128).unwrap_or(0);
            if b.is_negative() && b.as_i64().is_none() {
                v = -1; // saturated sentinel; out of differential contract
            }
            (v as u128) & mask_for(w)
        }
        ConstVal::Float(f) => match w {
            2 => f64_to_f16_bits(f) as u128,
            4 => (f as f32).to_bits() as u128,
            8 => f.to_bits() as u128,
            _ => panic!("TGERR: bad float width {w}"),
        },
        ConstVal::Invalid => 0,
    };
    // reinterpret
    match (is_float(out_dt), is_int(out_dt), is_bool(out_dt)) {
        (true, _, _) => match w {
            2 => ConstVal::Float(fp16_to_f64(bits as u16)),
            4 => ConstVal::Float(f32::from_bits(bits as u32) as f64),
            8 => ConstVal::Float(f64::from_bits(bits as u64)),
            _ => panic!("TGERR: bad float width {w}"),
        },
        (false, true, _) => ConstVal::Int((bits & mask_for(w)) as i128),
        (false, false, true) => ConstVal::Bool(bits != 0),
        _ => panic!("TGERR: bad bitcast target"),
    }
}

fn mask_for(bytes: usize) -> u128 {
    if bytes >= 16 {
        u128::MAX
    } else {
        (1u128 << (bytes * 8)) - 1
    }
}

fn fp16_to_f64(h: u16) -> f64 {
    // reuse the fp16 decoder used by truncate
    // simple reconstruction through the same RNE encoder inverse: use f32 path
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let man = (h & 0x3FF) as u32;
    let f32v: f32 = if exp == 0 {
        if man == 0 {
            f32::from_bits(sign << 31)
        } else {
            // subnormal
            let v = (man as f32) * 2f32.powi(-24) * 2f32.powi(-14);
            f32::from_bits((sign << 31) | v.to_bits() & 0x7FFF_FFFF)
        }
    } else if exp == 31 {
        f32::from_bits((sign << 31) | 0x7F80_0000 | if man != 0 { 1 } else { 0 })
    } else {
        f32::from_bits((sign << 31) | ((exp - 15 + 127) << 23) | (man << 13))
    };
    f32v as f64
}
