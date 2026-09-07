//! `sint` = Python `int | UOp`. Ints are carried as `i128`; tinygrad's own
//! symbolic values fit comfortably (documented divergence: Python is exact for
//! arbitrarily large ints — values beyond i128 in vmin/vmax/shape arithmetic
//! are outside the differential contract, and ALU const folding reports an
//! error instead of producing a wrong value).

use std::cmp::Ordering;

use super::graph::UopId;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SInt {
    I(i128),
    U(UopId),
}

impl SInt {
    pub fn as_i64(self) -> Option<i64> {
        match self {
            SInt::I(v) => i64::try_from(v).ok(),
            SInt::U(_) => None,
        }
    }

    pub fn as_i128_or_panic(self) -> i128 {
        match self {
            SInt::I(v) => v,
            SInt::U(_) => panic!("TGERR: symbolic value where int required"),
        }
    }

    pub fn as_i64_or_panic(self) -> i64 {
        self.as_i64().unwrap_or_else(|| panic!("TGERR: symbolic value where int required"))
    }

    pub fn as_usize(self) -> Option<usize> {
        self.as_i64().and_then(|v| usize::try_from(v).ok())
    }

    pub fn is_uop(self) -> bool {
        matches!(self, SInt::U(_))
    }
}

pub use SInt as SVal;

impl From<i64> for SInt {
    fn from(v: i64) -> Self {
        SInt::I(v as i128)
    }
}

impl From<usize> for SInt {
    fn from(v: usize) -> Self {
        SInt::I(v as i128)
    }
}

impl PartialOrd for SInt {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (SInt::I(a), SInt::I(b)) => Some(a.cmp(b)),
            _ => None,
        }
    }
}

/// Python `prod` over a shape; returns `None` if any element is symbolic
/// (Python would build a UOp expression instead — callers handle that).
pub fn prod_int(shape: &[SInt]) -> Option<i128> {
    let mut acc: i128 = 1;
    for s in shape {
        match s {
            SInt::I(v) => acc = acc.checked_mul(*v)?,
            SInt::U(_) => return None,
        }
    }
    Some(acc)
}
