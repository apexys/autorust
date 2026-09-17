//! Rust-idiomatic wrapper for a tensor's shape (`tuple[int | Variable, ...]` in
//! Python). Internally the engine keeps `Vec<SVal>`; the public `Tensor` API
//! hands out this newtype, which derefs to `[SVal]` for list-like ergonomics:
//!
//! ```ignore
//! let s = t.shape(ctx);      // Shape
//! s[0]; s[-1];               // Index<usize> / pythonic Index<isize>
//! &s[..]; s.len(); s.iter(); // via Deref<Target = [SVal]>
//! &s;                        // AsRef<[SVal]> -> any &[SVal] parameter
//! ```

use std::ops::{Deref, Index, Range, RangeFrom, RangeFull, RangeTo};

use crate::uop::sint::SVal;

/// Python `shape`: a sequence of `sint` (`int | Variable`).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct Shape(Vec<SVal>);

impl Shape {
    pub fn new(v: Vec<SVal>) -> Shape {
        Shape(v)
    }

    pub fn from_ints(ints: impl IntoIterator<Item = i128>) -> Shape {
        Shape(ints.into_iter().map(SVal::I).collect())
    }

    pub fn as_slice(&self) -> &[SVal] {
        &self.0
    }

    pub fn into_vec(self) -> Vec<SVal> {
        self.0
    }

    /// Python-style `shape[i]`: negative indices count from the end,
    /// out-of-range panics like `IndexError`.
    pub fn at(&self, i: isize) -> SVal {
        self.get(i).cloned().unwrap_or_else(|| panic!("TGERR: IndexError shape index {i} out of range for len {}", self.0.len()))
    }

    pub fn get(&self, i: isize) -> Option<&SVal> {
        let n = self.0.len() as isize;
        let i = if i < 0 { i + n } else { i };
        if i < 0 || i >= n {
            None
        } else {
            self.0.get(i as usize)
        }
    }

    /// The all-`int` view; `None` if any dim is a `Variable`.
    pub fn ints(&self) -> Option<Vec<i64>> {
        self.0.iter().map(|s| s.clone().as_i64()).collect()
    }

    /// `prod(shape)` as int when fully static.
    pub fn numel_int(&self) -> Option<i128> {
        self.0.iter().try_fold(1i128, |a, s| s.clone().as_i64().map(|v| a * v as i128))
    }

    pub fn is_static(&self) -> bool {
        self.0.iter().all(|s| matches!(s, SVal::I(_)))
    }
}

impl From<Vec<SVal>> for Shape {
    fn from(v: Vec<SVal>) -> Shape {
        Shape(v)
    }
}

impl From<&[SVal]> for Shape {
    fn from(v: &[SVal]) -> Shape {
        Shape(v.to_vec())
    }
}

impl FromIterator<SVal> for Shape {
    fn from_iter<T: IntoIterator<Item = SVal>>(iter: T) -> Shape {
        Shape(iter.into_iter().collect())
    }
}

impl AsRef<[SVal]> for Shape {
    fn as_ref(&self) -> &[SVal] {
        &self.0
    }
}

impl std::borrow::Borrow<[SVal]> for Shape {
    fn borrow(&self) -> &[SVal] {
        &self.0
    }
}

impl Deref for Shape {
    type Target = [SVal];

    fn deref(&self) -> &[SVal] {
        &self.0
    }
}

impl Index<usize> for Shape {
    type Output = SVal;

    fn index(&self, i: usize) -> &SVal {
        self.0.get(i).unwrap_or_else(|| panic!("TGERR: IndexError shape index {i} out of range for len {}", self.0.len()))
    }
}

macro_rules! impl_index_int {
    ($($t:ty),*) => {$(
        impl Index<$t> for Shape {
            type Output = SVal;

            fn index(&self, i: $t) -> &SVal {
                self.get(i as isize).unwrap_or_else(|| panic!("TGERR: IndexError shape index {i} out of range for len {}", self.0.len()))
            }
        }
    )*};
}
// every int type: bare literals (`s[0]`, `s[-1]`) work like Python via the i32
// literal-fallback type, and any integer variable indexes directly.
impl_index_int!(i8, i16, i32, i64, i128, isize);

impl Index<RangeFull> for Shape {
    type Output = [SVal];

    fn index(&self, _: RangeFull) -> &[SVal] {
        &self.0
    }
}

macro_rules! impl_index_range {
    ($idx:ty, $($t:ty),*) => {$(
        impl Index<$idx<$t>> for Shape {
            type Output = [SVal];

            fn index(&self, r: $idx<$t>) -> &[SVal] {
                let n = self.0.len() as isize;
                let conv = |v: $t| v as isize;
                $(
                    let lo = r.start as isize; let hi = r.end as isize;
                    let (lo, hi) = (if lo < 0 { lo + n } else { lo }, if hi < 0 { hi + n } else { hi });
                    return self.0.index(lo.max(0) as usize..hi.clamp(0, n) as usize);
                )?
            }
        }
    )*};
}
// ranges: `s[1..]`, `s[..2]`, `s[1..3]`, with pythonic negatives
impl Index<RangeFrom<i32>> for Shape { type Output = [SVal]; fn index(&self, r: RangeFrom<i32>) -> &[SVal] { let n = self.0.len() as isize; let lo = if r.start < 0 { r.start as isize + n } else { r.start as isize }; self.0.index(lo.max(0) as usize..) } }
impl Index<RangeFrom<usize>> for Shape { type Output = [SVal]; fn index(&self, r: RangeFrom<usize>) -> &[SVal] { self.0.index(r.start..) } }
impl Index<RangeTo<usize>> for Shape { type Output = [SVal]; fn index(&self, r: RangeTo<usize>) -> &[SVal] { self.0.index(..r.end) } }
impl Index<RangeTo<i32>> for Shape { type Output = [SVal]; fn index(&self, r: RangeTo<i32>) -> &[SVal] { let n = self.0.len() as isize; let hi = if r.end < 0 { r.end as isize + n } else { r.end as isize }; self.0.index(..hi.clamp(0, n) as usize) } }
impl Index<Range<i32>> for Shape { type Output = [SVal]; fn index(&self, r: Range<i32>) -> &[SVal] { let n = self.0.len() as isize; let lo = if r.start < 0 { r.start as isize + n } else { r.start as isize }; let hi = if r.end < 0 { r.end as isize + n } else { r.end as isize }; self.0.index(lo.max(0) as usize..hi.clamp(0, n) as usize) } }
impl Index<Range<usize>> for Shape { type Output = [SVal]; fn index(&self, r: Range<usize>) -> &[SVal] { self.0.index(r) } }



/// Python-like tuple repr: `(2, 3, 'a4')`-style (`int`s bare, Variables rendered
/// by their UOp debug form).
impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("(")?;
        for (i, s) in self.0.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            match s {
                SVal::I(v) => write!(f, "{v}")?,
                SVal::U(u) => write!(f, "u{u:?}")?,
            }
        }
        if self.0.len() == 1 {
            f.write_str(",")?;
        }
        f.write_str(")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_indexing() {
        let s = Shape::from_ints([2, 3, 4]);
        // bare literals (i32 fallback), negative indexing, and typed variables
        assert_eq!(s[0], SVal::I(2));
        assert_eq!(s[2], SVal::I(4));
        assert_eq!(s[-1], SVal::I(4));
        assert_eq!(s[-3], SVal::I(2));
        let i: i64 = 1;
        assert_eq!(s[i], SVal::I(3));
        let u: usize = 0;
        assert_eq!(s[u], SVal::I(2));
        assert_eq!(s.get(3), None);
        assert_eq!(s.get(-4), None);
    }

    #[test]
    fn test_shape_deref_and_asref() {
        let s = Shape::from_ints([2, 3]);
        assert_eq!(s.len(), 2);
        assert_eq!(s.iter().count(), 2);
        assert_eq!(&s[..], &[SVal::I(2), SVal::I(3)]);
        assert_eq!(&s[1..], &[SVal::I(3)]);
        // AsRef<[SVal]> passes to slice parameters
        fn takes_slice(x: &[SVal]) -> usize {
            x.len()
        }
        assert_eq!(takes_slice(&s), 2);
        assert_eq!(takes_slice(s.as_ref()), 2);
        assert_eq!(s.ints(), Some(vec![2, 3]));
        assert_eq!(s.numel_int(), Some(6));
        assert!(s.is_static());
    }

    #[test]
    fn test_shape_display() {
        assert_eq!(Shape::from_ints([2, 3]).to_string(), "(2, 3)");
        assert_eq!(Shape::from_ints([5]).to_string(), "(5,)");
        assert_eq!(Shape::from_ints([]).to_string(), "()");
    }
}
