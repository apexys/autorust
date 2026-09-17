//! Rust ergonomics for shape arguments: the Python layer accepts lists/tuples/
//! scalars via `argfix(*args)`; the Rust layer accepts them through `ShapeArg`:
//!
//! ```ignore
//! t.reshape([2, 3]);        // [i32; 2]
//! t.reshape(&[1, 6]);       // &[i32]
//! t.reshape((2, 3));        // tuple of ints
//! t.reshape(6);             // single dim
//! t.reshape(-1);            // negative dims too
//! t.reshape(shape);         // Shape / &[SVal] / Vec<SVal> / [SVal; N]
//! ```

use crate::uop::sint::{SVal, SvalLike};

use super::Shape;

/// Anything a shape-taking API accepts. Converts eagerly to `Vec<SVal>`.
pub trait ShapeArg {
    fn into_shape_vec(self) -> Vec<SVal>;
}

impl SvalLike for SVal {
    fn into_sval(self) -> SVal {
        self
    }
}

impl<T: SvalLike + Clone, const N: usize> ShapeArg for &[T; N] {
    fn into_shape_vec(self) -> Vec<SVal> {
        self.iter().map(|x| (*x).clone().into_sval()).collect()
    }
}

impl<T: SvalLike + Clone> ShapeArg for &[T] {
    fn into_shape_vec(self) -> Vec<SVal> {
        self.iter().map(|x| (*x).clone().into_sval()).collect()
    }
}

impl<T: SvalLike, const N: usize> ShapeArg for [T; N] {
    fn into_shape_vec(self) -> Vec<SVal> {
        self.into_iter().map(|x| x.into_sval()).collect()
    }
}

impl<T: SvalLike> ShapeArg for Vec<T> {
    fn into_shape_vec(self) -> Vec<SVal> {
        self.into_iter().map(|x| x.into_sval()).collect()
    }
}

impl<T: SvalLike + Clone> ShapeArg for &Vec<T> {
    fn into_shape_vec(self) -> Vec<SVal> {
        self.iter().map(|x| (*x).clone().into_sval()).collect()
    }
}

impl ShapeArg for Shape {
    fn into_shape_vec(self) -> Vec<SVal> {
        self.into_vec()
    }
}

impl ShapeArg for &Shape {
    fn into_shape_vec(self) -> Vec<SVal> {
        self.as_slice().to_vec()
    }
}

macro_rules! impl_shape_arg_scalar {
    ($($t:ty),*) => {$(
        impl ShapeArg for $t {
            fn into_shape_vec(self) -> Vec<SVal> {
                vec![self.into_sval()]
            }
        }
    )*};
}
impl_shape_arg_scalar!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, usize);

macro_rules! impl_shape_arg_tuples {
    ($($t:ident),+; $len:literal) => {};
}
impl<A: SvalLike> ShapeArg for (A,) {
    fn into_shape_vec(self) -> Vec<SVal> {
        vec![self.0.into_sval()]
    }
}
impl<A: SvalLike, B: SvalLike> ShapeArg for (A, B) {
    fn into_shape_vec(self) -> Vec<SVal> {
        vec![self.0.into_sval(), self.1.into_sval()]
    }
}
impl<A: SvalLike, B: SvalLike, C: SvalLike> ShapeArg for (A, B, C) {
    fn into_shape_vec(self) -> Vec<SVal> {
        vec![self.0.into_sval(), self.1.into_sval(), self.2.into_sval()]
    }
}
impl<A: SvalLike, B: SvalLike, C: SvalLike, D: SvalLike> ShapeArg for (A, B, C, D) {
    fn into_shape_vec(self) -> Vec<SVal> {
        vec![self.0.into_sval(), self.1.into_sval(), self.2.into_sval(), self.3.into_sval()]
    }
}
impl<A: SvalLike, B: SvalLike, C: SvalLike, D: SvalLike, E: SvalLike> ShapeArg for (A, B, C, D, E) {
    fn into_shape_vec(self) -> Vec<SVal> {
        vec![self.0.into_sval(), self.1.into_sval(), self.2.into_sval(), self.3.into_sval(), self.4.into_sval()]
    }
}
impl<A: SvalLike, B: SvalLike, C: SvalLike, D: SvalLike, E: SvalLike, F: SvalLike> ShapeArg for (A, B, C, D, E, F) {
    fn into_shape_vec(self) -> Vec<SVal> {
        vec![self.0.into_sval(), self.1.into_sval(), self.2.into_sval(), self.3.into_sval(), self.4.into_sval(), self.5.into_sval()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn len(a: impl ShapeArg) -> usize {
        a.into_shape_vec().len()
    }

    #[test]
    fn test_shape_arg_coercion() {
        // the Python `&[1, 2, 3]` experience
        assert_eq!(len([1, 2, 3]), 3);
        assert_eq!(len([2, 3]), 2);
        assert_eq!(len((2, 3)), 2);
        assert_eq!(len(6), 1);
        assert_eq!(len(-1), 1);
        assert_eq!(len(vec![2usize, 3usize]), 2);
        assert_eq!(len(Shape::from_ints([2, 3, 4])), 3);
        // SVal slices flow through unchanged
        let svals: Vec<SVal> = vec![SVal::I(2), SVal::I(3)];
        assert_eq!(len(&svals), 2);
        // values preserved (negative dims, big ints)
        assert_eq!(&[-1, 6].into_shape_vec(), &[SVal::I(-1), SVal::I(6)]);
        assert_eq!(&(2i64, 3i32).into_shape_vec(), &[SVal::I(2), SVal::I(3)]);
    }
}
