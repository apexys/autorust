//! The Rust indexing DSL: Python's `t[1:3, ::2, None]` becomes `t[s![1..3, ..;2, NewAxis]]`.
//!
//! Rust ranges have no step component and `:` alone is not an expression, so the
//! `s!` macro (ndarray-style) accepts `..`, `a..`, `..b`, `a..b` with an optional
//! `;step`, plain int indices, and the Python specials `NewAxis` (None) and
//! `Ellipsis` (`...` also works).
//!
//! `s_parse!`/`p_parse!` fill the accumulator with one `push` per element, which
//! trips `vec_init_then_push` (the count isn't a literal); allow it here where
//! the macro bodies live.
#![allow(clippy::vec_init_then_push)]

use super::Tensor;

/// One element of an index expression — Python's
/// `int | slice | Tensor | None | Ellipsis`.
///
/// `Clone` (not `Copy`) because [`Ix::Tensor`] carries a tensor handle for
/// advanced indexing. `PartialEq` is by UOp identity, which lets the `s!`/`p!`
/// macro tests keep comparing basic specs.
#[derive(Clone, Debug)]
pub enum Ix {
    /// `slice(None, None, None)` / `..`
    All,
    /// Python `slice(start, stop, step)` with int bounds; `step` defaults to 1.
    Range {
        start: Option<i64>,
        stop: Option<i64>,
        step: i64,
    },
    /// Python `int` — collapses the dim.
    Index(i64),
    /// Python `None` — injects a size-1 dim (no dim consumed).
    NewAxis,
    /// Python `Ellipsis` — fills remaining dims with `All` (at most one).
    Ellipsis,
    /// Python `Tensor` index (advanced/fancy indexing). A 0-d tensor index is
    /// treated as a `sint` in tinygrad and is not supported by the port yet.
    Tensor(Tensor),
}

impl PartialEq for Ix {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Ix::All, Ix::All) | (Ix::NewAxis, Ix::NewAxis) | (Ix::Ellipsis, Ix::Ellipsis) => true,
            (Ix::Index(a), Ix::Index(b)) => a == b,
            (
                Ix::Range {
                    start: a0,
                    stop: a1,
                    step: a2,
                },
                Ix::Range {
                    start: b0,
                    stop: b1,
                    step: b2,
                },
            ) => a0 == b0 && a1 == b1 && a2 == b2,
            (Ix::Tensor(a), Ix::Tensor(b)) => a.uop() == b.uop(),
            _ => false,
        }
    }
}
impl Eq for Ix {}

impl Ix {
    pub fn slice(start: Option<i64>, stop: Option<i64>, step: i64) -> Ix {
        Ix::Range { start, stop, step }
    }

    /// Python `t[tensor]`: advanced indexing on this dim.
    pub fn tensor(t: Tensor) -> Ix {
        Ix::Tensor(t)
    }
}

/// `t[s![.., 1]]` — build a `Vec<Ix>` index spec, mirroring Python's
/// comma-separated `__getitem__` tuple.
#[macro_export]
macro_rules! s {
    ($($t:tt)*) => {{
        let mut __ixs: ::std::vec::Vec<$crate::tensor::ix::Ix> = ::std::vec::Vec::new();
        $crate::s_parse!(__ixs, $($t)*);
        __ixs
    }};
}

/// internal: parse the element list into `ixs` (tt-muncher; every element rule
/// consumes the leading comma; negative bounds need their own rules since `-`
/// is punct, not part of a literal).
#[macro_export]
#[doc(hidden)]
macro_rules! s_parse {
    // exhausted (no elements / trailing comma)
    ($ixs:ident) => {};
    ($ixs:ident,) => {};
    // Ellipsis / NewAxis
    ($ixs:ident, ... $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::Ellipsis); $crate::s_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, Ellipsis $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::Ellipsis); $crate::s_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, NewAxis $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::NewAxis); $crate::s_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, .. $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::All); $crate::s_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, $a:literal.. $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(Some($a), None, 1)); $crate::s_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, - $a:literal.. $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(Some(-$a), None, 1)); $crate::s_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, .. $b:literal $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(None, Some($b), 1)); $crate::s_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, $a:literal.. $b:literal $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(Some($a), Some($b), 1)); $crate::s_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, - $a:literal.. $b:literal $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(Some(-$a), Some($b), 1)); $crate::s_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, .. - $b:literal $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(None, Some(-$b), 1)); $crate::s_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, $a:literal.. - $b:literal $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(Some($a), Some(-$b), 1)); $crate::s_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, - $a:literal.. - $b:literal $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(Some(-$a), Some(-$b), 1)); $crate::s_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, .. ; $st:expr $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(None, None, $st)); $crate::s_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, $a:literal.. ; $st:expr $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(Some($a), None, $st)); $crate::s_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, - $a:literal.. ; $st:expr $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(Some(-$a), None, $st)); $crate::s_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, .. $b:literal ; $st:expr $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(None, Some($b), $st)); $crate::s_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, $a:literal.. $b:literal ; $st:expr $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(Some($a), Some($b), $st)); $crate::s_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, - $a:literal.. $b:literal ; $st:expr $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(Some(-$a), Some($b), $st)); $crate::s_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, .. - $b:literal ; $st:expr $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(None, Some(-$b), $st)); $crate::s_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, $a:literal.. - $b:literal ; $st:expr $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(Some($a), Some(-$b), $st)); $crate::s_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, - $a:literal.. - $b:literal ; $st:expr $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(Some(-$a), Some(-$b), $st)); $crate::s_parse!($ixs, $($($rest)*)?); };

    // int index (negative literal: `-` is punct, so it gets its own rule)
    ($ixs:ident, - $i:literal $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::Index(-$i)); $crate::s_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, $i:literal $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::Index($i)); $crate::s_parse!($ixs, $($($rest)*)?); };
}

/// `t[p![:, 1]]` — python-style spelling of `s!`: `1:3`, `::2`, `::-1`,
/// `3:`, `None`, `...`. `:` and `::` are ordinary tokens, so the python
/// slice grammar works inside a macro. Expands to the same `Vec<Ix>`.
#[macro_export]
macro_rules! p {
    ($($t:tt)*) => {{
        let mut __ixs: ::std::vec::Vec<$crate::tensor::ix::Ix> = ::std::vec::Vec::new();
        $crate::p_parse!(__ixs, $($t)*);
        __ixs
    }};
}

/// internal: python-grammar muncher for `p!`. Lexer shapes drive the rules:
/// adjacent colons lex as `::` (stop slot empty), separated colons as single
/// `:` (`start:stop:step`). Each rule consumes the leading comma and recurses.
#[macro_export]
#[doc(hidden)]
macro_rules! p_parse {
    // exhausted (no elements / trailing comma)
    ($ixs:ident) => {};
    ($ixs:ident,) => {};
    // python specials
    ($ixs:ident, None $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::NewAxis); $crate::p_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, NewAxis $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::NewAxis); $crate::p_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, ... $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::Ellipsis); $crate::p_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, Ellipsis $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::Ellipsis); $crate::p_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, :: $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::All); $crate::p_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident,  :: $c:literal $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(None, None, $c)); $crate::p_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident,  :: - $c:literal $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(None, None, -$c)); $crate::p_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, $a:literal :: $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(Some($a), None, 1)); $crate::p_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, $a:literal :: $c:literal $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(Some($a), None, $c)); $crate::p_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, $a:literal :: - $c:literal $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(Some($a), None, -$c)); $crate::p_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, - $a:literal :: $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(Some(-$a), None, 1)); $crate::p_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, - $a:literal :: $c:literal $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(Some(-$a), None, $c)); $crate::p_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, - $a:literal :: - $c:literal $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(Some(-$a), None, -$c)); $crate::p_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident,  : $b:literal : $c:literal $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(None, Some($b), $c)); $crate::p_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident,  : $b:literal : - $c:literal $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(None, Some($b), -$c)); $crate::p_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident,  : - $b:literal : $c:literal $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(None, Some(-$b), $c)); $crate::p_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident,  : - $b:literal : - $c:literal $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(None, Some(-$b), -$c)); $crate::p_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, $a:literal : $b:literal : $c:literal $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(Some($a), Some($b), $c)); $crate::p_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, $a:literal : $b:literal : - $c:literal $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(Some($a), Some($b), -$c)); $crate::p_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, $a:literal : - $b:literal : $c:literal $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(Some($a), Some(-$b), $c)); $crate::p_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, $a:literal : - $b:literal : - $c:literal $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(Some($a), Some(-$b), -$c)); $crate::p_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, - $a:literal : $b:literal : $c:literal $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(Some(-$a), Some($b), $c)); $crate::p_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, - $a:literal : $b:literal : - $c:literal $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(Some(-$a), Some($b), -$c)); $crate::p_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, - $a:literal : - $b:literal : $c:literal $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(Some(-$a), Some(-$b), $c)); $crate::p_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, - $a:literal : - $b:literal : - $c:literal $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(Some(-$a), Some(-$b), -$c)); $crate::p_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident,  : $b:literal $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(None, Some($b), 1)); $crate::p_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident,  : - $b:literal $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(None, Some(-$b), 1)); $crate::p_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, $a:literal : $b:literal $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(Some($a), Some($b), 1)); $crate::p_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, $a:literal : - $b:literal $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(Some($a), Some(-$b), 1)); $crate::p_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, - $a:literal : $b:literal $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(Some(-$a), Some($b), 1)); $crate::p_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, - $a:literal : - $b:literal $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(Some(-$a), Some(-$b), 1)); $crate::p_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, - $i:literal $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::Index(-$i)); $crate::p_parse!($ixs, $($($rest)*)?); };
    // C': stop empty with a single trailing colon (`3:`, `-3:`, plain `:`)
    ($ixs:ident, : $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::All); $crate::p_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, $a:literal : $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(Some($a), None, 1)); $crate::p_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, - $a:literal : $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::slice(Some(-$a), None, 1)); $crate::p_parse!($ixs, $($($rest)*)?); };
    ($ixs:ident, $i:literal $(, $($rest:tt)*)?) => { $ixs.push($crate::tensor::ix::Ix::Index($i)); $crate::p_parse!($ixs, $($($rest)*)?); }
}

#[cfg(test)]
mod tests {
    use super::Ix;

    #[test]
    fn test_s_macro_basics() {
        // Python `t[:, 1]`
        assert_eq!(s![.., 1], vec![Ix::All, Ix::Index(1)]);
        // Python `t[1:3]`
        assert_eq!(s![1..3], vec![Ix::slice(Some(1), Some(3), 1)]);
        // Python `t[::2]`
        assert_eq!(s![..;2], vec![Ix::slice(None, None, 2)]);
        // Python `t[::-1]`
        assert_eq!(s![..;-1], vec![Ix::slice(None, None, -1)]);
        // Python `t[1:5:2, 3:]`
        assert_eq!(
            s![1..5;2, 3..],
            vec![Ix::slice(Some(1), Some(5), 2), Ix::slice(Some(3), None, 1)]
        );
        // Python `t[None, ..-2]` (hmm: `..-2` tokenizes as `..` `-` `2`)
        assert_eq!(
            s![NewAxis, ..-2],
            vec![Ix::NewAxis, Ix::slice(None, Some(-2), 1)]
        );
        // negative start / negative index
        assert_eq!(s![-2..], vec![Ix::slice(Some(-2), None, 1)]);
        assert_eq!(s![-1], vec![Ix::Index(-1)]);
        // ellipsis
        assert_eq!(s![..., 0], vec![Ix::Ellipsis, Ix::Index(0)]);
        assert_eq!(s![Ellipsis, 0], vec![Ix::Ellipsis, Ix::Index(0)]);
        // p!: python slice grammar over : and ::
        assert_eq!(p![:, 1], s![.., 1]);
        assert_eq!(p![1:3], s![1..3]);
        assert_eq!(p![::2], s![..;2]);
        assert_eq!(p![::-1], s![..;-1]);
        assert_eq!(
            p![1:5:2, 3:],
            vec![Ix::slice(Some(1), Some(5), 2), Ix::slice(Some(3), None, 1)]
        );
        assert_eq!(p![:-2], vec![Ix::slice(None, Some(-2), 1)]);
        assert_eq!(p![-2:], vec![Ix::slice(Some(-2), None, 1)]);
        assert_eq!(p![1:-2:2], vec![Ix::slice(Some(1), Some(-2), 2)]);
        assert_eq!(p![None, 2], s![NewAxis, 2]);
        assert_eq!(p![..., 0], s![..., 0]);
        assert_eq!(p![1], vec![Ix::Index(1)]);
        assert_eq!(p![:], vec![Ix::All]);
        assert_eq!(p![::], vec![Ix::All]);
        // empty
        assert_eq!(s![], Vec::<Ix>::new());
        // no trailing comma needed, both forms fine
        assert_eq!(s![1, 2, 3], vec![Ix::Index(1), Ix::Index(2), Ix::Index(3)]);
    }
}
