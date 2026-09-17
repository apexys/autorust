//! Port of `tinygrad/uop/__init__.py`: the `Ops` enum and `GroupOp` sets.
//!
//! Discriminants are pinned to the Python IntEnum values (verified by the
//! round-trip differential tests; FastEnum with auto() numbers 1..=79 in
//! declaration order).

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u16)]
pub enum Ops {
    Special = 1,
    Buffer = 2,
    Noop = 3,
    RewriteError = 4,
    Param = 5,
    Function = 6,
    Call = 7,
    Program = 8,
    Linear = 9,
    Source = 10,
    Binary = 11,
    Sink = 12,
    After = 13,
    Group = 14,
    Stack = 15,
    Tuple = 16,
    Gettuple = 17,
    Getaddr = 18,
    Index = 19,
    Shrink = 20,
    Load = 21,
    Store = 22,
    Wmma = 23,
    Cast = 24,
    Bitcast = 25,
    Exp2 = 26,
    Log2 = 27,
    Sin = 28,
    Sqrt = 29,
    Reciprocal = 30,
    Neg = 31,
    Trunc = 32,
    Add = 33,
    Mul = 34,
    Shl = 35,
    Shr = 36,
    Cdiv = 37,
    Max = 38,
    Cmod = 39,
    Cmplt = 40,
    Cmpne = 41,
    Cmpeq = 42,
    Xor = 43,
    Or = 44,
    And = 45,
    Threefry = 46,
    Sub = 47,
    Fdiv = 48,
    Pow = 49,
    Floordiv = 50,
    Floormod = 51,
    Where = 52,
    Mulacc = 53,
    Barrier = 54,
    Range = 55,
    If = 56,
    End = 57,
    Endif = 58,
    Const = 59,
    Custom = 60,
    Customi = 61,
    Ins = 62,
    Contiguous = 63,
    ContiguousBackward = 64,
    Detach = 65,
    Stage = 66,
    Copy = 67,
    Mselect = 68,
    Mstack = 69,
    CustomFunction = 70,
    Reshape = 71,
    Permute = 72,
    Expand = 73,
    Pad = 74,
    Flip = 75,
    Unshard = 76,
    Reduce = 77,
    Allreduce = 78,
    Pyliteral = 79,
}

pub const OPS_ALL: &[Ops] = &[
    Ops::Special, Ops::Buffer, Ops::Noop, Ops::RewriteError, Ops::Param, Ops::Function,
    Ops::Call, Ops::Program, Ops::Linear, Ops::Source, Ops::Binary, Ops::Sink, Ops::After,
    Ops::Group, Ops::Stack, Ops::Tuple, Ops::Gettuple, Ops::Getaddr, Ops::Index, Ops::Shrink,
    Ops::Load, Ops::Store, Ops::Wmma, Ops::Cast, Ops::Bitcast, Ops::Exp2, Ops::Log2, Ops::Sin,
    Ops::Sqrt, Ops::Reciprocal, Ops::Neg, Ops::Trunc, Ops::Add, Ops::Mul, Ops::Shl, Ops::Shr,
    Ops::Cdiv, Ops::Max, Ops::Cmod, Ops::Cmplt, Ops::Cmpne, Ops::Cmpeq, Ops::Xor, Ops::Or,
    Ops::And, Ops::Threefry, Ops::Sub, Ops::Fdiv, Ops::Pow, Ops::Floordiv, Ops::Floormod,
    Ops::Where, Ops::Mulacc, Ops::Barrier, Ops::Range, Ops::If, Ops::End, Ops::Endif,
    Ops::Const, Ops::Custom, Ops::Customi, Ops::Ins, Ops::Contiguous, Ops::ContiguousBackward,
    Ops::Detach, Ops::Stage, Ops::Copy, Ops::Mselect, Ops::Mstack, Ops::CustomFunction,
    Ops::Reshape, Ops::Permute, Ops::Expand, Ops::Pad, Ops::Flip, Ops::Unshard, Ops::Reduce,
    Ops::Allreduce, Ops::Pyliteral,
];

impl Ops {
    pub fn from_u16(v: u16) -> Option<Ops> {
        OPS_ALL.iter().cloned().find(|o| *o as u16 == v)
    }
    /// Python enum member name, e.g. "Ops.CONTIGUOUS_BACKWARD".
    pub fn pyname(self) -> &'static str {
        match self {
            Ops::Special => "Ops.SPECIAL",
            Ops::Buffer => "Ops.BUFFER",
            Ops::Noop => "Ops.NOOP",
            Ops::RewriteError => "Ops.REWRITE_ERROR",
            Ops::Param => "Ops.PARAM",
            Ops::Function => "Ops.FUNCTION",
            Ops::Call => "Ops.CALL",
            Ops::Program => "Ops.PROGRAM",
            Ops::Linear => "Ops.LINEAR",
            Ops::Source => "Ops.SOURCE",
            Ops::Binary => "Ops.BINARY",
            Ops::Sink => "Ops.SINK",
            Ops::After => "Ops.AFTER",
            Ops::Group => "Ops.GROUP",
            Ops::Stack => "Ops.STACK",
            Ops::Tuple => "Ops.TUPLE",
            Ops::Gettuple => "Ops.GETTUPLE",
            Ops::Getaddr => "Ops.GETADDR",
            Ops::Index => "Ops.INDEX",
            Ops::Shrink => "Ops.SHRINK",
            Ops::Load => "Ops.LOAD",
            Ops::Store => "Ops.STORE",
            Ops::Wmma => "Ops.WMMA",
            Ops::Cast => "Ops.CAST",
            Ops::Bitcast => "Ops.BITCAST",
            Ops::Exp2 => "Ops.EXP2",
            Ops::Log2 => "Ops.LOG2",
            Ops::Sin => "Ops.SIN",
            Ops::Sqrt => "Ops.SQRT",
            Ops::Reciprocal => "Ops.RECIPROCAL",
            Ops::Neg => "Ops.NEG",
            Ops::Trunc => "Ops.TRUNC",
            Ops::Add => "Ops.ADD",
            Ops::Mul => "Ops.MUL",
            Ops::Shl => "Ops.SHL",
            Ops::Shr => "Ops.SHR",
            Ops::Cdiv => "Ops.CDIV",
            Ops::Max => "Ops.MAX",
            Ops::Cmod => "Ops.CMOD",
            Ops::Cmplt => "Ops.CMPLT",
            Ops::Cmpne => "Ops.CMPNE",
            Ops::Cmpeq => "Ops.CMPEQ",
            Ops::Xor => "Ops.XOR",
            Ops::Or => "Ops.OR",
            Ops::And => "Ops.AND",
            Ops::Threefry => "Ops.THREEFRY",
            Ops::Sub => "Ops.SUB",
            Ops::Fdiv => "Ops.FDIV",
            Ops::Pow => "Ops.POW",
            Ops::Floordiv => "Ops.FLOORDIV",
            Ops::Floormod => "Ops.FLOORMOD",
            Ops::Where => "Ops.WHERE",
            Ops::Mulacc => "Ops.MULACC",
            Ops::Barrier => "Ops.BARRIER",
            Ops::Range => "Ops.RANGE",
            Ops::If => "Ops.IF",
            Ops::End => "Ops.END",
            Ops::Endif => "Ops.ENDIF",
            Ops::Const => "Ops.CONST",
            Ops::Custom => "Ops.CUSTOM",
            Ops::Customi => "Ops.CUSTOMI",
            Ops::Ins => "Ops.INS",
            Ops::Contiguous => "Ops.CONTIGUOUS",
            Ops::ContiguousBackward => "Ops.CONTIGUOUS_BACKWARD",
            Ops::Detach => "Ops.DETACH",
            Ops::Stage => "Ops.STAGE",
            Ops::Copy => "Ops.COPY",
            Ops::Mselect => "Ops.MSELECT",
            Ops::Mstack => "Ops.MSTACK",
            Ops::CustomFunction => "Ops.CUSTOM_FUNCTION",
            Ops::Reshape => "Ops.RESHAPE",
            Ops::Permute => "Ops.PERMUTE",
            Ops::Expand => "Ops.EXPAND",
            Ops::Pad => "Ops.PAD",
            Ops::Flip => "Ops.FLIP",
            Ops::Unshard => "Ops.UNSHARD",
            Ops::Reduce => "Ops.REDUCE",
            Ops::Allreduce => "Ops.ALLREDUCE",
            Ops::Pyliteral => "Ops.PYLITERAL",
        }
    }
}

/// `GroupOp` — static membership sets.
pub mod group {
    use super::Ops;

    pub const UNARY: &[Ops] = &[
        Ops::Exp2, Ops::Log2, Ops::Sin, Ops::Sqrt, Ops::Reciprocal, Ops::Neg, Ops::Trunc,
    ];
    pub const BINARY: &[Ops] = &[
        Ops::Add, Ops::Mul, Ops::Cdiv, Ops::Max, Ops::Cmod, Ops::Cmplt, Ops::Cmpne, Ops::Cmpeq,
        Ops::Xor, Ops::Shl, Ops::Shr, Ops::Or, Ops::And, Ops::Threefry, Ops::Sub, Ops::Fdiv,
        Ops::Pow, Ops::Floordiv, Ops::Floormod,
    ];
    pub const TERNARY: &[Ops] = &[Ops::Where, Ops::Mulacc];
    pub const ALU: &[Ops] = &[Ops::Exp2, Ops::Log2, Ops::Sin, Ops::Sqrt, Ops::Reciprocal, Ops::Neg, Ops::Trunc,
        Ops::Add, Ops::Mul, Ops::Cdiv, Ops::Max, Ops::Cmod, Ops::Cmplt, Ops::Cmpne, Ops::Cmpeq,
        Ops::Xor, Ops::Shl, Ops::Shr, Ops::Or, Ops::And, Ops::Threefry, Ops::Sub, Ops::Fdiv,
        Ops::Pow, Ops::Floordiv, Ops::Floormod, Ops::Where, Ops::Mulacc];
    pub const BROADCASTABLE: &[Ops] = &[
        Ops::Add, Ops::Mul, Ops::Cdiv, Ops::Max, Ops::Cmod, Ops::Cmplt, Ops::Cmpne, Ops::Cmpeq,
        Ops::Xor, Ops::Shl, Ops::Shr, Ops::Or, Ops::And, Ops::Threefry, Ops::Sub, Ops::Fdiv,
        Ops::Pow, Ops::Floordiv, Ops::Floormod, Ops::Where, Ops::Mulacc,
    ];
    pub const ELEMENTWISE: &[Ops] = &[
        Ops::Exp2, Ops::Log2, Ops::Sin, Ops::Sqrt, Ops::Reciprocal, Ops::Neg, Ops::Trunc,
        Ops::Add, Ops::Mul, Ops::Cdiv, Ops::Max, Ops::Cmod, Ops::Cmplt, Ops::Cmpne, Ops::Cmpeq,
        Ops::Xor, Ops::Shl, Ops::Shr, Ops::Or, Ops::And, Ops::Threefry, Ops::Sub, Ops::Fdiv,
        Ops::Pow, Ops::Floordiv, Ops::Floormod, Ops::Where, Ops::Mulacc, Ops::Cast, Ops::Bitcast,
    ];
    pub const DEFINES: &[Ops] = &[Ops::Param, Ops::Buffer];
    pub const IRREDUCIBLE: &[Ops] = &[Ops::Const, Ops::Special, Ops::Range, Ops::Param, Ops::Getaddr];
    pub const MOVEMENT: &[Ops] = &[Ops::Reshape, Ops::Expand, Ops::Permute, Ops::Pad, Ops::Shrink, Ops::Flip];
    pub const COMMUTATIVE: &[Ops] =
        &[Ops::Add, Ops::Mul, Ops::Max, Ops::Cmpne, Ops::Cmpeq, Ops::Xor, Ops::And, Ops::Or];
    pub const ASSOCIATIVE: &[Ops] = &[Ops::Add, Ops::Mul, Ops::And, Ops::Or, Ops::Max];
    pub const IDEMPOTENT: &[Ops] = &[Ops::Or, Ops::And, Ops::Max];
    pub const REDUCE_OPS: &[Ops] = &[Ops::Add, Ops::Mul, Ops::Max];
    pub const COMPARISON: &[Ops] = &[Ops::Cmplt, Ops::Cmpne, Ops::Cmpeq];

    pub fn in_(set: &[Ops], o: Ops) -> bool {
        set.contains(&o)
    }

    pub fn is_binary(o: Ops) -> bool {
        in_(BINARY, o)
    }

}




// ---------- pm_contiguous_view_offset (ops.py:1772) ----------

use crate::uop::ctx::UOpCtx;
use crate::uop::graph::UopId;
use crate::uop::arg::Arg;
use crate::uop::sym::{cvar, r, sget};
use crate::uop::upat::{PatternMatcher, RuleFn, UPat, UPatOrConst};

/// python `rtag()` — set tag to True (replace is a no-op when already set).
fn rtag(ctx: &mut UOpCtx, b: UopId) -> UopId {
    let (op, dt, srcs, arg, tag) = ctx.snapshot(&b);
    if matches!(tag, Arg::Bool(true)) {
        return b;
    }
    ctx.uop_tag(op, Some(dt), srcs, arg, Arg::Bool(true))
}

/// `pm_contiguous_view_offset`.
pub fn pm_contiguous_view_offset() -> PatternMatcher {
    let rules: Vec<(UPat, RuleFn)> = vec![
        // normalize to 1d bitcasts
        (
            UPat::ops1(Ops::Bitcast).named("b"),
            r(|ctx, st, _| {
                let b = sget(st, "b");
                let shape = ctx.shape(&b).ok().flatten()?;
                if shape.len() != 1 {
                    let flat = crate::tensor::movement::flatten(ctx, ctx.srcs(&b)[0].clone(), -1, -1);
                    let bc = ctx.bitcast(flat, ctx.dt_of(&b));
                    Some(crate::tensor::movement::reshape(ctx, bc, &shape))
                } else {
                    None
                }
            }),
        ),
        // INDEX(b, RANGE) -> INDEX(rtag(b), 0)
        (
            UPat::ops1(Ops::Index)
                .with_src_tup(vec![UPat::var(Some("b"), None), UPat::ops1(Ops::Range)])
                .allow_all_srcs(),
            r(|ctx, st, _| {
                let b = sget(st, "b");
                let zero = ctx.const_py(crate::dtype::ConstVal::Int(0), Some(crate::dtype::default_int()));
                let rb = rtag(ctx, b);
                Some(ctx.index(rb, vec![zero]))
            }),
        ),
        // INDEX(b) with a single src -> INDEX(rtag(b), 0)
        (
            UPat::ops1(Ops::Index).with_src_tup(vec![UPat::var(Some("b"), None)]),
            r(|ctx, st, _| {
                let b = sget(st, "b");
                let zero = ctx.const_py(crate::dtype::ConstVal::Int(0), Some(crate::dtype::default_int()));
                let rb = rtag(ctx, b);
                Some(ctx.index(rb, vec![zero]))
            }),
        ),
        // INDEX(b, RANGE + c) -> INDEX(rtag(b), c)
        (
            UPat::ops1(Ops::Index)
                .with_src_tup(vec![
                    UPat::var(Some("b"), None),
                    UPat::ops1(Ops::Range).add(UPatOrConst::Pat(cvar("c"))),
                ])
                .allow_all_srcs(),
            r(|ctx, st, _| {
                let (b, c) = (sget(st, "b"), sget(st, "c"));
                let rb = rtag(ctx, b);
                Some(ctx.index(rb, vec![c]))
            }),
        ),
        // INDEX(b, CONST c) -> INDEX(rtag(b), c) when numel == 1
        (
            UPat::ops1(Ops::Index)
                .with_src_tup(vec![UPat::var(Some("b"), None), cvar("c")])
                .allow_all_srcs(),
            r(|ctx, st, _| {
                let (b, c) = (sget(st, "b"), sget(st, "c"));
                let rb = rtag(ctx, b);
                Some(ctx.index(rb, vec![c]))
            }),
        ),
    ];
    PatternMatcher::new(rules)
}
