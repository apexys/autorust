//! Port of `tinygrad/uop/__init__.py`: the `Ops` enum and `GroupOp` sets.
//!
//! Discriminants are pinned to the Python IntEnum values (verified by the
//! round-trip differential tests; FastEnum with auto() numbers 1..=79 in
//! declaration order).

/// `Ops` — the op codes of the UOp graph, mirroring tinygrad's `Ops` `IntEnum`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u16)]
pub enum Ops {
    /// `Ops.SPECIAL` — a `RANGE` for GPU dimensions, similar to symbolic shapes.
    Special = 1,
    /// `Ops.BUFFER` — allocates global/local/register storage depending on its addrspace.
    Buffer = 2,
    /// `Ops.NOOP` — a uop that is not rendered.
    Noop = 3,
    /// `Ops.REWRITE_ERROR` — a not-rendered uop recording a failed rewrite match.
    RewriteError = 4,
    /// `Ops.PARAM` — a buffer-backed parameter leaf that optimizers can update in place.
    Param = 5,
    /// `Ops.FUNCTION` — a differentiable function whose body is a `TUPLE`.
    Function = 6,
    /// `Ops.CALL` — an opaque kernel invocation.
    Call = 7,
    /// `Ops.PROGRAM` — a renderer program node.
    Program = 8,
    /// `Ops.LINEAR` — a list of UOps forming a linear program.
    Linear = 9,
    /// `Ops.SOURCE` — source code carried as a human-readable `str` arg.
    Source = 10,
    /// `Ops.BINARY` — compiled code carried as a bytes arg.
    Binary = 11,
    /// `Ops.SINK` — the kernel root that collects the values to compute.
    Sink = 12,
    /// `Ops.AFTER` — passes `src[0]` through and orders consumers after `src[1..]`.
    After = 13,
    /// `Ops.GROUP` — a `NOOP` that just merges things together.
    Group = 14,
    /// `Ops.STACK` — vector creation from its srcs.
    Stack = 15,
    /// `Ops.TUPLE` — tuple construction for functions with multiple returns.
    Tuple = 16,
    /// `Ops.GETTUPLE` — selects one element of a `TUPLE`.
    Gettuple = 17,
    /// `Ops.GETADDR` — HCQ-specific address retrieval.
    Getaddr = 18,
    /// `Ops.INDEX` — like `ADD`, but it operates on pointers.
    Index = 19,
    /// `Ops.SHRINK` — movement op slicing each dimension to a sub-range.
    Shrink = 20,
    /// `Ops.LOAD` — load from a buffer.
    Load = 21,
    /// `Ops.STORE` — store into a buffer.
    Store = 22,
    /// `Ops.WMMA` — tensor-core math op, not elementwise.
    Wmma = 23,
    /// Unary `Ops.CAST` — dtype conversion.
    Cast = 24,
    /// Unary `Ops.BITCAST` — reinterpret the bits as another dtype.
    Bitcast = 25,
    /// Unary `Ops.EXP2` — base-2 exponential.
    Exp2 = 26,
    /// Unary `Ops.LOG2` — base-2 logarithm.
    Log2 = 27,
    /// Unary `Ops.SIN` — sine.
    Sin = 28,
    /// Unary `Ops.SQRT` — square root.
    Sqrt = 29,
    /// Unary `Ops.RECIPROCAL` — `1/x`.
    Reciprocal = 30,
    /// Unary `Ops.NEG` — negation.
    Neg = 31,
    /// Unary `Ops.TRUNC` — truncation toward zero.
    Trunc = 32,
    /// `Ops.ADD` — element-wise addition.
    Add = 33,
    /// `Ops.MUL` — element-wise multiplication.
    Mul = 34,
    /// `Ops.SHL` — element-wise shift left.
    Shl = 35,
    /// `Ops.SHR` — element-wise shift right.
    Shr = 36,
    /// `Ops.CDIV` — C-style (truncating) division.
    Cdiv = 37,
    /// `Ops.MAX` — element-wise maximum.
    Max = 38,
    /// `Ops.CMOD` — C-style (truncating) modulo.
    Cmod = 39,
    /// `Ops.CMPLT` — less-than comparison, yielding a bool.
    Cmplt = 40,
    /// `Ops.CMPNE` — not-equal comparison, yielding a bool.
    Cmpne = 41,
    /// `Ops.CMPEQ` — equality comparison, yielding a bool.
    Cmpeq = 42,
    /// `Ops.XOR` — bitwise exclusive or.
    Xor = 43,
    /// `Ops.OR` — bitwise or.
    Or = 44,
    /// `Ops.AND` — bitwise and.
    And = 45,
    /// `Ops.THREEFRY` — Threefry random-number mixing step.
    Threefry = 46,
    /// `Ops.SUB` — element-wise subtraction.
    Sub = 47,
    /// `Ops.FDIV` — true division.
    Fdiv = 48,
    /// `Ops.POW` — exponentiation.
    Pow = 49,
    /// `Ops.FLOORDIV` — floor division.
    Floordiv = 50,
    /// `Ops.FLOORMOD` — floor modulo (sign follows the divisor).
    Floormod = 51,
    /// Ternary `Ops.WHERE` — select between two values by a bool predicate.
    Where = 52,
    /// Ternary `Ops.MULACC` — fused multiply-add of its three srcs.
    Mulacc = 53,
    /// `Ops.BARRIER` — control-flow barrier.
    Barrier = 54,
    /// `Ops.RANGE` — a loop range variable; the arg holds end and axis.
    Range = 55,
    /// `Ops.IF` — start of a conditional block.
    If = 56,
    /// `Ops.END` — end of a `RANGE` loop.
    End = 57,
    /// `Ops.ENDIF` — end of an `IF` block.
    Endif = 58,
    /// `Ops.CONST` — a constant of the uop dtype.
    Const = 59,
    /// `Ops.CUSTOM` — a custom codegen string emitted as an out-of-line call.
    Custom = 60,
    /// `Ops.CUSTOMI` — a custom codegen string inlined into the kernel.
    Customi = 61,
    /// `Ops.INS` — a machine instruction.
    Ins = 62,
    /// `Ops.CONTIGUOUS` — scheduler hint that the value is contiguous.
    Contiguous = 63,
    /// `Ops.CONTIGUOUS_BACKWARD` — scheduler hint for the backward pass of `CONTIGUOUS`.
    ContiguousBackward = 64,
    /// `Ops.DETACH` — scheduler hint detaching the value from gradients.
    Detach = 65,
    /// `Ops.STAGE` — buffer op staging a value into memory.
    Stage = 66,
    /// `Ops.COPY` — buffer op copying a value to another device.
    Copy = 67,
    /// `Ops.MSELECT` — select one buffer of a multi-buffer.
    Mselect = 68,
    /// `Ops.MSTACK` — stack buffers into a multi-buffer.
    Mstack = 69,
    /// `Ops.CUSTOM_FUNCTION` — a user function with a user-supplied backward.
    CustomFunction = 70,
    /// Movement `Ops.RESHAPE` — reinterpret the shape with the same elements.
    Reshape = 71,
    /// Movement `Ops.PERMUTE` — reorder the axes.
    Permute = 72,
    /// Movement `Ops.EXPAND` — broadcast size-1 dimensions to the given sizes.
    Expand = 73,
    /// Movement `Ops.PAD` — pad each dimension.
    Pad = 74,
    /// Movement `Ops.FLIP` — reverse the given axes.
    Flip = 75,
    /// `Ops.UNSHARD` — movement op gathering sharded pieces into one buffer.
    Unshard = 76,
    /// `Ops.REDUCE` — reduce over an axis using the op in the arg.
    Reduce = 77,
    /// `Ops.ALLREDUCE` — cross-device reduction.
    Allreduce = 78,
    /// `Ops.PYLITERAL` — a Python literal arg for `CUSTOM` predicates.
    Pyliteral = 79,
}

/// `GroupOp.All` — every `Ops` variant in discriminant order.
pub const OPS_ALL: &[Ops] = &[
    Ops::Special,
    Ops::Buffer,
    Ops::Noop,
    Ops::RewriteError,
    Ops::Param,
    Ops::Function,
    Ops::Call,
    Ops::Program,
    Ops::Linear,
    Ops::Source,
    Ops::Binary,
    Ops::Sink,
    Ops::After,
    Ops::Group,
    Ops::Stack,
    Ops::Tuple,
    Ops::Gettuple,
    Ops::Getaddr,
    Ops::Index,
    Ops::Shrink,
    Ops::Load,
    Ops::Store,
    Ops::Wmma,
    Ops::Cast,
    Ops::Bitcast,
    Ops::Exp2,
    Ops::Log2,
    Ops::Sin,
    Ops::Sqrt,
    Ops::Reciprocal,
    Ops::Neg,
    Ops::Trunc,
    Ops::Add,
    Ops::Mul,
    Ops::Shl,
    Ops::Shr,
    Ops::Cdiv,
    Ops::Max,
    Ops::Cmod,
    Ops::Cmplt,
    Ops::Cmpne,
    Ops::Cmpeq,
    Ops::Xor,
    Ops::Or,
    Ops::And,
    Ops::Threefry,
    Ops::Sub,
    Ops::Fdiv,
    Ops::Pow,
    Ops::Floordiv,
    Ops::Floormod,
    Ops::Where,
    Ops::Mulacc,
    Ops::Barrier,
    Ops::Range,
    Ops::If,
    Ops::End,
    Ops::Endif,
    Ops::Const,
    Ops::Custom,
    Ops::Customi,
    Ops::Ins,
    Ops::Contiguous,
    Ops::ContiguousBackward,
    Ops::Detach,
    Ops::Stage,
    Ops::Copy,
    Ops::Mselect,
    Ops::Mstack,
    Ops::CustomFunction,
    Ops::Reshape,
    Ops::Permute,
    Ops::Expand,
    Ops::Pad,
    Ops::Flip,
    Ops::Unshard,
    Ops::Reduce,
    Ops::Allreduce,
    Ops::Pyliteral,
];

impl Ops {
    /// Decode a Python `IntEnum` discriminant back into an `Ops`, if it exists.
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

    /// `GroupOp.Unary` — single-src elementwise math ops.
    pub const UNARY: &[Ops] = &[
        Ops::Exp2,
        Ops::Log2,
        Ops::Sin,
        Ops::Sqrt,
        Ops::Reciprocal,
        Ops::Neg,
        Ops::Trunc,
    ];
    /// `GroupOp.Binary` — two-src elementwise math ops.
    pub const BINARY: &[Ops] = &[
        Ops::Add,
        Ops::Mul,
        Ops::Cdiv,
        Ops::Max,
        Ops::Cmod,
        Ops::Cmplt,
        Ops::Cmpne,
        Ops::Cmpeq,
        Ops::Xor,
        Ops::Shl,
        Ops::Shr,
        Ops::Or,
        Ops::And,
        Ops::Threefry,
        Ops::Sub,
        Ops::Fdiv,
        Ops::Pow,
        Ops::Floordiv,
        Ops::Floormod,
    ];
    /// `GroupOp.Ternary` — three-src elementwise math ops.
    pub const TERNARY: &[Ops] = &[Ops::Where, Ops::Mulacc];
    /// `GroupOp.ALU` — the union of the unary, binary and ternary op sets.
    pub const ALU: &[Ops] = &[
        Ops::Exp2,
        Ops::Log2,
        Ops::Sin,
        Ops::Sqrt,
        Ops::Reciprocal,
        Ops::Neg,
        Ops::Trunc,
        Ops::Add,
        Ops::Mul,
        Ops::Cdiv,
        Ops::Max,
        Ops::Cmod,
        Ops::Cmplt,
        Ops::Cmpne,
        Ops::Cmpeq,
        Ops::Xor,
        Ops::Shl,
        Ops::Shr,
        Ops::Or,
        Ops::And,
        Ops::Threefry,
        Ops::Sub,
        Ops::Fdiv,
        Ops::Pow,
        Ops::Floordiv,
        Ops::Floormod,
        Ops::Where,
        Ops::Mulacc,
    ];
    /// `GroupOp.Broadcastable` — binary and ternary ops whose srcs may broadcast.
    pub const BROADCASTABLE: &[Ops] = &[
        Ops::Add,
        Ops::Mul,
        Ops::Cdiv,
        Ops::Max,
        Ops::Cmod,
        Ops::Cmplt,
        Ops::Cmpne,
        Ops::Cmpeq,
        Ops::Xor,
        Ops::Shl,
        Ops::Shr,
        Ops::Or,
        Ops::And,
        Ops::Threefry,
        Ops::Sub,
        Ops::Fdiv,
        Ops::Pow,
        Ops::Floordiv,
        Ops::Floormod,
        Ops::Where,
        Ops::Mulacc,
    ];
    /// `GroupOp.Elementwise` — the ALU set plus `CAST` and `BITCAST`.
    pub const ELEMENTWISE: &[Ops] = &[
        Ops::Exp2,
        Ops::Log2,
        Ops::Sin,
        Ops::Sqrt,
        Ops::Reciprocal,
        Ops::Neg,
        Ops::Trunc,
        Ops::Add,
        Ops::Mul,
        Ops::Cdiv,
        Ops::Max,
        Ops::Cmod,
        Ops::Cmplt,
        Ops::Cmpne,
        Ops::Cmpeq,
        Ops::Xor,
        Ops::Shl,
        Ops::Shr,
        Ops::Or,
        Ops::And,
        Ops::Threefry,
        Ops::Sub,
        Ops::Fdiv,
        Ops::Pow,
        Ops::Floordiv,
        Ops::Floormod,
        Ops::Where,
        Ops::Mulacc,
        Ops::Cast,
        Ops::Bitcast,
    ];
    /// `GroupOp.Defines` — ops that define a buffer-backed leaf.
    pub const DEFINES: &[Ops] = &[Ops::Param, Ops::Buffer];
    /// `GroupOp.Irreducible` — ops that graph rewrites must not look inside.
    pub const IRREDUCIBLE: &[Ops] = &[
        Ops::Const,
        Ops::Special,
        Ops::Range,
        Ops::Param,
        Ops::Getaddr,
    ];
    /// `GroupOp.Movement` — ops that change how a buffer is viewed.
    pub const MOVEMENT: &[Ops] = &[
        Ops::Reshape,
        Ops::Expand,
        Ops::Permute,
        Ops::Pad,
        Ops::Shrink,
        Ops::Flip,
    ];
    /// `GroupOp.Commutative` — binary ops that can be flipped: `f(a,b) == f(b,a)`.
    pub const COMMUTATIVE: &[Ops] = &[
        Ops::Add,
        Ops::Mul,
        Ops::Max,
        Ops::Cmpne,
        Ops::Cmpeq,
        Ops::Xor,
        Ops::And,
        Ops::Or,
    ];
    /// `GroupOp.Associative` — binary ops where `f(f(a,b),c) == f(a,f(b,c))`.
    pub const ASSOCIATIVE: &[Ops] = &[Ops::Add, Ops::Mul, Ops::And, Ops::Or, Ops::Max];
    /// `GroupOp.Idempotent` — binary ops that satisfy `f(x,x) == x`.
    pub const IDEMPOTENT: &[Ops] = &[Ops::Or, Ops::And, Ops::Max];
    /// `GroupOp.Reduce` — ALU ops valid as the reduce op in `REDUCE`/`ALLREDUCE` args.
    pub const REDUCE_OPS: &[Ops] = &[Ops::Add, Ops::Mul, Ops::Max];
    /// `GroupOp.Comparison` — ops whose result dtype is bool.
    pub const COMPARISON: &[Ops] = &[Ops::Cmplt, Ops::Cmpne, Ops::Cmpeq];

    /// Python's `x in set`, i.e. membership in a `GroupOp` slice.
    pub fn in_(set: &[Ops], o: Ops) -> bool {
        set.contains(&o)
    }

    /// True when `o` belongs to `GroupOp.Binary`.
    pub fn is_binary(o: Ops) -> bool {
        in_(BINARY, o)
    }
}

// ---------- pm_contiguous_view_offset (ops.py:1772) ----------

use crate::uop::arg::Arg;
use crate::uop::ctx::UOpCtx;
use crate::uop::graph::UopId;
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
                    let flat =
                        crate::tensor::movement::flatten(ctx, ctx.srcs(&b)[0].clone(), -1, -1);
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
                let zero = ctx.const_py(
                    crate::dtype::ConstVal::Int(0),
                    Some(crate::dtype::default_int()),
                );
                let rb = rtag(ctx, b);
                Some(ctx.index(rb, vec![zero]))
            }),
        ),
        // INDEX(b) with a single src -> INDEX(rtag(b), 0)
        (
            UPat::ops1(Ops::Index).with_src_tup(vec![UPat::var(Some("b"), None)]),
            r(|ctx, st, _| {
                let b = sget(st, "b");
                let zero = ctx.const_py(
                    crate::dtype::ConstVal::Int(0),
                    Some(crate::dtype::default_int()),
                );
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
