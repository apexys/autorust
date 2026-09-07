# Indexing — the Rust port

This document describes how indexing works in the `autorust` port: the Rust
surface, the UOps each construct produces, and every place where the Rust API
*necessarily* differs from Python's `Tensor.__getitem__` syntax.

The semantics themselves are a faithful port of `mixin/movement.py`
(`_parse_view_index`, `_apply_view_ops`) and `mixin/op.py` (`_getitem`) — the
goal is **graph-key equality** with Python, not just numerical equality.

---

## 1. The three layers

| Layer | Python | Rust |
|---|---|---|
| Shape as a value | `tuple[int \| Variable, ...]` | `tensor::Shape` (newtype over `Vec<SVal>`) |
| Shape arguments | `argfix(*args)` (tuple or varargs) | `tensor::ShapeArg` (conversion trait) |
| Element access | `t[1:3, ::2, None]` | `t.getitem(&s![1..3, ..;2, NewAxis])` |

### 1.1 `Shape` — list-like ergonomics

`Tensor::shape` returns `tensor::Shape`, which derefs to `[SVal]`:

```rust
let s = t.shape(ctx);
s[0]              // Index<i32> — bare literals work (Rust's i32 literal fallback)
s[-1]             // pythonic negative indexing (Index<i32>; also i8..i128, isize)
s[i]              // i: usize / i64 / isize / ... — every int type implements Index
&s[..3]           // RangeTo — slices, with pythonic negatives (&s[-2..] etc.)
s.len(); s.iter(); &s[..]        // via Deref<Target = [SVal]>
&s                // AsRef<[SVal]> / Borrow<[SVal]> — passes to any &[SVal] parameter
s.ints()          // Option<Vec<i64>> — None if any dim is a Variable
s.numel_int()     // Option<i128>
s.is_static()
```

A dim is an `SVal` = `int | UOp` (Python's `sint`). `SVal::I(i128)` for static
dims, `SVal::U(UopId)` for `VARIABLE`-shaped dims. Display renders Python-style
tuple text (`(2, 3)`).

### 1.2 `ShapeArg` — anything can be a shape

Python's `reshape(2, 3)` / `reshape((2, 3))` / `reshape([2, 3])` all funnel
through `argfix`. The Rust equivalent is the `ShapeArg` trait on the *parameter*
position (a `&[1, 2, 3]` literal cannot coerce to `&[SVal]` directly — Rust
never converts slice element types automatically, so the conversion is done
eagerly by the trait):

```rust
t.reshape([2, 3]);       // [i32; 2]
t.reshape(&[1, 6]);      // &[i32] (&[T; N] also works)
t.reshape((2, 3));       // tuples up to arity 6, homogeneous or mixed int types
t.reshape(6);            // single dim — including negative: reshape(-1)
t.reshape(shape);        // Shape / &Shape / Vec<SVal> / &[SVal]
```

`SVal: From<{integer}>` (`SvalLike`) backs the conversions, so `SVal::from(3)`
also works where a raw dim is needed.

### 1.3 `s![...]` / `p![...]` — the index DSL

Rust has no `:` subscript syntax and its ranges have no step component, so
element access uses macro DSLs. `s!` is ndarray-style (`;` introduces the
step); `p!` accepts the **python slice grammar verbatim** — `:` and `::` are
ordinary tokens, so `p![1:5:2, ::-1]` works directly. Both produce the same
`Vec<Ix>`; pick per taste:

| Python | Rust |
|---|---|
| `t[:, 1]` | `t.getitem(&s![.., 1])` |
| `t[1:3]` | `t.getitem(&s![1..3])` |
| `t[::2]` | `t.getitem(&s![..;2])` |
| `t[::-1]` | `t.getitem(&s![..;-1])` |
| `t[1:5:2, 3:]` | `t.getitem(&s![1..5;2, 3..])` |
| `t[None, 2]` | `t.getitem(&s![NewAxis, 2])` |
| `t[..., 0]` | `t.getitem(&s![..., 0])` (or `Ellipsis`) |
| `t[1, 2]` | `t.getitem(&s![1, 2])` |
| `t[-1]` | `t.getitem(&s![-1])` / `t.getitem(&p![-1])` |

Negative bounds are supported everywhere Python allows them (`s![-2..]`,
`s![..-2]`, `s![1..-2;2]`) — inside the macro `-` is a separate token, so each
sign combination is an explicit macro rule.

Lexer note for `p!` (implementation detail, but it explains the macro's
internals): adjacent colons lex as the single token `::` (`1::2` → `1 :: 2`,
`::-1` → `:: - 1`), while colons separated by a bound lex as single `:` tokens
(`1:5:2` → five tokens, `1:-2:2` → six). `p_parse!` therefore has three rule
families — `[a] :: [c]`, `[a] : [b] : [c]`, `[a] : [b]` — each with sign
combinations for negative bounds.

The macro produces `Vec<tensor::Ix>`; `Ix` is the Rust spelling of Python's
`int | slice | None | Ellipsis` index elements:

```rust
pub enum Ix {
    All,                                                       // slice(None)
    Range { start: Option<i64>, stop: Option<i64>, step: i64 },// slice(ints)
    Index(i64),                                                // int — collapses the dim
    NewAxis,                                                   // None — injects a size-1 dim
    Ellipsis,                                                  // fills remaining dims with All
}
```

`Ix` is `Copy`, so specs can be built programmatically and mixed with the
macro: `let i = ix_of_dim; t.getitem(&[Ix::All, Ix::Index(i)])`.

---

## 2. What a `getitem` compiles to

`t[s![...]]` lowers through the exact Python pipeline:

1. **Normalize** (`_normalize_indices`): at most one `Ellipsis`; the ellipsis
   position (or the end) is filled with `ndim - num_real` copies of `All`;
   `num_real > ndim` is an `IndexError`.
2. **Parse** each element against `shape[dim]` (`_parse_view_index`):
   - `NewAxis`: `{size: 1, boundary: (0, 1), stride: 1}` — consumes no dim.
   - `Index(i)`: bounds-checked (`resolve(i >= size) or resolve(i < -size)` →
     `IndexError`), negative `i` wraps with `+size`, collapses the dim
     (`boundary: (b, b+1)`).
   - Range with all-int bounds: CPython's `slice.indices(size)` semantics —
     clamping against `size.vmax` when the dim is symbolic, then
     `bound = [0,0]` if `stride*(stop-start) < 0`, `[stop+1, start+1]` if
     `stride < 0`; `size = ceildiv(b1-b0, |stride|)`.
   - Range with a symbolic dim: only `step == 1` and `resolve(stop-start >= 0)`
     is accepted (`TypeError` otherwise), `size = stop - start` as `SVal`s.
3. **Apply view ops** (`_apply_view_ops`) on the non-`NewAxis` dims:
   `SHRINK(boundaries)` → `FLIP(negative strides)` → if any `|stride| != 1`:
   `pad_to(round_up(s, st))` → `RESHAPE((s//st, st)...)` →
   `shrink_to((s, 1)...)` → `RESHAPE(shape[::2])`.
4. **Reshape** to the parsed sizes of the non-collapsed dims (this injects the
   `NewAxis` size-1 dims and drops collapsed `Index` dims).

Advanced indexing (`t[Tensor([0,2]), :]`) is **not implemented yet** — `Ix`
will gain a tensor variant when it lands (see the M3 deferred list in
`PLAN.md`). Same for `__setitem__` (assign/store machinery, M4+).

## 3. Deliberate differences from Python

1. **Syntax, not semantics.** `:` / `::step` don't exist in Rust; `;step` in
   `s!` and `Range`-types on `Shape` are the substitutes. Everything lowers to
   the same Python pipeline.
2. **`i32` literal fallback.** `s[0]` / `s[-1]` work because every int type
   implements `Index` and Rust's integer-literal fallback picks `i32`. A
   variable of a concrete int type indexes directly.
3. **Panics vs exceptions.** Python raises `IndexError`/`ValueError`/
   `TypeError`; Rust `panic!`s with a `TGERR:`-prefixed message carrying the
   Python text. (The differential oracle treats both as "error" outcomes.)
4. **`NewAxis` / `Ellipsis` spellings.** Python's `None` and `...` are named
   (`NewAxis`, `Ellipsis`) because `None`/`...` are not Rust expressions; the
   `...` token *is* accepted inside `s!`.
5. **No implicit list→Tensor promotion (yet).** `t[[0, 1, 2]]` in Python
   builds a const tensor and takes the advanced-indexing path; the Rust basic
   path does not accept it until advanced indexing lands.
6. **Shape arg conversion is eager.** `ShapeArg` materializes a `Vec<SVal>`
   per call — Python's `argfix` returns a tuple view. Irrelevant for graph
   construction; noted for anyone porting hot loops.

## 4. Engine-level notes (for porters)

- Shape values inside the engine are `SVal` (`uop::sint::SInt`). Arithmetic on
  them goes through `UOpCtx::sval_add/sub/mul/div_floor` — int-int folds, UOp
  sides keep the receiver order with the int side wrapped as a CONST (Python's
  `3 + u` → `ADD(CONST(3), u)` via `__radd__`).
- `resolve(x, default)` is `UOpCtx::resolve_ord(a, CmpOrd, b, default)` — it
  builds the same comparison UOp Python would (`ge` = `NOT(lt)`), folds it,
  and returns `default` only when the result is not provably constant
  (`vmin != vmax`). **Do not** match on `std::cmp::Ordering` for six-way
  comparisons: its three variants make extra match arms silently bind as
  irrefutable variables (this bit us once — hence `CmpOrd`).
- PAD/SHRINK shape validation mirrors Python exactly: `resolve(sz >= 0) &&
  resolve(0 <= o) && resolve(o + sz <= dim)` — inclusive lower bounds
  (`SHRINK` from `0` is legal).
- Movement ops carry their shape args as **src consts** in v0.14
  (`RESHAPE(x, STACK-or-CONST)`, `PAD(x, starts, ends)`), not as op args;
  `PERMUTE`/`FLIP` keep the tuple as the op arg (`FLIP`'s is a tuple of bools).
