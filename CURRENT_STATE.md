# CURRENT_STATE.md

Status of the tinygrad → Rust port. See also `NNImpl.md` (nn-specific) and `PLAN.md`
(codegen-rule parity). Reference tree: `../tinygrad/tinygrad/`.

## 1. What we have right now

### Core graph engine (mature)
- `UOpCtx` graph (`src/uop/`): interned UOps, `UopId(u32)` (Copy), shapes, dtypes,
  `Ops` enum, `Arg`, `SVal`/`SInt`, const folding, `graph_rewrite`/`UPat`/`PatternMatcher`,
  min/max range analysis, `substitute`.
- Rewrite/simplify engine: `src/codegen/` (simplify, postrange, coalesce, gater, linearize,
  pipeline `full_rewrite_to_sink`), `src/schedule/` (rangeify, sched, indexing, memory).
- Backends/runtime (`src/runtime/`, `src/renderer/`): INTERP always on; CL behind `backend_cl`.
  `interp::{exec_kernel, exec_launch, from_storage_scalar, to_storage_scalar}`.
- Realize/execution (`src/engine/realize.rs`): `realize_tensor(...) -> Buffer`,
  `run_linear`, `tensor_values(...) -> Vec<f64>` (copies out a buffer and decodes each element
  via `from_storage_scalar`). **The eval path exists** — it is how the golden harness runs.

### Tensor front-end (`src/tensor/`)
- `Tensor { uop: UopId }` — now a `Copy` handle (mod.rs:64). **Design pivot (see §3d):** `grad`
  moved to `UOpCtx.grads` (`HashMap<UopId,UopId>`, `ctx.{set_grad,grad,clear_grads}` +
  `Tensor::grad(&self, ctx)`), and `is_param` was **dropped** — trainability is a property of the
  `Module` tree, not the tensor. `TensorAddArg` with `From<Tensor>` / `From<&Tensor>` / `From<f64>` /
  `From<i64>`.
- Movement (`movement.rs`): reshape, expand, unsqueeze, pad, permute, flip, transpose, squeeze,
  flatten, getitem, shrink/shrink_to, `pool`.
- Elementwise (`elementwise.rs`): add/sub/mul/div(neg·mul)/neg, maximum/minimum, exp2/log2/sqrt/
  sin/trunc/ceil/floor/reciprocal, binop/binop_c, `where`.
- Reduce (`reduce.rs`): sum/prod/max/mean/var/cumsum.
- Creation (`creation.rs`): `from_slice<T: SliceElem>` (dtype now **inferred from the generic
  width**: `&[i16]→INT16`, `&[u32]→UINT32`, `&[f32]→FLOAT32`, …; explicit `Some(dtype)` overrides
  and inserts a graph cast; native-width packing is exact, no `f64`), `from_bytes`, `empty`,
  `new_buffer`, `full_const`, `zeros/ones`, `arange`. `rand`/`uniform` now delegate to the faithful
  Threefry RNG in `tensor::rand` (see §3c) — the old LCG stub is gone.
- **Typed marshalling (DONE):** `Tensor::{realize, to_bytes, to_list::<T>, item::<T>}` + the
  `ReadDtype` trait (`dtype/mod.rs`); `engine::realize::tensor_bytes` returns native-dtype bytes.
  No `f64` round-trip; cross-dtype reads insert a graph `cast`, weak dtypes panic.
- NN layers (`nn.rs`): `Linear`, `BatchNorm`, `LayerNorm`, `GroupNorm`, `Conv2d` (**forward is a
  stub**), activations (relu/gelu/softmax/log_softmax/sigmoid/tanh/sin/cos/abs/sqrt/rsqrt).
- **`Module` trait (new):** `named_params(&self) -> Box<dyn Iterator<Item=(String, Tensor)>>` + a
  defaulted `params()` + a `scoped(prefix, iter)` helper for path-prefixing child modules
  (safetensors-ready naming, e.g. `"l1.weight"`). `Linear` implements it (weight, optional bias).
  Names live in the trait, never on `Tensor`. See §3d.

### Tests today
- `tests/backend_harness.rs` + `tests/golden.rs` (~90 goldens): ALU battery, reductions, casts,
  full dtype battery — asserted against tinygrad-interp goldens. This covers **UOp/engine-level
  op correctness**, driven through `tensor_values` (i.e. realize is exercised), but there are
  **no Tensor-level `tolist`/`item` assertions** and **no autograd tests**.
- `tests/marshal.rs` (new): typed marshalling — dtype inference in `from_slice`, `to_bytes`
  native-width storage, `to_list::<T>` round-trip + cross-dtype casts, `item::<T>`, and the
  `item()` multi-element panic.
- `tests/m2_smoke.rs`: graph construction/shape/rewrite/getitem (symbolic only).

### Build status
- `cargo build` and `cargo test` are green (**65 passed, 0 failed, 5 ignored**). The optimizer +
  `add_local_buffer` STAGE lowering landed cleanly; the XOR end-to-end training test is green
  (`tests/nn_xor.rs::xor_trains_to_convergence`, 2→4→1 ReLU) (§3f/§4-6).

## 2. What is missing

### Realize / data marshalling (no user-facing API yet)
- No `Tensor::realize`, `item`, `tolist`, `to_bytes` — the engine (`tensor_values`,
  `realize_tensor`) is only reachable from the harness, not from `Tensor`.
- `from_slice` dtype inference is lossy: all int widths map to `default_int`
  (`creation.rs:308`), so `&[i16]` does **not** become an int16 tensor. Needs precise
  `SliceElem → DTypeId` per width.
- Everything round-trips through `f64` (`tensor_values`); no typed decode.

### Autograd (engine DONE — see §3e)
- `Tensor::{gradient, backward, detach}` + `compute_gradient`/`grad_of_op`/`reduce_gradient`/
  `_deepwalk` are implemented (`src/tensor/grad.rs`) and gradchecked (`tests/grad.rs` 6/6).
  No `requires_grad`/`no_grad` context (tinygrad decides trainability from the `Module` tree; grads
  are opt-in by passing `targets`). `detach` is the `Ops::Detach` sugar.

### NN gaps
- `Conv2d::forward` is a stub (`nn.rs`) — full conv2d (pad → pool → reshape/permute → mul →
  sum → bias) unported (`tinygrad/mixin/op.py` conv2d).
- BatchNorm/LayerNorm/GroupNorm running-stat updates are stubbed/simplified.
- No `nn/optim.py` (SGD, Adam). `nn/state.py` param fetch is superseded by the `nn::Module` trait
  (§3d) — `named_params`/`params`/`scoped` replace reflection-style traversal; `init_weight`/`fetch`
  (safetensors) still to do.

### Tests we should have but don't
- Tensor-level op tests vs `test/backend/test_ops.py` (unary/binary/ternary/reduce/
  cast/dtype) driven through `to_list`/`item`.
- Autograd tests vs `test/unit/test_gradient.py` (+ `test_winograd.py` later).
- NN tests vs `test/backend/test_nn.py` and `test/backend/test_optim.py`.

## 3. Typed data marshalling — IMPLEMENTED

No `f64` round-trip; reads reinterpret native-dtype bytes, cross-dtype reads insert a graph cast.

### Output (tensor → Rust) — `src/tensor/mod.rs`
- `Tensor::to_bytes(&self, ctx) -> Vec<u8>` — native-dtype storage bytes via
  `engine::realize::tensor_bytes` (realize + `copyout`).
- `Tensor::to_list<T: ReadDtype>(&self, ctx) -> Vec<T>` — graph-casts to `T::DTYPE` (no-op if
  equal), realizes, reinterprets little-endian bytes as `[T]`. Panics on weak dtypes.
- `Tensor::item<T: ReadDtype>(&self, ctx) -> T` — asserts `numel==1`, then one `to_list` element.
- `Tensor::realize(&self, ctx) -> &Tensor` — forces compute, returns `&self`.
- `*_with` variants (`to_bytes_with`/`to_list_with`) take a caller-owned `&mut BufTable` +
  `&mut ProgCache` so a caller (e.g. the harness) can reuse realized buffers across reads.

### Input (Rust → tensor) — `src/tensor/creation.rs`
- `SliceElem::default_dtype()` is width-precise (`i8→INT8 … f64→FLOAT64`, `bool→BOOL`), so
  `from_slice` infers the dtype from the generic. `SliceElem::encode_native()` packs exact
  little-endian bytes; an explicit `Some(dtype)` of a different width falls back to the
  `to_storage_scalar` f64 bridge.

### Traits
- `ReadDtype` (`src/dtype/mod.rs`): `const DTYPE: DTypeId; fn from_bytes_le(&[u8]) -> Self;` —
  implemented for `i8/i16/i32/i64/u8/u16/u32/u64/f32/f64/bool`. `f16`/`bf16`/`fp8` tensors are read
  by casting to a wider type (no native Rust half type yet).

## 3b. Milestone 2 — tensor-op tests (`tests/ops.rs`) — DONE

Suite drives op → `to_list`/`item`. Current: **9/9 green** (no regressions: golden 7, marshal 6,
backend_harness 12 passed / 1 ignored).

Green: `binary_arithmetic`, `neg_abs_and_bounds`, `unary_pow_log`, `unary_misc`,
`casts_across_dtypes`, `activations` (relu/gelu/softmax), `reductions` (incl. `cumsum`),
`movement_ops` (permute/transpose/flip/flatten read-out), `matmul_ops` (2D@2D `dot`).

Op bugs found & FIXED (port vs tinygrad):
- `elementwise::log` did `log2 / ln2`; tinygrad is `log2 * ln2`. Now multiply. (`log(2)` was 1.4427, now 0.6931.)
- `Tensor::dot` reshape math wrong for N-D `w` (2D@2D produced a bad `w` reshape → size-mismatch panic).
  Rewritten to mirror tinygrad `dot` (`axis_w=-min(ndim,2)`, `x.reshape(...,[1]*rep,N)`,
  `w.reshape(*[:-2],[1]*rep,*w[axis_w:]).transpose(-1,axis_w)`, `(x*w).sum(-1)`). 2x3@3x2 verified.
- `elementwise::tanh` computed `exp2(2x·x/ln2)` = `2x²/ln2` (multiplied by `x` twice). Now `2x/ln2`
  (was wrong-sign for negatives).
- `runtime::buffer::buffer_of` now treats `Permute/Pad/Flip/Expand` like the other view ops (storage =
  base buffer; mapping applied via kernel index math).
- `to_bytes_with` now forces a materializing copy (contiguous + `copy_to_device`) so read-out is in
  logical C order for views.

Formerly RED — now GREEN (resolved during M6/M8 hardening, not by further ops-suite surgery):
- `movement_ops` (permute/transpose/flip/flatten read-out): the materializing copy (`to_bytes_with`)
  now lowers a permuted-view copy to a realized STORE correctly; `buffer_of` handles the `Ops::Copy`.
- `reductions` → `cumsum`: root cause was `_broadcast_to` building the final perm by block
  (expand-at then kept) instead of original dim order, so a middle/trailing size-1 broadcast landed
  at dim 0 and corrupted the pool's `repeat` (cumsum collapsed the batch dim). Fixed in
  `tensor/movement.rs` (commit `5c3abd5`); regression test `expand_all_positions`.
- `matmul_ops` (`postrange` `axis - deleted_shape` underflow): resolved by the `dot` rewrite above;
  no postrange surgery needed.

## 3c. Milestone — RNG: faithful Threefry (`tensor/rand.rs`) — DONE

Reproduces tinygrad's `Tensor.manual_seed(...)` bit-for-bit. Ported as a **host evaluator**
(`src/tensor/rand.rs`), not a UOp graph: the INTERP backend stores every value as `f64` and cannot
`bitcast` nor carry a `uint64` bit pattern, so the graph route panics (`interp op Bitcast not
ported`). The graph lowering (`codegen::decomp::threefry2x32`) is still wired into the pipeline's
"late decompositions" stage for hardware backends.

- State (thread-local, mirrors tinygrad class-level): `seed`, per-device `device_seeds`, counters.
- `manual_seed`, `_next_counter` (device seed = `sha256(index as 4-byte BE)` low-32 + `seed`),
  `threefry2x32` (5×4 rounds, rotations `[[13,15,26,6],[17,29,16,24]]`, `ks=[k1,kx,k0]`),
  `random_bits` (counter-derived key, low-then-high concatenation), `bits_to_rand` (f32/f64).
- API: `Tensor::{rand, uniform, randn, randint, manual_seed, kaiming_uniform, glorot_uniform,
  scaled_uniform}`. `uniform` draws bits in the **default float** dtype then `.cast`s to the target
  between scale and shift (matches `rand.py:158`); `randn` = polar Box–Muller over a single
  `rand([2,*shape], float32)`; `randint` = `trunc((high-low)*rand)+low`.
- SHA-256 (device seeding only) — bug fixed: `K[21]` was `0x4a748aa0`, correct is `0x4a7484aa`.
  `sha_tests` assert `""`/`"abc"` vectors.
- `nn::Linear::new` init corrected to tinygrad: `kaiming_uniform(a=sqrt(5))` weight `[out,in]`,
  `uniform(in, ±1/sqrt(in))` bias.
- Goldens: `tests/rand.rs` **8/8 green** (`manual_seed` 42/7/99/100): `rand` sequence + counter
  continuity, `uniform`, `kaiming_uniform`, `randn` (tol 1e-4 — Box–Muller transcendental), `randn`
  continuity, `rand→randn` stream continuity, `randint`.

## 3d. Design pivot: `Tensor` is `Copy`, grads live in the ctx, params via a trait — DONE

Rationale (agreed before coding): all mutation already funnels through `UOpCtx` (the arena), so a
`Tensor` need not own mutable state. Storing `grad` on `Tensor` (`Option<Box<Tensor>>`) forced
`&mut Tensor` to run `backward`, which fights the value-handle model. Resolution:

- **`Tensor { uop }`, `Copy`.** Dropped `grad` and `is_param` fields, and `Tensor::is_param_`.
  `UopId` is `Copy+Hash+Eq`, so `Tensor` derives `Clone, Copy, Debug, PartialEq, Eq, Hash`.
  **Do not derive `Ord`/`PartialOrd`** — the by-value `Ord::max`/`min` shadow the reduce
  `Tensor::{max,min}(&self, …)` and break method resolution (bit us during the port).
- **Grad side-table on `UOpCtx`:** `grads: HashMap<UopId,UopId>` (a gradient is itself a `UopId`).
  `ctx.set_grad(target, grad)` / `ctx.grad(target)` / `ctx.clear_grads()`, surfaced as
  `Tensor::grad(&self, ctx) -> Option<Tensor>`. `backward` can now fill grads with no `&mut Tensor`.
- **`is_param` dropped.** "Is trainable" is decided by the `Module` tree (what the optimizer walks),
  not a per-tensor flag. Buffer-style state that must update in-place (BatchNorm running mean/var)
  is handled by how the module *assigns* it, not by a flag.
- **`nn::Module` trait** (`src/tensor/nn.rs`): `named_params(&self) -> Box<dyn Iterator<Item=(String,
  Tensor)>>` (child-local names), defaulted `params()` (drops names), and `scoped(prefix, iter)`
  (`"l1.weight"`). `Box<dyn Iterator>` chosen over `impl Iterator` on the required method so
  `Module` stays object-safe and default `params()` can wrap it; a derive macro can emit
  `named_params` later. Names are **not** stored on `Tensor` — they are assembled by the trait, which
  is what safetensors load/save will key on.

## 3e. Autograd engine: faithful `compute_gradient` (`src/tensor/grad.rs`) — DONE

Ported tinygrad `mixin/gradient.py` (`compute_gradient`, `_deepwalk`, `reduce_gradient`, and the
`pm_gradient` rules) onto the UOp graph. tinygrad expresses `pm_gradient` as a `PatternMatcher`, but
each rule emits **one gradient per source** (a `Vec<Option<UopId>>`), and every rule is keyed on the
node's `op`, so we dispatch with an explicit `match` (`grad_of_op`) instead — rule bodies are ported
one-for-one. The `FUNCTION`/`CALL`/`GETTUPLE` machinery (`call_gradient`, `_compact_params`) is
intentionally omitted: eager `Tensor` graphs never build those nodes.

- `grad_of_op` covers: `CAST`, `RECIPROCAL`, `SIN`, `LOG2`, `EXP2`, `SQRT`, `TRUNC`, `CMPLT`, `CMPNE`,
  `ADD`, `SUB`, `MUL`, `MAX`, `POW`, `WHERE`, `REDUCE`, `CONTIGUOUS`, `CONTIGUOUS_BACKWARD`,
  `RESHAPE`, `EXPAND`, `PERMUTE`, `FLIP`, `BITCAST`. Unsupported ops (PAD/SHRINK/STACK/COPY/TUPLE)
  return `None` → `compute_gradient` raises a `TGERR:` (XOR never reaches them).
- `reduce_gradient` handles `ADD` (broadcast back to source shape), `MAX` (masked ratio), `MUL`
  (`prod/x_j` with a zero-count guard) — matches `gradient.py:9-15`.
- `compute_gradient` reverse-walks the deep-walked graph, and applies the **shape-sum glue**: when an
  edge's gradient shape differs from its source (e.g. an `EXPAND`ed bias), it `cast→sum(broadcast_axes)
  →reshape→cast` back to the source shape before accumulating into `grads` (`gradient.py` line for
  `k.shape != v.shape`). This is what makes a broadcast bias/dot gradient correct.
- `argsort` (tinygrad `helpers.py`) added locally for the PERMUTE grad (inverse permutation).
- Entry points (`impl Tensor`): `gradient(targets, gradient)` (seed `root_grad` = `const_like(self,1)`),
  `backward(targets, gradient)` (accumulates into `ctx.grads` via `set_grad`), `detach()` (`Ops::Detach`).

**Op-graph reality (Rust ≠ tinygrad decomposition, so grad rules match the Rust builder):**
`sub` = `ADD(x, Mul(b,-1))` (no `NEG` node); `neg` = `Mul(-1)`; but `tanh`/`elementwise::tanh` emit a
real `Ops::Sub` → a `SUB` grad rule (`[g, -g]`) was needed. `sigmoid`/`exp` insert `CAST`s. `mean` =
`Reduce(ADD)` + `Mul(reciprocal)`. `dot` = `reshape`/`permute`/`mul`/`sum`. All covered.

Tests: `tests/grad.rs` **6/6 green** — `grad_sum_square`, `grad_linear_affine`, `grad_bias_broadcast`
(broadcast-sum), `grad_sigmoid`, `grad_matmul` (dual grad wrt `a`/`w`), `grad_mse_like` (mean+sub+square).

## 3f. Realize memory + local-memory `STAGE` lowering — PARTIAL

- **Exponential-realize fix (DONE, `c2b8d50`):** `Graph::to_flat` (`src/uop/flat.rs`) previously
  serialized *every* arena node, so each compiled `Program`/`Binary` node (whose `Arg::Bytes(lib)`
  embedded the *previous* arena JSON) got re-serialized on every realize → exponential
  text/memory/time growth. Now it serializes only the toposort set reachable from `self.roots`
  (index-remapped; falls back to all nodes when roots is empty). `micro_repeated_realize` confirms
  per-iter realize is now ~5ms flat (was exponential). **Caveat:** this kills the *time/size blow-up*
  but the arena node count still grows ~linearly across repeated realizes — no refcount/GC yet
  (`Tensor` is `Copy`, so Drop-based liveness isn't available; an `Rc`/`Weak` arena is the open path).
- **`add_local_buffer` (DONE, `11ab94a`):** implemented `add_local_buffer` + `pm_add_local_buffers`
  in `src/schedule/rangeify.rs`, wired into the codegen `"add local buffers"` stage
  (`src/codegen/pipeline.rs`), which now receives a slot counter (`CtxVal::Cnt`). Lowers remaining
  local `STAGE`s into a placeholder `Buffer` + `index/store/end/after` so `STAGE` no longer reaches
  the interp executor (was `TGERR: interp op Stage not ported`).
- **Still pending (XOR WIP):** after `STAGE` lowering, a `Reshape` on the local-buffer *load* path
  survives into interp (`TGERR: interp op Reshape not ported`). Next: fold movement/`Reshape` into
  the coalesced index on that path (and audit any local-memory `BARRIER` handling). Tracked in §4/6.

## 4. Next milestones (in order)

1. ~~**Data marshalling.**~~ **DONE** (`to_bytes`/`to_list<T>`/`item<T>`/`realize` + `ReadDtype`;
   width-precise `from_slice` inference; `tensor_bytes`; `tests/marshal.rs`).
2. ~~**Tensor-op tests (tensor level).**~~ **DONE** (`tests/ops.rs` 9/9 green; op→`to_list`/`item`
   families for binary/unary/reduce/cast/activation/movement/matmul). Movement/broadcast gaps found
   while fixing (`expand`/`broadcast_to` axis order) got regression coverage in
   `backend_harness.rs` (`pad_fill_repro`, `expand_all_positions`).
3. ~~**Autograd engine.**~~ **DONE** (`src/tensor/grad.rs`; faithful `compute_gradient` + `match`-dispatch
   `grad_of_op` + `reduce_gradient` + `_deepwalk`; shape-sum glue; `Tensor::{gradient,backward,detach}`).
   See §3e. PAD/SHRINK/STACK/COPY grad rules intentionally deferred (unreachable for XOR).
4. ~~**Backward gradchecks.**~~ **DONE** (`tests/grad.rs` 6/6: sum-square, affine, bias broadcast-sum,
   sigmoid, matmul dual-grad, MSE). A fuller `test/unit/test_gradient.py` port can expand coverage.
 5. ~~**NN + optimizer.**~~ **DONE** (`src/tensor/optim.rs`: `Optim` trait + `SGD::{step,zero_grad}`
    walking `Module::named_params()` and applying `w -= lr * g` as an in-graph `assign`; module wired
    into `tensor/mod`). `Linear::new` bias shape fixed (uniform over `out_features`). `Conv2d::forward`
    + `test_nn.py` still deferred (XOR needs only `Linear`).
 6. **XOR end-to-end golden** *(WIP — `tests/nn_xor.rs::xor_trains_to_convergence`, `#[ignore]`d)*.
    `2→2→1` MLP (sigmoid, MSE, SGD `lr=0.5`) is wired and runs forward+backward+step, but the
    backward/local-memory **reduction kernels** still need codegen/interp work: `STAGE` lowering is
    done (§3f) but movement/`Reshape` folding on the local-buffer load path is pending. See §3f.

### Done out of band
- **RNG (Threefry)** — see §3c. `tests/rand.rs` 8/8 green; `nn::Linear` init now faithful.

## 5. Open questions (resolved during milestone 1)
- **Buffer handle:** kept call-scoped. The `to_bytes`/`to_list`/`item` conveniences allocate fresh
  `BufTable`/`ProgCache` per call; `*_with` variants accept caller-owned tables for buffer reuse.
  A persistent `Session`/`Runner` (tinygrad-style buffers-on-uop) is deferred to milestone 3
  (backward/optim need mutation + persistent buffers).
- **f16/bf16/fp8 read:** cast to `f32`/`f64` via `to_list::<f32>()`; no native half Rust type yet.
- **Shape of `to_list`:** flat row-major `Vec` (matches `tensor_values`); nested N-D list deferred
  until a test needs it.
