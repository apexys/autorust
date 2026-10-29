# CURRENT_STATE.md — tinygrad v0.14.0 → Rust (`autorust`): plan & status

Single source of truth for **what the port is**, **what is done**, and **what is next**.
Merged from the former `PLAN.md` / `CURRENT_STATE.md` / `NNImpl.md`.

- Reference oracle: the `tinygrad/` tree in this repo (v0.14.0, the post-"rangeify"
  refactor with `mixin/`). Python is the oracle, **never** a runtime dependency of the
  final artifact.
- `INDEXING.md` — the `getitem` index-DSL reference (kept separate).
- §5 holds the codegen rule-parity table.

---

## 1. Ground rules

- **MVP** = a native Rust `Tensor` that builds UOp graphs, schedules them, codegens
  kernels, and executes them on a Rust CPU backend — verified against real tinygrad by
  differential tests for every intermediate artifact, not just final numbers.
- **Backend-ready by construction**: port tinygrad's pluggable backend seams
  (`Renderer`, `Compiler`, `Allocator`, `Program`, `Compiled` as Rust traits, plus cstyle
  rendering and grid-dim codegen) so an OpenCL backend is a pure additive impl. Nothing
  below the trait boundary may assume CPU.
- **Dependency policy: std-only Rust + nanoserde** for the core. No PyO3/maturin, no serde/syn
  ecosystem, no libc/sha2 crates; nanoserde (single micro-crate) for the JSON graph
  interchange. **Optional backend deps are the one sanctioned exception**, each behind its
  own feature: `opencl3` (`backend_cl`), `cudarc` (`backend_cuda`), and `rayon` + `bytemuck`
  (`backend_rayon`, the multithreaded CPU backend, §9.8) — `--no-default-features` stays
  std-only. Two FFI faces only: (a) plain cargo binaries for batch oracle work over
  JSON graph files (one-shot — graph building, codegen diffs, CI), (b) a tiny `cdylib`
  C ABI (`#[no_mangle] extern "C"`, caller-owned buffers, ~5 functions) loaded from Python
  via stdlib `ctypes` for the per-kernel hot path (`ops_rust.py`) — bytes only, no JSON on
  the exec path. **Non-finite floats** (`NaN`/`±Inf` CONST args) have no JSON
  representation — nanoserde *and* miniserde both emit bare `NaN`/`inf` tokens they cannot
  parse back — so they travel as an explicit `Arg::NonFinite` variant. Internal hashes use
  std `DefaultHasher` (exact Python-`key` equality stays bridge-side). The OpenCL backend
  follows the same rule: hand-written `extern "C"` bindings loading `libOpenCL`, mirroring
  how tinygrad itself drives native code (precedent: `Compiler.compile_server`,
  device.py:314-320).
- **Sequencing decision: frontend-first.** Track A drives M1→M4; the executor does not
  block anything. The only executor work before M5 is the M0 JSON round-trip + cdylib
  smoke test; the `exec_alu` table is pulled into M2 because const folding needs it.
- Rust never imports Python at runtime after the final milestone. All Python contact is
  via a dev/test bridge that lives under `autorust/py/`.

## 2. Inventory of moving parts

Sizes are LOC of the v0.14.0 tree. Decision key: **PORT** (must exist in Rust),
**POLYFILL** (bridge to real tinygrad during development), **SKIP** (never in MVP).

| Layer | Files (v0.14.0) | LOC | Decision | Notes |
|---|---|---|---|---|
| dtypes | `dtype.py` | 323 | PORT (P1) | `DType`, `dtypes`, `AddrSpace`, truncation, `least_upper_dtype`, np-dtype maps (np only in tests) |
| helpers | `helpers.py` | 640 | PORT subset (P1) | `ContextVar`/env context (`getenv`, `Context`), `prod/cdiv/floordiv`, `Metadata`, `dedup`, `unwrap` |
| Ops enums | `uop/__init__.py` | 137 | PORT (P2) | `Ops`, `GroupOp`; freeze the v0.14.0 set |
| UOp core | `uop/ops.py` | 1788 | PORT (P2) | hash-consed immutable DAG node, recursive props (`key`, `shape`, `dtype`, `device`), `toposort`, `graph_rewrite`, `Variable`/`sint`, `CALL`/`FUNCTION` plumbing |
| Pattern engine | `uop/upat.py` | 171 | PORT (P2) | `UPat` DSL + `PatternMatcher` (skip pickling) |
| symbolic | `uop/symbolic.py`, `uop/divandmod.py` | 586 | PORT (P2) | `sym` rewrites, floordiv/mod semantics |
| shape system | `uop/movement.py`, shape logic in `uop/ops.py` | ~26+ | PORT (P2) | v0.14 has no ShapeTracker class; movement ops are UOps rewritten by `schedule/rangeify.pm_mops` |
| uop utils | `uop/render.py`, `uop/weak.py`, `uop/validate.py`, `uop/spec.py` | 505 | PORT (P2/P4) | `pretty_print` (repr oracle), weak-dtype lowering, spec verification |
| Tensor front-end | `tensor.py` + `mixin/{op,elementwise,movement,reduce,creation,dtype,rand,gradient}.py` | ~5900 | PORT (P3) | mechanical but huge; `gradient.py`/`rand.py` late (P6) |
| scheduler | `schedule/{__init__,rangeify,memory}.py` | 886 | PORT (P4) | `get_kernel_graph`, `create_schedule`, `memory_plan_rewrite` |
| indexing/rangeify | `schedule/indexing.py` | 331 | PORT (M4) | **not optional**: `rangeify.py` imports `run_rangeify`/`apply_movement_op`; `INDEX` machinery comes along. Only fancy `__getitem__` corpus coverage defers |
| multi-device | `schedule/{multi,allreduce}.py` | 381 | SKIP | no `UNSHARD`/`MSELECT`/`MSTACK`, no sharding |
| codegen | `codegen/{__init__,simplify,gpudims}.py` + `codegen/{decomp,late,opt}/*` | ~2200 | PORT (M5) | `to_program` pipeline: dtype decomp, transcendental, rangeify/mops, linearizer, regalloc, coalesce. `apply_opts`/postrange **parameterized** (opts are a rewrite input). `expand_wmma` ported with empty per-renderer `tensor_cores` lists; SKIP BEAM driver + TC layouts |
| renderer | `renderer/{__init__,cstyle}.py` | 690 | PORT (P5) | `Renderer` trait, `Estimates`, `Target`, **cstyle text rendering**. SKIP `llvmir/ptx/wgsl/nir/amd` |
| CPU ISA JIT | `renderer/isa/x86.py`, `runtime/support/{compiler_cpu,elf}.py` | ~1900 | SKIP | v0.14 CPU compiles to x86 ELF; we substitute the interpreter. Revisit if perf matters |
| execution ref | `runtime/ops_python.py` | 249 | PORT (P5) | the uop interpreter — "the living definition of uops"; port verbatim semantics |
| devices/runtime | `device.py`, `runtime/ops_*.py`, `runtime/support/{hcq,hcq2,...}` | ~10k+ | PORT abstraction only (P6) | `Buffer`/`Allocator`/`Compiler`/`Program`/`Compiled` become Rust **traits**. SKIP concrete GPU devices + HCQ/HCQ2 + graph capture |
| engine | `engine/{realize,jit}.py`, `engine/worker.py` | 677 | PORT subset (P6) | `run_linear`/`compile_linear`/`exec_*`, `TinyJit` stretch, SKIP worker pool |
| rng / autograd | `mixin/rand.py`, `mixin/gradient.py` | 462 | PORT (P6) | after native execution works |
| nn core | `nn/__init__.py` (420), `nn/state.py` (352), `nn/optim.py` (179) | ~950 | **PORT (M7e)** | the M7 integrator: exercises custom_kernel (Embedding), rand, gradient, conv ops together |
| llm / viz / nn extras | `llm/`, `viz/`, `nn/torch.py`, disk cache, process replay | huge | SKIP | `nn/onnx.py` is **ported** (T0/T1) in `src/onnx/`; torch import stays out |
| `fetch` / `Tensor.from_url` | **DESCOPED** (2026-10): weight/data download is left to the user; `examples/mnist.rs` shells out to `curl` once | — | **DESCOPED** | no opaque download API in the library |

MVP port budget ≈ **10–11k lines of Python** (of ~19k core), plus tests.

## 3. The seams (where we can cut)

The UOp is the universal currency, so every seam is a UOp-graph hand-off; one interchange
format exploits all of them:

- **Seam A — Tensor→UOp** (top-down): Rust builds the graph, Python schedules/codegens/executes it.
- **Seam B — SINK→LINEAR** (schedule): Rust schedules Python- or Rust-built tensor graphs.
- **Seam C — PROGRAM→kernels** (codegen): Rust linearizes kernel ASTs; Python or Rust executes.
- **Seam D — kernels→bytes** (execution, bottom-up): Rust executes uop lists produced by Python's codegen.

Interchange format: **JSON** via nanoserde — a post-order node list
(`{op, dtype, srcs, arg}` with `arg` as a tagged union, plus a dtype name table and root
indices), converters both directions (`autorust/py/bridge.py` ↔ `autorust/src/uop/flat.rs`).
Human-diffable by design; parse cost is per-compile, not per-launch. Non-finite floats use
an explicit `NonFinite` variant (strict JSON). This is *not* pickle. Note
`UOp.key = sha256(str((op,dtype,arg)) + child keys)` (uop/ops.py:264) — we do **not**
reimplement Python's `str()` formatting in Rust; key comparison happens Python-side on
graphs shipped through the bridge. Same for `pretty_print` reprs.

Two complementary strategies fall out of the seams:

1. **Top-down (primary)** — port dtype → UOp core → Tensor front-end first. Every phase
   ends with a graph we can hand to real tinygrad to realize. The oracle is strong (graph
   `key` equality, then numeric equality) and the biggest mechanical chunk (Tensor) is done
   early while the Rust UOp design is still fluid.
2. **Bottom-up (secondary, cheap)** — a Rust interpreter. Device auto-discovers
   `runtime/ops_*.py` modules by filename, so a `RustDevice(Compiled)` dropped into the
   tinygrad tree is the official extension mechanism; Python's codegen produces kernels,
   the bridge serializes, Rust executes. Validates the cdylib/ctypes plumbing, the flat
   format, and the `exec_alu` table early.

The tracks converge: once the front-end + scheduler + codegen are native (track A) and the
interpreter is native (track B), `Tensor.realize()` stops calling Python at all.

## 4. Milestones

Each milestone has a **gate**: tests that must pass before moving on. Differential tests
live in `autorust/py/` (driven against `tinygrad/`) + Rust unit tests in `autorust/src`.

### M0 — scaffolding + bridge (both tracks) — DONE
- `autorust` crate layout: `src/uop/`, `src/dtype/`, `src/tensor/`, `src/schedule/`,
  `src/codegen/`, `src/exec/`; `crate-type = ["lib", "cdylib"]`.
- **Two Rust faces, std + nanoserde only**:
  - cargo binaries (`src/bin/oracle_*.rs`) for batch operations over JSON graph files:
    `rebuild`, `rewrite <pm>` (JSON in, rewritten JSON out), later `schedule`, `codegen`,
    `render`. Driven by pytest; no FFI.
  - cdylib C ABI: `exec_kernel(uops_ptr, len, bufs, dims, vals) -> ret` +
    `autorust_free(ptr)`; loaded via `ctypes.CDLL`.
- `autorust/py/bridge.py`: `uop_to_json`, `json_to_uop`, `realize_in_tg`, key/repr oracles.
- **Gate**: `UOp → JSON → Rust → JSON → UOp` preserves `key` and `repr` (31-case corpus);
  cdylib smoke test executes a trivial hand-written kernel. ✅

### M1 — dtypes + helpers — DONE
- `DType` (struct: `scalar`, `count`, `itemsize`, `fmt`, flags), `dtypes::*`, `AddrSpace`,
  truncation/cast tables, `least_upper_dtype`, `ConstType`. No numpy in the lib.
- **Gate**: differential `least_upper_dtype` over all pairs, cast/truncate tables vs
  Python; `test/null/test_dtype.py` core cases. ✅

### M2 — UOp core (critical) — DONE
- Hash-consed `UOp` (arena + `UopId`; see §9 for the Rc/GC follow-up). Fields
  `op, dtype, src, arg, tag` matching v0.14.
- Recursive properties as cached fixpoints: `key` (Rust-side hash), `shape`, `dtype`,
  `device`, `vmin`/`vmax` + `resolve` (symbolic shapes are core: the scheduler's
  `broadcast_axes`, gates and the memory planner are `sint`-typed throughout).
- `UPat` + `PatternMatcher` with Python's application semantics (pattern order,
  bottom_up/top_down, ctx threading, `walk`/`enter_calls`), `graph_rewrite` to fixpoint.
- `uop/symbolic.py`, `uop/divandmod.py`, `uop/movement.py`, `exec_alu`/`python_alu` table.
- `CALL`/`FUNCTION`/`PARAM`/`TUPLE`/`GETTUPLE` with dtype/shape rules from day one.
- **Gate**: ported `test_uop_symbolic.py`/`test_pattern_matcher.py`/`test_uop_graph.py`
  cores; differential rewrite (`sym`, `mop_cleanup`, `symbolic_simple`) = 13k random graphs
  per PM, ~0.08% residual (documented NaN-const class); byte-equal `pretty_print`. ✅

### M3 — Tensor front-end (Seam A polyfill begins) — DONE
- `Tensor` = `uop: UopId` wrapper; `mixin/{creation,dtype,elementwise,movement,reduce}` and
  the dispatch core of `mixin/op.py`; `tensor.py` minus precompile/numpy internals.
- The mixin chain subclasses **UOp itself**; all sugar is UOp-level and returns UOps.
  Port order: `tensor/core.rs` → `movement.rs` (`_mop`, shape args as SRC consts) →
  `creation.rs` → `elementwise.rs` (`_broadcasted`/`_binop`) → `reduce.rs` (`_rop`) →
  `dtype.rs` → `callify.rs`.
- **Callify is part of M3**: v0.14 routes every graph through
  `transform_to_call` → `CALL(FUNCTION(body), params)`. Ported the FUNCTION-free subset.
- MVP op subset: const/ones/zeros/arange/full/eye, ALU binops/unops incl. `where`, casts,
  reshape/permute/expand/slice/flip/pad/cat, sum/max/min/mean/var, matmul, comparisons.
- **Gate**: `test/null/test_tensor.py` + `test_const_folding.py` core cases through the
  bridge; **key-equality suite**: Rust-built `key` == Python's for each corpus program;
  symbolic smoke test. ✅

### M4 — scheduler (Seam B) — DONE
- `schedule/indexing.py` (`run_rangeify`, `IndexingContext`, realize-map,
  `apply_movement_op`) — **required**.
- `schedule/rangeify.py` (`get_kernel_graph`, `pm_mops`, `BufferizeOpts`) and
  `schedule/__init__.py` (`create_schedule`, `create_linear_with_vars`,
  `resolve_linear_call`). SKIP multi/allreduce.
- **Graph-level memory planner** (`schedule/memory.py`): buffer lifetime analysis over
  LINEAR then suballocation into per-lane arenas (`SHRINK`/`BITCAST` views of one big
  `int8` arena) + `TLSFAllocator`. `_can_plan` excludes CL/WEBGPU — OCL relies on the
  runtime `LRUAllocator`.
- **Gate**: Rust `LINEAR` == Python's (same kernel count, per-kernel AST keys as a
  multiset); Rust-scheduled graphs realize correctly via the bridge; memory-plan output
  matches. ✅ (`test_schedule_diff.py` 6/6)

### M5 — codegen (Seam C) + renderer + interpreter (Seam D) — DONE
- Ported `codegen/__init__.to_program` pipeline: dtype decomp, late rewrites, `gpudims.py`
  (`get_grouped_dims` → `SPECIAL` globals), rangeify/mops, `linearize` (control flow,
  END/IF), regalloc, coalesce, simplify, `apply_opts`/postrange.
- `renderer/__init__.py` as a `Renderer` trait + `cstyle.py` (C-family text incl. `SPECIAL`,
  BARRIER, LOCAL addressing, pointer arithmetic). The interp renderer renders to the flat
  uop-list; cstyle is diff-tested for the OpenCL backend.
- `codegen/interp.rs` executes the same linearized uops the renderer consumes
  (global/local loops ↔ `SPECIAL(g/l)`).
- **Gate**: per-kernel Rust linearized uops == Python's; cstyle byte-equal on a subset;
  numeric differential vs tinygrad CPU; `test_render_diff.py` byte-equal. ✅
- Key fixes: linearize heap must pop **largest** nkey first (Python pushes `-nkey`);
  `pm_mops`'s movement-on-INDEX pattern direction; `INDEX(src=(x,)) -> x` identity.

### M6 — native realize + device traits — DONE
- `src/runtime/buffer.rs`: `Buffer` (base + byte-offset views, `get_buf` windows,
  opaque-ctor path), `BufTable` = the engine `buffers` dict; `Node.initial_data` carries
  `Tensor(data)`'s fake-realize bytes.
- `src/runtime/device.rs`: `BackendAllocator` (`_alloc/_copyin/_copyout/_offset`) +
  `LruAllocator`, empty `Compiler`, `TinyElf`, `Prog` trait, `InterpProgram`, `Compiled`,
  thread-local `Device` registry.
- `src/engine/realize.rs`: `ProgramInfo` + `from_sink`, `to_program`
  (full_rewrite_to_sink → linearize → SOURCE → BINARY), `compile_linear`, `run_linear`,
  `Tensor` realize + flat readback.
- **KEY CONVENTION**: Python's `PythonProgram` iterates `global_size[::-1]` /
  `local_size[::-1]`; `InterpProgram` reverses `launch_dims` to match.
- Reductions hardening: `expand_reduce`, `pm_reduce_local`, `opt.arg == 0` = full extent,
  shift_to's `KernelOptError` caught in the heuristic.
- **Gate** `py/tests/test_native_exec.py`: end-to-end corpus runs with Python fully out of
  the loop; numerics match tinygrad; double-realize compiles exactly once. ✅

### M7 — user kernels + the rest of the Tensor (stretch) — PARTIAL
- **`@function` / `Tensor.custom_kernel`**: the machinery is native since M3/M4; this
  milestone activates the user-facing API + precompiled-CALL fast path. **Proof target:
  flash attention** (`extra/thunder/tiny/fa.py` slice). ⬜ for the fused-kernel DSL
  (`tk` + CALL); the **op-level FlashAttention algorithm is done** — see §8.31.
- `TinyJit` (capture LINEAR, replay with fresh buffers). ⬜
- autograd (`mixin/gradient.py`) — ✅ done early, see §8.4.
- `rand` (`mixin/rand.py`) — ✅ done early, see §8.3.
- half/bf16 emulation — ✅ DONE (§8.18): the cstyle/CL `extra_matcher`
  (`create_non_native_float_pats` + `pm_manual_bf16_cast`), OpenCL `as_*` bitcasts and
  bf16-const bit patterns, and sub-word `rand` are ported; `tests/bf16.rs` is a CL↔INTERP
  differential and a full bf16 CNN train step runs (`DTYPE=bfloat16`, ~2x slower than f32,
  so f32 stays the default).
- WMMA arm + first tensor-core layouts, BEAM search driver, DISK device. ⬜
  (BEAM tracked in §9.6; GPU-native features motivate the native-backend item there.)
- **nn core** (`nn/__init__.py` + `nn/state.py` + `nn/optim.py`): BatchNorm, Conv1d/2d,
  ConvTranspose1d/2d, Linear, GroupNorm, InstanceNorm, LayerNorm(+2d), RMSNorm, Embedding,
  LSTMCell; `State`; Optimizer + SGD/Adam(+W)/LARS/LAMB. **Proof: MNIST e2e** (MLP train
  step + Conv inference, weights via `State`, optimizer numerics diffed). **MNIST
  example + ignored e2e gate DONE (§8.16)**; all `nn/__init__.py` layers are now ported
  (§8.25), each gated by `tests/nn_layers.rs` vs tinygrad goldens. `State`/safetensors
  landed in §9.9; dataset download is descoped (examples use `curl`). ✅

### M8 — first additional backend: OpenCL — DONE
- `runtime/` + `renderer/` backend seams: `OpaqueT`/`BackendAllocator`/`Comp`/`Prog`/
  `UopRenderer` traits + `register_device`.
- `renderer/cstyle.rs` parameterized via a `CStyleLang` config (clang + opencl flavors).
- `runtime/ops_cl.rs`: CLDevice (compiler + LRU allocator + program); mirrors tinygrad's
  composition quirk (empty compiler seam, real CL compile inside `CLProgram`).
- Gates (`test_ocl_diff.py`, RTX 3060): OpenCL render byte-diff 4/4, native CL numerics 4/4,
  kernel-count parity 4/4. ✅
- M8.1 backend conformance harness (`tests/backend_harness.rs`): smoke, ALU battery,
  dtype battery, casts, goldens. Fixed interp bf16/f16/fp8 truncation, from/to storage
  scalar, unary/ternary ALU, store-at-dest-dtype, LOCAL-window shrink shape, cstyle
  `code_for_op` completion. ✅
- M8.2 open gaps: (a) weak-const commit / `pm_lower_weak` — **FIXED** (§8.17: comparisons
  must commit weak operands); (b) non-native-float decomp (done in M5, superceded);
  (c) arange realize — FIXED; (c') reduce-over-reduce — FIXED (rangeify `PCONTIG` realize
  arm; see §8.11); (d) sub-dword vectorized coalescing bitcasts; (e) XOR training gate — now
  ✅ (see §8.7); (f) cstyle `extra_matcher` for bf16 (`create_non_native_float_pats` +
  `pm_manual_bf16_cast`) — see §9.6.

---

## 5. Codegen rule parity (`full_rewrite_to_sink`)

Authoritative stage list: `codegen/__init__.py:294-393`. Parity of each Python PM with the
Rust pipeline (`src/codegen/pipeline.rs` `stages()`/`post_stages()`). ✅ = wired and firing,
⬜ = empty placeholder (no-op on corpus), ❌ = not ported.

| # | python stage (PM) | Rust | note |
|---|-------------------|------|------|
| 1 | `multi_pm` | ❌ | in-kernel UNSHARD resolve; multi-device descoped |
| 2 | early movement ops `pm_mops`+`mop_cleanup` | ✅ | |
| 3 | load collapse `pm_load_collapse` | ✅ | indexing-by-tensor; + `test_rewrite_diff.py` differential |
| 4 | split ranges `pm_split_ranges`+`pm_flatten_range` | ✅ | `mark_range_mod` + split substitute (dict ctx) |
| 5 | initial symbolic `sym`+`pm_flatten_range` | ✅ | |
| 6 | simplify ranges `pm_flatten_range`+`pm_simplify_ranges` | ✅ | merge_adjacent + range_gates |
| — | `apply_opts` | ✅ | between the two stage lists |
| 7 | postopt symbolic `sym`+`pm_move_where_on_load`+`pm_flatten_range`+`pm_reduce_unparented` | ✅ | `where_on_load` ported |
| 8 | expander `expander2` | ✅ | |
| 9 | remove reduces `mop_cleanup`+`pm_reduce_local` | ✅ | |
| 10 | add local buffers `pm_add_local_buffers` | ✅ | |
| 11 | add gpudims `pm_add_gpudims` | ✅ | |
| 12 | expand broadcast / add loads `symbolic_simple`+`pm_expand_broadcast`+`pm_add_loads` | ✅ | |
| 13 | devectorize2 `symbolic_simple`+`devectorizer2`+`indexing_simplify` | ✅ | |
| 14 | early symbolic `sym` | ✅ | |
| 15 | memory coalescing `memory_coalescing` | ✅ | `coalesce.rs` |
| 16 | add images `symbolic_simple`+`ew_devectorizer`+`pm_simplify_add_image` | ⬜ | images descoped; `ew_devectorizer` inline no-op |
| 17 | extra symbolic `sym`+`indexing_simplify` | ✅ | |
| 18 | lower all index dtypes `symbolic_simple`+`pm_lower_index_dtype`+`indexing_simplify` | ✅ | weak lowering wired; `_lower_weak_ops` must exclude COMPARISON |
| 19 | final symbolic `symbolic` | ✅ | |
| 20 | cast float alu `pm_cast_float_alu` | ✅ | |
| 21 | early decompositions `symbolic_simple`+`get_simplifying_rewrite_patterns` | ✅ | floordiv/floormod→idiv/mod + pow2 shr/and + threefry + MAX→CMPLT/WHERE (`decomp.rs`; MAX gated on the renderer's `code_for_op`) |
| 22 | decomp dtypes `pm_dtype_decomps`+`pm_commit_weak` | ✅ | long-as-2xint32 + fp8/bf16/fp16 float emulation (`decomp_dtype.rs`) |
| 23 | late decompositions `pm_decomp` | ⬜ | has `symbolic_simple`+`get_simplifying_rewrite_patterns`; still missing `get_late_rewrite_patterns`/`get_transcendental_patterns` |
| 24 | move gates from index `pm_move_gates_from_index` | ✅ | `gater.rs` (4 non-image rules) |
| 25 | final rewrite `pm_final_rewrite`+`pm_remove_invalid` | ✅ | |
| 26 | casted consts `pm_casted_consts` | ✅ | walk |
| 27 | add implicit barriers `pm_implicit_barriers` | ✅ | `pipeline::pm_implicit_barriers` (raw + WAR) |
| 28 | add control flow `pm_add_control_flow` | ✅ | `pipeline::add_control_flow` (CFGContext) |
| 29 | number params `pm_number_params` | ⬜ | param renumbering (corpus params already distinct) |

**Remaining real gaps**: `get_late_rewrite_patterns`+`get_transcendental_patterns`
(#23), then `pm_number_params` (#29). Descoped: images (#16), `multi_pm` (#1).

**Differential gates** (`py/tests/test_rewrite_diff.py`; pytest is available in
`../tinygrad/.venv`, but the gates also run as plain scripts): 10 tests, all green —
`commutative` 192/8, `div_and_mod_symbolic` 192/8,
`mop_cleanup` 200/0, `pm_decomp` 336/14, `pm_dtype_decomps` 250/0,
`pm_load_collapse` 300/0, `pm_reduce_load_collapse` 300/0, `pm_split_ranges` 300/0,
`symbolic` 278/22, `symbolic_simple` 287/13 (ok/skip).

**Rule-parity inventory** (`python py/pm_inventory_diff.py`): exact 0/0 for
`pm_split_ranges`, `pm_dtype_decomps`, `pm_load_collapse`, `pm_implicit_barriers`,
`pm_flatten_range`, `pm_reduce_unparented`, etc. Inherited deltas: `pm_decomp` 76/76
(5/5, from `symbolic_simple`), `symbolic_simple` 71/71 (5/5), `symbolic` 118/118 (9/9),
`sym` 136/139 (14/11), `pm_reduce_load_collapse` 128/128 (10/10).

## 6. What we deliberately leave out (and what breaks because of it)

| Omitted | Consequence / guardrail |
|---|---|
| multi-device (`UNSHARD`, sharding, allreduce) | assert single-device in Tensor ctor; corpus avoids `.shard()` |
| fancy `__getitem__` corpus coverage | ported in M7 Phase 4 (§8.15): `Ix::Tensor` + the advanced path, with graph-key gates; 0-d/list indices not exposed |
| GPU runtimes (CUDA/AMD/NV/...), HCQ/HCQ2, graph capture | concrete devices are SKIP, but the traits they plug into are PORTED (M6) so OpenCL (M8) is additive; `exec_copy` simplified to memcpy |
| GPU memory mgmt: `DepsTracker`, `MemoryManager`/page tables | needed only by HCQ-class devices; OCL gets plain `LRUAllocator` semantics |
| BEAM driver / tensor-core layouts | not walled: `expand_wmma` + opts parameterization ported in M5; `opt/search.py` + `tc.py` activate in M7 |
| CPU x86 ISA JIT | perf hit; interpreter is our execution engine (could render C + cc later) |
| precompiled-CALL fast path, `@function`/`custom_kernel` UX | CALL/FUNCTION/PARAM spine native from M3; decorator + `transform_precompiled_call` activate in M7 |
| TINYFS/NPY/PYTHON devices, TLS | constructor rejects Path; no TLS — https-only sources stay user-supplied files |
| symbolic-heavy coverage | `sint`/`Variable`/vmin-vmax are core since M2; corpus is concrete-first with symbolic smoke tests — full coverage arrives with TinyJit (M7) |
| viz, process replay, disk cache, worker pool | no profiling hooks; `Context` env vars still ported for behavior toggles |

## 7. Risks

1. **Rewrite-order divergence.** `graph_rewrite` results can depend on pattern iteration
   order and match timing. RESOLVED in M2: drivers ported line-for-line, validated
   differentially at ~13k graphs/PM. Subtleties that mattered, for future ports:
   - Python `sum(uops)` starts at int 0 → first `+` is `ADD(CONST(0), t1)` (`__radd__`);
     `math.prod(xs, start=c)` multiplies c FIRST.
   - int×UOp puts the CONST on the LEFT (`__rmul__`).
   - `Counter.subtract` creates NEGATIVE entries for missing keys.
   - Python `//` is floor, `%` takes the divisor's sign — Rust `/`/`%` are trunc.
   - `early_reject` also guards against vacuous repeat matches.
   - commutative flipping orders by `tuplize` = `(op, arg, dtype, srcs)`.
   - Python mid-rewrite exceptions ABORT the rewrite — Rust mirrors via `TGERR:` panics
     caught at the oracle boundary; both-error graphs are skipped differentially.
2. **`str()`-dependent keys.** Never reimplement Python `str()` of args in Rust; use the
   bridge for cross-checks, keep a Rust-side structural hash for internal caches.
3. **Float semantics.** `exec_alu` truncates to the node dtype after each op. RESOLVED in
   M1 with pinned semantics (see the original notes): CPython int truncation =
   `v mod 2^64 -> reinterpret per ctypes signedness -> wrap to width`; `dtype.const` for
   int dtypes keeps the *exact* Python int; `ctypes` int types reject float inputs; NaN→int
   raises in CPython (unreachable from tinygrad flows). Canonical dtype names (via
   `INVERSE_DTYPES_DICT`) are `char/uchar/short/ushort/int/uint/long/ulong` +
   `half/float/double` — used as interchange identity.
4. **v0.14.0 is mid-refactor.** Tags (`rtag`/callify), `AFTER`/`STORE` assign semantics and
   `MOPS` movement are new; pin to the exact commit in this repo and treat `test/null/*`
   as the spec.
5. **Interning & memory.** Python's `UOpMetaClass.ucache` + weakrefs GC dead nodes; the Rust
   arena currently does not (see §9.2). The DAG is acyclic, so `Rc` is safe; interning via
   a global `WeakSet`-like map is the path.

---

## 8. Implementation status

### 8.0 Build / test gate
- `cargo test` → **200 passed, 11 ignored, 0 failed** (2026-10, with the ONNX
  importer). The ignored = 1 `backend_harness` (`#[ignore = "M7 gate: XOR training
  via tinygrad.nn (Gradient/Adam)"]`), 1 `nn_xor_parity::xor_parity_tinygrad_seed42`,
  1 `mnist_e2e` (downloads data + a CNN train step), 1 `vit_e2e`, 1
  `onnx::external_checkpoint_smoke` (needs `ONNX_MODEL`), and 6 doc-tests.
- Python differential gates all green: `test_rewrite_diff.py` (10 tests),
  `test_roundtrip`, `test_tensor_diff`, `test_schedule_diff`, `test_codegen_diff`,
  `test_exec_diff`, `test_native_exec`.
- Suites: `tests/{golden,backend_harness,marshal,ops,grad,rand,m2_smoke,flat_codec,nn_xor,
  nn_xor_relu,nn_xor_parity,session,mem_probe,nn_conv,nn_norm,nn_module,indexing,assign,
  mnist_e2e,state,extra_ops,ctx_ops,onnx}.rs`.
- `cargo build` is warning-free; `cargo rustdoc --lib -- -W missing_docs` is
  warning-free (all public items documented); `cargo clippy
  --all-targets` has 0 errors and 0 warnings (the `mutable_key_type` false positive —
  `UopId` hashes by immutable `id` — is allowed at the crate root; the remaining
  `too_many_arguments` sites are targeted allows on functions whose signatures
  mirror tinygrad).
- `cargo fmt` is clean after the 2026-10 formatting pass.
- MSRV verified: `cargo +1.88.0 check` (default, `--all-features`,
  `--no-default-features`) and `cargo +1.88.0 test --no-run` are warning-free.
- `cargo package --allow-dirty` succeeds (149 files, 2.3 MiB; the packaged crate
  compiles in the verify step), and `cargo doc --no-deps` is warning-free.

### 8.1 Core graph engine (mature)
`src/uop/`: interned UOps, `UopId` (a strong `Rc<Node>` handle, `Clone` not `Copy`), shapes, dtypes, `Ops`, `Arg`, `SVal`/`SInt`,
const folding, `graph_rewrite`/`UPat`/`PatternMatcher` (`Send + Sync`, `OnceLock`-cached), min/max range analysis, `substitute`.
`src/codegen/`: simplify, postrange, coalesce, gater, linearize, pipeline
`full_rewrite_to_sink`. `src/schedule/`: rangeify, sched, indexing, memory.
`src/runtime/`, `src/renderer/`: INTERP always on; CL behind `backend_cl`.

### 8.2 Tensor front-end
`Tensor { uop: UopId }`, a `Copy` handle. Movement (reshape/expand/unsqueeze/pad/permute/
flip/transpose/squeeze/flatten/getitem/shrink/pool); elementwise (add/sub/mul/div/neg,
maximum/minimum, exp2/log2/sqrt/sin/trunc/ceil/floor/reciprocal, binop/binop_c, where);
reduce (sum/prod/max/mean/var/cumsum); creation (`from_slice<T: SliceElem>` width-precise
dtype inference, `from_bytes`, `empty`, `new_buffer`, `full_const`, `zeros/ones`, `arange`;
`rand`/`uniform` delegate to the faithful Threefry RNG).

### 8.3 RNG — Threefry (DONE)
Reproduces tinygrad `Tensor.manual_seed(...)` bit-for-bit as a **host evaluator**
(`src/tensor/rand.rs`), because INTERP stores every value as `f64` and cannot `bitcast` or
carry a `uint64` pattern. State thread-local (mirrors class-level): `seed`, per-device
seeds, counters. `manual_seed`, `_next_counter` (`sha256(index as 4B BE)` low-32 + seed),
`threefry2x32` (5×4 rounds), `random_bits`, `bits_to_rand`. API: `rand`, `uniform`, `randn`,
`randint`, `manual_seed`, `kaiming_uniform`, `glorot_uniform`, `scaled_uniform`.
`nn::Linear::new` init is faithful. `tests/rand.rs` 8/8 green (seed 42/7/99/100).

### 8.4 Typed marshalling (DONE)
No `f64` round-trip; reads reinterpret native-dtype bytes, cross-dtype reads insert a graph
cast.
- `Tensor::to_bytes` (native storage via `tensor_bytes`), `to_list<T: ReadDtype>` (graph-cast
  to `T::DTYPE`, realize, reinterpret LE bytes), `item<T>` (asserts `numel==1`), `realize`.
- `*_with` variants take a caller-owned `&mut BufTable` + `&mut ProgCache` for buffer reuse.
- Input: `SliceElem::default_dtype()` width-precise; `encode_native()` exact LE bytes.
- `ReadDtype` implemented for `i8/i16/i32/i64/u8/u16/u32/u64/f32/f64/bool`; f16/bf16/fp8 are
  read by casting to a wider type.
- `tests/marshal.rs` 6/6.

### 8.5 Tensor-op tests (DONE)
`tests/ops.rs` 9/9 (binary/unary/reduce/cast/activation/movement/matmul). Op bugs found &
fixed: `log` = `log2 * ln2` (was `/`), `dot` N-D reshape math, `tanh` argument, `buffer_of`
view ops, `to_bytes` materializing copy, `expand`/`broadcast_to` axis order.

### 8.6 Autograd (engine DONE)
`src/tensor/grad.rs` ports `mixin/gradient.py` (`compute_gradient`, `_deepwalk`,
`reduce_gradient`, `pm_gradient` rules). tinygrad expresses `pm_gradient` as a
`PatternMatcher`; each rule emits one gradient per source keyed on `op`, so we dispatch
with an explicit `match` (`grad_of_op`) — rule bodies ported one-for-one. The
`FUNCTION`/`CALL`/`GETTUPLE` machinery is intentionally omitted (eager graphs never build
those nodes).
- Covers CAST/RECIPROCAL/SIN/LOG2/EXP2/SQRT/TRUNC/CMPLT/CMPNE/ADD/SUB/MUL/MAX/POW/WHERE/
  REDUCE/CONTIGUOUS/CONTIGUOUS_BACKWARD/RESHAPE/EXPAND/PERMUTE/FLIP/BITCAST; unsupported ops
  (PAD/SHRINK/STACK/COPY/TUPLE) return `None` → `TGERR:` (XOR never reaches them).
- **Shape-sum glue**: when an edge's grad shape differs from its source (e.g. expanded
  bias), `cast→sum(broadcast_axes)→reshape→cast` before accumulating.
- Entry points: `Tensor::{gradient, backward, detach}`.
- Grad side-table on `UOpCtx`: `grads: HashMap<UopId,UopId>` with
  `ctx.{set_grad,grad,clear_grads}` + `Tensor::grad(&self, ctx)` — **why `Tensor` is Copy**:
  all mutation funnels through `UOpCtx`, so no `&mut Tensor` is needed. `is_param` dropped:
  trainability is decided by the `Module` tree, not a per-tensor flag.
- `tests/grad.rs` 6/6.

### 8.7 NN (layers DONE — see §8.25; `State`/safetensors still open)
- `src/tensor/nn.rs`: every `nn/__init__.py` layer — `Linear`, `Conv2d` (N-D faithful
  `forward`), `Conv1d`, `ConvTranspose2d`, `ConvTranspose1d`, `BatchNorm`, `LayerNorm`,
  `LayerNorm2d`, `GroupNorm`, `InstanceNorm`, `RMSNorm`, `Embedding`, `LSTMCell`, plus
  activations (relu/gelu/softmax/log_softmax/sigmoid/tanh/sin/cos/abs/sqrt/rsqrt).
- **`Module` trait**: `named_params(&self) -> Box<dyn Iterator<Item=(String, Tensor)>>` +
  defaulted `params()` + `scoped(prefix, iter)` (`"l1.weight"`, safetensors-ready).
  `Box<dyn Iterator>` keeps `Module` object-safe; names live in the trait, never on
  `Tensor`.
- `src/tensor/optim.rs`: `Optim` trait + `SGD` owning its `params` (`SGD::new(params, lr)`,
  tinygrad `optim.SGD([w,b], lr)`). `schedule_step(&mut ctx, &mut sess, roots)` mirrors
  `Optimizer.schedule_step`: `_step` may `assign` state buffers and returns
  `(updates, extra)`, the applied values `t.detach() - update` are co-realized with the
  caller's `roots` (the loss) and `extra` in **one** schedule (`Tensor::realize_many`), then
  assigned back into the params. `step` = `schedule_step(roots=[])`. `_apply_update` and
  `zero_grad` match tinygrad. **LARS/LAMB ported** (`SGD = LARS(tcoef=0, momentum=0, wd=0)`,
  `Adam`/`AdamW = LAMB(adam=true)`); Muon newton-schulz intentionally not ported.
  Optimizer state is `Tensor::zeros_param` (a pure BUFFER leaf; see §9.5). Gate:
  `tests/nn_optim.rs` (3/3) diffs one- and multi-step SGD/LARS/Adam/AdamW against tinygrad
  goldens (`py/dump_optim_golden.py`).
- `nn/state.py` equivalent: `tensor::state` (`State`, safetensors, `get_state_dict` /
  `load_state_dict` incl. buffers) landed in §9.9. `fetch`/`from_url` is **descoped**.
- Tensor helpers added for nn: `dot`, `linear`, `batchnorm`, `layernorm`, `pool`, `uniform`,
  `rand`, `tanh`, `div`, `square`, `sqrt`, `abs`, `eq`, `where_`, `chunk`/`split`,
  `unflatten`, `shrink_to`, `conv_transpose2d`.
- Tests: `nn_xor.rs` 3/3 (incl. `xor_trains_to_convergence`, `micro_repeated_realize`),
  `nn_xor_relu.rs` 1/1; `nn_xor_parity.rs::xor_parity_tinygrad_seed42` `#[ignore]`d;
  `backend_harness` XOR-training `#[ignore]` placeholder.

### 8.8 Realize memory: `Rc` arena + tensor-owned buffers (9.1/9.2 — DONE)
- **Exponential-realize fix (DONE, `c2b8d50`)**: `Graph::to_flat` serializes only the
  toposort reachable from `self.roots`. Per-iter realize is ~5ms flat (was exponential).
- **`Rc<Node>` arena (9.2, DONE)**: `UopId` is now a strong handle to an immutable
  `Rc<Node>`; `Node.srcs` holds strong child handles and the interner is
  `HashMap<u64, Weak<Node>>` (Python's `ucache` semantics). A node is freed the moment its
  last handle drops — no GC pass, no safe-point, no slot reuse. Property caches
  (`shape_c`/`mm_c`/`range_c`) moved onto the node (interior mutability) and die with it.
  `Tensor` is `Clone` (not `Copy`) and holds the handle plus an `Rc` realized-buffer cell.
- **Tensor-owned buffers (9.1, DONE)**: `Tensor::{to_bytes,to_list,realize}` cache the
  realized `Buffer` on the tensor (shared by clones) and reuse it; `BufTable` keeps bases
  alive only for the duration of a realize and stores views as `Weak`, pruning any base no
  live caller references at the start of each realize.
- **Result**: the single-ctx 120-step training loop holds **flat at 898 live nodes and 0
  retained bases** (was ~133k nodes / ~5.4k bases, RSS 240MB → 15MB). Regression test
  `tests/mem_probe.rs::single_ctx_training_is_memory_bounded`.
- **Persistent realize state (`Session`)**: `engine::session::Session` bundles a `BufTable` +
  `ProgCache` + `var_vals` and is passed as the single realize-state argument to every read
  (`t.to_list::<f32>(&mut ctx, &mut sess)`) and optimizer step. `ProgCache` is content-keyed
  (`Graph::content_key`), so compiled kernels survive a fresh `UOpCtx` per step; its hit path
  re-interns the PROGRAM uop in the current arena (the cached `uop` id is arena-local).
  The XOR training loop shares one `Session`: **10 kernels compiled, 3290 hits / 300
  epochs**. `tests/session.rs` (4 tests) pins it. There is deliberately **no session-wide
  `realized` memo** — the per-tensor buffer cell (tinygrad `Tensor.realized`) is the memo;
  a uop-keyed map would pin every step's graph in a training loop. For one-shot reads use the
  `_fresh` conveniences (`to_list_fresh`/`to_bytes_fresh`/`item_fresh`/`realize_fresh`),
  which allocate a throwaway session.
- **`add_local_buffer` (DONE, `11ab94a`)** + `pm_add_local_buffers` wired into the
  `"add local buffers"` stage (receives a slot counter `CtxVal::Cnt`).
- **Per-node side tables (DONE)**: the staged `initial_data` bytes and the autograd `grad`
  now live on the node (`Node::{initial_data,grad}`, interior-mutable), so both are freed
  with the tensor's subgraph instead of leaking in ctx-wide `HashMap<u32, _>` tables.
  `zero_grad` clears each param's node (`UOpCtx::clear_grad`); pinned by
  `tests/session.rs::node_side_data_dies_with_the_node`.
- **PatternMatcher slots (fixed)**: `RuleFn = Arc<dyn Fn + Send + Sync>` and the cached
  matchers live in plain `OnceLock`s (`PmSlot`), so the parallel test harness shares one
  genuinely thread-safe matcher per slot. The previous `OnceLock`+`unsafe Sync` cell
  (`RefCell`) could race between two threads and panic with "already borrowed". Rules that
  need arena-specific data (e.g. `add_control_flow`'s edge map) now receive it via
  `CtxVal::Dict` instead of capturing a `UopId`.

### 8.9 Backends
- **`RAYON`** (multithreaded CPU, feature `backend_rayon`, optional `rayon` + `bytemuck`)
  — §9.8. R0–R2 landed: device selection priority CUDA → CL → RAYON → INTERP; an owned
  `Send` f64 kernel IR executed across the global launch grid with rayon (R1); and a
  **dtype-native typed register machine** (R2, `ray_value.rs`/`ray_typed.rs`) used by
  default, with exact `uint64`/integer semantics and `bytemuck` slice reinterpretation.
  R2.2 hoisted all per-instruction framing out of the interpreter loop (§9.8.2). Typed
  covers 159/159 kernels in the ops/conv/norm/attention/indexing/layer suites; matmul 512²
  ~87× / elementwise ~7.5× vs `INTERP` (4.3× over the f64 R1 path on matmul). Default device
  on a CPU-only host is `RAYON`.
- INTERP always on; CL behind `backend_cl` (opencl3). M8 render byte-diff + numerics +
  kernel-count parity on the RTX 3060. `tests/backend_harness.rs` 12 passed / 1 ignored.
- **Default device** (`Tensor::creation::default_device`): env `DEFAULT_DEVICE` if set, else
  `CL` when `backend_cl` is compiled in and an OpenCL device is present
  (`ops_cl::available`, a non-panicking probe), else `INTERP`. There is **no "NV"**: it was a
  leftover name for a backend that was never ported (it silently ran on the interpreter), so
  `Tensor::device(ctx)` now reports the real one (`UOp.device`, ops.py:847).
- The Python differential gates compare graph/`key()` hashes that embed the device string, so
  tinygrad's default here is `NV` (its `ops_nv` backend opens). The harnesses
  (`test_tensor_diff`/`test_schedule_diff`/`test_codegen_diff`) pin the oracle's
  `DEFAULT_DEVICE` to tinygrad's `Device.DEFAULT` so both sides use the same label.
- **MAX is renderer-gated**: `python_alu` has it, cstyle/PTX/OpenCL do not, so
  `pm_decomp(supports_max)` rewrites `MAX(a,b) → (a<b).where(b,a)` only for the C-style
  renderers (decomp/op.py:89).
- **`render_buffer` local declaration** (cstyle.py:169): a local `BUFFER` renders as
  `__attribute__ ((aligned (16))) __local float buf[N];` — the element type uses the default
  ALU addrspace (`_render_dtype(dtype, sz=1)`), and the only `__local ` is the one in
  `smem_prefix`. The port was passing the buffer's LOCAL addrspace to `render_dtype`, which
  appended a second `__local ` *and* a `*`, emitting `__local float* buf[N];` (an array of
  pointers). That made every reduction lowered through workgroup memory fail with
  `CL_BUILD_PROGRAM_FAILURE`. `tests/cl_reduce.rs` pins it (fails to build without the fix).
- **CL offset views**: `CLAllocator` has no `_offset` (only python/cuda/hip/metal/disk/dsp/
  npy/hcq implement it upstream), so an offset view (`SHRINK`/`BITCAST` reaching `buffer_of`)
  panics. Our forward/backward XOR graph never produces one, so CL training works. tinygrad
  avoids it unconditionally by carrying `Buffer.offset` into the kernel's buffer arg (the
  real fix if a future graph needs it); `tests/cl_offset.rs` pins the current panic.

### 8.10 In-place parameter updates: `Tensor::assign` (DONE)
tinygrad's optimizers assign the update into the parameter's existing buffer
(`Optimizer.step` → `Tensor.assign`, optim.py:35-56 / tensor.py:436), so parameters keep a
stable identity and the training graph does not chain across steps. The port previously had
a **pure-graph** optimizer (`step` returns new tensors), which forced callers to either
rebind (`w = w - lr*grad`, making each weight's subgraph contain the previous one → O(N)
graph growth, `toposort_from` dominated, ~180 MB) or round-trip the weights through host
`Vec<f32>` every step (what `nn_xor`/`mem_probe` used to do).
- `Tensor::assign(ctx, sess, value)` is the faithful tinygrad path (tensor.py:455):
  `self.uop.after(self.uop.store(value.uop))` — a fused `STORE`+`AFTER` that the scheduler
  runs as part of the surrounding schedule, leaving `self.uop` a stable leaf. `AFTER` orders
  any read of the parameter in the same schedule **before** the store, and
  `Optim::schedule_step` builds the effects with `Tensor::assign_effect` so the parameter
  write fuses into the same schedule as the loss (`Tensor.realize_many(roots + effects +
  state)`), instead of a separate realize-then-copy pass. `Tensor::assign` (the standalone
  one-shot form) realizes the effect and then `Tensor::pin_buffer`s the parameter.
- `Tensor::pin_buffer(ctx, sess)` pins the parameter's own storage (`buffer_of(self.uop)`)
  onto the shared buffer cell. tinygrad's `Tensor` object owns its realized storage; here the
  cell is that owner, so `BufTable::prune` cannot free a parameter's buffer between the
  schedule that wrote it and the next read. **Without this, a fused `STORE` into a
  not-yet-pinned parameter was silently lost** (the buffer was pruned and re-zeroed); the
  `tests/assign.rs` gates pin the read-before-store ordering for both elementwise and
  reduce reads. `buffer_copy_into` remains as a device-copy primitive (`tests/session.rs`).
- `Tensor::to_param` / `nn::Module::into_params`: random-init parameters are *expressions*
  (`uniform` = `rand*range + low`), which would be recomputed — discarding the update — on
  every realize. `to_param` realizes the init once and stages it on a buffer-backed leaf
  (`Node.initial_data`), i.e. tinygrad's "parameters are realized buffers". Do it once
  before training.
- `Optim::step(&mut ctx, &mut sess)` = `step_graph` + `assign` each update back into its
  parameter (tinygrad `Optimizer.step`); `step_graph(&mut ctx)` (return-new-tensors) is kept
  for the parity tests. `SGD` owns its params (`SGD::new(params, lr)`).
- Result: `tests/nn_xor.rs`, `examples/xor.rs` and the rewritten `tests/mem_probe.rs` hold
  the `Linear` layers and loop `forward → backward → step → zero_grad`. The per-step
  graph is identical, so every step after the first is a program-cache hit and the live graph
  is flat (the last XOR run: 10 misses / 3290 hits, loss 0.021).
- Remaining port simplification: tinygrad rewrites the parameter's *uop* to the `AFTER`
  (`self.uop = assign`, or the `_apply_map_to_tensors` view-assign at tensor.py:455-463) so
  later ops on it reference the write. The port leaves `self.uop` a leaf and relies on the
  shared buffer cell (pinned) plus `AFTER` ordering, which is equivalent for the eager,
  per-step graphs here. Consequently a *derived* uop materialized in an earlier step still
  must not be reused (training loops build fresh derived nodes each step); the write itself
  is now fused and ordered rather than a post-hoc copy.

### 8.11 Rangeify realize: `PCONTIG` arm in `maybe_realize_axes` (DONE)
`schedule/indexing.py:274` gates per-axis realization on
`if not (PCONTIG > 1) or any(rr.arg > e.arg ...)`. `PCONTIG` defaults to **0**
(`helpers.py:274`), so `not (PCONTIG > 1)` is **always true**: whenever a node has ending
ranges, **all** its axes are realized. The port had dropped the `PCONTIG` arm and evaluated
only the per-axis comparison, so an intermediate `REDUCE` with ending ranges was realized on
one axis instead of all. Concretely, for `r = a*b; s = r.sum(2); y = r * s.reshape(2,4,1);
o = y.sum(1)` the scheduler fused both reductions into one 5-axis kernel instead of
materializing `s` as its own kernel; the fused kernel's local reduction produced wrong values
(tinygrad materializes the intermediate, so it stayed correct). This is what made first-layer
gradients wrong on every backend (`a.dot(w.T)` is exactly this reduce→broadcast→reduce shape).
- Fix: `src/schedule/indexing.rs` `maybe_realize_axes` — restore the `PCONTIG` arm (default
  context realizes every axis).
- New gates: `tests/reduce_broadcast.rs` (forward + `p.grad` vs tinygrad),
  `test_schedule_diff.py::test_reduce_broadcast_reduce` now also compares the **single-output**
  schedule via the new `oracle_tensor` mode `schedule_last` (what `Tensor.realize` actually
  schedules; the all-outputs schedule can hide internal-node fusions). With the fix the CL and
  INTERP XOR runs agree exactly (loss 0.030414 at 100 epochs, all 4 cases correct), and
  `tests/nn_xor_relu.rs::xor_relu_converges` passes on CL.

### 8.12 M7 nn op coverage + sibling-reduce scheduler fix (DONE)
- **conv2d DONE** (`src/tensor/op.rs`): full port of `mixin/op.py:1595-1607` (`_pad_constant`
  incl. negative pads, `_pool`, `flat_to_grouped`/`resolve_pool_pads`), plus
  `Tensor.max_pool2d` (ceil_mode + return_indices), `Tensor.one_hot`, `Tensor.argmax`,
  `Tensor.eq`, and a faithful `sparse_categorical_crossentropy`. `nn::Conv2d::forward` now
  calls it. Also fixed a real `_pool` mis-port: the noop dims must keep their sizes in the
  interleaved reshapes and the stride shrink must happen **before** the `(k,o,s)` reshape
  (the stub had never exercised `_pool`). Gates: `tests/nn_conv.rs` (conv2d 5/5, max_pool 4/4,
  one_hot, CE 4/4) vs `tests/nn_golden.rs` (`py/dump_nn_golden.py`).
- **FIXED — sibling/nested reduces in one realize.** Minimal repro: for `a:(2,3)`, `b:(2,)`,
  `a.sum(axis=1).sum() + b.sum()` previously either panicked in `coalesce`
  ("attempting multiple stores") or returned a wrong value (`32` instead of `23`). Two
  independent port bugs, both now fixed:
  1. **`src/codegen/simplify.rs` `placeholder_reg`** (the REG accumulator from
     `reduce_ranges_to_acc`, codegen/__init__.py:210). tinygrad's `UOp.placeholder` reshapes
     the flat `(prod(shape),)` buffer back to `shape` when it has >1 dim (`ops.py:1148`);
     the port skipped that, so `acc.alu(op, inp)` broadcast a 1-D acc against a 2-D input
     (`acc[2] + inp[2,1] -> [2,2]`) and the local reduction summed wrong. This was the
     actual wrong-answer bug.
  2. **`src/codegen/coalesce.rs`** tested `addrspace_of` (reads `ParamArg` only) instead of
     the derived `buf.addrspace`. The buffer being coalesced is usually an `AFTER`/`INDEX`
     wrapping the real BUFFER, and `addrspace` propagates through those (`ops.py:867`), so an
     `AFTER`-wrapped REG accumulator was not skipped and its (legitimately repeated)
     serial-reduce stores tripped the coalesce assert. Also fixed the same latent mistake in
     `rangeify.rs::handle_after` (tinygrad uses `after.addrspace`).
  Gates: `tests/reduce_broadcast.rs::{sibling_reductions_same_realize,sibling_reductions_batched}`,
  `tests/nn_conv.rs::sparse_categorical_crossentropy_matches_tinygrad` (un-ignored). **MNIST
  e2e is no longer blocked on this.**

### 8.13 `Tensor.assign` faithful STORE+AFTER (DONE)
See §8.10 for the full write-up. `assign` is now
`self.uop.after(self.uop.store(value.uop))` (tensor.py:455) rather than realize-then-copy;
`Optim::schedule_step` builds the write effects and co-realizes them with the loss and state
so the parameter update fuses into one schedule; `Tensor::pin_buffer` makes the parameter own
its storage so `BufTable::prune` cannot drop a fused write. `buffer_copy_into` remains a
device-copy primitive. Gates: `tests/assign.rs`.

### 8.14 M7 Phase-3 Module/parameter collection (DONE)
- `nn::Module` gained `fn into_params(self, ctx) -> Self` (a required method, consuming and
  rebuilding) alongside `named_params`/`params` (tinygrad `nn.state.get_parameters`
  analogue). Every layer implements it: `Linear`, `Conv2d`, `BatchNorm`, `LayerNorm`,
  `GroupNorm`, plus `impl<T: Module> Module for Vec<T>` (numeric path prefixes like
  tinygrad's `get_state_dict` list indexing: `"0.weight"`). `nn::Sequential` was **cut** by
  request — models are explicit structs with a hand-written `forward` (like tinygrad's
  `class Model`), which is both simpler and closer to the reference.
- `examples/xor.rs` and `tests/nn_xor.rs` now define an `XorModel` struct wired with `Module`
  (`new` → `into_params` → `params().collect()` for the optimizer), so no call site
  hand-lists parameters.
- Fixed a real `LayerNorm::forward` bug while implementing its `Module` impl: it computed
  `x + weight*bias` instead of tinygrad's `x * weight + bias` (nn/__init__.py:258). Gate:
  `tests/nn_module.rs::layernorm_applies_weight_then_bias` (non-trivial weight/bias vs the
  tinygrad oracle).
- Gates: `tests/nn_module.rs` (nested name flattening, `Vec<T>` indexing, `into_params`
  leaf materialization, LayerNorm affine).
- **Known simplification (not touched):** `GroupNorm::forward` is still a placeholder
  (normalizes like BatchNorm, `_norm_shape` returns all-ones); `BatchNorm`'s running-stat
  update is still not wired (`forward_train` is a no-op for stats). Neither is used by XOR;
  both matter for MNIST Phase 5 and are tracked for that step.

### 8.15 M7 Phase-4 advanced (tensor) indexing (DONE)
`Ix` gained a `Tensor(Tensor)` variant (so `Ix` is now `Clone`, not `Copy`), and
`movement::getitem` ports `_getitem`'s advanced path (`mixin/op.py:104-146`):
- negative-index normalization `(index < 0).where(index + size, index)`;
- `_broadcast_shape` over the index tensors (right-aligned; size-1 broadcasts);
- the **consecutive-int-dim linear fast path** (`usum(t*stride)` + a `valid` boolean
  mask + `valid.where(x, 0)`), and the general `_one_hot_along_dim` → `uprod` → `where`
  → `sum` mask path with the non-contiguous `permute` case;
- out-of-bounds reads zero.
It is built entirely from already-ported elementwise/movement/reduce ops, so no engine
change was needed. `Tensor::ge` was added (tinygrad `ge`).
Gates: `tests/indexing.rs` (9 numeric cases vs the tinygrad oracle) and
`py/tests/test_tensor_diff.py` (6 new **graph-key equality** cases, incl. the fast path and
the permute path). `py/tests/test_tensor_diff.py` now clears `UOp.ucache` per subtest (the
eagerly-allocated index BUFFERs collided across cases after the `unique_num` reset).
Known gaps (documented in `INDEXING.md`): 0-d tensor indices (tinygrad treats them as
`sint`), Python `list`/`tuple` indices in the Rust API (`Ix::Tensor` covers them), and
`__setitem__` (not needed for MNIST).

### 8.16 M7 Phase-5 MNIST (DONE) + the engine bugs it exposed
**Layers.** `BatchNorm` is now a faithful port of `nn/__init__.py:8-64`:
`calc_stats` (running stats in eval, batch stats in train; `batch_mean.detach()` so
`d(var)/d(mean)=0`), `forward` with the in-graph running-stat `assign`s
(`(1-m)*r + m*batch`, `numel/(numel-C)*batch_var`) and `num_batches_tracked += 1`,
and `forward_inference`. Running stats are `to_param`-ed buffer leaves (still excluded
from `named_params`); `Tensor::rsqrt` added. `GroupNorm` is the faithful
`reshape(N, G, -1).layernorm(-1).reshape(N, C, ...)` + per-channel affine
(`nn/__init__.py:179-203`). Gates: `tests/nn_norm.rs` (BatchNorm train/eval/running-stat
goldens, GroupNorm 4d/3d goldens) generated by `py/dump_nn_golden.py`.

**Example.** `examples/mnist.rs` mirrors `beautiful_mnist.py` (same layer list, Adam,
`X[samples]` minibatching via `Ix::Tensor`, `TARGET_EVAL_ACC_PCT`); the IDX loader shells
out to `curl` (single `sh -c 'curl … | gunzip > file'`, skipped if cached). `TRAIN_N`/
`TEST_N`/`MNIST_DIR` are port conveniences. `tests/mnist_e2e.rs` is an **ignored** gate
that includes the example as a module and checks one train step + eval.

**Faithfulness fixes found along the way** (all against tinygrad):
- `reshape` `-1` inference: tinygrad's `-prod(shape) // prod(new_shape)` includes the
  `-1` itself (`-48 // -4 = 12`); the port filtered it out (`-48 // 4 = -12`).
- `Tensor.batchnorm` `shape_mask` was inverted: tinygrad keeps the *reduced* axes
  (`tuple(s if ax in axis_ else 1 ...)`).
- `flatten`: `end_dim` is inclusive (`prod(shape[start:end_dim+1])`), and a 0-dim input
  flattens to `(1,)`.
- `Tensor.argmax` was still the M3 panic stub; it is now the faithful
  `C - ((x==x.max(axis,keepdim=True)) * arange(C,0,-1)).max(axis)` (gate: `ARGMAX_GOLDEN`).
- autograd gained `PAD`/`SHRINK` grads (`mixin/gradient.py`: pad↔shrink).
- `Optim::schedule_step` realizes every gradient subgraph once (they share the backward
  graph) and freezes each into a buffer leaf (`Tensor::realize_to_leaf`) before `_step`;
  otherwise the per-parameter state `assign`s re-realize the whole backward graph 2×/param.

**Scheduler/codegen bugs exposed by the CNN** (fixed; these were latent):
- `Scheduler::rngs` must sort by `(axis_to_pos, arg[0:-1])`. `pm_split_ranges` inserts a
  split discriminator into the range arg; ignoring it made the sort tie (nondeterministic
  `HashSet` order) and the optimizer's divide checks fail.
- `pm_reduce_load_collapse`'s gated-load rule must name the **inner RANGE**
  (`UPat(..).or_casted()` with no name), not the cast — this is what collapses
  `one_hot(idx).where(x, 0).sum()` into an efficient gather (`x[idx]`). Missing it left
  the mask materialized (O(N·C) per index) and out of bounds for some sizes.
- `add_gpudims`: `get_grouped_dims(reverse=True)` reverses `dims` *before* naming the
  SPECIALs (`gidx0` ↔ the last global dim), then reverses the list. The port only
  reversed the list, attaching each `gidx` to the wrong range (out-of-bounds gathers).
- `group_dims` was rewritten to tinygrad's recursion (the old index arithmetic could
  underflow / read stale dims).
- `fix_group_for_reduce` is now implemented (GROUP_REDUCE → early reduce + LOCAL STAGE +
  final reduce), so `hand_coded_optimizations`' GROUPTOP path works.

**Known limitation (unchanged):** the single-threaded INTERP backend executes every range
extent serially, so a CNN backward step is ~1e9 interpreted ops (minutes). The example
targets `DEFAULT_DEVICE=CL`; the ignored gate runs a 4-image step.

### 8.17 OpenCL MNIST end-to-end + host-side performance findings (DONE)
Commit `84edf80` made the `DEFAULT_DEVICE=CL` training run complete a step and print a
finite loss. Two CL-only bugs, both now pinned by tests:
- **Weak-dtype comparison lowering** (`src/uop/weak.rs`, `tests/weak.rs`): a comparison
  whose operand is a weak int/float must be committed to a concrete dtype before codegen
  (`GroupOp.Binary` includes `CMPLT`/`CMPNE`/`CMPEQ`), and `lower_weak_node` must re-derive
  the rebuilt node's dtype (comparison → `bool`). Missing this rendered `weakint` and CL
  failed with `use of undeclared identifier 'weakint'`.
- **cstyle const rendering** (`src/renderer/cstyle.rs`): the uint8/uint16/int8/int16
  branches emitted `(uchar(0u))` (a malformed call) instead of tinygrad's `((uchar)(0u))`
  (cstyle.py `f"({ctx.render_cast(x, ...)})"`), which CL rejected with
  `CL_BUILD_PROGRAM_FAILURE`. Also fixed: the bf16/f16/fp8 const branch was dead code (the
  f64 fallthrough returned the plain float first), and non-finite float consts lacked the
  cast wrapper. Unit tests in `src/renderer/cstyle.rs` pin all of these.

**Performance — measured, not guessed** (`BS=2`, `STEPS=1..3`, CL/RTX 3060):
- `to_program` misses stay fixed at **115** across steps; hits grow ~195/step. The
  content-keyed `ProgCache` works: kernels are **not** recompiled.
- ~1 s/step. btop showed the **GPU at 100% and one CPU core pegged**: the host is
  single-threaded and every `realize` rebuilds the schedule (rangeify / kernel-graph /
  memory-plan `graph_rewrite`), and the optimizer realizes several roots per step. There is
  no `TinyJit`-style capture, so host graph rewriting is a co-bottleneck between dispatches.

**Backend-level gaps surfaced by this** (tracked in §9.6): BF16 on CL needs the renderer
`extra_matcher`; BEAM is absent; and the engine's `Rc`/`RefCell` design keeps all host work
on one thread.

### 8.18 BF16 on cstyle/CL (CORE DONE) + the bugs it uncovered
BF16 now computes on CL via tinygrad's renderer-side emulation:
- **`renderer.extra_matcher`** — added to the `Renderer` facts and run in the pipeline's
  "final rewrite" stage (`renderer/extra.rs`): `create_non_native_float_pats((bf16,))`
  computes bf16 ALU/WHERE/casts through f32, and `pm_manual_bf16_cast` does the f32↔bf16
  bit math (cstyle.py:76-102,319). The interpreter's matcher is empty.
- **OpenCL `BITCAST`** (`cstyle.rs`): was always emitting clang's `__builtin_bit_cast`;
  now `as_<type>(...)` for ALU (cstyle.py:322) and the pointer cast for GLOBAL/LOCAL,
  switched by a new `CStyleLang.bitcast_builtin`.
- **OpenCL bf16 consts** render as their 16-bit pattern `{bits}u` (cstyle.py:325) via
  `CStyleLang.bf16_const_bits`.
- **sub-word `rand`/`uniform`** (`rand.rs`): tinygrad's `_bits_to_rand` is generic over
  itemsize; the port only had f32/f64 and indexed out of bounds for f16/bf16. Added the
  2-byte path (`bits_to_rand_f16`).

Two latent engine bugs surfaced (both fixed):
- `cstyle::trunc_int` **clamped** instead of truncating/wrapping (dtype.py:289-293:
  `ctypes.c_<n>(x).value`), so `-1` rendered as `0u` and `-x` on a uint (bf16 bit math) was
  wrong.
- `pm_lower_index_dtype` rule order (`weak.rs`): tinygrad is
  `pm_commit_weak + pm_cast_weak + [lower_weak_srcs]`; the port put the driver first, so a
  weak const on a bf16 op was lowered to **default_float (f32)** and widened the consumer
  (`Mul(bf16,weakfloat) -> Mul(f32,[bf16,f32])`) instead of committing to bf16.

**Full bf16 CNN train step: DONE** (`DTYPE=bfloat16`). Getting there fixed two more bugs,
both of which made a large bf16 graph look like a *hang*:

- **`addrspace_derived` was uncached recursive** (`uop/ctx.rs`). tinygrad marks `addrspace`
  `@recursive_property` (ops.py:217-227, 866): a gated toposort memoizes it per node, so the
  walk is linear. The port's naive recursion re-walked shared subgraphs and is **exponential**
  in the DAG depth — and bf16 ALU emulation produces exactly those shared subgraphs (an inner
  value repeated per op). The "hang" was this blowing up inside the renderer's per-op
  `addrspace_derived` call. Fixed by adding `Node.addrspace_c` and memoizing (a per-node cache,
  matching the other `recursive_property` caches: `shape_c`/`mm_c`/`range_c`).
- **Adam/LAMB bias-correction scalars must be f32** (`tensor/optim.rs`). tinygrad builds
  `b1_t`/`b2_t` as `Tensor.ones((1,), dtype=dtypes.float32)` (optim.py:157) regardless of the
  param dtype. The port used `default_float` (bf16), where `0.999` rounds to `1.0`, so
  `1 - b2_t == 0` and `m_hat / v_hat` divided by zero → **NaN loss from step 1**. Now pinned to
  f32 (the moments `m`/`v` still track the param dtype; only the two 1-element accumulators and
  the transient bias-correction arithmetic are f32).

**Perf:** correct but ~2x *slower* than f32 on CL (bf16 is emulated through f32, so every op is
strictly more nodes — measured 23s vs 12s per BS=64 step on the RTX 3060). So the MNIST example
keeps the faithful f32 default and `DTYPE=bfloat16` stays opt-in.

**Gate:** `tests/bf16.rs` — bf16 mul/add/compare/where on every registered backend must
match the interpreter (CL included). `cargo test` 110 passed / 0 failed / 7 ignored.

**Still open (unchanged):** sub-dword vectorized coalescing bitcasts; the Clang
`supported_dtypes` bf16 arch gate. Tracked in §9.6.

### 8.19 Conv `pool` im2col blow-up — the real MNIST bottleneck (DONE)
The "TF32/GQ" hunt for a slow conv-backward kernel (§8.17/§9.6) found a bug in
`movement::pool` (`src/tensor/movement.rs`). tinygrad's `_pool` repeats only the
spatial dims: `self.repeat([1] * len(noop) + [ceildiv(k*(i*f+d), i)])`. The port
used each dim's **size** for the passthrough (noop) dims, so a `[64,32,24,24]`
conv input got `repeat([64,32,6,6])` instead of `repeat([1,1,6,6])`. That expanded
the im2col window tensor by `batch × channels` (2048x for BS=64), and the
conv-backward reduction over the (now-extra) batch/channel axes ran **~9.3 s per
kernel** instead of ~20 ms — a ~550x gap vs tinygrad measured on the same conv.

- **Impact:** f32 MNIST `BS=64` went from ~13 s/step to ~1.4 s/step; 20 steps
  (TRAIN_N=1024) run in **27.7 s** (67% test acc) vs >260 s before. bf16: 31.6 s.
- **Detection:** the forward is numerically correct, so the golden tests never
  caught it — only a graph-size assertion does. Added
  `conv_pool_window_is_not_repeated_over_batch_or_channels`
  (`tests/nn_conv.rs`): a small conv's max intermediate numel must stay ≤ 40000
  (correct 32768; the bug gave 65536).
- The earlier "extra `[64,32]` reduce axes with `valid<1`" seen in the codegen
  AST (§8.18 investigation) was this bug's downstream shadow: the gradient summed
  the spuriously-broadcast batch/channel axes.

### 8.20 BEAM search (`codegen/opt/search.py`) — core ported (opt-in)
`BEAM=n` (env, default 0=off) now runs beam search instead of
`hand_coded_optimizations`. New `src/codegen/opt_search.rs`:
- **`actions()`** — the candidate `Opt` list (UPCAST/UNROLL/LOCAL/GROUPTOP/GROUP;
  TC/SWAP/THREAD/PADTO/NOLOCALS are not ported).
- **`get_kernel_actions`** — one `Scheduler` per applicable action; skips
  non-dividing/invalid opts via a new `Scheduler::try_apply_opt` (the
  `KernelOptError` path; `apply_opt` stays panicking), plus the upcast/local
  product caps.
- **candidate compile + timing** — `args_from_ast` allocates dummy buffers per
  PARAM slot (required `OpaqueT` is public via `runtime::buffer`), compiles via
  `to_program` and times with `device.call(wait=true)` (CL event profiling).
  Adds `Compiled::runtime_nocache` so thousands of throwaway candidate programs
  don't leak into the device runtime cache, and caps the timing launch to
  `BEAM_TEST_GLOBAL_MAX` (65536) with time scaling (`get_test_global_size`).
- **search loop** — keep the fastest `amt` candidates per round; stop when the
  best time stops improving (`BEAM_MIN_PROGRESS`).
- Pipe: `full_rewrite_to_sink_with(..., optimize)` (`optimize = ast.tag is None`,
  codegen/__init__.py:479); a BEAM-tagged candidate skips the pre-opt stages and
  `apply_opts` (idempotent re-entry).

**Status: works on small kernels and now caches; the large-model CL abort's root
cause was found and fixed (`add_gpudims` split-range keying, below) — not a driver
limit.** Hardening landed:
- **Disk cache** — `engine/disk_cache.rs` (std-only, `AUTORUST_CACHE_DIR`,
  `CACHELEVEL`, `IGNORE_BEAM_CACHE`) stores the chosen `applied_opts` keyed by
  `(optimized-AST content key, beam width, device)`, mirroring `diskcache_get/put`
  (search.py:113/176). A second process replays the cached opts instead of
  re-searching (verified: 28-entry cache, no search rounds on the second run,
  identical loss).
- **`BEAM_UOPS_MAX`** (default 3000) rejects oversized candidates before rendering.
- **`seen_libs`** dedup by optimized-AST content key (search.py:140).
- **`Estimates.ops` filter** — a concrete `estimate_ops` (renderer/__init__.py:21,
  `ops` only, `ignore_indexing`) drops candidates using >1000x the smallest
  (search.py:143-146).
- **`BEAM_STRICT_MODE`** (propagate candidate panics) and **`BEAM_VERBOSE`**
  (keep panic messages); candidate compile *and* timing are now both caught, so
  one invalid candidate no longer aborts the search.
- **`BEAM_MAX_CANDIDATES`** — port-only per-round cap (default 0 = tinygrad).
- The `applied_opts` axis is stored **relative** (tinygrad records the original
  `Opt`), so cached opts replay correctly; `WMMA`/`TC` estimates are skipped.

**Root cause of the CL abort (debugged 2026-10): it is a port codegen bug, not a
driver limit or an unfreed resource.** Evidence:
- A minimal reproducer builds **3000 source programs** and **2000 full
  source→binary→build→kernel(+launch)** programs in one context without error —
  so "compile many, free unused" does work; there is no driver build cap.
- Instrumenting the search shows the first anomaly is **not** a build: a
  candidate *launch* fails at `clWaitForEvents` (`UNKNOWN_ERROR`). After that the
  context is poisoned: every later `clBuildProgram` returns `UNKNOWN_ERROR`
  (empty log), and the driver sometimes segfaults.
- **Actual root cause (fixed): `add_gpudims` dropped the `pm_split_ranges` split
  marker.** tinygrad keys every RANGE by `x.arg[0:-1]` (the head *excluding* the
  trailing AxisType), so the two halves of a split range get distinct keys
  `(axis,0)`/`(axis,1)` (`gpudims.py:47/84`). The port keyed by
  `(range_axis_id, axis_type)`, collapsing both onto one GPU dim. For the RNG
  `stack`/`cat` kernel (`rand` of a `[32,32,5,5]` weight init) the split
  `[2,12800]` base rendered `data0_25600[gidx1*12800 + gidx1]` instead of
  `[... + inner]`; a BEAM `UPCAST axis=1 arg=2` then widened the split range to
  6400, so `gidx1` (6400) × (12800+1) reached ~82M into a 25600 buffer — the
  illegal access that faults at `clWaitForEvents` and poisons the context.
  `hand_coded_optimizations` happened to repair the index, which is why non-BEAM
  training and `tests/rand.rs` were always correct. The fix keys by the
  `arg[0:-1]` head (shapes/dims derived from those heads), matching tinygrad; the
  RNG kernel now searches cleanly (its `[2,12800]` UPCAST candidates don't fault).
- **Pinned by a new differential gate** `test_pm_add_gpudims`
  (`py/tests/test_rewrite_diff.py` + `oracle_rewrite`): 300 random graphs mixing
  unsplit and split GLOBAL ranges. Fails on the old keying (both halves map to one
  `SPECIAL`), passes with the fix.
- `sval_render` now renders `Const(Bool)` as `True`/`False` (as `UOp.render`
  does); it previously panicked in the advanced-indexing gather's
  `get_optimized_ast` name generation.
- The in-memory `CLCompiler.compile_cache` was a genuine unbounded leak (tinygrad
  uses a *disk* cache, `device.py Compiler.compile_cached`); it is now **FIFO-capped**
  (`CL_COMPILE_CACHE`, default 1024) so BEAM can't grow it without bound.

**Second cause (fixed): uninitialized BEAM dummy buffers.** After the gpudims fix
the advanced-indexing gather `X[samples]` (`[64,784]`) still faulted — timing-
dependently. Its candidate is in-bounds *for valid inputs*, but `rawbufs_from_ast`
allocates dummy buffers that upstream leaves uninitialized (`BufferSpec.zero=False`):
the gather uses its dummy inputs (`data1/2/3`) to compute `X`'s row, so recycled
garbage sent `data4_200704[huge*784+gidx0]` out of bounds. tinygrad tolerates this
by running each candidate in a worker subprocess; the single-process port can't, so
the fault poisoned the CL context. Fix: **zero the dummy buffers**
(`opt_search::rawbufs_from_ast`); `BEAM_UNINIT_BUFS=1` restores upstream's
uninitialized dummies (safe only with the worker pool). Also made `ClProgram` keep
its `cl_program` alive (upstream `CLProgram.program`; the port had released it
immediately), and `sval_render` now renders `Invalid` too.

**BEAM on CL now completes:** `DEFAULT_DEVICE=CL BEAM=1 BS=64 STEPS=1 TRAIN_N=256`
runs a full step (`loss 2.658 test_accuracy 10.94%`, 0 faults, 119 misses/94 hits).
The worker pool remains the robust isolation for a *genuine* OOB candidate.

Still missing vs tinygrad: the **worker pool** (which in tinygrad isolates a
faulting candidate from the training process — the single-process port now zeroes
the dummy buffers instead, which covers index-driven faults but not a genuine OOB),
the `TC/SWAP/THREAD/PADTO/NOLOCALS` actions, and the `estimates.lds/mem` fields.
BEAM is now usable on CL (`xor` and full MNIST); the default
`hand_coded_optimizations` path is unchanged and the `pool` fix (§8.19) is the real
throughput win.

### 8.21 Dataset re-upload — the MNIST step bottleneck (DONE)
After the `pool` fix and the bf16 work, a steady-state step was still ~1.5 s at
BS=512, and `time` showed `user+sys ≈ wall` — the port was **CPU-bound**, not
GPU-bound. Instrumenting the realize path (temporary `TG_PROFILE`) split a step:
`callify` ~18 ms, `create_linear_with_vars` (rangeify/schedule) ~410 ms,
`run_linear` (exec) ~710 ms — and inside exec, `exec_copy` was **~700 ms** while
all kernel launches totalled **~9 ms**. The copies were 14/step for ~282 MB, of
which **6 × 47 MB**: the full 60000×784 uint8 train set, `PYTHON -> CL`.

Cause: tinygrad's `nn.datasets.mnist` returns tensors realized on the default
device (`Tensor.from_url` uploads the file once), so later steps index resident
storage. The port's `load_images`/`load_labels` returned **lazy** tensors; the
`from_slice` `PYTHON -> device` COPY is faithful (tinygrad's `Tensor(list,
device=NV)` is the same COPY), but a lazy dataset tensor re-runs that whole-file
COPY on every use — and the training step realizes the forward six times (the
`BatchNorm.forward` running-stat `assign`s), so 6 × 47 MB/step.

Fix (`examples/mnist.rs`): `load_images`/`load_labels` freeze each dataset once
with a device-backed leaf. First done as `realize_fresh(...).realize_to_leaf(...)`
(§8.21 commit `9b99220`); §8.22 replaces that with the faithful `realize`, which
now uploads into the realized buffer without the host round-trip. **BS=512 steady
state ~1.53 s/step -> ~0.80 s/step** (sys 10 s -> 0.8 s
per 15 steps); FAST preset ~0.35 s/step; losses/accuracy identical.

**What a step costs now** (BS=512): scheduler (`create_linear_with_vars`/rangeify)
**~430 ms**, callify ~18 ms, device exec ~15 ms, copies ~0. The port is now
**scheduler-bound**.

### 8.22 Faithful `realize` (uop → realized buffer) — DONE
`Tensor.realize` now replaces the tensor's uop with its realized-storage form
(tinygrad `_apply_map_to_tensors`, tensor.py:248), so later steps read the buffer
directly — no per-step re-run, and **no `realize_to_leaf` host round-trip**
(important for large datasets).

- **Model (commit `b5cee2b`).** `Tensor.uop` is now a shared `Rc<RefCell<UopId>>`
  with `Tensor::uop()`/`set_uop`, so an in-place replacement is seen by all clones.
- **Wiring.** `realize_tensor`/`realize_many` return `(Buffer, storage_uop)`;
  `Tensor::realize`/`to_bytes`/`realize_many` call `set_uop` to a new leaf that
  **aliases** the realized allocation (`BufTable::alias` — weak view only, or one
  base leaks per realize). `load_images`/`load_labels` now just `realize`.

**Two faithful callify fixes were needed** (both real port bugs found by the loop
it exposed):
1. `UOpCtx::contiguous_view` must return `None` for **CL/WEBGPU** — "CL does not
   support views" (ops.py:891). Without the guard, `contiguous_mops_to_view` never
   reached a fixed point on CL ("rewrite step limit").
2. `contiguous_view` must return `b.rtag(None)` (ops.py:895) — strip the tag. The
   port returned the tagged uop, so each `contiguous_mops_to_view` rewrite produced
   a structurally new copy and the rewrite never converged (the CPU/INTERP loop
   hit via `schedule_step` re-realizing a now-leaf loss).

**Result (CL, BS=512):** steady state ~0.75 s/step; 15-step acc **76.56%** vs
71.09% with the old `realize_to_leaf` (the CL view guard changes CL codegen to the
faithful path). `realize_to_leaf` remains, but is no longer needed for datasets.
`cargo test` 113 pass; `--features backend_cl` unchanged (the 2 bool-vector CL
failures are pre-existing §9.6.1); all 41 py differential gates pass.

Remaining nuance: realized storage is still resolved through the per-session
`BufTable` (aliased), so it is session-scoped; attaching the `Buffer` to the UOp
(tinygrad's global `buffers` WeakKeyDictionary, ops.py:214) would make it
arena-independent — a follow-up, not needed for the dataset win.

---

### 8.23 Device-graph RNG (threefry) — DONE (JIT prerequisite)

tinygrad's `Tensor.rand`/`randint` are *device graphs*: uint32 Threefry +
bit-level mantissa assembly (`mixin/rand.py`), lowered by
`codegen::decomp::threefry2x32` (no hardware runs THREEFRY natively). That is why
a captured/replayed TinyJit step regenerates fresh samples. The port had
reproduced the sequence on the host and materialized it with `from_slice`, so a
replayed graph would re-upload the *same* bytes (one fixed batch).

Ported the graph route:
- `rand_graph` / `random_bits_graph` / `bits_to_rand_graph` (rand.py:12/18/31/39),
  built on the already-present `Ops::THREEFRY` + `Tensor::threefry` +
  `codegen::decomp::threefry2x32`.
- `randint` is now `uniform(low, high, dtype)` (rand.py:122) — the faithful
  definition — so it is a pure graph on hardware rather than a host `from_slice`.
- Two supporting fixes found by running it:
  1. **`uop::sym` unpack rule was reversed.** The port's
     `((y:u32<<32)|z:u64).cast(u32) -> y` named the *high* word; tinygrad
     (symbolic.py:179) returns the *low* word (`z`). Fixed to match.
  2. **INTERP had no `Bitcast`.** Added byte-reinterpretation via
     `to_storage_scalar`/`from_storage_scalar` (int↔float mantissa assembly).

INTERP still carries f64 lanes, so the uint64 pack `(counts1<<32)|counts0` is not
exact there; `rand` dispatches to the host evaluator (`rand_host`) on INTERP and
to the graph everywhere else. **`tests/rand.rs` (8 tinygrad goldens) passes on
INTERP and on CL** (`DEFAULT_DEVICE=CL cargo test --release --features backend_cl
--test rand`), i.e. the graph is bit-exact on hardware. `cargo test` 113 pass;
clippy clean.

Device counter: with JIT on, `_next_counter` is the device-side version
(tensor.py:642) — seed/counter are device buffer leaves and the counter is
advanced by a `STORE+AFTER` inside the using graph, so replay moves it. With JIT
off, `rand_graph` stages the same values as host constants (no per-step
`After`-chain growth). The device counter is seeded from the host seed and starts
at the host counter's current value, so enabling JIT after model init continues
the non-JIT sequence (pinned by `tests/jit.rs::jit_rng_continues_from_host_counter`).

Residual divergence (a step, not a draw): the port's `BatchNorm` running-stat
`assign`s realize the forward several times per step, and the in-graph RNG store
rides along, so the device counter advances once per realize rather than once per
step. Over a training run the JIT sequence therefore drifts from non-JIT (values
remain valid draws). Exact parity needs tinygrad's deferred `assign` (store only,
realized once) or a JIT input-refresh path, tracked in §9.6.3.

---

### 8.24 Session opt-in JIT (TinyJit) — DONE

`engine/jit.rs` ports the core of `engine/jit.py`: `Jit::call(ctx, sess, f)` runs
`f` once to **capture** every planned LINEAR the step realizes (the session's
`capturing` hook, jit.py:255), holds all touched buffers, then **replays** those
linears instead of rebuilding the graph. `Jit::reset` recaptures. Public API:
`Session::jit(bool)` / `enable_jit()` / `jit_enabled()` (the user-requested knob),
plus a `JIT=1` preset in the MNIST example. With `jit(false)` the wrapper runs `f`
directly (`if not JIT`, jit.py:244).

Why a wrapper and not only session-level: tinygrad JITs a *function* because the
step boundary and the input set come from the decorated call — the MNIST example
has two independently-jitted functions (`train_step`, `get_test_acc`), which a
single session flag cannot distinguish.

Supporting changes:
- `Session.capturing: Option<Vec<UopId>>` + `begin/end_capture`; the realize
  functions take a `capture` out-param and push each planned linear.
- `Jit::call` realizes the captured return (`get_parameters`/`Tensor.realize`,
  jit.py:258) so a lazy eval graph captures rather than erroring.
- `optim::schedule_step` no longer `realize_to_leaf`s grads (that read grad bytes
  to the host and baked them into the capture — forbidden during JIT, and
  redundant now that `realize_many` freezes each root to its buffer leaf).

Validation (CL, BS=512, TRAIN_N=1024, TEST_N=256, 21 steps):
| | wall | test acc |
|---|---|---|
| JIT off | 18.01 s (~0.86 s/step) | 82.81% |
| JIT on  | 8.84 s (~0.42 s/step)  | 83.20% |

`tests/jit.rs` covers capture/replay param updates on INTERP and the device-RNG
advance on CL. `cargo test` 116 pass; 41 py gates pass; `--features backend_cl`
unchanged (2 pre-existing bool-vector failures, §9.6.1).

Known gaps / next: **compiled-linear replay — DONE** (replay executes stored
`CompiledLinear`s; `compile_linear` runs once at capture, `to_program` lookups
per 21 steps fell ~4300 → ~330). **Device-counter seed — DONE** (§8.23): the
device counter starts from the host counter so an isolated JIT draw continues the
non-JIT sequence; per-*step* parity still drifts because the port's eager BN
`assign`s realize the step several times (documented in §8.23). Buffer `prune` is
bypassed by `Jit::held` (the captured buffers are held strongly, so the session's
`prune` keeps them) — moving that into a session-owned arena is cosmetic.

---

### 8.25 `nn` layer coverage completed (DONE)

All layer types in `tinygrad/nn/__init__.py` are now ported to `src/tensor/nn.rs`,
each gated by `tests/nn_layers.rs` against goldens generated by `py/dump_nn_golden.py`:

- **Norm family** — `InstanceNorm` (reshape→layernorm→per-channel affine, like GroupNorm
  with `num_groups = num_features`), `LayerNorm2d` (permute `NCHW→NHWC`, inner `LayerNorm`,
  permute back), `RMSNorm` (normalize in f32, cast back, optional affine).
- **Embedding** — the non-atomic `_embedding_fwd`
  (`(arange(vocab)==idx.unsqueeze(-1)).unsqueeze(-1).where(weight,0).sum(-2)`); the
  `USE_ATOMICS` custom-kernel path is not needed for forward.
- **LSTMCell** — `weight_ih.T`/`weight_hh.T` linears + `chunk(4,1)` gating
  (`sigmoid/sigmoid/tanh/sigmoid`), `new_c = f*c + i*g`. Required porting
  `Tensor.chunk`/`split` (`movement.py:424-476`, `shrink`-based).
- **Conv variants** — `Conv1d`/`ConvTranspose1d` (the `(k,)`-kernel wrappers) and
  `ConvTranspose2d`. `Conv2d` was generalized from a 2-D-only layer to one kernel entry per
  spatial dim (matching tinygrad's `make_tuple(kernel_size, 2)`), so a 1-tuple is Conv1d.
  This needed a faithful `Tensor.conv_transpose2d` (`mixin/op.py:1607-1650`): unflatten the
  weight to `(groups, ·, ·)`, transpose/flip, optional stride zero-insertion
  (`reshape→pad→reshape→shrink_to`), reversed per-dim padding, then `conv2d` (with the
  `unflatten`/`shrink_to`/`chunk` movement helpers added alongside).

Gate: `tests/nn_layers.rs` (7 tests — InstanceNorm, LayerNorm2d, RMSNorm, Embedding,
LSTMCell incl. the `hc=None` path, Conv1d, ConvTranspose1d+2d), all vs explicit-weight
tinygrad goldens.

### 8.26 Training mode + dropout (DONE)

`TRAINING` (helpers.py:237) is a thread-local flag (`src/helpers/context.rs`,
mirroring the `rand::set_jit` pattern) with a `TrainingGuard`/`with_training`
scope, plus `Session::training(bool)` / `training_enabled()` mirroring
`Session::jit`. `Tensor.dropout(p=0.5)` (`mixin/rand.py:274-296`) is ported
(training-only inverted dropout `(rand_like(default_float) >= p).where(self,0)/(1-p)`);
`Tensor.rand_like`/`const_like` added. Gate: `tests/nn_dropout.rs` — invariants
plus **exact RNG parity** vs tinygrad (`py/dump_dropout_golden.py` →
`tests/dropout_golden.rs`).

### 8.27 Attention/ViT primitives: cat(dim), tril/triu, gather, clip, lerp, repeat_interleave (DONE)

For the ViT/attention work, `Tensor` gained the movement/op primitives tinygrad's
`mixin/op.py`/`mixin/elementwise.py` provide: a general `cat(*others, dim)`
(equal-size fast path `stack(dim)+flatten`, else cumulative-pad + add — the
oracle's `usum`), `tril`/`triu`, `gather(dim, index)`, `clamp`/`clip`, `lerp`,
and `repeat_interleave`. Gate: `tests/nn_tensor_ops.rs` vs tinygrad goldens
(`py/dump_tensor_golden.py` → `tests/tensor_golden.rs`). The RNG callers of `cat`
now pass `dim=0` explicitly. Also added `floordiv`(`//`)/`mod_(`%`)` (int int →
`Ops.FLOORDIV`/`Ops.FLOORMOD`) and a faithful `Tensor.interpolate(size, mode,
align_corners)` (op.py:1046-1078: linear via `gather`+`lerp`, plus
`nearest`/`nearest-exact`) — used to resize MNIST 28→224 for the ViT. Goldens
cover bilinear up/down, `align_corners`, and both nearest modes.

### 8.28 Attention: `scaled_dot_product_attention` (DONE)

Ported `Tensor.scaled_dot_product_attention(key, value, attn_mask, dropout_p,
is_causal, enable_gqa)` (`mixin/rand.py:298-327`): f32-accumulated `q·kᵀ/√d`
(`dot_dtype` with `least_upper_dtype(q,k,f32)`), `is_causal` via a bool `tril`,
bool→`0/-inf` / additive masks, GQA `repeat_interleave`, softmax, dropout, `@v`.
Gate: `tests/nn_attention.rs` vs tinygrad goldens (`py/dump_attention_golden.py`
→ `tests/attention_golden.rs`): basic, multi-head, causal, bool/float masks, GQA.
**Bug fixed:** `tril`/`triu` built their `0` branch with a weak-int const instead
of `x`'s dtype, so `tril` of a bool tensor returned int — now they use
`ctx.dt_of(&x)`, matching tinygrad's `self.const_like(0)`.

### 8.29 Transformer + ViT modules (DONE)

`src/models/transformer.rs` (`TransformerBlock`, `Transformer`, `Act`) and
`src/models/vit.rs` (`ViT`) mirror `extra/models/transformer.py` /
`extra/models/vit.py`. `ViT::new(layers, embed_dim, num_heads)` reproduces the
tinygrad defaults; `ViT::build(...)` takes the input shape explicitly for MNIST.
Gate: `tests/nn_models.rs` vs tinygrad goldens (`py/dump_models_golden.py` →
`tests/models_golden.rs`) — **seed parity** (the port's RNG matches, so the same
seed reproduces tinygrad's random init) across `TransformerBlock` and a small
`ViT`. **Two bugs fixed:** (1) `Tensor.linear` with a 1-D weight ignored the
weight — now `self.mul(weight) + bias` like tinygrad; (2) the prenorm branch
applied the FFN activation to `n2` instead of `n2·ff1`
(`act(x.layernorm().linear(ln2).linear(ff1)).linear(ff2)`).

### 8.30 MNIST ViT training example (DONE)

`examples/vit_mnist.rs` trains the ported `ViT` on MNIST, mirroring
`extra/models/vit.py` + the `beautiful_mnist.py` loop. MNIST is normalized to
[0,1], bilinearly resized (`Tensor.interpolate`) to `SIZE`, and 1→3 channels
(`repeat_interleave`). Defaults to a small ViT (`LAYERS=2 EMBED=64 HEADS=4
SIZE=32 PATCH=8`); `LAYERS=12 EMBED=768 HEADS=12 SIZE=224 PATCH=16` gives
ViT-B/16. Uses the `Session::training` switch: dropout is active for train steps
and the identity for eval. Gates: `tests/vit_e2e.rs` (ignored, `#[path]`-includes
the example: one train step + param update + eval) and a real run that learns
(loss 2.3→1.35, ~59% test acc in 40 steps on INTERP). Run:
`cargo run --release --example vit_mnist` (`FAST=1` for a quick preset).

### 8.31 FlashAttention (op-level) (DONE)

`src/models/flash_attention.rs` ports the **FlashAttention/FA2 algorithm** from
`extra/thunder/tiny/fa.py`: Q and KV block tiling with an **online (running)
softmax** (running max `m`, denominator `l`, accumulator `acc`, rescaled per KV
block), so the `N×N` score matrix is never materialized. Supports causal masking
(lower-triangular), bool/additive `attn_mask`, **GQA** (inferred from
`H // H_KV`, like fa.py's `head_kv = head // GROUP_SIZE`), 3-D→4-D promotion, and
configurable block sizes (`flash_attention_blocks`) using **variable-size final
Q/KV blocks (no padding)**. It is numerically identical to
`scaled_dot_product_attention` and backprop flows through the ops, so it doubles
as the FA backward.

`FA=1` (read once in `TransformerBlock::attn`) routes the model's attention
through the FA port instead of SDPA; the ViT-MNIST example trains with it.
Gate: `tests/nn_flash_attention.rs` — forward **and backward** match SDPA for
basic/multi-head/causal/bool+float masks/GQA, with small blocks to force
tiling + remainders + padding, and a default-block/3-D-promote case.
**Bug fixed:** `Ops::Stack` had no gradient rule — added it
(`gradient.py`: STACK's grad is `ctx[i]` along the new dim 0), which `cat`
uses on its equal-size fast path. The fused `tk` warp-DSL kernel
(`Tensor.custom_kernel`/CALL) remains the deferred M7 stretch (see above); this
is the same algorithm expressed with `Tensor` ops.

### 8.32 Attention on all backends: two bugs found + fixed (DONE)

Getting FA to run correctly on **INTERP, CL and CUDA** surfaced two bugs:

1. **CL/CUDA crash on FA graphs** (fixed). A kernel's used buffer slots can have
   gaps (`globals = [0..9, 11]`), but `exec_kernel` built a dense
   `globals.len()` `bufs` vec while CL/CUDA index `bufs[slot]` (tinygrad's
   contract), so a gap slot read out of bounds (`ops_cuda.rs:271`). Fix: build
   `bufs` **slot-indexed** (length `max_slot+1`, gap slots filled with a clone of
   a live view — never referenced); `max_slot+1` is also `paramarg_start`, the
   val-arg offset. INTERP updated to index `bufs[slot]` too. Files:
   `src/engine/realize.rs`, `src/runtime/ops_interp.rs`.

2. **FA wrong at large KV-padded shapes** (fixed). FA disagreed with SDPA by
   ~1e-3..0.7 for `H>1` with KV padding, *identically on all three backends*
   (so graph/port, not backend). Bisected: `H=1` and every `kvpad=0` shape were
   exact even when large; only KV padding broke, above ~2048 padded elements.
   `pad`/mask construction round-trip correctly on their own. Fix: rewrite
   `flash_attention_blocks` to use **variable-size final Q/KV blocks** (static
   block bounds, slice the `(nq,nk)` mask directly — no padding at all; also
   closer to real FA). After the rewrite **every tested shape matches SDPA to
   ~1e-6**, incl. `H4/S129`, `H4/S197`, `H1/S257`, `H2/S500` and the ViT-B/16
   attention shape `H12/S197/D64` on INTERP, CL and CUDA. The underlying
   "padded tensor feeding downstream ops at ~2048+ elems" port bug is **bypassed,
   not root-caused** (possible latent scheduler issue — follow-up).

**Measured on CUDA** (RTX 3060, ViT-B/16 attn `BS1 H12 N197 D64`, `examples/fa_bench`):
SDPA fwd 1.08 ms vs FA fwd 1.49 ms (kernel, JIT replay); SDPA fwd+bwd 7.3 ms vs
FA 87 ms. End-to-end per step (graph scheduling included) FA is ~18x/42x slower.
The op-level FA is **not faster** — it is not a fused kernel; the fused `tk`
kernel (`Tensor.custom_kernel`/CALL) is the deferred M7 stretch.

## 9. Next steps

Ordered by leverage on the training path. Items 1 and 2 are the two halves of the same
problem (persistent, collectible state) and should be designed together.

### 9.1 Buffer management via `Rc` — DONE
See §8.8 for the full write-up. Summary: `Tensor` owns its realized `Buffer` (`Rc`, shared
by clones, reused on repeat reads); `BufTable` stores views as `Weak` and prunes any base no
live caller references at the start of each realize; `Session` bundles `BufTable` +
`ProgCache` + `var_vals` and is reusable across steps. The
content-keyed `ProgCache` makes compiled kernels survive a fresh `UOpCtx` per step.
Residual (DONE): staged bytes and grads now live on the node — see §8.8/§9.2.

### 9.2 `Tensor` / UOp management via `Rc` — DONE
`UopId` is a strong handle (`Rc<Node>`); `Node.srcs` holds strong child handles; the
interner is `HashMap<u64, Weak<Node>>`; property caches live on the node. `Tensor` is
`Clone` (not `Copy`). Nodes free automatically when the last handle drops — no GC pass, no
safe-point, no id reuse. The engine's `UopId` currency changed from a `Copy` index to a
`Clone` handle, which rippled through ~2k call sites (mostly mechanical `.clone()` /
`&`-borrow fixes, driven by compiler suggestions). `Tensor`'s buffer/drop semantics are the
only behavioural change; the rewrite differentials (see §8.0) are unchanged.

**Residual — DONE.** The staged `initial_data` bytes and per-tensor autograd `grad` are now
per-node (`Node::{initial_data,grad}`), so they die with the tensor's subgraph rather than
growing ctx-wide `u32`-keyed maps. `Tensor`'s buffer/drop semantics remain the only
behavioural change; the rewrite differentials (see §8.0) are unchanged.

### 9.3 Completing nn
**Status: layer coverage DONE (see §8.25).** Every `nn/__init__.py` layer below is ported
with a tinygrad golden; what remains from this list is `State`/safetensors and the
data path. The original plan follows.

1. **`Conv2d::forward`** — port tinygrad `mixin/op.py` conv2d verbatim: `_pad_constant` →
   `_pool(HW, stride, dilation)` → `reshape(bs,groups,cin,1,*oyx,*HW).expand(...,rcout,...)`
   → `permute(0,1,3,*oyx_indices,2,*HW_indices)` → `mul(weight.reshape(...))` →
   `sum(reduction dims, keepdim=True)` → `reshape(bs,cout,*oyx)` → `+ bias`. `pool` and
   `pad` already exist.
2. **Norm layers, faithful**: BatchNorm / LayerNorm / GroupNorm running-stat updates as
   in-graph **assign** in train mode, buffered stats in inference, affine `weight`/`bias`;
   add `InstanceNorm`, `RMSNorm`, `LayerNorm2d`.
3. **`nn/state.py` equivalent** — **DONE (§9.9)**: `State`, safetensors reader/writer,
   `get_state_dict`/`load_state_dict` (buffers included).
4. **Optimizers** (`nn/optim.py`) — **DONE (§9.5)**: LARS/LAMB/Adam/AdamW + in-place
   `Tensor::assign`.
5. **Remaining ops needed by nn** — **DONE**: `max`/`min`, `argmax`, `flatten`,
   `interpolate`, `Embedding`, `LSTMCell` (§8.25).
6. **Data path**: descoped; `examples/mnist.rs` shells out to `curl` once and reads
   the IDX files directly.

### 9.4 Testing nn
1. **Golden harness**: add `py/dump_golden.py` cases and `tests/nn_{linear,conv2d,batchnorm,
   optim}.rs` asserting forward/step numerics against tinygrad-interp goldens. Extend
   `tests/backend_harness.rs` beyond the current ignored XOR placeholder.
2. **Port upstream test cores**: `test/backend/test_nn.py`, `test/backend/test_optim.py`,
   and `test/unit/test_gradient.py` (fuller than the current 6 gradchecks).
3. **Un-ignore parity**: `nn_xor_parity::xor_parity_tinygrad_seed42` should bit-match
   tinygrad at seed 42 — needs Adam + faithful init + persistent buffers (§9.1) to avoid
   re-initialization and non-determinism.
4. **M7 acceptance gate — MNIST e2e**: an MLP train step and a Conv inference pass, weights
   loaded via `State`, optimizer-step numerics diffed against tinygrad. This is the
   milestone that exercises custom kernels, rand, gradient, and conv ops together.

### 9.5 Agreed MNIST-session plan (decisions to implement)
The session goal is the M7 MNIST e2e, built as `autorust/examples/mnist.rs` (not a
default-suite test; INTERP training is slow), mirroring
`tinygrad/examples/beautiful_mnist.py`. Work order:

1. **Optimizers (LARS/SGD/Adam/AdamW/LAMB)** — port `nn/optim.py` one-for-one.
   **DONE**: `LARS`/`LAMB`/`Adam`/`AdamW` + reworked `Optim` trait
   (`_step`/`schedule_step`/`step`), `Tensor::zeros_param`, `Session::realize_many`,
   `Tensor::{gt,lt,ne,where_}`, and `tests/nn_optim.rs` goldens. `step_graph` is kept
   for the stateless parity tests. Remaining: none (Muon/newton-schulz descoped).
2. **conv2d + pooling/loss** — full ports of `Tensor.conv2d`
   (`mixin/op.py:1595-1607`) incl. `_pad_constant`/`flat_to_grouped`/`resolve_pool_pads`,
   `Tensor.max_pool2d` (`return_indices`+`ceil_mode` included), `Tensor.sequential`,
   `Tensor.sparse_categorical_crossentropy` (`ignore_index`/`label_smoothing`/`reduction`),
   `Tensor.argmax`. Gate against tinygrad goldens.
   **DONE except `sequential`**: conv2d + max_pool2d + one_hot + argmax + eq +
   `sparse_categorical_crossentropy` are in `src/tensor/op.rs` and green
   (`tests/nn_conv.rs`, 4/4). `sequential` is deferred to the Phase-3 model struct. The
   sibling-reduce scheduler bug that blocked CE is fixed (§8.12), so MNIST is unblocked.
3. **Module trait adoption** — implement `Module` for Conv2d/BatchNorm/LayerNorm/GroupNorm,
   plus `impl<T: Module> Module for Vec<T>`. Add a generic `into_params` so examples don't
   hand-list params. Rewrite `examples/xor.rs` and the XOR tests to use `Module`.
   **DONE** (§8.14): all four layers + `Vec<T>` implement `Module` with `into_params`;
   `XorModel` structs replace the hand-listed params. `nn::Sequential` was cut (models define
   `forward` themselves). Fixed `LayerNorm::forward` (`x*weight+bias`). BatchNorm/GroupNorm
   were made faithful in Phase 5 (§8.16).
4. **Tensor indexing (new prerequisite)** — `Ix` has no Tensor variant today. Port
   `_getitem`'s advanced path (`mixin/op.py:74-146`): negative-index normalization,
   broadcast indices, `_one_hot_along_dim` masks, `uprod` + `where` + `sum`, and the
   consecutive-int-dim linear-index fast path; out-of-bounds → 0. Differential-test vs
   tinygrad. Needed for `X[samples]`/`Y[samples]`.
   **DONE** (§8.15): `Ix::Tensor`, the full advanced path, numeric gates
   (`tests/indexing.rs`) and graph-key gates (`py/tests/test_tensor_diff.py`).
5. **MNIST example** — same layer list as `beautiful_mnist.py`; data loader shells out to
   `curl` via `std::process::Command` (single `sh -c 'curl … | gunzip > file'`, skipped if
   present) into a cache dir; IDX parsing in Rust; minibatched with `randint` + advanced
   indexing; Adam default; `TARGET_EVAL_ACC_PCT` check like tinygrad. Add one `#[ignore]`d
   `tests/mnist_e2e.rs` acceptance gate (small subset + conv goldens).
   **DONE** (§8.16): `examples/mnist.rs` + the ignored `tests/mnist_e2e.rs`. It exposed and
   fixed the reshape/batchnorm/flatten/PAD-SHRINK-grad issues and the `rngs` sort /
   load-collapse / gpudims / `group_dims` / GROUP_REDUCE bugs listed there.
6. **Docs** — keep this file current; `AGENTS.md` at the repo root states the faithful-port
   and end-user-feel rules and must be edited if its facts change.

**Deferred → now planned:** a faster CPU backend. See **§9.8** for the `ops_rayon`
design (chosen name/device: `RAYON`, feature `backend_rayon`, optional dep `rayon`). The
interpreter stays as the faithful debug/reference backend (`INTERP`); `RAYON` is a
sibling, not a replacement. A cstyle→`cc -shared` native backend remains the alternative
that needs no third-party dep but does need a C compiler at runtime.

### 9.6 Post-MNIST backlog (2026-10)
Discovered while getting the CL MNIST run working (§8.17). Roughly ordered by leverage on
GPU training throughput; items are independent.

1. **BF16 on cstyle/CL — DONE (§8.18).** `extra_matcher` (`create_non_native_float_pats`
   + `pm_manual_bf16_cast`), OpenCL `as_*` bitcasts, bf16 const bit patterns, and sub-word
   `rand` are ported; `tests/bf16.rs` passes on CL and a full bf16 CNN train step now runs
   (`DTYPE=bfloat16`). It is ~2x slower than f32 (emulated through f32), so f32 stays the
   example default. Still open: sub-dword vectorized coalescing bitcasts — the backend
   harness skips sub-dword outputs on cstyle backends and
   `dtype_battery_all_backends`/`alu_code_for_op_all_backends` fail on CL for bool vectors
   (pre-existing). The Clang `supported_dtypes` bf16 arch gate is not ported (bf16 stays
   "supported" on clang regardless of arch).
2. **BEAM search — core ported + hardened (§8.20); worker pool still missing.**
   `src/codegen/opt_search.rs` runs under `BEAM=n`; it explores
   UPCAST/UNROLL/LOCAL/GROUPTOP/GROUP, compiles each candidate and times it.
   Hardened with a **disk cache** (search runs once per kernel/width/device),
   `BEAM_UOPS_MAX`, `seen_libs` dedup, an `Estimates.ops` compute filter,
   `BEAM_STRICT_MODE`/`BEAM_VERBOSE`, and a port-only `BEAM_MAX_CANDIDATES` cap.
   Still missing: the **worker pool** (subprocess isolation of candidate
   compiles — the actual fix for the NVIDIA driver aborting the process on a
   large-model search), TC/SWAP/THREAD/PADTO/NOLOCALS actions, and
   `estimates.lds/mem`. See §8.20.
3. **`TinyJit` — DONE (§8.24).** `Session::jit(bool)` + `engine/jit.rs` capture/replay the
   planned LINEARS of a step; the device-graph RNG (§8.23) advances the sampler on replay.
   CL BS=512 went ~0.86 → ~0.42 s/step (21 steps: 18.0 → 8.8 s, acc unchanged). Follow-ups:
   store the compiled linear (trim per-replay `compile_linear` lookups), move buffer
   `prune` into the session-owned arena, and reconcile the device-counter seed with the
   pre-JIT host counter.
4. **Complete the `nn` layer types** (`nn/__init__.py`): missing `ConvTranspose2d`,
   the `Conv1d`/`ConvTranspose1d` helpers, `InstanceNorm`, `LayerNorm2d`, `RMSNorm`,
   `Embedding`, `LSTMCell`. Ported so far: `BatchNorm`, `Conv2d`, `Linear`, `GroupNorm`,
   `LayerNorm`.
5. **`nn/state.py` + safetensors** (later): `State`/`state_dict`/`load_state_dict`, a
   std-only safetensors reader/writer, `init_weight`/`fetch`. Today only
   `Module`/`named_params`/`into_params` exist.
6. **Native GPU backend — CUDA + WMMA DONE (2026-10).** `backend_cuda` uses
   `cudarc` (dynamic-loaded libcuda/NVRTC) for `ops_cuda.py` + the CUDARenderer;
   native bf16 MNIST runs (`DEFAULT_DEVICE=CUDA DTYPE=bfloat16`) and the WMMA
   tensor-core path is ported (a 16x16x16 bf16 matmul matches tinygrad's CUDA
   output exactly). See §9.7 for the design and the MNIST timing table.

### 9.7 Native GPU backend (CUDA/NV) — port plan for native bf16 (2026-10)

**Why.** CL computes bf16 by emulation (`pm_manual_bf16_cast` + compute-through-f32,
§8.18) and `OpenCLRenderer.tensor_cores == []`, so bf16 is ~2x *slower* than f32. On
NVIDIA the native path is `CUDARenderer`, which emits `nv_bfloat16` and sets
`tensor_cores = tc.get_cuda("sm_86")` (bf16 WMMA). That renderer, not the device name, is
what "unlocks bf16".

**Which "NV".** tinygrad has two NVIDIA backends:
- `CUDA` (`runtime/ops_cuda.py`, ~250 LOC) — the CUDA **driver API** (`libcuda.so`,
  `cuInit`/`cuMemAlloc`/`cuLaunchKernel`), compiled by `NVRTCCompiler` (`libnvrtc` →
  PTX → `cuModuleLoadData`). Small surface; dynamic loading. **Recommended target.**
- `NV` (`runtime/ops_nv.py`, 55 KB) — a from-scratch **userspace GPU driver**: ioctl on
  `/dev/nvidia*`, `support/hcq.py` command queues/signals/MMIO, `support/nv/nvdev.py`
  memory manager, plus the ~1.5 MB `nv_5xx` register/ioctl autogen. *Not* a dlopen
  backend; a multi-month port. Its only relevant edge is **kernel-launch latency** (QMD
  written straight into the GPFIFO + doorbell, vs the CUDA driver's per-launch arg
  marshalling) and low-level control (MMIO/PMA). It is **not** toolkit-free: `PTXRenderer`
  selects `NVPTXCompiler` for the `NV` device, which needs `libnvJitLink` (a *toolkit*
  library) — so both backends need a toolkit lib to compile (`libnvrtc` vs `libnvJitLink`).
  For bf16 matmul/conv throughput this edge is irrelevant; it would only matter for a
  launch-bound tiny-kernel workload.

Both select `CUDARenderer` on sm_80+, so bf16 is unlocked by either. This plan is the
CUDA driver-API path; the renderer/codegen work is shared.

**Moving parts**
1. **Bindings** (`src/runtime/cuda_ffi.rs`): ~40 CUDA driver fns + opaque typedefs +
   `CUresult` / `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` / `CU_MEMORYTYPE_DEVICE`;
   ~11 `nvrtc*` fns. No `cuGraph*` (graph capture is SKIP).
2. **Loader** (`src/runtime/dl.rs`): std-only `dlopen`/`dlsym`/`dlclose` wrapper
   (`extern "C"`, link `dl`) returning typed fn pointers — mirrors tinygrad's
   `support/c.py DLL`. (Alternative: the `dlopen`/`libloading` crate behind
   `backend_cuda`, as `opencl3` sits behind `backend_cl`.) **Or use `cudarc`**
   (`-F dynamic-loading`): it already wraps the CUDA driver + NVRTC with safe
   `CudaContext`/`CudaSlice`/`load_module`/`launch_builder`, and ships committed
   bindgen-generated `sys` bindings — i.e. it subsumes items 1–2 and most of 4, and
   replaces the autogen question below. Its abstractions differ from tinygrad's
   `device.py` seams, but the `Compiled`/`Allocator`/`Comp`/`Prog` traits already isolate
   `opencl3` the same way. It is a heavier dep than `opencl3` but behind `backend_cuda`.
3. **Renderer** (`renderer/cstyle.rs` + `renderer/mod.rs`): `CStyleLang::cuda()` —
   `kernel_typedef = extern "C" __global__ void __launch_bounds__(...)`,
   `blockIdx`/`threadIdx` workitems, `smem_prefix = __shared__ __align__(16)`,
   `__syncthreads()`, `type_map` bf16→`nv_bfloat16`, the bf16/half/fp8 includes and
   `_nms` vector prefix, the `htrunc/hsin/...` bf16 ops, native-bf16
   `supported_dtypes` (`ver>=80`), `global_max/local_max/shared_max`. No
   `pm_manual_bf16_cast` (native).
4. **Compiler** (`ops_cuda.rs`): `NvrtcCompiler` (source → PTX, cached like the CL
   `CompileCache`), or a subprocess `nvcc`-based `NvccCompiler` (nvcc + ptxas are present
   here). Cache key includes the device arch (`compile_cuda_sm_86`).
5. **Runtime** (`ops_cuda.rs`): `CUDAAllocator` (`cuMemAlloc_v2`/`cuMemFree_v2`/
   `cuMemcpyHtoDAsync_v2`/`cuMemcpyDtoH_v2`; host staging buffer for async copyin like
   upstream), `CUDAProgram` (`cuModuleLoadData` + `cuModuleGetFunction` +
   `cuLaunchKernel`, `encode_args` building the kernel-param struct), `CUDADevice`.
   Register `"CUDA"` behind a `backend_cuda` feature; add `Renderer::cuda(arch)`.
6. **Device arch**: `cuDeviceComputeCapability` → `sm_86`; both `supported_dtypes` and
   `tc.get_cuda` key off it, so the arch must reach renderer construction (the CL path
   already does this via `Renderer::opencl_arch(exts)`).
7. **Tensor cores** (full bf16 throughput): port `codegen/opt/tc.py` (layouts +
   `get_cuda`), `expand_wmma` (`pm_expand_wmma`/expander2), the `TC` opt in
   `_apply_tc_opt`, and the WMMA asm prefix in `CUDARenderer.render_kernel`. The port's
   pipeline currently says "no WMMA in corpus" and every renderer has empty
   `tensor_cores`, so this is the largest sub-task. Native bf16 (no TC) lands first.

**Autogen through a build script.** tinygrad's `support/autogen.py gen()` uses libclang to
parse C headers and **commits** the generated Python (`cuda.py`, `nv_580.py`); it only
regenerates with `REGEN=1`, never at import time. Options in Rust:
- (a) port `gen()` to `build.rs` via `clang-sys` and emit Rust `extern "C"` decls —
  faithful, but ~600 LOC + a libclang build-dep, overkill for CUDA's ~50 fns.
- (b) `bindgen` in `build.rs` — standard, but a large/unstable build-dep tree and it
  breaks the std-only policy.
- (c) **commit generated/hand-written bindings** (`cuda_ffi.rs`), exactly as tinygrad
  commits `cuda.py`. **Recommended for CUDA.** A `build.rs` is only needed to emit
  `cargo:rustc-link-lib=dl`. Revisit (a)/(b) only for the 1.5 MB `nv_5xx` ioctl tables if
  the `NV` driver is ever ported.

**Staged plan**
- **S0** `dl.rs` + `cuda_ffi.rs` + smoke test: `cuInit` → `cuDeviceGet` →
  `cuDeviceComputeCapability` → `cuMemAlloc` → `cuMemcpyHtoD` → `cuMemcpyDtoH`.
- **S1** NVRTC compile + `cuModuleLoadData` + `cuLaunchKernel`; run the existing f32
  kernels; byte-diff `Renderer::cuda` against tinygrad's `CUDARenderer`.
- **S2** port `CUDARenderer` rendering incl. native bf16; `backend_cuda` feature +
  register `"CUDA"`; `tests/bf16.rs` + backend harness on CUDA.
- **S3** tensor cores (`tc.py` + `expand_wmma` + TC opt + WMMA render) for bf16 matmul.
- **S4** MNIST `DEFAULT_DEVICE=CUDA DTYPE=bfloat16` benchmark vs CL.

**Gate:** CUDARenderer byte-diff (`oracle_codegen`), `tests/bf16.rs` on CUDA, backend
harness parity, and a bf16 MNIST train step. **Open decision:** confirm CUDA vs the full
`NV` ioctl driver, and the glue: std-only `dl.rs` + hand-written `cuda_ffi.rs`, vs the
`cudarc` crate (behind `backend_cuda`) — the same "feature-gated heavier dep" precedent as
`opencl3`. cudarc saves S0–S1 but not the renderer/tensor-core work that actually unlocks
bf16.

**Status (2026-10): S0-S2 DONE.** cudarc + smoke (S0); CUDARenderer facts/cstyle
+ `test_cuda_render_diff.py` byte gate (S1); `ops_cuda` runtime + `"CUDA"`
registration, native bf16/f16, and the MNIST example on CUDA (S2):

```
DEFAULT_DEVICE=CUDA DTYPE=bfloat16 BS=64 STEPS=1 TRAIN_N=256 TEST_N=64
  step 1: loss 2.875  test_accuracy 12.50%   (matches CL)
```

Bugs fixed along the way: cstyle `type_name` space-normalization (vector only),
`CAST` double parens, `memory_coalescing` allowed-dtype gate, and the
shape-changing `expand_bitcast` (uint32→uint16 sub-word rand) — the last was
masked on OpenCL but fatal on CUDA. NVRTC PTX is disk-cached (`cuda_ptx`).

**WMMA (S3) DONE (2026-10).** `Renderer::cuda` sets `tensor_cores =
tc::get_cuda("sm_86")` (others empty, so CL/clang/interp are unaffected); the
TC opt is applied first in `hand_coded_optimizations` (faithful `USE_TC=1`,
`TC_OPT=0`, `TC_SELECT=-1`); `expand_wmma`+`contract_axis`/`unroll_axis` run in
the expander; `broadcast_and_devec_wmma` + `do_stack_wmma` lower the WMMA; and
`CUDARenderer` emits the `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`
asm. Gated by a semantic CUDA render case and a numerical check: a 16x16x16
bf16 matmul on CUDA matches tinygrad's CUDA-TC output exactly (0/256 diffs).

Two symbolic bugs surfaced by the swizzled TC index were fixed in
`fold_add_divmod_recombine`/`quotient_base` (`sym.rs`):
`(b % (div*d))*mul` (the port had `d*mul`), and `k = t//D - s` on the raw value
(the port applied `rem_euclid` first, collapsing the quotient). A
`reduce_mul_chain` pattern was also narrowed to `REDUCE(MUL)` so it no longer
fires on `REDUCE(WMMA)`.

**MNIST benchmark (bf16, BS=256, TRAIN_N=20000, TEST_N=2000, 40 steps, RTX 3060,
all warm):**

| backend | wall time | final test acc |
|---|---|---|
| CL (OpenCL) | 31.0 s | 78.55 % |
| CUDA, bf16 (no TC, default) | 21.9 s | 78.55 % |
| CUDA, bf16, TC (`TC_OPT=1`) | 24.0 s | 79.20 % |

At the faithful default `TC_OPT=0` tensor cores only apply to kernels with a
single reduce axis; `beautiful_mnist`'s convs reduce over `C*kh*kw`, so MNIST
does **not** use WMMA by default (CUDA-TC == CUDA-no-TC). `TC_OPT=1` (multi-axis)
does engage the convs (4 `mma.sync` kernels) and is ~9 % slower here - the bf16
packing/swizzle overhead is not amortised on these small tiles.

**Known gap:** `TC_OPT=2` (BEAM only; allows padding) now propagates `PADTO`
failures as a caught `Result` (matching tinygrad's `KernelOptError`) instead of
panicking, but padded convs still fail at NVRTC: the emitted `__WMMA_...` helper
references `nv_bfloat164` while the kernel never emits that struct typedef — the
renderer does not yet emit the `nv_bfloat16N` typedefs for every WMMA operand
width. `TC_OPT=0/1` are unaffected.

---

### 9.8 Multithreaded CPU backend — `ops_rayon` / `RAYON` (plan, 2026-10)

**Goal.** A fast CPU backend that needs no C/Rust compiler at runtime, executes a
*higher-level* (pre-`ndim → loop`) kernel IR, and parallelizes with `rayon`. Keep the
existing interpreter as the faithful reference/debug backend; `ops_rayon` is a sibling.

**Decisions (agreed).**
- Module `src/runtime/ops_rayon.rs`; device `"RAYON"`; renderer `RayonRenderer`; program
  `RayonProgram`. Matches tinygrad's `ops_<tech>` naming (`ops_cl`, `ops_cuda`).
- `rayon` is an **optional dependency behind feature `backend_rayon`**, like
  `opencl3`/`cudarc`. `default = ["capi", "backend_rayon"]`; `--no-default-features`
  stays std-only. INTERP remains unconditional (always compiled).
- Device-selection priority (first *available* wins, like tinygrad `Device.DEFAULT`):
  **CUDA → CL → RAYON → INTERP**. `DEFAULT_DEVICE` env still overrides. Unknown device
  names keep falling back to the interpreter (staging devices like `PYTHON`).

#### Why not just thread `ops_interp`

`exec_kernel_with_dims` (`runtime/interp.rs`) already loops over global launch indices.
The blocker is that `UOpCtx` is an `Rc<Node>` DAG with interior-mutable property caches
(`shape_c`/`mm_c`/`addrspace_c`) and `HostOpaque` is `Rc<RefCell<Vec<u8>>>` — neither is
`Send`/`Sync`, so it cannot be shared across `rayon` threads without UB. Every threaded
approach therefore needs the same first step: **compile each kernel into an owned,
`Send` instruction list once**, then execute it many times. That step is shared by the
fast path and by a parallelized scalar path, so it is the foundation, not throwaway work.

#### Execution-model choice

Two independent levers; the plan does both, in order:

- **Lever A — parallel scalar execution (cheap, low-risk).** Keep tinygrad's existing
  lowering, but compile the linearized uop list into an owned `Vec<Instr>` (every
  `ctx.*` query pre-resolved: `op`, `dtype`, src indices, `arg`, `addrspace`, `numel`,
  const value, SPECIAL extent), hand the compiler's per-launch value machinery a `&[Instr]`
  instead of `&UOpCtx`, and run the global-launch loop under `rayon` inside a scope. Store
  to disjoint output regions through raw pointers (`struct SendPtr(*mut u8)`), inputs read
  shared. Reuses `exec_alu`/`truncate`/`from_storage_scalar` verbatim, so numerics equal
  `INTERP` by construction. This is the "make the current interpreter multithreaded" win.
- **Lever B — range-driven array/loop-nest evaluator (the strategic ask).** Do **not** run
  the `ndim → loop` lowering stages (`apply_opts`, `expander`, `devectorize2`,
  `coalesce`, `add_control_flow`, `linearize`). Instead take the *kernel SINK AST* (the
  output of `get_kernel_graph`, after the early symbolic/mops stages), build our own loop
  nest from its `RANGE` nodes, and evaluate the value tree per element with our own
  instruction array. This is what "leaves out the ndim→loop rewrites" means: we keep the
  N-d ranges and compile them ourselves, with dtype-native inner loops and `rayon` over the
  outer (parallel) axes.

#### Seam change for Lever B

`to_program` (`engine/realize.rs`) hardcodes `full_rewrite_to_sink_with` **and**
`linearize`. Add a `Renderer` capability:

```rust
pub enum Lowering { Scalar, Range }
// Renderer facts gain: pub lowering: Lowering  (Scalar for interp/CL/CUDA)
```

and a `UopRenderer::render_ast(ctx, sink)` method whose default linearizes then calls the
existing `render` (so CL/CUDA/interp are untouched). In `to_program`:
`Lowering::Range` stops the pipeline after the canonicalizing early stages and calls
`render_ast`; `ProgramInfo::from_sink` and the `signature` scan still work on the SINK.
Serialization reuses the existing flat-JSON (`FlatGraph`) codec, exactly as
`InterpUopRenderer::render` does today.

`RayonProgram::new` decodes the flat graph and **compiles it into an owned `Send`
`RayonKernel`** (`Vec<Instr>` + range/loop metadata + buffer slot table), dropping the
`UOpCtx`. `call` then runs the kernel, with raw-pointer buffer windows scoped to the call.

**Robust fallback.** Any kernel the fast path cannot yet handle (images, `WMMA`, `LOCAL`/
`GROUP` + `BARRIER` workgroup semantics, exotic dtype decomps) is compiled to the *scalar*
IR and executed by the owned interpreter. So `RAYON` is correct-by-construction from day
one and gains speed kernel-by-kernel; the per-step gate compares against `INTERP`.

#### dtype-native values

`ops_interp` carries every lane as `f64`. The owned evaluator should instead dispatch ALU
on the real dtype (f32, f16/bf16, i32/i8/u8, bool, i64/f64) using the same rounding/wrap
rules as `dtype::trunc_float` / `interp::truncate`, with an `f64` fallback for the rest.
This is both faster and removes the documented `f64` precision loss for `uint64` pack /
`bitcast` paths (§8.23). Staged: start native f32 + integer + bool, extend as harness
coverage grows.

#### Parallelism strategy

- Build the loop nest from ranges, ordered outer→inner. Parallel axes = output `Loop`/
  `Global`-class ranges not feeding a reduce accumulation; reduce axes are inner.
- Pure elementwise → flat/chunked `rayon` over the output, tight typed inner loops when
  the store index is affine-contiguous, otherwise a gather loop.
- Single-axis reduce → parallelize the non-reduced output axes (row reductions).
- Reduce-to-scalar / multi-axis → parallel partials + combine.
- Workgroup kernels (`LOCAL`/`GROUP`/`BARRIER`) → serialize within a workgroup; initially
  delegate to the scalar path rather than model local memory.
- Soundness: parallel tasks write disjoint output slices and read shared inputs. Only
  launch parallel tasks when the schedule's store ranges are disjoint by construction
  (tinygrad's independent-work-item contract); otherwise serial. `rayon::scope` always
  joins before `call` returns, so the raw pointers never outlive the `Rc<dyn OpaqueT>`s.

#### Staged plan & gates

- **R0 — scaffolding + device priority (DONE).** `backend_rayon` feature + optional dep;
  `ops_rayon.rs` registering `RAYON` (composes the interpreter, so the device is correct and
  usable immediately); `Renderer::rayon()` facts; `ops_cuda::available()` non-panicking probe;
  `default_device()` priority CUDA → CL → RAYON → INTERP via a unit-testable `pick_device`.
  `cargo test` 144 pass / 9 ignored (default and `--no-default-features` builds; CL+CUDA
  feature builds compile). New gate `tests/device_select.rs` (priority, registration,
  RAYON-executes, RAYON==INTERP). `rand`/`randn` gained `device::is_interp_engine` so the
  f64-lane host route applies to `RAYON` until its dtype-native IR lands (flip in R2).
- **R1 — owned `Send` IR + parallel scalar execution (Lever A) — DONE.** New
  `src/runtime/ray_kernel.rs`: `RayKernel::compile` pre-resolves the linearized uop list into
  owned data (op/dtype/src-indices/arg/addrspace/numel/const) and raw-pointer
  `SendMemView` windows; `RayKernel::exec` is a faithful port of `interp::exec_launch`
  (`Val`/`exec_alu`/`truncate`), fanning the **global launch grid** out over `rayon`
  (`exec_launch` is per-workgroup independent; disjoint output writes, shared read-only
  inputs, `rayon::scope` joins before `call` returns). Unsupported ops make `compile`
  return `None` → `RayonProgram` falls back to `InterpProgram` (correct-by-construction).
  Knobs: `RAYON_SERIAL=1` (owned IR, serial), `RAYON_INTERP=1` (force fallback),
  `RAYON_VERBOSE=1` (per-program owned/fallback). Observability statics `OWNED_KERNELS`/
  `FALLBACK_KERNELS`.
  - **Coverage:** 57/57 kernels across conv/norm/reduce-broadcast tests compiled to the
    owned IR (0 fallbacks); the whole `cargo test` suite is green with `RAYON` as default.
  - **Speedup** (`examples/rayon_bench.rs`, 24 threads): matmul 256² 18.2×, matmul 512²
    **20.9×**, elementwise 512² 4.7×; owned-IR-serial vs `INTERP` is already 1.5×/1.2×.
  - Gate: `tests/device_select.rs` (`rayon_owned_fast_path_is_active`, `rayon_matches_interp`);
    `cargo test` 145 pass / 9 ignored; `--no-default-features` and CL/CUDA builds compile.
- **R2 — dtype-native executor (Lever B, first slice) — DONE (typed register machine).**
  Rather than add the `Lowering` seam, R2 was implemented as a **dtype-native register
  machine over the existing linearized IR** (see §9.8.1). This is a deliberate deviation
  from the "skip the `ndim→loop` rewrites" framing: those stages are *compile-time* and
  `to_program` is content-cached, so skipping them buys ~nothing at runtime; the runtime
  representation is what matters. Keeping the linearized IR also reuses the committed,
  tested lowering and makes the executor a one-for-one transliteration of `interp::exec_launch`.
  Gate: numerics equal `INTERP` on the whole suite; `cargo test` 152 pass / 9 ignored;
  typed covers **159/159** kernels across ops/conv/norm/attention/indexing/layers.
- **R2.2 — hoisting the interpreter loop (DONE).** Profiling with callgrind found the
  R2.1 cost was *not* the math but per-instruction framing: ~34% of instructions were in
  `malloc`/`free` (`srcs`/`real`/pair/view clones and a `vec!` per load element), plus a
  linear `END` scan per RANGE iteration and a long `Ops`-match + `group::in_` walk per
  dispatch. R2.2 hoists all of that to compile time and reuses buffers (see §9.8.1). Net:
  matmul 512² **656 → 208 ms** (3.2× over R2.1; **87×** vs `INTERP`, **4.3×** over the f64
  R1 path); elementwise ~1.1–1.3× over R2.1. The remaining cost is the interpreter's
  per-node dispatch itself (65% of retired instructions), so closing the rest of the gap to
  native would need native/typed code generation — out of scope for this std-only, no-JIT
  port (tinygrad's own CPU backend JITs C via clang; its interpreters are equally far off).
- **R2.3 — instruction-level fusion (not done).** A true fusion (recognize `Load`-vector →
  per-lane ALU → `Stack`/`Store` and RANGE-reduce patterns and run them as native tight
  loops) would remove the dispatch bottleneck, but requires codegen. Tracked.
- **R3 — integration/perf.** `RAYON` is already the CPU default (R0). Running
  `xor`/`mnist`/`vit_mnist` on it and benchmarking vs CL, plus considering `"CPU"` → `RAYON`,
  remains.

#### Risks / open questions

1. **Faithfulness at array level** (integer wrap, NaN, float rounding) — mitigate by
   reusing `exec_alu`/`truncate` and differential-testing vs `INTERP` *and* tinygrad.
2. **Which early pipeline stages are safe to keep before `render_ast`** — spike with the
   existing `TG_M5_STAGES` truncation to dump kernel ASTs and enumerate the op set; the
   later stages are optimizations and are not required for semantics.
3. **`Send`/UB boundary** — keep all unsafe confined to raw-pointer window extraction and
   the rayon scope; no `UOpCtx`/`Rc` crosses a thread boundary.
4. **Reductions with `GROUP`/`LOCAL`/`BARRIER`** — delegate initially; model local memory
   later only if a hot kernel needs it.
5. **Gather cost** — arbitrary index expressions are elementwise gathers; detect the
   affine-contiguous fast case for typed inner loops.
6. **JIT replay** — the owned `RayonKernel` is built at `render`/compile time and cached in
   the elf, so it replays for free (same as `InterpProgram`).
7. **Dependency policy** — `rayon` only behind `backend_rayon`; the default artifact's
   runtime deps stay `nanoserde` (+ `capi`).

#### 9.8.1 R2 implementation — dtype-native register machine (DONE)

**Files.** `src/runtime/ray_value.rs` (value storage + primitive ops) and
`src/runtime/ray_typed.rs` (`TypedKernel`), both behind `backend_rayon`.

- **Value storage (`ray_value::Bytes`).** An 8-byte-aligned, heap-backed `Vec<u64>`; whole
  slices are reinterpreted with `bytemuck::cast_slice` (no hand-rolled `from_raw_parts`, no
  per-element conversion). Values are lane-major `lanes*wsize` elements at the node dtype.
- **`Prim` trait + dispatch.** ALU is monomorphized per native dtype (`f32`/`f64`,
  `i8..i64`/`u8..u64`, `bool`↔`u8`) via a `dispatch!` macro selecting a generic impl fn.
  Integer ops use wrapping arithmetic at the type width; comparisons write `u8` 0/1;
  `cdiv`/`cmod`/`floordiv`/`floormod`, shifts, and float transcendentals mirror
  `interp::exec_alu`. Float→int casts mirror tinygrad `truncate` (clamp negatives for an
  unsigned destination, truncate, wrap) — this was a real bug found by the backend harness
  (`cast_f2u8` wanted 44 from ~300, saturating `as` gave 255).
- **`TypedKernel`.** `compile` reuses `RayKernel::compile`, commits **weak dtypes**
  (`weakint`/`weakfloat`, which survive into the linear list as uncommitted constants) to
  `default_int`/`default_float`, and rejects anything outside the native set (sub-word
  floats, size-changing bitcasts) so those fall back.
- **Executor.** A one-for-one transliteration of `interp::exec_launch` over typed slots:
  RANGE/END/IF masks, SPECIAL, INDEX(→pair or scalar), gated LOAD, STORE (incl. the
  devectorized `Stack`-dest form), CAST/BITCAST, STACK, ALU (with mixed-dtype coercion and
  lane broadcast). Slots and a value-buffer pool live in a thread-local `Scratch` recycled
  per launch, so a rayon worker allocates only on its first launch. Value-producing ops
  **detach** their destination slot (`Scratch::take_sized`), read sources immutably, fill,
  and hand the buffer back — so the hot ops (LOAD/STORE/INDEX/ALU) allocate nothing and the
  borrow checker stays happy (no `get_disjoint_mut` juggling).
- **Layered selection.** `RayonProgram::new` compiles both `TypedKernel` and the f64
  `RayKernel`; `call` uses typed → f64 → `INTERP`. Knobs: `RAYON_F64=1` (force f64, still
  parallel), `RAYON_SERIAL=1`, `RAYON_INTERP=1`, `RAYON_VERBOSE=1`. Counters
  `TYPED_KERNELS`/`OWNED_KERNELS`/`FALLBACK_KERNELS`.
- **Bug found & fixed:** the first visit to a `RANGE` must initialize to 0 *without*
  incrementing (the f64 executor inserts a zero and falls through); the typed executor
  incremented immediately, an off-by-one that corrupted the advanced-index prefix-sum
  kernel (`tests/indexing.rs::three_d_non_consecutive_and_consecutive`). Also
  `put_data` must zero reused pool buffers (stale bytes).
- **Perf (24 threads, `examples/rayon_bench.rs`).** R2.2 typed vs `INTERP`: matmul 512²
  **87×**, elementwise 7.5×; vs the f64 R1 path: **4.3×** matmul / ~1.3× elementwise.

#### 9.8.2 R2.2 — hoisting the interpreter loop (DONE)

callgrind on the R2.1 executor (N=128, single-thread, `RAYON_SERIAL=1 BENCH_SKIP_INTERP=1`)
showed the cost was framing, not math: `exec_launch` 29%, **`malloc`/`free`/`calloc` ≈ 34%**,
`memcpy` 9%, `read_mem` 8%, `get_i128` 6%. R2.2 removed those.

- **No per-instruction allocations.** `exec_launch` used to `clone()` the `srcs` `Vec` and
  build a filtered `real` `Vec` for *every instruction, every loop iteration*, and the hot
  ops allocated too (`index_pairs` built two `Vec`s per call; `load` a `vec![u8; isz]` per
  element; `store` cloned the value and pairs; `After` cloned the view vector). All of it
  moved to a compile-time [`Plan`] (`real: Vec<Vec<usize>>` etc.) and to a **detach/reuse**
  pattern backed by per-thread pools in `Scratch` (value buffers, and `Vec<Pair>` /
  `Vec<SendMemView>` payloads).
- **Dense dispatch.** A `Plan::kind: Vec<Kind>` resolves each instruction to a small dense
  enum (`AluBinary`, `Load`, `IndexPairs`, …) at compile time, replacing the runtime
  `match Ops` + `group::in_(ALU, …)` walk. Unsupported ops are rejected in `compile`, so the
  executor never hits a panic path. `RANGE`'s matching `END` is precomputed
  (`range_after`), removing the per-iteration O(n) scan.
- **Per-launch scrub.** `has_if` / `has_alu_param` / `needs_warp` are precomputed so the
  `masks` / `pvals` / `warp` tables are only built when the kernel actually uses them
  (most kernels allocate none); `masks.is_empty()` now means "all true".
- **Typed memory access.** `SendMemView::read_bytes`/`write_bytes` specialize widths
  1/2/4/8 to a single unaligned load/store (they were `copy_nonoverlapping` → `memcpy`
  calls, ~10% of instructions). `get_i128` fast-paths `INT32`; `index_pairs`/`index_data`
  read `i32` indices directly.
- **Measurement.** Deterministic callgrind `Ir` for the N=128 matmul+elementwise workload:
  **6.96G → 2.24G**. Wall (24 threads, N=512): matmul **656 → 208 ms**, elementwise
  **6.8 → 5.9 ms**; at N=1024 the typed matmul is **~87×** `INTERP`.
- **What's left.** 65% of remaining retired instructions are the interpreter loop itself
  (inlined dispatch + indexing + loads). Cutting that needs instruction-level fusion or
  codegen (R2.3), which this std-only, no-JIT port deliberately avoids.

---

### 9.9 Public-release work (2026-10) — decisions and status

This section records the release-readiness review and the decisions taken
before the first public release. The engine is unchanged; this is packaging,
API shape, and missing user-facing features.

**Docs / packaging — done in this pass**
- `LICENSE` (MIT + tinygrad attribution), `README.md` (draft; the maintainer
  will rewrite the prose), Cargo metadata (`description`, `license`,
  `repository`, `homepage`, `documentation`, `readme`, `keywords`,
  `categories = ["artificial-intelligence", "science", "hardware-support"]`,
  `rust-version = "1.88"`, `exclude`), docs.rs metadata.
- Crate-level docs in `src/lib.rs`; module-level docs for every public module.
  **All public items are documented**: `cargo rustdoc --lib -- -W missing_docs`
  reports 0 warnings (2026-10), down from 932 at the start of the release pass.
- `ONNX-Compat.md`: the T1/T2 contract, full 170-op matrix, parser/feature
  matrix, validation plan, and security notes.
- Edition 2024 is current; `rust-version = "1.88"` is the first release with
  the edition-2024 let-chains used in the code. CI is deferred by request.

**API decision: `Ctx` lives inside `Tensor` — DONE (2026-10)**
- `Tensor` gains a `Ctx` handle (a shared `Rc<RefCell<UOpCtx>>`), so graph
  construction no longer needs an explicit `&mut UOpCtx` argument on every
  call. Constructors take `&Ctx`; methods use `self.ctx`.
- `Ctx` is created with `Ctx::new()`; `UOpCtx::new()` remains for engine
  internals that do not build `Tensor`s. `Tensor::from_uop` captures the
  shared handle from the context (a `Weak` self-reference in `UOpCtx`).
- `std::ops` overloads (`+ - * /`, `Neg`, scalar variants) use the stored
  `Ctx`; mixing tensors from different contexts is a hard error.
- The old explicit-context methods are migrated repo-wide (tests/examples
  included). This is a **breaking** pre-1.0 API change.
- Rationale: the `Rc` is already there, the handle is cheap to clone, it makes
  operator syntax possible, and it removes two mutable contexts from every
  call site. `Tensor` remains `!Send`/`!Sync`.

**State API — CORE DONE (2026-10)**
- `Tensor::from_read(ctx, reader, dtype, shape, device)` — streaming upload
  from any `std::io::Read`, one tensor at a time. `tests/state.rs`.
- `tensor::state`: `State` map, safetensors read/write (a small strict JSON
  header parser + per-tensor streaming), `get_state_dict`/`load_state_dict`
  including **buffers** (BatchNorm running stats via `Module::named_buffers`).
  Cross-`Ctx` loads copy through bytes; same-arena loads use `Tensor::replace`.
- `Tensor::to(device)`, `Tensor::replace`, `numel`/`size`/`nbytes` and
  `zeros_like`/`ones_like`/`full_like`.
- **Still open:** free `Node.initial_data` after a device upload, avoid the
  `to_param` host round-trip and the `pending_copyin` clone (needed for the
  6.76 GB DA3 checkpoint), nn `from_parts` constructors, and a state-loading
  test on a real checkpoint.

**Tensor long tail — BATCH 1 DONE (2026-10)**
- Added in `tensor/extra.rs`/`op.rs` (gated by `tests/extra_ops.rs`):
  activations (`silu`, `swish`, `quick_gelu`, `leaky_relu`, `relu6`,
  `hardswish`, `hardsigmoid`, `hardtanh`, `elu`, `celu`, `selu`, `softplus`,
  `softsign`, `mish`, `logsigmoid`), `sign`, `round`, `tan`, `log10`,
  `logaddexp`, `copysign`, `erf`; reductions `min`, `argmin`, `std`,
  `std_mean`, `normalize`, `logsumexp`, `cumprod`; sorting `sort`, `argsort`,
  `topk`; `cummax`/`cummin`/`logcumsumexp`; inverse/hyperbolic trig
  (`asin`/`acos`/`atan`/`asinh`/`acosh`/`atanh`/`sinh`/`cosh`); predicates
  `isnan`/`isinf`/`isfinite`/`isclose`/`allclose`; `numel`/`size`/`nbytes`/
  `zeros_like`/`ones_like`/`full_like`; `roll`, `diag`, `meshgrid`,
  `masked_fill`, `repeat`, `avg_pool2d`, and two-operand `einsum`
  (`ij,jk->ik`, bias sums, dots).
- `from_url`/download is **descoped**: users fetch weights themselves. The
  library will not grow an opaque download API.
- **Still open:** `nonzero`, `scatter*`, `masked_select`, `gather_nd/elements`,
  `diag`onal, pad reflect/replicate/circular, `max_unpool2d`, `unfold`, and
  einsum ellipsis/1-operand forms. The ONNX matrix tracks which of these T1/T2
  needs.

**ONNX importer — T0 DONE, T1 PARTIAL (2026-10)**
- `src/onnx/` is the port of `tinygrad/nn/onnx.py`, using only `std`:
  `pb.rs` is a strict protobuf wire reader (varint/fixed32/fixed64/length,
  typed and packed TensorProto fields); `mod.rs` owns the decoded graph, a
  `Ctx` and a `Session`; `ops.rs` is the op dispatcher.
- Public API: `OnnxModel::{load, load_with_device, from_bytes, run, run_f32}`,
  `ctx`, `session_mut`, `inputs`, `outputs`. Inputs must come from
  `model.ctx()`; outputs stay valid after the model is dropped (the `Ctx` is
  reference-counted).
- Features: static and runtime-bound symbolic dims (`Shape`/`Size` produce
  host-visible int64 values), initializers with raw/typed fields, external
  data (`location`/`offset`/`length`, read one tensor at a time), `Constant`,
  and a per-run host cache for shape/index tensors keyed by interned UOp.
- Op coverage: elementwise/logical/cast, reductions/argmax/topk/cumsum,
  movement (`Reshape`/`Slice`/`Gather`/`GatherND`/`GatherElements`/
  `Pad` modes/`Expand`/`Tile`/`Split`/…), `Conv`/`ConvTranspose`/`Gemm`/
  `MatMul`, pools, `BatchNorm`/`LayerNorm`/`InstanceNorm`/`GroupNorm`/`LRN`,
  `Softmax`/`LogSoftmax`, `Resize`/`Upsample` (linear/nearest), `Einsum`
  (two operands), `OneHot`, `Trilu`, `EyeLike`, `Range`, `DepthToSpace`,
  `SpaceToDepth`, `Dropout` (inference), and op-versioned softmax defaults.
- **Pow:** the CUDA/C renderers have no ALU `Pow`; the importer folds
  scalar-base `Pow` to a host constant, small integer exponents to repeated
  multiplication (`x**2` is common in shape subgraphs), and otherwise uses
  tinygrad's `xpow` decomposition (`exp2(e*log2(|b|))` plus the
  negative-base/odd-exponent/NaN/`x**0` fixups).
- Validation: Depth-Anything V2 ViT-B fp16 (619 nodes, 253 initializers, 21 op
  types) runs end-to-end on CUDA. Against tinygrad on the same zero image the
  `depth` output (268,324 values) has max abs error 0.015625 (2 fp16 ulps),
  mean abs 0.0019, max rel 0.0054. `tests/onnx.rs` has 26 self-contained
  parser/op tests plus an `#[ignore]`d external-checkpoint smoke test.
- **Known gaps:** `NonZero`/`Compress`/`CenterCropPad`/`Scatter*`/`MaxUnpool`
  (need primitives), `GatherND` with `batch_dims>0`, `Einsum` with >2
  operands/ellipsis, `Resize` cubic/antialias and asymmetric linear modes,
  zero-element tensors (the
  storage layer cannot allocate empty storage), per-dimension
  `Conv`/`ConvTranspose` strides/dilations/output-padding, ONNX
  training/quantized ops, `If`, optional/sequence values, and ONNX-ML.
  `ONNX-Compat.md` is the canonical matrix.
- Unsupported ops return `OnnxError::Unsupported` with the op name rather than
  panicking; malformed/missing inputs return `Parse`/`Shape`/`Runtime`.

**Environment configuration — CACHED (2026-10)**
- `helpers::env` now caches every process-env lookup once (Context overrides
  are still checked live). `dtype::{default_float,default_int,sum_dtype}` cache
  their parsed `DTypeId` in `OnceLock`; `ops_rayon` caches its mode and
  `RAYON_VERBOSE`.
- Runtime knobs (device, dtype, `BEAM`, `JIT`, `FA`, `RAYON_*`, cache dirs)
  remain environment variables (set before the first tensor/realize), while
  build-time diagnostics/backends are Cargo features (`trace`, `dump`,
  `backend_*`). This hybrid is documented in the README.

**Codegen**
- The missing `get_late_rewrite_patterns`/`get_transcendental_patterns` and
  `pm_number_params` stages, image path, multi-device, BEAM worker pool and
  `TC_OPT=2` typedefs are documented in the README as known limitations; no
  codegen work in this pass.
