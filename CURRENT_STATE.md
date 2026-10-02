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
- **Dependency policy: std-only Rust + nanoserde.** No PyO3/maturin, no serde/syn
  ecosystem, no libc/sha2 crates; nanoserde (single micro-crate) for the JSON graph
  interchange. Two FFI faces only: (a) plain cargo binaries for batch oracle work over
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
| llm / viz / nn extras | `llm/`, `viz/`, `nn/onnx.py`, `nn/torch.py`, disk cache, process replay | huge | SKIP | ONNX import stays out |
| `fetch` / `Tensor.from_url` | thin std-only shim: `TcpStream` HTTP/1.1 GET (Content-Length + chunked, <=3 redirects, tinygrad cache layout) + minimal DEFLATE inflate (~200 LOC) | ~350 | **PORT (M7e)** | plain HTTP only; no TLS, no ranges |

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
  flash attention** (`extra/thunder/tiny/fa.py` slice). ⬜
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
  step + Conv inference, weights via `State`, optimizer numerics diffed). Dataset via the
  std-only `fetch`/`from_url` shim. **MNIST example + ignored e2e gate DONE (§8.16)**;
  ported layers: BatchNorm/Conv2d/Linear/GroupNorm/LayerNorm. Remaining: ConvTranspose2d,
  Conv1d/ConvTranspose1d, InstanceNorm, LayerNorm2d, RMSNorm, Embedding, LSTMCell, plus
  `State`/safetensors (§9.6). 🚧

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
- `cargo test` → **105 passed, 7 ignored, 0 failed**. The ignored = 1 `backend_harness`
  (`#[ignore = "M7 gate: XOR training via tinygrad.nn (Gradient/Adam)"]`), 1
  `nn_xor_parity::xor_parity_tinygrad_seed42`, 1 `mnist_e2e` (downloads data + a CNN
  train step), 4 doc-tests.
- Python differential gates all green: `test_rewrite_diff.py` (10 tests),
  `test_roundtrip`, `test_tensor_diff`, `test_schedule_diff`, `test_codegen_diff`,
  `test_exec_diff`, `test_native_exec`.
- Suites: `tests/{golden,backend_harness,marshal,ops,grad,rand,m2_smoke,flat_codec,nn_xor,
  nn_xor_relu,nn_xor_parity,session,mem_probe,nn_conv,nn_norm,nn_module,indexing,assign,
  mnist_e2e}.rs`.
- `cargo build` is warning-free; `cargo clippy --all-targets` has 0 errors (the
  `mutable_key_type` false positive — `UopId` hashes by immutable `id` — is allowed at the
  crate root). A few dozen minor style lints remain (see below), left as-is.

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

### 8.7 NN (PARTIAL — see §9.3)
- `src/tensor/nn.rs`: `Linear` (faithful init, `forward` = `x.linear(w.T, bias)`),
  `BatchNorm` (train/inference; running stats simplified), `LayerNorm`, `GroupNorm`,
  activations (relu/gelu/softmax/log_softmax/sigmoid/tanh/sin/cos/abs/sqrt/rsqrt),
  `Conv2d` (**`forward` is a stub** — returns bias or a zero tensor).
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
- No `nn/state.py` equivalent (`state_dict`/`load_state_dict`/safetensors), no
  `fetch`/`from_url`, no Embedding/LSTM/InstanceNorm/RMSNorm.
- Tensor helpers added for nn: `dot`, `linear`, `batchnorm`, `layernorm`, `pool`, `uniform`,
  `rand`, `tanh`, `div`, `square`, `sqrt`, `abs`.
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

**Status: works on small kernels, not yet practical at scale.** At BS=64 the
search finds valid faster opts but the CL driver segfaults after thousands of
throwaway programs/compile-errors (many candidates render invalid CL, and each
failed `clBuildProgram` plus a fresh program stresses the backend). Missing vs
tinygrad: the worker pool (parallel candidate compile), the `estimates.ops`
compute filter + `BEAM_UOPS_MAX`, and the **disk cache of results** (so a search
runs once, not every process). Until those land, `BEAM` is a research knob; the
default `hand_coded_optimizations` path is unchanged and the `pool` fix (§8.19)
is the real throughput win.

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

**Still needed for JIT:** `_next_counter` must be device-side (tensor.py:642) —
seed/counter as device tensors advanced by a STORE+AFTER inside the graph — so a
replayed step moves the counter. Today `rand_graph` stages the host counter as
constants. That, plus the session opt-in JIT (§9.6.3), is the remaining work.

---

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
1. **`Conv2d::forward`** — port tinygrad `mixin/op.py` conv2d verbatim: `_pad_constant` →
   `_pool(HW, stride, dilation)` → `reshape(bs,groups,cin,1,*oyx,*HW).expand(...,rcout,...)`
   → `permute(0,1,3,*oyx_indices,2,*HW_indices)` → `mul(weight.reshape(...))` →
   `sum(reduction dims, keepdim=True)` → `reshape(bs,cout,*oyx)` → `+ bias`. `pool` and
   `pad` already exist.
2. **Norm layers, faithful**: BatchNorm / LayerNorm / GroupNorm running-stat updates as
   in-graph **assign** in train mode, buffered stats in inference, affine `weight`/`bias`;
   add `InstanceNorm`, `RMSNorm`, `LayerNorm2d`.
3. **`nn/state.py` equivalent**: `state_dict`/`load_state_dict` over `Module::named_params`,
   plus a minimal safetensors reader/writer (std-only) and `init_weight`/`fetch`.
4. **Optimizers** (`nn/optim.py`): momentum, weight decay, then `Adam`, `AdamW`, `LARS`,
   `LAMB`. Needs `Tensor` assign/mutation semantics that survive the buffer work (§9.1) —
   today `step` returns new handles rather than mutating in place.
5. **Remaining ops needed by nn**: `max`/`min` with axis, `argmax`, `pad2d`, `flatten`,
   `interpolate`; `Embedding` via `custom_kernel`; `LSTMCell`.
6. **Data path (only if going for MNIST e2e)**: std-only `fetch`/`from_url` (HTTP/1.1 GET +
   inflate) loading into the tinygrad cache layout.

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

**Deferred:** a faster CPU backend. Backends are pluggable (`register_device` + the trait
seams), but the engine is deliberately single-threaded (`Rc`/`RefCell`, `thread_local!`
registry), so a threaded interpreter needs an owned `Send` kernel representation and
`Arc`-based storage; `rayon` would also break the std-only policy (`std::thread::scope`
would not). A cstyle→`cc -shared` native backend is the alternative. Tracked as a future
option, not worked this session.

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
2. **BEAM search — core ported (§8.20), still needs hardening.** `src/codegen/opt_search.rs`
   runs under `BEAM=n`; it explores UPCAST/UNROLL/LOCAL/GROUPTOP/GROUP, compiles each
   candidate and times it on-device. Missing: the worker pool (parallel compile), the
   `Estimates.ops` compute filter + `BEAM_UOPS_MAX`, the TC/SWAP/THREAD/PADTO/NOLOCALS
   actions, and the **disk cache** so a search runs once. Today a large-shape search
   stresses the CL driver (segfault); see §8.20.
3. **`TinyJit` — now the top throughput item.** §8.21 shows a steady-state BS=512 step is
   **scheduler-bound**: `create_linear_with_vars` (rangeify/kernelize) ~430 ms, device exec
   ~15 ms, copies ~0. Porting `TinyJit` (capture the LINEAR schedule + buffers, replay with
   fresh inputs) removes that per-step `graph_rewrite`/rangeify host cost, mirroring why
   tinygrad's `beautiful_mnist` (~0.1–0.2 s/step) is ~5–8x ahead. Prerequisites:
   the faithful `realize` is **done** (§8.22), and the device-graph RNG is **done**
   (§8.23) so a replay advances the sampler. Remaining: make `_next_counter` device-side
   (§8.23) and add the session opt-in (`Session::jit(bool)`) with capture/replay; a minor
   follow-up is attaching the realized `Buffer` to the UOp to make storage arena-independent
   instead of session-scoped.
4. **Complete the `nn` layer types** (`nn/__init__.py`): missing `ConvTranspose2d`,
   the `Conv1d`/`ConvTranspose1d` helpers, `InstanceNorm`, `LayerNorm2d`, `RMSNorm`,
   `Embedding`, `LSTMCell`. Ported so far: `BatchNorm`, `Conv2d`, `Linear`, `GroupNorm`,
   `LayerNorm`.
5. **`nn/state.py` + safetensors** (later): `State`/`state_dict`/`load_state_dict`, a
   std-only safetensors reader/writer, `init_weight`/`fetch`. Today only
   `Module`/`named_params`/`into_params` exist.
6. **Native GPU backend (later).** OpenCL lacks GPU-native features (WMMA/tensor cores,
   native bf16/fp16 compute). A future `autorust-nv` crate porting tinygrad's NV/CUDA path
   is the likely direction once items 1–4 land. Tracked as a design direction, not started.
