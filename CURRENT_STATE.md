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
- half/bf16 emulation — ✅ (M8.2/M5 `decomp_dtype.rs`).
- WMMA arm + first tensor-core layouts, BEAM search driver, DISK device. ⬜
- **nn core** (`nn/__init__.py` + `nn/state.py` + `nn/optim.py`): BatchNorm, Conv1d/2d,
  ConvTranspose1d/2d, Linear, GroupNorm, InstanceNorm, LayerNorm(+2d), RMSNorm, Embedding,
  LSTMCell; `State`; Optimizer + SGD/Adam(+W)/LARS/LAMB. **Proof: MNIST e2e** (MLP train
  step + Conv inference, weights via `State`, optimizer numerics diffed). Dataset via the
  std-only `fetch`/`from_url` shim. **IN PROGRESS — see §9.3/§9.4.** 🚧

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
- M8.2 open gaps: (a) weak-const commit / `pm_lower_weak`; (b) non-native-float decomp
  (done in M5, superceded); (c) arange realize — FIXED; (c') reduce-over-reduce — mostly
  fixed (`red_sum_2d_all` remains); (d) sub-dword vectorized coalescing bitcasts;
  (e) XOR training gate — now ✅ (see §8.7).

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
| 21 | early decompositions `symbolic_simple`+`get_simplifying_rewrite_patterns` | ✅ | floordiv/floormod→idiv/mod + pow2 shr/and + threefry (`decomp.rs`) |
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

**Differential gates** (`py/tests/test_rewrite_diff.py`, run directly; pytest not
installed): 10 tests, all green — `commutative` 192/8, `div_and_mod_symbolic` 192/8,
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
| fancy `__getitem__` corpus coverage | `INDEX` machinery itself is ported in M4; corpus sticks to basic slicing until M7 |
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
- `cargo test` → **71 passed, 6 ignored, 0 failed**. The 6 ignored = 1 `backend_harness`
  (`#[ignore = "M7 gate: XOR training via tinygrad.nn (Gradient/Adam)"]`), 1
  `nn_xor_parity::xor_parity_tinygrad_seed42`, 4 doc-tests.
- Python differential gates all green: `test_rewrite_diff.py` (10 tests),
  `test_roundtrip`, `test_tensor_diff`, `test_schedule_diff`, `test_codegen_diff`,
  `test_exec_diff`, `test_native_exec`.
- Suites: `tests/{golden,backend_harness,marshal,ops,grad,rand,m2_smoke,flat_codec,nn_xor,
  nn_xor_relu,nn_xor_parity,session,mem_probe}.rs`.
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
  tinygrad `optim.SGD([w,b], lr)`; `w -= lr*grad`, missing grad leaves the param untouched).
  `step(&mut ctx, &mut sess)` is tinygrad's in-place `Optimizer.step`; `step_graph(&mut ctx)`
  returns updated tensors (parity tests); `zero_grad(&ctx)` uses the owned params.
  **No Adam/momentum/weight-decay.**
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

### 8.10 In-place parameter updates: `Tensor::assign` (DONE)
tinygrad's optimizers assign the update into the parameter's existing buffer
(`Optimizer.step` → `Tensor.assign`, optim.py:35-56 / tensor.py:436), so parameters keep a
stable identity and the training graph does not chain across steps. The port previously had
a **pure-graph** optimizer (`step` returns new tensors), which forced callers to either
rebind (`w = w - lr*grad`, making each weight's subgraph contain the previous one → O(N)
graph growth, `toposort_from` dominated, ~180 MB) or round-trip the weights through host
`Vec<f32>` every step (what `nn_xor`/`mem_probe` used to do).
- `Tensor::assign(ctx, sess, value)` realizes `value` and copies its
  bytes into this tensor's storage **by buffer identity** (`buffer_of(self.uop)` — the same
  base the forward pass resolves), leaving `self.uop` unchanged. The copy is
  `realize::buffer_copy_into`: on the **same device** a direct buffer-to-buffer copy
  (`BackendAllocator::_copy_into` — OpenCL `clEnqueueCopyBuffer`; host backends copy the
  windows directly, no host round-trip), and only **across devices** a host round-trip
  (`buffer_bytes` + `buffer_copyin`).
- `Tensor::to_param` / `nn::Linear::into_params`: random-init parameters are *expressions*
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
- Caveat: `assign` mutates a param buffer in place, so a *derived* uop realized in an earlier
  step must not be reused (training loops build fresh derived nodes each step). The faithful
  STORE+AFTER buffer-identity/view-assign path (tinygrad tensor.py:455-463) is not ported.

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
