# PLAN.md — minimum viable port of tinygrad v0.14.0 to Rust (`autorust`)

Ground rules:

- Target: the `tinygrad/` tree in this repo (v0.14.0, the post-"rangeify" refactor with
  `mixin/`). The Python tree stays in the workspace as the **oracle**, not as a dependency
  of the final artifact.
- MVP = a native Rust `Tensor` that builds UOp graphs, schedules them, codegens kernels,
  and executes them on a Rust CPU backend — verified against real tinygrad by differential
  tests for every intermediate artifact, not just final numbers.
- **Backend-ready by construction**: the MVP must port tinygrad's pluggable backend seams
  (`Renderer`, `Compiler`, `Allocator`, `Program`, `Compiled` as Rust traits, plus cstyle
  rendering and grid-dim codegen) so that an OpenCL backend (M8) is a pure additive impl.
  Nothing below the trait boundary may assume CPU.
- **Dependency policy: std-only Rust + nanoserde.** No PyO3/maturin, no serde/syn
  ecosystem, no libc/sha2 crates; nanoserde (single micro-crate) for the JSON graph
  interchange. Two FFI faces only: (a) plain cargo binaries for batch oracle work over
  JSON graph files (one-shot — graph building, codegen diffs, CI), (b) a tiny `cdylib`
  C ABI (`#[no_mangle] extern "C"`, caller-owned buffers, ~5 functions) loaded from Python
  via stdlib `ctypes` for the per-kernel hot path (`ops_rust.py`) — bytes only, no JSON on
  the exec path. **Non-finite floats** (`NaN`/`±Inf` CONST args) have no JSON
  representation — verified empirically: nanoserde *and* miniserde both emit bare
  `NaN`/`inf` tokens they cannot parse back — so they travel as an explicit
  `Arg::NonFinite` variant (derive-friendly, no wrapper types). Internal hashes use
  std `DefaultHasher` (exact Python-`key` equality stays bridge-side). M8 follows the same
  rule: hand-written `extern "C"` bindings loading `libOpenCL`, mirroring how tinygrad
  itself drives native code (precedent: `Compiler.compile_server`, device.py:314-320).
- **Sequencing decision: frontend-first.** Track A drives M1→M4; the executor does not
  block anything. The only executor work before M5 is the M0 JSON round-trip + cdylib
  smoke test; the `exec_alu` table is pulled into M2 because const folding needs it.
- Rust never imports Python at runtime after the final milestone. All Python contact is
  via a dev/test bridge that lives under `autorust/py/`.

## 1. Inventory of moving parts

Sizes are LOC of the v0.14.0 tree. Decision key: **PORT** (must exist in Rust),
**POLYFILL** (bridge to real tinygrad during development), **SKIP** (never in MVP).

| Layer | Files (v0.14.0) | LOC | Decision | Notes |
|---|---|---|---|---|
| dtypes | `dtype.py` | 323 | PORT (P1) | `DType`, `dtypes`, `AddrSpace`, truncation, `least_upper_dtype`, np-dtype maps (np only in tests) |
| helpers | `helpers.py` | 640 | PORT subset (P1) | `ContextVar`/env context (`getenv`, `Context`), `prod/cdiv/floordiv`, `Metadata`, `dedup`, `unwrap` |
| Ops enums | `uop/__init__.py` | 137 | PORT (P2) | `Ops`, `GroupOp`; freeze the v0.14.0 set |
| UOp core | `uop/ops.py` | 1788 | PORT (P2) | hash-consed immutable DAG node, recursive props (`key`, `shape`, `dtype`, `device`), `toposort`, `graph_rewrite`, `Variable`/`sint`, `CALL`/`FUNCTION` plumbing |
| Pattern engine | `uop/upat.py` | 171 | PORT (P2) | `UPat` DSL + `PatternMatcher` (incl. pickling of matchers — we don't need pickle) |
| symbolic | `uop/symbolic.py`, `uop/divandmod.py` | 586 | PORT (P2) | `sym` rewrites, floordiv/mod semantics |
| shape system | `uop/movement.py`, shape logic in `uop/ops.py` | ~26+ | PORT (P2) | v0.14 has no ShapeTracker class; movement ops are UOps (`MOPS`-style) rewritten by `schedule/rangeify.pm_mops` |
| uop utils | `uop/render.py`, `uop/weak.py`, `uop/validate.py`, `uop/spec.py` | 505 | PORT (P2/P4) | `pretty_print` (repr oracle!), weak-dtype lowering, spec verification |
| Tensor front-end | `tensor.py` + `mixin/{op,elementwise,movement,reduce,creation,dtype,rand,gradient}.py` | ~5900 | PORT (P3) | mechanical but huge; `gradient.py`/`rand.py` late (P6) |
| scheduler | `schedule/{__init__,rangeify,memory}.py` | 886 | PORT (P4) | `get_kernel_graph`, `create_schedule`, `memory_plan_rewrite` |
| indexing/rangeify | `schedule/indexing.py` | 331 | PORT (M4) | **not optional**: `rangeify.py` imports `run_rangeify`/`apply_movement_op` from it — it is part of the main pipeline; `INDEX` (gather/scatter) machinery comes along. Only fancy `__getitem__` corpus coverage defers |
| multi-device | `schedule/{multi,allreduce}.py` | 381 | SKIP | no `UNSHARD`/`MSELECT`/`MSTACK`, no sharding |
| codegen | `codegen/{__init__,simplify,gpudims}.py` + `codegen/{decomp,late,opt}/*` | ~2200 | PORT (M5) | `to_program` pipeline: dtype decomp, transcendental, rangeify/mops, linearizer, regalloc, coalesce. Keep `apply_opts`/postrange **parameterized** (opts are a rewrite input, not hardcoded) and port the `expand_wmma` machinery with empty per-renderer `tensor_cores` lists, so BEAM (search driver) and WMMA (layout defs) stay additive. SKIP `opt/search.py` (BEAM driver), `opt/tc.py` (TC layouts) |
| renderer | `renderer/{__init__,cstyle}.py` | 690 | PORT (P5) | `Renderer` trait, `Estimates`, `Target`, **cstyle text rendering** — required for the OpenCL backend, not optional. SKIP `llvmir/ptx/wgsl/nir/amd` |
| CPU ISA JIT | `renderer/isa/x86.py`, `runtime/support/{compiler_cpu,elf}.py` | ~1900 | SKIP | v0.14 CPU compiles to x86 ELF; we substitute the interpreter (P5). Revisit later if perf matters |
| execution ref | `runtime/ops_python.py` | 249 | PORT (P5) | the uop interpreter — "the living definition of uops"; port verbatim semantics |
| devices/runtime | `device.py`, `runtime/ops_{cuda,amd,cl,nv,hip,...}.py`, `runtime/support/{hcq,hcq2,...}` | ~10k+ | PORT abstraction only (P6) | `Buffer`/`Allocator`/`Compiler`/`Program`/`Compiled` become Rust **traits** shaped exactly like tinygrad's, so backends (CPU interp now, OpenCL next) are additive impls. SKIP concrete GPU devices + HCQ/HCQ2 + graph capture |
| engine | `engine/{realize,jit}.py`, `engine/worker.py` | 677 | PORT subset (P6) | `run_linear`/`compile_linear`/`exec_*` (P6), `TinyJit` stretch goal, SKIP worker pool |
| rng / autograd | `mixin/rand.py`, `mixin/gradient.py` | 462 | PORT (P6, stretch) | after native execution works |
| nn / llm / viz / extras | `nn/`, `llm/`, `viz/`, `function.py`, disk cache, process replay | huge | SKIP | maybe a `nn`-shaped test module later for e2e (MNIST) |

MVP port budget ≈ **10–11k lines of Python** (of ~19k core), plus tests.

## 2. The seams (where we can cut)

There are four natural seams. The UOp is the universal currency, so every seam is a
UOp-graph hand-off, and we need one interchange format to exploit all of them:

- **Seam A — Tensor→UOp** (top-down): Rust builds the graph, Python schedules/codegens/executes it.
- **Seam B — SINK→LINEAR** (schedule): Rust schedules Python- or Rust-built tensor graphs.
- **Seam C — PROGRAM→kernels** (codegen): Rust linearizes kernel ASTs; Python or Rust executes.
- **Seam D — kernels→bytes** (execution, bottom-up): Rust executes uop lists produced by Python's codegen.

Interchange format: **JSON** via nanoserde — a post-order node list
(`{op, dtype, srcs, arg}` with `arg` as a tagged union, plus a dtype name table and root
indices), converters in both directions (`autorust/py/bridge.py` ↔ `autorust/src/uop/flat.rs`).
Human-diffable by design; parse cost is per-compile, not per-launch, so JSON overhead is
irrelevant. Non-finite floats use an explicit `NonFinite` variant (strict JSON).
This is *not* pickle; tinygrad's `UOp.__reduce__` exists but we want a format we own.
Note `UOp.key` = `sha256(str((op,dtype,arg)) + child keys)` (uop/ops.py:264) — we do **not**
reimplement Python's `str()` formatting in Rust; key comparison happens Python-side on
graphs shipped through the bridge. Same for `pretty_print` reprs: port it (P2) but verify
against Python.

Two complementary strategies fall out of the seams:

1. **Top-down (recommended primary track)** — port dtype → UOp core → Tensor front-end
   first. Every phase ends with a graph we can *hand to real tinygrad to realize*. The
   oracle is strong (graph `key` equality, then numeric equality) and the biggest, most
   mechanical chunk (Tensor front-end) is done early while the design of the Rust UOp is
   still fluid enough to change.
2. **Bottom-up (secondary track, cheap)** — `tinygrad/runtime/ops_rust.py` + a Rust
   interpreter. `Device` auto-discovers `runtime/ops_*.py` modules by filename
   (device.py:17,31), so dropping `ops_rust.py` with a `RustDevice(Compiled)` into the
   tinygrad tree is the official extension mechanism — zero changes to tinygrad needed.
   Python's codegen produces kernels; the bridge serializes the uop list; Rust executes.
   This validates the cdylib/ctypes plumbing, the flat format, and the `exec_alu` semantics
   table early, and gives an end-to-end demo (`DEV=RUST python test/test_tiny.py`) almost
   immediately.

The tracks converge: once the front-end + scheduler + codegen are native (track A) and the
interpreter is native (track B), `Tensor.realize()` stops calling Python at all.

## 3. Milestones

Each milestone has a **gate**: tests that must pass before moving on. All differential
tests live in `autorust/py/` (pytest, driven against `tinygrad/` in this repo) + Rust unit
tests in `autorust/src`.

### M0 — scaffolding + bridge (both tracks)
- `autorust` crate layout: `src/uop/`, `src/dtype/`, `src/tensor/`, `src/schedule/`,
  `src/codegen/`, `src/exec/`; `crate-type = ["lib", "cdylib"]` from day one.
- **Two Rust faces, std + nanoserde only**:
  - cargo binaries (`src/bin/oracle_*.rs`) for batch operations over JSON graph files:
    `rebuild` (JSON → JSON round-trip), `rewrite <pm>` (JSON in, rewritten JSON out),
    later `schedule`, `codegen`, `render`. Driven by pytest; no FFI.
  - cdylib C ABI: `exec_kernel(uops_ptr, len, bufs, dims, vals) -> ret` +
    `autorust_free(ptr)`; loaded via `ctypes.CDLL` (GIL released during calls).
- `autorust/py/bridge.py`: `uop_to_json`, `json_to_uop`, `realize_in_tg(sink_uop, tensors)`,
  `kernel_uops_from_program(program_uop)`, key/repr oracles.
- **Gate**: round-trip `UOp → JSON → Rust → JSON → UOp` preserves `key` and `repr` for a
  corpus of graphs harvested from `test/null/test_tensor.py`; cdylib smoke test executes a
  trivial hand-written kernel Python-side.

### M1 — dtypes + helpers
- `DType` (struct: `scalar`, `count`, `itemsize`, `fmt`, flags), `dtypes::*`, `AddrSpace`,
  truncation/cast tables, `least_upper_dtype`, `ConstType` handling. No numpy in the lib.
- **Gate**: differential test of `least_upper_dtype` over all pairs, cast/truncate tables
  vs Python; port of `test/null/test_dtype.py` core cases.

### M2 — UOp core (the critical milestone)
- Hash-consed `UOp` (arena + `UOpId`, or `Rc<UOpData>`; decide by benchmarking
  `graph_rewrite` churn). Fields: `op, dtype, src: Box<[UOpId]>, arg: Arg, tag` matching
  v0.14 (`tag` used by callify — keep it).
- Recursive properties as cached fixpoint computations: `key` (Rust-side hash, our own
  format — Python's only used via bridge), `shape`, `dtype` (port `dtype_from_uop`
  production rules), `device`, `vmin`/`vmax` + `resolve` (symbolic shapes are core: the
  scheduler's `broadcast_axes`, gates and the memory planner are `sint`-typed throughout),
  `is_variable`/`bound_var` etc.
- `UPat` + `PatternMatcher` with Python's application semantics (pattern list order,
  bottom_up/top_down, ctx threading, `walk`/`enter_calls`), `graph_rewrite` to fixpoint.
- Port `uop/symbolic.py`, `uop/divandmod.py`, `uop/movement.py`, `exec_alu`/`python_alu`
  table (needed later by interpreter and const folding).
- The op set includes `CALL`/`FUNCTION`/`PARAM`/`TUPLE`/`GETTUPLE` with their dtype/shape
  rules from day one — in v0.14 they are the execution spine (callify), not an optional
  feature (see M3).
- **Gate**:
  - `test/null/test_uop_symbolic.py`, `test_pattern_matcher.py`, `test_uop_graph.py`
    cores ported to Rust;
  - differential rewrite test: N random small graphs, Rust `graph_rewrite(pm) ==` Python's
    (by key via bridge) for `sym`, `mop_cleanup`, `symbolic_simple`;
  - `pretty_print` byte-equal reprs on the corpus.

### M3 — Tensor front-end (Seam A polyfill begins here)
- `Tensor` = `uop: UOpId` wrapper (+`grad`, `is_param`); port
  `mixin/{creation,dtype,elementwise,movement,reduce}` and the op-dispatch core of
  `mixin/op.py`; `tensor.py` minus precompile/numpy-data internals.
- **v0.14 layout note (surveyed M3 start)**: the mixin chain
  `RandMixin ← OpMixin ← ElementwiseMixin ← CreationMixin ← (DTypeMixin, MovementMixin)`
  subclasses **UOp itself** — all sugar is UOp-level and returns UOps; `Tensor`
  (tensor.py, 775 lines) only wraps `uop: UOpId` (+`grad`, `is_param`) with
  `_apply_uop`/`alu`/`_uop`/`_wrap_uop` delegation. Port order mirrors the chain:
  1. `tensor/core.rs` — Tensor struct, const factory, accessors, `_apply_uop`
  2. `tensor/movement.rs` — `_mop` (v0.14 emits shape-args as SRC consts, not args:
     RESHAPE/EXPAND = 1 shape src, PAD/SHRINK = lo/end list srcs, PERMUTE/FLIP = arg
     tuple, STACK promotes srcs), reshape/expand/pad/shrink/permute/flip/squeeze/
     unsqueeze/T/transpose/flatten/stack/cat/pad_to, `_parse_view_index`/`_apply_view_ops`
     (basic getitem: int/None/slice/ellipsis)
  3. `tensor/creation.rs` — full/ones/zeros/eye/arange/linspace/_tri/triu/tril
     (+`UOp.new_buffer`/`empty` with the global slot counter)
  4. `tensor/elementwise.rs` — `ufix`/Tensor-`_broadcasted` (keeps weak CONSTs weak via
     const_like(weak_dtype(least_upper))) /`_binop` = ctor broadcast (dtype_from_uop
     expands), named binops/unops, div/mod/fmod int-vs-float arms, pow/exp2/log2 family
  5. `tensor/reduce.rs` — `_rop` = `REDUCE(permute(reduce_axis+kept), arg=(op, n))`,
     `_reduce`/sum/prod/max/any/all/mean/var/std/logsumexp/softmax/cumsum/argmax/argmin
  6. `tensor/dtype.rs` — cast/bitcast/float/shrink_to_ndim
  7. `tensor/callify.rs` — AllocCtx/tags/`transform_to_call` (FUNCTION-free subset) +
     `callify`/`call`
- **Callify is part of M3, not deferrable**: v0.14's `realize`/`linear_with_vars` route
  every graph through `transform_to_call` (tensor.py:224) → `CALL(FUNCTION(body), params)`
  → the scheduler unwraps it. Port the FUNCTION-free subset now (AllocCtx, tags/`rtag`,
  `add_tags`, `pm_early_transform` rules that are no-ops without FUNCTION nodes,
  `pm_finalize_call`, `pm_replace_buf`) — `precompile`/`transform_precompiled_call` rules
  stay dormant until user kernels land in M7.
- MVP op subset: const/ones/zeros/arange/full/eye, all ALU binops/unops incl. `where`,
  casts, reshape/permute/expand/slice/flip/pad/cat, sum/max/min/mean/var, matmul,
  comparisons, `argmax`/`trilu` if cheap.
- **Polyfill**: `Tensor.realize()` → bridge → the callified `CALL` graph realized by real
  tinygrad; `tolist()/numpy()` via bridge.
- **Gate**:
  - `test/null/test_tensor.py` + `test_const_folding.py` core cases pass through the
    bridge-executed backend;
  - **key-equality suite**: for each corpus program, Rust-built graph `key` == Python's
    `Tensor`-built graph `key` (this is the front-end conformance oracle — exact, not
    numerical);
  - symbolic smoke test: one `Variable`-shaped slice+sum program matches Python's key.

### M4 — scheduler (Seam B)
- Port `schedule/indexing.py` (`run_rangeify`, `IndexingContext`, realize-map,
  `apply_movement_op`) — **required**, `rangeify.py` imports it.
- Port `schedule/rangeify.py` (`get_kernel_graph`, `pm_mops`, `BufferizeOpts`) and
  `schedule/__init__.py` (`create_schedule` — its dep-graph natively handles `CALL`/`END`
  kernel units, `create_linear_with_vars`, `resolve_linear_call`). SKIP
  multi/allreduce (no `UNSHARD`/`MSELECT`/`MSTACK` in corpus).
- Port the **graph-level memory planner** (`schedule/memory.py`, 62 LOC): buffer lifetime
  analysis over LINEAR, then suballocation into per-(device, copy/compute lane) arenas —
  internal buffers become `SHRINK`/`BITCAST` views of one big `int8` arena. This is a
  top performance feature; port `TLSFAllocator` (runtime/support/memory.py:14-106, ~90
  LOC of pure logic) with it. Note `_can_plan` excludes CL/WEBGPU
  (memory.py:12-16) — the planner simply won't touch OCL buffers; runtime-level reuse
  (M6 `LRUAllocator`) is what carries buffer-reuse perf on OpenCL.
- **Gate**: for the corpus, Rust `LINEAR` == Python's (same kernel count, same per-kernel
  AST keys as multiset; ordering may differ — compare as sets + topological validity);
  Rust-scheduled graphs realize correctly via the bridge; memory-plan debug output
  ("memory reduced from X MB → Y MB") matches Python's on the corpus.

### M5 — codegen (Seam C) + renderer + interpreter (Seam D) — tracks converge
- Port `codegen/__init__.to_program` pipeline: dtype decomp, late rewrites, `gpudims.py`
  (`get_grouped_dims` → `SPECIAL` globals — **do not** degenerate to (1,1,1) on CPU, the
  grid-dim machinery is shared with OpenCL), rangeify/mops, `linearize` (control flow,
  END/IF), regalloc, coalesce, simplify. Include `expand_wmma` and the
  `apply_opts`/postrange machinery (opts as a rewrite parameter, defaults from
  `heuristic.py`) with **empty per-renderer tensor-core lists and no BEAM driver** — the
  code paths exist, the search/layouts plug in later without engine changes.
- Port `renderer/__init__.py` as a `Renderer` trait (`render`, `supported_dtypes`,
  `compiler`, `has_local`/`global_max`/`local_max`/`shared_max`, `Estimates`,
  `Target`) and `cstyle.py` (C-family text rendering incl. `SPECIAL` substitution,
  BARRIER, LOCAL addressing, pointer arithmetic). The MVP `CpuInterpRenderer` renders to
  the flat uop-list instead of text, but cstyle exists and is diff-tested against
  Python's output so the OpenCL backend can reuse it in M8.
- Port the interpreter from `runtime/ops_python.py` (sans WMMA paths) to Rust:
  `exec/interp.rs`, executing the same linearized uops the renderer consumes
  (global_size/local_size loop ↔ `SPECIAL(g/l)` — identical semantics to a GPU launch).
- `ops_rust.py` (lives in tinygrad tree, ~120 lines): `RustDevice(Compiled)` with a thin
  allocator + `RustProgram` that calls the cdylib's `exec_kernel` via ctypes. This is the
  bottom-up payoff: `DEV=RUST python -m pytest test/test_tiny.py`.
- **Gate**:
  - per-kernel: Rust linearized uops == Python's (key multiset) for the corpus;
  - cstyle-rendered kernels byte-equal Python's for a subset of the corpus (relax to
    AST-level compare if ordering makes it flaky);
  - numeric differential: same inputs → `allclose` outputs, Rust interp vs tinygrad CPU;
  - `DEV=RUST` passes `test_tiny.py` cases (const, plus, elu, cat, sum, eye, gemm).

### M6 — native realize + device traits
- Port `engine/realize.py` subset (`compile_linear`, `run_linear`, `exec_kernel`,
  `exec_copy`), and `device.py` as backend traits in Rust mirroring tinygrad's seams:
  `Allocator` (`_alloc/_copyin/_copyout/_as_buffer`) with **`LRUAllocator` as a default
  trait impl** — the (size, options)-keyed free-list cache with OOM-triggered
  `free_cache()` retry (device.py:250-270). This is the runtime-level buffer cache and is
  performance-critical for OpenCL (every uncached alloc is a `clCreateBuffer`; the
  graph-level planner deliberately skips CL). Also `Compiler` (`compile(src)->bytes`, with
  `compile_cached` semantics), `Program` (`__call__(bufs, global_size, local_size, vals,
  wait)`), `Compiled` (renderer+allocator+runtime, `synchronize()`), `Buffer` (device,
  size, dtype, ensure_allocated, copy paths). `Tensor.realize` stops touching Python;
  `tolist()/numpy()` read backend memory through a uniform interface.
- The CPU interpreter registers as the first `Backend` impl; nothing in schedule/codegen/
  engine may assume CPU.
- **Gate**: end-to-end corpus runs with Python fully out of the loop; numerics match
  tinygrad CPU; `schedule_cache`-style behavior sane (no O(n) recompiles).

### M7 — user kernels + the rest of the Tensor (stretch, still before/alongside M8)
- **`@function` / `Tensor.custom_kernel`** (function.py, 105 LOC + `precompile` +
  `transform_precompiled_call` rules + `grad_fxn` plumbing in `UOp.call`): the machinery
  is already native since M3/M4 — this milestone activates the user-facing API and its
  precompiled-CALL fast path. **Proof target: flash attention** — port a slice of
  `extra/thunder/tiny/fa.py` (forward + one custom backward kernel) and run it through the
  schedule/codegen/engine unchanged. This is the regression test that CALL never got
  walled off.
- `TinyJit` (engine/jit.py) — rides the same CALL/PARAM spine (capture LINEAR, replay with
  fresh buffers); symbolic shape baking comes with it (M2 core already supports `sint`).
- autograd (`mixin/gradient.py`), `rand` (`mixin/rand.py`), half/bf16 emulation,
  WMMA arm in the interpreter + first tensor-core layouts (`opt/tc.py`), BEAM search
  driver (`opt/search.py` + `time_call`), `nn`-subset + MNIST e2e, DISK device.

### M8 — first additional backend: OpenCL (next session, the reason for M5/M6 shape)
Everything M5/M6 set up is consumed here; no engine changes expected.
- `OclRenderer` = cstyle with OpenCL C flavor (compare against
  `tinygrad/renderer/cstyle.py` OpenCL render of identical linearized uops — byte-diff
  is the conformance test, exactly like the interpreter-vs-CPU differential).
- `OclCompiler` via `cl-sys`/`opencl3` (clCreateBuffer, clBuildProgram, enqueueNDRange,
  synchronize on event), `OclAllocator` + host copyin/copyout, `OclProgram` impl of the
  `Program` trait, `synchronize()`.
- `supported_dtypes` per device query (fp16 via `cl_khr_fp16` optional); corpus runs with
  `DEV=OCL` vs `DEV=CPU` numerics (`allclose`) and kernel-count parity vs tinygrad's own
  `DEV=OCL` on the same graphs.
- Explicitly out of scope still: images, multi-device. (Tensor cores and BEAM activate in
  M7 and are backend-independent; the OCL renderer just declares `tensor_cores` when its
  device supports them.)

## 4. What we deliberately leave out (and what breaks because of it)

| Omitted | Consequence / guardrail |
|---|---|
| multi-device (`UNSHARD`, sharding, allreduce) | assert single-device in Tensor ctor; corpus avoids `.shard()` |
| fancy `__getitem__` corpus coverage (tensor indices, broadcasting int-index semantics) | `INDEX` machinery itself is ported in M4 (rangeify depends on it); the corpus sticks to basic slicing until M7 |
| GPU runtimes (CUDA/AMD/NV/...), HCQ/HCQ2, graph capture | concrete devices are SKIP, but the `Renderer`/`Compiler`/`Allocator`/`Program`/`Compiled` abstractions they plug into are PORTED (M6) so OpenCL (M8) is additive; `exec_copy` simplified to memcpy |
| GPU memory mgmt: `DepsTracker` (suballoc dependency tracking), `MemoryManager`/page tables (memory.py:110-281) | needed only by HCQ-class devices; OCL gets plain `LRUAllocator` semantics, same as tinygrad's own CL device |
| BEAM driver / tensor-core layouts | not walled: `expand_wmma` + opts parameterization are ported in M5 (empty TC lists), `time_call` semantics in M6; `opt/search.py` + `tc.py` activate in M7 |
| CPU x86 ISA JIT | perf hit; interpreter is our execution engine (could render C + cc later as cheap upgrade) |
| precompiled-CALL fast path, `@function`/`custom_kernel` UX | CALL/FUNCTION/PARAM spine is native from M3; the user-facing decorator + `transform_precompiled_call` activate in M7 (flash attention as proof) |
| DISK/TINYFS/NPY/PYTHON devices, `fetch` | constructor rejects Path; `from_url` unimplemented |
| symbolic-heavy coverage | `sint`/`Variable`/vmin-vmax are core since M2 and the scheduler is `sint`-typed throughout; corpus is concrete-first with symbolic smoke tests in M3/M4 — full coverage arrives with TinyJit (M7) |
| viz, process replay, disk cache, worker pool | no profiling hooks; `Context` env vars still ported for behavior toggles |

## 5. Risks

1. **Rewrite-order divergence.** `graph_rewrite` results can depend on pattern iteration
   order and match timing. RESOLVED in M2: unified/walk drivers ported line-for-line
   (deque stack + waitlist, bottom-up bpm fixpoint, CALL/FUNCTION body gating,
   `_rebuild_dtype`), validated differentially at ~13k graphs/PM (0.08% residual = NaN
   class only). Subtleties that mattered, for future ports:
   - Python `sum(uops)` starts at int 0 → first `+` is `ADD(CONST(0), t1)` (via
     `__radd__` reverse binop); `math.prod(xs, start=c)` multiplies c FIRST.
   - int×UOp puts the CONST on the LEFT (`__rmul__` → reverse binop).
   - `Counter.subtract` creates NEGATIVE entries for missing keys (divide_exact relies
     on `all(counts >= 0)` failing).
   - Python `//` is floor, `%` takes the divisor's sign — Rust `/`/`%` are trunc.
   - `early_reject` (ops of single-op src patterns) also guards against vacuous
     repeat matches (e.g. an empty STACK matching `UPat(STACK, src=repeat(...))`) —
     without it a rule fires with an empty binding store.
   - commutative flipping orders by `tuplize` = the dataclass tuple order:
     `(op, arg, dtype, srcs)` where ParamArg is `order=True` (vmin_vmax numeric,
     then name) and `DType.__lt__` is `(priority, bitsize, name, fmt)`.
   - Python mid-rewrite exceptions ABORT the rewrite (const-folding ValueError,
     ZeroDivisionError, negative shift counts) — Rust mirrors via `TGERR:` panics
     caught at the oracle boundary; both-error graphs are skipped differentially.
   (ordered pattern lists, same bottom_up traversal), per-pattern differential tests.
2. **`str()`-dependent keys.** Never reimplement Python `str()` of args in Rust; use the
   bridge for cross-checks, keep a Rust-side structural hash for internal caches.
3. **Float semantics.** `exec_alu` truncates to the node dtype after each op; port the
   table verbatim. RESOLVED in M1 with pinned semantics (differentially tested):
   - CPython int truncation = `v mod 2^64 -> reinterpret per ctypes signedness -> wrap
     to width` (e.g. `c_long(2^63) -> i64::MIN`, `c_ubyte(-3) -> 253`).
   - `dtype.const` for int dtypes keeps the *exact* Python int (no wrapping).
   - `ctypes` int types reject float inputs entirely — tinygrad only feeds
     already-const ints into the truncate table's int arms.
   - NaN→int raises in CPython; unreachable from tinygrad flows (Rust returns a
     deterministic value, documented).
   - Python's canonical dtype names (via `INVERSE_DTYPES_DICT`, aliases win) are
     `char/uchar/short/ushort/int/uint/long/ulong` + `half/float/double` — the
     Rust table uses these as interchange identity, not `int8`/`float32`.
4. **v0.14.0 is mid-refactor.** Tags (`rtag`/callify), `AFTER`/`STORE` assign semantics and
   `MOPS` movement are new and sparsely documented; pin to the exact commit in this repo
   and treat `test/null/*` as the spec.
5. **Interning & memory.** Python's `UOpMetaClass.ucache` + weakrefs does GC of dead
   nodes; Rust arena needs explicit epoch/GC or `Rc` cycles audit (DAG is acyclic, so
   `Rc` is fine; interning via a global `WeakSet`-like map).

## 6. Current status

- [x] M0 bridge (JSON interchange via nanoserde, 31-case corpus key+repr round-trip, cdylib smoke)
- [x] M1 dtypes (all-pairs differential green: lattice, can_lossless_cast, minmax, truncation sweep)
- [x] M2 UOp core (engine + `symbolic`/`symbolic_simple`/`commutative`/`mop_cleanup`/`div_and_mod_symbolic`
      differentially exact vs Python: 13k random graphs per PM, ~0.08% residual = documented
      NaN-const-through-minmax class; Rust unit tests port the engine-semantics cores of
      test_uop_graph/test_pattern_matcher)
- [x] M3 part 1 Tensor front-end (Tensor core, movement/getitem, creation, elementwise,
      reduce, indexing DSL + `test_tensor_diff.py` key-equality differential)
- [x] M4 scheduler (`run_rangeify`/`IndexingContext`, `get_kernel_graph` incl. `pm_mops`/
      earliest rewrites/copy-to-store/add-buffers/split-kernels, `create_schedule`,
      `resolve_linear_call`, `pm_copy_from_store`, `transform_to_call` callify,
      TLSF memory planner; gate: `test_schedule_diff.py` 6/6 programs — LINEAR kernel
      bodies key-equal vs Python incl. planned mode)
- [~] M5 codegen + cstyle + interpreter + `ops_rust.py` — IN PROGRESS (opening block
      key-exact, see below)
- [ ] M6 native realize + device traits
- [ ] M7 user kernels (`@function`/custom_kernel + flash attention), TinyJit, autograd, BEAM/TC activation
- [ ] M8 OpenCL backend

### M5 in-progress state (committed)
- `src/codegen/pipeline.rs`: `full_rewrite_to_sink` opening block ("early movement ops"
  = `pm_mops`+`mop_cleanup` bottom-up, "initial symbolic" = sym+pm_flatten_range,
  "postopt symbolic" = sym+pm_flatten_range+pm_reduce_unparented).
- `py/dump_codegen_stages.py`: python per-stage key-multiset dumper for
  `full_rewrite_to_sink` (PYTHON renderer, Target.parse("cpu")); stage order verified.
- `py/tests/test_codegen_diff.py`: differential gate — for corpus kernel ASTs, the Rust
  pipeline output bridged back to python is KEY-EQUAL to python's "early movement ops"
  stage (2 programs passing; identical node counts through "postopt symbolic" because
  the middle stages are no-ops on the current corpus).
- Oracle: `TG_M5_CODEGEN=1` runs callify+schedule, picks the first compute kernel body
  (SINK containing ADD/MUL), lowers it through the pipeline, emits the flat graph.
- **Ported so far** (the ENTIRE full_rewrite_to_sink pipeline key-exact vs python
  through "number params with -1" for the ew and mulc corpus kernels — 28/13-node
  final SINKs match; postrange, expander, devectorize, coalescing, weak lowering,
  split_ends, casted consts): `apply_opts` (Scheduler, convert_loop_to_global, the heuristic
  subset — GROUPTOP/unroll/easy-upcast/local-groups, get_optimized_ast with ANSI-colored
  KernelInfo names), `expander` (UPCAST/UNROLL range → STACK of consts),
  expand broadcast + add loads, `devectorizer2` (incl. EXPAND-scalar→STACK),
  `pm_clean_up_group_sink`, `add gpudims` (get_grouped_dims, SPECIAL gidx/lidx globals).
  Renderer facts in `codegen::postrange::Renderer` (CPU-interp defaults).
- `codegen/linearize.rs`: `linearize` (priority-toposort: run_count×priority×extra
  nkey, heap-driven) + `pm_linearize_cleanups` (no-ops on the corpus).
- **Next steps**:
  1. The interpreter (`codegen/interp.rs`) over the linearized lines — ops_python.py
     model. Semantic notes from the python source (runtime/ops_python.py:38-160):
     - launch loops: `for idxs in product(range(global_size[::-1]))`; warp =
       `product(range(local_size[::-1]))` (last local dim varies fastest); thread t's
       local dim d id = warp[t][2-d]; SPECIAL 'g' → same id for all threads,
       'l' → per-thread.
     - values are PER-THREAD lists; vector loads add a lane dimension OUTSIDE:
       `values[vec_load] = [load(inp, j) for j in range(load_sz)]` (lane-major,
       thread-minor).
     - memory flows as (mem, offset) PAIRS: INDEX/SHRINK build pairs `(m, o*scale)`;
       a scalar's pair can carry a re-indexable list as "mem" (INDEX over a plain
       LOAD: mem = the loaded per-thread list, offset 0 → m[0] recovers the value).
     - LOAD: `_load(m, i)` raises on out-of-bounds unless gated; vector loads zip
       (mem,offset) pairs with the gate list; STORE fans `src_values[1]` (a STACK)
       over lanes when `u.max_numel() > 1`, gated by exec_masks.
     - global_size/local_size derive from the SPECIAL extents (gidx*→global,
       lidx*→local; missing → 1).
     - ALU: `exec_alu(op, dtype, per-thread tuple)`; CAST truncates via
       `truncate[u.dtype](dtype.const(x))`; END jumps via loop_ends; IF/ENDIF push/
       pop exec masks; no WMMA.
     Numeric gate: allclose vs tinygrad CPU for the corpus (addc, mulc, sum).
  2. `codegen/interp.rs` is written (compiles, untested): exec_kernel derives
     launch dims from SPECIAL extents, then exec_launch per python's dispatch.
     Remaining wiring for the numeric gate: oracle `TG_M5_EXEC` mode that runs
     callify→schedule→pipeline→linearize per LINEAR kernel call, allocates one
     Mem per distinct buffer uop (keyed by uop id, sized max_numel*itemsize),
     fills the copy kernel's PYTHON-addrspace source buffer with the from_slice
     data before execution, runs kernels in LINEAR order, and emits the output
     buffer contents as JSON; pytest compares allclose vs tinygrad CPU.
  3. cstyle renderer (byte-equality subset gate); `ops_rust.py` DEV=RUST driver.
  Deferred: `load collapse`/`split ranges`/`simplify ranges` are no-ops on the current
  corpus (key-equal with empty PMs; port with the sum/reduce kernel work).

### M2 deferred (finish alongside M5/M6 consumers)
- `pm_simplify_valid` / `uop_given_valid` / `simplify_valid` (symbolic.py:313-429) — needed by
  index lowering (M5); uses substitute+simplify recursively, engine supports it already.
- `pm_move_where_on_load`, `reduce_mul_chain`, `pm_drop_and_clauses`, `gated_given_valid`
  (load/store folding — M5 consumers).
- `sym` extras: transcendental `xpow`, `pm_clean_up_group_sink`, `pm_lower_weak`/`pm_commit_weak`/
  `pm_cast_weak` (weak.py beyond `commit_weak` — M5 index lowering).
- `render.py` port (pretty_print/pyrender) — Python-side reprs used for debugging only so far.
- Known divergence classes (documented, outside the differential contract):
  - NaN/±Inf consts flowing through `_min_max` (Python's NaN comparisons → False; Rust carries
    int sentinels) — ~0.08% of the random corpus, all in the full `symbolic` PM.
  - ALU const folding saturates at i128 (Python exact bignums); sentinel-poisoning in the
    min/max arithmetic arms mirrors Python's conclusions for unbounded operands.
