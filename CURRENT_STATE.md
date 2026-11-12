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
  own feature: `opencl3` (`backend_cl`), `cudarc` (`backend_cuda`), `objc2`/
  `objc2-foundation`/`objc2-metal` (`backend_metal`, macOS-target-gated), and
  `rayon` + `bytemuck` (`backend_rayon`, the multithreaded CPU backend, §9.8) —
  `--no-default-features` stays std-only. Two FFI faces only: (a) plain cargo binaries for batch oracle work over
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

### M9 — Metal backend (macOS) — DONE
- **Dependencies.** `backend_metal` is the second GPU backend. It uses `objc2`,
  `objc2-foundation` and `objc2-metal`, declared as **macOS-target-only optional
  deps**, so Linux/Windows builds and `--all-features` never link them. Unlike CUDA
  there is no dynamic-loading mode to arrange: `libobjc` and `Metal.framework` are
  OS components, and shaders are compiled at run time with
  `newLibraryWithSource:options:error:` (the same path tinygrad uses when
  `METAL_XCODE` is unset), so no Xcode/Metal toolchain is required at build time.
- **Renderer.** `CStyleLang::metal()` + `renderer::cstyle::render_metal` mirror
  `MetalRenderer` (cstyle.py:251): `kernel void` with implicit `uint3 gid
  [[threadgroup_position_in_grid]]` / `uint3 lid [[thread_position_in_threadgroup]]`,
  `device` globals, `threadgroup` locals, `as_type<T>` bitcasts, `constant int&`
  scalar args, `precise::sin`, native `half`/`bfloat`, and the bf16 unary
  compute-through-f32 `extra_matcher`. `Renderer::metal()` exposes the facts
  (shared_max 32768, `ALL` minus fp8 and f64, empty tensor cores).
- **Runtime.** `runtime/ops_metal.rs`: `MetalAllocator` (`newBufferWithLength:options:`
  with `StorageModeShared`; real `(buf, offset, size)` views; blit-encoder D2D
  copies), `MetalProgram` (source → library → function → compute pipeline;
  `dispatchThreadgroups:threadsPerThreadgroup:`; in-flight command buffers for
  `synchronize`), and registration as `"METAL"` through the public seams.
- **Fast math must stay off (default).** Passing `nil` compile options makes Metal
  enable fast math, and `x + (-inf)` is then folded to `x` — which silently
  dropped every causal `-inf` attention mask (`nn_attention` caught it). The
  program always builds an `MTLCompileOptions` with `mathMode = Safe` unless
  `METAL_FAST_MATH=1` (mirroring tinygrad's `setFastMathEnabled`), in which case
  it uses `Fast`.
- **Default device.** Priority is now **CUDA → METAL → CL → RUST → CC → RAYON →
  INTERP**. On macOS Metal beats the deprecated OpenCL stack. `pick_device` keeps
  its old signature (Metal absent); `pick_device_with_metal` is the new entry point.
- **Gates.** `tests/backend_harness.rs` (12/12) runs its smoke/ALU/dtype/reduction/
  matmul/cast batteries on METAL too; `tests/device_select.rs` covers registration,
  priority, execution and offset views; `examples/xor` trains to convergence and
  `FAST=1 STEPS=3 JIT=1 examples/mnist` (CNN + BatchNorm + advanced indexing +
  Adam) runs a capture/replay step on METAL, as does a `FAST=1` ViT-MNIST train
  step (the 38-buffer fused ViT kernel is split by the budget). `tests/bf16.rs`,
  `tests/rand.rs`, `tests/nn_attention.rs`, and `tests/nn_models.rs` pass with
  METAL as the default. Full suite with `AUTORUST_CL_LIB=/nonexistent` (to
  bypass Apple's broken OpenCL driver): **244 passed, 0 failed, 11 ignored**.
- **MPS host GEMM (opt-in).** `MetalAllocator`/`MetalProgram` now also expose the
  `HostGemm` seam through MetalPerformanceShaders (`MPSMatrixMultiplication`).
  The scheduler recognizes a plain affine contraction on METAL and attaches a
  `HostGemm`; `MetalInner::host_gemm` binds the three `MTLBuffer`s as
  `MPSMatrix` objects, supports strided/batched operands, and caches the MPS
  kernel/descriptors by shape+layout. `HostGemm` gained `c_dtype` so f16 inputs
  can write an f32 accumulator (MPS supports mixed f16/f32 descriptors; bf16 is
  rejected by `MPSMatrixMultiplication` on macOS and stays on the generated
  kernel). The path is **opt-in** (`AUTORUST_MPS=all`, or `LibraryGemm::Always`)
  because on an M2 it measured at parity with/slightly slower than the generated
  kernel (f32 512³ 1.76 vs 1.61 ms, 2048³ 24.0 vs 23.6 ms; f16/f16 2048³ 16.9 vs
  17.3 ms), and the generated path is deterministic. With simdgroup TCs enabled
  the generated f32/f16 path is now clearly faster than MPS (f32 2048³ 15.8 ms
  vs 24.0 ms), so MPS stays opt-in for future Apple GPUs. Fused bias/scale/residual
  epilogues and non-row-major `C` stay on the generated path. Gate:
  `tests/mps_gemm.rs` (MPS used, f32/f16 numerics match generated, batched).
- **Buffer-limit fusion budget (general, not Metal-only).** `Renderer` gained
  `max_buffers` (`usize::MAX` for every existing backend, **31** for Metal). The
  scheduler turns it into a fusion budget in `schedule/indexing.rs`
  (`budget_realizations`): producer-first, it carries each node's distinct input
  buffers (capped just over the budget) and, when a node would exceed it,
  realizes the largest non-leaf source; the downstream kernel then reads that
  one buffer instead of its whole subtree. `budget = max_buffers * 2/3` leaves
  headroom for the output/stage temporaries the scheduler adds later, and
  `get_kernel_graph` verifies the final kernel Params and retries with a tighter
  budget if any kernel still exceeds the limit. A future OpenGL-compute (or
  other bounded) backend only has to set this fact.
  `MetalProgram` still asserts `signature.len() <= 31` with an actionable
  message for the one case the budget cannot fix: a single op directly reading
  more buffers than the limit. The tiny ViT golden (previously 38 buffers) now
  runs on METAL.
- **Simdgroup tensor cores (DONE).** `tc::get_metal()` is the exact v0.14.0
  `tc.metal` table (8x8x8, f32/f32, f16/f32, f16/f16, bf16/f32, bf16/bf16),
  wired into `Renderer::metal()` on arm64; `render_metal` emits the
  `simdgroup_*8x8` + `simdgroup_multiply_accumulate` helper per WMMA shape.
  `Settings::tensor_cores` now actually gates the heuristic TC path (it only
  gated CUDA TF32 before). Measured on the M2 (`gemm_bench`, f16→f32):
  512³ 1.72 vs 2.39 ms, 1024³ 6.93 vs 7.98 ms, 2048³ 16.98 vs 21.72 ms; f32:
  1024³ 5.94 vs 9.60 ms, 2048³ 15.78 vs 27.35 ms. Gate: `tests/metal_tc.rs`.
  The earlier failed attempt used a newer tinygrad Metal table whose
  integer swizzle is incompatible with the port's v0.14.0 `apply_tc_opt`; the
  v0.14.0 `tc.py` string table is the correct source.
- **Benchmark-driven device selection (DONE).** `runtime/autotune.rs` adds
  `select_fastest_device()` (also re-exported at the crate root),
  `rank_devices()`, `benchmark_device()` and `set_default_device()`. It runs an
  ML-shaped suite on every available non-interp backend — `relu(A @ W + bias)`
  (512³ f32), 4-head causal attention at 256 tokens × 64 dims, and a 2048×512
  row softmax — validates a sample of each result against a host reference, and
  installs the fastest as the process default. `DEFAULT_DEVICE` is now a
  `RwLock` (was a `OnceLock`) so the default can be replaced after first use;
  `DEFAULT_DEVICE=auto` runs the selection lazily, and
  `AUTORUST_AUTOTUNE_DEVICES=METAL,RAYON` restricts the candidates. `INTERP`/
  `CPU` are never candidates, and a device whose driver panics is caught and
  skipped. Measured on the M2: METAL 7.9 ms (linear 1.4, attention 4.4,
  softmax 2.1), CC 11.0, RUST 11.2, RAYON 260.8 (linear 170.8 — RAYON's typed
  kernel does not yet cover a fused GEMM epilogue and falls back to the
  interpreter, a real R2 gap the suite surfaces; the winner will change when
  coverage lands). Gate: `tests/autotune.rs`.
- **Multi-GPU device names (DONE for the GPU backends).** `device()` now
  lazily builds indexed names through a per-backend family
  (`register_device_family`): `CL`, `CL:1`, ... and `CUDA`, `CUDA:1`, ... each
  get their own `Compiled` (context/queue/stream, allocator and compile cache),
  with index 0 as the bare name exactly as v0.14 canonicalizes it
  (`canonicalize_device_name` also fixed: `cuda:0` → `CUDA`, `cuda:1` stays
  `CUDA:1`; it previously flattened to `CUDA1`). `autotune::available_devices`
  enumerates every physical device, so `select_fastest_device()` compares an
  integrated and a dedicated GPU separately and installs the exact winner;
  `CUDA_DEVICE`/`CL_DEVICE` apply to the priority-picked default. The fused
  attention `HostCall` now carries the tensor's exact device name, so a
  `CL:1`/`CUDA:1` launch does not fall back to index 0. Gates:
  `tests/device_select.rs` (`canonical_device_names`, `cl_indexed_device_names`).
  Known limit: `ARCH_SM`/`STREAM` remain thread-locals of the last-opened CUDA
  device (mixed-arch multi-GPU banks on per-device rendering, not these
  globals).
- **CL usability probes (DONE).** `ops_cl::available()` now builds and creates
  one trivial kernel per enumerated device (cached per device) instead of only
  enumerating: the deprecated Apple stack can be present while pipeline
  creation is broken, and one unusable GPU no longer hides a working sibling
  (`CL:1`); the priority pick falls through to the first usable index.
  Separately, the fused attention kernel is per-device probed
  (`cl_flash::device_supports`): Apple's `newComputePipelineState` rejects
  `ar_flash_f32`, so the dispatcher falls back to SDPA instead of panicking
  (autotune then ranks CL rather than skipping it). A kernel-specific failure
  cannot be caught by the generic smoke test — this is the targeted fallback.
- **Fused FlashAttention (DONE).** `runtime/metal_flash.rs` ports the same FA2
  schedule as `cl_flash`: one threadgroup per `(batch*head, 64-query tile)`,
  16-key streaming steps with the online softmax, Q in threadgroup memory and
  separate K/V tiles. Supports `d=64`, causal masking, an additive f32 `[n,n]`
  mask, and GQA. `fused_attention_ex` selects it for METAL, so
  `Tensor::flash_attention_with`/`attention_with` run the fused kernel. Measured
  on an M2 (`n=789, h=6, d=64`): **13.0 ms vs 33.1 ms** for generated-kernel
  SDPA (2.5x), maxabs ~1e-8 vs a f32 reference. Gate:
  `tests/flash_metal.rs` (causal/mask/GQA/odd-n/batched dispatcher).
  This exposed and fixed a real Metal runtime bug: `MetalProgram::call` built
  `setBytes` pointers to temporaries that were dropped at the end of the
  `match` statement, so hand-written kernels read garbage scalar args. The
  value is now staged in a local byte buffer for the duration of the call.
- **f64 fallback (DONE).** Apple GPUs have no hardware `double`. The public
  Tensor API lowers `FLOAT64` to `FLOAT32` on METAL at creation/`cast`/const/
  rand/readout (`creation::metal_lower_f64`), and `to_list::<f64>` converts the
  realized f32 data on the host. f64 graphs therefore compile and run, with
  f32 precision; the first fallback in a process prints a one-time warning.
  Genuine f64 remains available on INTERP/CPU/CL/CUDA. `from_bytes` converts
  f64 payloads to f32 on the way in. Gate: `tests/metal_f64.rs`.
  Low-level `UOp` graphs that bypass the Tensor API can still reach the Metal
  compiler with a raw f64 param.
- **Known gaps.** No Metal graph-capture path; fused
  `Tensor.custom_kernel` remains the M7 stretch. The fusion budget is a
  conservative heuristic (leaf-buffer union, not the final argument count), so
  it can over-split a graph; the post-rangeify retry makes the final bound
  robust, and the whole pass is a no-op for `usize::MAX`. Port fixes found
  while wiring the examples: `tests/backend_harness.rs` now pins INTERP
  explicitly (`staging_dev(None)` meant "default device", not PYTHON), and the
  MNIST/ViT `load_images`/`load_labels` no longer hold a `RefMut` across Tensor
  methods. Float→small-int casts on Apple OpenCL/Metal saturate, so cstyle emits
  an explicit `(long)` truncation (clamping negatives for unsigned) on those
  backends, matching the interpreter's `ctypes` semantics.

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
- `cargo test` → **244 passed, 11 ignored, 0 failed** with
  `AUTORUST_CL_LIB=/nonexistent` (2026-11; the OpenCL backend is skipped so the
  suite does not hit Apple's pre-existing OpenCL driver failure — see below).
  This run includes the Metal backend on an Apple M2. The ignored = 1
  `backend_harness` (`#[ignore = "M7 gate: XOR training via tinygrad.nn
  (Gradient/Adam)"]`), 1 `nn_xor_parity::xor_parity_tinygrad_seed42`, 1
  `mnist_e2e` (downloads data + a CNN train step), 1 `vit_e2e`, 1
  `onnx::external_checkpoint_smoke` (needs `ONNX_MODEL`), and 6 doc-tests.
- **Apple OpenCL caveat:** with the default feature set on macOS, `tests/flash_cl.rs`
  fails at `clCreateKernel` (`-48`/`newComputePipelineState failed`) in Apple's
  OpenCL runtime. This is independent of Metal (the test is skipped by the env
  above) and is the reason the full-suite count uses the CL workaround.
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
- `cargo test --no-run` compiles warning-free for the default,
  `--all-features`, and `--no-default-features` sets (feature-gated tests and
  the `rayon_bench` example carry the right `cfg`/`required-features`), and
  `cargo clippy --all-targets` is 0/0 for the same three feature sets.

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
- INTERP always on; CL behind `backend_cl` (opencl3); METAL behind `backend_metal`
  (macOS-only `objc2`, §M9). M8 render byte-diff + numerics + kernel-count parity on
  the RTX 3060. `tests/backend_harness.rs` 12 passed / 1 ignored, now including METAL.
- **Default device** (`Tensor::creation::default_device`): env `DEFAULT_DEVICE` if set, else
  `CUDA` → `METAL` (macOS) → `CL` → `RUST` → `CC` → `RAYON` → `INTERP` for the first
  available backend. The probes are non-panicking. There is **no "NV"**: it was a
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
  `MetalAllocator` *does* implement `_offset` (shared allocation + byte offset) and
  `tests/device_select.rs::metal_allocator_supports_offset_views` pins it.

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

### 8.34 Opt-in phase profiling (`AUTORUST_PROF=1`) and the DA3 schedule cost

`src/engine/prof.rs`: thread-local named accumulators (`add` for timed phases,
`count` for counters), printed by `report()`. Zero cost when disabled (a
`OnceLock` read at each call site). Hooked into `realize_many`
(`materialize`/`callify`/`schedule`/`run`), `exec_linear` (`exec.kernel`),
`create_linear_with_vars` (`rangeify`/`resolve`/`copies`/`memory`),
`get_kernel_graph` (`earliest`/`copy_to_store`/`run_rangeify`/`cleanup`) and
`unified_rewrite` (steps, match attempts/hits, waitlist pushes/fires).

Why it exists: the DA3 port (sibling project `da3-autorust`) ran at ~75 ms of
GPU time but seconds of wall time per frame. Measured on an RTX 4500 Ada,
12-block ViT-B (721 tokens, 768 dim):

| | tinygrad 0.14.0 | autorust |
|---|---|---|
| kernels | 168 | 265 (incl. exact-erf GELU) |
| schedule | 38.9 ms | 2034 ms |
| realize (warm) | 50.5 ms | ~2700 ms |

Breakdown of the 2034 ms: `get_kernel_graph` (rangeify) 1039 ms,
`create_schedule` 0.65 ms, memory plan 1 ms, then `compile_linear` on
cache hits ~660 ms, GPU execution 60 ms.

Inside rangeify, all of it is the rewrites: `earliest` 4 ms,
`copy_to_store` 2 ms, `run_rangeify` 34 ms, `cleanup` 587 ms, and the
bufferize/store tail ~450 ms. Notably the *cleanup* rewrite costs ~250 ms per
full-graph pass even for a 4-rule PM, i.e. the cost is the traversal, not the
rules.

Counters for one 504×280 backbone schedule (4840-node graph, 571 kernels,
1.72 s of rangeify):

| stage | time |
|---|---|
| `rng.cleanup` (symbolic + reduce_simplify + const_buffer_folding + remove_bufferize, 134 rules) | 0.96 s |
| `rng.split_kernels` (1 rule, bottom-up) | 0.83 s |
| `rng.run_rangeify` | 0.07 s |
| bufferize / earliest / copy_to_store / no_indexing | < 0.07 s total |
| codegen (`to_program` per kernel) | ~0.6 s |

Two hypotheses were **ruled out** by measurement:

- Not traversal thrash: `TG_RW_VISITS=1` (per-node visit histogram in
  `unified_rewrite`) shows **2–6 visits per node** in every pass, and the engine
  costs ~0.6 µs per step. The earlier "~250 steps/node" figure divided all
  rewrite steps by one graph's node count while the steps span ~180 rewrites.
- Not the waitlist retry loop: pushes are bounded by edges.

`TG_RW_TRACE=1` prints one line per `graph_rewrite` (rules, steps, ms, µs/step),
which is what localized the cost:

1. **Kernel count.** 571 kernels for the backbone vs tinygrad's 168 for the same
   12-block shape (`py/tg_bench.py`; it lacks qk-norm/RoPE/exact-erf GELU).
   Every kernel pays `split_store`'s nested 3-PM rewrite (~1.5 ms) plus codegen
   (~1 ms), so the count dominates the total.
2. **`split_store`'s nested rewrite is the second half (partly localized).**
   `split_kernels` is a 1-rule bottom-up pass; each match runs `split_store`,
   which re-runs a nested 3-PM rewrite (`to_define_global` + `pm_flatten_range`
   + `rangeify_codegen`) over the matched node's subtree. `TG_RW_TRACE` shows
   this pass at **52.8k steps but 889 ms → 17 µs/step** (28× the engine's
   ~0.6 µs), i.e. 275 matches × ~3 ms of nested work. (`split.input_nodes`
   reports ~3200 nodes per match, but that is `toposort`, which — unlike the
   rewrite — does not stop at opaque call bodies, so treat it as an upper
   bound.) Next step: split that timer across the three nested PMs; the prime
   suspect is `rangeify_codegen`/`pm_flatten_range` over shared index
   expressions.

   The AST-size hypothesis is **disproved**: kernel ASTs average 35 nodes
   (`codegen.ast_nodes`), comparable to tinygrad's ~46 steps/kernel.

### 8.35 Same-device `exec_copy` is a D2D memcpy, not a host bounce

`exec_copy` always went `_copyout` (which `cuStreamSynchronize`s) then `_copyin`,
even when both sides were the same device. For CUDA that is a full stream sync
plus two PCIe transfers per copy. It now takes the `_copy_into` path (the
backend's `cuMemcpyDtoDAsync`) when `dest.device() == src.device()`, mirroring
tinygrad's `exec_copy`. Measured per copy on an RTX 4500: ~1.4 ms → ~0.05 ms
(the DA3 forward emits ~290 copies, mostly bufferize/copy-from-store).

Also added `exec.linear` (whole dispatch loop) and `exec.copy.secs` to the
profiler: `exec.kernel` alone is only ~0.17 ms per launch, so the remaining
per-frame cost is dispatch in `exec_linear`, not the kernels.

### 8.36 Per-launch hashing, per-copy syncs, and re-uploaded weights

Three costs that made a DA3 frame look like "the GPU is idle":

1. `Device::runtime` hashed the whole compiled artifact (`elf.lib`, the PTX) on
   **every launch** to key its program cache — 0.22 ms per kernel, 97% of
   `exec_kernel`. The key is now computed once at compile time
   (`runtime::device::elf_key`, stored as `CompiledProg::elf_key`) and looked up
   via `runtime_keyed`; per-launch cost is 3.5 µs.
2. `_copy_into` (CUDA) called `cuStreamSynchronize` after every async D2D copy,
   serializing the pipeline (~1.2 ms/copy). Removed — the copy is enqueued on
   the same stream, so program order already serializes it, matching tinygrad's
   `CUDAAllocator._copy`.
3. **Weights were re-uploaded on every realize.** `safe_load` stages a tensor as
   `COPY(device, BUFFER(PYTHON))`, so an unrealized weight is copied host→device
   by every realize that reads it: measured **~290 host→device copies per DA3
   forward**, i.e. the whole checkpoint, ~1.2 s/frame. `safe_load`'s docs now
   say to materialize; the DA3 loader realizes each tensor once
   (`Session::realize_tensor` + `set_uop`/`set_realized_buffer`) so weights
   become device BUFFER leaves.

Effect on DA3 (140×252, RTX 4500 Ada): non-JIT steady state 4.6 s → 3.0 s per
frame; JIT replay (TinyJit capture/replay) **0.32 s → 0.008 s per frame**
(120 FPS), which is 1.1 ms of kernels + one input upload.

### 8.37 `ranges` was quadratic — the real scheduler bug

`ctx.ranges()` is tinygrad's `@recursive_property _ranges`: a node's ranges are
the union of its children's (already memoized) ranges, i.e. O(1) amortized per
node. The port instead ran **`toposort` over the whole subtree on every cache
miss**. `split_store` calls `ranges` once per matched STORE/END, so with 275
matches on ~3200-node subtrees the pass walked 880k nodes: `split.ranges` was
**632 ms of a 659 ms `split_kernels`**, and the same property is used all over
indexing/codegen.

Fix: walk post-order iteratively but **stop at cached nodes**, so only genuinely
new nodes are computed (`src/uop/range.rs`). Also made `to_define_global` and
`rangeify_codegen` singletons (they rebuilt a `PatternMatcher` on every
`split_store` call, which also defeated the identity-keyed `add` memo).

One 504×280 DA3 backbone schedule (4840 nodes, 571 kernels):

| | before | after |
|---|---|---|
| `rng.split_kernels` | 659 ms | **17 ms** |
| `rng.cleanup` | 975 ms | 604 ms |
| rangeify total | 1815 ms | **769 ms** |
| model realize | 2382 ms | **1218 ms** |
| JIT capture @140×252 | 5.8 s | **2.6 s** |

Output is unchanged (differential gate still 1.5e-5) and replay stays 9 ms/frame.

Remaining hotspot: `pm_remove_bufferize` (369 ms of the 600 ms cleanup is 314
`substitute` calls at 1.2 ms each; `rb.walk` is 2 ms, `rb.reduce_walk` 51 ms).
Investigated and found to be **inherent per-site work**, not another cache bug:

- all 314 `(src, replacements)` pairs are distinct (memoising would not help);
- every replacement key is live in `src` (skipping is not valid);
- each `src` subtree is ~1770 nodes (measured with a temporary toposort), and
  all 314 are distinct, so it is 556k node visits of genuine substitution.

The only way to cut it is to make `remove_bufferize` fire ~1100 times less, i.e.
leave fewer `STAGE(INDEX)` sites for `run_rangeify` to clean up. That is a
bufferization-policy question (`maybe_realize_axes` / the fusion decision), and
it needs a controlled comparison: build the *same* 12-block ViT graph in both
runtimes and compare node/kernel/STAGE counts (tinygrad: 168 kernels,
`py/tg_bench.py`). Tracking that as the next scheduler task; it is not on the
critical path for the streaming pipeline any more (capture 3.1 s at 504×280).

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
   runtime deps stay `nanoserde` (the Python bridge is a separate dev-only crate).

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
- **Scratch leak fixed (2026-11).** The `REG` param path built a fresh
  `Vec<SendMemView>` and assigned it to the slot instead of taking one from
  `Scratch::take_m`, so every `recycle` pushed another payload into `pool_m`
  with no matching pop. RSS grew ~25 MB per MNIST step (106 → 429 MB over 10
  steps at bs=8, JIT on or off); the same run now plateaus at ~150 MB. This
  only affected the typed evaluator (`RAYON`), not `RUST`/`CC`, which is why
  those benchmarks were flat.

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
- `PERFORMANCE.md` (benchmarks, GEMM/attention internals, device selection),
  `FREEZE.md` (AOT frozen plans), and `CONFIGURATION.md` (features, `Ctx`
  settings, complete env-var reference) split the user-facing detail out of the
  README.
- Edition 2024 is current; `rust-version = "1.88"` is the first release with
  the edition-2024 let-chains used in the code. CI is deferred by request.
- **The Python bridge is its own crate** (`py/bridge`, package
  `autorust-bridge`, `publish = false`). It holds the `cdylib` C ABI, the
  `rebuild`/`graph_summary` JSON round-trip helpers (thin wrappers over the
  public `uop::{parse_flat, to_string_flat, Graph}` codec), and the oracle/dev
  binaries that used to live in `src/bin`. The library is now a plain `rlib` —
  no `capi` feature, no `cdylib`, no tooling-only entry points, no dev-only
  binaries in the published package. `cargo build` builds only the library
  (`default-members = ["."]`); the harness uses
  `cargo build -p autorust-bridge`.

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

### 9.10 Register-blocked GEMM + grid ordering (2026-11) — DONE

The matmul-shaped `MUL -> REDUCE(ADD)` pattern now gets a backend-specific
register tile in `codegen/postrange.rs::try_matmul_tiling` (before the generic
heuristic, after the tensor-core attempt):

- **CPU (`RUST`/`CC`)**: upcast both output axes 8x8 and unroll K by 4. The
  inner loop does 64 FMAs from eight vector loads. A 512³ f32 GEMM went from
  27.6 to ~320 GFLOP/s; ViT-B/16 forward from 4.1 s to **0.66 s** (24 threads).
- **GPU (`CUDA`/`CL`)**: `LOCAL` + `UPCAST` + `UNROLL`. Small/medium shapes use
  4x4 registers with 2x64 threads (block 8x256), large (`M*N >= 1M`) use 8x8
  with 4x16 threads (block 32x128). `AUTORUST_MM_{M,N,K,LM,LN}` override the
  defaults for tuning.
- **Grid order** (`codegen/gpudims.rs`): global axes are sorted by extent
  ascending so the largest dimension becomes `blockIdx.x` (fastest-varying).
  For `M=197` linears this runs all M blocks for one N tile before moving on,
  keeping the weight N-tile hot in L2 instead of streaming it per row. This
  alone took CUDA ViT from 61 ms to 30 ms.

bf16/f16: the tiling is dtype-agnostic. Two fixes were required:
1. `codegen/coalesce.rs` now includes `BFLOAT16` in the vector-coalescing set;
   previously every bf16 load stayed scalar.
2. The matmul multiply is promoted to the f32 accumulator before scheduling
   (`Cast(f32, A) * Cast(f32, B)`), matching tensor-core accumulate semantics.
   Without this, every bf16 product was rounded back to bf16 and re-widened
   (a 512³ bf16 GEMM was ~15x slower than f32; now it matches f32 on CPU).

`PADTO` on an output axis is deliberately **not** used by the matmul path: the
store mask is not lane-wise for vector stores, so padding a prime axis (197)
wrote past the tensor and corrupted the heap. Tensor-level SDPA padding
(`SDPA_TILE=32`, pad Q/K/V to 224, mask padded keys, slice back) is **correct**
after materializing the padded tensors with `contiguous`, but it is **slower**
(CUDA attention 6.8 -> 11.0 ms, CPU 14.8 -> 17.4 ms) with the current schedules,
so it is opt-in and default-off.

`FA=1` remains an algorithm reference: CUDA ViT is unchanged (~62 ms) and CPU
training is ~2x slower than SDPA.

### 9.10b Dynamically loaded cuBLAS (2026-11) — DONE

The CUDA backend now has an optional library GEMM. It is **dynamically loaded**
(`libloading`; `libcublas.so.12/13`), so `backend_cuda` still needs no CUDA
toolkit at build time and simply disables the path when the library is absent.

Detection happens in `schedule/host_gemm.rs`, called from `split_store` after
the store is lowered to affine `INDEX` expressions (not from the tensor graph):
a store of `REDUCE(ADD, MUL(load, load))` with affine strides becomes a
`HostGemm` descriptor `{slots, offsets, m/n/k, lda/ldb/ldc, batch strides,
dtype, alpha, optional bias}`. It is stored in the kernel's `KernelInfo`;
`engine/realize.rs` then skips render/compile and runs `runtime/ops_cuda.rs`'s
host dispatcher.

Supported: f32 (`cublasSgemm_v2` / strided-batched), f16/bf16 (`cublasGemmEx`
with f32 accumulate), arbitrary leading dimensions (transposed/strided views as
ViT produces), fused `+ bias` via a tiny NVRTC epilogue kernel, and strided
batches. Unsupported patterns (masked/fused softmax operands, flattened outputs,
multiple reductions) fall back to the generated kernel. The dispatcher is
installed whenever the library loads, independently of the `AUTORUST_CUBLAS`
toggle, so a device opened before the toggle still works.

Default policy: use the library only when `m`, `n` or `k` is not divisible by 4
(197 is the motivating case); our 8x8 register tile "wins on clean square
shapes". **Re-measured (kernel-only): it does not.** cuBLAS wins f32 at
1024/2048/4096 and at 6312x384x384 / 6312x384x1536 (2.6x / 4.0x); the earlier
claim came from per-iteration readback inflation. The generated bf16 path is
competitive (10.7 vs 9.8 TFLOP/s) because bf16 storage halves the traffic, not
because of `mma`. `Auto` now prefers the library for f32 on CUDA (16-bit/CL/CPU keep the
heuristic); the generated f32 kernel wants shared-memory tiling before it can
compete.
`AUTORUST_CUBLAS=0` disables, `=all` forces, `=1` (default) is the heuristic.
`AUTORUST_CUBLAS_TF32=1` enables TF32, `AUTORUST_CUBLAS_LIB` overrides the path.
A correctness bug found by the FlashAttention tests (double-counting the slice
start in `base_of`) is fixed and covered by `tests/cublas.rs` (plain, transposed
weight + bias, strided-batched attention, bf16).

Results (RTX 3060, f32, `JIT=1`): ViT-B/16 forward **28-29 ms -> 21-22 ms**
(~25%); training step 372 -> 360 ms; 96 of the 120 contractions dispatch (72
linears + 24 batched `QK^T`; the `AV` product is fused with the softmax and
falls back). SDPA forward is unchanged/slightly faster; FlashAttention matches.
cuDNN is intentionally not wired up: the demos' only conv (patch embed) is
negligible, and `HostGemm` is the seam a library conv/attention op would use.

Benchmarks (RTX 3060 / Ryzen 3900X, 24 threads):
- GEMM kernel-only (2026-11): 2048³ f32 generated 3.6 TFLOP/s, cuBLAS 4.8,
  cuBLAS TF32 5.8; 4096³ 3.1 / 4.5. bf16 2048³ generated 10.7 (FFMA, no `mma`),
  cuBLAS 9.8. The old 2.2 TFLOP/s figures included a per-iteration D2H copy.
- ViT-B/16 forward (`JIT=1`): `RUST` ~0.66 s, `CC` ~0.7 s, CUDA **~29-30 ms**
  (was 166 ms), frozen `RUST` plan ~0.87 s.
- ViT-B/16 training step (`vit_train_bench`, forward+backward+Adam, 404
  linears): CUDA ~381 ms/step.

### 9.11 Frozen CPU plans — `autorust::freeze` (2026-11) — DONE

`Jit` now captures multiple corealized roots (`call_many`) and exposes its
compiled LINEARS. `autorust::freeze` (feature `freeze`, which enables
`backend_rust`) lowers a capture into a backend-independent `FrozenPlan`:

- `StorageSpec` (base allocation, external/internal), `ViewSpec` (typed
  window), and `Step::{Kernel, Copy}`.
- Buffer identity is `(base allocation, byte offset, nbytes, dtype)`. Inputs
  and weights are external storages; everything else is allocated once and
  reused. `to_param`/bufferize can create a fresh buffer carrying the caller's
  bytes as `initial_data`; the builder matches those by a content hash and
  prepends an init `Copy`, so the runner receives the live tensor data.
- `FrozenRunner` executes the plan with the fixed `(bufs, vals, gidx)` ABI,
  chunking the first global dimension across `rayon` (the same policy as
  `RustProgram`). It can use JIT-loaded kernels (`FrozenRunner::new`) or a
  statically linked table (`from_fns`).
- `emit_rust`/`emit_to` writes an AOT Rust file: each kernel in its own module
  (duplicate names are safe, `no_mangle` stripped), a `KERNELS` table, and a
  `plan()` constructor. `CompiledProg` now stores its rendered `source` because
  a `ProgCache` hit re-interns PROGRAM without its SOURCE child.
- `Engine` selects `Eager(model)` when CUDA, Metal or CL is available and
  `Frozen(Frozen<M>)` otherwise (Metal was added once the backend landed — the
  check originally only saw CUDA/CL, so a Metal-only Mac took the frozen CPU
  path); `ModuleForward` is the minimal model trait (`Input`/`Output` associated
  types). `Frozen` supports one `Tensor` input and one `Tensor` output today
  (tuple/multi-output wrappers are follow-ups).

Gates: `tests/freeze.rs` (plan vs eager, multi-output/corealize, Engine
selection), `tests/freeze_emit.rs` (emits a standalone crate, compiles it with
Cargo, and byte-compares), and `examples/vit_freeze.rs` (full ViT-B/16: 405
storages, 399 steps, output matches eager).

Build-script layout: a **user-defined** model must be a separate crate that is
a build-dependency of the app (or use an `xtask`), since a build script cannot
import the crate under construction. A packaged model (e.g. `ViT`) needs no
subcrate: `examples/frozen_vit/` includes a shared `make_vit.rs` from both
`build.rs` and `main.rs`. `examples/vit_freeze.rs` with `FREEZE_EMIT=...` shows
the in-process emit side.

### 9.12 Portable C CPU backend — device `CC` (2026-11) — DONE

`backend_clang` adds a second native CPU backend that renders the same
linearized kernels as C (`render_cpu_c`) and compiles them with `$CC` (clang,
else `cc`/gcc) to a shared library loaded by `libloading`. It uses the same
fixed `(bufs, vals, gidx)` ABI, first-dimension chunking, CPU global-grid
schedule and upcast policy as `RUST`, so it is a direct codegen oracle.

- Portable bits: `vector_size` vector typedefs (clang + gcc), `TG_BITCAST`
  (`__builtin_bit_cast` on clang, a union statement-expression on gcc), and a
  `TG_F2U` clamp for tinygrad's float→unsigned truncate semantics.
- `$CC` selection: `CC` env, else `clang`, else `cc`; `AUTORUST_CC_ARCH`
  controls `-march` (default `native`, `none` disables).
- Device selection is now `CUDA → CL → RUST → CC → RAYON → INTERP`.
- Benchmarks (Ryzen 3900X, 24 threads, N=512, 3 iters): `CC` 17.9 ms vs `RUST`
  18.8 ms matmul, 5.2 vs 5.5 ms elementwise. Full ViT-B/16 captured:
  `CC` 4.22 s vs `RUST` 4.11 s (PyTorch CPU ~0.3–0.45 s, tinygrad TinyJit
  ~14 s). The remaining gap to PyTorch is the schedule/kernel algorithm
  (tiling/blocking), not the source language — see §9.10.
- Gate: `backend_harness` passes for `CC` (all dtype/op batteries, including
  the float→unsigned cast cases); `cargo test --features "backend_clang freeze"`
  is green.

`ModuleForward` now lives in `tensor::nn` and is implemented for `Linear`,
`Conv2d`, `ConvTranspose2d`, `BatchNorm` (inference), `LayerNorm`, `LayerNorm2d`,
`GroupNorm`, `InstanceNorm`, `RMSNorm`, `Embedding`, `LSTMCell` (tuple input),
`TransformerBlock`, `Transformer` and `ViT`. `freeze::ModuleForward` is a
re-export, and `examples/frozen_vit/` is a flat build-script package (no model
subcrate) that runs the packaged `ViT` through capture → emit → `Engine`.

### 9.13 OpenCL dynamic loading (DONE 2026-11)

`backend_cl` no longer uses `opencl3` (which hard-linked `libOpenCL`).
`runtime/cl_sys.rs` loads `libOpenCL.so.1`/`libOpenCL.so` (macOS framework,
Windows DLL) with `libloading` and resolves the ~30 `cl_*` entry points used by
`ops_cl`; the backend sees safe RAII wrappers. On a machine without the ICD
loader `ClLib::get()` returns `None`, `available()` is `false`, and the process
starts normally. `ldd target/debug/ocl_smoke` / `readelf -d` show no OpenCL
`NEEDED` entry. `AUTORUST_CL_LIB` overrides the library path for the
missing-library test (`AUTORUST_CL_LIB=/nonexistent` → probe returns false).

Leak discipline: every context/queue/program/kernel/mem/event releases on Drop;
`ClInner` drops the queue before the context, and `ClProgram` drops the kernel
before the program. Constructors defensively release a non-null handle if the
driver also returned an error, enqueue errors release any event, and empty
reads/writes/logs/binaries are short-circuited. The static `libloading::Library`
is intentionally kept for the process lifetime because resolved function
pointers must remain valid; the OS reclaims it at exit.

Gates: `cargo test --features backend_cl` 212 passed; the CL harness 12 passed;
`cl_reduce`/`cl_offset` pass; `ocl_smoke` works on the NVIDIA ICD.

Default features are now `backend_rayon, backend_rust, backend_clang,
backend_cl, backend_metal, backend_cuda, freeze` (2026-11: CUDA and `freeze`
were promoted to defaults for the public release; CUDA costs ~3.5 s of build
time and one dependency, and `freeze` reuses `rayon`/`backend_rust`). Cargo has
no per-target default features, so `backend_cuda` is enabled on every platform
but `cudarc` and all CUDA code are target-gated to Linux/Windows via
`all(feature = "backend_cuda", any(target_os = "linux", target_os = "windows"))`
(the same pattern Metal uses). `cudarc`'s `dynamic-loading` + `fallback-latest`
builds without CUDA libraries (it warns and uses the latest CUDA API if `nvcc`
is absent). Selection remains CUDA → CL → RUST → CC → RAYON → INTERP.

### 9.14 Comparative benchmarks (2026-11, RTX 3060 / Ryzen 3900X)

Numbers below are **after** the 9.10 tiling/grid-order work.

CPU (24 threads, captured steady state, `JIT=1`):
- ViT-B/16 forward: `RUST` **0.65 s**, `CC` **0.63 s**, frozen `RUST` plan
  0.86 s; tinygrad `TinyJit` ~14 s, PyTorch CPU ~0.3–0.45 s.
- 512³ f32 GEMM: `RUST` ~0.72 ms (~370 GFLOP/s); bf16 ~1.3 ms (207 GFLOP/s).
- `FA=1` routes attention through the op-level FlashAttention port and remains
  an algorithm/correctness reference: CPU ViT ~2.5 s vs ~0.65 s for SDPA and
  CUDA ~62 ms vs ~30 ms; not a speedup.

CUDA (RTX 3060, same physical GPU; shared ~30% utilization):
- ViT-B/16 forward (`JIT=1`, f32): **~21-22 ms/iter** with the cuBLAS path
  (~28-29 ms with generated kernels; was 166 ms before tiling).
- ViT-B/16 train step (fwd+bwd+Adam, f32, `vit_train_bench`): ~387 ms.
- 2048³ f32 GEMM kernel-only: 3.6 TFLOP/s generated, 4.8 cuBLAS, 5.8 TF32.
  bf16: 10.7 generated (FFMA) / 9.8 cuBLAS. (Old numbers included the bench's
  per-iteration D2H copy; see §9.15.)
- PyTorch 2.14.1+cu126 f32: ~8.5–9.1 ms/iter steady state. The remaining ~3.4x
  gap is cuBLAS/cuDNN + tensor cores; our f32 path is register-tiled FFMA and
  the bf16 TC path cannot tile the prime 197 dimensions without padding.
- Per-kernel profiler: `AUTORUST_KPROF=1` (syncs per launch; no GPU-counter
  permission needed). Nsight Compute needs
  `NVreg_RestrictProfilingToAdminUsers=0` (or root).

### 9.15 App-level status: clusterfun / DINOv3 (2026-11)

This is the application view (the `clusterfun` DINOv3 clustering port,
`autorust-backend` branch); the engine-side detail is in §9.10–§9.14 and the
attention note below. Folded in from the former `CURRENT_STATUS.md`.

Headline, 448 px, 128 frames, `--preprocess none`:

| backend | time | vs candle |
|---|---|---|
| CUDA + cuBLAS + fused FA2 | **5.6 s** | candle CUDA 6.35 s |
| OpenCL + CLBlast + fused FA2 | **8.4 s** | |
| frozen CPU (AOT plan) | 6.1 s / 16 frames (0.37 s/img) | candle CPU ~0.63 s/img |

With the default preprocessing (260 frames): CUDA 14.1 s, OpenCL 20.1 s,
candle CUDA 15.4 s. Projected to the real ~10k-image set: ~470 s vs ~590 s.
All GPU libraries and backends are dynamically loaded; nothing is needed at
build time. Test counts at the time of writing: `cargo test` default/CL 225,
`backend_cuda` 253, `backend_clang+freeze` 228.

**GEMM policy.** `LibraryGemm::Auto` on CUDA now always uses cuBLAS for f32
(1.3x at 2048³, 2.3-3.5x at the flattened batch-8 shapes); 16-bit/CL/CPU keep
the shape heuristic, and `AUTORUST_GEMM=0` opts out. The kernel-only table and
the `HostGemm` descriptor are in §9.10b.

**Attention.** Three hand-written CUDA kernels behind the fused path (scalar
f32, TF32 `mma`, bf16 `mma` with `cp.async` + `ldmatrix`), plus the OpenCL
kernel and the opt-in `sage` INT8/FP8 variant; see the attention note below.
The fused launch is a captured step: `Session::capturing` is an ordered
`Vec<CaptureEntry>` and the JIT replays `JitStep::{Linear, Host}` in order, so
eager replay reproduces the exact sequence. On CPU the `P·V` product is
materialised so it becomes a `HostGemm` (12 AV products: ~150 ms → ~7 ms per
image).

**Settings on `Ctx`.** `src/settings.rs` holds per-context configuration;
`Settings::from_env()` runs **once per process** via `settings::global()`
(`OnceLock`), so no lowering path pays a `getenv`. Each `Ctx` clones the
process defaults at creation and can be reconfigured:

```rust
let ctx = Ctx::new();
ctx.borrow_mut().settings.attention = AttentionKind::Sdpa;
ctx.borrow_mut().settings.library_gemm = LibraryGemm::Never;   // generated only
let s = Settings::default().with_attention(AttentionKind::Flash).with_beam(4);
```

Migrated so far: `attention` (`AUTORUST_ATTN`, `FA`), `library_gemm`
(`AUTORUST_GEMM`/`AUTORUST_CUBLAS`/`AUTORUST_CLBLAST`), `allow_tf32`
(`AUTORUST_CUBLAS_TF32`, `ALLOW_TF32`), `tensor_cores`
(`AUTORUST_TENSOR_CORES`), `async_pipeline` (`AUTORUST_ASYNC`), `cpu_gemm_tile`
(`AUTORUST_CPU_GEMM_*`), `beam` (`BEAM`), `cache` (`CACHELEVEL`), `sdpa_tile`,
`mm_tile`, `device`. Defaults are "use what is available": `library_gemm` is a
per-backend `{cuda, cl, cpu}` policy, all `Auto`, downgraded only by an
explicit opt-out. Precedence: backend-specific env > `AUTORUST_GEMM` > default.
Presence is separate from policy (`cublas::available()` etc. mean "the library
loaded"), which is what lets two contexts in one process choose differently.
`allow_tf32` controls **cuBLAS only** and is **off by default**: TF32 cuBLAS is
within noise of pedantic f32 at our shapes while changing the model output
(3.1e-4 vs 1.6e-6 max abs in clusterfun), so it is opt-in. The attention
kernels use TF32/bf16 `mma` independently of it.

**Process-wide lowering cache.** `to_program` already reused
arena-independent artifacts from the per-session `ProgCache`; it now also
checks a capped process-wide cache keyed by `(device, AST content)`. A fresh
session that rebuilds the same graph hits it instead of re-lowering/rendering/
compiling: 120/126 hits in the Rayon bench, elementwise 6.2 → 4.5 ms (Rayon)
and 5.5 → 3.0 ms (RUST). The schedule (`rangeify`/kernel split) still runs per
realize; a persistent session or the JIT avoids that.

**Device fallback for virtual buffers.** A virtual result carries no device of
its own. The allocator now uses the **sink-side** device (scanning back through
the graph and ignoring host creation leaves `PYTHON`/`NPY`/`DISK`), and
`realize_tensor`/`realize_many` seed the session's `BufTable` device from the
realized graph. Before, a missing device fell back to `default_device()`: one
transient CUDA availability-probe failure made the process default CL and
silently moved a CUDA graph onto the OpenCL/CPU backend (0 GFLOP/s, one core).
`tests/generated_device.rs` forces `DEFAULT_DEVICE=CPU` and checks that a CUDA
dot still runs and allocates on CUDA.

**Fresh one-shot reads (`*_fresh`).** `Tensor::to_list_fresh` /
`realize_fresh` / `to_bytes_fresh` run in a temporary `Session` that inherits
the tensor's device (`Session::with_device`; `BufTable` uses it for `BUFFER`
uops that carry no device), so a self-contained graph is scheduled on CUDA/CL
instead of falling back to the CPU renderer. An already realized tensor only
copies. A graph whose parameters were realized in **another** session cannot
reconstruct their data in the temporary one (it would allocate empty
replacements), so realize with that session first or use the JIT;
`examples/gemm_bench` realizes in its persistent session.
`tests/fresh_device.rs` covers both paths.

**Environment variables: one OS read per name.** Every env read in `src/` goes
through `helpers::env::{get,get_os}`, which caches the raw `OsString` per name
in a process-wide map; a name is read from the OS **at most once**, whether it
is a behavioural setting, a toolchain path, or a debug switch, and
`get`/`get_os` share the entry. Diagnostics/toolchain keep the env interface
but are cached too: `AUTORUST_{PROF,KPROF,KERNEL_INFO}`, `FROZEN_PROF`,
`AUTORUST_{CUBLAS,CPU_GEMM,MM,DEVICE,ATTN_DEBUG}`, `TG_DUMP_*`, `TG_DEBUG_*`,
`TG_{TRACE,RW_TRACE,RW_VISITS}`, `TG_M5_*`, `TG_{CUDA,OCL}_RENDER`,
`BEAM_DEBUG`, `CACHELEVEL`, `CUDA_PATH`, `RUSTC`, `CC`, `AUTORUST_CACHE_DIR`,
`XDG_CACHE_HOME`, `HOME`, `AUTORUST_RUST_CPU`, `AUTORUST_CC_ARCH`, the library
paths and `{CUDA,CL}_COMPILE_CACHE`. `EMULATED_DTYPES` and `RAYON_*` are
cached the same way; `CLUSTERFUN_{NO_JIT,TIMING}` are app-side. See §9.9.

**RAYON contractions** now use the packed CPU GEMM too: `gemm_device` maps
`RAYON` to `GemmDevice::Cpu` and `ops_rayon` installs the host dispatcher, so a
512³ matmul went **118.8 → 5.8 ms (20x)** and matches RUST/CC. Its typed-IR
schedule is only used for non-contractions (elementwise ~4.5 ms vs RUST
~3.0 ms). The remaining per-realize cost is the schedule (`sched.rangeify` +
`kernel_graph` ~1.5 ms), which the JIT avoids.

**Accuracy** (8 images, 448 px, CUDA vs the frozen CPU plan): CPU vs CUDA
embeddings 1.4e-6 max abs / cosine 0.9999995; CUDA fused attention vs SDPA
8.0e-7; CL fused attention vs SDPA 7.9e-7; `examples/frozen_check` 5.5e-6.
Pedantic cuBLAS + TF32 attention is 3.8e-4 / 0.9999971, i.e. the tensor-core
attention costs about the same as TF32 cuBLAS and is worth it; TF32 cuBLAS is
not, hence the default.

**Known limitations / next steps.**

1. **SageAttention** is implemented (INT8 on sm_80+, FP8 on sm_89+) but
   quantizes K per query block, so it loses to TF32 on the 3060 (1.26 vs
   0.97 ms). Next: a per-head pre-quantization pass (global int8/FP8 K +
   scales) and SmoothK, then gate on embedding drift over the real ~10k set.
   The FP8 path needs an Ada/Hopper card to run `tests/sage_int8.rs`.
   `runtime/sage_flash.rs` (the `dp4a` prototype) is kept only as the measured
   starting point.
2. **`freeze` cannot serialise a host-kernel step.** It panics with a clear
   message; harmless today because the CPU capture never emits one, but a
   GPU-side AOT plan needs `Step::Host` plus a runtime that can bind raw
   pointers.
3. **Attention output buffer** is allocated per layer per batch (~9.7 MB at
   batch 8). Reusing one scratch buffer should recover some of the gap between
   the 2.2x kernel win and the observed ~10% end-to-end win.
4. **GEMM headroom**: the generated f32 kernel now uses TF32 `mma` (5.4 TFLOP/s
   at 2048³ vs 3.0 FFMA; cuBLAS TF32 5.8), bf16 10.8. The FFMA fallback for
   tall/wide shapes is still memory-bound (no shared tiling).
5. Measurement note: the 3900X occasionally carries unrelated load (~11 load
   average), which inflated one CPU run to 66 s. Re-measure before trusting a
   single outlier.
6. The mma fragment layouts are not obvious: for `m16n8k8` the A layout is
   `(g,2t), (g+8,2t), (g,2t+1), (g+8,2t+1)`, the C layout is the same with
   registers 1 and 2 swapped; `tests/tc_mma_probe.rs` pins it down.

**Reproduce** (from the `clusterfun` checkout):

```bash
cargo build --release
./target/release/clusterfun <images> /tmp/out --device cuda --k 4
AUTORUST_ATTN=sdpa ./target/release/clusterfun <images> /tmp/out --device cuda --k 4
FROZEN_PROF=1 ./target/release/clusterfun <images> /tmp/out --device cpu
AUTORUST_KPROF=1 ./target/release/clusterfun <images> /tmp/out --device cuda
```


### 9.16 Conv direct tiling (implicit GEMM) — 2026-11 — DONE

A convolution lowers to `pad -> pool -> expand -> permute -> mul -> sum` over
`(cin, ky, kx)` with four output axes `(batch, cout, oy, ox)`. It is GEMM-shaped
(`M = batch*oy*ox`, `N = cout`, `K = cin*ky*kx`) but has no materialised
im2col, so neither `try_matmul_tiling` (two output axes) nor `HostGemm` (affine
A) matched it. Worse, `try_matmul_tiling` often claimed it anyway with an
M/N pair that left three quarters of a warp idle: the generated conv kernel
used 8 threads mapped to output channels, 4 channels per thread, and strided
weight loads (`KPROF`: the conv reductions were ~150 of ~190 ms on MNIST).

`try_conv_tiling` (runs before `try_matmul_tiling`; needs `has_local`) detects
`reds.len() == 3 && outs.len() == 4` and tiles it as a direct convolution:

- all output channels on threads (32/16/8, coalesced weight and bias loads);
- the innermost spatial axis upcast 4 (or 2) into registers (contiguous x
  loads, weights reused across adjacent outputs);
- the two kernel axes fully unrolled (`kx`, then `ky`), `cin` stays a loop.

Results (RTX 3060, f32): MNIST CNN training bs=32 **57.8 -> 41.8 ms/step**;
bs=512 **588.6 -> 258.4 ms/step** (2.28x). The forward conv kernel alone went
from 86 ms to 21 ms; the generated kernel is now
`r_32_20_5_32_4_32_5_5` (32 threads, 4 accumulators) instead of
`r_32_20_20_8_4_32_5_5`. `nn_conv`/`nn_norm`/`ops` pass and a 20-step MNIST run
still learns (loss 0.506 -> 0.248, 79.7% test accuracy).

### 9.17 BatchNorm statistics + cross-device state buffers (2026-11) — DONE

BatchNorm training fused the conv producer into its mean reduction and
recomputed it (`r_32_16_2_5_5_32_20_20`, ~65 ms); `x` has three consumers
(mean, variance, normalization) and no realization boundary. Realizing `x`
once in `BatchNorm::forward_ctx` fixes that, but exposed three scheduler/state
bugs that had to be fixed first:

- **`empty_like` preferred a host staging device.** A parameter's
  `Contiguous(Copy(PYTHON))` initial-data view made `empty_like` allocate the
  optimizer temporary on PYTHON while the rest of the graph was CUDA. Host-side
  devices (`PYTHON`/`NPY`/`DISK`/`TINYFS`) now fall back to the sink device; the
  cross-device copy is lowered separately.
- **`zeros_param` created state leaves on PYTHON** (device `None`), so Adam
  moments were updated by kernels mixing host and device buffers. It now
  defaults to the process device; BatchNorm's running stats are device leaves
  too (they are buffers, not trainable params).
- **Cross-device copies were only recognised for 0-D/1-D stores.** A
  multi-dimensional `assign` (`Store(Index(dst), Index(src))` over several
  ranges) fell through to the mixed-device check and panicked. A general rule
  now converts any pure same-index store into a COPY.

Results with the realization (RTX 3060 / 3900X, f32):

| workload | before 9.16 | after 9.16 (conv tiling) | now |
|---|---|---|---|
| MNIST CUDA bs=512 | 588.6 ms | 258.4 ms | **39.7 ms** |
| MNIST CUDA bs=32 | 57.8 ms | 41.8 ms | **2.9 ms** |
| MNIST RUST bs=512 | 1664.8 ms | 1664.8 ms | **64.8 ms** |
| MNIST RAYON bs=32 | — | — | 390.7 ms (memory flat) |

`cargo test` passes; a 20-step MNIST run still learns (loss 0.50 -> 0.86 at
step 20, 0.098 / 77.3% at step 70 with the demo's small preset). Note that
`Tensor::realize_many` without the pin is not sufficient here: the buffer can be
reused between the stats and the normalization, which silently breaks training;
`realize_device_ctx` (which pins) is required.

**Still open.** The weight-gradient/BN kernel (`r_32_16_2_5_5_32_20_20`) is
unchanged. **B (deferred): im2col + `dot`**, materialised per batch chunk, to
reuse cuBLAS/CLBlast and especially the packed CPU GEMM. Keep it as a CPU fast
path with a memory budget; the direct tiling plus the realization above already
brings CPU convs into the same order of magnitude.

## Attention variants and hardware capability (status note)

One operation, three implementations. `sdpa` and `flash` compute the same
function (`softmax(q·kᵀ/√d + mask)·v`); `flash` only changes the evaluation
order (online softmax) and is algebraically identical (measured 1.2e-8). `sage`
is a *different* function: QKᵀ runs on int8-quantized operands, so it
approximates.

| | coverage today |
|---|---|
| `sdpa` | everything: masks, causal, dropout, GQA, any `d`, any dtype |
| `flash` | CUDA (`d == 64`, contiguous): causal, additive f32 `[n, n]` mask, GQA, f32 or bf16. OpenCL: plain f32 only. No dropout anywhere |
| `sage` | same subset as flash, plus quantization assumptions |

`flash` has three CUDA kernels behind the same math: the scalar f32 FA2 kernel,
and on `sm_80+` a warp-tiled **TF32 tensor-core** kernel
(`runtime/cuda_flash_tc.rs`) for f32 tensors and a **bf16 tensor-core** kernel
(`runtime/cuda_flash_bf16.rs`) for bf16 tensors. TF32 is the f32-storage
tensor-core format, so the f32 kernel rounds Q/K/P/V to a 10-bit mantissa; the
bf16 kernel uses `mma.m16n8k16` with f32 accumulation. They are selected by the
tensor dtype when `Settings.tensor_cores` is on (the default), independently of
`Settings.allow_tf32` (which only controls cuBLAS). `AUTORUST_TENSOR_CORES=0`
forces the exact scalar f32 kernel; bf16 then falls back to SDPA. An older card
(e.g. `sm_61`, GTX 1060) never compiles the tensor-core sources and keeps the
scalar kernel; a one-time notice says so.

The CUDA kernels implement the features rather than treating them as limits:
causal skips key tiles above the diagonal at block level and masks the diagonal
tiles per row; an additive `[n, n]` f32 mask is added to the scores before the
max; GQA divides the Q head index by `h_q / h_kv` when indexing K/V. Dropout
still needs the general path (it consumes RNG state per element), and the
OpenCL kernel is unchanged (no mask/causal/GQA). On CPU the fused kernel does
not exist, and the graph-level FA port (`models::flash_attention`) measured 16x
slower than split-SDPA, so `flash` there means "same math, no faster kernel".

Fallbacks are therefore semantically safe, but they must be observable: the
plan is per-context counters plus a one-time log line per (kind, reason), and an
opt-in `strict_attention` for benchmarks. `AttentionKind::Sage` currently
degrades to the exact f32 kernel with a notice.

### What the current GPU (sm_86) has and we use

| capability | status |
|---|---|
| TF32 tensor cores (`mma.sync.m16n8k8`) | **used by the fused attention kernel** (`runtime/cuda_flash_tc.rs`): warp-tiled, Q in fragments, K/V through shared, scores in registers. 5.9x over the scalar FA2 kernel at the app shape (7.0 ms -> 1.18 ms for batch 8, h 6, n 789) and ~1e-5 vs the f32 reference. cuBLAS uses TCs for the host GEMMs |
| bf16/f16 tensor cores (`mma.m16n8k16`) | unused in our kernels; the attention dtype is f32 today, so it would need a bf16 model (or an explicit downcast) to be meaningful |
| `cp.async` | **used by the bf16 kernel** (2-stage K/V pipeline, `AUTORUST_ASYNC`, default on): the next tile's copy is issued before the wait for the current one, so the transfer overlaps the MMA. 1.3-1.4x over synchronous staging at the app shape. The same implementation exists in the TF32 kernel but is shape-dependent there (median ~1.04x), so the dispatcher leaves the TF32 kernel on synchronous staging |
| `ldmatrix` | **used by the bf16 kernel**: `m8n8.x2` for the K fragment and `x2.trans` for V, replacing the element-wise shared loads. `tests/ldmatrix_probe.rs` pins both orientations (K and V need opposite forms) |
| warp reductions | used in the new kernel (`__shfl_xor_sync` for the row max/sum); the scalar kernel reduces through memory |
| `__launch_bounds__` | tuned: bf16 `(128,5)` -> 96 regs / 5 blocks/SM, no spills |
| double buffering | used by the bf16 kernel (`cp.async`, 2 stages) |
| CUDA graphs | unused (own JIT replay) |

The generated GEMM now emits `mma` for roughly square outputs: f32 uses the
TF32 layout (5.4 TFLOP/s at 2048^3 vs 3.0 for the FFMA tiling) and bf16 the
bf16 layout (10.8 vs 9.8). The FFMA fallback for tall/wide outputs is still
register-only with **no shared memory** (DRAM-bound at ~3.6 TFLOP/s), so it
wants shared-memory tiling. The attention kernels use `mma` (TF32/bf16, ~15
TFLOP/s bf16) and Sage (INT8/FP8).

### Plan for a newer GPU

Done on every sm_80+ card:

1. ~~TF32 tensor-core FA2 (`mma.m16n8k8`)~~ — `runtime/cuda_flash_tc.rs`.
2. ~~Warp reductions / `__launch_bounds__` in the fused kernel.~~
3. ~~bf16 tensor-core FA2 (`mma.m16n8k16`, f32 accumulate)~~ —
   `runtime/cuda_flash_bf16.rs`, selected when the tensors are bf16. 1.8-2.4x
   the TF32 kernel, matching a bf16-emulating reference to <1e-4.
4. ~~Causal / additive mask / GQA~~ — all three CUDA kernels; causal also
   bounds the key loop to the diagonal block.

Next:

5. ~~`cp.async` double-buffered K/V staging~~ — bf16 1.3-1.4x; TF32 neutral, so it stays opt-in there (`run_ex`).
6. ~~`ldmatrix` for the fragment loads~~ — bf16 K uses `x2`, V `x2.trans`.
7. ~~Occupancy pass~~ — bf16 `__launch_bounds__(128,5)`: 118 -> 96 regs,
   4 -> 5 blocks/SM, no spills. TF32 stays at 122 regs (5 blocks spills).
   `AUTORUST_KERNEL_INFO=1` reports regs/shared/local.
8. ~~Masks/causal/GQA in the OpenCL kernel~~ — same semantics as CUDA
   (additive f32 `[n,n]`, causal key-loop bound, `kh = head / group`).
9. SageAttention: INT8 `mma.m16n8k32` (sm_80+) and FP8 E4M3 (sm_89+) QKᵀ,
   per-row Q / per-key K scales, f32 softmax, bf16 P·V. INT8 is tested on
   sm_86; FP8 is compile-tested for sm_89 and tested by `tests/sage_int8.rs`
   on an Ada/Hopper card.
10. **Sage pre-quantization pass**: the current kernel quantizes K once per
    query block, which makes INT8 slower than TF32 on the 3060 (1.26 vs
    0.97 ms). SageAttention quantizes K once per head into global int8/FP8 +
    scales; that removes the per-block conversion and is the next step.

Only for Ada/Hopper:

11. Hopper-only `wgmma`/TMA/warp specialization — a separate kernel; it does
    not help the 3060.

Arch detection and the PTX cache already key on `sm_XX`, so a newer card needs
no code change to run; `tests/flash_cuda.rs` and `tests/sage_cuda.rs` give
per-arch numbers to add to the README table.

### 9.18 CUDA toolkit-version independence (2026-11) — DONE

The CUDA backend used cudarc's NVRTC loader, which bakes the toolkit version
used at **build** time into its candidate list (`CUDA_MAJOR_VERSION`, so a
CUDA-12 build only tried `libnvrtc.so.12` plus the unversioned `-dev` symlink).
On a machine with a different major version — and no `cuda-nvrtc-dev` symlink —
`Library::new` failed and cudarc's `panic_no_lib_found` aborted the process.
`available()` only checked `libcuda`, so device selection happily picked CUDA
and the failure surfaced at the first kernel compile.

Changes:

- **`runtime/nvrtc.rs`** — a small, version-agnostic NVRTC loader. It binds
  `nvrtcCreateProgram` / `CompileProgram` / `GetPTX[Size]` /
  `GetProgramLog[Size]` / `DestroyProgram` / `Version` (plus the supported-arch
  query when present) through `libloading` and discovers the library the way
  tinygrad's `DLL.findlib` does: `AUTORUST_NVRTC_LIB` / `NVRTC_PATH` first,
  then every `libnvrtc.so[.N]` under `LD_LIBRARY_PATH`, the `CUDA_*` /
  `CONDA_PREFIX` roots, the multiarch dirs and `/usr/local/cuda*` / `/opt/cuda*`
  toolkits (newest version first, ELF check to skip linker scripts), then bare
  sonames. Windows uses the same rules for `nvrtc64_*.dll` / `nvrtc.dll`.
  An explicit override is authoritative; the loader does not silently fall back
  to a different toolkit.
- **Driver-compatible NVRTC fallback:** `nvrtc::compile_validated` tries every
  loaded NVRTC newest-first, compiles the probe source with each, and hands the
  PTX to a caller-supplied validator. The CUDA probe passes `cuModuleLoadData`,
  so an NVRTC newer than the installed driver (for example a CUDA 13.4 toolkit
  on a CUDA 13.2 driver) is skipped and the next-oldest installed toolkit is
  pinned for all later compiles. `CudaInner::open` runs the same validation when
  CUDA is requested directly without the availability gate. An explicit
  `AUTORUST_NVRTC_LIB`/`NVRTC_PATH` leaves one candidate, so pinned toolkits
  keep the old no-fallback behavior. A one-line `autorust: pinned ...` note is
  printed when a fallback is used.
- **`nvrtc::compile(src, arch, include_dirs)`** returns `Err(String)` with the
  program log. It pre-checks the device SM against `nvrtcGetSupportedArchs`
  and returns e.g. `NVRTC 12.4 does not support sm_120 (supported: ...)`
  instead of a generic compilation failure.
- **`ops_cuda::device_available(ordinal)`** is the new selection gate: driver
  present, `CudaContext::new`, compile a one-line kernel for the device's SM,
  and `cuModuleLoadData` the resulting PTX. Cached per ordinal (mixed-GPU
  boxes can have one card the installed NVRTC cannot target). On failure it
  logs one line and `available()` returns `false`, so `default_device()` falls
  through to the next backend. `autotune::is_available`/`available_devices`
  use the per-ordinal probe too.
- The epilogue kernel, the Sage compile-tests and `bin/cuda_smoke` use the same
  loader, so the main path no longer touches cudarc's NVRTC at all. cudarc
  remains the CUDA *driver* layer (context/stream, buffers, module load,
  launches); its only remaining `nvrtc` use is the `Ptx` value type required by
  `CudaContext::load_module`, which does not dlopen anything.
- **`runtime/libfind.rs`** — the discovery (overrides, search dirs, version
  parsing/sorting, ELF check) is shared with cuBLAS, whose hardcoded
  `libcublas.so.{13,12,11}` list had the same cross-toolkit limitation. cuBLAS
  is now scanned the same way and probed by creating/destroying a handle on the
  device's current context (`probe_current_context`, called from the CUDA device
  probe and `CudaInner::open`). A cuBLAS from the wrong toolkit or a GPU it does
  not support logs one line and disables the host-GEMM fast path; generated GEMM
  kernels keep CUDA running. `AUTORUST_CUBLAS_LIB` remains an authoritative
  override.
- The CUDA compile path (batch, cache key, parallel workers) is unchanged; only
  where the PTX comes from changed.

Gates: `runtime/libfind.rs` unit tests cover NVRTC/cuBLAS filename parsing and
version ordering (including rejecting `libcublasLt`); `runtime/nvrtc.rs` tests
cover SM parsing, a clean missing-library error, unsupported-arch diagnostics,
a real trivial-kernel compile when NVRTC is present, and that
`compile_validated` reports every rejected candidate. Manually:
`AUTORUST_NVRTC_LIB=/nonexistent` disables CUDA, prints the warning and runs
MNIST on OpenCL with identical accuracy; `AUTORUST_CUBLAS_LIB=/nonexistent`
keeps CUDA running on generated GEMM kernels; the normal run still opens the
RTX 3060 (`sm_86`) and emits the same cuBLAS GEMMs. On a box whose toolkit is
newer than the driver (NVRTC 13.4 with a CUDA 13.2 driver), `cargo run --bin
cuda_smoke` now falls back to the installed NVRTC 12 and passes without
`AUTORUST_NVRTC_LIB`. Full suite + clippy + rustdoc pass.

Known limits: the driver still has to understand the final PTX ISA, but the
probe now retries older installed NVRTCs before disabling CUDA; include headers
still come from `$CUDA_PATH/include` or the three standard locations, so a
non-standard toolkit should set `CUDA_PATH`.

### 9.19 Benchmark round: prime-shape GEMM tiles + attention default (2026-11) — DONE

`bench/results/` and [`PERFORMANCE.md`](PERFORMANCE.md#cross-framework-comparison)
are the source of truth for the numbers; this entry records the two changes.

1. **GPU matmul tiling for prime-ish row extents** (`codegen/postrange.rs`).
   `try_matmul_tiling` picked the largest `UPCAST` that divides the extent and
   only then the thread split. At ViT batch 8 the GEMM row count is
   `1576 = 8*197`, so an 8-wide register tile left exactly one thread on M:
   16-thread blocks for the FFN expansion and every backward GEMM with the same
   M. Small shapes were fine only by luck.
   The M axis now steps the register tile down (8 -> 4 -> 2) when the chosen
   tile yields fewer than two threads, but only for the `large` (`M*N >= 1M`)
   configuration. The N axis keeps the largest vectorizable tile, and the
   small/medium configuration is untouched. Generated kernel-only (RTX 3060,
   `NOSYNC=1`, `AUTORUST_GEMM=0`):
   - `1576x3072x768` (batch-8 FFN up): 8.1 -> 2.9 ms (0.93 -> 2.57 TFLOP/s)
   - `1576x768x3072` (down/backward): 7.2 -> 1.7 ms (1.03 -> 4.31 TFLOP/s)
   - `1576x768x768` (projection): 1.65 -> 0.49 ms (1.13 -> 3.82 TFLOP/s)
   - `2048^3` and `1024^3` unchanged (3.2 ms / 0.7 ms). cuBLAS remains the
     library path when `HostGemm` matches.
   ViT-B/16 inference, CUDA bs 8: 213 -> 105 ms (tinygrad 136 ms); bs 1: 20.5 ->
   15.1 ms. MNIST CUDA 41.0 -> 39.2 ms. Training kernels are unchanged (the
   differing shapes are all `small`).

2. **The op-level FlashAttention port is opt-in again** (`models/transformer.rs`).
   `TransformerBlock::attn` switched to the graph-level FA port whenever
   `Settings::attention != Sdpa`. `Settings::attention` defaults to `Flash`
   because that is the *tensor-level* setting (the fused kernels reached
   through `Tensor::attention_with`), so the model silently used the 16x slower
   op-level port on every backend. `FA=1` (read once, as `CURRENT_STATE §9.14`
   already documented) is now the only switch; the default model path is
   tinygrad's SDPA. This was worth another 213 -> 161 ms at bs 8 before the
   tiling change.

Also in this round: `fused_attention_ex` co-realizes q/k/v/mask with
`Tensor::realize_many` instead of realizing them one at a time (they share the
projection input, so sequential realizes re-scheduled the whole prefix graph).
The fused path needs a session, so this does not move the graph-model
benchmarks; it removes the redundant prefix work for `attention_with` callers.
`bench/run.py` no longer forces `AUTORUST_ATTN=flash` on CUDA.

Gates: `cargo test --release` (the one `causal_skips_half_the_work` timing test
is load-sensitive and passes in isolation), `cargo clippy --all-targets`,
`cargo rustdoc --lib -- -W missing_docs`, `cargo build --no-default-features`.

### 9.20 Benchmark round 2: rendered-kernel cache, conv/GEMM tiles, fused ViT inference (2026-11) — DONE

Goal: cut the scheduler/codegen cost of the benches and close the gap to torch
(no `torch.compile`). Numbers below are `bench/results/` at `d6050aa`;
`PERFORMANCE.md` has the curated tables.

**Persistent rendered kernels** (`engine/realize.rs`). `prepare_program` calls
`full_rewrite_to_sink_with` (~30 graph-rewrite stages) per kernel: 4.4 s of a
ViT-B/16 training cold start over 1148 kernels, even with the PTX already in
the source-keyed disk cache. The render result (ProgramInfo, ELF signature,
source) is now stored under `gkey` (device + AST content) plus a hash of all
`Settings`, the `USE_TC`/`TC_OPT`/`TC_SELECT` knobs and a **fingerprint of the
running binary** (crate version + executable mtime/size). The binary fingerprint
means any rebuild invalidates the entry, so a stale render cannot outlive the
codegen that produced it; `CACHELEVEL=0` disables it. A hit rebuilds the
PROGRAM from the cached metadata and fetches the PTX from the existing
compiler cache (the `linear`/`sunk` srcs are not used by cached programs —
`reuse_program` already builds a bare PROGRAM node).

Warm-cache cold start (`ms_cold`, one process, warm PTX+render cache):

| workload | before | now |
|---|---|---|
| MNIST bs512 | 505 ms | **205 ms** |
| ViT infer bs1 | 697 ms | **456 ms** |
| ViT infer bs8 | 1279 ms | **721 ms** |
| ViT train bs1 | 6013 ms | **2602 ms** |

**Codegen tiling.** Three findings from the new per-stage timings
(`codegen.sink` is ~3.7 ms/kernel, spread over ~30 stages; `apply_opts` and the
matmul tile are the rest):

1. The direct-conv tile put `cout` on threads and upcast only `ox`; every
   thread re-read the 5x5 weights for 4 outputs. Upcasting the other spatial
   axis reuses the weights across two rows: MNIST bs512 **39.2 -> 21.7 ms**
   (torch 13.1). The existing conv goldens all have `cout < 8` and never reach
   `try_conv_tiling`, so `tests/conv_tiling.rs` adds a CUDA-vs-INTERP
   differential (also re-run from a warm render cache).
2. `matmul_ranges` labels A's private axis M and B's as N *in operand order*,
   not store order. For the ViT backward `dX` ([197, 3072], K=768) that put the
   prime token axis on N: 16 rows of grid with 2-thread blocks, 5.6 ms/kernel.
   Score both orientations by block threads (and total upcast registers) and
   keep the better one; if the winner still has < 32 threads, trade the
   register tile for threads up to a warp. ViT training **332 -> 216 ms**.
3. The multi-reduce q/k/v weight-gradient kernel is still fused
   (`r_197_48_16_768_...`, ~67 ms/step): it has three reductions, so
   `matmul_ranges` declines it and the generic heuristic emits scalar loads.
   Splitting those stores is the next scheduler item.

**Fused inference path** (`models/transformer.rs`, `models/vit.rs`). The graph
model always used SDPA, so the hand-written FA2 kernels were reachable only by
calling `Tensor::attention_with` directly. `ViT::forward_with(sess)` (and
`TransformerBlock::forward_with`) now runs in eval mode on CUDA/CL/Metal:
realize the layer input, realize the q/k/v projections **before** the
reshape/transpose (so they stay library GEMMs rather than transposed-store
kernels — that ordering was the bug in the earlier attempt), then let
`attention_with` copy/launch the fused kernel. Training and CPU fall back to
the graph path. The projections are co-realized and **pinned**: `realize_many`
leaves only weak `BufTable` views, and without the pin the temporaries were
freed while the q/k/v views still indexed them (nondeterministic garbage;
`tests/nn_models::vit_session_path_matches_graph_path` pins it). ViT inference
CUDA bs8 **102 -> 69.9 ms**, bs1 **15.8 -> 13.2-15.9 ms**.

**Torch comparison** (median, no `torch.compile`): MNIST 21.7 vs 13.1 = 1.66x;
ViT infer bs1 15.9 vs 8.7 = 1.82x; bs8 68.9 vs 46.4 = 1.49x; ViT train 216 vs
43.8 = 4.9x. The three inference/MNIST workloads are within 2x; training is not
(the fused multi-reduce gradients + f32 vs cuDNN). CPU (unchanged code path) is
64.2 ms MNIST (faster than torch 175.9), ViT infer 438/1862 vs 165/701 (2.6x,
MKL), train 2230 vs 616 (3.6x).

Gates: `cargo test --release` (78 binaries), `cargo clippy --all-targets`,
`cargo rustdoc --lib -- -W missing_docs`, `cargo build --no-default-features`;
the conv and ViT-path differentials pass from a warm render cache.

### 9.21 Split summed contractions before kernel splitting (2026-11) — DONE

The largest remaining ViT training kernel was the attention input gradient
`dQ@Wq + dK@Wk + dV@Wv`: three contractions summed into one store. The
scheduler fused them into a single kernel with three reductions; `matmul_ranges`
declines multi-reduce kernels, so the generic heuristic emitted three scalar
loops (5.7 ms/layer, 30% of the step). Extending `matmul_ranges` to multi-reduce
sums was tried and **regressed** (216 -> 459 ms: the shared-output tile made the
single-reduce kernels it also accepted much worse), so that approach was
dropped.

Instead `pm_generate_realize_map` now marks both operands of an
`ADD(contraction, contraction)` for realization. Each contraction gets its own
STAGE, becomes a single-reduce store that the matmul tiler / host GEMM claims,
and the sum is an elementwise kernel. `contraction_root` unwraps casts,
movement ops and the reduce body's view down to the `MUL`; the rule stays
narrow on purpose (a version that also descended through ADD trees measured
181.2 ms, and through all ops 209.6 ms, both worse than the direct-operand
rule):

- ViT-B/16 training CUDA: **216.4 -> 179.4 ms** (torch 43.8, tinygrad 388.5).
- ViT-B/16 training CPU `RUST`: 2230 -> 2067 ms.
- Warm cold start still improved (6013 -> **2541 ms**) because three plain GEMM
  kernels render cheaper than the fused multi-reduce one.
- Inference/MNIST unchanged (the rule only fires on summed contractions).

Gates: full `cargo test --release` (78 binaries, including the grad/optim
goldens), clippy, rustdoc, `--no-default-features`.

### 9.22 Why the remaining backward GEMMs miss the library path (2026-11) — ANALYSIS

After §9.21 the ViT training profile is dominated by three generated GEMM
families (~102 of 180 ms/step). `AUTORUST_MM_DEBUG=1` now prints the reason
`host_gemm::extract` declines each one (`[hg] reject: ...`):

| reject | count/step | kernel | cost |
|---|---|---|---|
| `A operand is Mul` | 12 | FFN `dX` with the GELU-derivative prologue (`r_197_3_64_4_768_4`) | ~43 ms |
| `B operand is Where` | 24 | ff2 `dX` with the dropout mask on `dY` (`r_197_12_64_4_192_4`) | ~32 ms |
| `A operand is Exp2` | 12 | softmax backward (`r_197_3_64_4_12_16_4`) | ~27 ms |

All three are true contractions with a *pointwise prologue* on one operand.
`HostGemm` can only describe affine buffer operands, so the generated kernel
recomputes the prologue inside the K loop (the GELU case spends ~10 extra FLOPs
per MAC).

Materializing the prologue once was attempted at the scheduler level and did
not land: `pm_generate_realize_map` records the prologue, but
`convert_reduce_to_ranges` lowers the reduce body atomically, and by then the
body is a `Permute`/`Shrink` of the `MUL` (movement folding), so the generic
`create_bufferize_and_index_srcs` never visits the operand. A dedicated
"bufferize a pointwise contraction operand inside a reduce body" pass (the
reduce-body equivalent of `create_bufferize_and_index_srcs`) is the concrete
remaining work; it is a rangeify change, not a `host_gemm` matcher change.

Bound: even if all three reached cuBLAS, the affected kernels are ~100 ms of
the 180 ms step and cuBLAS f32 is at most ~2x the generated kernels at these
shapes (PERFORMANCE.md: 4.76 vs 3.58 TFLOP/s at 2048³; ~5-6 vs 2-3 here), so
the training step lands around 130-140 ms, i.e. ~3x torch, not 2x. Closing the
rest needs f32 tensor cores (TF32 changes the output) or a proper
im2col/GEMM-backward rewrite. This is why the three inference/MNIST workloads
are within 2x and training is not.

### 9.23 Stage pointwise contraction operands inside reduce bodies (2026-11) — DONE

§9.22 identified the remaining ~100 ms/step as contractions whose operand is a
pointwise prologue (`A operand is Mul` = GELU-derivative FFN dX, `B operand is
Where` = dropout-masked dY, `A operand is Exp2` = softmax backward). The
realize-map route could not stage them: `convert_reduce_to_ranges` lowers the
reduce body atomically, and after movement folding the body is a `Permute`/
`Shrink` of the `MUL`, so the generic `create_bufferize_and_index_srcs` never
sees the operand.

`stage_reduce_prologue` / `stage_prologue_in_body` now run at the start of
`convert_reduce_to_ranges`, before `create_bufferize_and_index_based_on_ranges`.
They descend the body's cast/movement chain to the `MUL`, replace the recorded
non-affine operand with `INDEX(STAGE(operand))` over its own axes (all axes
realized), and copy the `range_map` entry to the rebuilt reduce. Guards found
by the differential suite:

- already-materialized operands (`Param`/`Buffer`/`After`/`Stage`/`Store`/
  `Index`) are skipped — re-staging them put two stores in one kernel and
  tripped `coalesce` ("attempting multiple stores");
- the staged operand must have a *different shape* than its affine partner:
  `sum(x * mask)` and LRN/Lp `x*x` are not contractions, and staging them broke
  the cross-entropy / Lp goldens;
- movement-view operands are staged only while `TRAINING` and below 4M
  elements: the backward re-reads them, but an eval forward would just pay the
  copy (ViT inference 69 -> 122 ms without the gate), and staging MNIST's
  9.4M-element conv/pool views OOMs bs512.

Results (medians, `bench/results` at `43fe9d8`): ViT-B/16 training CUDA
**179.4 -> 94.2 ms** (torch 43.8 = 2.15x, tinygrad 388.5 = 4.1x faster), CPU
RUST 2221.8 -> **1754.9 ms**. MNIST 21.7 ms (1.66x), ViT inference 13.6/72.3 ms
(1.56x/1.56x) unchanged. Warm cold start: 216 / 474 / 760 ms, training 2498 ms
(from 505 / 697 / 1279 / 6013).

Remaining gap: training is still 2.15x torch. The profile no longer shows a
single dominant prologue kernel; the rest is the f32 backward (generated GEMMs
~2-3 TFLOP/s vs cuBLAS ~5-6) plus elementwise work. Closing to 2x needs f32
tensor cores (TF32 changes output) or a GEMM-backward rewrite, not another
scheduler cut.

### 9.24 CPU host GEMM: direct operands, packing once, measured ceiling (2026-11) — DONE

The packed CPU GEMM re-packed `A` for every `(m,n)` tile and `B` for every
m-block. On ViT's 197-row linears that packing traffic dominated the
arithmetic (B re-read 5x, A up to 8x). `run_raw` now has a direct path when
`a_k == 1` (row-major A) and `b_n == 1` or `b_k == 1` (row-major or a `[n,k]`
weight): A is read in place with its own strides, and a transposed weight is
copied once per GEMM into a contiguous `[k,n]` buffer. The microkernel takes
`(a_rs, a_ks, b_ks)` strides; the old per-task packing remains as the
column-major fallback.

Microbenchmark (`gemm_bench`, 24 threads, f32, GFLOP/s; MKL = torch 2.6):

| shape | generated | packed before | packed now | MKL |
|---|---:|---:|---:|---:|
| 197x768x768 | 94 | 273 | **290** | 93 |
| 197x3072x768 | 30 | 284 | **296** | 417 |
| 197x768x3072 | 65 | 315 | **464** | 613 |
| 1576x768x768 | 156 | 313 | **375** | 609 |

Tile retuning (MC/NC/KC sweeps) does not beat the 48/96/256 default on these
shapes. End-to-end `RUST` medians are within noise of before (ViT infer 1
425 ms, bs8 1902 ms, train 1784 ms; MNIST 65 ms): `kx.host` is ~100 ms of a
~420 ms inference step, so a MKL-speed microkernel would only reach ~2.4x
torch. The remaining CPU time is generated elementwise/attention kernels — a
codegen/vectorization project, not a host-GEMM one. Added a unit test for the
`[n,k]` weight layout with bias/alpha/residual epilogue.

Gates: full `cargo test --release` (78 binaries), clippy, rustdoc.

### 9.25 CPU per-kernel profile + measured ceiling; epilogue split blocked (2026-11)

`AUTORUST_KPROF=1` now works for host backends too: `exec_kernel` prints a
`KPROF <name> <secs>` line per generated kernel and per `HostGemm` call (CUDA
keeps its own stream-synchronised profiler). RUST ViT-B/16 inference bs1
(3-step trace, per-step column):

| kernel | /step | what |
|---|---:|---|
| `r_384_197_8_192_4` | **271 ms** | FFN expansion (`linear` + GELU fused) |
| `HostGemm` (177 calls) | 80 ms | the library GEMMs that do match |
| `r_96_197_8_12_16_4` | 32 ms | attention/output projection fragment |
| `r_197_8_12_8_197` | 30 ms | attention P·V / softmax |
| elementwise (`E_*`) | ~40 ms | layernorm, GELU, adds, masks |

So ~91% of the 420 ms inference step is one generated FFN-expansion kernel:
the GELU epilogue blocks the host GEMM, and the generated matmul runs at
~30 GFLOP/s vs 296 GFLOP/s for the packed GEMM on that exact shape.

Two ways to materialize the expansion were tried and **blocked**:

1. Model-level (`forward_mask_with` realizing the expansion and activation on
   CPU): the session-aware CPU path re-schedules the prefix per realize; ViT
   inference went 425 -> 1539 ms (bs8 1502 -> 15087 ms). Reverted.
2. Scheduler-level (mark the contraction for realization in the realize map /
   consumer-driven pass): the kernel splitter's END contract
   (`sched.rs:110 TGERR: END src[0] should be KERNEL`) rejects a realized
   reduce that is not a kernel root. A consumer map guard (skip contractions
   feeding store/end) did not change it. Not shipped.

The fix is a scheduler feature: split a non-affine epilogue into a second
kernel (a GEMM + elementwise pair). With it the FFN expansion would drop from
~271 ms to ~27 ms and CPU ViT inference to roughly ~150 ms (~0.9x torch);
without it the CPU row stays at 2.6-2.9x.

Training ceiling re-measured: Auto 96.9-97.4 ms, `GEMM=all`+TF32 93.4 ms
(2.13x), tensor cores off 94.4 ms. The generated f32 tensor-core path is
shape-gated on divisibility and declares the prime 197 shapes ineligible
(masked vector stores are not lane-wise), so it cannot bound the f32 ceiling
without a numerics change.

### 9.26 Epilogue split: reproduce the END contract failure; TC measurements (2026-11)

Attempting the CPU epilogue split (mark a contraction for realization so the
host GEMM takes the matmul and GELU becomes a separate kernel) reproduces a
scheduler contract failure, precisely located:

```
[bad-end] u567593 src0=u567592 Store srcs=[Index, Mul]
TGERR: END src[0] should be KERNEL   (schedule/sched.rs create_schedule)
```

The realized reduce is emitted by `bufferize_to_store` as
`AFTER(buf, [END(STORE)])`, but the END is created inside the consumer's
`split_store` rewrite, after the bottom-up `split_kernels` walk has passed that
position, so it is never converted to `END(CALL)`. Instrumenting the splitter
also shows the producer END is skipped as "open ranges":

```
446 End   open
450 Store open
```

i.e. the reduce realization's output-axis bookkeeping leaves a non-Device range
in the END subtree, so even a second `split_kernels` pass would decline it.
Fixing the epilogue split therefore needs (a) callifying producer END(STORE)
nodes created by `bufferize_to_store` (fixpoint or explicit producer pass) and
(b) staging a reduce with only its output axes. Neither is a matcher tweak; the
measured payoff remains: the FFN expansion kernel is 271 ms/step vs 296 GFLOP/s
for the packed GEMM on the same shape (~370 ms of CPU ViT inference).

Generated f32 tensor-core path (ruling it out as a ceiling lever): with
`TC_DEBUG=1` the ViT shapes are declined (`no tensor core available`,
`output is too asymmetric` for the prime 197 axes), and
`AUTORUST_TENSOR_CORES=0` measures 89.8-94.4 ms vs the default 94.3-97.4 ms —
i.e. the TC attempts are a ~5% *cost* on these shapes, not a headroom. The
generated f32 matmul ceiling stays at the `GEMM=all`+TF32 measurement (93.4 ms
= 2.13x torch).

### 9.27 CPU epilogue split: materialize interior contractions (2026-11) — DONE

§9.25/§9.26 left the CPU FFN expansion (`linear` + GELU fused) as one generated
kernel at ~271 ms/step, against ~296 GFLOP/s for the packed host GEMM on the
same shape. The blocker was structural: realizing a reduce that sits *inside* a
consumer's elementwise epilogue produces `AFTER(buf, [END(STORE)])` nodes whose
END does not close the loop ranges the realization adds, so `split_store`
declined them ("open ranges") and `create_schedule` asserted on the resulting
`END(STORE)`.

Two rangeify changes fix that:

- `close_store_ends` (run after `pm_add_buffers`, before `split_kernels`) adds
  the store's remaining loop ranges to every kernel-root `END(STORE)`. The
  STORE/END pairs created by `bufferize_to_store` are built from the
  pre-rewrite value, so a realized interior source can contribute ranges the
  END never saw; closing them makes the store a proper kernel.
- `realize_interior_contractions` (run after the consumer map in
  `run_rangeify`) realizes a contraction only when it is worth it:
  - host backend, one reduction axis with `K >= 512` (attention's K = 64/197
    kernels stay fused),
  - not already a kernel root (a store value is a GEMM already),
  - exactly one plain-buffer operand (the weight). A weight-gradient
    contraction has two activation operands and stays fused; splitting those
    measured ~14% slower on MNIST,
  - every non-affine operand can be staged (<= 8M elements) — otherwise the
    reduce would become a generated GEMM with a fused prologue, which is
    slower than the fused kernel. The one both-operand exception is `x * x`
    with a small (< 512k) activation, i.e. a normalisation variance, whose
    split materializes the normalised activation the projections read.

The staged operands are what make the split pay: the FFN expansion's `A` is the
layer-norm output (not a buffer), so without staging the new GEMM missed the
library path and was *slower* than the fused kernel (measured: 584 ms vs 425).

Also: a batched CPU host GEMM packs an unbatched weight once per call instead
of once per batch (`run_raw`), which matters at bs 8 ([1576, 3072] x
[3072, 768] repacked 8x before).

Medians (`bench/results`, RUST, torch 2.6 without `torch.compile`):

| workload | before | now | torch | ratio |
|---|---:|---:|---:|---:|
| ViT infer bs 1 | 425.3 ms | **227.1** | 165.1 | 1.38x |
| ViT infer bs 8 | 1902.2 ms | **1601.3** | 701.5 | 2.28x |
| ViT train bs 1 | 1783.8 ms | **1660.7** | 616.3 | 2.69x |
| MNIST bs 512 | 65.1 ms | **64.7** | 175.9 | 0.37x |

`AUTORUST_KPROF=1` now shows 216 `HostGemm` calls (~95 ms/step) and no
FFN-expansion kernel at bs 1; the largest remaining kernels are the attention
QK^T/PV (K = 64/197, fused on purpose). CUDA is unchanged (the pass is
host-only): MNIST 21.8 (1.67x), ViT infer 14.7/71.6 (1.69x/1.54x), train 96.8
(2.21x).

Gates: full `cargo test --release` (78 binaries), clippy, rustdoc,
`--no-default-features`, `cargo publish --dry-run`.

### 9.28 CPU attention into the packed GEMM; 6 of 8 rows within 2x (2026-11) — DONE

§9.27 brought the CPU FFN expansion onto the packed host GEMM. The remaining
RUST ViT cost was attention: `r_12_197_197_2_4_16_4` (QK^T) took 43 ms/layer at
bs 8 and `r_12_197_8_8_8_197` (P·V) 27 ms/layer, i.e. ~500 ms + ~320 ms per
step for 0.06 GFLOP each. The generated kernels re-read a 64-float Q/K row per
output element (the score kernel's bias adds are inside the reduction, so the
operands are not buffers and cuBLAS-style reuse never happens).

`realize_interior_contractions` now also **stages the pointwise operand of any
contraction with >= 16M MACs** (operand <= 4M elements), independently of the
epilogue split. Attention's `Q + q_bias` and `K + k_bias` become buffers, so the
score kernel is a plain batched matmul the host GEMM claims; `P·V` reaches it
directly because the softmax output is already a buffer. The split rule gained
corresponding cases: both operands already buffers (attention) needs
`>= 16M` MACs, one buffer + one expression keeps `K >= 512` (the forward
projections), and a weight-gradient pair (two expressions) stays fused.

`AUTORUST_KPROF=1`, RUST ViT-B/16:

| workload | before | now | torch | ratio |
|---|---:|---:|---:|---:|
| infer bs 1 | 227.1 ms | **173.2** | 165.1 | **1.05x** |
| infer bs 8 | 1601.3 ms | **1163.8** | 701.5 | **1.66x** |
| train bs 1 | 1660.7 ms | **1634.2** | 616.3 | 2.65x |
| MNIST bs 512 | 64.7 ms | **64.1** | 175.9 | 0.36x |

The bs-8 profile is now ~75% `HostGemm` (12 layers x [1576,3072,768] and
[1576,768,3072] plus 4 projections); the generated kernels that remain are the
softmax and small elementwise passes. On this CPU (AVX2, 24 threads) the packed
kernel sustains 290-464 GFLOP/s against MKL's ~600, and 8-12 ms of the step is
the layer's fused softmax; that is the measured CPU floor for the current
microkernel, not a scheduling gap.

CUDA is unchanged (the pass is host-only): MNIST 21.9 (1.67x), ViT infer
14.0/69.3 (1.61x/1.49x), train 97.2 (2.22x).

Gates: full `cargo test --release` (78 binaries), clippy, rustdoc.

### 9.29 Training rows: where the remaining time actually is (2026-11) — ANALYSIS

The two rows still above 2x are the training ones. Both were profiled
per-kernel this round; neither gap is scheduling or GEMM dispatch.

**CUDA training (96.8-124 ms/step, torch 43.8).** `AUTORUST_KPROF=1`
stream-synchronised per-launch trace, 4 steps (421.6 ms of kernels total):

| category | launches | kernel time | share |
|---|---:|---:|---:|
| elementwise (`E_*`) | 12,095 | 254 ms | **60.2%** |
| generated matmul / attention (`r_*`) | 1,413 | 168 ms | 39.8% |
| cuBLAS | 0 | — | — |

163.6 ms of the 421.6 (39%) is in kernels under 50 µs each: the optimizer's
per-parameter moment updates (`E_2`: 6,373 launches), bias/scale epilogues and
the loss. Auto never picks cuBLAS for these training shapes, so the 2.2x is not
a library-dispatch gap — it is elementwise work plus per-launch overhead plus
the generated f32 matmul rate. The earlier measured library ceiling (91.9 ms
with `GEMM=all` + TF32 = 2.10x) bounds only the matmul part, which is 40% of
the step; the elementwise 60% would have to be fused (or the optimizer update
batched) to move this row.

**CPU training (1.61-1.65 s, torch 616 ms).** The fused FFN-backward kernel
`r_384_197_8_192_4` is 1,220-1,284 ms over 36 launches (~310 ms/step), and
`host_gemm::extract` never sees it: instrumenting the extractor shows no reduce
of that output size is ever offered, i.e. the contraction sits inside a store
whose top is not the reduce (it is a realised sum from §9.21 with a
dropout/`where` prologue). The rest of the step is ~120 ms of contiguous copies
(`E_147456_4` = 768 copies of a 589,824-element buffer per step) and generated
attention fragments.

Attempted this round and **not** landed: moving the operand-staging step ahead
of the "already realized / kernel root" guards in
`realize_interior_contractions`, and carrying the realize mark across the
`PAD -> WHERE` rewrite, so a §9.21-realized backward sum would stage its
`where(...)` operand. Neither changed the kernel list (the extractor still never
sees the reduce), so both were reverted; the remaining work is to make
`extract` accept a store whose value is an elementwise op over a realised
reduce, or to fuse the factorising elementwise kernels, not another rangeify
guard.

Measured state unchanged for the eight rows of `bench/results`: CUDA 21.9 /
14.0 / 69.3 / 96.8 ms and RUST 64.1 / 173.2 / 1163.8 / 1634.2 ms, i.e. six of
eight within 2x of torch.

### 9.30 Training rows, round 2: what the remaining kernels are (2026-11) — ANALYSIS

Follow-up to §9.29. Two CPU staging attempts were measured and reverted, and the
CUDA small kernels were identified. Both rows are still above 2x; nothing in
this round changed the benchmark medians, and the tree is unchanged apart from
this note.

**CPU: the FFN-backward kernel is fused and no staging path reaches it.**
`r_384_197_8_192_4` (~310 ms/step) is `Mul(gelu'(ff1_out + bias), Reduce)`:
the reduce is the masked activation gradient and the store value is the
nonlinear epilogue. `host_gemm::extract` peels `Cast/Mul/Add` around the reduce
and stops at `Mul`, so it *does* reach the reduce — but instrumentation shows no
reduce of that output size is ever passed to it, i.e. the kernel comes from a
path where the store is an elementwise op over an already-realized reduce and
`split_store` never offers those nodes to the extractor.

Tried and reverted (both measured, no kernel-list change):

- adding `non_removable` to the §9.21 contraction-sum realizations, so
  `pm_remove_bufferize` cannot re-inline them (train 1.553-1.627 s, unchanged);
- (previous round) moving operand staging ahead of the realized/root guards and
  carrying the mark across `PAD -> WHERE` (unchanged).

The measured CPU step is now ~43% `HostGemm` (393 calls, ~700 ms), ~310 ms the
fused kernel above, ~120 ms of contiguous copies (`E_147456_4`: 192 copies of a
589,824-element buffer per step) and the rest generated fragments. Closing it
needs `extract` to accept a store that is an elementwise op over an *already
realized* reduce (i.e. hoisting the epilogue into the consumer kernel after
`split_kernels`), which is a schedule-level change, not another rangeify guard.

**CUDA: two tiny kernels account for a quarter of the step.**
Dumping the sources identifies the sub-50 µs kernels:

- `E_2` (6,373 launches in 4 steps ≈ 1,600/step, 11 ms/step) is a **two-element
  counter update**: `data0[0] += 589824` with carry into the high word, i.e. a
  device-side kernel counter (the increment is the GEMM's element count). It
  could be accumulated on the host for the same reported value.
- `E_4608_32_4` (768 launches ≈ 192/step, 15.9 ms/step) is a **full zero-fill**
  of a 589,824-float (2.4 MB) buffer: `make_float4(0,0,0,0)` over the whole
  buffer, i.e. a materialised `zeros` for a [768, 768] accumulator that is then
  overwritten by the first reduction.

Together those are ~27 ms of the 96.8 ms CUDA training step (28%), and they are
the concrete remaining levers for that row (host-side counter, avoid the
materialised zero accumulator); the other 39% is the optimizer's per-parameter
elementwise chains, which need a batched update kernel to shrink.

State unchanged (`bench/results`): CUDA 21.9 / 14.0 / 69.3 / 96.8 ms, RUST
64.1 / 173.2 / 1163.8 / 1634.2 ms — six of eight rows within 2x of torch.

### 9.31 The requested epilogue hoist is unnecessary: the reduce is already split (2026-11)

Implemented the `split_store`-level hoist this round: the ParamArg counter is now
threaded into `split_kernels` (`CtxVal::Cnt`, seeded from the graph's max slot)
and `split_store` can rewrite `END(STORE(dest, f(value)))` into a reduce kernel
plus the store. It is **inert**, and the instrumentation explains why:

```
Store u719141
  value = Mul(Reciprocal(Index(Add(Range, Mul(Const, Range)), After(...))), ...)
  After u719110
    End u719109
      Store u719108
        Reduce u719107      <- the contraction is already its own kernel
        Index u712508
```

The rangeify realize path already splits the FFN-backward contraction: the
reduce is a kernel root inside its own `AFTER`, and the epilogue is a separate
store. So the "epilogue hoist" that the last rounds targeted does not exist as
work — the blob at ~310 ms/step is the *reduce kernel itself*, and it is
generated rather than a host GEMM.

The remaining `host_gemm` rejections were then enumerated: `A operand is Exp2`
(12/step) and `A operand is Mul` (24/step). Instrumenting the reduce operands
behind those rejects shows they are the **softmax denominator / layernorm
variance sums** (`out=[1,12,197]` with `a=Exp2`, `b=Reduce`), not matmuls — so
they are harmless and cannot move the benchmark. Staging the operands of
already-realized contractions (which removes the 24 `Mul` rejects) measured
1.553-1.639 s, i.e. no change, and was reverted.

The open question is therefore narrow and different from the previous plan: the
~310 ms/step kernel is a *reduce kernel root* whose store value is `Reduce` yet
which `host_gemm::extract` never analyses (the `[hgd]` probes on a 605,184-element
reduce never fire). The next step is to instrument `extract`'s call site in
`split_store` to see whether that kernel's body reaches it at all, rather than
adding more rangeify staging.

State unchanged: RUST train 1.56-1.64 s (2.5-2.66x), CUDA train 94-100 ms
(2.15-2.29x); six of eight rows within 2x.

### 9.32 Per-program KPROF keys; the CPU hot kernel is a single program (2026-11)

`AUTORUST_KPROF` printed only the kernel *name*, and the name encodes the loop
structure — so distinct programs collided (two different sources rendered as
`r_384_197_8_192_4`), which made the last rounds' attribution wrong. The host
path now appends the program's content key (`CompiledProg::elf_key`, the ELF
content hash) as a fourth column:

```
KPROF  <name>  <secs>  <program-key>
```

RUST ViT training, per-program (141 distinct programs in the capture):

| program | key | calls | time |
|---|---|---:|---:|
| `r_384_197_8_192_4` | `a8522231781e4338` | 33 | **1134 ms** |
| `HostGemm` | — | 393 | 728 ms |
| `r_96_197_8_768_4` | `5eeef187ae79c426` | 36 | 485 ms |
| `E_589824_4` | `729c7ea48388bd2e` | 432 | 437 ms |
| `E_147456_4` | `51c8169efd39cf60` | 882 | 265 ms |

So the "1,222 ms kernel" is *one* program (33 launches, ~34 ms each) plus a
couple of same-named smaller ones — not a single 310 ms/step blob, and not an
aggregation artefact in the way the previous note claimed. It is a genuine
generated matmul of [197, 3072] x K=768 at ~27 GFLOP/s.

The store value of that kernel is `Add` (extents `[197, 3072]`) and the
extractor sees it (`extract` peels `Add/Mul`); it bails in the *extras*
classification because the peeled operands are per-element expressions (the
quick-GELU derivative times a dropout mask), which the `HostGemm` descriptor
cannot express. Hoisting the reduce at `split_store` time was re-implemented
with the counter plumbed through `split_kernels` and the destination-index
shape fixed (`INDEX(Param, expr)` has two srcs), but the pre-lowering body does
not expose the reduce to `find_biggest_reduce` (it sits behind an `AFTER`), so
the pass stays inert and was reverted. The working next step is to peel the
epilogue *after* `to_define_global` inside `split_store`, where the reduce is
visible (the earlier `[ext-fused] value=Add exts=[197, 3072]` probe), and to emit
the temp-buffer kernel there.

Gates: clippy, rustdoc, full suite green; the change is profiler-only, so the
bench medians are unchanged (CUDA 21.9/15.7/69.2/94.1, RUST
67.0/170.9/1148.2/1638.5).

### 9.33 Post-lowering epilogue hoist: three concrete blockers (2026-11)

Implemented the hoist where the reduce *is* visible — inside `split_store`, on the
body returned by `to_define_global` (the `[ext-fused] value=Add exts=[197,3072]`
probe), with the ParamArg counter threaded through `split_kernels`
(`CtxVal::Cnt`). It now fires (`[hoist2] fired: ret=End afters_left=0`) but the
kernel graph is not valid; three distinct blockers, in order:

1. **`buffer without device`** — the temp `BUFFER` needs the store's device in
   its `ParamArg` (fixed).
2. **`args=1` vs the kernel's slots** — re-lowering the hoisted body with a
   *fresh* `LocalAddBufferContext` loses the destination: an already-renumbered
   `PARAM` is not re-registered by `debuf` (only `BUFFER` nodes are), so the
   CALL ended up with one arg while the kernel referenced slot 3. Re-using the
   first pass's `lctx` fixes that (`args=4`), but then:
3. **`END` inside the `AFTER` is never callified.** The reduce kernel is built
   *after* `split_kernels` has run, so its `END(STORE)` stays an END (this is the
   same END-callify contract as §9.26). Wrapping it by calling
   `split_store_inner` recursively yields a CALL but still hits
   `index out of bounds: len is 1 but the index is 3` at launch — the per-kernel
   `debuf` slot numbering and the CALL's arg list do not agree for a kernel
   created outside the split walk.

All three experiments were reverted; the tree is at `7077cdd` and the benchmark
medians are unchanged (RUST train 1.56-1.59 s).

The remaining work is therefore explicitly: teach the split walk to *revisit*
`END(STORE)` nodes created late (a second `split_kernels` fixpoint, as noted in
§9.26) instead of hand-building the CALL, then the hoist can simply emit the
`AFTER(buf, [END(STORE(...))])` like `bufferize_to_store` does and let the
existing machinery callify it.

### 9.34 Per-step profile: subtract the model-build phase (2026-11)

`AUTORUST_KPROF` traces include the model-build phase (weight init materialises
`Tensor::full(low)`/`full(range)` buffers for `rand::uniform`), so raw traces
overstate the per-step cost and made earlier attribution wrong. Differencing an
`ITERS=0` run against an `ITERS=3` run and dividing by three gives the true
per-step cost. A one-off check confirms the build phase is iteration-independent
(`E_589824_4` counts 72 in both the 0- and 3-iteration inference runs).

RUST ViT-B/16 training, per step (sum 1647 ms, measured 1635 ms):

| ms/step | calls | program | key |
|---:|---:|---|---|
| **453.2** | 11 | `r_384_197_8_192_4` | `a8522231781e4338` |
| 204.6 | 131 | `HostGemm` | — |
| 159.5 | 12 | `r_96_197_8_768_4` | `5eeef187ae79c426` |
| 138.5 | 144 | `E_589824_4` (a fill) | `729c7ea48388bd2e` |
| 78.9 | 294 | `E_147456_4` (a fill) | `51c8169efd39cf60` |
| 55.9 | 12 | `r_197_197_3_4_16_4` | `c239dbc3aee70cbd` |
| 44.8 | 24 | `E_589824_4` (a fill) | `724d1260b2f9741d` |
| 40.7 / 39.2 | 12 / 12 | `r_96_197_8_192_4` (two programs) | — |
| 34.4 | 12 | `r_96_96_8_8_197` | — |

The 589,824-element "fills" are *not* the weight-init buffers (those are
iteration-independent): in training 144 + 24 of them run **per step**, and both
are stores of a constant (`Tensor::full`, `buffer=true`) whose destination is a
weight-sized buffer. Where they are created is still open — the scalar elementwise
helpers (`binop_c`/`add_c`/`mul_c`) thread a scalar CONST, so the 0.9/0.999
scale factors are not the source; the next probe should print the *tensor-graph*
producer of a `full(..., buffer=true)` between the second and third iterations.

This also corrects §9.31/§9.32: the hot program is 453 ms/step (27%), not
310-450, and the fills+copy kernels together are 262 ms/step (16%), i.e. the CPU
training step is ~40% elementwise/fill traffic, 27% one generated matmul, 12%
packed GEMM. No code change this round; the tree is at `9ce81d8` and the bench
medians are unchanged.

### 9.35 Per-step constant-fill stores: attribution attempts (2026-11) — ANALYSIS

§9.34 showed ~183 ms/step in kernels that store a constant into a weight-sized
buffer (`E_589824_4`/`E_147456_4`). Four probes failed to name the producer:

1. **`creation::full` instrumentation** (value + numel + `buffer` flag, env
   `AUTORUST_FULL_DEBUG`): differencing `ITERS=0` against `ITERS=3` gives **zero
   per-step `Tensor::full` calls**. The build creates them once (`Float(0.0)`
   589824 ×99 — the Adam moments and similar; `Float(-1.0)`/`Float(2.0)` from
   `rand::uniform`'s `range`/`low`), and they are not re-created per step.
2. **Backtrace on the `-1.0` full**: `rand::uniform` → `scaled_uniform` →
   `TransformerBlock::new`, i.e. weight init — confirms the build-phase origin.
3. **Scalar elementwise helpers**: `binop_c`/`add_c`/`mul_c` thread a scalar
   CONST (`ufix`), so the optimizer's 0.9/0.999 scale factors cannot materialise
   a full tensor.
4. **`zeros_param`**: uses `to_param_ctx`, which realises the init once and
   re-creates the leaf from bytes with `register_initial` (no in-graph STORE), so
   the moments are not the per-step fills.

So the per-step fills are *stores of an expanded constant that no
`Tensor::full` call produces* — i.e. an op whose operand is a scalar const
(`where(cond, x, c)`, `zeros_like`, `ones_like`, a padded/valid gate) being
materialised as its own kernel. The remaining probe is to print the *tensor-graph
node* that is stored into a weight-sized buffer between step 1 and step 2
(e.g. tag the scheduler with a "step ≥ 1" flag and print the store's value tree
from `create_schedule`), or to name the fill kernels' destinations by giving
`register_initial` params a debug name.

No code change: the tree is at `ac8d422` (plus this note) and the benchmark
medians are unchanged. CPU ViT train remains 1.56-1.64 s (2.66x) and CUDA ViT
train 94-100 ms (2.15x); both are decomposed but not reduced.

### 9.36 Split fixpoint + hoist: the CALL argument list is the last blocker (2026-11)

Implemented the `split_kernels` fixpoint this round: `split_store` now skips a
body whose store is already a `CALL` (so revisiting is safe), the ParamArg
counter is threaded through `CtxVal::Cnt`, and the split runs up to four passes,
stopping when the root stops changing. With the fixpoint in place the
post-`to_define_global` epilogue hoist can finally emit
`AFTER(buf, [END(STORE(...))])` and let a later pass callify it instead of
hand-building the CALL.

Result: the target program is now produced correctly — the hot kernel is split
and the reduce becomes its own kernel — but at launch the *outer* CALL carries
**one argument** while its `KernelInfo` declares `globals=[0, 3, 4]`:

```
[arg-mismatch] r_384_197_8_192_4 globals=[0, 3, 4] args=1 resolved=[Bitcast]
```

`pm_no_indexing_calls` turns a CALL argument that was an `INDEX` into its buffer,
which here ends up as a `Bitcast`, and the argument list built from
`LocalAddBufferContext.map_vals` for the re-lowered body does not line up with the
globals the KernelInfo records. That is the same class of mismatch as §9.35's
report (parameter slots vs. arg list) and is *not* solved by the fixpoint.

All changes were reverted (tree at `864e1ec`); the benchmark medians are
unchanged and the full suite passes. The remaining work is to make
`program_info_from_sink` and the CALL argument list share one source of truth for
a kernel built after the split walk — i.e. derive `globals` from the `lctx` map
order rather than from the body's `ParamArg` slots — which is a contained change
in `realize.rs`/`split_store_inner`, not another rangeify pass.

### 9.37 Root cause of the hoisted-kernel arg mismatch: inherited PARAM slots (2026-11)

§9.36 left `args=1` vs `globals=[0, 3, 4]`. The cause is now identified: the
hoisted contraction kernel is built from a body that `to_define_global` already
lowered under the *outer* kernel's `LocalAddBufferContext`, so its operands are
renumbered `PARAM` nodes. A fresh `debuf` on that body only registers `BUFFER`
nodes (the new temp), which is why the CALL had one argument while
`program_info_from_sink` — which reads the `ParamArg` slots straight from the
body — reported three.

Two candidate fixes were tried:

- **slot-keyed argument list** in `split_store_inner` (build `args[slot]` from
  the body's PARAM nodes so both sides agree). This compiles and matches the
  globals, but the slots then refer to buffers of the *outer* kernel's mapping,
  and the run dies inside `rayon` (bad buffer pointer). Reverted.
- the fixpoint + hoist without it still produces the one-arg CALL (§9.36).

The correct form is the one tinygrad gets for free: a kernel body must be
lowered with a **fresh** context while its buffers are still `BUFFER` nodes, so
slots are assigned 0..n-1 in that body's first-touch order. That means the hoist
must emit the inner kernel from a *pre-lowering* body — the reduce is only
visible after `to_define_global`, so the pass would have to (a) find the reduce
post-lowering, then (b) rebuild an equivalent pre-lowering body for it (the
reduce's own `STAGE` from `create_bufferize_and_index_srcs`), which is exactly
the rangeify realize path that cannot see this reduce (§9.33).

That closes the loop: the CPU hot kernel cannot be split without teaching
`pm_generate_realize_map` to realize a reduce that is only reachable inside an
already-materialized operand, or without per-kernel param renumbering. Both are
scheduler reworks, not one-line changes. Tree at `6c2cbda`; suite green;
bench medians unchanged (RUST train 1.56-1.64 s, CUDA train 94-100 ms).

### 9.38 Per-kernel renumbering: consistent slots, missing dependencies (2026-11)

Implemented the per-kernel parameter renumbering §9.37 asked for:
`kernel_body_and_args` walks a lowered body, assigns 0..n-1 to its global
`PARAM`s **in toposort order** (the same order `program_info_from_sink` uses for
`globals`), rebuilds the body with renumbered copies, and returns the matching
argument list. With the split fixpoint and the post-lowering hoist re-enabled,
the CALL/globals mismatch disappears — but the process **segfaults**:

```
timeout 300 env DEFAULT_DEVICE=RUST ... autorust-bench
exit=139 (Segmentation fault)
```

The renumbered args are the plain buffers, so the inner kernel loses the `AFTER`
dependency arguments that the first-touch `lctx` map supplies. It therefore runs
before (or to the wrong extent of) the kernels that fill its inputs, and the
generated code writes out of bounds. Two variants were tried (DFS first-touch
order, then toposort order with a slot-indexed argument list); both crash the
same way, and both were reverted.

The remaining piece is therefore explicit: the hoist must build the inner CALL's
arguments from the **outer** `LocalAddBufferContext` values (the `AFTER`s that
carry the dependency), keyed by the buffer each renumbered parameter names —
`hoist_lowered_epilogue` needs the outer `lctx` passed in. Combined with the
renumbering that gives both a consistent `globals` and the dependency ordering;
that is the next change, and it is a signature/plumbing change rather than a new
pass.

Tree at `5e298ab`; suite green (78/78); bench medians unchanged (RUST train
1.53-1.64 s, CUDA train 94-100 ms).

### 9.39 Hoisted kernel: slot/argument invariants do not survive a resumed lowering (2026-11)

Implemented the §9.38 plan: `hoist_lowered_epilogue` takes the outer
`LocalAddBufferContext`, clones it for the inner kernel (same slot numbering, so
the dependency `AFTER`s come along as CALL arguments), lowers the reduce body
with that clone and builds the inner CALL from its map values. Three variants
were measured, each failing in the runtime rather than the scheduler:

1. cloned map as-is → `index out of bounds: the len is 6 but the index is 7`;
2. clone with `dg = map.len()` (so `debuf` cannot leave a slot past the list) →
   same;
3. additionally fixing `debuf` to assign `slot = map index` (instead of a
   per-encounter counter, which leaves holes when a buffer is debuf'd twice —
   impossible in tinygrad, possible in a resumed lowering) → same.

The remaining mismatch is not in `globals` (the executor's `[arg-mismatch]`
probe does not fire: every `pi.globals` entry is covered) but between the
**signature** recorded by `to_elf` (which scans the linearized body) and
`globals` (which `program_info_from_sink` derives from the SINK): the signature
contains slot 7 while `bufs` has 6 entries, so a PARAM in the linearized body is
missing from the SINK toposort — a direct consequence of the hoist's unusual
shape (an `AFTER` wrapping a `CALL` *inside* another kernel's value).

That is the point where this approach stops being a local change: either
`program_info_from_sink` must include PARAMs reachable through nested `AFTER`s
(in which case `bufs` grows), or the hoist must not nest a kernel inside a value
structure the info pass does not fully traverse. Both were left undone rather
than shipped broken; the tree is at `4aaeb7c` and the benchmark medians are
unchanged (RUST train 1.53-1.64 s, CUDA train 94-100 ms).

### 9.40 Session close: what is measured, what is not (2026-11)

Final state of the benchmark work. Gates green (`cargo test --release` 78/78,
clippy, rustdoc, `--no-default-features`, `cargo publish --dry-run`), tree clean.

Measured and delivered:

- **Scheduling/codegen**: warm cold starts 505/697/1279/6013 ms ->
  **216/474/760/2498 ms** (persistent rendered kernels keyed by AST content +
  codegen settings + a binary fingerprint; per-stage timings; contraction
  split; reduce-body prologue staging; interior-contraction splits; big-operand
  staging for attention).
- **Six of eight benchmark rows within 2x of torch** (no `torch.compile`):
  CUDA 21.9 / 14.0 / 69.3 ms (1.67x / 1.61x / 1.49x), CPU MNIST 65.8 ms
  (0.37x, faster), CPU ViT infer 179.2 / 1149.1 ms (1.09x / 1.64x).

Not delivered:

- **CPU ViT train 1625.0 ms vs 616.3 (2.64x)** and **CUDA ViT train 94.9 ms vs
  43.8 (2.17x)**. Per-step decomposition (build phase subtracted): CPU = 453 ms
  one generated matmul, 205 ms packed GEMM, 183 ms constant-fill stores, 79 ms
  copies, ~700 ms other generated kernels; CUDA = 60% elementwise, 40% matmul,
  39% of the time in sub-50 us launches.
- The epilogue hoist that would move the 453 ms matmul onto the packed GEMM is
  blocked at the last link: a kernel built after `split_kernels` nests a CALL
  inside another kernel's value, and `program_info_from_sink`'s `globals`
  (6 entries) disagree with `to_elf`'s signature (slot 7). See §9.36-§9.39 for
  the four links already solved (fixpoint, per-kernel renumbering, dependency
  AFTERs, slot/arg invariants).
- The ~183 ms/step constant-fill stores are *not* produced by `Tensor::full`
  (§9.35) and remain unnamed.

So the objective's first clause is met and evidenced; the second is met for six
of eight rows, and for the two training rows the work is decomposed but not
reduced. "Whatever is actually achievable" cannot be claimed for those rows.

### 9.41 Fill stores: named, partly removed, and not on the critical path (2026-11)

Used a step-tagged probe (`AUTORUST_STEP_TAG`: a realize counter in
`engine/prof.rs` plus a print of constant-store kernels in `split_store`) to
name the producers of the constant-fill kernels.

Findings:

- The stores appear only inside **`assign` realizes** (root op `After`), i.e. the
  optimizer step, and never in inference — 2 per assign realize.
- Their values identify the producer: `Float(0.0)`/`Float(1.0)`/`Float(2.0)`/
  `Float(-1.0)` on weight-sized buffers. The `2.0`/`-1.0` pair is exactly
  `rand::uniform`'s `full(high-low)`/`full(low)` for the ViT `scaled_uniform`
  (low = -1, range = 2), and the zeros/ones are the `Tensor::zeros/ones`
  parameter initialisers (`LayerNorm.weight/bias`, `Linear.bias`).
- Two contained fixes were implemented and measured:
  - `rand::uniform` builds its range/low as **scalars** instead of
    `Tensor::full(shape, ...)` buffers (numerically identical): removes the
    `2.0`/`-1.0` stores (72 + 72 unique kernels).
  - `Tensor::zeros_param` builds its leaf as `empty` + `register_initial(zero
    bytes)` instead of `creation::zeros(...).to_param_ctx()` — this is what its
    own doc comment already claimed ("no in-graph STORE"); it removes the moment
    init stores.

  Together they cut *unique* fill kernels from 530 to 130 per run (75%).
- **Neither changed the wall clock** (RUST train 1.58-1.65 s, unchanged), and the
  per-step KPROF diff is unchanged too (`E_589824_4` 216 launches /
  ~228 ms, `E_147456_4` 441 / ~128 ms). The remaining ~130 unique stores are
  *re-executed ~5x per step*, so the executed-fill count is not the number of
  stores created: it is dominated by the parameter initialisers still being
  reachable from the step's realizes.

The `zeros_param` change is kept (it restores the documented "no in-graph
STORE" contract and is numerically identical); the `uniform` change was reverted
to stay faithful to tinygrad's `rand.py:158` (`Tensor.full`). Next step on this
thread: find why the remaining parameter-initialiser stores are reachable from
an `assign` realize (the assigned value's graph must still reference the
param's pre-`to_param` chain) and cut that link; that is where the ~350 ms/step
of fill traffic actually lives.

Gates: 78/78, clippy, rustdoc.

### 9.42 The fill traffic is duplicated inside the captured step (2026-11)

Executed-fill accounting (a realize counter in `engine/prof.rs` + a per-launch
print in `exec_kernel` for the two fill programs, env `AUTORUST_STEP_TAG`):

```
[fill-exec] total 2190 executions over 221 realizes
  by realize root: {After: 438, <n=608>: 1752}
```

- **1,752 of the 2,190 fill launches happen inside a single captured step** (one
  `Jit` replay, which does not advance the realize counter — hence the `?`/608
  attribution). That is ~44% of the step's ~5,000 kernel launches, matching the
  ~350 ms/step derived from the ITERS-diff.
- They come from only **~130 distinct fill kernels** (§9.41), i.e. each init
  store is emitted **~13-17 times** into the captured `LINEAR`. So the cost is
  *duplication in the schedule*, not store creation: the same
  `STORE(const)` kernel appears once per reachable path (every optimizer
  `assign` co-realized into the step's single LINEAR), and `create_schedule`'s
  identity dedup does not collapse them because each path carries its own copy.

The next fix is therefore in the scheduler: dedup kernel bodies by *content*
(the store's AST key), not by node identity, when building the LINEAR — or stop
the parameter-initialiser store from being reachable from an `assign`'s value at
all. Both are now precisely scoped; the accounting probe above is what proves
which one is needed.

State unchanged (`b9a7567` tree): RUST train 1.58 s, CUDA train 94-100 ms, six of
eight rows within 2x.

### 9.43 Content-keyed LINEAR dedup: hypothesis falsified (2026-11)

Implemented the §9.42 dedup: while building the LINEAR, drop an earlier store
kernel when a *structurally identical* one appears later (`UOpCtx.g.content_key`)
and nothing else wrote that destination in between (write-aware, so a
read-between-writes sequence cannot change meaning). It drops **zero** kernels —
for the 705-kernel step LINEAR as well:

```
[dedup] linear=705 dropped=0
```

So the ~657 fill launches per step are **not** repeated copies of identical
kernels: they are distinct store kernels (same loop structure, hence the same
`E_589824_4` / `E_147456_4` name, but different destinations/parameter slots).
The duplication is therefore *per-path buffer renumbering*: each co-realized
path gives the same logical buffer a different `ParamArg`, so the content keys
differ and identity/content dedup cannot see them.

Counting also corrects the "2 per weight" reading: 216 launches of the
2.36M-element fill over 12 `ff1` weights is ~18 per weight, and 441 launches of
the 589k-element fill over 48 attention weights is ~9 per weight. Those are the
two `rand::uniform` init buffers (`full(high-low)`, `full(low)`) of
`scaled_uniform`, re-executed per path.

Reverted (inert). The next attempt on this thread must key on the *logical*
buffer (the pre-renumbering base) rather than the renumbered `ParamArg`, or
remove the initialiser stores from the step graph before scheduling. State
unchanged: RUST train 1.56-1.65 s, CUDA train 94-100 ms, six of eight rows
within 2x.

### 9.44 "Fill traffic" was a kernel-name collision: corrected accounting (2026-11)

Dumped every distinct source block sharing the two hot names in a ViT training
run:

```
E_589824_4   82 blocks, 7 distinct signatures
E_147456_4  157 blocks, 10 distinct signatures
```

The signatures are *not* just fills:

| signature | what it is |
|---|---|
| `= [±1.0, 2.0, 0.0, 1.0]` | the `rand::uniform` / `Tensor::zeros/ones` initialiser fills |
| `= val0` | a plain contiguous copy |
| `= (val0*0.9 + val1*0.1)` | the Adam first-moment update |
| `= (val0*0.999 + val1*val1*0.001)` | the second-moment update |
| `= val0 + val1*(1/(…sqrt…))` | the parameter update (LAMB/adam path) |
| `= ((val0*val1)+val2)*0.00065` | a scaled GEMM epilogue (a *matmul*, not a fill) |

So the earlier "~183/350 ms/step of fill traffic" was an artefact of keying on
the kernel *name*, which only encodes the loop structure: the 216 + 441 launches
per step include the **necessary optimizer arithmetic**, **copies** and even a
scaled matmul. The true initialiser-fill share is much smaller than assumed, and
the `b9a7567` result (400 fewer unique fill kernels, no wall-clock change) is
consistent with that.

Corrected per-step picture for RUST ViT training (sum 1608 ms, measured 1635):

| item | ms/step |
|---|---:|
| the generated matmul `r_384_197_8_192_4` (hoist target) | 423 |
| packed `HostGemm` | 207 |
| `E_589824_4` (fills + copies + Adam updates, 82 blocks) | 234 |
| `r_96_197_8_768_4` (generated matmul) | 173 |
| `E_147456_4` (fills + copies + updates, 157 blocks) | 136 |
| other generated kernels | ~440 |

The single lever large enough to reach 2x is the hoist: moving the 423 ms matmul
(plus the 173 ms sibling) onto the packed GEMM would save ~400-500 ms, taking the
step to ~1.2 s = 1.95-2.0x. Everything else (optimizer arithmetic ~380 params x
3 kernels, copies, attention fragments) is inherent work at this model size.
The hoist is blocked only by the `program_info_from_sink` vs `to_elf`
signature/globals disagreement for a kernel created after the split walk
(§9.36-§9.39).

This supersedes §9.41-§9.43: there is no large "phantom fill" to remove.

### 9.45 Why the 423 ms/step matmul cannot be tiled faster on CPU (2026-11)

The kernel (`r_384_197_8_192_4`, key `a8522231781e4338`) is a fused
`gelu'(ff1) * Reduce(mask*dY, W2)`. Its dumped source has exactly three loops:

```
for __gidx0 in gidx0_start..gidx0_end      // CPU "grid": rayon chunks
  for Lidx2 in 0..384                      // 3072 columns / 8, serial
    for Ridx0 in 0..192                    // K=768 / 4, 4-wide upcast
```

`AUTORUST_MM_DEBUG` shows the tiler does recognise it, with
`m=Some(3072) n=Some(197)` — the FFN axis is M and the *token* axis is N. The
CPU tiling branch upcasts both axes, but N = **197 is prime**, so its upcast
falls back to 1 (the port deliberately refuses masked/padded upcasts, since
vector stores are not lane-wise). The token axis therefore cannot be blocked in
registers, so every column block re-reads the whole masked activation:

```
A = [197, 768] = 0.61 MB;  384 column blocks -> 232 MB/layer -> 2.79 GB/step
at 10-20 GB/s that is 279-139 ms/step of pure A traffic (measured 423 ms)
```

The packed CPU GEMM does not have this problem because it *packs* B into panels
and micro-kernels a 6x16 register tile, so both operands are read once; that is
why it sustains 290-460 GFLOP/s on the same shapes while this kernel runs at
~27 GFLOP/s.

So the gap is **structural for the generated path on a prime token count**: no
tiling change can block a 197-row axis without masked/padded vector ops. Closing
it requires routing the contraction to the packed GEMM — i.e. materialising the
masked operand and splitting the epilogue (the hoist, §9.36-§9.39).

This is the first *demonstrated* limit for the CPU training row: the 423 ms is
explained and bounded by A-operand traffic under the current codegen, and the
only way to go below it is the hoist (or an im2col-style A staging plus the host
GEMM). State otherwise unchanged.

### 9.46 A realized reduce keeps its decision across the prologue rebuild (2026-11)

`stage_reduce_prologue` rebuilds the `REDUCE` after staging its operand
(`x` -> `nx`) and copied only the `range_map` entry. The `realize_map` decision
and the `non_removable` mark were therefore left on a dead node id, and
`create_bufferize_and_index_srcs` computed `removable = true` for the new reduce
— so `pm_remove_bufferize` was free to inline the boundary and re-fuse the
contraction. Traced with `AUTORUST_STAGE_TRACE`:

```
[stage-make] reduce ... removable=true  axes=3 out=[1,197,768] nrem=0   (before)
[stage-make] reduce ... removable=false axes=3 ...             nrem=174 (after)
```

The fix carries both maps across the rebuild. It is a correctness fix (a node
rebuild must preserve the realization decision), and it turns 174 reduce STAGEs
into kept boundaries — but it does **not** materialize the hot kernel: 270 reduce
STAGEs are created per capture and `r_384_197_8_192_4` (429 ms/step) still runs
fused, with `HostGemm` unchanged at 131 calls. So the hot reduce is one whose
STAGE is created by a path that never receives the mark at all (not one whose
mark is lost), which is consistent with §9.36-§9.39: the contraction is only
visible after lowering.

Gates: 78/78, clippy, rustdoc. Medians unchanged (RUST train 1.59-1.63 s,
MNIST 64.3 ms, ViT infer 181.7 ms).

### 9.47 ProgramInfo now derived from the linear body (2026-11)

`prepare_program` computed `ProgramInfo` from the SINK (`program_info_from_sink`)
and the ELF signature from the *linearized* body
(`signature_from_linear(ctx, &lst)`). `pm_linearize_cleanups` can rewrite or
introduce nodes, so the two traversals could disagree — exactly the
`len is 6 but the index is 7` crash seen for a kernel created after
`split_kernels`. `program_info_from_list` now re-derives `vars/globals/outs/ins`
from `lst` (sorted and deduped, so ordering is unchanged), giving both sides one
source of truth.

This is landed and neutral for the current code paths (the cleanups do not
rewrite PARAMs for kernels built during the split): 78/78 tests, clippy, rustdoc
clean; RUST train 1.56-1.64 s, CUDA train 94.5 ms, medians unchanged.

It does **not** by itself fix the hoisted kernel: re-enabling the split fixpoint
plus the post-`to_define_global` hoist still fails with `len is 6 but the index is
7` at launch, so the disagreement survives this change — the linear list for that
kernel must be produced by a path where the signature counts a slot the info
traversal does not see (or vice versa), and the next probe should print both slot
sets for the failing program rather than relying on the executor's index panic.

### 9.48 The hoist's launch failure is not a signature/globals mismatch (2026-11)

Ran the requested side-by-side diagnostic (env `AUTORUST_SLOT_DEBUG`: print
`name`, ELF signature slots, `pi.globals` and the CALL arg count for every
launch) with the split fixpoint and the post-`to_define_global` hoist enabled.

Result: **7,049 launches logged, every one with `sig == globals` and
`args > max(slot)`** — including the multi-slot kernels (`E_3072_197`:
`sig=[0,1,2,5] globals=[0,1,2,5] args=6`). The very next launch then panics with
`index out of bounds: the len is 6 but the index is 7` inside
`runtime::ops_rust::call`, and it produces **no** `[slot]` line, so the crashing
program never reaches `exec_kernel`'s diagnostic.

That rules out the `program_info_from_sink` vs `to_elf` theory (§9.36-§9.39,
§9.47): the info and signature agree for every kernel that goes through
`exec_kernel`. The crash must come from a launch whose `CompiledProg`/runtime
pair is inconsistent — the only remaining candidate is the runtime-keyed program
cache (`dev.runtime_keyed(cp.elf_key, &cp.elf)`): a program built for one ELF
reused under a colliding/reused key would carry a different signature than
`cp.elf` (whose slots the diagnostic checks).

Next probe: print inside `runtime_keyed` whether it returned a cached runtime and
compare the cached runtime's signature with the requested ELF's. Everything else
was reverted; the tree is at `97544f3` and the medians are unchanged (RUST train
1.54-1.64 s, CUDA train 94.5 ms).

### 9.49 Runtime cache identity includes the buffer signature (2026-11)

The §9.48 probe (`AUTORUST_SLOT_DEBUG`) finally caught the crash site. For the
program that panics:

```
[elf] name=E_3072_197 lib_len=6288 key=3048ba7f24414863 sig=[0,1,2,7] globals=[0,1,2,7]
[rt-mismatch] bad_slot=7 sig=[0,1,2,7] bufs=6
```

`pi.globals` and the ELF signature **agree** (`[0,1,2,7]`), so the info/signature
traversal was never the problem: `bufs` had six entries because the *runtime*
returned by `dev.runtime_keyed(elf_key, &elf)` was built from a different
artifact than `cp.elf` — a cached runtime whose signature is `[0,1,2,7]` reused
for a program whose slots end at 5. `runtime_keyed` keys only on
`elf_key` (FNV over the compiled bytes), so any key reuse hands back a runtime
with the wrong signature.

Fix: fold the buffer signature into the runtime identity (`slot`, `dtype.0` per
entry), so a cached runtime is only ever reused for an artifact with the same
signature. With the fix the previously crashing hoist runs to completion.

It still does **not** pay off: with the split fixpoint + post-`to_define_global`
hoist enabled, RUST train is 1743-1763 ms (worse than 1560-1642) and
`r_384_197_8_192_4` is 531 ms/step with `HostGemm` unchanged at 131 calls — the
hoist adds an epilogue kernel without turning that contraction into a packed
GEMM, so it was reverted (the cache fix is kept).

Gates: 78/78, clippy, rustdoc. Medians: RUST train 1.57-1.64 s, CUDA train
94.5 ms, six of eight rows within 2x.

### 9.50 rustc's loop vectorizer is exponential on integer shift chains (2026-11)

Batch-2 CPU training hung for over ten minutes in `rustc`, not in a kernel: the
disk cache was cold for the new shape, and one generated kernel
(`E_2`, the `threefry` hash behind `rand`/dropout, 166 lines) never finished
compiling. Reproduced standalone on rustc 1.97.1 with the real flags
(`-C opt-level=3 -C target-cpu=native`); `-C opt-level=0/1`, `-C
target-cpu=x86-64` (no SSE4.1) and `-C llvm-args=-vectorize-loops=false` all
compile it in milliseconds, so the loop vectorizer is the culprit. A synthetic
loop shows the blowup is in the length of the **integer shift/xor chain**, not
the body in general (f32 chains of 84 ops: 0.07 s): 20 shifts 0.26 s, 22 0.65 s,
24 1.8 s, 26 5.2 s, 28 16 s, 30 > 30 s; the RNG kernel has 80+.

Fix: `render_rust` routes the `gidx0` loop index through `core::hint::black_box`
when the body has at least 24 shifts. An opaque induction variable makes the
vectorizer bail on the outer work-item loop, which costs nothing at runtime
(the body is already vectorized; these kernels are tiny) and leaves every other
kernel's loop vectorizable. `tests/rust_render.rs::shift_heavy_kernels_mark_the_loop_index`
pins both sides.

With the fix, RUST ViT-B/16 training runs at batch 2/4 (3.15/4.21 s) instead of
hanging. The same shapes are fine on CUDA (`CC` uses `restrict`, and CUDA has no
work-item loop).

### 9.51 Backend memory reporting (2026-11)

`Compiled` now reports device capacity through `MemoryInfo { total, free }`.
The query threads through the existing allocator seam, so it works for every
registered backend without constructing a parallel device-properties API:
`Compiled::memory_info()` → `LruAllocator::memory_info()` →
`BackendAllocator::memory_info()` (default `None`).

- **CUDA** — `cuMemGetInfo_v2` after binding the context; reports both total and
  free bytes.
- **OpenCL** — `CL_DEVICE_GLOBAL_MEM_SIZE`; reports total, with `free: None`
  because OpenCL has no portable free-memory query (vendor extensions were
  deliberately avoided).
- **Metal** — `recommendedMaxWorkingSetSize` for the budget and
  `currentAllocatedSize` for the process's current use; `free` is the
  difference.
- **Host/interp** — `None`; there is no per-backend limit to report.

`free` is a snapshot, not a reservation: other processes and the driver still
change it. Callers should apply a safety margin, and remember that `LruAllocator`
cached blocks remain allocated even though they are reusable.

Gates: `tests/memory_info.rs` checks the host default is `None` and, when the
backend is present, CUDA returns `free <= total`, OpenCL returns a positive
total with `free: None`, and Metal returns a positive working set. Clippy and
rustdoc pass.

ClusterFun uses this to replace its fixed GPU batch size with a memory-bounded
estimate (rounded down to a multiple of 8) instead of assuming every GPU is the
12 GB reference card.



### 9.51 The benchmark now covers the pieces the two models hide (2026-11)

The matrix only had MNIST (conv/linear) and ViT (attention), so a gap could be
a GEMM tile, a conv lowering, a launch count or the front end and the table
could not tell them apart. Eight workloads were added, each with the same
protocol as the existing rows (`bench/run.py`, raw JSON in `bench/results/`):

- **GEMM sweep** (`gemm`, `gemm_bf16`): 1024³/2048³/4096³, two rectangular
  shapes, transposed-A, transposed-B and batched 8×512³, f32 and bf16, one
  element read back to force completion. This is the control for every model
  row: the f32 GEMM is at parity with cuBLAS on CUDA (7.8-9.4 TFLOP/s at 2048³+
  vs torch's 7.4-8.5) and the bf16 path is 0.4-0.9x torch (8-15 vs 13-25 TFLOP/s) with
  tensor cores selected. On CPU the packed GEMM is 0.4-0.6 TFLOP/s vs MKL's
  0.6-1.0; transposed-A is the weak case (0.3 vs 0.7).
- **Memory-bound ops** (`memops`): add/copy/gelu/softmax/layernorm/sum/amax/
  transpose over 16M f32 elements, reported in GB/s. Pure elementwise matches
  torch on CUDA (325 vs 322 GB/s) and is 2x faster on CPU; reductions are the
  gap (softmax 155 vs 288, amax 64 vs 292 GB/s) because they are two-pass.
- **AdamW-only step** (`optim`): 100M parameters in 32 tensors with grads
  attached once. CPU is faster than torch (232 vs 296 ms); CUDA is 4x slower
  (102 vs 26 ms) because the update stays per-tensor elementwise launches
  (27 GB/s vs torch's 106).
- **Conv models**: ResNet-18 (infer + train), ResNet-50, MobileNetV2 and a
  small UNet. Dense conv is the weak path: ResNet-18 inference is 4.8x off
  torch on CPU and 21.9x on CUDA (the direct-conv kernel runs at ~9-30 GFLOP/s
  against 8 TFLOP/s for the GEMM). ResNet-18 *inference* is slower than its own
  training step on CPU although training does more work, which points at the
  BatchNorm-fused inference conv selecting a slower lowering. MobileNetV2 is
  the exception (depthwise has no GEMM lowering; the elementwise path beats
  oneDNN on CPU, 3.4x). The UNet exposes conv_transpose/upsample/cat (33.9x off
  torch on CPU).
- **Autoregressive decoder** (`llm_prefill`, `llm_decode`): 6-layer LLaMA-style
  model (12/4 GQA, RoPE, causal mask, 512-token KV cache, Gumbel-max sampling).
  Prefill is 2.2x/19.4x off torch (CPU/CUDA). Decode launches ~300 kernels per
  token and is 8.1x/4.7x off; the gap is per-kernel dispatch, not FLOPs (torch
  runs the same explicit matmul+softmax, not SDPA).
- **bf16 ViT** (`vit_infer_bf16`, `vit_train_bf16`): weights and input cast to
  bf16, exercising the TC matmul and bf16 attention. Inference is 61.6 vs
  torch's 18.2 ms at bs 8 and training 84.3 vs 20.9 ms at bs 1: only 1.14-1.15x
  faster than f32 because the 197-token axis is prime and the TC tiles pad it,
  while torch's bf16 halves its f32 cost (so the bf16 gap, 3.4-4.0x, is wider
  than the f32 one).
- **ONNX runner** (`onnx_mnist`, `onnx_resnet`): the MNIST CNN and a
  width-0.25 ResNet-18 exported to `bench/assets/*.onnx` (regenerate with
  `bench/tools/export_onnx.py`) and run through `OnnxModel::run`. 14.7/204.8 ms
  per call on CPU vs torch's 3.7/17.4: the runner rebuilds and schedules the
  graph per call, and the cost scales with the graph (54x behind on the ResNet
  CUDA row).
- **Frozen plan** (`freeze`): the `vit_freeze` example now also times eager JIT
  replay vs `FrozenRunner` (1.60 vs 2.00 ms for the small ViT, 17 ms plan
  build). Per-step parity is expected (same kernels); the frozen path's win is
  deployment (no scheduler/rustc), and the runner's input copies are why it is
  not faster.

The bf16 sweep immediately found a renderer bug: a bf16 kernel whose lanes
vectorize 2-wide emits `make_nv_bfloat162`, which `cuda_bf16.h` does not
provide (it only typedefs `nv_bfloat162`); tinygrad has the same latent bug
because it also only emits vector prefixes for 4/8-wide. `render_kernel` now
defines the 2-wide constructor in terms of `__halves2bfloat162`. The bf16 GEMM
rows are CUDA-only: torch's CPU bf16 GEMM is unusably slow here, candle's CPU
matmul is f32-only and tinygrad 0.14 rejects bf16 on this CUDA path.

Rows live in `bench/run.py`; `bench/report.py` renders the microbenches as
per-case tables (their JSON carries a `cases` array). tinygrad/candle drivers
only cover the original workloads plus GEMM, so the other new rows are
autorust+torch; noted in `bench/README.md`.

### 9.52 A Vulkan backend that emits SPIR-V directly (2026-11)

There is no tinygrad Vulkan target, so this backend has no oracle: it reuses the
existing seams (`OpaqueT`/`BackendAllocator`/`Comp`/`Prog`/`UopRenderer` +
`register_device`) and the C-style renderer's linear-UOp walk, re-encoded as a
SPIR-V module (`renderer/spirv.rs`). `ash` supplies the bindings only: its
`loaded` feature `dlopen`s `libvulkan`, so nothing links against the loader and
`--no-default-features` stays std-only+`nanoserde`.

Emission details that are easy to get wrong, each found by a real failure:

- **No pointer casts in SPIR-V.** A `float4` view of a buffer becomes a second
  descriptor binding (`RuntimeArray<vec4<float>>` over the same `VkBuffer`), so
  a buffer *slot* may appear several times in the binding table. The table is
  part of the rendered payload (`vk1 <n> <slots...> <local size>` + base64
  words) because `TinyElf.signature` is generated generically from the PARAM
  list and cannot express duplicates.
- **Narrow integers are 32-bit in registers.** Vulkan rejects `OpConstant` of
  8/16-bit integer types and restricts 8/16-bit arithmetic, so loads
  zero/sign-extend, stores truncate, and narrowing casts wrap to the dtype and
  widen again. `bool` buffers are byte arrays (`OpTypeBool` has no storage
  size), converted at the boundary. `bf16` is a 16-bit pattern widened to f32 by
  `extra::pm_manual_bf16_cast`, exactly as the OpenCL renderer does.
- **`Function` storage must live in the first block**, workgroup arrays may be
  module scope; `OpVariable`s are therefore pre-scanned and emitted in the entry
  block.
- **Structured control flow.** `RANGE`/`END` become `OpPhi` + `OpLoopMerge`
  loops with the increment in the continue block; the gated `END` is a branch to
  the merge block (`break`). `IF`/`ENDIF` are `OpSelectionMerge` regions; a
  gated `LOAD` (`gate ? access : var`) is a selection plus a `Phi` instead of a
  variable. Both were validated against `spirv-val --target-env vulkan1.1`
  (`tests/vk_spirv.rs` renders real scheduled kernels and validates them).
- **A command-buffer ring is not optional.** Re-recording one command buffer
  while the previous submission is pending is undefined and segfaults lavapipe;
  four slots with one fence each, reset only after that slot's fence signals,
  fixed it. Descriptor sets follow the same ring.
- **Deferred buffer destruction.** Vulkan forbids destroying a `VkBuffer` a
  pending submission references. MNIST training frees and reallocates
  intermediates every step; destroying eagerly lost the device on NVIDIA.
  Destruction is queued and drained at the next idle point (`wait_idle`), while
  the queue's ordering already makes *reuse* of a just-freed buffer safe.
- **64-bit index arithmetic.** The cross-entropy/getitem graphs use `INT64`
  offsets; emitting `OpIMul %int %long %int` mixed widths (accepted by
  `spirv-val`, miscompiled by llvmpipe into an out-of-bounds write, i.e. SIGFPE
  in lavapipe and a lost device on NVIDIA) is fixed by carrying the index width
  in `Off` and widening both sides.
- **Device facts matter.** The scheduler's tiling depends on
  `global_max`/`local_max`/`shared_max`; these now come from
  `VkPhysicalDeviceLimits`. Host-visible memory must also be chosen by size:
  tensors are `DEVICE_LOCAL` (VRAM) and only small allocations use the BAR
  window, with staged copies otherwise. Staging in the BAR window is much
  *slower* for readback than system memory (404 ms vs 64 ms for 16 MB).

**The reduce gap was loop unrolling, not local memory.** An isolated 64 MiB
row-reduce (`amax` over axis 1 of 4096x4096) measured 0.71 ms of device time on
VK and 0.258 ms on CL (`VK_TIME=1` timestamp queries vs OpenCL event profiling,
both added for this comparison). glslang's canonical SPIR-V for the same kernel
was equally slow (0.78 ms through a raw-module runner), so the emitted IR was
not at fault; a hand-written 4x-unrolled GLSL variant ran in 0.263 ms. The
NVIDIA SPIR-V path does not unroll that loop, while its OpenCL compiler does,
and the extra loads in flight are what make a 32-lane/16 KB-stride reduce reach
memory bandwidth. `renderer::spirv` now unrolls small constant-trip-count loops
8x (`UNROLL`: body <= 48 uops, must contain a load, extent divisible by the
factor). `softmax` went 2.14 -> 1.16 ms and `layernorm` 2.18 -> 1.18 ms against
CL's 1.41/1.44, i.e. **both are now faster than OpenCL**, and LLM decode dropped
from 34 to 25 ms.

**Register arrays must live in `OpPhi`s, and the carry must come from the
continue block.** A 64-float accumulator declared as a `Function` array is not
promoted by the driver, so the GEMM reduce and the 4-float scalar tiles in the
conv kernels round-trip through scratch (`OpAccessChain`+`OpLoad`+`OpStore` per
lane per iteration). The emitter now promotes a `BUFFER` whose every index is a
constant and whose stores are not inside an `IF` to SSA, with a loop-carried
`OpPhi`. Two pitfalls: (a) the store pointer is `Index(After(Buffer), k)`, so
both chain walks must pass through `AFTER`; (b) the loop's exit branch is taken
from the *continue* block, so the value used after the loop must be the one
carried out of the continue block — using the header phi silently drops the last
iteration (a 64-element sum of ones returned 24 and the bf16 matmul was off by
0.15). `vk_reduce_covers_the_last_chunk` and `vk_bf16_matmul_matches_interp` pin
both; ResNet-18 inference went 403 -> 321 ms and UNet 21.0 -> 18.1 ms.

**Dispatch batching needs a flush point.** A Vulkan submit + fence costs ~25 us
where `clEnqueueNDRangeKernel` is nearly free, so consecutive dispatches are now
recorded into one command buffer (8 per batch, one descriptor set per
(ring slot, batch position) because a set may not be rewritten while a batch
references it). That alone took LLM decode from 25.6 to 21.3 ms — but holding
the submit back idles the GPU while the host builds the next graph, and a
single-kernel step went from 0.91 to 1.45 ms. `Prog::flush` (a default-no-op
trait method called at the end of every executed linear in `exec_linear`) closes
the batch at the step boundary, which restores the CPU/GPU overlap; both wins
then hold.

Measured against `CL` on the same machine (`bench/run.py --only
'autorust-(vk|cl)-'`, 138-row matrix, medians): **7 of 8 memory-bound
microbenches are at parity or faster** (add 0.85 vs 1.07 ms, layernorm 1.13 vs
1.43, softmax 1.20 vs 1.38, sum 0.61 vs 0.69, copy 0.80 vs 0.85, gelu 0.81 vs
0.83; `amax` 0.69 vs 0.62 is the one behind), MNIST train 22.6 vs 21.9 ms, ViT
infer bs 8 **132.5 vs 179.9 ms**, **ResNet-18 infer 321 vs 376**, ResNet-18
train 453 vs 449, ResNet-50 102 vs 99, **UNet 18.1 vs 21.2**, LLM prefill 423 vs
397, LLM decode 22.4 vs 20.2, AdamW 95.8 vs 97.4, GEMM 1024³/NT/batched ahead
and 2048³/4096³/TN 1.07-1.38x behind. Remaining gaps: bf16 (GEMM 4.39 vs 3.16,
ViT-B/16 154 vs 92) where the graph's 15-uop round-to-bf16 idiom is compiled
literally by the driver but optimized into a few `prmt`/`lop3`/`selp`s by the
OpenCL C compiler; MobileNet/ResNet-50/ONNX (1.03-1.35x).

A bf16 tensor-core path is the concrete follow-up for the last gap: bf16 `dot`
accumulates in bf16 (the output dtype), so the graph rounds *every product*, and
that idiom costs the same on both backends — only hardware MMA accumulates in f32
for free. The device advertises `VK_KHR_cooperative_matrix` plus
`VK_NV_cooperative_matrix{,_2}` (and lavapipe advertises the KHR one), so
`OpTypeCooperativeMatrixKHR` + `OpCooperativeMatrixMulAddKHR` is available
without a new dependency; it needs a capability probe, the emitter's type/decor
support, and a WMMA-shaped tile from the scheduler (the CUDA backend's `tc` path
is the model). Not attempted yet.


### 9.53 Cooperative matrices: probed, proven, not yet wired (2026-11)

`VK_KHR_cooperative_matrix` is the answer to the "specialized syntax for matrix
multiplies" question, and it is a large win. `vk_sys::query_coop` enumerates
`VkPhysicalDeviceCooperativeMatrixPropertiesKHR` at discovery and keeps the best
`A=B=16-bit, C=f32, subgroup-scope` shape (`CoopShape` on `DeviceInfo`, also
printed by `VK_TRACE`); `VkDevice::open` enables the extension and
`cooperativeMatrix` feature only when such a shape exists, so a device without
it is unaffected. The RTX 3060 reports `M16 N16 K16` (plus bf16 inputs under
`VK_COMPONENT_TYPE_BFLOAT16_KHR`), lavapipe `M8 N8 K8`.

`tests/vk_coopmat.rs` assembles
[`tests/data/coopmat_gemm.spvasm`](tests/data/coopmat_gemm.spvasm) with
`spirv-as` and runs it through the normal device/pipeline/dispatch path, checking
numerics against an f64 host reference and asserting a throughput floor. The
fixture is a 16x64-tile GEMM (four `OpCooperativeMatrixMulAddKHR` per A load,
f16 in / f32 out, no shared-memory staging): 7.7 TF/s at 1024^3 and 9.8 TF/s at
2048^3, versus 3.0/3.9 TF/s for the current plain-SPIR-V GEMM (3.2x and 2.5x),
exact to the last bit at every size.

Emission details worth keeping (they cost a full debugging round each):
`OpTypeCooperativeMatrixKHR`'s last operand is the *MatrixUse* (A=0, B=1,
Accumulator=2) — passing K there makes the validator report "Cooperative matrix
type must be A Type"; `MemoryLayout` and `Matrix Operands` are literals while
`Stride` is an id and the memory-access operand is omitted; `Shader` +
`CooperativeMatrixKHR` requires the `VulkanMemoryModel` capability
(`OpMemoryModel Logical Vulkan`); and the accumulator must be carried out of the
*continue* block (the same off-by-one-iteration trap as §9.52).

The module is now generic in the shape: `runtime::coopmat` embeds the assembled
words with *sentinels* for M, N, K and A's row stride and patches them at load
time (`coopmat::module(m, n, k)`), so one artifact serves every shape. The last
tile in each direction is clamped to the tensor edge, so M and N need not be
multiples of 16/64 — which is what the ViT shapes need (M = 8x197 = 1576). At
those shapes it is 3.4-4.6x the generated kernel: 1576x768x768 in 0.22 vs
1.03 ms, 1576x3072x768 in 0.68 vs 2.32 ms (8.3-11.0 vs 1.9-3.2 TF/s), and the
four linears of a ViT block are ~70% of the bf16 step, so this is the change that
would put `vit_infer_bf16` ahead of OpenCL.

Wired for f16/f32, not yet for bf16. The library-GEMM seam now has a
`GemmDevice::Vk` (policy `AUTORUST_VK_GEMM`), `library_ready`/`library_available`
consult `ops_vk::coop_gemm_available()` (the probed MMA shape), `extract` builds
a descriptor for a 16-bit multiply with an f32 reduce/store, `coopmat::eligible`
filters it down to the contiguous row-major case the module supports, and the VK
device installs a `host_exec` that patches `coopmat::module` and dispatches it
(pipelines cached by shape). Two things had to change for the descriptor to
exist at all: the mixed-precision gate (`mul_dt == FLOAT16 && red_dt == c_dt ==
FLOAT32`) was Metal-only and now includes VK, and the "clean multiples of 4 stay
generated" heuristic (measured for cuBLAS) no longer applies to VK — it excluded
exactly the shapes where the module wins. `tests/vk_coopmat.rs` pins the
end-to-end behaviour: one dispatch, exact numerics, and a plain f16 `dot` *not*
taking the path. Measured at 2048^3 f16/f32 in the pipeline: 1.71 vs 3.42 ms of
device time (2.0x).

**bf16 storage works on the same path** (`coopmat_handles_bf16_storage`): the
module reads f16, so `bf16` A/B are staged through f16 scratch buffers by two
element-wise conversion modules (`coopmat::convert_bf16_to_f16`, exact for
in-range values — f16 has more mantissa than bf16, only a narrower exponent) and
a bf16 result is rounded from an f32 scratch (`convert_f32_to_bf16`). Both the
bf16-out and f32-out forms dispatch once and match INTERP. The bf16, f32-out form
also unlocked the mixed-precision gate for bf16 multiplies.

The reason the bf16 rows stayed put was a **dtype mismatch in the matcher**: a
reduced-precision GEMM lowers to a 16-bit multiply with an **f32 reduce** and a
16-bit store (`mul=bf16 red=f32 c=bf16`, "promote the multiply to the f32
accumulator"), which the gate rejected because it only allowed `mul == red == c`
or Metal's f16/f32 form. Fixed: for a Vulkan descriptor, `mul` 16-bit with an f32
reduce now qualifies for either a 32-bit or a matching 16-bit result — and since
the graph itself asks for f32 accumulation there, the numerics are faithful
(`USE_TC` only gates a graph whose reduce dtype is 16-bit).

That uncovered the real blocker: every contraction in the benchmarks presents one
**transposed** operand (`a_m == 1`, or `b_k == 1`, i.e. `ColumnMajorKHR` — the
ViT's `Linear` weight is stored `[out, in]` and used as `x @ w.T`, and the `TN`/
`NT` GEMM shapes are the same). It is solved by **staging**: `Plan` carries, per
operand, the two offset strides plus the `MemoryLayoutKHR`, and
`coopmat::stage_bf16` reads a strided bf16 operand and writes contiguous row-major
f16, so the GEMM itself stays row-major. Two bugs on the way, both found by
reading the staged buffer back:

- the staging pipeline cache was keyed by `rows*cols + cols`, which is identical
  for a row-major and a transposed operand of the same shape, so the second case
  ran the *first* case's kernel (a verbatim copy) while the trace still printed
  the right strides. The key now covers every patched constant;
- the scratch buffers were allocated with `vkAllocateMemory` on every call (32 MB
  for a 2048^3 GEMM), which cost ~1.6 ms of the 4.4 ms; they are pooled per use
  (A/B/C) and grown in place.

All four operand-layout combinations now dispatch once and match the interpreter
exactly, and the bf16 GEMM row moved for the first time (`bench`, VK vs CL):
2048^3 **2.78 vs 2.93 ms**, 4096^3 **17.9 vs 23.2**, NT 2048^3 **2.60 vs 3.49**,
1024x1024x4096 **1.86 vs 2.23**, 4096x1024x1024 **1.66 vs 1.73**, TN 2048^3 3.01
vs 2.97 (parity), 1024^3 1.25 vs 0.80 (behind — the three staging dispatches do
not amortise, so but re-measuring 1024^3 with the cutoff in place gave 1.25 ms (worse than the
0.98 ms the MMA took), i.e. the machine's +-30% noise, so there is **no size
cutoff**; the batched-8x512^3 row is a batched descriptor and stays generated).

The **bias epilogue** is in: `coopmat::epilogue_bf16` adds a per-column bf16 bias
to the f32 accumulator and rounds once to bf16 (`out[i] = bf16(acc[i] +
bias[i % n])`), and `plan` accepts a `bias_slot` when the result is bf16
(`bias_off` is honoured through the binding offset). `coopmat_handles_a_biased_linear`
pins it: a `Linear` (transposed weight *and* bias) dispatches through the MMA and
matches INTERP.

`vit_infer_bf16` is **still 155 ms vs CL's 92**, and the reason is now pinned: the
per-`return` instrumentation in `extract` (re-added and run against that workload)
shows its projection kernels bail at the operand-dtype check, 49 times, with

```text
ab-dtype a=f32 b=bf16 mul=f32 red=f32 c=f32
```

i.e. the ViT runs **f32 activations against bf16 weights with f32 accumulation** —
not a 16-bit contraction at all. A cooperative matrix has 16-bit (or f32) component
types and nothing in between, so this contraction cannot use the MMA without
rounding the activations to f16 (11 mantissa bits against f32's 24) — a real
precision loss, not the exact bf16 storage conversion the staging pass does. So the
ViT gap is **not** a missing tensor-core path; it is the quality of the generated
f32 x bf16 GEMM kernel (the same 15-uop bf16-round idiom that the OpenCL C compiler
folds and the driver does not).

Option (b) is built and working **for a batch-1 mixed-precision `Linear`**, and
`vit_infer_bf16` is still blocked — by batching, not by precision:

- `HostGemm` carries `a_dtype`/`b_dtype`/`bias_dtype` (defaulting to
  `dtype`/`c_dtype`, so existing descriptors and the frozen-plan format still
  parse); `extract` accepts a mixed 16/32-bit operand pair and a bias in a
  different dtype for a Vulkan descriptor;
- `coopmat::Stage` distinguishes `Direct`/`Bf16`/`F32` per operand, and
  `coopmat::Epilogue` names what runs after the accumulator: `None`,
  `BiasBf16OutBf16`, `BiasBf16OutF32` (new — the mixed-precision shape) and
  `BiasF32OutF32`;
- `stage_f32` (strided f32 -> row-major f16) and `epilogue_bf16_bias_f32` assemble
  and validate, and `USE_TC` gates the f32 staging (it rounds the activations).
  `coopmat_handles_mixed_precision_linear` pins both directions: `USE_TC` on gives
  one dispatch matching INTERP, `USE_TC=0` keeps the generated kernel.

Also fixed while wiring it: `coop_gemm` staged from the *scratch* buffer instead of
the operand (the tuple had already been shadowed), which corrupted the packed
tiles — that is why the mixed probe wrote the wrong width earlier.

`plan` still requires `batch == 1`. The ViT's projections are `batch = 8` (the
activations are `[8, 197, 768]` and `fold_batches` folds the leading axis), and the
per-slice dispatch loop that now exists loses the device on its *staging* dispatch
at that shape, so the batch offsets are still wrong somewhere — the next thing to
pin down (and the residual epilogue: 11 of the 48 projections also carry a
`res_slot`).

Also found on the way: `Tensor::rand(..., FLOAT16)` on `VK` produces an init
kernel (`E_..._2_16_2`) whose pipeline the driver rejects ("unknown error"), so
f16 parameters must be seeded through `from_slice`. Separate bug, not touched.

Still not wired into the graph pipeline: the
scheduler's `Ops::Wmma` arm (what `apply_tc_opt` feeds) hands the renderer the
CUDA `wmma` *packed register tile*, whose element order comes from
`element_per_thread` + swizzle; `OpCooperativeMatrixLoadKHR` instead reads from a
pointer with an explicit stride, so that route needs a `Workgroup` staging tile
*and* a re-derivation of the swizzle. The library-GEMM seam
(`schedule::host_gemm`, which already carries explicit strides per dimension) is
the tractable one: a `GemmDevice::Vk` + `host_exec` that patches
`coopmat::module` and dispatches it. That wiring was drafted and reverted: for a
`f16 -> f32` `dot` the descriptor's `extract` rejects the kernel before the
descriptor is built (the f32-accumulator promotion leaves a `Cast` in the reduce
that the affine matcher does not accept), so the path never triggered and I would
not ship an unexercised branch in a shared seam. Remaining work: teach `extract`
to recognise that reduced-precision-then-promoted matmul (`dtype` 16-bit,
`c_dtype` f32), re-enable the seam, and extend the module to bf16 storage (either
a bf16 -> f16 convert pre-pass or a staging tile) for the ViT/GEMM bf16 rows.

Numerics stay unchanged by construction: the module's accumulator *and* result
are f32, so `coopmat::eligible` only accepts a descriptor whose `c_dtype` is
f32 — a plain 16-bit `dot` (which accumulates in 16 bits) never takes it. That is
the conservative choice; the CUDA path's `USE_TC`-gated f32 accumulation for bf16
is the more aggressive alternative if the bf16 rows are to improve.
