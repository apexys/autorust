# Performance steps: what to try next

Date: 2026-11-14
Baseline: `autorust` at `70fa29b`, RTX 3060 (sm_86), Ryzen 9 3900X (24 threads),
CUDA 12.4, torch 2.6.0+cu124.

This document is the backlog for the next round of performance work. It is
ordered by **assumed impact**, not by effort. Every item names the evidence we
already have, the change to make, and the gate that decides whether it lands.
The last sections list the benchmark gaps that motivate the items, what was
already tried and measured (so it is not repeated), and the remaining
`PERFORMANCE_REVIEW.md` items.

Measurement protocol for every item: alternating builds, two processes, medians,
same machine, `AUTORUST_PROF=1`/`AUTORUST_KPROF=1` for attribution, and the full
test suite before committing. Do not land a "win" justified by one sample.

**Ranking update (2026-11-14, after the first attempts).** Item 1's premise was
measured and revised: the cacheable host work in a cold step is ~70-120 ms of
the ~555 ms (the rest is GPU clock ramp and first-launch driver work — the
median step falls 88 → 65 → 42 → 39 ms over the first five runs). Item 8 is
done for CL (96.9 → 26.4 ms) with VK remaining, item 9 landed with no measurable
step-time change, item 3's per-shape LRU landed, and TF32 (item 13) measured
−5.8% on ViT training. The impact order is therefore now:

1. **#2 conv** (~300 ms available, 17x gap)
2. **#3 automatic eager plans** (~1.5 s per eager step)
3. **#4 elementwise fusion** (~40% of generated kernels)
4. **#5 attention backward** (training gap)
5. **#1/#10 cold construction** (~100 ms one-off; the SINK-content schedule
   cache is the cheaper variant of #1)
6. **#6 HostGemm staging**, **#7 ONNX preprocessing**
7. **#8 VK optimizer**, **#13 TF32/bf16 opt-ins**, **#11/#12/#14/#15**

Item numbers below are kept stable so cross-references continue to work.

## Baseline (2026-11-14, `bench/results/`)

Gap to PyTorch (f32, medians):

| workload | autorust CUDA | torch CUDA | ratio | autorust CPU | torch CPU | ratio |
|---|---:|---:|---:|---:|---:|---:|
| MNIST bs 64 | 3.49 ms | 2.34 ms | 1.49x | 19.0 ms | 17.5 ms | 1.08x |
| ViT-B/16 infer bs 8 | 69.8 ms | 48.6 ms | 1.44x | 1103 ms | 827 ms | 1.33x |
| ViT-B/16 train bs 4 | 217 ms | 100 ms | 2.16x | 3861 ms | 1470 ms | 2.63x |
| ResNet-18 infer bs 32 | 323 ms | 18.8 ms | **17.2x** | 1196 ms | 374 ms | 3.20x |
| AdamW 100M | 25.7 ms | 25.4 ms | 1.01x | 246 ms | 324 ms | **0.76x** |
| ONNX MNIST bs 32 | 1.80 ms | 0.34 ms | 5.33x | 9.7 ms | 4.2 ms | 2.31x |
| ONNX ResNet-0.25 bs 8 | 9.68 ms | 1.81 ms | 5.35x | 127 ms | 17.4 ms | 7.26x |

Backends on this machine:

| workload | CUDA | CL | VK | RUST |
|---|---:|---:|---:|---:|
| MNIST bs 512 | 22.4 ms | 22.2 ms | 23.4 ms | 63.2 ms |
| ViT-B/16 infer bs 8 | 69.8 ms | 179 ms | 116 ms | 1103 ms |
| GEMM sweep (8) | 27.4 ms | 52.8 ms | 56.6 ms | 506 ms |
| memops sweep (8) | 5.03 ms | 7.97 ms | **4.79 ms** | 70.8 ms |
| AdamW 100M | **25.4 ms** | **26.4 ms** | 95.0 ms | 246 ms |

First process vs steady (`vit_infer` bs 4 unless noted):

| | ms |
|---|---:|
| first process, warm rendered/PTX cache (`ms_cold`) | 570-595 |
| first process, fresh cache | 1253 |
| steady JIT replay | 38.5 |
| eager steady step, no plan (`JIT=0`, ViT train bs 1) | 1620-1800 |

`AUTORUST_PROF=1` on the warm-cache cold start: `codegen.render` 83 ms (14
kernels, 5.7 ms each in `codegen.sink`), `cuda.module_load` 1.9 ms,
`compile.batch` ~0 on cached PTX, `sched.rangeify` ~50 ms for the capture,
`exec.linear` the actual first step.

---

# Ranked work items

## 1. Serialized ("AOT") lowered plans for GPU backends — close the cold-warm gap

**Evidence:** a replay step is 38.5 ms but the first step in a new process costs
570 ms even with every rendered source and PTX cached. The remaining cost is
host work that cannot be cached today: `callify` + `rangeify` + kernel split +
memory planning (~50 ms), re-linking the plan into the new arena, and runtime
construction (module load is only 1.9 ms). ONNX has the same story with a bigger
graph.

**Change:** persist the *lowered plan* — for each captured LINEAR, the ordered
CALL list with each program's content key, `ProgramInfo`, launch dims, buffer
slots and scalar vals — keyed by `(graph content key, device, settings,
dtype/shape signature)`. On a hit, rebuild the CALL/PROGRAM nodes from the cached
`CompiledProg`s (the engine already re-interns cached artifacts by content) and
skip rangeify/rendering entirely. This is the GPU analogue of `freeze/` for CPU
plans; start with a debug-gated implementation and a strict validation of the
rebuilt plan against the freshly scheduled one (kernel multiset + numerics).
Extend the key with a binary fingerprint so a codegen change invalidates it.

**Measured caveat (2026-11-14):** the cacheable host work in the warm-cache cold
step is `realize.schedule` 58 ms + `callify` 12 ms + render misses (0-51 ms) ≈
**70-120 ms of ~555 ms**. The rest is GPU clock ramp and first-launch driver
work: successive step medians are 88 → 65 → 42 → 39 ms. So the realistic win
here is ~100 ms per cold process, not 470 ms. The cheaper variant is the
**SINK-content schedule cache** (reuse the kernel split + memory plan, skip
`rangeify`/bufferize) before full plan serialization.

**Expected:** warm-cache cold start 555 ms → ~440-480 ms; deployment value comes
from removing the scheduling on *every* new shape/process, not from the first
step's GPU ramp. Do the SINK-content cache first.

**Effort/risk:** medium-large. The risk is buffer-slot and arena-id reuse; the
existing content-keyed `CompiledProg` reuse path is the precedent, and the plan
validation gate catches mismatches.

## 2. Convolution: implicit GEMM (or Winograd) instead of materialized patches

**Evidence:** ResNet-18 inference is 323 ms vs torch 18.8 ms (17.2x) and is the
largest absolute gap in the matrix. KPROF attributes ~300 ms to `r_*` conv
kernels; the im2col change only bought 14% (and needed a `K >= 576` gate to avoid
regressing small convs). `split_reduceop` is *not* the blocker (disabling it
changed nothing; only 38 extraction attempts reach `host_gemm`), so the conv
kernel creation path itself needs the change.

**Change:** either
(a) an implicit-GEMM kernel (cuDNN-style: tile over `(M = batch*oy*ox, N = cout)`
with the `(cin, kh, kw)` reduction streamed from the image, no materialized
patches), or
(b) Winograd F(2x2, 3x3) for the 3x3 layers (the bulk of the time), with im2col
fallback for the 7x7/1x1 layers.
Start with (a) as a generated kernel in `codegen` or as a backend host kernel;
the existing `pool` lowering already computes the exact patch indices, so the
implicit kernel can reuse that index logic.

**Expected:** 2-4x on the conv share → ResNet-18 CUDA 323 → 100-160 ms; the gap
to torch drops from 17x to 5-8x. Also helps UNet/MobileNet/ONNX ResNet.

**Effort/risk:** large. Do not attempt without a per-kernel numeric gate against
the current generated conv and a fallback for unsupported layouts.

## 3. Use the plan automatically for eager code

**Evidence:** eager ViT train bs 1 is 1.62-1.80 s vs ~80-100 ms with JIT (the
bs-4 JIT step is 217 ms; eager at bs 4 was not measured, it scales further).
`StepPlan` (`engine::jit`) exists but must be used explicitly; the
ONNX runner is the only automatic consumer. Rangeify + substitution +
bufferization dominate the eager host time (`rb.substitute` 0.48 s/step even
after pruning).

**Change:** a session-level helper that wraps a user step in a `StepPlan` keyed
by input shapes, with an LRU across shape signatures and `AUTORUST_PLAN=0` as
the escape hatch. Reuse the ONNX input-binding semantics (dedicated input
buffers, copy-in per call, recapture on shape/dtype change). Document the output
aliasing contract.

**Expected:** eager users see replay-like host cost (1.6 s → ~0.1-0.2 s on ViT
train). This is also the prerequisite for making "no JIT" a reasonable default
for inference.

**Status (2026-11-14):** `StepPlan` now keeps one plan per input signature with
an LRU (`StepPlan::set_capacity`, default 4) and a test for two shapes
(`tests/jit.rs::step_plan_keeps_one_plan_per_shape`). Remaining: wiring it into
the default eager path or documenting the opt-in API.

**Effort/risk:** small-medium; the machinery is done, this is API + lifecycle.

## 4. Elementwise fusion and launch reduction

**Evidence:** `AUTORUST_KPROF=1` on ViT train bs 4 attributes 7,910 `E_*`
launches / 172.5 ms (39.7% of KPROF time, 60.6% of `E_*`+`r_*`). Many are
sub-100 µs. The backward graph is a chain of small elementwise kernels that the
scheduler does not merge across store boundaries.

**Change:** a post-scheduling fusion pass that merges compatible elementwise
kernels in the same LINEAR (shared loop domain, same buffer sizes, no reduction
or library GEMM between them), with a hard rule to **never fuse across a
`host_gemm` descriptor**. Also merge consecutive stores into one multi-output
kernel where the scheduler currently emits one kernel per store. Validate with
the kernel multiset before/after (fewer kernels, same rendered math).

**Expected:** 30-50% fewer launches and 20-40 ms off ViT training; larger effect
on LLM decode.

**Effort/risk:** medium-large; fusion correctness is the risk, so gate with the
differential suite and the multiset check the review asks for.

## 5. Fused attention backward

**Evidence:** `vit_train` is 2.16x torch while inference is 1.44x. Training uses
graph SDPA plus generated backward kernels; flash attention is forward-only.
The review lists a fused backward kernel explicitly.

**Change:** implement backward for the fused FA2 path (dQ/dK/dV from the saved
output/softmax statistics), keep the graph SDPA path as the reference and
fallback, and gate on masks/GQA/odd lengths/dtypes. Reuse the existing
`host_call` seam.

**Expected:** removes a large share of the training attention cost; a plausible
10-20% off ViT training, more for long sequences.

**Effort/risk:** large, but self-contained and numerically gateable.

## 6. HostGemm staging and per-slice accumulation

**Evidence:** 170 descriptors are accepted at ViT train bs 4; the remaining
GEMM-shaped rejections are 48 `b-index-not-affine` (merged `(h,d)` axis +
strided destination), 12 `reduce-axes-not-contiguous` (needs `beta = 1` per
batch slice), and 36 computed operands (`(x-y) @ W`). The column-major work
already recovered 73 descriptors.

**Change:** two parts, can land separately:
(a) `beta = 1` accumulation in the CUDA path: one GEMM per outer reduce slice
accumulating into C, for the non-contiguous weight gradients;
(b) a staging path: when the destination has no unit-stride axis, GEMM into a
contiguous temporary and copy out (one extra kernel), or materialize the
non-affine operand first.
Both need new `HostGemm` descriptor fields (freeze serialization must be
updated) and a numeric gate against the generated kernel.

**Expected:** a few more percent on ViT training, and it removes a class of
"library GEMM silently unavailable" cases for other models.

**Effort/risk:** medium; the descriptor/freeze surface is the main cost.

## 7. ONNX graph preprocessing (fold, fuse, prepack)

**Evidence:** ONNX is 5.3x torch on both models even after the replay plan.
The importer currently executes the graph node-by-node as written.

**Change:** at load time: fold constants, fold Conv+BN+ReLU into one conv,
pre-transpose/repack weights for the GEMM path, drop identity/no-op nodes, and
co-realize the model into as few LINEARS as possible. The conv part also
benefits from item 2.

**Expected:** fewer kernels and less graph work per capture; 1.3-2x on ONNX
ResNet, less on MNIST.

**Effort/risk:** medium; a pure graph transform, gate with the ONNX operator
suite and golden outputs.

## 8. Port the fused optimizer kernel to CL and VK

**Evidence:** AdamW is 25.7 ms on CUDA but 96.9 ms on CL and 94.7 ms on VK,
because `runtime::cuda_optim` is CUDA-only and the others run the tensor-graph
update. This is now the largest backend-specific gap we created.

**Change:** the same kernel in CL C / Vulkan GLSL through the existing
`HostCall` seam (`run_host_kernel` already compiles per backend), gated like the
CUDA one (f32, contiguous, `n >= 1024`). CPU backends can keep the graph path
(or get a scalar loop with prefetch).

**Expected:** CL/VK optimizer 95 → ~30-40 ms; closes the 3.7x cross-backend gap.

**Status (2026-11-14):** CL is done (`runtime::cl_optim`) and measures **26.4 ms**
— CUDA parity. A numerical gate runs on CL too
(`tests/nn_optim.rs::fused_kernel_matches_graph_update_cl`). Vulkan is not
ported: its compute path needs explicit descriptor sets rather than the generic
`run_host_kernel` source seam, which is a separate piece of work.

**Effort/risk:** small-medium; the numerical formula is already validated
against the graph path on CUDA.

## 9. Fix the allocator free list (or prove it is irrelevant)

**Evidence:** new counters show `alloc.reuse = 0` and `alloc.backend ≈ 500-600/step`
at ViT train bs 4 — every step re-allocates device buffers from the driver.
Measured cost 0.010 ms each ≈ 6 ms/step (2-3%), and it scales with the number of
intermediate buffers rather than with tensor size.

**Change:** route dropped base allocations back into `LruAllocator::free`
instead of releasing them; keep views/`_offset` windows out of the free list;
add a cache-size cap and `free_cache()` on OOM. Verify `alloc.reuse > 0` and
that `alloc.backend` falls after the first step.

**Expected:** ~5 ms/step on ViT training and more on models with many small
buffers; also reduces driver pressure.

**Status (2026-11-14):** landed (`BufferBase::drop` returns base allocations to
the device free list, capped at 32 per size class; `try_device` makes the drop
glue safe during shutdown). Measured: `alloc.backend` 1817 → 1611 over the
profile window (−11%), `alloc.reuse` 206; **step time unchanged** (220.97/221.47
→ 219.99/218.97 ms on ViT train bs 4; MNIST unchanged). Direction closed: the
allocator is not a bottleneck, as the earlier measurement suggested.

**Effort/risk:** medium; buffer lifetime/aliasing rules need care (the pinning
bug in §9.59 is the cautionary tale).

## 10. Parallel rendering with an owned, `Send` kernel IR

**Evidence:** rendering 14 kernels takes 83 ms with 5.7 ms per kernel in
`codegen.sink` (the ~30 rewrite stages); it is serial because it mutates the
`Rc` arena. `compile_linear` already batches compilation, and CUDA NVRTC is
already parallel.

**Change:** hand post-scheduling kernels to a worker pool as an owned IR (or
give each worker an independent arena), render them in parallel, and merge only
the finished sources/`ProgramInfo` back into the session serially. Deterministic
kernel order and cache keys must be preserved.

**Expected:** cold rendering 83 ms → ~15-20 ms on a 24-thread CPU; combined with
item 1 the cold path gets close to the replay cost.

**Effort/risk:** large (cross-cutting `uop`/`codegen`); the review describes the
same design.

## 11. Scheduler: shared substitution worklist and fewer removable STAGEs

**Evidence:** `rb.substitute` is still 0.48 s/eager step after the AFTER pruning;
`subst.changed` is ~35k nodes while visits were 12.2M (0.38% relevant). The
review's second direction — create fewer removable STAGEs upstream — has not
been tried at all.

**Change:** (a) batch the `remove_bufferize` substitutions that share a producer
into one traversal/worklist instead of one DFS per site; (b) adjust bufferization
so fewer global STAGEs are created and then removed, measured separately so the
two effects are attributable. A per-node range-set cache (the 0.38% measurement)
is only worth it once a schedule-scoped lifetime exists.

**Expected:** eager host time down substantially (substitution is ~30% of the
eager step); no effect on JIT replay.

**Effort/risk:** large; gate on the kernel multiset.

## 12. Tiling/opt search with parallel candidate timing and a shipped cache

**Evidence:** `BEAM=1` is -13.7% on ViT inference (38.55 → 33.25 ms) and
`BEAM=2` -15.8% (32.35 ms), but the one-off search costs 55-82 s. A bounded
24-action set only reached -2.3% and was reverted.

**Change:** make the search worth using: time candidates concurrently (the
search currently compiles and times serially), warm it from a disk cache that
can be shipped/checked in for known models, and start from a curated candidate
list per kernel family (matmul/reduce/elementwise) rather than all 100 actions.
Then re-evaluate `BEAM=2` as an opt-in benchmark mode.

**Expected:** ViT inference -14-16% for users who can pay a one-time search, and
a path to using tuned opts without the cold cost.

**Effort/risk:** medium; mostly infrastructure, but the search must stay out of
the default path.

## 13. cuBLASLt epilogues, TF32 and reduced-precision opt-ins

**Evidence:** HostGemm currently runs cuBLAS + a separate epilogue kernel, and
f32 is the default everywhere. The review's GEMM table shows TF32 is faster at
some shapes with a numerics change.

**Change:** use `cublasLtMatmul` with the bias/scale epilogue fused into the
GEMM (removes one kernel per linear); expose TF32 and bf16 compute as explicit
opt-ins with accuracy tests. Also evaluate bf16 weights/activations for
inference (the bf16 ViT rows are already close to torch).

**Expected:** 3-8% on ViT inference/training; a larger win if bf16 mixed
precision is adopted.

**Measured (2026-11-14):** `AUTORUST_CUBLAS_TF32=1` on ViT train bs 4 gives
**217.2 → 204.6 ms (-5.8%)**; it changes numerics, so it stays an explicit
opt-in. The cuBLASLt epilogue fusion (removing the separate epilogue kernel) is
still open.

**Effort/risk:** medium; numerics changes must stay opt-in.

## 14. CUDA graphs for launch-heavy paths

**Evidence:** `llm_decode` is 16.5 ms/token with ~1,660 launches per captured
step; the optimizer that motivated this is already one kernel per parameter.

**Change:** capture the replayed plan into a CUDA graph keyed by shape, replay
per token, and fall back when buffers/scalars change. Requires replay-safe
buffer lifetimes and scalar updates (the JIT plan already pins buffers).

**Expected:** decode launch gaps shrink; the review calls it the main lever for
decode.

**Effort/risk:** large.

## 15. Symbolic / shape-generic plans

**Evidence:** every shape change recaptures; batch sweeps and dynamic serving
pay the whole plan build per shape. `ProgramInfo::launch_dims` ignores
`var_vals` and `vals()` returns empty.

**Change:** resolve launch dims, buffer sizes, index expressions and backend
vars at replay from `var_vals`; only then advertise one plan per model. This is
the review's "larger project, not a cache tweak".

**Expected:** no recapture on batch/shape changes; mostly serving-oriented.

**Effort/risk:** large; correctness of symbolic bounds is the risk.

## 16. Lower-priority and measurement items

- **Cubin cache:** measured module load at 1.9 ms / 0.3% of cold — not worth it.
- **Tiling-decision cache** by normalized shape/layout: only useful while the
  opt search (item 12) is being used.
- **Multi-stream CUDA:** overlap copies/GEMMs and the epilogue; the engine is
  single-stream today.
- **Phase-separated profiles:** model build vs first capture vs compile vs
  replay as separate tables; add an exclusive scheduler timer (the current
  `rw.pm_secs` is inclusive and nests).
- **CUDA-event KPROF:** the current profiler synchronizes per launch and
  serializes the workload; event timing would be accurate without that.
- **CPU `RAYON` for training:** RUST training is 2.63x torch CPU; check whether
  the RAYON backend is worth routing elementwise/reduction work to.
- **Weight prepacking:** cache repacked/transposed weights for the CPU host
  GEMM and for repeated im2col, instead of per-call packing.
- **Masked elementwise tails:** `PADTO` on a store axis is disabled because a
  vector store cannot be partially masked. A vector body plus a scalar tail
  would let prime/odd extents (ViT's 197) use 32-thread blocks instead of the
  2/8-thread blocks the divisibility heuristic forces; this is the generic form
  of the optimizer problem fixed by the dedicated kernel.
- **Host launch batching:** each CALL is one `cuLaunchKernel`; a launch-list or
  graph capture would cut per-kernel host cost in launch-heavy graphs (related
  to item 14 but usable without full graph capture).
- **`DEFAULT_DEVICE=auto`:** runs three benchmark cases at first tensor
  creation; make sure it is not accidentally enabled in cold-start-sensitive
  services (it is not the default).

---

# The cold-warm gap, in detail

Where the 570 ms warm-cache first step goes (measured or derived):

| component | cost | cacheable today? | idea |
|---|---:|---|---|
| model/data init (weights, rand kernels, 44 runtimes) | ~100-150 ms | partially (runtime construction) | reuse a preallocated `Session`; lazily upload weights; warm runtime cache |
| capture: callify + rangeify + kernel split + memory plan | ~50 ms | **no** | item 1: serialize the lowered plan |
| rendering (warm render cache hits) | ~5 ms (83 ms fresh) | yes (disk) | item 10: parallel rendering for the fresh case |
| NVRTC | 0 warm (268 ms fresh) | yes (PTX disk cache) | — |
| module load | 1.9 ms | yes (driver) | cubin cache not worth it |
| first execution (the step itself) | 38-190 ms (first-launch driver work at the high end; steady replay is 38.5) | no | inherent compute |
| host/library submission (`kx.host`) | ~130 ms as measured by the review (submission, not GPU time) | — | KPROF syncs; see item 16 (CUDA events) |

The components overlap and should not be summed; the point is which of them has
no cache today. Only two do: plan construction and the first execution.

The actionable gap is therefore **plan construction**, not compilation: the
schedule/lowering is the only large piece with no cache. Two ways to attack it:

1. **Persist the plan** (item 1) — the biggest lever, and it also helps ONNX
   (whose cold start is dominated by the same scheduling).
2. **Make construction cheaper** — parallel rendering (item 10), fewer
   realizes/LINEARS (co-realize more roots), constant folding for ONNX (item 7),
   and the scheduler work in item 11.

For warm *processes* the plan already exists; the remaining cold-warm cost is
the first-step execution itself, which no cache can remove.

Three more construction-cost levers beyond parallel rendering, in increasing
scope:

- **SINK-content schedule cache.** Hash the callified SINK (the kernel graph) and
  reuse the *schedule* — kernel split, per-kernel AST content keys and the
  memory plan — for identical graphs in the same process or from disk. This is
  item 1 without serializing the full CALL/ProgramInfo structure; it removes the
  `rangeify`/`split`/bufferize cost even when the plan rebuild still happens.
- **Fewer realizes/LINEARS.** Each realize pays a schedule; the optimizer now
  needs two and ONNX three. Co-realize forward/backward/optimizer roots where
  semantics allow; every merged linear removes one `create_linear_with_vars`.
- **Pass elision for static BUFFER graphs.** `callify` and parts of bufferize do
  work (shape/index materialization) that is identity when the graph is already
  all BUFFER leaves with static shapes; detect and skip those passes. This is a
  smaller, safer version of "don't refactor the scheduler": cache decisions
  rather than the graph.

---

# Benchmark gaps that motivate the ranking

1. **ResNet-18 conv (17.2x CUDA, 3.2x CPU)** — the only 10x-class gap.
2. **ViT training (2.16x CUDA, 2.63x CPU)** — attention backward + elementwise.
3. **ONNX (5.3x CUDA)** — graph-level conv/elementwise and per-node overhead.
4. **ViT inference (1.44x CUDA)** — close; BEAM could take it to ~1.25x.
5. **CL/VK ViT (1.7-2.6x CUDA)** and **CL/VK optimizer (3.7x CUDA)** — backend
   porting, not architecture.
6. **CPU ONNX (2.3-7.3x torch)** — the CPU host GEMM is fine for large matmuls;
   the gap is conv lowering and graph overhead, same as GPU item 2/7.
7. **memops:** VK is the fastest backend (4.79 ms vs 5.03 CUDA) — worth
   understanding before optimizing CUDA elementwise further.

---

# Already tried and measured (do not repeat)

- HostGemm batch cliff: two-axis strided-batched cuBLAS, column-major outputs,
  contiguous reduce collapse → 170 descriptors at bs 4; ViT train bs 4 went
  331 ms (review baseline) → 217 ms now across those commits.
- Capture-once (`Rc<CompiledLinear>`), single AST hash, O(1) LRU, cached
  scheduler metadata, `realize_many` map.
- ONNX replay plan → 11.7 → 1.7 ms / 73.8 → 12.0 ms (9.7 after the conv gate).
- `StepPlan` eager API + test; `StepPlan` is not auto-enabled (item 3).
- Fused CUDA AdamW → 104 → 26 ms (torch parity).
- im2col with `K >= 576` → ResNet-18 323 ms / 289 ms train; ONNX ResNet 9.7 ms.
- RUST/CC `compile_batch` pool → -18.6% cold.
- `remove_bufferize` AFTER pruning → `subst.nodes` -29%, eager step -8.4%.
- Instrumentation: HostGemm dispatch/reject detail, keyed KPROF, synchronized
  `host_gemm` rows, `alloc.*`/`cuda.module_load` counters, `BENCH_LOSS`/`SEED`,
  `BENCH_ITERS=0`.
- Disproved/measured-and-skipped: bounded 24-action opt search (-2.3%, reverted),
  GEMM-before-`split_reduceop` (no effect), cubin cache (0.3%), allocator as a
  bottleneck (2-3%), module load.

# Remaining `PERFORMANCE_REVIEW.md` items, mapped

| review item | status |
|---|---|
| Reusable plan cache above rangeify (P0 #2) | ONNX done; eager API done; automatic/serialized forms = items 1, 3 |
| Batch reduction coverage / HostGemm (P0 #1) | mostly done; staging/beta=1 = item 6 |
| Per-site substitution rewrite (P1 #3) | partial (AFTER pruning); worklist/bufferization = item 11 |
| Grouped optimizer (P1 #4) | CUDA done; CL/VK = item 8; SDPA/FFN fusion = item 4 |
| Parallel cold lowering (P2 #5) | compile batching done; rendering = item 10; cold profiling = item 16 |
| Shape-specialized plans / symbolic batch (F6) | specialization done; symbolic = item 15 |
| im2col / vendor-GEMM conv | im2col done; implicit GEMM/Winograd = item 2 |
| Fused attention backward | item 5 |
| CUDA graphs | item 14 |
| Tiling cache / BEAM | item 12 |
| Cubin cache | measured, skip |
| Allocator at large batches | measured, no bottleneck; free-list = item 9 |
| Frozen plans for production | exists for CPU; GPU form = item 1 |

---

# Suggested sequence

1. **Item 1** (serialized plans) — biggest broad win, and it makes every cold
   benchmark in the repo faster.
2. **Item 2** (conv implicit GEMM) — biggest absolute gap; do a per-kernel
   prototype before committing to the full change.
3. **Item 3** (automatic eager plan) — small, unlocks the eager story.
4. **Item 4** (elementwise fusion) or **item 5** (attention backward) — pick by
   which profile (launch-bound vs attention-bound) the previous items leave.
5. **Item 8** (CL/VK optimizer) — cheap backend parity.
6. Everything else by available time; item 9 is a small independent warm-up.
