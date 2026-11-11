# Performance

Detailed performance notes, benchmark methodology, device selection, and the
tuning knobs behind them. The README only carries the headline numbers; every
number here was measured on a 24-thread Ryzen 3900X / RTX 3060 unless stated
otherwise. Environment variables are catalogued in
[`CONFIGURATION.md`](CONFIGURATION.md).

## Measuring

Steady-state numbers use a captured `Jit` (`JIT=1` in the examples), which
replays the compiled linears and excludes rangeify/scheduling. Two artifacts
matter when comparing against older numbers:

- `realize_fresh` / `to_list` in a timing loop performs a **synchronous
  D2H copy every iteration**. The old 2.2 TFLOP/s GEMM figures were dominated by
  that copy, not by the kernel.
- `JIT=0` re-schedules through the interpreter on every iteration (one core at
  100%, ~0 GFLOP/s for GPU work).

Use `SYNC=1` / `NOSYNC=1` in `gemm_bench` to time the launch on the stream
without the per-iteration readback.

Diagnostics:

| variable | effect |
|---|---|
| `AUTORUST_PROF=1` | scheduler phase table (`rangeify`, `kernel_graph`, …); also enabled by `TG_RW_TRACE`/`TG_RW_VISITS` |
| `AUTORUST_KPROF=1` | per-kernel CUDA time table (syncs per launch; no GPU performance-counter permission needed) |
| `AUTORUST_KERNEL_INFO=1` | registers / shared memory / local bytes per compiled CUDA kernel |
| `FROZEN_PROF=1` | per-kernel and host-GEMM counters inside a `FrozenRunner` (`report_profile()`) |
| `AUTORUST_MM_DEBUG=1` | generated matmul tiling decisions |
| `AUTORUST_CPU_GEMM_DEBUG=1`, `AUTORUST_CUBLAS_DEBUG=1` | host-GEMM dispatch and descriptor details |

Benchmarks (`cargo run --release --example …`):

| example | what it measures | useful knobs |
|---|---|---|
| `gemm_bench` | one square/rectangular GEMM | `DEV M N K ITERS WARM JIT BIAS DT ACC_SAME BT SYNC NOSYNC CHECK PROF` |
| `vit_bench` | ViT inference | `DEV JIT LAYERS EMBED HEADS IMG PATCH ITERS` |
| `vit_train_bench` | forward+backward+Adam step | `DEFAULT_DEVICE JIT LAYERS EMBED HEADS IMG ITERS LR DTYPE TRAINING` |
| `fa_bench` | SDPA vs fused attention (fwd and fwd+bwd) | `BS HEADS SEQ DIM FA_BQ FA_BK WARM ITERS` |
| `rayon_bench` | elementwise `RAYON` throughput | `N ITERS BENCH_SKIP_INTERP VERBOSE` |
| `cublas_check` | library GEMM vs generated kernels on the same ViT weights | `LAYERS` |

Read the source of each example for the full list; `DEV`/`DEFAULT_DEVICE`
selects the backend.

## Generated GEMM

Matmul-shaped contractions (`MUL -> REDUCE(ADD)`) get a register tile from the
scheduler (`codegen/postrange.rs::try_matmul_tiling`) before the generic
heuristic:

- **CPU (`RUST`/`CC`)**: upcast both output axes 8x8 and unroll K by 4. The
  inner loop does 64 FMAs from eight vector loads.
- **GPU (`CUDA`/`CL`)**: `LOCAL` + `UPCAST` + `UNROLL`. Small/medium shapes use
  4x4 registers with 2x64 threads (block 8x256); large (`M*N >= 1M`) use 8x8
  with 4x16 threads (block 32x128). When the M extent is prime-ish (ViT batch-8
  `M = 1576 = 8*197`), an 8-wide register tile would leave a *single* thread on
  M (16-thread blocks); the tile steps down to 4x8 with 2x16 threads, which
  took the batch-8 FFN expansion from 8.1 to 2.9 ms (2.8x). `matmul_ranges`
  returns A's and B's private axes in operand order, not store order, so both
  orientations are scored and the one with more block threads wins; a block
  below one warp trades register tile for threads. The ViT backward `dX`
  (M=197 prime, N=3072) went from a 2-thread block at 5.6 ms to a tiled kernel
  at 2.6 ms.
- **Grid order** (`codegen/gpudims.rs`): global axes are sorted by extent
  ascending, so the largest dimension becomes `blockIdx.x` (fastest-varying).
  For `M=197` linears this runs all M blocks for one N tile before moving on,
  keeping the weight N-tile hot in L2.
- **Tensor cores** (CUDA, `sm_80+`): f32 uses the TF32 `mma` layout and bf16 the
  bf16 layout, gated on `Settings.tensor_cores`. An output-aspect gate
  (max/min ≤ 2) keeps the FFMA tiling for tall/wide shapes, where WMMA loses.
- **Sums of contractions** are materialized per contraction before kernel
  splitting (`pm_generate_realize_map`), so each becomes a single-reduce GEMM.
  The ViT attention input gradient (`dQ@Wq + dK@Wk + dV@Wv`) used to be one
  scalar three-reduction kernel (5.7 ms/layer).
- **Pointwise contraction operands** (the GELU derivative, the dropout mask,
  the softmax backward, movement views in training) are staged exactly once
  inside the reduce body (`convert_reduce_to_ranges::stage_reduce_prologue`),
  which the generic realize pass cannot reach after movement folding. Besides
  making those contractions affine enough for cuBLAS, it avoids recomputing
  the prologue per output column. Training 179 -> 94 ms; the split + staging
  together took it from 332 to 94 ms.

The generated f32 kernel is memory-bound at large K (no shared-memory tiling;
every block re-reads A/B from global). Kernel-only (2026-11, `SYNC=1`,
`NOSYNC=1`):

| shape | generated f32 | cuBLAS f32 | cuBLAS TF32 |
|---|---|---|---|
| 1024³ | 2.54 TFLOP/s | 4.11 | — |
| 2048³ | 3.58 | 4.76 | **5.79** |
| 4096³ | 3.09 | 4.48 | — |
| 6312x384x384 (batch-8 qkv) | 0.85 | **2.23** | 1.85 |
| 6312x1536x384 (up) | 0.62 | 1.23 | **1.29** |
| 6312x384x1536 (down) | 0.89 | **3.57** | 3.49 |
| 1576x3072x768 (batch-8 FFN, 4x8 tile) | 2.57 | **3.46** | — |
| 1576x768x3072 (batch-8 down, 4x8 tile) | **4.31** | 3.96 | — |
| 2048³, bf16 | **10.68** (FFMA, no `mma`) | 9.84 | — |

Old "before" rows in the README (`512³ CUDA 144 → 650-830 GFLOP/s`,
`2048³ bf16 ~2-3.5 TFLOP/s`) came from the readback-inflated harness; the
kernel-only table above supersedes them. bf16 is fast mainly because the
storage halves the DRAM traffic, not because the generated kernel emits `mma`.

### Convolutions

Conv is not a `HostGemm`: it lowers to `pad -> pool -> expand -> mul -> sum`
over `(cin, ky, kx)`. `try_conv_tiling` expresses it as an implicit GEMM
(`M = batch*oy*ox`, `N = cout`, `K = cin*ky*kx`) without materialising im2col:
output channels on threads (coalesced weights), **both** spatial output axes
upcast into registers (so a 5x5 weight tap is reused across two rows), and the
kernel axes unrolled. On MNIST/RTX 3060 this took the training step from 588.6
to 258.4 ms at bs=512 (one-axis upcast) and 39.2 to **21.7 ms** with the second
axis; see `CURRENT_STATE.md` §9.16-§9.17 and §9.20. Materialised im2col + `dot`
(route B, deferred) remains the way to use the vendor GEMMs and the packed CPU
kernel for conv.

## Host GEMM (library path)

A pure contraction can bypass generated code entirely. `schedule/host_gemm.rs`
recognises a store of `REDUCE(ADD, MUL(load, load))` with affine `INDEX`
strides and turns it into a `HostGemm` descriptor: shapes, strides, dtype,
`alpha`, and an optional fused epilogue (`bias + per-column scale + residual`).
`engine/realize.rs` then runs the backend's own implementation:

- **CUDA → cuBLAS** (`cublasSgemm_v2` / `cublasGemmEx`, f32/f16/bf16,
  strided-batched, in-place epilogue kernel). Loaded dynamically
  (`libcublas.so.12/13`).
- **CL → CLBlast** (`CLBlastSgemm` / `CLBlastSgemmStridedBatched`, f32,
  row-major). Install `libclblast1` (`apt install libclblast1`); without it the
  OpenCL backend keeps using generated kernels.
- **CPU → packed blocked kernel** (`RUST`/`CC`/`RAYON`, f32). Rayon over
  disjoint `(m,n)` tiles, a 6x16 AVX2/FMA microkernel (12 YMM accumulators) and
  a scalar fringe, so prime extents need no padding or masks. A row-major `A`
  is read directly (no pack) and a `[n, k]` weight is transposed into one
  contiguous `[k, n]` panel per GEMM; the old per-task packing re-read `B` once
  per m-block and `A` once per n-block, which dominated the small-M ViT linears.
  Microbenchmark (24 threads, f32, `gemm_bench`; MKL = torch 2.6 `torch.mm`):

  | shape | generated `RUST` | packed before | packed now | MKL |
  |---|---|---:|---:|---:|
  | 197x768x768 | 94 | 273 | **290** | 93 |
  | 197x3072x768 | 30 | 284 | **296** | 417 |
  | 197x768x3072 | 65 | 315 | **464** | 613 |
  | 1576x768x768 | 156 | 313 | **375** | 609 |

  MKL still wins the larger shapes by ~1.3-1.6x, and the host GEMM is only
  ~80-100 ms of the step. The measured per-kernel profile
  (`AUTORUST_KPROF=1`) put ~271 ms in one generated FFN-expansion kernel
  (`linear` + GELU fused, ~30 GFLOP/s) whose GELU epilogue blocked the host
  GEMM. The scheduler now realizes *interior* host-backend contractions
  (`realize_interior_contractions`, K >= 512, one plain weight operand), stages
  their non-affine operands, and closes the store ENDs the realization leaves
  open (`close_store_ends`). The FFN expansion becomes a packed host GEMM
  (~296 GFLOP/s) plus an elementwise GELU kernel. A big contraction's pointwise
  operands are staged too, which is what turns attention's biased `Q`/`K` into a
  plain batched GEMM: RUST ViT-B/16 inference 425 -> **173 ms** at bs 1
  (1.05x torch), 1902 -> **1164 ms** at bs 8 (1.66x), and training 1784 ->
  1634 ms. The step is now dominated by the packed GEMM itself (~300 GFLOP/s vs
  MKL's ~600), i.e. the AVX2 microkernel, not scheduling. Weight-gradient
  contractions (two activation operands) stay fused where staging would only
  add copies. A batched host GEMM also packs its weight once per call instead
  of once per batch. Generated f32 tensor cores are not the lever: the prime
  197 axes are declined and disabling them is ~5% faster on these shapes.
- **Metal → MPS** (`MPSMatrixMultiplication`, f32/f16). Opt-in
  (`AUTORUST_MPS=all`) because simdgroup tensor cores measured faster on an M2.

Selection (`Settings::library_gemm`, per backend):

| backend | `Auto` policy |
|---|---|
| CUDA f32 | always use cuBLAS (faster at square and batch-8 projection shapes) |
| CUDA f16/bf16 | library only when some of `m,n,k` is not divisible by 4 |
| CL | same shape heuristic |
| CPU | library only above `m*n*k >= 1_000_000` (packing costs dominate below) |
| Metal | never in `Auto`; MPS only with `Always` / `AUTORUST_MPS=all` |

`AUTORUST_GEMM` sets the neutral policy; the backend-specific variables win over
it. Values: `0`/`never`, `all`/`force`/`always`, anything else = `Auto`. A
missing library simply disables that backend's path.

| switch | effect |
|---|---|
| `AUTORUST_GEMM=all\|0` | force/disable the library on every backend |
| `AUTORUST_CUBLAS=all\|0` | CUDA-only override |
| `AUTORUST_CLBLAST=all\|0` | OpenCL-only override |
| `AUTORUST_CPU_GEMM=all\|0` | CPU-only override |
| `AUTORUST_MPS=all\|0` (alias `AUTORUST_METAL_GEMM`) | Metal MPS override |
| `AUTORUST_CUBLAS_TF32=1` | allow TF32 for f32 cuBLAS GEMMs (off by default: it changes the output for no measured gain) |
| `AUTORUST_CUBLAS_LIB`, `AUTORUST_CLBLAST_LIB` | override the library path |
| `AUTORUST_CPU_GEMM_MC/NC/KC` | CPU tile sizes (default 48 / 96 / 256) |
| `AUTORUST_MM_M/N/K/LM/LN` | generated-matmul tile override (all five required) |

Results (448px DINOv3 ViT-S/16, 128 frames): CLBlast takes OpenCL from 27.0 s to
9.5 s (CUDA/cuBLAS 6.9 s). The CPU host GEMM takes the frozen CPU plan from
21.7 s to 13.9 s per 16 frames and emits 48 host-GEMM steps in the frozen plan
instead of generated kernels (freeze serialises the descriptor, so AOT and eager
agree). On CUDA, ViT-B/16 forward went from 28-29 ms to **21-22 ms** (~25%).

## Attention

One operation, three implementations selected by `Settings.attention` (or the
`Tensor` method used):

- `sdpa`: everything — masks, causal, dropout, GQA, any `d`, any dtype.
- `flash`: online-softmax kernel, algebraically identical (`1.2e-8`), but only
  for the shapes the hand-written kernels cover.
- `sage`: QKᵀ on 8-bit tensor cores; approximates.

Hand-written kernels are reached through `Compiled::run_host_kernel`, the seam
for code the uop IR cannot express (the IR cannot carry per-thread state like
the online softmax). The source is compiled by the device's own compiler
(NVRTC, `clBuildProgram`, rustc) and launched through its runtime.

CUDA, same math, three kernels:

| kernel | when | batch 8, h=6, n=789, d=64 | error vs f32 |
|---|---|---|---|
| scalar f32 FA2 | any `sm` | 7.0 ms | 1.2e-8 |
| warp-tiled TF32 (`mma.m16n8k8`) | f32, `sm_80+`, tensor cores on | **1.18 ms** (5.9x) | ~9e-6 |
| bf16 (`mma.m16n8k16`, `ldmatrix`, `cp.async`) | bf16, `sm_80+` | **0.52-0.54 ms** | <1e-4 vs bf16 reference |
| INT8/FP8 Sage (`mma.m16n8k32`) | opt-in, `sm_80+`/`sm_89+` | 1.26 ms (vs 0.97 TF32) | ~2e-5 |

The bf16 kernel uses `__launch_bounds__(128, 5)` (96 regs, 18.4 KB shared, no
spills) and a two-stage `cp.async` pipeline (1.3-1.4x over synchronous staging).
The TF32 kernel keeps synchronous staging: the doubled shared memory halves
occupancy there. `AUTORUST_ASYNC=0` disables the pipeline;
`AUTORUST_TENSOR_CORES=0` forces the scalar kernel.

Feature coverage in the fused kernels: causal (block-level skip + diagonal
mask), additive `[n, n]` f32 mask, GQA, and no dropout. Unsupported
shapes/dtypes/masks fall back to SDPA. OpenCL has the same schedule in OpenCL C
(1.32 ms at batch 1 / 8.33 ms at batch 8, 1.04 ms/image/layer vs 0.73 on CUDA,
`1.2e-8` against CL SDPA). Metal has its own FA2 kernel and native `simdgroup`
8x8 tensor cores; MPS host GEMM is opt-in.

Attention's `P·V` can also reach the host GEMM: with
`Tensor::scaled_dot_product_attention_sess` the score matrix is materialised, so
the contraction has a buffer to describe. That is a CPU win (~150 ms → ~7 ms per
image for the 12 AV products of a ViT-S/16) and a GPU loss, because writing and
re-reading `P` (`batch*heads*t*t`) costs more bandwidth than fusing saves. The
choice belongs to the caller.

The prime 197-token ViT sequence is why the host path exists: the register tile
cannot vectorise 197, while the library can. `SDPA_TILE=32` pads attention at the
tensor level (correct, currently slower, off by default). Codegen `PADTO` on a
vector-store axis is deliberately unused: the store mask is not lane-wise.

## Devices and selection

Default selection is the first available backend in
**CUDA → METAL → CL → RUST → CC → RAYON → INTERP** order.

| device | platform | run-time requirements | notes |
|---|---|---|---|
| `CUDA` | Linux, Windows | NVIDIA driver (`libcuda`) + NVRTC (`libnvrtc`) | cuBLAS optional; TF32/bf16 `mma` needs `sm_80+`, FP8 needs `sm_89+` |
| `METAL` | macOS | OS Metal/Objective-C frameworks | no Xcode needed (shaders compile at run time); f64 lowered to f32 |
| `CL` | anywhere | OpenCL ICD loader (`libOpenCL`) | CLBlast optional; no offset views, bf16 stored as `ushort` |
| `RUST` | anywhere with `rustc` | `rustc` on `PATH` | first run compiles kernels (seconds), x86-64-first |
| `CC` | anywhere with a C compiler | `$CC`, else `clang`, else `cc` | first run compiles kernels (seconds) |
| `RAYON` | anywhere | — | multithreaded typed-IR interpreter |
| `INTERP` | anywhere | — | scalar reference; always available, slowest |

`DEFAULT_DEVICE` overrides the priority. `autorust::set_default_device("RAYON")`
sets it at run time, and `autorust::select_fastest_device()` runs three
ML-shaped cases (`relu(A @ W + bias)`, 4-head causal attention at 256 tokens, a
2048×512 row softmax), validates each result, and installs the winner:

```rust
let dev = autorust::select_fastest_device();
// prints e.g. `autotune: METAL 7.9 ms (linear 1.4, attention 4.4, softmax 2.1),
// RAYON 260.8 ms (...) -> METAL`
```

`DEFAULT_DEVICE=auto` runs the same selection lazily on the first tensor
creation. `AUTORUST_AUTOTUNE_DEVICES=METAL,RAYON` restricts the candidates,
`AUTORUST_AUTOTUNE_QUIET=1` suppresses the summary, and
`AUTORUST_AUTOTUNE_TRACE=1` prints the per-case samples. `INTERP`/`CPU` are never
benchmarked.

The priority assumes the GPU is faster, which is not true on older Apple
hardware (a slow Intel Iris can lose to the CPU) — that is what the autotune
path is for. The first `RUST`/`CC` run compiles kernels with `rustc`/`cc`, so
allow a few seconds there; CUDA compiles each kernel with NVRTC on first use and
caches the PTX.

GPU backends are indexable: `CL`, `CL:1`, … and `CUDA`, `CUDA:1`, … each open a
distinct physical device (index 0 is the bare name, matching tinygrad's
canonicalization). `CUDA_DEVICE=1` / `CL_DEVICE=1` choose the index used by the
priority-picked default, and `select_fastest_device()` benchmarks **every**
physical device and installs the exact winner — a workstation with an integrated
and a dedicated GPU is compared per device, not per backend.

## End-to-end snapshot

ViT-B/16 (`JIT=1`, steady state):

| workload | before | now |
|---|---|---|
| forward, `RUST` CPU | 4.1 s | **0.17 s** |
| forward, `CC` CPU | — | **0.63 s** |
| forward, frozen `RUST` plan | — | 0.86 s |
| forward, CUDA | 166 ms | **14 ms** (fused FA2) |
| training step (fwd+bwd+Adam, CUDA) | ~387 ms | **96 ms** |
| 512³ f32 GEMM, `RUST` | 27.6 GFLOP/s | **~370 GFLOP/s** |

Clusterfun (DINOv3 ViT-S/16, 448px): CUDA + cuBLAS + fused FA2 **5.6 s** / 128
frames (candle CUDA 6.35 s), OpenCL + CLBlast + fused FA2 **8.4 s**, frozen CPU
6.1 s / 16 frames (0.37 s/image, candle CPU ~0.63). With default preprocessing
(260 frames): CUDA 14.1 s, OpenCL 20.1 s, candle CUDA 15.4 s.

Accuracy (8 images, 448px, CUDA vs frozen CPU plan): embeddings 1.4e-6 max abs
(cosine 0.9999995), CUDA fused attention vs SDPA 8.0e-7, CL 7.9e-7, frozen check
5.5e-6. Pedantic cuBLAS + TF32 attention is 3.8e-4 / 0.9999971, i.e.
tensor-core attention costs about as much as TF32 cuBLAS and is worth it, while
TF32 cuBLAS is not (hence the default).

## Cold start and scheduling

The steady-state medians above hide the front end, but a fresh process still
has to build the graph, rangeify it and render every kernel. Two things make
that cheaper:

- **Persistent rendered kernels** (`engine/realize.rs`). The PTX cache is keyed
  by source text, so it only helps after rendering. The render result
  (ProgramInfo, ELF signature, source) is now stored under the AST content key
  plus the codegen settings and a fingerprint of the running binary, so a
  second process (or a later realize of the same kernel) skips the ~30
  pipeline stages. On a ViT-B/16 training step rendering is ~4.4 s of the
  cold start and drops to ~50 ms on a hit; `CACHELEVEL=0` disables it.
- **Pipeline timings** under `AUTORUST_PROF=1`
  (`codegen.sink/pi/linearize/source`, `compile.batch`, one row per stage, and
  `sched.*`/`rb.*` for rangeify). `AUTORUST_MM_DEBUG=1` additionally prints the
  chosen matmul tile or the reason `matmul_ranges` rejected a kernel.

Warm-cache cold start (`ms_cold`, same process protocol as the table):

| workload | before | now |
|---|---|---|
| MNIST CNN train, bs512 | 505 ms | **216 ms** |
| ViT-B/16 infer, bs1 | 697 ms | **474 ms** |
| ViT-B/16 infer, bs8 | 1279 ms | **760 ms** |
| ViT-B/16 train, bs1 | 6013 ms | **2498 ms** |

`bench/run.py` warms the on-disk caches with a discarded `ITERS=0` process
before the measured one, matching the warm-cache protocol used for the other
frameworks.

## Cross-framework comparison

[`bench/`](bench/README.md) runs the same two models — `beautiful_mnist` CNN
training and ViT-B/16 — through autorust, candle, torch and tinygrad on CPU and
CUDA. Inputs are synthetic and device-resident, the timed region is one
training/inference step (no data loading, evaluation or logging), and each run
is a fresh process with warmup. Raw JSON and the generated per-run tables are in
[`bench/results/`](bench/results); these medians are from the `v0.1.1` release
sources (autorust CPU and CUDA rows re-taken together after the codegen,
packed-GEMM and attention changes in CURRENT_STATE §9.20-§9.49) on the
environment above
(the candle runs were re-taken at `82f6408` with the class-token fix; torch is
eager, no `torch.compile`; autorust uses `Jit`, tinygrad `TinyJit`).

All step times are milliseconds (lower is better).

| workload | device | autorust | candle | torch | tinygrad |
|---|---|---:|---:|---:|---:|
| MNIST CNN train, bs 512 | CPU | **65.5 ms** (RUST) | 3902.0 ms | 175.9 ms | 3464.8 ms |
| MNIST CNN train, bs 512 | CUDA | 22.2 ms | 131.9 ms | **13.1 ms** | 56.8 ms |
| ViT-B/16 infer, bs 1 | CPU | 170.6 ms (RUST) | 229.4 ms | **165.1 ms** | 14639.3 ms |
| ViT-B/16 infer, bs 1 | CUDA | 14.2 ms | 19.2 ms | **8.7 ms** | 116.9 ms |
| ViT-B/16 infer, bs 8 | CPU | 1188.0 ms (RUST) | 1403.5 ms | **701.5 ms** | — |
| ViT-B/16 infer, bs 8 | CUDA | 70.5 ms | 114.6 ms | **46.4 ms** | 136.2 ms |
| ViT-B/16 train, bs 1 | CPU | 1634.5 ms (RUST) | **243.4 ms** | 616.3 ms | — |
| ViT-B/16 train, bs 1 | CUDA | 95.9 ms | **19.0 ms** | 43.8 ms | 388.5 ms |

Reading it:

- **MNIST CUDA**: autorust is 2.6x faster than tinygrad and 1.7x slower than
  torch (cuDNN + cuBLAS + fused BN/optimizer). The direct-conv tile now reuses
  the 5x5 weights across two output rows, which is most of the gap that
  remains.
- **ViT inference CUDA**: eval mode uses the hand-written FA2 kernels through
  `ViT::forward_with`, so torch's cuBLAS/cuDNN path is only ~1.5x ahead at
  both batches. autorust is 8.6x/1.9x faster than tinygrad and ~1.6x faster
  than candle at bs 8.
- **ViT training CUDA**: autorust is 4.1x faster than tinygrad and 2.15x slower
  than torch. Splitting the summed q/k/v input gradient and staging the
  pointwise contraction operands (GELU derivative, dropout mask, softmax
  backward) took the step from 332 to 94 ms; the remaining gap is the f32
  backward against cuBLAS/cuDNN (see §9.20-§9.23).
- **CPU**: autorust's `RUST` backend is 60x/60x faster than tinygrad/candle on
  MNIST and now lands within 2x of torch on ViT inference at both batches
  (173 vs 165 ms at bs 1, 1164 vs 702 ms at bs 8); training is 2.65x (torch's
  MKL GEMMs and its fused attention/optimizer are the gap). tinygrad's
  `CPU` backend is single-threaded by design. `RAYON` is the typed interpreter
  (`mnist_train` bs 32: 348 ms/step) and is not a throughput backend.

Caveats that matter for the comparison: tinygrad CPU is single-threaded;
candle-nn's BatchNorm is inference-only, so its MNIST training normalizes with
batch statistics but does not update running stats; candle's ViT forward
dominates its step (a manual profile shows ~1.4 ms of autograd against an
~18 ms forward), so its training and inference medians are nearly equal;
autorust runs with a warm on-disk kernel cache, while the other frameworks
compile per process (`ms_cold` is recorded in the JSON but excluded from these
steady-state medians).

### Batch-size scaling

The headline rows compare at one batch each. Re-running the same steps at other
batches (`bench/run.py --only '.*(mnist-bs64|vit_train-bs4)'`; raw JSON in
`bench/results/`) shows that the MNIST gap is stable while the ViT training gap
grows with batch:

| workload | device | batch | autorust | torch | ratio |
|---|---|---:|---:|---:|---:|
| MNIST CNN train | CPU | 64 | 9.9 ms | 19.9 ms | 0.50x |
| MNIST CNN train | CPU | 512 | 65.5 ms | 175.9 ms | 0.37x |
| MNIST CNN train | CUDA | 64 | 3.1 ms | 2.3 ms | 1.32x |
| MNIST CNN train | CUDA | 512 | 22.2 ms | 13.1 ms | 1.69x |
| ViT-B/16 train | CPU | 1 | 1634.5 ms | 616.3 ms | 2.65x |
| ViT-B/16 train | CPU | 4 | 4205.1 ms | 1498.0 ms | 2.81x |
| ViT-B/16 train | CUDA | 1 | 95.9 ms | 43.8 ms | 2.19x |
| ViT-B/16 train | CUDA | 4 | 317.1 ms | 102.4 ms | 3.10x |

A wider ad-hoc sweep (five timed iterations per point) on the same steps shows
the per-image cost directly. ViT-B/16 training, ms per step and per image:

| device | batch | autorust step | autorust ms/img | torch step | torch ms/img |
|---|---:|---:|---:|---:|---:|
| CUDA | 1 | 98.8 | 98.8 | 45.8 | 45.8 |
| CUDA | 2 | 169.6 | 84.8 | 65.5 | 32.7 |
| CUDA | 4 | 320.8 | 80.2 | 102.2 | 25.5 |
| CUDA | 8 | 628.8 | 78.6 | 177.0 | 22.1 |
| CPU | 1 | 1600.9 | 1600.9 | 609.5 | 609.5 |
| CPU | 2 | 3152.8 | 1576.4 | 999.0 | 499.5 |
| CPU | 4 | 4211.8 | 1053.0 | 1342.4 | 335.6 |

The step time grows almost linearly with batch on both devices: the generated
f32 GEMM is close to saturation at batch 1 (per-image 98.8 -> 78.6 ms on CUDA
from bs 1 to 8), so extra rows buy little, while cuBLAS keeps improving its tile
choice (torch's CUDA per-image cost halves from bs 1 to bs 8). In other words, the remaining ViT-training gap is not launch or
scheduling overhead that a bigger batch would amortize; it is the steady-state
GEMM/elementwise throughput of the f32 path. MNIST, which is conv- and
optimizer-heavy, behaves differently: on CPU autorust is 2x faster than torch at
bs 64 and 2.7x at bs 512, and on CUDA it is within 1.3x at bs 64.

CPU batch sizes above 1 used to hang: a cold `threefry` RNG kernel made rustc's
loop vectorizer run for over ten minutes (CURRENT_STATE §9.50). The RUST
renderer now keeps that vectorizer off shift-chain kernels, so the batch 2/4
rows run in seconds.


## Microbenches

The model tables hide *why* a step costs what it does, so the matrix also runs
three isolated microbenches (`bench/run.py --only '.*gemm|.*memops|.*optim'`;
raw per-case JSON in `bench/results/`).

### GEMM sweep

Eight shapes per dtype: 1024³, 2048³, 4096³, 1024×1024×4096, 4096×1024×1024,
transposed-A ("TN"), transposed-B ("NT") and a batched 8×512³. One-element
readback forces completion; TFLOP/s is `2·M·N·K·B / time`.

| case | device | autorust | candle | torch | tinygrad |
|---|---|---:|---:|---:|---:|
| f32 1024³ | CUDA | 5.1 | 3.1 | **6.4** | 1.3 |
| f32 2048³ | CUDA | **7.8** | 4.6 | **7.8** | 1.5 |
| f32 4096³ | CUDA | **8.0** | 2.5 | 7.7 | 1.5 |
| f32 TN 2048³ | CUDA | **9.4** | 4.8 | 8.5 | 4.9 |
| f32 batched 8×512³ | CUDA | **5.1** | 2.1 | 6.2 | 1.3 |
| bf16 2048³ | CUDA | 14.4 | — | **23.0** | — |
| bf16 4096³ | CUDA | 14.5 | — | **25.3** | — |
| bf16 TN 2048³ | CUDA | 11.3 | — | **12.8** | — |
| f32 2048³ | CPU | 0.5 | 0.6 | **0.7** | — |
| f32 4096³ | CPU | 0.4 | **0.8** | 0.6 | — |
| f32 TN 2048³ | CPU | 0.3 | **0.7** | **0.7** | — |

(TFLOP/s, higher is better.)

Reading it:

- **The f32 GEMM is at parity with cuBLAS** on CUDA at 2048³ and up (7.8-9.4 vs
  7.4-8.5 TFLOP/s), and ahead of candle and tinygrad. Only the smallest shape
  (1024³) loses to torch, where launch/tile overhead dominates. This is the
  important control for the ViT training gap: the generated tiled matmul is not
  what is 2-3x behind.
- **bf16 is 0.5-0.6x torch**: tensor cores are selected (`AUTORUST_TENSOR_CORES=0`
  drops 1024³ from 8.1 to 5.3 TFLOP/s, so the TC path is in use), but torch's
  cutlass/cuBLAS kernels reach 21-25 TFLOP/s where ours sit at 9-14. The gap is
  the TC kernel itself, not dispatch.
- On CPU the packed GEMM reaches 0.4-0.6 TFLOP/s against torch/MKL's 0.6-1.0.
  The transposed-A case is the weakest (0.3 vs 0.7): `AᵀB` needs a copy that the
  direct-operand path cannot avoid.

### Memory-bound ops

One pass over 16M f32 elements (64 MB) per op; GB/s counts the minimum traffic
(reads + writes), so a low number means the kernel is not saturating memory.

| op | device | autorust | torch |
|---|---|---:|---:|
| add | CUDA | **325** | 322 |
| copy (`x*1`) | CUDA | **318** | 310 |
| gelu | CUDA | **318** | 314 |
| softmax (4096²) | CUDA | 155 | **288** |
| layernorm (4096²) | CUDA | 156 | **209** |
| sum | CUDA | 176 | **287** |
| amax | CUDA | 64 | **292** |
| transpose | CUDA | **312** | 137 |
| add | CPU | **20.3** | 10.4 |
| copy | CPU | **18.4** | 8.2 |
| gelu | CPU | **17.3** | 8.2 |
| softmax | CPU | **8.8** | 7.2 |
| layernorm | CPU | **10.6** | 8.3 |
| sum | CPU | **28.7** | 20.8 |
| amax | CPU | 23.8 | **115.2** |
| transpose | CPU | **8.8** | 2.6 |

(GB/s.)

- Pure elementwise kernels match or beat torch on both devices (the `RUST`
  backend's vectorized kernels are 2x torch's CPU elementwise throughput).
- **Reductions are the gap on CUDA**: a full `sum`/`amax`/`softmax` runs at
  150-180 GB/s (amax at 64), because the two-pass reduce + broadcast re-reads
  the tensor; torch fuses the reduction. This matches the ViT training profile,
  which is ~60% elementwise/reduction kernels.
- On CPU, `amax` is the outlier in the other direction (torch is 5x faster):
  our max reduction is a scalar loop where torch uses a vectorized reduction.

### AdamW over 100M parameters

32 parameter tensors, gradients attached once, no `zero_grad`, 28 bytes of
traffic per parameter (grad, m, v, p reads; m, v, p writes).

| device | autorust | torch |
|---|---:|---:|
| CUDA | 102.2 ms (27.4 GB/s) | **26.3 ms (106.3 GB/s)** |
| CPU | **232.0 ms (12.1 GB/s)** | 295.8 ms (9.5 GB/s) |

On CUDA the optimizer is 4x slower than torch: the update is decomposed into
separate elementwise launches per parameter (32 tensors × several kernels), so
the 2.8 GB of traffic runs at 27 GB/s instead of near peak. This is one of the
few places where kernel launch/fusion, not bandwidth, is the limit.

## Conv models

The MNIST CNN is 28×28×1; these rows use 224² and real channel counts.
ResNet-18 trains, the others infer (batch 32; UNet batch 4 at 64²).

| workload | device | autorust | torch | ratio |
|---|---|---:|---:|---:|
| ResNet-18 infer | CPU | 1781.2 ms | **374.4 ms** | 4.8x |
| ResNet-18 infer | CUDA | 373.6 ms | **17.1 ms** | 21.9x |
| ResNet-18 train | CPU | 1177.5 ms | **1110.8 ms** | 1.06x |
| ResNet-18 train | CUDA | 441.8 ms | **64.2 ms** | 6.9x |
| ResNet-50 infer | CPU | 531.9 ms | **306.1 ms** | 1.7x |
| ResNet-50 infer | CUDA | 97.7 ms | **14.1 ms** | 6.9x |
| MobileNetV2 infer | CPU | **243.7 ms** | 837.6 ms | 0.29x |
| MobileNetV2 infer | CUDA | 54.8 ms | **22.5 ms** | 2.4x |
| UNet infer (64²) | CPU | 562.7 ms | **16.6 ms** | 33.9x |
| UNet infer (64²) | CUDA | 18.7 ms | **1.6 ms** | 11.7x |

Reading it:

- **Dense conv is the weak path.** The direct-conv kernels run at ~9-30 GFLOP/s
  (a 7×7 stem conv: 3.0 GFLOP in 0.32 s) where the GEMM microbench reaches
  8 TFLOP/s. torch uses oneDNN on CPU and cuDNN on CUDA, both far ahead.
- ResNet-18 *inference* is slower than its own training step on CPU
  (1781 vs 1177 ms) although training does more work: the inference forward
  fuses the BatchNorm into the convolution and that variant selects a much
  slower lowering than the plain conv used during training. The conv-im2col +
  batch-norm epilogue is the lever to pull.
- MobileNetV2 is the exception: depthwise convolutions have no GEMM lowering,
  and our elementwise path beats torch on CPU by 3.4x (torch's CPU depthwise
  kernel is slow at this size). CUDA is 2.4x behind.
- The UNet exposes `conv_transpose`, nearest upsample and skip `cat`, all far
  behind cuDNN/oneDNN.

## Autoregressive decoder

LLaMA-style 6-layer decoder (d=768, 12 heads, 4 KV heads, FFN 3072, vocab
32000): 512-token prefill with a causal mask, then single-token decode against
a 512-token KV cache, sampled on-device (Gumbel-max), so the step stays in the
graph.

| workload | device | autorust | torch | ratio |
|---|---|---:|---:|---:|
| prefill 512 tokens | CPU | 914.3 ms | **417.8 ms** | 2.2x |
| prefill 512 tokens | CUDA | 317.1 ms | **16.4 ms** | 19.4x |
| decode 1 token | CPU | 430.6 ms | **53.0 ms** | 8.1x |
| decode 1 token | CUDA | 17.1 ms | **3.6 ms** | 4.7x |

Decode is latency-bound and exposes a scheduling limit rather than a FLOP one:
one token launches ~300 kernels (the KV `cat`, RoPE slices, per-head attention
and the 32000-wide head), each ~50 µs of dispatch on CPU. torch runs the same
algorithm (explicit matmul + softmax, not SDPA) at 53 ms/token. The prefill gap
on CUDA is the attention matmuls at 513 keys plus the same launch count.

## Mixed precision and the ONNX runner

| workload | device | autorust | torch | ratio |
|---|---|---:|---:|---:|
| ViT-B/16 infer (bf16) | CUDA | 61.6 ms | **18.2 ms** | 3.4x |
| MNIST infer via ONNX, bs 32 | CPU | 14.7 ms | **3.7 ms** | 4.0x |
| MNIST infer via ONNX, bs 32 | CUDA | 10.9 ms | **0.35 ms** | 31x |
| ResNet-18 (w=0.25) via ONNX, bs 8 | CPU | 204.8 ms | **17.4 ms** | 11.8x |
| ResNet-18 (w=0.25) via ONNX, bs 8 | CUDA | 67.7 ms | **1.25 ms** | 54x |

- The bf16 ViT row exercises the tensor-core matmul and the bf16 attention
  path. It is 1.15x faster than f32 (61.6 vs 70.5 ms) because the token
  dimension is 197 (prime) and the TC tiling pads it; torch's bf16 forward is
  4x faster than its own f32.
- The ONNX rows load exported models (`bench/assets/mnist_bs32.onnx` and the
  width-0.25 `resnet18w25_bs8.onnx`, regenerated by
  `bench/tools/export_onnx.py`) through `OnnxModel::run`. The runner rebuilds
  and schedules the graph per call, which costs ~10-200 ms per call depending
  on the graph; torch's number is the same model executed eagerly. A frozen
  ONNX plan (the `freeze` seam) would remove the per-call scheduling.

## Frozen CPU plans

`bench/run.py --only '.*freeze'` times the `vit_freeze` example: a small ViT
(1 layer, d=192, 64²) captured once, emitted as a Rust plan by the build script,
and executed through `FrozenRunner`.

| case | device | ms |
|---|---|---:|
| eager JIT replay | CPU | 1.60 |
| frozen plan run | CPU | 2.00 |
| plan build (capture + emit) | CPU | 17.2 |

The frozen plan is *not* faster per step: both paths execute the same kernels,
and the frozen runner additionally copies inputs into the plan's storage. Its
value is deployment (no scheduler, no `rustc`, no render cache) and a
pre-compiled kernel table, documented in [`FREEZE.md`](FREEZE.md).

## Reading the numbers

- This machine occasionally carries unrelated load (load average ~11), which
  inflated one CPU run to 66 s. Re-measure before trusting a single outlier.
- ViT-B/16 CPU numbers are memory-bandwidth-bound; the GPU gap to PyTorch
  (~8.5-9.1 ms/iter) is cuBLAS/cuDNN plus tensor cores. The f32 path is
  register-tiled FFMA, and the bf16 TC path cannot tile the prime 197 dimensions
  without padding.
- cuDNN is intentionally not wired up: the demos' only convolution (ViT patch
  embed) is a negligible fraction of the forward, and `HostGemm` is the seam a
  library conv/attention op would use.
