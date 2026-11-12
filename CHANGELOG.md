# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Cooperative-matrix ("tensor core") probing for the Vulkan backend.** The
  device probe now enumerates `VkPhysicalDeviceCooperativeMatrixPropertiesKHR`
  and enables `VK_KHR_cooperative_matrix` only when a 16-bit-in/f32-out shape
  exists (`DeviceInfo::coop`). `tests/vk_coopmat.rs` runs a hand-written
  `OpCooperativeMatrixMulAddKHR` GEMM and pins both numerics (exact against an
  f64 reference) and throughput: **7.7 TF/s at 1024³ and 9.8 TF/s at 2048³,
  3.2x / 2.5x the plain-SPIR-V GEMM**. The renderer does not emit it yet — the
  integration plan and the numerics trade-off are in `CURRENT_STATE.md` §9.53.
- **Vulkan backend (`VK`, `backend_vk`, default).** A fourth GPU backend that
  needs no vendor SDK: kernels are emitted as **SPIR-V directly** from the
  linear UOp list (`renderer::spirv`) instead of C text, and `libvulkan` is
  loaded at run time through `ash`'s `loaded` feature (bindings only; nothing
  links against the loader). Machines without a loader, a compute device, or
  `storageBuffer8BitAccess` simply do not register `VK`, like a box without
  CUDA. Supports f32/f16/bf16 (bf16 as a 16-bit storage pattern widened to f32),
  all integer widths, vectors, local memory + barriers, and `VK:N` device
  selection. The same scheduler/tuning is reused, so performance lands within
  ~20% of the OpenCL backend on most workloads (faster on elementwise, `add`,
  `gelu`, TN/NT GEMM and AdamW; slower on `LOCAL`-memory reductions and on
  decode-heavy graphs, which pay one command buffer per kernel — see
  `PERFORMANCE.md`). The emitter unrolls small constant-trip-count loops 8x
  because the NVIDIA SPIR-V path does not (its OpenCL compiler does): that alone
  took `softmax`/`layernorm` from 2.1/2.2 ms to 1.09/1.19 ms, ahead of OpenCL's
  1.38/1.42, and consecutive launches are batched into one command buffer (closed
  at each executed linear via the new `Prog::flush` seam) which took LLM decode
  from 25.6 to 21.3 ms. Register arrays (GEMM accumulators, the per-lane tiles in
  the conv kernels) are promoted to SSA with loop-carried `OpPhi`s instead of
  round-tripping through `Function` memory, worth 20% on ResNet-18 inference
  (403 -> 321 ms) and 14% on UNet; the loop carry has to be read out of the
  continue block, which `tests/vk_backend.rs` pins (using the header phi dropped
  the last iteration of every reduction).
- **Backend memory reporting.** `Compiled::memory_info()` returns
  `MemoryInfo { total, free }`, implemented through
  `BackendAllocator::memory_info()`: CUDA uses `cuMemGetInfo_v2`, OpenCL reports
  `CL_DEVICE_GLOBAL_MEM_SIZE`, Metal reports its recommended working set, and
  host memory reports `None`.

### Fixed

- **CUDA NVRTC fallback.** The device probe now tries every installed NVRTC
  newest-first and pins the first whose PTX the driver actually loads, instead
  of disabling CUDA when the newest toolkit is newer than the driver. An
  explicit `AUTORUST_NVRTC_LIB`/`NVRTC_PATH` still pins one library.

## [0.1.1] - 2026-11-11

Performance and correctness release; no API changes. Net effect on the
benchmark suite: ViT-B/16 training CUDA 332 -> 96 ms, CPU inference 425 -> 171 ms
at bs 1 and 1902 -> 1188 ms at bs 8, CPU training 1784 -> 1634 ms; MNIST CUDA
39 -> 22 ms. Six of the eight original bench rows are now within 2x of eager
torch (`torch.compile` off), and warm cold starts drop from 505/697/1279/6013 ms
to 216/474/760/2498 ms. The ViT training gap is batch-dependent rather than
fixed overhead: the generated f32 GEMM is close to saturation at batch 1
(per-image 98.8 -> 78.6 ms on CUDA through bs 8) while cuBLAS keeps improving
(45.8 -> 22.1 ms/image), so at batch 4 the step is 317 vs 102 ms; at batch 64
MNIST is 1.3x off torch on CUDA and 2x faster on CPU. The benchmark matrix also
grew beyond the two models: a GEMM sweep (f32/bf16, transposed and batched), a
memory-bound op set, an AdamW-only step, ResNet-18/50, MobileNetV2, a small
UNet, an autoregressive prefill/decode pair (GQA, RoPE, KV cache, sampling), a
bf16 ViT rows (inference and training), the ONNX runner (MNIST and a
width-0.25 ResNet-18) and a frozen-plan comparison (dev-only; see
`PERFORMANCE.md`).

### Performance

- **Persistent rendered kernels.** The render result (ProgramInfo, ELF
  signature, source) is cached under the AST content key plus the codegen
  settings and a fingerprint of the running binary, so a warm process skips the
  ~30 codegen pipeline stages per kernel. ViT-B/16 training cold start
  7.2 s -> 2.5 s; ViT infer bs8 1.26 s -> 0.49 s. `CACHELEVEL=0` disables it.
- **GPU GEMM tiling.** Both contraction orientations are scored by block
  threads (with a register/thread trade below one warp), and the direct-conv
  tile reuses the 5x5 weights across two output rows. ViT-B/16 training CUDA
  332 -> 216 ms, MNIST bs512 39 -> 22 ms.
- **Summed gradient contractions** (`dQ@Wq + dK@Wk + dV@Wv`) are materialized
  per contraction so each reaches the matmul tiler instead of one scalar
  multi-reduce kernel (ViT training 216 -> 179 ms).
- **Pointwise contraction operands inside reduce bodies** (GELU derivative,
  dropout mask, softmax backward) are staged once, making those backward GEMMs
  affine (179 -> 94 ms).
- **CPU packed GEMM coverage.** Interior host-backend contractions (the FFN
  expansion, attention's biased Q/K) are split and their operands staged so the
  packed CPU GEMM takes the matmul and the epilogue runs as its own kernel.
  RUST ViT-B/16 inference 425 -> 171 ms at bs 1, 1902 -> 1188 ms at bs 8;
  training 1784 -> 1634 ms.
- **CPU host GEMM** reads a row-major `A` directly and packs a `[n, k]` weight
  once per GEMM (and once per batched call), 315 -> 464 GFLOP/s on
  197x768x3072.
- **Fused attention for eval-mode inference**: `ViT::forward_with` /
  `TransformerBlock::forward_with` dispatch to the FA2 kernels on CUDA/CL/Metal
  (CPU and training keep SDPA), with the q/k/v projections co-realized and
  pinned.
- `AUTORUST_PROF` per-stage codegen timings, `AUTORUST_KPROF` per-program kernel
  timings for host backends, and `AUTORUST_MM_DEBUG` matmul/`HostGemm`
  rejection reasons.

### Fixed

- The graph model (`TransformerBlock::attn`) no longer routes through the
  op-level FlashAttention port by default: the default `Settings::attention`
  selects the fused tensor-level kernels, and the 16x-slower op-level port is
  now only used with `FA=1`. The default follows tinygrad's model and uses SDPA.
- CUDA bf16 kernels whose lanes vectorized 2-wide failed to compile:
  `cuda_bf16.h` typedefs `nv_bfloat162` but has no `make_nv_bfloat162`, so the
  generated constructor was undefined. The renderer now defines it via
  `__halves2bfloat162`; bf16 elementwise kernels (and the benchmark's bf16 GEMM
  sweep) compile and run on CUDA.
- The `RUST` renderer keeps rustc's loop vectorizer off integer shift chains
  (the `threefry` RNG kernels): the vectorizer is exponential in the chain
  length (20 shifts 0.26 s, 24 1.8 s, 28 16 s, the RNG kernel 80+), so a cold
  batch-2 training run could spend over ten minutes in `rustc` on one kernel.
  The loop index goes through `core::hint::black_box` for those kernels only;
  every other kernel keeps its vectorized work-item loop.
- The runtime cache is keyed by the buffer signature as well as the compiled
  artifact: a cached runtime could be reused for an artifact with different
  slots, launching with an out-of-bounds buffer index.
- `ProgramInfo` is derived from the linearized body, the same traversal the ELF
  signature uses (the linearize cleanups can rewrite nodes).
- The realize decision and the `non_removable` mark survive the reduce rebuild
  in `stage_reduce_prologue` (a rebuilt reduce otherwise lost its boundary).
- Optimizer moments are built as store-free buffer leaves, as
  `zeros_param`'s contract states (they previously carried a `full` store that
  every realize re-executed).

## [0.1.0] - 2026-11-10

Initial release.

- `Tensor` front-end, autograd, `nn` layers and optimizers, RNG and state
  (safetensors).
- Interned UOp graph, pattern matcher, rangeify scheduler, codegen and the
  reference interpreter.
- Backends: `INTERP`, `RAYON`, `RUST`, `CC`, `CL`, `CUDA` and `METAL`.
- Dynamically loaded cuBLAS / CLBlast / MetalPerformanceShaders host GEMM, a
  packed CPU GEMM, and fused FlashAttention kernels (scalar, TF32 and bf16
  tensor cores).
- `TinyJit` capture/replay, BEAM search, ONNX import and frozen CPU plans.
