# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-11-11

Performance and correctness release; no API changes.

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
