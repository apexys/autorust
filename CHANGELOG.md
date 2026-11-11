# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- GPU GEMM register tiling that keeps prime row extents (e.g. ViT at batch 8)
  on multi-thread tiles, scores both contraction orientations, and the direct
  conv tile reuses weights across two spatial rows.
- Persistent rendered kernels: a warm process skips the codegen pipeline.
- `ViT::forward_with` / `TransformerBlock::forward_with`: eval-mode inference
  dispatches to the fused attention kernels.
- Summed gradient contractions (e.g. `dQ@Wq + dK@Wk + dV@Wv`) are materialized
  per contraction so each reaches the matmul tiler instead of one scalar
  multi-reduce kernel.
- Pointwise contraction operands inside reduce bodies (GELU derivative,
  dropout mask, softmax backward) are staged once, so those backward GEMMs
  become affine and can use the matmul tiler/library GEMM.
- CPU host GEMM reads a row-major `A` directly and packs a `[n, k]` weight once
  per GEMM instead of per tile (1.2-1.5x on the larger ViT linears), and a
  batched GEMM packs an unbatched weight once per call instead of per batch.
- The scheduler materializes interior host-backend contractions (the FFN
  expansion under GELU) so the packed CPU GEMM takes the matmul and the
  epilogue runs as its own kernel, and stages the pointwise operands of big
  contractions (attention's biased Q/K): RUST ViT-B/16 inference 425 -> 173 ms
  at bs 1 (1.05x torch), 1902 -> 1164 ms at bs 8 (1.66x), training 1784 ->
  1634 ms.
- `TinyJit` capture/replay, BEAM search, ONNX import and frozen CPU plans.
