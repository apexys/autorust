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
- `TinyJit` capture/replay, BEAM search, ONNX import and frozen CPU plans.
