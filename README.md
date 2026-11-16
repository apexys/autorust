# autorust

A multi-backend GPU-compute / ML framework!
Train and infer on CUDA, Vulkan, OpenCL, Metal or CPU!
Inspired and based on tinygrad!

It includes:
- A `Tensor` type for dynamic multi-dimension, multi-datatype arrays
- A `nn` library of common ML layers (Linear, Conv, LSTMCell, ...) and Optimizers (Adam, SGD, LARS, ...)
- A dynamic graph compiler/scheduler (adapted from tinygrad's) with codegen
- GPU backends for CUDA, Vulkan, OpenCL and Metal (Vulkan emits SPIR-V directly)
- "Just-in-time" CPU backends for Rust and C-compilers and an interpreted RAYON backend
- AOT CPU compilation (embed ML models as Rust code with a build script)
- Accelerated GEMM and fused FlashAttention kernels (including TF32/bf16
  tensor-core variants)
- Safetensors checkpoint loading, ONNX model import and ONNX export (E0-E2)
- Classical ML (C0/C1/C2a): linear/ridge and logistic regression, PCA, k-means,
  exact kNN, linear SVM/SVR, decision trees, random forests, standard scaling
  and metrics

This started as a somewhat faithful automated ("vibe-coded") Rust port of [tinygrad](https://github.com/tinygrad/tinygrad) v0.14 and now has gained a few more features. This README's introduction is the only thing I mostly wrote by hand, the rest was mostly done using Qwen 3.8 27b and DeepSeek v4.1 Flash, both hosted locally (no Claude or OpenAI). I have tested this extensively and am using this in production, but there might be rough edges.

## Quick start

```rust
use autorust::engine::session::Session;
use autorust::tensor::{Ctx, Tensor};

let ctx = Ctx::new(); // Contexts hold your compute graph
let mut sess = Session::new(); // Sessions hold buffers
let a = Tensor::from_slice(&ctx, &[1.0f32, 2.0, 3.0], None, None); //Create tensor on default device
let b = Tensor::from_slice(&ctx, &[10.0f32, 20.0, 30.0], None, None);
let c = &a + &b; // or a.add(&b)
assert_eq!(c.to_list::<f32>(&mut sess), vec![11.0, 22.0, 33.0]); //to_list allocates a new buffer (or reuses one from your session), then triggers the computation
```

Also look at our examples, especially:
- [`xor.rs`](https://github.com/apexys/autorust/blob/main/examples/xor.rs) —
  train a dense NN to learn the XOR function
- [`frozen_vit`](https://github.com/apexys/autorust/tree/main/examples/frozen_vit) —
  include a ViT-B/16 as precompiled Rust code in your application (a standalone
  crate, so it lives in the repository rather than the published package)

## Dependencies

I've kept the dependency list deliberately short. `nanoserde` (graph
serialization) is the only mandatory dependency. `rayon` + `bytemuck` power the
`RAYON` backend and the packed CPU GEMM; the `RUST`, `CC`, `CL` and `CUDA`
backends use `libloading` for the libraries they generate or load at run time.
`CUDA` also uses `cudarc`, dynamically loaded, so no CUDA toolkit is needed on
the build machine, `METAL` uses the `objc2` crates, and `VK` uses [`ash`](https://crates.io/crates/ash)
purely for its bindings (its `loaded` feature `dlopen`s `libvulkan` at run
time, so nothing links against the loader).

If you want to use the `RUST` backend, you need `rustc` installed, same with a c compiler for the `CC` backend. Both fall back to the multithreaded `RAYON` backend if not available.

## Status

Basic inference and training work. I am using this as a compute layer for
several applications at the GEOMAR Helmholtz Centre for Ocean Research right
now. The ergonomics are fine, but I am still looking at a few places where they
could be improved. Scheduling is single-threaded for now and could be faster
(kernel compilation is parallel where the backend allows it). There's also a
lot of optimization potential in the backends. ONNX import is relatively incomplete; ONNX export covers E0-E2. Classical ML covers C0/C1/C2a (linear models, PCA, k-means, exact kNN, linear SVM/SVR, decision trees and random forests); GBT and the wider classic stack are planned.

## Performance

`autorust` is pretty okay right now (fast enough to be usable).
We integrate with native GEMM libraries when available (cuBLAS, CLBlast,
MetalPerformanceShaders, the Vulkan cooperative-matrix path, plus our own packed
CPU kernel) and have fast paths for attention and, to a lesser extent,
convolutions.

Here's a benchmark ([`bench`](https://github.com/apexys/autorust/tree/main/bench)) comparing the performance across workloads and frameworks:

(CPU: Ryzen 9 3900X, 64GB of DDR4 RAM, RTX 3060 12GB, Ubuntu 26.04)

The table is **Exact-mode** (`AUTORUST_MATH=exact`) to keep the arithmetic
comparable; the default is now Fast (`AUTORUST_MATH=fast`), which uses TF32
tensor cores and measured 55% faster ResNet-18 inference, 60% faster ResNet-50
and 21% faster ViT
inference. See [`PERFORMANCE.md`](https://github.com/apexys/autorust/blob/main/PERFORMANCE.md#math-mode) for methodology and tuning.

| workload | device | autorust | burn | candle | torch | tinygrad |
|---|---|---:|---:|---:|---:|---:|
| MNIST CNN train, bs 64 | CPU | 19.0 ms (RUST) | 76.6 ms (Flex) | 434.6 ms | **17.5 ms** | 404.4 ms |
| MNIST CNN train, bs 64 | CUDA | 3.5 ms | 4.3 ms | 17.4 ms | **2.3 ms** | 8.4 ms |
| MNIST CNN train, bs 512 | CPU | **63.2 ms** (RUST) | 611.6 ms (Flex) | 3902.0 ms | 175.9 ms | 3464.8 ms |
| MNIST CNN train, bs 512 | CUDA | 22.4 ms | 32.0 ms | 131.9 ms | **13.1 ms** | 56.8 ms |
| ViT-B/16 infer, bs 1 | CPU | **156.4 ms** (RUST) | 195.7 ms (Flex) | 229.4 ms | 165.1 ms | 14639.3 ms |
| ViT-B/16 infer, bs 1 | CUDA | 14.2 ms | 9.4 ms | 19.2 ms | **8.7 ms** | 116.9 ms |
| ViT-B/16 infer, bs 8 | CPU | 1103.1 ms (RUST) | 1194.6 ms (Flex) | 1403.5 ms | **826.9 ms** | — |
| ViT-B/16 infer, bs 8 | CUDA | 69.8 ms | 52.1 ms | 114.6 ms | **48.6 ms** | 136.2 ms |
| ViT-B/16 train, bs 1 | CPU | 1634.5 ms (RUST) | 1048.4 ms (Flex) | **243.4 ms** | 616.3 ms | — |
| ViT-B/16 train, bs 1 | CUDA | 95.9 ms | 36.8 ms | **19.0 ms** | 43.8 ms | 388.5 ms |

Burn's f32 CUDA matmul stages operands as TF32, while the autorust/candle/torch
rows are strict f32; the Burn CPU rows are strict f32. With that caveat, Burn is
2.4-2.6x faster than autorust on CUDA ViT training and within 7% of torch at
bs 8 inference; candle remains the fastest ViT-training framework. autorust
keeps the CPU MNIST and inference lead; on CPU ViT training burn is 1.6x ahead
of autorust, torch 2.7x and candle 6.7x. ViT training remains the largest gap
to candle and torch: at bs 1 autorust/CUDA is 95.9 ms vs torch's 43.8 ms, and at
bs 4 it is 216.7 ms vs 100.4 ms. The remaining cost is steady-state f32
GEMM/elementwise throughput rather than launch overhead. MNIST scales the other
way: autorust is 2.8x faster than torch on CPU at bs 512 and within 1.1x at
bs 64. Full sweep in
[`PERFORMANCE.md`](https://github.com/apexys/autorust/blob/main/PERFORMANCE.md#batch-size-scaling).

### More workloads

The bench also isolates the pieces the two models hide:

| workload | device | autorust | torch | note |
|---|---|---:|---:|---|
| f32 GEMM, 2048³ | CUDA | 2.22 ms (7.7 TF/s) | **2.21 ms** (7.8 TF/s) | at parity with cuBLAS |
| bf16 GEMM, 4096³ | CUDA | 9.50 ms (14.5 TF/s) | **5.43 ms** (25.3 TF/s) | tensor cores selected, kernel ~0.6x |
| memory-bound add, 64 MB | CUDA | **0.62 ms** (325 GB/s) | 0.63 ms (322 GB/s) | elementwise at peak |
| memory-bound softmax | CUDA | 0.87 ms (154 GB/s) | **0.47 ms** (287 GB/s) | two-pass reduce |
| AdamW, 100M params | CUDA | 25.7 ms | **25.4 ms** | fused kernel, parity with torch |
| ResNet-18 infer, bs 32 | CUDA | 322.7 ms | **18.8 ms** | conv lowering is the gap |
| ResNet-18 train, bs 32 | CPU | 1177.5 ms | **1110.8 ms** | near parity on CPU |
| MobileNetV2 infer, bs 32 | CPU | **243.7 ms** | 837.6 ms | depthwise beats oneDNN |
| UNet infer, 64² | CUDA | 18.7 ms | **1.6 ms** | conv_transpose/upsample |
| LLM prefill, 512 tok | CUDA | 317.1 ms | **16.4 ms** | attention + head |
| LLM decode, 1 tok | CUDA | 17.1 ms | **3.6 ms** | ~300 kernels/token |
| ViT-B/16 infer, bf16 | CUDA | 61.6 ms | **18.2 ms** | TC tiling on 197 tokens |
| ViT-B/16 train, bf16 | CUDA | 84.3 ms | **20.9 ms** | f32 was 95.9 vs 43.8 |
| MNIST infer via ONNX, bs 32 | CUDA | 1.80 ms | **0.34 ms** | runner replays one plan per shape |
| ResNet-18 (w=0.25) via ONNX, bs 8 | CUDA | 9.68 ms | **1.81 ms** | same, larger graph |

Vulkan (`VK`) emits SPIR-V directly and runs the same kernels as OpenCL: all
eight memory-bound microbenches are faster (softmax 0.89 vs 1.39 ms, layernorm
0.90 vs 1.43 ms, add 0.64 vs 1.05 ms), ViT-B/16 inference at bs 8 is
**116.6 vs 179.5 ms**, ResNet-18 inference **319.7 vs 376.3 ms**, UNet 16.4 vs
19.0 ms, AdamW 94.5 vs 96.3 ms, and MNIST train 21.1 vs 21.0 ms. bf16 remains
the gap (ViT-B/16 bf16 113.4 vs 90.9 ms); MobileNetV2 (61.4 vs 56.1 ms) and LLM
prefill (421.7 vs 394.3 ms) are the other rows more than 6% behind. See
[`PERFORMANCE.md`](https://github.com/apexys/autorust/blob/main/PERFORMANCE.md#vulkan-vk).

Per-case tables (GEMM shapes, each memory op, both optimizer devices, the
frozen-plan comparison) are in
[`bench/results/summary.md`](https://github.com/apexys/autorust/blob/main/bench/results/summary.md)
and [`PERFORMANCE.md`](https://github.com/apexys/autorust/blob/main/PERFORMANCE.md#microbenches).

## Scope and limitations

- ONNX covers the T0/T1 inference surface (see
  [`ONNX-Compat.md`](https://github.com/apexys/autorust/blob/main/ONNX-Compat.md)); quantization, training ops and control
  flow are not supported.
- `METAL` has no hardware `double`: f64 is lowered to f32 with a one-time
  warning. Use `INTERP`/`CPU`/`CL`/`CUDA` when true f64 is required.
- The front-end is single-threaded (`Tensor` is `Rc`-based and `!Send`); the CPU
  backends parallelize their kernels internally.
- cuDNN is not wired up: convolutions use generated kernels plus an im2col +
  host-GEMM path where the contraction is long enough, and attention uses the
  fused hand-written kernels.
- CUDA tensor cores target `sm_80+`; older cards fall back to the scalar path.
- The `RUST` CPU backend is x86-64-first; other architectures fall back to libm
  math.

## Further Reading

(all of these documents are LLM-written)

- [`CONFIGURATION.md`](https://github.com/apexys/autorust/blob/main/CONFIGURATION.md) — environment variables and settings.
- [`PERFORMANCE.md`](https://github.com/apexys/autorust/blob/main/PERFORMANCE.md) — measurements, kernels and device
  selection.
- [`FREEZE.md`](https://github.com/apexys/autorust/blob/main/FREEZE.md) — embedding a "precompiled" model in an application.
- [`ONNX-Compat.md`](https://github.com/apexys/autorust/blob/main/ONNX-Compat.md) — ONNX operator coverage and runner API.
- [`ONNX-Export.md`](https://github.com/apexys/autorust/blob/main/ONNX-Export.md) — ONNX export contract and roadmap.
- [`CLASSIC-ML.md`](https://github.com/apexys/autorust/blob/main/CLASSIC-ML.md) — classical-ML scope, algorithms and roadmap.
- [`INDEXING.md`](https://github.com/apexys/autorust/blob/main/INDEXING.md) — tensor indexing, especially with Rust types.
- [`CURRENT_STATE.md`](https://github.com/apexys/autorust/blob/main/CURRENT_STATE.md) — canonical status and known gaps.
- [`CHANGELOG.md`](https://github.com/apexys/autorust/blob/main/CHANGELOG.md) — release notes.
- [`AGENTS.md`](https://github.com/apexys/autorust/blob/main/AGENTS.md) — how the LLM works on this codebase.

## License

MIT. This is a derivative work of tinygrad (MIT); see [`LICENSE`](https://github.com/apexys/autorust/blob/main/LICENSE).
