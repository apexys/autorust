# autorust

A multi-backend GPU-compute / ML framework!
Train and infer on CUDA, OpenCL, Metal or CPU!
Inspired and based on tinygrad!

It includes:
- A `Tensor` type for dynamic multi-dimension, multi-datatype arrays
- A `nn` library of common ML layers (Linear, Conv, LSTMCell, ...) and Optimizers (Adam, SGD, LARS, ...)
- A dynamic graph compiler/scheduler (adapted from tinygrad's) with codegen
- GPU backends for CUDA, OpenCL and Metal
- "Just-in-time" CPU backends for Rust and C-compilers and an interpreted RAYON backend
- AOT CPU compilation (embed ML models as Rust code with a build script)
- Accelerated GEMM and fused FlashAttention kernels (including TF32/bf16
  tensor-core variants)
- Safetensors checkpoint loading and ONNX model import

This started as a somewhat faithful automated ("vibe-coded") Rust port of [tinygrad](https://github.com/tinygrad/tinygrad) v0.14 and now has gained a few more features. This README's introduction is the only thing I really wrote by hand, the rest was mostly done using Qwen 3.8 27b and DeepSeek v4.1 Flash, both hosted locally (no Claude or OpenAI). I have tested this extensively and am using this in production, but there might be rough edges.

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
- `xor.rs` Train a dense NN to learn the XOR function
- `frozen_vit` Include a ViT-B/16 as precompiled Rust-code in your application

## Dependencies

I've kept the dependency list deliberately short. `nanoserde` (graph
serialization) is the only mandatory dependency. `rayon` + `bytemuck` power the
`RAYON` backend and the packed CPU GEMM; the `RUST`, `CC`, `CL` and `CUDA`
backends use `libloading` for the libraries they generate or load at run time.
`CUDA` also uses `cudarc`, dynamically loaded, so no CUDA toolkit is needed on
the build machine, and `METAL` uses the `objc2` crates.

If you want to use the `RUST` backend, you need `rustc` installed, same with a c compiler for the `CC` backend. Both fall back to the multithreaded `RAYON` backend if not available.

## Status

Basic inference and training work. I am using this as a compute layer for
several applications at the GEOMAR Helmholtz Centre for Ocean Research right
now. The ergonomics are fine, but I am still looking at a few places where they
could be improved. Scheduling is single-threaded for now and could be faster
(kernel compilation is parallel where the backend allows it). There's also a
lot of optimization potential in the backends (and I'd like to add a Vulkan
one). ONNX is relatively incomplete and I'd like to add an exporter as well.

## Performance

`autorust` is pretty okay right now (fast enough to be usable).
We integrate with native GEMM libraries when available (cuBLAS, CLBlast,
MetalPerformanceShaders, plus our own packed CPU kernel) and have fast paths
for attention and, to a lesser extent, convolutions.

Here's a benchmark (see `bench`) comparing the performance across workloads and frameworks:

(CPU: Ryzen 9 3900X, 64GB of DDR4 RAM, RTX 3060 12GB, Ubuntu 26.04)

See [`PERFORMANCE.md`](PERFORMANCE.md) for methodology and tuning.

| workload | device | autorust | candle | torch | tinygrad |
|---|---|---:|---:|---:|---:|
| MNIST CNN train, bs 512 | CPU | **64.2 ms** (RUST) | 3902.0 ms | 175.9 ms | 3464.8 ms |
| MNIST CNN train, bs 512 | CUDA | 41.0 ms | 131.9 ms | **13.1 ms** | 56.8 ms |
| ViT-B/16 infer, bs 1 | CPU | 438.1 ms (RUST) | 229.4 ms | **165.1 ms** | 14639.3 ms |
| ViT-B/16 infer, bs 1 | CUDA | 20.5 ms | 19.2 ms | **8.7 ms** | 116.9 ms |
| ViT-B/16 infer, bs 8 | CPU | 1861.9 ms (RUST) | 1403.5 ms | **701.5 ms** | — |
| ViT-B/16 infer, bs 8 | CUDA | 207.0 ms | 114.6 ms | **46.4 ms** | 136.2 ms |
| ViT-B/16 train, bs 1 | CPU | 2230.5 ms (RUST) | **243.4 ms** | 616.3 ms | — |
| ViT-B/16 train, bs 1 | CUDA | 300.2 ms | **19.0 ms** | 43.8 ms | 388.5 ms |


## Scope and limitations

- ONNX covers the T0/T1 inference surface (see
  [`ONNX-Compat.md`](ONNX-Compat.md)); quantization, training ops and control
  flow are not supported.
- `METAL` has no hardware `double`: f64 is lowered to f32 with a one-time
  warning. Use `INTERP`/`CPU`/`CL`/`CUDA` when true f64 is required.
- The front-end is single-threaded (`Tensor` is `Rc`-based and `!Send`); the CPU
  backends parallelize their kernels internally.
- cuDNN is not wired up: convolutions use generated kernels and attention uses
  the fused hand-written kernels.
- CUDA tensor cores target `sm_80+`; older cards fall back to the scalar path.
- The `RUST` CPU backend is x86-64-first; other architectures fall back to libm
  math.

## Further Reading

(all of these documents are LLM-written)

- [`CONFIGURATION.md`](CONFIGURATION.md) — environment variables and settings.
- [`PERFORMANCE.md`](PERFORMANCE.md) — measurements, kernels and device
  selection.
- [`FREEZE.md`](FREEZE.md) — embedding a "precompiled" model in an application.
- [`ONNX-Compat.md`](ONNX-Compat.md) — ONNX operator coverage and runner API.
- [`INDEXING.md`](INDEXING.md) — tensor indexing, especially with Rust types.
- [`CURRENT_STATE.md`](CURRENT_STATE.md) — canonical status and known gaps.
- [`CHANGELOG.md`](CHANGELOG.md) — release notes.
- [`AGENTS.md`](AGENTS.md) — how the LLM works on this codebase.

## License

MIT. This is a derivative work of tinygrad (MIT); see [`LICENSE`](LICENSE).
