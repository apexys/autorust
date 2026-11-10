# autorust

A faithful Rust port of [tinygrad](https://github.com/tinygrad/tinygrad) v0.14:
an interned UOp graph, a rangeify scheduler, a codegen pipeline, and native
CPU/OpenCL/CUDA/Metal backends. The Python tree is the reference oracle; Rust
never imports Python at runtime.

This README is a working draft. The maintainer will rewrite it; the technical
statements below are the ones that matter.

## Status

The core engine is complete and differentially tested against tinygrad:

- UOp graph, pattern matcher, symbolic ranges, dtypes, Tensor front-end,
  autograd, random, `nn` layers, and `nn/optim.py` optimizers.
- Scheduler (`rangeify`/`indexing`/memory planning), `to_program` codegen,
  C-style rendering, and the uop interpreter.
- Backends: `INTERP` (faithful reference), `RAYON` (multithreaded CPU),
  `RUST` (native CPU: Rust kernels compiled by `rustc`), `CC` (native CPU:
  portable C compiled by clang/gcc), `CL` (OpenCL), `CUDA` (driver API +
  NVRTC, bf16/WMMA), and `METAL` (Apple Metal on macOS).
- Frozen CPU plans: capture a `Jit` step, lower it to a `FrozenPlan`, and emit
  a self-contained Rust file for `build.rs`/`include!` (no runtime `rustc`);
  `Engine` selects eager GPU or frozen CPU automatically.
- `TinyJit`-style capture/replay, BEAM search, and a content-keyed program cache.

The public API is **0.1.0 and unstable**. A `Tensor` owns a shared `Ctx`
handle, so graph operations do not take a context argument; safetensors and
`load_state_dict` are implemented (including non-trainable buffers). See
[`CURRENT_STATE.md`](https://github.com/apexys/autorust/blob/main/CURRENT_STATE.md)
for the full status and known gaps.

## Quick start

```sh
cargo run --release --example xor
```

```rust
use autorust::engine::session::Session;
use autorust::tensor::{Ctx, Tensor};

let ctx = Ctx::new();
let mut sess = Session::new();
let a = Tensor::from_slice(&ctx, &[1.0f32, 2.0, 3.0], None, None);
let b = Tensor::from_slice(&ctx, &[10.0f32, 20.0, 30.0], None, None);
let c = &a + &b; // or a.add(&b)
assert_eq!(c.to_list::<f32>(&mut sess), vec![11.0, 22.0, 33.0]);
```

Weights can be streamed from safetensors without loading the whole checkpoint:

```rust,ignore
let state = autorust::tensor::state::safe_load("model.safetensors")?;
autorust::tensor::state::load_state_dict(&model, &state, true)?;
```

`Tensor::from_read(&ctx, &mut reader, dtype, shape, device)` is the underlying
one-tensor-at-a-time primitive.

ONNX inference is available for the T0/T1 surface (std-only protobuf parser,
external data, static/runtime-bound shapes, CNN/transformer ops):

```rust,ignore
use autorust::onnx::OnnxModel;
use autorust::tensor::Tensor;

let mut model = OnnxModel::load("model.onnx")?;
let image = Tensor::zeros(model.ctx(), [1, 3, 518, 518]);
let out = model.run(&[("image", image)])?;
let depth = out["depth"].to_list::<f32>(model.session_mut());
```

See [`ONNX-Compat.md`](ONNX-Compat.md) for the supported operator matrix and
known gaps (no `Scatter*`/`NonZero`, quantized/training ops are T2).

Runnable demos: `examples/xor.rs`, `examples/mnist.rs`,
`examples/vit_mnist.rs`, plus `examples/fa_bench.rs`,
`examples/rayon_bench.rs`, `examples/gemm_bench.rs`, `examples/vit_bench.rs`,
and `examples/vit_train_bench.rs` for attention/CPU/GPU/GEMM/training
benchmarks.

## Performance

Matmul-shaped contractions (`MUL -> REDUCE(ADD)`) are register-blocked by the
scheduler: `RUST`/`CC` use an 8x8 output tile with a 4-wide K unroll; CUDA/CL
use `LOCAL`/`UPCAST` tiles (4x4 or 8x8, chosen by output size). The largest
grid dimension is placed on `blockIdx.x` so CUDA runs all M blocks for one N
weight tile before moving on (L2 reuse). Tuning overrides:
`AUTORUST_MM_{M,N,K,LM,LN}`.

A pure contraction can also run through the **`HostGemm` host path**. The
scheduler describes it once (shapes, strides, dtype, alpha and a fused
`bias + per-column scale + residual` epilogue) and each backend executes it with
its own fastest implementation:

- **CUDA → cuBLAS** (`cublasSgemm_v2` / `cublasGemmEx`, f32/f16/bf16,
  strided-batched, in-place epilogue kernel).
- **CL → CLBlast** (`CLBlastSgemm` / `CLBlastSgemmStridedBatched`, f32,
  row-major). Install `libclblast1` (`apt install libclblast1`); without it the
  OpenCL backend simply keeps using generated kernels.
- **CPU → packed blocked kernel** (`RUST`/`CC`, f32). Rayon over disjoint
  `(m,n)` tiles, both operands packed into microkernel panels, a 6x16 AVX2/FMA
  microkernel (12 YMM accumulators) and a scalar fringe so prime extents need no
  padding or masks. This is our own kernel, not a library, but it goes through
  the same descriptor and epilogue. It is on by default because the generated
  CPU kernel tiles one output row and re-streams `B` for every row: 789x384x384
  went from 16.7 ms to 1.2 ms (14 → 200 GFLOP/s) on a 3900X.

### Hand-written kernels

`Compiled::run_host_kernel` is the escape hatch for kernels the uop IR cannot
express — a fused kernel needs running per-thread state (flash attention's
online softmax), which the IR has no way to carry. It takes source in the
backend's own language, compiles it through the device's existing compiler
(NVRTC on CUDA, `clBuildProgram` on OpenCL, rustc on `RUST`) and launches it
through the device's existing runtime, so buffers bind by slot and scalar args
follow the generated-kernel ABI. `tests/host_kernel.rs` covers the CUDA path
end to end.

`runtime::cuda_flash` is the first user: an FA2 kernel (64-query tiles, 32-key
streaming steps, online softmax, D=64) that keeps the score matrix in
registers/shared. For DINOv3's attention (6 heads, 789 tokens) it runs one
layer in 1.22 ms at batch 1 and 5.85 ms at batch 8 (0.73 ms/image/layer),
against 1.15 ms/image for the generated softmax·V *plus* 0.33 ms for softmax
and a separate cuBLAS QK^T. It matches `scaled_dot_product_attention` to
1.2e-8.

`runtime::cl_flash` is the same schedule in OpenCL C (sequence length baked into
the source so the launch is four buffers on every backend). It matches CL SDPA
to 1.2e-8 and runs 1.32 ms at batch 1 / 8.33 ms at batch 8 (1.04 ms/image/layer,
vs 0.73 on CUDA) — so the OpenCL backend gets the same win once it is wired in.

This is the prerequisite for a real fused attention: the graph-level port in
`models::flash_attention` is 16x *slower* than the current fused softmax·V on
CUDA (105.7 s vs 6.4 s per 128 frames at 448px) because every tile boundary
becomes a materialized kernel, which is exactly why tinygrad's FA is a
`@custom_kernel` over its warp DSL rather than a graph.

Attention's `P·V` reaches the same path when the caller uses
`Tensor::scaled_dot_product_attention_sess`: the score matrix is materialized so
the contraction has a buffer to describe. That is a CPU win (~150 ms -> ~7 ms per
image for the 12 AV products of a ViT-S/16) but a GPU loss, because `P` is
`batch*heads*t*t` and writing/re-reading it costs more bandwidth than fusing
softmax into the AV kernel saves. The choice belongs to the caller, which knows
the device.

This is what makes the prime 197-token ViT sequence fast: the register tile
cannot vectorize 197, while the library can. On CUDA/OpenCL the library is used
only for shapes that do not tile cleanly (any of `m,n,k` not divisible by 4),
because our generated kernels still win on square matrices; on CPU the host path
always wins above `m*n*k >= 1e6`.

| switch | effect |
|---|---|
| `AUTORUST_GEMM=all` | force the library for every eligible contraction |
| `AUTORUST_GEMM=0` | disable the library on every backend |
| `AUTORUST_CUBLAS=all\|0` | CUDA-only alias (kept for compatibility) |
| `AUTORUST_CUBLAS_TF32=1` | allow TF32 for f32 CUDA GEMMs |
| `AUTORUST_MPS=all\|0` | force/disable the MetalPerformanceShaders host GEMM |
| `AUTORUST_CUBLAS_LIB`, `AUTORUST_CLBLAST_LIB` | override the library path |
| `AUTORUST_CPU_GEMM_MC`, `AUTORUST_CPU_GEMM_NC`, `AUTORUST_CPU_GEMM_KC` | CPU tile sizes (48 / 96 / 256) |

Neither library is needed at build time; a missing one just disables its path.
On a 448px DINOv3 ViT-S/16, CLBlast takes the OpenCL backend from 27.0 s to
9.5 s per 128 frames (generated kernels → library GEMM), with CUDA/cuBLAS at
6.9 s. The CPU host GEMM takes the frozen CPU plan (`clusterfun --device cpu`)
from 21.7 s to 13.9 s per 16 frames, and emits 48 host-GEMM steps in the frozen
plan instead of generated kernels (freeze serializes the descriptor, so AOT and
eager agree).

Representative numbers on a 24-thread Ryzen 3900X / RTX 3060 (captured,
`JIT=1`):

| workload | before | now |
|---|---|---|
| ViT-B/16 forward, `RUST` CPU | 4.1 s | **0.66 s** |
| ViT-B/16 forward, CUDA | 166 ms | **21-22 ms** (cuBLAS) |
| 512³ f32 GEMM, `RUST` | 27.6 GFLOP/s | **~320 GFLOP/s** |
| 512³ f32 GEMM, CUDA | 144 GFLOP/s | **~650-830 GFLOP/s** |
| 2048³ bf16 GEMM, CUDA (TC) | — | **~2-3.5 TFLOP/s** |

Use `AUTORUST_KPROF=1` for a per-kernel CUDA time table (works without GPU
performance-counter permissions). `examples/cublas_check.rs` compares the
library path against the generated kernels on the same ViT weights. `AUTORUST_PROF=1` reports scheduler phases.
`SDPA_TILE=32` enables tensor-level attention padding (correct, but slower with
the current schedules, so off by default).

## Frozen CPU plans

`freeze` captures a realized `Jit` trace and emits a self-contained Rust file
(kernels + a `FrozenPlan`); the application `include!`s it, so no `rustc` is
needed at run time. The model must live in a separate crate when used from
`build.rs` (a build script cannot import the crate it is building):

```toml
[build-dependencies]
autorust = { path = "../autorust", features = ["freeze"] }
model = { path = "../model" }
```

The build script creates the model, runs one captured forward, lowers it with
`autorust::freeze::plan::build`, and calls `autorust::freeze::emit_to`; the app
then uses `include!(concat!(env!("OUT_DIR"), "/frozen.rs"))`,
`FrozenRunner::from_fns(plan(), KERNELS.to_vec())`, and `Engine` to choose eager
GPU vs frozen CPU. `ModuleForward` lives in `tensor::nn` and is implemented by
the packaged layers/models, so a built-in model (e.g.
`autorust::models::vit::ViT`) needs no wrapper. See
`examples/frozen_vit/` for the complete build-script flow, `examples/vit_freeze.rs`
for the in-process emit knob (`FREEZE_EMIT`), and `tests/freeze_emit.rs` (which
compiles the emitted crate and diffs it against eager).


| Feature | Default | What it enables |
|---|---|---|
| `capi` | yes | Small C ABI used by the development-only Python bridge. |
| `backend_rayon` | yes | Multithreaded CPU backend (`RAYON`), via `rayon`/`bytemuck`. |
| `backend_rust` | yes | Native CPU backend (`RUST`): Rust kernel source → `rustc` cdylib → `libloading`. |
| `backend_clang` | yes | Native CPU backend (`CC`): C kernel source → `$CC` (clang/gcc) shared library. |
| `backend_cl` | yes | OpenCL backend (`CL`), dynamically loaded at run time. |
| `backend_metal` | yes | Metal backend (`METAL`), macOS only; links the OS frameworks, no Xcode needed. |
| `backend_cuda` | no | CUDA backend (`CUDA`), via `cudarc`; dynamically loaded, no build-time CUDA libs. |
| `freeze` | no | Capture/emit frozen CPU plans (`autorust::freeze`); enables `backend_rust`. |
| `trace` | no | Rule-firing trace (`TG_TRACE`). |
| `dump` | no | Debug dumps and pipeline truncation (`TG_*` knobs). |

`--no-default-features` is std-only (plus `nanoserde`). Optional backend crates
are the only sanctioned non-core dependencies.

Device selection is `CUDA → METAL → CL → RUST → CC → RAYON → INTERP` for the first
available backend; `METAL` is only considered on macOS, `RUST` is only selected
when `rustc` is on `PATH`, and `CC` when `$CC` (or `clang`, else `cc`) works. Set
`DEFAULT_DEVICE` to override. Runtime knobs (device, dtype,
`BEAM`, `JIT`, `FA`, …) are environment variables read once and cached; build-
time diagnostics/backends are Cargo features. Set runtime env vars before the
first tensor/realize.

## Scope and known limitations

- **Codegen** is nearly complete, but these stages are still missing/descoped:
  late/transcendental rewrites (`get_late_rewrite_patterns`,
  `get_transcendental_patterns`), parameter renumbering (`pm_number_params`),
  and the image path. Multi-device/sharding is out of scope.
- **Backends**: CUDA tensor cores target `sm_86`; OpenCL has no offset views and
  no native bf16; Metal has native simdgroup 8x8 tensor cores (f32/f16/bf16) but
  no hardware `double` — the public Tensor API lowers f64 to f32 on METAL.
  There is no AMD/WebGPU backend. The `RUST` CPU backend
  is x86-64-first (other architectures fall back to libm for math).
- **OpenCL loading**: `backend_cl` loads `libOpenCL` dynamically (`cl_sys`);
  a binary built with it starts on machines without OpenCL and `CL` is simply
  unavailable. `AUTORUST_CL_LIB` overrides the library path for testing.
- **Metal**: `backend_metal` is target-gated to macOS and links the OS-provided
  Objective-C runtime and Metal framework; shaders are compiled at runtime with
  `newLibraryWithSource:`, so no Xcode/Metal toolchain is needed. Buffers use
  shared storage (host upload/download is a CPU copy), and the allocator
  supports real offset views. f64 is lowered to f32 on METAL (f32 precision);
  use INTERP/CPU/CL/CUDA for true f64. Native `simdgroup` 8x8 tensor cores cover
  f32/f16/bf16 matmuls (≈1.3-1.7x over the FFMA path on an M2). A
  MetalPerformanceShaders host-GEMM path also exists (f32/f32, f16/f16, f16
  inputs with f32 accumulation); it is opt-in via `AUTORUST_MPS=all`, because
  simdgroup TCs are faster on the measured M2. Shaders compile with precise math by default
  (`METAL_FAST_MATH=1` opts into fast math — `x + -inf` otherwise folds to `x`
  and drops masks). Apple GPUs allow at most 31 buffer arguments per kernel;
  `Renderer::max_buffers` turns that into a scheduler fusion budget, so large
  graphs are split at an intermediate realization rather than failing to
  compile. The same fact is the hook for other bounded backends (e.g. OpenGL
  compute).
- **Training**: eager autograd covers the ops used by the demos. Fused
  `Tensor.custom_kernel`/`@function` kernels are deferred.
- **ONNX** covers the T0/T1 inference surface. Missing: `Scatter*`,
  `NonZero`/`Compress`/`CenterCropPad`, `MaxUnpool`, `GatherND` with
  `batch_dims>0`, `Einsum` with more than two operands, `Resize` cubic,
  zero-element tensors, per-dimension conv strides, and all T2 features
  (quantization, training ops, `If`, ONNX-ML).
- The front-end is single-threaded (`Tensor` is `Rc`-based and `!Send`);
  `RAYON` parallelism is internal to the backend.
- CUDA matmuls are register-tiled by default and dispatch to dynamically
  loaded cuBLAS when the shape does not tile cleanly (see Performance). cuDNN
  is not wired up: the only convolution in the demos (the ViT patch embed) is a
  negligible fraction of the forward, and the `HostGemm` seam is where a
  library conv/attention op would slot in.
- Codegen `PADTO` on a vector-store axis is unsafe (the mask is not lane-wise);
  padding is done at the tensor level (e.g. `SDPA_TILE`) instead.

## Documentation

- [`CURRENT_STATE.md`](https://github.com/apexys/autorust/blob/main/CURRENT_STATE.md)
  — canonical status, design decisions, and milestone history.
- [`INDEXING.md`](https://github.com/apexys/autorust/blob/main/INDEXING.md) — the
  `getitem` index DSL.
- [`ONNX-Compat.md`](ONNX-Compat.md) — the ONNX T0/T1/T2 operator and feature
  matrix, runner API, validation results, and known gaps.

## License

MIT. This is a derivative work of tinygrad (MIT); see [`LICENSE`](LICENSE).
