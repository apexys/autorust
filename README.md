# autorust

A faithful Rust port of [tinygrad](https://github.com/tinygrad/tinygrad) v0.14:
an interned UOp graph, a rangeify scheduler, a codegen pipeline, and native
CPU/OpenCL/CUDA backends. The Python tree is the reference oracle; Rust never
imports Python at runtime.

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
  portable C compiled by clang/gcc), `CL` (OpenCL), and `CUDA` (driver API +
  NVRTC, bf16/WMMA).
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
`examples/vit_mnist.rs`, plus `examples/fa_bench.rs` and
`examples/rayon_bench.rs` for attention/CPU benchmarks.

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
| `backend_cuda` | no | CUDA backend (`CUDA`), via `cudarc`; dynamically loaded, no build-time CUDA libs. |
| `freeze` | no | Capture/emit frozen CPU plans (`autorust::freeze`); enables `backend_rust`. |
| `trace` | no | Rule-firing trace (`TG_TRACE`). |
| `dump` | no | Debug dumps and pipeline truncation (`TG_*` knobs). |

`--no-default-features` is std-only (plus `nanoserde`). Optional backend crates
are the only sanctioned non-core dependencies.

Device selection is `CUDA → CL → RUST → CC → RAYON → INTERP` for the first
available backend; `RUST` is only selected when `rustc` is on `PATH`, `CC` when
`$CC` (or `clang`, else `cc`) works. Set `DEFAULT_DEVICE` to override. Runtime knobs (device, dtype,
`BEAM`, `JIT`, `FA`, …) are environment variables read once and cached; build-
time diagnostics/backends are Cargo features. Set runtime env vars before the
first tensor/realize.

## Scope and known limitations

- **Codegen** is nearly complete, but these stages are still missing/descoped:
  late/transcendental rewrites (`get_late_rewrite_patterns`,
  `get_transcendental_patterns`), parameter renumbering (`pm_number_params`),
  and the image path. Multi-device/sharding is out of scope.
- **Backends**: CUDA tensor cores target `sm_86`; OpenCL has no offset views and
  no native bf16; there is no AMD/Metal/WebGPU backend. The `RUST` CPU backend
  is x86-64-first (other architectures fall back to libm for math).
- **OpenCL loading**: `backend_cl` loads `libOpenCL` dynamically (`cl_sys`);
  a binary built with it starts on machines without OpenCL and `CL` is simply
  unavailable. `AUTORUST_CL_LIB` overrides the library path for testing.
- **Training**: eager autograd covers the ops used by the demos. Fused
  `Tensor.custom_kernel`/`@function` kernels are deferred.
- **ONNX** covers the T0/T1 inference surface. Missing: `Scatter*`,
  `NonZero`/`Compress`/`CenterCropPad`, `MaxUnpool`, `GatherND` with
  `batch_dims>0`, `Einsum` with more than two operands, `Resize` cubic,
  zero-element tensors, per-dimension conv strides, and all T2 features
  (quantization, training ops, `If`, ONNX-ML).
- The front-end is single-threaded (`Tensor` is `Rc`-based and `!Send`);
  `RAYON` parallelism is internal to the backend.

## Documentation

- [`CURRENT_STATE.md`](https://github.com/apexys/autorust/blob/main/CURRENT_STATE.md)
  — canonical status, design decisions, and milestone history.
- [`INDEXING.md`](https://github.com/apexys/autorust/blob/main/INDEXING.md) — the
  `getitem` index DSL.
- [`ONNX-Compat.md`](ONNX-Compat.md) — the ONNX T0/T1/T2 operator and feature
  matrix, runner API, validation results, and known gaps.

## License

MIT. This is a derivative work of tinygrad (MIT); see [`LICENSE`](LICENSE).
