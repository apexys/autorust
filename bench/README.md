# Cross-framework benchmark

Compares **autorust**, **candle**, **torch** and **tinygrad** on the same
workloads, on CPU and CUDA. This directory is development-only: it is excluded
from the published crate, and `bench/candle` is a standalone package so candle
never enters autorust's dependency graph.

Results live in [`results/`](results) as raw JSON plus a generated
`summary.md`; the curated table is in [`../PERFORMANCE.md`](../PERFORMANCE.md).

## Workloads

| workload | model | shapes | metric |
|---|---|---|---|
| `mnist_train` | `beautiful_mnist` CNN (2 conv, BN, pool ×2, 576→10) | batch 64 and 512, 1×28×28, Adam 1e-3, CE | ms/step, img/s, s/epoch |
| `vit_infer` | ViT-B/16 (12 layers, 768 wide, 12 heads) | batch 1 and 8, 3×224×224, patch 16, 1000 classes | ms/step, ms/image, img/s |
| `vit_train` | same ViT + CE + backward + Adam 3e-4 | batch 1 and 4, f32 | ms/step, img/s |
| `vit_infer_bf16` / `vit_train_bf16` | ViT-B/16 weights and input cast to bf16 (tensor-core path) | batch 8 infer / batch 1 train | ms/step, img/s |
| `gemm` / `gemm_bf16` | matmul shape sweep (8 shapes) | 1024³…4096³, TN/NT, batched 8×512³ | per-case ms, TFLOP/s |
| `memops` | add, copy, gelu, softmax, layernorm, sum, amax, transpose | 16M f32 elements (4096² for the 2-D ops) | per-op ms, GB/s |
| `optim` | AdamW over 100M parameters in 32 tensors | gradients set once, no zero_grad | ms/step, GB/s |
| `resnet18_infer`/`_train`, `resnet50_infer` | torchvision-shaped ResNet | batch 32, 3×224×224, 1000 classes | ms/step, img/s |
| `mobilenet_infer` | MobileNetV2 (inverted residuals, depthwise 3×3) | batch 32, 3×224×224 | ms/step, img/s |
| `unet_infer` | 4-level UNet (GroupNorm, max-pool, conv_transpose, upsample, cat) | batch 4, 3×64×64 | ms/step, img/s |
| `llm_prefill` | 6-layer LLaMA-style decoder (GQA 12/4, RoPE, causal mask) | 512 tokens, vocab 32000, f32 | ms/step, token/s |
| `llm_decode` | one token against a 512-token KV cache, Gumbel-max sampling | batch 1 | ms/step, token/s |
| `onnx_mnist` | the same MNIST CNN exported to ONNX, run through `OnnxModel` | batch 32, `bench/assets/mnist_bs32.onnx` | ms/step, img/s |
| `onnx_resnet` | ResNet-18 at width 0.25 exported to ONNX | batch 8, `bench/assets/resnet18w25_bs8.onnx` | ms/step, img/s |
| `freeze` | `vit_freeze` example: eager JIT replay vs frozen CPU plan | 1 layer, d=192, 64² | per-case ms |

`VK` (Vulkan) and `CL` (OpenCL) rows cover the same workloads as the `RUST`/
`CUDA` ones, without clBLAST on the CL side, so the two GPU APIs can be compared
directly; `PERFORMANCE.md` has the summary and the known gaps.

Inputs are **synthetic and device-resident** (`rand` for MNIST, `zeros` for
ViT), with fixed labels; the timed region is one training/inference step and
contains no data loading, evaluation or logging. This keeps the comparison
about the framework, not the data pipeline. The MNIST model itself is the
`beautiful_mnist` one, so the absolute step time is representative of the demo.

## Protocol

- One fresh process per run; warmup then `N` timed steps (see `run.py`), report
  **median**, min and IQR. GPU runs synchronize per step (and the step ends with
  a host readback/loss scalar, so enqueue-only timing is impossible).
- `24` threads for everything multithreaded (`RAYON_NUM_THREADS`,
  `OMP_NUM_THREADS`, `torch.set_num_threads`); tinygrad's `CPU` backend is
  single-threaded and is labelled as such.
- autorust runs with `Jit` capture/replay (its steady-state path), tinygrad
  with `TinyJit`, torch eager (no `torch.compile`), candle without JIT (it has
  none).
- On CPU autorust uses `AUTORUST_ATTN=sdpa`; the fused FlashAttention path is
  CUDA-only and the graph-level port is much slower (see `PERFORMANCE.md`). On
  CUDA it uses the fused flash path.
- `mnist_train` is run at the demo's batch 512 and at batch 64 (the size most
  people type); `vit_train` at batch 1 and 4, because the training gap grows
  with batch (see `PERFORMANCE.md#batch-size-scaling`). RAYON (the typed
  interpreter) is run with fewer steps because it is 1-2 orders of magnitude
  slower.
- `gemm`, `memops` and `optim` are microbenches: one JSON per framework with a
  `cases` array, rendered as per-case tables by `report.py`. The GEMM bf16 rows
  are CUDA-only (torch's CPU bf16 GEMM is unusably slow here, candle's CPU
  matmul is f32-only, tinygrad 0.14 rejects bf16 on this CUDA path).
- `llm_decode` reuses the same 512-token cache for every timed step (the cache
  is concatenated into the graph each step), so it measures per-token cost
  without including cache growth.
- The ONNX asset is committed; regenerate it with
  `uv pip install --python bench/.venv/bin/python onnx` and
  `bench/.venv/bin/python bench/tools/export_onnx.py`.
- `freeze` runs `target/release/examples/vit_freeze`
  (`cargo build --release --example vit_freeze`); it is autorust-only.
- `VK` rows need a Vulkan loader and a compute device (the backend is only
  registered when one is usable); `VK:1` selects the second device, e.g. the
  llvmpipe software driver.
- Not included: `torch.compile`, `@function` transforms, multi-GPU, and
  tinygrad/candle for the new conv/LLM rows (their bench drivers only cover the
  original workloads and GEMM). They are candidates for a follow-up.

Caveats that affect interpretation:

- tinygrad's CPU backend is single-threaded; candle's CPU matmuls use `gemm`
  (rayon), torch uses MKL, autorust uses its own packed AVX2 kernel.
- candle-nn's `BatchNorm` is inference-only, so the candle `mnist_train` step
  normalizes with batch statistics but does not update running stats.
- `RAYON` is a typed interpreter; it exists for correctness/reference, not for
  training throughput.
- Numbers are load- and clock-sensitive. Run on an idle machine, preferably
  with the same driver/CUDA version and pinned versions recorded below.

## Environment used

| | |
|---|---|
| CPU | 24-thread Ryzen 3900X |
| GPU | NVIDIA RTX 3060 12 GB, driver 580.178.04 |
| CUDA / nvcc | 12.4 |
| autorust | 0.1.0 (`git rev-parse`) |
| torch | 2.6.0+cu124 (Python 3.12.11) |
| tinygrad | v0.14.0 (`6f87158d7`) |
| candle | 0.11.1 |

The committed results in [`results/`](results) are from commit `5cf311a`
(2026-11-10); the candle runs were re-taken at `82f6408` after the CUDA
class-token fix.

## Running

```sh
# 1. Python side (uv)
uv venv bench/.venv --python 3.12
UV_CACHE_DIR=/tmp/opencode/uv-cache uv pip install \
  --python bench/.venv/bin/python \
  torch --index-url https://download.pytorch.org/whl/cu124
UV_CACHE_DIR=/tmp/opencode/uv-cache uv pip install \
  --python bench/.venv/bin/python numpy

# 2. Rust sides
cargo build --release -p autorust-bench
cargo build --release --example vit_freeze                   # frozen-plan row
(cd bench/candle && cargo build --release --features cuda)   # CPU path included

# 3. Run / report
bench/run.py --list
bench/run.py                    # full matrix
bench/run.py --only 'mnist'     # subset (regex over run names)
bench/report.py
```

tinygrad runs with the checkout's venv (`../tinygrad/.venv/bin/python`), so
`tinygrad` must be checked out next to `autorust` (or adjust `TINYGRAD_PY` in
`run.py`).

Microbench JSON files additionally carry `cases: [{case, ms, tflops|gbs}]`;
their `ms_median` is the sum of the cases (only used by the runner's summary
line). Each JSON file contains: `framework`, `version`, `device`, `workload`,
`batch`, `dtype`, `jit`, `warmup`, `iters`, `ms_cold`, `ms_median`, `ms_min`,
`ms_iqr`, `images_per_s`, plus `name`, `variant`, `commit`, `date` and `notes`
added by `run.py`. `ms_cold` is the first step including capture/compile; it is
reported for context but not part of the steady-state table.
