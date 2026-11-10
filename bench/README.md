# Cross-framework benchmark

Compares **autorust**, **candle**, **torch** and **tinygrad** on the same two
models, on CPU and CUDA. This directory is development-only: it is excluded
from the published crate, and `bench/candle` is a standalone package so candle
never enters autorust's dependency graph.

Results live in [`results/`](results) as raw JSON plus a generated
`summary.md`; the curated table is in [`../PERFORMANCE.md`](../PERFORMANCE.md).

## Workloads

| workload | model | shapes | metric |
|---|---|---|---|
| `mnist_train` | `beautiful_mnist` CNN (2 conv, BN, pool ×2, 576→10) | batch 512, 1×28×28, Adam 1e-3, CE | ms/step, img/s, s/epoch |
| `vit_infer` | ViT-B/16 (12 layers, 768 wide, 12 heads) | batch 1 and 8, 3×224×224, patch 16, 1000 classes | ms/step, ms/image, img/s |
| `vit_train` | same ViT + CE + backward + Adam 3e-4 | batch 1, f32 | ms/step, img/s |

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
- `mnist_train` uses the batch-512 demo; RAYON (the typed interpreter) is run
  with fewer steps because it is 1-2 orders of magnitude slower.
- Not included: bf16/tensor-core runs, `torch.compile`, `@function` transforms,
  multi-GPU. They are candidates for a follow-up.

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

Each JSON file contains: `framework`, `version`, `device`, `workload`,
`batch`, `dtype`, `jit`, `warmup`, `iters`, `ms_cold`, `ms_median`, `ms_min`,
`ms_iqr`, `images_per_s`, plus `name`, `variant`, `commit`, `date` and `notes`
added by `run.py`. `ms_cold` is the first step including capture/compile; it is
reported for context but not part of the steady-state table.
