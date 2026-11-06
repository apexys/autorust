# Current status

Snapshot of the autorust performance work and the `clusterfun` (DINOv3
clustering) port. Commits: autorust `main`, clusterfun `autorust-backend`.

## Headline

| backend | 448px, 128 frames (`--preprocess none`) | vs candle |
|---|---|---|
| CUDA + cuBLAS + fused FA2 | **5.6 s** | candle CUDA 6.35 s |
| OpenCL + CLBlast + fused FA2 | **8.4 s** | |
| frozen CPU (AOT plan) | 6.1 s / 16 frames (0.37 s/img) | candle CPU ~0.63 s/img |

With default preprocessing, 260 frames: CUDA 14.1 s, OpenCL 20.1 s,
candle CUDA 15.4 s. Projection to the real ~10k-image set: ~470 s vs ~590 s.

All GPU libraries and backends are dynamically loaded; nothing is needed at
build time. `cargo test`: default 224, `backend_cuda` 231,
`backend_clang+freeze` 226.

## What was built, in the order it landed

### GEMM
- Register-tiled generated GEMM with grid-axis ordering, `bf16` coalescing and
  promotion, `AUTORUST_KPROF`/`AUTORUST_PROF`.
- **`HostGemm`**: the scheduler describes a pure contraction once (shapes,
  strides, dtype, alpha, `bias + per-column scale + residual` epilogue) and each
  backend executes it with its fastest implementation:
  - CUDA → dynamically loaded **cuBLAS** (f32/f16/bf16, strided-batched);
  - OpenCL → dynamically loaded **CLBlast** (f32, row-major);
  - CPU → **packed blocked kernel** (rayon tiles, 6x16 AVX2/FMA microkernel,
    scalar fringes).
- Selection: CUDA/CL use the library only for shapes that do not tile cleanly
  (any of `m,n,k` not divisible by 4); CPU always uses the host path above
  `m*n*k >= 1e6` because the generated CPU kernel tiles a single row.
- Switches: `AUTORUST_GEMM=all|0`, `AUTORUST_CUBLAS=all|0` (alias),
  `AUTORUST_CUBLAS_LIB`, `AUTORUST_CLBLAST_LIB`,
  `AUTORUST_CPU_GEMM_{MC,NC,KC}`, `AUTORUST_CPU_GEMM_DEBUG`.

### Realization and freezing
- Fixed a pre-existing bug where a realized intermediate's storage was dropped
  when the `Tensor` moved into a consumer: `Session::pin`/`unpin_all` plus
  `tests/realize_reuse.rs`.
- `Session::pin` keeps explicitly realized storage alive; the app unpins after
  each eager batch.
- Freeze serialises a `HostGemm` as `Step::Gemm` (descriptor as source), so the
  AOT plan and eager execution agree.

### Attention
- CPU: `scaled_dot_product_attention_sess` materialises the score matrix, so
  `P·V` becomes a `HostGemm` (12 AV products: ~150 ms → ~7 ms per image).
- CUDA/CL: hand-written **FA2 kernels** (`runtime::cuda_flash`,
  `runtime::cl_flash`) — 64-query tiles, 32-key streaming steps, online softmax,
  D=64, no mask/causal. Launched through `Compiled::run_host_kernel`, the seam
  for kernels the uop IR cannot express.
- `AttentionKind::{Sdpa, Flash, Sage}` via `AUTORUST_ATTN` (default `flash`).
  `sage` is the placeholder for the quantized variant and currently degrades to
  `flash` with a one-time notice.
- The fused launch is a **captured step**: `Session::capturing` is an ordered
  `Vec<CaptureEntry>` and the JIT replays `JitStep::{Linear, Host}` in order, so
  eager replay reproduces the exact sequence.

### App-level
- JIT capture/replay, persistent input buffer (`Tensor::write_bytes`), batch 8,
  `free_cache()` between batches.
- Frozen plan captured through `forward_with` (realize boundaries), so it
  contains 97 pure contractions (85 + 12 after the AV split).

## Numbers along the way (CPU, 16 frames at 448px)

| step | time |
|---|---|
| generated kernels | 21.7 s |
| + packed `HostGemm` | 13.9 s |
| + boundary capture | 7.5 s |
| + `P·V` as a `HostGemm` | 5.9 s |

CPU now beats candle CPU (~0.37 vs ~0.63 s/image).

## Accuracy

- CPU vs CUDA embeddings: 1.4e-6 maxabs, cosine 0.9999995.
- CUDA fused attention vs SDPA: 8.0e-7, cosine 0.9999995.
- CL fused attention vs SDPA: 7.9e-7, cosine 0.9999995.
- `examples/frozen_check`: 5.5e-6.

## Known limitations / next steps

1. **SageAttention** is not implemented. On sm_86 there are no FP8 tensor cores,
   so the target is INT8 QK^T (`mma.s8` or `dp4a`) with SmoothK (subtract the
   per-block K mean, add the correction) and per-block scales; P·V stays
   bf16/f32. Accuracy must be gated on embedding drift *and* cluster-assignment
   stability over the real ~10k set, not just maxabs.
2. **`freeze` cannot serialise a host-kernel step.** It panics with a clear
   message; harmless today because the CPU capture never emits one, but a
   GPU-side AOT plan needs `Step::Host` plus a runtime that can bind raw
   pointers.
3. **Attention output buffer** is allocated per layer per batch (~9.7 MB at
   batch 8). Reusing one scratch buffer should recover some of the gap between
   the 2.2x kernel win and the observed ~10% end-to-end win.
4. **RAYON** (the interpreter-backed CPU backend) is not optimised: its renderer
   facts still advertise the interpreter's capabilities, so it gets a GPU-shaped
   schedule and runs the typed IR at ~0.9 GFLOP/s. It is not on the app's path
   (the frozen plan uses the compiled `RUST` backend).
5. **GPU attention tuning**: the FA2 kernels are scalar f32 on CUDA cores
   (~1.3 TFLOP/s, ~10% of FP32 peak). Tensor-core bf16 `mma` is the next 4-8x;
   the quantization above is a further ~1.3-1.5x on top.
6. Measurement note: this machine occasionally carries unrelated load (seen at
   load average ~11), which inflated one CPU run to 66 s. Re-measure before
   trusting a single outlier.

## Reproduce

```bash
cd clusterfun
cargo build --release
./target/release/clusterfun <images> /tmp/out --device cuda --k 4
AUTORUST_ATTN=sdpa ./target/release/clusterfun <images> /tmp/out --device cuda --k 4
FROZEN_PROF=1 ./target/release/clusterfun <images> /tmp/out --device cpu
AUTORUST_KPROF=1 ./target/release/clusterfun <images> /tmp/out --device cuda
```
