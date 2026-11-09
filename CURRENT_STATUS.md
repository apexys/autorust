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
build time. `cargo test`: default 224, `backend_cuda` 244,
`backend_clang+freeze` 227.

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
- **TF32 tensor-core FA2** (`runtime::cuda_flash_tc`) on `sm_80+`: warp-tiled
  `mma.sync.m16n8k8`, Q in A fragments, K/V streamed through 16.6 KB of shared,
  score tile and output accumulator in registers. 5.9x over the scalar kernel
  at the app shape (7.0 ms -> 1.18 ms, batch 8, h 6, n 789) with ~1e-5 error.
- **bf16 tensor-core FA2** (`runtime::cuda_flash_bf16`): `mma.sync.m16n8k16`,
  bf16 operands and f32 accumulate, 9 KB of shared per stage (18 KB with the
  async pipeline), 1.8-2.4x the TF32 kernel (0.52-0.54 ms vs 0.9-1.3 ms at the
  same shape) and <1e-4 against a reference that rounds operands and
  probabilities to bf16. Chosen by tensor dtype.
- **`cp.async` double-buffered K/V** (`AUTORUST_ASYNC`, default on) in the bf16
  kernel: the next tile is issued before the wait for the current one, giving
  1.3-1.4x at the app shape (0.70-0.75 ms -> 0.52-0.54 ms). The TF32 kernel has
  the same path; there it is within noise because the TF32 conversion moves to
  the fragment loads and the doubled shared memory halves occupancy, so the
  dispatcher leaves TF32 on the synchronous path.
- **Causal, additive `[n, n]` f32 mask, and GQA** in all three CUDA kernels.
  Causal bounds the key loop to `block_q + BQ` as well as masking the diagonal
  tiles; the additive mask is added to the scores before the max; GQA divides
  the Q head index by `h_q / h_kv`. `Tensor::attention*` infers GQA from the
  head counts, and unsupported mask shapes fall back to SDPA.
- Both are gated on `Settings.tensor_cores` (`AUTORUST_TENSOR_CORES`, default
  on); `sm < 80` keeps the scalar f32 kernel and bf16 falls back to SDPA.
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

## SageAttention (prototype, not enabled)

`runtime::sage_flash` implements the SageAttention idea on the FA2 schedule:
symmetric per-row INT8 quantization of Q and K, `__dp4a` for QK^T, f32 softmax
and P·V. Measured at the DINOv3 shape (6 heads, 789 tokens, D=64):

| variant | time/layer | accuracy vs SDPA |
|---|---|---|
| f32 FA2 (`cuda_flash`) | 1.22 ms | 1.2e-8 |
| INT8 dp4a (`sage_flash`) | 1.43 ms | cosine 0.83 |

Two findings, both structural:

- **It is slower, not faster.** `dp4a` is 4 MACs/instruction on the same CUDA
  cores, but the kernel is bound by softmax, quantization and memory, and the
  per-key-block max+quantize pass (plus two barriers) costs more than the f32
  FMAs it removes. SageAttention's 2x comes from `mma.s8` **tensor cores**, not
  from dp4a.
- **Accuracy needs SmoothK.** Naive per-row int8 loses too much (cosine 0.83 on
  the synthetic test); SageAttention's per-block mean subtraction is what makes
  int8 safe, and it needs a pass over the whole key sequence before the
  streaming loop.

So `AttentionKind::Sage` stays a placeholder that degrades to `Flash` with a
one-time notice, and `Tensor::sage_attention` runs the fused f32 kernel. The
module is kept as the measured starting point for a tensor-core (`mma.s8`)
implementation.

## Attention API

Three tensor methods select the implementation explicitly (plus `_with`
variants that take a `Session`, needed for the fused kernels):

```rust
q.attention(&k, &v, mask, dropout, is_causal)        // AUTORUST_ATTN default
q.flash_attention(&k, &v, mask, dropout, is_causal)  // fused FA2, falls back
q.sage_attention(&k, &v, mask, dropout, is_causal)   // quantized (-> flash today)
```

All three fall back to `scaled_dot_product_attention` for masked, causal or
dropout cases, and for shapes the kernel does not cover.

## Settings on `Ctx`

`src/settings.rs` holds the per-context configuration. `Settings::from_env()` is
called **once per process** by `settings::global()` (`OnceLock`), so no lowering
path pays a `getenv`; each `Ctx` clones the process defaults at creation and can
then be reconfigured:

```rust
let ctx = Ctx::new();
ctx.borrow_mut().settings.attention = AttentionKind::Sdpa;
ctx.borrow_mut().settings.library_gemm = LibraryGemm::Never;   // generated only
let s = Settings::default().with_attention(AttentionKind::Flash).with_beam(4);
```

Migrated so far: `attention` (`AUTORUST_ATTN`, `FA`), `library_gemm`
(`AUTORUST_GEMM`/`AUTORUST_CUBLAS`/`AUTORUST_CLBLAST`), `allow_tf32`
(`AUTORUST_CUBLAS_TF32`, `ALLOW_TF32`), `cpu_gemm_tile`
(`AUTORUST_CPU_GEMM_*`), `beam` (`BEAM`), `cache` (`CACHELEVEL`).

Defaults are "use what is available": `library_gemm` is
`LibraryGemmPolicy { cuda, cl, cpu }`, all `Auto`, which resolves to the
library on every backend and is only downgraded by an explicit opt-out. The
policy is per backend, so `AUTORUST_CLBLAST=0` no longer disables cuBLAS or the
CPU path (it did in the first cut, which the dispatch counts caught:
`AUTORUST_CLBLAST=0` still emits the CPU host GEMM, `AUTORUST_CPU_GEMM=0` does
not). Precedence: backend-specific env > `AUTORUST_GEMM` > default.

The policy is deliberately split from presence: `cublas::available()` /
`clblast::available()` / `cpu_gemm::available()` mean "the library loaded", and
the per-context `library_gemm` decides whether to use it. That is what lets two
contexts in one process make different choices (the cublas/clblast tests now do
exactly that instead of toggling env vars, which the `OnceLock` would ignore).

## Environment variables: keep vs. settings

~55 variables are read across the crate. Split, with the first group already
migrated:

**Keep as env (diagnostics/toolchain — easy to flip in a finished app):**

| group | variables |
|---|---|
| profiling | `AUTORUST_PROF`, `AUTORUST_KPROF`, `FROZEN_PROF` |
| debug dumps | `AUTORUST_CUBLAS_DEBUG`, `AUTORUST_CPU_GEMM_DEBUG`, `AUTORUST_MM_DEBUG`, `AUTORUST_DEVICE_VERBOSE`, `TG_DUMP_{SRC,LST,IDX,INTERP,PRE_EXPAND}`, `TG_DEBUG_{LOAD,RANGEIFY}`, `TG_{TRACE,RW_TRACE,RW_VISITS}`, `TG_M5_{CODEGEN,EXEC,RENDER,STAGES}`, `TG_{CUDA,OCL}_RENDER`, `BEAM_DEBUG`, `CACHELEVEL` |
| toolchain/paths | `CUDA_PATH`, `RUSTC`, `CC`, `AUTORUST_CACHE_DIR`, `XDG_CACHE_HOME`, `HOME`, `AUTORUST_RUST_CPU`, `AUTORUST_CC_ARCH`, `AUTORUST_CUBLAS_LIB`, `AUTORUST_CLBLAST_LIB`, `AUTORUST_CL_LIB`, `{CUDA,CL}_COMPILE_CACHE`, `AUTORUST_{RUST,CC}_DUMP` |

**Should become `Ctx` settings (behavioural policy):**

| variable | proposed settings call |
|---|---|
| `AUTORUST_ATTN`, `FA` | **done** — `settings.attention` |
| `AUTORUST_GEMM`, `AUTORUST_CUBLAS`, `AUTORUST_CLBLAST` | **done** — `settings.library_gemm` |
| `AUTORUST_CUBLAS_TF32`, `ALLOW_TF32` | **done** — `settings.allow_tf32` |
| `AUTORUST_CPU_GEMM_{MC,NC,KC}` | **done** — `settings.cpu_gemm_tile` |
| `BEAM` | **done** — `settings.beam` |
| `CACHELEVEL` | **done** — `settings.cache` |
| `SDPA_TILE` | todo — `settings.sdpa_tile(n)` |
| `DEFAULT_DEVICE` | todo — `settings.device("CUDA")` (read before a `Ctx` exists; needs a process-global) |
| `EMULATED_DTYPES` | todo — `settings.emulated_dtypes(&[..])` |
| `RAYON_{SERIAL,F64,INTERP,VERBOSE}` | todo — `settings.rayon(RayonMode)` |
| `CLUSTERFUN_{NO_JIT,TIMING}` | app settings (not the library) |

Kept env-only on purpose: anything that must be settable before a `Ctx` exists
(paths, toolchain) or that a developer flips on a built binary (profiling,
dumps). Note `DEFAULT_DEVICE` is read during tensor creation and would need a
process-global until every creation path takes `&Ctx`.

## Known limitations / next steps

1. **SageAttention** needs tensor cores. `dp4a` measured 1.43 ms vs 1.22 ms for
   f32 FA2 and cosine 0.83, so the next attempt is `mma.s8` (sm_86 has no FP8)
   with SmoothK and per-block scales; P·V stays bf16/f32. Gate on embedding
   drift *and* cluster-assignment stability over the real ~10k set.
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


## Tensor-core / TF32 note (measured)

`allow_tf32` controls **cuBLAS only** and is **off by default**: TF32 cuBLAS is
within noise of pedantic f32 at our shapes (below), while it changes the model
output, so there is no reason to pay for it. `AUTORUST_CUBLAS_TF32=1` enables
it. `AUTORUST_CUBLAS_TF32=0` maps to `CUBLAS_PEDANTIC_MATH` (the pre-existing
`false` used math mode 0, which already allows TF32 on CUDA >= 11 — the switch
was a no-op before that mapping).

Measured on the 3060, cuBLAS f32:

| shape | pedantic f32 | TF32 |
|---|---|---|
| 2048^3 | 15.03 ms (1143 GFLOP/s) | 15.47 ms (1110) |
| 789x1536x384 | 7.83 ms (119) | 8.86 ms (105) |
| 789x384x384 | 0.904 ms (258) | 0.888 ms (262) |

No meaningful difference, and 2048^3 at ~1.1 TFLOP/s is far below the card's
capability, so these runs are **not compute-bound** — the flag is not the lever
for cuBLAS. TF32 pays off in the **fused attention kernel**, where the scalar
one-thread-per-row kernel was the bottleneck and TF32 is the only f32-storage
tensor-core format. The new warp-tiled `mma.sync.m16n8k8` kernel
(`runtime/cuda_flash_tc.rs`) measures, on the 3060, batch 8 / h 6 / n 789:

| kernel | time | error vs f32 reference |
|---|---|---|
| scalar f32 FA2 | 7.0 ms | 1.2e-8 |
| warp-tiled TF32 | **1.18 ms** (5.9x) | 9e-6 |

and every tested length (1, 7, 8, 31, 32, 33, 63, 64, 65, 100, 224, 789)
agrees with the scalar reference to < 3e-5. `tensor_cores` defaults on;
`AUTORUST_TENSOR_CORES=0` forces the scalar kernel for bit-exact comparisons.

End to end in clusterfun (8 images, 448px), CUDA vs the frozen CPU plan:

| CUDA settings | max abs | min cosine |
|---|---|---|
| pedantic cuBLAS + scalar attention | 1.6e-6 | 1.0000000 |
| TF32 cuBLAS + scalar attention | 3.1e-4 | 0.9999985 |
| pedantic cuBLAS + TF32 attention | 3.8e-4 | 0.9999971 |

so the tensor-core attention costs about the same as TF32 cuBLAS and is worth
it; TF32 cuBLAS is not, hence the default.

Note: `Settings::default()` had `allow_tf32: true` while `Settings::from_env`
(which `global()` actually uses) still mapped "unset" to `false`; both now say
`false`, so the documented default is the effective one.

The mma fragment layouts are not obvious and cost a debugging round: for
`m16n8k8` the A layout is `(g,2t), (g+8,2t), (g,2t+1), (g+8,2t+1)`, the C layout
is the same with registers 1 and 2 swapped. `tests/tc_mma_probe.rs` pins this
down.

