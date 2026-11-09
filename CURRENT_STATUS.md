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
build time. `cargo test`: default/CL 225, `backend_cuda` 253,
`backend_clang+freeze` 228.

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
- **`ldmatrix` fragment loads** in the bf16 kernel (`m8n8.x2` for K,
  `x2.trans` for V); `tests/ldmatrix_probe.rs` pins both orientations.
- **Occupancy pass**: bf16 is `__launch_bounds__(128, 5)` -> 96 regs, 18.4 KB
  shared, 5 blocks/SM, 0 B local (was 118 regs / 4 blocks). TF32 keeps 122 regs
  / 4 blocks: forcing 5 blocks there spills 16 B. `AUTORUST_KERNEL_INFO=1`
  prints regs/shared/local per compiled kernel.
- **`cp.async` double-buffered K/V** (`AUTORUST_ASYNC`, default on) in the bf16
  kernel: the next tile is issued before the wait for the current one, giving
  1.3-1.4x at the app shape (0.70-0.75 ms -> 0.52-0.54 ms). The TF32 kernel has
  the same path; there it is within noise because the TF32 conversion moves to
  the fragment loads and the doubled shared memory halves occupancy, so the
  dispatcher leaves TF32 on the synchronous path.
- **Causal, additive `[n, n]` f32 mask, and GQA** in all three CUDA kernels and
  in the OpenCL kernel.
- **SageAttention** (`runtime::cuda_flash_sage`, opt-in): QKᵀ on 8-bit tensor
  cores — INT8 `mma.m16n8k32` on sm_80+ (tested on the 3060) and FP8 E4M3 on
  sm_89+ (compile-tested for sm_89; runtime test ready) — with per-row Q /
  per-key K scales, f32 softmax and bf16 P·V. On sm_86 it is slower than TF32
  (1.26 vs 0.97 ms at the app shape) because K is re-quantized per query block;
  a per-head pre-quantization pass is the next optimisation.
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

## SageAttention (implemented, opt-in)

`runtime::cuda_flash_sage` runs QKᵀ on 8-bit tensor cores:

| arch | instruction | quantization |
|---|---|---|
| sm_80..88 | `mma.m16n8k32.s32.s8.s8.s32` | symmetric int8, `max/127` per Q row / K row |
| sm_89+ | `mma.m16n8k32.f32.e4m3.e4m3.f32` | E4M3, `max/448` per Q row / K row |

Softmax is f32 and `P·V` stays on the bf16 tensor cores, so only the logits are
approximate. INT8 matches a reference that performs the same quantization to
~2e-5 at the DINOv3 shape (`tests/sage_int8.rs`, sm_86). FP8 compiles for sm_89
(unit test) and has the same runtime test ready for an Ada/Hopper card.

On the 3060, INT8 is 1.26 ms vs 0.97 ms for TF32: the 8-bit `mma` is faster,
but K is re-quantized once per query block (13x per head) and the f32 staging +
conversion dominates. SageAttention pre-quantizes K once per head into global
int8/FP8 plus scales; that removes the per-block pass and is the next step
(together with SmoothK). The earlier `dp4a` prototype
(`runtime/sage_flash.rs`) is kept only as the measured starting point.

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

## Environment variables: one OS read per name

Every env read in `src/` goes through `helpers::env::{get,get_os}`, which caches
the raw `OsString` per name in a process-wide map. A name is read from the OS
**at most once**, whether it is a behavioural setting, a toolchain path, or a
debug switch, and `get`/`get_os` share the same entry so mixing them cannot
read twice.

Behavioural policy lives in `Settings` (read once by `global()`): `attention`,
`library_gemm` (per backend), `allow_tf32`, `tensor_cores`, `async_pipeline`,
`cpu_gemm_tile`, `beam`, `cache`, `sdpa_tile`, `mm_tile`, `device`.

Diagnostics/toolchain keep their env interface (settable before a `Ctx`, easy to
flip on a finished binary) but are still cached: `AUTORUST_PROF`,
`AUTORUST_KPROF`, `AUTORUST_KERNEL_INFO`, `FROZEN_PROF`,
`AUTORUST_{CUBLAS,CPU_GEMM,MM,DEVICE,ATTN_DEBUG}`/`AUTORUST_ATTN_DEBUG`,
`TG_DUMP_*`, `TG_DEBUG_*`, `TG_{TRACE,RW_TRACE,RW_VISITS}`, `TG_M5_*`,
`TG_{CUDA,OCL}_RENDER`, `BEAM_DEBUG`, `CACHELEVEL`, `CUDA_PATH`, `RUSTC`, `CC`,
`AUTORUST_CACHE_DIR`, `XDG_CACHE_HOME`, `HOME`, `AUTORUST_RUST_CPU`,
`AUTORUST_CC_ARCH`, the library paths and `{CUDA,CL}_COMPILE_CACHE`.

`EMULATED_DTYPES` and `RAYON_*` are also env (they must be visible to lowering /
the thread pool) and are cached the same way. `CLUSTERFUN_{NO_JIT,TIMING}` are
app-side.

## Known limitations / next steps

1. **SageAttention** is implemented (INT8 on sm_80+, FP8 on sm_89+) but still
   quantizes K per query block, so it loses to TF32 on the 3060. Next: a
   per-head pre-quantization pass (global int8/FP8 K + scales) and SmoothK,
   then gate on embedding drift and cluster-assignment stability over the real
   ~10k set. The FP8 path needs an Ada/Hopper card to run `tests/sage_int8.rs`.
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

