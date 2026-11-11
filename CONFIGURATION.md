# Configuration

Everything that changes behaviour, in three layers:

1. **Cargo features** — which backends and debug facilities are compiled in.
2. **`Settings` on a `Ctx`** — per-context lowering choices (attention kind,
   library GEMM, tensor cores, tile sizes, …). Cheap to change; this is the
   right layer for per-model or per-pass policy.
3. **Environment variables** — process defaults, toolchain paths, cache
   locations, and diagnostics. Every read goes through `helpers::env`, which
   caches the raw value per name, so a variable is read from the OS **at most
   once per process**. Set them before the first tensor/realize.

Device selection is process-wide, like tinygrad's `Device.DEFAULT`; see
[`PERFORMANCE.md`](PERFORMANCE.md#devices-and-selection).

## Feature flags

`cargo add autorust` enables everything below except `trace`/`dump`.

| feature | default | what it enables |
|---|---|---|
| `backend_rayon` | yes | Multithreaded CPU backend (`RAYON`), via `rayon`/`bytemuck`. |
| `backend_rust` | yes | Native CPU backend (`RUST`): Rust kernel source → `rustc` cdylib → loaded with `libloading`. |
| `backend_clang` | yes | Native CPU backend (`CC`): C kernel source → `$CC` (clang/gcc) shared library. |
| `backend_cl` | yes | OpenCL backend (`CL`), dynamically loaded at run time. |
| `backend_metal` | yes | Metal backend (`METAL`), macOS only; links the OS frameworks, no Xcode needed. |
| `backend_cuda` | yes (Linux/Windows) | CUDA backend, via `cudarc` (driver + NVRTC, dynamically loaded). Target-gated: macOS builds never see the dependency. |
| `freeze` | yes | Capture/emit frozen CPU plans (`autorust::freeze`, see [`FREEZE.md`](FREEZE.md)); implies `backend_rust`. |
| `trace` | no | Rule-firing trace (`TG_TRACE`). |
| `dump` | no | Codegen/interpreter dumps and pipeline truncation (`TG_DUMP_*`, `TG_M5_STAGES`, `TG_DEBUG_LOAD`). |

`--no-default-features` is std-only plus `nanoserde` and still compiles; the
`INTERP`/`CPU` device remains available. `--no-default-features --features
backend_rayon` gives a pure-Rust multithreaded CPU build. Optional backend
crates are the only non-core dependencies.

`feature = "trace"`/`"dump"` only add diagnostics; they do not change the
public API. The three `*_smoke` binaries are gated by their backend features.

## `Settings` on a `Ctx`

`Ctx` is a handle to the interned arena (`Rc<RefCell<UOpCtx>>`); the settings
live in `ctx.borrow_mut().settings`. Each `Ctx` clones the process defaults
(`helpers::env` at `Settings::global()` time) at creation and can then be
reconfigured:

```rust
use autorust::models::flash_attention::AttentionKind;
use autorust::settings::{LibraryGemm, LibraryGemmPolicy};
use autorust::tensor::Ctx;

let ctx = Ctx::new();
{
    let mut c = ctx.borrow_mut();
    c.settings.attention = AttentionKind::Sdpa;
    c.settings.library_gemm = LibraryGemmPolicy::uniform(LibraryGemm::Never);
    c.settings.tensor_cores = false;
}
```

`Settings::default().with_attention(…).with_library_gemm(…).with_beam(4)` is the
builder form for standalone settings objects; `autorust::settings::global()`
returns the read-only process defaults.

| field | type / default | env | meaning |
|---|---|---|---|
| `attention` | `AttentionKind::Flash` | `AUTORUST_ATTN` | `Flash` (fused where supported, else SDPA), `Sdpa`, `Sage` (8-bit QKᵀ on CUDA `sm_80+`/`sm_89+`; falls back to the exact kernel elsewhere). |
| `library_gemm` | `LibraryGemmPolicy::uniform(Auto)` | `AUTORUST_GEMM`, then `AUTORUST_CUBLAS`/`AUTORUST_CLBLAST`/`AUTORUST_MPS`/`AUTORUST_CPU_GEMM` | Per-backend `Auto`/`Always`/`Never` policy for cuBLAS/CLBlast/MPS/packed-CPU GEMM. `Auto` resolves to "use the library where it loaded and wins" (CUDA f32 always; shape heuristic elsewhere), see [`PERFORMANCE.md`](PERFORMANCE.md#host-gemm-library-path). |
| `allow_tf32` | `false` | `AUTORUST_CUBLAS_TF32` (alias `ALLOW_TF32`) | Allow TF32 for **cuBLAS f32 only**. Off by default: it changes the output for no measured gain. |
| `tensor_cores` | `true` | `AUTORUST_TENSOR_CORES` (alias `AUTORUST_TC`) | Allow `mma` kernels in generated matmuls and fused attention (`sm_80+`, Apple7+). `0` forces the scalar paths. |
| `cpu_gemm_tile` | `None` | `AUTORUST_CPU_GEMM_MC/NC/KC` | CPU host-GEMM tile `(mc, nc, kc)`; all three must be set. Defaults 48/96/256. |
| `beam` | `0` (off) | `BEAM` (alias `AUTORUST_BEAM`) | BEAM search width for codegen. |
| `cache` | `true` | `CACHELEVEL` | On-disk kernel/tuning cache. `CACHELEVEL=0` disables it. |
| `sdpa_tile` | `0` (off) | `SDPA_TILE` | Pad attention to a multiple of this at the tensor level (correct but currently slower). |
| `mm_tile` | `None` | `AUTORUST_MM_M/N/K/LM/LN` | Generated matmul tile override; all five must be set. |
| `device` | from `DEFAULT_DEVICE` | `DEFAULT_DEVICE` | Process default device. Read from the **global** settings by tensor creation; per-`Ctx` changes do not affect constructors — use `set_default_device` or pass the device explicitly. |
| `cuda_device` | `0` | `CUDA_DEVICE` | GPU ordinal used by the priority-picked default (`CUDA:N`). |
| `cl_device` | `0` | `CL_DEVICE` | Same for enumerated OpenCL devices. |
| `async_pipeline` | `true` | `AUTORUST_ASYNC` | `cp.async` double buffering in the bf16 fused-attention kernel (`0`/`false` disables). |

Notes:

- Settings are consulted at lowering time. A kernel that is already compiled
  (per-session `ProgCache` or the content-keyed process-wide cache) is not
  recompiled because a setting changed; start a fresh `Session`/`Ctx` if you
  need to re-lower. Changing settings between realizations within one session is
  supported (the library/clblast/cublas tests do exactly that per context).
- `Settings::default()` and the environment defaults agree; the difference is
  that `global()` is initialised from the environment once.

## Environment reference

All names are read once (first use) and cached. Set values before creating the
first tensor.

`helpers::env::with_context(&[(name, value)], || …)` pushes a scoped override
stack, but it is honoured **only** by the `getenv_*`/`ContextVar` helpers — that
is the dtype variables (`DEFAULT_FLOAT`, `DEFAULT_INT`, `SUM_DTYPE`), which cache
on first resolution. `Settings`, the BEAM/TC knobs, and the diagnostics use the
cached `helpers::env::get` and do **not** see context overrides: set real
environment variables, or mutate `ctx.borrow_mut().settings`.

### Device selection

| variable | values | default | effect |
|---|---|---|---|
| `DEFAULT_DEVICE` | device name, `auto` | first available (CUDA → METAL → CL → RUST → CC → RAYON → INTERP) | Device used when a constructor gets no explicit device. `auto` runs `select_fastest_device()` lazily. |
| `CUDA_DEVICE`, `CL_DEVICE` | ordinal | `0` | Index appended to the priority-picked name (`CUDA:1`); index 0 stays the bare name. |
| `AUTORUST_AUTOTUNE_DEVICES` | comma-separated names | all candidates | Restrict `select_fastest_device()` / `DEFAULT_DEVICE=auto` to these. |
| `AUTORUST_AUTOTUNE_QUIET` | any | unset | Suppress the autotune summary and per-device failure lines. |
| `AUTORUST_AUTOTUNE_TRACE` | any | unset | Print the per-case timing samples. |

### Lowering / policy

These mirror the `Settings` fields above; the per-backend variables win over
`AUTORUST_GEMM`.

| variable | values | default | effect |
|---|---|---|---|
| `AUTORUST_ATTN` | `sdpa`/`0`, `sage`, anything else | `flash` | Attention implementation for `Tensor::attention`. |
| `FA` | any non-`0` | off | Route `TransformerBlock` through the op-level FlashAttention port instead of SDPA (an algorithm reference; the fused kernels are reached by `Tensor::attention_with`). |
| `AUTORUST_GEMM` | `0`/`never`, `all`/`force`/`always`, anything else | `auto` | Neutral library-GEMM policy. |
| `AUTORUST_CUBLAS` | same | auto | CUDA override. |
| `AUTORUST_CLBLAST` | same | auto | OpenCL override. |
| `AUTORUST_MPS` / `AUTORUST_METAL_GEMM` | same | auto (MPS needs `all`) | Metal MPS override. |
| `AUTORUST_CPU_GEMM` | same | auto | CPU host-GEMM override. |
| `AUTORUST_CUBLAS_TF32` / `ALLOW_TF32` | `1`/`true` | off | TF32 cuBLAS (f32 only). |
| `AUTORUST_TENSOR_CORES` / `AUTORUST_TC` | `0`/`false` to disable | on | `mma` kernels. |
| `AUTORUST_ASYNC` | `0`/`false` to disable | on | `cp.async` pipeline. |
| `METAL_FAST_MATH` | any non-`0` | off (safe math) | Metal shader math mode; fast math folds `x + -inf` to `x` and drops attention masks. |
| `BEAM` / `AUTORUST_BEAM` | integer | `0` (off) | BEAM search width. |
| `CACHELEVEL` | `0` to disable | on | Disk cache. |
| `SDPA_TILE` | integer | `0` (off) | Attention padding multiple. |
| `AUTORUST_MM_M/N/K/LM/LN` | five integers | generated | Matmul tile override (ignored unless all five parse). |
| `AUTORUST_CPU_GEMM_MC/NC/KC` | three integers | 48/96/256 | CPU host-GEMM tile (ignored unless all three parse). |

### Dtypes and the RAYON evaluator

| variable | values | default | effect |
|---|---|---|---|
| `DEFAULT_FLOAT` | dtype attr (`float`, `float16`, `bfloat16`, …) | `float` | dtype for weak floats (`dtypes.default_float`). |
| `DEFAULT_INT` | dtype attr (`int`, `int32`, …) | `int` | dtype for weak ints. |
| `SUM_DTYPE` | dtype attr | `float` | Accumulation dtype for float reductions. |
| `EMULATED_DTYPES` | comma-separated dtype attrs | empty | Force dtype decomposition for these dtypes even if the backend advertises support. |
| `RAYON_INTERP` | any | unset | Use the interpreter loop inside `RAYON`. |
| `RAYON_SERIAL` | any | unset | Single-threaded typed evaluator. |
| `RAYON_F64` | any | unset | Bypass the typed evaluator; use the f64 owned-IR evaluator. |
| `RAYON_VERBOSE` | any | unset | Per-program dispatch lines. |

Precedence inside `RAYON`: `RAYON_INTERP` > `RAYON_SERIAL` > `RAYON_F64` >
parallel.

### Toolchains and libraries

| variable | values | default | effect |
|---|---|---|---|
| `RUSTC` | path/name | `rustc` | Compiler used by the `RUST` backend (and by `freeze` emission at build time). |
| `AUTORUST_RUST_CPU` | target-cpu | `native` | `-C target-cpu` for `RUST` kernels. |
| `CC` | path/name | `clang`, else `cc` | C compiler for the `CC` backend. |
| `AUTORUST_CC_ARCH` | arch | `native` | `-march`/`-mcpu` for `CC` kernels. |
| `CUDA_PATH` | directory | `/usr/local/cuda`, `/usr`, `/opt/cuda` include paths | NVRTC include root (`$CUDA_PATH/include`). |
| `AUTORUST_COMPILE_THREADS` | integer | `8` | NVRTC workers per kernel batch on CUDA (`1` = serial). NVRTC gains flatten after a handful of workers. |
| `AUTORUST_NVRTC_LIB` | path/soname | first `libnvrtc.so*` found on the loader path | Force a specific NVRTC library. When set, no other NVRTC is tried. `NVRTC_PATH` is accepted as a tinygrad-compatible alias. |
| `AUTORUST_CL_LIB` | path | `libOpenCL.so` | Load a specific OpenCL library (test/deployment override). |
| `AUTORUST_CUBLAS_LIB` | path/soname | newest `libcublas.so*` found on the loader path | Force a specific cuBLAS library. When set, no other cuBLAS is tried. |
| `AUTORUST_CLBLAST_LIB` | path | `libclblast.so.1`, `.dylib`, `clblast.dll` | Override the CLBlast library. |

### CUDA toolkit versions

NVRTC is loaded at run time and is **independent of the toolkit used to build**
`autorust`. The loader scans `LD_LIBRARY_PATH`/`PATH`, `CUDA_HOME` /
`CUDA_PATH` / `CUDA_ROOT` / `CONDA_PREFIX`, the multiarch library directories,
and every `cuda*` toolkit under `/usr/local` and `/opt`, preferring the newest
`libnvrtc.so[.N]`; it then falls back to bare sonames. A binary built on CUDA 12
therefore uses a CUDA 13 NVRTC on a colleague's machine (and vice versa) as long
as the installed *driver* can load the PTX (PTX is forward compatible; an
NVRTC newer than the driver can emit an ISA the driver rejects, which the probe
below catches).

Before offering `CUDA` to the scheduler, `ops_cuda::device_available` compiles
and loads a one-line kernel for the device's SM (cached once per ordinal). If
NVRTC is missing, too old for the GPU (for example `sm_120` with NVRTC 12.4),
or the driver rejects the PTX, CUDA reports itself unavailable, prints one
warning such as

```
autorust: CUDA device 0 disabled, falling back to the next backend: NVRTC cannot build for sm_86: ...
```

and `DEFAULT_DEVICE` selection continues with `METAL` → `CL` → `RUST` → `CC` →
`RAYON`. An explicit `DEFAULT_DEVICE=CUDA` still selects CUDA, so the compile
error is reported instead of silently running somewhere else.

cuBLAS gets the same treatment: `runtime/libfind` is shared with NVRTC, so the
newest installed `libcublas.so[.N]` / `cublas64_*.dll` is used regardless of
the build toolkit. It is validated by creating and destroying a handle on the
device's current context; if that fails (`CUBLAS_STATUS_ARCH_MISMATCH`, a
driver too old for that cuBLAS, ...) it prints one warning and the scheduler
uses the generated GEMM kernels instead. cuBLAS is only a fast path, so CUDA
itself keeps running.

### Caches

| variable | default | effect |
|---|---|---|
| `AUTORUST_CACHE_DIR` | unset | Root of the on-disk kernel/tuning cache. |
| `XDG_CACHE_HOME` | unset | Used as `$XDG_CACHE_HOME/autorust` when set. |
| `HOME` | unset | Fallback `$HOME/.cache/autorust`. |
| `CUDA_COMPILE_CACHE` | `1024` | In-memory compiled-kernel cache cap (source → PTX). |
| `CL_COMPILE_CACHE` | `1024` | Same for built OpenCL programs. |

The disk cache is content-keyed and can be cleared by deleting the directory (or
`CACHELEVEL=0` disables it per process).

### Diagnostics

All of these are inert unless set; most accept any value.

| variable | output |
|---|---|
| `AUTORUST_PROF` (any value except `0`/empty) | Scheduler phase table; also enabled by `TG_RW_TRACE`/`TG_RW_VISITS`. |
| `AUTORUST_KPROF` | Per-program kernel time table: `KPROF <name> <secs> <program-key>`. CUDA synchronises per launch (so its timings are profiling-only); host backends (`RUST`/`CC`/`RAYON`) print generated kernels and `HostGemm` calls. |
| `AUTORUST_KERNEL_INFO` | Registers/shared/local bytes per compiled CUDA kernel. |
| `FROZEN_PROF` | Per-kernel and host-GEMM counters in a `FrozenRunner` (`report_profile()`). |
| `AUTORUST_CUBLAS_DEBUG` | Host-GEMM dispatch/descriptor lines. |
| `AUTORUST_CPU_GEMM_DEBUG` | CPU host-GEMM packing/tile decisions. |
| `AUTORUST_MM_DEBUG` | Generated matmul tiling decisions and `host_gemm::extract` rejection reasons. |
| `AUTORUST_DEVICE_VERBOSE` | Per-launch `RUST` backend lines (grid, args, buffer count). |
| `AUTORUST_RUST_DUMP`, `AUTORUST_CC_DUMP` | Print the compiled Rust/C kernel source. |
| `AUTORUST_ATTN_DEBUG` | One-time line describing the selected fused-attention kernel. |
| `AUTORUST_COMPILE_DEBUG` | Per-batch CUDA compile stats (`srcs`, unique sources, threads, ms). |
| `TG_RW_TRACE`, `TG_RW_VISITS` | Rule-rewrite trace / visit histogram (also enables `AUTORUST_PROF`). |
| `TG_DEBUG_RANGEIFY` | Rangeify debug output. |

Feature-gated dumps:

| variable | feature | output |
|---|---|---|
| `TG_TRACE` | `trace` | One line per rule firing (pattern, ops, rewritten uop). |
| `TG_DUMP_PRE_EXPAND` | `dump` | Pre-expander AST; must be `1`. |
| `TG_DUMP_IDX` | `dump` | Indexed kernel AST. |
| `TG_DUMP_LST` | `dump` | Linearized program listing. |
| `TG_DUMP_SRC` | `dump` | Rendered kernel source. |
| `TG_DUMP_INTERP` | `dump` | Interpreter program. |
| `TG_DEBUG_LOAD` | `dump` | Interpreter loads. |
| `TG_M5_STAGES` | `dump` | Truncate the pipeline after `N` stages (debug). |

### BEAM search and tensor-core tuning

Defaults match the port; normally leave them alone.

| variable | default | effect |
|---|---|---|
| `BEAM_UPCAST_MAX` | `256` | Maximum upcast factor. |
| `BEAM_LOCAL_MAX` | `1024` | Maximum local (shared-memory) axis. |
| `BEAM_MIN_PROGRESS` | `1` (scaled by `1e-6`) | Minimum improvement to keep a candidate. |
| `BEAM_UOPS_MAX` | `3000` | Maximum uops per candidate. |
| `BEAM_MAX_CANDIDATES` | `0` (unlimited) | Port-only safety cap on candidates compiled per round. |
| `BEAM_TEST_GLOBAL_MAX` | `65536` | Cap on the timing launch's global size. |
| `IGNORE_BEAM_CACHE` | unset | Re-run the search instead of using cached results. |
| `BEAM_STRICT_MODE` | unset | Panic instead of skipping a failing candidate. |
| `BEAM_VERBOSE`, `BEAM_DEBUG` | unset | Progress / debug output. |
| `BEAM_DUMP_BASE`, `BEAM_DUMP_SRC`, `BEAM_UNINIT_BUFS` | unset | Candidate dumps / uninitialised-buffer handling. |
| `USE_TC` | `1` | Try the tensor-core path for generated matmuls. |
| `TC_OPT`, `TC_SELECT` | `0`, `-1` | Tensor-core layout selection overrides. |
| `TC_DEBUG` | unset | Tensor-core decision lines. |

## Not environment variables

- **Training mode** is a scoped flag, not an env var:
  `helpers::context::set_training(true)`, the RAII
  `TrainingGuard::new(true)`, or `Session::training(true)`.
- **Device override** at run time: `autorust::set_default_device("RAYON")` or
  pass the device to the constructor/`Tensor::to`.
- **Benchmark knobs** such as `JIT`, `ITERS`, `SYNC`, `NOSYNC`, `BIAS` belong to
  the `examples/*` programs, not the library; see
  [`PERFORMANCE.md`](PERFORMANCE.md#measuring).
