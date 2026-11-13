# AGENTS.md — working in `autorust`

`autorust` is a standalone multi-backend tensor / ML framework in Rust. It began
as an automated port of tinygrad v0.14 and still shares that lineage — the
interned UOp graph, rangeify scheduler, codegen pipeline and interpreter all
descend from it — but it is **no longer a port**. The `Tensor`/`Ctx`/`Session`
API, the Vulkan/SPIR-V backend, frozen CPU plans, library-GEMM integration,
hand-written fused kernels, the ONNX runner and the model library are this
project's own, and deliberate divergences are recorded in `CURRENT_STATE.md`.

`../tinygrad/` is a **reference and a differential oracle, not a specification**.
Read it to understand where a piece of code came from or to compare behaviour,
but do not assume upstream's current code is the answer for this repository.

Start here, then read **`CURRENT_STATE.md`** (canonical status, dated design
decisions, measurements and known gaps) and **`README.md`** (what users see). Do
not duplicate those documents here; point at them.

## Working principles

1. **Optimise for this framework, not for parity.**
   - If code is still recognizably derived from a Python file/function, say so in
     a doc comment — the existing tree does this constantly.
   - When you intentionally diverge from tinygrad, record the reason and the
     evidence in `CURRENT_STATE.md`. When a fast path diverges, keep the general
     path correct and working.
   - Do not refactor working, tinygrad-derived lowering code just to make it more
     idiomatic. Do refactor when it removes a demonstrated bug or a measured
     bottleneck.
   - Upstream fixes are worth reading, but port the selected fix, not the file.
     New work should follow this project's architecture (`Ctx`/`Session`,
     `Settings`, backend seams, `HostGemm`, fused-kernel dispatch) rather than
     mirroring Python structure.
2. **Keep the user-facing contract stable and legible.**
   - Stable public modules: `tensor` (including `nn`, `optim`, `grad`, `state`),
     `models`, `onnx`, `dtype`, `helpers`, `engine`, `runtime`, `renderer`,
     `settings`. `uop`, `schedule` and `codegen` are public for backend authors
     and tooling, but explicitly unstable in 0.1.x.
   - `examples/*.rs` are part of the documentation. Keep them runnable and keep
     their documented env knobs (`EPOCHS`, `LR`, `SEED`, `FAST`, `JIT`, …) working.
     A reader should be able to learn the framework from them.
   - Every public item is documented (`#![warn(missing_docs)]` plus the rustdoc
     gate below). Pre-1.0 breaking changes are allowed, but call them out in
     `CHANGELOG.md`.
   - Most tensor/device misuse panics with a `TGERR:` message (matching the
     front-end's existing style); fallible I/O and parsing entry points
     (safetensors, ONNX, model/state loading) return `Result`. Preserve that split.
   - Rust never imports Python at runtime. Everything under `py/` is a
     development-only bridge, oracle and golden generator.
3. **Prove changes with tests and measurements.**
   - Every new op, kernel, backend or public behaviour gets a test, normally in
     `tests/` (module tests near the code are fine for internals). When numerics
     are involved, compare against `INTERP` or a host/f64 reference; compare
     against tinygrad only when tinygrad itself is the useful oracle.
   - The interpreter (`src/runtime/ops_interp.rs`) is the reference semantics for
     every backend. New operations must work there too, not only on the fast path.
   - Performance claims need a before/after measurement on named hardware and
     stated shapes. Put user-visible numbers in `PERFORMANCE.md` and the raw data
     in `bench/`; do not land an "optimisation" justified by a single timing.
   - Optional capabilities (tensor cores, cooperative matrices, library GEMM,
     a broken vendor driver) must be probed and must degrade to the generated or
     scalar path, never panic or silently change numerics.

## Dependencies and portability

- The core is **std-only Rust plus `nanoserde`**. That is a feature: keep the
  mandatory dependency set minimal.
- Backend crates are the sanctioned exception, each behind its own feature:
  `rayon`/`bytemuck` (`backend_rayon`), `libloading` (`CL`/`RUST`/`CC`/CUDA),
  `cudarc` (`backend_cuda`), `objc2*` (`backend_metal`), `ash` (`backend_vk`).
  No GPU backend needs a vendor SDK at build time: CL/CUDA/VK load their driver
  at run time, and Metal links only OS-provided frameworks.
- `--no-default-features` must keep compiling as a std-only build (plus
  `nanoserde`). That is a gate, not an aspiration.
- `--all-features` must compile on Linux/Windows even though Metal is macOS-only:
  keep platform-specific dependencies target-gated.
- The published crate is an `rlib` with no C ABI and no `cdylib`. The Python
  bridge is the separate, unpublished `autorust-bridge` crate under `py/bridge`.

## Commits (required)

- **Always commit when you finish a task.** Do not leave a completed task's work
  uncommitted; make a focused commit for it.
- Follow the existing history style: conventional-commit subjects
  (`feat(scope): …`, `fix(scope): …`, `docs: …`, `perf(vk): …`, `refactor(…): …`),
  present tense, a short imperative subject. Commit each coherent task separately.
- **Never amend commits.** If a change would require rewriting pushed history,
  make a new commit instead. Do not force-push over published commits.

## Keeping docs current (required)

- **`CURRENT_STATE.md`** is the canonical record: status, dated design decisions,
  measurements, rejected approaches and known gaps. Add a dated section for any
  change that alters status, a gate, a default, or a performance story. It keeps
  its original "port plan" title and milestone history; the newest dated
  sections under §8/§9 are the current record.
- **`CHANGELOG.md`** is for user-visible changes, in Keep a Changelog form, under
  `[Unreleased]`. Record additions, fixes and measured performance effects.
- Update the focused document when its subject changes:
  - `CONFIGURATION.md` — new features, `Settings` fields, env vars, defaults.
  - `PERFORMANCE.md` — benchmark methodology, kernel internals, device selection,
    any claim that a user might rely on.
  - `ONNX-Compat.md` — ONNX operator/feature coverage (the canonical matrix).
  - `FREEZE.md` — frozen-plan capture/emit/runner behaviour.
  - `INDEXING.md` — the `getitem` index DSL.
- **Keep this `AGENTS.md` in sync** when layout, build commands, policies or the
  working principles change. The docs must not drift.

## Layout

```
src/uop/        interned UOp graph; UPat/PatternMatcher; graph_rewrite; symbolic;
                arg/flat JSON codec; weak/dtype lowering; sint/range
src/dtype/      DType/AddrSpace, casts, truncation, least_upper_dtype
src/helpers/    read-once env/context, base64, bigint
src/settings.rs Settings (attention kind, library GEMM, tile sizes, TC, cache),
                process defaults and per-Ctx overrides
src/tensor/     Tensor front-end + Ctx, movement/elementwise/reduce, indexing,
                grad, rand, nn layers, optimizers, state/safetensors, marshalling
src/models/     reference models: transformer, ViT, FlashAttention dispatch
src/onnx/       std-only protobuf parser + T0/T1 op runner
src/schedule/   rangeify/indexing/memory, library-GEMM seam (Seam B)
src/codegen/    to_program pipeline, linearize, simplify, coalesce, BEAM search
src/renderer/   Renderer trait, cstyle (C/CL/CUDA/Metal/Rust), SPIR-V (VK),
                tensor-core tables
src/runtime/    device traits + allocators + programs for INTERP, RAYON, RUST,
                CC, CL, CUDA, METAL and VK; interpreter execution; autotune;
                fused attention; library GEMM (cuBLAS/CLBlast/MPS/packed CPU/
                Vulkan coopmat)
src/engine/     realize, Session (BufTable + ProgCache + Jit capture), Jit,
                disk cache, profiling
src/freeze/     AOT frozen CPU plan capture/emit/runner
src/bin/        per-backend smoke probes (ocl/cuda/metal/vk_smoke)
examples/       runnable demos (xor, mnist, vit_mnist, vit_bench, gemm_bench,
                fa_bench, rayon_bench, cublas_check, vit_freeze; frozen_vit/ is
                a standalone crate)
tests/          Rust unit/differential/golden/backend/nn/model/ONNX tests
bench/          dev-only cross-framework benchmark (autorust/candle/torch/tinygrad),
                excluded from the published crate
py/             dev-only Python bridge + differential gates (not published)
    bridge.py       UOp JSON round trip / oracle helpers
    bridge/         cdylib shim + oracle bins; build with
                    `cargo build -p autorust-bridge`
    tests/          pytest differential gates against ../tinygrad
    dump_*.py       regenerate goldens after editing either side
CURRENT_STATE.md   canonical status, decisions, measurements, known gaps
CHANGELOG.md       user-visible release notes
README.md          user-facing overview and quick start
PERFORMANCE.md     benchmarks, GEMM/attention internals, device selection
CONFIGURATION.md   feature flags, Settings, full env-var reference
FREEZE.md          AOT frozen CPU plans (build.rs/include! flow)
INDEXING.md        the getitem index-DSL reference
ONNX-Compat.md     ONNX operator/feature matrix
../tinygrad/       reference oracle (do not modify; dev/test only)
```

## Build / test

Run from this directory (`autorust/`):

- `cargo build` — builds the default workspace members (the `autorust` library
  and its smoke binaries), not the Python bridge or `bench/`. Warning-free.
- `cargo test` — the Rust suite. Some tests are `#[ignore]`d because they download
  data, need an external ONNX checkpoint, or are slow end-to-end gates; run them
  explicitly with `-- --ignored` when relevant. On macOS, Apple's OpenCL runtime
  can fail `flash_cl`; `AUTORUST_CL_LIB=/nonexistent` skips CL for a clean run.
- `cargo clippy --all-targets` — 0 errors, 0 warnings. `mutable_key_type` is
  allowed at the crate root (`UopId` hashes by immutable id); oracle-derived
  functions with many arguments carry targeted allows, not crate-wide ones.
- `cargo rustdoc --lib -- -W missing_docs` — 0 warnings. Keep it that way when
  adding public API.
- `cargo check --no-default-features` — the std-only gate.
- `cargo run --release --example xor` — quick end-to-end demo.
- Python differential gates: `../tinygrad/.venv/bin/python -m pytest py/tests -q`
  (they also run as plain scripts). Regenerate goldens with
  `../tinygrad/.venv/bin/python py/dump_<subject>.py`. The bridge takes JSON on
  batch paths and bytes on the per-kernel path; build it with
  `cargo build -p autorust-bridge`.
- `bench/` is the cross-framework harness; see `bench/README.md` for the protocol
  and `bench/run.py` for invocation. Do not add it to the published crate.
- Format the files you touch (`rustfmt <file>` or `cargo fmt`). The whole
  workspace is not currently format-clean (VK and `bench/` files have pre-existing
  diffs), so do not reformat unrelated code.

Features: `backend_rayon`, `backend_rust`, `backend_clang`, `backend_cl`,
`backend_vk`, `backend_metal`, `backend_cuda` and `freeze` are all **default**.
`backend_cuda` is target-gated to Linux/Windows; `backend_metal` is macOS-only;
`freeze` implies `backend_rust`. `trace` and `dump` are debug-only, off by
default, and compiled out so hot paths never call `getenv`. `--no-default-features`
keeps the `INTERP`/`CPU` device available.

## Conventions

- **`Ctx` owns the interned UOp arena; `Tensor` is a cheap handle to it;
  `Session` owns realize state** (buffers, compiled-program cache, symbolic
  values, JIT capture, training flag). Create a `Ctx` and a `Session` once and
  reuse them across steps — a fresh `Session` per step recompiles and reallocates
  everything.
- `Tensor` and `Ctx` are `Rc`-based and `!Send`/`!Sync`; the front-end is
  single-threaded by design. CPU backends parallelize inside their kernels.
  A `Session` is scoped to the arena whose ids populated its buffer table; its
  compiled-program cache is content-keyed and safe across arenas.
- Configuration is read **once** from the environment by `helpers::env`. Change
  policy per context through `ctx.borrow_mut().settings`, not by mutating the
  environment later. An already-compiled kernel is not re-lowered when settings
  change; start a fresh `Session`/`Ctx` if you need that.
- Device selection is process-wide (`DEFAULT_DEVICE`, with `auto` running the
  benchmark selector). Backends register through `register_device` /
  `register_device_family` and must not require a vendor SDK at build time.
- Never reimplement Python's `str()`/`repr` in Rust. Cross-graph identity is
  checked Python-side through the bridge; Rust-side caches use structural or
  content hashes.
- For a new backend, implement the `Renderer`/runtime seams, register the
  device, add a smoke/conformance test, and wire optional fast paths behind a
  probe. Hand-written kernels must agree with the generated kernel; feature
  detection must fall back when a driver lacks the extension.
- When a behavioural subtlety comes from the Python lineage (argument order,
  float semantics, iteration order), document it in `CURRENT_STATE.md` and pin it
  with a test.
- Follow `../tinygrad/AGENTS.md` before touching the oracle tree (you normally
  should not modify it).

## Before you call it done

- `cargo build`, the relevant `cargo test` targets, `cargo clippy --all-targets`
  and `cargo rustdoc --lib -- -W missing_docs` are clean.
- New public items have docs; new knobs are in `CONFIGURATION.md`; user-visible
  changes are in `CHANGELOG.md`; design/measurement findings are in
  `CURRENT_STATE.md`.
- Touched examples still run, and a new backend/kernel has a test plus a working
  fallback.
- One focused commit with a conventional subject.
