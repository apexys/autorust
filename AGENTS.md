# AGENTS.md — working in `autorust`

This repo is a **faithful Rust port of tinygrad** (v0.14.0). The port lives in
`autorust/`; the reference Python tree is `../tinygrad/` (an unmodified checkout
used as the oracle). Start here, then read **`CURRENT_STATE.md`** — it is the
single source of truth for what is done, what is next, and the design decisions
behind each milestone. Do not duplicate that document here; point at it.

## The two rules that matter most

1. **Be faithful to tinygrad.** This is a mechanical port, not a redesign. When
   porting an op/layer/optimizer, read the corresponding file under
   `../tinygrad/tinygrad/…` and mirror its logic, argument order, and semantics
   one-for-one (modulo Rust typing). Consult `CURRENT_STATE.md` §1 for the ground
   rules, especially the **std-only + nanoserde** dependency policy and the
   "Rust never imports Python at runtime" rule.
2. **Preserve the end-user feel.** The public Rust API and especially the
   `examples/*.rs` programs should stay recognizably close to their tinygrad
   equivalents (`../tinygrad/examples/…`). A reader who knows the tinygrad version
   should be able to follow the Rust version line-by-line. Prefer tinygrad's
   naming, structure, and flow over Rust-idiomatic restructuring when the two
   conflict. This is a feature of the port, not an afterthought.

## Commits (required)

- **Always commit when you finish a task.** Do not leave a completed task's work
  uncommitted; make a focused commit for it.
- Follow the existing history style: conventional-commit subjects
  (`feat(scope): …`, `fix(scope): …`, `docs: …`, `refactor(…): …`), present tense,
  a short imperative subject. Commit each coherent task separately.
- **Never amend commits.** If a change would require rewriting pushed history, make
  a new commit instead. Do not force-push over published commits.

## Keeping docs current (required)

- **Keep `CURRENT_STATE.md` up to date** as part of any change that alters status,
  a gate, a design decision, or a known gap. It is the canonical record.
- **If an important fact in this `AGENTS.md` changes, edit this file too** (build
  commands, layout, conventions, commit policy, the two rules above). The two docs
  must not drift.

## Layout

```
src/uop/        interned UOp graph, UPat/PatternMatcher, graph_rewrite, symbolic
src/dtype/      DType/AddrSpace, casts, truncation, least_upper_dtype
src/tensor/     Tensor front-end, grad, rand, nn, optim, marshal helpers
src/onnx/       ONNX importer: std-only protobuf parser + T0/T1 op runner
src/schedule/   rangeify/indexing/memory (Seam B)
src/codegen/    to_program pipeline, linearize, simplify, coalesce, interp exec
src/renderer/   Renderer trait + cstyle (OpenCL under `backend_cl`)
src/runtime/    Buffer/allocator/compiler/device traits, INTERP + CL
src/engine/     realize, session (BufTable + ProgCache), jit (TinyJit), disk_cache, ProgramInfo
src/bin/        oracle_* cargo bins driven by pytest (JSON graph files)
examples/       runnable demos (xor.rs, mnist.rs) — mirror tinygrad
tests/onnx.rs   ONNX parser/op tests; external checkpoint test is #[ignore]d
tests/          Rust unit/differential/golden/nn tests
py/             dev-only Python bridge + differential gates
    bridge.py       uop_to_json / json_to_uop / realize_in_tg
    tests/          test_*_diff.py gates against the tinygrad tree
CURRENT_STATE.md   the canonical status/plan document
INDEXING.md        the getitem index-DSL reference
../tinygrad/       the reference oracle (do not modify)
```

## Build / test

Run from this directory (`autorust/`):

- `cargo build` — should be warning-free.
- `cargo clippy --all-targets` — 0 errors (`mutable_key_type` is allowed at the
  crate root; a few minor style lints are left as-is).
- `cargo test` — the Rust suite (some tests `#[ignore]`d as explicit M7 gates).
- `cargo run --release --example xor` — runnable demo; knobs via env vars
  (`EPOCHS`, `LR`, `SEED`).
- Python differential gates live in `py/tests/` and run against the tinygrad tree;
  pytest is available at `../tinygrad/.venv/bin/python -m pytest`, and the gates
  also run as plain scripts. Regenerate goldens with
  `../tinygrad/.venv/bin/python py/dump_golden.py` after editing either side.

Feature flags: `backend_cl` (OpenCL via `opencl3`), `capi` (default; the C ABI
loaded by the Python bridge), `trace`/`dump` (debug-only, off by default).

## Conventions

- `Tensor` is a handle to an interned `UopId` (`Rc<Node>`), not a raw index. Cloning
  is cheap; parameters are buffer-backed leaves so optimizers can update in place
  via `Tensor::assign`.
- Never reimplement Python `str()`/`repr` in Rust; cross-graph identity is checked
  Python-side through the bridge. Rust-side caches use a structural hash.
- When a Python-side rewrite/oracle behavior has a subtlety (argument order, float
  semantics, iteration order), document the finding in `CURRENT_STATE.md` and pin it
  with a test.
- Follow `../tinygrad/AGENTS.md` for the oracle tree itself (test flags, lint/typecheck).
