# ONNX export

This file is the contract for the ONNX **exporter**. The importer and its
operator matrix live in `ONNX-Compat.md`; this document covers producing ONNX
models from `autorust` graphs and models.

**Status (2026-11-16):** design and roadmap, not implemented. No exporter code
exists yet. The tiers below are the intended order of work.

## Goals

- Export an inference graph built from the public `Tensor`/`nn` API to a valid
  ONNX model that loads in onnxruntime/torch and matches the source numerics.
- Make failures explicit: an unsupported op or attribute is an error with the
  node path, never a silently incorrect graph.
- Work before graph realization: export the lazy `Tensor`/UOp graph, not a
  compiled or fused backend plan.
- Keep the exporter in the core crate with no new mandatory dependencies
  (std-only protobuf writing, like the importer).

## Non-goals (initially)

- Exporting a realized `Session`/JIT plan or backend-specific fused kernels.
- Data-dependent control flow (`If`/`Loop`/`Scan`). `If` follows the importer's
  T2-A work; `Loop`/`Scan` remain T2-C and are not exporter targets yet.
- Training graphs (`Gradient`, optimizer updates) and stateful optimizer export.
- Quantized/QDQ export (tied to the INT8 W8A8 design).
- Guaranteeing bit-identical numeric output to onnxruntime; the gate is a
  numerical tolerance on the source graph.
- Exporting arbitrary custom/backend-specific ops. These fail with
  `UnsupportedOp`; a custom-domain debug escape hatch is an open question.

## Placement

```
src/onnx/
  mod.rs            # pub mod export;
  pb.rs             # importer wire reader (unchanged)
  write.rs          # std-only protobuf writer helpers
  export/
    mod.rs          # public API, InputSpec/OutputSpec, ExportOptions, ExportError
    builder.rs      # ModelProto/GraphProto builder (E0)
    trace.rs        # UOp capture, freeze, naming, topological order
    lower.rs        # UOp -> ONNX node/attribute mapping
    module.rs       # nn::Module / named_params convenience
    opset.rs        # per-op minimum opset and decompositions
tests/onnx_export.rs
examples/export_onnx.rs
ONNX-Export.md      # this file
```

The exporter belongs in the core crate:

- It needs `Ctx`/`Tensor`/`UOp` and `nn::Module`/`named_params`; a separate crate
  would duplicate or depend on unstable internals and create version skew.
- The protobuf writer uses the same std-only wire format as `pb.rs`; no `prost`
  or other dependency.
- It works in the `--no-default-features` build (std-only).
- A thin `autorust-onnx-export` facade can be added later if compile time or
  binary size becomes a problem.

`ONNX-Compat.md` stays the importer contract. The two documents share the
protobuf types and error types where useful, but the op tables are indexed by
different directions (importer: ONNX op -> handler; exporter: UOp -> ONNX node).

## Export tiers

The tiers separate **graph source** from **shape and control-flow
complexity**. A higher tier includes the lower tiers.

| Tier | Source | Shapes | Scope | Status |
|---|---|---|---|---|
| **E0** | Manual `GraphBuilder` | any | Protobuf/model writer: nodes, attributes, tensors, initializers, value info, opset/IR metadata, external data | planned |
| **E1** | `export_fn` closure | fixed | Trace a lazy `Tensor`/UOp graph; params/constants as initializers; dedup; core standard ops | planned |
| **E2** | `export_module` + `Module` | fixed | `named_params`/`named_buffers`, shared parameters, multiple inputs/outputs, eval mode, metadata | planned |
| **E3** | E1/E2 | symbolic/dynamic | `Dim::Param`, shape inference, opset targeting/decomposition, constant folding, external data | planned |
| **E4** | E1/E2 | data-dependent | `If` (later `Loop`/`Scan`), training, QDQ/quantized, sequences/optionals, custom domains | deferred |

### E0 — proto and graph builder

The mechanical foundation:

- `ModelProto`, `GraphProto`, `Node`, `Attribute`, `TensorProto`, `ValueInfo`.
- Initializers, inputs/outputs, opset imports, IR version, producer metadata.
- `GraphBuilder` so callers can construct a graph node by node without tracing.
- External-data sidecars for large initializers.

E0 is not the primary user API, but it is essential for importer reverse tests
and for compatibility fixtures.

### E1 — static trace

The `torch.onnx.export`-equivalent path:

```rust
pub fn export_fn<F>(
    path: impl AsRef<Path>,
    inputs: &[InputSpec],
    outputs: &[OutputSpec],
    f: F,
    opts: &ExportOptions,
) -> Result<(), ExportError>
where
    F: FnOnce(&[Tensor]) -> Vec<Tensor>;
```

Rules:

- Run the forward while the lazy UOp graph still exists. Do not export after
  `realize` (the graph is then a buffer leaf or a fused plan).
- Register which `Tensor`/`UopId` are graph inputs. Other leaves with host
  initial data become initializers; a volatile/runtime-only leaf is an error.
- Emit only the subgraph reachable from the declared outputs, in topological
  order.
- Reject side effects (`assign`/`store`), in-place mutation, and unseeded random
  ops for inference export.
- Every UOp must lower to a standard ONNX node or a documented decomposition;
  otherwise return `UnsupportedOp` with the node path.

### E2 — module export

A convenience layer over E1:

```rust
pub fn export_module<M, F>(
    path: impl AsRef<Path>,
    model: &M,
    inputs: &[InputSpec],
    forward: F,
    opts: &ExportOptions,
) -> Result<(), ExportError>
where
    M: Module,
    F: FnOnce(&M, &[Tensor]) -> Vec<Tensor>;
```

- Initializer names come from `named_params()`/`named_buffers()`.
- Shared parameter tensors are emitted once and referenced by name.
- Multiple inputs and outputs are supported.
- Model metadata (producer, version, doc string) is configurable.
- `ModuleForward` can get a convenience adapter later; the explicit closure
  keeps the API usable for modules whose `forward` has a non-trivial input type.

### E3 — dynamic axes and deployment polish

This is where most exporter fragility lives:

- `InputSpec` with `Dim::Val` or `Dim::Param`; symbolic batch/sequence axes.
- Output `value_info` from the traced graph; intermediate value info is optional.
- Shape inference sufficient for onnxruntime, or explicit shape subgraphs where
  the traced graph already contains them.
- Opset targeting and decomposition (see below).
- Constant folding and cleanup: identities, contiguous/copy nodes, trivial
  reshapes, initializer deduplication, deterministic names.
- External data for large models, with inline-versus-sidecar policy.
- Strict mode (standard ops only) as the default; any custom-domain escape
  hatch is debug-only and clearly non-portable.

### E4 — advanced

Deferred until the importer has the corresponding features:

- `If` subgraphs (after importer T2-A), then `Loop`/`Scan` if ever.
- Training graphs and optimizer state.
- Quantized/QDQ export (INT8 W8A8 first).
- Sequence/optional values and arbitrary custom domains.
- Data-dependent output shapes.

## Public API sketch

```rust
pub struct InputSpec {
    pub name: String,
    pub dtype: DTypeId,
    pub shape: Vec<Dim>, // Dim::Val or Dim::Param
}

pub struct OutputSpec {
    pub name: String,
    pub dtype: Option<DTypeId>, // inferred when None
    pub shape: Option<Vec<Dim>>,
}

pub struct ExportOptions {
    pub opset: i64,           // default 17
    pub ir_version: i64,      // default 8, paired with opset 17
    pub producer: String,     // default "autorust"
    pub external_data: ExternalData, // threshold/path policy
    pub strict: bool,         // standard ops only (default true)
    pub fold_constants: bool, // default true
}

pub enum ExternalData {
    Never,
    AboveBytes(u64), // default 2 GiB in a future release
}

pub struct GraphBuilder { /* E0 */ }
```

`ExportError` is export-specific so diagnostics can carry the node and op path:

```rust
pub enum ExportError {
    Io(String),
    UnsupportedOp { op: String, path: String },
    UnsupportedAttribute { op: String, attr: String },
    DynamicControlFlow(String),
    MissingInitializer(String),
    Shape(String),
    Internal(String),
}
```

`OnnxError` remains the importer/runner error type. Conversion between the two
is explicit; export errors should not be reported as parse errors.

## Graph capture model

The stable capture point is the **lazy `Tensor` graph before any realize**:

1. Build inputs from `InputSpec` in the model's `Ctx`.
2. Run the user forward, producing output `Tensor`s.
3. Walk `outputs -> sources` with `ctx.toposort` to collect the reachable graph.
4. Classify leaves:
   - registered inputs -> `GraphProto.input`;
   - parameters/buffers with host data -> initializers;
   - `Const`/constant-folded subtrees -> initializers;
   - volatile leaves or buffers without exportable data -> `MissingInitializer`.
5. Lower nodes in topological order and assign stable names.
6. Emit only inputs, outputs, initializers and nodes; add intermediate
   `value_info` only when it materially helps the consumer.

Do not use `Tensor::realized_buffer()` as the export source. A realized tensor
is already a storage leaf (or a fused plan); re-run the forward in a fresh
`Ctx`/`Session` if the graph has been realized.

## Lowering and normalization

A direct UOp trace is correct but can be a poor ONNX graph. The exporter should
normalize before lowering:

- `nn::Embedding` / one-hot + sum -> `Gather` (otherwise a vocabulary-sized
  one-hot is exported).
- `MatMul` + bias + activation patterns -> `Gemm` where it is exact and helps.
- `Contiguous`/`Copy`/trivial reshapes -> identity or removal.
- Fused/primitive ops with standard decompositions (e.g. layer norm, attention)
  -> primitive ONNX subgraphs unless a newer opset op is explicitly requested.
- `Reshape` with static targets -> shapes; dynamic targets -> the traced
  `Shape`/`Gather`/`Concat` subgraph.

The lowering table is exhaustive over the UOps that can reach export. There is
no silent fallback: an unmapped UOp is an `UnsupportedOp` error with the node
path. Layout conversions (conv weight layout, `Gemm` `transA`/`transB`,
broadcast axes) are part of lowering and need round-trip tests.

## Initial op coverage

E1 should cover the inference subset already exercised by the importer:

| Family | Initial coverage |
|---|---|
| Elementwise / logical | `Add`, `Sub`, `Mul`, `Div`, `Pow`, `Max`, `Min`, `Where`, comparisons, logical ops, casts, unary math |
| Reductions | `ReduceSum`/`Mean`/`Max`/`Min`/`Prod`, `ArgMax`/`ArgMin`, `CumSum`, `TopK` |
| Movement | `Reshape`, `Transpose`, `Squeeze`, `Unsqueeze`, `Concat`, `Split`, `Slice`, `Expand`, `Tile`, `Pad` |
| Indexing | `Gather`; `GatherND`/`GatherElements` where the traced form is unambiguous; simple `Scatter*` only if semantics are exact |
| NN | `Conv`, `ConvTranspose`, `MatMul`, `Gemm`, pooling, `BatchNormalization`, `LayerNormalization`, `GroupNormalization`, `Softmax`, `LogSoftmax`, activation compositions |
| Resize | linear/nearest/cubic forms that map to `Resize`; unsupported attributes error |
| Recurrent/control flow | none initially (E4) |
| Random | none initially; seeded random is E4/depends on the ONNX random-op policy |

`nn` modules are traced through their tensor operations; the exporter does not
serialize Rust module structure. Module-specific normalization is what turns
`Embedding` and similar layers into compact ONNX nodes.

## Opset and IR policy

- Default pair: **opset 17 / IR 8** (broadly supported by onnxruntime and
  torch).
- `opset` is an explicit option; the exporter does not silently raise it for a
  single op.
- `export/opset.rs` holds the per-op minimum and optional decomposition:
  - `LayerNormalization` requires opset 17; below that, decompose or error.
  - `GroupNormalization` requires opset 18; below that, decompose or error.
  - `Attention`/`RotaryEmbedding` require newer opsets (23+); decompose to
    standard MatMul/Softmax/Slice unless the user opts in.
  - Attribute-level minimums (`Resize` antialias/aspect-ratio policies, newer
    `Softmax` axis semantics) are handled at the same layer.
- The exporter should reject an op/attribute combination it cannot express at
  the requested opset rather than produce a model that only one runtime accepts.

## Shapes and dynamic axes

- `Dim::Val` for static dimensions; `Dim::Param` for symbolic ones (batch,
  sequence length).
- Inputs and outputs get `value_info` with the declared/inferred shapes.
- Static trace (E1/E2) requires all declared input dimensions to be concrete;
  otherwise the user must opt into E3.
- E3 supports symbolic axes at runtime, but the graph must still be a static
  DAG: data-dependent output sizes (`NonZero`, dynamic `Loop` scan outputs) are
  out of scope.
- Shape arithmetic already present in the graph (`Shape`/`Gather`/`Concat`/
  `Reshape`) is preserved; a later constant-folding pass may reduce it when the
  relevant inputs are static.

## Initializers, external data, names

- Parameters and buffers are read once and written as `TensorProto` raw data.
- Deduplicate by storage identity: a shared parameter becomes one initializer
  and multiple consumers.
- Deterministic names:
  - inputs/outputs: declared names;
  - params: `param/<named_params name>` (sanitized for uniqueness, not for
    ONNX validity, which allows most strings);
  - intermediates: `node/<index>_<Op>`.
- Avoid empty or duplicate names; fail on collisions rather than silently
  renaming a graph that a consumer may rely on.
- External data:
  - inline small tensors; write a sidecar for large models;
  - one file or per-tensor files is an option, but the default is a single
    sidecar next to the model;
  - validate offsets/lengths and cap total bytes, mirroring the importer's
    limits.
- Metadata: producer name/version, model doc string, opset imports, IR version.

## Validation gates

Every exporter level must pass the gates below. The first two are always in CI;
the ORT/third-party gates are dev-only tools, not crate dependencies.

| Gate | Check |
|---|---|
| **V0** | Export -> our `OnnxModel` -> run -> compare against the source `Tensor` result |
| **V1** | `onnx.checker.check_model` accepts the file |
| **V2** | onnxruntime runs the model and matches the source within tolerance |
| **V3** | a third-party import (torch/Netron) or a second runtime accepts it |
| **V4** | Dynamic axes, external data, and opset-target variants behave |

Tests:

- E0: hand-built `Add`/`MatMul`; parse with the importer and run.
- E1: MLP, CNN (conv/pool/batchnorm), transformer block
  (matmul/softmax/layer norm/reshape/transpose), embedding/Gather.
- E2: module with shared parameters, named buffers, multiple outputs.
- E3: dynamic batch/sequence through ORT; opset 13 decomposition; forced
  external-data threshold; name-collision and unsupported-op errors.
- Round-trip fidelity: fp32, fp16, bf16; reject unsupported f8/dtypes clearly.
- Error cases: side effects, mutation, dynamic control flow, volatile leaves.

The dev-only ORT script lives under `bench/` or `py/`; the crate itself must not
depend on onnxruntime.

## Fragility and mitigations

ONNX exporters are fragile for predictable reasons. The contract handles them
explicitly:

| Failure mode | Mitigation |
|---|---|
| Tracing sees only executed control flow | Reject data-dependent branches; E4 adds `If`; never silently bake a runtime branch |
| Graph already realized/fused | Capture the lazy UOp graph; document that export must happen before `realize` |
| Silent op mismatch | Exhaustive lowering table; `UnsupportedOp` with node path |
| Opset drift | Explicit opset option + per-op minimum/decomposition table |
| Bloated graphs (one-hot embedding) | Normalization patterns before lowering |
| Shared/aliased parameters | Deduplicate by storage identity |
| Layout/dtype mistakes | Round-trip + ORT numerical tests for conv/gemm/resize/norm |
| Non-portable custom ops | Strict standard-op mode; custom domains are debug-only |
| Dynamic shapes | Symbolic `Dim::Param` only; data-dependent shapes are an error |
| Random/training side effects | Reject by default; explicit seeded/training modes later |
| Large models | External data, size caps, single sidecar by default |

## MVP sequence

1. **Foundation (E0):** `write.rs`, `GraphBuilder`, hand-built models, importer
   round-trip and `onnx.checker`.
2. **Static trace (E1):** `export_fn` with fixed shapes; core elementwise,
   movement, reduction and nn ops; MLP/CNN tests; V0-V2.
3. **Module export (E2):** `export_module`, `named_params`, shared weights,
   multiple outputs; a small transformer/ViT export; embedding normalization.
4. **Deployment polish (E3):** `Dim::Param`, opset targeting/decomposition,
   constant folding, external data, metadata; dynamic-shape and ORT variants.
5. **Advanced (E4):** only when the importer has the corresponding control-flow
   or quantization feature.

## Open questions

- Feature gate: make `onnx_export` default-on (no new dependency) or opt-in to
  keep the default crate smaller? Recommendation: default-on until measured.
- `ExportError` vs reusing `OnnxError`: recommendation is a dedicated error
  type, with `From<OnnxError>` for I/O/parse failures.
- Default opset: 17 or 18? Recommendation: 17 first, with 18 as an option for
  newer normalization/resize attributes.
- External-data threshold and sidecar layout: one file versus per-tensor files.
- Whether a custom-domain escape hatch is worth shipping for debugging, or
  whether strict failure is enough.
- `If` export once importer T2-A lands; `Loop`/`Scan` only if a target model
  forces it.
- Dtype coverage: fp16/bf16 now; f8 after the ONNX IR/consumer support is
  settled.
- How much module-aware normalization to build beyond `Embedding` (e.g.
  attention, layer norm, conv+activation patterns).

## Security

Export is a local operation on trusted tensors, but the writer still:

- caps tensor sizes and external-data offsets/lengths;
- never writes through caller-controlled paths without validation;
- never serializes host pointers, only bytes;
- treats external-data filenames as relative to the model directory and
  rejects path traversal.

The importer's `ONNX-Compat.md` security section remains authoritative for
reading untrusted models; this section covers writing.
