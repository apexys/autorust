# ONNX compatibility

This file is the contract for the ONNX importer in `src/onnx/`.

**Status (2026-10):** the T0 importer is implemented and validated end-to-end on
Depth-Anything V2. T1 is largely covered (the transformer/CNN op set, `Pad`
modes, `TopK`, `GatherND`/`GatherElements`, two-operand `Einsum`); the remaining
T1/T2 gaps are listed below. T2 (quantization, training ops, ONNX-ML, control
flow) is not implemented.

Legend:

- ✅ **implemented** — importer handler + required `Tensor` primitive exist.
- 🟡 **partial** — the importer supports the common form but not every
  attribute/mode, or a required primitive is still missing.
- ⬜ **missing** — no importer support / primitive yet.
- ❌ **not planned** — intentionally out of scope (training ops, image codecs,
  ONNX-ML feature transformers, …).

## Scope

| Tier | Contents | Status |
|---|---|---|
| T0 | protobuf parser, runner, runtime `Shape` folding, external data, Depth-Anything-style models | ✅ done |
| T1 | transformer/CLIP op set (`Einsum`, `GatherND`, `Pad` modes, `TopK`, `Scatter*`, …) | 🟡 mostly done (see gaps) |
| T2 | quantized ops, training ops (`Adam`/`Momentum`/`Gradient`), windows, `If` subgraphs, ONNX-ML | ⬜ not started |

## Parser and runner features

| Feature | Status | Notes |
|---|---|---|
| Protobuf wire decoding (no `onnx` crate) | ✅ | `src/onnx/pb.rs`: varint, fixed32/64, length-delimited, typed and packed fields. |
| Opset-version dispatch | 🟡 | Opset map is parsed and used for `Softmax`/`LogSoftmax` axis defaults; per-version op tables are not yet separate. |
| External data (`data_location=1`) | ✅ | reads `offset`/`length` from the model directory with `std::fs`; one tensor at a time. |
| Initializers / `Constant` | ✅ | decoded per tensor and uploaded as buffer leaves; raw bytes dropped after upload. |
| Static shapes | ✅ | `Shape`/`Size` produce host-visible `int64` tensors at run time. |
| Dynamic/symbolic dims | ✅ (runtime-bound) | declared `dim_param`s are bound from the fed tensor; data-dependent shapes remain unsupported. |
| Subgraphs (`If`, `Loop`) | T2 | graph attributes are parsed; no control-flow runner yet. `Loop` is not in tinygrad's table. |
| Optional / sequence values | T2 | not implemented. |
| Constant folding of `Shape`/`Slice`/`Concat` | 🟡 | host values are cached by interned UOp per run; there is no pre-pass that rewrites the graph. |
| >2 GB models / external data | 🟡 | external tensors stream one at a time; an embedded model is read into memory (2 GB practical limit). |
| Security limits | ✅ | per-tensor byte caps, length checks, overflow-checked dims; no recursion-depth limit yet (subgraphs are T2). |

## Runner API

```rust
pub struct OnnxModel { /* Ctx + Session + decoded graph */ }

impl OnnxModel {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, OnnxError>;
    pub fn load_with_device(path: impl AsRef<Path>, device: Option<&str>) -> Result<Self, OnnxError>;
    pub fn ctx(&self) -> &Ctx;
    pub fn session_mut(&mut self) -> &mut Session;
    pub fn inputs(&self) -> &[ValueInfo];
    pub fn outputs(&self) -> &[String];
    pub fn run(&mut self, inputs: &[(&str, Tensor)]) -> Result<HashMap<String, Tensor>, OnnxError>;
    pub fn run_f32(&mut self, inputs: &[(&str, Tensor)]) -> Result<HashMap<String, Vec<f32>>, OnnxError>;
}
```

Inputs must be built from `model.ctx()`. Returned tensors are lazy; read them
with `model.session_mut()` (or use `run_f32`). The model owns the `Ctx`, so
outputs remain valid after it is dropped.

## Validation

- **Depth-Anything V2 ViT-B (fp16, 619 nodes, 253 initializers, 21 op types)**:
  the full graph runs on `CUDA`. Against tinygrad's `OnnxRunner` on the same
  zero image, the `depth` output (268,324 values) has **max abs error 0.015625**
  (2 fp16 ulps at ~8), mean abs error 0.0019, max relative error 0.0054. This
  is the T0 acceptance model.
- `tests/onnx.rs`: 24 self-contained tests covering the parser, external data,
  typed/raw tensor encodings, symbolic dims, movement/reduction/nn/resize ops,
  and error paths. The external-checkpoint test is `#[ignore]`d and driven by
  `ONNX_MODEL`/`ONNX_DEVICE`.
- The reference gate is tinygrad's `nn/onnx.py`; compare with
  `../tinygrad/.venv/bin/python` on a model path for new op families.

## Known T1/T2 gaps

- `NonZero`, `Compress`, `CenterCropPad`, `Scatter*`, `MaxUnpool` need
  `masked_select`/scatter primitives.
- `Pad` supports constant/edge/reflect/wrap, not arbitrary dynamic pads
  (pads must resolve to host constants).
- `Resize` supports linear (align-corners/half-pixel/pytorch-half-pixel for
  outputs > 1) and all four `nearest_mode` choices; cubic, antialias,
  `asymmetric`/`half_pixel_symmetric` linear, and non-`stretch` aspect-ratio
  policies are missing.
- `GatherND` only supports `batch_dims=0`; `GatherElements` uses a host-index
  loop (correct, not fast).
- Zero-element tensors are rejected: the storage layer cannot allocate empty
  storage yet.
- `Conv`/`ConvTranspose` require the same stride/dilation/output-padding on
  every spatial dim (the `Tensor` API takes scalars); differing per-dim values
  return `Unsupported`.
- `Einsum` covers two operands; ellipsis and repeated labels are rejected.
- `Dropout` is inference-only; `BatchNormalization` rejects `training_mode`.
- Quantized ops, ONNX training ops, windows, `If`, optional/sequence values and
  ONNX-ML are T2.
- `MatMulInteger`/`ConvInteger` and the `QLinear*` family are T2.

## Operator matrix

The matrix is grouped by ONNX family. "primitive present" means a `Tensor` op
exists; "compose" means it can be built from existing ops but needs a wrapper;
"missing" means a new primitive is required.

### Elementwise / logical

| ONNX ops | Status | Notes |
|---|---|---|
| `Abs Add Ceil Div Exp Floor Log Max Mean Min Mod Mul Neg Pow Reciprocal Relu Sqrt Sub Sum Where` | ✅ | direct `Tensor` ops. |
| `Equal Greater GreaterOrEqual Less LessOrEqual Not And Or Xor` | ✅ | comparisons/logical ops. |
| `BitwiseAnd BitwiseOr BitwiseXor` | ✅ | wrapper uses the integer `And`/`Or`/`Xor` ALU ops. |
| `BitwiseNot` | ✅ | composed as `-x - 1`. |
| `Cast CastLike Clip Round Sign` | ✅ | importer + `Tensor` compositions. |
| `Erf` | ✅ | tinygrad's Abramowitz–Stegun polynomial. |
| `Sin Cos Tan Tanh Sigmoid` | ✅ | `cos`/`tan` are compositions. |
| `Asin Acos Atan Asinh Acosh Atanh Sinh Cosh` | ✅ | `Tensor` helpers added (`extra.rs`). |
| `Celu Elu Selu LeakyRelu Mish Softplus Softsign Swish ThresholdedRelu PRelu` | ✅ | activation helpers in `tensor/extra.rs`. |
| `Gelu FastGelu BiasGelu HardSigmoid HardSwish` | ✅ | exact/tanh GELU + clamp compositions. |
| `IsInf IsNaN` | ✅ | predicates support the `detect_*` flags. |
| `Shrink` | ✅ | `where` + const. |

### Reductions and sorting

| ONNX ops | Status | Notes |
|---|---|---|
| `ReduceSum ReduceMean ReduceMax ReduceMin ReduceProd` | ✅ | `sum`/`mean`/`max`/`min`/`prod` (add `min`). |
| `ArgMax ArgMin` | ✅ | supports `keepdims` and `select_last_index`. |
| `ReduceL1 ReduceL2 ReduceSumSquare ReduceLogSum ReduceLogSumExp` | ✅ | compositions over `sum`/`square`/`log`. |
| `CumSum` | ✅ | `Tensor::cumsum`. |
| `TopK` | ✅ | bitonic `sort`/`topk`; indices cast to int64. |
| `NonZero` | ⬜ | needs `nonzero` (masked select / gather). |
| `Hardmax` | ✅ | `argmax` + one-hot. |

### Movement / shape / indexing

| ONNX ops | Status | Notes |
|---|---|---|
| `Reshape Flatten Squeeze Unsqueeze Transpose Concat Split Expand Tile Identity` | ✅ | direct movement ops. |
| `Slice` | ✅ | `Ix::Range`; constant starts/ends/axes/steps. |
| `Gather` | ✅ | `Tensor::gather`. |
| `GatherElements GatherND` | 🟡 | `GatherND` supports `batch_dims=0`; `GatherElements` uses a host-index loop. |
| `Scatter ScatterElements ScatterND TensorScatter` | ⬜ | needs the scatter primitive/`masked_select`. |
| `Pad` | ✅ | constant/edge/reflect/wrap with static (host-resolved) pads. |
| `Resize Upsample` | 🟡 | linear (align-corners/half-pixel) and nearest with all `nearest_mode`s; no cubic/antialias. |
| `Shape Size` | ✅ | runtime `int64` tensors; symbolic declared dims are bound from the input. |
| `Range ConstantOfShape EyeLike OneHot Trilu` | ✅ | `arange`/`full`/`eye`/`one_hot`/`tril`/`triu`. |
| `DepthToSpace SpaceToDepth` | ✅ | reshape+permute composition (DCR/CRD). |
| `CenterCropPad Compress` | ⬜ | needs `masked_select`/`nonzero`. |
| `Shrink` | ✅ | `where` composition. |

### Neural-network ops

| ONNX ops | Status | Notes |
|---|---|---|
| `Conv ConvTranspose Gemm MatMul` | ✅ | `conv2d`/`conv_transpose2d`/`linear`/`matmul`. |
| `MatMulInteger ConvInteger` | 🟡 | T2; integer matmul/conv on dequantized inputs. |
| `MaxPool` | ✅ | `max_pool2d`. |
| `AveragePool GlobalAveragePool GlobalMaxPool` | ✅ | N-D `avg_pool2d` + reductions. |
| `MaxUnpool` | ⬜ | add `max_unpool2d` (or compose from indices). |
| `BatchNormalization` | ✅ | `Tensor::batchnorm`. |
| `LayerNormalization RMSNormalization GroupNormalization InstanceNormalization` | ✅ | `nn` layers + `Tensor::layernorm`. |
| `LpNormalization MeanVarianceNormalization` | ✅ | `normalize` + mean/var composition. |
| `LRN` | ✅ | `avg_pool2d` + `pow`. |
| `Softmax LogSoftmax SoftmaxCrossEntropyLoss NegativeLogLikelihoodLoss` | 🟡 | softmax/log-softmax done; the loss wrappers need `nll_loss`. |
| `Dropout` | 🟡 | inference only; `training_mode=true` is rejected. |
| `Attention` | ✅ | `scaled_dot_product_attention`. |
| `RotaryEmbedding` | 🟡 | port the DA3 RoPE helper (Embedding cos/sin tables). |
| `SkipLayerNormalization EmbedLayerNormalization` | 🟡 | compositions (layernorm + residual + embedding). |
| `AffineGrid` | 🟡 | grid construction from arange/linear. |
| `If` | T2 | subgraph runner; `Loop` is not in tinygrad's table. |
| `OptionalGetElement OptionalHasElement` | T2 | optional/sequence value plumbing. |

### Quantization, training, and misc

| ONNX ops | Status | Notes |
|---|---|---|
| `QuantizeLinear DequantizeLinear DynamicQuantizeLinear` | 🟡 | T2; round/clamp/cast composition with per-axis reshape. |
| `QLinearAdd QLinearMul QLinearMatMul QLinearConv QLinearGlobalAveragePool` | 🟡 | T2; integer dequantize → op → requantize. |
| `Adam Adagrad Momentum Gradient` | 🟡 | T2; port of tinygrad's ONNX training adapters over `nn/optim`. |
| `HannWindow HammingWindow BlackmanWindow` | 🟡 | T2; `arange`+`cos` composition. |
| `Binarizer ArrayFeatureExtractor` | T2 | ONNX-ML domain. |
| `ImageDecoder` | ❌ | image codecs are out of scope; feed tensors. |
| `Einsum` | 🟡 | two-operand path; ellipsis/repeated labels are rejected. |

## Validation plan (remaining)

1. **CLIP text/vision ONNX** for the T1 set (`Einsum`, `GatherND`, `TopK`,
   `Pad` modes, and the still-missing `Scatter*` family).
2. **ONNX backend node tests** from `../tinygrad/test/external/` for op-level
   correctness; compare against `onnxruntime` in the dev environment only.
3. **Golden fixtures**: `tests/onnx.rs` builds models in-memory; add committed
   fixtures for the T1 op families once the CLIP gate exists.

## Security

ONNX is untrusted input. The importer must cap tensor element counts and
external-data lengths, bound subgraph recursion, and never trust `dims`/`pads`/
`shape` values before validating them. `safetensors` is safe by construction;
a torch-pickle loader (not planned here) is a code-execution format and must
not be added casually.
