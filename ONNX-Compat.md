# ONNX compatibility

This file is the contract for the ONNX importer in `src/onnx/`.

**Status (2026-11-16):** T0 is validated end-to-end on Depth-Anything V2. The
T1 transformer/CLIP/CNN op set is implemented: `Einsum` (N operands, ellipsis,
repeated labels), `GatherND` (all `batch_dims`)/`GatherElements`, `Pad` modes,
`TopK`, `NonZero`/`Compress`/`CenterCropPad`/`Scatter*`/`TensorScatter`,
`MaxUnpool`, `Resize` (linear/nearest/cubic, antialias, every coordinate mode
except `tf_crop_and_resize`, aspect-ratio policy), the fused transformer ops
(`SkipLayerNormalization`, `EmbedLayerNormalization`, `RotaryEmbedding`,
`AffineGrid`) and the loss wrappers. Zero-element tensors and `NonZero`/
`Compress` empty selections are supported. The CLIP text/vision exports are
validated end-to-end against onnxruntime (`tests/clip.rs`). Remaining T1
limitations are rare: shape inputs fed at runtime (not derived from
`Shape`/`Size`; evaluated eagerly) and `Resize`'s `tf_crop_and_resize` (needs
`roi`). T2 is split into batches by cost/value: **T2-A** (windows, `If`,
optional/sequence values, the two small ONNX-ML ops), **T2-B** (additional
high-value vision/general ops), and **T2-C** (deferred: quantization, training
ops, `Loop`/`Scan`, the wider ONNX-ML/string domain, `ImageDecoder`, random
ops). See [T2 batches](#t2-batches).

Legend:

- ✅ **implemented** — importer handler + required `Tensor` primitive exist.
- 🟡 **partial** — the importer supports the common form but not every
  attribute/mode, or a required primitive is still missing.
- ⬜ **missing** — no importer support / primitive yet.
- ❌ **not planned / deferred** — intentionally out of scope for now
  (training graphs, `Loop`/`Scan`, wider ONNX-ML, image codecs, random ops).

## Scope

| Tier | Contents | Status |
|---|---|---|
| T0 | protobuf parser, runner, runtime `Shape` folding, external data, Depth-Anything-style models | ✅ done |
| T1 | transformer/CLIP op set (`Einsum`, `GatherND`, `Pad` modes, `TopK`, `Scatter*`, …) | ✅ done |
| T2-A | near-term: `HannWindow`/`HammingWindow`/`BlackmanWindow`, `If`, optional/sequence values, `Binarizer`/`ArrayFeatureExtractor` | ⬜ next |
| T2-B | additional high-value op coverage: `GridSample`, `RoiAlign`, `MaxRoiPool`, `BitShift`, `Unique`; lower priority `Col2Im`, `DeformConv` | ⬜ after T2-A |
| T2-C | deferred / not planned: quantized ops, training ops, `Loop`/`Scan`, wider ONNX-ML, `ImageDecoder`, random ops | ❌ deferred |

## T2 batches

T2 is deliberately not one milestone. The remaining work is split by cost,
value and architectural risk:

| Batch | Contents | Rationale |
|---|---|---|
| **T2-A** | `HannWindow`/`HammingWindow`/`BlackmanWindow`; `If` subgraphs; `OptionalGetElement`/`OptionalHasElement`; `Binarizer`/`ArrayFeatureExtractor` | Small, mostly compositional, and the only T2 features with realistic deployment value. `If` needs a nested graph runner and a recursion cap; runtime conditions disable the replay plan. |
| **T2-B** | `GridSample`; `RoiAlign`/`MaxRoiPool`; `BitShift`; `Unique`; lower priority `Col2Im`/`DeformConv` | Not in the original T2 list, but higher value than most of T2-C: spatial transformers, detection post-processing and integer models. `GridSample`/`BitShift` are graph compositions; ROI pooling and `Unique` are host-assisted and data-dependent. |
| **T2-C** | Quantized ops; training ops (`Adam`/`Adagrad`/`Momentum`/`Gradient`); `Loop`/`Scan`; the wider ONNX-ML/string domain; `ImageDecoder`; random ops (`Bernoulli`/`Multinomial`/`Random*`) | Deferred or not planned. Quantization waits for the INT8 W8A8 design; training graphs are rare in deployment and stateful; `Loop`/`Scan` need dynamic carried state; ONNX-ML is a long preprocessing tail; image codecs are out of scope; random ops need a seeding/reproducibility policy. |

Execution order: **T2-A → T2-B**, with T2-C explicitly out of scope until a
concrete model requires a piece of it. The short T2-A items (windows, the two
ONNX-ML ops) can land before `If`.

## Parser and runner features

| Feature | Status | Notes |
|---|---|---|
| Protobuf wire decoding (no `onnx` crate) | ✅ | `src/onnx/pb.rs`: varint, fixed32/64, length-delimited, typed and packed fields. |
| Opset-version dispatch | 🟡 | Opset map is parsed and used for `Softmax`/`LogSoftmax` axis defaults; per-version op tables are not yet separate. |
| External data (`data_location=1`) | ✅ | reads `offset`/`length` from the model directory with `std::fs`; one tensor at a time. |
| Initializers / `Constant` | ✅ | decoded per tensor and uploaded as buffer leaves; raw bytes dropped after upload. |
| Static shapes | ✅ | `Shape`/`Size` produce host-visible `int64` tensors at run time. |
| Dynamic/symbolic dims | ✅ (runtime-bound) | declared `dim_param`s are bound from the fed tensor; data-dependent shapes remain unsupported. |
| Subgraphs (`If`, `Loop`) | T2-A / T2-C | graph attributes are parsed; `If` is T2-A, `Loop`/`Scan` are T2-C. |
| Optional / sequence values | T2-A | not implemented. |
| Constant folding of `Shape`/`Slice`/`Concat` | 🟡 | host values are cached by interned UOp per run; there is no pre-pass that rewrites the graph. |
| >2 GB models / external data | 🟡 | external tensors stream one at a time; an embedded model is read into memory (2 GB practical limit). |
| Security limits | ✅ | per-tensor byte caps, length checks, overflow-checked dims; no recursion-depth limit yet (the `If` runner is T2-A). |

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
- `tests/onnx.rs`: 26 self-contained tests covering the parser, external data,
  typed/raw tensor encodings, symbolic dims, movement/reduction/nn/resize ops,
  and error paths. The external-checkpoint test is `#[ignore]`d and driven by
  `ONNX_MODEL`/`ONNX_DEVICE`.
- The reference gate is tinygrad's `nn/onnx.py`; compare with
  `../tinygrad/.venv/bin/python` on a model path for new op families.

## Known T1/T2 gaps

- Zero-element tensors use zero-byte storage (`ZeroOpaque`): their kernels are
  skipped, their copies are no-ops, and `NonZero`/`Compress` return an empty
  result when nothing is selected.
- `Scatter*`, `NonZero`, `Compress`, `CenterCropPad`, `MaxUnpool` and
  `TensorScatter` depend on runtime indices/values, so the replay plan is
  disabled for a model that uses them (eager each call). `GatherElements`
  builds a graph gather and keeps the plan.
- `Pad` supports constant/edge/reflect/wrap. A `Pad`/`Resize`/`Reshape` shape
  input derived from `Shape`/`Size` keeps the replay plan (stable per input
  signature); one fed by the caller disables it and runs eagerly.
- `Resize` supports linear/nearest/cubic, every
  `coordinate_transformation_mode` except `tf_crop_and_resize` (needs `roi`),
  `antialias`, `cubic_coeff_a`/`exclude_outside`, all four `nearest_mode`
  choices and `keep_aspect_ratio_policy` `stretch`/`not_larger`/`not_smaller`.
- `Conv` supports per-dimension stride/dilation in 2-D (the kernel is dilated
  with zeros, run at stride 1, then subsampled). `ConvTranspose` does the same
  for per-dimension stride/dilation/output-padding (dilate the kernel,
  zero-insert the input, run at stride 1, append the output padding).
- `Dropout` is inference-only; `BatchNormalization` rejects `training_mode`.
- Windows, `If`, optional/sequence values and the two small ONNX-ML ops
  (`Binarizer`, `ArrayFeatureExtractor`) are T2-A.
- `GridSample`, `RoiAlign`, `MaxRoiPool`, `BitShift`, `Unique`, `Col2Im` and
  `DeformConv` are T2-B.
- Quantized ops (`MatMulInteger`/`ConvInteger`, `QuantizeLinear`/`DequantizeLinear`,
  the `QLinear*` family), training ops, `Loop`/`Scan`, the wider ONNX-ML/string
  domain, `ImageDecoder` and random ops are T2-C (deferred / not planned).

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
| `BitShift` | ⬜ T2-B | logical/arithmetic shift composition. |
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
| `NonZero` | ✅ | host-side; output `[rank, nnz]` int64, including `nnz = 0`. |
| `Hardmax` | ✅ | `argmax` + one-hot. |
| `Unique` | ⬜ T2-B | host-side, data-dependent (plan-disabled), like `NonZero`. |

### Movement / shape / indexing

| ONNX ops | Status | Notes |
|---|---|---|
| `Reshape Flatten Squeeze Unsqueeze Transpose Concat Split Expand Tile Identity` | ✅ | direct movement ops. |
| `Slice` | ✅ | `Ix::Range`; constant starts/ends/axes/steps. |
| `Gather` | ✅ | `Tensor::gather`. |
| `GatherElements GatherND` | ✅ | `GatherND` all `batch_dims` including `k=0`; `GatherElements` as one advanced-index gather (graph, plan-enabled, negative indices). |
| `Scatter ScatterElements ScatterND TensorScatter` | ✅ | `Scatter*` host-side with negative indices and reduction `none`/`add`/`mul`/`max`/`min`; `TensorScatter` uses the graph one-hot/matmul on axis -2 (the KV layout) and a host fallback on other axes, with `linear`/`circular` and per-batch `write_indices`. |
| `Pad` | ✅ | constant/edge/reflect/wrap; runtime pads/axes/value run eagerly. |
| `Resize Upsample` | ✅ | linear/nearest/cubic, antialias, all coordinate modes except `tf_crop_and_resize`, all `nearest_mode`s, `keep_aspect_ratio_policy`. |
| `Shape Size` | ✅ | runtime `int64` tensors; symbolic declared dims are bound from the input. |
| `Range ConstantOfShape EyeLike OneHot Trilu` | ✅ | `arange`/`full`/`eye`/`one_hot`/`tril`/`triu`. |
| `DepthToSpace SpaceToDepth` | ✅ | reshape+permute composition (DCR/CRD). |
| `CenterCropPad Compress` | ✅ | host-side; centre crop/pad and axis/flatten selection, including an empty selection. |
| `Shrink` | ✅ | `where` composition. |

### Control flow and values

| ONNX ops | Status | Notes |
|---|---|---|
| `If` | ⬜ T2-A | branch subgraph runner; runtime conditions make the replay plan run eagerly. |
| `OptionalGetElement OptionalHasElement` | ⬜ T2-A | optional/sequence value plumbing. |
| `Loop Scan` | ❌ T2-C | dynamic iteration/carried state; `Loop` is not in tinygrad's table. |

### Neural-network ops

| ONNX ops | Status | Notes |
|---|---|---|
| `Conv ConvTranspose Gemm MatMul` | ✅ | `conv2d`/`conv_transpose2d`/`linear`/`matmul`; both `Conv` and `ConvTranspose` accept per-dimension strides/dilations (and `ConvTranspose` per-dim `output_padding`). |
| `MatMulInteger ConvInteger` | 🟡 | T2-C; integer matmul/conv on dequantized inputs, after the INT8 W8A8 plan. |
| `MaxPool` | ✅ | `max_pool2d`. |
| `AveragePool GlobalAveragePool GlobalMaxPool` | ✅ | N-D `avg_pool2d` + reductions. |
| `MaxUnpool` | ✅ | host-side placement from the MaxPool indices; inferred or explicit `output_shape`. |
| `BatchNormalization` | ✅ | `Tensor::batchnorm`. |
| `LayerNormalization RMSNormalization GroupNormalization InstanceNormalization` | ✅ | `nn` layers + `Tensor::layernorm`. |
| `LpNormalization MeanVarianceNormalization` | ✅ | `normalize` + mean/var composition. |
| `LRN` | ✅ | `avg_pool2d` + `pow`. |
| `Softmax LogSoftmax SoftmaxCrossEntropyLoss NegativeLogLikelihoodLoss` | ✅ | `Tensor::nll_loss`/`softmax_cross_entropy` (class axis 1); weights, `ignore_index`, `none`/`mean`/`sum`, and the optional `log_prob` output. |
| `Dropout` | 🟡 | inference only; `training_mode=true` is rejected. |
| `Attention` | ✅ | `scaled_dot_product_attention`. |
| `RotaryEmbedding` | ✅ | 3-D/4-D, interleaved, `position_ids`, optional `rotary_embedding_dim`. |
| `SkipLayerNormalization` | ✅ | residual (+bias) + layernorm + affine; optional mean/inv_std/input outputs. |
| `EmbedLayerNormalization` | ✅ | word/position/(segment) embedding + layernorm; the optional `mask_index` output is unsupported. |
| `AffineGrid` | ✅ | 2-D/3-D base grid from `theta`, `align_corners`. |
| `GridSample` | ⬜ T2-B | bilinear/nearest/linear sampling; graph composition, `padding_mode` and `align_corners`. |
| `RoiAlign MaxRoiPool` | ⬜ T2-B | host-assisted ROI pooling; detection models. |
| `Col2Im DeformConv` | ⬜ T2-B | lower priority; `DeformConv` needs sampled kernel offsets. |

### Quantization, training, and misc

| ONNX ops | Status | Notes |
|---|---|---|
| `QuantizeLinear DequantizeLinear DynamicQuantizeLinear` | 🟡 | T2-C; round/clamp/cast composition with per-axis reshape, after INT8 W8A8. |
| `QLinearAdd QLinearMul QLinearMatMul QLinearConv QLinearGlobalAveragePool` | 🟡 | T2-C; integer dequantize → op → requantize, after INT8 W8A8. |
| `Adam Adagrad Momentum Gradient` | 🟡 | T2-C; stateful/training-only graphs, low deployment value. |
| `HannWindow HammingWindow BlackmanWindow` | ⬜ T2-A | `arange`+`cos` composition. |
| `Binarizer ArrayFeatureExtractor` | ⬜ T2-A | the only planned ONNX-ML ops. |
| `ImageDecoder` | ❌ T2-C | image codecs are out of scope; feed tensors. |
| `Bernoulli Multinomial RandomNormal RandomUniform` | ❌ T2-C | random ops; a seeding/reproducibility policy is needed first. |
| `Einsum` | ✅ | N operands, ellipsis (right-aligned broadcast), repeated labels (diagonal) and implicit output. |

## Validation plan (remaining)

1. ~~**CLIP text/vision ONNX** for the T1 set.~~ Done:
   `tests/clip.rs` compares the fp32 Xenova exports against onnxruntime
   (max abs diff 2.4e-4 text, 1.7e-4 vision).
2. **ONNX backend node tests** from `../tinygrad/test/external/` for op-level
   correctness; compare against `onnxruntime` in the dev environment only.
3. **Golden fixtures**: `tests/onnx.rs` builds models in-memory; add committed
   fixtures (when size allows) for the T1 op families.
4. **T2**: land T2-A first (windows, then `If`), then T2-B (`GridSample`,
   `RoiAlign`/`MaxRoiPool`, `BitShift`, `Unique`). T2-C stays documented as
   deferred unless a target model forces a specific item.

## Security

ONNX is untrusted input. The importer must cap tensor element counts and
external-data lengths, bound subgraph recursion, and never trust `dims`/`pads`/
`shape` values before validating them. `safetensors` is safe by construction;
a torch-pickle loader (not planned here) is a code-execution format and must
not be added casually.
