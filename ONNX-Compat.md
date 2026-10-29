# ONNX compatibility

This file is the contract for the ONNX importer. The importer is **not in
0.1.0 yet**; the statuses below describe the target T1/T2 surface and whether
the required `Tensor` primitive already exists.

Legend:

- ✅ **primitive present** — the Rust `Tensor` op exists; the ONNX wrapper is
  mechanical.
- 🟡 **compose/partial** — implementable from existing primitives, but the
  wrapper needs real work (shape inference, attribute handling, or a missing
  helper).
- ⬜ **missing primitive** — needs a new `Tensor`/codegen feature.
- ❌ **not planned** — intentionally out of scope (training ops, image codecs,
  ONNX-ML feature transformers, etc.).

## Scope

| Tier | Contents | Status |
|---|---|---|
| T0 | protobuf parser, runner, static `Shape` folding, external data, Depth-Anything-style models | planned |
| T1 | transformer/CLIP op set (`Einsum`, `GatherND`, `Pad` modes, `TopK`, `Scatter*`, …) | planned |
| T2 | quantized ops, training ops (`Adam`/`Momentum`/`Gradient`), windows, `If` subgraphs, ONNX-ML | planned, lower priority |

## Parser and runner features

| Feature | Status | Notes |
|---|---|---|
| Protobuf wire decoding (no `onnx` crate) | planned | ~400–500 Rust lines; varint, fixed32/64, length-delimited. |
| Opset-version dispatch | planned | `OnnxNode` carries `(domain, version)`; select the newest compatible impl. |
| External data (`data_location=1`) | planned | read `offset`/`length` from the origin file with `std::fs`; no mmap. |
| Initializers / `Constant` | planned | upload as buffer leaves through `State`/`Tensor::from_read`. |
| Static shapes | planned | required for v1; `SVal::I` shapes make `Shape` foldable. |
| Dynamic/symbolic dims | ❌ v1 | needs runtime shape tensors and symbolic `Reshape`; not in the first importer. |
| Subgraphs (`If`, `Loop`) | T2 | `If` only; `Loop` is not in tinygrad's op table. |
| Optional / sequence values | T2 | `OptionalGetElement`, `OptionalHasElement`, sequence inputs. |
| Constant folding of `Shape`/`Slice`/`Concat` | planned | avoids realizing tiny shape tensors per `Reshape`. |
| >2 GB models / external data | planned | stream one tensor at a time; see `State` + `Tensor::from_read`. |
| Security limits | planned | cap tensor sizes, recursion depth, opset/attribute counts; ONNX is untrusted input. |

## Operator matrix

The matrix is grouped by ONNX family. "primitive present" means a `Tensor` op
exists; "compose" means it can be built from existing ops but needs a wrapper;
"missing" means a new primitive is required.

### Elementwise / logical

| ONNX ops | Status | Notes |
|---|---|---|
| `Abs Add Ceil Div Exp Floor Log Max Mean Min Mod Mul Neg Pow Reciprocal Relu Sqrt Sub Sum Where` | ✅ | direct `Tensor` ops. |
| `Equal Greater GreaterOrEqual Less LessOrEqual Not And Or Xor` | ✅ | comparisons/logical ops. |
| `BitwiseAnd BitwiseOr BitwiseXor` | 🟡 | `and`/`or`/`xor` exist for bool; integer bitwise wrappers needed. |
| `BitwiseNot` | ⬜ | add `Tensor::bitwise_not`. |
| `Cast CastLike Clip Round Sign` | 🟡 | cast/clip exist; `round`/`sign` are compositions. |
| `Erf` | 🟡 | tinygrad's polynomial (`abs`/`sign`/`exp`/`polyN`); add `Tensor::erf`. |
| `Sin Cos Tan Tanh Sigmoid` | 🟡 | sin/cos/tanh/sigmoid exist; `tan` is `sin/cos`. |
| `Asin Acos Atan Asinh Acosh Atanh Sinh Cosh` | 🟡 | not exposed; compose from `sin`/`cos`/`log`/`sqrt` or add helpers. |
| `Celu Elu Selu LeakyRelu Mish Softplus Softsign Swish ThresholdedRelu PRelu` | 🟡 | compositions; add `Tensor` activation helpers. |
| `Gelu FastGelu BiasGelu HardSigmoid HardSwish` | 🟡 | tanh/erf GELU + clamp compositions. |
| `IsInf IsNaN` | ⬜ | add predicates (`isfinite`/`isinf`/`isnan`). |
| `Shrink` | 🟡 | `where` + const. |

### Reductions and sorting

| ONNX ops | Status | Notes |
|---|---|---|
| `ReduceSum ReduceMean ReduceMax ReduceMin ReduceProd` | ✅ | `sum`/`mean`/`max`/`min`/`prod` (add `min`). |
| `ArgMax ArgMin` | 🟡 | `argmax` exists; `argmin` is `-argmax(-x)`. |
| `ReduceL1 ReduceL2 ReduceSumSquare ReduceLogSum ReduceLogSumExp` | 🟡 | compositions (`abs`/`pow`/`log`/`logsumexp`). |
| `CumSum` | ✅ | `Tensor::cumsum`. |
| `TopK` | ⬜ | needs `sort`/`topk` (bitonic sort port). |
| `NonZero` | ⬜ | needs `nonzero` (masked select / gather). |
| `Hardmax` | 🟡 | `argmax` + one-hot. |

### Movement / shape / indexing

| ONNX ops | Status | Notes |
|---|---|---|
| `Reshape Flatten Squeeze Unsqueeze Transpose Concat Split Expand Tile Identity` | ✅ | direct movement ops. |
| `Slice` | ✅ | `Ix::Range`; constant starts/ends/axes/steps. |
| `Gather` | ✅ | `Tensor::gather`. |
| `GatherElements GatherND` | ⬜ | add gather-elements/ND wrappers. |
| `Scatter ScatterElements ScatterND TensorScatter` | ⬜ | add scatter family. |
| `Pad` | 🟡 | constant pad exists; `reflect`/`replicate`/`circular` need helpers. |
| `Resize Upsample` | 🟡 | linear/nearest exist; add bicubic (DA3 reference `bicubic.py`). |
| `Shape Size` | 🟡 | static fold from `SVal` shapes; dynamic shapes are v1-unsupported. |
| `Range ConstantOfShape EyeLike OneHot Trilu` | ✅ | `arange`/`full`/`eye`/`one_hot`/`tril`/`triu`. |
| `DepthToSpace SpaceToDepth` | 🟡 | reshape+permute composition. |
| `CenterCropPad Compress` | 🟡 | shrink/gather; add `masked_select`. |
| `Shrink` | 🟡 | `where` composition. |

### Neural-network ops

| ONNX ops | Status | Notes |
|---|---|---|
| `Conv ConvTranspose Gemm MatMul` | ✅ | `conv2d`/`conv_transpose2d`/`linear`/`matmul`. |
| `MatMulInteger ConvInteger` | 🟡 | T2; integer matmul/conv on dequantized inputs. |
| `MaxPool` | ✅ | `max_pool2d`. |
| `AveragePool GlobalAveragePool GlobalMaxPool` | 🟡 | add `avg_pool2d`; global pools are reductions. |
| `MaxUnpool` | ⬜ | add `max_unpool2d` (or compose from indices). |
| `BatchNormalization` | ✅ | `Tensor::batchnorm`. |
| `LayerNormalization RMSNormalization GroupNormalization InstanceNormalization` | ✅ | `nn` layers + `Tensor::layernorm`. |
| `LpNormalization MeanVarianceNormalization` | 🟡 | `normalize`/mean-var compositions. |
| `LRN` | 🟡 | local response normalization from `avg_pool`/`pow`. |
| `Softmax LogSoftmax SoftmaxCrossEntropyLoss NegativeLogLikelihoodLoss` | 🟡 | softmax/CE exist; add `nll_loss`. |
| `Dropout` | ✅ | `Tensor::dropout` (TRAINING-gated). |
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
| `Einsum` | ⬜ | add `Tensor::einsum` (general two-operand path). |

## Validation plan

1. **Depth-Anything V2 ONNX** (local file, 619 nodes / 21 op types):
   `Cast Shape Concat Expand Squeeze Div Pow Reshape Add Mul Conv Transpose
   LayerNormalization MatMul Gather Slice Softmax Erf ConvTranspose Relu Resize`.
   This is the T0 acceptance model; only `Shape` folding + `Erf` are new
   primitives.
2. **CLIP text/vision ONNX** for the T1 set (`Einsum`, `GatherND`, `TopK`,
   `Scatter*`, `Pad` modes).
3. **ONNX backend node tests** from `../tinygrad/test/external/` for op-level
   correctness; compare against `onnxruntime` in the dev environment only.
4. **Numerical gates**: run the same model in tinygrad and autorust from the
   same weights; compare logits/depth with a tight tolerance. Add a Rust golden
   test for each supported op family.

## Security

ONNX is untrusted input. The importer must cap tensor element counts and
external-data lengths, bound subgraph recursion, and never trust `dims`/`pads`/
`shape` values before validating them. `safetensors` is safe by construction;
a torch-pickle loader (not planned here) is a code-execution format and must
not be added casually.
