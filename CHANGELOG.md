# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **ONNX indexing family.** `NonZero`, `Compress`, `CenterCropPad`,
  `Scatter`/`ScatterElements`/`ScatterND` (with `none`/`add`/`mul`/`max`/`min`
  reductions) are implemented; they are host-realized and disable the replay
  plan for models that use them (`tests/onnx.rs`).
- **ONNX `GatherND batch_dims>0` and `MaxUnpool`.** `GatherND` now gathers
  per batch axis; `MaxUnpool` places pooled values at the MaxPool indices with
  an explicit or inferred output shape (`tests/onnx.rs`).
- **Generalized ONNX `Einsum`.** `Tensor::einsum_many` (and the two-operand
  `einsum` wrapper) now supports N operands, ellipsis with right-aligned
  broadcasting, repeated labels (diagonal/trace) and implicit output. The ONNX
  `Einsum` op no longer rejects >2 operands (`tests/einsum.rs`,
  `tests/onnx.rs::einsum_multi_operand_chain`).

## [0.3.0] - 2026-11-16

### Changed

- **Fast math mode is now the default.** `Settings` gains `math_mode`
  (`MathMode::Fast` by default) and `HostGemm` gains `tf32` (both
  source-breaking for struct-literal construction). `Fast` enables TF32
  cuBLAS, generated/custom tensor-core paths, `CC` `-ffast-math` and OpenCL
  `-cl-fast-relaxed-math`; `AUTORUST_MATH=exact` restores pedantic f32.
  Measured RTX 3060: ResNet-18 infer 264.6 → 118.0 ms (-55%), ResNet-50
  88.6 → 35.8 ms (-60%), ViT-B/16 infer bs 8 82.8 → 65.6 ms. Existing
  benchmark tables were collected in Exact mode and are labelled as such in
  `PERFORMANCE.md`.

### Added

- **Custom tensor-core kxk implicit-GEMM convolution.** A bounded TF32 `wmma`
  kernel generates patches directly into the A tile (no im2col copy), 64 rows x
  64 output channels per block; `Tensor::conv2d_with`/`Conv2d::forward_with`
  use it for groups=1/stride 1-2/symmetric-padding f32 inference. It is picked
  for the measured 3x3 stride-1 case and for low-K convs below the im2col
  floor, notably ResNet's 7x7 stem (K=147). Measured RTX 3060, bs 32:
  ResNet-18 infer 214-216 → 118 ms, ResNet-50 83 → 35-37 ms, MobileNet
  neutral. Gate: `tests/cuda_conv_tc.rs`. The ONNX runner's `Conv` op also
  uses it when its input is already realized: ONNX ResNet-18 (w=0.25) bs 8
  went 9.24-9.66 → 6.94-7.38 ms (-29%) with no forced mid-graph realize.
- **Opt-in conv-only TF32 tensor-core path.** `AUTORUST_CONV_TF32=1` switches
  batched f32 im2col host GEMMs with `n >= 28` to TF32 tensor-core math while
  keeping attention/linear f32 strict. Measured ResNet-18 -7-11%, UNet ~4% median
  (mixed), MobileNet/ONNX neutral/slight; convolution numerics change, so it is off by
  default. Gate: `tests/conv_tf32.rs`.
- **Low-K 3x3 convolution im2col on CUDA.** The im2col floor is now
  `cin*kh*kw >= 256` on CUDA (`576` on CPU/CL/VK, where host-GEMM setup costs
  more). UNet's 32-channel 3x3 layers moved to im2col + cuBLAS and improved
  11.3-12.4 → 10.2-10.3 ms (~10-14%); ResNet-18/ONNX were neutral. Gate:
  `tests/conv_im2col.rs` (CUDA forward and gradients vs INTERP at K=288).
- **1x1 convolution fast path on CUDA.** A 1x1 conv with `groups=1`, stride 1,
  dilation 1 and zero padding now lowers as `[B*H*W, Cin] @ [Cin, Cout]`
  instead of the 3-axis convolution path, with a CUDA-only row gate
  (`AUTORUST_CONV1X1`, `AUTORUST_CONV1X1_MIN_M`, default 2048). Measured
  MobileNetV2 -7%, ResNet-50 -2%, UNet -4%; ResNet-18 and CPU unchanged.
  `tests/conv1x1.rs` covers the direct-sum reference, the stride fallback and
  the CUDA gradients.
- **Fused CUDA full reductions.** `Tensor::sum_with` and `Tensor::max_with`
  (and the memops driver by default) run an already-realized contiguous f32
  scalar reduction as one streaming pass plus a tiny finish kernel instead of
  the graph reduce. On the 16M-element memops case, CUDA `sum` improved
  0.37 → 0.25 ms (270 GB/s) and `amax` 0.40 → 0.25 ms (270 GB/s), within 6% of
  torch. `AUTORUST_REDUCE_KERNEL=0` forces the graph path; the graph remains the
  fallback for views, non-scalar axes and non-f32 inputs. Tests:
  `tests/cuda_reduce.rs`.

- **Burn cross-framework benchmark.** `bench/burn` runs the same
  `beautiful_mnist`, ViT-B/16 and GEMM workloads through burn 0.22.1 (CubeCL
  CUDA and Flex CPU), and `bench/run.py` / `bench/report.py` include it. The
  comparison adds burn columns to `README.md` and `PERFORMANCE.md`, plus a
  review of burn's autotuning, implicit-GEMM conv, fused reductions,
  broadcast-rhs matmul folding and multi-stream execution as backlog items in
  `PERFORMANCE_OPTIMIZATIONS.md`. Burn's f32 CUDA rows use TF32 tensor
  cores and are labelled as such; no autorust kernel changed in this pass.

### Fixed

- **`--no-default-features` builds again.** The fused reduction and conv
  helpers referenced `runtime::cuda_reduce` / `runtime::cuda_conv_tc`
  unconditionally; they are now behind `backend_cuda`, like the kernels
  they call.
- **Small batched contractions no longer go through one cuBLAS call per
  batch.** CUDA `Auto` now keeps a batched f32 contraction on the generated
  kernel below `m*n*k = 1.5M` per batch (`AUTORUST_CUBLAS_BATCH_WORK`; `0`
  disables it, `AUTORUST_CUBLAS=all` still forces cuBLAS). This fixes a
  regression in current `main`: MNIST train bs 512 CUDA measured 52.9 ms/step
  and now measures 23.3 ms (the committed 22.4 ms row); ONNX ResNet-18 (w=0.25)
  bs 8 goes 12.0 → 10.2 ms. ResNet-18's 2.06M/batch host families stay on
  cuBLAS. Unit test: `schedule::host_gemm::tests::
  tiny_batches_stay_on_the_generated_kernel`.

## [0.2.0] - 2026-11-16

Feature release: a fourth GPU backend (`VK`, SPIR-V emitted directly and now in
the default feature set), eager `StepPlan` capture/replay, ONNX replay plans,
fused CUDA/OpenCL optimizer kernels, inference Conv+BN folding, and a large
host-GEMM/cuBLAS and codegen pass. ViT-B/16 f32 training at bs 4 on CUDA drops
from 317 to ~200 ms (216.7 in the full matrix) and ResNet-18 inference from 374
to 323 ms. `StepPlan` is on by default; two public structs gained Vulkan fields,
which is source-breaking for struct-literal construction and is called out
below.

### Changed

- **Vulkan fields on public settings structs (source-breaking).** `Settings`
  gains `vk_device` and `LibraryGemmPolicy` gains `vk`. Neither struct is
  `#[non_exhaustive]`, so code that constructs them with a struct literal must
  add the field; use `Settings::default()`/`from_env()` or
  `..Default::default()`. The `backend_vk` feature is also in the default set,
  so default builds compile the Vulkan backend and pull `ash` (bindings only;
  `libvulkan` is loaded at run time).
- **Measured items 15 (symbolic plans) and 1 (schedule cache) as explicit
  skips.** The port has no symbolic-variable machinery at all (`SInt::U` is
  never built, there is no `Variable` op, `Session.var_vals` is unused,
  `ProgramInfo::launch_dims` ignores its `var_vals` argument, `Jit::replay`
  passes an empty map), so `var_vals` resolution is the review's larger
  compiler project; recapture is 494/619/726 ms at bs 1/4/8 vs 12.8/37.8/68.2 ms
  replay. The kernel caches already exist, but `create_linear_with_vars` is
  re-run per capture (`realize.schedule` 61 ms + `callify` 11.5 ms); a
  SINK-content cache needs the CALL/buffer-map re-interning of the serialized
  plan. Both are recorded in `PERFORMANCE_OPTIMIZATIONS.md`.
- **Measured the ONNX preprocessing and bf16-epilogue items (both skips).** The
  ONNX assets are already clean (0 `Constant` nodes; `Identity` aliases emit no
  kernels; the ResNet has no BN; MNIST's BN is fused into the conv stores) and
  the steady replays are 29/12 kernels, so the gap is conv lowering (item 2).
  On bf16, `vit_infer_bf16` accepts 0 host GEMMs and `vit_train_bf16`'s accepted
  descriptors are bias-free attention contractions; the biased linears are f32
  compute with a bf16 store, so a bf16 cuBLASLt epilogue has no target. Both are
  recorded in `PERFORMANCE_OPTIMIZATIONS.md`.
- **Measured the ONNX preprocessing item (skip).** The exported assets have no
  `Constant` nodes, `Identity` nodes emit no kernels, host-GEMM weights are
  already consumed through strides (no pack/transpose kernels in the replay),
  and the ResNet has no BN nodes (MNIST's are fused into the conv stores). The
  steady replays are 29 (ResNet) and 12 (MNIST) kernels; the remaining gap is
  conv lowering (item 2), not graph preprocessing. Recorded in
  `PERFORMANCE_OPTIMIZATIONS.md`.
- **Host-GEMM epilogue C indexing and outer batch stride.** The epilogue used
  `row*n + col` with one batch stride, which assumes `c_m == n` and contiguous
  batch slices; the folded-conv layout (`C[b][cout][oy][ox]`, two batch axes)
  violates both. `HostGemm` gains `c_bo` (the outer C batch stride) and the
  epilogue takes `c_m`/`batch_inner`/`c_bo`
  (`rel = bo*c_outer + bi*c_b + row*c_m + col`). A size gate keeps small
  transposed convs (1x1 / 7-spatial) on the generated kernel. Gate:
  `tests/cublas.rs::cublas_matches_im2col_conv_transposed_store`.
- **Host-GEMM epilogue supports per-row (`m`) affine terms and both orders.**
  A transposed conv store (`C[cout, ox]`) puts the per-channel BN terms on the
  m axis, and a BN store is `C*scale + bias` rather than the transformer's
  `(C + bias)*scale`. `HostGemm` now carries `bias_m`/`scale_m`/`bias_outside`
  and the CUDA epilogue indexes the row and applies scale/bias in the recorded
  order (this also fixes a latent wrong order for BN-style stores). Gates:
  `tests/cublas.rs::cublas_matches_per_row_bias_and_scale` in both orders. The
  ResNet convs still stop at `epilogue-operand` because the BN scale is a
  computed `1/sqrt(var+eps)`, so folding BN (item 7) is the remaining step.
- **cuBLASLt fused bias(+ReLU) epilogue for host GEMMs.** `runtime::cublas`
  loads `libcublasLt` and runs eligible f32 host GEMMs with the bias fused
  into the matmul (`CUBLASLT_EPILOGUE_BIAS` / `BIAS_RELU`), skipping the
  separate `autorust_epi_*` launch. At ViT-B/16 train bs 4, 98 of 182 host
  GEMMs per step take it and the step goes **200.9/201.3/202.0 →
  199.6/200.6/200.4 ms (-0.6%)**; ViT inference -1.0%; MNIST/ResNet neutral.
  The per-column scale, elementwise residual and `beta = 1` accumulation cases
  cannot be expressed in a cuBLASLt epilogue and keep the kernel. Numeric gate:
  `tests/cublas.rs::cublas_lt_fuses_bias_and_relu`; `AUTORUST_CUBLASLT=0`
  reverts.
- **Measured the training attention path in situ.** A captured CUDA graph
  replays a ViT-B/16 train bs 4 step in 196 ms of GPU time, so the step is
  GPU-bound; attention forward+backward is 86 ms of it (forward 29 ms, backward
  57 ms), i.e. 44% and the second-largest measured item after convolution.
  Nsight Compute absolute times are not in-situ (cache flushing inflated the
  attention `dS` kernel ~15x), so the graph-replay method is recorded for
  future measurements. No behavior change in this commit.
- **Transposed (column-major) GEMM outputs stay on cuBLAS.** A store through a
  transposed view (`assign` into `p.transpose()`, the backward weight-gradient
  path) used to keep the generated kernel. `extract` now describes the
  transposed product (`C^T = B^T A^T`) and swaps the operand roles. ViT-B/16
  f32 training on an RTX 3060: **bs 4 317.6 → 230.2 ms/step (-27.5%)**, **bs 1
  100.2 → 81.0 (-19%)**; bs 4 inference unchanged. Descriptor coverage at bs 4
  goes 97 → 170. Covered by
  `tests/cublas.rs::cublas_handles_column_major_output`.
- **A two-axis reduction folds into one contraction axis when the operands are
  contiguous over the pair.** Weight gradients summed over `(batch, tokens)`
  (`REDUCE(ADD, Mul, Range, Range)`) now reach cuBLAS as a single `K` axis
  instead of staying on the generated kernel: 23 more descriptors at ViT
  training bs 4, worth ~0.7% (306.4 → 304.4 ms/step in an alternating A/B).
  A non-contiguous pair (the N axis between batch and token in B) is still
  rejected, since it needs per-slice accumulation.
- **Two-axis `(batch, head)` GEMMs stay on cuBLAS at batch > 1.** The library
  path now issues one strided-batched cuBLAS call per outer batch slice instead
  of rejecting the split layout, so ViT training keeps its forward-attention
  GEMMs at every batch size. Measured on an RTX 3060: `vit_train` f32 bs 4 JIT
  **315.5 → 301.6 ms/step (-4.5%)**. `tests/cublas.rs` covers the layout
  numerically and asserts the dispatch through the new `CUBLAS_GEMMS` counter.
- **JIT capture no longer lowers the same plan twice.** `CaptureEntry::Linear`
  and `JitStep::Linear` carry the `Rc<CompiledLinear>` produced by the capturing
  realize, so replay skips the second AST hash/walk (`codegen.ast_nodes`
  516 → 270 on `vit_infer` bs 4). `run_linear` now returns the compiled LINEAR.
- **Cheaper program-cache and scheduler metadata lookups.** `ProgCache` uses an
  O(1) tick map instead of a `VecDeque` scan per hit and keeps no LRU state when
  uncapped; `Scheduler` caches `rngs`/`full_shape`/`axis_types` and
  `realize_many` uses a hash map for large co-realizations.
- **Device buffers return to the free list.** `BufferBase::drop` hands the base
  allocation back to the device allocator's size-capped free list instead of
  releasing it, so steady-state realizes stop paying a driver allocation per
  intermediate: `alloc.backend` 1817 → 1611 (−11%) on `vit_train`. No step-time
  change was measurable; the allocator was not a bottleneck.
- **`remove_bufferize` substitution stops at `AFTER`.** The range-remapping DFS
  used to walk subtrees that belong to previously realized kernels and cannot
  contain the current schedule's ranges. `upat::substitute_local` prunes them:
  eager ViT training bs 1 `subst.nodes` **17.2M → 12.2M (-29%)**,
  `rb.substitute` **0.72 → 0.48 s (-33%)**, step **1.80 → 1.62 s (-8.4%)**.

### Added

- **Automatic eager plans.** `StepPlan` is enabled by default (`AUTORUST_PLAN=0`
  disables) and the benchmark's eager path wraps its step in one: a fixed-input
  step is captured with an empty signature and replayed. ViT train bs1
  **1318 -> 75.5 ms**, bs4 **1340 -> 199.2 ms** (JIT parity), ViT infer bs8
  **402.3 -> 97.5 ms**. Gate:
  `tests/jit.rs::step_plan_matches_non_jit_stateful` (stateful updates, plan vs
  plain, same loss trajectory).
- **`nn::fold_bn_eval`: inference Conv+BN folding.** Folds
  `gamma/sqrt(running_var+eps)` into the conv weight and
  `beta - mean*gamma/sqrt(running_var+eps)` into its bias, materialized as
  parameter leaves, so the conv store is a plain bias(+ReLU) GEMM. ResNet-18
  inference bs32 **317.9 -> 261.2 ms (-17.8%)**, MobileNet bs8 **13.28 ->
  12.06 ms (-9.2%)**; accepted host-GEMM descriptors 1 -> 13. ResNet-50 is kept
  unfolded by default (measured +1.3%). Gated by
  `tests/nn_conv.rs::fold_bn_eval_matches_unfolded_conv` and the benchmark's
  within-process `BENCH_VERIFY_FOLD=1` (max logit diff 2.5e-3 on ResNet-18).
- **Fused FA2 training kernels (forward with saved LSE, dQ, dK/dV).**
  `runtime::cuda_flash_train` implements the backward pass as four CUDA
  kernels: `ar_fl_fwd` also writes the per-row log-sum-exp, `ar_fl_prep`
  computes `D = O . dO`, and `ar_fl_bwd_dq` / `ar_fl_bwd_dkv` recompute
  `p = exp(s - lse)` and accumulate `dQ`/`dK`/`dV` without atomics. Exposed as
  `models::flash_attention::{fused_attention_train, fused_attention_backward}`
  and gated against the graph SDPA autograd by `tests/flash_train_cuda.rs`
  (plain/causal/additive mask/GQA, odd lengths, max diff 4e-7). The scalar
  kernels are at parity with the graph's cuBLAS-based backward (7.24 vs
  7.17 ms/module at N=197; +5% at N=1024), so they are not wired into training;
  a tensor-core backward is the remaining step.
- **Measured attempts for the remaining backlog.** Parallel rendering is 83.5
  ms for 14 kernels on a fresh cache (serial `codegen.sink`); eager
  substitution is 0.464 s/step (12.1M nodes); the beam-search disk cache makes
  a 22.3 s search 0.58 s on a repeat run, and beam timing is ~0 (compilation is
  the cost); every new input shape still recaptures (~0.5-0.9 s). These are
  recorded in `PERFORMANCE_OPTIMIZATIONS.md`.
- **Per-slice accumulation for strided two-axis reductions.** A contraction
  whose two reduction axes are not contiguous in both operands (the ViT
  weight-gradient shapes) is now a sum of per-slice GEMMs: `HostGemm` carries
  `acc`/`a_acc`/`b_acc` and cuBLAS runs the outer slices with `beta = 0/1`.
  ViT-B/16 train bs 4: **212.4 → 200.6/201.2 ms (-5.5%)**, accepted descriptors
  170 → 182. Gate: `cublas_accumulates_strided_two_axis_reduction` plus the
  end-to-end loss parity (~5e-4 over two steps).
- **ReLU epilogue and arbitrary reduce-axis folds for the host GEMM.** A store
  `Max(REDUCE(...), 0)` now peels the clamp into a `relu` descriptor flag that
  the CUDA epilogue kernel applies (numeric gate:
  `tests/cublas.rs::cublas_matches_relu_epilogue`), and reduction axes fold
  generically (not just one/two), so a convolution's `(cin, ky, kx)` can be a
  single K axis when the operands are contiguous. ResNet-18 convs still stay
  generated: the remaining rejects are multi-axis output layouts
  (`epilogue-layout c_axes=4`), not the epilogue.
- **Fused Adam/AdamW/LAMB kernel for OpenCL.** `runtime::cl_optim` is the CL
  port of the CUDA kernel: the 100M-parameter / 32-tensor benchmark goes
  **96.9 → 26.4 ms (107 GB/s), level with CUDA's 25.4 ms**. Gated like CUDA
  (f32, contiguous, `n >= 1024`, `AUTORUST_OPTIM_KERNEL=0` to disable), with a
  graph-vs-kernel numerical test on CL. Vulkan still uses the graph path
  (its compute path needs descriptor sets rather than the generic host-kernel
  seam).
- **`StepPlan` keeps one plan per input signature.** An LRU
  (`set_capacity`, default 4) lets a batch sweep replay every shape instead of
  recapturing; covered by `tests/jit.rs::step_plan_keeps_one_plan_per_shape`.
- **`StepPlan`: a reusable eager plan.** `engine::jit::StepPlan` captures a step
  function once (binding its inputs to dedicated buffers) and replays it
  afterwards, recapturing when the input shapes/dtypes change — the same
  mechanism ONNX uses, now available for eager user loops. The step closure is
  proven to run exactly once (`tests/jit.rs`).
- **Parallel subprocess compilation for the `RUST`/`CC` backends.** Their
  source caches are `Mutex`-guarded and `compile_batch` compiles kernels in a
  bounded thread pool (`AUTORUST_COMPILE_WORKERS`, default `min(cpus, 8)`).
  RUST `vit_infer` fresh-cache cold start 2654 → 2161 ms (-18.6%). CUDA's NVRTC
  batch was already parallel.
- **im2col only when the contraction can amortise the copy.** A re-run of the
  cross-framework matrix showed the unguarded materialization regressed ONNX
  ResNet-0.25 (11.5 → 29 ms) because its K = cin*kh*kw is small. Materializing
  now requires `K >= 576` (`AUTORUST_IM2COL_MIN_K`): ONNX ResNet-0.25 9.7 ms and
  ResNet-18 inference 321 ms / training 289 ms (from 449 ms pre-im2col).
- **Materialized im2col for convolution.** The pooled patch view is now made
  contiguous (once per group, before the output-channel expansion) when the
  patch buffer is at most 64M elements, turning `(x*w).sum(cin,*kh,*kw)` into a
  plain `[bs*oy*ox, cin*kh*kw] @ [..., cout]` matmul on a real buffer. ResNet-18
  on CUDA: inference **378.1 → 326.9 ms (-13.5%)**, training **448.6 → 399.5 ms
  (-11%)**. The contraction still runs as a generated reduce kernel where the
  scheduler split the reduction (>32K reduction ratio); closing that needs the
  host-GEMM decision to happen before `split_reduceop`, so the remaining gap to
  torch is conv lowering, not patch materialization.
- **Fused CUDA optimizer kernel.** Adam/AdamW/LAMB now run one 256-thread pass
  per parameter (`runtime::cuda_optim`) instead of four or five elementwise
  passes, updating the first/second moments and the parameter from one load of
  g/m/v. The 100M-parameter / 32-tensor benchmark goes **102.2 → 26.0 ms**
  (27 → 108 GB/s, level with torch's 26.3 ms); the graph path remains the
  reference and the fallback for non-CUDA, non-f32 and small tensors.
  `AUTORUST_OPTIM_KERNEL=0` forces it, `set_fused_optimizer(false)` from code.
  The eligibility floor is 256 elements, so a ViT-B/16 training step (768-wide
  parameters) uses it too: 217.2 → 212.1 ms and 2,287 → 1,487 launches/step.
  Also: every moment/parameter store is deferred into one schedule, so the graph
  fallback dropped from 66 `create_linear_with_vars` passes to two.
- **ONNX replay plans.** `OnnxModel::run` captures the whole graph's LINEARS on
  the first call for an input shape and replays them afterwards (inputs are
  copied into the plan's buffers; a shape change recaptures;
  `set_plan_cache(false)` forces the eager path). The profiled ONNX step was
  ~90% host scheduling (`sched.rangeify` 11.1 ms vs `exec.kernel` 0.049 ms), so
  this is the dominant fix: **ONNX MNIST bs 32 11.7 → 1.7 ms (-86%)**, **ONNX
  ResNet-0.25 bs 8 73.8 → 12.0 ms (-84%)** on an RTX 3060. Outputs alias the
  plan's buffers and are valid until the next `run`. `Jit` gained public
  `lower_capture`/`replay` for callers that drive their own capture.
- **Host-GEMM dispatch accounting and keyed CUDA `KPROF` rows.** With
  `AUTORUST_GEMM_STATS=1`, `prof::report` prints every library-GEMM extraction
  attempt: accepted `(m,n,k,batch)` families and rejected-by-reason counts
  (`batch-layout`, `epilogue-*`, `divisibility`, …), so a batch-size dispatch
  cliff is visible as data instead of hidden in wall time. CUDA `KPROF` rows now
  carry the same content key as the host rows, and cuBLAS calls print
  `KPROF host_gemm <secs> <layout-key>` with a stream-synchronized sample, so
  library time is no longer invisible next to generated kernels. Setting
  `AUTORUST_GEMM_STATS=all` also records a structural hint per rejection
  (`op=Add<Mul`, `c_axes=4 batch_axes=2`, `reduce_op=Some(Add) srcs=3`), and
  `b-index-not-affine`/`n-axis-ambiguous` print the offending index expression
  and the `a`/`b`/`c` stride maps, which is how the remaining groups were
  classified as "needs staging" rather than "needs more parsing".
  `BENCH_ITERS=0` priming runs now report `cache primed; no timed samples`
  instead of `NaN`.
- **Cooperative-matrix GEMM for the Vulkan backend (f16 in, f32 out).** The
  library-GEMM seam recognises an f16 matmul with an f32 accumulator
  (`AUTORUST_VK_GEMM`) and runs one `OpCooperativeMatrixMulAddKHR` kernel
  (`runtime::coopmat`, shapes patched at load time) instead of the generated
  reduce loop: **2.0x the device speed at 2048³** (1.71 vs 3.42 ms) and exact
  against the interpreter. `bf16` storage runs on the same path through two
  element-wise conversion modules (bf16 -> f16 for the operands, f32 -> bf16 for
  the result), gated by `USE_TC` where the graph's 16-bit accumulation would
  otherwise change. Plain 16-bit matmuls that the matcher does not describe keep
  the generated kernel, so their numerics are unchanged. The bias **and the
  residual** of a transformer block are fused into that epilogue (`x + linear(x)`
  is one MMA dispatch), which took `vit_infer_bf16` from 155 to 132 ms.
- **Bounded fence waits (Vulkan).** `vkWaitForFences` used `u64::MAX`, so a lost
  device or a hung kernel hung the process instead of failing. It now times out
  after 60 s with an error naming the cause and pointing at `VK_SYNC=1`.
- **Loop unrolling for non-divisible trip counts (Vulkan).** The emitter only
  unrolled loops whose extent was a multiple of 8, which excluded the attention
  loops (197) and most conv tails; the driver's SPIR-V frontend does not unroll on
  its own. It now runs `ext / 8` full chunks and emits the leftover iterations as
  straight-line copies (same load order, same accumulator order, so identical
  numerics): `vit_infer_bf16` **130 -> 114 ms**, ViT bs 8 130 -> 117,
  UNet 19.5 -> 18.1, LLM decode 22.4 -> 19.8 (now ahead of OpenCL).
- **Device-buffer reuse for the Vulkan backend.** The engine's buffer cache is
  bypassed by every backend's `_free`, so a schedule re-allocated one
  `vkCreateBuffer`+`vkAllocateMemory` per kernel per step (~0.25 ms each).
  `VkDevice` now reuses freed buffers by size: ONNX-MNIST **15.5 -> 11.6 ms**
  (now level with OpenCL's 12.0) and ONNX ResNet-0.25 **82.6 -> 73.5 ms**
  (1.02x, was 1.23x).
- **Cooperative-matrix ("tensor core") probing for the Vulkan backend.** The
  device probe now enumerates `VkPhysicalDeviceCooperativeMatrixPropertiesKHR`
  and enables `VK_KHR_cooperative_matrix` only when a 16-bit-in/f32-out shape
  exists (`DeviceInfo::coop`). `tests/vk_coopmat.rs` runs a hand-written
  `OpCooperativeMatrixMulAddKHR` GEMM and pins both numerics (exact against an
  f64 reference) and throughput: **7.7 TF/s at 1024³ and 9.8 TF/s at 2048³,
  3.2x / 2.5x the plain-SPIR-V GEMM**. The renderer does not emit it yet — the
  integration plan and the numerics trade-off are in `CURRENT_STATE.md` §9.53.
- **Vulkan backend (`VK`, `backend_vk`, default).** A fourth GPU backend that
  needs no vendor SDK: kernels are emitted as **SPIR-V directly** from the
  linear UOp list (`renderer::spirv`) instead of C text, and `libvulkan` is
  loaded at run time through `ash`'s `loaded` feature (bindings only; nothing
  links against the loader). Machines without a loader, a compute device, or
  `storageBuffer8BitAccess` simply do not register `VK`, like a box without
  CUDA. Supports f32/f16/bf16 (bf16 as a 16-bit storage pattern widened to f32),
  all integer widths, vectors, local memory + barriers, and `VK:N` device
  selection. The same scheduler/tuning is reused, so performance lands within
  ~20% of the OpenCL backend on most workloads (faster on elementwise, `add`,
  `gelu`, TN/NT GEMM and AdamW; slower on `LOCAL`-memory reductions and on
  decode-heavy graphs, which pay one command buffer per kernel — see
  `PERFORMANCE.md`). The emitter unrolls small constant-trip-count loops 8x
  because the NVIDIA SPIR-V path does not (its OpenCL compiler does): that alone
  took `softmax`/`layernorm` from 2.1/2.2 ms to 1.09/1.19 ms, ahead of OpenCL's
  1.38/1.42, and consecutive launches are batched into one command buffer (closed
  at each executed linear via the new `Prog::flush` seam) which took LLM decode
  from 25.6 to 21.3 ms. Register arrays (GEMM accumulators, the per-lane tiles in
  the conv kernels) are promoted to SSA with loop-carried `OpPhi`s instead of
  round-tripping through `Function` memory, worth 20% on ResNet-18 inference
  (403 -> 321 ms) and 14% on UNet; the loop carry has to be read out of the
  continue block, which `tests/vk_backend.rs` pins (using the header phi dropped
  the last iteration of every reduction).
- **Backend memory reporting.** `Compiled::memory_info()` returns
  `MemoryInfo { total, free }`, implemented through
  `BackendAllocator::memory_info()`: CUDA uses `cuMemGetInfo_v2`, OpenCL reports
  `CL_DEVICE_GLOBAL_MEM_SIZE`, Metal reports its recommended working set, and
  host memory reports `None`.

### Fixed

- **Optimizer state is created on the parameter's device.** `LAMB::new` built
  the bias-correction tensors, `m` and `v` with no device, so an optimizer over
  non-default devices (e.g. `CL` params in a CUDA process) failed with "all
  buffers must be on the same device".
- **CUDA NVRTC fallback.** The device probe now tries every installed NVRTC
  newest-first and pins the first whose PTX the driver actually loads, instead
  of disabling CUDA when the newest toolkit is newer than the driver. An
  explicit `AUTORUST_NVRTC_LIB`/`NVRTC_PATH` still pins one library.

## [0.1.1] - 2026-11-11

Performance and correctness release; no API changes. Net effect on the
benchmark suite: ViT-B/16 training CUDA 332 -> 96 ms, CPU inference 425 -> 171 ms
at bs 1 and 1902 -> 1188 ms at bs 8, CPU training 1784 -> 1634 ms; MNIST CUDA
39 -> 22 ms. Six of the eight original bench rows are now within 2x of eager
torch (`torch.compile` off), and warm cold starts drop from 505/697/1279/6013 ms
to 216/474/760/2498 ms. The ViT training gap is batch-dependent rather than
fixed overhead: the generated f32 GEMM is close to saturation at batch 1
(per-image 98.8 -> 78.6 ms on CUDA through bs 8) while cuBLAS keeps improving
(45.8 -> 22.1 ms/image), so at batch 4 the step is 317 vs 102 ms; at batch 64
MNIST is 1.3x off torch on CUDA and 2x faster on CPU. The benchmark matrix also
grew beyond the two models: a GEMM sweep (f32/bf16, transposed and batched), a
memory-bound op set, an AdamW-only step, ResNet-18/50, MobileNetV2, a small
UNet, an autoregressive prefill/decode pair (GQA, RoPE, KV cache, sampling), a
bf16 ViT rows (inference and training), the ONNX runner (MNIST and a
width-0.25 ResNet-18) and a frozen-plan comparison (dev-only; see
`PERFORMANCE.md`).

### Performance

- **Persistent rendered kernels.** The render result (ProgramInfo, ELF
  signature, source) is cached under the AST content key plus the codegen
  settings and a fingerprint of the running binary, so a warm process skips the
  ~30 codegen pipeline stages per kernel. ViT-B/16 training cold start
  7.2 s -> 2.5 s; ViT infer bs8 1.26 s -> 0.49 s. `CACHELEVEL=0` disables it.
- **GPU GEMM tiling.** Both contraction orientations are scored by block
  threads (with a register/thread trade below one warp), and the direct-conv
  tile reuses the 5x5 weights across two output rows. ViT-B/16 training CUDA
  332 -> 216 ms, MNIST bs512 39 -> 22 ms.
- **Summed gradient contractions** (`dQ@Wq + dK@Wk + dV@Wv`) are materialized
  per contraction so each reaches the matmul tiler instead of one scalar
  multi-reduce kernel (ViT training 216 -> 179 ms).
- **Pointwise contraction operands inside reduce bodies** (GELU derivative,
  dropout mask, softmax backward) are staged once, making those backward GEMMs
  affine (179 -> 94 ms).
- **CPU packed GEMM coverage.** Interior host-backend contractions (the FFN
  expansion, attention's biased Q/K) are split and their operands staged so the
  packed CPU GEMM takes the matmul and the epilogue runs as its own kernel.
  RUST ViT-B/16 inference 425 -> 171 ms at bs 1, 1902 -> 1188 ms at bs 8;
  training 1784 -> 1634 ms.
- **CPU host GEMM** reads a row-major `A` directly and packs a `[n, k]` weight
  once per GEMM (and once per batched call), 315 -> 464 GFLOP/s on
  197x768x3072.
- **Fused attention for eval-mode inference**: `ViT::forward_with` /
  `TransformerBlock::forward_with` dispatch to the FA2 kernels on CUDA/CL/Metal
  (CPU and training keep SDPA), with the q/k/v projections co-realized and
  pinned.
- `AUTORUST_PROF` per-stage codegen timings, `AUTORUST_KPROF` per-program kernel
  timings for host backends, and `AUTORUST_MM_DEBUG` matmul/`HostGemm`
  rejection reasons.

### Fixed

- The graph model (`TransformerBlock::attn`) no longer routes through the
  op-level FlashAttention port by default: the default `Settings::attention`
  selects the fused tensor-level kernels, and the 16x-slower op-level port is
  now only used with `FA=1`. The default follows tinygrad's model and uses SDPA.
- CUDA bf16 kernels whose lanes vectorized 2-wide failed to compile:
  `cuda_bf16.h` typedefs `nv_bfloat162` but has no `make_nv_bfloat162`, so the
  generated constructor was undefined. The renderer now defines it via
  `__halves2bfloat162`; bf16 elementwise kernels (and the benchmark's bf16 GEMM
  sweep) compile and run on CUDA.
- The `RUST` renderer keeps rustc's loop vectorizer off integer shift chains
  (the `threefry` RNG kernels): the vectorizer is exponential in the chain
  length (20 shifts 0.26 s, 24 1.8 s, 28 16 s, the RNG kernel 80+), so a cold
  batch-2 training run could spend over ten minutes in `rustc` on one kernel.
  The loop index goes through `core::hint::black_box` for those kernels only;
  every other kernel keeps its vectorized work-item loop.
- The runtime cache is keyed by the buffer signature as well as the compiled
  artifact: a cached runtime could be reused for an artifact with different
  slots, launching with an out-of-bounds buffer index.
- `ProgramInfo` is derived from the linearized body, the same traversal the ELF
  signature uses (the linearize cleanups can rewrite nodes).
- The realize decision and the `non_removable` mark survive the reduce rebuild
  in `stage_reduce_prologue` (a rebuilt reduce otherwise lost its boundary).
- Optimizer moments are built as store-free buffer leaves, as
  `zeros_param`'s contract states (they previously carried a `full` store that
  every realize re-executed).

## [0.1.0] - 2026-11-10

Initial release.

- `Tensor` front-end, autograd, `nn` layers and optimizers, RNG and state
  (safetensors).
- Interned UOp graph, pattern matcher, rangeify scheduler, codegen and the
  reference interpreter.
- Backends: `INTERP`, `RAYON`, `RUST`, `CC`, `CL`, `CUDA` and `METAL`.
- Dynamically loaded cuBLAS / CLBlast / MetalPerformanceShaders host GEMM, a
  packed CPU GEMM, and fused FlashAttention kernels (scalar, TF32 and bf16
  tensor cores).
- `TinyJit` capture/replay, BEAM search, ONNX import and frozen CPU plans.
