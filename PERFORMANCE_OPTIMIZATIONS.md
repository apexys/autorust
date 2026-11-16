# Performance optimizations: open work and measured results

Date: 2026-11-16. Machine: RTX 3060 (sm_86), Ryzen 9 3900X, CUDA 12.4.

This is the single performance document. It supersedes `PERFORMANCE_STEPS.md`,
`PROPOSED_OPTIMIZATIONS.md` and the 2026-11-13 `PERFORMANCE_REVIEW.md`. The
canonical dated experiment log is `CURRENT_STATE.md` (§9.62-9.99 for this work);
the cross-framework tables are in `PERFORMANCE.md`.

How to read it:

- **TODO** is open work, grouped by area. Each item has the evidence, the
  change, the expected result and the risk.
- **DONE** is landed work with its measured result, plus work that was
  measured and *not* landed. Keeping the rejections here is the point: do not
  repeat them without a new reason.
- Protocol for every item: alternating builds/processes, medians on a stated
  shape, the full test suite before committing. No single-sample wins.
- IDs (`S`, `B`, `C`, `A`, `E`, `T`) are stable for cross-references.

## TODO

### Scheduler and cold start

- **S1. Serialized (AOT) lowered plans, or a SINK-content schedule cache.**
  Evidence: a replay step is 38.5 ms, but the first step in a warm-cache
  process is 570-595 ms. The cacheable host work is only ~70-120 ms of that
  (`realize.schedule` ~61 ms + `callify` ~11.5 ms at `vit_infer` bs 8); the
  rest is GPU clock ramp and first-launch driver work. Change: persist each
  captured LINEAR's ordered CALL list, per-program content key, `ProgramInfo`,
  launch dims, buffer slots and scalar vals, keyed by `(graph content, device,
  settings, dtype/shape)` plus a binary fingerprint; or hash the callified SINK
  and reuse the split/memory plan. Risk: buffer-slot and arena-id reuse. The
  existing content-keyed `CompiledProg` reuse is the precedent. Effort:
  medium-large.
- **S2. Symbolic / shape-generic plans.** Evidence: every shape change
  recaptures — `ms_cold` 494/619/726 ms at bs 1/4/8 against replay medians of
  12.8/37.8/68.2 ms. The port has no symbolic machinery: `ProgramInfo::
  launch_dims` ignores `var_vals`, `vals()` is empty, `SInt::U` is never
  constructed, and there is no `Variable`/`DefineVar` uop. Change: build that
  subsystem before advertising one plan per model. Effort: large; symbolic
  bounds are the risk.
- **S3. Shared substitution worklist and fewer removable STAGEs.** Evidence:
  `rb.substitute` is ~0.46 s/eager step, with 12.1M visited nodes for 16.7k
  changed (725:1); `rb.walk` is only 8.9 ms, so the per-call DFS is the cost.
  The DFS is already memoized within a call, so the remaining ideas are a
  schedule-scoped cross-call cache or creating fewer bufferize STAGEs upstream.
  Eager path only; JIT replay is unaffected. Effort: large; gate on the kernel
  multiset.
- **S4. Parallel rendering with an owned, `Send` kernel IR.** Evidence: fresh
  rendering is 83.5 ms for 14 kernels (5.97 ms each, `codegen.sink` 80.3 ms)
  out of a ~595 ms cold start; it is serial because `codegen.sink` mutates the
  `Rc` arena. Change: hand post-scheduled kernels to workers as owned IR (or
  per-worker arenas), render in parallel, merge serially. Cold path only.
  Effort: large.
- **S5. Per-kernel opt search / BEAM as a default.** Evidence: the disk cache
  the item needs already exists (`BEAM=1`: 22.3 s cold → 0.58 s warm) and the
  opt-in win is real (`vit_infer` bs 4: 38.30 → 32.93 ms, -14.0%; `BEAM=2`
  32.36 ms, -15.5%). The cold cost is NVRTC compilation, not the timing loop
  (`distinct_keys == compiled`). A bounded 24-action set only reached -2.3%.
  Change: parallel compilation (needs S4) or a shipped cache keyed to the
  codegen fingerprint. Keep it opt-in until then.

### Backend portability

- **B1. Port the fused optimizer to Vulkan.** Evidence: AdamW/100M is 25.7 ms
  on CUDA, **26.4 ms on CL**, and 95.0 ms on VK (graph path). CL is landed
  (`runtime::cl_optim`); Vulkan needs explicit descriptor sets rather than the
  generic `run_host_kernel` source seam. Target 30-40 ms. Effort: medium.
- **B2. Which CUDA-only paths are portable.** TF32 cuBLAS, the cuBLASLt
  epilogues, the kxk `wmma` conv and conv-only TF32 are CUDA-only. OpenCL gets
  `-cl-fast-relaxed-math` and the im2col host GEMM; Vulkan gets cooperative
  matrices (f16) and no compiler fast-math flag; `CC` gets `-ffast-math`;
  `RUST` has no stable fast-math flag; Metal keeps `METAL_FAST_MATH` explicit
  because fast math folds the attention `-inf` masks. Porting the kxk kernel
  needs a CL/VK tensor-core path; there is no measured demand yet.

### Conv and GEMM

- **C1. HostGemm staging for the remaining non-affine operands.** Evidence:
  48 `b-index-not-affine` rejects at ViT train bs 4 are the attention
  out-projection weight gradient, whose B operand is `[b, n, h*d]` while it is
  stored `[b, h, n, d]` (the merged `(h,d)` axis interleaves `n`). The
  graph-level `contiguous()` form was neutral (182 descriptors unchanged, step
  199.5 vs 199.4 ms), so a real staging path has to happen inside the host GEMM
  (temp + gather + GEMM + copy-out). Effort: medium; descriptor/freeze surface.
- **C2. Winograd F(2x2, 3x3).** Only if the implicit-GEMM gap remains after
  the landed kxk kernel; the kernel already covers the low-K stem and the 3x3
  stride-1 case, which was most of the gap.
- **C3. Autograd for host conv kernels.** The kxk kernel is forward-only;
  training keeps the graph. See A1 for the general "host kernel + VJP" problem.

### Attention

- **A1. Tensor-core attention backward and autograd wiring.** Evidence:
  attention is 86 ms of the 196 ms ViT train step (44%; one module is 7.2 ms,
  the backward is 57 ms). The scalar FA2 training kernels are landed and
  numerically correct but at parity (86.9 vs 86.0 ms at N=197; +1% at 512, +5%
  at 1024) because graph SDPA's backward is already batched cuBLAS GEMMs. A
  tensor-core backward + fused forward targets 25-35 ms, i.e. **-50-60 ms/step
  (25-30%)**. It needs: an LSE-writing FA2 forward, dQ and dK/dV kernels, a
  graph op with a `grad_of_op` rule, scheduler/session lowering to a
  `HostCall`, and freeze serialization. `grad.rs` has no custom-VJP hook and
  `Ops::CustomFunction` has no lowering, so this is cross-cutting. Effort:
  large. The landed API and its 4e-7 gate are the reference.

### Elementwise, fusion and launches

- **E1. CUDA graphs for launch-heavy paths (decode).** Evidence: `llm_decode`
  is ~16.5-17.2 ms/token with ~1,660 launches/step. Capture is implemented and
  is a valid GPU-time instrument (ViT train replays in 196 ms against a 200 ms
  wall), but the wall win on ViT is only 3.8 ms and it is numerically wrong for
  training: the Adam bias-correction state (`b1_t`/`b2_t`) is mutated outside
  the pinned buffers, so replays use stale values. Next: make the captured
  region state-explicit (stable bias-correction buffers) or capture only the
  forward/backward LINEARS and run the optimizer outside. Effort: large.
- **E2. Graph-level softmax/layernorm with autograd.** Forward-only host
  kernels were 2.1x on softmax and ~5% on layernorm, but neutral end-to-end
  (readback/attention dominates) and unusable in training. The follow-up is a
  graph op with a grad rule; the measured share is small on matmul-bound
  models, so it is deprioritized.
- **E3. Constant-fill inlining and RNG-counter batching.** Evidence: ViT
  train's elementwise launches are mostly fills/copies of constants
  (`E_4608_32_4`, `E_18432_32_4`) and 2-thread RNG counter increments (`E_2`),
  not fusible elementwise stores. Fold `full(c)` constants into consumers;
  batch or host-accumulate the counter increments. `E_2` is one-time in
  inference (3025 launches at iters=1 and 5) and mostly one-time in training
  (6174 at iters=1 vs 6771 at iters=4), so the steady-state value is limited.
- **E4. Masked elementwise tails (`PADTO`).** A vector body plus a scalar tail
  would let prime extents (ViT's 197) use wider blocks instead of the 2/8-thread
  blocks the divisibility heuristic forces. Lower priority.

### Measurement and tooling

- **T1. CUDA-event KPROF.** The current profiler synchronizes per launch and
  serializes the workload; event timing would be accurate without that.
- **T2. Phase-separated profiles.** Build vs first capture vs compile vs
  replay as separate tables, plus an exclusive scheduler timer (`rw.pm_secs` is
  inclusive and nests).

## DONE

### Conv and GEMM

- **1x1 conv as a matmul.** `[B*H*W, Cin] @ [Cin, Cout]` on CUDA, then the NCHW
  permute. MobileNetV2 -7%, ResNet-50 -2-6%, UNet ~-4%; ResNet-18/CPU neutral.
  `AUTORUST_CONV1X1`, `AUTORUST_CONV1X1_MIN_M` (2048).
- **Low-K 3x3 im2col floor.** `cin*kh*kw >= 256` on CUDA (576 on CPU/CL/VK).
  UNet's K=288 layers moved to im2col + cuBLAS and improved 11.3-12.4 →
  10.2-10.3 ms (~10-14%); ResNet-18/ONNX neutral.
- **Conv-only TF32.** `AUTORUST_CONV_TF32=1` (now part of Fast mode): ResNet-18
  -7-11%, UNet -12%; changes convolution numerics.
- **Tensor-core kxk implicit-GEMM conv.** A bounded TF32 `wmma` kernel that
  generates patches into the A tile (no im2col copy), 64 rows x 64 channels per
  block; covers square k, groups=1, stride 1-2, symmetric padding. It is picked
  for the measured 3x3 stride-1 case and for low-K convs, notably ResNet's 7x7
  stem (K=147). RTX 3060, bs 32: **ResNet-18 214-216 → 118 ms**, **ResNet-50
  83 → 35.8 ms**, MobileNet neutral. The row floor is 4096
  (`AUTORUST_CONV_TC_MIN_ROWS`); 16384 reproduces the old 157 ms. ONNX `Conv`
  also uses it when the input is already realized: **ONNX ResNet-18 w=0.25 bs 8
  9.24-9.66 → 6.94-7.38 ms (-29%)**. Gates: `tests/cuda_conv_tc.rs` (3x3, 7x7
  stem, small map, stride-2 fallback, graph gradients).
- **Fast math mode is the default.** `MathMode::Fast`: TF32 cuBLAS,
  generated/custom tensor cores, conv TF32, `CC -ffast-math`, OpenCL
  `-cl-fast-relaxed-math`. `AUTORUST_MATH=exact` restores pedantic f32. Versus
  Exact: **ResNet-18 264.6 → 118.0 ms (-55%)**, **ResNet-50 88.6 → 35.8 ms
  (-60%)**, **ViT-B/16 infer bs 8 82.8 → 65.6 ms (-21%)**.
- **Inference BN fold.** `nn::fold_bn_eval` turns conv+BN into a bias(+ReLU)
  store: ResNet-18 317.9 → 261.2 ms (-17.8%), MobileNet -9.2%; ResNet-50 stays
  unfolded by default.
- **cuBLASLt bias(+ReLU) epilogue.** 98 of 182 host GEMMs per ViT train step
  fuse the bias into the GEMM: -0.6% train, -1.0% infer. Per-column scale and
  elementwise residual still use the separate epilogue.
- **TF32 opt-in before Fast mode.** `AUTORUST_CUBLAS_TF32=1` was -5.8% on ViT
  train; it is now the Fast default.
- **Measured, not landed:** depthwise host kernel (the realized-input boundary
  costs more than it saves: 69.9 vs 56.3 ms unpinned); conv-transpose
  pixel-shuffle (neutral on CUDA, 2x worse on CPU); 3x3 stride-2 dispatch
  (ResNet-50 85.7 → 1132 ms via the bottleneck mid-graph realize); 1x1 permute
  removal (general `hw=1` im2col 1.88 vs 1.42 ms, `k=1` tensor-core 1.56 vs
  1.48); lowering the im2col floor further (ResNet-50 83.2 → 85.3 ms); 1x1
  `MIN_M` 512 (MobileNet consistently worse); reduction-partial block counts
  128/512/1024 (all within 0.244-0.253 ms).

### Scheduler and cold start

- **AFTER-pruned substitution.** `subst.nodes` 17.21M → 12.21M (-29%),
  `rb.substitute` 0.715 → 0.478 s (-33%), eager ViT train step -8.4%.
- **`StepPlan` eager API.** One plan per input signature (LRU, capacity 4),
  `AUTORUST_PLAN=0` to opt out. ViT train bs 1 1318 → 75.5 ms, bs 4 1340 →
  199.2 ms; ViT infer bs 8 402.3 → 97.5 ms. Numeric/state gate in
  `tests/jit.rs`.
- **`compile_batch` pool for RUST/CC.** Bounded scoped workers (Mutex caches):
  cold start 2654 → 2161 ms (-18.6%).
- **BEAM disk cache.** Keyed by AST content, width and device: the same
  `BEAM=1` run is 22.3 s cold → 0.58 s warm.
- **Measured, not landed:** library GEMM before `split_reduceop` (325.99 →
  326.80 ms, same 38 rejects); keyed per-shape tile tuner (2048³ candidates
  within 0.2-2%, ViT prefers the default); bounded 24-action opt search (-2.3%,
  reverted); symbolic plans (no subsystem); cubin cache (module load 1.9 ms /
  0.3%); SINK-content schedule cache (needs the plan re-interning of S1).

### Attention

- **Scalar FA2 training kernels.** `runtime::cuda_flash_train` +
  `fused_attention_train`/`fused_attention_backward`, correct to 4e-7 vs the
  graph SDPA autograd (odd lengths, masks, GQA). At N=197 it is 86.9 vs 86.0 ms
  (parity); +1% at 512 and +5% at 1024. Not wired into autograd; kept as the
  reference/gate for A1.
- **Fused eval attention.** ViT/LLM inference use the fused flash path;
  training keeps graph SDPA by design.
- **Measured, not landed:** ncu coalescing of the `dS` kernel (35.8 → 2.3 ms in
  ncu, neutral in situ: 86.11 vs 86.02 ms); scalar fwd+bwd as a drop-in
  (parity).

### Elementwise and launches

- **Fused full reductions.** `sum_with`/`max_with` one streaming pass + finish:
  `sum` 0.370 → 0.254 ms, `amax` 0.397 → 0.248 ms (265/270 GB/s), deterministic
  partial order.
- **Fused optimizer.** CUDA 104 → 26 ms (torch parity), CL 96.9 → 26.4 ms
  (numeric gate on CL); `MIN_ELEMS` 1024 → 256 took ViT train -2.3% and cut
  launches 2287 → 1487.
- **Measured, not landed:** a generic elementwise fusion pass (the composition
  is fills/copies/counter updates; measured opportunity ~3 ms + ~1 ms, not the
  assumed 20-40 ms); forward-only softmax (2.1x kernel, ~8% memops, neutral
  LLM/ViT) and layernorm (~5%, neutral ViT); multi-stream scheduling (q/k/v
  overlap 5.4-5.6%, <1% end-to-end); broadcast-rhs matmul fold (ViT train
  -16%, no candidate at LLM B=1).

### Backend

- **CL fused optimizer.** 96.9 → 26.4 ms, CUDA parity; `tests/nn_optim.rs`.
- **Math-mode compiler flags.** `CC` `-ffast-math`: amax 2.38 → 2.11 ms, add
  10.64 → 10.20 ms, but gelu 9.38 → 10.63 ms (mixed); OpenCL
  `-cl-fast-relaxed-math`: softmax 1.065 → 0.982 ms, other memops neutral.
  `RUST` has no stable fast-math flag, Vulkan emits SPIR-V with no compiler
  fast-math option, and Metal keeps `METAL_FAST_MATH` explicit.

### Memory

- **Allocator free list.** Base allocations return to a capped free list:
  `alloc.backend` 1817 → 1611 (-11%), `alloc.reuse` 206, **step time unchanged**
  (219.99 vs 220.97 ms). Closed: the allocator is not a bottleneck (0.010 ms
  per alloc, ~6 ms/step).

### ONNX

- **Replay plan.** MNIST 11.7 → 1.8 ms, ResNet-0.25 73.8 → 9.7 ms; recaptures
  on a shape change, `set_plan_cache(false)` forces eager.
- **`Conv` → tensor-core kernel.** -29% on ONNX ResNet (above).
- **Graph preprocessing.** Measured skip: 0 `Constant` nodes; 16 `Identity`
  nodes alias to 0 kernels; the replay is 29 kernels/step; torch already
  folded BN; the remaining gap was conv lowering, not graph work.

### Tooling

- Keyed KPROF (program content, not name), HostGemm accept/reject detail, the
  `alloc.*`/`cuda.module_load` counters, `host_gemm.lt`, and
  `BENCH_LOSS`/`SEED`/`ITERS=0`.

## Appendix: the 2026-11-13 review, mapped

| review finding | disposition |
|---|---|
| HostGemm batch-size coverage cliff | addressed (170-182 descriptors; see DONE) |
| Eager execution repeats host work | `StepPlan` (DONE) |
| Whole-subtree substitution (8.5M visits) | AFTER pruning (DONE) + S3 (open) |
| Serial/duplicate cold compilation | `compile_batch` (DONE) + S4/S1 (open) |
| Batch scaling in ViT training | attention/conv were the cause (DONE; A1 open) |
| Fused attention backward | scalar landed at parity; TC backward = A1 |
| CUDA graphs | capture landed/instrumental; training-state fix = E1 |
| Tiling cache / BEAM | disk cache + opt-in BEAM (DONE); default = S5 |
| Cubin cache | measured 1.9 ms / 0.3%, skip |
| Allocator at large batches | no bottleneck; free list DONE |
| Symbolic batch plans | no symbolic subsystem; S2 |
| Parallel cold lowering | compile batch DONE; rendering = S4 |
