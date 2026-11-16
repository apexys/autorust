Status: items 1-10 were evaluated in `CURRENT_STATE §9.62` (4 landed, 2 measured
decisions, 4 skipped). New burn-derived items 11-15 are below. The first landed
follow-up is the CUDA small-batch host-GEMM gate (2026-11-16, `CURRENT_STATE
§9.81`): MNIST CUDA 52.9 -> 23.3 ms/step.

1. Small per-kernel opt search (LOCAL/UPCAST candidates), cached
Why: the optimizer's 3,125,000-element kernels were stuck at LOCAL 2 because the heuristic must divide the extent; a 6-candidate sweep measured on the real buffers would have found the better launch. The same class hits every odd extent (ViT's 197-derived elementwise kernels).
Try: candidate set {(up4,loc2),(up4,loc8),(up2,loc16),(up1,loc32),(up4,loc32),(up8,loc4)}, bounded to ~6, timed after 2 warmups on the capture path only, winner cached by (device, kernel content key, shape signature) in the existing disk cache. Reuse opt_search rather than writing a new searcher.
Cost/risk: first call only; +ms per kernel at capture. Measure: per-kernel candidate table (AUTORUST_AUTOTUNE_KERNELS=1), A/B on vit_infer, vit_train, mnist, resnet18, memops.
2. BEAM width 2–4, measured and cached
Why: the review says keep BEAM opt-in until candidate generation/timing are parallelized. Try the smallest widths on a fixed set of kernels to see if the search pays for itself once cached by kernel+shape.
Try: BEAM=2 and BEAM=4 on vit_infer/gemm/memops; record cold cost, cache hit rate, steady-state win. If the search is too slow (>1 s/kernel), parallelize candidate timing or drop it.
3. Library GEMM before split_reduceop (the conv blocker)
Why: im2col now materializes the patches, but split_reduceop rewrites any reduce with ratio ≥ 32768 into partial kernels before host_gemm::extract sees a full contraction — the ResNet conv stays generated (327 ms vs torch 17 ms).
Try: skip the reduce split when the store is a candidate contraction (or run extract on the pre-split AST); A/B ResNet-18 infer/train. This is the cheapest continuation of the conv work.
4. Eager plan cache (the other half of P0 #2)
Why: eager ViT training is ~20x slower than JIT replay; only ONNX got a plan so far. The Jit + input-binding machinery already exists.
Try: a session-level plan(step_fn) API keyed by input shapes, recapture on change, AUTORUST_PLAN=0 escape. Measure: eager ViT/MNIST step time and capture cost.
5. Cubin cache + module-load measurement
Why: on a PTX-cache hit the driver still re-loads modules every process (F5 cost #4). The review explicitly asks to measure before building it.
Try: time cuModuleLoadData across a cold/warm process; if it is a meaningful share of ms_cold, cache driver-ready cubins keyed by exact SM/toolkit, PTX fallback.
6. SDPA/FFN epilogue fusion after HostGemm decisions
Why: elementwise kernels are ~60% of generated CUDA training time and many are sub-50 µs launches; the review asks to fuse only where it does not hide a library GEMM.
Try: fuse the GELU/dropout/epilogue chains that the host-GEMM split leaves on either side of a matmul; verify the kernel multiset and that the HostGemm still fires. Measure: generated E_* time and launch count on vit_train.
7. Allocator behavior at larger batches
Why: the review did not measure it and claims nothing; it is a cheap question that could invalidate a hidden assumption.
Try: ViT bs 8 and UNet at bs 8, track memory_info, reallocation counts (AUTORUST_PROF counters) and prune/alloc traffic. Either close the question or find a new lever.
8. RUST/CC compile_batch parallelism
Why: CPU cold start still compiles subprocesses serially; CUDA already batches. Blocked on Comp being Send + Sync (compilers hold RefCell caches).
Try: switch the CPU compiler caches to Mutex, add a bounded compile_batch with std::thread::scope; gate worker count. Measure: RUST/CC fresh-cache ms_cold on ViT/ONNX.
9. CUDA graphs for LLM decode
Why: decode is launch-bound (hundreds of launches per token); the optimizer is now one kernel per parameter, so decode is the remaining candidate.
Try: capture the decode step into a CUDA graph after the JIT plan, replay per token; requires replay-safe buffer lifetimes and scalar updates. Fall back to the current path.
10. Bufferize/substitution rewrite
Why: rb.substitute is ~363 ms/step and 8.5M node visits on eager ViT — the largest measured scheduler hotspot — but the plan cache (item 4) removes most eager value.
Try: batch compatible remove_bufferize substitutions into a shared traversal and prune unaffected subtrees; gate on the kernel-multiset check, not just final outputs.11. Implicit-GEMM convolution with tensor-core tiles (burn/cubek cross-check)
Why: burn 0.22's `burn-cubecl/src/kernel/conv` runs a per-shape autotuned set of direct,
depthwise, 1x1-im2col and implicit-GEMM routines, the last with MMA/CMMA tiles, async staging
and TMA. Our ResNet-18/UNet CUDA gap (17x/12x vs torch, PERFORMANCE.md#conv-models) is the
largest absolute gap in the matrix, and this is the path that closes it in burn.
Try: extend the conv candidate extractor to emit an implicit-GEMM contraction with a
tensor-core tile on sm_80+, keep the direct/im2col paths as fallbacks, and add a dedicated
depthwise kernel for `groups == channels`. Gate with the same numerics check as the FA2 TC path.
Measure: ResNet-18 infer/train, ResNet-50 infer, MobileNetV2, UNet at bs 32/4; kernel table and
GFLOP/s before/after.
Status (2026-11-16): started with the 1x1 case. CUDA 1x1 convs (groups=1,
stride=1, dilation=1, no padding, >=2048 rows) now lower as a plain matmul:
MobileNetV2 -7%, ResNet-50 -2%, UNet -4%, CPU/ResNet-18 neutral
(`CURRENT_STATE §9.86`, `tests/conv1x1.rs`). A hand-written depthwise host
kernel was also tried and rejected: it needs a realized-input boundary that
costs more than the kernel saves (MobileNet 69.9 vs 56.3 ms unpinned graph), and
a graph-level 4-D depthwise tiling tweak was neutral (`CURRENT_STATE §9.87`).
The 3x3 layout specialization also landed: the im2col floor is 256 on CUDA
(576 elsewhere), which moves UNet's K=288 layers onto im2col + cuBLAS
(11.3-12.4 -> 10.2-10.3 ms, `CURRENT_STATE §9.90`, `tests/conv_im2col.rs`).
Tensor-core implicit GEMM for 3x3/conv_transpose remains open as a custom
kernel, but the bounded cuBLAS-tensor-core path landed: opt-in conv-only TF32
(`AUTORUST_CONV_TF32=1`) gives ResNet-18 -7-9% and UNet -12%
(`CURRENT_STATE §9.91`, `tests/conv_tf32.rs`). A graph-level conv_transpose
pixel-shuffle specialization was also tried and rejected: UNet CUDA was
neutral (10.69 vs 10.61 ms) and CPU was 2x worse (`CURRENT_STATE §9.92`). The
custom tensor-core 3x3 implicit-GEMM kernel then landed as a bounded TF32
`wmma` kernel (no im2col copy, 64x64 block tile) under the new Fast math mode:
ResNet-18 277 -> 201-208 ms and ViT infer 82.8 -> 65.6 ms vs Exact; see
`CURRENT_STATE §9.93` and `tests/cuda_conv3x3_tc.rs`.
12. Fused reductions: softmax/layernorm/sum/amax (burn reduce tuner)
Why: our CUDA reductions are the remaining memory-bound gap (softmax 155 vs torch 288 GB/s,
layernorm 156 vs 209, sum 176 vs 287, amax 64 vs 292). Burn decomposes softmax and lets fusion
emit one kernel, and its reduce op autotunes unit/plane/cube routines plus output vectorization
(`burn-cubecl/src/kernel/reduce/tune.rs`); we have hand-written fused reductions only inside the
attention kernels.
Try: one-pass row softmax (max+sum+exp in a single block/plane reduction), a vectorized amax
and sum, and a layernorm kernel with the two reductions + normalize in one launch. Start with
`memops` shapes, then reuse in the ViT/LLM paths.
Measure: `bench/run.py --only '.*memops'` GB/s, kernel counts and wall time on vit_train.
Status (2026-11-16): **sum/amax landed** as `Tensor::sum_with`/`max_with`
(one streaming pass + finish kernel, 0.37/0.40 -> 0.25/0.25 ms, 270 GB/s on
CUDA; `CURRENT_STATE §9.83`). **Softmax was a measured rejection** for the
forward-only host-kernel version: the kernel ran 4096x4096 in 0.43 ms vs 0.89
for the graph (2.1x), but the end-to-end memops step improved only ~8%
(readback dominates), LLM prefill was neutral, and ViT training needs an
autograd node (`CURRENT_STATE §9.82`). A fused layernorm host kernel was also
tried and rejected: memops was 0.82-0.87 vs 0.87-0.88 ms (not robust), ViT eval
was neutral and training needs autograd (`CURRENT_STATE §9.85`). A graph-level
softmax/layernorm with autograd remains open.
13. Fold broadcast-rhs batched matmul (LLM decode / per-head broadcast)
Why: burn's `MatmulTransformPolicy` (burn-std/src/tensor/matmul.rs) rewrites
`[.., b, m, k] @ [.., 1, k, n]` into one `[b*m, k] @ [k, n]` when the batch rows do not fill a
tile, instead of launching b matmuls that each re-read the shared rhs. Our decode path launches
~300 kernels/token; per-head/broadcast contractions are a direct candidate.
Try: the same metadata-only fold in `schedule/host_gemm.rs`/the generated matmul path, keyed by
the inferred shape and only when the row count is not already tile-aligned.
Measure: llm_decode CUDA/CPU ms/token and launch count; verify identical logits.
Status (2026-11-16): **measured rejection**. The LLM workloads are `B=1`, so
there is no broadcast-rhs batched matmul to fold; on ViT bs 4 the fold regressed
training 205.4 -> 238.1 ms while bs 8 inference was noise. The host GEMM already
folds broadcast batches where it wins. See `CURRENT_STATE §9.84`.
14. Keyed per-shape autotuner for host-GEMM / conv candidates
Why: burn's `LocalTuner` benchmarks candidate implementations per anchored problem key
(shape+dtype+layout+alignment), keeps the winner, and persists it in the existing cache. Our
kernel-choice knobs (`AUTORUST_MM_*`, `AUTORUST_CPP_GEMM_TILE`, item 1) are global or per-run,
not measured per shape. Our own f32 GEMM shows the spread: 8.0 TF/s at 4096³ but 5.1 at 1024³,
where the right candidate differs.
Try: reuse `opt_search` and the disk cache; tune only the top 2-3 candidates per key after a
minimum-work gate, anchored to 32/128-element buckets so ViT's 197 does not create a new entry
per kernel. Add `AUTORUST_AUTOTUNE_KERNELS=0` escape and a `GEMM_STATS`-style report of the
winner per key.
Measure: first-call cost, cache hit rate, steady-state gain on gemm/vit/resnet.
Status (2026-11-16): the small-batch host-GEMM gate landed as a fixed
heuristic rather than a tuner (MNIST CUDA 52.9 -> 23.3 ms/step, ONNX ResNet-18
12.0 -> 10.2 ms; `CURRENT_STATE §9.81`). The full keyed tile tuner was then
measured and **rejected**: 2048^3 candidates are indistinguishable
(2.18-2.21 ms), the 8-shape sweep is within noise, ViT prefers the current
default, and BEAM=1 on 2048^3 was slower (2.31 ms) after ~30 s cold. The
existing 4x4/8x8 shape rule stays (`CURRENT_STATE §9.88`).
15. Multi-stream scheduling for independent branches
Why: burn's fusion runtime is `Send + Clone` and keeps one lazy stream per thread
(`burn-fusion/src/stream/multi.rs`), with read plans and `device.sync()` as the only hard
barriers. Our scheduler/JIT replay is one ordered stream, which is right for a captured training
step but leaves decode's independent head/FFN branches and q/k/v projections serialized.
Try: after `StepPlan` capture, annotate SINKs with no data dependency and launch each on a
secondary CUDA/CL stream with events joined before the next barrier; keep the single-stream path
as the fallback. CUDA graphs (item 9) and this can share the same dependency annotation.
Measure: llm_decode and vit_infer launch timeline / ms, and a kernel-order/numerics check.
Status (2026-11-16): **measured rejection**. The only independent group is a
layer's three q/k/v GEMMs; running them on three streams saved 5.4-5.6% of that
group, or <0.1% of `llm_decode` (17.2 ms) and ~0.8% of `vit_infer` bs 8
(70.7 ms). The rest of the step is dependency-chained. The executor change was
not justified (`CURRENT_STATE §9.89`).
