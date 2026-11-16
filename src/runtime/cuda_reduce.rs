//! Fused full reductions for CUDA (`sum` / `amax`).
//!
//! The generated path lowers a full reduction to a split reduce + a second
//! reduce; `AUTORUST_KPROF` on the 16M-element `memops` case shows it runs at
//! 150-180 GB/s, well below the 300+ GB/s a single streaming pass reaches.
//! This module does one grid-stride pass into `BLOCKS` partials and a tiny
//! finish kernel. It is forward-only (a host kernel has no autograd node), so
//! the public entry points are explicit session-aware calls; the graph path
//! remains the reference and the fallback for every other shape.
//!
//! Limits: contiguous f32, a full scalar reduction (`axis = None`,
//! `keepdim = false`), at least [`MIN_ELEMS`] elements. Amax uses an ordered
//! u32 key so the partials can be reduced with an integer max.

/// Partials written by the streaming pass.
pub const BLOCKS: i64 = 256;
/// Threads per block in both passes.
pub const THREADS: i64 = 256;
/// Smallest tensor worth the two host launches.
pub const MIN_ELEMS: i64 = 1 << 20;
/// Sum entry points.
pub const SUM_PARTIALS: &str = "ar_sum_partials";
/// Sum finish entry point.
pub const SUM_FINISH: &str = "ar_sum_finish";
/// Amax entry points.
pub const AMAX_PARTIALS: &str = "ar_amax_partials";
/// Amax finish entry point.
pub const AMAX_FINISH: &str = "ar_amax_finish";

/// CUDA source shared by both reductions.
pub const SRC: &str = r#"
#define AR_RED_WARPS 8

__device__ __forceinline__ float ar_red_warp_sum(float v) {
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ unsigned ar_red_warp_max(unsigned v) {
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = max(v, __shfl_down_sync(0xffffffffu, v, o));
  return v;
}

__device__ __forceinline__ unsigned ar_f2u(float f) {
  unsigned u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ float ar_u2f(unsigned u) {
  return __uint_as_float((u & 0x80000000u) ? (u & 0x7fffffffu) : ~u);
}

extern "C" __global__ void ar_sum_partials(const float* __restrict__ X, float* __restrict__ P, int n) {
  __shared__ float red[AR_RED_WARPS];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  float s = 0.0f;
  for (int i = tid; i < n; i += stride) s += X[i];
  s = ar_red_warp_sum(s);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = (threadIdx.x < AR_RED_WARPS) ? red[threadIdx.x] : 0.0f;
    v = ar_red_warp_sum(v);
    if (threadIdx.x == 0) P[blockIdx.x] = v;
  }
}

extern "C" __global__ void ar_sum_finish(const float* __restrict__ P, float* __restrict__ out, int blocks) {
  __shared__ float red[AR_RED_WARPS];
  int t = threadIdx.x;
  float v = (t < blocks) ? P[t] : 0.0f;
  v = ar_red_warp_sum(v);
  if ((t & 31) == 0) red[t >> 5] = v;
  __syncthreads();
  if (t < 32) {
    float w = (t < AR_RED_WARPS) ? red[t] : 0.0f;
    w = ar_red_warp_sum(w);
    if (t == 0) out[0] = w;
  }
}

extern "C" __global__ void ar_amax_partials(const float* __restrict__ X, unsigned* __restrict__ P, int n) {
  __shared__ unsigned red[AR_RED_WARPS];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  float m = -1e38f;
  for (int i = tid; i < n; i += stride) m = fmaxf(m, X[i]);
  unsigned u = ar_f2u(m);
  u = ar_red_warp_max(u);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = u;
  __syncthreads();
  if (threadIdx.x < 32) {
    unsigned v = (threadIdx.x < AR_RED_WARPS) ? red[threadIdx.x] : 0u;
    v = ar_red_warp_max(v);
    if (threadIdx.x == 0) P[blockIdx.x] = v;
  }
}

extern "C" __global__ void ar_amax_finish(const unsigned* __restrict__ P, float* __restrict__ out, int blocks) {
  __shared__ unsigned red[AR_RED_WARPS];
  int t = threadIdx.x;
  unsigned v = (t < blocks) ? P[t] : 0u;
  v = ar_red_warp_max(v);
  if ((t & 31) == 0) red[t >> 5] = v;
  __syncthreads();
  if (t < 32) {
    unsigned w = (t < AR_RED_WARPS) ? red[t] : 0u;
    w = ar_red_warp_max(w);
    if (t == 0) out[0] = ar_u2f(w);
  }
}
"#;

fn sig(
    f: crate::dtype::DTypeId,
    p: crate::dtype::DTypeId,
    n: crate::dtype::DTypeId,
) -> Vec<(i64, crate::dtype::DTypeId)> {
    vec![(0, f), (1, p), (2, n)]
}

/// First pass for a sum: `X -> partials`.
pub fn sum_partials(
    bufs: Vec<crate::runtime::buffer::Buffer>,
    n: i64,
) -> crate::engine::session::HostCall {
    use crate::dtype::{FLOAT32, INT32};
    crate::engine::session::HostCall {
        device: "CUDA".into(),
        src: SRC.to_string(),
        name: SUM_PARTIALS.to_string(),
        signature: sig(FLOAT32, FLOAT32, INT32),
        bufs,
        global: [BLOCKS, 1, 1],
        local: Some([THREADS, 1, 1]),
        vals: vec![n],
    }
}

/// Finish pass for a sum: `partials -> scalar`.
pub fn sum_finish(
    bufs: Vec<crate::runtime::buffer::Buffer>,
    blocks: i64,
) -> crate::engine::session::HostCall {
    use crate::dtype::{FLOAT32, INT32};
    crate::engine::session::HostCall {
        device: "CUDA".into(),
        src: SRC.to_string(),
        name: SUM_FINISH.to_string(),
        signature: sig(FLOAT32, FLOAT32, INT32),
        bufs,
        global: [1, 1, 1],
        local: Some([THREADS, 1, 1]),
        vals: vec![blocks],
    }
}

/// First pass for an amax: `X -> ordered-u32 partials`.
pub fn amax_partials(
    bufs: Vec<crate::runtime::buffer::Buffer>,
    n: i64,
) -> crate::engine::session::HostCall {
    use crate::dtype::{FLOAT32, INT32};
    crate::engine::session::HostCall {
        device: "CUDA".into(),
        src: SRC.to_string(),
        name: AMAX_PARTIALS.to_string(),
        signature: sig(FLOAT32, INT32, INT32),
        bufs,
        global: [BLOCKS, 1, 1],
        local: Some([THREADS, 1, 1]),
        vals: vec![n],
    }
}

/// Finish pass for an amax: `ordered-u32 partials -> scalar`.
pub fn amax_finish(
    bufs: Vec<crate::runtime::buffer::Buffer>,
    blocks: i64,
) -> crate::engine::session::HostCall {
    use crate::dtype::{FLOAT32, INT32};
    crate::engine::session::HostCall {
        device: "CUDA".into(),
        src: SRC.to_string(),
        name: AMAX_FINISH.to_string(),
        signature: sig(INT32, FLOAT32, INT32),
        bufs,
        global: [1, 1, 1],
        local: Some([THREADS, 1, 1]),
        vals: vec![blocks],
    }
}
