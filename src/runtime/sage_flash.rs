//! SageAttention-style fused attention for CUDA (`D == 64`, non-causal, no mask).
//!
//! Same FA2 schedule as [`cuda_flash`](crate::runtime::cuda_flash), but the QKᵀ
//! product runs in **INT8** via `__dp4a` (4 MACs per instruction) instead of f32
//! FMA, which is where SageAttention gets its speed on GPUs without FP8 tensor
//! cores (sm_86 has none).
//!
//! Quantization: symmetric per-row — each query row and each key vector gets its
//! own scale (`127 / max|x|`), so outliers stay inside the int8 range without a
//! global calibration. Softmax and `P·V` stay in f32, matching SageAttention's
//! choice to keep the value product in higher precision.
//!
//! Deliberately **not** implemented yet: SmoothK (subtracting the per-block K
//! mean). It needs a pass over the whole key sequence before the streaming loop,
//! so it belongs with a two-kernel or precomputed-scale variant.

use crate::engine::session::HostCall;
use crate::runtime::buffer::Buffer;

/// Head dimension the kernel is specialized for.
pub const HEAD_DIM: i64 = 64;
/// Queries per block.
pub const BQ: i64 = 64;
/// Keys per streaming step.
pub const BK: i64 = 32;
/// Entry point name.
pub const NAME: &str = "ar_sage_f32";

/// Whether the kernel can run this shape.
pub fn supports(n: i64, d: i64) -> bool {
    d == HEAD_DIM && n > 0
}

/// The CUDA source.
pub const SRC: &str = r#"
#define AR_D 64
#define AR_BQ 64
#define AR_BK 32
#define AR_I8_OFF 64          /* 127 for round-to-nearest, 64 for fp32-free ties */

__device__ __forceinline__ signed char ar_q(float x, float s) {
    float v = x * s;
    int i = (int)(v >= 0.f ? v + 0.5f : v - 0.5f);
    if (i > 127) i = 127;
    if (i < -127) i = -127;
    return (signed char)i;
}

extern "C" __global__ void ar_sage_f32(
    const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, float* __restrict__ O,
    int N, int head_dim, int unused)
{
    __shared__ float qs[AR_BQ][AR_D + 1];
    __shared__ float ks[AR_BK][AR_D];
    __shared__ float vs[AR_BK][AR_D];
    __shared__ signed char qi[AR_BQ][AR_D];
    __shared__ signed char ki[AR_BK][AR_D];
    __shared__ float qscale[AR_BQ];
    __shared__ float kscale[AR_BK];

    const int t = threadIdx.x;
    const int row = blockIdx.x * AR_BQ + t;
    const int bh = blockIdx.y;
    const size_t base = (size_t)bh * (size_t)N * AR_D;

    for (int i = t; i < AR_BQ * AR_D; i += blockDim.x) {
        int r = i / AR_D, d = i % AR_D;
        int gr = blockIdx.x * AR_BQ + r;
        qs[r][d] = (gr < N) ? Q[base + (size_t)gr * AR_D + d] : 0.f;
    }
    __syncthreads();

    /* Per-query-row symmetric int8 quantization. */
    float qmax = 1e-20f;
    for (int d = 0; d < AR_D; ++d) qmax = fmaxf(qmax, fabsf(qs[t][d]));
    float qs_scale = 127.f / qmax;
    qscale[t] = qs_scale;
    for (int d = 0; d < AR_D; ++d) qi[t][d] = ar_q(qs[t][d], qs_scale);

    float acc[AR_D];
    #pragma unroll
    for (int d = 0; d < AR_D; ++d) acc[d] = 0.f;
    float m = -1e30f, l = 0.f;
    const float scale = 0.125f;

    for (int k0 = 0; k0 < N; k0 += AR_BK) {
        for (int i = t; i < AR_BK * AR_D; i += blockDim.x) {
            int r = i / AR_D, d = i % AR_D;
            int gr = k0 + r;
            ks[r][d] = (gr < N) ? K[base + (size_t)gr * AR_D + d] : 0.f;
            vs[r][d] = (gr < N) ? V[base + (size_t)gr * AR_D + d] : 0.f;
        }
        __syncthreads();

        /* Per-key symmetric int8 quantization (one thread per key). */
        if (t < AR_BK) {
            float kmax = 1e-20f;
            for (int d = 0; d < AR_D; ++d) kmax = fmaxf(kmax, fabsf(ks[t][d]));
            float sc = 127.f / kmax;
            kscale[t] = sc;
            for (int d = 0; d < AR_D; ++d) ki[t][d] = ar_q(ks[t][d], sc);
        }
        __syncthreads();

        float s[AR_BK];
        float mnew = m;
        const int* qv = (const int*)qi[t];
        #pragma unroll
        for (int j = 0; j < AR_BK; ++j) {
            int gj = k0 + j;
            float dot = -1e30f;
            if (gj < N) {
                const int* kv = (const int*)ki[j];
                int isum = 0;
                #pragma unroll
                for (int w = 0; w < AR_D / 4; ++w) isum = __dp4a(qv[w], kv[w], isum);
                dot = (float)isum * qscale[t] * kscale[j] * scale;
            }
            s[j] = dot;
            mnew = fmaxf(mnew, dot);
        }
        const float alpha = expf(m - mnew);
        float sum = 0.f;
        #pragma unroll
        for (int j = 0; j < AR_BK; ++j) {
            float p = expf(s[j] - mnew);
            s[j] = p;
            sum += p;
        }
        l = l * alpha + sum;
        #pragma unroll
        for (int d = 0; d < AR_D; ++d) {
            float a = acc[d] * alpha;
            #pragma unroll
            for (int j = 0; j < AR_BK; ++j) a += s[j] * vs[j][d];
            acc[d] = a;
        }
        m = mnew;
        __syncthreads();
    }

    if (row < N) {
        const float inv = (l > 0.f) ? 1.f / l : 0.f;
        for (int d = 0; d < AR_D; ++d) O[base + (size_t)row * AR_D + d] = acc[d] * inv;
    }
}
"#;

/// The captured-launch descriptor for a SageAttention call.
pub fn host_call(bufs: Vec<Buffer>, b: i64, h: i64, n: i64) -> HostCall {
    HostCall {
        device: "CUDA".into(),
        src: SRC.to_string(),
        name: NAME.to_string(),
        signature: vec![
            (0, crate::dtype::FLOAT32),
            (1, crate::dtype::FLOAT32),
            (2, crate::dtype::FLOAT32),
            (3, crate::dtype::FLOAT32),
            (4, crate::dtype::INT32),
            (5, crate::dtype::INT32),
            (6, crate::dtype::INT32),
        ],
        bufs,
        global: [((n + BQ - 1) / BQ), b * h, 1],
        local: Some([BQ, 1, 1]),
        vals: vec![n, HEAD_DIM, 0],
    }
}
