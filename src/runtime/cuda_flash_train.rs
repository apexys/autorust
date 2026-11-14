//! Fused FA2 **training** kernels for CUDA (`D == 64`): forward with saved
//! softmax statistics, then dQ and dK/dV.
//!
//! The inference kernel (`cuda_flash`) returns only `O`. Training needs the
//! log-sum-exp per query row (`LSE[bh, n]`) to recompute `p = exp(s - lse)` in
//! the backward without materializing the `N x N` score matrix. This module
//! adds that variant plus the two backward passes:
//!
//! - `ar_fl_fwd`: the same tiling as [`crate::runtime::cuda_flash`] (online
//!   softmax, `BQ=64` queries per block, `BK=32` keys per step) and also writes
//!   `LSE = m + log(l)`.
//! - `ar_fl_prep`: `D[bh, i] = O[i] . dO[i]`, the softmax-backward row scalar.
//! - `ar_fl_bwd_dq`: one block per query tile; for each key it recomputes
//!   `s = q.k * scale (+ mask)`, `p = exp(s - lse)`, `dp = dO.v`,
//!   `ds = p * (dp - D)` and accumulates `dQ += ds * k` in registers. No atomics.
//! - `ar_fl_bwd_dkv`: one block per key tile and K/V head; loops over the GQA
//!   group and all query tiles, accumulating `dK += ds^T q` and `dV += p^T dO`
//!   in shared memory. No atomics, deterministic order.
//!
//! Causal masking, an additive f32 `[n, n]` mask and GQA are supported, matching
//! the forward kernels. f32 only; other dtypes keep the graph path.

use crate::runtime::buffer::OpaqueView;
use crate::runtime::device::Compiled;

/// Head dimension the kernels are specialized for.
pub const HEAD_DIM: i64 = 64;
/// Queries per forward block.
pub const BQ: i64 = 64;
/// Keys per forward streaming step.
pub const BK: i64 = 32;
/// Query tile in the backward kernels (shared budget: two `[BQ][D+1]` tiles).
pub const BQ_BWD: i64 = 32;
/// Key tile in the backward kernels.
pub const BK_BWD: i64 = 32;

/// Options shared by the forward/backward launches.
#[derive(Clone, Copy, Debug, Default)]
pub struct Options {
    /// Apply the causal (lower-triangular) mask.
    pub causal: bool,
    /// `mask` is a real additive `[n, n]` f32 tensor.
    pub has_mask: bool,
    /// Number of K/V heads (`h_q / h_kv` is the GQA group size).
    pub h_kv: i64,
}

/// CUDA source for the four kernels.
pub const SRC: &str = r#"
#define AR_D 64
#define AR_BQ 64
#define AR_BK 32
#define AR_BQB 32
#define AR_BKB 32
#define AR_NEG (-1e30f)

extern "C" __global__ void ar_fl_fwd(
    const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, float* __restrict__ O,
    float* __restrict__ LSE, const float* __restrict__ M,
    int N, int causal, int has_mask, int hq, int hkv)
{
    __shared__ float qs[AR_BQ][AR_D + 1];
    __shared__ float ks[AR_BK][AR_D];
    __shared__ float vs[AR_BK][AR_D];
    const int t = threadIdx.x;
    const int qblk = blockIdx.x * AR_BQ;
    const int row = qblk + t;
    const int bh = blockIdx.y;
    const int b_idx = bh / hq;
    const int head = bh - b_idx * hq;
    const int group = hq / hkv;
    const int kh = head / group;
    const int kv_bh = b_idx * hkv + kh;
    const long qbase = (long)bh * (long)N * AR_D;
    const long kvbase = (long)kv_bh * (long)N * AR_D;
    const int kend = causal ? min(N, qblk + AR_BQ) : N;

    for (int i = t; i < AR_BQ * AR_D; i += blockDim.x) {
        int r = i / AR_D, d = i % AR_D;
        int gr = qblk + r;
        qs[r][d] = (gr < N) ? Q[qbase + (long)gr * AR_D + d] : 0.f;
    }
    __syncthreads();

    float acc[AR_D];
    #pragma unroll
    for (int d = 0; d < AR_D; ++d) acc[d] = 0.f;
    float m = AR_NEG, l = 0.f;
    const float scale = 0.125f;

    for (int k0 = 0; k0 < kend; k0 += AR_BK) {
        for (int i = t; i < AR_BK * AR_D; i += blockDim.x) {
            int r = i / AR_D, d = i % AR_D;
            int gr = k0 + r;
            ks[r][d] = (gr < N) ? K[kvbase + (long)gr * AR_D + d] : 0.f;
            vs[r][d] = (gr < N) ? V[kvbase + (long)gr * AR_D + d] : 0.f;
        }
        __syncthreads();

        float s[AR_BK];
        float mnew = m;
        #pragma unroll
        for (int j = 0; j < AR_BK; ++j) {
            int gj = k0 + j;
            float dot = AR_NEG;
            if (gj < N) {
                dot = 0.f;
                #pragma unroll
                for (int d = 0; d < AR_D; ++d) dot += qs[t][d] * ks[j][d];
                dot *= scale;
                if (has_mask && row < N) dot += M[(long)row * N + gj];
                if (causal && gj > row) dot = AR_NEG;
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
        for (int d = 0; d < AR_D; ++d) O[qbase + (long)row * AR_D + d] = acc[d] * inv;
        LSE[(long)bh * N + row] = (l > 0.f) ? (m + logf(l)) : AR_NEG;
    }
}

extern "C" __global__ void ar_fl_prep(
    const float* __restrict__ O, const float* __restrict__ dO,
    float* __restrict__ D, int N)
{
    const int row = blockIdx.x;
    const int bh = blockIdx.y;
    const long base = (long)bh * N * AR_D + (long)row * AR_D;
    float d = 0.f;
    #pragma unroll
    for (int i = 0; i < AR_D; ++i) d += O[base + i] * dO[base + i];
    D[(long)bh * N + row] = d;
}

extern "C" __global__ void ar_fl_bwd_dq(
    const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, const float* __restrict__ dO,
    const float* __restrict__ LSE, const float* __restrict__ D,
    const float* __restrict__ M, float* __restrict__ dQ,
    int N, int causal, int has_mask, int hq, int hkv)
{
    __shared__ float qs[AR_BQB][AR_D + 1];
    __shared__ float dos[AR_BQB][AR_D + 1];
    __shared__ float ks[AR_BK][AR_D];
    __shared__ float vs[AR_BK][AR_D];
    const int t = threadIdx.x;
    const int qblk = blockIdx.x * AR_BQB;
    const int row = qblk + t;
    const int bh = blockIdx.y;
    const int b_idx = bh / hq;
    const int head = bh - b_idx * hq;
    const int group = hq / hkv;
    const int kh = head / group;
    const int kv_bh = b_idx * hkv + kh;
    const long qbase = (long)bh * (long)N * AR_D;
    const long kvbase = (long)kv_bh * (long)N * AR_D;
    const bool valid = row < N;
    const int rr = valid ? row : 0;

    for (int i = t; i < AR_BQB * AR_D; i += blockDim.x) {
        int r = i / AR_D, d = i % AR_D;
        int gr = qblk + r;
        qs[r][d] = (gr < N) ? Q[qbase + (long)gr * AR_D + d] : 0.f;
        dos[r][d] = (gr < N) ? dO[qbase + (long)gr * AR_D + d] : 0.f;
    }
    __syncthreads();

    const float lse = LSE[(long)bh * N + rr];
    const float Di = D[(long)bh * N + rr];
    const float scale = 0.125f;
    // Uniform across the block: the `__syncthreads` below must be reached by
    // every thread, even though each row attends a different number of keys.
    const int kend = causal ? min(N, qblk + AR_BQB) : N;
    float dq[AR_D];
    #pragma unroll
    for (int d = 0; d < AR_D; ++d) dq[d] = 0.f;

    for (int k0 = 0; k0 < kend; k0 += AR_BK) {
        for (int i = t; i < AR_BK * AR_D; i += blockDim.x) {
            int r = i / AR_D, d = i % AR_D;
            int gr = k0 + r;
            ks[r][d] = (gr < N) ? K[kvbase + (long)gr * AR_D + d] : 0.f;
            vs[r][d] = (gr < N) ? V[kvbase + (long)gr * AR_D + d] : 0.f;
        }
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < AR_BK; ++j) {
            int gj = k0 + j;
            if (gj >= N) break;
            // The causal mask is per row (`gj > rr`), not per block.
            if (causal && gj > rr) continue;
            float s = 0.f;
            #pragma unroll
            for (int d = 0; d < AR_D; ++d) s += qs[t][d] * ks[j][d];
            s *= scale;
            if (has_mask) s += M[(long)rr * N + gj];
            float p = (lse > AR_NEG * 0.5f) ? expf(s - lse) : 0.f;
            float dp = 0.f;
            #pragma unroll
            for (int d = 0; d < AR_D; ++d) dp += dos[t][d] * vs[j][d];
            // d/dq of `s = q.k * scale`: the contraction carries the scale.
            float ds = p * (dp - Di) * scale;
            #pragma unroll
            for (int d = 0; d < AR_D; ++d) dq[d] += ds * ks[j][d];
        }
        __syncthreads();
    }
    if (valid) {
        for (int d = 0; d < AR_D; ++d) dQ[qbase + (long)row * AR_D + d] = dq[d];
    }
}

extern "C" __global__ void ar_fl_bwd_dkv(
    const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, const float* __restrict__ dO,
    const float* __restrict__ LSE, const float* __restrict__ D,
    const float* __restrict__ M, float* __restrict__ dK, float* __restrict__ dV,
    int N, int causal, int has_mask, int hq, int hkv)
{
    __shared__ float qs[AR_BQB][AR_D + 1];
    __shared__ float dos[AR_BQB][AR_D + 1];
    __shared__ float dks[AR_BKB][AR_D + 1];
    __shared__ float dvs[AR_BKB][AR_D + 1];
    const int t = threadIdx.x;
    const int gj = blockIdx.x * AR_BKB + t;
    const int kv_bh = blockIdx.y;
    const int b_idx = kv_bh / hkv;
    const int kh = kv_bh - b_idx * hkv;
    const int group = hq / hkv;
    const long kvbase = (long)kv_bh * (long)N * AR_D;
    const bool valid = gj < N;
    const int jj = valid ? gj : 0;
    const float scale = 0.125f;

    float kr[AR_D], vr[AR_D];
    #pragma unroll
    for (int d = 0; d < AR_D; ++d) {
        kr[d] = valid ? K[kvbase + (long)jj * AR_D + d] : 0.f;
        vr[d] = valid ? V[kvbase + (long)jj * AR_D + d] : 0.f;
        dks[t][d] = 0.f;
        dvs[t][d] = 0.f;
    }

    for (int g = 0; g < group; ++g) {
        const int bh = b_idx * hq + kh * group + g;
        const long qbase = (long)bh * (long)N * AR_D;
        for (int q0 = 0; q0 < N; q0 += AR_BQB) {
            for (int i = t; i < AR_BQB * AR_D; i += blockDim.x) {
                int r = i / AR_D, d = i % AR_D;
                int gi = q0 + r;
                qs[r][d] = (gi < N) ? Q[qbase + (long)gi * AR_D + d] : 0.f;
                dos[r][d] = (gi < N) ? dO[qbase + (long)gi * AR_D + d] : 0.f;
            }
            __syncthreads();
            for (int i = 0; i < AR_BQB; ++i) {
                int gi = q0 + i;
                if (gi >= N) break;
                if (causal && gi < jj) continue;
                float s = 0.f;
                #pragma unroll
                for (int d = 0; d < AR_D; ++d) s += qs[i][d] * kr[d];
                s *= scale;
                if (has_mask) s += M[(long)gi * N + jj];
                const float lse = LSE[(long)bh * N + gi];
                float p = (lse > AR_NEG * 0.5f) ? expf(s - lse) : 0.f;
                float dp = 0.f;
                #pragma unroll
                for (int d = 0; d < AR_D; ++d) dp += dos[i][d] * vr[d];
                float ds = p * (dp - D[(long)bh * N + gi]) * scale;
                #pragma unroll
                for (int d = 0; d < AR_D; ++d) {
                    dks[t][d] += ds * qs[i][d];
                    dvs[t][d] += p * dos[i][d];
                }
            }
            __syncthreads();
        }
    }
    if (valid) {
        for (int d = 0; d < AR_D; ++d) {
            dK[kvbase + (long)jj * AR_D + d] = dks[t][d];
            dV[kvbase + (long)jj * AR_D + d] = dvs[t][d];
        }
    }
}
"#;

/// Whether the training kernels cover this shape.
pub fn supports(n: i64, d: i64) -> bool {
    d == HEAD_DIM && n > 0
}

fn signature_fwd() -> Vec<(i64, crate::dtype::DTypeId)> {
    let f = crate::dtype::FLOAT32;
    let i = crate::dtype::INT32;
    vec![
        (0, f),
        (1, f),
        (2, f),
        (3, f),
        (4, f),
        (5, f),
        (6, i),
        (7, i),
        (8, i),
        (9, i),
        (10, i),
    ]
}

/// Launch the forward kernel, writing `O` and `LSE` (`[b*h, n]` f32).
#[allow(clippy::too_many_arguments)]
pub fn run_fwd(
    dev: &std::rc::Rc<Compiled>,
    q: &OpaqueView,
    k: &OpaqueView,
    v: &OpaqueView,
    o: &OpaqueView,
    lse: &OpaqueView,
    mask: Option<&OpaqueView>,
    b: i64,
    h: i64,
    n: i64,
    opts: Options,
) {
    let dummy;
    let m = match mask {
        Some(m) => m,
        None => {
            dummy = o.clone();
            &dummy
        }
    };
    dev.run_host_kernel(
        SRC,
        "ar_fl_fwd",
        &signature_fwd(),
        &[
            q.clone(),
            k.clone(),
            v.clone(),
            o.clone(),
            lse.clone(),
            m.clone(),
        ],
        [((n + BQ - 1) / BQ) as usize, (b * h) as usize, 1],
        Some([BQ as usize, 1, 1]),
        &[n, opts.causal as i64, opts.has_mask as i64, h, opts.h_kv],
        false,
    );
}

fn signature_prep() -> Vec<(i64, crate::dtype::DTypeId)> {
    let f = crate::dtype::FLOAT32;
    let i = crate::dtype::INT32;
    vec![(0, f), (1, f), (2, f), (3, i)]
}

/// Launch the `D = O . dO` row-scalar kernel.
pub fn run_prep(
    dev: &std::rc::Rc<Compiled>,
    o: &OpaqueView,
    dout: &OpaqueView,
    d: &OpaqueView,
    b: i64,
    h: i64,
    n: i64,
) {
    dev.run_host_kernel(
        SRC,
        "ar_fl_prep",
        &signature_prep(),
        &[o.clone(), dout.clone(), d.clone()],
        [n as usize, (b * h) as usize, 1],
        Some([1, 1, 1]),
        &[n],
        false,
    );
}

fn signature_dq() -> Vec<(i64, crate::dtype::DTypeId)> {
    let f = crate::dtype::FLOAT32;
    let i = crate::dtype::INT32;
    vec![
        (0, f),
        (1, f),
        (2, f),
        (3, f),
        (4, f),
        (5, f),
        (6, f),
        (7, f),
        (8, i),
        (9, i),
        (10, i),
        (11, i),
        (12, i),
    ]
}

/// Launch the dQ kernel.
#[allow(clippy::too_many_arguments)]
pub fn run_dq(
    dev: &std::rc::Rc<Compiled>,
    q: &OpaqueView,
    k: &OpaqueView,
    v: &OpaqueView,
    dout: &OpaqueView,
    lse: &OpaqueView,
    d: &OpaqueView,
    mask: Option<&OpaqueView>,
    dq: &OpaqueView,
    b: i64,
    h: i64,
    n: i64,
    opts: Options,
) {
    let dummy;
    let m = match mask {
        Some(m) => m,
        None => {
            dummy = dq.clone();
            &dummy
        }
    };
    dev.run_host_kernel(
        SRC,
        "ar_fl_bwd_dq",
        &signature_dq(),
        &[
            q.clone(),
            k.clone(),
            v.clone(),
            dout.clone(),
            lse.clone(),
            d.clone(),
            m.clone(),
            dq.clone(),
        ],
        [((n + BQ_BWD - 1) / BQ_BWD) as usize, (b * h) as usize, 1],
        Some([BQ_BWD as usize, 1, 1]),
        &[n, opts.causal as i64, opts.has_mask as i64, h, opts.h_kv],
        false,
    );
}

fn signature_dkv() -> Vec<(i64, crate::dtype::DTypeId)> {
    let f = crate::dtype::FLOAT32;
    let i = crate::dtype::INT32;
    vec![
        (0, f),
        (1, f),
        (2, f),
        (3, f),
        (4, f),
        (5, f),
        (6, f),
        (7, f),
        (8, f),
        (9, i),
        (10, i),
        (11, i),
        (12, i),
        (13, i),
    ]
}

/// Launch the dK/dV kernel.
#[allow(clippy::too_many_arguments)]
pub fn run_dkv(
    dev: &std::rc::Rc<Compiled>,
    q: &OpaqueView,
    k: &OpaqueView,
    v: &OpaqueView,
    dout: &OpaqueView,
    lse: &OpaqueView,
    d: &OpaqueView,
    mask: Option<&OpaqueView>,
    dk: &OpaqueView,
    dv: &OpaqueView,
    b: i64,
    h: i64,
    n: i64,
    opts: Options,
) {
    let dummy;
    let m = match mask {
        Some(m) => m,
        None => {
            dummy = dk.clone();
            &dummy
        }
    };
    dev.run_host_kernel(
        SRC,
        "ar_fl_bwd_dkv",
        &signature_dkv(),
        &[
            q.clone(),
            k.clone(),
            v.clone(),
            dout.clone(),
            lse.clone(),
            d.clone(),
            m.clone(),
            dk.clone(),
            dv.clone(),
        ],
        [
            ((n + BK_BWD - 1) / BK_BWD) as usize,
            (b * opts.h_kv) as usize,
            1,
        ],
        Some([BK_BWD as usize, 1, 1]),
        &[n, opts.causal as i64, opts.has_mask as i64, h, opts.h_kv],
        false,
    );
}
