//! Warp-tiled bf16 tensor-core flash-attention kernel for CUDA.
//!
//! The bf16 sibling of [`cuda_flash_tc`](crate::runtime::cuda_flash_tc): same
//! schedule (`BQ = 64`, `BK = 32`, `d == 64`), but the two matmuls use
//! `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` — bf16 operands, f32
//! accumulate. Selected when the tensors themselves are bf16, so the dtype
//! contract is unchanged: an f32 model keeps TF32 attention, a bf16 model gets
//! the bf16 kernel.
//!
//! Supported: causal, an additive f32 mask of shape `[n, n]`, and GQA (K/V may
//! have fewer heads than Q).
//!
//! # Why this is the next step
//!
//! bf16 is the model-level way to use tensor cores at full rate: on sm_86 the
//! bf16 `mma` is ~2x the TF32 rate, and the fragments are half the size. The
//! kernel is also where a bf16 model wants its attention to run, rather than
//! casting to f32 and back.
//!
//! # Availability
//!
//! bf16 `mma` needs `sm_80+`, same as TF32. [`supported`] checks
//! [`Settings::tensor_cores`] and the device generation; an older card keeps the
//! general SDPA path (which computes bf16 through f32).
//!
//! # Layout
//!
//! `mma.m16n8k16` fragments (pinned by `tests/tc_mma_bf16_probe.rs`; `g = lane
//! >> 2`, `t = lane & 3`, each register holds two consecutive k values):
//!
//! | A | B | C/D |
//! |---|---|---|
//! | a0 = A[g][2t:2t+2]      | b0 = B[2t:2t+2][g]    | c0 = C[g][2t]     |
//! | a1 = A[g+8][2t:2t+2]    | b1 = B[2t+8:2t+10][g] | c1 = C[g][2t+1]   |
//! | a2 = A[g][2t+8:2t+10]   |                       | c2 = C[g+8][2t]   |
//! | a3 = A[g+8][2t+8:2t+10] |                       | c3 = C[g+8][2t+1] |
//!
//! Two score tiles (16 keys) pack into one A fragment for `P·V` with no
//! cross-lane exchange, so the probabilities again stay in registers.
//!
//! Shared = `K` + `V` as bf16: `2 * 32 * 66 * 2 = 8.4 KB`.
//!
//! [`Settings::tensor_cores`]: crate::settings::Settings::tensor_cores

use crate::runtime::buffer::OpaqueView;
use crate::settings::Settings;

/// Head dimension the kernel is specialized for.
pub const HEAD_DIM: i64 = 64;
/// Queries per block (4 warps x 16).
pub const BQ: i64 = 64;
/// Keys per streaming step.
pub const BK: i64 = 32;
/// Threads per block (4 warps).
pub const THREADS: i64 = 128;

/// Entry point name.
pub const NAME: &str = "ar_flash_bf16";

/// The per-call options the fused kernel understands beyond the tensors.
#[derive(Clone, Copy, Debug, Default)]
pub struct Options {
    /// Apply the causal (lower-triangular) mask.
    pub causal: bool,
    /// `mask` is a real additive `[n, n]` f32 tensor (else it is ignored).
    pub has_mask: bool,
    /// Number of K/V heads (`h_q / h_kv` is the GQA group size).
    pub h_kv: i64,
}

/// The CUDA source.
pub const SRC: &str = r#"
#define AR_D 64
#define AR_BQ 64
#define AR_BK 32
#define AR_NEG -1e30f

__device__ __forceinline__ unsigned short ar_bf16(float x) {
    unsigned short r;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(x));
    return r;
}

/* Two consecutive bf16 values from memory, packed low/high. */
__device__ __forceinline__ unsigned ar_ld2(const unsigned short* p, int i) {
    return (unsigned)p[i] | ((unsigned)p[i + 1] << 16);
}

__device__ __forceinline__ unsigned ar_pack(float lo, float hi) {
    return (unsigned)ar_bf16(lo) | ((unsigned)ar_bf16(hi) << 16);
}

__device__ __forceinline__ void ar_st2(unsigned short* p, int i, unsigned v) {
    p[i] = (unsigned short)(v & 0xffffu);
    p[i + 1] = (unsigned short)(v >> 16);
}

/* D = A(row-major 16x16) * B(col-major 16x8) + D, bf16 in, f32 accumulate. */
__device__ __forceinline__ void ar_mma(float d[4], const unsigned a[4],
                                       const unsigned b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

extern "C" __global__ void __launch_bounds__(128) ar_flash_bf16(
    const unsigned short* __restrict__ Q, const unsigned short* __restrict__ K,
    const unsigned short* __restrict__ V, unsigned short* __restrict__ O,
    const float* __restrict__ M,
    int N, int head_dim, int causal, int has_mask, int hq, int hkv)
{
    /* bf16 [key][d], padded to 66 elements (132 B) so the 4-byte pair loads
     * rotate banks: bank = (row*66*2/4 + word) & 31 = (row*33 + word) & 31. */
    __shared__ unsigned short ks[AR_BK][AR_D + 2];
    __shared__ unsigned short vs[AR_BK][AR_D + 2];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g    = lane >> 2;        /* row within the 16-row warp tile */
    const int t2   = (lane & 3) * 2;   /* column pair within an 8-wide tile */

    const int qblk = blockIdx.x * AR_BQ;
    const int q0   = qblk + warp * 16;
    const int bh   = blockIdx.y;
    const int b_idx   = bh / hq;
    const int head    = bh - b_idx * hq;
    const int group   = hq / hkv;
    const int kh      = head / group;
    const int kv_bh   = b_idx * hkv + kh;
    const long qbase  = (long)bh * (long)N * AR_D;
    const long kvbase = (long)kv_bh * (long)N * AR_D;
    const int kend    = causal ? min(N, qblk + AR_BQ) : N;

    /* ---- Q into A fragments: d = 64 -> 4 k steps of 16 ---- */
    unsigned qa[4][4];
    #pragma unroll
    for (int s = 0; s < 4; ++s) {
        const int d0 = s * 16;
        const int r0 = q0 + g;
        const int r1 = q0 + g + 8;
        const bool ok0 = r0 < N, ok1 = r1 < N;
        qa[s][0] = ok0 ? ar_ld2(Q + qbase + (long)r0 * AR_D, d0 + t2)     : 0u;
        qa[s][1] = ok1 ? ar_ld2(Q + qbase + (long)r1 * AR_D, d0 + t2)     : 0u;
        qa[s][2] = ok0 ? ar_ld2(Q + qbase + (long)r0 * AR_D, d0 + t2 + 8) : 0u;
        qa[s][3] = ok1 ? ar_ld2(Q + qbase + (long)r1 * AR_D, d0 + t2 + 8) : 0u;
    }

    /* ---- online-softmax state and the 16x64 output accumulator ---- */
    float o[8][4];
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        o[nt][0] = 0.f; o[nt][1] = 0.f; o[nt][2] = 0.f; o[nt][3] = 0.f;
    }
    float m0 = AR_NEG, l0 = 0.f;
    float m1 = AR_NEG, l1 = 0.f;

    for (int k0 = 0; k0 < kend; k0 += AR_BK) {
        /* ---- stage K/V (already bf16) ---- */
        for (int i = tid; i < AR_BK * AR_D; i += 128) {
            const int r  = i >> 6;
            const int d  = i & 63;
            const int gr = k0 + r;
            if (gr < N) {
                ks[r][d] = K[kvbase + (long)gr * AR_D + d];
                vs[r][d] = V[kvbase + (long)gr * AR_D + d];
            } else {
                ks[r][d] = 0;
                vs[r][d] = 0;
            }
        }
        __syncthreads();

        /* ---- S = Q * K^T : 16 queries x 32 keys ---- */
        float sc[4][4];
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            sc[t][0] = 0.f; sc[t][1] = 0.f; sc[t][2] = 0.f; sc[t][3] = 0.f;
        }
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int d0 = s * 16;
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const unsigned b0 = ar_ld2(ks[t * 8 + g], d0 + t2);
                const unsigned b1 = ar_ld2(ks[t * 8 + g], d0 + t2 + 8);
                const unsigned b[2] = { b0, b1 };
                ar_mma(sc[t], qa[s], b);
            }
        }

        /* ---- 1/sqrt(64), causal and additive mask ---- */
        const int row0 = q0 + g;
        const int row1 = row0 + 8;
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int key = k0 + t * 8 + t2;
            sc[t][0] *= 0.125f; sc[t][1] *= 0.125f;
            sc[t][2] *= 0.125f; sc[t][3] *= 0.125f;
            if (has_mask) {
                if (key < N) {
                    if (row0 < N) sc[t][0] += M[(long)row0 * N + key];
                    if (row1 < N) sc[t][2] += M[(long)row1 * N + key];
                }
                if (key + 1 < N) {
                    if (row0 < N) sc[t][1] += M[(long)row0 * N + key + 1];
                    if (row1 < N) sc[t][3] += M[(long)row1 * N + key + 1];
                }
            }
            if (key >= N || (causal && key > row0))     { sc[t][0] = AR_NEG; }
            if (key + 1 >= N || (causal && key + 1 > row0)) { sc[t][1] = AR_NEG; }
            if (key >= N || (causal && key > row1))     { sc[t][2] = AR_NEG; }
            if (key + 1 >= N || (causal && key + 1 > row1)) { sc[t][3] = AR_NEG; }
        }

        /* ---- row-wise online softmax over the 32 keys ---- */
        float mx0 = m0, mx1 = m1;
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            mx0 = fmaxf(mx0, fmaxf(sc[t][0], sc[t][1]));
            mx1 = fmaxf(mx1, fmaxf(sc[t][2], sc[t][3]));
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));

        const float alpha0 = expf(m0 - mx0);
        const float alpha1 = expf(m1 - mx1);
        float sum0 = 0.f, sum1 = 0.f;
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            sc[t][0] = expf(sc[t][0] - mx0); sum0 += sc[t][0];
            sc[t][1] = expf(sc[t][1] - mx0); sum0 += sc[t][1];
            sc[t][2] = expf(sc[t][2] - mx1); sum1 += sc[t][2];
            sc[t][3] = expf(sc[t][3] - mx1); sum1 += sc[t][3];
        }
        sum0 += __shfl_xor_sync(0xffffffffu, sum0, 1);
        sum0 += __shfl_xor_sync(0xffffffffu, sum0, 2);
        sum1 += __shfl_xor_sync(0xffffffffu, sum1, 1);
        sum1 += __shfl_xor_sync(0xffffffffu, sum1, 2);
        l0 = l0 * alpha0 + sum0;
        l1 = l1 * alpha1 + sum1;
        m0 = mx0;
        m1 = mx1;

        /* ---- O = O * alpha + P * V ---- */
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            o[nt][0] *= alpha0; o[nt][1] *= alpha0;
            o[nt][2] *= alpha1; o[nt][3] *= alpha1;
        }
        #pragma unroll
        for (int kk = 0; kk < 2; ++kk) {    /* k step: 16 keys = 2 score tiles */
            const int s0 = kk * 2;
            const int kb = kk * 16;
            /* A layout packs consecutive keys, so one fragment is two score
             * tiles with no register shuffle. */
            const unsigned a[4] = {
                ar_pack(sc[s0][0],     sc[s0][1]),
                ar_pack(sc[s0][2],     sc[s0][3]),
                ar_pack(sc[s0 + 1][0], sc[s0 + 1][1]),
                ar_pack(sc[s0 + 1][2], sc[s0 + 1][3]),
            };
            #pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                const unsigned b0 = (unsigned)vs[kb + t2 + 0][nt * 8 + g]
                                  | ((unsigned)vs[kb + t2 + 1][nt * 8 + g] << 16);
                const unsigned b1 = (unsigned)vs[kb + t2 + 8][nt * 8 + g]
                                  | ((unsigned)vs[kb + t2 + 9][nt * 8 + g] << 16);
                const unsigned b[2] = { b0, b1 };
                ar_mma(o[nt], a, b);
            }
        }
        __syncthreads();
    }

    /* ---- normalize and store ---- */
    const int r0 = q0 + g;
    const int r1 = q0 + g + 8;
    const float inv0 = (l0 > 0.f) ? 1.f / l0 : 0.f;
    const float inv1 = (l1 > 0.f) ? 1.f / l1 : 0.f;
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        if (r0 < N) {
            ar_st2(O + qbase + (long)r0 * AR_D, nt * 8 + t2,
                   ar_pack(o[nt][0] * inv0, o[nt][1] * inv0));
        }
        if (r1 < N) {
            ar_st2(O + qbase + (long)r1 * AR_D, nt * 8 + t2,
                   ar_pack(o[nt][2] * inv1, o[nt][3] * inv1));
        }
    }
}
"#;

/// Whether the shape is one the kernel implements (device aside).
pub fn shape_supported(n: i64, d: i64) -> bool {
    d == HEAD_DIM && n > 0
}

/// Whether this device can run the kernel.
pub fn supported(n: i64, d: i64, s: &Settings) -> bool {
    if !shape_supported(n, d) || !s.tensor_cores {
        return false;
    }
    crate::runtime::ops_cuda::has_tensor_cores()
}

/// Number of attention calls dispatched to this kernel since process start.
static SELECTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Record a dispatch; one-time line under `AUTORUST_ATTN_DEBUG=1`.
pub fn note_selected() {
    SELECTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if std::env::var_os("AUTORUST_ATTN_DEBUG").is_some() {
            let sm = crate::runtime::ops_cuda::arch_sm().unwrap_or(0);
            eprintln!("[attention] flash: warp-tiled bf16 tensor cores (sm_{sm})");
        }
    });
}

/// Calls dispatched to the bf16 kernel since process start.
pub fn selections() -> u64 {
    SELECTED.load(std::sync::atomic::Ordering::Relaxed)
}

/// Launch with the default options (no mask, no causal, one K/V head per Q
/// head).
pub fn run(
    dev: &std::rc::Rc<crate::runtime::device::Compiled>,
    q: &OpaqueView,
    k: &OpaqueView,
    v: &OpaqueView,
    o: &OpaqueView,
    b: i64,
    h: i64,
    n: i64,
) {
    run_ex(
        dev,
        q,
        k,
        v,
        o,
        None,
        b,
        h,
        n,
        Options {
            h_kv: h,
            ..Options::default()
        },
    );
}

/// Launch the fused kernel with masks/GQA.
#[allow(clippy::too_many_arguments)]
pub fn run_ex(
    dev: &std::rc::Rc<crate::runtime::device::Compiled>,
    q: &OpaqueView,
    k: &OpaqueView,
    v: &OpaqueView,
    o: &OpaqueView,
    mask: Option<&OpaqueView>,
    b: i64,
    h: i64,
    n: i64,
    opts: Options,
) {
    assert!(
        shape_supported(n, HEAD_DIM),
        "flash bf16: unsupported shape"
    );
    launch(dev, q, k, v, o, mask, b, h, n, opts, true);
}

#[allow(clippy::too_many_arguments)]
fn launch(
    dev: &std::rc::Rc<crate::runtime::device::Compiled>,
    q: &OpaqueView,
    k: &OpaqueView,
    v: &OpaqueView,
    o: &OpaqueView,
    mask: Option<&OpaqueView>,
    b: i64,
    h: i64,
    n: i64,
    opts: Options,
    wait: bool,
) -> Option<f64> {
    let dummy;
    let m = match mask {
        Some(m) => m,
        None => {
            dummy = o.clone();
            &dummy
        }
    };
    let grid = [((n + BQ - 1) / BQ) as usize, (b * h) as usize, 1];
    dev.run_host_kernel(
        SRC,
        NAME,
        &signature(),
        &[q.clone(), k.clone(), v.clone(), o.clone(), m.clone()],
        grid,
        Some([THREADS as usize, 1, 1]),
        &vals(n, h, opts),
        wait,
    )
}

fn vals(n: i64, h: i64, opts: Options) -> Vec<i64> {
    vec![
        n,
        HEAD_DIM,
        opts.causal as i64,
        opts.has_mask as i64,
        h,
        opts.h_kv,
    ]
}

fn signature() -> Vec<(i64, crate::dtype::DTypeId)> {
    let b = crate::dtype::BFLOAT16;
    let f = crate::dtype::FLOAT32;
    let i = crate::dtype::INT32;
    vec![
        (0, b),
        (1, b),
        (2, b),
        (3, b),
        (4, f),
        (5, i),
        (6, i),
        (7, i),
        (8, i),
        (9, i),
        (10, i),
    ]
}

/// The captured-launch descriptor for a fusable bf16 attention call.
#[allow(clippy::too_many_arguments)]
pub fn host_call(
    bufs: Vec<crate::runtime::buffer::Buffer>,
    b: i64,
    h: i64,
    n: i64,
    opts: Options,
) -> crate::engine::session::HostCall {
    crate::engine::session::HostCall {
        device: "CUDA".into(),
        src: SRC.to_string(),
        name: NAME.to_string(),
        signature: signature(),
        bufs,
        global: [((n + BQ - 1) / BQ), b * h, 1],
        local: Some([THREADS, 1, 1]),
        vals: vals(n, h, opts),
    }
}
