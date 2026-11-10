//! Quantized (SageAttention-style) flash-attention kernel for CUDA.
//!
//! QKᵀ runs on 8-bit tensor cores after per-row Q and per-key K quantization;
//! softmax stays in f32 and `P·V` runs on the bf16 tensor cores. Two variants:
//!
//! | arch | QKᵀ instruction | quantized values |
//! |---|---|---|
//! | `sm_80..88` | `mma.m16n8k32.s32.s8.s8.s32` | symmetric int8, scale `max/127` |
//! | `sm_89+` | `mma.m16n8k32.f32.e4m3.e4m3.f32` | E4M3, scale `max/448` |
//!
//! Both use the same `m16n8k32` fragment layout (`tests/sage_layout_probe.rs`
//! pins it on the INT8 instruction; the FP8 instruction has the same layout).
//! The scales are per query row and per key row, which is finer than
//! SageAttention's per-block scaling; SmoothK is not implemented yet.
//!
//! `P·V` keeps bf16/f32 precision, so only the QKᵀ logits are approximate.
//! The result is opt-in (`AttentionKind::Sage` / `sage_attention`); the exact
//! paths are unchanged.
//!
//! # Availability
//!
//! `mma.s8` needs `sm_80+`; `mma.e4m3` needs `sm_89+`. [`supported`] checks the
//! device generation and [`Settings::tensor_cores`]. The source is only
//! compiled on a device that can run it, so an older card keeps the exact
//! kernel (with a one-time notice from the dispatcher).
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
pub const NAME: &str = "ar_flash_sage";

/// Per-call options.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Apply the causal (lower-triangular) mask.
    pub causal: bool,
    /// `mask` is a real additive `[n, n]` f32 tensor.
    pub has_mask: bool,
    /// Number of K/V heads (`h_q / h_kv` is the GQA group size).
    pub h_kv: i64,
    /// Use the FP8 E4M3 `mma` (sm_89+); otherwise int8 `mma` (sm_80+).
    pub fp8: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            causal: false,
            has_mask: false,
            h_kv: 1,
            fp8: false,
        }
    }
}

/// Whether the device can run the quantized kernel at all.
pub fn supported(n: i64, d: i64, s: &Settings) -> bool {
    if d != HEAD_DIM || n <= 0 || !s.tensor_cores {
        return false;
    }
    crate::runtime::ops_cuda::arch_sm().is_some_and(|sm| sm >= 80)
}

/// Whether the open device has FP8 tensor cores (`sm_89+`).
pub fn fp8_available() -> bool {
    crate::runtime::ops_cuda::arch_sm().is_some_and(|sm| sm >= 89)
}

/// The CUDA source for a variant.
pub fn src(fp8: bool) -> String {
    format!("#define AR_FP8 {}\n{BODY}", fp8 as u8)
}

const BODY: &str = r#"
#define AR_D 64
#define AR_BQ 64
#define AR_BK 32
#define AR_PAD 8
#define AR_NEG -1e30f
#if AR_FP8
  #include <cuda_fp8.h>
  #define AR_QMAX 448.0f
#else
  #define AR_QMAX 127.0f
#endif

__device__ __forceinline__ unsigned short ar_bf16(float x) {
    unsigned short r;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(x));
    return r;
}
__device__ __forceinline__ unsigned ar_pack_bf16(float lo, float hi) {
    return (unsigned)ar_bf16(lo) | ((unsigned)ar_bf16(hi) << 16);
}
__device__ __forceinline__ void ar_st2(unsigned short* p, int i, unsigned v) {
    p[i] = (unsigned short)(v & 0xffffu);
    p[i + 1] = (unsigned short)(v >> 16);
}
__device__ __forceinline__ unsigned ar_smem(const void* p) {
    unsigned r;
    asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }"
        : "=r"(r) : "l"(p));
    return r;
}
__device__ __forceinline__ void ar_ldm2t(unsigned& r0, unsigned& r1, const void* addr) {
    unsigned a = ar_smem(addr);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1) : "r"(a));
}

/* Quantize four consecutive values into one 32-bit register. */
__device__ __forceinline__ unsigned ar_q4(float a, float b, float c, float d) {
#if AR_FP8
    __nv_fp8x2_storage_t lo = __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
    __nv_fp8x2_storage_t hi = __nv_cvt_float2_to_fp8x2(make_float2(c, d), __NV_SATFINITE, __NV_E4M3);
    return (unsigned)lo | ((unsigned)hi << 16);
#else
    int i0 = __float2int_rn(a); i0 = i0 < -127 ? -127 : (i0 > 127 ? 127 : i0);
    int i1 = __float2int_rn(b); i1 = i1 < -127 ? -127 : (i1 > 127 ? 127 : i1);
    int i2 = __float2int_rn(c); i2 = i2 < -127 ? -127 : (i2 > 127 ? 127 : i2);
    int i3 = __float2int_rn(d); i3 = i3 < -127 ? -127 : (i3 > 127 ? 127 : i3);
    return ((unsigned)i0 & 0xffu) | (((unsigned)i1 & 0xffu) << 8)
         | (((unsigned)i2 & 0xffu) << 16) | (((unsigned)i3 & 0xffu) << 24);
#endif
}

#if AR_FP8
__device__ __forceinline__ void ar_mma_qk(float d[4], const unsigned a[4],
                                          const unsigned b[2]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                   "r"(b[0]), "r"(b[1]));
}
#else
__device__ __forceinline__ void ar_mma_qk(int d[4], const unsigned a[4],
                                          const unsigned b[2]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                   "r"(b[0]), "r"(b[1]));
}
#endif

__device__ __forceinline__ void ar_mma_pv(float d[4], const unsigned a[4],
                                          const unsigned b[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                   "r"(b[0]), "r"(b[1]));
}

extern "C" __global__ void __launch_bounds__(128) ar_flash_sage(
    const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, float* __restrict__ O,
    const float* __restrict__ M,
    int N, int head_dim, int causal, int has_mask, int hq, int hkv)
{
    /* `scratch` holds the f32 Q tile first, then each f32 K tile (32 rows). */
    __shared__ __align__(16) float scratch[AR_BQ][AR_D + 1];
    __shared__ __align__(16) unsigned char q8[AR_BQ][AR_D + 4];
    __shared__ __align__(16) unsigned char k8[AR_BK][AR_D + 4];
    __shared__ __align__(16) unsigned short vs[AR_BK][AR_D + AR_PAD];
    __shared__ float sq[AR_BQ];
    __shared__ float sk[AR_BK];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g    = lane >> 2;
    const int t2   = (lane & 3) * 2;   /* C columns / output d */
    const int t4   = (lane & 3) * 4;   /* 8-bit fragments: 4 consecutive k */

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

    /* ---- stage + quantize Q (per row) ---- */
    for (int i = tid; i < AR_BQ * AR_D; i += 128) {
        const int r = i >> 6;
        const int d = i & 63;
        const int gr = qblk + r;
        scratch[r][d] = (gr < N) ? Q[qbase + (long)gr * AR_D + d] : 0.f;
    }
    __syncthreads();
    for (int r = tid; r < AR_BQ; r += 128) {
        float mx = 0.f;
        for (int d = 0; d < AR_D; ++d) mx = fmaxf(mx, fabsf(scratch[r][d]));
        sq[r] = (mx > 0.f) ? mx / AR_QMAX : 1.f;
    }
    __syncthreads();
    for (int i = tid; i < AR_BQ * 16; i += 128) {
        const int r = i >> 4;
        const int d0 = (i & 15) * 4;
        const float inv = 1.f / sq[r];
        *(unsigned*)&q8[r][d0] = ar_q4(
            scratch[r][d0] * inv, scratch[r][d0 + 1] * inv,
            scratch[r][d0 + 2] * inv, scratch[r][d0 + 3] * inv);
    }
    __syncthreads();

    /* Q fragments: 2 k steps of 32, loaded from the quantized tile. */
    unsigned qa[2][4];
    const int qr0 = warp * 16 + g;
    const int qr1 = qr0 + 8;
    #pragma unroll
    for (int s = 0; s < 2; ++s) {
        qa[s][0] = *(unsigned*)&q8[qr0][s * 32 + t4];
        qa[s][1] = *(unsigned*)&q8[qr1][s * 32 + t4];
        qa[s][2] = *(unsigned*)&q8[qr0][s * 32 + 16 + t4];
        qa[s][3] = *(unsigned*)&q8[qr1][s * 32 + 16 + t4];
    }
    const float sq0 = sq[qr0];
    const float sq1 = sq[qr1];

    /* ---- output accumulator and softmax state ---- */
    float o[8][4];
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        o[nt][0] = 0.f; o[nt][1] = 0.f; o[nt][2] = 0.f; o[nt][3] = 0.f;
    }
    float m0 = AR_NEG, l0 = 0.f;
    float m1 = AR_NEG, l1 = 0.f;

    for (int k0 = 0; k0 < kend; k0 += AR_BK) {
        /* ---- stage K f32 (reuse scratch), quantize per key row ---- */
        for (int i = tid; i < AR_BK * AR_D; i += 128) {
            const int r = i >> 6;
            const int d = i & 63;
            const int gr = k0 + r;
            scratch[r][d] = (gr < N) ? K[kvbase + (long)gr * AR_D + d] : 0.f;
        }
        /* V straight to bf16. */
        for (int i = tid; i < AR_BK * AR_D; i += 128) {
            const int r = i >> 6;
            const int d = i & 63;
            const int gr = k0 + r;
            vs[r][d] = (gr < N) ? ar_bf16(V[kvbase + (long)gr * AR_D + d]) : (unsigned short)0;
        }
        __syncthreads();
        for (int r = tid; r < AR_BK; r += 128) {
            float mx = 0.f;
            for (int d = 0; d < AR_D; ++d) mx = fmaxf(mx, fabsf(scratch[r][d]));
            sk[r] = (mx > 0.f) ? mx / AR_QMAX : 1.f;
        }
        __syncthreads();
        for (int i = tid; i < AR_BK * 16; i += 128) {
            const int r = i >> 4;
            const int d0 = (i & 15) * 4;
            const float inv = 1.f / sk[r];
            *(unsigned*)&k8[r][d0] = ar_q4(
                scratch[r][d0] * inv, scratch[r][d0 + 1] * inv,
                scratch[r][d0 + 2] * inv, scratch[r][d0 + 3] * inv);
        }
        __syncthreads();

        /* ---- S = Q * K^T on 8-bit tensor cores ---- */
#if AR_FP8
        float sc[4][4];
#else
        int sc[4][4];
#endif
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
#if AR_FP8
            sc[t][0] = 0.f; sc[t][1] = 0.f; sc[t][2] = 0.f; sc[t][3] = 0.f;
#else
            sc[t][0] = 0; sc[t][1] = 0; sc[t][2] = 0; sc[t][3] = 0;
#endif
        }
        #pragma unroll
        for (int s = 0; s < 2; ++s) {
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const unsigned b[2] = {
                    *(unsigned*)&k8[t * 8 + g][s * 32 + t4],
                    *(unsigned*)&k8[t * 8 + g][s * 32 + 16 + t4],
                };
                ar_mma_qk(sc[t], qa[s], b);
            }
        }

        /* ---- dequantize (q_scale * k_scale / sqrt(64)) + masks ---- */
        const int row0 = q0 + g;
        const int row1 = row0 + 8;
        float s[4][4];
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int key = k0 + t * 8 + t2;
            const float sk0 = sk[key & 31];
            const float sk1 = sk[(key + 1) & 31];
            const float a0 = sq0 * sk0 * 0.125f;
            const float a1 = sq0 * sk1 * 0.125f;
            const float b0 = sq1 * sk0 * 0.125f;
            const float b1 = sq1 * sk1 * 0.125f;
            s[t][0] = (float)sc[t][0] * a0;
            s[t][1] = (float)sc[t][1] * a1;
            s[t][2] = (float)sc[t][2] * b0;
            s[t][3] = (float)sc[t][3] * b1;
            if (has_mask) {
                if (key < N) {
                    if (row0 < N) s[t][0] += M[(long)row0 * N + key];
                    if (row1 < N) s[t][2] += M[(long)row1 * N + key];
                }
                if (key + 1 < N) {
                    if (row0 < N) s[t][1] += M[(long)row0 * N + key + 1];
                    if (row1 < N) s[t][3] += M[(long)row1 * N + key + 1];
                }
            }
            if (key >= N || (causal && key > row0))     { s[t][0] = AR_NEG; }
            if (key + 1 >= N || (causal && key + 1 > row0)) { s[t][1] = AR_NEG; }
            if (key >= N || (causal && key > row1))     { s[t][2] = AR_NEG; }
            if (key + 1 >= N || (causal && key + 1 > row1)) { s[t][3] = AR_NEG; }
        }

        /* ---- online softmax ---- */
        float mx0 = m0, mx1 = m1;
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            mx0 = fmaxf(mx0, fmaxf(s[t][0], s[t][1]));
            mx1 = fmaxf(mx1, fmaxf(s[t][2], s[t][3]));
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
            s[t][0] = expf(s[t][0] - mx0); sum0 += s[t][0];
            s[t][1] = expf(s[t][1] - mx0); sum0 += s[t][1];
            s[t][2] = expf(s[t][2] - mx1); sum1 += s[t][2];
            s[t][3] = expf(s[t][3] - mx1); sum1 += s[t][3];
        }
        sum0 += __shfl_xor_sync(0xffffffffu, sum0, 1);
        sum0 += __shfl_xor_sync(0xffffffffu, sum0, 2);
        sum1 += __shfl_xor_sync(0xffffffffu, sum1, 1);
        sum1 += __shfl_xor_sync(0xffffffffu, sum1, 2);
        l0 = l0 * alpha0 + sum0;
        l1 = l1 * alpha1 + sum1;
        m0 = mx0;
        m1 = mx1;

        /* ---- O = O * alpha + P * V (bf16 tensor cores) ---- */
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            o[nt][0] *= alpha0; o[nt][1] *= alpha0;
            o[nt][2] *= alpha1; o[nt][3] *= alpha1;
        }
        #pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            const int s0 = kk * 2;
            const int kb = kk * 16;
            const unsigned a[4] = {
                ar_pack_bf16(s[s0][0],     s[s0][1]),
                ar_pack_bf16(s[s0][2],     s[s0][3]),
                ar_pack_bf16(s[s0 + 1][0], s[s0 + 1][1]),
                ar_pack_bf16(s[s0 + 1][2], s[s0 + 1][3]),
            };
            const int l = lane & 15;
            const int vrow = kb + l;
            #pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                unsigned b0, b1;
                ar_ldm2t(b0, b1, &vs[vrow][nt * 8]);
                const unsigned b[2] = { b0, b1 };
                ar_mma_pv(o[nt], a, b);
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
            O[qbase + (long)r0 * AR_D + nt * 8 + t2 + 0] = o[nt][0] * inv0;
            O[qbase + (long)r0 * AR_D + nt * 8 + t2 + 1] = o[nt][1] * inv0;
        }
        if (r1 < N) {
            O[qbase + (long)r1 * AR_D + nt * 8 + t2 + 0] = o[nt][2] * inv1;
            O[qbase + (long)r1 * AR_D + nt * 8 + t2 + 1] = o[nt][3] * inv1;
        }
    }
}
"#;

/// Number of calls dispatched to this kernel since process start.
static SELECTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Record a dispatch; one-time line under `AUTORUST_ATTN_DEBUG=1`.
pub fn note_selected() {
    SELECTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if crate::helpers::env::get_os("AUTORUST_ATTN_DEBUG").is_some() {
            let sm = crate::runtime::ops_cuda::arch_sm().unwrap_or(0);
            let kind = if fp8_available() { "FP8 E4M3" } else { "INT8" };
            eprintln!("[attention] sage: quantized QK^T {kind} (sm_{sm}), bf16 P.V");
        }
    });
}

/// Calls dispatched to the quantized kernel since process start.
pub fn selections() -> u64 {
    SELECTED.load(std::sync::atomic::Ordering::Relaxed)
}

/// One-time notice when `sage` was requested but the device cannot run it.
pub fn note_unsupported() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let sm = crate::runtime::ops_cuda::arch_sm().unwrap_or(0);
        eprintln!(
            "[attention] sage: quantized QK^T needs sm_80+ and tensor cores \
             (device is sm_{sm}): using the exact kernel"
        );
    });
}

/// Launch the quantized kernel.
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
    assert!(n > 0, "sage: unsupported shape");
    let dummy;
    let m = match mask {
        Some(m) => m,
        None => {
            dummy = o.clone();
            &dummy
        }
    };
    let source = src(opts.fp8);
    let grid = [((n + BQ - 1) / BQ) as usize, (b * h) as usize, 1];
    dev.run_host_kernel(
        &source,
        NAME,
        &signature(),
        &[q.clone(), k.clone(), v.clone(), o.clone(), m.clone()],
        grid,
        Some([THREADS as usize, 1, 1]),
        &vals(n, h, opts),
        true,
    );
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
    let f = crate::dtype::FLOAT32;
    let i = crate::dtype::INT32;
    vec![
        (0, f),
        (1, f),
        (2, f),
        (3, f),
        (4, f),
        (5, i),
        (6, i),
        (7, i),
        (8, i),
        (9, i),
        (10, i),
    ]
}

/// The captured-launch descriptor for a quantized attention call.
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
        src: src(opts.fp8),
        name: NAME.to_string(),
        signature: signature(),
        bufs,
        global: [((n + BQ - 1) / BQ), b * h, 1],
        local: Some([THREADS, 1, 1]),
        vals: vals(n, h, opts),
    }
}

#[cfg(test)]
mod tests {
    /// The FP8 source can only run on `sm_89+`, but NVRTC can still assemble it
    /// for that target on any machine. This catches syntax/include regressions
    /// before an Ada/Hopper card is available.
    #[test]
    fn fp8_source_compiles_for_sm89() {
        let src = super::src(true);
        let includes = vec![
            "/usr/local/cuda/include".to_string(),
            "/usr/include".to_string(),
            "/opt/cuda/include".to_string(),
        ];
        crate::runtime::nvrtc::compile(&src, "sm_89", &includes)
            .expect("FP8 source must compile for sm_89");
    }

    #[test]
    fn int8_source_compiles() {
        let src = super::src(false);
        let includes = vec![
            "/usr/local/cuda/include".to_string(),
            "/usr/include".to_string(),
            "/opt/cuda/include".to_string(),
        ];
        crate::runtime::nvrtc::compile(&src, "sm_80", &includes)
            .expect("INT8 source must compile for sm_80");
    }
}
