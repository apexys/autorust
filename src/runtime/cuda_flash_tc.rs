//! Warp-tiled TF32 tensor-core flash-attention kernel for CUDA.
//!
//! Same math and the same tiling as [`cuda_flash`](crate::runtime::cuda_flash)
//! (`BQ = 64`, `BK = 32`, non-causal, no mask, `d == 64`, f32 in/out), but the
//! two matmuls run on `mma.sync.aligned.m16n8k8` TF32 tensor cores instead of
//! scalar FMAs. The online softmax stays in f32.
//!
//! # Why TF32 and not f32
//!
//! There is no full-precision f32 tensor core: the f32-storage format is TF32
//! (10-bit mantissa). This kernel therefore rounds Q/K/P/V to TF32 in hardware
//! and is selected when [`Settings::tensor_cores`] is on (the default).
//! `AUTORUST_TENSOR_CORES=0` keeps the exact scalar kernel; `allow_tf32` is
//! cuBLAS-only and does not gate this kernel.
//!
//! # Availability
//!
//! TF32 `mma` needs `sm_80+`. [`supported`] checks the device generation and
//! [`Settings::tensor_cores`], so an older card (e.g. GTX 1060 / `sm_61`) never
//! compiles this source and keeps the scalar kernel. Nothing about the shape
//! contract changes: the caller (`models::flash_attention::fused_attention`)
//! falls back to [`cuda_flash`](crate::runtime::cuda_flash) in exactly the same
//! place it already falls back from the fused kernel to SDPA.
//!
//! # Layout
//!
//! One block = 4 warps, one warp = 16 queries. `Q` is loaded once into
//! `mma.m16n8k8` A fragments (32 registers/thread), `K`/`V` stream through
//! shared memory, and the 16x32 score tile stays in registers as C fragments.
//! For `m16n8k8` the C layout is the A layout with registers 1 and 2 swapped
//! (`tests/tc_mma_probe.rs` pins this down), so the probabilities feed the
//! `P·V` mma without a shared-memory round trip; only the 16x64 output
//! accumulator is scaled by the online-softmax correction, in registers.
//!
//! Shared = `K` + `V` only: `2 * 32 * 65 * 4 = 16.6 KB` (padded to 65 floats to
//! dodge bank conflicts), so shared memory is not the occupancy limit and the
//! kernel does not need the >48 KB dynamic-shared opt-in.
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
pub const NAME: &str = "ar_flash_tc_f32";

/// The CUDA source.
pub const SRC: &str = r#"
#define AR_D 64
#define AR_BQ 64
#define AR_BK 32
#define AR_NEG -1e30f

/* Round f32 to the TF32 storage format the mma consumes (round-to-nearest). */
__device__ __forceinline__ unsigned ar_tf32(float x) {
    unsigned r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return r;
}

/* D = A(row-major 16x8) * B(col-major 8x8) + D, TF32 in, f32 accumulate. */
__device__ __forceinline__ void ar_mma(float d[4], const unsigned a[4],
                                       const unsigned b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

extern "C" __global__ void __launch_bounds__(128) ar_flash_tc_f32(
    const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, float* __restrict__ O,
    int N, int head_dim, int unused)
{
    /* K is [key][d] and V is [key][d]; both padded to 65 floats. The mma
     * fragments index them directly (see the load sites below). */
    __shared__ float ks[AR_BK][AR_D + 1];
    __shared__ float vs[AR_BK][AR_D + 1];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g    = lane >> 2;        /* row within the 16-row warp tile */
    const int t2   = (lane & 3) * 2;   /* column pair within an 8-wide tile */

    const int q0  = blockIdx.x * AR_BQ + warp * 16;  /* first query of this warp */
    const int bh  = blockIdx.y;
    const long base = (long)bh * (long)N * AR_D;

    /* ---- Q into A fragments, one per 8-wide k step (d = 64 -> 8 steps) ---- */
    unsigned qa[8][4];
    #pragma unroll
    for (int s = 0; s < 8; ++s) {
        const int d0 = s * 8;
        const int r0 = q0 + g;
        const int r1 = q0 + g + 8;
        const float a0 = (r0 < N) ? Q[base + (long)r0 * AR_D + d0 + t2 + 0] : 0.f;
        const float a1 = (r1 < N) ? Q[base + (long)r1 * AR_D + d0 + t2 + 0] : 0.f;
        const float a2 = (r0 < N) ? Q[base + (long)r0 * AR_D + d0 + t2 + 1] : 0.f;
        const float a3 = (r1 < N) ? Q[base + (long)r1 * AR_D + d0 + t2 + 1] : 0.f;
        /* 1/sqrt(64) folded into Q. It is a power of two, so applying it before
         * the TF32 rounding is exactly the same as scaling the scores after. */
        qa[s][0] = ar_tf32(a0 * 0.125f);
        qa[s][1] = ar_tf32(a1 * 0.125f);
        qa[s][2] = ar_tf32(a2 * 0.125f);
        qa[s][3] = ar_tf32(a3 * 0.125f);
    }

    /* ---- online-softmax state and the 16x64 output accumulator ---- */
    float o[8][4];
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        o[nt][0] = 0.f; o[nt][1] = 0.f; o[nt][2] = 0.f; o[nt][3] = 0.f;
    }
    float m0 = AR_NEG, l0 = 0.f;   /* row g     */
    float m1 = AR_NEG, l1 = 0.f;   /* row g + 8 */

    for (int k0 = 0; k0 < N; k0 += AR_BK) {
        /* ---- stage K/V, converted to TF32 once ---- */
        for (int i = tid; i < AR_BK * AR_D; i += 128) {
            const int r  = i >> 6;
            const int d  = i & 63;
            const int gr = k0 + r;
            if (gr < N) {
                ks[r][d] = __uint_as_float(ar_tf32(K[base + (long)gr * AR_D + d]));
                vs[r][d] = __uint_as_float(ar_tf32(V[base + (long)gr * AR_D + d]));
            } else {
                ks[r][d] = 0.f;
                vs[r][d] = 0.f;
            }
        }
        __syncthreads();

        /* ---- S = Q * K^T : 16 queries x 32 keys, C fragments ---- */
        float sc[4][4];
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            sc[t][0] = 0.f; sc[t][1] = 0.f; sc[t][2] = 0.f; sc[t][3] = 0.f;
        }
        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                /* B[k][j] = K[t*8 + j][s*8 + k], col-major from shared. */
                const unsigned b[2] = {
                    __float_as_uint(ks[t * 8 + g][s * 8 + t2 + 0]),
                    __float_as_uint(ks[t * 8 + g][s * 8 + t2 + 1]),
                };
                ar_mma(sc[t], qa[s], b);
            }
        }

        /* ---- row-wise online softmax over the 32 keys ---- */
        float mx0 = m0, mx1 = m1;
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int key = k0 + t * 8 + t2;
            if (key >= N) {                 /* padded keys contribute nothing */
                sc[t][0] = AR_NEG; sc[t][2] = AR_NEG;
            }
            if (key + 1 >= N) {
                sc[t][1] = AR_NEG; sc[t][3] = AR_NEG;
            }
            mx0 = fmaxf(mx0, fmaxf(sc[t][0], sc[t][1]));
            mx1 = fmaxf(mx1, fmaxf(sc[t][2], sc[t][3]));
        }
        /* The 4 lanes of a group hold the 32 keys of one row: reduce with xor. */
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
        for (int t = 0; t < 4; ++t) {       /* k step: 8 keys */
            /* A-fragment layout is the C layout with registers 1 and 2
             * swapped, so the probabilities feed the mma from registers. */
            const unsigned a[4] = {
                ar_tf32(sc[t][0]), ar_tf32(sc[t][2]),
                ar_tf32(sc[t][1]), ar_tf32(sc[t][3]),
            };
            #pragma unroll
            for (int nt = 0; nt < 8; ++nt) { /* n tiles: 8 of the 64 dims */
                /* B[k][j] = V[t*8 + k][nt*8 + j]. */
                const unsigned b[2] = {
                    __float_as_uint(vs[t * 8 + t2 + 0][nt * 8 + g]),
                    __float_as_uint(vs[t * 8 + t2 + 1][nt * 8 + g]),
                };
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
            O[base + (long)r0 * AR_D + nt * 8 + t2 + 0] = o[nt][0] * inv0;
            O[base + (long)r0 * AR_D + nt * 8 + t2 + 1] = o[nt][1] * inv0;
        }
        if (r1 < N) {
            O[base + (long)r1 * AR_D + nt * 8 + t2 + 0] = o[nt][2] * inv1;
            O[base + (long)r1 * AR_D + nt * 8 + t2 + 1] = o[nt][3] * inv1;
        }
    }
}
"#;

/// Whether the shape is one the kernel implements (device aside).
pub fn shape_supported(n: i64, d: i64) -> bool {
    d == HEAD_DIM && n > 0
}

/// Whether this device can run the kernel: TF32 tensor cores are `sm_80+` and
/// the master switch must be on. `Settings::allow_tf32` is cuBLAS-only and does
/// not gate this kernel; TF32 is the only f32-storage tensor-core format.
pub fn supported(n: i64, d: i64, s: &Settings) -> bool {
    if !shape_supported(n, d) || !s.tensor_cores {
        return false;
    }
    crate::runtime::ops_cuda::has_tensor_cores()
}

/// Number of attention calls dispatched to this kernel since process start.
static SELECTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Record a dispatch. Emits a one-time line under `AUTORUST_ATTN_DEBUG=1` so a
/// benchmark can tell the TF32 kernel from the scalar one.
pub fn note_selected() {
    SELECTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if std::env::var_os("AUTORUST_ATTN_DEBUG").is_some() {
            let sm = crate::runtime::ops_cuda::arch_sm().unwrap_or(0);
            eprintln!("[attention] flash: warp-tiled TF32 tensor cores (sm_{sm})");
        }
    });
}

/// Calls dispatched to the tensor-core kernel since process start.
pub fn selections() -> u64 {
    SELECTED.load(std::sync::atomic::Ordering::Relaxed)
}

/// Launch the fused kernel: `q/k/v/o` are `[b, h, n, 64]` contiguous f32.
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
    assert!(shape_supported(n, HEAD_DIM), "flash tc: unsupported shape");
    launch(dev, q, k, v, o, b, h, n, true);
}

#[allow(clippy::too_many_arguments)]
fn launch(
    dev: &std::rc::Rc<crate::runtime::device::Compiled>,
    q: &OpaqueView,
    k: &OpaqueView,
    v: &OpaqueView,
    o: &OpaqueView,
    b: i64,
    h: i64,
    n: i64,
    wait: bool,
) -> Option<f64> {
    let grid = [((n + BQ - 1) / BQ) as usize, (b * h) as usize, 1];
    dev.run_host_kernel(
        SRC,
        NAME,
        &signature(),
        &[q.clone(), k.clone(), v.clone(), o.clone()],
        grid,
        Some([THREADS as usize, 1, 1]),
        &[n, HEAD_DIM, 0],
        wait,
    )
}

fn signature() -> Vec<(i64, crate::dtype::DTypeId)> {
    vec![
        (0, crate::dtype::FLOAT32),
        (1, crate::dtype::FLOAT32),
        (2, crate::dtype::FLOAT32),
        (3, crate::dtype::FLOAT32),
        (4, crate::dtype::INT32),
        (5, crate::dtype::INT32),
        (6, crate::dtype::INT32),
    ]
}

/// The captured-launch descriptor for a fusable attention call.
pub fn host_call(
    bufs: Vec<crate::runtime::buffer::Buffer>,
    b: i64,
    h: i64,
    n: i64,
) -> crate::engine::session::HostCall {
    crate::engine::session::HostCall {
        device: "CUDA".into(),
        src: SRC.to_string(),
        name: NAME.to_string(),
        signature: signature(),
        bufs,
        global: [((n + BQ - 1) / BQ), b * h, 1],
        local: Some([THREADS, 1, 1]),
        vals: vec![n, HEAD_DIM, 0],
    }
}
