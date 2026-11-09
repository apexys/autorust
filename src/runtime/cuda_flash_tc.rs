//! Warp-tiled TF32 tensor-core flash-attention kernel for CUDA.
//!
//! Same math as [`cuda_flash`](crate::runtime::cuda_flash) (`BQ = 64`, `BK =
//! 32`, `d == 64`, f32 in/out), but the two matmuls run on
//! `mma.sync.aligned.m16n8k8` TF32 tensor cores instead of scalar FMAs. The
//! online softmax stays in f32.
//!
//! Supported: causal, an additive f32 mask of shape `[n, n]`, and GQA (K/V may
//! have fewer heads than Q).
//!
//! # Async K/V pipeline
//!
//! `Options::async_pipeline` double-buffers K/V with `cp.async`, like the bf16
//! kernel. Unlike bf16, the values must still be rounded to TF32; the copy
//! cannot do that, so the conversion happens at the fragment loads. Shared
//! doubles to ~34 KB, which caps this kernel at two blocks/SM, so the win is
//! smaller than for bf16 — measure with `AUTORUST_ASYNC=0` before relying on it.
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
//! compiles this source and keeps the scalar kernel.
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
//! Shared = `K` + `V`: `2 * 32 * 68 * 4 = 17.4 KB` per stage (272-byte rows,
//! so every `cp.async` destination is 16-byte aligned).
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

/// The per-call options the fused kernel understands beyond the tensors.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Apply the causal (lower-triangular) mask.
    pub causal: bool,
    /// `mask` is a real additive `[n, n]` f32 tensor (else it is ignored).
    pub has_mask: bool,
    /// Number of K/V heads (`h_q / h_kv` is the GQA group size).
    pub h_kv: i64,
    /// Double-buffer K/V with `cp.async` (`AUTORUST_ASYNC`).
    pub async_pipeline: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            causal: false,
            has_mask: false,
            h_kv: 1,
            async_pipeline: true,
        }
    }
}

/// The CUDA source, parameterised on the pipeline depth.
pub fn src(async_pipeline: bool) -> String {
    format!("#define AR_ASYNC {}\n{BODY}", async_pipeline as u8)
}

const BODY: &str = r#"
#define AR_D 64
#define AR_BQ 64
#define AR_BK 32
/* Row stride: 68 floats = 272 B, a multiple of 16 for cp.async alignment;
 * bank rotation 68 & 31 = 4. */
#define AR_PAD 4
#define AR_CHUNKS (AR_D * 4 / 16)   /* 16-byte chunks per row */
#define AR_NEG -1e30f

#if AR_ASYNC
  #define AR_STAGES 2
#else
  #define AR_STAGES 1
#endif

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

#if AR_ASYNC
__device__ __forceinline__ unsigned ar_smem(const void* p) {
    unsigned r;
    asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }"
        : "=r"(r) : "l"(p));
    return r;
}
__device__ __forceinline__ void ar_cp16(unsigned dst, const void* src, int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(dst), "l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void ar_commit() {
    asm volatile("cp.async.commit_group;" ::: "memory");
}
__device__ __forceinline__ void ar_wait1() {
    asm volatile("cp.async.wait_group 1;" ::: "memory");
}
__device__ __forceinline__ void ar_wait0() {
    asm volatile("cp.async.wait_group 0;" ::: "memory");
}
__device__ __forceinline__ void ar_copy_tile(
    float (*dk)[AR_D + AR_PAD], float (*dv)[AR_D + AR_PAD],
    const float* K, const float* V, long kvbase, int k0, int N, int tid)
{
    for (int i = tid; i < AR_BK * AR_CHUNKS; i += 128) {
        const int r = i / AR_CHUNKS;
        const int c = (i % AR_CHUNKS) * 4;   /* element offset, 4 floats = 16 B */
        const int gr = k0 + r;
        const int ok = (gr < N);
        const long src = kvbase + (long)(ok ? gr : 0) * AR_D + c;
        const int bytes = ok ? 16 : 0;
        ar_cp16(ar_smem(&dk[r][c]), K + src, bytes);
        ar_cp16(ar_smem(&dv[r][c]), V + src, bytes);
    }
}
#else
__device__ __forceinline__ void ar_load_tile(
    float (*dk)[AR_D + AR_PAD], float (*dv)[AR_D + AR_PAD],
    const float* K, const float* V, long kvbase, int k0, int N, int tid)
{
    for (int i = tid; i < AR_BK * AR_D; i += 128) {
        const int r = i >> 6;
        const int d = i & 63;
        const int gr = k0 + r;
        if (gr < N) {
            dk[r][d] = K[kvbase + (long)gr * AR_D + d];
            dv[r][d] = V[kvbase + (long)gr * AR_D + d];
        } else {
            dk[r][d] = 0.f;
            dv[r][d] = 0.f;
        }
    }
}
#endif

extern "C" __global__ void __launch_bounds__(128) ar_flash_tc_f32(
    const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, float* __restrict__ O,
    const float* __restrict__ M,
    int N, int head_dim, int causal, int has_mask, int hq, int hkv)
{
    __shared__ __align__(16) float ks[AR_STAGES][AR_BK][AR_D + AR_PAD];
    __shared__ __align__(16) float vs[AR_STAGES][AR_BK][AR_D + AR_PAD];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g    = lane >> 2;        /* row within the 16-row warp tile */
    const int t2   = (lane & 3) * 2;   /* column pair within an 8-wide tile */

    const int qblk = blockIdx.x * AR_BQ;
    const int q0   = qblk + warp * 16;
    const int bh   = blockIdx.y;
    /* GQA: Q has hq heads, K/V have hkv; repeat each K/V head hq/hkv times. */
    const int b_idx   = bh / hq;
    const int head    = bh - b_idx * hq;
    const int group   = hq / hkv;
    const int kh      = head / group;
    const int kv_bh   = b_idx * hkv + kh;
    const long qbase  = (long)bh * (long)N * AR_D;
    const long kvbase = (long)kv_bh * (long)N * AR_D;
    const int kend    = causal ? min(N, qblk + AR_BQ) : N;

    /* ---- Q into A fragments, one per 8-wide k step (d = 64 -> 8 steps) ---- */
    unsigned qa[8][4];
    #pragma unroll
    for (int s = 0; s < 8; ++s) {
        const int d0 = s * 8;
        const int r0 = q0 + g;
        const int r1 = q0 + g + 8;
        const float a0 = (r0 < N) ? Q[qbase + (long)r0 * AR_D + d0 + t2 + 0] : 0.f;
        const float a1 = (r1 < N) ? Q[qbase + (long)r1 * AR_D + d0 + t2 + 0] : 0.f;
        const float a2 = (r0 < N) ? Q[qbase + (long)r0 * AR_D + d0 + t2 + 1] : 0.f;
        const float a3 = (r1 < N) ? Q[qbase + (long)r1 * AR_D + d0 + t2 + 1] : 0.f;
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
    float m0 = AR_NEG, l0 = 0.f;
    float m1 = AR_NEG, l1 = 0.f;

#if AR_ASYNC
    if (kend > 0) {
        ar_copy_tile(ks[0], vs[0], K, V, kvbase, 0, N, tid);
        ar_commit();
    }
#endif

    int step = 0;
    for (int k0 = 0; k0 < kend; k0 += AR_BK, ++step) {
#if AR_ASYNC
        const int nk = k0 + AR_BK;
        if (nk < kend) {
            ar_copy_tile(ks[(step + 1) & 1], vs[(step + 1) & 1], K, V, kvbase, nk, N, tid);
            ar_commit();
        }
        if (nk < kend) ar_wait1(); else ar_wait0();
        __syncthreads();
        const int stage = step & 1;
#else
        ar_load_tile(ks[0], vs[0], K, V, kvbase, k0, N, tid);
        __syncthreads();
        const int stage = 0;
#endif
        float (*ksb)[AR_D + AR_PAD] = ks[stage];
        float (*vsb)[AR_D + AR_PAD] = vs[stage];

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
                    ar_tf32(ksb[t * 8 + g][s * 8 + t2 + 0]),
                    ar_tf32(ksb[t * 8 + g][s * 8 + t2 + 1]),
                };
                ar_mma(sc[t], qa[s], b);
            }
        }

        /* ---- causal + additive mask ---- */
        const int row0 = q0 + g;
        const int row1 = row0 + 8;
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int key = k0 + t * 8 + t2;
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
        for (int t = 0; t < 4; ++t) {       /* k step: 8 keys */
            const unsigned a[4] = {
                ar_tf32(sc[t][0]), ar_tf32(sc[t][2]),
                ar_tf32(sc[t][1]), ar_tf32(sc[t][3]),
            };
            #pragma unroll
            for (int nt = 0; nt < 8; ++nt) { /* n tiles: 8 of the 64 dims */
                const unsigned b[2] = {
                    ar_tf32(vsb[t * 8 + t2 + 0][nt * 8 + g]),
                    ar_tf32(vsb[t * 8 + t2 + 1][nt * 8 + g]),
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

/// Launch with the default options (no mask, no causal, one K/V head per Q
/// head, async pipeline on).
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
    assert!(shape_supported(n, HEAD_DIM), "flash tc: unsupported shape");
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
    let source = src(opts.async_pipeline);
    let grid = [((n + BQ - 1) / BQ) as usize, (b * h) as usize, 1];
    dev.run_host_kernel(
        &source,
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

/// The captured-launch descriptor for a fusable attention call.
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
        src: src(opts.async_pipeline),
        name: NAME.to_string(),
        signature: signature(),
        bufs,
        global: [((n + BQ - 1) / BQ), b * h, 1],
        local: Some([THREADS, 1, 1]),
        vals: vals(n, h, opts),
    }
}
