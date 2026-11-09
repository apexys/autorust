//! Fused flash-attention kernel for CUDA (`D == 64`).
//!
//! The generated path materializes the `N x N` score matrix twice (softmax, then
//! `P·V`), which `AUTORUST_KPROF` shows is 38% + 11% of the CUDA kernel time at
//! 448px. This is a standard FA2 kernel: one block per `(batch*head, query
//! tile)`, online softmax over key tiles, so the score matrix never leaves
//! registers/shared memory.
//!
//! Tiling: `BQ = 64` queries per block, one thread per query row, `BK = 32` keys
//! per step. `Q` is staged in shared (padded to dodge bank conflicts), `K`/`V`
//! tiles are streamed through shared, `acc[D]` and `s[BK]` live in registers.
//! Shared = 64*65*4 + 2*32*64*4 = 32.9 KB, under the 48 KB static limit.
//!
//! Supported: causal, an additive f32 mask of shape `[n, n]`, and GQA (K/V may
//! have fewer heads than Q). This is the fallback for devices without tensor
//! cores; on `sm_80+` the TF32/bf16 kernels are preferred.
//!
//! The kernel is launched through [`Compiled::run_host_kernel`], so it uses the
//! same NVRTC compiler and runtime as the generated kernels.
//!
//! [`Compiled::run_host_kernel`]: crate::runtime::device::Compiled::run_host_kernel

use crate::runtime::buffer::OpaqueView;
use crate::runtime::device::{Compiled, device};

/// Head dimension the kernel is specialized for.
pub const HEAD_DIM: i64 = 64;
/// Queries per block.
pub const BQ: i64 = 64;
/// Keys per streaming step.
pub const BK: i64 = 32;

/// Entry point name.
pub const NAME: &str = "ar_flash_f32";

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

extern "C" __global__ void ar_flash_f32(
    const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, float* __restrict__ O,
    const float* __restrict__ M,
    int N, int head_dim, int causal, int has_mask, int hq, int hkv)
{
    __shared__ float qs[AR_BQ][AR_D + 1];
    __shared__ float ks[AR_BK][AR_D];
    __shared__ float vs[AR_BK][AR_D];

    const int t = threadIdx.x;              /* query row within the tile */
    const int qblk = blockIdx.x * AR_BQ;
    const int row = qblk + t;               /* global query row */
    const int bh = blockIdx.y;
    /* GQA: Q has hq heads, K/V have hkv; repeat each K/V head hq/hkv times. */
    const int b_idx   = bh / hq;
    const int head    = bh - b_idx * hq;
    const int group   = hq / hkv;
    const int kh      = head / group;
    const int kv_bh   = b_idx * hkv + kh;
    const long qbase  = (long)bh * (long)N * AR_D;
    const long kvbase = (long)kv_bh * (long)N * AR_D;
    const int kend    = causal ? min(N, qblk + AR_BQ) : N;

    /* Stage the query tile (zero-padded past N). */
    for (int i = t; i < AR_BQ * AR_D; i += blockDim.x) {
        int r = i / AR_D, d = i % AR_D;
        int gr = qblk + r;
        qs[r][d] = (gr < N) ? Q[qbase + (long)gr * AR_D + d] : 0.f;
    }
    __syncthreads();

    float acc[AR_D];
    #pragma unroll
    for (int d = 0; d < AR_D; ++d) acc[d] = 0.f;
    float m = -1e30f, l = 0.f;
    const float scale = 0.125f; /* 1 / sqrt(64) */

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
            float dot = -1e30f;
            if (gj < N) {
                dot = 0.f;
                #pragma unroll
                for (int d = 0; d < AR_D; ++d) dot += qs[t][d] * ks[j][d];
                dot *= scale;
                if (has_mask && row < N) dot += M[(long)row * N + gj];
                if (causal && gj > row) dot = -1e30f;
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
    }
}
"#;

/// Whether the kernel can run this shape on `dev`.
pub fn supports(n: i64, d: i64) -> bool {
    d == HEAD_DIM && n > 0
}

/// One-time notice when tensor-core attention was wanted but the device
/// predates `sm_80`, so this scalar kernel runs instead. A deliberate opt-out
/// (`AUTORUST_TENSOR_CORES=0`) stays quiet.
pub fn note_scalar_selected(settings: &crate::settings::Settings) {
    if !settings.tensor_cores {
        return;
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if let Some(sm) = crate::runtime::ops_cuda::arch_sm() {
            if sm < 80 {
                eprintln!(
                    "[attention] tensor-core attention needs sm_80+, device is sm_{sm}: \
                     using scalar f32 FA2"
                );
            }
        }
    });
}

/// Launch with the default options (no mask, no causal, one K/V head per Q
/// head).
pub fn run(
    dev: &std::rc::Rc<Compiled>,
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

/// Launch the fused kernel with masks/GQA: `q/k/v/o` are `[b, h, n, 64]`
/// contiguous f32, `mask` is an additive `[n, n]` f32 tensor when
/// `opts.has_mask`.
#[allow(clippy::too_many_arguments)]
pub fn run_ex(
    dev: &std::rc::Rc<Compiled>,
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
    assert!(supports(n, HEAD_DIM), "flash kernel: unsupported shape");
    launch(dev, q, k, v, o, mask, b, h, n, opts, true);
}

#[allow(clippy::too_many_arguments)]
fn launch(
    dev: &std::rc::Rc<Compiled>,
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
        Some([BQ as usize, 1, 1]),
        &vals(n, h, opts),
        wait,
    );
    let _ = device("CUDA");
    None
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
        src: SRC.to_string(),
        name: NAME.to_string(),
        signature: signature(),
        bufs,
        global: [((n + BQ - 1) / BQ), b * h, 1],
        local: Some([BQ, 1, 1]),
        vals: vals(n, h, opts),
    }
}
