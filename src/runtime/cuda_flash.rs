//! Fused flash-attention kernel for CUDA (`D == 64`, non-causal, no mask).
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

/// The CUDA source.
pub const SRC: &str = r#"
#define AR_D 64
#define AR_BQ 64
#define AR_BK 32

extern "C" __global__ void ar_flash_f32(
    const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, float* __restrict__ O,
    int N, int head_dim, int unused)
{
    __shared__ float qs[AR_BQ][AR_D + 1];
    __shared__ float ks[AR_BK][AR_D];
    __shared__ float vs[AR_BK][AR_D];

    const int t = threadIdx.x;              /* query row within the tile */
    const int row = blockIdx.x * AR_BQ + t; /* global query row */
    const int bh = blockIdx.y;
    const size_t base = (size_t)bh * (size_t)N * AR_D;

    /* Stage the query tile (zero-padded past N). */
    for (int i = t; i < AR_BQ * AR_D; i += blockDim.x) {
        int r = i / AR_D, d = i % AR_D;
        int gr = blockIdx.x * AR_BQ + r;
        qs[r][d] = (gr < N) ? Q[base + (size_t)gr * AR_D + d] : 0.f;
    }
    __syncthreads();

    float acc[AR_D];
    #pragma unroll
    for (int d = 0; d < AR_D; ++d) acc[d] = 0.f;
    float m = -1e30f, l = 0.f;
    const float scale = 0.125f; /* 1 / sqrt(64) */

    for (int k0 = 0; k0 < N; k0 += AR_BK) {
        for (int i = t; i < AR_BK * AR_D; i += blockDim.x) {
            int r = i / AR_D, d = i % AR_D;
            int gr = k0 + r;
            ks[r][d] = (gr < N) ? K[base + (size_t)gr * AR_D + d] : 0.f;
            vs[r][d] = (gr < N) ? V[base + (size_t)gr * AR_D + d] : 0.f;
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

/// Launch the fused kernel: `q/k/v/o` are `[b, h, n, 64]` contiguous f32.
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
    assert!(supports(n, HEAD_DIM), "flash kernel: unsupported shape");
    let grid = [((n + BQ - 1) / BQ) as usize, (b * h) as usize, 1];
    dev.run_host_kernel(
        SRC,
        NAME,
        &[
            (0, crate::dtype::FLOAT32),
            (1, crate::dtype::FLOAT32),
            (2, crate::dtype::FLOAT32),
            (3, crate::dtype::FLOAT32),
            (4, crate::dtype::INT32),
            (5, crate::dtype::INT32),
            (6, crate::dtype::INT32),
        ],
        &[q.clone(), k.clone(), v.clone(), o.clone()],
        grid,
        Some([BQ as usize, 1, 1]),
        &[n, HEAD_DIM, 0],
        true,
    );
    let _ = device("CUDA");
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
