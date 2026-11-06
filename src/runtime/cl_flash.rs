//! Fused flash-attention kernel for OpenCL (`D == 64`, non-causal, no mask).
//!
//! Same FA2 schedule as `cuda_flash`: one work-group per `(batch*head, 64-query
//! tile)`, 32-key streaming steps with online softmax, Q staged in (padded)
//! local memory, K/V streamed through local, accumulators in private memory.
//!
//! The sequence length is baked into the source (instead of a scalar arg) so the
//! launch is four buffers on every backend; the compiled binary is cached by
//! source text, so one kernel per shape is all that is ever built.

use crate::runtime::buffer::OpaqueView;
use crate::runtime::device::{Compiled, device};

/// Head dimension the kernel is specialized for.
pub const HEAD_DIM: i64 = 64;
/// Queries per work-group.
pub const BQ: i64 = 64;
/// Keys per streaming step.
pub const BK: i64 = 32;
/// Entry point name.
pub const NAME: &str = "ar_flash_f32";

/// Whether the kernel can run this shape.
pub fn supports(n: i64, d: i64) -> bool {
    d == HEAD_DIM && n > 0
}

/// The OpenCL source for a sequence length `n`.
pub fn src(n: i64) -> String {
    assert!(n > 0);
    format!(
        r#"
#define AR_D 64
#define AR_BQ 64
#define AR_BK 32
#define AR_N {n}

__kernel void {NAME}(
    __global const float* restrict Q, __global const float* restrict K,
    __global const float* restrict V, __global float* restrict O)
{{
    __local float qs[AR_BQ][AR_D + 1];
    __local float ks[AR_BK][AR_D];
    __local float vs[AR_BK][AR_D];

    const int t = get_local_id(0);          /* query row within the tile */
    const int row = get_group_id(0) * AR_BQ + t;
    const int bh = get_group_id(1);
    const long base = (long)bh * (long)AR_N * AR_D;
    const int lsz = get_local_size(0);

    for (int i = t; i < AR_BQ * AR_D; i += lsz) {{
        int r = i / AR_D, d = i % AR_D;
        int gr = get_group_id(0) * AR_BQ + r;
        qs[r][d] = (gr < AR_N) ? Q[base + (long)gr * AR_D + d] : 0.0f;
    }}
    barrier(CLK_LOCAL_MEM_FENCE);

    float acc[AR_D];
    for (int d = 0; d < AR_D; ++d) acc[d] = 0.0f;
    float m = -1e30f, l = 0.0f;
    const float scale = 0.125f;

    for (int k0 = 0; k0 < AR_N; k0 += AR_BK) {{
        for (int i = t; i < AR_BK * AR_D; i += lsz) {{
            int r = i / AR_D, d = i % AR_D;
            int gr = k0 + r;
            ks[r][d] = (gr < AR_N) ? K[base + (long)gr * AR_D + d] : 0.0f;
            vs[r][d] = (gr < AR_N) ? V[base + (long)gr * AR_D + d] : 0.0f;
        }}
        barrier(CLK_LOCAL_MEM_FENCE);

        float s[AR_BK];
        float mnew = m;
        for (int j = 0; j < AR_BK; ++j) {{
            int gj = k0 + j;
            float dot = -1e30f;
            if (gj < AR_N) {{
                dot = 0.0f;
                for (int d = 0; d < AR_D; ++d) dot += qs[t][d] * ks[j][d];
                dot *= scale;
            }}
            s[j] = dot;
            mnew = fmax(mnew, dot);
        }}
        const float alpha = exp(m - mnew);
        float sum = 0.0f;
        for (int j = 0; j < AR_BK; ++j) {{
            float p = exp(s[j] - mnew);
            s[j] = p;
            sum += p;
        }}
        l = l * alpha + sum;
        for (int d = 0; d < AR_D; ++d) {{
            float a = acc[d] * alpha;
            for (int j = 0; j < AR_BK; ++j) a += s[j] * vs[j][d];
            acc[d] = a;
        }}
        m = mnew;
        barrier(CLK_LOCAL_MEM_FENCE);
    }}

    if (row < AR_N) {{
        const float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
        for (int d = 0; d < AR_D; ++d) O[base + (long)row * AR_D + d] = acc[d] * inv;
    }}
}}
"#
    )
}

/// Launch the fused kernel: `q/k/v/o` are `[b, h, n, 64]` contiguous f32.
#[allow(clippy::too_many_arguments)]
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
        &src(n),
        NAME,
        &[
            (0, crate::dtype::FLOAT32),
            (1, crate::dtype::FLOAT32),
            (2, crate::dtype::FLOAT32),
            (3, crate::dtype::FLOAT32),
        ],
        &[q.clone(), k.clone(), v.clone(), o.clone()],
        grid,
        Some([BQ as usize, 1, 1]),
        &[],
        true,
    );
    let _ = device("CL");
}
