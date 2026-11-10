//! Fused flash-attention kernel for Metal (`D == 64`).
//!
//! Same FA2 schedule as `cl_flash`/`cuda_flash`: one threadgroup per
//! `(batch*head, 64-query tile)`, 16-key streaming steps with online softmax,
//! Q staged in threadgroup memory, K/V streamed through separate threadgroup
//! tiles, accumulators in private memory.
//!
//! Supported: causal, an additive f32 `[n, n]` mask, and GQA (K/V may have
//! fewer heads than Q). The sequence length is baked into the source (so the
//! program is cached per shape); the per-call options are scalar arguments.
//!
//! [`Compiled::run_host_kernel`]: crate::runtime::device::Compiled::run_host_kernel

use crate::runtime::buffer::OpaqueView;
use crate::runtime::device::Compiled;

/// Head dimension the kernel is specialized for.
pub const HEAD_DIM: i64 = 64;
/// Queries per threadgroup.
pub const BQ: i64 = 64;
/// Keys per streaming step.
pub const BK: i64 = 16;
/// Entry point name.
pub const NAME: &str = "ar_flash_f32";

/// Per-call options beyond the tensors.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Apply the causal (lower-triangular) mask.
    pub causal: bool,
    /// `mask` is a real additive `[n, n]` f32 tensor.
    pub has_mask: bool,
    /// Number of K/V heads (`h_q / h_kv` is the GQA group size).
    pub h_kv: i64,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            causal: false,
            has_mask: false,
            h_kv: 1,
        }
    }
}

/// Whether the kernel can run this shape.
pub fn supports(n: i64, d: i64) -> bool {
    d == HEAD_DIM && n > 0
}

/// The Metal source for a sequence length `n`.
pub fn src(n: i64) -> String {
    assert!(n > 0);
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

#define AR_D 64
#define AR_BQ 64
#define AR_BK 16
#define AR_N {n}

kernel void {NAME}(
    device const float* Q, device const float* K,
    device const float* V, device float* O,
    device const float* M,
    constant int& causal, constant int& has_mask, constant int& hq, constant int& hkv,
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]])
{{
    threadgroup float qs[AR_BQ][AR_D];
    threadgroup float ks[AR_BK][AR_D];
    threadgroup float vs[AR_BK][AR_D];

    const int t = int(lid.x);               /* query row within the tile */
    const int qblk = int(gid.x) * AR_BQ;
    const int row = qblk + t;               /* global query row */
    const int bh = int(gid.y);
    /* GQA: Q has hq heads, K/V have hkv; repeat each K/V head hq/hkv times. */
    const int b_idx = bh / hq;
    const int head = bh % hq;
    const int group = hq / hkv;
    const int kh = head / group;
    const int kv_bh = b_idx * hkv + kh;
    const long qbase = (long)bh * (long)AR_N * AR_D;
    const long kvbase = (long)kv_bh * (long)AR_N * AR_D;
    const int kend = causal ? min(AR_N, qblk + AR_BQ) : AR_N;

    for (int i = t; i < AR_BQ * AR_D; i += AR_BQ) {{
        int r = i / AR_D, d = i % AR_D;
        int gr = qblk + r;
        qs[r][d] = (gr < AR_N) ? Q[qbase + (long)gr * AR_D + d] : 0.0f;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc[AR_D];
    for (int d = 0; d < AR_D; ++d) acc[d] = 0.0f;
    float m = -1e30f, l = 0.0f;
    const float scale = 0.125f;

    for (int k0 = 0; k0 < kend; k0 += AR_BK) {{
        for (int i = t; i < AR_BK * AR_D; i += AR_BQ) {{
            int r = i / AR_D, d = i % AR_D;
            int gr = k0 + r;
            ks[r][d] = (gr < AR_N) ? K[kvbase + (long)gr * AR_D + d] : 0.0f;
            vs[r][d] = (gr < AR_N) ? V[kvbase + (long)gr * AR_D + d] : 0.0f;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float s[AR_BK];
        float mnew = m;
        for (int j = 0; j < AR_BK; ++j) {{
            int gj = k0 + j;
            float dot = -1e30f;
            if (gj < AR_N) {{
                dot = 0.0f;
                for (int d = 0; d < AR_D; ++d) dot += qs[t][d] * ks[j][d];
                dot *= scale;
                if (has_mask && row < AR_N) dot += M[(long)row * AR_N + gj];
                if (causal && gj > row) dot = -1e30f;
            }}
            s[j] = dot;
            mnew = max(mnew, dot);
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
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (row < AR_N) {{
        const float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
        for (int d = 0; d < AR_D; ++d) O[qbase + (long)row * AR_D + d] = acc[d] * inv;
    }}
}}
"#
    )
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
    ]
}

fn vals(opts: Options, h: i64) -> Vec<i64> {
    vec![opts.causal as i64, opts.has_mask as i64, h, opts.h_kv]
}

/// Launch with the default options (no mask, no causal, one K/V head per Q
/// head).
#[allow(clippy::too_many_arguments)] // mirrors the CL/CUDA kernel entry points
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

/// Launch with masks/GQA. `mask` is an additive `[n, n]` f32 tensor when
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
        &src(n),
        NAME,
        &signature(),
        &[q.clone(), k.clone(), v.clone(), o.clone(), m.clone()],
        grid,
        Some([BQ as usize, 1, 1]),
        &vals(opts, h),
        true,
    );
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
        device: "METAL".into(),
        src: src(n),
        name: NAME.to_string(),
        signature: signature(),
        bufs,
        global: [((n + BQ - 1) / BQ), b * h, 1],
        local: Some([BQ, 1, 1]),
        vals: vals(opts, h),
    }
}
