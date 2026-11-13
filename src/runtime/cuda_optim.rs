//! Fused Adam/AdamW/LAMB update for CUDA (`AUTORUST_OPTIM_KERNEL`, default on).
//!
//! The tensor graph lowers one optimizer step to four or five elementwise
//! kernels per parameter (the m store, the v store, a copy, and the update
//! re-reading the moments). For 100M parameters split over 32 tensors that is
//! ~160 launches and ~52 bytes of traffic per element, which measured 104 ms on
//! an RTX 3060 (27 GB/s effective). The scheduler cannot fuse them: the stores
//! are separate roots, and each parameter's flat extent (e.g. 3,125,000 =
//! 2^3 * 5^8) limits a generated kernel to 8-thread blocks.
//!
//! This kernel does the whole update in one pass with a 256-thread block over
//! the flat buffer: m/v are loaded and stored, `p` is updated in place, and the
//! bias-correction scalars are read from their 1-element buffers. Traffic is
//! the ideal 28 bytes per parameter.
//!
//! The graph remains the reference implementation: the fused path is only taken
//! for contiguous f32 CUDA parameters of at least [`MIN_ELEMS`] and can be
//! disabled with `AUTORUST_OPTIM_KERNEL=0` or
//! [`set_fused_optimizer`](crate::tensor::optim::set_fused_optimizer).

/// Smallest parameter that is worth a host kernel (below this the launch and
/// buffer resolution dominate; the graph path is also what the golden tests
/// exercise).
pub const MIN_ELEMS: i64 = 1024;

/// Kernel entry point.
pub const NAME: &str = "autorust_adamw";

/// Block size of the fused launch.
pub const THREADS: i64 = 256;

/// Render the kernel for one optimizer configuration.
///
/// `lr`, `b1`, `b2`, `eps` and `wd` are compile-time constants (they do not
/// change between steps), so the source is stable and NVRTC compiles it once.
/// The bias-correction factors stay buffers: they are updated every step.
pub fn src(lr: f32, b1: f32, b2: f32, eps: f32, wd: f32) -> String {
    let one_minus_b1 = (1.0f64 - b1 as f64) as f32;
    let one_minus_b2 = (1.0f64 - b2 as f64) as f32;
    let wd_line = if wd > 0.0 {
        format!("  up += p[i] * {wd:?}f;")
    } else {
        String::new()
    };
    format!(
        r#"extern "C" __global__ void {NAME}(float* p, const float* g, float* m, float* v,
                                             const float* b1t, const float* b2t, long n) {{
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float bc1 = 1.0f - b1t[0];
  float bc2 = 1.0f - b2t[0];
  float gi = g[i];
  float mi = m[i] * {b1:?}f + gi * {one_minus_b1:?}f;
  float vi = v[i] * {b2:?}f + gi * gi * {one_minus_b2:?}f;
  m[i] = mi;
  v[i] = vi;
  float up = (mi / bc1) / (sqrtf(vi / bc2) + {eps:?}f);
{wd_line}
  p[i] = p[i] - {lr:?}f * up;
}}
"#
    )
}

/// Argument list: six float buffers, then the element count.
pub fn signature() -> Vec<(i64, crate::dtype::DTypeId)> {
    let f = crate::dtype::FLOAT32;
    vec![
        (0, f),
        (1, f),
        (2, f),
        (3, f),
        (4, f),
        (5, f),
        (6, crate::dtype::INT64),
    ]
}

/// Captured-launch descriptor for one parameter's update.
#[allow(clippy::too_many_arguments)]
pub fn host_call(
    bufs: Vec<crate::runtime::buffer::Buffer>,
    n: i64,
    lr: f32,
    b1: f32,
    b2: f32,
    eps: f32,
    wd: f32,
) -> crate::engine::session::HostCall {
    crate::engine::session::HostCall {
        device: "CUDA".into(),
        src: src(lr, b1, b2, eps, wd),
        name: NAME.to_string(),
        signature: signature(),
        bufs,
        global: [(n + THREADS - 1) / THREADS, 1, 1],
        local: Some([THREADS, 1, 1]),
        vals: vec![n],
    }
}
