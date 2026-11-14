//! Fused Adam/AdamW/LAMB update for OpenCL (`AUTORUST_OPTIM_KERNEL`, default on).
//!
//! Same contract as [`crate::runtime::cuda_optim`]: one 256-work-item pass per
//! parameter updates m, v and p from one load of g/m/v, instead of the tensor
//! graph's four or five elementwise kernels. Ported because the CUDA-only kernel
//! left CL/VK at ~95 ms on the 100M-parameter benchmark while CUDA is 26 ms.
//! Vulkan is not ported yet: its compute path needs explicit descriptor sets,
//! unlike this generic `run_host_kernel` seam.

/// Smallest parameter worth a host kernel.
pub const MIN_ELEMS: i64 = 256;

/// Kernel entry point.
pub const NAME: &str = "autorust_adamw";

/// Work-items per group.
pub const THREADS: i64 = 256;

/// Render the OpenCL kernel for one optimizer configuration.
pub fn src(lr: f32, b1: f32, b2: f32, eps: f32, wd: f32) -> String {
    let one_minus_b1 = (1.0f64 - b1 as f64) as f32;
    let one_minus_b2 = (1.0f64 - b2 as f64) as f32;
    let wd_line = if wd > 0.0 {
        format!("  up += p[i] * {wd:?}f;")
    } else {
        String::new()
    };
    format!(
        r#"__kernel void {NAME}(__global float* p, __global const float* g,
                              __global float* m, __global float* v,
                              __global const float* b1t, __global const float* b2t,
                              long n) {{
  long i = (long)get_global_id(0);
  if (i >= n) return;
  float bc1 = 1.0f - b1t[0];
  float bc2 = 1.0f - b2t[0];
  float gi = g[i];
  float mi = m[i] * {b1:?}f + gi * {one_minus_b1:?}f;
  float vi = v[i] * {b2:?}f + gi * gi * {one_minus_b2:?}f;
  m[i] = mi;
  v[i] = vi;
  float up = (mi / bc1) / (sqrt(vi / bc2) + {eps:?}f);
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
        device: "CL".into(),
        src: src(lr, b1, b2, eps, wd),
        name: NAME.to_string(),
        signature: signature(),
        bufs,
        global: [(n + THREADS - 1) / THREADS, 1, 1],
        local: Some([THREADS, 1, 1]),
        vals: vec![n],
    }
}
