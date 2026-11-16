//! Session-aware fused full reductions (`sum` / `amax`), CUDA and Vulkan.
//!
//! [`crate::tensor::Tensor::sum_with`] and `max_with` use one streaming pass
//! plus a tiny finish kernel when the input is a contiguous f32 tensor reduced
//! to a scalar. Every other case falls back to the graph reduction, which
//! remains the numeric reference.

use crate::engine::session::Session;
#[cfg(any(
    all(
        feature = "backend_cuda",
        any(target_os = "linux", target_os = "windows")
    ),
    feature = "backend_vk"
))]
use crate::runtime::buffer::{Buffer, buffer_of};
use crate::tensor::Tensor;
#[cfg(any(
    all(
        feature = "backend_cuda",
        any(target_os = "linux", target_os = "windows")
    ),
    feature = "backend_vk"
))]
use crate::uop::sint::SVal;

#[cfg(any(
    all(
        feature = "backend_cuda",
        any(target_os = "linux", target_os = "windows")
    ),
    feature = "backend_vk"
))]
fn static_numel(x: &Tensor) -> Option<i64> {
    let shape = x.shape();
    let dims: Option<Vec<i64>> = shape.iter().map(|s| s.clone().as_i64()).collect();
    dims?.iter().try_fold(1i64, |a, &b| a.checked_mul(b))
}

#[cfg(any(
    all(
        feature = "backend_cuda",
        any(target_os = "linux", target_os = "windows")
    ),
    feature = "backend_vk"
))]
fn enabled() -> bool {
    !crate::helpers::env::get("AUTORUST_REDUCE_KERNEL").is_some_and(|v| v == "0" || v.is_empty())
}

/// Number of partials written by the streaming pass.
#[cfg(any(
    all(
        feature = "backend_cuda",
        any(target_os = "linux", target_os = "windows")
    ),
    feature = "backend_vk"
))]
const PARTIALS: i64 = 256;

#[cfg(any(
    all(
        feature = "backend_cuda",
        any(target_os = "linux", target_os = "windows")
    ),
    feature = "backend_vk"
))]
fn alloc_out(
    ctx: &crate::tensor::Ctx,
    sess: &mut Session,
    dev: &str,
    shape: &[SVal],
) -> (Tensor, Buffer) {
    let u = {
        let mut c = ctx.borrow_mut();
        crate::tensor::creation::empty(&mut c, shape, crate::dtype::FLOAT32, dev)
    };
    let t = Tensor::from_uop_ctx(ctx, u.clone());
    let b = {
        let mut c = ctx.borrow_mut();
        buffer_of(&mut c, &mut sess.buffers, u)
    };
    b.ensure_allocated(&crate::runtime::device::device(dev));
    t.set_realized_buffer(b.clone());
    (t, b)
}

#[cfg(any(
    all(
        feature = "backend_cuda",
        any(target_os = "linux", target_os = "windows")
    ),
    feature = "backend_vk"
))]
fn temp_partials(dev: &str) -> Buffer {
    let b = Buffer::new_base(dev, PARTIALS as usize, crate::dtype::FLOAT32);
    b.ensure_allocated(&crate::runtime::device::device(dev));
    b
}

#[cfg(any(
    all(
        feature = "backend_cuda",
        any(target_os = "linux", target_os = "windows")
    ),
    feature = "backend_vk"
))]
fn input_buffer(x: &Tensor, _sess: &mut Session) -> Option<Buffer> {
    // Require an already-realized input: realizing a view inside a captured
    // step would replay its producer, and a realized view's buffer layout is
    // not necessarily dense. `Tensor::realize()` first is the explicit
    // contract; anything else uses the graph reduction.
    x.realized_buffer()
}

#[cfg(any(
    all(
        feature = "backend_cuda",
        any(target_os = "linux", target_os = "windows")
    ),
    feature = "backend_vk"
))]
fn eligible(x: &Tensor, axis: Option<&[usize]>, keepdim: bool) -> Option<(String, i64)> {
    if !enabled() || axis.is_some() || keepdim || x.dtype() != crate::dtype::FLOAT32 {
        return None;
    }
    let n = static_numel(x)?;
    // Both fused reduction kernels use the same 1M-element minimum.
    let min = 1i64 << 20;
    if n < min {
        return None;
    }
    let dev = x.device()?;
    if !(dev.starts_with("CUDA") || dev.starts_with("VK")) {
        return None;
    }
    Some((dev, n))
}

#[cfg(any(
    all(
        feature = "backend_cuda",
        any(target_os = "linux", target_os = "windows")
    ),
    feature = "backend_vk"
))]
#[allow(clippy::too_many_lines)]
fn sum_with_impl(
    x: &Tensor,
    axis: Option<&[usize]>,
    keepdim: bool,
    dtype: Option<crate::dtype::DTypeId>,
    sess: &mut Session,
) -> Option<Tensor> {
    let (dev, n) = eligible(x, axis, keepdim)?;
    if !dtype.is_none_or(|d| d == crate::dtype::FLOAT32) {
        return None;
    }
    let in_buf = input_buffer(x, sess)?;
    let (out, out_buf) = alloc_out(x.ctx(), sess, &dev, &[]);
    let temp = temp_partials(&dev);
    let (mut first, mut finish) = if dev.starts_with("VK") {
        #[cfg(feature = "backend_vk")]
        {
            (
                crate::runtime::vk_reduce::sum_partials(vec![in_buf, temp.clone()], n),
                crate::runtime::vk_reduce::sum_finish(vec![temp, out_buf], PARTIALS),
            )
        }
        #[cfg(not(feature = "backend_vk"))]
        unreachable!("VK device without backend_vk")
    } else {
        #[cfg(all(
            feature = "backend_cuda",
            any(target_os = "linux", target_os = "windows")
        ))]
        {
            (
                crate::runtime::cuda_reduce::sum_partials(vec![in_buf, temp.clone()], n),
                crate::runtime::cuda_reduce::sum_finish(vec![temp, out_buf], PARTIALS),
            )
        }
        #[cfg(not(all(
            feature = "backend_cuda",
            any(target_os = "linux", target_os = "windows")
        )))]
        unreachable!("CUDA device without backend_cuda")
    };
    first.device = crate::runtime::device::canonicalize_device_name(&dev);
    sess.capture_host(first.clone());
    first.exec();
    finish.device = first.device.clone();
    sess.capture_host(finish.clone());
    finish.exec();
    Some(out)
}

#[cfg(any(
    all(
        feature = "backend_cuda",
        any(target_os = "linux", target_os = "windows")
    ),
    feature = "backend_vk"
))]
pub fn sum_with(
    x: &Tensor,
    axis: Option<&[usize]>,
    keepdim: bool,
    dtype: Option<crate::dtype::DTypeId>,
    sess: &mut Session,
) -> Tensor {
    if let Some(out) = sum_with_impl(x, axis, keepdim, dtype, sess) {
        out
    } else {
        x.sum(axis, keepdim, dtype)
    }
}

#[cfg(not(any(
    all(
        feature = "backend_cuda",
        any(target_os = "linux", target_os = "windows")
    ),
    feature = "backend_vk"
)))]
pub fn sum_with(
    x: &Tensor,
    axis: Option<&[usize]>,
    keepdim: bool,
    dtype: Option<crate::dtype::DTypeId>,
    sess: &mut Session,
) -> Tensor {
    let _ = sess;
    x.sum(axis, keepdim, dtype)
}

#[cfg(any(
    all(
        feature = "backend_cuda",
        any(target_os = "linux", target_os = "windows")
    ),
    feature = "backend_vk"
))]
pub fn max_with(x: &Tensor, axis: Option<&[usize]>, keepdim: bool, sess: &mut Session) -> Tensor {
    let Some((dev, n)) = eligible(x, axis, keepdim) else {
        return x.max(axis, keepdim);
    };
    let Some(in_buf) = input_buffer(x, sess) else {
        return x.max(axis, keepdim);
    };
    let (out, out_buf) = alloc_out(x.ctx(), sess, &dev, &[]);
    let temp = temp_partials(&dev);
    let (mut first, mut finish) = if dev.starts_with("VK") {
        #[cfg(feature = "backend_vk")]
        {
            (
                crate::runtime::vk_reduce::amax_partials(vec![in_buf, temp.clone()], n),
                crate::runtime::vk_reduce::amax_finish(vec![temp, out_buf], PARTIALS),
            )
        }
        #[cfg(not(feature = "backend_vk"))]
        unreachable!("VK device without backend_vk")
    } else {
        #[cfg(all(
            feature = "backend_cuda",
            any(target_os = "linux", target_os = "windows")
        ))]
        {
            (
                crate::runtime::cuda_reduce::amax_partials(vec![in_buf, temp.clone()], n),
                crate::runtime::cuda_reduce::amax_finish(vec![temp, out_buf], PARTIALS),
            )
        }
        #[cfg(not(all(
            feature = "backend_cuda",
            any(target_os = "linux", target_os = "windows")
        )))]
        unreachable!("CUDA device without backend_cuda")
    };
    first.device = crate::runtime::device::canonicalize_device_name(&dev);
    sess.capture_host(first.clone());
    first.exec();
    finish.device = first.device.clone();
    sess.capture_host(finish.clone());
    finish.exec();
    out
}

#[cfg(not(any(
    all(
        feature = "backend_cuda",
        any(target_os = "linux", target_os = "windows")
    ),
    feature = "backend_vk"
)))]
pub fn max_with(x: &Tensor, axis: Option<&[usize]>, keepdim: bool, sess: &mut Session) -> Tensor {
    let _ = sess;
    x.max(axis, keepdim)
}
