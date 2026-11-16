//! Session-aware fused kxk convolution (CUDA tensor cores, forward-only).
//!
//! `Tensor::conv2d_with` uses `runtime::cuda_conv_tc` for groups=1, dilation=1,
//! square `k x k`, symmetric `(k-1)/2` padding, stride 1 or 2, f32 convolutions
//! with a realized input. The graph conv remains the fallback and the reference;
//! training stays on it. The host path is only taken where it measured faster
//! than the graph: the measured 3x3 stride-1 case, and low-K convs such as
//! ResNet's 7x7 stem (K = 3*49 = 147) that fall below the im2col floor.

use crate::engine::session::Session;
use crate::runtime::buffer::{Buffer, buffer_of};
use crate::tensor::Tensor;
use crate::uop::sint::SVal;

fn enabled(x: &Tensor) -> bool {
    match crate::helpers::env::get("AUTORUST_CONV_TC")
        .or_else(|| crate::helpers::env::get("AUTORUST_CONV3X3_TC"))
    {
        Some(v) => v != "0" && !v.is_empty(),
        None => x.ctx().borrow().settings.math_mode == crate::settings::MathMode::Fast,
    }
}

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

fn realized_weight(t: &Tensor, sess: &mut Session) -> Option<Buffer> {
    if let Some(b) = t.realized_buffer() {
        return Some(b);
    }
    let mut c = t.ctx().borrow_mut();
    t.realize_device_ctx(&mut c, sess);
    t.realized_buffer()
}

#[cfg(all(
    feature = "backend_cuda",
    any(target_os = "linux", target_os = "windows")
))]
#[allow(clippy::too_many_arguments)]
pub fn conv_with(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    groups: i64,
    stride: i64,
    dilation: i64,
    padding: &[i64],
    dtype: Option<crate::dtype::DTypeId>,
    sess: &mut Session,
) -> Option<Tensor> {
    if !enabled(x)
        || dtype.is_some()
        || groups != 1
        || !(stride == 1 || stride == 2)
        || dilation != 1
        || x.dtype() != crate::dtype::FLOAT32
        || weight.dtype() != crate::dtype::FLOAT32
    {
        return None;
    }
    let xs = x.shape();
    let ws = weight.shape();
    if xs.len() != 4 || ws.len() != 4 {
        return None;
    }
    let xd: Option<Vec<i64>> = xs.iter().map(|s| s.clone().as_i64()).collect();
    let wd: Option<Vec<i64>> = ws.iter().map(|s| s.clone().as_i64()).collect();
    let (xd, wd) = (xd?, wd?);
    let (b, c, h, w) = (xd[0], xd[1], xd[2], xd[3]);
    let (o, cin, kh, kw) = (wd[0], wd[1], wd[2], wd[3]);
    if cin != c || kh != kw || kh < 3 || o < 16 || b <= 0 || h <= 0 || w <= 0 {
        return None;
    }
    let pad = (kh - 1) / 2;
    if padding.len() != 4 || padding.iter().any(|&p| p != pad) {
        return None;
    }
    // 3x3 stride 1 is the measured ResNet-18 win. Larger kernels are only
    // worth the host launch when the graph would use the direct convolution,
    // i.e. when the im2col contraction is below its floor (the 7x7 stem).
    let red = cin.saturating_mul(kh).saturating_mul(kw);
    let im2col_floor = crate::helpers::env::get("AUTORUST_IM2COL_MIN_K")
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or(256);
    if !((kh == 3 && stride == 1) || red < im2col_floor) {
        return None;
    }
    let oh = (h + 2 * pad - kh) / stride + 1;
    let ow = (w + 2 * pad - kw) / stride + 1;
    let rows = b.saturating_mul(oh).saturating_mul(ow);
    if rows < crate::runtime::cuda_conv_tc::min_out() {
        return None;
    }
    let in_buf = x.realized_buffer()?;
    let dev = in_buf.device().to_string();
    if !dev.starts_with("CUDA") {
        return None;
    }
    let w_buf = realized_weight(weight, sess)?;
    let (has_bias, b_buf) = match bias {
        Some(bias) => {
            if bias.dtype() != crate::dtype::FLOAT32 || bias.shape().len() != 1 {
                return None;
            }
            (true, realized_weight(bias, sess)?)
        }
        None => (false, in_buf.clone()),
    };
    let (out, out_buf) = alloc_out(
        x.ctx(),
        sess,
        &dev,
        &[
            SVal::I(b as i128),
            SVal::I(o as i128),
            SVal::I(oh as i128),
            SVal::I(ow as i128),
        ],
    );
    let mut call = crate::runtime::cuda_conv_tc::host_call(
        vec![in_buf, w_buf, b_buf, out_buf],
        b,
        c,
        h,
        w,
        o,
        has_bias,
        kh,
        kw,
        pad,
        stride,
    );
    call.device = crate::runtime::device::canonicalize_device_name(&dev);
    sess.capture_host(call.clone());
    call.exec();
    Some(out)
}

#[cfg(not(all(
    feature = "backend_cuda",
    any(target_os = "linux", target_os = "windows")
)))]
#[allow(clippy::too_many_arguments)]
pub fn conv_with(
    _x: &Tensor,
    _weight: &Tensor,
    _bias: Option<&Tensor>,
    _groups: i64,
    _stride: i64,
    _dilation: i64,
    _padding: &[i64],
    _dtype: Option<crate::dtype::DTypeId>,
    _sess: &mut Session,
) -> Option<Tensor> {
    None
}
