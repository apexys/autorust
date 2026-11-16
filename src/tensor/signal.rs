//! Signal-processing primitives: window functions, DFT/FFT and STFT.
//!
//! All transforms are ordinary tensor graphs: they emit `reshape`, movement,
//! elementwise and `matmul` UOps, so the existing scheduler/codegen compiles
//! them without a dedicated FFT kernel. Complex values use a trailing size-2
//! axis (`[..., 2]` = real, imag), matching ONNX's complex convention.
//!
//! `fft`/`cfft` use radix-2 Cooley–Tukey and require a power-of-two transform
//! length. `dft`/`cdft` are the arbitrary-length fallback: a precomputed
//! cosine/sine matrix multiply (`O(n²)` constants and work). `rfft` and `stft`
//! choose `fft` for power-of-two lengths and `dft` otherwise. `rfft` currently
//! runs the full complex transform and slices the onesided part; the packed
//! real-FFT optimization is future work.
//!
//! Transform axes are resolved Python-style (`-1` is the last signal axis, not
//! the complex axis). All lengths and shapes are static at graph construction
//! time; there is no data-dependent FFT length.

use crate::dtype::{self, ConstVal, DTypeId};
use crate::tensor::ix::Ix;
use crate::tensor::movement;
use crate::tensor::{AsCtx, Tensor};
use crate::uop::ctx::UOpCtx;
use crate::uop::sint::SVal;

const TWO_PI: f64 = std::f64::consts::TAU;

/// The arithmetic dtype for a transform: f64 stays f64; everything else
/// (f32/f16/bf16) is computed in f32 and cast back by the public wrapper.
fn compute_dtype(dt: DTypeId) -> DTypeId {
    if dt == dtype::FLOAT64 {
        dtype::FLOAT64
    } else {
        dtype::FLOAT32
    }
}

/// A real tensor as a complex tensor with a zero imaginary part.
fn to_complex_ctx(x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
    let dt = x.dtype_ctx(ctx);
    let sh = x.shape_ctx(ctx);
    let zero = Tensor::full_const(&mut *ctx, sh, ConstVal::Float(0.0), Some(dt));
    let re = x.unsqueeze_ctx(ctx, -1);
    let im = zero.unsqueeze_ctx(ctx, -1);
    re.cat_ctx(ctx, &[im], -1)
}

/// Split a complex tensor `[..., 2]` into its real and imaginary parts.
fn complex_parts_ctx(z: &Tensor, ctx: &mut UOpCtx) -> (Tensor, Tensor) {
    let rank = z.shape_ctx(ctx).len();
    assert!(
        rank >= 1 && z.shape_ctx(ctx).as_slice()[rank - 1].clone().as_i64() == Some(2),
        "TGERR: complex tensor must have a trailing size-2 axis"
    );
    let mut ixs = vec![Ix::All; rank];
    ixs[rank - 1] = Ix::Index(0);
    let re = z.getitem_ctx(ctx, &ixs);
    ixs[rank - 1] = Ix::Index(1);
    let im = z.getitem_ctx(ctx, &ixs);
    (re, im)
}

/// Combine real/imaginary parts into a trailing size-2 axis.
fn make_complex_ctx(re: &Tensor, im: &Tensor, ctx: &mut UOpCtx) -> Tensor {
    let re = re.unsqueeze_ctx(ctx, -1);
    let im = im.unsqueeze_ctx(ctx, -1);
    re.cat_ctx(ctx, &[im], -1)
}

/// `(a + bi) * (c + di)`.
fn complex_mul_ctx(a: &Tensor, b: &Tensor, ctx: &mut UOpCtx) -> Tensor {
    let (ar, ai) = complex_parts_ctx(a, ctx);
    let (br, bi) = complex_parts_ctx(b, ctx);
    let ai_bi = ai.mul_ctx(ctx, &bi);
    let ai_br = ai.mul_ctx(ctx, &br);
    let re = ar.mul_ctx(ctx, &br).sub_ctx(ctx, &ai_bi);
    let im = ar.mul_ctx(ctx, &bi).add_ctx(ctx, &ai_br);
    make_complex_ctx(&re, &im, ctx)
}

/// Twiddle factors `exp(±2πi j / m)` as `[m/2, 2]`.
fn twiddle_ctx(
    ctx: &mut UOpCtx,
    m: usize,
    inverse: bool,
    dt: DTypeId,
    device: Option<&str>,
) -> Tensor {
    let half = m / 2;
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut vals: Vec<f64> = Vec::with_capacity(half * 2);
    for j in 0..half {
        let a = sign * TWO_PI * (j as f64) / (m as f64);
        vals.push(a.cos());
        vals.push(a.sin());
    }
    Tensor::from_slice(&mut *ctx, &vals, Some(dt), device).reshape_ctx(ctx, [half as i64, 2])
}

/// Cooley–Tukey radix-2 FFT along the second-to-last axis of a complex tensor.
fn cfft_last_ctx(x: &Tensor, inverse: bool, ctx: &mut UOpCtx) -> Tensor {
    let shape: Vec<SVal> = x.shape_ctx(ctx).into_vec();
    let rank = shape.len();
    assert!(
        rank >= 2 && shape[rank - 1].clone().as_i64() == Some(2),
        "TGERR: fft expects a complex tensor with a trailing size-2 axis"
    );
    let n = shape[rank - 2]
        .clone()
        .as_i64()
        .expect("TGERR: fft length must be static");
    assert!(n > 0, "TGERR: fft length must be positive");
    assert!(
        (n as u64).is_power_of_two(),
        "TGERR: fft requires a power-of-two length (got {n}); use `dft` for arbitrary lengths"
    );
    let k = n.trailing_zeros() as usize;
    if k == 0 {
        return x.clone();
    }
    let p = rank - 2;
    let dt = x.dtype_ctx(ctx);
    let device = x.device_ctx(ctx);
    let prefix = &shape[..p];

    // Bit-reversal permutation: reshape to [2; k], transpose the bits in
    // reverse order, then flatten back.
    let mut split_shape: Vec<SVal> = prefix.to_vec();
    for _ in 0..k {
        split_shape.push(SVal::I(2));
    }
    split_shape.push(SVal::I(2));
    let mut z = x.reshape_ctx(ctx, split_shape);
    let mut order: Vec<i64> = (0..p as i64).collect();
    for i in (0..k).rev() {
        order.push((p + i) as i64);
    }
    order.push((p + k) as i64);
    z = z.permute_ctx(ctx, &order);
    let mut flat: Vec<SVal> = prefix.to_vec();
    flat.push(SVal::I(n as i128));
    flat.push(SVal::I(2));
    z = z.reshape_ctx(ctx, flat.clone());

    // Butterflies: stage m = 2, 4, ..., n.
    for s in 1..=k {
        let m = 1usize << s;
        let half = m / 2;
        let blocks = (n as usize) / m;
        let mut stage_shape: Vec<SVal> = prefix.to_vec();
        stage_shape.push(SVal::I(blocks as i128));
        stage_shape.push(SVal::I(2));
        stage_shape.push(SVal::I(half as i128));
        stage_shape.push(SVal::I(2));
        let zz = z.reshape_ctx(ctx, stage_shape);
        let axis = (p + 1) as i64;
        let parts = zz.split_ctx(ctx, &[1, 1], axis);
        let even = parts[0].squeeze_ctx(ctx, Some(axis));
        let odd = parts[1].squeeze_ctx(ctx, Some(axis));
        let tw = twiddle_ctx(ctx, m, inverse, dt, device.as_deref());
        let mut tw_shape: Vec<SVal> = vec![SVal::I(1); p + 1];
        tw_shape.push(SVal::I(half as i128));
        tw_shape.push(SVal::I(2));
        let tw = tw.reshape_ctx(ctx, tw_shape);
        let t = complex_mul_ctx(&odd, &tw, ctx);
        let even_plus = even.add_ctx(ctx, &t);
        let even_minus = even.sub_ctx(ctx, &t);
        let even_plus = even_plus.unsqueeze_ctx(ctx, axis);
        let even_minus = even_minus.unsqueeze_ctx(ctx, axis);
        z = even_plus
            .cat_ctx(ctx, &[even_minus], axis)
            .reshape_ctx(ctx, flat.clone());
    }

    if inverse {
        z = z.mul_ctx(ctx, 1.0 / (n as f64));
    }
    z
}

/// Move a transform axis to the second-to-last position of a complex tensor.
fn move_axis_to_last_signal_ctx(z: &Tensor, axis: i64, ctx: &mut UOpCtx) -> (Tensor, Vec<i64>) {
    let rank = z.shape_ctx(ctx).len();
    let signal_rank = rank - 1;
    let a = movement::resolve_dim(signal_rank, axis, false);
    let mut perm: Vec<i64> = (0..signal_rank)
        .filter(|&d| d != a)
        .map(|d| d as i64)
        .collect();
    perm.push(a as i64);
    perm.push((rank - 1) as i64);
    let z = if perm.iter().enumerate().all(|(i, &p)| i as i64 == p) {
        z.clone()
    } else {
        z.permute_ctx(ctx, &perm)
    };
    // Inverse order: [original signal dims, n, complex].
    let mut inv: Vec<i64> = Vec::with_capacity(rank);
    let mut next = 0i64;
    for i in 0..signal_rank {
        if i == a {
            inv.push((signal_rank - 1) as i64);
        } else {
            inv.push(next);
            next += 1;
        }
    }
    inv.push((rank - 1) as i64);
    (z, inv)
}

/// Cooley–Tukey FFT of a complex tensor along `axis`.
fn cfft_ctx_impl(x: &Tensor, axis: i64, inverse: bool, ctx: &mut UOpCtx) -> Tensor {
    let orig = x.dtype_ctx(ctx);
    let cdt = compute_dtype(orig);
    let z = if orig == cdt {
        x.clone()
    } else {
        x.cast_ctx(ctx, cdt)
    };
    let (z, inv) = move_axis_to_last_signal_ctx(&z, axis, ctx);
    let out = cfft_last_ctx(&z, inverse, ctx);
    let out = if inv.iter().enumerate().all(|(i, &p)| i as i64 == p) {
        out
    } else {
        out.permute_ctx(ctx, &inv)
    };
    if orig == cdt {
        out
    } else {
        out.cast_ctx(ctx, orig)
    }
}

/// DFT matrices `(cos, sin)` for `n` points: `[n, n]` each.
fn dft_matrices_ctx(
    ctx: &mut UOpCtx,
    n: usize,
    dt: DTypeId,
    device: Option<&str>,
) -> (Tensor, Tensor) {
    let mut c = Vec::with_capacity(n * n);
    let mut s = Vec::with_capacity(n * n);
    for row in 0..n {
        for col in 0..n {
            let a = TWO_PI * (row as f64) * (col as f64) / (n as f64);
            c.push(a.cos());
            s.push(a.sin());
        }
    }
    let c =
        Tensor::from_slice(&mut *ctx, &c, Some(dt), device).reshape_ctx(ctx, [n as i64, n as i64]);
    let s =
        Tensor::from_slice(&mut *ctx, &s, Some(dt), device).reshape_ctx(ctx, [n as i64, n as i64]);
    (c, s)
}

/// Arbitrary-length DFT along the second-to-last axis of a complex tensor.
fn cdft_last_ctx(x: &Tensor, inverse: bool, ctx: &mut UOpCtx) -> Tensor {
    let shape: Vec<SVal> = x.shape_ctx(ctx).into_vec();
    let rank = shape.len();
    assert!(
        rank >= 2 && shape[rank - 1].clone().as_i64() == Some(2),
        "TGERR: dft expects a complex tensor with a trailing size-2 axis"
    );
    let n = shape[rank - 2]
        .clone()
        .as_i64()
        .expect("TGERR: dft length must be static");
    assert!(n > 0, "TGERR: dft length must be positive");
    if n == 1 {
        return x.clone();
    }
    let p = rank - 2;
    let dt = x.dtype_ctx(ctx);
    let device = x.device_ctx(ctx);
    let p_vol: i64 = shape[..p]
        .iter()
        .map(|s| s.clone().as_i64().unwrap_or(1))
        .product();
    let (cos, sin) = dft_matrices_ctx(ctx, n as usize, dt, device.as_deref());
    let (xr, xi) = complex_parts_ctx(x, ctx);
    let xr = xr.reshape_ctx(ctx, [p_vol, n]);
    let xi = xi.reshape_ctx(ctx, [p_vol, n]);
    let xr_cos = xr.matmul_ctx(ctx, &cos);
    let xi_sin = xi.matmul_ctx(ctx, &sin);
    let xi_cos = xi.matmul_ctx(ctx, &cos);
    let xr_sin = xr.matmul_ctx(ctx, &sin);
    let (rr, ii) = if inverse {
        (xr_cos.sub_ctx(ctx, &xi_sin), xi_cos.add_ctx(ctx, &xr_sin))
    } else {
        (xr_cos.add_ctx(ctx, &xi_sin), xi_cos.sub_ctx(ctx, &xr_sin))
    };
    let mut out_shape: Vec<SVal> = shape[..p].to_vec();
    out_shape.push(SVal::I(n as i128));
    let mut rr = rr.reshape_ctx(ctx, out_shape.clone());
    let mut ii = ii.reshape_ctx(ctx, out_shape);
    if inverse {
        let inv_n = 1.0 / (n as f64);
        rr = rr.mul_ctx(ctx, inv_n);
        ii = ii.mul_ctx(ctx, inv_n);
    }
    make_complex_ctx(&rr, &ii, ctx)
}

/// Arbitrary-length DFT of a complex tensor along `axis`.
fn cdft_ctx_impl(x: &Tensor, axis: i64, inverse: bool, ctx: &mut UOpCtx) -> Tensor {
    let orig = x.dtype_ctx(ctx);
    let cdt = compute_dtype(orig);
    let z = if orig == cdt {
        x.clone()
    } else {
        x.cast_ctx(ctx, cdt)
    };
    let (z, inv) = move_axis_to_last_signal_ctx(&z, axis, ctx);
    let out = cdft_last_ctx(&z, inverse, ctx);
    let out = if inv.iter().enumerate().all(|(i, &p)| i as i64 == p) {
        out
    } else {
        out.permute_ctx(ctx, &inv)
    };
    if orig == cdt {
        out
    } else {
        out.cast_ctx(ctx, orig)
    }
}

impl Tensor {
    /// Complex FFT along `axis` (Cooley–Tukey radix-2).
    ///
    /// The tensor must be complex: `[..., n, 2]` with real/imag in the last
    /// axis. `axis` indexes the signal axes (not the complex axis), so `-1`
    /// transforms the last signal axis. `n` must be a power of two; use
    /// [`Tensor::dft`] for arbitrary lengths. `inverse` applies the inverse
    /// transform including the `1/n` normalization.
    pub fn cfft(&self, axis: i64, inverse: bool) -> Tensor {
        self.cfft_ctx(&mut self.ctx.borrow_mut(), axis, inverse)
    }

    /// [`Tensor::cfft`] with an explicit context.
    #[doc(hidden)]
    pub fn cfft_ctx(&self, ctx: &mut UOpCtx, axis: i64, inverse: bool) -> Tensor {
        cfft_ctx_impl(self, axis, inverse, ctx)
    }

    /// Real-input FFT along `axis`; returns a complex `[..., n, 2]` tensor.
    ///
    /// `n` must be a power of two; see [`Tensor::dft`] for arbitrary lengths.
    pub fn fft(&self, axis: i64, inverse: bool) -> Tensor {
        self.fft_ctx(&mut self.ctx.borrow_mut(), axis, inverse)
    }

    /// [`Tensor::fft`] with an explicit context.
    #[doc(hidden)]
    pub fn fft_ctx(&self, ctx: &mut UOpCtx, axis: i64, inverse: bool) -> Tensor {
        let z = to_complex_ctx(self, ctx);
        cfft_ctx_impl(&z, axis, inverse, ctx)
    }

    /// Complex DFT along `axis` for any static length (`O(n²)` matrix form).
    ///
    /// Input and output are complex `[..., n, 2]`. Use [`Tensor::cfft`] for the
    /// power-of-two fast path.
    pub fn cdft(&self, axis: i64, inverse: bool) -> Tensor {
        self.cdft_ctx(&mut self.ctx.borrow_mut(), axis, inverse)
    }

    /// [`Tensor::cdft`] with an explicit context.
    #[doc(hidden)]
    pub fn cdft_ctx(&self, ctx: &mut UOpCtx, axis: i64, inverse: bool) -> Tensor {
        cdft_ctx_impl(self, axis, inverse, ctx)
    }

    /// Real-input DFT along `axis` for any static length; returns a complex
    /// `[..., n, 2]` tensor.
    pub fn dft(&self, axis: i64, inverse: bool) -> Tensor {
        self.dft_ctx(&mut self.ctx.borrow_mut(), axis, inverse)
    }

    /// [`Tensor::dft`] with an explicit context.
    #[doc(hidden)]
    pub fn dft_ctx(&self, ctx: &mut UOpCtx, axis: i64, inverse: bool) -> Tensor {
        let z = to_complex_ctx(self, ctx);
        cdft_ctx_impl(&z, axis, inverse, ctx)
    }

    /// Real-input, onesided FFT: `[..., n]` -> `[..., n/2+1, 2]`.
    ///
    /// Uses [`Tensor::fft`] for power-of-two lengths and [`Tensor::dft`]
    /// otherwise.
    pub fn rfft(&self, axis: i64) -> Tensor {
        self.rfft_ctx(&mut self.ctx.borrow_mut(), axis)
    }

    /// [`Tensor::rfft`] with an explicit context.
    #[doc(hidden)]
    pub fn rfft_ctx(&self, ctx: &mut UOpCtx, axis: i64) -> Tensor {
        let shape: Vec<SVal> = self.shape_ctx(ctx).into_vec();
        let rank = shape.len();
        let a = movement::resolve_dim(rank, axis, false);
        let n = shape[a]
            .clone()
            .as_i64()
            .expect("TGERR: rfft length must be static");
        assert!(n > 0, "TGERR: rfft length must be positive");
        let z = if (n as u64).is_power_of_two() {
            let z = to_complex_ctx(self, ctx);
            cfft_ctx_impl(&z, axis, false, ctx)
        } else {
            let z = to_complex_ctx(self, ctx);
            cdft_ctx_impl(&z, axis, false, ctx)
        };
        let bins = n / 2 + 1;
        let out_rank = rank + 1;
        let mut ixs = vec![Ix::All; out_rank];
        ixs[out_rank - 2] = Ix::Range {
            start: Some(0),
            stop: Some(bins),
            step: 1,
        };
        z.getitem_ctx(ctx, &ixs)
    }

    /// Short-time Fourier transform of a real signal.
    ///
    /// `self` is `[..., length]`; frames are extracted along the last axis with
    /// `frame_length`/`frame_step` (no centering or padding). `window`, when
    /// present, is a 1-D tensor of `frame_length` values applied to every
    /// frame. The result is `[..., frames, bins, 2]`; `onesided` keeps
    /// `frame_length/2 + 1` bins, otherwise all `frame_length` bins.
    ///
    /// `frame_length` must be static; power-of-two lengths use the FFT path,
    /// other lengths use the DFT matrix.
    pub fn stft(
        &self,
        frame_length: i64,
        frame_step: i64,
        window: Option<&Tensor>,
        onesided: bool,
    ) -> Tensor {
        self.stft_ctx(
            &mut self.ctx.borrow_mut(),
            frame_length,
            frame_step,
            window,
            onesided,
        )
    }

    /// [`Tensor::stft`] with an explicit context.
    #[doc(hidden)]
    pub fn stft_ctx(
        &self,
        ctx: &mut UOpCtx,
        frame_length: i64,
        frame_step: i64,
        window: Option<&Tensor>,
        onesided: bool,
    ) -> Tensor {
        assert!(
            frame_length > 0,
            "TGERR: stft frame_length must be positive"
        );
        assert!(frame_step > 0, "TGERR: stft frame_step must be positive");
        let shape: Vec<SVal> = self.shape_ctx(ctx).into_vec();
        let length = shape
            .last()
            .cloned()
            .and_then(|s| s.as_i64())
            .expect("TGERR: stft signal length must be static");
        assert!(
            length >= frame_length,
            "TGERR: stft signal length {length} is shorter than frame_length {frame_length}"
        );
        let frames = self.pool_vec_ctx(ctx, &[frame_length], &[frame_step], &[1]);
        let frames = if let Some(w) = window {
            let wshape: Vec<i64> = w
                .shape_ctx(ctx)
                .as_slice()
                .iter()
                .map(|s| s.clone().as_i64().unwrap_or(0))
                .collect();
            assert_eq!(
                wshape.as_slice(),
                &[frame_length],
                "TGERR: stft window must have shape [frame_length]"
            );
            frames.mul_ctx(ctx, w)
        } else {
            frames
        };
        if onesided {
            frames.rfft_ctx(ctx, -1)
        } else if (frame_length as u64).is_power_of_two() {
            frames.fft_ctx(ctx, -1, false)
        } else {
            frames.dft_ctx(ctx, -1, false)
        }
    }
}

/// Window function family.
#[derive(Clone, Copy, Debug)]
enum WindowKind {
    Hann,
    Hamming,
    Blackman,
}

fn window_ctx(
    ctx: &mut UOpCtx,
    size: i64,
    periodic: bool,
    dtype_opt: Option<DTypeId>,
    kind: WindowKind,
) -> Tensor {
    assert!(size > 0, "TGERR: window size must be positive");
    let dt = dtype_opt.unwrap_or_else(dtype::default_float);
    let denom = if periodic || size <= 1 {
        size as f64
    } else {
        (size - 1) as f64
    };
    let n = Tensor::arange(&mut *ctx, 0.0, Some(size as f64), 1.0, Some(dt));
    let angle = n.mul_ctx(ctx, TWO_PI / denom);
    let c = crate::tensor::nn::cos_ctx(&angle, ctx);
    match kind {
        WindowKind::Hann => c.mul_ctx(ctx, -0.5).add_ctx(ctx, 0.5),
        WindowKind::Hamming => c.mul_ctx(ctx, -0.46).add_ctx(ctx, 0.54),
        WindowKind::Blackman => {
            let angle2 = angle.mul_ctx(ctx, 2.0);
            let c2 = crate::tensor::nn::cos_ctx(&angle2, ctx);
            let a = c.mul_ctx(ctx, -0.5);
            let b = c2.mul_ctx(ctx, 0.08);
            a.add_ctx(ctx, &b).add_ctx(ctx, 0.42)
        }
    }
}

/// Periodic (or symmetric) Hann window of `size` values.
pub fn hann_window(ctx: impl AsCtx, size: i64, periodic: bool, dtype: Option<DTypeId>) -> Tensor {
    ctx.with_ctx(|ctx| window_ctx(ctx, size, periodic, dtype, WindowKind::Hann))
}

/// Periodic (or symmetric) Hamming window of `size` values.
pub fn hamming_window(
    ctx: impl AsCtx,
    size: i64,
    periodic: bool,
    dtype: Option<DTypeId>,
) -> Tensor {
    ctx.with_ctx(|ctx| window_ctx(ctx, size, periodic, dtype, WindowKind::Hamming))
}

/// Periodic (or symmetric) Blackman window of `size` values.
pub fn blackman_window(
    ctx: impl AsCtx,
    size: i64,
    periodic: bool,
    dtype: Option<DTypeId>,
) -> Tensor {
    ctx.with_ctx(|ctx| window_ctx(ctx, size, periodic, dtype, WindowKind::Blackman))
}
