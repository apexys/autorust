//! FlashAttention — an op-level port of `extra/thunder/tiny/fa.py`.
//!
//! tinygrad's FA is a fused `@custom_kernel` over the `tk` warp DSL; the port
//! does not have that kernel machinery, so this is the **same algorithm** built
//! from `Tensor` ops: the FlashAttention/FA2 tiling (Q and KV blocks) with an
//! online (running) softmax, so the `N x N` score matrix is never materialized.
//! The result is numerically identical to
//! [`Tensor::scaled_dot_product_attention`](crate::tensor::Tensor::scaled_dot_product_attention)
//! (which this mirrors: `softmax(q·kᵀ/√d + mask)·v`), and backprop flows through
//! the ops, so it doubles as the FA backward gate.
//!
//! Differences from `fa.py` (documented, not behavioral for f32): the oracle
//! casts q/k/v to bf16 for the tensor cores and shards across devices; this port
//! accumulates in f32 and runs on one device. GQA is inferred from the head
//! counts (`H // H_KV`), exactly like `head_kv = head // GROUP_SIZE`.

use crate::dtype::{self, ConstVal};
use crate::tensor::{Ix, Tensor};
use crate::uop::ctx::UOpCtx;
use crate::uop::sint::SVal;

/// `Q_BLOCK_SIZE` / `KV_BLOCK_SIZE` (fa.py:11-12).
pub const Q_BLOCK_SIZE: i64 = 64;
/// Key/value tile size in the flash-attention loop (`fa.py:12`).
pub const KV_BLOCK_SIZE: i64 = 64;

/// Additive mask / running-max sentinel. A large *finite* negative (not `-inf`)
/// keeps fully-masked padded query rows finite (`0 * fin = 0` instead of
/// `0 * nan` in the backward), while masked logits still underflow to exactly
/// `exp(-1e30 - m) = 0`, so attended rows are unchanged.
const NEG: f64 = -1e30;

fn scalar(ctx: &mut UOpCtx, v: f64) -> Tensor {
    Tensor::full_const(&mut *ctx, &[] as &[i64], ConstVal::Float(v), None)
}

/// `flash_attention(xq, xk, xv, attn_mask=None, is_causal=False)`
/// (fa.py:26-28): tiled attention with the default block sizes.
pub fn flash_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    attn_mask: Option<&Tensor>,
    is_causal: bool,
) -> Tensor {
    flash_attention_ctx(q, k, v, attn_mask, is_causal, &mut q.ctx().borrow_mut())
}

#[doc(hidden)]
pub fn flash_attention_ctx(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    attn_mask: Option<&Tensor>,
    is_causal: bool,
    ctx: &mut UOpCtx,
) -> Tensor {
    flash_attention_blocks_ctx(
        q,
        k,
        v,
        attn_mask,
        is_causal,
        Q_BLOCK_SIZE,
        KV_BLOCK_SIZE,
        ctx,
    )
}

/// The tiled implementation with explicit block sizes (used by the tests to
/// force many small blocks). `q`/`k`/`v` are `(B, H, N, D)` (or `(H, N, D)`);
/// `k`/`v` may have fewer heads than `q` (GQA). Blocks tile the sequence; the
/// final Q/KV block takes the remainder (no padding), so there is no padded
/// region to mask.
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_blocks(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    attn_mask: Option<&Tensor>,
    is_causal: bool,
    bq: i64,
    bk: i64,
) -> Tensor {
    flash_attention_blocks_ctx(
        q,
        k,
        v,
        attn_mask,
        is_causal,
        bq,
        bk,
        &mut q.ctx().borrow_mut(),
    )
}

#[doc(hidden)]
#[allow(clippy::too_many_arguments)] // signature mirrors the tinygrad oracle
pub fn flash_attention_blocks_ctx(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    attn_mask: Option<&Tensor>,
    is_causal: bool,
    bq: i64,
    bk: i64,
    ctx: &mut UOpCtx,
) -> Tensor {
    assert!(bq > 0 && bk > 0, "TGERR: block sizes must be positive");
    let promoted = q.shape_ctx(ctx).len() == 3;
    let (q, k, v) = if promoted {
        (
            q.unsqueeze_ctx(ctx, 0),
            k.unsqueeze_ctx(ctx, 0),
            v.unsqueeze_ctx(ctx, 0),
        )
    } else {
        (q.clone(), k.clone(), v.clone())
    };
    let qs: Vec<SVal> = q.shape_ctx(ctx).into_vec();
    let (bs, hq) = (
        qs[0].clone().as_i64().unwrap_or(0),
        qs[1].clone().as_i64().unwrap_or(0),
    );
    let nq = qs[2].clone().as_i64().unwrap_or(0);
    let d = qs[3].clone().as_i64().unwrap_or(0);
    let ks: Vec<SVal> = k.shape_ctx(ctx).into_vec();
    let hkv = ks[1].clone().as_i64().unwrap_or(0);
    let nk = ks[2].clone().as_i64().unwrap_or(0);
    assert!(
        hkv > 0 && hq % hkv == 0,
        "TGERR: q heads ({hq}) must be a multiple of kv heads ({hkv})"
    );
    let group = hq / hkv;
    let (k, v) = if group > 1 {
        (
            k.repeat_interleave_ctx(ctx, group, Some(1)),
            v.repeat_interleave_ctx(ctx, group, Some(1)),
        )
    } else {
        (k, v)
    };

    // Combined additive mask on the (nq, nk) grid: 0 where attended, `NEG` where
    // not. Causal is the lower-triangular `j <= i` mask.
    let ninf = scalar(ctx, NEG);
    let zero = scalar(ctx, 0.0);
    let mut mask: Option<Tensor> = None;
    if is_causal {
        let rows = Tensor::arange(
            &mut *ctx,
            0.0,
            Some(nq as f64),
            1.0,
            Some(dtype::default_int()),
        )
        .reshape_ctx(ctx, [nq, 1]);
        let cols = Tensor::arange(
            &mut *ctx,
            0.0,
            Some(nk as f64),
            1.0,
            Some(dtype::default_int()),
        )
        .reshape_ctx(ctx, [1, nk]);
        mask = Some(cols.gt_ctx(ctx, &rows).where__ctx(ctx, &ninf, &zero));
    }
    if let Some(am) = attn_mask {
        let am = if am.dtype_ctx(ctx) == dtype::BOOL {
            am.where__ctx(ctx, &zero, &ninf)
        } else {
            am.clone()
        };
        mask = Some(match mask {
            None => am,
            Some(m) => m.add_ctx(ctx, &am),
        });
    }

    let scale = scalar(ctx, 1.0 / (d as f64).sqrt());
    let mut outs: Vec<Tensor> = Vec::new();
    for &(i0, i1) in &block_bounds(nq, bq) {
        let qb = i1 - i0;
        let qi = q.getitem_ctx(
            ctx,
            &[Ix::Ellipsis, Ix::slice(Some(i0), Some(i1), 1), Ix::All],
        );
        let ms = vec![
            SVal::I(bs as i128),
            SVal::I(hq as i128),
            SVal::I(qb as i128),
        ];
        let mut m = Tensor::full_const(&mut *ctx, ms.clone(), ConstVal::Float(NEG), None);
        let mut l = Tensor::full_const(&mut *ctx, ms.clone(), ConstVal::Float(0.0), None);
        let mut acc = Tensor::full_const(
            &mut *ctx,
            vec![
                SVal::I(bs as i128),
                SVal::I(hq as i128),
                SVal::I(qb as i128),
                SVal::I(d as i128),
            ],
            ConstVal::Float(0.0),
            None,
        );
        for &(j0, j1) in &block_bounds(nk, bk) {
            let kj = k.getitem_ctx(
                ctx,
                &[Ix::Ellipsis, Ix::slice(Some(j0), Some(j1), 1), Ix::All],
            );
            let vj = v.getitem_ctx(
                ctx,
                &[Ix::Ellipsis, Ix::slice(Some(j0), Some(j1), 1), Ix::All],
            );
            let kjt = kj.transpose_ctx(ctx, -2, -1);
            let mut s = qi.matmul_ctx(ctx, &kjt).mul_ctx(ctx, &scale); // (B, H, qb, kb)
            if let Some(mk) = &mask {
                let mb = mk.getitem_ctx(
                    ctx,
                    &[
                        Ix::Ellipsis,
                        Ix::slice(Some(i0), Some(i1), 1),
                        Ix::slice(Some(j0), Some(j1), 1),
                    ],
                );
                s = s.add_ctx(ctx, &mb);
            }
            let slast = s.shape_ctx(ctx).len() - 1;
            let row_max = s.max_ctx(ctx, Some(&[slast]), false);
            let m_new = m.maximum_ctx(ctx, &row_max);
            let alpha = m.sub_ctx(ctx, &m_new).exp_ctx(ctx);
            let mcol = m_new.unsqueeze_ctx(ctx, -1);
            let p = s.sub_ctx(ctx, &mcol).exp_ctx(ctx);
            let psum = p.sum_ctx(ctx, Some(&[slast]), false, None);
            let alpha_col = alpha.unsqueeze_ctx(ctx, -1);
            let l_new = l.mul_ctx(ctx, &alpha).add_ctx(ctx, &psum);
            let pv = p.matmul_ctx(ctx, &vj);
            let acc_new = acc.mul_ctx(ctx, &alpha_col).add_ctx(ctx, &pv);
            m = m_new;
            l = l_new;
            acc = acc_new;
        }
        let lr = l.reciprocal_ctx(ctx).unsqueeze_ctx(ctx, -1);
        outs.push(acc.mul_ctx(ctx, &lr));
    }

    let out = if outs.len() == 1 {
        outs.pop().unwrap()
    } else {
        let first = outs.remove(0);
        first.cat_ctx(ctx, &outs, 2)
    };
    if promoted {
        out.getitem_ctx(ctx, &[Ix::Index(0)])
    } else {
        out
    }
}

/// `(start, end)` for consecutive blocks of size `b` covering `0..n`; the last
/// block holds the remainder.
fn block_bounds(n: i64, b: i64) -> Vec<(i64, i64)> {
    let mut v = Vec::new();
    let mut s = 0;
    while s < n {
        let e = (s + b).min(n);
        v.push((s, e));
        s = e;
    }
    v
}

/// Fused CUDA attention through the hand-written FA2 kernel.
///
/// Returns `None` when the backend, dtype or shape is not covered (the caller
/// then falls back to [`Tensor::scaled_dot_product_attention`]), and never
/// handles masks or causality — those need the general path.
///
/// The launch is captured by the session in its exact position between the q/k/v
/// projections and the output projection, so a JIT replay reproduces the same
/// order.
pub fn fused_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    sess: &mut crate::engine::session::Session,
) -> Option<Tensor> {
    let dims = q.shape();
    let dims = dims.as_slice();
    if dims.len() != 4 {
        return None;
    }
    let i = |x: &SVal| x.clone().as_i64().unwrap_or(0);
    let (b, h, n, d) = (i(&dims[0]), i(&dims[1]), i(&dims[2]), i(&dims[3]));
    if q.dtype() != dtype::FLOAT32 {
        return None;
    }
    let dev = q.device().unwrap_or_default();
    #[cfg(feature = "backend_cuda")]
    let cuda = dev.starts_with("CUDA") && crate::runtime::cuda_flash::supports(n, d);
    #[cfg(not(feature = "backend_cuda"))]
    let cuda = false;
    #[cfg(feature = "backend_cl")]
    let cl = dev.starts_with("CL") && crate::runtime::cl_flash::supports(n, d);
    #[cfg(not(feature = "backend_cl"))]
    let cl = false;
    if !cuda && !cl {
        return None;
    }

    {
        let mut c = q.ctx().borrow_mut();
        q.realize_device_ctx(&mut c, sess);
        k.realize_device_ctx(&mut c, sess);
        v.realize_device_ctx(&mut c, sess);
    }
    let ctx = q.ctx().clone();
    let out = Tensor::from_slice(
        &ctx,
        &vec![0.0f32; (b * h * n * d) as usize],
        None,
        Some(&dev),
    )
    .reshape([b, h, n, d])
    .to_param();
    {
        let mut c = ctx.borrow_mut();
        out.realize_device_ctx(&mut c, sess);
    }
    let bufs = vec![
        q.realized_buffer()?,
        k.realized_buffer()?,
        v.realized_buffer()?,
        out.realized_buffer()?,
    ];
    let call = if cuda {
        #[cfg(feature = "backend_cuda")]
        {
            crate::runtime::cuda_flash::host_call(bufs, b, h, n)
        }
        #[cfg(not(feature = "backend_cuda"))]
        {
            return None;
        }
    } else {
        #[cfg(feature = "backend_cl")]
        {
            crate::runtime::cl_flash::host_call(bufs, b, h, n)
        }
        #[cfg(not(feature = "backend_cl"))]
        {
            return None;
        }
    };
    // Record the launch for replay *and* run it now: the capture pass executes
    // the graph, so the consumers that follow need the value.
    sess.capture_host(call.clone());
    call.exec();
    Some(out)
}

/// Selectable attention implementation (`AUTORUST_ATTN`).
///
/// Mirrors the `FA=1` switch on `models::transformer`, but as a first-class
/// choice so quantized variants can be added without touching call sites.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AttentionKind {
    /// Generated `softmax(q·kᵀ)v` (the reference path).
    Sdpa,
    /// Fused FA2 hand-written kernel (`runtime::cuda_flash` / `cl_flash`).
    Flash,
    /// SageAttention-style quantized attention. Not implemented yet: selecting
    /// it currently runs [`AttentionKind::Flash`] and says so once.
    Sage,
}

impl AttentionKind {
    /// `AUTORUST_ATTN=sdpa|flash|sage` (default `flash`).
    pub fn from_env() -> AttentionKind {
        match std::env::var("AUTORUST_ATTN").as_deref() {
            Ok("sdpa") | Ok("0") => AttentionKind::Sdpa,
            Ok("sage") => AttentionKind::Sage,
            _ => AttentionKind::Flash,
        }
    }

    /// Name for logging.
    pub fn name(self) -> &'static str {
        match self {
            AttentionKind::Sdpa => "sdpa",
            AttentionKind::Flash => "flash",
            AttentionKind::Sage => "sage",
        }
    }

    /// One-time warning when a requested variant degrades to another.
    pub fn effective(self) -> AttentionKind {
        match self {
            AttentionKind::Sage => {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    eprintln!(
                        "AUTORUST_ATTN=sage: quantized attention not implemented, using flash"
                    )
                });
                AttentionKind::Flash
            }
            k => k,
        }
    }
}
