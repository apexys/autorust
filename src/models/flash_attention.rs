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
pub const KV_BLOCK_SIZE: i64 = 64;

/// Additive mask / running-max sentinel. A large *finite* negative (not `-inf`)
/// keeps fully-masked padded query rows finite (`0 * fin = 0` instead of
/// `0 * nan` in the backward), while masked logits still underflow to exactly
/// `exp(-1e30 - m) = 0`, so attended rows are unchanged.
const NEG: f64 = -1e30;

fn scalar(ctx: &mut UOpCtx, v: f64) -> Tensor {
    Tensor::full_const(ctx, &[] as &[i64], ConstVal::Float(v), None)
}

fn scalar_i(ctx: &mut UOpCtx, v: i64) -> Tensor {
    Tensor::full_const(ctx, &[] as &[i64], ConstVal::Int(v as i128), Some(dtype::default_int()))
}

/// `flash_attention(xq, xk, xv, attn_mask=None, is_causal=False)`
/// (fa.py:26-28): tiled attention with the default block sizes.
pub fn flash_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    attn_mask: Option<&Tensor>,
    is_causal: bool,
    ctx: &mut UOpCtx,
) -> Tensor {
    flash_attention_blocks(q, k, v, attn_mask, is_causal, Q_BLOCK_SIZE, KV_BLOCK_SIZE, ctx)
}

/// The tiled implementation with explicit block sizes (used by the tests to
/// force many small blocks). `q`/`k`/`v` are `(B, H, N, D)` (or `(H, N, D)`);
/// `k`/`v` may have fewer heads than `q` (GQA).
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_blocks(
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
    let promoted = q.shape(ctx).len() == 3;
    let (q, k, v) = if promoted {
        (q.unsqueeze(ctx, 0), k.unsqueeze(ctx, 0), v.unsqueeze(ctx, 0))
    } else {
        (q.clone(), k.clone(), v.clone())
    };
    let qs: Vec<SVal> = q.shape(ctx).into_vec();
    let (bs, hq) = (qs[0].clone().as_i64().unwrap_or(0), qs[1].clone().as_i64().unwrap_or(0));
    let nq = qs[2].clone().as_i64().unwrap_or(0);
    let d = qs[3].clone().as_i64().unwrap_or(0);
    let ks: Vec<SVal> = k.shape(ctx).into_vec();
    let hkv = ks[1].clone().as_i64().unwrap_or(0);
    let nk = ks[2].clone().as_i64().unwrap_or(0);
    assert!(hkv > 0 && hq % hkv == 0, "TGERR: q heads ({hq}) must be a multiple of kv heads ({hkv})");
    let group = hq / hkv;

    // GQA: repeat each KV head `group` times so every Q head has its KV head.
    let (k, v) = if group > 1 {
        (k.repeat_interleave(ctx, group, Some(1)), v.repeat_interleave(ctx, group, Some(1)))
    } else {
        (k, v)
    };

    let nbq = (nq + bq - 1) / bq;
    let nbk = (nk + bk - 1) / bk;
    let nqp = nbq * bq;
    let nkp = nbk * bk;

    // Pad along the sequence (dim 2) to whole blocks.
    let seq_pad = |extra: i64| vec![(0, 0), (0, 0), (0, extra), (0, 0)];
    let q_pad = if nqp > nq { q.pad(ctx, &seq_pad(nqp - nq)) } else { q.clone() };
    let k_pad = if nkp > nk { k.pad(ctx, &seq_pad(nkp - nk)) } else { k.clone() };
    let v_pad = if nkp > nk { v.pad(ctx, &seq_pad(nkp - nk)) } else { v.clone() };

    // Combined additive mask on the (nq, nk) grid: 0 where attended, -inf where
    // not. Causal uses a lower-triangular `j <= i` mask (fa.py's `_tri`).
    let ninf = scalar(ctx, NEG);
    let zero = scalar(ctx, 0.0);
    let mut mask: Option<Tensor> = None;
    if is_causal {
        let rows = Tensor::arange(ctx, 0.0, Some(nq as f64), 1.0, Some(dtype::default_int())).reshape(ctx, [nq, 1]);
        let cols = Tensor::arange(ctx, 0.0, Some(nk as f64), 1.0, Some(dtype::default_int())).reshape(ctx, [1, nk]);
        let blocked = cols.gt(ctx, &rows); // (nq, nk): j > i
        mask = Some(blocked.where_(ctx, &ninf, &zero));
    }
    if let Some(am) = attn_mask {
        let am = if am.dtype(ctx) == dtype::BOOL { am.where_(ctx, &zero, &ninf) } else { am.clone() };
        mask = Some(match mask {
            None => am,
            Some(m) => m.add(ctx, &am),
        });
    }
    let mut mask = mask.unwrap_or_else(|| Tensor::full_const(ctx, [nq, nk], ConstVal::Float(0.0), None));

    // Zero-pad the mask to the block grid, then overwrite the padding with -inf
    // so padded keys never contribute and padded queries are harmless (dropped
    // by the final slice).
    if nqp > nq || nkp > nk {
        let mnd = mask.shape(ctx).len();
        let mut parg: Vec<(i64, i64)> = vec![(0, 0); mnd];
        parg[mnd - 2] = (0, nqp - nq);
        parg[mnd - 1] = (0, nkp - nk);
        let padded = mask.pad(ctx, &parg);
        let rp = Tensor::arange(ctx, 0.0, Some(nqp as f64), 1.0, Some(dtype::default_int())).reshape(ctx, [nqp, 1]);
        let cp = Tensor::arange(ctx, 0.0, Some(nkp as f64), 1.0, Some(dtype::default_int())).reshape(ctx, [1, nkp]);
        let nkt = scalar_i(ctx, nk);
        let nqt = scalar_i(ctx, nq);
        let t = cp.ge(ctx, &nkt).where_(ctx, &ninf, &padded);
        mask = rp.ge(ctx, &nqt).where_(ctx, &ninf, &t);
    }

    // FlashAttention-2: for each Q block, stream over KV blocks keeping a running
    // max `m`, running denominator `l`, and unnormalized accumulator `acc`.
    let scale = scalar(ctx, 1.0 / (d as f64).sqrt());
    let m_shape = vec![SVal::I(bs as i128), SVal::I(hq as i128), SVal::I(bq as i128)];
    let acc_shape = vec![SVal::I(bs as i128), SVal::I(hq as i128), SVal::I(bq as i128), SVal::I(d as i128)];
    let mut outs: Vec<Tensor> = Vec::with_capacity(nbq as usize);
    for i in 0..nbq {
        let i0 = i * bq;
        let qi = q_pad.getitem(ctx, &[Ix::Ellipsis, Ix::slice(Some(i0), Some(i0 + bq), 1), Ix::All]);
        let mut m = Tensor::full_const(ctx, m_shape.clone(), ConstVal::Float(NEG), None);
        let mut l = Tensor::full_const(ctx, m_shape.clone(), ConstVal::Float(0.0), None);
        let mut acc = Tensor::full_const(ctx, acc_shape.clone(), ConstVal::Float(0.0), None);
        for j in 0..nbk {
            let j0 = j * bk;
            let kj = k_pad.getitem(ctx, &[Ix::Ellipsis, Ix::slice(Some(j0), Some(j0 + bk), 1), Ix::All]);
            let vj = v_pad.getitem(ctx, &[Ix::Ellipsis, Ix::slice(Some(j0), Some(j0 + bk), 1), Ix::All]);
            let kjt = kj.transpose(ctx, -2, -1);
            let s = qi.matmul(ctx, &kjt).mul(ctx, &scale); // (B, H, bq, bk)
            let mb = mask.getitem(ctx, &[Ix::Ellipsis, Ix::slice(Some(i0), Some(i0 + bq), 1), Ix::slice(Some(j0), Some(j0 + bk), 1)]);
            let s = s.add(ctx, &mb);
            let slast = s.shape(ctx).len() - 1;
            let row_max = s.max(ctx, Some(&[slast]), false); // (B, H, bq)
            let m_new = m.maximum(ctx, &row_max);
            let alpha = m.sub(ctx, &m_new).exp(ctx); // (B, H, bq)
            let mcol = m_new.unsqueeze(ctx, -1);
            let p = s.sub(ctx, &mcol).exp(ctx); // (B, H, bq, bk)
            let psum = p.sum(ctx, Some(&[slast]), false, None);
            let alpha_col = alpha.unsqueeze(ctx, -1);
            let l_new = l.mul(ctx, &alpha).add(ctx, &psum);
            let pv = p.matmul(ctx, &vj);
            let acc_new = acc.mul(ctx, &alpha_col).add(ctx, &pv);
            m = m_new;
            l = l_new;
            acc = acc_new;
        }
        let lr = l.reciprocal(ctx).unsqueeze(ctx, -1);
        outs.push(acc.mul(ctx, &lr));
    }

    let out = if outs.len() == 1 {
        outs.pop().unwrap()
    } else {
        let first = outs.remove(0);
        first.cat(ctx, &outs, 2)
    };
    let out = if nqp > nq {
        out.getitem(ctx, &[Ix::Ellipsis, Ix::slice(Some(0), Some(nq), 1), Ix::All])
    } else {
        out
    };
    if promoted {
        out.getitem(ctx, &[Ix::Index(0)])
    } else {
        out
    }
}
