//! Differential tests for the fused FA2 training path: forward with saved LSE,
//! then dQ/dK/dV, against the graph SDPA's autograd gradients.

#![cfg(all(
    feature = "backend_cuda",
    any(target_os = "linux", target_os = "windows")
))]

use autorust::dtype;
use autorust::engine::session::Session;
use autorust::models::flash_attention::{fused_attention_backward, fused_attention_train};
use autorust::tensor::{Ctx, Tensor};

fn cuda() -> bool {
    autorust::runtime::ops_cuda::available()
}

fn data(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.0137 + seed).sin() * 0.6)
        .collect()
}

fn tensor(ctx: &Ctx, shape: [i64; 4], seed: f32) -> Tensor {
    let n: usize = shape.iter().map(|d| *d as usize).product();
    Tensor::from_slice(ctx, &data(n, seed), Some(dtype::FLOAT32), Some("CUDA"))
        .reshape(shape)
        .to_param()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[allow(clippy::too_many_arguments)]
fn check(b: i64, h: i64, hk: i64, n: i64, causal: bool, with_mask: bool, gqa: bool, tag: &str) {
    let d = 64i64;
    let ctx = Ctx::new();
    let mut sess = Session::new();

    // The fused path reads the same data as the reference, so give each side its
    // own tensors (the fused forward realizes its inputs).
    let qf = tensor(&ctx, [b, h, n, d], 0.0);
    let kf = tensor(&ctx, [b, hk, n, d], 1.3);
    let vf = tensor(&ctx, [b, hk, n, d], 2.7);
    let dof = tensor(&ctx, [b, h, n, d], 4.1);
    let qr = tensor(&ctx, [b, h, n, d], 0.0);
    let kr = tensor(&ctx, [b, hk, n, d], 1.3);
    let vr = tensor(&ctx, [b, hk, n, d], 2.7);
    let dor = tensor(&ctx, [b, h, n, d], 4.1);
    let maskf = if with_mask {
        Some(
            Tensor::from_slice(&ctx, &data((n * n) as usize, 5.3), None, Some("CUDA"))
                .reshape([n, n])
                .to_param(),
        )
    } else {
        None
    };
    let maskr = maskf.clone();

    // Reference: SDPA forward, then autograd with `dO` as the incoming gradient.
    let out_ref = qr.scaled_dot_product_attention(&kr, &vr, maskr.as_ref(), 0.0, causal, gqa);
    let loss = out_ref.mul(&dor).sum(None, false, None);
    let grads = loss.gradient(&[qr.clone(), kr.clone(), vr.clone()], None);
    let want_out = out_ref.to_list::<f32>(&mut sess);
    let want_dq = grads[0].to_list::<f32>(&mut sess);
    let want_dk = grads[1].to_list::<f32>(&mut sess);
    let want_dv = grads[2].to_list::<f32>(&mut sess);

    // Fused forward + backward.
    let (o, lse) = fused_attention_train(&qf, &kf, &vf, maskf.as_ref(), causal, &mut sess)
        .unwrap_or_else(|| panic!("{tag}: fused forward unavailable"));
    let got_out = o.to_list::<f32>(&mut sess);
    let fd = max_diff(&want_out, &got_out);
    assert!(fd < 2e-3, "{tag}: forward diff {fd:e}");

    let (dq, dk, dv) = fused_attention_backward(
        &qf,
        &kf,
        &vf,
        &o,
        &lse,
        &dof,
        maskf.as_ref(),
        causal,
        &mut sess,
    )
    .unwrap_or_else(|| panic!("{tag}: fused backward unavailable"));
    let got_dq = dq.to_list::<f32>(&mut sess);
    let got_dk = dk.to_list::<f32>(&mut sess);
    let got_dv = dv.to_list::<f32>(&mut sess);
    let (dq_d, dk_d, dv_d) = (
        max_diff(&want_dq, &got_dq),
        max_diff(&want_dk, &got_dk),
        max_diff(&want_dv, &got_dv),
    );
    assert!(dq_d < 2e-3, "{tag}: dQ diff {dq_d:e}");
    assert!(dk_d < 2e-3, "{tag}: dK diff {dk_d:e}");
    assert!(dv_d < 2e-3, "{tag}: dV diff {dv_d:e}");
}

#[test]
fn fused_backward_matches_sdpa() {
    if !cuda() {
        return;
    }
    // Odd sequence length, no mask.
    check(2, 4, 4, 37, false, false, false, "plain/37");
    // ViT sequence length.
    check(1, 2, 2, 197, false, false, false, "plain/197");
    // Causal.
    check(2, 4, 4, 37, true, false, false, "causal/37");
    // Additive mask.
    check(2, 4, 4, 37, false, true, false, "mask/37");
    // GQA: 4 query heads over 2 K/V heads.
    check(2, 4, 2, 37, false, false, true, "gqa/37");
    check(2, 4, 2, 37, true, false, true, "gqa-causal/37");
    check(2, 4, 1, 37, false, true, true, "gqa-mask/37");
}
