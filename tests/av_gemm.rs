//! Attention `P @ V` must reach the CPU `HostGemm` path.
//!
//! The blocker used to be that the scheduler kept `softmax(QK^T)` fused into
//! the AV store, so the contraction had no operand buffer to describe and fell
//! back to the generated kernel. Materializing the score matrix (what
//! `scaled_dot_product_attention_sess` does) makes it dispatch.

#![cfg(feature = "backend_rust")]

use std::sync::atomic::Ordering;

use autorust::runtime::cpu_gemm::DISPATCHES;
use autorust::tensor::{Ctx, Tensor};

fn fill(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i + seed) % 23) as f32) * 0.01 - 0.11)
        .collect()
}

#[test]
fn av_dispatches() {
    unsafe { std::env::set_var("DEFAULT_DEVICE", "RUST") };
    let ctx = Ctx::new();
    let (b, h, t, d) = (1i64, 6i64, 789i64, 64i64);
    let pd = fill((b * h * t * t) as usize, 3);
    let vd = fill((b * h * t * d) as usize, 7);
    let p = Tensor::from_slice(&ctx, &pd, None, Some("RUST")).reshape([b, h, t, t]);
    let v = Tensor::from_slice(&ctx, &vd, None, Some("RUST")).reshape([b, h, t, d]);

    let sm = autorust::tensor::nn::softmax(&p, Some(3));
    let mut sess = autorust::engine::session::Session::new();
    // Without this the graph stays lazy and softmax fuses into the AV store.
    {
        let mut c = ctx.borrow_mut();
        sm.realize_device_ctx(&mut c, &mut sess);
    }

    let before = DISPATCHES.load(Ordering::Relaxed);
    let attn = sm.dot(&v);
    let out = attn.to_list::<f32>(&mut sess);
    let after = DISPATCHES.load(Ordering::Relaxed);

    assert_eq!(out.len(), (b * h * t * d) as usize);
    assert!(
        after > before,
        "P@V must dispatch as a HostGemm once the softmax output is materialized"
    );

    // `attn[b,h,t,:] = sum_k softmax(p[b,h,t,:])[k] * v[b,h,k,:]`.
    let (bi, hi, ti, di) = (0usize, 3usize, 7usize, 5usize);
    let tt = t as usize;
    let row = &pd[((bi * h as usize + hi) * tt + ti) * tt..][..tt];
    let max = row.iter().copied().fold(f32::MIN, f32::max);
    let ex: Vec<f64> = row.iter().map(|x| ((x - max) as f64).exp()).collect();
    let den: f64 = ex.iter().sum();
    let want: f64 = (0..tt)
        .map(|k| {
            let vv = vd[((bi * h as usize + hi) * tt + k) * d as usize + di] as f64;
            ex[k] / den * vv
        })
        .sum();
    let got = out[((bi * h as usize + hi) * tt + ti) * d as usize + di] as f64;
    assert!((want - got).abs() < 1e-4, "want {want} got {got}");
}
