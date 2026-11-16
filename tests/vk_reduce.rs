//! Fused Vulkan full reductions (`sum_with` / `max_with`) vs the graph.

#![cfg(feature = "backend_vk")]

use autorust::engine::session::Session;
use autorust::tensor::{Ctx, Tensor};

fn values(seed: usize, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i + seed) % 31) as f32) * 0.07 - 0.9)
        .collect()
}

fn check(n: usize) {
    if !autorust::runtime::ops_vk::available() {
        return;
    }
    let ctx = Ctx::new();
    let data = values(5, n);
    let x = Tensor::from_slice(&ctx, &data, None, Some("VK")).to_param();
    let mut s1 = Session::new();
    x.realize(&mut s1);
    let gsum = x.sum(None, false, None).to_list::<f32>(&mut s1);
    let gmax = x.max(None, false).to_list::<f32>(&mut s1);
    let mut s2 = Session::new();
    let fsum = x
        .sum_with(None, false, None, &mut s2)
        .to_list::<f32>(&mut s2);
    let fmax = x.max_with(None, false, &mut s2).to_list::<f32>(&mut s2);
    let rel = |a: f32, b: f32| (a - b).abs() / a.abs().max(1.0);
    assert!(rel(gsum[0], fsum[0]) < 2e-4, "sum {gsum:?} vs {fsum:?}");
    assert!((gmax[0] - fmax[0]).abs() < 1e-3, "max {gmax:?} vs {fmax:?}");
}

#[test]
fn fused_vk_reductions_match_graph() {
    check(1 << 20);
    check((1 << 20) + 12345);
}

#[test]
fn fused_vk_reductions_fall_back_for_axis() {
    if !autorust::runtime::ops_vk::available() {
        return;
    }
    let ctx = Ctx::new();
    let data = values(2, 1024 * 1024);
    let x = Tensor::from_slice(&ctx, &data, None, Some("VK"))
        .reshape([1024, 1024])
        .to_param();
    let mut s1 = Session::new();
    let g = x.sum(Some(&[1]), false, None).to_list::<f32>(&mut s1);
    let mut s2 = Session::new();
    let f = x
        .sum_with(Some(&[1]), false, None, &mut s2)
        .to_list::<f32>(&mut s2);
    assert_eq!(g, f);
}
