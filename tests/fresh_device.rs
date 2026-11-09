//! `*_fresh` reads must use the tensor's device, and an already realized tensor
//! must only copy (never re-schedule and lose the params' data).

#![cfg(feature = "backend_cuda")]

use autorust::engine::session::Session;
use autorust::tensor::{Ctx, Tensor};

#[test]
fn realize_fresh_uses_selected_device() {
    if !autorust::runtime::ops_cuda::available() {
        return;
    }
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let a = Tensor::from_slice(&ctx, &[1.0f32, 2.0, 3.0, 4.0], None, Some("CUDA"))
        .reshape([2, 2])
        .to_param();
    let b = Tensor::from_slice(&ctx, &[5.0f32, 6.0, 7.0, 8.0], None, Some("CUDA"))
        .reshape([2, 2])
        .to_param();

    // Self-contained graph: the fresh session inherits CUDA and schedules
    // there (not on the CPU fallback).
    let y = a.dot(&b);
    let got = y.to_list_fresh::<f32>();
    assert_eq!(got, vec![19.0, 22.0, 43.0, 50.0]);

    // Already realized: the fresh path must only copy the device buffer.
    let want = y.to_list::<f32>(&mut sess);
    assert_eq!(want, vec![19.0, 22.0, 43.0, 50.0]);
    let got2 = y.to_list_fresh::<f32>();
    assert_eq!(got2, want, "fresh read after realize must only copy");
}
