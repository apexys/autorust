//! A device-less virtual result must inherit the operands' device, not the
//! process default (`DEFAULT_DEVICE`). This is the transient-CUDA-probe case:
//! the default resolved to CL/CPU while the graph itself was on CUDA.

#![cfg(feature = "backend_cuda")]

use autorust::engine::session::Session;
use autorust::tensor::{Ctx, Tensor};

#[test]
fn device_less_output_inherits_operands() {
    if !autorust::runtime::ops_cuda::available() {
        return;
    }
    // Set before the first `Ctx::new()` reads the process settings. This binary
    // has one test, so nothing has initialized them yet.
    unsafe { std::env::set_var("DEFAULT_DEVICE", "CPU") };

    let ctx = Ctx::new();
    let mut sess = Session::new();
    let a = Tensor::from_slice(&ctx, &[1.0f32, 2.0, 3.0, 4.0], None, Some("CUDA")).reshape([2, 2]);
    let b = Tensor::from_slice(&ctx, &[5.0f32, 6.0, 7.0, 8.0], None, Some("CUDA")).reshape([2, 2]);
    let y = a.dot(&b);
    let got = y.to_list::<f32>(&mut sess);
    assert_eq!(got, vec![19.0, 22.0, 43.0, 50.0]);
    assert_eq!(
        y.realized_buffer().expect("realized").device(),
        "CUDA",
        "a device-less result must inherit the operands' device"
    );
}
