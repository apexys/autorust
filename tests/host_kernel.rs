//! The hand-written-kernel seam (`Compiled::run_host_kernel`).
//!
//! A fused kernel like flash attention needs running per-thread state that the
//! uop IR cannot carry, so the device has to compile and launch source that
//! never went through the scheduler. This test proves that path end to end:
//! NVRTC compiles the source, the runtime binds the buffers by slot, the launch
//! parameters are honoured, and the result lands in the output buffer.

#![cfg(feature = "backend_cuda")]

use autorust::dtype;
use autorust::runtime::device::device;
use autorust::tensor::{Ctx, Tensor};

const SRC: &str = r#"
extern "C" __global__ void ar_scale(const float* __restrict__ a,
                                    float* __restrict__ b,
                                    int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = a[i] * (float)k + 1.0f;
}
"#;

#[test]
fn host_kernel_launches() {
    if !autorust::runtime::ops_cuda::available() {
        return;
    }
    let ctx = Ctx::new();
    let mut sess = autorust::engine::session::Session::new();
    let a = Tensor::from_slice(&ctx, &[1.0f32, 2.0, 3.0, 4.0], None, Some("CUDA"));
    let b = Tensor::from_slice(&ctx, &[0.0f32; 4], None, Some("CUDA")).to_param();
    let _ = a.to_list::<f32>(&mut sess);
    let _ = b.to_list::<f32>(&mut sess);

    let dev = device("CUDA");
    let av = a.realized_buffer().expect("a realized").get_buf(&dev);
    let bv = b.realized_buffer().expect("b realized").get_buf(&dev);
    dev.run_host_kernel(
        SRC,
        "ar_scale",
        &[
            (0, dtype::FLOAT32),
            (1, dtype::FLOAT32),
            (2, dtype::INT32),
            (3, dtype::INT32),
        ],
        &[av, bv],
        [4, 1, 1],
        None,
        &[4, 3],
        true,
    )
    .expect("launch");

    let out = b.to_list::<f32>(&mut sess);
    assert_eq!(out, vec![4.0, 7.0, 10.0, 13.0]);
}
