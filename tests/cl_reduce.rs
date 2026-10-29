//! CL backend regression: reductions that lower with OpenCL `__local` workgroup
//! memory must declare a plain array (`__local float buf[N];`), not an array of
//! pointers (`__local float* buf[N];`), or the kernel fails to build
//! (`CL_BUILD_PROGRAM_FAILURE`). Matches tinygrad's `CStyleLanguage.render_buffer`
//! (cstyle.py:169), which renders the element type with the default ALU addrspace
//! and takes the `__local ` prefix only from `smem_prefix`.
#![cfg(feature = "backend_cl")]

use autorust::tensor::{Ctx, Tensor};
use autorust::tensor::nn::{self, Linear};
use autorust::uop::ctx::UOpCtx;

fn skip() -> bool {
    if !autorust::runtime::ops_cl::available() {
        eprintln!("skipping: no OpenCL device");
        true
    } else {
        false
    }
}

#[test]
fn cl_reduce_axis_mean_runs() {
    if skip() {
        return;
    }
    let mut ctx = Ctx::new();
    let data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    let x = Tensor::from_slice(&ctx, &data, None, Some("CL")).reshape([4, 4]);
    let m = x.mean(Some(&[1]), false);
    let got = m.to_list_fresh::<f32>();
    assert_eq!(got.len(), 4, "rows");
    for (i, g) in got.iter().enumerate() {
        let lo = (i * 4 + 1) as f32;
        let want = (0..4).map(|k| lo + k as f32).sum::<f32>() / 4.0;
        assert!((g - want).abs() < 1e-5, "row {i}: got {g}, want {want}");
    }
}

/// Forward+backward through a small ReLU MLP. The backward/optimizer graph lowers
/// reductions that use `__local` workgroup memory (`buf1[N]`), the exact shape
/// that failed to build before the `render_buffer` fix. Values are not asserted
/// here (the autograd path is exercised by other gates); this is a build/run
/// smoke test for the CL reduction kernels.
#[test]
fn cl_xor_backward_kernels_build_and_run() {
    if skip() {
        return;
    }
    let mut ctx = Ctx::new();
    let l1 = Linear::new(&ctx, 2, 4, true);
    let l2 = Linear::new(&ctx, 4, 1, true);
    let x = Tensor::from_slice(&ctx, &[0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], None, Some("CL")).reshape([4, 2]);
    let t = Tensor::from_slice(&ctx, &[0.0f32, 1.0, 1.0, 0.0], None, Some("CL")).reshape([4, 1]);
    let h = nn::relu(&l1.forward(&x));
    let out = l2.forward(&h);
    let diff = out.sub(&t);
    let loss = diff.mul(&diff).mean(None, false);
    let params = vec![l1.weight.clone(), l1.bias.clone().unwrap(), l2.weight.clone(), l2.bias.clone().unwrap()];
    loss.backward(&params, None);
    assert_eq!(out.device().as_deref(), Some("CL"), "forward must run on CL");
    let g = params[0].grad().unwrap().to_list_fresh::<f32>();
    assert!(g.iter().all(|v| v.is_finite()), "CL backward produced non-finite grads: {g:?}");
}
