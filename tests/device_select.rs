//! Device-selection tests for the backend priority (CURRENT_STATE.md §9.8):
//! CUDA → CL → RAYON → INTERP, and the `RAYON` device executes correctly.

use autorust::tensor::Tensor;
use autorust::uop::ctx::UOpCtx;

#[test]
fn pick_device_priority() {
    use autorust::tensor::creation::pick_device;
    // CUDA beats everything, CL beats RAYON/INTERP.
    assert_eq!(pick_device(true, true), "CUDA");
    assert_eq!(pick_device(true, false), "CUDA");
    assert_eq!(pick_device(false, true), "CL");
    let low = pick_device(false, false);
    if cfg!(feature = "backend_rayon") {
        assert_eq!(low, "RAYON");
    } else {
        assert_eq!(low, "INTERP");
    }
}

#[test]
fn rayon_is_registered() {
    let devs = autorust::runtime::device::registered_devices();
    if cfg!(feature = "backend_rayon") {
        assert!(devs.contains(&"RAYON".to_string()), "registered: {devs:?}");
    }
    // INTERP is always present.
    assert!(devs.contains(&"INTERP".to_string()), "registered: {devs:?}");
}

#[cfg(feature = "backend_rayon")]
#[test]
fn rayon_executes_elementwise() {
    let mut ctx = UOpCtx::new();
    let a = Tensor::from_slice(&mut ctx, &[1.0f32, 2.0, 3.0, 4.0], None, Some("RAYON"));
    let out = a.add(&mut ctx, &a).mul(&mut ctx, 2.0).to_list_fresh::<f32>(&mut ctx);
    assert_eq!(out, vec![4.0, 8.0, 12.0, 16.0]);
}

#[cfg(feature = "backend_rayon")]
#[test]
fn rayon_owned_fast_path_is_active() {
    use std::sync::atomic::Ordering;
    use autorust::runtime::ops_rayon::OWNED_KERNELS;
    let before = OWNED_KERNELS.load(Ordering::Relaxed);
    let mut ctx = UOpCtx::new();
    // A few distinct elementwise kernels.
    let a = Tensor::from_slice(&mut ctx, &[1.0f32, 2.0, 3.0, 4.0], None, Some("RAYON"));
    let b = Tensor::from_slice(&mut ctx, &[5.0f32, 6.0, 7.0, 8.0], None, Some("RAYON"));
    let _ = a.add(&mut ctx, &b).mul(&mut ctx, 2.0).to_list_fresh::<f32>(&mut ctx);
    let _ = a.sub(&mut ctx, &b).sqrt(&mut ctx).to_list_fresh::<f32>(&mut ctx);
    let after = OWNED_KERNELS.load(Ordering::Relaxed);
    assert!(after > before, "expected the owned fast path to compile kernels (before={before}, after={after})");
}

#[cfg(feature = "backend_rayon")]
#[test]
fn rayon_matches_interp() {
    let mut ctx = UOpCtx::new();
    let data = [0.5f32, -1.0, 2.25, 3.0, 4.0, 5.5, -6.0, 7.75];
    let build = |ctx: &mut UOpCtx, dev: &str| {
        let x = Tensor::from_slice(ctx, &data, None, Some(dev));
        x.mul(ctx, &x).add(ctx, 1.0).sqrt(ctx).to_list_fresh::<f32>(ctx)
    };
    let r = build(&mut ctx, "RAYON");
    let i = build(&mut ctx, "INTERP");
    assert_eq!(r, i, "RAYON must match INTERP");
}
