//! Device-selection tests for the backend priority (CURRENT_STATE.md §9.8):
//! CUDA → CL → RUST → RAYON → INTERP, and the `RAYON`/`RUST` devices execute
//! correctly.

#[cfg(feature = "backend_rayon")]
use autorust::tensor::{Ctx, Tensor};

#[test]
fn pick_device_priority() {
    use autorust::tensor::creation::pick_device;
    // CUDA beats everything; CL beats RUST/CC/RAYON/INTERP; RUST beats CC.
    assert_eq!(pick_device(true, true, true, true), "CUDA");
    assert_eq!(pick_device(true, false, false, false), "CUDA");
    assert_eq!(pick_device(false, true, true, true), "CL");
    assert_eq!(pick_device(false, false, true, true), "RUST");
    assert_eq!(pick_device(false, false, false, true), "CC");
    let low = pick_device(false, false, false, false);
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
    if cfg!(feature = "backend_rust") {
        assert!(devs.contains(&"RUST".to_string()), "registered: {devs:?}");
    }
    if cfg!(feature = "backend_clang") {
        assert!(devs.contains(&"CC".to_string()), "registered: {devs:?}");
    }
    // INTERP is always present.
    assert!(devs.contains(&"INTERP".to_string()), "registered: {devs:?}");
}

/// The default-device probe must be safe on a machine where the CUDA backend
/// was compiled in but the driver library is not installed: cudarc's loader
/// panics on a missing `libcuda`, so `available()` has to return `false`.
#[cfg(feature = "backend_cuda")]
#[test]
fn cuda_availability_probe_does_not_panic() {
    let _ = autorust::runtime::ops_cuda::available();
}

/// Same requirement for OpenCL: the loader is dynamic, and `AUTORUST_CL_LIB`
/// can point at a missing path to exercise the fallback.
#[cfg(feature = "backend_cl")]
#[test]
fn cl_availability_probe_does_not_panic() {
    let _ = autorust::runtime::ops_cl::available();
}

/// `RUST` is only selected when `rustc` is present; the probe must not panic.
#[cfg(feature = "backend_rust")]
#[test]
fn rust_availability_probe_does_not_panic() {
    let _ = autorust::runtime::ops_rust::available();
}

#[cfg(feature = "backend_rust")]
#[test]
fn rust_executes_elementwise() {
    use autorust::tensor::{Ctx, Tensor};
    let ctx = Ctx::new();
    let a = Tensor::from_slice(&ctx, &[1.0f32, 2.0, 3.0, 4.0], None, Some("RUST"));
    let out = a.add(&a).mul(2.0).to_list_fresh::<f32>();
    assert_eq!(out, vec![4.0, 8.0, 12.0, 16.0]);
}

#[cfg(feature = "backend_rust")]
#[test]
fn rust_matches_interp() {
    use autorust::tensor::{Ctx, Tensor};
    let ctx = Ctx::new();
    let data = [0.5f32, -1.0, 2.25, 3.0, 4.0, 5.5, -6.0, 7.75];
    let build = |ctx: &Ctx, dev: &str| {
        let x = Tensor::from_slice(ctx, &data, None, Some(dev));
        x.mul(&x).add(1.0).sqrt().to_list_fresh::<f32>()
    };
    let r = build(&ctx, "RUST");
    let i = build(&ctx, "INTERP");
    assert_eq!(r, i, "RUST must match INTERP");
}

#[cfg(feature = "backend_rayon")]
#[test]
fn rayon_executes_elementwise() {
    let ctx = Ctx::new();
    let a = Tensor::from_slice(&ctx, &[1.0f32, 2.0, 3.0, 4.0], None, Some("RAYON"));
    let out = a.add(&a).mul(2.0).to_list_fresh::<f32>();
    assert_eq!(out, vec![4.0, 8.0, 12.0, 16.0]);
}

#[cfg(feature = "backend_rayon")]
#[test]
fn rayon_owned_fast_path_is_active() {
    use autorust::runtime::ops_rayon::OWNED_KERNELS;
    use std::sync::atomic::Ordering;
    let before = OWNED_KERNELS.load(Ordering::Relaxed);
    let ctx = Ctx::new();
    // A few distinct elementwise kernels.
    let a = Tensor::from_slice(&ctx, &[1.0f32, 2.0, 3.0, 4.0], None, Some("RAYON"));
    let b = Tensor::from_slice(&ctx, &[5.0f32, 6.0, 7.0, 8.0], None, Some("RAYON"));
    let _ = a.add(&b).mul(2.0).to_list_fresh::<f32>();
    let _ = a.sub(&b).sqrt().to_list_fresh::<f32>();
    let after = OWNED_KERNELS.load(Ordering::Relaxed);
    assert!(
        after > before,
        "expected the owned fast path to compile kernels (before={before}, after={after})"
    );
}

#[cfg(feature = "backend_rayon")]
#[test]
fn rayon_matches_interp() {
    let ctx = Ctx::new();
    let data = [0.5f32, -1.0, 2.25, 3.0, 4.0, 5.5, -6.0, 7.75];
    let build = |ctx: &Ctx, dev: &str| {
        let x = Tensor::from_slice(ctx, &data, None, Some(dev));
        x.mul(&x).add(1.0).sqrt().to_list_fresh::<f32>()
    };
    let r = build(&ctx, "RAYON");
    let i = build(&ctx, "INTERP");
    assert_eq!(r, i, "RAYON must match INTERP");
}
