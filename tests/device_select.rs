//! Device-selection tests for the backend priority (CURRENT_STATE.md §9.8):
//! CUDA → METAL → CL → RUST → RAYON → INTERP, and the `RAYON`/`RUST`/`METAL`
//! devices execute correctly.

#[cfg(any(
    feature = "backend_rayon",
    all(feature = "backend_metal", target_os = "macos")
))]
use autorust::tensor::{Ctx, Tensor};

#[test]
fn pick_device_priority() {
    use autorust::tensor::creation::{pick_device, pick_device_with_metal};
    // CUDA beats everything; CL beats RUST/CC/RAYON/INTERP; RUST beats CC.
    assert_eq!(pick_device(true, true, true, true), "CUDA");
    assert_eq!(pick_device(true, false, false, false), "CUDA");
    assert_eq!(pick_device(false, true, true, true), "CL");
    assert_eq!(pick_device(false, false, true, true), "RUST");
    assert_eq!(pick_device(false, false, false, true), "CC");
    // Metal (macOS) outranks OpenCL but not CUDA.
    assert_eq!(pick_device_with_metal(true, true, true, true, true), "CUDA");
    assert_eq!(
        pick_device_with_metal(true, false, true, true, true),
        "METAL"
    );
    assert_eq!(pick_device_with_metal(false, false, true, true, true), "CL");
    let low = pick_device(false, false, false, false);
    if cfg!(feature = "backend_rayon") {
        assert_eq!(low, "RAYON");
    } else {
        assert_eq!(low, "INTERP");
    }
}

#[test]
fn canonical_device_names() {
    use autorust::runtime::device::canonicalize_device_name;
    // v0.14 `Device._canonicalize`: uppercase the prefix, drop a trailing `:0`,
    // keep other indices as part of the name.
    assert_eq!(canonicalize_device_name("cuda"), "CUDA");
    assert_eq!(canonicalize_device_name("cuda:0"), "CUDA");
    assert_eq!(canonicalize_device_name("cuda:1"), "CUDA:1");
    assert_eq!(canonicalize_device_name("cl:0"), "CL");
    assert_eq!(canonicalize_device_name("CL:01"), "CL:01");
}

/// Indexed GPU names resolve to the same composition as the base device;
/// an out-of-range index fails with a clear message instead of silently
/// running the interpreter.
#[cfg(feature = "backend_cl")]
#[test]
fn cl_indexed_device_names() {
    use autorust::runtime::device::device;
    if !autorust::runtime::ops_cl::available() {
        return;
    }
    let count = autorust::runtime::ops_cl::device_count();
    assert!(count >= 1, "{count}");
    assert_eq!(device("CL:0").device, "CL");
    if count == 1 {
        assert!(std::panic::catch_unwind(|| device("CL:1")).is_err());
    } else {
        assert_eq!(device("CL:1").device, "CL:1");
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
#[cfg(all(
    feature = "backend_cuda",
    any(target_os = "linux", target_os = "windows")
))]
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

#[cfg(all(feature = "backend_metal", target_os = "macos"))]
#[test]
fn metal_is_registered() {
    let devs = autorust::runtime::device::registered_devices();
    assert!(devs.contains(&"METAL".to_string()), "registered: {devs:?}");
}

/// The Metal default-device probe must not panic (and is a no-op on non-macOS
/// builds, where the backend module is not compiled).
#[cfg(all(feature = "backend_metal", target_os = "macos"))]
#[test]
fn metal_availability_probe_does_not_panic() {
    let _ = autorust::runtime::ops_metal::available();
}

#[cfg(all(feature = "backend_metal", target_os = "macos"))]
#[test]
fn metal_executes_elementwise() {
    let ctx = Ctx::new();
    let a = Tensor::from_slice(&ctx, &[1.0f32, 2.0, 3.0, 4.0], None, Some("METAL"));
    let out = a.add(&a).mul(2.0).to_list_fresh::<f32>();
    assert_eq!(out, vec![4.0, 8.0, 12.0, 16.0]);
}

#[cfg(all(feature = "backend_metal", target_os = "macos"))]
#[test]
fn metal_matches_interp() {
    let ctx = Ctx::new();
    let data = [0.5f32, -1.0, 2.25, 3.0, 4.0, 5.5, -6.0, 7.75];
    let build = |dev: &str| {
        let x = Tensor::from_slice(&ctx, &data, None, Some(dev));
        x.mul(&x).add(1.0).sqrt().to_list_fresh::<f32>()
    };
    let m = build("METAL");
    let i = build("INTERP");
    assert_eq!(m.len(), i.len());
    for (a, b) in m.iter().zip(i.iter()) {
        assert!(
            (a - b).abs() <= 1e-5 * b.abs().max(1.0),
            "METAL {a} != INTERP {b}"
        );
    }
}

#[cfg(all(feature = "backend_metal", target_os = "macos"))]
#[test]
fn metal_splits_graphs_over_the_buffer_limit() {
    // A single fused kernel would need 64 input buffers, over Metal's 31-slot
    // limit. `Renderer::max_buffers` must make the scheduler realize
    // intermediates so the graph still runs and sums correctly.
    assert_eq!(
        autorust::renderer::Renderer::metal().max_buffers,
        31,
        "Metal buffer-argument limit changed"
    );
    let ctx = Ctx::new();
    let mut acc = Tensor::from_slice(&ctx, &[0.0f32], None, Some("METAL"));
    for i in 0..64 {
        let t = Tensor::from_slice(&ctx, &[i as f32], None, Some("METAL"));
        acc = acc.add(&t);
    }
    let got = acc.to_list_fresh::<f32>();
    assert_eq!(got, vec![(0..64).sum::<i32>() as f32]);
}

#[cfg(all(feature = "backend_metal", target_os = "macos"))]
#[test]
fn metal_allocator_supports_offset_views() {
    // Metal has real (buf, offset, size) views, unlike the CL allocator.
    let dev = autorust::runtime::device::device("METAL");
    let base = dev.allocator.alloc(64);
    dev.copyin_to(&*base, &[0u8; 64]);
    let view = dev.allocator._offset(&*base, 16, 8);
    dev.copyin_to(
        &*view,
        &[1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    assert_eq!(dev.copyout(&*view), (1u8..=16).collect::<Vec<_>>());
    // The bytes landed at offset 8 in the parent allocation.
    let all = dev.copyout(&*base);
    assert_eq!(&all[8..24], &(1u8..=16).collect::<Vec<_>>()[..]);
    assert_eq!(&all[..8], &[0u8; 8]);
    assert_eq!(&all[24..], &[0u8; 40]);
}
