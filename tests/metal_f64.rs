//! f64 on Metal: there is no hardware `double`. The public Tensor API lowers
//! `FLOAT64` to `FLOAT32` on METAL, so f64 graphs compile and run with f32
//! precision. This pins the fallback and the values.

#![cfg(all(feature = "backend_metal", target_os = "macos"))]

use autorust::dtype;
use autorust::tensor::{Ctx, Tensor};

#[test]
fn metal_f64_compiles_and_runs() {
    let ctx = Ctx::new();
    let a = Tensor::from_slice(
        &ctx,
        &[1.5f64, -2.25, 0.0, 4.0, 10.5, -0.5],
        Some(dtype::FLOAT64),
        Some("METAL"),
    );
    let b = Tensor::from_slice(
        &ctx,
        &[0.5f64, 2.0, 3.0, -1.0, 2.0, 8.0],
        Some(dtype::FLOAT64),
        Some("METAL"),
    );
    // `from_slice(FLOAT64)` is lowered to a native f32 tensor on METAL.
    assert_eq!(a.dtype(), dtype::FLOAT32);
    assert_eq!(b.dtype(), dtype::FLOAT32);
    let y = a.add(&b).mul(2.0);
    let got = y.to_list_fresh::<f64>();
    assert_eq!(got, vec![4.0, -0.5, 6.0, 6.0, 25.0, 15.0]);

    // A reduction and a cast through f32 must compile too.
    let s = a.sum(None, false, None).to_list_fresh::<f64>();
    assert!((s[0] - 13.25).abs() < 1e-5, "sum {s:?}");
    // An explicit f64 cast is lowered as well.
    let f = a.cast(dtype::FLOAT64);
    assert_eq!(f.dtype(), dtype::FLOAT32);
    assert_eq!(
        f.to_list_fresh::<f32>(),
        vec![1.5, -2.25, 0.0, 4.0, 10.5, -0.5]
    );
}
