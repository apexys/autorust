//! BF16 through the full schedule/codegen/backend path.
//!
//! bf16 storage/ALU on every registered backend must match the interpreter. On
//! the cstyle/CL backends this exercises the renderer `extra_matcher`
//! (`create_non_native_float_pats` + `pm_manual_bf16_cast`, renderer/extra.rs),
//! the OpenCL `as_*` BITCAST rendering, and the OpenCL bf16-const bit pattern.

use autorust::dtype;
use autorust::runtime::device::registered_devices;
use autorust::tensor::Tensor;
use autorust::uop::ctx::UOpCtx;

/// bf16 multiply + add + comparison/where, read back through an f32 cast.
/// The inputs (0.1, 0.2) are not exactly bf16, so the cast rounding is real.
fn program(dev: Option<&str>) -> Vec<f32> {
    let mut ctx = UOpCtx::new();
    let a = Tensor::from_slice(&mut ctx, &[1.5f32, -2.25, 100.0, 0.5, 0.1], Some(dtype::FLOAT32), dev)
        .cast(&mut ctx, dtype::BFLOAT16);
    let b = Tensor::from_slice(&mut ctx, &[0.5f32, 4.0, 100.0, 0.25, 0.2], Some(dtype::FLOAT32), dev)
        .cast(&mut ctx, dtype::BFLOAT16);
    let m = a.mul(&mut ctx, &b);
    let s = m.add(&mut ctx, &b);
    // where(s > 0, s, -s) — drives a bf16 comparison + where through the
    // non-native patterns (compare operands in f32, arms in f32).
    let zero = Tensor::from_slice(&mut ctx, &[0.0f32], Some(dtype::BFLOAT16), dev);
    let cond = s.gt(&mut ctx, &zero);
    let neg = s.mul(&mut ctx, -1.0);
    let out = cond.where_(&mut ctx, &s, &neg);
    out.cast(&mut ctx, dtype::FLOAT32).to_list_fresh::<f32>(&mut ctx)
}

#[test]
fn bf16_matches_interp_on_all_backends() {
    let want = program(None);
    assert_eq!(want.len(), 5);
    for dev in registered_devices() {
        if dev == "INTERP" || dev == "CPU" {
            continue;
        }
        let got = program(Some(&dev));
        assert_eq!(got, want, "bf16 mismatch on {dev}: {got:?} vs {want:?}");
    }
}
