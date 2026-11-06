//! BF16 through the full schedule/codegen/backend path.
//!
//! bf16 storage/ALU on every registered backend must match the interpreter. On
//! the cstyle/CL backends this exercises the renderer `extra_matcher`
//! (`create_non_native_float_pats` + `pm_manual_bf16_cast`, renderer/extra.rs),
//! the OpenCL `as_*` BITCAST rendering, and the OpenCL bf16-const bit pattern.

use autorust::dtype;
use autorust::engine::session::Session;
use autorust::runtime::device::registered_devices;
use autorust::tensor::Ctx;
use autorust::tensor::optim::{Adam, Optim};
use autorust::tensor::{Tensor, TensorAddArg};

/// bf16 multiply + add + comparison/where, read back through an f32 cast.
/// The inputs (0.1, 0.2) are not exactly bf16, so the cast rounding is real.
fn program(dev: Option<&str>) -> Vec<f32> {
    let ctx = Ctx::new();
    let a = Tensor::from_slice(
        &ctx,
        &[1.5f32, -2.25, 100.0, 0.5, 0.1],
        Some(dtype::FLOAT32),
        dev,
    )
    .cast(dtype::BFLOAT16);
    let b = Tensor::from_slice(
        &ctx,
        &[0.5f32, 4.0, 100.0, 0.25, 0.2],
        Some(dtype::FLOAT32),
        dev,
    )
    .cast(dtype::BFLOAT16);
    let m = a.mul(&b);
    let s = m.add(&b);
    // where(s > 0, s, -s) — drives a bf16 comparison + where through the
    // non-native patterns (compare operands in f32, arms in f32).
    let zero = Tensor::from_slice(&ctx, &[0.0f32], Some(dtype::BFLOAT16), dev);
    let cond = s.gt(&zero);
    let neg = s.mul(-1.0);
    let out = cond.where_(&s, &neg);
    out.cast(dtype::FLOAT32).to_list_fresh::<f32>()
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

/// A bf16 matmul with an f32 accumulator must match the hardware semantics:
/// inputs are rounded to bf16 once, then multiplied/accumulated in f32 (not
/// each product rounded back to bf16). This is the `try_matmul_tiling` promote
/// path and also exercises the vectorized bf16 load coalescing.
#[test]
fn bf16_matmul_accumulates_in_f32() {
    fn bf16_round(x: f32) -> f32 {
        let bits = x.to_bits();
        let round = 0x7fff + ((bits >> 16) & 1);
        f32::from_bits((bits.wrapping_add(round)) & 0xffff_0000)
    }
    let (m, n, k) = (64usize, 64usize, 64usize);
    let ad: Vec<f32> = (0..m * k).map(|i| ((i % 19) as f32) * 0.05 - 0.4).collect();
    let bd: Vec<f32> = (0..k * n).map(|i| ((i % 17) as f32) * 0.04 - 0.3).collect();
    let want: Vec<f32> = (0..m * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            (0..k)
                .map(|p| bf16_round(ad[i * k + p]) * bf16_round(bd[p * n + j]))
                .sum()
        })
        .collect();
    for dev in registered_devices() {
        let ctx = Ctx::new();
        let a = Tensor::from_slice(&ctx, &ad, Some(dtype::BFLOAT16), Some(&dev)).reshape([64, 64]);
        let b = Tensor::from_slice(&ctx, &bd, Some(dtype::BFLOAT16), Some(&dev)).reshape([64, 64]);
        let got = a.dot_dtype(&b, Some(dtype::FLOAT32)).to_list_fresh::<f32>();
        let maxd = want
            .iter()
            .zip(&got)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(maxd < 2e-4, "bf16 matmul mismatch on {dev}: max {maxd:e}");
    }
}

/// Adam with a bf16 parameter must stay finite. Regression for the LAMB
/// bias-correction scalars (`b1_t`/`b2_t`): tinygrad pins them to f32 regardless
/// of the param dtype (optim.py:157) because in bf16 `0.999` rounds to `1.0`, so
/// `1 - b2_t == 0` and the update divides by zero → NaN. Uses an explicit bf16
/// param so it is independent of the global default float.
#[test]
fn adam_step_with_bf16_param_is_finite() {
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let p =
        Tensor::from_slice(&ctx, &[1.0f32, -2.0, 0.5, 3.0], Some(dtype::BFLOAT16), None).to_param();
    let g = Tensor::from_slice(
        &ctx,
        &[0.5f32, 0.25, -0.75, 1.0],
        Some(dtype::BFLOAT16),
        None,
    );
    ctx.raw().set_grad(&p.uop(), g.uop().clone());
    let opt = Adam::new(vec![p.clone()], 0.001);
    for _ in 0..3 {
        opt.step(&mut sess);
    }
    let got = p.cast(dtype::FLOAT32).to_list::<f32>(&mut sess);
    for (i, v) in got.iter().enumerate() {
        assert!(
            v.is_finite(),
            "bf16 Adam produced non-finite param[{i}] = {v}"
        );
    }
}

/// `addrspace_derived` must stay memoized on shared DAGs. A doubling chain
/// `x_{k+1} = x_k + x_k` has `2^k` distinct paths; tinygrad's `@recursive_property`
/// walks it linearly (ops.py:217-227), so depth 40 is instant. If the per-node
/// cache is removed this test effectively hangs (2^40 walks).
#[test]
fn addrspace_derived_memoizes_shared_subgraphs() {
    let ctx = Ctx::new();
    let mut x = Tensor::from_slice(&ctx, &[1.0f32], None, None);
    for _ in 0..40 {
        x = x.add(TensorAddArg::T(x.clone()));
    }
    let _ = ctx.raw().addrspace_derived(x.uop().clone());
}
