//! Library GEMM (`CLBlast`) vs the generated OpenCL kernels.
//!
//! Mirrors `tests/cublas.rs`: force the library (`AUTORUST_GEMM=all`) and compare
//! against generated kernels on the layouts DINOv3 produces (prime M, transposed
//! weight, scale+residual, strided-batched attention).

#![cfg(feature = "backend_cl")]

use autorust::tensor::nn::Linear;
use autorust::tensor::{Ctx, Tensor};

static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn have_cl() -> bool {
    if !autorust::runtime::ops_cl::available() || !autorust::runtime::clblast::available() {
        return false;
    }
    unsafe { std::env::set_var("DEFAULT_DEVICE", "CL") };
    true
}

fn set_lib(on: bool) {
    unsafe { std::env::set_var("AUTORUST_GEMM", if on { "all" } else { "0" }) };
}

fn values(seed: usize, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i + seed) % 23) as f32) * 0.01 - 0.11)
        .collect()
}

fn run<F>(f: F) -> Vec<f32>
where
    F: Fn(&Ctx) -> Vec<Tensor>,
{
    let ctx = Ctx::new();
    let outs = f(&ctx);
    let mut out = Vec::new();
    for t in outs {
        out.extend(t.to_list_fresh::<f32>());
    }
    out
}

fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn clblast_matches_generated_gemm() {
    let _guard = LOCK.lock().unwrap();
    if !have_cl() {
        return;
    }
    for (m, n, k) in [(197usize, 768, 768), (64, 64, 64), (33, 17, 65)] {
        let ad = values(0, m * k);
        let bd = values(7, k * n);
        let run_case = |lib: bool| {
            set_lib(lib);
            run(|ctx| {
                let a =
                    Tensor::from_slice(ctx, &ad, None, Some("CL")).reshape([m as i64, k as i64]);
                let b =
                    Tensor::from_slice(ctx, &bd, None, Some("CL")).reshape([k as i64, n as i64]);
                vec![a.dot(&b)]
            })
        };
        let want = run_case(false);
        let got = run_case(true);
        let d = maxdiff(&want, &got);
        assert!(d < 1e-3, "{m}x{n}x{k}: CLBlast diff {d:e}");
    }
}

#[test]
fn clblast_matches_transposed_weight_with_bias() {
    let _guard = LOCK.lock().unwrap();
    if !have_cl() {
        return;
    }
    let (m, k, n) = (197usize, 768usize, 384usize);
    let ad = values(1, m * k);
    let wd = values(3, n * k);
    let bd = values(5, n);
    let run_case = |lib: bool| {
        set_lib(lib);
        run(|ctx| {
            let a = Tensor::from_slice(ctx, &ad, None, Some("CL")).reshape([m as i64, k as i64]);
            let lin = Linear {
                weight: Tensor::from_slice(ctx, &wd, None, Some("CL"))
                    .reshape([n as i64, k as i64]),
                bias: Some(Tensor::from_slice(ctx, &bd, None, Some("CL"))),
            };
            vec![lin.forward(&a)]
        })
    };
    let d = maxdiff(&run_case(false), &run_case(true));
    assert!(d < 1e-3, "transposed weight: CLBlast diff {d:e}");
}

#[test]
fn clblast_matches_scale_and_residual_epilogue() {
    let _guard = LOCK.lock().unwrap();
    if !have_cl() {
        return;
    }
    let (m, k, n) = (197usize, 384usize, 384usize);
    let hd = values(2, m * k);
    let wd = values(4, n * k);
    let bd = values(6, n);
    let sd = values(8, n);
    let xd = values(11, m * n);
    let run_case = |lib: bool| {
        set_lib(lib);
        run(|ctx| {
            let h = Tensor::from_slice(ctx, &hd, None, Some("CL")).reshape([m as i64, k as i64]);
            let x = Tensor::from_slice(ctx, &xd, None, Some("CL")).reshape([m as i64, n as i64]);
            let lin = Linear {
                weight: Tensor::from_slice(ctx, &wd, None, Some("CL"))
                    .reshape([n as i64, k as i64]),
                bias: Some(Tensor::from_slice(ctx, &bd, None, Some("CL"))),
            };
            let scale = Tensor::from_slice(ctx, &sd, None, Some("CL"));
            vec![x.add(&lin.forward(&h).mul(&scale))]
        })
    };
    let d = maxdiff(&run_case(false), &run_case(true));
    assert!(d < 1e-3, "scale+residual: CLBlast diff {d:e}");
}

#[test]
fn clblast_matches_strided_batched_attention() {
    let _guard = LOCK.lock().unwrap();
    if !have_cl() {
        return;
    }
    let (b, h, m, k, n) = (1usize, 3usize, 37usize, 16usize, 37usize);
    let qd = values(2, b * h * m * k);
    let kd = values(9, b * h * n * k);
    let run_case = |lib: bool| {
        set_lib(lib);
        run(|ctx| {
            let q = Tensor::from_slice(ctx, &qd, None, Some("CL"))
                .reshape([b as i64, m as i64, h as i64, k as i64]);
            let kk = Tensor::from_slice(ctx, &kd, None, Some("CL"))
                .reshape([b as i64, n as i64, h as i64, k as i64]);
            let q = q.transpose(1, 2);
            let kt = kk.transpose(1, 2).transpose(2, 3);
            vec![q.dot(&kt)]
        })
    };
    let d = maxdiff(&run_case(false), &run_case(true));
    assert!(d < 1e-3, "batched attention: CLBlast diff {d:e}");
}
