//! autorust side of the cross-framework benchmark.
//!
//! Three workloads, selected with `BENCH_WORKLOAD`:
//! - `mnist_train`: the `beautiful_mnist` CNN, Adam, synthetic device-resident
//!   data, batch 512.
//! - `vit_infer`: ViT-B/16 forward (12 layers, 768 wide, 12 heads, 224 px).
//! - `vit_train`: ViT-B/16 forward + cross-entropy + backward + Adam.
//!
//! The device comes from `DEFAULT_DEVICE` (set by `bench/run.py`). `BENCH_BS`,
//! `BENCH_WARMUP`, `BENCH_ITERS`, `BENCH_JIT` and `BENCH_OUT` override the
//! defaults. The final line on stdout is `RESULT <json>`.

use std::time::Instant;

use autorust::dtype;
use autorust::engine::jit::Jit;
use autorust::engine::session::Session;
use autorust::models::vit::ViT;
use autorust::tensor::ix::Ix;
use autorust::tensor::nn::{self, BatchNorm, Conv2d, Linear, Module};
use autorust::tensor::optim::{Adam, Optim};
use autorust::tensor::{Ctx, Tensor};

fn env_i64(key: &str, default: i64) -> i64 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// `beautiful_mnist.py`'s model, exactly.
struct Mnist {
    l0: Conv2d,
    l1: Conv2d,
    l2: BatchNorm,
    l3: Conv2d,
    l4: Conv2d,
    l5: BatchNorm,
    l6: Linear,
}

impl Mnist {
    fn new(ctx: &Ctx) -> Self {
        Mnist {
            l0: Conv2d::new(ctx, 1, 32, &[5, 5], 1, &[(0, 0), (0, 0)], 1, 1, true),
            l1: Conv2d::new(ctx, 32, 32, &[5, 5], 1, &[(0, 0), (0, 0)], 1, 1, true),
            l2: BatchNorm::new(ctx, 32, 1e-5, true, true, 0.1),
            l3: Conv2d::new(ctx, 32, 64, &[3, 3], 1, &[(0, 0), (0, 0)], 1, 1, true),
            l4: Conv2d::new(ctx, 64, 64, &[3, 3], 1, &[(0, 0), (0, 0)], 1, 1, true),
            l5: BatchNorm::new(ctx, 64, 1e-5, true, true, 0.1),
            l6: Linear::new(ctx, 576, 10, true),
        }
    }

    fn forward(&self, x: &Tensor, sess: &mut Session, training: bool) -> Tensor {
        let x = nn::relu(&self.l0.forward(x));
        let x = nn::relu(&self.l1.forward(&x));
        let x = self.l2.forward(&x, sess, training);
        let (x, _) = x.max_pool2d(&[2, 2], None, &[1, 1], &[0], false, false);
        let x = nn::relu(&self.l3.forward(&x));
        let x = nn::relu(&self.l4.forward(&x));
        let x = self.l5.forward(&x, sess, training);
        let (x, _) = x.max_pool2d(&[2, 2], None, &[1, 1], &[0], false, false);
        self.l6.forward(&x.flatten(1, -1))
    }
}

impl Module for Mnist {
    fn named_params(&self) -> Box<dyn Iterator<Item = (String, Tensor)> + '_> {
        Box::new(
            nn::scoped("l0", self.l0.named_params())
                .chain(nn::scoped("l1", self.l1.named_params()))
                .chain(nn::scoped("l2", self.l2.named_params()))
                .chain(nn::scoped("l3", self.l3.named_params()))
                .chain(nn::scoped("l4", self.l4.named_params()))
                .chain(nn::scoped("l5", self.l5.named_params()))
                .chain(nn::scoped("l6", self.l6.named_params())),
        )
    }

    fn into_params(self) -> Self {
        Mnist {
            l0: self.l0.into_params(),
            l1: self.l1.into_params(),
            l2: self.l2.into_params(),
            l3: self.l3.into_params(),
            l4: self.l4.into_params(),
            l5: self.l5.into_params(),
            l6: self.l6.into_params(),
        }
    }
}

/// First call (capture/compile) plus the per-step samples after warmup.
pub(crate) fn measure(mut step: impl FnMut(), warmup: usize, iters: usize) -> (f64, Vec<f64>) {
    // Profile only the measured region; model construction (parameter init)
    // otherwise dominates the table on workloads with hundreds of weights.
    autorust::engine::prof::reset();
    let t = Instant::now();
    step();
    let cold = t.elapsed().as_secs_f64() * 1e3;
    for _ in 0..warmup {
        step();
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        step();
        samples.push(t.elapsed().as_secs_f64() * 1e3);
    }
    (cold, samples)
}

pub(crate) fn stats(samples: &mut [f64]) -> (f64, f64, f64) {
    samples.sort_by(f64::total_cmp);
    let n = samples.len();
    if n == 0 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let median = if n % 2 == 1 {
        samples[n / 2]
    } else {
        0.5 * (samples[n / 2 - 1] + samples[n / 2])
    };
    let q = |p: f64| samples[((n as f64 - 1.0) * p).round() as usize];
    (median, samples[0], q(0.75) - q(0.25))
}

#[allow(clippy::too_many_arguments)]
fn emit(
    workload: &str,
    device: &str,
    batch: i64,
    dtype_name: &str,
    jit: bool,
    warmup: usize,
    iters: usize,
    cold: f64,
    mut samples: Vec<f64>,
) {
    let (median, min, iqr) = stats(&mut samples);
    // `BENCH_ITERS=0` is the cache-priming mode used by `bench/run.py`: report
    // the cold time and say so, instead of printing/emitting `NaN` statistics.
    if samples.is_empty() {
        println!(
            "{workload} {device} bs={batch} jit={jit}: cache primed; \
             no timed samples (cold {cold:.1} ms)"
        );
        let json = format!(
            "{{\"framework\":\"autorust\",\"version\":\"{}\",\"device\":\"{device}\",\
             \"workload\":\"{workload}\",\"batch\":{batch},\"dtype\":\"{dtype_name}\",\"jit\":{jit},\
             \"warmup\":{warmup},\"iters\":0,\"ms_cold\":{cold:.4},\"ms_median\":null,\
             \"ms_min\":null,\"ms_iqr\":null,\"images_per_s\":null,\"cold_only\":true}}",
            env!("CARGO_PKG_VERSION")
        );
        if let Ok(path) = std::env::var("BENCH_OUT") {
            std::fs::write(path, &json).expect("write BENCH_OUT");
        }
        println!("RESULT {json}");
        return;
    }
    let images_per_s = if median.is_finite() && median > 0.0 {
        batch as f64 * 1e3 / median
    } else {
        f64::NAN
    };
    let json = format!(
        "{{\"framework\":\"autorust\",\"version\":\"{}\",\"device\":\"{device}\",\
         \"workload\":\"{workload}\",\"batch\":{batch},\"dtype\":\"{dtype_name}\",\"jit\":{jit},\
         \"warmup\":{warmup},\"iters\":{iters},\"ms_cold\":{cold:.4},\
         \"ms_median\":{median:.4},\"ms_min\":{min:.4},\"ms_iqr\":{iqr:.4},\
         \"images_per_s\":{images_per_s:.2}}}",
        env!("CARGO_PKG_VERSION")
    );
    println!(
        "{workload} {device} bs={batch} jit={jit}: {median:.3} ms/step \
         (min {min:.3}, iqr {iqr:.3}, cold {cold:.1} ms, {images_per_s:.0} img/s)"
    );
    if let Ok(path) = std::env::var("BENCH_OUT") {
        std::fs::write(path, &json).expect("write BENCH_OUT");
    }
    println!("RESULT {json}");
}

/// Emit a multi-case microbench (GEMM sweep, memory-bound ops) as one result.
///
/// `metric` is the per-case column name in the JSON (`tflops` / `gbs`); the
/// stored `ms_median` is the sum of the case medians so the generic runner and
/// report keep working, and `cases` carries the real per-case numbers.
#[allow(clippy::too_many_arguments)]
fn emit_cases(
    workload: &str,
    device: &str,
    batch: i64,
    dtype_name: &str,
    jit: bool,
    warmup: usize,
    iters: usize,
    cases: &[(String, f64, f64)],
    metric: &str,
) {
    let total: f64 = cases.iter().map(|c| c.1).sum();
    let json_cases: Vec<String> = cases
        .iter()
        .map(|(name, ms, v)| format!("{{\"case\":\"{name}\",\"ms\":{ms:.4},\"{metric}\":{v:.4}}}"))
        .collect();
    for (name, ms, v) in cases {
        println!("{workload} {device} {name}: {ms:.3} ms ({v:.2} {metric})");
    }
    let json = format!(
        "{{\"framework\":\"autorust\",\"version\":\"{}\",\"device\":\"{device}\",\
         \"workload\":\"{workload}\",\"batch\":{batch},\"dtype\":\"{dtype_name}\",\"jit\":{jit},\
         \"warmup\":{warmup},\"iters\":{iters},\"ms_cold\":0.0,\
         \"ms_median\":{total:.4},\"ms_min\":0.0,\"ms_iqr\":0.0,\"images_per_s\":0.0,\
         \"cases\":[{}]}}",
        env!("CARGO_PKG_VERSION"),
        json_cases.join(",")
    );
    if let Ok(path) = std::env::var("BENCH_OUT") {
        std::fs::write(path, &json).expect("write BENCH_OUT");
    }
    println!("RESULT {json}");
}

mod gemm;
mod llm;
mod memops;
mod optim_wl;
mod resnet;
mod unet;
mod vit_dtype;

/// Shared model runner: JIT-capture one forward (and backward/Adam step when
/// `training`), then time `warmup + iters` calls.
#[allow(clippy::too_many_arguments)]
fn run_vision<M: Module>(
    ctx: &Ctx,
    sess: &mut Session,
    model: M,
    x: &Tensor,
    y: &Tensor,
    training: bool,
    lr: f32,
    warmup: usize,
    iters: usize,
    fwd: impl Fn(&M, &Tensor, &mut Session, bool) -> Tensor,
) -> (f64, Vec<f64>) {
    let params: Vec<Tensor> = model.params().collect();
    let opt = Adam::new(params.clone(), lr);
    let mut jit = Jit::new();
    let step = |sess: &mut Session| {
        let out = fwd(&model, x, sess, training);
        if !training && std::env::var("BENCH_OUT").is_ok() {
            let v = out.cast(dtype::FLOAT32).to_list::<f32>(sess);
            let sum: f32 = v.iter().sum();
            let first: f32 = v[0];
            eprintln!("OUT sum={sum:.6} first={first:.6}");
        }
        if training {
            let loss = out.sparse_categorical_crossentropy(y, -1, 0.0, "mean");
            loss.backward(&params, None);
            opt.schedule_step(sess, std::slice::from_ref(&loss));
            opt.zero_grad();
            loss
        } else {
            out
        }
    };
    let run = |jit: &mut Jit, sess: &mut Session| {
        let out = jit.call(ctx, sess, |_c, s| step(s));
        let _ = out.cast(dtype::FLOAT32).to_list::<f32>(sess);
    };
    measure(|| run(&mut jit, sess), warmup, iters)
}

/// Optional `BENCH_VERIFY_FOLD=1` gate: run the model before and after BN
/// folding in the same process (same weights/input) and print the max
/// difference, then leave it folded.
fn verify_and_fold<M>(
    sess: &mut Session,
    model: &mut M,
    x: &Tensor,
    forward: impl Fn(&M, &Tensor, &mut Session) -> Tensor,
    fold: impl Fn(&mut M, &mut Session),
) {
    if std::env::var("BENCH_VERIFY_FOLD").is_ok() {
        // A non-zero probe so the convs do real work (the benchmark input is
        // zeros); it is identical for the before/after runs.
        let ctx = x.ctx().clone();
        let shape: Vec<i64> = x
            .shape()
            .iter()
            .map(|s| s.clone().as_i64().unwrap_or(0))
            .collect();
        let probe = Tensor::rand(&ctx, &shape, None, None);
        let before = forward(model, &probe, sess).to_list::<f32>(sess);
        fold(model, sess);
        let after = forward(model, &probe, sess).to_list::<f32>(sess);
        let d = before
            .iter()
            .zip(&after)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("FOLD_DIFF {d:e}");
    } else {
        fold(model, sess);
    }
}

fn main() {
    let workload = std::env::var("BENCH_WORKLOAD").unwrap_or_else(|_| "mnist_train".into());
    // JIT capture/replay is the steady-state path; BENCH_JIT=0 opts out.
    let jit_on = std::env::var("BENCH_JIT")
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(true);
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let device = autorust::tensor::creation::default_device();
    if jit_on {
        sess.enable_jit();
    }

    match workload.as_str() {
        "mnist_train" => {
            let bs = env_i64("BENCH_BS", 512);
            let warmup = env_i64("BENCH_WARMUP", 5) as usize;
            let iters = env_i64("BENCH_ITERS", 30) as usize;
            let model = Mnist::new(&ctx).into_params();
            let params: Vec<Tensor> = model.params().collect();
            let opt = Adam::new(params.clone(), 1e-3);
            let xb = Tensor::rand(&ctx, &[bs, 1, 28, 28], None, None).to_param();
            let yb = Tensor::randint(&ctx, &[bs], 0, 10, Some(dtype::INT32), None).to_param();
            sess.training(true);

            let mut jit = Jit::new();
            let step = |sess: &mut Session| {
                let out = model.forward(&xb, sess, true);
                let loss = out.sparse_categorical_crossentropy(&yb, -1, 0.0, "mean");
                loss.backward(&params, None);
                opt.schedule_step(sess, std::slice::from_ref(&loss));
                opt.zero_grad();
                loss
            };
            let run = |jit: &mut Jit, sess: &mut Session| {
                let loss = jit.call(&ctx, sess, |_c, s| step(s));
                let l = loss.item::<f32>(sess);
                if std::env::var("BENCH_LOSS").is_ok() {
                    eprintln!("LOSS {l:.6}");
                }
            };
            let (cold, samples) = measure(|| run(&mut jit, &mut sess), warmup, iters);
            emit(
                "mnist_train",
                &device,
                bs,
                "f32",
                jit_on,
                warmup,
                iters,
                cold,
                samples,
            );
        }
        "vit_infer" => {
            let bs = env_i64("BENCH_BS", 1);
            let warmup = env_i64("BENCH_WARMUP", 3) as usize;
            let iters = env_i64("BENCH_ITERS", 10) as usize;
            let layers = env_i64("BENCH_LAYERS", 12);
            let embed = env_i64("BENCH_EMBED", 768);
            let heads = env_i64("BENCH_HEADS", 12);
            let img = env_i64("BENCH_IMG", 224);
            if let Ok(seed) = std::env::var("BENCH_SEED") {
                Tensor::manual_seed(seed.parse().expect("BENCH_SEED"));
            }
            let model = ViT::build(&ctx, layers, embed, heads, 3, img, 16, 1000).into_params();
            let x = Tensor::zeros(&ctx, [bs, 3, img, img]).to_param();
            let mut jit = Jit::new();
            let run = |jit: &mut Jit, sess: &mut Session| {
                let out = jit.call(&ctx, sess, |_c, s| model.forward_with(&x, s));
                // Force completion in the *same* session: a fresh session would
                // re-schedule the output and measure that instead of the step.
                let _ = out.to_list::<f32>(sess);
            };
            let (cold, samples) = measure(|| run(&mut jit, &mut sess), warmup, iters);
            emit(
                "vit_infer",
                &device,
                bs,
                "f32",
                jit_on,
                warmup,
                iters,
                cold,
                samples,
            );
        }
        "vit_train" => {
            let bs = env_i64("BENCH_BS", 1);
            let warmup = env_i64("BENCH_WARMUP", 2) as usize;
            let iters = env_i64("BENCH_ITERS", 5) as usize;
            let layers = env_i64("BENCH_LAYERS", 12);
            let embed = env_i64("BENCH_EMBED", 768);
            let heads = env_i64("BENCH_HEADS", 12);
            let img = env_i64("BENCH_IMG", 224);
            if let Ok(seed) = std::env::var("BENCH_SEED") {
                Tensor::manual_seed(seed.parse().expect("BENCH_SEED"));
            }
            let model = ViT::build(&ctx, layers, embed, heads, 3, img, 16, 1000).into_params();
            let params: Vec<Tensor> = model.params().collect();
            let opt = Adam::new(params.clone(), 3e-4);
            let x = Tensor::zeros(&ctx, [bs, 3, img, img]).to_param();
            let y = Tensor::randint(&ctx, &[bs], 0, 1000, Some(dtype::INT32), None).to_param();
            sess.training(true);
            let mut jit = Jit::new();
            let step = |sess: &mut Session| {
                let out = model.forward(&x);
                let loss = out.sparse_categorical_crossentropy(&y, -1, 0.0, "mean");
                loss.backward(&params, None);
                opt.schedule_step(sess, std::slice::from_ref(&loss));
                opt.zero_grad();
                loss
            };
            let run = |jit: &mut Jit, sess: &mut Session| {
                let loss = jit.call(&ctx, sess, |_c, s| step(s));
                let l = loss.item::<f32>(sess);
                if std::env::var("BENCH_LOSS").is_ok() {
                    eprintln!("LOSS {l:.6}");
                }
            };
            let (cold, samples) = measure(|| run(&mut jit, &mut sess), warmup, iters);
            emit(
                "vit_train",
                &device,
                bs,
                "f32",
                jit_on,
                warmup,
                iters,
                cold,
                samples,
            );
        }
        "gemm" => {
            let warmup = env_i64("BENCH_WARMUP", 1) as usize;
            let iters = env_i64("BENCH_ITERS", 3) as usize;
            let bf16 = std::env::var("BENCH_BF16")
                .map(|v| v != "0" && !v.is_empty())
                .unwrap_or(false);
            let only = std::env::var("BENCH_GEMM_ONLY").ok();
            let cases = gemm::run(&ctx, &mut sess, bf16, warmup, iters, only.as_deref());
            let rows: Vec<(String, f64, f64)> = cases
                .into_iter()
                .map(|c| (c.name, c.ms, c.tflops))
                .collect();
            emit_cases(
                "gemm",
                &device,
                1,
                if bf16 { "bf16" } else { "f32" },
                true,
                warmup,
                iters,
                &rows,
                "tflops",
            );
        }
        "memops" => {
            let warmup = env_i64("BENCH_WARMUP", 2) as usize;
            let iters = env_i64("BENCH_ITERS", 5) as usize;
            let only = std::env::var("BENCH_MEMOPS_ONLY").ok();
            let cases = memops::run(&ctx, &mut sess, warmup, iters, only.as_deref());
            let rows: Vec<(String, f64, f64)> =
                cases.into_iter().map(|c| (c.name, c.ms, c.gbs)).collect();
            emit_cases(
                "memops", &device, 1, "f32", true, warmup, iters, &rows, "gbs",
            );
        }
        "optim" => {
            let warmup = env_i64("BENCH_WARMUP", 2) as usize;
            let iters = env_i64("BENCH_ITERS", 10) as usize;
            let total = env_i64("BENCH_OPT_PARAMS", 100_000_000);
            let tensors = env_i64("BENCH_OPT_TENSORS", 32);
            let c = optim_wl::run(&ctx, &mut sess, total, tensors, warmup, iters);
            emit_cases(
                "optim",
                &device,
                1,
                "f32",
                true,
                warmup,
                iters,
                &[(c.name, c.ms, c.gbs)],
                "gbs",
            );
        }
        "resnet18_train" | "resnet18_infer" | "resnet50_infer" | "mobilenet_infer" => {
            let bs = env_i64("BENCH_BS", 32);
            let training = workload.ends_with("_train");
            if let Ok(seed) = std::env::var("BENCH_SEED") {
                Tensor::manual_seed(seed.parse().expect("BENCH_SEED"));
            }
            // Measured (RTX 3060): BN folding + the conv host GEMM wins on
            // ResNet-18 (-17.6%) and MobileNet (-14%), but ResNet-50's 1x1 and
            // small-spatial stages are slower through cuBLAS (+~17%), so the
            // benchmark keeps that model unfolded by default.
            let fold_bn = std::env::var("AUTORUST_FOLD_BN")
                .map(|v| v != "0")
                .unwrap_or(workload != "resnet50_infer");
            let warmup = env_i64("BENCH_WARMUP", 3) as usize;
            let iters = env_i64("BENCH_ITERS", if training { 5 } else { 10 }) as usize;
            sess.training(training);
            let x = Tensor::zeros(&ctx, [bs, 3, 224, 224]).to_param();
            let y = Tensor::randint(&ctx, &[bs], 0, 1000, Some(dtype::INT32), None).to_param();
            let (cold, samples) = match workload.as_str() {
                "resnet18_train" | "resnet18_infer" => {
                    let mut model = resnet::ResNet::new(&ctx, 18, 1000, 1.0).into_params();
                    if !training && fold_bn {
                        verify_and_fold(
                            &mut sess,
                            &mut model,
                            &x,
                            |m, x, s| m.forward(x, s, false),
                            |m, s| m.fold_bn(s),
                        );
                    }
                    run_vision(
                        &ctx,
                        &mut sess,
                        model,
                        &x,
                        &y,
                        training,
                        1e-3,
                        warmup,
                        iters,
                        |m, x, s, t| m.forward(x, s, t),
                    )
                }
                "resnet50_infer" => {
                    let mut model = resnet::ResNet::new(&ctx, 50, 1000, 1.0).into_params();
                    if !training && fold_bn {
                        verify_and_fold(
                            &mut sess,
                            &mut model,
                            &x,
                            |m, x, s| m.forward(x, s, false),
                            |m, s| m.fold_bn(s),
                        );
                    }
                    run_vision(
                        &ctx,
                        &mut sess,
                        model,
                        &x,
                        &y,
                        training,
                        1e-3,
                        warmup,
                        iters,
                        |m, x, s, t| m.forward(x, s, t),
                    )
                }
                _ => {
                    let mut model = resnet::MobileNet::new(&ctx, 1000).into_params();
                    if !training && fold_bn {
                        verify_and_fold(
                            &mut sess,
                            &mut model,
                            &x,
                            |m, x, s| m.forward(x, s, false),
                            |m, s| m.fold_bn(s),
                        );
                    }
                    run_vision(
                        &ctx,
                        &mut sess,
                        model,
                        &x,
                        &y,
                        training,
                        1e-3,
                        warmup,
                        iters,
                        |m, x, s, t| m.forward(x, s, t),
                    )
                }
            };
            emit(
                workload.as_str(),
                &device,
                bs,
                "f32",
                true,
                warmup,
                iters,
                cold,
                samples,
            );
        }
        "unet_infer" => {
            let bs = env_i64("BENCH_BS", 4);
            let warmup = env_i64("BENCH_WARMUP", 2) as usize;
            let iters = env_i64("BENCH_ITERS", 5) as usize;
            let model = unet::Unet::new(&ctx);
            let x = Tensor::rand(&ctx, &[bs, 3, 64, 64], None, None).to_param();
            let mut jit = Jit::new();
            let run = |jit: &mut Jit, sess: &mut Session| {
                let out = jit.call(&ctx, sess, |_c, _s| model.forward(&x));
                let _ = out
                    .getitem(&vec![Ix::slice(None, Some(1), 1); 4])
                    .to_list::<f32>(sess);
            };
            let (cold, samples) = measure(|| run(&mut jit, &mut sess), warmup, iters);
            emit(
                "unet_infer",
                &device,
                bs,
                "f32",
                true,
                warmup,
                iters,
                cold,
                samples,
            );
        }
        "llm_prefill" => {
            let mut cfg = llm::Config::default();
            cfg.vocab = env_i64("BENCH_LLM_VOCAB", cfg.vocab);
            cfg.seq = env_i64("BENCH_LLM_SEQ", cfg.seq);
            cfg.n_layers = env_i64("BENCH_LLM_LAYERS", cfg.n_layers);
            cfg.d_model = env_i64("BENCH_LLM_DMODEL", cfg.d_model);
            cfg.n_heads = env_i64("BENCH_LLM_HEADS", cfg.n_heads);
            cfg.n_kv = env_i64("BENCH_LLM_KV", cfg.n_kv);
            cfg.ffn = env_i64("BENCH_LLM_FFN", cfg.ffn);
            let warmup = env_i64("BENCH_WARMUP", 1) as usize;
            let iters = env_i64("BENCH_ITERS", 3) as usize;
            let (_model, _tok, _caches, ms) =
                llm::bench_prefill(&ctx, &mut sess, &cfg, warmup, iters);
            emit(
                "llm_prefill",
                &device,
                cfg.seq,
                "f32",
                true,
                warmup,
                iters,
                0.0,
                vec![ms],
            );
        }
        "llm_decode" => {
            let mut cfg = llm::Config::default();
            cfg.vocab = env_i64("BENCH_LLM_VOCAB", cfg.vocab);
            cfg.seq = env_i64("BENCH_LLM_SEQ", cfg.seq);
            cfg.n_layers = env_i64("BENCH_LLM_LAYERS", cfg.n_layers);
            cfg.d_model = env_i64("BENCH_LLM_DMODEL", cfg.d_model);
            cfg.n_heads = env_i64("BENCH_LLM_HEADS", cfg.n_heads);
            cfg.n_kv = env_i64("BENCH_LLM_KV", cfg.n_kv);
            cfg.ffn = env_i64("BENCH_LLM_FFN", cfg.ffn);
            let (_model, _tok, caches, _) = llm::bench_prefill(&ctx, &mut sess, &cfg, 0, 1);
            let ms = llm::bench_decode(&ctx, &mut sess, &_model, &caches, &cfg, 2, 5);
            emit("llm_decode", &device, 1, "f32", true, 2, 5, 0.0, vec![ms]);
        }
        "vit_infer_bf16" => {
            let bs = env_i64("BENCH_BS", 8);
            let warmup = env_i64("BENCH_WARMUP", 1) as usize;
            let iters = env_i64("BENCH_ITERS", 3) as usize;
            let layers = env_i64("BENCH_LAYERS", 12);
            let embed = env_i64("BENCH_EMBED", 768);
            let heads = env_i64("BENCH_HEADS", 12);
            let img = env_i64("BENCH_IMG", 224);
            let model = vit_dtype::cast(
                &ViT::build(&ctx, layers, embed, heads, 3, img, 16, 1000),
                dtype::BFLOAT16,
            );
            let x = Tensor::zeros(&ctx, [bs, 3, img, img])
                .cast(dtype::BFLOAT16)
                .to_param();
            let mut jit = Jit::new();
            let run = |jit: &mut Jit, sess: &mut Session| {
                let out = jit.call(&ctx, sess, |_c, s| model.forward_with(&x, s));
                let _ = out.cast(dtype::FLOAT32).to_list::<f32>(sess);
            };
            let (cold, samples) = measure(|| run(&mut jit, &mut sess), warmup, iters);
            emit(
                "vit_infer_bf16",
                &device,
                bs,
                "bf16",
                true,
                warmup,
                iters,
                cold,
                samples,
            );
        }
        "onnx_mnist" => {
            let bs = env_i64("BENCH_BS", 32);
            let warmup = env_i64("BENCH_WARMUP", 2) as usize;
            let iters = env_i64("BENCH_ITERS", 10) as usize;
            let path = std::env::var("BENCH_ONNX")
                .unwrap_or_else(|_| "bench/assets/mnist_bs32.onnx".to_string());
            let mut m = autorust::onnx::OnnxModel::load_with_device(&path, Some(&device))
                .unwrap_or_else(|e| panic!("load {path}: {e}"));
            let x = Tensor::rand(m.ctx(), &[bs, 1, 28, 28], None, None);
            // `run` pins the input buffer, so it must be realized first.
            x.realize(m.session_mut());
            let mut step = || {
                let mut out = m.run(&[("x", x.clone())]).expect("onnx run");
                let y = out.remove("logits").expect("logits");
                let _ = y
                    .getitem(&vec![Ix::slice(None, Some(1), 1); 2])
                    .to_list::<f32>(m.session_mut());
            };
            let (cold, samples) = measure(&mut step, warmup, iters);
            emit(
                "onnx_mnist",
                &device,
                bs,
                "f32",
                true,
                warmup,
                iters,
                cold,
                samples,
            );
        }
        "vit_train_bf16" => {
            let bs = env_i64("BENCH_BS", 1);
            let warmup = env_i64("BENCH_WARMUP", 1) as usize;
            let iters = env_i64("BENCH_ITERS", 3) as usize;
            let layers = env_i64("BENCH_LAYERS", 12);
            let embed = env_i64("BENCH_EMBED", 768);
            let heads = env_i64("BENCH_HEADS", 12);
            let img = env_i64("BENCH_IMG", 224);
            let model = vit_dtype::cast(
                &ViT::build(&ctx, layers, embed, heads, 3, img, 16, 1000),
                dtype::BFLOAT16,
            );
            let params: Vec<Tensor> = model.params().collect();
            let opt = Adam::new(params.clone(), 3e-4);
            let x = Tensor::zeros(&ctx, [bs, 3, img, img])
                .cast(dtype::BFLOAT16)
                .to_param();
            let y = Tensor::randint(&ctx, &[bs], 0, 1000, Some(dtype::INT32), None).to_param();
            sess.training(true);
            let mut jit = Jit::new();
            let step = |sess: &mut Session| {
                let out = model.forward(&x);
                let loss = out.sparse_categorical_crossentropy(&y, -1, 0.0, "mean");
                loss.backward(&params, None);
                opt.schedule_step(sess, std::slice::from_ref(&loss));
                opt.zero_grad();
                loss
            };
            let run = |jit: &mut Jit, sess: &mut Session| {
                let loss = jit.call(&ctx, sess, |_c, s| step(s));
                let _ = loss.cast(dtype::FLOAT32).item::<f32>(sess);
            };
            let (cold, samples) = measure(|| run(&mut jit, &mut sess), warmup, iters);
            emit(
                "vit_train_bf16",
                &device,
                bs,
                "bf16",
                true,
                warmup,
                iters,
                cold,
                samples,
            );
        }
        "onnx_resnet" => {
            let bs = env_i64("BENCH_BS", 8);
            let warmup = env_i64("BENCH_WARMUP", 2) as usize;
            let iters = env_i64("BENCH_ITERS", 5) as usize;
            let path = std::env::var("BENCH_ONNX")
                .unwrap_or_else(|_| "bench/assets/resnet18w25_bs8.onnx".to_string());
            let mut m = autorust::onnx::OnnxModel::load_with_device(&path, Some(&device))
                .unwrap_or_else(|e| panic!("load {path}: {e}"));
            let x = Tensor::rand(m.ctx(), &[bs, 3, 224, 224], None, None);
            x.realize(m.session_mut());
            let step = || {
                let mut out = m.run(&[("x", x.clone())]).expect("onnx run");
                let y = out.remove("logits").expect("logits");
                let _ = y
                    .getitem(&vec![Ix::slice(None, Some(1), 1); 2])
                    .to_list::<f32>(m.session_mut());
            };
            let (cold, samples) = measure(step, warmup, iters);
            emit(
                "onnx_resnet",
                &device,
                bs,
                "f32",
                true,
                warmup,
                iters,
                cold,
                samples,
            );
        }
        other => panic!(
            "unknown BENCH_WORKLOAD={other} (mnist_train|vit_infer|vit_train|gemm|memops|\
             optim|resnet18_train|resnet18_infer|resnet50_infer|mobilenet_infer|unet_infer|\
             llm_prefill|llm_decode|vit_infer_bf16|vit_train_bf16|onnx_mnist|onnx_resnet)"
        ),
    }
    autorust::engine::prof::report();
    println!("compiles: {}", autorust::runtime::device::total_compiles());
}
