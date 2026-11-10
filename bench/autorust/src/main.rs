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
fn measure(mut step: impl FnMut(), warmup: usize, iters: usize) -> (f64, Vec<f64>) {
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

fn stats(samples: &mut [f64]) -> (f64, f64, f64) {
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
    jit: bool,
    warmup: usize,
    iters: usize,
    cold: f64,
    mut samples: Vec<f64>,
) {
    let (median, min, iqr) = stats(&mut samples);
    let images_per_s = if median.is_finite() && median > 0.0 {
        batch as f64 * 1e3 / median
    } else {
        f64::NAN
    };
    let json = format!(
        "{{\"framework\":\"autorust\",\"version\":\"{}\",\"device\":\"{device}\",\
         \"workload\":\"{workload}\",\"batch\":{batch},\"dtype\":\"f32\",\"jit\":{jit},\
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
                let _ = loss.item::<f32>(sess);
            };
            let (cold, samples) = measure(|| run(&mut jit, &mut sess), warmup, iters);
            emit(
                "mnist_train",
                &device,
                bs,
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
                let _ = loss.item::<f32>(sess);
            };
            let (cold, samples) = measure(|| run(&mut jit, &mut sess), warmup, iters);
            emit(
                "vit_train",
                &device,
                bs,
                jit_on,
                warmup,
                iters,
                cold,
                samples,
            );
        }
        other => panic!("unknown BENCH_WORKLOAD={other} (mnist_train|vit_infer|vit_train)"),
    }
    autorust::engine::prof::report();
    println!("compiles: {}", autorust::runtime::device::total_compiles());
}
