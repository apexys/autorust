//! ViT training-step benchmark (forward + backward + Adam), synthetic data.
//!
//! Unlike `vit_bench` (inference), this exercises the backward graph, dropout,
//! the optimizer update, and the RNG device graph. It is intentionally
//! dataset-free: the input/labels are fixed parameters and the step is captured
//! with `Jit`, so the reported time is steady-state training compute.
//!
//! ```text
//! DEFAULT_DEVICE=RUST JIT=1 ITERS=3 cargo run --release --example vit_train_bench
//! DEFAULT_DEVICE=CUDA JIT=1 ITERS=5 \
//!   cargo run --release --features backend_cuda --example vit_train_bench
//! LAYERS=1 EMBED=192 HEADS=3 IMG=64 ITERS=2 ...   # small smoke config
//! ```
//!
//! `TRAINING=1` turns dropout on (otherwise it is identity and the step skips
//! those kernels). `DTYPE=bfloat16` selects bf16 via `DEFAULT_FLOAT`.

use autorust::dtype;
use autorust::engine::jit::Jit;
use autorust::engine::session::Session;
use autorust::models::vit::ViT;
use autorust::tensor::nn::Module;
use autorust::tensor::optim::{Adam, Optim};
use autorust::tensor::{Ctx, Tensor};
use std::time::Instant;

fn env_i64(key: &str, default: i64) -> i64 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_flag(key: &str) -> bool {
    std::env::var(key)
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false)
}

fn main() {
    let layers = env_i64("LAYERS", 2);
    let embed = env_i64("EMBED", 192);
    let heads = env_i64("HEADS", 3);
    let img = env_i64("IMG", 64);
    let patch = env_i64("PATCH", 16);
    let classes = env_i64("CLASSES", 10);
    let bs = env_i64("BS", 1);
    let iters = env_i64("ITERS", 5).max(1);
    let lr = std::env::var("LR")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(3e-4) as f32;
    let training = env_flag("TRAINING");
    let dev = std::env::var("DEFAULT_DEVICE")
        .unwrap_or_else(|_| autorust::tensor::creation::default_device());

    let ctx = Ctx::new();
    let mut sess = Session::new();
    sess.enable_jit();

    let t0 = Instant::now();
    let model = ViT::build(&ctx, layers, embed, heads, 3, img, patch, classes).into_params();
    let params: Vec<Tensor> = model.params().collect();
    let opt = Adam::new(params.clone(), lr);
    let x = Tensor::zeros(&ctx, [bs, 3, img, img]).to_param();
    let y = Tensor::randint(
        &mut ctx.borrow_mut(),
        &[bs],
        0,
        classes as i32,
        Some(dtype::INT32),
        None,
    );
    let y = y.to_param();
    println!(
        "ViT train: dev={dev} layers={layers} embed={embed} heads={heads} img={img} \
         patch={patch} classes={classes} bs={bs} training={training} params={}",
        params.len()
    );
    println!("build+init: {:.2}s", t0.elapsed().as_secs_f64());

    let mut jit = Jit::new();
    let step = |_ctx: &Ctx, sess: &mut Session| {
        let out = model.forward(&x);
        let loss = out.sparse_categorical_crossentropy(&y, -1, 0.0, "mean");
        loss.backward(&params, None);
        opt.schedule_step(sess, std::slice::from_ref(&loss));
        opt.zero_grad();
        loss
    };

    // Capture + first step.
    let t = Instant::now();
    let loss = jit.call(&ctx, &mut sess, step);
    let first = loss.item::<f32>(&mut sess);
    println!(
        "jit capture + step: {:.3}s (loss {first:.4}, {} linears)",
        t.elapsed().as_secs_f64(),
        jit.n_linears()
    );

    let t = Instant::now();
    for _ in 0..iters {
        let l = jit.call(&ctx, &mut sess, step);
        let _ = l.item::<f32>(&mut sess);
    }
    let secs = t.elapsed().as_secs_f64() / iters as f64;
    println!(
        "train step: {:.3}s/iter ({:.3} s total)",
        secs,
        secs * iters as f64
    );
    println!(
        "to_program: {} misses / {} hits",
        sess.cache.misses, sess.cache.hits
    );
    autorust::engine::prof::report();
}
