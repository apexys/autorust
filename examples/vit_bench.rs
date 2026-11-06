//! ViT-B/16 forward benchmark for the CPU backends.
//!
//! Defaults to a full ViT-B/16 (12 layers, width 768, 12 heads, 224×224,
//! patch 16, 1000 classes). Override the shape with env vars for a quick
//! smoke run:
//!
//! ```text
//! cargo run --release --example vit_bench
//! cargo run --release --features backend_rust --example vit_bench
//! DEV=RUST JIT=1 cargo run --release --features backend_rust --example vit_bench
//! LAYERS=1 EMBED=192 HEADS=3 IMG=64 PATCH=16 ITERS=1 cargo run --release --example vit_bench
//! ```
//!
//! `JIT=1` captures the forward once and replays the compiled schedule, so the
//! timing excludes rangeify/scheduling and shows the steady-state model cost.

use autorust::engine::jit::Jit;
use autorust::engine::session::Session;
use autorust::models::vit::ViT;
use autorust::tensor::nn::Module;
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
    let layers = env_i64("LAYERS", 12);
    let embed = env_i64("EMBED", 768);
    let heads = env_i64("HEADS", 12);
    let img = env_i64("IMG", 224);
    let patch = env_i64("PATCH", 16);
    let classes = env_i64("CLASSES", 1000);
    let iters = env_i64("ITERS", 1).max(1);
    let jit_on = env_flag("JIT");
    let dev = std::env::var("DEV").unwrap_or_else(|_| autorust::tensor::creation::default_device());

    let ctx = Ctx::new();
    let t0 = Instant::now();
    let model = ViT::build(&ctx, layers, embed, heads, 3, img, patch, classes).into_params();
    let x = Tensor::zeros(&ctx, [1, 3, img, img]);
    println!(
        "ViT-B/16-ish: dev={dev} layers={layers} embed={embed} heads={heads} \
         img={img} patch={patch} classes={classes} jit={jit_on}"
    );
    println!("build+init: {:.2}s", t0.elapsed().as_secs_f64());

    // Materialize the weights and the input once; the timed loop only runs the
    // forward (and, without JIT, rebuilds/schedules it each iteration).
    let out = model.forward(&x);
    let _ = out.realize_fresh();
    println!("logits: {:?}", out.shape());

    let mut sess = Session::new();
    let mut jit = Jit::new();
    if jit_on {
        sess.jit(true);
        // Capture + first execution.
        let t = Instant::now();
        let _ = jit.call(&ctx, &mut sess, |_ctx, _sess| model.forward(&x));
        println!(
            "jit capture: {:.3}s ({} linears)",
            t.elapsed().as_secs_f64(),
            jit.n_linears()
        );
    }

    let t = Instant::now();
    for _ in 0..iters {
        let out = if jit_on {
            jit.call(&ctx, &mut sess, |_ctx, _sess| model.forward(&x))
        } else {
            model.forward(&x)
        };
        // Force completion/readback: CUDA launches are asynchronous, so timing
        // only the enqueue would under-report device time.
        let _ = out.to_list_fresh::<f32>();
    }
    let secs = t.elapsed().as_secs_f64() / iters as f64;
    println!(
        "forward: {:.3}s/iter ({:.3} s total)",
        secs,
        secs * iters as f64
    );
    println!(
        "to_program: {} misses / {} hits",
        sess.cache.misses, sess.cache.hits
    );
}
