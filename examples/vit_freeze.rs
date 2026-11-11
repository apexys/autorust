//! Freeze a ViT-B/16 forward and execute the frozen plan.
//!
//! ```text
//! LAYERS=1 EMBED=192 HEADS=3 IMG=64 PATCH=16 \
//!   cargo run --release --features freeze --example vit_freeze
//! ```
//!
//! The plan is built from a captured `Jit`; the runner uses the loaded kernel
//! table and compares the bytes with the eager captured output. This is the
//! in-process gate for the AOT emitter.

use autorust::engine::jit::Jit;
use autorust::engine::session::Session;
use autorust::freeze::FrozenRunner;
use autorust::freeze::plan;
use autorust::models::vit::ViT;
use autorust::runtime::device::device;
use autorust::tensor::nn::Module;
use autorust::tensor::{Ctx, Tensor};
use std::time::Instant;

fn env_i64(key: &str, default: i64) -> i64 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn main() {
    // A frozen plan is CPU code emitted for the RUST backend; capture on RUST
    // even if CL/CUDA are present (otherwise the plan holds CL/PTX artifacts).
    unsafe { std::env::set_var("DEFAULT_DEVICE", "RUST") };
    let layers = env_i64("LAYERS", 1);
    let embed = env_i64("EMBED", 192);
    let heads = env_i64("HEADS", 3);
    let img = env_i64("IMG", 64);
    let patch = env_i64("PATCH", 16);
    let classes = env_i64("CLASSES", 1000);

    let ctx = Ctx::new();
    let mut sess = Session::new();
    sess.enable_jit();
    let model = ViT::build(&ctx, layers, embed, heads, 3, img, patch, classes).into_params();
    let weights: Vec<Tensor> = model.named_params().map(|(_, t)| t).collect();
    let x = Tensor::zeros(&ctx, [1, 3, img, img]);

    let mut jit = Jit::new();
    let outs = jit.call_many(&ctx, &mut sess, |_c, _s| vec![model.forward(&x)]);

    let t = Instant::now();
    let built = plan::build(&ctx, &mut sess, &jit, std::slice::from_ref(&x), &weights);
    let build_ms = t.elapsed().as_secs_f64() * 1e3;
    println!(
        "freeze: {:.2}s, {} storages, {} views, {} steps, {} kernels, {} init copies",
        t.elapsed().as_secs_f64(),
        built.plan.storages.len(),
        built.plan.views.len(),
        built.plan.steps.len(),
        built.progs.len(),
        built
            .plan
            .steps
            .iter()
            .filter(|s| matches!(s, autorust::freeze::Step::Copy { .. }))
            .count()
    );
    if let Ok(path) = std::env::var("FREEZE_EMIT") {
        autorust::freeze::emit_to(&built, &path).expect("emit frozen source");
        println!("emitted {path}");
    }
    let mut runner = FrozenRunner::new(built.plan, built.progs);
    let t = Instant::now();
    let got = runner.run(std::slice::from_ref(&x), &weights);
    println!("frozen run: {:.3}s", t.elapsed().as_secs_f64());
    if std::env::var("FROZEN_PROF").is_ok() {
        runner.report_profile();
    }

    let want = {
        let b = outs[0].realized_buffer().expect("eager output realized");
        let d = device(b.device());
        d.copyout(&*b.get_buf(&d))
    };
    assert_eq!(got.len(), 1);
    assert_eq!(got[0], want, "frozen bytes differ from eager");
    println!("frozen output matches eager ({} bytes)", got[0].len());

    // Steady-state comparison: replay the captured eager plan vs the frozen
    // runner (both execute the same kernels; the difference is the runtime).
    let warmup = env_i64("BENCH_WARMUP", 2) as usize;
    let iters = env_i64("BENCH_ITERS", 5) as usize;
    let measure = |mut step: Box<dyn FnMut()>| -> f64 {
        for _ in 0..warmup {
            step();
        }
        let mut xs = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = Instant::now();
            step();
            xs.push(t.elapsed().as_secs_f64() * 1e3);
        }
        xs.sort_by(f64::total_cmp);
        xs[xs.len() / 2]
    };
    let eager_ms = measure(Box::new(|| {
        let outs = jit.call_many(&ctx, &mut sess, |_c, _s| vec![model.forward(&x)]);
        // Match the frozen runner's host copy of the output.
        let b = outs[0].realized_buffer().expect("eager output realized");
        let d = device(b.device());
        let _ = d.copyout(&*b.get_buf(&d));
    }));
    let frozen_ms = measure(Box::new(|| {
        let _ = runner.run(std::slice::from_ref(&x), &weights);
    }));
    println!(
        "freeze bench: eager {eager_ms:.3} ms, frozen {frozen_ms:.3} ms ({:.2}x)",
        eager_ms / frozen_ms
    );
    println!(
        "RESULT {{\"framework\":\"autorust\",\"version\":\"{}\",\"device\":\"RUST\",\
         \"workload\":\"freeze\",\"batch\":1,\"dtype\":\"f32\",\"jit\":true,\
         \"warmup\":{warmup},\"iters\":{iters},\"ms_cold\":0.0,\"ms_median\":{eager_ms:.4},\
         \"ms_min\":0.0,\"ms_iqr\":0.0,\"images_per_s\":0.0,\
         \"cases\":[{{\"case\":\"eager jit\",\"ms\":{eager_ms:.4}}},\
         {{\"case\":\"frozen plan\",\"ms\":{frozen_ms:.4}}},\
         {{\"case\":\"plan build\",\"ms\":{build_ms:.4}}}]}}",
        env!("CARGO_PKG_VERSION")
    );
}
