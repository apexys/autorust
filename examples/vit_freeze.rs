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
}
