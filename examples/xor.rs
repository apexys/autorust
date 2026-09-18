//! XOR training demo: a 2->4->1 ReLU MLP trained with SGD.
//!
//! Derived from `tests/nn_xor.rs::xor_trains_to_convergence`, but as a runnable
//! program instead of a test. Run it with:
//!
//! ```sh
//! cargo run --release --example xor
//! EPOCHS=1000 LR=0.05 cargo run --release --example xor
//! ```
//!
//! The layers are created once; their parameters are buffer-backed leaves that
//! the optimizer updates in place (`into_params` + `Optim::schedule_step`), and
//! the loss is co-realized with the update in one schedule, so the per-step
//! graph stays identical and bounded and every step after the first is a
//! compiled-program cache hit.

use autorust::engine::session::Session;
use autorust::tensor::nn::{self, Linear};
use autorust::tensor::optim::{Optim, SGD};
use autorust::tensor::Tensor;
use autorust::uop::ctx::UOpCtx;

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn main() {
    let lr: f32 = env_or("LR", 0.1);
    let n: usize = env_or("EPOCHS", 300);
    let seed: i64 = env_or("SEED", 42);

    println!("XOR training: 2->4(relu)->1 MLP, MSE, SGD lr={lr}, epochs={n}, seed={seed}");

    let mut ctx = UOpCtx::new();
    let mut sess = Session::new();

    // Create the layers once and make their params updateable buffer leaves.
    Tensor::manual_seed(seed);
    let l1 = Linear::new(&mut ctx, 2, 4, true).into_params(&mut ctx);
    let l2 = Linear::new(&mut ctx, 4, 1, true).into_params(&mut ctx);
    let params = vec![
        l1.weight.clone(),
        l1.bias.clone().unwrap(),
        l2.weight.clone(),
        l2.bias.clone().unwrap(),
    ];
    let sgd = SGD::new(params.clone(), lr);

    let x = Tensor::from_slice(&mut ctx, &[0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], None, None)
        .reshape(&mut ctx, [4, 2]);
    let t = Tensor::from_slice(&mut ctx, &[0.0f32, 1.0, 1.0, 0.0], None, None).reshape(&mut ctx, [4, 1]);

    let mut pred = Vec::new();
    let mut last_loss = f32::NAN;
    let mut first_epoch_misses = 0;
    let log_every = (n / 10).max(1);

    for epoch in 0..n {
        let h = nn::relu(&l1.forward(&x, &mut ctx), &mut ctx);
        let out = l2.forward(&h, &mut ctx);
        let diff = out.sub(&mut ctx, &t);
        let loss = diff.mul(&mut ctx, &diff).mean(&mut ctx, None, false);

        loss.backward(&mut ctx, &params, None);
        // Co-realize the loss with the optimizer update in one schedule
        // (`loss.realize(*opt.schedule_step())` in tinygrad), then clear grads.
        sgd.schedule_step(&mut ctx, &mut sess, &[loss.clone()]);
        sgd.zero_grad(&ctx);

        pred = out.reshape(&mut ctx, [4]).to_list::<f32>(&mut ctx, &mut sess);
        last_loss = loss.reshape(&mut ctx, [1]).to_list::<f32>(&mut ctx, &mut sess)[0];

        if epoch == 0 {
            first_epoch_misses = sess.cache.misses;
        }
        if epoch % log_every == 0 || epoch == n - 1 {
            println!("epoch {epoch:>5}: loss = {last_loss:.5}");
        }
    }

    println!(
        "to_program: {} misses / {} hits ({} cached)",
        sess.cache.misses,
        sess.cache.hits,
        sess.cache.len()
    );

    let truth = [false, true, true, false];
    let inputs = ["0,0", "0,1", "1,0", "1,1"];
    let mut ok = true;
    println!("\n  x1 x2 | out    | target | ok");
    for i in 0..4 {
        let p = pred[i];
        let tt = truth[i];
        let good = (p > 0.5) == tt;
        ok &= good;
        println!("  {} | {p:6.3} | {:>6} | {}", inputs[i], u8::from(tt), if good { "ok" } else { "MISS" });
    }

    println!("\nfinal loss: {last_loss:.6}");
    if sess.cache.misses != first_epoch_misses {
        eprintln!("warning: epochs after the first recompiled {} kernel(s)", sess.cache.misses - first_epoch_misses);
    }
    if !ok {
        eprintln!("XOR training did not converge; try more EPOCHS or a different LR/SEED.");
        std::process::exit(1);
    }
    println!("all 4 XOR cases classified correctly.");
}
