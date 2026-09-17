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
//! The graph discipline mirrors the test: weights are carried between epochs as
//! plain `Vec<f32>` and each epoch builds a fresh `UOpCtx`. Buffers are
//! arena-local, but the compiled-program cache is content-keyed, so every epoch
//! after the first is a cache hit (no re-lowering / re-rendering / recompiling).

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

    // Draw the starting weights once, then carry them as plain vecs.
    let (mut w1, mut b1, mut w2, mut b2);
    {
        let mut ctx = UOpCtx::new();
        let mut sess = Session::new();
        Tensor::manual_seed(seed);
        let l1 = Linear::new(&mut ctx, 2, 4, true);
        let l2 = Linear::new(&mut ctx, 4, 1, true);
        w1 = l1.weight.to_list_in::<f32>(&mut ctx, &mut sess);
        b1 = l1.bias.unwrap().to_list_in::<f32>(&mut ctx, &mut sess);
        w2 = l2.weight.to_list_in::<f32>(&mut ctx, &mut sess);
        b2 = l2.bias.unwrap().to_list_in::<f32>(&mut ctx, &mut sess);
    }

    let sgd = SGD::new(lr);
    let mut sess = Session::new();
    let mut pred = Vec::new();
    let mut last_loss = f32::NAN;
    let mut first_epoch_misses = 0;
    let log_every = (n / 10).max(1);

    for epoch in 0..n {
        // Fresh arena per step; the session's compiled-kernel cache survives.
        let mut ctx = UOpCtx::new();
        let x = Tensor::from_slice(&mut ctx, &[0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], None, None)
            .reshape(&mut ctx, [4, 2]);
        let t = Tensor::from_slice(&mut ctx, &[0.0f32, 1.0, 1.0, 0.0], None, None).reshape(&mut ctx, [4, 1]);

        let w1t = Tensor::from_slice(&mut ctx, &w1, None, None).reshape(&mut ctx, [4, 2]);
        let b1t = Tensor::from_slice(&mut ctx, &b1, None, None);
        let w2t = Tensor::from_slice(&mut ctx, &w2, None, None).reshape(&mut ctx, [1, 4]);
        let b2t = Tensor::from_slice(&mut ctx, &b2, None, None);

        let h = nn::relu(&Linear { weight: w1t.clone(), bias: Some(b1t.clone()) }.forward(&x, &mut ctx), &mut ctx);
        let out = Linear { weight: w2t.clone(), bias: Some(b2t.clone()) }.forward(&h, &mut ctx);
        let diff = out.sub(&mut ctx, &t);
        let loss = diff.mul(&mut ctx, &diff).mean(&mut ctx, None, false);

        let params = vec![w1t, b1t, w2t, b2t];
        loss.backward(&mut ctx, &params, None);
        let np = sgd.step(&mut ctx, &params);
        sgd.zero_grad(&ctx, &params);

        w1 = np[0].to_list_in::<f32>(&mut ctx, &mut sess);
        b1 = np[1].to_list_in::<f32>(&mut ctx, &mut sess);
        w2 = np[2].to_list_in::<f32>(&mut ctx, &mut sess);
        b2 = np[3].to_list_in::<f32>(&mut ctx, &mut sess);
        pred = out.reshape(&mut ctx, [4]).to_list_in::<f32>(&mut ctx, &mut sess);
        last_loss = loss.reshape(&mut ctx, [1]).to_list_in::<f32>(&mut ctx, &mut sess)[0];

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
