//! End-to-end NN training: a 2->4->1 ReLU MLP on XOR, trained with SGD.
//! Functional golden — after training the outputs must classify XOR correctly
//! (out>0.5 matches the truth table) and the loss must drop sharply.
//!
//! Notes on the graph discipline here:
//! - The `Linear` layers are created once and their parameters are turned into
//!   buffer-backed leaves (`into_params`); the optimizer updates those buffers
//!   in place with `Tensor::assign` (tinygrad `Optimizer.step`). So each step's
//!   graph references the *same* parameter leaves and is structurally identical
//!   — the graph does not chain into previous steps. See CURRENT_STATE §8.10.
//! - Because the graph is identical each step, every step after the first is a
//!   program-cache hit: kernels are not re-lowered/re-rendered/re-compiled.
//! - Config matches the tinygrad reference sweep that converges reliably
//!   (`2->4` ReLU hidden, linear output, MSE, `lr=0.1`, seed 42, ~300 epochs).

use autorust::engine::session::Session;
use autorust::tensor::nn::{self, Linear};
use autorust::tensor::optim::{Optim, SGD};
use autorust::tensor::Tensor;
use autorust::uop::ctx::UOpCtx;

#[test]
fn xor_trains_to_convergence() {
    let mut ctx = UOpCtx::new();
    let mut sess = Session::new();

    // Draw the initial weights once and turn them into buffer-backed leaves so
    // the optimizer can assign into them in place.
    Tensor::manual_seed(42);
    let l1 = Linear::new(&mut ctx, 2, 4, true).into_params(&mut ctx);
    let l2 = Linear::new(&mut ctx, 4, 1, true).into_params(&mut ctx);

    // Confirm which backend actually holds the parameters (env DEFAULT_DEVICE >
    // CL when compiled+present > INTERP). `to_param` stages each leaf on the
    // device its init was realized on.
    eprintln!(
        "[xor] default device = {}, l1.weight = {:?}, l1.bias = {:?}, l2.weight = {:?}",
        autorust::tensor::creation::default_device(),
        l1.weight.device(&ctx),
        l1.bias.as_ref().and_then(|b| b.device(&ctx)),
        l2.weight.device(&ctx),
    );

    let lr: f32 = std::env::var("LR").ok().and_then(|s| s.parse().ok()).unwrap_or(0.1);
    let params = vec![
        l1.weight.clone(),
        l1.bias.clone().unwrap(),
        l2.weight.clone(),
        l2.bias.clone().unwrap(),
    ];
    let sgd = SGD::new(params.clone(), lr);

    // The inputs are constant; the graph is reused every step.
    let x = Tensor::from_slice(&mut ctx, &[0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], None, None)
        .reshape(&mut ctx, [4, 2]);
    let t = Tensor::from_slice(&mut ctx, &[0.0f32, 1.0, 1.0, 0.0], None, None).reshape(&mut ctx, [4, 1]);

    let n: usize = std::env::var("EPOCHS").ok().and_then(|s| s.parse().ok()).unwrap_or(300);
    let mut first_epoch_misses: Option<u64> = None;
    let mut last_loss = 0.0f32;
    let mut pred: Vec<f32> = Vec::new();

    for epoch in 0..n {
        let h = nn::relu(&l1.forward(&x, &mut ctx), &mut ctx);
        let out = l2.forward(&h, &mut ctx);
        let diff = out.sub(&mut ctx, &t);
        let loss = diff.mul(&mut ctx, &diff).mean(&mut ctx, None, false);

        loss.backward(&mut ctx, &params, None);
        // `step` assigns into the parameter buffers; the leaves are unchanged,
        // so the next step's graph is identical.
        sgd.step(&mut ctx, &mut sess);
        sgd.zero_grad(&ctx);

        pred = out.reshape(&mut ctx, [4]).to_list::<f32>(&mut ctx, &mut sess);
        last_loss = loss.reshape(&mut ctx, [1]).to_list::<f32>(&mut ctx, &mut sess)[0];

        if epoch % (n / 10).max(1) == 0 || epoch == n - 1 {
            println!("epoch {epoch}: loss = {last_loss}");
        }
        if epoch == 0 {
            first_epoch_misses = Some(sess.cache.misses);
        }
    }
    println!("to_program: {} misses / {} hits ({} cached)", sess.cache.misses, sess.cache.hits, sess.cache.len());

    let truth = [false, true, true, false];
    for (i, (p, tt)) in pred.iter().zip(truth).enumerate() {
        let is_one = *p > 0.5;
        assert_eq!(is_one, tt, "xor misclassified at {i}: out={p}");
    }
    assert!(last_loss < 0.05, "loss did not drop enough: {last_loss}");
    // Every step builds an identical-shaped graph, so no epoch after the first may
    // compile a new kernel (the point of 9.1's persistent, content-keyed ProgCache).
    assert_eq!(sess.cache.misses, first_epoch_misses.unwrap(), "epochs after the first must not recompile");
    assert!(sess.cache.hits > sess.cache.misses * 10, "expected heavy program-cache reuse, got {} hits / {} misses", sess.cache.hits, sess.cache.misses);
}

#[test]
fn micro_seq_kaiming_timed() {
    use std::time::Instant;
    let mut ctx = UOpCtx::new();
    Tensor::manual_seed(1);
    for i in 0..4 {
        let t = Tensor::kaiming_uniform(&mut ctx, &[2,2], (5.0f64).sqrt(), None);
        let n_before = ctx.g.live_len();
        let t0 = Instant::now();
        let v = t.to_list_fresh::<f32>(&mut ctx);
        eprintln!("[t] i={i} before={n_before} dt={:?}", t0.elapsed());
        let _ = v;
    }
    eprintln!("[t] final nodes={}", ctx.g.live_len());
}

#[test]
fn micro_repeated_realize() {
    use std::time::Instant;
    let mut ctx = UOpCtx::new();
    let mut sess = Session::new();
    for i in 0..7 {
        let a = Tensor::from_slice(&mut ctx, &[1.0f32, 2.0, 3.0], None, None);
        let b = Tensor::from_slice(&mut ctx, &[4.0f32, 5.0, 6.0], None, None);
        let c = a.add(&mut ctx, &b);
        let before = ctx.g.live_len();
        let t0 = Instant::now();
        let v = c.to_list::<f32>(&mut ctx, &mut sess);
        eprintln!(
            "[r] i={i} before={before} after={} bases={} hits={} misses={} dt={:?} v={v:?}",
            ctx.g.live_len(),
            sess.buffers.n_bases(),
            sess.cache.hits,
            sess.cache.misses,
            t0.elapsed()
        );
    }
}
