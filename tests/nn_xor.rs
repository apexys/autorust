//! End-to-end NN training: a 2->4->1 ReLU MLP on XOR, trained with SGD.
//! Functional golden — after training the outputs must classify XOR correctly
//! (out>0.5 matches the truth table) and the loss must drop sharply.
//!
//! Notes on the graph discipline here:
//! - Weights are carried between epochs as plain `Vec<f32>` and each epoch builds
//!   a fresh `UOpCtx` (tinygrad realizes + mutates buffers; keeping the arena
//!   bounded per step avoids O(n^2) graph growth). The math is unchanged.
//! - Repeated realization is cheap now that `to_flat` serializes only the nodes
//!   reachable from the current kernel's roots, so each param is read out
//!   individually instead of `cat`-ing uneven shapes together.
//! - Config matches the tinygrad reference sweep that converges reliably
//!   (`2->4` ReLU hidden, linear output, MSE, `lr=0.1`, seed 42, ~300 epochs).

use autorust::engine::realize::ProgCache;
use autorust::engine::session::Session;
use autorust::runtime::buffer::BufTable;
use autorust::tensor::nn::{self, Linear};
use autorust::tensor::optim::{Optim, SGD};
use autorust::tensor::Tensor;
use autorust::uop::ctx::UOpCtx;

/// Read all four params out (flat f32: w1, b1, w2, b2) as separate realizations.
fn read_params(ctx: &mut UOpCtx, bufs: &mut BufTable, cache: &mut ProgCache, w1: Tensor, b1: Tensor, w2: Tensor, b2: Tensor) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    (
        w1.to_list_with::<f32>(ctx, bufs, cache),
        b1.to_list_with::<f32>(ctx, bufs, cache),
        w2.to_list_with::<f32>(ctx, bufs, cache),
        b2.to_list_with::<f32>(ctx, bufs, cache),
    )
}

#[test]
fn xor_trains_to_convergence() {
    // Initialization RNG: draw the starting weights once (single realization).
    let (mut w1, mut b1, mut w2, mut b2);
    let mut cache = ProgCache::default();
    {
        let mut ctx = UOpCtx::new();
        let mut bufs = BufTable::default();
        Tensor::manual_seed(42);
        let l1 = Linear::new(&mut ctx, 2, 4, true);
        let l2 = Linear::new(&mut ctx, 4, 1, true);
        (w1, b1, w2, b2) = read_params(&mut ctx, &mut bufs, &mut cache, l1.weight, l1.bias.unwrap(), l2.weight, l2.bias.unwrap());
    }

    let lr: f32 = std::env::var("LR").ok().and_then(|s| s.parse().ok()).unwrap_or(0.1);
    let sgd = SGD::new(lr);
    let mut last_loss = 0.0f32;
    let mut pred: Vec<f32> = Vec::new();

    let n: usize = std::env::var("EPOCHS").ok().and_then(|s| s.parse().ok()).unwrap_or(300);
    let mut first_epoch_misses: Option<u64> = None;
    for epoch in 0..n {
        let mut ctx = UOpCtx::new();
        // Buffers are arena-local (fresh per step, see CURRENT_STATE.md §9.1/§9.2),
        // but the compiled-program cache is content-keyed, so every epoch after the
        // first is a cache hit: the kernels are not re-lowered/re-rendered/re-compiled.
        let mut bufs = BufTable::default();
        let x = Tensor::from_slice(&mut ctx, &[0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], None, None)
            .reshape(&mut ctx, &[4, 2]);
        let t = Tensor::from_slice(&mut ctx, &[0.0f32, 1.0, 1.0, 0.0], None, None).reshape(&mut ctx, &[4, 1]);

        let w1t = Tensor::from_slice(&mut ctx, &w1, None, None).reshape(&mut ctx, &[4, 2]);
        let b1t = Tensor::from_slice(&mut ctx, &b1, None, None);
        let w2t = Tensor::from_slice(&mut ctx, &w2, None, None).reshape(&mut ctx, &[1, 4]);
        let b2t = Tensor::from_slice(&mut ctx, &b2, None, None);

        let h = nn::relu(&Linear { weight: w1t, bias: Some(b1t) }.forward(&x, &mut ctx), &mut ctx);
        let out = Linear { weight: w2t, bias: Some(b2t) }.forward(&h, &mut ctx);
        let diff = out.sub(&mut ctx, &t);
        let sq = diff.mul(&mut ctx, &diff);
        let loss = sq.mean(&mut ctx, None, false);

        let params = vec![w1t, b1t, w2t, b2t];
        loss.backward(&mut ctx, &params, None);
        let np = sgd.step(&mut ctx, &params);
        sgd.zero_grad(&mut ctx);

        // Read the realized params + final out + loss out of the graph.
        w1 = np[0].to_list_with::<f32>(&mut ctx, &mut bufs, &mut cache);
        b1 = np[1].to_list_with::<f32>(&mut ctx, &mut bufs, &mut cache);
        w2 = np[2].to_list_with::<f32>(&mut ctx, &mut bufs, &mut cache);
        b2 = np[3].to_list_with::<f32>(&mut ctx, &mut bufs, &mut cache);
        pred = out.reshape(&mut ctx, &[4]).to_list_with::<f32>(&mut ctx, &mut bufs, &mut cache);
        last_loss = loss.reshape(&mut ctx, &[1]).to_list_with::<f32>(&mut ctx, &mut bufs, &mut cache)[0];

        if epoch % (n / 10).max(1) == 0 || epoch == n - 1 {
            println!("epoch {epoch}: loss = {last_loss}");
        }
        if epoch == 0 {
            first_epoch_misses = Some(cache.misses);
        }
    }
    println!("to_program: {} misses / {} hits ({} cached)", cache.misses, cache.hits, cache.len());

    let truth = [false, true, true, false];
    for (i, (p, tt)) in pred.iter().zip(truth).enumerate() {
        let is_one = *p > 0.5;
        assert_eq!(is_one, tt, "xor misclassified at {i}: out={p}");
    }
    assert!(last_loss < 0.05, "loss did not drop enough: {last_loss}");
    // Every step builds an identical-shaped graph, so no epoch after the first may
    // compile a new kernel (the point of 9.1's persistent, content-keyed ProgCache).
    assert_eq!(cache.misses, first_epoch_misses.unwrap(), "epochs after the first must not recompile");
    assert!(cache.hits > cache.misses * 10, "expected heavy program-cache reuse, got {} hits / {} misses", cache.hits, cache.misses);
}

#[test]
fn micro_seq_kaiming_timed() {
    use std::time::Instant;
    let mut ctx = UOpCtx::new();
    Tensor::manual_seed(1);
    for i in 0..4 {
        let t = Tensor::kaiming_uniform(&mut ctx, &[2,2], (5.0f64).sqrt(), None);
        let n_before = ctx.g.nodes.len();
        let t0 = Instant::now();
        let v = t.to_list::<f32>(&mut ctx);
        eprintln!("[t] i={i} before={n_before} dt={:?}", t0.elapsed());
        let _ = v;
    }
    eprintln!("[t] final nodes={}", ctx.g.nodes.len());
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
        let before = ctx.g.nodes.len();
        let t0 = Instant::now();
        let v = c.to_list_in::<f32>(&mut ctx, &mut sess);
        eprintln!(
            "[r] i={i} before={before} after={} bases={} hits={} misses={} dt={:?} v={v:?}",
            ctx.g.nodes.len(),
            sess.buffers.n_bases(),
            sess.cache.hits,
            sess.cache.misses,
            t0.elapsed()
        );
    }
}
