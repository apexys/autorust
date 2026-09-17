//! Regression test for the 9.1/9.2 memory work: a long-lived `UOpCtx` must not
//! grow without bound across repeated training steps.
//!
//! Before the `Rc<Node>` arena + tensor-owned buffers, this loop grew ~1040
//! nodes/step and ~1.9 MB/step (unbounded). Nodes are now freed automatically
//! when the last handle drops, and `BufTable` prunes bases no live caller holds,
//! so the live-node and base counts stay flat.

use autorust::engine::realize::ProgCache;
use autorust::runtime::buffer::BufTable;
use autorust::tensor::nn::{self, Linear};
use autorust::tensor::optim::{Optim, SGD};
use autorust::tensor::Tensor;
use autorust::uop::ctx::UOpCtx;

#[test]
fn single_ctx_training_is_memory_bounded() {
    Tensor::manual_seed(42);
    let mut ctx = UOpCtx::new();
    let mut cache = ProgCache::default();
    let mut bufs = BufTable::default();
    let l1 = Linear::new(&mut ctx, 2, 4, true);
    let l2 = Linear::new(&mut ctx, 4, 1, true);
    let mut w = [
        l1.weight.to_list_with::<f32>(&mut ctx, &mut bufs, &mut cache),
        l1.bias.unwrap().to_list_with::<f32>(&mut ctx, &mut bufs, &mut cache),
        l2.weight.to_list_with::<f32>(&mut ctx, &mut bufs, &mut cache),
        l2.bias.unwrap().to_list_with::<f32>(&mut ctx, &mut bufs, &mut cache),
    ];
    let sgd = SGD::new(0.1);
    let mut baseline: Option<(usize, usize)> = None;
    for i in 0..120 {
        let x = Tensor::from_slice(&mut ctx, &[0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], None, None).reshape(&mut ctx, [4, 2]);
        let t = Tensor::from_slice(&mut ctx, &[0.0f32, 1.0, 1.0, 0.0], None, None).reshape(&mut ctx, [4, 1]);
        let w1t = Tensor::from_slice(&mut ctx, &w[0], None, None).reshape(&mut ctx, [4, 2]);
        let b1t = Tensor::from_slice(&mut ctx, &w[1], None, None);
        let w2t = Tensor::from_slice(&mut ctx, &w[2], None, None).reshape(&mut ctx, [1, 4]);
        let b2t = Tensor::from_slice(&mut ctx, &w[3], None, None);
        let h = nn::relu(&Linear { weight: w1t.clone(), bias: Some(b1t.clone()) }.forward(&x, &mut ctx), &mut ctx);
        let out = Linear { weight: w2t.clone(), bias: Some(b2t.clone()) }.forward(&h, &mut ctx);
        let diff = out.sub(&mut ctx, &t);
        let loss = diff.mul(&mut ctx, &diff).mean(&mut ctx, None, false);
        let params = vec![w1t, b1t, w2t, b2t];
        loss.backward(&mut ctx, &params, None);
        let np = sgd.step(&mut ctx, &params);
        sgd.zero_grad(&ctx, &params);
        w[0] = np[0].to_list_with::<f32>(&mut ctx, &mut bufs, &mut cache);
        w[1] = np[1].to_list_with::<f32>(&mut ctx, &mut bufs, &mut cache);
        w[2] = np[2].to_list_with::<f32>(&mut ctx, &mut bufs, &mut cache);
        w[3] = np[3].to_list_with::<f32>(&mut ctx, &mut bufs, &mut cache);
        let _ = loss.reshape(&mut ctx, [1]).to_list_with::<f32>(&mut ctx, &mut bufs, &mut cache);

        if i == 9 {
            baseline = Some((ctx.g.live_len(), bufs.n_bases()));
        }
    }

    let (base_nodes, base_bases) = baseline.expect("baseline captured at step 10");
    let (nodes, bases) = (ctx.g.live_len(), bufs.n_bases());
    // Nodes may fluctuate by a small constant but must not scale with step count.
    assert!(nodes <= base_nodes + 64, "live nodes grew with steps: {base_nodes} -> {nodes}");
    assert!(bases <= base_bases + 4, "buffer bases grew with steps: {base_bases} -> {bases}");
}
