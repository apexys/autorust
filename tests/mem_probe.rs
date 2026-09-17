//! Regression test for the 9.1/9.2 memory work plus in-place optimizer updates:
//! a long-lived `UOpCtx` with persistent `Linear` parameters updated via
//! `Tensor::assign` must not grow without bound across training steps.
//!
//! Before the `Rc<Node>` arena + `Tensor::assign`, the idiomatic loop
//! (`w = w - lr*grad`) chained each step into the previous one and the live
//! graph grew every step (rebinding an expression), or the parameters had to be
//! round-tripped through host `Vec<f32>` each step. Now each step references the
//! same parameter leaves and the live-node / base counts stay flat.

use autorust::engine::session::Session;
use autorust::tensor::nn::{self, Linear};
use autorust::tensor::optim::{Optim, SGD};
use autorust::tensor::Tensor;
use autorust::uop::ctx::UOpCtx;

#[test]
fn single_ctx_training_is_memory_bounded() {
    Tensor::manual_seed(42);
    let mut ctx = UOpCtx::new();
    let mut sess = Session::new();
    let l1 = Linear::new(&mut ctx, 2, 4, true).into_params(&mut ctx);
    let l2 = Linear::new(&mut ctx, 4, 1, true).into_params(&mut ctx);
    let params = vec![
        l1.weight.clone(),
        l1.bias.clone().unwrap(),
        l2.weight.clone(),
        l2.bias.clone().unwrap(),
    ];
    let sgd = SGD::new(params.clone(), 0.1);

    let x = Tensor::from_slice(&mut ctx, &[0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], None, None)
        .reshape(&mut ctx, [4, 2]);
    let t = Tensor::from_slice(&mut ctx, &[0.0f32, 1.0, 1.0, 0.0], None, None).reshape(&mut ctx, [4, 1]);

    let mut baseline: Option<(usize, usize)> = None;
    for i in 0..120 {
        let h = nn::relu(&l1.forward(&x, &mut ctx), &mut ctx);
        let out = l2.forward(&h, &mut ctx);
        let diff = out.sub(&mut ctx, &t);
        let loss = diff.mul(&mut ctx, &diff).mean(&mut ctx, None, false);

        loss.backward(&mut ctx, &params, None);
        sgd.step(&mut ctx, &mut sess);
        sgd.zero_grad(&ctx);
        let _ = loss.reshape(&mut ctx, [1]).to_list::<f32>(&mut ctx, &mut sess);

        if i == 9 {
            baseline = Some((ctx.g.live_len(), sess.buffers.n_bases()));
        }
    }

    let (base_nodes, base_bases) = baseline.expect("baseline captured at step 10");
    let (nodes, bases) = (ctx.g.live_len(), sess.buffers.n_bases());
    // Nodes may fluctuate by a small constant but must not scale with step count.
    assert!(nodes <= base_nodes + 64, "live nodes grew with steps: {base_nodes} -> {nodes}");
    assert!(bases <= base_bases + 4, "buffer bases grew with steps: {base_bases} -> {bases}");
}
