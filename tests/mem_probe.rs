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
use autorust::tensor::nn::{self, Linear, Module};
use autorust::tensor::optim::{Optim, SGD};
use autorust::tensor::{Ctx, Tensor};

#[test]
fn single_ctx_training_is_memory_bounded() {
    Tensor::manual_seed(42);
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let l1 = Linear::new(&ctx, 2, 4, true).into_params();
    let l2 = Linear::new(&ctx, 4, 1, true).into_params();
    let params = vec![
        l1.weight.clone(),
        l1.bias.clone().unwrap(),
        l2.weight.clone(),
        l2.bias.clone().unwrap(),
    ];
    let sgd = SGD::new(params.clone(), 0.1);

    let x = Tensor::from_slice(
        &ctx,
        &[0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        None,
        None,
    )
    .reshape([4, 2]);
    let t = Tensor::from_slice(&ctx, &[0.0f32, 1.0, 1.0, 0.0], None, None).reshape([4, 1]);

    let mut baseline: Option<(usize, usize)> = None;
    for i in 0..120 {
        let h = nn::relu(&l1.forward(&x));
        let out = l2.forward(&h);
        let diff = out.sub(&t);
        let loss = diff.mul(&diff).mean(None, false);

        loss.backward(&params, None);
        sgd.step(&mut sess);
        sgd.zero_grad();
        let _ = loss.reshape([1]).to_list::<f32>(&mut sess);

        if i == 9 {
            baseline = Some((ctx.raw().g.live_len(), sess.buffers.n_bases()));
        }
    }

    let (base_nodes, base_bases) = baseline.expect("baseline captured at step 10");
    let (nodes, bases) = (ctx.raw().g.live_len(), sess.buffers.n_bases());
    // Nodes may fluctuate by a small constant but must not scale with step count.
    assert!(
        nodes <= base_nodes + 64,
        "live nodes grew with steps: {base_nodes} -> {nodes}"
    );
    assert!(
        bases <= base_bases + 4,
        "buffer bases grew with steps: {base_bases} -> {bases}"
    );
}
