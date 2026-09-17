//! Experiment: XOR with a ReLU hidden layer (2->4->1), matching the tinygrad
//! config that converges quickly (`lr=0.1, 100 epochs`). Diagnostic only.
use autorust::tensor::nn::{self, Linear};
use autorust::tensor::optim::{Optim, SGD};
use autorust::tensor::Tensor;
use autorust::uop::ctx::UOpCtx;

#[test]
fn xor_relu_converges() {
    let (mut w1, mut b1, mut w2, mut b2);
    {
        let mut ctx = UOpCtx::new();
        Tensor::manual_seed(42);
        let l1 = Linear::new(&mut ctx, 2, 4, true);
        let l2 = Linear::new(&mut ctx, 4, 1, true);
        w1 = l1.weight.to_list_fresh::<f32>(&mut ctx);
        b1 = l1.bias.unwrap().to_list_fresh::<f32>(&mut ctx);
        w2 = l2.weight.to_list_fresh::<f32>(&mut ctx);
        b2 = l2.bias.unwrap().to_list_fresh::<f32>(&mut ctx);
    }
    let lr: f32 = std::env::var("LR").ok().and_then(|s| s.parse().ok()).unwrap_or(0.1);
    let n: usize = std::env::var("EPOCHS").ok().and_then(|s| s.parse().ok()).unwrap_or(100);
    let mut pred: Vec<f32> = Vec::new();
    let mut last_loss = 0.0f32;
    for epoch in 0..n {
        let mut ctx = UOpCtx::new();
        let x = Tensor::from_slice(&mut ctx, &[0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], None, None).reshape(&mut ctx, [4, 2]);
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
        let sgd = SGD::new(params.clone(), lr);
        let np = sgd.step_graph(&mut ctx);
        sgd.zero_grad(&ctx);
        w1 = np[0].to_list_fresh::<f32>(&mut ctx);
        b1 = np[1].to_list_fresh::<f32>(&mut ctx);
        w2 = np[2].to_list_fresh::<f32>(&mut ctx);
        b2 = np[3].to_list_fresh::<f32>(&mut ctx);
        pred = out.reshape(&mut ctx, [4]).to_list_fresh::<f32>(&mut ctx);
        last_loss = loss.reshape(&mut ctx, [1]).to_list_fresh::<f32>(&mut ctx)[0];
        if epoch % 10 == 0 || epoch == n - 1 {
            println!("epoch {epoch}: loss = {last_loss:.8}");
        }
    }
    let truth = [false, true, true, false];
    let mut allok = true;
    for (i, (p, tt)) in pred.iter().zip(truth).enumerate() {
        let is_one = *p > 0.5;
        if is_one != tt { allok = false; }
        println!("  out[{i}] = {p} (want {})", if tt { 1 } else { 0 });
    }
    assert!(allok, "relu XOR misclassified; loss={last_loss}");
}
