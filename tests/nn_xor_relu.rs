//! Experiment: XOR with a ReLU hidden layer (2->4->1), matching the tinygrad
//! config that converges quickly (`lr=0.1, 100 epochs`). Diagnostic only.
use autorust::tensor::nn::{self, Linear};
use autorust::tensor::optim::{Optim, SGD};
use autorust::tensor::{Ctx, Tensor};

#[test]
fn xor_relu_converges() {
    let (mut w1, mut b1, mut w2, mut b2);
    {
        let ctx = Ctx::new();
        Tensor::manual_seed(42);
        let l1 = Linear::new(&ctx, 2, 4, true);
        let l2 = Linear::new(&ctx, 4, 1, true);
        w1 = l1.weight.to_list_fresh::<f32>();
        b1 = l1.bias.unwrap().to_list_fresh::<f32>();
        w2 = l2.weight.to_list_fresh::<f32>();
        b2 = l2.bias.unwrap().to_list_fresh::<f32>();
    }
    let lr: f32 = std::env::var("LR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.1);
    let n: usize = std::env::var("EPOCHS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    let mut pred: Vec<f32> = Vec::new();
    let mut last_loss = 0.0f32;
    for epoch in 0..n {
        let ctx = Ctx::new();
        let x = Tensor::from_slice(
            &ctx,
            &[0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            None,
            None,
        )
        .reshape([4, 2]);
        let t = Tensor::from_slice(&ctx, &[0.0f32, 1.0, 1.0, 0.0], None, None).reshape([4, 1]);
        let w1t = Tensor::from_slice(&ctx, &w1, None, None).reshape([4, 2]);
        let b1t = Tensor::from_slice(&ctx, &b1, None, None);
        let w2t = Tensor::from_slice(&ctx, &w2, None, None).reshape([1, 4]);
        let b2t = Tensor::from_slice(&ctx, &b2, None, None);
        let h = nn::relu(
            &Linear {
                weight: w1t.clone(),
                bias: Some(b1t.clone()),
            }
            .forward(&x),
        );
        let out = Linear {
            weight: w2t.clone(),
            bias: Some(b2t.clone()),
        }
        .forward(&h);
        let diff = out.sub(&t);
        let loss = diff.mul(&diff).mean(None, false);
        let params = vec![w1t, b1t, w2t, b2t];
        loss.backward(&params, None);
        let sgd = SGD::new(params.clone(), lr);
        let np = sgd.step_graph(&mut ctx.raw());
        sgd.zero_grad();
        w1 = np[0].to_list_fresh::<f32>();
        b1 = np[1].to_list_fresh::<f32>();
        w2 = np[2].to_list_fresh::<f32>();
        b2 = np[3].to_list_fresh::<f32>();
        pred = out.reshape([4]).to_list_fresh::<f32>();
        last_loss = loss.reshape([1]).to_list_fresh::<f32>()[0];
        if epoch % 10 == 0 || epoch == n - 1 {
            println!("epoch {epoch}: loss = {last_loss:.8}");
        }
    }
    let truth = [false, true, true, false];
    let mut allok = true;
    for (i, (p, tt)) in pred.iter().zip(truth).enumerate() {
        let is_one = *p > 0.5;
        if is_one != tt {
            allok = false;
        }
        println!("  out[{i}] = {p} (want {})", if tt { 1 } else { 0 });
    }
    assert!(allok, "relu XOR misclassified; loss={last_loss}");
}
