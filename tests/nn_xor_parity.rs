//! Numeric parity probe: same initial weights as tinygrad seed 42, then a few
//! SGD epochs. Run with TEPS=5 and compare against tg_xor_weights.py.
use autorust::tensor::nn::{self, Linear};
use autorust::tensor::optim::{Optim, SGD};
use autorust::tensor::Tensor;
use autorust::uop::ctx::UOpCtx;

#[test]
#[ignore]
fn xor_parity_tinygrad_seed42() {
    let w1_init: Vec<f32> = vec![0.0472639799118042, 0.2406357377767563, 0.3720497786998749, -0.09608840942382812];
    let b1_init: Vec<f32> = vec![0.0891210287809372, -0.6955669522285461];
    let w2_init: Vec<f32> = vec![0.047621723264455795, -0.17559118568897247];
    let b2_init: Vec<f32> = vec![0.4453694522380829];

    let mut w1 = w1_init.clone();
    let mut b1 = b1_init.clone();
    let mut w2 = w2_init.clone();
    let mut b2 = b2_init.clone();

    let sgd = SGD::new(0.5);
    let n: usize = std::env::var("TEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    for epoch in 0..n {
        let mut ctx = UOpCtx::new();
        let x = Tensor::from_slice(&mut ctx, &[0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], None, None)
            .reshape(&mut ctx, &[4, 2]);
        let t = Tensor::from_slice(&mut ctx, &[0.0f32, 1.0, 1.0, 0.0], None, None).reshape(&mut ctx, &[4, 1]);
        let w1t = Tensor::from_slice(&mut ctx, &w1, None, None).reshape(&mut ctx, &[2, 2]);
        let b1t = Tensor::from_slice(&mut ctx, &b1, None, None);
        let w2t = Tensor::from_slice(&mut ctx, &w2, None, None).reshape(&mut ctx, &[1, 2]);
        let b2t = Tensor::from_slice(&mut ctx, &b2, None, None);

        let h = nn::sigmoid(&Linear { weight: w1t, bias: Some(b1t) }.forward(&x, &mut ctx), &mut ctx);
        let out = Linear { weight: w2t, bias: Some(b2t) }.forward(&h, &mut ctx);
        let diff = out.sub(&mut ctx, &t);
        let sq = diff.mul(&mut ctx, &diff);
        let loss = sq.mean(&mut ctx, None, false);

        if epoch == 0 {
            let h0 = h.reshape(&mut ctx, &[4, 2]).to_list::<f32>(&mut ctx);
            println!("h0 {h0:?}");
            let out0 = out.reshape(&mut ctx, &[4]).to_list::<f32>(&mut ctx);
            println!("out0 {out0:?}");
        }

        let params = vec![w1t, b1t, w2t, b2t];
        loss.backward(&mut ctx, &params, None);
        let np = sgd.step(&mut ctx, &params);
        sgd.zero_grad(&mut ctx);

        w1 = np[0].to_list::<f32>(&mut ctx);
        b1 = np[1].to_list::<f32>(&mut ctx);
        w2 = np[2].to_list::<f32>(&mut ctx);
        b2 = np[3].to_list::<f32>(&mut ctx);
        let l = loss.reshape(&mut ctx, &[1]).to_list::<f32>(&mut ctx)[0];
        println!("epoch {epoch}: loss = {l:.8}");
    }
}

#[test]
fn dot_probe() {
    let mut ctx = UOpCtx::new();
    // compute h from inputs (chained), like the real forward
    let x = Tensor::from_slice(&mut ctx, &[0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], None, None).reshape(&mut ctx, &[4, 2]);
    let w1 = Tensor::from_slice(&mut ctx, &[0.0472639799118042f32, 0.2406357377767563, 0.3720497786998749, -0.09608840942382812], None, None).reshape(&mut ctx, &[2, 2]);
    let b1 = Tensor::from_slice(&mut ctx, &[0.0891210287809372f32, -0.6955669522285461], None, None);
    let h = nn::sigmoid(&Linear { weight: w1, bias: Some(b1) }.forward(&x, &mut ctx), &mut ctx);
    println!("h {:?}", h.reshape(&mut ctx, &[4, 2]).to_list::<f32>(&mut ctx));

    let w = Tensor::from_slice(&mut ctx, &[0.047621723264455795f32, -0.17559118568897247], None, None)
        .reshape(&mut ctx, &[1, 2]);
    let b = Tensor::from_slice(&mut ctx, &[0.4453694522380829f32], None, None);
    let wt = w.transpose(&mut ctx, 0, 1);
    let o = h.linear(&mut ctx, &wt, Some(&b));
    println!("dot_probe {:?}", o.reshape(&mut ctx, &[4]).to_list::<f32>(&mut ctx));
    let hv = h.reshape(&mut ctx, &[4, 2]).to_list::<f32>(&mut ctx);
    let h2 = Tensor::from_slice(&mut ctx, &hv, None, None).reshape(&mut ctx, &[4, 2]);
    let o2 = h2.linear(&mut ctx, &wt, Some(&b));
    println!("dot_probe_copy {:?}", o2.reshape(&mut ctx, &[4]).to_list::<f32>(&mut ctx));
    println!("expected [0.41180462, 0.41831934, 0.39708504, 0.40397506]");
}
