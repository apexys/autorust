//! Numeric parity probe: same initial weights as tinygrad seed 42, then a few
//! SGD epochs. Run with TEPS=5 and compare against tg_xor_weights.py.
use autorust::tensor::nn::{self, Linear};
use autorust::tensor::optim::{Optim, SGD};
use autorust::tensor::{Ctx, Tensor};
use autorust::uop::ctx::UOpCtx;

#[test]
#[ignore]
fn xor_parity_tinygrad_seed42() {
    let w1_init: Vec<f32> = vec![0.047_263_98, 0.240_635_74, 0.372_049_78, -0.096_088_41];
    let b1_init: Vec<f32> = vec![0.089_121_03, -0.695_566_95];
    let w2_init: Vec<f32> = vec![0.047_621_723, -0.175_591_19];
    let b2_init: Vec<f32> = vec![0.445_369_45];

    let mut w1 = w1_init.clone();
    let mut b1 = b1_init.clone();
    let mut w2 = w2_init.clone();
    let mut b2 = b2_init.clone();

    let sgd_lr = 0.5;
    let n: usize = std::env::var("TEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    for epoch in 0..n {
        let mut ctx = Ctx::new();
        let x = Tensor::from_slice(&ctx, &[0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], None, None)
            .reshape([4, 2]);
        let t = Tensor::from_slice(&ctx, &[0.0f32, 1.0, 1.0, 0.0], None, None).reshape([4, 1]);
        let w1t = Tensor::from_slice(&ctx, &w1, None, None).reshape([2, 2]);
        let b1t = Tensor::from_slice(&ctx, &b1, None, None);
        let w2t = Tensor::from_slice(&ctx, &w2, None, None).reshape([1, 2]);
        let b2t = Tensor::from_slice(&ctx, &b2, None, None);

        let h = nn::sigmoid(&Linear { weight: w1t.clone(), bias: Some(b1t.clone()) }.forward(&x));
        let out = Linear { weight: w2t.clone(), bias: Some(b2t.clone()) }.forward(&h);
        let diff = out.sub(&t);
        let sq = diff.mul(&diff);
        let loss = sq.mean(None, false);

        if epoch == 0 {
            let h0 = h.reshape([4, 2]).to_list_fresh::<f32>();
            println!("h0 {h0:?}");
            let out0 = out.reshape([4]).to_list_fresh::<f32>();
            println!("out0 {out0:?}");
        }

        let params = vec![w1t, b1t, w2t, b2t];
        loss.backward(&params, None);
        let sgd = SGD::new(params.clone(), sgd_lr);
        let np = sgd.step_graph(&mut ctx.raw());
        sgd.zero_grad();

        w1 = np[0].to_list_fresh::<f32>();
        b1 = np[1].to_list_fresh::<f32>();
        w2 = np[2].to_list_fresh::<f32>();
        b2 = np[3].to_list_fresh::<f32>();
        let l = loss.reshape([1]).to_list_fresh::<f32>()[0];
        println!("epoch {epoch}: loss = {l:.8}");
    }
}

#[test]
fn dot_probe() {
    let mut ctx = Ctx::new();
    // compute h from inputs (chained), like the real forward
    let x = Tensor::from_slice(&ctx, &[0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], None, None).reshape([4, 2]);
    let w1 = Tensor::from_slice(&ctx, &[0.047_263_98_f32, 0.240_635_74, 0.372_049_78, -0.096_088_41], None, None).reshape([2, 2]);
    let b1 = Tensor::from_slice(&ctx, &[0.089_121_03_f32, -0.695_566_95], None, None);
    let h = nn::sigmoid(&Linear { weight: w1, bias: Some(b1) }.forward(&x));
    println!("h {:?}", h.reshape([4, 2]).to_list_fresh::<f32>());

    let w = Tensor::from_slice(&ctx, &[0.047_621_723_f32, -0.175_591_19], None, None)
        .reshape([1, 2]);
    let b = Tensor::from_slice(&ctx, &[0.445_369_45_f32], None, None);
    let wt = w.transpose(0, 1);
    let o = h.linear(&wt, Some(&b));
    let ov = o.reshape([4]).to_list_fresh::<f32>();
    println!("dot_probe {ov:?}");
    let hv = h.reshape([4, 2]).to_list_fresh::<f32>();
    let h2 = Tensor::from_slice(&ctx, &hv, None, None).reshape([4, 2]);
    let o2 = h2.linear(&wt, Some(&b));
    let o2v = o2.reshape([4]).to_list_fresh::<f32>();
    println!("dot_probe_copy {o2v:?}");

    let expected = [0.41180462f32, 0.41831934, 0.39708504, 0.40397506];
    println!("expected {expected:?}");
    for (i, want) in expected.iter().enumerate() {
        assert!((ov[i] - want).abs() < 1e-6, "dot_probe[{i}]: {} != {want}", ov[i]);
        assert!((o2v[i] - want).abs() < 1e-6, "dot_probe_copy[{i}]: {} != {want}", o2v[i]);
    }
}
