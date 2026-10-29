//! MNIST-ViT end-to-end acceptance gate — **ignored by default**. Downloads the
//! IDX files and runs one full ViT train step (slow on the interpreter backend).
//!
//! ```sh
//! DEFAULT_DEVICE=CL cargo test --release --test vit_e2e -- --ignored --nocapture
//! ```
//!
//! The model/data-loader live in `examples/vit_mnist.rs`; this test includes that
//! file as a module so the example and the gate cannot drift.

#[allow(dead_code)]
#[path = "../examples/vit_mnist.rs"]
mod vit_mnist;

use autorust::engine::session::Session;
use autorust::models::vit::ViT;
use autorust::tensor::nn::Module;
use autorust::tensor::optim::{Adam, Optim};
use autorust::tensor::{Ctx, Tensor};
use autorust::uop::ctx::UOpCtx;

use vit_mnist::{fetch, load_images, load_labels, mnist_dir, prep};

#[test]
#[ignore = "downloads MNIST and runs a ViT train step (slow on INTERP)"]
fn vit_one_step_and_eval() {
    let dir = mnist_dir();
    let base = "https://storage.googleapis.com/cvdf-datasets/mnist/";
    for (gz, raw) in [
        ("train-images-idx3-ubyte.gz", "train-images-idx3-ubyte"),
        ("train-labels-idx1-ubyte.gz", "train-labels-idx1-ubyte"),
        ("t10k-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte"),
        ("t10k-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte"),
    ] {
        fetch(&format!("{base}{gz}"), &dir.join(raw));
    }

    let mut ctx = Ctx::new();
    let mut sess = Session::new();
    let n = 4;
    let x = load_images(&mut ctx.raw(), &mut sess, &dir.join("train-images-idx3-ubyte"), n);
    let y = load_labels(&mut ctx.raw(), &mut sess, &dir.join("train-labels-idx1-ubyte"), n);

    Tensor::manual_seed(0);
    let size = 16;
    let model = ViT::build(&mut ctx.raw(), 1, 16, 2, 3, size, 8, 10).into_params();
    let params: Vec<Tensor> = model.params().collect();
    let opt = Adam::new(params.clone(), 1e-3);
    let before = params[0].to_list::<f32>(&mut sess);

    sess.training(true);
    let xp = prep(&x, size, &mut ctx.raw());
    let out = model.forward(&xp);
    assert_eq!(out.shape().as_slice().iter().map(|s| s.clone().as_i64().unwrap()).collect::<Vec<_>>(), vec![n, 10]);
    let loss = out.sparse_categorical_crossentropy(&y, -1, 0.0, "mean");
    loss.backward(&params, None);
    opt.schedule_step(&mut sess, std::slice::from_ref(&loss));
    opt.zero_grad();

    let l = loss.item::<f32>(&mut sess);
    assert!(l.is_finite(), "loss is not finite: {l}");
    let after = params[0].to_list::<f32>(&mut sess);
    assert_ne!(before, after, "the first parameter was not updated by the optimizer");

    // eval mode (dropout off) must run
    sess.training(false);
    let xp = prep(&x, size, &mut ctx.raw());
    let eval_out = model.forward(&xp);
    let acc = eval_out.argmax(Some(1), false).eq(&y).cast(autorust::dtype::FLOAT32).mean(None, false);
    let acc = acc.item::<f32>(&mut sess);
    assert!((0.0..=100.0).contains(&acc), "accuracy out of range: {acc}");
}
