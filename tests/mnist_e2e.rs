//! MNIST end-to-end acceptance gate — **ignored by default**. It downloads the
//! IDX files and runs one full train step of the `beautiful_mnist` model, which
//! is slow on the interpreter backend.
//!
//! ```sh
//! DEFAULT_DEVICE=CL cargo test --release --test mnist_e2e -- --ignored --nocapture
//! ```
//!
//! The model/data-loader live in `examples/mnist.rs`; this test includes that
//! file as a module so the example and the gate cannot drift.

#[allow(dead_code)]
#[path = "../examples/mnist.rs"]
mod mnist;

use autorust::engine::session::Session;
use autorust::tensor::nn::Module;
use autorust::tensor::optim::{Adam, Optim};
use autorust::tensor::Tensor;
use autorust::uop::ctx::UOpCtx;

use mnist::{fetch, get_test_acc, load_images, load_labels, mnist_dir, Model};

#[test]
#[ignore = "downloads MNIST and runs a train step (slow on INTERP)"]
fn mnist_one_step_and_eval() {
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

    let mut ctx = UOpCtx::new();
    let mut sess = Session::new();
    let n = 4;
    let x = load_images(&mut ctx, &dir.join("train-images-idx3-ubyte"), n);
    let y = load_labels(&mut ctx, &dir.join("train-labels-idx1-ubyte"), n);
    assert_eq!(x.shape(&mut ctx).iter().map(|s| s.clone().as_usize().unwrap()).collect::<Vec<_>>(), vec![n as usize, 1, 28, 28]);
    assert_eq!(y.shape(&mut ctx).iter().map(|s| s.clone().as_usize().unwrap()).collect::<Vec<_>>(), vec![n as usize]);

    Tensor::manual_seed(0);
    let model = Model::new(&mut ctx).into_params(&mut ctx);
    let params: Vec<Tensor> = model.params().collect();
    let opt = Adam::new(&mut ctx, params.clone(), 1e-3);
    let before = params[0].to_list::<f32>(&mut ctx, &mut sess);

    let out = model.forward(&x, &mut ctx, &mut sess, true);
    let loss = out.sparse_categorical_crossentropy(&mut ctx, &y, -1, 0.0, "mean");
    loss.backward(&mut ctx, &params, None);
    opt.schedule_step(&mut ctx, &mut sess, std::slice::from_ref(&loss));
    opt.zero_grad(&ctx);

    let l = loss.item::<f32>(&mut ctx, &mut sess);
    assert!(l.is_finite(), "loss is not finite: {l}");
    let after = params[0].to_list::<f32>(&mut ctx, &mut sess);
    assert_ne!(before, after, "the first parameter was not updated by the optimizer");

    let acc = get_test_acc(&model, &x, &y, &mut ctx, &mut sess);
    assert!((0.0..=100.0).contains(&acc), "accuracy out of range: {acc}");
}
