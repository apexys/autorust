//! Train a Vision Transformer on MNIST, ported from tinygrad's
//! `extra/models/vit.py` + the `beautiful_mnist.py` training loop.
//!
//! MNIST is 28x28 grayscale; the ViT wants a square RGB image, so we normalize
//! to [0,1], bilinearly resize to `SIZE`, and replicate 1 -> 3 channels
//! (`Tensor.interpolate` + `repeat_interleave`). The config defaults to a small
//! ViT (`LAYERS=2 EMBED=64 HEADS=4 SIZE=32 PATCH=8`) so a run finishes quickly;
//! `LAYERS=12 EMBED=768 HEADS=12 SIZE=224 PATCH=16` gives ViT-B/16.
//!
//! ```sh
//! cargo run --release --example vit_mnist
//! FAST=1 cargo run --release --example vit_mnist
//! FA=1 cargo run --release --example vit_mnist   # FlashAttention attention
//! LAYERS=12 EMBED=768 HEADS=12 SIZE=224 PATCH=16 STEPS=20 TRAIN_N=2000 \
//!   cargo run --release --example vit_mnist
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;

use autorust::dtype;
use autorust::engine::session::Session;
use autorust::models::vit::ViT;
use autorust::tensor::nn::Module;
use autorust::tensor::optim::{Adam, Optim};
use autorust::tensor::{Ctx, Ix, Tensor};
use autorust::uop::ctx::UOpCtx;

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

pub fn mnist_dir() -> PathBuf {
    std::env::var("MNIST_DIR").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("/tmp/opencode/mnist"))
}

/// Download `url` through `sh -c 'curl … | gunzip > path'` unless it's cached.
pub fn fetch(url: &str, path: &Path) {
    if path.exists() {
        return;
    }
    std::fs::create_dir_all(path.parent().expect("cache dir")).expect("create cache dir");
    println!("downloading {} -> {}", url, path.display());
    let status = Command::new("sh")
        .arg("-c")
        .arg(format!("curl -fsSL '{url}' | gunzip > '{}'", path.display()))
        .status()
        .expect("spawn sh/curl");
    assert!(status.success(), "download failed: {url}");
}

pub fn load_images(ctx: &mut UOpCtx, sess: &mut Session, path: &Path, n: i64) -> Tensor {
    let bytes = std::fs::read(path).expect("read idx images");
    let x = Tensor::from_slice(ctx, &bytes[0x10..], Some(dtype::UINT8), None).reshape([-1, 1, 28, 28]);
    let x = if n < 60_000 { x.getitem(&[Ix::slice(Some(0), Some(n), 1)]) } else { x };
    x.realize(sess).clone()
}

pub fn load_labels(ctx: &mut UOpCtx, sess: &mut Session, path: &Path, n: i64) -> Tensor {
    let bytes = std::fs::read(path).expect("read idx labels");
    let y = Tensor::from_slice(ctx, &bytes[8..], Some(dtype::UINT8), None);
    let y = if n < 60_000 { y.getitem(&[Ix::slice(Some(0), Some(n), 1)]) } else { y };
    y.realize(sess).clone()
}

/// 28x28 uint8 -> `(N,3,SIZE,SIZE)` f32 in [0,1]: normalize, bilinear resize,
/// replicate the single channel to RGB.
pub fn prep(x: &Tensor, size: i64, ctx: &mut UOpCtx) -> Tensor {
    let x = x.cast(dtype::FLOAT32).div(255.0f64);
    let x = x.interpolate(&[size, size], "linear", false);
    x.repeat_interleave(3, Some(1))
}

/// `(model(X).argmax(1) == Y).mean() * 100`, read back as a percentage.
fn test_acc(model: &ViT, x: &Tensor, y: &Tensor, size: i64, ctx: &mut UOpCtx, sess: &mut Session) -> f32 {
    let xp = prep(x, size, ctx);
    let out = model.forward(&xp);
    let pred = out.argmax(Some(1), false);
    let correct = pred.eq(y).cast(dtype::FLOAT32).mean(None, false);
    correct.mul(100.0).item::<f32>(sess)
}

fn main() {
    if let Ok(dt) = std::env::var("DTYPE") {
        // SAFETY: single-threaded; set before any `default_float()` read.
        unsafe { std::env::set_var("DEFAULT_FLOAT", dt) };
    }
    let fast = std::env::var("FAST").is_ok();

    let bs: i64 = env_or("BS", if fast { 16 } else { 64 });
    let steps: i64 = env_or("STEPS", if fast { 5 } else { 60 });
    let train_n: i64 = env_or("TRAIN_N", if fast { 512 } else { 20_000 });
    let test_n: i64 = env_or("TEST_N", if fast { 128 } else { 1000 });
    let lr: f32 = env_or("LR", 3e-4);
    let seed: i64 = env_or("SEED", 42);
    let target: f32 = env_or("TARGET_EVAL_ACC_PCT", 0.0);
    // ViT shape: default small, override all four for ViT-B/16.
    let layers: i64 = env_or("LAYERS", if fast { 1 } else { 2 });
    let embed: i64 = env_or("EMBED", if fast { 32 } else { 64 });
    let heads: i64 = env_or("HEADS", 4);
    let size: i64 = env_or("SIZE", if fast { 16 } else { 32 });
    let patch: i64 = env_or("PATCH", 8);
    let classes: i64 = env_or("CLASSES", 10);
    assert_eq!(size % patch, 0, "SIZE ({size}) must be divisible by PATCH ({patch})");

    println!(
        "ViT-MNIST: BS={bs} STEPS={steps} TRAIN_N={train_n} TEST_N={test_n} lr={lr} dtype={}",
        dtype::default_float().attr()
    );
    println!("ViT shape: LAYERS={layers} EMBED={embed} HEADS={heads} SIZE={size} PATCH={patch} CLASSES={classes}");

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

    let x_train = load_images(&mut ctx.raw(), &mut sess, &dir.join("train-images-idx3-ubyte"), train_n);
    let y_train = load_labels(&mut ctx.raw(), &mut sess, &dir.join("train-labels-idx1-ubyte"), train_n);
    let x_test = load_images(&mut ctx.raw(), &mut sess, &dir.join("t10k-images-idx3-ubyte"), test_n);
    let y_test = load_labels(&mut ctx.raw(), &mut sess, &dir.join("t10k-labels-idx1-ubyte"), test_n);

    Tensor::manual_seed(seed);
    let model = ViT::build(&mut ctx.raw(), layers, embed, heads, 3, size, patch, classes).into_params();
    let params: Vec<Tensor> = model.params().collect();
    println!("params: {} tensors", params.len());
    let opt = Adam::new(params.clone(), lr);

    let mut test_accuracy = f32::NAN;
    for step in 0..steps {
        // training mode: dropout is active
        sess.training(true);
        let samples = Tensor::randint(&ctx, &[bs], 0, train_n as i32, None, None);
        let xb = x_train.getitem(&[Ix::Tensor(samples.clone())]);
        let yb = y_train.getitem(&[Ix::Tensor(samples)]);
        let xb = prep(&xb, size, &mut ctx.raw());

        let out = model.forward(&xb);
        let loss = out.sparse_categorical_crossentropy(&yb, -1, 0.0, "mean");
        loss.backward(&params, None);
        opt.schedule_step(&mut sess, std::slice::from_ref(&loss));
        opt.zero_grad();

        if step % 10 == 9 || step == steps - 1 {
            let l = loss.item::<f32>(&mut sess);
            // eval mode: dropout is the identity
            sess.training(false);
            test_accuracy = test_acc(&model, &x_test, &y_test, size, &mut ctx.raw(), &mut sess);
            println!("step {:>4}: loss {l:6.3}  test_accuracy {test_accuracy:5.2}%", step + 1);
        }
    }

    println!("to_program: {} misses / {} hits", sess.cache.misses, sess.cache.hits);
    if target > 0.0 {
        if test_accuracy >= target && test_accuracy != 100.0 {
            println!("{test_accuracy} >= {target}: PASS");
        } else {
            eprintln!("{test_accuracy} < {target}: FAIL");
            std::process::exit(1);
        }
    }
}
