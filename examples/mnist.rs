//! MNIST training demo, ported from tinygrad `examples/beautiful_mnist.py`.
//!
//! Same model as tinygrad (Conv/Conv/BatchNorm/pool ×2, flatten, Linear) trained
//! with Adam on minibatches sampled with `Tensor.randint` + advanced indexing
//! (`X[samples]`). Data is downloaded with `curl` into a cache dir and parsed from
//! the IDX format, exactly like tinygrad's `nn.datasets.mnist`.
//!
//! ```sh
//! cargo run --release --example mnist
//! DEFAULT_DEVICE=CL STEPS=70 cargo run --release --example mnist
//! FAST=1 cargo run --release --example mnist              # ~1 minute preset
//! DTYPE=bfloat16 FAST=1 cargo run --release --example mnist --features backend_cl
//! ```
//!
//! Port conveniences (not in tinygrad): `TRAIN_N`/`TEST_N` limit the dataset to a
//! prefix for quick runs, and `MNIST_DIR` sets the download cache (default
//! `/tmp/opencode/mnist`).

use std::path::{Path, PathBuf};
use std::process::Command;

use autorust::dtype;
use autorust::engine::session::Session;
use autorust::tensor::nn::{self, BatchNorm, Conv2d, Linear, Module};
use autorust::tensor::optim::{Adam, Optim};
use autorust::tensor::{Ix, Tensor};
use autorust::uop::ctx::UOpCtx;

/// tinygrad's `Model`: the layer list from `beautiful_mnist.py`.
pub struct Model {
    l0: Conv2d,
    l1: Conv2d,
    l2: BatchNorm,
    l3: Conv2d,
    l4: Conv2d,
    l5: BatchNorm,
    l6: Linear,
}

impl Model {
    pub fn new(ctx: &mut UOpCtx) -> Self {
        Model {
            l0: Conv2d::new(ctx, 1, 32, &[5, 5], 1, &[(0, 0), (0, 0)], 1, 1, true),
            l1: Conv2d::new(ctx, 32, 32, &[5, 5], 1, &[(0, 0), (0, 0)], 1, 1, true),
            l2: BatchNorm::new(ctx, 32, 1e-5, true, true, 0.1),
            l3: Conv2d::new(ctx, 32, 64, &[3, 3], 1, &[(0, 0), (0, 0)], 1, 1, true),
            l4: Conv2d::new(ctx, 64, 64, &[3, 3], 1, &[(0, 0), (0, 0)], 1, 1, true),
            l5: BatchNorm::new(ctx, 64, 1e-5, true, true, 0.1),
            l6: Linear::new(ctx, 576, 10, true),
        }
    }

    /// `x.sequential(self.layers)` — written out, since Rust has no `nn.Sequential`.
    pub fn forward(&self, x: &Tensor, ctx: &mut UOpCtx, sess: &mut Session, training: bool) -> Tensor {
        let x = nn::relu(&self.l0.forward(x, ctx), ctx);
        let x = nn::relu(&self.l1.forward(&x, ctx), ctx);
        let x = self.l2.forward(&x, ctx, sess, training);
        let (x, _) = x.max_pool2d(ctx, &[2, 2], None, &[1, 1], &[0], false, false);
        let x = nn::relu(&self.l3.forward(&x, ctx), ctx);
        let x = nn::relu(&self.l4.forward(&x, ctx), ctx);
        let x = self.l5.forward(&x, ctx, sess, training);
        let (x, _) = x.max_pool2d(ctx, &[2, 2], None, &[1, 1], &[0], false, false);
        let x = x.flatten(ctx, 1, -1);
        self.l6.forward(&x, ctx)
    }
}

impl Module for Model {
    fn named_params(&self) -> Box<dyn Iterator<Item = (String, Tensor)> + '_> {
        Box::new(
            nn::scoped("l0", self.l0.named_params())
                .chain(nn::scoped("l1", self.l1.named_params()))
                .chain(nn::scoped("l2", self.l2.named_params()))
                .chain(nn::scoped("l3", self.l3.named_params()))
                .chain(nn::scoped("l4", self.l4.named_params()))
                .chain(nn::scoped("l5", self.l5.named_params()))
                .chain(nn::scoped("l6", self.l6.named_params())),
        )
    }

    fn into_params(self, ctx: &mut UOpCtx) -> Self {
        Model {
            l0: self.l0.into_params(ctx),
            l1: self.l1.into_params(ctx),
            l2: self.l2.into_params(ctx),
            l3: self.l3.into_params(ctx),
            l4: self.l4.into_params(ctx),
            l5: self.l5.into_params(ctx),
            l6: self.l6.into_params(ctx),
        }
    }
}

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

/// `nn.datasets.mnist`: `Tensor.from_url(file, gunzip=True)[0x10:].reshape(-1,1,28,28)`
pub fn load_images(ctx: &mut UOpCtx, path: &Path, n: i64) -> Tensor {
    let bytes = std::fs::read(path).expect("read idx images");
    let x = Tensor::from_slice(ctx, &bytes[0x10..], Some(dtype::UINT8), None).reshape(ctx, [-1, 1, 28, 28]);
    if n < 60_000 {
        x.getitem(ctx, &[Ix::slice(Some(0), Some(n), 1)])
    } else {
        x
    }
}

/// `Tensor.from_url(file, gunzip=True)[8:]`
pub fn load_labels(ctx: &mut UOpCtx, path: &Path, n: i64) -> Tensor {
    let bytes = std::fs::read(path).expect("read idx labels");
    let y = Tensor::from_slice(ctx, &bytes[8..], Some(dtype::UINT8), None);
    if n < 60_000 {
        y.getitem(ctx, &[Ix::slice(Some(0), Some(n), 1)])
    } else {
        y
    }
}

fn main() {
    // `DTYPE` selects the compute dtype through tinygrad's `DEFAULT_FLOAT`
    // (a synonym: setting `DEFAULT_FLOAT` directly works too). It must be set
    // before any layer/`rand` is built. The faithful default is f32 (tinygrad's
    // `beautiful_mnist` sets no dtype): bf16 is fully ported and correct, but on
    // CL it is ~2x *slower* (tinygrad emulates bf16 through f32, so every op is
    // strictly more work), so it stays opt-in.
    if let Ok(dt) = std::env::var("DTYPE") {
        // SAFETY: single-threaded; set before any `default_float()` read.
        unsafe { std::env::set_var("DEFAULT_FLOAT", dt) };
    }
    // `FAST=1` presets a run that finishes in roughly a minute.
    let fast = std::env::var("FAST").is_ok();

    let bs: i64 = env_or("BS", if fast { 32 } else { 512 });
    let steps: i64 = env_or("STEPS", if fast { 3 } else { 70 });
    let train_n: i64 = env_or("TRAIN_N", if fast { 256 } else { 60_000 });
    let test_n: i64 = env_or("TEST_N", if fast { 128 } else { 10_000 });
    let lr: f32 = env_or("LR", 0.001);
    let seed: i64 = env_or("SEED", 42);
    let target: f32 = env_or("TARGET_EVAL_ACC_PCT", 0.0);

    println!("MNIST: BS={bs} STEPS={steps} TRAIN_N={train_n} TEST_N={test_n} lr={lr} dtype={}", dtype::default_float().attr());

    let dir = mnist_dir();
    let base = "https://storage.googleapis.com/cvdf-datasets/mnist/";
    let files = [
        ("train-images-idx3-ubyte.gz", "train-images-idx3-ubyte"),
        ("train-labels-idx1-ubyte.gz", "train-labels-idx1-ubyte"),
        ("t10k-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte"),
        ("t10k-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte"),
    ];
    for (gz, raw) in files {
        fetch(&format!("{base}{gz}"), &dir.join(raw));
    }

    let mut ctx = UOpCtx::new();
    let mut sess = Session::new();

    let x_train = load_images(&mut ctx, &dir.join("train-images-idx3-ubyte"), train_n);
    let y_train = load_labels(&mut ctx, &dir.join("train-labels-idx1-ubyte"), train_n);
    let x_test = load_images(&mut ctx, &dir.join("t10k-images-idx3-ubyte"), test_n);
    let y_test = load_labels(&mut ctx, &dir.join("t10k-labels-idx1-ubyte"), test_n);

    Tensor::manual_seed(seed);
    let model = Model::new(&mut ctx).into_params(&mut ctx);
    let params: Vec<Tensor> = model.params().collect();
    let opt = Adam::new(&mut ctx, params.clone(), lr);

    let mut test_acc = f32::NAN;
    for step in 0..steps {
        let samples = Tensor::randint(&mut ctx, &[bs], 0, train_n as i32, None, None);
        let xb = x_train.getitem(&mut ctx, &[Ix::Tensor(samples.clone())]);
        let yb = y_train.getitem(&mut ctx, &[Ix::Tensor(samples)]);

        let out = model.forward(&xb, &mut ctx, &mut sess, true);
        let loss = out.sparse_categorical_crossentropy(&mut ctx, &yb, -1, 0.0, "mean");
        loss.backward(&mut ctx, &params, None);
        opt.schedule_step(&mut ctx, &mut sess, std::slice::from_ref(&loss));
        opt.zero_grad(&ctx);

        if step % 10 == 9 || step == steps - 1 {
            test_acc = get_test_acc(&model, &x_test, &y_test, &mut ctx, &mut sess);
            let l = loss.item::<f32>(&mut ctx, &mut sess);
            println!("step {:>4}: loss {l:6.3}  test_accuracy {test_acc:5.2}%", step + 1);
        }
    }

    println!("to_program: {} misses / {} hits", sess.cache.misses, sess.cache.hits);
    if target > 0.0 {
        if test_acc >= target && test_acc != 100.0 {
            println!("{test_acc} >= {target}: PASS");
        } else {
            eprintln!("{test_acc} < {target}: FAIL");
            std::process::exit(1);
        }
    }
}

/// `Model.get_test_acc`: `(self(X).argmax(axis=1) == Y).mean()*100`.
pub fn get_test_acc(model: &Model, x: &Tensor, y: &Tensor, ctx: &mut UOpCtx, sess: &mut Session) -> f32 {
    let out = model.forward(x, ctx, sess, false);
    let pred = out.argmax(ctx, Some(1), false);
    let correct = pred.eq(ctx, y).cast(ctx, dtype::FLOAT32).mean(ctx, None, false);
    correct.mul(ctx, 100.0).item::<f32>(ctx, sess)
}
