//! burn side of the cross-framework benchmark.
//!
//! Mirrors the candle/ autorust workloads: `beautiful_mnist` CNN training and
//! ViT-B/16 inference/training, plus the GEMM sweep. Models are written against
//! burn's public `nn` API and run through the same `Device` chosen by
//! `BENCH_DEVICE` (CPU = Flex, CUDA = CubeCL).
//!
//! Burn's fusion (enabled by default on the CubeCL backends) fuses elementwise
//! chains; the CPU rows use `Device::flex()` because that is the pure-Rust CPU
//! backend burn's own examples select (NdArray is deprecated in 0.22).

use std::time::Instant;

use burn::module::{Initializer, Param};
use burn::nn::{
    BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Gelu, LayerNorm, LayerNormConfig, Linear,
    LinearConfig, PaddingConfig2d, Relu,
    conv::{Conv2d, Conv2dConfig},
    loss::CrossEntropyLossConfig,
    pool::MaxPool2dConfig,
};
use burn::optim::{AdamWConfig, GradientsParams};
use burn::prelude::*;
use burn::tensor::Distribution;
use burn::tensor::activation::softmax;
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// `beautiful_mnist` CNN: two conv+BN blocks, 2x2/2 max pools, 576->10.
#[derive(Module, Debug)]
struct Mnist {
    l0: Conv2d,
    l1: Conv2d,
    l2: BatchNorm,
    l3: Conv2d,
    l4: Conv2d,
    l5: BatchNorm,
    l6: Linear,
}

impl Mnist {
    fn new(device: &Device) -> Self {
        let conv = |channels: [usize; 2], kernel: [usize; 2]| {
            Conv2dConfig::new(channels, kernel)
                .with_padding(PaddingConfig2d::Valid)
                .init(device)
        };
        Self {
            l0: conv([1, 32], [5, 5]),
            l1: conv([32, 32], [5, 5]),
            l2: BatchNormConfig::new(32).init(device),
            l3: conv([32, 64], [3, 3]),
            l4: conv([64, 64], [3, 3]),
            l5: BatchNormConfig::new(64).init(device),
            l6: LinearConfig::new(576, 10).init(device),
        }
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<2> {
        let pool = || MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();
        let x = Relu::new().forward(self.l0.forward(x));
        let x = Relu::new().forward(self.l1.forward(x));
        let x = self.l2.forward(x);
        let x = pool().forward(x);
        let x = Relu::new().forward(self.l3.forward(x));
        let x = Relu::new().forward(self.l4.forward(x));
        let x = self.l5.forward(x);
        let x = pool().forward(x);
        let [b, c, h, w] = x.dims();
        self.l6.forward(x.reshape([b, c * h * w]))
    }
}

#[derive(Module, Debug)]
struct Block {
    ln1: LayerNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    ln2: LayerNorm,
    ff1: Linear,
    ff2: Linear,
    drop: Dropout,
    drop2: Dropout,
    act: Gelu,
    heads: usize,
}

impl Block {
    fn new(dim: usize, heads: usize, device: &Device) -> Self {
        Self {
            ln1: LayerNormConfig::new(dim).init(device),
            q: LinearConfig::new(dim, dim).init(device),
            k: LinearConfig::new(dim, dim).init(device),
            v: LinearConfig::new(dim, dim).init(device),
            out: LinearConfig::new(dim, dim).init(device),
            ln2: LayerNormConfig::new(dim).init(device),
            ff1: LinearConfig::new(dim, 4 * dim).init(device),
            ff2: LinearConfig::new(4 * dim, dim).init(device),
            drop: DropoutConfig::new(0.1).init(),
            drop2: DropoutConfig::new(0.1).init(),
            act: Gelu::new(),
            heads,
        }
    }

    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [b, n, d] = x.dims();
        let hd = d / self.heads;
        let shape = [b, n, self.heads, hd];

        let h = self.ln1.forward(x.clone());
        let q = self.q.forward(h.clone()).reshape(shape).swap_dims(1, 2);
        let k = self.k.forward(h.clone()).reshape(shape).swap_dims(1, 2);
        let v = self.v.forward(h).reshape(shape).swap_dims(1, 2);

        // Inference goes through burn's fused attention op (flash on CUDA,
        // fused CPU kernel on Flex). Autodiff has only the fallback, so the
        // training path materializes the scores like the other drivers.
        let a = if x.device().is_autodiff() {
            let scores = q.matmul(k.swap_dims(2, 3)).mul_scalar(1.0 / (hd as f32).sqrt());
            softmax(scores, 3).matmul(v)
        } else {
            attention(
                q,
                k,
                v,
                None,
                None,
                AttentionModuleOptions {
                    scale: None,
                    softcap: None,
                    is_causal: false,
                },
            )
        };
        let a = a.swap_dims(1, 2).reshape([b, n, d]);
        let x = x.add(self.drop.forward(self.out.forward(a)));

        let h = self.ln2.forward(x.clone());
        let h = self.act.forward(self.ff1.forward(h));
        x.add(self.drop2.forward(self.ff2.forward(h)))
    }
}

#[derive(Module, Debug)]
struct Vit {
    embed: Conv2d,
    cls: Param<Tensor<3>>,
    pos: Param<Tensor<3>>,
    blocks: Vec<Block>,
    norm: LayerNorm,
    head: Linear,
    dim: usize,
    patches: usize,
}

impl Vit {
    fn new(
        layers: usize,
        dim: usize,
        heads: usize,
        img: usize,
        patch: usize,
        classes: usize,
        device: &Device,
    ) -> Self {
        let side = img / patch;
        let patches = side * side;
        let embed = Conv2dConfig::new([3, dim], [patch, patch])
            .with_stride([patch, patch])
            .with_padding(PaddingConfig2d::Valid)
            .init(device);
        Self {
            embed,
            cls: Initializer::Ones.init([1, 1, dim], device),
            pos: Initializer::Ones.init([1, patches + 1, dim], device),
            blocks: (0..layers)
                .map(|_| Block::new(dim, heads, device))
                .collect(),
            norm: LayerNormConfig::new(dim).init(device),
            head: LinearConfig::new(dim, classes).init(device),
            dim,
            patches,
        }
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<2> {
        let [b, _, _, _] = x.dims();
        let p = self
            .embed
            .forward(x)
            .reshape([b, self.dim, self.patches])
            .swap_dims(1, 2);
        let cls = self.cls.val().expand([b, 1, self.dim]);
        let mut x = Tensor::cat(vec![cls, p], 1).add(self.pos.val());
        for block in &self.blocks {
            x = block.forward(x);
        }
        let x = self.norm.forward(x);
        self.head.forward(x.narrow(1, 0, 1).squeeze_dim::<2>(1))
    }
}

fn stats(mut samples: Vec<f64>) -> (f64, f64, f64) {
    samples.sort_by(f64::total_cmp);
    let n = samples.len();
    if n == 0 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let median = if n % 2 == 1 {
        samples[n / 2]
    } else {
        0.5 * (samples[n / 2 - 1] + samples[n / 2])
    };
    let q = |p: f64| samples[((n as f64 - 1.0) * p).round() as usize];
    (median, samples[0], q(0.75) - q(0.25))
}

fn emit(
    workload: &str,
    device: &str,
    batch: usize,
    dtype: &str,
    jit: bool,
    warmup: usize,
    iters: usize,
    cold: f64,
    samples: Vec<f64>,
) {
    let (median, min, iqr) = stats(samples);
    let images_per_s = batch as f64 * 1e3 / median;
    let json = format!(
        "{{\"framework\":\"burn\",\"version\":\"0.22.1\",\"device\":\"{device}\",\
         \"workload\":\"{workload}\",\"batch\":{batch},\"dtype\":\"{dtype}\",\"jit\":{jit},\
         \"warmup\":{warmup},\"iters\":{iters},\"ms_cold\":{cold:.4},\
         \"ms_median\":{median:.4},\"ms_min\":{min:.4},\"ms_iqr\":{iqr:.4},\
         \"images_per_s\":{images_per_s:.2}}}"
    );
    println!(
        "{workload} {device} bs={batch}: {median:.3} ms/step (min {min:.3}, \
         iqr {iqr:.3}, cold {cold:.1} ms, {images_per_s:.0} img/s)"
    );
    if let Ok(path) = std::env::var("BENCH_OUT") {
        std::fs::write(path, &json).expect("write BENCH_OUT");
    }
    println!("RESULT {json}");
}

fn emit_cases(workload: &str, device: &str, dtype: &str, warmup: usize, iters: usize, cases: &[(String, f64, f64)]) {
    let total: f64 = cases.iter().map(|c| c.1).sum();
    let rows: Vec<String> = cases
        .iter()
        .map(|(name, ms, tf)| format!("{{\"case\":\"{name}\",\"ms\":{ms:.4},\"tflops\":{tf:.4}}}"))
        .collect();
    for (name, ms, tf) in cases {
        println!("gemm {device} {name}: {ms:.3} ms ({tf:.2} tflops)");
    }
    let json = format!(
        "{{\"framework\":\"burn\",\"version\":\"0.22.1\",\"device\":\"{device}\",\
         \"workload\":\"{workload}\",\"batch\":1,\"dtype\":\"{dtype}\",\"jit\":false,\
         \"warmup\":{warmup},\"iters\":{iters},\"ms_cold\":0.0,\"ms_median\":{total:.4},\
         \"ms_min\":0.0,\"ms_iqr\":0.0,\"images_per_s\":0.0,\"cases\":[{}]}}",
        rows.join(",")
    );
    if let Ok(path) = std::env::var("BENCH_OUT") {
        std::fs::write(path, &json).expect("write BENCH_OUT");
    }
    println!("RESULT {json}");
}

fn main() {
    let workload = std::env::var("BENCH_WORKLOAD").unwrap_or_else(|_| "mnist_train".into());
    let device_name = std::env::var("BENCH_DEVICE").unwrap_or_else(|_| "cpu".into());
    let base = match device_name.as_str() {
        "cuda" => Device::cuda(0),
        _ => Device::flex(),
    };
    let dev_name = if device_name == "cuda" { "CUDA" } else { "CPU" };
    let seed = env_f64("BENCH_SEED", 42.0) as u64;
    base.seed(seed);

    if workload == "gemm" || workload == "gemm_bf16" {
        gemm(&base, dev_name);
        return;
    }

    let training = workload.ends_with("train");
    let device = if training { base.clone().autodiff() } else { base };
    let default_bs = if workload.starts_with("mnist") { 512 } else { 1 };
    let bs = env_usize("BENCH_BS", default_bs);
    let warmup = env_usize("BENCH_WARMUP", if training { 5 } else { 3 });
    let iters = env_usize("BENCH_ITERS", if workload.starts_with("mnist") { 30 } else { 5 });
    let lr = if workload.starts_with("mnist") { 1e-3 } else { 3e-4 };

    match workload.as_str() {
        "mnist_train" => run_mnist(&device, dev_name, bs, lr, warmup, iters),
        "vit_infer" => run_vit_infer(&device, dev_name, bs, warmup, iters),
        "vit_train" => run_vit_train(&device, dev_name, bs, lr, warmup, iters),
        other => panic!("unknown BENCH_WORKLOAD={other} (mnist_train|vit_infer|vit_train|gemm)"),
    }
}

fn run_mnist(device: &Device, dev_name: &str, bs: usize, lr: f64, warmup: usize, iters: usize) {
    let mut model = Some(Mnist::new(device));
    let mut optim = AdamWConfig::new().init();
    let x = Tensor::<4>::random([bs, 1, 28, 28], Distribution::Default, device);
    let labels: Vec<i64> = (0..bs).map(|i| (i % 10) as i64).collect();
    let y = Tensor::<1, Int>::from_ints(labels.as_slice(), device);
    let loss_fn = CrossEntropyLossConfig::new().init(device);

    let mut step = || {
        let net = model.take().expect("model");
        let out = net.forward(x.clone());
        let loss = loss_fn.forward(out, y.clone());
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &net);
        let next = optim.step(lr, net, grads);
        // Reading an updated parameter drains the optimizer's stream, so the
        // whole step is inside the timed region (the loss readback alone only
        // waits for the forward/backward chain).
        let _: f32 = next.l6.weight.val().slice(s![0, 0]).into_scalar();
        device.sync().expect("device sync");
        model = Some(next);
    };

    let t = Instant::now();
    step();
    let cold = t.elapsed().as_secs_f64() * 1e3;
    for _ in 0..warmup {
        step();
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        step();
        samples.push(t.elapsed().as_secs_f64() * 1e3);
    }
    emit("mnist_train", dev_name, bs, "f32", false, warmup, iters, cold, samples);
}

fn run_vit_infer(device: &Device, dev_name: &str, bs: usize, warmup: usize, iters: usize) {
    let layers = env_usize("BENCH_LAYERS", 12);
    let dim = env_usize("BENCH_EMBED", 768);
    let heads = env_usize("BENCH_HEADS", 12);
    let img = env_usize("BENCH_IMG", 224);
    let model = Vit::new(layers, dim, heads, img, 16, 1000, device);
    let x = Tensor::<4>::zeros([bs, 3, img, img], device);

    let step = || {
        let out = model.forward(x.clone());
        // Host copy, matching the other drivers' completion point, plus an
        // explicit device sync: fused attention launches outside the fusion
        // stream, and a readback on the output alone can return before it ends.
        let _ = out.into_data();
        device.sync().expect("device sync");
    };

    let t = Instant::now();
    step();
    let cold = t.elapsed().as_secs_f64() * 1e3;
    for _ in 0..warmup {
        step();
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        step();
        samples.push(t.elapsed().as_secs_f64() * 1e3);
    }
    emit("vit_infer", dev_name, bs, "f32", false, warmup, iters, cold, samples);
}

fn run_vit_train(device: &Device, dev_name: &str, bs: usize, lr: f64, warmup: usize, iters: usize) {
    let layers = env_usize("BENCH_LAYERS", 12);
    let dim = env_usize("BENCH_EMBED", 768);
    let heads = env_usize("BENCH_HEADS", 12);
    let img = env_usize("BENCH_IMG", 224);
    let mut model = Some(Vit::new(layers, dim, heads, img, 16, 1000, device));
    let mut optim = AdamWConfig::new().init();
    let x = Tensor::<4>::zeros([bs, 3, img, img], device);
    let labels: Vec<i64> = (0..bs).map(|i| (i % 1000) as i64).collect();
    let y = Tensor::<1, Int>::from_ints(labels.as_slice(), device);
    let loss_fn = CrossEntropyLossConfig::new().init(device);

    let mut step = || {
        let net = model.take().expect("model");
        let out = net.forward(x.clone());
        let loss = loss_fn.forward(out, y.clone());
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &net);
        let next = optim.step(lr, net, grads);
        let _: f32 = next.head.weight.val().slice(s![0, 0]).into_scalar();
        device.sync().expect("device sync");
        model = Some(next);
    };

    let t = Instant::now();
    step();
    let cold = t.elapsed().as_secs_f64() * 1e3;
    for _ in 0..warmup {
        step();
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        step();
        samples.push(t.elapsed().as_secs_f64() * 1e3);
    }
    emit("vit_train", dev_name, bs, "f32", false, warmup, iters, cold, samples);
}

fn gemm(device: &Device, dev_name: &str) {
    use burn::tensor::DType;

    let bf16 = std::env::var("BENCH_BF16")
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false);
    let dtype = if bf16 { DType::BF16 } else { DType::F32 };
    let dt_name = if bf16 { "bf16" } else { "f32" };
    let warmup = env_usize("BENCH_WARMUP", 1);
    let iters = env_usize("BENCH_ITERS", 3);
    let only = std::env::var("BENCH_GEMM_ONLY").ok();

    let shapes: &[(&str, usize, usize, usize, usize, bool, bool)] = &[
        ("1024³", 1, 1024, 1024, 1024, false, false),
        ("2048³", 1, 2048, 2048, 2048, false, false),
        ("4096³", 1, 4096, 4096, 4096, false, false),
        ("1024x1024x4096", 1, 1024, 1024, 4096, false, false),
        ("4096x1024x1024", 1, 4096, 1024, 1024, false, false),
        ("TN 2048³", 1, 2048, 2048, 2048, true, false),
        ("NT 2048³", 1, 2048, 2048, 2048, false, true),
        ("batched 8x512³", 8, 512, 512, 512, false, false),
    ];

    let mut cases = Vec::new();
    for &(name, batch, m, n, k, ta, tb) in shapes {
        let label = format!("{dt_name} {name}");
        if let Some(p) = &only
            && !label.contains(p)
        {
            continue;
        }
        let a = Tensor::<3>::random([batch, m, k], Distribution::Default, device).cast(dtype);
        let bshape = if tb { [batch, n, k] } else { [batch, k, n] };
        let b = Tensor::<3>::random(bshape, Distribution::Default, device).cast(dtype);
        let lhs = if ta { a.swap_dims(1, 2) } else { a };
        let rhs = if tb { b.swap_dims(1, 2) } else { b };

        let step = || {
            let c = lhs.clone().matmul(rhs.clone());
            let _: f32 = c.slice(s![0, 0, 0]).cast(DType::F32).into_scalar();
            device.sync().expect("device sync");
        };
        step();
        for _ in 0..warmup {
            step();
        }
        let mut samples = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = Instant::now();
            step();
            samples.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let (ms, _, _) = stats(samples);
        let flops = 2.0 * (batch * m * n * k) as f64;
        cases.push((label, ms, flops / (ms * 1e9)));
    }
    emit_cases("gemm", dev_name, dt_name, warmup, iters, &cases);
}
