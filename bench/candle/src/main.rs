//! candle side of the cross-framework benchmark.
//!
//! Same three workloads as `bench/autorust` and the Python drivers:
//! `mnist_train` (beautiful_mnist CNN), `vit_infer` and `vit_train`
//! (ViT-B/16). Env knobs: `BENCH_WORKLOAD`, `BENCH_DEVICE` (cpu|cuda),
//! `BENCH_BS`, `BENCH_WARMUP`, `BENCH_ITERS`, `BENCH_OUT`.
//!
//! candle-nn's `BatchNorm` is inference-only, so the training path uses a small
//! batch-statistics BN (no running-stat update; noted in the results).

use std::time::Instant;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{
    AdamW, Conv2d, Conv2dConfig, Init, LayerNorm, Linear, Module, Optimizer, ParamsAdamW,
    VarBuilder, VarMap, conv2d, layer_norm, linear, loss::cross_entropy, ops::softmax_last_dim,
};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Training-mode batch norm: normalize with batch statistics.
struct Bn {
    w: Tensor,
    b: Tensor,
    eps: f64,
}

impl Bn {
    fn new(c: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            w: vb.get_with_hints(c, "weight", Init::Const(1.))?,
            b: vb.get_with_hints(c, "bias", Init::Const(0.))?,
            eps: 1e-5,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mean = x.mean_keepdim((0, 2, 3))?;
        let var = x.broadcast_sub(&mean)?.sqr()?.mean_keepdim((0, 2, 3))?;
        let denom = var.affine(1.0, self.eps)?.sqrt()?;
        let c = x.dims()[1];
        let w = self.w.reshape((1, c, 1, 1))?;
        let b = self.b.reshape((1, c, 1, 1))?;
        x.broadcast_sub(&mean)?
            .broadcast_div(&denom)?
            .broadcast_mul(&w)?
            .broadcast_add(&b)
    }
}

struct Mnist {
    l0: Conv2d,
    l1: Conv2d,
    l2: Bn,
    l3: Conv2d,
    l4: Conv2d,
    l5: Bn,
    l6: Linear,
}

impl Mnist {
    fn new(vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig::default();
        Ok(Self {
            l0: conv2d(1, 32, 5, cfg, vb.pp("l0"))?,
            l1: conv2d(32, 32, 5, cfg, vb.pp("l1"))?,
            l2: Bn::new(32, vb.pp("l2"))?,
            l3: conv2d(32, 64, 3, cfg, vb.pp("l3"))?,
            l4: conv2d(64, 64, 3, cfg, vb.pp("l4"))?,
            l5: Bn::new(64, vb.pp("l5"))?,
            l6: linear(576, 10, vb.pp("l6"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.l0.forward(x)?.relu()?;
        let x = self.l1.forward(&x)?.relu()?;
        let x = self.l2.forward(&x)?;
        let x = x.max_pool2d(2)?;
        let x = self.l3.forward(&x)?.relu()?;
        let x = self.l4.forward(&x)?.relu()?;
        let x = self.l5.forward(&x)?;
        let x = x.max_pool2d(2)?;
        self.l6.forward(&x.flatten_from(1)?)
    }
}

struct Block {
    ln1: LayerNorm,
    qkv: Linear,
    out: Linear,
    ln2: LayerNorm,
    ff1: Linear,
    ff2: Linear,
    heads: usize,
}

impl Block {
    fn new(dim: usize, heads: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            ln1: layer_norm(dim, 1e-5, vb.pp("ln1"))?,
            qkv: linear(dim, 3 * dim, vb.pp("qkv"))?,
            out: linear(dim, dim, vb.pp("out"))?,
            ln2: layer_norm(dim, 1e-5, vb.pp("ln2"))?,
            ff1: linear(dim, 4 * dim, vb.pp("ff1"))?,
            ff2: linear(4 * dim, dim, vb.pp("ff2"))?,
            heads,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims();
        let (b, n, d) = (dims[0], dims[1], dims[2]);
        let hd = d / self.heads;
        let h = self.ln1.forward(x)?;
        let qkv = self.qkv.forward(&h)?;
        let shape = (b, n, self.heads, hd);
        let q = qkv.narrow(2, 0, d)?.reshape(shape)?.transpose(1, 2)?;
        let k = qkv.narrow(2, d, d)?.reshape(shape)?.transpose(1, 2)?;
        let v = qkv.narrow(2, 2 * d, d)?.reshape(shape)?.transpose(1, 2)?;
        let scores = q
            .matmul(&k.transpose(2, 3)?)?
            .affine(1.0 / (hd as f64).sqrt(), 0.0)?;
        let p = softmax_last_dim(&scores)?;
        let a = p
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, n, d))?
            .contiguous()?;
        let x = x.add(&self.out.forward(&a)?)?;
        let h = self.ln2.forward(&x)?;
        let mlp = self.ff2.forward(&self.ff1.forward(&h)?.gelu_erf()?)?;
        x.add(&mlp)
    }
}

struct Vit {
    embed: Conv2d,
    cls: Tensor,
    pos: Tensor,
    blocks: Vec<Block>,
    norm: LayerNorm,
    head: Linear,
    dim: usize,
}

impl Vit {
    fn new(layers: usize, dim: usize, heads: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig {
            stride: 16,
            ..Default::default()
        };
        Ok(Self {
            embed: conv2d(3, dim, 16, cfg, vb.pp("embed"))?,
            cls: vb.get_with_hints((1, 1, dim), "cls", Init::Const(1.))?,
            pos: vb.get_with_hints((1, 197, dim), "pos", Init::Const(1.))?,
            blocks: (0..layers)
                .map(|i| Block::new(dim, heads, vb.pp(format!("b{i}"))))
                .collect::<Result<Vec<_>>>()?,
            norm: layer_norm(dim, 1e-5, vb.pp("norm"))?,
            head: linear(dim, 1000, vb.pp("head"))?,
            dim,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let b = x.dims()[0];
        let p = self
            .embed
            .forward(x)?
            .flatten_from(2)?
            .transpose(1, 2)?;
        let cls = self.cls.broadcast_as((b, 1, self.dim))?;
        let mut x = Tensor::cat(&[&cls, &p], 1)?.add(&self.pos)?;
        for blk in &self.blocks {
            x = blk.forward(&x)?;
        }
        let x = self.norm.forward(&x)?;
        self.head.forward(&x.narrow(1, 0, 1)?.squeeze(1)?)
    }
}

fn stats(mut samples: Vec<f64>) -> (f64, f64, f64) {
    samples.sort_by(f64::total_cmp);
    let n = samples.len();
    let median = if n % 2 == 1 {
        samples[n / 2]
    } else {
        0.5 * (samples[n / 2 - 1] + samples[n / 2])
    };
    let q = |p: f64| samples[((n as f64 - 1.0) * p).round() as usize];
    (median, samples[0], q(0.75) - q(0.25))
}

fn main() -> Result<()> {
    let workload = std::env::var("BENCH_WORKLOAD").unwrap_or_else(|_| "mnist_train".into());
    let device_name = std::env::var("BENCH_DEVICE").unwrap_or_else(|_| "cpu".into());
    let dev = match device_name.as_str() {
        "cuda" => Device::new_cuda(0)?,
        _ => Device::Cpu,
    };
    let dev_name = if dev.is_cuda() { "CUDA" } else { "CPU" };
    let training = workload.ends_with("train");
    let default_bs = if workload == "mnist_train" { 512 } else { 1 };
    let bs = env_usize("BENCH_BS", default_bs);
    let warmup = env_usize("BENCH_WARMUP", if training { 5 } else { 3 });
    let iters = env_usize("BENCH_ITERS", if workload == "mnist_train" { 30 } else { 5 });
    let lr = if workload == "mnist_train" { 1e-3 } else { 3e-4 };

    let layers = env_usize("BENCH_LAYERS", 12);
    let dim = env_usize("BENCH_EMBED", 768);
    let heads = env_usize("BENCH_HEADS", 12);

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);

    enum Model {
        Mnist(Mnist),
        Vit(Vit),
    }
    let model = match workload.as_str() {
        "mnist_train" => Model::Mnist(Mnist::new(vb)?),
        "vit_infer" | "vit_train" => Model::Vit(Vit::new(layers, dim, heads, vb)?),
        other => panic!("unknown BENCH_WORKLOAD={other}"),
    };
    let forward = |m: &Model, x: &Tensor| -> Result<Tensor> {
        match m {
            Model::Mnist(m) => m.forward(x),
            Model::Vit(m) => m.forward(x),
        }
    };

    let x = if workload == "mnist_train" {
        Tensor::rand(0f32, 1f32, (bs, 1, 28, 28), &dev)?
    } else {
        Tensor::zeros((bs, 3, 224, 224), DType::F32, &dev)?
    };
    let classes = if workload == "mnist_train" { 10 } else { 1000 };
    let labels: Vec<u32> = (0..bs).map(|i| (i % classes) as u32).collect();
    let y = Tensor::from_vec(labels, (bs,), &dev)?;

    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr,
            ..Default::default()
        },
    )?;

    let mut step = |m: &Model| -> Result<f32> {
        if training {
            let out = forward(m, &x)?;
            let loss = cross_entropy(&out, &y)?;
            opt.backward_step(&loss)?;
            loss.to_scalar::<f32>()
        } else {
            let out = forward(m, &x)?;
            // Force completion and the same host copy autorust makes.
            let _ = out.to_vec2::<f32>()?;
            Ok(0.0)
        }
    };

    dev.synchronize()?;
    let t = Instant::now();
    step(&model)?;
    dev.synchronize()?;
    let cold = t.elapsed().as_secs_f64() * 1e3;
    for _ in 0..warmup {
        step(&model)?;
    }
    dev.synchronize()?;
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        step(&model)?;
        dev.synchronize()?;
        samples.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let (median, min, iqr) = stats(samples);
    let images_per_s = bs as f64 * 1e3 / median;
    let json = format!(
        "{{\"framework\":\"candle\",\"version\":\"0.11.1\",\"device\":\"{dev_name}\",\
         \"workload\":\"{workload}\",\"batch\":{bs},\"dtype\":\"f32\",\"jit\":false,\
         \"warmup\":{warmup},\"iters\":{iters},\"ms_cold\":{cold:.4},\
         \"ms_median\":{median:.4},\"ms_min\":{min:.4},\"ms_iqr\":{iqr:.4},\
         \"images_per_s\":{images_per_s:.2}}}"
    );
    println!(
        "{workload} {dev_name} bs={bs}: {median:.3} ms/step (min {min:.3}, \
         iqr {iqr:.3}, cold {cold:.1} ms, {images_per_s:.0} img/s)"
    );
    if let Ok(path) = std::env::var("BENCH_OUT") {
        std::fs::write(path, &json).expect("write BENCH_OUT");
    }
    println!("RESULT {json}");
    Ok(())
}
