//! Autoregressive decoder microbench (LLaMA-style): GQA, RoPE, causal mask,
//! KV cache (`cat` + `slice`), embedding gather and on-device sampling.
//!
//! Two phases:
//! - `prefill`: a full prompt (512 tokens) with a causal mask, building the
//!   per-layer KV cache;
//! - `decode`: one token against a cache of 512 past tokens (no mask, all past
//!   positions are visible), sampled on-device (Gumbel-max) so the step stays
//!   in the graph.
//!
//! The decode benchmark reuses the same cache tensor for every timed step: the
//! cache is a graph input (`cat`ed with the new token each step), so the
//! per-token cost is the same as in a growing loop, without the growth itself
//! being part of the measurement.

use std::time::Instant;

use autorust::engine::jit::Jit;
use autorust::engine::session::Session;
use autorust::tensor::ix::Ix;
use autorust::dtype;
use autorust::tensor::nn::{self, Embedding, Linear, Module, RMSNorm};
use autorust::tensor::{Ctx, Tensor};

/// LLaMA-ish decoder dimensions.
pub struct Config {
    /// Model width.
    pub d_model: i64,
    /// Query heads.
    pub n_heads: i64,
    /// Key/value heads (GQA group size is `n_heads / n_kv`).
    pub n_kv: i64,
    /// Decoder blocks.
    pub n_layers: i64,
    /// Feed-forward width.
    pub ffn: i64,
    /// Vocabulary size (the lm_head dominates decode bandwidth).
    pub vocab: i64,
    /// Prompt length for prefill and the cache length for decode.
    pub seq: i64,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            d_model: 768,
            n_heads: 12,
            n_kv: 4,
            n_layers: 6,
            ffn: 3072,
            vocab: 32000,
            seq: 512,
        }
    }
}

struct Attn {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    cos: Tensor,
    sin: Tensor,
    n_heads: i64,
    n_kv: i64,
    hd: i64,
}

struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

struct Block {
    ln1: RMSNorm,
    attn: Attn,
    ln2: RMSNorm,
    mlp: Mlp,
}

/// The decoder.
pub struct Llm {
    embed: Embedding,
    blocks: Vec<Block>,
    norm: RMSNorm,
    head: Linear,
}

fn full(_dim: usize) -> Ix {
    Ix::slice(None, None, 1)
}

fn range(lo: i64, hi: i64) -> Ix {
    Ix::slice(Some(lo), Some(hi), 1)
}

impl Attn {
    fn new(ctx: &Ctx, d: i64, n_heads: i64, n_kv: i64, seq: i64) -> Attn {
        let hd = d / n_heads;
        let half = hd / 2;
        // One extra position: decode appends a token at index `seq`.
        let positions = seq + 1;
        let mut cos = vec![0.0f32; (positions * half) as usize];
        let mut sin = vec![0.0f32; (positions * half) as usize];
        for pos in 0..positions {
            for i in 0..half {
                let freq = 1.0 / 10_000f64.powf((2 * i) as f64 / hd as f64);
                let a = pos as f64 * freq;
                cos[(pos * half + i) as usize] = a.cos() as f32;
                sin[(pos * half + i) as usize] = a.sin() as f32;
            }
        }
        Attn {
            q: Linear::new(ctx, d, n_heads * hd, false),
            k: Linear::new(ctx, d, n_kv * hd, false),
            v: Linear::new(ctx, d, n_kv * hd, false),
            o: Linear::new(ctx, n_heads * hd, d, false),
            cos: Tensor::from_slice(ctx, &cos, None, None)
                .reshape([positions, half])
                .to_param(),
            sin: Tensor::from_slice(ctx, &sin, None, None)
                .reshape([positions, half])
                .to_param(),
            n_heads,
            n_kv,
            hd,
        }
    }

    /// x: `[B, T, H*D]` at absolute positions `start..start+T`.
    fn forward(
        &self,
        x: &Tensor,
        start: i64,
        cache: Option<&(Tensor, Tensor)>,
    ) -> (Tensor, Option<(Tensor, Tensor)>) {
        let sh = x.shape();
        let (b, t) = (
            sh.as_slice()[0].clone().as_i64().unwrap_or(1),
            sh.as_slice()[1].clone().as_i64().unwrap_or(1),
        );
        let (h, kv, hd) = (self.n_heads, self.n_kv, self.hd);
        let q = self.q.forward(x).reshape([b, t, h, hd]).transpose(1, 2);
        let k = self
            .k
            .forward(x)
            .reshape([b, t, kv, hd])
            .transpose(1, 2);
        let v = self
            .v
            .forward(x)
            .reshape([b, t, kv, hd])
            .transpose(1, 2);
        let half = hd / 2;
        let cos = self
            .cos
            .getitem(&[range(start, start + t), full(2)])
            .reshape([1, 1, t, half]);
        let sin = self
            .sin
            .getitem(&[range(start, start + t), full(2)])
            .reshape([1, 1, t, half]);
        let cos = cos.cat(std::slice::from_ref(&cos), 3);
        let sin = sin.cat(std::slice::from_ref(&sin), 3);
        let rope = |x: Tensor| {
            let x1 = x.getitem(&[full(4), full(4), full(4), range(0, half)]);
            let x2 = x.getitem(&[full(4), full(4), full(4), range(half, hd)]);
            let rot = x1.neg().cat(&[x2], 3);
            x.mul(cos.clone()).add(rot.mul(sin.clone()))
        };
        let q = rope(q);
        let k = rope(k);
        // The cache stores one entry per kv head (pre-GQA-repeat).
        let cache_out = cache.is_none().then(|| (k.clone(), v.clone()));
        let (kk, vv) = match cache {
            Some((kc, vc)) => (kc.cat(&[k], 2), vc.cat(&[v], 2)),
            None => (k.clone(), v.clone()),
        };
        // GQA: repeat each kv head `group` times, matching q-head order.
        let group = h / kv;
        let (kk, vv) = if group > 1 {
            (kk.repeat(&[1, group, 1, 1]), vv.repeat(&[1, group, 1, 1]))
        } else {
            (kk, vv)
        };
        let scale = 1.0 / (hd as f64).sqrt();
        let scores = q.matmul(kk.transpose(-1, -2)).mul(scale);
        let scores = if start == 0 && t > 1 {
            // Causal mask: token `i` sees keys `j <= i`.
            let idx = Tensor::arange(x.ctx(), 0.0, Some(t as f64), 1.0, None);
            let mask = idx.reshape([t, 1]).ge(idx.reshape([1, t]));
            mask.where_(&scores, f64::NEG_INFINITY)
        } else {
            scores
        };
        let attn = nn::softmax(&scores, Some(3));
        let y = attn.matmul(&vv).transpose(1, 2).reshape([b, t, h * hd]);
        (self.o.forward(&y), cache_out)
    }
}

impl Block {
    fn forward(
        &self,
        x: &Tensor,
        start: i64,
        cache: Option<&(Tensor, Tensor)>,
    ) -> (Tensor, Option<(Tensor, Tensor)>) {
        let (a, c) = self.attn.forward(&self.ln1.forward(x), start, cache);
        let x = x.add(&a);
        let h = self.ln2.forward(&x);
        let gate = self.mlp.gate.forward(&h).silu();
        let up = self.mlp.up.forward(&h);
        let y = self.mlp.down.forward(&gate.mul(&up));
        (x.add(&y), c)
    }
}

impl Llm {
    /// Build a decoder with `cfg`. Weights are parameter leaves.
    pub fn new(ctx: &Ctx, cfg: &Config) -> Llm {
        let blocks = (0..cfg.n_layers)
            .map(|_| Block {
                ln1: RMSNorm::new(ctx, cfg.d_model, 1e-6, true).into_params(),
                attn: Attn::new(ctx, cfg.d_model, cfg.n_heads, cfg.n_kv, cfg.seq),
                ln2: RMSNorm::new(ctx, cfg.d_model, 1e-6, true).into_params(),
                mlp: Mlp {
                    gate: Linear::new(ctx, cfg.d_model, cfg.ffn, false),
                    up: Linear::new(ctx, cfg.d_model, cfg.ffn, false),
                    down: Linear::new(ctx, cfg.ffn, cfg.d_model, false),
                },
            })
            .collect();
        Llm {
            embed: Embedding::new(ctx, cfg.vocab, cfg.d_model),
            blocks,
            norm: RMSNorm::new(ctx, cfg.d_model, 1e-6, true).into_params(),
            head: Linear::new(ctx, cfg.d_model, cfg.vocab, false),
        }
    }

    /// Prompt pass: returns `(logits, per-layer (k, v) caches)`.
    pub fn prefill(&self, tokens: &Tensor) -> (Tensor, Vec<(Tensor, Tensor)>) {
        let mut x = self.embed.forward(tokens);
        let mut caches = Vec::with_capacity(self.blocks.len());
        for b in &self.blocks {
            let (y, c) = b.forward(&x, 0, None);
            x = y;
            caches.push(c.expect("prefill returns the cache"));
        }
        (self.head.forward(&self.norm.forward(&x)), caches)
    }

    /// One token at absolute position `pos` against `caches`; returns the
    /// sampled next token `[B, 1]`.
    pub fn decode(&self, token: &Tensor, pos: i64, caches: &[(Tensor, Tensor)]) -> Tensor {
        let mut x = self.embed.forward(token);
        for (b, cache) in self.blocks.iter().zip(caches) {
            let (y, _) = b.forward(&x, pos, Some(cache));
            x = y;
        }
        let logits = self.head.forward(&self.norm.forward(&x));
        sample(&logits)
    }
}

impl Module for Llm {
    fn named_params(&self) -> Box<dyn Iterator<Item = (String, Tensor)> + '_> {
        let mut v: Vec<(String, Tensor)> = self
            .embed
            .named_params()
            .map(|(n, t)| (format!("embed.{n}"), t))
            .collect();
        for (i, b) in self.blocks.iter().enumerate() {
            let p = format!("block{i}");
            for (n, t) in b.ln1.named_params() {
                v.push((format!("{p}.ln1.{n}"), t));
            }
            for (n, t) in b.attn.q.named_params() {
                v.push((format!("{p}.attn.q.{n}"), t));
            }
            for (n, t) in b.attn.k.named_params() {
                v.push((format!("{p}.attn.k.{n}"), t));
            }
            for (n, t) in b.attn.v.named_params() {
                v.push((format!("{p}.attn.v.{n}"), t));
            }
            for (n, t) in b.attn.o.named_params() {
                v.push((format!("{p}.attn.o.{n}"), t));
            }
            v.push((format!("{p}.attn.rope.cos"), b.attn.cos.clone()));
            v.push((format!("{p}.attn.rope.sin"), b.attn.sin.clone()));
            for (n, t) in b.ln2.named_params() {
                v.push((format!("{p}.ln2.{n}"), t));
            }
            for (n, t) in b.mlp.gate.named_params() {
                v.push((format!("{p}.mlp.gate.{n}"), t));
            }
            for (n, t) in b.mlp.up.named_params() {
                v.push((format!("{p}.mlp.up.{n}"), t));
            }
            for (n, t) in b.mlp.down.named_params() {
                v.push((format!("{p}.mlp.down.{n}"), t));
            }
        }
        for (n, t) in self.norm.named_params() {
            v.push((format!("norm.{n}"), t));
        }
        for (n, t) in self.head.named_params() {
            v.push((format!("head.{n}"), t));
        }
        Box::new(v.into_iter())
    }

    fn into_params(self) -> Self {
        // Every layer was built with `into_params` / `to_param` already.
        self
    }
}

/// On-device Gumbel-max sampling (`-log(-log(u))`), so no host round-trip.
fn sample(logits: &Tensor) -> Tensor {
    let u = logits.rand_like(None);
    let g = u.log().neg().log().neg();
    logits.add(&g).argmax(Some(-1), true)
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(f64::total_cmp);
    let n = xs.len();
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        0.5 * (xs[n / 2 - 1] + xs[n / 2])
    }
}

/// Measure the prefill step.
pub fn bench_prefill(
    ctx: &Ctx,
    sess: &mut Session,
    cfg: &Config,
    warmup: usize,
    iters: usize,
) -> (Llm, Tensor, Vec<(Tensor, Tensor)>, f64) {
    let model = Llm::new(ctx, cfg);
    let tokens =
        Tensor::randint(ctx, &[1, cfg.seq], 0, cfg.vocab as i32, Some(dtype::INT32), None).to_param();
    let mut jit = Jit::new();
    let mut caches: Vec<(Tensor, Tensor)> = Vec::new();
    let step = |jit: &mut Jit, sess: &mut Session, caches: &mut Vec<(Tensor, Tensor)>| {
        let mut outs = jit.call_many(ctx, sess, |_c, _s| {
            let (l, c) = model.prefill(&tokens);
            let mut roots = vec![l.clone()];
            for (k, v) in &c {
                roots.push(k.clone());
                roots.push(v.clone());
            }
            roots
        });
        let logits = outs.remove(0);
        let mut cs = Vec::new();
        let mut it = outs.into_iter();
        while let (Some(k), Some(v)) = (it.next(), it.next()) {
            cs.push((k, v));
        }
        let _ = logits
            .getitem(&[range(0, 1), range(0, 1)])
            .to_list::<f32>(sess);
        *caches = cs;
    };
    step(&mut jit, sess, &mut caches);
    for _ in 0..warmup {
        step(&mut jit, sess, &mut caches);
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        step(&mut jit, sess, &mut caches);
        samples.push(t.elapsed().as_secs_f64() * 1e3);
    }
    (model, tokens, caches, median(samples))
}

/// Measure one decode step against a fixed cache (returns ms).
pub fn bench_decode(
    ctx: &Ctx,
    sess: &mut Session,
    model: &Llm,
    caches: &[(Tensor, Tensor)],
    cfg: &Config,
    warmup: usize,
    iters: usize,
) -> f64 {
    let token =
        Tensor::randint(ctx, &[1, 1], 0, cfg.vocab as i32, Some(dtype::INT32), None).to_param();
    let mut jit = Jit::new();
    let step = |jit: &mut Jit, sess: &mut Session| {
        let next = jit.call(ctx, sess, |_c, _s| model.decode(&token, cfg.seq, caches));
        let _ = next.to_list::<i32>(sess);
    };
    eprintln!("=== DECODE START");
    step(&mut jit, sess);
    for _ in 0..warmup.max(1) {
        step(&mut jit, sess);
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        step(&mut jit, sess);
        samples.push(t.elapsed().as_secs_f64() * 1e3);
    }
    median(samples)
}
