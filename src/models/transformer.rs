//! `extra/models/transformer.py` — `TransformerBlock` and `Transformer`.
//! Attention goes through `Tensor::scaled_dot_product_attention`; the FFN
//! activation is an [`Act`]. Dropout is applied by `Tensor::dropout` (active
//! only under `TRAINING`).

use crate::tensor::nn::Module;
use crate::tensor::Tensor;
use crate::uop::ctx::UOpCtx;
use crate::uop::sint::SVal;

/// `act=lambda x: x.relu()` / `x.gelu()` in `TransformerBlock`.
#[derive(Clone, Copy)]
pub enum Act {
    Relu,
    Gelu,
}

impl Act {
    pub fn apply(self, x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
        match self {
            Act::Relu => crate::tensor::nn::relu(x, ctx),
            Act::Gelu => crate::tensor::nn::gelu(x, ctx),
        }
    }
}

/// `(name.0, name.1)` for a tinygrad `(weight, bias)` tuple attribute.
fn pair(prefix: &str, p: &(Tensor, Tensor)) -> Vec<(String, Tensor)> {
    vec![(format!("{prefix}.0"), p.0.clone()), (format!("{prefix}.1"), p.1.clone())]
}

/// `FA=1` env switch: route attention through the FlashAttention port instead of
/// `scaled_dot_product_attention` (read once, like `JIT`/`TRAINING`).
fn flash_enabled() -> bool {
    static FA: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FA.get_or_init(|| std::env::var("FA").map(|v| v != "0").unwrap_or(false))
}

/// `TransformerBlock` (extra/models/transformer.py:3-39).
pub struct TransformerBlock {
    pub num_heads: i64,
    pub head_size: i64,
    pub prenorm: bool,
    pub act: Act,
    pub dropout: f64,
    pub query: (Tensor, Tensor),
    pub key: (Tensor, Tensor),
    pub value: (Tensor, Tensor),
    pub out: (Tensor, Tensor),
    pub ff1: (Tensor, Tensor),
    pub ff2: (Tensor, Tensor),
    pub ln1: (Tensor, Tensor),
    pub ln2: (Tensor, Tensor),
}

impl TransformerBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new(ctx: &mut UOpCtx, embed_dim: i64, num_heads: i64, ff_dim: i64, prenorm: bool, act: Act, dropout: f64) -> Self {
        assert_eq!(embed_dim % num_heads, 0, "embed_dim must be divisible by num_heads");
        let d = embed_dim;
        TransformerBlock {
            num_heads,
            head_size: embed_dim / num_heads,
            prenorm,
            act,
            dropout,
            query: (Tensor::scaled_uniform(ctx, &[d, d], None), Tensor::zeros(ctx, [d])),
            key: (Tensor::scaled_uniform(ctx, &[d, d], None), Tensor::zeros(ctx, [d])),
            value: (Tensor::scaled_uniform(ctx, &[d, d], None), Tensor::zeros(ctx, [d])),
            out: (Tensor::scaled_uniform(ctx, &[d, d], None), Tensor::zeros(ctx, [d])),
            ff1: (Tensor::scaled_uniform(ctx, &[d, ff_dim], None), Tensor::zeros(ctx, [ff_dim])),
            ff2: (Tensor::scaled_uniform(ctx, &[ff_dim, d], None), Tensor::zeros(ctx, [d])),
            ln1: (Tensor::ones(ctx, [d]), Tensor::zeros(ctx, [d])),
            ln2: (Tensor::ones(ctx, [d]), Tensor::zeros(ctx, [d])),
        }
    }

    /// `TransformerBlock.attn` (transformer.py:24-28).
    pub fn attn(&self, x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
        let bs = x.shape(ctx).into_vec()[0].clone();
        let nh = SVal::I(self.num_heads as i128);
        let hs = SVal::I(self.head_size as i128);
        let project = |ctx: &mut UOpCtx, x: &Tensor, p: &(Tensor, Tensor)| -> Tensor {
            let y = x.linear(ctx, &p.0, Some(&p.1));
            y.reshape(ctx, vec![bs.clone(), SVal::I(-1), nh.clone(), hs.clone()]).transpose(ctx, 1, 2)
        };
        let q = project(ctx, x, &self.query);
        let k = project(ctx, x, &self.key);
        let v = project(ctx, x, &self.value);
        // `FA=1` swaps the (default) SDPA for the tiled FlashAttention port;
        // both are the same softmax attention, so training is equivalent.
        let attention = if flash_enabled() {
            crate::models::flash_attention::flash_attention(&q, &k, &v, None, false, ctx)
        } else {
            q.scaled_dot_product_attention(ctx, &k, &v, None, 0.0, false, false)
        };
        let attention = attention.transpose(ctx, 1, 2);
        let merged = attention.reshape(ctx, vec![bs, SVal::I(-1), SVal::I((self.num_heads * self.head_size) as i128)]);
        merged.linear(ctx, &self.out.0, Some(&self.out.1))
    }

    /// `TransformerBlock.__call__` (transformer.py:30-39).
    pub fn forward(&self, x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
        if self.prenorm {
            let n1 = x.layernorm(ctx, -1, 1e-5).linear(ctx, &self.ln1.0, Some(&self.ln1.1));
            let a = self.attn(&n1, ctx);
            let a = a.dropout(ctx, self.dropout);
            let x = x.add(ctx, &a);
            let n2 = x.layernorm(ctx, -1, 1e-5).linear(ctx, &self.ln2.0, Some(&self.ln2.1));
            let ff = n2.linear(ctx, &self.ff1.0, Some(&self.ff1.1));
            let g = self.act.apply(&ff, ctx);
            let h = g.linear(ctx, &self.ff2.0, Some(&self.ff2.1));
            let h = h.dropout(ctx, self.dropout);
            x.add(ctx, &h)
        } else {
            let a = self.attn(x, ctx);
            let a = a.dropout(ctx, self.dropout);
            let x = x.add(ctx, &a);
            let x = x.layernorm(ctx, -1, 1e-5).linear(ctx, &self.ln1.0, Some(&self.ln1.1));
            let h = x.linear(ctx, &self.ff1.0, Some(&self.ff1.1));
            let h = self.act.apply(&h, ctx);
            let h = h.linear(ctx, &self.ff2.0, Some(&self.ff2.1));
            let h = h.dropout(ctx, self.dropout);
            let x = x.add(ctx, &h);
            x.layernorm(ctx, -1, 1e-5).linear(ctx, &self.ln2.0, Some(&self.ln2.1))
        }
    }
}

impl Module for TransformerBlock {
    fn named_params(&self) -> Box<dyn Iterator<Item = (String, Tensor)> + '_> {
        let mut v = Vec::new();
        v.extend(pair("query", &self.query));
        v.extend(pair("key", &self.key));
        v.extend(pair("value", &self.value));
        v.extend(pair("out", &self.out));
        v.extend(pair("ff1", &self.ff1));
        v.extend(pair("ff2", &self.ff2));
        v.extend(pair("ln1", &self.ln1));
        v.extend(pair("ln2", &self.ln2));
        Box::new(v.into_iter())
    }

    fn into_params(mut self, ctx: &mut UOpCtx) -> Self {
        fn tp(ctx: &mut UOpCtx, p: (Tensor, Tensor)) -> (Tensor, Tensor) {
            (p.0.to_param(ctx), p.1.to_param(ctx))
        }
        self.query = tp(ctx, self.query);
        self.key = tp(ctx, self.key);
        self.value = tp(ctx, self.value);
        self.out = tp(ctx, self.out);
        self.ff1 = tp(ctx, self.ff1);
        self.ff2 = tp(ctx, self.ff2);
        self.ln1 = tp(ctx, self.ln1);
        self.ln2 = tp(ctx, self.ln2);
        self
    }
}

/// `Transformer` (extra/models/transformer.py:41-61): the adder model.
pub struct Transformer {
    pub maxlen: i64,
    pub syms: i64,
    pub embed: Tensor,
    pub tbs: Vec<TransformerBlock>,
    pub final_: Tensor,
}

impl Transformer {
    pub fn new(ctx: &mut UOpCtx, syms: i64, maxlen: i64, layers: i64, embed_dim: i64, num_heads: i64, ff_dim: i64) -> Self {
        Transformer {
            maxlen,
            syms,
            embed: Tensor::scaled_uniform(ctx, &[maxlen + syms, embed_dim], None),
            tbs: (0..layers).map(|_| TransformerBlock::new(ctx, embed_dim, num_heads, ff_dim, false, Act::Relu, 0.1)).collect(),
            final_: Tensor::scaled_uniform(ctx, &[embed_dim, syms], None),
        }
    }

    pub fn forward(&self, x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
        let xs: Vec<SVal> = x.shape(ctx).into_vec();
        let (bs, seq) = (xs[0].clone(), xs[1].clone().as_i64().unwrap_or(0));
        let bs_i = xs[0].clone().as_i64().unwrap_or(0);
        let eye = Tensor::eye(ctx, seq, None, None);
        let eye = eye.unsqueeze(ctx, 0).expand(ctx, vec![bs.clone(), SVal::I(seq as i128), SVal::I(seq as i128)]);
        let onehot_feat = x.cast(ctx, crate::dtype::INT32).one_hot(ctx, self.syms);
        let onehot = eye.cat(ctx, &[onehot_feat], 2).flatten(ctx, 0, 1);
        let mut x = onehot.dot(ctx, &self.embed).reshape(ctx, vec![SVal::I(bs_i as i128), SVal::I(seq as i128), SVal::I(-1)]);
        for tb in &self.tbs {
            x = tb.forward(&x, ctx);
        }
        let last = x.shape(ctx).into_vec()[2].clone();
        let x = x.reshape(ctx, vec![SVal::I(-1), last]).dot(ctx, &self.final_);
        let x = crate::tensor::nn::log_softmax(&x, ctx, Some(1));
        x.reshape(ctx, vec![SVal::I(bs_i as i128), SVal::I(-1), SVal::I(self.syms as i128)])
    }
}
