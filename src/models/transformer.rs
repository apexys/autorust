//! `extra/models/transformer.py` — `TransformerBlock` and `Transformer`.
//! Attention goes through `Tensor::scaled_dot_product_attention`; the FFN
//! activation is an [`Act`]. Dropout is applied by `Tensor::dropout` (active
//! only under `TRAINING`).

use crate::tensor::nn::Module;
use crate::tensor::{AsCtx, Tensor};
use crate::uop::ctx::UOpCtx;
use crate::uop::sint::SVal;

/// `act=lambda x: x.relu()` / `x.gelu()` in `TransformerBlock`.
#[derive(Clone, Copy)]
pub enum Act {
    /// ReLU activation (`x.relu()`).
    Relu,
    /// GELU activation (`x.gelu()`).
    Gelu,
}

impl Act {
    /// Apply the activation to `x`.
    pub fn apply(self, x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
        match self {
            Act::Relu => crate::tensor::nn::relu_ctx(x, ctx),
            Act::Gelu => crate::tensor::nn::gelu_ctx(x, ctx),
        }
    }
}

/// `(name.0, name.1)` for a tinygrad `(weight, bias)` tuple attribute.
fn pair(prefix: &str, p: &(Tensor, Tensor)) -> Vec<(String, Tensor)> {
    vec![
        (format!("{prefix}.0"), p.0.clone()),
        (format!("{prefix}.1"), p.1.clone()),
    ]
}

/// `FA=1` env switch: route attention through the FlashAttention port instead of
/// `scaled_dot_product_attention` (read once, like `JIT`/`TRAINING`).
fn flash_enabled() -> bool {
    static FA: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FA.get_or_init(|| std::env::var("FA").map(|v| v != "0").unwrap_or(false))
}

/// `TransformerBlock` (extra/models/transformer.py:3-39).
pub struct TransformerBlock {
    /// Number of attention heads.
    pub num_heads: i64,
    /// Per-head embedding size (`embed_dim / num_heads`).
    pub head_size: i64,
    /// Whether layer norm runs before (pre-norm) rather than after the sublayers.
    pub prenorm: bool,
    /// FFN activation.
    pub act: Act,
    /// Dropout probability applied inside the block.
    pub dropout: f64,
    /// `(weight, bias)` of the query projection.
    pub query: (Tensor, Tensor),
    /// `(weight, bias)` of the key projection.
    pub key: (Tensor, Tensor),
    /// `(weight, bias)` of the value projection.
    pub value: (Tensor, Tensor),
    /// `(weight, bias)` of the attention output projection.
    pub out: (Tensor, Tensor),
    /// `(weight, bias)` of the first FFN projection.
    pub ff1: (Tensor, Tensor),
    /// `(weight, bias)` of the second FFN projection.
    pub ff2: (Tensor, Tensor),
    /// `(weight, bias)` of the first layer norm.
    pub ln1: (Tensor, Tensor),
    /// `(weight, bias)` of the second layer norm.
    pub ln2: (Tensor, Tensor),
}

impl TransformerBlock {
    #[allow(clippy::too_many_arguments)]
    /// Construct a block with the given embedding/FFN widths and dropout.
    pub fn new(
        ctx: impl AsCtx,
        embed_dim: i64,
        num_heads: i64,
        ff_dim: i64,
        prenorm: bool,
        act: Act,
        dropout: f64,
    ) -> Self {
        ctx.with_ctx(|ctx| {
            assert_eq!(
                embed_dim % num_heads,
                0,
                "embed_dim must be divisible by num_heads"
            );
            let d = embed_dim;
            TransformerBlock {
                num_heads,
                head_size: embed_dim / num_heads,
                prenorm,
                act,
                dropout,
                query: (
                    Tensor::scaled_uniform(&mut *ctx, &[d, d], None),
                    Tensor::zeros(&mut *ctx, [d]),
                ),
                key: (
                    Tensor::scaled_uniform(&mut *ctx, &[d, d], None),
                    Tensor::zeros(&mut *ctx, [d]),
                ),
                value: (
                    Tensor::scaled_uniform(&mut *ctx, &[d, d], None),
                    Tensor::zeros(&mut *ctx, [d]),
                ),
                out: (
                    Tensor::scaled_uniform(&mut *ctx, &[d, d], None),
                    Tensor::zeros(&mut *ctx, [d]),
                ),
                ff1: (
                    Tensor::scaled_uniform(&mut *ctx, &[d, ff_dim], None),
                    Tensor::zeros(&mut *ctx, [ff_dim]),
                ),
                ff2: (
                    Tensor::scaled_uniform(&mut *ctx, &[ff_dim, d], None),
                    Tensor::zeros(&mut *ctx, [d]),
                ),
                ln1: (Tensor::ones(&mut *ctx, [d]), Tensor::zeros(&mut *ctx, [d])),
                ln2: (Tensor::ones(&mut *ctx, [d]), Tensor::zeros(&mut *ctx, [d])),
            }
        })
    }

    /// `TransformerBlock.attn` (transformer.py:24-28).
    pub fn attn(&self, x: &Tensor) -> Tensor {
        self.attn_ctx(x, &mut x.ctx().borrow_mut())
    }
    #[doc(hidden)]
    pub fn attn_ctx(&self, x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
        let bs = x.shape_ctx(ctx).into_vec()[0].clone();
        let nh = SVal::I(self.num_heads as i128);
        let hs = SVal::I(self.head_size as i128);
        let project = |ctx: &mut UOpCtx, x: &Tensor, p: &(Tensor, Tensor)| -> Tensor {
            let y = x.linear_ctx(ctx, &p.0, Some(&p.1));
            y.reshape_ctx(ctx, vec![bs.clone(), SVal::I(-1), nh.clone(), hs.clone()])
                .transpose_ctx(ctx, 1, 2)
        };
        let q = project(ctx, x, &self.query);
        let k = project(ctx, x, &self.key);
        let v = project(ctx, x, &self.value);
        // `FA=1` swaps the (default) SDPA for the tiled FlashAttention port;
        // both are the same softmax attention, so training is equivalent.
        let attention = if flash_enabled() {
            crate::models::flash_attention::flash_attention_ctx(&q, &k, &v, None, false, ctx)
        } else {
            q.scaled_dot_product_attention_ctx(ctx, &k, &v, None, 0.0, false, false)
        };
        let attention = attention.transpose_ctx(ctx, 1, 2);
        let merged = attention.reshape_ctx(
            ctx,
            vec![
                bs,
                SVal::I(-1),
                SVal::I((self.num_heads * self.head_size) as i128),
            ],
        );
        merged.linear_ctx(ctx, &self.out.0, Some(&self.out.1))
    }

    /// `TransformerBlock.__call__` (transformer.py:30-39).
    pub fn forward(&self, x: &Tensor) -> Tensor {
        self.forward_ctx(x, &mut x.ctx().borrow_mut())
    }
    #[doc(hidden)]
    pub fn forward_ctx(&self, x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
        if self.prenorm {
            let n1 = x
                .layernorm_ctx(ctx, -1, 1e-5)
                .linear_ctx(ctx, &self.ln1.0, Some(&self.ln1.1));
            let a = self.attn_ctx(&n1, ctx);
            let a = a.dropout_ctx(ctx, self.dropout);
            let x = x.add_ctx(ctx, &a);
            let n2 = x
                .layernorm_ctx(ctx, -1, 1e-5)
                .linear_ctx(ctx, &self.ln2.0, Some(&self.ln2.1));
            let ff = n2.linear_ctx(ctx, &self.ff1.0, Some(&self.ff1.1));
            let g = self.act.apply(&ff, ctx);
            let h = g.linear_ctx(ctx, &self.ff2.0, Some(&self.ff2.1));
            let h = h.dropout_ctx(ctx, self.dropout);
            x.add_ctx(ctx, &h)
        } else {
            let a = self.attn_ctx(x, ctx);
            let a = a.dropout_ctx(ctx, self.dropout);
            let x = x.add_ctx(ctx, &a);
            let x = x
                .layernorm_ctx(ctx, -1, 1e-5)
                .linear_ctx(ctx, &self.ln1.0, Some(&self.ln1.1));
            let h = x.linear_ctx(ctx, &self.ff1.0, Some(&self.ff1.1));
            let h = self.act.apply(&h, ctx);
            let h = h.linear_ctx(ctx, &self.ff2.0, Some(&self.ff2.1));
            let h = h.dropout_ctx(ctx, self.dropout);
            let x = x.add_ctx(ctx, &h);
            x.layernorm_ctx(ctx, -1, 1e-5)
                .linear_ctx(ctx, &self.ln2.0, Some(&self.ln2.1))
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

    fn into_params(mut self) -> Self {
        fn tp(p: (Tensor, Tensor)) -> (Tensor, Tensor) {
            (p.0.to_param(), p.1.to_param())
        }
        self.query = tp(self.query);
        self.key = tp(self.key);
        self.value = tp(self.value);
        self.out = tp(self.out);
        self.ff1 = tp(self.ff1);
        self.ff2 = tp(self.ff2);
        self.ln1 = tp(self.ln1);
        self.ln2 = tp(self.ln2);
        self
    }
}

/// `Transformer` (extra/models/transformer.py:41-61): the adder model.
pub struct Transformer {
    /// Maximum sequence length.
    pub maxlen: i64,
    /// Alphabet size (number of symbols to predict).
    pub syms: i64,
    /// Token embedding table, shape `[maxlen + syms, embed_dim]`.
    pub embed: Tensor,
    /// Stacked transformer blocks.
    pub tbs: Vec<TransformerBlock>,
    /// Final projection to symbol logits, shape `[embed_dim, syms]`.
    pub final_: Tensor,
}

impl Transformer {
    /// Construct the adder model with `layers` blocks.
    pub fn new(
        ctx: impl AsCtx,
        syms: i64,
        maxlen: i64,
        layers: i64,
        embed_dim: i64,
        num_heads: i64,
        ff_dim: i64,
    ) -> Self {
        ctx.with_ctx(|ctx| Transformer {
            maxlen,
            syms,
            embed: Tensor::scaled_uniform(&mut *ctx, &[maxlen + syms, embed_dim], None),
            tbs: (0..layers)
                .map(|_| {
                    TransformerBlock::new(
                        &mut *ctx,
                        embed_dim,
                        num_heads,
                        ff_dim,
                        false,
                        Act::Relu,
                        0.1,
                    )
                })
                .collect(),
            final_: Tensor::scaled_uniform(&mut *ctx, &[embed_dim, syms], None),
        })
    }
    /// `Transformer.forward` — embed the one-hot inputs and run the blocks.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        self.forward_ctx(x, &mut x.ctx().borrow_mut())
    }

    #[doc(hidden)]
    pub fn forward_ctx(&self, x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
        let xs: Vec<SVal> = x.shape_ctx(ctx).into_vec();
        let (bs, seq) = (xs[0].clone(), xs[1].clone().as_i64().unwrap_or(0));
        let bs_i = xs[0].clone().as_i64().unwrap_or(0);
        let eye = Tensor::eye(&mut *ctx, seq, None, None);
        let eye = eye.unsqueeze_ctx(ctx, 0).expand_ctx(
            ctx,
            vec![bs.clone(), SVal::I(seq as i128), SVal::I(seq as i128)],
        );
        let onehot_feat = x
            .cast_ctx(ctx, crate::dtype::INT32)
            .one_hot_ctx(ctx, self.syms);
        let onehot = eye.cat_ctx(ctx, &[onehot_feat], 2).flatten_ctx(ctx, 0, 1);
        let mut x = onehot.dot_ctx(ctx, &self.embed).reshape_ctx(
            ctx,
            vec![SVal::I(bs_i as i128), SVal::I(seq as i128), SVal::I(-1)],
        );
        for tb in &self.tbs {
            x = tb.forward_ctx(&x, ctx);
        }
        let last = x.shape_ctx(ctx).into_vec()[2].clone();
        let x = x
            .reshape_ctx(ctx, vec![SVal::I(-1), last])
            .dot_ctx(ctx, &self.final_);
        let x = crate::tensor::nn::log_softmax_ctx(&x, ctx, Some(1));
        x.reshape_ctx(
            ctx,
            vec![
                SVal::I(bs_i as i128),
                SVal::I(-1),
                SVal::I(self.syms as i128),
            ],
        )
    }
}
