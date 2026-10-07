//! `extra/models/vit.py` — the Vision Transformer. `ViT::new(layers, embed_dim,
//! num_heads)` mirrors the tinygrad defaults (3×224×224 input, patch 16, 1000
//! classes); `ViT::build` takes the shape explicitly for other datasets (e.g.
//! MNIST upscaled to 224 with 10 classes).

use crate::models::transformer::{Act, TransformerBlock};
use crate::tensor::nn::Module;
use crate::tensor::{Ix, Tensor};
use crate::uop::ctx::UOpCtx;
use crate::uop::sint::SVal;

/// `ViT` (extra/models/vit.py:6-30).
pub struct ViT {
    pub embedding: (Tensor, Tensor),
    pub embed_dim: i64,
    pub cls: Tensor,
    pub pos_embedding: Tensor,
    pub tbs: Vec<TransformerBlock>,
    pub encoder_norm: (Tensor, Tensor),
    pub head: (Tensor, Tensor),
    pub patch_size: i64,
}

impl ViT {
    /// tinygrad `ViT(layers=12, embed_dim=192, num_heads=3)`.
    pub fn new(ctx: &mut UOpCtx, layers: i64, embed_dim: i64, num_heads: i64) -> Self {
        Self::build(ctx, layers, embed_dim, num_heads, 3, 224, 16, 1000)
    }

    /// The generalized constructor (shape params that tinygrad hardcodes).
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        ctx: &mut UOpCtx,
        layers: i64,
        embed_dim: i64,
        num_heads: i64,
        in_channels: i64,
        img_size: i64,
        patch_size: i64,
        num_classes: i64,
    ) -> Self {
        let side = img_size / patch_size;
        let num_patches = side * side;
        ViT {
            embedding: (
                Tensor::uniform(ctx, &[embed_dim, in_channels, patch_size, patch_size], None, None, None, None),
                Tensor::zeros(ctx, [embed_dim]),
            ),
            embed_dim,
            cls: Tensor::ones(ctx, [1, 1, embed_dim]),
            pos_embedding: Tensor::ones(ctx, [1, num_patches + 1, embed_dim]),
            tbs: (0..layers)
                .map(|_| TransformerBlock::new(ctx, embed_dim, num_heads, embed_dim * 4, true, Act::Gelu, 0.1))
                .collect(),
            encoder_norm: (Tensor::uniform(ctx, &[embed_dim], None, None, None, None), Tensor::zeros(ctx, [embed_dim])),
            head: (Tensor::uniform(ctx, &[embed_dim, num_classes], None, None, None, None), Tensor::zeros(ctx, [num_classes])),
            patch_size,
        }
    }

    /// `ViT.patch_embed` (vit.py:19-22).
    pub fn patch_embed(&self, x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
        let x = x.conv2d(ctx, &self.embedding.0, Some(&self.embedding.1), 1, self.patch_size, 1, &[0, 0], None);
        let xs: Vec<SVal> = x.shape(ctx).into_vec();
        x.reshape(ctx, vec![xs[0].clone(), xs[1].clone(), SVal::I(-1)]).permute(ctx, &[0, 2, 1])
    }

    /// `ViT.forward` (vit.py:24-30).
    pub fn forward(&self, x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
        let bs = x.shape(ctx).into_vec()[0].clone();
        let z = Tensor::zeros(ctx, vec![bs, SVal::I(1), SVal::I(1)]);
        let ce = self.cls.add(ctx, &z);
        let pe = self.patch_embed(x, ctx);
        let x = ce.cat(ctx, &[pe], 1);
        let mut x = x.add(ctx, &self.pos_embedding);
        for tb in &self.tbs {
            x = tb.forward(&x, ctx);
        }
        let x = x.layernorm(ctx, -1, 1e-5).linear(ctx, &self.encoder_norm.0, Some(&self.encoder_norm.1));
        let cls_tok = x.getitem(ctx, &[Ix::All, Ix::Index(0)]);
        cls_tok.linear(ctx, &self.head.0, Some(&self.head.1))
    }
}

impl Module for ViT {
    fn named_params(&self) -> Box<dyn Iterator<Item = (String, Tensor)> + '_> {
        let mut v = Vec::new();
        v.push(("embedding.0".to_string(), self.embedding.0.clone()));
        v.push(("embedding.1".to_string(), self.embedding.1.clone()));
        v.push(("cls".to_string(), self.cls.clone()));
        v.push(("pos_embedding".to_string(), self.pos_embedding.clone()));
        for (i, tb) in self.tbs.iter().enumerate() {
            for (n, t) in tb.named_params() {
                v.push((format!("tbs.{i}.{n}"), t));
            }
        }
        v.push(("encoder_norm.0".to_string(), self.encoder_norm.0.clone()));
        v.push(("encoder_norm.1".to_string(), self.encoder_norm.1.clone()));
        v.push(("head.0".to_string(), self.head.0.clone()));
        v.push(("head.1".to_string(), self.head.1.clone()));
        Box::new(v.into_iter())
    }

    fn into_params(mut self, ctx: &mut UOpCtx) -> Self {
        fn tp(ctx: &mut UOpCtx, p: (Tensor, Tensor)) -> (Tensor, Tensor) {
            (p.0.to_param(ctx), p.1.to_param(ctx))
        }
        self.embedding = tp(ctx, self.embedding);
        self.cls = self.cls.to_param(ctx);
        self.pos_embedding = self.pos_embedding.to_param(ctx);
        self.tbs = self.tbs.into_iter().map(|tb| tb.into_params(ctx)).collect();
        self.encoder_norm = tp(ctx, self.encoder_norm);
        self.head = tp(ctx, self.head);
        self
    }
}
