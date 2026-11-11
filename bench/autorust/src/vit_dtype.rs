//! Cast a ViT's weights to another dtype (bf16) for the mixed-precision row.
//!
//! The model builder has no dtype knob (tinygrad's `ViT` does not either), so
//! the benchmark rebuilds the struct with cast parameter leaves.

use autorust::dtype::DTypeId;
use autorust::models::transformer::TransformerBlock;
use autorust::models::vit::ViT;
use autorust::tensor::Tensor;

fn one(t: &Tensor, dt: DTypeId) -> Tensor {
    t.cast(dt).to_param()
}

fn pair(t: &(Tensor, Tensor), dt: DTypeId) -> (Tensor, Tensor) {
    (one(&t.0, dt), one(&t.1, dt))
}

fn block(b: &TransformerBlock, dt: DTypeId) -> TransformerBlock {
    TransformerBlock {
        num_heads: b.num_heads,
        head_size: b.head_size,
        prenorm: b.prenorm,
        act: b.act,
        dropout: b.dropout,
        query: pair(&b.query, dt),
        key: pair(&b.key, dt),
        value: pair(&b.value, dt),
        out: pair(&b.out, dt),
        ff1: pair(&b.ff1, dt),
        ff2: pair(&b.ff2, dt),
        ln1: pair(&b.ln1, dt),
        ln2: pair(&b.ln2, dt),
    }
}

/// Return a copy of `v` whose weights are `dt` parameter leaves.
pub fn cast(v: &ViT, dt: DTypeId) -> ViT {
    ViT {
        embedding: pair(&v.embedding, dt),
        embed_dim: v.embed_dim,
        cls: one(&v.cls, dt),
        pos_embedding: one(&v.pos_embedding, dt),
        tbs: v.tbs.iter().map(|b| block(b, dt)).collect(),
        encoder_norm: pair(&v.encoder_norm, dt),
        head: pair(&v.head, dt),
        patch_size: v.patch_size,
    }
}
