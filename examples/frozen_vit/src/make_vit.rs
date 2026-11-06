// Shared deterministic ViT construction, included by both `build.rs` and
// `main.rs`. Keeping it in one included file avoids a model subcrate.

use autorust::models::vit::ViT;
use autorust::tensor::nn::Module;
use autorust::tensor::{Ctx, Tensor};

fn env_i64(key: &str, default: i64) -> i64 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Build a deterministic ViT and its input/weights.
///
/// Defaults are small so the build-script example is quick; set
/// `VIT_LAYERS=12 VIT_EMBED=768 VIT_HEADS=12 VIT_IMG=224 VIT_PATCH=16` for
/// ViT-B/16.
pub fn make(ctx: &Ctx) -> (ViT, Vec<Tensor>, Tensor) {
    let layers = env_i64("VIT_LAYERS", 1);
    let embed = env_i64("VIT_EMBED", 192);
    let heads = env_i64("VIT_HEADS", 3);
    let img = env_i64("VIT_IMG", 64);
    let patch = env_i64("VIT_PATCH", 16);
    let classes = env_i64("VIT_CLASSES", 10);

    Tensor::manual_seed(42);
    let vit = ViT::build(ctx, layers, embed, heads, 3, img, patch, classes).into_params();
    let weights: Vec<Tensor> = vit.named_params().map(|(_, t)| t).collect();
    let x = Tensor::zeros(ctx, [1, 3, img, img]);
    (vit, weights, x)
}
