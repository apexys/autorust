//! Compare the library-GEMM path against the generated kernels on the same
//! weights, for a real ViT forward.
use autorust::models::vit::ViT;
use autorust::tensor::nn::Module;
use autorust::tensor::{Ctx, Tensor};

fn main() {
    let ctx = Ctx::new();
    Tensor::manual_seed(42);
    let layers = std::env::var("LAYERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);
    let model = ViT::build(&ctx, layers, 768, 12, 3, 224, 16, 1000).into_params();
    let x = Tensor::ones(&ctx, [1, 3, 224, 224]);
    unsafe { std::env::set_var("AUTORUST_CUBLAS", "0") };
    let a = model.forward(&x).to_list_fresh::<f32>();
    unsafe { std::env::set_var("AUTORUST_CUBLAS", "1") };
    let b = model.forward(&x).to_list_fresh::<f32>();
    let md = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let rel = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs() / x.abs().max(1e-6))
        .fold(0.0f32, f32::max);
    println!("n={} maxdiff={md:e} maxrel={rel:e}", a.len());
    println!("a[:4]={:?} b[:4]={:?}", &a[..4], &b[..4]);
}
