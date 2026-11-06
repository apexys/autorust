//! Compile-time check that the packaged layers/models implement
//! `tensor::nn::ModuleForward`.

use autorust::models::transformer::{Transformer, TransformerBlock};
use autorust::models::vit::ViT;
use autorust::tensor::Tensor;
use autorust::tensor::nn::{
    BatchNorm, Conv2d, ConvTranspose2d, Embedding, GroupNorm, InstanceNorm, LSTMCell, LayerNorm,
    LayerNorm2d, Linear, ModuleForward, RMSNorm,
};

fn assert_one<T: ModuleForward<Input = Tensor, Output = Tensor>>() {}
fn assert_lstm<
    T: ModuleForward<Input = (Tensor, Option<(Tensor, Tensor)>), Output = (Tensor, Tensor)>,
>() {
}

#[test]
fn packaged_modules_implement_module_forward() {
    assert_one::<Linear>();
    assert_one::<Conv2d>();
    assert_one::<ConvTranspose2d>();
    assert_one::<LayerNorm>();
    assert_one::<LayerNorm2d>();
    assert_one::<GroupNorm>();
    assert_one::<InstanceNorm>();
    assert_one::<RMSNorm>();
    assert_one::<BatchNorm>();
    assert_one::<Embedding>();
    assert_one::<TransformerBlock>();
    assert_one::<Transformer>();
    assert_one::<ViT>();
    assert_lstm::<LSTMCell>();
}
