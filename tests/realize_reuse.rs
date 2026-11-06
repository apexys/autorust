//! A realized intermediate must keep its storage when it is consumed by a graph
//! built later.
//!
//! `Tensor::realize_device_ctx` (and the existing `realize_ctx`) turn a tensor
//! into a buffer leaf whose only strong reference used to be the `Tensor`
//! itself. Moving the tensor into a consumer drops it, the weak `BufTable` alias
//! dies, and the next `prune` reallocates the storage — silently losing the
//! value. The session now pins the storage; this is the regression test.

use autorust::engine::session::Session;
use autorust::tensor::{Ctx, Tensor};

const W: usize = 3 * 16 * 16;

/// Run on CUDA when it is compiled in and present (the bug surfaced there);
/// otherwise use the default backend.
fn gpu() -> Option<&'static str> {
    #[cfg(feature = "backend_cuda")]
    if autorust::runtime::ops_cuda::available() {
        unsafe { std::env::set_var("DEFAULT_DEVICE", "CUDA") };
        return Some("CUDA");
    }
    None
}

fn inputs(ctx: &Ctx) -> (Tensor, Tensor, Tensor) {
    let dev = gpu();
    let pix: Vec<f32> = (0..3 * 32 * 32)
        .map(|i| ((i % 23) as f32) * 0.03 - 0.3)
        .collect();
    let w: Vec<f32> = (0..8 * W)
        .map(|i| ((i % 17) as f32) * 0.002 - 0.015)
        .collect();
    let b: Vec<f32> = (0..8).map(|i| (i as f32) * 0.01).collect();
    (
        Tensor::from_slice(ctx, &pix, None, dev).reshape([1, 3, 32, 32]),
        Tensor::from_slice(ctx, &w, None, dev).reshape([8, W]),
        Tensor::from_slice(ctx, &b, None, dev),
    )
}

/// The Conv2d-as-matmul patch embedding, like a ViT's first layer.
fn patch_embed(ctx: &Ctx) -> (Tensor, Tensor) {
    let (px, pw, pb) = inputs(ctx);
    let patches = px
        .reshape([1, 3, 2, 16, 2, 16])
        .permute(&[0, 2, 4, 1, 3, 5])
        .contiguous()
        .reshape([1, 4, W])
        .dot(&pw.transpose(0, 1))
        .add(&pb);
    let cls = Tensor::from_slice(ctx, &[0.5f32; 8], None, gpu()).reshape([1, 1, 8]);
    // Returning the cat too makes the test exercise the "token 0 matches" part.
    (cls.cat(&[patches.clone()], 1), patches)
}

#[test]
fn realized_intermediate_survives_reuse() {
    let ctx = Ctx::new();

    // Reference: build and realize the whole graph in one schedule.
    let mut a = Session::new();
    let (tokens, _) = patch_embed(&ctx);
    let want = tokens.to_list::<f32>(&mut a);

    // Realize the patch embedding on its own, then consume it from a new graph.
    let mut b = Session::new();
    let (_, patches) = patch_embed(&ctx);
    {
        let mut c = patches.ctx().borrow_mut();
        patches.realize_device_ctx(&mut c, &mut b);
    }
    // Consume it through a *view* and drop the realized tensor, as a model does
    // when it realizes a projection and then reshapes/transposes it.
    let view = patches.reshape([1, 1, 4, 8]).squeeze(Some(1));
    drop(patches);
    let cls = Tensor::from_slice(&ctx, &[0.5f32; 8], None, gpu()).reshape([1, 1, 8]);
    let got = cls.cat(&[view], 1).to_list::<f32>(&mut b);

    assert_eq!(want, got, "realized intermediate changed after reuse");
}
