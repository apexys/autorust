//! Build-script half of the frozen-ViT example: capture one forward, lower it
//! to a `FrozenPlan`, and emit the self-contained Rust file into `OUT_DIR`.

include!("src/make_vit.rs");

use std::path::PathBuf;

use autorust::engine::jit::Jit;
use autorust::engine::session::Session;

fn main() {
    // The emitted plan is RUST code; capture it on RUST even on machines where
    // CL/CUDA are available.
    unsafe { std::env::set_var("DEFAULT_DEVICE", "RUST") };
    for key in [
        "VIT_LAYERS",
        "VIT_EMBED",
        "VIT_HEADS",
        "VIT_IMG",
        "VIT_PATCH",
        "VIT_CLASSES",
    ] {
        println!("cargo:rerun-if-env-changed={key}");
    }
    println!("cargo:rerun-if-changed=src/make_vit.rs");

    let ctx = Ctx::new();
    let mut sess = Session::new();
    sess.enable_jit();
    let (model, weights, x) = make(&ctx);

    let mut jit = Jit::new();
    let _ = jit.call_many(&ctx, &mut sess, |_c, _s| vec![model.forward(&x)]);
    let built = autorust::freeze::plan::build(
        &ctx,
        &mut sess,
        &jit,
        std::slice::from_ref(&x),
        &weights,
    );
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("frozen.rs");
    autorust::freeze::emit_to(&built, out).expect("emit frozen ViT");
}
