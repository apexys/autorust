//! Metal smoke test (`backend_metal`, macOS): a tensor built on METAL, an
//! elementwise kernel + sum, values read back through the Metal allocator.

#[cfg(all(feature = "backend_metal", target_os = "macos"))]
fn main() {
    use autorust::engine::realize::{ProgCache, tensor_values};
    use autorust::runtime::device::device;
    use autorust::tensor::Ctx;

    let ctx = Ctx::new();
    let mut cache = ProgCache::default();
    let t =
        autorust::tensor::Tensor::from_slice(&ctx, &[1.0f32, 2.0, 3.0, 4.0], None, Some("METAL"));
    let t2 = t.add(2.0);
    let sum = t2.sum(None, false, None);
    let vals = tensor_values(
        &mut ctx.raw(),
        sum.uop(),
        &mut autorust::runtime::buffer::BufTable::default(),
        &mut cache,
    );
    println!("METAL sum+2 = {vals:?}");
    assert!(
        (vals[0] - 18.0).abs() < 1e-5,
        "METAL numerics wrong: {vals:?}"
    );
    println!("compiles: {}", autorust::runtime::device::total_compiles());
    let _ = device("METAL");
}

#[cfg(not(all(feature = "backend_metal", target_os = "macos")))]
fn main() {
    eprintln!("metal_smoke: METAL backend is only available on macOS");
}
