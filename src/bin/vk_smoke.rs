//! Vulkan smoke test (`backend_vk`): a tensor built on VK, an elementwise
//! kernel + sum, values read back through the Vulkan allocator.
//!
//! ```sh
//! cargo run --release --features backend_vk --bin vk_smoke
//! DEFAULT_DEVICE=VK:1 cargo run --release --features backend_vk --bin vk_smoke
//! ```

use autorust::engine::realize::{ProgCache, tensor_values};
use autorust::runtime::buffer::BufTable;
use autorust::runtime::device::device;
use autorust::tensor::Ctx;

fn main() {
    if !autorust::runtime::ops_vk::available() {
        println!("VK: no usable Vulkan device (loader, features or probe missing)");
        return;
    }
    println!("VK: {} device(s)", autorust::runtime::ops_vk::device_count());
    let dev = std::env::var("DEFAULT_DEVICE").unwrap_or_else(|_| "VK".to_string());

    let ctx = Ctx::new();
    let mut cache = ProgCache::default();
    let t = autorust::tensor::Tensor::from_slice(&ctx, &[1.0f32, 2.0, 3.0, 4.0], None, Some(&dev));
    // Elementwise (a vectorized upcast), a reduce, and a movement op.
    let t2 = t.add(2.0).mul(3.0);
    let sum = t2.sum(None, false, None);
    let vals = tensor_values(
        &mut ctx.raw(),
        sum.uop(),
        &mut BufTable::default(),
        &mut cache,
    );
    println!("{dev} sum((x+2)*3) = {vals:?}");
    assert!((vals[0] - 54.0).abs() < 1e-5, "VK numerics wrong: {vals:?}");

    // A small matmul exercises local memory + a barrier.
    let n = 64usize;
    let a: Vec<f32> = (0..n * n).map(|i| (i % 7) as f32 * 0.5).collect();
    let b: Vec<f32> = (0..n * n).map(|i| (i % 5) as f32 - 1.0).collect();
    let at = autorust::tensor::Tensor::from_slice(&ctx, &a, None, Some(&dev)).reshape([n, n]);
    let bt = autorust::tensor::Tensor::from_slice(&ctx, &b, None, Some(&dev)).reshape([n, n]);
    let c = at.dot(&bt);
    let got = tensor_values(&mut ctx.raw(), c.uop(), &mut BufTable::default(), &mut cache);
    let mut want = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0f32;
            for k in 0..n {
                acc += a[i * n + k] * b[k * n + j];
            }
            want[i * n + j] = acc;
        }
    }
    let mut maxdiff = 0.0f32;
    for (x, y) in got.iter().zip(&want) {
        maxdiff = maxdiff.max((x - *y as f64).abs() as f32);
    }
    println!("{dev} matmul {n}x{n}: max abs diff {maxdiff:.6}");
    assert!(maxdiff < 1e-3, "VK matmul wrong: {maxdiff}");

    println!("compiles: {}", autorust::runtime::device::total_compiles());
    let _ = device(&dev);
}
