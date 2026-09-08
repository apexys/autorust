//! OCL smoke test (`backend_cl`): a tensor built on CL, an elementwise
//! kernel + sum, values read back through the CL allocator.

use autorust::engine::realize::{tensor_values, ProgCache};
use autorust::runtime::device::device;
use autorust::uop::ctx::UOpCtx;

fn main() {
    let mut ctx = UOpCtx::new();
    let mut cache = ProgCache::default();
    let t = autorust::tensor::Tensor::from_slice(&mut ctx, &[1.0f32, 2.0, 3.0, 4.0], None, Some("CL"));
    let t2 = t.add(&mut ctx, 2.0);
    let sum = t2.sum(&mut ctx, None, false, None);
    let vals = tensor_values(&mut ctx, sum.uop, &mut autorust::runtime::buffer::BufTable::default(), &mut cache);
    println!("CL sum+2 = {vals:?}");
    assert!((vals[0] - 18.0).abs() < 1e-5, "CL numerics wrong: {vals:?}");
    println!("compiles: {}", autorust::runtime::device::total_compiles());
    let _ = device("CL");
}
