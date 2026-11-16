//! Fused Adam/AdamW/LAMB update for Vulkan (`AUTORUST_OPTIM_KERNEL`, default on).
//!
//! Same contract as [`crate::runtime::cuda_optim`]/[`crate::runtime::cl_optim`]:
//! one 256-invocation dispatch updates `m`, `v` and `p` from one load of the
//! gradient and moments. The Vulkan compute path needs explicit descriptor
//! sets, so the kernel is a precompiled SPIR-V module (`vk_optim.spv`) wrapped
//! in the normal `VkModule` payload. The five hyperparameters and the element
//! count are one-element storage buffers; this keeps one module for every
//! optimizer configuration and avoids scalar push-constant plumbing.

use crate::dtype;
use crate::engine::session::HostCall;
use crate::renderer::spirv::VkModule;
use crate::runtime::buffer::Buffer;
use crate::runtime::device::device;

/// Smallest parameter worth a host kernel.
pub const MIN_ELEMS: i64 = 256;

/// Kernel entry point.
pub const NAME: &str = "autorust_adamw";

/// Work-items per group.
pub const THREADS: i64 = 256;

const SPV: &[u8] = include_bytes!("vk_optim.spv");

fn payload() -> String {
    let words: Vec<u32> = SPV
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    VkModule {
        slots: (0..12).collect(),
        local: [THREADS as u32, 1, 1],
        words,
    }
    .encode()
}

fn scalar_f32(dev: &crate::runtime::device::Compiled, value: f32) -> Buffer {
    let b = Buffer::new_base("VK", 1, dtype::FLOAT32);
    b.ensure_allocated(dev);
    dev.copyin_to(&*b.get_buf(dev), &value.to_le_bytes());
    b
}

fn scalar_i32(dev: &crate::runtime::device::Compiled, value: i32) -> Buffer {
    let b = Buffer::new_base("VK", 1, dtype::INT32);
    b.ensure_allocated(dev);
    dev.copyin_to(&*b.get_buf(dev), &value.to_le_bytes());
    b
}

/// Captured-launch descriptor for one parameter's update.
#[allow(clippy::too_many_arguments)]
pub fn host_call(
    mut bufs: Vec<Buffer>,
    n: i64,
    lr: f32,
    b1: f32,
    b2: f32,
    eps: f32,
    wd: f32,
) -> HostCall {
    assert_eq!(bufs.len(), 6, "VK optimizer expects six parameter buffers");
    assert!(
        n <= i32::MAX as i64,
        "VK optimizer parameter exceeds i32 indexing"
    );
    let dev = device("VK");
    bufs.push(scalar_f32(&dev, lr));
    bufs.push(scalar_f32(&dev, b1));
    bufs.push(scalar_f32(&dev, b2));
    bufs.push(scalar_f32(&dev, eps));
    bufs.push(scalar_f32(&dev, wd));
    bufs.push(scalar_i32(&dev, n as i32));
    let mut signature = Vec::with_capacity(12);
    for slot in 0..11 {
        signature.push((slot, dtype::FLOAT32));
    }
    signature.push((11, dtype::INT32));
    HostCall {
        device: "VK".into(),
        src: payload(),
        name: NAME.to_string(),
        signature,
        bufs,
        global: [(n + THREADS - 1) / THREADS, 1, 1],
        local: Some([THREADS, 1, 1]),
        vals: Vec::new(),
    }
}
