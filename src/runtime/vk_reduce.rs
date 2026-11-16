//! Fused full reductions for Vulkan (`sum` / `amax`).
//!
//! One streaming pass into 256 partials plus a tiny finish pass. The generated
//! split-reduce path is limited by a 32-thread block for these large flat
//! reductions; this kernel uses 256-thread blocks and shared-memory tree
//! reductions.

use crate::dtype;
use crate::engine::session::HostCall;
use crate::renderer::spirv::VkModule;
use crate::runtime::buffer::Buffer;
use crate::runtime::device::device;

/// Partials written by the streaming pass.
pub const BLOCKS: i64 = 256;
/// Threads per block in both passes.
pub const THREADS: i64 = 256;
/// Smallest tensor worth the two host launches.
pub const MIN_ELEMS: i64 = 1 << 20;
/// Sum entry points.
pub const SUM_PARTIALS: &str = "ar_vk_sum_partials";
/// Sum finish entry point.
pub const SUM_FINISH: &str = "ar_vk_sum_finish";
/// Amax entry points.
pub const AMAX_PARTIALS: &str = "ar_vk_amax_partials";
/// Amax finish entry point.
pub const AMAX_FINISH: &str = "ar_vk_amax_finish";

fn payload(bytes: &[u8]) -> String {
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    VkModule {
        slots: vec![0, 1, 2],
        local: [THREADS as u32, 1, 1],
        words,
    }
    .encode()
}

fn scalar_i32(dev: &crate::runtime::device::Compiled, value: i32) -> Buffer {
    let b = Buffer::new_base("VK", 1, dtype::INT32);
    b.ensure_allocated(dev);
    dev.copyin_to(&*b.get_buf(dev), &value.to_le_bytes());
    b
}

fn call(
    kind: &'static str,
    bytes: &[u8],
    mut bufs: Vec<Buffer>,
    n: i64,
    global: [i64; 3],
) -> HostCall {
    let dev = device("VK");
    assert!(n <= i32::MAX as i64, "VK reduce count exceeds i32");
    bufs.push(scalar_i32(&dev, n as i32));
    HostCall {
        device: "VK".into(),
        src: payload(bytes),
        name: kind.to_string(),
        signature: vec![(0, dtype::FLOAT32), (1, dtype::FLOAT32), (2, dtype::INT32)],
        bufs,
        global,
        local: Some([THREADS, 1, 1]),
        vals: Vec::new(),
    }
}

/// First pass for a sum: `X -> partials`.
pub fn sum_partials(bufs: Vec<Buffer>, n: i64) -> HostCall {
    call(
        SUM_PARTIALS,
        include_bytes!("vk_reduce_sum.spv"),
        bufs,
        n,
        [BLOCKS, 1, 1],
    )
}

/// Finish pass for a sum: `partials -> scalar`.
pub fn sum_finish(bufs: Vec<Buffer>, blocks: i64) -> HostCall {
    call(
        SUM_FINISH,
        include_bytes!("vk_reduce_sum_finish.spv"),
        bufs,
        blocks,
        [1, 1, 1],
    )
}

/// First pass for an amax: `X -> partials`.
pub fn amax_partials(bufs: Vec<Buffer>, n: i64) -> HostCall {
    call(
        AMAX_PARTIALS,
        include_bytes!("vk_reduce_amax.spv"),
        bufs,
        n,
        [BLOCKS, 1, 1],
    )
}

/// Finish pass for an amax: `partials -> scalar`.
pub fn amax_finish(bufs: Vec<Buffer>, blocks: i64) -> HostCall {
    call(
        AMAX_FINISH,
        include_bytes!("vk_reduce_amax_finish.spv"),
        bufs,
        blocks,
        [1, 1, 1],
    )
}
