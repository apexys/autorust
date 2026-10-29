//! CL limitation: `CLAllocator` has no `_offset` (matching tinygrad's — only the
//! python/cuda/hip/metal/disk/dsp/npy/hcq backends implement it), so an offset
//! view cannot be lowered to a windowed `cl_mem` and `Buffer::get_buf` panics.
//!
//! Our `buffer_of` produces genuine offset views for a residual `SHRINK`/
//! `BITCAST` (see `runtime/buffer.rs`), so those must not reach the CL backend;
//! tinygrad avoids this by carrying `Buffer.offset` into the kernel's buffer arg
//! rather than slicing the opaque. Only compiled/run with `--features backend_cl`.
#![cfg(feature = "backend_cl")]

use autorust::dtype::FLOAT32;
use autorust::runtime::buffer::Buffer;
use autorust::runtime::device::device;

#[test]
fn cl_offset_view_get_buf_panics() {
    if !autorust::runtime::ops_cl::available() {
        eprintln!("skipping: no OpenCL device");
        return;
    }
    let dev = device("CL");
    let base = Buffer::new_base("CL", 2, FLOAT32); // 8 bytes
    let view = base.view(1, FLOAT32, 4); // window [4,8): needs allocator._offset

    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = view.get_buf(&dev);
    }));
    let err = res.expect_err("CL offset view must panic (CLAllocator has no _offset)");
    let msg = err
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| err.downcast_ref::<&str>().copied())
        .unwrap_or("<non-string panic>");
    assert!(
        msg.contains("need offset"),
        "unexpected panic message: {msg}"
    );
}
