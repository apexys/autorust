//! Backend memory-capacity reporting.

use autorust::runtime::device::{LruAllocator, device};

#[test]
fn host_allocator_reports_no_capacity() {
    assert!(LruAllocator::new().memory_info().is_none());
}

#[cfg(all(
    feature = "backend_cuda",
    any(target_os = "linux", target_os = "windows")
))]
#[test]
fn cuda_reports_free_and_total_memory() {
    if !autorust::runtime::ops_cuda::available() {
        return;
    }
    let info = device("CUDA").memory_info().expect("CUDA memory info");
    assert!(info.total > 0, "{info:?}");
    let free = info.free.expect("CUDA reports free memory");
    assert!(free <= info.total, "{info:?}");
}

#[cfg(feature = "backend_cl")]
#[test]
fn opencl_reports_total_memory() {
    if !autorust::runtime::ops_cl::available() {
        return;
    }
    let info = device("CL").memory_info().expect("OpenCL memory info");
    assert!(info.total > 0, "{info:?}");
    assert!(info.free.is_none(), "{info:?}");
}

#[cfg(all(feature = "backend_metal", target_os = "macos"))]
#[test]
fn metal_reports_a_working_set() {
    let Some(info) = device("METAL").memory_info() else {
        return;
    };
    assert!(info.total > 0, "{info:?}");
    if let Some(free) = info.free {
        assert!(free <= info.total, "{info:?}");
    }
}
