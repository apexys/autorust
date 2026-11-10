//! Development-only C ABI loaded by the Python `ctypes` bridge
//! (`py/bridge.py`), plus the oracle binaries the pytest gates drive.
//!
//! This crate is not published and is not a dependency of `autorust`: it is a
//! `cdylib` shim over the public JSON graph codec (`autorust::rebuild`), so the
//! library itself stays a plain `rlib` with no C surface and no `cdylib`.
//!
//! Build it with `cargo build -p autorust-bridge`; the shared object lands in
//! the workspace `target/debug/libautorust_bridge.so`.

use std::alloc::{Layout, alloc, dealloc};

use autorust::rebuild;

/// # Safety
/// `in_ptr` must point to `in_len` bytes of JSON; `out_len` must be writable.
/// Returns a malloc'd buffer the caller must free via `autorust_free`, or null
/// on parse failure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autorust_rebuild(
    in_ptr: *const u8,
    in_len: usize,
    out_len: *mut usize,
) -> *mut u8 {
    let json = unsafe { std::slice::from_raw_parts(in_ptr, in_len) };
    match std::str::from_utf8(json).ok().and_then(|s| rebuild(s).ok()) {
        Some(out) => {
            let bytes = out.into_bytes();
            let layout = Layout::array::<u8>(bytes.len()).unwrap();
            unsafe {
                let ptr = alloc(layout);
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
                *out_len = bytes.len();
                ptr
            }
        }
        None => std::ptr::null_mut(),
    }
}

/// # Safety
/// `ptr` must have come from `autorust_rebuild` with the same `len`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autorust_free(ptr: *mut u8, len: usize) {
    if ptr.is_null() {
        return;
    }
    unsafe { dealloc(ptr, Layout::array::<u8>(len).unwrap()) }
}

/// Smoke kernel (M0): elementwise add. Real interpreter lands in M5.
///
/// # Safety
/// `a`/`b` point to `n` f32s; `out` points to `n` writable f32s.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn autorust_exec_add(a: *const f32, b: *const f32, out: *mut f32, n: usize) {
    unsafe {
        for i in 0..n {
            *out.add(i) = *a.add(i) + *b.add(i);
        }
    }
}
