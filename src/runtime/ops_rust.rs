//! `ops_rust` — the native CPU backend (device `RUST`).
//!
//! Kernel source is rendered as Rust by [`render_rust`], compiled to a
//! `cdylib` with `rustc`, and loaded with `libloading`. The generated kernel
//! has a fixed C ABI:
//!
//! ```text
//! pub unsafe extern "C" fn kernel(
//!     bufs: *const *mut u8,  // typed buffers, in signature order
//!     vals: *const i64,     // scalar ALU params
//!     gidx: *const i64,     // global work-item id (0..3)
//! )
//! ```
//!
//! `global_size` is fanned out across `rayon`; each call receives its own
//! `gidx` and writes a disjoint output region (tinygrad's independent-work-item
//! contract). The whole module is behind the `backend_rust` feature.
//!
//! Compilation is cached in memory and on disk. The disk key covers the source,
//! the `rustc` version, the target OS/arch, and the configured `target-cpu`, so
//! a kernel compiled for `-C target-cpu=native` is not reused on a different
//! CPU (set `AUTORUST_RUST_CPU` to a baseline to make the cache portable).

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;
use std::rc::Rc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use rayon::prelude::*;

use crate::dtype::DTypeId;
use crate::renderer::cstyle::render_rust;
use crate::renderer::{Renderer, UopRenderer};
use crate::runtime::buffer::{OpaqueT, host_of};
use crate::runtime::device::{
    Comp, Compiled, LruAllocator, Prog, TinyElf, elf_key, register_device,
};
use crate::uop::UopId;
use crate::uop::ctx::UOpCtx;

/// The fixed ABI every generated kernel exports.
pub type KernelFn = unsafe extern "C" fn(*const *mut u8, *const i64, *const i64);

/// Prelude for a generated kernel crate.
///
/// `#![no_std]` keeps each `.so` small (~5 KB instead of ~4 MB); the kernel
/// only needs `core`. Math comes from `libm` on Unix.
const PRELUDE: &str = r#"#![no_std]
#![allow(dead_code, unused_mut, unused_variables, unused_parens, unsafe_op_in_unsafe_fn)]

#[panic_handler]
fn __autorust_panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

// `core` is built with unwinding, so a `no_std` cdylib can still reference the
// personality routine even though our crate is `panic=abort`. Provide a no-op
// so `dlopen(RTLD_NOW)` resolves it; it is never called.
#[unsafe(no_mangle)]
pub extern "C" fn rust_eh_personality() {}

// `sqrt` as an inline instruction: a libm call blocks LLVM from vectorizing the
// surrounding loop, while the SSE intrinsic lowers to `llvm.sqrt` and becomes
// `vsqrtps`/`vsqrtpd`.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn sqrtf(x: f32) -> f32 {
    core::arch::x86_64::_mm_cvtss_f32(
        core::arch::x86_64::_mm_sqrt_ss(core::arch::x86_64::_mm_set_ss(x)),
    )
}
#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
unsafe fn sqrtf(x: f32) -> f32 {
    unsafe extern "C" {
        fn sqrtf(x: f32) -> f32;
    }
    sqrtf(x)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn sqrt(x: f64) -> f64 {
    core::arch::x86_64::_mm_cvtsd_f64(core::arch::x86_64::_mm_sqrt_sd(
        core::arch::x86_64::_mm_setzero_pd(),
        core::arch::x86_64::_mm_set_sd(x),
    ))
}
#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
unsafe fn sqrt(x: f64) -> f64 {
    unsafe extern "C" {
        fn sqrt(x: f64) -> f64;
    }
    sqrt(x)
}

#[cfg_attr(unix, link(name = "m"))]
unsafe extern "C" {
    fn exp2f(x: f32) -> f32;
    fn exp2(x: f64) -> f64;
    fn log2f(x: f32) -> f32;
    fn log2(x: f64) -> f64;
    fn sinf(x: f32) -> f32;
    fn sin(x: f64) -> f64;
    fn truncf(x: f32) -> f32;
    fn trunc(x: f64) -> f64;
}
"#;

/// Unique id for temp build directories and cache temp-file names, so two
/// threads compiling the same source cannot clobber each other.
static BUILD_ID: AtomicU64 = AtomicU64::new(0);

fn next_build_id() -> u64 {
    BUILD_ID.fetch_add(1, Ordering::Relaxed)
}

fn rustc_cmd() -> String {
    crate::helpers::env::get("RUSTC").unwrap_or_else(|| "rustc".to_string())
}

fn rustc_version() -> &'static str {
    static V: OnceLock<String> = OnceLock::new();
    V.get_or_init(|| {
        let out = Command::new(rustc_cmd()).arg("--version").output();
        match out {
            Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).trim().to_string(),
            _ => String::new(),
        }
    })
}

fn target_cpu() -> String {
    crate::helpers::env::get("AUTORUST_RUST_CPU").unwrap_or_else(|| "native".to_string())
}

/// Shared-library magic check (ELF / Mach-O / PE). Guards both the compile
/// cache and the runtime loader against truncated or mis-cached artifacts.
fn dylib_magic_ok(bytes: &[u8]) -> bool {
    bytes.starts_with(b"\x7fELF")
        || bytes.starts_with(&[0xcf, 0xfa, 0xed, 0xfe])
        || bytes.starts_with(&[0xce, 0xfa, 0xed, 0xfe])
        || bytes.starts_with(&[0xca, 0xfe, 0xba, 0xbe])
        || bytes.starts_with(b"MZ")
}

/// FNV-1a, matching the rest of the port's cheap content hashes.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn cache_key(src: &str) -> u64 {
    let mut key = Vec::new();
    key.extend_from_slice(b"autorust-rust-v2");
    key.extend_from_slice(PRELUDE.as_bytes());
    key.extend_from_slice(src.as_bytes());
    key.extend_from_slice(rustc_version().as_bytes());
    key.extend_from_slice(std::env::consts::OS.as_bytes());
    key.extend_from_slice(std::env::consts::ARCH.as_bytes());
    key.extend_from_slice(target_cpu().as_bytes());
    fnv1a(&key)
}

fn cache_root() -> PathBuf {
    if let Some(d) = crate::helpers::env::get_os("AUTORUST_CACHE_DIR") {
        return PathBuf::from(d);
    }
    if let Some(d) = crate::helpers::env::get_os("XDG_CACHE_HOME") {
        return PathBuf::from(d).join("autorust");
    }
    if let Some(d) = crate::helpers::env::get_os("HOME") {
        return PathBuf::from(d).join(".cache").join("autorust");
    }
    std::env::temp_dir().join("autorust-cache")
}

fn dylib_ext() -> &'static str {
    if cfg!(target_os = "macos") {
        "dylib"
    } else if cfg!(target_os = "windows") {
        "dll"
    } else {
        "so"
    }
}

/// Whether a `rustc` is available (used by default-device selection).
pub fn available() -> bool {
    !rustc_version().is_empty()
}

/// `RUST` compiler: Rust source → cdylib bytes via `rustc`.
pub struct RustcCompiler {
    cache: RefCell<HashMap<String, Vec<u8>>>,
}

impl RustcCompiler {
    /// A compiler with an empty in-memory cache.
    pub fn new() -> RustcCompiler {
        RustcCompiler {
            cache: RefCell::new(HashMap::new()),
        }
    }

    fn compile_raw(&self, src: &str) -> Vec<u8> {
        let text = format!("{PRELUDE}\n{src}\n");
        let key = cache_key(src);
        let dir = std::env::temp_dir().join(format!(
            "autorust-rust-build-{}-{key:016x}-{}",
            std::process::id(),
            next_build_id()
        ));
        std::fs::create_dir_all(&dir).expect("TGERR: create rustc build dir");
        let rs = dir.join("k.rs");
        let so = dir.join(format!("k.{}", dylib_ext()));
        std::fs::write(&rs, &text).expect("TGERR: write rust kernel source");
        if crate::helpers::env::get("AUTORUST_RUST_DUMP").is_some() {
            eprintln!("--- rust kernel ---\n{text}");
        }
        let out = Command::new(rustc_cmd())
            .args([
                "--crate-type=cdylib",
                "--edition=2024",
                "-C",
                "opt-level=3",
                "-C",
                "panic=abort",
                "-C",
                "debuginfo=0",
                "-C",
                "codegen-units=1",
                "-C",
                &format!("target-cpu={}", target_cpu()),
            ])
            .arg("-o")
            .arg(&so)
            .arg(&rs)
            .output()
            .unwrap_or_else(|e| panic!("TGERR: failed to run {}: {e}", rustc_cmd()));
        if !out.status.success() {
            let err = String::from_utf8_lossy(&out.stderr);
            let _ = std::fs::remove_dir_all(&dir);
            panic!("TGERR: rustc rejected a kernel:\n{err}\n--- source ---\n{text}");
        }
        let lib = std::fs::read(&so).expect("TGERR: read rust kernel cdylib");
        if !dylib_magic_ok(&lib) {
            panic!(
                "TGERR: rustc produced a non-dylib artifact ({} bytes, magic {:02x?})",
                lib.len(),
                &lib[..lib.len().min(8)]
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
        lib
    }
}

impl Default for RustcCompiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Comp for RustcCompiler {
    fn compile(&self, src: &str) -> Vec<u8> {
        self.compile_raw(src)
    }

    fn compile_cached(&self, src: &str) -> Vec<u8> {
        if let Some(lib) = self.cache.borrow().get(src) {
            return lib.clone();
        }
        let key = cache_key(src);
        let path = cache_root()
            .join("rust")
            .join(format!("{key:016x}.{}", dylib_ext()));
        if crate::engine::disk_cache::cache_enabled()
            && let Ok(lib) = std::fs::read(&path)
        {
            if dylib_magic_ok(&lib) {
                self.cache.borrow_mut().insert(src.to_string(), lib.clone());
                return lib;
            }
            // Corrupt cache entry (interrupted write / full disk): recompile.
            let _ = std::fs::remove_file(&path);
        }
        let lib = self.compile_raw(src);
        if crate::engine::disk_cache::cache_enabled() {
            if let Some(dir) = path.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            // write-then-rename so a concurrent reader never sees a partial file
            let tmp = path.with_extension(format!("tmp{}-{}", std::process::id(), next_build_id()));
            if std::fs::write(&tmp, &lib).is_ok() {
                let _ = std::fs::rename(&tmp, &path);
            }
        }
        self.cache.borrow_mut().insert(src.to_string(), lib.clone());
        lib
    }
}

/// `RUST` runtime program: a loaded cdylib and its kernel symbol.
pub struct RustProgram {
    /// Keep the library mapped for as long as the function pointer is used.
    _lib: libloading::Library,
    f: KernelFn,
    signature: Vec<(i64, DTypeId)>,
}

impl RustProgram {
    /// The loaded kernel entry point (for the freeze plan runner).
    pub fn kernel_fn(&self) -> KernelFn {
        self.f
    }

    /// Write the cdylib bytes to a content-addressed file and `dlopen` it.
    pub fn new(elf: &TinyElf) -> RustProgram {
        let key = elf_key(elf);
        let dir = std::env::temp_dir().join("autorust-rust-libs");
        std::fs::create_dir_all(&dir).expect("TGERR: create rust kernel lib dir");
        let path = dir.join(format!("{key:016x}.{}", dylib_ext()));
        // Publish atomically: several test threads may load the same
        // content-addressed kernel concurrently, and a reader must never see a
        // half-written file.
        let publish = || {
            let tmp = dir.join(format!(
                "{key:016x}.tmp{}-{}",
                std::process::id(),
                next_build_id()
            ));
            std::fs::write(&tmp, &elf.lib).expect("TGERR: write rust kernel cdylib");
            if std::fs::rename(&tmp, &path).is_err() {
                let _ = std::fs::remove_file(&tmp);
            }
        };
        if !dylib_magic_ok(&elf.lib) {
            panic!(
                "TGERR: expected a Rust cdylib for a frozen plan, got {} bytes (magic {:02x?}). \
                 Frozen plans must be captured with DEFAULT_DEVICE=RUST.",
                elf.lib.len(),
                &elf.lib[..elf.lib.len().min(8)]
            );
        }
        let same = std::fs::metadata(&path)
            .map(|m| m.len() == elf.lib.len() as u64)
            .unwrap_or(false);
        if !same {
            publish();
        }
        // A cache entry can be truncated/corrupted by an interrupted writer or
        // a full disk while still matching the length check. If `dlopen`
        // rejects it, re-publish the in-memory bytes and retry once.
        let lib = match unsafe { libloading::Library::new(&path) } {
            Ok(lib) => lib,
            Err(first) => {
                let _ = std::fs::remove_file(&path);
                publish();
                unsafe { libloading::Library::new(&path) }.unwrap_or_else(|e| {
                    panic!(
                        "TGERR: load rust kernel {} (first error: {first}): {e}",
                        path.display()
                    )
                })
            }
        };
        let f = unsafe {
            *lib.get::<KernelFn>(elf.name.as_bytes())
                .unwrap_or_else(|e| panic!("TGERR: rust kernel symbol {}: {e}", elf.name))
        };
        RustProgram {
            _lib: lib,
            f,
            signature: elf.signature.clone(),
        }
    }
}

impl Prog for RustProgram {
    fn call(
        &mut self,
        bufs: &[Rc<dyn OpaqueT>],
        global_size: [usize; 3],
        _local_size: Option<[usize; 3]>,
        vals: &[i64],
        wait: bool,
    ) -> Option<f64> {
        if crate::helpers::env::get("AUTORUST_DEVICE_VERBOSE").is_some() {
            eprintln!(
                "[rust] gs={global_size:?} ls={_local_size:?} vals={vals:?} bufs={}",
                self.signature.len()
            );
        }
        let st = std::time::Instant::now();
        // The kernel's parameter order is the signature order (first encounter
        // in the linear list), which is exactly how `to_program` records it.
        let mut ptrs: Vec<usize> = Vec::with_capacity(self.signature.len());
        for (slot, _) in &self.signature {
            let o = &bufs[*slot as usize];
            let h = host_of(&**o);
            // Borrow only long enough to get the base pointer; the `Rc` in
            // `bufs` keeps the allocation alive until the parallel loop joins.
            let base = h.mem.borrow().as_ptr() as usize;
            ptrs.push(base + h.offset);
        }
        let vals: Vec<i64> = vals.to_vec();
        let gs = [
            global_size[0].max(1),
            global_size[1].max(1),
            global_size[2].max(1),
        ];
        let threads = rayon::current_num_threads().max(1);
        // Chunk the first global dimension inside the kernel: the host makes
        // ~8 calls per thread instead of one call per work item.
        let nchunks = if gs[0] > 1 {
            (threads * 8).min(gs[0]).max(1)
        } else {
            1
        };
        let total = nchunks * gs[1] * gs[2];
        let f = self.f;
        // Pass addresses as `usize` so the rayon closure is `Send`.
        let ptrs_addr = ptrs.as_ptr() as usize;
        let vals_addr = vals.as_ptr() as usize;
        let min_len = (total / (threads * 2)).max(1);
        (0..total)
            .into_par_iter()
            .with_min_len(min_len)
            .for_each(move |t| {
                let c = t % nchunks;
                let rest = t / nchunks;
                let i1 = rest % gs[1];
                let i2 = rest / gs[1];
                let start = gs[0] * c / nchunks;
                let end = gs[0] * (c + 1) / nchunks;
                let g = [start as i64, end as i64, i1 as i64, i2 as i64];
                unsafe {
                    f(
                        ptrs_addr as *const *mut u8,
                        vals_addr as *const i64,
                        g.as_ptr(),
                    )
                };
            });
        if wait {
            Some(st.elapsed().as_secs_f64())
        } else {
            None
        }
    }
}

// ---------------- renderer + registration ----------------

/// `RustUopRenderer` — the `RUST` facts plus `render_rust`.
pub struct RustUopRenderer {
    facts: Renderer,
}

impl Default for RustUopRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl RustUopRenderer {
    /// Build the `RUST` renderer facts.
    pub fn new() -> RustUopRenderer {
        RustUopRenderer {
            facts: Renderer::rust(),
        }
    }
}

impl UopRenderer for RustUopRenderer {
    fn facts(&self) -> &Renderer {
        &self.facts
    }
    fn render(&self, ctx: &mut UOpCtx, lst: &[UopId]) -> String {
        render_rust(ctx, lst)
    }
}

/// The `RUST` composition + `Device._DeviceMap["RUST"]` assignment.
#[allow(clippy::type_complexity)]
pub fn register() {
    fn make() -> Rc<Compiled> {
        let renderer: Box<dyn UopRenderer> = Box::new(RustUopRenderer::new());
        let compiler: Rc<dyn Comp> = Rc::new(RustcCompiler::default());
        let ctor: Rc<dyn Fn(&TinyElf) -> Rc<RefCell<dyn Prog>>> =
            Rc::new(|elf| Rc::new(RefCell::new(RustProgram::new(elf))));
        let dev = Compiled::new("RUST", renderer, LruAllocator::new(), compiler, ctor);
        crate::runtime::cpu_gemm::install(&dev);
        dev
    }
    register_device("RUST", Rc::new(make));
}
