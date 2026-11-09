//! `ops_clang` — the portable C CPU backend (device `CC`).
//!
//! Kernel source is rendered as C by [`render_cpu_c`], compiled with `$CC`
//! (clang or gcc) to a shared library, and loaded with `libloading`. The ABI
//! and launch/chunking policy are identical to the `RUST` backend:
//!
//! ```text
//! void kernel(uint8_t** bufs, int64_t* vals, int64_t* gidx)
//! ```
//!
//! `gidx[0..2]` is the `[start, end)` chunk of the first global dimension;
//! `gidx[2]`/`gidx[3]` are the other two dimensions. This backend is both a
//! performance oracle for the Rust codegen and a fallback for systems with a C
//! compiler but no Rust toolchain.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;
use std::rc::Rc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use rayon::prelude::*;

use crate::dtype::DTypeId;
use crate::renderer::cstyle::render_cpu_c;
use crate::renderer::{Renderer, UopRenderer};
use crate::runtime::buffer::{OpaqueT, host_of};
use crate::runtime::device::{
    Comp, Compiled, LruAllocator, Prog, TinyElf, elf_key, register_device,
};
use crate::uop::UopId;
use crate::uop::ctx::UOpCtx;

/// The fixed ABI every generated C kernel exports.
pub type KernelFn = unsafe extern "C" fn(*const *mut u8, *const i64, *const i64);

const PRELUDE: &str = r#"#include <stdint.h>
#include <math.h>
#if defined(__clang__)
#define TG_BITCAST(T, x) __builtin_bit_cast(T, x)
#else
#define TG_BITCAST(T, x) ({ union { __typeof__(x) f; T t; } u; u.f = (x); u.t; })
#endif
#define TG_F2U(T, x) ({ __typeof__(x) __tgv = (x); (T)(long long)(__tgv < 0 ? 0 : __tgv); })
"#;

static BUILD_ID: AtomicU64 = AtomicU64::new(0);

fn next_build_id() -> u64 {
    BUILD_ID.fetch_add(1, Ordering::Relaxed)
}

fn cc_cmd() -> String {
    if let Some(cc) = crate::helpers::env::get("CC") {
        return cc;
    }
    // Prefer clang; fall back to the system cc (typically gcc).
    if Command::new("clang")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
    {
        "clang".to_string()
    } else {
        "cc".to_string()
    }
}

fn cc_version() -> &'static str {
    static V: OnceLock<String> = OnceLock::new();
    V.get_or_init(|| {
        let out = Command::new(cc_cmd()).arg("--version").output();
        match out {
            Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout)
                .lines()
                .next()
                .unwrap_or("")
                .to_string(),
            _ => String::new(),
        }
    })
}

/// Whether a C compiler is available.
pub fn available() -> bool {
    !cc_version().is_empty()
}

fn arch_flag() -> String {
    crate::helpers::env::get("AUTORUST_CC_ARCH").unwrap_or_else(|| "native".to_string())
}

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
    key.extend_from_slice(b"autorust-cc-v1");
    key.extend_from_slice(PRELUDE.as_bytes());
    key.extend_from_slice(src.as_bytes());
    key.extend_from_slice(cc_version().as_bytes());
    key.extend_from_slice(std::env::consts::OS.as_bytes());
    key.extend_from_slice(std::env::consts::ARCH.as_bytes());
    key.extend_from_slice(arch_flag().as_bytes());
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

/// `CC` compiler: C source → shared-library bytes via `$CC`.
pub struct ClangCompiler {
    cache: RefCell<HashMap<String, Vec<u8>>>,
}

impl ClangCompiler {
    /// A compiler with an empty in-memory cache.
    pub fn new() -> ClangCompiler {
        ClangCompiler {
            cache: RefCell::new(HashMap::new()),
        }
    }

    fn compile_raw(&self, src: &str) -> Vec<u8> {
        let text = format!("{PRELUDE}\n{src}\n");
        let key = cache_key(src);
        let dir = std::env::temp_dir().join(format!(
            "autorust-cc-build-{}-{key:016x}-{}",
            std::process::id(),
            next_build_id()
        ));
        std::fs::create_dir_all(&dir).expect("TGERR: create cc build dir");
        let cs = dir.join("k.c");
        let so = dir.join(format!("k.{}", dylib_ext()));
        std::fs::write(&cs, &text).expect("TGERR: write C kernel source");
        if crate::helpers::env::get("AUTORUST_CC_DUMP").is_some() {
            eprintln!("--- C kernel ---\n{text}");
        }
        let mut cmd = Command::new(cc_cmd());
        if cfg!(target_os = "macos") {
            cmd.arg("-dynamiclib");
        } else {
            cmd.arg("-shared");
        }
        cmd.args(["-fPIC", "-O3", "-std=gnu11", "-fno-math-errno"]);
        let arch = arch_flag();
        if arch != "none" {
            cmd.arg(format!("-march={arch}"));
        }
        cmd.arg("-o").arg(&so).arg(&cs).arg("-lm");
        let out = cmd
            .output()
            .unwrap_or_else(|e| panic!("TGERR: failed to run {}: {e}", cc_cmd()));
        if !out.status.success() {
            let err = String::from_utf8_lossy(&out.stderr);
            let _ = std::fs::remove_dir_all(&dir);
            panic!("TGERR: C compiler rejected a kernel:\n{err}\n--- source ---\n{text}");
        }
        let lib = std::fs::read(&so).expect("TGERR: read C kernel library");
        let _ = std::fs::remove_dir_all(&dir);
        lib
    }
}

impl Default for ClangCompiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Comp for ClangCompiler {
    fn compile(&self, src: &str) -> Vec<u8> {
        self.compile_raw(src)
    }

    fn compile_cached(&self, src: &str) -> Vec<u8> {
        if let Some(lib) = self.cache.borrow().get(src) {
            return lib.clone();
        }
        let key = cache_key(src);
        let path = cache_root()
            .join("cc")
            .join(format!("{key:016x}.{}", dylib_ext()));
        if crate::engine::disk_cache::cache_enabled()
            && let Ok(lib) = std::fs::read(&path)
        {
            self.cache.borrow_mut().insert(src.to_string(), lib.clone());
            return lib;
        }
        let lib = self.compile_raw(src);
        if crate::engine::disk_cache::cache_enabled() {
            if let Some(dir) = path.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let tmp = path.with_extension(format!("tmp{}-{}", std::process::id(), next_build_id()));
            if std::fs::write(&tmp, &lib).is_ok() {
                let _ = std::fs::rename(&tmp, &path);
            }
        }
        self.cache.borrow_mut().insert(src.to_string(), lib.clone());
        lib
    }
}

/// `CC` runtime program: a loaded shared library and its kernel symbol.
pub struct ClangProgram {
    _lib: libloading::Library,
    f: KernelFn,
    signature: Vec<(i64, DTypeId)>,
}

impl ClangProgram {
    /// Write the library bytes to a content-addressed file and `dlopen` it.
    pub fn new(elf: &TinyElf) -> ClangProgram {
        let key = elf_key(elf);
        let dir = std::env::temp_dir().join("autorust-cc-libs");
        std::fs::create_dir_all(&dir).expect("TGERR: create CC lib dir");
        let path = dir.join(format!("{key:016x}.{}", dylib_ext()));
        let same = std::fs::metadata(&path)
            .map(|m| m.len() == elf.lib.len() as u64)
            .unwrap_or(false);
        if !same {
            let tmp = dir.join(format!(
                "{key:016x}.tmp{}-{}",
                std::process::id(),
                next_build_id()
            ));
            std::fs::write(&tmp, &elf.lib).expect("TGERR: write CC kernel library");
            if std::fs::rename(&tmp, &path).is_err() {
                let _ = std::fs::remove_file(&tmp);
            }
        }
        let lib = unsafe { libloading::Library::new(&path) }
            .unwrap_or_else(|e| panic!("TGERR: load CC kernel {}: {e}", path.display()));
        let f = unsafe {
            *lib.get::<KernelFn>(elf.name.as_bytes())
                .unwrap_or_else(|e| panic!("TGERR: CC kernel symbol {}: {e}", elf.name))
        };
        ClangProgram {
            _lib: lib,
            f,
            signature: elf.signature.clone(),
        }
    }
}

impl Prog for ClangProgram {
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
                "[cc] gs={global_size:?} ls={_local_size:?} vals={vals:?} bufs={}",
                self.signature.len()
            );
        }
        let st = std::time::Instant::now();
        let mut ptrs: Vec<usize> = Vec::with_capacity(self.signature.len());
        for (slot, _) in &self.signature {
            let o = &bufs[*slot as usize];
            let h = host_of(&**o);
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
        let nchunks = if gs[0] > 1 {
            (threads * 8).min(gs[0]).max(1)
        } else {
            1
        };
        let total = nchunks * gs[1] * gs[2];
        let f = self.f;
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

/// `ClangUopRenderer` — the `CC` facts plus `render_cpu_c`.
pub struct ClangUopRenderer {
    facts: Renderer,
}

impl Default for ClangUopRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl ClangUopRenderer {
    /// Build the `CC` renderer facts.
    pub fn new() -> ClangUopRenderer {
        ClangUopRenderer {
            facts: Renderer::cc(),
        }
    }
}

impl UopRenderer for ClangUopRenderer {
    fn facts(&self) -> &Renderer {
        &self.facts
    }
    fn render(&self, ctx: &mut UOpCtx, lst: &[UopId]) -> String {
        render_cpu_c(ctx, lst)
    }
}

/// Register `CC` (and the `CLANG` alias).
#[allow(clippy::type_complexity)]
pub fn register() {
    fn make() -> Rc<Compiled> {
        let renderer: Box<dyn UopRenderer> = Box::new(ClangUopRenderer::new());
        let compiler: Rc<dyn Comp> = Rc::new(ClangCompiler::default());
        let ctor: Rc<dyn Fn(&TinyElf) -> Rc<RefCell<dyn Prog>>> =
            Rc::new(|elf| Rc::new(RefCell::new(ClangProgram::new(elf))));
        let dev = Compiled::new("CC", renderer, LruAllocator::new(), compiler, ctor);
        crate::runtime::cpu_gemm::install(&dev);
        dev
    }
    register_device("CC", Rc::new(make));
    register_device("CLANG", Rc::new(make));
}
