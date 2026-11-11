//! Version-agnostic, dynamically loaded NVRTC (`libnvrtc`).
//!
//! tinygrad finds NVRTC by scanning the loader search path for the first
//! `libnvrtc.so*` ELF file (`runtime/support/c.py::DLL.findlib`) and honours
//! `NVRTC_PATH`. cudarc instead bakes the toolkit version it was *built*
//! against into its candidate list (`CUDA_MAJOR_VERSION`), so a binary built on
//! CUDA 12 only looks for `libnvrtc.so.12` plus the unversioned `-dev` symlink.
//! On a machine with a different toolkit — and no `cuda-nvrtc-dev` symlink —
//! that load fails and cudarc aborts the process.
//!
//! This module performs the lookup itself and binds the handful of NVRTC
//! entry points the compiler needs. Any installed toolkit works: the CUDA
//! *driver* API (loaded separately by cudarc) is the version-stable ABI, and
//! PTX is forward compatible, so a kernel compiled by NVRTC 13 runs on a 12
//! driver as long as the driver understands the emitted PTX ISA. A *newer*
//! NVRTC can emit an ISA the installed driver rejects; [`compile_validated`]
//! lets the CUDA probe try every loaded NVRTC newest-first and pin the first
//! one whose PTX the driver actually loads. When no candidate works, the CUDA
//! backend reports itself unavailable instead of panicking, so device selection
//! falls through to the next backend.
//!
//! Search order:
//!
//! 1. `AUTORUST_NVRTC_LIB` (path or soname) and tinygrad's `NVRTC_PATH`.
//!    An explicit override is authoritative and disables the fallback.
//! 2. Every `libnvrtc.so[.N]` found under `LD_LIBRARY_PATH`, `CUDA_HOME` /
//!    `CUDA_PATH` / `CUDA_ROOT` / `CONDA_PREFIX`, the multiarch library dirs,
//!    and the `cuda*` toolkits under `/usr/local` and `/opt`.
//! 3. Bare sonames (`libnvrtc.so.20` … `libnvrtc.so.9`) resolved by the
//!    system loader.
//!
//! On Windows the same rules apply to `nvrtc64_*.dll` / `nvrtc.dll` under
//! `PATH` and the toolkit roots.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use libloading::Library;

/// NVRTC result code (`nvrtcResult`; `0` is success).
type NvrtcResult = c_int;

type FnVersion = unsafe extern "C" fn(*mut c_int, *mut c_int) -> NvrtcResult;
type FnErrorString = unsafe extern "C" fn(NvrtcResult) -> *const c_char;
type FnCreate = unsafe extern "C" fn(
    *mut *mut c_void,
    *const c_char,
    *const c_char,
    c_int,
    *const *const c_char,
    *const *const c_char,
) -> NvrtcResult;
type FnCompile = unsafe extern "C" fn(*mut c_void, c_int, *const *const c_char) -> NvrtcResult;
type FnSize = unsafe extern "C" fn(*mut c_void, *mut usize) -> NvrtcResult;
type FnGet = unsafe extern "C" fn(*mut c_void, *mut c_char) -> NvrtcResult;
type FnDestroy = unsafe extern "C" fn(*mut *mut c_void) -> NvrtcResult;
type FnGetArchs = unsafe extern "C" fn(*mut c_int) -> NvrtcResult;

/// Bound NVRTC entry points plus the library that owns them.
struct Api {
    _lib: Library,
    /// Candidate path/soname, for diagnostics.
    name: String,
    version: FnVersion,
    error_string: FnErrorString,
    create: FnCreate,
    compile: FnCompile,
    ptx_size: FnSize,
    ptx: FnGet,
    log_size: FnSize,
    log: FnGet,
    destroy: FnDestroy,
    num_archs: Option<FnGetArchs>,
    get_archs: Option<FnGetArchs>,
}

/// Every NVRTC that loaded successfully, in search order (newest first).
fn apis() -> &'static [Api] {
    static APIS: OnceLock<Vec<Api>> = OnceLock::new();
    APIS.get_or_init(load_all)
}

/// Index of the NVRTC that produced driver-compatible PTX, or 0 (newest)
/// before the CUDA probe has selected one.
static ACTIVE: AtomicUsize = AtomicUsize::new(0);
/// Whether [`compile_validated`] has accepted a candidate.
static VALIDATED: AtomicBool = AtomicBool::new(false);

fn api() -> Option<&'static Api> {
    let apis = apis();
    let index = ACTIVE
        .load(Ordering::Acquire)
        .min(apis.len().saturating_sub(1));
    apis.get(index)
}

/// Whether an NVRTC library could be loaded.
pub fn available() -> bool {
    api().is_some()
}

/// Runtime NVRTC version (`(major, minor)`), for diagnostics.
pub fn version() -> Option<(c_int, c_int)> {
    api()?.runtime_version()
}

/// SM numbers this NVRTC can target (`80` for `sm_80`); empty when the
/// loaded library predates `nvrtcGetSupportedArchs`.
pub fn supported_sms() -> Vec<c_int> {
    api().map(Api::archs).unwrap_or_default()
}

/// Whether the active NVRTC was accepted by [`compile_validated`].
pub(crate) fn has_validated() -> bool {
    VALIDATED.load(Ordering::Acquire)
}

/// Whether the active NVRTC advertises support for `arch`. A library that
/// predates `nvrtcGetSupportedArchs` is given the benefit of the doubt.
pub(crate) fn supports_arch(arch: &str) -> bool {
    let Some(api) = api() else {
        return false;
    };
    let arches = api.archs();
    match sm_of(arch) {
        Some(sm) => arches.is_empty() || arches.contains(&sm),
        None => true,
    }
}

/// Compile `src` to PTX for `arch` (for example `sm_86`), adding one
/// `-I<dir>` option per include directory, using the active NVRTC.
pub fn compile(src: &str, arch: &str, include_dirs: &[String]) -> Result<Vec<u8>, String> {
    let Some(api) = api() else {
        return Err(missing_library());
    };
    compile_with(api, src, arch, include_dirs)
}

/// Compile with every loaded NVRTC, newest first, and return the first PTX for
/// which `validate` succeeds. The accepting library becomes the active NVRTC
/// for all later [`compile`] calls.
///
/// The CUDA device probe uses this to reject an NVRTC whose PTX is newer than
/// the installed driver, then retry with the next-oldest installed toolkit. An
/// explicit `AUTORUST_NVRTC_LIB` / `NVRTC_PATH` leaves exactly one candidate,
/// so the fallback is disabled for pinned toolkits.
pub fn compile_validated<T, F>(
    src: &str,
    arch: &str,
    include_dirs: &[String],
    mut validate: F,
) -> Result<(Vec<u8>, T), String>
where
    F: FnMut(&[u8]) -> Result<T, String>,
{
    let apis = apis();
    if apis.is_empty() {
        return Err(missing_library());
    }
    let mut failures = Vec::with_capacity(apis.len());
    for (index, api) in apis.iter().enumerate() {
        let ptx = match compile_with(api, src, arch, include_dirs) {
            Ok(ptx) => ptx,
            Err(e) => {
                failures.push(format!("{}: {e}", api.label()));
                continue;
            }
        };
        match validate(&ptx) {
            Ok(value) => {
                ACTIVE.store(index, Ordering::Release);
                VALIDATED.store(true, Ordering::Release);
                if index > 0 {
                    eprintln!(
                        "autorust: pinned {} for {arch}; newer NVRTC candidates were rejected by the driver",
                        api.label()
                    );
                }
                return Ok((ptx, value));
            }
            Err(e) => failures.push(format!("{}: PTX rejected: {e}", api.label())),
        }
    }
    Err(format!(
        "no installed NVRTC produced driver-compatible PTX for {arch}:\n  {}",
        failures.join("\n  ")
    ))
}

fn missing_library() -> String {
    "libnvrtc not found (install the CUDA NVRTC library or set AUTORUST_NVRTC_LIB)".to_string()
}

fn compile_with(
    api: &Api,
    src: &str,
    arch: &str,
    include_dirs: &[String],
) -> Result<Vec<u8>, String> {
    // Fail fast with an actionable message when this NVRTC is too old for the
    // GPU, instead of a generic COMPILATION error from the source.
    if let Some(sm) = sm_of(arch) {
        let arches = api.archs();
        if !arches.is_empty() && !arches.contains(&sm) {
            let listed: Vec<String> = arches.iter().map(|n| format!("sm_{n}")).collect();
            let ver = api
                .runtime_version()
                .map(|(a, b)| format!("{a}.{b}"))
                .unwrap_or_else(|| "?".to_string());
            return Err(format!(
                "NVRTC {ver} does not support {arch} (supported: {})",
                listed.join(", ")
            ));
        }
    }
    let src = CString::new(src).map_err(|_| "kernel source contains a NUL byte".to_string())?;
    let name = CString::new("<autorust>").expect("static string has no NUL");
    let mut options: Vec<CString> = Vec::with_capacity(include_dirs.len() + 1);
    options.push(CString::new(format!("--gpu-architecture={arch}")).expect("arch has no NUL"));
    for dir in include_dirs {
        options.push(
            CString::new(format!("-I{dir}")).map_err(|_| "include path contains a NUL byte")?,
        );
    }
    let opt_ptrs: Vec<*const c_char> = options.iter().map(|o| o.as_ptr()).collect();

    let mut prog: *mut c_void = std::ptr::null_mut();
    // SAFETY: all pointers are valid for the duration of the calls and the
    // signatures match the NVRTC C API (`nvrtc.h`).
    unsafe {
        let rc = (api.create)(
            &mut prog,
            src.as_ptr(),
            name.as_ptr(),
            0,
            std::ptr::null(),
            std::ptr::null(),
        );
        if rc != 0 {
            return Err(format!("nvrtcCreateProgram failed: {}", api.error_text(rc)));
        }
        let rc = (api.compile)(prog, opt_ptrs.len() as c_int, opt_ptrs.as_ptr());
        if rc != 0 {
            let log = api.program_log(prog);
            let _ = (api.destroy)(&mut prog);
            return Err(format!(
                "{}: {}\n{}",
                arch,
                api.error_text(rc),
                log.trim_end()
            ));
        }
        let size = api.data_size(prog, api.ptx_size);
        let mut out = vec![0u8; size];
        let rc = (api.ptx)(prog, out.as_mut_ptr() as *mut c_char);
        let _ = (api.destroy)(&mut prog);
        if rc != 0 {
            return Err(format!("nvrtcGetPTX failed: {}", api.error_text(rc)));
        }
        truncate_nul(&mut out);
        Ok(out)
    }
}

impl Api {
    fn runtime_version(&self) -> Option<(c_int, c_int)> {
        let (mut major, mut minor) = (0, 0);
        // SAFETY: `nvrtcVersion` writes two ints through valid pointers.
        let rc = unsafe { (self.version)(&mut major, &mut minor) };
        (rc == 0).then_some((major, minor))
    }

    fn archs(&self) -> Vec<c_int> {
        let (Some(num), Some(get)) = (self.num_archs, self.get_archs) else {
            return Vec::new();
        };
        let mut count: c_int = 0;
        // SAFETY: `nvrtcGetNumSupportedArchs` only writes the count.
        if unsafe { num(&mut count) } != 0 || count <= 0 {
            return Vec::new();
        }
        let mut sms = vec![0 as c_int; count as usize];
        // SAFETY: `nvrtcGetSupportedArchs` fills `count` ints.
        if unsafe { get(sms.as_mut_ptr()) } != 0 {
            return Vec::new();
        }
        sms
    }

    fn label(&self) -> String {
        match self.runtime_version() {
            Some((major, minor)) => format!("{} (NVRTC {major}.{minor})", self.name),
            None => self.name.clone(),
        }
    }

    fn error_text(&self, rc: NvrtcResult) -> String {
        // SAFETY: `nvrtcGetErrorString` returns a static NUL-terminated string.
        let ptr = unsafe { (self.error_string)(rc) };
        if ptr.is_null() {
            return format!("NVRTC error {rc}");
        }
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }

    fn program_log(&self, prog: *mut c_void) -> String {
        // SAFETY: `prog` is a live program handle.
        let size = self.data_size(prog, self.log_size);
        if size == 0 {
            return String::new();
        }
        let mut buf = vec![0u8; size];
        // SAFETY: the buffer is `size` bytes, as reported by NVRTC.
        let rc = unsafe { (self.log)(prog, buf.as_mut_ptr() as *mut c_char) };
        if rc != 0 {
            return String::new();
        }
        truncate_nul(&mut buf);
        String::from_utf8_lossy(&buf).into_owned()
    }

    fn data_size(&self, prog: *mut c_void, f: FnSize) -> usize {
        let mut size = 0usize;
        // SAFETY: `prog` is live and `size` is a valid out-pointer.
        let rc = unsafe { f(prog, &mut size) };
        if rc != 0 { 0 } else { size }
    }
}

fn truncate_nul(buf: &mut Vec<u8>) {
    if let Some(pos) = buf.iter().position(|&b| b == 0) {
        buf.truncate(pos);
    }
}

/// `sm_86` -> `Some(86)`; `sm_90a` -> `Some(90)`.
fn sm_of(arch: &str) -> Option<c_int> {
    arch.strip_prefix("sm_")?.trim_end_matches('a').parse().ok()
}

fn load_all() -> Vec<Api> {
    let mut out = Vec::new();
    for candidate in candidates() {
        // SAFETY: loading an arbitrary shared library is inherently unsafe; a
        // failed load is skipped and every symbol is resolved before use.
        let lib = match unsafe { Library::new(&candidate) } {
            Ok(lib) => lib,
            Err(_) => continue,
        };
        match unsafe { bind(lib, candidate) } {
            Ok(api) => out.push(api),
            Err(()) => continue,
        }
    }
    out
}

unsafe fn bind(lib: Library, name: String) -> Result<Api, ()> {
    // `get` needs a concrete `T`; spell out each signature at the call site.
    macro_rules! bind_required {
        ($name:literal, $ty:ty) => {{
            // SAFETY: the name/signature pairs come from `nvrtc.h`.
            let sym: std::result::Result<libloading::Symbol<'_, $ty>, _> =
                unsafe { lib.get($name.as_bytes()) };
            *sym.map_err(|_| ())?
        }};
    }
    macro_rules! bind_optional {
        ($name:literal, $ty:ty) => {{
            // SAFETY: the name/signature pairs come from `nvrtc.h`.
            let sym: std::result::Result<libloading::Symbol<'_, $ty>, _> =
                unsafe { lib.get($name.as_bytes()) };
            sym.ok().map(|s| *s)
        }};
    }
    // Bind every symbol first: `Symbol` borrows `lib`, so the library can only
    // be moved into `Api` after all the borrows have ended.
    let version: FnVersion = bind_required!("nvrtcVersion", FnVersion);
    let error_string: FnErrorString = bind_required!("nvrtcGetErrorString", FnErrorString);
    let create: FnCreate = bind_required!("nvrtcCreateProgram", FnCreate);
    let compile: FnCompile = bind_required!("nvrtcCompileProgram", FnCompile);
    let ptx_size: FnSize = bind_required!("nvrtcGetPTXSize", FnSize);
    let ptx: FnGet = bind_required!("nvrtcGetPTX", FnGet);
    let log_size: FnSize = bind_required!("nvrtcGetProgramLogSize", FnSize);
    let log: FnGet = bind_required!("nvrtcGetProgramLog", FnGet);
    let destroy: FnDestroy = bind_required!("nvrtcDestroyProgram", FnDestroy);
    let num_archs = bind_optional!("nvrtcGetNumSupportedArchs", FnGetArchs);
    let get_archs = bind_optional!("nvrtcGetSupportedArchs", FnGetArchs);
    let api = Api {
        name,
        version,
        error_string,
        create,
        compile,
        ptx_size,
        ptx,
        log_size,
        log,
        destroy,
        num_archs,
        get_archs,
        _lib: lib,
    };
    Ok(api)
}

/// Every library name/path to try, most specific toolkit first.
fn candidates() -> Vec<String> {
    // 1. Explicit overrides (tinygrad's `NVRTC_PATH` + our own variable). An
    //    override is authoritative: if the named library does not load we do
    //    not silently pick a different toolkit.
    let mut out = crate::runtime::libfind::env_overrides(&["AUTORUST_NVRTC_LIB", "NVRTC_PATH"]);
    if !out.is_empty() {
        return out;
    }
    // 2. Libraries found on disk, newest version first.
    out.extend(crate::runtime::libfind::scanned_libraries("nvrtc"));
    // 3. Bare sonames, resolved by the system loader (covers ldconfig-only
    //    installs and layouts the scan does not know about).
    out.push(crate::runtime::libfind::unversioned_name("nvrtc"));
    for major in (9..=20).rev() {
        out.push(crate::runtime::libfind::versioned_name("nvrtc", major));
    }
    out.dedup();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sm_ids_are_parsed() {
        assert_eq!(sm_of("sm_86"), Some(86));
        assert_eq!(sm_of("sm_90a"), Some(90));
        assert_eq!(sm_of("sm_120"), Some(120));
        assert_eq!(sm_of("compute_86"), None);
    }

    #[test]
    fn compile_reports_missing_library_cleanly() {
        // No CUDA needed: when NVRTC is absent this must be an `Err`, never a
        // panic (the CUDA device probe relies on that).
        if available() {
            return;
        }
        let err = compile("__global__ void k() {}", "sm_86", &[]).unwrap_err();
        assert!(err.contains("libnvrtc"), "{err}");
    }

    #[test]
    fn unsupported_arch_is_a_clean_error() {
        if !available() || supported_sms().is_empty() {
            return;
        }
        // A CUDA 12.4 NVRTC reports the same kind of error for sm_120; the
        // probe relies on this being an `Err`, not a panic.
        let err = compile("__global__ void k() {}", "sm_999", &[]).unwrap_err();
        assert!(err.contains("does not support sm_999"), "{err}");
    }

    #[test]
    fn compiles_a_trivial_kernel_when_present() {
        if !available() {
            return;
        }
        // Use whatever SM this NVRTC supports so the test also runs on a CUDA
        // 13 machine with a newer arch set.
        let sms = supported_sms();
        let Some(sm) = sms.last().copied() else {
            return;
        };
        let arch = format!("sm_{sm}");
        let ptx = compile(
            "extern \"C\" __global__ void probe(float *out) { out[0] = 1.0f; }",
            &arch,
            &[],
        )
        .expect("trivial kernel must compile");
        let text = String::from_utf8_lossy(&ptx);
        assert!(
            text.contains(".version"),
            "not PTX: {}",
            &text[..64.min(text.len())]
        );
        assert!(text.contains(".entry probe") || text.contains(".visible .entry probe"));
    }

    #[test]
    fn validated_compile_reports_every_rejected_candidate() {
        if !available() || supported_sms().is_empty() {
            return;
        }
        let Some(sm) = supported_sms().last().copied() else {
            return;
        };
        let arch = format!("sm_{sm}");
        let err = compile_validated(
            "extern \"C\" __global__ void probe() {}",
            &arch,
            &[],
            |_| Err::<(), String>("simulated driver rejection".to_string()),
        )
        .unwrap_err();
        assert!(err.contains("driver-compatible PTX"), "{err}");
        assert!(err.contains("simulated driver rejection"), "{err}");
    }
}
