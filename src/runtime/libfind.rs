//! Shared discovery for the dynamically loaded CUDA libraries.
//!
//! Both NVRTC and cuBLAS are separate installs from the driver and from each
//! other, and their soname encodes the toolkit major version. Loading them by
//! the version the binary was built against breaks when the runtime toolkit
//! differs, so [`crate::runtime::nvrtc`] and [`crate::runtime::cublas`] use
//! these helpers instead: honour an explicit override, otherwise scan the usual
//! library directories for the newest matching library and fall back to bare
//! sonames.
//!
//! This mirrors tinygrad's `runtime/support/c.py::DLL.findlib`, which walks the
//! loader path and accepts any `lib<name>.so[.N]` ELF file.

use std::path::{Path, PathBuf};

/// Explicit library path/name overrides, in priority order.
///
/// An override is authoritative: callers return it without falling back to a
/// scanned library, so a user asking for a specific toolkit gets exactly that.
pub(crate) fn env_overrides(keys: &[&str]) -> Vec<String> {
    let mut out = Vec::new();
    for key in keys {
        if let Some(path) = crate::helpers::env::get(key).filter(|p| !p.is_empty()) {
            out.push(path);
        }
    }
    out
}

/// Directories searched for `lib<stem>.so*` / `<stem>64_*.dll`.
pub(crate) fn search_dirs() -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    let push = |dirs: &mut Vec<PathBuf>, p: PathBuf| {
        if !dirs.contains(&p) {
            dirs.push(p);
        }
    };
    for var in ["LD_LIBRARY_PATH", "PATH"] {
        if let Some(value) = std::env::var_os(var) {
            for dir in std::env::split_paths(&value) {
                push(&mut dirs, dir);
            }
        }
    }
    for key in ["CUDA_HOME", "CUDA_PATH", "CUDA_ROOT", "CONDA_PREFIX"] {
        if let Some(root) = crate::helpers::env::get(key).filter(|p| !p.is_empty()) {
            let root = PathBuf::from(root);
            for sub in [
                "lib64",
                "lib",
                "targets/x86_64-linux/lib",
                "targets/aarch64-linux/lib",
            ] {
                push(&mut dirs, root.join(sub));
            }
            // Windows toolkits put the DLLs in `bin`, not `lib`.
            if cfg!(windows) {
                push(&mut dirs, root.join("bin"));
            }
        }
    }
    // CUDA toolkits installed under the usual prefixes (`/usr/local/cuda-13.1`).
    for root in ["/usr/local", "/opt"] {
        let Ok(entries) = std::fs::read_dir(root) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            if !name.to_ascii_lowercase().starts_with("cuda") {
                continue;
            }
            let base = entry.path();
            for sub in [
                "lib64",
                "lib",
                "targets/x86_64-linux/lib",
                "targets/aarch64-linux/lib",
            ] {
                push(&mut dirs, base.join(sub));
            }
        }
    }
    let multiarch = format!("/usr/lib/{}-linux-gnu", std::env::consts::ARCH);
    for fixed in [
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/targets/x86_64-linux/lib",
        "/usr/local/cuda/targets/aarch64-linux/lib",
        "/usr/local/lib",
        "/usr/lib64",
        "/usr/lib",
        "/lib64",
        "/lib",
        "/opt/cuda/lib64",
        "/usr/lib/wsl/lib",
    ] {
        push(&mut dirs, PathBuf::from(fixed));
    }
    push(&mut dirs, PathBuf::from(multiarch));
    dirs
}

/// Every scanned `lib<stem>.so*` / `<stem>64_*.dll`, newest version first.
pub(crate) fn scanned_libraries(stem: &str) -> Vec<String> {
    let mut found: Vec<(Vec<u32>, String)> = Vec::new();
    for dir in search_dirs() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            let Some(key) = version_key(name, stem) else {
                continue;
            };
            if !is_real_library(&path) {
                continue;
            }
            found.push((key, path.to_string_lossy().into_owned()));
        }
    }
    found.sort_by(|a, b| b.0.cmp(&a.0));
    found.into_iter().map(|(_, path)| path).collect()
}

/// Version groups encoded in a candidate filename, or `None` when it is not a
/// `<stem>` library. `lib<stem>.so` -> `[]`, `lib<stem>.so.12.4` -> `[12, 4]`,
/// `<stem>64_130_0.dll` -> `[130, 0]`.
pub(crate) fn version_key(name: &str, stem: &str) -> Option<Vec<u32>> {
    #[cfg(windows)]
    {
        let lower = name.to_ascii_lowercase();
        if lower == format!("{stem}.dll") {
            return Some(Vec::new());
        }
        let rest = lower
            .strip_prefix(&format!("{stem}64_"))?
            .strip_suffix(".dll")?;
        rest.split('_').map(|part| part.parse().ok()).collect()
    }
    #[cfg(not(windows))]
    {
        let rest = name.strip_prefix(&format!("lib{stem}.so"))?;
        if rest.is_empty() {
            return Some(Vec::new());
        }
        let rest = rest.strip_prefix('.')?;
        // Reject `lib<stem>-builtins.so...` (the prefix check already does) and
        // any non-numeric suffix.
        rest.split('.').map(|part| part.parse().ok()).collect()
    }
}

/// A real ELF/PE file, not a linker script (`lib<name>.so` is sometimes a text
/// file containing `INPUT(...)`).
pub(crate) fn is_real_library(path: &Path) -> bool {
    use std::io::Read;
    let Ok(mut f) = std::fs::File::open(path) else {
        return false;
    };
    let mut magic = [0u8; 4];
    if f.read_exact(&mut magic).is_err() {
        return false;
    }
    #[cfg(target_os = "linux")]
    {
        magic == *b"\x7fELF"
    }
    #[cfg(windows)]
    {
        magic[0..2] == *b"MZ"
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    {
        let _ = magic;
        true
    }
}

/// Unversioned name (`lib<stem>.so` or `<stem>.dll`).
pub(crate) fn unversioned_name(stem: &str) -> String {
    if cfg!(windows) {
        format!("{stem}.dll")
    } else {
        format!("lib{stem}.so")
    }
}

/// Bare soname for `major` (`lib<stem>.so.13` or `<stem>64_130_0.dll`).
pub(crate) fn versioned_name(stem: &str, major: u32) -> String {
    if cfg!(windows) {
        format!("{stem}64_{major}0_0.dll")
    } else {
        format!("lib{stem}.so.{major}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvrtc_names_parse_and_sort() {
        #[cfg(not(windows))]
        {
            assert_eq!(version_key("libnvrtc.so", "nvrtc"), Some(vec![]));
            assert_eq!(version_key("libnvrtc.so.12", "nvrtc"), Some(vec![12]));
            assert_eq!(version_key("libnvrtc.so.12.4", "nvrtc"), Some(vec![12, 4]));
            assert_eq!(
                version_key("libnvrtc.so.12.4.127", "nvrtc"),
                Some(vec![12, 4, 127])
            );
            assert_eq!(version_key("libnvrtc-builtins.so.12.4", "nvrtc"), None);
            assert_eq!(version_key("libnvrtc_static.a", "nvrtc"), None);
            assert!(vec![13u32] > vec![12u32]);
            assert!(vec![12u32, 4] > vec![12u32, 3]);
        }
        #[cfg(windows)]
        {
            assert_eq!(version_key("nvrtc.dll", "nvrtc"), Some(vec![]));
            assert_eq!(
                version_key("nvrtc64_120_0.dll", "nvrtc"),
                Some(vec![120, 0])
            );
        }
    }

    #[test]
    fn cublas_names_parse_and_sort() {
        #[cfg(not(windows))]
        {
            assert_eq!(version_key("libcublas.so", "cublas"), Some(vec![]));
            assert_eq!(version_key("libcublas.so.12", "cublas"), Some(vec![12]));
            assert_eq!(
                version_key("libcublas.so.13.0", "cublas"),
                Some(vec![13, 0])
            );
            // `libcublasLt.so` and `cublasLt64_*.dll` are a different library.
            assert_eq!(version_key("libcublasLt.so.12", "cublas"), None);
            assert_eq!(version_key("libcublas_static.a", "cublas"), None);
        }
        #[cfg(windows)]
        {
            assert_eq!(version_key("cublas64_12.dll", "cublas"), Some(vec![12]));
            assert_eq!(
                version_key("cublas64_130_0.dll", "cublas"),
                Some(vec![130, 0])
            );
            assert_eq!(version_key("cublasLt64_12.dll", "cublas"), None);
        }
    }
}
