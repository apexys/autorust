//! A tiny std-only on-disk cache — the port of tinygrad's `diskcache_get` /
//! `diskcache_put` (`helpers.py`), used to make a BEAM search run once per
//! `(kernel, width, device)` rather than on every process.
//!
//! tinygrad stores a JSON file per `(scope, key-dict)` under `$CACHE_DIR`,
//! gated by `CACHELEVEL` (default 1; 0 disables). The port stores one file per
//! `(scope, FNV-1a(key))` under `AUTORUST_CACHE_DIR` (default
//! `$XDG_CACHE_HOME/autorust` or `~/.cache/autorust`), with the full key on the
//! first line so a stale/hash-colliding entry is rejected. The value is opaque
//! text (the caller decides the encoding).

use std::path::PathBuf;

/// `CACHELEVEL` (tinygrad default 1): `>= 1` enables the on-disk cache; `0`
/// disables it. An unparsable value falls back to the default.
pub fn cache_enabled() -> bool {
    std::env::var("CACHELEVEL").ok().and_then(|s| s.parse::<i32>().ok()).unwrap_or(1) >= 1
}

/// The cache root: `AUTORUST_CACHE_DIR`, else `$XDG_CACHE_HOME/autorust`, else
/// `$HOME/.cache/autorust`, else a `/tmp` fallback.
fn cache_root() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("AUTORUST_CACHE_DIR")
        && !d.is_empty()
    {
        return Some(PathBuf::from(d));
    }
    if let Ok(d) = std::env::var("XDG_CACHE_HOME")
        && !d.is_empty()
    {
        return Some(PathBuf::from(d).join("autorust"));
    }
    if let Ok(h) = std::env::var("HOME")
        && !h.is_empty()
    {
        return Some(PathBuf::from(h).join(".cache").join("autorust"));
    }
    Some(PathBuf::from("/tmp/opencode/autorust-cache"))
}

/// FNV-1a 64-bit — a stable, dependency-free file-name hash (the value itself is
/// stored in full, so a collision can never return the wrong entry).
fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn entry_path(scope: &str, key: &str) -> Option<PathBuf> {
    Some(cache_root()?.join(format!("{scope}-{:016x}.json", fnv1a(key))))
}

/// `diskcache_get(scope, key)`: the stored value, or `None` on a miss/disabled.
pub fn get(scope: &str, key: &str) -> Option<String> {
    if !cache_enabled() {
        return None;
    }
    let text = std::fs::read_to_string(entry_path(scope, key)?).ok()?;
    let (stored_key, value) = text.split_once('\n')?;
    if stored_key == key {
        Some(value.to_string())
    } else {
        None
    }
}

/// `diskcache_put(scope, key, value)`. Best-effort: a failed write is ignored.
pub fn put(scope: &str, key: &str, value: &str) {
    if !cache_enabled() {
        return;
    }
    let Some(path) = entry_path(scope, key) else { return };
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(path, format!("{key}\n{value}"));
}
