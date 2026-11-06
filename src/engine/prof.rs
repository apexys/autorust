//! Opt-in phase profiling (`AUTORUST_PROF=1`).
//!
//! The port has no profiler, so when a model is unexpectedly slow the only
//! visible signal is wall time. This module accumulates named timings on the
//! thread and prints a sorted table at process exit (via [`report`], which the
//! caller invokes where it knows the run is over).
//!
//! Zero cost when disabled: every call site checks [`enabled`] (a `OnceLock`
//! read) before taking an `Instant`.
//!
//! Names are hierarchical by convention (`"realize.run"`, `"exec.kernel"`), so
//! the table reads as a breakdown. Timings are inclusive (a phase may contain
//! another).

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::OnceLock;

#[derive(Default, Clone, Copy)]
struct Entry {
    count: u64,
    total: f64,
}

thread_local! {
    static ENTRIES: RefCell<BTreeMap<&'static str, Entry>> = const { RefCell::new(BTreeMap::new()) };
}

/// Whether profiling is on.
///
/// `AUTORUST_PROF` turns it on (any value except `0`/empty). The rewrite
/// diagnostics `TG_RW_TRACE` and `TG_RW_VISITS` also turn it on, because their
/// output is derived from the `rw.*` counters maintained here.
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let prof = std::env::var("AUTORUST_PROF")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(false);
        prof || std::env::var("TG_RW_TRACE").is_ok() || std::env::var("TG_RW_VISITS").is_ok()
    })
}

/// Accumulate `secs` under `name`.
pub fn add(name: &'static str, secs: f64) {
    if !enabled() {
        return;
    }
    ENTRIES.with(|e| {
        let mut m = e.borrow_mut();
        let ent = m.entry(name).or_default();
        ent.count += 1;
        ent.total += secs;
    });
}

/// Increment a counter (no timing).
pub fn count(name: &'static str) {
    if !enabled() {
        return;
    }
    ENTRIES.with(|e| {
        let mut m = e.borrow_mut();
        m.entry(name).or_default().count += 1;
    });
}

/// Clear all accumulators (e.g. after a JIT capture, so the reported table
/// describes steady-state replay only).
pub fn reset() {
    ENTRIES.with(|e| e.borrow_mut().clear());
}

/// Current count for `name` (0 when profiling is off).
pub fn get(name: &'static str) -> u64 {
    ENTRIES.with(|e| e.borrow().get(name).map(|x| x.count).unwrap_or(0))
}

/// Print the accumulated table (no-op when disabled or empty).
pub fn report() {
    if !enabled() {
        return;
    }
    ENTRIES.with(|e| {
        let m = e.borrow();
        if m.is_empty() {
            return;
        }
        println!("--- autorust profile ---");
        println!(
            "{:<24} {:>7} {:>12} {:>12}",
            "phase", "calls", "total (s)", "per call"
        );
        for (name, ent) in m.iter() {
            println!(
                "{:<24} {:>7} {:>12.4} {:>12}",
                name,
                ent.count,
                ent.total,
                format!("{:.3} ms", ent.total / ent.count as f64 * 1e3)
            );
        }
    });
}
