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
    /// `false` for an accumulator (`acc`), whose total is a raw quantity (node
    /// visits) rather than seconds.
    is_time: bool,
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
        let prof = crate::helpers::env::get("AUTORUST_PROF")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(false);
        prof || crate::helpers::env::get("TG_RW_TRACE").is_some()
            || crate::helpers::env::get("TG_RW_VISITS").is_some()
    })
}

/// Per-kernel profiling (`AUTORUST_KPROF=1`): the realize loop prints one
/// `KPROF <name> <secs>` line per launch for host-executed backends. CUDA has
/// its own per-launch profiler in `runtime::ops_cuda` (it syncs the stream).
pub fn kprof() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::helpers::env::get("AUTORUST_KPROF").is_some())
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
        ent.is_time = true;
    });
}

/// Accumulate an integer-valued quantity under `name` (e.g. nodes visited or
/// AST size). The total is printed raw, not as seconds.
pub fn acc(name: &'static str, value: f64) {
    if !enabled() {
        return;
    }
    ENTRIES.with(|e| {
        let mut m = e.borrow_mut();
        let ent = m.entry(name).or_default();
        ent.count += 1;
        ent.total += value;
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
/// describes steady-state replay only). This also clears the opt-in HostGemm
/// dispatch accounting, so it describes the same region.
pub fn reset() {
    ENTRIES.with(|e| e.borrow_mut().clear());
    crate::schedule::host_gemm::reset_stats();
}

/// Current count for `name` (0 when profiling is off).
pub fn get(name: &'static str) -> u64 {
    ENTRIES.with(|e| e.borrow().get(name).map(|x| x.count).unwrap_or(0))
}

/// Print the accumulated table (no-op when disabled or empty), then the opt-in
/// HostGemm dispatch accounting (`AUTORUST_GEMM_STATS`), which is useful on its
/// own and does not require `AUTORUST_PROF`.
///
/// Timings are inclusive: a phase may contain another (e.g. `realize.schedule`
/// contains `rng.*`), so do not sum the table. Counters (`calls` only) have no
/// time and print `-` under the timing columns.
pub fn report() {
    if enabled() {
        ENTRIES.with(|e| {
            let m = e.borrow();
            if m.is_empty() {
                return;
            }
            println!("--- autorust profile (inclusive; phases nest) ---");
            println!(
                "{:<24} {:>7} {:>12} {:>12}",
                "phase", "calls", "total", "per call"
            );
            for (name, ent) in m.iter() {
                let (total, per_call) = if ent.is_time {
                    (
                        format!("{:.4} s", ent.total),
                        format!("{:.3} ms", ent.total / ent.count as f64 * 1e3),
                    )
                } else {
                    // A counter/accumulator (`count`/`acc`), not a duration.
                    (format!("{:.0}", ent.total), "-".to_string())
                };
                println!(
                    "{:<24} {:>7} {:>12} {:>12}",
                    name, ent.count, total, per_call
                );
            }
        });
    }
    crate::schedule::host_gemm::report_stats();
}
