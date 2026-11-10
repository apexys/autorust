//! Benchmark-driven default-device selection.
//!
//! tinygrad picks `Device.DEFAULT` from a fixed priority (CUDA → METAL → CL →
//! RUST → CC → RAYON → INTERP). That is a good prior, but it assumes the GPU
//! is faster — which is false on older Apple hardware, where an Intel Iris
//! OpenCL/Metal path can lose to the multithreaded CPU backend.
//!
//! [`select_fastest_device`] measures a small representative workload on every
//! available backend **and physical GPU** (`CL:1`, `CUDA:1`, ...) and makes the
//! winner the process default (see [`set_default_device`]). It is a one-shot,
//! opt-in call:
//!
//! ```no_run
//! autorust::select_fastest_device();
//! // ... tensor creation with `None` now uses the measured-fastest backend.
//! ```
//!
//! The suite mirrors a tiny training/inference step rather than a synthetic
//! BLAS benchmark — three cases, about 7 MiB of buffers:
//!
//! 1. **linear** — `relu(A @ W + bias)`, a 512×512 f32 GEMM with the fused
//!    epilogue every model layer has.
//! 2. **attention** — causal self-attention, 4 heads × 256 tokens × 64 dims,
//!    the shape that routes through the fused FA2 kernels where available.
//! 3. **softmax** — a 2048×512 row-wise softmax, which is reduction- and
//!    memory-bandwidth-bound (the part GEMM benchmarks miss).
//!
//! Each case is timed end-to-end (graph build + realize + readback) after two
//! untimed warmups — so compilation is excluded — and scored as the mean of
//! the two fastest of three runs. Every result is validated against a
//! host/known answer before the device is eligible — a broken or miscompiling
//! driver is skipped instead of selected. `INTERP`/`CPU` are the scalar
//! reference interpreters and are never candidates; if nothing benchmarks
//! successfully the existing priority pick is kept.
//!
//! `DEFAULT_DEVICE=auto` runs the same selection lazily on the first tensor
//! creation. `AUTORUST_AUTOTUNE_DEVICES=METAL,RAYON` restricts the candidates;
//! a bare `CL` there means every CL device, while `CL:1` means just that one.

use std::time::Instant;

use crate::dtype;
use crate::engine::session::Session;
use crate::tensor::{Ctx, Tensor};

pub use crate::tensor::creation::set_default_device;

/// Linear-layer side length. Big enough that GPU parallelism beats launch
/// overhead (a 256³ GEMM is still launch-bound on Apple GPUs) but small enough
/// that the slowest supported backend finishes in tens of milliseconds.
const MATMUL_N: usize = 512;
/// Attention heads / sequence length / head dim for the attention case.
const ATTN_H: usize = 4;
const ATTN_N: usize = 256;
const ATTN_D: usize = 64;
/// Row-softmax shape (1M elements, the reduction-heavy case).
const SOFTMAX_ROWS: usize = 2048;
const SOFTMAX_COLS: usize = 512;
/// Timed repetitions per case.
const ITERS: usize = 3;
/// Untimed warmups per case (compile + ramp-up, not measured).
const WARMUP: usize = 2;

/// One representative case's measured wall time.
#[derive(Clone, Debug)]
pub struct Case {
    /// `linear`, `attention`, or `softmax`.
    pub name: &'static str,
    /// Mean of the fastest two of three end-to-end runs, in milliseconds.
    pub millis: f64,
}

/// One backend's score from [`rank_devices`].
#[derive(Clone, Debug)]
pub struct DeviceBenchmark {
    /// Canonical device name (`METAL`, `RAYON`, ...).
    pub device: String,
    /// Total representative wall time (the sum of [`DeviceBenchmark::cases`]),
    /// in milliseconds. Lower is faster.
    pub millis: f64,
    /// Per-case breakdown, in suite order.
    pub cases: Vec<Case>,
}

/// Whether this build can construct `device` and it is usable right now.
///
/// Indexed GPU names (`CUDA:1`, `CL:1`) are checked against the driver's
/// device count. The interpreter is intentionally not a candidate: it is the
/// scalar reference path, not something to select for speed.
pub fn is_available(device: &str) -> bool {
    let name = crate::runtime::device::canonicalize_device_name(device);
    let (base, index) = match name.split_once(':') {
        Some((base, idx)) => (base, idx.parse::<usize>().ok()),
        None => (name.as_str(), Some(0)),
    };
    match base {
        #[cfg(feature = "backend_cuda")]
        "CUDA" => {
            index.is_some_and(|i| i < crate::runtime::ops_cuda::device_count())
                && crate::runtime::ops_cuda::available()
        }
        #[cfg(all(feature = "backend_metal", target_os = "macos"))]
        "METAL" => index == Some(0) && crate::runtime::ops_metal::available(),
        #[cfg(feature = "backend_cl")]
        "CL" => index.is_some_and(crate::runtime::ops_cl::device_usable),
        #[cfg(feature = "backend_rust")]
        "RUST" => index == Some(0) && crate::runtime::ops_rust::available(),
        #[cfg(feature = "backend_clang")]
        "CC" => index == Some(0) && crate::runtime::ops_clang::available(),
        #[cfg(feature = "backend_rayon")]
        "RAYON" => index == Some(0),
        _ => false,
    }
}

/// Canonical names of every physical device this build can use, in priority
/// order: `CUDA`, `CUDA:1`, ..., `METAL`, `CL`, `CL:1`, ..., `RUST`, `CC`,
/// `RAYON`. `INTERP`/`CPU` are omitted.
pub fn available_devices() -> Vec<String> {
    let mut out = Vec::new();
    #[cfg(feature = "backend_cuda")]
    if crate::runtime::ops_cuda::available() {
        out.extend(indexed("CUDA", crate::runtime::ops_cuda::device_count()));
    }
    #[cfg(all(feature = "backend_metal", target_os = "macos"))]
    if crate::runtime::ops_metal::available() {
        out.push("METAL".to_string());
    }
    #[cfg(feature = "backend_cl")]
    if crate::runtime::ops_cl::available() {
        out.extend(
            (0..crate::runtime::ops_cl::device_count())
                .filter(|i| crate::runtime::ops_cl::device_usable(*i))
                .map(|i| indexed_name("CL", i)),
        );
    }
    #[cfg(feature = "backend_rust")]
    if crate::runtime::ops_rust::available() {
        out.push("RUST".to_string());
    }
    #[cfg(feature = "backend_clang")]
    if crate::runtime::ops_clang::available() {
        out.push("CC".to_string());
    }
    #[cfg(feature = "backend_rayon")]
    out.push("RAYON".to_string());
    out
}

/// `BASE`, `BASE:1`, ..., `BASE:(count-1)` (index 0 is tinygrad's bare name).
#[cfg(feature = "backend_cuda")]
fn indexed(base: &str, count: usize) -> Vec<String> {
    (0..count).map(|i| indexed_name(base, i)).collect()
}

/// `BASE` for index 0 (tinygrad canonicalizes `:0` away), else `BASE:N`.
fn indexed_name(base: &str, index: usize) -> String {
    if index == 0 {
        base.to_string()
    } else {
        format!("{base}:{index}")
    }
}

/// Time one backend on the representative suite, in milliseconds.
///
/// Returns `None` when the device is unavailable, panics, or produces a wrong
/// result (a broken driver must not win the selection); a one-line
/// `AUTORUST_AUTOTUNE_QUIET`-suppressed note is printed when a benchmark
/// fails.
pub fn benchmark_device(device: &str) -> Option<f64> {
    benchmark_cases(device).map(|cases| cases.iter().map(|c| c.millis).sum())
}

/// Benchmark every available backend and return the results, fastest first.
///
/// Silent; [`select_fastest_device`] is the printing variant. Candidates can be
/// restricted with `AUTORUST_AUTOTUNE_DEVICES=METAL,RAYON`.
pub fn rank_devices() -> Vec<DeviceBenchmark> {
    let wanted = wanted_devices();
    let mut ranked: Vec<DeviceBenchmark> = available_devices()
        .into_iter()
        // `AUTORUST_AUTOTUNE_DEVICES=CL` means every CL device; `CL:1` means
        // just that one.
        .filter(|d| {
            wanted.as_ref().is_none_or(|w| {
                w.iter()
                    .any(|x| x == d || d.split_once(':').is_some_and(|(base, _)| x == base))
            })
        })
        .filter_map(|device| {
            let cases = benchmark_cases(&device)?;
            let millis = cases.iter().map(|c| c.millis).sum();
            Some(DeviceBenchmark {
                device,
                millis,
                cases,
            })
        })
        .collect();
    ranked.sort_by(|a, b| a.millis.total_cmp(&b.millis));
    ranked
}

/// Benchmark the available backends, print the table, and set the process-wide
/// default device (used by tensor creation when no device is given) to the
/// fastest one. Returns the chosen name.
///
/// If no backend can complete the suite, the existing priority pick is kept
/// (`CUDA → METAL → CL → RUST → CC → RAYON → INTERP`). `AUTORUST_AUTOTUNE_QUIET=1`
/// suppresses the summary line.
pub fn select_fastest_device() -> String {
    let ranked = rank_devices();
    let chosen = match ranked.first() {
        Some(best) => best.device.clone(),
        None => crate::tensor::creation::pick_available_device(),
    };
    if crate::helpers::env::get("AUTORUST_AUTOTUNE_QUIET").is_none() && !ranked.is_empty() {
        let table: Vec<String> = ranked.iter().map(fmt_entry).collect();
        eprintln!("autotune: {} -> {chosen}", table.join(", "));
    }
    crate::tensor::creation::set_default_device(&chosen);
    chosen
}

fn fmt_entry(b: &DeviceBenchmark) -> String {
    let cases: Vec<String> = b
        .cases
        .iter()
        .map(|c| format!("{} {:.2}", c.name, c.millis))
        .collect();
    format!("{} {:.2} ms ({})", b.device, b.millis, cases.join(", "))
}

/// `benchmark_device` with the per-case breakdown.
fn benchmark_cases(device: &str) -> Option<Vec<Case>> {
    let name = crate::runtime::device::canonicalize_device_name(device);
    if !is_available(&name) {
        return None;
    }
    // A driver failure surfaces as a Rust panic (`TGERR: ...`) in this port;
    // treat it like an unavailable device rather than aborting the selection.
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_suite(&name))) {
        Ok(cases) => cases,
        Err(_) => {
            if crate::helpers::env::get("AUTORUST_AUTOTUNE_QUIET").is_none() {
                eprintln!("autotune: {name} failed; skipping");
            }
            None
        }
    }
}

/// `AUTORUST_AUTOTUNE_DEVICES=...` as canonical device names.
fn wanted_devices() -> Option<Vec<String>> {
    let raw = crate::helpers::env::get("AUTORUST_AUTOTUNE_DEVICES")?;
    let list: Vec<String> = raw
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(crate::runtime::device::canonicalize_device_name)
        .collect();
    (!list.is_empty()).then_some(list)
}

/// Warm up, run all three cases, validate, and return their timings.
fn run_suite(dev: &str) -> Option<Vec<Case>> {
    let ctx = Ctx::new();
    let mut sess = Session::with_device(Some(dev.to_string()));

    // ---- linear: relu(A @ W + bias) -------------------------------------
    let n = MATMUL_N;
    let a_data = matrix(n, 0);
    let w_data = matrix(n, 17);
    let bias_data: Vec<f32> = (0..n).map(|i| ((i % 23) as f32) * 0.05 - 0.6).collect();
    let shape = [n as i64, n as i64];
    let a = Tensor::from_slice(&ctx, &a_data, Some(dtype::FLOAT32), Some(dev))
        .reshape(shape)
        .to_param();
    let w = Tensor::from_slice(&ctx, &w_data, Some(dtype::FLOAT32), Some(dev))
        .reshape(shape)
        .to_param();
    let bias = Tensor::from_slice(&ctx, &bias_data, Some(dtype::FLOAT32), Some(dev)).to_param();

    // Materialize the parameters once, outside the timed region: otherwise the
    // host→device copy is replayed every iteration and the score measures it
    // instead of the compute.
    for t in [&a, &w, &bias] {
        let _ = t.to_list::<f32>(&mut sess);
    }
    let (linear_ms, linear_out) = timed(&mut sess, |_| a.dot(&w).add(&bias).relu());
    // Sample a spread of outputs against a host dot product + bias + clamp.
    for (i, j) in [(0, 0), (3, 17), (31, 127), (64, 64), (200, 400), (511, 511)] {
        let want: f32 = (0..n)
            .map(|k| a_data[i * n + k] * w_data[k * n + j])
            .sum::<f32>()
            + bias_data[j];
        let want = want.max(0.0);
        let got = *linear_out.get(i * n + j)?;
        if !got.is_finite() || (got - want).abs() > 2e-2 * (1.0 + want.abs()) {
            return None;
        }
    }

    // ---- attention: causal self-attention --------------------------------
    let (h, seq, d) = (ATTN_H, ATTN_N, ATTN_D);
    let elems = h * seq * d;
    let q_data = tensor_data(elems, 3);
    let k_data = tensor_data(elems, 11);
    let v_data = tensor_data(elems, 19);
    let attn_shape = [1i64, h as i64, seq as i64, d as i64];
    let q = Tensor::from_slice(&ctx, &q_data, Some(dtype::FLOAT32), Some(dev))
        .reshape(attn_shape)
        .to_param();
    let k = Tensor::from_slice(&ctx, &k_data, Some(dtype::FLOAT32), Some(dev))
        .reshape(attn_shape)
        .to_param();
    let v = Tensor::from_slice(&ctx, &v_data, Some(dtype::FLOAT32), Some(dev))
        .reshape(attn_shape)
        .to_param();
    for t in [&q, &k, &v] {
        let _ = t.to_list::<f32>(&mut sess);
    }
    let (attn_ms, attn_out) = timed(&mut sess, |s| q.attention_with(&k, &v, None, 0.0, true, s));
    for (hh, row) in [(0, 0), (1, 17), (h - 1, seq - 1)] {
        let want = attention_row(&q_data, &k_data, &v_data, seq, d, hh, row);
        for t in 0..d {
            let got = *attn_out.get((hh * seq + row) * d + t)?;
            let want = want[t];
            if !got.is_finite() || (got - want).abs() > 5e-3 * (1.0 + want.abs()) {
                return None;
            }
        }
    }

    // ---- softmax: row-wise activation over 1M elements -------------------
    let rows = SOFTMAX_ROWS;
    let cols = SOFTMAX_COLS;
    let x_data: Vec<f32> = (0..rows * cols)
        .map(|i| (((i * 7) % 61) as f32) * 0.13 - 4.0)
        .collect();
    let x = Tensor::from_slice(&ctx, &x_data, Some(dtype::FLOAT32), Some(dev))
        .reshape([rows as i64, cols as i64])
        .to_param();
    let _ = x.to_list::<f32>(&mut sess);
    let (softmax_ms, softmax_out) = timed(&mut sess, |_| crate::tensor::nn::softmax(&x, Some(1)));
    for row in [0, rows / 2, rows - 1] {
        let want = softmax_row(&x_data, cols, row);
        for c in 0..cols {
            let got = *softmax_out.get(row * cols + c)?;
            let want = want[c];
            if !got.is_finite() || (got - want).abs() > 2e-4 {
                return None;
            }
        }
    }

    Some(vec![
        Case {
            name: "linear",
            millis: linear_ms,
        },
        Case {
            name: "attention",
            millis: attn_ms,
        },
        Case {
            name: "softmax",
            millis: softmax_ms,
        },
    ])
}

/// Time `build()` + read-back after [`WARMUP`] untimed runs, returning the mean
/// of the fastest two of [`ITERS`]. The builder gets the session so
/// `attention_with` can route through the fused attention kernels.
fn timed(sess: &mut Session, mut build: impl FnMut(&mut Session) -> Tensor) -> (f64, Vec<f32>) {
    // Warmups absorb compilation and allocator/clock ramp-up, so the samples
    // below measure execution only.
    for _ in 0..WARMUP {
        let _ = build(sess).to_list::<f32>(sess);
    }
    let mut samples = Vec::with_capacity(ITERS);
    let mut last = Vec::new();
    for _ in 0..ITERS {
        let start = Instant::now();
        let out = build(sess);
        last = out.to_list::<f32>(sess);
        samples.push(start.elapsed().as_secs_f64() * 1e3);
    }
    let trace = crate::helpers::env::get("AUTORUST_AUTOTUNE_TRACE").is_some();
    if trace {
        eprintln!("autotune: samples {samples:?} ms");
    }
    // Mean of the fastest two: a single slow sample (scheduling, a cold cache)
    // must not decide the device, and the mean is less noisy than the minimum.
    samples.sort_by(f64::total_cmp);
    let n = samples.len().clamp(1, 2);
    (samples[..n].iter().sum::<f64>() / n as f64, last)
}

/// Deterministic `n × n` f32 operand.
fn matrix(n: usize, seed: usize) -> Vec<f32> {
    (0..n * n)
        .map(|i| (((i + seed) % 131) as f32) * 0.0078125 - 0.5)
        .collect()
}

/// Deterministic 1-D f32 operand.
fn tensor_data(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i + seed) % 131) as f32) * 0.0078125 - 0.5)
        .collect()
}

/// Host reference for one causal attention row (`softmax(q·kᵀ/√d) · v`).
fn attention_row(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n: usize,
    d: usize,
    head: usize,
    row: usize,
) -> Vec<f32> {
    let base = head * n * d;
    let scale = 1.0 / (d as f32).sqrt();
    let jmax = row + 1;
    let mut s = Vec::with_capacity(jmax);
    let mut mx = f32::NEG_INFINITY;
    for j in 0..jmax {
        let dot: f32 = (0..d)
            .map(|t| q[base + row * d + t] * k[base + j * d + t])
            .sum::<f32>()
            * scale;
        s.push(dot);
        mx = mx.max(dot);
    }
    let p: Vec<f32> = s.iter().map(|x| (x - mx).exp()).collect();
    let sum: f32 = p.iter().sum();
    (0..d)
        .map(|t| (0..jmax).map(|j| p[j] * v[base + j * d + t]).sum::<f32>() / sum)
        .collect()
}

/// Host reference for one row of a row-wise softmax.
fn softmax_row(x: &[f32], cols: usize, row: usize) -> Vec<f32> {
    let src = &x[row * cols..(row + 1) * cols];
    let mx = src.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = src.iter().map(|v| (v - mx).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}
