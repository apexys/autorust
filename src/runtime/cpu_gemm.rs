#![allow(unsafe_op_in_unsafe_fn)]

//! Blocked CPU GEMM for the `HostGemm` seam.
//!
//! The generated Rust kernel tiles one output row at a time, so every row
//! re-streams the whole `B` matrix and the tile collapses to 1x8 whenever `m`
//! is not divisible by the register width. On a 3900X that is ~14 GFLOP/s at
//! 789x384x384 and it does not scale (1.6x on 12 cores).
//!
//! This module implements the same descriptor as a packed blocked GEMM:
//! `rayon` over disjoint `(m, n)` tiles, both operands packed into panels that
//! match the microkernel, and a 6x16 AVX2/FMA microkernel (12 YMM accumulators
//! + 2 operand vectors, no spills). The fringe rows/columns go through a scalar
//! microkernel, so prime extents need no padding and no masks.
//!
//! The epilogue (`alpha`, bias, per-column scale, residual) is applied in a
//! final in-tile pass, matching the CUDA/CL order exactly:
//! `v = alpha*c; v += bias; v *= scale; v += residual`.

use std::cell::RefCell;

use rayon::prelude::*;

use crate::dtype;
use crate::schedule::host_gemm::HostGemm;

/// Microkernel rows.
const MR: usize = 6;
/// Microkernel columns (two 8-wide vectors).
const NR: usize = 16;

/// Number of host GEMMs dispatched (observability / tests).
pub static DISPATCHES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

thread_local! {
    /// Per-thread pack buffer: `MC*KC` floats of `A`, then `KC*NC` of `B`.
    static SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

fn env_usize(name: &str, default: usize) -> usize {
    crate::helpers::env::get(name)
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Whether the CPU host GEMM should be used.
///
/// `AUTORUST_GEMM=0` (or the older `AUTORUST_CPU_GEMM=0`) disables it.
pub fn available() -> bool {
    true
}

/// Install the CPU GEMM as a device's host-op dispatcher.
///
/// Shared by the compiled-Rust (`RUST`), clang (`CC`) and `RAYON` backends:
/// all keep their operands in host memory, so the same packed kernel serves
/// them under the same `HostGemm` descriptor.
pub fn install(dev: &std::rc::Rc<crate::runtime::device::Compiled>) {
    use crate::runtime::buffer::host_of;
    *dev.host_exec.borrow_mut() = Some(Box::new(|bufs, desc| {
        use crate::schedule::host_gemm::HostGemm;
        let g = HostGemm::from_arg(&crate::uop::arg::Arg::Data(desc.clone()))
            .unwrap_or_else(|| panic!("TGERR: bad HostGemm descriptor"));
        let mut ptrs = Vec::with_capacity(bufs.len());
        for b in bufs {
            let h = host_of(&**b);
            let mem = h.mem.borrow();
            // SAFETY: the buffer is alive for the duration of the call and the
            // descriptor's offsets/strides stay inside it.
            ptrs.push(unsafe { mem.as_ptr().add(h.offset) as *mut u8 });
        }
        DISPATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dbg = crate::helpers::env::get("AUTORUST_CPU_GEMM_DEBUG").is_some();
        let t0 = dbg.then(std::time::Instant::now);
        if let Err(e) = unsafe { run_raw(&g, &ptrs) } {
            panic!("TGERR: cpu gemm: {e}");
        }
        if let Some(t) = t0 {
            let ms = t.elapsed().as_secs_f64() * 1e3;
            if ms > 2.0 {
                eprintln!(
                    "CPUGemm m={} n={} k={} batch={} a=[{},{}] b=[{},{}] c=[{},{}] {ms:.2} ms",
                    g.m, g.n, g.k, g.batch, g.a_m, g.a_k, g.b_k, g.b_n, g.c_m, g.c_n
                );
            }
        }
    }));
}

/// Whether the shape is worth dispatching: below this the generated kernel is
/// cheaper than packing + a rayon fan-out.
pub fn worth_dispatching(m: i64, n: i64, k: i64) -> bool {
    m > 0 && n > 0 && k > 0 && m.saturating_mul(n).saturating_mul(k) >= 1_000_000
}

struct Tile {
    mc: usize,
    nc: usize,
    kc: usize,
}

fn tile() -> Tile {
    match crate::settings::global().cpu_gemm_tile {
        Some((mc, nc, kc)) => Tile {
            mc: mc.max(MR),
            nc: nc.max(NR),
            kc: kc.max(1),
        },
        None => Tile {
            mc: 48.max(MR),
            nc: 96.max(NR),
            kc: 256.max(1),
        },
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn micro_6x16(
    kc: usize,
    ap: *const f32,
    bp: *const f32,
    c: *mut f32,
    ldc: usize,
    first: bool,
) {
    use std::arch::x86_64::*;
    let mut acc = [_mm256_setzero_ps(); 12];
    for k in 0..kc {
        let b0 = _mm256_loadu_ps(bp.add(k * NR));
        let b1 = _mm256_loadu_ps(bp.add(k * NR + 8));
        for i in 0..MR {
            let a = _mm256_set1_ps(*ap.add(k * MR + i));
            acc[2 * i] = _mm256_fmadd_ps(a, b0, acc[2 * i]);
            acc[2 * i + 1] = _mm256_fmadd_ps(a, b1, acc[2 * i + 1]);
        }
    }
    for i in 0..MR {
        let p = c.add(i * ldc);
        let (mut v0, mut v1) = (acc[2 * i], acc[2 * i + 1]);
        if !first {
            v0 = _mm256_add_ps(v0, _mm256_loadu_ps(p));
            v1 = _mm256_add_ps(v1, _mm256_loadu_ps(p.add(8)));
        }
        _mm256_storeu_ps(p, v0);
        _mm256_storeu_ps(p.add(8), v1);
    }
}

/// Portable microkernel for fringe tiles (`mr <= MR`, `nr <= NR`).
#[inline]
unsafe fn micro_gen(
    mr: usize,
    nr: usize,
    kc: usize,
    ap: *const f32,
    bp: *const f32,
    c: *mut f32,
    ldc: usize,
    first: bool,
) {
    let mut acc = [[0.0f32; NR]; MR];
    for k in 0..kc {
        for j in 0..nr {
            let b = *bp.add(k * NR + j);
            for i in 0..mr {
                acc[i][j] += *ap.add(k * MR + i) * b;
            }
        }
    }
    for i in 0..mr {
        for j in 0..nr {
            let p = c.add(i * ldc + j);
            if first {
                *p = acc[i][j];
            } else {
                *p += acc[i][j];
            }
        }
    }
}

fn have_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *V.get_or_init(|| {
            std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
        })
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Run one `HostGemm` on host buffers.
///
/// `ptrs[slot]` is the base address of the buffer the descriptor's `*_off`
/// element offsets apply to. `f32` only; the caller gates the dtype.
///
/// # Safety
/// Every `ptrs` entry referenced by the descriptor must be a valid allocation
/// large enough for the offsets and strides it describes.
pub unsafe fn run_raw(g: &HostGemm, ptrs: &[*mut u8]) -> Result<(), String> {
    if g.dtype != dtype::FLOAT32 {
        return Err("cpu_gemm: only f32 is supported".into());
    }
    if g.c_n != 1 {
        return Err("cpu_gemm: C must be row-contiguous (c_n == 1)".into());
    }
    if g.m <= 0 || g.n <= 0 || g.k <= 0 {
        return Ok(());
    }
    // Raw pointers wrapped as `usize` so the pack/execute closure is `Sync`.
    let bases: Vec<usize> = ptrs.iter().map(|p| *p as usize).collect();
    let g = g.clone();
    let (m, n, k) = (g.m as usize, g.n as usize, g.k as usize);
    let t = tile();
    let avx = have_avx2();
    let bsz = g.batch as usize;
    for batch in 0..bsz {
        let nb = n.div_ceil(t.nc);
        let tasks = m.div_ceil(t.mc) * nb;
        let need = t.mc * t.kc + t.kc * t.nc;
        (0..tasks).into_par_iter().for_each(|task| {
            let base = |slot: i64, off: i64| -> *mut f32 {
                let p = bases[slot as usize] as *mut u8;
                unsafe { p.offset((off * 4) as isize) as *mut f32 }
            };
            let a = unsafe { base(g.a_slot, g.a_off).offset(batch as isize * g.a_b as isize) };
            let b = unsafe { base(g.b_slot, g.b_off).offset(batch as isize * g.b_b as isize) };
            let c = unsafe { base(g.c_slot, g.c_off).offset(batch as isize * g.c_b as isize) };
            let batch_rel = batch as isize * g.c_b as isize;
            let (a_m, a_k) = (g.a_m as isize, g.a_k as isize);
            let (b_k, b_n) = (g.b_k as isize, g.b_n as isize);
            let c_m = g.c_m as isize;
            SCRATCH.with(|slot| {
                let mut buf = slot.borrow_mut();
                buf.resize(need, 0.0);
                let ap_all = buf.as_mut_ptr();
                let bp_all = unsafe { ap_all.add(t.mc * t.kc) };
                let mb = task / nb;
                let nblk = task % nb;
                let m0 = mb * t.mc;
                let n0 = nblk * t.nc;
                let m_cnt = t.mc.min(m - m0);
                let n_cnt = t.nc.min(n - n0);
                let mpan = m_cnt.div_ceil(MR);
                let npan = n_cnt.div_ceil(NR);
                let mut k0 = 0;
                let mut first = true;
                while k0 < k {
                    let kc = t.kc.min(k - k0);
                    unsafe {
                        // Pack A panels: ap[(ip*kc + kk)*MR + ii], zero-padded.
                        // Walk whichever axis is contiguous to keep the reads
                        // sequential (ViT weights are `[n, k]`, A is `[m, k]`).
                        if a_k == 1 {
                            for ip in 0..mpan {
                                let rows = MR.min(m_cnt - ip * MR);
                                for ii in 0..MR {
                                    let dst = ap_all.add(ip * kc * MR + ii);
                                    if ii < rows {
                                        let src = a.offset((m0 + ip * MR + ii) as isize * a_m);
                                        for kk in 0..kc {
                                            *dst.add(kk * MR) = *src.offset((k0 + kk) as isize);
                                        }
                                    } else {
                                        for kk in 0..kc {
                                            *dst.add(kk * MR) = 0.0;
                                        }
                                    }
                                }
                            }
                        } else {
                            for ip in 0..mpan {
                                let rows = MR.min(m_cnt - ip * MR);
                                for ii in 0..MR {
                                    let dst = ap_all.add(ip * kc * MR + ii);
                                    if ii < rows {
                                        let src = a.offset((m0 + ip * MR + ii) as isize * a_m);
                                        for kk in 0..kc {
                                            *dst.add(kk * MR) =
                                                *src.offset((k0 + kk) as isize * a_k);
                                        }
                                    } else {
                                        for kk in 0..kc {
                                            *dst.add(kk * MR) = 0.0;
                                        }
                                    }
                                }
                            }
                        }
                        // Pack B panels: bp[(jp*kc + kk)*NR + jj], zero-padded.
                        if b_k == 1 {
                            // B stored [n, k] (a transposed weight): read each
                            // column panel row-wise and scatter into k-major.
                            for jp in 0..npan {
                                let cols = NR.min(n_cnt - jp * NR);
                                for jj in 0..NR {
                                    let dst = bp_all.add(jp * kc * NR + jj);
                                    if jj < cols {
                                        let src = b.offset((n0 + jp * NR + jj) as isize * b_n);
                                        for kk in 0..kc {
                                            *dst.add(kk * NR) = *src.offset((k0 + kk) as isize);
                                        }
                                    } else {
                                        for kk in 0..kc {
                                            *dst.add(kk * NR) = 0.0;
                                        }
                                    }
                                }
                            }
                        } else {
                            for jp in 0..npan {
                                let cols = NR.min(n_cnt - jp * NR);
                                for kk in 0..kc {
                                    let src = b.offset(
                                        (k0 + kk) as isize * b_k + (n0 + jp * NR) as isize * b_n,
                                    );
                                    let dst = bp_all.add((jp * kc + kk) * NR);
                                    for jj in 0..cols {
                                        *dst.add(jj) = *src.offset(jj as isize * b_n);
                                    }
                                    for jj in cols..NR {
                                        *dst.add(jj) = 0.0;
                                    }
                                }
                            }
                        }
                        for ip in 0..mpan {
                            let mr = MR.min(m_cnt - ip * MR);
                            for jp in 0..npan {
                                let nr = NR.min(n_cnt - jp * NR);
                                let cbase = c.offset(
                                    (m0 + ip * MR) as isize * c_m + (n0 + jp * NR) as isize,
                                );
                                let ap = ap_all.add(ip * kc * MR);
                                let bp = bp_all.add(jp * kc * NR);
                                if mr == MR && nr == NR && avx {
                                    #[cfg(target_arch = "x86_64")]
                                    micro_6x16(kc, ap, bp, cbase, c_m as usize, first);
                                    #[cfg(not(target_arch = "x86_64"))]
                                    micro_gen(mr, nr, kc, ap, bp, cbase, c_m as usize, first);
                                } else {
                                    micro_gen(mr, nr, kc, ap, bp, cbase, c_m as usize, first);
                                }
                            }
                        }
                    }
                    first = false;
                    k0 += kc;
                }
                // Epilogue: alpha, bias, per-column scale, residual.
                let needs_epi =
                    g.alpha != 1.0 || g.bias_slot >= 0 || g.scale_slot >= 0 || g.res_slot >= 0;
                if needs_epi {
                    let bias = (g.bias_slot >= 0).then(|| base(g.bias_slot, g.bias_off));
                    let scale = (g.scale_slot >= 0).then(|| base(g.scale_slot, g.scale_off));
                    let res = (g.res_slot >= 0).then(|| base(g.res_slot, g.res_off));
                    // The residual is indexed exactly like `C` (already
                    // batch-offset); bias/scale are per column.
                    let rbase = res.map(|r| unsafe { r.offset(batch_rel) });
                    unsafe {
                        for i in 0..m_cnt {
                            for j in 0..n_cnt {
                                let rel = (m0 + i) as isize * c_m + (n0 + j) as isize;
                                let p = c.offset(rel);
                                let mut v = *p * g.alpha;
                                if let Some(b) = bias {
                                    v += *b.add(n0 + j);
                                }
                                if let Some(s) = scale {
                                    v *= *s.add(n0 + j);
                                }
                                if let Some(r) = rbase {
                                    v += *r.offset(rel);
                                }
                                *p = v;
                            }
                        }
                    }
                }
            });
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc(m: i64, n: i64, k: i64) -> HostGemm {
        HostGemm {
            a_slot: 0,
            b_slot: 1,
            c_slot: 2,
            a_off: 0,
            b_off: 0,
            c_off: 0,
            m,
            n,
            k,
            a_m: k,
            a_k: 1,
            b_k: n,
            b_n: 1,
            c_m: n,
            c_n: 1,
            batch: 1,
            a_b: 0,
            b_b: 0,
            c_b: 0,
            dtype: dtype::FLOAT32,
            c_dtype: dtype::FLOAT32,
            alpha: 1.0,
            bias_slot: -1,
            bias_off: 0,
            scale_slot: -1,
            scale_off: 0,
            res_slot: -1,
            res_off: 0,
        }
    }

    #[test]
    fn matches_naive_for_prime_m() {
        let (m, n, k) = (789usize, 384usize, 384usize);
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i % 131) as f32) * 0.0078125 - 0.5)
            .collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| ((i % 127) as f32) * 0.0078125 - 0.5)
            .collect();
        let mut c = vec![0.0f32; m * n];
        let g = desc(m as i64, n as i64, k as i64);
        let mut ptrs = [
            a.as_ptr() as *mut u8,
            b.as_ptr() as *mut u8,
            c.as_mut_ptr() as *mut u8,
        ];
        unsafe { run_raw(&g, &mut ptrs).unwrap() };
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for kk in 0..k {
                    s += a[i * k + kk] * b[kk * n + j];
                }
                assert!((s - c[i * n + j]).abs() < 1e-3, "{i},{j}");
            }
        }
    }
}
