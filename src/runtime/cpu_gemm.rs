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
//! plus 2 operand vectors, no spills). The fringe rows/columns go through a scalar
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
            kc: 256,
        },
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn micro_6x16(
    kc: usize,
    ap: *const f32,
    a_rs: usize,
    a_ks: usize,
    bp: *const f32,
    b_ks: usize,
    c: *mut f32,
    ldc: usize,
    first: bool,
) {
    use std::arch::x86_64::*;
    let mut acc = [_mm256_setzero_ps(); 12];
    for k in 0..kc {
        let b0 = _mm256_loadu_ps(bp.add(k * b_ks));
        let b1 = _mm256_loadu_ps(bp.add(k * b_ks + 8));
        for i in 0..MR {
            let a = _mm256_set1_ps(*ap.add(i * a_rs + k * a_ks));
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
// The loops are written in matrix-index order on purpose: this is the scalar
// reference for the AVX2 microkernel, and the argument list is its microkernel
// ABI.
#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
unsafe fn micro_gen(
    mr: usize,
    nr: usize,
    kc: usize,
    ap: *const f32,
    a_rs: usize,
    a_ks: usize,
    bp: *const f32,
    b_ks: usize,
    c: *mut f32,
    ldc: usize,
    first: bool,
) {
    let mut acc = [[0.0f32; NR]; MR];
    for k in 0..kc {
        for j in 0..nr {
            let b = *bp.add(k * b_ks + j);
            for i in 0..mr {
                acc[i][j] += *ap.add(i * a_rs + k * a_ks) * b;
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
// The packing loops offset by the descriptor's element strides, which are
// non-negative `usize`s by construction.
#[allow(clippy::ptr_offset_with_cast)]
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
    // Fast path: A row-major (k contiguous), B row-major or a `[n, k]` weight
    // packed into `[k, n]`. The per-task packing below re-reads B for every
    // m-block and A for every n-block, which dominates small-M GEMMs (ViT's
    // 197-row linears).
    let direct = g.a_k == 1 && (g.b_n == 1 || g.b_k == 1);
    let pack_b = |b_addr: usize| -> Option<Vec<f32>> {
        if !direct || g.b_n == 1 {
            return None;
        }
        let mut p = vec![0.0f32; k * n];
        let b_n = g.b_n as usize;
        p.par_chunks_mut(n).enumerate().for_each(|(kk, row)| {
            let src = unsafe { (b_addr as *const f32).offset(kk as isize) };
            for (j, dst) in row.iter_mut().enumerate() {
                unsafe { *dst = *src.offset((j * b_n) as isize) };
            }
        });
        Some(p)
    };
    // A batched GEMM over the same (unbatched) weight packs it once.
    let weight = if g.b_b == 0 {
        let b_head = bases[g.b_slot as usize] as *const f32;
        pack_b(unsafe { b_head.offset(g.b_off as isize) } as usize)
    } else {
        None
    };
    let weight_addr = weight.as_ref().map(|p| p.as_ptr() as usize);
    for batch in 0..bsz {
        let base = |slot: i64, off: i64| -> *mut u8 {
            let p = bases[slot as usize] as *mut u8;
            unsafe { p.offset((off * 4) as isize) }
        };
        let a = unsafe { base(g.a_slot, g.a_off).add(batch * g.a_b as usize * 4) as *mut f32 };
        let b = unsafe { base(g.b_slot, g.b_off).add(batch * g.b_b as usize * 4) as *mut f32 };
        let c = unsafe { base(g.c_slot, g.c_off).add(batch * g.c_b as usize * 4) as *mut f32 };
        let batch_rel = batch as isize * g.c_b as isize;
        let (a_m, a_k) = (g.a_m as isize, g.a_k as isize);
        let (b_k, b_n) = (g.b_k as isize, g.b_n as isize);
        let c_m = g.c_m as isize;
        let nb = n.div_ceil(t.nc);
        let tasks = m.div_ceil(t.mc) * nb;
        let need = t.mc * t.kc + t.kc * t.nc;
        // The shared weight pack lives outside the batch loop; a batched B is
        // packed per batch.
        let packed_b: Option<Vec<f32>> = if weight_addr.is_some() {
            None
        } else {
            pack_b(b as usize)
        };
        let b_fast_addr = weight_addr
            .or_else(|| packed_b.as_ref().map(|p| p.as_ptr() as usize))
            .unwrap_or(b as usize);
        let b_fast_ks = if weight_addr.is_some() || packed_b.is_some() {
            n
        } else {
            b_k as usize
        };
        let (a_addr, b_addr, c_addr) = (a as usize, b as usize, c as usize);
        (0..tasks).into_par_iter().for_each(|task| {
            let mb = task / nb;
            let nblk = task % nb;
            let m0 = mb * t.mc;
            let n0 = nblk * t.nc;
            let m_cnt = t.mc.min(m - m0);
            let n_cnt = t.nc.min(n - n0);
            let mpan = m_cnt.div_ceil(MR);
            let npan = n_cnt.div_ceil(NR);
            if direct {
                let a = a_addr as *mut f32;
                let bf = b_fast_addr as *const f32;
                let c = c_addr as *mut f32;
                let mut k0 = 0;
                let mut first = true;
                while k0 < k {
                    let kc = t.kc.min(k - k0);
                    unsafe {
                        for ip in 0..mpan {
                            let mr = MR.min(m_cnt - ip * MR);
                            for jp in 0..npan {
                                let nr = NR.min(n_cnt - jp * NR);
                                let cbase = c.offset(
                                    (m0 + ip * MR) as isize * c_m + (n0 + jp * NR) as isize,
                                );
                                let ap = a.offset(
                                    (m0 + ip * MR) as isize * a_m + k0 as isize,
                                ) as *const f32;
                                let bp = bf.offset(
                                    (k0 as isize) * b_fast_ks as isize
                                        + (n0 + jp * NR) as isize,
                                );
                                if mr == MR && nr == NR && avx {
                                    #[cfg(target_arch = "x86_64")]
                                    micro_6x16(
                                        kc,
                                        ap,
                                        a_m as usize,
                                        1,
                                        bp,
                                        b_fast_ks,
                                        cbase,
                                        c_m as usize,
                                        first,
                                    );
                                    #[cfg(not(target_arch = "x86_64"))]
                                    micro_gen(
                                        mr,
                                        nr,
                                        kc,
                                        ap,
                                        a_m as usize,
                                        1,
                                        bp,
                                        b_fast_ks,
                                        cbase,
                                        c_m as usize,
                                        first,
                                    );
                                } else {
                                    micro_gen(
                                        mr,
                                        nr,
                                        kc,
                                        ap,
                                        a_m as usize,
                                        1,
                                        bp,
                                        b_fast_ks,
                                        cbase,
                                        c_m as usize,
                                        first,
                                    );
                                }
                            }
                        }
                    }
                    first = false;
                    k0 += kc;
                }
            } else {
                SCRATCH.with(|slot| {
                    let mut buf = slot.borrow_mut();
                    buf.resize(need, 0.0);
                    let ap_all = buf.as_mut_ptr();
                    let bp_all = unsafe { ap_all.add(t.mc * t.kc) };
                    let a = a_addr as *mut f32;
                    let b = b_addr as *mut f32;
                    let c = c_addr as *mut f32;
                    let mut k0 = 0;
                    let mut first = true;
                    while k0 < k {
                        let kc = t.kc.min(k - k0);
                        unsafe {
                            // Pack A panels: ap[(ip*kc + kk)*MR + ii], zero-padded.
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
                            // Pack B panels: bp[(jp*kc + kk)*NR + jj], zero-padded.
                            if b_k == 1 {
                                for jp in 0..npan {
                                    let cols = NR.min(n_cnt - jp * NR);
                                    for jj in 0..NR {
                                        let dst = bp_all.add(jp * kc * NR + jj);
                                        if jj < cols {
                                            let src =
                                                b.offset((n0 + jp * NR + jj) as isize * b_n);
                                            for kk in 0..kc {
                                                *dst.add(kk * NR) =
                                                    *src.offset((k0 + kk) as isize);
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
                                            (k0 + kk) as isize * b_k
                                                + (n0 + jp * NR) as isize * b_n,
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
                                        (m0 + ip * MR) as isize * c_m
                                            + (n0 + jp * NR) as isize,
                                    );
                                    let ap = ap_all.add(ip * kc * MR);
                                    let bp = bp_all.add(jp * kc * NR);
                                    if mr == MR && nr == NR && avx {
                                        #[cfg(target_arch = "x86_64")]
                                        micro_6x16(
                                            kc, ap, 1, MR, bp, NR, cbase, c_m as usize, first,
                                        );
                                        #[cfg(not(target_arch = "x86_64"))]
                                        micro_gen(
                                            mr, nr, kc, ap, 1, MR, bp, NR, cbase, c_m as usize,
                                            first,
                                        );
                                    } else {
                                        micro_gen(
                                            mr, nr, kc, ap, 1, MR, bp, NR, cbase, c_m as usize,
                                            first,
                                        );
                                    }
                                }
                            }
                        }
                        first = false;
                        k0 += kc;
                    }
                });
            }
            // Epilogue: alpha, bias, per-column scale, residual.
            let needs_epi =
                g.alpha != 1.0 || g.bias_slot >= 0 || g.scale_slot >= 0 || g.res_slot >= 0;
            if needs_epi {
                let c = c_addr as *mut f32;
                let bias = (g.bias_slot >= 0)
                    .then(|| base(g.bias_slot, g.bias_off) as *const f32);
                let scale = (g.scale_slot >= 0)
                    .then(|| base(g.scale_slot, g.scale_off) as *const f32);
                let res = (g.res_slot >= 0)
                    .then(|| unsafe { (base(g.res_slot, g.res_off) as *const f32).offset(batch_rel) });
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
                            if let Some(r) = res {
                                v += *r.offset(rel);
                            }
                            *p = v;
                        }
                    }
                }
            }
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
            a_dtype: dtype::FLOAT32,
            b_dtype: dtype::FLOAT32,
            c_dtype: dtype::FLOAT32,
            alpha: 1.0,
            bias_slot: -1,
            bias_off: 0,
            bias_dtype: crate::dtype::DTypeId(20),
            scale_slot: -1,
            scale_off: 0,
            res_slot: -1,
            res_off: 0,
        }
    }

    /// The ViT `Linear` layout: `B` is stored `[n, k]` (`b_k == 1`,
    /// `b_n == k`) and the epilogue has bias/alpha/residual. Exercises the
    /// direct path (B packed once per GEMM).
    #[test]
    fn matches_naive_weight_layout_with_epilogue() {
        let (m, n, k) = (197usize, 96usize, 768usize);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 131) as f32) * 0.0078125 - 0.5).collect();
        let bt: Vec<f32> = (0..n * k).map(|i| ((i % 127) as f32) * 0.0078125 - 0.5).collect();
        let bias: Vec<f32> = (0..n).map(|j| j as f32 * 0.01).collect();
        let res: Vec<f32> = (0..m * n).map(|i| ((i % 17) as f32) * 0.001).collect();
        let mut c: Vec<f32> = (0..m * n).map(|i| ((i % 19) as f32) * 0.002).collect();
        let mut g = desc(m as i64, n as i64, k as i64);
        // B = [n, k] row-major.
        g.b_k = 1;
        g.b_n = k as i64;
        g.alpha = 0.5;
        g.bias_slot = 3;
        g.res_slot = 4;
        let ptrs = [
            a.as_ptr() as *mut u8,
            bt.as_ptr() as *mut u8,
            c.as_mut_ptr() as *mut u8,
            bias.as_ptr() as *mut u8,
            res.as_ptr() as *mut u8,
        ];
        unsafe { run_raw(&g, &ptrs).unwrap() };
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for kk in 0..k {
                    s += a[i * k + kk] * bt[j * k + kk];
                }
                let want = 0.5 * s + bias[j] + res[i * n + j];
                assert!((want - c[i * n + j]).abs() < 1e-3, "{i},{j}");
            }
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
        let ptrs = [
            a.as_ptr() as *mut u8,
            b.as_ptr() as *mut u8,
            c.as_mut_ptr() as *mut u8,
        ];
        unsafe { run_raw(&g, &ptrs).unwrap() };
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
