//! Execute a [`FrozenPlan`] with a kernel table.
//!
//! The JIT path loads the `RUST` cdylibs and uses their function pointers; the
//! AOT path will provide the same pointers from statically linked functions.
//! Internal storages are allocated once and reused across runs; external
//! storages are bound to the live input/weight tensors.

use std::time::Instant;

use rayon::prelude::*;

use crate::runtime::buffer::{OpaqueView, host_of};
use crate::runtime::device::device;
use crate::runtime::ops_rust::{KernelFn, RustProgram};
use crate::tensor::Tensor;

use super::plan::{FrozenPlan, Step};

/// A reusable executor for a frozen plan.
pub struct FrozenRunner {
    plan: FrozenPlan,
    /// Keeps the loaded cdylibs alive for the function pointers in `fns`.
    _progs: Vec<RustProgram>,
    fns: Vec<KernelFn>,
    /// Internal storage allocations, reused across runs (empty for external).
    internal: Vec<Vec<u8>>,
    /// Per-kernel `(calls, seconds)` when `FROZEN_PROF` is set.
    prof: Vec<(u64, f64)>,
    /// Host GEMM `(calls, seconds)` when `FROZEN_PROF` is set.
    gemm_prof: (u64, f64),
    prof_on: bool,
}

impl FrozenRunner {
    /// Build a runner from a plan and its loaded JIT kernels.
    pub fn new(plan: FrozenPlan, progs: Vec<RustProgram>) -> FrozenRunner {
        let fns: Vec<KernelFn> = progs.iter().map(|p| p.kernel_fn()).collect();
        let mut runner = FrozenRunner::from_fns(plan, fns);
        runner._progs = progs;
        runner
    }

    /// Build a runner from a plan and a statically linked kernel table (AOT).
    pub fn from_fns(plan: FrozenPlan, fns: Vec<KernelFn>) -> FrozenRunner {
        if let Some(max_kernel) = plan
            .steps
            .iter()
            .filter_map(|s| match s {
                Step::Kernel { kernel, .. } => Some(*kernel),
                Step::Copy { .. } | Step::Gemm { .. } => None,
            })
            .max()
        {
            assert!(
                fns.len() > max_kernel,
                "TGERR: freeze kernel table is too small"
            );
        }
        let internal = plan
            .storages
            .iter()
            .map(|s| {
                if s.external {
                    Vec::new()
                } else {
                    vec![0u8; s.nbytes]
                }
            })
            .collect();
        FrozenRunner {
            plan,
            _progs: Vec::new(),
            prof: vec![(0u64, 0.0f64); fns.len()],
            gemm_prof: (0, 0.0),
            prof_on: crate::helpers::env::get("FROZEN_PROF").is_some(),
            fns,
            internal,
        }
    }

    /// The plan this runner executes.
    pub fn plan(&self) -> &FrozenPlan {
        &self.plan
    }

    /// Bind external storages from `inputs`/`weights` and execute the plan.
    /// Returns one byte buffer per output view.
    pub fn run(&mut self, inputs: &[Tensor], weights: &[Tensor]) -> Vec<Vec<u8>> {
        assert_eq!(
            inputs.len(),
            self.plan.inputs.len(),
            "TGERR: freeze input count"
        );
        assert_eq!(
            weights.len(),
            self.plan.weights.len(),
            "TGERR: freeze weight count"
        );

        let dev = device("RUST");
        let mut base_ptrs = vec![0usize; self.plan.storages.len()];
        let mut keep: Vec<OpaqueView> = Vec::new();
        let plan_views = &self.plan.views;

        let bind =
            |t: &Tensor, view_id: usize, base_ptrs: &mut Vec<usize>, keep: &mut Vec<OpaqueView>| {
                let b = t
                    .realized_buffer()
                    .or_else(|| {
                        let _ = t.realize_fresh();
                        t.realized_buffer()
                    })
                    .unwrap_or_else(|| panic!("TGERR: freeze external tensor is not realized"));
                b.ensure_allocated(&dev);
                let o = b.get_buf(&dev);
                let h = host_of(&*o);
                let base = h.mem.borrow().as_ptr() as usize;
                let storage = plan_views[view_id].storage;
                base_ptrs[storage] = base;
                keep.push(o);
            };

        for (view_id, t) in self.plan.inputs.iter().zip(inputs) {
            bind(t, *view_id, &mut base_ptrs, &mut keep);
        }
        for (view_id, t) in self.plan.weights.iter().zip(weights) {
            bind(t, *view_id, &mut base_ptrs, &mut keep);
        }
        for (i, s) in self.plan.storages.iter().enumerate() {
            if !s.external {
                base_ptrs[i] = self.internal[i].as_mut_ptr() as usize;
            }
        }
        for (i, p) in base_ptrs.iter().enumerate() {
            assert!(
                *p != 0,
                "TGERR: freeze storage {i} is external but was not bound"
            );
        }

        let ptr_of = |view_id: usize, base_ptrs: &[usize]| {
            let v = &self.plan.views[view_id];
            base_ptrs[v.storage] + v.offset
        };

        for step in &self.plan.steps {
            match step {
                Step::Kernel {
                    kernel,
                    bufs,
                    vals,
                    global,
                } => {
                    let ptrs: Vec<usize> = bufs.iter().map(|&v| ptr_of(v, &base_ptrs)).collect();
                    let timer = self.prof_on.then(Instant::now);
                    run_kernel(self.fns[*kernel], &ptrs, vals, *global);
                    if let Some(t) = timer {
                        let e = &mut self.prof[*kernel];
                        e.0 += 1;
                        e.1 += t.elapsed().as_secs_f64();
                    }
                }
                Step::Gemm { gemm, bufs } => {
                    let timer = self.prof_on.then(Instant::now);
                    let ptrs: Vec<*mut u8> = bufs
                        .iter()
                        .map(|&v| {
                            (base_ptrs[self.plan.views[v].storage] + self.plan.views[v].offset)
                                as *mut u8
                        })
                        .collect();
                    // SAFETY: every view is bound and large enough for the
                    // descriptor's offsets/strides (checked when the plan was
                    // built from the live tensors).
                    let g: &crate::schedule::host_gemm::HostGemm = gemm;
                    if let Err(e) = unsafe { crate::runtime::cpu_gemm::run_raw(g, &ptrs) } {
                        panic!("TGERR: frozen cpu gemm: {e}");
                    }
                    if let Some(t) = timer {
                        self.gemm_prof.0 += 1;
                        self.gemm_prof.1 += t.elapsed().as_secs_f64();
                    }
                }
                Step::Copy { dst, src } => {
                    let d = ptr_of(*dst, &base_ptrs);
                    let s = ptr_of(*src, &base_ptrs);
                    let n = self.plan.views[*dst].nbytes;
                    assert_eq!(
                        n, self.plan.views[*src].nbytes,
                        "TGERR: freeze copy size mismatch"
                    );
                    unsafe {
                        std::ptr::copy_nonoverlapping(s as *const u8, d as *mut u8, n);
                    }
                }
            }
        }

        self.plan
            .outputs
            .iter()
            .map(|&v| {
                let view = &self.plan.views[v];
                let ptr = base_ptrs[view.storage] + view.offset;
                unsafe { std::slice::from_raw_parts(ptr as *const u8, view.nbytes).to_vec() }
            })
            .collect()
    }

    /// Print the per-kernel profile collected under `FROZEN_PROF`.
    pub fn report_profile(&self) {
        let mut rows: Vec<(usize, u64, f64)> = self
            .prof
            .iter()
            .enumerate()
            .filter(|(_, e)| e.0 > 0)
            .map(|(i, e)| (i, e.0, e.1))
            .collect();
        rows.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        println!("--- frozen kernel profile ---");
        if self.gemm_prof.0 > 0 {
            println!(
                "{:<28} {:>6} {:>10} {:>12}",
                "host_gemm",
                self.gemm_prof.0,
                format!("{:.4}", self.gemm_prof.1),
                format!("{:.3} ms", self.gemm_prof.1 * 1e3 / self.gemm_prof.0 as f64)
            );
        }
        println!(
            "{:<28} {:>6} {:>10} {:>12}",
            "kernel", "calls", "total (s)", "per call"
        );
        for (i, calls, total) in rows {
            let name = self
                .plan
                .kernel_names
                .get(i)
                .map(String::as_str)
                .unwrap_or("?");
            println!(
                "{:<28} {:>6} {:>10.4} {:>10.3} ms",
                name,
                calls,
                total,
                total / calls as f64 * 1e3
            );
        }
    }
}

/// Launch a kernel over the first global dimension, chunked across rayon.
pub(crate) fn run_kernel(f: KernelFn, ptrs: &[usize], vals: &[i64], global: [usize; 3]) {
    let gs = [global[0].max(1), global[1].max(1), global[2].max(1)];
    let threads = rayon::current_num_threads().max(1);
    let nchunks = if gs[0] > 1 {
        (threads * 8).min(gs[0]).max(1)
    } else {
        1
    };
    let total = nchunks * gs[1] * gs[2];
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
}
