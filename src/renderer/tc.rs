//! `codegen/opt/tc.py` port — the `TensorCore` description table. The Scheduler's
//! `_apply_tc_opt` consumes these to turn a `[M,K]x[K,N]->[M,N]` reduce into a
//! `Ops.WMMA`, and the CUDA renderer consumes the per-op shapes to emit the
//! `mma.sync.aligned` asm.
//!
//! Ported: the CUDA family (`cuda_sm80` + `cuda_sm89` with fp8). AMD/Metal are
//! not needed (no such backend).

use std::collections::HashMap;

use crate::dtype::DTypeId;

/// `TensorCore` (tc.py:6). D = A * B + C, A is (M x K), B is (K x N), C/D are (M x N).
#[derive(Clone, Debug)]
pub struct TensorCore {
    /// N, M, K
    pub dims: (usize, usize, usize),
    /// number of threads that construct the warp
    pub threads: usize,
    /// elements per-thread to load/store from A/B/C
    pub elements_per_thread: (usize, usize, usize),
    pub dtype_in: DTypeId,
    pub dtype_out: DTypeId,
    /// ordered tuple of "ux"/"lx" kernel opts ("ux" upcasts dim x, "lx" localizes dim x)
    pub opts: Vec<&'static str>,
    /// (local_swizzle, upcast_swizzle, reduce_swizzle) for remap 0 and remap 1.
    pub swizzle: [Vec<Vec<&'static str>>; 2],
}

impl TensorCore {
    /// `get_reduce_axes` — `[(i, 2) for i in range(log2(dims[2]))]`.
    pub fn get_reduce_axes(&self) -> Vec<(usize, usize)> {
        (0..(self.dims.2 as f64).log2() as usize)
            .map(|i| (i, 2))
            .collect()
    }

    pub fn get_upcast_axes(&self) -> Vec<&'static str> {
        self.opts
            .iter()
            .copied()
            .filter(|o| o.starts_with('u'))
            .collect()
    }

    pub fn get_local_axes(&self) -> Vec<&'static str> {
        self.opts
            .iter()
            .copied()
            .filter(|o| o.starts_with('l'))
            .collect()
    }

    /// `base_shape_str` — opts in order (`u0`,`l0`,`l0`,…) then `r0..`.
    pub fn base_shape_str(&self) -> Vec<String> {
        let mut ret = Vec::new();
        let (mut cu, mut cl) = (0usize, 0usize);
        for opt in &self.opts {
            if opt.starts_with('u') {
                ret.push(format!("u{cu}"));
                cu += 1;
            } else {
                ret.push(format!("l{cl}"));
                cl += 1;
            }
        }
        for i in 0..self.get_reduce_axes().len() {
            ret.push(format!("r{i}"));
        }
        ret
    }

    /// `base_upcast_axes` — `([r0..] + [u0..])[::-1]`.
    pub fn base_upcast_axes(&self) -> Vec<String> {
        let mut v: Vec<String> = self
            .get_reduce_axes()
            .iter()
            .map(|(i, _)| format!("r{i}"))
            .collect();
        v.extend(
            self.get_upcast_axes()
                .iter()
                .map(|o| format!("u{}", &o[1..])),
        );
        v.reverse();
        v
    }

    /// `_remaps` — map the forward shape string to the flattened swizzle tuples.
    pub fn remaps(&self) -> Vec<HashMap<String, String>> {
        let (la, ua, ra) = (
            self.get_local_axes().len(),
            self.get_upcast_axes().len(),
            self.get_reduce_axes().len(),
        );
        let mut fwd: Vec<String> = Vec::new();
        fwd.extend((0..la).map(|i| format!("l{i}")));
        fwd.extend((0..ua).map(|i| format!("u{i}")));
        fwd.extend((0..ra).map(|i| format!("r{i}")));
        self.swizzle
            .iter()
            .map(|s| {
                let flat: Vec<String> = s.iter().flatten().map(|x| x.to_string()).collect();
                fwd.iter().cloned().zip(flat).collect()
            })
            .collect()
    }

    /// `permutes_for_shape_str`.
    pub fn permutes_for_shape_str(&self, shape_str: &[String]) -> (Vec<usize>, Vec<usize>) {
        let remaps = self.remaps();
        let mut out: Vec<Vec<usize>> = Vec::new();
        for remap in &remaps {
            let mut p = Vec::new();
            for (i, ss) in shape_str.iter().enumerate() {
                match remap.get(ss) {
                    Some(t) => {
                        let j = shape_str.iter().position(|x| x == t).unwrap_or_else(|| {
                            panic!("TGERR: swizzle target {t} not in shape_str")
                        });
                        p.push(j);
                    }
                    None => p.push(i),
                }
            }
            out.push(p);
        }
        (out[0].clone(), out[1].clone())
    }
}

/// `cuda_tc_opts` — shared by all CUDA shapes with M=16 N=8.
const CUDA_TC_OPTS: [&str; 7] = ["u0", "l0", "l0", "l1", "l1", "l1", "u1"];

const D_HALF: DTypeId = crate::dtype::FLOAT16;
const D_FLOAT: DTypeId = crate::dtype::FLOAT32;
const D_BF16: DTypeId = crate::dtype::BFLOAT16;
const D_FP8E4M3: DTypeId = crate::dtype::FP8E4M3;
const D_FP8E5M2: DTypeId = crate::dtype::FP8E5M2;

fn cuda_81616() -> Vec<TensorCore> {
    [(D_HALF, D_FLOAT), (D_BF16, D_FLOAT), (D_HALF, D_HALF)]
        .into_iter()
        .map(|(di, do_)| TensorCore {
            dims: (8, 16, 16),
            threads: 32,
            elements_per_thread: (8, 4, 4),
            dtype_in: di,
            dtype_out: do_,
            opts: CUDA_TC_OPTS.to_vec(),
            swizzle: [
                vec![
                    vec!["r1", "r2", "l2", "l3", "l4"],
                    vec!["u1", "r3"],
                    vec!["l0", "l1", "u0", "r0"],
                ],
                vec![
                    vec!["r1", "r2", "u0", "l0", "l1"],
                    vec!["r0", "r3"],
                    vec!["l2", "l3", "l4", "u1"],
                ],
            ],
        })
        .collect()
}

fn cuda_81632_f8() -> Vec<TensorCore> {
    [(D_FP8E4M3, D_FLOAT), (D_FP8E5M2, D_FLOAT)]
        .into_iter()
        .map(|(di, do_)| TensorCore {
            dims: (8, 16, 32),
            threads: 32,
            elements_per_thread: (16, 8, 4),
            dtype_in: di,
            dtype_out: do_,
            opts: CUDA_TC_OPTS.to_vec(),
            swizzle: [
                vec![
                    vec!["r2", "r3", "l2", "l3", "l4"],
                    vec!["u1", "r4"],
                    vec!["l0", "l1", "u0", "r0", "r1"],
                ],
                vec![
                    vec!["r2", "r3", "u0", "l0", "l1"],
                    vec!["r1", "r4"],
                    vec!["l2", "l3", "l4", "u1", "r0"],
                ],
            ],
        })
        .collect()
}

fn cuda_8168_f16() -> Vec<TensorCore> {
    [(D_HALF, D_FLOAT), (D_HALF, D_HALF)]
        .into_iter()
        .map(|(di, do_)| TensorCore {
            dims: (8, 16, 8),
            threads: 32,
            elements_per_thread: (4, 2, 4),
            dtype_in: di,
            dtype_out: do_,
            opts: CUDA_TC_OPTS.to_vec(),
            swizzle: [
                vec![
                    vec!["r1", "r2", "l2", "l3", "l4"],
                    vec!["r0", "u1"],
                    vec!["l0", "l1", "u0"],
                ],
                vec![
                    vec!["r1", "r2", "u0", "l0", "l1"],
                    vec!["u1", "r0"],
                    vec!["l2", "l3", "l4"],
                ],
            ],
        })
        .collect()
}

fn cuda_8168_tf32() -> Vec<TensorCore> {
    vec![TensorCore {
        dims: (8, 16, 8),
        threads: 32,
        elements_per_thread: (4, 2, 4),
        dtype_in: D_FLOAT,
        dtype_out: D_FLOAT,
        opts: CUDA_TC_OPTS.to_vec(),
        swizzle: [
            vec![
                vec!["r0", "r1", "l2", "l3", "l4"],
                vec!["u1", "r2"],
                vec!["l0", "l1", "u0"],
            ],
            vec![
                vec!["r0", "r1", "u0", "l0", "l1"],
                vec!["u1", "r2"],
                vec!["l2", "l3", "l4"],
            ],
        ],
    }]
}

fn cuda_sm75() -> Vec<TensorCore> {
    cuda_8168_f16()
}

fn cuda_sm80() -> Vec<TensorCore> {
    let mut v = cuda_81616();
    v.extend(cuda_8168_f16());
    v.extend(cuda_8168_tf32());
    v
}

fn cuda_sm89() -> Vec<TensorCore> {
    let mut v = cuda_sm80();
    v.extend(cuda_81632_f8());
    v
}

/// `get_cuda(arch)` — `sm_89`/`sm_80`/`sm_75` else empty.
pub fn get_cuda(arch: &str) -> Vec<TensorCore> {
    let ver: i64 = arch
        .strip_prefix("sm_")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if ver >= 89 {
        cuda_sm89()
    } else if ver >= 80 {
        cuda_sm80()
    } else if ver >= 75 {
        cuda_sm75()
    } else {
        vec![]
    }
}
