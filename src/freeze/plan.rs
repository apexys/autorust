//! Planning: lower captured `CompiledLinear`s into a stable `FrozenPlan`.
//!
//! Buffer identity is `(base allocation, byte offset, size, dtype)`. Each base
//! becomes a storage; each kernel argument is a typed view into one. External
//! storages (model inputs/weights) are bound by the runner; everything else is
//! allocated internally and reused across runs.

use std::collections::HashMap;
use std::rc::Rc;

use crate::dtype::DTypeId;
use crate::engine::jit::Jit;
use crate::engine::realize::CompiledProg;
use crate::engine::session::Session;
use crate::runtime::buffer::{Buffer, buffer_of};
use crate::runtime::ops_rust::RustProgram;
use crate::tensor::Tensor;
use crate::uop::Ops;

/// A storage allocation (a `BufferBase`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StorageSpec {
    /// Allocation size in bytes.
    pub nbytes: usize,
    /// Whether the caller supplies this storage at run time.
    pub external: bool,
}

/// A typed window into a storage (a `Buffer` view).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ViewSpec {
    /// Owning storage index.
    pub storage: usize,
    /// Byte offset within the storage.
    pub offset: usize,
    /// Window size in bytes.
    pub nbytes: usize,
    /// Element dtype.
    pub dtype: DTypeId,
}

/// One execution step.
#[derive(Clone, Debug)]
pub enum Step {
    /// Launch a compiled kernel over `bufs` (signature order).
    Kernel {
        /// Index into the kernel table.
        kernel: usize,
        /// View ids in kernel signature order.
        bufs: Vec<usize>,
        /// Scalar ALU args.
        vals: Vec<i64>,
        /// Global launch dimensions.
        global: [usize; 3],
    },
    /// Byte copy between two views (same device).
    Copy {
        /// Destination view.
        dst: usize,
        /// Source view.
        src: usize,
    },
}

/// A backend-independent execution plan.
#[derive(Clone, Debug, Default)]
pub struct FrozenPlan {
    /// Storage allocations, indexed by `ViewSpec::storage`.
    pub storages: Vec<StorageSpec>,
    /// Typed views, indexed by step argument ids.
    pub views: Vec<ViewSpec>,
    /// Ordered kernel/copy steps.
    pub steps: Vec<Step>,
    /// View ids for the caller-supplied inputs, in order.
    pub inputs: Vec<usize>,
    /// View ids for the caller-supplied weights, in order.
    pub weights: Vec<usize>,
    /// View ids for the returned outputs, in order.
    pub outputs: Vec<usize>,
    /// Concrete output shapes, in the same order as `outputs`.
    pub output_shapes: Vec<Vec<i64>>,
    /// Kernel names (for diagnostics/profiling), in kernel-table order.
    pub kernel_names: Vec<String>,
}

impl FrozenPlan {
    /// Number of external storage bindings expected per run.
    pub fn external_count(&self) -> usize {
        self.storages.iter().filter(|s| s.external).count()
    }
}

type ViewKey = (usize, usize, usize, DTypeId);

/// FNV-1a for matching staged `initial_data` to the caller's tensors.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

struct Builder {
    plan: FrozenPlan,
    view_of: HashMap<ViewKey, usize>,
    storage_of: HashMap<usize, usize>,
    kernels: Vec<Rc<CompiledProg>>,
    kernel_of: HashMap<u64, usize>,
    external_hash: HashMap<u64, usize>,
    copies: Vec<(usize, usize)>,
}

impl Builder {
    fn new() -> Builder {
        Builder {
            plan: FrozenPlan::default(),
            view_of: HashMap::new(),
            storage_of: HashMap::new(),
            kernels: Vec::new(),
            kernel_of: HashMap::new(),
            external_hash: HashMap::new(),
            copies: Vec::new(),
        }
    }

    fn intern(&mut self, b: &Buffer, external: bool) -> usize {
        let base = Rc::as_ptr(&b.base) as usize;
        let key = (base, b.offset, b.nbytes(), b.dtype);
        if let Some(&v) = self.view_of.get(&key) {
            return v;
        }
        let storage = *self.storage_of.entry(base).or_insert_with(|| {
            let i = self.plan.storages.len();
            self.plan.storages.push(StorageSpec {
                nbytes: b.base.nbytes(),
                external,
            });
            i
        });
        if external {
            self.plan.storages[storage].external = true;
        }
        let v = self.plan.views.len();
        self.plan.views.push(ViewSpec {
            storage,
            offset: b.offset,
            nbytes: b.nbytes(),
            dtype: b.dtype,
        });
        self.view_of.insert(key, v);
        v
    }

    fn kernel(&mut self, cp: &Rc<CompiledProg>) -> usize {
        if let Some(&i) = self.kernel_of.get(&cp.elf_key) {
            return i;
        }
        let i = self.kernels.len();
        self.kernels.push(cp.clone());
        self.kernel_of.insert(cp.elf_key, i);
        i
    }
}

/// A built plan plus the loaded JIT kernel programs (keep them alive).
pub struct BuiltPlan {
    /// The serializable execution plan.
    pub plan: FrozenPlan,
    /// Loaded `RUST` cdylibs for the plan's kernel table.
    pub progs: Vec<RustProgram>,
    /// Compiled programs, in kernel-table order (for source emission).
    pub kernels: Vec<Rc<CompiledProg>>,
}

/// Resolve a tensor's realized storage, realizing it if necessary.
fn realized(t: &Tensor) -> Buffer {
    if let Some(b) = t.realized_buffer() {
        return b;
    }
    let _ = t.realize_fresh();
    t.realized_buffer()
        .expect("TGERR: freeze could not realize a tensor")
}

/// Lower a captured [`Jit`] into a [`FrozenPlan`].
///
/// `inputs`/`weights` must be the tensors used during the capture (or clones
/// sharing their buffers). `weights` are external like inputs; the split is
/// only for the generated wrapper's argument order.
pub fn build(
    ctx: &crate::tensor::Ctx,
    sess: &mut Session,
    jit: &Jit,
    inputs: &[Tensor],
    weights: &[Tensor],
) -> BuiltPlan {
    // Realize the external tensors before taking the arena borrow: a lazy input
    // would otherwise try to borrow the same `Ctx` from inside `buffer_of`.
    for t in inputs
        .iter()
        .chain(weights.iter())
        .chain(jit.outputs().iter())
    {
        if t.realized_buffer().is_none() {
            let _ = t.realize_fresh();
        }
    }
    let mut c = ctx.borrow_mut();
    let ctx = &mut *c;
    let mut b = Builder::new();
    for t in inputs {
        let buf = realized(t);
        let v = b.intern(&buf, true);
        b.external_hash
            .insert(fnv1a(&crate::engine::realize::buffer_bytes(&buf)), v);
        b.plan.inputs.push(v);
    }
    for t in weights {
        let buf = realized(t);
        let v = b.intern(&buf, true);
        b.external_hash
            .insert(fnv1a(&crate::engine::realize::buffer_bytes(&buf)), v);
        b.plan.weights.push(v);
    }
    for t in jit.outputs() {
        let buf = realized(t);
        let v = b.intern(&buf, false);
        b.plan.outputs.push(v);
        let shape: Vec<i64> = t
            .shape_ctx(ctx)
            .into_vec()
            .into_iter()
            .map(|s| s.as_i64_or_panic())
            .collect();
        b.plan.output_shapes.push(shape);
    }

    for cl in jit.linears() {
        let calls = ctx.srcs(&cl.line).to_vec();
        for call in calls {
            let body = ctx.srcs(&call)[0].clone();
            let resolved = ctx.srcs(&call)[1..].to_vec();
            match ctx.op_of(&body) {
                Ops::Copy => {
                    assert!(resolved.len() >= 2, "TGERR: freeze COPY args");
                    let dst = buffer_of(ctx, &mut sess.buffers, resolved[0].clone());
                    let src = buffer_of(ctx, &mut sess.buffers, resolved[1].clone());
                    let dst = b.intern(&dst, false);
                    let src = b.intern(&src, false);
                    b.plan.steps.push(Step::Copy { dst, src });
                }
                Ops::Program => {
                    let cp = cl
                        .progs
                        .get(&call.id())
                        .unwrap_or_else(|| panic!("TGERR: freeze missing program for call"))
                        .clone();
                    let mut by_slot: HashMap<i64, usize> = HashMap::new();
                    for &g in &cp.pi.globals {
                        let arg = &resolved[g as usize];
                        let buf = buffer_of(ctx, &mut sess.buffers, arg.clone());
                        let vid = b.intern(&buf, false);
                        // `to_param`/bufferize may have created a fresh buffer
                        // with the caller's bytes as `initial_data`. Match it by
                        // content and copy the live tensor into it before the
                        // kernels run.
                        if let Some(init) = arg.node.initial_data.borrow().as_ref() {
                            let h = fnv1a(init.borrow().as_slice());
                            if let Some(&ext) = b.external_hash.get(&h)
                                && ext != vid
                                && !b.copies.contains(&(vid, ext))
                            {
                                b.copies.push((vid, ext));
                            }
                        }
                        by_slot.insert(g, vid);
                    }
                    let fallback = by_slot.values().next().copied();
                    let mut bufs = Vec::with_capacity(cp.elf.signature.len());
                    for (slot, _) in &cp.elf.signature {
                        let vid = by_slot.get(slot).copied().or(fallback).unwrap_or_else(|| {
                            panic!("TGERR: freeze kernel has no buffers for slot {slot}")
                        });
                        bufs.push(vid);
                    }
                    let kernel = b.kernel(&cp);
                    let vals = cp.pi.vals(&HashMap::new());
                    let global = cp.pi.launch_dims(&HashMap::new()).0;
                    b.plan.steps.push(Step::Kernel {
                        kernel,
                        bufs,
                        vals,
                        global,
                    });
                }
                other => panic!("TGERR: freeze unsupported call op {other:?}"),
            }
        }
    }

    if !b.copies.is_empty() {
        let mut steps: Vec<Step> = b
            .copies
            .iter()
            .map(|&(dst, src)| Step::Copy { dst, src })
            .collect();
        steps.append(&mut b.plan.steps);
        b.plan.steps = steps;
    }

    let progs = b
        .kernels
        .iter()
        .map(|cp| RustProgram::new(cp.elf.as_ref()))
        .collect();
    let mut plan = b.plan;
    plan.kernel_names = b.kernels.iter().map(|cp| cp.elf.name.clone()).collect();
    BuiltPlan {
        plan,
        progs,
        kernels: b.kernels,
    }
}
