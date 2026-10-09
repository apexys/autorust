//! `ops_rayon` — the multithreaded CPU backend (device `RAYON`).
//!
//! This is the sibling of `ops_interp`, not a replacement: `ops_interp` stays
//! the faithful reference interpreter (`INTERP`) used for debugging and as the
//! differential oracle, while `RAYON` is the production CPU backend that
//! executes a *higher-level* (pre-`ndim → loop`) kernel IR and parallelizes
//! with `rayon`. See `CURRENT_STATE.md` §9.8 for the design.
//!
//! **R0 (this commit):** scaffolding only. `RAYON` is registered as a first-class
//! device with its own `Renderer` facts (device label `"RAYON"`) but composes
//! the interpreter's program for execution, so the device is correct and usable
//! immediately. The owned `Send` IR and the range-driven evaluator land in
//! R1/R2; until then every kernel runs through the scalar interpreter path.
//!
//! The whole module is behind the `backend_rayon` feature (optional `rayon`
//! dependency), keeping the default `--no-default-features` build std-only.

use std::cell::RefCell;
use std::rc::Rc;

use crate::renderer::{Renderer, UopRenderer};
use crate::runtime::device::{register_device, Comp, Compiled, IdentityCompiler, LruAllocator, Prog, TinyElf};
use crate::runtime::interp::exec_kernel_with_dims;
use crate::runtime::ops_interp::InterpProgram;
use crate::uop::ctx::UOpCtx;
use crate::uop::flat::FlatGraph;
use crate::uop::graph::Graph;
use crate::uop::UopId;

/// `RayonRenderer` — the `RAYON` facts. The artifact is the same flat-JSON uop
/// wire format the interpreter uses (`ops_interp`), so R0 executes through
/// `InterpProgram`. R2 replaces this with the owned range-driven IR.
pub struct RayonUopRenderer {
    facts: Renderer,
}

impl Default for RayonUopRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl RayonUopRenderer {
    pub fn new() -> RayonUopRenderer {
        RayonUopRenderer { facts: Renderer::rayon() }
    }
}

impl UopRenderer for RayonUopRenderer {
    fn facts(&self) -> &Renderer {
        &self.facts
    }

    /// Same flat-JSON serialization as `PythonRenderer.render`.
    fn render(&self, ctx: &mut UOpCtx, lst: &[UopId]) -> String {
        let saved_roots = ctx.g.roots.clone();
        ctx.g.roots = lst.to_vec();
        let flat = ctx.g.to_flat();
        ctx.g.roots = saved_roots;
        crate::uop::flat::to_string_flat(&flat)
    }
}

/// `RayonProgram` — R0 wraps the interpreter program. Kept as a distinct type
/// (not a bare `InterpProgram`) so `RAYON` has its own program identity and the
/// R1/R2 `Send` IR slots in without touching `ops_interp`.
pub struct RayonProgram {
    inner: InterpProgram,
}

impl RayonProgram {
    pub fn new(elf: &TinyElf) -> RayonProgram {
        RayonProgram { inner: InterpProgram::new(elf) }
    }
}

impl Prog for RayonProgram {
    fn call(
        &mut self,
        bufs: &[Rc<dyn crate::runtime::buffer::OpaqueT>],
        global_size: [usize; 3],
        local_size: Option<[usize; 3]>,
        vals: &[i64],
        wait: bool,
    ) -> Option<f64> {
        self.inner.call(bufs, global_size, local_size, vals, wait)
    }
}

/// The `RAYON` composition + `Device._DeviceMap["RAYON"]` assignment.
#[allow(clippy::type_complexity)]
pub fn register() {
    fn make() -> Rc<Compiled> {
        let renderer: Box<dyn UopRenderer> = Box::new(RayonUopRenderer::new());
        let compiler: Rc<dyn Comp> = Rc::new(IdentityCompiler::default());
        let ctor: Rc<dyn Fn(&TinyElf) -> Rc<RefCell<dyn Prog>>> =
            Rc::new(|elf| Rc::new(RefCell::new(RayonProgram::new(elf))));
        Compiled::new("RAYON", renderer, LruAllocator::new(), compiler, ctor)
    }
    register_device("RAYON", Rc::new(make));
}

/// Decode the elf lib into a `(ctx, uops)` pair. Reserved for the R1 owned-IR
/// compiler; kept here so `rayon` is a used dependency from R0 (the crate must
/// link the feature) and so the decode path is centralized.
#[allow(dead_code)]
pub(crate) fn decode_lib(elf: &TinyElf) -> (UOpCtx, Vec<UopId>) {
    let src = std::str::from_utf8(&elf.lib).expect("TGERR: rayon elf lib not utf8");
    let fg: FlatGraph = crate::uop::flat::parse_flat(src).expect("TGERR: rayon elf lib parse");
    let g = Graph::from_flat(&fg).expect("TGERR: rayon elf lib graph");
    let uops = g.roots.clone();
    let n = fg.nodes.len();
    let mut ctx = UOpCtx::new();
    ctx.grow_to(n);
    ctx.g = g;
    (ctx, uops)
}

/// Initialize the `rayon` thread pool explicitly (optional; rayon lazily builds
/// a global pool on first use). Exposed so examples/tests can warm the pool and
/// report the worker count. Honours `RAYON_NUM_THREADS`.
pub fn num_threads() -> usize {
    rayon::current_num_threads()
}

/// Reserved: execute the scalar path through `exec_kernel_with_dims`. Kept so
/// the interpreter import is exercised and documents the R0 fallback boundary.
#[allow(dead_code)]
pub(crate) fn exec_scalar(
    ctx: &mut UOpCtx,
    uops: &[UopId],
    views: &std::collections::HashMap<i64, crate::runtime::interp::MemView>,
    vals: &[f64],
    global: [usize; 3],
    local: Option<[usize; 3]>,
) {
    exec_kernel_with_dims(ctx, uops, views, vals, global, local);
}
