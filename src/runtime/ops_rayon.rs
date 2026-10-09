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
use std::collections::HashMap;
use std::rc::Rc;

use crate::renderer::{Renderer, UopRenderer};
use crate::runtime::buffer::{host_of, OpaqueT};
use crate::runtime::device::{register_device, Comp, Compiled, IdentityCompiler, LruAllocator, Prog, TinyElf};
use crate::runtime::ops_interp::InterpProgram;
use crate::runtime::ray_kernel::{RayKernel, SendMemView};
use crate::uop::ctx::UOpCtx;
use crate::uop::flat::{parse_flat, FlatGraph};
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

/// `RayonProgram` — R1 runs the owned `Send` kernel on the host, falling back
/// to the interpreter for kernels the owned IR cannot compile (keeping `RAYON`
/// correct-by-construction). `RAYON`'s R2 range-driven evaluator slots in here.
pub struct RayonProgram {
    kernel: Option<RayKernel>,
    inner: InterpProgram,
    mode: Mode,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Owned IR, `rayon` over the global launch grid (default).
    Parallel,
    /// Owned IR, serial launches (`RAYON_SERIAL=1`).
    Serial,
    /// Always the interpreter (`RAYON_INTERP=1`); also the fallback when
    /// `kernel` is `None`.
    Interp,
}

fn mode_from_env() -> Mode {
    if std::env::var("RAYON_INTERP").is_ok() {
        Mode::Interp
    } else if std::env::var("RAYON_SERIAL").is_ok() {
        Mode::Serial
    } else {
        Mode::Parallel
    }
}

/// Observability: `RayonProgram`s constructed with a compiled owned kernel
/// (fast path) vs those that fell back to the interpreter.
pub static OWNED_KERNELS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static FALLBACK_KERNELS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

impl RayonProgram {
    pub fn new(elf: &TinyElf) -> RayonProgram {
        // Decode the flat graph once and compile the owned IR; `InterpProgram`
        // re-decodes for the fallback (cheap, compile-time only).
        let kernel = decode_lib(elf).and_then(|(mut ctx, uops)| RayKernel::compile(&mut ctx, &uops));
        if kernel.is_some() {
            OWNED_KERNELS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        } else {
            FALLBACK_KERNELS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        if std::env::var("RAYON_VERBOSE").is_ok() {
            eprintln!("[rayon] program {} -> {}", elf.name, if kernel.is_some() { "owned" } else { "fallback" });
        }
        RayonProgram { kernel, inner: InterpProgram::new(elf), mode: mode_from_env() }
    }
}

impl Prog for RayonProgram {
    fn call(
        &mut self,
        bufs: &[Rc<dyn OpaqueT>],
        global_size: [usize; 3],
        local_size: Option<[usize; 3]>,
        vals: &[i64],
        wait: bool,
    ) -> Option<f64> {
        // python's PythonProgram iterates `global_size[::-1]` / `local_size[::-1]`;
        // the owned executor shares that little-endian dim convention.
        if self.mode == Mode::Interp || self.kernel.is_none() {
            return self.inner.call(bufs, global_size, local_size, vals, wait);
        }
        let kernel = self.kernel.as_ref().unwrap();
        let st = std::time::Instant::now();
        // Bind signature slots to raw host windows. The `Rc<dyn OpaqueT>`s are
        // borrowed from `bufs`, which the caller keeps alive until `call`
        // returns; the rayon scope below always joins before then.
        let mut views: HashMap<i64, SendMemView> = HashMap::new();
        for &slot in &kernel.globals {
            let o = &bufs[slot as usize];
            let dt = kernel.sig_dtypes[&slot];
            let isz = dt.itemsize();
            let h = host_of(&**o);
            views.insert(slot, SendMemView::from_host(h, isz, h.size / isz));
        }
        let vals_f: Vec<f64> = vals.iter().map(|&v| v as f64).collect();
        let gs_rev = [global_size[2], global_size[1], global_size[0]];
        let ls_rev = local_size.map(|l| [l[2], l[1], l[0]]);
        let parallel = self.mode == Mode::Parallel;
        kernel.exec(&views, gs_rev, ls_rev, &vals_f, parallel);
        if wait {
            Some(st.elapsed().as_secs_f64())
        } else {
            None
        }
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

/// Decode the elf lib into a `(ctx, uops)` pair for the owned-IR compiler.
pub(crate) fn decode_lib(elf: &TinyElf) -> Option<(UOpCtx, Vec<UopId>)> {
    let src = std::str::from_utf8(&elf.lib).ok()?;
    let fg: FlatGraph = parse_flat(src).ok()?;
    let g = Graph::from_flat(&fg).ok()?;
    let uops = g.roots.clone();
    let n = fg.nodes.len();
    let mut ctx = UOpCtx::new();
    ctx.grow_to(n);
    ctx.g = g;
    Some((ctx, uops))
}

/// The number of workers in the global `rayon` pool (honours
/// `RAYON_NUM_THREADS`). Exposed for examples/tests to report parallelism.
pub fn num_threads() -> usize {
    rayon::current_num_threads()
}

