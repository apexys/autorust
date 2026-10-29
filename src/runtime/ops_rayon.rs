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

use crate::dtype::DTypeId;
use crate::renderer::{Renderer, UopRenderer};
use crate::runtime::buffer::{OpaqueT, host_of};
use crate::runtime::device::{
    Comp, Compiled, IdentityCompiler, LruAllocator, Prog, TinyElf, register_device,
};
use crate::runtime::ops_interp::InterpProgram;
use crate::runtime::ray_kernel::{RayKernel, SendMemView};
use crate::runtime::ray_typed::TypedKernel;
use crate::uop::UopId;
use crate::uop::ctx::UOpCtx;
use crate::uop::flat::{FlatGraph, parse_flat};
use crate::uop::graph::Graph;

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
        RayonUopRenderer {
            facts: Renderer::rayon(),
        }
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

/// `RayonProgram` — layered: the dtype-native `TypedKernel` (R2), else the
/// `f64` `RayKernel` (R1), else the interpreter. This keeps `RAYON`
/// correct-by-construction while the typed coverage grows.
pub struct RayonProgram {
    typed: Option<TypedKernel>,
    f64_kernel: Option<RayKernel>,
    inner: InterpProgram,
    mode: Mode,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Owned IR, `rayon` over the global launch grid (default).
    Parallel,
    /// Owned IR, serial launches (`RAYON_SERIAL=1`).
    Serial,
    /// Bypass the typed path, use the f64 owned IR (`RAYON_F64=1`).
    F64,
    /// Always the interpreter (`RAYON_INTERP=1`); also the final fallback.
    Interp,
}

fn mode() -> Mode {
    static V: std::sync::OnceLock<Mode> = std::sync::OnceLock::new();
    *V.get_or_init(mode_from_env)
}

fn verbose() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("RAYON_VERBOSE").is_ok())
}

fn mode_from_env() -> Mode {
    if std::env::var("RAYON_INTERP").is_ok() {
        Mode::Interp
    } else if std::env::var("RAYON_SERIAL").is_ok() {
        Mode::Serial
    } else if std::env::var("RAYON_F64").is_ok() {
        Mode::F64
    } else {
        Mode::Parallel
    }
}

/// Observability: programs compiled to an owned kernel (typed or f64),
/// programs using the typed path, and interpreter fallbacks.
pub static OWNED_KERNELS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static TYPED_KERNELS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static FALLBACK_KERNELS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

impl RayonProgram {
    pub fn new(elf: &TinyElf) -> RayonProgram {
        // Decode the flat graph once, then compile the owned IR. The typed path
        // is tried first; the f64 path is the fallback for sub-word dtypes.
        let (typed, f64_kernel) = match decode_lib(elf) {
            Some((mut ctx, uops)) => {
                let typed = TypedKernel::compile(&mut ctx, &uops);
                // Always compile the f64 path too: it backs `RAYON_F64=1` and
                // any runtime case typed can't handle.
                let f64k = RayKernel::compile(&mut ctx, &uops);
                (typed, f64k)
            }
            None => (None, None),
        };
        let owned = typed.is_some() || f64_kernel.is_some();
        if owned {
            OWNED_KERNELS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        } else {
            FALLBACK_KERNELS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        if typed.is_some() {
            TYPED_KERNELS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        if verbose() {
            let kind = if typed.is_some() {
                "typed"
            } else if f64_kernel.is_some() {
                "f64"
            } else {
                "fallback"
            };
            eprintln!("[rayon] program {} -> {kind}", elf.name);
        }
        RayonProgram {
            typed,
            f64_kernel,
            inner: InterpProgram::new(elf),
            mode: mode(),
        }
    }
}

/// Bind signature slots to raw host windows. The `Rc<dyn OpaqueT>`s are
/// borrowed from `bufs`, which the caller keeps alive until `call` returns; the
/// rayon scope always joins before then.
fn bind_views(
    bufs: &[Rc<dyn OpaqueT>],
    globals: &[i64],
    sig: &HashMap<i64, DTypeId>,
) -> HashMap<i64, SendMemView> {
    let mut views: HashMap<i64, SendMemView> = HashMap::new();
    for &slot in globals {
        let o = &bufs[slot as usize];
        let dt = sig[&slot];
        let isz = dt.itemsize();
        let h = host_of(&**o);
        views.insert(slot, SendMemView::from_host(h, isz, h.size / isz));
    }
    views
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
        // the owned executors share that little-endian dim convention.
        if self.mode == Mode::Interp {
            return self.inner.call(bufs, global_size, local_size, vals, wait);
        }
        let use_typed = self.mode != Mode::F64 && self.typed.is_some();
        let (globals, sig): (&[i64], &HashMap<i64, DTypeId>) = if use_typed {
            let t = self.typed.as_ref().unwrap();
            (t.globals(), t.sig_dtypes())
        } else if let Some(k) = &self.f64_kernel {
            (&k.globals, &k.sig_dtypes)
        } else {
            return self.inner.call(bufs, global_size, local_size, vals, wait);
        };
        let st = std::time::Instant::now();
        let views = bind_views(bufs, globals, sig);
        let vals_f: Vec<f64> = vals.iter().map(|&v| v as f64).collect();
        let gs_rev = [global_size[2], global_size[1], global_size[0]];
        let ls_rev = local_size.map(|l| [l[2], l[1], l[0]]);
        let parallel = matches!(self.mode, Mode::Parallel | Mode::F64);
        if use_typed {
            self.typed
                .as_ref()
                .unwrap()
                .exec(&views, gs_rev, ls_rev, &vals_f, parallel);
        } else {
            self.f64_kernel
                .as_ref()
                .unwrap()
                .exec(&views, gs_rev, ls_rev, &vals_f, parallel);
        }
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
