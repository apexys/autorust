//! `runtime/ops_python.py` → `ops_interp` — the built-in interp backend.
//!
//! `InterpUopRenderer` renders the linear uops as the flat JSON wire format
//! (our pickle: the interp "binary" is the serialized graph, exactly like
//! python's `PythonRenderer.render` = base64(pickle(uops))). `InterpProgram`
//! decodes the elf lib back into the uop list and runs the M5 interpreter
//! over host byte windows.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::dtype::DTypeId;
use crate::renderer::{Renderer, UopRenderer};
use crate::runtime::device::{register_device, Prog, TinyElf};
use crate::runtime::interp::{exec_kernel_with_dims, MemView};
use crate::runtime::buffer::{host_of, OpaqueT};
use crate::uop::ctx::UOpCtx;
use crate::uop::flat::FlatGraph;
use crate::uop::graph::Graph;
use crate::uop::UopId;

use super::device::Compiled;

/// `PythonRenderer` — flat-JSON render of the linear uop list.
pub struct InterpUopRenderer {
    facts: Renderer,
}

impl Default for InterpUopRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl InterpUopRenderer {
    pub fn new() -> InterpUopRenderer {
        InterpUopRenderer { facts: Renderer::cpu_interp() }
    }
}

impl UopRenderer for InterpUopRenderer {
    fn facts(&self) -> &Renderer {
        &self.facts
    }

    /// `PythonRenderer.render` — the linear list serialized as flat JSON.
    fn render(&self, ctx: &mut UOpCtx, lst: &[UopId]) -> String {
        let saved_roots = ctx.g.roots.clone();
        ctx.g.roots = lst.to_vec();
        let flat = ctx.g.to_flat();
        ctx.g.roots = saved_roots;
        crate::uop::flat::to_string_flat(&flat)
    }
}

/// `PythonProgram`: decode the linear uops from the elf lib and execute them
/// with the M5 uops interpreter.
pub struct InterpProgram {
    ctx: UOpCtx,
    uops: Vec<UopId>,
    globals: Vec<i64>,
    sig_dtypes: HashMap<i64, DTypeId>,
}

impl InterpProgram {
    pub fn new(elf: &TinyElf) -> InterpProgram {
        let src = std::str::from_utf8(&elf.lib).expect("TGERR: interp elf lib not utf8");
        let fg: FlatGraph = crate::uop::flat::parse_flat(src).expect("TGERR: interp elf lib parse");
        let g = Graph::from_flat(&fg).expect("TGERR: interp elf lib graph");
        let uops = g.roots.clone();
        let n = fg.nodes.len();
        let mut ctx = UOpCtx::new();
        ctx.grow_to(n);
        ctx.g = g;
        let globals: Vec<i64> = elf.signature.iter().map(|(s, _)| *s).collect();
        let sig_dtypes: HashMap<i64, DTypeId> = elf.signature.iter().cloned().collect();
        InterpProgram { ctx, uops, globals, sig_dtypes }
    }
}

impl Prog for InterpProgram {
    fn call(
        &mut self,
        bufs: &[Rc<dyn OpaqueT>],
        global_size: [usize; 3],
        local_size: Option<[usize; 3]>,
        vals: &[i64],
        wait: bool,
    ) -> Option<f64> {
        let st = std::time::Instant::now();
        // bind the signature slots to the passed buffer windows (bufs is
        // slot-indexed, matching tinygrad's `bufs[slot]`)
        let mut views: HashMap<i64, MemView> = HashMap::new();
        for &slot in &self.globals {
            let o = &bufs[slot as usize];
            let dt = self.sig_dtypes[&slot];
            let isz = dt.itemsize();
            let h = host_of(&**o);
            views.insert(slot, MemView(h.mem.clone(), h.offset, isz, h.size / isz));
        }
        let vals_f: Vec<f64> = vals.iter().map(|&v| v as f64).collect();
        // python's PythonProgram iterates `global_size[::-1]` / `local_size[::-1]`;
        // the M5 interpreter shares that little-endian dim convention.
        let gs_rev = [global_size[2], global_size[1], global_size[0]];
        let ls_rev = local_size.map(|l| [l[2], l[1], l[0]]);
        exec_kernel_with_dims(&mut self.ctx, &self.uops, &views, &vals_f, gs_rev, ls_rev);
        if wait {
            Some(st.elapsed().as_secs_f64())
        } else {
            None
        }
    }
}

/// The `PythonDevice` composition + `Device._DeviceMap` assignment ("CPU"
/// and "INTERP" both run the interp backend). Unknown device names fall back
/// to an interp device under that name (e.g. a staging name like "PYTHON"):
/// only the interpreter and the optional OpenCL backend are ported.
pub fn register() {
    fn make(device: &str) -> Rc<Compiled> {
        let renderer: Box<dyn UopRenderer> = Box::new(InterpUopRenderer::new());
        let compiler: Rc<dyn crate::runtime::device::Comp> = Rc::new(crate::runtime::device::IdentityCompiler::default());
        let ctor: Rc<dyn Fn(&TinyElf) -> Rc<RefCell<dyn Prog>>> = Rc::new(|elf| Rc::new(RefCell::new(InterpProgram::new(elf))));
        Compiled::new(device, renderer, crate::runtime::device::LruAllocator::new(), compiler, ctor)
    }
    register_device("CPU", Rc::new(|| make("CPU")));
    register_device("INTERP", Rc::new(|| make("INTERP")));
}

/// Build an interp device under an arbitrary (unregistered) name — the
/// registry fallback for staging devices.
pub fn fallback(device: &str) -> Rc<Compiled> {
    register();
    let renderer: Box<dyn UopRenderer> = Box::new(InterpUopRenderer::new());
    let compiler: Rc<dyn crate::runtime::device::Comp> = Rc::new(crate::runtime::device::IdentityCompiler::default());
    let ctor: Rc<dyn Fn(&TinyElf) -> Rc<RefCell<dyn Prog>>> = Rc::new(|elf| Rc::new(RefCell::new(InterpProgram::new(elf))));
    Compiled::new(device, renderer, crate::runtime::device::LruAllocator::new(), compiler, ctor)
}
