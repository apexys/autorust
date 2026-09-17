//! `engine/realize.py` + `codegen/__init__.py` to_program (M6).
//!
//! `to_program` lowers a kernel AST (SINK with KernelInfo) into a PROGRAM uop:
//! full_rewrite_to_sink → ProgramInfo::from_sink → linearize → SOURCE →
//! BINARY, cached per AST (python's to_program_cache). `compile_linear`
//! swaps PROGRAMs into the LINEAR's CALLs; `run_linear` executes every CALL
//! through the device (exec_copy / exec_kernel).

use std::collections::HashMap;
use std::rc::Rc;

use crate::codegen::linearize::{linearize, pm_linearize_cleanups};
use crate::codegen::pipeline::full_rewrite_to_sink_with;
use crate::renderer::cstyle::to_function_name_pub;
use crate::runtime::buffer::{buffer_of, BufTable, OpaqueView};
use crate::runtime::device::{device, TinyElf};
use crate::uop::arg::{Arg, DataArg};
use crate::uop::ctx::UOpCtx;
use crate::uop::graph::UopId;
use crate::uop::ops::Ops;
use crate::uop::sint::SVal;

// ---------------- ProgramInfo ----------------

/// `ProgramInfo` (ops.py:1215) — the PROGRAM arg dataclass.
#[derive(Clone, Debug, Default)]
pub struct ProgramInfo {
    pub name: String,
    pub global_size: Vec<i64>,
    pub local_size: Option<Vec<i64>>,
    pub vars: Vec<i64>,
    pub globals: Vec<i64>,
    pub outs: Vec<i64>,
    pub ins: Vec<i64>,
}

fn seq(items: Vec<i64>) -> Arg {
    Arg::Seq { tuple: true, items: items.into_iter().map(Arg::Int).collect() }
}

fn seq_of(arg: &Arg) -> Vec<i64> {
    match arg {
        Arg::Seq { items, .. } => items.iter().filter_map(|a| a.as_int()).collect(),
        other => panic!("TGERR: ProgramInfo seq field {other:?}"),
    }
}

impl ProgramInfo {
    /// Encode as a `DataArg` (the PROGRAM uop arg).
    pub fn to_arg(&self) -> Arg {
        Arg::Data(DataArg {
            tag: "ProgramInfo".into(),
            fields: vec![
                ("name".into(), Arg::Str(self.name.clone())),
                ("global_size".into(), seq(self.global_size.clone())),
                (
                    "local_size".into(),
                    match &self.local_size {
                        Some(l) => seq(l.clone()),
                        None => Arg::None,
                    },
                ),
                ("vars".into(), seq(self.vars.clone())),
                ("globals".into(), seq(self.globals.clone())),
                ("outs".into(), seq(self.outs.clone())),
                ("ins".into(), seq(self.ins.clone())),
            ],
        })
    }

    pub fn from_arg(arg: &Arg) -> ProgramInfo {
        let Arg::Data(d) = arg else { panic!("TGERR: PROGRAM arg {arg:?}") };
        assert!(d.tag == "ProgramInfo", "TGERR: PROGRAM arg tag {}", d.tag);
        ProgramInfo {
            name: d.get("name").as_str().unwrap_or("test").to_string(),
            global_size: seq_of(d.get("global_size")),
            local_size: match d.get("local_size") {
                Arg::None => None,
                a => Some(seq_of(a)),
            },
            vars: seq_of(d.get("vars")),
            globals: seq_of(d.get("globals")),
            outs: seq_of(d.get("outs")),
            ins: seq_of(d.get("ins")),
        }
    }

    /// `ProgramInfo.launch_dims` (ops.py:1231) — corpus dims are concrete.
    pub fn launch_dims(&self, _var_vals: &HashMap<String, i64>) -> ([usize; 3], Option<[usize; 3]>) {
        let mut gs = [1usize; 3];
        for (i, v) in self.global_size.iter().enumerate().take(3) {
            gs[i] = *v as usize;
        }
        let ls = self.local_size.as_ref().map(|l| {
            let mut out = [1usize; 3];
            for (i, v) in l.iter().enumerate().take(3) {
                out[i] = *v as usize;
            }
            out
        });
        (gs, ls)
    }

    /// `ProgramInfo.vals` (ops.py:1236) — ALU param bindings; corpus has no
    /// Variables, so this is empty.
    pub fn vals(&self, _var_vals: &HashMap<String, i64>) -> Vec<i64> {
        Vec::new()
    }
}

/// Walk a LOAD/STORE src[0] chain (INDEX/SHRINK/CAST/BITCAST) down to the
/// buffer PARAM uop (python `idx.src[0].buf_uop`).
fn buf_param_of(ctx: &mut UOpCtx, mut idx: UopId) -> Option<UopId> {
    while matches!(ctx.op_of(idx), Ops::Index | Ops::Shrink | Ops::Cast | Ops::Bitcast) {
        idx = ctx.srcs(idx)[0];
    }
    if matches!(ctx.op_of(idx), Ops::Param | Ops::Buffer) {
        Some(idx)
    } else {
        None
    }
}

/// `ProgramInfo.from_sink` (ops.py:1239).
pub fn program_info_from_sink(ctx: &mut UOpCtx, sink: UopId) -> ProgramInfo {
    let mut vars: Vec<i64> = Vec::new();
    let mut globals: Vec<i64> = Vec::new();
    let mut outs: Vec<i64> = Vec::new();
    let mut ins: Vec<i64> = Vec::new();
    let mut global_size: Vec<i64> = vec![1, 1, 1];
    let mut local_size: Option<Vec<i64>> = Some(vec![1, 1, 1]);
    let name = match ctx.arg_of(sink) {
        Arg::Data(d) if d.tag == "KernelInfo" => d.get("name").as_str().unwrap_or("test").to_string(),
        _ => "test".to_string(),
    };
    for &u in ctx.toposort(sink).iter() {
        match ctx.op_of(u) {
            Ops::Param => {
                let Arg::Param(p) = ctx.arg_of(u) else { continue };
                if p.addrspace == Some(crate::dtype::AddrSpace::Alu) {
                    if !vars.contains(&p.slot) {
                        vars.push(p.slot);
                    }
                } else if !globals.contains(&p.slot) {
                    globals.push(p.slot);
                }
            }
            Ops::Store | Ops::Load => {
                let s0 = ctx.srcs(u)[0];
                if let Some(buf) = buf_param_of(ctx, s0) {
                    if let Arg::Param(p) = ctx.arg_of(buf) {
                        if p.addrspace != Some(crate::dtype::AddrSpace::Alu) {
                            let list = if ctx.op_of(u) == Ops::Store { &mut outs } else { &mut ins };
                            if !list.contains(&p.slot) {
                                list.push(p.slot);
                            }
                        }
                    }
                }
            }
            Ops::Special => {
                let Arg::Str(s) = ctx.arg_of(u) else { continue };
                let which = s.as_bytes()[0];
                if which == b'i' {
                    local_size = None;
                }
                let dim: usize = s[s.len() - 1..].parse().unwrap_or(0);
                let ext = match ctx.ssimplify(ctx.srcs(u)[0]) {
                    Ok(SVal::I(v)) => v as i64,
                    other => panic!("TGERR: from_sink SPECIAL {other:?}"),
                };
                if which == b'l' {
                    if let Some(l) = &mut local_size {
                        l[dim] = ext;
                    }
                } else if which == b'g' {
                    global_size[dim] = ext;
                }
            }
            _ => {}
        }
    }
    vars.sort_unstable();
    globals.sort_unstable();
    globals.dedup();
    outs.sort_unstable();
    outs.dedup();
    ins.sort_unstable();
    ins.dedup();
    ProgramInfo { name, global_size, local_size, vars, globals, outs, ins }
}

// ---------------- to_program ----------------

/// A lowered+compiled kernel: the PROGRAM uop, its ProgramInfo, and the elf.
pub struct CompiledProg {
    pub uop: UopId,
    pub pi: ProgramInfo,
    pub elf: Rc<TinyElf>,
}

/// `to_program_cache` — the JIT cache. Keyed by a **content hash of the AST
/// DAG** (`Graph::content_key`, mirroring tinygrad `ast.key`), so compiled
/// programs are reused across arena resets / graph rebuilds (we rebuild the
/// front-end graph every realize but skip re-lowering/re-rendering/re-compiling
/// on a hit). Stores **compiled code only** (`Rc<CompiledProg>`: source text,
/// `Rc<TinyElf>`, `ProgramInfo`), never device buffers, so it can't hold VRAM.
/// Optionally LRU-capped so it can't grow unbounded on dynamic-shape workloads.
pub struct ProgCache {
    map: HashMap<u64, Rc<CompiledProg>>,
    order: std::collections::VecDeque<u64>,
    cap: usize,
    /// Observability: number of `to_program` lookups that found a cached program.
    pub hits: u64,
    /// Observability: number of `to_program` lookups that had to lower+compile.
    pub misses: u64,
}

impl Default for ProgCache {
    fn default() -> Self {
        ProgCache::new()
    }
}

impl ProgCache {
    /// Unbounded cache (no eviction).
    pub fn new() -> Self {
        ProgCache { map: HashMap::new(), order: std::collections::VecDeque::new(), cap: usize::MAX, hits: 0, misses: 0 }
    }
    /// LRU-capped cache: once more than `cap` entries are live, the least-recently
    /// used is dropped.
    pub fn capped(cap: usize) -> Self {
        ProgCache { map: HashMap::new(), order: std::collections::VecDeque::new(), cap, hits: 0, misses: 0 }
    }
    fn touch(&mut self, key: u64) {
        if let Some(pos) = self.order.iter().position(|&k| k == key) {
            self.order.remove(pos);
        }
        self.order.push_back(key);
    }
    pub fn get(&mut self, key: u64) -> Option<Rc<CompiledProg>> {
        let p = self.map.get(&key).cloned();
        if p.is_some() {
            self.hits += 1;
            self.touch(key);
        } else {
            self.misses += 1;
        }
        p
    }
    /// Number of cached programs currently resident.
    pub fn len(&self) -> usize {
        self.map.len()
    }
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
    pub fn insert(&mut self, key: u64, cp: Rc<CompiledProg>) {
        if !self.map.contains_key(&key) {
            self.order.push_back(key);
        } else {
            self.touch(key);
        }
        self.map.insert(key, cp);
        while self.map.len() > self.cap {
            if let Some(old) = self.order.pop_front() {
                self.map.remove(&old);
            } else {
                break;
            }
        }
    }
}

/// `to_program` (codegen/__init__.py:500) — cached do_to_program. The device
/// supplies its renderer (facts + render) and compiler through the public
/// backend seams.
pub fn to_program(ctx: &mut UOpCtx, ast: UopId, dev: &Rc<crate::runtime::device::Compiled>, cache: &mut ProgCache) -> Rc<CompiledProg> {
    let key = ctx.g.content_key(ast);
    if let Some(p) = cache.get(key) {
        // The cached program may have been compiled in a different arena (the key
        // is content-based by design), so its `uop` id is not valid here. The
        // expensive artifacts (`elf`, `pi`) are arena-independent; re-intern a
        // fresh PROGRAM node in this ctx and reuse them (no re-lower/re-render/
        // re-compile).
        let prg = ctx.uop(Ops::Program, Some(crate::dtype::VOID), vec![], p.pi.to_arg());
        return Rc::new(CompiledProg { uop: prg, pi: p.pi.clone(), elf: p.elf.clone() });
    }
    // 1. full_rewrite_to_sink
    let sunk = full_rewrite_to_sink_with(ctx, ast, dev.renderer.facts());
    if std::env::var("TG_DUMP_IDX").is_ok() {
        crate::codegen::postrange::dump_index_exprs(ctx, sunk, "SUNK");
    }
    // 2. ProgramInfo.from_sink
    let pi = program_info_from_sink(ctx, sunk);
    // 3. do_linearize: linearize + pm_linearize_cleanups
    let lst = linearize(ctx, sunk);
    if std::env::var("TG_DUMP_IDX").is_ok() {
        crate::codegen::postrange::dump_lst_index_offsets(ctx, &lst, "LINEAR");
    }
    let lst = pm_linearize_cleanups(ctx, lst);
    if std::env::var("TG_DUMP_IDX").is_ok() {
        crate::codegen::postrange::dump_lst_index_offsets(ctx, &lst, "CLEAN");
    }
    // 4. do_render: SOURCE through the device's renderer seam
    let src_text = dev.renderer.render(ctx, &lst);
    if std::env::var("TG_DUMP_LST").is_ok() {
        let pname = crate::codegen::postrange::kernel_name(ctx, &lst);
        eprintln!("LST### BEGIN name={}", pname);
        for &o in &lst {
            let (op, _dt, so, arg, tag) = ctx.snapshot(o);
            let tag_s = match &tag {
                crate::uop::arg::Arg::Int(i) => format!(" tag={}", i),
                _ => String::new(),
            };
            let srcs: Vec<String> = so.iter().map(|s| s.0.to_string()).collect();
            eprintln!("LST {} {:?}({}){} arg={:?}", o.0, op, srcs.join(","), tag_s, arg);
        }
        eprintln!("LST### END name={}", pname);
    }
    // 5. do_compile: through the device's compiler seam
    let lib = dev.compiler.compile_cached(&src_text);
    let linear_uop = ctx.uop(Ops::Linear, Some(crate::dtype::VOID), lst.clone(), Arg::None);
    let source_uop = ctx.uop(Ops::Source, Some(crate::dtype::VOID), vec![], Arg::Str(src_text));
    let binary_uop = ctx.uop(Ops::Binary, Some(crate::dtype::VOID), vec![source_uop], Arg::Bytes(lib.clone()));
    let prg = ctx.uop(Ops::Program, Some(crate::dtype::VOID), vec![sunk, linear_uop, source_uop, binary_uop], pi.to_arg());
    // to_elf signature: non-ALU PARAMs (linear order)
    let mut signature: Vec<(i64, crate::dtype::DTypeId)> = Vec::new();
    for &u in &lst {
        if ctx.op_of(u) == Ops::Param {
            if let Arg::Param(p) = ctx.arg_of(u) {
                if p.addrspace != Some(crate::dtype::AddrSpace::Alu) {
                    signature.push((p.slot, ctx.dt_of(u)));
                }
            }
        }
    }
    let elf = Rc::new(TinyElf { lib, name: to_function_name_pub(&pi.name), signature });
    let cp = Rc::new(CompiledProg { uop: prg, pi, elf });
    cache.insert(key, cp.clone());
    cp
}

// ---------------- compile_linear + run_linear ----------------

/// `lower_and_compile` (realize.py:263) — swap compiled PROGRAMs into every
/// SINK-ast CALL of the linear. Returns the new linear plus per-call
/// CompiledProgs (keyed by the new CALL uop).
pub fn compile_linear(ctx: &mut UOpCtx, linear: UopId, cache: &mut ProgCache) -> (UopId, HashMap<u32, Rc<CompiledProg>>) {
    let calls = ctx.srcs(linear).to_vec();
    let mut replaced = false;
    let mut new_calls: Vec<UopId> = Vec::with_capacity(calls.len());
    let mut progs: HashMap<u32, Rc<CompiledProg>> = HashMap::new();
    for &c in &calls {
        let ast = ctx.srcs(c)[0];
        if ctx.op_of(ast) == Ops::Sink {
            // device from the first global arg's buffer
            let first_arg = ctx.srcs(c)[1];
            let dev_name = buffer_of(ctx, &mut BufTable::default(), first_arg).device().to_string();
            let dev = device(&dev_name);
            let cp = to_program(ctx, ast, &dev, cache);
            let mut new_srcs = vec![cp.uop];
            new_srcs.extend(ctx.srcs(c)[1..].iter().copied());
            let new_call = ctx.uop(Ops::Call, Some(ctx.dt_of(c)), new_srcs, ctx.arg_of(c).clone());
            progs.insert(new_call.0, cp);
            new_calls.push(new_call);
            replaced = true;
        } else {
            new_calls.push(c);
        }
    }
    let lin = if replaced {
        ctx.uop(Ops::Linear, Some(crate::dtype::VOID), new_calls, Arg::None)
    } else {
        linear
    };
    (lin, progs)
}

/// `resolve_params` (realize.py:157) for the non-jit path: call args pass
/// through (PARAM slots only appear in jit re-runs).
fn resolve_params(ctx: &UOpCtx, call: UopId) -> Vec<UopId> {
    ctx.srcs(call)[1..].to_vec()
}

/// `exec_copy` (realize.py:168) — dest/src engine buffers; copies through
/// the backend allocators (host bounce for cross-device/CL, shared host
/// windows memcpy directly).
fn exec_copy(ctx: &mut UOpCtx, t: &mut BufTable, call: UopId) {
    let resolved = resolve_params(ctx, call);
    assert!(resolved.len() >= 2, "TGERR: copy call args");
    let dest = buffer_of(ctx, t, resolved[0]);
    let src = buffer_of(ctx, t, resolved[1]);
    let dest_dev = device(dest.device());
    let src_dev = device(src.device());
    dest.ensure_allocated(&dest_dev);
    src.ensure_allocated(&src_dev);
    let d = dest.get_buf(&dest_dev);
    let s = src.get_buf(&src_dev);
    let mut tmp = vec![0u8; s.size()];
    src_dev.allocator._copyout(&mut tmp, &*s);
    dest_dev.allocator._copyin(&*d, &tmp);
}

/// `exec_kernel` (realize.py:180) — launch the PROGRAM through the device.
fn exec_kernel(ctx: &mut UOpCtx, t: &mut BufTable, call: UopId, cp: &Rc<CompiledProg>, var_vals: &HashMap<String, i64>) {
    let resolved = resolve_params(ctx, call);
    // device from the first global arg (python: to_tuple(call.src[1].device))
    let g0 = *cp.pi.globals.first().unwrap_or(&0) as usize;
    let dev_name = buffer_of(ctx, t, resolved[g0]).device().to_string();
    let dev = device(&dev_name);
    let mut bufs: Vec<OpaqueView> = Vec::new();
    for &g in &cp.pi.globals {
        let b = buffer_of(ctx, t, resolved[g as usize]);
        b.ensure_allocated(&dev);
        bufs.push(b.get_buf(&dev));
    }
    let rt = dev.runtime(TinyElf {
        lib: cp.elf.lib.clone(),
        name: cp.elf.name.clone(),
        signature: cp.elf.signature.clone(),
    });
    let (gs, ls) = cp.pi.launch_dims(var_vals);
    rt.borrow_mut().call(&bufs, gs, ls, &cp.pi.vals(var_vals), false);
}

/// `run_linear` (realize.py:315) — compile then execute every call.
pub fn run_linear(ctx: &mut UOpCtx, linear: UopId, t: &mut BufTable, var_vals: &HashMap<String, i64>, cache: &mut ProgCache) {
    let (compiled, progs) = compile_linear(ctx, linear, cache);
    let calls = ctx.srcs(compiled).to_vec();
    for &call in &calls {
        let ast = ctx.srcs(call)[0];
        match ctx.op_of(ast) {
            Ops::Copy => exec_copy(ctx, t, call),
            Ops::Program => {
                let cp = progs.get(&call.0).expect("TGERR: run_linear missing CompiledProg").clone();
                exec_kernel(ctx, t, call, &cp, var_vals);
            }
            other => panic!("TGERR: run_linear call ast op {other:?}"),
        }
    }
}

// ---------------- Tensor-level realize ----------------

/// Schedule + run one tensor (python `Tensor.realize` path). Returns the
/// tensor's realized base buffer (the held buffer the kernel wrote).
pub fn realize_tensor(ctx: &mut UOpCtx, tensor_uop: UopId, t: &mut BufTable, cache: &mut ProgCache) -> crate::runtime::buffer::Buffer {
    // python `_buffer()`: a device-less (virtual) tensor is made real via
    // contiguous + clone to the default device before realizing.
    let mut target = tensor_uop;
    let is_view = crate::uop::group::in_(crate::uop::group::MOVEMENT, ctx.op_of(target));
    if crate::schedule::indexing::device_of(ctx, target).is_none()
        || crate::dtype::is_weak(ctx.dt_of(target))
        || is_view
    {
        let dev = crate::schedule::indexing::device_of(ctx, target)
            .map(|d| match d {
                crate::uop::arg::DeviceSpec::Single(s) => s,
                crate::uop::arg::DeviceSpec::Multi(v) => v[0].clone(),
            })
            .unwrap_or_else(|| "PYTHON".to_string());
        let cont = ctx.uop(Ops::Contiguous, Some(ctx.dt_of(target)), vec![target], Arg::None);
        target = crate::tensor::creation::copy_to_device(ctx, cont, &dev);
    }
    let sink = crate::tensor::sink(ctx, vec![target]);
    let (call, buffer_map) = crate::tensor::callify::transform_to_call(ctx, sink);
    let (linear, planned, _held) = crate::schedule::sched::create_linear_with_vars(ctx, call);
    let _ = linear;
    run_linear(ctx, planned, t, &HashMap::new(), cache);
    // python `_apply_map_to_tensors`: the tensor's uop becomes its final
    // storage form after the schedule — that is the buffer the kernel wrote.
    let final_uop = buffer_map
        .iter()
        .find(|(a, _)| *a == target)
        .map(|(_, v)| *v)
        .unwrap_or(target);
    buffer_of(ctx, t, final_uop)
}

/// Read a realized buffer's flat values (python `_buffer().as_memoryview()`).
pub fn buffer_values(b: &crate::runtime::buffer::Buffer) -> Vec<f64> {
    let dev = device(b.device());
    b.ensure_allocated(&dev);
    let o = b.get_buf(&dev);
    let bytes = dev.copyout(&*o);
    let isz = b.dtype.itemsize();
    (0..b.size)
        .map(|i| crate::runtime::interp::from_storage_scalar(&bytes[i * isz..(i + 1) * isz], b.dtype))
        .collect()
}

/// Read a realized buffer's raw storage bytes at its own dtype (python
/// `_buffer()` copyout).
pub fn buffer_bytes(b: &crate::runtime::buffer::Buffer) -> Vec<u8> {
    let dev = device(b.device());
    b.ensure_allocated(&dev);
    let o = b.get_buf(&dev);
    dev.copyout(&*o)
}

/// Read a realized tensor's flat values (python `_buffer().as_memoryview()`).
pub fn tensor_values(ctx: &mut UOpCtx, tensor_uop: UopId, t: &mut BufTable, cache: &mut ProgCache) -> Vec<f64> {
    buffer_values(&realize_tensor(ctx, tensor_uop, t, cache))
}

/// Read a realized tensor's raw storage bytes at its own dtype (python
/// `_buffer()` copyout). `Tensor::to_bytes`/`to_list` build on this so typed
/// read-out never round-trips through `f64`.
pub fn tensor_bytes(ctx: &mut UOpCtx, tensor_uop: UopId, t: &mut BufTable, cache: &mut ProgCache) -> Vec<u8> {
    buffer_bytes(&realize_tensor(ctx, tensor_uop, t, cache))
}
