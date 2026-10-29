//! `codegen/opt/search.py` — BEAM search over kernel opts.
//!
//! Explores sequences of `Opt`s (UPCAST/UNROLL/LOCAL/GROUPTOP/GROUP), compiles
//! each candidate through [`to_program`] and **times it on the device** with
//! dummy buffers (`args_from_ast`), keeping the fastest. Enabled by `BEAM=n`
//! (`n` = beam width); off by default. A result is cached on disk
//! (`engine::disk_cache`) so a search runs once per `(kernel, width, device)`.
//! The TC/SWAP/THREAD/PADTO/NOLOCALS actions are not ported yet (see
//! CURRENT_STATE §9.6).

use std::collections::HashMap;
use std::rc::Rc;

use crate::codegen::postrange::{OptOp, Scheduler};
use crate::engine::realize::{CompiledProg, ProgCache, to_program};
use crate::runtime::buffer::OpaqueT;
use crate::runtime::device::{Compiled, TinyElf, device};
use crate::uop::arg::Arg;
use crate::uop::ctx::UOpCtx;
use crate::uop::graph::UopId;
use crate::uop::ops::Ops;
use crate::uop::range::AxisType;

/// One candidate action (`Opt(op, axis, arg)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Action {
    /// Opt to apply.
    pub op: OptOp,
    /// Axis index the opt targets.
    pub axis: usize,
    /// Opt argument, usually a factor.
    pub arg: i64,
}

/// `actions` (search.py:12) — restricted to the ported `OptOp`s.
pub fn actions() -> Vec<Action> {
    let mut a = Vec::new();
    for arg in [0i64, 2, 3, 4, 5, 7] {
        for axis in 0..8 {
            a.push(Action {
                op: OptOp::Upcast,
                axis,
                arg,
            });
        }
    }
    for arg in [0i64, 4, 7] {
        for axis in 0..5 {
            a.push(Action {
                op: OptOp::Unroll,
                axis,
                arg,
            });
        }
    }
    for arg in [2i64, 3, 4, 8, 13, 16, 29] {
        for axis in 0..6 {
            a.push(Action {
                op: OptOp::Local,
                axis,
                arg,
            });
        }
    }
    for arg in [13i64, 16, 28, 29, 32, 49, 64, 256] {
        for axis in 0..3 {
            a.push(Action {
                op: OptOp::GroupTop,
                axis,
                arg,
            });
        }
    }
    for arg in [0i64, 4, 8, 16] {
        for axis in 0..3 {
            a.push(Action {
                op: OptOp::Group,
                axis,
                arg,
            });
        }
    }
    a.push(Action {
        op: OptOp::Local,
        axis: 0,
        arg: 32,
    });
    a.push(Action {
        op: OptOp::Local,
        axis: 6,
        arg: 2,
    });
    a
}

fn env_i64(key: &str, default: i64) -> i64 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// upcast/unroll product and local/group product (search.py:132-136).
fn up_lcl(ctx: &mut UOpCtx, s: &Scheduler) -> (i64, i64) {
    let (mut up, mut lcl) = (1i64, 1i64);
    let shapes = s.full_shape(ctx);
    let types = s.axis_types(ctx);
    for (x, t) in shapes.iter().zip(types.iter()) {
        match t {
            AxisType::Upcast | AxisType::Unroll => up *= x,
            AxisType::Warp | AxisType::Local | AxisType::GroupReduce => lcl *= x,
            _ => {}
        }
    }
    (up, lcl)
}

/// `get_kernel_actions` (search.py:95) — one Scheduler per applicable action.
pub fn get_kernel_actions<'a>(
    ctx: &mut UOpCtx,
    s: &Scheduler<'a>,
    include_0: bool,
    max_up: i64,
    max_lcl: i64,
) -> Vec<Scheduler<'a>> {
    let mut acted: Vec<Scheduler> = Vec::new();
    if include_0 {
        acted.push(s.copy());
    }
    let acts = actions();
    for a in &acts {
        // skip an action whose arg is the current full extent when a "full"
        // (arg=0) version of the same (op, axis) also exists.
        if let Some(ax) = s.real_axis(ctx, a.op, a.axis) {
            let fs = s.full_shape(ctx);
            if ax >= fs.len() {
                continue;
            }
            if fs[ax] == a.arg
                && acts
                    .iter()
                    .any(|b| b.op == a.op && b.axis == a.axis && b.arg == 0)
            {
                continue;
            }
        } else {
            continue;
        }
        let mut s2 = s.copy();
        if s2
            .try_apply_opt(ctx, a.op, Some(a.axis), Some(a.arg))
            .is_err()
        {
            continue;
        }
        let (up, lcl) = up_lcl(ctx, &s2);
        if up > max_up || lcl > max_lcl {
            continue;
        }
        acted.push(s2);
    }
    acted
}

/// `args_from_ast` (postrange.py:330) — one dummy buffer per kernel PARAM slot.
fn rawbufs_from_ast(
    ctx: &mut UOpCtx,
    ast: &UopId,
    dev: &Rc<Compiled>,
) -> HashMap<i64, Rc<dyn OpaqueT>> {
    let mut map: HashMap<i64, Rc<dyn OpaqueT>> = HashMap::new();
    for u in ctx.backward_slice_with_self(ast) {
        if ctx.op_of(&u) != Ops::Param {
            continue;
        }
        let Arg::Param(p) = ctx.arg_of(&u) else {
            continue;
        };
        if p.slot < 0 || map.contains_key(&p.slot) {
            continue;
        }
        let numel: i64 = ctx
            .shape(&u)
            .ok()
            .flatten()
            .map(|s| s.iter().map(|x| x.clone().as_i64().unwrap_or(1)).product())
            .unwrap_or(1);
        let nbytes = (numel.max(1) as usize) * ctx.dt_of(&u).itemsize();
        if env_flag("BEAM_DUMP_BASE") {
            eprintln!(
                "[beam] rawbuf slot={} shape={:?} nbytes={}",
                p.slot,
                ctx.shape(&u).ok().flatten(),
                nbytes
            );
        }
        let buf = dev.allocator.alloc(nbytes);
        // BEAM times candidates with dummy buffers. Upstream leaves them
        // uninitialized (`BufferSpec.zero=False`) and tolerates a faulting
        // candidate via its worker pool; the single-process port has no such
        // isolation, so a candidate that uses a dummy as an *index* input
        // (e.g. the advanced-indexing gather) would read recycled garbage and
        // fault, poisoning the CL context. Zeroing gives benign, deterministic
        // inputs so the timing launch is safe. `BEAM_UNINIT_BUFS=1` restores
        // upstream's uninitialized dummies (only safe with the worker pool).
        if !env_flag("BEAM_UNINIT_BUFS") {
            dev.copyin_to(&*buf, &vec![0u8; nbytes]);
        }
        map.insert(p.slot, buf);
    }
    map
}

/// A candidate that compiled: its scheduler, compiled program, the
/// `Estimates.ops` proxy, and the optimized-AST content key (`seen_libs`).
struct CompiledCand<'a> {
    s: Scheduler<'a>,
    cp: Rc<CompiledProg>,
    ops: i64,
    key: u64,
}

fn env_flag(key: &str) -> bool {
    std::env::var(key).map(|v| v != "0").unwrap_or(false)
}

/// The concrete extent of a `RANGE`/`SPECIAL` (`src[0]`); 1 for symbolic extents
/// (the corpus schedules to integers).
fn range_extent(ctx: &UOpCtx, u: &UopId) -> i64 {
    let s0 = ctx.srcs(u)[0].clone();
    if ctx.op_of(&s0) == Ops::Const
        && let crate::dtype::ConstVal::Int(v) = ctx.val(&s0)
    {
        return i64::try_from(v).unwrap_or(1).max(1);
    }
    1
}

/// `Estimates.from_uops(..., ignore_indexing=True).ops` (renderer/__init__.py:21)
/// over a concrete-int AST: the loop-nest multiplier times the ALU node count,
/// excluding index-expression arithmetic. Used to drop candidates using >1000x
/// the compute of the smallest (search.py:143-146). Approximate on the port (no
/// symbolic extents, WMMA skipped), which is enough for an outlier filter.
fn estimate_ops(ctx: &mut UOpCtx, ast: &UopId) -> i64 {
    let order = ctx.toposort(ast);
    // `ignore_indexing`: drop ALU nodes reachable only through INDEX/SHRINK index
    // operands (src[1:]).
    let mut excluded: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for u in &order {
        if matches!(ctx.op_of(u), Ops::Index | Ops::Shrink) {
            for s in ctx.srcs(u).iter().skip(1) {
                for x in ctx.toposort(s) {
                    excluded.insert(x.id());
                }
            }
        }
    }
    let mut flops: i64 = 0;
    let mut mults: i64 = 1;
    let mut mult_stack: Vec<i64> = Vec::new();
    for u in &order {
        let op = ctx.op_of(u);
        match op {
            Ops::Range => {
                mult_stack.push(mults);
                if ctx.dt_of(u) != crate::dtype::VOID {
                    mults = mults.saturating_mul(range_extent(ctx, u));
                }
            }
            Ops::End => {
                mults = mult_stack.pop().unwrap_or(1);
            }
            Ops::Special => {
                mults = mults.saturating_mul(range_extent(ctx, u));
            }
            o if crate::uop::group::ALU.contains(&o) => {
                if excluded.contains(&u.id()) {
                    continue;
                }
                let m = if o == Ops::Mulacc { 2 } else { 1 };
                flops = flops.saturating_add(
                    mults
                        .saturating_mul(ctx.max_numel(u.clone()))
                        .saturating_mul(m),
                );
            }
            _ => {}
        }
    }
    flops
}

/// `_try_compile` (search.py:58): lower+compile one candidate, applying the
/// `BEAM_UOPS_MAX` cap. A panic from an unsupported candidate is caught and the
/// candidate skipped (python's `except Exception`, unless `BEAM_STRICT_MODE`).
fn compile_candidate<'a>(
    ctx: &mut UOpCtx,
    cand: &Scheduler<'a>,
    dev: &Rc<Compiled>,
    uops_max: i64,
    strict: bool,
) -> Option<CompiledCand<'a>> {
    let run = |ctx: &mut UOpCtx| -> Option<CompiledCand<'a>> {
        let ast = cand.copy().get_optimized_ast(ctx, Some("test"));
        // `len(uops) >= BEAM_UOPS_MAX` — checked on the AST so a huge kernel is
        // rejected before the expensive render/compile.
        if uops_max > 0 && ctx.toposort(&ast).len() as i64 >= uops_max {
            return None;
        }
        let ops = estimate_ops(ctx, &ast);
        let key = ctx.g.content_key(&ast);
        let mut cache = ProgCache::default();
        let cp = to_program(ctx, ast, dev, &mut cache);
        Some(CompiledCand {
            s: cand.copy(),
            cp,
            ops,
            key,
        })
    };
    if strict {
        run(ctx)
    } else {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(ctx)))
            .map_err(|e| {
                if env_flag("BEAM_VERBOSE") {
                    eprintln!("[beam] compile skip: {}", panic_msg(&e));
                }
            })
            .ok()
            .flatten()
    }
}

/// The first line of a caught panic payload (for `BEAM_VERBOSE` diagnostics).
fn panic_msg(e: &Box<dyn std::any::Any + Send>) -> String {
    let s = e
        .downcast_ref::<String>()
        .map(|s| s.as_str())
        .or_else(|| e.downcast_ref::<&str>().copied())
        .unwrap_or("<panic>");
    s.lines().next().unwrap_or("").to_string()
}

/// `_time_program` (search.py:38): launch a compiled candidate with the dummy
/// buffers and return the device time (scaled back up from the capped test size).
fn time_candidate(
    cp: &Rc<CompiledProg>,
    rawbufs: &HashMap<i64, Rc<dyn OpaqueT>>,
    dev: &Rc<Compiled>,
) -> Option<f64> {
    let mut bufs: Vec<Rc<dyn OpaqueT>> = Vec::with_capacity(cp.pi.globals.len());
    for slot in &cp.pi.globals {
        bufs.push(rawbufs.get(slot)?.clone());
    }
    let var_vals: HashMap<String, i64> = HashMap::new();
    let (mut gs, ls) = cp.pi.launch_dims(&var_vals);
    // A zero-dimension launch (`clEnqueueNDRangeKernel` with a 0 work size) can
    // corrupt the queue on some drivers; skip such candidates.
    if gs.contains(&0) {
        return None;
    }
    // `get_test_global_size`: cap the timing launch so a huge candidate can't
    // exhaust the driver; scale the measured time back up.
    let max_global: usize = std::env::var("BEAM_TEST_GLOBAL_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(65536);
    let orig: usize = gs.iter().product();
    while gs.iter().product::<usize>() > max_global {
        let mut shrunk = false;
        for j in (0..gs.len()).rev() {
            if gs[j] > 16 {
                gs[j] /= 2;
                shrunk = true;
                break;
            }
        }
        if !shrunk {
            break;
        }
    }
    let factor = orig as f64 / gs.iter().product::<usize>() as f64;
    let elf = TinyElf {
        lib: cp.elf.lib.clone(),
        name: cp.elf.name.clone(),
        signature: cp.elf.signature.clone(),
    };
    let rt = dev.runtime_nocache(&elf);
    let dur = rt
        .borrow_mut()
        .call(&bufs, gs, ls, &cp.pi.vals(&var_vals), true)?;
    Some(dur * factor)
}

// ---- BEAM result cache (search.py:112-116, 176) ----

fn serialize_opt(op: &Arg) -> Option<String> {
    let Arg::Data(d) = op else { return None };
    if d.tag != "Opt" {
        return None;
    }
    let (Arg::Str(o), Arg::Int(axis), Arg::Int(arg)) = (d.get("op"), d.get("axis"), d.get("arg"))
    else {
        return None;
    };
    Some(format!("{o}:{axis}:{arg}"))
}

fn parse_opt(s: &str) -> Option<(OptOp, usize, i64)> {
    let mut it = s.split(':');
    let op = match it.next()? {
        "UPCAST" => OptOp::Upcast,
        "UNROLL" => OptOp::Unroll,
        "LOCAL" => OptOp::Local,
        "GROUP" => OptOp::Group,
        "GROUPTOP" => OptOp::GroupTop,
        _ => return None,
    };
    Some((op, it.next()?.parse().ok()?, it.next()?.parse().ok()?))
}

fn serialize_opts(opts: &[Arg]) -> String {
    let mut out = String::new();
    for o in opts {
        if let Some(s) = serialize_opt(o) {
            out.push_str(&s);
            out.push('\n');
        }
    }
    out
}

fn parse_opts(s: &str) -> Vec<(OptOp, usize, i64)> {
    s.lines().filter_map(parse_opt).collect()
}

/// `beam_search` (search.py:110). Returns the best Scheduler found.
pub fn beam_search<'a>(ctx: &mut UOpCtx, base: Scheduler<'a>, amt: usize) -> Scheduler<'a> {
    let dev = device(&base.ren.device);
    let rawbufs = rawbufs_from_ast(ctx, &base.ast, &dev);

    // Result cache keyed by (optimized-AST key, width, device) — search.py:112.
    // `IGNORE_BEAM_CACHE` disables both the read and the write.
    let ignore_cache = env_flag("IGNORE_BEAM_CACHE");
    let cache_key = format!(
        "{:016x}:{}:{}",
        ctx.g.content_key(&base.ast),
        amt,
        base.ren.device
    );
    if env_flag("BEAM_VERBOSE") {
        let name = match ctx.arg_of(&base.ast) {
            Arg::Data(d) => match d.get("name") {
                Arg::Str(s) => s.clone(),
                _ => "?".to_string(),
            },
            _ => "?".to_string(),
        };
        eprintln!("[beam] search kernel {name} ({})", &cache_key[..16]);
    }
    if env_flag("BEAM_DUMP_BASE") {
        eprintln!(
            "[beam] BASE {} shape={:?} types={:?} opts={:?}",
            &cache_key[..16],
            base.full_shape(ctx),
            base.axis_types(ctx),
            base.applied_opts()
        );
        crate::codegen::postrange::dump_index_exprs(ctx, base.ast.clone(), "BASE");
    }
    if !ignore_cache && let Some(v) = crate::engine::disk_cache::get("beam_search", &cache_key) {
        let mut ret = base.copy();
        let skip = base.applied_opts().len();
        for (i, (op, axis, arg)) in parse_opts(&v).into_iter().enumerate() {
            if i < skip {
                continue;
            }
            let _ = ret.try_apply_opt(ctx, op, Some(axis), Some(arg));
        }
        return ret;
    }

    // Candidates the lowerer can't handle panic; `_try_compile` catches these,
    // so suppress the (expected) panic messages during the search.
    let strict = env_flag("BEAM_STRICT_MODE");
    let quiet = !strict && !env_flag("BEAM_VERBOSE");
    let prev_hook = std::panic::take_hook();
    if quiet {
        std::panic::set_hook(Box::new(|_| {}));
    }

    let max_up = env_i64("BEAM_UPCAST_MAX", 256);
    let max_lcl = env_i64("BEAM_LOCAL_MAX", 1024);
    let min_progress = env_i64("BEAM_MIN_PROGRESS", 1) as f64 / 1e6;
    let uops_max = env_i64("BEAM_UOPS_MAX", 3000);
    // Port-only safety cap on candidates compiled per round. tinygrad isolates
    // candidate compiles in a worker pool, so a misbehaving driver can't take
    // down the training process; the single-process port bounds the stress
    // instead (0 = unlimited, matching tinygrad).
    let max_candidates = env_i64("BEAM_MAX_CANDIDATES", 0) as usize;

    let mut beam: Vec<(Scheduler, f64)> = vec![(base.copy(), f64::INFINITY)];
    let mut seen_libs: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let start = std::time::Instant::now();
    let mut rounds = 0usize;
    loop {
        let mut candidates: Vec<Scheduler> = Vec::new();
        for (si, _) in &beam {
            candidates.extend(get_kernel_actions(ctx, si, false, max_up, max_lcl));
        }
        if max_candidates > 0 && candidates.len() > max_candidates {
            candidates.truncate(max_candidates);
        }
        // compile (with uops cap), then dedup + compute-filter, then time.
        let mut compiled: Vec<CompiledCand> = Vec::new();
        for cand in &candidates {
            if let Some(c) = compile_candidate(ctx, cand, &dev, uops_max, strict) {
                compiled.push(c);
            }
        }
        let mut timed: Vec<(Scheduler, f64)> = Vec::new();
        let mut least_compute_ops: i64 = i64::MAX;
        for c in compiled {
            if seen_libs.contains(&c.key) {
                continue;
            }
            least_compute_ops = least_compute_ops.min(c.ops);
            if least_compute_ops > 0 && least_compute_ops.saturating_mul(1000) < c.ops {
                continue;
            }
            seen_libs.insert(c.key);
            // `_try_compile` catches a candidate's compile/launch failure and
            // skips it; without this a single invalid CL candidate would abort
            // the whole search.
            let timed_c = if strict {
                time_candidate(&c.cp, &rawbufs, &dev)
            } else {
                let src = String::from_utf8_lossy(&c.cp.elf.lib).to_string();
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    time_candidate(&c.cp, &rawbufs, &dev)
                }))
                .map_err(|e| {
                    if env_flag("BEAM_VERBOSE") {
                        eprintln!(
                            "[beam] time skip: {} applied={:?}",
                            panic_msg(&e),
                            c.s.applied_opts()
                        );
                        if env_flag("BEAM_DUMP_SRC") {
                            for l in src.lines().take(400) {
                                eprintln!("    | {l}");
                            }
                        }
                    }
                })
                .ok()
                .flatten()
            };
            if let Some(t) = timed_c
                && t.is_finite()
            {
                timed.push((c.s, t));
            }
        }
        timed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let exiting = timed.is_empty()
            || timed[0].1 < min_progress
            || (!beam.is_empty() && (beam[0].1 - timed[0].1) < min_progress);
        if !exiting {
            beam = timed.into_iter().take(amt).collect();
        } else if !timed.is_empty() && timed[0].1 < beam[0].1 {
            beam = timed.into_iter().take(1).collect();
        }
        rounds += 1;
        if std::env::var("BEAM_DEBUG").is_ok() {
            eprintln!(
                "[beam] round {rounds}: {:.2}s best={:.6}s applied={:?}",
                start.elapsed().as_secs_f64(),
                beam[0].1,
                beam[0].0.applied_opts()
            );
        }
        if exiting || rounds > 64 {
            break;
        }
    }
    if quiet {
        std::panic::set_hook(prev_hook);
    }

    if !ignore_cache {
        crate::engine::disk_cache::put(
            "beam_search",
            &cache_key,
            &serialize_opts(beam[0].0.applied_opts()),
        );
    }
    beam[0].0.copy()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uop::arg::DataArg;

    fn opt_arg(op: &str, axis: i64, arg: i64) -> Arg {
        Arg::Data(DataArg {
            tag: "Opt".into(),
            fields: vec![
                ("op".into(), Arg::Str(op.into())),
                ("axis".into(), Arg::Int(axis)),
                ("arg".into(), Arg::Int(arg)),
            ],
        })
    }

    /// The disk cache stores/applies `applied_opts`; the wire format must survive
    /// a round-trip with the *relative* axis (tinygrad records the original opt).
    #[test]
    fn opt_cache_roundtrip() {
        let opts = vec![
            opt_arg("UPCAST", 2, 4),
            opt_arg("UNROLL", 1, 0),
            opt_arg("LOCAL", 0, 32),
            opt_arg("GROUPTOP", 1, 16),
        ];
        let s = serialize_opts(&opts);
        assert_eq!(
            parse_opts(&s),
            vec![
                (OptOp::Upcast, 2, 4),
                (OptOp::Unroll, 1, 0),
                (OptOp::Local, 0, 32),
                (OptOp::GroupTop, 1, 16),
            ]
        );
    }
}
