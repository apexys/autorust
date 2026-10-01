//! `codegen/opt/search.py` — BEAM search over kernel opts.
//!
//! Explores sequences of `Opt`s (UPCAST/UNROLL/LOCAL/GROUPTOP/GROUP), compiles
//! each candidate through [`to_program`] and **times it on the device** with
//! dummy buffers (`args_from_ast`), keeping the fastest. Enabled by `BEAM=n`
//! (`n` = beam width); off by default. The TC/SWAP/THREAD/PADTO/NOLOCALS actions
//! and the disk cache are not ported yet (see CURRENT_STATE §9.6).

use std::collections::HashMap;
use std::rc::Rc;

use crate::codegen::postrange::{OptOp, Scheduler};
use crate::engine::realize::{to_program, ProgCache};
use crate::runtime::buffer::OpaqueT;
use crate::runtime::device::{device, Compiled, TinyElf};
use crate::uop::arg::Arg;
use crate::uop::ctx::UOpCtx;
use crate::uop::ops::Ops;
use crate::uop::range::AxisType;

/// One candidate action (`Opt(op, axis, arg)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Action {
    pub op: OptOp,
    pub axis: usize,
    pub arg: i64,
}

/// `actions` (search.py:12) — restricted to the ported `OptOp`s.
pub fn actions() -> Vec<Action> {
    let mut a = Vec::new();
    for arg in [0i64, 2, 3, 4, 5, 7] {
        for axis in 0..8 {
            a.push(Action { op: OptOp::Upcast, axis, arg });
        }
    }
    for arg in [0i64, 4, 7] {
        for axis in 0..5 {
            a.push(Action { op: OptOp::Unroll, axis, arg });
        }
    }
    for arg in [2i64, 3, 4, 8, 13, 16, 29] {
        for axis in 0..6 {
            a.push(Action { op: OptOp::Local, axis, arg });
        }
    }
    for arg in [13i64, 16, 28, 29, 32, 49, 64, 256] {
        for axis in 0..3 {
            a.push(Action { op: OptOp::GroupTop, axis, arg });
        }
    }
    for arg in [0i64, 4, 8, 16] {
        for axis in 0..3 {
            a.push(Action { op: OptOp::Group, axis, arg });
        }
    }
    a.push(Action { op: OptOp::Local, axis: 0, arg: 32 });
    a.push(Action { op: OptOp::Local, axis: 6, arg: 2 });
    a
}

fn env_i64(key: &str, default: i64) -> i64 {
    std::env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
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
pub fn get_kernel_actions<'a>(ctx: &mut UOpCtx, s: &Scheduler<'a>, include_0: bool, max_up: i64, max_lcl: i64) -> Vec<Scheduler<'a>> {
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
            if fs[ax] == a.arg && acts.iter().any(|b| b.op == a.op && b.axis == a.axis && b.arg == 0) {
                continue;
            }
        } else {
            continue;
        }
        let mut s2 = s.copy();
        if s2.try_apply_opt(ctx, a.op, Some(a.axis), Some(a.arg)).is_err() {
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
fn rawbufs_from_ast(ctx: &mut UOpCtx, ast: &crate::uop::UopId, dev: &Rc<Compiled>) -> HashMap<i64, Rc<dyn OpaqueT>> {
    let mut map: HashMap<i64, Rc<dyn OpaqueT>> = HashMap::new();
    for u in ctx.backward_slice_with_self(ast) {
        if ctx.op_of(&u) != Ops::Param {
            continue;
        }
        let Arg::Param(p) = ctx.arg_of(&u) else { continue };
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
        map.insert(p.slot, dev.allocator.alloc(nbytes));
    }
    map
}

/// `_try_compile` + `_time_program`: compile one candidate and time it. Any
/// panic (a candidate the port's lowerer can't handle yet, e.g. a multi-store
/// kernel rejected by `coalesce`) is caught and the candidate skipped, matching
/// python's `try/except` in `_try_compile`.
fn try_compile_time<'a>(
    ctx: &mut UOpCtx,
    cand: &Scheduler<'a>,
    dev: &Rc<Compiled>,
    rawbufs: &HashMap<i64, Rc<dyn OpaqueT>>,
) -> Option<(Scheduler<'a>, f64)> {
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Option<(Scheduler<'a>, f64)> {
        let ast = cand.copy().get_optimized_ast(ctx, Some("test"));
        let mut cache = ProgCache::default();
        let cp = to_program(ctx, ast, dev, &mut cache);
        let mut bufs: Vec<Rc<dyn OpaqueT>> = Vec::with_capacity(cp.pi.globals.len());
        for slot in &cp.pi.globals {
            bufs.push(rawbufs.get(slot)?.clone());
        }
        let var_vals: HashMap<String, i64> = HashMap::new();
        let (mut gs, ls) = cp.pi.launch_dims(&var_vals);
        // `get_test_global_size`: cap the timing launch so a huge candidate can't
        // exhaust the driver; scale the measured time back up.
        let max_global: usize = std::env::var("BEAM_TEST_GLOBAL_MAX").ok().and_then(|s| s.parse().ok()).unwrap_or(65536);
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
        let elf = TinyElf { lib: cp.elf.lib.clone(), name: cp.elf.name.clone(), signature: cp.elf.signature.clone() };
        let rt = dev.runtime_nocache(&elf);
        let dur = rt.borrow_mut().call(&bufs, gs, ls, &cp.pi.vals(&var_vals), true)?;
        Some((cand.copy(), dur * factor))
    }));
    r.ok().flatten()
}

/// `beam_search` (search.py:110). Returns the best Scheduler found.
pub fn beam_search<'a>(ctx: &mut UOpCtx, base: Scheduler<'a>, amt: usize) -> Scheduler<'a> {
    // Candidates the lowerer can't handle yet panic; python catches these in
    // `_try_compile`, so suppress the (expected) panic messages during the search.
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let dev = device(&base.ren.device);
    let rawbufs = rawbufs_from_ast(ctx, &base.ast, &dev);

    let max_up = env_i64("BEAM_UPCAST_MAX", 256);
    let max_lcl = env_i64("BEAM_LOCAL_MAX", 1024);
    let min_progress = env_i64("BEAM_MIN_PROGRESS", 1) as f64 / 1e6;

    let mut beam: Vec<(Scheduler, f64)> = vec![(base.copy(), f64::INFINITY)];
    let start = std::time::Instant::now();
    let mut rounds = 0usize;
    loop {
        let mut candidates: Vec<Scheduler> = Vec::new();
        for (si, _) in &beam {
            candidates.extend(get_kernel_actions(ctx, si, false, max_up, max_lcl));
        }
        let mut timed: Vec<(Scheduler, f64)> = Vec::new();
        for cand in candidates {
            match try_compile_time(ctx, &cand, &dev, &rawbufs) {
                Some((s, t)) if t.is_finite() => timed.push((s, t)),
                _ => {}
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
    std::panic::set_hook(prev_hook);
    beam[0].0.copy()
}
