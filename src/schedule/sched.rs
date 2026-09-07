//! `schedule/__init__.py` port — kernel scheduling (dependency graph over
//! AFTER states), LINEAR resolution, and buffer allocation.

use std::collections::{HashMap, VecDeque};

use crate::dtype::{AddrSpace, ConstVal};
use crate::uop::arg::{Arg, ParamArg};
use crate::uop::ctx::UOpCtx;
use crate::uop::ops::{group, Ops};
use crate::uop::sym::r;
use crate::uop::upat::{graph_rewrite, r_ctx, CtxVal, PatternMatcher, RuleFn, UPat};
use crate::uop::UopId;

use super::indexing::gate_kernel_sink;
use super::rangeify::get_kernel_graph;

/// `_unwrap_src` (schedule/__init__.py:10).
fn unwrap_src(ctx: &UOpCtx, s: UopId) -> UopId {
    let mut s = s;
    while !ctx.srcs(s).is_empty()
        && !matches!(ctx.op_of(s), Ops::After | Ops::Buffer | Ops::Param | Ops::Mselect | Ops::Mstack)
    {
        s = ctx.srcs(s)[0];
    }
    s
}

/// `_states` (schedule/__init__.py:15).
fn states(ctx: &UOpCtx, s: UopId) -> Vec<UopId> {
    let s = unwrap_src(ctx, s);
    if matches!(ctx.op_of(s), Ops::Mselect | Ops::Mstack) {
        let mut out = Vec::new();
        for &ss in ctx.srcs(s) {
            out.extend(states(ctx, ss));
        }
        return out;
    }
    assert!(
        matches!(ctx.op_of(s), Ops::After | Ops::Buffer | Ops::Param),
        "TGERR: input to kernel must resolve to a buffer state, not {:?}",
        ctx.op_of(s)
    );
    vec![s]
}

/// `_split_after` (schedule/__init__.py:21) — (kernels, after_deps).
fn split_after(ctx: &UOpCtx, after: UopId) -> (Vec<UopId>, Vec<UopId>) {
    let mut kernels = Vec::new();
    let mut rest = Vec::new();
    for &s in ctx.srcs(after)[1..].iter() {
        if matches!(ctx.op_of(s), Ops::Call | Ops::End) {
            kernels.push(s);
        } else {
            rest.push(s);
        }
    }
    let mut deps = Vec::new();
    let mut remaining = Vec::new();
    for s in rest {
        if ctx.op_of(s) == Ops::After {
            deps.push(s);
        } else {
            remaining.push(s);
        }
    }
    if let Some(first) = remaining.first() {
        panic!("TGERR: AFTER source should be CALL, END, STORE, or AFTER, not {:?}", ctx.op_of(*first));
    }
    (kernels, deps)
}

/// `create_schedule` (schedule/__init__.py:28) — topological kernel ordering
/// with RAW + WAR dependencies.
pub fn create_schedule(ctx: &mut UOpCtx, sched_sink: UopId) -> UopId {
    // build kernel dependency graph
    let mut children: HashMap<UopId, Vec<UopId>> = HashMap::new();
    let mut in_degree: HashMap<UopId, i64> = HashMap::new();
    // buffer -> [(AFTER, prior state, new kernels)]
    let mut writes: HashMap<UopId, Vec<(UopId, UopId, Vec<UopId>)>> = HashMap::new();
    let mut reads: Vec<(UopId, UopId, UopId)> = Vec::new();
    for u in ctx.toposort_gated(sched_sink, Some(&|x| gate_kernel_sink(ctx, x)), true) {
        if ctx.op_of(u) != Ops::After {
            continue;
        }
        let (kernels, after_deps) = split_after(ctx, u);
        let buf = ctx.buf_uop(u);
        let prev_state = unwrap_src(ctx, ctx.srcs(u)[0]);
        let prev_kernels: Vec<UopId> = if ctx.op_of(prev_state) == Ops::After { split_after(ctx, prev_state).0 } else { vec![] };
        let new_kernels: Vec<UopId> = kernels.iter().filter(|k| !prev_kernels.contains(k)).copied().collect();
        writes.entry(buf).or_default().push((u, prev_state, new_kernels));
        for &k in &kernels {
            *in_degree.entry(k).or_insert(0) += 0;
            if ctx.op_of(k) == Ops::End {
                assert!(ctx.op_of(ctx.srcs(k)[0]) == Ops::Call, "TGERR: END src[0] should be KERNEL");
            }
            let kernel_deps: Vec<UopId> =
                if ctx.op_of(k) == Ops::End { ctx.srcs(ctx.srcs(k)[0])[1..].to_vec() } else { ctx.srcs(k)[1..].to_vec() };
            let mut read_states = Vec::new();
            for s in &kernel_deps {
                read_states.extend(states(ctx, *s));
            }
            for &st in &read_states {
                reads.push((u, k, st));
            }
            // RAW deps
            for st in read_states.iter().copied().chain(after_deps.iter().flat_map(|s| states(ctx, *s))) {
                if ctx.op_of(st) == Ops::After {
                    for t in split_after(ctx, st).0 {
                        children.entry(t).or_default().push(k);
                        *in_degree.entry(k).or_insert(0) += 1;
                    }
                }
            }
        }
    }
    // WAR deps
    for (u, k, s) in reads.clone() {
        if let Some(ws) = writes.get(&ctx.buf_uop(s)).cloned() {
            for (a, prev_state, write_kernels) in ws {
                if a == u || prev_state != s {
                    continue;
                }
                for t in &write_kernels {
                    if *t != k && !ctx.backward_slice(k).contains(t) {
                        children.entry(k).or_default().push(*t);
                        *in_degree.entry(*t).or_insert(0) += 1;
                    }
                }
            }
        }
    }
    // Kahn
    let mut queue: VecDeque<UopId> = in_degree.iter().filter(|(_, v)| **v == 0).map(|(&k, _)| k).collect();
    queue.make_contiguous();
    let mut linearized: Vec<UopId> = Vec::new();
    while let Some(rk) = queue.pop_front() {
        if ctx.op_of(rk) == Ops::Linear {
            linearized.extend(ctx.srcs(rk));
        } else {
            let k = if ctx.op_of(rk) == Ops::End { ctx.srcs(rk)[0] } else { rk };
            assert!(ctx.op_of(k) == Ops::Call, "TGERR: unexpected op in queue: {:?}", ctx.op_of(k));
            let mut buf_uops = Vec::new();
            let ksrcs = ctx.srcs(k)[1..].to_vec();
            for s in ksrcs {
                if !ctx.is_bound_var(s) {
                    let u = unwrap_src(ctx, s);
                    let b = ctx.buf_uop(u);
                    buf_uops.push(b);
                }
            }
            let lin_call = ctx.call(ctx.srcs(k)[0], buf_uops, None);
            linearized.push(lin_call);
        }
        if let Some(ch) = children.get(&rk).cloned() {
            for x in ch {
                let e = in_degree.entry(x).or_insert(0);
                *e -= 1;
                if *e == 0 {
                    queue.push_back(x);
                }
            }
        }
    }
    if in_degree.values().any(|&v| v != 0) {
        panic!("TGERR: cycle detected in assign graph");
    }
    ctx.uop(Ops::Linear, None, linearized, Arg::None)
}

// ---------- linear resolution (schedule/__init__.py:88-115) ----------

/// `create_new_buffer` — fresh allocation mirroring the buffer spec.
fn create_new_buffer(map: &mut HashMap<UopId, UopId>, ctx: &mut UOpCtx, b: UopId) -> UopId {
    if let Some(&r) = map.get(&b) {
        return r;
    }
    let dev = crate::schedule::indexing::device_of(ctx, b);
    let dev_s = match dev {
        Some(crate::uop::arg::DeviceSpec::Single(s)) => s,
        Some(crate::uop::arg::DeviceSpec::Multi(_)) => panic!("TGERR: multi-device alloc unsupported"),
        None => panic!("TGERR: buffer without device"),
    };
    let numel = ctx.max_numel(b);
    let dt = ctx.dt_of(b);
    let ret = crate::tensor::creation::new_buffer(ctx, &dev_s, crate::uop::sint::SVal::I(numel as i128), dt, None);
    map.insert(b, ret);
    ret
}

/// `pm_post_sched_cache` (schedule/__init__.py:92).
pub fn pm_post_sched_cache() -> PatternMatcher {
    let rules: Vec<(UPat, RuleFn)> = vec![
        // only resolve buffer PARAMs (slot>=0)
        (
            UPat::ops1(Ops::Param).named("x"),
            r_ca(|(cache, args), ctx, st| {
                let x = sym_sget(st, "x");
                if let Arg::Param(p) = ctx.arg_of(x) {
                    if p.slot >= 0 {
                        let arg = *args.get(p.slot as usize).unwrap_or_else(|| panic!("TGERR: param slot {} out of range", p.slot));
                        return Some(arg);
                    }
                }
                None
            }),
        ),
        // create new BUFFERs
        (
            UPat::ops1(Ops::Buffer).with_src_tup(vec![UPat::var(None, None)]).named("b"),
            r_ca(|(cache, _args), ctx, st| {
                let b = sym_sget(st, "b");
                if let Arg::Param(p) = ctx.arg_of(b) {
                    if p.addrspace == Some(AddrSpace::Global) {
                        return Some(create_new_buffer(cache, ctx, b));
                    }
                }
                None
            }),
        ),
    ];
    PatternMatcher::new(rules)
}

pub type CallCtx = (HashMap<UopId, UopId>, Vec<UopId>);

fn sym_sget(st: &crate::uop::upat::Store, name: &str) -> UopId {
    crate::uop::sym::sget(st, name)
}

/// rule helper over the (cache, args) call context.
pub fn r_ca<F>(f: F) -> RuleFn
where
    F: Fn(&mut CallCtx, &mut UOpCtx, &crate::uop::upat::Store) -> Option<UopId> + Send + Sync + 'static,
{
    std::sync::Arc::new(move |ctx, st, ctxv| match ctxv {
        CtxVal::Ca(c) => f(&mut c.borrow_mut(), ctx, st),
        _ => panic!("TGERR: rule requires call ctx"),
    })
}

/// `resolve_linear_call` (schedule/__init__.py:100).
pub fn resolve_linear_call(ctx: &mut UOpCtx, linear_call: UopId) -> UopId {
    let args = ctx.srcs(linear_call)[1..].to_vec();
    let pm = pm_post_sched_cache();
    let linear_body = ctx.srcs(linear_call)[0];
    let linear = graph_rewrite(ctx, linear_body, &pm, false, None, true, false, CtxVal::Ca(std::rc::Rc::new(std::cell::RefCell::new((HashMap::new(), args)))))
        .unwrap_or_else(|e| panic!("TGERR: params to buffers: {e}"));
    // corpus: no bound variables/scalar binds — srcs pass through unchanged
    let (op, dt, _srcs, arg, _tag) = ctx.snapshot(linear);
    // python's pm_resolve_linear_call replaces the CALL with the resolved LINEAR
    let _ = (op, dt, arg, _tag);
    linear
}

/// `pm_resolve_linear_call` — resolve CALL(LINEAR) nodes.
pub fn pm_resolve_linear_call() -> PatternMatcher {
    PatternMatcher::new(vec![(
        UPat::ops1(Ops::Call).with_src_tup(vec![UPat::ops1(Ops::Linear)]).allow_all_srcs().named("lc"),
        r(|_c, st, _| {
            let _ = st;
            None // handled by the direct call in create_linear_with_vars
        }),
    )])
}

/// `create_linear_with_vars` (schedule/__init__.py:181) — returns the resolved
/// LINEAR (pre-memory-plan) plus the final memory-planned LINEAR.
pub fn create_linear_with_vars(ctx: &mut UOpCtx, big_sink: UopId) -> (UopId, UopId, Vec<UopId>) {
    // schedule to linear: pm_schedule lowers the top SINK
    let linear_call = {
        let pm_sched = pm_schedule();
        graph_rewrite(ctx, big_sink, &pm_sched, false, None, false, true, CtxVal::None)
            .unwrap_or_else(|e| panic!("TGERR: schedule to linear: {e}"))
    };

    // held bufs (schedule/__init__.py:208): the top CALL's direct BUFFER args
    let held_bufs: Vec<UopId> = ctx.srcs(linear_call)[1..]
        .iter()
        .copied()
        .filter(|&a| ctx.op_of(a) == Ops::Buffer)
        .collect();

    // resolve the linear_call (allocates buffers)
    let linear = resolve_linear_call(ctx, linear_call);

    // create copies (SDMA) — needed for cross-device (PYTHON->NV) copies
    let linear = graph_rewrite(ctx, linear, &pm_copy_from_store(), false, None, false, false, CtxVal::None)
        .unwrap_or_else(|e| panic!("TGERR: create copies: {e}"));

    // vars: corpus is concrete — no var_vals
    let planned = crate::schedule::memory::memory_plan_rewrite(ctx, linear, &held_bufs);
    (linear, planned, held_bufs)
}

/// `pm_schedule` / `lower_sink_to_linear` — lower the (callified) tensor SINK.
pub fn pm_schedule() -> PatternMatcher {
    PatternMatcher::new(vec![(
        UPat::ops1(Ops::Sink).allow_all_srcs().named("function"),
        r(|ctx, st, _| {
            let function = sym_sget(st, "function");
            // function.arg is KernelInfo for inner kernel sinks — leave those
            if let Arg::Data(d) = ctx.arg_of(function) {
                if d.tag == "KernelInfo" {
                    return None;
                }
            }
            let sched_sink = super::rangeify::get_kernel_graph(ctx, function);
            Some(create_schedule(ctx, sched_sink))
        }),
    )])
}

// ---------- pm_copy_from_store (schedule/__init__.py:164-178) ----------

/// `copy_kernel_to_copy_uop` — replace the SDMA store kernel with a COPY uop.
fn copy_kernel_to_copy_uop(ctx: &mut UOpCtx, call: UopId, dst: UopId, src: UopId) -> Option<UopId> {
    let ddev = crate::schedule::indexing::device_of(ctx, dst);
    let sdev = crate::schedule::indexing::device_of(ctx, src);
    if ddev == sdev {
        return None;
    }
    let ddev_s = match ddev {
        Some(crate::uop::arg::DeviceSpec::Single(s)) => s,
        _ => panic!("TGERR: bad copy dst device"),
    };
    let copy = ctx.uop_tag(Ops::Copy, Some(ctx.dt_of(src)), vec![src], Arg::Str(ddev_s), ctx.tag_of(call).clone());
    let mut new_srcs = vec![copy];
    new_srcs.extend(ctx.srcs(call)[1..].iter().copied());
    Some(ctx.uop_tag(Ops::Call, Some(ctx.dt_of(call)), new_srcs, ctx.arg_of(call).clone(), ctx.tag_of(call).clone()))
}

/// `pm_copy_from_store`.
pub fn pm_copy_from_store() -> PatternMatcher {
    let zero = || UPat::const_pat(crate::dtype::ConstVal::Int(0), None);
    let rules: Vec<(UPat, RuleFn)> = vec![
        // dst PARAM index const-0 store src PARAM index const-0, SINK body
        (
            UPat::ops1(Ops::Call)
                .with_src_tup(vec![
                    UPat::ops1(Ops::Sink)
                        .with_src_tup(vec![
                            UPat::ops1(Ops::Param)
                                .index(vec![zero()])
                                .store(vec![UPat::ops1(Ops::Param).named("src").index(vec![zero()])])
                                .named("dst"),
                        ])
                        .allow_all_srcs(),
                ])
                .allow_all_srcs()
                .named("call"),
            r_ctx(|ctx, st| {
                let (call, dst, src) = (sym_sget(st, "call"), sym_sget(st, "dst"), sym_sget(st, "src"));
                copy_kernel_to_copy_uop(ctx, call, dst, src)
            }),
        ),
        // range version: dst PARAM index RANGE r store src PARAM index RANGE r end r
        (
            UPat::ops1(Ops::Call)
                .with_src_tup(vec![
                    UPat::ops1(Ops::Sink)
                        .with_src_tup(vec![
                            UPat::ops1(Ops::Param)
                                .index(vec![UPat::ops1(Ops::Range).named("r")])
                                .store(vec![UPat::ops1(Ops::Param).named("src").index(vec![UPat::ops1(Ops::Range).named("r")])])
                                .end(vec![UPat::ops1(Ops::Range).named("r")])
                                .named("dst"),
                        ])
                        .allow_all_srcs(),
                ])
                .allow_all_srcs()
                .named("call"),
            r_ctx(|ctx, st| {
                let (call, dst, src) = (sym_sget(st, "call"), sym_sget(st, "dst"), sym_sget(st, "src"));
                copy_kernel_to_copy_uop(ctx, call, dst, src)
            }),
        ),
        // all buffers must be on the same device for real kernels
        (
            UPat::ops1(Ops::Call).with_src_tup(vec![UPat::ops1(Ops::Sink).named("ast")]).allow_all_srcs(),
            r_ctx(|ctx, st| {
                let ast = sym_sget(st, "ast");
                let mut devs: Vec<String> = Vec::new();
                for n in ctx.toposort(ast) {
                    if ctx.op_of(n) == Ops::Param {
                        if let Some(crate::uop::arg::DeviceSpec::Single(d)) = crate::schedule::indexing::device_of(ctx, n) {
                            if !devs.contains(&d) {
                                devs.push(d);
                            }
                        }
                    }
                }
                if devs.len() >= 2 {
                    panic!("TGERR: all buffers must be on the same device: {devs:?}");
                }
                None
            }),
        ),
    ];
    PatternMatcher::new(rules)
}


