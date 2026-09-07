//! `codegen/__init__.py` port: the `to_program` pipeline (M5) — IN PROGRESS.
//!
//! `full_rewrite_to_sink` lowers a kernel AST (SINK with KernelInfo) into the
//! program SINK the renderer/interpreter consumes. This file implements the
//! first block of stages; the remaining python stages (postrange opts,
//! expander, reduce-local accs, gpudims, devectorize, decomps, control flow)
//! are listed in PLAN.md §M5 and append to `stages()` as they land.
//!
//! Differential loop: `py/dump_codegen_stages.py` emits per-stage UOp key
//! multisets from python; `TG_M5_STAGES` truncates this pipeline so the test
//! compares stage-for-stage.

use crate::codegen::postrange::{apply_opts, Renderer};
use crate::codegen::simplify::{pm_flatten_range, pm_reduce_unparented};
use crate::uop::sym::{pm_clean_up_group_sink, pm_mop_cleanup, symbolic, symbolic_simple, sget};
use crate::dtype;
use crate::uop::range::AxisType;
use crate::uop::sym::r;
use crate::uop::upat::{graph_rewrite, CtxVal, PatternMatcher, RuleFn, UPat};
use crate::uop::ctx::UOpCtx;
use crate::uop::sint::SVal;
use crate::uop::ops::group;
use crate::uop::{arg::Arg, ctx::UOpCtx as Ctx2, UopId};
use std::collections::HashMap;

/// Truncation for stage-by-stage differentials (default: run everything).
fn stage_limit() -> usize {
    std::env::var("TG_M5_STAGES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(usize::MAX)
}

/// One named graph_rewrite stage (name matches python for the key diff).
struct Stage {
    name: &'static str,
    pm: PatternMatcher,
    bottom_up: bool,
}

/// The implemented python stages, in order.
fn stages() -> Vec<Stage> {
    vec![
        Stage {
            name: "early movement ops",
            pm: PatternMatcher::new(vec![]).add(&crate::schedule::rangeify::pm_mops()).add(&pm_mop_cleanup()),
            bottom_up: true,
        },
        Stage { name: "initial symbolic", pm: symbolic().add(&pm_flatten_range()), bottom_up: false },
        Stage { name: "simplify ranges", pm: PatternMatcher::new(vec![]), bottom_up: false },
    ]
}

/// The stages after apply_opts.
fn post_stages() -> Vec<Stage> {
    vec![
        Stage { name: "postopt symbolic", pm: symbolic().add(&pm_flatten_range()).add(&pm_reduce_unparented()), bottom_up: false },
        Stage { name: "expander", pm: expander_pm(), bottom_up: false },
        Stage { name: "remove reduces", pm: PatternMatcher::new(vec![]).add(&pm_mop_cleanup()), bottom_up: false },
        Stage { name: "add local buffers", pm: PatternMatcher::new(vec![]).add(&crate::schedule::rangeify::pm_mops()), bottom_up: true },
        Stage { name: "add gpudims", pm: crate::codegen::gpudims::pm_add_gpudims(), bottom_up: false },
        Stage { name: "*** expand broadcast / add loads", pm: symbolic_simple().add(&pm_expand_broadcast()).add(&pm_add_loads()), bottom_up: false },
        Stage { name: "devectorize2", pm: devectorizer2(), bottom_up: false },
        Stage { name: "early symbolic", pm: PatternMatcher::new(vec![]).add(&symbolic()).add(&pm_clean_up_group_sink()), bottom_up: false },
        Stage { name: "add images", pm: PatternMatcher::new(vec![]), bottom_up: true },
        Stage { name: "extra symbolic", pm: PatternMatcher::new(vec![]).add(&symbolic()), bottom_up: false },
        Stage { name: "lower all index dtypes", pm: crate::uop::weak::pm_lower_index_dtype(), bottom_up: false },
        Stage { name: "final symbolic", pm: PatternMatcher::new(vec![]).add(&symbolic()), bottom_up: false },
        Stage { name: "cast float alu operands", pm: PatternMatcher::new(vec![]), bottom_up: false },
        Stage { name: "early decompositions", pm: PatternMatcher::new(vec![]).add(&symbolic_simple()), bottom_up: false },
        Stage { name: "decomp dtypes", pm: PatternMatcher::new(vec![]).add(&crate::uop::weak::pm_commit_weak()), bottom_up: false },
        Stage { name: "late decompositions", pm: PatternMatcher::new(vec![]), bottom_up: false },
        Stage { name: "move gates from index", pm: PatternMatcher::new(vec![]), bottom_up: false },
        Stage { name: "final rewrite", pm: pm_split_ends().add(&crate::uop::weak::pm_commit_weak()).add(&crate::uop::weak::pm_cast_weak()), bottom_up: false },
        Stage { name: "casted consts", pm: pm_casted_consts(), bottom_up: false,  },
        Stage { name: "add implicit barriers", pm: PatternMatcher::new(vec![]), bottom_up: false },
        Stage { name: "add control flow", pm: PatternMatcher::new(vec![]), bottom_up: true },
        Stage { name: "number params with -1", pm: PatternMatcher::new(vec![]), bottom_up: false },
    ]
}

/// `do_split_ends`/`pm_split_ends` (late/linearizer.py:87) — re-nest END
/// srcs: one nested END per range (sorted by arg desc), backedges last.
fn pm_split_ends() -> PatternMatcher {
    PatternMatcher::new(vec![(
        UPat::ops1(crate::uop::Ops::End).allow_all_srcs().named("e"),
        r(|ctx, st, _| {
            let e = sget(st, "e");
            let srcs = ctx.srcs(e).to_vec();
            let ret0 = srcs[0];
            let mut backedge = Vec::new();
            let mut rest = Vec::new();
            for &x in &srcs[1..] {
                let d = ctx.dt_of(x);
                if d == crate::dtype::VOID || d == crate::dtype::BOOL {
                    backedge.push(x);
                } else {
                    rest.push(x);
                }
            }
            let sink = crate::tensor::sink(ctx, rest.clone());
            let mut rngs = ctx.ranges(sink).to_vec();
            rngs.sort_by_key(|&x| std::cmp::Reverse(AxisType::range_axis_id(ctx, x)));
            let mut ret = ret0;
            for r in rngs {
                ret = ctx.end(ret, vec![r]);
            }
            if !backedge.is_empty() {
                ret = ctx.end(ret, backedge);
            }
            if ret == e {
                None
            } else {
                Some(ret)
            }
        }),
    )])
}

/// `pm_casted_consts` (codegen/__init__.py:280) — spell literals as
/// `cconst(val, dtype)` = CAST(CONST(val at from_py default), dtype).
fn pm_casted_consts() -> PatternMatcher {
    PatternMatcher::new(vec![(
        UPat::ops1(crate::uop::Ops::Const).allow_all_srcs().named("c"),
        r(|ctx, st, _| {
            let c = sget(st, "c");
            let v = ctx.val(c);
            let d = ctx.dt_of(c);
            if d == crate::dtype::VOID {
                return None;
            }
            // UOp.const(b) — the from_py default dtype (weakint/weakfloat/bool)
            let inner = ctx.const_py(v, None);
            Some(ctx.cast(inner, d))
        }),
    )])
}

/// `memory_coalescing` + `ew_devectorizer` for the "add images" stage.
fn add_images(ctx: &mut Ctx2, sink: UopId) -> UopId {
    let coalesced = crate::codegen::coalesce::memory_coalescing(ctx, sink);
    // ew_devectorizer: unpack remaining vectorized elementwise (corpus: no-op
    // after coalescing) — bottom-up rewrite with symbolic_simple
    let pm = PatternMatcher::new(vec![]).add(&symbolic_simple());
    graph_rewrite(ctx, coalesced, &pm, true, None, false, false, CtxVal::None)
        .unwrap_or_else(|e| panic!("TGERR: add images: {e}"))
}

/// `do_devectorize` (codegen/__init__.py:107).
fn do_devectorize(ctx: &mut Ctx2, b: UopId) -> Option<UopId> {
    let shape = ctx.shape(b).ok()?.as_ref().map(|v| v.as_ref().clone()).unwrap_or_default();
    if shape.is_empty() {
        return None;
    }
    let srcs = ctx.srcs(b).to_vec();
    // all srcs must have b's shape (or be INVALID consts)
    for &x in &srcs {
        if ctx.is_invalid(ctx.base(x)) {
            continue;
        }
        let xs = ctx.shape(x).ok()?.as_ref().map(|v| v.as_ref().clone()).unwrap_or_default();
        if xs != shape {
            return None;
        }
    }
    // cartesian product over the (1-D corpus) shape
    let idx_lists: Vec<Vec<UopId>> = shape
        .iter()
        .map(|&d| {
            let n = match d {
                SVal::I(v) => v as i64,
                _ => panic!("TGERR: symbolic devectorize"),
            };
            (0..n).map(|i| ctx.const_py(crate::dtype::ConstVal::Int(i as i128), None)).collect()
        })
        .collect();
    let mut elems: Vec<UopId> = Vec::new();
    for idx_c in cartesian(&idx_lists) {
        let mut new_srcs = Vec::with_capacity(srcs.len());
        for &x in &srcs {
            if ctx.is_invalid(ctx.base(x)) {
                new_srcs.push(ctx.base(x));
            } else {
                let mut isrcs = vec![x];
                isrcs.extend(idx_c.iter().copied());
                new_srcs.push(ctx.uop(crate::uop::Ops::Index, Some(ctx.dt_of(x)), isrcs, Arg::None));
            }
        }
        elems.push(ctx.uop(ctx.op_of(b), Some(ctx.dt_of(b)), new_srcs, ctx.arg_of(b).clone()));
    }
    if ctx.op_of(b) == crate::uop::Ops::Store {
        Some(ctx.uop(crate::uop::Ops::Group, Some(crate::dtype::VOID), elems, Arg::None))
    } else {
        let stack = ctx.uop(crate::uop::Ops::Stack, Some(ctx.dt_of(b)), elems, Arg::None);
        Some(crate::tensor::movement::reshape(ctx, stack, &shape))
    }
}

/// cartesian product over per-dim const lists.
fn cartesian(lists: &[Vec<UopId>]) -> Vec<Vec<UopId>> {
    let mut ret: Vec<Vec<UopId>> = vec![vec![]];
    for l in lists {
        let mut next = Vec::new();
        for combo in &ret {
            for &x in l {
                let mut c = combo.clone();
                c.push(x);
                next.push(c);
            }
        }
        ret = next;
    }
    ret
}

/// `devectorizer2` (codegen/__init__.py:139) — corpus subset (no WMMA).
fn devectorizer2() -> PatternMatcher {
    let targets: Vec<crate::uop::Ops> = group::ELEMENTWISE
        .iter()
        .copied()
        .chain([crate::uop::Ops::Load, crate::uop::Ops::Store])
        .collect();
    PatternMatcher::new(vec![
        // unpack broadcasting
        (
            UPat::op_var(targets).named("b"),
            r(|ctx, st, _| do_devectorize(ctx, sget(st, "b"))),
        ),
        // INDEX without src is nothing
        (UPat::ops1(crate::uop::Ops::Index).named("x"), r(|_ctx, _st, _| None)),
        // stacked INDEX is many INDEX
        (
            UPat::ops1(crate::uop::Ops::Index)
                .with_src_tup(vec![UPat::op_var(vec![crate::uop::Ops::Param, crate::uop::Ops::Buffer]).named("b"), UPat::ops1(crate::uop::Ops::Stack).named("s")])
                .named("x"),
            r(|ctx, st, _| {
                let b = sget(st, "b");
                let x = sget(st, "x");
                let ss = ctx.srcs(sget(st, "s")).to_vec();
                let mut elems = Vec::new();
                for &u in &ss {
                    let mut srcs = vec![b, u];
                    srcs.extend(ctx.srcs(x)[2..].iter().copied());
                    elems.push(ctx.uop(crate::uop::Ops::Index, Some(ctx.dt_of(x)), srcs, ctx.arg_of(x).clone()));
                }
                Some(ctx.uop(crate::uop::Ops::Stack, Some(ctx.dt_of(x)), elems, ctx.arg_of(x).clone()))
            }),
        ),
        // INDEX into RESHAPE moves the RESHAPE
        (
            UPat::ops1(crate::uop::Ops::Index)
                .with_src_tup(vec![
                    UPat::op_var(vec![crate::uop::Ops::Param, crate::uop::Ops::Buffer]).named("b"),
                    UPat::ops1(crate::uop::Ops::Reshape).named("s"),
                ])
                .named("x"),
            r(|ctx, st, _| {
                let b = sget(st, "b");
                let s = sget(st, "s");
                let shape = ctx.shape(s).ok()?.as_ref().map(|v| v.as_ref().clone()).unwrap_or_default();
                let new_idx = crate::tensor::movement::reshape(ctx, ctx.srcs(s)[0], &shape);
                let idx = ctx.uop(crate::uop::Ops::Index, Some(ctx.dt_of(ctx.srcs(s)[0])), vec![b, new_idx], Arg::None);
                Some(crate::tensor::movement::reshape(ctx, idx, &shape))
            }),
        ),
        // RESHAPE a void is removed
        (
            UPat::ops1(crate::uop::Ops::Reshape).with_dtype(Some(vec![crate::dtype::VOID])).named("x"),
            r(|ctx, st, _| Some(ctx.srcs(sget(st, "x"))[0])),
        ),
        // reshape of a single element shaped value to scalar is an index
        (
            UPat::ops1(crate::uop::Ops::Reshape).named("x"),
            r(|ctx, st, _| {
                let x = sget(st, "x");
                let shape = ctx.shape(x).ok()?.as_ref().map(|v| v.as_ref().clone()).unwrap_or_default();
                let src = ctx.srcs(x)[0];
                let src_shape = ctx.shape(src).ok()?.as_ref().map(|v| v.as_ref().clone()).unwrap_or_default();
                if shape.is_empty() && src_shape == vec![SVal::I(1)] {
                    let zero = ctx.const_py(crate::dtype::ConstVal::Int(0), None);
                    return Some(ctx.uop(crate::uop::Ops::Index, Some(ctx.dt_of(src)), vec![src, zero], Arg::None));
                }
                None
            }),
        ),
        // EXPAND on scalar -> STACK
        (
            UPat::ops1(crate::uop::Ops::Expand).with_src_any_len(true).named("out"),
            r(|ctx, st, _| {
                let out = sget(st, "out");
                let x = ctx.srcs(out)[0];
                let xs = ctx.shape(x).ok()?.as_ref().map(|v| v.as_ref().clone()).unwrap_or_default();
                let os = ctx.shape(out).ok()?.as_ref().map(|v| v.as_ref().clone()).unwrap_or_default();
                if !xs.is_empty() || os.len() != 1 {
                    return None;
                }
                let n = match os[0] {
                    SVal::I(v) => v as i64,
                    _ => return None,
                };
                // out.shape == (out.max_numel(),) — 1-D is its own max shape
                let copies = vec![x; n as usize];
                Some(ctx.uop(crate::uop::Ops::Stack, Some(ctx.dt_of(out)), copies, Arg::None))
            }),
        ),
    ])
    .add(&pm_mop_cleanup())
    .add(&crate::schedule::rangeify::pm_mops())
}

/// `build_range_map` (codegen/__init__.py:39).
fn build_range_map(ctx: &mut Ctx2, sink: UopId) -> HashMap<i64, usize> {
    let mut map = HashMap::new();
    for &x in ctx.toposort(sink).iter() {
        if ctx.op_of(x) == crate::uop::Ops::Range {
            if let Arg::Seq { tuple: true, items } = ctx.arg_of(x) {
                if let Arg::Enum(e) = &items[1] {
                    if matches!(e.value, 9 | 8) {
                        // UNROLL | UPCAST
                        let axis = match &items[0] {
                            Arg::Int(i) => *i,
                            _ => continue,
                        };
                        let n = map.len();
                        map.entry(axis).or_insert(n);
                    }
                }
            }
        }
    }
    map
}

/// `expander2` (codegen/__init__.py:74) — corpus subset (no REDUCE lift, no WMMA).
fn expander_pm() -> PatternMatcher {
    // the range map is captured at PM-construction time from the caller
    PatternMatcher::new(vec![])
}

/// expander with the range map bound.
fn expander_pm_with(ctx: &mut Ctx2, sink: UopId) -> (PatternMatcher, HashMap<i64, usize>) {
    let range_map = build_range_map(ctx, sink);
    let rm = range_map.clone();
    let pm = PatternMatcher::new(vec![(
        UPat::ops1(crate::uop::Ops::Range).named("r"),
        r(move |ctx: &mut Ctx2, st, _| {
            let r = sget(st, "r");
            let (op, dt, srcs, arg, tag) = ctx.snapshot(r);
            let _ = op;
            let Arg::Seq { tuple: true, items } = &arg else { return None };
            let axis = match &items[0] {
                Arg::Int(i) => *i,
                _ => return None,
            };
            let pos = *rm.get(&axis)?;
            let vmax = match ctx.ssimplify(srcs[0]) {
                Ok(SVal::I(v)) => v as i64 - 1,
                _ => return None,
            };
            // UOp.const(tuple(range(vmax+1)), r.dtype).reshape(...)
            let consts: Vec<UopId> = (0..=vmax).map(|i| ctx.const_py(crate::dtype::ConstVal::Int(i as i128), Some(dt))).collect();
            let stack = ctx.uop(crate::uop::Ops::Stack, Some(dt), consts, Arg::None);
            let shape: Vec<SVal> = (0..rm.len()).map(|i| SVal::I(if i == pos { vmax + 1 } else { 1 } as i128)).collect();
            Some(crate::tensor::movement::reshape(ctx, stack, &shape))
        }),
    )])
    .add(&pm_flatten_range())
    .add(&pm_mop_cleanup());
    (pm, range_map)
}

/// `expand_broadcast` (codegen/__init__.py:97).
fn expand_broadcast(ctx: &mut Ctx2, x: UopId) -> Option<UopId> {
    let srcs = ctx.srcs(x).to_vec();
    let mut shapes: Vec<Vec<SVal>> = Vec::new();
    for &u in &srcs {
        match ctx.shape(u) {
            Ok(s) => shapes.push(s.as_ref().map(|v| v.as_ref().clone()).unwrap_or_default()),
            Err(_) => return None,
        }
    }
    if shapes.iter().all(|s| *s == shapes[0]) {
        return None;
    }
    let shape = ctx.broadcast_shape(&shapes).ok()?;
    let mut new_srcs = Vec::new();
    for &u in &srcs {
        new_srcs.push(crate::tensor::movement::expand(ctx, u, &shape));
    }
    Some(ctx.uop(ctx.op_of(x), Some(ctx.dt_of(x)), new_srcs, ctx.arg_of(x).clone()))
}

/// `pm_expand_broadcast` (codegen/__init__.py:106) — corpus subset (no WMMA).
fn pm_expand_broadcast() -> PatternMatcher {
    PatternMatcher::new(vec![(
        UPat::op_var(
            group::BINARY
                .iter()
                .copied()
                .chain(group::TERNARY.iter().copied())
                .chain([crate::uop::Ops::Store])
                .collect(),
        )
        .named("x"),
        r(|ctx, st, _| expand_broadcast(ctx, sget(st, "x"))),
    )])
}

/// `maybe_load` (codegen/__init__.py:245).
fn maybe_load(ctx: &mut Ctx2, u: UopId) -> UopId {
    match ctx.addrspace_derived(u) {
        Some(a) if matches!(a, crate::dtype::AddrSpace::Global | crate::dtype::AddrSpace::Local | crate::dtype::AddrSpace::Reg) => {
            crate::tensor::load(ctx, u, vec![])
        }
        _ => u,
    }
}

/// `pm_add_loads` (codegen/__init__.py:246).
fn pm_add_loads() -> PatternMatcher {
    let elementwise_like: Vec<crate::uop::Ops> = group::ELEMENTWISE
        .iter()
        .copied()
        .chain([crate::uop::Ops::Reduce, crate::uop::Ops::Wmma, crate::uop::Ops::Stack])
        .collect();
    PatternMatcher::new(vec![
        (
            UPat::op_var(elementwise_like).named("x"),
            r(|ctx, st, _| {
                let x = sget(st, "x");
                let srcs = ctx.srcs(x).to_vec();
                let new_srcs: Vec<UopId> = srcs.iter().map(|&s| maybe_load(ctx, s)).collect();
                if new_srcs == srcs {
                    return None;
                }
                Some(ctx.uop(ctx.op_of(x), Some(ctx.dt_of(x)), new_srcs, ctx.arg_of(x).clone()))
            }),
        ),
        (
            UPat::ops1(crate::uop::Ops::Store).named("x"),
            r(|ctx, st, _| {
                let x = sget(st, "x");
                let srcs = ctx.srcs(x).to_vec();
                let loaded = maybe_load(ctx, srcs[1]);
                if loaded == srcs[1] {
                    return None;
                }
                let mut new_srcs = vec![srcs[0], loaded];
                new_srcs.extend(srcs[2..].iter().copied());
                Some(ctx.uop(crate::uop::Ops::Store, Some(ctx.dt_of(x)), new_srcs, ctx.arg_of(x).clone()))
            }),
        ),
    ])
}

/// `full_rewrite_to_sink` (codegen/__init__.py:288) — partial: the opening
/// mops/symbolic block. Returns the rewritten SINK.
pub fn full_rewrite_to_sink(ctx: &mut UOpCtx, ast: UopId) -> UopId {
    full_rewrite_to_sink_with(ctx, ast, &Renderer::cpu_interp())
}

pub fn full_rewrite_to_sink_with(ctx: &mut UOpCtx, ast: UopId, ren: &Renderer) -> UopId {
    let limit = stage_limit();
    let mut i = 0usize;
    let mut cur = ast;
    for st in stages() {
        if i + 1 > limit {
            return cur;
        }
        cur = graph_rewrite(ctx, cur, &st.pm, st.bottom_up, None, false, false, CtxVal::None)
            .unwrap_or_else(|e| panic!("TGERR: {}: {e}", st.name));
        i += 1;
    }
    if i + 1 > limit {
        return cur;
    }
    // do postrange optimization, BEAM or hand_coded_optimizations
    cur = apply_opts(ctx, cur, ren);
    i += 1;
    for st in post_stages() {
        if i + 1 > limit {
            return cur;
        }
        cur = if st.name == "expander" {
            let (pm, _rm) = expander_pm_with(ctx, cur);
            graph_rewrite(ctx, cur, &pm, st.bottom_up, None, false, false, CtxVal::None)
                .unwrap_or_else(|e| panic!("TGERR: {}: {e}", st.name))
        } else if st.name == "add images" {
            add_images(ctx, cur)
        } else {
            graph_rewrite(ctx, cur, &st.pm, st.bottom_up, None, false, false, CtxVal::None)
                .unwrap_or_else(|e| panic!("TGERR: {}: {e}", st.name))
        };
        i += 1;
    }
    cur
}
