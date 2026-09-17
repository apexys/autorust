//! `codegen/__init__.py` port: the `to_program` pipeline (M5) — IN PROGRESS.
//!
//! `full_rewrite_to_sink` lowers a kernel AST (SINK with KernelInfo) into the
//! program SINK the renderer/interpreter consumes. This file implements the
//! first block of stages; the remaining python stages (postrange opts,
//! expander, reduce-local accs, gpudims, devectorize, decomps, control flow)
//! are listed in CURRENT_STATE.md §5 and append to `stages()` as they land.
//!
//! Differential loop: `py/dump_codegen_stages.py` emits per-stage UOp key
//! multisets from python; `TG_M5_STAGES` truncates this pipeline so the test
//! compares stage-for-stage.

use crate::codegen::postrange::apply_opts;
use crate::renderer::Renderer;
use crate::codegen::simplify::{pm_flatten_range, pm_load_collapse, pm_reduce_unparented, pm_split_ranges};
use crate::uop::sym::{pm_cast_float_alu, pm_mop_cleanup, pm_move_where_on_load, pm_remove_invalid, sym, symbolic, symbolic_simple, sget};
use crate::schedule::rangeify::pm_mops;
use crate::dtype;
use crate::uop::range::AxisType;
use crate::uop::sym::r;
use crate::uop::upat::{graph_rewrite, CtxVal, PatternMatcher, RuleFn, UPat};
use crate::uop::ctx::UOpCtx;
use crate::uop::sint::SVal;
use crate::uop::ops::group;
use crate::uop::{arg::Arg, ctx::UOpCtx as Ctx2, UopId};
use std::collections::{HashMap, HashSet};

/// Truncation for stage-by-stage differentials (default: run everything).
fn stage_limit() -> usize {
    #[cfg(feature = "dump")]
    if let Ok(v) = std::env::var("TG_M5_STAGES")
        && let Ok(n) = v.parse::<usize>() {
            return n;
        }
    usize::MAX
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
            pm: PatternMatcher::new(vec![]).add(&crate::schedule::rangeify::pm_mops()).add(pm_mop_cleanup()),
            bottom_up: true,
        },
        Stage { name: "load collapse", pm: pm_load_collapse(), bottom_up: false },
        Stage { name: "split ranges", pm: pm_split_ranges().add(&pm_flatten_range()), bottom_up: false },
        Stage { name: "initial symbolic", pm: sym().add(&pm_flatten_range()), bottom_up: false },
        Stage { name: "simplify ranges", pm: pm_flatten_range().add(&crate::codegen::simplify::pm_simplify_ranges()), bottom_up: false },
    ]
}

/// The stages after apply_opts. `ren` selects the decomposition op set (e.g.
/// the interpreter supports MAX, the C-style/OpenCL renderers do not).
fn post_stages(ren: &Renderer) -> Vec<Stage> {
    // `supported_ops = tuple(ren.code_for_op.keys())` (codegen/__init__.py:364).
    let decomp = crate::codegen::decomp::pm_decomp(ren.code_for_op.contains(&crate::uop::Ops::Max));
    vec![
        Stage { name: "postopt symbolic", pm: sym().add(&pm_move_where_on_load()).add(&pm_flatten_range()).add(&pm_reduce_unparented()), bottom_up: false },
        Stage { name: "expander", pm: expander_pm(), bottom_up: false },
        Stage { name: "remove reduces", pm: PatternMatcher::new(vec![]).add(pm_mop_cleanup()).add(&crate::codegen::simplify::pm_reduce_local()), bottom_up: false },
        Stage { name: "add local buffers", pm: PatternMatcher::new(vec![]).add(&crate::schedule::rangeify::pm_add_local_buffers()).add(&crate::schedule::rangeify::pm_mops()), bottom_up: true },
        Stage { name: "add gpudims", pm: crate::codegen::gpudims::pm_add_gpudims(), bottom_up: false },
        Stage { name: "*** expand broadcast / add loads", pm: symbolic_simple().add(&pm_expand_broadcast()).add(&pm_add_loads()), bottom_up: false },
        Stage { name: "devectorize2", pm: symbolic_simple().add(&devectorizer2()).add(crate::codegen::coalesce::indexing_simplify()), bottom_up: false },
        Stage { name: "early symbolic", pm: PatternMatcher::new(vec![]).add(sym()), bottom_up: false },
        Stage { name: "add images", pm: PatternMatcher::new(vec![]), bottom_up: true },
        Stage { name: "extra symbolic", pm: sym().add(crate::codegen::coalesce::indexing_simplify()), bottom_up: false },
        Stage { name: "lower all index dtypes", pm: symbolic_simple().add(&crate::uop::weak::pm_lower_index_dtype()).add(crate::codegen::coalesce::indexing_simplify()), bottom_up: false },
        Stage { name: "final symbolic", pm: PatternMatcher::new(vec![]).add(symbolic()), bottom_up: false },
        Stage { name: "cast float alu operands", pm: pm_cast_float_alu(), bottom_up: false },
        Stage { name: "early decompositions", pm: decomp.clone(), bottom_up: false },
        Stage { name: "decomp dtypes", pm: crate::codegen::decomp_dtype::pm_dtype_decomps().add(&crate::uop::weak::pm_commit_weak()), bottom_up: false },
        Stage { name: "late decompositions", pm: decomp, bottom_up: false },
        Stage { name: "move gates from index", pm: crate::codegen::gater::pm_move_gates_from_index(), bottom_up: false },
        Stage { name: "final rewrite", pm: pm_split_ends().add(&crate::uop::weak::pm_commit_weak()).add(&crate::uop::weak::pm_cast_weak()).add(&pm_remove_invalid()), bottom_up: false },
        Stage { name: "casted consts", pm: pm_casted_consts(), bottom_up: false,  },
        Stage { name: "add implicit barriers", pm: pm_implicit_barriers(), bottom_up: false },
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
            let srcs = ctx.srcs(&e).to_vec();
            let ret0 = srcs[0].clone();
            let mut backedge = Vec::new();
            let mut rest = Vec::new();
            for x in &srcs[1..] {
                let d = ctx.dt_of(x);
                if d == crate::dtype::VOID || d == crate::dtype::BOOL {
                    backedge.push(x.clone());
                } else {
                    rest.push(x.clone());
                }
            }
            let sink = crate::tensor::sink(ctx, rest.clone());
            let mut rngs = ctx.ranges(&sink).to_vec();
            rngs.sort_by_key(|x| std::cmp::Reverse(AxisType::range_axis_id(ctx, x.clone())));
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

/// `_is_local_store` (codegen/__init__.py:259).
fn is_local_store(ctx: &UOpCtx, x: UopId) -> bool {
    ctx.op_of(&x) == crate::uop::Ops::Store && ctx.addrspace_derived(x) == Some(dtype::AddrSpace::Local)
}

/// `add_raw_barrier` (codegen/__init__.py:261): a load from a LOCAL buffer that
/// depends (via AFTER) on LOCAL stores needs a workgroup barrier.
fn add_raw_barrier(ctx: &mut UOpCtx, after: UopId) -> Option<UopId> {
    if ctx.addrspace_derived(after.clone()) != Some(dtype::AddrSpace::Local) {
        return None;
    }
    let srcs = ctx.srcs(&after).to_vec();
    let deps = crate::tensor::sink(ctx, srcs[1..].to_vec());
    let topo = ctx.toposort_gated(deps, Some(&|x| ctx.op_of(&x) != crate::uop::Ops::Barrier), true);
    if !topo.iter().any(|x| is_local_store(ctx, x.clone())) {
        return None;
    }
    let barrier = ctx.uop(crate::uop::Ops::Barrier, None, srcs[1..].to_vec(), Arg::None);
    Some(crate::tensor::after(ctx, (srcs[0]).clone(), vec![barrier]))
}

/// `add_war_barrier` (codegen/__init__.py:269): a LOCAL buffer stored and loaded
/// in the same loop needs a barrier at the end of the loop body.
fn add_war_barrier(ctx: &mut UOpCtx, end: UopId) -> Option<UopId> {
    let srcs = ctx.srcs(&end).to_vec();
    let rngs: Vec<UopId> = srcs[1..]
        .iter().filter(|&r| {
            if ctx.op_of(r) != crate::uop::Ops::Range {
                return false;
            }
            let at = AxisType::range_axis_type(ctx, r.clone());
            (at == AxisType::Reduce || at == AxisType::Weak || at == AxisType::Loop)
                && matches!(ctx.vmax(r), Ok(v) if v.clone().as_i128_or_panic() > 0)
        }).cloned()
        .collect();
    if rngs.is_empty() || ctx.op_of(&srcs[0]) == crate::uop::Ops::Barrier {
        return None;
    }
    let sl = ctx.backward_slice_with_self(&srcs[0]);
    let mut store_bufs: HashSet<UopId> = HashSet::new();
    for x in &sl {
        if is_local_store(ctx, x.clone()) {
            let xr = ctx.ranges(&x.clone());
            if rngs.iter().any(|r| xr.contains(r)) {
                store_bufs.insert(ctx.buf_uop(x.clone()));
            }
        }
    }
    let loads: Vec<UopId> = sl
        .iter().filter(|&x| {
            ctx.op_of(x) == crate::uop::Ops::Load && store_bufs.contains(&ctx.buf_uop(ctx.srcs(x)[0].clone()))
        }).cloned()
        .collect();
    if loads.is_empty() {
        return None;
    }
    let mut barrier_srcs = vec![(srcs[0]).clone()];
    barrier_srcs.extend(loads);
    let barrier = ctx.uop(crate::uop::Ops::Barrier, None, barrier_srcs, Arg::None);
    let mut new_srcs = vec![barrier];
    new_srcs.extend(srcs[1..].iter().cloned());
    let (op, dt, _, arg, tag) = ctx.snapshot(&end);
    Some(ctx.replace(&end, op, dt, new_srcs, arg, tag))
}

/// `pm_implicit_barriers` (codegen/__init__.py:280).
pub fn pm_implicit_barriers() -> PatternMatcher {
    PatternMatcher::new(vec![
        (
            UPat::ops1(crate::uop::Ops::After).allow_all_srcs().named("after"),
            r(|ctx, st, _| add_raw_barrier(ctx, sget(st, "after"))),
        ),
        (
            UPat::ops1(crate::uop::Ops::End).allow_all_srcs().named("end"),
            r(|ctx, st, _| add_war_barrier(ctx, sget(st, "end"))),
        ),
    ])
}

/// `pm_casted_consts` (codegen/__init__.py:280) — spell literals as
/// `cconst(val, dtype)` = CAST(CONST(val at from_py default), dtype).
fn pm_casted_consts() -> PatternMatcher {
    PatternMatcher::new(vec![(
        UPat::ops1(crate::uop::Ops::Const).allow_all_srcs().named("c"),
        r(|ctx, st, _| {
            let c = sget(st, "c");
            let v = ctx.val(&c);
            let d = ctx.dt_of(&c);
            if d == crate::dtype::VOID {
                return None;
            }
            // UOp.const(b) — the from_py default dtype (weakint/weakfloat/bool)
            let inner = ctx.const_py(v, None);
            Some(ctx.cast(inner, d))
        }),
    )])
}

/// `memory_coalescing` + `ew_devectorizer` + `pm_simplify_add_image` for the
/// "add images" stage (codegen/__init__.py:344-346).
fn add_images(ctx: &mut Ctx2, sink: UopId) -> UopId {
    let coalesced = crate::codegen::coalesce::memory_coalescing(ctx, sink);
    let pm = PatternMatcher::new(vec![])
        .add(symbolic_simple())
        .add(&ew_devectorizer())
        .add(&pm_simplify_add_image());
    graph_rewrite(ctx, coalesced, &pm, true, None, false, false, CtxVal::None)
        .unwrap_or_else(|e| panic!("TGERR: add images: {e}"))
}

/// `CFGContext` + `pm_add_control_flow` (late/linearizer.py:53-85) — compute
/// range nesting over the sink, then append the enclosing (or preceding
/// sibling) END/RANGE as an extra src on each `RANGE`. The extra src makes
/// `ranges()` include enclosing ranges, so `linearize` orders nested loop
/// headers outer-first (and lets the interpreter nest loops correctly).
fn add_control_flow(ctx: &mut UOpCtx, sink: UopId) -> UopId {
    use std::collections::HashSet;
    use crate::uop::ops::Ops;

    let order = ctx.toposort(&sink);
    let mut deps: HashMap<UopId, Vec<UopId>> = HashMap::new();
    let mut dep_set: HashMap<UopId, HashSet<UopId>> = HashMap::new();
    let mut nesting: Vec<(UopId, UopId)> = Vec::new();
    let mut nesting_set: HashSet<UopId> = HashSet::new();

    for u in &order {
        let mut d: Vec<UopId> = Vec::new();
        let mut ds: HashSet<UopId> = HashSet::new();
        for s in ctx.srcs(u) {
            if let Some(sd) = deps.get(s) {
                for x in sd {
                    if ds.insert(x.clone()) {
                        d.push(x.clone());
                    }
                }
            }
        }
        let op = ctx.op_of(u);
        if op == Ops::End || op == Ops::Sink {
            let endrange = if op == Ops::End { ctx.srcs(u).get(1).cloned() } else { None };
            for x in &d {
                let nested = op == Ops::Sink
                    || endrange.clone().is_some_and(|r| dep_set.get(x).is_some_and(|s| s.contains(&r)));
                if ctx.op_of(x) == Ops::End && nested && !nesting_set.contains(x) {
                    nesting_set.insert(x.clone());
                    nesting.push((x.clone(), u.clone()));
                }
            }
        }
        if (op == Ops::Range || op == Ops::End)
            && ds.insert(u.clone()) {
                d.push(u.clone());
            }
        deps.insert(u.clone(), d);
        dep_set.insert(u.clone(), ds);
    }

    // siblings[vv] preserves first-insertion order (dict semantics).
    let mut sibling_keys: Vec<UopId> = Vec::new();
    let mut siblings: HashMap<UopId, Vec<UopId>> = HashMap::new();
    for (k, vv) in &nesting {
        if !siblings.contains_key(vv) {
            sibling_keys.push(vv.clone());
        }
        siblings.entry(vv.clone()).or_default().push(k.clone());
    }

    let mut edges: HashMap<UopId, UopId> = HashMap::new();
    for k in &sibling_keys {
        let v = siblings.get(k).unwrap();
        let mut ord = v.clone();
        ord.sort_by_key(|x| v.iter().filter(|&u| dep_set.get(x).is_some_and(|s| s.contains(u))).count());
        if ord.is_empty() {
            continue;
        }
        let mut pairs: Vec<(UopId, UopId)> = Vec::new();
        if ctx.op_of(k) == Ops::Sink {
            for w in ord.windows(2) {
                pairs.push((w[0].clone(), w[1].clone()));
            }
        } else {
            pairs.push((ctx.srcs(k)[1].clone(), (ord[0]).clone()));
            for w in ord.windows(2) {
                pairs.push((w[0].clone(), w[1].clone()));
            }
        }
        for (x, y) in pairs {
            let ysrc1 = &ctx.srcs(&y)[1].clone();
            edges.insert(ysrc1.clone(), x);
        }
    }

    // The edge map is arena-specific, so it travels through the rewrite
    // `CtxVal` instead of being captured by the rule closure (capturing a
    // `UopId` would make the rule — and every cached `PatternMatcher` — non-Send).
    let edges = std::rc::Rc::new(std::cell::RefCell::new(edges));
    let pm = PatternMatcher::new(vec![(
        UPat::ops1(Ops::Range).allow_all_srcs().named("x"),
        crate::uop::upat::r_dict(|edges, ctx, st| {
            let x = sget(st, "x");
            edges.get(&x).map(|y| {
                let mut srcs = ctx.srcs(&x).to_vec();
                srcs.push(y.clone());
                let dt = ctx.dt_of(&x);
                let arg = ctx.arg_of(&x).clone();
                ctx.uop(Ops::Range, Some(dt), srcs, arg)
            })
        }),
    )]);
    graph_rewrite(ctx, sink, &pm, true, None, false, false, CtxVal::Dict(edges))
        .unwrap_or_else(|e| panic!("TGERR: add control flow: {e}"))
}

/// `ew_devectorizer` (codegen/__init__.py:143): unpack vectorized elementwise.
fn ew_devectorizer() -> PatternMatcher {
    PatternMatcher::new(vec![(
        UPat::op_var(group::ELEMENTWISE.to_vec()).named("b"),
        r(|ctx, st, _| do_devectorize(ctx, sget(st, "b"))),
    )])
}

/// `pm_simplify_add_image` (late/coalesce.py:95). The image rule is a no-op
/// here (IMAGE is off and this port has no image support); the float/half
/// cast rules are renderer-independent.
fn pm_simplify_add_image() -> PatternMatcher {
    PatternMatcher::new(vec![
        // SHRINK(PARAM, x, 4) -> image buffer (transform_to_image); IMAGE off
        (
            UPat::ops1(crate::uop::Ops::Shrink).with_src_tup(vec![
                UPat::ops1(crate::uop::Ops::Param).named("buf"),
                UPat::var(Some("x"), None),
                UPat::const_pat(crate::dtype::ConstVal::Int(4), None),
            ]),
            r(|_ctx, _st, _| None),
        ),
        // INDEX(float).load(half) -> INDEX(float).load().cast(half)
        (
            UPat::ops1(crate::uop::Ops::Load)
                .with_dtype(Some(vec![crate::dtype::FLOAT16]))
                .with_src_tup(vec![UPat::ops1(crate::uop::Ops::Index)
                    .with_dtype(Some(vec![crate::dtype::FLOAT32]))
                    .named("x")]),
            r(|ctx, st, _| {
                let x = sget(st, "x");
                let ld = crate::tensor::load(ctx, x, vec![]);
                Some(ctx.cast(ld, crate::dtype::FLOAT16))
            }),
        ),
        // INDEX(float).store(d, half) -> store(d.cast(float))
        (
            UPat::ops1(crate::uop::Ops::Store).with_src_tup(vec![
                UPat::ops1(crate::uop::Ops::Index)
                    .with_dtype(Some(vec![crate::dtype::FLOAT32]))
                    .named("x"),
                UPat::var(Some("d"), Some(vec![crate::dtype::FLOAT16])),
            ]),
            r(|ctx, st, _| {
                let (x, d) = (sget(st, "x"), sget(st, "d"));
                let df = ctx.cast(d, crate::dtype::FLOAT32);
                Some(crate::tensor::store(ctx, x, df, None))
            }),
        ),
        // f32.cast(f16).cast(f32) -> f32
        (
            UPat::var(Some("x"), Some(vec![crate::dtype::FLOAT32]))
                .cast(Some(crate::dtype::FLOAT16))
                .cast(Some(crate::dtype::FLOAT32)),
            r(|_ctx, st, _| Some(sget(st, "x"))),
        ),
    ])
}

/// `do_devectorize` (codegen/__init__.py:107).
fn do_devectorize(ctx: &mut Ctx2, b: UopId) -> Option<UopId> {
    let shape = ctx.shape(&b).ok()?.as_ref().map(|v| v.as_ref().clone()).unwrap_or_default();
    if shape.is_empty() {
        return None;
    }
    let srcs = ctx.srcs(&b).to_vec();
    // all srcs must have b's shape (or be INVALID consts)
    for x in &srcs {
        if ctx.is_invalid(&ctx.base(x)) {
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
        .map(|d| {
            let n = match d {
                SVal::I(v) => *v as i64,
                _ => panic!("TGERR: symbolic devectorize"),
            };
            (0..n).map(|i| ctx.const_py(crate::dtype::ConstVal::Int(i as i128), None)).collect()
        })
        .collect();
    let mut elems: Vec<UopId> = Vec::new();
    for idx_c in cartesian(&idx_lists) {
        let mut new_srcs = Vec::with_capacity(srcs.len());
        for x in &srcs {
            if ctx.is_invalid(&ctx.base(x)) {
                new_srcs.push(ctx.base(x));
            } else {
                // python: `x.index(*idx_c)` — must route through the UOp.index
                // fold so `STACK[single_const]` collapses to its child
                new_srcs.push(ctx.index(x.clone(), idx_c.clone()));
            }
        }
        elems.push(ctx.uop(ctx.op_of(&b), Some(ctx.dt_of(&b)), new_srcs, ctx.arg_of(&b).clone()));
    }
    if ctx.op_of(&b) == crate::uop::Ops::Store {
        Some(ctx.uop(crate::uop::Ops::Group, Some(crate::dtype::VOID), elems, Arg::None))
    } else {
        let stack = ctx.uop(crate::uop::Ops::Stack, Some(ctx.dt_of(&b)), elems, Arg::None);
        Some(crate::tensor::movement::reshape(ctx, stack, &shape))
    }
}

/// cartesian product over per-dim const lists.
fn cartesian(lists: &[Vec<UopId>]) -> Vec<Vec<UopId>> {
    let mut ret: Vec<Vec<UopId>> = vec![vec![]];
    for l in lists {
        let mut next = Vec::new();
        for combo in &ret {
            for x in l {
                let mut c = combo.clone();
                c.push(x.clone());
                next.push(c);
            }
        }
        ret = next;
    }
    ret
}

/// `devectorizer2` (codegen/__init__.py:139) — mop_cleanup+pm_mops+rules
/// (no WMMA in corpus).
fn devectorizer2() -> PatternMatcher {
    let targets: Vec<crate::uop::Ops> = group::ELEMENTWISE
        .iter()
        .cloned()
        .chain([crate::uop::Ops::Load, crate::uop::Ops::Store])
        .collect();
    PatternMatcher::new(vec![
        // unpack broadcasting
        (
            UPat::op_var(targets).named("b"),
            r(|ctx, st, _| do_devectorize(ctx, sget(st, "b"))),
        ),
        // INDEX with a single src is nothing (python: `INDEX(src=(x,)) -> x`)
        (
            UPat::ops1(crate::uop::Ops::Index)
                .with_src_tup(vec![UPat::var(None, None)])
                .named("x"),
            r(|ctx, st, _| Some(ctx.srcs(&sget(st, "x"))[0].clone())),
        ),
        // stacked INDEX is many INDEX
        (
            UPat::ops1(crate::uop::Ops::Index)
                .with_src_tup(vec![UPat::op_var(vec![crate::uop::Ops::Param, crate::uop::Ops::Buffer]).named("b"), UPat::ops1(crate::uop::Ops::Stack).named("s")])
                .named("x"),
            r(|ctx, st, _| {
                let b = sget(st, "b");
                let x = sget(st, "x");
                let ss = ctx.srcs(&sget(st, "s")).to_vec();
                let mut elems = Vec::new();
                for u in &ss {
                    let mut srcs = vec![b.clone(), u.clone()];
                    srcs.extend(ctx.srcs(&x)[2..].iter().cloned());
                    elems.push(ctx.uop(crate::uop::Ops::Index, Some(ctx.dt_of(&x)), srcs, ctx.arg_of(&x).clone()));
                }
                Some(ctx.uop(crate::uop::Ops::Stack, Some(ctx.dt_of(&x)), elems, ctx.arg_of(&x).clone()))
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
                let shape = ctx.shape(&s).ok()?.as_ref().map(|v| v.as_ref().clone()).unwrap_or_default();
                let inner = &ctx.srcs(&s)[0].clone();
                let idx = ctx.uop(crate::uop::Ops::Index, Some(ctx.dt_of(&b)), vec![b, inner.clone()], Arg::None);
                Some(crate::tensor::movement::reshape(ctx, idx, &shape))
            }),
        ),
        // RESHAPE a void is removed
        (
            UPat::ops1(crate::uop::Ops::Reshape).with_dtype(Some(vec![crate::dtype::VOID])).named("x"),
            r(|ctx, st, _| Some(ctx.srcs(&sget(st, "x"))[0].clone())),
        ),
        // reshape of a single element shaped value to scalar is an index
        (
            UPat::ops1(crate::uop::Ops::Reshape).named("x"),
            r(|ctx, st, _| {
                let x = sget(st, "x");
                let shape = ctx.shape(&x).ok()?.as_ref().map(|v| v.as_ref().clone()).unwrap_or_default();
                let src = &ctx.srcs(&x)[0].clone();
                let src_shape = ctx.shape(src).ok()?.as_ref().map(|v| v.as_ref().clone()).unwrap_or_default();
                if shape.is_empty() && src_shape == vec![SVal::I(1)] {
                    let zero = ctx.const_py(crate::dtype::ConstVal::Int(0), None);
                    return Some(ctx.uop(crate::uop::Ops::Index, Some(ctx.dt_of(src)), vec![src.clone(), zero], Arg::None));
                }
                None
            }),
        ),
        // EXPAND on scalar -> STACK
        (
            UPat::ops1(crate::uop::Ops::Expand).with_src_any_len(true).named("out"),
            r(|ctx, st, _| {
                let out = sget(st, "out");
                let x = &ctx.srcs(&out)[0].clone();
                let xs = ctx.shape(x).ok()?.as_ref().map(|v| v.as_ref().clone()).unwrap_or_default();
                let os = ctx.shape(&out).ok()?.as_ref().map(|v| v.as_ref().clone()).unwrap_or_default();
                if !xs.is_empty() || os.len() != 1 {
                    return None;
                }
                let n = match os[0] {
                    SVal::I(v) => v as i64,
                    _ => return None,
                };
                // out.shape == (out.max_numel(),) — 1-D is its own max shape
                let copies = vec![x.clone(); n as usize];
                Some(ctx.uop(crate::uop::Ops::Stack, Some(ctx.dt_of(&out)), copies, Arg::None))
            }),
        ),
    ])
    .add(pm_mop_cleanup())
    .add(&crate::schedule::rangeify::pm_mops())
}

/// `build_range_map` (codegen/__init__.py:39).
fn build_range_map(ctx: &mut Ctx2, sink: UopId) -> HashMap<i64, usize> {
    let mut map = HashMap::new();
    for x in ctx.toposort(&sink).iter() {
        if ctx.op_of(x) == crate::uop::Ops::Range
            && let Arg::Seq { tuple: true, items } = ctx.arg_of(x)
                && let Arg::Enum(e) = &items[1]
                    && matches!(e.value, 9 | 8) {
                        // UNROLL | UPCAST
                        let axis = match &items[0] {
                            Arg::Int(i) => *i,
                            _ => continue,
                        };
                        let n = map.len();
                        map.entry(axis).or_insert(n);
                    }
    }
    map
}

/// `expander2` (codegen/__init__.py:74) — corpus subset (no REDUCE lift, no WMMA).
fn expander_pm() -> PatternMatcher {
    // Template form (range map empty) so the rule inventory can see the same
    // rules the runtime `expander_pm_with` binds. The closures are only invoked
    // via `expander_pm_with`; a stray invocation here would just find no ranges.
    PatternMatcher::new(expander_rules(HashMap::new()))
        .add(&pm_flatten_range())
        .add(pm_mop_cleanup())
}

/// The `expander2` rules with the range map bound into the RANGE rule.
fn expander_rules(rm: HashMap<i64, usize>) -> Vec<(UPat, RuleFn)> {
    vec![
        (
            // `expand_reduce` (codegen/__init__.py:49): fold non-range srcs'
            // shape>1 axes into the reduce as leading axes.
            UPat::ops1(crate::uop::Ops::Reduce).named("r"),
            r(|ctx: &mut Ctx2, st, _| expand_reduce(ctx, sget(st, "r"))),
        ),
        (
            UPat::ops1(crate::uop::Ops::Range).named("r"),
            r(move |ctx: &mut Ctx2, st, _| {
                let r = sget(st, "r");
                let (op, dt, srcs, arg, _tag) = ctx.snapshot(&r);
                let _ = op;
                let Arg::Seq { tuple: true, items } = &arg else { return None };
                let axis = match &items[0] {
                    Arg::Int(i) => *i,
                    _ => return None,
                };
                let pos = *rm.get(&axis)?;
                let vmax = match ctx.ssimplify((srcs[0]).clone()) {
                    Ok(SVal::I(v)) => v as i64 - 1,
                    _ => return None,
                };
                // UOp.const(tuple(range(vmax+1)), r.dtype).reshape(...)
                let consts: Vec<UopId> = (0..=vmax).map(|i| ctx.const_py(crate::dtype::ConstVal::Int(i as i128), Some(dt))).collect();
                let stack = ctx.uop(crate::uop::Ops::Stack, Some(dt), consts, Arg::None);
                let shape: Vec<SVal> = (0..rm.len()).map(|i| SVal::I(if i == pos { vmax + 1 } else { 1 } as i128)).collect();
                Some(crate::tensor::movement::reshape(ctx, stack, &shape))
            }),
        ),
    ]
}

/// expander with the range map bound.
fn expander_pm_with(ctx: &mut Ctx2, sink: UopId) -> (PatternMatcher, HashMap<i64, usize>) {
    let range_map = build_range_map(ctx, sink);
    let pm = PatternMatcher::new(expander_rules(range_map.clone()))
        .add(&pm_flatten_range())
        .add(pm_mop_cleanup());
    (pm, range_map)
}

/// `expand_reduce` (codegen/__init__.py:49-64).
fn expand_reduce(ctx: &mut Ctx2, r: UopId) -> Option<UopId> {
    let (_op, dt, srcs, arg, _tag) = ctx.snapshot(&r);
    let s0 = srcs[0].clone();
    let mut range_srcs: Vec<UopId> = Vec::new();
    let mut new_axes: Vec<usize> = Vec::new();
    for u in &srcs[1..] {
        if ctx.op_of(u) == crate::uop::Ops::Range {
            range_srcs.push(u.clone());
        } else if let Ok(shape) = ctx.shape(u) {
            for (i, s) in shape.unwrap_or_default().iter().enumerate() {
                if !matches!(s, crate::uop::sint::SVal::I(1)) {
                    new_axes.push(i);
                }
            }
        }
    }
    if new_axes.is_empty() {
        return None;
    }
    // assert r.arg[1] == 0 and keep items[0] (the reduce op)
    let red_op: Arg = match &arg {
        Arg::Seq { tuple: true, items } => {
            match &items[1] {
                Arg::Int(0) => {}
                other => panic!("TGERR: expand_reduce expects arg[1] == 0, got {other:?}"),
            }
            items[0].clone()
        }
        other => panic!("TGERR: expand_reduce bad arg {other:?}"),
    };
    let s0_shape: Vec<crate::uop::sint::SVal> = ctx.shape(&s0).unwrap().unwrap_or_default().as_ref().clone();
    let perm: Vec<usize> = new_axes
        .iter()
        .cloned()
        .chain((0..s0_shape.len()).filter(|i| !new_axes.contains(i)))
        .collect();
    let permuted = crate::tensor::movement::permute_internal(ctx, s0, &perm);
    let red_arg = Arg::Seq {
        tuple: true,
        items: vec![red_op, Arg::Int(new_axes.len() as i64)],
    };
    let red = ctx.uop(crate::uop::Ops::Reduce, Some(dt), [vec![permuted], range_srcs].concat(), red_arg);
    let out_shape: Vec<crate::uop::sint::SVal> = s0_shape
        .iter()
        .enumerate()
        .map(|(i, s)| if new_axes.contains(&i) { crate::uop::sint::SVal::I(1) } else { s.clone() })
        .collect();
    Some(crate::tensor::movement::reshape_internal(ctx, red, &out_shape))
}

/// `expand_broadcast` (codegen/__init__.py:97).
fn expand_broadcast(ctx: &mut Ctx2, x: UopId) -> Option<UopId> {
    let srcs = ctx.srcs(&x).to_vec();
    let mut shapes: Vec<Vec<SVal>> = Vec::new();
    for u in &srcs {
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
    for u in &srcs {
        new_srcs.push(crate::tensor::movement::expand(ctx, u.clone(), &shape));
    }
    Some(ctx.uop(ctx.op_of(&x), Some(ctx.dt_of(&x)), new_srcs, ctx.arg_of(&x).clone()))
}

/// `pm_expand_broadcast` (codegen/__init__.py:106) — corpus subset (no WMMA).
fn pm_expand_broadcast() -> PatternMatcher {
    PatternMatcher::new(vec![(
        UPat::op_var(
            group::BINARY
                .iter()
                .cloned()
                .chain(group::TERNARY.iter().cloned())
                .chain([crate::uop::Ops::Store])
                .collect(),
        )
        .named("x"),
        r(|ctx, st, _| expand_broadcast(ctx, sget(st, "x"))),
    )])
}

/// `maybe_load` (codegen/__init__.py:245).
fn maybe_load(ctx: &mut Ctx2, u: UopId) -> UopId {
    match ctx.addrspace_derived(u.clone()) {
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
        .cloned()
        .chain([crate::uop::Ops::Reduce, crate::uop::Ops::Wmma, crate::uop::Ops::Stack])
        .collect();
    PatternMatcher::new(vec![
        (
            UPat::op_var(elementwise_like).named("x"),
            r(|ctx, st, _| {
                let x = sget(st, "x");
                let srcs = ctx.srcs(&x).to_vec();
                let new_srcs: Vec<UopId> = srcs.iter().map(|s| maybe_load(ctx, s.clone())).collect();
                if new_srcs == srcs {
                    return None;
                }
                Some(ctx.uop(ctx.op_of(&x), Some(ctx.dt_of(&x)), new_srcs, ctx.arg_of(&x).clone()))
            }),
        ),
        (
            UPat::ops1(crate::uop::Ops::Store).named("x"),
            r(|ctx, st, _| {
                let x = sget(st, "x");
                let srcs = ctx.srcs(&x).to_vec();
                let loaded = maybe_load(ctx, (srcs[1]).clone());
                if loaded == srcs[1] {
                    return None;
                }
                let mut new_srcs = vec![(srcs[0]).clone(), loaded];
                new_srcs.extend(srcs[2..].iter().cloned());
                Some(ctx.uop(crate::uop::Ops::Store, Some(ctx.dt_of(&x)), new_srcs, ctx.arg_of(&x).clone()))
            }),
        ),
    ])
}

/// `full_rewrite_to_sink` (codegen/__init__.py:288) — partial: the opening
/// mops/symbolic block. Returns the rewritten SINK.
pub fn full_rewrite_to_sink(ctx: &mut UOpCtx, ast: UopId) -> UopId {
    full_rewrite_to_sink_with(ctx, ast, &Renderer::cpu_interp())
}

/// temporary: readable one-line-per-node dump of the graph below `sink`.
#[cfg(feature = "dump")]
fn dump_ast(ctx: &mut UOpCtx, sink: UopId) -> String {
    let order = ctx.toposort(&sink);
    let mut out = String::new();
    for n in order.iter() {
        let (op, dt, srcs, arg, _tag) = ctx.snapshot(n);
        let dt_s = format!("{dt:?}");
        let src_s: Vec<String> = srcs.iter().map(|s| s.id().to_string()).collect();
        let shape = ctx.shape(n).ok().flatten().map(|s| format!("{s:?}")).unwrap_or_default();
        out.push_str(&format!(
            "{} = {:?}{} ({}) src=[{}] arg={:?}\n",
            n.id(),
            op,
            dt_s,
            shape,
            src_s.join(","),
            arg
        ));
    }
    out
}

pub fn full_rewrite_to_sink_with(ctx: &mut UOpCtx, ast: UopId, ren: &Renderer) -> UopId {
    let limit = stage_limit();
    let mut i = 0usize;
    let mut cur = ast;
    for st in stages() {
        if i + 1 > limit {
            return cur;
        }
        let ctxv = if st.name == "split ranges" {
            CtxVal::Dict(std::rc::Rc::new(std::cell::RefCell::new(HashMap::new())))
        } else {
            CtxVal::None
        };
        cur = graph_rewrite(ctx, cur, &st.pm, st.bottom_up, None, false, false, ctxv)
            .unwrap_or_else(|e| panic!("TGERR: {}: {e}", st.name));
        if st.name == "simplify ranges" {
            cur = crate::codegen::simplify::range_gates(ctx, cur);
        }
        i += 1;
        // temporary: dump the pre-expander AST (TG_DUMP_PRE_EXPAND=1)
        #[cfg(feature = "dump")]
        if std::env::var("TG_DUMP_PRE_EXPAND").as_deref() == Ok("1") {
            eprintln!("=== after stage {} ({}) ===", i, st.name);
            eprintln!("{}", dump_ast(ctx, cur.clone()));
        }
    }
    if i + 1 > limit {
        return cur;
    }
    // do postrange optimization, BEAM or hand_coded_optimizations
    cur = apply_opts(ctx, cur, ren);
    i += 1;
    #[cfg(feature = "dump")]
    if std::env::var("TG_DUMP_PRE_EXPAND").as_deref() == Ok("1") {
        eprintln!("=== after stage {} (apply_opts) ===", i);
        eprintln!("{}", dump_ast(ctx, cur.clone()));
    }
    for st in post_stages(ren) {
        if i + 1 > limit {
            return cur;
        }
        cur = if st.name == "expander" {
            let (pm, _rm) = expander_pm_with(ctx, cur.clone());
            graph_rewrite(ctx, cur, &pm, st.bottom_up, None, false, false, CtxVal::None)
                .unwrap_or_else(|e| panic!("TGERR: {}: {e}", st.name))
        } else if st.name == "remove reduces" || st.name == "add local buffers" {
            graph_rewrite(ctx, cur, &st.pm, st.bottom_up, None, false, false, CtxVal::Cnt(std::rc::Rc::new(std::cell::Cell::new(0i64))))
                .unwrap_or_else(|e| panic!("TGERR: {}: {e}", st.name))
        } else if st.name == "add images" {
            add_images(ctx, cur)
        } else if st.name == "decomp dtypes" {
            let dctx = crate::codegen::decomp_dtype::DtypeDecompCtx {
                dtypes: HashSet::new(),
                supported: ren.supported_dtypes.clone(),
                emulated: crate::codegen::decomp_dtype::emulated_dtypes(),
            };
            graph_rewrite(ctx, cur, &st.pm, st.bottom_up, None, false, false,
                          CtxVal::Decomp(std::rc::Rc::new(std::cell::RefCell::new(dctx))))
                .unwrap_or_else(|e| panic!("TGERR: {}: {e}", st.name))
        } else if st.name == "add control flow" {
            add_control_flow(ctx, cur)
        } else {
            graph_rewrite(ctx, cur, &st.pm, st.bottom_up, None, false, false, CtxVal::None)
                .unwrap_or_else(|e| panic!("TGERR: {}: {e}", st.name))
        };
        i += 1;
        #[cfg(feature = "dump")]
        if std::env::var("TG_DUMP_IDX").is_ok() {
            crate::codegen::postrange::dump_index_exprs(ctx, cur.clone(), st.name);
        }
        #[cfg(feature = "dump")]
        if std::env::var("TG_DUMP_PRE_EXPAND").as_deref() == Ok("1") {
            eprintln!("=== after stage {} ({}) ===", i, st.name);
            eprintln!("{}", dump_ast(ctx, cur.clone()));
        }
    }
    cur
}

/// Rule-parity inventory: every codegen pipeline `PatternMatcher`, with one
/// canonical signature per rule (order-sensitive) to diff against tinygrad.
pub fn pm_inventory() -> Vec<(&'static str, Vec<String>)> {
    let mut v: Vec<(&'static str, Vec<String>)> = Vec::new();
    macro_rules! add {
        ($name:literal, $pm:expr) => {{
            let pm = $pm;
            v.push(($name, pm.rule_sigs()));
        }};
    }
    add!("symbolic_simple", symbolic_simple());
    add!("symbolic", symbolic());
    add!("sym", sym());
    add!("pm_simplify_valid", crate::uop::sym::pm_simplify_valid());
    add!("pm_commutative", crate::uop::sym::pm_commutative());
    add!("pm_sym_extra", crate::uop::sym::pm_sym_extra());
    add!("mop_cleanup", pm_mop_cleanup());
    add!("devectorizer2", devectorizer2());
    add!("ew_devectorizer", ew_devectorizer());
    add!("expander2", expander_pm());
    add!("pm_expand_broadcast", pm_expand_broadcast());
    add!("pm_add_loads", pm_add_loads());
    add!("pm_reduce_local", crate::codegen::simplify::pm_reduce_local());
    // tinygrad: `pm_add_local_buffers = PatternMatcher([...])+pm_mops`
    add!("pm_add_local_buffers", crate::schedule::rangeify::pm_add_local_buffers().add(&pm_mops()));
    add!("pm_add_gpudims", crate::codegen::gpudims::pm_add_gpudims());
    add!("indexing_simplify", crate::codegen::coalesce::indexing_simplify());
    add!("pm_simplify_add_image", pm_simplify_add_image());
    // the bridge differentials use the interpreter op set (python_alu contains MAX)
    add!("pm_decomp", crate::codegen::decomp::pm_decomp(true));
    add!("pm_dtype_decomps", crate::codegen::decomp_dtype::pm_dtype_decomps());
    add!("pm_move_gates_from_index", crate::codegen::gater::pm_move_gates_from_index());
    add!("pm_cast_float_alu", pm_cast_float_alu());
    add!("pm_commit_weak", crate::uop::weak::pm_commit_weak());
    add!("pm_cast_weak", crate::uop::weak::pm_cast_weak());
    add!("pm_lower_index_dtype", crate::uop::weak::pm_lower_index_dtype());
    add!("pm_casted_consts", pm_casted_consts());
    add!("pm_implicit_barriers", pm_implicit_barriers());
    add!("pm_split_ends", pm_split_ends());
    add!("pm_move_where_on_load", pm_move_where_on_load());
    add!("pm_flatten_range", pm_flatten_range());
    add!("pm_split_ranges", pm_split_ranges());
    add!("pm_reduce_unparented", pm_reduce_unparented());
    add!("pm_reduce_load_collapse", crate::codegen::simplify::pm_reduce_load_collapse());
    add!("pm_load_collapse", pm_load_collapse());
    add!("pm_mops", pm_mops());
    v
}
