//! `tensor.py` callify port (transform_to_call): number uops with tags,
//! materialize CONTIGUOUS/AFTER values into buffers, and extract the call
//! boundary (assigns + PARAM replacements) for the scheduler.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::dtype::{self, AddrSpace, ConstVal};
use crate::uop::arg::{Arg, ParamArg};
use crate::uop::ctx::UOpCtx;
use crate::uop::ops::{group, Ops};
use crate::uop::upat::{graph_rewrite, r_al, substitute, CtxVal, PatternMatcher, RuleFn, UPat};
use crate::uop::UopId;

/// `AllocCtx` (tensor.py:20).
#[derive(Default)]
pub struct AllocCtx {
    pub uop_list: Vec<UopId>,
    pub buffer_map: Vec<(UopId, UopId)>, // insertion-ordered
    pub bases: Vec<UopId>,
    pub assigns: Vec<UopId>,
    pub replacements: Vec<UopId>,
    pub views: Vec<UopId>,
}

impl AllocCtx {
    pub fn buffer_map_get(&self, k: UopId) -> Option<UopId> {
        self.buffer_map.iter().find(|(a, _)| *a == k).map(|(_, v)| *v)
    }
    pub fn buffer_map_insert(&mut self, k: UopId, v: UopId) {
        if self.buffer_map_get(k).is_none() {
            self.buffer_map.push((k, v));
        }
    }
    pub fn in_bases(&self, k: UopId) -> bool {
        self.bases.contains(&k)
    }
    pub fn in_views(&self, k: UopId) -> bool {
        self.views.contains(&k)
    }
}

/// `tag_uop` — tag=(n,) sequential numbering.
fn tag_uop(actx: &mut AllocCtx, ctx: &mut UOpCtx, x: UopId) -> Option<UopId> {
    if !matches!(ctx.tag_of(x), Arg::None) {
        return None;
    }
    actx.uop_list.push(x);
    let n = actx.uop_list.len() - 1;
    let tag = Arg::Seq { tuple: true, items: vec![Arg::Int(n as i64)] };
    Some(ctx.uop_tag(ctx.op_of(x), Some(ctx.dt_of(x)), ctx.srcs(x).to_vec(), ctx.arg_of(x).clone(), tag))
}

fn device_str_of(ctx: &mut UOpCtx, u: UopId) -> Option<String> {
    match crate::schedule::indexing::device_of(ctx, u) {
        Some(crate::uop::arg::DeviceSpec::Single(s)) => Some(s),
        _ => None,
    }
}

fn disk_like(ctx: &mut UOpCtx, u: UopId) -> bool {
    match device_str_of(ctx, u) {
        Some(d) => d.starts_with("DISK") || d.starts_with("TINYFS"),
        None => false,
    }
}

/// `disk_copy_is_buffer` (tensor.py:36).
fn disk_copy_is_buffer(actx: &mut AllocCtx, ctx: &mut UOpCtx, u: UopId) -> Option<UopId> {
    let tagged_none = matches!(ctx.tag_of(u), Arg::None);
    if disk_like(ctx, u) && tagged_none {
        let el = empty_like(ctx, u);
        actx.buffer_map_insert(u, el);
        return Some(ctx.uop_tag(ctx.op_of(u), Some(ctx.dt_of(u)), ctx.srcs(u).to_vec(), ctx.arg_of(u).clone(), Arg::Seq { tuple: true, items: vec![] }));
    }
    // copies from disk/numpy are realized into a real buffer
    let s0 = ctx.srcs(u)[0];
    let from_creation = matches!(device_str_of(ctx, s0), Some(d) if d.starts_with("NPY") || d.starts_with("DISK") || d.starts_with("PYTHON") || d.starts_with("TINYFS"));
    if from_creation {
        tag_uop(actx, ctx, u)
    } else {
        None
    }
}

/// `UOp.empty_like` (ops.py:823) — canonicalized device, strong dtype.
pub fn empty_like(ctx: &mut UOpCtx, u: UopId) -> UopId {
    let dev = crate::schedule::indexing::device_of(ctx, u);
    let dev_s: Option<String> = match dev {
        Some(crate::uop::arg::DeviceSpec::Single(s)) => Some(s),
        Some(crate::uop::arg::DeviceSpec::Multi(_)) => panic!("TGERR: multi-device empty_like unsupported"),
        None => None,
    };
    let canon = super::creation::canonicalize_device(dev_s.as_deref());
    let shape = ctx.shape(u).expect("TGERR: empty_like shape").unwrap_or_default();
    let dt = crate::dtype::strong_dtype(ctx.dt_of(u));
    crate::tensor::creation::empty(ctx, &shape, dt, &canon)
}

/// `replace_contig_with_store_after` (tensor.py:62).
fn replace_contig_with_store_after(actx: &mut AllocCtx, ctx: &mut UOpCtx, u: UopId) -> Option<UopId> {
    // can't allocate a buffer for a virtual value
    let is_virtual = crate::schedule::indexing::device_of(ctx, u).is_none() || dtype::is_weak(ctx.dt_of(u));
    if is_virtual {
        return None;
    }
    let shape = ctx.shape(u).ok().flatten()?;
    if shape.iter().any(|s| s.as_i64() == Some(0)) {
        return Some(ctx.srcs(u)[0]);
    }
    if disk_like(ctx, u) {
        return Some(ctx.uop_tag(Ops::Contiguous, Some(ctx.dt_of(u)), ctx.srcs(u).to_vec(), ctx.arg_of(u).clone(), Arg::None));
    }
    let buf = empty_like(ctx, u);
    let st = crate::tensor::store(ctx, buf, ctx.srcs(u)[0], None);
    let aft = crate::tensor::after(ctx, buf, vec![st]);
    let tag = ctx.tag_of(u).clone();
    Some(ctx.uop_tag(Ops::After, Some(ctx.dt_of(aft)), ctx.srcs(aft).to_vec(), ctx.arg_of(aft).clone(), tag))
}

/// `replace_store_after_with_contig` (tensor.py:71).
fn replace_store_after_with_contig(actx: &mut AllocCtx, ctx: &mut UOpCtx, u: UopId) -> Option<UopId> {
    let _ = actx;
    let mut assigned_to = u;
    loop {
        let op = ctx.op_of(assigned_to);
        if matches!(op, Ops::Bitcast | Ops::After | Ops::Unshard) {
            let s0 = ctx.srcs(assigned_to)[0];
            assigned_to = ctx.base(s0);
        } else {
            break;
        }
    }
    if ctx.op_of(assigned_to) == Ops::Buffer {
        return None;
    }
    // python: return src.contiguous(tag=u.tag) where src is the STORE's value
    let store = ctx.srcs(u)[1];
    let val = ctx.srcs(store)[1];
    let tag = ctx.tag_of(u).clone();
    let contig = ctx.uop_tag(Ops::Contiguous, Some(ctx.dt_of(val)), vec![val], Arg::None, tag);
    Some(contig)
}

/// `contiguous_mops_to_view` (tensor.py:78) — MOPS(BUFFER) → SHRINK view.
fn contiguous_mops_to_view(actx: &mut AllocCtx, ctx: &mut UOpCtx, c: UopId, src: UopId) -> Option<UopId> {
    let mut buf = ctx.base(src);
    while ctx.op_of(buf) == Ops::Bitcast {
        buf = ctx.base(ctx.srcs(buf)[0]);
    }
    if !matches!(ctx.op_of(buf), Ops::Buffer | Ops::Unshard) {
        return None;
    }
    let c_shape = ctx.shape(c).ok().flatten()?;
    if !c_shape.iter().all(|s| matches!(s, crate::uop::sint::SVal::I(_))) {
        return None;
    }
    if ctx.op_of(buf) != Ops::Unshard {
        if let Some((view0, offset)) = buffer_view(ctx, src) {
            actx.views.push(view0);
            let view = crate::tensor::movement::reshape(ctx, view0, &c_shape);
            let c_op = ctx.op_of(c);
            if matches!(c_op, Ops::Copy | Ops::Store) {
                let (op, dt, _srcs, arg, tag) = ctx.snapshot(c);
                let mut new_srcs = vec![view];
                let csrcs = ctx.srcs(c)[1..].to_vec();
                new_srcs.extend(csrcs);
                return Some(ctx.uop_tag(op, Some(dt), new_srcs, arg, tag));
            }
            return Some(view);
        }
        return None;
    }
    None // UNSHARD: multi-device, skipped (corpus is single-device)
}

/// `_make_buffer_view` (tensor.py:81) — (view_uop, offset).
fn buffer_view(ctx: &mut UOpCtx, src: UopId) -> Option<(UopId, i64)> {
    let (buf, offset) = ctx.contiguous_view(src)?;
    if ctx.op_of(buf) != Ops::Buffer {
        return None;
    }
    let src_numel = ctx.max_numel(src);
    let src_es = ctx.dt_of(src).itemsize() as i64;
    let buf_es = ctx.dt_of(buf).itemsize() as i64;
    let size = src_numel * src_es / buf_es;
    // buf[offset:offset+size].bitcast(src.dtype) — shrink view of the BUFFER
    let view = crate::tensor::movement::shrink(ctx, buf, &[(crate::uop::sint::SVal::I(offset as i128), crate::uop::sint::SVal::I(size as i128))]);
    let bitcast = ctx.bitcast(view, ctx.dt_of(src));
    Some((bitcast, offset))
}

/// `add_tags` (tensor.py:45).
pub fn add_tags() -> PatternMatcher {
    let rules: Vec<(UPat, RuleFn)> = vec![
        (UPat::ops1(Ops::Copy).allow_all_srcs().named("u"), r_al(|a, ctx, st| disk_copy_is_buffer(a, ctx, sym_sget(st, "u")))),
        (
            UPat::ops1(Ops::After)
                .with_src_tup(vec![
                    UPat::var(None, None),
                    UPat::ops1(Ops::Store).with_src_tup(vec![UPat::var(Some("dest"), None), UPat::ops1(Ops::Copy).named("c")]),
                ])
                .named("a"),
            r_al(|_a, ctx, st| {
                let (a, c, dest) = (sym_sget(st, "a"), sym_sget(st, "c"), sym_sget(st, "dest"));
                let a_tag = ctx.tag_of(a).clone();
                let c_tag = ctx.tag_of(c).clone();
                let (a_has, c_has) = (tag_nonempty(&a_tag), tag_nonempty(&c_tag));
                if a_has && c_has {
                    // merge COPY tag into AFTER: src[1] becomes STORE(dest, rtag(()) copy)
                    let c_r = ctx.uop_tag(Ops::Copy, Some(ctx.dt_of(c)), ctx.srcs(c).to_vec(), ctx.arg_of(c).clone(), Arg::Seq { tuple: true, items: vec![] });
                    let st2 = ctx.uop_tag(Ops::Store, Some(ctx.dt_of(c_r)), vec![dest, c_r], Arg::None, Arg::None);
                    let merged = merge_tags(&a_tag, &c_tag);
                    return Some(ctx.uop_tag(Ops::After, Some(ctx.dt_of(a)), vec![ctx.srcs(a)[0], st2], ctx.arg_of(a).clone(), merged));
                }
                None
            }),
        ),
        (UPat::op_var(vec![Ops::Contiguous, Ops::After]).named("x"), r_al(|a, ctx, st| tag_uop(a, ctx, sym_sget(st, "x")))),
        (
            UPat::op_var(crate::uop::ops::OPS_ALL.to_vec()).named("x"),
            r_al(|a, ctx, st| {
                let x = sym_sget(st, "x");
                if a.in_bases(x) {
                    tag_uop(a, ctx, x)
                } else {
                    None
                }
            }),
        ),
    ];
    PatternMatcher::new(rules)
}

fn tag_nonempty(tag: &Arg) -> bool {
    matches!(tag, Arg::Seq { tuple: true, items } if !items.is_empty())
}

fn merge_tags(a: &Arg, b: &Arg) -> Arg {
    let mut items = Vec::new();
    if let Arg::Seq { tuple: true, items: ia } = a {
        items.extend(ia.iter().cloned());
    }
    if let Arg::Seq { tuple: true, items: ib } = b {
        items.extend(ib.iter().cloned());
    }
    Arg::Seq { tuple: true, items }
}

fn sym_sget(st: &crate::uop::upat::Store, name: &str) -> UopId {
    crate::uop::sym::sget(st, name)
}

/// `finalize_after` (tensor.py:190).
fn finalize_after(actx: &mut AllocCtx, ctx: &mut UOpCtx, x: UopId) -> Option<UopId> {
    if ctx.is_bound_var(x) {
        return None;
    }
    let tag = ctx.tag_of(x).clone();
    if matches!(tag, Arg::None) {
        actx.assigns.push(x);
        return None;
    }
    // tagged: untag, map each original tagged uop to the stripped buffer
    let ret = ctx.uop_tag(Ops::After, Some(ctx.dt_of(x)), ctx.srcs(x).to_vec(), ctx.arg_of(x).clone(), Arg::None);
    let mut replace_uop = ret;
    let mut views: Vec<UopId> = Vec::new();
    loop {
        let op = ctx.op_of(replace_uop);
        if group::in_(group::MOVEMENT, op) || matches!(op, Ops::Unshard | Ops::Bitcast | Ops::After) {
            if op != Ops::After {
                views.push(replace_uop);
            }
            replace_uop = ctx.srcs(replace_uop)[0];
        } else {
            break;
        }
    }
    for v in views.iter().rev() {
        let (op, dt, vsrcs, arg, vtag) = ctx.snapshot(*v);
        let mut ns = vec![replace_uop];
        ns.extend(vsrcs[1..].iter().copied());
        replace_uop = ctx.uop_tag(op, Some(dt), ns, arg, vtag);
    }
    if let Arg::Seq { tuple: true, items } = &tag {
        for t in items {
            let idx = match t {
                Arg::Int(i) => *i as usize,
                other => panic!("TGERR: bad tag item {other:?}"),
            };
            let original = *actx.uop_list.get(idx).expect("TGERR: tag out of range");
            let o_shape = ctx.shape(original).expect("TGERR: tagged shape").unwrap_or_default();
            let shr = crate::tensor::movement::shrink_to(ctx, replace_uop, &o_shape);
            actx.buffer_map_insert(original, shr);
        }
    }
    Some(ret)
}

/// `replace_input_buffer` (tensor.py:212).
fn replace_input_buffer(actx: &mut AllocCtx, ctx: &mut UOpCtx, b: UopId) -> Option<UopId> {
    actx.replacements.push(b);
    let slot = (actx.replacements.len() - 1) as i64;
    if ctx.is_bound_var(b) || ctx.is_variable(b) {
        // param_like
        let inner = if ctx.op_of(b) == Ops::After { ctx.srcs(b)[0] } else { b };
        let (op, dt, srcs, arg, _tag) = ctx.snapshot(inner);
        let _ = op;
        if let Arg::Param(p) = &arg {
            let mut np = p.clone();
            np.slot = slot;
            np.name = Some(format!("p{slot}"));
            return Some(ctx.uop_tag(Ops::Param, Some(dt), srcs, Arg::Param(np), Arg::None));
        }
        return None;
    }
    let shape = ctx.shape(b).expect("TGERR: param shape").unwrap_or_default();
    let dt = ctx.dt_of(b);
    let addrspace = ctx.addrspace_of(b).unwrap_or(AddrSpace::Global);
    let dev: Option<crate::uop::arg::DeviceSpec> = crate::schedule::indexing::device_of(ctx, b);
    let pa = ParamArg { slot, dtype: dt, vmin_vmax: None, multiple_of: None, name: None, addrspace: Some(addrspace), axis: None, device: dev, volatile: false };
    let shape_src = ctx.shape_arg(&shape);
    Some(ctx.uop(Ops::Param, Some(dt), vec![shape_src], Arg::Param(pa)))
}

/// `pm_early_transform_tensor_graph` (tensor.py:145).
pub fn pm_early_transform() -> PatternMatcher {
    let rules: Vec<(UPat, RuleFn)> = vec![
        // fold MOPS+BITCAST over BUFFER into SHRINK view
        (
            UPat::op_var(vec![Ops::Copy, Ops::Contiguous])
                .with_src_tup(vec![UPat::op_var(group::MOVEMENT.iter().copied().chain([Ops::Bitcast]).collect::<Vec<Ops>>()).named("src")])
                .allow_all_srcs()
                .named("c"),
            r_al(|a, ctx, st| contiguous_mops_to_view(a, ctx, sym_sget(st, "c"), sym_sget(st, "src"))),
        ),
        (
            UPat::ops1(Ops::Store)
                .with_src_tup(vec![UPat::ops1(Ops::Bitcast).named("src"), UPat::var(None, None)])
                .allow_all_srcs()
                .named("c"),
            r_al(|a, ctx, st| contiguous_mops_to_view(a, ctx, sym_sget(st, "c"), sym_sget(st, "src"))),
        ),
        // add CONTIGUOUS to tagged UOps
        (
            UPat::op_var(crate::uop::ops::OPS_ALL.iter().copied().filter(|o| !matches!(o, Ops::Contiguous | Ops::After | Ops::Store)).collect::<Vec<Ops>>()).named("x"),
            r_al(|_a, ctx, st| {
                let x = sym_sget(st, "x");
                let tag = ctx.tag_of(x).clone();
                if matches!(tag, Arg::None) {
                    return None;
                }
                if tag_nonempty(&tag) {
                    // x.rtag(None).contiguous(tag=x.tag)
                    let un = ctx.uop_tag(ctx.op_of(x), Some(ctx.dt_of(x)), ctx.srcs(x).to_vec(), ctx.arg_of(x).clone(), Arg::None);
                    return Some(ctx.uop_tag(Ops::Contiguous, Some(ctx.dt_of(x)), vec![un], Arg::None, tag));
                }
                Some(ctx.uop_tag(ctx.op_of(x), Some(ctx.dt_of(x)), ctx.srcs(x).to_vec(), ctx.arg_of(x).clone(), Arg::None))
            }),
        ),
        // remove extra CONTIGUOUS on AFTER
        (
            UPat::ops1(Ops::Contiguous).with_src_tup(vec![UPat::ops1(Ops::After).allow_all_srcs().named("a")]).named("c"),
            r_al(|_a, ctx, st| {
                let (a, c) = (sym_sget(st, "a"), sym_sget(st, "c"));
                if ctx.has_buffer_identity(ctx.srcs(a)[0], false) {
                    let a_tag = ctx.tag_of(a).clone();
                    let c_tag = ctx.tag_of(c).clone();
                    let merged = merge_tags(&a_tag, &c_tag);
                    let _ = merged;
                    // python: a.replace(tag=...) — the CONTIGUOUS becomes the tagged AFTER
                    return Some(ctx.uop_tag(Ops::After, Some(ctx.dt_of(a)), ctx.srcs(a).to_vec(), ctx.arg_of(a).clone(), merged));
                }
                None
            }),
        ),
        // replace AFTER+STORE with CONTIGUOUS when target is not a buffer
        (
            UPat::ops1(Ops::After)
                .with_src_tup(vec![UPat::var(None, None), UPat::ops1(Ops::Store).with_src_tup(vec![UPat::var(None, None), UPat::var(Some("src"), None)])])
                .allow_all_srcs()
                .named("u"),
            r_al(|a, ctx, st| replace_store_after_with_contig(a, ctx, sym_sget(st, "u"))),
        ),
        // replace CONTIGUOUS with STORE+AFTER
        (UPat::ops1(Ops::Contiguous).named("u"), r_al(|a, ctx, st| replace_contig_with_store_after(a, ctx, sym_sget(st, "u")))),
        // remove DETACH/CONTIGUOUS_BACKWARD
        (
            UPat::op_var(vec![Ops::Detach, Ops::ContiguousBackward]).named("x"),
            r_al(|_a, ctx, st| Some(ctx.srcs(sym_sget(st, "x"))[0])),
        ),
    ];
    PatternMatcher::new(rules)
}

/// `pm_finalize_call` (tensor.py:207).
pub fn pm_finalize_call() -> PatternMatcher {
    PatternMatcher::new(vec![(
        UPat::ops1(Ops::After).allow_all_srcs().named("x"),
        r_al(|a, ctx, st| finalize_after(a, ctx, sym_sget(st, "x"))),
    )])
}

/// `pm_replace_buf` (tensor.py:210).
pub fn pm_replace_buf() -> PatternMatcher {
    let rules: Vec<(UPat, RuleFn)> = vec![
        (
            UPat::ops1(Ops::Buffer).with_src_tup(vec![UPat::var(None, None)]).named("b"),
            r_al(|a, ctx, st| {
                let b = sym_sget(st, "b");
                if let Arg::Param(p) = ctx.arg_of(b) {
                    if p.addrspace == Some(AddrSpace::Global) {
                        return replace_input_buffer(a, ctx, b);
                    }
                }
                None
            }),
        ),
        (
            UPat::ops1(Ops::After).allow_all_srcs().named("b"),
            r_al(|a, ctx, st| {
                let b = sym_sget(st, "b");
                if ctx.is_bound_var(b) {
                    replace_input_buffer(a, ctx, b)
                } else {
                    None
                }
            }),
        ),
    ];
    PatternMatcher::new(rules)
}

/// `transform_to_call` (tensor.py:224) — returns (call_uop, buffer_map).
pub fn transform_to_call(ctx: &mut UOpCtx, big_sink: UopId) -> (UopId, Vec<(UopId, UopId)>) {
    // bases: real (non-virtual, non-buffer-identity) bases of sink srcs
    let mut actx = AllocCtx::default();
    let sink_srcs = ctx.srcs(big_sink).to_vec();
    for x in sink_srcs {
        let base = ctx.base(x);
        let is_virtual = crate::schedule::indexing::device_of(ctx, base).is_none() || dtype::is_weak(ctx.dt_of(base));
        if !is_virtual && !ctx.has_buffer_identity(base, false) && ctx.op_of(base) != Ops::After && ctx.addrspace_of(base) != Some(AddrSpace::Alu) {
            if !actx.bases.contains(&base) {
                actx.bases.push(base);
            }
        }
    }
    let shared = Rc::new(RefCell::new(actx));

    let big_sink = graph_rewrite(ctx, big_sink, &add_tags(), true, None, false, false, CtxVal::Al(shared.clone()))
        .unwrap_or_else(|e| panic!("TGERR: number the uops: {e}"));
    let big_sink = graph_rewrite(ctx, big_sink, &pm_early_transform(), false, None, false, false, CtxVal::Al(shared.clone()))
        .unwrap_or_else(|e| panic!("TGERR: early transform: {e}"));
    let _big = graph_rewrite(ctx, big_sink, &pm_finalize_call(), false, None, false, false, CtxVal::Al(shared.clone()))
        .unwrap_or_else(|e| panic!("TGERR: finalize call: {e}"));

    let assigns = shared.borrow().assigns.clone();
    let assign_sink = crate::tensor::sink(ctx, assigns);
    let ret = graph_rewrite(ctx, assign_sink, &pm_replace_buf(), true, None, false, false, CtxVal::Al(shared.clone()))
        .unwrap_or_else(|e| panic!("TGERR: replace bufs: {e}"));
    // NOTE: replacements are populated BY pm_replace_buf — snapshot after it runs
    let replacements = shared.borrow().replacements.clone();
    let call = ctx.call(ret, replacements, None);
    let map = shared.borrow().buffer_map.clone();
    (call, map)
}
