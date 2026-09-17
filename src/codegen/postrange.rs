//! `codegen/opt/postrange.py` port (M5): Scheduler + apply_opts.
//!
//! Corpus subset: UPCAST/UNROLL/LOCAL/GROUPTOP opts, convert_loop_to_global,
//! the hand-coded heuristic paths that fire on small kernels, and
//! `get_optimized_ast` (flatten range + KernelInfo with the colored name).

use crate::dtype::ConstVal;
use crate::uop::arg::{Arg, DataArg};
use crate::uop::ctx::UOpCtx;
use crate::uop::ops::Ops;
use crate::uop::range::AxisType;
use crate::uop::sint::SVal;
use crate::uop::upat::{graph_rewrite, CtxVal};
use crate::uop::UopId;
use std::collections::HashMap;

/// `axis_to_pos` (ops.py:45).
fn axis_to_pos(t: AxisType) -> i32 {
    match t {
        AxisType::Device => -2,
        AxisType::Weak | AxisType::Loop => -1,
        AxisType::Thread | AxisType::Global => 0,
        AxisType::Warp | AxisType::Local | AxisType::GroupReduce => 1,
        AxisType::Reduce => 4,
        AxisType::Upcast => 3,
        AxisType::Unroll => 5,
        AxisType::Placeholder => 6,
    }
}

/// `axis_letters` (ops.py:38).
fn axis_letter(t: AxisType) -> &'static str {
    match t {
        AxisType::Device => "d",
        AxisType::Global => "g",
        AxisType::Thread => "t",
        AxisType::Local => "l",
        AxisType::Warp => "w",
        AxisType::Weak | AxisType::Loop => "L",
        AxisType::Upcast => "u",
        AxisType::GroupReduce => "G",
        AxisType::Reduce => "R",
        AxisType::Unroll => "r",
        AxisType::Placeholder => "?",
    }
}

/// `axis_colors` (ops.py:40).
fn axis_color(t: AxisType) -> &'static str {
    match t {
        AxisType::Device => "green",
        AxisType::Global => "blue",
        AxisType::Thread => "BLUE",
        AxisType::Local => "cyan",
        AxisType::Warp => "CYAN",
        AxisType::Weak | AxisType::Loop => "WHITE",
        AxisType::Upcast => "yellow",
        AxisType::GroupReduce => "RED",
        AxisType::Reduce => "red",
        AxisType::Unroll => "magenta",
        AxisType::Placeholder => "white",
    }
}

/// `colored` (helpers.py:38) — ANSI, colors always on (NO_COLOR unset).
pub fn colored(st: &str, color: &str) -> String {
    const COLORS: [&str; 8] = ["black", "red", "green", "yellow", "blue", "magenta", "cyan", "white"];
    let idx = COLORS.iter().position(|&c| c == color.to_lowercase()).unwrap_or(0);
    let code = 60 * (color.chars().next().is_some_and(|c| c.is_uppercase()) as i32) + 30 + idx as i32;
    format!("\u{1b}[{}m{}\u{1b}[0m", code, st)
}

use crate::renderer::Renderer;


/// `Scheduler` (postrange.py:26).
pub struct Scheduler<'a> {
    pub ast: UopId,
    pub ren: &'a Renderer,
    pub dont_use_locals: bool,
    pub     applied_opts: Vec<Arg>,
    opt_range: i64,
}

impl<'a> Scheduler<'a> {
    pub fn new(ctx: &mut UOpCtx, ast: UopId, ren: &'a Renderer) -> Scheduler<'a> {
        let dont_use_locals = match ctx.arg_of(&ast) {
            Arg::Data(d) => match d.get("dont_use_locals") {
                Arg::Bool(b) => *b,
                _ => false,
            },
            _ => false,
        };
        let applied_opts = match ctx.arg_of(&ast) {
            Arg::Data(d) => match d.get("applied_opts") {
                Arg::Seq { items, .. } => items.clone(),
                _ => vec![],
            },
            _ => vec![],
        };
        let mut base_axis = -1i64;
        for u in ctx.backward_slice_with_self(&ast).iter() {
            if ctx.op_of(u) == Ops::Range {
                base_axis = base_axis.max(AxisType::range_axis_id(ctx, u.clone()));
            }
        }
        Scheduler { ast, ren, dont_use_locals, applied_opts, opt_range: base_axis + 1 }
    }

    /// `rngs` — opt axes in order.
    pub fn rngs(&self, ctx: &mut UOpCtx) -> Vec<UopId> {
        let mut rngs: Vec<UopId> = ctx
            .backward_slice_with_self(&self.ast)
            .into_iter()
            .filter(|u| {
                ctx.op_of(u) == Ops::Range
                    && ctx.dt_of(u) != crate::dtype::VOID
                    && self.rng_vmax(ctx, u.clone()) > 0
                    && AxisType::range_axis_type(ctx, u.clone()) != AxisType::Device
            })
            .collect();
        rngs.sort_by_key(|r| (axis_to_pos(AxisType::range_axis_type(ctx, r.clone())), AxisType::range_axis_id(ctx, r.clone())));
        rngs
    }

    /// `int(rng.vmax)` — corpus extents are concrete.
    fn rng_vmax(&self, ctx: &mut UOpCtx, r: UopId) -> i64 {
        let ext = ctx.srcs(&r)[0].clone();
        match ctx.ssimplify(ext) {
            Ok(SVal::I(v)) => v as i64 - 1,
            _ => panic!("TGERR: symbolic range extent unsupported in postrange"),
        }
    }

    pub fn shape_len(&self, ctx: &mut UOpCtx) -> usize {
        self.rngs(ctx).len()
    }

    /// `full_shape` — ssimplify of extents.
    pub fn full_shape(&self, ctx: &mut UOpCtx) -> Vec<i64> {
        self.rngs(ctx).iter().map(|r| self.rng_vmax(ctx, r.clone()) + 1).collect()
    }

    pub fn axis_types(&self, ctx: &mut UOpCtx) -> Vec<AxisType> {
        self.rngs(ctx).iter().map(|r| AxisType::range_axis_type(ctx, r.clone())).collect()
    }

    /// `output_shape`.
    pub fn output_shape(&self, ctx: &mut UOpCtx) -> Vec<i64> {
        self.full_shape(ctx)
            .into_iter()
            .zip(self.axis_types(ctx))
            .map(|(s, at)| if matches!(at, AxisType::Reduce | AxisType::Unroll | AxisType::GroupReduce) { 1 } else { s })
            .collect()
    }

    pub fn ranges_of(&self, ctx: &mut UOpCtx, types: &[AxisType]) -> Vec<UopId> {
        self.rngs(ctx).into_iter().filter(|r| types.contains(&AxisType::range_axis_type(ctx, r.clone()))).collect()
    }

    pub fn axes_of(&self, ctx: &mut UOpCtx, types: &[AxisType]) -> Vec<usize> {
        self.axis_types(ctx).into_iter().enumerate().filter(|(_, t)| types.contains(t)).map(|(i, _)| i).collect()
    }

    /// `upcastable_dims`.
    pub fn upcastable_dims(&self, ctx: &mut UOpCtx) -> Vec<usize> {
        self.axes_of(ctx, &[AxisType::Global, AxisType::Local, AxisType::Weak])
            .into_iter()
            .filter(|&i| self.full_shape(ctx)[i] > 1)
            .collect()
    }

    /// `unrollable_dims`.
    pub fn unrollable_dims(&self, ctx: &mut UOpCtx) -> Vec<usize> {
        self.axes_of(ctx, &[AxisType::GroupReduce, AxisType::Reduce])
            .into_iter()
            .filter(|&i| self.full_shape(ctx)[i] > 1)
            .collect()
    }

    pub fn upcasted(&self, ctx: &mut UOpCtx) -> usize {
        self.axes_of(ctx, &[AxisType::Upcast, AxisType::Unroll]).len()
    }

    pub fn upcast_size(&self, ctx: &mut UOpCtx) -> i64 {
        self.axes_of(ctx, &[AxisType::Upcast, AxisType::Unroll]).iter().map(|&a| self.full_shape(ctx)[a]).product()
    }

    pub fn group_for_reduces(&self, ctx: &mut UOpCtx) -> usize {
        self.axes_of(ctx, &[AxisType::GroupReduce]).len()
    }

    /// `reduceop` — the first REDUCE in the backward slice.
    pub fn reduceop(&self, ctx: &mut UOpCtx) -> Option<UopId> {
        ctx.backward_slice_with_self(&self.ast).into_iter().find(|u| ctx.op_of(u) == Ops::Reduce)
    }

    /// `bufs` — INDEX nodes, reversed toposort.
    pub fn bufs(&self, ctx: &mut UOpCtx) -> Vec<UopId> {
        let mut v: Vec<UopId> = ctx.toposort(&self.ast).into_iter().filter(|u| ctx.op_of(u) == Ops::Index).collect();
        v.reverse();
        v
    }

    /// `_output_rngs`.
    fn output_rngs(&self, ctx: &mut UOpCtx) -> Vec<UopId> {
        let mut ret = Vec::new();
        let ast_srcs = ctx.srcs(&self.ast).to_vec();
        for s in &ast_srcs {
            if ctx.op_of(s) != Ops::End {
                continue;
            }
            let sink = crate::tensor::sink(ctx, ctx.srcs(s)[1..].to_vec());
            for r in ctx.ranges(&sink).iter() {
                if AxisType::range_axis_type(ctx, r.clone()) != AxisType::Reduce && !ret.contains(r) {
                    ret.push(r.clone());
                }
            }
        }
        ret
    }

    /// `_globalizable_rngs`.
    fn globalizable_rngs(&self, ctx: &mut UOpCtx) -> Vec<UopId> {
        let mut ret: Vec<UopId> = self.output_rngs(ctx).into_iter().filter(|r| AxisType::range_axis_type(ctx, r.clone()) == AxisType::Weak).collect();
        for x in ctx.toposort(&self.ast).iter() {
            if ctx.op_of(x) == Ops::Stage {
                let xr = ctx.ranges(&x.clone());
                ret.retain(|r| xr.contains(r));
            }
        }
        ret
    }

    /// `convert_loop_to_global`.
    pub fn convert_loop_to_global(&mut self, ctx: &mut UOpCtx) {
        if !self.ren.has_local {
            return;
        }
        let globalizable = self.globalizable_rngs(ctx);
        if globalizable.is_empty() {
            return;
        }
        let rngs = self.rngs(ctx);
        let mut map = HashMap::new();
        for r in &rngs {
            if globalizable.contains(r) {
                let (op, dt, srcs, arg, tag) = ctx.snapshot(r);
                let Arg::Seq { tuple: true, items } = &arg else { panic!("TGERR: bad range arg") };
                let mut new_items = items.clone();
                // `x.arg[0:-1]+(AxisType.GLOBAL,)` — replace the last (axis type) item
                *new_items.last_mut().unwrap() = Arg::Enum(AxisType::Global.to_enum_val());
                let nr = ctx.uop_tag(op, Some(dt), srcs, Arg::Seq { tuple: true, items: new_items }, tag);
                map.insert(r.clone(), nr);
            }
        }
        if !map.is_empty() {
            self.ast = crate::uop::upat::substitute(ctx, self.ast.clone(), map, false, false);
        }
    }

    /// `shift_to`.
    fn shift_to(&mut self, ctx: &mut UOpCtx, rng: UopId, amount: i64, new_type: AxisType, top: bool) -> (UopId, UopId) {
        let ext = ctx.srcs(&rng)[0].clone();
        let old_sz = ctx.divides(ext, amount as i128).unwrap_or_else(|| panic!("TGERR: {} can't divide the range extent", amount));
        let axis = self.opt_range;
        self.opt_range += 1;
        let new_rng = {
            let dt = ctx.dt_of(&rng);
            let eu = ctx.sint_to_uop_dt(SVal::I(amount as i128), dt);
            ctx.uop(Ops::Range, Some(dt), vec![eu], Arg::Seq { tuple: true, items: vec![Arg::Int(axis), Arg::Enum(new_type.to_enum_val())] })
        };
        let (rop, rdt, _rsrcs, rarg, rtag) = ctx.snapshot(&rng);
        let Arg::Seq { tuple: true, items: _ } = &rarg else { panic!("TGERR: bad range arg") };
        let replaced = ctx.uop_tag(rop, Some(rdt), vec![old_sz.clone()], rarg.clone(), rtag);
        // sub_axis = old*amt + new (bottom) or new*old_sz + old (top)
        let amt_u = ctx.sint_to_uop_dt(SVal::I(amount as i128), rdt);
        let sub_axis = if top {
            let prod = ctx.mul(new_rng.clone(), old_sz);
            ctx.add(prod, replaced.clone())
        } else {
            let prod = ctx.mul(replaced.clone(), amt_u);
            ctx.add(prod, new_rng.clone())
        };
        let mut map = HashMap::new();
        map.insert(rng, sub_axis);
        self.ast = crate::uop::upat::substitute(ctx, self.ast.clone(), map, false, false);
        (replaced, new_rng)
    }

    /// `apply_opt` — corpus subset (UPCAST/UNROLL/LOCAL/GROUPTOP).
    pub fn apply_opt(&mut self, ctx: &mut UOpCtx, op: OptOp, axis: Option<usize>, arg: Option<i64>) {
        let at = match op {
            OptOp::Local => AxisType::Local,
            OptOp::Upcast => AxisType::Upcast,
            OptOp::Unroll => AxisType::Unroll,
            OptOp::Group | OptOp::GroupTop => AxisType::GroupReduce,
        };
        if matches!(op, OptOp::Local | OptOp::Group | OptOp::GroupTop) {
            assert!(self.ren.has_local, "TGERR: locals needed for opt");
        }
        // real_axis
        let real_axis = match op {
            OptOp::Unroll => {
                let ud = self.unrollable_dims(ctx);
                *ud.get(axis.unwrap_or(0)).unwrap_or_else(|| panic!("TGERR: bad unroll axis"))
            }
            OptOp::Group | OptOp::GroupTop => {
                let rd = self.axes_of(ctx, &[AxisType::Reduce]);
                *rd.get(axis.unwrap_or(0)).unwrap_or_else(|| panic!("TGERR: bad group axis"))
            }
            _ => {
                let a = axis.unwrap_or(0);
                assert!(a < self.shape_len(ctx), "TGERR: invalid axis");
                a
            }
        };
        let rng = self.rngs(ctx)[real_axis].clone();
        // python: `amt = int(rng.vmax+1) if opt.arg == 0 else opt.arg` — a
        // literal 0 arg means "the full extent", never an actual zero divisor.
        let amt = match arg {
            Some(0) | None => self.rng_vmax(ctx, rng.clone()) + 1,
            Some(a) => a,
        };
        match op {
            OptOp::Unroll => {
                assert!(amt <= 32, "TGERR: don't unroll more than 32");
                assert!(matches!(AxisType::range_axis_type(ctx, rng.clone()), AxisType::GroupReduce | AxisType::Reduce), "TGERR: unroll is for GROUP_REDUCE/REDUCE");
            }
            OptOp::Upcast => {
                assert!(amt <= 16, "TGERR: don't upcast more than 16");
                assert!(
                    matches!(AxisType::range_axis_type(ctx, rng.clone()), AxisType::Global | AxisType::Local | AxisType::Weak),
                    "TGERR: upcast is for GLOBAL/LOCAL/LOOP"
                );
            }
            OptOp::Local => {
                assert!(!self.dont_use_locals, "TGERR: can't use locals");
                assert!(matches!(AxisType::range_axis_type(ctx, rng.clone()), AxisType::Global | AxisType::Weak), "TGERR: local is for globals");
            }
            OptOp::Group | OptOp::GroupTop => {
                assert!(!self.dont_use_locals, "TGERR: can't use locals");
                assert!(AxisType::range_axis_type(ctx, rng.clone()) == AxisType::Reduce, "TGERR: group is for reduce");
            }
        }
        self.shift_to(ctx, rng, amt, at, matches!(op, OptOp::GroupTop));
        self.applied_opts.push(Arg::Data(DataArg {
            tag: "Opt".into(),
            fields: vec![
                ("op".into(), Arg::Str(format!("{op:?}").to_uppercase())),
                ("axis".into(), Arg::Int(real_axis as i64)),
                ("arg".into(), Arg::Int(amt)),
            ],
        }));
    }

    /// `shape_str` — strings like ['g0', 'u0'].
    pub fn shape_str(&self, ctx: &mut UOpCtx) -> Vec<String> {
        let mut ret = Vec::new();
        let mut cnt: HashMap<AxisType, i32> = HashMap::new();
        for x in self.axis_types(ctx) {
            let c = cnt.entry(x).or_insert(-1);
            *c += 1;
            ret.push(format!("{}{}", axis_letter(x), c));
        }
        ret
    }

    /// `get_optimized_ast`.
    pub fn get_optimized_ast(self, ctx: &mut UOpCtx, name_override: Option<&str>) -> UopId {
        let name = match name_override {
            Some(n) => n.to_string(),
            None => {
                let k_type = if self.reduceop(ctx).is_some() { "r" } else { "E" };
                let mut parts: Vec<String> = Vec::new();
                for r in self.rngs(ctx) {
                    let color = axis_color(AxisType::range_axis_type(ctx, r.clone()));
                    let ext = ctx.srcs(&r)[0].clone();
                    let rendered = sval_render(ctx, ext);
                    parts.push(colored(&rendered, color));
                }
                let sep = colored("_", "BLACK");
                let mut name = k_type.to_string();
                for (i, p) in parts.iter().enumerate() {
                    if i > 0 || true {
                        name.push_str(&sep);
                    }
                    name.push_str(p);
                }
                name
            }
        };
        let pm = crate::codegen::simplify::pm_flatten_range();
        let ast = graph_rewrite(ctx, self.ast, &pm, false, None, false, false, CtxVal::None)
            .unwrap_or_else(|e| panic!("TGERR: flatten range: {e}"));
        let ki = kernel_info_with_opts(&name, &self.applied_opts, self.dont_use_locals);
        let (op, dt, srcs, _old_arg, _tag) = ctx.snapshot(&ast);
        ctx.uop_tag(op, Some(dt), srcs, Arg::Data(ki), Arg::Int(1))
    }
}

/// render a (const) extent like python's `UOp.render` for the kernel name.
fn sval_render(ctx: &mut UOpCtx, u: UopId) -> String {
    match ctx.op_of(&u) {
        Ops::Const => match ctx.val(&u) {
            crate::dtype::ConstVal::Int(v) => format!("{}", v),
            crate::dtype::ConstVal::Float(f) => {
                if f == f.trunc() && f.abs() < 1e15 {
                    format!("{}", f as i64)
                } else {
                    format!("{}", f)
                }
            }
            other => panic!("TGERR: bad const render {other:?}"),
        },
        _ => {
            match ctx.ssimplify(u) {
                Ok(SVal::I(v)) => format!("{}", v),
                _ => panic!("TGERR: symbolic extent render unsupported"),
            }
        }
    }
}

/// Debug: dump every Index offset expr in a linear op list (post-linearize).
pub fn dump_lst_index_offsets(ctx: &mut UOpCtx, lst: &[UopId], label: &str) {
    fn pexpr(ctx: &mut UOpCtx, u: UopId, out: &mut String) {
        let (op, _dt, srcs, _arg, _tag) = ctx.snapshot(&u);
        match op {
            Ops::Const => { let _ = std::fmt::Write::write_fmt(out, format_args!("{}", sval_render(ctx, u))); }
            Ops::Range | Ops::Special => {
                let tag = match ctx.arg_of(&u) { Arg::Str(s) => s.clone(), _ => format!("r{}", AxisType::range_axis_id(ctx, u)) };
                let _ = std::fmt::Write::write_fmt(out, format_args!("{}", tag));
            }
            other => {
                if srcs.len() == 2 {
                    let _ = std::fmt::Write::write_fmt(out, format_args!("("));
                    for (i, s) in srcs.iter().enumerate() {
                        if i > 0 { let _ = std::fmt::Write::write_fmt(out, format_args!(" {:?} ", other)); }
                        pexpr(ctx, s.clone(), out);
                    }
                    let _ = std::fmt::Write::write_fmt(out, format_args!(")"));
                } else {
                    let _ = std::fmt::Write::write_fmt(out, format_args!("{:?}[{}]", other, srcs.len()));
                }
            }
        }
    }
    for o in lst {
        if ctx.op_of(o) == Ops::Index {
            let isrcs = ctx.srcs(o).to_vec();
            if isrcs.len() < 2 { continue; }
            let mut s = String::new();
            pexpr(ctx, (isrcs[1]).clone(), &mut s);
            eprintln!("[{}] idx{} off = {}", label, o.id(), s);
        }
    }
}

/// Debug: dump RANGE/SPECIAL ops and every LOAD/STORE INDEX offset expr.
pub fn dump_index_exprs(ctx: &mut UOpCtx, ast: UopId, label: &str) {
    fn pexpr(ctx: &mut UOpCtx, u: UopId, out: &mut String) {
        let (op, _dt, srcs, _arg, _tag) = ctx.snapshot(&u);
        match op {
            Ops::Const => { let _ = std::fmt::Write::write_fmt(out, format_args!("{}", sval_render(ctx, u))); }
            Ops::Range | Ops::Special => {
                let tag = match ctx.arg_of(&u) { Arg::Str(s) => s.clone(), _ => format!("r{}", AxisType::range_axis_id(ctx, u)) };
                let _ = std::fmt::Write::write_fmt(out, format_args!("{}", tag));
            }
            other => {
                if op == Ops::Where && srcs.len() == 3 {
                    let _ = std::fmt::Write::write_fmt(out, format_args!("("));
                    pexpr(ctx, (srcs[0]).clone(), out);
                    let _ = std::fmt::Write::write_fmt(out, format_args!("?"));
                    pexpr(ctx, (srcs[1]).clone(), out);
                    let _ = std::fmt::Write::write_fmt(out, format_args!(":"));
                    pexpr(ctx, (srcs[2]).clone(), out);
                    let _ = std::fmt::Write::write_fmt(out, format_args!(")"));
                } else if srcs.len() == 2 {
                    let _ = std::fmt::Write::write_fmt(out, format_args!("("));
                    for (i, s) in srcs.iter().enumerate() {
                        if i > 0 { let _ = std::fmt::Write::write_fmt(out, format_args!(" {:?} ", other)); }
                        pexpr(ctx, s.clone(), out);
                    }
                    let _ = std::fmt::Write::write_fmt(out, format_args!(")"));
                } else {
                    let _ = std::fmt::Write::write_fmt(out, format_args!("{:?}[{}]", other, srcs.len()));
                }
            }
        }
    }
    for o in &ctx.toposort(&ast) {
        match ctx.op_of(o) {
            Ops::Range | Ops::Special => {
                let ext = &ctx.srcs(o)[0].clone();
                let (tag, at) = match ctx.arg_of(o) {
                    Arg::Str(s) => (s.clone(), String::from("special")),
                    _ => (String::new(), format!("{:?}", AxisType::range_axis_type(ctx, o.clone()))),
                };
                eprintln!("[{}] rng {} ext={} at={}", label, tag, sval_render(ctx, ext.clone()), at);
            }
            Ops::Index => {
                let isrcs = ctx.srcs(o).to_vec();
                if isrcs.len() < 2 { continue; }
                let mut s = String::new();
                pexpr(ctx, (isrcs[1]).clone(), &mut s);
                eprintln!("[{}] {:?} idx{} off = {}", label, ctx.op_of(o), o.id(), s);
            }
            _ => {}
        }
    }
}

/// `KernelInfo(name=..., applied_opts=(...Opt...), dont_use_locals=...)`.
fn kernel_info_with_opts(name: &str, opts: &[Arg], dont_use_locals: bool) -> DataArg {
    DataArg {
        tag: "KernelInfo".into(),
        fields: vec![
            ("name".into(), Arg::Str(name.into())),
            ("axis_types".into(), Arg::Seq { tuple: true, items: vec![] }),
            ("dont_use_locals".into(), Arg::Bool(dont_use_locals)),
            ("applied_opts".into(), Arg::Seq { tuple: true, items: opts.to_vec() }),
            ("opts_to_apply".into(), Arg::None),
            ("beam".into(), Arg::Int(0)),
        ],
    }
}

/// `OptOps` subset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptOp {
    Upcast,
    Unroll,
    Local,
    Group,
    GroupTop,
}

/// `hand_coded_optimizations` (heuristic.py) — corpus subset.
pub fn hand_coded_optimizations(k: &mut Scheduler, ctx: &mut UOpCtx) {
    // are we grouping? (requires local shape support)
    {
        let ups = k.upcastable_dims(ctx);
        let prod: i64 = ups.iter().map(|&i| k.output_shape(ctx)[i]).product();
        if prod <= 2048 {
            'outer: for axis in 0..3usize {
                let rd = k.axes_of(ctx, &[AxisType::Reduce]);
                if axis >= rd.len() {
                    continue;
                }
                // GROUPTOP: shift_to fails if the extent doesn't divide
                let rng = k.rngs(ctx)[rd[axis]].clone();
                let ext = ctx.srcs(&rng)[0].clone();
                if ctx.divides(ext, 16).is_some() {
                    k.apply_opt(ctx, OptOp::GroupTop, Some(axis), Some(16));
                    break 'outer;
                }
            }
        }
    }

    // no more opt if we are grouping
    if k.group_for_reduces(ctx) > 0 {
        return;
    }

    // if last reduce dim is small(ish), loop unroll the reduce
    {
        let unroll_ok = k.upcast_size(ctx) <= 4 || k.axes_of(ctx, &[AxisType::Unroll]).is_empty();
        if !k.unrollable_dims(ctx).is_empty() && unroll_ok && k.upcast_size(ctx) < 64 {
            let last = k.unrollable_dims(ctx).len() - 1;
            let s = k.full_shape(ctx)[k.unrollable_dims(ctx)[last]];
            if s <= 32 {
                k.apply_opt(ctx, OptOp::Unroll, Some(last), Some(0));
                // if it's small, upcast a second reduce dimension too
                let ud = k.unrollable_dims(ctx);
                if !ud.is_empty() && s <= 3 && k.full_shape(ctx)[ud[ud.len() - 1]] <= 3 {
                    k.apply_opt(ctx, OptOp::Unroll, Some(ud.len() - 1), Some(0));
                }
            } else if k.full_shape(ctx)[k.unrollable_dims(ctx)[last]] % 4 == 0 {
                k.apply_opt(ctx, OptOp::Unroll, Some(last), Some(4));
            }
        }
    }

    // if nothing at all is upcasted and it's easy to, do an upcast
    {
        let ups = k.upcastable_dims(ctx);
        if k.upcasted(ctx) == 0 && !ups.is_empty() && k.full_shape(ctx)[ups[ups.len() - 1]] % 4 == 0 {
            k.apply_opt(ctx, OptOp::Upcast, Some(ups[ups.len() - 1]), Some(4));
        }
    }

    // **** local groups ****
    if k.ren.has_local {
        if k.dont_use_locals {
            // NOLOCALS not needed for the corpus
        } else {
            // prioritize making expand axes local
            let mut to_local: Vec<(usize, i64)> = Vec::new();
            {
                let bufs = k.bufs(ctx);
                let mut ranking: Vec<(bool, usize)> = Vec::new();
                let all_rngs = k.rngs(ctx);
                for axis in k.axes_of(ctx, &[AxisType::Global, AxisType::Weak]) {
                    let ext = ctx.srcs(&all_rngs[axis])[0].clone();
                    if !matches!(ctx.arg_of(&ext), Arg::Const(ConstVal::Int(_))) {
                        continue;
                    }
                    let rng = k.rngs(ctx)[axis].clone();
                    // python: any(rngs[axis] not in b.src[1].get_idx().backward_slice for b in bufs)
                    let not_in_any = {
                        let mut any_missing = false;
                        for b in &bufs {
                            // a 1-src INDEX is a degenerate pass-through: no idx
                            if ctx.srcs(b).len() < 2 {
                                continue;
                            }
                            let idx = ctx.srcs(b)[1].clone();
                            let bs = ctx.backward_slice_with_self(&idx);
                            if !bs.contains(&rng) {
                                any_missing = true;
                            }
                        }
                        any_missing
                    };
                    ranking.push((not_in_any, axis));
                }
                ranking.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)));
                for (_, axis) in ranking {
                    let local_size: i64 = to_local.iter().map(|(_, sz)| *sz).product();
                    let fs = k.full_shape(ctx)[axis];
                    let mut local_sz: Option<i64> = None;
                    let mut cands: Vec<i64> = Vec::new();
                    if axis == 0 {
                        cands.push(32);
                    }
                    cands.extend([16, 8, 4, 3, 2]);
                    for x in cands {
                        if fs % x == 0 && local_size * x <= 128 {
                            local_sz = Some(x);
                            break;
                        }
                    }
                    if let Some(sz) = local_sz {
                        to_local.push((axis, sz));
                    }
                }
            }
            // python: `for axis, local_sz in sorted(to_local[:3])` — sort ascending so that
            // deleting an earlier axis never shifts a later one below `deleted_shape`.
            let mut sorted_local = to_local.clone();
            sorted_local.sort();
            let mut deleted_shape = 0usize;
            let picks: Vec<(usize, i64)> = sorted_local.iter().take(3).cloned().collect();
            for (axis, local_sz) in picks {
                let axis = axis - deleted_shape;
                let will_delete = local_sz == k.full_shape(ctx)[axis];
                k.apply_opt(ctx, OptOp::Local, Some(axis), Some(local_sz));
                if will_delete {
                    deleted_shape += 1;
                }
            }
        }
    }
}

/// `apply_opts` (postrange.py:334).
pub fn apply_opts(ctx: &mut UOpCtx, ast: UopId, ren: &Renderer) -> UopId {
    // ast.tag is not None → already applied (replay)
    if !matches!(ctx.tag_of(&ast), Arg::None) {
        return ast;
    }
    let mut k = Scheduler::new(ctx, ast.clone(), ren);
    k.convert_loop_to_global(ctx);
    // opts_to_apply from the schedule (M4) — apply if present
    let opts_to_apply = match ctx.arg_of(&ast) {
        Arg::Data(d) => match d.get("opts_to_apply") {
            Arg::Seq { items, .. } if !items.is_empty() => Some(items.clone()),
            _ => None,
        },
        _ => None,
    };
    #[cfg(feature = "dump")]
    if std::env::var("TG_DUMP_IDX").is_ok() {
        dump_index_exprs(ctx, k.ast.clone(), "PRE");
    }
    match opts_to_apply {
        Some(_) => panic!("TGERR: opts_to_apply path not ported for corpus"),
        None => hand_coded_optimizations(&mut k, ctx),
    }

    #[cfg(feature = "dump")]
    if std::env::var("TG_DUMP_IDX").is_ok() {
        dump_index_exprs(ctx, k.ast.clone(), "POST");
    }
    let name_override = match ctx.arg_of(&ast) {
        Arg::Data(d) => match d.get("name") {
            Arg::Str(s) if s != "test" => Some(s.to_string()),
            _ => None,
        },
        _ => None,
    };
    let out = k.get_optimized_ast(ctx, name_override.as_deref());
    #[cfg(feature = "dump")]
    if std::env::var("TG_DUMP_IDX").is_ok() {
        dump_index_exprs(ctx, out.clone(), "RESULT");
    }
    out
}

/// Extract the KernelInfo `name` (ANSI-stripped) from the Sink in a uop list.
pub fn kernel_name(ctx: &mut crate::uop::ctx::UOpCtx, lst: &[crate::uop::UopId]) -> String {
    for u in lst {
        if ctx.op_of(u) == Ops::Sink
            && let Arg::Data(d) = ctx.arg_of(u)
                && let Arg::Str(s) = d.get("name") {
                    return strip_ansi(s);
                }
    }
    "<unnamed>".to_string()
}

fn strip_ansi(s: &str) -> String {
    let mut out = String::new();
    let mut it = s.chars().peekable();
    while let Some(c) = it.next() {
        if c == '\u{1b}' {
            for d in it.by_ref() {
                if d == 'm' {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}
