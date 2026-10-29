//! `codegen/opt/postrange.py` port (M5): Scheduler + apply_opts.
//!
//! Corpus subset: UPCAST/UNROLL/LOCAL/GROUPTOP opts, convert_loop_to_global,
//! the hand-coded heuristic paths that fire on small kernels, and
//! `get_optimized_ast` (flatten range + KernelInfo with the colored name).

use crate::dtype::ConstVal;
use crate::uop::UopId;
use crate::uop::arg::{Arg, DataArg, EnumClass, EnumVal};
use crate::uop::ctx::UOpCtx;
use crate::uop::ops::Ops;
use crate::uop::range::AxisType;
use crate::uop::sint::SVal;
use crate::uop::upat::{CtxVal, graph_rewrite};
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

/// `argsort(x)` (helpers.py:29).
fn argsort(x: &[usize]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..x.len()).collect();
    idx.sort_by_key(|&i| x[i]);
    idx
}

/// A RANGE's `arg[0]` (the axis id).
fn range_axis_id(ctx: &mut UOpCtx, r: &UopId) -> Option<i64> {
    match ctx.arg_of(r) {
        Arg::Seq { tuple: true, items } if !items.is_empty() => items[0].as_int(),
        _ => None,
    }
}

/// Sort a list of RANGEs by `arg[0]` descending.
fn sort_by_axis_desc(ctx: &mut UOpCtx, v: &mut Vec<UopId>) {
    let mut keyed: Vec<(i64, UopId)> = v
        .drain(..)
        .map(|u| (range_axis_id(ctx, &u).unwrap_or(i64::MIN), u))
        .collect();
    keyed.sort_by_key(|(k, _)| std::cmp::Reverse(*k));
    v.extend(keyed.into_iter().map(|(_, u)| u));
}

/// `reduceop.arg[0]` — the reduce ALU op.
fn reduce_alu_op(ctx: &mut UOpCtx, reduceop: &UopId) -> Ops {
    match ctx.arg_of(reduceop) {
        Arg::Seq { tuple: true, items } => match &items[0] {
            Arg::Enum(e) => {
                Ops::from_u16(e.value).unwrap_or_else(|| panic!("TGERR: bad reduce op {e:?}"))
            }
            other => panic!("TGERR: bad reduce arg {other:?}"),
        },
        other => panic!("TGERR: bad reduce arg {other:?}"),
    }
}

/// `ALLOW_TF32` (helpers.py) — default off, so float-input tensor cores skip.
fn allow_tf32() -> bool {
    std::env::var("ALLOW_TF32")
        .ok()
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// `getenv(name, default)` for integer knobs.
fn env_int(name: &str, default: i64) -> i64 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or(default)
}

/// `colored` (helpers.py:38) — ANSI, colors always on (NO_COLOR unset).
pub fn colored(st: &str, color: &str) -> String {
    const COLORS: [&str; 8] = [
        "black", "red", "green", "yellow", "blue", "magenta", "cyan", "white",
    ];
    let idx = COLORS
        .iter()
        .position(|&c| c == color.to_lowercase())
        .unwrap_or(0);
    let code =
        60 * (color.chars().next().is_some_and(|c| c.is_uppercase()) as i32) + 30 + idx as i32;
    format!("\u{1b}[{}m{}\u{1b}[0m", code, st)
}

use crate::renderer::Renderer;
use crate::renderer::tc::TensorCore;

/// `Scheduler` (postrange.py:26).
pub struct Scheduler<'a> {
    /// The kernel AST (SINK) being scheduled.
    pub ast: UopId,
    /// Renderer whose capabilities drive the opts.
    pub ren: &'a Renderer,
    /// Whether the AST forbids LOCAL axes (`dont_use_locals`).
    pub dont_use_locals: bool,
    /// Opts applied so far, recorded into the kernel AST's `applied_opts`.
    pub applied_opts: Vec<Arg>,
    opt_range: i64,
    /// `self.tensor_core` — set by `_apply_tc_opt` (postrange.py:308).
    tensor_core: Option<TensorCore>,
}

impl<'a> Scheduler<'a> {
    /// Build a scheduler for `ast` under renderer `ren`.
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
        Scheduler {
            ast,
            ren,
            dont_use_locals,
            applied_opts,
            opt_range: base_axis + 1,
            tensor_core: None,
        }
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
        // tinygrad key is `(axis_to_pos[arg[-1]],) + arg[0:-1]`. `arg[0:-1]` is
        // `(axis,)` normally but becomes `(axis, split)` after `pm_split_ranges`,
        // and the split discriminator is what keeps the two halves of a split
        // range ordered (and unique) — dropping it makes the sort nondeterministic
        // when the halves have different extents.
        rngs.sort_by_key(|r| {
            let at = AxisType::range_axis_type(ctx, r.clone());
            let mut ints: Vec<i64> = Vec::new();
            if let Arg::Seq { tuple: true, items } = ctx.arg_of(r) {
                for a in &items[..items.len() - 1] {
                    if let Arg::Int(i) = a {
                        ints.push(*i);
                    }
                }
            }
            (axis_to_pos(at), ints)
        });
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

    /// Number of opt axes (`len(full_shape)`).
    pub fn shape_len(&self, ctx: &mut UOpCtx) -> usize {
        self.rngs(ctx).len()
    }

    /// `full_shape` — ssimplify of extents.
    pub fn full_shape(&self, ctx: &mut UOpCtx) -> Vec<i64> {
        self.rngs(ctx)
            .iter()
            .map(|r| self.rng_vmax(ctx, r.clone()) + 1)
            .collect()
    }

    /// Axis type of each opt axis, in order.
    pub fn axis_types(&self, ctx: &mut UOpCtx) -> Vec<AxisType> {
        self.rngs(ctx)
            .iter()
            .map(|r| AxisType::range_axis_type(ctx, r.clone()))
            .collect()
    }

    /// `output_shape`.
    pub fn output_shape(&self, ctx: &mut UOpCtx) -> Vec<i64> {
        self.full_shape(ctx)
            .into_iter()
            .zip(self.axis_types(ctx))
            .map(|(s, at)| {
                if matches!(
                    at,
                    AxisType::Reduce | AxisType::Unroll | AxisType::GroupReduce
                ) {
                    1
                } else {
                    s
                }
            })
            .collect()
    }

    /// Ranges whose axis type is in `types`.
    pub fn ranges_of(&self, ctx: &mut UOpCtx, types: &[AxisType]) -> Vec<UopId> {
        self.rngs(ctx)
            .into_iter()
            .filter(|r| types.contains(&AxisType::range_axis_type(ctx, r.clone())))
            .collect()
    }

    /// Indices of opt axes whose axis type is in `types`.
    pub fn axes_of(&self, ctx: &mut UOpCtx, types: &[AxisType]) -> Vec<usize> {
        self.axis_types(ctx)
            .into_iter()
            .enumerate()
            .filter(|(_, t)| types.contains(t))
            .map(|(i, _)| i)
            .collect()
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

    /// Number of axes currently upcast or unrolled.
    pub fn upcasted(&self, ctx: &mut UOpCtx) -> usize {
        self.axes_of(ctx, &[AxisType::Upcast, AxisType::Unroll])
            .len()
    }

    /// Product of the upcast/unroll axis extents.
    pub fn upcast_size(&self, ctx: &mut UOpCtx) -> i64 {
        self.axes_of(ctx, &[AxisType::Upcast, AxisType::Unroll])
            .iter()
            .map(|&a| self.full_shape(ctx)[a])
            .product()
    }

    /// Number of GROUP_REDUCE axes.
    pub fn group_for_reduces(&self, ctx: &mut UOpCtx) -> usize {
        self.axes_of(ctx, &[AxisType::GroupReduce]).len()
    }

    /// `reduceop` — the first REDUCE in the backward slice.
    pub fn reduceop(&self, ctx: &mut UOpCtx) -> Option<UopId> {
        ctx.backward_slice_with_self(&self.ast)
            .into_iter()
            .find(|u| ctx.op_of(u) == Ops::Reduce)
    }

    /// `reduceops` (postrange.py:315) — all REDUCE ops, in toposort order.
    pub fn reduceops(&self, ctx: &mut UOpCtx) -> Vec<UopId> {
        ctx.toposort(&self.ast)
            .into_iter()
            .filter(|u| ctx.op_of(u) == Ops::Reduce)
            .collect()
    }

    /// `bufs` — INDEX nodes, reversed toposort.
    pub fn bufs(&self, ctx: &mut UOpCtx) -> Vec<UopId> {
        let mut v: Vec<UopId> = ctx
            .toposort(&self.ast)
            .into_iter()
            .filter(|u| ctx.op_of(u) == Ops::Index)
            .collect();
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
                if AxisType::range_axis_type(ctx, r.clone()) != AxisType::Reduce && !ret.contains(r)
                {
                    ret.push(r.clone());
                }
            }
        }
        ret
    }

    /// `_globalizable_rngs`.
    fn globalizable_rngs(&self, ctx: &mut UOpCtx) -> Vec<UopId> {
        let mut ret: Vec<UopId> = self
            .output_rngs(ctx)
            .into_iter()
            .filter(|r| AxisType::range_axis_type(ctx, r.clone()) == AxisType::Weak)
            .collect();
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
                let Arg::Seq { tuple: true, items } = &arg else {
                    panic!("TGERR: bad range arg")
                };
                let mut new_items = items.clone();
                // `x.arg[0:-1]+(AxisType.GLOBAL,)` — replace the last (axis type) item
                *new_items.last_mut().unwrap() = Arg::Enum(AxisType::Global.to_enum_val());
                let nr = ctx.uop_tag(
                    op,
                    Some(dt),
                    srcs,
                    Arg::Seq {
                        tuple: true,
                        items: new_items,
                    },
                    tag,
                );
                map.insert(r.clone(), nr);
            }
        }
        if !map.is_empty() {
            self.ast = crate::uop::upat::substitute(ctx, self.ast.clone(), map, false, false);
        }
    }

    /// `shift_to`.
    fn shift_to(
        &mut self,
        ctx: &mut UOpCtx,
        rng: UopId,
        amount: i64,
        new_type: AxisType,
        top: bool,
        input_new_rng: Option<UopId>,
    ) -> (UopId, UopId) {
        let ext = ctx.srcs(&rng)[0].clone();
        let old_sz = ctx
            .divides(ext, amount as i128)
            .unwrap_or_else(|| panic!("TGERR: {} can't divide the range extent", amount));
        let new_rng = match input_new_rng {
            Some(r) => r,
            None => {
                let axis = self.opt_range;
                self.opt_range += 1;
                let dt = ctx.dt_of(&rng);
                let eu = ctx.sint_to_uop_dt(SVal::I(amount as i128), dt);
                ctx.uop(
                    Ops::Range,
                    Some(dt),
                    vec![eu],
                    Arg::Seq {
                        tuple: true,
                        items: vec![Arg::Int(axis), Arg::Enum(new_type.to_enum_val())],
                    },
                )
            }
        };
        let (rop, rdt, _rsrcs, rarg, rtag) = ctx.snapshot(&rng);
        let Arg::Seq {
            tuple: true,
            items: _,
        } = &rarg
        else {
            panic!("TGERR: bad range arg")
        };
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

    /// `apply_opt` — corpus subset (UPCAST/UNROLL/LOCAL/GROUPTOP/PADTO). Returns
    /// the shifted `(replaced_rng, new_rng)` (python `apply_opt`'s `ret`).
    pub fn apply_opt(
        &mut self,
        ctx: &mut UOpCtx,
        op: OptOp,
        axis: Option<usize>,
        arg: Option<i64>,
    ) -> Vec<UopId> {
        assert!(!matches!(op, OptOp::Tc), "TGERR: use apply_opt_tc for TC");
        if matches!(op, OptOp::PadTo) {
            let _ = self.apply_padto(ctx, axis.unwrap_or(0), arg.unwrap_or(1));
            return vec![];
        }
        let at = match op {
            OptOp::Local => AxisType::Local,
            OptOp::Upcast => AxisType::Upcast,
            OptOp::Unroll => AxisType::Unroll,
            OptOp::Group | OptOp::GroupTop => AxisType::GroupReduce,
            OptOp::PadTo | OptOp::Tc => unreachable!(),
        };
        if matches!(op, OptOp::Local | OptOp::Group | OptOp::GroupTop) {
            assert!(self.ren.has_local, "TGERR: locals needed for opt");
        }
        // real_axis
        let real_axis = match op {
            OptOp::Unroll => {
                let ud = self.unrollable_dims(ctx);
                *ud.get(axis.unwrap_or(0))
                    .unwrap_or_else(|| panic!("TGERR: bad unroll axis"))
            }
            OptOp::Group | OptOp::GroupTop => {
                let rd = self.axes_of(ctx, &[AxisType::Reduce]);
                *rd.get(axis.unwrap_or(0))
                    .unwrap_or_else(|| panic!("TGERR: bad group axis"))
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
                assert!(
                    matches!(
                        AxisType::range_axis_type(ctx, rng.clone()),
                        AxisType::GroupReduce | AxisType::Reduce
                    ),
                    "TGERR: unroll is for GROUP_REDUCE/REDUCE"
                );
            }
            OptOp::Upcast => {
                assert!(amt <= 16, "TGERR: don't upcast more than 16");
                assert!(
                    matches!(
                        AxisType::range_axis_type(ctx, rng.clone()),
                        AxisType::Global | AxisType::Local | AxisType::Weak
                    ),
                    "TGERR: upcast is for GLOBAL/LOCAL/LOOP"
                );
            }
            OptOp::Local => {
                assert!(!self.dont_use_locals, "TGERR: can't use locals");
                assert!(
                    matches!(
                        AxisType::range_axis_type(ctx, rng.clone()),
                        AxisType::Global | AxisType::Weak
                    ),
                    "TGERR: local is for globals"
                );
            }
            OptOp::Group | OptOp::GroupTop => {
                assert!(!self.dont_use_locals, "TGERR: can't use locals");
                assert!(
                    AxisType::range_axis_type(ctx, rng.clone()) == AxisType::Reduce,
                    "TGERR: group is for reduce"
                );
            }
            OptOp::PadTo | OptOp::Tc => unreachable!(),
        }
        let (replaced, new_rng) =
            self.shift_to(ctx, rng, amt, at, matches!(op, OptOp::GroupTop), None);
        self.applied_opts.push(Arg::Data(DataArg {
            tag: "Opt".into(),
            fields: vec![
                ("op".into(), Arg::Str(format!("{op:?}").to_uppercase())),
                ("axis".into(), Arg::Int(axis.unwrap_or(0) as i64)),
                ("arg".into(), Arg::Int(amt)),
            ],
        }));
        vec![replaced, new_rng]
    }

    /// `OptOps.PADTO` (postrange.py:185) — round a const range up to a multiple,
    /// masking out-of-range lanes with `Invalid`.
    fn apply_padto(&mut self, ctx: &mut UOpCtx, axis: usize, arg: i64) -> Result<(), String> {
        let rng = self.rngs(ctx)[axis].clone();
        // python: these are `KernelOptError`s (caught by `_apply_tc_opt`), not fatal.
        if ctx.op_of(&ctx.srcs(&rng)[0].clone()) != Ops::Const {
            return Err("only pad const axes".into());
        }
        let at = AxisType::range_axis_type(ctx, rng.clone());
        if matches!(at, AxisType::Upcast | AxisType::Unroll) {
            return Err("cannot pad upcasted".into());
        }
        if at == AxisType::Thread {
            return Err("cannot pad thread".into());
        }
        let old = self.rng_vmax(ctx, rng.clone()) + 1;
        let new_sz = (old + arg - 1) / arg * arg;
        if old <= new_sz / 4 {
            return Err("pad adds more than quadruple the work".into());
        }
        let (rop, rdt, _srcs, rarg, rtag) = ctx.snapshot(&rng);
        let eu = ctx.sint_to_uop_dt(SVal::I(new_sz as i128), rdt);
        let replaced_rng = ctx.uop_tag(rop, Some(rdt), vec![eu], rarg.clone(), rtag);
        let mut replaces: HashMap<UopId, UopId> = HashMap::new();
        replaces.insert(rng.clone(), replaced_rng.clone());
        // valid = replaced_rng < old
        let old_u = ctx.sint_to_uop_dt(SVal::I(old as i128), rdt);
        let valid = ctx.binop(Ops::Cmplt, replaced_rng.clone(), old_u);
        let store_targets: Vec<UopId> = ctx
            .backward_slice_with_self(&self.ast)
            .into_iter()
            .filter(|u| ctx.op_of(u) == Ops::Store)
            .map(|u| ctx.srcs(&u)[0].clone())
            .collect();
        for b in self.bufs(ctx) {
            let bsrcs = ctx.srcs(&b).to_vec();
            if bsrcs.len() < 2 {
                continue;
            }
            let i = ctx.get_idx(bsrcs[1].clone());
            if !ctx.backward_slice_with_self(&i).contains(&rng) {
                continue;
            }
            let gv = ctx.get_valid(bsrcs[1].clone());
            let combined = ctx.binop(Ops::And, valid.clone(), gv);
            let new_idx = ctx.valid(i.clone(), combined);
            let nb = ctx.uop_tag(
                ctx.op_of(&b),
                Some(ctx.dt_of(&b)),
                vec![bsrcs[0].clone(), new_idx],
                ctx.arg_of(&b).clone(),
                Arg::None,
            );
            if store_targets.contains(&b) {
                replaces.insert(b.clone(), nb);
            } else {
                let inv = ctx.const_py(ConstVal::Invalid, Some(ctx.dt_of(&b)));
                let wrapped = ctx.where_(valid.clone(), nb, inv);
                replaces.insert(b.clone(), wrapped);
            }
        }
        self.ast = crate::uop::upat::substitute(ctx, self.ast.clone(), replaces, false, false);
        Ok(())
    }

    /// `apply_opt(Opt(OptOps.TC, axis, (tc_select, tc_opt, use_tensor_cores)))`
    /// (postrange.py:172/dispatch → 214 `_apply_tc_opt`).
    pub fn apply_opt_tc(
        &mut self,
        ctx: &mut UOpCtx,
        axis: usize,
        tc_select: i64,
        tc_opt: i64,
        use_tc: i64,
    ) -> Result<Vec<UopId>, String> {
        if !self.applied_opts.is_empty() {
            return Err("tensor core opts must be first".into());
        }
        if !(-1..self.ren.tensor_cores.len() as i64).contains(&tc_select) {
            return Err("tensor core opts must have valid tc_select".into());
        }
        if !(0..=2).contains(&tc_opt) {
            return Err("tensor core opts must have valid tc_opt".into());
        }
        if !(1..=2).contains(&use_tc) {
            return Err("use_tensor_cores value is not valid".into());
        }
        self.apply_tc_opt(ctx, use_tc, axis, tc_select, tc_opt)
    }

    /// `_apply_tc_opt` (postrange.py:214).
    fn apply_tc_opt(
        &mut self,
        ctx: &mut UOpCtx,
        use_tc: i64,
        axis: usize,
        tc_select: i64,
        opt_level: i64,
    ) -> Result<Vec<UopId>, String> {
        let reduceops = self.reduceops(ctx);
        let reduceop = reduceops
            .first()
            .cloned()
            .ok_or("no reduce ops for TensorCore")?;
        if reduce_alu_op(ctx, &reduceop) != Ops::Add {
            return Err("no tensor core available".into());
        }
        let rsrc0 = ctx.srcs(&reduceop)[0].clone();
        let mul = if ctx.op_of(&rsrc0) == Ops::Cast {
            ctx.srcs(&rsrc0)[0].clone()
        } else {
            rsrc0
        };
        if ctx.op_of(&mul) != Ops::Mul {
            return Err("no tensor core available".into());
        }
        let mul_srcs = ctx.srcs(&mul).to_vec();
        if mul_srcs.len() != 2 {
            return Err("no tensor core available".into());
        }
        let (in0, in1) = (mul_srcs[0].clone(), mul_srcs[1].clone());
        let redt = ctx.dt_of(&reduceop);
        let tensor_cores: Vec<TensorCore> = if tc_select == -1 {
            self.ren.tensor_cores.clone()
        } else {
            vec![self.ren.tensor_cores[tc_select as usize].clone()]
        };
        for tc in tensor_cores {
            if (self.ren.device == "CUDA" || self.ren.device == "NV")
                && tc.dtype_in == crate::dtype::FLOAT32
                && !allow_tf32()
            {
                continue;
            }
            if tc.dtype_in != ctx.dt_of(&in0)
                || tc.dtype_in != ctx.dt_of(&in1)
                || tc.dtype_out != redt
            {
                continue;
            }
            // tensor cores have three ranges: X (in1), Y (in0), REDUCE
            let in0r: Vec<UopId> = ctx.ranges(&in0).iter().cloned().collect();
            let in1r: Vec<UopId> = ctx.ranges(&in1).iter().cloned().collect();
            let mut ir0: Vec<UopId> = in0r.iter().filter(|u| !in1r.contains(u)).cloned().collect();
            let mut ir1: Vec<UopId> = in1r.iter().filter(|u| !in0r.contains(u)).cloned().collect();
            let mut rr: Vec<UopId> = ctx.srcs(&reduceop)[1..].to_vec();
            sort_by_axis_desc(ctx, &mut ir0);
            sort_by_axis_desc(ctx, &mut ir1);
            sort_by_axis_desc(ctx, &mut rr);
            if ir0.is_empty() || ir1.is_empty() || rr.is_empty() {
                continue;
            }
            // NOTE: why are in1 and in0 switched?
            let mut axis_choices: Vec<[UopId; 3]> = Vec::new();
            for b in &ir1 {
                for a in &ir0 {
                    for r in &rr {
                        axis_choices.push([b.clone(), a.clone(), r.clone()]);
                    }
                }
            }
            if axis >= axis_choices.len() {
                continue;
            }
            let mut axes: [UopId; 3] = axis_choices[axis].clone();
            if axes[..2]
                .iter()
                .any(|a| AxisType::range_axis_type(ctx, a.clone()) == AxisType::Reduce)
            {
                return Err("tensor core X/Y axes can't be REDUCE".into());
            }
            // tag the reduceop
            {
                let (op, dt, srcs, arg, _t) = ctx.snapshot(&reduceop);
                let tagged = ctx.uop_tag(op, Some(dt), srcs, arg, Arg::Str("TC".into()));
                let mut m = HashMap::new();
                m.insert(reduceop.clone(), tagged);
                self.ast = crate::uop::upat::substitute(ctx, self.ast.clone(), m, false, false);
            }
            // PADTO when an axis doesn't divide the tensor-core dim
            let dims = [tc.dims.0, tc.dims.1, tc.dims.2];
            for i in 0..3 {
                let idx = match self.rngs(ctx).iter().position(|r| r == &axes[i]) {
                    Some(x) => x,
                    None => return Err("tc axis vanished".into()),
                };
                let extent = self.rng_vmax(ctx, axes[i].clone()) + 1;
                if extent % dims[i] as i64 != 0 {
                    if opt_level < 2 {
                        return Err("tc padding requires opt_level >= 2".into());
                    }
                    self.apply_padto(ctx, idx, dims[i] as i64)?;
                    axes[i] = self.rngs(ctx)[idx].clone();
                }
            }
            // create the warp as a whole, in case ranges move/removed later
            let warp0 = ctx.range(SVal::I(tc.threads as i128), -1, AxisType::Warp);
            let mut warp = warp0.clone();
            let two = ctx.const_py(ConstVal::Int(2), Some(crate::dtype::WEAKINT));
            let mut ne: Vec<UopId> = Vec::new();
            for opt in tc.opts.clone() {
                let di: usize = opt[1..].parse().unwrap();
                if opt.starts_with('l') {
                    let input = ctx.binop(Ops::Floormod, warp.clone(), two.clone());
                    let rng = axes[di].clone();
                    let (replaced, new_rng) =
                        self.shift_to(ctx, rng, 2, AxisType::Local, false, Some(input));
                    axes[di] = replaced;
                    warp = ctx.binop(Ops::Floordiv, warp.clone(), two.clone());
                    ne.push(new_rng);
                } else {
                    let rng = axes[di].clone();
                    let (replaced, new_rng) =
                        self.shift_to(ctx, rng, 2, AxisType::Upcast, false, None);
                    axes[di] = replaced;
                    ne.push(new_rng);
                }
            }
            for (_, amt) in tc.get_reduce_axes() {
                let rng = axes[2].clone();
                let (replaced, new_rng) =
                    self.shift_to(ctx, rng, amt as i64, AxisType::Unroll, false, None);
                axes[2] = replaced;
                ne.push(new_rng);
            }
            if use_tc != 2 {
                let tagged = ctx
                    .toposort(&self.ast)
                    .into_iter()
                    .find(|u| {
                        ctx.op_of(u) == Ops::Reduce
                            && matches!(ctx.tag_of(u), Arg::Str(s) if s == "TC")
                    })
                    .ok_or("no tagged reduce")?;
                let tne: Vec<UopId> = ne
                    .iter()
                    .map(|x| {
                        let (op, dt, srcs, arg, _t) = ctx.snapshot(x);
                        ctx.uop_tag(op, Some(dt), srcs, arg, Arg::Int(1))
                    })
                    .collect();
                let mut m: HashMap<UopId, UopId> = HashMap::new();
                for (a, b) in ne.iter().zip(tne.iter()) {
                    m.insert(a.clone(), b.clone());
                }
                let ret = crate::uop::upat::substitute(ctx, tagged.clone(), m, false, false);
                let ret_src0 = ctx.srcs(&ret)[0].clone();
                let mul_node = if ctx.op_of(&ret_src0) == Ops::Cast {
                    ctx.srcs(&ret_src0)[0].clone()
                } else {
                    ret_src0
                };
                let srcs_list = ctx.srcs(&mul_node).to_vec();
                let (p0, p1) = tc.permutes_for_shape_str(&tc.base_shape_str());
                let perms = [p0, p1];
                let mut new_srcs: Vec<UopId> = Vec::new();
                for (x, p) in srcs_list.iter().zip(perms.iter()) {
                    let asrt = argsort(p);
                    let mut m2: HashMap<UopId, UopId> = HashMap::new();
                    for (j, t) in tne.iter().enumerate() {
                        m2.insert(t.clone(), ne[asrt[j]].clone());
                    }
                    new_srcs.push(crate::uop::upat::substitute(
                        ctx,
                        x.clone(),
                        m2,
                        false,
                        false,
                    ));
                }
                if new_srcs.len() != 2 {
                    return Err("wmma srcs".into());
                }
                // get reduce/upcast axes for the tensor cores
                let redu_nms: Vec<String> = (0..tc.get_reduce_axes().len())
                    .map(|i| format!("r{i}"))
                    .collect();
                let tc_reduce_axis_idx = self.shape_str_to_axis(ctx, &redu_nms);
                let base_upcast_axis_idx = self.shape_str_to_axis(ctx, &tc.base_upcast_axes());
                let base_upcast: Vec<(usize, usize)> =
                    base_upcast_axis_idx.iter().map(|&s| (s, 2usize)).collect();
                let ept = [
                    tc.elements_per_thread.0,
                    tc.elements_per_thread.1,
                    tc.elements_per_thread.2,
                ];
                let rngs_snapshot = self.rngs(ctx);
                let mut tc_upcast_axes: Vec<Vec<(i64, i64)>> = (0..3)
                    .map(|i| {
                        let n = (ept[i] as f64).log2() as usize;
                        base_upcast[..n]
                            .iter()
                            .map(|&(a, sz)| {
                                (range_axis_id(ctx, &rngs_snapshot[a]).unwrap(), sz as i64)
                            })
                            .collect()
                    })
                    .collect();
                let tc_reduce_axes: Vec<i64> = tc_reduce_axis_idx
                    .iter()
                    .map(|&a| range_axis_id(ctx, &rngs_snapshot[a]).unwrap())
                    .collect();
                // with_missing_tc_axes (uses the pre-update value)
                let all_up: Vec<(i64, i64)> = tc_upcast_axes[0]
                    .iter()
                    .chain(tc_upcast_axes[1].iter())
                    .cloned()
                    .collect();
                for v in tc_upcast_axes.iter_mut() {
                    for &(rn, _) in &all_up {
                        if !v.iter().any(|x| x.0 == rn) {
                            v.push((rn, 1));
                        }
                    }
                }
                let device = self.ren.device.clone();
                let acc_consts: Vec<UopId> = (0..ept[2])
                    .map(|_| ctx.const_py(ConstVal::Float(0.0), Some(tc.dtype_out)))
                    .collect();
                let acc = crate::tensor::stack(ctx, acc_consts);
                let tc_uop = ctx.wmma(
                    new_srcs[0].clone(),
                    new_srcs[1].clone(),
                    acc,
                    (tc.dims.0 as i64, tc.dims.1 as i64, tc.dims.2 as i64),
                    &device,
                    tc.threads as i64,
                    Some(tc_upcast_axes),
                );
                // preserve extra reduces
                let red_srcs: Vec<UopId> = ctx.srcs(&tagged)[1..].to_vec();
                let sink = crate::tensor::sink(ctx, red_srcs);
                let reduce_ranges: Vec<UopId> = ctx
                    .toposort(&sink)
                    .into_iter()
                    .filter(|x| {
                        ctx.op_of(x) == Ops::Range
                            && !tc_reduce_axes.contains(&range_axis_id(ctx, x).unwrap())
                    })
                    .collect();
                let mut tc_final = tc_uop.clone();
                if !reduce_ranges.is_empty() {
                    let add_enum = Arg::Enum(EnumVal {
                        class: EnumClass::Ops,
                        value: Ops::Add as u16,
                    });
                    let arg = Arg::Seq {
                        tuple: true,
                        items: vec![add_enum, Arg::Int(0)],
                    };
                    tc_final = ctx.uop(
                        Ops::Reduce,
                        Some(redt),
                        [vec![tc_uop], reduce_ranges].concat(),
                        arg,
                    );
                }
                let mut m: HashMap<UopId, UopId> = HashMap::new();
                m.insert(tagged.clone(), tc_final);
                self.ast = crate::uop::upat::substitute(ctx, self.ast.clone(), m, false, false);
            }
            self.tensor_core = Some(tc);
            return Ok(axes.to_vec());
        }
        Err("no tensor core available".into())
    }

    /// `Scheduler.copy` (search.py) — clone for BEAM candidates.
    pub fn copy(&self) -> Scheduler<'a> {
        Scheduler {
            ast: self.ast.clone(),
            ren: self.ren,
            dont_use_locals: self.dont_use_locals,
            applied_opts: self.applied_opts.clone(),
            opt_range: self.opt_range,
            tensor_core: self.tensor_core.clone(),
        }
    }

    /// The `Opt` list applied to this scheduler (BEAM result / KernelInfo).
    pub fn applied_opts(&self) -> &[Arg] {
        &self.applied_opts
    }

    /// `Scheduler.real_axis` (postrange.py:88) — map an opt-relative axis to the
    /// real axis index in `rngs`.
    pub fn real_axis(&self, ctx: &mut UOpCtx, op: OptOp, axis: usize) -> Option<usize> {
        match op {
            OptOp::Unroll => self.unrollable_dims(ctx).get(axis).copied(),
            OptOp::Group | OptOp::GroupTop => {
                self.axes_of(ctx, &[AxisType::Reduce]).get(axis).copied()
            }
            _ => {
                if axis < self.shape_len(ctx) {
                    Some(axis)
                } else {
                    None
                }
            }
        }
    }

    /// `apply_opt` returning `Err` instead of panicking on an invalid opt
    /// (python raises `KernelOptError`). Mirrors the asserts in `apply_opt`.
    pub fn try_apply_opt(
        &mut self,
        ctx: &mut UOpCtx,
        op: OptOp,
        axis: Option<usize>,
        arg: Option<i64>,
    ) -> Result<(), String> {
        if matches!(op, OptOp::Local | OptOp::Group | OptOp::GroupTop) && !self.ren.has_local {
            return Err("no locals".into());
        }
        let real_axis = self
            .real_axis(ctx, op, axis.unwrap_or(0))
            .ok_or("bad axis")?;
        let rng = self.rngs(ctx).get(real_axis).cloned().ok_or("axis oob")?;
        let amt = match arg {
            Some(0) | None => self.rng_vmax(ctx, rng.clone()) + 1,
            Some(a) => a,
        };
        let rtype = AxisType::range_axis_type(ctx, rng.clone());
        match op {
            OptOp::Unroll => {
                if amt > 32 {
                    return Err("unroll > 32".into());
                }
                if !matches!(rtype, AxisType::GroupReduce | AxisType::Reduce) {
                    return Err("unroll axis type".into());
                }
            }
            OptOp::Upcast => {
                if amt > 16 {
                    return Err("upcast > 16".into());
                }
                if !matches!(rtype, AxisType::Global | AxisType::Local | AxisType::Weak) {
                    return Err("upcast axis type".into());
                }
            }
            OptOp::Local => {
                if self.dont_use_locals {
                    return Err("no locals".into());
                }
                if !matches!(rtype, AxisType::Global | AxisType::Weak) {
                    return Err("local axis type".into());
                }
            }
            OptOp::Group | OptOp::GroupTop => {
                if self.dont_use_locals {
                    return Err("no locals".into());
                }
                if rtype != AxisType::Reduce {
                    return Err("group axis type".into());
                }
            }
            OptOp::PadTo | OptOp::Tc => return Err("opt handled elsewhere".into()),
        }
        // `shift_to` panics unless the extent divides evenly.
        let ext = ctx.srcs(&rng)[0].clone();
        if ctx.divides(ext, amt as i128).is_none() {
            return Err("extent not divisible".into());
        }
        let at = match op {
            OptOp::Local => AxisType::Local,
            OptOp::Upcast => AxisType::Upcast,
            OptOp::Unroll => AxisType::Unroll,
            OptOp::Group | OptOp::GroupTop => AxisType::GroupReduce,
            OptOp::PadTo | OptOp::Tc => unreachable!(),
        };
        self.shift_to(ctx, rng, amt, at, matches!(op, OptOp::GroupTop), None);
        self.applied_opts.push(Arg::Data(DataArg {
            tag: "Opt".into(),
            fields: vec![
                ("op".into(), Arg::Str(format!("{op:?}").to_uppercase())),
                ("axis".into(), Arg::Int(axis.unwrap_or(0) as i64)),
                ("arg".into(), Arg::Int(amt)),
            ],
        }));
        Ok(())
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

    /// `shape_str_to_axis` (postrange.py:41).
    pub fn shape_str_to_axis(&self, ctx: &mut UOpCtx, nms: &[String]) -> Vec<usize> {
        let ss = self.shape_str(ctx);
        nms.iter()
            .map(|n| {
                ss.iter()
                    .position(|x| x == n)
                    .unwrap_or_else(|| panic!("TGERR: shape_str_to_axis {n} not in {ss:?}"))
            })
            .collect()
    }

    /// `get_optimized_ast`.
    pub fn get_optimized_ast(self, ctx: &mut UOpCtx, name_override: Option<&str>) -> UopId {
        let name = match name_override {
            Some(n) => n.to_string(),
            None => {
                let k_type = if self.reduceop(ctx).is_some() {
                    "r"
                } else {
                    "E"
                };
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
            crate::dtype::ConstVal::Bool(b) => (if b { "True" } else { "False" }).to_string(),
            crate::dtype::ConstVal::Invalid => "Invalid".to_string(),
            crate::dtype::ConstVal::Float(f) => {
                if f == f.trunc() && f.abs() < 1e15 {
                    format!("{}", f as i64)
                } else {
                    format!("{}", f)
                }
            }
            other => panic!("TGERR: bad const render {other:?}"),
        },
        _ => match ctx.ssimplify(u) {
            Ok(SVal::I(v)) => format!("{}", v),
            _ => panic!("TGERR: symbolic extent render unsupported"),
        },
    }
}

/// Debug: dump every Index offset expr in a linear op list (post-linearize).
pub fn dump_lst_index_offsets(ctx: &mut UOpCtx, lst: &[UopId], label: &str) {
    fn pexpr(ctx: &mut UOpCtx, u: UopId, out: &mut String) {
        let (op, _dt, srcs, _arg, _tag) = ctx.snapshot(&u);
        match op {
            Ops::Const => {
                let _ = std::fmt::Write::write_fmt(out, format_args!("{}", sval_render(ctx, u)));
            }
            Ops::Range | Ops::Special => {
                let tag = match ctx.arg_of(&u) {
                    Arg::Str(s) => s.clone(),
                    _ => format!("r{}", AxisType::range_axis_id(ctx, u)),
                };
                let _ = std::fmt::Write::write_fmt(out, format_args!("{}", tag));
            }
            other => {
                if srcs.len() == 2 {
                    let _ = std::fmt::Write::write_fmt(out, format_args!("("));
                    for (i, s) in srcs.iter().enumerate() {
                        if i > 0 {
                            let _ = std::fmt::Write::write_fmt(out, format_args!(" {:?} ", other));
                        }
                        pexpr(ctx, s.clone(), out);
                    }
                    let _ = std::fmt::Write::write_fmt(out, format_args!(")"));
                } else {
                    let _ = std::fmt::Write::write_fmt(
                        out,
                        format_args!("{:?}[{}]", other, srcs.len()),
                    );
                }
            }
        }
    }
    for o in lst {
        if ctx.op_of(o) == Ops::Index {
            let isrcs = ctx.srcs(o).to_vec();
            if isrcs.len() < 2 {
                continue;
            }
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
            Ops::Const => {
                let _ = std::fmt::Write::write_fmt(out, format_args!("{}", sval_render(ctx, u)));
            }
            Ops::Range | Ops::Special => {
                let tag = match ctx.arg_of(&u) {
                    Arg::Str(s) => s.clone(),
                    _ => format!("r{}", AxisType::range_axis_id(ctx, u)),
                };
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
                        if i > 0 {
                            let _ = std::fmt::Write::write_fmt(out, format_args!(" {:?} ", other));
                        }
                        pexpr(ctx, s.clone(), out);
                    }
                    let _ = std::fmt::Write::write_fmt(out, format_args!(")"));
                } else {
                    let _ = std::fmt::Write::write_fmt(
                        out,
                        format_args!("{:?}[{}]", other, srcs.len()),
                    );
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
                    _ => (
                        String::new(),
                        format!("{:?}", AxisType::range_axis_type(ctx, o.clone())),
                    ),
                };
                eprintln!(
                    "[{}] rng {} ext={} at={} arg={:?}",
                    label,
                    tag,
                    sval_render(ctx, ext.clone()),
                    at,
                    ctx.arg_of(o)
                );
            }
            Ops::Index => {
                let isrcs = ctx.srcs(o).to_vec();
                if isrcs.len() < 2 {
                    continue;
                }
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
            (
                "axis_types".into(),
                Arg::Seq {
                    tuple: true,
                    items: vec![],
                },
            ),
            ("dont_use_locals".into(), Arg::Bool(dont_use_locals)),
            (
                "applied_opts".into(),
                Arg::Seq {
                    tuple: true,
                    items: opts.to_vec(),
                },
            ),
            ("opts_to_apply".into(), Arg::None),
            ("beam".into(), Arg::Int(0)),
        ],
    }
}

/// `OptOps` subset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptOp {
    /// Upcast an axis (`OptOps.UPCAST`).
    Upcast,
    /// Unroll an axis (`OptOps.UNROLL`).
    Unroll,
    /// Localize an axis (`OptOps.LOCAL`).
    Local,
    /// Group a reduce axis (`OptOps.GROUP`).
    Group,
    /// Group a reduce axis with a shifted top (`OptOps.GROUPTOP`).
    GroupTop,
    /// Pad an axis to a multiple (`OptOps.PADTO`).
    PadTo,
    /// Apply the tensor-core opt (`OptOps.TC`).
    Tc,
}

/// `hand_coded_optimizations` (heuristic.py) — corpus subset.
pub fn hand_coded_optimizations(k: &mut Scheduler, ctx: &mut UOpCtx) {
    // first try the tensor cores
    let use_tc: i64 = env_int("USE_TC", 1);
    let tc_opt: i64 = env_int("TC_OPT", 0);
    let tc_select: i64 = env_int("TC_SELECT", -1);
    if use_tc > 0
        && (k
            .axes_of(ctx, &[AxisType::GroupReduce, AxisType::Reduce])
            .len()
            == 1
            || tc_opt >= 1)
    {
        for axis in 0..3usize {
            let mut tk = k.copy();
            let mut rngs = match tk.apply_opt_tc(ctx, axis, tc_select, tc_opt, use_tc) {
                Ok(r) => r,
                Err(_) => continue,
            };
            // attempt to upcast M and N
            for &tc_dim in &[1usize, 0usize] {
                let mut pick: Option<i64> = None;
                for sz in [5i64, 4, 3, 2] {
                    let ext = ctx.srcs(&rngs[tc_dim])[0].clone();
                    if ctx.divides(ext, sz as i128).is_some() {
                        pick = Some(sz);
                        break;
                    }
                }
                if let Some(sz) = pick {
                    let idx = tk
                        .rngs(ctx)
                        .iter()
                        .position(|r| r == &rngs[tc_dim])
                        .unwrap();
                    let ret = tk.apply_opt(ctx, OptOp::Upcast, Some(idx), Some(sz));
                    rngs[tc_dim] = ret[0].clone();
                }
            }
            // attempt to local N
            let mut pick: Option<i64> = None;
            for sz in [4i64, 2] {
                let ext = ctx.srcs(&rngs[0])[0].clone();
                if ctx.divides(ext, sz as i128).is_some() {
                    pick = Some(sz);
                    break;
                }
            }
            if let Some(sz) = pick {
                let idx = tk.rngs(ctx).iter().position(|r| r == &rngs[0]).unwrap();
                tk.apply_opt(ctx, OptOp::Local, Some(idx), Some(sz));
            }
            *k = tk;
            return;
        }
    }

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
        if k.upcasted(ctx) == 0 && !ups.is_empty() && k.full_shape(ctx)[ups[ups.len() - 1]] % 4 == 0
        {
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
pub fn apply_opts(ctx: &mut UOpCtx, ast: UopId, ren: &Renderer, beam: usize) -> UopId {
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
        None if beam >= 1 => {
            k = crate::codegen::opt_search::beam_search(ctx, k, beam);
        }
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
            && let Arg::Str(s) = d.get("name")
        {
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
