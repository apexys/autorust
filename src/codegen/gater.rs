//! Port of tinygrad codegen/late/gater.py (pm_move_gates_from_index).

use crate::dtype::{BOOL, ConstVal as CV};
use crate::uop::UopId;
use crate::uop::arg::Arg;
use crate::uop::ctx::UOpCtx;
use crate::uop::ops::Ops;
use crate::uop::sym::{invalid_pat, r, sget, var};
use crate::uop::upat::{PatternMatcher, UPat, UPatOrConst};

fn vconst_like(ctx: &mut UOpCtx, l: UopId, val: i64) -> UopId {
    let dt = ctx.dt_of(&l);
    let c = ctx.const_py(CV::Int(val as i128), Some(dt));
    // mirror UOp.broadcast (ops.py:589): count==1 -> plain const, else STACK of copies.
    let n = ctx.max_numel(l) as usize;
    if n <= 1 {
        return c;
    }
    ctx.uop(Ops::Stack, Some(dt), vec![c; n], Arg::None)
}

fn store2(ctx: &mut UOpCtx, idx: UopId, srcs: Vec<UopId>) -> UopId {
    let all = std::iter::once(idx).chain(srcs).collect();
    ctx.uop(Ops::Store, Some(crate::dtype::VOID), all, Arg::None)
}

fn move_where_load(ctx: &mut UOpCtx, l: UopId, a: UopId, w: UopId) -> Option<UopId> {
    let lsrc = ctx.srcs(&l).to_vec();
    let alt = if ctx.is_invalid(&a) {
        vconst_like(ctx, l.clone(), 0)
    } else if ctx.op_of(&a) == Ops::Const {
        ctx.const_like(l.clone(), ctx.const_arg(&a)?)
    } else if ctx.op_of(&a) == Ops::Cast && ctx.dt_of(&ctx.srcs(&a)[0]) == ctx.dt_of(&l) {
        ctx.srcs(&a)[0].clone()
    } else {
        ctx.cast(a, ctx.dt_of(&l))
    };
    let nl = ctx.uop(
        Ops::Load,
        Some(ctx.dt_of(&l)),
        vec![(lsrc[0]).clone(), alt, (lsrc[2]).clone()],
        Arg::None,
    );
    Some(ctx.cast(nl, ctx.dt_of(&w)))
}

/// `pm_move_gates_from_index` (late/gater.py:9). The two image rules are not
/// ported (no image support yet).
pub fn pm_move_gates_from_index() -> PatternMatcher {
    {
        let mop = || {
            UPat::op_var(vec![Ops::Index, Ops::Shrink])
                .with_src_tup(vec![
                    UPat::var(None, None),
                    var("gate").where_(
                        UPatOrConst::Pat(var("idx")),
                        UPatOrConst::Pat(invalid_pat()),
                    ),
                ])
                .allow_all_srcs()
                .named("mop")
                .load(vec![], true)
                .named("l")
        };
        let load_pat = |gate_src: UPat| {
            UPat::op_var(vec![Ops::Load])
                .with_src_tup(vec![UPat::var(None, None), UPat::var(None, None), gate_src])
                .allow_all_srcs()
                .named("l")
                .or_casted(None)
        };
        PatternMatcher::new(vec![
            // INDEX/SHRINK with gated-Invalid index -> load with 0-alt + gate
            (
                mop(),
                r(|ctx, st, _| {
                    let (mop, gate, idx, l) = (
                        sget(st, "mop"),
                        sget(st, "gate"),
                        sget(st, "idx"),
                        sget(st, "l"),
                    );
                    let msrc = ctx.srcs(&mop).to_vec();
                    let new_mop = ctx.replace(
                        &mop,
                        ctx.op_of(&mop),
                        ctx.dt_of(&mop),
                        std::iter::once((msrc[0]).clone())
                            .chain(std::iter::once(idx))
                            .chain(msrc[2..].iter().cloned())
                            .collect(),
                        ctx.arg_of(&mop).clone(),
                        Arg::None,
                    );
                    let alt = vconst_like(ctx, l, 0);
                    Some(crate::tensor::load(ctx, new_mop, vec![alt, gate]))
                }),
            ),
            // same for stores
            (
                UPat::op_var(vec![Ops::Index, Ops::Shrink])
                    .with_src_tup(vec![
                        UPat::var(None, None),
                        var("gate").where_(
                            UPatOrConst::Pat(var("idx")),
                            UPatOrConst::Pat(invalid_pat()),
                        ),
                    ])
                    .allow_all_srcs()
                    .named("mop")
                    .store(vec![var("data")]),
                r(|ctx, st, _| {
                    let (mop, gate, idx, data) = (
                        sget(st, "mop"),
                        sget(st, "gate"),
                        sget(st, "idx"),
                        sget(st, "data"),
                    );
                    let msrc = ctx.srcs(&mop).to_vec();
                    let new_mop = ctx.replace(
                        &mop,
                        ctx.op_of(&mop),
                        ctx.dt_of(&mop),
                        std::iter::once((msrc[0]).clone())
                            .chain(std::iter::once(idx))
                            .chain(msrc[2..].iter().cloned())
                            .collect(),
                        ctx.arg_of(&mop).clone(),
                        Arg::None,
                    );
                    Some(store2(ctx, new_mop, vec![data, gate]))
                }),
            ),
            // WHERE(gate, gated_load, a) -> alt value moved into the load
            (
                var("gate")
                    .where_(
                        UPatOrConst::Pat(load_pat(UPat::var(Some("gate"), Some(vec![BOOL])))),
                        UPatOrConst::Pat(var("a")),
                    )
                    .named("w"),
                r(|ctx, st, _| {
                    let _ = sget(st, "gate");
                    let (l, a, w) = (sget(st, "l"), sget(st, "a"), sget(st, "w"));
                    move_where_load(ctx, l, a, w)
                }),
            ),
            // WHERE(gate, a, gated_load(~gate)) -> alt value moved into the load
            (
                var("gate")
                    .where_(
                        UPatOrConst::Pat(var("a")),
                        UPatOrConst::Pat(load_pat(UPat::logical_not(UPat::var(
                            Some("gate"),
                            Some(vec![BOOL]),
                        )))),
                    )
                    .named("w"),
                r(|ctx, st, _| {
                    let _ = sget(st, "gate");
                    let (l, a, w) = (sget(st, "l"), sget(st, "a"), sget(st, "w"));
                    move_where_load(ctx, l, a, w)
                }),
            ),
        ])
    }
}
