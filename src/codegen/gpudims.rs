//! `codegen/gpudims.py` port (M5) — convert GLOBAL/LOCAL/WARP/GROUP_REDUCE
//! ranges into SPECIAL globals. Corpus subset: no threads, no dont_use_locals,
//! no multi-device (DEVICE ranges are resolved by the scheduler).

use crate::dtype::AddrSpace;
use crate::uop::arg::Arg;
use crate::uop::ctx::UOpCtx;
use crate::uop::ops::Ops;
use crate::uop::range::AxisType;
use crate::uop::sint::SVal;
use crate::uop::upat::{substitute, PatternMatcher};
use crate::uop::UopId;
use std::collections::HashMap;

/// `_group_dims` (gpudims.py:8) — recursively merge adjacent dims that fit.
fn group_dims(dims: &[i64], max_sizes: &[i64]) -> Option<Vec<i64>> {
    if dims.len() > max_sizes.len() || dims.iter().zip(max_sizes.iter()).any(|(d, m)| d > m) {
        for i in 0..dims.len().saturating_sub(1) {
            if dims[i] * dims[i + 1] <= max_sizes[i] {
                let mut nd = dims[..i].to_vec();
                nd.push(dims[i] * dims[i + 1]);
                nd.extend_from_slice(&dims[i + 2..]);
                return group_dims(&nd, max_sizes);
            }
        }
        return None;
    }
    Some(dims.to_vec())
}

/// `_split_dims` — split oversized dims into factors.
fn split_dims(dims: &[i64], max_sizes: &[i64]) -> Vec<i64> {
    if dims.iter().zip(max_sizes.iter()).all(|(d, m)| d <= m) {
        return dims.to_vec();
    }
    let mut d = dims.to_vec();
    d.resize(3, 1);
    for i in 0..3 {
        while d[i] > max_sizes[i] {
            let n = d[i];
            let div = (2..=(n as f64).sqrt().ceil() as i64).find(|&k| n % k == 0).unwrap_or(1);
            if div == 1 {
                panic!("TGERR: cannot limit dim {d:?}");
            }
            let j = (i + 1) % 3;
            d[i] /= div;
            d[j] *= div;
        }
    }
    if d[2] == 1 {
        d[..2].to_vec()
    } else {
        d
    }
}

/// `UOp.special(s, name)`.
fn special(ctx: &mut UOpCtx, size: i64, name: &str) -> UopId {
    let eu = ctx.sint_to_uop_dt(SVal::I(size as i128), crate::dtype::WEAKINT);
    ctx.uop(Ops::Special, Some(crate::dtype::WEAKINT), vec![eu], Arg::Str(name.to_string()))
}

/// `get_grouped_dims` — returns per-dim UOp idx expressions.
fn get_grouped_dims(ctx: &mut UOpCtx, prefix: &str, dims: &[i64], max_sizes: Option<&[i64]>, reverse: bool) -> Vec<UopId> {
    if reverse {
        // tinygrad: `get_grouped_dims(prefix, dims[::-1], max_sizes)[::-1]` — the
        // specials are *named* in reversed dim order (gidx0 ↔ the last global
        // dim), then the result list is put back in `dims` order.
        let mut rd = dims.to_vec();
        rd.reverse();
        let mut r = get_grouped_dims(ctx, prefix, &rd, max_sizes, false);
        r.reverse();
        return r;
    }
    let limited = match max_sizes {
        None => dims.to_vec(),
        Some(ms) => {
            let grouped = group_dims(dims, ms);
            let lim = grouped.unwrap_or_else(|| dims.to_vec());
            if lim.len() > ms.len() {
                panic!("TGERR: cannot limit dim {dims:?}");
            }
            if lim == dims {
                split_dims(dims, ms)
            } else {
                lim
            }
        }
    };
    let raw_idxs: Vec<UopId> = limited.iter().enumerate().map(|(i, &s)| special(ctx, s, &format!("{prefix}{i}"))).collect();
    let mut out = Vec::with_capacity(dims.len());
    for i in 0..dims.len() {
        // flat = sum(idx * prod(limited[i+1:])); out[i] = flat // prod(dims[i+1:]) (mod dims[i] for i>0)
        let mut flat: Option<UopId> = None;
        for (j, idx) in raw_idxs.iter().enumerate() {
            let mult: i64 = limited[j + 1..].iter().product();
            let term = if mult == 1 {
                idx
            } else {
                let m = ctx.const_py(crate::dtype::ConstVal::Int(mult as i128), Some(crate::dtype::WEAKINT));
                &ctx.mul(idx.clone(), m)
            };
            flat = Some(match flat {
                None => term.clone(),
                Some(f) => ctx.add(f, term.clone()),
            });
        }
        let flat = flat.unwrap();
        let div_amount: i64 = dims[i + 1..].iter().product();
        let q = if div_amount == 1 {
            flat
        } else {
            let dv = ctx.const_py(crate::dtype::ConstVal::Int(div_amount as i128), Some(crate::dtype::WEAKINT));
            ctx.div_floor(flat, dv)
        };
        if i == 0 {
            out.push(crate::uop::upat::simplify(ctx, q));
        } else {
            let dm = ctx.const_py(crate::dtype::ConstVal::Int(dims[i] as i128), Some(crate::dtype::WEAKINT));
            let m = ctx.mod_(q, dm);
            out.push(crate::uop::upat::simplify(ctx, m));
        }
    }
    out
}

/// `add_gpudims` (gpudims.py:44).
pub fn add_gpudims(ctx: &mut UOpCtx, s: UopId) -> Option<UopId> {
    if matches!(ctx.arg_of(&s), Arg::None) {
        return None;
    }
    let s_topo = ctx.toposort(&s);
    if s_topo.iter().any(|x| ctx.op_of(x) == Ops::Special) {
        return None;
    }

    // get ranges: tinygrad keys each RANGE by `x.arg[0:-1]` — everything but
    // the trailing AxisType (`all_ranges = {x.arg[0:-1]: x ...}`). The `arg[1]`
    // split marker distinguishes the two halves of a `pm_split_ranges` split;
    // keying on the axis id alone (as this port did) collapses them onto one
    // gpu dim, rendering e.g. `gidx1*12800 + gidx1` instead of `*12800 + inner`.
    let range_head = |ctx: &UOpCtx, r: UopId| -> Vec<i64> {
        match ctx.arg_of(&r) {
            Arg::Seq { tuple: true, items } => items[..items.len() - 1]
                .iter()
                .map(|a| match a {
                    Arg::Int(i) => *i,
                    other => panic!("TGERR: bad RANGE arg head {other:?}"),
                })
                .collect(),
            other => panic!("TGERR: bad RANGE arg {other:?}"),
        }
    };
    let mut all_ranges: HashMap<Vec<i64>, UopId> = HashMap::new();
    for x in s_topo.iter() {
        if ctx.op_of(x) == Ops::Range {
            all_ranges.insert(range_head(ctx, x.clone()), x.clone());
        }
    }
    let mut global_dims: Vec<Vec<i64>> = all_ranges
        .iter()
        .filter(|(_, r)| matches!(AxisType::range_axis_type(ctx, (**r).clone()), AxisType::Global | AxisType::Thread))
        .map(|(k, _)| k.clone())
        .collect();
    let mut local_dims: Vec<Vec<i64>> = all_ranges
        .iter()
        .filter(|(_, r)| matches!(AxisType::range_axis_type(ctx, (**r).clone()), AxisType::Warp | AxisType::Local | AxisType::GroupReduce))
        .map(|(k, _)| k.clone())
        .collect();
    global_dims.sort();
    local_dims.sort();
    if global_dims.is_empty() && local_dims.is_empty() {
        return None;
    }

    let extent = |ctx: &mut UOpCtx, r: UopId| -> i64 {
        match ctx.ssimplify(ctx.srcs(&r)[0].clone()) {
            Ok(SVal::I(v)) => v as i64,
            _ => panic!("TGERR: symbolic gpudims extent"),
        }
    };
    let mut global_shape: Vec<i64> = Vec::with_capacity(global_dims.len());
    for k in global_dims.iter() {
        global_shape.push(extent(ctx, all_ranges[k].clone()));
    }
    let mut local_shape: Vec<i64> = Vec::with_capacity(local_dims.len());
    for k in local_dims.iter() {
        local_shape.push(extent(ctx, all_ranges[k].clone()));
    }

    // gpu-like execution: local then global idxs (has_threads=false, dont_use_locals=false)
    let local_idxs = get_grouped_dims(ctx, "lidx", &local_shape, Some(&[32768, 32768, 32768]), false);
    let global_idxs = get_grouped_dims(ctx, "gidx", &global_shape, Some(&[0x8FFFFFFF, 0x8FFFFFFF, 0x8FFFFFFF]), true);
    let mut idxs = global_idxs;
    idxs.extend(local_idxs);

    // ordered dims like python: (global_dims+local_dims) in sorted-key order
    let ordered: Vec<Vec<i64>> = global_dims.iter().chain(local_dims.iter()).cloned().collect();

    let mut subs: HashMap<UopId, UopId> = HashMap::new();
    for r in s_topo.iter() {
        if ctx.op_of(r) != Ops::Range {
            continue;
        }
        if AxisType::range_axis_type(ctx, r.clone()) == AxisType::Reduce {
            continue;
        }
        let key = range_head(ctx, r.clone());
        if let Some(ii) = ordered.iter().position(|k| *k == key) {
            subs.insert(r.clone(), (idxs[ii]).clone());
        }
    }
    if subs.is_empty() {
        None
    } else {
        Some(substitute(ctx, s, subs, false, false))
    }
}

/// `pm_add_gpudims` — SINK rule only (pm_device_to_var needs DEVICE ranges).
pub fn pm_add_gpudims() -> PatternMatcher {
    PatternMatcher::new(vec![(crate::uop::upat::UPat::ops1(Ops::Sink).allow_all_srcs().named("s"),
        crate::uop::sym::r(|ctx, st, _| add_gpudims(ctx, crate::uop::sym::sget(st, "s"))))])
}

// keep the AddrSpace import used (the store-mask path is corpus-deferred)
#[allow(dead_code)]
fn _assert_as(_: AddrSpace) {}
