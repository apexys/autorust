//! `codegen/late/coalesce.py` port (M5): memory coalescing — group scalar
//! loads/stores on the same buffer with consecutive const offsets into
//! vectorized SHRINK accesses. Corpus subset: no DSP, no images, no REG
//! buffers, no gated loads/stores, no half8.

use crate::dtype::{self, ConstVal};
use crate::uop::arg::Arg;
use crate::uop::ctx::UOpCtx;
use crate::uop::ops::{group, Ops};

use crate::uop::UopId;
use std::collections::HashMap;

/// is the valid mask a plain `const(True)` (no invalid gating)?
fn valid_is_true(ctx: &mut UOpCtx, valid: UopId) -> bool {
    ctx.op_of(valid) == Ops::Const && matches!(ctx.val(valid), ConstVal::Bool(true))
}

/// `memory_coalescing` (coalesce.py:117).
pub fn memory_coalescing(ctx: &mut UOpCtx, sink: UopId) -> UopId {
    // collect: (op, buf, root_src, valid) -> offset -> [uops]
    let mut memory: Vec<((Ops, UopId, Key, UopId), HashMap<i64, Vec<UopId>>)> = Vec::new();
    for &u in ctx.toposort(sink).iter() {
        let op = ctx.op_of(u);
        if op != Ops::Load && op != Ops::Store {
            continue;
        }
        let srcs = ctx.srcs(u).to_vec();
        let idx_u = srcs[0];
        if ctx.op_of(idx_u) != Ops::Index {
            continue; // gated forms unsupported in corpus
        }
        let isrcs = ctx.srcs(idx_u).to_vec();
        let buf = isrcs[0];
        // REG buffers skipped
        if ctx.addrspace_of(buf) == Some(dtype::AddrSpace::Reg) {
            continue;
        }
        // python: buf, idx_u = u.src[0].src — the idx EXPR, not the INDEX node
        let idx_expr = isrcs[1];
        let idx = ctx.get_idx(idx_expr);
        let valid = ctx.get_valid(idx_expr);
        let iop = ctx.op_of(idx);
        let isrcs_i = ctx.srcs(idx).to_vec();
        let key: Key;
        let mut offset: i64 = 0;
        let mut base_uop: Option<UopId> = None;
        if iop == Ops::Add {
            // (x + const) or (const + x)
            let (a, b) = (isrcs_i[0], isrcs_i[1]);
            if ctx.op_of(b) == Ops::Const {
                if let ConstVal::Int(v) = ctx.val(b) {
                    base_uop = Some(a);
                    offset = v as i64;
                }
            } else if ctx.op_of(a) == Ops::Const {
                if let ConstVal::Int(v) = ctx.val(a) {
                    base_uop = Some(b);
                    offset = v as i64;
                }
            }
            key = match base_uop {
                Some(u) => Key::Uop(u),
                None => Key::Uop(idx),
            };
        } else if iop == Ops::Const {
            match ctx.val(idx) {
                ConstVal::Int(v) => {
                    key = Key::Const;
                    offset = v as i64;
                }
                _ => key = Key::Uop(idx),
            }
        } else {
            key = Key::Uop(idx);
        }
        let k = (op, buf, key, valid);
        match memory.iter_mut().find(|(kk, _)| *kk == k) {
            Some((_, m)) => m.entry(offset).or_default().push(u),
            None => {
                let mut m = HashMap::new();
                m.insert(offset, vec![u]);
                memory.push((k, m));
            }
        }
    }

    // build replacements
    let mut replacements: Vec<(UopId, UopId)> = Vec::new();
    for ((op, buf, base, valid), offsets) in memory.iter() {
        // allowed lengths (python defaults, supports_float4=true)
        let mut lengths: Vec<i64> = vec![4, 2];
        lengths.push(1);
        let _ = group::ELEMENTWISE;

        // group consecutive offsets
        let mut sorted: Vec<i64> = offsets.keys().copied().collect();
        sorted.sort();
        let mut groups: Vec<Vec<i64>> = Vec::new();
        let mut cur: Vec<i64> = Vec::new();
        for (i, &o) in sorted.iter().enumerate() {
            if let Some(&last) = cur.last() {
                if o == last + 1 {
                    cur.push(o);
                } else {
                    groups.push(std::mem::take(&mut cur));
                    cur.push(o);
                }
            } else {
                cur.push(o);
            }
            let _ = i;
        }
        if !cur.is_empty() {
            groups.push(cur);
        }

        for full_grp in groups {
            let mut full_grp: Vec<i64> = full_grp;
            while !full_grp.is_empty() {
                let first = full_grp[0];
                let base_node: UopId = match *base {
                    Key::Const => ctx.const_py(ConstVal::Int(first as i128), Some(crate::dtype::WEAKINT)),
                    Key::Uop(u) => {
                        if first == 0 {
                            u
                        } else {
                            let c = ctx.const_py(ConstVal::Int(first as i128), Some(crate::dtype::WEAKINT));
                            ctx.add(u, c)
                        }
                    }
                };
                // NOTE: python computes offset = base+full_grp[0] once per group; the
                // divides() gate applies to that offset node.
                let mut length = 1usize;
                for &l in lengths.iter() {
                    if l <= full_grp.len() as i64 && ctx.divides(base_node, l as i128).is_some() {
                        length = l as usize;
                        break;
                    }
                }
                let grp: Vec<i64> = full_grp[..length].to_vec();
                // apply valid again after length determination (corpus: const-true valids only)
                let base_node = if valid_is_true(ctx, *valid) { base_node } else { ctx.valid(base_node, *valid) };
                let idx: UopId = if grp.len() > 1 {
                    // SHRINK pairs are (start, size): the group is n elements at base.
                    let n = ctx.const_py(ConstVal::Int(grp.len() as i128), Some(crate::dtype::WEAKINT));
                    ctx.uop(Ops::Shrink, Some(ctx.dt_of(*buf)), vec![*buf, base_node, n], Arg::None)
                } else {
                    ctx.uop(Ops::Index, Some(ctx.dt_of(*buf)), vec![*buf, base_node], Arg::None)
                };
                if *op == Ops::Store {
                    let mut datas = Vec::new();
                    for &g in &grp {
                        let us = offsets.get(&g).unwrap();
                        assert!(us.len() == 1, "TGERR: attempting multiple stores");
                        datas.push(ctx.srcs(us[0])[1]);
                    }
                    let val = if datas.len() > 1 {
                        ctx.uop(Ops::Stack, Some(ctx.dt_of(datas[0])), datas, Arg::None)
                    } else {
                        datas[0]
                    };
                    let store = ctx.uop(Ops::Store, Some(crate::dtype::VOID), vec![idx, val], Arg::None);
                    for &g in &grp {
                        for &uo in offsets.get(&g).unwrap() {
                            replacements.push((uo, store));
                        }
                    }
                } else {
                    let ld = crate::tensor::load(ctx, idx, vec![]);
                    for (i, &g) in grp.iter().enumerate() {
                        for &uo in offsets.get(&g).unwrap() {
                            if grp.len() > 1 {
                                let ci = ctx.const_py(ConstVal::Int(i as i128), Some(crate::dtype::WEAKINT));
                                let r = ctx.uop(Ops::Index, Some(ctx.dt_of(ld)), vec![ld, ci], Arg::None);
                                replacements.push((uo, r));
                            } else {
                                replacements.push((uo, ld));
                            }
                        }
                    }
                }
                full_grp = full_grp[length..].to_vec();
                let _ = first;
            }
        }
    }

    if replacements.is_empty() {
        sink
    } else {
        let map: HashMap<UopId, UopId> = replacements.into_iter().collect();
        crate::uop::upat::substitute(ctx, sink, map, false, false)
    }
}

/// base key for the memory map: uop base or plain const.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Key {
    Const,
    Uop(UopId),
}

// ---------- indexing_simplify (late/coalesce.py:57) ----------

/// `simplify_valid_load` (late/coalesce.py:40).
fn simplify_valid_load(ctx: &mut UOpCtx, buf: UopId, start_idx: UopId, valid: UopId) -> Option<UopId> {
    let idx = crate::uop::sym::uop_given_valid(ctx, valid, start_idx, true);
    if idx == start_idx {
        return None;
    }
    let start_s = crate::uop::upat::simplify(ctx, start_idx);
    if idx == start_s {
        return None;
    }
    let iv = ctx.valid(idx, valid);
    Some(ctx.index(buf, vec![iv]))
}

/// `indexing_simplify` (late/coalesce.py:57) — scalar rule; the image rule is
/// not ported (no images until CL image support).
pub fn indexing_simplify() -> &'static crate::uop::upat::PatternMatcher {
    static PM: std::sync::OnceLock<crate::uop::upat::PatternMatcher> = std::sync::OnceLock::new();
    PM.get_or_init(|| {
        use crate::uop::sym::{r, sget};
        use crate::uop::upat::{PatternMatcher, UPat};
        PatternMatcher::new(vec![(
            UPat::op_var(vec![Ops::Index])
                .with_src_tup(vec![
                    crate::uop::sym::var("buf"),
                    crate::uop::sym::invalid_gate(),
                ]),
            r(|ctx, st, _| {
                let (buf, x, cond) = (sget(st, "buf"), sget(st, "x"), sget(st, "cond"));
                simplify_valid_load(ctx, buf, x, cond)
            }),
        )])
    })
}
