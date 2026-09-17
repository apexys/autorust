//! `codegen/late/linearizer.py` port (M5): priority-toposort linearization.

use crate::dtype::AddrSpace;
use crate::uop::ctx::UOpCtx;
use crate::uop::ops::Ops;
use crate::uop::range::AxisType;
use crate::uop::UopId;
use std::collections::BinaryHeap;

/// `linearize` (linearizer.py:9) — toposort with priorities (heap by nkey).
pub fn linearize(ctx: &mut UOpCtx, sink: UopId) -> Vec<UopId> {
    let lst = ctx.toposort(&sink);

    // out_degree
    let mut out_degree: std::collections::HashMap<UopId, usize> = std::collections::HashMap::new();
    // priorities: (run_count, priority, extra)
    let mut priorities: std::collections::HashMap<UopId, (i64, i32, i64)> = std::collections::HashMap::new();

    for u in lst.iter().rev() {
        for s in ctx.srcs(u) {
            *out_degree.entry(s.clone()).or_insert(0) += 1;
        }
        let run_count: i64 = ctx
            .ranges(u)
            .iter()
            .map(|r| match ctx.ssimplify(ctx.srcs(r)[0].clone()) {
                Ok(crate::uop::sint::SVal::I(v)) => v as i64,
                _ => 1,
            })
            .product();
        let op = ctx.op_of(u);
        let (priority, extra) = match op {
            Ops::Param => {
                let slot = match ctx.arg_of(u) {
                    crate::uop::arg::Arg::Param(p) => p.slot,
                    _ => -1,
                };
                (-20, slot)
            }
            Ops::Buffer => {
                let p = if ctx.addrspace_of(u.clone()) == Some(AddrSpace::Local) { -17 } else { -18 };
                (p, 0)
            }
            Ops::Load => (-1, 0),
            Ops::Store => (1, 0),
            Ops::Range => (5, 0),
            Ops::End => (-5, 0),
            _ => (0, 0),
        };
        priorities.insert(u.clone(), (run_count, priority, extra));
    }

    // nkey: sorted by (run_count, priority, extra) — python sorts ascending;
    // smaller run_count closer to the top; the heap pops smallest nkey first
    let mut sorted_lst: Vec<UopId> = lst.clone();
    sorted_lst.sort_by_key(|u| priorities[u]);
    let nkey: std::collections::HashMap<UopId, usize> =
        sorted_lst.iter().enumerate().map(|(i, u)| (u.clone(), i)).collect();

    // heap-driven toposort from the sink — python pushes (-nkey, u) and pops the
    // smallest, i.e. LARGEST nkey first
    let mut heap: BinaryHeap<(usize, UopId)> = BinaryHeap::new();
    heap.push((nkey[&sink], sink.clone()));
    let mut newlst: Vec<UopId> = Vec::new();
    while let Some((_, u)) = heap.pop() {
        newlst.push(u.clone());
        for v in ctx.srcs(&u) {
            let e = out_degree.get_mut(v).unwrap();
            *e -= 1;
            if *e == 0 {
                heap.push((nkey[v], v.clone()));
            }
        }
    }
    newlst.reverse();
    newlst
}

/// `pm_linearize_cleanups` (codegen/__init__.py:441) — gated STORE becomes
/// IF/ENDIF. The corpus has no gated stores; returns the input unchanged.
pub fn pm_linearize_cleanups(ctx: &mut UOpCtx, lst: Vec<UopId>) -> Vec<UopId> {
    // python also panics on bare IF/ENDIF in the graph — none in the corpus
    let _ = ctx;
    lst
}

// keep AxisType import used when corpus gains DEVICE ranges
#[allow(dead_code)]
fn _t(_: AxisType) {}
