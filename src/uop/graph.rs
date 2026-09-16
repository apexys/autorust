//! The interned UOp graph (arena + hash-consing), the substrate for M2's
//! rewrite engine. Mirrors Python's `UOpMetaClass.ucache` semantics: a node is
//! identified by `(op, dtype, srcs, arg, tag)`; identical nodes collapse.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

use super::arg::Arg;
use super::ops::Ops;
use crate::dtype::DTypeId;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct UopId(pub u32);

fn normalize_arg(a: Arg) -> Arg {
    match a {
        Arg::Float(x) if !x.is_finite() => {
            if x.is_nan() {
                Arg::NonFinite(super::arg::NonFinite::Nan)
            } else if x > 0.0 {
                Arg::NonFinite(super::arg::NonFinite::PosInf)
            } else {
                Arg::NonFinite(super::arg::NonFinite::NegInf)
            }
        }
        other => other,
    }
}

#[derive(Clone, Debug)]
pub struct Node {
    pub op: Ops,
    pub dtype: DTypeId,
    pub srcs: Vec<UopId>,
    pub arg: Arg,
    pub tag: Arg,
}

#[derive(Default, Clone)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub roots: Vec<UopId>,
    interner: HashMap<u64, u32>,
}

impl Graph {
    pub fn new() -> Self {
        Graph::default()
    }

    fn key_hash(op: Ops, dtype: DTypeId, srcs: &[UopId], arg: &Arg, tag: &Arg) -> u64 {
        let mut h = DefaultHasher::new();
        (op as u16).hash(&mut h);
        dtype.0.hash(&mut h);
        srcs.hash(&mut h);
        arg.hash(&mut h);
        tag.hash(&mut h);
        h.finish()
    }

    /// Insert or find an identical node; returns its id.
    /// Floats are normalized: non-finite values become `Arg::NonFinite` so the
    /// internal (and interning) representation is canonical.
    pub fn intern(&mut self, op: Ops, dtype: DTypeId, srcs: Vec<UopId>, arg: Arg, tag: Arg) -> UopId {
        let arg = normalize_arg(arg);
        let tag = normalize_arg(tag);
        let k = Self::key_hash(op, dtype, &srcs, &arg, &tag);
        if let Some(&id) = self.interner.get(&k) {
            return UopId(id);
        }
        let id = self.nodes.len() as u32;
        self.nodes.push(Node { op, dtype, srcs, arg, tag });
        self.interner.insert(k, id);
        UopId(id)
    }

    pub fn get(&self, id: UopId) -> &Node {
        &self.nodes[id.0 as usize]
    }

    /// Post-order DFS with dedup, mirroring a tinygrad `toposort` (srcs before parents).
    pub fn toposort_from(&self, root: UopId, out: &mut Vec<UopId>, seen: &mut Vec<bool>) {
        // iterative post-order
        enum Step {
            Enter(UopId),
            Emit(UopId),
        }
        let mut stack = vec![Step::Enter(root)];
        while let Some(s) = stack.pop() {
            match s {
                Step::Enter(id) => {
                    if seen[id.0 as usize] {
                        continue;
                    }
                    stack.push(Step::Emit(id));
                    // Python pushes `reversed(node.src)` so srcs are visited in forward order
                    for &src in self.get(id).srcs.iter().rev() {
                        if !seen[src.0 as usize] {
                            stack.push(Step::Enter(src));
                        }
                    }
                }
                Step::Emit(id) => {
                    if seen[id.0 as usize] {
                        continue;
                    }
                    seen[id.0 as usize] = true;
                    out.push(id);
                }
            }
        }
    }

    /// All nodes reachable from `roots`, in deterministic post-order.
    pub fn toposort(&self) -> Vec<UopId> {
        let mut out = Vec::new();
        let mut seen = vec![false; self.nodes.len()];
        // roots in reverse so the final order is root-first-stable like a recursive DFS
        for &r in self.roots.iter().rev() {
            self.toposort_from(r, &mut out, &mut seen);
        }
        out
    }

    /// JIT content key (tinygrad `ast.key`): a hash of the DAG reachable from
    /// `root`, referencing each edge by the **toposort position** of the source
    /// node rather than its arena id. Two structurally identical ASTs therefore
    /// hash equal even across arena resets / graph rebuilds, so the JIT cache
    /// survives node reclamation.
    pub fn content_key(&self, root: UopId) -> u64 {
        let mut order = Vec::new();
        let mut seen = vec![false; self.nodes.len()];
        self.toposort_from(root, &mut order, &mut seen);
        let mut pos: HashMap<u32, u64> = HashMap::with_capacity(order.len());
        for (i, &id) in order.iter().enumerate() {
            pos.insert(id.0, i as u64);
        }
        let mut h = DefaultHasher::new();
        (order.len() as u64).hash(&mut h);
        for &id in &order {
            let n = self.get(id);
            (n.op as u16).hash(&mut h);
            n.dtype.0.hash(&mut h);
            n.arg.hash(&mut h);
            for &s in &n.srcs {
                pos.get(&s.0).copied().unwrap_or(u64::MAX).hash(&mut h);
            }
            0u16.hash(&mut h); // node separator
        }
        h.finish()
    }
}

#[cfg(test)]
mod tests {
    use crate::uop::arg::Arg;
    use crate::uop::ctx::UOpCtx;
    use crate::uop::graph::UopId;
    use crate::uop::ops::Ops;

    fn build_target(ctx: &mut UOpCtx) -> UopId {
        let two = ctx.cint(2);
        let three = ctx.cint(3);
        let mul = ctx.uop(Ops::Mul, None, vec![two, three], Arg::None);
        let one = ctx.cint(1);
        ctx.uop(Ops::Add, None, vec![mul, one], Arg::None)
    }

    #[test]
    fn content_key_is_arena_id_independent() {
        let mut ca = UOpCtx::new();
        let root_a = build_target(&mut ca);

        let mut cb = UOpCtx::new();
        // pollute the arena so the target nodes land at *different* absolute ids
        let _j1 = cb.cint(900);
        let j7 = cb.cint(7);
        let j8 = cb.cint(8);
        let _j2 = cb.uop(Ops::Mul, None, vec![j7, j8], Arg::None);
        let root_b = build_target(&mut cb);

        assert_ne!(root_a, root_b, "ids must differ for this to prove anything");
        assert_eq!(ca.g.content_key(root_a), cb.g.content_key(root_b), "identical DAGs must share a JIT key");

        // structurally different graph -> different key
        let mut cc = UOpCtx::new();
        let c2a = cc.cint(2);
        let c3 = cc.cint(3);
        let m = cc.uop(Ops::Mul, None, vec![c2a, c3], Arg::None);
        let c2b = cc.cint(2);
        let root_c = cc.uop(Ops::Add, None, vec![m, c2b], Arg::None);
        assert_ne!(ca.g.content_key(root_a), cc.g.content_key(root_c), "different DAGs must not collide");
    }
}
