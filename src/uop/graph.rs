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
}
