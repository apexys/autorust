//! The interned UOp graph, the substrate for the rewrite engine.
//!
//! Mirrors Python's `UOpMetaClass.ucache`: an immutable node is identified by
//! `(op, dtype, srcs, arg, tag)` and interned through a **weak** table, so a node
//! stays alive exactly as long as some strong handle reaches it. A strong handle
//! is [`UopId`] itself (an `Rc<Node>`): a `Tensor` holds one, and `Node.srcs`
//! holds its children. When the last handle drops, the node (and recursively any
//! child no longer referenced elsewhere) is freed — no GC pass, no safe point,
//! and no id reuse. This is what bounds memory across repeated computation.

use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::rc::{Rc, Weak};
use std::sync::atomic::{AtomicU32, Ordering};

use super::arg::Arg;
use super::ops::Ops;
use crate::dtype::{AddrSpace, DTypeId};
use crate::uop::sint::SVal;

/// Process-wide monotonic node id. Unique across every arena ever built, so it is
/// safe as an identity/order key and for cross-arena bookkeeping (JIT cache,
/// bridge flat format).
static NEXT_NODE_ID: AtomicU32 = AtomicU32::new(0);

fn next_id() -> u32 {
    NEXT_NODE_ID.fetch_add(1, Ordering::Relaxed)
}

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

/// `_shape` cache value (`Vec<SVal>`; `None` = op has no shape).
pub type Shape = Vec<SVal>;
/// `_min_max` cache value.
pub type MinMax = (SVal, SVal);

/// An immutable UOp node. Fields mirror Python's `UOp.__slots__`; `srcs` holds
/// **strong** handles to children, so a handle to a node keeps its whole subgraph
/// alive. Property caches are interior-mutable so they can be filled through a
/// shared handle and are freed with the node.
#[derive(Debug)]
pub struct Node {
    /// Process-unique node id, used for identity, ordering and hashing.
    pub id: u32,
    /// The op code.
    pub op: Ops,
    /// Result dtype.
    pub dtype: DTypeId,
    /// Src operands (strong handles).
    pub srcs: Vec<UopId>,
    /// The op's argument.
    pub arg: Arg,
    /// Python `tag` metadata for tracing.
    pub tag: Arg,
    // lazily computed, memoized recursive properties (one slot per property):
    /// Cached `u.shape`.
    pub shape_c: RefCell<Option<Option<Rc<Shape>>>>,
    /// Cached `u._min_max` bounds.
    pub mm_c: RefCell<Option<Option<MinMax>>>,
    /// Cached `u.ranges` for this node.
    pub range_c: RefCell<Option<Option<Rc<Vec<UopId>>>>>,
    /// `u.addrspace` derived (`recursive_property`, ops.py:866). Cached so the
    /// recursive walk is memoized (linear, not exponential, on shared DAGs).
    pub addrspace_c: RefCell<Option<Option<AddrSpace>>>,
    /// `Tensor(data)` staged host bytes for a base BUFFER node (python
    /// `Buffer(opaque=...)`). Lives with the node, so it is freed when the
    /// buffer is, instead of leaking in a ctx-wide side table.
    pub initial_data: RefCell<Option<Rc<RefCell<Vec<u8>>>>>,
    /// Autograd gradient attached to this node (tinygrad `Tensor.grad`). Stored
    /// on the node (interior-mutable) so it dies with the graph rather than
    /// growing a ctx-wide map; the value is a strong handle so the gradient
    /// subgraph stays alive while attached.
    pub grad: RefCell<Option<UopId>>,
}

// Node identity is its process-unique id (two handles to the same node are the
// same node). Implemented manually so the interior-mutable side tables are not
// compared (and `grad` cannot recurse through itself).
impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for Node {}

/// A strong handle to a node. `Clone` bumps the reference count; the last drop
/// frees the node (and any child it uniquely owns). Identity/order/hash are the
/// node's process-unique id.
#[derive(Clone)]
pub struct UopId {
    /// The referenced node.
    pub node: Rc<Node>,
}

impl UopId {
    /// The node's process-unique id.
    #[inline]
    pub fn id(&self) -> u32 {
        self.node.id
    }
    /// The referenced node.
    #[inline]
    pub fn node(&self) -> &Node {
        &self.node
    }
}

impl std::ops::Deref for UopId {
    type Target = Node;
    #[inline]
    fn deref(&self) -> &Node {
        &self.node
    }
}

impl PartialEq for UopId {
    fn eq(&self, other: &Self) -> bool {
        self.node.id == other.node.id
    }
}
impl Eq for UopId {}
impl Hash for UopId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.node.id.hash(state);
    }
}
impl PartialOrd for UopId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for UopId {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.node.id.cmp(&other.node.id)
    }
}
impl std::fmt::Debug for UopId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "UopId({}: {:?})", self.node.id, self.node.op)
    }
}

/// The weak interning table + external roots. Nodes are owned by handles
/// (`UopId`), not by the graph.
#[derive(Default)]
pub struct Graph {
    /// Externally held roots, classified as outside the weak interner.
    pub roots: Vec<UopId>,
    /// Structural-hash → node weak handle, for interning.
    interner: HashMap<u64, Weak<Node>>,
}

impl Clone for Graph {
    /// Clone the root list and a fresh (empty) interner: the nodes themselves are
    /// shared via the strong handles in `roots`.
    fn clone(&self) -> Self {
        Graph {
            roots: self.roots.clone(),
            interner: HashMap::new(),
        }
    }
}

impl Graph {
    /// An empty graph with no roots and an empty interner.
    pub fn new() -> Self {
        Graph::default()
    }

    fn key_hash(op: Ops, dtype: DTypeId, srcs: &[UopId], arg: &Arg, tag: &Arg) -> u64 {
        let mut h = DefaultHasher::new();
        (op as u16).hash(&mut h);
        dtype.0.hash(&mut h);
        // Hash by child node id (stable, deterministic) rather than by Rc address.
        for s in srcs {
            s.id().hash(&mut h);
        }
        arg.hash(&mut h);
        tag.hash(&mut h);
        h.finish()
    }

    /// Insert or find an identical node; returns a strong handle. Floats are
    /// normalized so the internal (and interning) representation is canonical.
    pub fn intern(
        &mut self,
        op: Ops,
        dtype: DTypeId,
        srcs: Vec<UopId>,
        arg: Arg,
        tag: Arg,
    ) -> UopId {
        let arg = normalize_arg(arg);
        let tag = normalize_arg(tag);
        // Opportunistic prune of dead weak entries so the table cannot grow
        // without bound over a long-running program.
        if self.interner.len() > 4096 && self.interner.len().is_multiple_of(1024) {
            self.interner.retain(|_, w| w.strong_count() > 0);
        }
        let k = Self::key_hash(op, dtype, &srcs, &arg, &tag);
        if let Some(w) = self.interner.get(&k)
            && let Some(n) = w.upgrade()
        {
            return UopId { node: n };
        }
        let node = Rc::new(Node {
            id: next_id(),
            op,
            dtype,
            srcs,
            arg,
            tag,
            shape_c: RefCell::new(None),
            mm_c: RefCell::new(None),
            range_c: RefCell::new(None),
            addrspace_c: RefCell::new(None),
            initial_data: RefCell::new(None),
            grad: RefCell::new(None),
        });
        self.interner.insert(k, Rc::downgrade(&node));
        UopId { node }
    }

    /// Number of live interned nodes (for diagnostics/tests).
    pub fn live_len(&self) -> usize {
        self.interner
            .values()
            .filter(|w| w.strong_count() > 0)
            .count()
    }

    /// Post-order DFS with dedup, mirroring a tinygrad `toposort` (srcs before parents).
    pub fn toposort_from(
        root: &UopId,
        out: &mut Vec<UopId>,
        seen: &mut std::collections::HashSet<u32>,
    ) {
        // iterative post-order
        enum Step {
            Enter(UopId),
            Emit(UopId),
        }
        let mut stack = vec![Step::Enter(root.clone())];
        while let Some(s) = stack.pop() {
            match s {
                Step::Enter(id) => {
                    if seen.contains(&id.id()) {
                        continue;
                    }
                    stack.push(Step::Emit(id.clone()));
                    // Python pushes `reversed(node.src)` so srcs are visited in forward order
                    for src in id.srcs.iter().rev() {
                        if !seen.contains(&src.id()) {
                            stack.push(Step::Enter(src.clone()));
                        }
                    }
                }
                Step::Emit(id) => {
                    if seen.contains(&id.id()) {
                        continue;
                    }
                    seen.insert(id.id());
                    out.push(id);
                }
            }
        }
    }

    /// All nodes reachable from `roots`, in deterministic post-order.
    pub fn toposort(&self) -> Vec<UopId> {
        let mut out = Vec::new();
        let mut seen = std::collections::HashSet::new();
        // roots in reverse so the final order is root-first-stable like a recursive DFS
        for r in self.roots.iter().rev() {
            Self::toposort_from(r, &mut out, &mut seen);
        }
        out
    }

    /// `toposort` from an explicit root.
    pub fn toposort_root(&self, root: &UopId) -> Vec<UopId> {
        let mut out = Vec::new();
        let mut seen = std::collections::HashSet::new();
        Self::toposort_from(root, &mut out, &mut seen);
        out
    }

    /// JIT content key (tinygrad `ast.key`): a hash of the DAG reachable from
    /// `root`, referencing each edge by the **toposort position** of the source
    /// node rather than its arena id, so structurally identical ASTs hash equal.
    pub fn content_key(&self, root: &UopId) -> u64 {
        let order = Self::toposort_from_pub(root);
        let mut pos: HashMap<u32, u64> = HashMap::with_capacity(order.len());
        for (i, id) in order.iter().enumerate() {
            pos.insert(id.id(), i as u64);
        }
        let mut h = DefaultHasher::new();
        (order.len() as u64).hash(&mut h);
        for id in &order {
            (id.op as u16).hash(&mut h);
            id.dtype.0.hash(&mut h);
            id.arg.hash(&mut h);
            for s in &id.srcs {
                pos.get(&s.id()).cloned().unwrap_or(u64::MAX).hash(&mut h);
            }
            0u16.hash(&mut h); // node separator
        }
        h.finish()
    }

    fn toposort_from_pub(root: &UopId) -> Vec<UopId> {
        let mut out = Vec::new();
        let mut seen = std::collections::HashSet::new();
        Self::toposort_from(root, &mut out, &mut seen);
        out
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
        assert_eq!(
            ca.g.content_key(&root_a),
            cb.g.content_key(&root_b),
            "identical DAGs must share a JIT key"
        );

        // structurally different graph -> different key
        let mut cc = UOpCtx::new();
        let c2a = cc.cint(2);
        let c3 = cc.cint(3);
        let m = cc.uop(Ops::Mul, None, vec![c2a, c3], Arg::None);
        let c2b = cc.cint(2);
        let root_c = cc.uop(Ops::Add, None, vec![m, c2b], Arg::None);
        assert_ne!(
            ca.g.content_key(&root_a),
            cc.g.content_key(&root_c),
            "different DAGs must not collide"
        );
    }

    #[test]
    fn nodes_free_when_last_handle_drops() {
        let mut ctx = UOpCtx::new();
        let a = ctx.cint(5);
        let b = ctx.cint(6);
        let sum = ctx.uop(Ops::Add, None, vec![a.clone(), b.clone()], Arg::None);
        assert_eq!(ctx.g.live_len(), 3);
        drop(sum);
        // `a` and `b` are still held here; drop them too
        drop(a);
        drop(b);
        assert_eq!(ctx.g.live_len(), 0, "all nodes freed once handles drop");
    }
}
