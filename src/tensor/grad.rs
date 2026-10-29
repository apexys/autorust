//! Faithful port of `tinygrad/mixin/gradient.py`: `compute_gradient` + the
//! `pm_gradient` rewrite rules + `reduce_gradient`, operating on the UOp graph.
//!
//! tinygrad expresses `pm_gradient` as a `PatternMatcher`, but that matcher only
//! rewrites a node to a *single* replacement uop, whereas each gradient rule
//! emits one gradient *per source* (a `Vec<Option<UopId>>`). Every rule is keyed
//! on the node's `op`, so we dispatch with an explicit `match` instead — the rule
//! bodies are ported one-for-one. The `FUNCTION`/`CALL`/`GETTUPLE` machinery
//! (`call_gradient`, `_compact_params`) is intentionally omitted: eager `Tensor`
//! graphs never build those nodes, so XOR training never reaches them.
//!
//! Gradients are written back into [`UOpCtx::grads`] keyed by the target `uop`.

use std::collections::{HashMap, HashSet};

use crate::dtype::ConstVal;
use crate::uop::arg::Arg;
use crate::uop::ctx::UOpCtx;
use crate::uop::graph::UopId;
use crate::uop::ops::Ops;
use crate::uop::sint::SVal;

use super::Tensor;
use super::movement;
use super::reduce;

const LN_2: f64 = std::f64::consts::LN_2;
const FRAC_PI_2: f64 = std::f64::consts::FRAC_PI_2;

/// True division as the graph knows it: `a * reciprocal(b)`.
fn fdiv(ctx: &mut UOpCtx, a: UopId, b: UopId) -> UopId {
    let r = ctx.reciprocal(b);
    ctx.mul(a, r)
}

/// `helpers.argsort`: indices that would sort `a` ascending.
fn argsort(a: &[usize]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..a.len()).collect();
    idx.sort_by_key(|&i| a[i]);
    idx
}

/// Parse a `REDUCE` node's `(op, num_axes)` argument (`Arg::Seq{[Enum(op), Int(n)]}`).
fn reduce_arg(ctx: &UOpCtx, ret: UopId) -> (Ops, usize) {
    if let Arg::Seq { items, .. } = ctx.arg_of(&ret)
        && items.len() == 2
    {
        let op = match &items[0] {
            Arg::Enum(e) => {
                Ops::from_u16(e.value).unwrap_or_else(|| panic!("TGERR: bad REDUCE op {}", e.value))
            }
            _ => panic!("TGERR: REDUCE arg[0] not an op enum"),
        };
        let n = match &items[1] {
            Arg::Int(n) => *n as usize,
            _ => panic!("TGERR: REDUCE arg[1] not an int"),
        };
        return (op, n);
    }
    panic!("TGERR: malformed REDUCE arg");
}

/// Decode an `Arg::Seq` of ints into `Vec<usize>` (used for PERMUTE order).
fn seq_ints(arg: &Arg) -> Vec<usize> {
    if let Arg::Seq { items, .. } = arg {
        items
            .iter()
            .map(|a| match a {
                Arg::Int(i) => *i as usize,
                _ => panic!("TGERR: expected int in seq"),
            })
            .collect()
    } else {
        panic!("TGERR: expected Arg::Seq");
    }
}

/// Decode an `Arg::Seq` of bools into a `Vec<usize>` of true positions (FLIP axes).
fn seq_trues(arg: &Arg) -> Vec<usize> {
    if let Arg::Seq { items, .. } = arg {
        items
            .iter()
            .enumerate()
            .filter_map(|(i, a)| {
                if matches!(a, Arg::Bool(true)) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    } else {
        panic!("TGERR: expected Arg::Seq");
    }
}

/// `gradient.py:8 reduce_gradient` — gradient of a `REDUCE` wrt its single source.
fn reduce_gradient(
    ctx: &mut UOpCtx,
    grad: UopId,
    ret: UopId,
    op: Ops,
    naxes: usize,
) -> Vec<Option<UopId>> {
    let x = ctx.srcs(&ret)[0].clone();
    let axes: Vec<usize> = (0..naxes).collect();
    match op {
        Ops::Add => {
            let sx = ctx.shape_of(&x).unwrap();
            let gr = movement::broadcast_to(ctx, grad, &sx);
            vec![Some(gr)]
        }
        Ops::Max => {
            let dt = ctx.dt_of(&grad);
            let eq = ctx.eq(x, ret);
            let mask = ctx.cast(eq, dt);
            let denom = reduce::rop(ctx, Ops::Add, mask.clone(), &axes);
            let ratio = fdiv(ctx, mask, denom);
            let out = ctx.mul(ratio, grad);
            vec![Some(out)]
        }
        Ops::Mul => {
            let one = ctx.const_like(x.clone(), ConstVal::Int(1));
            let x0 = ctx.const_like(x.clone(), ConstVal::Int(0));
            let is_zero = ctx.eq(x.clone(), x0);
            let safe_x = ctx.where_(is_zero.clone(), one, x);
            let iz_dt = ctx.dt_of(&is_zero);
            let sadt = reduce::sum_acc_dtype(iz_dt);
            let iz_cast = ctx.cast(is_zero.clone(), sadt);
            let zero_count = reduce::rop(ctx, Ops::Add, iz_cast, &axes);
            let safe_prod = reduce::rop(ctx, Ops::Mul, safe_x.clone(), &axes);
            let one_c = ctx.const_like(zero_count.clone(), ConstVal::Int(1));
            let only_zero = ctx.eq(zero_count.clone(), one_c);
            let zero_c = ctx.const_like(zero_count, ConstVal::Int(0));
            let others = ctx.where_(only_zero, safe_prod, zero_c);
            let ret_safe = fdiv(ctx, ret, safe_x);
            let val = ctx.where_(is_zero, others, ret_safe);
            let out = ctx.mul(grad, val);
            vec![Some(out)]
        }
        _ => panic!("TGERR: unsupported reduce op in reduce_gradient: {:?}", op),
    }
}

/// The `pm_gradient` rules (`gradient.py:51`), dispatched on `ret`'s op. Returns one
/// gradient per source (or `None` = no gradient flows through that edge). `None`
/// return (the whole node) means "unsupported op" — `compute_gradient` raises then.
fn grad_of_op(ctx: &mut UOpCtx, ret: UopId, g: UopId) -> Option<Vec<Option<UopId>>> {
    let srcs = ctx.srcs(&ret).to_vec();
    let n = srcs.len();
    let op = ctx.op_of(&ret);
    // helpers
    let zero_g = ctx.const_like(g.clone(), ConstVal::Float(0.0));
    let ln2 = ctx.cfloat(LN_2);
    match op {
        Ops::Cast => {
            let dt = ctx.dt_of(&srcs[0]);
            let gr = ctx.cast(g, dt);
            Some(vec![Some(gr)])
        }
        Ops::Reciprocal => {
            let gg = ctx.mul(g, ret.clone());
            let gg2 = ctx.mul(gg, ret);
            let ng = ctx.neg(gg2);
            Some(vec![Some(ng)])
        }
        Ops::Sin => {
            let pih = ctx.cfloat(FRAC_PI_2);
            let shifted = ctx.sub(pih, (srcs[0]).clone());
            let cos = ctx.alu1(Ops::Sin, shifted);
            let out = ctx.mul(cos, g);
            Some(vec![Some(out)])
        }
        Ops::Log2 => {
            let denom = ctx.mul((srcs[0]).clone(), ln2);
            let out = fdiv(ctx, g, denom);
            Some(vec![Some(out)])
        }
        Ops::Exp2 => {
            let r0 = ctx.mul(ret, g);
            let r = ctx.mul(r0, ln2);
            Some(vec![Some(r)])
        }
        Ops::Sqrt => {
            let two = ctx.cfloat(2.0);
            let denom = ctx.mul(ret, two);
            let out = fdiv(ctx, g, denom);
            Some(vec![Some(out)])
        }
        Ops::Trunc => Some(vec![Some(zero_g)]),
        Ops::Cmplt | Ops::Cmpne => Some(vec![None; n]),
        Ops::Add => Some(vec![Some(g); n]),
        Ops::Sub => {
            let ng = ctx.neg(g.clone());
            Some(vec![Some(g), Some(ng)])
        }
        Ops::Mul => {
            let g0 = ctx.mul((srcs[1]).clone(), g.clone());
            let g1 = ctx.mul((srcs[0]).clone(), g);
            Some(vec![Some(g0), Some(g1)])
        }
        Ops::Max => {
            let (x, y) = ((srcs[0]).clone(), (srcs[1]).clone());
            let cf05 = ctx.cfloat(0.5);
            let half = ctx.mul(g.clone(), cf05);
            let eqxy = ctx.eq(x.clone(), y.clone());
            let tie = ctx.where_(eqxy, half, zero_g);
            let gt = ctx.gt(x.clone(), y.clone());
            let gx = ctx.where_(gt, g.clone(), tie.clone());
            let lt = ctx.lt(x, y);
            let gy = ctx.where_(lt, g, tie);
            Some(vec![Some(gx), Some(gy)])
        }
        Ops::Where => {
            let cond = srcs[0].clone();
            let gx = ctx.where_(cond.clone(), g.clone(), zero_g.clone());
            let gy = ctx.where_(cond, zero_g, g);
            Some(vec![None, Some(gx), Some(gy)])
        }
        Ops::Pow => {
            let (b, e) = ((srcs[0]).clone(), (srcs[1]).clone());
            let c1 = ctx.const_like(e.clone(), ConstVal::Int(1));
            let e1 = ctx.sub(e.clone(), c1);
            let bp = ctx.pow(b.clone(), e1);
            let e_pow = ctx.mul(e.clone(), bp);
            let e0 = ctx.const_like(e.clone(), ConstVal::Int(0));
            let e_eq0 = ctx.eq(e.clone(), e0);
            let inner_b = ctx.where_(e_eq0, e.clone(), e_pow);
            let gb = ctx.mul(g.clone(), inner_b);
            let logb = ctx.alu1(Ops::Log2, b.clone());
            let ret_log = ctx.mul(ret, logb);
            let logb_ln = ctx.mul(ret_log, ln2);
            let e_lt0c = ctx.const_like(e.clone(), ConstVal::Int(0));
            let e_lt0 = ctx.lt(e, e_lt0c);
            let neginf = ctx.const_like(g.clone(), ConstVal::Float(f64::NEG_INFINITY));
            let zerog = ctx.const_like(g.clone(), ConstVal::Float(0.0));
            let inner_neg = ctx.where_(e_lt0, neginf, zerog);
            let b0 = ctx.const_like(b.clone(), ConstVal::Int(0));
            let b_eq0 = ctx.eq(b, b0);
            let inner_e = ctx.where_(b_eq0, inner_neg, logb_ln);
            let ge = ctx.mul(g, inner_e);
            Some(vec![Some(gb), Some(ge)])
        }
        Ops::Reduce => {
            let (rop_op, naxes) = reduce_arg(ctx, ret.clone());
            Some(reduce_gradient(ctx, g, ret, rop_op, naxes))
        }
        Ops::Contiguous => Some(vec![Some(g)]),
        Ops::ContiguousBackward => {
            let c = ctx.uop(Ops::Contiguous, None, vec![g], Arg::None);
            Some(vec![Some(c)])
        }
        Ops::Reshape => {
            let s0 = ctx.shape_of(&srcs[0]).unwrap();
            let gr = movement::reshape(ctx, g, &s0);
            Some(vec![Some(gr), None])
        }
        Ops::Expand => Some(vec![Some(g), None]),
        Ops::Permute => {
            let order = seq_ints(ctx.arg_of(&ret));
            let inv = argsort(&order);
            let inv_i64: Vec<i64> = inv.iter().map(|&v| v as i64).collect();
            let gr = movement::permute(ctx, g, &inv_i64);
            Some(vec![Some(gr)])
        }
        Ops::Flip => {
            let axes = seq_trues(ctx.arg_of(&ret));
            let gr = movement::flip(ctx, g, &axes);
            Some(vec![Some(gr)])
        }
        // `mixin/gradient.py`: PAD's grad is a SHRINK of the incoming grad to
        // the input region; SHRINK's grad is a zero PAD back to the input shape.
        Ops::Pad => {
            let in_shape = ctx.shape_of(&srcs[0]).unwrap();
            let pairs = ctx.marg_pairs_svals(ret.clone()).unwrap_or_default();
            let shr: Vec<(SVal, SVal)> = in_shape
                .iter()
                .zip(pairs.iter())
                .map(|(s, (lo, _hi))| (lo.clone(), ctx.sval_add(lo.clone(), s.clone())))
                .collect();
            let gr = movement::shrink(ctx, g, &shr);
            Some(vec![Some(gr), None, None])
        }
        Ops::Shrink => {
            let in_shape = ctx.shape_of(&srcs[0]).unwrap();
            let pairs = ctx.marg_pairs_svals(ret.clone()).unwrap_or_default();
            let pads: Vec<(SVal, SVal)> = in_shape
                .iter()
                .zip(pairs.iter())
                .map(|(s, (a, b))| {
                    let sa = ctx.sval_sub(s.clone(), a.clone());
                    let tail = ctx.sval_sub(sa, b.clone());
                    (a.clone(), tail)
                })
                .collect();
            let gr = movement::pad(ctx, g, &pads);
            Some(vec![Some(gr), None, None])
        }
        Ops::Bitcast => Some(vec![None]),
        // `gradient.py`: STACK's grad is `ctx[i]` (slice along the new dim 0).
        Ops::Stack => {
            let shp = ctx.shape_of(&g).unwrap();
            let gs: Vec<Option<UopId>> = (0..n)
                .map(|i| {
                    let shr: Vec<(SVal, SVal)> =
                        std::iter::once((SVal::I(i as i128), SVal::I(i as i128 + 1)))
                            .chain(shp.iter().skip(1).map(|s| (SVal::I(0), s.clone())))
                            .collect();
                    let sliced = movement::shrink(ctx, g.clone(), &shr);
                    Some(movement::reshape(ctx, sliced, &shp[1..]))
                })
                .collect();
            Some(gs)
        }
        _ => None,
    }
}

/// `gradient.py:89 _deepwalk` — forward (srcs-before-parent) order restricted to
/// nodes that can reach a target, skipping `DETACH` and its downstream.
fn deepwalk(
    ctx: &UOpCtx,
    root: UopId,
    targets: &HashSet<UopId>,
) -> (Vec<UopId>, HashMap<UopId, bool>) {
    enum Step {
        Enter(UopId),
        Emit(UopId),
    }
    let mut visited: HashSet<UopId> = HashSet::new();
    let mut order: Vec<UopId> = Vec::new();
    let mut stack = vec![Step::Enter(root)];
    while let Some(s) = stack.pop() {
        match s {
            Step::Enter(id) => {
                if !visited.insert(id.clone()) {
                    continue;
                }
                stack.push(Step::Emit(id.clone()));
                for src in ctx.srcs(&id).iter().rev() {
                    stack.push(Step::Enter(src.clone()));
                }
            }
            Step::Emit(id) => order.push(id),
        }
    }
    let mut in_path: HashMap<UopId, bool> = HashMap::new();
    for u in &order {
        let v = ctx
            .srcs(u)
            .iter()
            .any(|s| *in_path.get(s).unwrap_or(&false) || targets.contains(s));
        in_path.insert(u.clone(), v);
    }
    let walk = order
        .into_iter()
        .filter(|u| ctx.op_of(u) != Ops::Detach && *in_path.get(u).unwrap_or(&false))
        .collect();
    (walk, in_path)
}

/// `gradient.py:96 compute_gradient`: reverse-walk the forward graph, emitting a
/// gradient for every source, and return `target uop -> gradient uop`.
pub fn compute_gradient(
    ctx: &mut UOpCtx,
    root: UopId,
    root_grad: UopId,
    targets: &HashSet<UopId>,
) -> HashMap<UopId, UopId> {
    let (walk, in_path) = deepwalk(ctx, root.clone(), targets);
    let mut grads: HashMap<UopId, UopId> = HashMap::new();
    grads.insert(root, root_grad);
    for i in (0..walk.len()).rev() {
        let t0 = &walk[i].clone();
        let g0 = match grads.get(t0) {
            Some(g) if ctx.op_of(g) != Ops::Noop => g,
            _ => continue,
        };
        let lgrads = match grad_of_op(ctx, t0.clone(), g0.clone()) {
            Some(g) => g,
            None => panic!(
                "TGERR: failed to compute gradient for {:?}\n  (op not supported by grad_of_op)",
                ctx.op_of(t0)
            ),
        };
        let srcs = ctx.srcs(t0).to_vec();
        assert_eq!(
            lgrads.len(),
            srcs.len(),
            "TGERR: got {} gradient, expected {}",
            lgrads.len(),
            srcs.len()
        );
        for (k, v) in srcs.iter().zip(lgrads) {
            let mut v = match v {
                Some(v) => v,
                None => continue,
            };
            // a shaped edge's gradient is summed to its source's shape.
            let ks = ctx.shape(k).ok().flatten();
            let vs = ctx.shape(&v).ok().flatten();
            if let (Some(ks), Some(vs)) = (ks, vs)
                && ks.as_ref() != vs.as_ref()
            {
                let dt = ctx.dt_of(&v);
                let axes = movement::broadcast_axes(ctx, &ks, &vs);
                let vc = ctx.cast(v, reduce::sum_acc_dtype(dt));
                let r = reduce::rop(ctx, Ops::Add, vc, &axes);
                let rs = movement::reshape(ctx, r, &ks);
                v = ctx.cast(rs, dt);
            }
            match grads.get(k).cloned() {
                Some(prev) if ctx.op_of(&prev) != Ops::Noop => {
                    grads.insert(k.clone(), ctx.add(prev, v));
                }
                _ => {
                    grads.insert(k.clone(), v);
                }
            }
        }
    }
    let _ = &in_path;
    grads
}

impl Tensor {
    /// `Tensor.detach`: a new tensor with the same data but cut off from the
    /// autograd graph (`elementwise.py:39`, `alu(Ops.DETACH)`).
    pub fn detach(&self) -> Tensor {
        self.detach_ctx(&mut self.ctx.borrow_mut())
    }
    #[doc(hidden)]
    pub fn detach_ctx(&self, ctx: &mut UOpCtx) -> Tensor {
        let dt = ctx.dt_of(&self.uop());
        {
            let __u = ctx.uop(Ops::Detach, Some(dt), vec![self.uop().clone()], Arg::None);
            Tensor::from_uop(ctx, __u)
        }
    }

    /// `Tensor.gradient(*targets, gradient)` (`op.py:453`): gradients of `self`
    /// wrt each of `targets` (missing grads become zeros).
    pub fn gradient(&self, targets: &[Tensor], gradient: Option<Tensor>) -> Vec<Tensor> {
        self.gradient_ctx(&mut self.ctx.borrow_mut(), targets, gradient)
    }
    #[doc(hidden)]
    pub fn gradient_ctx(
        &self,
        ctx: &mut UOpCtx,
        targets: &[Tensor],
        gradient: Option<Tensor>,
    ) -> Vec<Tensor> {
        assert!(
            self.shape_ctx(ctx).numel_int().unwrap_or(0) == 1 || gradient.is_some(),
            "TGERR: gradient() on a non-scalar needs an explicit gradient"
        );
        let root_grad = match gradient {
            Some(gt) => gt.uop(),
            None => ctx.const_like(self.uop().clone(), ConstVal::Float(1.0)),
        };
        let tset: HashSet<UopId> = targets.iter().map(|t| t.uop().clone()).collect();
        let grads = compute_gradient(ctx, self.uop().clone(), root_grad, &tset);
        targets
            .iter()
            .map(|t| match grads.get(&t.uop()) {
                Some(g) => {
                    let __u = g;
                    Tensor::from_uop(ctx, __u.clone())
                }
                None => {
                    let __u = ctx.const_like(t.uop().clone(), ConstVal::Float(0.0));
                    Tensor::from_uop(ctx, __u)
                }
            })
            .collect()
    }

    /// `Tensor.backward(gradient)` (`tensor.py:657`): run `compute_gradient` and
    /// attach each target's gradient to its graph node (accumulating).
    pub fn backward(&self, targets: &[Tensor], gradient: Option<Tensor>) {
        self.backward_ctx(&mut self.ctx.borrow_mut(), targets, gradient)
    }
    #[doc(hidden)]
    pub fn backward_ctx(&self, ctx: &mut UOpCtx, targets: &[Tensor], gradient: Option<Tensor>) {
        let gs = self.gradient_ctx(ctx, targets, gradient);
        for (t, g) in targets.iter().zip(gs) {
            match ctx.grad(&t.uop()) {
                Some(prev) => {
                    let sum = ctx.add(prev, g.uop());
                    ctx.set_grad(&t.uop(), sum);
                }
                None => ctx.set_grad(&t.uop(), g.uop()),
            }
        }
    }
}
