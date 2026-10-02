//! `nn/optim.py` — optimizers. They own their parameter list (tinygrad
//! `optim.SGD([w, b], lr)`) and read each parameter's gradient off its graph node
//! (written by [`Tensor::backward`]).
//!
//! Port shape (faithful to `tinygrad/nn/optim.py`):
//! - [`Optim::step`] is `Optimizer.step` → `Tensor.realize(*schedule_step())`.
//! - [`Optim::schedule_step`] builds the updates via `_step`, co-realizes the
//!   caller's roots (e.g. the loss) **together with** the updates and optimizer
//!   state in one schedule, then `assign`s each update back into its parameter.
//! - [`Optim::_step`] mirrors `Optimizer._step`: it may `assign` internal state
//!   buffers (momentum / Adam moments) and returns `(updates, extra_state)`.
//!
//! Stateless [`SGD`] is tinygrad's `SGD = LARS(tcoef=0.0)` (momentum 0, no weight
//! decay), so it shares the LARS `_step`; `step_graph` remains for the pure-graph
//! parity tests that rebuild the graph each epoch.

use crate::dtype::ConstVal;
use crate::engine::session::Session;
use crate::tensor::Tensor;
use crate::uop::ctx::UOpCtx;

/// Common surface for an optimizer step.
pub trait Optim {
    /// The parameters this optimizer updates, in order.
    fn params(&self) -> &[Tensor];

    /// Non-parameter buffers the optimizer owns (tinygrad `Optimizer.buffers`).
    fn buffers(&self) -> &[Tensor] {
        &[]
    }

    /// `Optimizer._step(params, grads)` (optim.py:59): build the per-parameter
    /// update. May `assign` (and therefore realize) internal state buffers with
    /// `sess`; the returned updates then read the updated state. Returns
    /// `(updates, extra_state)`.
    fn _step(&self, ctx: &mut UOpCtx, sess: &mut Session, params: &[Tensor], grads: &[Tensor]) -> (Vec<Tensor>, Vec<Tensor>);

    /// `Optimizer.schedule_step` (optim.py:41): build the updates, co-realize
    /// `roots + updates + extra` in one schedule, then `assign` each update back
    /// into its parameter. Returns `extra + params + buffers` (tinygrad's realized
    /// set). Pass the loss as a root to fuse it into the same schedule.
    fn schedule_step(&self, ctx: &mut UOpCtx, sess: &mut Session, roots: &[Tensor]) -> Vec<Tensor> {
        let params = self.params();
        let grads: Vec<Tensor> = params
            .iter()
            .map(|p| match ctx.grad(&p.uop()) {
                Some(g) => Tensor::from_uop(ctx, g),
                // Missing grad → zero, so `_apply_update` leaves the param as-is
                // (tinygrad would `unwrap` and raise; the port stays forgiving).
                None => {
                    let z = ctx.const_like(p.uop().clone(), ConstVal::Float(0.0));
                    Tensor::from_uop(ctx, z)
                }
            })
            .collect();
        // Realize every gradient subgraph once, together (they share most of the
        // backward graph). The faithful `realize_many` freezes each root into its
        // realized buffer leaf (`_apply_map_to_tensors`), so `_step` reads those
        // buffers directly — no host round-trip, and JIT-capturable.
        Tensor::realize_many(ctx, sess, roots.iter().chain(grads.iter()));
        let (updates, extra) = self._step(ctx, sess, params, &grads);
        // `tt.assign(self._apply_update(tt, updates[i]))`: build the applied
        // value (`t.detach() - up`) and its STORE+AFTER write effect, co-realize
        // those effects **with** the caller's roots and state so the parameter
        // write fuses into the same schedule as the loss, then pin each
        // parameter's buffer (tinygrad's Tensor owns its storage).
        let applied: Vec<Tensor> = params.iter().zip(updates.iter()).map(|(p, u)| self._apply_update(ctx, p, u)).collect();
        let effects: Vec<Tensor> = params.iter().zip(applied.iter()).map(|(p, a)| p.assign_effect(ctx, a)).collect();
        let mut to_realize: Vec<&Tensor> = roots.iter().collect();
        to_realize.extend(effects.iter());
        to_realize.extend(extra.iter());
        Tensor::realize_many(ctx, sess, to_realize);
        for p in params.iter() {
            p.pin_buffer(ctx, sess);
        }
        let mut ret = extra;
        ret.extend(params.iter().cloned());
        ret.extend(self.buffers().iter().cloned());
        ret
    }

    /// `Optimizer.step` (optim.py:35): `Tensor.realize(*schedule_step())`.
    fn step(&self, ctx: &mut UOpCtx, sess: &mut Session) {
        let _ = self.schedule_step(ctx, sess, &[]);
    }

    /// `Optimizer._apply_update` (optim.py:60): `t.detach() - up`.
    fn _apply_update(&self, ctx: &mut UOpCtx, t: &Tensor, up: &Tensor) -> Tensor {
        t.detach(ctx).sub(ctx, up)
    }

    /// Pure-graph single step (stateless optimizers only) — kept for the numeric
    /// parity tests that rebuild the graph each epoch.
    fn step_graph(&self, _ctx: &mut UOpCtx) -> Vec<Tensor> {
        panic!("TGERR: step_graph is only implemented for stateless SGD");
    }

    /// `Optimizer.zero_grad`: clear each parameter's attached gradient.
    fn zero_grad(&self, ctx: &UOpCtx) {
        for p in self.params() {
            ctx.clear_grad(&p.uop());
        }
    }
}

/// Allocate one optimizer-state leaf per parameter (`Optimizer._new_optim_param`,
/// optim.py:24): a zero **buffer leaf** with `initial_data` and no in-graph STORE,
/// so `assign` updates it durably (`Tensor::zeros_param`).
fn new_optim_param(ctx: &mut UOpCtx, params: &[Tensor]) -> Vec<Tensor> {
    params
        .iter()
        .map(|p| {
            let shape = p.shape(ctx);
            let dt = p.dtype(ctx);
            Tensor::zeros_param(ctx, shape, Some(dt), None)
        })
        .collect()
}

/// Scalar const (0-d) of a requested dtype — tinygrad's `Tensor.ones((1,))`
/// bias-correction accumulators; here a broadcastable 0-d is enough.
fn scalar(ctx: &mut UOpCtx, v: f64) -> Tensor {
    Tensor::full_const(ctx, &[] as &[i64], ConstVal::Float(v), None)
}

// ============================ LARS ============================

/// `nn.optim.LARS` (optim.py:95): Layer-wise Adaptive Rate Scaling with optional
/// momentum and weight decay. `SGD` is `LARS(tcoef=0.0, momentum=0, wd=0)`; `Muon`
/// (newton-schulz) is intentionally not ported yet.
pub struct LARS {
    params: Vec<Tensor>,
    pub lr: f32,
    pub momentum: f32,
    pub wd: f32,
    pub nesterov: bool,
    pub classic: bool,
    pub pre_wd: bool,
    pub tcoef: f32,
    /// Momentum state per parameter (`self.b`), empty when `momentum == 0`.
    b: Vec<Tensor>,
}

impl LARS {
    /// tinygrad `LARS(params, lr=0.001, momentum=0.9, weight_decay=1e-4,
    /// nesterov=False, classic=True, pre_wd=True, tcoef=0.001)`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(ctx: &mut UOpCtx, params: Vec<Tensor>, lr: f32, momentum: f32, weight_decay: f32, nesterov: bool, classic: bool, pre_wd: bool, tcoef: f32) -> Self {
        assert!(momentum >= 0.0, "TGERR: invalid momentum value: {momentum}");
        let b = if momentum != 0.0 { new_optim_param(ctx, &params) } else { Vec::new() };
        LARS { params, lr, momentum, wd: weight_decay, nesterov, classic, pre_wd, tcoef, b }
    }
}

impl Optim for LARS {
    fn params(&self) -> &[Tensor] {
        &self.params
    }

    fn _step(&self, ctx: &mut UOpCtx, sess: &mut Session, params: &[Tensor], grads: &[Tensor]) -> (Vec<Tensor>, Vec<Tensor>) {
        let mut ret = Vec::with_capacity(params.len());
        for (i, (t, g)) in params.iter().zip(grads.iter()).enumerate() {
            let mut g = g.clone();
            // trust ratio r (optim.py:112-116)
            let r = if self.tcoef != 0.0 {
                let r1 = t.detach(ctx).square(ctx).sum(ctx, None, false, None).sqrt(ctx);
                let r2 = g.square(ctx).sum(ctx, None, false, None).sqrt(ctx);
                let r1_tc = r1.mul(ctx, self.tcoef as f64);
                let r1_wd = r1.mul(ctx, self.wd as f64);
                let denom = r2.add(ctx, &r1_wd);
                let ratio = r1_tc.div(ctx, &denom);
                let inner = r2.gt(ctx, 0.0).where_(ctx, ratio, 1.0);
                r1.gt(ctx, 0.0).where_(ctx, inner, 1.0)
            } else {
                scalar(ctx, 1.0)
            };
            if self.pre_wd && self.wd > 0.0 {
                let wd_t = t.detach(ctx).mul(ctx, self.wd as f64);
                g = g.add(ctx, &wd_t);
            }
            // classic momentum does post learning-rate update
            if self.classic {
                let gr = g.mul(ctx, &r);
                g = gr.mul(ctx, self.lr as f64);
            }
            if self.momentum != 0.0 {
                let b = &self.b[i];
                let bm = b.mul(ctx, self.momentum as f64);
                let nb = bm.add(ctx, &g);
                b.assign(ctx, sess, &nb); // NOTE: b is zero on the first run
                g = if self.nesterov {
                    let bm2 = b.mul(ctx, self.momentum as f64);
                    g.add(ctx, &bm2)
                } else {
                    b.clone()
                };
            }
            // popular momentum does pre learning-rate update
            if !self.classic {
                let gr = g.mul(ctx, &r);
                g = gr.mul(ctx, self.lr as f64);
            }
            if !self.pre_wd && self.wd > 0.0 {
                let wd_t = t.detach(ctx).mul(ctx, self.lr as f64 * self.wd as f64);
                g = g.add(ctx, &wd_t);
            }
            let dt = t.dtype(ctx);
            ret.push(g.cast(ctx, dt));
        }
        (ret, self.b.clone())
    }
}

// ============================ LAMB ============================

/// `nn.optim.LAMB` (optim.py:147): Adam/AdamW with the LARS trust ratio. `Adam`
/// and `AdamW` are `LAMB(..., adam=true)`; plain `Adam` sets `weight_decay=0`.
pub struct LAMB {
    params: Vec<Tensor>,
    pub lr: f32,
    pub b1: f32,
    pub b2: f32,
    pub eps: f32,
    pub wd: f32,
    pub adam: bool,
    b1_t: Tensor,
    b2_t: Tensor,
    m: Vec<Tensor>,
    v: Vec<Tensor>,
}

impl LAMB {
    /// tinygrad `LAMB(params, lr=0.001, b1=0.9, b2=0.999, eps=1e-6,
    /// weight_decay=0.0, adam=False)`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(ctx: &mut UOpCtx, params: Vec<Tensor>, lr: f32, b1: f32, b2: f32, eps: f32, weight_decay: f32, adam: bool) -> Self {
        assert!(weight_decay >= 0.0, "TGERR: invalid weight_decay value: {weight_decay}");
        // b1_t/b2_t: `Tensor.ones((1,), dtype=dtypes.float32).is_param_(False)` —
        // **explicitly f32** (optim.py:157), not the default float. Keeping the
        // bias-correction accumulators in f32 is load-bearing: in bf16 `0.999`
        // rounds to `1.0`, so `1 - b2_t` would be `0` and the update divides by
        // zero (NaN). buffer leaves updated in place each step, hence `to_param`.
        let b1_t = crate::tensor::creation::ones(ctx, &[crate::uop::sint::SVal::I(1)], Some(crate::dtype::FLOAT32), None).to_param(ctx);
        let b2_t = crate::tensor::creation::ones(ctx, &[crate::uop::sint::SVal::I(1)], Some(crate::dtype::FLOAT32), None).to_param(ctx);
        let m = new_optim_param(ctx, &params);
        let v = new_optim_param(ctx, &params);
        LAMB { params, lr, b1, b2, eps, wd: weight_decay, adam, b1_t, b2_t, m, v }
    }
}

impl Optim for LAMB {
    fn params(&self) -> &[Tensor] {
        &self.params
    }

    fn _step(&self, ctx: &mut UOpCtx, sess: &mut Session, params: &[Tensor], grads: &[Tensor]) -> (Vec<Tensor>, Vec<Tensor>) {
        // `self.b1_t *= self.b1` / `self.b2_t *= self.b2` (optim.py:163-164)
        let b1t_new = self.b1_t.mul(ctx, self.b1 as f64);
        self.b1_t.assign(ctx, sess, &b1t_new);
        let b2t_new = self.b2_t.mul(ctx, self.b2 as f64);
        self.b2_t.assign(ctx, sess, &b2t_new);
        let one = scalar(ctx, 1.0);
        let mut ret = Vec::with_capacity(params.len());
        for (i, (t, g)) in params.iter().zip(grads.iter()).enumerate() {
            let dt = t.dtype(ctx);
            let m_old = self.m[i].mul(ctx, self.b1 as f64);
            let g_b1 = g.mul(ctx, 1.0 - self.b1 as f64);
            let m_new = m_old.add(ctx, &g_b1).cast(ctx, dt);
            self.m[i].assign(ctx, sess, &m_new);
            let v_old = self.v[i].mul(ctx, self.b2 as f64);
            let g_sq = g.square(ctx);
            let g_b2 = g_sq.mul(ctx, 1.0 - self.b2 as f64);
            let v_new = v_old.add(ctx, &g_b2).cast(ctx, dt);
            self.v[i].assign(ctx, sess, &v_new);
            let one_m_b1 = one.sub(ctx, &self.b1_t);
            let m_hat = self.m[i].div(ctx, &one_m_b1);
            let one_m_b2 = one.sub(ctx, &self.b2_t);
            let v_hat = self.v[i].div(ctx, &one_m_b2);
            let v_den = v_hat.sqrt(ctx).add(ctx, self.eps as f64);
            let mut up = m_hat.div(ctx, &v_den);
            if self.wd > 0.0 {
                let wd_t = t.detach(ctx).mul(ctx, self.wd as f64);
                up = up.add(ctx, &wd_t);
            }
            let r = if !self.adam {
                let r1 = t.detach(ctx).square(ctx).sum(ctx, None, false, None).sqrt(ctx);
                let r2 = up.square(ctx).sum(ctx, None, false, None).sqrt(ctx);
                let ratio = r1.div(ctx, &r2);
                let inner = r2.gt(ctx, 0.0).where_(ctx, ratio, 1.0);
                r1.gt(ctx, 0.0).where_(ctx, inner, 1.0)
            } else {
                one.clone()
            };
            let rl = r.mul(ctx, self.lr as f64);
            ret.push(rl.mul(ctx, &up).cast(ctx, dt));
        }
        let mut extra = vec![self.b1_t.clone(), self.b2_t.clone()];
        extra.extend(self.m.iter().cloned());
        extra.extend(self.v.iter().cloned());
        (ret, extra)
    }
}

// ============================ convenience ============================

/// tinygrad `nn.optim.SGD(params, lr, momentum=0.0, weight_decay=0.0,
/// nesterov=False, classic=False)` = `LARS(..., pre_wd=True, tcoef=0.0)`.
pub struct SGD {
    inner: LARS,
}

impl SGD {
    /// `SGD(params, lr)` — the stateless default (`momentum=0`, `wd=0`).
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        // momentum=0 → LARS needs no state, so no `ctx`/state allocation here,
        // preserving the `SGD::new(params, lr)` call shape.
        SGD { inner: LARS { params, lr, momentum: 0.0, wd: 0.0, nesterov: false, classic: false, pre_wd: true, tcoef: 0.0, b: Vec::new() } }
    }
}

impl Optim for SGD {
    fn params(&self) -> &[Tensor] {
        self.inner.params()
    }

    fn _step(&self, ctx: &mut UOpCtx, sess: &mut Session, params: &[Tensor], grads: &[Tensor]) -> (Vec<Tensor>, Vec<Tensor>) {
        self.inner._step(ctx, sess, params, grads)
    }

    /// Pure-graph `w - lr*grad` (the old stateless path, for parity tests).
    fn step_graph(&self, ctx: &mut UOpCtx) -> Vec<Tensor> {
        self.inner
            .params()
            .iter()
            .map(|p| match ctx.grad(&p.uop()) {
                Some(g) => {
                    let g = Tensor::from_uop(ctx, g);
                    let g_lr = g.mul(ctx, self.inner.lr as f64);
                    p.detach(ctx).sub(ctx, &g_lr)
                }
                None => p.clone(),
            })
            .collect()
    }
}

/// tinygrad `nn.optim.Adam(params, lr, b1, b2, eps)` = `LAMB(..., wd=0, adam=true)`.
pub struct Adam {
    inner: LAMB,
}

impl Adam {
    pub fn new(ctx: &mut UOpCtx, params: Vec<Tensor>, lr: f32) -> Self {
        Adam { inner: LAMB::new(ctx, params, lr, 0.9, 0.999, 1e-8, 0.0, true) }
    }
}

impl Optim for Adam {
    fn params(&self) -> &[Tensor] {
        self.inner.params()
    }
    fn _step(&self, ctx: &mut UOpCtx, sess: &mut Session, params: &[Tensor], grads: &[Tensor]) -> (Vec<Tensor>, Vec<Tensor>) {
        self.inner._step(ctx, sess, params, grads)
    }
}

/// tinygrad `nn.optim.AdamW(params, lr, ..., weight_decay=0.01)`.
pub struct AdamW {
    inner: LAMB,
}

impl AdamW {
    pub fn new(ctx: &mut UOpCtx, params: Vec<Tensor>, lr: f32) -> Self {
        AdamW { inner: LAMB::new(ctx, params, lr, 0.9, 0.999, 1e-8, 0.01, true) }
    }
}

impl Optim for AdamW {
    fn params(&self) -> &[Tensor] {
        self.inner.params()
    }
    fn _step(&self, ctx: &mut UOpCtx, sess: &mut Session, params: &[Tensor], grads: &[Tensor]) -> (Vec<Tensor>, Vec<Tensor>) {
        self.inner._step(ctx, sess, params, grads)
    }
}