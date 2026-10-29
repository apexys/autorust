//! Session-driven `Jit` (engine/jit.rs): capture one step, replay it with fresh
//! buffers. Verified here on INTERP with an in-place SGD step; the device-RNG
//! replay path is covered by the CL run of `tests/rand.rs` plus the MNIST example.

use autorust::engine::jit::Jit;
use autorust::engine::session::Session;
use autorust::tensor::{Ctx, Tensor};
use autorust::uop::ctx::UOpCtx;

/// On replay the JIT executes the stored compiled LINEARS: it must not re-lower
/// or re-look-up any program (`jit_lower` runs once, jit.py:263).
#[test]
fn jit_replay_does_not_recompile() {
    let ctx = Ctx::new();
    let mut sess = Session::new();
    sess.enable_jit();
    let w = Tensor::from_slice(&ctx, &[1.0f32, 2.0], None, None).to_param();
    let x = Tensor::from_slice(&ctx, &[3.0f32, 4.0], None, None).to_param();
    let mut jit = Jit::new();
    let _ = jit
        .call(&ctx, &mut sess, |ctx, sess| step(ctx, sess, &w, &x))
        .to_list::<f32>(&mut sess);
    let (h0, m0) = (sess.cache.hits, sess.cache.misses);
    let _ = jit
        .call(&ctx, &mut sess, |ctx, sess| step(ctx, sess, &w, &x))
        .to_list::<f32>(&mut sess);
    assert_eq!(
        (sess.cache.hits, sess.cache.misses),
        (h0, m0),
        "replay must not recompile"
    );
}

/// One manual SGD step: `loss = sum(w*x)`, `w -= 0.1 * dloss/dw`, co-realized.
fn step(ctx: &Ctx, sess: &mut Session, w: &Tensor, x: &Tensor) -> Tensor {
    let loss = w.mul(x).sum(None, false, None);
    let grads = loss.gradient(std::slice::from_ref(w), None);
    let step_t = grads[0].mul(0.1);
    let update = w.sub(step_t);
    let eff = w.assign_effect(&update);
    Tensor::realize_many(&mut ctx.raw(), sess, &[loss.clone(), eff]);
    w.pin_buffer(sess); // keep the param's storage across prunes (optimizers do this)
    loss
}

#[test]
fn jit_capture_then_replay_updates_params() {
    let ctx = Ctx::new();
    let mut sess = Session::new();
    sess.enable_jit();
    assert!(sess.jit_enabled());
    let w = Tensor::from_slice(&ctx, &[1.0f32, 2.0], None, None).to_param();
    let x = Tensor::from_slice(&ctx, &[3.0f32, 4.0], None, None).to_param();

    let mut jit = Jit::new();
    let mut losses = Vec::new();
    for _ in 0..3 {
        let loss = jit.call(&ctx, &mut sess, |ctx, sess| step(ctx, sess, &w, &x));
        losses.push(loss.to_list::<f32>(&mut sess)[0]);
    }
    assert!(jit.is_captured() && jit.n_linears() >= 1);
    // loss_0 = 3 + 8 = 11; w <- [0.7, 1.6]; loss_1 = 2.1 + 6.4 = 8.5; …
    for (got, want) in losses.iter().zip([11.0f32, 8.5, 6.0]) {
        assert!(
            (got - want).abs() < 1e-5,
            "loss {got} != {want} (all {losses:?})"
        );
    }
    let wv = w.to_list::<f32>(&mut sess);
    assert!(
        (wv[0] - 0.1).abs() < 1e-5 && (wv[1] - 0.8).abs() < 1e-5,
        "w = {wv:?}"
    );
}

#[test]
fn jit_matches_non_jit() {
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let w0 = Tensor::from_slice(&ctx, &[0.5f32, -0.25, 3.0], None, None).to_param();
    let x0 = Tensor::from_slice(&ctx, &[2.0f32, 5.0, -1.0], None, None).to_param();

    // non-JIT reference
    let mut plain = Vec::new();
    for _ in 0..4 {
        let l = step(&ctx, &mut sess, &w0, &x0);
        plain.push(l.to_list::<f32>(&mut sess)[0]);
    }

    // JIT run (fresh state)
    let mut sess2 = Session::new();
    let w1 = Tensor::from_slice(&ctx, &[0.5f32, -0.25, 3.0], None, None).to_param();
    let x1 = Tensor::from_slice(&ctx, &[2.0f32, 5.0, -1.0], None, None).to_param();
    let mut jit = Jit::new();
    let mut jitted = Vec::new();
    for _ in 0..4 {
        let l = jit.call(&ctx, &mut sess2, |ctx, sess| step(ctx, sess, &w1, &x1));
        jitted.push(l.to_list::<f32>(&mut sess2)[0]);
    }
    for (a, b) in plain.iter().zip(&jitted) {
        assert!((a - b).abs() < 1e-5, "jit {jitted:?} != plain {plain:?}");
    }
}

/// On hardware the RNG is a device graph (§8.23), so replaying a JIT step must
/// advance the counter and yield the *same* sequence as a non-JIT run. (INTERP
/// uses the host RNG, whose `from_slice` is baked into the captured graph, so
/// this only holds on hardware; skip otherwise.)
#[test]
fn jit_rng_advances_on_replay() {
    if autorust::tensor::creation::default_device() != "CL" {
        return;
    }
    let ctx = Ctx::new();
    Tensor::manual_seed(42);
    let mut sess = Session::new();
    let plain: Vec<Vec<i32>> = (0..3)
        .map(|i| {
            eprintln!("plain {i}");
            Tensor::randint(&ctx, &[4], 0, 100, None, None).to_list::<i32>(&mut sess)
        })
        .collect();

    Tensor::manual_seed(42);
    let mut sess2 = Session::new();
    sess2.enable_jit();
    let mut jit = Jit::new();
    let mut jitted = Vec::new();
    for i in 0..3 {
        eprintln!("jit {i}");
        let r = jit.call(&ctx, &mut sess2, |ctx, sess| {
            Tensor::randint(ctx, &[4], 0, 100, None, None)
                .realize(sess)
                .clone()
        });
        jitted.push(r.to_list::<i32>(&mut sess2));
    }
    assert_eq!(
        plain, jitted,
        "JIT replay must advance the device RNG like non-JIT"
    );
}

/// Enabling JIT *after* a pre-JIT draw (model init) must continue the host
/// counter's sequence, so a JIT run samples where a non-JIT run would.
#[test]
fn jit_rng_continues_from_host_counter() {
    if autorust::tensor::creation::default_device() != "CL" {
        return;
    }
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let draw = |ctx: &mut UOpCtx, sess: &mut Session| {
        Tensor::randint(ctx, &[6], 0, 50, None, None).to_list::<i32>(sess)
    };

    // A: one "init" draw, then three non-JIT draws.
    Tensor::manual_seed(7);
    let init_a = draw(&mut ctx.raw(), &mut sess);
    let a: Vec<Vec<i32>> = (0..3).map(|_| draw(&mut ctx.raw(), &mut sess)).collect();

    // B: the same init draw, then JIT for the rest.
    Tensor::manual_seed(7);
    let init_b = draw(&mut ctx.raw(), &mut sess);
    assert_eq!(init_a, init_b);
    sess.enable_jit();
    let mut jit = Jit::new();
    let b: Vec<Vec<i32>> = (0..3)
        .map(|_| {
            jit.call(&ctx, &mut sess, |ctx, sess| {
                Tensor::randint(ctx, &[6], 0, 50, None, None)
                    .realize(sess)
                    .clone()
            })
            .to_list::<i32>(&mut sess)
        })
        .collect();
    sess.jit(false);
    assert_eq!(a, b, "JIT must continue the host counter left by init");
}
