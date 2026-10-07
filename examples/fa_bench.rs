//! SDPA vs FlashAttention attention benchmark (forward and forward+backward).
//!
//! ```sh
//! # CUDA (RTX 3060):
//! DEFAULT_DEVICE=CUDA cargo run --release --features backend_cuda --example fa_bench
//! # interpreter reference:
//! cargo run --release --example fa_bench
//! # ViT-B/16 attention shape, bigger batch:
//! DEFAULT_DEVICE=CUDA BS=8 HEADS=12 SEQ=197 DIM=64 \
//!   cargo run --release --features backend_cuda --example fa_bench
//! ```
//!
//! Knobs: `BS HEADS SEQ DIM` (attention shape), `FA_BQ FA_BK` (FA block sizes),
//! `WARM`/`ITERS`. `DEFAULT_DEVICE` selects the backend.

use std::time::Instant;

use autorust::engine::jit::Jit;
use autorust::engine::session::Session;
use autorust::models::flash_attention::flash_attention_blocks;
use autorust::tensor::Tensor;
use autorust::uop::ctx::UOpCtx;

fn env_or<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d)
}

/// Build + realize `iters` times (so the kernels actually run each iteration),
/// after `warm` warmups, and return the mean wall time in milliseconds.
fn bench<F: FnMut(&mut UOpCtx, &mut Session) -> Tensor>(
    mut build: F,
    ctx: &mut UOpCtx,
    sess: &mut Session,
    warm: usize,
    iters: usize,
) -> f64 {
    for _ in 0..warm {
        let o = build(ctx, sess);
        let _ = o.to_list::<f32>(ctx, sess);
    }
    let t = Instant::now();
    for _ in 0..iters {
        let o = build(ctx, sess);
        let _ = o.to_list::<f32>(ctx, sess);
    }
    t.elapsed().as_secs_f64() / iters as f64 * 1e3
}

/// Like [`bench`], but captures the schedule once with `Jit` and replays it, so
/// the measured time is kernel execution (no per-iteration graph scheduling).
fn bench_jit<F: Fn(&mut UOpCtx, &mut Session) -> Tensor>(
    build: F,
    ctx: &mut UOpCtx,
    sess: &mut Session,
    warm: usize,
    iters: usize,
) -> f64 {
    sess.enable_jit();
    let mut jit = Jit::new();
    for _ in 0..warm {
        let o = jit.call(ctx, sess, |c, s| build(c, s));
        let _ = o.to_list::<f32>(ctx, sess);
    }
    let t = Instant::now();
    for _ in 0..iters {
        let o = jit.call(ctx, sess, |c, s| build(c, s));
        let _ = o.to_list::<f32>(ctx, sess);
    }
    sess.jit(false);
    t.elapsed().as_secs_f64() / iters as f64 * 1e3
}

fn main() {
    let bs = env_or("BS", 1i64);
    let heads = env_or("HEADS", 12i64);
    let seq = env_or("SEQ", 197i64);
    let dim = env_or("DIM", 64i64);
    let bq = env_or("FA_BQ", 64i64);
    let bk = env_or("FA_BK", 64i64);
    let warm = env_or("WARM", 5usize);
    let iters = env_or("ITERS", 50usize);
    let dev = std::env::var("DEFAULT_DEVICE").ok();

    println!(
        "attention bench: BS={bs} HEADS={heads} SEQ={seq} DIM={dim} FA_blocks=({bq},{bk}) device={:?}",
        autorust::tensor::creation::default_device()
    );

    let n = (bs * heads * seq * dim) as usize;
    let mode = std::env::var("INPUT").unwrap_or_else(|_| "pattern".into());
    let mk = |seed: usize| -> Vec<f32> {
        match mode.as_str() {
            "ones" => vec![1.0f32; n],
            "ramp" => (0..n).map(|i| (i as f32) * 0.001 - 0.05).collect(),
            _ => (0..n).map(|i| (((i + seed) % 23) as f32) * 0.02 - 0.23).collect(),
        }
    };
    let (qd, kd, vd) = (mk(0), mk(7), mk(13));

    let mut ctx = UOpCtx::new();
    let mut sess = Session::new();
    let q = Tensor::from_slice(&mut ctx, &qd, None, dev.as_deref()).reshape(&mut ctx, [bs, heads, seq, dim]);
    let k = Tensor::from_slice(&mut ctx, &kd, None, dev.as_deref()).reshape(&mut ctx, [bs, heads, seq, dim]);
    let v = Tensor::from_slice(&mut ctx, &vd, None, dev.as_deref()).reshape(&mut ctx, [bs, heads, seq, dim]);

    // Correctness: FA must match SDPA.
    let os = q.scaled_dot_product_attention(&mut ctx, &k, &v, None, 0.0, false, false);
    let of = flash_attention_blocks(&q, &k, &v, None, false, bq, bk, &mut ctx);
    let ovs = os.to_list::<f32>(&mut ctx, &mut sess);
    let ovf = of.to_list::<f32>(&mut ctx, &mut sess);
    let maxd = ovs.iter().zip(&ovf).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let maxo = ovs.iter().map(|a| a.abs()).fold(0.0f32, f32::max);
    let maxr = ovs.iter().zip(&ovf).map(|(a, b)| (a - b).abs() / a.abs().max(1e-6)).fold(0.0f32, f32::max);
    let (mut mi, mut mdd) = (0usize, 0.0f32);
    for (i, (a, b)) in ovs.iter().zip(&ovf).enumerate() { let d = (a - b).abs(); if d > mdd { mdd = d; mi = i; } }
    let (du, su, hu) = (dim as usize, seq as usize, heads as usize);
    println!("max |SDPA| = {maxo:.3e}   max |SDPA-FA| = {maxd:.3e}   max rel = {maxr:.3e}   at b={} h={} row={} col={} (sdpa={} fa={})",
        mi/(du*su*hu), (mi/du)%su, (mi/(du*su))%hu, mi%du, ovs[mi], ovf[mi]);
    if std::env::var("CHECK_ONLY").is_ok() { return; }

    let sdpa = |c: &mut UOpCtx, _s: &mut Session| q.scaled_dot_product_attention(c, &k, &v, None, 0.0, false, false);
    let fa = |c: &mut UOpCtx, _s: &mut Session| flash_attention_blocks(&q, &k, &v, None, false, bq, bk, c);

    let t_sdpa_f = bench(sdpa, &mut ctx, &mut sess, warm, iters);
    let t_fa_f = bench(fa, &mut ctx, &mut sess, warm, iters);

    // Forward + backward: zero the grads each step so they don't accumulate.
    let targets = [q.clone(), k.clone(), v.clone()];
    let bwd_body = |c: &mut UOpCtx, s: &mut Session, o: Tensor| -> Tensor {
        let loss = o.mul(c, &o).sum(c, None, false, None);
        loss.backward(c, &targets, None);
        for t in [&q, &k, &v] {
            let g = t.grad(c).unwrap();
            g.realize(c, s);
            c.clear_grad(&t.uop());
        }
        loss
    };
    let t_sdpa_b = bench(
        |c: &mut UOpCtx, s: &mut Session| {
            let o = q.scaled_dot_product_attention(c, &k, &v, None, 0.0, false, false);
            bwd_body(c, s, o)
        },
        &mut ctx, &mut sess, warm, iters,
    );
    let t_fa_b = bench(
        |c: &mut UOpCtx, s: &mut Session| {
            let o = flash_attention_blocks(&q, &k, &v, None, false, bq, bk, c);
            bwd_body(c, s, o)
        },
        &mut ctx, &mut sess, warm, iters,
    );

    if std::env::var("JIT").is_ok() {
        let t_sdpa_j = bench_jit(|c, _s| q.scaled_dot_product_attention(c, &k, &v, None, 0.0, false, false), &mut ctx, &mut sess, warm, iters);
        let t_fa_j = bench_jit(|c, _s| flash_attention_blocks(&q, &k, &v, None, false, bq, bk, c), &mut ctx, &mut sess, warm, iters);
        let t_sdpa_bj = bench_jit(|c, s| { let o = q.scaled_dot_product_attention(c, &k, &v, None, 0.0, false, false); bwd_body(c, s, o) }, &mut ctx, &mut sess, warm, iters);
        let t_fa_bj = bench_jit(|c, s| { let o = flash_attention_blocks(&q, &k, &v, None, false, bq, bk, c); bwd_body(c, s, o) }, &mut ctx, &mut sess, warm, iters);
        println!("  SDPA fwd (jit) {t_sdpa_j:8.3} ms");
        println!("  FA   fwd (jit) {t_fa_j:8.3} ms   ({:.2}x)", t_sdpa_j / t_fa_j);
        println!("  SDPA f+b (jit) {t_sdpa_bj:8.3} ms");
        println!("  FA   f+b (jit) {t_fa_bj:8.3} ms   ({:.2}x)", t_sdpa_bj / t_fa_bj);
    }
    println!("  SDPA fwd       {t_sdpa_f:8.3} ms");
    println!("  FA   fwd       {t_fa_f:8.3} ms   ({:.2}x)", t_sdpa_f / t_fa_f);
    println!("  SDPA fwd+bwd   {t_sdpa_b:8.3} ms");
    println!("  FA   fwd+bwd   {t_fa_b:8.3} ms   ({:.2}x)", t_sdpa_b / t_fa_b);
}
