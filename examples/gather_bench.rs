//! DORSCH-shaped gather benchmark: general one-hot advanced indexing vs the
//! direct (`gather_direct`) and interpolating (`gather_interp`) fast paths.
//!
//! Run with `DEFAULT_DEVICE=RUST cargo run --release --example gather_bench`.
//! `BENCH_DEVICE` selects a backend by name (`RUST`, `CL`, …);
//! `BENCH_ONLY=onehot|direct|onehot_interp|direct_interp` runs one path so the
//! per-path kernel log (`AUTORUST_KPROF=1`) is easy to read.
use autorust::engine::jit::Jit;
use autorust::engine::session::Session;
use autorust::tensor::{Ctx, Ix, Tensor};
use std::time::Instant;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let (n, s, r) = (4540i64, 2000i64, 64i64);
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let dev = std::env::var("BENCH_DEVICE").ok();
    let d = dev.as_deref();
    let total = (n * s) as usize;
    let data: Vec<f32> = (0..total)
        .map(|i| ((i * 2654435761usize) % 10007) as f32 * 1e-3)
        .collect();
    let atlas = Tensor::from_slice(&ctx, &data, None, d).reshape([n, s]);
    let rows_v: Vec<i32> = (0..r * r).map(|i| (i * 7 % n) as i32).collect();
    let q_v: Vec<f32> = (0..r * r)
        .map(|i| ((i * 37) % ((s - 1) * 1000)) as f32 / 1000.0)
        .collect();
    let rows = Tensor::from_slice(&ctx, &rows_v, None, d).reshape([r, r]);
    let q = Tensor::from_slice(&ctx, &q_v, None, d).reshape([r, r]);
    let flat = atlas.reshape([n * s]);
    let i_int: Vec<i32> = q_v.iter().map(|x| x.floor() as i32).collect();
    let flat_idx_v: Vec<i32> = rows_v
        .iter()
        .zip(&i_int)
        .map(|(&row, &col)| row * s as i32 + col)
        .collect();
    let flat_idx = Tensor::from_slice(&ctx, &flat_idx_v, None, d);
    let flat_idx1 = Tensor::from_slice(
        &ctx,
        &flat_idx_v.iter().map(|x| x + 1).collect::<Vec<_>>(),
        None,
        d,
    );

    let mut onehot = Vec::new();
    let mut direct = Vec::new();
    let mut onehot_interp = Vec::new();
    let mut direct_interp = Vec::new();
    sess.enable_jit();
    let mut jit1 = Jit::new();
    let mut jit2 = Jit::new();
    let mut jit3 = Jit::new();
    let mut jit4 = Jit::new();
    let only = std::env::var("BENCH_ONLY").unwrap_or_default();
    for rep in 0..50 {
        if only.is_empty() || only == "onehot" {
            let t = Instant::now();
            let a = jit1.call(&ctx, &mut sess, |_c, _s| {
                flat.getitem(&[Ix::Tensor(flat_idx.clone())])
                    .reshape([r, r])
            });
            let _ = a;
            if rep >= 5 {
                onehot.push(t.elapsed().as_secs_f64());
            }
        }

        if only.is_empty() || only == "direct" {
            let t = Instant::now();
            let b = jit2.call(&ctx, &mut sess, |_c, _s| {
                flat.gather_direct(0, &flat_idx).reshape([r, r])
            });
            let _ = b;
            if rep >= 5 {
                direct.push(t.elapsed().as_secs_f64());
            }
        }

        if only.is_empty() || only == "onehot_interp" {
            let t = Instant::now();
            let outs1 = jit3.call_many(&ctx, &mut sess, |_c, _s| {
                let lo = flat
                    .getitem(&[Ix::Tensor(flat_idx.clone())])
                    .reshape([r, r]);
                let hi = flat
                    .getitem(&[Ix::Tensor(flat_idx1.clone())])
                    .reshape([r, r]);
                let w = q.sub(q.floor());
                let one = autorust::tensor::Tensor::full_const(
                    &ctx,
                    &[] as &[i64],
                    autorust::dtype::ConstVal::Float(1.0),
                    None,
                );
                let value = lo.mul(one.sub(&w)).add(hi.mul(&w));
                let slope = hi.sub(&lo);
                vec![value, slope]
            });
            let _ = outs1;
            if rep >= 5 {
                onehot_interp.push(t.elapsed().as_secs_f64());
            }
        }

        if only.is_empty() || only == "direct_interp" {
            let t = Instant::now();
            let outs = jit4.call_many(&ctx, &mut sess, |_c, _s| {
                let (v, s) = atlas.gather_interp(&rows, &q);
                vec![v, s]
            });
            let _ = outs;
            if rep >= 5 {
                direct_interp.push(t.elapsed().as_secs_f64());
            }
        }
    }
    println!("device {:?}", d);
    if !onehot.is_empty() {
        println!("one-hot gather      {:.3} ms", median(onehot) * 1e3);
    }
    if !direct.is_empty() {
        println!("direct gather       {:.3} ms", median(direct) * 1e3);
    }
    if !onehot_interp.is_empty() {
        println!("one-hot interp taps {:.3} ms", median(onehot_interp) * 1e3);
    }
    if !direct_interp.is_empty() {
        println!("direct gather_interp{:.3} ms", median(direct_interp) * 1e3);
    }
    println!("index outputs {} ({} bytes f32)", r * r, r * r * 4);
    println!(
        "one-hot mask conceptual {:.1} GB; direct O(out) 2 arrays = {} bytes",
        (r * r * n * s) as f64 * 4.0 / 1e9,
        r * r * 2 * 4
    );
}
