//! Square/rectangular GEMM benchmark used to tune the CPU/CUDA matmul paths.
//!
//! ```text
//! DEV=RUST M=512 N=512 K=512 ITERS=20 cargo run --release --example gemm_bench
//! DEV=CUDA M=4096 N=4096 K=4096 ITERS=20 cargo run --release --features backend_cuda --example gemm_bench
//! ```
//!
//! `DEV` defaults to the selected device. `WARM` warms the kernel cache.
//! `JIT=1` captures the schedule once and replays it (steady-state execution).
//! `BIAS=1` adds a `+ bias` epilogue (checks that fusion survives).
//! `DUMP=1` prints the rendered kernel source (requires `--features dump`).

use autorust::dtype;
use autorust::engine::jit::Jit;
use autorust::engine::session::Session;
use autorust::tensor::{Ctx, Tensor};
use std::time::Instant;

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(d)
}

fn env_flag(k: &str) -> bool {
    std::env::var(k)
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false)
}

fn main() {
    let m = env_usize("M", 512);
    let n = env_usize("N", 512);
    let k = env_usize("K", 512);
    let iters = env_usize("ITERS", 10).max(1);
    let warm = env_usize("WARM", 2);
    let jit_on = env_flag("JIT");
    let bias_on = env_flag("BIAS");
    let dt_name = std::env::var("DT").unwrap_or_else(|_| "f32".into());
    let dt = match dt_name.as_str() {
        "f32" => dtype::FLOAT32,
        "f16" | "half" => dtype::FLOAT16,
        "bf16" | "bfloat16" => dtype::BFLOAT16,
        other => panic!("unknown DT={other} (f32/f16/bf16)"),
    };
    // Accumulate in f32 unless explicitly asked to keep the input dtype.
    let acc = if env_flag("ACC_SAME") {
        dt
    } else {
        dtype::FLOAT32
    };
    let dev = std::env::var("DEV").unwrap_or_else(|_| autorust::tensor::creation::default_device());

    let ctx = Ctx::new();
    let mut sess = Session::new();
    if jit_on {
        sess.enable_jit();
    }
    let shape = |i: usize| ((i % 131) as f32) * 0.0078125 - 0.5;
    let ad: Vec<f32> = (0..m * k).map(shape).collect();
    let bd: Vec<f32> = (0..k * n).map(|i| shape(i + 17)).collect();
    let bsd: Vec<f32> = (0..n).map(|i| shape(i + 41)).collect();
    // Parameters keep their buffers alive across realizes (otherwise a later
    // `prune` can free them and a non-JIT reference reads freshly zeroed VRAM).
    let a = Tensor::from_slice(&ctx, &ad, Some(dt), Some(&dev))
        .reshape([m, k])
        .to_param();
    // `BT=1` stores B as `[n, k]` and transposes the view, i.e. exactly what a
    // `Linear` weight looks like (k-contiguous columns, `b_k == 1`).
    let b = if env_flag("BT") {
        Tensor::from_slice(&ctx, &bd, Some(dt), Some(&dev))
            .reshape([n, k])
            .to_param()
            .transpose(0, 1)
    } else {
        Tensor::from_slice(&ctx, &bd, Some(dt), Some(&dev))
            .reshape([k, n])
            .to_param()
    };
    let bias = Tensor::from_slice(&ctx, &bsd, Some(acc), Some(&dev))
        .reshape([n])
        .to_param();

    // Materialize the operands once, **before** capture. Otherwise the JIT
    // captures the host→device copy (or the f32→bf16 creation cast) and replays
    // it every iteration, which makes a small GEMM look bandwidth-bound.
    // Read through `sess` (not `realize_fresh`, whose throwaway session would
    // drop the realized storage and leave later graphs reading zeros).
    let _ = a.to_list::<f32>(&mut sess);
    let _ = b.to_list::<f32>(&mut sess);
    let _ = bias.to_list::<f32>(&mut sess);

    let build = |_ctx: &Ctx, sess: &mut Session| -> Tensor {
        let _ = sess;
        let y = if dt == dtype::FLOAT32 {
            a.dot(&b)
        } else {
            a.dot_dtype(&b, Some(acc))
        };
        if bias_on { y.add(&bias) } else { y }
    };

    let mut jit = Jit::new();
    for _ in 0..warm {
        let y = if jit_on {
            jit.call(&ctx, &mut sess, build)
        } else {
            build(&ctx, &mut sess)
        };
        let _ = y.realize_fresh();
        let _ = y.to_list::<f32>(&mut sess);
    }

    // `NOSYNC=1` queues all iterations and reads back once, so GPU numbers
    // measure device time rather than per-iteration D2H latency.
    let nosync = env_flag("NOSYNC");
    let prof = env_flag("PROF");
    let sync_mode = env_flag("SYNC");
    let t = Instant::now();
    let mut last = None;
    for it in 0..iters {
        let t0 = Instant::now();
        let y = if jit_on {
            jit.call(&ctx, &mut sess, build)
        } else {
            build(&ctx, &mut sess)
        };
        let t1 = Instant::now();
        let was_realized = y.realized_buffer().is_some();
        if prof {
            eprintln!(
                "iter {it}: buffer device={:?} (cuda available={})",
                y.realized_buffer().map(|b| b.device().to_string()),
                {
                    #[cfg(all(
                        feature = "backend_cuda",
                        any(target_os = "linux", target_os = "windows")
                    ))]
                    {
                        autorust::runtime::ops_cuda::available()
                    }
                    #[cfg(not(all(
                        feature = "backend_cuda",
                        any(target_os = "linux", target_os = "windows")
                    )))]
                    {
                        false
                    }
                }
            );
        }
        if sync_mode {
            // No host copy: enqueue (realize if needed), then wait on the stream.
            let t3 = Instant::now();
            y.realize_device_ctx(&mut ctx.borrow_mut(), &mut sess);
            let t4 = Instant::now();
            #[cfg(all(
                feature = "backend_cuda",
                any(target_os = "linux", target_os = "windows")
            ))]
            autorust::runtime::ops_cuda::synchronize();
            let t5 = Instant::now();
            if prof {
                eprintln!(
                    "iter {it}: realize_dev {:.2} ms, sync {:.2} ms, realized_after={}",
                    (t4 - t3).as_secs_f64() * 1e3,
                    (t5 - t4).as_secs_f64() * 1e3,
                    y.realized_buffer().is_some()
                );
            }
        } else {
            // Realize in the persistent session: a fresh session cannot see the
            // params realized here (it would allocate empty replacements).
            y.realize_device_ctx(&mut ctx.borrow_mut(), &mut sess);
        }
        let t2 = Instant::now();
        if prof && !sync_mode {
            eprintln!(
                "iter {it}: build/jit {:.2} ms (realized={was_realized}), realize {:.2} ms, devices a={:?} b={:?} y={:?} default={:?}",
                (t1 - t0).as_secs_f64() * 1e3,
                (t2 - t1).as_secs_f64() * 1e3,
                a.device(),
                b.device(),
                y.device(),
                autorust::tensor::creation::default_device(),
            );
        }
        if nosync {
            last = Some(y.clone());
        } else {
            let t3 = Instant::now();
            let _ = y.to_list::<f32>(&mut sess);
            if prof {
                eprintln!("iter {it}: to_list {:.2} ms", (t3 - t1).as_secs_f64() * 1e3);
            }
        }
    }
    if let Some(y) = &last {
        let _ = y.to_list::<f32>(&mut sess);
    }
    let secs = t.elapsed().as_secs_f64() / iters as f64;
    let gflops = 2.0 * (m * n * k) as f64 / secs / 1e9;
    println!(
        "gemm {m}x{n}x{k} dev={dev} dt={dt_name} acc={acc:?} jit={jit_on} bias={bias_on}: {:.3} ms/iter, {:.1} GFLOP/s{}",
        secs * 1e3,
        gflops,
        if jit_on {
            format!(" ({} linears)", jit.n_linears())
        } else {
            String::new()
        }
    );
    if env_flag("CHECK") {
        // Host triple-loop reference over the *original* f32 data. This is
        // independent of every backend and catches PADTO masking mistakes.
        assert_eq!(dt, dtype::FLOAT32, "CHECK currently supports DT=f32");
        let got = if let Some(y) = &last {
            y.to_list::<f32>(&mut sess)
        } else {
            let y = build(&ctx, &mut sess);
            y.to_list::<f32>(&mut sess)
        };
        if env_flag("CHECK_VERBOSE") {
            let av = a.to_list::<f32>(&mut sess);
            let bv = b.to_list::<f32>(&mut sess);
            println!(
                "check: a[:4]={:?} host_a[:4]={:?} b[:4]={:?} host_b[:4]={:?}",
                &av[..4.min(av.len())],
                &ad[..4.min(ad.len())],
                &bv[..4.min(bv.len())],
                &bd[..4.min(bd.len())]
            );
            println!(
                "check: got_len={} want_len={} total={}",
                got.len(),
                m * n,
                m * n
            );
            let mut head = Vec::new();
            for i in 0..m.min(2) {
                for j in 0..n.min(4) {
                    let mut acc = 0.0f64;
                    for p in 0..k {
                        acc += ad[i * k + p] as f64 * bd[p * n + j] as f64;
                    }
                    if bias_on {
                        acc += bsd[j] as f64;
                    }
                    head.push((i, j, acc as f32, got[i * n + j]));
                }
            }
            println!("check head: {head:?}");
        }
        let mut maxd = 0.0f64;
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for p in 0..k {
                    acc += ad[i * k + p] as f64 * bd[p * n + j] as f64;
                }
                if bias_on {
                    acc += bsd[j] as f64;
                }
                let d = (acc as f32 - got[i * n + j]).abs() as f64;
                maxd = maxd.max(d);
            }
        }
        println!("check(host): max |diff| = {maxd:e}");
        assert!(maxd < 1e-2, "check failed: max |diff| = {maxd:e}");
    }
    println!(
        "to_program: {} misses / {} hits",
        sess.cache.misses, sess.cache.hits
    );
    if env_flag("VERBOSE") && jit_on {
        for (i, st) in jit.steps().iter().enumerate() {
            if let autorust::engine::jit::JitStep::Linear(l) = st {
                for p in l.progs.values() {
                    println!(
                        "  linear {i}: {} global={:?} local={:?}",
                        p.pi.name, p.pi.global_size, p.pi.local_size
                    );
                }
            }
        }
    }
    autorust::engine::prof::report();
}
