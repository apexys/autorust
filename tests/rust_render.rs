//! `RUST` renderer gate: render real scheduled kernels to Rust and compile
//! them with `rustc`.
//!
//! This is a renderer-only test (no device registration): it builds a tensor
//! graph, schedules it, re-lowers each kernel with `Renderer::rust()`, and
//! checks that the emitted source is valid Rust with the fixed
//! `(bufs, vals, gidx)` ABI.

use autorust::renderer::Renderer;
use autorust::renderer::cstyle::render_rust;
use autorust::tensor::{Ctx, Tensor};
use autorust::uop::Ops;
use autorust::uop::UopId;
use autorust::uop::ctx::UOpCtx;

/// Schedule `roots` and render every compute kernel with the Rust renderer.
fn render_roots(ctx: &mut UOpCtx, roots: Vec<UopId>) -> Vec<String> {
    let sink = autorust::tensor::sink(ctx, roots);
    let (call, _map) = autorust::tensor::callify::transform_to_call(ctx, sink);
    let (_linear, planned, _held) = autorust::schedule::sched::create_linear_with_vars(ctx, call);
    let ren = Renderer::rust();
    let calls = ctx.srcs(&planned).to_vec();
    let mut out = Vec::new();
    for call in calls {
        let body = ctx.srcs(&call)[0].clone();
        if ctx.op_of(&body) == Ops::Copy {
            continue;
        }
        let sunk = autorust::codegen::pipeline::full_rewrite_to_sink_with(ctx, body, &ren, true);
        let lst = autorust::codegen::linearize::linearize(ctx, sunk);
        out.push(render_rust(ctx, &lst));
    }
    out
}

/// Compile one rendered kernel as a `cdylib`; returns the source on failure.
fn compile(src: &str) {
    // The real compiler will add the no_std/libm prelude. For this test a std
    // crate is enough to prove the emitted body is syntactically valid.
    let text = format!(
        "#![allow(dead_code, unused_mut, unused_variables, unused_parens, unsafe_op_in_unsafe_fn)]\n\
         #[link(name = \"m\")]\n\
         unsafe extern \"C\" {{\n\
             fn sqrtf(x: f32) -> f32;\n\
             fn sqrt(x: f64) -> f64;\n\
             fn exp2f(x: f32) -> f32;\n\
             fn log2f(x: f32) -> f32;\n\
             fn sinf(x: f32) -> f32;\n\
             fn truncf(x: f32) -> f32;\n\
         }}\n\
         {src}\n"
    );
    let dir = std::env::temp_dir().join(format!(
        "autorust-rust-render-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let rs = dir.join("k.rs");
    let so = dir.join("k.so");
    std::fs::write(&rs, &text).unwrap();
    if std::env::var("PRINT_RUST").is_ok() {
        eprintln!("--- rendered kernel ---\n{text}");
    }
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let out = std::process::Command::new(rustc)
        .args([
            "--crate-type=cdylib",
            "--edition=2024",
            "-C",
            "opt-level=2",
            "-C",
            "panic=abort",
        ])
        .arg(&rs)
        .arg("-o")
        .arg(&so)
        .output()
        .expect("run rustc");
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        out.status.success(),
        "rustc rejected rendered source:\n{}\n--- source ---\n{text}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn renders_elementwise_chain() {
    let ctx = Ctx::new();
    let a = Tensor::from_slice(&ctx, &[1.0f32, 2.0, 3.0, 4.0], None, Some("INTERP"));
    let out = a.add(1.0).sqrt().mul(2.0);
    let kernels = {
        let mut c = ctx.borrow_mut();
        render_roots(&mut c, vec![out.uop().clone()])
    };
    assert!(!kernels.is_empty(), "expected at least one compute kernel");
    for k in &kernels {
        assert!(k.contains("pub unsafe extern \"C\" fn"), "{k}");
        assert!(k.contains("*const *mut u8"), "{k}");
        compile(k);
    }
}

#[test]
fn renders_matmul() {
    let ctx = Ctx::new();
    let a =
        Tensor::from_slice(&ctx, &[1.0f32, 2.0, 3.0, 4.0], None, Some("INTERP")).reshape([2, 2]);
    let b =
        Tensor::from_slice(&ctx, &[5.0f32, 6.0, 7.0, 8.0], None, Some("INTERP")).reshape([2, 2]);
    let out = a.dot(&b);
    let kernels = {
        let mut c = ctx.borrow_mut();
        render_roots(&mut c, vec![out.uop().clone()])
    };
    assert!(!kernels.is_empty(), "expected a matmul kernel");
    for k in &kernels {
        compile(k);
    }
}

#[test]
fn renders_global_grid() {
    let ctx = Ctx::new();
    let data: Vec<f32> = (0..1024).map(|i| i as f32 * 0.5).collect();
    let a = Tensor::from_slice(&ctx, &data, None, Some("INTERP"));
    let out = a.add(1.0).mul(2.0);
    let kernels = {
        let mut c = ctx.borrow_mut();
        render_roots(&mut c, vec![out.uop().clone()])
    };
    assert!(!kernels.is_empty(), "expected a grid kernel");
    for k in &kernels {
        assert!(k.contains("gidx"), "expected a global-id argument:\n{k}");
        compile(k);
    }
}

#[test]
fn renders_boolean_where_and_casts() {
    let ctx = Ctx::new();
    let a = Tensor::from_slice(&ctx, &[1.0f32, -2.0, 3.0, -4.0], None, Some("INTERP"));
    let out = a.lt(0.0).where_(&a, a.neg());
    let kernels = {
        let mut c = ctx.borrow_mut();
        render_roots(&mut c, vec![out.uop().clone()])
    };
    assert!(!kernels.is_empty(), "expected a where kernel");
    for k in &kernels {
        compile(k);
    }
}

/// Integer shift chains (the `threefry` hash in `rand`/dropout) make rustc's
/// loop vectorizer take exponentially long, so the rendered loop index goes
/// through `black_box`. A plain elementwise kernel must keep the plain loop.
#[test]
fn shift_heavy_kernels_mark_the_loop_index() {
    let ctx = Ctx::new();
    let data: Vec<u32> = (0..1024).collect();
    let a = Tensor::from_slice(&ctx, &data, Some(autorust::dtype::UINT32), Some("INTERP"));
    let mut x = a;
    for i in 0..20u32 {
        x = x.shl(3 + (i % 7) as i64).rshift(5 + (i % 5) as i64);
        x = x.bitwise_xor(i as i64);
    }
    let kernels = {
        let mut c = ctx.borrow_mut();
        render_roots(&mut c, vec![x.uop().clone()])
    };
    assert!(!kernels.is_empty(), "expected a shift kernel");
    assert!(
        kernels.iter().any(|k| k.contains("core::hint::black_box")),
        "expected a black_box loop index:\n{}",
        kernels.join("\n---\n")
    );
    for k in &kernels {
        compile(k);
    }

    let plain = Tensor::from_slice(&ctx, &[1.0f32, 2.0, 3.0, 4.0], None, Some("INTERP"));
    let out = plain.add(1.0).mul(2.0);
    let kernels = {
        let mut c = ctx.borrow_mut();
        render_roots(&mut c, vec![out.uop().clone()])
    };
    for k in &kernels {
        assert!(!k.contains("core::hint::black_box"), "{k}");
    }
}
