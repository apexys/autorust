//! AOT emitter gate: emit a frozen plan, compile it as a standalone crate, and
//! compare its bytes with the eager captured output.

#![cfg(feature = "freeze")]

use std::process::{Command, Output};

use autorust::engine::jit::Jit;
use autorust::engine::session::Session;
use autorust::freeze::emit_rust;
use autorust::freeze::plan::{self, Step};
use autorust::runtime::device::device;
use autorust::tensor::{Ctx, Tensor};

/// Compile `source` as the `frozen.rs` of a throwaway crate and run `main`.
fn run_generated_crate(source: &str, main: &str) -> Output {
    let dir = std::env::temp_dir().join(format!(
        "autorust-freeze-emit-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(dir.join("src")).unwrap();
    let manifest = format!(
        "[package]\nname = \"frozen_check\"\nversion = \"0.0.0\"\nedition = \"2024\"\n\n\
         [dependencies]\nautorust = {{ path = {:?}, default-features = false, features = [\"freeze\"] }}\n",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::write(dir.join("Cargo.toml"), manifest).unwrap();
    std::fs::write(dir.join("src/frozen.rs"), source).unwrap();
    std::fs::write(
        dir.join("src/main.rs"),
        format!("include!(\"frozen.rs\");\n{main}"),
    )
    .unwrap();

    let out = Command::new("cargo")
        .current_dir(&dir)
        .env("CARGO_TARGET_DIR", dir.join("target"))
        .args(["run", "--quiet"])
        .output()
        .expect("run generated crate");
    let _ = std::fs::remove_dir_all(&dir);
    out
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn emitted_frozen_crate_matches_eager() {
    let ctx = Ctx::new();
    let mut sess = Session::new();
    sess.enable_jit();
    let x = Tensor::from_slice(&ctx, &[1.0f32, 2.0, 3.0, 4.0], None, Some("RUST")).to_param();
    let w = Tensor::from_slice(&ctx, &[2.0f32, 3.0, 4.0, 5.0], None, Some("RUST")).to_param();
    let mut jit = Jit::new();
    let outs = jit.call_many(&ctx, &mut sess, |_c, _s| {
        let a = x.mul(&w);
        let b = a.add(&x);
        vec![a, b]
    });
    let built = plan::build(
        &ctx,
        &mut sess,
        &jit,
        std::slice::from_ref(&x),
        std::slice::from_ref(&w),
    );
    let eager = {
        let b = outs[0].realized_buffer().expect("eager output");
        let d = device(b.device());
        d.copyout(&*b.get_buf(&d))
    };
    let source = emit_rust(&built);

    let out = run_generated_crate(
        &source,
        r#"use autorust::tensor::{Ctx, Tensor};

fn main() {
    let ctx = Ctx::new();
    let x = Tensor::from_slice(&ctx, &[1.0f32, 2.0, 3.0, 4.0], None, Some("RUST")).to_param();
    let w = Tensor::from_slice(&ctx, &[2.0f32, 3.0, 4.0, 5.0], None, Some("RUST")).to_param();
    let mut runner = autorust::freeze::FrozenRunner::from_fns(plan(), KERNELS.to_vec());
    let out = runner.run(&[x], &[w]);
    for b in &out[0] {
        print!("{b:02x}");
    }
    println!();
}
"#,
    );
    assert!(
        out.status.success(),
        "generated crate failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let got = String::from_utf8(out.stdout).unwrap();
    assert_eq!(
        got.trim(),
        hex(&eager),
        "generated plan output differs from eager"
    );
}

/// A frozen plan containing a library GEMM serialises its `HostGemm`
/// descriptor through [`autorust::schedule::host_gemm::HostGemm::to_rust_literal`].
/// Every field must appear, or the emitted crate does not compile.
#[test]
fn emitted_host_gemm_plan_compiles() {
    unsafe { std::env::set_var("DEFAULT_DEVICE", "RUST") };
    let ctx = Ctx::new();
    let mut sess = Session::new();
    sess.enable_jit();
    let x = Tensor::from_slice(&ctx, &vec![0.5f32; 64 * 256], None, Some("RUST"))
        .reshape([64, 256])
        .to_param();
    let w = Tensor::from_slice(&ctx, &vec![0.1f32; 256 * 128], None, Some("RUST"))
        .reshape([256, 128])
        .to_param();
    let mut jit = Jit::new();
    let outs = jit.call_many(&ctx, &mut sess, |_c, _s| vec![x.matmul(&w)]);
    let built = plan::build(
        &ctx,
        &mut sess,
        &jit,
        std::slice::from_ref(&x),
        std::slice::from_ref(&w),
    );
    assert!(
        built
            .plan
            .steps
            .iter()
            .any(|s| matches!(s, Step::Gemm { .. })),
        "matmul must lower to a HostGemm step"
    );
    let eager = {
        let b = outs[0].realized_buffer().expect("eager output");
        let d = device(b.device());
        d.copyout(&*b.get_buf(&d))
    };
    let source = emit_rust(&built);

    let out = run_generated_crate(
        &source,
        r#"use autorust::tensor::{Ctx, Tensor};

fn main() {
    let ctx = Ctx::new();
    let x = Tensor::from_slice(&ctx, &vec![0.5f32; 64 * 256], None, Some("RUST"))
        .reshape([64, 256])
        .to_param();
    let w = Tensor::from_slice(&ctx, &vec![0.1f32; 256 * 128], None, Some("RUST"))
        .reshape([256, 128])
        .to_param();
    let mut runner = autorust::freeze::FrozenRunner::from_fns(plan(), KERNELS.to_vec());
    let out = runner.run(&[x], &[w]);
    for b in &out[0] {
        print!("{b:02x}");
    }
    println!();
}
"#,
    );
    assert!(
        out.status.success(),
        "generated crate with a HostGemm failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let got = String::from_utf8(out.stdout).unwrap();
    assert_eq!(
        got.trim(),
        hex(&eager),
        "generated HostGemm plan output differs from eager"
    );
}
