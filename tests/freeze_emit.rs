//! AOT emitter gate: emit a frozen plan, compile it as a standalone crate, and
//! compare its bytes with the eager captured output.

#![cfg(feature = "freeze")]

use std::process::Command;

use autorust::engine::jit::Jit;
use autorust::engine::session::Session;
use autorust::freeze::emit_rust;
use autorust::freeze::plan;
use autorust::runtime::device::device;
use autorust::tensor::{Ctx, Tensor};

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
         [dependencies]\nautorust = {{ path = {:?}, features = [\"freeze\"] }}\n",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::write(dir.join("Cargo.toml"), manifest).unwrap();
    std::fs::write(dir.join("src/frozen.rs"), source).unwrap();
    std::fs::write(
        dir.join("src/main.rs"),
        r#"include!("frozen.rs");
use autorust::tensor::{Ctx, Tensor};

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
    )
    .unwrap();

    let out = Command::new("cargo")
        .current_dir(&dir)
        .env("CARGO_TARGET_DIR", dir.join("target"))
        .args(["run", "--quiet"])
        .output()
        .expect("run generated crate");
    assert!(
        out.status.success(),
        "generated crate failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let got = String::from_utf8(out.stdout).unwrap();
    let got = got.trim();
    let want: String = eager.iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(got, want, "generated plan output differs from eager");
    let _ = std::fs::remove_dir_all(&dir);
}
