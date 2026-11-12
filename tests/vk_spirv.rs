//! `VK` renderer gate: render real scheduled kernels to SPIR-V and validate
//! each module with `spirv-val` (the Khronos validator, same rules the Vulkan
//! drivers apply at pipeline creation).
//!
//! Renderer-only: no Vulkan device is opened, so the test runs everywhere the
//! optional `spirv-val` binary exists (it is skipped otherwise).

use autorust::renderer::Renderer;
use autorust::renderer::spirv::{Caps, render};
use autorust::tensor::{Ctx, Tensor};
use autorust::uop::Ops;
use autorust::uop::UopId;
use autorust::uop::ctx::UOpCtx;

fn caps() -> Caps {
    Caps {
        f16: true,
        storage16: true,
        int64: true,
        f64: true,
        storage8: true,
        int8: true,
        int16: true,
        spirv13: true,
    }
}

/// Schedule `roots` and render every compute kernel to SPIR-V.
fn render_roots(ctx: &mut UOpCtx, roots: Vec<UopId>) -> Vec<autorust::renderer::spirv::VkModule> {
    let sink = autorust::tensor::sink(ctx, roots);
    let (call, _map) = autorust::tensor::callify::transform_to_call(ctx, sink);
    let (_linear, planned, _held) = autorust::schedule::sched::create_linear_with_vars(ctx, call);
    let ren = Renderer::opencl(); // the VK facts differ only in limits/dtypes
    let calls = ctx.srcs(&planned).to_vec();
    let mut out = Vec::new();
    for call in calls {
        let body = ctx.srcs(&call)[0].clone();
        if ctx.op_of(&body) == Ops::Copy {
            continue;
        }
        let sunk = autorust::codegen::pipeline::full_rewrite_to_sink_with(ctx, body, &ren, true);
        let lst = autorust::codegen::linearize::linearize(ctx, sunk);
        out.push(render(ctx, &lst, caps()));
    }
    out
}

fn validate(name: &str, m: &autorust::renderer::spirv::VkModule) {
    let dir = std::env::temp_dir().join("autorust-vk-spirv");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{name}.spv"));
    let mut bytes = Vec::with_capacity(m.words.len() * 4);
    for w in &m.words {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    std::fs::write(&path, &bytes).unwrap();
    let out = std::process::Command::new("spirv-val")
        .args(["--target-env", "vulkan1.1"])
        .arg(&path)
        .output();
    let out = match out {
        Ok(o) => o,
        Err(_) => return,
    };
    if !out.status.success() {
        let dis = std::process::Command::new("spirv-dis")
            .arg(&path)
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
            .unwrap_or_default();
        panic!(
            "spirv-val rejected {name}:\n{}\n--- module ---\n{dis}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    // Round-trip the payload envelope too.
    let back = autorust::renderer::spirv::VkModule::decode(&m.encode()).expect("decode");
    assert_eq!(&back.words, &m.words);
    assert_eq!(back.slots, m.slots);
    assert_eq!(back.local, m.local);
}

fn check(name: &str, mut kernels: Vec<autorust::renderer::spirv::VkModule>) {
    assert!(!kernels.is_empty(), "{name}: no kernels");
    for (i, m) in kernels.drain(..).enumerate() {
        validate(&format!("{name}_{i}"), &m);
    }
}

#[test]
fn renders_elementwise_chain() {
    let ctx = Ctx::new();
    let a = Tensor::from_slice(&ctx, &[1.0f32, 2.0, 3.0, 4.0], None, Some("INTERP"));
    let out = a.add(1.0).sqrt().mul(2.0);
    let k = {
        let mut c = ctx.borrow_mut();
        render_roots(&mut c, vec![out.uop().clone()])
    };
    check("elementwise", k);
}

#[test]
fn renders_where_bitops_and_casts() {
    let ctx = Ctx::new();
    let a = Tensor::from_slice(&ctx, &[1.0f32, -2.0, 3.0, -4.0], None, Some("INTERP"));
    let w = a.lt(0.0).where_(&a, a.neg());
    let c = w.cast(autorust::dtype::INT32);
    let d = c.cast(autorust::dtype::FLOAT32);
    let k = {
        let mut c = ctx.borrow_mut();
        render_roots(&mut c, vec![d.uop().clone()])
    };
    check("where_cast", k);
}

#[test]
fn renders_matmul_reduce_and_local_memory() {
    let ctx = Ctx::new();
    let data: Vec<f32> = (0..4096).map(|i| (i % 17) as f32 * 0.25).collect();
    let a = Tensor::from_slice(&ctx, &data, None, Some("INTERP")).reshape([64, 64]);
    let b = a.transpose(0, 1);
    let out = a.dot(&b);
    let k = {
        let mut c = ctx.borrow_mut();
        render_roots(&mut c, vec![out.uop().clone()])
    };
    check("matmul", k);
}

#[test]
fn renders_softmax_and_amax() {
    let ctx = Ctx::new();
    let data: Vec<f32> = (0..2048).map(|i| (i % 23) as f32 * 0.5 - 5.0).collect();
    let a = Tensor::from_slice(&ctx, &data, None, Some("INTERP")).reshape([32, 64]);
    let s = autorust::tensor::nn::softmax(&a, Some(1));
    let m = a.max(Some(&[1usize]), true);
    let k = {
        let mut c = ctx.borrow_mut();
        render_roots(
            &mut c,
            vec![s.uop().clone(), m.uop().clone()],
        )
    };
    check("softmax", k);
}

/// A bool literal in the graph must render as `OpConstantTrue/False`.
#[test]
fn renders_bool_constants() {
    let ctx = Ctx::new();
    let a = Tensor::from_slice(&ctx, &[1.0f32, -2.0, 3.0, -4.0], None, Some("INTERP"));
    let mask = Tensor::from_slice(&ctx, &[1u8, 0, 1, 0], Some(autorust::dtype::BOOL), Some("INTERP"));
    let out = mask.where_(&a, a.neg());
    let k = {
        let mut c = ctx.borrow_mut();
        render_roots(&mut c, vec![out.uop().clone()])
    };
    check("bool_const", k);
}

#[test]
fn renders_shift_chain() {
    let ctx = Ctx::new();
    let data: Vec<u32> = (0..1024).collect();
    let a = Tensor::from_slice(&ctx, &data, Some(autorust::dtype::UINT32), Some("INTERP"));
    let mut x = a;
    for i in 0..12u32 {
        x = x.shl(3 + (i % 7) as i64).rshift(5 + (i % 5) as i64);
        x = x.bitwise_xor(i as i64);
    }
    let k = {
        let mut c = ctx.borrow_mut();
        render_roots(&mut c, vec![x.uop().clone()])
    };
    check("shift", k);
}

/// The probe module the device availability check compiles must be valid.
#[test]
fn probe_module_validates() {
    let m = autorust::renderer::spirv::VkModule {
        slots: Vec::new(),
        local: [1, 1, 1],
        words: autorust::runtime::vk_sys::PROBE_SPV.to_vec(),
    };
    validate("probe", &m);
}
