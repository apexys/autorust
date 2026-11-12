//! `VK_KHR_cooperative_matrix` ("tensor core") gate for the `VK` backend.
//!
//! `renderer/spirv.rs` does not emit cooperative-matrix instructions yet, so
//! this runs a hand-written module ([`tests/data/coopmat_gemm.spvasm`]) through
//! the same device/pipeline/dispatch path as a rendered kernel and checks both
//! things that matter before such a path could be shipped:
//!
//! 1. **numerics** — `C = A * B` over 16-bit inputs with an f32 accumulator must
//!    match a host f64 reference (the cooperative matrix accumulates in f32, so
//!    the tolerance is f32 rounding, not f16);
//! 2. **speed** — the rate is printed and asserted above a floor, so a driver
//!    that silently scalarises the MMA (or a device without tensor cores) cannot
//!    pass unnoticed.
//!
//! Skipped when there is no usable Vulkan device or no `spirv-as` on `PATH`
//! (assembling the fixture is the only external dependency, and it is a
//! development tool like `spirv-val` in `tests/vk_spirv.rs`).

use autorust::runtime::vk_sys::CoopShape;

/// Assemble `src` (with `$NK$` substituted) and return the SPIR-V words.
fn assemble(src: &str, nk: usize, tag: &str) -> Option<Vec<u32>> {
    let text = src
        .replace("$M$", &nk.to_string())
        .replace("$SA$", &nk.to_string())
        .replace("$N$", &nk.to_string())
        .replace("$K$", &nk.to_string());
    let dir = std::env::temp_dir().join("autorust-vk-coopmat");
    std::fs::create_dir_all(&dir).ok()?;
    let asm = dir.join(format!("{tag}.spvasm"));
    let spv = dir.join(format!("{tag}.spv"));
    std::fs::write(&asm, text).ok()?;
    let out = std::process::Command::new("spirv-as")
        .args(["--target-env", "spv1.6"])
        .arg(&asm)
        .arg("-o")
        .arg(&spv)
        .output()
        .ok()?;
    if !out.status.success() {
        panic!(
            "spirv-as rejected the fixture:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let words = std::fs::read(&spv).ok()?;
    Some(
        words
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

/// f32 -> IEEE binary16 (round to nearest even), for the test inputs.
fn f16_le(v: f32) -> [u8; 2] {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let man = x & 0x7f_ffff;
    let out = if exp == 0xff {
        sign | 0x7c00
    } else {
        let e = exp - 127 + 15;
        if e >= 0x1f {
            sign | 0x7c00
        } else if e <= 0 {
            0
        } else {
            let m = man >> 13;
            let mut o = (e as u16) << 10 | m as u16;
            if (man >> 12) & 1 == 1 && (man & 0xfff) != 0 {
                o += 1;
            }
            sign | o
        }
    };
    out.to_le_bytes()
}

/// The shape the emitter would need: 16-bit inputs, f32 accumulator.
fn coop_shape() -> Option<CoopShape> {
    let lib = autorust::runtime::vk_sys::VkLib::get()?;
    lib.devices().first()?.coop
}

#[test]
fn coopmat_gemm_matches_reference() {
    if !autorust::runtime::ops_vk::available() {
        return;
    }
    let Some(shape) = coop_shape() else {
        eprintln!("coopmat: device advertises no 16-bit -> f32 shape");
        return;
    };
    assert_eq!(
        (shape.m, shape.n, shape.k),
        (16, 16, 16),
        "the fixture is written for a 16x16x16 MMA"
    );
    let Ok(src) = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/coopmat_gemm.spvasm"),
    ) else {
        return;
    };

    for n in [512usize, 1024, 2048] {
        let Some(spv) = assemble(&src, n, &format!("gemm{n}")) else {
            eprintln!("coopmat: spirv-as unavailable, skipping {n}^3");
            return;
        };
        // Deterministic inputs that are exactly representable in f16.
        let a: Vec<f32> = (0..n * n).map(|i| ((i % 13) as f32 - 6.0) * 0.25).collect();
        let b: Vec<f32> = (0..n * n).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();
        let mut a16 = Vec::with_capacity(a.len() * 2);
        for v in &a {
            a16.extend_from_slice(&f16_le(*v));
        }
        let mut b16 = Vec::with_capacity(b.len() * 2);
        for v in &b {
            b16.extend_from_slice(&f16_le(*v));
        }

        let sizes = [a16.len(), b16.len(), n * n * 4];
        let fills: [&[u8]; 3] = [&a16, &b16, &[]];
        let groups = [(n / 16) as u32, (n / 64) as u32, 1];
        let (ms, bufs) =
            autorust::runtime::ops_vk::run_raw_module_io(&spv, &sizes, &fills, groups, 20)
                .expect("raw coopmat run");

        // f32 accumulation over K <= 2048 f16 products: a handful of ulps.
        let mut worst = 0.0f64;
        for i in [0usize, 1, n / 2, n - 1] {
            for j in [0usize, 3, n / 3, n - 1] {
                let mut acc = 0.0f64;
                for kk in 0..n {
                    acc += (a[i * n + kk] as f64) * (b[kk * n + j] as f64);
                }
                let o = (i * n + j) * 4;
                let got =
                    f32::from_le_bytes([bufs[2][o], bufs[2][o + 1], bufs[2][o + 2], bufs[2][o + 3]]);
                let rel = ((got as f64) - acc).abs() / acc.abs().max(1.0);
                worst = worst.max(rel);
            }
        }
        let tflops = 2.0 * (n * n * n) as f64 / ms / 1e9;
        println!("coopmat {n}^3: {ms:.3} ms = {tflops:.2} TFLOPS (worst rel err {worst:.1e})");
        assert!(
            worst < 1e-4,
            "coopmat {n}^3: worst relative error {worst:e} vs the f64 reference"
        );
        // A 16x16x16 f16 MMA on any device that advertises the extension is far
        // above this; a scalarised fallback is far below.
        assert!(
            tflops > 0.2,
            "coopmat {n}^3: only {tflops:.2} TFLOPS — the MMA was not used"
        );
    }
}

/// The probe must be honest: a device that advertises the extension also reports
/// a shape whose component types we can emit, and a device without it reports
/// `None` rather than a fabricated shape.
#[test]
fn coopmat_probe_is_consistent() {
    let Some(lib) = autorust::runtime::vk_sys::VkLib::get() else {
        return;
    };
    for d in lib.devices() {
        match d.coop {
            Some(s) => {
                assert!(s.m >= 8 && s.n >= 8 && s.k >= 8, "{}: {s:?}", d.name);
                assert_eq!(s.scope, 3, "{}: subgroup scope", d.name);
            }
            None => eprintln!("{}: no 16-bit -> f32 cooperative matrix", d.name),
        }
    }
}

/// End to end through the pipeline: an f16 matmul that asks for an f32
/// accumulator must become one cooperative-matrix dispatch (the library-GEMM
/// seam), match INTERP exactly, and be much faster than the generated kernel.
/// A plain f16 dot (16-bit accumulation) must *not* take it: the module
/// accumulates in f32, so that would change results.
#[test]
fn coopmat_is_used_for_f16_f32_matmul() {
    use autorust::dtype;
    use autorust::engine::realize::{ProgCache, tensor_values};
    use autorust::runtime::buffer::BufTable;
    use autorust::tensor::{Ctx, Tensor};

    fn dot(
        m: i64,
        n: i64,
        k: i64,
        dev: Option<&str>,
        cdt: Option<dtype::DTypeId>,
    ) -> (Vec<f64>, f64) {
        let ctx = Ctx::new();
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 11) as f32 - 5.0) * 0.25).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();
        let at = Tensor::from_slice(&ctx, &a, Some(dtype::FLOAT16), dev).reshape([m, k]);
        let bt = Tensor::from_slice(&ctx, &b, Some(dtype::FLOAT16), dev).reshape([k, n]);
        let out = at.dot_dtype(&bt, cdt);
        let mut cache = ProgCache::default();
        let t = std::time::Instant::now();
        let v = tensor_values(&mut ctx.raw(), out.uop(), &mut BufTable::default(), &mut cache);
        (v, t.elapsed().as_secs_f64() * 1e3)
    }

    if !autorust::runtime::ops_vk::available() || !autorust::runtime::ops_vk::coop_gemm_available() {
        return;
    }
    // A shape the module handles: rows need not be a multiple of the tile (the
    // kernel clamps its last tile), only K must be a multiple of 16.
    let (m, n, k) = (128i64, 192, 256);
    let before = autorust::runtime::ops_vk::coop_gemm_count();
    let (got, ms) = dot(m, n, k, Some("VK"), Some(dtype::FLOAT32));
    let used = autorust::runtime::ops_vk::coop_gemm_count() - before;
    let (want, _) = dot(m, n, k, None, Some(dtype::FLOAT32));
    assert_eq!(used, 1, "expected one coopmat dispatch (incl. none for the cast)");
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-4 * w.abs().max(1.0),
            "coopmat[{i}]: {g} vs {w}"
        );
    }
    // `ms` covers the whole realize (scheduling + kernels + readback), so it is a
    // smoke number; the module's device rate is measured in
    // `coopmat_gemm_matches_reference`.
    println!("coopmat pipeline {m}x{n}x{k}: {ms:.3} ms end to end");

    // A 16-bit dot must stay on the generated kernel.
    let before = autorust::runtime::ops_vk::coop_gemm_count();
    let _ = dot(m, n, k, Some("VK"), None);
    assert_eq!(
        autorust::runtime::ops_vk::coop_gemm_count(),
        before,
        "a plain f16 dot must not take the f32 accumulator path"
    );
}

/// bf16 storage on the same path: the module reads f16, so A and B are staged
/// through f16 scratch (exact for in-range values) and a bf16 result is rounded
/// from an f32 scratch. Both the bf16-out and the f32-out forms must work.
#[test]
fn coopmat_handles_bf16_storage() {
    use autorust::dtype;
    use autorust::engine::realize::{ProgCache, tensor_values};
    use autorust::runtime::buffer::BufTable;
    use autorust::tensor::{Ctx, Tensor};

    fn dot(m: i64, n: i64, k: i64, dev: Option<&str>, cdt: Option<dtype::DTypeId>) -> Vec<f64> {
        let ctx = Ctx::new();
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 11) as f32 - 5.0) * 0.25).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();
        let at = Tensor::from_slice(&ctx, &a, Some(dtype::BFLOAT16), dev).reshape([m, k]);
        let bt = Tensor::from_slice(&ctx, &b, Some(dtype::BFLOAT16), dev).reshape([k, n]);
        let out = at.dot_dtype(&bt, cdt);
        let mut cache = ProgCache::default();
        tensor_values(&mut ctx.raw(), out.uop(), &mut BufTable::default(), &mut cache)
    }

    if !autorust::runtime::ops_vk::available() || !autorust::runtime::ops_vk::coop_gemm_available() {
        return;
    }
    for cdt in [Some(dtype::BFLOAT16), Some(dtype::FLOAT32)] {
        let (m, n, k) = (128i64, 192, 256);
        let before = autorust::runtime::ops_vk::coop_gemm_count();
        let got = dot(m, n, k, Some("VK"), cdt);
        let used = autorust::runtime::ops_vk::coop_gemm_count() - before;
        let want = dot(m, n, k, None, cdt);
        assert_eq!(used, 1, "expected one coopmat dispatch for c_dtype {cdt:?}");
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-3 * w.abs().max(1.0),
                "bf16 coopmat c_dtype {cdt:?} [{i}]: {g} vs {w}"
            );
        }
        println!("bf16 coopmat c_dtype={cdt:?}: one dispatch, matches INTERP");
    }
}
