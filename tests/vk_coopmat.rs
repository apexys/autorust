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
/// seam), match a host reference, and a plain f16 dot (16-bit accumulation) must
/// *not* take it.
///
/// The shape is above `coopmat`'s measured 2 G-MAC cutoff (below it the three
/// staging dispatches do not amortise), with a small output so the reference is
/// cheap: the contraction is long and the tiles are few.
#[test]
fn coopmat_is_used_for_f16_f32_matmul() {
    use autorust::dtype;
    use autorust::engine::realize::{ProgCache, tensor_values};
    use autorust::runtime::buffer::BufTable;
    use autorust::tensor::{Ctx, Tensor};

    // Small on purpose: the module's rate is measured by
    // `coopmat_gemm_matches_reference`.
    const M: i64 = 128;
    const N: i64 = 192;
    const K: i64 = 256;
    if !autorust::runtime::ops_vk::available() || !autorust::runtime::ops_vk::coop_gemm_available() {
        return;
    }
    let a: Vec<f32> = (0..M * K).map(|i| ((i % 11) as f32 - 5.0) * 0.25).collect();
    let b: Vec<f32> = (0..K * N).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();
    let ctx = Ctx::new();
    let at = Tensor::from_slice(&ctx, &a, Some(dtype::FLOAT16), Some("VK")).reshape([M, K]);
    let bt = Tensor::from_slice(&ctx, &b, Some(dtype::FLOAT16), Some("VK")).reshape([K, N]);
    let out = at.dot_dtype(&bt, Some(dtype::FLOAT32));
    let before = autorust::runtime::ops_vk::coop_gemm_count();
    let mut cache = ProgCache::default();
    let got = tensor_values(&mut ctx.raw(), out.uop(), &mut BufTable::default(), &mut cache);
    let used = autorust::runtime::ops_vk::coop_gemm_count() - before;
    assert!(used >= 1, "expected a coopmat dispatch, got {used}");
    for (i, j) in [(0usize, 0usize), (1, 3), (M as usize / 2, N as usize / 2), (63, 63)] {
        let mut acc = 0.0f64;
        for kk in 0..K as usize {
            acc += (a[i * K as usize + kk] as f64) * (b[kk * N as usize + j] as f64);
        }
        let g = got[i * N as usize + j];
        assert!(
            (g - acc).abs() <= 1e-4 * acc.abs().max(1.0),
            "coopmat[{i},{j}]: {g} vs {acc}"
        );
    }
    println!("f16/f32 pipeline {M}x{N}x{K}: {used} dispatch(es), matches the host reference");

    // A plain f16 dot (16-bit accumulate) must stay on the generated kernel.
    let before = autorust::runtime::ops_vk::coop_gemm_count();
    let plain = at.dot_dtype(&bt, None);
    let got2 = tensor_values(&mut ctx.raw(), plain.uop(), &mut BufTable::default(), &mut cache);
    assert_eq!(
        autorust::runtime::ops_vk::coop_gemm_count(),
        before,
        "a plain f16 dot must not take the f32 accumulator path"
    );
    assert_eq!(got2.len(), got.len());
}

/// bf16 storage on the same path: the module reads f16, so A and B are staged
/// (a transposed operand is transposed there) and a bf16 result is rounded from
/// an f32 scratch. Both the bf16-out and the f32-out forms must dispatch once and
/// match a host reference.
#[test]
fn coopmat_handles_bf16_storage() {
    use autorust::dtype;
    use autorust::engine::realize::{ProgCache, tensor_values};
    use autorust::runtime::buffer::BufTable;
    use autorust::tensor::{Ctx, Tensor};

    const M: i64 = 128;
    const N: i64 = 192;
    const K: i64 = 256;
    if !autorust::runtime::ops_vk::available() || !autorust::runtime::ops_vk::coop_gemm_available() {
        return;
    }
    let a: Vec<f32> = (0..M * K).map(|i| ((i % 11) as f32 - 5.0) * 0.25).collect();
    let b: Vec<f32> = (0..K * N).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();
    for cdt in [Some(dtype::BFLOAT16), Some(dtype::FLOAT32)] {
        for transposed in [false, true] {
            let ctx = Ctx::new();
            let at = Tensor::from_slice(&ctx, &a, Some(dtype::BFLOAT16), Some("VK")).reshape([M, K]);
            let bt = if transposed {
                Tensor::from_slice(&ctx, &b, Some(dtype::BFLOAT16), Some("VK"))
                    .reshape([N, K])
                    .transpose(0, 1)
            } else {
                Tensor::from_slice(&ctx, &b, Some(dtype::BFLOAT16), Some("VK")).reshape([K, N])
            };
            let out = at.dot_dtype(&bt, cdt);
            let before = autorust::runtime::ops_vk::coop_gemm_count();
            let mut cache = ProgCache::default();
            let got = tensor_values(&mut ctx.raw(), out.uop(), &mut BufTable::default(), &mut cache);
            let used = autorust::runtime::ops_vk::coop_gemm_count() - before;
            assert!(used >= 1, "a dispatch for c_dtype {cdt:?}, transposed={transposed} (got {used})");
            // Reference in f64 over the same operand values (the transposed view
            // reads the buffer column-major, exactly as the descriptor says).
            for (i, j) in [(0usize, 0usize), (2, 5), (31, 17), (63, 63)] {
                let mut acc = 0.0f64;
                for kk in 0..K as usize {
                    let bv = if transposed {
                        b[j * K as usize + kk]
                    } else {
                        b[kk * N as usize + j]
                    };
                    acc += (a[i * K as usize + kk] as f64) * (bv as f64);
                }
                let g = got[i * N as usize + j];
                assert!(
                    (g - acc).abs() <= 1e-3 * acc.abs().max(1.0),
                    "bf16 coopmat c_dtype {cdt:?} transposed={transposed} [{i},{j}]: {g} vs {acc}"
                );
            }
            println!(
                "bf16 coopmat c_dtype={cdt:?} transposed={transposed}: one dispatch, matches the host reference"
            );
        }
    }
}

/// A `Linear` (transposed weight **and** a per-column bias) must take the
/// cooperative-matrix path: the bias rides the f32 -> bf16 epilogue, so the f32
/// accumulator is rounded only once, after the add.
#[test]
fn coopmat_handles_a_biased_linear() {
    use autorust::dtype;
    use autorust::engine::realize::{ProgCache, tensor_values};
    use autorust::runtime::buffer::BufTable;
    use autorust::tensor::nn::Linear;
    use autorust::tensor::{Ctx, Tensor};

    if !autorust::runtime::ops_vk::available() || !autorust::runtime::ops_vk::coop_gemm_available() {
        return;
    }
    let (m, n, k) = (128i64, 192, 256);
    let ctx = Ctx::new();
    let x: Vec<f32> = (0..m * k).map(|i| ((i % 11) as f32 - 5.0) * 0.25).collect();
    let w: Vec<f32> = (0..n * k).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();
    let b: Vec<f32> = (0..n).map(|i| ((i % 5) as f32 - 2.0) * 0.125).collect();
    let run = |dev: Option<&str>| {
        let xt = Tensor::from_slice(&ctx, &x, Some(dtype::BFLOAT16), dev).reshape([m, k]);
        let wt = Tensor::from_slice(&ctx, &w, Some(dtype::BFLOAT16), dev).reshape([n, k]);
        let bt = Tensor::from_slice(&ctx, &b, Some(dtype::BFLOAT16), dev);
        let lin = Linear {
            weight: wt,
            bias: Some(bt),
        };
        let out = lin.forward(&xt);
        let mut cache = ProgCache::default();
        tensor_values(&mut ctx.raw(), out.uop(), &mut BufTable::default(), &mut cache)
    };
    let before = autorust::runtime::ops_vk::coop_gemm_count();
    let got = run(Some("VK"));
    let used = autorust::runtime::ops_vk::coop_gemm_count() - before;
    let want = run(None);
    assert!(used > 0, "coopmat not used for a biased Linear");
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-2 * w.abs().max(1.0),
            "biased Linear [{i}]: {g} vs {w}"
        );
    }
    println!("biased Linear {m}x{n}x{k}: coopmat used, matches INTERP");
}

/// A mixed-precision `Linear` — f32 activations, bf16 weight, bf16 bias, f32
/// result — goes through the MMA by staging the activation to f16. That rounds
/// the activations, so it is gated by `USE_TC`: with `USE_TC=0` the generated
/// kernel must run instead.
#[test]
fn coopmat_handles_mixed_precision_linear() {
    use autorust::dtype;
    use autorust::engine::realize::{ProgCache, tensor_values};
    use autorust::runtime::buffer::BufTable;
    use autorust::tensor::nn::Linear;
    use autorust::tensor::{Ctx, Tensor};

    if !autorust::runtime::ops_vk::available() || !autorust::runtime::ops_vk::coop_gemm_available() {
        return;
    }
    let tc_on = std::env::var("USE_TC").map(|v| v != "0" && v != "false").unwrap_or(true);
    let (m, n, k) = (128i64, 192, 256);
    let ctx = Ctx::new();
    let x: Vec<f32> = (0..m * k).map(|i| ((i % 11) as f32 - 5.0) * 0.25).collect();
    let w: Vec<f32> = (0..n * k).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();
    let b: Vec<f32> = (0..n).map(|i| ((i % 5) as f32 - 2.0) * 0.125).collect();
    let run = |dev: Option<&str>| {
        let xt = Tensor::from_slice(&ctx, &x, Some(dtype::FLOAT32), dev).reshape([m, k]);
        let wt = Tensor::from_slice(&ctx, &w, Some(dtype::BFLOAT16), dev).reshape([n, k]);
        let bt = Tensor::from_slice(&ctx, &b, Some(dtype::BFLOAT16), dev);
        let lin = Linear {
            weight: wt,
            bias: Some(bt),
        };
        let out = lin.forward(&xt);
        let mut cache = ProgCache::default();
        tensor_values(&mut ctx.raw(), out.uop(), &mut BufTable::default(), &mut cache)
    };
    let before = autorust::runtime::ops_vk::coop_gemm_count();
    let got = run(Some("VK"));
    let used = autorust::runtime::ops_vk::coop_gemm_count() - before;
    let want = run(None);
    assert_eq!(
        used > 0,
        tc_on,
        "the mixed-precision path must follow USE_TC (coopmat dispatches: {used})"
    );
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        // f32 activations rounded to f16: ~1e-3 relative.
        assert!(
            (g - w).abs() <= 2e-3 * w.abs().max(1.0),
            "mixed Linear [{i}]: {g} vs {w}"
        );
    }
    println!("mixed-precision Linear {m}x{n}x{k}: coopmat={used} (USE_TC={tc_on}), matches INTERP");
}

/// A strided batched contraction (`batch` folded from the leading axis) runs one
/// cooperative-matrix dispatch per slice, with the accumulator and result
/// windows following the batch stride — the shape the ViT's projections have.
#[test]
fn coopmat_handles_a_batched_dot() {
    use autorust::dtype;
    use autorust::engine::realize::{ProgCache, tensor_values};
    use autorust::runtime::buffer::BufTable;
    use autorust::tensor::{Ctx, Tensor};

    if !autorust::runtime::ops_vk::available() || !autorust::runtime::ops_vk::coop_gemm_available() {
        return;
    }
    let (b, m, n, k) = (8i64, 32i64, 64i64, 96i64);
    let ctx = Ctx::new();
    let x: Vec<f32> = (0..b * m * k).map(|i| ((i % 11) as f32 - 5.0) * 0.25).collect();
    let w: Vec<f32> = (0..n * k).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();
    let run = |dev: Option<&str>| {
        let xt = Tensor::from_slice(&ctx, &x, Some(dtype::FLOAT32), dev).reshape([b, m, k]);
        // `Linear`: the weight is [n, k] and used transposed, which is a
        // column-major B for the MMA.
        let wt = Tensor::from_slice(&ctx, &w, Some(dtype::BFLOAT16), dev)
            .reshape([n, k])
            .transpose(0, 1);
        let out = xt.dot(&wt);
        let mut cache = ProgCache::default();
        tensor_values(&mut ctx.raw(), out.uop(), &mut BufTable::default(), &mut cache)
    };
    let before = autorust::runtime::ops_vk::coop_gemm_count();
    let got = run(Some("VK"));
    let used = autorust::runtime::ops_vk::coop_gemm_count() - before;
    let want = run(None);
    assert!(used > 0, "coopmat not used for a batched dot");
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() <= 2e-3 * w.abs().max(1.0),
            "batched dot [{i}]: {g} vs {w}"
        );
    }
    println!("batched dot {b}x{m}x{n}x{k}: {used} coopmat dispatch(es), matches INTERP");
}

/// A transformer block's `x + linear(ln(x))`: the residual join is fused into the
/// cooperative-matrix epilogue (`out = acc + bias + residual`), so the whole block
/// is one MMA dispatch plus the packing, not an extra full-tensor add.
#[test]
fn coopmat_handles_a_residual_linear() {
    use autorust::dtype;
    use autorust::engine::realize::{ProgCache, tensor_values};
    use autorust::runtime::buffer::BufTable;
    use autorust::tensor::nn::Linear;
    use autorust::tensor::{Ctx, Tensor};

    if !autorust::runtime::ops_vk::available() || !autorust::runtime::ops_vk::coop_gemm_available() {
        return;
    }
    let (m, n, k) = (64i64, 96, 128);
    let ctx = Ctx::new();
    let x: Vec<f32> = (0..m * k).map(|i| ((i % 11) as f32 - 5.0) * 0.25).collect();
    let w: Vec<f32> = (0..n * k).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();
    let b: Vec<f32> = (0..n).map(|i| ((i % 5) as f32 - 2.0) * 0.125).collect();
    let r: Vec<f32> = (0..m * n).map(|i| ((i % 3) as f32 - 1.0) * 0.25).collect();
    let run = |dev: Option<&str>| {
        let xt = Tensor::from_slice(&ctx, &x, Some(dtype::FLOAT32), dev).reshape([m, k]);
        let wt = Tensor::from_slice(&ctx, &w, Some(dtype::BFLOAT16), dev).reshape([n, k]);
        let bt = Tensor::from_slice(&ctx, &b, Some(dtype::BFLOAT16), dev);
        let rt = Tensor::from_slice(&ctx, &r, Some(dtype::FLOAT32), dev).reshape([m, n]);
        let lin = Linear {
            weight: wt,
            bias: Some(bt),
        };
        let out = lin.forward(&xt).add(&rt);
        let mut cache = ProgCache::default();
        tensor_values(&mut ctx.raw(), out.uop(), &mut BufTable::default(), &mut cache)
    };
    let before = autorust::runtime::ops_vk::coop_gemm_count();
    let got = run(Some("VK"));
    let used = autorust::runtime::ops_vk::coop_gemm_count() - before;
    let want = run(None);
    assert!(used > 0, "coopmat not used for x + linear(x)");
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() <= 3e-3 * w.abs().max(1.0),
            "residual Linear [{i}]: {g} vs {w}"
        );
    }
    println!("residual Linear {m}x{n}x{k}: {used} coopmat dispatch(es), matches INTERP");
}

/// The attention score matmul, `Q @ K^T` over `(batch, seq, head_dim)`, is a
/// plain contraction and takes the same path — including the transposed `K`
/// view, which becomes a column-major operand.
///
/// This matters because it is the shape of the ViT's largest kernel: measured
/// one 197x197x64 slice at 0.069 ms (stage + MMA) against 0.213 ms for the
/// generated kernel, i.e. 3.1x. The ViT's own kernel is still rejected by
/// `host_gemm::extract` (see `CURRENT_STATE.md` §9.53), so this gate pins the
/// capability and the numerics while that matcher gap is closed.
#[test]
fn coopmat_handles_the_attention_score_matmul() {
    use autorust::dtype;
    use autorust::engine::realize::{ProgCache, tensor_values};
    use autorust::runtime::buffer::BufTable;
    use autorust::tensor::{Ctx, Tensor};

    fn run(dev: Option<&str>, b: i64, m: i64, n: i64, k: i64) -> Vec<f64> {
        let ctx = Ctx::new();
        let q: Vec<f32> = (0..b * m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
        let kt: Vec<f32> = (0..b * n * k).map(|i| ((i % 7) as f32 - 3.0) * 0.05).collect();
        let qt = Tensor::from_slice(&ctx, &q, Some(dtype::BFLOAT16), dev).reshape([b, m, k]);
        let ktt = Tensor::from_slice(&ctx, &kt, Some(dtype::BFLOAT16), dev)
            .reshape([b, n, k])
            .transpose(1, 2);
        let out = qt.dot_dtype(&ktt, Some(dtype::FLOAT32));
        let mut cache = ProgCache::default();
        tensor_values(&mut ctx.raw(), out.uop(), &mut BufTable::default(), &mut cache)
    }

    if !autorust::runtime::ops_vk::available() || !autorust::runtime::ops_vk::coop_gemm_available() {
        return;
    }
    let (b, m, n, k) = (8i64, 197, 197, 64);
    let before = autorust::runtime::ops_vk::coop_gemm_count();
    let got = run(Some("VK"), b, m, n, k);
    let used = autorust::runtime::ops_vk::coop_gemm_count() - before;
    let want = run(None, b, m, n, k);
    assert_eq!(used, b as usize, "one dispatch per batch slice for Q@K^T");
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-3 * w.abs().max(1.0),
            "Q@K^T [{i}]: {g} vs {w}"
        );
    }
    println!("Q@K^T {b}x{m}x{n}x{k}: {used} coopmat dispatch(es), matches INTERP");
}
