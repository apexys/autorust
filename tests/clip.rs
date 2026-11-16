//! End-to-end CLIP T1 acceptance gate.
//!
//! Downloads are not checked in. Put `text_model.onnx`, `vision_model.onnx`,
//! `text_in.bin`, `text_out_ort.bin`, `vision_in.bin`, `vision_out_ort.bin`
//! in `CLIP_ASSET_DIR` (default `/tmp/opencode/clip`) and run:
//!
//! ```text
//! cargo test --release --test clip -- --ignored --nocapture
//! ```
//!
//! The `*_in.bin` / `*_out_ort.bin` files are the deterministic inputs and the
//! onnxruntime CPU reference outputs. They were produced from the
//! `Xenova/clip-vit-base-patch32` fp32 export (`text_model.onnx`,
//! `vision_model.onnx`, opset 14/11) with `onnxruntime` 1.32 and
//! `ORT_DISABLE_ALL` graph optimization.
#![cfg(all(
    feature = "backend_cuda",
    any(target_os = "linux", target_os = "windows")
))]

use autorust::dtype;
use autorust::onnx::OnnxModel;
use autorust::tensor::Tensor;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn dir() -> PathBuf {
    PathBuf::from(std::env::var("CLIP_ASSET_DIR").unwrap_or_else(|_| "/tmp/opencode/clip".into()))
}

fn read(dir: &Path, name: &str) -> Vec<u8> {
    let p = dir.join(name);
    std::fs::read(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn compare(name: &str, got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len(), "{name} length");
    let mut max_abs = 0.0f64;
    let mut dot = 0.0f64;
    let mut ng = 0.0f64;
    let mut nw = 0.0f64;
    for (a, b) in got.iter().zip(want) {
        let a = *a as f64;
        let b = *b as f64;
        max_abs = max_abs.max((a - b).abs());
        dot += a * b;
        ng += a * a;
        nw += b * b;
    }
    let cos = dot / (ng.sqrt() * nw.sqrt());
    eprintln!(
        "{name}: len {}, max|diff| {max_abs:.6}, cosine {cos:.8}",
        got.len()
    );
    max_abs
}

#[allow(clippy::too_many_arguments)]
fn run(
    dir: &Path,
    model: &str,
    input: &str,
    in_file: &str,
    in_dtype: autorust::dtype::DTypeId,
    in_shape: &[i64],
    out_name: &str,
    out_file: &str,
) -> (Vec<f32>, f64) {
    let mut m = OnnxModel::load_with_device(dir.join(model), Some("CUDA"))
        .unwrap_or_else(|e| panic!("load {model}: {e}"));
    let x = Tensor::from_bytes(m.ctx(), &read(dir, in_file), in_dtype, None).reshape(in_shape);
    let t = Instant::now();
    let mut out = m.run(&[(input, x.clone())]).expect("first run");
    let first = t.elapsed().as_secs_f64() * 1e3;
    let got: Vec<f32> = out.remove(out_name).unwrap().to_list(m.session_mut());

    let t = Instant::now();
    let mut out = m.run(&[(input, x)]).expect("replay");
    let replay = t.elapsed().as_secs_f64() * 1e3;
    let again: Vec<f32> = out.remove(out_name).unwrap().to_list(m.session_mut());
    let want = f32s(&read(dir, out_file));
    let max_abs = compare(model, &got, &want);
    let drift = compare("replay", &again, &got);
    eprintln!("{model}: first {first:.1} ms, replay {replay:.1} ms, replay drift {drift:.3e}");
    assert!(drift < 1e-6, "{model} replay drift {drift}");
    (got, max_abs)
}

#[test]
#[ignore = "needs the downloaded CLIP assets + onnxruntime reference files"]
fn clip_text_and_vision_match_onnxruntime() {
    if !autorust::runtime::ops_cuda::available() {
        eprintln!("skipping: CUDA is not available");
        return;
    }
    let d = dir();
    let (_, text_diff) = run(
        &d,
        "text_model.onnx",
        "input_ids",
        "text_in.bin",
        dtype::INT64,
        &[1, 77],
        "text_embeds",
        "text_out_ort.bin",
    );
    let (_, vision_diff) = run(
        &d,
        "vision_model.onnx",
        "pixel_values",
        "vision_in.bin",
        dtype::FLOAT32,
        &[1, 3, 224, 224],
        "image_embeds",
        "vision_out_ort.bin",
    );
    assert!(text_diff < 1e-2, "CLIP text max abs diff {text_diff}");
    assert!(vision_diff < 5e-2, "CLIP vision max abs diff {vision_diff}");
}
