//! State API: streaming `from_read`, safetensors round-trip, and
//! `get_state_dict`/`load_state_dict` (parameters **and** buffers).

use std::collections::HashMap;
use std::io::Cursor;

use autorust::dtype;
use autorust::engine::session::Session;
use autorust::tensor::nn::{BatchNorm, Module};
use autorust::tensor::state::{State, get_state_dict, load_state_dict, safe_load, safe_save};
use autorust::tensor::{Ctx, Tensor};

fn tmp(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "autorust_{}_{}.safetensors",
        std::process::id(),
        name
    ))
}

#[test]
fn from_read_streams_exact_bytes() {
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let raw: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let mut cur = Cursor::new(raw);
    let t = Tensor::from_read(&ctx, &mut cur, dtype::FLOAT32, [2, 2], None);
    assert_eq!(t.shape().as_slice().len(), 2);
    assert_eq!(t.to_list::<f32>(&mut sess), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn safetensors_roundtrip_and_subset() {
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let mut st = State::new();
    st.insert(
        "model.a",
        Tensor::from_slice(&ctx, &[1.0f32, 2.0, 3.0], None, None).reshape([3]),
    );
    st.insert(
        "model.b",
        Tensor::from_slice(&ctx, &[7i16, -2, 9], None, None).reshape([1, 3]),
    );
    st.insert(
        "model.flag",
        Tensor::from_slice(&ctx, &[true, false, true], None, None),
    );
    let mut md = HashMap::new();
    md.insert("format".to_string(), "autorust".to_string());

    let path = tmp("roundtrip");
    safe_save(&st, &path, Some(&md)).expect("save");
    let loaded = safe_load(&path).expect("load");
    let _ = std::fs::remove_file(&path);

    assert_eq!(loaded.len(), 3);
    assert_eq!(
        loaded.metadata.get("format").map(String::as_str),
        Some("autorust")
    );
    assert_eq!(
        loaded.get("model.a").unwrap().to_list::<f32>(&mut sess),
        vec![1.0, 2.0, 3.0]
    );
    assert_eq!(
        loaded.get("model.b").unwrap().to_list::<i16>(&mut sess),
        vec![7, -2, 9]
    );
    assert_eq!(
        loaded.get("model.flag").unwrap().to_list::<bool>(&mut sess),
        vec![true, false, true]
    );

    let sub = loaded.subset("model.");
    assert_eq!(sub.len(), 3);
    assert!(sub.get("a").is_some());
    assert!(sub.get("model.a").is_none());
}

#[test]
fn state_dict_includes_buffers_and_loads_across_contexts() {
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let bn = BatchNorm::new(&ctx, 3, 1e-5, true, true, 0.1).into_params();

    // Every parameter and running buffer is in the state dict.
    let sd = get_state_dict(&bn);
    for name in [
        "weight",
        "bias",
        "running_mean",
        "running_var",
        "num_batches_tracked",
    ] {
        assert!(sd.get(name).is_some(), "missing {name}");
    }

    // Build replacement values in a *different* arena: load_state_dict must copy
    // them into the model's arena rather than moving a foreign node.
    let other = Ctx::new();
    let mut repl = State::new();
    repl.insert(
        "weight",
        Tensor::from_slice(&other, &[2.0f32, 3.0, 4.0], None, None),
    );
    repl.insert(
        "bias",
        Tensor::from_slice(&other, &[10.0f32, 20.0, 30.0], None, None),
    );
    repl.insert(
        "running_mean",
        Tensor::from_slice(&other, &[0.5f32, 0.25, -0.5], None, None),
    );
    repl.insert(
        "running_var",
        Tensor::from_slice(&other, &[2.0f32, 4.0, 8.0], None, None),
    );
    repl.insert(
        "num_batches_tracked",
        Tensor::from_slice(&other, &[5i64], Some(dtype::INT64), None),
    );

    let loaded = load_state_dict(&bn, &repl, true).expect("load_state_dict");
    assert_eq!(loaded.len(), 5);
    assert_eq!(
        bn.weight.as_ref().unwrap().to_list::<f32>(&mut sess),
        vec![2.0, 3.0, 4.0]
    );
    assert_eq!(
        bn.bias.as_ref().unwrap().to_list::<f32>(&mut sess),
        vec![10.0, 20.0, 30.0]
    );
    assert_eq!(
        bn.running_mean.as_ref().unwrap().to_list::<f32>(&mut sess),
        vec![0.5, 0.25, -0.5]
    );
    assert_eq!(
        bn.running_var.as_ref().unwrap().to_list::<f32>(&mut sess),
        vec![2.0, 4.0, 8.0]
    );
    assert_eq!(
        bn.num_batches_tracked
            .as_ref()
            .unwrap()
            .to_list::<i64>(&mut sess),
        vec![5]
    );
}

#[test]
fn load_state_dict_strict_reports_missing_and_extra() {
    let ctx = Ctx::new();
    let lin = autorust::tensor::nn::Linear::new(&ctx, 2, 3, true).into_params();

    // Missing bias.
    let mut partial = State::new();
    partial.insert(
        "weight",
        Tensor::from_slice(&ctx, &[0.0f32; 6], None, None).reshape([3, 2]),
    );
    assert!(load_state_dict(&lin, &partial, true).is_err());
    assert!(load_state_dict(&lin, &partial, false).is_ok());

    // Unexpected name.
    let mut extra = get_state_dict(&lin);
    extra.insert("nope", Tensor::from_slice(&ctx, &[1.0f32], None, None));
    assert!(load_state_dict(&lin, &extra, true).is_err());
}
