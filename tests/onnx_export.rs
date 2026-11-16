//! ONNX exporter E1/E2 gates: round-trip through the autorust importer and
//! write assets for the Python onnxruntime check.

use autorust::dtype;
use autorust::onnx::OnnxModel;
use autorust::onnx::export::{
    ExportOptions, GraphBuilder, InputSpec, OutputSpec, export_fn, export_module,
};
use autorust::tensor::nn::{Conv2d, Module};
use autorust::tensor::{Ctx, Tensor};
use std::path::PathBuf;

fn out_dir() -> PathBuf {
    let p = PathBuf::from("/tmp/opencode");
    let _ = std::fs::create_dir_all(&p);
    p
}

fn close(got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len());
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert!(
            (a - b).abs() <= tol,
            "[{i}] {a} != {b} (diff {})",
            (a - b).abs()
        );
    }
}

struct Mlp {
    w1: Tensor,
    b1: Tensor,
    w2: Tensor,
    b2: Tensor,
}

impl Mlp {
    fn new(ctx: &Ctx) -> Self {
        let w1 = Tensor::from_slice(
            ctx,
            &[0.1f32, -0.2, 0.3, 0.05, 0.4, -0.1],
            Some(dtype::FLOAT32),
            None,
        )
        .reshape([3, 2]);
        let b1 = Tensor::from_slice(ctx, &[0.01f32, -0.02], Some(dtype::FLOAT32), None);
        let w2 = Tensor::from_slice(ctx, &[0.7f32, -0.5, 0.2, 0.9], Some(dtype::FLOAT32), None)
            .reshape([2, 2]);
        let b2 = Tensor::from_slice(ctx, &[0.1f32, -0.1], Some(dtype::FLOAT32), None);
        Mlp { w1, b1, w2, b2 }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let h = x.matmul(&self.w1).add(&self.b1).relu();
        h.matmul(&self.w2).add(&self.b2)
    }
}

impl Module for Mlp {
    fn named_params(&self) -> Box<dyn Iterator<Item = (String, Tensor)> + '_> {
        Box::new(
            vec![
                ("w1".to_string(), self.w1.clone()),
                ("b1".to_string(), self.b1.clone()),
                ("w2".to_string(), self.w2.clone()),
                ("b2".to_string(), self.b2.clone()),
            ]
            .into_iter(),
        )
    }

    fn into_params(self) -> Self {
        self
    }
}

#[test]
fn graph_builder_e0_round_trip() {
    let path = out_dir().join("onnx_export_e0.onnx");
    let mut g = GraphBuilder::new("e0");
    g.input("x", dtype::FLOAT32, &[2]);
    g.initializer("one", dtype::FLOAT32, &[], 1.0f32.to_le_bytes().to_vec())
        .unwrap();
    g.node("Add", &["x", "one"], &["y"], &[]);
    g.output("y", dtype::FLOAT32, &[2]);
    g.save(&path, &ExportOptions::default()).expect("save e0");

    let mut m = OnnxModel::load_with_device(&path, None).expect("load e0");
    let x = Tensor::from_slice(m.ctx(), &[1.0f32, 2.0], Some(dtype::FLOAT32), None);
    let mut out = m.run(&[("x", x)]).expect("run e0");
    let got = out.remove("y").unwrap().to_list::<f32>(m.session_mut());
    close(&got, &[2.0, 3.0], 1e-6);
}

#[test]
fn export_e1_round_trip_and_assets() {
    let path = out_dir().join("onnx_export_e1.onnx");
    let inputs = [InputSpec::new("x", dtype::FLOAT32, &[2, 3])];
    let outputs = [OutputSpec::new("y")];
    export_fn(
        &path,
        &inputs,
        &outputs,
        |args| {
            let x = &args[0];
            let ctx = x.ctx();
            let w = Tensor::from_slice(
                ctx,
                &[0.2f32, -0.3, 0.4, 0.1, -0.5, 0.6],
                Some(dtype::FLOAT32),
                None,
            )
            .reshape([3, 2]);
            let b = Tensor::from_slice(ctx, &[0.05f32, -0.05], Some(dtype::FLOAT32), None);
            vec![x.matmul(&w).add(&b).relu()]
        },
        &ExportOptions::default(),
    )
    .expect("export e1");

    let xv = [1.0f32, 2.0, 3.0, -1.0, 0.5, 2.0];
    let ctx = Ctx::new();
    let x = Tensor::from_slice(&ctx, &xv, Some(dtype::FLOAT32), None).reshape([2, 3]);
    let w = Tensor::from_slice(
        &ctx,
        &[0.2f32, -0.3, 0.4, 0.1, -0.5, 0.6],
        Some(dtype::FLOAT32),
        None,
    )
    .reshape([3, 2]);
    let b = Tensor::from_slice(&ctx, &[0.05f32, -0.05], Some(dtype::FLOAT32), None);
    let want = x.matmul(&w).add(&b).relu();
    let mut sess = autorust::engine::session::Session::new();
    let want = want.to_list::<f32>(&mut sess);

    let mut m = OnnxModel::load_with_device(&path, None).expect("load e1");
    let xx = Tensor::from_slice(m.ctx(), &xv, Some(dtype::FLOAT32), None).reshape([2, 3]);
    let mut out = m.run(&[("x", xx)]).expect("run e1");
    let got = out.remove("y").unwrap().to_list::<f32>(m.session_mut());
    close(&got, &want, 1e-5);
}

#[test]
fn export_e2_module_round_trip_and_python_assets() {
    let ctx = Ctx::new();
    let model = Mlp::new(&ctx);
    let xv = [1.0f32, 2.0, 3.0, -1.0, 0.5, 2.0];
    let x = Tensor::from_slice(&ctx, &xv, Some(dtype::FLOAT32), None).reshape([2, 3]);
    let mut sess = autorust::engine::session::Session::new();
    let want = model.forward(&x).to_list::<f32>(&mut sess);

    let path = out_dir().join("onnx_export_e2.onnx");
    let inputs = [InputSpec::new("x", dtype::FLOAT32, &[2, 3])];
    let outputs = [OutputSpec::new("y")];
    export_module(
        &path,
        &model,
        &inputs,
        &outputs,
        |m, args| vec![m.forward(&args[0])],
        &ExportOptions::default(),
    )
    .expect("export e2");

    let mut m = OnnxModel::load_with_device(&path, None).expect("load e2");
    let xx = Tensor::from_slice(m.ctx(), &xv, Some(dtype::FLOAT32), None).reshape([2, 3]);
    let mut out = m.run(&[("x", xx)]).expect("run e2");
    let got = out.remove("y").unwrap().to_list::<f32>(m.session_mut());
    close(&got, &want, 1e-5);

    // Assets for the dev-only Python onnxruntime check.
    let mut xb = Vec::new();
    for v in xv {
        xb.extend_from_slice(&v.to_le_bytes());
    }
    let mut yb = Vec::new();
    for v in &want {
        yb.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(out_dir().join("onnx_export_e2_x.bin"), xb).unwrap();
    std::fs::write(out_dir().join("onnx_export_e2_y.bin"), yb).unwrap();
}

#[test]
fn export_e3_dynamic_axes_and_external_data() {
    // Dynamic batch axis on a graph without shape-baking Reshape.
    let dyn_path = out_dir().join("onnx_export_e3_dynamic.onnx");
    let inputs = [InputSpec::new("x", dtype::FLOAT32, &[1, 2]).with_dynamic_dim(0, "batch")];
    let outputs = [OutputSpec::new("y").with_dynamic_dim(0, "batch")];
    export_fn(
        &dyn_path,
        &inputs,
        &outputs,
        |args| vec![args[0].mul(2.0).add(1.0)],
        &ExportOptions::default(),
    )
    .expect("export dynamic");
    let mut m = OnnxModel::load_with_device(&dyn_path, None).expect("load dynamic");
    for (batch, vals) in [
        (2usize, vec![1.0f32, 2.0, 3.0, 4.0]),
        (3usize, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]),
    ] {
        let x = Tensor::from_slice(m.ctx(), &vals, Some(dtype::FLOAT32), None)
            .reshape([batch as i64, 2]);
        let mut out = m.run(&[("x", x)]).expect("run dynamic");
        let got = out.remove("y").unwrap().to_list::<f32>(m.session_mut());
        let want: Vec<f32> = vals.iter().map(|v| 2.0 * v + 1.0).collect();
        close(&got, &want, 1e-6);
    }

    // Force every initializer into a sidecar; the importer must stream it.
    let ctx = Ctx::new();
    let model = Mlp::new(&ctx);
    let xv = [1.0f32, 2.0, 3.0, -1.0, 0.5, 2.0];
    let mut sess = autorust::engine::session::Session::new();
    let want = model
        .forward(&Tensor::from_slice(&ctx, &xv, Some(dtype::FLOAT32), None).reshape([2, 3]))
        .to_list::<f32>(&mut sess);
    let ext_path = out_dir().join("onnx_export_e3_external.onnx");
    let opts = ExportOptions {
        external_data: Some(0),
        ..ExportOptions::default()
    };
    export_module(
        &ext_path,
        &model,
        &[InputSpec::new("x", dtype::FLOAT32, &[2, 3])],
        &[OutputSpec::new("y")],
        |m, args| vec![m.forward(&args[0])],
        &opts,
    )
    .expect("export external");
    assert!(out_dir().join("onnx_export_e3_external.onnx.data").exists());
    let mut em = OnnxModel::load_with_device(&ext_path, None).expect("load external");
    let x = Tensor::from_slice(em.ctx(), &xv, Some(dtype::FLOAT32), None).reshape([2, 3]);
    let mut out = em.run(&[("x", x)]).expect("run external");
    let got = out.remove("y").unwrap().to_list::<f32>(em.session_mut());
    close(&got, &want, 1e-5);
}

struct Block {
    ln: autorust::tensor::nn::LayerNorm,
    w: Tensor,
    b: Tensor,
}

impl Block {
    fn new(ctx: &Ctx) -> Self {
        let ln = autorust::tensor::nn::LayerNorm::new(ctx, vec![4], 1e-5, true);
        let w = Tensor::from_slice(
            ctx,
            &[
                0.1f32, 0.0, -0.2, 0.3, 0.0, 0.2, 0.1, -0.1, -0.3, 0.4, 0.0, 0.2, 0.2, -0.1, 0.1,
                0.0,
            ],
            Some(dtype::FLOAT32),
            None,
        )
        .reshape([4, 4]);
        let b = Tensor::from_slice(
            ctx,
            &[0.01f32, -0.02, 0.0, 0.03],
            Some(dtype::FLOAT32),
            None,
        );
        Block { ln, w, b }
    }
    fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.ln.forward(x);
        x.add(h.matmul(&self.w).add(&self.b))
    }
}

impl Module for Block {
    fn named_params(&self) -> Box<dyn Iterator<Item = (String, Tensor)> + '_> {
        let v = vec![
            ("ln_weight".to_string(), self.ln.weight.clone().unwrap()),
            ("ln_bias".to_string(), self.ln.bias.clone().unwrap()),
            ("w".to_string(), self.w.clone()),
            ("b".to_string(), self.b.clone()),
        ];
        Box::new(v.into_iter())
    }
    fn into_params(self) -> Self {
        self
    }
}

#[test]
fn export_e3_transformer_style_block() {
    let ctx = Ctx::new();
    let model = Block::new(&ctx);
    let xv = [1.0f32, 2.0, 3.0, 4.0, -1.0, 0.5, 0.0, 2.0];
    let x = Tensor::from_slice(&ctx, &xv, Some(dtype::FLOAT32), None).reshape([2, 4]);
    let mut sess = autorust::engine::session::Session::new();
    let want = model.forward(&x).to_list::<f32>(&mut sess);
    let path = out_dir().join("onnx_export_e3_block.onnx");
    export_module(
        &path,
        &model,
        &[InputSpec::new("x", dtype::FLOAT32, &[2, 4])],
        &[OutputSpec::new("y")],
        |m, args| vec![m.forward(&args[0])],
        &ExportOptions::default(),
    )
    .expect("export block");
    // The 2-D `dot` + bias lowering must be recognized as one `Gemm`, not
    // exported as the expanded broadcast-multiply/reduce/add graph.
    let raw = std::fs::read(&path).expect("read block model");
    let proto = autorust::onnx::pb::parse_model(&raw).expect("parse block model");
    let ops: Vec<&str> = proto
        .graph
        .nodes
        .iter()
        .map(|n| n.op_type.as_str())
        .collect();
    let gemm = proto
        .graph
        .nodes
        .iter()
        .find(|n| n.op_type == "Gemm")
        .unwrap_or_else(|| panic!("block export has no folded Gemm: {ops:?}"));
    assert_eq!(
        gemm.inputs.len(),
        3,
        "Gemm must fold the bias: {:?}",
        gemm.inputs
    );
    assert!(
        !ops.contains(&"MatMul"),
        "un-folded MatMul left in block: {ops:?}"
    );
    let mut m = OnnxModel::load_with_device(&path, None).expect("load block");
    let xx = Tensor::from_slice(m.ctx(), &xv, Some(dtype::FLOAT32), None).reshape([2, 4]);
    let mut out = m.run(&[("x", xx)]).expect("run block");
    let got = out.remove("y").unwrap().to_list::<f32>(m.session_mut());
    close(&got, &want, 1e-5);
}

struct Cnn {
    conv: Conv2d,
    lin: Tensor,
    bias: Tensor,
}

impl Cnn {
    fn new(ctx: &Ctx) -> Self {
        let conv = Conv2d::new(ctx, 1, 2, &[2, 2], 1, &[(1, 1), (1, 1)], 1, 1, true);
        // 2 pools of 2x2 with one pixel of padding over the 9x9 conv output
        // -> 5x5 each; the padded max uses a bool Pad + Where with dtype_min.
        let lin = Tensor::from_slice(
            ctx,
            &(0..(2 * 5 * 5 * 3))
                .map(|i| ((i % 11) as f32 - 5.0) * 0.03)
                .collect::<Vec<_>>(),
            Some(dtype::FLOAT32),
            None,
        )
        .reshape([50, 3]);
        let bias = Tensor::from_slice(ctx, &[0.1f32, -0.2, 0.05], Some(dtype::FLOAT32), None);
        Cnn { conv, lin, bias }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.conv.forward(x).relu();
        let (p, _) = h.max_pool2d(&[2, 2], Some(&[2, 2]), &[1, 1], &[1, 1], false, false);
        p.reshape([2, 50]).matmul(&self.lin).add(&self.bias)
    }
}

impl Module for Cnn {
    fn named_params(&self) -> Box<dyn Iterator<Item = (String, Tensor)> + '_> {
        let mut v = vec![
            ("conv_weight".to_string(), self.conv.weight.clone()),
            ("lin".to_string(), self.lin.clone()),
            ("bias".to_string(), self.bias.clone()),
        ];
        if let Some(b) = &self.conv.bias {
            v.push(("conv_bias".to_string(), b.clone()));
        }
        Box::new(v.into_iter())
    }

    fn into_params(self) -> Self {
        self
    }
}

#[test]
fn export_e3b_cnn_block() {
    let ctx = Ctx::new();
    let model = Cnn::new(&ctx);
    let xv: Vec<f32> = (0..128).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
    let x = Tensor::from_slice(&ctx, &xv, Some(dtype::FLOAT32), None).reshape([2, 1, 8, 8]);
    let mut sess = autorust::engine::session::Session::new();
    let want = model.forward(&x).to_list::<f32>(&mut sess);
    let path = out_dir().join("onnx_export_e3b_cnn.onnx");
    export_module(
        &path,
        &model,
        &[InputSpec::new("x", dtype::FLOAT32, &[2, 1, 8, 8])],
        &[OutputSpec::new("y")],
        |m, args| vec![m.forward(&args[0])],
        &ExportOptions::default(),
    )
    .expect("export cnn");
    let raw = std::fs::read(&path).expect("read cnn model");
    let proto = autorust::onnx::pb::parse_model(&raw).expect("parse cnn model");
    let ops: Vec<&str> = proto
        .graph
        .nodes
        .iter()
        .map(|n| n.op_type.as_str())
        .collect();
    assert!(ops.contains(&"Pad"), "CNN export has no Pad: {ops:?}");
    assert!(ops.contains(&"Slice"), "CNN export has no Slice: {ops:?}");
    let conv = proto
        .graph
        .nodes
        .iter()
        .find(|n| n.op_type == "Conv")
        .unwrap_or_else(|| panic!("CNN export has no folded Conv: {ops:?}"));
    assert_eq!(
        conv.inputs.len(),
        3,
        "Conv bias must be folded: {:?}",
        conv.inputs
    );
    assert_eq!(conv.attr_i64("group", 1), 1);
    assert_eq!(
        conv.attrs.get("strides").and_then(|a| a.as_i64s()),
        Some(&[1, 1][..])
    );
    assert_eq!(
        conv.attrs.get("dilations").and_then(|a| a.as_i64s()),
        Some(&[1, 1][..])
    );
    assert_eq!(
        conv.attrs.get("pads").and_then(|a| a.as_i64s()),
        Some(&[1, 1, 1, 1][..])
    );
    let mut m = OnnxModel::load_with_device(&path, None).expect("load cnn");
    let xx = Tensor::from_slice(m.ctx(), &xv, Some(dtype::FLOAT32), None).reshape([2, 1, 8, 8]);
    let mut out = m.run(&[("x", xx)]).expect("run cnn");
    let got = out.remove("y").unwrap().to_list::<f32>(m.session_mut());
    close(&got, &want, 1e-4);

    // Assets for the dev-only Python onnxruntime check.
    let mut xb = Vec::new();
    for v in &xv {
        xb.extend_from_slice(&v.to_le_bytes());
    }
    let mut yb = Vec::new();
    for v in &want {
        yb.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(out_dir().join("onnx_export_e3b_cnn_x.bin"), xb).unwrap();
    std::fs::write(out_dir().join("onnx_export_e3b_cnn_y.bin"), yb).unwrap();
}

#[test]
fn export_e3b_conv_groups_stride() {
    let weights: Vec<f32> = (0..32).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
    let xv: Vec<f32> = (0..(2 * 4 * 7 * 7))
        .map(|i| ((i % 11) as f32 - 5.0) * 0.05)
        .collect();
    let path = out_dir().join("onnx_export_e3b_conv_groups.onnx");
    export_fn(
        &path,
        &[InputSpec::new("x", dtype::FLOAT32, &[2, 4, 7, 7])],
        &[OutputSpec::new("y")],
        |args| {
            let ctx = args[0].ctx();
            let w =
                Tensor::from_slice(ctx, &weights, Some(dtype::FLOAT32), None).reshape([4, 2, 2, 2]);
            vec![args[0].conv2d(&w, None, 2, 2, 2, &[0, 0], None)]
        },
        &ExportOptions::default(),
    )
    .expect("export grouped conv");

    let raw = std::fs::read(&path).expect("read grouped conv");
    let proto = autorust::onnx::pb::parse_model(&raw).expect("parse grouped conv");
    let conv = proto
        .graph
        .nodes
        .iter()
        .find(|n| n.op_type == "Conv")
        .expect("folded grouped Conv");
    assert_eq!(conv.inputs.len(), 2, "no-bias Conv must have two inputs");
    assert_eq!(conv.attr_i64("group", 1), 2);
    assert_eq!(
        conv.attrs.get("strides").and_then(|a| a.as_i64s()),
        Some(&[2, 2][..])
    );
    assert_eq!(
        conv.attrs.get("dilations").and_then(|a| a.as_i64s()),
        Some(&[2, 2][..])
    );
    assert_eq!(
        conv.attrs.get("pads").and_then(|a| a.as_i64s()),
        Some(&[0, 0, 0, 0][..])
    );

    let ctx = Ctx::new();
    let x = Tensor::from_slice(&ctx, &xv, Some(dtype::FLOAT32), None).reshape([2, 4, 7, 7]);
    let w = Tensor::from_slice(&ctx, &weights, Some(dtype::FLOAT32), None).reshape([4, 2, 2, 2]);
    let mut sess = autorust::engine::session::Session::new();
    let want = x
        .conv2d(&w, None, 2, 2, 2, &[0, 0], None)
        .to_list::<f32>(&mut sess);

    let mut m = OnnxModel::load_with_device(&path, None).expect("load grouped conv");
    let xx = Tensor::from_slice(m.ctx(), &xv, Some(dtype::FLOAT32), None).reshape([2, 4, 7, 7]);
    let mut out = m.run(&[("x", xx)]).expect("run grouped conv");
    let got = out.remove("y").unwrap().to_list::<f32>(m.session_mut());
    close(&got, &want, 1e-4);

    let mut xb = Vec::new();
    for v in &xv {
        xb.extend_from_slice(&v.to_le_bytes());
    }
    let mut yb = Vec::new();
    for v in &want {
        yb.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(out_dir().join("onnx_export_e3b_conv_groups_x.bin"), xb).unwrap();
    std::fs::write(out_dir().join("onnx_export_e3b_conv_groups_y.bin"), yb).unwrap();
}

fn model_ops(path: &std::path::Path) -> Vec<String> {
    let raw = std::fs::read(path).expect("read model");
    let proto = autorust::onnx::pb::parse_model(&raw).expect("parse model");
    proto
        .graph
        .nodes
        .iter()
        .map(|n| n.op_type.clone())
        .collect()
}

#[test]
fn export_e3b_gemm_fallback_cases() {
    // ONNX Gemm is 2-D only: a batched matmul keeps the expanded form.
    let path3 = out_dir().join("onnx_export_e3b_gemm_3d.onnx");
    export_fn(
        &path3,
        &[InputSpec::new("x", dtype::FLOAT32, &[2, 3, 4])],
        &[OutputSpec::new("y")],
        |args| {
            let ctx = args[0].ctx();
            let w = Tensor::from_slice(
                ctx,
                &(0..20).map(|i| i as f32 * 0.01).collect::<Vec<_>>(),
                Some(dtype::FLOAT32),
                None,
            )
            .reshape([4, 5]);
            let b = Tensor::from_slice(ctx, &[0.1f32; 5], Some(dtype::FLOAT32), None);
            vec![args[0].matmul(&w).add(&b)]
        },
        &ExportOptions::default(),
    )
    .expect("export 3d matmul");
    let ops = model_ops(&path3);
    assert!(ops.contains(&"MatMul".to_string()), "3-D: {ops:?}");
    assert!(
        !ops.contains(&"Gemm".to_string()),
        "3-D folded to Gemm: {ops:?}"
    );

    // A shared matmul cannot be consumed by the fold.
    let path_shared = out_dir().join("onnx_export_e3b_gemm_shared.onnx");
    export_fn(
        &path_shared,
        &[InputSpec::new("x", dtype::FLOAT32, &[2, 3])],
        &[OutputSpec::new("y")],
        |args| {
            let ctx = args[0].ctx();
            let w = Tensor::from_slice(
                ctx,
                &[0.2f32, -0.3, 0.4, 0.1, -0.5, 0.6],
                Some(dtype::FLOAT32),
                None,
            )
            .reshape([3, 2]);
            let b = Tensor::from_slice(ctx, &[0.05f32, -0.05], Some(dtype::FLOAT32), None);
            let y = args[0].matmul(&w);
            vec![y.add(&b).add(&y)]
        },
        &ExportOptions::default(),
    )
    .expect("export shared matmul");
    let ops = model_ops(&path_shared);
    assert!(ops.contains(&"MatMul".to_string()), "shared: {ops:?}");
    assert!(
        !ops.contains(&"Gemm".to_string()),
        "shared matmul folded: {ops:?}"
    );

    // A matmul that is itself an output must stay a MatMul.
    let path_out = out_dir().join("onnx_export_e3b_gemm_output.onnx");
    export_fn(
        &path_out,
        &[InputSpec::new("x", dtype::FLOAT32, &[2, 3])],
        &[OutputSpec::new("y"), OutputSpec::new("z")],
        |args| {
            let ctx = args[0].ctx();
            let w = Tensor::from_slice(
                ctx,
                &[0.2f32, -0.3, 0.4, 0.1, -0.5, 0.6],
                Some(dtype::FLOAT32),
                None,
            )
            .reshape([3, 2]);
            let b = Tensor::from_slice(ctx, &[0.05f32, -0.05], Some(dtype::FLOAT32), None);
            let y = args[0].matmul(&w);
            vec![y.contiguous(), y.add(&b)]
        },
        &ExportOptions::default(),
    )
    .expect("export matmul output");
    let ops = model_ops(&path_out);
    assert!(ops.contains(&"MatMul".to_string()), "output: {ops:?}");
    assert!(
        !ops.contains(&"Gemm".to_string()),
        "output matmul folded: {ops:?}"
    );
}
