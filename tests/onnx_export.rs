//! ONNX exporter E1/E2 gates: round-trip through the autorust importer and
//! write assets for the Python onnxruntime check.

use autorust::dtype;
use autorust::onnx::export::{
    export_fn, export_module, ExportOptions, GraphBuilder, InputSpec, OutputSpec,
};
use autorust::onnx::OnnxModel;
use autorust::tensor::nn::Module;
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
    // The `dot` lowering must be recognized as one `MatMul`, not exported as
    // the expanded broadcast-multiply/reduce graph.
    let raw = std::fs::read(&path).expect("read block model");
    let proto = autorust::onnx::pb::parse_model(&raw).expect("parse block model");
    assert!(
        proto.graph.nodes.iter().any(|n| n.op_type == "MatMul"),
        "block export has no MatMul node: {:?}",
        proto
            .graph
            .nodes
            .iter()
            .map(|n| n.op_type.as_str())
            .collect::<Vec<_>>()
    );
    let mut m = OnnxModel::load_with_device(&path, None).expect("load block");
    let xx = Tensor::from_slice(m.ctx(), &xv, Some(dtype::FLOAT32), None).reshape([2, 4]);
    let mut out = m.run(&[("x", xx)]).expect("run block");
    let got = out.remove("y").unwrap().to_list::<f32>(m.session_mut());
    close(&got, &want, 1e-5);
}
