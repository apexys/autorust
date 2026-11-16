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
