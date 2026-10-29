//! ONNX importer tests: parser, external data, and numerical op dispatch.

#[path = "support/onnx_builder.rs"]
mod b;

use autorust::dtype;
use autorust::onnx::{OnnxError, OnnxModel};
use autorust::tensor::{Ctx, Tensor};
use b::{Attr, Dim};

fn f32_model(bytes: &[u8], inputs: &[(&str, Vec<f32>, Vec<i64>)], output: &str) -> Vec<f32> {
    let mut m = OnnxModel::from_bytes(bytes, None).expect("parse");
    let tensors: Vec<(&str, Tensor)> = inputs
        .iter()
        .map(|(name, vals, shape)| {
            (
                *name,
                Tensor::from_slice(m.ctx(), vals, Some(dtype::FLOAT32), None)
                    .reshape(shape.clone()),
            )
        })
        .collect();
    let mut out = m.run(&tensors).expect("run");
    out.remove(output)
        .expect("output")
        .to_list::<f32>(m.session_mut())
}

fn close(got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "length: {got:?} vs {want:?}");
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert!(
            (a - b).abs() <= tol,
            "[{i}] {a} != {b} (got {got:?}, want {want:?})"
        );
    }
}

#[test]
fn add_broadcast_and_relu() {
    let nodes = vec![
        b::node("Add", &["x", "y"], &["sum"], &[]),
        b::node("Relu", &["sum"], &["out"], &[]),
    ];
    let inputs = vec![
        b::value_info("x", b::FLOAT, &[Dim::Val(2), Dim::Val(3)]),
        b::value_info("y", b::FLOAT, &[Dim::Val(3)]),
    ];
    let outputs = vec![b::value_info("out", b::FLOAT, &[Dim::Val(2), Dim::Val(3)])];
    let bytes = b::model(&b::graph(&nodes, &[], &inputs, &outputs, "add"), 17);

    let got = f32_model(
        &bytes,
        &[
            ("x", vec![1.0, -2.0, 3.0, 4.0, -5.0, 6.0], vec![2, 3]),
            ("y", vec![10.0, 20.0, 30.0], vec![3]),
        ],
        "out",
    );
    close(&got, &[11.0, 18.0, 33.0, 14.0, 15.0, 36.0], 1e-6);
}

#[test]
fn matmul_gemm_and_layernorm_match_manual() {
    let w = b::f32_tensor("w", &[4, 2], &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, -1.0]);
    let bias = b::f32_tensor("b", &[2], &[0.5, -0.5]);
    let gamma = b::f32_tensor("gamma", &[2], &[2.0, 3.0]);
    let beta = b::f32_tensor("beta", &[2], &[1.0, -1.0]);
    let nodes = vec![
        b::node("MatMul", &["x", "w"], &["mm"], &[]),
        b::node("Add", &["mm", "b"], &["lin"], &[]),
        b::node(
            "LayerNormalization",
            &["lin", "gamma", "beta"],
            &["out"],
            &[Attr::Int("axis", -1), Attr::Float("epsilon", 1e-5)],
        ),
    ];
    let inputs = vec![b::value_info("x", b::FLOAT, &[Dim::Val(2), Dim::Val(4)])];
    let outputs = vec![b::value_info("out", b::FLOAT, &[Dim::Val(2), Dim::Val(2)])];
    let bytes = b::model(
        &b::graph(&nodes, &[w, bias, gamma, beta], &inputs, &outputs, "ln"),
        17,
    );

    let got = f32_model(
        &bytes,
        &[(
            "x",
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2, 4],
        )],
        "out",
    );
    close(&got, &[3.0, -4.0, 3.0, -4.0], 2e-4);
}

#[test]
fn movement_slice_gather_concat_and_expand() {
    let shape = b::i64_tensor("shape", &[2], &[2, 12]);
    let gather_idx = b::i64_tensor("gidx", &[2], &[0, 11]);
    let extra = b::f32_tensor("extra", &[2, 1], &[100.0, 200.0]);
    let exp_shape = b::i64_tensor("eshape", &[3], &[3, 2, 4]);
    let nodes = vec![
        b::node(
            "Transpose",
            &["x"],
            &["t"],
            &[Attr::Ints("perm", vec![0, 2, 1])],
        ),
        b::node("Reshape", &["t", "shape"], &["r"], &[]),
        b::node("Gather", &["r", "gidx"], &["g"], &[Attr::Int("axis", 1)]),
        b::node("Concat", &["g", "extra"], &["out"], &[Attr::Int("axis", 1)]),
        b::node("Expand", &["a", "eshape"], &["exp"], &[]),
    ];
    let inputs = vec![
        b::value_info("x", b::FLOAT, &[Dim::Val(2), Dim::Val(3), Dim::Val(4)]),
        b::value_info("a", b::FLOAT, &[Dim::Val(1), Dim::Val(2), Dim::Val(1)]),
    ];
    let outputs = vec![
        b::value_info("out", b::FLOAT, &[Dim::Val(2), Dim::Val(3)]),
        b::value_info("exp", b::FLOAT, &[Dim::Val(3), Dim::Val(2), Dim::Val(4)]),
    ];
    let bytes = b::model(
        &b::graph(
            &nodes,
            &[shape, gather_idx, extra, exp_shape],
            &inputs,
            &outputs,
            "mov",
        ),
        17,
    );

    let mut m = OnnxModel::from_bytes(&bytes, None).unwrap();
    let x = Tensor::from_slice(
        m.ctx(),
        &(0..24).map(|i| i as f32).collect::<Vec<_>>(),
        Some(dtype::FLOAT32),
        None,
    )
    .reshape([2, 3, 4]);
    let a =
        Tensor::from_slice(m.ctx(), &[1.0f32, 2.0], Some(dtype::FLOAT32), None).reshape([1, 2, 1]);
    let out = m.run(&[("x", x), ("a", a)]).unwrap();
    close(
        &out["out"].to_list::<f32>(m.session_mut()),
        &[0.0, 11.0, 100.0, 12.0, 23.0, 200.0],
        1e-6,
    );
    let exp = out["exp"].to_list::<f32>(m.session_mut());
    assert_eq!(exp.len(), 24);
    for b in 0..3 {
        for c in 0..2 {
            for h in 0..4 {
                assert_eq!(exp[b * 8 + c * 4 + h], if c == 0 { 1.0 } else { 2.0 });
            }
        }
    }
}

#[test]
fn conv_pool_and_auto_pad() {
    let w = b::f32_tensor("w", &[1, 1, 2, 2], &[1.0; 4]);
    let bias = b::f32_tensor("bias", &[1], &[0.5]);
    let nodes = vec![
        b::node(
            "Conv",
            &["x", "w", "bias"],
            &["conv"],
            &[
                Attr::Int("group", 1),
                Attr::Ints("strides", vec![1, 1]),
                Attr::Ints("pads", vec![0, 0, 0, 0]),
            ],
        ),
        b::node(
            "MaxPool",
            &["conv"],
            &["pool"],
            &[
                Attr::Ints("kernel_shape", vec![2, 2]),
                Attr::Ints("strides", vec![1, 1]),
            ],
        ),
        b::node(
            "AveragePool",
            &["x"],
            &["avg"],
            &[
                Attr::Ints("kernel_shape", vec![2, 2]),
                Attr::Ints("strides", vec![1, 1]),
            ],
        ),
        b::node(
            "Conv",
            &["same", "w"],
            &["same_out"],
            &[
                Attr::Str("auto_pad", "SAME_UPPER"),
                Attr::Ints("strides", vec![2, 2]),
            ],
        ),
    ];
    let inputs = vec![
        b::value_info(
            "x",
            b::FLOAT,
            &[Dim::Val(1), Dim::Val(1), Dim::Val(4), Dim::Val(4)],
        ),
        b::value_info(
            "same",
            b::FLOAT,
            &[Dim::Val(1), Dim::Val(1), Dim::Val(5), Dim::Val(5)],
        ),
    ];
    let outputs = vec![
        b::value_info(
            "conv",
            b::FLOAT,
            &[Dim::Val(1), Dim::Val(1), Dim::Val(3), Dim::Val(3)],
        ),
        b::value_info(
            "pool",
            b::FLOAT,
            &[Dim::Val(1), Dim::Val(1), Dim::Val(2), Dim::Val(2)],
        ),
        b::value_info(
            "avg",
            b::FLOAT,
            &[Dim::Val(1), Dim::Val(1), Dim::Val(3), Dim::Val(3)],
        ),
        b::value_info(
            "same_out",
            b::FLOAT,
            &[Dim::Val(1), Dim::Val(1), Dim::Val(3), Dim::Val(3)],
        ),
    ];
    let bytes = b::model(&b::graph(&nodes, &[w, bias], &inputs, &outputs, "conv"), 17);

    let mut m = OnnxModel::from_bytes(&bytes, None).unwrap();
    let x = Tensor::from_slice(
        m.ctx(),
        &(1..=16).map(|i| i as f32).collect::<Vec<_>>(),
        Some(dtype::FLOAT32),
        None,
    )
    .reshape([1, 1, 4, 4]);
    let same = Tensor::from_slice(m.ctx(), &[1.0f32; 25], Some(dtype::FLOAT32), None)
        .reshape([1, 1, 5, 5]);
    let out = m.run(&[("x", x), ("same", same)]).unwrap();
    close(
        &out["conv"].to_list::<f32>(m.session_mut()),
        &[14.5, 18.5, 22.5, 30.5, 34.5, 38.5, 46.5, 50.5, 54.5],
        1e-5,
    );
    close(
        &out["pool"].to_list::<f32>(m.session_mut()),
        &[34.5, 38.5, 50.5, 54.5],
        1e-5,
    );
    close(
        &out["avg"].to_list::<f32>(m.session_mut()),
        &[3.5, 4.5, 5.5, 7.5, 8.5, 9.5, 11.5, 12.5, 13.5],
        1e-5,
    );
    assert_eq!(out["same_out"].shape().as_slice().len(), 4);
    assert_eq!(
        out["same_out"].shape().as_slice()[2].clone().as_i64(),
        Some(3)
    );
}

#[test]
fn softmax_reduce_and_argmax() {
    let axes = b::i64_tensor("axes", &[1], &[1]);
    let nodes = vec![
        b::node("Softmax", &["x"], &["sm"], &[]),
        b::node(
            "ReduceSum",
            &["x", "axes"],
            &["sum"],
            &[Attr::Int("keepdims", 1)],
        ),
        b::node(
            "ArgMax",
            &["x"],
            &["arg"],
            &[Attr::Int("axis", 1), Attr::Int("keepdims", 0)],
        ),
    ];
    let inputs = vec![b::value_info("x", b::FLOAT, &[Dim::Val(2), Dim::Val(3)])];
    let outputs = vec![
        b::value_info("sm", b::FLOAT, &[Dim::Val(2), Dim::Val(3)]),
        b::value_info("sum", b::FLOAT, &[Dim::Val(2), Dim::Val(1)]),
        b::value_info("arg", b::INT64, &[Dim::Val(2)]),
    ];
    let bytes = b::model(&b::graph(&nodes, &[axes], &inputs, &outputs, "softmax"), 13);

    let mut m = OnnxModel::from_bytes(&bytes, None).unwrap();
    let x = Tensor::from_slice(
        m.ctx(),
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        Some(dtype::FLOAT32),
        None,
    )
    .reshape([2, 3]);
    let out = m.run(&[("x", x)]).unwrap();
    let sm = out["sm"].to_list::<f32>(m.session_mut());
    close(&sm[..3], &[0.09003057, 0.24472847, 0.66524096], 1e-5);
    close(&sm[3..], &[0.09003057, 0.24472847, 0.66524096], 1e-5);
    close(
        &out["sum"].to_list::<f32>(m.session_mut()),
        &[6.0, 15.0],
        1e-5,
    );
    assert_eq!(out["arg"].to_list::<i64>(m.session_mut()), vec![2, 2]);
}

#[test]
fn resize_linear_align_corners() {
    let sizes = b::i64_tensor("sizes", &[4], &[1, 1, 4, 4]);
    let nodes = vec![b::node(
        "Resize",
        &["x", "", "", "sizes"],
        &["out"],
        &[
            Attr::Str("mode", "linear"),
            Attr::Str("coordinate_transformation_mode", "align_corners"),
        ],
    )];
    let inputs = vec![b::value_info(
        "x",
        b::FLOAT,
        &[Dim::Val(1), Dim::Val(1), Dim::Val(2), Dim::Val(2)],
    )];
    let outputs = vec![b::value_info(
        "out",
        b::FLOAT,
        &[Dim::Val(1), Dim::Val(1), Dim::Val(4), Dim::Val(4)],
    )];
    let bytes = b::model(&b::graph(&nodes, &[sizes], &inputs, &outputs, "resize"), 13);

    let got = f32_model(
        &bytes,
        &[("x", vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2])],
        "out",
    );
    close(
        &got,
        &[
            1.0, 1.3333334, 1.6666667, 2.0, 1.6666667, 2.0, 2.3333333, 2.6666667, 2.3333333,
            2.6666667, 3.0, 3.3333333, 3.0, 3.3333333, 3.6666667, 4.0,
        ],
        1e-5,
    );
}

#[test]
fn constant_constant_of_shape_and_cast() {
    let val = b::f32_tensor("val", &[1], &[7.0]);
    let nodes = vec![
        b::node(
            "Constant",
            &[],
            &["shape"],
            &[Attr::Ints("value_ints", vec![3, 4])],
        ),
        b::node(
            "ConstantOfShape",
            &["shape"],
            &["filled"],
            &[Attr::Tensor("value", val)],
        ),
        b::node("Cast", &["x"], &["cast"], &[Attr::Int("to", b::INT64)]),
    ];
    let inputs = vec![b::value_info("x", b::FLOAT, &[Dim::Val(2)])];
    let outputs = vec![
        b::value_info("filled", b::FLOAT, &[Dim::Val(3), Dim::Val(4)]),
        b::value_info("cast", b::INT64, &[Dim::Val(2)]),
    ];
    let bytes = b::model(&b::graph(&nodes, &[], &inputs, &outputs, "constant"), 17);

    let mut m = OnnxModel::from_bytes(&bytes, None).unwrap();
    let x = Tensor::from_slice(m.ctx(), &[1.7f32, -2.2], Some(dtype::FLOAT32), None);
    let out = m.run(&[("x", x)]).unwrap();
    assert_eq!(out["filled"].to_list::<f32>(m.session_mut()), vec![7.0; 12]);
    assert_eq!(out["cast"].to_list::<i64>(m.session_mut()), vec![1, -2]);
}

#[test]
fn typed_float_data_and_int64_data_fields() {
    let w = b::f32_tensor_typed("w", &[3], &[1.0, 2.0, 3.0]);
    let nodes = vec![b::node("Mul", &["x", "w"], &["out"], &[])];
    let inputs = vec![b::value_info("x", b::FLOAT, &[Dim::Val(3)])];
    let outputs = vec![b::value_info("out", b::FLOAT, &[Dim::Val(3)])];
    let bytes = b::model(&b::graph(&nodes, &[w], &inputs, &outputs, "typed"), 17);
    let got = f32_model(&bytes, &[("x", vec![2.0, 3.0, 4.0], vec![3])], "out");
    close(&got, &[2.0, 6.0, 12.0], 1e-6);
}

#[test]
fn external_data_is_read_from_the_model_directory() {
    let dir = std::env::temp_dir().join(format!("autorust_onnx_ext_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let bin = dir.join("weights.bin");
    let mut raw = Vec::new();
    for v in [1.0f32, 2.0, 3.0, 4.0] {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(&bin, &raw).unwrap();

    let w = b::external_f32_tensor("w", &[2, 2], "weights.bin", 0, 16);
    let nodes = vec![b::node("Add", &["x", "w"], &["out"], &[])];
    let inputs = vec![b::value_info("x", b::FLOAT, &[Dim::Val(2), Dim::Val(2)])];
    let outputs = vec![b::value_info("out", b::FLOAT, &[Dim::Val(2), Dim::Val(2)])];
    let bytes = b::model(&b::graph(&nodes, &[w], &inputs, &outputs, "ext"), 17);
    let path = dir.join("model.onnx");
    std::fs::write(&path, &bytes).unwrap();

    let mut m = OnnxModel::load(&path).unwrap();
    let x = Tensor::zeros(m.ctx(), [2, 2]);
    let out = m.run(&[("x", x)]).unwrap();
    close(
        &out["out"].to_list::<f32>(m.session_mut()),
        &[1.0, 2.0, 3.0, 4.0],
        1e-6,
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn negative_gather_indices_are_normalized() {
    let idx = b::i64_tensor("idx", &[2], &[-1, 0]);
    let nodes = vec![b::node(
        "Gather",
        &["x", "idx"],
        &["out"],
        &[Attr::Int("axis", 0)],
    )];
    let inputs = vec![b::value_info("x", b::FLOAT, &[Dim::Val(2), Dim::Val(2)])];
    let outputs = vec![b::value_info("out", b::FLOAT, &[Dim::Val(2), Dim::Val(2)])];
    let bytes = b::model(&b::graph(&nodes, &[idx], &inputs, &outputs, "gather"), 17);
    let got = f32_model(
        &bytes,
        &[("x", vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])],
        "out",
    );
    close(&got, &[3.0, 4.0, 1.0, 2.0], 1e-6);
}

#[test]
fn unsupported_op_reports_the_name() {
    let nodes = vec![b::node("NonZero", &["x"], &["out"], &[])];
    let inputs = vec![b::value_info("x", b::FLOAT, &[Dim::Val(3)])];
    let outputs = vec![b::value_info("out", b::INT64, &[Dim::Val(1)])];
    let bytes = b::model(&b::graph(&nodes, &[], &inputs, &outputs, "nz"), 17);
    let mut m = OnnxModel::from_bytes(&bytes, None).unwrap();
    let x = Tensor::from_slice(m.ctx(), &[1.0f32, 0.0, 2.0], Some(dtype::FLOAT32), None);
    match m.run(&[("x", x)]) {
        Err(OnnxError::Unsupported(s)) => assert!(s.contains("NonZero"), "{s}"),
        other => panic!("expected Unsupported, got {other:?}"),
    }
}

#[test]
fn malformed_bytes_and_bad_inputs_are_errors() {
    assert!(OnnxModel::from_bytes(&[0x08], None).is_err());

    let nodes = vec![b::node("Relu", &["x"], &["out"], &[])];
    let inputs = vec![b::value_info("x", b::FLOAT, &[Dim::Val(2), Dim::Val(2)])];
    let outputs = vec![b::value_info("out", b::FLOAT, &[Dim::Val(2), Dim::Val(2)])];
    let bytes = b::model(&b::graph(&nodes, &[], &inputs, &outputs, "relu"), 17);

    let mut m = OnnxModel::from_bytes(&bytes, None).unwrap();
    assert!(matches!(m.run(&[]), Err(OnnxError::Runtime(_))));

    let wrong = Tensor::from_slice(m.ctx(), &[1.0f32; 3], Some(dtype::FLOAT32), None);
    assert!(matches!(m.run(&[("x", wrong)]), Err(OnnxError::Shape(_))));

    // A tensor from another arena is rejected rather than moved.
    let other = Ctx::new();
    let foreign =
        Tensor::from_slice(&other, &[1.0f32; 4], Some(dtype::FLOAT32), None).reshape([2, 2]);
    assert!(matches!(
        m.run(&[("x", foreign)]),
        Err(OnnxError::Runtime(_))
    ));
}

#[test]
fn symbolic_dims_are_bound_by_the_fed_tensor() {
    let nodes = vec![b::node("Shape", &["x"], &["shape"], &[])];
    let inputs = vec![b::value_info(
        "x",
        b::FLOAT,
        &[Dim::Param("batch"), Dim::Val(3), Dim::Param("height")],
    )];
    let outputs = vec![b::value_info("shape", b::INT64, &[Dim::Val(3)])];
    let bytes = b::model(&b::graph(&nodes, &[], &inputs, &outputs, "shape"), 17);
    let mut m = OnnxModel::from_bytes(&bytes, None).unwrap();
    let x = Tensor::zeros(m.ctx(), [2, 3, 7]);
    let out = m.run(&[("x", x)]).unwrap();
    assert_eq!(out["shape"].to_list::<i64>(m.session_mut()), vec![2, 3, 7]);
}

#[test]
fn conv_transpose_and_depth_to_space() {
    let w = b::f32_tensor("w", &[1, 1, 2, 2], &[1.0, 1.0, 1.0, 1.0]);
    let nodes = vec![
        b::node(
            "ConvTranspose",
            &["x", "w"],
            &["deconv"],
            &[Attr::Ints("strides", vec![2, 2])],
        ),
        b::node(
            "DepthToSpace",
            &["d"],
            &["out"],
            &[Attr::Int("blocksize", 2)],
        ),
    ];
    let inputs = vec![
        b::value_info(
            "x",
            b::FLOAT,
            &[Dim::Val(1), Dim::Val(1), Dim::Val(2), Dim::Val(2)],
        ),
        b::value_info(
            "d",
            b::FLOAT,
            &[Dim::Val(1), Dim::Val(4), Dim::Val(1), Dim::Val(1)],
        ),
    ];
    let outputs = vec![
        b::value_info(
            "deconv",
            b::FLOAT,
            &[Dim::Val(1), Dim::Val(1), Dim::Val(4), Dim::Val(4)],
        ),
        b::value_info(
            "out",
            b::FLOAT,
            &[Dim::Val(1), Dim::Val(1), Dim::Val(2), Dim::Val(2)],
        ),
    ];
    let bytes = b::model(&b::graph(&nodes, &[w], &inputs, &outputs, "deconv"), 17);

    let mut m = OnnxModel::from_bytes(&bytes, None).unwrap();
    let x = Tensor::from_slice(
        m.ctx(),
        &[1.0f32, 1.0, 1.0, 1.0],
        Some(dtype::FLOAT32),
        None,
    )
    .reshape([1, 1, 2, 2]);
    let d = Tensor::from_slice(
        m.ctx(),
        &[1.0f32, 2.0, 3.0, 4.0],
        Some(dtype::FLOAT32),
        None,
    )
    .reshape([1, 4, 1, 1]);
    let out = m.run(&[("x", x), ("d", d)]).unwrap();
    assert_eq!(out["deconv"].shape().as_slice().len(), 4);
    assert_eq!(
        out["deconv"].shape().as_slice()[2].clone().as_i64(),
        Some(4)
    );
    assert_eq!(out["out"].shape().as_slice()[2].clone().as_i64(), Some(2));
    // DCR: channels (1,2) form the top row and (3,4) the bottom row.
    close(
        &out["out"].to_list::<f32>(m.session_mut()),
        &[1.0, 2.0, 3.0, 4.0],
        1e-6,
    );
}

#[test]
fn session_is_reusable_across_runs() {
    let nodes = vec![b::node("Add", &["x", "y"], &["out"], &[])];
    let inputs = vec![
        b::value_info("x", b::FLOAT, &[Dim::Val(2)]),
        b::value_info("y", b::FLOAT, &[Dim::Val(2)]),
    ];
    let outputs = vec![b::value_info("out", b::FLOAT, &[Dim::Val(2)])];
    let bytes = b::model(&b::graph(&nodes, &[], &inputs, &outputs, "reuse"), 17);
    let mut m = OnnxModel::from_bytes(&bytes, None).unwrap();
    for step in 0..3 {
        let x = Tensor::from_slice(m.ctx(), &[step as f32, 1.0], Some(dtype::FLOAT32), None);
        let y = Tensor::from_slice(m.ctx(), &[10.0f32, 20.0], Some(dtype::FLOAT32), None);
        let out = m.run(&[("x", x), ("y", y)]).unwrap();
        close(
            &out["out"].to_list::<f32>(m.session_mut()),
            &[step as f32 + 10.0, 21.0],
            1e-6,
        );
    }
}

#[test]
fn opset_13_softmax_defaults_to_last_axis() {
    let nodes = vec![b::node("Softmax", &["x"], &["out"], &[])];
    let inputs = vec![b::value_info("x", b::FLOAT, &[Dim::Val(2), Dim::Val(3)])];
    let outputs = vec![b::value_info("out", b::FLOAT, &[Dim::Val(2), Dim::Val(3)])];
    let bytes = b::model(&b::graph(&nodes, &[], &inputs, &outputs, "sm"), 13);
    let got = f32_model(
        &bytes,
        &[("x", vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0], vec![2, 3])],
        "out",
    );
    // Rows must each sum to 1 when the default axis is the last one.
    let r0: f32 = got[..3].iter().sum();
    let r1: f32 = got[3..].iter().sum();
    assert!((r0 - 1.0).abs() < 1e-5, "r0={r0}");
    assert!((r1 - 1.0).abs() < 1e-5, "r1={r1}");
}

/// End-to-end smoke test against an external checkpoint (not shipped).
///
/// `ONNX_MODEL=/path/model.onnx [ONNX_DEVICE=CUDA] [ONNX_SHAPE=1,3,518,518]`
/// `cargo test --release --test onnx -- --ignored external_checkpoint_smoke`
///
/// `ONNX_SHAPE` overrides the first input's declared dims (needed for models
/// whose spatial dims are symbolic).
#[test]
#[ignore = "needs ONNX_MODEL pointing at a real model file"]
fn external_checkpoint_smoke() {
    let path = std::env::var("ONNX_MODEL").expect("set ONNX_MODEL");
    let dev = std::env::var("ONNX_DEVICE").ok();
    let mut m = OnnxModel::load_with_device(&path, dev.as_deref()).expect("load");
    let first_shape: Option<Vec<i64>> = std::env::var("ONNX_SHAPE").ok().map(|s| {
        s.split(',')
            .map(|v| v.trim().parse().expect("ONNX_SHAPE dims"))
            .collect()
    });
    let specs: Vec<(String, Vec<i64>)> = m
        .inputs()
        .iter()
        .enumerate()
        .map(|(idx, i)| {
            if idx == 0
                && let Some(shape) = &first_shape
            {
                return (i.name.clone(), shape.clone());
            }
            let dims = i
                .dims
                .iter()
                .map(|d| match d {
                    autorust::onnx::Dim::Value(v) => *v,
                    _ => 1,
                })
                .collect();
            (i.name.clone(), dims)
        })
        .collect();
    let tensors: Vec<(String, Tensor)> = specs
        .iter()
        .map(|(name, shape)| {
            (
                name.clone(),
                Tensor::full(
                    m.ctx(),
                    shape.clone(),
                    dtype::ConstVal::Float(0.0),
                    Some(dtype::FLOAT32),
                    dev.as_deref(),
                ),
            )
        })
        .collect();
    let refs: Vec<(&str, Tensor)> = tensors
        .iter()
        .map(|(n, t)| (n.as_str(), t.clone()))
        .collect();
    let out = m.run(&refs).expect("run");
    assert!(!out.is_empty());
    for name in m.outputs().iter() {
        assert!(out.contains_key(name), "missing output {name}");
    }
    // Read one output to force a realize; all values must be finite.
    let (name, t) = out.iter().next().unwrap();
    let vals = t.to_list::<f32>(m.session_mut());
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "non-finite value in {name}"
    );
}

#[test]
fn pad_constant_and_edge_reflect_wrap() {
    let pads2 = b::i64_tensor("pads2", &[4], &[0, 1, 0, 1]);
    let pads1a = b::i64_tensor("pads1a", &[2], &[2, 1]);
    let cval = b::f32_tensor("cval", &[], &[9.0]);
    // The three single-output Pads need separate nodes; build them explicitly.
    let nodes = vec![
        b::node(
            "Pad",
            &["x", "pads2", "cval"],
            &["const"],
            &[Attr::Str("mode", "constant")],
        ),
        b::node(
            "Pad",
            &["v", "pads1a"],
            &["edge"],
            &[Attr::Str("mode", "edge")],
        ),
        b::node(
            "Pad",
            &["v", "pads1a"],
            &["reflect"],
            &[Attr::Str("mode", "reflect")],
        ),
        b::node(
            "Pad",
            &["v", "pads1a"],
            &["wrap"],
            &[Attr::Str("mode", "wrap")],
        ),
    ];
    let inputs = vec![
        b::value_info("x", b::FLOAT, &[Dim::Val(2), Dim::Val(3)]),
        b::value_info("v", b::FLOAT, &[Dim::Val(3)]),
    ];
    let outputs = vec![
        b::value_info("const", b::FLOAT, &[Dim::Val(2), Dim::Val(5)]),
        b::value_info("edge", b::FLOAT, &[Dim::Val(6)]),
        b::value_info("reflect", b::FLOAT, &[Dim::Val(6)]),
        b::value_info("wrap", b::FLOAT, &[Dim::Val(6)]),
    ];
    let bytes = b::model(
        &b::graph(&nodes, &[pads2, pads1a, cval], &inputs, &outputs, "pad"),
        17,
    );

    let mut m = OnnxModel::from_bytes(&bytes, None).unwrap();
    let x = Tensor::from_slice(
        m.ctx(),
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        Some(dtype::FLOAT32),
        None,
    )
    .reshape([2, 3]);
    let v = Tensor::from_slice(m.ctx(), &[1.0f32, 2.0, 3.0], Some(dtype::FLOAT32), None);
    let out = m.run(&[("x", x), ("v", v)]).unwrap();
    close(
        &out["const"].to_list::<f32>(m.session_mut()),
        &[9.0, 1.0, 2.0, 3.0, 9.0, 9.0, 4.0, 5.0, 6.0, 9.0],
        1e-6,
    );
    close(
        &out["edge"].to_list::<f32>(m.session_mut()),
        &[1.0, 1.0, 1.0, 2.0, 3.0, 3.0],
        1e-6,
    );
    close(
        &out["reflect"].to_list::<f32>(m.session_mut()),
        &[3.0, 2.0, 1.0, 2.0, 3.0, 2.0],
        1e-6,
    );
    close(
        &out["wrap"].to_list::<f32>(m.session_mut()),
        &[2.0, 3.0, 1.0, 2.0, 3.0, 1.0],
        1e-6,
    );
}

#[test]
fn split_squeeze_unsqueeze_flatten_tile() {
    let split = b::i64_tensor("split", &[2], &[1, 3]);
    let nodes = vec![
        b::node(
            "Split",
            &["x", "split"],
            &["a", "b"],
            &[Attr::Int("axis", 1)],
        ),
        b::node(
            "Squeeze",
            &["s"],
            &["sq"],
            &[Attr::Ints("axes", vec![0, 2])],
        ),
        b::node(
            "Unsqueeze",
            &["u"],
            &["us"],
            &[Attr::Ints("axes", vec![0, 2])],
        ),
        b::node("Flatten", &["f"], &["fl"], &[Attr::Int("axis", 1)]),
        b::node("Tile", &["t", "reps"], &["ti"], &[]),
    ];
    let reps = b::i64_tensor("reps", &[1], &[3]);
    let inputs = vec![
        b::value_info("x", b::FLOAT, &[Dim::Val(2), Dim::Val(4)]),
        b::value_info("s", b::FLOAT, &[Dim::Val(1), Dim::Val(2), Dim::Val(1)]),
        b::value_info("u", b::FLOAT, &[Dim::Val(2)]),
        b::value_info("f", b::FLOAT, &[Dim::Val(2), Dim::Val(2), Dim::Val(2)]),
        b::value_info("t", b::FLOAT, &[Dim::Val(2)]),
    ];
    let outputs = vec![
        b::value_info("a", b::FLOAT, &[Dim::Val(2), Dim::Val(1)]),
        b::value_info("b", b::FLOAT, &[Dim::Val(2), Dim::Val(3)]),
        b::value_info("sq", b::FLOAT, &[Dim::Val(2)]),
        b::value_info("us", b::FLOAT, &[Dim::Val(1), Dim::Val(2), Dim::Val(1)]),
        b::value_info("fl", b::FLOAT, &[Dim::Val(2), Dim::Val(4)]),
        b::value_info("ti", b::FLOAT, &[Dim::Val(6)]),
    ];
    let bytes = b::model(
        &b::graph(&nodes, &[split, reps], &inputs, &outputs, "move2"),
        17,
    );

    let mut m = OnnxModel::from_bytes(&bytes, None).unwrap();
    let x = Tensor::from_slice(
        m.ctx(),
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        Some(dtype::FLOAT32),
        None,
    )
    .reshape([2, 4]);
    let s =
        Tensor::from_slice(m.ctx(), &[1.0f32, 2.0], Some(dtype::FLOAT32), None).reshape([1, 2, 1]);
    let u = Tensor::from_slice(m.ctx(), &[1.0f32, 2.0], Some(dtype::FLOAT32), None);
    let f = Tensor::from_slice(
        m.ctx(),
        &(0..8).map(|i| i as f32).collect::<Vec<_>>(),
        Some(dtype::FLOAT32),
        None,
    )
    .reshape([2, 2, 2]);
    let t = Tensor::from_slice(m.ctx(), &[1.0f32, 2.0], Some(dtype::FLOAT32), None);
    let out = m
        .run(&[("x", x), ("s", s), ("u", u), ("f", f), ("t", t)])
        .unwrap();
    close(&out["a"].to_list::<f32>(m.session_mut()), &[1.0, 5.0], 1e-6);
    close(
        &out["b"].to_list::<f32>(m.session_mut()),
        &[2.0, 3.0, 4.0, 6.0, 7.0, 8.0],
        1e-6,
    );
    close(
        &out["sq"].to_list::<f32>(m.session_mut()),
        &[1.0, 2.0],
        1e-6,
    );
    close(
        &out["us"].to_list::<f32>(m.session_mut()),
        &[1.0, 2.0],
        1e-6,
    );
    assert_eq!(out["us"].shape().as_slice().len(), 3);
    close(
        &out["fl"].to_list::<f32>(m.session_mut()),
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        1e-6,
    );
    close(
        &out["ti"].to_list::<f32>(m.session_mut()),
        &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
        1e-6,
    );
}

#[test]
fn topk_cumsum_and_trilu() {
    let k = b::i64_tensor("k", &[1], &[2]);
    let axis = b::i64_tensor("axis", &[], &[0]);
    let nodes = vec![
        b::node(
            "TopK",
            &["x", "k"],
            &["tv", "ti"],
            &[Attr::Int("axis", 0), Attr::Int("largest", 1)],
        ),
        b::node("CumSum", &["c", "axis"], &["cs"], &[]),
        b::node(
            "CumSum",
            &["c", "axis"],
            &["ex"],
            &[Attr::Int("exclusive", 1)],
        ),
        b::node(
            "CumSum",
            &["c", "axis"],
            &["rev"],
            &[Attr::Int("reverse", 1)],
        ),
        b::node("Trilu", &["m"], &["up"], &[Attr::Int("upper", 1)]),
        b::node("Trilu", &["m"], &["lo"], &[Attr::Int("upper", 0)]),
    ];
    let inputs = vec![
        b::value_info("x", b::FLOAT, &[Dim::Val(5)]),
        b::value_info("c", b::FLOAT, &[Dim::Val(4)]),
        b::value_info("m", b::FLOAT, &[Dim::Val(3), Dim::Val(3)]),
    ];
    let outputs = vec![
        b::value_info("tv", b::FLOAT, &[Dim::Val(2)]),
        b::value_info("ti", b::INT64, &[Dim::Val(2)]),
        b::value_info("cs", b::FLOAT, &[Dim::Val(4)]),
        b::value_info("ex", b::FLOAT, &[Dim::Val(4)]),
        b::value_info("rev", b::FLOAT, &[Dim::Val(4)]),
        b::value_info("up", b::FLOAT, &[Dim::Val(3), Dim::Val(3)]),
        b::value_info("lo", b::FLOAT, &[Dim::Val(3), Dim::Val(3)]),
    ];
    let bytes = b::model(&b::graph(&nodes, &[k, axis], &inputs, &outputs, "topk"), 17);

    let mut m = OnnxModel::from_bytes(&bytes, None).unwrap();
    let x = Tensor::from_slice(
        m.ctx(),
        &[0.1f32, 0.5, 1.2, 3.4, 2.1],
        Some(dtype::FLOAT32),
        None,
    );
    let c = Tensor::from_slice(
        m.ctx(),
        &[1.0f32, 2.0, 3.0, 4.0],
        Some(dtype::FLOAT32),
        None,
    );
    let mat = Tensor::from_slice(
        m.ctx(),
        &(1..=9).map(|i| i as f32).collect::<Vec<_>>(),
        Some(dtype::FLOAT32),
        None,
    )
    .reshape([3, 3]);
    let out = m.run(&[("x", x), ("c", c), ("m", mat)]).unwrap();
    assert_eq!(out["ti"].to_list::<i64>(m.session_mut()), vec![3, 4]);
    close(
        &out["tv"].to_list::<f32>(m.session_mut()),
        &[3.4, 2.1],
        1e-6,
    );
    close(
        &out["cs"].to_list::<f32>(m.session_mut()),
        &[1.0, 3.0, 6.0, 10.0],
        1e-6,
    );
    close(
        &out["ex"].to_list::<f32>(m.session_mut()),
        &[0.0, 1.0, 3.0, 6.0],
        1e-6,
    );
    close(
        &out["rev"].to_list::<f32>(m.session_mut()),
        &[10.0, 9.0, 7.0, 4.0],
        1e-6,
    );
    close(
        &out["up"].to_list::<f32>(m.session_mut()),
        &[1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0],
        1e-6,
    );
    close(
        &out["lo"].to_list::<f32>(m.session_mut()),
        &[1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0],
        1e-6,
    );
}

#[test]
fn one_hot_and_gather_nd() {
    let depth = b::i64_tensor("depth", &[], &[3]);
    let values = b::f32_tensor("values", &[2], &[0.0, 1.0]);
    let gidx = b::i64_tensor("gidx", &[2, 1], &[0, 2]);
    let nodes = vec![
        b::node(
            "OneHot",
            &["idx", "depth", "values"],
            &["oh"],
            &[Attr::Int("axis", -1)],
        ),
        b::node("GatherND", &["x", "gidx"], &["gn"], &[]),
    ];
    let inputs = vec![
        b::value_info("idx", b::INT64, &[Dim::Val(2)]),
        b::value_info("x", b::FLOAT, &[Dim::Val(3), Dim::Val(2)]),
    ];
    let outputs = vec![
        b::value_info("oh", b::FLOAT, &[Dim::Val(2), Dim::Val(3)]),
        b::value_info("gn", b::FLOAT, &[Dim::Val(2), Dim::Val(2)]),
    ];
    let bytes = b::model(
        &b::graph(&nodes, &[depth, values, gidx], &inputs, &outputs, "onehot"),
        17,
    );

    let mut m = OnnxModel::from_bytes(&bytes, None).unwrap();
    let idx = Tensor::from_slice(m.ctx(), &[0i64, 2], Some(dtype::INT64), None);
    let x = Tensor::from_slice(
        m.ctx(),
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        Some(dtype::FLOAT32),
        None,
    )
    .reshape([3, 2]);
    let out = m.run(&[("idx", idx), ("x", x)]).unwrap();
    close(
        &out["oh"].to_list::<f32>(m.session_mut()),
        &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        1e-6,
    );
    close(
        &out["gn"].to_list::<f32>(m.session_mut()),
        &[1.0, 2.0, 5.0, 6.0],
        1e-6,
    );
}

#[test]
fn batch_and_instance_normalization() {
    let scale = b::f32_tensor("scale", &[2], &[1.0, 1.0]);
    let bias = b::f32_tensor("bias", &[2], &[0.0, 0.0]);
    let mean = b::f32_tensor("mean", &[2], &[0.0, 1.0]);
    let var = b::f32_tensor("var", &[2], &[1.0, 1.0]);
    let nodes = vec![
        b::node(
            "BatchNormalization",
            &["x", "scale", "bias", "mean", "var"],
            &["bn"],
            &[Attr::Float("epsilon", 1e-5)],
        ),
        b::node(
            "InstanceNormalization",
            &["i", "scale", "bias"],
            &["inorm"],
            &[Attr::Float("epsilon", 1e-5)],
        ),
    ];
    let inputs = vec![
        b::value_info(
            "x",
            b::FLOAT,
            &[Dim::Val(1), Dim::Val(2), Dim::Val(1), Dim::Val(1)],
        ),
        b::value_info(
            "i",
            b::FLOAT,
            &[Dim::Val(1), Dim::Val(2), Dim::Val(1), Dim::Val(2)],
        ),
    ];
    let outputs = vec![
        b::value_info(
            "bn",
            b::FLOAT,
            &[Dim::Val(1), Dim::Val(2), Dim::Val(1), Dim::Val(1)],
        ),
        b::value_info(
            "inorm",
            b::FLOAT,
            &[Dim::Val(1), Dim::Val(2), Dim::Val(1), Dim::Val(2)],
        ),
    ];
    let bytes = b::model(
        &b::graph(&nodes, &[scale, bias, mean, var], &inputs, &outputs, "norm"),
        17,
    );

    let mut m = OnnxModel::from_bytes(&bytes, None).unwrap();
    let x = Tensor::from_slice(m.ctx(), &[1.0f32, 1.0], Some(dtype::FLOAT32), None)
        .reshape([1, 2, 1, 1]);
    let i = Tensor::from_slice(
        m.ctx(),
        &[1.0f32, 3.0, 2.0, 6.0],
        Some(dtype::FLOAT32),
        None,
    )
    .reshape([1, 2, 1, 2]);
    let out = m.run(&[("x", x), ("i", i)]).unwrap();
    close(
        &out["bn"].to_list::<f32>(m.session_mut()),
        &[1.0, 0.0],
        2e-5,
    );
    close(
        &out["inorm"].to_list::<f32>(m.session_mut()),
        &[-1.0, 1.0, -1.0, 1.0],
        2e-4,
    );
}

#[test]
fn gather_elements_uses_full_shape_indices() {
    let nodes = vec![
        b::node(
            "GatherElements",
            &["x", "i1"],
            &["axis1"],
            &[Attr::Int("axis", 1)],
        ),
        b::node(
            "GatherElements",
            &["x", "i0"],
            &["axis0"],
            &[Attr::Int("axis", 0)],
        ),
    ];
    let inputs = vec![
        b::value_info("x", b::FLOAT, &[Dim::Val(2), Dim::Val(2)]),
        b::value_info("i1", b::INT64, &[Dim::Val(2), Dim::Val(2)]),
        b::value_info("i0", b::INT64, &[Dim::Val(2), Dim::Val(2)]),
    ];
    let outputs = vec![
        b::value_info("axis1", b::FLOAT, &[Dim::Val(2), Dim::Val(2)]),
        b::value_info("axis0", b::FLOAT, &[Dim::Val(2), Dim::Val(2)]),
    ];
    let bytes = b::model(&b::graph(&nodes, &[], &inputs, &outputs, "ge"), 17);

    let mut m = OnnxModel::from_bytes(&bytes, None).unwrap();
    let x = Tensor::from_slice(
        m.ctx(),
        &[1.0f32, 2.0, 3.0, 4.0],
        Some(dtype::FLOAT32),
        None,
    )
    .reshape([2, 2]);
    let i1 =
        Tensor::from_slice(m.ctx(), &[0i64, 1, 1, 0], Some(dtype::INT64), None).reshape([2, 2]);
    let i0 =
        Tensor::from_slice(m.ctx(), &[1i64, 0, 0, 1], Some(dtype::INT64), None).reshape([2, 2]);
    let out = m.run(&[("x", x), ("i1", i1), ("i0", i0)]).unwrap();
    close(
        &out["axis1"].to_list::<f32>(m.session_mut()),
        &[1.0, 2.0, 4.0, 3.0],
        1e-6,
    );
    close(
        &out["axis0"].to_list::<f32>(m.session_mut()),
        &[3.0, 2.0, 1.0, 4.0],
        1e-6,
    );
}

#[test]
fn zero_element_initializer_is_a_clear_error() {
    let empty = b::f32_tensor("empty", &[0], &[]);
    let nodes = vec![b::node("Relu", &["x"], &["out"], &[])];
    let inputs = vec![b::value_info("x", b::FLOAT, &[Dim::Val(1)])];
    let outputs = vec![b::value_info("out", b::FLOAT, &[Dim::Val(1)])];
    let bytes = b::model(&b::graph(&nodes, &[empty], &inputs, &outputs, "empty"), 17);
    match OnnxModel::from_bytes(&bytes, None) {
        Err(OnnxError::Unsupported(s)) => assert!(s.contains("zero-element"), "{s}"),
        Err(e) => panic!("expected Unsupported, got {e}"),
        Ok(_) => panic!("expected a zero-element error"),
    }
}

#[test]
fn resize_nearest_honors_nearest_mode() {
    let sizes = b::i64_tensor("sizes", &[4], &[1, 1, 1, 4]);
    let nodes = vec![
        b::node(
            "Resize",
            &["x", "", "", "sizes"],
            &["default"],
            &[
                Attr::Str("mode", "nearest"),
                Attr::Str("coordinate_transformation_mode", "half_pixel"),
            ],
        ),
        b::node(
            "Resize",
            &["x", "", "", "sizes"],
            &["floor"],
            &[
                Attr::Str("mode", "nearest"),
                Attr::Str("coordinate_transformation_mode", "half_pixel"),
                Attr::Str("nearest_mode", "floor"),
            ],
        ),
    ];
    let inputs = vec![b::value_info(
        "x",
        b::FLOAT,
        &[Dim::Val(1), Dim::Val(1), Dim::Val(1), Dim::Val(2)],
    )];
    let outputs = vec![
        b::value_info(
            "default",
            b::FLOAT,
            &[Dim::Val(1), Dim::Val(1), Dim::Val(1), Dim::Val(4)],
        ),
        b::value_info(
            "floor",
            b::FLOAT,
            &[Dim::Val(1), Dim::Val(1), Dim::Val(1), Dim::Val(4)],
        ),
    ];
    let bytes = b::model(
        &b::graph(&nodes, &[sizes], &inputs, &outputs, "resize_near"),
        13,
    );

    let mut m = OnnxModel::from_bytes(&bytes, None).unwrap();
    let x = Tensor::from_slice(m.ctx(), &[0.0f32, 10.0], Some(dtype::FLOAT32), None)
        .reshape([1, 1, 1, 2]);
    let out = m.run(&[("x", x)]).unwrap();
    close(
        &out["default"].to_list::<f32>(m.session_mut()),
        &[0.0, 0.0, 10.0, 10.0],
        1e-6,
    );
    close(
        &out["floor"].to_list::<f32>(m.session_mut()),
        &[0.0, 0.0, 0.0, 10.0],
        1e-6,
    );
}
