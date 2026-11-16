#!/usr/bin/env python3
"""Check the exporter's E0/E1/E2 artifacts with onnxruntime.

Run after `cargo test --release --test onnx_export` has written the models and
assets to /tmp/opencode. Use the bench virtualenv (onnx + onnxruntime).
"""
import numpy as np
import onnx
import onnxruntime as ort

ROOT = "/tmp/opencode"


def run(path, feed):
    model = onnx.load(path)
    onnx.checker.check_model(model)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return sess.run(None, feed)[0]


x = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]], dtype=np.float32)

e0 = run(
    f"{ROOT}/onnx_export_e0.onnx",
    {"x": np.array([1.0, 2.0], dtype=np.float32)},
)
assert np.allclose(e0, [2.0, 3.0]), e0

w = np.array([0.2, -0.3, 0.4, 0.1, -0.5, 0.6], dtype=np.float32).reshape(3, 2)
b = np.array([0.05, -0.05], dtype=np.float32)
e1 = run(f"{ROOT}/onnx_export_e1.onnx", {"x": x})
assert np.allclose(e1, np.maximum(x @ w + b, 0)), e1

x2 = np.fromfile(f"{ROOT}/onnx_export_e2_x.bin", dtype=np.float32).reshape(2, 3)
y2 = np.fromfile(f"{ROOT}/onnx_export_e2_y.bin", dtype=np.float32).reshape(2, 2)
e2 = run(f"{ROOT}/onnx_export_e2.onnx", {"x": x2})
assert np.allclose(e2, y2), (e2, y2)

# E3: dynamic batch and external data.
e3d = run(
    f"{ROOT}/onnx_export_e3_dynamic.onnx",
    {"x": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)},
)
assert np.allclose(e3d, np.array([[3.0, 5.0], [7.0, 9.0], [11.0, 13.0]])), e3d
e3x = np.fromfile(f"{ROOT}/onnx_export_e2_x.bin", dtype=np.float32).reshape(2, 3)
e3ext = run(f"{ROOT}/onnx_export_e3_external.onnx", {"x": e3x})
assert np.allclose(e3ext, y2), (e3ext, y2)

print("onnx export E0/E1/E2/E3: checker + onnxruntime OK")
print("E0", e0.tolist())
print("E1", e1.tolist())
print("E2", e2.tolist())
print("E3 dynamic", e3d.tolist())
print("E3 external", e3ext.tolist())
