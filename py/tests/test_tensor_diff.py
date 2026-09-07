# M3 gate: tensor-construction differential — Rust `oracle_tensor` vs real
# tinygrad Tensors. For each program both sides build the same graph; we compare
# UOp keys exactly (the front-end conformance oracle).

import base64, json, os, subprocess, sys, unittest

_here = os.path.dirname(__file__)
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))
sys.path.insert(0, os.path.join(_here, "..", "..", "..", "tinygrad"))

from bridge import json_to_uop  # noqa: E402
from tinygrad import Tensor, dtypes  # noqa: E402

ORACLE = os.path.join(os.path.dirname(__file__), "..", "..", "target", "debug", "oracle_tensor")

# index spec -> the JSON IxSpec the oracle consumes
def ix_spec(i=None, newaxis=False, ellipsis=False, slice=None):
    d = {}
    if i is not None:
        d["int"] = i
    if newaxis:
        d["newaxis"] = True
    if ellipsis:
        d["ellipsis"] = True
    if slice is not None:
        d["slice"] = list(slice)
    return d

def run_rust(prog):
    p = subprocess.run([ORACLE], input=json.dumps({"prog": prog}), capture_output=True, text=True, timeout=60)
    if p.returncode != 0:
        raise RuntimeError(f"oracle_tensor failed: {p.stderr[-400:]}")
    out = json.loads(p.stdout)
    if not out.get("ok"):
        raise RuntimeError(f"oracle_tensor error: {out.get('error')}")
    return json_to_uop(json.dumps(out["graph"]))[0]

def slice_prog(start, stop, step):
    return ix_spec(slice=[start, stop, step])

CASES = [
    # (name, rust_steps, python_builder)
    ("const f32 PYTHON", [
        {"op": "from_slice", "kind": "f32", "data": [1.0, 2.0, 3.0], "dtype": None, "device": "PYTHON"},
    ], lambda: Tensor([1.0, 2.0, 3.0], device="PYTHON")),
    ("const i32 default device", [
        {"op": "from_slice", "kind": "i32", "data": [1, 2, 3, 4], "dtype": None, "device": None},
    ], lambda: Tensor([1, 2, 3, 4])),
    ("const f32 explicit dtype", [
        {"op": "from_slice", "kind": "f32", "data": [1.5, 2.5], "dtype": "double", "device": "PYTHON"},
    ], lambda: Tensor([1.5, 2.5], dtype=dtypes.float64, device="PYTHON")),
    ("bool consts", [
        {"op": "from_slice", "kind": "bool", "data": [1, 0, 1], "dtype": None, "device": "PYTHON"},
    ], lambda: Tensor([True, False, True], device="PYTHON")),
    ("bytes uint8", [
        {"op": "from_bytes", "hex": b"\x01\x02\x03\x04".hex(), "dtype": "uint8", "device": "PYTHON"},
    ], lambda: Tensor(b"\x01\x02\x03\x04", device="PYTHON")),
    ("bytes explicit dtype", [
        {"op": "from_bytes", "hex": b"\x01\x00\x00\x00\x02\x00\x00\x00".hex(), "dtype": "int32", "device": "PYTHON"},
    ], lambda: Tensor(b"\x01\x00\x00\x00\x02\x00\x00\x00", dtype=dtypes.int32, device="PYTHON")),
    ("zeros ones", [
        {"op": "zeros", "shape": [2, 3]},
        {"op": "ones", "shape": [4]},
    ], lambda: (Tensor.zeros(2, 3), Tensor.ones(4))[-1]),
    ("arange", [
        {"op": "arange", "start": 0, "stop": 6, "step": 1, "dtype": None},
    ], lambda: Tensor.arange(6)),
    ("arange step float", [
        {"op": "arange", "start": 0, "stop": 5, "step": 0.5, "dtype": None},
    ], lambda: Tensor.arange(0, 5, 0.5)),
    ("arange stop first", [
        {"op": "arange", "start": 5, "stop": None, "step": 1, "dtype": None},
    ], lambda: Tensor.arange(5)),
    ("eye", [
        {"op": "eye", "n": 3, "m": None, "dtype": None},
    ], lambda: Tensor.eye(3)),
    ("eye rect", [
        {"op": "eye", "n": 2, "m": 4, "dtype": None},
    ], lambda: Tensor.eye(2, 4)),
    ("reshape", [
        {"op": "arange", "start": 0, "stop": 6, "step": 1, "dtype": None},
        {"op": "reshape", "x": 0, "shape": [2, 3]},
    ], lambda: Tensor.arange(6).reshape(2, 3)),
    ("getitem int+fill", [
        {"op": "arange", "start": 0, "stop": 12, "step": 1, "dtype": None},
        {"op": "reshape", "x": 0, "shape": [3, 4]},
        {"op": "getitem", "x": 1, "ixs": [ix_spec(i=1)]},
    ], lambda: Tensor.arange(12).reshape(3, 4)[1]),
    ("getitem basic slice", [
        {"op": "arange", "start": 0, "stop": 12, "step": 1, "dtype": None},
        {"op": "reshape", "x": 0, "shape": [3, 4]},
        {"op": "getitem", "x": 1, "ixs": [slice_prog(1, 3, 1)]},
    ], lambda: Tensor.arange(12).reshape(3, 4)[1:3]),
    ("getitem colon-comma", [
        {"op": "arange", "start": 0, "stop": 12, "step": 1, "dtype": None},
        {"op": "reshape", "x": 0, "shape": [3, 4]},
        {"op": "getitem", "x": 1, "ixs": [slice_prog(None, None, None), ix_spec(i=1)]},
    ], lambda: Tensor.arange(12).reshape(3, 4)[:, 1]),
    ("getitem stride", [
        {"op": "arange", "start": 0, "stop": 12, "step": 1, "dtype": None},
        {"op": "reshape", "x": 0, "shape": [3, 4]},
        {"op": "getitem", "x": 1, "ixs": [slice_prog(None, None, 2)]},
    ], lambda: Tensor.arange(12).reshape(3, 4)[::2]),
    ("getitem negative stride", [
        {"op": "arange", "start": 0, "stop": 6, "step": 1, "dtype": None},
        {"op": "getitem", "x": 0, "ixs": [slice_prog(None, None, -1)]},
    ], lambda: Tensor.arange(6)[::-1]),
    ("getitem newaxis", [
        {"op": "arange", "start": 0, "stop": 12, "step": 1, "dtype": None},
        {"op": "reshape", "x": 0, "shape": [3, 4]},
        {"op": "getitem", "x": 1, "ixs": [ix_spec(newaxis=True), ix_spec(i=2)]},
    ], lambda: Tensor.arange(12).reshape(3, 4)[None, 2]),
    ("getitem ellipsis", [
        {"op": "arange", "start": 0, "stop": 12, "step": 1, "dtype": None},
        {"op": "reshape", "x": 0, "shape": [3, 4]},
        {"op": "getitem", "x": 1, "ixs": [ix_spec(ellipsis=True), ix_spec(i=0)]},
    ], lambda: Tensor.arange(12).reshape(3, 4)[..., 0]),
    ("getitem negative index", [
        {"op": "arange", "start": 0, "stop": 6, "step": 1, "dtype": None},
        {"op": "getitem", "x": 0, "ixs": [ix_spec(i=-1)]},
    ], lambda: Tensor.arange(6)[-1]),
    ("getitem mixed chain", [
        {"op": "arange", "start": 0, "stop": 24, "step": 1, "dtype": None},
        {"op": "reshape", "x": 0, "shape": [2, 3, 4]},
        {"op": "getitem", "x": 1, "ixs": [slice_prog(1, None, -1), slice_prog(None, None, 2), ix_spec(i=3)]},
    ], lambda: Tensor.arange(24).reshape(2, 3, 4)[1::-1, ::2, 3]),
]

class TestTensorDiff(unittest.TestCase):
    def test_key_equality(self):
        import itertools
        from tinygrad.uop.ops import UOp
        for name, steps, py_build in CASES:
            with self.subTest(name=name):
                # both sides start each case with a fresh BUFFER slot counter
                UOp.unique_num = itertools.count(0)
                rust_key = run_rust(steps).key
                py_val = py_build()
                py_key = py_val.uop.key
                self.assertEqual(rust_key, py_key, f"key mismatch for {name}")

if __name__ == "__main__":
    unittest.main()
