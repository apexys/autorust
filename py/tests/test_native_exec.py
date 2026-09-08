"""M6 native gate: full Rust-native realize (Python out of the loop).

Runs each corpus program through the native engine — Tensor schedule ->
memory plan -> to_program (PROGRAM/LINEAR/SOURCE/BINARY) -> device runtime
(interpreter Program) -> buffer readback — and compares against real
tinygrad's tolist(). Also asserts schedule_cache-style behavior: realizing
the same tensor twice must not recompile (compiles == 1).
"""
import sys, json, subprocess, unittest
sys.path.insert(0, "py"); sys.path.insert(0, "py/tests"); sys.path.insert(0, "../tinygrad")

ORACLE = "./target/debug/oracle_tensor"


def rs_native(prog):
    p = subprocess.run([ORACLE], input=json.dumps({"prog": prog, "mode": "native"}),
                       capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin"})
    assert p.returncode == 0, p.stderr[-400:]
    out = json.loads(p.stdout)
    assert out.get("ok"), out
    return out


def py_exec(prog):
    from tinygrad import Tensor
    from tinygrad.dtype import dtypes
    tensors = []
    for st in prog:
        op = st["op"]
        if op == "from_slice":
            t = Tensor(st["data"], dtype=dtypes.float32)
            tensors.append(t.contiguous())
        elif op == "addc":
            tensors.append(tensors[st["x"]] + st["val"])
        elif op == "mulc":
            tensors.append(tensors[st["x"]] * st["val"])
        elif op == "neg":
            tensors.append(-tensors[st["x"]])
        elif op == "sum":
            tensors.append(tensors[st["x"]].sum(axis=tuple(st["axis"]) if st.get("axis") else None))
        else:
            raise AssertionError(f"unknown op {op}")
    return tensors[-1].tolist()


class TestNativeExec(unittest.TestCase):
    def native_equality(self, name, prog):
        with self.subTest(name=name):
            rs = rs_native(prog)
            self.assertTrue(rs["stable"], "double realize diverged")
            self.assertEqual(rs["compiles"], 1, "double realize recompiled")
            py = py_exec(prog)
            r = rs["outs"][-1]
            if isinstance(py, list):
                self.assertEqual(len(r), len(py), "output sizes differ")
                for a, b in zip(r, py):
                    self.assertAlmostEqual(a, b, places=5, msg=f"{name}: {r} vs {py}")
            else:
                self.assertAlmostEqual(r[0], py, places=5, msg=f"{name}: {r} vs {py}")

    def test_ew_kernel(self):
        self.native_equality("ew kernel", [
            {"op": "from_slice", "data": [1.0, 2.0, 3.0, 4.0], "kind": "f32", "dtype": None},
            {"op": "addc", "x": 0, "val": 1.0},
        ])

    def test_mul_kernel(self):
        self.native_equality("mul kernel", [
            {"op": "from_slice", "data": [1.0, 2.0, 3.0], "kind": "f32", "dtype": None},
            {"op": "mulc", "x": 0, "val": 2.0},
        ])

    def test_neg_kernel(self):
        self.native_equality("neg kernel", [
            {"op": "from_slice", "data": [-1.0, -2.5, 3.0], "kind": "f32", "dtype": None},
            {"op": "neg", "x": 0},
        ])

    def test_sum_kernel(self):
        self.native_equality("sum kernel", [
            {"op": "from_slice", "data": [1.0, 2.0, 3.0, 4.0], "kind": "f32", "dtype": None},
            {"op": "sum", "x": 0, "axis": [0]},
        ])

    def test_sum_chain(self):
        self.native_equality("sum chain", [
            {"op": "from_slice", "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "kind": "f32", "dtype": None},
            {"op": "mulc", "x": 0, "val": 2.0},
            {"op": "sum", "x": 1, "axis": [0]},
            {"op": "addc", "x": 2, "val": 1.0},
        ])

    def test_two_kernel_chain(self):
        self.native_equality("two kernel chain", [
            {"op": "from_slice", "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "kind": "f32", "dtype": None},
            {"op": "mulc", "x": 0, "val": 2.0},
            {"op": "addc", "x": 1, "val": 1.0},
        ])


if __name__ == "__main__":
    unittest.main()
