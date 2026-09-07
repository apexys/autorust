"""M5 numeric gate: Rust uops interpreter vs tinygrad CPU execution.

Runs each corpus program through the Rust schedule -> codegen pipeline ->
linearize -> interpreter (oracle TG_M5_EXEC mode) and compares the output
values against real tinygrad's tolist().
"""
import sys, json, subprocess, unittest
sys.path.insert(0, "py"); sys.path.insert(0, "py/tests"); sys.path.insert(0, "../tinygrad")

ORACLE = "./target/debug/oracle_tensor"


def rs_exec(prog):
    p = subprocess.run([ORACLE], input=json.dumps({"prog": prog, "mode": "schedule"}),
                       capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin", "TG_M5_EXEC": "1"})
    assert p.returncode == 0, p.stderr[-400:]
    out = json.loads(p.stdout)
    assert out.get("ok"), out
    return out["outs"]


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
        else:
            raise AssertionError(f"unknown op {op}")
    return [tensors[-1].tolist()]


class TestExecDiff(unittest.TestCase):
    def exec_equality(self, name, prog):
        with self.subTest(name=name):
            rs = rs_exec(prog)
            py = py_exec(prog)
            self.assertEqual(len(rs), len(py), "output count differs")
            for r, p in zip(rs, py):
                self.assertEqual(len(r), len(p), "output sizes differ")
                for a, b in zip(r, p):
                    self.assertAlmostEqual(a, b, places=5, msg=f"{name}: {r} vs {p}")

    def test_ew_kernel(self):
        self.exec_equality("ew kernel", [
            {"op": "from_slice", "data": [1.0, 2.0, 3.0, 4.0], "kind": "f32", "dtype": None},
            {"op": "addc", "x": 0, "val": 1.0},
        ])

    def test_mul_kernel(self):
        self.exec_equality("mul kernel", [
            {"op": "from_slice", "data": [1.0, 2.0, 3.0], "kind": "f32", "dtype": None},
            {"op": "mulc", "x": 0, "val": 2.0},
        ])

    def test_neg_values(self):
        self.exec_equality("negative values", [
            {"op": "from_slice", "data": [-1.5, 2.5, -3.5], "kind": "f32", "dtype": None},
            {"op": "mulc", "x": 0, "val": -2.0},
        ])


if __name__ == "__main__":
    unittest.main()
