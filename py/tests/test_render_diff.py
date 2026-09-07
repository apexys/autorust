"""M5 cstyle gate: Rust cstyle render vs python ClangRenderer, byte-exact.

For each corpus program: the Rust oracle (TG_M5_RENDER) runs the pipeline with
the clang renderer facts, linearizes, and renders; python renders the same
linearized uops with its ClangRenderer. Outputs must be byte-equal strings.
"""
import sys, json, subprocess, unittest
sys.path.insert(0, "py"); sys.path.insert(0, "py/tests"); sys.path.insert(0, "../tinygrad")

ORACLE = "./target/debug/oracle_tensor"


def rs_render(prog):
    p = subprocess.run([ORACLE], input=json.dumps({"prog": prog, "mode": "schedule"}),
                       capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin", "TG_M5_RENDER": "1"})
    assert p.returncode == 0, p.stderr[-400:]
    out = json.loads(p.stdout)
    assert out.get("ok"), out
    return out["renders"]


def py_render(prog):
    import os
    p = subprocess.run([sys.executable, "py/dump_cstyle_render.py", json.dumps(prog)],
                       capture_output=True, text=True, env=dict(os.environ))
    assert p.returncode == 0, p.stderr[-400:]
    return json.loads(p.stdout)["renders"]


class TestRenderDiff(unittest.TestCase):
    def render_equality(self, name, prog):
        with self.subTest(name=name):
            rs = rs_render(prog)
            py = py_render(prog)
            self.assertEqual(len(rs), len(py), "render count differs")
            for i, (r, p) in enumerate(zip(rs, py)):
                if p is None:
                    self.assertIsNone(r, f"call {i}: expected copy")
                    continue
                self.assertEqual(r, p, f"call {i} render differs:\n--- rust ---\n{r}\n--- python ---\n{p}")

    def test_mul_kernel(self):
        self.render_equality("mul kernel", [
            {"op": "from_slice", "data": [1.0, 2.0, 3.0], "kind": "f32", "dtype": None},
            {"op": "mulc", "x": 0, "val": 2.0},
        ])

    def test_ew_kernel(self):
        self.render_equality("ew kernel", [
            {"op": "from_slice", "data": [1.0, 2.0, 3.0, 4.0], "kind": "f32", "dtype": None},
            {"op": "addc", "x": 0, "val": 1.0},
        ])


if __name__ == "__main__":
    unittest.main()
