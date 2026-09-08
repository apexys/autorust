"""M8 OCL gates: OpenCL render byte-diff + native CL numerics + kernel-count parity.

For each corpus program:
  1. render: the Rust oracle (TG_OCL_RENDER) renders each kernel's linearized
     uops with our OpenCLRenderer port; python renders the same linearized
     uops with its OpenCLRenderer. Outputs must be byte-equal strings.
  2. numerics: the Rust oracle runs the same corpus natively on CL (device
     strings "CL" route the buffers through the OpenCL runtime); the values
     must match tinygrad's own DEV=CL execution (allclose).
  3. kernel-count: the number of compiled runtimes on CL must match the
     number of compute-kernel calls in tinygrad's scheduled linear.

Requires: cargo build --features backend_cl --bin oracle_tensor and the RTX 3060.
"""
import sys, json, subprocess, unittest, os
sys.path.insert(0, "py"); sys.path.insert(0, "py/tests"); sys.path.insert(0, "../tinygrad")

ORACLE = "./target/debug/oracle_tensor"


def build_oracle():
    env = dict(os.environ)
    subprocess.run(["cargo", "build", "--features", "backend_cl", "--bin", "oracle_tensor"],
                   check=True, capture_output=True, env=env)


def rs_render(prog):
    p = subprocess.run([ORACLE], input=json.dumps({"prog": prog, "mode": "schedule"}),
                       capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin", "TG_OCL_RENDER": "1"})
    assert p.returncode == 0, p.stderr[-400:]
    out = json.loads(p.stdout)
    assert out.get("ok"), out
    return out["renders"]


def py_render(prog):
    p = subprocess.run([sys.executable, "py/dump_ocl_render.py", json.dumps(prog)],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-400:]
    return json.loads(p.stdout)["renders"]


def rs_native(prog):
    p = subprocess.run([ORACLE], input=json.dumps({"prog": prog, "mode": "native"}),
                       capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin"})
    assert p.returncode == 0, p.stderr[-400:]
    out = json.loads(p.stdout)
    assert out.get("ok"), out
    return out["outs"][-1], out["compiles"]


def py_cl_values(prog):
    from corpus import build
    tensors = build(prog, "CL")
    flat = tensors[-1].tolist()
    return [float(flat)] if isinstance(flat, float) else [float(v) for v in flat]


def py_cl_kernel_count(prog):
    from corpus import planned_last
    from tinygrad.uop.ops import Ops
    linear = planned_last(prog, "CL")
    return sum(1 for c in linear.src if c.src[0].op is not Ops.COPY)


PROGS = {
    "ew add": [
        {"op": "from_slice", "data": [1.0, 2.0, 3.0], "kind": "f32", "dtype": None, "device": "CL"},
        {"op": "addc", "x": 0, "val": 2.0},
    ],
    "ew mul chain": [
        {"op": "from_slice", "data": [1.0, 2.0, 3.0, 4.0], "kind": "f32", "dtype": None, "device": "CL"},
        {"op": "mulc", "x": 0, "val": 2.0},
        {"op": "addc", "x": 1, "val": 1.5},
    ],
    "two tensors add": [
        {"op": "from_slice", "data": [1.0, 2.0, 3.0], "kind": "f32", "dtype": None, "device": "CL"},
        {"op": "from_slice", "data": [10.0, 20.0, 30.0], "kind": "f32", "dtype": None, "device": "CL"},
        {"op": "add", "x": 0, "y": 1},
    ],
    "sum kernel": [
        {"op": "from_slice", "data": [1.0, 2.0, 3.0, 4.0], "kind": "f32", "dtype": None, "device": "CL"},
        {"op": "sum", "x": 0, "axis": None},
    ],
}


class TestOclDiff(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        build_oracle()

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

    def test_render_diff(self):
        for name, prog in PROGS.items():
            self.render_equality(name, prog)

    def test_native_cl_numerics(self):
        for name, prog in PROGS.items():
            with self.subTest(name=name):
                (rs, _compiles) = rs_native(prog)
                py = py_cl_values(prog)
                self.assertEqual(len(rs), len(py), "value count differs")
                for i, (a, b) in enumerate(zip(rs, py)):
                    self.assertAlmostEqual(a, b, places=5, msg=f"{name} idx {i}")

    def test_kernel_count_parity(self):
        for name, prog in PROGS.items():
            with self.subTest(name=name):
                (_rs, compiles) = rs_native(prog)
                py_kernels = py_cl_kernel_count(prog)
                self.assertEqual(compiles, py_kernels, f"{name}: kernel count differs")


if __name__ == "__main__":
    unittest.main()
