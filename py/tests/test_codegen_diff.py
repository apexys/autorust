"""M5 differential gate (in progress): per-stage key equality of the codegen
pipeline vs python's full_rewrite_to_sink.

Compares the Rust pipeline output (via the bridge) against python's
"postopt symbolic" stage keys for corpus kernel ASTs (the pipeline covers
early movement ops .. apply_opts .. postopt symbolic). Later python stages
append as the Rust pipeline grows (see CURRENT_STATE.md §5).
"""
import sys, json, itertools, subprocess, unittest
sys.path.insert(0, "py"); sys.path.insert(0, "py/tests"); sys.path.insert(0, "../tinygrad")
from bridge import json_to_uop
from test_schedule_diff import py_build
from tinygrad.uop.ops import UOp, Ops, graph_rewrite
from tinygrad.tensor import transform_to_call
from tinygrad.schedule import pm_schedule, pm_resolve_linear_call, pm_copy_from_store

ORACLE = "./target/debug/oracle_tensor"


def rs_codegen(prog):
    p = subprocess.run([ORACLE], input=json.dumps({"prog": prog, "mode": "schedule"}),
                       capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin", "TG_M5_CODEGEN": "1"})
    if p.returncode != 0:
        return None
    return json_to_uop(json.dumps(json.loads(p.stdout)["graph"]))[0]


def py_stage_keys(prog, stage="number params with -1"):
    """python's per-stage toposort key multiset for the first compute kernel AST
    (subprocess-isolated: python's structural UOp cache leaks state across programs)."""
    import os, subprocess
    env = dict(os.environ)
    p = subprocess.run([sys.executable, "py/dump_codegen_stages.py", json.dumps(prog)],
                       capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stderr[-400:]
    d = json.loads(p.stdout)
    for st in d["stages"]:
        if st["name"] == stage:
            return [bytes.fromhex(k) for k in st["keys"]]
    raise AssertionError(f"stage {stage} not found")


class TestCodegenDiff(unittest.TestCase):
    def stage_key_equality(self, name, prog):
        with self.subTest(name=name):
            UOp.unique_num = itertools.count(0)
            py_keys = py_stage_keys(prog)
            rs_u = rs_codegen(prog)
            if rs_u is None:
                self.fail("oracle failed")
            rs_keys = sorted(u.key for u in rs_u.toposort())
            self.assertEqual(py_keys, rs_keys, "codegen stage keys differ")

    def test_ew_kernel(self):
        self.stage_key_equality("ew kernel", [
            {"op": "from_slice", "data": [1.0, 2.0, 3.0, 4.0], "kind": "f32", "dtype": None},
            {"op": "addc", "x": 0, "val": 1.0},
        ])

    def test_mul_kernel(self):
        self.stage_key_equality("mul kernel", [
            {"op": "from_slice", "data": [1.0, 2.0, 3.0], "kind": "f32", "dtype": None},
            {"op": "mulc", "x": 0, "val": 2.0},
        ])


if __name__ == "__main__":
    unittest.main()
