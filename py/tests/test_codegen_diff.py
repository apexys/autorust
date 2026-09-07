"""M5 differential gate (in progress): per-stage key equality of the codegen
pipeline vs python's full_rewrite_to_sink.

Compares the Rust pipeline output (via the bridge) against python's
"early movement ops" stage keys for corpus kernel ASTs. Later python stages
append as the Rust pipeline grows (see PLAN.md §M5).
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


def py_stage_keys(prog, stage="early movement ops"):
    """python's per-stage toposort key multiset for the first compute kernel AST."""
    import tinygrad.codegen.__init__ as CI
    orig = CI.graph_rewrite
    out = {}
    def gw2(sink, pm, ctx=None, bottom_up=False, name=None, bpm=None, walk=False, enter_calls=False):
        ret = orig(sink, pm, ctx=ctx, bottom_up=bottom_up, name=name, bpm=bpm, walk=walk, enter_calls=enter_calls)
        if name is not None and name not in out:
            out[name] = sorted(u.key for u in ret.toposort())
        return ret
    CI.graph_rewrite = gw2
    try:
        ts = py_build(prog)
        call, _ = transform_to_call(UOp.sink(*[t.uop for t in ts]))
        linear_call = orig(call, pm_schedule, enter_calls=True, name="schedule to linear")
        lin = orig(linear_call, pm_resolve_linear_call, name="resolve linear call")
        lin = orig(lin, pm_copy_from_store, name="create COPY kernels for SDMA")
        ast = None
        for c in lin.src:
            body = c.src[0]
            if body.op is Ops.SINK and any(x.op in (Ops.ADD, Ops.MUL) for x in body.toposort(gate=lambda u: u.op is not Ops.CALL)):
                ast = body
                break
        assert ast is not None
        from tinygrad.runtime.ops_python import PythonRenderer
        ren = PythonRenderer.__new__(PythonRenderer)
        from tinygrad.helpers import Target
        PythonRenderer.__init__(ren, Target.parse("cpu"))
        stages = []
        CI.full_rewrite_to_sink(ast, ren)
        return out.get(stage)
    finally:
        CI.graph_rewrite = orig


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
