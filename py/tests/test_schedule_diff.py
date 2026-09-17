"""M4 differential gate: Rust scheduler vs Python create_linear_with_vars.

For each program both sides build the same Tensor graph, then run:
  big_sink -> transform_to_call -> pm_schedule/create_schedule -> resolve -> copies
and compare the resulting LINEAR: kernel count + multiset of kernel-body keys
(the pre-memory-plan schedule; ordering may differ). The memory-planned LINEAR
is compared separately (TG_SCHED_PLANNED).
"""

import collections
import itertools
import json
import subprocess
import sys
import unittest

sys.path.insert(0, ".")
sys.path.insert(0, "py")
sys.path.insert(0, "py/tests")
sys.path.insert(0, "../tinygrad")

from bridge import json_to_uop  # noqa: E402

from tinygrad import Tensor, Device  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402
from tinygrad.tensor import transform_to_call  # noqa: E402
from tinygrad.schedule import pm_schedule, pm_resolve_linear_call, pm_copy_from_store  # noqa: E402
from tinygrad.uop.ops import UOp, graph_rewrite  # noqa: E402


def run_oracle(prog: list[dict], planned: bool = False) -> UOp | None:
    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    # Pin the oracle to tinygrad's device label: AST keys embed the device string,
    # and tinygrad defaults to e.g. "NV" (not the port's INTERP/CL default).
    env["DEFAULT_DEVICE"] = Device.DEFAULT
    if planned:
        env["TG_SCHED_PLANNED"] = "1"
    p = subprocess.run(["./target/debug/oracle_tensor"], input=json.dumps({"prog": prog, "mode": "schedule"}),
                       capture_output=True, text=True, env=env)
    if p.returncode != 0:
        print(f"  ORACLE rc={p.returncode} stderr={p.stderr!r:.400} stdout={p.stdout!r:.200}")
        return None
    return json_to_uop(json.dumps(json.loads(p.stdout)["graph"]))[0]


def run_oracle_last(prog: list[dict]) -> UOp | None:
    """Schedule only the final tensor (what `Tensor.realize` does), so intermediate
    graph nodes can be fused/materialized by the scheduler."""
    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    env["DEFAULT_DEVICE"] = Device.DEFAULT
    p = subprocess.run(["./target/debug/oracle_tensor"], input=json.dumps({"prog": prog, "mode": "schedule_last"}),
                       capture_output=True, text=True, env=env)
    if p.returncode != 0:
        print(f"  ORACLE rc={p.returncode} stderr={p.stderr!r:.400} stdout={p.stdout!r:.200}")
        return None
    return json_to_uop(json.dumps(json.loads(p.stdout)["graph"]))[0]


def py_schedule(tensors: list[Tensor]) -> UOp:
    big_sink = UOp.sink(*[t.uop for t in tensors])
    call, _ = transform_to_call(big_sink)
    linear_call = graph_rewrite(call, pm_schedule, name="schedule to linear", enter_calls=True)
    linear = graph_rewrite(linear_call, pm_resolve_linear_call, name="resolve linear call")
    linear = graph_rewrite(linear, pm_copy_from_store, name="create COPY kernels for SDMA")
    return linear


def kernel_bodies(linear: UOp) -> list[bytes]:
    out = []
    for c in linear.src:
        if len(c.src):
            out.append(c.src[0].key)
        else:
            out.append(c.key)
    return out


def ix_spec(int=None, newaxis=False, ellipsis=False, slice=None):
    return {"int": int, "newaxis": newaxis, "ellipsis": ellipsis, "slice": slice}


def slice_prog(start, stop, step):
    return ix_spec(slice=[start, stop, step])


class TestScheduleDiff(unittest.TestCase):
    def schedule_key_equality(self, name: str, prog: list[dict]):
        with self.subTest(name=name):
            UOp.unique_num = itertools.count(0)
            py_tensors = py_build(prog)
            py_lin = py_schedule(py_tensors)
            rust_lin = run_oracle(prog)
            if rust_lin is None:
                self.fail("oracle failed")
            pb, rb = kernel_bodies(py_lin), kernel_bodies(rust_lin)
            if len(pb) != len(rb) or collections.Counter(pb) != collections.Counter(rb):
                print(f"  PY {len(pb)} kernels: {[b.hex()[:12] for b in pb]}")
                print(f"  RS {len(rb)} kernels: {[b.hex()[:12] for b in rb]}")
                print("  rust graph op:", rust_lin.op, "srcs:", [s.op for s in rust_lin.src])
                print("  py graph op:", py_lin.op, "srcs:", [s.op for s in py_lin.src])
            self.assertEqual(len(pb), len(rb), f"kernel count {len(pb)} != {len(rb)}")
            self.assertEqual(collections.Counter(pb), collections.Counter(rb), "kernel AST keys differ")

    def planned_key_equality(self, name: str, prog: list[dict]):
        with self.subTest(name=name + " planned"):
            from tinygrad.schedule.memory import memory_plan_rewrite
            UOp.unique_num = itertools.count(0)
            py_tensors = py_build(prog)
            py_lin = py_schedule(py_tensors)
            py_plan = memory_plan_rewrite(py_lin, set())
            rust_plan = run_oracle(prog, planned=True)
            if rust_plan is None:
                self.fail("oracle failed")
            pb, rb = kernel_bodies(py_plan), kernel_bodies(rust_plan)
            self.assertEqual(len(pb), len(rb), f"kernel count {len(pb)} != {len(rb)}")
            self.assertEqual(collections.Counter(pb), collections.Counter(rb), "planned kernel AST keys differ")

    def schedule_last_key_equality(self, name: str, prog: list[dict]):
        """Realize only the last tensor (as `Tensor.realize` does) and compare the
        resulting schedules. The all-outputs schedule can hide fusions of internal
        nodes, e.g. a reduce that feeds a broadcast then another reduce."""
        with self.subTest(name=name + " last"):
            UOp.unique_num = itertools.count(0)
            py_tensors = py_build(prog)
            py_lin = py_schedule([py_tensors[-1]])
            rust_lin = run_oracle_last(prog)
            if rust_lin is None:
                self.fail("oracle failed")
            pb, rb = kernel_bodies(py_lin), kernel_bodies(rust_lin)
            if len(pb) != len(rb) or collections.Counter(pb) != collections.Counter(rb):
                print(f"  PY {len(pb)} kernels: {[b.hex()[:12] for b in pb]}")
                print(f"  RS {len(rb)} kernels: {[b.hex()[:12] for b in rb]}")
            self.assertEqual(len(pb), len(rb), f"kernel count {len(pb)} != {len(rb)}")
            self.assertEqual(collections.Counter(pb), collections.Counter(rb), "last kernel AST keys differ")


def py_build(prog: list[dict]) -> list[Tensor]:
    ts: list[Tensor] = []
    for st in prog:
        op = st["op"]
        if op == "from_slice":
            kind, data = st["kind"], st["data"]
            dt = st.get("dtype")
            ctor = {"f32": dtypes.float32, "f64": dtypes.float64, "i32": dtypes.int32, "i64": dtypes.int64,
                    "u8": dtypes.uint8, "bool": dtypes.bool}
            t = Tensor(data, dtype=ctor[kind] if dt is None else dtypes.from_py(0, dt) if False else None)
            if dt is not None:
                t = Tensor(data, dtype={"float32": dtypes.float32, "float64": dtypes.float64, "int32": dtypes.int32,
                                        "int64": dtypes.int64, "uint8": dtypes.uint8, "bool": dtypes.bool}.get(dt, None) or ctor[kind])
            ts.append(t)
        elif op == "arange":
            import math
            start, stop, step = st["start"], st["stop"], st["step"]
            n = max(0, math.ceil((stop - start) / step))
            ts.append(Tensor.arange(start, start + n * step, step))
        elif op == "ones":
            ts.append(Tensor.ones(*st["shape"]))
        elif op == "zeros":
            ts.append(Tensor.zeros(*st["shape"]))
        elif op == "reshape":
            ts.append(ts[st["x"]].reshape(st["shape"]))
        elif op == "getitem":
            idxs = []
            for ix in st["ixs"]:
                if ix["int"] is not None:
                    idxs.append(ix["int"])
                elif ix["newaxis"]:
                    idxs.append(None)
                elif ix["ellipsis"]:
                    idxs.append(Ellipsis)
                else:
                    s = ix["slice"]
                    idxs.append(slice(s[0], s[1], s[2] if s[2] is not None else 1))
            ts.append(ts[st["x"]][tuple(idxs)] if len(idxs) > 1 else ts[st["x"]][idxs[0]])
        elif op == "add":
            ts.append(ts[st["x"]] + ts[st["y"]])
        elif op == "mul":
            ts.append(ts[st["x"]] * ts[st["y"]])
        elif op == "addc":
            ts.append(ts[st["x"]] + st["val"])
        elif op == "mulc":
            ts.append(ts[st["x"]] * st["val"])
        elif op == "neg":
            ts.append(-ts[st["x"]])
        elif op == "sum":
            ax = st.get("axis")
            ts.append(ts[st["x"]].sum(ax if ax is None else tuple(ax), keepdim=st.get("keepdim", False)))
        elif op == "max":
            ax = st.get("axis")
            ts.append(ts[st["x"]].max(ax if ax is None else tuple(ax), keepdim=st.get("keepdim", False)))
        elif op == "contiguous":
            ts.append(ts[st["x"]].contiguous())
        else:
            raise AssertionError(f"unknown program op {op}")
    return ts


class TestSchedule(TestScheduleDiff):
    def test_reduce_simple(self):
        self.schedule_key_equality("reduce simple", [
            {"op": "arange", "start": 0, "stop": 6, "step": 1, "dtype": None},
            {"op": "sum", "x": 0, "axis": [0], "keepdim": False},
        ])

    def test_ew_on_copy(self):
        self.schedule_key_equality("ew on copy", [
            {"op": "from_slice", "data": [1.0, 2.0, 3.0], "kind": "f32", "dtype": None},
            {"op": "addc", "x": 0, "val": 1.0},
        ])

    def test_ew_reduce_chain(self):
        self.schedule_key_equality("ew reduce chain", [
            {"op": "from_slice", "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "kind": "f32", "dtype": None},
            {"op": "reshape", "x": 0, "shape": [2, 3]},
            {"op": "mulc", "x": 1, "val": 2.0},
            {"op": "sum", "x": 2, "axis": [1], "keepdim": False},
        ])

    def test_reduce_broadcast_reduce(self):
        prog = [
            {"op": "from_slice", "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "kind": "f32", "dtype": None},
            {"op": "reshape", "x": 0, "shape": [2, 1, 3]},
            {"op": "from_slice", "data": [1.0, 0.5, -1.0, 2.0, 1.0, 0.5, -0.5, 3.0, 1.5, 0.25, -2.0, 1.0], "kind": "f32", "dtype": None},
            {"op": "reshape", "x": 2, "shape": [1, 4, 3]},
            {"op": "mul", "x": 1, "y": 3},
            {"op": "sum", "x": 4, "axis": [2], "keepdim": False},
            {"op": "reshape", "x": 5, "shape": [2, 4, 1]},
            {"op": "mul", "x": 4, "y": 6},
            {"op": "sum", "x": 7, "axis": [1], "keepdim": False},
        ]
        self.schedule_key_equality("reduce broadcast reduce", prog)
        self.schedule_last_key_equality("reduce broadcast reduce", prog)

    def test_contiguous_realize(self):
        self.schedule_key_equality("contiguous realize", [
            {"op": "arange", "start": 0, "stop": 12, "step": 1, "dtype": None},
            {"op": "reshape", "x": 0, "shape": [3, 4]},
            {"op": "getitem", "x": 1, "ixs": [slice_prog(None, None, 2)]},
            {"op": "contiguous", "x": 2},
            {"op": "sum", "x": 3, "axis": [0], "keepdim": False},
        ])

    def test_two_outputs(self):
        self.schedule_key_equality("two outputs", [
            {"op": "from_slice", "data": [1.0, 2.0, 3.0, 4.0], "kind": "f32", "dtype": None},
            {"op": "addc", "x": 0, "val": 1.0},
            {"op": "mulc", "x": 1, "val": 3.0},
        ])

    def test_reduce_simple_planned(self):
        self.planned_key_equality("reduce simple", [
            {"op": "arange", "start": 0, "stop": 6, "step": 1, "dtype": None},
            {"op": "sum", "x": 0, "axis": [0], "keepdim": False},
        ])


if __name__ == "__main__":
    unittest.main()
