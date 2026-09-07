"""Dump per-stage UOp key multisets for to_program's full_rewrite_to_sink.

Usage: python dump_codegen_stages.py
Emits JSON: {"stages": [{"name": ..., "keys": [...]}, ...]} on stdout for a fixed corpus.
"""
import sys, itertools, json, pickle, base64
sys.path.insert(0, "py"); sys.path.insert(0, "py/tests"); sys.path.insert(0, "../tinygrad")
from test_schedule_diff import py_build
from tinygrad.uop.ops import UOp, Ops, graph_rewrite
from tinygrad.tensor import transform_to_call
from tinygrad.schedule import pm_schedule, pm_resolve_linear_call, pm_copy_from_store
from tinygrad.schedule.rangeify import get_kernel_graph
from tinygrad.codegen.__init__ import full_rewrite_to_sink, _broadcast_shape
import tinygrad.codegen.__init__ as C
from tinygrad.renderer import Renderer
from tinygrad.dtype import dtypes
from tinygrad.runtime.ops_python import PythonRenderer

# monkeypatch graph_rewrite to trace stages inside full_rewrite_to_sink
import tinygrad.helpers as H
stages: list[tuple[str, list[bytes]]] = []

real_gw = graph_rewrite
GW_STACK = []

def tracing_gw(sink, pm, ctx=None, bottom_up=False, name=None, bpm=None, walk=False, enter_calls=False):
    ret = real_gw(sink, pm, ctx=ctx, bottom_up=bottom_up, name=name, bpm=bpm, walk=walk, enter_calls=enter_calls)
    if GW_STACK and name is not None:
        stages.append((name, sorted(u.key for u in ret.toposort())))
    return ret

# instead of tracing graph_rewrite globally (too noisy), snapshot at source points:
# we re-implement the stage list by calling real graph_rewrite but with a name filter
import tinygrad.codegen.__init__ as CI
orig = CI.graph_rewrite
seen_names = set()
def gw2(sink, pm, ctx=None, bottom_up=False, name=None, bpm=None, walk=False, enter_calls=False):
    ret = orig(sink, pm, ctx=ctx, bottom_up=bottom_up, name=name, bpm=bpm, walk=walk, enter_calls=enter_calls)
    if name is not None and name not in seen_names:
        seen_names.add(name)
    if name is not None:
        stages.append((name, sorted(u.key for u in ret.toposort())))
    return ret
CI.graph_rewrite = gw2

UOp.unique_num = itertools.count(0)
prog = [
    {"op": "from_slice", "data": [1.0, 2.0, 3.0, 4.0], "kind": "f32", "dtype": None},
    {"op": "addc", "x": 0, "val": 1.0},
]
ts = py_build(prog)
call, _ = transform_to_call(UOp.sink(*[t.uop for t in ts]))
linear_call = orig(call, pm_schedule, enter_calls=True, name="schedule to linear")
lin = orig(linear_call, pm_resolve_linear_call, name="resolve linear call")
lin = orig(lin, pm_copy_from_store, name="create COPY kernels for SDMA")

# take the first compute kernel body as the AST
ast = None
for c in lin.src:
    body = c.src[0]
    if body.op is Ops.SINK and any(x.op in (Ops.ADD, Ops.MUL) for x in body.toposort(gate=lambda u: u.op is not Ops.CALL)):
        ast = body
        break
assert ast is not None, "no compute kernel found"

ren = PythonRenderer.__new__(PythonRenderer)
from tinygrad.helpers import Target
PythonRenderer.__init__(ren, Target.parse("cpu"))

stages.clear()
full = CI.full_rewrite_to_sink(ast, ren)

out = {"stages": [{"name": n, "keys": [k.hex() for k in ks]} for n, ks in stages]}
out["final_keys"] = [k.hex() for k in sorted(u.key for u in full.toposort())]
print(json.dumps(out, indent=0))
