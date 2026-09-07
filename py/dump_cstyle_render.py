"""Dump python's cstyle render of the corpus kernels' linearized uops.

Usage: python dump_cstyle_render.py '<prog-json>'
Emits JSON: {"renders": ["<c code>", ...]} on stdout (one per kernel call).
"""
import sys, json, itertools
sys.path.insert(0, "py"); sys.path.insert(0, "../tinygrad")
from tinygrad.uop.ops import UOp, Ops, graph_rewrite
from tinygrad.tensor import transform_to_call, Tensor
from tinygrad.dtype import dtypes
from tinygrad.schedule import pm_schedule, pm_resolve_linear_call, pm_copy_from_store
from tinygrad.schedule.memory import memory_plan_rewrite
from tinygrad.codegen.__init__ import full_rewrite_to_sink, linearize, pm_linearize_cleanups, line_rewrite
from tinygrad.runtime.ops_python import PythonRenderer
from tinygrad.helpers import Target
from tinygrad.renderer.cstyle import ClangRenderer

prog = json.loads(sys.argv[1])
UOp.unique_num = itertools.count(0)
tensors = []
for st in prog:
    if st["op"] == "from_slice":
        tensors.append(Tensor(st["data"], dtype=dtypes.float32).contiguous())
    elif st["op"] == "addc":
        tensors.append(tensors[st["x"]] + st["val"])
    elif st["op"] == "mulc":
        tensors.append(tensors[st["x"]] * st["val"])
sink = UOp.sink(*[t.uop for t in tensors])
call, _ = transform_to_call(sink)
linear_call = graph_rewrite(call, pm_schedule, name="schedule to linear", enter_calls=True)
linear = graph_rewrite(linear_call, pm_resolve_linear_call, name="resolve linear call")
linear = graph_rewrite(linear, pm_copy_from_store, name="create COPY kernels for SDMA")
planned = memory_plan_rewrite(linear)

from dataclasses import replace
ren = ClangRenderer(replace(Target(device="CPU", arch="x86_64,znver2"), renderer="CLANG"))
outs = []
for c in planned.src:
    body = c.src[0]
    if body.op is Ops.COPY:
        outs.append(None)  # SDMA copy: no C render
        continue
    sunk = full_rewrite_to_sink(body, ren)
    lst = line_rewrite(linearize(sunk), pm_linearize_cleanups)
    outs.append(ren.render(lst))
print(json.dumps({"renders": outs}))
