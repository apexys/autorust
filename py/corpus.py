"""Shared corpus builder for the OCL gates (mirrors py/exec_steps semantics)."""
import sys
sys.path.insert(0, "py"); sys.path.insert(0, "py/tests"); sys.path.insert(0, "../tinygrad")

import itertools
from tinygrad import Tensor
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp, Ops, graph_rewrite
from tinygrad.tensor import transform_to_call
from tinygrad.schedule import pm_schedule, pm_resolve_linear_call, pm_copy_from_store
from tinygrad.schedule.memory import memory_plan_rewrite
from tinygrad.codegen.__init__ import full_rewrite_to_sink, linearize, pm_linearize_cleanups, line_rewrite


def build(prog, device_default=None):
    tensors = []
    for st in prog:
        dev = st.get("device", device_default)
        if st["op"] == "from_slice":
            tensors.append(Tensor(st["data"], dtype=dtypes.float32, device=dev).contiguous())
        elif st["op"] == "addc":
            tensors.append(tensors[st["x"]] + st["val"])
        elif st["op"] == "mulc":
            tensors.append(tensors[st["x"]] * st["val"])
        elif st["op"] == "add":
            tensors.append(tensors[st["x"]] + tensors[st["y"]])
        elif st["op"] == "mul":
            tensors.append(tensors[st["x"]] * tensors[st["y"]])
        elif st["op"] == "sum":
            axis = tuple(st["axis"]) if st.get("axis") else None
            tensors.append(tensors[st["x"]].sum(axis=axis))
        else:
            raise ValueError(f"unknown step {st['op']}")
    return tensors


def planned_linear(prog, device_default=None):
    tensors = build(prog, device_default)
    sink = UOp.sink(*[t.uop for t in tensors])
    call, _ = transform_to_call(sink)
    linear = graph_rewrite(call, pm_schedule, name="schedule to linear", enter_calls=True)
    linear = graph_rewrite(linear, pm_resolve_linear_call, name="resolve linear call")
    linear = graph_rewrite(linear, pm_copy_from_store, name="create COPY kernels for SDMA")
    return memory_plan_rewrite(linear)


def render_calls(prog, ren, device_default=None):
    linear = planned_linear(prog, device_default)
    outs = []
    for c in linear.src:
        body = c.src[0]
        if body.op is Ops.COPY:
            outs.append(None)  # SDMA copy: no C render
            continue
        sunk = full_rewrite_to_sink(body, ren)
        lst = line_rewrite(linearize(sunk), pm_linearize_cleanups)
        outs.append(ren.render(lst))
    return outs


def planned_last(prog, device_default=None):
    """Schedule realizing only the final tensor (the native realize semantics)."""
    tensors = build(prog, device_default)
    sink = UOp.sink(tensors[-1].uop)
    call, _ = transform_to_call(sink)
    linear = graph_rewrite(call, pm_schedule, name="schedule to linear", enter_calls=True)
    linear = graph_rewrite(linear, pm_resolve_linear_call, name="resolve linear call")
    linear = graph_rewrite(linear, pm_copy_from_store, name="create COPY kernels for SDMA")
    return memory_plan_rewrite(linear)
