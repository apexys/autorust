"""Dump python's CUDARenderer render of the corpus kernels' linearized uops.

Usage: python dump_cuda_render.py '<prog-json>'
Emits JSON: {"renders": ["<cuda c>", ...]} on stdout (one per kernel call).
"""
import sys, json, itertools
sys.path.insert(0, "py"); sys.path.insert(0, "../tinygrad")
from tinygrad.uop.ops import UOp, Ops, graph_rewrite
from tinygrad.tensor import transform_to_call, Tensor
from tinygrad.dtype import dtypes
from tinygrad.schedule import pm_schedule, pm_resolve_linear_call, pm_copy_from_store
from tinygrad.schedule.memory import memory_plan_rewrite
from tinygrad.codegen.__init__ import full_rewrite_to_sink, linearize, pm_linearize_cleanups, line_rewrite
from tinygrad.helpers import Target
from tinygrad.renderer.cstyle import CUDARenderer

DT = {"float32": dtypes.float32, "f32": dtypes.float32, "float16": dtypes.float16, "f16": dtypes.float16,
      "bfloat16": dtypes.bfloat16, "bf16": dtypes.bfloat16, "float64": dtypes.float64, "f64": dtypes.float64,
      "int32": dtypes.int32, "i32": dtypes.int32, "uint8": dtypes.uint8, "u8": dtypes.uint8}

prog = json.loads(sys.argv[1])
UOp.unique_num = itertools.count(0)
tensors = []
for st in prog:
    op = st["op"]
    if op == "from_slice":
        tensors.append(Tensor(st["data"], dtype=DT.get(st.get("dtype") or st.get("kind") or "float32", dtypes.float32)).contiguous())
    elif op == "cast":
        tensors.append(tensors[st["x"]].cast(DT[st["dtype"]]))
    elif op == "addc":
        tensors.append(tensors[st["x"]] + st["val"])
    elif op == "mulc":
        tensors.append(tensors[st["x"]] * st["val"])
    elif op == "add":
        tensors.append(tensors[st["x"]] + tensors[st["y"]])
    elif op == "mul":
        tensors.append(tensors[st["x"]] * tensors[st["y"]])
    elif op == "matmul":
        tensors.append(tensors[st["x"]] @ tensors[st["y"]])
    elif op == "reshape":
        tensors.append(tensors[st["x"]].reshape(st["shape"]))
    elif op == "rand":
        tensors.append(Tensor.rand(*st["shape"], dtype=DT.get(st.get("dtype"), dtypes.float32), device="CUDA"))
    elif op == "neg":
        tensors.append(-tensors[st["x"]])
    else:
        raise SystemExit(f"dump_cuda_render: unsupported op {op}")

sink = UOp.sink(*[t.uop for t in tensors])
call, _ = transform_to_call(sink)
linear_call = graph_rewrite(call, pm_schedule, name="schedule to linear", enter_calls=True)
linear = graph_rewrite(linear_call, pm_resolve_linear_call, name="resolve linear call")
linear = graph_rewrite(linear, pm_copy_from_store, name="create COPY kernels for SDMA")
planned = memory_plan_rewrite(linear)

from dataclasses import replace
ren = CUDARenderer(replace(Target(device="CUDA", arch="sm_86"), renderer="CUDA"))
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
