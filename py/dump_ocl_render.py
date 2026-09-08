"""Dump tinygrad's OpenCLRenderer render of the corpus kernels' linearized uops.

Usage: python dump_ocl_render.py '<prog-json>'
Emits JSON: {"renders": ["<opencl c code>", ...]} on stdout (one per kernel call).
"""
import sys, json
sys.path.insert(0, "py"); sys.path.insert(0, "py/tests"); sys.path.insert(0, "../tinygrad")
from corpus import render_calls
from tinygrad.helpers import Target
from tinygrad.renderer.cstyle import OpenCLRenderer

prog = json.loads(sys.argv[1])
ren = OpenCLRenderer(Target(device="CL", arch="sm_86,cl_khr_fp64"))  # NVIDIA 3060 exts subset
outs = render_calls(prog, ren)
print(json.dumps({"renders": outs}))
