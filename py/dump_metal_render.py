"""Dump tinygrad's MetalRenderer render of the corpus kernels' linearized uops.

Usage: python dump_metal_render.py '<prog-json>'
Emits JSON: {"renders": ["<metal shading language code>", ...]} on stdout.
"""
import sys, json
sys.path.insert(0, "py"); sys.path.insert(0, "py/tests"); sys.path.insert(0, "../tinygrad")
from corpus import render_calls
from tinygrad.renderer.cstyle import MetalRenderer

prog = json.loads(sys.argv[1])
ren = MetalRenderer()
outs = render_calls(prog, ren)
print(json.dumps({"renders": outs}))
