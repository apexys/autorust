#!/usr/bin/env python3
"""Regenerate `bench/assets/mnist_bs32.onnx` (the ONNX-runner row).

Requires `onnx` in the bench venv:

    uv pip install --python bench/.venv/bin/python onnx

The file is committed, so running the benchmark does not need torch's ONNX
exporter; this script exists so the asset can be rebuilt reproducibly.
"""

import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench" / "assets" / "mnist_bs32.onnx"


class Mnist(nn.Module):
    """`beautiful_mnist` (forward only)."""

    def __init__(self):
        super().__init__()
        self.l0 = nn.Conv2d(1, 32, 5)
        self.l1 = nn.Conv2d(32, 32, 5)
        self.l2 = nn.BatchNorm2d(32)
        self.l3 = nn.Conv2d(32, 64, 3)
        self.l4 = nn.Conv2d(64, 64, 3)
        self.l5 = nn.BatchNorm2d(64)
        self.l6 = nn.Linear(576, 10)

    def forward(self, x):
        x = F.relu(self.l0(x))
        x = F.relu(self.l1(x))
        x = self.l2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = self.l5(x)
        x = F.max_pool2d(x, 2)
        return self.l6(x.flatten(1))


def main():
    torch.manual_seed(0)
    model = Mnist().eval()
    x = torch.rand(32, 1, 28, 28)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (x,),
        OUT,
        input_names=["x"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
