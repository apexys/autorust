#!/usr/bin/env python3
"""Regenerate the ONNX-runner assets (`bench/assets/*.onnx`).

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
ASSETS = ROOT / "bench" / "assets"


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


def conv_bn(cin, cout, k=3, stride=1, pad=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, stride, pad, groups=groups, bias=False),
        nn.BatchNorm2d(cout),
    )


class Basic(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = conv_bn(cin, cout, 3, stride, 1)
        self.c2 = conv_bn(cout, cout, 3, 1, 1)
        self.down = conv_bn(cin, cout, 1, stride, 0) if stride != 1 or cin != cout else None

    def forward(self, x):
        idt = self.down(x) if self.down is not None else x
        y = torch.relu(self.c1(x))
        return torch.relu(self.c2(y) + idt)


class ResNet18(nn.Module):
    """ResNet-18 at a reduced width (the ONNX row stays a small asset)."""

    def __init__(self, width=0.25, classes=1000):
        super().__init__()

        def w(c):
            return max(8, int(c * width) // 8 * 8)

        self.stem = conv_bn(3, w(64), 7, 2, 3)
        stages, cin = [], w(64)
        for i, n in enumerate([2, 2, 2, 2]):
            cout = w(64 << i)
            blocks = []
            for b in range(n):
                blocks.append(Basic(cin, cout, 2 if i > 0 and b == 0 else 1))
                cin = cout
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.fc = nn.Linear(cin, classes)

    def forward(self, x):
        x = torch.relu(self.stem(x))
        x = torch.max_pool2d(x, 3, 2, 1)
        x = self.stages(x)
        return self.fc(x.mean((2, 3)))


def export(model, x, name):
    path = ASSETS / name
    ASSETS.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (x,),
        path,
        input_names=["x"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"wrote {path} ({path.stat().st_size} bytes)")


def main():
    torch.manual_seed(0)
    export(Mnist().eval(), torch.rand(32, 1, 28, 28), "mnist_bs32.onnx")
    export(ResNet18().eval(), torch.rand(8, 3, 224, 224), "resnet18w25_bs8.onnx")


if __name__ == "__main__":
    main()
