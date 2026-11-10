#!/usr/bin/env python3
"""PyTorch side of the cross-framework benchmark.

Same models and shapes as bench/autorust: the `beautiful_mnist` CNN and a
ViT-B/16 (12 layers, 768 wide, 12 heads, 224 px). Synthetic data stays on the
device; only the training/inference step is timed.

Run with bench/.venv/bin/python. The final line on stdout is `RESULT <json>`.
"""

import argparse
import json
import os
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


class Mnist(nn.Module):
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


class Block(nn.Module):
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ff1 = nn.Linear(dim, 4 * dim)
        self.ff2 = nn.Linear(4 * dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        b, n, d = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        shape = (b, n, self.heads, d // self.heads)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v)
        a = a.transpose(1, 2).reshape(b, n, d)
        x = x + self.drop(self.out(a))
        h = self.ln2(x)
        x = x + self.drop(self.ff2(F.gelu(self.ff1(h))))
        return x


class Vit(nn.Module):
    def __init__(self, layers=12, dim=768, heads=12, img=224, patch=16, classes=1000):
        super().__init__()
        side = img // patch
        tokens = side * side
        self.embed = nn.Conv2d(3, dim, patch, patch)
        self.cls = nn.Parameter(torch.ones(1, 1, dim))
        self.pos = nn.Parameter(torch.ones(1, tokens + 1, dim))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, classes)

    def forward(self, x):
        x = self.embed(x).flatten(2).transpose(1, 2)
        cls = self.cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])


def build(workload, batch, dim=768, layers=12, heads=12, img=224):
    if workload == "mnist_train":
        model = Mnist()
        x = torch.rand(batch, 1, 28, 28)
        y = torch.randint(0, 10, (batch,))
        return model, x, y
    model = Vit(layers=layers, dim=dim, heads=heads, img=img)
    x = torch.zeros(batch, 3, img, img)
    y = torch.randint(0, 1000, (batch,))
    return model, x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="mnist_train",
                    choices=["mnist_train", "vit_infer", "vit_train"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(env_int("BENCH_THREADS", os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device(args.device)
    torch.manual_seed(0)

    training = args.workload.endswith("train")
    batch = args.batch if args.batch is not None else (
        512 if args.workload == "mnist_train" else (1 if args.workload == "vit_train" else 8))
    warmup = args.warmup if args.warmup is not None else (5 if training else 3)
    iters = args.iters if args.iters is not None else (30 if args.workload == "mnist_train" else 5)

    model, x, y = build(args.workload, batch, args.dim, args.layers, args.heads, args.img)
    model = model.to(dev)
    x, y = x.to(dev), y.to(dev)
    model.train(training)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3 if args.workload == "mnist_train" else 3e-4)
    if args.compile:
        model = torch.compile(model)

    def step():
        if training:
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            opt.step()
            return float(loss.detach())
        out = model(x)
        # Force completion and the same host copy autorust makes.
        out.detach().cpu()
        return None

    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()

    sync()
    t = time.perf_counter()
    step()
    sync()
    cold = (time.perf_counter() - t) * 1e3
    for _ in range(warmup):
        step()
    sync()
    samples = []
    for _ in range(iters):
        sync()
        t = time.perf_counter()
        step()
        sync()
        samples.append((time.perf_counter() - t) * 1e3)

    samples.sort()
    n = len(samples)
    median = samples[n // 2] if n % 2 else 0.5 * (samples[n // 2 - 1] + samples[n // 2])
    q = lambda p: samples[round((n - 1) * p)]
    result = {
        "framework": "torch",
        "version": torch.__version__,
        "device": args.device.upper(),
        "workload": args.workload,
        "batch": batch,
        "dtype": "f32",
        "jit": bool(args.compile),
        "warmup": warmup,
        "iters": iters,
        "ms_cold": round(cold, 4),
        "ms_median": round(median, 4),
        "ms_min": round(samples[0], 4),
        "ms_iqr": round(q(0.75) - q(0.25), 4),
        "images_per_s": round(batch * 1e3 / median, 2),
    }
    print(f"{args.workload} {args.device.upper()} bs={batch} compile={args.compile}: "
          f"{median:.3f} ms/step (min {samples[0]:.3f}, iqr {result['ms_iqr']:.3f}, "
          f"cold {cold:.1f} ms, {result['images_per_s']:.0f} img/s)")
    print("RESULT " + json.dumps(result))


if __name__ == "__main__":
    main()
