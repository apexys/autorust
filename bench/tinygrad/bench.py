#!/usr/bin/env python3
"""tinygrad side of the cross-framework benchmark.

Run with the tinygrad checkout's venv:
  ../tinygrad/.venv/bin/python bench/tinygrad/bench.py --device cuda ...

Same three workloads as bench/autorust and bench/torch. `TinyJit` is used for
the steady-state runs, matching autorust's `Jit`; no `@function` transform, so
the graph is the plain module forward in every framework. The final line on
stdout is `RESULT <json>`.
"""

import os
import sys

# Parse --device before tinygrad creates its ContextVars (DEV is read at import).
_DEV = "CPU"
for _i, _a in enumerate(sys.argv):
    if _a == "--device" and _i + 1 < len(sys.argv):
        _DEV = sys.argv[_i + 1].upper()
    elif _a.startswith("--device="):
        _DEV = _a.split("=", 1)[1].upper()
os.environ["DEV"] = _DEV

import argparse
import contextlib
import json
import time

from tinygrad import Context, Tensor, TinyJit, nn


class Mnist:
    def __init__(self):
        self.l0 = nn.Conv2d(1, 32, 5)
        self.l1 = nn.Conv2d(32, 32, 5)
        self.l2 = nn.BatchNorm(32)
        self.l3 = nn.Conv2d(32, 64, 3)
        self.l4 = nn.Conv2d(64, 64, 3)
        self.l5 = nn.BatchNorm(64)
        self.l6 = nn.Linear(576, 10)

    def __call__(self, x):
        x = self.l0(x).relu()
        x = self.l1(x).relu()
        x = self.l2(x)
        x = x.max_pool2d()
        x = self.l3(x).relu()
        x = self.l4(x).relu()
        x = self.l5(x)
        x = x.max_pool2d()
        return self.l6(x.flatten(1))


class Block:
    """`extra/models/transformer.py` (prenorm), inlined to avoid its numpy import."""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        self.num_heads = num_heads
        self.head_size = embed_dim // num_heads
        self.dropout = dropout
        self.query = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))
        self.key = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))
        self.value = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))
        self.out = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))
        self.ff1 = (Tensor.scaled_uniform(embed_dim, embed_dim * 4), Tensor.zeros(embed_dim * 4))
        self.ff2 = (Tensor.scaled_uniform(embed_dim * 4, embed_dim), Tensor.zeros(embed_dim))
        self.ln1 = (Tensor.ones(embed_dim), Tensor.zeros(embed_dim))
        self.ln2 = (Tensor.ones(embed_dim), Tensor.zeros(embed_dim))

    def attn(self, x):
        q, k, v = [x.linear(*y).reshape(shape=(x.shape[0], -1, self.num_heads, self.head_size))
                   .transpose(1, 2) for y in [self.query, self.key, self.value]]
        a = Tensor.scaled_dot_product_attention(q, k, v).transpose(1, 2)
        return a.reshape(shape=(x.shape[0], -1, self.num_heads * self.head_size)).linear(*self.out)

    def __call__(self, x):
        x = x + self.attn(x.layernorm().linear(*self.ln1)).dropout(self.dropout)
        x = x + x.layernorm().linear(*self.ln2).linear(*self.ff1).gelu().linear(*self.ff2).dropout(self.dropout)
        return x


class ViT:
    """`extra/models/vit.py` (224 px, patch 16, 1000 classes), inlined."""

    def __init__(self, layers=12, embed_dim=768, num_heads=12):
        self.embedding = (Tensor.uniform(embed_dim, 3, 16, 16), Tensor.zeros(embed_dim))
        self.embed_dim = embed_dim
        self.cls = Tensor.ones(1, 1, embed_dim)
        self.pos_embedding = Tensor.ones(1, 197, embed_dim)
        self.tbs = [Block(embed_dim, num_heads) for _ in range(layers)]
        self.encoder_norm = (Tensor.uniform(embed_dim), Tensor.zeros(embed_dim))
        self.head = (Tensor.uniform(embed_dim, 1000), Tensor.zeros(1000))

    def forward(self, x):
        ce = self.cls.add(Tensor.zeros(x.shape[0], 1, 1))
        pe = x.conv2d(*self.embedding, stride=16)
        pe = pe.reshape(shape=(pe.shape[0], pe.shape[1], -1)).permute(order=(0, 2, 1))
        x = ce.cat(pe, dim=1).add(self.pos_embedding).sequential(self.tbs)
        x = x.layernorm().linear(*self.encoder_norm)
        return x[:, 0].linear(*self.head)

    def __call__(self, x):
        return self.forward(x)


def build(workload, batch, dim=768, layers=12, heads=12, img=224):
    if workload == "mnist_train":
        model = Mnist()
        x = Tensor.rand(batch, 1, 28, 28).realize()
        y = Tensor.randint(batch, low=0, high=10).realize()
        return model, x, y
    model = ViT(layers=layers, embed_dim=dim, num_heads=heads)
    x = Tensor.zeros(batch, 3, img, img).realize()
    y = Tensor.randint(batch, low=0, high=1000).realize()
    return model, x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="mnist_train",
                    choices=["mnist_train", "vit_infer", "vit_train"])
    ap.add_argument("--device", default=_DEV.lower(), choices=["cpu", "cuda"])
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--img", type=int, default=224)
    args = ap.parse_args()

    training = args.workload.endswith("train")
    batch = args.batch if args.batch is not None else (
        512 if args.workload == "mnist_train" else (1 if args.workload == "vit_train" else 8))
    warmup = args.warmup if args.warmup is not None else (5 if training else 3)
    iters = args.iters if args.iters is not None else (30 if args.workload == "mnist_train" else 5)
    lr = 1e-3 if args.workload == "mnist_train" else 3e-4

    model, x, y = build(args.workload, batch, args.dim, args.layers, args.heads, args.img)
    opt = nn.optim.Adam(nn.state.get_parameters(model), lr=lr)
    params = nn.state.get_parameters(model)

    with (Context(TRAINING=1) if training else contextlib.nullcontext()):
        if training:
            @TinyJit
            def step(x, y):
                opt.zero_grad()
                loss = model(x).sparse_categorical_crossentropy(y).backward()
                return loss.realize(*opt.schedule_step())

            def sync(t):
                return t.item()
        else:
            @TinyJit
            def step(x, y):
                return model(x).realize()

            def sync(t):
                return t.tolist()

        t0 = time.perf_counter()
        sync(step(x, y))
        cold = (time.perf_counter() - t0) * 1e3
        for _ in range(warmup):
            sync(step(x, y))
        samples = []
        for _ in range(iters):
            t0 = time.perf_counter()
            sync(step(x, y))
            samples.append((time.perf_counter() - t0) * 1e3)

    samples.sort()
    n = len(samples)
    median = samples[n // 2] if n % 2 else 0.5 * (samples[n // 2 - 1] + samples[n // 2])
    q = lambda p: samples[round((n - 1) * p)]
    result = {
        "framework": "tinygrad",
        "version": "0.14.0",
        "device": _DEV,
        "workload": args.workload,
        "batch": batch,
        "dtype": "f32",
        "jit": True,
        "warmup": warmup,
        "iters": iters,
        "ms_cold": round(cold, 4),
        "ms_median": round(median, 4),
        "ms_min": round(samples[0], 4),
        "ms_iqr": round(q(0.75) - q(0.25), 4),
        "images_per_s": round(batch * 1e3 / median, 2),
        "params": sum(p.numel() for p in params),
    }
    print(f"{args.workload} {_DEV} bs={batch} jit=True: {median:.3f} ms/step "
          f"(min {samples[0]:.3f}, iqr {result['ms_iqr']:.3f}, cold {cold:.1f} ms, "
          f"{result['images_per_s']:.0f} img/s)")
    print("RESULT " + json.dumps(result))


if __name__ == "__main__":
    main()
