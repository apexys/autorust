#!/usr/bin/env python3
"""PyTorch side of the cross-framework benchmark.

Same models/shapes as `bench/autorust`: the `beautiful_mnist` CNN, ViT-B/16,
the GEMM / memory-bound / optimizer microbenches, ResNet-18/50, MobileNetV2, a
small UNet and an autoregressive decoder (GQA + RoPE + KV cache + sampling).

Run with bench/.venv/bin/python. The final line on stdout is `RESULT <json>`.
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


# --------------------------------------------------------------------------
# existing model workloads (MNIST CNN, ViT-B/16)
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# conv models
# --------------------------------------------------------------------------


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
        self.down = (
            conv_bn(cin, cout, 1, stride, 0) if stride != 1 or cin != cout else None
        )

    def forward(self, x):
        idt = self.down(x) if self.down is not None else x
        y = F.relu(self.c1(x))
        y = self.c2(y)
        return F.relu(y + idt)


class Bottleneck(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        mid = cout // 4
        self.c1 = conv_bn(cin, mid, 1, 1, 0)
        self.c2 = conv_bn(mid, mid, 3, stride, 1)
        self.c3 = conv_bn(mid, cout, 1, 1, 0)
        self.down = (
            conv_bn(cin, cout, 1, stride, 0) if stride != 1 or cin != cout else None
        )

    def forward(self, x):
        idt = self.down(x) if self.down is not None else x
        y = F.relu(self.c1(x))
        y = F.relu(self.c2(y))
        y = self.c3(y)
        return F.relu(y + idt)


class ResNet(nn.Module):
    def __init__(self, depth=18, classes=1000, width=1.0):
        super().__init__()
        layers = [2, 2, 2, 2] if depth == 18 else [3, 4, 6, 3]
        block = Basic if depth == 18 else Bottleneck

        def w(c):
            return max(8, int(c * width) // 8 * 8)

        self.stem = conv_bn(3, w(64), 7, 2, 3)
        stages, cin = [], w(64)
        for i, n in enumerate(layers):
            cout = w(64 << i)
            blocks = []
            for b in range(n):
                stride = 2 if i > 0 and b == 0 else 1
                blocks.append(block(cin, cout, stride))
                cin = cout
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.fc = nn.Linear(cin, classes)

    def forward(self, x):
        x = F.relu(self.stem(x))
        x = F.max_pool2d(x, 3, 2, 1)
        x = self.stages(x)
        return self.fc(x.mean((2, 3)))


class InvRes(nn.Module):
    def __init__(self, cin, cout, stride, expand):
        super().__init__()
        mid = cin * expand
        self.expand = conv_bn(cin, mid, 1, 1, 0) if expand != 1 else None
        self.dw = conv_bn(mid, mid, 3, stride, 1, groups=mid)
        self.project = conv_bn(mid, cout, 1, 1, 0)
        self.add = stride == 1 and cin == cout

    def forward(self, x):
        y = F.relu6(self.expand(x)) if self.expand is not None else x
        y = F.relu6(self.dw(y))
        y = self.project(y)
        return x + y if self.add else y


class MobileNet(nn.Module):
    def __init__(self, classes=1000):
        super().__init__()
        cfg = [
            (1, 16, 1), (6, 24, 2), (6, 24, 1), (6, 32, 2), (6, 32, 1), (6, 32, 1),
            (6, 64, 2), (6, 64, 1), (6, 64, 1), (6, 64, 1), (6, 96, 1), (6, 96, 1),
            (6, 96, 1), (6, 160, 2), (6, 160, 1), (6, 160, 1), (6, 320, 1),
        ]
        self.first = conv_bn(3, 32, 3, 2, 1)
        blocks, cin = [], 32
        for e, cout, s in cfg:
            blocks.append(InvRes(cin, cout, s, e))
            cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.last = conv_bn(320, 1280, 1, 1, 0)
        self.fc = nn.Linear(1280, classes)

    def forward(self, x):
        x = F.relu6(self.first(x))
        x = F.relu6(self.blocks(x))
        x = F.relu6(self.last(x))
        return self.fc(x.mean((2, 3)))


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        g = min(cout, 8)
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, 1, 1),
            nn.GroupNorm(g, cout),
            nn.ReLU(),
            nn.Conv2d(cout, cout, 3, 1, 1),
            nn.GroupNorm(g, cout),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = DoubleConv(3, 32)
        self.e2 = DoubleConv(32, 64)
        self.e3 = DoubleConv(64, 128)
        self.bottom = DoubleConv(128, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.d3 = DoubleConv(256, 128)
        self.d2 = DoubleConv(192, 64)
        self.d1 = DoubleConv(96, 32)
        self.out = nn.Conv2d(32, 3, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        b = self.bottom(F.max_pool2d(e3, 2))
        u3 = self.up3(b)
        d3 = self.d3(torch.cat([u3, e3], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, scale_factor=2, mode="nearest"), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, scale_factor=2, mode="nearest"), e1], 1))
        return self.out(d1)


# --------------------------------------------------------------------------
# autoregressive decoder (GQA + RoPE + KV cache + sampling)
# --------------------------------------------------------------------------


class LlmConfig:
    d_model = 768
    n_heads = 12
    n_kv = 4
    n_layers = 6
    ffn = 3072
    vocab = 32000
    seq = 512


class LlmAttn(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.h, self.kv = cfg.n_heads, cfg.n_kv
        self.hd = cfg.d_model // cfg.n_heads
        self.q = nn.Linear(cfg.d_model, cfg.n_heads * self.hd, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.n_kv * self.hd, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.n_kv * self.hd, bias=False)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        pos = torch.arange(cfg.seq + 1, dtype=torch.float32)[:, None]
        freq = torch.exp(
            -math.log(10000.0) * torch.arange(0, self.hd, 2, dtype=torch.float32) / self.hd
        )
        ang = pos * freq[None, :]
        self.register_buffer("cos", ang.cos()[None, None])
        self.register_buffer("sin", ang.sin()[None, None])

    def rope(self, x, start):
        t = x.shape[2]
        cos = self.cos[:, :, start : start + t]
        sin = self.sin[:, :, start : start + t]
        cos = torch.cat([cos, cos], -1)
        sin = torch.cat([sin, sin], -1)
        half = self.hd // 2
        x1, x2 = x[..., :half], x[..., half:]
        rot = torch.cat([-x2, x1], -1)
        return x * cos + rot * sin

    def forward(self, x, start, cache=None):
        b, t, _ = x.shape
        hd = self.hd
        q = self.q(x).view(b, t, self.h, hd).transpose(1, 2)
        k = self.k(x).view(b, t, self.kv, hd).transpose(1, 2)
        v = self.v(x).view(b, t, self.kv, hd).transpose(1, 2)
        q, k = self.rope(q, start), self.rope(k, start)
        new_cache = (k, v)
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        group = self.h // self.kv
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
        scores = q @ k.transpose(-1, -2) * (1.0 / math.sqrt(hd))
        if start == 0 and t > 1:
            mask = torch.arange(t, device=x.device)
            scores = scores.masked_fill(mask[None, :] > mask[:, None], float("-inf"))
        a = torch.softmax(scores, -1)
        y = (a @ v).transpose(1, 2).reshape(b, t, self.h * hd)
        return self.o(y), new_cache


class LlmBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.RMSNorm(cfg.d_model, eps=1e-6) if hasattr(nn, "RMSNorm") else None
        if self.ln1 is None:
            self.ln1 = nn.LayerNorm(cfg.d_model, eps=1e-6)
        self.attn = LlmAttn(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model, eps=1e-6)
        self.gate = nn.Linear(cfg.d_model, cfg.ffn, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.ffn, bias=False)
        self.down = nn.Linear(cfg.ffn, cfg.d_model, bias=False)

    def forward(self, x, start, cache=None):
        a, c = self.attn(self.ln1(x), start, cache)
        x = x + a
        h = self.ln2(x)
        y = self.down(F.silu(self.gate(h)) * self.up(h))
        return x + y, c


class Llm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab, cfg.d_model)
        self.blocks = nn.ModuleList([LlmBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(cfg.d_model, eps=1e-6)
        self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)

    def prefill(self, tokens):
        x = self.embed(tokens)
        caches = []
        for blk in self.blocks:
            x, c = blk(x, 0, None)
            caches.append(c)
        return self.head(self.norm(x)), caches

    def decode(self, token, pos, caches):
        x = self.embed(token)
        for blk, cache in zip(self.blocks, caches):
            x, _ = blk(x, pos, cache)
        logits = self.head(self.norm(x))
        u = torch.rand_like(logits)
        g = -torch.log(-torch.log(u))
        return (logits + g).argmax(-1, keepdim=True)


# --------------------------------------------------------------------------
# measurement helpers
# --------------------------------------------------------------------------


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def measure(step, warmup, iters, dev):
    sync(dev)
    t = time.perf_counter()
    step()
    sync(dev)
    cold = (time.perf_counter() - t) * 1e3
    for _ in range(warmup):
        step()
    sync(dev)
    samples = []
    for _ in range(iters):
        sync(dev)
        t = time.perf_counter()
        step()
        sync(dev)
        samples.append((time.perf_counter() - t) * 1e3)
    return cold, samples


def stats(samples):
    samples = sorted(samples)
    n = len(samples)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    median = samples[n // 2] if n % 2 else 0.5 * (samples[n // 2 - 1] + samples[n // 2])
    q = lambda p: samples[round((n - 1) * p)]
    return median, samples[0], q(0.75) - q(0.25)


def emit(workload, device, batch, dtype_name, jit, warmup, iters, cold, samples):
    median, mn, iqr = stats(samples)
    images = batch * 1e3 / median if median > 0 else float("nan")
    result = {
        "framework": "torch",
        "version": torch.__version__,
        "device": device.upper(),
        "workload": workload,
        "batch": batch,
        "dtype": dtype_name,
        "jit": jit,
        "warmup": warmup,
        "iters": iters,
        "ms_cold": round(cold, 4),
        "ms_median": round(median, 4),
        "ms_min": round(mn, 4),
        "ms_iqr": round(iqr, 4),
        "images_per_s": round(images, 2),
    }
    print(
        f"{workload} {device.upper()} bs={batch} dtype={dtype_name}: {median:.3f} ms/step "
        f"(min {mn:.3f}, iqr {iqr:.3f}, cold {cold:.1f} ms, {images:.0f} img/s)"
    )
    print("RESULT " + json.dumps(result))


def emit_cases(workload, device, dtype_name, warmup, iters, cases, metric):
    total = sum(c[1] for c in cases)
    for name, ms, val in cases:
        print(f"{workload} {device.upper()} {name}: {ms:.3f} ms ({val:.2f} {metric})")
    payload = {
        "framework": "torch",
        "version": torch.__version__,
        "device": device.upper(),
        "workload": workload,
        "batch": 1,
        "dtype": dtype_name,
        "jit": False,
        "warmup": warmup,
        "iters": iters,
        "ms_cold": 0.0,
        "ms_median": round(total, 4),
        "ms_min": 0.0,
        "ms_iqr": 0.0,
        "images_per_s": 0.0,
        "cases": [
            {"case": n, "ms": round(ms, 4), metric: round(v, 4)} for n, ms, v in cases
        ],
    }
    print("RESULT " + json.dumps(payload))


# --------------------------------------------------------------------------
# workloads
# --------------------------------------------------------------------------


GEMM_SHAPES = [
    ("1024³", 1, 1024, 1024, 1024, False, False),
    ("2048³", 1, 2048, 2048, 2048, False, False),
    ("4096³", 1, 4096, 4096, 4096, False, False),
    ("1024x1024x4096", 1, 1024, 1024, 4096, False, False),
    ("4096x1024x1024", 1, 4096, 1024, 1024, False, False),
    ("TN 2048³", 1, 2048, 2048, 2048, True, False),
    ("NT 2048³", 1, 2048, 2048, 2048, False, True),
    ("batched 8x512³", 8, 512, 512, 512, False, False),
]


def run_gemm(dev, dtype_name, warmup, iters, only):
    dt = torch.bfloat16 if dtype_name == "bf16" else torch.float32
    cases = []
    for name, batch, m, n, k, ta, tb in GEMM_SHAPES:
        label = f"{dtype_name} {name}"
        if only and only not in label:
            continue
        a = torch.rand(batch, m, k, device=dev, dtype=dt)
        bshape = (batch, n, k) if tb else (batch, k, n)
        b = torch.rand(*bshape, device=dev, dtype=dt)
        lhs = a.transpose(-2, -1) if ta else a
        rhs = b.transpose(-2, -1) if tb else b

        def step():
            c = lhs @ rhs
            return c[0, 0, 0].item()

        _, samples = measure(step, warmup, iters, dev)
        ms = stats(samples)[0]
        flops = 2.0 * batch * m * n * k
        cases.append((label, ms, flops / (ms * 1e9)))
    emit_cases("gemm", dev.type, dtype_name, warmup, iters, cases, "tflops")


MEM_N = 1 << 24


def run_memops(dev, warmup, iters, only):
    n = MEM_N
    a = torch.rand(n, device=dev)
    b = torch.rand(n, device=dev)
    m2 = a.view(4096, 4096)
    ln = nn.LayerNorm(4096, eps=1e-5).to(dev)
    nbytes = n * 4.0
    # Each entry rebuilds the op inside the timed region (torch is eager, so a
    # precomputed tensor would measure nothing).
    ops = [
        ("add", 3 * nbytes, lambda: a + b),
        ("copy", 2 * nbytes, lambda: a * 1.0),
        ("gelu", 2 * nbytes, lambda: F.gelu(a)),
        ("softmax", 2 * nbytes, lambda: torch.softmax(m2, dim=1)),
        ("layernorm", 2 * nbytes, lambda: ln(m2)),
        ("sum", nbytes, lambda: a.sum()),
        ("amax", nbytes, lambda: a.max()),
        ("transpose", 2 * nbytes, lambda: m2.t().contiguous()),
    ]
    cases = []
    for name, nbytes, fn in ops:
        if only and only not in name:
            continue

        def step(fn=fn):
            return fn().flatten()[0].item()

        _, samples = measure(step, warmup, iters, dev)
        ms = stats(samples)[0]
        cases.append((f"{name} {n}", ms, nbytes / (ms * 1e6)))
    emit_cases("memops", dev.type, "f32", warmup, iters, cases, "gbs")


def run_optim(dev, warmup, iters):
    total = env_int("BENCH_OPT_PARAMS", 100_000_000)
    ntensors = env_int("BENCH_OPT_TENSORS", 32)
    per = total // ntensors
    params = [
        torch.zeros(per, device=dev, requires_grad=True) for _ in range(ntensors)
    ]
    for p in params:
        p.grad = torch.ones_like(p)
    opt = torch.optim.AdamW(params, lr=1e-3)

    def step():
        opt.step()

    _, samples = measure(step, warmup, iters, dev)
    ms = stats(samples)[0]
    case = (f"adamw {total} params, {ntensors} tensors", ms, 28.0 * total / (ms * 1e6))
    emit_cases("optim", dev.type, "f32", warmup, iters, [case], "gbs")


def run_model(workload, dev, dtype_name, warmup, iters):
    dt = torch.bfloat16 if dtype_name == "bf16" else torch.float32
    batch = env_int("BENCH_BS", 8 if workload == "vit_infer_bf16" else 32)
    torch.manual_seed(0)

    if workload in ("onnx_mnist", "onnx_resnet"):
        if workload == "onnx_mnist":
            model, batch = Mnist(), env_int("BENCH_BS", 32)
            x = torch.rand(batch, 1, 28, 28, device=dev)
        else:
            model, batch = ResNet(18, width=0.25), env_int("BENCH_BS", 8)
            x = torch.rand(batch, 3, 224, 224, device=dev)
        model = model.to(dev).eval()
        with torch.no_grad():
            def step():
                return model(x).detach()

            cold, samples = measure(step, warmup, iters, dev)
        emit(workload, dev.type, batch, "f32", False, warmup, iters, cold, samples)
        return

    if workload == "vit_infer_bf16":
        model = Vit().to(dev, dtype=dt)
        x = torch.zeros(batch, 3, 224, 224, device=dev, dtype=dt)

        def step():
            with torch.no_grad():
                out = model(x)
            return out.float().cpu()

        cold, samples = measure(step, warmup, iters, dev)
        emit(workload, dev.type, batch, dtype_name, False, warmup, iters, cold, samples)
        return

    if workload.startswith("resnet") or workload == "mobilenet_infer":
        if workload.startswith("resnet"):
            model = ResNet(18 if "18" in workload else 50)
        else:
            model = MobileNet()
        x = torch.zeros(batch, 3, 224, 224, device=dev)
    elif workload == "unet_infer":
        model = Unet()
        batch = env_int("BENCH_BS", 4)
        x = torch.rand(batch, 3, 64, 64, device=dev)
    else:
        raise SystemExit(f"unknown model workload {workload}")

    model = model.to(dev)
    training = workload.endswith("_train")
    model.train(training)
    y = torch.randint(0, 1000, (batch,), device=dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def step():
        if training:
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            return float(loss.detach())
        with torch.no_grad():
            out = model(x)
        return out.detach().cpu()

    cold, samples = measure(step, warmup, iters, dev)
    emit(workload, dev.type, batch, "f32", False, warmup, iters, cold, samples)


def run_llm(workload, dev, warmup, iters):
    cfg = LlmConfig()
    cfg.vocab = env_int("BENCH_LLM_VOCAB", cfg.vocab)
    cfg.seq = env_int("BENCH_LLM_SEQ", cfg.seq)
    cfg.n_layers = env_int("BENCH_LLM_LAYERS", cfg.n_layers)
    cfg.d_model = env_int("BENCH_LLM_DMODEL", cfg.d_model)
    cfg.n_heads = env_int("BENCH_LLM_HEADS", cfg.n_heads)
    cfg.n_kv = env_int("BENCH_LLM_KV", cfg.n_kv)
    cfg.ffn = env_int("BENCH_LLM_FFN", cfg.ffn)
    torch.manual_seed(0)
    model = Llm(cfg).to(dev)
    tokens = torch.randint(0, cfg.vocab, (1, cfg.seq), device=dev)
    token = torch.randint(0, cfg.vocab, (1, 1), device=dev)

    if workload == "llm_prefill":
        def step():
            with torch.no_grad():
                logits, _ = model.prefill(tokens)
            return logits[0, 0, 0].item()

        cold, samples = measure(step, warmup, iters, dev)
        emit(workload, dev.type, cfg.seq, "f32", False, warmup, iters, cold, samples)
        return

    with torch.no_grad():
        _, caches = model.prefill(tokens)
    caches = [(k.contiguous(), v.contiguous()) for k, v in caches]

    def step():
        with torch.no_grad():
            nxt = model.decode(token, cfg.seq, caches)
        return nxt.item()

    cold, samples = measure(step, warmup, iters, dev)
    emit(workload, dev.type, 1, "f32", False, warmup, iters, cold, samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--workload",
        default="mnist_train",
        choices=[
            "mnist_train", "vit_infer", "vit_train", "gemm", "memops", "optim",
            "resnet18_train", "resnet18_infer", "resnet50_infer", "mobilenet_infer",
            "unet_infer", "llm_prefill", "llm_decode", "vit_infer_bf16",
            "onnx_mnist", "onnx_resnet", "vit_train_bf16",
        ],
    )
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--dtype", default="f32", choices=["f32", "bf16"])
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

    if args.workload == "gemm":
        run_gemm(
            dev, args.dtype, args.warmup or 1, args.iters or 3,
            os.environ.get("BENCH_GEMM_ONLY"),
        )
        return
    if args.workload == "memops":
        run_memops(
            dev, args.warmup or 2, args.iters or 5,
            os.environ.get("BENCH_MEMOPS_ONLY"),
        )
        return
    if args.workload == "optim":
        run_optim(dev, args.warmup or 2, args.iters or 10)
        return
    if args.workload in ("llm_prefill", "llm_decode"):
        run_llm(args.workload, dev, args.warmup or 1, args.iters or 3)
        return
    if args.workload in (
        "resnet18_train", "resnet18_infer", "resnet50_infer", "mobilenet_infer",
        "unet_infer", "vit_infer_bf16", "onnx_mnist", "onnx_resnet",
    ):
        run_model(
            args.workload, dev, args.dtype, args.warmup or 2, args.iters or 5,
        )
        return

    # original MNIST / ViT rows
    training = args.workload.endswith("train") or args.workload == "vit_train_bf16"
    batch = args.batch if args.batch is not None else (
        512 if args.workload == "mnist_train" else (1 if args.workload == "vit_train" else 8))
    warmup = args.warmup if args.warmup is not None else (5 if training else 3)
    iters = args.iters if args.iters is not None else (30 if args.workload == "mnist_train" else 5)

    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    if args.workload == "mnist_train":
        model = Mnist()
        x = torch.rand(batch, 1, 28, 28)
        y = torch.randint(0, 10, (batch,))
    else:
        model = Vit(layers=args.layers, dim=args.dim, heads=args.heads, img=args.img)
        x = torch.zeros(batch, 3, args.img, args.img)
        y = torch.randint(0, 1000, (batch,))
    model = model.to(dev, dtype=dt)
    x, y = x.to(dev, dtype=dt), y.to(dev)
    model.train(training)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3 if args.workload == "mnist_train" else 3e-4)
    if args.compile:
        model = torch.compile(model)

    def step():
        if training:
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            return float(loss.detach())
        out = model(x)
        out.detach().cpu()
        return None

    cold, samples = measure(step, warmup, iters, dev)
    emit(
        args.workload, dev.type, batch, args.dtype, bool(args.compile), warmup, iters,
        cold, samples,
    )


if __name__ == "__main__":
    main()
