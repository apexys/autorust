#!/usr/bin/env python3
"""Run the cross-framework benchmark matrix and write bench/results/*.json.

Prerequisites (see bench/README.md):
  - torch venv:  uv venv bench/.venv && uv pip install torch --index-url ...
  - tinygrad:    ../tinygrad/.venv/bin/python (checkout v0.14.0)
  - autorust:    cargo build --release -p autorust-bench
  - candle:      cargo build --release --features cuda   (in bench/candle)

Examples:
  bench/run.py --list
  bench/run.py --only 'mnist.*' 
  bench/run.py --only 'autorust.*cuda.*vit_infer'
"""

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
BENCH = ROOT / "bench"
RESULTS = BENCH / "results"
TORCH_PY = BENCH / ".venv" / "bin" / "python"
TINYGRAD_PY = ROOT.parent / "tinygrad" / ".venv" / "bin" / "python"
AUTORUST_BIN = ROOT / "target" / "release" / "autorust-bench"
CANDLE_BIN = BENCH / "candle" / "target" / "release" / "candle-bench"

# (name, framework, device, workload, batch, warmup, iters, notes)
RUNS = [
    # --- MNIST CNN training, batch 512 (the beautiful_mnist demo) ---
    ("autorust-rust-mnist", "autorust", "RUST", "mnist_train", 512, 3, 10, ""),
    ("autorust-cuda-mnist", "autorust", "CUDA", "mnist_train", 512, 3, 10, ""),
    ("autorust-rayon-mnist", "autorust", "RAYON", "mnist_train", 32, 0, 1, "typed interpreter, bs=32 (per-step memory growth)"),
    ("candle-cpu-mnist", "candle", "cpu", "mnist_train", 512, 2, 8, ""),
    ("candle-cuda-mnist", "candle", "cuda", "mnist_train", 512, 2, 8, ""),
    ("torch-cpu-mnist", "torch", "cpu", "mnist_train", 512, 2, 10, ""),
    ("torch-cuda-mnist", "torch", "cuda", "mnist_train", 512, 2, 10, ""),
    ("tinygrad-cpu-mnist", "tinygrad", "cpu", "mnist_train", 512, 2, 5, "single-threaded"),
    ("tinygrad-cuda-mnist", "tinygrad", "cuda", "mnist_train", 512, 2, 10, ""),
    # --- MNIST CNN training, batch 64 (the batch people actually type) ---
    ("autorust-rust-mnist-bs64", "autorust", "RUST", "mnist_train", 64, 3, 10, ""),
    ("autorust-cuda-mnist-bs64", "autorust", "CUDA", "mnist_train", 64, 3, 10, ""),
    ("candle-cpu-mnist-bs64", "candle", "cpu", "mnist_train", 64, 2, 8, ""),
    ("candle-cuda-mnist-bs64", "candle", "cuda", "mnist_train", 64, 2, 8, ""),
    ("torch-cpu-mnist-bs64", "torch", "cpu", "mnist_train", 64, 2, 10, ""),
    ("torch-cuda-mnist-bs64", "torch", "cuda", "mnist_train", 64, 2, 10, ""),
    ("tinygrad-cpu-mnist-bs64", "tinygrad", "cpu", "mnist_train", 64, 2, 5, "single-threaded"),
    ("tinygrad-cuda-mnist-bs64", "tinygrad", "cuda", "mnist_train", 64, 2, 10, ""),
    # --- ViT-B/16 inference, batch 1 ---
    ("autorust-rust-vit_infer-bs1", "autorust", "RUST", "vit_infer", 1, 1, 3, "SDPA (CPU)"),
    ("autorust-cuda-vit_infer-bs1", "autorust", "CUDA", "vit_infer", 1, 1, 3, "fused FA2"),
    ("candle-cpu-vit_infer-bs1", "candle", "cpu", "vit_infer", 1, 1, 2, ""),
    ("candle-cuda-vit_infer-bs1", "candle", "cuda", "vit_infer", 1, 1, 3, ""),
    ("torch-cpu-vit_infer-bs1", "torch", "cpu", "vit_infer", 1, 1, 3, ""),
    ("torch-cuda-vit_infer-bs1", "torch", "cuda", "vit_infer", 1, 1, 3, ""),
    ("tinygrad-cpu-vit_infer-bs1", "tinygrad", "cpu", "vit_infer", 1, 0, 1, "single-threaded"),
    ("tinygrad-cuda-vit_infer-bs1", "tinygrad", "cuda", "vit_infer", 1, 1, 3, ""),
    # --- ViT-B/16 inference, batch 8 ---
    ("autorust-rust-vit_infer-bs8", "autorust", "RUST", "vit_infer", 8, 1, 2, "SDPA (CPU)"),
    ("autorust-cuda-vit_infer-bs8", "autorust", "CUDA", "vit_infer", 8, 1, 3, "fused FA2"),
    ("candle-cpu-vit_infer-bs8", "candle", "cpu", "vit_infer", 8, 1, 2, ""),
    ("candle-cuda-vit_infer-bs8", "candle", "cuda", "vit_infer", 8, 1, 3, ""),
    ("torch-cpu-vit_infer-bs8", "torch", "cpu", "vit_infer", 8, 1, 2, ""),
    ("torch-cuda-vit_infer-bs8", "torch", "cuda", "vit_infer", 8, 1, 3, ""),
    ("tinygrad-cuda-vit_infer-bs8", "tinygrad", "cuda", "vit_infer", 8, 1, 3, ""),
    # --- ViT-B/16 training step, batch 1 (f32) ---
    ("autorust-rust-vit_train", "autorust", "RUST", "vit_train", 1, 1, 2, "SDPA (CPU)"),
    ("autorust-cuda-vit_train", "autorust", "CUDA", "vit_train", 1, 1, 3, "SDPA (training)"),
    ("candle-cpu-vit_train", "candle", "cpu", "vit_train", 1, 1, 2, ""),
    ("candle-cuda-vit_train", "candle", "cuda", "vit_train", 1, 1, 3, ""),
    ("torch-cpu-vit_train", "torch", "cpu", "vit_train", 1, 1, 2, ""),
    ("torch-cuda-vit_train", "torch", "cuda", "vit_train", 1, 1, 3, ""),
    ("tinygrad-cuda-vit_train", "tinygrad", "cuda", "vit_train", 1, 1, 3, ""),
    # --- ViT-B/16 training step, batch 4 (how the deficit scales with batch) ---
    ("autorust-rust-vit_train-bs4", "autorust", "RUST", "vit_train", 4, 1, 2, "SDPA (CPU)"),
    ("autorust-cuda-vit_train-bs4", "autorust", "CUDA", "vit_train", 4, 1, 3, "SDPA (training)"),
    ("candle-cpu-vit_train-bs4", "candle", "cpu", "vit_train", 4, 1, 2, ""),
    ("candle-cuda-vit_train-bs4", "candle", "cuda", "vit_train", 4, 1, 3, ""),
    ("torch-cpu-vit_train-bs4", "torch", "cpu", "vit_train", 4, 1, 2, ""),
    ("torch-cuda-vit_train-bs4", "torch", "cuda", "vit_train", 4, 1, 3, ""),
    ("tinygrad-cuda-vit_train-bs4", "tinygrad", "cuda", "vit_train", 4, 1, 3, ""),
    # --- Microbenches: GEMM sweep (f32 + bf16), memory-bound ops, AdamW ---
    ("autorust-rust-gemm", "autorust", "RUST", "gemm", 1, 1, 3, ""),
    ("autorust-cuda-gemm", "autorust", "CUDA", "gemm", 1, 1, 3, ""),
    ("candle-cpu-gemm", "candle", "cpu", "gemm", 1, 1, 3, ""),
    ("candle-cuda-gemm", "candle", "cuda", "gemm", 1, 1, 3, ""),
    ("torch-cpu-gemm", "torch", "cpu", "gemm", 1, 1, 3, ""),
    ("torch-cuda-gemm", "torch", "cuda", "gemm", 1, 1, 3, ""),
    ("tinygrad-cuda-gemm", "tinygrad", "cuda", "gemm", 1, 1, 3, ""),
    ("autorust-cuda-gemm-bf16", "autorust", "CUDA", "gemm_bf16", 1, 1, 3, ""),
    ("torch-cuda-gemm-bf16", "torch", "cuda", "gemm_bf16", 1, 1, 3, ""),
    ("autorust-rust-memops", "autorust", "RUST", "memops", 1, 2, 5, ""),
    ("autorust-cuda-memops", "autorust", "CUDA", "memops", 1, 2, 5, ""),
    ("torch-cpu-memops", "torch", "cpu", "memops", 1, 2, 5, ""),
    ("torch-cuda-memops", "torch", "cuda", "memops", 1, 2, 5, ""),
    ("autorust-rust-optim", "autorust", "RUST", "optim", 1, 2, 10, ""),
    ("autorust-cuda-optim", "autorust", "CUDA", "optim", 1, 2, 10, ""),
    ("torch-cpu-optim", "torch", "cpu", "optim", 1, 2, 10, ""),
    ("torch-cuda-optim", "torch", "cuda", "optim", 1, 2, 10, ""),
    # --- Conv nets: ResNet-18/50, MobileNetV2 (depthwise), small UNet ---
    ("autorust-rust-resnet18_infer", "autorust", "RUST", "resnet18_infer", 32, 3, 5, ""),
    ("autorust-cuda-resnet18_infer", "autorust", "CUDA", "resnet18_infer", 32, 3, 5, ""),
    ("torch-cpu-resnet18_infer", "torch", "cpu", "resnet18_infer", 32, 3, 5, ""),
    ("torch-cuda-resnet18_infer", "torch", "cuda", "resnet18_infer", 32, 3, 5, ""),
    ("autorust-rust-resnet18_train", "autorust", "RUST", "resnet18_train", 32, 2, 3, ""),
    ("autorust-cuda-resnet18_train", "autorust", "CUDA", "resnet18_train", 32, 2, 3, ""),
    ("torch-cpu-resnet18_train", "torch", "cpu", "resnet18_train", 32, 2, 3, ""),
    ("torch-cuda-resnet18_train", "torch", "cuda", "resnet18_train", 32, 2, 3, ""),
    ("autorust-rust-resnet50_infer", "autorust", "RUST", "resnet50_infer", 32, 2, 3, ""),
    ("autorust-cuda-resnet50_infer", "autorust", "CUDA", "resnet50_infer", 32, 2, 3, ""),
    ("torch-cpu-resnet50_infer", "torch", "cpu", "resnet50_infer", 32, 2, 3, ""),
    ("torch-cuda-resnet50_infer", "torch", "cuda", "resnet50_infer", 32, 2, 3, ""),
    ("autorust-rust-mobilenet_infer", "autorust", "RUST", "mobilenet_infer", 32, 2, 5, ""),
    ("autorust-cuda-mobilenet_infer", "autorust", "CUDA", "mobilenet_infer", 32, 2, 5, ""),
    ("torch-cpu-mobilenet_infer", "torch", "cpu", "mobilenet_infer", 32, 2, 5, ""),
    ("torch-cuda-mobilenet_infer", "torch", "cuda", "mobilenet_infer", 32, 2, 5, ""),
    ("autorust-rust-unet_infer", "autorust", "RUST", "unet_infer", 4, 2, 3, ""),
    ("autorust-cuda-unet_infer", "autorust", "CUDA", "unet_infer", 4, 2, 5, ""),
    ("torch-cpu-unet_infer", "torch", "cpu", "unet_infer", 4, 2, 3, ""),
    ("torch-cuda-unet_infer", "torch", "cuda", "unet_infer", 4, 2, 5, ""),
    # --- Autoregressive decoder (GQA + RoPE + KV cache + sampling) ---
    ("autorust-rust-llm_prefill", "autorust", "RUST", "llm_prefill", 512, 1, 3, ""),
    ("autorust-cuda-llm_prefill", "autorust", "CUDA", "llm_prefill", 512, 1, 3, ""),
    ("torch-cpu-llm_prefill", "torch", "cpu", "llm_prefill", 512, 1, 3, ""),
    ("torch-cuda-llm_prefill", "torch", "cuda", "llm_prefill", 512, 1, 3, ""),
    ("autorust-rust-llm_decode", "autorust", "RUST", "llm_decode", 1, 2, 5, ""),
    ("autorust-cuda-llm_decode", "autorust", "CUDA", "llm_decode", 1, 2, 5, ""),
    ("torch-cpu-llm_decode", "torch", "cpu", "llm_decode", 1, 2, 5, ""),
    ("torch-cuda-llm_decode", "torch", "cuda", "llm_decode", 1, 2, 5, ""),
    # --- bf16 ViT (tensor-core path), ONNX runner, frozen AOT plan ---
    ("autorust-cuda-vit_infer_bf16", "autorust", "CUDA", "vit_infer_bf16", 8, 1, 3, ""),
    ("torch-cuda-vit_infer_bf16", "torch", "cuda", "vit_infer_bf16", 8, 1, 3, ""),
    ("autorust-cuda-vit_train_bf16", "autorust", "CUDA", "vit_train_bf16", 1, 1, 3, ""),
    ("torch-cuda-vit_train_bf16", "torch", "cuda", "vit_train_bf16", 1, 1, 3, ""),
    ("autorust-rust-onnx_mnist", "autorust", "RUST", "onnx_mnist", 32, 2, 10, ""),
    ("autorust-cuda-onnx_mnist", "autorust", "CUDA", "onnx_mnist", 32, 2, 10, ""),
    ("torch-cpu-onnx_mnist", "torch", "cpu", "onnx_mnist", 32, 2, 10, ""),
    ("torch-cuda-onnx_mnist", "torch", "cuda", "onnx_mnist", 32, 2, 10, ""),
    ("autorust-rust-onnx_resnet", "autorust", "RUST", "onnx_resnet", 8, 2, 5, ""),
    ("autorust-cuda-onnx_resnet", "autorust", "CUDA", "onnx_resnet", 8, 2, 5, ""),
    ("torch-cpu-onnx_resnet", "torch", "cpu", "onnx_resnet", 8, 2, 5, ""),
    ("torch-cuda-onnx_resnet", "torch", "cuda", "onnx_resnet", 8, 2, 5, ""),
    ("autorust-rust-freeze", "autorust", "RUST", "freeze", 1, 2, 5, ""),
]


def base_env():
    env = dict(os.environ)
    env.update(
        OMP_NUM_THREADS="24",
        RAYON_NUM_THREADS="24",
        BENCH_THREADS="24",
        PYTHONUNBUFFERED="1",
        TOKENIZERS_PARALLELISM="false",
    )
    # `AUTORUST_ATTN` is intentionally left at its default: the model uses the
    # fused attention kernels on CUDA and falls back to SDPA elsewhere, and the
    # op-level FA port stays opt-in via FA=1.
    return env


def parse_result(output):
    for line in reversed(output.splitlines()):
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    raise RuntimeError(f"no RESULT line in output:\n{output[-2000:]}")


def cmd_autorust(device, workload, batch, warmup, iters):
    env = base_env()
    if workload == "freeze":
        exe = ROOT / "target" / "release" / "examples" / "vit_freeze"
        if not exe.exists():
            raise SystemExit("missing frozen example: cargo build --release --example vit_freeze")
        env.update(BENCH_WARMUP=str(warmup), BENCH_ITERS=str(iters))
        return [str(exe)], env
    if not AUTORUST_BIN.exists():
        raise SystemExit("missing autorust-bench: cargo build --release -p autorust-bench")
    base = "gemm" if workload == "gemm_bf16" else workload
    env.update(
        DEFAULT_DEVICE=device,
        BENCH_WORKLOAD=base,
        BENCH_BS=str(batch),
        BENCH_WARMUP=str(warmup),
        BENCH_ITERS=str(iters),
    )
    if workload.endswith("bf16"):
        env["BENCH_BF16"] = "1"
    return [str(AUTORUST_BIN)], env


def cmd_candle(device, workload, batch, warmup, iters):
    if not CANDLE_BIN.exists():
        raise SystemExit("missing candle-bench: (cd bench/candle && cargo build --release --features cuda)")
    env = base_env()
    env.update(
        BENCH_DEVICE=device,
        BENCH_WORKLOAD="gemm" if workload == "gemm_bf16" else workload,
        BENCH_BS=str(batch),
        BENCH_WARMUP=str(warmup),
        BENCH_ITERS=str(iters),
    )
    if workload.endswith("bf16"):
        env["BENCH_DTYPE"] = "bf16"
    return [str(CANDLE_BIN)], env


def cmd_torch(device, workload, batch, warmup, iters):
    dtype = "bf16" if workload.endswith("bf16") else "f32"
    base = {"gemm_bf16": "gemm"}.get(workload, workload)
    return [
        str(TORCH_PY), str(BENCH / "torch" / "bench.py"),
        "--device", device, "--workload", base, "--batch", str(batch),
        "--warmup", str(warmup), "--iters", str(iters), "--dtype", dtype,
    ], base_env()


def cmd_tinygrad(device, workload, batch, warmup, iters):
    dtype = "bf16" if workload.endswith("bf16") else "f32"
    base = {"gemm_bf16": "gemm"}.get(workload, workload)
    return [
        str(TINYGRAD_PY), str(BENCH / "tinygrad" / "bench.py"),
        "--device", device, "--workload", base, "--batch", str(batch),
        "--warmup", str(warmup), "--iters", str(iters), "--dtype", dtype,
    ], base_env()


COMMANDS = {
    "autorust": cmd_autorust,
    "candle": cmd_candle,
    "torch": cmd_torch,
    "tinygrad": cmd_tinygrad,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="regex over run names")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per run")
    args = ap.parse_args()

    selected = [r for r in RUNS if args.only is None or re.search(args.only, r[0])]
    if args.list:
        for r in selected:
            print(r[0])
        return
    if not selected:
        raise SystemExit("no runs matched")

    RESULTS.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        commit = "unknown"

    for name, framework, device, workload, batch, warmup, iters, notes in selected:
        cmd, env = COMMANDS[framework](device, workload, batch, warmup, iters)
        print(f"==> {name}: {' '.join(cmd)}", flush=True)
        t = time.time()
        if framework == "autorust":
            # Warm the on-disk caches (PTX + rendered kernels) like the docs'
            # protocol says; otherwise ms_cold measures this binary's first
            # process after a rebuild. Discard the result.
            warm = dict(env, BENCH_ITERS="0", BENCH_WARMUP="0")
            subprocess.run(cmd, cwd=ROOT, env=warm, capture_output=True,
                           text=True, timeout=args.timeout)
        proc = subprocess.run(
            cmd, cwd=ROOT, env=env, capture_output=True, text=True,
            timeout=args.timeout,
        )
        if proc.returncode != 0:
            raw = BENCH / "results" / f"{name}.failed.txt"
            raw.write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
            print(f"    FAILED (see {raw})", flush=True)
            continue
        result = parse_result(proc.stdout)
        result.update(name=name, variant=device, commit=commit,
                      date=time.strftime("%Y-%m-%dT%H:%M:%S"), notes=notes)
        out = RESULTS / f"{name}.json"
        out.write_text(json.dumps(result, indent=2) + "\n")
        if "cases" in result:
            print(f"    {len(result['cases'])} cases, total {result['ms_median']:.1f} ms"
                  f"  [{time.time() - t:.1f}s]", flush=True)
        else:
            print(f"    {result['ms_median']:.3f} ms  ({result['images_per_s']:.0f} img/s)"
                  f"  [{time.time() - t:.1f}s]", flush=True)


if __name__ == "__main__":
    main()
