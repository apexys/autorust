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
    # --- ViT-B/16 inference, batch 1 ---
    ("autorust-rust-vit_infer-bs1", "autorust", "RUST", "vit_infer", 1, 1, 3, "AUTORUST_ATTN=sdpa"),
    ("autorust-cuda-vit_infer-bs1", "autorust", "CUDA", "vit_infer", 1, 1, 3, "SDPA"),
    ("candle-cpu-vit_infer-bs1", "candle", "cpu", "vit_infer", 1, 1, 2, ""),
    ("candle-cuda-vit_infer-bs1", "candle", "cuda", "vit_infer", 1, 1, 3, ""),
    ("torch-cpu-vit_infer-bs1", "torch", "cpu", "vit_infer", 1, 1, 3, ""),
    ("torch-cuda-vit_infer-bs1", "torch", "cuda", "vit_infer", 1, 1, 3, ""),
    ("tinygrad-cpu-vit_infer-bs1", "tinygrad", "cpu", "vit_infer", 1, 0, 1, "single-threaded"),
    ("tinygrad-cuda-vit_infer-bs1", "tinygrad", "cuda", "vit_infer", 1, 1, 3, ""),
    # --- ViT-B/16 inference, batch 8 ---
    ("autorust-rust-vit_infer-bs8", "autorust", "RUST", "vit_infer", 8, 1, 2, "AUTORUST_ATTN=sdpa"),
    ("autorust-cuda-vit_infer-bs8", "autorust", "CUDA", "vit_infer", 8, 1, 3, "SDPA"),
    ("candle-cpu-vit_infer-bs8", "candle", "cpu", "vit_infer", 8, 1, 2, ""),
    ("candle-cuda-vit_infer-bs8", "candle", "cuda", "vit_infer", 8, 1, 3, ""),
    ("torch-cpu-vit_infer-bs8", "torch", "cpu", "vit_infer", 8, 1, 2, ""),
    ("torch-cuda-vit_infer-bs8", "torch", "cuda", "vit_infer", 8, 1, 3, ""),
    ("tinygrad-cuda-vit_infer-bs8", "tinygrad", "cuda", "vit_infer", 8, 1, 3, ""),
    # --- ViT-B/16 training step, batch 1 (f32) ---
    ("autorust-rust-vit_train", "autorust", "RUST", "vit_train", 1, 1, 2, "AUTORUST_ATTN=sdpa"),
    ("autorust-cuda-vit_train", "autorust", "CUDA", "vit_train", 1, 1, 3, "SDPA"),
    ("candle-cpu-vit_train", "candle", "cpu", "vit_train", 1, 1, 2, ""),
    ("candle-cuda-vit_train", "candle", "cuda", "vit_train", 1, 1, 3, ""),
    ("torch-cpu-vit_train", "torch", "cpu", "vit_train", 1, 1, 2, ""),
    ("torch-cuda-vit_train", "torch", "cuda", "vit_train", 1, 1, 3, ""),
    ("tinygrad-cuda-vit_train", "tinygrad", "cuda", "vit_train", 1, 1, 3, ""),
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
    # The autorust CPU backends should use the cheap SDPA path; the op-level
    # FlashAttention port is 16x slower there (see PERFORMANCE.md). The model
    # itself always uses SDPA unless `FA=1` is set.
    env.setdefault("AUTORUST_ATTN", "sdpa")
    return env


def parse_result(output):
    for line in reversed(output.splitlines()):
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    raise RuntimeError(f"no RESULT line in output:\n{output[-2000:]}")


def cmd_autorust(device, workload, batch, warmup, iters):
    if not AUTORUST_BIN.exists():
        raise SystemExit("missing autorust-bench: cargo build --release -p autorust-bench")
    env = base_env()
    env.update(
        DEFAULT_DEVICE=device,
        BENCH_WORKLOAD=workload,
        BENCH_BS=str(batch),
        BENCH_WARMUP=str(warmup),
        BENCH_ITERS=str(iters),
    )
    return [str(AUTORUST_BIN)], env


def cmd_candle(device, workload, batch, warmup, iters):
    if not CANDLE_BIN.exists():
        raise SystemExit("missing candle-bench: (cd bench/candle && cargo build --release --features cuda)")
    env = base_env()
    env.update(
        BENCH_DEVICE=device,
        BENCH_WORKLOAD=workload,
        BENCH_BS=str(batch),
        BENCH_WARMUP=str(warmup),
        BENCH_ITERS=str(iters),
    )
    return [str(CANDLE_BIN)], env


def cmd_torch(device, workload, batch, warmup, iters):
    return [
        str(TORCH_PY), str(BENCH / "torch" / "bench.py"),
        "--device", device, "--workload", workload, "--batch", str(batch),
        "--warmup", str(warmup), "--iters", str(iters),
    ], base_env()


def cmd_tinygrad(device, workload, batch, warmup, iters):
    return [
        str(TINYGRAD_PY), str(BENCH / "tinygrad" / "bench.py"),
        "--device", device, "--workload", workload, "--batch", str(batch),
        "--warmup", str(warmup), "--iters", str(iters),
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
        print(f"    {result['ms_median']:.3f} ms  ({result['images_per_s']:.0f} img/s)"
              f"  [{time.time() - t:.1f}s]", flush=True)


if __name__ == "__main__":
    main()
