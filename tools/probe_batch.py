#!/usr/bin/env python3
"""Probe max per-GPU batch size for ResNet18 absolute-pose training on one GPU."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
from config import load_config  # noqa: E402


def try_batch(output_dim: int, bsz: int, precision: str, steps: int = 3) -> dict:
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    conv1 = torch.nn.Conv2d(2, 3, kernel_size=3, padding=1).to(device)
    rn = models.resnet18(num_classes=output_dim).to(device)
    opt = torch.optim.Adam(list(conv1.parameters()) + list(rn.parameters()), lr=1e-3)
    use_bf16 = precision in ("bf16", "bf16-mixed")
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    x = torch.randn(bsz, 180, 240, 2, device=device)
    y = torch.randn(bsz, output_dim, device=device)
    t0 = time.time()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
            h = conv1(x.permute(0, 3, 1, 2))
            pred = rn(h)
            loss = F.mse_loss(pred, y)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    dt = time.time() - t0
    mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    sps = bsz * steps / max(dt, 1e-6)
    del conv1, rn, opt, x, y, pred, loss
    torch.cuda.empty_cache()
    return {"batch": bsz, "mem_gb": mem, "samples_per_sec": sps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-frac", type=float, default=0.9)
    ap.add_argument("--start", type=int, default=256)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    cfg = load_config(args.config)
    output_dim = int(cfg["MODEL"]["OUTPUT_DIM"])
    precision = cfg["TRAIN"].get("PRECISION", "bf16")
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    budget = total_gb * args.max_frac
    print(f"GPU mem total={total_gb:.1f}GB budget={budget:.1f}GB precision={precision} out={output_dim}")

    bsz = args.start
    best = None
    while True:
        try:
            stats = try_batch(output_dim, bsz, precision)
            print(
                f"OK batch={bsz} mem={stats['mem_gb']:.2f}GB sps={stats['samples_per_sec']:.1f}"
            )
            if stats["mem_gb"] <= budget:
                best = stats
                bsz *= 2
            else:
                print("exceeded budget")
                break
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM at batch={bsz}")
                torch.cuda.empty_cache()
                break
            raise

    # Binary refine between best and last fail
    if best is not None:
        lo = best["batch"]
        hi = bsz  # failed or over budget
        # try midpoints as powers aren't necessary; step down by /2 already done
        # Fine search: lo is ok, try lo*1.5 if hi==2*lo
        cand = int(lo * 1.5)
        if cand > lo and cand < hi:
            try:
                stats = try_batch(output_dim, cand, precision)
                print(
                    f"OK batch={cand} mem={stats['mem_gb']:.2f}GB sps={stats['samples_per_sec']:.1f}"
                )
                if stats["mem_gb"] <= budget:
                    best = stats
            except RuntimeError:
                torch.cuda.empty_cache()

    if best is None:
        raise SystemExit("No feasible batch size found")

    # Suggested LR with sqrt scaling from base 64 @ 1e-3
    base_bs, base_lr = 64, 1e-3
    global_bs = best["batch"] * int(cfg["TRAIN"]["DEVICES"])
    lr = base_lr * (global_bs / base_bs) ** 0.5
    print("---")
    print(f"RECOMMENDED BATCH_SIZE_PER_GPU: {best['batch']}")
    print(f"RECOMMENDED LR (sqrt scale from 64@1e-3, devices={cfg['TRAIN']['DEVICES']}): {lr:.6g}")
    print(f"peak_mem_gb: {best['mem_gb']:.2f}")
    print(f"samples_per_sec_approx: {best['samples_per_sec']:.1f}")


if __name__ == "__main__":
    main()
