#!/usr/bin/env python3
"""SBA P4 data (docs/EXPERIMENT_RECORDS.md#sba-p4-prereg-20261022): emission features of training windows.

For every 4th protocol endpoint of the base sequences of the nine training subjects (fit: ch lfz lpc lr ly lyh lyq;
selection: ycy ylf), the feature channels of `semkine/sba_emission.py` (silhouette / internal sweep per part group,
material sweep per template patch) along the label trajectory, on the window's crop. Stored sparsely: the cells with
events (counts, features, background, uniform field) and the per-channel totals over the crop -- the exact
sufficient statistics of the Poisson likelihood for any theta >= 0.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python tools/sba/p4_cache.py --shard 0 --shards 4 [--smooth]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sba import common as CM  # noqa: E402
from sba.common import SB, EF  # noqa: E402
from semkine import sba_emission as EM  # noqa: E402

OUT = CM.OUT / "p4"
FIT_SUBJ = ("ch", "lfz", "lpc", "lr", "ly", "lyh", "lyq")
SEL_SUBJ = ("ycy", "ylf")
STRIDE = 4
N_PATCHES = 96


def items():
    out = []
    for s, d in EF.training_base_seqs():
        subj = s.split("_")[0]
        role = "fit" if subj in FIT_SUBJ else "sel"
        out.append((s, d, role))
    return out


def run(shard, shards, smooth, blur):
    dev = CM.require_gpu()
    torch.set_grad_enabled(False)
    ctx = CM.Ctx(dev)
    patch_path = OUT / "patches.npy"
    if patch_path.exists():
        patch = torch.from_numpy(np.load(patch_path)).to(dev)
    else:
        patch = EM.edge_patches(ctx.geo, N_PATCHES)
        if shard == 0:
            patch_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(patch_path, patch.cpu().numpy())
    n_ch = EM.n_channels(N_PATCHES)
    t0 = time.time()
    recs = []
    jobs = []
    for s, d, role in items():
        sd_rows = CM.endpoint_rows(CM.SeqData(s, d, dev))[::STRIDE]
        jobs += [(s, d, role, r) for r in sd_rows]
    jobs = jobs[shard::shards]
    cache = {}
    for s, d, role, (rid, end, prev, first) in jobs:
        if s not in cache:
            cache = {s: CM.SeqData(s, d, dev)}
        sd = cache[s]
        if smooth:
            a, b = sd.run_bounds(end)
            x0 = torch.tensor(CM.C.smoothed_gt(sd.pos51, prev, a, b), dtype=torch.float32, device=dev)
            x1 = torch.tensor(CM.C.smoothed_gt(sd.pos51, end, a, b), dtype=torch.float32, device=dev)
        else:
            x0, x1 = sd.gt(prev), sd.gt(end)
        win = CM.window(ctx, sd, prev, end, x0, x1)
        if win.n_crop == 0:
            continue
        F = EM.features(ctx.geo, win, x0, x1, sd.betas, sd.K, patch, N_PATCHES, blur)    # (n_ch, S, h, w)
        N = win.N.sum(0)                                                                   # (S, h, w)
        cell = N > 0
        recs.append({"seq": s, "role": role, "subject": s.split("_")[0], "category": s.split("_")[1], "end": end,
                     "n_crop": int(N.sum()), "n_cells": int(N.numel()),
                     "N": N[cell].cpu().half(), "F": F[:, cell].T.contiguous().cpu().half(),
                     "B": win.bkg[cell].cpu().float(), "U": win.uni[cell].cpu().float(),
                     "F_tot": F.flatten(1).sum(1).cpu().double(), "B_tot": float(win.bkg.sum()),
                     "U_tot": float(win.uni.sum())})
        if len(recs) % 200 == 0:
            print(json.dumps({"shard": shard, "done": len(recs), "of": len(jobs), "s": round(time.time() - t0)}),
                  flush=True)
    tag = f"{'smooth' if smooth else 'raw'}_b{blur:g}"
    path = OUT / "cache" / tag / f"shard{shard:02d}of{shards:02d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"recs": recs, "n_ch": n_ch, "n_patches": N_PATCHES, "smooth": smooth, "blur": blur}, path)
    CM.receipt(path.parent / f"shard{shard:02d}", {"probe": "P4-cache", "shard": shard, "windows": len(recs),
                                                     "wall_s": time.time() - t0},
               scripts=(__file__, CM.REPO / "semkine" / "sba_emission.py"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--smooth", action="store_true")
    ap.add_argument("--blur", type=float, default=1.0)
    a = ap.parse_args()
    run(a.shard, a.shards, a.smooth, a.blur)


if __name__ == "__main__":
    main()
