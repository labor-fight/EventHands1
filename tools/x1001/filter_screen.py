#!/usr/bin/env python3
"""x1001 zero-training screen: causal smoothing of a per-frame (absolute) arm's outputs.

For each seed, the protocol steps saved by `evalx.py eval` are filtered segment by segment, starting
from the segment's first *prediction* (no GT anywhere): translation and the 45 finger parameters by
an exponential moving average with gain a_t / a_f, the root rotation by geodesic interpolation
R_t = R_{t-1} Exp(a_r Log(R_{t-1}^T R_meas)). Gains are chosen on the development set by two-fold
cross-fitting over alternating 10 s blocks (chosen on fold A, scored on fold B and the reverse), so
the reported RA is out of selection; it is paired step by step with the unfiltered arm.

    python tools/x1001/filter_screen.py --arm x1001_cnn --seeds 3407 3408
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))
sys.path.insert(0, str(REPO / "tools" / "x1001"))
from mano_layer import ManoLayer                                      # noqa: E402
import evalx as EX                                                    # noqa: E402

PROG = Path("/data1/lyq/code/mesh/EventHands1_x1001")
GAINS = (1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2)
BLOCK_MS = 10_000


def so3_exp(w):
    return EX.aa_to_R(w)


def so3_log(R):
    tr = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    th = ((tr - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7).acos()
    w = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    return w / (2 * th.sin()[..., None]).clamp_min(1e-9) * th[..., None]


def smooth(pred, run, at, ar, af):
    out = pred.copy()
    R = EX.aa_to_R(torch.from_numpy(pred[:, 3:6]).double())
    Rout = R.clone()
    for i in range(1, len(pred)):
        if run[i] != run[i - 1]:
            continue                                   # new segment: start from its first prediction
        out[i, :3] = out[i - 1, :3] + at * (pred[i, :3] - out[i - 1, :3])
        out[i, 6:] = out[i - 1, 6:] + af * (pred[i, 6:] - out[i - 1, 6:])
        Rp = Rout[i - 1]
        Rout[i] = Rp @ so3_exp(ar * so3_log(Rp.T @ R[i]))
    out[:, 3:6] = so3_log(Rout).numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[3407, 3408])
    ap.add_argument("--ckpt", default="selected")
    a = ap.parse_args()
    mano = ManoLayer(str(REPO / "assets" / "mano_right.npz"), add_mean=False).eval()
    rep = {}
    for seed in a.seeds:
        run = PROG / "runs" / f"{a.arm}_s{seed}"
        tag = f"evalx_val_core_{a.ckpt}"
        z = np.load(run / f"{tag}.npz")
        seqs = sorted({k.split("|")[1] for k in z.files if k.startswith("model|")})
        res = {}
        for s in seqs:
            pred, gt, rr, end = (z[f"model|{s}|{k}"] for k in ("pred", "gt", "run", "end"))
            aux = np.load(f"/data1/lyq/code/mesh/EventHands/data/hand_data51/val/{s}_aux.npz", allow_pickle=True)
            r = {"betas": aux["betas"], "gt": gt}
            grid = {}
            for at, ar, af in itertools.product(GAINS, GAINS, GAINS):
                if not (at == ar or at == 1.0):            # translation either unfiltered or tied to the root
                    continue
                r["pred"] = smooth(pred, rr, at, ar, af).astype(np.float32)
                grid[(at, ar, af)] = EX.per_step_metrics(mano, r, torch.device("cpu"))["mpjpe_ra_mm"]
            res[s] = (grid, end)
        # two-fold cross-fitting over 10 s blocks, folds shared across sequences
        allend = np.concatenate([res[s][1] + (10 ** 8 if "local" in s else 0) for s in seqs])
        fold = (allend // BLOCK_MS) % 2
        keys = list(next(iter(res.values()))[0].keys())
        cat = {k: np.concatenate([res[s][0][k] for s in seqs]) for k in keys}
        base = cat[(1.0, 1.0, 1.0)]
        oos = np.zeros_like(base)
        chosen = {}
        for f in (0, 1):
            m = fold == f
            best = min(keys, key=lambda k: float(cat[k][m].mean()))
            chosen[int(1 - f)] = best
            oos[~m] = cat[best][~m]
        d = oos - base
        rep[seed] = {"unfiltered_ra": float(base.mean()), "filtered_ra_out_of_selection": float(oos.mean()),
                     "paired_delta": EX.block_ci(d, allend), "chosen_gains_by_fold": {k: list(v) for k, v in chosen.items()},
                     "per_seq": {s: {"unfiltered": float(res[s][0][(1.0, 1.0, 1.0)].mean()),
                                     "best_in_grid": float(min(res[s][0][k].mean() for k in keys))} for s in seqs}}
        print(seed, json.dumps(rep[seed], indent=1), flush=True)
    out = PROG / "reports" / f"filter_screen_{a.arm}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
