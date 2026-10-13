#!/usr/bin/env python3
"""trk6 / S42: paired summary of filter_eval outputs (S39 stack + raw + TF) for arms vs a control, per seed.
    python tools/trk6/summarize.py --root OUTROOT --ctl CTLPREFIX_s --arms ARMPREFIX_s [...] --seeds 3407 3408
           [--ctl-root DIR]   (the control's own output root, default --root)
Prints, per arm and view (filtered S39 stack / raw closed loop / raw teacher forcing): local / global RA-J and RA-V,
jitter (acc_err_ra), speed ratios, displacement gain k5 and velocity-xcorr lag (from the stored npz, model views only),
and the paired per-seed differences to the control."""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "model")); sys.path.insert(0, str(REPO / "tools"))
from mano_layer import ManoLayer                                      # noqa: E402
from trk6.diag_traj import betas_of                                   # noqa: E402
from trk6.gain_scan import gains, ra_joints                           # noqa: E402
from trk6.lag_scan import lag_stats                                   # noqa: E402

MANO = None
BETAS = {}
COLS = ("RA", "lJ", "lV", "gJ", "gV", "lacc", "gacc", "lrs", "lfs", "lk5", "gk5", "llag", "glag")


def f0(x):
    return x[0] if isinstance(x, list) else x


def load(run_dir):
    out = {}
    for f in glob.glob(str(Path(run_dir) / "*.json")):
        d = json.load(open(f))
        tag = "raw" if Path(f).name.startswith("raw") else "filt"
        for mode in ("model", "tf"):
            if mode not in d:
                continue
            m = d[mode]
            r = {}
            for part, seq in (("g", "zgz_global"), ("l", "zgz_local")):
                s = m[seq]
                r[part + "J"] = f0(s["mpjpe_ra_mm"])
                r[part + "V"] = f0(s["mpvpe_ra_mm"])
                if mode == "model" and "jitter" in s:
                    r[part + "acc"] = s["jitter"]["acc_err_ra_mm"]
                    r[part + "rs"] = s["motion"]["root_speed_ratio"]
                    r[part + "fs"] = s["motion"]["finger_speed_ratio"]
            n_g, n_l = m["zgz_global"]["n_frames"], m["zgz_local"]["n_frames"]
            r["RA"] = (r["gJ"] * n_g + r["lJ"] * n_l) / (n_g + n_l)
            npz = f.replace(".json", ".npz")
            if mode == "model" and Path(npz).exists():
                z = np.load(npz)
                for part, seq in (("g", "zgz_global"), ("l", "zgz_local")):
                    key = f"model|{seq}|pred"
                    if key not in z.files:
                        continue
                    b = BETAS.setdefault(seq, torch.from_numpy(betas_of(seq)).view(1, -1))
                    p = z[key].astype(np.float32)
                    g = z[f"model|{seq}|gt"].astype(np.float32)
                    run = z[f"model|{seq}|run"]
                    r[part + "k5"] = gains(ra_joints(MANO, p, b), ra_joints(MANO, g, b), run)[5][0]
                    r[part + "lag"] = lag_stats(MANO, p, g, run, b)[2]
            out[f"{tag}_{mode}"] = r
    return out


def main():
    global MANO
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ctl", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", default=["3407", "3408"])
    ap.add_argument("--ctl-root", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    MANO = ManoLayer(str(REPO / "assets/mano_right.npz"), add_mean=False).eval()
    data = {}
    for name in [a.ctl] + a.arms:
        root = a.ctl_root if (name == a.ctl and a.ctl_root) else a.root
        data[name] = {s: load(Path(root) / f"{name}{s}") for s in a.seeds}
    for view in ("filt_model", "raw_model", "raw_tf"):
        print(f"\n== {view}")
        print(f"{'arm':30s} " + " ".join(f"{c:>7s}" for c in COLS))
        for name in [a.ctl] + a.arms:
            rows = [data[name][s].get(view, {}) for s in a.seeds]
            mean = {c: (np.mean([r[c] for r in rows]) if all(c in r for r in rows) else np.nan) for c in COLS}
            print(f"{name:30s} " + " ".join(f"{mean[c]:7.3f}" for c in COLS))
            if name != a.ctl:
                ctl = [data[a.ctl][s].get(view, {}) for s in a.seeds]
                d = {c: [r[c] - q[c] for r, q in zip(rows, ctl) if c in r and c in q] for c in COLS}
                print(f"{'  paired d':30s} " + " ".join(f"{np.mean(d[c]):+7.3f}" if d[c] else f"{'nan':>7s}" for c in COLS))
                print(f"{'  per-seed d lJ/gJ':30s} " + "  ".join(f"{x:+.3f}/{y:+.3f}" for x, y in zip(d["lJ"], d["gJ"])))
    if a.out:
        Path(a.out).write_text(json.dumps(data, indent=1))


if __name__ == "__main__":
    main()
