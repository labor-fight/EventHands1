#!/usr/bin/env python3
"""trk6: motion lag of a stored trajectory (no new inference).

For shift k (steps of 50 ms), compare the prediction at t with the ground truth at t-k inside the same segment:
RA-MPJPE(k). A tracker that lags by k steps reaches its minimum near k > 0. Reported: the RA-MPJPE curve, the
argmin, and a sub-step lag from the velocity cross-correlation of root-aligned joints (pred vs GT, lags -4..+6).
    python tools/trk6/lag_scan.py NPZ [NPZ ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))
sys.path.insert(0, str(REPO / "tools"))
from mano_layer import ManoLayer                                      # noqa: E402
from semkine.eval_track import mano_fk                                # noqa: E402
from trk6.diag_traj import betas_of                                   # noqa: E402

KS = range(-2, 7)


def lag_stats(mano, p, g, run, betas):
    pj = mano_fk(mano, p, betas, torch.device("cpu"))[0]
    gj = mano_fk(mano, g, betas, torch.device("cpu"))[0]
    pr, gr = pj - pj[:, :1], gj - gj[:, :1]
    curve = {}
    for k in KS:
        if k >= 0:
            ok = np.r_[np.zeros(k, bool), run[k:] == run[:len(run) - k]] if k else np.ones(len(run), bool)
            a, b = pr[k:], gr[:len(gr) - k]
            m = ok[k:]
        else:
            kk = -k
            ok = run[:len(run) - kk] == run[kk:]
            a, b = pr[:len(pr) - kk], gr[kk:]
            m = ok
        curve[k] = float(np.linalg.norm(a[m] - b[m], axis=-1).mean() * 1000)
    # velocity cross-correlation (root-aligned joints, flattened), within segments
    same = run[1:] == run[:-1]
    vp = (pr[1:] - pr[:-1]).reshape(len(pr) - 1, -1)
    vg = (gr[1:] - gr[:-1]).reshape(len(gr) - 1, -1)
    vp = vp - vp[same].mean(0)
    vg = vg - vg[same].mean(0)
    xc = {}
    for k in KS:
        if k >= 0:
            a, b = vp[k:], vg[:len(vg) - k]
            m = same[k:] & same[:len(same) - k] if k else same
        else:
            kk = -k
            a, b = vp[:len(vp) - kk], vg[kk:]
            m = same[:len(same) - kk] & same[kk:]
        xc[k] = float((a[m] * b[m]).sum() / np.sqrt((a[m] ** 2).sum() * (b[m] ** 2).sum()))
    ks = np.array(list(xc))
    vals = np.array([xc[k] for k in ks])
    i = int(vals.argmax())
    sub = float(ks[i])
    if 0 < i < len(ks) - 1:                                   # parabolic peak
        y0, y1, y2 = vals[i - 1], vals[i], vals[i + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            sub = float(ks[i] + 0.5 * (y0 - y2) / den)
    return curve, xc, sub


def main():
    mano = ManoLayer(str(REPO / "assets/mano_right.npz"), add_mean=False).eval()
    for f in sys.argv[1:]:
        z = np.load(f, allow_pickle=True)
        for seq in ("zgz_global", "zgz_local"):
            k = f"model|{seq}"
            if f"{k}|pred" not in z.files:
                continue
            betas = torch.from_numpy(betas_of(seq)).view(1, -1)
            curve, xc, sub = lag_stats(mano, z[f"{k}|pred"].astype(np.float32), z[f"{k}|gt"].astype(np.float32),
                                       z[f"{k}|run"], betas)
            kmin = min(curve, key=curve.get)
            print(f"{Path(f).parent.name:22s} {Path(f).name[:10]:10s} {seq:10s} RA(k): " +
                  " ".join(f"{k}:{v:.2f}" for k, v in curve.items()) +
                  f" | argmin {kmin} | vel-xcorr peak {sub:+.2f} steps (xc0 {xc[0]:.3f})")


if __name__ == "__main__":
    main()
