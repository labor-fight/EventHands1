#!/usr/bin/env python3
"""trk6: response gain of a stored trajectory at several time scales (no new inference).

For root-aligned joints, displacement over k steps inside a segment: dP = P[t]-P[t-k], dG likewise.
gain_k = <dP,dG>/<dG,dG> (regression slope of predicted on true motion), corr_k (cosine of centred vectors).
Splits: root-only motion (GT fingers, pred root) and finger-only motion (GT root, pred fingers).
    python tools/trk6/gain_scan.py NPZ [NPZ ...]
"""
import sys
from pathlib import Path
import numpy as np
import torch
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "model")); sys.path.insert(0, str(REPO / "tools"))
from mano_layer import ManoLayer                    # noqa: E402
from semkine.eval_track import mano_fk              # noqa: E402
from trk6.diag_traj import betas_of                 # noqa: E402
KS = (1, 2, 5, 10)


def ra_joints(mano, x, b):
    j = mano_fk(mano, x, b, torch.device("cpu"))[0]
    return j - j[:, :1]


def gains(P, G, run):
    out = {}
    for k in KS:
        m = np.r_[np.zeros(k, bool), run[k:] == run[:-k]]
        dP = (P[k:] - P[:-k])[m[k:]].reshape(-1)
        dG = (G[k:] - G[:-k])[m[k:]].reshape(-1)
        out[k] = (float(dP @ dG / max(dG @ dG, 1e-12)), float(dP @ dG / np.sqrt(max(dP @ dP * (dG @ dG), 1e-24))),
                  float(np.sqrt((dG ** 2).sum() / max(m.sum(), 1)) * 1000))
    return out


def main():
    mano = ManoLayer(str(REPO / "assets/mano_right.npz"), add_mean=False).eval()
    for f in sys.argv[1:]:
        z = np.load(f, allow_pickle=True)
        for seq in ("zgz_global", "zgz_local"):
            p = z[f"model|{seq}|pred"].astype(np.float32); g = z[f"model|{seq}|gt"].astype(np.float32)
            run = z[f"model|{seq}|run"]; b = torch.from_numpy(betas_of(seq)).view(1, -1)
            G = ra_joints(mano, g, b)
            rows = []
            for name, sl in (("all", None), ("root", slice(3, 6)), ("fing", slice(6, 51))):
                if sl is None:
                    q = p
                else:
                    q = g.copy(); q[:, sl] = p[:, sl]
                r = gains(ra_joints(mano, q, b), G, run)
                rows.append(name + " " + " ".join(f"k{k}:g{v[0]:.2f}/c{v[1]:.2f}" for k, v in r.items()))
            print(f"{Path(f).parent.name[-5:]} {Path(f).name[:4]} {seq:10s} | " + " | ".join(rows))


if __name__ == "__main__":
    main()
