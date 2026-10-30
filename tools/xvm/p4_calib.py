#!/usr/bin/env python3
"""XVM P4 post-hoc diagnostic (not a gate): calibration of the learned information and spatial error correlation.

For a frozen P4 head on the selection subjects (every 4th endpoint) and zgz: mean squared Mahalanobis residual
r^T Lambda r of the visible observed vertices (2 if calibrated), split by contour / interior, and the correlation of the
normal-component errors between mesh-neighbour (1-ring) vertex pairs that are both observed (0 if independent, the
assumption behind the P3n r*).

    python tools/xvm/p4_calib.py --run outputs/xvm/s2/p4/A_m600 --head patch_r2 --out outputs/xvm/s3/calib/patch_r2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xvm import common as C  # noqa: E402
from xvm.p4_data import P4Data, collate  # noqa: E402
from xvm import p4_train as T  # noqa: E402
from xvm.p5_solve import load_head  # noqa: E402
from semkine import xvm as XV  # noqa: E402
from semkine.xyz_mesh import mesh_ring  # noqa: E402


@torch.no_grad()
def run_split(head, geo, ds, ring, rmask, dev):
    rng = np.random.default_rng(12345)
    acc = {"maha": {"contour": [], "interior": []}, "pairs": [0.0] * 5}
    for batch in DataLoader(ds, batch_size=16, shuffle=False, num_workers=12, collate_fn=collate):
        g = geo(batch, "A", rng)
        m, L, rho, _ = head(g) if isinstance(head, T.KernelHead) else head(g, batch)
        t = T.target_frame(g, rho)
        r = (m.float() - t).unsqueeze(-1)
        q = (L.float().transpose(-1, -2) @ r).squeeze(-1)
        maha = (q * q).sum(-1)
        obs = g["vis"] & g["obs2"]
        con = g["cos"] < XV.CONTOUR_COS
        acc["maha"]["contour"].append(maha[obs & con].cpu().numpy())
        acc["maha"]["interior"].append(maha[obs & ~con].cpu().numpy())
        e = ((m[..., 0].float() - t[..., 0]) * rho)                        # normal-component error (px)
        B, V = e.shape
        en = e[:, ring.reshape(-1)].view(B, V, -1)
        on = obs[:, ring.reshape(-1)].view(B, V, -1) & obs.unsqueeze(-1) & rmask.unsqueeze(0)
        x, y = e.unsqueeze(-1).expand_as(en)[on].double(), en[on].double()
        s = acc["pairs"]
        s[0] += len(x); s[1] += float(x.sum()); s[2] += float(y.sum()); s[3] += float((x * y).sum())
        s[4] += float((x * x).sum())
    n, sx, sy, sxy, sxx = acc["pairs"]
    cov = sxy / n - sx * sy / n ** 2
    var = sxx / n - (sx / n) ** 2
    out = {k: {"n": int(sum(len(a) for a in v)), "mean_maha": float(np.concatenate(v).mean()),
               "median_maha": float(np.median(np.concatenate(v)))} for k, v in acc["maha"].items()}
    out["neighbour_error_corr"] = float(cov / max(var, 1e-12))
    out["n_pairs"] = int(n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = C.require_gpu()
    head, mem = load_head(Path(a.run), a.head, dev)
    geo = T.Geo(dev)
    idx, em = mesh_ring(geo.faces.cpu(), 778)
    ring, rmask = idx.to(dev), em.to(dev) > 0
    sae = a.head.startswith("patch")
    res = {"run": a.run, "head": a.head, "note": "post-hoc diagnostic, not a gate"}
    res["selection"] = run_split(head, geo, P4Data(C.subject_seqs(C.SELECT_SUBJ), mem, stride=4, with_sae=sae),
                                 ring, rmask, dev)
    for s, d in C.zgz_seqs():
        res[s] = run_split(head, geo, P4Data([(s, d)], mem, with_sae=sae), ring, rmask, dev)
        print(s, json.dumps(res[s]), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
