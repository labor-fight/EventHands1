#!/usr/bin/env python3
"""S44 post-hoc sensitivity (declared after the P0.1 / P0.7 readings; TRAINING subjects only, never zgz): is the
plane-fit normal flow uninformative at every fit scale, or only at the preregistered ones?

For (r, dfit, memory) in a wider grid, at GT(end - 50) with oracle owners on the 18 training base sequences (every 8th
step), compare the measured normal speed s with the GT-implied normal speed s* = A_c theta*_dot / mass of the owner
point: Pearson correlation, fraction s* < 0 (direction disagreement), median relative error by |s*| bin. Local and
global separately. Descriptive; does not change any preregistered verdict.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s44 import evflow as EF  # noqa: E402

GRID = [(2, 30, 300), (2, 100, 300), (3, 100, 300), (4, 100, 300), (4, 300, 600), (6, 300, 600)]
BINS = [0, 20, 50, 150, 1e9]


@torch.no_grad()
def main():
    dev = torch.device("cuda")
    mano = EF.Mano64(dev)
    acc = {}
    for name, d in EF.training_base_seqs():
        cat = "local" if "_local" in name else "global"
        sq = EF.Seq(name, d)
        intr = sq.intrinsics()
        betas = torch.tensor(sq.betas, dtype=torch.float64, device=dev)[None]
        steps = [s for s in sq.steps() if s[2] - EF.STEP >= s[1]][::8]
        for rid, ra, end in steps:
            gp = sq.pos51[end - EF.STEP].astype(np.float64)
            rate = torch.tensor((sq.pos51[end] - gp) / (EF.STEP * 1e-3), dtype=torch.float64, device=dev)
            _, uv, z, J = mano.jacobian(torch.tensor(gp, device=dev)[None], betas, intr)
            for (r, dfit, mem) in GRID:
                lo = max(end - mem + 1, ra)
                T, R = EF.build_sae(sq, lo, end, dev)
                pol, y, x = EF.measurement_pixels(T, end)
                f = EF.plane_fit(T, pol, y, x, r, dfit * 1000.0)
                v = f["valid"] & (f["eps_px"] <= 1.0)
                if int(v.sum()) == 0:
                    continue
                ids, w, ok = EF.owners_oracle(sq, R[pol[v], y[v], x[v]].cpu().numpy(), mano.faces)
                A, mass = EF.rows_A(J[0], f["n"][v], ids, w)
                keep = mass > 1e-6
                st = (A[keep] @ rate / mass[keep]).cpu().numpy()
                s = f["s"][v][keep].cpu().numpy()
                k = (cat, r, dfit, mem)
                e = acc.setdefault(k, {"s": [], "st": []})
                e["s"].append(s)
                e["st"].append(st)
        print(name, flush=True)
    out = {}
    for (cat, r, dfit, mem), e in sorted(acc.items()):
        s, st = np.concatenate(e["s"]), np.concatenate(e["st"])
        rel = np.abs(s - st) / np.maximum(np.abs(st), 1e-9)
        out[f"{cat}|r{r}|d{dfit}|m{mem}"] = {
            "n": int(len(s)), "corr": float(np.corrcoef(s, st)[0, 1]), "frac_true_negative": float((st < 0).mean()),
            "median_true_speed": float(np.median(np.abs(st))),
            "median_rel_err_by_bin": {f"[{lo},{hi})": (int(((np.abs(st) >= lo) & (np.abs(st) < hi)).sum()),
                                                       float(np.median(rel[(np.abs(st) >= lo) & (np.abs(st) < hi)]))
                                                       if ((np.abs(st) >= lo) & (np.abs(st) < hi)).any() else None)
                                      for lo, hi in zip(BINS[:-1], BINS[1:])}}
        print(f"{cat} r{r} d{dfit} m{mem}: n {len(s)} corr {out[f'{cat}|r{r}|d{dfit}|m{mem}']['corr']:.3f} "
              f"neg {out[f'{cat}|r{r}|d{dfit}|m{mem}']['frac_true_negative']:.3f}", flush=True)
    p = EF.REPO / "outputs" / "s44" / "p07_sensitivity_train.json"
    p.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
