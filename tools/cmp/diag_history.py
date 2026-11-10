#!/usr/bin/env python3
"""CMP descriptive diagnostic (docs/EXPERIMENT_RECORDS.md#cmp-diag-history-20261110): accuracy by elapsed time in a
segment, from existing E1 readings only (no GPU, no new reading).

EvHandPose's sliding form needs 12 packets of history (it was supervised on the last 3 packets of 12-packet sequences);
on the protocol's segments (windows clipped at the segment start) its first ~0.55 s are out of its training regime. This
tool splits every reading's zgz endpoints into the cold start (elapsed < 550 ms, fewer than 12 packets) and the steady
part (elapsed >= 550 ms) and reports the five table metrics of each part (endpoint-weighted over zgz_global + zgz_local
for the pooled RA, per sequence for local / global), per seed and as the three-seed mean.

    python -m tools.cmp.diag_history
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]
import numpy as np  # noqa: E402

from tools.cmp.common import EVHANDTRACK, OUT, SEEDS, now, read, save  # noqa: E402

STEADY_MS = 550
SEQS = ("zgz_local", "zgz_global")


def readings():
    out = {}
    for name in ("eh", "eh__w50", "eh__warm", "emrol5", "eg", "eg__rolling", "eg__warm", "ep", "ep__warm", "epc",
                 "epc__warm", "epc__gtcrop", "epcl", "epcl__warm", "epcl__gtcrop", "eho", "eho__kf", "eho__kff"):
        arm, _, form = name.partition("__")
        paths = [OUT / "readings" / (f"cmp_{arm}_s{s}" + (f"__{form}" if form else "")) / "e1" / "states.npz" for s in SEEDS]
        if all(p.exists() for p in paths):
            out[name] = paths
    ident = read(EVHANDTRACK)
    out["evhandtrack"] = [ROOT / Path(s["forms"]["streaming"]["reading"]).parent / "states.npz" for s in ident["seeds"]]
    return out


def split(path):
    z = np.load(path)
    res = {}
    for part, keep in (("cold", lambda e: e < STEADY_MS), ("steady", lambda e: e >= STEADY_MS), ("all", lambda e: e >= 0)):
        r, n = {}, {}
        for seq in SEQS:
            m = keep(z[f"{seq}/elapsed_ms"])
            n[seq] = int(m.sum())
            r[seq] = {k: float(z[f"{seq}/{k}"][m].mean()) if m.any() else float("nan") for k in ("mpjpe_ra_mm", "mpvpe_ra_mm")}
        tot = sum(n.values())
        pooled = sum(r[s]["mpjpe_ra_mm"] * n[s] for s in SEQS) / tot
        res[part] = [r["zgz_local"]["mpjpe_ra_mm"], r["zgz_global"]["mpjpe_ra_mm"], r["zgz_local"]["mpvpe_ra_mm"],
                     r["zgz_global"]["mpvpe_ra_mm"], pooled, n["zgz_local"], n["zgz_global"]]
    return res


def main():
    rows = {}
    for name, paths in readings().items():
        per = [split(p) for p in paths]
        rows[name] = {part: {"mean": [float(np.mean([p[part][i] for p in per])) for i in range(5)],
                             "ra_per_seed": [p[part][4] for p in per], "n_local": per[0][part][5], "n_global": per[0][part][6]}
                      for part in ("cold", "steady", "all")}
        print(f"{name:12s}", "  ".join(f"{part}: " + " / ".join(f"{v:.2f}" for v in rows[name][part]["mean"]) for part in ("all", "cold", "steady")))
    n = next(iter(rows.values()))
    print(f"endpoints: cold local {n['cold']['n_local']} / global {n['cold']['n_global']}; steady local {n['steady']['n_local']} / global {n['steady']['n_global']}")
    save(OUT / "analysis" / "diag_history.json", {"steady_ms": STEADY_MS, "rows": rows, "written_utc": now(),
                                                    "columns": ["MPJPE-L", "MPJPE-G", "MPVPE-L", "MPVPE-G", "RA"]})


if __name__ == "__main__":
    main()
