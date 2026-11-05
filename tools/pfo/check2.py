#!/usr/bin/env python3
"""PFO G0b (docs/EXPERIMENT_RECORDS.md#pfo-amend1-20261103): the S43 part of the equivalence gate, restated for a closed loop
that the recorded evalx run computed under PyTorch's default (non-deterministic) kernels. CPU only.

    python -m tools.pfo.check2 g0b

Per S43 seed and sequence, against the seed's evalx arrays:
  (a) first endpoint of every segment (identical incoming state): max |prediction difference| <= 1e-3
  (b) single step from the GT previous state (engine E3 vs evalx teacher forcing, non-first endpoints): per-step RA
      |difference| mean <= 0.01 mm and max <= 0.05 mm
  (c) closed loop: the engine's three-seed mean per sequence within 0.05 mm of the published S43 row
The pfasrdz part of G0 (bitwise) is read from checks/g0.json. Writes checks/g0b.json (exclusive).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.pfo.common import OUT, S43_ROW, S43_RUN, S43_SEEDS, ZGZ, now, read, require, save  # noqa: E402
from tools.pfo.engine import reading_dir  # noqa: E402


def g0b():
    out = OUT / "checks" / "g0b.json"
    require(not out.exists(), "G0b runs once")
    g0 = read(OUT / "checks" / "g0.json")
    res = {"checked_at_utc": now(), "pfasrdz_bitwise": all(v["bitwise_equal"] for v in g0["pfasrdz"].values()),
           "s43": {}, "s43_mean": {}}
    ok = res["pfasrdz_bitwise"]
    e1m = {s: [] for s in ZGZ}
    for seed in S43_SEEDS:
        ref = np.load(Path(str(S43_RUN).format(seed=seed)) / "evalx_val_core_last_tf_pert.npz")
        e1 = np.load(reading_dir(f"s43_s{seed}", task="e1") / "states.npz")
        e3 = np.load(reading_dir(f"s43_s{seed}", task="e3") / "states.npz")
        for s in ZGZ:
            run = e1[f"{s}/run_id"]
            first = np.r_[True, run[1:] != run[:-1]]
            a = float(np.abs(e1[f"{s}/prediction"][first].astype(np.float64) - ref[f"model|{s}|pred"][first]).max())
            d = np.abs(e3[f"{s}/mpjpe_ra_mm"][~first] - ref[f"tf|{s}|mpjpe_ra_mm"][~first])
            passed = a <= 1e-3 and float(d.mean()) <= 0.01 and float(d.max()) <= 0.05
            res["s43"][f"{seed}/{s}"] = {"first_step_max_abs": a, "e3_vs_tf_mean": float(d.mean()),
                                         "e3_vs_tf_max": float(d.max()), "pass": passed}
            ok &= passed
            e1m[s].append(float(e1[f"{s}/mpjpe_ra_mm"].mean()))
    row = read(S43_ROW)["two_seed_mean"]
    for s, key in (("zgz_local", "local"), ("zgz_global", "global")):
        m = float(np.mean(e1m[s]))
        res["s43_mean"][s] = {"engine": m, "published": row[key]["mpjpe_ra_mm"], "diff": m - row[key]["mpjpe_ra_mm"]}
        ok &= abs(m - row[key]["mpjpe_ra_mm"]) <= 0.05
    res["status"] = "PASS" if ok else "FAIL"
    save(out, res, exclusive=True)
    print(res["status"], res["s43_mean"], flush=True)
    require(ok, "G0b failed")


if __name__ == "__main__":
    require(sys.argv[1:] == ["g0b"], "usage: check2.py g0b")
    g0b()
