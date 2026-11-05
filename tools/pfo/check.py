#!/usr/bin/env python3
"""PFO G0 equivalence gate (docs/EXPERIMENT_RECORDS.md#pfo-prereg-20261103), CPU only.

    python -m tools.pfo.check g0

pfasrdz: the engine's E1 predictions must equal the published XCL trace bit for bit. S43 (three seeds): the engine's E1
predictions against each seed's evalx arrays `model|zgz_*|pred`: max |diff| <= 1e-3 and per-sequence RA diff <= 0.01 mm.
Writes outputs/pfo/20261103/checks/g0.json (exclusive); exits non-zero on FAIL.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.pfo.common import OUT, S43_RUN, S43_SEEDS, SM, ZGZ, now, require, save  # noqa: E402
from tools.pfo.engine import reading_dir  # noqa: E402

MAX_ABS, RA_TOL = 1e-3, 0.01


def g0():
    out = OUT / "checks" / "g0.json"
    require(not out.exists(), "G0 runs once")
    res = {"checked_at_utc": now(), "pfasrdz": {}, "s43": {}}
    ok = True
    mine = np.load(reading_dir("pfasrdz", task="e1") / "states.npz")
    ref = np.load(SM / "evaluations/pfasrdz/e1_step6000/states.npz")
    for s in ZGZ:
        a, b = mine[f"{s}/prediction"], ref[f"{s}/prediction"]
        same = a.shape == b.shape and a.dtype == b.dtype and bool(np.array_equal(a, b))
        res["pfasrdz"][s] = {"bitwise_equal": same, "max_abs": float(np.abs(a - b).max()) if a.shape == b.shape else None,
                             "ra_engine": float(mine[f"{s}/mpjpe_ra_mm"].mean()), "ra_published": float(ref[f"{s}/mpjpe_ra_mm"].mean())}
        ok &= same
    for seed in S43_SEEDS:
        mine = np.load(reading_dir(f"s43_s{seed}", task="e1") / "states.npz")
        ref = np.load(Path(str(S43_RUN).format(seed=seed)) / "evalx_val_core_last_tf_pert.npz")
        for s in ZGZ:
            a, b = mine[f"{s}/prediction"], ref[f"model|{s}|pred"]
            ma = float(np.abs(a.astype(np.float64) - b).max())
            ra_e, ra_r = float(mine[f"{s}/mpjpe_ra_mm"].mean()), float(ref[f"model|{s}|mpjpe_ra_mm"].mean())
            passed = ma <= MAX_ABS and abs(ra_e - ra_r) <= RA_TOL
            res["s43"][f"{seed}/{s}"] = {"max_abs": ma, "ra_engine": ra_e, "ra_evalx": ra_r, "pass": passed}
            ok &= passed
    res["status"] = "PASS" if ok else "FAIL"
    save(out, res, exclusive=True)
    print(res["status"], flush=True)
    require(ok, "G0 failed: see " + str(out))


if __name__ == "__main__":
    require(sys.argv[1:] == ["g0"], "usage: check.py g0")
    g0()
