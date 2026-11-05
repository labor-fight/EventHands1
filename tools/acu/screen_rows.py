#!/usr/bin/env python3
"""ACU screen rows for the nine-column table (tools/report_table.py's format); CPU, read-only on the readings.

    python -m tools.acu.screen_rows --name N --label L --read TAGDIR [--read TAGDIR ...]
        outputs/semkine/acu_<N>_main_row.json: accuracy = the readings' E1 (PFO's engine, deterministic flags; N-seed mean);
        cost columns = B2's deployable form from EVF's final block (GPU 4 exclusive, PyTorch default flags, 600 lyq_local
        packets). A screen arm with CA has B2's parameters and layer MACs exactly (the dXYZ slots carry the CA geometry);
        CA's rasterisation / JFA time is not in the latency cell (it is measured only in ACU's final cost block).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.acu.common import now, read, require, save  # noqa: E402
from tools.b2ub.common import B2_COST  # noqa: E402

SEMKINE = ROOT / "outputs/semkine"
KEYS = ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm")


def write(name, label, reads):
    c = read(B2_COST)
    require(c["policy"] == "default", "the table's cost comes from the PyTorch-default-flag reading")
    per = {}
    for i, d in enumerate(reads):
        r = read(Path(d) / "e1" / "report.json")
        s = r["per_sequence"]
        per[str(i)] = {"reading": str(d), "checkpoint": r["identity"]["checkpoint"],
                       "local": {k: s["zgz_local"][k] for k in KEYS}, "global": {k: s["zgz_global"][k] for k in KEYS},
                       "overall": {k: r["overall"][k] for k in KEYS}}
    m = {b: {k: float(np.mean([per[s][b][k] for s in per])) for k in KEYS} for b in ("local", "global", "overall")}
    out = {"arm": name, "label": label, "split": "val_core", "n_seeds": len(per), "two_seed_mean": m, "per_seed": per,
           "latency_ms_raw": c["latency_row"]["raw_ms"], "latency_anchor_ms": c["latency_row"]["anchor_ms"],
           "latency_ms_scaled_full1p75": c["latency_row"]["scaled_ms"], "macs_forward_packet": c["macs_layers"]["median"],
           "params_total": c["params_total"], "latency_e2e_ms": c["latency_e2e_ms"], "cost_reading": str(B2_COST),
           "written_utc": now(),
           "note": "ACU screen row: PFO's engine E1 (deterministic flags); the cost columns are B2's deployable form from "
                   "EVF's final block; CA's rasterisation / JFA time is not included"}
    p = SEMKINE / f"acu_{name}_main_row.json"
    require(not p.exists(), "main row written once: " + str(p))
    save(p, out, exclusive=True)
    print("wrote", p)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--read", action="append", required=True)
    a = p.parse_args(argv)
    write(a.name, a.label, a.read)


if __name__ == "__main__":
    main()
