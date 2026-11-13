#!/usr/bin/env python3
"""PMR D2 screen main rows (docs/EXPERIMENT_RECORDS.md#pmr-d2-screen-cost-20261110): seed 3407, streaming form, for the
report table (tools/report_table.py).

    python -m tools.pmr.rows13 write
        outputs/semkine/pmr_d1hubfull_rol5_s3407_str_main_row.json, pmr_d1r_rol5_s3407_str_main_row.json

Accuracy = the arm's streaming e1 reading (PFO's engine); cost = tools/pmr/cost8.py's reading of the same arm (streaming
form, reference trunk, default flags; both arms back to back on one GPU, host not quiet).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import cost8 as C8  # noqa: E402
from tools.pmr.common import OUT, now, read, require, save  # noqa: E402

SEED = 3407
SEMKINE = ROOT / "outputs/semkine"
KEYS = ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm")
LABELS = {"d1r_rol5": "D1 R + rol5（EvHandTrack 的种子 3407，流式形式；参考实现，同块成本）",
          "d1hubfull_rol5": "D2 + rol5（手节点逐部位满秩收消息，种子 3407，流式形式；参考实现，同块成本）"}


def row(arm):
    tag = f"pmr_{arm}_s{SEED}__str"
    r = read(OUT / "readings" / tag / "e1" / "report.json")
    c = read(C8.report(arm))
    require(r["status"] == "COMPLETE" or r.get("finished_utc"), "reading not finished: " + tag)
    s = r["per_sequence"]
    per = {str(SEED): {"reading": f"pmr:{tag}", "checkpoint": r["identity"]["checkpoint"],
                       "local": {k: s["zgz_local"][k] for k in KEYS}, "global": {k: s["zgz_global"][k] for k in KEYS},
                       "overall": {k: r["overall"][k] for k in KEYS}}}
    return {"arm": f"pmr_{arm}", "label": LABELS[arm], "split": "val_core", "n_seeds": 1, "form": "streaming",
            "two_seed_mean": {b: dict(per[str(SEED)][b]) for b in ("local", "global", "overall")}, "per_seed": per,
            "latency_ms_raw": c["latency_row"]["raw_ms"], "latency_anchor_ms": c["latency_row"]["anchor_ms"],
            "latency_ms_scaled_full1p75": c["latency_row"]["scaled_ms"], "macs_forward_packet": c["macs_layers"]["median"],
            "params_total": c["params_total"], "latency_e2e_ms": c["latency_e2e_ms"],
            "cost_reading": str(C8.report(arm).parent), "written_utc": now(),
            "note": "PMR D2 screen row (seed 3407 only, PFO engine E1, streaming form); cost = tools/pmr/cost8.py "
                    "(streaming form, reference trunk, default flags; D1 R + rol5 and D2 + rol5 back to back on one GPU "
                    "while CMP's trainings load the host: a same-block pair, not comparable to the identity's quiet block)"}


def write():
    for arm in C8.ARMS:
        p = SEMKINE / f"pmr_{arm}_s{SEED}_str_main_row.json"
        save(p, row(arm), exclusive=True)
        print(p)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("verb", choices=("write",))
    p.parse_args(argv)
    write()


if __name__ == "__main__":
    main()
