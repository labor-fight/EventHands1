#!/usr/bin/env python3
"""PMR screen with the user's health reading (docs/EXPERIMENT_RECORDS.md#pmr-amend2-20261109): the user decided
(2026-11-08 18:45 UTC) that the screen's health check is one-sided, i.e. a control fails it only when its single-step
E3 is worse than R's by more than 1.0 mm (the check guards against a collapsed control); the registered two-sided
reading of tools/pmr/analyze.py is kept in decisions/screen_<design>.json, unchanged.

    python -m tools.pmr.decide_h1 --design d1
        reads decisions/screen_<design>.json (written by the frozen screen) and writes
        decisions/screen_<design>_h1.json: (a) (b) (c) as written, health one-sided, the verdict
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr.common import OUT, now, read, require, save  # noqa: E402

HEALTH = 1.0


def health_one_sided(e3_noprop_minus_r, e3_mlp_minus_r):
    """Pure: a control fails only when its E3 is worse than R's by more than HEALTH."""
    return bool(e3_noprop_minus_r <= HEALTH and e3_mlp_minus_r <= HEALTH)


def decide(design):
    src = read(OUT / "decisions" / f"screen_{design}.json")
    v, n = src["verdict"], src["verdict"]["numbers"]
    h = health_one_sided(n["e3_noprop_minus_r"], n["e3_mlp_minus_r"])
    ok = v["a_necessary"] and v["b_sufficient"] and v["c_no_regress"] and h
    dst = OUT / "decisions" / f"screen_{design}_h1.json"
    require(not dst.exists(), "written once: " + str(dst))
    save(dst, {"design": design, "registered_verdict": v["verdict"], "registered_health_two_sided": v["health"],
               "health_one_sided": h, "a_necessary": v["a_necessary"], "b_sufficient": v["b_sufficient"],
               "c_no_regress": v["c_no_regress"], "numbers": n,
               "verdict": "SCREEN_PASS" if ok else "SCREEN_FAIL",
               "basis": "user decision 2026-11-08 18:45 UTC (#pmr-amend2-20261109): health one-sided",
               "written_utc": now()}, exclusive=True)
    print(design, "registered", v["verdict"], "-> one-sided health", h, "->", "SCREEN_PASS" if ok else "SCREEN_FAIL")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--design", required=True)
    decide(p.parse_args(argv).design)


if __name__ == "__main__":
    main()
