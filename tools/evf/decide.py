#!/usr/bin/env python3
"""EVF written rules (docs/EXPERIMENT_RECORDS.md#evf-prereg-20261104); CPU only, read-only on the readings.

    python -m tools.evf.decide screen --arm A          2000-step screen of arm A (seed 3407) against B2 seed 3407 at
                                                       step 2000 -> decisions/screen_<A>.json
    python -m tools.evf.decide gate --arm A            exit 0 if A passed its screen, 3 if not (a closed chain)
    python -m tools.evf.decide effective --arm A       6000-step single-seed effectiveness of A against B2 seed 3407
                                                       -> decisions/effective_<A>.json
    python -m tools.evf.decide gate_effective --arm A  exit 0 if effective, 3 if not
    python -m tools.evf.decide adopt --arm A           three seeds against B2: adoption and the target
                                                       -> decisions/adopt_<A>.json
Readings: E1 five metrics (MPJPE-L / MPJPE-G / MPVPE-L / MPVPE-G on zgz_local / zgz_global, RA = overall MPJPE-RA),
E3 = overall MPJPE-RA of the single step from the GT previous state, local R15 / F15 = the median retention of an injected
15 deg root / 0.15 rad finger error on zgz_local (PFO's engine `pert`). G = sum of max(0, metric - S43) over the five.
  screen     PASS iff R15 and F15 both >= 0.10 below B2's and E3 not worse than B2's by more than 1.0 mm
  effective  RA and G both better than B2's (seed 3407, step 6000) by >= 0.5 mm
  adopt      three seeds paired: G of the seed means better than B2's by >= 0.5 mm and no seed's RA worse than B2's same
             seed by more than 0.1 mm; target = every seed-mean metric below S43 (never "how many below S43")
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.evf import arms as A  # noqa: E402
from tools.evf.common import OUT, PFO2_OUT, S43_ENGINE, SCREEN_SEED, SEEDS, now, read, require, save  # noqa: E402

SCREEN_STEP = 2000
RETENTION_DROP = 0.10
E3_SLACK = 1.0
EFFECT = 0.5
ADOPT_G = 0.5
ADOPT_RA_SLACK = 0.1


def report(root, tag, task):
    p = root / "readings" / tag / task / "report.json"
    return read(p) if p.exists() else None


def metrics(root, tag):
    e1, e3, pt = report(root, tag, "e1"), report(root, tag, "e3"), report(root, tag, "pert")
    out = {}
    if e1 is not None:
        s = e1["per_sequence"]
        out["five"] = [s["zgz_local"]["mpjpe_ra_mm"], s["zgz_global"]["mpjpe_ra_mm"], s["zgz_local"]["mpvpe_ra_mm"],
                       s["zgz_global"]["mpvpe_ra_mm"], e1["overall"]["mpjpe_ra_mm"]]
    if e3 is not None:
        out["e3"] = e3["overall"]["mpjpe_ra_mm"]
    if pt is not None:
        loc = pt["per_sequence"]["zgz_local"]
        out.update({k: loc[k]["rho_median"] for k in ("R15", "F15", "T10", "T30")})
    return out


def gap(five):
    return float(sum(max(0.0, a - b) for a, b in zip(five, S43_ENGINE)))


def screen(arm):
    b2 = metrics(OUT, f"b2_s{SCREEN_SEED}@{SCREEN_STEP}")
    me = metrics(OUT, f"{A.run_name(arm, SCREEN_SEED, 'screen')}@{SCREEN_STEP}")
    need = ("e3", "R15", "F15")
    require(all(k in b2 for k in need) and all(k in me for k in need), "screen readings missing")
    d = {"R15_drop": b2["R15"] - me["R15"], "F15_drop": b2["F15"] - me["F15"], "E3_change": me["e3"] - b2["e3"]}
    ok = d["R15_drop"] >= RETENTION_DROP and d["F15_drop"] >= RETENTION_DROP and d["E3_change"] <= E3_SLACK
    out = {"arm": arm, "seed": SCREEN_SEED, "step": SCREEN_STEP, "b2": b2, "arm_readings": me, **d,
           "rule": f"R15 and F15 >= {RETENTION_DROP} below B2, E3 <= B2 + {E3_SLACK} mm",
           "verdict": "PASS" if ok else "STOP", "decided_at_utc": now()}
    save(OUT / "decisions" / f"screen_{arm}.json", out, exclusive=True)
    print(out["verdict"], {k: round(v, 3) for k, v in d.items()}, flush=True)
    return out


def effective(arm):
    b2 = metrics(PFO2_OUT, f"pfo2_fgs_s{SCREEN_SEED}")
    me = metrics(OUT, f"{A.run_name(arm, SCREEN_SEED, 'formal')}@6000")
    require("five" in b2 and "five" in me, "E1 readings missing")
    d = {"RA_gain": b2["five"][4] - me["five"][4], "G_gain": gap(b2["five"]) - gap(me["five"]),
         "G_b2": gap(b2["five"]), "G_arm": gap(me["five"])}
    ok = d["RA_gain"] >= EFFECT and d["G_gain"] >= EFFECT
    out = {"arm": arm, "seed": SCREEN_SEED, "b2": b2, "arm_readings": me, **d,
           "rule": f"RA and G both >= {EFFECT} mm better than B2 seed {SCREEN_SEED}",
           "verdict": "EFFECTIVE" if ok else "NOT_EFFECTIVE", "decided_at_utc": now()}
    save(OUT / "decisions" / f"effective_{arm}.json", out, exclusive=True)
    print(out["verdict"], {k: round(v, 3) for k, v in d.items()}, flush=True)
    return out


def adopt(arm):
    rows = {}
    for s in SEEDS:
        b2 = metrics(PFO2_OUT, f"pfo2_fgs_s{s}")
        me = metrics(OUT, f"{A.run_name(arm, s, 'formal')}@6000")
        require("five" in b2 and "five" in me, f"E1 readings missing for seed {s}")
        rows[s] = (b2["five"], me["five"])
    b2m = np.mean([r[0] for r in rows.values()], 0)
    mem = np.mean([r[1] for r in rows.values()], 0)
    ra_worse = {s: r[1][4] - r[0][4] for s, r in rows.items()}
    g_gain = gap(b2m) - gap(mem)
    adopted = g_gain >= ADOPT_G and all(v <= ADOPT_RA_SLACK for v in ra_worse.values())
    target = all(a < b for a, b in zip(mem, S43_ENGINE))
    out = {"arm": arm, "seeds": list(SEEDS), "b2_mean": b2m.tolist(), "arm_mean": mem.tolist(), "G_b2": gap(b2m),
           "G_arm": gap(mem), "G_gain": g_gain, "RA_minus_b2_by_seed": ra_worse, "adopted": bool(adopted),
           "target_met": bool(target), "remaining_gap_to_s43": [float(a - b) for a, b in zip(mem, S43_ENGINE)],
           "decided_at_utc": now()}
    save(OUT / "decisions" / f"adopt_{arm}.json", out, exclusive=True)
    print("ADOPTED" if adopted else "NOT_ADOPTED", "TARGET_MET" if target else "TARGET_NOT_MET",
          [round(x, 2) for x in mem], flush=True)
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("verb", choices=("screen", "gate", "effective", "gate_effective", "adopt"))
    p.add_argument("--arm", required=True)
    a = p.parse_args(argv)
    A.spec(a.arm)
    if a.verb == "screen":
        screen(a.arm)
    elif a.verb == "effective":
        effective(a.arm)
    elif a.verb == "adopt":
        adopt(a.arm)
    else:
        name = "screen" if a.verb == "gate" else "effective"
        d = read(OUT / "decisions" / f"{name}_{a.arm}.json")
        sys.exit(0 if d["verdict"] in ("PASS", "EFFECTIVE") else 3)


if __name__ == "__main__":
    main()
