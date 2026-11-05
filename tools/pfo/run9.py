#!/usr/bin/env python3
"""PFO readings / costs of an arm trained with the pfo_trunk4 "self" variant (tools/pfo/train4.py) through its per-vertex
implementation semkine/pfo_trunk5.py (same parameters, same function; docs/EXPERIMENT_RECORDS.md#pfo-p3c-prereg-20261103),
with the frozen engine and cost harness. Reading / cost name `pfo_<arm>_s<seed>~vtx`.

    python -m tools.pfo.run9 eval --arm A --seed S --task T
    python -m tools.pfo.run9 cost --arm A --seed S [--policy det|default]
    python -m tools.pfo.run9 equiv --arm A --seed S [--task e1]
        the implementation check: the `~vtx` reading against the arm's tools/pfo/run7.py reading of the same task, per
        sequence max |prediction difference| and RA difference; PASS when every RA difference <= 0.01 mm
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.pfo import arms3 as A3  # noqa: E402
from tools.pfo import run4 as R4  # noqa: E402
from tools.pfo.arms import run_name  # noqa: E402
from tools.pfo.common import OUT, ZGZ, now, read, require, save  # noqa: E402
from tools.pfo.train4 import trunk4_variant  # noqa: E402

R4.ARMS2, R4.build_config, R4.spec = A3.ARMS3, A3.build_config, A3.spec
SUFFIX = "~vtx"
RA_TOL = 0.01


def equiv(arm, seed, task):
    name = run_name(arm, seed)
    ra, rb = (OUT / "readings" / n / task for n in (name, name + SUFFIX))
    a, b = np.load(ra / "states.npz"), np.load(rb / "states.npz")
    ja, jb = read(ra / "report.json"), read(rb / "report.json")
    res = {}
    for s in ZGZ:
        res[s] = {"max_abs_prediction_diff": float(np.abs(a[f"{s}/prediction"] - b[f"{s}/prediction"]).max()),
                  "ra_diff_mm": jb["per_sequence"][s]["mpjpe_ra_mm"] - ja["per_sequence"][s]["mpjpe_ra_mm"],
                  "mpvpe_ra_diff_mm": jb["per_sequence"][s]["mpvpe_ra_mm"] - ja["per_sequence"][s]["mpvpe_ra_mm"]}
    ok = all(abs(v["ra_diff_mm"]) <= RA_TOL and abs(v["mpvpe_ra_diff_mm"]) <= RA_TOL for v in res.values())
    out = {"arm": arm, "seed": seed, "task": task, "per_sequence": res, "tolerance_mm": RA_TOL,
           "status": "PASS" if ok else "FAIL", "checked_utc": now()}
    save(OUT / "equiv" / f"{name}{SUFFIX}_{task}.json", out)
    print(out["status"], res, flush=True)
    require(ok, "per-vertex implementation differs from the trained one")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("verb", choices=("eval", "cost", "equiv"))
    p.add_argument("--arm", required=True, choices=tuple(A3.ARMS3))
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--task", default="e1")
    p.add_argument("--policy", default="det", choices=("det", "default"))
    a = p.parse_args(argv)
    if a.verb == "equiv":
        return equiv(a.arm, a.seed, a.task)
    from semkine import pfo_trunk5 as T5
    s = A3.spec(a.arm)
    v5 = T5.from_trunk4(trunk4_variant(s["trunk"]))
    require(v5 is not None, "the arm's trunk is not a 'self' variant")
    sp = R4.spec_file(a.arm, a.seed, 6000)
    if s.get("router") and not s.get("train_router"):
        from semkine import pfo_router as PR
        PR.register()
    T5.install(v5)
    name = run_name(a.arm, a.seed) + SUFFIX
    if a.verb == "eval":
        from tools.pfo.engine import run
        run(name, a.task, None, None, (), str(sp))
    else:
        from tools.pfo.cost import run
        run(name, "fast", str(sp), None, a.policy)


if __name__ == "__main__":
    main()
