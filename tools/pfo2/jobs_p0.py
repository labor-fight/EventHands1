#!/usr/bin/env python3
"""PFO2 zero-training probe jobs (docs/EXPERIMENT_RECORDS.md#pfo2-p0-prereg-20261103).

    python -m tools.pfo2.jobs_p0 --out FILE

  P0-A  router-feature extraction (r_s2d, r_wide; one GPU each), then the CPU fit and gate
  P0-B  F seeds 3407-3409 x 21 noise cells x {E1, pert_rf} (one GPU each), the template calibration, then sigma* and gate
  P0-C  F seeds 3407-3409 injected-state captures (one GPU each), then the CPU decoders and gate
  P0-D  B2 seeds 3407-3409 moment captures after B2's E1 / tl_e1 readings, for both base forms (the form closed by the
        f1mp rule is skipped with its chain), then the CPU decoders and gate
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pfo2 import jobs as J  # noqa: E402
from tools.pfo2 import p0b  # noqa: E402
from tools.pfo2.common import SEEDS, require  # noqa: E402

CPU_MEM = 64


def probes():
    L = []
    for r in ("r_s2d", "r_wide"):
        L.append(J.gpu(f"p0:a:extract:{r}", ["tools.pfo2.p0a", "extract", "--router", r]))
    a = J.cpu("p0:a:decide", ["tools.pfo2.p0a", "decide"], [f"p0:a:extract:{r}" for r in ("r_s2d", "r_wide")])
    a["mem_gib"] = CPU_MEM
    L.append(a)
    L.append(J.cpu("p0:b:calib", ["tools.pfo2.p0b", "calib"]))
    cells = []
    for task in ("e1", "pert_rf"):
        for seed in SEEDS:
            for sg, m in p0b.cells():
                k = f"p0:b:{task}:{seed}:g{sg:g}m{int(round(m * 100))}"
                L.append(J.gpu(k, ["tools.pfo2.p0b", "run", "--seed", str(seed), "--sigma", f"{sg:g}", "--mis", f"{m:g}",
                                   "--task", task]))
                cells.append(k)
    L.append(J.cpu("p0:b:decide", ["tools.pfo2.p0b", "decide"], cells + ["p0:b:calib"]))
    for seed in SEEDS:
        L.append(J.gpu(f"p0:c:run:{seed}", ["tools.pfo2.p0c", "run", "--seed", str(seed)]))
    c = J.cpu("p0:c:decide", ["tools.pfo2.p0c", "decide"], [f"p0:c:run:{s}" for s in SEEDS])
    c["mem_gib"] = CPU_MEM
    L.append(c)
    for arm in ("fl1", "fgs"):
        runs = []
        for seed in SEEDS:
            k = f"p0:d:run:{arm}:{seed}"
            L.append(J.gpu(k, ["tools.pfo2.p0d", "run", "--arm", arm, "--seed", str(seed)],
                           [f"a1:eval:{arm}:{seed}:e1@6000", f"a1:eval:{arm}:{seed}:tl_e1@6000"]))
            runs.append(k)
        d = J.cpu(f"p0:d:decide:{arm}", ["tools.pfo2.p0d", "decide", "--arm", arm], runs)
        d["mem_gib"] = CPU_MEM
        L.append(d)
    keys = [j["key"] for j in L]
    require(len(keys) == len(set(keys)), "duplicate keys")
    return L


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out)
    require(not out.exists(), "job file exists")
    jobs = probes()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jobs, indent=1) + "\n")
    print(len(jobs), "jobs ->", out)


if __name__ == "__main__":
    main()
