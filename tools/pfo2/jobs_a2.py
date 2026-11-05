#!/usr/bin/env python3
"""PFO2 amendment-2 job list (docs/EXPERIMENT_RECORDS.md#pfo2-amend2-20261103): the f1mp seed-3408 E1 (and f1) readings,
the f1mp rule, the two gated base chains and P0-D again under new keys (`a2:*`, `p0d2:*`), with the race-safe readings
entry tools/pfo2/run2.py; the skipped `a1:*` / `p0:d:*` keys stay in the queue untouched.

    python -m tools.pfo2.jobs_a2 --out FILE
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pfo2 import jobs as J  # noqa: E402
from tools.pfo2 import jobs_a1 as JA  # noqa: E402
from tools.pfo2.common import SEEDS, require  # noqa: E402

CPU_MEM = 64


def ev(key, arm, seed, task, step, deps):
    return J.gpu(key, ["tools.pfo2.run2", "eval", "--arm", arm, "--seed", str(seed), "--task", task, "--step",
                       str(step)], deps)


def amend2():
    e1 = "a2:eval:f1mp:3408:e1@6000"
    L = [ev(e1, "f1mp", 3408, "e1", 6000, ["a1:accept:f1mp:3408"]),
         ev("a2:eval:f1mp:3408:f1@6000", "f1mp", 3408, "f1", 6000, ["a1:accept:f1mp:3408", e1]),
         J.cpu("a2:decide:mp", ["tools.pfo2.decide", "mp"], ["a1:eval:f1mp:3407:e1@6000", e1])]
    for arm, ref, form in (("fl1", "rl1", "l1"), ("fgs", "rgs", "gshare")):
        g = f"a2:gate:{arm}"
        L.append(J.cpu(g, ["tools.pfo2.decide", "gate", "--rule", "mp", "--open", form], ["a2:decide:mp"],
                       "exit 3 = chain closed by the written rule"))
        p0d = []
        for s in SEEDS:
            t = f"a2:train:{arm}:{s}"
            L.append(JA.train_det(t, arm, s, "formal", [g, f"a1:cmp:{arm}:pre"], J.ALL))
            c = f"a2:cmp:{arm}:{s}:head"
            L.append(J.cpu(c, ["tools.pfo2.replay", "base", "--arm", arm, "--seed", str(s), "--phase", "head500"],
                           [t, f"a1:head:{ref}:{s}"]))
            acc = f"a2:accept:{arm}:{s}"
            L.append(J.cpu(acc, ["tools.pfo2.run_a1", "accept", "--arm", arm, "--seed", str(s)], [t, c]))
            first = None
            for task, step in J.READINGS:
                k = f"a2:eval:{arm}:{s}:{task}@{step}"
                L.append(ev(k, arm, s, task, step, [acc] + ([first] if task == "f1" else [])))
                if (task, step) == ("e1", 6000):
                    first = k
            r = f"p0d2:run:{arm}:{s}"
            L.append(J.gpu(r, ["tools.pfo2.p0d", "run", "--arm", arm, "--seed", str(s)],
                           [f"a2:eval:{arm}:{s}:e1@6000", f"a2:eval:{arm}:{s}:tl_e1@6000"]))
            p0d.append(r)
        d = J.cpu(f"p0d2:decide:{arm}", ["tools.pfo2.p0d", "decide", "--arm", arm], p0d)
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
    jobs = amend2()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jobs, indent=1) + "\n")
    print(len(jobs), "jobs ->", out)


if __name__ == "__main__":
    main()
