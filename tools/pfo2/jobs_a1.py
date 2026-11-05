#!/usr/bin/env python3
"""PFO2 amendment-1 job list (docs/EXPERIMENT_RECORDS.md#pfo2-amend1-20261103), replacing the stage-0 chains that the
failed bitwise F replay closed (their keys stay in the queue as skipped; new keys `a1:*`).

    python -m tools.pfo2.jobs_a1 --out FILE

  * the procedure-equivalence check of PFO2's entry against F (tools/pfo2/equiv.py) gates f1mp's training
  * f1mp seeds 3407 / 3408 with F's procedure (tools/pfo2/train.py, Lightning's default kernels, as F), the standard
    readings, the written f1mp rule (tools/pfo2/decide.py, unchanged)
  * the base forms with deterministic training (tools/pfo2/train_det.py): preflights and single-change references
    (pair 6-7), one formal chain opened by the rule, bitwise checks against the references, acceptance
    (tools/pfo2/run_a1.py), the standard readings
  * F's same-seed replay (f1rep, seed 3407, 6000 steps, F's procedure) after the references: E1 / E3 at the late steps,
    the run-to-run spread of F's procedure (descriptive)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pfo2 import jobs as J  # noqa: E402
from tools.pfo2.arms import write_config  # noqa: E402
from tools.pfo2.common import PY, ROOT as R, SEEDS, require  # noqa: E402


def train_det(key, arm, seed, phase, deps=(), pairs=None, note=""):
    job = J.train(key, arm, seed, phase, deps, pairs, note)
    job["command"] = ["env", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", PY, str(R / "tools/pfo2/train_det.py"),
                      "--config", str(write_config(arm, seed, phase))]
    return job


def readings(arm, seed, accept_key, steps=J.READINGS):
    jobs, e1 = [], None
    for task, step in steps:
        k = f"a1:eval:{arm}:{seed}:{task}@{step}"
        jobs.append(J.gpu(k, ["tools.pfo2.run", "eval", "--arm", arm, "--seed", str(seed), "--task", task, "--step",
                              str(step)], [accept_key] + ([e1] if task == "f1" else [])))
        if (task, step) == ("e1", 6000):
            e1 = k
    return jobs


def amend1():
    L = [J.cpu("a1:equiv:f1rep", ["tools.pfo2.equiv"], [], "procedure equivalence of PFO2's entry and F's")]
    L.append(J.train("a1:pre:f1mp", "f1mp", 3407, "preflight", ["a1:equiv:f1rep"], J.MAIN))
    for s in (3407, 3408):
        L.append(J.train(f"a1:train:f1mp:{s}", "f1mp", s, "formal", ["a1:pre:f1mp"], J.MAIN))
        L.append(J.cpu(f"a1:accept:f1mp:{s}", ["tools.pfo2.run", "accept", "--arm", "f1mp", "--seed", str(s)],
                       [f"a1:train:f1mp:{s}"]))
        L += readings("f1mp", s, f"a1:accept:f1mp:{s}")
    refs = []
    for arm, ref in (("fl1", "rl1"), ("fgs", "rgs")):
        L.append(train_det(f"a1:pre:{ref}", ref, 3407, "preflight", [], J.SIDE))
        L.append(train_det(f"a1:pre:{arm}", arm, 3407, "preflight", [], J.SIDE))
        L.append(J.cpu(f"a1:cmp:{arm}:pre", ["tools.pfo2.replay", "base", "--arm", arm, "--seed", "3407", "--phase",
                                            "preflight"], [f"a1:pre:{ref}", f"a1:pre:{arm}"]))
        for s in SEEDS:
            L.append(train_det(f"a1:head:{ref}:{s}", ref, s, "head500", [f"a1:cmp:{arm}:pre"], J.SIDE))
            refs.append(f"a1:head:{ref}:{s}")
    L.append(J.cpu("a1:decide:mp", ["tools.pfo2.decide", "mp"], [f"a1:eval:f1mp:{s}:e1@6000" for s in (3407, 3408)]))
    for arm, ref, form in (("fl1", "rl1", "l1"), ("fgs", "rgs", "gshare")):
        g = f"a1:gate:{arm}"
        L.append(J.cpu(g, ["tools.pfo2.decide", "gate", "--rule", "mp", "--open", form], ["a1:decide:mp"],
                       "exit 3 = chain closed by the written rule"))
        for s in SEEDS:
            t = f"a1:train:{arm}:{s}"
            L.append(train_det(t, arm, s, "formal", [g, f"a1:cmp:{arm}:pre"], J.ALL))
            c = f"a1:cmp:{arm}:{s}:head"
            L.append(J.cpu(c, ["tools.pfo2.replay", "base", "--arm", arm, "--seed", str(s), "--phase", "head500"],
                           [t, f"a1:head:{ref}:{s}"]))
            acc = f"a1:accept:{arm}:{s}"
            L.append(J.cpu(acc, ["tools.pfo2.run_a1", "accept", "--arm", arm, "--seed", str(s)], [t, c]))
            L += readings(arm, s, acc)
    L.append(J.train("a1:train:f1rep:3407", "f1rep", 3407, "formal", refs, J.SIDE, "F same-seed replay"))
    L.append(J.cpu("a1:accept:f1rep:3407", ["tools.pfo2.run", "accept", "--arm", "f1rep", "--seed", "3407"],
                   ["a1:train:f1rep:3407"]))
    L += readings("f1rep", 3407, "a1:accept:f1rep:3407", (("e1", 6000), ("e3", 6000), ("e1", 5000), ("e1", 5500)))
    keys = [j["key"] for j in L]
    require(len(keys) == len(set(keys)), "duplicate keys")
    return L


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out)
    require(not out.exists(), "job file exists")
    jobs = amend1()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jobs, indent=1) + "\n")
    print(len(jobs), "jobs ->", out)


if __name__ == "__main__":
    main()
