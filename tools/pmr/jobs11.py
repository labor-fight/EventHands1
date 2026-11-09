#!/usr/bin/env python3
"""PMR stage S3c job lists (docs/EXPERIMENT_RECORDS.md#pmr-s3c-prereg-20261109; report only, no reading enters a rule):
the S3 report items for D1's R + rol5 and for the final form, the refinement X that S5 trains (decisions/s5_auto.json)
with its rollout X_rol5. Generating a list writes nothing.

  readings  per rollout arm A_rol5, keys s3c:<A>:...:
              async  CPU, s3407: TPD's prefix / order checks in the plain and the causal form and prefix / order /
                     partition in the streaming form (tools/pmr/asynccheck.py)
              forms  3 seeds read with zero training in the causal and the streaming input form, e1 and e3
                     (tools/pmr/run_forms.py)
              erh    3 seeds on EvRealHands, tier 3d, ordinary packet form (tools/pmr/erh.py)
            D1's are appended now (its fine-tunes are accepted); X's after each S5 acceptance
  block     one cost block on GPU 4 with no other job (tools/pmr/cost5.py) after S5's verdict and every S3c reading:
            S43 executor (default, det), S43 dense (default), B2 (default), D1 and X in the reference implementation
            (default, det); then D1's and X's fast implementation (default; it refuses to run when its equivalence check
            fails, and nothing depends on it)
  tables    analysis/s3c_<A>.md for D1 and X (tools/pmr/analyze8.py)

    python -m tools.pmr.jobs11 d1 | final --arm X
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr.common import PY, SEEDS, require  # noqa: E402
from tools.pmr.jobs import cpu, gpu  # noqa: E402
from tools.pmr.jobs8 import CONTROLS  # noqa: E402

FORMS = ("causal", "str")
SHARED = (("s43:exec:default", ["--net", "s43", "--impl", "exec", "--policy", "default"]),
          ("s43:exec:det", ["--net", "s43", "--impl", "exec", "--policy", "det"]),
          ("s43:dense:default", ["--net", "s43", "--impl", "dense", "--policy", "default"]),
          ("b2:default", ["--net", "b2", "--policy", "default"]))


def exclusive(key, args, deps):
    return {"key": key, "kind": "exclusive", "deps": list(deps), "mem_gib": 15,
            "command": [PY, "-m", "tools.pmr.cost5", "run", *args], "note": "S3c cost block"}


def readings(arm, accepted=None):
    """S3c readings of arm + rol5; accepted(seed) names the acceptance job a reading waits for (None: accepted already)."""
    name = f"{arm}_rol5"
    p = f"s3c:{arm}"

    def deps(seed):
        return [accepted(seed)] if accepted else []
    J = [cpu(f"{p}:async:{form}", ["--form", form, "--arm", name, "--seed", "3407"], deps(3407),
             "asynchronous checks", module="tools.pmr.asynccheck") for form in ("plain", "causal", "str")]
    for seed in SEEDS:
        for form in FORMS:
            for task in ("e1", "e3"):
                J.append(gpu(f"{p}:form:s{seed}:{form}:{task}",
                             ["read", "--arm", name, "--seed", str(seed), "--form", form, "--task", task], deps(seed),
                             module="tools.pmr.run_forms"))
        J.append(gpu(f"{p}:erh:s{seed}", ["run", "--net", "arm", "--arm", name, "--seed", str(seed), "--tier", "3d"],
                     deps(seed), module="tools.pmr.erh"))
    return J


def d1():
    return readings("d1r")


def final(arm):
    require(arm in CONTROLS, "the final form is an S5 refinement")
    J = readings(arm, lambda seed: f"s5:raccept:{arm}_rol5:s{seed}")
    reads = [j["key"] for j in d1()] + [j["key"] for j in J]
    rows = []
    for key, args in SHARED:
        J.append(exclusive(f"s3c:cost:{key}", args, ["s5:verdict", *reads]))
        rows.append(J[-1]["key"])
    for a in ("d1r", arm):
        for policy in ("default", "det"):
            J.append(exclusive(f"s3c:cost:{a}:{policy}", ["--arm", a, "--policy", policy], ["s5:verdict", *reads]))
            rows.append(J[-1]["key"])
    for a in ("d1r", arm):
        J.append(exclusive(f"s3c:cost:{a}:fast:default", ["--arm", a, "--fast", "--policy", "default"], rows))
    for a in ("d1r", arm):
        J.append(cpu(f"s3c:table:{a}", ["s3c", "--arm", a], [*reads, *rows], "S3c report", module="tools.pmr.analyze8"))
    keys = [j["key"] for j in J]
    require(len(keys) == len(set(keys)), "duplicate job keys")
    return J


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    sub.add_parser("d1")
    f = sub.add_parser("final")
    f.add_argument("--arm", required=True, choices=tuple(CONTROLS))
    a = p.parse_args(argv)
    print(json.dumps(d1() if a.verb == "d1" else final(a.arm), indent=1))


if __name__ == "__main__":
    main()
