#!/usr/bin/env python3
"""PMR design D2 single-seed screen, job list for driver 5 (docs/EXPERIMENT_RECORDS.md#pmr-d2-screen-prereg-20261110,
#pmr-amend9-20261110), written once and appended with `chain5.py add`.

    python -m tools.pmr.jobs17 s9x --out outputs/pmr/20261109/jobs/s9x.json

Seed 3407 only, S9's registered commands and configs unchanged (tools/pmr/jobs16.py): d1hubfull from scratch (its 40-step
preflight was accepted on 2026-11-09 and is not repeated), acceptance, e1 / e3; the rol5 preflight and fine-tune on the
direct pair (5, 7) (#pmr-amend8-20261109), acceptance; the streaming-form e1 / e3 (tools/pmr/run_forms4.py), the packet
e1 / e3 / pert / f1; the from-scratch pert / f1 after the fine-tune (they would otherwise hold the direct pair); the
written screen (tools/pmr/analyze13.py). No cost reading in the screen.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import jobs16 as J16  # noqa: E402
from tools.pmr.common import require  # noqa: E402
from tools.pmr.jobs import cpu, gpu  # noqa: E402

ARM = J16.ARM
ROL = f"{ARM}_rol5"
SEED = 3407
ROL_PAIRS = [[5, 7]]
P = "s9x"


def rekey(key):
    return P + key[len("s9"):]


def s9x():
    src = {j["key"]: j for j in J16.s9("s9")}

    def take(key, deps, **over):
        j = src[key]
        return {"key": rekey(key), "kind": j["kind"], "deps": list(deps), "mem_gib": j.get("mem_gib"), "note": j["note"],
                "command": j["command"], **({"pairs": ROL_PAIRS} if "pairs" in j else {}), **over}

    b, r = f"{ARM}:s{SEED}", f"{ROL}:s{SEED}"
    J = [cpu(f"{P}:freeze:s9", ["verify", "--stage", "s9"], [], "manifest unchanged", module="tools.pmr.freeze"),
         cpu(f"{P}:freeze:amend9", ["verify", "--stage", "amend9"], [], "manifest unchanged", module="tools.pmr.freeze")]
    J.append(take(f"s9:train:{b}", [f"{P}:freeze:s9", f"{P}:freeze:amend9"]))
    J.append(take(f"s9:accept:{b}", [f"{P}:train:{b}"]))
    for task in ("e1", "e3"):
        J.append(take(f"s9:read:{b}:{task}", [f"{P}:accept:{b}"]))
    J.append(take(f"s9:rpre:{ROL}", [f"{P}:accept:{b}"]))
    J.append(take(f"s9:rpre:accept:{ROL}", [f"{P}:rpre:{ROL}"]))
    J.append(take(f"s9:rtrain:{r}", [f"{P}:rpre:accept:{ROL}", f"{P}:accept:{b}"]))
    J.append(take(f"s9:raccept:{r}", [f"{P}:rtrain:{r}"]))
    J.append(gpu(f"{P}:fread:{r}:str:e1", ["read", "--arm", ROL, "--seed", str(SEED), "--form", "str", "--task", "e1"],
                 [f"{P}:raccept:{r}"], "streaming form", module="tools.pmr.run_forms4"))
    J.append(take(f"s9:rread:{r}:e1", [f"{P}:raccept:{r}"]))
    J.append(gpu(f"{P}:fread:{r}:str:e3", ["read", "--arm", ROL, "--seed", str(SEED), "--form", "str", "--task", "e3"],
                 [f"{P}:raccept:{r}"], "streaming form", module="tools.pmr.run_forms4"))
    J.append(take(f"s9:rread:{r}:e3", [f"{P}:raccept:{r}"]))
    J.append(take(f"s9:rread:{r}:pert", [f"{P}:raccept:{r}"]))
    J.append(take(f"s9:rread:{r}:f1", [f"{P}:rread:{r}:e1"]))
    J.append(take(f"s9:read:{b}:pert", [f"{P}:accept:{b}", f"{P}:raccept:{r}"]))
    J.append(take(f"s9:read:{b}:f1", [f"{P}:read:{b}:e1", f"{P}:raccept:{r}"]))
    reads = [j["key"] for j in J if ":read:" in j["key"] or ":rread:" in j["key"] or ":fread:" in j["key"]]
    J.append(cpu(f"{P}:screen", ["screen"], reads, "written screen", module="tools.pmr.analyze13"))
    for j in J:
        if j.get("mem_gib") is None:
            j.pop("mem_gib", None)
    keys = [j["key"] for j in J]
    require(len(keys) == len(set(keys)), "duplicate job keys")
    return J


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("s9x")
    s.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out)
    require(not out.exists(), "written once: " + str(out))
    jobs = s9x()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jobs, indent=1) + "\n")
    print(len(jobs), "jobs ->", out)


if __name__ == "__main__":
    main()
