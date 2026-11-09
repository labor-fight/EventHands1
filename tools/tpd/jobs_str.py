#!/usr/bin/env python3
"""TPD stage STR jobs (docs/EXPERIMENT_RECORDS.md#tpd-str-prereg-20261106) for driver 5 (tools/tpd/chain4.py, GPUs 4-7):
writes the configs and outputs/tpd/20261105/jobs/str.json (once).

    python -m tools.tpd.jobs_str
    TPD_DRIVER_DIR=driver5 python tools/tpd/chain4.py add outputs/tpd/20261105/jobs/str.json

freeze verify -> preflight (40 steps) -> acceptance -> formal 1500 steps -> acceptance -> e1 / e3 / pert / f1 ->
asynchronous receipt -> screen (which appends seeds 3408 / 3409 on STR_BETTER / STR_EQUIV); rol5 3407 read in the
streaming form with zero training (e1 / e3, description) beside them.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.tpd import ns_str  # noqa: E402

ns_str.enter()
from tools.tpd import arms_final as A  # noqa: E402
from tools.tpd.common import OUT, PY, require  # noqa: E402

ARM = ns_str.ARM
TASKS = ("e1", "e3", "pert", "f1")


def train(key, cfg, deps, note):
    return {"key": key, "kind": "pair", "mem_gib": 130, "deps": deps, "note": note,
            "command": ["env", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", PY,
                        str(ROOT / "tools/tpd/train_str.py"), "--config", str(cfg)]}


def jobs():
    pre, formal = A.write_config(ARM, 3407, "preflight"), A.write_config(ARM, 3407, "formal")
    for seed in (3408, 3409):
        A.write_config(ARM, seed, "formal")
    run = [PY, "-m", "tools.tpd.run_str"]
    j = [{"key": "str:freeze", "kind": "cpu", "deps": [], "command": [PY, "-m", "tools.tpd.freeze", "verify", "--stage", "str"]},
         train("str:train:s3407:preflight", pre, ["str:freeze"], "TPD STR: seed 3407 preflight (40 steps)"),
         {"key": "str:accept:s3407:preflight", "kind": "cpu", "deps": ["str:train:s3407:preflight"],
          "command": run + ["accept", "--seed", "3407", "--phase", "preflight"]},
         train("str:train:s3407", formal, ["str:accept:s3407:preflight"], "TPD STR: seed 3407 (screen)"),
         {"key": "str:accept:s3407", "kind": "cpu", "deps": ["str:train:s3407"],
          "command": run + ["accept", "--seed", "3407", "--phase", "formal"]}]
    reads = []
    for task in TASKS:
        key = f"str:read:s3407:{task}"
        j.append({"key": key, "kind": "gpu", "deps": ["str:accept:s3407"] + (["str:read:s3407:e1"] if task == "f1" else []),
                  "command": run + ["read", "--seed", "3407", "--task", task]})
        reads.append(key)
    j.append({"key": "str:async:s3407", "kind": "cpu", "mem_gib": 20, "deps": ["str:read:s3407:e1"],
              "command": [PY, "-m", "tools.tpd.asynccheck_str", "--name", "str_s3407",
                          "--spec", str(OUT / "specs" / "tpd_str_s3407.json")]})
    for task in ("e1", "e3"):
        j.append({"key": f"str:ref:{task}", "kind": "gpu", "deps": ["str:freeze"], "command": run + ["ref", "--task", task],
                  "note": "rol5 3407 read in the streaming form, zero training (description)"})
    j.append({"key": "str:screen", "kind": "cpu", "deps": reads + ["str:async:s3407"], "command": run + ["screen"]})
    return j


def main():
    out = OUT / "jobs" / "str.json"
    require(not out.exists(), "written once: " + str(out))
    out.write_text(json.dumps(jobs(), indent=1) + "\n")
    print(out)


if __name__ == "__main__":
    main()
