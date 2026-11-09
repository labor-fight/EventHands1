#!/usr/bin/env python3
"""PMR stage S3a job list (docs/EXPERIMENT_RECORDS.md#pmr-prereg-20261109 and the design's S3a section), written once
and appended with `chain.py add`. Generating the list writes the registered rollout configs.

    python -m tools.pmr.jobs4 s3a --design d1 --freeze s3ad1 --out outputs/pmr/20261109/jobs/s3a_d1.json

Rollout fine-tunes (tools/pmr/train_rol.py, rol5): freeze check; one preflight (pair, 40 steps) per variant at seed
3407 and its acceptance; the formal fine-tunes R + rol5 x 3407 / 3408 / 3409, R-noprop + rol5 and R-mlp + rol5 x 3407
(pair, each gated on its variant's preflight acceptance and on its base arm's acceptance: S1 for 3407, S2 for
3408 / 3409); acceptance; e1 / e3 / pert, f1 after e1; the written verdict (tools/pmr/analyze3.py). A rollout training
declares ROL_MEM GiB of host memory, so at most two run together beside the from-scratch trainings (/dev/shm: TPD
amendment 3).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import arms_rol as R  # noqa: E402
from tools.pmr.common import PY, ROOT as RR, require  # noqa: E402
from tools.pmr.jobs import TASKS, cpu, gpu  # noqa: E402

ROL_MEM = 140


def rol_train(key, name, seed, phase, deps):
    return {"key": key, "kind": "pair", "deps": list(deps), "mem_gib": ROL_MEM,
            "command": ["env", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", "NCCL_P2P_DISABLE=1", PY,
                        str(RR / "tools/pmr/train_rol.py"), "--config", str(R.write_config(name, seed, phase))],
            "note": f"{name} {phase} s{seed}"}


def base_accept(design, code, seed):
    return f"s1:{design}:accept:{design}{code}" if seed == 3407 else f"s2:{design}:accept:{design}{code}:s{seed}"


def s3a(design, freeze):
    p = f"s3a:{design}"
    fine = [(f"{design}r_rol5", s, "r") for s in (3407, 3408, 3409)] + [(f"{design}n_rol5", 3407, "n"),
                                                                        (f"{design}m_rol5", 3407, "m")]
    J = [cpu(f"{p}:freeze", ["verify", "--stage", freeze], [], "manifest unchanged", module="tools.pmr.freeze")]
    for code in ("r", "n", "m"):
        name = f"{design}{code}_rol5"
        J.append(rol_train(f"{p}:pre:{name}", name, 3407, "preflight", [f"{p}:freeze", base_accept(design, code, 3407)]))
        J.append(cpu(f"{p}:pre:accept:{name}", ["accept", "--arm", name, "--seed", "3407", "--phase", "preflight"],
                     [f"{p}:pre:{name}"], "preflight acceptance", module="tools.pmr.run_rol"))
    for name, seed, code in fine:
        J.append(rol_train(f"{p}:train:{name}:s{seed}", name, seed, "formal",
                           [f"{p}:pre:accept:{name}", base_accept(design, code, seed)]))
        J.append(cpu(f"{p}:accept:{name}:s{seed}", ["accept", "--arm", name, "--seed", str(seed), "--phase", "formal"],
                     [f"{p}:train:{name}:s{seed}"], "acceptance", module="tools.pmr.run_rol"))
    reads = []
    for name, seed, _ in fine:
        for task in TASKS:
            J.append(gpu(f"{p}:read:{name}:s{seed}:{task}", ["read", "--arm", name, "--seed", str(seed), "--task", task],
                         [f"{p}:accept:{name}:s{seed}"], module="tools.pmr.run_rol"))
            reads.append(J[-1]["key"])
        J.append(gpu(f"{p}:read:{name}:s{seed}:f1", ["read", "--arm", name, "--seed", str(seed), "--task", "f1"],
                     [f"{p}:read:{name}:s{seed}:e1"], module="tools.pmr.run_rol"))
        reads.append(J[-1]["key"])
    J.append(cpu(f"{p}:verdict", ["s3a", "--design", design], reads, "written verdict", module="tools.pmr.analyze3"))
    keys = [j["key"] for j in J]
    require(len(keys) == len(set(keys)), "duplicate job keys")
    return J


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("s3a")
    s.add_argument("--design", required=True)
    s.add_argument("--freeze", required=True)
    s.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out)
    require(not out.exists(), "written once: " + str(out))
    jobs = s3a(a.design, a.freeze)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jobs, indent=1) + "\n")
    print(len(jobs), "jobs ->", out)


if __name__ == "__main__":
    main()
