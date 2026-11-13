#!/usr/bin/env python3
"""ABL job list, batch a (docs/EXPERIMENT_RECORDS.md#abl-prereg-20261111), written once and appended with `chain7.py add`
after batch r (tools/pmr/jobs19.py). Generating the list writes the registered configs.

    python -m tools.pmr.jobs20 a --out outputs/pmr/20261109/jobs/abl_a.json

Queue order = priority:
  1. (a) abgeo, seed 3407 (no router, needed by both paths): preflight, from scratch, rol5 preflight, rol5, readings
  2. path 1 with the selected RDR router v3, seed 3407, gated on a:gate:p1:v3: (b) abnrv3, (c) abmv3, (d) abg0v3,
     (e) abh0v3 (from scratch + rol5 each, no preflight: their trunks and the RDR router were preflighted elsewhere),
     (f) d1rv3_rol1 (rol1 of the trunk stage's accepted d1rv3)
  3. path 2, seed 3407, speculative (a late nop1 check before every training stage): (b) abnr (preflights), (f) d1r_rol1
     (preflight); (c) / (d) / (e) are batch r's d1m_rol5 reading and d1g0 / d1h0 rol5
  4. the RDR router v1 (crop only), its capture and its screen against r_s2d (decisions/screen_b.json)
  5. path 2, seeds 3408 / 3409, gated on a:gate:p2: abgeo, abnr, d1m, d1g0, d1h0 (from scratch + rol5) and d1r_rol1
Every training goes through tools/pmr/train11.py / train_rol6.py (union registry 6, the arm's own router block), every
acceptance through tools/pmr/run11.py / run_rol6.py, every streaming reading through tools/pmr/run_forms6.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr.common import PY, ROOT as RR, require  # noqa: E402
from tools.pmr.jobs19 import CAPTURE, CPU_MEM, GPU_MEM, R_S2D, RDR, ROL_MEM, TRAIN_MEM, cpu, gpu, pair  # noqa: E402

TRAIN, ROLT = "tools/pmr/train11.py", "tools/pmr/train_rol6.py"


class Chains:
    def __init__(self):
        from tools.pmr import arms_any6  # noqa: F401
        from tools.pmr import arms_any2 as U
        from tools.pmr import arms_rol2 as R2
        self.U, self.R2, self.J = U, R2, []

    def gate(self, prefix, tag, deps, spec):
        if not spec:
            return list(deps)
        key = f"{prefix}:nop1:{tag}"
        self.J.append(cpu(key, ["nop1"], deps, "late check: skip once path 1 is open"))
        return [key]

    def readings(self, prefix, rol, seed, deps):
        self.J += [gpu(f"{prefix}:fread:{rol}:s{seed}:str:e1", ["read", "--arm", rol, "--seed", str(seed), "--form", "str",
                                                                "--task", "e1"], deps, "ablation table",
                       module="tools.pmr.run_forms6"),
                   gpu(f"{prefix}:rread:{rol}:s{seed}:e1", ["read", "--arm", rol, "--seed", str(seed), "--task", "e1"], deps,
                       "packet form", module="tools.pmr.run_rol6")]

    def rollout(self, prefix, base, rol, seed, deps, preflight, spec, base_accept):
        name = f"{base}_{rol}"
        d = list(deps)
        if preflight:
            d = self.gate(prefix, f"{name}:s{seed}:c", d, spec)
            self.J += [pair(f"{prefix}:rpre:{name}:s{seed}", ROLT, self.R2.write_config(name, seed, "preflight"), d, ROL_MEM,
                            f"{rol} preflight"),
                       cpu(f"{prefix}:rpre:accept:{name}:s{seed}", ["accept", "--arm", name, "--seed", str(seed), "--phase",
                                                                     "preflight"], [f"{prefix}:rpre:{name}:s{seed}"],
                           module="tools.pmr.run_rol6")]
            d = [f"{prefix}:rpre:accept:{name}:s{seed}"] + list(base_accept)
        d = self.gate(prefix, f"{name}:s{seed}:d", d, spec)
        self.J += [pair(f"{prefix}:rtrain:{name}:s{seed}", ROLT, self.R2.write_config(name, seed, "formal"), d, ROL_MEM, rol),
                   cpu(f"{prefix}:raccept:{name}:s{seed}", ["accept", "--arm", name, "--seed", str(seed), "--phase", "formal"],
                       [f"{prefix}:rtrain:{name}:s{seed}"], module="tools.pmr.run_rol6")]
        self.readings(prefix, name, seed, [f"{prefix}:raccept:{name}:s{seed}"])

    def scratch(self, prefix, arm, seed, deps, preflight, spec):
        d = list(deps)
        if preflight:
            d = self.gate(prefix, f"{arm}:s{seed}:a", d, spec)
            self.J += [pair(f"{prefix}:pre:{arm}:s{seed}", TRAIN, self.U.write_config(arm, seed, "preflight"), d, TRAIN_MEM,
                            "preflight"),
                       cpu(f"{prefix}:pre:accept:{arm}:s{seed}", ["accept", "--arm", arm, "--phase", "preflight", "--seed",
                                                                   str(seed)], [f"{prefix}:pre:{arm}:s{seed}"],
                           module="tools.pmr.run11")]
            d = [f"{prefix}:pre:accept:{arm}:s{seed}"]
        d = self.gate(prefix, f"{arm}:s{seed}:b", d, spec)
        acc = f"{prefix}:accept:{arm}:s{seed}"
        self.J += [pair(f"{prefix}:train:{arm}:s{seed}", TRAIN, self.U.write_config(arm, seed, "formal"), d, TRAIN_MEM,
                        "from scratch"),
                   cpu(acc, ["accept", "--arm", arm, "--phase", "formal", "--seed", str(seed)],
                       [f"{prefix}:train:{arm}:s{seed}"], module="tools.pmr.run11")]
        self.rollout(prefix, arm, "rol5", seed, [acc], preflight, spec, [acc])
        self.J.append(gpu(f"{prefix}:read:{arm}:s{seed}:e1", ["read", "--arm", arm, "--seed", str(seed), "--task", "e1"],
                          [f"{prefix}:raccept:{arm}_rol5:s{seed}"], "from scratch, packet form", module="tools.pmr.run11"))


def batch_a():
    c = Chains()
    S = 3407
    c.J += [cpu("abl:freeze:abl2", ["verify", "--stage", "abl2"], [], "manifest unchanged", module="tools.pmr.freeze")]
    # 1. (a) the ray-association arm: no router, both paths
    c.scratch("a0", "abgeo", S, ["abl:freeze:abl2"], preflight=True, spec=False)
    # 2. path 1 with v3
    g1 = ["a:gate:p1:v3", "abl:freeze:abl2"]
    for arm in ("abnrv3", "abmv3", "abg0v3", "abh0v3"):
        c.scratch("a1", arm, S, g1, preflight=False, spec=False)
    c.rollout("a1", "d1rv3", "rol1", S, g1 + ["t:accept:d1rv3"], preflight=True, spec=False, base_accept=["t:accept:d1rv3"])
    # 3. path 2 speculative
    c.scratch("a2", "abnr", S, ["r:decide", "abl:freeze:abl2"], preflight=True, spec=True)
    c.rollout("a2", "d1r", "rol1", S, ["r:decide", "abl:freeze:abl2"], preflight=True, spec=True, base_accept=[])
    # 4. router v1
    c.J += [{"key": "v1:train", "kind": "pair", "deps": ["r:decide", "abl:freeze:abl2"], "mem_gib": TRAIN_MEM,
             "command": ["env", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", "NCCL_P2P_DISABLE=1", PY,
                         str(RR / "tools/rdr/train.py"), "--variant", "v1", "--phase", "formal", "--devices", "2", "--batch",
                         "256", "--accum", "2", "--workers", "16"], "note": "RDR router v1 (crop only)"},
            gpu("v1:cap", ["--name", "rdr_v1", "--rkind", "pfr", "--ckpt", str(R_S2D), "--rdr",
                           str(RDR / "routers/v1/v1_final.ckpt")], ["v1:train"], "router capture vs labels", script=CAPTURE),
            cpu("v1:decide", ["--candidates", "v1", "--out", str(RDR / "decisions/screen_b.json")], ["v1:cap"],
                "the screen rule for v1", module="tools.rdr.decide")]
    # 5. path 2, more seeds
    for seed in (3408, 3409):
        for arm in ("abgeo", "abnr", "d1m", "d1g0", "d1h0"):
            c.scratch("a3", arm, seed, ["a:gate:p2", "abl:freeze:abl2"], preflight=False, spec=False)
        c.rollout("a3", "d1r", "rol1", seed, ["a:gate:p2", "abl:freeze:abl2"], preflight=False, spec=False, base_accept=[])
    keys = [j["key"] for j in c.J]
    require(len(keys) == len(set(keys)), "duplicate job keys")
    return c.J


BATCHES = {"a": batch_a}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("batch", choices=tuple(BATCHES))
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out)
    require(not out.exists(), "written once: " + str(out))
    jobs = BATCHES[a.batch]()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jobs, indent=1) + "\n")
    print(len(jobs), "jobs ->", out)


if __name__ == "__main__":
    main()
