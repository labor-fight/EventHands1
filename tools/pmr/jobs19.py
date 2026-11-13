#!/usr/bin/env python3
"""ABL job lists for PMR driver 7 (docs/EXPERIMENT_RECORDS.md#abl-prereg-20261111), written once per batch and appended with
`chain7.py add`. Generating a list writes the registered configs.

    python -m tools.pmr.jobs19 r --out outputs/pmr/20261109/jobs/abl_r.json
        the RDR screen (capture v2 / v3, the written rule), the router gates, the RDR trunk stage for v2 and v3 (each gated
        on its router being selected), and the speculative path-2 rollout fine-tunes of the existing d1g0 / d1h0 bases
        (each stage behind a late `nop1` check) plus the streaming reading of the existing d1m_rol5.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr.common import PY, ROOT as RR, require  # noqa: E402

SEED = 3407
TRAIN_MEM, ROL_MEM, GPU_MEM, CPU_MEM = 130, 140, 15, 4
RDR = RR / "outputs/rdr/20261110"
R_S2D = RR / "outputs/pfo/20261103/routers/r_s2d/r_s2d_final.ckpt"
CAPTURE = RR / "outputs/evhandtrack/analysis/posthoc_20261110/router_capture.py"


def env_cmd(script, config):
    return ["env", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", "NCCL_P2P_DISABLE=1", PY, str(RR / script),
            "--config", str(config)]


def cpu(key, args, deps, note="", module="tools.pmr.abl"):
    return {"key": key, "kind": "cpu", "deps": list(deps), "mem_gib": CPU_MEM, "command": [PY, "-m", module, *args],
            "note": note}


def gpu(key, args, deps, note="", module=None, script=None):
    cmd = [PY, "-m", module, *args] if module else [PY, str(script), *args]
    return {"key": key, "kind": "gpu", "deps": list(deps), "mem_gib": GPU_MEM, "command": cmd, "note": note}


def pair(key, script, config, deps, mem, note=""):
    return {"key": key, "kind": "pair", "deps": list(deps), "mem_gib": mem, "command": env_cmd(script, config), "note": note}


def batch_r():
    from tools.pmr import arms_any5  # noqa: F401  (registry 10 in the union)
    from tools.pmr import arms10 as A10
    from tools.pmr import arms_rol2 as R2
    J = [cpu("abl:freeze:rdr", ["verify", "--stage", "rdr"], [], "manifest unchanged", module="tools.pmr.freeze"),
         cpu("abl:freeze:abl1", ["verify", "--stage", "abl1"], [], "manifest unchanged", module="tools.pmr.freeze"),
         cpu("r:wait", ["wait-routers"], ["abl:freeze:rdr", "abl:freeze:abl1"], "RDR v2 / v3 trainings finished")]
    for v in ("v2", "v3"):
        J.append(gpu(f"r:cap:{v}", ["--name", f"rdr_{v}", "--rkind", "pfr", "--ckpt", str(R_S2D), "--rdr",
                                    str(RDR / f"routers/{v}/{v}_final.ckpt")], ["r:wait"], "router capture vs labels",
                     script=CAPTURE))
    J.append(cpu("r:decide", ["--candidates", "v2,v3", "--out", str(RDR / "decisions/screen_a.json")],
                 ["r:cap:v2", "r:cap:v3"], "written screen rule (#rdr-prereg-20261110)", module="tools.rdr.decide"))
    for v in ("v2", "v3"):
        J.append(cpu(f"r:gate:{v}", ["gate-router", "--router", v], ["r:decide"], "open iff the screen selected " + v))
    # ---- RDR trunk stage (#rdr-trunk-prereg-20261110), gated on the router
    for v in ("v2", "v3"):
        arm, rol = f"d1rv{v[1:]}", f"d1rv{v[1:]}_rol5"
        pre, formal = A10.write_config(arm, SEED, "preflight"), A10.write_config(arm, SEED, "formal")
        rpre, rformal = R2.write_config(rol, SEED, "preflight"), R2.write_config(rol, SEED, "formal")
        J += [pair(f"t:pre:{arm}", "tools/pmr/train10.py", pre, [f"r:gate:{v}"], TRAIN_MEM, "preflight"),
              cpu(f"t:pre:accept:{arm}", ["accept", "--arm", arm, "--phase", "preflight"], [f"t:pre:{arm}"],
                  module="tools.pmr.run10"),
              pair(f"t:train:{arm}", "tools/pmr/train10.py", formal, [f"t:pre:accept:{arm}"], TRAIN_MEM, "from scratch"),
              cpu(f"t:accept:{arm}", ["accept", "--arm", arm, "--phase", "formal", "--seed", str(SEED)],
                  [f"t:train:{arm}"], module="tools.pmr.run10"),
              pair(f"t:rpre:{rol}", "tools/pmr/train_rol5.py", rpre, [f"t:accept:{arm}"], ROL_MEM, "rol5 preflight"),
              cpu(f"t:rpre:accept:{rol}", ["accept", "--arm", rol, "--seed", str(SEED), "--phase", "preflight"],
                  [f"t:rpre:{rol}"], module="tools.pmr.run_rol5"),
              pair(f"t:rtrain:{rol}", "tools/pmr/train_rol5.py", rformal, [f"t:rpre:accept:{rol}", f"t:accept:{arm}"],
                   ROL_MEM, "rol5"),
              cpu(f"t:raccept:{rol}", ["accept", "--arm", rol, "--seed", str(SEED), "--phase", "formal"],
                  [f"t:rtrain:{rol}"], module="tools.pmr.run_rol5"),
              gpu(f"t:fread:{rol}:str:e1", ["read", "--arm", rol, "--seed", str(SEED), "--form", "str", "--task", "e1"],
                  [f"t:raccept:{rol}"], "decisive reading", module="tools.pmr.run_forms5"),
              gpu(f"t:rread:{rol}:e1", ["read", "--arm", rol, "--seed", str(SEED), "--task", "e1"], [f"t:raccept:{rol}"],
                  "packet form, descriptive", module="tools.pmr.run_rol5"),
              gpu(f"t:read:{arm}:e1", ["read", "--arm", arm, "--seed", str(SEED), "--task", "e1"], [f"t:raccept:{rol}"],
                  "from scratch, descriptive", module="tools.pmr.run10")]
    # ---- path gates
    J.append(cpu("a:gate:p2", ["gate-path", "--path", "2"], ["r:decide"], "path 2 (r_s2d EvHandTrack ablations)"))
    for v in ("v2", "v3"):
        J.append(cpu(f"a:gate:p1:{v}", ["gate-path", "--path", "1", "--router", v], ["r:decide"],
                     "path 1 (RDR ablations) with " + v))
    # ---- speculative path 2: rollout fine-tunes of the accepted d1g0 / d1h0 bases, the d1m_rol5 streaming reading
    J.append(gpu("p2:fread:d1m_rol5:str:e1", ["read", "--arm", "d1m_rol5", "--seed", str(SEED), "--form", "str", "--task",
                                               "e1"], ["abl:freeze:abl1"], "ablation (c)", module="tools.pmr.run_forms"))
    for base in ("d1g0", "d1h0"):
        rol = f"{base}_rol5"
        rpre, rformal = R2.write_config(rol, SEED, "preflight"), R2.write_config(rol, SEED, "formal")
        J += [cpu(f"p2:nop1:{base}:a", ["nop1"], ["r:decide"], "late check"),
              pair(f"p2:rpre:{rol}", "tools/pmr/train_rol4.py", rpre, [f"p2:nop1:{base}:a"], ROL_MEM, "rol5 preflight"),
              cpu(f"p2:rpre:accept:{rol}", ["accept", "--arm", rol, "--seed", str(SEED), "--phase", "preflight"],
                  [f"p2:rpre:{rol}"], module="tools.pmr.run_rol4"),
              cpu(f"p2:nop1:{base}:b", ["nop1"], [f"p2:rpre:accept:{rol}"], "late check"),
              pair(f"p2:rtrain:{rol}", "tools/pmr/train_rol4.py", rformal, [f"p2:nop1:{base}:b"], ROL_MEM, "rol5"),
              cpu(f"p2:raccept:{rol}", ["accept", "--arm", rol, "--seed", str(SEED), "--phase", "formal"],
                  [f"p2:rtrain:{rol}"], module="tools.pmr.run_rol4"),
              gpu(f"p2:fread:{rol}:str:e1", ["read", "--arm", rol, "--seed", str(SEED), "--form", "str", "--task", "e1"],
                  [f"p2:raccept:{rol}"], "ablation", module="tools.pmr.run_forms"),
              gpu(f"p2:rread:{rol}:e1", ["read", "--arm", rol, "--seed", str(SEED), "--task", "e1"], [f"p2:raccept:{rol}"],
                  "packet form", module="tools.pmr.run_rol4")]
    keys = [j["key"] for j in J]
    require(len(keys) == len(set(keys)), "duplicate job keys")
    return J


BATCHES = {"r": batch_r}


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
