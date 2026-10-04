#!/usr/bin/env python3
"""Job JSON for sched2: the fixed DT2 filters (recommended AdaptiveFilter spec chosen on dt_dz_l3, and the
preregistered constant 0.5/1.0/0.5) applied unchanged to other runs. Usage: fa_jobs.py RUN [RUN ...] > jobs.json"""
import json, sys
REPO = "/data1/lyq/code/mesh/EventHands1"
PY = "/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python"
SPEC = f"{REPO}/outputs/dt2/filter_sweep/recommended_spec.json"
jobs = []
for run in sys.argv[1:]:
    jobs.append({"id": f"fapply_{run}", "kind": "eval", "cwd": REPO, "prio": 75, "cores": 1, "mem_mib": 2500,
                 "cmd": [PY, "tools/dt/filter_eval.py", "--run-dir", f"{REPO}/outputs/semkine/{run}", "--ckpt", "last",
                         "--filter-spec", SPEC, "--gains", "0.5,1.0,0.5", "--perturb",
                         "--out-dir", f"{REPO}/outputs/dt2/filter_apply/{run}"]})
print(json.dumps(jobs, indent=1))
