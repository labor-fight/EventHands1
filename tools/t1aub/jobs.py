#!/usr/bin/env python3
"""Job blocks of a T1A-UB round for the shared driver (tools/s45rt/chain_r2.py add <jobs.json>); keys `ub:r<k>:...`.

  jobs.py train --round K --out FILE        preflights (+ acceptance) of every registered arm, the formal trainings
                                            (after the arm's accepted preflight and the accepted t1areplay), their
                                            acceptance
  jobs.py eval  --round K --block B --out FILE [--after KEY ...]
                                            per network: zgz E1 (5000 / 5500 / 6000), E2, E3; readouts tl_e1 (three steps),
                                            tl_f1, tl_e3, zl_e3, tl_T10, tl_R5, tl_Fs; then one exclusive main-row block
                                            (t1aref_<B>, t1arp_<B>, every network) and the round summary
Declared host memory: trainings that read the XAG sources 260 GiB, other trainings 130, evaluations 15 (the driver's
admission). Trainings run with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (T1a's own setting, S45-TR amendment 3).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.t1aub import common as C  # noqa: E402

PY = "/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python"
ALLOC = "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
NOTE = "T1A-UB r{k} (docs/EXPERIMENT_RECORDS.md#t1aub-r{k}-prereg-20261102)"
TRACES = (("e1", 5000), ("e1", 5500), ("e1", 6000), ("e2", 6000), ("e3", 6000))
READOUTS = ("tl_e1", "tl_f1", "tl_e3", "zl_e3", "tl_T10", "tl_R5", "tl_Fs")


def reads_xag(arm):
    s = C.arm_spec(arm)
    return bool(s.get("gtfeat")) or (s.get("route") or {}).get("router") == "gt"


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT))


def train_jobs(k):
    reg = C.registry()
    arms = [a for a, s in reg["arms"].items() if s["round"] == k]
    note = NOTE.format(k=k)
    jobs = []
    for arm in arms:
        mem = 260 if reads_xag(arm) else 130
        cfg = rel(C.config_path(arm, "preflight"))
        jobs.append({"key": f"ub:r{k}:pre:{arm}", "kind": "pair", "mem_gib": mem, "note": note,
                     "command": ["/usr/bin/env", ALLOC, PY, "tools/t1aub/train.py", "--arm", arm, "--phase", "preflight",
                                 "--config", cfg]})
        jobs.append({"key": f"ub:r{k}:acc:pre:{arm}", "kind": "cpu", "deps": [f"ub:r{k}:pre:{arm}"], "note": note,
                     "command": [PY, "tools/t1aub/accept.py", "--arm", arm, "--phase", "preflight"]})
    replay = "ub:r1:acc:pre:t1areplay"
    for arm in arms:
        if "formal" not in C.phases(arm):
            continue
        mem = 260 if reads_xag(arm) else 130
        deps = [f"ub:r{k}:acc:pre:{arm}"] + ([replay] if f"ub:r{k}:acc:pre:{arm}" != replay else [])
        jobs.append({"key": f"ub:r{k}:formal:{arm}", "kind": "pair", "mem_gib": mem, "note": note, "deps": deps,
                     "command": ["/usr/bin/env", ALLOC, PY, "tools/t1aub/train.py", "--arm", arm, "--phase", "formal",
                                 "--config", rel(C.config_path(arm, "formal"))]})
        jobs.append({"key": f"ub:r{k}:acc:formal:{arm}", "kind": "cpu", "deps": [f"ub:r{k}:formal:{arm}"], "note": note,
                     "command": [PY, "tools/t1aub/accept.py", "--arm", arm, "--phase", "formal"]})
    return jobs


def eval_jobs(k, block, after=()):
    """Per network of round k: traces and readouts (after its training's acceptance), then one exclusive main-row block
    after every reading of the round, then the summary."""
    reg = C.registry()
    note = NOTE.format(k=k)
    arms = [a for a, s in reg["arms"].items() if s["round"] == k and "formal" in C.phases(a)]
    cells = [c for c, s in reg["cells"].items() if s["round"] == k]
    jobs, readings = [], []
    for net in arms + cells:
        trunk = net if net in arms else C.cell_spec(net)["trunk"]
        dep = [f"ub:r{k}:acc:formal:{trunk}", *after]
        for mode, step in TRACES:
            name = net if step == 6000 else f"{net}_{step}"
            key = f"ub:r{k}:ev:{net}:{mode}:{step}"
            jobs.append({"key": key, "kind": "gpu", "mem_gib": 15, "note": note, "deps": dep,
                         "command": [PY, "tools/t1aub/evaluate.py", "trace", "--net", name, "--mode", mode]})
            readings.append(key)
        for task in READOUTS:
            key = f"ub:r{k}:ro:{net}:{task}"
            deps = dep + ([f"ub:r{k}:ro:{net}:tl_e1"] if task == "tl_f1" else [])
            jobs.append({"key": key, "kind": "gpu", "mem_gib": 15, "note": note, "deps": deps,
                         "command": [PY, "tools/t1aub/evaluate.py", "readout", "--net", net, "--task", task]})
            readings.append(key)
        for step in (5000, 5500):
            key = f"ub:r{k}:ro:{net}_{step}:tl_e1"
            jobs.append({"key": key, "kind": "gpu", "mem_gib": 15, "note": note, "deps": dep,
                         "command": [PY, "tools/t1aub/evaluate.py", "readout", "--net", f"{net}_{step}", "--task", "tl_e1"]})
            readings.append(key)
    rows = []
    for net in [f"t1aref_{block}", f"t1arp_{block}"] + arms + cells:
        key = f"ub:r{k}:row:{block}:{net}"
        jobs.append({"key": key, "kind": "exclusive", "mem_gib": 15, "note": note, "deps": list(readings) + rows[-1:],
                     "command": [PY, "tools/t1aub/evaluate.py", "row", "--net", net, "--block", block]})
        rows.append(key)
    jobs.append({"key": f"ub:r{k}:decide", "kind": "cpu", "note": note, "deps": rows,
                 "command": [PY, "tools/t1aub/decide.py", "--round", str(k), "--block", block]})
    return jobs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="what", required=True)
    t = sub.add_parser("train")
    t.add_argument("--round", type=int, required=True)
    t.add_argument("--out", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--round", type=int, required=True)
    e.add_argument("--block", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--after", nargs="*", default=[])
    a = p.parse_args(argv)
    jobs = train_jobs(a.round) if a.what == "train" else eval_jobs(a.round, a.block, a.after)
    Path(a.out).write_text(json.dumps(jobs, indent=1) + "\n")
    print(len(jobs), "jobs ->", a.out)


if __name__ == "__main__":
    main()
