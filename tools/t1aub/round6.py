#!/usr/bin/env python3
"""T1A-UB round 6: registration and driver jobs of the router-location-feature arm `rf6` (CPU only; never trains or reads).

    round6.py register --tag T1AUB-R6-PREREG-20261102     registry_r6.json + configs + manifests r6x / r6 (one process)
    round6.py jobs --out FILE                              preflight, acceptance, formal training, acceptance, readings
    round6.py block --block b6 --out FILE                  the main-row block (t1aref_<B>, rf6), rf6's breakdown, the
                                                           round summary; added after the final round's decision exists

rf6 = candidate B's trunk (no event graph, readout 20 / 256) + srdwkda's predicted surface location of every sampled event as
an input feature (semkine/t1aub.py's gtfeat feature, fed by the router: tools/t1aub/r6.py) + S45's own ray association;
srdwkda at training and test (deployable as trained). Registration order as tools/t1aub/final.py: configs, the files
manifest r6x (round 6), registry_r6.json, the round manifest r6.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.t1aub import common as C  # noqa: E402
from tools.t1aub import r6  # noqa: E402

ROUND = 6
PREREG = "docs/EXPERIMENT_RECORDS.md#t1aub-r6-prereg-20261102"
NOTE = "T1A-UB r6 (docs/EXPERIMENT_RECORDS.md#t1aub-r6-prereg-20261102)"
PY = "/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python"
ALLOC = "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
REGISTRY = {"round": ROUND, "prereg": PREREG, "arms": {"rf6": {
    "eg": False, "readout": {"slot_dim": 20, "hidden": 256}, "gtfeat": True,
    "route": {"router": "srdwkda", "assoc": "face_route", "onhand_tau": 0.5},
    "note": "candidate B's trunk + srdwkda's predicted surface location as an input feature; S45's own ray association"}}}
FILES = ("tools/t1aub/r6.py", "tools/t1aub/train6.py", "tools/t1aub/evaluate6.py", "tools/t1aub/breakdown6.py",
         "tools/t1aub/decide6.py", "tools/t1aub/round6.py", "tests/test_t1aub_r6.py", "tools/t1aub/r5.py")


def _text(obj):
    return json.dumps(obj, indent=1, ensure_ascii=False) + "\n"


def config_texts():
    native = C.registry

    def merged():
        out = native()
        out["rounds"][ROUND] = PREREG
        for n, s in REGISTRY["arms"].items():
            C.require(n not in out["arms"] and n not in out["cells"], "network registered twice: " + n)
            out["arms"][n] = {**s, "round": ROUND}
        return out
    C.registry = merged
    try:
        return r6.round_files(ROUND)
    finally:
        C.registry = native


def register(tag):
    from tools.t1aub import freeze as F
    C.verify_protocol()
    reg_path = C.PACKAGE / "registry_r6.json"
    C.require(not reg_path.exists(), "registry_r6.json exists")
    for name in ("r6.json", "r6x.json"):
        C.require(not (C.MANIFESTS / name).exists(), name + " exists")
    C.section(tag)
    configs = config_texts()
    C.require(len(configs) == 2, "rf6: preflight and formal configs")
    for path, _ in configs:
        C.require(not path.exists(), "exists: " + str(path))
    for path, text in configs:
        path.write_text(text)
    F.manifest("r6x", ROUND, tag, [p for p, _ in configs] + [ROOT / f for f in FILES], "FILES")
    reg_path.write_text(_text(REGISTRY))
    F.manifest("r6", ROUND, tag, [reg_path, *(p for p, _ in configs), *(ROOT / f for f in C.TRAIN_FILES),
                                  ROOT / "tests/test_t1aub.py", *(ROOT / f for f in F.DEPENDENCIES)], "ROUND")
    bad = [q for q, ok in r6.check_configs(ROUND).items() if not ok]
    C.require(not bad, "configs differ from the generator: " + ", ".join(bad))
    C.verify_protocol()
    return [str(p) for p, _ in configs]


def train_jobs():
    arm = "rf6"
    pre, formal = (str(C.config_path(arm, ph).relative_to(ROOT)) for ph in ("preflight", "formal"))
    train = ["/usr/bin/env", ALLOC, PY, "tools/t1aub/train6.py", "--arm", arm]
    jobs = [{"key": "ub:r6:pre:rf6", "kind": "pair", "mem_gib": 130, "note": NOTE,
             "command": train + ["--phase", "preflight", "--config", pre]},
            {"key": "ub:r6:acc:pre:rf6", "kind": "cpu", "deps": ["ub:r6:pre:rf6"], "note": NOTE,
             "command": [PY, "tools/t1aub/accept.py", "--arm", arm, "--phase", "preflight"]},
            {"key": "ub:r6:formal:rf6", "kind": "pair", "mem_gib": 130, "note": NOTE,
             "deps": ["ub:r6:acc:pre:rf6", "ub:r1:acc:pre:t1areplay"],
             "command": train + ["--phase", "formal", "--config", formal]},
            {"key": "ub:r6:acc:formal:rf6", "kind": "cpu", "deps": ["ub:r6:formal:rf6"], "note": NOTE,
             "command": [PY, "tools/t1aub/accept.py", "--arm", arm, "--phase", "formal"]}]
    from tools.t1aub import jobs as JB
    dep = ["ub:r6:acc:formal:rf6"]
    for mode, step in JB.TRACES:
        jobs.append({"key": f"ub:r6:ev:rf6:{mode}:{step}", "kind": "gpu", "mem_gib": 15, "note": NOTE, "deps": dep,
                     "command": [PY, "tools/t1aub/evaluate6.py", "trace", "--net", arm if step == 6000 else f"{arm}_{step}",
                                 "--mode", mode]})
    for task in JB.READOUTS:
        jobs.append({"key": f"ub:r6:ro:rf6:{task}", "kind": "gpu", "mem_gib": 15, "note": NOTE,
                     "deps": dep + (["ub:r6:ro:rf6:tl_e1"] if task == "tl_f1" else []),
                     "command": [PY, "tools/t1aub/evaluate6.py", "readout", "--net", arm, "--task", task]})
    for step in (5000, 5500):
        jobs.append({"key": f"ub:r6:ro:rf6_{step}:tl_e1", "kind": "gpu", "mem_gib": 15, "note": NOTE, "deps": dep,
                     "command": [PY, "tools/t1aub/evaluate6.py", "readout", "--net", f"rf6_{step}", "--task", "tl_e1"]})
    return jobs


def block_jobs(block):
    readings = [j["key"] for j in train_jobs() if ":ev:" in j["key"] or ":ro:" in j["key"]]
    jobs, prev = [], ["ub:r5:decide"]
    for net in (f"t1aref_{block}", "rf6"):
        key = f"ub:r6:row:{block}:{net}"
        jobs.append({"key": key, "kind": "exclusive", "mem_gib": 15, "note": NOTE, "deps": readings + prev,
                     "command": [PY, "tools/t1aub/evaluate6.py", "row", "--net", net, "--block", block]})
        prev = [key]
    key = f"ub:r6:bd:{block}:rf6"
    jobs.append({"key": key, "kind": "exclusive", "mem_gib": 15, "note": NOTE, "deps": prev,
                 "command": [PY, "tools/t1aub/breakdown6.py", "--net", "rf6", "--block", block]})
    jobs.append({"key": "ub:r6:decide", "kind": "cpu", "note": NOTE,
                 "deps": [key, "ub:r4:row:b4:b4match", "ub:r1:decide"] + readings,
                 "command": [PY, "tools/t1aub/decide6.py", "--block", block, "--match-block", "b4"]})
    return jobs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="what", required=True)
    r = sub.add_parser("register")
    r.add_argument("--tag", required=True)
    j = sub.add_parser("jobs")
    j.add_argument("--out", required=True)
    b = sub.add_parser("block")
    b.add_argument("--block", required=True)
    b.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.what == "register":
        print(register(a.tag))
    elif a.what == "jobs":
        js = train_jobs()
        Path(a.out).write_text(json.dumps(js, indent=1) + "\n")
        print(len(js), "jobs ->", a.out)
    else:
        js = block_jobs(a.block)
        Path(a.out).write_text(json.dumps(js, indent=1) + "\n")
        print(len(js), "jobs ->", a.out)


if __name__ == "__main__":
    main()
