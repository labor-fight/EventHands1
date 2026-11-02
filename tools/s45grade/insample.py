#!/usr/bin/env python3
"""S45-GRADE in-sample late checkpoints (docs/EXPERIMENT_RECORDS.md#s45grade-prereg-20261030): S45-TR's frozen tl_e1
readout (own closed loop on the nine training local recordings; tools/s45tr/readouts.py run_task, unchanged) for N2 /
N4 / T1a at steps 5000 / 5500, and the registered grading of each arm's in-sample gain over S45 at 5000 / 5500 / 6000.

  insample.py freeze                   outputs/s45grade/20261030/insample/protocol.json (exclusive)
  insample.py read --net <arm>_<step>  one GPU among 2-7; readouts/<net>/tl_e1 (step 6000 = reproduction check)
  insample.py accept                   the 6000 re-reads must equal S45-TR's published arrays bit for bit
  insample.py decide                   registered grading -> decision.json (exclusive); needs accept PASS
  insample.py chain                    background driver: reproduction -> accept -> 5000 / 5500 -> decide
Only the network loader differs from S45-TR's: the checkpoint of the given step through tools/s45tr/evaluate.resolve
(SHA against the formal receipt and this protocol) and the arm's evaluation hooks.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.frozen import accepts  # noqa: E402
from tools.s45tr.common import OUT as S45TR, now, read, require, save, sha  # noqa: E402

OUT = ROOT / "outputs/s45grade/20261030/insample"
ARMS = ("n2", "n4", "t1a")
STEPS = (5000, 5500, 6000)
NEW = tuple(f"{a}_{s}" for a in ARMS for s in (5000, 5500))
REPRO = tuple(f"{a}_6000" for a in ARMS)
GPUS = (2, 3, 4, 5, 6, 7)
TH = {"gain_mm": 2.5, "recordings_min": 7, "harm_recordings_max": 2}
FILES = ("tools/s45grade/insample.py", "tools/s45tr/readouts.py", "tools/s45tr/evaluate.py", "tools/s45tr/common.py")
STATEMENT = {
    "robust_gain": "A: the in-sample closed-loop gain holds at all three late checkpoints of this run; zgz stays undetermined",
    "robust_harm": "A (negative): worse in-sample at all three late checkpoints of this run",
    "weak": "B: same direction but not every checkpoint beyond 2.5 mm / 7 of 9 recordings; descriptive only",
    "null": "C: every checkpoint within 2.5 mm (both statistics); no in-sample effect detected",
    "unstable": "C: checkpoints beyond 2.5 mm in opposite directions; no conclusion",
}


def s45_states(step):
    if step == 6000:
        return S45TR / "readouts/s45/tl_e1/states.npz"
    return ROOT / f"outputs/s45ct/20261030/readouts/s45_{step}/tl_e1/states.npz"


def published(arm):
    return S45TR / "readouts" / arm / "tl_e1" / "states.npz"


def reread(net):
    return OUT / "readouts" / net / "tl_e1" / "states.npz"


def arm_states(arm, step):
    return published(arm) if step == 6000 else reread(f"{arm}_{step}")


def protocol():
    return read(OUT / "protocol.json")


def verify():
    p = protocol()
    for f, digest in p["file_sha256"].items():
        require(accepts(ROOT / f, digest), "frozen file changed: " + f)
    for step in STEPS:
        require(sha(s45_states(step)) == p["s45_reference_sha256"][str(step)], f"S45 reference {step} changed")
    for arm in ARMS:
        require(sha(published(arm)) == p["published_6000_sha256"][arm], f"published {arm} readout changed")
    return p


# --------------------------------------------------------------------------- freeze
def freeze():
    target = OUT / "protocol.json"
    require(not target.exists(), "protocol is frozen once: " + str(target))
    checkpoints = {}
    for arm in ARMS:
        receipt = read(S45TR / "receipts" / f"formal_{arm}.json")
        require(receipt["status"] == "PASS", "arm has no verified formal training: " + arm)
        checkpoints[arm] = {str(s): receipt["checkpoint_sha256_by_step"][str(s)] for s in STEPS}
    OUT.mkdir(parents=True, exist_ok=True)
    save(target, {
        "study": "S45-GRADE in-sample late checkpoints", "prereg": "docs/EXPERIMENT_RECORDS.md#s45grade-prereg-20261030",
        "task": "tl_e1", "arms": ARMS, "steps": STEPS, "new_reads": NEW, "reproduction_reads": REPRO,
        "thresholds": TH, "statements": STATEMENT, "gpus": GPUS, "max_processes_per_gpu": 2,
        "checkpoint_sha256": checkpoints,
        "s45_reference_sha256": {str(s): sha(s45_states(s)) for s in STEPS},
        "published_6000_sha256": {a: sha(published(a)) for a in ARMS},
        "file_sha256": {f: sha(ROOT / f) for f in FILES}, "frozen_utc": now()}, exclusive=True)
    print(target, flush=True)


# --------------------------------------------------------------------------- read
def read_net(net):
    require(net in NEW + REPRO, "unregistered read: " + net)
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in tuple(map(str, GPUS)), "one GPU among 2-7")
    p = verify()
    if net in NEW:
        require(read(OUT / "acceptance.json")["status"] == "PASS", "the 6000 reproduction must pass first")
    arm, step = net.rsplit("_", 1)
    from tools.xcl import evaluate as XE
    XE.configure_environment()
    XE.set_policy()
    from tools.s45tr import readouts as R
    from tools.s45tr.evaluate import install_eval_hooks, resolve

    def load(_net, device):
        from config import load_config
        from model import MNISTModel
        checkpoint, config = resolve(arm, int(step))
        require(sha(checkpoint) == p["checkpoint_sha256"][arm][step], "checkpoint differs from the protocol")
        cfg = load_config(str(config))
        install_eval_hooks(cfg)
        model = MNISTModel.load_from_checkpoint(str(checkpoint), cfg=cfg, map_location=device, strict=True)
        return model.to(device).eval(), cfg, checkpoint
    R.load = load
    R.run_task(net, "tl_e1", out_root=OUT / "readouts")


# --------------------------------------------------------------------------- accept / decide
def equal_npz(a_path, b_path):
    with np.load(a_path, allow_pickle=False) as a, np.load(b_path, allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            return False
        for k in b.files:
            x, y = a[k], b[k]
            if x.dtype != y.dtype or x.shape != y.shape:
                return False
            if not np.array_equal(x, y, equal_nan=x.dtype.kind in "fc"):
                return False
    return True


def accept():
    verify()
    target = OUT / "acceptance.json"
    require(not target.exists(), "acceptance is read once")
    checks = {arm: bool(reread(f"{arm}_6000").is_file() and equal_npz(reread(f"{arm}_6000"), published(arm)))
              for arm in ARMS}
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "bitwise_equal_to_published_6000": checks,
              "computed_utc": now()}
    save(target, result, exclusive=True)
    print(result, flush=True)
    require(result["status"] == "PASS", "reproduction failed; the 5000 / 5500 reads are not opened")


def per_recording(path):
    with np.load(path, allow_pickle=False) as z:
        recs = sorted({k.split("/")[0] for k in z.files})
        return {r: float(z[f"{r}/mpjpe_ra_mm"].mean()) for r in recs}


def grade(rows):
    g = TH["gain_mm"]
    if all(r["paired_median_mm"] <= -g and r["median_difference_mm"] <= -g and r["better"] >= TH["recordings_min"]
           for r in rows):
        return "robust_gain"
    if all(r["paired_median_mm"] >= g and r["median_difference_mm"] >= g and r["better"] <= TH["harm_recordings_max"]
           for r in rows):
        return "robust_harm"
    if any(r["paired_median_mm"] <= -g for r in rows) and any(r["paired_median_mm"] >= g for r in rows):
        return "unstable"
    if all(abs(r["paired_median_mm"]) < g and abs(r["median_difference_mm"]) < g for r in rows):
        return "null"
    return "weak"


def decide():
    verify()
    require(read(OUT / "acceptance.json")["status"] == "PASS", "acceptance must pass")
    target = OUT / "decision.json"
    require(not target.exists(), "decision is read once")
    base = {s: per_recording(s45_states(s)) for s in STEPS}
    arms = {}
    for arm in ARMS:
        rows = []
        for s in STEPS:
            e = per_recording(arm_states(arm, s))
            require(sorted(e) == sorted(base[s]), "recording sets differ")
            d = np.array([e[r] - base[s][r] for r in sorted(e)])
            rows.append({"step": s, "median_mm": float(np.median(list(e.values()))),
                         "s45_median_mm": float(np.median(list(base[s].values()))),
                         "paired_median_mm": float(np.median(d)),
                         "median_difference_mm": float(np.median(list(e.values())) - np.median(list(base[s].values()))),
                         "better": int((d < 0).sum()), "recordings": int(len(d)),
                         "per_recording_minus_s45_mm": {r: float(x) for r, x in zip(sorted(e), d)}})
        cls = grade(rows)
        arms[arm] = {"rows": rows, "class": cls, "statement": STATEMENT[cls]}
    result = {"arms": arms, "thresholds": TH, "single_seed_descriptive": True, "computed_utc": now()}
    save(target, result, exclusive=True)
    for arm, v in arms.items():
        print(arm, v["class"], [(r["step"], round(r["paired_median_mm"], 2), round(r["median_difference_mm"], 2),
                                 f'{r["better"]}/{r["recordings"]}') for r in v["rows"]], flush=True)


# --------------------------------------------------------------------------- chain
def status(update):
    path = OUT / "status.json"
    s = read(path) if path.is_file() else {"started_utc": now(), "pid": os.getpid(), "jobs": {}}
    s.update(update)
    save(path, s)


def job_status(name, value):
    path = OUT / "status.json"
    s = read(path)
    s["jobs"][name] = value
    save(path, s)


def run_reads(nets):
    """One process per GPU, at most len(GPUS) at once; failures are preserved and stop the chain."""
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    procs = {}
    for net, gpu in zip(nets, GPUS):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        log = open(OUT / "logs" / f"read_{net}.log", "w")
        procs[net] = (subprocess.Popen([sys.executable, __file__, "read", "--net", net], cwd=ROOT, env=env,
                                       stdout=log, stderr=subprocess.STDOUT), log)
        job_status(f"read:{net}", {"status": "RUNNING", "gpu": gpu, "pid": procs[net][0].pid, "started_utc": now()})
    ok = True
    for net, (proc, log) in procs.items():
        code = proc.wait()
        log.close()
        job_status(f"read:{net}", {"status": f"EXIT_{code}", "ended_utc": now()})
        ok = ok and code == 0
    return ok


def run_cpu(command):
    with open(OUT / "logs" / f"{command}.log", "w") as log:
        code = subprocess.call([sys.executable, __file__, command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    job_status(command, {"status": f"EXIT_{code}", "ended_utc": now()})
    return code == 0


def chain():
    verify()
    require(not (OUT / "status.json").exists(), "the chain runs once")
    status({"status": "RUNNING"})
    for stage, step in (("reproduction", lambda: run_reads(REPRO)), ("accept", lambda: run_cpu("accept")),
                        ("new_reads", lambda: run_reads(NEW)), ("decide", lambda: run_cpu("decide"))):
        status({"stage": stage})
        if not step():
            status({"status": f"FAILED_AT_{stage.upper()}", "ended_utc": now()})
            return
        time.sleep(1)
    status({"status": "COMPLETE", "ended_utc": now()})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("command", choices=("freeze", "read", "accept", "decide", "chain"))
    p.add_argument("--net")
    a = p.parse_args(argv)
    if a.command == "read":
        read_net(a.net)
    else:
        {"freeze": freeze, "accept": accept, "decide": decide, "chain": chain}[a.command]()


if __name__ == "__main__":
    main()
