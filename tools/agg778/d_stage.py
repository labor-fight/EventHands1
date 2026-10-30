#!/usr/bin/env python3
"""AGG778 Stage D supervisor (docs/EXPERIMENT_RECORDS.md#agg778-prereg-20261024): select -> train -> evaluate.

    select    the pre-registered arm rule on Stage-B selection-subject contrasts (outputs/agg778/b_summary.json)
    train     50-step preflight of every selected arm (<= 2 at once, physical GPUs 4-5 / 6-7), pairing check against the
              control, then fresh 6000-step formal training, receipts with checkpoint SHAs
    evaluate  per arm: the latency row alone, then E1 x2 (fresh loads), E2, E3, GT-pose / GT-translation clamps and E1 at
              steps 5000 / 5500, one process per GPU 4-7

Fresh-only and fail-closed: a failed child stops the stage and its evidence is preserved.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]
from tools.agg778.d_common import ARMS, GPU_GROUPS, OUT_D, arm_name, paths, read, require, save, sha, verify_pairing  # noqa: E402
from tools.agg778.common import PROTOCOL  # noqa: E402

PY = sys.executable
MAX_ARMS = 3


def now():
    return datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------------------------------------------ select
def select():
    """Rank the four aggregation factors by their selection-subject effect; top-1 on a3 and a0, top-2 on a3 only if it
    is material; none material -> the largest-effect factor on a3 and a0 as a confirmation."""
    b = read(ROOT / "outputs/agg778/b_summary.json")
    fac = b["selection_factors"]                                                 # {factor: {effect, material}}
    ranked = sorted(fac, key=lambda f: fac[f]["effect"], reverse=True)
    material = [f for f in ranked if fac[f]["material"]]
    top = (material or ranked)[:2]
    arms = [arm_name(top[0], "a3"), arm_name(top[0], "a0")]
    if len(material) >= 2:
        arms.append(arm_name(top[1], "a3"))
    out = {"factors": fac, "ranked": ranked, "material": material, "arms": arms[:MAX_ARMS],
           "confirmation_only": not material, "b_summary_sha256": sha(ROOT / "outputs/agg778/b_summary.json"),
           "selected_at_utc": now(), "zgz_not_read_for_selection": True}
    save(OUT_D / "selection.json", out, exclusive=True)
    return out


# ------------------------------------------------------------------------------------------------ train
def launch(arm, phase, gpus, log):
    p = paths(arm)
    cmd = [PY, "tools/agg778/train.py", "--arm", arm, "--phase", phase, "--protocol", str(PROTOCOL),
           "--config", str(p["preflight_config"] if phase == "preflight" else p["config"])]
    if phase == "formal":
        cmd += ["--acceptance", str(OUT_D / "acceptance" / f"preflight_{arm}.json")]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)), PYTHONHASHSEED="3407",
               PL_INTER_BATCH_PARALLELISM="0", CUDA_DEVICE_ORDER="PCI_BUS_ID", MASTER_PORT=str(29400 + min(gpus)))
    log.parent.mkdir(parents=True, exist_ok=True)
    h = log.open("x")
    return subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=h, stderr=subprocess.STDOUT, start_new_session=True), cmd


def verify_run(arm, phase):
    d = paths(arm)["preflight_output" if phase == "preflight" else "output"]
    steps = 50 if phase == "preflight" else 6000
    comp = [read(d / f"completion_rank{r}.json") for r in (0, 1)]
    require(all(c["status"] == "COMPLETE" and c["step"] == steps for c in comp), f"{arm} {phase} incomplete")
    vals = sorted(d.glob("validation_rank0_step*.json"))
    require(vals and all(read(v)["status"] == "PASS" for v in vals), f"{arm} {phase} validation missing / failed")
    pair = verify_pairing(arm, d)
    require(pair["status"] == "PASS", f"{arm} {phase} pairing with its control failed")
    rec = {"status": "PASS", "arm": arm, "phase": phase, "steps": steps, "pairing": pair,
           "final_checkpoint_sha256": sha(d / "repro_final.ckpt"), "completed_at_utc": now()}
    if phase == "formal":
        rec["checkpoint_sha256_by_step"] = {}
        for step in (5000, 5500):
            c = [q for q in d.glob(f"*-step={step}.ckpt")]
            require(len(c) == 1, f"{arm}: step{step} checkpoint missing")
            rec["checkpoint_sha256_by_step"][str(step)] = sha(c[0])
    return rec


def run_phase(arms, phase):
    pending, active, done = list(arms), {}, []
    free = list(GPU_GROUPS)
    while pending or active:
        while pending and free:
            arm = pending.pop(0)
            gpus = free.pop(0)
            proc, cmd = launch(arm, phase, gpus, OUT_D / "logs" / f"{phase}_{arm}.log")
            active[arm] = (proc, gpus)
            save(OUT_D / "status.json", {"phase": phase, "active": {a: list(g) for a, (_, g) in active.items()},
                                          "pending": pending, "done": done, "updated_utc": now()})
        for arm, (proc, gpus) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            require(code == 0, f"{phase} {arm} failed with exit {code}; evidence preserved")
            rec = verify_run(arm, phase)
            name = ("preflight_" if phase == "preflight" else "formal_") + arm + ".json"
            save(OUT_D / ("acceptance" if phase == "preflight" else "receipts") / name,
                 {**rec, "protocol_sha256": sha(PROTOCOL), "wrapper_sha256": sha(ROOT / "tools/agg778/train.py")},
                 exclusive=True)
            done.append(arm)
            free.append(gpus)
            del active[arm]
        time.sleep(20)
    return done


def train():
    sel = read(OUT_D / "selection.json")
    arms = sel["arms"]
    run_phase(arms, "preflight")
    run_phase(arms, "formal")
    save(OUT_D / "status.json", {"phase": "trained", "arms": arms, "updated_utc": now()})


# ------------------------------------------------------------------------------------------------ evaluate
def eval_jobs(arm):
    jobs = [("e1", 6000, 1, "none"), ("e1", 6000, 2, "none"), ("e2", 6000, 1, "none"), ("e3", 6000, 1, "none"),
            ("gt_pose", 6000, 1, "none"), ("gt_transl", 6000, 1, "none"), ("e1", 5000, 1, "none"),
            ("e1", 5500, 1, "none")]
    return [(arm, *j) for j in jobs]


def run_eval(job, gpu):
    arm, mode, step, rep, inter = job
    cmd = [PY, "tools/agg778/d_eval.py", "--arm", arm, "--mode", mode, "--step", str(step), "--repeat", str(rep),
           "--intervention", inter]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID")
    log = OUT_D / "logs" / f"eval_{arm}_{mode}_{step}_{rep}_{inter}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log.open("x"), stderr=subprocess.STDOUT,
                            start_new_session=True)


def evaluate():
    sel = read(OUT_D / "selection.json")
    for arm in sel["arms"]:                                                       # latency rows: alone, serially
        p = run_eval((arm, "row", 6000, 1, "none"), 4)
        require(p.wait() == 0, f"row {arm} failed")
    jobs = [j for arm in sel["arms"] for j in eval_jobs(arm)]
    free, active = [4, 5, 6, 7], {}
    while jobs or active:
        while jobs and free:
            g = free.pop(0)
            j = jobs.pop(0)
            active[g] = (j, run_eval(j, g))
        for g, (j, p) in list(active.items()):
            code = p.poll()
            if code is None:
                continue
            require(code == 0, f"evaluation {j} failed")
            free.append(g)
            del active[g]
        time.sleep(10)
    save(OUT_D / "status.json", {"phase": "evaluated", "arms": sel["arms"], "updated_utc": now()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("select", "train", "evaluate"))
    a = ap.parse_args()
    p = read(PROTOCOL)
    bad = [k for k, v in p["files"].items() if sha(ROOT / k) != v]
    require(not bad, f"AGG778 frozen sources changed: {bad}")
    {"select": select, "train": train, "evaluate": evaluate}[a.cmd]()


if __name__ == "__main__":
    main()
