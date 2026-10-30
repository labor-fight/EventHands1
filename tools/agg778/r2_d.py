#!/usr/bin/env python3
"""AGG778 R2 Stage D (docs/EXPERIMENT_RECORDS.md#agg778-r2-prereg-20261024): registry, configs and supervisor.

Pre-fixed arms (no choice among sub-factors on zgz): `noabs` on the A3F0 and C1 recipes when H-T is supported, `fanout` on
the C1 recipe when H-B is supported. The R1 wrappers (`tools/agg778/train.py`, `d_eval.py`) are reused unchanged through
`r2_train.py` / `r2_d_eval.py`, which swap in this registry and the R2 protocol.

    python tools/agg778/r2_d.py write-configs | select | train | evaluate
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.agg778 import d_common as D1  # noqa: E402
from tools.agg778.d_common import CONTROLS, GPU_GROUPS, read, require, save, sha  # noqa: E402

OUT_D = ROOT / "outputs" / "agg778" / "r2" / "d"
CONFIG_DIR = ROOT / "configs" / "agg778_r2"
PROTOCOL = ROOT / "outputs" / "agg778" / "r2" / "protocol.json"
SUMMARY = ROOT / "outputs" / "agg778" / "r2" / "summary.json"
ARMS = {"agg778r2_noabs_a3_s3407": {"variant": "noabs", "cond": "a3"},
        "agg778r2_noabs_a0_s3407": {"variant": "noabs", "cond": "a0"},
        "agg778r2_fanout_a0_s3407": {"variant": "fanout", "cond": "a0"}}
PY = sys.executable


def now():
    return datetime.now(timezone.utc).isoformat()


def paths(arm: str) -> dict:
    return {"config": CONFIG_DIR / f"{arm}.yaml", "preflight_config": CONFIG_DIR / "preflight" / f"{arm}_preflight.yaml",
            "output": OUT_D / "runs" / arm, "preflight_output": OUT_D / "preflight" / f"{arm}_preflight"}


def expected_config(arm: str, phase: str) -> dict:
    """The arm's control config with only XYZ_AGG, run name / paths (and the XCL audit cell on a3) changed."""
    from config import load_config
    a = ARMS[arm]
    cfg = load_config(str(ROOT / CONTROLS[a["cond"]]["config"]))
    cfg.pop("_config_path")
    cfg["MODEL"]["XYZ_AGG"] = a["variant"]
    p = paths(arm)
    out = p["preflight_output"] if phase == "preflight" else p["output"]
    if phase == "preflight":
        cfg["TRAIN"].update(RUN_NAME=arm + "_preflight", OUTPUT_DIR=str(out), MAX_STEPS=50, VAL_CHECK_INTERVAL=25)
    else:
        cfg["TRAIN"].update(RUN_NAME=arm, OUTPUT_DIR=str(out))
    cfg["EVAL"]["OUTPUT_DIR"] = str(out / "eval")
    if a["cond"] == "a3":
        cfg["XCL"] = {"ENABLED": True, "CELL": arm}
    return cfg


def write_configs():
    import yaml
    written = []
    for arm in ARMS:
        for phase, key in (("formal", "config"), ("preflight", "preflight_config")):
            dst = paths(arm)[key]
            text = yaml.safe_dump(expected_config(arm, phase), sort_keys=True)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                require(dst.read_text() == text, f"registered config differs: {dst}")
            else:
                dst.write_text(text)
            written.append(str(dst.relative_to(ROOT)))
    return written


def verify_pairing(arm: str, directory=None) -> dict:
    """R1's pairing check (shared initial tensors, actual index streams, per-step packet fingerprints) on R2 arms."""
    saved_arms, saved_paths = D1.ARMS, D1.paths
    D1.ARMS, D1.paths = ARMS, paths
    try:
        return D1.verify_pairing(arm, directory)
    finally:
        D1.ARMS, D1.paths = saved_arms, saved_paths


def patch_train(T):
    T.ARMS, T.expected_config, T.paths = ARMS, expected_config, paths


def patch_eval(E):
    E.ARMS, E.OUT_D, E.paths, E.PROTOCOL = ARMS, OUT_D, paths, PROTOCOL


def verify_protocol():
    p = read(PROTOCOL)
    bad = [k for k, v in p["files"].items() if sha(ROOT / k) != v]
    require(not bad, f"AGG778 R2 frozen sources changed: {bad}")


# ------------------------------------------------------------------------------------------------ supervisor
def select():
    s = read(SUMMARY)
    t = s["triggers"]
    arms = [a for a in t["d_arms"] if t["d_allowed"].get(a)]
    refused = [a for a in t["d_arms"] if not t["d_allowed"].get(a)]
    save(OUT_D / "selection.json", {"arms": arms, "refused_invalid_condition": refused, "triggers": t,
                                    "summary_sha256": sha(SUMMARY), "selected_at_utc": now()}, exclusive=True)
    return arms


def launch(arm, phase, gpus):
    p = paths(arm)
    cmd = [PY, "tools/agg778/r2_train.py", "--arm", arm, "--phase", phase, "--protocol", str(PROTOCOL),
           "--config", str(p["preflight_config"] if phase == "preflight" else p["config"])]
    if phase == "formal":
        cmd += ["--acceptance", str(OUT_D / "acceptance" / f"preflight_{arm}.json")]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)), PYTHONHASHSEED="3407",
               PL_INTER_BATCH_PARALLELISM="0", CUDA_DEVICE_ORDER="PCI_BUS_ID", MASTER_PORT=str(29600 + min(gpus)))
    log = OUT_D / "logs" / f"{phase}_{arm}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log.open("x"), stderr=subprocess.STDOUT,
                            start_new_session=True)


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
            c = list(d.glob(f"*-step={step}.ckpt"))
            require(len(c) == 1, f"{arm}: step{step} checkpoint missing")
            rec["checkpoint_sha256_by_step"][str(step)] = sha(c[0])
    return rec


def run_phase(arms, phase):
    pending, active, free = list(arms), {}, list(GPU_GROUPS)
    while pending or active:
        while pending and free:
            arm, gpus = pending.pop(0), free.pop(0)
            active[arm] = (launch(arm, phase, gpus), gpus)
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
            free.append(gpus)
            del active[arm]
        time.sleep(20)


def run_eval(job, gpu):
    arm, mode, step, rep = job
    cmd = [PY, "tools/agg778/r2_d_eval.py", "--arm", arm, "--mode", mode, "--step", str(step), "--repeat", str(rep)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID")
    log = OUT_D / "logs" / f"eval_{arm}_{mode}_{step}_{rep}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log.open("x"), stderr=subprocess.STDOUT,
                            start_new_session=True)


def evaluate():
    arms = read(OUT_D / "selection.json")["arms"]
    for arm in arms:                                                              # latency rows alone, serially
        require(run_eval((arm, "row", 6000, 1), 4).wait() == 0, f"row {arm} failed")
    jobs = [(arm, m, st, rp) for arm in arms for m, st, rp in
            (("e1", 6000, 1), ("e1", 6000, 2), ("e2", 6000, 1), ("e3", 6000, 1), ("gt_pose", 6000, 1),
             ("gt_transl", 6000, 1), ("e1", 5000, 1), ("e1", 5500, 1))]
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


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "write-configs":
        print("\n".join(write_configs()))
        return
    verify_protocol()
    if cmd == "select":
        print(select())
    elif cmd == "train":
        arms = read(OUT_D / "selection.json")["arms"]
        run_phase(arms, "preflight")
        run_phase(arms, "formal")
    elif cmd == "evaluate":
        evaluate()
    else:
        raise SystemExit("usage: r2_d.py write-configs | select | train | evaluate")


if __name__ == "__main__":
    main()
