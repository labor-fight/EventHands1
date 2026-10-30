#!/usr/bin/env python3
"""MRF Stage D (docs/EXPERIMENT_RECORDS.md#mrf-prereg-20261024): in-system paired retraining of one receptive-field
variant (`MODEL.XYZ_RF`), single seed 3407 (user decision 2026-10-24).

An arm = one variant on one recipe; its config is the control's with only MODEL.XYZ_RF, the run name / output paths
(and on a3 the XCL audit cell) changed:

    a0  the C1 recipe, paired with xyz_c1_stable_s3407            a3  the XCL A3F0 recipe (oracle), paired with A3F0

The AGG778 R1 wrappers (`tools/agg778/train.py`, `d_eval.py`: the C1 strict numerical / input contract, the XCL
readouts) are reused unchanged through `tools/mrf/d_train.py` / `d_eval.py`, which swap in this registry. model/model.py
must carry the frozen `MODEL.XYZ_RF` wiring, applied only after AGG778 R2 is complete (`apply-wiring`).

    python tools/mrf/d_stage.py write-configs | apply-wiring | select | train | evaluate
"""
from __future__ import annotations

import json
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

OUT_D = ROOT / "outputs" / "mrf" / "d"
CONFIG_DIR = ROOT / "configs" / "mrf"
PROTOCOL = ROOT / "outputs" / "mrf" / "protocol.json"
B_SUMMARY = ROOT / "outputs" / "mrf" / "b" / "summary.json"
VARIANT = {"self3": "self3", "noprop": "noprop", "ring4np": "ring4np", "globalloc": "global_loc"}
ARMS = {f"mrf_{v}_{c}_s3407": {"variant": VARIANT[v], "cond": c} for v in VARIANT for c in ("a0", "a3")}
MAX_ARMS = 3
PY = sys.executable


def now():
    return datetime.now(timezone.utc).isoformat()


def paths(arm: str) -> dict:
    return {"config": CONFIG_DIR / f"{arm}.yaml", "preflight_config": CONFIG_DIR / "preflight" / f"{arm}_preflight.yaml",
            "output": OUT_D / "runs" / arm, "preflight_output": OUT_D / "preflight" / f"{arm}_preflight"}


def expected_config(arm: str, phase: str) -> dict:
    from config import load_config
    a = ARMS[arm]
    cfg = load_config(str(ROOT / CONTROLS[a["cond"]]["config"]))
    cfg.pop("_config_path")
    cfg["MODEL"]["XYZ_RF"] = a["variant"]
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
    saved = D1.ARMS, D1.paths
    D1.ARMS, D1.paths = ARMS, paths
    try:
        return D1.verify_pairing(arm, directory)
    finally:
        D1.ARMS, D1.paths = saved


def verify_files():
    """The frozen `files` (model/model.py is not among them) plus the wired model/model.py."""
    p = read(PROTOCOL)
    bad = [k for k, v in p["files"].items() if sha(ROOT / k) != v]
    require(not bad, f"MRF frozen sources changed: {bad}")
    require(sha(ROOT / "model" / "model.py") == p["model_wiring"]["post"],
            "model/model.py is not the frozen MODEL.XYZ_RF wiring (run apply-wiring once AGG778 R2 is complete)")
    return p


def patch_train(T):
    def verify(protocol):
        require(Path(protocol).resolve() == PROTOCOL.resolve(), "MRF arms train under the MRF protocol only")
        return verify_files()
    T.ARMS, T.expected_config, T.paths, T.verify_protocol = ARMS, expected_config, paths, verify


def patch_eval(E):
    verify_files()
    E.ARMS, E.OUT_D, E.paths, E.PROTOCOL = ARMS, OUT_D, paths, PROTOCOL


# ------------------------------------------------------------------------------------------------ wiring
def r2_complete() -> dict:
    """AGG778 R2 is complete: its probe summary and Stage-D selection exist, every selected arm has all nine readouts,
    and no R2 process is alive."""
    r2 = ROOT / "outputs" / "agg778" / "r2"
    out = {"summary": (r2 / "summary.json").is_file(), "selection": (r2 / "d" / "selection.json").is_file()}
    arms = read(r2 / "d" / "selection.json")["arms"] if out["selection"] else None
    want = ("row_step6000_repeat1_none", "e1_step6000_repeat1_none", "e1_step6000_repeat2_none",
            "e2_step6000_repeat1_none", "e3_step6000_repeat1_none", "gt_pose_step6000_repeat1_none",
            "gt_transl_step6000_repeat1_none", "e1_step5000_repeat1_none", "e1_step5500_repeat1_none")
    out["readouts"] = None if arms is None else all(
        (r2 / "d" / "evaluations" / a / w / "report.json").is_file() for a in arms for w in want)
    ps = subprocess.run(["pgrep", "-f", "tools/agg778/r2_"], capture_output=True, text=True).stdout.split()
    out["no_r2_process"] = not ps
    out["complete"] = bool(out["summary"] and out["selection"] and out["readouts"] and out["no_r2_process"])
    return out


def apply_wiring():
    from tools.mrf import wiring as W
    st = r2_complete()
    require(st["complete"], f"AGG778 R2 is not complete; model/model.py stays frozen: {st}")
    p = read(PROTOCOL)
    result = W.apply(p["model_wiring"])
    save(OUT_D / "wiring_applied.json", {"result": result, "r2_state": st, "model_py_sha256": sha(W.MODEL_PY),
                                         "expected_post": p["model_wiring"]["post"], "applied_at_utc": now()},
         exclusive=True)
    return result


# ------------------------------------------------------------------------------------------------ supervisor
def select():
    s = read(B_SUMMARY)
    t = s["d_triggers"]
    save(OUT_D / "selection.json", {"arms": t["arms"], "triggers": t, "b_summary_sha256": sha(B_SUMMARY),
                                    "zgz_not_read_for_selection": True, "selected_at_utc": now()}, exclusive=True)
    return t["arms"]


def launch(arm, phase, gpus):
    p = paths(arm)
    cmd = [PY, "tools/mrf/d_train.py", "--arm", arm, "--phase", phase, "--protocol", str(PROTOCOL),
           "--config", str(p["preflight_config"] if phase == "preflight" else p["config"])]
    if phase == "formal":
        cmd += ["--acceptance", str(OUT_D / "acceptance" / f"preflight_{arm}.json")]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)), PYTHONHASHSEED="3407",
               PL_INTER_BATCH_PARALLELISM="0", CUDA_DEVICE_ORDER="PCI_BUS_ID", MASTER_PORT=str(29700 + min(gpus)))
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
            save(OUT_D / "status.json", {"phase": phase, "active": {a: list(g) for a, (_, g) in active.items()},
                                          "pending": pending, "updated_utc": now()})
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
    cmd = [PY, "tools/mrf/d_eval.py", "--arm", arm, "--mode", mode, "--step", str(step), "--repeat", str(rep)]
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
    save(OUT_D / "status.json", {"phase": "evaluated", "arms": arms, "updated_utc": now()})


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "write-configs":
        print("\n".join(write_configs()))
    elif cmd == "r2-state":
        print(json.dumps(r2_complete(), indent=1))
    elif cmd == "apply-wiring":
        print(apply_wiring())
    elif cmd == "select":
        print(select())
    elif cmd == "train":
        verify_files()
        arms = read(OUT_D / "selection.json")["arms"]
        run_phase(arms, "preflight")
        run_phase(arms, "formal")
    elif cmd == "evaluate":
        verify_files()
        evaluate()
    else:
        raise SystemExit("usage: d_stage.py write-configs | r2-state | apply-wiring | select | train | evaluate")


if __name__ == "__main__":
    main()
