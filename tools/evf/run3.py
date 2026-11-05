#!/usr/bin/env python3
"""EVF acceptance of the second registry from round 2 on (tools/evf/run2.py's procedure, frozen; readings stay with
run2): an arm with TRACK overrides (a training prior-state distribution arm) draws different prior noise, so its packets
(prior state, and the polarity augmentation drawn after the noise) differ from B2's by design; its acceptance records
the stream hashes without requiring them equal to B2's. Every other check is run2's. (Original docstring follows.)

EVF acceptance and readings of the second registry (tools/evf/arms2.py; tools/evf/run.py's procedure, frozen with
round 1). EVF acceptance and readings (docs/EXPERIMENT_RECORDS.md#evf-prereg-20261104) through PFO's frozen readout engine
(tools/pfo/engine.py, unchanged), the EVF network kind installed (tools/evf/nets.py) and the output root redirected to
outputs/evf/20261104 (readings/<tag>/<task>/).

    python -m tools.evf.run3 accept --arm A --seed S --phase screen|formal
    python -m tools.evf.run3 eval --arm A --seed S --phase screen|formal --task T --step N
        tag evf_<arm>_s<seed>[_screen]@<step>
    python -m tools.evf.run3 evalb2 --seed S --task T --step N
        B2 (PFO2's fgs run, read-only) at any saved step; tag b2_s<seed>@<step>
Acceptance: the phase's checkpoint grid, seed, steps, cosine schedule over 6000 steps, two devices, the deterministic
Trainer record, the config SHA, and the data stream: the first two batches of each rank hash equal to B2's at the same
seed (semkine/train.py's pair_stream_rank*.json). Evaluation configs keep the protocol's segment-start noise and drop the
MODEL.XAG_* keys (written once under configs/evf_20261104/eval/).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.evf import arms2 as A  # noqa: E402
from tools.evf.common import CONFIGS, OUT, b2_checkpoint, b2_run, now, read, require, save, sha  # noqa: E402

GRID = {"screen": (500, 1000, 1500, 2000), "formal": (500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500,
                                                       6000)}
PROTOCOL_INIT_NOISE = {"PREV_NOISE_T": 0.005, "PREV_NOISE_R": 0.05, "PREV_NOISE_POSE": 0.05}


def write_once(path, obj):
    from tools.pfo2.specs import write_once_text
    if str(path).endswith(".json"):
        return write_once_text(path, json.dumps(obj, indent=2, allow_nan=False) + "\n")
    import yaml
    return write_once_text(path, yaml.safe_dump(obj, sort_keys=False))


def eval_config(arm, seed, phase):
    cfg = A.build_config(arm, seed, phase)
    cfg["TRACK"].update(PROTOCOL_INIT_NOISE)
    cfg["MODEL"].pop("XAG_ASSOC", None)
    cfg["MODEL"].pop("XAG_GNN", None)
    return write_once(CONFIGS / "eval" / f"{A.run_name(arm, seed, phase)}.yaml", cfg)


def checkpoint(arm, seed, phase, step):
    return A.run_dir(arm, seed, phase) / f"{A.run_name(arm, seed, phase)}-step={step}.ckpt"


def stream_rows(run_dir):
    return {r: read(run_dir / f"pair_stream_rank{r}.json") for r in (0, 1)}


def accept(arm, seed, phase):
    require(phase in GRID, "acceptance of screen / formal runs")
    name = A.run_name(arm, seed, phase)
    out = OUT / "receipts" / f"{name}.json"
    require(not out.exists(), "accepted once")
    d = A.run_dir(arm, seed, phase)
    launch, meta = read(d / "evf_launch.json"), read(d / "training_metadata.json")
    det = read(d / "evf_det.json")
    ok = {"steps": all(checkpoint(arm, seed, phase, st).exists() for st in GRID[phase]),
          "seed": meta["seed"] == seed and launch["evf"]["SEED"] == seed,
          "max_steps": meta["max_steps"] == A.stop_step(phase), "lr_schedule": meta["lr_schedule"] == "cosine",
          "devices": meta["devices"] == 2 and meta["batch_size_per_gpu"] == 512,
          "config": launch["config_sha256"] == sha(launch["config"]),
          "deterministic": bool(det["deterministic_algorithms"]) and det["trainer_deterministic"]}
    mine, b2 = stream_rows(d), stream_rows(b2_run(seed))
    same = all([r["sha256"] for r in mine[k][:2]] == [r["sha256"] for r in b2[k][:2]] for k in (0, 1))
    if A.spec(arm).get("track"):
        stream_note = {"data_stream_equals_b2": same, "required": False, "reason": "TRACK override"}
    else:
        ok["data_stream_equals_b2"] = same
        stream_note = None
    receipt = {"run": name, "arm": arm, "seed": seed, "phase": phase, "checks": ok, "stream_note": stream_note,
               "status": "PASS" if all(ok.values()) else "FAIL", "accepted_at_utc": now(),
               "checkpoint_sha256_by_step": {str(st): sha(checkpoint(arm, seed, phase, st)) for st in GRID[phase]
                                             if checkpoint(arm, seed, phase, st).exists()}}
    save(out, receipt, exclusive=True)
    print(receipt["status"], ok, flush=True)
    require(receipt["status"] == "PASS", "acceptance failed")


def arm_spec(arm, seed, phase, step):
    receipt = read(OUT / "receipts" / f"{A.run_name(arm, seed, phase)}.json")
    require(receipt["status"] == "PASS", "training not accepted")
    ck = checkpoint(arm, seed, phase, step)
    require(sha(ck) == receipt["checkpoint_sha256_by_step"][str(step)], "checkpoint changed since acceptance")
    tag = f"{A.run_name(arm, seed, phase)}@{step}"
    sp = {"kind": "evf2", "checkpoint": str(ck), "checkpoint_sha256": receipt["checkpoint_sha256_by_step"][str(step)],
          "config": str(eval_config(arm, seed, phase))}
    return tag, write_once(OUT / "specs" / f"{tag}.json", sp)


def b2_spec(seed, step):
    ck = b2_checkpoint(seed, step)
    require(ck.exists(), "missing B2 checkpoint " + str(ck))
    cfg = ROOT / "configs/pfo2_20261103/eval" / f"pfo2_fgs_s{seed}.yaml"
    require(cfg.exists(), "missing B2 evaluation config " + str(cfg))
    tag = f"b2_s{seed}@{step}"
    sp = {"kind": "routed", "checkpoint": str(ck), "checkpoint_sha256": sha(ck), "config": str(cfg), "trunk": "gshare",
          "module": "trunk4"}
    return tag, write_once(OUT / "specs" / f"{tag}.json", sp)


def engine_run(tag, task, sp, b2=False):
    from tools.pfo import engine as E
    from tools.evf import nets2 as EN
    if b2:
        from tools.pfo2.run import prepare
        prepare(sp)
    EN.install()
    E.OUT = OUT                                                     # readings go under outputs/evf/20261104
    return E.run(tag, task, None, None, (), str(sp))


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="verb", required=True)
    for verb in ("accept", "eval", "evalb2"):
        s = sub.add_parser(verb)
        s.add_argument("--seed", required=True, type=int)
        if verb != "evalb2":
            s.add_argument("--arm", required=True)
            s.add_argument("--phase", required=True, choices=tuple(GRID))
        if verb != "accept":
            s.add_argument("--task", required=True)
            s.add_argument("--step", type=int, required=True)
    a = p.parse_args(argv)
    if a.verb == "accept":
        return accept(a.arm, a.seed, a.phase)
    if a.verb == "eval":
        require(a.step in GRID[a.phase], "a step of the phase's checkpoint grid")
        tag, sp = arm_spec(a.arm, a.seed, a.phase, a.step)
        return engine_run(tag, a.task, sp)
    tag, sp = b2_spec(a.seed, a.step)
    return engine_run(tag, a.task, sp, b2=True)


if __name__ == "__main__":
    main()
