#!/usr/bin/env python3
"""ACU stage-1b readings and acceptance of b1 (docs/EXPERIMENT_RECORDS.md#acu-s1b-prereg-20261104), through PFO's frozen
engine with the output root redirected to outputs/acu/20261104.

    python -m tools.acu.run_b1 accept --phase preflight|formal
        registered config, seed, steps, deterministic kernels, B1 installs; data pairing with B2 checked sample by sample
        (`pairing`: on the first PAIR_N indices of the training set every field equals B2's sample except the prior on
        the B1 branches 1 / 2, where the prior is GT(t_p) through B2's realisation)
    python -m tools.acu.run_b1 eval --task T
        reading of the accepted checkpoint with the deployed router (r_s2d); tag acu_b1_s3407
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.acu import arms_b1 as A  # noqa: E402
from tools.acu.common import CONFIGS, OUT, SCREEN_SEED, now, read, require, run_dir_base, save, sha  # noqa: E402

GRID = {"preflight": (40,), "formal": tuple(range(500, 6001, 500))}
TASKS = ("e1", "e3", "pert", "f1", "tl_e1")
PAIR_N = 24
ARM = "b1"


def checkpoint(step, phase="formal"):
    return A.run_dir(ARM, SCREEN_SEED, phase) / f"{A.run_name(ARM, SCREEN_SEED, phase)}-step={step}.ckpt"


def write_once(path, text):
    from tools.pfo2.specs import write_once_text
    return write_once_text(path, text)


def eval_config():
    import yaml
    from tools.pfo2.run import _eval_cfg
    name = A.run_name(ARM, SCREEN_SEED)
    cfg = _eval_cfg(A.build_config(ARM, SCREEN_SEED), name)
    return write_once(CONFIGS / "eval" / f"{name}.yaml", yaml.safe_dump(cfg, sort_keys=False))


def spec(step=6000):
    receipt = read(OUT / "receipts" / f"{A.run_name(ARM, SCREEN_SEED)}.json")
    require(receipt["status"] == "PASS", "training not accepted")
    ck = checkpoint(step)
    require(sha(ck) == receipt["checkpoint_sha256_by_step"][str(step)], "checkpoint changed since acceptance")
    name = A.run_name(ARM, SCREEN_SEED) + ("" if step == 6000 else f"@{step}")
    sp = {"kind": "routed", "checkpoint": str(ck), "checkpoint_sha256": receipt["checkpoint_sha256_by_step"][str(step)],
          "config": str(eval_config()), "trunk": "gshare", "module": "acu_b1"}
    return name, write_once(OUT / "specs" / f"{name}.json", json.dumps(sp, indent=2) + "\n")


def install_b1():
    from semkine import acu_b1 as B1
    from semkine import pfo_router as PR
    from semkine import xyz_mesh as XM
    from tools.b2ub.run import deployed_router_for_mixes
    from tools.pfo2.train import install_trunk
    from model import MNISTModel
    PR.register()
    deployed_router_for_mixes()
    install_trunk("trunk4", "gshare")
    B1.install_trunk(XM)
    B1.install_model(MNISTModel)


def evaluate(task, step=6000):
    from tools.pfo import engine as E
    require(task in TASKS, "task")
    name, sp = spec(step)
    install_b1()
    E.OUT = OUT
    return E.run(name, task, None, None, (), str(sp))


def pairing(phase):
    """B1's training samples against B2's, index by index (CPU)."""
    import torch
    from semkine import acu_b1 as B1
    from semkine import dataset as D
    from semkine.dataset import build_dataset
    cfg = A.build_config(ARM, SCREEN_SEED, phase)
    comps = np.load(cfg["MANO"]["NPZ"])["hands_components"].astype(np.float32)
    ds = build_dataset(cfg, "train", comps, train=True)
    torch.manual_seed(0)
    rows = []
    for idx in range(PAIR_N):
        native = D.SemKineDataset.__getitem__
        base = native(ds, idx) if not B1._STATE["dataset"] else B1._ORIGINAL_GETITEM(ds, idx)
        mine = B1._getitem(ds, idx)
        start = int(base.t_start_us) // 1000
        end = int(base.t_end_us) // 1000 - 1
        prior_ms, branch = B1.draw_prior_ms(ds.seed, idx, start, end)
        same = {f: bool(np.array_equal(getattr(base, f), getattr(mine, f)))
                for f in ("events", "target", "betas", "camera_K", "t_start_us", "t_end_us", "sequence_id")}
        prev_same = bool(np.array_equal(base.prev_state, mine.prev_state))
        rows.append({"idx": idx, "branch": branch, "prior_ms": prior_ms, "start": start, "end": end, **same,
                     "prev_equal": prev_same, "offset_ok": abs(mine.meta["acu_prior_offset_s"] - (prior_ms - start) * 1e-3) < 1e-9,
                     "ok": all(same.values()) and (prev_same if branch == 0 else (prev_same == (prior_ms == start)))})
    return rows


def accept(phase):
    require(phase in GRID, "phase")
    name = A.run_name(ARM, SCREEN_SEED, phase)
    out = OUT / "receipts" / f"{name}.json"
    require(not out.exists(), "accepted once")
    d = A.run_dir(ARM, SCREEN_SEED, phase)
    launch, meta, det = read(d / "acu_launch.json"), read(d / "training_metadata.json"), read(d / "acu_det.json")
    cfg = A.build_config(ARM, SCREEN_SEED, phase)
    pr = pairing(phase)
    ok = {"steps": all(checkpoint(st, phase).exists() for st in GRID[phase]),
          "seed": meta["seed"] == SCREEN_SEED and launch["pfo"]["SEED"] == SCREEN_SEED,
          "max_steps": meta["max_steps"] == cfg["TRAIN"]["MAX_STEPS"],
          "lr_schedule": meta["lr_schedule"] == "cosine",
          "devices": meta["devices"] == 2 and meta["batch_size_per_gpu"] == 512,
          "config": launch["config_sha256"] == sha(launch["config"]) and Path(launch["config"]) == A.config_path(
              ARM, SCREEN_SEED, phase).resolve(),
          "deterministic": bool(det["deterministic_algorithms"]) and det["trainer_deterministic"],
          "routes": launch["routing"].get("router") == cfg["PFO"]["ROUTER_KIND"] and launch["routing"].get("assoc") == "pfa",
          "b1_trunk": str(launch["trunk"]).startswith("B1_"),
          "pairing": all(r["ok"] and r["offset_ok"] for r in pr),
          "branches_seen": len({r["branch"] for r in pr}) == 3}
    receipt = {"run": name, "arm": ARM, "seed": SCREEN_SEED, "phase": phase, "checks": ok, "pairing": pr,
               "base_run": str(run_dir_base("fgs", phase)), "status": "PASS" if all(ok.values()) else "FAIL",
               "accepted_at_utc": now(),
               "checkpoint_sha256_by_step": {str(st): sha(checkpoint(st, phase)) for st in GRID[phase]
                                             if checkpoint(st, phase).exists()}}
    save(out, receipt, exclusive=True)
    print(receipt["status"], ok, flush=True)
    require(receipt["status"] == "PASS", "acceptance failed")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    a = sub.add_parser("accept")
    a.add_argument("--phase", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--task", required=True)
    e.add_argument("--step", type=int, default=6000)
    x = p.parse_args(argv)
    if x.verb == "accept":
        accept(x.phase)
    else:
        evaluate(x.task, x.step)


if __name__ == "__main__":
    main()
