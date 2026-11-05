#!/usr/bin/env python3
"""ACU stage-2 readings and acceptance (docs/EXPERIMENT_RECORDS.md#acu-s2-prereg-20261104) through PFO's frozen engine,
output root outputs/acu/20261104; one GPU among 4-7.

    python -m tools.acu.run_s2 accept --cell C --seed S [--trunk T] --phase P
        registered config, seed, steps, deterministic kernels, the causal hooks and the PR checkpoint accepted by s1c;
        pairing: the data stream equals the partner cell's (c10 with c00, c11 with c01: the same pipeline, only the
        observation differs) when the partner has started, and the b1 cells' samples equal the orig cells' except the prior
        on the B1 branches (CPU, first PAIR_N indices)
    python -m tools.acu.run_s2 eval --cell C --seed S [--trunk T] --task T
        reading with PR's causal routes; the engine's step is re-bound in this process to build every packet with its
        stream history (semkine/acu_causal.py `eval_batch`); tag acu_s2_<arm>_s<seed>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.acu import arms_s2 as A  # noqa: E402
from tools.acu.common import CONFIGS, OUT, now, read, require, save, sha  # noqa: E402

GRID = {"preflight": (40,), "formal": tuple(range(500, 6001, 500))}
TASKS = ("e1", "e3", "pert", "f1", "tl_e1")
PARTNER = {"c10": "c00", "c11": "c01"}
PAIR_N = 16


def checkpoint(cell, seed, step, phase="formal", trunk="gshare"):
    return A.run_dir(cell, seed, phase, trunk) / f"{A.run_name(cell, seed, phase, trunk)}-step={step}.ckpt"


def write_once(path, text):
    from tools.pfo2.specs import write_once_text
    return write_once_text(path, text)


def eval_config(cell, seed, trunk):
    import yaml
    from tools.pfo2.run import _eval_cfg
    name = A.run_name(cell, seed, trunk=trunk)
    cfg = _eval_cfg(A.build_config(cell, seed, trunk=trunk), name)
    return write_once(CONFIGS / "eval" / f"{name}.yaml", yaml.safe_dump(cfg, sort_keys=False))


def spec(cell, seed, trunk, step=6000):
    receipt = read(OUT / "receipts" / f"{A.run_name(cell, seed, trunk=trunk)}.json")
    require(receipt["status"] == "PASS", "training not accepted")
    ck = checkpoint(cell, seed, step, trunk=trunk)
    require(sha(ck) == receipt["checkpoint_sha256_by_step"][str(step)], "checkpoint changed since acceptance")
    name = A.run_name(cell, seed, trunk=trunk) + ("" if step == 6000 else f"@{step}")
    sp = {"kind": "routed", "checkpoint": str(ck), "checkpoint_sha256": receipt["checkpoint_sha256_by_step"][str(step)],
          "config": str(eval_config(cell, seed, trunk)), "trunk": trunk, "module": "acu_s2"}
    return name, write_once(OUT / "specs" / f"{name}.json", json.dumps(sp, indent=2) + "\n")


def install(cell, trunk):
    """Trunk class now; the causal hooks right after the engine's model load; the engine's step re-bound."""
    import torch
    from semkine import acu_causal as AC
    from semkine import pfo_router as PRR
    from tools.acu.train_s2 import install_trunk
    from tools.b2ub.run import deployed_router_for_mixes
    from tools.pfo import nets as N
    from model import MNISTModel
    cfg = A.build_config(cell, A.SEEDS[0], trunk=trunk)["ACU"]
    PRR.register()
    deployed_router_for_mixes()
    install_trunk(cfg)
    native = N.load

    def load(*args, **kwargs):
        net = native(*args, **kwargs)
        net.identity["acu_causal"] = AC.install_model(MNISTModel, cfg["PR_CHECKPOINT"], cfg["IFACE"],
                                                      float(cfg["ONHAND_TAU"]), b1=cfg["UPDATE"] == "b1")
        return net
    N.load = load

    def step(self, ctx, end, prev):
        prev_t = torch.from_numpy(np.ascontiguousarray(prev, np.float32)).view(1, -1).to(self.device)
        batch = AC.eval_batch(ctx, int(end), AC.run_start(ctx.aux, int(end)), prev_t, self.device)
        value = self.model.forward_packet(batch).detach().cpu().numpy()[0].copy()
        if value.shape != (51,) or not np.isfinite(value).all():
            raise FloatingPointError(f"nonfinite endpoint retained as failure: {ctx.seq}:{end}")
        return value
    N.Net.step = step


def evaluate(cell, seed, trunk, task, step=6000):
    from tools.pfo import engine as E
    require(task in TASKS, "task")
    name, sp = spec(cell, seed, trunk, step)
    install(cell, trunk)
    E.OUT = OUT
    return E.run(name, task, None, None, (), str(sp))


def stream_rows(run_dir):
    return {r: [x["sha256"] for x in read(Path(run_dir) / f"pair_stream_rank{r}.json")[:2]] for r in (0, 1)}


def b1_pairing(seed, phase):
    """The b1 cells' training samples against the orig cells' (same causal pipeline), index by index (CPU)."""
    from semkine import acu_b1 as B1
    from semkine import acu_causal as AC
    from semkine.dataset import build_dataset
    cfg = A.build_config("c01", seed, phase)
    comps = np.load(cfg["MANO"]["NPZ"])["hands_components"].astype(np.float32)
    ds = build_dataset(cfg, "train", comps, train=True)
    rows = []
    for idx in range(PAIR_N):
        AC._STATE["b1"] = False
        base = AC._getitem(ds, idx)
        AC._STATE["b1"] = True
        mine = AC._getitem(ds, idx)
        start, end = int(base.t_start_us) // 1000, int(base.t_end_us) // 1000 - 1
        prior_ms, branch = B1.draw_prior_ms(ds.seed, idx, start, end)
        same = {f: bool(np.array_equal(getattr(base, f), getattr(mine, f))) for f in ("events", "target", "camera_K")}
        same["hist"] = bool(np.array_equal(base.meta["acu_hist"], mine.meta["acu_hist"]))
        prev_same = bool(np.array_equal(base.prev_state, mine.prev_state))
        rows.append({"idx": idx, "branch": branch, **same, "prev_equal": prev_same,
                     "ok": all(same.values()) and (prev_same == (prior_ms == start))})
    AC._STATE["b1"] = False
    return rows


def accept(cell, seed, trunk, phase):
    require(cell in A.CELLS and phase in GRID, "cell / phase")
    name = A.run_name(cell, seed, phase, trunk)
    out = OUT / "receipts" / f"{name}.json"
    require(not out.exists(), "accepted once")
    d = A.run_dir(cell, seed, phase, trunk)
    launch, meta, det = read(d / "acu_launch.json"), read(d / "training_metadata.json"), read(d / "acu_det.json")
    cfg = A.build_config(cell, seed, phase, trunk)
    a = cfg["ACU"]
    ok = {"steps": all(checkpoint(cell, seed, st, phase, trunk).exists() for st in GRID[phase]),
          "seed": meta["seed"] == seed and launch["pfo"]["SEED"] == seed,
          "max_steps": meta["max_steps"] == cfg["TRAIN"]["MAX_STEPS"], "lr_schedule": meta["lr_schedule"] == "cosine",
          "devices": meta["devices"] == 2 and meta["batch_size_per_gpu"] == 512,
          "config": launch["config_sha256"] == sha(launch["config"]) and Path(launch["config"]) == A.config_path(
              cell, seed, phase, trunk).resolve(),
          "deterministic": bool(det["deterministic_algorithms"]) and det["trainer_deterministic"],
          "hooks": launch["hooks"]["assoc"] == a["IFACE"] and launch["hooks"]["b1"] == (a["UPDATE"] == "b1"),
          "pr_checkpoint": launch["pr_checkpoint_sha256"] == sha(a["PR_CHECKPOINT"])}
    extra = {}
    partner = PARTNER.get(cell)
    if partner and A.run_dir(partner, seed, phase, trunk).exists():
        ok["stream_equals_partner"] = stream_rows(d) == stream_rows(A.run_dir(partner, seed, phase, trunk))
    if a["UPDATE"] == "b1":
        extra["b1_pairing"] = b1_pairing(seed, phase)
        ok["b1_pairing"] = all(r["ok"] for r in extra["b1_pairing"])
    receipt = {"run": name, "cell": cell, "seed": seed, "trunk": trunk, "phase": phase, "checks": ok, **extra,
               "status": "PASS" if all(ok.values()) else "FAIL", "accepted_at_utc": now(),
               "checkpoint_sha256_by_step": {str(st): sha(checkpoint(cell, seed, st, phase, trunk)) for st in GRID[phase]
                                             if checkpoint(cell, seed, st, phase, trunk).exists()}}
    save(out, receipt, exclusive=True)
    print(receipt["status"], ok, flush=True)
    require(receipt["status"] == "PASS", "acceptance failed")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    for verb in ("accept", "eval"):
        s = sub.add_parser(verb)
        s.add_argument("--cell", required=True, choices=tuple(A.CELLS))
        s.add_argument("--seed", type=int, default=3407)
        s.add_argument("--trunk", default="gshare", choices=A.TRUNKS)
        if verb == "accept":
            s.add_argument("--phase", required=True)
        else:
            s.add_argument("--task", required=True)
            s.add_argument("--step", type=int, default=6000)
    x = p.parse_args(argv)
    if x.verb == "accept":
        accept(x.cell, x.seed, x.trunk, x.phase)
    else:
        evaluate(x.cell, x.seed, x.trunk, x.task, x.step)


if __name__ == "__main__":
    main()
