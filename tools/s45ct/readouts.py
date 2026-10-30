#!/usr/bin/env python3
"""S45-CT readouts: S45-TR's frozen readout tasks (tools/s45tr/readouts.py run_task, unchanged) for any S45-CT
network, one task per process on one GPU among 2-7; outputs under outputs/s45ct/20261030/readouts/<net>/<task>.

Networks: `s45_5000`, `s45_5500`, `t1d_5000`, `t1d_5500` (step-0 discrimination check), `sigma16`, `shuffle` (step
6000) and `sigma16_5000` ... `shuffle_5500` (descriptive late-checkpoint stability). Tasks: tl_e1 (own loop on the
nine training local recordings), tl_f1 (the same packet 20 times from the GT / mean / own-loop state), tl_e3, zl_e3.
Only the network loader is replaced: checkpoint by registered SHA, config, and the arm's route corruption.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.s45ct.common import (FORMAL_ARMS, OUT, PHYSICAL_GPUS, S45_CONFIG, S45_RUN, S45TR, T1D_CONFIG, T1D_RUN,  # noqa: E402
                                config_path, install_hook, output_dir, read, require, run_name, sha, verify_protocol)

NETS = ("s45_5000", "s45_5500", "t1d_5000", "t1d_5500") + FORMAL_ARMS + tuple(f"{a}_{s}" for a in FORMAL_ARMS for s in (5000, 5500))
TASKS = ("tl_e1", "tl_f1", "tl_e3", "zl_e3")


def parse(net):
    base, _, step = net.partition("_")
    return base, int(step) if step else 6000


def checkpoint_of(net):
    base, step = parse(net)
    if base == "s45":
        receipt = read(ROOT / "outputs/c1sf/20261029/receipts/formal_so3.json")
        return S45_RUN / f"c1sf_so3_s3407-step={step}.ckpt", S45_CONFIG, receipt["checkpoint_sha256_by_step"][str(step)]
    if base == "t1d":
        receipt = read(S45TR / "receipts/formal_t1d.json")
        return T1D_RUN / f"s45tr_t1d_s3407-step={step}.ckpt", T1D_CONFIG, receipt["checkpoint_sha256_by_step"][str(step)]
    receipt = read(OUT / "receipts" / f"formal_{base}.json")
    require(receipt["status"] == "PASS", "arm has no verified formal training")
    return (output_dir(base, "formal") / f"{run_name(base)}-step={step}.ckpt", config_path(base, "formal"),
            receipt["checkpoint_sha256_by_step"][str(step)])


def load(net, device):
    from config import load_config
    from model import MNISTModel
    checkpoint, config, digest = checkpoint_of(net)
    require(sha(checkpoint) == digest, "checkpoint changed since its receipt: " + str(checkpoint))
    cfg = load_config(str(config))
    install_hook(cfg)
    model = MNISTModel.load_from_checkpoint(str(checkpoint), cfg=cfg, map_location=device, strict=True)
    return model.to(device).eval(), cfg, checkpoint


def main(argv=None):
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--net", required=True, choices=NETS)
    p.add_argument("--task", required=True, choices=TASKS)
    a = p.parse_args(argv)
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in tuple(map(str, PHYSICAL_GPUS)), "one GPU among 2-7")
    from tools.xcl import evaluate as XE
    XE.configure_environment()
    XE.set_policy()
    verify_protocol()
    from tools.s45tr import readouts as R
    R.load = load
    R.run_task(a.net, a.task, out_root=OUT / "readouts")


if __name__ == "__main__":
    main()
