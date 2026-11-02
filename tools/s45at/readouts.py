#!/usr/bin/env python3
"""S45-AT readouts: S45-TR's frozen readout tasks (tools/s45tr/readouts.py run_task, unchanged) for the S45-AT arms,
one task per process on one GPU among 2-7; outputs under outputs/s45at/20261031/readouts/<net>/<task>.

Networks `<arm>` (step 6000: tl_e1, tl_f1, tl_e3, zl_e3) and `<arm>_5000` / `<arm>_5500` (tl_e1). Only the network
loader is replaced (checkpoint of the given step by the formal receipt and the config); w200 reads with its 200 ms
windows clipped at the run start (install_window_patch). t1dnoylf's training local recordings include ylf_local,
which it never saw in training (its held-out reading).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.s45at.common import ARMS, OUT, PHYSICAL_GPUS, install_window_patch, require, resolve, sha, verify_protocol  # noqa: E402

TASKS_6000 = ("tl_e1", "tl_f1", "tl_e3", "zl_e3")


def parse(net):
    arm, _, step = net.rpartition("_") if net.rsplit("_", 1)[-1] in ("5000", "5500") else (net, "", "")
    return arm, int(step) if step else 6000


def main(argv=None):
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--net", required=True)
    p.add_argument("--task", required=True)
    a = p.parse_args(argv)
    arm, step = parse(a.net)
    require(arm in ARMS, "unregistered S45-AT arm " + arm)
    require(a.task in (TASKS_6000 if step == 6000 else ("tl_e1",)), f"task {a.task} is not registered for {a.net}")
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in tuple(map(str, PHYSICAL_GPUS)), "one GPU among 2-7")
    from tools.xcl import evaluate as XE
    XE.configure_environment()
    XE.set_policy()
    verify_protocol()
    from tools.s45tr import readouts as R

    def load(_net, device):
        from config import load_config
        from model import MNISTModel
        checkpoint, config = resolve(arm, step)
        cfg = load_config(str(config))
        install_window_patch(cfg)
        model = MNISTModel.load_from_checkpoint(str(checkpoint), cfg=cfg, map_location=device, strict=True)
        return model.to(device).eval(), cfg, checkpoint
    R.load = load
    R.run_task(a.net, a.task, out_root=OUT / "readouts")
    print("COMPLETE", a.net, a.task, sha(OUT / "readouts" / a.net / a.task / "states.npz"), flush=True)


if __name__ == "__main__":
    main()
