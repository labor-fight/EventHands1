#!/usr/bin/env python3
"""S45-PX readouts: S45-TR's frozen readout tasks and analysis (tools/s45tr/readouts.py run_task / analyze, unchanged)
for the S45-PX networks; outputs under outputs/s45px/20261030/readouts/<net>/.

  readouts.py run --net <pxc|pxm|pxc_5000|...> --task <task>     one GPU among 2-7
  readouts.py analyze --net <pxc|pxm>                             CPU; readouts/<net>/summary.json (readouts 1-4)

Step-6000 networks get every S45-TR task (the four mechanism readouts on zgz_local and the nine training local
recordings); the late checkpoints 5000 / 5500 get the in-sample own loop `tl_e1` only (stability of the primary
in-sample reading). Only the network loader and the output root are replaced.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.s45px.common import (ARMS, OUT, PHYSICAL_GPUS, config_path, install_hook, output_dir, read, require,  # noqa: E402
                                run_name, sha, verify_protocol)

LATE = tuple(f"{a}_{s}" for a in ARMS for s in (5000, 5500))
NETS = tuple(ARMS) + LATE


def parse(net):
    base, _, step = net.partition("_")
    return base, int(step) if step else 6000


def load(net, device):
    from config import load_config
    from model import MNISTModel
    base, step = parse(net)
    receipt = read(OUT / "receipts" / f"formal_{base}.json")
    require(receipt["status"] == "PASS", "arm has no verified formal training")
    checkpoint = output_dir(base, "formal") / f"{run_name(base)}-step={step}.ckpt"
    require(sha(checkpoint) == receipt["checkpoint_sha256_by_step"][str(step)], "checkpoint changed since its receipt")
    cfg = load_config(str(config_path(base, "formal")))
    install_hook(cfg)
    model = MNISTModel.load_from_checkpoint(str(checkpoint), cfg=cfg, map_location=device, strict=True)
    return model.to(device).eval(), cfg, checkpoint


def patched():
    from tools.s45tr import readouts as R
    R.load = load
    R.out_dir = lambda net, out_root=None: Path(out_root or OUT / "readouts") / net
    return R


def main(argv=None):
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("action", choices=("run", "analyze"))
    p.add_argument("--net", required=True, choices=NETS)
    p.add_argument("--task")
    a = p.parse_args(argv)
    verify_protocol()
    R = patched()
    if a.action == "run":
        require(a.task in R.TASKS and (a.net in ARMS or a.task == "tl_e1"), "unregistered net / task")
        require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in tuple(map(str, PHYSICAL_GPUS)), "one GPU among 2-7")
        from tools.xcl import evaluate as XE
        XE.configure_environment()
        XE.set_policy()
        R.run_task(a.net, a.task, out_root=OUT / "readouts")
    else:
        require(a.net in ARMS, "analysis is for step-6000 networks")
        R.analyze(a.net)


if __name__ == "__main__":
    main()
