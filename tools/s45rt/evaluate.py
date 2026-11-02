#!/usr/bin/env python3
"""S45-RT T1p evaluation: one network x one zgz readout per process, on one physical GPU among 2-7.

Delegates to the XCL evaluator's `trace` / `make_main_row` unchanged (as S45-TR / S45-CT did), with T1p's routing hook
installed first and no ground-truth XAG sources attached: E1 = own closed loop (steps 5000 / 5500 / 6000), E2 = fed
S45's own E1 states, E3 = GT previous state, `row` = latency / FLOPs / params.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.xcl import evaluate as XE  # noqa: E402
from tools.s45rt.common import OUT, PHYSICAL_GPUS, require, sha, verify_protocol  # noqa: E402
from tools.s45rt.plan2 import ARMS, install_eval_hook, resolve, run_name  # noqa: E402

S45_E1_TRACE = ROOT / "outputs/c1sf/20261029/evaluations/so3/e1_step6000_repeat1_none/states.npz"
MODES = ("row", "e1", "e2", "e3")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--net", required=True, choices=ARMS)
    p.add_argument("--mode", required=True, choices=MODES)
    p.add_argument("--step", type=int, choices=(5000, 5500, 6000), default=6000)
    a = p.parse_args(argv)
    require(a.step == 6000 or a.mode == "e1", "late checkpoints are E1 only")
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in tuple(map(str, PHYSICAL_GPUS)), "one GPU among 2-7")
    XE.configure_environment()
    XE.set_policy()
    verify_protocol()
    checkpoint, config = resolve(a.net, a.step)
    from config import load_config
    cfg = load_config(str(config))
    routing = install_eval_hook(cfg)
    output = OUT / "evaluations" / a.net / f"{a.mode}_step{a.step}"
    require(not output.exists(), "Preserve existing evaluation: " + str(output))
    output.mkdir(parents=True)
    identity = {"net": a.net, "seed": 3407, "step": a.step, "mode": a.mode, "routing": routing,
                "checkpoint_sha256": sha(checkpoint), "config_sha256": sha(config), "script_sha256": sha(__file__),
                "xcl_evaluator_sha256": sha(XE.__file__), "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}
    XE.save_new(output / "launch.json", identity)
    try:
        report = XE.trace({"name": "s45rt_" + a.net}, checkpoint, cfg, output, a.mode,
                          S45_E1_TRACE if a.mode == "e2" else None, "none")
        if a.mode == "row":
            report["main_row"] = XE.make_main_row({"name": run_name(a.net)}, checkpoint, cfg, output, report, a.step)
        require(sha(checkpoint) == identity["checkpoint_sha256"], "Inputs changed")
        XE.save_new(output / "report.json", {**identity, **report, "no_training_or_selection": True})
    except BaseException as exc:
        XE.save_new(output / "failure.json", {**identity, "status": "FAILED_PRESERVED", "error": f"{type(exc).__name__}: {exc}"})
        raise
    print("COMPLETE", output, flush=True)


if __name__ == "__main__":
    main()
