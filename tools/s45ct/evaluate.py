#!/usr/bin/env python3
"""S45-CT evaluation: one network x one zgz readout per process, on one physical GPU among 2-7.

Delegates to the XCL evaluator's `trace` / `make_main_row` unchanged (as S45-TR did for every arm), with the arm's
route corruption installed first: E1 = own closed loop (steps 5000 / 5500 / 6000), E2 = fed S45's own E1 states,
E3 = GT previous state, `row` = latency / FLOPs / params. `--net t1d` re-evaluates S45-TR T1d's E1 through this entry
(no corruption) for the step-0 reproduction check against T1d's published arrays.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.xcl import evaluate as XE  # noqa: E402
from tools.s45ct.common import (FORMAL_ARMS, OUT, PHYSICAL_GPUS, S45TR, T1D_CONFIG, T1D_RUN, config_path,  # noqa: E402
                                install_hook, output_dir, read, require, run_name, sha, verify_protocol)

S45_E1_TRACE = ROOT / "outputs/c1sf/20261029/evaluations/so3/e1_step6000_repeat1_none/states.npz"
MODES = ("row", "e1", "e2", "e3")


def resolve(net, step):
    if net == "t1d":
        receipt, checkpoint, config = (read(S45TR / "receipts/formal_t1d.json"),
                                       T1D_RUN / f"s45tr_t1d_s3407-step={step}.ckpt", T1D_CONFIG)
    else:
        receipt = read(OUT / "receipts" / f"formal_{net}.json")
        checkpoint, config = output_dir(net, "formal") / f"{run_name(net)}-step={step}.ckpt", config_path(net, "formal")
    require(receipt["status"] == "PASS", "network has no verified formal training")
    require(sha(checkpoint) == receipt["checkpoint_sha256_by_step"][str(step)], "checkpoint changed since its receipt")
    return checkpoint, config


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--net", required=True, choices=("t1d",) + FORMAL_ARMS)
    p.add_argument("--mode", required=True, choices=MODES)
    p.add_argument("--step", type=int, choices=(5000, 5500, 6000), default=6000)
    a = p.parse_args(argv)
    require(a.step == 6000 or a.mode == "e1", "late checkpoints are E1 only")
    require(a.net != "t1d" or (a.mode == "e1" and a.step == 6000), "t1d is re-evaluated for the reproduction check only")
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in tuple(map(str, PHYSICAL_GPUS)), "one GPU among 2-7")
    XE.configure_environment()
    XE.set_policy()
    verify_protocol()
    checkpoint, config = resolve(a.net, a.step)
    from config import load_config
    cfg = load_config(str(config))
    mode = install_hook(cfg) if a.net != "t1d" else None
    output = OUT / "evaluations" / a.net / f"{a.mode}_step{a.step}"
    require(not output.exists(), "Preserve existing evaluation: " + str(output))
    output.mkdir(parents=True)
    identity = {"net": a.net, "seed": 3407, "step": a.step, "mode": a.mode, "route_corruption": mode,
                "checkpoint_sha256": sha(checkpoint), "config_sha256": sha(config), "script_sha256": sha(__file__),
                "xcl_evaluator_sha256": sha(XE.__file__), "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}
    XE.save_new(output / "launch.json", identity)
    try:
        report = XE.trace({"name": "s45ct_" + a.net}, checkpoint, cfg, output, a.mode,
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
