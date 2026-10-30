#!/usr/bin/env python3
"""S45-TR evaluation: one process = one formal arm x zgz readout, on one physical GPU among 2-7.

Delegates to the XCL evaluator's `trace` / `make_main_row` unchanged (tools/xcl/evaluate.py), exactly as C1SF did for
S45, so every arm is scored by S45's scorer: E1 = own closed loop, E2 = the arm fed S45's own closed-loop states
(S45's E1 trace, outputs/c1sf/20261029/evaluations/so3/e1_step6000_repeat1_none), E3 = GT previous state; `row` adds
latency / FLOPs / params via tools/make_s36_row.py. The arm's evaluation-time hook is installed first: N1 and T1a
build their XYZMeshNet subclass, T1b routes with the frozen T0 (its zgz packets carry no ground-truth source, so the
evaluator's ground-truth source attachment is skipped for T1b only); N2 / N3 / N4 change training only.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.xcl import evaluate as XE  # noqa: E402
from tools.s45tr.common import (FORMAL_ARMS, OUT, PHYSICAL_GPUS, config_path, hook_of, read, require,  # noqa: E402
                                run_name, sha, verify_protocol)

S45_E1_TRACE = ROOT / "outputs/c1sf/20261029/evaluations/so3/e1_step6000_repeat1_none/states.npz"
MODES = ("row", "e1", "e2", "e3")
EVAL_HOOKS = ("n1", "t1a", "t1b")


def resolve(arm, step):
    require(arm in FORMAL_ARMS, "unknown arm " + repr(arm))
    receipt = read(OUT / "receipts" / f"formal_{arm}.json")
    require(receipt["status"] == "PASS", "arm has no verified formal training")
    checkpoint = OUT / "runs" / run_name(arm) / f"{run_name(arm)}-step={step}.ckpt"
    require(sha(checkpoint) == receipt["checkpoint_sha256_by_step"][str(step)], "checkpoint changed since its receipt")
    return checkpoint, config_path(arm, "formal")


def install_eval_hooks(cfg):
    hook = hook_of(cfg)
    if hook not in EVAL_HOOKS:
        return None
    from tools.s45tr.common import install_hooks
    install_hooks(cfg)
    if hook == "t1b":
        from semkine import eval_track as ET
        ET.attach_xag_sources = lambda batch, *args, **kwargs: batch
    return hook


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--arm", required=True, choices=FORMAL_ARMS)
    p.add_argument("--mode", required=True, choices=MODES)
    p.add_argument("--step", type=int, choices=(5000, 5500, 6000), default=6000)
    args = p.parse_args(argv)
    if args.step != 6000 and args.mode != "e1":
        p.error("late checkpoints are E1 only")
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in tuple(map(str, PHYSICAL_GPUS)),
            "S45-TR evaluation uses exactly one physical GPU among 2-7 per process")
    XE.configure_environment()
    XE.set_policy()
    verify_protocol()
    checkpoint, config = resolve(args.arm, args.step)
    from config import load_config
    cfg = load_config(config)
    hook = install_eval_hooks(cfg)
    output = OUT / "evaluations" / args.arm / f"{args.mode}_step{args.step}"
    require(not output.exists(), "Preserve existing evaluation: " + str(output))
    output.mkdir(parents=True)
    identity = {"arm": args.arm, "seed": 3407, "step": args.step, "mode": args.mode, "eval_hook": hook,
                "checkpoint_sha256": sha(checkpoint), "config_sha256": sha(config), "script_sha256": sha(__file__),
                "xcl_evaluator_sha256": sha(XE.__file__), "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}
    XE.save_new(output / "launch.json", identity)
    try:
        report = XE.trace({"name": "s45tr_" + args.arm}, checkpoint, cfg, output, args.mode,
                          S45_E1_TRACE if args.mode == "e2" else None, "none")
        if args.mode == "row":
            report["main_row"] = XE.make_main_row({"name": run_name(args.arm)}, checkpoint, cfg, output, report, args.step)
        require(sha(checkpoint) == identity["checkpoint_sha256"], "Inputs changed")
        XE.save_new(output / "report.json", {**identity, **report, "no_training_or_selection": True})
    except BaseException as exc:
        XE.save_new(output / "failure.json", {**identity, "status": "FAILED_PRESERVED", "error": f"{type(exc).__name__}: {exc}"})
        raise
    print("COMPLETE", output, flush=True)


if __name__ == "__main__":
    main()
