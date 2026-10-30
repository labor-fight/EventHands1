#!/usr/bin/env python3
"""AGG778 Stage D evaluation: the XCL readouts of one trained arm through the unchanged XCL functions
(`tools/xcl/evaluate.py` `trace` = E1 / E2 / E3 / E4 on the complete 2590-endpoint zgz grid, `make_main_row` = the
original `tools/make_s36_row.py` row with latency). One process = one readout on one physical GPU 4-7; the latency row
is run alone. Fresh output directory only; the checkpoint identity comes from the arm's formal receipt.

    CUDA_VISIBLE_DEVICES=4 CUDA_DEVICE_ORDER=PCI_BUS_ID python tools/agg778/d_eval.py --arm agg778_matpt_a3_s3407 --mode row
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.agg778.d_common import ARMS, OUT_D, paths, read, require, save, sha  # noqa: E402
from tools.agg778.common import PROTOCOL, STRICT_TRACE  # noqa: E402

MODES = ("row", "e1", "e2", "e3", "gt_pose", "gt_transl")


def checkpoint_of(arm: str, step: int) -> Path:
    receipt = read(OUT_D / "receipts" / f"formal_{arm}.json")
    require(receipt["status"] == "PASS", "arm has no verified formal training")
    run = paths(arm)["output"]
    if step == 6000:
        ck = run / "repro_final.ckpt"
        expected = receipt["final_checkpoint_sha256"]
    else:
        cands = [p for p in run.glob("*step=*.ckpt") if (m := re.search(r"step=(\d+)", p.name)) and int(m[1]) == step]
        require(len(cands) == 1, f"missing step{step} checkpoint")
        ck = cands[0]
        expected = receipt["checkpoint_sha256_by_step"][str(step)]
    require(sha(ck) == expected, "checkpoint SHA changed since the formal receipt")
    return ck


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--arm", choices=tuple(ARMS), required=True)
    ap.add_argument("--mode", choices=MODES, default="e1")
    ap.add_argument("--step", type=int, choices=(5000, 5500, 6000), default=6000)
    ap.add_argument("--repeat", type=int, choices=(1, 2), default=1)
    ap.add_argument("--intervention", choices=("none", "h3_zero"), default="none")
    a = ap.parse_args()
    if a.step != 6000 and (a.mode != "e1" or a.intervention != "none"):
        ap.error("E5 late checkpoints are unmodified E1 only")
    if a.mode == "row" and a.intervention != "none":
        ap.error("main rows cannot be intervention results")
    physical = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    require(physical in ("4", "5", "6", "7"), "AGG778 evaluation uses exactly one physical GPU 4/5/6/7")
    p = read(PROTOCOL)
    bad = [k for k, v in p["files"].items() if sha(ROOT / k) != v]
    require(not bad, f"AGG778 frozen sources changed: {bad}")
    from tools.xcl import evaluate as XE
    XE.configure_environment()
    XE.set_policy()
    from config import load_config
    out = OUT_D / "evaluations" / a.arm / f"{a.mode}_step{a.step}_repeat{a.repeat}_{a.intervention}"
    require(not os.path.lexists(out), "preserve the existing evaluation")
    ck = checkpoint_of(a.arm, a.step)
    config = paths(a.arm)["config"]
    cfg = load_config(str(config))
    out.mkdir(parents=True, exist_ok=False)
    run = {"name": a.arm, "prefix": a.arm.removesuffix("_s3407")}
    identity = {"arm": a.arm, "mode": a.mode, "step": a.step, "repeat": a.repeat, "intervention": a.intervention,
                "checkpoint_sha256": sha(ck), "config_sha256": sha(config), "protocol_sha256": sha(PROTOCOL),
                "script_sha256": sha(__file__), "xcl_evaluate_sha256": sha(XE.__file__), "physical_gpu": physical}
    save(out / "launch.json", identity, exclusive=True)
    canonical = ROOT / STRICT_TRACE["path"]
    try:
        report = XE.trace(run, ck, cfg, out, a.mode, canonical if a.mode == "e2" else None, a.intervention)
        if a.mode == "row":
            row = XE.make_main_row(run, ck, cfg, out, report, a.step)
            report["main_row"] = row
            dst = ROOT / "outputs" / "semkine" / f"{run['prefix']}_main_row.json"
            require(not dst.exists(), "main row already published")
            shutil.copyfile(row["path"], dst)
            report["main_row_published"] = str(dst.relative_to(ROOT))
        require(sha(ck) == identity["checkpoint_sha256"] and sha(config) == identity["config_sha256"],
                "input identity changed during evaluation")
        save(out / "report.json", {**identity, **report, "no_training_or_selection": True}, exclusive=True)
    except BaseException as exc:
        save(out / "failure.json", {**identity, "status": "FAILED_PRESERVED", "error": f"{type(exc).__name__}: {exc}"},
             exclusive=True)
        raise
    print("COMPLETE", out, json.dumps(report["metrics"]["local"]), flush=True)


if __name__ == "__main__":
    main()
