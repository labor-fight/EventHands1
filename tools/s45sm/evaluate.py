"""S45-SM evaluation: one network x one zgz readout per process, on one physical GPU among 2-7.

Delegates to the XCL evaluator's `trace` / `make_main_row` unchanged (as S45-TR / S45-CT / S45-RT / S45-AT did), with the
network's routing hook installed first and, for router networks, no ground-truth XAG source attached:
  evaluate --net N --mode {e1,e2,e3,row} [--step {5000,5500,6000}]
      E1 = own closed loop (steps 5000 / 5500 / 6000; late steps E1 only), E2 = fed S45's own E1 states, E3 = GT previous
      state, `row` = latency / FLOPs / params on the original path. `row` exists for the round-1 arms, the label-route
      oracles and `--net s45ref` (S45 itself, no hook; the round-1 latency reference); later arms' main rows use `row` below.
  row --net N --block B
      the main row on the latency path (semkine/s45sm_fast.py; semkine/s45sm_soft.py for `spfa`), GPU 4 exclusive (driver
      kind `exclusive`). For pfa / face_route arms trained on the original path the routes and every output are bit-identical
      to the arm's (tests/test_s45sm_fast.py), so the accuracy columns repeat the arm's own row and the latency column is the
      deployment implementation's. `--net s45ref_<B>` is S45 itself (no hook) re-measured in block B, the latency reference
      of every network measured in the same block.
A network is an arm (trained or a zero-training cell: its trunk's checkpoint under the cell's own config) or an oracle. Writes
evaluations/<net>/<mode>_step<step>/ (`row_fast_step6000`) with launch.json (the identity), report.json or failure.json; the
output directory must not exist. A cell's evaluation writes its swap receipt afterwards.
"""
from __future__ import annotations

import os

from .paths import OUT, ROOT, S45_CONFIG, S45_E1_TRACE, require, require_one_gpu, sha
from .protocol import code_identity, verify_protocol
from .registry import arm_spec, net_spec, oracles, registry, run_name
from .routing import (cell_net, install_for_config, register_for_net, resolve, write_swap_receipt)

MODES = ("row", "e1", "e2", "e3")
SLOW_ROW_ROUND = 1


def _identity_hooks(arm):
    """hook-module SHAs recorded from round 3 on (the round-1 evaluator knew one hook module)."""
    return {m: sha(ROOT / f"semkine/{m}.py") for m in ("s45sm", "s45sm_fast", "s45sm_soft")}


def evaluate(net, mode, step):
    from tools.xcl import evaluate as XE
    from tools.s45tr.common import S45_CHECKPOINT, S45_CHECKPOINT_SHA256
    oracle, ref = net in oracles(), net == "s45ref"
    if ref:
        require(mode == "row", "unregistered S45-SM arm " + net)
    elif oracle:
        pass
    else:
        require(net in registry()["arms"], "unregistered S45-SM arm " + net)
        require(mode != "row" or arm_spec(net)["round"] == SLOW_ROW_ROUND,
                "main rows of rounds >= 2 are measured by `row` (latency path)")
    require(step == 6000 or mode == "e1", "late checkpoints are E1 only")
    require_one_gpu()
    XE.configure_environment()
    XE.set_policy()
    verify_protocol()
    from config import load_config
    if ref:
        require(sha(S45_CHECKPOINT) == S45_CHECKPOINT_SHA256, "S45 checkpoint changed")
        checkpoint, config = S45_CHECKPOINT, S45_CONFIG
        cfg = load_config(str(config))
        routing = None
    else:
        register_for_net(net)
        checkpoint, config = resolve(net, step)
        cfg = load_config(str(config))
        routing = install_for_config(cfg)
    output = OUT / "evaluations" / net / f"{mode}_step{step}"
    require(not output.exists(), "Preserve existing evaluation: " + str(output))
    output.mkdir(parents=True)
    script = os.path.abspath(__file__)
    identity = {"net": net, "seed": 3407, "step": step, "mode": mode, "routing": routing}
    if oracle:
        identity["oracle_module_sha256"] = sha(ROOT / "semkine/s45sm_oracle.py")
    else:
        identity["router_checkpoint_sha256"] = sha(routing["checkpoint"]) if routing else None
        if not ref and arm_spec(net)["round"] > SLOW_ROW_ROUND:
            identity["hook_modules_sha256"] = _identity_hooks(net)
    identity.update(checkpoint_sha256=sha(checkpoint), config_sha256=sha(config), script_sha256=sha(script),
                    xcl_evaluator_sha256=sha(XE.__file__), code_identity_sha256=code_identity(),
                    physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))
    XE.save_new(output / "launch.json", identity)
    try:
        report = XE.trace({"name": "s45sm_" + net}, checkpoint, cfg, output, mode,
                          S45_E1_TRACE if mode == "e2" else None, "none")
        if mode == "row":
            name = "s45sm_s45ref" if ref else run_name(net)
            report["main_row"] = XE.make_main_row({"name": name}, checkpoint, cfg, output, report, step)
        require(sha(checkpoint) == identity["checkpoint_sha256"], "Inputs changed")
        XE.save_new(output / "report.json", {**identity, **report, "no_training_or_selection": True})
    except BaseException as exc:
        XE.save_new(output / "failure.json", {**identity, "status": "FAILED_PRESERVED", "error": f"{type(exc).__name__}: {exc}"})
        raise
    if not ref and cell_net(net):
        write_swap_receipt(net, "evaluate", f"{mode}_step{step}", step)
    print("COMPLETE", output, flush=True)


def row_fast(net, block):
    from tools.xcl import evaluate as XE
    from tools.s45tr.common import S45_CHECKPOINT, S45_CHECKPOINT_SHA256
    require(block.isidentifier(), "block tag")
    is_ref = net.startswith("s45ref_")
    require(net in registry()["arms"] or net == "s45ref_" + block, "unregistered S45-SM arm " + net)
    require_one_gpu()
    XE.configure_environment()
    XE.set_policy()
    verify_protocol()
    from config import load_config
    routing = None
    if is_ref:
        require(sha(S45_CHECKPOINT) == S45_CHECKPOINT_SHA256, "S45 checkpoint changed")
        checkpoint, config = S45_CHECKPOINT, S45_CONFIG
        cfg = load_config(str(config))
    else:
        register_for_net(net)
        checkpoint, config = resolve(net, 6000)
        cfg = load_config(str(config))
        routing = install_for_config(cfg, force_fast=True)
        require(routing["fast"] is True, "the latency path is not installed")
    output = OUT / "evaluations" / net / "row_fast_step6000"
    require(not output.exists(), "Preserve existing evaluation: " + str(output))
    output.mkdir(parents=True)
    identity = {"net": net, "block": block, "seed": 3407, "step": 6000, "mode": "row_fast", "routing": routing,
                "checkpoint_sha256": sha(checkpoint), "config_sha256": sha(config), "script_sha256": sha(os.path.abspath(__file__)),
                "fast_module_sha256": sha(ROOT / "semkine/s45sm_fast.py"),
                "soft_module_sha256": sha(ROOT / "semkine/s45sm_soft.py"), "xcl_evaluator_sha256": sha(XE.__file__),
                "code_identity_sha256": code_identity(), "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}
    XE.save_new(output / "launch.json", identity)
    try:
        report = XE.trace({"name": "s45sm_fast_" + net}, checkpoint, cfg, output, "row", None, "none")
        name = ("s45sm_fast_" + net if is_ref else run_name(net).replace("s45sm_", "s45sm_fast_"))
        report["main_row"] = XE.make_main_row({"name": name}, checkpoint, cfg, output, report, 6000)
        XE.save_new(output / "report.json", {**identity, **report, "no_training_or_selection": True})
    except BaseException as exc:
        XE.save_new(output / "failure.json", {**identity, "status": "FAILED_PRESERVED", "error": f"{type(exc).__name__}: {exc}"})
        raise
    if not is_ref and cell_net(net):
        write_swap_receipt(net, "row", "row_fast", 6000)
    print("COMPLETE", output, flush=True)
