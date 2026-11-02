#!/usr/bin/env python3
"""T1A-UB evaluation (one network x one reading per process; single seed 3407; never trains or selects).

  evaluate.py trace   --net N --mode {e1,e2,e3}      zgz traces through the XCL evaluator (tools/xcl/evaluate.py trace):
                                                     E1 own closed loop (steps 5000 / 5500 / 6000 via N_5000 / N_5500),
                                                     E2 fed S45's own E1 states, E3 GT previous state. One GPU among 2-7.
  evaluate.py readout --net N --task T               S45-TR's readout tasks (tools/s45tr/readouts.py run_task, unchanged)
                                                     with this study's loader: tl_e1 (also N_5000 / N_5500), tl_f1, tl_e3,
                                                     zl_e3, tl_T10, tl_R5, tl_Fs. One GPU among 2-7.
  evaluate.py row     --net N --block B              the main row in the deployment form, GPU 4 exclusive: routes on the
                                                     latency path (semkine/s45sm_fast.py, bit-identical routes), T1a's
                                                     EdgeConv in its reparameterised form; accuracy from a fresh closed
                                                     loop of that form; latency / MACs / parameters through
                                                     tools/make_s36_row.py, plus the test router's parameters and MACs,
                                                     the reparameterised products' MACs and per-packet tail latency.
                                                     `--net t1aref_<B>`: T1a (original implementation) re-measured in block
                                                     B, the latency budget of every network measured in that block;
                                                     `--net t1arp_<B>`: T1a in its deployment form (descriptive only).
Networks: `t1a` (the baseline, S45-TR's checkpoints), the registered arms and cells (tools/t1aub/registry_r<k>.json).
Outputs below outputs/t1aub/20261102/{evaluations,readouts}/<net>/...; existing outputs are never overwritten.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.t1aub import common as C  # noqa: E402

READOUT_TASKS_6000 = ("tl_e1", "tl_f1", "tl_e3", "zl_e3", "tl_T10", "tl_R5", "tl_Fs")


def load(name, device, *, fast=False, reparam=False):
    """(model, cfg, checkpoint, installed) of a network, hooks installed before the model is built."""
    from model import MNISTModel
    checkpoint, _, _, _ = C.resolve(name)
    cfg = C.load_config_for(name)
    state = C.install_for_config(cfg, evaluation=True, fast=fast, reparam=reparam)
    model = MNISTModel.load_from_checkpoint(str(checkpoint), cfg=cfg, map_location=device, strict=True)
    return model.to(device).eval(), cfg, checkpoint, state


def identity(name, extra):
    checkpoint, config, fam, step = C.resolve(name)
    return {"net": name, "family": fam, "seed": C.SEED, "step": step, "checkpoint_sha256": C.sha(checkpoint),
            "config_sha256": C.sha(config), "script_sha256": C.sha(__file__), "code_identity_sha256": C.code_identity(),
            "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), **extra}


def trace(name, mode):
    from tools.xcl import evaluate as XE
    base, step = C.split_step(name)
    C.require(mode in ("e1", "e2", "e3") and (step == 6000 or mode == "e1"), "late checkpoints are E1 only")
    C.require(base != "t1a", "T1a's traces are S45-TR's (outputs/s45tr/20261030/evaluations/t1a)")
    C.require_one_gpu()
    XE.configure_environment()
    XE.set_policy()
    C.verify_protocol()
    checkpoint, _, _, _ = C.resolve(name)
    cfg = C.load_config_for(name)
    state = C.install_for_config(cfg, evaluation=True)
    output = C.OUT / "evaluations" / base / f"{mode}_step{step}"
    C.require(not output.exists(), "Preserve existing evaluation: " + str(output))
    output.mkdir(parents=True)
    ident = identity(name, {"mode": mode, "installed": state, "xcl_evaluator_sha256": C.sha(XE.__file__)})
    XE.save_new(output / "launch.json", ident)
    try:
        report = XE.trace({"name": "t1aub_" + base}, checkpoint, cfg, output, mode,
                          C.S45_E1_TRACE if mode == "e2" else None, "none")
        C.require(C.sha(checkpoint) == ident["checkpoint_sha256"], "Inputs changed")
        XE.save_new(output / "report.json", {**ident, **report, "no_training_or_selection": True})
    except BaseException as exc:
        XE.save_new(output / "failure.json", {**ident, "status": "FAILED_PRESERVED", "error": f"{type(exc).__name__}: {exc}"})
        raise
    print("COMPLETE", output, flush=True)


def readout(name, task):
    from tools.xcl import evaluate as XE
    from tools.s45tr import readouts as R
    base, step = C.split_step(name)
    C.require(base != "t1a", "T1a's readouts are S45-TR's / S45-GRADE's")
    C.require(task in (READOUT_TASKS_6000 if step == 6000 else ("tl_e1",)), f"task {task} not registered for {name}")
    C.require_one_gpu()
    XE.configure_environment()
    XE.set_policy()
    C.verify_protocol()

    def loader(_net, device):
        model, cfg, checkpoint, _ = load(name, device)
        return model, cfg, checkpoint
    R.load = loader
    R.run_task(name, task, out_root=C.OUT / "readouts")
    out = C.OUT / "readouts" / name / task
    C.save(out / "t1aub_identity.json", identity(name, {"task": task}), exclusive=True)
    print("COMPLETE", name, task, C.sha(out / "states.npz"), flush=True)


# --------------------------------------------------------------------------- main row in the deployment form
def tail_latency(model, cfg, device, passes=3, max_packets=600):
    """Per-packet forward_packet times (ms, CUDA-synchronised around each packet) on make_s36_row's 600 lyq_local packets."""
    import numpy as np
    import torch
    from tools import make_s36_row as R
    from semkine import eval_track as ET
    from semkine.dataset import sequences_for_split, splits_manifest
    root = Path(cfg["DATA"]["ROOT"])
    seq, d = next((s, d) for split in ("val_core", "train")
                  for s, d in sequences_for_split(root, split, splits_manifest(cfg)) if s == "lyq_local")
    events, offsets, aux, pos51 = ET.load_sequence(root, d, seq)
    tsub_p = root / d / f"{seq}_tsub.npy"
    tsub = np.load(tsub_p, mmap_mode="r") if tsub_p.exists() else None
    betas = torch.tensor(aux["betas"], dtype=torch.float32, device=device).view(1, -1)
    K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=device).view(1, 3, 3)
    model.set_hand_context(betas, K)
    packets, counts = [], []
    for a, b in np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2):
        prev = torch.from_numpy(pos51[a].copy()).view(1, -1).to(device)
        for end in np.arange(a + R.STEP_MS - 1, b, R.STEP_MS, dtype=np.int64):
            ev5 = ET._window_events(events, offsets, tsub, int(end), R.STEP_MS)
            packet = ET.make_eval_packet(ev5, prev, betas, K, R.STEP_MS, device)
            packets.append(ET.attach_xag_sources(packet, cfg, seq, offsets, int(end), R.STEP_MS, device))
            counts.append(len(ev5))
            if len(packets) >= max_packets:
                break
        if len(packets) >= max_packets:
            break
    with torch.no_grad():
        for b_ in packets[:100]:
            model.forward_packet(b_)
        torch.cuda.synchronize()
        times = np.zeros((passes, len(packets)))
        for p in range(passes):
            for i, b_ in enumerate(packets):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                model.forward_packet(b_)
                torch.cuda.synchronize()
                times[p, i] = (time.perf_counter() - t0) * 1e3
    best = times.min(0)
    q = lambda a, x: float(np.percentile(a, x))  # noqa: E731
    return {"packets": len(packets), "events_mean": float(np.mean(counts)), "events_p99": q(np.asarray(counts), 99),
            "per_packet_min_over_passes_ms": {"mean": float(best.mean()), "p50": q(best, 50), "p90": q(best, 90),
                                              "p95": q(best, 95), "p99": q(best, 99), "max": float(best.max())},
            "pooled_ms": {"mean": float(times.mean()), "p50": q(times, 50), "p90": q(times, 90), "p95": q(times, 95),
                          "p99": q(times, 99), "max": float(times.max())}}


def cost_extras(cfg, state):
    """Router parameters / MACs (S45-SM's router_cost) and the reparameterised products' MACs on make_s36_row's fixture."""
    import torch
    from tools import make_s36_row as R
    from semkine import t1aub as M
    with torch.no_grad(), M.count_macs() as counter:
        macs_thop, note = R.macs_forward_packet(cfg)
        custom = int(counter["count"])
    router = {"params": 0, "macs": 0.0}
    if state.get("route") and state["route"]["router"] not in ("gt",):
        from tools.s45sm.summary import router_cost
        router = router_cost({"router": state["route"]["router"], "checkpoint": state["route"]["checkpoint"]})
    return {"macs_thop_forward_packet": float(macs_thop), "macs_functional": custom, "router": router,
            "macs_note": note}


def row(name, block):
    import torch
    from tools.xcl import evaluate as XE
    from tools import make_s36_row as R
    C.require(block.isidentifier(), "block tag")
    C.require(os.environ.get("CUDA_VISIBLE_DEVICES") == str(C.LATENCY_GPU), "main rows are measured on GPU 4 only")
    ref = name == f"t1aref_{block}"
    rp = name == f"t1arp_{block}"
    net = "t1a" if ref or rp else name
    C.require(C.split_step(net)[1] == 6000, "main rows are step 6000")
    XE.configure_environment()
    XE.set_policy()
    C.verify_protocol()
    checkpoint, _, fam, _ = C.resolve(net)
    cfg = C.load_config_for(net)
    deploy = not ref
    eg = bool(cfg["T1AUB"]["EG"])
    state = C.install_for_config(cfg, evaluation=True, fast=deploy, reparam=deploy and eg)
    output = C.OUT / "evaluations" / name / "row_step6000"
    C.require(not output.exists(), "Preserve existing evaluation: " + str(output))
    output.mkdir(parents=True)
    ident = identity(net, {"row_name": name, "block": block, "mode": "row", "installed": state, "deployment_form": deploy,
                           "xcl_evaluator_sha256": C.sha(XE.__file__), "row_tool_sha256": C.sha(R.__file__)})
    XE.save_new(output / "launch.json", ident)
    try:
        report = XE.trace({"name": "t1aub_" + name}, checkpoint, cfg, output, "row", None, "none")
        report["main_row"] = XE.make_main_row({"name": "t1aub_" + name}, checkpoint, cfg, output, report, 6000)
        device = torch.device("cuda")
        from model import MNISTModel
        model = MNISTModel.load_from_checkpoint(str(checkpoint), cfg=cfg, map_location=device, strict=True).to(device).eval()
        tail = tail_latency(model, cfg, device)
        del model
        torch.cuda.empty_cache()
        extras = cost_extras(cfg, state)
        main = C.read(report["main_row"]["path"])
        scale = R.FULL_PUBLISHED_MS / main["anchor_full_raw_ms"]
        cost = {"params_model": int(main["params_total"]), "params_router": int(extras["router"]["params"]),
                "params_total": int(main["params_total"]) + int(extras["router"]["params"]),
                "macs_total": extras["macs_thop_forward_packet"] + extras["macs_functional"] + extras["router"]["macs"],
                "latency_ms": float(main["latency_ms_scaled_full1p75"]), "latency_raw_ms": float(main["latency_ms_raw"]),
                "anchor_raw_ms": float(main["anchor_full_raw_ms"]),
                "tail_scaled_ms": {k: {q: v * scale for q, v in tail[k].items()}
                                   for k in ("per_packet_min_over_passes_ms", "pooled_ms")},
                "tail_raw": tail, **extras}
        C.require(C.sha(checkpoint) == ident["checkpoint_sha256"], "Inputs changed")
        XE.save_new(output / "report.json", {**ident, **report, "cost": cost, "no_training_or_selection": True})
    except BaseException as exc:
        XE.save_new(output / "failure.json", {**ident, "status": "FAILED_PRESERVED", "error": f"{type(exc).__name__}: {exc}"})
        raise
    print("COMPLETE", output, flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    t = sub.add_parser("trace")
    t.add_argument("--net", required=True)
    t.add_argument("--mode", required=True, choices=("e1", "e2", "e3"))
    o = sub.add_parser("readout")
    o.add_argument("--net", required=True)
    o.add_argument("--task", required=True)
    r = sub.add_parser("row")
    r.add_argument("--net", required=True)
    r.add_argument("--block", required=True)
    a = p.parse_args(argv)
    if a.verb == "trace":
        trace(a.net, a.mode)
    elif a.verb == "readout":
        readout(a.net, a.task)
    else:
        row(a.net, a.block)


if __name__ == "__main__":
    main()
