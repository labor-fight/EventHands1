#!/usr/bin/env python3
"""C1SF evaluation: one process = one run x readout, on one physical GPU among 4-7.

zgz readouts (row / e1 / e2 / e3, interventions none / geometry_zero / h3_zero) delegate to the XCL evaluator's
`trace` and `make_main_row` unchanged (tools/xcl/evaluate.py, frozen SHA), so C1SF numbers are on the same
scorer as the XCL C1 numbers. `train_e3` is the zero-training reading of step 1: GT-previous single step on the
18 base training recordings, with the same packet / capture / FK scoring code, plus the identity (no-update)
baseline on the same endpoints.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.xcl import evaluate as XE  # noqa: E402
from tools.c1sf.common import (C1_CHECKPOINT, C1_CHECKPOINT_SHA256, C1_E1_TRACE, C1_E1_TRACE_SHA256, CANONICAL,  # noqa: E402
                               FORMAL_ARMS, OUT, PHYSICAL_GPUS, PROTOCOL, TRAIN_BASE, config_path, read, require, run_name, sha,
                               verify_protocol, zn_mode)

MODES = ("row", "e1", "e2", "e3", "train_e3")
INTERVENTIONS = ("none", "geometry_zero", "h3_zero")


def resolve(run, step):
    """(checkpoint, config) of a registered run; C1 is the published baseline, arms come from formal receipts."""
    if run == "c1":
        require(step == 6000, "C1 is read at its published step 6000 only")
        require(sha(C1_CHECKPOINT) == C1_CHECKPOINT_SHA256, "C1 checkpoint changed")
        return C1_CHECKPOINT, CANONICAL
    require(run in FORMAL_ARMS, "unknown run " + repr(run))
    receipt = read(OUT / "receipts" / f"formal_{run}.json")
    require(receipt["status"] == "PASS", "arm has no verified formal training")
    out = OUT / "runs" / run_name(run)
    checkpoint = out / f"{run_name(run)}-step={step}.ckpt"
    require(sha(checkpoint) == receipt["checkpoint_sha256_by_step"][str(step)], "checkpoint changed since its receipt")
    return checkpoint, config_path(run, "formal")


def trace_train(checkpoint, cfg, output, intervention):
    import numpy as np
    import torch
    from mano_layer import ManoLayer
    from model import MNISTModel
    from semkine import eval_track as ET
    from semkine.dataset import sequences_for_split, splits_manifest
    from tools.xcl.interventions import apply
    device = torch.device("cuda")
    model = MNISTModel.load_from_checkpoint(str(checkpoint), cfg=cfg, map_location=device, strict=True).to(device).eval()
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    root = Path(cfg["DATA"]["ROOT"])
    found = dict(sequences_for_split(root, "train", splits_manifest(cfg)))
    require(all(s in found for s in TRAIN_BASE), "a base training recording is missing from the train split")
    arrays, results = {}, []
    native = model.forward_packet
    rng = np.random.default_rng(0)
    try:
        with torch.no_grad(), apply(model, intervention):
            for sequence in TRAIN_BASE:
                directory = found[sequence]
                _, _, aux, gt = ET.load_sequence(root, directory, sequence)
                rows = XE.endpoint_grid(aux)
                incoming, predictions, targets, priors = [], [], [], []

                def capture(packet):
                    i = len(predictions)
                    require(i < len(rows), "Too many evaluation packets")
                    _, end, previous_ms, first = rows[i]
                    raw = packet.prev_state.detach().cpu().numpy()[0].copy()
                    truth = np.asarray(gt[previous_ms], np.float32).copy()
                    current = np.asarray(gt[end], np.float32).copy()
                    previous = XE.select_previous("e3", raw, truth)
                    packet = replace(packet, prev_state=torch.from_numpy(previous).view(1, -1).to(device),
                                     target=torch.from_numpy(current).view(1, -1).to(device))
                    pred = native(packet)
                    value = pred.detach().cpu().numpy()[0].copy()
                    if value.shape != (51,) or not np.isfinite(value).all():
                        raise FloatingPointError(f"Nonfinite endpoint retained as failure: {sequence}:{end}")
                    if not first and not np.array_equal(raw, predictions[-1]):
                        raise RuntimeError("Native closed loop previous output contract changed")
                    incoming.append(raw); predictions.append(value); targets.append(current); priors.append(truth)
                    return pred
                model.forward_packet = capture
                result = ET.track_sequence(model, mano, cfg, root, directory, sequence, 50, device, rng, 1.0, 1.0, None)
                require(result is not None and result["n_frames"] == len(predictions) == len(rows),
                        "Incomplete evaluation; no endpoints may be dropped: " + sequence)
                beta = torch.tensor(np.array(aux["betas"]), device=device, dtype=torch.float32).view(1, -1)
                predictions, targets, priors = np.stack(predictions), np.stack(targets), np.stack(priors)
                metrics = XE.score(mano, predictions, targets, beta, device)
                identity = XE.score(mano, priors, targets, beta, device)
                for key in XE.ET_METRICS:
                    require(np.isclose(metrics[key].mean(), result[key], rtol=0, atol=1e-9),
                            f"Canonical FK/scoring mismatch: {sequence}/{key}")
                for key, value in {"prediction": predictions, "target": targets, "gt_previous": priors,
                                   "end_ms": np.array([r[1] for r in rows], np.int64),
                                   **metrics, **{"identity_" + k: v for k, v in identity.items()}}.items():
                    if not np.isfinite(value).all():
                        raise FloatingPointError("Nonfinite metrics cannot be omitted")
                    arrays[sequence + "/" + key] = value
                results.append({"seq": sequence, "n_frames": len(predictions),
                                **{k: float(v.mean()) for k, v in metrics.items()},
                                **{"identity_" + k: float(v.mean()) for k, v in identity.items()}})
                (output / "progress.json").write_text(json.dumps({"done": [r["seq"] for r in results]}) + "\n")
    finally:
        model.forward_packet = native
    np.savez_compressed(output / "states.npz", **arrays)
    return {"status": "COMPLETE", "n_sequences": len(results), "n_endpoints": int(sum(r["n_frames"] for r in results)),
            "n_failed": 0, "per_sequence": results, "states_sha256": XE.sha(output / "states.npz")}


def zgz_identity(cfg, states):
    """No-update baseline on the E3 endpoints: score GT(previous endpoint) against GT(end), same scorer."""
    import numpy as np
    import torch
    from mano_layer import ManoLayer
    from semkine import eval_track as ET
    from semkine.dataset import sequences_for_split, splits_manifest
    device = torch.device("cuda")
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    root = Path(cfg["DATA"]["ROOT"])
    out = {}
    with np.load(states, allow_pickle=False) as z:
        for sequence, directory in sequences_for_split(root, "val_core", splits_manifest(cfg)):
            _, _, aux, _ = ET.load_sequence(root, directory, sequence)
            beta = torch.tensor(np.array(aux["betas"]), device=device, dtype=torch.float32).view(1, -1)
            m = XE.score(mano, z[sequence + "/gt_previous"], z[sequence + "/target"], beta, device)
            out[sequence] = {k: float(v.mean()) for k, v in m.items()}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--run", required=True, choices=("c1", *FORMAL_ARMS))
    p.add_argument("--mode", required=True, choices=MODES)
    p.add_argument("--step", type=int, choices=(5000, 5500, 6000), default=6000)
    p.add_argument("--intervention", choices=INTERVENTIONS, default="none")
    p.add_argument("--repeat", type=int, choices=(1, 2), default=1)
    args = p.parse_args(argv)
    if args.step != 6000 and (args.mode != "e1" or args.intervention != "none"):
        p.error("late checkpoints are unmodified E1 only")
    if args.mode == "row" and args.intervention != "none":
        p.error("main rows cannot be intervention results")
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in tuple(map(str, PHYSICAL_GPUS)),
            "C1SF evaluation uses exactly one physical GPU among 2-7 per process")
    XE.configure_environment()
    XE.set_policy()
    protocol_sha = sha(PROTOCOL)
    verify_protocol()
    checkpoint, config = resolve(args.run, args.step)
    from config import load_config
    cfg = load_config(config)
    if zn_mode(cfg) == "root_relative":
        from semkine import c1sf_zrel
        c1sf_zrel.install(cfg["MANO"]["NPZ"])
    output = OUT / "evaluations" / args.run / f"{args.mode}_step{args.step}_repeat{args.repeat}_{args.intervention}"
    require(not output.exists(), "Preserve existing evaluation: " + str(output))
    output.mkdir(parents=True)
    identity = {"run": args.run, "seed": 3407, "step": args.step, "mode": args.mode, "intervention": args.intervention,
                "repeat": args.repeat, "protocol_sha256": protocol_sha, "checkpoint_sha256": sha(checkpoint),
                "config_sha256": sha(config), "script_sha256": sha(__file__), "xcl_evaluator_sha256": sha(XE.__file__),
                "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "zn_mode": zn_mode(cfg)}
    XE.save_new(output / "launch.json", identity)
    try:
        if args.mode == "train_e3":
            report = trace_train(checkpoint, cfg, output, args.intervention)
        else:
            require(sha(C1_E1_TRACE) == C1_E1_TRACE_SHA256, "Canonical C1 E1 trace changed")
            report = XE.trace({"name": "c1sf_" + args.run}, checkpoint, cfg, output, args.mode,
                              C1_E1_TRACE if args.mode == "e2" else None, args.intervention)
            if args.mode == "row":
                report["main_row"] = XE.make_main_row({"name": run_name(args.run) if args.run != "c1" else "c1sf_c1_s3407"},
                                                      checkpoint, cfg, output, report, args.step)
            if args.mode == "e3":
                report["identity"] = zgz_identity(cfg, output / "states.npz")
        require(sha(PROTOCOL) == protocol_sha and sha(checkpoint) == identity["checkpoint_sha256"], "Inputs changed")
        XE.save_new(output / "report.json", {**identity, **report, "no_training_or_selection": True})
    except BaseException as exc:
        XE.save_new(output / "failure.json", {**identity, "status": "FAILED_PRESERVED", "error": f"{type(exc).__name__}: {exc}"})
        raise
    print("COMPLETE", output, flush=True)


if __name__ == "__main__":
    main()
