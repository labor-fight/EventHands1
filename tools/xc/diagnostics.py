#!/usr/bin/env python3
"""Fixed common-start recovery and time-binding diagnostics for the XC factorial.

No training occurs here. Targets are zero during prediction; truth is used only
for the declared common starts and subsequent measurements. Nonfinite trajectories
are retained and invalidate the diagnostic instead of being dropped or averaged out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))

import numpy as np
import torch
from config import load_config
from mano_layer import ManoLayer
from semkine.xyz_time_study import TimeXYZModel
from tools.xb import diagnostics as D

OUT = REPO / "outputs/xc"
TRACE = REPO / "outputs/xb/c0_val_reference.pt"
ARMS = ("g0t0", "g1t0", "g0t1", "g1t1")
SEEDS = (3407, 3408, 3409)
STEPS = 10
PER_SEQ = 24


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(path.suffix + ".tmp").write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    path.with_suffix(path.suffix + ".tmp").replace(path)


def sanitize(value, location="$", invalid=None):
    """Keep every record; convert an originally invalid scalar to an annotated null."""
    if invalid is None:
        invalid = []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        invalid.append(location)
        return None
    if isinstance(value, dict):
        return {str(k): sanitize(v, f"{location}.{k}", invalid) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v, f"{location}[{i}]", invalid) for i, v in enumerate(value)]
    return value


def save_payload(path, value):
    invalid = []
    cleaned = sanitize(value, invalid=invalid)
    cleaned.update(numeric_valid=not invalid, numeric_status="FAIL" if invalid else "PASS",
                   invalid_numeric_paths=invalid, invalid_numeric_count=len(invalid),
                   first_nonfinite=invalid[0] if invalid else None)
    write_json(path, cleaned)


def prepare_selection():
    """Freeze starts before any new candidate is scored, reusing the old C0 trace."""
    target = OUT / "clip_selection.json"
    if (OUT / "frozen_manifest.json").exists():
        raise RuntimeError("cannot replace clip selection after study freeze")
    cached = torch.load(TRACE, map_location="cpu")
    checkpoint = Path(cached["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    selected = D.select_clips(cached["traces"], STEPS, PER_SEQ)
    if set(selected) != {"zgz_global", "zgz_local"} or any(len(v) != PER_SEQ for v in selected.values()):
        raise RuntimeError("expected exactly 24 fixed ten-step clips from each zgz sequence")
    value = {"study": "XYZ-GEOMETRY-CAUSAL-TIME-20261012", "steps": STEPS,
             "clips_per_sequence": PER_SEQ, "selected_trace_indices": selected,
             "trace_cache": str(TRACE.relative_to(REPO)), "trace_sha256": sha(TRACE),
             "reference_checkpoint": str(checkpoint), "reference_checkpoint_sha256": sha(checkpoint),
             "selection": "uniform chronological legal starts; selected independently of XC predictions",
             "perturbation_seed": "20261011 + 10000 * seq_index + clip_index; preserves historical fixed protocol",
             "sequence_order": list(selected)}
    write_json(target, value)
    print(f"Prepared {sum(map(len, selected.values()))} common ten-step clips before training.")


def fixed_inputs():
    selection = json.loads((OUT / "clip_selection.json").read_text())
    if sha(TRACE) != selection["trace_sha256"]:
        raise RuntimeError("fixed common reference trace drifted")
    cached = torch.load(TRACE, map_location="cpu")
    expected = D.select_clips(cached["traces"], STEPS, PER_SEQ)
    if expected != selection["selected_trace_indices"]:
        raise RuntimeError("fixed chronological clip selection drifted")
    cfg = load_config(REPO / "configs/xc/xc_g0t0_s3407.yaml")
    sequences = D.load_sequences(cfg)
    return cfg, sequences, cached["traces"], expected, selection


def load_arm(arm, seed, device):
    from tools.xc.run_study import audit_checkpoint, check_run_metadata, frozen
    manifest = frozen()
    name = f"xc_{arm}_s{seed}"
    job = next(value for value in manifest["jobs"] if value["run"] == name)
    check_run_metadata(job)
    audit = audit_checkpoint(job)
    expected = None
    for line in (OUT / "events.jsonl").read_text().splitlines():
        event = json.loads(line)
        if event.get("action") == "train_finish" and event.get("run") == name and event.get("audit"):
            expected = event["audit"]["checkpoint_sha256"]
    if expected != audit["checkpoint_sha256"]:
        raise RuntimeError(f"diagnostic endpoint changed since completion or lacks a training completion audit: {name}")
    directory = REPO / f"outputs/semkine/xc_{arm}_s{seed}"
    candidates = list(directory.glob(f"xc_{arm}_s{seed}-step=6000.ckpt"))
    if len(candidates) != 1:
        raise RuntimeError(f"exact unique step6000 required: {directory}")
    cfg = load_config(directory / "config_resolved.yaml")
    model = TimeXYZModel.load_from_checkpoint(str(candidates[0]), cfg=cfg, map_location="cpu").to(device).eval()
    return model, candidates[0]


@torch.no_grad()
def replay(model, mano, sequences, traces, selected, device):
    results = []
    for seq_index, (seq, starts) in enumerate(selected.items()):
        data, source = sequences[seq], traces[seq]
        beta, camera = D.context(data, device)
        model.set_hand_context(beta, camera)
        for clip_index, start in enumerate(starts):
            segment = source[start:start + STEPS]
            initial_time = int(segment[0]["previous_end_ms"])
            true_initial = data["gt"][initial_time].copy()
            noise = D.perturbations(seed=20261011 + 10000 * seq_index + clip_index)
            initials = {"natural": segment[0]["prev"].copy(), "clean": true_initial.copy(),
                        "small": true_initial + noise["small"], "large": true_initial + noise["large"]}
            events = [D.ET._window_events(data["events"], data["offsets"], data["tsub"], int(row["end_ms"]), 50)
                      for row in segment]
            truth = np.stack([true_initial] + [data["gt"][int(row["end_ms"])] for row in segment])
            for condition, initial in initials.items():
                model.reset_time_history()
                prev = torch.from_numpy(initial.copy()).view(1, -1).to(device)
                previous_state, previous_time = None, None
                state_time = initial_time
                predictions, stats, offsets, history_intervals, failures = [initial.copy()], [], [], [], []
                failed = False
                for frame_index, (row, ev5) in enumerate(zip(segment, events)):
                    event_start = int(row["end_ms"]) - 49
                    offset_s = (event_start - state_time) * .001
                    if offset_s < -1e-9 or offset_s > .001000001:
                        raise RuntimeError("clip state and event timestamps violate the fixed 0/1ms convention")
                    interval_s = 0.0 if previous_state is None else (state_time - previous_time) * .001
                    offsets.append(offset_s)
                    history_intervals.append(interval_s)
                    if not failed:
                        if previous_state is None:
                            model.reset_time_history()
                        else:
                            model.set_time_history(previous_state, interval_s, query_start_offset_s=offset_s)
                        # An initial clip may start 1ms after a C0 prior, but has no XC
                        # history. Zero velocity means geometry stays static; the offset
                        # is supplied for the common horizon with a zero-interval history.
                        if previous_state is None and offset_s:
                            model.set_time_history(prev, 0.0, query_start_offset_s=offset_s)
                        packet = D.ET.make_eval_packet(ev5, prev, beta, camera, 50, device)
                        if bool(packet.target.any()):
                            raise AssertionError("diagnostic targets must remain zero")
                        incoming = prev.detach().clone()
                        try:
                            pred = model.forward_packet(packet)
                            if not bool(torch.isfinite(pred).all()):
                                failed = True
                                failures.append({"frame_index": frame_index, "reason": "nonfinite_prediction"})
                            prev = pred
                            stats.append({k: float(v) for k, v in model.route_stats.items()})
                            predictions.append(pred[0].cpu().numpy())
                        except Exception as error:
                            failed = True
                            failures.append({"frame_index": frame_index, "reason": type(error).__name__, "message": str(error)})
                            predictions.append(np.full(51, np.nan, np.float32))
                            stats.append({"failed": True})
                        previous_state, previous_time = incoming, state_time
                    else:
                        predictions.append(np.full(51, np.nan, np.float32))
                        stats.append({"unexecuted_after_failure": True})
                    state_time = int(row["end_ms"])
                model.reset_time_history()
                abs_error, ra_error, pj, gj = D.measure(mano, predictions, truth, beta, device)
                held_abs, held_ra, _, _ = D.measure(mano, np.repeat(initial[None], STEPS + 1, axis=0), truth, beta, device)
                results.append({"seq": seq, "clip_id": clip_index, "trace_start_index": start, "condition": condition,
                                "run_id": int(segment[0]["run_id"]), "previous_end_ms": initial_time,
                                "first_end_ms": int(segment[0]["end_ms"]), "query_start_offset_s": offsets,
                                "history_interval_s": history_intervals, "n_raw_events": [len(e) for e in events],
                                "initial_state_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
                                "mpjpe_abs_mm": abs_error.tolist(), "mpjpe_ra_mm": ra_error.tolist(),
                                "held_state_mpjpe_abs_mm": held_abs.tolist(), "held_state_mpjpe_ra_mm": held_ra.tolist(),
                                "correction_gain_ra_mm": (held_ra - ra_error).tolist(),
                                "motion": D.motion_response(pj, gj), "initial_correction": D.initial_correction(pj, gj),
                                "route_stats": stats, "first_hit_ra_step": D.first_hit(ra_error, D.RECOVERY_RA_MM),
                                "first_hit_abs_step": D.first_hit(abs_error, D.RECOVERY_ABS_MM),
                                "trajectory_failures": failures, "trajectory_valid": not failed})
        print(f"{seq}: {len(starts)} common clips x four starts complete", flush=True)
    return results


@torch.no_grad()
def common_time_audit(model, sequences, traces, selected, device):
    """Fixed past velocity fixture isolates time binding while leaving common inputs fixed."""
    records = []
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=False)
    previous_keep = getattr(model, "xyz_keep_aux", False)
    model.xyz_keep_aux = True
    try:
        for seq, starts in selected.items():
            data = sequences[seq]
            beta, camera = D.context(data, device)
            model.set_hand_context(beta, camera)
            for start in starts:
                row = traces[seq][start]
                incoming = torch.from_numpy(row["prev"].copy()).view(1, -1).to(device)
                past = incoming.clone()
                # Predeclared nonzero physical fixture: 1mm translation and .01rad
                # root-AA offset over 50ms. No truth/candidate future enters this audit.
                past[:, 0] -= .001
                past[:, 4] -= .01
                ev5 = D.ET._window_events(data["events"], data["offsets"], data["tsub"], int(row["end_ms"]), 50)
                values = []
                for shift in (0, 2):
                    model.query_bin_shift = shift
                    model.set_time_history(past, .05, query_start_offset_s=.001)
                    pred = model.forward_packet(D.ET.make_eval_packet(ev5, incoming, beta, camera, 50, device))
                    aux = model.xyz_last
                    values.append((pred.detach().clone(), {k: v.detach().clone() if torch.is_tensor(v) else v
                                                           for k, v in aux.items()}))
                pred0, aux0 = values[0]
                pred2, aux2 = values[1]
                identical = {key: torch.equal(aux0[key], aux2[key]) for key in
                             ("tok", "motion_delta", "physical_velocity", "query_start_offset_s", "query_horizon_s", "verts")}
                records.append({"seq": seq, "trace_start_index": start, "n_raw_events": len(ev5),
                                "common_fields_identical": identical, "prediction_max_abs_difference": float((pred0 - pred2).abs().max()),
                                "event_bin_shift_verified": torch.equal(aux2["time_bin"], (aux0["time_bin"] + 2) % 4),
                                "predictions_finite": bool(torch.isfinite(pred0).all() and torch.isfinite(pred2).all())})
    finally:
        model.query_bin_shift = 0
        model.xyz_keep_aux = previous_keep
        model.reset_time_history()
        torch.use_deterministic_algorithms(deterministic, warn_only=warn)
    return {"records": records, "fixture": "fixed 1mm and .01rad root-AA past offset, 50ms interval, 1ms query offset",
            "control": "cyclically shift 4 association/record geometry bins by +2; original time token and motion readout unchanged",
            "interpretation": "same-weight OOD sensitivity diagnostic; does not replace the retrained T contrast",
            "deterministic_algorithms_for_audit_only": True,
            "contract_pass": bool(records) and all(all(r["common_fields_identical"].values()) and
                                                   r["event_bin_shift_verified"] and r["predictions_finite"] for r in records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_selection()
        return
    if args.seed is None:
        parser.error("run requires --seed")
    from tools.xc.run_study import frozen
    frozen()
    cfg, sequences, traces, selected, selection = fixed_inputs()
    device = torch.device(args.device)
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    output = OUT / f"recovery_s{args.seed}.json"
    if output.exists():
        raise RuntimeError(f"diagnostic output already exists; preserve before any explicit rerun: {output}")
    started = time.monotonic()
    payload = {"study": "XYZ-GEOMETRY-CAUSAL-TIME-20261012", "seed": args.seed,
               "protocol": {"window_ms": 50, "steps": STEPS, "selection": selection,
                            "history": "reset at each clip; subsequent own incoming states only, exact state timestamps",
                            "initials": ["natural C0 prior", "clean", "small fixed perturbation", "large fixed perturbation"],
                            "target_input": "zero; GT supplies initial clean/perturbed starts and measures errors only",
                            "control_scope": "+2 bin shift is descriptive OOD sensitivity; no additional training condition"},
               "arms": {}}
    for arm in ARMS:
        model, checkpoint = load_arm(arm, args.seed, device)
        rows = replay(model, mano, sequences, traces, selected, device)
        value = {"checkpoint": str(checkpoint.relative_to(REPO)), "checkpoint_sha256": sha(checkpoint),
                 "rows": rows, "summary": D.summarize(rows)}
        if arm.endswith("t1"):
            value["time_binding_contract"] = common_time_audit(model, sequences, traces, selected, device)
            model.query_bin_shift = 2
            shifted = replay(model, mano, sequences, traces, selected, device)
            model.query_bin_shift = 0
            value["time_shift_2_control"] = {"descriptive_only": True, "same_weights": True,
                                            "trajectory_valid": all(row["trajectory_valid"] for row in shifted),
                                            "rows": shifted, "summary": D.summarize(shifted)}
        payload["arms"][arm] = value
        payload["elapsed_s"] = time.monotonic() - started
        save_payload(output, payload)
        print(f"Saved seed{args.seed} {arm}: {output}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
