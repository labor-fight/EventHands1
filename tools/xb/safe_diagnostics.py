#!/usr/bin/env python3
"""Run frozen XYZ diagnostics with explicit numerical failures and an OOD addendum.

The original diagnostics.py and every candidate checkpoint remain unchanged. Numerical
failures retain all frames/clips/seeds and invalidate their original aggregates; they
are serialized as null with explicit failure paths, never averaged over a finite subset.
The additional B r=1-only exchange is a non-confirmatory sensitivity probe.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))
from tools.xb import diagnostics as D  # noqa: E402
from semkine.xyz_residual import residual_permutation  # noqa: E402

ORIGINAL_DUMPS = json.dumps
ORIGINAL_TRACK = D.ET.track_sequence
ORIGINAL_LOAD = D.load_sequences
ORIGINAL_SELECT = D.select_clips
ORIGINAL_CONTROLS = D.inference_controls
ORIGINAL_AUDIT = D.audit_control
CONTEXT = {}


def deterministic_audit_call(function, *args, **kwargs):
    """Keep strict bitwise checks while making only common-input audits deterministic."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        result = function(*args, **kwargs)
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    result["audit_runtime"] = {
        "deterministic_algorithms": True, "warn_only": False,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "prior_deterministic_algorithms": enabled, "prior_warn_only": warn_only,
        "restored_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "restored_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "original_strict_checks_unchanged": True,
        "runtime_fix_record": "outputs/xb/runtime_audit_fix.json"}
    return result


def deterministic_common_input_audit(*args, **kwargs):
    return deterministic_audit_call(ORIGINAL_AUDIT, *args, **kwargs)


def exact_checkpoint_from_run(path):
    """Select the registered optimizer step, independently of Lightning last copies."""
    path = Path(path)
    directory = path.parent if path.is_file() else path
    steps = 1500 if directory.name.startswith("xb_p") else 6000
    matches = list(directory.glob(f"*-step={steps}.ckpt"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one exact step={steps} checkpoint in {directory}, got {matches}")
    selected = matches[0]
    if path.is_file() and path.resolve() != selected.resolve():
        raise RuntimeError(f"explicit checkpoint is not the registered endpoint: {path}")
    return selected


def _sanitize_numeric(value, path="$", annotate=True):
    """Return a lossless-structure JSON value plus every original invalid scalar path."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None, [path], []
    if isinstance(value, dict):
        cleaned, invalid, known_invalid = {}, [], []
        for key, item in value.items():
            cleaned[key], paths, known = _sanitize_numeric(item, f"{path}.{key}", annotate)
            invalid.extend(paths)
            known_invalid.extend(known)
            if (str(key).endswith("numeric_valid") and item is False) or (key == "numeric_status" and item == "FAIL"):
                known_invalid.append(f"{path}.{key}")
        if annotate:
            valid = not invalid and not known_invalid
            cleaned.update({"numeric_valid": valid, "numeric_status": "PASS" if valid else "FAIL",
                            "first_nonfinite": invalid[0] if invalid else None,
                            "invalid_numeric_count": len(invalid), "invalid_numeric_paths": invalid,
                            "invalid_known_count": len(known_invalid), "invalid_known_paths": known_invalid})
        return cleaned, invalid, known_invalid
    if isinstance(value, (list, tuple)):
        cleaned, invalid, known_invalid = [], [], []
        for index, item in enumerate(value):
            item, paths, known = _sanitize_numeric(item, f"{path}[{index}]", annotate)
            cleaned.append(item)
            invalid.extend(paths)
            known_invalid.extend(known)
        return cleaned, invalid, known_invalid
    return value, [], []


def sanitize_numeric(value, path="$", annotate=True):
    cleaned, invalid, _ = _sanitize_numeric(value, path, annotate)
    return cleaned, invalid


def safe_dumps(value, *args, **kwargs):
    cleaned, _ = sanitize_numeric(value)
    kwargs["allow_nan"] = False
    return ORIGINAL_DUMPS(cleaned, *args, **kwargs)


def capture_load(cfg):
    result = ORIGINAL_LOAD(cfg)
    CONTEXT["sequences"] = result
    return result


def capture_selection(traces, steps, per_sequence):
    selected = ORIGINAL_SELECT(traces, steps, per_sequence)
    CONTEXT.update({"traces": traces, "selected": selected})
    return selected


def audited_track(*args, **kwargs):
    """Observe every frame of the unchanged original evaluator, including its NaNs."""
    from inspect import signature
    bound = signature(ORIGINAL_TRACK).bind(*args, **kwargs)
    bound.apply_defaults()
    params = bound.arguments
    seq, root, directory = params["seq"], Path(params["root"]), params["legacy_dir"]
    aux = np.load(root / directory / f"{seq}_aux.npz", allow_pickle=True)
    step_ms = int(params["step_ms"])
    window_ms = int(params.get("window_ms") or step_ms)
    runs = [(int(a), int(b)) for a, b in np.asarray(aux["valid_runs_ms"], np.int64).reshape(-1, 2)
            if len(np.arange(a + max(step_ms, window_ms) - 1, b, step_ms))]
    original_hook = params.get("state_hook")
    audit = {"run_index": -1, "run_step": 0, "frames": []}

    def hook(stage, prev, prediction):
        if stage == "run_start":
            audit["run_index"] += 1
            audit["run_step"] = 0
            if original_hook is not None:
                original_hook(stage, prev, prediction)
            return None
        if original_hook is not None:
            prediction = original_hook(stage, prev, prediction)
        pred_valid = bool(torch.isfinite(prediction).all())
        prev_valid = bool(torch.isfinite(prev).all())
        run_start = runs[audit["run_index"]][0]
        end = run_start + max(step_ms, window_ms) - 1 + audit["run_step"] * step_ms
        audit["run_step"] += 1
        audit["frames"].append({"frame_index": len(audit["frames"]) + 1,
                                "run_index": audit["run_index"], "end_ms": end,
                                "prediction_numeric_valid": pred_valid, "prior_numeric_valid": prev_valid,
                                "prediction_nonfinite_coordinate_count": int((~torch.isfinite(prediction)).sum()),
                                "prior_norm_fp64": float(prev.double().norm()) if prev_valid else None,
                                "prediction_norm_fp64": float(prediction.double().norm()) if pred_valid else None})
        return prediction

    params["state_hook"] = hook
    result = ORIGINAL_TRACK(**params)
    if result is not None:
        bad = [row for row in audit["frames"] if not row["prediction_numeric_valid"]]
        result["trajectory_numeric_audit"] = {"n_frames": len(audit["frames"]),
                                               "numeric_valid": not bad,
                                               "first_nonfinite_frame": bad[0] if bad else None,
                                               "invalid_prediction_frame_count": len(bad),
                                               "all_frames_retained": True, "frames": audit["frames"],
                                               "aggregation": "original evaluator over all frames, including failures"}
        if len(audit["frames"]) != result["n_frames"]:
            raise AssertionError("output handler changed the evaluated frame count")
    return result


@torch.no_grad()
def common_frame_r1_probe(model, device):
    """Same trained B weights, same common C0 priors/events; exchange only r=1 content."""
    sequences, traces, selected = CONTEXT["sequences"], CONTEXT["traces"], CONTEXT["selected"]
    old_keep = getattr(model, "xyz_keep_aux", False)
    model.xyz_keep_aux = True
    rows = []
    try:
        for seq, indices in selected.items():
            data = sequences[seq]
            beta, camera = D.context(data, device)
            model.set_hand_context(beta, camera)
            for clip_id, index in enumerate(indices):
                source = traces[seq][index]
                prev = torch.from_numpy(source["prev"].copy()).view(1, -1).to(device)
                ev5 = D.ET._window_events(data["events"], data["offsets"], data["tsub"], source["end_ms"], 50)
                if not len(ev5):
                    rows.append({"seq": seq, "clip_id": clip_id, "end_ms": source["end_ms"],
                                 "n_raw_events": 0, "empty_packet": True, "content_change": 0.0,
                                 "extra_change": 0.0, "prediction_change": 0.0})
                    continue
                packet = D.ET.make_eval_packet(ev5, prev, beta, camera, 50, device)
                baseline = model.forward_packet(packet)
                original = {key: value.clone() if torch.is_tensor(value) else value for key, value in model.xyz_last.items()}
                handle = model.xyz.residual.register_forward_pre_hook(D.exchange_r1_hook)
                try:
                    exchanged = model.forward_packet(packet)
                    control = model.xyz_last
                finally:
                    handle.remove()
                token, residual, valid = original["tok"], original["r"], original["mask"]
                r1 = (residual == 1) & valid
                groups = (token[..., 3].float() * 2).floor().long().clamp(0, 1) * 2 + (token[..., 2] > 0).long()
                permutation = residual_permutation(r1, model.xyz.residual.permutation_seed, groups)
                identity = torch.arange(token.shape[1], device=device).expand_as(permutation)
                changed = (permutation != identity) & r1
                cell = original["residual_cell"]
                crosscell = changed & (cell.gather(1, permutation) != cell)
                denominator = r1.sum().clamp_min(1).float()
                invariants = {"legacy_slots_equal": torch.equal(original["slots"], control["slots"]),
                              "xy_equal": torch.equal(original["tok"][..., :2], control["tok"][..., :2]),
                              "all_residual_weights_equal": torch.equal(original["r"], control["r"]),
                              "metadata_equal": torch.equal(original["residual_metadata"], control["residual_metadata"]),
                              "mass_equal": torch.equal(original["residual_mass"], control["residual_mass"]),
                              "count_equal": torch.equal(original["residual_count"], control["residual_count"]),
                              "r1_quality_preserved_exactly": bool(((residual.gather(1, permutation) == residual) | ~r1).all()),
                              "weak_content_unchanged": bool(((permutation == identity) | r1).all()),
                              "stratum_preserved_exactly": bool(((groups.gather(1, permutation) == groups) | ~r1).all())}
                if not all(invariants.values()):
                    raise AssertionError(f"r1 supplementary control changed an invariant: {invariants}")
                content_change = (original["residual_content"] - control["residual_content"]).abs()
                extra_change = (original["residual_extra"] - control["residual_extra"]).abs()
                difference = (baseline - exchanged).abs()
                rows.append({"seq": seq, "clip_id": clip_id, "end_ms": source["end_ms"],
                             "n_raw_events": len(ev5), "empty_packet": False, **invariants,
                             "r1_count": int(r1.sum()), "r1_swap_fraction": float(changed.sum() / denominator),
                             "r1_crosscell_fraction": float(crosscell.sum() / denominator),
                             "content_change": float(content_change.mean()), "content_max_change": float(content_change.max()),
                             "content_relative_change": float(content_change.mean() / original["residual_content"].abs().mean().clamp_min(1e-12)),
                             "extra_change": float(extra_change.mean()), "extra_max_change": float(extra_change.max()),
                             "extra_relative_change": float(extra_change.mean() / original["residual_extra"].abs().mean().clamp_min(1e-12)),
                             "prediction_change": float(difference.mean()), "prediction_max_change": float(difference.max()),
                             "translation_change_mm": float((baseline[:, :3] - exchanged[:, :3]).norm() * 1000),
                             "root_change_rad": float((baseline[:, 3:6] - exchanged[:, 3:6]).norm()),
                             "finger_change_rad": float((baseline[:, 6:] - exchanged[:, 6:]).norm())})
    finally:
        model.xyz_keep_aux = old_keep
    # Deliberately use every selected frame: any NaN invalidates its summary.
    summary = {"n_frames": len(rows), "n_empty_frames": sum(row["empty_packet"] for row in rows),
               "n_frames_with_content_change": sum(row["content_change"] > 0 for row in rows),
               "n_frames_with_extra_change": sum(row["extra_change"] > 0 for row in rows),
               "n_frames_with_prediction_change": sum(row["prediction_change"] > 0 for row in rows)}
    for key in ("content_change", "extra_change", "prediction_change"):
        summary[key] = float(np.mean([row[key] for row in rows])) if rows else None
    return {"non_confirmatory": True, "purpose": "isolate unit-quality content-position binding from weak-quality exchange",
            "all_selected_frames_retained": True, "rows": rows, "summary": summary}


@torch.no_grad()
def extended_controls(model, mano, cfg, device):
    result = ORIGINAL_CONTROLS(model, mano, cfg, device)
    branch = getattr(model.xyz, "residual", None)
    if branch is None or branch.mode != "B":
        return result
    old_keep = getattr(model, "xyz_keep_aux", False)
    model.xyz_keep_aux = False
    root = Path(cfg["DATA"]["ROOT"])
    sequences = D.sequences_for_split(root, "val_core", D.splits_manifest(cfg))
    handle = branch.register_forward_pre_hook(D.exchange_r1_hook)
    try:
        rng = np.random.default_rng(0)
        scores = [D.ET.track_sequence(model, mano, cfg, root, directory, seq, 50, device, rng, 1.0, 1.0, None)
                  for seq, directory in sequences]
    finally:
        handle.remove()
        model.xyz_keep_aux = old_keep
    scores = [row for row in scores if row is not None]
    aggregate = {}
    for group in ("all", "local", "global"):
        chosen = [row for row in scores if group == "all" or ("_local" in row["seq"]) == (group == "local")]
        total = sum(row["n_frames"] for row in chosen)
        aggregate[group] = {key: sum(row[key] * row["n_frames"] for row in chosen) / max(total, 1)
                            for key in ("mpjpe_ra_mm", "mpjpe_abs_mm", "mpvpe_ra_mm", "mpvpe_abs_mm")}
        aggregate[group]["n_frames"] = total
    result["supplementary_r1_exchange"] = {"non_confirmatory": True, "trained_mode": "B", "same_weights": True,
                                           "changes": "new residual branch only: r=1 content exchange within time/polarity strata",
                                           "unchanged": "weak residual content, xy, every residual weight, matched branch, legacy slots",
                                           "per_sequence": scores, "aggregate": aggregate,
                                           "common_frame_probe": deterministic_audit_call(common_frame_r1_probe, model, device)}
    print("supplementary B->B_r1_exchange: complete (non-confirmatory)", flush=True)
    return result


def write_addendum():
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {"registration_date": "2026-10-11", "non_confirmatory_addendum": True,
               "wrapper": str(Path(__file__).resolve()), "wrapper_sha256": digest(Path(__file__)),
               "frozen_diagnostics_sha256": digest(Path(D.__file__)),
               "frozen_probe_sha256": digest(REPO / "tools/xb/probe.py"),
               "candidate_training_changed": False, "preregistration_snapshot_changed": False,
               "post_result_runtime_fix": {
                   "record": "outputs/xb/runtime_audit_fix.json",
                   "scope": "deterministic algorithms only inside the original common-input control audit and supplementary r1 common-frame audit",
                   "strict_equality_and_thresholds_changed": False,
                   "recovery_and_whole_loop_deterministic_flag_changed": False},
               "differences_from_frozen_implementation": [
                   "output-only NaN/Inf -> null with numeric_valid/status, first_nonfinite path and invalid path/count annotations",
                   "observe complete inference-control trajectories with state hooks; preserve all frames and failed whole-frame means",
                   "select the unique named registered endpoint (pilot step1500, full/reference step6000) instead of an ambiguous Lightning last copy",
                   "supplementary OOD B r=1-only content exchange with full own-loop and same-common-frame strength audit",
                   "post-result runtime repair: deterministic common-input audit, original strict checks retained, flags restored afterwards"],
               "failure_policy": "no clipped trajectory, removed clip/seed/frame, finite-subset averaging or replacement score",
               "json_schema_additions": {
                   "annotation_scope": "every dictionary, including top/arms/arm/row/summary/control/aggregate-group/frame",
                   "numeric_valid": "boolean: false for a nonfinite scalar anywhere below, or any propagated known-invalid numeric_valid/status",
                   "numeric_status": "PASS or FAIL; known-invalid is FAIL, never pending",
                   "first_nonfinite": "first original NaN/Inf scalar JSON path, or null",
                   "invalid_numeric_count": "count of original NaN/Inf scalar values in this subtree",
                   "invalid_numeric_paths": "all original NaN/Inf scalar JSON paths in this subtree",
                   "invalid_known_count": "count of explicit false *numeric_valid or FAIL numeric_status flags in this subtree",
                   "invalid_known_paths": "paths to original explicit numeric failure flags",
                   "nonfinite_scalar": "JSON null; original aggregates remain failed rather than recomputed",
                   "trajectory_numeric_audit": "per-sequence all-frame state validity, first_nonfinite_frame and invalid_prediction_frame_count",
                   "finalize_rule": "null accuracy/recovery/control metric or numeric_valid=false/numeric_status=FAIL -> FAIL; never pending",
                   "supplementary_control_path": "arms.<B>.same_weight_inference_controls.supplementary_r1_exchange",
                   "supplementary_control_confirmation": "non_confirmatory=true; exclude from original adoption gate"},
               "supplementary_control": {"name": "B_r1_exchange", "scope": "new residual branch only",
                                          "exchange": "r==1 events only, within original coarse time/polarity strata",
                                          "unchanged": ["xy", "all residual weights", "weak residual content", "legacy matched and residual slots"],
                                          "usage": "sensitivity evidence; excluded from preregistered confirmatory adoption gate"}}
    path = REPO / "outputs/xb/diagnostic_addendum.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ORIGINAL_DUMPS(payload, indent=2, allow_nan=False))


def main():
    if torch.cuda.is_initialized():
        raise RuntimeError("audit workspace must be configured before CUDA initialization")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    write_addendum()
    D.checkpoint_from_run = exact_checkpoint_from_run
    D.json.dumps = safe_dumps
    D.load_sequences = capture_load
    D.select_clips = capture_selection
    D.ET.track_sequence = audited_track
    D.inference_controls = extended_controls
    D.audit_control = deterministic_common_input_audit
    D.main()


if __name__ == "__main__":
    main()
