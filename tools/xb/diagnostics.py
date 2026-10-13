#!/usr/bin/env python3
"""Read-only paired S37 residual diagnostics; writes mechanism JSON, never a main row.

The reference C0 tracker generates actual erroneous priors over the two zgz sequences.
All arms then replay the same ten event windows from each common initial state, with
their own predictions generating subsequent associations.  Truth only initializes a
diagnostic clean/perturbed start and evaluates predictions; packet.target stays zero.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))
from config import load_config  # noqa: E402
from model import MNISTModel  # noqa: E402
from mano_layer import ManoLayer  # noqa: E402
from semkine import eval_track as ET  # noqa: E402
from semkine.dataset import sequences_for_split, splits_manifest  # noqa: E402
from semkine.xyz_residual import residual_permutation  # noqa: E402

RECOVERY_RA_MM = 10.0
RECOVERY_ABS_MM = 20.0


def checkpoint_from_run(path):
    path = Path(path)
    if path.is_file():
        return path
    if (path / "last.ckpt").exists():
        return path / "last.ckpt"
    raise FileNotFoundError(f"no last.ckpt in {path}")


def load_arm(path, device, config_path=None):
    checkpoint = checkpoint_from_run(path)
    config_path = config_path or checkpoint.parent / "config_resolved.yaml"
    cfg = load_config(config_path)
    model = MNISTModel.load_from_checkpoint(str(checkpoint), cfg=cfg, map_location="cpu").to(device).eval()
    if getattr(model, "xyz_assoc_prev", "self") != "self":
        raise ValueError("diagnostics require deployable own-state association")
    return model, cfg, checkpoint


def load_sequences(cfg):
    root = Path(cfg["DATA"]["ROOT"])
    result = {}
    for seq, directory in sequences_for_split(root, "val_core", splits_manifest(cfg)):
        ev, offsets, aux, gt = ET.load_sequence(root, directory, seq)
        tsub_file = root / directory / f"{seq}_tsub.npy"
        result[seq] = {"events": ev, "offsets": offsets, "aux": aux, "gt": gt,
                       "tsub": np.load(tsub_file, mmap_mode="r") if tsub_file.exists() else None}
    if set(result) != {"zgz_local", "zgz_global"}:
        raise RuntimeError(f"expected the fixed two zgz sequences, got {list(result)}")
    return result


def context(data, device):
    beta = torch.as_tensor(np.array(data["aux"]["betas"]), device=device, dtype=torch.float32).view(1, -1)
    camera = torch.as_tensor(np.array(data["aux"]["camera_K"]), device=device, dtype=torch.float32).view(1, 3, 3)
    return beta, camera


@torch.no_grad()
def reference_traces(model, cfg, sequences, device):
    """Collect the unaltered W50 own-state loop, including actual priors and residual mass."""
    traces = {}
    rng = np.random.default_rng(0)
    for seq, data in sequences.items():
        beta, camera = context(data, device)
        model.set_hand_context(beta, camera)
        rows = []
        for rid, (a, b) in enumerate(np.asarray(data["aux"]["valid_runs_ms"], np.int64).reshape(-1, 2)):
            prev = torch.from_numpy(data["gt"][a].copy() + ET.sample_init_noise(cfg, rng, 1.0)).view(1, -1).to(device)
            previous_end = int(a)
            for end in np.arange(a + 49, b, 50, dtype=np.int64):
                ev5 = ET._window_events(data["events"], data["offsets"], data["tsub"], int(end), 50)
                pred = model.forward_packet(ET.make_eval_packet(ev5, prev, beta, camera, 50, device))
                rows.append({"run_id": rid, "end_ms": int(end), "previous_end_ms": previous_end,
                             "prev": prev[0].cpu().numpy(), "pred": pred[0].cpu().numpy(),
                             "n_raw_events": len(ev5),
                             "residual_mass": float(model.route_stats.get("xyz_resid_mass", 0.0))})
                prev, previous_end = pred, int(end)
        if not rows:
            raise RuntimeError(f"no W50 reference packets in {seq}")
        traces[seq] = rows
        print(f"reference trace {seq}: {len(rows)} steps", flush=True)
    return traces


def select_clips(traces, steps, per_sequence):
    """Uniform chronological selection, fixed before evaluating any candidate."""
    selected = {}
    for seq, rows in traces.items():
        possible = [i for i in range(len(rows) - steps + 1)
                    if rows[i]["run_id"] == rows[i + steps - 1]["run_id"]]
        if not possible:
            raise RuntimeError(f"no {steps}-step valid clip in {seq}")
        chosen = np.linspace(0, len(possible) - 1, min(per_sequence, len(possible)), dtype=np.int64)
        selected[seq] = [possible[int(i)] for i in np.unique(chosen)]
    return selected


def perturbations(shape=(51,), seed=20261011):
    rng = np.random.default_rng(seed)
    result = {}
    for name, t, r, pose in (("small", .005, .05, .05), ("large", .05, .3, .3)):
        value = np.zeros(shape, np.float32)
        value[:3] = rng.normal(size=3) * t
        value[3:6] = rng.normal(size=3) * r
        value[6:] = rng.normal(size=45) * pose
        result[name] = value
    return result


@torch.no_grad()
def measure(mano, params, truth, beta, device):
    predicted_joints, _ = ET.mano_fk(mano, np.asarray(params, np.float32), beta, device)
    true_joints, _ = ET.mano_fk(mano, np.asarray(truth, np.float32), beta, device)
    error_abs = np.linalg.norm(predicted_joints - true_joints, axis=-1).mean(-1) * 1000
    error_ra = np.linalg.norm(ET.root_align(predicted_joints) - ET.root_align(true_joints), axis=-1).mean(-1) * 1000
    return error_abs, error_ra, predicted_joints, true_joints


def motion_response(pred_joints, gt_joints):
    """Velocity gain/direction/lag diagnostic in joint space, with matched 50 ms steps."""
    pred_joints = ET.root_align(pred_joints)
    gt_joints = ET.root_align(gt_joints)
    pv, gv = np.diff(pred_joints, axis=0), np.diff(gt_joints, axis=0)
    energy = float(np.square(gv).sum())
    gain = float((pv * gv).sum() / max(energy, 1e-20))
    pred_energy = float(np.square(pv).sum())
    cosine = float((pv * gv).sum() / max(np.sqrt(energy * pred_energy), 1e-20))
    # Compare signed velocity correlation at -2..+2 steps; positive lag means prediction
    # at t+lag corresponds to GT at t. The selected lag is a diagnostic, never a score.
    lag_scores = {}
    for lag in (-2, -1, 0, 1, 2):
        if lag > 0:
            p, g = pv[lag:], gv[:-lag]
        elif lag < 0:
            p, g = pv[:lag], gv[-lag:]
        else:
            p, g = pv, gv
        if len(p):
            lag_scores[str(lag)] = float((p * g).sum() / max(np.sqrt(np.square(p).sum() * np.square(g).sum()), 1e-20))
    return {"joint_velocity_gain": gain, "joint_velocity_cosine": cosine,
            "predicted_motion_mm_per_step": float(np.linalg.norm(pv, axis=-1).mean() * 1000),
            "true_motion_mm_per_step": float(np.linalg.norm(gv, axis=-1).mean() * 1000),
            "velocity_lag_steps": int(max(lag_scores, key=lag_scores.get)), "lag_correlations": lag_scores}


def initial_correction(pred_joints, true_joints):
    result = {}
    for label, transform in (("absolute", lambda value: value), ("root_relative", ET.root_align)):
        prediction, truth = transform(pred_joints), transform(true_joints)
        update = prediction[1] - prediction[0]
        needed = truth[1] - prediction[0]
        numerator = float((update * needed).sum())
        energy = float(np.square(needed).sum())
        result[label] = {"cosine": numerator / max(np.sqrt(float(np.square(update).sum()) * energy), 1e-20),
                         "gain": numerator / max(energy, 1e-20),
                         "update_norm_mm": float(np.linalg.norm(update, axis=-1).mean() * 1000),
                         "needed_norm_mm": float(np.linalg.norm(needed, axis=-1).mean() * 1000)}
    return result


def first_hit(errors, threshold):
    hits = np.flatnonzero(np.asarray(errors)[1:] <= threshold)
    return int(hits[0] + 1) if len(hits) else None


def exchange_r1_hook(mod, inputs):
    """Scramble only unit-weight residual content; leave xy, quality and strata fixed."""
    token, residual, valid = inputs
    live = (residual == 1) & valid
    groups = (token[..., 3].float() * 2).floor().long().clamp(0, 1) * 2 + (token[..., 2] > 0).long()
    permutation = residual_permutation(live, mod.permutation_seed, groups)
    changed_token = token.clone()
    changed_token[..., 2:] = token[..., 2:].gather(1, permutation.unsqueeze(-1).expand(-1, -1, 5))
    return changed_token, residual, valid


@torch.no_grad()
def replay_clips(model, mano, sequences, traces, selected, device, steps):
    result = []
    for seq_index, (seq, starts) in enumerate(selected.items()):
        data, rows = sequences[seq], traces[seq]
        beta, camera = context(data, device)
        model.set_hand_context(beta, camera)
        for clip_index, begin in enumerate(starts):
            segment = rows[begin:begin + steps]
            previous_end = segment[0]["previous_end_ms"]
            true_initial = data["gt"][previous_end].copy()
            noises = perturbations(seed=20261011 + 10000 * seq_index + clip_index)
            initials = {"natural": segment[0]["prev"].copy(), "clean": true_initial,
                        "small": true_initial + noises["small"], "large": true_initial + noises["large"]}
            events = [ET._window_events(data["events"], data["offsets"], data["tsub"], row["end_ms"], 50)
                      for row in segment]
            truth = np.stack([true_initial] + [data["gt"][row["end_ms"]] for row in segment])
            for condition, initial in initials.items():
                prev = torch.from_numpy(initial.copy()).view(1, -1).to(device)
                predictions, stats = [initial.copy()], []
                for ev5 in events:
                    prev = model.forward_packet(ET.make_eval_packet(ev5, prev, beta, camera, 50, device))
                    predictions.append(prev[0].cpu().numpy())
                    stats.append({key: float(value) for key, value in model.route_stats.items()})
                errors_abs, errors_ra, pj, gj = measure(mano, predictions, truth, beta, device)
                held_abs, held_ra, _, _ = measure(mano, np.repeat(initial[None], steps + 1, axis=0), truth, beta, device)
                result.append({"seq": seq, "clip_id": clip_index, "condition": condition,
                               "run_id": segment[0]["run_id"], "first_end_ms": segment[0]["end_ms"],
                               "previous_end_ms": previous_end, "n_raw_events": [len(ev) for ev in events],
                               "reference_residual_mass": segment[0]["residual_mass"],
                               "initial_state_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
                               "mpjpe_abs_mm": errors_abs.tolist(), "mpjpe_ra_mm": errors_ra.tolist(),
                               "held_state_mpjpe_abs_mm": held_abs.tolist(), "held_state_mpjpe_ra_mm": held_ra.tolist(),
                               "correction_gain_ra_mm": (held_ra - errors_ra).tolist(),
                               "motion": motion_response(pj, gj), "route_stats": stats,
                               "initial_correction": initial_correction(pj, gj),
                               "first_hit_ra_step": first_hit(errors_ra, RECOVERY_RA_MM),
                               "first_hit_abs_step": first_hit(errors_abs, RECOVERY_ABS_MM),
                               "first_half_initial_ra_step": first_hit(errors_ra, errors_ra[0] * .5)
                               if errors_ra[0] > 0 else None})
        print(f"replayed {seq}: {len(starts)} clips x 4 initial conditions", flush=True)
    return result


def summarize(rows):
    result = {}
    for seq in ("all", "zgz_local", "zgz_global"):
        result[seq] = {}
        for condition in ("natural", "clean", "small", "large"):
            chosen = [row for row in rows if row["condition"] == condition and (seq == "all" or row["seq"] == seq)]
            if not chosen:
                continue
            error_ra = np.asarray([row["mpjpe_ra_mm"] for row in chosen])
            error_abs = np.asarray([row["mpjpe_abs_mm"] for row in chosen])
            gain_ra = np.asarray([row["correction_gain_ra_mm"] for row in chosen])
            result[seq][condition] = {"n_clips": len(chosen), "mpjpe_ra_mm_by_step": error_ra.mean(0).tolist(),
                                      "mpjpe_abs_mm_by_step": error_abs.mean(0).tolist(),
                                      "correction_gain_ra_mm_by_step": gain_ra.mean(0).tolist(),
                                      "recovery_auc_ra_mm": float(error_ra[:, 1:].mean()),
                                      "first_step_mpjpe_ra_mm": float(error_ra[:, 1].mean()),
                                      "first_step_mpjpe_abs_mm": float(error_abs[:, 1].mean()),
                                      "first_step_correction_gain_ra_mm": float(gain_ra[:, 1].mean()),
                                      "recovery_ra_success_fraction": float(np.mean([row["first_hit_ra_step"] is not None for row in chosen])),
                                      "recovery_abs_success_fraction": float(np.mean([row["first_hit_abs_step"] is not None for row in chosen])),
                                      "recovery_ra_capped_time_ms": float(np.mean([50 * (row["first_hit_ra_step"] or len(row["mpjpe_ra_mm"])) for row in chosen])),
                                      "recovery_abs_capped_time_ms": float(np.mean([50 * (row["first_hit_abs_step"] or len(row["mpjpe_abs_mm"])) for row in chosen])),
                                      "initial_correction": {space: {key: float(np.mean([row["initial_correction"][space][key] for row in chosen]))
                                                                     for key in ("cosine", "gain", "update_norm_mm", "needed_norm_mm")}
                                                             for space in ("absolute", "root_relative")},
                                      "motion": {key: float(np.mean([row["motion"][key] for row in chosen]))
                                                 for key in ("joint_velocity_gain", "joint_velocity_cosine",
                                                             "predicted_motion_mm_per_step", "true_motion_mm_per_step",
                                                             "velocity_lag_steps")}}
    return result


@torch.no_grad()
def audit_control(model, sequences, traces, selected, device):
    """Switch only B/C content exchange at identical inputs/weights; verify a real effect.

    Common slots, destination quality/occupancy metadata and group membership must stay
    unchanged.  The audit discloses the fraction of *eligible* events exchanged and the
    resulting local feature/readout difference, rather than counting permutation alone.
    """
    branch = getattr(model.xyz, "residual", None)
    if branch is None:
        return {"applicable": False, "reason": "control audit applies to B/C residual branches"}
    original_mode = branch.mode
    original_keep = getattr(model, "xyz_keep_aux", False)
    model.xyz_keep_aux = True
    rows = []
    try:
        for seq, indices in selected.items():
            data = sequences[seq]
            beta, camera = context(data, device)
            model.set_hand_context(beta, camera)
            for clip_index, index in enumerate(indices):
                source = traces[seq][index]
                prev = torch.from_numpy(source["prev"].copy()).view(1, -1).to(device)
                ev5 = ET._window_events(data["events"], data["offsets"], data["tsub"], source["end_ms"], 50)
                if not len(ev5):
                    continue
                packet = ET.make_eval_packet(ev5, prev, beta, camera, 50, device)
                branch.mode = "B"
                prediction_b = model.forward_packet(packet)
                captured_b = model.xyz_last
                common_slots_b = captured_b["slots"].clone()
                extra_b, aux_b = branch(captured_b["tok"], captured_b["r"], captured_b["mask"])
                branch.mode = "C"
                prediction_c = model.forward_packet(packet)
                captured_c = model.xyz_last
                extra_c, aux_c = branch(captured_c["tok"], captured_c["r"], captured_c["mask"])
                branch.mode = "D"
                prediction_d = model.forward_packet(packet)
                captured_d = model.xyz_last
                extra_d, aux_d = branch(captured_d["tok"], captured_d["r"], captured_d["mask"])
                unchanged = {"common_slots_equal": torch.equal(common_slots_b, captured_c["slots"]),
                             "residual_mass_equal": torch.equal(aux_b["residual_mass"], aux_c["residual_mass"]),
                             "residual_count_equal": torch.equal(aux_b["residual_count"], aux_c["residual_count"]),
                             "metadata_equal": torch.equal(aux_b["residual_metadata"], aux_c["residual_metadata"]),
                             "tok_equal": torch.equal(captured_b["tok"], captured_c["tok"]),
                             "residual_weights_equal": torch.equal(captured_b["r"], captured_c["r"])}
                unchanged.update({"d_common_slots_equal": torch.equal(common_slots_b, captured_d["slots"]),
                                  "d_metadata_equal": torch.equal(aux_b["residual_metadata"], aux_d["residual_metadata"]),
                                  "d_mass_equal": torch.equal(aux_b["residual_mass"], aux_d["residual_mass"]),
                                  "d_count_equal": torch.equal(aux_b["residual_count"], aux_d["residual_count"]),
                                  "d_content_zero": bool((aux_d["residual_content"] == 0).all())})
                if not all(unchanged.values()):
                    raise AssertionError(f"B/C audit changed common input or metadata: {unchanged}")
                row = {"seq": seq, "clip_id": clip_index, "end_ms": source["end_ms"], **unchanged,
                       "extra_mean_abs_change": float((extra_b - extra_c).abs().mean()),
                       "extra_max_abs_change": float((extra_b - extra_c).abs().max()),
                       "prediction_mean_abs_change": float((prediction_b - prediction_c).abs().mean()),
                       "prediction_max_abs_change": float((prediction_b - prediction_c).abs().max()),
                       "content_mean_abs_change": float((aux_b["residual_content"] - aux_c["residual_content"]).abs().mean()),
                       "d_extra_max_abs_change": float((extra_b - extra_d).abs().max()),
                       "d_prediction_max_abs_change": float((prediction_b - prediction_d).abs().max())}
                for key, value in aux_c.items():
                    if value.numel() == 1:
                        row[key] = float(value)
                permutation = aux_c["residual_permutation"]
                live = (captured_c["r"] > 0) & captured_c["mask"]
                row["permutation_preserves_stratum"] = bool(((aux_c["residual_group"].gather(1, permutation)
                                                                == aux_c["residual_group"]) | ~live).all())
                if not row["permutation_preserves_stratum"]:
                    raise AssertionError("C exchange crossed a quality/time/polarity stratum")
                # A receives the same fixed-r=1 content exchange but its global summary
                # must be invariant to position binding, up to accumulation roundoff.
                branch.mode = "A"
                prediction_a = model.forward_packet(packet)
                a_content = model.xyz_last["residual_content"].clone()

                handle = branch.register_forward_pre_hook(exchange_r1_hook)
                try:
                    prediction_a_control = model.forward_packet(packet)
                    a_control_content = model.xyz_last["residual_content"]
                finally:
                    handle.remove()
                row["a_r1_content_max_abs_change"] = float((a_content - a_control_content).abs().max())
                row["a_r1_prediction_max_abs_change"] = float((prediction_a - prediction_a_control).abs().max())
                row["a_r1_global_invariant"] = row["a_r1_content_max_abs_change"] <= 1e-6
                rows.append(row)
    finally:
        branch.mode = original_mode
        model.xyz_keep_aux = original_keep
    counts = {"audited_packets": len(rows),
              "packets_with_content_change": sum(row["content_mean_abs_change"] > 0 for row in rows),
              "packets_with_extra_change": sum(row["extra_max_abs_change"] > 0 for row in rows),
              "packets_with_prediction_change": sum(row["prediction_max_abs_change"] > 0 for row in rows),
              "packets_with_d_prediction_change": sum(row["d_prediction_max_abs_change"] > 0 for row in rows),
              "a_global_control_pass": all(row["a_r1_global_invariant"] for row in rows),
              "all_common_invariants_pass": all(row[key] for row in rows for key in
                                                ("common_slots_equal", "residual_mass_equal", "residual_count_equal",
                                                 "metadata_equal", "tok_equal", "residual_weights_equal",
                                                 "permutation_preserves_stratum", "d_common_slots_equal", "d_metadata_equal",
                                                 "d_mass_equal", "d_count_equal", "d_content_zero"))}
    scalar_keys = [key for key, value in rows[0].items() if key.startswith("residual_") and isinstance(value, float)] if rows else []
    averages = {key: float(np.mean([row[key] for row in rows])) for key in scalar_keys}
    return {"applicable": True, "original_mode": original_mode, "summary": {**counts, **averages}, "rows": rows,
            "non_noop_pass": counts["packets_with_content_change"] > 0 and counts["packets_with_extra_change"] > 0,
            "readout_uses_binding": counts["packets_with_prediction_change"] > 0}


@torch.no_grad()
def inference_controls(model, mano, cfg, device):
    """Full own-loop same-weight controls: B -> C/D; A -> r=1 exchange.

    These inference perturbations support a sensitivity argument. Retrained C/D arms
    remain the main mechanism comparisons, since exchanging inputs can change a learned
    feature distribution. No weights are changed and no truth feeds associations.
    """
    branch = getattr(model.xyz, "residual", None)
    if branch is None or branch.mode not in ("A", "B"):
        return {"applicable": False}
    original_mode = branch.mode
    original_keep = getattr(model, "xyz_keep_aux", False)
    model.xyz_keep_aux = False
    root = Path(cfg["DATA"]["ROOT"])
    sequences = sequences_for_split(root, "val_core", splits_manifest(cfg))
    modes = ("B", "C", "D") if original_mode == "B" else ("A", "A_r1_exchange")
    result = {}
    try:
        for mode in modes:
            branch.mode = "A" if mode == "A_r1_exchange" else mode
            handle = branch.register_forward_pre_hook(exchange_r1_hook) if mode == "A_r1_exchange" else None
            try:
                rng = np.random.default_rng(0)
                scores = [ET.track_sequence(model, mano, cfg, root, directory, seq, 50, device, rng, 1.0, 1.0, None)
                          for seq, directory in sequences]
            finally:
                if handle is not None:
                    handle.remove()
            scores = [row for row in scores if row]
            aggregate = {}
            for group in ("all", "local", "global"):
                chosen = [row for row in scores if group == "all" or ("_local" in row["seq"]) == (group == "local")]
                total_frames = sum(row["n_frames"] for row in chosen)
                aggregate[group] = {key: sum(row[key] * row["n_frames"] for row in chosen) / max(total_frames, 1)
                                    for key in ("mpjpe_ra_mm", "mpjpe_abs_mm", "mpvpe_ra_mm", "mpvpe_abs_mm")}
                aggregate[group]["n_frames"] = total_frames
            result[mode] = {"per_sequence": scores, "aggregate": aggregate}
            print(f"same-weight inference control {original_mode}->{mode}: complete", flush=True)
    finally:
        branch.mode = original_mode
        model.xyz_keep_aux = original_keep
    return {"applicable": True, "trained_mode": original_mode, "same_weights": True, "controls": result,
            "interpretation": "sensitivity only; retrained C/D comparisons test the mechanism without input-distribution confound"}


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(type(value))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", required=True, help="fixed C0 run directory or checkpoint")
    ap.add_argument("--reference-config")
    ap.add_argument("--arm", action="append", default=[], help="display_name=run_directory (repeatable)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--clips-per-seq", type=int, default=24)
    ap.add_argument("--trace-cache", help="reuse/save the fixed reference trace npz")
    ap.add_argument("--skip-inference-controls", action="store_true", help="skip the additional full A/B same-weight control loops")
    args = ap.parse_args()
    device = torch.device(args.device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    reference, cfg, reference_ckpt = load_arm(args.reference, device, args.reference_config)
    sequences = load_sequences(cfg)
    trace_path = Path(args.trace_cache) if args.trace_cache else output.with_suffix(".reference.pt")
    if trace_path.exists():
        cached = torch.load(trace_path, map_location="cpu")
        if cached["checkpoint"] != str(reference_ckpt.resolve()):
            raise ValueError("trace cache belongs to another reference checkpoint")
        traces = cached["traces"]
    else:
        traces = reference_traces(reference, cfg, sequences, device)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"checkpoint": str(reference_ckpt.resolve()), "traces": traces}, trace_path)
    selected = select_clips(traces, args.steps, args.clips_per_seq)
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    arms = [("C0", args.reference)] + [value.split("=", 1) for value in args.arm]
    payload = {"protocol": {"window_ms": 50, "step_ms": 50, "steps": args.steps,
                            "reference_checkpoint": str(reference_ckpt.resolve()), "selected_trace_indices": selected,
                            "clip_selection": "uniform chronological legal starts; fixed before candidate evaluation",
                            "initial_conditions": ["reference own-loop prior", "correct prior", "small perturbation", "large perturbation"],
                            "ground_truth_input": "run initialization or common perturbation start only; packet target zero",
                            "metrics_destination": "mechanism diagnostics only; formal results use make_s36_row"},
               "arms": {}}
    del reference
    if device.type == "cuda":
        torch.cuda.empty_cache()
    for name, directory in arms:
        model, arm_cfg, checkpoint = load_arm(directory, device, args.reference_config if name == "C0" else None)
        rows = replay_clips(model, mano, sequences, traces, selected, device, args.steps)
        audit = audit_control(model, sequences, traces, selected, device)
        full_controls = {"skipped": True} if args.skip_inference_controls else inference_controls(model, mano, arm_cfg, device)
        payload["arms"][name] = {"checkpoint": str(checkpoint.resolve()), "rows": rows,
                                 "summary": summarize(rows), "control_audit": audit,
                                 "same_weight_inference_controls": full_controls,
                                 "common_prior_one_step": [row for row in rows if row["condition"] == "natural"]}
        payload["elapsed_s"] = time.monotonic() - started
        output.write_text(json.dumps(payload, default=json_value, indent=2, allow_nan=False))
        print(f"diagnostics checkpointed {name}: {output}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
