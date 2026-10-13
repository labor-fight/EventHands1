#!/usr/bin/env python3
"""CPU audit of frozen residual controls, warm start and S39 compatibility.

The first 64 samples of the common C0 training bank are fixed before this
audit.  No accuracy selection or training occurs here.  S39 comparisons load
the actual step-6000 weights to avoid a zero-initialized-head comparison.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
from torch.utils.data import default_collate

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))

from config import load_config  # noqa: E402
from model import MNISTModel  # noqa: E402
from semkine import eval_track as ET  # noqa: E402
from semkine.dataset import build_dataset, sequences_for_split, splits_manifest  # noqa: E402
from tools.xb.probe import cached_forward, warm_start  # noqa: E402


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def same_tensors(a, b):
    if set(a) != set(b):
        return {"equal": False, "bad": sorted(set(a) ^ set(b)), "max_abs_diff": None}
    bad, maximum = [], 0.0
    for key in a:
        x, y = a[key], b[key]
        if x is None or y is None:
            if x is not y:
                bad.append(key)
            continue
        if x.shape != y.shape or x.dtype != y.dtype or not torch.equal(x, y):
            bad.append(key)
            if x.shape == y.shape:
                maximum = max(maximum, float((x.double() - y.double()).abs().max()) if x.numel() else 0.)
    return {"equal": not bad, "bad": bad, "n_tensors": len(a), "max_abs_diff": maximum}


def legacy_cached_forward(model, batch):
    xyz, slots, prev = model.xyz, batch["slots"], batch["prev"]
    zs = torch.relu(xyz.slot_proj(slots.reshape(len(prev), -1, 1))).reshape(len(prev), -1)
    features = torch.cat([zs, prev[:, 3:51].to(zs.dtype)], -1)
    delta = xyz.fc2(torch.relu(xyz.fc1(features)))
    if model.zero_event_gate:
        delta = delta * batch["mask"].any(1).unsqueeze(-1).to(delta.dtype)
    return prev + delta


def bank_audit(bank_path, n_samples):
    bank = torch.load(bank_path, map_location="cpu")
    batch = {key: value[:n_samples] for key, value in bank.items() if torch.is_tensor(value)}
    cfg = load_config(str(REPO / "configs/xa/xa_c0_s3407.yaml"))
    checkpoint = REPO / "outputs/semkine/xa_c0_s3407/xa_c0_s3407-step=6000.ckpt"
    source = torch.load(checkpoint, map_location="cpu")["state_dict"]
    torch.manual_seed(3407)
    legacy = MNISTModel(cfg=copy.deepcopy(cfg)).float().cpu().eval()
    legacy.load_state_dict(source, strict=True)
    with torch.no_grad():
        reference = legacy_cached_forward(legacy, batch)
    models, initial, rng = {}, {}, {}
    for mode in "ABCD":
        conf = copy.deepcopy(cfg)
        conf["MODEL"].update(XYZ_RESIDUAL_MODE=mode, XYZ_RESIDUAL_SEED=19073)
        torch.manual_seed(3407)
        models[mode] = MNISTModel(cfg=conf).float().cpu().eval()
        initial[mode] = {k: v.detach().clone() for k, v in models[mode].state_dict().items()}
        rng[mode] = torch.random.get_rng_state().clone()
    init = {mode: same_tensors(initial["A"], initial[mode]) for mode in "BCD"}
    init_rng = all(torch.equal(rng["A"], rng[mode]) for mode in "BCD")
    arms, auxiliaries = {}, {}
    for mode, model in models.items():
        warm = warm_start(model, checkpoint)
        common = {name: value for name, value in source.items() if name != "xyz.fc1.weight"}
        copied = same_tensors(common, {name: model.state_dict()[name] for name in common})
        old_width = source["xyz.fc1.weight"].shape[1]
        expanded = model.xyz.fc1.weight.detach()
        prefix_equal = torch.equal(expanded[:, :old_width], source["xyz.fc1.weight"])
        extra_zero = not bool(expanded[:, old_width:].any())
        before = torch.random.get_rng_state().clone()
        with torch.no_grad():
            extra, aux = model.xyz.residual(batch["tok"], batch["r"], batch["mask"])
            prediction = cached_forward(model, batch)
        unchanged_rng = torch.equal(before, torch.random.get_rng_state())
        diff = float((reference - prediction).abs().max())
        parity = torch.allclose(reference, prediction, atol=1e-6, rtol=1e-5)
        moved_xy = batch["tok"].clone()
        moved_xy[..., :2] = (torch.floor(moved_xy[..., :2] * 4).clamp(0, 3) + .4) / 4
        with torch.no_grad():
            moved, _ = model.xyz.residual(moved_xy, batch["r"], batch["mask"])
        within_cell_invariant = torch.equal(extra, moved)
        specialized = {}
        if mode == "A":
            shuffled = batch["tok"].clone()
            shuffled[..., :2] = shuffled[..., :2].flip(1)
            with torch.no_grad():
                other, _ = model.xyz.residual(shuffled, batch["r"], batch["mask"])
            specialized["event_xy_invariant"] = torch.equal(extra, other)
            specialized["fixed_slots_distinct"] = not torch.equal(aux["residual_slot_features"],
                aux["residual_slot_features"][:, :1].expand(-1, 16, -1))
        if mode == "D":
            changed = batch["tok"].clone()
            changed[..., 2:] += 7
            with torch.no_grad():
                other, _ = model.xyz.residual(changed, batch["r"], batch["mask"])
            specialized["content_invariant"] = torch.equal(extra, other)
            specialized["content_identically_zero"] = not bool(aux["residual_content"].any())
        if mode == "C":
            permutation = aux["residual_permutation"]
            live = batch["mask"] & (batch["r"] > 0)
            changed = permutation != torch.arange(permutation.shape[1]).unsqueeze(0)
            specialized.update(
                same_strata=torch.equal(aux["residual_group"], aux["residual_group"].gather(1, permutation)),
                live_sources=bool((~live | live.gather(1, permutation)).all()),
                no_eligible_fixed_points=bool((~aux["residual_eligible"] | changed).all()),
                r1_quality_preserved=float(aux["residual_r1_quality_diff"]) == 0,
                measured_feature_change=float(aux["residual_feature_diff"]) > 0)
        scalars = {key: float(value) for key, value in aux.items() if value.ndim == 0}
        arms[mode] = {"params": sum(p.numel() for p in model.parameters()), "warm_common_copy": copied,
                      "warm_fc1_prefix_equal": prefix_equal, "warm_fc1_extra_zero": extra_zero,
                      "warm_prediction_close": bool(parity), "warm_prediction_max_abs_diff": diff,
                      "forward_global_rng_unchanged": unchanged_rng, "within_cell_raw_xy_invariant": within_cell_invariant,
                      "specialized_controls": specialized, "diagnostics": scalars,
                      "n_new_tensors": len(warm["new_tensors"])}
        auxiliaries[mode] = aux
    spatial_metadata = all(torch.equal(auxiliaries["B"]["residual_metadata"], auxiliaries[m]["residual_metadata"]) for m in "CD")
    spatial_counts = all(torch.equal(auxiliaries["B"]["residual_count"], auxiliaries[m]["residual_count"]) for m in "CD")
    spatial_mass = all(torch.equal(auxiliaries["B"]["residual_mass"], auxiliaries[m]["residual_mass"]) for m in "CD")
    spatial_cells = all(torch.equal(auxiliaries["B"]["residual_cell"], auxiliaries[m]["residual_cell"]) for m in "CD")
    fixed_centers = all(torch.equal(auxiliaries["A"]["residual_metadata"][..., -2:],
                                  auxiliaries[m]["residual_metadata"][..., -2:]) for m in "BCD")
    pass_arms = all(row["warm_common_copy"]["equal"] and row["warm_fc1_prefix_equal"] and row["warm_fc1_extra_zero"]
                    and row["warm_prediction_close"] and row["forward_global_rng_unchanged"] and row["within_cell_raw_xy_invariant"]
                    and all(row["specialized_controls"].values()) for row in arms.values())
    equal_params = len({row["params"] for row in arms.values()}) == 1
    return {"passed": all(v["equal"] for v in init.values()) and init_rng and pass_arms and equal_params
                       and spatial_metadata and spatial_counts and spatial_mass and spatial_cells and fixed_centers,
            "bank": str(bank_path), "bank_sha256": sha(bank_path), "n_samples": len(batch["prev"]),
            "sample_ids": bank["sample_ids"][:n_samples], "checkpoint": str(checkpoint),
            "equal_seed_initial_state": init, "equal_seed_initial_rng": init_rng,
            "equal_total_params": equal_params, "BCD_same_metadata": spatial_metadata,
            "BCD_same_counts": spatial_counts, "BCD_same_mass": spatial_mass, "BCD_same_cells": spatial_cells,
            "ABCD_same_fixed_centers": fixed_centers, "arms": arms,
            "comparison_tolerance": {"warm_prediction_atol": 1e-6, "warm_prediction_rtol": 1e-5}}


def load_old_model(path):
    spec = importlib.util.spec_from_file_location("xb_original_model", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.MNISTModel


def s39_batches(cfg):
    random.seed(117)
    np.random.seed(117)
    torch.manual_seed(117)
    components = np.load(cfg["MANO"]["NPZ"])["hands_components"].astype(np.float32)
    dataset = build_dataset(cfg, "train", components, train=True)
    indices = torch.randint(len(dataset), (4,), generator=torch.Generator().manual_seed(19)).tolist()
    train = default_collate([dataset[index] for index in indices])
    root = Path(cfg["DATA"]["ROOT"])
    seqs = dict(sequences_for_split(root, "val_core", splits_manifest(cfg)))
    seq = "zgz_local"
    events, offsets, aux, targets = ET.load_sequence(root, seqs[seq], seq)
    a, b = next((int(a), int(b)) for a, b in np.asarray(aux["valid_runs_ms"]).reshape(-1, 2) if b - a > 400)
    beta = torch.tensor(aux["betas"], dtype=torch.float32).view(1, -1)
    K = torch.tensor(aux["camera_K"], dtype=torch.float32).view(1, 3, 3)
    evals, descriptions = [], []
    for end in (a + 199, a + 249):
        x = torch.from_numpy(ET.build_lnes(events, offsets, end, 200)).unsqueeze(0)
        prev = targets[end - 199].copy() + ET.sample_init_noise(cfg, np.random.default_rng(0), 1.)
        evals.append((x, torch.from_numpy(prev).view(1, -1), beta, K))
        descriptions.append({"sequence": seq, "end_ms": end, "window_ms": 200,
                             "raw_events": int(offsets[end + 1] - offsets[end - 199])})
    return train, evals, {"train_indices": indices, "train_batch": 4, "eval_packets": descriptions}


def run_s39(model, train, evals):
    output = {}
    model.eval()
    for i, (x, prev, beta, K) in enumerate(evals):
        model.set_hand_context(None, None)
        with torch.no_grad():
            output[f"eval/{i}/prediction"] = model(x, prev, betas=beta, camera_K=K).detach().clone()
        output[f"eval/{i}/delta"] = output[f"eval/{i}/prediction"] - prev
    model.train()
    model.zero_grad(set_to_none=True)
    x, prev, target, beta, K = train
    torch.manual_seed(812)
    prediction = model(x, prev, betas=beta, camera_K=K)
    loss, parts = model._compute_loss(prediction, target, beta)
    loss.backward()
    output["train/prediction"] = prediction.detach().clone()
    output["train/loss"] = loss.detach().clone()
    for name, value in parts.items():
        output[f"train/loss_part/{name}"] = torch.as_tensor(value).detach().clone()
    for name, p in model.named_parameters():
        output[f"train/gradient/{name}"] = p.grad.detach().clone() if p.grad is not None else None
    output["after_train/global_rng"] = torch.random.get_rng_state().clone()
    for name, value in model.state_dict().items():
        output[f"after_train/state/{name}"] = value.detach().clone()
    return output


def s39_audit(backup):
    run_dir = REPO / "outputs/semkine/dt_dz_l3_s3407"
    cfg = load_config(str(run_dir / "config_resolved.yaml"))
    checkpoint = run_dir / "dt_dz_l3_s3407-step=6000.ckpt"
    state = torch.load(checkpoint, map_location="cpu")["state_dict"]
    old_class = load_old_model(backup)
    torch.manual_seed(27)
    original = old_class(cfg=copy.deepcopy(cfg)).float().cpu().eval()
    old_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(27)
    current = MNISTModel(cfg=copy.deepcopy(cfg)).float().cpu().eval()
    new_rng = torch.random.get_rng_state().clone()
    init = same_tensors(original.state_dict(), current.state_dict())
    rng_equal = torch.equal(old_rng, new_rng)
    original.load_state_dict(state, strict=True)
    current.load_state_dict(state, strict=True)
    trained = same_tensors(original.state_dict(), current.state_dict())
    train, evals, descriptions = s39_batches(cfg)
    old_result, new_result = run_s39(original, train, evals), run_s39(current, train, evals)
    result = same_tensors(old_result, new_result)
    nonvacuous = {"eval_delta_abs_sum": [float(old_result[f"eval/{i}/delta"].abs().sum()) for i in range(len(evals))],
                 "nonzero_gradient_tensors": sum(value is not None and bool(value.any()) for key, value in old_result.items()
                                                   if key.startswith("train/gradient/")),
                 "train_loss": float(old_result["train/loss"])}
    passed = init["equal"] and rng_equal and trained["equal"] and result["equal"] \
             and all(v > 0 for v in nonvacuous["eval_delta_abs_sum"]) and nonvacuous["nonzero_gradient_tensors"] > 0
    return {"passed": passed, "original_source": str(backup), "original_source_sha256": sha(backup),
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint), "config": cfg["_config_path"],
            "initial_state": init, "initial_rng_equal": rng_equal, "trained_state": trained,
            "eval_train_loss_gradient_updated_state": result, "nonvacuity": nonvacuous, "batches": descriptions,
            "scope": "S39 trained network and loss; deployment DT2 filter/gate code is untouched and excluded from this audit"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, default=REPO / "outputs/xb/c0_train_bank.pt")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=REPO / "outputs/xb/preflight.json")
    args = parser.parse_args()
    os.chdir(REPO)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    result = {"device": "cpu", "threads": args.threads, "torch": torch.__version__, "checks": {}}
    try:
        print("Auditing first fixed bank samples, residual controls and warm start", flush=True)
        result["checks"]["bank_controls_warmstart"] = bank_audit(args.bank, args.samples)
        print("Auditing trained S39 against original model source", flush=True)
        result["checks"]["s39_default_bitwise"] = s39_audit(REPO / "outputs/xb/orig_backup/model.py")
        result["passed"] = all(check["passed"] for check in result["checks"].values())
    except Exception as error:
        result.update(passed=False, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(f"preflight saved {args.output}; passed={result['passed']}", flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
