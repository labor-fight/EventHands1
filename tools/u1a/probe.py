#!/usr/bin/env python3
"""Auxiliary U1a/S38 raw-measurement and shared-head gradient diagnostics.

Explicit run/last checkpoint required. This does not run evalx, select a checkpoint,
create a main row, instantiate a Trainer/optimizer, or update any parameter.
Raw zgz measurements use eval_track's exact 50-ms windows without recursion;
training data are read only for one independent three-component gradient probe.
Example:
  python tools/u1a/probe.py --run-dir outputs/semkine/u1a_shared_2k_s3407 --ckpt last --output RUN/probe_last.json
  python tools/u1a/probe.py --run-dir outputs/semkine/u1a_shared_debug_s3407 --ckpt last --max-raw-steps 4 --grad-batch 4 --device cpu
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "model")]
from config import load_config
from model import MNISTModel
from semkine import eval_track as ET
from semkine.dataset import build_dataset, sequences_for_split, splits_manifest
from semkine.events import EventPacket, collate_packets
from semkine.lie import so3_exp, so3_log
from tools.x1001.evalx import find_ckpt

STEP_MS = 50
TRAIN_SUBJECTS = {"ch", "lfz", "lpc", "lr", "ly", "lyh", "lyq", "ycy", "ylf"}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stats(values):
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(x):
        return None
    if not np.isfinite(x).all():
        raise ValueError("non-finite diagnostic values")
    return {"n": int(len(x)), "mean": float(x.mean()), "rms": float(np.sqrt((x * x).mean())),
            "std": float(x.std()), "median": float(np.median(x)),
            "p90": float(np.percentile(x, 90)), "max": float(x.max())}


def raw_scale(raw_root, raw_finger, gt, reference):
    rr, rf = np.asarray(raw_root), np.asarray(raw_finger)
    gt_ref = so3_log(so3_exp(torch.as_tensor(gt[:, 3:6], dtype=torch.float64)) @ reference.T).numpy()
    return {"root_reference_residual_component_rad": stats(rr),
            "root_reference_residual_norm_rad": stats(np.linalg.norm(rr, axis=-1)),
            "root_GT_reference_residual_norm_rad": stats(np.linalg.norm(gt_ref, axis=-1)),
            "finger_residual_component_rad": stats(rf),
            "finger_residual_joint_norm_rad": stats(np.linalg.norm(rf.reshape(-1, 15, 3), axis=-1)),
            "finger_GT_residual_component_rad": stats(gt[:, 6:])}


class RawCapture:
    """Capture pre-reference/pre-filter heads; do not use forward_packet's output."""
    def __init__(self, model):
        self.model = model
        self.cache = {}
        self.handles = []
        self.unified = getattr(model, "u1a_mode", "off") != "off"
        if not self.unified:
            for key, name in (("root", "root_abs_head"), ("finger", "finger_abs_head")):
                if not hasattr(model, name):
                    raise ValueError(f"S38 raw probe requires {name}")
                def hook(_module, _inputs, output, key=key):
                    self.cache[key] = output.detach().clone()
                self.handles.append(getattr(model, name).register_forward_hook(hook))

    def forward(self, batch):
        self.cache.clear()
        # The returned filtered/held prediction is deliberately discarded.
        self.model.forward_packet(batch)
        if self.unified:
            raw = self.model.u1a_last_raw
            root, finger = raw[:, 3:6], raw[:, 6:]
        else:
            if set(self.cache) != {"root", "finger"}:
                raise RuntimeError("raw absolute-measurement hooks were not called")
            root, finger = self.cache["root"], self.cache["finger"]
        if root.shape != (batch.batch_size, 3) or finger.shape != (batch.batch_size, 45):
            raise ValueError("unexpected root/finger measurement shapes")
        return root, finger

    def close(self):
        for handle in self.handles:
            handle.remove()


@torch.no_grad()
def metric_arrays(model, raw_root, raw_finger, target, betas, reference):
    # Exp(raw) R_ref is the raw root measurement, before gain/hold/feedback.
    measured_R = so3_exp(raw_root.double()) @ reference.to(raw_root.device)
    target_R = so3_exp(target[:, 3:6].double())
    relative_R = measured_R.transpose(-1, -2) @ target_R
    cosine = (relative_R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0
    angle = torch.rad2deg(cosine.clamp(-1.0, 1.0).acos())
    # Identity root on BOTH sides isolates articulation. Translation is zero;
    # residual45 is decoded by _fk, which adds MANO hands_mean exactly once.
    predicted = target.new_zeros(target.shape)
    truth = target.new_zeros(target.shape)
    predicted[:, 6:] = raw_finger
    truth[:, 6:] = target[:, 6:]
    pv, pj = model._fk(predicted, betas)
    gv, gj = model._fk(truth, betas)
    root = model.FK_ROOT_JOINT
    pr, gr = pj[:, root:root + 1], gj[:, root:root + 1]
    joint = ((pj - pr) - (gj - gr)).norm(dim=-1).mean(-1) * 1000.0
    mesh = ((pv - pr) - (gv - gr)).norm(dim=-1).mean(-1) * 1000.0
    return {"root_geodesic_deg": angle.cpu().numpy(),
            "finger_articulation_root_relative_mpjpe_mm": joint.cpu().numpy(),
            "finger_articulation_root_relative_mpvpe_mm": mesh.cpu().numpy()}


def summarize(arrays, counts, raw_root, raw_finger, gt, reference, frame_keys):
    if not len(counts):
        return None
    live = np.asarray(counts) > 0
    canonical_frames = json.dumps(frame_keys, separators=(",", ":"), ensure_ascii=True).encode()
    canonical_mask = json.dumps(list(zip(frame_keys, live.tolist())), separators=(",", ":"), ensure_ascii=True).encode()
    return {"n_windows": int(len(counts)), "n_empty": int((~live).sum()),
            "n_valid_measurements": int(live.sum()), "measurement_coverage": float(live.mean()),
            "frame_index_sha256": hashlib.sha256(canonical_frames).hexdigest(),
            "measurement_mask_sha256": hashlib.sha256(canonical_mask).hexdigest(),
            "event_count": stats(counts),
            "measurement_metrics": {k: stats(np.asarray(v)[live]) for k, v in arrays.items()},
            "all_windows_auxiliary": {k: stats(v) for k, v in arrays.items()},
            "raw_scale": raw_scale(np.asarray(raw_root)[live], np.asarray(raw_finger)[live], np.asarray(gt)[live], reference)
                         if live.any() else None}


@torch.inference_mode()
def raw_probe(model, cfg, seqs, device, args):
    capture = RawCapture(model)
    root = Path(cfg["DATA"]["ROOT"])
    reference = so3_exp(torch.tensor(cfg["MODEL"]["ROOT_REF"], dtype=torch.float64))
    collected = []
    try:
        for seq_id, (seq, directory) in enumerate(seqs):
            events, offsets, aux, pos51 = ET.load_sequence(root, directory, seq)
            tsub_path = root / directory / f"{seq}_tsub.npy"
            if not tsub_path.is_file():
                raise FileNotFoundError(f"raw evidence requires {tsub_path}")
            tsub = np.load(tsub_path, mmap_mode="r")
            runs = np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)
            betas = np.asarray(aux["betas"], dtype=np.float32).copy()
            camera_K = np.asarray(aux["camera_K"], dtype=np.float32).reshape(3, 3).copy()
            descriptors, protocol_count = [], 0
            for run_id, (a, b) in enumerate(runs):
                for end in np.arange(a + STEP_MS - 1, b, STEP_MS, dtype=np.int64):
                    if protocol_count % args.stride == 0:
                        descriptors.append((run_id, int(end), protocol_count))
                    protocol_count += 1
            if args.max_raw_steps:
                descriptors = descriptors[:args.max_raw_steps]
            metrics, roots, fingers, targets, counts = {}, [], [], [], []
            index_records = []
            for i0 in range(0, len(descriptors), args.batch_size):
                items = []
                for run_id, end, protocol_index in descriptors[i0:i0 + args.batch_size]:
                    start = end - STEP_MS + 1
                    # Exactly the timestamp/polarity extraction used by evalx/ET.
                    ev5 = ET._window_events(events, offsets, tsub, end, STEP_MS)
                    items.append(EventPacket(events=ev5[:, 1:].copy(), sequence_id=seq_id,
                        t_start_us=start * 1000, t_end_us=(end + 1) * 1000,
                        is_sequence_start=False, is_sequence_end=False,
                        target=pos51[end].copy(), prev_state=pos51[start].copy(),
                        betas=betas, camera_K=camera_K))
                    index_records.append({"run_id": run_id, "end_ms": end,
                                          "protocol_step_index": protocol_index, "n_events": len(ev5)})
                batch = collate_packets(items).to(device)
                raw_r, raw_f = capture.forward(batch)
                part = metric_arrays(model, raw_r, raw_f, batch.target, batch.betas, reference)
                for key, value in part.items():
                    metrics.setdefault(key, []).append(value)
                roots.append(raw_r.cpu().numpy())
                fingers.append(raw_f.cpu().numpy())
                targets.append(batch.target.cpu().numpy())
                counts.append(batch.counts.cpu().numpy())
            aux.close()
            if not descriptors:
                raise ValueError(f"no fixed windows in {seq}")
            row = {"seq": seq, "protocol_windows": protocol_count, "indices": index_records,
                   "metrics": {k: np.concatenate(v) for k, v in metrics.items()},
                   "root": np.concatenate(roots), "finger": np.concatenate(fingers),
                   "gt": np.concatenate(targets), "counts": np.concatenate(counts)}
            collected.append(row)
            print(f"raw {seq}: {len(descriptors)}/{protocol_count} fixed windows", file=sys.stderr, flush=True)
    finally:
        capture.close()
    def group(rows):
        if not rows:
            return None
        keys = rows[0]["metrics"]
        return summarize({k: np.concatenate([r["metrics"][k] for r in rows]) for k in keys},
            np.concatenate([r["counts"] for r in rows]), np.concatenate([r["root"] for r in rows]),
            np.concatenate([r["finger"] for r in rows]), np.concatenate([r["gt"] for r in rows]), reference,
            [[r["seq"], i["run_id"], i["end_ms"], i["protocol_step_index"]] for r in rows for i in r["indices"]])
    total_windows = int(sum(r["protocol_windows"] for r in collected))
    if total_windows != 2590:
        raise ValueError(f"fixed zgz protocol must contain 2590 windows; found {total_windows}")
    return {"method": "head raw before root-reference/filter/empty hold; no recursive predictions",
            "measurement_metric_set": "measurement_metrics: nonempty evidence windows only",
            "gate_eligible_full_window_coverage": args.stride == 1 and args.max_raw_steps == 0,
            "gate_role": "auxiliary raw measurement only; frame/mask hashes must match across arms; recursive gate is external evalx",
            "contexts": "GT at each fixed evidence-window start, used only for routing/T; no noise or feedback",
            "reference_composition": "Exp(raw_root) @ Exp(MODEL.ROOT_REF), float64",
            "finger_FK": "residual45; MANO mean once; both roots R=I/T=0; joints and vertices aligned by MANO root joint",
            "sampling": {"step_ms": STEP_MS, "window_ms": STEP_MS, "stride": args.stride,
                "rule": "per-sequence protocol_step_index % stride == 0; valid-runs order; cap after stride",
                "max_sampled_windows_per_sequence": args.max_raw_steps or None,
                "batch_size": args.batch_size, "precision": "fp32 model, fp64 root metric"},
            "protocol_windows_total": int(sum(r["protocol_windows"] for r in collected)),
            "sampled_windows_total": int(sum(len(r["counts"]) for r in collected)),
            "overall": group(collected), "global": group([r for r in collected if "_global" in r["seq"]]),
            "local": group([r for r in collected if "_local" in r["seq"]]),
            "per_sequence": {r["seq"]: {"protocol_windows": r["protocol_windows"], "indices": r["indices"],
                                        "summary": group([r])} for r in collected}}


def gradient_stats(vectors):
    norms = {k: float(v.double().norm()) for k, v in vectors.items()}
    cosine = {}
    names = list(vectors)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            denominator = norms[first] * norms[second]
            cosine[f"{first}__{second}"] = (float(torch.dot(vectors[first].double(), vectors[second].double()) / denominator)
                                           if denominator > 0.0 else None)
    return {"norm": norms, "pairwise_cosine": cosine,
            "n_parameter_coordinates": int(next(iter(vectors.values())).numel())}


def gradient_probe(model, cfg, device, args):
    mode = getattr(model, "u1a_mode", "off")
    if args.grad_batch == 0:
        return {"status": "disabled explicitly by --grad-batch 0"}
    if mode == "off":
        return {"status": "not_applicable", "reason": "S38 has separate heads, no shared U1a MLP; no artificial shared gradient cosine"}
    if model.loss_type != "mse_51d":
        raise ValueError("gradient decomposition is defined only for the S38 mse_51d loss")
    grad_cfg = copy.deepcopy(cfg)
    if (grad_cfg.get("TRACK", {}) or {}).get("UNROLL_PAIR", False):
        raise ValueError("probe expects the U1a single-window training protocol")
    components = model.mano.hands_components.detach().cpu().numpy()
    dataset = build_dataset(grad_cfg, "train", components, train=True, input_mode="raw_packed")
    indices = np.linspace(0, len(dataset) - 1, args.grad_batch, dtype=np.int64)
    packets = [dataset[int(i)] for i in indices]
    batch = collate_packets(packets).to(device)
    # No optimizer/Trainer. Preserve a frozen observation function and its BN state.
    encoder_state = {k: v.detach().clone() for k, v in model.event_encoder.state_dict().items()}
    model.train()
    model.event_encoder.eval()
    model.zero_grad(set_to_none=True)
    enabled = device.type == "cuda" and str(cfg.get("TRAIN", {}).get("PRECISION", "")).startswith("bf16")
    try:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled):
            pred = model.forward_packet(batch)
            total, parts = model._compute_loss(pred, batch.target, batch.betas)
        terms = {"translation": model.lambda_t * parts["pos_loss"] / model.normalizer,
                 "root": model.lambda_r * parts["rot_loss"] / model.normalizer,
                 "finger": model.lambda_pose * parts["mano_loss"] / model.normalizer}
        torch.testing.assert_close(sum(terms.values()), total, rtol=1e-6, atol=1e-6)
        if not torch.isfinite(total) or total <= 0:
            raise ValueError("log10 gradient probe requires a positive finite weighted loss")
        coefficient = 1.0 / (math.log(10.0) * float(total.detach())) if model.log10_loss else 1.0
        selected = list(model.u1a_readout.named_parameters())
        names, parameters = zip(*selected)
        full_vectors = {}
        for name, term in terms.items():
            # Independent reverse passes; shared graph, no gradient accumulation.
            gradients = torch.autograd.grad(term * coefficient, parameters, retain_graph=True, allow_unused=True)
            full_vectors[name] = [torch.zeros_like(p).reshape(-1).cpu() if g is None else g.detach().float().reshape(-1).cpu()
                                  for p, g in zip(parameters, gradients)]
        if mode == "shared":
            head_prefix, group_label = "shared_head.", "shared"
        else:
            head_prefix, group_label = "node_heads.", "untied_bank"
        def select_group(predicate):
            include = [i for i, name in enumerate(names) if predicate(name)]
            return gradient_stats({key: torch.cat([value[i] for i in include]) for key, value in full_vectors.items()})
        groups = {f"{group_label}_head_with_LayerNorm": select_group(lambda n: n.startswith(head_prefix)),
                  f"{group_label}_MLP_Linear_only": select_group(lambda n: n.startswith(head_prefix)
                        and (n.rsplit(".", 2)[1] in ("1", "3"))),
                  "identity_embeddings": select_group(lambda n: n.startswith(("type_embed.", "joint_embed.")))}
        assert all(torch.equal(value, model.event_encoder.state_dict()[key]) for key, value in encoder_state.items())
        raw = model.u1a_last_raw.cpu().numpy()
        reference = so3_exp(torch.tensor(cfg["MODEL"]["ROOT_REF"], dtype=torch.float64))
        return {"status": "ok", "data": "train only, actual deterministic training sampler including augmentation/prev noise",
                "train_samples": len(dataset), "dataset_seed": int(dataset.seed),
                "indices_rule": "linspace(0, train_samples-1, grad_batch), integer endpoints included",
                "indices": indices.tolist(), "batch_size": args.grad_batch,
                "samples": [{"seq": p.meta["seq"], "end_idx": int(p.meta["end_idx"]),
                             "window_ms": int(p.meta["window_ms"]), "n_events": p.n_events} for p in packets],
                "loss": {"type": model.loss_type, "root": model.root_loss,
                         "lambda_t": model.lambda_t, "lambda_r": model.lambda_r, "lambda_pose": model.lambda_pose,
                         "normalizer": model.normalizer, "weighted_total": float(total.detach()),
                         "weighted_terms": {k: float(v.detach()) for k, v in terms.items()},
                         "log10": model.log10_loss, "common_gradient_coefficient": coefficient},
                "gradient_semantics": "each vector is its weighted loss contribution to grad(log10(total)); common denominator detached, not log10(each term)",
                "groups": groups, "raw_scale": raw_scale(raw[:, 3:6], raw[:, 6:], batch.target.cpu().numpy(), reference),
                "translation_delta_component_m": stats(raw[:, :3]), "encoder_state_unchanged": True,
                "precision": "configured bf16 CUDA autocast, U1a head/root fp32" if enabled else "fp32",
                "limitation": "one fixed augmented train batch; cosine is a local diagnostic, not a population estimate or causal proof"}
    finally:
        model.zero_grad(set_to_none=True)
        model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--ckpt", required=True, choices=("last",), help="explicit last only; never select/check gates")
    parser.add_argument("--config", type=Path, help="otherwise prefer run/config_resolved.yaml")
    parser.add_argument("--output", type=Path, help="write JSON here; default prints JSON to stdout only")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:N")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-raw-steps", type=int, default=0, help="smoke cap per sequence after stride; 0=all")
    parser.add_argument("--grad-batch", type=int, default=16, help="fixed augmented train batch; 0=skip")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.stride < 1 or args.max_raw_steps < 0 or args.grad_batch < 0 or args.threads < 1:
        parser.error("batch-size/stride/threads positive; max-raw-steps/grad-batch nonnegative")
    start = time.monotonic()
    torch.set_num_threads(args.threads)
    run = args.run_dir.resolve()
    ckpt, parsed_step, _selection_score = find_ckpt(run, args.ckpt)
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    metadata = json.loads((run / "training_metadata.json").read_text()) if (run / "training_metadata.json").is_file() else {}
    config_path = args.config or (run / "config_resolved.yaml" if (run / "config_resolved.yaml").is_file()
                                 else Path(metadata["config_path"]))
    cfg = load_config(config_path)
    root = Path(cfg["DATA"]["ROOT"])
    manifest = splits_manifest(cfg) or root / "splits_semkine.json"
    if not manifest.is_file() or manifest.name != "splits_semkine.json":
        raise ValueError("only the existing splits_semkine.json protocol is accepted")
    train = sequences_for_split(root, "train", manifest)
    seqs = sequences_for_split(root, "val_core", manifest)
    if len(train) != 72 or {s.split("_")[0] for s, _ in train} != TRAIN_SUBJECTS:
        raise ValueError("expected nine training subjects/72 sequences")
    if [s for s, _ in seqs] != ["zgz_global", "zgz_local"]:
        raise ValueError("expected val_core ordering zgz_global, zgz_local")
    if metadata.get("train_sequences") and set(metadata["train_sequences"]) != {s for s, _ in train}:
        raise ValueError("run training metadata disagrees with the fixed training manifest")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    checkpoint_metadata = torch.load(str(ckpt), map_location="cpu")
    checkpoint_step = int(checkpoint_metadata.get("global_step", -1))
    del checkpoint_metadata
    if checkpoint_step != parsed_step:
        raise ValueError("checkpoint global_step disagrees with evalx explicit-last filename")
    model = MNISTModel.load_from_checkpoint(str(ckpt), cfg=cfg, map_location="cpu").to(device).eval()
    if model.pose_repr != "mano_full_axis_angle" or model.root_meas != "abs" or model.finger_meas != "abs":
        raise ValueError("probe requires S38/U1a full51 absolute root and finger measurements")
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except subprocess.CalledProcessError:
        git_commit = None
    result = {"kind": "auxiliary_raw_and_gradient_diagnostic", "run": run.name, "ckpt": str(ckpt),
              "seed": int(cfg.get("SEED", metadata.get("seed", -1))), "step": checkpoint_step,
              "split": "val_core", "gate_eligible": args.stride == 1 and args.max_raw_steps == 0,
              "root_ref": cfg["MODEL"]["ROOT_REF"],
              "checkpoint_policy": "evalx.find_ckpt explicit last (newest numbered checkpoint); no selection or gate changes", "ckpt_sha256": sha256(ckpt),
              "config": str(Path(config_path).resolve()), "config_sha256": sha256(config_path),
              "manifest": str(manifest), "manifest_sha256": sha256(manifest),
              "source_git_commit": git_commit, "probe_sha256": sha256(__file__),
              "device": str(device), "u1a_mode": getattr(model, "u1a_mode", "off"),
              "limitations": ["auxiliary diagnostic; official recursive evalx/main-row results remain separate",
                  "zgz is both development and test under the user-fixed protocol; no independent test claim",
                  "raw probing is nonrecursive and uses GT start-state routing context; root/finger observations are state-free",
                  "empty-window raw values are auxiliary only; measurement metrics/gates use nonempty windows and recorded mask hashes",
                  "shared versus untied may differ in capacity and scheduling; these probes do not establish causal sharing benefit"]}
    result["raw"] = raw_probe(model, cfg, seqs, device, args)
    result["gradient"] = gradient_probe(model, cfg, device, args)
    result["elapsed_s"] = time.monotonic() - start
    payload = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr, flush=True)
    else:
        print(payload)


if __name__ == "__main__":
    main()