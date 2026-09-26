#!/usr/bin/env python3
"""Inspect graph construction on training packets; this is not an accuracy evaluation.

No checkpoints are trained or scored. Outputs contain graph/support diagnostics under
controlled previous states, including the geometry-only neighbour counterfactual.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "model"), str(ROOT)]
from config import load_config
from model import MNISTModel
from semkine.dataset import sequences_for_split, splits_manifest
from semkine.eval_track import load_sequence, _window_events, make_eval_packet
from semkine.event_guided_mesh import build_event_guided_graph
from semkine.mesh_graph import visible_vertices, nearest_node_lut, assign_events_by_lut


def propagate(support, idx, mask, layers):
    result = [support.clone()]
    for _ in range(layers):
        incoming = support.gather(1, idx.flatten(1)).reshape_as(idx) & mask.bool()
        support = support | incoming.any(-1)
        result.append(support.clone())
    return result


def changed_neighbors(left, right):
    overlap = (left.unsqueeze(-1) == right.unsqueeze(-2)).any(-1)
    return (~overlap).float().mean(), (~overlap.all(-1)).float().mean()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packets", type=int, default=24, help="evenly spaced windows per sequence")
    ap.add_argument("--sequences", nargs="+", default=["lyq_local", "lyq_global"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", type=Path,
                    default=ROOT / ".experiments/egm_debug_20260923/real_packets.json")
    args = ap.parse_args()
    torch.manual_seed(3407)
    rng = np.random.default_rng(0)
    cfg = load_config(ROOT / "configs/semkine/event_guided_mesh_s3407.yaml")
    m = MNISTModel(cfg).to(args.device).eval()
    enc = m.event_encoder
    root = Path(cfg["DATA"]["ROOT"])
    available = dict(sequences_for_split(root, "train", splits_manifest(cfg)))
    captured = {}
    def capture(_module, inputs):
        captured["observations"], captured["vertices"] = (t.detach() for t in inputs)
    hook = enc.register_forward_pre_hook(capture)
    weights = m.mano.weights.float()
    parents = m.mano.kintree_table[0].long().tolist()
    branch = list(range(16))
    for j in range(1, 16):
        while parents[branch[j]] != 0:
            branch[j] = parents[branch[j]]
    group = torch.tensor(branch, device=args.device)[weights.argmax(-1)]
    result = []
    for seq in args.sequences:
        if seq not in available:
            raise ValueError(f"{seq} is not in the training split")
        directory = available[seq]
        events, offsets, aux, states = load_sequence(root, directory, seq)
        tsub = np.load(root / directory / f"{seq}_tsub.npy", mmap_mode="r")
        ends = np.concatenate([np.arange(int(a) + 49, int(b), 50)
                               for a, b in aux["valid_runs_ms"] if b - a >= 50])
        ends = ends[np.linspace(0, len(ends) - 1, min(args.packets, len(ends)), dtype=int)]
        betas = torch.as_tensor(aux["betas"], dtype=torch.float32, device=args.device).view(1, -1)
        camera = torch.as_tensor(aux["camera_K"], dtype=torch.float32, device=args.device).view(1, 3, 3)
        for end in ends:
            ev = _window_events(events, offsets, tsub, int(end), 50)
            for regime, scales in (("clean", (0, 0, 0)), ("small", (.005, .05, .05)),
                                   ("large", (.05, .3, .3))):
                start = int(end) - 49
                noise = rng.standard_normal(51) * np.repeat(scales, [3, 3, 45])
                prev = torch.as_tensor(states[start] + noise, dtype=torch.float32,
                                       device=args.device).view(1, -1)
                pk = make_eval_packet(ev, prev, betas, camera, 50, args.device)
                evidence, joint_support = m._event_guided_mesh_forward(pk, prev)
                obs, verts = captured["observations"], captured["vertices"]
                guided, dp, edge_mask = enc.last_graph
                initial = obs[..., 0] > 0
                uv, z = m._project_verts(verts, camera)
                visible = visible_vertices(verts, uv, z, m.mano.f, 180, 240,
                                           m.mg_front_px, m.mg_z_tol)
                pixel = uv.round().long()
                keys = pixel[0, visible[0], 1] * 240 + pixel[0, visible[0], 0]
                collision_count = len(keys) - len(keys.unique())
                # Independently compare the pixel lookup to continuous projected vertices.
                # Stratified event subsampling bounds this diagnostic's work.
                sample_ids = torch.linspace(0, max(len(ev) - 1, 0), min(len(ev), 512),
                                            device=args.device).long()
                sampled = pk.events[sample_ids]
                lut = nearest_node_lut(uv, visible, 180, 240, m.mg_band_px)
                assigned = assign_events_by_lut(sampled, lut, enc.n_nodes)
                distances = (sampled[:, None, 1:3] - uv).square().sum(-1)
                distances = distances.masked_fill(~visible, float("inf"))
                nearest_d2, nearest = distances.min(-1)
                nearest = torch.where(nearest_d2 <= m.mg_band_px ** 2, nearest,
                                      torch.full_like(nearest, enc.n_nodes))
                lut_disagrees = (assigned != nearest).float().mean() if len(sampled) else 0.
                geometric, _, gm = build_event_guided_graph(
                    verts, obs, enc.k, enc.candidates, 0., enc.geometry_scale)
                gs = propagate(initial, guided, edge_mask, len(enc.layers))
                base = propagate(initial, geometric, gm, len(enc.layers))
                edge_change, node_change = changed_neighbors(guided, geometric)
                mass = gs[-1].float() @ enc.lbs_pool_weights.T
                selected_obs = obs.gather(1, guided.flatten(1).unsqueeze(-1).expand(-1, -1, 6))
                selected_obs = selected_obs.reshape(1, enc.n_nodes, enc.k, 6)
                event_term = (obs.unsqueeze(2) - selected_obs).square().mean(-1)
                geo_term = dp.square().sum(-1)
                query_active = initial.unsqueeze(-1).expand_as(event_term)
                edge_ratio = event_term / (event_term + geo_term).clamp_min(1e-12)
                target_group = group[guided]
                source_group = group.view(1, -1, 1).expand_as(target_group)
                fingers = (target_group != 0) & (source_group != 0)
                crosses = (target_group != source_group) & fingers
                # A small pose perturbation tests assignment/topology sensitivity, not pose error.
                shifted = prev.clone()
                shifted[:, 0] += .001
                m._event_guided_mesh_forward(pk, shifted)
                shifted_edge, shifted_node = changed_neighbors(guided, enc.last_graph[0])
                result.append({
                    "sequence": seq, "end_ms": int(end), "regime": regime,
                    "events": len(ev), "observed_vertices": int(initial.sum()),
                    "visible_vertices": int(visible.sum()),
                    "visible_pixel_collisions": int(collision_count),
                    "sampled_event_lut_nearest_disagreement": float(lut_disagrees),
                    "support_by_layer": [int(s.sum()) for s in gs],
                    "geometry_only_support_by_layer": [int(s.sum()) for s in base],
                    "changed_edge_fraction": float(edge_change),
                    "changed_node_fraction": float(node_change),
                    "event_score_share_active_queries": float(edge_ratio[query_active].mean())
                        if query_active.any() else None,
                    "cross_finger_edge_fraction": float(crosses.float().mean()),
                    "lbs_supported_mass": mass[0].cpu().tolist(),
                    "joint_gates_open": int(joint_support.sum()),
                    "joint_gates_below_one_percent_mass": int((joint_support & (mass < .01)).sum()),
                    "one_mm_shift_edge_change": float(shifted_edge),
                    "one_mm_shift_node_change": float(shifted_node),
                    "guided_unobserved_nodes_reached": int((gs[-1] & ~initial).sum()),
                    "geometry_unobserved_nodes_reached": int((base[-1] & ~initial).sum()),
                })
    hook.remove()
    grouped = {}
    for seq in args.sequences:
        for regime in ("clean", "small", "large"):
            rows = [r for r in result if r["sequence"] == seq and r["regime"] == regime]
            summary = {"packets": len(rows)}
            for key in result[0]:
                vals = [r[key] for r in rows if isinstance(r[key], (int, float))]
                if vals:
                    summary[key] = {"mean": float(np.mean(vals)), "median": float(np.median(vals)),
                                    "min": float(np.min(vals)), "max": float(np.max(vals))}
            summary["support_by_layer_mean"] = np.mean([r["support_by_layer"] for r in rows], 0).tolist()
            summary["geometry_only_support_by_layer_mean"] = np.mean(
                [r["geometry_only_support_by_layer"] for r in rows], 0).tolist()
            summary["lbs_supported_mass_mean"] = np.mean([r["lbs_supported_mass"] for r in rows], 0).tolist()
            grouped[f"{seq}/{regime}"] = summary
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "purpose": "Graph/support debug on training data, no trained accuracy evaluation",
        "seed": 3407, "noise_rng_seed": 0, "packet_window_ms": 50,
        "previous_state": "controlled GT at window start plus stated Gaussian noise, not rollout",
        "event_model_sha256": hashlib.sha256((ROOT / "semkine/event_guided_mesh.py").read_bytes()).hexdigest(),
        "grouped": grouped, "packets": result,
    }, ensure_ascii=False, indent=2))
    print(f"Wrote debug observations: {args.output}")


if __name__ == "__main__":
    main()
