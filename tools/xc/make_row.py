#!/usr/bin/env python3
"""XC exact-6000 main-row producer using the unchanged canonical scorer.

The factory selects TimeXYZModel only for XYZ_CAUSAL_STUDY configs.  The dense
EventHands-Full timing anchor and old references remain ordinary MNISTModel.
Latency replays staged incoming states/history from a real own-state W50 loop;
it never times the temporal model with its history accidentally absent.
Only this process's module bindings are adapted; no old source is modified.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))

from config import load_config  # noqa: E402
from model import MNISTModel  # noqa: E402
from semkine import eval_track as ET  # noqa: E402
from semkine.dataset import sequences_for_split, splits_manifest  # noqa: E402
from semkine.events import EventPacketBatch  # noqa: E402
from semkine.xyz_time_study import TimeXYZModel  # noqa: E402
from tools import make_s36_row as CANONICAL  # noqa: E402
from tools.xc import evaluation  # noqa: E402

EXACT_STEP = 6000
STEP_MS = 50
SEEDS = (3407, 3408, 3409)
_LATENCY_AUDIT = {}
_MAC_AUDIT = {}
_MAC_CACHE = {}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for part in iter(lambda: file.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def model_class(cfg):
    return TimeXYZModel if bool(cfg.get("MODEL", {}).get("XYZ_CAUSAL_STUDY", False)) else MNISTModel


def exact_step_checkpoint(run, policy="last"):
    if str(policy) not in ("last", str(EXACT_STEP)):
        raise ValueError("XC accepts exact step6000 only; selection/earlier fallback is forbidden")
    matches = [p for p in sorted(Path(run).glob("*step=*.ckpt"))
               if re.search(r"step=(\d+)", p.name) and
               int(re.search(r"step=(\d+)", p.name).group(1)) == EXACT_STEP]
    if len(matches) != 1:
        raise RuntimeError(f"require exactly one named step6000 checkpoint in {run}; found {matches}")
    return EXACT_STEP, matches[0]


class ModelFactory:
    """Canonical scorer/anchor instantiate through this config-aware adapter."""

    def __new__(cls, cfg=None, **kwargs):
        return model_class(cfg or {})(cfg=cfg, **kwargs)

    @staticmethod
    def load_from_checkpoint(checkpoint_path, cfg=None, map_location=None, **kwargs):
        checkpoint_path = Path(checkpoint_path)
        _, exact = exact_step_checkpoint(checkpoint_path.parent, "last")
        if checkpoint_path.resolve() != exact.resolve():
            raise RuntimeError("requested checkpoint is not the unique exact6000 artifact")
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        if int(checkpoint.get("global_step", -1)) != EXACT_STEP:
            raise RuntimeError("named6000 checkpoint has a different optimizer global_step")
        del checkpoint
        if kwargs.pop("strict", True) is not True:
            raise ValueError("XC checkpoint loads must be strict")
        return model_class(cfg or {}).load_from_checkpoint(str(checkpoint_path), cfg=cfg,
                                                           map_location=map_location, strict=True, **kwargs)


def _synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def restore_latency_context(model, context):
    """Restore a staged prior history without advancing or changing the packet."""
    if not hasattr(model, "set_time_history"):
        return
    if context["previous_incoming_state"] is None:
        model.reset_time_history()
    else:
        model.set_time_history(context["previous_incoming_state"], context["past_interval_s"],
                               query_start_offset_s=context["query_start_offset_s"])


@torch.no_grad()
def stage_latency_packets(model, cfg, root, device, max_packets=600):
    """Untimed preparation of true incoming states and exact history on lyq_local."""
    seq, directory = next((s, d) for split in ("val_core", "train")
                          for s, d in sequences_for_split(root, split, splits_manifest(cfg)) if s == "lyq_local")
    events, offsets, aux, truth = ET.load_sequence(root, directory, seq)
    sub_path = Path(root) / directory / f"{seq}_tsub.npy"
    sub = np.load(sub_path, mmap_mode="r") if sub_path.exists() else None
    beta = torch.as_tensor(np.array(aux["betas"]), device=device, dtype=torch.float32).view(1, -1)
    camera = torch.as_tensor(np.array(aux["camera_K"]), device=device, dtype=torch.float32).view(1, 3, 3)
    model.set_hand_context(beta, camera)
    rng = np.random.default_rng(0)
    packets, contexts, counts = [], [], []
    nonzero_history, dynamic_count, run_starts = 0, 0, 0
    was_aux = getattr(model, "xyz_keep_aux", False)
    model.xyz_keep_aux = True
    try:
        for begin, end_run in np.asarray(aux["valid_runs_ms"], np.int64).reshape(-1, 2):
            prev = torch.as_tensor(truth[begin].copy() + ET.sample_init_noise(cfg, rng, 1.),
                                   device=device, dtype=torch.float32).view(1, -1)
            prev_ms = int(begin)
            older, older_ms = None, None
            for end in np.arange(begin + STEP_MS - 1, end_run, STEP_MS, dtype=np.int64):
                incoming = prev.detach().clone()
                context = {"previous_incoming_state": None if older is None else older.detach().clone(),
                           "past_interval_s": None if older is None else (prev_ms - older_ms) * .001,
                           "query_start_offset_s": (int(end) - STEP_MS + 1 - prev_ms) * .001,
                           "incoming_state_time_ms": prev_ms, "window_end_ms": int(end),
                           "run_start": older is None}
                restore_latency_context(model, context)
                ev = ET._window_events(events, offsets, sub, int(end), STEP_MS)
                packet = ET.make_eval_packet(ev, incoming.detach().clone(), beta, camera, STEP_MS, device)
                if hasattr(model, "prepare_history"):
                    nonzero_history += int(bool((model.prepare_history(incoming) != 0).any()))
                if hasattr(model, "xyz_last"):
                    del model.xyz_last  # empty packets must not inherit an earlier dynamic audit
                prediction = model.forward_packet(packet)
                if not bool(torch.isfinite(prediction).all()):
                    raise RuntimeError(f"nonfinite latency own-loop state at {seq}/{int(end)}ms")
                dynamic_count += int("verts_by_time" in getattr(model, "xyz_last", {}))
                run_starts += int(older is None)
                packets.append(packet)
                contexts.append(context)
                counts.append(len(ev))
                older, older_ms, prev, prev_ms = incoming, prev_ms, prediction.detach().clone(), int(end)
                if len(packets) == max_packets:
                    break
            if len(packets) == max_packets:
                break
    finally:
        model.xyz_keep_aux = was_aux
        if hasattr(model, "reset_time_history"):
            model.reset_time_history()
    if len(packets) != max_packets:
        raise RuntimeError(f"latency requires {max_packets} chronological packets; got {len(packets)}")
    audit = {"sequence": seq, "window_ms": STEP_MS, "step_ms": STEP_MS,
             "n_packets": len(packets), "run_starts": run_starts,
             "n_nonzero_history_packets": nonzero_history, "n_dynamic_geometry_packets": dynamic_count,
             "initializer": "original GT+sample_init_noise(cfg,rng0) at each valid-run start",
             "staging": "own predictions; incoming prev and previous incoming history cloned",
             "past_intervals_s": sorted({c["past_interval_s"] for c in contexts if c["past_interval_s"] is not None}),
             "query_offsets_s": sorted({c["query_start_offset_s"] for c in contexts}),
             "first_and_last": [{k: v for k, v in c.items() if k != "previous_incoming_state"}
                                for c in (contexts[0], contexts[-1])]}
    return packets, contexts, counts, audit


@torch.no_grad()
def latency_s36(cfg, ckpt, seqs, root, device, max_packets=600, passes=3):
    global _LATENCY_AUDIT
    if torch.device(device).type != "cuda":
        raise RuntimeError("published latency requires CUDA; CPU interface checks do not produce a timing row")
    model = ModelFactory.load_from_checkpoint(ckpt, cfg=cfg, map_location=device).to(device).eval()
    packets, contexts, counts, audit = stage_latency_packets(model, cfg, root, device, max_packets)
    for packet, context in zip(packets[:100], contexts[:100]):
        restore_latency_context(model, context)
        model.forward_packet(packet)
    _synchronize(device)
    means = []
    for _ in range(passes):
        _synchronize(device)
        start = time.perf_counter()
        for packet, context in zip(packets, contexts):
            restore_latency_context(model, context)
            model.forward_packet(packet)
        _synchronize(device)
        means.append((time.perf_counter() - start) / len(packets) * 1000.)
    _LATENCY_AUDIT = {**audit, "warm_packets": min(100, len(packets)), "passes": passes,
                      "pass_mean_ms": means, "reduction": "minimum of complete pass means",
                      "includes": "forward_packet and causal history restoration; excludes untimed packet staging"}
    del model
    torch.cuda.empty_cache()
    return min(means), int(np.mean(counts)), len(packets)


def fixed_mac_fixture():
    n_events = 4096
    generator = torch.Generator().manual_seed(0)
    events = torch.zeros(n_events, 5)
    events[:, 1] = (122 + 30 * torch.randn(n_events, generator=generator)).clamp(0, 239).round()
    events[:, 2] = (91 + 30 * torch.randn(n_events, generator=generator)).clamp(0, 179).round()
    events[:, 3] = torch.arange(n_events) * (.05 / n_events)
    events[:, 4] = (torch.rand(n_events, generator=generator) > .5).float()
    prev = torch.zeros(1, 51)
    prev[:, 2] = .45
    camera = torch.tensor([[603.4507, 0, 325.09183], [0, 602.95654, 242.09796], [0, 0, 1.]]).view(1, 3, 3)
    packet = EventPacketBatch(events=events, ptr=torch.tensor([0, n_events]),
                              sequence_id=torch.zeros(1, dtype=torch.long),
                              t_start_us=torch.zeros(1, dtype=torch.long), t_end_us=torch.tensor([50000]),
                              delta_t_s=torch.tensor([.05]), is_sequence_start=torch.zeros(1, dtype=torch.bool),
                              is_sequence_end=torch.zeros(1, dtype=torch.bool), target=torch.zeros_like(prev),
                              prev_state=prev, betas=torch.zeros(1, 10), camera_K=camera)
    older = prev.clone()
    older[:, 0] -= .002
    older[:, 4] -= .02
    older[:, 6] -= .01
    return packet, older


def macs_forward_packet(cfg):
    """Standard-layer MAC lower bound, with fixed nonzero past motion for T1."""
    global _MAC_AUDIT
    key = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()
    if key in _MAC_CACHE:
        macs, note, audit = _MAC_CACHE[key]
        _MAC_AUDIT = audit
        return macs, note
    from thop import profile

    class Wrap(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, packet):
            return self.model.forward_packet(packet)

    model = model_class(cfg)(cfg=copy.deepcopy(cfg)).float().eval()
    packet, older = fixed_mac_fixture()
    model.xyz_keep_aux = True
    if hasattr(model, "set_time_history"):
        model.set_time_history(older, .05, query_start_offset_s=.001)
    macs, _ = profile(Wrap(model), inputs=(packet,), verbose=False)
    dynamic = "verts_by_time" in getattr(model, "xyz_last", {})
    expected_dynamic = isinstance(model, TimeXYZModel) and model.time_mode == "extrapolate4"
    if dynamic != expected_dynamic:
        raise RuntimeError("MAC fixture did not execute the declared temporal geometry branch")
    note = ("forward_packet standard-layer MAC lower bound; fixed4096events/nonzero causal history; "
            "thop excludes MANO FK, rays/triangle tests, topk, scatter/reductions, SO3 and history work")
    _MAC_AUDIT = {"n_events": 4096, "fixture_seed": 0, "history_interval_s": .05,
                  "query_start_offset_s": .001, "dynamic_geometry_executed": dynamic,
                  "n_time_slices": 4 if dynamic else 1,
                  "scope": "standard NN layers only; not total arithmetic or end-to-end compute"}
    _MAC_CACHE[key] = (float(macs), note, _MAC_AUDIT)
    return float(macs), note


def install_adapters():
    CANONICAL.MNISTModel = ModelFactory
    CANONICAL.resolve_step_ckpt = exact_step_checkpoint
    CANONICAL.latency_s36 = latency_s36
    CANONICAL.macs_forward_packet = macs_forward_packet
    evaluation.install()


def _finite_json(value, path="", failures=None):
    failures = [] if failures is None else failures
    if isinstance(value, dict):
        return {str(k): _finite_json(v, f"{path}/{k}", failures) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_finite_json(v, f"{path}/{i}", failures) for i, v in enumerate(value)]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        failures.append(path)
        return None
    return value


def save_raw_accuracy(output, run, requested_seeds, per_seed, source_cfgs):
    """Persist every returned accuracy before any timing/profiling can fail.

    A partial multi-seed artifact lists missing seeds and never substitutes a
    smaller finite cohort.  Raw numeric failures remain explicit null+FAIL.
    """
    raw_path = Path(str(output) + ".raw_accuracy.json")
    missing = [seed for seed in requested_seeds if seed not in per_seed]
    payload = {"artifact_type": "XC accuracy retained before cost measurement", "run": run,
               "per_seed": per_seed, "requested_seeds": list(requested_seeds),
               "completed_seeds": list(per_seed), "missing_seeds": missing,
               "exact_optimizer_step": EXACT_STEP, "window_ms": STEP_MS, "step_ms": STEP_MS,
               "split": "val_core", "expected_sequences": ["zgz_global", "zgz_local"],
               "expected_n_frames_per_seed": 2590, "config_provenance": source_cfgs,
               "cost_status": "not measured in this raw artifact; require the separate final row",
               "finite_subset_reduction": False,
               "source_sha256": {str(p.relative_to(REPO)): sha256(p) for p in
                                 (Path(__file__).resolve(), REPO / "tools/xc/evaluation.py",
                                  REPO / "semkine/xyz_time_study.py", REPO / "tools/make_s36_row.py")}}
    invalid = []
    payload = _finite_json(payload, failures=invalid)
    payload["numeric_valid"] = not invalid
    payload["numeric_status"] = "FAIL" if invalid else "PASS"
    payload["accuracy_completion_status"] = "partial" if missing else "complete"
    payload["invalid_numeric_paths"] = invalid
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = raw_path.with_name(raw_path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(raw_path)
    return raw_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--step", choices=("last", "6000"), default="last")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))
    if not seeds or len(set(seeds)) != len(seeds):
        parser.error("--seeds must be a nonempty unique list")
    if len(seeds) == 1 and args.output is None:
        parser.error("single-seed rows require --output to avoid overwriting the formal three-seed row")
    output = args.output or REPO / f"outputs/semkine/{args.run}_main_row_last.json"
    install_adapters()
    device = torch.device("cuda")
    per_seed, cfg, sequences, root = {}, None, None, None
    source_cfgs = {}
    for seed in seeds:
        aggregate, cfg, sequences, root = CANONICAL.eval_seed(args.run, seed, device, "val_core", None, "last")
        checkpoint = Path(aggregate["ckpt"])
        aggregate["checkpoint_sha256"] = sha256(checkpoint)
        aggregate["exact_optimizer_step"] = EXACT_STEP
        per_seed[seed] = aggregate
        save_raw_accuracy(output, args.run, seeds, per_seed, source_cfgs)
        if set(aggregate["sequences"]) != {"zgz_global", "zgz_local"} or aggregate["n_frames"] != 2590:
            raise RuntimeError("formal row must retain both fixed zgz sequences and all2590 frames; raw accuracy retained")
        resolved = checkpoint.parent / "config_resolved.yaml"
        metadata = json.loads((checkpoint.parent / "training_metadata.json").read_text())
        accuracy_cfg = Path(metadata["config_path"])
        if not accuracy_cfg.is_absolute():
            accuracy_cfg = REPO / accuracy_cfg
        source_cfgs[str(seed)] = {"config_resolved_path": str(resolved), "config_resolved_sha256": sha256(resolved),
                                 "accuracy_config_path": str(accuracy_cfg), "accuracy_config_sha256": sha256(accuracy_cfg)}
        save_raw_accuracy(output, args.run, seeds, per_seed, source_cfgs)
    means = {block: {metric: float(np.mean([per_seed[s][block][metric] for s in seeds]))
                    for metric in CANONICAL.METRICS} for block in ("overall", "local", "global")}
    anchor = CANONICAL.latency_anchor(device)
    raw_latency, mean_events, n_packets = latency_s36(cfg, per_seed[seeds[0]]["ckpt"], sequences, root, device)
    frontend_macs, head_macs, n_params, n_frontend = CANONICAL.macs_s36(cfg)
    full_macs, note = macs_forward_packet(cfg)
    row = {"per_seed": per_seed, "two_seed_mean": means,
           "latency_ms_scaled_full1p75": raw_latency / anchor * CANONICAL.FULL_PUBLISHED_MS,
           "latency_ms_raw": raw_latency, "anchor_full_raw_ms": anchor,
           "latency_packets": n_packets, "latency_mean_events_per_packet": mean_events,
           "macs_frontend": frontend_macs, "macs_heads": head_macs,
           "macs_forward_packet": full_macs, "macs_forward_packet_note": note,
           "params_total": n_params, "params_frontend": n_frontend,
           "step_policy": "last", "latency_context_audit": _LATENCY_AUDIT, "mac_fixture_audit": _MAC_AUDIT,
           "protocol": {"split": "val_core", "subject": None, "manifest": "splits_semkine.json",
                        "step_ms": STEP_MS, "window_ms": STEP_MS, "n_seeds": len(seeds),
                        "seeds": list(seeds), "sequences": per_seed[seeds[0]]["sequences"],
                        "step_policy": "last", "exact_optimizer_step": EXACT_STEP,
                        "model_class": model_class(cfg).__name__,
                        "XYZ_GEOM": bool(cfg.get("MODEL", {}).get("XYZ_GEOM", True)),
                        "XYZ_CAUSAL_TIME_MODE": cfg.get("MODEL", {}).get("XYZ_CAUSAL_TIME_MODE", "static"),
                        "note": f"{len(seeds)}-seed arithmetic mean, exact6000, own-loop rawW50; no checkpoint selection",
                        "full_anchor_published_ms": CANONICAL.FULL_PUBLISHED_MS,
                        "anchor_iters": 300, "anchor_rounds": 8, "anchor_reduction": "minimum round mean",
                        "latency_context": "fixed incoming priors/history staged from real own-loop, not GT-held priors"},
           "provenance": {"config_resolved": source_cfgs,
                          "source_sha256": {str(p.relative_to(REPO)): sha256(p) for p in
                                            (Path(__file__).resolve(), REPO / "tools/xc/evaluation.py",
                                             REPO / "semkine/xyz_time_study.py", REPO / "tools/make_s36_row.py")}}}
    invalid = []
    row = _finite_json(row, failures=invalid)
    row["numeric_valid"] = not invalid
    row["numeric_status"] = "PASS" if not invalid else "FAIL"
    row["invalid_numeric_paths"] = invalid
    row["finite_subset_reduction"] = False
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
    print(f"wrote {output}; exact6000, n_seeds={len(seeds)}, numeric_status={row['numeric_status']}")


if __name__ == "__main__":
    main()
