#!/usr/bin/env python3
"""Root-tracking round: evaluate and report an anchored tracker (semkine.anchored.AnchoredTracker).

The composite is two trained runs, a state-free absolute arm and a `prev + delta` tracker, with fixed
gains. It is evaluated by exactly the evalx machinery (`evalx.evaluate`: protocol loop, controls,
teacher forcing, perturbation, CIs, failures, motion) and gets a main row in the format
tools/report_table.py reads (`evalx.aggregate_runs`; latency of the composite on real packets;
MACs and parameters of both networks).

    python tools/rt/pair_eval.py eval --abs outputs/semkine/rt_cnn_s3407 --trk outputs/semkine/rt_cnndelta_s3407 \\
        --a-root 0.4 --a-rest 0.6 --name rt_anchor_s3407 [--ckpt last] [--controls --tf --perturb]
    python tools/rt/pair_eval.py row --arm rt_anchor --runs outputs/semkine/rt_anchor_s3407 outputs/semkine/rt_anchor_s3408 \\
        [--ckpt last --variant tf_pert]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "model"), str(REPO / "tools" / "x1001")]
from config import load_config                                   # noqa: E402
from mano_layer import ManoLayer                                 # noqa: E402
from model import MNISTModel                                     # noqa: E402
from semkine.anchored import AnchoredTracker, CausalFilter       # noqa: E402
from semkine.dataset import sequences_for_split, splits_manifest  # noqa: E402
import evalx as EX                                               # noqa: E402


def run_cfg(run: Path) -> Path:
    return next(iter(sorted(run.glob("*.yaml"))))


def build(abs_run: Path, trk_run, ckpt: str, a_root: float, a_rest: float, dev, a_trans=None):
    """AnchoredTracker of two runs, or (trk_run None) the CausalFilter of the absolute run."""
    parts = {}
    for k, run in (("abs", abs_run), ("trk", trk_run)):
        if run is None:
            parts[k] = parts["abs"]
            continue
        cfg = load_config(run_cfg(run))
        path, step, _ = EX.find_ckpt(run, ckpt)
        parts[k] = (cfg, MNISTModel.load_from_checkpoint(str(path), cfg=cfg, map_location=dev).to(dev).eval(),
                    str(path), step)
    if trk_run is None:
        pair = CausalFilter(parts["abs"][1], a_root, a_rest, 1.0 if a_trans is None else a_trans).to(dev).eval()
    else:
        pair = AnchoredTracker(parts["abs"][1], parts["trk"][1], a_root, a_rest, a_trans).to(dev).eval()
    return pair, parts


def cmd_eval(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    abs_run, trk_run = Path(a.abs), (Path(a.trk) if a.trk else None)
    pair, parts = build(abs_run, trk_run, a.ckpt, a.a_root, a.a_rest, dev, a.a_trans)
    cfg = parts["trk"][0]
    trk_dir = trk_run or abs_run
    seeds = {k: json.loads((r / "training_metadata.json").read_text())["seed"] for k, r in (("abs", abs_run), ("trk", trk_dir))}
    out_dir = REPO / "outputs" / "semkine" / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(run_cfg(trk_dir), out_dir / "config_tracker.yaml")
    trained = sorted(set(json.loads((abs_run / "training_metadata.json").read_text()).get("train_sequences", []))
                     | set(json.loads((trk_dir / "training_metadata.json").read_text()).get("train_sequences", [])))
    (out_dir / "training_metadata.json").write_text(json.dumps({
        "seed": seeds["trk"], "kind": "anchored_tracker" if trk_run else "causal_filter",
        "abs_run": str(abs_run), "trk_run": str(trk_run) if trk_run else None,
        "abs_seed": seeds["abs"], "trk_seed": seeds["trk"], "a_root": a.a_root, "a_rest": a.a_rest,
        "a_trans": a.a_trans,
        "train_sequences": trained}, indent=1))
    root = Path(cfg["DATA"]["ROOT"])
    seqs = sequences_for_split(root, "val_core", splits_manifest(cfg))
    assert not ({s.split("_")[0] for s, _ in seqs} & {s.split("_")[0] for s in trained}), "evaluation subject in training"
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(dev).eval()
    a.window_mode, a.window_ms, a.min_events, a.max_window_ms = "fixed", EX.STEP, 0, 300
    out = {"run": a.name, "ckpt": f"abs={parts['abs'][2]} trk={parts['trk'][2]}", "step": parts["trk"][3],
           "split": "val_core", "manifest": str(splits_manifest(cfg)),
           "anchor": {"a_root": a.a_root, "a_rest": a.a_rest, "a_trans": a.a_trans,
                      "abs_run": str(abs_run), "trk_run": str(trk_run) if trk_run else None}}
    summary, arrays = EX.evaluate(pair, cfg, mano, root, seqs, dev, a, label=a.name)
    out.update(summary)
    out["window"] = {"mode": "fixed", "ms": EX.STEP, "min_events": 0, "max_ms": 300}
    tag = f"evalx_val_core_{a.ckpt.replace('=', '')}" + ("_tf" if a.tf else "") + ("_pert" if a.perturb else "")
    np.savez_compressed(out_dir / f"{tag}.npz", **arrays)
    (out_dir / f"{tag}.json").write_text(json.dumps(out, indent=1))
    print("wrote", out_dir / f"{tag}.json", flush=True)


def cmd_row(a):
    dev = torch.device("cuda")
    per_seed, ext, mean, _ = EX.aggregate_runs(a.runs, "val_core", a.ckpt, a.variant)
    meta = json.loads((Path(a.runs[0]) / "training_metadata.json").read_text())
    pair, parts = build(Path(meta["abs_run"]), Path(meta["trk_run"]) if meta["trk_run"] else None, a.ckpt,
                        meta["a_root"], meta["a_rest"], dev,
                        meta.get("a_trans"))
    anchor = EX.latency_anchor(dev)
    lat = EX.latency_model(pair, parts["trk"][0], dev)
    row = {"arm": a.arm, "split": "val_core", "ckpt_rule": a.ckpt, "n_seeds": len(per_seed),
           "two_seed_mean": mean, "per_seed": per_seed,
           "latency_ms_raw": lat, "latency_anchor_ms": anchor, "latency_ms_scaled_full1p75": lat * 1.75 / anchor,
           "macs_forward_packet": EX.macs_of(parts["abs"][0]) + (EX.macs_of(parts["trk"][0]) if meta["trk_run"] else 0.0),
           "params_total": int(sum(p.numel() for p in pair.parameters())),
           "anchor": {"a_root": meta["a_root"], "a_rest": meta["a_rest"], "a_trans": meta.get("a_trans")},
           "note": "anchored tracker (two networks); 'two_seed_mean' is the N-seed mean"}
    out = REPO / "outputs" / "semkine"
    tag = f"_{a.variant}" if a.variant else ""
    (out / f"{a.arm}{tag}_main_row.json").write_text(json.dumps(row, indent=1))
    print(json.dumps({k: v for k, v in row.items() if k != "per_seed"}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--abs", required=True)
    e.add_argument("--trk", default=None, help="tracker run; omitted: CausalFilter of --abs")
    e.add_argument("--a-root", type=float, required=True)
    e.add_argument("--a-rest", type=float, required=True)
    e.add_argument("--a-trans", type=float, default=None, help="translation gain (default: --a-rest)")
    e.add_argument("--name", required=True)
    e.add_argument("--ckpt", default="last")
    e.add_argument("--controls", action="store_true")
    e.add_argument("--tf", action="store_true")
    e.add_argument("--perturb", action="store_true")
    r = sub.add_parser("row")
    r.add_argument("--arm", required=True)
    r.add_argument("--runs", nargs="+", required=True)
    r.add_argument("--ckpt", default="last")
    r.add_argument("--variant", default="")
    a = ap.parse_args()
    {"eval": cmd_eval, "row": cmd_row}[a.cmd](a)


if __name__ == "__main__":
    main()
