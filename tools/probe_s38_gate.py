#!/usr/bin/env python3
"""Mechanism gate on the selected checkpoints: the same checkpoint, closed loop on the selection
protocol (val_core = zgz, 50 ms, rng seed 0), live and with `model.ablate_evidence = True`.

For the mesh graph family the ablation clears the per-vertex observations and the observed-vertex
mask, so the LBS pool, the S38 rigid node and the S38 part lever terms are exactly zero; what is
left is the background node, the finger heads' prev angles and `prev_mlp`. The gate in
docs/S38_MESH3D_PREREG.md section 4 is a recursive-RA degradation of at least 1.5 mm. Works for
any arm that exposes `ablate_evidence` (fk_graph, mesh_graph, routed readout, mesh query).

    CUDA_VISIBLE_DEVICES=6 python tools/probe_s38_gate.py \\
        --runs outputs/semkine/s38_mesh3d_s3407 outputs/semkine/s38_mesh3d_s3408 \\
               outputs/semkine/s38_rootlever_s3407 outputs/semkine/s38_rootlever_s3408 \\
        --out outputs/semkine/probe_s38_gate.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))

from config import load_config                                    # noqa: E402
from model import MNISTModel                                      # noqa: E402
from semkine import eval_track as ET                              # noqa: E402
from semkine.dataset import sequences_for_split, splits_manifest  # noqa: E402


def load_run(run_dir: Path, device):
    sels = sorted(run_dir.glob("selection_val_core_step50*.json"))
    if not sels:
        raise SystemExit(f"{run_dir}: no selection JSON; run tools/select_checkpoint.py first")
    sel = json.loads(sels[-1].read_text())["selected"]
    cfg = load_config(json.loads((run_dir / "training_metadata.json").read_text())["config_path"])
    model = MNISTModel.load_from_checkpoint(sel["ckpt"], cfg=cfg, map_location=device)
    return model.to(device).eval(), cfg, sel


@torch.no_grad()
def closed_loop(model, cfg, root, seqs, step_ms, device, ablate: bool) -> dict:
    if not hasattr(model, "ablate_evidence"):
        raise SystemExit("this arm has no `ablate_evidence` hook; the gate is not defined for it")
    model.ablate_evidence = bool(ablate)
    try:
        ra = ab = n = 0.0
        per_seq = {}
        # one rng for the whole pass, as `select_checkpoint.py` / `make_s36_row.py` do: the second
        # sequence's initial-state noise is drawn after the first's (a fresh rng per sequence
        # reproduced the selection to 0.24 mm, not 0.00)
        rng = np.random.default_rng(0)
        for seq, d in seqs:
            r = ET.track_sequence(model, model.mano, cfg, root, d, seq, step_ms, device,
                                  rng, 1.0, 1.0, None)
            if r is None:
                continue
            ra += r["mpjpe_ra_mm"] * r["n_frames"]
            ab += r["mpjpe_abs_mm"] * r["n_frames"]
            n += r["n_frames"]
            per_seq[seq] = {"ra_mm": float(r["mpjpe_ra_mm"]), "abs_mm": float(r["mpjpe_abs_mm"]),
                            "n_frames": int(r["n_frames"])}
    finally:
        model.ablate_evidence = False
    return {"ra_mm": ra / max(n, 1), "abs_mm": ab / max(n, 1), "n_frames": int(n), "per_seq": per_seq}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run dirs with a selection JSON")
    ap.add_argument("--split", default="val_core")
    ap.add_argument("--step-ms", type=int, default=50)
    ap.add_argument("--gate-mm", type=float, default=1.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = {}
    for r in a.runs:
        run = Path(r)
        model, cfg, sel = load_run(run, device)
        root = Path(cfg["DATA"]["ROOT"])
        seqs = sequences_for_split(root, a.split, splits_manifest(cfg))
        live = closed_loop(model, cfg, root, seqs, a.step_ms, device, ablate=False)
        abl = closed_loop(model, cfg, root, seqs, a.step_ms, device, ablate=True)
        drift = live["ra_mm"] - float(sel["mpjpe_ra_mm"])
        res[run.name] = {"ckpt": sel["ckpt"], "step": sel.get("step"),
                         "selection_ra_mm": float(sel["mpjpe_ra_mm"]), "reproduction_drift_mm": drift,
                         "live": live, "observations_zeroed": abl,
                         "degradation_mm": abl["ra_mm"] - live["ra_mm"],
                         "gate_mm": a.gate_mm, "gate_passed": bool(abl["ra_mm"] - live["ra_mm"] >= a.gate_mm)}
        print(f"{run.name} step {sel.get('step')}: live RA {live['ra_mm']:.2f} (selection {sel['mpjpe_ra_mm']:.2f}, "
              f"drift {drift:+.4f}) | observations zeroed RA {abl['ra_mm']:.2f} abs {abl['abs_mm']:.1f} "
              f"| degradation {abl['ra_mm'] - live['ra_mm']:+.2f} mm -> gate {'PASS' if res[run.name]['gate_passed'] else 'FAIL'}")
        del model
        torch.cuda.empty_cache()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
