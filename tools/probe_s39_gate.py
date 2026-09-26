#!/usr/bin/env python3
"""S39 mechanism gate: the selected checkpoint, closed loop on the selection protocol (val_core =
zgz, 50 ms, rng seed 0), live and with one input silenced.

    --ablate covmap    `model.ablate_covmap = True`: the surface coverage map alone is zeroed, the
                       per-joint evidence and the pooled vector stay -- the S39 gate
                       (docs/S39_COVMAP_PREREG.md section 3: degradation >= 1.5 mm recursive RA)
    --ablate evidence  `model.ablate_evidence = True`: every routed evidence (and the map) zeroed

    CUDA_VISIBLE_DEVICES=1 python tools/probe_s39_gate.py --ablate covmap \\
        --runs outputs/semkine/s39_covmap_s3407 outputs/semkine/s39_covmap_s3408 \\
        --out outputs/semkine/probe_s39_gate.json
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
def closed_loop(model, cfg, root, seqs, step_ms, device, ablate: str | None) -> dict:
    flag = {"covmap": "ablate_covmap", "evidence": "ablate_evidence", None: None}[ablate]
    if flag is not None and not hasattr(model, flag):
        raise SystemExit(f"this arm has no `{flag}` hook")
    if flag:
        setattr(model, flag, True)
    try:
        ra = ab = n = 0.0
        per_seq = {}
        rng = np.random.default_rng(0)     # one rng for the pass, as select_checkpoint.py does
        for seq, d in seqs:
            r = ET.track_sequence(model, model.mano, cfg, root, d, seq, step_ms, device, rng, 1.0, 1.0, None)
            if r is None:
                continue
            ra += r["mpjpe_ra_mm"] * r["n_frames"]
            ab += r["mpjpe_abs_mm"] * r["n_frames"]
            n += r["n_frames"]
            per_seq[seq] = {"ra_mm": float(r["mpjpe_ra_mm"]), "abs_mm": float(r["mpjpe_abs_mm"]), "n_frames": int(r["n_frames"])}
    finally:
        if flag:
            setattr(model, flag, False)
    return {"ra_mm": ra / max(n, 1), "abs_mm": ab / max(n, 1), "n_frames": int(n), "per_seq": per_seq}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--ablate", choices=("covmap", "evidence"), default="covmap")
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
        live = closed_loop(model, cfg, root, seqs, a.step_ms, device, None)
        abl = closed_loop(model, cfg, root, seqs, a.step_ms, device, a.ablate)
        drift = live["ra_mm"] - float(sel["mpjpe_ra_mm"])
        res[run.name] = {"ckpt": sel["ckpt"], "step": sel.get("step"), "ablate": a.ablate,
                         "selection_ra_mm": float(sel["mpjpe_ra_mm"]), "reproduction_drift_mm": drift,
                         "live": live, "ablated": abl, "degradation_mm": abl["ra_mm"] - live["ra_mm"],
                         "gate_mm": a.gate_mm, "gate_passed": bool(abl["ra_mm"] - live["ra_mm"] >= a.gate_mm)}
        print(f"{run.name} step {sel.get('step')}: live RA {live['ra_mm']:.2f} (selection {sel['mpjpe_ra_mm']:.2f}, drift {drift:+.4f}) "
              f"| {a.ablate} zeroed RA {abl['ra_mm']:.2f} abs {abl['abs_mm']:.1f} | degradation {abl['ra_mm'] - live['ra_mm']:+.2f} mm "
              f"-> gate {'PASS' if res[run.name]['gate_passed'] else 'FAIL'}", flush=True)
        del model
        torch.cuda.empty_cache()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
