#!/usr/bin/env python3
"""DT2: the deployment form (DeployTracker: GPU LNES + CUDA graph of forward + GPU AdaptiveFilter, one host sync per step)
against the recorded filter_eval evaluation of the same checkpoint and spec, for every seed of the main row. Closed loop on
zgz under the evalx protocol (one rng seeded 0 across the split's sequences); reports the overall RA of both and how many of
the 2590 steps are bit-identical. Used to justify that the "deployment form" row shares the accuracy cells of the
adaptive-filter row.

    python tools/dt/deploy_check.py --runs outputs/semkine/dt_dz_l3_s3407 outputs/semkine/dt_dz_l3_s3408 \
        --rec-dir outputs/dt2/filter_sweep --tag afad1f21_rn0.3-0.5-0.8_f0.8_t0.5 --out outputs/dt2/reports/deploy_check.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "model"), str(REPO / "tools" / "tracking")]
import deploy_fast as DF                                              # noqa: E402
import evalx as EX                                                    # noqa: E402
from config import load_config                                        # noqa: E402
from mano_layer import ManoLayer                                      # noqa: E402
from model import MNISTModel                                          # noqa: E402
from semkine.dataset import sequences_for_split                       # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--rec-dir", default=str(REPO / "outputs/dt2/filter_sweep"))
    ap.add_argument("--tag", default="afad1f21_rn0.3-0.5-0.8_f0.8_t0.5")
    ap.add_argument("--out", default=str(REPO / "outputs/dt2/reports/deploy_check.json"))
    a = ap.parse_args()
    dev = torch.device("cuda")
    out = {}
    for rd in a.runs:
        run = Path(rd)
        cfg = load_config(run / "config_resolved.yaml")
        ckpt, step, _ = EX.find_ckpt(run, "last")
        model = MNISTModel.load_from_checkpoint(str(ckpt), cfg=cfg, map_location=dev).to(dev).eval()
        mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(dev).eval()
        root = Path(cfg["DATA"]["ROOT"])
        seqs = sequences_for_split(root, "val_core", Path(cfg["DATA"]["SPLITS_MANIFEST"]))
        rec_p = Path(a.rec_dir) / run.name / f"{a.tag}.json"
        rec = json.loads(rec_p.read_text())
        npz = np.load(rec_p.with_suffix(".npz"))
        DF.enable_fast_render(model)
        tr = DF.DeployTracker(model, rec["filter"]["spec"], graph=True, scope="all")
        rng = np.random.default_rng(0)
        res = {s: DF.run_closed_loop(tr, cfg, root, d, s, rng) for s, d in seqs}
        ms = {s: EX.per_step_metrics(mano, r, dev) for s, r in res.items()}
        n = sum(len(m["mpjpe_ra_mm"]) for m in ms.values())
        ra = float(sum(m["mpjpe_ra_mm"].sum() for m in ms.values()) / n)
        same = sum(int((res[s]["pred"].view(np.int32) == npz[f"model|{s}|pred"].view(np.int32)).all(1).sum()) for s in res)
        rec_ra = rec["model"]["overall"]["mpjpe_ra_mm"]
        out[run.name] = {"ckpt": str(ckpt), "step": step, "spec": rec["filter"]["spec"], "deploy_ra": ra, "recorded_ra": rec_ra,
                         "abs_diff_mm": abs(ra - rec_ra), "steps_bit_identical": same, "steps": n,
                         "fallbacks": int(tr.n_fallback)}
        print(json.dumps({run.name: out[run.name]}), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
