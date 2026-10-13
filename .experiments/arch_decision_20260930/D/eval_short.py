#!/usr/bin/env python3
"""Closed-loop zgz RA of the matched W32 / W256 short-train checkpoints (fixed steps, no selection).

Same loop, inits (rng 0) and scoring as probe_arch.py H0, which reproduces the S37 main row bitwise.
    CUDA_VISIBLE_DEVICES=6 python .experiments/arch_decision_20260930/D/eval_short.py --steps 200,400
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "B"))
import probe_arch as PA                                                                 # noqa: E402

ARMS = ("arch0930_w32_s3407", "arch0930_w256_s3407")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="200,400")
    a = ap.parse_args()
    device = torch.device("cuda")
    res = {}
    for arm in ARMS:
        for step in [int(x) for x in a.steps.split(",")]:
            ck = f"outputs/semkine/{arm}/{arm}-step={step}.ckpt"
            if not (PA.REPO / ck).exists():
                res[f"{arm}@{step}"] = "missing"
                continue
            m, cfg, _ = PA.load_run(f"outputs/semkine/{arm}:{ck}", device)
            m.eval()
            root = Path(cfg["DATA"]["ROOT"])
            zgz = [PA.PSD.SeqNC(root, d, s, device)
                   for s, d in PA.sequences_for_split(root, "val_core", PA.splits_manifest(cfg))]
            rng0 = np.random.default_rng(0)
            inits = {s.name: [PA.ET.sample_init_noise(cfg, rng0, 1.0) for _ in s.runs] for s in zgz}
            rec, n_tot, ra_w = {"window": int(m.event_encoder.window)}, 0, 0.0
            for s in zgz:
                m.set_hand_context(s.betas, s.K)
                preds, _, gts = PA.loop_variants(m, s, inits[s.name], device, [("base", None, 0.0)])
                d = PA.summ(PA.decomp(m, preds[0], gts, s.betas))
                rec[s.name] = d
                n = sum(len(e) for _, e in s.runs)
                n_tot += n
                ra_w += d["ra"] * n
            rec["frame_weighted_ra"] = ra_w / n_tot
            res[f"{arm}@{step}"] = rec
            print(arm, step, PA.r3(rec), flush=True)
            del m
            torch.cuda.empty_cache()
    (HERE / f"eval_short_{time.strftime('%H%M%S')}.json").write_text(json.dumps(PA.r3(res), indent=1))


if __name__ == "__main__":
    main()
