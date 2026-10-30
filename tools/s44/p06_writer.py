#!/usr/bin/env python3
"""S44 P0.6 (docs/EXPERIMENT_RECORDS.md#s44-prereg-20261019): the writer contract on zgz.

(a) prefix equivalence: `tests/test_s44_evflow.py::test_sae_prefix_equivalence` (unit test; bit for bit).
(b) fraction of active SAE entries (latest event within the last 300 ms) whose value did not change between
    consecutive queries, hop 50 and 10 ms, on every 4th protocol step.
(c) owner stability: for the measurement pixels of step k, the fraction whose first (nearest visible) owner vertex
    differs between the closed-loop state before and after one real S39 update (seed3407 control trajectory).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s44 import evflow as EF  # noqa: E402


@torch.no_grad()
def main():
    dev = torch.device("cuda")
    mano = EF.Mano64(dev)
    ctl = np.load(EF.REPO / "outputs/s44/p10/s39_s3407_control.npz")
    out = {}
    for s, d in EF.split_seqs("val_core"):
        sq = EF.Seq(s, d)
        intr = sq.intrinsics()
        betas = torch.tensor(sq.betas, dtype=torch.float64, device=dev)[None]
        ends, runs, prev, pred = ctl[f"{s}|end"], ctl[f"{s}|run"], ctl[f"{s}|prev"], ctl[f"{s}|pred"]
        unch = {50: [], 10: []}
        flip = []
        for i, end in enumerate(ends):
            end = int(end)
            ra = int(sq.runs[int(runs[i])][0])
            if end - EF.STEP < ra:
                continue
            if i % 4 == 0:
                for hop in (50, 10):
                    lo = max(end - EF.W_MEM + 1, ra)
                    T1, _ = EF.build_sae(sq, lo, end, dev)
                    T0, _ = EF.build_sae(sq, max(end - hop - EF.W_MEM + 1, ra), end - hop, dev)
                    act = (T1 >= lo * 1000) & (T1 >= 0)
                    if int(act.sum()):
                        unch[hop].append(float(((T1 == T0) & act).sum() / act.sum()))
            meas = EF.step_measurements(sq, ra, end, 2, 30.0, dev)
            if meas["n_valid"] == 0:
                continue
            ids = []
            for st in (prev[i], pred[i]):
                v, _ = mano.fk(torch.tensor(st, dtype=torch.float64, device=dev)[None], betas)
                uv, z = mano.project(v[0], intr)
                o, w = EF.owners_deploy(meas["px"], uv, EF.visible(uv, z))
                ids.append(torch.where(w[:, 0] > 0, o[:, 0], torch.full_like(o[:, 0], -1)))
            both = (ids[0] >= 0) | (ids[1] >= 0)
            if int(both.sum()):
                flip.append(float((ids[0] != ids[1])[both].float().mean()))
        out[s] = {"sae_unchanged_frac_hop50": float(np.mean(unch[50])), "sae_unchanged_frac_hop10": float(np.mean(unch[10])),
                  "owner_change_frac_one_update": float(np.mean(flip)), "n_steps_owner": len(flip)}
        print(s, json.dumps(out[s]), flush=True)
    (EF.REPO / "outputs/s44/p06_writer.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
