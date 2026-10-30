#!/usr/bin/env python3
"""SBA main rows (docs/EXPERIMENT_RECORDS.md#sba-prereg-20261022 P3) in the format `tools/report_table.py` reads.

Accuracy: per seed from outputs/sba/p3/<arm>.npz (frame-weighted over zgz_global + zgz_local, as evalx);
"two_seed_mean" is the N-seed mean. Latency: mean wall time of one SBA step (window + solve, batch 1, eager) over
the first 200 protocol packets of lyq_local (training subject), plus the S39 network + filter step for the fused arm,
scaled by `evalx.latency_anchor` to the 1.75 ms anchor (the main-row convention). MACs: counted MANO FK evaluations
of the implementation as run (central-difference Jacobian over the trajectory samples, ~0.6 MMAC each) plus the
fused arm's network count; the finite-difference Jacobian dominates and is an implementation choice (an analytic
Jacobian needs one FK per sample). Params: SBA has no learned parameters (4 calibration scalars and the training-GT
prior statistics); the fused arm carries S39's.

    python tools/sba/make_row.py --arm sba_standalone --npz outputs/sba/p3/standalone_w0.05_i{seed}.npz --seeds 0,1,2
    python tools/sba/make_row.py --arm sba_fused_s39 --npz outputs/sba/p3/fused_s{seed}.npz --seeds 3407,3408,3409 \
        --base-row outputs/semkine/dt3_dz_l3_w200cfg_main_row.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "tools"), str(REPO / "tools" / "dt"), str(REPO), str(REPO / "model")]
from sba import common as CM  # noqa: E402
from sba.common import SB, EF  # noqa: E402

KEYS = ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm", "root_rot_deg", "transl_mm")
MANO_FK_MACS = 0.6e6


def seed_metrics(npz):
    z = np.load(npz)
    out = {}
    for part, seqs in (("overall", ("zgz_global", "zgz_local")), ("local", ("zgz_local",)), ("global", ("zgz_global",))):
        n = sum(len(z[f"{s}|mpjpe_ra_mm"]) for s in seqs)
        out[part] = {k: float(sum(z[f"{s}|{k}"].sum() for s in seqs) / n) for k in KEYS}
    return out


@torch.no_grad()
def step_latency(dev, n_packets=200):
    """Mean ms of one standalone SBA step (window + solve) and the counted FK evaluations per step."""
    import filter_eval as FE
    ctx = CM.Ctx(dev)
    model = CM.load_model()
    sd = CM.SeqData("lyq_local", "train", dev)
    rows = CM.endpoint_rows(sd)[:n_packets]
    counter = {"fk": 0}
    orig = ctx.geo.fk

    def fk(params, betas, chunk=4096):
        counter["fk"] += int(params.shape[0])
        return orig(params, betas, chunk)
    ctx.geo.fk = fk
    times = []
    for (rid, end, prev, first) in rows:
        x = sd.gt(prev)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        win = CM.window(ctx, sd, prev, end, x, x)
        if win.n_events:
            ev = SB.Evaluator(ctx.geo, win, model, sd.betas, sd.K)
            pri = SB.Priors(inc_prec=ctx.inc_prec(end - prev), start_mean=x, start_prec=CM.sig_prec(CM.STATE_SIG, dev))
            SB.solve(ev, x, x, pri, pools=CM.POOLS_TRACK)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return float(np.mean(times[10:])), counter["fk"] / len(rows), FE.EX.latency_anchor(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--npz", required=True, help="path with {seed}")
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--base-row", default=None, help="network main row (fused arm)")
    a = ap.parse_args()
    dev = CM.require_gpu()
    seeds = [s for s in a.seeds.split(",")]
    per_seed = {s: seed_metrics(a.npz.format(seed=s)) for s in seeds}
    mean = {part: {k: float(np.mean([per_seed[s][part][k] for s in seeds])) for k in KEYS}
            for part in ("overall", "local", "global")}
    ms, fk_per_step, anchor = step_latency(dev)
    base = json.loads(Path(a.base_row).read_text()) if a.base_row else None
    raw = ms
    # each part scaled by the anchor of its own measuring session, then added
    scaled = ms * 1.75 / anchor + (base["latency_ms_scaled_full1p75"] if base else 0.0)
    macs = fk_per_step * MANO_FK_MACS + (base["macs_forward_packet"] if base else 0.0)
    row = {"arm": a.arm, "split": "val_core", "ckpt_rule": "zero-training" if not base else "last",
           "n_seeds": len(seeds), "two_seed_mean": mean, "per_seed": per_seed,
           "latency_ms_raw_sba_step": raw, "anchor_full_raw_ms": anchor, "latency_ms_scaled_full1p75": scaled,
           "latency_packets": 200, "macs_forward_packet": macs,
           "macs_note": f"counted MANO FK of the SBA step ({fk_per_step:.0f} per step, central-difference Jacobian)"
                        + (" + network thop" if base else ""),
           "params_total": base["params_total"] if base else 0,
           "seeds_are": "initial-noise seeds" if not base else "S39 model seeds",
           "note": "SBA zero-training closed loop; docs/EXPERIMENT_RECORDS.md#sba-prereg-20261022"}
    out = REPO / "outputs" / "semkine" / f"{a.arm}_main_row.json"
    if out.exists():
        raise SystemExit(f"{out} exists; preserve it")
    out.write_text(json.dumps(row, indent=1))
    print(out, json.dumps({k: round(v["mpjpe_ra_mm"], 3) for k, v in mean.items()}),
          f"latency {row['latency_ms_scaled_full1p75']:.2f} ms, MACs {macs / 1e9:.2f} G")


if __name__ == "__main__":
    main()
