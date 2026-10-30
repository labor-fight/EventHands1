#!/usr/bin/env python3
"""XVM P0d: S39-stack closed loop with a post-filter update gain (docs/EXPERIMENT_RECORDS.md#xvm-prereg-20261021 §1).

The loop is `tools/s44/p10_replay.py` in mode control (W200 clipped to the segment start, ramp1000 AdaptiveFilter,
Count50, protocol initial noise from one rng seeded 0 over val_core in order); g = 1 reproduces it exactly. Per step

    out = AdaptiveFilter(net(LNES, prev), prev);   x_k = prev + g * (out - prev)   (variant all: 51-D; fingers: 6:51)

Diagnostic only (gains are scanned on zgz); no gain becomes a candidate.

    python tools/xvm/p0d_gain.py --seed 3407 --out outputs/xvm/s1/p0d/s3407.json
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
from xvm import common as C  # noqa: E402
from s44 import evflow as EF  # noqa: E402
import filter_eval as FE  # noqa: E402
from mano_layer import ManoLayer  # noqa: E402
from semkine import eval_track as ET  # noqa: E402
from semkine import events as EV  # noqa: E402
from semkine.anchored import AdaptiveFilter  # noqa: E402
from semkine.async_tracker import _Fixed  # noqa: E402

EX = FE.EX
SPEC = REPO / "outputs" / "dt3" / "gate_specs" / "ramp1000.json"
WIN = 200
GAINS = (1.0, 1.1, 1.2, 1.3, 1.4, 1.5)


@torch.no_grad()
def closed_loop(model, filt, cfg, dev, mano32, gain, variant, cache):
    ev_ch = EV.event_channels(cfg)
    rng = np.random.default_rng(0)
    summ = {}
    for s, d in EF.split_seqs("val_core"):
        events, offsets, aux, pos51 = cache[s]
        betas = torch.tensor(aux["betas"], dtype=torch.float32, device=dev).view(1, -1)
        camera_K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=dev).view(1, 3, 3)
        model.set_hand_context(betas, camera_K)
        runs = np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)
        rec = {k: [] for k in ("pred", "prev", "gt", "end", "run", "elapsed", "count", "win")}
        for run_id, (ra, rb) in enumerate(runs):
            ends = np.arange(ra + EF.STEP - 1, rb, EF.STEP, dtype=np.int64)
            if not len(ends):
                continue
            prev = pos51[ra].copy() + ET.sample_init_noise(cfg, rng, 1.0)
            prev_t = torch.from_numpy(prev).view(1, -1).to(dev)
            filt.reset_state(prev_t)
            for end in ends:
                end = int(end)
                n_ev = int(offsets[end + 1] - offsets[end - EF.STEP + 1])
                w = EX.window_of(offsets, end, "fixed", WIN, 0, 300, floor=int(ra))
                x = torch.from_numpy(ET.build_lnes(events, offsets, end, w, ev_ch)).unsqueeze(0).to(dev)
                rec["prev"].append(prev_t.cpu().numpy()[0])
                z = model(x, prev_t)
                filt.trk_model.z = z
                out = filt(x, prev_t, n_events=n_ev)
                if gain != 1.0:
                    upd = out - prev_t
                    if variant == "fingers":
                        upd[:, :6] = 0
                        out = out.clone()
                        out[:, 6:] = prev_t[:, 6:] + gain * upd[:, 6:]
                    else:
                        out = prev_t + gain * upd
                prev_t = out
                rec["pred"].append(out.cpu().numpy()[0])
                rec["gt"].append(pos51[end])
                rec["end"].append(end)
                rec["run"].append(run_id)
                rec["elapsed"].append(end - int(ra))
                rec["count"].append(n_ev)
                rec["win"].append(w)
        r = {k: np.asarray(v) for k, v in rec.items()}
        r["pred"], r["gt"], r["prev"] = r["pred"].astype(np.float32), r["gt"].astype(np.float32), r["prev"].astype(np.float32)
        r["betas"] = aux["betas"]
        m = EX.per_step_metrics(mano32, r, dev)
        summ[s] = {"n": int(len(r["end"])), **{k: float(v.mean()) for k, v in m.items()}}
    n = sum(v["n"] for v in summ.values())
    summ["overall"] = {k: sum(summ[s][k] * summ[s]["n"] for s in C.ZGZ) / n
                       for k in ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "root_rot_deg", "transl_mm")}
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, choices=(3407, 3408, 3409))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = C.require_gpu()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; preserve it")
    cfg, model, ckpt, step, _ = FE.build_run(REPO / f"outputs/semkine/dt_dz_l3_s{a.seed}", "last", dev)
    spec = json.loads(SPEC.read_text())
    filt = AdaptiveFilter.from_spec(_Fixed(), spec).to(dev).eval()
    mano32 = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(dev).eval()
    cache = {s: ET.load_sequence(EF.DATA_ROOT, d, s) for s, d in EF.split_seqs("val_core")}
    res = {"seed": a.seed, "ckpt": str(ckpt), "step": step, "runs": {}}
    t0 = time.time()
    for variant in ("all", "fingers"):
        for g in GAINS:
            if g == 1.0 and variant == "fingers":
                continue
            key = f"{variant}_g{g}" if g != 1.0 else "control"
            res["runs"][key] = closed_loop(model, filt, cfg, dev, mano32, g, variant, cache)
            o = res["runs"][key]
            print(a.seed, key, f"local {o['zgz_local']['mpjpe_ra_mm']:.3f} global {o['zgz_global']['mpjpe_ra_mm']:.3f} "
                  f"RA {o['overall']['mpjpe_ra_mm']:.3f} ({time.time() - t0:.0f} s)", flush=True)
    res["wall_s"] = time.time() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    C.receipt(out.parent, {"probe": "p0d", "seeds_done": sorted(int(p.stem[1:]) for p in out.parent.glob("s*.json"))},
              scripts=(__file__,))


if __name__ == "__main__":
    main()
