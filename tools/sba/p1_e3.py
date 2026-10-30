#!/usr/bin/env python3
"""SBA P1 (docs/EXPERIMENT_RECORDS.md#sba-prereg-20261022 P1): single step from the exact GT state (E3), zero training.

Every protocol endpoint of zgz (XCL endpoint grid): start knot = GT at the window start (fixed), end knot initialised
at the start (hold), increment prior = the 50 ms training increment precision scaled by (50 / h)^2, the S0 model
(calibration.json), coarse-to-fine Fisher scoring (POOLS_TRACK). Horizons h = 50 ms (window = the protocol step) and
h = 200 ms (window = (end - 200, end], clipped to the run start). Scored like the XVM ladder: root-aligned MPJPE of
the end state against GT(end) and R_art (finger articulation increment error / zero-motion error); identity in the
same run.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python tools/sba/p1_e3.py run --seq zgz_local --h 50 --shard 0 --shards 2
    python tools/sba/p1_e3.py summarize
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sba import common as CM  # noqa: E402
from sba.common import SB, C  # noqa: E402

OUT = CM.OUT / "p1"


def run(seq, h, shard, shards):
    dev = CM.require_gpu()
    torch.set_grad_enabled(False)
    ctx = CM.Ctx(dev)
    model = CM.load_model()
    if json.loads(CM.CALIB.read_text()).get("kappa_mode") == "profiled":
        model.profile = True
    sd = CM.SeqData(seq, "val", dev)
    rows = CM.endpoint_rows(sd)
    idx = list(range(len(rows)))[shard::shards]
    out = {"i": [], "start": [], "end": [], "x": [], "n_events": [], "accepted": [], "sign": [], "s": []}
    t0 = time.time()
    for i in idx:
        rid, end, prev, first = rows[i]
        a = sd.run_start(end)
        start = prev if h == 50 else max(a, end - h)
        x0 = sd.gt(start)
        win = CM.window(ctx, sd, start, end, x0, x0)
        ts = time.time()
        if win.n_events == 0:
            x, acc, sg = x0, 0, 0
        else:
            ev = SB.Evaluator(ctx.geo, win, model, sd.betas, sd.K)
            res = SB.solve(ev, x0, x0, SB.Priors(inc_prec=ctx.inc_prec(end - start)), pools=CM.POOLS_TRACK)
            x, acc, sg = res.x1, res.accepted, res.sign
        out["i"].append(i); out["start"].append(start); out["end"].append(end)
        out["x"].append(x.cpu().numpy()); out["n_events"].append(win.n_events)
        out["accepted"].append(acc); out["sign"].append(sg); out["s"].append(time.time() - ts)
        if len(out["i"]) % 100 == 0:
            print(json.dumps({"seq": seq, "h": h, "shard": shard, "done": len(out["i"]), "of": len(idx),
                              "s": round(time.time() - t0)}), flush=True)
    path = OUT / "raw" / f"{seq}_h{h}__{shard:02d}of{shards:02d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in out.items()})
    CM.receipt(OUT / "raw" / f"{seq}_h{h}__{shard:02d}", {"probe": "P1", "seq": seq, "h": h, "shard": shard,
                                                         "model": model.__dict__, "wall_s": time.time() - t0},
               scripts=(__file__,))


def summarize():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = CM.Ctx(dev)
    from xvm.common import EF
    mano64 = EF.Mano64(dev)
    res = {"per_seq": {}}
    for seq in CM.ZGZ:
        sd = CM.SeqData(seq, "val", dev)
        betas64 = torch.tensor(sd.sq.betas, dtype=torch.float64, device=dev)[None]
        res["per_seq"][seq] = {}
        for h in (50, 200):
            files = sorted((OUT / "raw").glob(f"{seq}_h{h}__*of*.npz"))
            if not files:
                continue
            z = [np.load(f) for f in files]
            cat = {k: np.concatenate([q[k] for q in z]) for k in z[0].files}
            o = np.argsort(cat["i"])
            cat = {k: v[o] for k, v in cat.items()}
            pred = cat["x"].astype(np.float64)
            gt_end = np.asarray(sd.pos51[cat["end"]], np.float64)
            x_start = np.asarray(sd.pos51[cat["start"]], np.float64)
            ent = {"n": int(len(pred)), "solve_s_mean": float(cat["s"].mean()),
                   "accepted_mean": float(cat["accepted"].mean())}
            for name, P in (("sba", pred), ("identity", x_start)):
                sc = C.score_ra(ctx.mano, P, gt_end, sd.sq.betas, dev)
                ea, ez, _, _ = C.r_art_terms(mano64, torch.tensor(x_start, device=dev), torch.tensor(P, device=dev),
                                             torch.tensor(gt_end, device=dev), betas64)
                ent[name] = {"mpjpe_ra_mm": float(sc["mpjpe_ra_mm"].mean()), "wrist_mm": float(sc["wrist_mm"].mean()),
                             "R_art": float(ea.sum() / ez.sum())}
                inf = cat["n_events"] >= CM.MIN_EVENTS
                ent[name]["informative"] = {"n": int(inf.sum()),
                                            "mpjpe_ra_mm": float(sc["mpjpe_ra_mm"][inf].mean()) if inf.any() else None,
                                            "R_art": float(ea[torch.tensor(inf, device=dev)].sum()
                                                           / ez[torch.tensor(inf, device=dev)].sum()) if inf.any() else None}
            res["per_seq"][seq][f"h{h}"] = ent
            print(seq, h, json.dumps({k: ent[k] for k in ("sba", "identity")}), flush=True)
    loc = res["per_seq"].get("zgz_local", {})
    if "h50" in loc and "h200" in loc:
        m50 = loc["h50"]["sba"]["mpjpe_ra_mm"]
        r200 = loc["h200"]["sba"]["R_art"]
        idn = loc["h50"]["identity"]["mpjpe_ra_mm"]
        res["gate"] = {"pass": bool(m50 < idn and r200 <= 0.80), "falsified": bool(m50 > idn + 1.0),
                       "local_mpjpe_ra_h50": m50, "identity_h50": idn, "local_R_art_h200": r200}
    (OUT / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res.get("gate"), indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "summarize"))
    ap.add_argument("--seq", choices=CM.ZGZ)
    ap.add_argument("--h", type=int, choices=(50, 200), default=50)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.seq, a.h, a.shard, a.shards)
    else:
        summarize()


if __name__ == "__main__":
    main()
