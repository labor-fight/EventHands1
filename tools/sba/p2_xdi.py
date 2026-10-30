#!/usr/bin/env python3
"""SBA P2 (docs/EXPERIMENT_RECORDS.md#sba-prereg-20261022 P2): XDI's four Phase-0 gates with the SBA step in place of
the one-sided contour cost (docs/EXPERIMENT_RECORDS.md#xdi-20261014, tools/xdi/feasibility.py), zero training.

Per zgz packet (outputs/xdl/packet_bank.json) and XDI perturbation of the GT previous state (x_in): one SBA solve on
the packet's real events with the start knot free under XDI's own step-prior scales (0.05 m, 0.3 rad, 0.3 rad) around
x_in, end knot initialised at x_in, the 50 ms increment prior, wide coarse-to-fine schedule (POOLS_WIDE). The output
is the end knot. Wrong content: z+-50 with the events of a donor packet >= 40 mm / 15 deg away (XDI's same-sequence
far packet is not a valid control on the near-static zgz_local). Scoring = XDI's summarize: translation on the 51-D
translation parameter, root by geodesic angle, both against GT(end).

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python tools/sba/p2_xdi.py run --seq zgz_local --shard 0 --shards 2
    python tools/sba/p2_xdi.py summarize
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
from sba.common import SB  # noqa: E402
from semkine.lie import so3_exp, so3_log  # noqa: E402

OUT = CM.OUT / "p2"
NAMES = list(CM.PERTURB)
WRONG = ["z+50", "z-50"]


def run(seq, shard, shards):
    dev = CM.require_gpu()
    torch.set_grad_enabled(False)
    ctx = CM.Ctx(dev)
    model = CM.load_model()
    if json.loads(CM.CALIB.read_text()).get("kappa_mode") == "profiled":
        model.profile = True
    sd = CM.SeqData(seq, "val", dev)
    rows = CM.bank_rows(CM.bank(), seq)
    idx = list(range(len(rows)))[shard::shards]
    prec0 = CM.sig_prec(CM.XDI_SIG, dev)
    rec = {k: [] for k in ("i", "x_in", "x_out", "gt_prev", "gt_next", "wrong_in", "wrong_out", "n_events", "has_donor")}
    t0 = time.time()

    def step(win, xin):
        if win.n_events == 0:
            return xin
        ev = SB.Evaluator(ctx.geo, win, model, sd.betas, sd.K)
        pri = SB.Priors(inc_prec=ctx.inc_prec(50), start_mean=xin, start_prec=prec0)
        return SB.solve(ev, xin, xin, pri, pools=CM.POOLS_WIDE).x1

    for i in idx:
        row = rows[i]
        prev, end = int(row["previous_end_ms"]), int(row["end_ms"])
        gp, gn = sd.gt(prev), sd.gt(end)
        xs, outs = [], []
        n_ev = None
        for name in NAMES:
            kind, val = CM.PERTURB[name]
            xin = CM.perturb(ctx, gp, kind, val, i % 15, sd.betas)
            win = CM.window(ctx, sd, prev, end, xin, xin)
            n_ev = win.n_events
            xs.append(xin.cpu().numpy()); outs.append(step(win, xin).cpu().numpy())
        dr = CM.donor_row(ctx, sd, rows, i, sd.betas)
        wi, wo = [], []
        for name in WRONG:
            kind, val = CM.PERTURB[name]
            xin = CM.perturb(ctx, gp, kind, val, i % 15, sd.betas)
            if dr is None:
                wi.append(xin.cpu().numpy()); wo.append(xin.cpu().numpy())
                continue
            evms = CM.retime(sd.events(int(dr["previous_end_ms"]), int(dr["end_ms"])), int(dr["previous_end_ms"]),
                             prev, end)
            win = CM.window(ctx, sd, prev, end, xin, xin, events=evms)
            wi.append(xin.cpu().numpy()); wo.append(step(win, xin).cpu().numpy())
        rec["i"].append(i); rec["x_in"].append(np.stack(xs)); rec["x_out"].append(np.stack(outs))
        rec["gt_prev"].append(gp.cpu().numpy()); rec["gt_next"].append(gn.cpu().numpy())
        rec["wrong_in"].append(np.stack(wi)); rec["wrong_out"].append(np.stack(wo))
        rec["n_events"].append(n_ev); rec["has_donor"].append(dr is not None)
        if len(rec["i"]) % 25 == 0:
            print(json.dumps({"seq": seq, "shard": shard, "done": len(rec["i"]), "of": len(idx),
                              "s": round(time.time() - t0)}), flush=True)
    path = OUT / "raw" / f"{seq}__{shard:02d}of{shards:02d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in rec.items()}, names=np.array(NAMES))
    CM.receipt(OUT / "raw" / f"{seq}__{shard:02d}", {"probe": "P2", "seq": seq, "shard": shard,
                                                    "wall_s": time.time() - t0}, scripts=(__file__,))


def summarize():
    res = {"domains": {}, "gates": {}}
    for seq in CM.ZGZ:
        files = sorted((OUT / "raw").glob(f"{seq}__*of*.npz"))
        if not files:
            continue
        z = [np.load(f) for f in files]
        cat = {k: np.concatenate([q[k] for q in z]) for k in ("x_in", "x_out", "gt_prev", "gt_next", "wrong_in",
                                                              "wrong_out", "n_events", "has_donor")}
        gt = cat["gt_next"][:, None, :]

        def tr(x):
            return np.linalg.norm((x[..., :3] - gt[..., :3]) * 1000., axis=-1)

        def geo(x):
            a = torch.tensor(x[..., 3:6].reshape(-1, 3), dtype=torch.float64)
            b = torch.tensor(np.broadcast_to(gt[..., 3:6], x[..., 3:6].shape).reshape(-1, 3), dtype=torch.float64)
            return (so3_log(so3_exp(a) @ so3_exp(b).transpose(1, 2)).norm(dim=-1).numpy() * 180 / np.pi).reshape(x.shape[:-1])

        xin, xout = cat["x_in"], cat["x_out"]
        tin, tout, gin, gout = tr(xin), tr(xout), geo(xin), geo(xout)
        hold = np.linalg.norm((cat["gt_prev"][:, :3] - cat["gt_next"][:, :3]) * 1000., axis=-1)
        c = NAMES.index("clean")
        r = {"clean": {"hold_wrist_mm": float(hold.mean()), "after_wrist_mm": float(tout[:, c].mean()),
                       "increase_mm": float(tout[:, c].mean() - hold.mean()),
                       "median_abs_dt_mm": float(np.median(np.linalg.norm((xout[:, c, :3] - xin[:, c, :3]) * 1000., axis=-1))),
                       "root_deg_before": float(gin[:, c].mean()), "root_deg_after": float(gout[:, c].mean())}}
        per = {}
        for i, n in enumerate(NAMES):
            if n == "clean":
                continue
            if n[0] in "xyz":
                rec = 1 - tout[:, i] / np.maximum(tin[:, i], 1e-9)
            elif n[0] == "r":
                rec = 1 - gout[:, i] / np.maximum(gin[:, i], 1e-9)
            else:
                rec = None
            r[n] = {"wrist_mm_before": float(tin[:, i].mean()), "wrist_mm_after": float(tout[:, i].mean()),
                    "root_deg_before": float(gin[:, i].mean()), "root_deg_after": float(gout[:, i].mean())}
            if rec is not None:
                r[n]["median_recovery"] = float(np.median(rec))
                per[n] = rec
        r10 = np.median(np.stack([per[n] for n in NAMES if n[0] == "r" and n.endswith("10")]), axis=0)
        r["root10_median_recovery"] = float(np.median(r10))
        hd = cat["has_donor"].astype(bool)
        tw_in, tw_out = tr(cat["wrong_in"]), tr(cat["wrong_out"])
        wrong = {n: float(np.median((1 - tw_out[:, k] / np.maximum(tw_in[:, k], 1e-9))[hd])) if hd.any() else None
                 for k, n in enumerate(WRONG)}
        inf = cat["n_events"] >= CM.MIN_EVENTS
        strata = {"informative_n": int(inf.sum()),
                  "z+50_inf": float(np.median((1 - tout[:, NAMES.index("z+50")] / tin[:, NAMES.index("z+50")])[inf])) if inf.any() else None,
                  "z-50_inf": float(np.median((1 - tout[:, NAMES.index("z-50")] / tin[:, NAMES.index("z-50")])[inf])) if inf.any() else None,
                  "root10_inf": float(np.median(r10[inf])) if inf.any() else None}
        res["domains"][seq] = {"n_packets": int(len(hold)), "result": r, "wrong_content": wrong,
                               "n_with_donor": int(hd.sum()), "informative": strata}
        g = {"P0a": r["clean"]["increase_mm"] <= 3.0 and r["clean"]["median_abs_dt_mm"] <= 5.0,
             "P0b": r["z+50"]["median_recovery"] >= 0.5 and r["z-50"]["median_recovery"] >= 0.5,
             "P0c": r["root10_median_recovery"] >= 0.5,
             "P0d": all(v is not None and v < 0.2 for v in wrong.values())}
        res["gates"][seq] = g
    res["stop"] = any(sum(not v for v in g.values()) >= 2 for g in res["gates"].values())
    (OUT / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"gates": res["gates"], "stop": res["stop"]}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "summarize"))
    ap.add_argument("--seq", choices=CM.ZGZ)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.seq, a.shard, a.shards)
    else:
        summarize()


if __name__ == "__main__":
    main()
