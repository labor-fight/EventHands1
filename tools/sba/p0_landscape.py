#!/usr/bin/env python3
"""SBA P0 (docs/EXPERIMENT_RECORDS.md#sba-prereg-20261022 P0): zero-training likelihood landscape on XDI's packets.

Per packet (outputs/xdl/packet_bank.json rows of zgz_local / zgz_global; the 576 E2 'check' packets of the training
subjects), the GT trajectory GT(prev_end) -> GT(end) against displaced hypotheses, all scored on the packet's real
events (prev_end, end]:

    A  absolute: both knots displaced by the same XDI perturbation (translation z+-50 / x+-20 / y+-20 mm, root +-10 deg
       about the camera axes with the wrist fixed, finger joint (index mod 15) +-20 deg) and a z-line -60..+60 mm
    M  motion: start knot GT, end knot = hold (start) or GT + a step-scale perturbation (t +-5 mm, root +-3 deg,
       finger +-5 deg)
    D  donor control: condition A with the events of a packet >= 40 mm / 15 deg away (must not prefer the GT)

Three likelihood variants from the same swept areas: calibrated kappa (outputs/sba/s0/calibration.json), kappa
profiled per hypothesis, and one-sided (sum N log lambda, no -lambda: the XDI-type cost without negative evidence).

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python tools/sba/p0_landscape.py run --domain zgz_local --shard 0 --shards 2
    python tools/sba/p0_landscape.py summarize
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sba import common as CM  # noqa: E402
from sba.common import SB  # noqa: E402

OUT = CM.OUT / "p0"
ZLINE = [k * 5 for k in range(-12, 13)]
A_SET = [k for k in CM.PERTURB if not (k.startswith("r") and k[2:] in ("+20", "-20"))]
M_SET = {"hold": None, "t_x+5": ("t", (0, .005)), "t_x-5": ("t", (0, -.005)), "t_y+5": ("t", (1, .005)),
         "t_y-5": ("t", (1, -.005)), "t_z+5": ("t", (2, .005)), "t_z-5": ("t", (2, -.005)),
         "r_x+3": ("root", (0, 3)), "r_x-3": ("root", (0, -3)), "r_y+3": ("root", (1, 3)), "r_y-3": ("root", (1, -3)),
         "r_z+3": ("root", (2, 3)), "r_z-3": ("root", (2, -3)), "f+5": ("finger", 5), "f-5": ("finger", -5)}
D_SET = ["clean"] + CM.ROOT10 + ["f+20", "f-20"]
BINS = (0, 300, 1000, 3000, 10 ** 9)


def packets(domain):
    b = CM.bank()
    if domain in CM.ZGZ:
        rows = CM.bank_rows(b, domain)
        return [(domain, "val", i, r, rows) for i, r in enumerate(rows)]
    e2 = json.loads((CM.REPO / "outputs/xde/e2/protocol.json").read_text())["selected_packets"]
    return [(r["sequence"], r["directory"], int(r["row"]["index"]), r["row"], CM.bank_rows(b, r["sequence"]))
            for r in e2 if r["split"] == "check"]


def score(ev, win, model, a, b):
    A = ev.areas(a[None], b[None], ev.plan_at(a, b))
    lam = SB.rates(A, win, model)
    N = SB.counts(win, model)
    fixed = float(SB.loglik(lam, N)[0])
    onesided = float((N[None] * torch.log(lam.clamp(min=1e-12))).flatten(1).sum(1)[0])
    k = SB.profile_kappa(A, win, model)
    m2 = replace(model, kappa_sil=model.kappa_sil * float(k[0]), kappa_int=model.kappa_int * float(k[0]))
    prof = float(SB.loglik(SB.rates(A, win, m2), N)[0])
    return fixed, prof, onesided


def run(domain, shard, shards):
    dev = CM.require_gpu()
    torch.set_grad_enabled(False)
    ctx = CM.Ctx(dev)
    model = replace(CM.load_model(), signed=False)          # landscape on merged counts (polarity: see P1-P3)
    items = packets(domain)[shard::shards]
    cache = {}
    recs = []
    t0 = time.time()
    for seq, d, index, row, rows in items:
        if seq not in cache:
            cache = {seq: CM.SeqData(seq, d, dev)}
        sd = cache[seq]
        prev, end = int(row["previous_end_ms"]), int(row["end_ms"])
        x0, x1 = sd.gt(prev), sd.gt(end)
        win = CM.window(ctx, sd, prev, end, x0, x1)
        ev = SB.Evaluator(ctx.geo, win, model, sd.betas, sd.K)
        joint = index % 15
        rec = {"seq": seq, "index": index, "end": end, "n_events": win.n_events, "n_crop": win.n_crop,
               "A": {}, "M": {}, "D": None}
        if win.n_events > 0:
            for name in A_SET:
                kind, val = CM.PERTURB[name]
                a = CM.perturb(ctx, x0, kind, val, joint, sd.betas)
                b = CM.perturb(ctx, x1, kind, val, joint, sd.betas)
                rec["A"][name] = score(ev, win, model, a, b)
            for z in ZLINE:
                a = CM.perturb(ctx, x0, "t", (2, z / 1000), joint, sd.betas)
                b = CM.perturb(ctx, x1, "t", (2, z / 1000), joint, sd.betas)
                rec["A"][f"zl{z:+d}"] = score(ev, win, model, a, b)
            rec["M"]["clean"] = rec["A"]["clean"]
            for name, spec in M_SET.items():
                b = x0 if spec is None else CM.perturb(ctx, x1, spec[0], spec[1], joint, sd.betas)
                rec["M"][name] = score(ev, win, model, x0, b)
            dr = CM.donor_row(ctx, sd, rows, index, sd.betas)
            if dr is not None:
                evms = CM.retime(sd.events(int(dr["previous_end_ms"]), int(dr["end_ms"])),
                                 int(dr["previous_end_ms"]), prev, end)
                if len(evms[1]) > 0:
                    wd = CM.window(ctx, sd, prev, end, x0, x1, events=evms)
                    evd = SB.Evaluator(ctx.geo, wd, model, sd.betas, sd.K)
                    rec["D"] = {"donor_end": int(dr["end_ms"]), "n_events": wd.n_events, "ll": {}}
                    for name in D_SET:
                        kind, val = CM.PERTURB[name]
                        a = CM.perturb(ctx, x0, kind, val, joint, sd.betas)
                        b = CM.perturb(ctx, x1, kind, val, joint, sd.betas)
                        rec["D"]["ll"][name] = score(evd, wd, model, a, b)
        recs.append(rec)
        if len(recs) % 50 == 0:
            print(json.dumps({"domain": domain, "shard": shard, "done": len(recs), "of": len(items),
                              "s": round(time.time() - t0)}), flush=True)
    path = OUT / "raw" / f"{domain}__{shard:02d}of{shards:02d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"records": recs, "wall_s": time.time() - t0, "model": model.__dict__}))
    CM.receipt(OUT / "raw" / f"{domain}__{shard:02d}", {"probe": "P0", "domain": domain, "shard": shard},
               scripts=(__file__,))


def pref(recs, key, names, variant=0):
    v = [r[key]["clean"][variant] > r[key][n][variant] for r in recs for n in names if r[key]]
    return float(np.mean(v)) if v else None


def zstar(recs, variant=0):
    z = [abs(ZLINE[int(np.argmax([r["A"][f"zl{q:+d}"][variant] for q in ZLINE]))]) for r in recs if r["A"]]
    return float(np.median(z)) if z else None


def summary_of(recs):
    out = {"n": len(recs)}
    for vi, vn in enumerate(("fixed", "profiled", "onesided")):
        out[vn] = {"A_root10": pref(recs, "A", CM.ROOT10, vi), "A_finger20": pref(recs, "A", ["f+20", "f-20"], vi),
                   "A_translation": pref(recs, "A", ["z+50", "z-50", "x+20", "x-20", "y+20", "y-20"], vi),
                   "zline_abs_argmax_median_mm": zstar(recs, vi),
                   "M_vs_hold": pref(recs, "M", ["hold"], vi),
                   "M_step_translation": pref(recs, "M", [k for k in M_SET if k.startswith("t_")], vi),
                   "M_step_root": pref(recs, "M", [k for k in M_SET if k.startswith("r_")], vi),
                   "M_step_finger": pref(recs, "M", ["f+5", "f-5"], vi)}
    dr = [{"D": r["D"]["ll"]} for r in recs if r["D"]]
    for vi, vn in ((0, "fixed"), (1, "profiled")):
        out[f"donor_{vn}"] = {"n": len(dr), "root10": pref(dr, "D", CM.ROOT10, vi),
                              "finger20": pref(dr, "D", ["f+20", "f-20"], vi)}
    return out


def summarize():
    res = {"domains": {}, "gates": {}}
    for domain in ("zgz_local", "zgz_global", "check"):
        files = sorted((OUT / "raw").glob(f"{domain}__*of*.json"))
        if not files:
            continue
        recs = [r for f in files for r in json.loads(f.read_text())["records"]]
        inf = [r for r in recs if r["n_events"] >= CM.MIN_EVENTS]
        d = {"all_packets": len(recs), "informative": summary_of(inf), "strata": {}}
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            d["strata"][f"[{lo},{hi})"] = summary_of([r for r in recs if lo <= r["n_events"] < hi and r["n_events"] > 0])
        res["domains"][domain] = d
    mode = json.loads(CM.CALIB.read_text()).get("kappa_mode", "fixed")
    res["primary_variant"] = mode
    for seq in CM.ZGZ:
        if seq not in res["domains"]:
            continue
        f = res["domains"][seq]["informative"][mode]
        don = res["domains"][seq]["informative"][f"donor_{mode}"]
        z, r, fi = f["zline_abs_argmax_median_mm"], f["A_root10"], f["A_finger20"]
        dmean = None if don["root10"] is None else 0.5 * (don["root10"] + don["finger20"])
        res["gates"][seq] = {"pass": bool(z <= 10 and r >= 0.70 and fi >= 0.70),
                             "falsified": bool(z > 20 or r < 0.50 or fi < 0.50),
                             "donor_control_valid": bool(dmean is not None and dmean <= 0.60),
                             "zline_abs_argmax_median_mm": z, "root10": r, "finger20": fi, "donor_mean_pref": dmean}
    g = res["gates"]
    res["stop"] = any(v["falsified"] or not v["donor_control_valid"] for v in g.values())
    (OUT / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["gates"], indent=1), "stop:", res["stop"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "summarize"))
    ap.add_argument("--domain", choices=("zgz_local", "zgz_global", "check"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.domain, a.shard, a.shards)
    else:
        summarize()


if __name__ == "__main__":
    main()
