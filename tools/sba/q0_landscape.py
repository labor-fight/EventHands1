#!/usr/bin/env python3
"""SBA Q0 (docs/EXPERIMENT_RECORDS.md#sba-p4-prereg-20261022): likelihood landscape of nuisance-orthogonal inference.

P0's packets and hypotheses (GT; root +-10 deg; finger +-20 deg; translation; finger-motion scale 0.5 / 1.5; donor
events), scored with the P4 emission model (outputs/sba/p4/fit_raw_b1.json, selected smoothness) and the nuisances
re-estimated for every hypothesis:

    V0  P0's model (contour only, S0 kappa ratio, eta fixed), one scale profiled          -- reproduces P0
    V1  P4 theta, one scale profiled, eta fixed at the P4 value
    V2  P4 theta, per-group emission multipliers m_g (6 part groups) and eta profiled (convex Poisson regression)
    V3  V2 with the two-dimensional scale gauge sigma in {0.90, ..., 1.04} profiled (mesh about the wrist, depth fixed)

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python tools/sba/q0_landscape.py run --domain zgz_local --shard 0 --shards 3
    python tools/sba/q0_landscape.py summarize
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
from sba.p0_landscape import packets  # noqa: E402
from semkine import sba_emission as EM  # noqa: E402

OUT = CM.OUT / "q0"
P4 = CM.OUT / "p4"
SIGMAS = [round(0.90 + 0.02 * k, 2) for k in range(8)]
HYP = ["clean"] + CM.ROOT10 + ["f+20", "f-20", "z+50", "z-50", "x+20", "x-20", "y+20", "y-20", "sf0.5", "sf1.5"]
DONOR = ["clean"] + CM.ROOT10 + ["f+20", "f-20"]
VARIANTS = ("V0", "V1", "V2", "V3")


def profile(X, Xt, B, Bt, N, iters=40):
    """max_{w >= 0} sum_ev N log(X^T w + B) - Xt . w - Bt over event cells; X (k, n), Xt (k,), B (n,), N (n,)."""
    X, B, N, Xt = X.double(), B.double(), N.double(), Xt.double()
    k = X.shape[0]
    w = torch.full((k,), 0.5, dtype=torch.float64, device=X.device)
    for _ in range(iters):
        lam = (w @ X + B).clamp(min=1e-12)
        g = X @ (N / lam) - Xt
        H = -(X * (N / lam ** 2)) @ X.T - 1e-9 * torch.eye(k, dtype=torch.float64, device=X.device)
        w = (w - torch.linalg.solve(H, g)).clamp(min=0.0)
    lam = (w @ X + B).clamp(min=1e-12)
    return float((N * lam.log()).sum() - Xt @ w - Bt)


class Q0:
    def __init__(self, ctx, fit, s0):
        self.ctx = ctx
        self.patch = torch.from_numpy(np.load(P4 / "patches.npy")).to(ctx.dev)
        self.n_p = int(fit["meta"]["n_patches"])
        sel = fit["selected"]
        self.theta = torch.tensor(sel["theta"], dtype=torch.float32, device=ctx.dev)
        self.eta4 = float(sel["eta"])
        self.blur = float(fit["meta"]["blur"])
        self.ks, self.ki, self.eta0 = s0["kappa_sil"], s0["kappa_int"], s0["outlier"]
        grp_edge = ctx.geo.groups[ctx.geo.topo.edges[:, 0]]
        maj = torch.zeros(self.n_p, dtype=torch.long, device=ctx.dev)
        for i in range(self.n_p):
            g = grp_edge[self.patch == i]
            maj[i] = torch.bincount(g, minlength=6).argmax() if len(g) else 0
        n_ch = EM.n_channels(self.n_p)
        self.member = torch.zeros(6, n_ch, device=ctx.dev)                  # channel -> part group
        for g in range(6):
            self.member[g, g] = 1.0
            self.member[g, 6 + g] = 1.0
            self.member[g, 12:][maj == g] = 1.0

    def scores(self, sd, win, x0, x1, sigma):
        """{variant: log-likelihood} at one scale (V0..V2) -- V3 takes the max over scales of V2."""
        F = EM.features(self.ctx.geo, win, x0, x1, sd.betas, sd.K, self.patch, self.n_p, self.blur, sigma)
        N = win.N.sum(0)
        cell = N > 0
        Ne = N[cell]
        B, U = win.bkg, win.uni
        Fe = F[:, cell]
        Ft = F.flatten(1).sum(1)
        out = {}
        if sigma == 1.0:
            c0 = self.ks * F[:6].sum(0) + self.ki * F[6:12].sum(0)
            out["V0"] = profile(c0[cell][None], c0.sum()[None], (B + self.eta0 * U)[cell],
                                float((B + self.eta0 * U).sum()), Ne)
            c1 = torch.einsum("c,c...->...", self.theta, F)
            out["V1"] = profile(c1[cell][None], c1.sum()[None], (B + self.eta4 * U)[cell],
                                float((B + self.eta4 * U).sum()), Ne)
        G = (self.member * self.theta[None]) @ Fe                            # (6, n_ev)
        Gt = (self.member * self.theta[None]) @ Ft
        X = torch.cat([G, U[cell][None]])
        Xt = torch.cat([Gt, U.sum()[None]])
        out["V2"] = profile(X, Xt, B[cell], float(B.sum()), Ne)
        return out


def hyp_states(ctx, sd, x0, x1, name, joint):
    if name.startswith("sf"):
        s = float(name[2:])
        mid = 0.5 * (x0 + x1)
        a, b = x0.clone(), x1.clone()
        a[6:] = mid[6:] - s * (mid[6:] - x0[6:])
        b[6:] = mid[6:] + s * (x1[6:] - mid[6:])
        return a, b
    kind, val = CM.PERTURB[name]
    return CM.perturb(ctx, x0, kind, val, joint, sd.betas), CM.perturb(ctx, x1, kind, val, joint, sd.betas)


def score_set(q, sd, win, states):
    res = {}
    for name, (a, b) in states.items():
        per = {}
        v3 = []
        for sg in SIGMAS:
            sc = q.scores(sd, win, a, b, sg)
            if sg == 1.0:
                per.update({k: sc[k] for k in ("V0", "V1", "V2")})
            v3.append(sc["V2"])
        per["V3"] = max(v3)
        per["sigma_star"] = SIGMAS[int(np.argmax(v3))]
        res[name] = per
    return res


def run(domain, shard, shards):
    dev = CM.require_gpu()
    torch.set_grad_enabled(False)
    ctx = CM.Ctx(dev)
    fit = json.loads((P4 / "fit_raw_b1.json").read_text())
    s0 = json.loads(CM.CALIB.read_text())
    q = Q0(ctx, fit, s0)
    items = packets(domain)[shard::shards]
    cache, recs = {}, []
    t0 = time.time()
    for seq, d, index, row, rows in items:
        if seq not in cache:
            cache = {seq: CM.SeqData(seq, d, dev)}
        sd = cache[seq]
        prev, end = int(row["previous_end_ms"]), int(row["end_ms"])
        x0, x1 = sd.gt(prev), sd.gt(end)
        win = CM.window(ctx, sd, prev, end, x0, x1)
        rec = {"seq": seq, "index": index, "end": end, "n_events": win.n_events}
        if win.n_events >= CM.MIN_EVENTS:
            joint = index % 15
            rec["H"] = score_set(q, sd, win, {n: hyp_states(ctx, sd, x0, x1, n, joint) for n in HYP})
            dr = CM.donor_row(ctx, sd, rows, index, sd.betas)
            if dr is not None:
                evms = CM.retime(sd.events(int(dr["previous_end_ms"]), int(dr["end_ms"])),
                                 int(dr["previous_end_ms"]), prev, end)
                if len(evms[1]) >= CM.MIN_EVENTS:
                    wd = CM.window(ctx, sd, prev, end, x0, x1, events=evms)
                    rec["D"] = score_set(q, sd, wd, {n: hyp_states(ctx, sd, x0, x1, n, joint) for n in DONOR})
        recs.append(rec)
        if len(recs) % 25 == 0:
            print(json.dumps({"domain": domain, "shard": shard, "done": len(recs), "of": len(items),
                              "s": round(time.time() - t0)}), flush=True)
    path = OUT / "raw" / f"{domain}__{shard:02d}of{shards:02d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"records": recs, "wall_s": time.time() - t0}))
    CM.receipt(OUT / "raw" / f"{domain}__{shard:02d}", {"probe": "Q0", "domain": domain, "shard": shard},
               scripts=(__file__, CM.REPO / "semkine" / "sba_emission.py"))


def pref(recs, key, names, v):
    x = [r[key]["clean"][v] > r[key][n][v] for r in recs if key in r for n in names]
    return float(np.mean(x)) if x else None


def summary_of(recs):
    out = {"n": len(recs)}
    for v in VARIANTS:
        out[v] = {"root10": pref(recs, "H", CM.ROOT10, v), "finger20": pref(recs, "H", ["f+20", "f-20"], v),
                  "translation": pref(recs, "H", ["z+50", "z-50", "x+20", "x-20", "y+20", "y-20"], v),
                  "finger_motion_scale_GT_over_0.5": pref(recs, "H", ["sf0.5"], v),
                  "finger_motion_scale_GT_over_1.5": pref(recs, "H", ["sf1.5"], v),
                  "donor_root10": pref(recs, "D", CM.ROOT10, v), "donor_finger20": pref(recs, "D", ["f+20", "f-20"], v)}
    out["sigma_star_median_at_GT"] = float(np.median([r["H"]["clean"]["sigma_star"] for r in recs if "H" in r]))
    return out


def summarize():
    res = {"domains": {}, "gates": {}}
    sel = ("ycy", "ylf")
    for domain in ("zgz_local", "zgz_global", "check"):
        files = sorted((OUT / "raw").glob(f"{domain}__*of*.json"))
        if not files:
            continue
        recs = [r for f in files for r in json.loads(f.read_text())["records"] if "H" in r]
        if domain == "check":
            for tag, keep in (("check_fit_subjects", lambda r: r["seq"].split("_")[0] not in sel),
                              ("check_selection_subjects", lambda r: r["seq"].split("_")[0] in sel)):
                rs = [r for r in recs if keep(r)]
                res["domains"][tag] = {"local": summary_of([r for r in rs if "_local" in r["seq"]]),
                                       "global": summary_of([r for r in rs if "_global" in r["seq"]])}
        else:
            res["domains"][domain] = summary_of(recs)
    for seq in CM.ZGZ:
        if seq not in res["domains"]:
            continue
        v = res["domains"][seq]["V3"]
        dmean = 0.5 * (v["donor_root10"] + v["donor_finger20"]) if v["donor_root10"] is not None else None
        res["gates"][seq] = {"pass": bool(v["finger20"] >= 0.70 and v["root10"] >= 0.70 and dmean is not None
                                          and dmean <= 0.60),
                             "falsified": bool(v["finger20"] < 0.60), "finger20": v["finger20"], "root10": v["root10"],
                             "donor_mean_pref": dmean}
    res["stop"] = any(g["falsified"] for g in res["gates"].values())
    res["proceed_to_P1"] = bool(res["gates"]) and all(g["pass"] for g in res["gates"].values())
    (OUT / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


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
