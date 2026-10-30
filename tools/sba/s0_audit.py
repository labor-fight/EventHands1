#!/usr/bin/env python3
"""SBA S0 (docs/EXPERIMENT_RECORDS.md#sba-prereg-20261022 S0): audits before any zgz landscape / tracking reading.

    synth   S0a positive / negative control on events from an independent simulator: exact triangle raster of the
            per-ms GT mesh at 640 x 480 (semkine/kssf.py, not the sweep code), log intensity = hand shading by depth or
            background, v2e-style threshold crossings (C = 0.2) at 1 ms, the dataset's floor(3i/8) binning, uniform
            noise. Landscape (condition A, kappa profiled) on 60 windows of lyq_local / lyq_global.
    calib   S0b on the 18 training base sequences (every 8th protocol step, GT trajectories): kappa_sil / kappa_int
            (Poisson MLE), blur sigma in {0, 1} chosen by 2-fold-over-subjects held-out likelihood, per-event deviance
            gain over a background-only model, per-subject kappa CV, Pearson dispersion, polarity consistency;
            S0c: predicted / observed hand events on zgz_local at GT with the training kappa.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python tools/sba/s0_audit.py synth
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sba import common as CM  # noqa: E402
from sba.common import SB, EF  # noqa: E402
from semkine.kssf import KSSF  # noqa: E402

OUT = CM.OUT / "s0"
C_THR = 0.2
BG_LOG, SHADE, Z_REF = -1.5, 10.0, 0.6
NOISE_PER_PX_MS = 1e-4
FOLD_A = ("ch", "lfz", "lpc", "lr", "ly")
SUBJECTS = ("ch", "lfz", "lpc", "lr", "ly", "lyh", "lyq", "ycy", "ylf")
BLURS = (0.0, 1.0)
ZLINE = [k * 5 for k in range(-12, 13)]


# --------------------------------------------------------------------------------------------- simulator
@torch.no_grad()
def simulate(ctx, kssf, sd, m_prev, m1, rng):
    """Events (ev (n, 3) uint8 output grid, ms (n,)) of (m_prev, m1] from the independent simulator."""
    ms_all = np.arange(m_prev, m1 + 1)
    X = torch.tensor(np.asarray(sd.pos51[ms_all], np.float32), device=ctx.dev)
    v, _ = ctx.geo.fk(X, sd.betas)
    K = sd.K[None].expand(len(ms_all), 3, 3)
    f = kssf.rasterize(v, K)
    hand = f.face_id >= 0
    L = torch.where(hand, -SHADE * (f.depth - Z_REF), torch.full_like(f.depth, BG_LOG))
    mem = L[0].clone()
    xs, ys, ps, ts = [], [], [], []
    for t in range(1, len(ms_all)):
        d = L[t] - mem
        n = (d.abs() / C_THR).floor()
        nz = n > 0
        if bool(nz.any()):
            y, x = nz.nonzero(as_tuple=True)
            k = n[y, x].long()
            pol = (d[y, x] > 0).long()
            xs.append(torch.repeat_interleave(x, k)); ys.append(torch.repeat_interleave(y, k))
            ps.append(torch.repeat_interleave(pol, k))
            ts.append(torch.full((int(k.sum()),), int(ms_all[t]), device=ctx.dev, dtype=torch.long))
            mem = mem + torch.sign(d) * n * C_THR
    W = m1 - m_prev
    n_noise = rng.poisson(NOISE_PER_PX_MS * SB.SRC_W * SB.SRC_H * W)
    nx = torch.from_numpy(rng.integers(0, SB.SRC_W, n_noise)).to(ctx.dev)
    ny = torch.from_numpy(rng.integers(0, SB.SRC_H, n_noise)).to(ctx.dev)
    npol = torch.from_numpy(rng.integers(0, 2, n_noise)).to(ctx.dev)
    nt = torch.from_numpy(rng.integers(m_prev + 1, m1 + 1, n_noise)).to(ctx.dev)
    x = torch.cat(xs + [nx]); y = torch.cat(ys + [ny]); p = torch.cat(ps + [npol]); t = torch.cat(ts + [nt])
    g = ctx.geo.grid
    ev = torch.stack([g.lut_x[x], g.lut_y[y], p], 1).cpu().numpy().astype(np.uint8)
    return ev, t.cpu().numpy().astype(np.int64)


def landscape_A(ctx, sd, m_prev, m1, x0, x1, model, ev_ms, joint, profile=True):
    """Condition A (both knots displaced by the same XDI perturbation) + z-line: log-likelihood per hypothesis."""
    win = CM.window(ctx, sd, m_prev, m1, x0, x1, events=ev_ms)
    ev = SB.Evaluator(ctx.geo, win, model, sd.betas, sd.K)
    names, H0, H1 = [], [], []
    for name, (kind, val) in CM.PERTURB.items():
        if name.startswith("r") and name[2:] in ("+20", "-20"):
            continue
        names.append(name)
        H0.append(CM.perturb(ctx, x0, kind, val, joint, sd.betas))
        H1.append(CM.perturb(ctx, x1, kind, val, joint, sd.betas))
    for z in ZLINE:
        names.append(f"zl{z:+d}")
        H0.append(CM.perturb(ctx, x0, "t", (2, z / 1000.0), joint, sd.betas))
        H1.append(CM.perturb(ctx, x1, "t", (2, z / 1000.0), joint, sd.betas))
    ll = {}
    for name, a, b in zip(names, H0, H1):
        A = ev.areas(a[None], b[None], ev.plan_at(a, b))
        if profile:
            k = SB.profile_kappa(A, win, model)
            m2 = SB.Model(model.kappa_sil * float(k[0]), model.kappa_int * float(k[0]), model.signed,
                          model.dispersion, model.blur)
            ll[name] = float(SB.loglik(SB.rates(A, win, m2), SB.counts(win, m2))[0])
        else:
            ll[name] = float(ev.loglik(A, 1, 1)[0])
    return ll, win.n_crop


def summarise_landscape(rows):
    root = [r["ll"]["clean"] > r["ll"][k] for r in rows for k in CM.ROOT10]
    fing = [r["ll"]["clean"] > r["ll"][k] for r in rows for k in ("f+20", "f-20")]
    trans = [r["ll"]["clean"] > r["ll"][k] for r in rows for k in ("z+50", "z-50", "x+20", "x-20", "y+20", "y-20")]
    zst = [abs(ZLINE[int(np.argmax([r["ll"][f"zl{z:+d}"] for z in ZLINE]))]) for r in rows]
    return {"n": len(rows), "gt_pref_root10": float(np.mean(root)) if root else None,
            "gt_pref_finger20": float(np.mean(fing)) if fing else None,
            "gt_pref_translation": float(np.mean(trans)) if trans else None,
            "zline_abs_argmax_median_mm": float(np.median(zst)) if zst else None}


def run_synth(dev):
    ctx = CM.Ctx(dev)
    kssf = KSSF(ctx.mano, SB.SRC_H, SB.SRC_W, 1.0).to(dev)
    rng = np.random.default_rng(0)
    model = SB.Model(kappa_sil=1.0, kappa_int=1.0, outlier=0.1)         # kappa profiled per hypothesis
    res = {"windows": {}, "settings": {"C": C_THR, "background_log": BG_LOG, "shade_per_m": SHADE,
                                       "noise_per_src_px_ms": NOISE_PER_PX_MS}}
    t0 = time.time()
    for seq in ("lyq_local", "lyq_global"):
        sd = CM.SeqData(seq, "train", dev)
        rows = CM.endpoint_rows(sd)[::20][:30]
        sims = []
        for (rid, end, prev, first) in rows:
            sims.append((prev, end, simulate(ctx, kssf, sd, prev, end, rng)))
        out = []
        for i, (prev, end, evms) in enumerate(sims):
            if len(evms[1]) < CM.MIN_EVENTS:
                continue
            x0, x1 = sd.gt(prev), sd.gt(end)
            ll, nc = landscape_A(ctx, sd, prev, end, x0, x1, model, evms, joint=i % 15)
            wt = ctx.wrist(x1, sd.betas)[0]
            donor = None
            for k in range(1, len(sims)):
                p2, e2, ev2 = sims[(i + k) % len(sims)]
                g2 = sd.gt(e2)
                if float((ctx.wrist(g2, sd.betas)[0] - wt).norm()) * 1000 >= 40 or CM.geo_deg(x1[3:6], g2[3:6]) >= 15:
                    donor = CM.retime(ev2, p2, prev, end)
                    break
            dl = None
            if donor is not None and len(donor[1]) >= CM.MIN_EVENTS:
                dl, _ = landscape_A(ctx, sd, prev, end, x0, x1, model, donor, joint=i % 15)
            out.append({"end": end, "n_events": int(len(evms[1])), "n_crop": nc, "ll": ll, "donor_ll": dl})
            print(seq, end, json.dumps({"n": len(evms[1]), "clean_minus_best_other":
                                        ll["clean"] - max(v for k, v in ll.items() if k not in ("clean", "zl+0"))}),
                  f"{time.time() - t0:.0f} s", flush=True)
        res["windows"][seq] = out
    allrows = [r for v in res["windows"].values() for r in v]
    res["summary"] = summarise_landscape(allrows)
    don = [r for r in allrows if r["donor_ll"] is not None]
    dsum = summarise_landscape([{"ll": r["donor_ll"]} for r in don])
    res["summary_donor"] = dsum
    s, d = res["summary"], dsum
    res["gate_S0a"] = {"pass": bool(s["gt_pref_root10"] >= 0.90 and s["gt_pref_finger20"] >= 0.90
                                    and s["zline_abs_argmax_median_mm"] <= 5.0
                                    and (d["gt_pref_root10"] is None or
                                         0.5 * (d["gt_pref_root10"] + d["gt_pref_finger20"]) <= 0.60)),
                       "rule": "synthetic: GT preferred >= 0.90 for root +-10 deg and finger +-20 deg, z-line |z*| median"
                               " <= 5 mm; donor events (>= 40 mm / 15 deg away): mean GT preference (root, finger) <= 0.60"}
    res["wall_s"] = time.time() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "synth.json").write_text(json.dumps(res, indent=1))
    CM.receipt(OUT / "synth", {"probe": "S0a"}, scripts=(__file__,))
    print(json.dumps({k: res[k] for k in ("summary", "summary_donor", "gate_S0a")}, indent=1))


# --------------------------------------------------------------------------------------------- calibration
def window_stats(ctx, sd, prev, end, blurs):
    """Per GT window: sufficient statistics on the cells with events, totals, per blur."""
    x0, x1 = sd.gt(prev), sd.gt(end)
    win = CM.window(ctx, sd, prev, end, x0, x1)
    ev = SB.Evaluator(ctx.geo, win, SB.Model(1.0, 1.0), sd.betas, sd.K)
    A = ev.areas(x0[None], x1[None], ev.plan_at(x0, x1))[0]                      # (3, S, h, w)
    N = win.N                                                                     # (2, S, h, w)
    Nm = N.sum(0)
    cell = Nm > 0
    out = {"n_crop": int(Nm.sum()), "n_cells": int(Nm.numel()), "B_tot": float(win.bkg.sum()), "b0": win.b0,
           "b_fa": win.b_fa,
           "N": Nm[cell].float().cpu(), "N_on": N[1][cell].float().cpu(), "N_off": N[0][cell].float().cpu(),
           "B": win.bkg[cell].cpu(), "U": win.uni[cell].cpu()}
    # background-only reference: uniform rate within the crop with the gain pattern and bin widths
    gw = win.bkg / win.bkg.sum().clamp(min=1e-12)
    out["bg_ref"] = (gw[cell] * Nm.sum()).cpu()
    for b in blurs:
        Ab = SB.blur_area(A, b)
        out[b] = {"As": (Ab[SB.CH_SIL_POS] + Ab[SB.CH_SIL_NEG])[cell].cpu(), "Ai": Ab[SB.CH_INT][cell].cpu(),
                  "Ap": Ab[SB.CH_SIL_POS][cell].cpu(), "An": Ab[SB.CH_SIL_NEG][cell].cpu(),
                  "As_tot": float((Ab[SB.CH_SIL_POS] + Ab[SB.CH_SIL_NEG]).sum()), "Ai_tot": float(Ab[SB.CH_INT].sum())}
    return out


def fit_kappa(stats, b, iters=60):
    """Poisson MLE of (kappa_sil, kappa_int, outlier) on pooled windows (projected Newton, all >= 0)."""
    As = torch.cat([s[b]["As"] for s in stats]).double()
    Ai = torch.cat([s[b]["Ai"] for s in stats]).double()
    U = torch.cat([s["U"] for s in stats]).double()
    B = torch.cat([s["B"] for s in stats]).double()
    N = torch.cat([s["N"] for s in stats]).double()
    T = torch.tensor([sum(s[b]["As_tot"] for s in stats), sum(s[b]["Ai_tot"] for s in stats),
                      float(sum(s["n_crop"] for s in stats))], dtype=torch.float64)
    X = torch.stack([As, Ai, U], 1)
    k = torch.tensor([1.0, 0.5, 0.1], dtype=torch.float64)
    for _ in range(iters):
        lam = (X @ k + B).clamp(min=1e-12)
        g = X.T @ (N / lam) - T
        Hm = -(X * (N / lam ** 2)[:, None]).T @ X
        step = -torch.linalg.solve(Hm - 1e-9 * torch.eye(3, dtype=torch.float64), g)
        k = (k + step).clamp(min=1e-6)
    return float(k[0]), float(k[1]), float(k[2])


def window_ll(s, b, ks, ki, eta):
    lam = (ks * s[b]["As"].double() + ki * s[b]["Ai"].double() + eta * s["U"].double() + s["B"].double()).clamp(min=1e-12)
    tot = ks * s[b]["As_tot"] + ki * s[b]["Ai_tot"] + eta * s["n_crop"] + s["B_tot"]
    return float((s["N"].double() * lam.log()).sum() - tot)


def run_calib(dev):
    ctx = CM.Ctx(dev)
    t0 = time.time()
    stats = {}
    for s, d in EF.training_base_seqs():
        sd = CM.SeqData(s, d, dev)
        lst = []
        for (rid, end, prev, first) in CM.endpoint_rows(sd)[::8]:
            st = window_stats(ctx, sd, prev, end, BLURS)
            st["subject"], st["category"] = s.split("_")[0], s.split("_")[1]
            lst.append(st)
        stats[s] = lst
        print(s, len(lst), f"{time.time() - t0:.0f} s", flush=True)
    allw = [w for v in stats.values() for w in v]
    res = {"n_windows": len(allw)}
    # blur selection: 2-fold over subjects, held-out log-likelihood per event
    held = {}
    for b in BLURS:
        tot, nev = 0.0, 0
        for fit_set, test_set in ((FOLD_A, None), (None, FOLD_A)):
            fit = [w for w in allw if (w["subject"] in FOLD_A) == (fit_set is not None)]
            test = [w for w in allw if (w["subject"] in FOLD_A) != (fit_set is not None)]
            ks, ki, eta = fit_kappa(fit, b)
            tot += sum(window_ll(w, b, ks, ki, eta) for w in test)
            nev += sum(w["n_crop"] for w in test)
        held[str(b)] = tot / max(nev, 1)
    blur = max(BLURS, key=lambda b: held[str(b)])
    ks, ki, eta = fit_kappa(allw, blur)
    res.update({"heldout_ll_per_event": held, "blur": blur, "kappa_sil": ks, "kappa_int": ki, "outlier": eta})
    # per-subject kappa (ratio fixed)
    per_subj = {}
    for subj in SUBJECTS:
        ws = [w for w in allw if w["subject"] == subj]
        r = ki / ks
        As = torch.cat([w[blur]["As"] + r * w[blur]["Ai"] for w in ws]).double()
        B = torch.cat([w["B"] + eta * w["U"] for w in ws]).double()
        N = torch.cat([w["N"] for w in ws]).double()
        T = sum(w[blur]["As_tot"] + r * w[blur]["Ai_tot"] for w in ws)
        k = ks
        for _ in range(50):
            lam = (k * As + B).clamp(min=1e-12)
            g1 = float((N * As / lam).sum() - T)
            g2 = float(-(N * As * As / lam ** 2).sum())
            k = max(1e-6, k - g1 / g2) if g2 < 0 else k
        per_subj[subj] = k
    kv = np.array(list(per_subj.values()))
    res["kappa_sil_per_subject"] = per_subj
    res["kappa_cv"] = float(kv.std() / kv.mean())
    # deviance gain per event, polarity consistency, dispersion
    gains = {"local": [], "global": []}
    pol = {"local": [], "global": []}
    signs = {"local": [], "global": []}
    pear_num, pear_cells, dev_num = 0.0, 0, 0.0
    for w in allw:
        if w["n_crop"] < CM.MIN_EVENTS:
            continue
        lm = window_ll(w, blur, ks, ki, eta)
        bg = w["bg_ref"].double().clamp(min=1e-12)
        lb = float((w["N"].double() * bg.log()).sum() - w["n_crop"])
        gains[w["category"]].append((lm - lb) / w["n_crop"])
        ap, an = w[blur]["Ap"].double(), w[blur]["An"].double()
        num = float((w["N_on"].double() * ap + w["N_off"].double() * an).sum())
        den = float((w["N"].double() * (ap + an)).sum())
        if den > 0:
            sc = num / den
            pol[w["category"]].append(max(sc, 1 - sc))
            signs[w["category"]].append(1 if sc >= 0.5 else -1)
        lam = (ks * w[blur]["As"].double() + ki * w[blur]["Ai"].double() + eta * w["U"].double()
               + w["B"].double()).clamp(min=1e-12)
        tot = ks * w[blur]["As_tot"] + ki * w[blur]["Ai_tot"] + eta * w["n_crop"] + w["B_tot"]
        Nd = w["N"].double()
        pear_num += float((Nd ** 2 / lam).sum() - 2 * Nd.sum()) + tot
        dev_num += 2 * (float((Nd * (Nd / lam).log()).sum() - Nd.sum()) + tot)
        pear_cells += w["n_cells"]
    res["deviance_gain_per_event_median"] = {k: float(np.median(v)) if v else None for k, v in gains.items()}
    res["deviance_gain_n"] = {k: len(v) for k, v in gains.items()}
    res["polarity_consistency_median"] = {k: float(np.median(v)) if v else None for k, v in pol.items()}
    res["polarity_sign_plus_fraction"] = {k: float(np.mean(np.array(v) > 0)) if v else None for k, v in signs.items()}
    # quasi-likelihood dispersion over all crop cells of the informative windows: Pearson X2 / cells (used to temper
    # the information in P3's fused arm) and deviance / cells (reported)
    res["dispersion_pearson"] = pear_num / max(pear_cells, 1)
    res["dispersion_deviance"] = dev_num / max(pear_cells, 1)
    signed = all(v is not None and v >= 0.70 for v in res["polarity_consistency_median"].values())
    res["signed"] = signed
    res["gate_S0b"] = {"pass": bool(all(v is not None and v > 0 for v in res["deviance_gain_per_event_median"].values())
                                    and res["kappa_cv"] <= 1.0),
                       "rule": "median per-event deviance gain over background-only > 0 in training local and global;"
                               " per-subject kappa_sil CV <= 1.0"}
    # S0c: zgz_local count prediction at GT with the training kappa
    sd = CM.SeqData("zgz_local", "val", dev)
    ratios = []
    for (rid, end, prev, first) in CM.endpoint_rows(sd):
        w = window_stats(ctx, sd, prev, end, (blur,))
        if w["n_crop"] < CM.MIN_EVENTS:
            continue
        pred = ks * w[blur]["As_tot"] + ki * w[blur]["Ai_tot"]
        obs = w["n_crop"] - w["B_tot"]
        if obs > 0:
            ratios.append(pred / obs)
    res["S0c_pred_over_obs_median"] = float(np.median(ratios)) if ratios else None
    res["S0c_n"] = len(ratios)
    res["S0c_pass"] = bool(ratios and 0.7 <= res["S0c_pred_over_obs_median"] <= 1.3)
    res["kappa_mode_for_P1_P3"] = "fixed" if res["S0c_pass"] else "profiled"
    res["wall_s"] = time.time() - t0
    res["dispersion"] = max(1.0, res["dispersion_pearson"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "calib.json").write_text(json.dumps(res, indent=1))
    calib = {"kappa_sil": ks, "kappa_int": ki, "outlier": eta, "blur": blur, "signed": signed,
             "dispersion": res["dispersion"],
             "kappa_mode": res["kappa_mode_for_P1_P3"], "source": "outputs/sba/s0/calib.json"}
    CM.CALIB.write_text(json.dumps(calib, indent=1))
    CM.receipt(OUT / "calib", {"probe": "S0b/S0c"}, scripts=(__file__,))
    print(json.dumps({k: v for k, v in res.items() if k != "kappa_sil_per_subject"}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("synth", "calib"))
    a = ap.parse_args()
    dev = CM.require_gpu()
    torch.set_grad_enabled(False)
    if a.what == "synth":
        run_synth(dev)
    else:
        run_calib(dev)


if __name__ == "__main__":
    main()
