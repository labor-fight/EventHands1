#!/usr/bin/env python3
"""S44 P0.0 / P0.1 / P0.2 / P0.7 on zgz with the training-selected hyperparameters
(docs/EXPERIMENT_RECORDS.md#s44-prereg-20261019 section 2 and the clarification note).

Steps: every protocol step of zgz_global / zgz_local except the first of a segment (no state 50 ms earlier inside it).
Layers (one Delta-theta per step and layer, from the events of (end - 50, end]):
    oracle    state GT(end - 50), owners = XAG GT source labels, Jacobian at GT(end - 50)
    deploy    state = the S39 seed3407 closed-loop state the step was conditioned on (P0.10 control, bit-identical to the
              S39 main row), deployable owners / Jacobian at that state
    shuffle   as deploy, the window's event timestamps randomly permuted (rng seeded by the step's end)
Writes outputs/s44/p01/{steps.npz, summary.json}.
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
from s44 import evflow as EF  # noqa: E402

LAYERS = ("oracle", "deploy", "shuffle")
BIG = np.deg2rad(2.0)
TIPS = (4, 8, 12, 16, 20)


def auc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = label.sum(), (~label).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    s_sorted = score[order]
    i = 0
    while i < len(score):                                   # average ranks over ties
        j = i
        while j + 1 < len(score) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[label].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def g4_counts(T, end, uv_all, support):
    """kinematic_gate.joint_counts on the SAE: recent (latest event of either polarity in the last 50 ms) pixels,
    nearest projected vertex (all 778) within 8 px -> support rows summed (16,)."""
    rec = (T.max(dim=0).values >= (end - EF.STEP + 1) * 1000).nonzero()
    cnt = torch.zeros(support.shape[1], dtype=torch.float64, device=T.device)
    if not len(rec):
        return cnt
    p = torch.stack([rec[:, 1], rec[:, 0]], -1).to(torch.float64)
    d2, idx = torch.cdist(p, uv_all).square().min(dim=1)
    ok = d2 <= 64.0
    if ok.any():
        cnt += support[idx[ok]].to(torch.float64).sum(0)
    return cnt


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyper", default="outputs/s44/p01_select/selected.json")
    ap.add_argument("--control", default="outputs/s44/p10/s39_s3407_control.npz")
    ap.add_argument("--out", default="outputs/s44/p01")
    a = ap.parse_args()
    dev = torch.device("cuda")
    hp = json.loads(Path(a.hyper).read_text())
    ctl = np.load(a.control)
    mano = EF.Mano64(dev)
    prec, _ = EF.rate_prior(dev)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = {}
    t0 = time.time()
    for s, d in EF.split_seqs("val_core"):
        sq = EF.Seq(s, d)
        intr = sq.intrinsics()
        betas = torch.tensor(sq.betas, dtype=torch.float64, device=dev)[None]
        ends = ctl[f"{s}|end"]
        prevs = ctl[f"{s}|prev"].astype(np.float64)
        runs = ctl[f"{s}|run"]
        R = {k: [] for k in ("end", "e_zero", "e_jit", "z_frac", "n_meas", "n_valid", "dth_true",
                             "ev_joint", "fit_joint_oracle", "g4", "lam_max")}
        for L in LAYERS:
            for k in ("dth", "e_art", "n_used"):
                R[f"{L}_{k}"] = []
        snr = {"s": [], "s_true": []}
        for i, end in enumerate(ends):
            end = int(end)
            rid = int(runs[i])
            ra = int(sq.runs[rid][0])
            if end - EF.STEP < ra:
                continue
            gp = sq.pos51[end - EF.STEP].astype(np.float64)
            ge = sq.pos51[end].astype(np.float64)
            dth_true = ge - gp
            # GT label jitter: the step increment of a +-25 ms moving average of the per-ms GT
            lo_s, hi_s = max(ra, end - EF.STEP - 25), min(int(sq.runs[rid][1]) - 1, end + 25)
            gsm_p = sq.pos51[max(ra, end - EF.STEP - 25):end - EF.STEP + 26].astype(np.float64).mean(0)
            gsm_e = sq.pos51[end - 25:min(end + 26, hi_s + 1)].astype(np.float64).mean(0)
            states = {"oracle": gp, "deploy": prevs[i], "shuffle": prevs[i]}
            # fingertip optical-axis energy (camera frame, root aligned)
            Pg = torch.tensor(np.stack([gp, ge, gsm_p, gsm_e]), dtype=torch.float64, device=dev)
            vg, jg = mano.fk(Pg, betas)
            dj = (jg[1] - jg[1, :1]) - (jg[0] - jg[0, :1])
            tip = dj[list(TIPS)]
            R["z_frac"].append(float((tip[:, 2] ** 2).sum() / (tip ** 2).sum().clamp(min=1e-18)))
            jh = mano.fk_hand(Pg, betas)
            true_h = jh[1] - jh[0]
            R["e_zero"].append(float(true_h.norm(dim=-1).mean() * 1000))
            R["e_jit"].append(float(((jh[3] - jh[2]) - true_h).norm(dim=-1).mean() * 1000))
            R["dth_true"].append(dth_true[EF.FINGER].reshape(15, 3))
            R["end"].append(end)
            meas = EF.step_measurements(sq, ra, end, hp["r"], hp["dfit_ms"], dev, mem_ms=hp.get("mem_ms", EF.W_MEM))
            R["n_meas"].append(meas["n_meas"])
            capm = meas["eps_px"] <= hp["eps_px"]
            R["n_valid"].append(int(capm.sum()))
            # P0.0: raw events of the step per finger-joint subtree (oracle owner: max-barycentric vertex)
            a0, a1 = int(sq.offsets[end - EF.STEP + 1]), int(sq.offsets[end + 1])
            ev_joint = np.zeros(16)
            if a1 > a0:
                ids, w, ok = EF.owners_oracle(sq, np.arange(a0, a1), mano.faces)
                vmax = ids.gather(1, w.argmax(1, keepdim=True)).squeeze(1)[ok]
                ev_joint = mano.support[vmax].sum(0).cpu().numpy()
            R["ev_joint"].append(ev_joint)
            fit_joint = np.zeros(16)
            for L in LAYERS:
                state = torch.tensor(states[L], dtype=torch.float64, device=dev)[None]
                inc = torch.zeros(51, dtype=torch.float64, device=dev)
                used = 0
                lam_j = np.zeros(15)
                if L == "shuffle":
                    m_ = EF.step_measurements(sq, ra, end, hp["r"], hp["dfit_ms"], dev,
                                              perm_rng=np.random.default_rng(end), mem_ms=hp.get("mem_ms", EF.W_MEM))
                else:
                    m_ = meas
                _, uv, z, J = mano.jacobian(state, betas, intr)
                if L == "deploy":
                    lo = max(end - EF.W_MEM + 1, ra)
                    Tg, _ = EF.build_sae(sq, lo, end, dev)
                    R["g4"].append(g4_counts(Tg, end, uv[0], mano.support).cpu().numpy())
                if m_["n_valid"] > 0:
                    if L == "oracle":
                        ids, w, ok = EF.owners_oracle(sq, m_["rows"].cpu().numpy(), mano.faces)
                    else:
                        vis = EF.visible(uv[0], z[0])
                        ids, w = EF.owners_deploy(m_["px"], uv[0], vis)
                    A, mass = EF.rows_A(J[0], m_["n"], ids, w)
                    sel = (m_["eps_px"] <= hp["eps_px"]) & (mass > 1e-6)
                    used = int(sel.sum())
                    if used:
                        th, Lam = EF.solve_rate(A[sel], m_["s"][sel], mass[sel], hp["kappa"], hp["lam_p"], prec)
                        inc = th * (EF.STEP * 1e-3)
                        if L == "deploy":
                            for j in range(15):
                                b = 6 + 3 * j
                                lam_j[j] = float(torch.linalg.eigvalsh(Lam[b:b + 3, b:b + 3]).max())
                        if L == "oracle":
                            rate_true = torch.tensor(dth_true / (EF.STEP * 1e-3), dtype=torch.float64, device=dev)
                            snr["s"].append(m_["s"][sel].cpu().numpy())
                            snr["s_true"].append((A[sel] @ rate_true / mass[sel]).cpu().numpy())
                            vmax = ids.gather(1, w.argmax(1, keepdim=True)).squeeze(1)[sel]
                            fit_joint = mano.support[vmax].sum(0).cpu().numpy()
                P = state.expand(2, -1).clone()
                P[1, EF.FINGER] += inc[EF.FINGER]
                jj = mano.fk_hand(P, betas)
                e = float((((jj[1] - jj[0]) - true_h).norm(dim=-1).mean()) * 1000)
                R[f"{L}_dth"].append(inc[EF.FINGER].cpu().numpy().reshape(15, 3))
                R[f"{L}_e_art"].append(e)
                R[f"{L}_n_used"].append(used)
                if L == "deploy":
                    R["lam_max"].append(lam_j)
            R["fit_joint_oracle"].append(fit_joint)
        for k, v in R.items():
            rows[f"{s}|{k}"] = np.asarray(v)
        rows[f"{s}|snr_s"] = np.concatenate(snr["s"]) if snr["s"] else np.zeros(0)
        rows[f"{s}|snr_s_true"] = np.concatenate(snr["s_true"]) if snr["s_true"] else np.zeros(0)
        print(f"{s}: {len(R['end'])} steps ({time.time() - t0:.0f} s)", flush=True)
    np.savez_compressed(out_dir / "steps.npz", **rows)
    summ = {"hyper": hp, "control": a.control, "wall_s": time.time() - t0, "sequences": {}}
    for s, _ in EF.split_seqs("val_core"):
        g = lambda k: rows[f"{s}|{k}"]  # noqa: E731
        ez = g("e_zero")
        ent = {"n_steps": int(len(ez)), "mean_e_zero_mm": float(ez.mean()), "mean_e_jit_mm": float(g("e_jit").mean()),
               "fingertip_z_energy_frac_median": float(np.median(g("z_frac"))),
               "meas_per_step_median": float(np.median(g("n_meas"))), "valid_per_step_median": float(np.median(g("n_valid")))}
        dt = g("dth_true")
        for L in LAYERS:
            ea = g(f"{L}_e_art")
            dh = g(f"{L}_dth")
            num = (dh * dt).sum(-1).sum(0)
            den = np.sqrt((dh ** 2).sum(-1).sum(0) * (dt ** 2).sum(-1).sum(0))
            r_j = np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)
            Rj = np.linalg.norm(dh - dt, axis=-1).sum(0) / np.linalg.norm(dt, axis=-1).sum(0)
            ent[L] = {"mean_e_art_mm": float(ea.mean()), "R_art": float(ea.sum() / ez.sum()),
                      "r_j": r_j.tolist(), "r_j_median": float(np.median(r_j)), "R_art_j": Rj.tolist(),
                      "n_joints_Rj_le_0.7": int((Rj <= 0.7).sum()), "n_joints_rj_lt_0.3": int((r_j < 0.3).sum()),
                      "used_per_step_median": float(np.median(g(f"{L}_n_used")))}
        e_dep, e_shu = ent["deploy"]["mean_e_art_mm"], ent["shuffle"]["mean_e_art_mm"]
        gain_dep = ent["mean_e_zero_mm"] - e_dep
        ent["shuffle_retention"] = float((ent["mean_e_zero_mm"] - e_shu) / gain_dep) if gain_dep > 0 else None
        evj = g("ev_joint")[:, 1:]
        med = np.median(evj, axis=0)
        ent["P00"] = {"median_events_per_joint": med.tolist(), "n_joints_median_lt_20": int((med < 20).sum()),
                      "median_validfits_per_joint": np.median(g("fit_joint_oracle")[:, 1:], axis=0).tolist()}
        lab = (np.linalg.norm(dt, axis=-1) >= BIG)                   # (steps, 15)
        lm, g4 = g("lam_max"), g("g4")[:, 1:]
        ent["P02"] = {"auc_lambda_max": auc(lm.reshape(-1), lab.reshape(-1)), "auc_g4": auc(g4.reshape(-1), lab.reshape(-1)),
                      "pos_frac": float(lab.mean())}
        st, s_ = rows[f"{s}|snr_s_true"], rows[f"{s}|snr_s"]
        if len(st):
            res = s_ - st
            bins = [0, 20, 50, 150, 1e9]
            ent["P07"] = {"corr_s_vs_true": float(np.corrcoef(s_, st)[0, 1]), "n": int(len(st)),
                          "residual_mean_over_std": float(res.mean() / res.std()),
                          "by_true_speed": {f"[{lo},{hi})": {"n": int(((st >= lo) & (st < hi)).sum()),
                                                            "median_rel_err": float(np.median(np.abs(res[(st >= lo) & (st < hi)])
                                                                                              / np.maximum(np.abs(st[(st >= lo) & (st < hi)]), 1e-9)))
                                                            if ((st >= lo) & (st < hi)).sum() else None}
                                            for lo, hi in zip(bins[:-1], bins[1:])},
                          "frac_true_negative": float((st < 0).mean())}
        summ["sequences"][s] = ent
    (out_dir / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({s: {L: summ["sequences"][s][L]["R_art"] for L in LAYERS} for s in summ["sequences"]}), flush=True)


if __name__ == "__main__":
    main()
