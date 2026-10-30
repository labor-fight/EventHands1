#!/usr/bin/env python3
"""KA step 1, real-data part (docs/EXPERIMENT_RECORDS.md#ka-prereg-20261028). CPU, zero training, no zgz.

Windows: the 18 training base recordings (FIT7 + SELECT), every endpoint of the evaluator's 50 ms grid inside a valid
run (each run's first step skipped); events of (end - 50, end]; previous state = GT(end - 50). Finger-dominant filter:
root rotation change <= R deg and wrist displacement <= D mm over the window, largest hand-frame fingertip displacement
>= 3 mm, >= 100 events inside the previous mesh's projected box (+10 px). Rung (R, D) = (1, 2), else (2, 4), else (3, 6):
the first rung with >= 400 windows; at most 3000 windows (seeded subsample).

Motion descriptor: displacement of the 20 non-wrist OpenPose joints over the window in the previous hand frame
(J_h = R_prev^T (J - J_wrist)), 60-d, mm. Nuisance-different pair: different recording, or root orientations >= 20 deg
apart, or wrist depths >= 5 cm apart. Per-anchor AUC: positives = nuisance-different windows with descriptor distance <=
the 2 % quantile of all nuisance-different pairs, negatives = those >= the 50 % quantile (<= 200, seeded, shared across
representations); mean over anchors with >= 1 positive; paired representation differences bootstrapped over anchor
groups (recording, 1 s), 2000 replicates.

    python tools/kin_atlas/real.py run --out outputs/kin_atlas/real [--workers 48]
    python tools/kin_atlas/real.py smoke
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.kin_atlas import common as C  # noqa: E402

RUNGS = ((1.0, 2.0), (2.0, 4.0), (3.0, 6.0))
MIN_WINDOWS, MAX_WINDOWS = 400, 3000
TIP_MIN_MM, MIN_EVENTS = 3.0, 100
ND_ROOT_DEG, ND_DEPTH_M = 20.0, 0.05
Q_POS, Q_NEG, N_NEG = 0.02, 0.50, 200
SEED = 20261028


def _rot_mats(rv):
    from scipy.spatial.transform import Rotation as R
    return R.from_rotvec(np.asarray(rv, np.float64)).as_matrix()


def scan_recording(seq):
    """All candidate windows of one recording with the filter quantities (GT geometry and event counts only)."""
    ev, off, aux, pos = C.load_seq(seq)
    betas = np.asarray(aux["betas"], np.float32).reshape(-1)[:10]
    K = np.asarray(aux["camera_K"], np.float64).reshape(3, 3)
    Ke = C.event_K(K)
    ends = []
    for a, b in np.asarray(aux["valid_runs_ms"], np.int64).reshape(-1, 2):
        e = np.arange(a + 49, b, 50, dtype=np.int64)[1:]
        ends.append(e[e - 50 >= a])
    ends = np.concatenate(ends) if ends else np.zeros(0, np.int64)
    s0 = np.stack([np.asarray(pos[int(e) - 50], np.float64) for e in ends])
    s1 = np.stack([np.asarray(pos[int(e)], np.float64) for e in ends])
    v0, j0 = C.fk(s0, betas)
    _, j1 = C.fk(s1, betas)
    R0, R1 = _rot_mats(s0[:, 3:6]), _rot_mats(s1[:, 3:6])
    dang = np.degrees(np.arccos(np.clip((np.einsum("nij,nij->n", R0, R1) - 1) / 2, -1, 1)))
    dwrist = np.linalg.norm(j1[:, 0] - j0[:, 0], axis=1) * 1000
    jh0 = np.einsum("nji,nkj->nki", R0, j0 - j0[:, :1])
    jh1 = np.einsum("nji,nkj->nki", R0, j1 - j1[:, :1])
    desc = (jh1 - jh0)[:, 1:].reshape(len(ends), -1) * 1000
    tipmax = np.linalg.norm((jh1 - jh0)[:, list(C.OP_TIPS)], axis=2).max(1) * 1000
    nev = np.zeros(len(ends), np.int64)
    for i, e in enumerate(ends):
        uv = C.project(v0[i], Ke)
        lo, hi = uv.min(0) - 10, uv.max(0) + 10
        a0, a1 = int(off[int(e) - 49]), int(off[int(e) + 1])
        x = ev[a0:a1, 0].astype(np.float64)
        y = ev[a0:a1, 1].astype(np.float64)
        nev[i] = int(((x >= lo[0]) & (x <= hi[0]) & (y >= lo[1]) & (y <= hi[1])).sum())
    return {"seq": seq, "ends": ends, "s0": s0, "dang": dang, "dwrist": dwrist, "tipmax": tipmax, "nev": nev,
            "desc": desc, "rootmat": R0, "depth": j0[:, 0, 2], "betas": betas, "K": K}


_SEQ_CACHE = {}


def _load(seq):
    if seq not in _SEQ_CACHE:
        _SEQ_CACHE.clear()
        _SEQ_CACHE[seq] = C.load_seq(seq)
    return _SEQ_CACHE[seq]


def window_features(item):
    seq, e, s0, betas, K, pseed = item
    ev, off, aux, pos = _load(seq)
    a0, a1 = int(off[int(e) - 49]), int(off[int(e) + 1])
    ex = ev[a0:a1, 0].astype(np.int64)
    ey = ev[a0:a1, 1].astype(np.int64)
    ep = np.clip(ev[a0:a1, 2].astype(np.int64), 0, 1)
    f = C.representations(ex, ey, ep, s0, betas, K, blur=True)
    r = C.representations(ex, ey, ep, s0, betas, K, blur=False)
    pp = C.representations(ex, ey, ep, C.perturb_prev(s0, betas, np.random.default_rng(pseed)), betas, K, blur=True)
    return ({k: v for k, v in f.items() if not k.startswith("_")}, {k: v for k, v in r.items() if not k.startswith("_")},
            {k: v for k, v in pp.items() if not k.startswith("_")}, float(f["_offband_kin"][0]), float(f["_offband_rigid"][0]))


def anchor_auc(S, D, nd, groups, q_pos, q_neg, rng):
    """Per-anchor AUC for every representation's similarity matrix in S (dict rep -> (n, n)); negatives sampled once."""
    n = D.shape[0]
    iu = np.triu_indices(n, 1)
    dvals = D[iu][nd[iu]]
    t_pos, t_neg = np.quantile(dvals, q_pos), np.quantile(dvals, q_neg)
    out = {rep: [] for rep in S}
    anchors = []
    for i in range(n):
        cand = nd[i].copy()
        cand[i] = False
        pos = np.flatnonzero(cand & (D[i] <= t_pos))
        neg = np.flatnonzero(cand & (D[i] >= t_neg))
        if not len(pos) or not len(neg):
            continue
        if len(neg) > N_NEG:
            neg = rng.choice(neg, N_NEG, replace=False)
        anchors.append(i)
        for rep, M in S.items():
            out[rep].append(C.auc_from(M[i, pos], M[i, neg]))
    return {rep: np.array(v) for rep, v in out.items()}, np.array(anchors), float(t_pos), float(t_neg)


def run(out: Path, workers: int):
    C.verify_protocol()
    if (out / "result.json").exists():
        raise SystemExit(f"{out}/result.json exists; preserve it")
    t0 = time.time()
    seqs = list(C.base_seqs())
    with mp.get_context("spawn").Pool(min(workers, len(seqs))) as pool:
        scans = pool.map(scan_recording, seqs, chunksize=1)
    rung_used, keep = None, None
    for rg in RUNGS:
        k = [np.flatnonzero((s["dang"] <= rg[0]) & (s["dwrist"] <= rg[1]) & (s["tipmax"] >= TIP_MIN_MM) & (s["nev"] >= MIN_EVENTS))
             for s in scans]
        if sum(len(x) for x in k) >= MIN_WINDOWS:
            rung_used, keep = rg, k
            break
    if keep is None:
        rung_used, keep = RUNGS[-1], k
    sel = [(si, int(i)) for si, k in enumerate(keep) for i in k]
    rng = np.random.default_rng(SEED)
    if len(sel) > MAX_WINDOWS:
        sel = [sel[i] for i in sorted(rng.choice(len(sel), MAX_WINDOWS, replace=False))]
    items = [(scans[si]["seq"], int(scans[si]["ends"][i]), scans[si]["s0"][i], scans[si]["betas"], scans[si]["K"],
              SEED + 7 * n) for n, (si, i) in enumerate(sel)]
    with mp.get_context("spawn").Pool(workers) as pool:
        feats = pool.map(window_features, items, chunksize=16)            # items are grouped by recording
    n = len(sel)
    desc = np.stack([scans[si]["desc"][i] for si, i in sel])
    rec = np.array([si for si, _ in sel])
    Rm = np.stack([scans[si]["rootmat"][i] for si, i in sel])
    depth = np.array([scans[si]["depth"][i] for si, i in sel])
    groups = np.array([f"{scans[si]['seq']}|{int(scans[si]['ends'][i]) // 1000}" for si, i in sel])
    from scipy.spatial.distance import cdist
    D = cdist(desc, desc)
    tr = np.einsum("aij,bij->ab", Rm, Rm)                     # trace(R_a^T R_b)
    ang = np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))
    dz = np.abs(depth[:, None] - depth[None])
    nd = (rec[:, None] != rec[None]) | (ang >= ND_ROOT_DEG) | (dz >= ND_DEPTH_M)
    ns = (rec[:, None] == rec[None]) & (ang < 10.0) & (dz < 0.02)
    res = {"rung": rung_used, "n_windows_total_after_filter": int(sum(len(x) for x in keep)), "n_windows": n,
           "per_recording": {scans[si]["seq"]: int((rec == si).sum()) for si in range(len(scans))},
           "offband_kin_median": float(np.median([f[3] for f in feats])),
           "offband_rigid_median": float(np.median([f[4] for f in feats])),
           "frac_pairs_nuisance_different": float(nd[np.triu_indices(n, 1)].mean())}
    for kind, col in (("feats", 0), ("raw", 1), ("pert", 2)):
        S = {}
        for rep in C.REPS:
            F = np.stack([f[col][rep] for f in feats]).astype(np.float64)
            S[rep] = F @ F.T
        auc, anchors, tp, tn = anchor_auc(S, D, nd, groups, Q_POS, Q_NEG, np.random.default_rng(SEED))
        g = groups[anchors]
        blk = {"n_anchors": int(len(anchors)), "d_pos_mm": tp, "d_neg_mm": tn,
               "auc": {rep: {"mean": float(v.mean()), "ci95": list(C.boot_ci(v, g))} for rep, v in auc.items()}}
        best_box = max(("box16", "box24"), key=lambda r: auc[r].mean())
        blk["box_best"] = best_box
        for a_rep, b_rep in (("atlas_kin", best_box), ("atlas_kin", "atlas_rigid"), ("atlas_rigid", best_box), ("atlas_kin", "img")):
            d = auc[a_rep] - auc[b_rep]
            blk[f"diff|{a_rep}-{b_rep}"] = {"mean": float(d.mean()), "ci95": list(C.boot_ci(d, g))}
        if kind == "feats":
            # descriptive: nuisance-same reference and rank correlation with the descriptor over nuisance-different pairs
            auc_s, anc_s, _, _ = anchor_auc(S, D, ns, groups, Q_POS, Q_NEG, np.random.default_rng(SEED))
            blk["nuisance_same_auc"] = {rep: float(v.mean()) if len(v) else None for rep, v in auc_s.items()}
            blk["nuisance_same_n_anchors"] = int(len(anc_s))
            from scipy.stats import spearmanr
            iu = np.triu_indices(n, 1)
            m = nd[iu]
            pick = np.random.default_rng(SEED).choice(int(m.sum()), min(200000, int(m.sum())), replace=False)
            dv = D[iu][m][pick]
            blk["spearman_sim_vs_neg_desc"] = {rep: float(spearmanr(S[rep][iu][m][pick], -dv).correlation) for rep in C.REPS}
        res[kind] = blk
    res["wall_s"] = time.time() - t0
    C.save_json(out / "result.json", res)
    print(f"[KA real] {n} windows (rung {rung_used}) in {time.time() - t0:.0f} s -> {out}/result.json")


def smoke():
    """Shapes and timings only: scans one recording but prints no filter count or similarity."""
    t0 = time.time()
    s = scan_recording("ch_local")
    i = 5
    f = window_features(("ch_local", int(s["ends"][i]), s["s0"][i], s["betas"], s["K"], 1))
    print(f"[smoke] scanned {len(s['ends'])} windows of ch_local; feature dims "
          f"{ {k: v.shape for k, v in f[0].items()} }; {time.time() - t0:.1f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("run", "smoke"))
    ap.add_argument("--out", default=str(C.OUT / "real"))
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    C.force_cpu()
    if a.stage == "smoke":
        return smoke()
    return run(Path(a.out), a.workers)


if __name__ == "__main__":
    main()
