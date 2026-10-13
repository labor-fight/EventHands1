#!/usr/bin/env python3
"""[XYZ-ASSOC] P1 cross-check of the repository's opt-in rule semkine.xyz_mesh.associate_rays(mode="contour") against the
P1 reimplementations `alt` (no-hit fallback = unfiltered near layer, the primary T1-e reading) and `altf` (no-hit
fallback = back-face-filtered near layer).  Read-only on the repository; CPU only.

Phase 1 (detailed): >= 48 windows per sequence (zgz_local: 24 with fingertip_in_front, 24 without; zgz_global: 48 random),
  sampled events, regime R0.  Per event: ctr set, per-record vid / mass (q*A) / z (tol 1e-5), r = 1-q, valid, hit.
Phase 2 (full): repo contour rule on the whole sampled set (all windows) at R0 -> M, exact / group agreement, to be
  compared with assoc_stats.json (alt / altf / cur).
Usage: step1_check_repo_contour.py [n_windows_per_seq] [procs]
"""
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import step1_assoc_regimes as A                                    # noqa: E402
import step1_common as C                                           # noqa: E402
import numpy as np                                                 # noqa: E402
import torch                                                       # noqa: E402
from semkine.xyz_mesh import associate_rays, stride_sample, vertex_normals as repo_normals   # noqa: E402

TOL = 1e-5
KEYS = ("ctr", "vid", "w", "z", "r", "valid", "hit")


def step_inputs(s):
    """Sampled events of window s at R0: (dp, mask, vb, nb, band, front, take, samp, ms) for the R0 prev grid."""
    g = A._G
    end = int(g["ends"][s])
    ev5 = C._window_events(g["events"], g["offsets"], g["tsub"], end, 50)
    n = len(ev5)
    src, mk = stride_sample(torch.tensor([0, n], dtype=torch.int64), C.CAP)
    samp = src[0][mk[0]].numpy()
    px, py = torch.from_numpy(ev5[:, 1].copy()).float(), torch.from_numpy(ev5[:, 2].copy()).float()
    d_all = torch.stack([(px - g["cx"]) / g["fx"], (py - g["cy"]) / g["fy"], torch.ones_like(px)], -1)
    verts = A._regime_grids(int(s))
    vn = A.vertex_normals(verts, g["faces"])
    ms_rel = np.repeat(np.arange(50), np.diff(g["offsets"][end - 49:end + 2]).astype(np.int64))
    return (*A._batch(samp, ms_rel, d_all, "R0", verts, vn), samp)


def run_variants(s):
    g = A._G
    dp, mask, vb, nb, band, front, take, samp = step_inputs(s)
    m = len(samp)
    prod = associate_rays(dp, mask, vb, g["faces"], g["vert_faces"], band=band, front_tol=front, z_tol=C.Z_TOL, k=C.KNN, chunk=A.CHUNK,
                          mode="contour", normals=repo_normals(vb, g["faces"]), gap=0.01)
    cur, alt, altf = A.associate_dual(dp, mask, vb, g["faces"], g["vert_faces"], nb, band, front, C.Z_TOL, C.KNN, 0.01, A.CHUNK)
    P = {k: take(v) for k, v in prod.items()}
    rec = {"repo": dict(vid=P["vid"], w=P["q"].unsqueeze(1) * P["A"], zr=P["z"], r=1.0 - P["q"], valid=P["valid"], hit=P["hit"], ctr=P["ctr"].any(1)),
           "alt": {k: take(v) for k, v in alt.items()}, "altf": {k: take(v) for k, v in altf.items()}}
    nu = {k: take(v) for k, v in cur.items()}                    # unfiltered near layer (cur) for the no-hit analysis
    return m, samp, rec, nu


def diff_events(a, b):
    """Per-event boolean arrays: which fields of variant a differ from b."""
    wa, wb = a["w"], b["w"]
    live = (wa > 0) | (wb > 0)
    return {"ctr": a["ctr"] != b["ctr"],
            "vid": ((a["vid"] != b["vid"]) & live).any(1),
            "w": (wa - wb).abs().max(1).values > TOL,
            "z": ((a["zr"] - b["zr"]).abs() * live).max(1).values > TOL,
            "r": (a["r"] - b["r"]).abs() > TOL,
            "valid": a["valid"] != b["valid"], "hit": a["hit"] != b["hit"]}


def phase1(seq, legacy, lab, n_win, rng):
    g = A._G
    S = len(g["ends"])
    fip = lab["fingertip_in_front"]
    if fip.any():
        k1 = min(n_win // 2, int(fip.sum()))
        pick = np.concatenate([rng.choice(np.where(fip)[0], k1, replace=False), rng.choice(np.where(~fip)[0], n_win - k1, replace=False)])
    else:
        pick = rng.choice(S, n_win, replace=False)
    pick = np.sort(pick)
    out = {"n_windows": int(len(pick)), "n_windows_fip_true": int(fip[pick].sum()), "n_events": 0}
    agg = {nm: {k: 0 for k in KEYS + ("any",)} for nm in ("alt", "altf")}
    reasons = {nm: {} for nm in ("alt", "altf")}
    ctr_sets = {"repo": 0, "alt": 0, "altf": 0, "repo_and_alt": 0, "repo_and_altf": 0}
    acc = {v: {"mden": 0, "mnum": 0, "hand": 0, "ex": 0, "gr": 0, "mden_f1": 0, "mnum_f1": 0} for v in ("repo", "alt", "altf", "cur")}
    nohit_filtering = {"nohit_events": 0, "nohit_unfiltered_ne_filtered": 0, "repo_eq_filtered": 0, "repo_eq_unfiltered": 0}
    for s in pick:
        m, samp, rec, nu = run_variants(int(s))
        e0 = int(lab["step_ev_start"][s])
        out["n_events"] += m
        for nm in ("alt", "altf"):
            df = diff_events(rec["repo"], rec[nm])
            anyd = torch.zeros(m, dtype=torch.bool)
            for k_ in KEYS:
                agg[nm][k_] += int(df[k_].sum()); anyd |= df[k_]
            agg[nm]["any"] += int(anyd.sum())
            # reason categories of differing events
            hit = rec["repo"]["hit"]
            for lbl, msk in (("no_hit_event", ~hit), ("hit_event_no_ctr_either", hit & ~rec["repo"]["ctr"] & ~rec[nm]["ctr"]), ("hit_event_ctr_differs", hit & df["ctr"])):
                reasons[nm][lbl] = reasons[nm].get(lbl, 0) + int((anyd & msk).sum())
        ctr_sets["repo"] += int(rec["repo"]["ctr"].sum()); ctr_sets["alt"] += int(rec["alt"]["ctr"].sum()); ctr_sets["altf"] += int(rec["altf"]["ctr"].sum())
        ctr_sets["repo_and_alt"] += int((rec["repo"]["ctr"] & rec["alt"]["ctr"]).sum()); ctr_sets["repo_and_altf"] += int((rec["repo"]["ctr"] & rec["altf"]["ctr"]).sum())
        # no-hit events: does the repo near layer equal the filtered (altf) or the unfiltered (alt = cur) one?
        nh = ~rec["repo"]["hit"]
        cur_rec = dict(vid=nu["vid"], w=nu["q"].unsqueeze(1) * nu["A"], zr=nu["z"].unsqueeze(1).expand(m, 4), r=1.0 - nu["q"], valid=nu["valid"], hit=nu["hit"], ctr=torch.zeros(m, dtype=torch.bool))
        # compare on the first 4 records (cur has 4)
        sub = lambda r_: dict(vid=r_["vid"][:, :4], w=r_["w"][:, :4], zr=r_["zr"][:, :4], r=r_["r"], valid=r_["valid"], hit=r_["hit"], ctr=r_["ctr"])
        d_f = diff_events(sub(rec["repo"]), sub(rec["altf"])); d_u = diff_events(sub(rec["repo"]), cur_rec)
        any_f = torch.stack([d_f[k_] for k_ in KEYS]).any(0); any_u = torch.stack([d_u[k_] for k_ in KEYS]).any(0)
        nohit_filtering["nohit_events"] += int(nh.sum())
        nohit_filtering["nohit_unfiltered_ne_filtered"] += int((nh & (torch.stack([diff_events(sub(rec["altf"]), cur_rec)[k_] for k_ in KEYS]).any(0))).sum())
        nohit_filtering["repo_eq_filtered"] += int((nh & ~any_f).sum()); nohit_filtering["repo_eq_unfiltered"] += int((nh & ~any_u).sum())
        # metrics on this subset with each variant's output
        kind = lab["src_kind"][e0 + samp].astype(int); part = lab["src_part"][e0 + samp].astype(int)
        fp = bool(fip[s])
        for v, r_ in (("repo", rec["repo"]), ("alt", rec["alt"]), ("altf", rec["altf"]),
                      ("cur", dict(vid=nu["vid"], w=nu["q"].unsqueeze(1) * nu["A"], zr=nu["z"].unsqueeze(1).expand(m, 4), r=1.0 - nu["q"], valid=nu["valid"], hit=nu["hit"], ctr=torch.zeros(m, dtype=torch.bool)))):
            sm = A.summarize(r_["vid"], r_["w"], r_["zr"], r_["r"], r_["valid"], r_["hit"], r_["ctr"], g["hard"])
            md = (kind >= 1) & (part >= 1)
            hd = part >= 0
            pred = sm["pred"].astype(int)
            acc[v]["mden"] += int(md.sum()); acc[v]["mnum"] += int((md & sm["maj0"]).sum())
            acc[v]["hand"] += int(hd.sum()); acc[v]["ex"] += int((hd & (pred == part)).sum())
            acc[v]["gr"] += int((hd & (pred >= 0) & (C.JOINT_GROUP[np.maximum(pred, 0)] == C.JOINT_GROUP[np.maximum(part, 0)])).sum())
            if fp:
                acc[v]["mden_f1"] += int(md.sum()); acc[v]["mnum_f1"] += int((md & sm["maj0"]).sum())
    out["field_mismatch_events_repo_vs"] = agg
    out["reasons_of_differing_events"] = reasons
    out["ctr_trigger_sets"] = ctr_sets
    out["no_hit_events_near_layer"] = nohit_filtering
    out["subset_metrics"] = {v: {"M": a["mnum"] / max(a["mden"], 1), "M_den": a["mden"], "exact": a["ex"] / max(a["hand"], 1), "group": a["gr"] / max(a["hand"], 1),
                                 "hand_n": a["hand"], "M_fip1": (a["mnum_f1"] / a["mden_f1"]) if a["mden_f1"] else None, "M_fip1_den": a["mden_f1"]} for v, a in acc.items()}
    return out


# ---------------- phase 2: the whole sampled set, repo rule only
def phase2_task(task):
    j, steps = task
    g = A._G
    lab = g["lab"]
    acc = np.zeros(8)
    for s in steps:
        dp, mask, vb, nb, band, front, take, samp = step_inputs(s)
        prod = associate_rays(dp, mask, vb, g["faces"], g["vert_faces"], band=band, front_tol=front, z_tol=C.Z_TOL, k=C.KNN, chunk=A.CHUNK,
                              mode="contour", normals=repo_normals(vb, g["faces"]), gap=0.01)
        P = {k: take(v) for k, v in prod.items()}
        sm = A.summarize(P["vid"], P["q"].unsqueeze(1) * P["A"], P["z"], 1.0 - P["q"], P["valid"], P["hit"], P["ctr"].any(1), g["hard"])
        e0 = int(lab["step_ev_start"][s])
        kind = lab["src_kind"][e0 + samp].astype(int); part = lab["src_part"][e0 + samp].astype(int)
        md, hd = (kind >= 1) & (part >= 1), part >= 0
        pred = sm["pred"].astype(int)
        acc += [md.sum(), (md & sm["maj0"]).sum(), hd.sum(), (hd & (pred == part)).sum(),
                (hd & (pred >= 0) & (C.JOINT_GROUP[np.maximum(pred, 0)] == C.JOINT_GROUP[np.maximum(part, 0)])).sum(), sm["ctr"].sum(), len(samp), 0]
    return acc


def main():
    n_win = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    stats = json.load(open(C.OUT_DIR / "assoc_stats.json"))
    res = {"tol": TOL, "repo_default_mode_note": "hit_first path is checked separately in check_vs_contour_impl.json"}
    rng = np.random.default_rng(7)
    for seq, legacy in C.sequences():
        z = np.load(C.OUT_DIR / f"labels_{seq}.npz")
        lab = {k: z[k] for k in z.files}
        S = len(lab["step_end_ms"])
        A._init(seq, legacy, "sampled", lab, np.zeros((S, 51), np.float32))
        torch.set_num_threads(1)
        res[seq] = {"phase1": phase1(seq, legacy, lab, n_win, rng)}
        print(seq, "phase1", json.dumps(res[seq]["phase1"], indent=0)[:3000], flush=True)
        t0 = time.time()
        tasks = [(j, list(range(S))[j::procs]) for j in range(min(procs, S))]
        with mp.get_context("fork").Pool(len(tasks), initializer=torch.set_num_threads, initargs=(1,)) as pool:
            tot = sum(pool.map(phase2_task, tasks))
        sm = stats["sets"]["sampled"][seq]["R0"]
        res[seq]["phase2_full_sampled_set_R0"] = {
            "repo": {"M": tot[1] / tot[0], "M_den": int(tot[0]), "exact": tot[3] / tot[2], "group": tot[4] / tot[2], "hand_n": int(tot[2]),
                     "ctr_rate": tot[5] / tot[6], "n_events": int(tot[6])},
            "report_alt": {"M": sm["alt"]["M"]["maj0_rate"], "exact": sm["alt"]["hand"]["exact"], "group": sm["alt"]["hand"]["group"], "ctr_rate": sm["alt"]["all_events"]["ctr_rate"]},
            "report_altf": {"M": sm["altf"]["M"]["maj0_rate"], "exact": sm["altf"]["hand"]["exact"], "group": sm["altf"]["hand"]["group"], "ctr_rate": sm["altf"]["all_events"]["ctr_rate"]},
            "report_cur": {"M": sm["cur"]["M"]["maj0_rate"], "exact": sm["cur"]["hand"]["exact"], "group": sm["cur"]["hand"]["group"]},
            "seconds": time.time() - t0}
        print(seq, "phase2", json.dumps(res[seq]["phase2_full_sampled_set_R0"]), flush=True)
    (C.OUT_DIR / "check_repo_contour.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
