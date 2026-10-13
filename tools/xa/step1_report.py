#!/usr/bin/env python3
"""[XYZ-ASSOC] P1 step 3: metrics, T1 criteria and the report from the tables of step1_assoc_regimes.py.

Reads  outputs/xa/step1/{labels,assoc_tables,assoc_checks,assoc_events}_<seq>.npz|json
Writes outputs/xa/step1/assoc_stats.json and REPORT.md  (numbers only; no verdict on 'bottleneck').
Pure numpy (no torch).
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
REPO = Path("/data1/lyq/code/mesh/EventHands1")
SEQS = ["zgz_local", "zgz_global"]
REGIMES = ["R3", "R0", "R1s", "R1l"]
RULES = ["cur", "alt", "altf"]
RULE_NAME = {"cur": "现行(associate_rays)", "alt": "替代(contour)", "altf": "替代-滤波回退变体(altf)"}
REG_NAME = {"R3": "R3 逐毫秒真值", "R0": "R0 窗首真值", "R1s": "R1s 窗首真值+小噪声", "R1l": "R1l 窗首真值+大噪声"}
CNT = ["n", "valid", "hit", "resid", "agree_exact", "agree_group", "maj0", "plur0", "m0frac_sum"]
GRP_NAMES = ["非手", "掌(关节0)", "食指(1-3)", "中指(4-6)", "小指(7-9)", "无名指(10-12)", "拇指(13-15)"]
KIND_NAMES = {-1: "非手(-1)", 0: "内部(0)", 1: "剪影(1)", 2: "深度不连续(2)"}
JOINT_GROUP = np.array([0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5])
SH = None            # table shape constants, filled from the checks file


class View:
    """Sufficient-statistic table of one (set, regime, rule): counters over cells (fip, kind+1, grp, atype, ctr)."""

    def __init__(self, cnt, hz, conf):
        self.c = cnt.reshape(len(CNT), 2, 4, 7, 3, 2)
        self.hz, self.conf = hz, conf

    def s(self, name, fip=None, kind=None, grp=None, at=None, ctr=None):
        a = self.c[CNT.index(name)]
        ix = [np.arange(d) if v is None else np.atleast_1d(v) for v, d in zip((fip, kind, grp, at, ctr), a.shape)]
        return float(a[np.ix_(*ix)].sum())


def metrics(v, fip=None, kind=None, grp=None, at=None):
    """kind is given as a list of src_kind values (-1..2); converted to table index (+1)."""
    kidx = None if kind is None else [k + 1 for k in kind]
    S = lambda nm, **kw: v.s(nm, fip=fip, kind=kidx, grp=grp, at=at, **kw)
    n = S("n")
    out = {"n": int(n)}
    if n == 0:
        return out
    nv, nh = S("valid"), S("hit")
    out.update(valid_rate=nv / n, hit_rate=nh / n, resid_mean=S("resid") / n, ctr_rate=S("n", ctr=1) / n)
    hand = grp is None or all(g >= 1 for g in np.atleast_1d(grp))
    if hand:
        out.update(exact=S("agree_exact") / n, group=S("agree_group") / n,
                   exact_given_valid=S("agree_exact") / nv if nv else None,
                   group_given_valid=S("agree_group") / nv if nv else None)
    out.update(maj0_rate=S("maj0") / n, plur0_rate=S("plur0") / n,
               maj0_given_valid=S("maj0") / nv if nv else None,
               m0frac_mean_given_valid=S("m0frac_sum") / nv if nv else None)
    return out


def M_of(v, fip=None):
    """Contour mismatch rate M: source = finger contour foreground (kind in {1,2}, part >= 1), majority associated mass on joint 0."""
    return metrics(v, fip=fip, kind=[1, 2], grp=[2, 3, 4, 5, 6])


# ---------------------------------------------------------------- z-error histograms
def _hist_q(h, qs):
    tot = h.sum()
    if tot == 0:
        return [None] * len(qs)
    cum = np.cumsum(h)
    return [int(np.searchsorted(cum, q * tot)) for q in qs]


def zstats(h):
    """h: (NBIN,) counts of z_prior - src_depth in 0.25 mm bins over +-300 mm (edge bins clip)."""
    NBIN, BIN, ZR = SH["NBIN"], SH["BIN_MM"], SH["ZRANGE"]
    n = int(h.sum())
    if n == 0:
        return {"n": 0}
    cen = (np.arange(NBIN) + 0.5) * BIN - ZR
    q = _hist_q(h, [0.1, 0.5, 0.9])
    ab = np.bincount(np.round((np.abs(cen) - BIN / 2) / BIN).astype(int), weights=h, minlength=NBIN // 2)
    qa = _hist_q(ab, [0.5, 0.9])
    return {"n": n, "signed_median_mm": float(cen[q[1]]), "signed_p10_mm": float(cen[q[0]]), "signed_p90_mm": float(cen[q[2]]),
            "abs_median_mm": float((qa[0] + 0.5) * BIN), "abs_p90_mm": float((qa[1] + 0.5) * BIN),
            "frac_abs_ge_10mm": float(ab[int(10 / BIN):].sum() / n), "mean_mm": float((h * cen).sum() / n)}


def zviews(v):
    HA, HB = SH["HA"], SH["HB"]
    A = v.hz[:HA].reshape(2, 4, 2, 2, -1)            # at-1, kind+1, mism, fip
    B = v.hz[HA:HA + HB].reshape(2, 7, -1)           # at-1, grp
    Cc = v.hz[HA + HB:].reshape(2, 4, 2, -1)         # which(front/hit), kind+1, mism
    return A, B, Cc


def z_block(v):
    A, B, Cc = zviews(v)
    out = {}
    names = {1: "near", 2: "hit"}
    for at in (1, 2):
        out[f"{names[at]}_all_hand"] = zstats(A[at - 1].sum((0, 1, 2)))
        for kd in (0, 1, 2):
            out[f"{names[at]}_kind{kd}"] = zstats(A[at - 1, kd + 1].sum((0, 1)))
        for fp in (0, 1):
            out[f"{names[at]}_fip{fp}"] = zstats(A[at - 1, :, :, fp].sum((0, 1)))
    out["hit_or_near_all_hand"] = zstats(A.sum((0, 1, 2, 3)))
    # mismatch events: source finger contour, majority mass on joint 0 (the M numerator)
    for at in (1, 2):
        out[f"mismatch_{names[at]}"] = zstats(A[at - 1, 2:4, 1].sum((0, 1)))
    out["mismatch_all"] = zstats(A[:, 2:4, 1].sum((0, 1, 2)))
    out["nonmismatch_finger_contour_all"] = zstats(A[:, 2:4, 0].sum((0, 1, 2)))
    for gi in range(1, 7):
        out[f"group_{gi}_all"] = zstats(B[:, gi].sum(0))
    out["ctr_front_record_mismatch"] = zstats(Cc[0, 2:4, 1].sum(0))
    out["ctr_front_record_nonmismatch"] = zstats(Cc[0, 2:4, 0].sum(0))
    out["ctr_hit_record_mismatch"] = zstats(Cc[1, 2:4, 1].sum(0))
    out["ctr_hit_record_nonmismatch"] = zstats(Cc[1, 2:4, 0].sum(0))
    out["ctr_front_record_all"] = zstats(Cc[0].sum((0, 1)))
    out["ctr_hit_record_all"] = zstats(Cc[1].sum((0, 1)))
    return out


def stratify(v):
    d = {"all_events": metrics(v), "hand": metrics(v, grp=[1, 2, 3, 4, 5, 6]), "nonhand": metrics(v, grp=[0]),
         "M": M_of(v)}
    for fp in (0, 1):
        d[f"fip{fp}_hand"] = metrics(v, fip=[fp], grp=[1, 2, 3, 4, 5, 6])
        d[f"fip{fp}_M"] = M_of(v, fip=[fp])
    for kd in (0, 1, 2):
        d[f"kind{kd}_hand"] = metrics(v, kind=[kd], grp=[1, 2, 3, 4, 5, 6])
    for gi in range(1, 7):
        d[f"group{gi}"] = metrics(v, grp=[gi])
        d[f"group{gi}_contour"] = metrics(v, kind=[1, 2], grp=[gi])
        d[f"group{gi}_interior"] = metrics(v, kind=[0], grp=[gi])
    d["fingers_all_kinds"] = metrics(v, grp=[2, 3, 4, 5, 6])
    d["palm_contour"] = metrics(v, kind=[1, 2], grp=[1])
    d["finger_contour_fip1"] = M_of(v, fip=[1])
    for at, nm in ((2, "hit"), (1, "near"), (0, "none")):
        d[f"atype_{nm}_hand"] = metrics(v, grp=[1, 2, 3, 4, 5, 6], at=[at])
    d["nonhand_hit_rate"] = d["nonhand"].get("hit_rate")
    d["z"] = z_block(v)
    # confusion: source part x associated joint, per kind class
    cf = v.conf.reshape(4, 17, 17)
    d["confusion_contour_finger_by_srcgroup_to_pred"] = {}
    for gi in range(2, 7):
        parts = [p for p in range(16) if JOINT_GROUP[p] == gi - 1]
        sub = cf[2:4][:, [p + 1 for p in parts]].sum((0, 1))         # (pred+1)
        tot = sub.sum()
        d["confusion_contour_finger_by_srcgroup_to_pred"][GRP_NAMES[gi]] = {
            "n": int(tot), "pred_none": float(sub[0] / tot) if tot else None,
            "pred_joint0": float(sub[1] / tot) if tot else None,
            "pred_same_group": float(sub[[j + 1 for j in range(16) if JOINT_GROUP[j] == gi - 1]].sum() / tot) if tot else None,
            "pred_other_finger": float(sub[[j + 1 for j in range(16) if JOINT_GROUP[j] not in (0, gi - 1)]].sum() / tot) if tot else None}
    return d


# ---------------------------------------------------------------- bootstrap (step clusters)
def boot_block(seq, lab, ev, nb=2000, seed=0):
    rng = np.random.default_rng(seed)
    pos, st = ev["samp_lab_pos"], ev["samp_step"]
    S = int(lab["step_end_ms"].shape[0])
    kind, part = lab["src_kind"][pos].astype(int), lab["src_part"][pos].astype(int)
    mden = (kind >= 1) & (part >= 1)
    hand = part >= 0
    bc = lambda m: np.bincount(st, weights=m.astype(np.float64), minlength=S)
    den_M, den_h = bc(mden), bc(hand)
    res = {}
    idx = rng.integers(0, S, size=(nb, S))
    def ci(r):
        lo, hi = np.nanpercentile(r, [2.5, 97.5])
        return [float(lo), float(hi)]
    for rg in REGIMES:
        for ru in RULES:
            num = bc(mden & ev[f"{rg}_{ru}_maj0"])
            agr = bc(hand & (ev[f"{rg}_{ru}_pred"].astype(int) == part))
            Mpt = num.sum() / max(den_M.sum(), 1)
            r = num[idx].sum(1) / np.maximum(den_M[idx].sum(1), 1)
            e = agr[idx].sum(1) / np.maximum(den_h[idx].sum(1), 1)
            res[f"{rg}_{ru}"] = {"M": float(Mpt), "M_ci95_step_cluster": ci(r),
                                 "exact": float(agr.sum() / max(den_h.sum(), 1)), "exact_ci95_step_cluster": ci(e)}
    # paired quantities used by T1-c / T1-e
    for rg in REGIMES:
        nc, na = bc(mden & ev[f"{rg}_cur_maj0"]), bc(mden & ev[f"{rg}_alt_maj0"])
        nf = bc(mden & ev[f"{rg}_altf_maj0"])
        ac = bc(hand & (ev[f"{rg}_cur_pred"].astype(int) == part))
        aa = bc(hand & (ev[f"{rg}_alt_pred"].astype(int) == part))
        af = bc(hand & (ev[f"{rg}_altf_pred"].astype(int) == part))
        for nm, nx, ax in (("alt", na, aa), ("altf", nf, af)):
            rel = 1.0 - nx[idx].sum(1) / np.maximum(nc[idx].sum(1), 1e-9)
            dex = (ax[idx].sum(1) - ac[idx].sum(1)) / np.maximum(den_h[idx].sum(1), 1)
            res[f"{rg}_{nm}_vs_cur"] = {"M_rel_drop": float(1.0 - nx.sum() / max(nc.sum(), 1e-9)),
                                        "M_rel_drop_ci95": ci(rel),
                                        "exact_diff_pp": float((ax.sum() - ac.sum()) / max(den_h.sum(), 1) * 100),
                                        "exact_diff_pp_ci95": [x * 100 for x in ci(dex)]}
    for ru in RULES:      # T1-c decompositions
        n3, n0, n1s, n1l = (bc(mden & ev[f"{rg}_{ru}_maj0"]) for rg in REGIMES)
        d = lambda a, b: (a[idx].sum(1) / np.maximum(den_M[idx].sum(1), 1) - b[idx].sum(1) / np.maximum(den_M[idx].sum(1), 1))
        res[f"T1c_{ru}"] = {
            "lag_R0_minus_R3": float((n0.sum() - n3.sum()) / max(den_M.sum(), 1)), "lag_ci95": ci(d(n0, n3)),
            "noise_R1s_minus_R0": float((n1s.sum() - n0.sum()) / max(den_M.sum(), 1)), "noise_s_ci95": ci(d(n1s, n0)),
            "noise_R1l_minus_R1s": float((n1l.sum() - n1s.sum()) / max(den_M.sum(), 1)), "noise_l_ci95": ci(d(n1l, n1s))}
    res["n_finger_contour_events"] = int(den_M.sum()); res["n_hand_events"] = int(den_h.sum()); res["n_steps"] = S
    return res


# ---------------------------------------------------------------- formatting
pc = lambda x, d=1: "—" if x is None else f"{100 * x:.{d}f}%"
mm = lambda x, d=1: "—" if x is None else f"{x:.{d}f}"
sn = lambda s: s.replace("zgz_", "")


def table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    global SH
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=str(REPO / "outputs/xa/step1"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--lab-dir", default=None)
    a = ap.parse_args()
    ind = Path(a.in_dir)
    outd = Path(a.out_dir or a.in_dir)
    labd = Path(a.lab_dir or a.in_dir)
    stats = {"sets": {}, "checks": {}, "labels": {}, "bootstrap": {}}
    V = {}                 # (set, seq, regime, rule) -> View
    sets_present = set()
    for seq in SEQS:
        chk = json.loads((ind / f"assoc_checks_{seq}.json").read_text())
        SH = chk["shape"]
        stats["checks"][seq] = chk
        z = np.load(ind / f"assoc_tables_{seq}.npz")
        for k in z.files:
            if k.endswith("|cnt"):
                st, rg, ru, _ = k.split("|")
                V[(st, seq, rg, ru)] = View(z[k], z[f"{st}|{rg}|{ru}|hz"], z[f"{st}|{rg}|{ru}|conf"])
                sets_present.add(st)
    for st in sorted(sets_present):
        stats["sets"][st] = {seq: {rg: {ru: stratify(V[(st, seq, rg, ru)]) for ru in RULES} for rg in REGIMES} for seq in SEQS}
    # labels
    for seq in SEQS:
        lz = np.load(labd / f"labels_{seq}.npz")
        k = lz["src_kind"]; p = lz["src_part"]
        lab_small = {"step_end_ms": lz["step_end_ms"], "src_kind": None}
        stats["labels"][seq] = {
            "n_events": int(len(k)), "n_steps": int(len(lz["step_end_ms"])), "n_runs": int(len(np.unique(lz["run_id"]))),
            "kind_frac": {str(kk): float(np.mean(k == kk)) for kk in (-1, 0, 1, 2)},
            "part_hist": np.bincount(p[p >= 0], minlength=16).tolist(),
            "finger_contour_frac": float(np.mean((k >= 1) & (p >= 1))),
            "fingertip_in_front_steps": float(lz["fingertip_in_front"].mean()),
            "fingertip_in_front_local_steps": float(lz["fingertip_in_front_local"].mean()),
            "any_tip_in_palm_hull_steps": float(np.mean(lz["tips_in_hull"] > 0)),
            "n_sampled_events": int(len(lz["samp_local"])),
            "kind_frac_sampled": {str(kk): float(np.mean(k[lz["step_ev_start"][:-1].repeat(np.diff(lz["samp_start"])) + lz["samp_local"]] == kk))
                                  for kk in (-1, 0, 1, 2)},
        }
        evp = ind / f"assoc_events_{seq}.npz"
        if evp.exists():
            lab = {"step_end_ms": lz["step_end_ms"], "src_kind": k, "src_part": p}
            ev = np.load(evp)
            ev = {kk: ev[kk] for kk in ev.files}
            b = boot_block(seq, lab, ev)
            # independent recomputation of table-based numbers from the per-event arrays
            mt = stats["sets"]["sampled"][seq]["R0"]["cur"]["M"]["maj0_rate"]
            b["crosscheck_M_R0_cur_table_minus_events"] = float(mt - b["R0_cur"]["M"])
            ex_t = stats["sets"]["sampled"][seq]["R0"]["cur"]["hand"]["exact"]
            b["crosscheck_exact_R0_cur_table_minus_events"] = float(ex_t - b["R0_cur"]["exact"])
            stats["bootstrap"][seq] = b
    # ---------------- T1 judgements (set = sampled, regime R0, rule cur / alt)
    S0 = stats["sets"]["sampled"]
    Mc = {s: S0[s]["R0"]["cur"]["M"]["maj0_rate"] for s in SEQS}
    Ma = {s: S0[s]["R0"]["alt"]["M"]["maj0_rate"] for s in SEQS}
    Maf = {s: S0[s]["R0"]["altf"]["M"]["maj0_rate"] for s in SEQS}
    nM = {s: S0[s]["R0"]["cur"]["M"]["n"] for s in SEQS}
    T1 = {}
    T1["a"] = {"condition": "R0 下 M(local) >= 20% 且 M(global) <= 5%", "M_local": Mc["zgz_local"], "M_global": Mc["zgz_global"],
               "n_local": nM["zgz_local"], "n_global": nM["zgz_global"],
               "pass": bool(Mc["zgz_local"] >= 0.20 and Mc["zgz_global"] <= 0.05)}
    T1["b"] = {"condition": "R0 下 M(local) < 5% 且 M(global) < 5%", "M_local": Mc["zgz_local"], "M_global": Mc["zgz_global"],
               "pass": bool(Mc["zgz_local"] < 0.05 and Mc["zgz_global"] < 0.05)}
    T1["ab_summary"] = "T1-a 通过" if T1["a"]["pass"] else ("T1-b 条件满足" if T1["b"]["pass"] else "混合(T1-a、T1-b 均不成立)")
    T1["c"] = {}
    for ru in RULES:
        T1["c"][ru] = {s: {rg: S0[s][rg][ru]["M"]["maj0_rate"] for rg in REGIMES} for s in SEQS}
        for s in SEQS:
            m = T1["c"][ru][s]
            m["lag_R0_minus_R3"] = m["R0"] - m["R3"]; m["noise_R1s_minus_R0"] = m["R1s"] - m["R0"]
            m["noise_R1l_minus_R1s"] = m["R1l"] - m["R1s"]
    T1["d"] = {}
    for s in SEQS:
        zm = S0[s]["R0"]["cur"]["z"]["mismatch_all"]
        fullconf = None
        v = V[("sampled", s, "R0", "cur")]
        n_m = v.s("maj0", kind=[2, 3], grp=[2, 3, 4, 5, 6])
        n_mh = v.s("maj0", kind=[2, 3], grp=[2, 3, 4, 5, 6], at=[2])
        T1["d"][s] = {"n_mismatch": int(n_m), "frac_mismatch_via_hit_q1": n_mh / n_m if n_m else None, **zm,
                      "pass_signed": bool(zm["n"] >= 100 and zm["signed_median_mm"] >= 10.0) if zm["n"] else None,
                      "pass_abs": bool(zm["n"] >= 100 and zm["abs_median_mm"] >= 10.0) if zm["n"] else None,
                      "enough_n": bool(zm["n"] >= 100)}
    # pooled over both sequences
    pooled = sum((zviews(V[("sampled", s, "R0", "cur")])[0][:, 2:4, 1].sum((0, 1, 2)) for s in SEQS))
    pz = zstats(pooled)
    n_m = sum(V[("sampled", s, "R0", "cur")].s("maj0", kind=[2, 3], grp=[2, 3, 4, 5, 6]) for s in SEQS)
    n_mh = sum(V[("sampled", s, "R0", "cur")].s("maj0", kind=[2, 3], grp=[2, 3, 4, 5, 6], at=[2]) for s in SEQS)
    T1["d"]["pooled"] = {"n_mismatch": int(n_m), "frac_mismatch_via_hit_q1": n_mh / n_m if n_m else None, **pz,
                         "pass_signed": bool(pz["n"] >= 100 and pz["signed_median_mm"] >= 10.0),
                         "pass_abs": bool(pz["n"] >= 100 and pz["abs_median_mm"] >= 10.0), "enough_n": bool(pz["n"] >= 100)}
    T1["e"] = {"M_local_cur": Mc["zgz_local"], "M_local_alt": Ma["zgz_local"], "M_local_altf": Maf["zgz_local"]}
    rel = (1 - Ma["zgz_local"] / Mc["zgz_local"]) if Mc["zgz_local"] > 0 else None
    relf = (1 - Maf["zgz_local"] / Mc["zgz_local"]) if Mc["zgz_local"] > 0 else None
    exg = {r: S0["zgz_global"]["R0"][r]["hand"]["exact"] for r in RULES}
    exl = {r: S0["zgz_local"]["R0"][r]["hand"]["exact"] for r in RULES}
    T1["e"].update(M_local_rel_drop_alt=rel, M_local_rel_drop_altf=relf,
                   exact_global={r: exg[r] for r in RULES}, exact_local={r: exl[r] for r in RULES},
                   exact_global_diff_pp_alt=(exg["alt"] - exg["cur"]) * 100, exact_global_diff_pp_altf=(exg["altf"] - exg["cur"]) * 100,
                   cond1_alt=bool(rel is not None and rel >= 0.5), cond2_alt=bool((exg["alt"] - exg["cur"]) * 100 >= -2.0),
                   cond1_altf=bool(relf is not None and relf >= 0.5), cond2_altf=bool((exg["altf"] - exg["cur"]) * 100 >= -2.0))
    T1["e"]["pass_alt"] = bool(T1["e"]["cond1_alt"] and T1["e"]["cond2_alt"])
    T1["e"]["pass_altf"] = bool(T1["e"]["cond1_altf"] and T1["e"]["cond2_altf"])
    if rel is None:
        T1["e"]["note"] = "M(local, 现行) = 0, 相对下降无定义, 视为未通过"
        T1["e"]["pass_alt"] = T1["e"]["pass_altf"] = False
    stats["T1"] = T1
    stats["meta"] = {"set_primary": "sampled (stride_sample(ptr, 2048), what the model sees)", "set_secondary": "all events",
                     "rules": RULE_NAME, "regimes": REG_NAME, "T1_primary_rule_for_E2": "alt",
                     "sequences": SEQS}
    outd.mkdir(parents=True, exist_ok=True)
    (outd / "assoc_stats.json").write_text(json.dumps(stats, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    (outd / "REPORT.md").write_text(render(stats, ind))
    print("wrote", outd / "assoc_stats.json", outd / "REPORT.md")


# ---------------------------------------------------------------- report
def render(st, ind):
    S0, SA = st["sets"]["sampled"], st["sets"].get("all")
    T = st["T1"]
    L = []
    P = L.append
    P("# [XYZ-ASSOC] P1 零训练：事件来源标签与关联对照（结果报告）\n")
    P("预注册：`docs/EXPERIMENT_RECORDS.md` XYZ-ASSOC-20261009 第 2 节。脚本：`tools/xa/step1_source_labels.py`、`step1_assoc_regimes.py`、`step1_report.py`（公共函数 `step1_common.py`）。纯 CPU，未使用 GPU，未改动仓库任何已有文件。\n")
    P("**口径提示**：数据为 v2e 仿真事件 + 伪真值；\"来源\"完全由该事件所在毫秒的真值网格定义（见 §0），\"非手\"(src_kind = −1) 不等于噪声，也可能是伪标签误差或背景事件。本报告只给 L1 层事实与 T1 判据，不对\"关联是不是瓶颈\"下结论。\n")
    P("## 0. 定义摘要（与预注册第 2 节逐条对应；偏离与口径选择见 §9）\n")
    for t in DEFS:
        P(f"- {t}")
    P("")
    # ---------------- T1 verdicts
    P("## 1. T1 判据（主口径：抽样事件 ≤2048/窗，现行规则，R0 窗首真值）\n")
    ta, tb = T["a"], T["b"]
    bl = st["bootstrap"]
    ci = lambda s, key: (lambda c: f"[{pc(c[0])}, {pc(c[1])}]")(bl[s][key]["M_ci95_step_cluster"]) if s in bl else "—"
    P(table([
        ["T1-a", "M(local) ≥ 20% 且 M(global) ≤ 5%", f"M(local)={pc(ta['M_local'])} (n={ta['n_local']}, 95%CI {ci('zgz_local', 'R0_cur')}); M(global)={pc(ta['M_global'])} (n={ta['n_global']}, 95%CI {ci('zgz_global', 'R0_cur')})", "**通过**" if ta["pass"] else "**未通过**"],
        ["T1-b", "M(local) < 5% 且 M(global) < 5%", "同上", "**条件满足**" if tb["pass"] else "**条件不满足**"],
    ], ["判据", "条件", "数字(序列/R0/现行/抽样集)", "判定"]))
    P(f"\n合并判读：**{T['ab_summary']}**。\n")
    P("### T1-c：规则 vs 滞后 vs 噪声（M 的分解，现行规则 / 替代规则，抽样集）\n")
    rows = []
    for ru in ("cur", "alt"):
        for s in SEQS:
            m = T["c"][ru][s]
            bb = st["bootstrap"].get(s, {}).get(f"T1c_{ru}")
            lagci = f" {[round(x * 100, 1) for x in bb['lag_ci95']]}" if bb else ""
            nci = f" {[round(x * 100, 1) for x in bb['noise_s_ci95']]}" if bb else ""
            rows.append([RULE_NAME[ru], sn(s), pc(m["R3"]), f"{pc(m['lag_R0_minus_R3'])}{lagci}", pc(m["R0"]), f"{pc(m['noise_R1s_minus_R0'])}{nci}", pc(m["R1s"]), pc(m["noise_R1l_minus_R1s"]), pc(m["R1l"])])
    P(table(rows, ["规则", "序列", "M(R3) 规则本身", "M(R0)−M(R3) 50ms 滞后(pp, 括号为 95%CI/步聚类自助)", "M(R0)", "M(R1s)−M(R0) 小噪声(pp, CI)", "M(R1s)", "M(R1l)−M(R1s)", "M(R1l)"]))
    P("\n（差值以百分点计；n 见 §3 表。CI 为按窗(step)聚类的自助法，2000 次，仅反映窗间变异，未计 run 间相关。）\n")
    P("### T1-d：错配事件的深度（现行规则，R0）\n")
    rows = []
    for s in SEQS + ["pooled"]:
        d = T["d"][s]
        if d.get("n", 0) == 0:
            rows.append([sn(s), 0, "—", "—", "—", "—", "—", "—"]); continue
        rows.append([sn(s), d["n"], mm(d["signed_median_mm"]), mm(d["signed_p90_mm"]), mm(d["abs_median_mm"]), mm(d["abs_p90_mm"]),
                     pc(d.get("frac_mismatch_via_hit_q1")),
                     ("通过" if d.get("pass_signed") else "未通过") + (" / 样本<100" if "enough_n" in d and not d["enough_n"] else "")])
    P(table(rows, ["序列", "错配事件数 n", "z 误差中位(带符号, mm)", "带符号 p90(mm)", "|z 误差|中位(mm)", "|z 误差|p90(mm)", "经命中(q=1)的比例", "T1-d(按带符号中位 ≥ 10 mm)"]))
    P("\nz 误差 = z_prior − src_depth；错配事件 = src_kind∈{1,2} 且 src_part≥1 且关联多数质量落在关节 0。直方图分辨率 0.25 mm（±300 mm 之外截入端点箱）。判读约定（看数字前写定）：按带符号中位 ≥ 10 mm 判定，|误差| 中位同列给出作对照；样本 <100 记\"样本不足\"。\n")
    P("### T1-e：替代规则（R0，抽样集）\n")
    e = T["e"]
    bt = st["bootstrap"]
    def bci(seq, key, sub):
        if seq in bt and key in bt[seq]:
            c = bt[seq][key][sub]
            return f" (95%CI [{c[0]:.1f}, {c[1]:.1f}])" if "pp" in sub else f" (95%CI [{c[0] * 100:.1f}%, {c[1] * 100:.1f}%])"
        return ""
    P(table([
        ["替代(contour)", f"M(local) {pc(e['M_local_cur'])} → {pc(e['M_local_alt'])}；相对下降 {pc(e['M_local_rel_drop_alt'])}{bci('zgz_local', 'R0_alt_vs_cur', 'M_rel_drop_ci95')}", f"global 精确一致率 {pc(e['exact_global']['cur'], 2)} → {pc(e['exact_global']['alt'], 2)}；变化 {e['exact_global_diff_pp_alt']:+.2f} pp{bci('zgz_global', 'R0_alt_vs_cur', 'exact_diff_pp_ci95')}", "**通过**" if e["pass_alt"] else "**未通过**"],
        ["altf 变体", f"M(local) {pc(e['M_local_cur'])} → {pc(e['M_local_altf'])}；相对下降 {pc(e['M_local_rel_drop_altf'])}", f"global 精确一致率 → {pc(e['exact_global']['altf'], 2)}；变化 {e['exact_global_diff_pp_altf']:+.2f} pp", "通过" if e["pass_altf"] else "未通过"],
    ], ["规则", "条件 1：M(local) 相对下降 ≥ 50%", "条件 2：global 精确部位一致率下降 ≤ 2 pp", "T1-e"]))
    P("\n主判读用 `alt`（命中/近层未触发 gap 的事件与现行逐位一致，符合任务说明）；`altf`（无命中事件的近层也做背面滤波）是预注册第 3 条措辞的另一种读法。**注意：仓库里随后上线的 opt-in 实现 `associate_rays(mode=\"contour\")` 与 `altf` 逐位一致（见 §2），不是 `alt`**；两种读法的 T1-e 结论相同（上表）。\n")
    # ---------------- self checks
    P("## 2. 自检\n")
    rows = []
    for s in SEQS:
        ck = st["checks"][s]["checks"]
        for sett in ("sampled", "all"):
            if f"{sett}/R0/n_events" not in ck:
                continue
            for rg in REGIMES:
                g = lambda k: ck.get(f"{sett}/{rg}/{k}", 0)
                lib = sum(g(f"lib_vs_cur_mismatch_{k}") for k in ("vid", "A", "q", "z", "valid", "hit"))
                rows.append([sn(s), sett, rg, g("n_events"), lib, g("alt_vs_cur_mismatch_on_noctr"), g("alt_n_ctr"), pc(g("alt_n_ctr") / max(g("n_events"), 1)), g("altf_ctr_differs_from_alt"), g("altf_vs_alt_differ"), g("alt_ctr_mass_err_gt1e-5")])
    P(table(rows, ["序列", "事件集", "regime", "事件数", "associate_rays vs 复现(cur)不一致事件数(vid/A/q/z/valid/hit 合计)", "替代规则在 ctr=0 事件上与现行不一致数", "ctr=1(双假设)事件数", "ctr 占比", "altf 与 alt 的 ctr 差异数", "altf 与 alt 记录不同的事件数", "ctr 事件质量和≠1(>1e-5)"]))
    P("\n- 复现一致性：每次调用均把 `associate_rays` 的输出与脚本内复现的现行逻辑逐位比较（`torch !=`），第 5 列应全为 0。\n- 渲染器核对（`labels_render_verify.json`）：对随机像素用 `ray_triangle`（关联所用的 Möller–Trumbore）暴力遍历全部 1538 面，与光栅化深度比较。\n")
    rv = ind / "labels_render_verify.json"
    if rv.exists():
        j = json.loads(rv.read_text())
        rows = []
        for s, d in j.items():
            for r in d["render_verify"]:
                rows.append([sn(s), r["frame"], r["n_pix"], r["both_hit"], r["raster_only"], r["brute_only"], f"{r['max_abs_dz_mm']:.4f}", f"{r['front_face_n_dot_d_negative']:.3f}"])
        P(table(rows, ["序列", "帧(ms)", "随机像素数", "两者都命中", "仅光栅命中", "仅暴力命中", "最大|Δz|(mm)", "命中面法向·射线<0 的比例(=1 表示面绕向朝外)"]))
    for s in SEQS:
        ck = st["checks"][s]["checks"]
        for sett in ("sampled", "all"):
            if f"{sett}/R3/interior_n" in ck:
                P(f"\n- 端到端几何核对（{sn(s)}，{sett}，R3，现行）：src_kind=0 内部事件 {ck[f'{sett}/R3/interior_n']} 个，其中射线命中 {ck[f'{sett}/R3/interior_hit']} 个 ({pc(ck[f'{sett}/R3/interior_hit'] / max(ck[f'{sett}/R3/interior_n'], 1), 2)})，命中深度与标签深度相差 <0.1 mm 的 {ck[f'{sett}/R3/interior_hit_and_z_within_0p1mm']} 个 ({pc(ck[f'{sett}/R3/interior_hit_and_z_within_0p1mm'] / max(ck[f'{sett}/R3/interior_n'], 1), 2)})。")
    cc2 = ind / "check_vs_contour_impl.json"
    if cc2.exists():
        j = json.loads(cc2.read_text())
        for sq, d in j.items():
            n_ = {rg: d[f"{rg}/n"] for rg in REGIMES}
            P(f"- 与仓库 opt-in 实现 `associate_rays(mode=\"contour\")` 的对照（{sn(sq)}，均匀抽 16 个窗的抽样事件，R3/R0/R1s/R1l 各 {n_['R0']} 个事件）：与 `altf` 不一致的事件数 " + "/".join(str(d[f"{rg}/altf/events_differing_from_production"]) for rg in REGIMES) + "（全部为 0 即逐位一致）；与 `alt` 不一致的事件数 " + "/".join(str(d[f"{rg}/alt/events_differing_from_production"]) for rg in REGIMES) + "；默认模式 `hit_first` 与 P1 的现行复现不一致数 " + "/".join(str(d[f"{rg}/default_mode_vs_cur_mismatch"]) for rg in REGIMES) + "。")
    cr = ind / "check_repo_contour.json"
    if cr.exists():
        j = json.loads(cr.read_text())
        for sq in SEQS:
            d = j[sq]; p1 = d["phase1"]; p2 = d["phase2_full_sampled_set_R0"]
            P(f"- 仓库 contour 实现交叉核对（`check_repo_contour.json`，{sn(sq)}，R0，{p1['n_windows']} 个窗（其中 fingertip_in_front 为真 {p1['n_windows_fip_true']}）、{p1['n_events']} 个抽样事件；容差 1e-5）：与 `altf` 在 ctr/vid/权重/z/r/valid/hit 上不一致事件 {p1['field_mismatch_events_repo_vs']['altf']['any']}；与 `alt` 不一致 {p1['field_mismatch_events_repo_vs']['alt']['any']}（全部为无命中事件，其近层在背面滤波前后不同）。全抽样集上用仓库实现重算：M={pc(p2['repo']['M'], 3)}、精确一致率={pc(p2['repo']['exact'], 3)}，与本报告 altf 的 {pc(p2['report_altf']['M'], 3)} / {pc(p2['report_altf']['exact'], 3)} 一致（alt：{pc(p2['report_alt']['M'], 3)} / {pc(p2['report_alt']['exact'], 3)}）。")
    cm = ind / "check_vs_model.json"
    if cm.exists():
        j = json.loads(cm.read_text())
        for s, d in j.items():
            P(f"- 与生产路径 `model._xyz_mesh_forward` 的逐步对照（{sn(s)}，R0，随机初始化的 C1 模型 CPU 前向 + `xyz_keep_aux`，均匀抽 {d['steps_checked']} 个窗、{d['events_checked']} 个抽样事件）：抽样集、射线 d、valid/hit/vid 全部逐位相同；FK 顶点最大偏差 {d['max_abs_diff']['verts']:.1e} m、z 最大偏差 {d['max_abs_diff']['z']:.1e} m、q 最大偏差 {d['max_abs_diff']['q']:.1e}、重心权 A 最大偏差 {d['max_abs_diff']['A']:.1e}（批量 FK 的 float32 舍入）；模型自己的 route_stats(valid/hit/resid) 与本脚本一致；失败项 {d['failed_checks'] or '无'}。")
    ct = ind / "check_timing.json"
    if ct.exists():
        j = json.loads(ct.read_text())
        for s, d in j.items():
            nh = {k: v["nonhand"] for k, v in d.items()}
            P(f"- 时间偏移敏感性（附带检查，{sn(s)}，随机 80 个窗的全部事件，用 ms+δ 的真值标注）：非手比例 δ=−4..+4 ms 为 " + ", ".join(f"{int(k):+d}:{100 * v:.2f}%" for k, v in nh.items()) + "（变化 ≤0.4 pp，标签对几毫秒的时间偏移不敏感）。")
    for s in SEQS:
        if s in st["bootstrap"]:
            b = st["bootstrap"][s]
            P(f"- 表统计量 vs 逐事件数组独立重算（{sn(s)}，R0 现行）：M 差 {b['crosscheck_M_R0_cur_table_minus_events']:.2e}，精确一致率差 {b['crosscheck_exact_R0_cur_table_minus_events']:.2e}（应为 0）。")
    # ---------------- labels
    P("\n## 3. 来源标签统计（全部事件 / 抽样事件）\n")
    rows = []
    for s in SEQS:
        l = st["labels"][s]
        rows.append([sn(s), l["n_steps"], l["n_runs"], l["n_events"], l["n_sampled_events"], pc(l["kind_frac"]["0"]), pc(l["kind_frac"]["1"]), pc(l["kind_frac"]["2"]), pc(l["kind_frac"]["-1"]),
                     pc(l["finger_contour_frac"]), pc(l["kind_frac_sampled"]["-1"]), pc(l["fingertip_in_front_steps"]), pc(l["fingertip_in_front_local_steps"]), pc(l["any_tip_in_palm_hull_steps"])])
    P(table(rows, ["序列", "评估步数", "run 数", "事件总数", "抽样事件数", "内部(0)", "剪影(1)", "深度不连续(2)", "非手(−1)", "手指轮廓前景(M 分母)", "抽样集非手", "fingertip_in_front 步占比(主定义)", "…(局部变体)", "有指尖落入掌凸包的步占比"]))
    P("\n像素统计与自检打印见 `labels.log`；`fingertip_in_front` 主定义 = 任务说明（指尖投影落入掌区凸包且深度小于全部掌顶点的最小深度）；局部变体 = 深度小于该指尖图像最近 5 个掌顶点的最小深度。**zgz_global 没有任何评估步有指尖落入掌凸包，因此该分层在 global 上为空。**\n")
    lg = ind / "labels.log"
    if lg.exists():
        txt = lg.read_text()
        P("```\n" + "\n".join(l for l in txt.splitlines() if re.search(r"====|pixels|events src_kind|finger-contour|fingertip", l)) + "\n```\n")
    # ---------------- main tables
    def main_table(S, regs, title):
        P(f"### {title}\n")
        rows = []
        for s in SEQS:
            for rg in regs:
                for ru in RULES:
                    d = S[s][rg][ru]
                    h, M = d["hand"], d["M"]
                    rows.append([sn(s), rg, ru, d["all_events"]["n"], h["n"], pc(d["all_events"]["valid_rate"]), pc(d["all_events"]["hit_rate"]), mm(d["all_events"]["resid_mean"], 3),
                                 pc(h["exact"]), pc(h["group"]), pc(h["exact_given_valid"]), pc(M["maj0_rate"]), M["n"], pc(M["maj0_given_valid"]), pc(M["plur0_rate"]), pc(d["all_events"]["ctr_rate"])])
        P(table(rows, ["序列", "regime", "规则", "事件数", "手事件数(分母)", "valid", "hit", "残差质量均值", "精确部位一致率", "手指组一致率", "精确一致率|valid", "M", "M 分母(手指轮廓前景)", "M|valid", "M(argmax 关节=0 口径)", "ctr(双假设)占比"]))
        P("")
    P("## 4. 主表（抽样集 ≤2048/窗；一致率分母 = src_part≥0 的事件，未关联事件计为不一致）\n")
    main_table(S0, REGIMES, "4.1 序列 × regime × 规则")
    # --- stratified
    def strat(S, rg, title):
        P(f"### {title}\n")
        rows = []
        keys = [("fip0_hand", "手事件, fingertip_in_front=0"), ("fip1_hand", "手事件, fingertip_in_front=1"), ("kind0_hand", "src_kind=0 内部"), ("kind1_hand", "src_kind=1 剪影"), ("kind2_hand", "src_kind=2 深度不连续"),
                ("fip0_M", "M: 手指轮廓前景, fip=0"), ("fip1_M", "M: 手指轮廓前景, fip=1"),
                ("nonhand", "非手事件 (src=−1)")]
        keys += [(f"group{g}", f"来源部位组: {GRP_NAMES[g]}") for g in range(1, 7)]
        keys += [(f"group{g}_contour", f"  其中轮廓(kind 1/2): {GRP_NAMES[g]}") for g in range(1, 7)]
        for s in SEQS:
            for key, nm in keys:
                for ru in ("cur", "alt"):
                    m = S[s][rg][ru][key]
                    if m["n"] == 0:
                        continue
                    rows.append([sn(s), nm, ru, m["n"], pc(m.get("valid_rate")), pc(m.get("hit_rate")), mm(m.get("resid_mean"), 3), pc(m.get("exact")), pc(m.get("group")), pc(m.get("maj0_rate")), pc(m.get("ctr_rate"))])
        P(table(rows, ["序列", "分层", "规则", "n", "valid", "hit", "残差质量均值", "精确一致率", "手指组一致率", "关节0占多数质量比例 maj0(在 M 分层行即 M)", "ctr 占比"]))
        P("")
    P("## 5. 分层（抽样集）\n")
    strat(S0, "R0", "5.1 R0")
    strat(S0, "R3", "5.2 R3（仅规则本身）")
    P("### 5.3 轮廓前景为手指的事件被关联到哪里（R0，现行 vs 替代；占该来源组事件的比例）\n")
    rows = []
    for s in SEQS:
        for ru in ("cur", "alt"):
            for g, d in S0[s]["R0"][ru]["confusion_contour_finger_by_srcgroup_to_pred"].items():
                rows.append([sn(s), ru, g, d["n"], pc(d["pred_none"]), pc(d["pred_joint0"]), pc(d["pred_same_group"]), pc(d["pred_other_finger"])])
    P(table(rows, ["序列", "规则", "来源部位组(kind 1/2)", "n", "未关联", "argmax 关节=0", "argmax 在同一手指组", "argmax 在其它手指"]))
    P("\n（此表为 argmax 关节口径，M 为\"关节 0 质量 > 50%\"口径，二者可略有差异。）\n")
    # --- alt side effects
    P("### 5.4 替代规则相对现行的变化（alt − cur，抽样集；一致率为 src_part≥0 事件上的精确/手指组一致率，单位 pp）\n")
    for rg in ("R0", "R3"):
        rows = []
        for sq in SEQS:
            for key, nm in (("hand", "全部手事件"), ("kind0_hand", "内部(kind 0)"), ("kind1_hand", "剪影(kind 1)"), ("kind2_hand", "深度不连续(kind 2)"),
                            ("fip0_hand", "fingertip_in_front=0"), ("fip1_hand", "fingertip_in_front=1"), ("group1", "来源=掌(关节0)"),
                            ("fingers_all_kinds", "来源=手指(全部 kind)"), ("M", "M 分母集(手指轮廓前景)")):
                c, a_ = S0[sq][rg]["cur"][key], S0[sq][rg]["alt"][key]
                if c["n"] == 0:
                    continue
                rows.append([sn(sq), nm, c["n"], pc(c["exact"]), pc(a_["exact"]), f"{(a_['exact'] - c['exact']) * 100:+.2f}",
                             pc(c["group"]), pc(a_["group"]), f"{(a_['group'] - c['group']) * 100:+.2f}", pc(c["maj0_rate"]), pc(a_["maj0_rate"]), pc(a_["ctr_rate"])])
        P(f"**{rg}**\n")
        P(table(rows, ["序列", "分层", "n", "精确一致率 cur", "alt", "Δ(pp)", "手指组一致率 cur", "alt", "Δ(pp)", "maj0 cur", "maj0 alt", "ctr 占比(alt)"]))
        P("")
    P("### 5.5 现行规则的命中面是否就是可见面（内部事件，src_kind=0，现行规则，抽样集）\n")
    rows = []
    for sq in SEQS:
        for rg in ("R3", "R0", "R1s"):
            m = S0[sq][rg]["cur"]["kind0_hand"]
            zh = S0[sq][rg]["cur"]["z"]["hit_kind0"]
            rows.append([sn(sq), rg, m["n"], pc(m["hit_rate"]), zh["n"], mm(zh["signed_median_mm"]), mm(zh["abs_p90_mm"]), pc(zh["frac_abs_ge_10mm"]), pc(m["exact"])])
    P(table(rows, ["序列", "regime", "内部事件数", "hit 率", "hit 事件数", "z−src_depth 带符号中位(mm)", "|误差|p90(mm)", "|误差|≥10mm 比例", "精确部位一致率"]))
    P("\nR3 下内部事件的网格与标签渲染同源，命中深度本应等于可见面深度。")
    cc = ind / "check_candidates.json"
    if cc.exists():
        j = json.loads(cc.read_text())
        for sq, d in j.items():
            P(f"- 候选面核查（{sn(sq)}，R3，随机窗每 5 ms 取一帧，内部事件 {d['n_interior']}，其中有命中 {d['n_hit']}）：命中深度比渲染可见面深度深 >1 mm 的 {d['hit_behind']} 个（{pc(d['hit_behind'] / max(d['n_hit'], 1), 2)}），比可见面浅 >1 mm 的 {d['hit_in_front']} 个；对这些深度偏深的事件，渲染最前面的面落在 16 个角距最近顶点的候选面集合内的有 {d['front_face_in_cand']} 个。")
    P("")
    # --- z
    P("## 6. z 误差（z_prior − src_depth，mm；仅 src_part≥0 且 valid 的事件）\n")
    def ztab(S, rg_list, rules, title):
        P(f"### {title}\n")
        rows = []
        for s in SEQS:
            for rg in rg_list:
                for ru in rules:
                    z = S[s][rg][ru]["z"]
                    for key, nm in (("hit_all_hand", "hit"), ("near_all_hand", "near"), ("mismatch_all", "错配事件(M 分子)"), ("nonmismatch_finger_contour_all", "手指轮廓前景·未错配")):
                        d = z[key]
                        if d["n"] == 0:
                            continue
                        rows.append([sn(s), rg, ru, nm, d["n"], mm(d["signed_median_mm"]), mm(d["signed_p90_mm"]), mm(d["abs_median_mm"]), mm(d["abs_p90_mm"]), pc(d["frac_abs_ge_10mm"])])
        P(table(rows, ["序列", "regime", "规则", "类别", "n", "带符号中位", "带符号 p90", "|误差|中位", "|误差|p90", "|误差|≥10mm 比例"]))
        P("")
    ztab(S0, ["R3", "R0", "R1s"], ["cur", "alt"], "6.1 现行 vs 替代（替代规则的 z 取质量较大的那条记录）")
    P("### 6.2 替代规则双假设事件(ctr=1)的两条记录的 z 误差（R0）\n")
    rows = []
    for s in SEQS:
        z = S0[s]["R0"]["alt"]["z"]
        for key, nm in (("ctr_front_record_all", "前景记录 z_near"), ("ctr_hit_record_all", "命中记录 t_hit"), ("ctr_front_record_mismatch", "前景记录·现行错配事件"), ("ctr_hit_record_mismatch", "命中记录·现行错配事件")):
            d = z[key]
            if d["n"]:
                rows.append([sn(s), nm, d["n"], mm(d["signed_median_mm"]), mm(d["signed_p90_mm"]), mm(d["abs_median_mm"]), mm(d["abs_p90_mm"])])
    P(table(rows, ["序列", "记录", "n", "带符号中位", "带符号 p90", "|误差|中位", "|误差|p90"]))
    P("\n（\"现行错配事件\"在替代规则下的 ctr 子集里按替代规则自己的多数质量划分 mism 标志，故与 T1-d 的事件集合不完全相同，仅作参考。）\n")
    # --- coverage
    P("## 7. 覆盖：valid / hit / 残差质量按来源类别（手 / 非手，抽样集）\n")
    rows = []
    for s in SEQS:
        for rg in REGIMES:
            for ru in ("cur", "alt"):
                for key, nm in (("hand", "手(src≥0)"), ("nonhand", "非手(src=−1)")):
                    m = S0[s][rg][ru][key]
                    rows.append([sn(s), rg, ru, nm, m["n"], pc(m["valid_rate"]), pc(m["hit_rate"]), mm(m["resid_mean"], 3)])
    P(table(rows, ["序列", "regime", "规则", "来源类别", "n", "valid", "hit(非手行即非手事件 hit 率)", "残差质量均值"]))
    P("")
    if SA:
        P("## 8. 全事件口径（次要）\n")
        main_table(SA, REGIMES, "8.1 序列 × regime × 规则（全部事件）")
        strat(SA, "R0", "8.2 R0 分层（全部事件）")
        ztab(SA, ["R0"], ["cur", "alt"], "8.3 R0 的 z 误差（全部事件）")
        P("### 8.4 T1 数字（全部事件）\n")
        rows = []
        for s in SEQS:
            c, a_, af = (SA[s]["R0"][r]["M"]["maj0_rate"] for r in RULES)
            n = SA[s]["R0"]["cur"]["M"]["n"]
            rows.append([sn(s), n, pc(c), pc(a_), pc(af), pc(1 - a_ / c) if c else "—", pc(SA[s]["R0"]["cur"]["hand"]["exact"], 2), pc(SA[s]["R0"]["alt"]["hand"]["exact"], 2), pc(SA[s]["R0"]["altf"]["hand"]["exact"], 2)])
        P(table(rows, ["序列", "M 分母 n", "M 现行", "M 替代", "M altf", "M 相对下降(替代)", "精确一致率 现行", "…替代", "…altf"]))
        P("")
    # --- deviations
    P("## 9. 与预注册定义的偏离、口径选择与注意事项\n")
    for t in DEVIATIONS:
        P(f"- {t}")
    P("\n## 10. 脚本与产物\n")
    P("- 脚本（`tools/xa/`）：`step1_common.py`（公共）、`step1_source_labels.py`（来源标签）、`step1_assoc_regimes.py`（关联对照，累积表）、`step1_report.py`（指标/判据/本报告）；附带检查 `step1_check_vs_model.py`（对照生产路径）、`step1_check_vs_contour_impl.py`、`step1_check_repo_contour.py`（对照 opt-in contour 实现）、`step1_check_timing.py`（时间偏移敏感性）、`step1_check_candidates.py`（候选面核查）。")
    P("- 产物（`outputs/xa/step1/`）：`labels_<seq>.npz`（逐事件 src_part/src_kind/src_depth/step_idx/ms/ev_x/ev_y，步级 step_end_ms/run_id/fingertip_in_front(+局部变体)/step_ev_start/step_a0/samp_local/samp_start）、`assoc_tables_<seq>.npz`（充分统计量：分层计数、z 误差直方图、混淆表）、`assoc_events_<seq>.npz`（抽样事件逐事件关联摘要，12 组 regime×规则）、`assoc_checks_<seq>.json`、`assoc_stats.json`、`REPORT.md`、`labels.log`/`assoc.log`、`labels_render_verify.json`、`check_*.json`。")
    return "\n".join(L)


DEFS = [
    "**数据**：zgz_global（1 run，1386 步）与 zgz_local（41 个 run，1204 步）的全部评估步（`eval_track.track_sequence` 的 run/end 枚举，50 ms 窗，`_window_events` 顺序）；主口径事件 = `stride_sample(ptr, 2048)` 抽样（模型实际看到的），次要口径 = 全部事件。",
    "**来源标签**：事件所在毫秒 t 的 pos51[t] → MANO FK → 软件光栅化 240×180（内参 s·K，s=0.375）得深度图 Z 与部位图 P（面部位 = 三顶点硬 LBS 关节多数）。轮廓像素 = 手像素且 4 邻域有非手像素(剪影)，或 4 邻域内有手像素深度差 >10 mm(深度不连续，前景=更近一侧)；事件 2 px 内有轮廓像素 → src_kind 1/2，部位与深度取最近轮廓像素的前景侧；否则落在手像素 → kind 0；否则 −1(非手)。来源由真值网格定义，事件为 v2e 仿真、真值为伪标签。",
    "**regime**：R3 每毫秒用该毫秒真值网格；R0 窗首真值(pos51[end−50]，首步 pos51[a])；R1s/R1l = R0 + 高斯噪声 σ(0.005,0.05,0.05)/(0.05,0.3,0.3)（t/rot/pose 每维，`default_rng(0)`）。关联参数同训练：band 16 px、front 3 px、z_tol 0.01、k 16。",
    "**规则**：现行 = `semkine.xyz_mesh.associate_rays`（直接调用）；替代 = 预注册\"替代规则\"1–4（`alt`，另有 `altf` 敏感性变体）。",
    "**指标**：精确部位一致率 = 关联质量加权硬关节 argmax == src_part；手指组一致率 = 同一手指组；**M** = src_kind∈{1,2} 且 src_part≥1 的事件中关联多数质量(>50%)落在关节 0 的比例；z 误差 = z_prior − src_depth；valid/hit/残差质量按手/非手分列。关节编号已用 `assets/mano_right.npz` 的 kintree 核对：0 腕，1–3 食指，4–6 中指，7–9 小指，10–12 无名指，13–15 拇指；21 关节中指尖为 OpenPose 索引 4/8/12/16/20（拇/食/中/无名/小）。",
]

DEVIATIONS = [
    "**像素与射线约定**：事件整数像素 (x, y) 视为像素中心，与 `associate_rays` 的射线 ((x−cx)/fx, (y−cy)/fy, 1) 同一约定；光栅化在整数像素坐标上取样，深度为透视正确的精确射线深度（与 Möller–Trumbore 命中深度同值，见自检）。z-buffer 对正反面一视同仁（与 `associate_rays` 的\"正反面都算命中\"一致）。重心坐标容差 1e-5（同 `BARY_EPS`）。",
    "**图像边界**：手像素的 4 邻域出图像时，该邻居被忽略（不计为非手），手被画面边界截断处不算剪影。",
    "**轮廓像素的前景与 kind**：深度不连续轮廓中，若某邻居手像素比本像素近 >10 mm，则本像素为\"远侧\"，其前景取最近的那个更近邻居；否则前景为本像素。kind 先判剪影（有非手 4 邻居→1），否则 2；因此同时是剪影与远侧的像素 kind=1 但前景部位/深度取更近邻居。M 的口径对 kind 1/2 合并，kind 的取舍不影响 M。2 px 内取欧氏距离（≤2，13 个偏移）最近的轮廓像素，同距取前景深度更小者。",
    "**面部位**：三顶点硬 LBS 关节的多数；三顶点两两不同时取三个候选关节中三顶点 LBS 权重和最大者（极少数面）。",
    "**fingertip_in_front**：主定义按任务说明（指尖投影落入掌区凸包，且指尖深度 < 全部掌顶点最小深度）；预注册措辞仅为\"更靠近相机\"，故另存局部变体（深度 < 该指尖图像最近 5 个掌顶点的最小深度）。两者均在 `labels_<seq>.npz`。",
    "**一致率与 M 的分母**：精确/手指组一致率分母 = 全部 src_part≥0 的事件（关联为 none 的事件计不一致），另给\"|valid\"口径；M 分母 = 全部 src_kind∈{1,2} 且 src_part≥1 的事件，分子 = 关联质量 w=q·A 在关节 0 上 > 总关联质量 50%（\"多数质量\"）；另给 argmax=0 口径(`plur0_rate`)和 M|valid。\"关联硬关节\"= 顶点硬 LBS argmax；预测关节 = 质量加权硬关节 argmax（并列取编号小者）。",
    "**z 误差**：现行规则取事件的 z；替代规则取质量较大的那条记录的 z（ctr 事件另给两条记录各自的误差）。分位数由 0.25 mm 直方图得到（分辨率 0.25 mm，±300 mm 外截入端点箱）。T1-d 按预注册字面（带符号中位 ≥ 10 mm），|误差| 同列给出。",
    "**R1 噪声**：`np.random.default_rng(0)` 每序列一次性抽 (步数×51) 标准正态，按步序使用，σ 乘到 t/rot/pose 各维；R1s 与 R1l 共用同一组标准正态；噪声每步独立地加在该步窗首真值上，不沿链累积（预注册中 R1 为\"窗首真值加噪\"）。",
    "**替代规则 (alt)**：预注册第 3 条\"近层始终计算并加背面过滤\"、第 4 条\"否则近层非空 → 同现行\"。按任务说明\"gap 条件不触发的事件上必须与现行一致\"，`alt` 中：触发 gap 条件（命中存在 ∧ 滤波后近层非空 ∧ zfront ≤ t_hit − 0.01）的事件走双假设，其余事件（含无命中事件）原样用现行的（未滤波）近层；背面滤波只影响决策与前景记录。第 3 条的另一读法（无命中事件的近层也滤波）作为敏感性变体 `altf` 并行给出。顶点法向 = 面积加权面法向累加后归一；MANO 面绕向朝外（光栅化核对中命中面法向·射线<0 的比例 = 1）。参考角距、同集、前层、≤4、σ=band/2、q_near 的逻辑与现行相同，只是在滤波后的候选集上取（参考 = 滤波后角距最近者）。记录上限 R=8（命中 4 槽含 1 个零权占位 + 前景 4 槽）。",
    "**抽样集由全事件运行导出**：每个事件的关联只依赖其自身射线与网格，故 `all` 运行中抽样子集的结果与单独只对抽样事件运行逐位相同（本批用 `--mode both`）。",
    "**R3** 按毫秒分组、B=50（每毫秒一个真值网格）；抽样事件按其所在毫秒分组。",
    "**CI**：自助法按窗聚类（2000 次，rng 固定），不计 run 间相关；zgz_global 只有 1 条 run。",
    "未做：未改动任何仓库已有文件；`outputs/xa/step1` 之外只在 `tools/xa/` 新增了 step1_* 脚本。",
]


if __name__ == "__main__":
    main()
