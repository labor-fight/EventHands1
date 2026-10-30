#!/usr/bin/env python3
"""AURA gates and summaries (docs/EXPERIMENT_RECORDS.md#aura-prereg-20261024), CPU only.

    python tools/aura/summarize.py r2gate   AGG778 R2 A0 / E3 validity (V2, V3, V5) -> may Stage C be read
    python tools/aura/summarize.py b        Stage B operator metrics, P1 reproduction, flags, capture radii
    python tools/aura/summarize.py c        Stage C validity V1-V9, headroom, primary (Holm) and secondary contrasts,
                                            realisation fractions, D triggers -> outputs/aura/c/summary.json
    python tools/aura/summarize.py c2       background injection readouts
    python tools/aura/summarize.py final    D gates, the main table, the verdict

Stage C contrasts are paired per endpoint (arm a - arm b, E3 / noise single-step RA in mm; positive = a worse), over the
seed indices both arms have (same family: same initial values), with a 20-endpoint block bootstrap. The gates scale with
the headroom H = identity - T6@c1 of the cell (evaluation set x domain x regime): "substantial" = |mean| >=
max(0.25 H, 0.25 mm), 95 % interval excluding 0 (primary family: Holm over the 12 zgz tests, 10000 resamples), every
seed the same sign (>= 2 seeds); "negligible" = interval within +-0.10 H and every seed within that band; else
"undetermined". D triggers read the selection subjects only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.agg778 import common as C  # noqa: E402
from tools.aura import common as AC  # noqa: E402

B_DIR, C_DIR, D_DIR = AC.OUT / "b", AC.OUT / "c", AC.OUT / "d"
DOMS = ("local", "global")
XCL_E3 = {"local": 15.3511, "global": 4.8169}
P1_TARGET = {"M_local": 6.31, "agree_local": 80.56}
SUB_FRAC, NEG_FRAC, SUB_FLOOR = 0.25, 0.10, 0.25
PRIMARY = [("P1", "e3", "occD", "occ"), ("P2", "e3", "M@shufG", "M@c1"), ("P3", "e3", "M@p2dh", "M@c1"),
           ("P4", "noise", "MR@surf", "MR@c1"), ("P5", "noise", "MR@aura10", "MR@c1"), ("P6", "noise", "M@c1", "M@A1")]
SECONDARY = {
    "occupancy": [("e3", "occ_tot", "occ"), ("e3", "occ_surf", "occ"), ("e3", "occ", "M@c1"), ("noise", "occ", "M@c1"),
                  ("noise", "occD", "occ")],
    "binding": [("e3", "M@shufP", "M@c1"), ("noise", "M@shufP", "M@c1"), ("e3", "M@shufG", "M@shufP"),
                ("noise", "M@shufG", "M@shufP"), ("noise", "M@shufG", "M@c1"), ("e3", "M@shufG", "occ"),
                ("noise", "M@shufG", "occ")],
    "3d": [("e3", "M@p2dvis", "M@p2dh"), ("e3", "M@p2d", "M@p2dvis"), ("noise", "M@p2dh", "M@c1")],
    "aura": [("e3", "MR@surf", "MR@c1"), ("e3", "MR@aura10", "MR@c1"), ("e3", "M@surf", "M@c1"),
             ("e3", "MR@b8", "MR@c1"), ("noise", "MR@b8", "MR@c1"), ("e3", "MR@b32", "MR@c1"),
             ("noise", "MR@b32", "MR@c1"), ("e3", "MR@mM", "MR@c1"), ("noise", "MR@mM", "MR@c1"),
             ("e3", "MR@m10", "MR@mM"), ("noise", "MR@m10", "MR@mM"), ("e3", "MR@aura10", "MR@m10"),
             ("noise", "MR@aura10", "MR@m10"), ("e3", "MR@m20", "MR@c1"), ("noise", "MR@m20", "MR@c1"),
             ("e3", "MR@aura20", "MR@c1"), ("noise", "MR@aura20", "MR@c1")],
    "nodes": [("e3", "M@shell", "M@c1"), ("noise", "M@shell", "M@c1"), ("e3", "MR@split", "MR@c1")],
    "visibility": [("e3", "M@vis", "M@c1"), ("e3", "M@ctr", "M@c1")],
    "oracle": [("e3", "M@c1", "M@A1"), ("e3", "M@A1", "M@A2"), ("noise", "M@A1", "M@A2"), ("e3", "M@A2", "M@A3"),
               ("noise", "M@A2", "M@A3")],
    "granularity": [("e3", "j16", "M@c1"), ("noise", "j16", "M@c1")],
    "ceiling": [("e3", f"T6@{x}", "T6@c1") for x in ("surf", "b32", "mM", "aura10", "p2dvis", "A3")]
               + [("noise", f"T6@{x}", "T6@c1") for x in ("surf", "b32", "aura10", "A3")],
    "calibration": [("e3", "MR@none", "MR@c1"), ("noise", "MR@none", "MR@c1")],
}
#: D variant -> (probe arm, comparator, regimes where the probe exists)
D_MAP = {"surf": ("MR@surf", "MR@c1"), "b8": ("MR@b8", "MR@c1"), "b32": ("MR@b32", "MR@c1"),
         "mmatch": ("MR@mM", "MR@c1"), "m10": ("MR@m10", "MR@c1"), "aura10": ("MR@aura10", "MR@c1"),
         "p2dh": ("M@p2dh", "M@c1"), "shell": ("M@shell", "M@c1"), "vis": ("M@vis", "M@c1")}


def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        x = x.item()
    return None if isinstance(x, float) and not np.isfinite(x) else x


# ================================================================================================ R2 precondition
def r2gate():
    s = json.loads((C.OUT / "r2" / "summary.json").read_text())["validity"]
    v = s.get("A0/e3", {})
    ok = all(bool(v.get(k)) for k in ("V2_calibration", "V3_ceiling", "V5_donor")) if v else False
    out = {"r2_A0_e3_validity": v, "may_read_C": ok,
           "rule": "AURA Stage C is read only when AGG778 R2's A0 / E3 V2, V3 and V5 hold"}
    C.save_json(AC.OUT / "r2gate.json", out, exclusive=False)
    return out


# ================================================================================================ Stage B
def _auc(hpos, hneg):
    """P(score_pos > score_neg) + 0.5 P(tie) from equal-bin histograms (ties within a bin count 0.5)."""
    hp, hn = np.asarray(hpos, float), np.asarray(hneg, float)
    if hp.sum() == 0 or hn.sum() == 0:
        return float("nan")
    cn = np.cumsum(hn) - hn                                                          # negatives strictly below
    return float(((hp * cn).sum() + 0.5 * (hp * hn).sum()) / (hp.sum() * hn.sum()))


def _quantile(h, lo, hi, q):
    h = np.asarray(h, float)
    if h.sum() == 0:
        return float("nan")
    c = np.cumsum(h) / h.sum()
    i = int(np.searchsorted(c, q))
    nb = len(h) - (2 if len(h) % 2 == 0 else 1)
    return float(lo + (i + 0.5) * (hi - lo) / max(nb, 1))


def b_metrics(z, regime, rule, di):
    from tools.aura import b_operator as BO
    g = lambda k: float(z[f"{regime}|{rule}|cnt|{k}"][di])                             # noqa: E731
    r = lambda a, b: g(a) / g(b) if g(b) > 0 else float("nan")                       # noqa: E731
    qh = z[f"{regime}|{rule}|q_hist"][di]
    dh = z[f"{regime}|{rule}|dist_hist"][di]
    dirh = z[f"{regime}|{rule}|dir_hist"][di]
    out = {"hand_valid": r("hand_valid", "hand"), "hand_q": r("hand_q", "hand"), "non_valid": r("non_valid", "non"),
           "non_q": r("non_q", "non"), "auc_q": _auc(qh[0], qh[1]), "auc_distance": _auc(dh[1], dh[0]),
           "mesh_mass_share": r("mesh_mass", "live"), "non_mass_share_of_mesh": r("mesh_mass_non", "mesh_mass"),
           "agree_exact": r("agree_exact", "hand"), "agree_finger": r("agree_finger", "hand"),
           "M": r("fc_mism", "fc"), "crosstalk": r("near_xtalk", "near"), "O_mean": r("O_sum", "packets"),
           "own_coverage": r("own_cov", "own"), "dir_coverage": r("dir_cov", "dir_elig"),
           "matpt_mm_median": _quantile(z[f"{regime}|{rule}|matpt_hist"][di], 0, BO.MMAX, 0.5),
           "matpt_mm_p90": _quantile(z[f"{regime}|{rule}|matpt_hist"][di], 0, BO.MMAX, 0.9),
           "depth_mm_median": _quantile(z[f"{regime}|{rule}|depth_hist"][di], -BO.EMAX, BO.EMAX, 0.5),
           "bind_depth_mm_median": _quantile(z[f"{regime}|{rule}|bind_hist"][di], -BO.EMAX, BO.EMAX, 0.5),
           "n_hand": g("hand"), "n_non": g("non")}
    out["selectivity_beyond_distance"] = out["auc_q"] - out["auc_distance"]
    out["dir_auc"] = _auc(dirh[:, 1].sum(0), dirh[:, 0].sum(0))
    out["dir_auc_by_u"] = [_auc(dirh[k, 1], dirh[k, 0]) for k in range(dirh.shape[0])]
    dv = z[f"{regime}|{rule}|dist_valid"][di]
    out["valid_by_distance"] = {"hand": (dv[0, :, 1] / np.maximum(dv[0, :, 0], 1)).tolist(),
                                "non": (dv[1, :, 1] / np.maximum(dv[1, :, 0], 1)).tolist(), "bins_px": BO.DBINS}
    kv, kn = z[f"{regime}|{rule}|kind_valid"][di], z[f"{regime}|{rule}|kind_n"][di]
    out["valid_by_kind"] = {k: float(kv[i, 0] / max(kn[i], 1)) for i, k in
                            enumerate(("non_hand", "interior", "silhouette", "depth_disc"))}
    return out


def capture_radius(curve):
    """Linear interpolation of the magnitude at which a metric falls to half its E3 value (None if it never does)."""
    xs, ys = curve
    y0 = ys[0]
    for i in range(1, len(xs)):
        if ys[i] <= 0.5 * y0:
            x0, x1, a, b = xs[i - 1], xs[i], ys[i - 1], ys[i]
            return float(x0 + (0.5 * y0 - a) * (x1 - x0) / (b - a)) if b != a else float(x1)
    return None


def stage_b():
    from tools.aura import b_operator as BO
    out = {"sets": {}, "flags": {}, "capture": {}, "p1_check": None, "shuffle_locality": None}
    for s in ("zgz", "sel", "fit"):
        f = B_DIR / f"stats_{s}.npz"
        if not f.exists():
            continue
        z = dict(np.load(f))
        res = {}
        for regime in BO.REGIMES[s]:
            for rule in BO.RULES_B:
                if f"{regime}|{rule}|cnt|live" not in z:
                    continue
                for di, dom in enumerate(DOMS):
                    res[f"{regime}|{rule}|{dom}"] = b_metrics(z, regime, rule, di)
        out["sets"][s] = res
        if s == "zgz":
            ck = {k.split("|")[1]: z[k] for k in z if k.startswith("p1check")}
            M = 100 * ck["p1_mism"][0] / max(ck["p1_fc"][0], 1)
            A = 100 * ck["p1_agree"][0] / max(ck["p1_hand"][0], 1)
            out["p1_check"] = {"M_local_pct": M, "agree_local_pct": A, "target": P1_TARGET,
                               "pass": abs(M - P1_TARGET["M_local"]) <= 0.01 and
                               abs(A - P1_TARGET["agree_local"]) <= 0.01}
            loc = {}
            for nm in ("G", "P"):
                h = z[f"shuffle|{nm}|dist"]
                n = float(z[f"shuffle|{nm}|n"])
                loc[nm] = {"median_px": _quantile(np.r_[h, 0], 0, 128.0, 0.5),
                           "within_16px": float(h[:16].sum() / max(h.sum(), 1)),
                           "same_part": float(z[f"shuffle|{nm}|same_part"]) / max(n, 1),
                           "same_finger": float(z[f"shuffle|{nm}|same_finger"]) / max(n, 1),
                           "self": float(z[f"shuffle|{nm}|self"]) / max(n, 1)}
            out["shuffle_locality"] = loc
            fl = {}
            for dom in DOMS:
                e3 = res.get(f"E3|c1|{dom}", {})
                e2 = res.get(f"E2|c1|{dom}", {})
                fl[dom] = {"hand_events_lost_at_clean_prior": e3.get("hand_valid", 1) < 0.95,
                           "aura_overlap": e3.get("crosstalk", 0) > 0.20,
                           "closed_loop_collapse": e2.get("agree_exact", 1) < 0.6 * e3.get("agree_exact", 1),
                           "selectivity_only_distance": e3.get("selectivity_beyond_distance", 1) < 0.02}
            out["flags"] = fl
        if s in ("zgz", "sel"):
            cap = {}
            fams = {"tx": [5, 10, 20, 40], "ty": [5, 10, 20, 40], "tz": [5, 10, 20, 40], "rot": [5, 10, 20],
                    "fp": [5, 10, 20]}
            for rule in BO.RULES_B:
                for dom in DOMS:
                    e3 = res.get(f"E3|{rule}|{dom}")
                    if e3 is None:
                        continue
                    for fam, mags in fams.items():
                        for met in ("hand_q", "agree_exact", "dir_auc"):
                            ys = [e3[met]] + [res[f"{fam}{m}|{rule}|{dom}"][met] for m in mags]
                            if met == "dir_auc":
                                ys = [y - 0.5 for y in ys]
                            cap[f"{rule}|{dom}|{fam}|{met}"] = {"curve": ys, "R_half": capture_radius(([0] + mags, ys))}
            out["capture"][s] = cap
    out = clean(out)
    C.save_json(B_DIR / "summary.json", out, exclusive=False)
    return out


# ================================================================================================ Stage C
def load(regime, split):
    """All arms of one regime merged over its process groups ({arm: (seeds, n)} mm), the A/A arm as M@c1', the meta."""
    arms, meta, own = {}, None, None
    for d in sorted(C_DIR.glob(f"probe/{regime}_*")):
        f = d / f"eval_{split}.npz"
        if not f.exists():
            continue
        z = dict(np.load(f, allow_pickle=False))
        names = json.loads((d / "result.json").read_text())["names"]
        m = {k: z[k] for k in ("seq", "si", "k", "rid", "blk")}
        if meta is None:
            meta, own = {**m, "anchor": z["anchor"], "target": z["target"]}, z["mpjpe_ra_mm"][-1]
            if "noise_kind" in z:
                meta["noise_kind"] = z["noise_kind"]
        elif not all(np.array_equal(meta[k], m[k]) for k in m):
            raise RuntimeError(f"endpoint order differs across groups: {d}")
        aa = d.name.endswith("_aa")
        for j, n in enumerate(names):
            a, s = n.split("/")
            a = a + "'" if aa else a
            if int(s) in arms.get(a, {}):
                raise RuntimeError(f"arm {a}/{s} appears in two groups")
            arms.setdefault(a, {})[int(s)] = z["mpjpe_ra_mm"][j]
    if meta is None:
        raise FileNotFoundError(f"no probe outputs for {regime} {split}")
    return {a: np.stack([v[s] for s in sorted(v)]) for a, v in arms.items()}, meta, own


def dom_mask(meta, dom):
    return np.array([("_" + dom) in s for s in meta["seq"]])


def blocks(meta, sel):
    return meta["si"][sel] * 10 ** 6 + meta["rid"][sel] * 10 ** 3 + meta["blk"][sel]


def boot(values, blk, n=2000, seed=20261024):
    ub, inv = np.unique(blk, return_inverse=True)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ub), size=(n, len(ub)))
    v = np.asarray(values, np.float64)
    s = np.bincount(inv, weights=v, minlength=len(ub))
    c = np.bincount(inv, minlength=len(ub)).astype(np.float64)
    return s[draws].sum(1) / c[draws].sum(1)


def contrast(arms, meta, a, b, dom, n_boot=2000):
    sel = dom_mask(meta, dom)
    if not sel.any() or a not in arms or b not in arms:
        return None
    k = min(len(arms[a]), len(arms[b]))
    da, db = arms[a][:k][:, sel], arms[b][:k][:, sel]
    diff = (da - db).mean(0)
    bs = boot(diff, blocks(meta, sel), n_boot)
    p = float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))
    return {"mean": float(diff.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
            "p": p, "per_seed": [float(x) for x in (da - db).mean(1)], "seeds": k, "n": int(sel.sum()),
            "a": float(arms[a][:, sel].mean()), "b": float(arms[b][:, sel].mean())}


def classify(c, H, floor_aa=0.0, holm_ok=None):
    if c is None or H is None or not np.isfinite(H):
        return "absent"
    sub = max(SUB_FRAC * H, SUB_FLOOR, floor_aa)
    neg = NEG_FRAC * H
    seeds = np.array(c["per_seed"])
    sig = (c["lo"] > 0 or c["hi"] < 0) if holm_ok is None else holm_ok
    if abs(c["mean"]) >= sub and sig and c["seeds"] >= 2 and np.all(np.sign(seeds) == np.sign(c["mean"])):
        return "substantial"
    if c["lo"] > -neg and c["hi"] < neg and np.all(np.abs(seeds) < neg):
        return "negligible"
    return "undetermined"


def holm(ps, alpha=0.05):
    order = np.argsort(ps)
    m = len(ps)
    ok = np.zeros(m, bool)
    for r, i in enumerate(order):
        if ps[i] <= alpha / (m - r):
            ok[i] = True
        else:
            break
    return ok


def stage_c():
    from tools.agg778.r2_summarize import identity_ra
    gate = json.loads((AC.OUT / "r2gate.json").read_text())
    S = {"r2gate": gate, "validity": {}, "headroom": {}, "aa": {}, "primary": {}, "secondary": {}, "classes": {},
         "arm_means": {}, "realisation": {}, "readouts_descriptive": {}}
    if not gate.get("may_read_C"):
        S["note"] = "AGG778 R2 A0 / E3 validity failed: Stage C not read"
        C.save_json(C_DIR / "summary.json", S, exclusive=False)
        return S
    data = {}
    for regime in ("e3", "noise"):
        for split in ("zgz", "sel"):
            arms, meta, own = load(regime, split)
            meta["identity"] = identity_ra(meta)
            arms["identity"] = meta["identity"][None]
            data[(regime, split)] = (arms, meta, own)
            S["arm_means"][f"{regime}/{split}"] = {a: {d: float(arms[a][:, dom_mask(meta, d)].mean())
                                                       if dom_mask(meta, d).any() else float("nan") for d in DOMS}
                                                   for a in arms}
    # ---------------- headroom, validity, A/A
    floor_aa = {}
    for (regime, split), (arms, meta, own) in data.items():
        for d in DOMS:
            cell = f"{regime}/{split}/{d}"
            idn, t6 = S["arm_means"][f"{regime}/{split}"]["identity"][d], S["arm_means"][f"{regime}/{split}"]["T6@c1"][d]
            H = idn - t6
            S["headroom"][cell] = H
            aa = contrast(arms, meta, "M@c1'", "M@c1", d)
            neg = NEG_FRAC * H
            ok = aa is not None and aa["lo"] > -neg and aa["hi"] < neg and all(abs(x) < neg for x in aa["per_seed"])
            S["aa"][cell] = {"contrast": aa, "negligible": bool(ok)}
            floor_aa[cell] = 0.0 if ok else (2 * max(abs(aa["lo"]), abs(aa["hi"])) if aa else float("inf"))
            for a in arms:
                if a != "identity":
                    S["realisation"].setdefault(cell, {})[a] = (idn - S["arm_means"][f"{regime}/{split}"][a][d]) / H \
                        if H > 0 else float("nan")
    for regime in ("e3", "noise"):
        az, mz, own = data[(regime, "zgz")]
        asel, msel, _ = data[(regime, "sel")]
        mean = lambda a, m, arm, d: float(a[arm][:, dom_mask(m, d)].mean())                      # noqa: E731
        v = {"V2_calibration": all(abs(mean(a, m, "T0s", d) - mean(a, m, "identity", d)) <= 0.1
                                   for a, m in ((az, mz), (asel, msel)) for d in DOMS),
             "V3_ceiling": all(mean(az, mz, "T6@c1", d) <= mean(az, mz, "identity", d) for d in DOMS)}
        dose = contrast(az, mz, "T6s12@c1", "T6@c1", "local")
        v["V4_dose"] = dose is not None and dose["lo"] > 0
        if regime == "e3":
            v["V1_harness"] = all(abs(float(own[dom_mask(mz, d)].mean()) - XCL_E3[d]) <= 0.01 for d in DOMS)
            v["V5_donor"] = all((c := contrast(a, m, "occD", "T0s", d)) is not None and
                                c["mean"] >= -NEG_FRAC * S["headroom"][f"e3/{sp}/{d}"]
                                for (a, m, sp) in ((az, mz, "zgz"), (asel, msel, "sel")) for d in DOMS)
            v["V7_residual"] = all(mean(a, m, "MR@none", d) - mean(a, m, "identity", d) <=
                                   NEG_FRAC * S["headroom"][f"e3/{sp}/{d}"]
                                   for (a, m, sp) in ((az, mz, "zgz"), (asel, msel, "sel")) for d in DOMS)
        S["validity"][regime] = {**v, "valid": all(v[k] for k in v if k != "V7_residual"),
                                 "V9_aa_all_negligible": all(S["aa"][f"{regime}/{sp}/{d}"]["negligible"]
                                                             for sp in ("zgz", "sel") for d in DOMS)}
    valid = {r: S["validity"][r]["valid"] for r in ("e3", "noise")}
    mr_ok = S["validity"]["e3"].get("V7_residual", False)
    # ---------------- primary family (zgz, Holm over 6 x 2) and its selection-subject sign check
    tests = []
    for pid, regime, a, b in PRIMARY:
        arms, meta, _ = data[(regime, "zgz")]
        for d in DOMS:
            c = contrast(arms, meta, a, b, d, n_boot=10000)
            tests.append((pid, regime, a, b, d, c))
    ps = np.array([t[5]["p"] if t[5] else 1.0 for t in tests])
    hok = holm(ps)
    for (pid, regime, a, b, d, c), h in zip(tests, hok):
        cell = f"{regime}/zgz/{d}"
        mr = a.startswith("MR@") or b.startswith("MR@")
        cls = classify(c, S["headroom"][cell], floor_aa[cell], holm_ok=bool(h))
        sarms, smeta, _ = data[(regime, "sel")]
        cs = contrast(sarms, smeta, a, b, d)
        if cls == "substantial" and (cs is None or np.sign(cs["mean"]) != np.sign(c["mean"])):
            cls = "substantial_zgz_only"
        if not valid[regime] or (mr and not mr_ok):
            cls = "invalid_" + cls
        S["primary"][f"{pid}|{regime}|{a}-{b}|{d}"] = {"zgz": c, "sel": cs, "holm_reject": bool(h), "class": cls,
                                                        "threshold_mm": max(SUB_FRAC * S["headroom"][cell], SUB_FLOOR,
                                                                            floor_aa[cell])}
    # ---------------- secondary contrasts (zgz and sel, per-cell interval, no correction)
    for fam, rows in SECONDARY.items():
        for regime, a, b in rows:
            for split in ("zgz", "sel"):
                arms, meta, _ = data[(regime, split)]
                for d in DOMS:
                    c = contrast(arms, meta, a, b, d)
                    key = f"{fam}|{regime}|{a}-{b}|{split}/{d}"
                    cell = f"{regime}/{split}/{d}"
                    S["secondary"][key] = c
                    cls = classify(c, S["headroom"][cell], floor_aa[cell])
                    if (a.startswith("MR@") or b.startswith("MR@")) and not mr_ok:
                        cls = "residual_confounded_" + cls
                    S["classes"][key] = cls if valid[regime] else "invalid_" + cls
    # robustness: (M@c1 - M@A3)_noise - (M@c1 - M@A3)_e3, per endpoint (same endpoints, other anchors)
    rob = {}
    for split in ("zgz", "sel"):
        an, mn, _ = data[("noise", split)]
        ae, me, _ = data[("e3", split)]
        for d in DOMS:
            sel = dom_mask(mn, d)
            if not sel.any():
                continue
            k = min(len(an["M@c1"]), len(an["M@A3"]), len(ae["M@c1"]), len(ae["M@A3"]))
            dd = ((an["M@c1"][:k] - an["M@A3"][:k]) - (ae["M@c1"][:k] - ae["M@A3"][:k]))[:, sel].mean(0)
            bs = boot(dd, blocks(mn, sel))
            rob[f"{split}/{d}"] = {"mean": float(dd.mean()), "lo": float(np.percentile(bs, 2.5)),
                                   "hi": float(np.percentile(bs, 97.5))}
    S["robustness_c1_minus_A3_noise_minus_e3"] = rob
    # ---------------- descriptive readouts: E2 (noise-trained arms), E3s, noise strata
    desc = {}
    for split, regime in (("zgz_E2", "noise"), ("zgz_E3s", "e3")):
        try:
            arms, meta, _ = load(regime, split)
        except FileNotFoundError:
            continue
        meta["identity"] = identity_ra(meta)
        arms["identity"] = meta["identity"][None]
        desc[split] = {a: {d: float(arms[a][:, dom_mask(meta, d)].mean()) for d in DOMS} for a in arms}
    an, mn, _ = data[("noise", "zgz")]
    if "noise_kind" in mn:
        strata = {}
        for kind, nm in ((0, "small"), (1, "large"), (2, "directed")):
            for d in DOMS:
                sel = dom_mask(mn, d) & (mn["noise_kind"] == kind)
                strata[f"{nm}/{d}"] = {a: float(an[a][:, sel].mean()) for a in an} if sel.any() else {}
        desc["noise_strata_zgz"] = strata
    S["readouts_descriptive"] = desc
    S["d_triggers"] = d_triggers(S, data, valid, floor_aa)
    S = clean(S)
    C.save_json(C_DIR / "summary.json", S, exclusive=False)
    return S


def d_triggers(S, data, valid, floor_aa):
    """Pre-registered Stage-D arms from the selection subjects only: none and shuffle always; at most one deployable
    variant whose probe is substantially better than its comparator at the selection subjects in E3 or noise (local or
    global, its regime valid) and not substantially worse in the other regime, ranked by |effect| / threshold."""
    mr_ok = S["validity"]["e3"].get("V7_residual", False)
    cand, detail = [], {}
    for variant, (a, b) in D_MAP.items():
        hits, worse = [], False
        for regime in ("e3", "noise"):
            arms, meta, _ = data[(regime, "sel")]
            if a not in arms or b not in arms or not valid[regime] or (a.startswith("MR@") and not mr_ok):
                continue
            for d in DOMS:
                cell = f"{regime}/sel/{d}"
                c = contrast(arms, meta, a, b, d)
                cls = classify(c, S["headroom"][cell], floor_aa[cell])
                if cls == "substantial" and c["mean"] < 0:
                    hits.append((regime, d, c["mean"], max(SUB_FRAC * S["headroom"][cell], SUB_FLOOR)))
                if cls == "substantial" and c["mean"] > 0:
                    worse = True
        detail[variant] = {"hits": hits, "worse_somewhere": worse}
        if hits and not worse:
            cand.append((max(abs(m) / t for _, _, m, t in hits), variant))
    cand.sort(reverse=True)
    arms = ["aura_none_a0_s3407", "aura_shuffle_a0_s3407"] + [f"aura_{v}_a0_s3407" for _, v in cand[:1]]
    return {"arms": arms, "ranked_optional": [v for _, v in cand], "detail": detail, "valid": valid}


# ================================================================================================ C2
def stage_c2():
    out = {}
    for f in sorted((C_DIR / "background").glob("*/result.json")):
        out[f.parent.name] = json.loads(f.read_text())
    C.save_json(C_DIR / "background" / "summary.json", out, exclusive=False)
    return out


# ================================================================================================ D and final
def xcl_metrics(run, name):
    return json.loads((C.REPO / "outputs/xcl/20261019/evaluations" / run / name / "report.json").read_text())["metrics"]


def arm_metrics(arm, name):
    return json.loads((D_DIR / "evaluations" / arm / name / "report.json").read_text())["metrics"]


def d_gate(arm, control_run="xyz_c1_stable_s3407"):
    """XCL single-seed descriptive rule against C1 (arm - control, mm), either direction, with the H3=0 guard."""
    keys = {"e1": "e1_step6000_repeat1_none", "e2": "e2_step6000_repeat1_none", "e3": "e3_step6000_repeat1_none",
            "e5_5000": "e1_step5000_repeat1_none", "e5_5500": "e1_step5500_repeat1_none"}
    mine = {k: arm_metrics(arm, v) for k, v in keys.items()}
    ctrl = {k: xcl_metrics(control_run, v) for k, v in keys.items()}
    out = {"control": control_run}
    for metric, dom, thr in (("mpjpe_ra_mm", "local", 5.0), ("mpjpe_ra_mm", "global", 8.0), ("wrist_mm", "local", 10.0),
                             ("wrist_mm", "global", 10.0)):
        d = {k: mine[k][dom][metric] - ctrl[k][dom][metric] for k in keys}
        sgn = np.sign(d["e1"])
        late = [d["e5_5000"], d["e5_5500"], d["e1"]]
        support = (abs(d["e1"]) >= thr and np.sign(d["e2"]) == sgn and np.sign(d["e3"]) == sgn
                   and sum(np.sign(x) == sgn for x in late) >= 2)
        out[f"{metric}/{dom}"] = {"delta": d, "threshold": thr,
                                  "class": "direction_supported" if support else "noise_band",
                                  "direction": "better" if sgn < 0 else "worse"}
    h3z = xcl_metrics(control_run, "e3_step6000_repeat1_h3_zero")
    e3l = mine["e3"]["local"]["mpjpe_ra_mm"]
    out["h3_zero_guard"] = {"arm_e3_local": e3l, "control_h3_zero_e3_local": h3z["local"]["mpjpe_ra_mm"],
                            "only_less_event_reliance": bool(e3l < ctrl["e3"]["local"]["mpjpe_ra_mm"]
                                                             and e3l >= h3z["local"]["mpjpe_ra_mm"])}
    return out


def main_table(arms):
    from tools import report_table as original
    rows = {"dt3_dz_l3_w200cfg": C.REPO / "outputs/semkine/dt3_dz_l3_w200cfg_main_row.json",
            "xyz_c1_baseline": C.REPO / "outputs/semkine/xyz_c1_baseline_main_row.json"}
    for arm in arms:
        rows[arm.removesuffix("_s3407")] = C.REPO / "outputs/semkine" / f"{arm.removesuffix('_s3407')}_main_row.json"

    class Rows:
        def __truediv__(self, filename):
            return rows[str(filename).removesuffix("_main_row.json")]
    label = {"none": "去掉关联（全部事件进残差槽）", "shuffle": "随机绑定（包内置换，占据不变）", "surf": "只有表面命中",
             "b8": "外包围带 8 px", "b32": "外包围带 32 px", "mmatch": "度量外包围（等效 16 px）", "m10": "度量外包围 1 cm",
             "aura10": "Aura Mesh 式度量壳 1 cm", "p2dh": "2D 邻近 + 剪影位", "shell": "外包围壳节点", "vis": "近层遮挡判定"}
    old = original.OUT
    original.OUT = Rows()
    try:
        lines = [original.HEADER, original.RULE, original.BASELINE,
                 original.row("dt3_dz_l3_w200cfg", "S39：dt_dz_l3，W=200 + 自适应滤波 + 证据门 ramp1000 *"),
                 original.row("xyz_c1_baseline", "XYZ C1（seed3407）")]
        for arm in arms:
            v = arm.split("_")[1]
            lines.append(original.row(arm.removesuffix("_s3407"), f"AURA {label[v]}（seed3407）"))
    finally:
        original.OUT = old
    return "\n".join(lines)


def final():
    a = json.loads((AC.OUT / "a" / "result.json").read_text())
    b = json.loads((B_DIR / "summary.json").read_text())
    c = json.loads((C_DIR / "summary.json").read_text())
    sel = json.loads((D_DIR / "selection.json").read_text())
    d = {arm: d_gate(arm) for arm in sel["arms"]}
    v = {"A_pass": a["pass"], "A_gates": a["gates"], "B_p1_check": b.get("p1_check"), "B_flags": b.get("flags"),
         "C_validity": c.get("validity"), "C_primary": {k: x["class"] for k, x in c.get("primary", {}).items()},
         "C_classes_zgz": {k: x for k, x in c.get("classes", {}).items() if "|zgz/" in k},
         "C_classes_sel": {k: x for k, x in c.get("classes", {}).items() if "|sel/" in k}, "D": d}
    C.save_json(AC.OUT / "verdict.json", v, exclusive=False)
    table = main_table(sel["arms"])
    (AC.OUT / "main_table.md").write_text(table + "\n")
    return v, table


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "r2gate":
        print(json.dumps(r2gate(), indent=1))
    elif cmd == "b":
        r = stage_b()
        print(json.dumps({"p1_check": r["p1_check"], "flags": r["flags"]}, indent=1))
    elif cmd == "c":
        r = stage_c()
        print(json.dumps({k: r.get(k) for k in ("validity", "d_triggers")}, indent=1, default=str))
    elif cmd == "c2":
        stage_c2()
        print("ok")
    elif cmd == "final":
        v, table = final()
        print(table)
    else:
        raise SystemExit("usage: summarize.py r2gate | b | c | c2 | final")
