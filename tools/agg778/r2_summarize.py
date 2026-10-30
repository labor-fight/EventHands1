#!/usr/bin/env python3
"""AGG778 R2 gates (docs/EXPERIMENT_RECORDS.md#agg778-r2-prereg-20261024), CPU only.

    python tools/agg778/r2_summarize.py pilot    the pre-freeze ceiling check (selection subjects only)
    python tools/agg778/r2_summarize.py probes   validity per (condition x hypothesis), contrasts, D triggers
                                                 -> outputs/agg778/r2/summary.json

Contrasts are paired per endpoint (arm a - arm b, mm E3 RA; positive = a worse), averaged over the probe seeds, with a
95 % block bootstrap (20-endpoint blocks within runs, 2000 resamples). "Material": |mean| >= 1.5 mm (local) / 1.0 mm
(global), interval excludes 0, every seed the same sign. "Negligible": interval inside +-0.75 / +-0.5 mm and every seed's
difference inside that band.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.agg778 import common as C  # noqa: E402

R2 = C.OUT / "r2"
MAT = {"local": 1.5, "global": 1.0}
NEG = {"local": 0.75, "global": 0.5}
XCL_E3 = {"A0": {"local": 15.3511, "global": 4.8169}, "A3": {"local": 8.7779, "global": 3.7395}}
DOMS = ("local", "global")
_ID = {}


def identity_ra(z):
    """Per-endpoint RA of the anchor itself (no update), by the evaluator's scoring, cached per anchor set."""
    key = (z["anchor"].tobytes()[:4096], len(z["anchor"]))
    if key in _ID:
        return _ID[key]
    import torch
    from config import load_config
    from tools.xcl.evaluate import score
    cfg = load_config(str(C.REPO / C.MODELS["A0"]["config"]))
    dev = torch.device("cpu")
    mano = C.mano32(cfg, dev)
    out = np.zeros(len(z["anchor"]), np.float32)
    for s in np.unique(z["seq"]):
        sel = z["seq"] == s
        name, d = next((n, dd) for n, dd in (C.subject_seqs(C.SELECT_SUBJ) + C.zgz_seqs()) if n == s)
        beta = torch.tensor(C.Seq(name, d).betas).view(1, -1)
        out[sel] = score(mano, z["anchor"][sel].astype(np.float32), z["target"][sel].astype(np.float32), beta,
                         dev)["mpjpe_ra_mm"]
    _ID[key] = out
    return out


def load(cond, regime, split):
    """All arms of one (condition, regime) merged over its process groups: {arm: (seeds, n)} mm, plus the meta."""
    arms, meta, own = {}, None, None
    for d in sorted(R2.glob(f"probe/{cond}_{regime}_*")):
        f = d / f"eval_{split}.npz"
        if not f.exists():
            continue
        z = dict(np.load(f, allow_pickle=False))
        names = json.loads((d / "result.json").read_text())["names"]
        m = {k: z[k] for k in ("seq", "si", "k", "rid", "blk")}
        if meta is None:
            meta, own = {**m, "anchor": z["anchor"], "target": z["target"]}, z["mpjpe_ra_mm"][-1]
        elif not all(np.array_equal(meta[k], m[k]) for k in m):
            raise RuntimeError(f"endpoint order differs across groups: {d}")
        group = d.name.split("_")[-1]
        for j, n in enumerate(names):
            a, s = n.split("/")
            key = a if group != "gb" else a + "@8000"
            arms.setdefault(key, {})[int(s)] = {"ra": z["mpjpe_ra_mm"][j], "pred": z["pred"][j]}
    if meta is None:
        raise FileNotFoundError(f"no probe outputs for {cond} {regime} {split}")
    out = {a: np.stack([v[s]["ra"] for s in sorted(v)]) for a, v in arms.items()}
    preds = {a: np.stack([v[s]["pred"] for s in sorted(v)]) for a, v in arms.items()}
    return out, preds, meta, own


def dom_mask(meta, dom):
    return np.array([("_" + dom) in s for s in meta["seq"]])


def blocks(meta, sel):
    return meta["si"][sel] * 10 ** 6 + meta["rid"][sel] * 10 ** 3 + meta["blk"][sel]


def contrast(arms, meta, a, b, dom):
    sel = dom_mask(meta, dom)
    da, db = arms[a][:, sel], arms[b][:, sel]
    k = min(len(da), len(db))
    per_seed = (da[:k] - db[:k]).mean(1) if k == len(da) == len(db) else np.array([da.mean() - db.mean()])
    diff = da.mean(0) - db.mean(0)
    ci = C.block_bootstrap({"d": diff}, blocks(meta, sel))["d"]
    return {"mean": ci["mean"], "lo": ci["lo"], "hi": ci["hi"], "per_seed": [float(x) for x in per_seed],
            "a": float(da.mean()), "b": float(db.mean()), "n": int(sel.sum())}


def classify(c, dom):
    seeds = np.array(c["per_seed"])
    if (abs(c["mean"]) >= MAT[dom] and (c["lo"] > 0 or c["hi"] < 0) and len(seeds) > 1
            and np.all(np.sign(seeds) == np.sign(c["mean"]))):
        return "material"
    if c["lo"] > -NEG[dom] and c["hi"] < NEG[dom] and np.all(np.abs(seeds) < NEG[dom]):
        return "negligible"
    return "undetermined"


def arm_mean(arms, meta, a, dom):
    return float(arms[a][:, dom_mask(meta, dom)].mean())


def bias(preds, meta, a, dom):
    """Norm of the endpoint-mean 51-d residual (local 45 / root 3 / translation 3), seed-averaged."""
    sel = dom_mask(meta, dom)
    r = (preds[a][:, sel] - meta["target"][sel][None]).mean(1).mean(0)
    return {"local45": float(np.linalg.norm(r[6:])), "root": float(np.linalg.norm(r[3:6])),
            "transl_mm": float(np.linalg.norm(r[:3]) * 1000)}


def pilot():
    out, ok = {}, True
    for cond in ("A0", "A3"):
        d = R2 / "pilot" / cond
        z = dict(np.load(d / "eval_sel.npz", allow_pickle=False))
        names = json.loads((d / "result.json").read_text())["names"]
        idr = identity_ra(z)
        res = {}
        for dom in DOMS:
            sel = np.array([("_" + dom) in s for s in z["seq"]])
            res[dom] = {"identity": float(idr[sel].mean()),
                        **{n: float(z["mpjpe_ra_mm"][j, sel].mean()) for j, n in enumerate(names)}}
            res[dom]["T6_below_identity"] = res[dom]["T6/0"] < res[dom]["identity"]
            ok &= res[dom]["T6_below_identity"]
        out[cond] = res
    out["pass"] = bool(ok)
    C.save_json(R2 / "pilot" / "summary.json", out, exclusive=False)
    return out


def probes():
    S = {"validity": {}, "contrasts": {}, "classes": {}, "arm_means": {}, "bias": {}, "triggers": {}}
    data = {}
    for cond, regime in (("A3", "e3"), ("A0", "e3"), ("A3", "noise")):
        for split in ("zgz", "sel"):
            arms, preds, meta, own = load(cond, regime, split)
            meta["identity"] = identity_ra(meta)
            arms["identity"] = meta["identity"][None]
            data[(cond, regime, split)] = (arms, preds, meta, own)
            S["arm_means"][f"{cond}/{regime}/{split}"] = {a: {dom: arm_mean(arms, meta, a, dom) for dom in DOMS}
                                                          for a in arms}
            S["bias"][f"{cond}/{regime}/{split}"] = {a: {dom: bias(preds, meta, a, dom) for dom in DOMS} for a in preds}
    # ---------------- validity per condition
    for cond, regime in (("A3", "e3"), ("A0", "e3"), ("A3", "noise")):
        az, _, mz, own = data[(cond, regime, "zgz")]
        asel, _, msel, _ = data[(cond, regime, "sel")]
        v = {}
        v["V2_calibration"] = all(abs(arm_mean(a, m, "T0s", dom) - arm_mean(a, m, "identity", dom)) <= 0.1
                                  for a, m in ((az, mz), (asel, msel)) for dom in DOMS)
        if regime == "e3":
            v["V1_harness"] = all(abs(float(own[dom_mask(mz, dom)].mean()) - XCL_E3[cond][dom]) <= 0.01 for dom in DOMS)
            v["V3_ceiling"] = all(arm_mean(az, mz, "T6", dom) <= arm_mean(az, mz, "identity", dom) for dom in DOMS)
            c = contrast(az, mz, "T6s12", "T6", "local")
            v["V4_dose"] = c["lo"] > 0
            dn = [contrast(az, mz, "DN", "T0s", dom) for dom in DOMS]
            v["V5_donor"] = all(x["mean"] >= -0.5 for x in dn)
            S["contrasts"][f"{cond}/{regime}/validity"] = {"dose_T6s12-T6_local": c, "DN-T0s": dn,
                                                         "DN_diag_within_1mm": all(abs(x["mean"]) <= 1.0 for x in dn)}
        S["validity"][f"{cond}/{regime}"] = {**v, "valid": all(v.values())}
    valid = {k: v["valid"] for k, v in S["validity"].items()}
    # ---------------- contrasts
    plan = {
        "H-F": [("A3", "e3", "M13", "T4"), ("A3", "e3", "M13", "T5"), ("A3", "e3", "Mcore", "T4")],
        "H-T": [("A3", "e3", "T2", "T5"), ("A3", "noise", "T2", "T5"), ("A3", "e3", "T2", "M13i"),
                ("A3", "e3", "M13i", "M13"), ("A3", "e3", "T2", "M13"), ("A3", "e3", "T2", "T2abs"),
                ("A3", "e3", "M13", "Mabs"), ("A3", "e3", "M13", "Mcore"), ("A3", "noise", "M13", "Mabs"),
                ("A3", "noise", "T2", "M13"), ("A0", "e3", "T2", "M13i"), ("A0", "e3", "M13i", "M13"),
                ("A0", "e3", "T2", "M13"), ("A0", "e3", "T2", "T2abs"), ("A0", "e3", "M13", "Mabs")],
        "H-T-sub": [("A3", "e3", "M13", "Mpol"), ("A3", "e3", "M13", "Mrate"), ("A3", "e3", "M13", "Mmass"),
                    ("A3", "e3", "M13R", "MabsR"), ("A3", "e3", "M13R", "M13"), ("A3", "e3", "M13@8000", "M13")],
        "H-B": [("A0", "e3", "M13", "M13fan"), ("A0", "e3", "Mabs", "Mabsfan"), ("A0", "e3", "M13", "M13fan4"),
                ("A0", "e3", "T6fan", "T6"), ("A0", "e3", "T6K", "T6"), ("A0", "e3", "M13", "K")],
    }
    for hyp, rows in plan.items():
        for cond, regime, a, b in rows:
            for split in ("zgz", "sel"):
                arms, _, meta, _ = data[(cond, regime, split)]
                if a not in arms or b not in arms:
                    continue
                for dom in DOMS:
                    c = contrast(arms, meta, a, b, dom)
                    key = f"{hyp}|{cond}/{regime}|{a}-{b}|{split}/{dom}"
                    S["contrasts"][key] = c
                    S["classes"][key] = classify(c, dom)

    def cls(hyp, cond, regime, a, b, split, dom):
        return S["classes"].get(f"{hyp}|{cond}/{regime}|{a}-{b}|{split}/{dom}")

    def con(hyp, cond, regime, a, b, split, dom):
        return S["contrasts"].get(f"{hyp}|{cond}/{regime}|{a}-{b}|{split}/{dom}")

    # ---------------- triggers (pre-registered)
    ht = {}
    for cond in ("A3", "A0"):
        sup = False
        if valid.get(f"{cond}/e3"):
            for a, b in (("T2", "T2abs"), ("M13", "Mabs")):
                z_ = con("H-T", cond, "e3", a, b, "zgz", "local")
                s_ = con("H-T", cond, "e3", a, b, "sel", "local")
                if z_ and cls("H-T", cond, "e3", a, b, "zgz", "local") == "material" and z_["mean"] > 0 \
                        and abs(s_["mean"]) < NEG["local"]:
                    sup = True
        ht[cond] = sup
    hb = False
    if valid.get("A0/e3"):
        for dom in DOMS:
            z_ = con("H-B", "A0", "e3", "M13", "M13fan", "zgz", dom)
            s_ = con("H-B", "A0", "e3", "M13", "M13fan", "sel", dom)
            if z_ and cls("H-B", "A0", "e3", "M13", "M13fan", "zgz", dom) == "material" and z_["mean"] > 0 \
                    and s_["mean"] > 0:
                hb = True
    arms_d = []
    if any(ht.values()):
        arms_d += ["agg778r2_noabs_a3_s3407", "agg778r2_noabs_a0_s3407"]
    if hb:
        arms_d.append("agg778r2_fanout_a0_s3407")
    S["triggers"] = {"H-T_supported": ht, "H-B_supported": hb, "d_arms": arms_d,
                     "d_allowed": {a: valid.get("A3/e3" if "_a3_" in a else "A0/e3", False) for a in arms_d}}
    C.save_json(R2 / "summary.json", S, exclusive=False)
    return S


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "pilot":
        print(json.dumps(pilot(), indent=1))
    elif cmd == "probes":
        r = probes()
        print(json.dumps({"validity": r["validity"], "triggers": r["triggers"]}, indent=1))
    else:
        raise SystemExit("usage: r2_summarize.py pilot | probes")
