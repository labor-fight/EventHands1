#!/usr/bin/env python3
"""RDR screen, the written rule (docs/EXPERIMENT_RECORDS.md#rdr-prereg-20261110, #rdr-amend1-20261110). CPU.

    python -m tools.rdr.decide --candidates v2,v3 [--reference r_s2d] --out outputs/rdr/20261110/decisions/screen_a.json

Coherent offset = node-weighted median norm of the per-(packet, label part) mean routing error over the routed label-hand
nodes (router_capture/<name>.npz of outputs/evhandtrack/analysis/posthoc_20261110). Against the reference on ylf_local,
zgz_local, zgz_global: BETTER = lower on all three and mean drop >= 10 %; WORSE = mean rise >= 5 %; else EQUIVALENT.
Selected = the BETTER candidate with the lowest three-recording mean (none if no candidate is BETTER).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CAP = ROOT / "outputs/evhandtrack/analysis/posthoc_20261110/router_capture"
SEQS = ("ylf_local", "zgz_local", "zgz_global")
BETTER, WORSE = 0.10, 0.05


def coherent(name):
    z = np.load(CAP / f"{name}.npz")
    seqs = [str(s) for s in z["seqs"]]
    out = {}
    for seq in SEQS + ("lyq_local",):
        if seq not in seqs:
            continue
        m = z["seq"] == seqs.index(seq)
        both = z["l_ok"][m] & z["r_sel"][m]
        pkt, lpart, rpart = z["pkt"][m][both], z["lpart"][m][both].astype(int), z["rpart"][m][both].astype(int)
        err = (z["rp"][m][both].astype(np.float64) - z["lp"][m][both]) * 1000
        n = int(z["pkt"][m].max()) + 1
        s = np.zeros((n, 16, 3)); c = np.zeros((n, 16))
        np.add.at(s, (pkt, lpart), err); np.add.at(c, (pkt, lpart), 1)
        v = c >= 3
        mag = np.linalg.norm(s[v] / c[v][:, None], axis=-1)
        o = np.argsort(mag); cw = np.cumsum(c[v][o])
        out[seq] = {"coherent_mm": float(mag[o][np.searchsorted(cw, cw[-1] / 2)]), "part_acc": float(np.mean(lpart == rpart)),
                    "recall": float(both.sum() / max(z["l_ok"][m].sum(), 1))}
    return out


def rule(ref, cand):
    r = np.array([ref[s]["coherent_mm"] for s in SEQS]); c = np.array([cand[s]["coherent_mm"] for s in SEQS])
    rel = float(c.mean() / r.mean() - 1.0)
    if bool((c < r).all()) and rel <= -BETTER:
        verdict = "BETTER"
    elif rel >= WORSE:
        verdict = "WORSE"
    else:
        verdict = "EQUIVALENT"
    return {"verdict": verdict, "mean_mm": float(c.mean()), "reference_mean_mm": float(r.mean()), "relative": rel}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", required=True)
    p.add_argument("--reference", default="r_s2d")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        sys.exit("written once: " + str(out))
    ref = coherent(a.reference)
    rows = {}
    for v in a.candidates.split(","):
        stats = coherent(f"rdr_{v}")
        rows[v] = {"stats": stats, **rule(ref, stats)}
    better = [v for v, r in rows.items() if r["verdict"] == "BETTER"]
    selected = min(better, key=lambda v: rows[v]["mean_mm"]) if better else None
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"reference": {"name": a.reference, "stats": ref}, "candidates": rows, "selected": selected,
                               "thresholds": {"better": BETTER, "worse": WORSE}}, indent=1) + "\n")
    for v, r in rows.items():
        print(v, r["verdict"], f"{r['relative'] * 100:+.1f} %", {s: round(r["stats"][s]["coherent_mm"], 2) for s in r["stats"]})
    print("reference", {s: round(ref[s]["coherent_mm"], 2) for s in ref})
    print("selected", selected)


if __name__ == "__main__":
    main()
