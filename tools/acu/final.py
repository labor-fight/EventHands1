#!/usr/bin/env python3
"""ACU stage-3 deliverables (docs/EXPERIMENT_RECORDS.md#acu-s3-prereg-20261104); CPU, read-only on readings and cost reports.

    python -m tools.acu.final row --name N --label L --read TAGDIR [--read TAGDIR ...] --cost COSTDIR
        outputs/semkine/acu_<N>_main_row.json in tools/report_table.py's format (N-seed means; per-seed RA)
    python -m tools.acu.final tables --rows N[,N...] --mech LABEL=TAGDIR[,...] --out analysis/final.md
        the nine-column table (tools/report_table.py) + mechanism table + condition breakdowns + drift

Mechanism (per reading): E3 RA; amplification E1 / E3; zgz_local retention R15 / F15 / T30 (rho medians, the engine's
definition); f1: RA after 1 and 20 feeds from GT and C = mean-start RA after 20 feeds / mean-start RA (< 1 = recovers from
the mean hand); the per-step decomposition of tools/acu/diag_update.py on zgz_local (E3 injection root / fingers,
low-motion change, own-loop correction a, response lag of fingers).
Breakdowns (E1, both zgz recordings pooled, mean RA per tertile): true per-step motion (hand-frame joint displacement of
the target over the step), events per packet, the loop's prior error; drift = E1 RA by time since the segment start
(0-1, 1-2, 2-4, 4-8, >= 8 s).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.acu.common import OUT, ZGZ, now, read, require, save  # noqa: E402

SEMKINE = ROOT / "outputs/semkine"
KEYS = ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm")
DRIFT = (0, 1000, 2000, 4000, 8000, 10 ** 9)


def _block(e1):
    s = e1["per_sequence"]
    return {"local": {k: s["zgz_local"][k] for k in KEYS}, "global": {k: s["zgz_global"][k] for k in KEYS},
            "overall": {k: e1["overall"][k] for k in KEYS}}


def row(name, label, reads, cost):
    per = {}
    for i, d in enumerate(reads):
        e1 = read(Path(d) / "e1" / "report.json")
        per[str(i)] = {"reading": str(d), "checkpoint": e1["identity"]["checkpoint"], **_block(e1)}
    mean = {b: {k: float(np.mean([per[s][b][k] for s in per])) for k in KEYS} for b in ("local", "global", "overall")}
    c = read(Path(cost) / "report.json")
    macs = c["macs_layers"]["mean"]
    r = {"arm": name, "label": label, "split": "val_core", "n_seeds": len(per), "two_seed_mean": mean, "per_seed": per,
         "latency_ms_raw": c["latency_row"]["raw_ms"], "latency_anchor_ms": c["latency_row"]["anchor_ms"],
         "latency_ms_scaled_full1p75": c["latency_row"]["scaled_ms"], "macs_forward_packet": float(macs),
         "params_total": int(c["params_total"]), "latency_e2e_ms": c["latency_e2e_ms"], "cost_report": str(cost),
         "note": "'two_seed_mean' is the N-seed mean (tools/report_table.py format); cost = the ACU cost block",
         "written_at_utc": now()}
    out = SEMKINE / f"acu_{name}_main_row.json"
    require(not out.exists(), "main row written once: " + str(out))
    save(out, r, exclusive=True)
    print("wrote", out)


def mech(tag_dir):
    from tools.acu import diag_update as DU
    d = Path(tag_dir)
    e1, e3, pt = (read(d / t / "report.json") for t in ("e1", "e3", "pert"))
    loc = pt["per_sequence"]["zgz_local"]
    out = {"e3": e3["overall"]["mpjpe_ra_mm"], "amp": e1["overall"]["mpjpe_ra_mm"] / e3["overall"]["mpjpe_ra_mm"],
           "R15": loc["R15"]["rho_median"], "F15": loc["F15"]["rho_median"], "T30": loc["T30"]["rho_median"]}
    f1p = d / "f1" / "report.json"
    if f1p.exists():
        f = read(f1p)["per_sequence"]
        out["f1_gt1"] = float(np.mean([f[s]["gt_iter1"] for s in ZGZ]))
        out["f1_gt20"] = float(np.mean([f[s]["gt_iter20"] for s in ZGZ]))
        out["C"] = float(np.mean([f[s]["mean_iter20"] / f[s]["mean_start"] for s in ZGZ]))
    z3 = np.load(d / "e3" / "states.npz")
    z1 = np.load(d / "e1" / "states.npz")
    b3 = DU.e3_block(z3, "zgz_local")
    b1 = DU.e1_block(z1, "zgz_local")
    out.update(inj_R=b3["R"]["rms_injection"], inj_F=b3["F"]["rms_injection"], lowmo_R=b3["R"]["low_motion"]["rms_D"],
               lowmo_F=b3["F"]["low_motion"]["rms_D"], G_F=b3["F"]["G"], a_R=b1["R"]["a"], a_F=b1["F"]["a"],
               lag_F=b1["F"]["lag_peak_shift"])
    return out


def breakdown(tag_dir):
    from tools.acu.diag_update import hand_joints
    from tools.acu.diag_support import OUT as DOUT
    d = Path(tag_dir)
    z = np.load(d / "e1" / "states.npz")
    cnt = np.load(DOUT / "d1b_counts.npz")
    rows = {"motion": [], "events": [], "prior": [], "drift": []}
    ra_all, mot_all, ev_all, pe_all, el_all = [], [], [], [], []
    for seq in ZGZ:
        ra = z[f"{seq}/mpjpe_ra_mm"]
        jt, jg, ji = (hand_joints(z[f"{seq}/{k}"], seq) for k in ("target", "gt_previous", "incoming"))
        mot = np.linalg.norm(jt - jg, axis=-1).mean(-1)
        pe = np.linalg.norm(ji - jg, axis=-1).mean(-1)
        ra_all.append(ra), mot_all.append(mot), pe_all.append(pe), el_all.append(z[f"{seq}/elapsed_ms"])
        ev_all.append(cnt[f"{seq}/n_events"][: len(ra)])
    ra, mot, ev, pe, el = (np.concatenate(v) for v in (ra_all, mot_all, ev_all, pe_all, el_all))
    for key, v in (("motion", mot), ("events", ev), ("prior", pe)):
        q = np.quantile(v, [1 / 3, 2 / 3])
        for i, sel in enumerate((v <= q[0], (v > q[0]) & (v <= q[1]), v > q[1])):
            rows[key].append({"tertile": i, "lo": float(v[sel].min()), "hi": float(v[sel].max()), "ra": float(ra[sel].mean()),
                              "n": int(sel.sum())})
    for lo, hi in zip(DRIFT[:-1], DRIFT[1:]):
        sel = (el >= lo) & (el < hi)
        if sel.any():
            rows["drift"].append({"lo_s": lo / 1000, "hi_s": hi / 1000 if hi < 10 ** 9 else None, "ra": float(ra[sel].mean()),
                                  "n": int(sel.sum())})
    return rows


def tables(rows, mechs, out):
    import subprocess
    cmd = [sys.executable, str(ROOT / "tools/report_table.py")] + [f"acu_{r}" if not r.startswith("@") else r[1:] for r in rows]
    t = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    L = ["# ACU final tables", "", t.strip(), "", "## Mechanism", "",
         "| network | E3 | amp | R15 | F15 | T30 | f1 GT 1 -> 20 | C | E3 inj R deg / F mm | low-motion R / F | G_F | loop a R / F | F lag |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    bd = {}
    for label, d in mechs:
        m = mech(d)
        L.append(f"| {label} | {m['e3']:.2f} | {m['amp']:.2f} | {m['R15']:.2f} | {m['F15']:.2f} | {m['T30']:.2f} | "
                 f"{m.get('f1_gt1', float('nan')):.2f} -> {m.get('f1_gt20', float('nan')):.2f} | {m.get('C', float('nan')):.2f} | "
                 f"{m['inj_R']:.2f} / {m['inj_F']:.2f} | {m['lowmo_R']:.2f} / {m['lowmo_F']:.2f} | {m['G_F']:.2f} | "
                 f"{m['a_R']:.2f} / {m['a_F']:.2f} | {m['lag_F']} |")
        bd[label] = breakdown(d)
    for key, title in (("motion", "true per-step motion (hand-frame joints, mm)"), ("events", "events per packet"),
                       ("prior", "the loop's prior error (hand-frame joints, mm)")):
        L += ["", f"## E1 RA by {title} (tertiles, both zgz recordings)", "", "| network | low | mid | high |", "|---|---|---|---|"]
        for label in bd:
            r = bd[label][key]
            L.append(f"| {label} | " + " | ".join(f"{x['ra']:.2f} ({x['lo']:.1f}-{x['hi']:.1f})" for x in r) + " |")
    L += ["", "## Drift: E1 RA by time since the segment start", "", "| network | " +
          " | ".join(f"{a}-{b if b else ''} s" for a, b in zip([0, 1, 2, 4, 8], [1, 2, 4, 8, None])) + " |",
          "|---|" + "---|" * 5]
    for label in bd:
        L.append(f"| {label} | " + " | ".join(f"{x['ra']:.2f}" for x in bd[label]["drift"]) + " |")
    p = OUT / out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    r = sub.add_parser("row")
    r.add_argument("--name", required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--read", action="append", required=True)
    r.add_argument("--cost", required=True)
    t = sub.add_parser("tables")
    t.add_argument("--rows", required=True)
    t.add_argument("--mech", required=True)
    t.add_argument("--out", default="analysis/final.md")
    a = p.parse_args(argv)
    if a.verb == "row":
        row(a.name, a.label, a.read, a.cost)
    else:
        tables(a.rows.split(","), [tuple(x.split("=", 1)) for x in a.mech.split(",")], a.out)


if __name__ == "__main__":
    main()
