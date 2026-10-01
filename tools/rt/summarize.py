#!/usr/bin/env python3
"""Root-tracking round: one table per comparison, from the artifacts the post-processing chain writes.

Per run (`outputs/semkine/<run>`): the selection grid (`selection_val_core_step50*.json`) and evalx on
the last and the selected checkpoint (`evalx_val_core_last_tf_pert.json`, `evalx_val_core_selected.json`).
Per arm, seeds are paired with the base arm's run of the same seed.

    python tools/rt/summarize.py --base rt_s37_2k --arms rt_g3_2k rt_e8_2k rt_c37_2k [--seeds 3407 3408]
                                 [--ckpt last|selected] [--out outputs/rt/reports/screen1]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SEM = REPO / "outputs" / "semkine"


def load_run(run: str, ckpt: str):
    d = SEM / run
    sel = sorted(d.glob("selection_val_core_step50*.json"))
    if not sel:
        return None
    sel = json.loads(sel[0].read_text())
    grid = {int(r["step"]): r["mpjpe_ra_mm"] for r in sel["grid"]}
    out = {"run": run, "sel_step": sel["selected"]["step"], "sel_ra": sel["selected"]["mpjpe_ra_mm"],
           "last_step": max(grid), "last_ra_grid": grid[max(grid)],
           "grid_median": float(np.median(list(grid.values()))), "grid": grid}
    ev = d / ("evalx_val_core_last_tf_pert.json" if ckpt == "last" else "evalx_val_core_selected.json")
    if ev.exists():
        e = json.loads(ev.read_text())
        m = e["model"]
        out.update({
            "ra": m["overall"]["mpjpe_ra_mm"], "rot": m["overall"]["root_rot_deg"],
            "ra_g": m["zgz_global"]["mpjpe_ra_mm"][0], "ra_l": m["zgz_local"]["mpjpe_ra_mm"][0],
            "rot_g": m["zgz_global"]["root_rot_deg"][0], "rot_l": m["zgz_local"]["root_rot_deg"][0],
            "fail_eps": m["zgz_global"]["failure"]["episodes"] + m["zgz_local"]["failure"]["episodes"],
        })
        for q in ("zgz_global", "zgz_local"):
            mo = m[q].get("motion")
            if mo:
                t = "g" if "global" in q else "l"
                out[f"rspd_{t}"] = mo["root_speed_ratio"]
                out[f"fspd_{t}"] = mo["finger_speed_ratio"]
                out[f"slope_{t}"] = float(np.mean(mo["root_err_slope_deg_per_10s"])) if mo["root_err_slope_deg_per_10s"] else None
        if "tf" in e:
            out.update({"tf_ra": e["tf"]["overall"]["mpjpe_ra_mm"], "tf_rot": e["tf"]["overall"]["root_rot_deg"],
                        "amp_ra": e["amplification"]["mpjpe_ra_mm"], "amp_rot": e["amplification"]["root_rot_deg"]})
        if "hold" in e:
            out.update({"hold_ra": e["hold"]["overall"]["mpjpe_ra_mm"], "noev_ra": e["noevents"]["overall"]["mpjpe_ra_mm"]})
        if "perturb" in e:
            for th, v in e["perturb"].items():
                r = v["retention_k1_k2_k5_k10_k20"]
                out[f"ret{th}_k1"], out[f"ret{th}_k5"], out[f"ret{th}_k20"] = r[0], r[2], r[4]
    pa = d / "probe_abs_last.json"
    if pa.exists():
        out["probe_abs"] = json.loads(pa.read_text())["zgz_all"]
    return out


COLS = [("ra", "RA"), ("ra_g", "RA-g"), ("ra_l", "RA-l"), ("rot", "rot°"), ("rot_g", "rot-g"), ("rot_l", "rot-l"),
        ("sel_ra", "RA sel"), ("grid_median", "grid med"), ("tf_ra", "TF RA"), ("tf_rot", "TF rot"),
        ("amp_ra", "amp"), ("ret10_k1", "ret10 k1"), ("ret10_k5", "k5"), ("ret10_k20", "k20"),
        ("rspd_g", "rspd g"), ("rspd_l", "rspd l"), ("fspd_g", "fspd g"), ("slope_g", "slope g"),
        ("hold_ra", "hold"), ("fail_eps", "fails"), ("probe_abs", "probe°")]


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, (int, np.integer)):
        return str(v)
    return f"{v:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", default=["3407", "3408"])
    ap.add_argument("--ckpt", default="last", choices=("last", "selected"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows, paired = [], {}
    for arm in [a.base] + a.arms:
        for s in a.seeds:
            r = load_run(f"{arm}_s{s}", a.ckpt)
            if r:
                r["arm"], r["seed"] = arm, s
                rows.append(r)
    by = {(r["arm"], r["seed"]): r for r in rows}
    for arm in a.arms:
        ds = [by[(arm, s)][k] - by[(a.base, s)][k] for s in a.seeds for k in ("ra",)
              if (arm, s) in by and (a.base, s) in by and "ra" in by[(arm, s)] and "ra" in by[(a.base, s)]]
        dr = [by[(arm, s)]["rot"] - by[(a.base, s)]["rot"] for s in a.seeds
              if (arm, s) in by and (a.base, s) in by and "rot" in by[(arm, s)] and "rot" in by[(a.base, s)]]
        paired[arm] = {"d_ra_per_seed": ds, "d_ra_mean": float(np.mean(ds)) if ds else None,
                       "d_rot_per_seed": dr, "same_sign": bool(ds) and (all(x < 0 for x in ds) or all(x > 0 for x in ds))}
    lines = [f"ckpt={a.ckpt}; base={a.base}; paired = arm - base, same seed", "",
             "| run | step | " + " | ".join(h for _, h in COLS) + " |",
             "|---|---|" + "---|" * len(COLS)]
    for r in rows:
        step = r["last_step"] if a.ckpt == "last" else r["sel_step"]
        lines.append(f"| {r['run']} | {step} | " + " | ".join(fmt(r.get(k)) for k, _ in COLS) + " |")
    lines += ["", "| arm | dRA per seed | dRA mean | d rot per seed | same sign |", "|---|---|---|---|---|"]
    for arm, p in paired.items():
        lines.append(f"| {arm} | {', '.join(f'{x:+.2f}' for x in p['d_ra_per_seed'])} | {fmt(p['d_ra_mean'])} | "
                     f"{', '.join(f'{x:+.2f}' for x in p['d_rot_per_seed'])} | {p['same_sign']} |")
    text = "\n".join(lines)
    print(text)
    if a.out:
        o = Path(a.out)
        o.parent.mkdir(parents=True, exist_ok=True)
        o.with_suffix(".md").write_text(text + "\n")
        o.with_suffix(".json").write_text(json.dumps({"rows": rows, "paired": paired}, indent=1, default=str))


if __name__ == "__main__":
    main()
