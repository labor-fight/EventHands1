#!/usr/bin/env python3
"""x1001 paired comparison of arms on the development set (or the sealed test).

For every arm and seed it reads `evalx_<split>_<ckpt>.{json,npz}` (written by `evalx.py eval`) and
the selection grid. Arms are compared with the baseline *seed by seed* on identical protocol steps:
per-seed mean difference with a 10 s block-bootstrap CI (evaluation-sample uncertainty only), the
mean over seeds, and whether every seed moves the same way (training-randomness evidence; seeds are
the independent repeats, correlated packets are not).

    python tools/x1001/compare.py --base x1001_s37 --arms x1001_cnn x1001_cnn@wadaptive50_n2000_x300 [--seeds 3407 3408]
                                  [--split val_core] [--ckpt selected] [--out reports/round1]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROG = Path("/data1/lyq/code/mesh/EventHands1_x1001")
BLOCK_MS = 10_000
METRICS = ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "root_rot_deg", "transl_mm")


def block_ci(vals, end, n_boot=2000, seed=0):
    blocks = end // BLOCK_MS
    ub = np.unique(blocks)
    sums = np.array([vals[blocks == b].sum() for b in ub])
    cnts = np.array([(blocks == b).sum() for b in ub])
    rng = np.random.default_rng(seed)
    bs = [sums[i].sum() / cnts[i].sum() for i in (rng.integers(0, len(ub), len(ub)) for _ in range(n_boot))]
    return float(vals.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def load(arm, seed, split, ckpt):
    """`arm` may carry an evaluation variant: `x1001_cnn@wadaptive50_n2000_x300` reads that evalx tag."""
    name, _, variant = arm.partition("@")
    run = PROG / "runs" / f"{name}_s{seed}"
    tag = f"evalx_{split}_{ckpt.replace('=', '')}" + (f"_{variant}" if variant else "")
    js = json.loads((run / f"{tag}.json").read_text())
    z = np.load(run / f"{tag}.npz")
    sel = sorted(run.glob("selection_val_core_step50*.json"))
    grid = json.loads(sel[0].read_text()) if sel else None
    return js, z, grid


def seqs_of(z, mode="model"):
    return sorted({k.split("|")[1] for k in z.files if k.startswith(mode + "|")})


def pooled(z, seqs, key, mode="model", which=None):
    parts = [z[f"{mode}|{s}|{key}"] for s in seqs if which is None or (("_local" in s) == (which == "local"))]
    return np.concatenate(parts) if parts else np.zeros(0)


def ends_of(z, seqs, which=None):
    parts = [z[f"model|{s}|end"] + (10**8 * i) for i, s in enumerate(seqs)
             if which is None or (("_local" in s) == (which == "local"))]
    return np.concatenate(parts) if parts else np.zeros(0, np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[3407, 3408])
    ap.add_argument("--split", default="val_core")
    ap.add_argument("--ckpt", default="selected")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    arms = [a.base] + [x for x in a.arms if x != a.base]
    data = {}
    for arm in arms:
        for s in a.seeds:
            try:
                data[(arm, s)] = load(arm, s, a.split, a.ckpt)
            except FileNotFoundError:
                pass
    rep = {"split": a.split, "ckpt": a.ckpt, "base": a.base, "arms": {}, "paired": {}}
    lines = [f"split={a.split} ckpt={a.ckpt}", "",
             "| arm | seeds | RA all | RA global | RA local | MPVPE-RA all | abs MPJPE | root rot ° | transl mm | "
             "fail time % | hold RA | noevents RA | sel step | grid median | last |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in arms:
        ss = [s for s in a.seeds if (arm, s) in data]
        if not ss:
            continue
        per = {}
        for s in ss:
            js, z, grid = data[(arm, s)]
            seqs = seqs_of(z)
            e = {m: float(pooled(z, seqs, m).mean()) for m in METRICS}
            e["ra_global"] = float(pooled(z, seqs, "mpjpe_ra_mm", which="global").mean())
            e["ra_local"] = float(pooled(z, seqs, "mpjpe_ra_mm", which="local").mean())
            for ctl in ("hold", "noevents"):
                if any(k.startswith(ctl + "|") for k in z.files):
                    e[f"{ctl}_ra"] = float(pooled(z, seqs, "mpjpe_ra_mm", mode=ctl).mean())
            n = sum(js["model"][q]["n_frames"] for q in seqs)
            e["fail_time_frac"] = float(sum(js["model"][q]["failure"]["fail_time_frac"] * js["model"][q]["n_frames"]
                                            for q in seqs) / n)
            e["by_events"] = {q: js["model"][q]["by_events"] for q in seqs}
            if grid:
                e["sel_step"] = grid["selected"]["step"]
                e["grid_median"] = grid["grid_median"]
                e["grid_last"] = grid["grid"][-1]["mpjpe_ra_mm"]
                e["grid"] = [(g["step"], round(g["mpjpe_ra_mm"], 3)) for g in grid["grid"]]
            per[s] = e
        mean = {k: float(np.mean([per[s][k] for s in ss])) for k in per[ss[0]]
                if isinstance(per[ss[0]][k], float)}
        rep["arms"][arm] = {"seeds": ss, "per_seed": per, "mean": mean}

        def cell(k, fmt="{:.2f}"):
            if k not in mean:
                return "-"
            v = fmt.format(mean[k])
            if len(ss) > 1:
                v += "（" + " / ".join(fmt.format(per[s][k]) for s in ss) + "）"
            return v
        lines.append(f"| {arm} | {len(ss)} | {cell('mpjpe_ra_mm')} | {cell('ra_global')} | {cell('ra_local')} | "
                     f"{cell('mpvpe_ra_mm')} | {cell('mpjpe_abs_mm')} | {cell('root_rot_deg')} | {cell('transl_mm')} | "
                     f"{100 * mean.get('fail_time_frac', float('nan')):.1f} | {cell('hold_ra')} | {cell('noevents_ra')} | "
                     f"{' / '.join(str(per[s].get('sel_step', '-')) for s in ss)} | {cell('grid_median')} | {cell('grid_last')} |")
    lines += ["", "Paired vs base (same seed, same steps; arm − base, mm RA; CI = 10 s block bootstrap of that seed's evaluation)", "",
              "| arm | seed | Δ all | Δ global | Δ local | Δ root rot ° |", "|---|---|---|---|---|---|"]
    for arm in arms[1:]:
        ds = []
        for s in a.seeds:
            if (arm, s) not in data or (a.base, s) not in data:
                continue
            zb, za = data[(a.base, s)][1], data[(arm, s)][1]
            seqs = seqs_of(zb)
            assert seqs == seqs_of(za)
            for q in seqs:
                assert np.array_equal(zb[f"model|{q}|end"], za[f"model|{q}|end"]), "steps differ"
            row = {}
            for which in (None, "global", "local"):
                d = pooled(za, seqs, "mpjpe_ra_mm", which=which) - pooled(zb, seqs, "mpjpe_ra_mm", which=which)
                row[which or "all"] = block_ci(d, ends_of(zb, seqs, which))
            d = pooled(za, seqs, "root_rot_deg") - pooled(zb, seqs, "root_rot_deg")
            row["rot"] = block_ci(d, ends_of(zb, seqs))
            rep["paired"][f"{arm}|{s}"] = row
            ds.append(row["all"][0])
            f = lambda v: f"{v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]"   # noqa: E731
            lines.append(f"| {arm} | {s} | {f(row['all'])} | {f(row['global'])} | {f(row['local'])} | {f(row['rot'])} |")
        if ds:
            same = all(x < 0 for x in ds) or all(x > 0 for x in ds)
            lines.append(f"| {arm} | mean | {np.mean(ds):+.2f} | | | | seeds agree in sign: {same} |")
            rep["paired"][f"{arm}|mean"] = {"mean": float(np.mean(ds)), "sign_agree": bool(same), "per_seed": ds}
    text = "\n".join(lines)
    print(text)
    if a.out:
        out = PROG / a.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".md").write_text(text + "\n")
        out.with_suffix(".json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
