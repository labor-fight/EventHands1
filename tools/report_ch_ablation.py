#!/usr/bin/env python3
"""Aggregate a render-channel or loss ablation into the arm-level verdict.

Arms are run basenames (`--arms ch_both so3fk`); the `_rep2` replicate of each is
picked up automatically. The input and loss columns are read from each arm's own
config, so an arm that changes only the objective is still legible in the table.

Checkpoints are ranked by closed-loop RA-MPJPE (val_loss is anti-correlated with it
on this task), the best step is taken per replicate, and each arm is reported by the
*worse* of its two replicates, which is the convention the earlier rounds used.

Cost columns are read from `outputs/hand_data51/main_table.json`, which is produced by
`tools/make_main_table.py`; that keeps the sil-only arm's efficiency claim on exactly
the same latency scale as the main results table.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

OUT_ROOT = ROOT / "outputs" / "hand_data51"
CHANNEL_WIDTH = {"sil": 1, "inv": 1, "semsil": 1}
FACE_DESC = {"sil": "silhouette", "inv": "inverse depth",
             "semsil": "semantic silhouette"}
LOSS_DESC = {
    "mse_51d": "51D MSE",
    "so3_trans_fk": "1Lrot + 1Ltrans + 2Lfk",
}


def arm_config(arm):
    """The arm's own yaml. Arms are run basenames, e.g. `ch_both` or `so3fk`."""
    p = ROOT / "configs" / f"eventhands_track_render51_{arm}.yaml"
    return yaml.safe_load(p.read_text()) if p.exists() else None


def arm_input(arm):
    """conv1 input width and a description, read from the arm's own config."""
    cfg = arm_config(arm)
    if cfg is None:
        return None, arm
    faces = cfg["MODEL"].get("RENDER_CHANNELS", ["sil", "inv"])
    n_ch = 2 + sum(CHANNEL_WIDTH[f] for f in faces)
    return n_ch, "LNES + " + " + ".join(FACE_DESC.get(f, f) for f in faces)


def arm_loss(arm):
    """Training objective of the arm, so a loss ablation is legible in the table."""
    cfg = arm_config(arm)
    if cfg is None:
        return "n/a"
    t = str(cfg.get("LOSS", {}).get("TYPE", "mse_51d"))
    return LOSS_DESC.get(t, t)


def mean_of_present(values):
    vals = [v for v in values if v is not None]
    return statistics.fmean(vals) if vals else None


def collect(run):
    """Every scored checkpoint of one replicate, sorted by step."""
    d = OUT_ROOT / f"track_render51_{run}"
    rows = []
    for p in sorted(d.glob("eval_step*/track_metrics_step50.json")):
        m = json.loads(p.read_text())
        o = m["overall"]
        per = {r["seq"]: r for r in m["per_sequence"]}
        rows.append({
            "run": run,
            "step": int(p.parent.name.replace("eval_step", "")),
            "ra": o["mpjpe_ra_mm"],
            "mpvpe_ra": o["mpvpe_ra_mm"],
            "ra_abs": o["mpjpe_abs_mm"],
            "local": per["zgz_local"]["mpjpe_ra_mm"] if "zgz_local" in per else None,
            "global": per["zgz_global"]["mpjpe_ra_mm"] if "zgz_global" in per else None,
            "mpvpe_local": per["zgz_local"]["mpvpe_ra_mm"] if "zgz_local" in per else None,
            "mpvpe_global": per["zgz_global"]["mpvpe_ra_mm"] if "zgz_global" in per else None,
            "jitter_all": mean_of_present(
                [r["jitter_all_mm_per_step"] for r in m["per_sequence"]]),
            "jitter_static": mean_of_present(
                [r["jitter_static_mm_per_step"] for r in m["per_sequence"]]),
            "n": o["n_frames"],
        })
    return sorted(rows, key=lambda r: r["step"])


FULL_PUBLISHED_MS = 1.75  # EventHands-Full in the main table; the latency anchor.


def efficiency_table():
    """Cost columns, read from the main table so the two artifacts cannot disagree.

    Measuring here independently produced latencies that conflicted with the main
    table by up to 0.6 ms, because the ratio to the Full anchor is only stable when
    models are measured in isolation. `tools/make_main_table.py` owns that.
    """
    src = OUT_ROOT / "main_table.json"
    if not src.exists():
        return {}
    by_cfg = {}
    for r in json.loads(src.read_text())["rows"]:
        by_cfg.setdefault(r["cost_config"], r)
    wanted = {
        "EventHands-Full": "eventhands_abs_full51.yaml",
        "EventHands-Track (render)": "eventhands_track_render51.yaml",
    }
    for cfg_name in by_cfg:
        pre, suf = "eventhands_track_render51_", ".yaml"
        if cfg_name.startswith(pre) and cfg_name.endswith(suf) and len(cfg_name) > len(pre) + len(suf):
            wanted[cfg_name[len(pre):-len(suf)]] = cfg_name
    out = {}
    for name, cfg_name in wanted.items():
        r = by_cfg.get(cfg_name)
        if r is None:
            continue
        out[name] = {"config": cfg_name, "latency_raw_ms": r["raw_ms"],
                     "latency_scaled_ms": r["ms"], "flops_per_step": r["macs"],
                     "params": r["params"]}
    return out


def fmt(v, spec=".2f"):
    return "n/a" if v is None else format(v, spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/hand_data51/report_ch_ablation.md")
    ap.add_argument("--no-latency", action="store_true")
    ap.add_argument("--arms", nargs="+", default=["ch_sil", "ch_inv", "ch_both"],
                    help="run basenames, e.g. ch_both so3fk; _rep2 is picked up too")
    ap.add_argument("--control", default="ch_both")
    ap.add_argument("--gate-mm", type=float, default=None,
                    help="pre-registered pooled dRA the arms must beat vs the control")
    args = ap.parse_args()
    ARMS = tuple(args.arms)
    control = args.control

    per_run = {}
    for arm in ARMS:
        for run in (arm, f"{arm}_rep2"):
            rows = collect(run)
            if rows:
                per_run[run] = rows

    lines = [f"# Render-channel / loss ablation: {', '.join(ARMS)}", ""]
    lines.append("Protocol: `model/eval_track.py`, step = 50 ms, GT init + "
                 "`TRACK.PREV_NOISE` (scale 1.0), seed 0, val = `zgz_global` + "
                 "`zgz_local`. Checkpoints are ranked by recursive RA-MPJPE.")
    lines.append("")
    lines.append("## 1. Every checkpoint")
    lines.append("")
    lines.append("| run | step | RA-MPJPE | MPVPE RA | MPJPE abs | local | global "
                 "| jitter all | jitter static |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for run, rows in per_run.items():
        best = min(r["ra"] for r in rows)
        for r in rows:
            mark = " **(sel)**" if r["ra"] == best else ""
            lines.append(
                f"| {run}{mark} | {r['step']} | {r['ra']:.2f} | {r['mpvpe_ra']:.2f} | "
                f"{r['ra_abs']:.2f} | {fmt(r['local'])} | {fmt(r['global'])} | "
                f"{fmt(r['jitter_all'], '.3f')} | {fmt(r['jitter_static'], '.3f')} |")
    lines.append("")

    eff = {} if args.no_latency else efficiency_table()
    lat = {a: eff.get(a, {}).get("latency_scaled_ms") for a in ARMS}

    lines.append("## 2. Arm verdict (worse of two replicates)")
    lines.append("")
    lines.append("| arm | input | loss | RA-MPJPE (worse rep) | rep1 / rep2 | sel steps "
                 "| local | global | jitter all | latency b1 |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    verdict = {}
    for arm in ARMS:
        picks = []
        for run in (arm, f"{arm}_rep2"):
            if run in per_run:
                picks.append(min(per_run[run], key=lambda r: r["ra"]))
        if not picks:
            continue
        claim = max(picks, key=lambda r: r["ra"])
        verdict[arm] = {
            "claim_ra": claim["ra"],
            "replicates": [p["ra"] for p in picks],
            "sel_steps": [p["step"] for p in picks],
            "local": claim["local"],
            "global": claim["global"],
            "mpvpe_local": claim["mpvpe_local"],
            "mpvpe_global": claim["mpvpe_global"],
            "jitter_all": claim["jitter_all"],
            "latency_ms_b1": lat.get(arm),
        }
        n_ch, desc = arm_input(arm)
        lines.append(
            f"| {arm} | {n_ch}ch: {desc} | {arm_loss(arm)} | **{claim['ra']:.2f}** | "
            f"{' / '.join(f'{p:.2f}' for p in verdict[arm]['replicates'])} | "
            f"{' / '.join(str(s) for s in verdict[arm]['sel_steps'])} | "
            f"{fmt(claim['local'])} | {fmt(claim['global'])} | "
            f"{fmt(claim['jitter_all'], '.3f')} | {fmt(lat.get(arm), '.2f')} ms |")
    lines.append("")

    if control in verdict:
        base = verdict[control]["claim_ra"]
        lines.append(f"## 3. Deltas against the {control} control")
        lines.append("")
        lines.append(f"| arm | dRA vs {control} |")
        lines.append("|---|---:|")
        for arm in ARMS:
            if arm in verdict:
                lines.append(f"| {arm} | {verdict[arm]['claim_ra'] - base:+.2f} |")
        lines.append("")
        spread = [abs(v["replicates"][0] - v["replicates"][1])
                  for v in verdict.values() if len(v["replicates"]) == 2]
        if spread:
            lines.append(f"Replicate spread (max over arms): **{max(spread):.2f} mm** — "
                         "arm differences below this are not resolvable.")
            lines.append("")

    # min-over-checkpoints then max-over-replicates is a 2-sample order statistic and
    # its variance is what the spread above measures. Pooling all scored checkpoints
    # per arm uses 12 measurements instead of 2 and is the stabler arm estimate.
    lines.append("## 4. Pooled over all scored checkpoints (2 replicates x 6 steps)")
    lines.append("")
    lines.append("| arm | n | mean RA | sd | sem | min | max |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    pooled = {}
    for arm in ARMS:
        vals = [r["ra"] for run in (arm, f"{arm}_rep2")
                for r in per_run.get(run, [])]
        if not vals:
            continue
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        pooled[arm] = {"n": len(vals), "mean": statistics.fmean(vals), "sd": sd,
                       "sem": sd / len(vals) ** 0.5, "min": min(vals), "max": max(vals)}
        p = pooled[arm]
        lines.append(f"| {arm} | {p['n']} | **{p['mean']:.2f}** | {p['sd']:.2f} | "
                     f"{p['sem']:.2f} | {p['min']:.2f} | {p['max']:.2f} |")
    lines.append("")
    if control in pooled and len(pooled) > 1:
        b = pooled[control]
        head = f"| arm | mean dRA vs {control} | pooled sem of the difference |"
        if args.gate_mm is not None:
            head += f" pre-registered gate (dRA <= {args.gate_mm:+.2f}) |"
        lines.append(head)
        lines.append("|---|---:|---:|" + ("---|" if args.gate_mm is not None else ""))
        for arm in ARMS:
            if arm == control or arm not in pooled:
                continue
            p = pooled[arm]
            d = p["mean"] - b["mean"]
            sem = (p["sem"] ** 2 + b["sem"] ** 2) ** 0.5
            row = f"| {arm} | {d:+.2f} | {sem:.2f} |"
            if args.gate_mm is not None:
                row += " **pass** |" if d <= args.gate_mm else " fail (tie) |"
            lines.append(row)
        lines.append("")
        worst_sem = max(v["sem"] for v in pooled.values())
        lines.append(f"Resolution: the pooled standard error is about {worst_sem:.2f} mm per "
                     f"arm, so differences below roughly {2 * worst_sem * 2 ** 0.5:.1f} mm "
                     "cannot be claimed from this budget.")
        lines.append("")

    if eff:
        lines.append("## 5. Cost, on the main table's latency scale")
        lines.append("")
        lines.append("| model | conv1 in | params | FLOPs/step | latency raw | "
                     "latency scaled |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for name, v in eff.items():
            fl = "n/a" if v["flops_per_step"] is None else f"{v['flops_per_step']/1e9:.3f} G"
            in_ch = arm_input(name)[0]
            if in_ch is None:
                in_ch = 2 if "Full" in name else 4
            lines.append(f"| {name} | {in_ch} | {v['params']/1e6:.2f} M | {fl} | "
                         f"{fmt(v['latency_raw_ms'], '.2f')} ms | "
                         f"{fmt(v['latency_scaled_ms'], '.2f')} ms |")
        lines.append("")
        lines.append(f"Read from `main_table.json`. Raw numbers are machine-specific, so "
                     f"each row is rescaled by EventHands-Full's raw latency to land on "
                     f"the published {FULL_PUBLISHED_MS:.2f} ms anchor.")
        lines.append("")

    if verdict:
        lines.append("## 6. Rows in the main table's format")
        lines.append("")
        lines.append("| Model | MPJPE local | MPJPE global | MPVPE local | MPVPE global "
                     "| Latency | FLOPs/step | Params |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for arm in ARMS:
            if arm not in verdict:
                continue
            v, e = verdict[arm], eff.get(arm, {})
            fl = e.get("flops_per_step")
            lines.append(
                f"| EventHands-Track (render, {arm}) | {fmt(v['local'])} | "
                f"{fmt(v['global'])} | {fmt(v['mpvpe_local'])} | {fmt(v['mpvpe_global'])} | "
                f"{fmt(e.get('latency_scaled_ms'), '.2f')} ms | "
                f"{'n/a' if fl is None else format(fl / 1e9, '.3f') + ' G'} | "
                f"{fmt(e.get('params', 0) / 1e6, '.2f')} M |")
        lines.append("")

    out = ROOT / args.out
    out.write_text("\n".join(lines) + "\n")
    out.with_suffix(".json").write_text(json.dumps(
        {"per_run": per_run, "verdict": verdict, "pooled": pooled,
         "latency_ms_b1": lat, "efficiency": eff}, indent=2))
    print("\n".join(lines))
    print("wrote", out)


if __name__ == "__main__":
    main()
