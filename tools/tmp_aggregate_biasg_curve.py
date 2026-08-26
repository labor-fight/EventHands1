"""Aggregate the bias-vs-retention curve across every echo-gain probe run.

Reads any number of `tmp_probe_echo_gain.py --out` JSONs, de-duplicates arms by
checkpoint path, and reports the three quantities the verdict rests on:

  * the closed-form amplification law   recursive ~= bias / (1 - G)
  * the anti-correlation               bias falls as G rises
  * the location of the interior optimum in recursive error

Every arm must come from the same split / step-ms / route or the rows are not
comparable; mismatches are refused rather than silently pooled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _rank(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        shared = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def _spearman(xs: list[float], ys: list[float]) -> float:
    return _pearson(_rank(xs), _rank(ys))


def _fit_line(xs: list[float], ys: list[float]) -> tuple[float, float]:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den > 0 else 0.0
    return slope, my - slope * mx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="+", help="echo-gain probe outputs to pool")
    ap.add_argument("--out", default="", help="optional path for the merged table")
    args = ap.parse_args()

    FIELDS = {
        "G": "gain_rand_pooled",
        "G_self": "gain_self_pooled",
        "bias": "err_tf_mm",
        "recur": "err_recursive_mm",
    }

    reps: dict[str, dict] = {}
    protocol: dict[str, object] = {}
    for path in args.json:
        blob = json.loads(Path(path).read_text())
        here = {k: blob.get(k) for k in ("split", "step_ms", "route")}
        if protocol and here != protocol:
            raise SystemExit(f"protocol mismatch in {path}: {here} vs {protocol}")
        protocol = here
        for label, arm in blob["arms"].items():
            # Same checkpoint measured twice is one data point measured twice, not
            # two data points; the spread across repeats is the probe's noise floor.
            slot = reps.setdefault(
                arm["ckpt"],
                {"label": label, "ckpt": arm["ckpt"], "sources": [], "reps": []},
            )
            slot["sources"].append(Path(path).name)
            slot["reps"].append({k: arm["pooled"][v] for k, v in FIELDS.items()})

    table = []
    for slot in reps.values():
        row = {k: slot[k] for k in ("label", "ckpt", "sources")}
        row["n_reps"] = len(slot["reps"])
        for key in FIELDS:
            vals = [r[key] for r in slot["reps"]]
            row[key] = sum(vals) / len(vals)
            row[f"{key}_spread"] = max(vals) - min(vals)
        table.append(row)

    table.sort(key=lambda r: r["G"])
    for r in table:
        r["amp_measured"] = r["recur"] / r["bias"]
        r["amp_pred"] = 1.0 / (1.0 - r["G"])
        r["law_ratio"] = (r["bias"] * r["amp_pred"]) / r["recur"]

    gs = [r["G"] for r in table]
    biases = [r["bias"] for r in table]
    recurs = [r["recur"] for r in table]
    slope, intercept = _fit_line(gs, biases)

    print(f"protocol: {protocol}   n={len(table)}   G span {min(gs):.3f}-{max(gs):.3f}\n")
    head = f"{'arm':<24}{'G':>7}{'G_self':>8}{'bias':>8}{'recur':>8}{'amp':>7}{'1/(1-G)':>9}{'law':>7}{'bias_fit':>10}"
    print(head)
    print("-" * len(head))
    for r in table:
        fit = slope * r["G"] + intercept
        print(
            f"{r['label']:<24}{r['G']:>7.3f}{r['G_self']:>8.3f}{r['bias']:>8.2f}"
            f"{r['recur']:>8.2f}{r['amp_measured']:>7.3f}{r['amp_pred']:>9.3f}"
            f"{r['law_ratio']:>7.2f}{r['bias'] - fit:>+10.2f}"
        )

    best = min(table, key=lambda r: r["recur"])
    lo = [r for r in table if r["G"] < best["G"]]
    hi = [r for r in table if r["G"] > best["G"]]

    repeated = [r for r in table if r["n_reps"] > 1]
    print()
    if repeated:
        print("probe noise floor (same checkpoint, repeated measurements):")
        for r in repeated:
            print(
                f"  {r['label']:<22} x{r['n_reps']}  spread G {r['G_spread']:.3f}"
                f"  bias {r['bias_spread']:.2f} mm  recursive {r['recur_spread']:.2f} mm"
            )
    else:
        print("probe noise floor: UNKNOWN, no checkpoint was measured more than once")
    print()
    print(f"amplification law   Pearson(1/(1-G), amp) = {_pearson([r['amp_pred'] for r in table], [r['amp_measured'] for r in table]):+.3f}")
    print(f"                    law ratio range       = {min(r['law_ratio'] for r in table):.2f}-{max(r['law_ratio'] for r in table):.2f}")
    print(f"bias vs G           Spearman              = {_spearman(gs, biases):+.3f}   fit {slope:+.2f}*G {intercept:+.2f} mm")
    print(f"recursive vs G      Spearman              = {_spearman(gs, recurs):+.3f}")
    print()
    print(f"optimum             {best['label']} at G={best['G']:.3f}, recursive {best['recur']:.2f} mm")
    print(f"  {len(lo)} arm(s) below: " + ", ".join(f"{r['recur']:.2f}" for r in lo))
    print(f"  {len(hi)} arm(s) above: " + ", ".join(f"{r['recur']:.2f}" for r in hi))
    if lo and hi:
        print("  -> interior optimum: both directions are worse")
    else:
        side = "low-G" if not lo else "high-G"
        print(f"  -> BOUNDARY optimum, no data on the {side} side; do not claim an interior minimum")

    if args.out:
        Path(args.out).write_text(
            json.dumps({"protocol": protocol, "arms": table}, indent=1)
        )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
