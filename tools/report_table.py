#!/usr/bin/env python3
"""The one results table every report uses (user's format, 2026-09-21).

    | 网络结构 | MPJPE-local | MPJPE-global | MPVPE-local | MPVPE-global | RA-MPJPE(递推) | Latency | FLOPs/step | Params |

Nothing else is a headline metric. Rows are read from the `outputs/semkine/<run>_main_row.json`
files `tools/make_s36_row.py` writes (two-seed mean on the zgz protocol; the RA cell carries the
two seeds in parentheses because the protocol reports per-seed numbers), so the table cannot
drift from the artifacts. The first row is the published EventHands-PCA6 baseline the user's
table starts from.

    python tools/report_table.py s36_eventgnn s37_routed s37_fkgraph s37_meshgraph
    python tools/report_table.py --label s37_routed="S37 路由读出（当前臂）" s37_routed
    python tools/report_table.py --tag last xa_c1 xa_orc      # <run>_main_row_last.json (make_s36_row --step last)

`--tag <suffix>` (default empty) reads `<run>_main_row<tag>.json`; a leading underscore is added when
missing, so `--tag last` and `--tag _last` both read `<run>_main_row_last.json`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "semkine"

HEADER = ("| 网络结构 | MPJPE-local | MPJPE-global | MPVPE-local | MPVPE-global | RA-MPJPE(递推) "
          "| Latency | FLOPs/step | Params |")
RULE = "|---|---|---|---|---|---|---|---|---|"
#: the row the user's table starts from (published EventHands-PCA6 numbers)
BASELINE = "| EventHands-PCA6 (baseline) | 30 | 10.99 | 23.58 | 8.15 | - | 1.76 ms | 1.653 G | 11.18 M |"


def row(run: str, label: str | None = None, seeds: bool = True, tag: str = "") -> str:
    p = OUT / f"{run}_main_row{tag}.json"
    r = json.loads(p.read_text())
    m = r["two_seed_mean"]
    ra = f"{m['overall']['mpjpe_ra_mm']:.2f}"
    if seeds and len(r["per_seed"]) > 1:
        ra += "（" + " / ".join(f"{r['per_seed'][s]['overall']['mpjpe_ra_mm']:.2f}" for s in sorted(r["per_seed"])) + "）"
    return (f"| {label or run} | {m['local']['mpjpe_ra_mm']:.2f} | {m['global']['mpjpe_ra_mm']:.2f} "
            f"| {m['local']['mpvpe_ra_mm']:.2f} | {m['global']['mpvpe_ra_mm']:.2f} | {ra} "
            f"| {r['latency_ms_scaled_full1p75']:.2f} ms | {r['macs_forward_packet'] / 1e9:.3f} G "
            f"| {r['params_total'] / 1e6:.2f} M |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run names with an outputs/semkine/<run>_main_row.json")
    ap.add_argument("--label", action="append", default=[], metavar="RUN=LABEL",
                    help="display name for a run (default: the run name)")
    ap.add_argument("--no-seeds", action="store_true", help="RA cell without the per-seed values")
    ap.add_argument("--no-baseline", action="store_true", help="omit the EventHands-PCA6 row")
    ap.add_argument("--tag", default="",
                    help="suffix of the main-row file, <run>_main_row<tag>.json (e.g. `last` for "
                         "`make_s36_row.py --step last`); default: the selected-step row")
    a = ap.parse_args()
    tag = a.tag if (not a.tag or a.tag.startswith("_")) else "_" + a.tag
    labels = dict(kv.split("=", 1) for kv in a.label)
    lines = [HEADER, RULE]
    if not a.no_baseline:
        lines.append(BASELINE)
    lines += [row(run, labels.get(run), seeds=not a.no_seeds, tag=tag) for run in a.runs]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
