#!/usr/bin/env python3
"""Insert a wave's last-step numbers (from the selection grids) into the P3 results area of the
pre-registration, before the '- P3 主表与门判定：（待填）' placeholder.
Usage: record_wave.py "<title>" run [run ...]"""
import glob
import json
import sys
from pathlib import Path

title, runs = sys.argv[1], sys.argv[2:]
rows, grids = [], []
for n in runs:
    fs = glob.glob(f"outputs/semkine/{n}/selection_val_core_step50*.json")
    if not fs:
        rows.append(f"  | {n} | （选点网格未完成） | | | |")
        continue
    d = json.load(open(fs[0]))
    g = {r["step"]: r for r in d["grid"]}
    last = g[max(g)]
    ps = last["per_seq"]
    rows.append(f"  | {n} | {last['mpjpe_ra_mm']:.3f}（{ps['zgz_global']:.2f} / {ps['zgz_local']:.2f}） | "
                f"{last['mpjpe_abs_mm']:.1f} | {d['selected']['step']} / {d['selected']['mpjpe_ra_mm']:.3f} | "
                f"{d['grid_median']:.3f} |")
    grids.append(f"{n} " + " ".join(f"{g[s]['mpjpe_ra_mm']:.2f}" for s in sorted(g)))
block = (f"- {title}\n\n"
         "  | run | 末步 RA overall（global / local） | abs | 选中步 / RA（仅对照） | 网格中位 |\n"
         "  |---|---|---|---|---|\n" + "\n".join(rows) + "\n\n"
         "  - 12 点网格（500…6000）：" + "；".join(grids) + "。\n")
doc = Path("docs/EXPERIMENT_RECORDS.md")
t = doc.read_text()
ph = "- P3 主表与门判定：（待填）\n"
assert t.count(ph) == 1
doc.write_text(t.replace(ph, block + ph))
print(block)
