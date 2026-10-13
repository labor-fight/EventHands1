#!/usr/bin/env python3
"""Summarise selection grids of [XYZ-ASSOC] runs: last-step RA (the pre-registered primary readout),
per-sequence RA, selected step (comparability only), grid median. Usage: summ_sel.py run [run ...]"""
import glob
import json
import sys

for n in sys.argv[1:]:
    fs = glob.glob(f"outputs/semkine/{n}/selection_val_core_step50*.json")
    if not fs:
        print(f"{n}: no selection json")
        continue
    d = json.load(open(fs[0]))
    g = {r["step"]: r for r in d["grid"]}
    last = g[max(g)]
    ps = {k.replace("zgz_", ""): round(v, 2) for k, v in last["per_seq"].items()}
    print(f"{n}: last step {max(g)} RA={last['mpjpe_ra_mm']:.3f} {ps} abs={last['mpjpe_abs_mm']:.1f}"
          f" | selected step {d['selected']['step']} RA={d['selected']['mpjpe_ra_mm']:.3f}"
          f" | grid median {d['grid_median']:.3f}")
    print("   grid:", " ".join(f"{s}:{g[s]['mpjpe_ra_mm']:.2f}" for s in sorted(g)))
