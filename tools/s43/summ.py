#!/usr/bin/env python3
"""S43: one line per evaluation npz (gate_eval / filter_eval output): overall / global / local RA, root deg, translation."""
import sys
from pathlib import Path
import numpy as np


def row(npz):
    z = np.load(npz)
    out = {}
    tot = {k: [0.0, 0] for k in ("mpjpe_ra_mm", "mpvpe_ra_mm", "root_rot_deg", "transl_mm")}
    for s in ("zgz_global", "zgz_local"):
        for k in tot:
            v = z[f"model|{s}|{k}"]
            out[(s, k)] = float(v.mean())
            tot[k][0] += float(v.sum())
            tot[k][1] += len(v)
    for k, (a, n) in tot.items():
        out[("all", k)] = a / n
    return out


if __name__ == "__main__":
    print(f"{'file':55s} {'RA':>7s} {'RA_g':>7s} {'RA_l':>7s} {'V_g':>6s} {'V_l':>6s} {'rot':>5s} {'rot_l':>5s} {'T_g':>6s} {'T_l':>6s}")
    for p in sys.argv[1:]:
        r = row(p)
        print(f"{str(Path(p).parent.name + '/' + Path(p).stem)[-55:]:55s} {r[('all','mpjpe_ra_mm')]:7.4f} "
              f"{r[('zgz_global','mpjpe_ra_mm')]:7.3f} {r[('zgz_local','mpjpe_ra_mm')]:7.3f} "
              f"{r[('zgz_global','mpvpe_ra_mm')]:6.2f} {r[('zgz_local','mpvpe_ra_mm')]:6.2f} "
              f"{r[('all','root_rot_deg')]:5.2f} {r[('zgz_local','root_rot_deg')]:5.2f} "
              f"{r[('zgz_global','transl_mm')]:6.1f} {r[('zgz_local','transl_mm')]:6.1f}")
