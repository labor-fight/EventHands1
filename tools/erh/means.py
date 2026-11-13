#!/usr/bin/env python3
"""EvHandPose's dataset-dependent crop-frame translation mean for EvRealHands (docs/EXPERIMENT_RECORDS.md#erh-prereg-20261111):
tools/real/means.py's definition and rule unchanged (z' measured on the training labels, the port's 0.52 scaled by the
measured EvRealHands / hand_data51 ratio, x / y from the event-frame intrinsics), on erh_n346 with splits_erh.json;
writes outputs/erh/20261111/trans_means.json (read by tools/real/ns.py `trans_mean_crop` in the EvRealHands namespace).

    python tools/erh/means.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.erh import ns as E  # noqa: E402
from tools.real import means as M  # noqa: E402


def main():
    out = {"sim_check": M.measure(M.SIM, "splits_semkine.json", 0.375),
           "erh": M.measure(E.CANVAS["n346"]["root"], E.SPLITS, 346 / 640), "port_constant_sim": [-0.13, -0.06, 0.52]}
    zp = 0.52 * out["erh"]["z_prime_mean"] / out["sim_check"]["z_prime_mean"]
    fx, fy, cx, cy = out["erh"]["event_frame_K"]
    out["used"] = {"rule": "0.52 x EvRealHands / sim measured z' (tools/real/means.py)", "z_prime": zp,
                   "trans_mean_crop": [-(cx - 64) * zp / fx, -(cy - 64) * zp / fy, zp]}
    p = E.OUT / "trans_means.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
