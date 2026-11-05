#!/usr/bin/env python3
"""PFO p3e readings (docs/EXPERIMENT_RECORDS.md#pfo-p3e-prereg-20261103): the restore-one arms f1mp (message passing
restored) and f1sr1 (router back to sr1) against F = f1, 2-seed same-seed paired; CPU only, read-only on the readings.
tools/pfo/analyze3.py is frozen with the p3 section and is not edited; this file adds the p3e rule.

    python -m tools.pfo.analyze4 rules [--out FILE]

Rule (written): the cost decision is reversed iff the restored arm's mean E1 RA over seeds 3407 / 3408 is better than
F's by >= 0.3 mm and both seeds go the same way; otherwise the decision stays.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.pfo.analyze import fmt, rep, row5  # noqa: E402
from tools.pfo.arms import run_name  # noqa: E402
from tools.pfo.common import OUT  # noqa: E402

FINAL = "f1"
PAIRED = (3407, 3408)
RESTORE = {"f1mp": "去掉消息传递（nomp）", "f1sr1": "路由器 r_s2d"}
COLS = ("MPJPE-L", "MPJPE-G", "MPVPE-L", "MPVPE-G", "RA")


def e1(arm, seed):
    return row5(rep(run_name(arm, seed), "e1"))


def e3(arm, seed):
    r = rep(run_name(arm, seed), "e3")
    return r["overall"]["mpjpe_ra_mm"] if r else None


def rules():
    out, lines = {}, ["## p3e 写定规则：结构 / 成本决定的逐一恢复（对 F 两种子配对）", ""]
    for arm, what in RESTORE.items():
        pairs = {s: (e1(FINAL, s), e1(arm, s)) for s in PAIRED}
        have = [s for s, (f, x) in pairs.items() if f is not None and x is not None]
        if len(have) < len(PAIRED):
            out[arm] = {"status": "pending", "seeds_read": have}
            lines.append(f"- 恢复臂 {arm}（撤销{what}）：未读齐（已读种子 {have}）")
            continue
        d_ra = [pairs[s][1][4] - pairs[s][0][4] for s in PAIRED]        # restored − F; negative = restored better
        d5 = np.mean([np.array(pairs[s][1]) - np.array(pairs[s][0]) for s in PAIRED], 0)
        reversed_ = bool(np.mean(d_ra) <= -0.3 and all(v < 0 for v in d_ra))
        out[arm] = {"status": "read", "d_ra_per_seed": d_ra, "d_ra_mean": float(np.mean(d_ra)),
                    "d5_mean": d5.tolist(), "reversed": reversed_,
                    "restored_mean5": np.mean([pairs[s][1] for s in PAIRED], 0).tolist(),
                    "f_mean5": np.mean([pairs[s][0] for s in PAIRED], 0).tolist(),
                    "e3_ra": {str(s): (e3(FINAL, s), e3(arm, s)) for s in PAIRED}}
        lines.append(f"- 恢复臂 {arm}（撤销{what}）：恢复 − F 的 E1 RA 两种子 {fmt(d_ra[0])} / {fmt(d_ra[1])}，均值 "
                     f"{fmt(np.mean(d_ra))}；五项均值差 {' / '.join(fmt(v) for v in d5)} → "
                     f"{'撤销该决定' if reversed_ else '保留该决定'}")
    return out, lines


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=("rules",))
    p.add_argument("--out")
    a = p.parse_args(argv)
    out, lines = rules()
    text = "\n".join(lines) + "\n"
    print(text)
    if a.out:
        Path(a.out).write_text(text)
        Path(a.out).with_suffix(".json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    else:
        (OUT / "analysis" / "rules_p3e.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
