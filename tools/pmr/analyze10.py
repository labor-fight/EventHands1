#!/usr/bin/env python3
"""PMR stage S7 written decision (docs/EXPERIMENT_RECORDS.md#pmr-s7-prereg-20261109), computed once after every S7 reading;
CPU only, read-only.

    python -m tools.pmr.analyze10 s7 --arms d1ghandhp[,d1ghandhub]     -> analysis/s7.md, decisions/s7.json

For each combination Y (seed 3407, from scratch, no ROL): the screen of tools/pmr/analyze6.py against D1's controls
d1n / d1m and B2 (one-sided health). Y replaces ghand as the final-form candidate when its screen passes, its RA is lower
than ghand's (seed 3407) by >= REPLACE_RA and its MPJPE-L is not higher than ghand's by more than LOCAL_GUARD; among such
the lowest RA -> REPLACE_<Y>, none -> KEEP_ghand. Descriptive: the root / finger / wrist split on both recordings.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import analyze9 as A9  # noqa: E402
from tools.pmr import arms8 as A8  # noqa: E402
from tools.pmr import arms_any2 as U  # noqa: E402
from tools.pmr.analyze import fmt, mech, rules  # noqa: E402
from tools.pmr.common import B2_TAG, NOISE_RA, OUT, S43_ENGINE, now, require, save  # noqa: E402
from tools.pmr.decide_h1 import health_one_sided  # noqa: E402

REPLACE_RA = 0.3
LOCAL_GUARD = 0.3


def decide(rows, arms):
    """rows: b2, d1n, d1m, d1ghand and every arm (pure)."""
    out, better = {}, []
    g = rows["d1ghand"]["five"]
    for y in arms:
        v = rules(rows["b2"], rows[y], rows["d1n"], rows["d1m"])
        h = health_one_sided(v["numbers"]["e3_noprop_minus_r"], v["numbers"]["e3_mlp_minus_r"])
        screen = bool(v["a_necessary"] and v["b_sufficient"] and v["c_no_regress"] and h)
        f = rows[y]["five"]
        ok = screen and g[4] - f[4] >= REPLACE_RA and f[0] - g[0] <= LOCAL_GUARD
        out[y] = {"screen": screen, "health_one_sided": h, "rules": v, "ra_gain_vs_ghand": g[4] - f[4],
                  "local_vs_ghand": f[0] - g[0], "replaces": bool(ok)}
        if ok:
            better.append(y)
    best = min(better, key=lambda y: rows[y]["five"][4]) if better else None
    return {"candidates": out, "verdict": f"REPLACE_{best[2:]}" if best else "KEEP_ghand", "selected": best}


def s7(arms):
    dst = OUT / "decisions" / "s7.json"
    require(not dst.exists(), "written once: " + str(dst))
    M = mech()
    tags = {a: U.run_name(a, 3407) for a in ("d1r", "d1n", "d1m", "d1ghand")}
    tags.update({a: A8.run_name(a, 3407) for a in arms})
    rows = {a: M.row("pmr", t) for a, t in tags.items()}
    rows["b2"] = M.row("pfo2", B2_TAG)
    sel = decide(rows, arms)
    L = ["## PMR S7：ghand 与手节点改进的组合（种子 3407，从头 6000 步，无 ROL；PFO 引擎；写定规则）", "",
         f"S43（引擎三种子）{' / '.join(fmt(v) for v in S43_ENGINE)}；单种子噪声底约 {NOISE_RA} mm RA。", "",
         "| 行 | E1 五项 | E3 | local R15 / F15 / T10 | global 腕 mm / 根 ° / 手指 ° | local 腕 mm / 根 ° / 手指 ° |",
         "|---|---|---|---|---|---|"]
    for k, r in rows.items():
        ret = " / ".join(fmt(r["retention"][q]) for q in ("R15", "F15", "T10")) if "retention" in r else "–"
        s = A9.split(tags[k]) if k in tags else None
        cells = [" / ".join(fmt(s[q][f]) for f in ("wrist_mm", "root_deg", "finger_deg")) if s else "–"
                 for q in ("zgz_global", "zgz_local")]
        L.append(f"| {k} | {' / '.join(fmt(v) for v in r['five'])} | {fmt(r.get('e3'))} | {ret} | {cells[0]} | {cells[1]} |")
    L += ["", "| 组合 | 筛选（单侧健全性） | (a) 去掉 P 差 | (b) mlp − Y | (c) Y − B2 | RA 比 ghand 好 | local 比 ghand 差 | 替换 |",
          "|---|---|---|---|---|---|---|---|"]
    for y, c in sel["candidates"].items():
        n = c["rules"]["numbers"]
        L.append(f"| {y} | {c['screen']} | {fmt(n['gap_noprop_minus_r'])} | {fmt(n['mlp_minus_r'])} | {fmt(n['r_minus_b2'])} | "
                 f"{fmt(c['ra_gain_vs_ghand'])} | {fmt(c['local_vs_ghand'])} | {c['replaces']} |")
    L += ["", f"**{sel['verdict']}**"]
    text = "\n".join(L) + "\n"
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    (OUT / "analysis" / "s7.md").write_text(text)
    save(dst, {"rows": rows, **sel, "replace_ra": REPLACE_RA, "local_guard": LOCAL_GUARD, "written_at_utc": now()},
         exclusive=True)
    print(text)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("s7")
    s.add_argument("--arms", required=True)
    a = p.parse_args(argv)
    s7(a.arms.split(","))


if __name__ == "__main__":
    main()
