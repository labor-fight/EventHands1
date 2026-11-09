#!/usr/bin/env python3
"""PMR stages S4 + S4b joint written selection (docs/EXPERIMENT_RECORDS.md#pmr-s4b-prereg-20261109): tools/pmr/analyze6.py's
rule (frozen with S4) over the three refinements dual, hp and pos, computed once after every reading of both stages; CPU
only, read-only. It supersedes decisions/s4.json (dual / hp only) for the choice of the S5 arm.

    python -m tools.pmr.analyze6b joint
        -> analysis/s4_joint.md, decisions/s4_joint.json

For each refinement X (seed 3407, from scratch, no ROL) with its controls (dual: dualn / dualm; pos: posn / posm; hp: D1's
d1n / d1m): the screen of #pmr-prereg-20261109 with the one-sided health (#pmr-amend2-20261109) and the global gain
G(D1 R) - G(X). X is adopted when its screen passes and its global gain is >= 0.5 mm; among the adopted the lower RA is
selected -> REFINE_<X>; none adopted -> NO_REFINEMENT.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import analyze6 as A6  # noqa: E402
from tools.pmr import arms_any as U  # noqa: E402
from tools.pmr.analyze import fmt, mech  # noqa: E402
from tools.pmr.common import B2_TAG, NOISE_RA, OUT, S43_ENGINE, now, require, save  # noqa: E402

CONTROLS = {**A6.CONTROLS, "d1pos": ("d1posn", "d1posm")}


def select(rows):
    saved = dict(A6.CONTROLS)
    try:
        A6.CONTROLS.clear()
        A6.CONTROLS.update(CONTROLS)
        return A6.select(rows)
    finally:
        A6.CONTROLS.clear()
        A6.CONTROLS.update(saved)


def joint():
    dst = OUT / "decisions" / "s4_joint.json"
    require(not dst.exists(), "written once: " + str(dst))
    M = mech()
    arms = ["d1r", "d1n", "d1m"] + [a for x, c in CONTROLS.items() for a in (x, *c) if a not in ("d1n", "d1m")]
    rows = {a: M.row("pmr", U.run_name(a, 3407)) for a in dict.fromkeys(arms)}
    rows["b2"] = M.row("pfo2", B2_TAG)
    sel = select(rows)
    L = ["## PMR S4 + S4b：D1 的 global 改进，联合选择（种子 3407，从头 6000 步，无 ROL；PFO 引擎；写定规则）", "",
         f"S43（引擎三种子）{' / '.join(fmt(x) for x in S43_ENGINE)}；单种子噪声底约 {NOISE_RA} mm RA。", "",
         "| 行 | E1 五项 | E3 | 放大 | local R15 / F15 / T10 | C |", "|---|---|---|---|---|---|"]
    for k, r in rows.items():
        ret = " / ".join(fmt(r["retention"][q]) for q in ("R15", "F15", "T10")) if "retention" in r else "–"
        L.append(f"| {k} | {' / '.join(fmt(x) for x in r['five'])} | {fmt(r.get('e3'))} | "
                 f"{fmt(r['amp'][0]) if 'amp' in r else '–'} | {ret} | {fmt(r.get('C'), 3)} |")
    L += ["", "| 候选 | 筛选（单侧健全性） | (a) 去掉 P 差 | (b) mlp − X | (c) X − B2 | global 改善 | 采纳 |", "|---|---|---|---|---|---|---|"]
    for x, c in sel["candidates"].items():
        n = c["rules"]["numbers"]
        L.append(f"| {x} | {c['screen']} | {fmt(n['gap_noprop_minus_r'])} | {fmt(n['mlp_minus_r'])} | {fmt(n['r_minus_b2'])} | "
                 f"{fmt(c['global_gain'])} | {c['adopted']} |")
    L += ["", f"**{sel['verdict']}**"]
    text = "\n".join(L) + "\n"
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    (OUT / "analysis" / "s4_joint.md").write_text(text)
    save(dst, {"rows": rows, **sel, "global_gain_threshold": A6.GLOBAL_GAIN, "written_at_utc": now()}, exclusive=True)
    print(text)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("verb", choices=("joint",))
    p.parse_args(argv)
    joint()


if __name__ == "__main__":
    main()
