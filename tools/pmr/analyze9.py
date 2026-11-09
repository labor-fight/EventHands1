#!/usr/bin/env python3
"""PMR stage S6 written decision (docs/EXPERIMENT_RECORDS.md#pmr-s6-prereg-20261109), computed once after d1hub's
readings and the S4 + S4b + S4c joint selection; CPU only, read-only.

    python -m tools.pmr.analyze9 s6     -> analysis/s6.md, decisions/s6.json

d1hub (seed 3407, from scratch, no ROL) against D1's controls d1n / d1m with tools/pmr/analyze6.py's rule: the screen with
the one-sided health and the global gain G(D1 R) - G(d1hub) >= 0.5 mm -> ADOPT_hub, else HUB_NOT_ADOPTED.
Descriptive: d1hub beside D1 R and the joint selection's arm (decisions/s4c_joint.json), with the root / finger / wrist
split on both zgz recordings.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import analyze6 as A6  # noqa: E402
from tools.pmr import arms_any2 as U  # noqa: E402
from tools.pmr import arms7 as A7  # noqa: E402
from tools.pmr.analyze import fmt, mech  # noqa: E402
from tools.pmr.common import B2_TAG, NOISE_RA, OUT, S43_ENGINE, now, read, require, save  # noqa: E402

CONTROLS = {"d1hub": ("d1n", "d1m")}


def select(rows):
    saved = dict(A6.CONTROLS)
    try:
        A6.CONTROLS.clear()
        A6.CONTROLS.update(CONTROLS)
        sel = A6.select(rows)
    finally:
        A6.CONTROLS.clear()
        A6.CONTROLS.update(saved)
    return {**sel, "verdict": "ADOPT_hub" if sel["selected"] else "HUB_NOT_ADOPTED"}


def split(tag):
    p = OUT / "readings" / tag / "e1" / "report.json"
    if not p.exists():
        return None
    r = read(p)["per_sequence"]
    return {q: {k: r[q][k] for k in ("mpjpe_ra_mm", "wrist_mm", "root_deg", "finger_deg")}
            for q in ("zgz_local", "zgz_global")}


def s6():
    dst = OUT / "decisions" / "s6.json"
    require(not dst.exists(), "written once: " + str(dst))
    M = mech()
    joint = read(OUT / "decisions" / "s4c_joint.json")
    x = joint.get("selected")
    tags = {"d1r": U.run_name("d1r", 3407), "d1n": U.run_name("d1n", 3407), "d1m": U.run_name("d1m", 3407),
            "d1hub": A7.run_name("d1hub", 3407)}
    if x:
        tags[x] = U.run_name(x, 3407)
    rows = {a: M.row("pmr", t) for a, t in tags.items()}
    rows["b2"] = M.row("pfo2", B2_TAG)
    sel = select(rows)
    L = ["## PMR S6：手节点边的逐部位低秩适配 d1hub（种子 3407，从头 6000 步，无 ROL；PFO 引擎；写定规则）", "",
         f"S43（引擎三种子）{' / '.join(fmt(v) for v in S43_ENGINE)}；单种子噪声底约 {NOISE_RA} mm RA。"
         f"联合选择：{joint['verdict']}。", "",
         "| 行 | E1 五项 | E3 | local R15 / F15 / T10 | global 腕 mm / 根 ° / 手指 ° | local 腕 mm / 根 ° / 手指 ° |",
         "|---|---|---|---|---|---|"]
    for k, r in rows.items():
        ret = " / ".join(fmt(r["retention"][q]) for q in ("R15", "F15", "T10")) if "retention" in r else "–"
        s = split(tags[k]) if k in tags else None
        cells = [" / ".join(fmt(s[q][f]) for f in ("wrist_mm", "root_deg", "finger_deg")) if s else "–"
                 for q in ("zgz_global", "zgz_local")]
        L.append(f"| {k} | {' / '.join(fmt(v) for v in r['five'])} | {fmt(r.get('e3'))} | {ret} | {cells[0]} | {cells[1]} |")
    c = sel["candidates"]["d1hub"]
    n = c["rules"]["numbers"]
    L += ["", "| 候选 | 筛选（单侧健全性） | (a) 去掉 P 差 | (b) mlp − X | (c) X − B2 | global 改善 | 采纳 |",
          "|---|---|---|---|---|---|---|",
          f"| d1hub | {c['screen']} | {fmt(n['gap_noprop_minus_r'])} | {fmt(n['mlp_minus_r'])} | {fmt(n['r_minus_b2'])} | "
          f"{fmt(c['global_gain'])} | {c['adopted']} |", "", f"**{sel['verdict']}**"]
    text = "\n".join(L) + "\n"
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    (OUT / "analysis" / "s6.md").write_text(text)
    save(dst, {"rows": rows, **sel, "joint": joint["verdict"], "global_gain_threshold": A6.GLOBAL_GAIN,
               "written_at_utc": now()}, exclusive=True)
    print(text)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("verb", choices=("s6",))
    p.parse_args(argv)
    s6()


if __name__ == "__main__":
    main()
