#!/usr/bin/env python3
"""PMR stage S4 written selection (docs/EXPERIMENT_RECORDS.md#pmr-s4-prereg-20261109), computed once after every reading;
CPU only, read-only.

    python -m tools.pmr.analyze6 s4
        -> analysis/s4.md, decisions/s4.json

For each refinement X (seed 3407, from scratch, no ROL) with its controls (dual: dualn / dualm; hp: D1's d1n / d1m, the
same node inputs): the screen of #pmr-prereg-20261109 (tools/pmr/analyze.py `rules`) with the user's one-sided health
(#pmr-amend2-20261109), and the global gain G(D1 R) - G(X) (MPJPE-RA on zgz_global, seed 3407).
Written rule: X is adopted when its screen passes and its global gain is >= 0.5 mm; among the adopted the lower RA is
selected -> REFINE_<X>; none adopted -> NO_REFINEMENT.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import arms as A  # noqa: E402
from tools.pmr import arms4 as A4  # noqa: E402
from tools.pmr.analyze import fmt, mech, rules  # noqa: E402
from tools.pmr.common import B2_TAG, NOISE_RA, OUT, S43_ENGINE, now, require, save  # noqa: E402
from tools.pmr.decide_h1 import health_one_sided  # noqa: E402

GLOBAL_GAIN = 0.5
CONTROLS = {"d1dual": ("d1dualn", "d1dualm"), "d1hp": ("d1n", "d1m")}


def select(rows):
    """rows: {name: {"five": [...], "e3": ...}} with b2, d1r and every candidate / control (pure)."""
    out, adopted = {}, []
    for x, (n, m) in CONTROLS.items():
        v = rules(rows["b2"], rows[x], rows[n], rows[m])
        h = health_one_sided(v["numbers"]["e3_noprop_minus_r"], v["numbers"]["e3_mlp_minus_r"])
        screen = v["a_necessary"] and v["b_sufficient"] and v["c_no_regress"] and h
        gain = rows["d1r"]["five"][1] - rows[x]["five"][1]
        ok = bool(screen and gain >= GLOBAL_GAIN)
        out[x] = {"screen": bool(screen), "health_one_sided": h, "rules": v, "global_gain": gain, "adopted": ok}
        if ok:
            adopted.append(x)
    best = min(adopted, key=lambda x: rows[x]["five"][4]) if adopted else None
    return {"candidates": out, "verdict": f"REFINE_{best[2:]}" if best else "NO_REFINEMENT", "selected": best}


def s4():
    dst = OUT / "decisions" / "s4.json"
    require(not dst.exists(), "written once: " + str(dst))
    M = mech()
    rows = {a: M.row("pmr", A.run_name(a, 3407)) for a in ("d1r", "d1n", "d1m")}
    rows.update({a: M.row("pmr", A4.run_name(a, 3407)) for a in A4.arms()})
    rows["b2"] = M.row("pfo2", B2_TAG)
    sel = select(rows)
    names = {"b2": "B2", "d1r": "D1 R", "d1n": "D1 R-noprop", "d1m": "D1 R-mlp", "d1dual": "R-dual（双坐标系统计量）",
             "d1dualn": "R-dual-noprop", "d1dualm": "R-dual-mlp", "d1hp": "R-hp（掌—手节点边单列）"}
    L = ["## PMR S4：D1 的 global 改进（种子 3407，从头 6000 步，无 ROL；PFO 引擎；写定规则）", "",
         f"S43（引擎三种子）{' / '.join(fmt(x) for x in S43_ENGINE)}；单种子噪声底约 {NOISE_RA} mm RA。", "",
         "| 行 | E1 五项 | E3 | 放大 | local R15 / F15 / T10 | C |", "|---|---|---|---|---|---|"]
    for k in ("b2", "d1r", "d1n", "d1m", "d1dual", "d1dualn", "d1dualm", "d1hp"):
        r = rows[k]
        ret = " / ".join(fmt(r["retention"][q]) for q in ("R15", "F15", "T10")) if "retention" in r else "–"
        L.append(f"| {names[k]} | {' / '.join(fmt(x) for x in r['five'])} | {fmt(r.get('e3'))} | "
                 f"{fmt(r['amp'][0]) if 'amp' in r else '–'} | {ret} | {fmt(r.get('C'), 3)} |")
    L += ["", "| 候选 | 筛选（单侧健全性） | (a) 去掉 P 差 | (b) mlp − X | (c) X − B2 | global 改善 | 采纳 |", "|---|---|---|---|---|---|---|"]
    for x, c in sel["candidates"].items():
        n = c["rules"]["numbers"]
        L.append(f"| {names[x]} | {c['screen']} | {fmt(n['gap_noprop_minus_r'])} | {fmt(n['mlp_minus_r'])} | "
                 f"{fmt(n['r_minus_b2'])} | {fmt(c['global_gain'])} | {c['adopted']} |")
    L += ["", f"**{sel['verdict']}**"]
    text = "\n".join(L) + "\n"
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    (OUT / "analysis" / "s4.md").write_text(text)
    save(dst, {"rows": rows, **sel, "global_gain_threshold": GLOBAL_GAIN, "written_at_utc": now()}, exclusive=True)
    print(text)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("verb", choices=("s4",))
    p.parse_args(argv)
    s4()


if __name__ == "__main__":
    main()
