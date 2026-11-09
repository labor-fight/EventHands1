#!/usr/bin/env python3
"""PMR stage S5 written verdict (docs/EXPERIMENT_RECORDS.md#pmr-s5-prereg-20261109), computed once after every reading;
CPU only, read-only.

    python -m tools.pmr.analyze7 s5 --arm d1dual|d1hp|d1pos|d1ghand|d1gview
        -> analysis/s5_<arm>.md, decisions/s5_<arm>.json, main rows
           outputs/semkine/pmr_<arm>_rol5_3seed_main_row.json (+ the controls' rol5 rows, seed 3407)

Written rules (from scratch rows: E1 of X and its no-P control at 3407 / 3408 / 3409; rollout rows: E1 of X + rol5):
  module    RA(no-P, s) - RA(X, s) > 0 for every seed -> MODULE_ESTABLISHED (else MODULE_NOT_ESTABLISHED)
  done      the three-seed means of X + rol5's E1 five all below S43's engine values -> DONE (else NOT_DONE)
The goal's "完成" needs both. Descriptive: per-seed rows, the controls + rol5 (3407), D1 R + rol5 and B2 + rol5 means.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.pmr import arms_any2 as U  # noqa: E402
from tools.pmr import arms_rol2 as R2  # noqa: E402
from tools.pmr.analyze import fmt, mech  # noqa: E402
from tools.pmr.analyze2 import rule as module_rule  # noqa: E402
from tools.pmr.analyze3 import done, main_row  # noqa: E402
from tools.pmr.common import OUT, S43_ENGINE, SEEDS, now, require, save  # noqa: E402

NOPROP = {"d1dual": "d1dualn", "d1hp": "d1n", "d1pos": "d1posn", "d1ghand": "d1n", "d1gview": "d1n"}
MLP = {"d1dual": "d1dualm", "d1hp": "d1m", "d1pos": "d1posm", "d1ghand": "d1m", "d1gview": "d1m"}
TPD_OUT = ROOT / "outputs/tpd/20261105"


def s5(arm):
    require(arm in NOPROP, "S5 arm")
    dst = OUT / "decisions" / f"s5_{arm}.json"
    require(not dst.exists(), "written once: " + str(dst))
    M = mech()
    M.ROOTS["tpd"] = TPD_OUT
    n_arm, m_arm = NOPROP[arm], MLP[arm]
    rows = {}
    for s in SEEDS:
        rows[f"x_s{s}"] = M.row("pmr", U.run_name(arm, s))
        rows[f"n_s{s}"] = M.row("pmr", U.run_name(n_arm, s))
        rows[f"xrol_s{s}"] = M.row("pmr", R2.run_name(f"{arm}_rol5", s))
        rows[f"d1rol_s{s}"] = M.row("pmr", R2.run_name("d1r_rol5", s))
        rows[f"b2rol_s{s}"] = M.row("tpd", f"tpd_rol5_s{s}")
    rows["nrol_s3407"] = M.row("pmr", R2.run_name(f"{n_arm}_rol5", 3407))
    rows["mrol_s3407"] = M.row("pmr", R2.run_name(f"{m_arm}_rol5", 3407))
    gaps = {s: rows[f"n_s{s}"]["five"][4] - rows[f"x_s{s}"]["five"][4] for s in SEEDS}
    module = module_rule(gaps)
    mean = lambda k: [float(np.mean([rows[f"{k}_s{s}"]["five"][i] for s in SEEDS])) for i in range(5)]  # noqa: E731
    xm, d1m, b2m = mean("xrol"), mean("d1rol"), mean("b2rol")
    verdict = done(xm)
    final = "DONE" if (verdict == "DONE" and module == "MODULE_ESTABLISHED") else "NOT_DONE"
    L = [f"## PMR S5：{arm} 的三种子确认与 rol5（PFO 引擎；普通分包形式；写定规则）", "",
         f"S43（引擎三种子）{' / '.join(fmt(x) for x in S43_ENGINE)}。", "",
         "| 行 | E1 五项 | E3 | 放大 |", "|---|---|---|---|"]
    for k, r in rows.items():
        five = " / ".join(fmt(x) for x in r["five"]) if "five" in r else "–"
        L.append(f"| {k} | {five} | {fmt(r.get('e3'))} | {fmt(r['amp'][0]) if 'amp' in r else '–'} |")
    L += ["", f"{arm} + rol5 三种子均值 {' / '.join(fmt(x) for x in xm)}；D1 R + rol5 {' / '.join(fmt(x) for x in d1m)}；"
          f"B2 + rol5 {' / '.join(fmt(x) for x in b2m)}",
          "去掉 P 的差距（从头，无 ROL）：" + "；".join(f"{s} {fmt(g)}" for s, g in gaps.items()) + f" → **{module}**",
          f"五项均值全部低于 S43 → **{verdict}**；目标的完成 → **{final}**"]
    text = "\n".join(L) + "\n"
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    (OUT / "analysis" / f"s5_{arm}.md").write_text(text)
    rows_written = [str(main_row(f"{arm}_rol5", SEEDS, f"PMR {arm} + rol5（三种子，普通分包形式）", U.run_name(arm, 3407)))]
    save(dst, {"arm": arm, "rows": rows, "gaps": {str(k): v for k, v in gaps.items()}, "module": module,
               "mean_x_rol5": xm, "mean_d1r_rol5": d1m, "mean_b2_rol5": b2m, "done": verdict, "final": final,
               "main_rows": rows_written, "written_at_utc": now()}, exclusive=True)
    print(text)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("s5")
    s.add_argument("--arm", required=True, choices=tuple(NOPROP))
    a = p.parse_args(argv)
    s5(a.arm)


if __name__ == "__main__":
    main()
