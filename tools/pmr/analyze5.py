#!/usr/bin/env python3
"""PMR stage S3b ladder table after engineering amendment 3 (docs/EXPERIMENT_RECORDS.md#pmr-amend3-20261109): the
descriptive table of tools/pmr/analyze4.py (frozen; its rounds rungs d1l1 / d1l2 could not be built) with the rounds rungs
d1l1b / d1l2b (tools/pmr/arms3.py) in their place; computed once after every reading, CPU only, read-only.

    python -m tools.pmr.analyze5 ladder
        -> analysis/s3b_ladder.md, decisions/s3b_ladder.json
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import arms as A  # noqa: E402
from tools.pmr import arms2 as A2  # noqa: E402
from tools.pmr import arms3 as A3  # noqa: E402
from tools.pmr.analyze import fmt, mech  # noqa: E402
from tools.pmr.analyze4 import NAMES  # noqa: E402
from tools.pmr.common import B2_TAG, NOISE_RA, OUT, S43_ENGINE, now, require, save  # noqa: E402

RUNGS = ("d1r", "d1l1b", "d1l2b", "d1g0", "d1h0", "d1a0", "d1n", "d1m", "b2")
LABELS = {**NAMES, "d1l1b": "轮数 1", "d1l2b": "轮数 2"}


def ladder():
    dst = OUT / "decisions" / "s3b_ladder.json"
    require(not dst.exists(), "written once: " + str(dst))
    M = mech()
    rows = {a: M.row("pmr", A.run_name(a, 3407)) for a in ("d1r", "d1n", "d1m")}
    rows.update({a: M.row("pmr", A2.run_name(a, 3407)) for a in ("d1g0", "d1h0", "d1a0")})
    rows.update({a: M.row("pmr", A3.run_name(a, 3407)) for a in A3.arms()})
    rows["b2"] = M.row("pfo2", B2_TAG)
    base = rows["d1r"]["five"]
    diffs = {a: {"ra": r["five"][4] - base[4], "local": r["five"][0] - base[0], "global": r["five"][1] - base[1]}
             for a, r in rows.items()}
    L = ["## PMR D1 阶梯（S3b，修订 3 后；种子 3407，从头 6000 步，无 ROL；PFO 引擎；描述）", "",
         f"S43（引擎三种子）{' / '.join(fmt(x) for x in S43_ENGINE)}；单种子噪声底约 {NOISE_RA} mm RA。", "",
         "| 行 | E1 五项 | E3 | 放大 | local R15 / F15 | C | RA − R（local / global） |", "|---|---|---|---|---|---|---|"]
    for a in RUNGS:
        r, dd = rows[a], diffs[a]
        ret = " / ".join(fmt(r["retention"][q]) for q in ("R15", "F15")) if "retention" in r else "–"
        L.append(f"| {LABELS[a]} | {' / '.join(fmt(x) for x in r['five'])} | {fmt(r.get('e3'))} | "
                 f"{fmt(r['amp'][0]) if 'amp' in r else '–'} | {ret} | {fmt(r.get('C'), 3)} | "
                 f"{fmt(dd['ra'])}（{fmt(dd['local'])} / {fmt(dd['global'])}） |")
    text = "\n".join(L) + "\n"
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    (OUT / "analysis" / "s3b_ladder.md").write_text(text)
    save(dst, {"rows": rows, "diffs_vs_r": diffs, "written_at_utc": now()}, exclusive=True)
    print(text)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("verb", choices=("ladder",))
    p.parse_args(argv)
    ladder()


if __name__ == "__main__":
    main()
