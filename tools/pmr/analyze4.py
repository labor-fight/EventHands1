#!/usr/bin/env python3
"""PMR stage S3b ladder table (docs/EXPERIMENT_RECORDS.md#pmr-s3b-prereg-20261109), descriptive, computed once after
every reading; CPU only, read-only.

    python -m tools.pmr.analyze4 ladder
        -> analysis/s3b_ladder.md, decisions/s3b_ladder.json

Rows (tools/evf/mech.py `row`, seed 3407, from scratch, no ROL): R (3 rounds), the rungs (rounds 1 / 2, gate without
geometry, no hand-node edges, no event-mesh association), and the screen's anchors R-noprop / R-mlp / B2. For every rung:
RA(rung) - RA(R) and the local / global differences (single seed, noise floor ~0.8 mm RA; descriptive, no gate).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import arms as A  # noqa: E402
from tools.pmr import arms2 as A2  # noqa: E402
from tools.pmr.analyze import fmt, mech  # noqa: E402
from tools.pmr.common import B2_TAG, NOISE_RA, OUT, S43_ENGINE, now, require, save  # noqa: E402

NAMES = {"d1r": "R（3 轮，D1）", "d1l1": "轮数 1", "d1l2": "轮数 2", "d1g0": "门控消融（门不读几何）",
         "d1h0": "手节点消融（只有树边）", "d1a0": "无网格关联（事件不落到前态网格）", "d1n": "R-noprop（去掉 P）",
         "d1m": "R-mlp（拼接 + MLP）", "b2": "B2"}


def ladder():
    dst = OUT / "decisions" / "s3b_ladder.json"
    require(not dst.exists(), "written once: " + str(dst))
    M = mech()
    rows = {a: M.row("pmr", A.run_name(a, 3407)) for a in ("d1r", "d1n", "d1m")}
    rows.update({a: M.row("pmr", A2.run_name(a, 3407)) for a in A2.arms()})
    rows["b2"] = M.row("pfo2", B2_TAG)
    base = rows["d1r"]["five"]
    diffs = {a: {"ra": r["five"][4] - base[4], "local": r["five"][0] - base[0], "global": r["five"][1] - base[1]}
             for a, r in rows.items()}
    L = ["## PMR D1 阶梯（S3b；种子 3407，从头 6000 步，无 ROL；PFO 引擎；描述）", "",
         f"S43（引擎三种子）{' / '.join(fmt(x) for x in S43_ENGINE)}；单种子噪声底约 {NOISE_RA} mm RA。", "",
         "| 行 | E1 五项 | E3 | 放大 | local R15 / F15 | C | RA − R（local / global） |", "|---|---|---|---|---|---|---|"]
    for a in ("d1r", "d1l1", "d1l2", "d1g0", "d1h0", "d1a0", "d1n", "d1m", "b2"):
        r, dd = rows[a], diffs[a]
        ret = " / ".join(fmt(r["retention"][q]) for q in ("R15", "F15")) if "retention" in r else "–"
        L.append(f"| {NAMES[a]} | {' / '.join(fmt(x) for x in r['five'])} | {fmt(r.get('e3'))} | "
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
