#!/usr/bin/env python3
"""ESMNet without the part loss: trunk table with two seeds (docs/EXPERIMENT_RECORDS.md#nopart-amend2-20261113).
CPU only. Same readings, rows and columns as tools/nopart/table.py (frozen, three seeds). Only the seeds and the rule change.

User 2026-11-13 ~07:25 UTC: "不用那么多种子啊，种子2个就行". RFSU's sim:rfsr:3409 and sim:rfsrnp:3409 are SKIPPED, so
rfsrnp_rol5 pairs with rfsr_rol5 on seeds 3407 / 3408 only.

    python tools/nopart/table2.py [--out outputs/nopart/20261113/table.md]

Rule (written before any NOPART reading): per seed, d = rfsrnp_rol5 - rfsr_rol5 on MPJPE / MPVPE of zgz_local and
zgz_global (streaming E1, root-aligned, mm).
  per metric: both seeds d > 0 and mean d >= +0.5 mm -> "p_i helps" (dropping the part loss is worse);
              both seeds d < 0 and mean d <= -0.5 mm -> "dropping helps"; otherwise "no detectable difference".
  RA is descriptive. The verdict needs both paired seeds; with one it is printed as interim.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]
from tools.nopart import table as T  # noqa: E402

SEEDS = (3407, 3408)
MARGIN = T.MARGIN


def verdict(d):
    mean = sum(d) / len(d)
    if all(x > 0 for x in d) and mean >= MARGIN:
        return "p_i 有用（去掉后变差）"
    if all(x < 0 for x in d) and mean <= -MARGIN:
        return "去掉后变好"
    return "无可检出差异"


def table(out=None):
    T.SEEDS = SEEDS                                  # T.cells reads the module's SEEDS at call time
    full, np_ = T.cells(T.ROWS[0][1]), T.cells(T.ROWS[1][1])
    L = ["## 主干层面：闭环 E1，流式形式（zgz，去根对齐，mm；均值，括号里逐种子）", "",
         "| 变体 | " + " | ".join(c[2] for c in T.COLS) + " | RA-MPJPE | 种子 |", "|---|---|---|---|---|---|---|"]
    for (label, arm), v in zip(T.ROWS, (full, np_)):
        L.append(f"| {label} | " + " | ".join(T.fmt(v, k) for k in range(5)) + f" | {' / '.join(map(str, sorted(v))) or '–'} |")
    paired = sorted(set(full) & set(np_))
    L += ["", "### 同种子配对差（无部位损失 − 有部位损失；正 = 去掉后变差）", "",
          "| 指标 | 均值 | 逐种子 | 判读（写定规则，修订 2：两种子同号且均值 ≥ 0.5 mm） |", "|---|---|---|---|"]
    names = [c[2] for c in T.COLS] + ["RA-MPJPE（描述）"]
    for k, name in enumerate(names):
        d = [np_[s][k] - full[s][k] for s in paired]
        if not d:
            L.append(f"| {name} | – | – | – |")
            continue
        v = verdict(d) if k < 4 else "描述"
        if k < 4 and len(paired) < len(SEEDS):
            v = "中间结果：" + v
        L.append(f"| {name} | {statistics.mean(d):+.2f} | " + " / ".join(f"{x:+.2f}" for x in d)
                 + f"（{', '.join(map(str, paired))}） | {v} |")
    text = "\n".join(L) + "\n"
    rc = ROOT / "outputs/nopart/20261113/router_compare/table.md"
    if rc.exists():
        text = rc.read_text() + "\n" + text
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text)
    print(text)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out")
    a = p.parse_args(argv)
    table(a.out)


if __name__ == "__main__":
    main()
