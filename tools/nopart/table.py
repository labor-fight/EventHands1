#!/usr/bin/env python3
"""ESMNet without the part loss, trunk table and the written rule (docs/EXPERIMENT_RECORDS.md#nopart-prereg-20261113).
CPU only. Reads RFSU's simulated streaming E1 reports (closed loop, zgz, root-aligned, mm).

    python tools/nopart/table.py [--out outputs/nopart/20261113/table.md]

Rule (written before any reading): per seed, d = rfsrnp_rol5 - rfsr_rol5 on MPJPE / MPVPE of zgz_local and zgz_global.
  per metric: all available seeds d > 0 and mean d >= +0.5 mm -> "p_i helps" (dropping the part loss is worse);
              all seeds d < 0 and mean d <= -0.5 mm -> "dropping helps"; otherwise "no detectable difference".
  The verdict needs the three paired seeds; with fewer it is printed as interim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
READ = ROOT / "outputs/rfsu/20261113/sim/readings"
SEEDS = (3407, 3408, 3409)
ROWS = [("EvHandTrack（论文主模型：r_s2dx2 有部位损失，先路由，rol5）", "rfsr_rol5"),
        ("w/o part prediction（r_s2dx2np：训练去掉部位损失，只用 c_i 解码）", "rfsrnp_rol5")]
COLS = [("zgz_local", "mpjpe_ra_mm", "MPJPE（Local）"), ("zgz_local", "mpvpe_ra_mm", "MPVPE（Local）"),
        ("zgz_global", "mpjpe_ra_mm", "MPJPE（Global）"), ("zgz_global", "mpvpe_ra_mm", "MPVPE（Global）")]
MARGIN = 0.5


def cells(arm):
    out = {}
    for s in SEEDS:
        p = READ / f"pmr_{arm}_s{s}__str" / "e1" / "report.json"
        if p.exists():
            r = json.loads(p.read_text())
            if r.get("status", "COMPLETE") == "COMPLETE":
                q = r["per_sequence"]
                out[s] = [q[a][b] for a, b, _ in COLS] + [r["overall"]["mpjpe_ra_mm"]]
    return out


def fmt(v, k):
    if not v:
        return "–"
    import statistics
    m = statistics.mean(x[k] for x in v.values())
    return f"{m:.2f}" if len(v) == 1 else f"{m:.2f}（" + " / ".join(f"{v[s][k]:.2f}" for s in sorted(v)) + "）"


def verdict(d):
    if not d:
        return "–"
    mean = sum(d) / len(d)
    if all(x > 0 for x in d) and mean >= MARGIN:
        return "p_i 有用（去掉后变差）"
    if all(x < 0 for x in d) and mean <= -MARGIN:
        return "去掉后变好"
    return "无可检出差异"


def table(out=None):
    full, np_ = cells(ROWS[0][1]), cells(ROWS[1][1])
    L = ["## 主干层面：闭环 E1，流式形式（zgz，去根对齐，mm；均值，括号里逐种子）", "",
         "| 变体 | " + " | ".join(c[2] for c in COLS) + " | RA-MPJPE | 种子 |", "|---|---|---|---|---|---|---|"]
    for (label, arm), v in zip(ROWS, (full, np_)):
        L.append(f"| {label} | " + " | ".join(fmt(v, k) for k in range(5)) + f" | {' / '.join(map(str, sorted(v))) or '–'} |")
    paired = sorted(set(full) & set(np_))
    L += ["", "### 同种子配对差（无部位损失 − 有部位损失；正 = 去掉后变差）", "",
          "| 指标 | 均值 | 逐种子 | 判读（写定规则：三种子同号且均值 ≥ 0.5 mm） |", "|---|---|---|---|"]
    names = [c[2] for c in COLS] + ["RA-MPJPE（描述）"]
    for k, name in enumerate(names):
        d = [np_[s][k] - full[s][k] for s in paired]
        if not d:
            L.append(f"| {name} | – | – | – |")
            continue
        v = verdict(d) if k < 4 else "描述"
        if k < 4 and len(paired) < 3:
            v = "中间结果：" + v
        L.append(f"| {name} | {sum(d) / len(d):+.2f} | " + " / ".join(f"{x:+.2f}" for x in d) + f"（{', '.join(map(str, paired))}） | {v} |")
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
