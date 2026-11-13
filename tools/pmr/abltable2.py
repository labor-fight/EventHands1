#!/usr/bin/env python3
"""ABL reduced table (docs/EXPERIMENT_RECORDS.md#abl-amend2-20261111): one ablation per step, plus the finished
sub-module variants as supplementary rows. CPU only; tools/pmr/abltable.py's readouts unchanged.

    python -m tools.pmr.abltable2 table [--out outputs/pmr/20261109/analysis/abl_table.md]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import abltable as T  # noqa: E402

CORE = [("完整 EvHandTrack", "d1r_rol5"),
        ("第 1 步 · 去表面对应：几何邻近关联替代 ESMNet", "abgeo_rol5"),
        ("第 2 步 · 去部位协同传播：传播 P 的消息全部置零", "d1n_rol5"),
        ("第 3 步 · 去闭环训练：单步训练替代多步闭环展开", "d1r_rol1")]
SUPP = [("补充 (b) 去事件—前态几何残差", "abnr_rol5"),
        ("补充 (c) 部位特征拼接 MLP 替代图传播与局部读出", "d1m_rol5"),
        ("补充 (d) 前态几何门控 → 静态边类型门控", "d1g0_rol5"),
        ("补充 (e) 取消整手节点与部位节点的通信", "d1h0_rol5")]


def rows_table(rows, title):
    L = [title, "", "| 变体 | MPJPE（Local） | MPVPE（Local） | MPJPE（Global） | MPVPE（Global） | 种子 |",
         "|---|---|---|---|---|---|"]
    for label, name in rows:
        v = T.cells(name)
        L.append(f"| {label} | {T.fmt(v, 0)} | {T.fmt(v, 1)} | {T.fmt(v, 2)} | {T.fmt(v, 3)} | {len(v)} |")
    return L


def table(out):
    L = rows_table(CORE, "## 消融表：每一步拿掉一次（闭环 E1，流式形式；zgz，去根对齐，mm）")
    J = T.Joints()
    L += ["", f"### 观测不足部位的重建（zgz_local；每包该部位标注节点 < {T.UNDER} 记为观测不足；关节误差 mm）", ""]
    L += T.under(CORE, J)
    L += ["", "### 长时误差曲线（MPJPE-RA，按距上次初始化的时间分箱，mm）", ""]
    L += T.curve(2)
    L += [""] + rows_table(SUPP, "## 补充：步内子模块的变体（不进主张）")
    text = "\n".join(L) + "\n"
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text)
    print(text)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("verb", choices=("table",))
    p.add_argument("--out")
    a = p.parse_args(argv)
    table(a.out)


if __name__ == "__main__":
    main()
