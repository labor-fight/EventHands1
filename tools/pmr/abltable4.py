#!/usr/bin/env python3
"""ABL table under the paper's variant names, every row exactly as trained: the rol5 (or rol1) model, closed-loop E1 in the
streaming form, no switching of forms (user 2026-11-11: "这些数据你不用微调，你训出来什么结果就是什么结果"). CPU only.

    python -m tools.pmr.abltable4 table [--out outputs/pmr/20261109/analysis/abl_table_paper.md]

  table 1  mean over the seeds each variant has (per-seed values in parentheses)
  table 2  variant minus the full model on the same seeds (paired), per seed
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

import numpy as np  # noqa: E402

from tools.pmr import abltable as T  # noqa: E402

FULL = ("EvHandTrack（完整模型）", "d1r_rol5")
ROWS = [("Projection-based association（几何邻近 / 投影关联替代 ESMNet）", "abgeo_rol5"),
        ("w/o geometric residuals（去事件—前态几何残差）", "abnr_rol5"),
        ("w/o part propagation（运动学树传播的消息全部置零）", "d1n_rol5"),
        ("Global MLP update（部位特征拼接 + MLP 替代图传播与局部读出）", "d1m_rol5"),
        ("State-independent gating（前态几何门控 → 静态边类型门控）", "d1g0_rol5"),
        ("w/o hand–part communication（取消整手节点与部位节点的通信）", "d1h0_rol5"),
        ("Single-step fine-tuning（K = 1 替代 K = 5 闭环展开）", "d1r_rol1")]
COLS = "| 变体 | MPJPE（Local） | MPVPE（Local） | MPJPE（Global） | MPVPE（Global） | 种子 |"


def table(out):
    full = T.cells(FULL[1])
    L = ["## 消融表（按训练结果，闭环 E1，流式形式；zgz，去根对齐，mm）", "", COLS, "|---|---|---|---|---|---|"]
    for label, name in [FULL] + ROWS:
        v = T.cells(name)
        L.append(f"| {label} | {T.fmt(v, 0)} | {T.fmt(v, 1)} | {T.fmt(v, 2)} | {T.fmt(v, 3)} | "
                 f"{' / '.join(str(s) for s in sorted(v))} |")
    L += ["", "## 同种子配对差（变体 − 完整模型；均值，括号里逐种子）", "", COLS.replace("变体 |", "变体 |", 1),
          "|---|---|---|---|---|---|"]
    for label, name in ROWS:
        v = T.cells(name)
        d = {s: tuple(a - b for a, b in zip(v[s], full[s])) for s in v if s in full}
        cell = lambda k: f"{np.mean([x[k] for x in d.values()]):+.2f}（" + " / ".join(f"{d[s][k]:+.2f}" for s in sorted(d)) + "）"
        L.append(f"| {label} | {cell(0)} | {cell(1)} | {cell(2)} | {cell(3)} | {' / '.join(str(s) for s in sorted(d))} |")
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
