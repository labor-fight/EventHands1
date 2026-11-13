#!/usr/bin/env python3
"""ABL final table (docs/EXPERIMENT_RECORDS.md#abl-results-20261111): tools/pmr/abltable2.py's table unchanged, plus two
descriptive step-1 sections written after the step-1 rollout collapse:
  - before the fine-tune: the two from-scratch bases (EvHandTrack's D1 R and (a)'s ray-association trunk, same recipe and
    data stream, 6000 steps, no rollout) read in the streaming form (tools/pmr/run_forms7.py), three seeds;
  - the association during the rollout fine-tune: the training log's `train_xyz_valid` (fraction of event-node records with
    a valid association) and `train_joint_error_mm_k0`, by step range, per seed.
CPU only.

    python -m tools.pmr.abltable3 table [--out outputs/pmr/20261109/analysis/abl_table.md]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

import numpy as np  # noqa: E402

from tools.pmr import abltable as T  # noqa: E402
from tools.pmr import abltable2 as T2  # noqa: E402
from tools.pmr.common import OUT  # noqa: E402

BASES = [("EvHandTrack 的从头基座（D1 R + r_s2d）", "d1r"), ("(a) 的从头基座（几何邻近关联）", "abgeo")]
ROLL = [("完整 EvHandTrack（rol5）", "d1r_rol5"), ("第 1 步 (a)（rol5）", "abgeo_rol5")]
RANGES = ((0, 100), (100, 300), (300, 600), (600, 1500))


def scalars(run, tag):
    import glob
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    d = sorted(glob.glob(str(OUT / "runs" / run / "logs" / "*")))[-1]
    ea = EventAccumulator(d, size_guidance={"scalars": 0})
    ea.Reload()
    return [(e.step, e.value) for e in ea.Scalars(tag)]


def rollout_log():
    head = " | ".join(f"第 {a}–{b} 步" for a, b in RANGES)
    L = ["| 训练（K = 5 闭环展开微调） | 种子 | 关联有效比例：" + head + " | 首包关节误差 mm：" + head + " |",
         "|---|---|" + "---|" * (2 * len(RANGES))]
    for label, name in ROLL:
        for s in T.SEEDS:
            run = f"pmr_{name}_s{s}"
            v, e = scalars(run, "train_xyz_valid"), scalars(run, "train_joint_error_mm_k0")
            m = lambda x, a, b: np.mean([y for st, y in x if a <= st < b])
            L.append(f"| {label} | {s} | " + " | ".join(f"{m(v, a, b):.2f}" for a, b in RANGES) + " | "
                     + " | ".join(f"{m(e, a, b):.1f}" for a, b in RANGES) + " |")
    return L


def table(out):
    L = T2.rows_table(T2.CORE, "## 消融表：每一步拿掉一次（闭环 E1，流式形式；zgz，去根对齐，mm）")
    L += ["", "### 第 1 步 · 微调前对照（从头 6000 步、未做闭环微调，同配方同数据流；闭环 E1，流式形式；描述）", ""]
    L += T2.rows_table(BASES, "")[2:]
    L += ["", "### 第 1 步 · 闭环展开微调中的关联（训练日志；描述）", ""]
    L += rollout_log()
    J = T.Joints()
    L += ["", f"### 观测不足部位的重建（zgz_local；每包该部位标注节点 < {T.UNDER} 记为观测不足；关节误差 mm）", ""]
    L += T.under(T2.CORE, J)
    L += ["", "### 长时误差曲线（MPJPE-RA，按距上次初始化的时间分箱，mm）", ""]
    L += T.curve(2)
    L += [""] + T2.rows_table(T2.SUPP, "## 补充：步内子模块的变体（不进主张）")
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
