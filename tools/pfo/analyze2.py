#!/usr/bin/env python3
"""PFO phase-2 readings -> tables (docs/EXPERIMENT_RECORDS.md#pfo-p2a-prereg-20261103, #pfo-p2r-prereg-20261103,
#pfo-p2b-prereg-20261103); CPU only, read-only on the readings.

    python -m tools.pfo.analyze2 arms [--out FILE]        every trained arm (2a and 2b): E1 five columns per seed and seed
                                                          means, E1@5000/5500 RA, E3 RA, tl_e1, C, local R15/F15 retention,
                                                          root-only / fingers-only E1 RA
    python -m tools.pfo.analyze2 routers [--out FILE]     router quality readings and zero-training swaps
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.pfo.analyze import R, decomp, f1_c, fmt, rep, row5, tl_mean  # noqa: E402
from tools.pfo.common import OUT, ZGZ, read  # noqa: E402

SEEDS = (3407, 3408, 3409)


def arm_names():
    from tools.pfo.arms import ARMS
    from tools.pfo.arms2 import ARMS2
    return list(ARMS) + list(ARMS2)


def retention(tag):
    r = rep(tag, "pert")
    if r is None:
        return None
    b = r["per_sequence"]["zgz_local"]
    g = r["per_sequence"]["zgz_global"]
    return {"R15": (b["R15"]["rho_median"], g["R15"]["rho_median"]), "F15": (b["F15"]["rho_median"], g["F15"]["rho_median"]),
            "T30": (b["T30"]["rho_median"], g["T30"]["rho_median"])}


def arms():
    lines = ["## 第 2 阶段训练臂（第 6000 步）", "",
             "| 臂 | 种子 | E1 MPJPE-L / G，MPVPE-L / G，RA | E1@5000 / 5500 RA | E3 RA | tl_e1 中位 | C | ρ local R15 / F15 | 只根 / 只手指 RA（L/G） |",
             "|---|---|---|---|---|---|---|---|---|"]
    js = {}
    from tools.pfo.arms import run_name
    for arm in arm_names():
        per = {}
        for seed in SEEDS:
            tag = run_name(arm, seed)
            a = row5(rep(tag, "e1"))
            if a is None:
                continue
            e3 = row5(rep(tag, "e3"))
            late = [row5(rep(f"{tag}@{s}", "e1")) for s in (5000, 5500)]
            tl = tl_mean(rep(tag, "tl_e1"))
            f = f1_c(tag)
            ret = retention(tag)
            dec = decomp(tags=(tag,))[1].get(tag)
            per[seed] = {"e1": a, "e3": e3, "late": late, "tl": tl, "C": f and f["pooled_C"], "ret": ret, "decomp": dec}
            rcell = "–" if ret is None else f"{ret['R15'][0]:.2f} / {ret['F15'][0]:.2f}"
            dcell = "–" if dec is None else (f"{dec['zgz_local'][2]:.2f}/{dec['zgz_global'][2]:.2f} ; "
                                             f"{dec['zgz_local'][1]:.2f}/{dec['zgz_global'][1]:.2f}")
            lines.append(f"| {arm} | {seed} | {' / '.join(fmt(v) for v in a)} | "
                         f"{' / '.join(fmt(l and l[4]) for l in late)} | {fmt(e3 and e3[4])} | {fmt(tl and tl[0])} | "
                         f"{fmt(f and f['pooled_C'], 3)} | {rcell} | {dcell} |")
        if per:
            m = np.mean([v["e1"] for v in per.values()], 0)
            lines.append(f"| {arm} | 均值（{len(per)}） | {' / '.join(fmt(v) for v in m)} | | | | | | |")
            js[arm] = {"per_seed": {str(k): v for k, v in per.items()}, "mean": m.tolist(), "n": len(per)}
    s43 = [row5(rep(f"s43_s{s}", "e1")) for s in SEEDS]
    if all(s43):
        m = np.mean(s43, 0)
        lines.append(f"| S43（引擎重算） | 均值（3） | {' / '.join(fmt(v) for v in m)} | | | | | | |")
        js["s43"] = {"mean": m.tolist(), "per_seed": {str(s): v for s, v in zip(SEEDS, s43)}}
    return lines, js


def routers():
    lines = ["## 路由器质量（XAG 伪标签，S45 节点，每 4 个端点）", "",
             "| 路由器 | 参数 | zgz_local 部位 / 点误差中位 / 手外入网格 | zgz_global 部位 / 点误差 / 手外 | ylf_local 部位 / 点误差 | lyq_local 部位 / 点误差 |",
             "|---|---|---|---|---|---|"]
    js = {"quality": {}, "swaps": {}}
    for p in sorted((OUT / "routers_read").glob("*.json")):
        r = read(p)
        s = r["per_sequence"]
        js["quality"][p.stem] = r

        def c(seq, full=True):
            v = s[seq]
            base = f"{v['part_acc']:.3f} / {v['point_err_mm_median']:.1f}"
            return base + (f" / {v['offhand_in_mesh']:.3f}" if full else "")
        lines.append(f"| {p.stem} | {r['params']:,} | {c('zgz_local')} | {c('zgz_global')} | {c('ylf_local', False)} | "
                     f"{c('lyq_local', False)} |")
    lines += ["", "## 零训练换路由器（E1 local / global / RA；E3 RA）", "",
              "| 主干 + 路由器 | E1 local | E1 global | E1 RA | E3 RA |", "|---|---|---|---|---|"]
    for d in sorted(R.glob("*+*")):
        a, b = row5(rep(d.name, "e1")), row5(rep(d.name, "e3"))
        js["swaps"][d.name] = {"e1": a, "e3": b}
        lines.append(f"| {d.name} | {fmt(a and a[0])} | {fmt(a and a[1])} | {fmt(a and a[4])} | {fmt(b and b[4])} |")
    return lines, js


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=("arms", "routers"))
    p.add_argument("--out")
    a = p.parse_args(argv)
    lines, js = {"arms": arms, "routers": routers}[a.what]()
    text = "\n".join(lines) + "\n"
    print(text)
    if a.out:
        Path(a.out).write_text(text)
        Path(a.out).with_suffix(".json").write_text(json.dumps(js, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
