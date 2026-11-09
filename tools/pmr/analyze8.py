#!/usr/bin/env python3
"""PMR stage S3c report for the final form (docs/EXPERIMENT_RECORDS.md#pmr-s3c-prereg-20261109; descriptive, no rule):
CPU only, read-only, written once after every S3c reading.

    python -m tools.pmr.analyze8 s3c --arm X     -> analysis/s3c_<X>.md, decisions/s3c_<X>.json

  cost   X / S43 executor multiples in the same block under the same flags (parameters incl. the router, executed MACs incl.
         MANO FK, row-form latency, end-to-end p50 / p99), S43 dense and B2 beside, X's fast row when it exists
  async  each form's receipt status (plain / causal / streaming) for X_rol5 s3407
  forms  E1 five of X_rol5 in the ordinary / causal / streaming input form, three-seed means
  erh    EvRealHands tier 3d (loop and h200: MPJPE-RA, MPVPE-RA, root / finger degrees) of X_rol5 beside S43, B2 and
         B2 + rol5 (causal form) from TPD's readings, three-seed means
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
from tools.pmr.common import OUT, S43_ENGINE, SEEDS, now, read, require, save  # noqa: E402

TPD_ERH = ROOT / "outputs/tpd/20261105/erh/readings"
ERH_KEYS = ("mpjpe_ra_mm", "mpvpe_ra_mm", "root_deg", "finger_deg")


def cost(name, impl, policy):
    p = OUT / "costs" / f"final__{name}__{impl}__{policy}" / "report.json"
    if not p.exists():
        return None
    r = read(p)
    return {"params": r["params_total"], "macs": r["macs_layers"]["median"] + r["macs_mano_fk"],
            "row_ms": r["latency_row"]["raw_ms"], "e2e_p50": r["latency_e2e_ms"]["p50"],
            "e2e_p99": r["latency_e2e_ms"]["p99"]}


def erh(path):
    p = path / "3d" / "report.json"
    return read(p)["overall"] if p.exists() else None


def s3c(arm):
    dst = OUT / "decisions" / f"s3c_{arm}.json"
    require(not dst.exists(), "written once: " + str(dst))
    name = f"{arm}_rol5"
    base = U.run_name(arm, 3407)
    tags = {s: R2.run_name(name, s) for s in SEEDS}
    out = {"arm": arm, "rollout": name}
    costs = {(n, i, p): cost(n, i, p) for n, i, p in (("s43_s3407", "exec", "default"), ("s43_s3407", "exec", "det"),
                                                       ("s43_s3407", "dense", "default"), ("b2_s3407", "fast", "default"),
                                                       (base, "fast", "default"), (base, "fast", "det"),
                                                       (base, "pmrfast", "default"))}
    out["cost"] = {"__".join(k): v for k, v in costs.items()}
    L = [f"## PMR S3c：最终形式 {name} 的只报项（描述；不进任何规则）", "",
         f"S43（引擎三种子）{' / '.join(fmt(x) for x in S43_ENGINE)}。", "",
         "### 成本（同一块、GPU 4 独占；X 对 S43 执行器的倍率，同一标志）", "",
         "| 行 | 标志 | 参数（含路由器） | MACs（含 MANO FK） | 主行时延 ms | 端到端 p50 / p99 ms |", "|---|---|---|---|---|---|"]
    labels = {"s43_s3407": "S43", "b2_s3407": "B2", base: arm}
    for (n, i, p), c in costs.items():
        if c:
            L.append(f"| {labels[n]}（{i}） | {p} | {c['params']:,} | {c['macs'] / 1e6:.1f} M | {fmt(c['row_ms'])} | "
                     f"{fmt(c['e2e_p50'])} / {fmt(c['e2e_p99'])} |")
    for p in ("default", "det"):
        x, s = costs[(base, "fast", p)], costs[("s43_s3407", "exec", p)]
        if x and s:
            m = {k: x[k] / s[k] for k in x}
            out.setdefault("multiples", {})[p] = m
            L.append(f"| **{arm} / S43 执行器** | {p} | {m['params']:.2f}× | {m['macs']:.2f}× | {m['row_ms']:.2f}× | "
                     f"{m['e2e_p50']:.2f}× / {m['e2e_p99']:.2f}× |")
    L += ["", "### 异步检验（种子 3407）", "", "| 形式 | 状态 |", "|---|---|"]
    for form in ("plain", "causal", "str"):
        p = OUT / "receipts" / f"async_{tags[3407]}__{form}.json"
        st = read(p).get("status", "?") if p.exists() else "缺"
        out.setdefault("async", {})[form] = st
        L.append(f"| {form} | {st} |")
    L += ["", "### 输入形式（零训练换入；E1 五项三种子均值）", "", "| 形式 | E1 五项 |", "|---|---|"]
    M = mech()
    for form, suffix in (("普通分包", ""), ("因果", "__causal"), ("流式", "__str")):
        try:
            five = [float(np.mean([M.row("pmr", tags[s] + suffix)["five"][i] for s in SEEDS])) for i in range(5)]
            out.setdefault("forms", {})[form] = five
            L.append(f"| {form} | {' / '.join(fmt(x) for x in five)} |")
        except (KeyError, FileNotFoundError, RuntimeError) as exc:
            L.append(f"| {form} | 缺（{type(exc).__name__}） |")
    L += ["", "### EvRealHands（tier 3d，三种子均值；loop / h200）", "",
          "| 网络 | MPJPE-RA | MPVPE-RA | 根 ° | 手指 ° |", "|---|---|---|---|---|"]
    rows = {"S43": [TPD_ERH / f"s43_s{s}" for s in SEEDS], "B2": [TPD_ERH / f"pfo2_fgs_s{s}" for s in SEEDS],
            "B2 + rol5（因果）": [TPD_ERH / f"tpd_rol5_s{s}__causal" for s in SEEDS],
            name: [OUT / "erh" / "readings" / tags[s] for s in SEEDS]}
    for label, paths in rows.items():
        reports = [erh(p) for p in paths]
        if not all(reports):
            L.append(f"| {label} | 缺 | | | |")
            continue
        cells = []
        for k in ERH_KEYS:
            loop = float(np.mean([r["loop"][k] for r in reports]))
            h200 = float(np.mean([r["h200"][k] for r in reports]))
            cells.append(f"{fmt(loop)} / {fmt(h200)}")
            out.setdefault("erh", {}).setdefault(label, {})[k] = [loop, h200]
        L.append(f"| {label} | {' | '.join(cells)} |")
    text = "\n".join(L) + "\n"
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    (OUT / "analysis" / f"s3c_{arm}.md").write_text(text)
    save(dst, {**out, "written_at_utc": now()}, exclusive=True)
    print(text)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("s3c")
    s.add_argument("--arm", required=True)
    a = p.parse_args(argv)
    s3c(a.arm)


if __name__ == "__main__":
    main()
