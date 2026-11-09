#!/usr/bin/env python3
"""PMR stage S2 written confirmation (docs/EXPERIMENT_RECORDS.md#pmr-prereg-20261109, "过筛之后"; the design's S2
section), computed once after every reading; CPU only, read-only on the readings.

    python -m tools.pmr.analyze2 s2 --design d1
        -> analysis/s2_<design>.md, decisions/s2_<design>.json, main rows outputs/semkine/pmr_<design>{r,n}_3seed_main_row.json

Rows (tools/evf/mech.py `row`): R and R-noprop at seeds 3407 / 3408 / 3409 (from scratch, no ROL), their three-seed
means, R-mlp 3407 and B2 at the three seeds (descriptive).
Written rule: MODULE_ESTABLISHED when, for every seed s, RA(R-noprop, s) - RA(R, s) > 0 (the three seeds' gaps have
the screen's sign); else MODULE_NOT_ESTABLISHED. The per-seed gaps, their mean and the local / global differences are
reported (descriptive).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.pmr import arms as A  # noqa: E402
from tools.pmr.analyze import KEYS, fmt, mech  # noqa: E402
from tools.pmr.common import OUT, S43_ENGINE, SEEDS, now, read, require, save  # noqa: E402

SEMKINE = ROOT / "outputs/semkine"


def rule(gaps):
    """gaps: {seed: RA(noprop) - RA(R)} for the three seeds (pure)."""
    require(set(gaps) == set(SEEDS), "the three seeds")
    return "MODULE_ESTABLISHED" if all(g > 0 for g in gaps.values()) else "MODULE_NOT_ESTABLISHED"


def three_seed_row(arm, label):
    per, cost = {}, None
    for i, s in enumerate(SEEDS):
        tag = A.run_name(arm, s)
        r = read(OUT / "readings" / tag / "e1" / "report.json")
        q = r["per_sequence"]
        per[str(i)] = {"reading": f"pmr:{tag}", "checkpoint": r["identity"]["checkpoint"],
                       "local": {k: q["zgz_local"][k] for k in KEYS}, "global": {k: q["zgz_global"][k] for k in KEYS},
                       "overall": {k: r["overall"][k] for k in KEYS}}
    m = {b: {k: float(np.mean([per[i][b][k] for i in per])) for k in KEYS} for b in ("local", "global", "overall")}
    cp = OUT / "costs" / f"{A.run_name(arm, SEEDS[0])}__fast__default" / "report.json"
    require(cp.exists(), "the arm's cost reading (seed 3407, default flags)")
    c = read(cp)
    out = {"arm": f"pmr_{arm}_3seed", "label": label, "split": "val_core", "n_seeds": 3, "two_seed_mean": m,
           "per_seed": per, "latency_ms_raw": c["latency_row"]["raw_ms"], "latency_anchor_ms": c["latency_row"]["anchor_ms"],
           "latency_ms_scaled_full1p75": c["latency_row"]["scaled_ms"], "macs_forward_packet": c["macs_layers"]["median"],
           "params_total": c["params_total"], "latency_e2e_ms": c["latency_e2e_ms"], "cost_reading": str(cp.parent),
           "written_utc": now(), "note": "PMR S2 three-seed row (PFO's engine E1, deterministic flags; mean over seeds "
                                          "3407 / 3408 / 3409; cost = the seed-3407 reading, default flags)"}
    p = SEMKINE / f"pmr_{arm}_3seed_main_row.json"
    save(p, out)
    return p


def s2(design):
    require(design in A.DESIGNS, "unregistered design")
    dst = OUT / "decisions" / f"s2_{design}.json"
    require(not dst.exists(), "written once: " + str(dst))
    M = mech()
    r_arm, n_arm, m_arm = f"{design}r", f"{design}n", f"{design}m"
    rows = {}
    for s in SEEDS:
        rows[f"r_s{s}"] = M.row("pmr", A.run_name(r_arm, s))
        rows[f"n_s{s}"] = M.row("pmr", A.run_name(n_arm, s))
        rows[f"b2_s{s}"] = M.row("pfo2", f"pfo2_fgs_s{s}")
    rows["m_s3407"] = M.row("pmr", A.run_name(m_arm, 3407))
    gaps = {s: rows[f"n_s{s}"]["five"][4] - rows[f"r_s{s}"]["five"][4] for s in SEEDS}
    lg = {s: (rows[f"n_s{s}"]["five"][0] - rows[f"r_s{s}"]["five"][0], rows[f"n_s{s}"]["five"][1] - rows[f"r_s{s}"]["five"][1])
          for s in SEEDS}
    verdict = rule(gaps)
    mean = {k: [float(np.mean([rows[f"{k}_s{s}"]["five"][i] for s in SEEDS])) for i in range(5)] for k in ("r", "n", "b2")}
    L = [f"## PMR {design.upper()} S2 确认（从头 6000 步，无 ROL；三种子；PFO 引擎；写定规则）", "",
         f"S43（引擎三种子）{' / '.join(fmt(x) for x in S43_ENGINE)}。", "",
         "| 行 | E1 五项 | E3 | 放大 | local R15 / F15 | C |", "|---|---|---|---|---|---|"]
    for k, r in rows.items():
        ret = " / ".join(fmt(r["retention"][q]) for q in ("R15", "F15")) if "retention" in r else "–"
        L.append(f"| {k} | {' / '.join(fmt(x) for x in r['five'])} | {fmt(r.get('e3'))} | "
                 f"{fmt(r['amp'][0]) if 'amp' in r else '–'} | {ret} | {fmt(r.get('C'), 3)} |")
    L += ["", "| 三种子均值 | E1 五项 |", "|---|---|"] + [f"| {k} | {' / '.join(fmt(x) for x in v)} |" for k, v in mean.items()]
    L += ["", "去掉 P 的差距 RA(R-noprop) − RA(R)（local / global）：" + "；".join(
        f"{s} {fmt(gaps[s])}（{fmt(lg[s][0])} / {fmt(lg[s][1])}）" for s in SEEDS) + f"；均值 {fmt(np.mean(list(gaps.values())))}",
          "", f"**{verdict}**"]
    text = "\n".join(L) + "\n"
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    (OUT / "analysis" / f"s2_{design}.md").write_text(text)
    main_rows = [str(three_seed_row(r_arm, f"PMR {design.upper()} R：部位图传播 P + 逐关节局部读出（三种子，无 ROL）")),
                 str(three_seed_row(n_arm, f"PMR {design.upper()} R-noprop：去掉 P 的消息（三种子，无 ROL）"))]
    save(dst, {"design": design, "rows": rows, "gaps": {str(k): v for k, v in gaps.items()},
               "local_global": {str(k): v for k, v in lg.items()}, "means": mean, "verdict": verdict,
               "main_rows": main_rows, "written_at_utc": now()}, exclusive=True)
    print(text)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("s2")
    s.add_argument("--design", required=True)
    a = p.parse_args(argv)
    s2(a.design)


if __name__ == "__main__":
    main()
