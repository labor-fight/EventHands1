#!/usr/bin/env python3
"""PMR stage S3a written verdict (docs/EXPERIMENT_RECORDS.md#pmr-prereg-20261109 and the design's S3a section),
computed once after every reading; CPU only, read-only on the readings (this study's, TPD's rol5 and S43's engine row).

    python -m tools.pmr.analyze3 s3a --design d1
        -> analysis/s3a_<design>.md, decisions/s3a_<design>.json, main rows
           outputs/semkine/pmr_<design>r_rol5_3seed_main_row.json and pmr_<design>{n,m}_rol5_s3407_main_row.json

Written rule: DONE when the three-seed means of R + rol5's E1 five (MPJPE-L, MPJPE-G, MPVPE-L, MPVPE-G, RA; PFO's engine,
deployed router, ordinary packet form) are all below S43's engine values 12.81 / 8.02 / 10.77 / 6.64 / 10.24; else
NOT_DONE. Descriptive: per-seed rows, R-noprop + rol5 and R-mlp + rol5 (seed 3407) against R + rol5 (3407), B2 + rol5
(TPD, three seeds).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.pmr import arms_rol as R  # noqa: E402
from tools.pmr.analyze import KEYS, fmt, mech  # noqa: E402
from tools.pmr.common import OUT, S43_ENGINE, SEEDS, now, read, require, save  # noqa: E402

SEMKINE = ROOT / "outputs/semkine"
TPD_OUT = ROOT / "outputs/tpd/20261105"


def done(mean5):
    """mean5: the three-seed means of the five (pure)."""
    return "DONE" if all(m < s for m, s in zip(mean5, S43_ENGINE)) else "NOT_DONE"


def main_row(name, seeds, label, cost_tag):
    per = {}
    for i, s in enumerate(seeds):
        tag = R.run_name(name, s)
        r = read(OUT / "readings" / tag / "e1" / "report.json")
        q = r["per_sequence"]
        per[str(i)] = {"reading": f"pmr:{tag}", "checkpoint": r["identity"]["checkpoint"],
                       "local": {k: q["zgz_local"][k] for k in KEYS}, "global": {k: q["zgz_global"][k] for k in KEYS},
                       "overall": {k: r["overall"][k] for k in KEYS}}
    m = {b: {k: float(np.mean([per[i][b][k] for i in per])) for k in KEYS} for b in ("local", "global", "overall")}
    cp = OUT / "costs" / f"{cost_tag}__fast__default" / "report.json"
    require(cp.exists(), "the architecture's cost reading (default flags): " + str(cp))
    c = read(cp)
    out = {"arm": f"pmr_{name}" + ("_3seed" if len(seeds) == 3 else f"_s{seeds[0]}"), "label": label, "split": "val_core",
           "n_seeds": len(seeds), "two_seed_mean": m, "per_seed": per,
           "latency_ms_raw": c["latency_row"]["raw_ms"], "latency_anchor_ms": c["latency_row"]["anchor_ms"],
           "latency_ms_scaled_full1p75": c["latency_row"]["scaled_ms"], "macs_forward_packet": c["macs_layers"]["median"],
           "params_total": c["params_total"], "latency_e2e_ms": c["latency_e2e_ms"], "cost_reading": str(cp.parent),
           "written_utc": now(),
           "note": "PMR S3a row (PFO's engine E1, deterministic flags, ordinary packet form); a rollout fine-tune has its base "
                   "arm's architecture, so the cost is the base arm's seed-3407 reading (default flags)"}
    p = SEMKINE / f"{out['arm']}_main_row.json"
    save(p, out)
    return p


def s3a(design):
    dst = OUT / "decisions" / f"s3a_{design}.json"
    require(not dst.exists(), "written once: " + str(dst))
    M = mech()
    M.ROOTS["tpd"] = TPD_OUT
    r5, n5, m5 = f"{design}r_rol5", f"{design}n_rol5", f"{design}m_rol5"
    rows = {f"r_rol5_s{s}": M.row("pmr", R.run_name(r5, s)) for s in SEEDS}
    rows["n_rol5_s3407"] = M.row("pmr", R.run_name(n5, 3407))
    rows["m_rol5_s3407"] = M.row("pmr", R.run_name(m5, 3407))
    for s in SEEDS:
        rows[f"b2_rol5_s{s}"] = M.row("tpd", f"tpd_rol5_s{s}")
    mean = [float(np.mean([rows[f"r_rol5_s{s}"]["five"][i] for s in SEEDS])) for i in range(5)]
    b2m = [float(np.mean([rows[f"b2_rol5_s{s}"]["five"][i] for s in SEEDS])) for i in range(5)] \
        if all("five" in rows[f"b2_rol5_s{s}"] for s in SEEDS) else None
    verdict = done(mean)
    L = [f"## PMR {design.upper()} S3a：rol5 流水线（普通分包形式；PFO 引擎；写定规则）", "",
         f"S43（引擎三种子）{' / '.join(fmt(x) for x in S43_ENGINE)}。", "",
         "| 行 | E1 五项 | E3 | 放大 | local R15 / F15 | C |", "|---|---|---|---|---|---|"]
    for k, r in rows.items():
        five = " / ".join(fmt(x) for x in r["five"]) if "five" in r else "–"
        ret = " / ".join(fmt(r["retention"][q]) for q in ("R15", "F15")) if "retention" in r else "–"
        L.append(f"| {k} | {five} | {fmt(r.get('e3'))} | {fmt(r['amp'][0]) if 'amp' in r else '–'} | {ret} | "
                 f"{fmt(r.get('C'), 3)} |")
    L += ["", f"R + rol5 三种子均值：{' / '.join(fmt(x) for x in mean)}" +
          (f"；B2 + rol5 三种子均值：{' / '.join(fmt(x) for x in b2m)}" if b2m else ""),
          f"ROL 之后去掉 P（3407）：RA {fmt(rows['n_rol5_s3407']['five'][4] - rows['r_rol5_s3407']['five'][4])}；"
          f"拼接 + MLP（3407）：RA {fmt(rows['m_rol5_s3407']['five'][4] - rows['r_rol5_s3407']['five'][4])}", "",
          f"**{verdict}**（五项均值全部低于 S43）"]
    text = "\n".join(L) + "\n"
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    (OUT / "analysis" / f"s3a_{design}.md").write_text(text)
    rows_written = [str(main_row(r5, SEEDS, f"PMR {design.upper()} R + rol5（三种子，普通分包形式）", A_run(design, "r"))),
                    str(main_row(n5, (3407,), f"PMR {design.upper()} R-noprop + rol5（种子 3407）", A_run(design, "n"))),
                    str(main_row(m5, (3407,), f"PMR {design.upper()} R-mlp + rol5（种子 3407）", A_run(design, "m")))]
    save(dst, {"design": design, "rows": rows, "mean_r_rol5": mean, "mean_b2_rol5": b2m, "verdict": verdict,
               "s43_engine": list(S43_ENGINE), "main_rows": rows_written, "written_at_utc": now()}, exclusive=True)
    print(text)


def A_run(design, code):
    from tools.pmr import arms as A
    return A.run_name(f"{design}{code}", 3407)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("s3a")
    s.add_argument("--design", required=True)
    a = p.parse_args(argv)
    s3a(a.design)


if __name__ == "__main__":
    main()
