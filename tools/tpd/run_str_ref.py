#!/usr/bin/env python3
"""TPD STR addendum (docs/EXPERIMENT_RECORDS.md#tpd-str-ref-20261106; description, no rule): rol5 seeds 3408 / 3409 read in
the streaming form with zero training, beside the registered seed 3407 (`run_str ref`).

    python -m tools.tpd.run_str_ref read --seed S --task T    the spec file of rol5 seed S's zgz readings (checkpoint
                                                              checked against its acceptance); tag tpd_rol5_s<S>__str
    python -m tools.tpd.run_str_ref table                     analysis/str_ref.md, decisions/str_ref_table.json (once)
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.tpd import run_str as RS  # noqa: E402
from tools.tpd.common import OUT, S43_ENGINE, now, read, require, save, sha  # noqa: E402

SEEDS = (3407, 3408, 3409)


def reading(seed, task):
    require(seed in (3408, 3409), "seed 3407 is the registered `run_str ref`")
    spec = OUT / "specs" / f"tpd_rol5_s{seed}.json"
    sp = read(spec)
    receipt = read(OUT / "receipts" / f"tpd_rol5_s{seed}.json")
    require(sha(sp["checkpoint"]) == receipt["checkpoint_sha256_by_step"]["1500"], "rol5 checkpoint changed")
    RS.engine(f"tpd_rol5_s{seed}__str", task, spec)


def table():
    out = OUT / "decisions" / "str_ref_table.json"
    require(not out.exists(), "written once")
    M = RS.mech()
    forms = {"部署形式（不异步）": "", "零训练换成因果形式": "__causal", "零训练换成流式形式": "__str"}
    rows = {f: {s: M.row("tpd", f"tpd_rol5_s{s}{suf}") for s in SEEDS} for f, suf in forms.items()}
    names = ("MPJPE-L", "MPJPE-G", "MPVPE-L", "MPVPE-G", "RA")
    L = ["## TPD STR 补充：rol5 三种子换成流式形式（零训练；描述）", "",
         "| 行 | " + " | ".join(names) + " | E3 | 放大 |", "|---|---|---|---|---|---|---|---|"]
    means = {}
    for f, by in rows.items():
        for s in SEEDS:
            r = by[s]
            L.append(f"| rol5 {s} {f} | " + " | ".join(f"{x:.2f}" for x in r["five"]) + f" | {r['e3']:.2f} | {r['amp'][0]:.2f} |")
        means[f] = [sum(by[s]["five"][i] for s in SEEDS) / 3 for i in range(5)]
    for f, m in means.items():
        L.append(f"| **rol5 三种子均值，{f}** | " + " | ".join(f"{x:.2f}" for x in m) + " | | |")
    L.append("| S43（引擎三种子） | " + " | ".join(f"{x:.2f}" for x in S43_ENGINE) + " | | |")
    text = "\n".join(L) + "\n"
    (OUT / "analysis" / "str_ref.md").write_text(text)
    save(out, {"rows": rows, "means": means, "description_only": True, "written_at_utc": now()}, exclusive=True)
    print(text)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("read")
    s.add_argument("--seed", type=int, required=True, choices=(3408, 3409))
    s.add_argument("--task", required=True, choices=("e1", "e3"))
    sub.add_parser("table")
    a = p.parse_args(argv)
    return reading(a.seed, a.task) if a.verb == "read" else table()


if __name__ == "__main__":
    main()
