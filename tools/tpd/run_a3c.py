#!/usr/bin/env python3
"""TPD ROL-c readings under amendment 3 (docs/EXPERIMENT_RECORDS.md#tpd-amend3-20261105): tools/tpd/run_rolc.py's verbs
(accept / read / causal / table), the same calls, inside tools/tpd/ns_amend3.py (which registers seed 3409 itself, so
run_rolc's import-time namespace call is not repeated).

    python -m tools.tpd.run_a3c accept --seed 3409 | read --seed 3409 --task T | causal --seed S --task T | table
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.tpd import ns_amend3  # noqa: E402

ns_amend3.enter()
from tools.tpd import run_rol as RR  # noqa: E402
from tools.tpd.common import OUT, PFO2_OUT, S43_ENGINE, now, require, save  # noqa: E402

SEEDS = (3407, 3408, 3409)


def causal(seed, task):
    from semkine import tpd_causal as TC
    from tools.pfo import engine as E
    tag, sp = RR.arm_spec("rol5", seed)
    RR.prepare(sp)
    TC.install()
    TC.engine_hooks()
    E.OUT = OUT
    out = E.run(tag + "__causal", task, None, None, (), str(sp))
    save(Path(out) / "causal.json", {"causal": True, "amendment": ns_amend3.AMENDMENT, "written_utc": now()},
         exclusive=True)


def table():
    from tools.evf import mech as M
    M.ROOTS.update(tpd=OUT, pfo2=PFO2_OUT)
    rows = {}
    for s in SEEDS:
        rows[f"B2 {s}"] = M.row("pfo2", f"pfo2_fgs_s{s}")
        rows[f"rol5 {s}"] = M.row("tpd", f"tpd_rol5_s{s}")
        rows[f"rol5 {s} 因果形式"] = M.row("tpd", f"tpd_rol5_s{s}__causal")
    L = ["## TPD ROL-c：B2 + rol5 微调的三种子行（描述）", "", f"S43（引擎三种子）{' / '.join(f'{x:.2f}' for x in S43_ENGINE)}。", "",
         "| 行 | E1 五项 | E3 | 放大 |", "|---|---|---|---|"]
    for k, r in rows.items():
        L.append(f"| {k} | {' / '.join(f'{x:.2f}' for x in r['five'])} | "
                 + (f"{r['e3']:.2f} | {r['amp'][0]:.2f} |" if "e3" in r and "amp" in r else "– | – |"))
    for name, key in (("B2", "B2 {}"), ("rol5", "rol5 {}"), ("rol5 · 因果形式", "rol5 {} 因果形式")):
        five = [sum(rows[key.format(s)]["five"][i] for s in SEEDS) / 3 for i in range(5)]
        L.append(f"| **{name} 三种子均值** | {' / '.join(f'{x:.2f}' for x in five)} | | |")
    text = "\n".join(L) + "\n"
    (OUT / "analysis" / "rolc.md").write_text(text)
    save(OUT / "decisions" / "rolc_table.json", {"rows": rows, "amendment": ns_amend3.AMENDMENT, "written_utc": now()},
         exclusive=True)
    print(text)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("accept")
    s.add_argument("--seed", type=int, required=True, choices=(3409,))
    for verb in ("read", "causal"):
        s = sub.add_parser(verb)
        s.add_argument("--seed", type=int, required=True, choices=SEEDS)
        s.add_argument("--task", required=True, choices=RR.TASKS)
    sub.add_parser("table")
    a = p.parse_args(argv)
    if a.verb == "accept":
        return RR.accept("rol5", a.seed, "formal")
    if a.verb == "read":
        require(a.seed == 3409, "seed 3409 only")
        tag, sp = RR.arm_spec("rol5", a.seed)
        return RR.engine(tag, a.task, sp)
    if a.verb == "causal":
        return causal(a.seed, a.task)
    return table()


if __name__ == "__main__":
    main()
