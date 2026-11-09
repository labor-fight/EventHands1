#!/usr/bin/env python3
"""TPD stage ROL-c readings (docs/EXPERIMENT_RECORDS.md#tpd-rolc-prereg-20261105), under tools/tpd/ns_seed.py:

    python -m tools.tpd.run_rolc accept --seed 3409            rol5 seed 3409 training acceptance (run_rol `accept`)
    python -m tools.tpd.run_rolc read --seed S --task T        rol5 seed S (deployed form); tag tpd_rol5_s<S>
    python -m tools.tpd.run_rolc causal --seed S --task T      rol5 seed S read in the causal deployment form, zero
                                                               training (semkine/tpd_causal.py); tag tpd_rol5_s<S>__causal
    python -m tools.tpd.run_rolc table                         analysis/rolc.md (descriptive, three seeds, both forms)
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.tpd import ns_seed  # noqa: E402

ns_seed.enter()
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
    save(Path(out) / "causal.json", {"causal": True, "written_utc": now()}, exclusive=True)


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
                 f"{r['e3']:.2f} | {r['amp'][0]:.2f} |" if "e3" in r else f"| {k} | {' / '.join(f'{x:.2f}' for x in r['five'])} | – | – |")
    for name in ("B2", "rol5", "rol5 · 因果形式"):
        key = (lambda s: f"B2 {s}") if name == "B2" else (lambda s: f"rol5 {s}") if name == "rol5" else (lambda s: f"rol5 {s} 因果形式")
        five = [sum(rows[key(s)]["five"][i] for s in SEEDS) / 3 for i in range(5)]
        L.append(f"| **{name} 三种子均值** | {' / '.join(f'{x:.2f}' for x in five)} | | |")
    text = "\n".join(L) + "\n"
    (OUT / "analysis" / "rolc.md").write_text(text)
    save(OUT / "decisions" / "rolc_table.json", {"rows": rows, "written_utc": now()}, exclusive=True)
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
        require(a.seed == 3409, "seeds 3407 / 3408 are read by the ROL stage")
        tag, sp = RR.arm_spec("rol5", a.seed)
        return RR.engine(tag, a.task, sp)
    if a.verb == "causal":
        return causal(a.seed, a.task)
    return table()


if __name__ == "__main__":
    main()
