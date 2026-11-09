#!/usr/bin/env python3
"""TPD LC stage readouts (docs/EXPERIMENT_RECORDS.md#tpd-lc-prereg-20261105); CPU only, read-only on the probes and readings.

    python -m tools.tpd.analyze_lc screen    decisions/lc_screen.json (once): per router, zgz_local finger-group accuracy
                                             >= 0.90 -> the zero-training swap into the frozen L1 trunk runs
    python -m tools.tpd.analyze_lc all       decisions/lc_summary.json, analysis/lc.md (once, after every reading)

Written rules: screen as above (r_s2d 0.855); closed-loop gate RA(L1 + R) <= RA(L1 + r_s2d) - 0.65 (IDP's idp_l1_s3407 e1,
11.55) -> SWAP_PASS, else SWAP_FAIL; a router that fails the screen is ID_SCREEN_FAIL (no closed-loop reading).
lc - lc0 (the history's value at the same fine-tune) is descriptive.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.tpd.common import IDP_OUT, OUT, S43_ENGINE, now, read, save  # noqa: E402

LC_OUT = OUT / "lc"
ROUTERS = ("lc", "lc0")
FINGER_MIN = 0.90
GATE = 0.65


def screen_rule(acc):
    return acc >= FINGER_MIN


def gate_rule(ra, ra_base):
    return ra <= ra_base - GATE


def probes():
    return {f"{r}__{s}": read(LC_OUT / "probes" / f"{r}__{s}.json") for r in ("r_s2d", "lc_init", *ROUTERS)
            for s in ("zgz_local", "zgz_global") if (LC_OUT / "probes" / f"{r}__{s}.json").exists()}


def screen():
    p = probes()
    out = {"finger_min": FINGER_MIN, "screen": {r: {"finger_acc_zgz_local": p[f"{r}__zgz_local"]["finger_acc"],
                                                    "swap": screen_rule(p[f"{r}__zgz_local"]["finger_acc"])}
                                                for r in ROUTERS},
           "r_s2d_finger_acc_zgz_local": p["r_s2d__zgz_local"]["finger_acc"], "probes": p, "written_at_utc": now()}
    save(OUT / "decisions" / "lc_screen.json", out, exclusive=True)
    print(json.dumps(out["screen"]))


def all_():
    from tools.evf import mech as M
    M.ROOTS.update(tpd=OUT, idp=IDP_OUT)
    s = read(OUT / "decisions" / "lc_screen.json")
    base = M.row("idp", "idp_l1_s3407")
    rows, verdicts = {"l1__r_s2d": base}, {}
    for r in ROUTERS:
        if s["screen"][r]["swap"]:
            rows[f"l1__{r}"] = M.row("tpd", f"l1__{r}")
            ok = gate_rule(rows[f"l1__{r}"]["five"][4], base["five"][4])
            verdicts[r] = {"verdict": "SWAP_PASS" if ok else "SWAP_FAIL",
                           "d_ra": rows[f"l1__{r}"]["five"][4] - base["five"][4]}
        else:
            verdicts[r] = {"verdict": "ID_SCREEN_FAIL"}
    p = s["probes"]
    L = ["## TPD 问题 1 · LC 长上下文路由器（种子 3407；描述 + 写定规则）", "",
         "| 路由器 | zgz_local 部位 / 手指级准确率 | 掌–指 / 跨指错误 | zgz_global 部位 / 手指级 |", "|---|---|---|---|"]
    for r in ("r_s2d", "lc_init", *ROUTERS):
        a, b = p.get(f"{r}__zgz_local"), p.get(f"{r}__zgz_global")
        if a:
            L.append(f"| {r} | {a['part_acc']:.3f} / {a['finger_acc']:.3f} | {a['palm_finger_err']:.3f} / "
                     f"{a['cross_finger_err']:.3f} | " + (f"{b['part_acc']:.3f} / {b['finger_acc']:.3f}" if b else "–") + " |")
    L += ["", f"S43（引擎三种子）{' / '.join(f'{x:.2f}' for x in S43_ENGINE)}。", "",
          "| 行 | E1 五项 | E3 | 放大 | R15 / F15 | C |", "|---|---|---|---|---|---|"]
    for k, r in rows.items():
        L.append(f"| {k} | {' / '.join(f'{x:.2f}' for x in r['five'])} | {r['e3']:.2f} | {r['amp'][0]:.2f} | "
                 f"{r['retention']['R15']:.2f} / {r['retention']['F15']:.2f} | {r.get('C', float('nan')):.3f} |")
    L += ["", "判读：" + "；".join(f"{r} **{v['verdict']}**" + (f"（ΔRA {v['d_ra']:+.2f}）" if "d_ra" in v else "")
                                   for r, v in verdicts.items()) + "。"]
    text = "\n".join(L) + "\n"
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    (OUT / "analysis" / "lc.md").write_text(text)
    save(OUT / "decisions" / "lc_summary.json", {"rows": rows, "verdicts": verdicts, "screen": s["screen"],
                                                 "written_at_utc": now()}, exclusive=True)
    print(text)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("verb", choices=("screen", "all"))
    a = p.parse_args(argv)
    (screen if a.verb == "screen" else all_)()


if __name__ == "__main__":
    main()
