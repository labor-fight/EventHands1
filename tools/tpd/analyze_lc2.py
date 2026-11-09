#!/usr/bin/env python3
"""TPD LC stage readout, engineering amendment 1 (docs/EXPERIMENT_RECORDS.md#tpd-lc-amend1-20261105): tools/tpd/analyze_lc.py
`all` unchanged in its rules, with the table tolerating rows whose readings lack E3 / perturbation / f1 (IDP published
only the e1 reading of L1 with the deployed router). CPU, once.

    python -m tools.tpd.analyze_lc2 all      decisions/lc_summary.json, analysis/lc.md
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.tpd import analyze_lc as AL  # noqa: E402
from tools.tpd.common import IDP_OUT, OUT, S43_ENGINE, now, read, save  # noqa: E402


def fmt(v, d=2):
    return "–" if v is None else f"{v:.{d}f}"


def all_():
    from tools.evf import mech as M
    M.ROOTS.update(tpd=OUT, idp=IDP_OUT)
    s = read(OUT / "decisions" / "lc_screen.json")
    base = M.row("idp", "idp_l1_s3407")
    rows, verdicts = {"l1__r_s2d": base}, {}
    for r in AL.ROUTERS:
        if s["screen"][r]["swap"]:
            rows[f"l1__{r}"] = M.row("tpd", f"l1__{r}")
            ok = AL.gate_rule(rows[f"l1__{r}"]["five"][4], base["five"][4])
            verdicts[r] = {"verdict": "SWAP_PASS" if ok else "SWAP_FAIL", "d_ra": rows[f"l1__{r}"]["five"][4] - base["five"][4]}
        else:
            verdicts[r] = {"verdict": "ID_SCREEN_FAIL"}
    p = s["probes"]
    L = ["## TPD 问题 1 · LC 长上下文路由器（种子 3407；描述 + 写定规则）", "",
         "| 路由器 | zgz_local 部位 / 手指级准确率 | 掌–指 / 跨指错误 | zgz_global 部位 / 手指级 |", "|---|---|---|---|"]
    for r in ("r_s2d", "lc_init", *AL.ROUTERS):
        a, b = p.get(f"{r}__zgz_local"), p.get(f"{r}__zgz_global")
        if a:
            L.append(f"| {r} | {a['part_acc']:.3f} / {a['finger_acc']:.3f} | {a['palm_finger_err']:.3f} / "
                     f"{a['cross_finger_err']:.3f} | " + (f"{b['part_acc']:.3f} / {b['finger_acc']:.3f}" if b else "–") + " |")
    L += ["", f"S43（引擎三种子）{' / '.join(f'{x:.2f}' for x in S43_ENGINE)}。", "",
          "| 行 | E1 五项 | E3 | 放大 | R15 / F15 | C |", "|---|---|---|---|---|---|"]
    for k, r in rows.items():
        ret = r.get("retention")
        L.append(f"| {k} | {' / '.join(f'{x:.2f}' for x in r['five'])} | {fmt(r.get('e3'))} | "
                 f"{fmt(r['amp'][0] if 'amp' in r else None)} | "
                 f"{(fmt(ret['R15']) + ' / ' + fmt(ret['F15'])) if ret else '–'} | {fmt(r.get('C'), 3)} |")
    L += ["", "判读：" + "；".join(f"{r} **{v['verdict']}**" + (f"（ΔRA {v['d_ra']:+.2f}）" if "d_ra" in v else "")
                                   for r, v in verdicts.items()) + "。"]
    text = "\n".join(L) + "\n"
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    (OUT / "analysis" / "lc.md").write_text(text)
    save(OUT / "decisions" / "lc_summary.json", {"rows": rows, "verdicts": verdicts, "screen": s["screen"],
                                                 "amendment": "tpd-lc-amend1-20261105", "written_at_utc": now()},
         exclusive=True)
    print(text)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("verb", choices=("all",))
    p.parse_args(argv)
    all_()


if __name__ == "__main__":
    main()
