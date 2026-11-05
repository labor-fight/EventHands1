#!/usr/bin/env python3
"""IDP written readouts (docs/EXPERIMENT_RECORDS.md#idp-prereg-20261105), computed once after every reading; CPU only,
read-only on the readings (this study's, PFO2's B2 and B2UB's label_part); descriptive except the written rules.

    python -m tools.idp.analyze all
        -> analysis/ladder.md, decisions/summary.json and the main rows outputs/semkine/idp_{l1,l2,l3}_main_row.json

Rows (tools/evf/mech.py `row`: E1 five, E3, amplification, zgz_local retention R15 / F15 / T10 / T30, C):
  B2 (seed 3407, deployed), B2 + label_part at test only (B2UB U1), l1, l2 (label_part, training and test), l3
  (deployed), and the exposure rows l1 / l2 read with the deployed router.
Written rules (RA = overall recursive MPJPE, mm; seed 3407 throughout):
  L1  RA(l1) >= RA(B2) - 0.5 -> POINT_ERROR_LIMITS; RA(l1) <= 8.0 -> IDENTITY_IS_LEVER; else MIXED
  L2  RA(l2) <= RA(l1) - 0.5 -> FEAT_HELPS, else FEAT_NULL
  L3  screening (#pfo2-amend3-20261103): (RA(l3) <= RA(B2) - 0.3, or zgz_local R15 and F15 both <= B2's - 0.05) and
      E3(l3) <= E3(B2) + 0.3 -> SCREEN_PASS, else SCREEN_FAIL
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.idp import arms as A  # noqa: E402
from tools.idp.common import (B2_COST, B2_TAG, B2UB_OUT, NOISE_RA, OUT, PFO2_OUT, S43_ENGINE, SEED, now, read,  # noqa: E402
                              require, save)

SEMKINE = ROOT / "outputs/semkine"
KEYS = ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm")
IDENTITY_TARGET = 8.0


def mech():
    from tools.evf import mech as M
    M.ROOTS.update(idp=OUT, b2ub=B2UB_OUT, pfo2=PFO2_OUT)
    return M


def rules(ra_b2, ra_l1, ra_l2, ra_l3, e3_b2, e3_l3, ret_b2, ret_l3):
    """The written verdicts (pure; tested on synthetic numbers)."""
    l1 = ("POINT_ERROR_LIMITS" if ra_l1 >= ra_b2 - 0.5 else "IDENTITY_IS_LEVER" if ra_l1 <= IDENTITY_TARGET else "MIXED")
    feat = "FEAT_HELPS" if ra_l2 <= ra_l1 - 0.5 else "FEAT_NULL"
    better_ra = ra_l3 <= ra_b2 - 0.3
    better_ret = ret_l3["R15"] <= ret_b2["R15"] - 0.05 and ret_l3["F15"] <= ret_b2["F15"] - 0.05
    screen = "SCREEN_PASS" if (better_ra or better_ret) and e3_l3 <= e3_b2 + 0.3 else "SCREEN_FAIL"
    return {"L1": l1, "L2": feat, "L3": screen, "L3_better_ra": bool(better_ra), "L3_better_retention": bool(better_ret),
            "L3_e3_ok": bool(e3_l3 <= e3_b2 + 0.3)}


def fmt(v, d=2):
    return "–" if v is None else f"{v:.{d}f}"


def main_row(arm, tag, label, feat):
    c = read(B2_COST)
    require(c["policy"] == "default", "the table's cost comes from the PyTorch-default-flag reading")
    r = read(OUT / "readings" / tag / "e1" / "report.json")
    s = r["per_sequence"]
    per = {"0": {"reading": f"idp:{tag}", "checkpoint": r["identity"]["checkpoint"],
                 "local": {k: s["zgz_local"][k] for k in KEYS}, "global": {k: s["zgz_global"][k] for k in KEYS},
                 "overall": {k: r["overall"][k] for k in KEYS}}}
    m = {b: dict(per["0"][b]) for b in ("local", "global", "overall")}
    extra = 2 * 64 if feat == "fore" else 0
    out = {"arm": f"idp_{arm}", "label": label, "split": "val_core", "n_seeds": 1, "two_seed_mean": m, "per_seed": per,
           "latency_ms_raw": c["latency_row"]["raw_ms"], "latency_anchor_ms": c["latency_row"]["anchor_ms"],
           "latency_ms_scaled_full1p75": c["latency_row"]["scaled_ms"], "macs_forward_packet": c["macs_layers"]["median"],
           "params_total": c["params_total"] + extra, "latency_e2e_ms": c["latency_e2e_ms"], "cost_reading": str(B2_COST),
           "written_utc": now(),
           "note": "IDP row (seed 3407, PFO's engine E1, deterministic flags); cost columns = B2's deployable form from EVF's "
                   "final block" + (f" plus the {extra} zero-initialised feature weights (latency / MACs not re-measured)"
                                    if extra else "") + ("; label_part routes are diagnostic, not deployable"
                                                         if A.eval_route(arm) else "")}
    p = SEMKINE / f"idp_{arm}_main_row.json"
    save(p, out)
    return p


def all_():
    M = mech()
    rows = {"b2": M.row("pfo2", B2_TAG), "b2_label_part_test": M.row("b2ub", f"b2_s{SEED}__label_part"),
            "l1": M.row("idp", A.run_name("l1") + "__label_part"), "l2": M.row("idp", A.run_name("l2") + "__label_part"),
            "l3": M.row("idp", A.run_name("l3")), "l1_deployed": M.row("idp", A.run_name("l1")),
            "l2_deployed": M.row("idp", A.run_name("l2"))}
    ra = {k: v["five"][4] for k, v in rows.items()}
    verdict = rules(ra["b2"], ra["l1"], ra["l2"], ra["l3"], rows["b2"]["e3"], rows["l3"]["e3"], rows["b2"]["retention"],
                    rows["l3"]["retention"])
    names = {"b2": "B2 部署路由 r_s2d（PFO2，参照）", "b2_label_part_test": "B2 + label_part 只在测试时（B2UB U1）",
             "l1": "L1：label_part 训练 + 测试（标签身份，路由器点位）", "l2": "L2：L1 + 透视缩短可靠度列",
             "l3": "L3：B2 + 透视缩短可靠度列（可部署）", "l1_deployed": "L1 配部署路由 r_s2d（暴露）",
             "l2_deployed": "L2 配部署路由 r_s2d（暴露）"}
    L = ["## IDP 身份 / 点位阶梯（种子 3407；PFO 引擎；描述 + 写定规则）", "",
         f"S43（引擎三种子）{' / '.join(fmt(x) for x in S43_ENGINE)}；单种子噪声底约 {NOISE_RA} mm RA。", "",
         "| 行 | E1 五项 | E3 | 放大 | local R15 / F15 / T10 / T30 | C |", "|---|---|---|---|---|---|"]
    for k, r in rows.items():
        ret = " / ".join(fmt(r["retention"][q]) for q in ("R15", "F15", "T10", "T30")) if "retention" in r else "–"
        L.append(f"| {names[k]} | {' / '.join(fmt(x) for x in r['five'])} | {fmt(r.get('e3'))} | "
                 f"{fmt(r['amp'][0]) if 'amp' in r else '–'} | {ret} | {fmt(r.get('C'), 3)} |")
    L += ["", f"判读：L1 **{verdict['L1']}**（RA(l1) − RA(B2) = {fmt(ra['l1'] - ra['b2'])}）；"
          f"L2 **{verdict['L2']}**（RA(l2) − RA(l1) = {fmt(ra['l2'] - ra['l1'])}）；"
          f"L3 **{verdict['L3']}**（RA(l3) − RA(B2) = {fmt(ra['l3'] - ra['b2'])}，E3 差 {fmt(rows['l3']['e3'] - rows['b2']['e3'])}）。",
          "", "暴露行与 B2UB 的测试时 label_part 行只作描述。标签身份不可部署。"]
    text = "\n".join(L) + "\n"
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    (OUT / "analysis" / "ladder.md").write_text(text)
    main_rows = [str(main_row("l1", A.run_name("l1") + "__label_part", "IDP L1：label_part 训练 + 测试（种子 3407，不可部署）", "none")),
                 str(main_row("l2", A.run_name("l2") + "__label_part", "IDP L2：L1 + 透视缩短可靠度列（种子 3407，不可部署）", "fore")),
                 str(main_row("l3", A.run_name("l3"), "IDP L3：B2 + 透视缩短可靠度列（种子 3407，可部署）", "fore"))]
    summary = {"rows": rows, "ra": ra, "verdict": verdict, "identity_target_ra": IDENTITY_TARGET, "noise_ra": NOISE_RA,
               "s43_engine": list(S43_ENGINE), "main_rows": main_rows, "written_at_utc": now()}
    save(OUT / "decisions" / "summary.json", summary, exclusive=True)
    print(text)
    print(verdict)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("verb", choices=("all",))
    p.parse_args(argv)
    all_()


if __name__ == "__main__":
    main()
