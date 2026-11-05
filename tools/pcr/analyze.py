#!/usr/bin/env python3
"""PCR written readouts (docs/EXPERIMENT_RECORDS.md#pcr-prereg-20261105), computed once after every reading; CPU only.

    python -m tools.pcr.analyze all  -> analysis/pcr.md, decisions/summary.json, outputs/semkine/pcr_l1_<router>_main_row.json

Rows (tools/evf/mech.py `row`): l1__r_s2d (reference, re-read here), l1__pcr, l1__pcrb, b2__pcr, b2__pcrb (descriptive),
plus the router probes. Written rule per router R in (pcr, pcrb), seed 3407, RA = overall recursive MPJPE:
  d = RA(l1__R) - RA(l1__r_s2d);  d <= -0.85 -> PCR_WORKS;  d >= -0.30 -> PCR_NULL;  else PCR_WEAK
  flag PRIOR_COLLAPSE when zgz_local R15 or F15 of l1__R exceeds l1__r_s2d's by >= 0.05 (weaker correction)
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pcr.common import NOISE_RA, OUT, S43_ENGINE, now, read, require, save  # noqa: E402

SEMKINE = ROOT / "outputs/semkine"
KEYS = ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm")
B2_COST = ROOT / "outputs/evf/20261104/costs/final__pfo2_fgs_s3407__fast__default/report.json"


def rule(ra_ref, ra, ret_ref, ret):
    d = ra - ra_ref
    verdict = "PCR_WORKS" if d <= -0.85 else "PCR_NULL" if d >= -0.30 else "PCR_WEAK"
    collapse = (ret["R15"] - ret_ref["R15"] >= 0.05) or (ret["F15"] - ret_ref["F15"] >= 0.05)
    return {"verdict": verdict, "delta_ra": d, "prior_collapse": bool(collapse)}


def fmt(v, d=2):
    return "–" if v is None else f"{v:.{d}f}"


def main_row(tag, label):
    c = read(B2_COST)
    require(c["policy"] == "default", "the table's cost comes from the PyTorch-default-flag reading")
    r = read(OUT / "readings" / tag / "e1" / "report.json")
    s = r["per_sequence"]
    per = {"0": {"reading": f"pcr:{tag}", "checkpoint": r["identity"]["checkpoint"],
                 "local": {k: s["zgz_local"][k] for k in KEYS}, "global": {k: s["zgz_global"][k] for k in KEYS},
                 "overall": {k: r["overall"][k] for k in KEYS}}}
    out = {"arm": f"pcr_{tag}", "label": label, "split": "val_core", "n_seeds": 1,
           "two_seed_mean": {b: dict(per["0"][b]) for b in ("local", "global", "overall")}, "per_seed": per,
           "latency_ms_raw": c["latency_row"]["raw_ms"], "latency_anchor_ms": c["latency_row"]["anchor_ms"],
           "latency_ms_scaled_full1p75": c["latency_row"]["scaled_ms"], "macs_forward_packet": c["macs_layers"]["median"],
           "params_total": c["params_total"] + 256, "latency_e2e_ms": c["latency_e2e_ms"], "cost_reading": str(B2_COST),
           "written_utc": now(),
           "note": "PCR row (seed 3407, PFO engine E1); cost columns = B2's deployable form plus the router's 256 prior "
                   "weights; the prior map adds one MANO FK and a splat per packet (latency / MACs not re-measured)"}
    p = SEMKINE / f"pcr_{tag.replace('__', '_')}_main_row.json"
    save(p, out)
    return str(p)


def all_():
    from tools.evf import mech as M
    M.ROOTS["pcr"] = OUT
    tags = ("l1__r_s2d", "l1__pcr", "l1__pcrb", "b2__pcr", "b2__pcrb")
    rows = {t: M.row("pcr", t) for t in tags}
    probes = {}
    for f in sorted((OUT / "probes").glob("*.json")):
        probes[f.stem] = read(f)
    ref = rows["l1__r_s2d"]
    verdicts = {r: rule(ref["five"][4], rows[f"l1__{r}"]["five"][4], ref["retention"], rows[f"l1__{r}"]["retention"])
                for r in ("pcr", "pcrb")}
    names = {"l1__r_s2d": "L1 主干 + r_s2d（参照）", "l1__pcr": "L1 主干 + PCR（数据集噪声前态训练）",
             "l1__pcrb": "L1 主干 + PCRb（一半闭环状态库前态）", "b2__pcr": "B2 主干 + PCR（描述）",
             "b2__pcrb": "B2 主干 + PCRb（描述）"}
    L = ["## PCR 前态条件路由器（种子 3407，零训练换进冻结主干；PFO 引擎）", "",
         f"S43（引擎三种子）{' / '.join(fmt(x) for x in S43_ENGINE)}；单种子噪声底约 {NOISE_RA} mm RA。", "",
         "| 行 | E1 五项 | E3 | 放大 | local R15 / F15 / T10 / T30 | C |", "|---|---|---|---|---|---|"]
    for t, r in rows.items():
        ret = " / ".join(fmt(r["retention"][q]) for q in ("R15", "F15", "T10", "T30")) if "retention" in r else "–"
        L.append(f"| {names[t]} | {' / '.join(fmt(x) for x in r.get('five', [None] * 5))} | {fmt(r.get('e3'))} | "
                 f"{fmt(r['amp'][0]) if 'amp' in r else '–'} | {ret} | {fmt(r.get('C'), 3)} |")
    L += ["", "| 路由器探针 | 部位准确率 | 手指级准确率 | 部位正确时点误差中位 mm |", "|---|---|---|---|"]
    for k, v in probes.items():
        L.append(f"| {k} | {fmt(v['part_acc'], 3)} | {fmt(v['finger_acc'], 3)} | {fmt(v['point_err_part_correct_median_mm'])} |")
    L += ["", "判读：" + "；".join(f"{r} **{v['verdict']}**（ΔRA {fmt(v['delta_ra'])}"
                                    + ("，PRIOR_COLLAPSE" if v["prior_collapse"] else "") + "）" for r, v in verdicts.items()),
          "", "B2 主干行只作描述。"]
    text = "\n".join(L) + "\n"
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    (OUT / "analysis" / "pcr.md").write_text(text)
    main_rows = [main_row("l1__pcr", "PCR：L1 主干 + 前态条件路由器（种子 3407）"),
                 main_row("l1__pcrb", "PCRb：L1 主干 + 前态条件路由器，闭环状态库训练（种子 3407）")]
    save(OUT / "decisions" / "summary.json", {"rows": rows, "probes": probes, "verdicts": verdicts,
                                               "main_rows": main_rows, "written_at_utc": now()}, exclusive=True)
    print(text)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("verb", choices=("all",))
    p.parse_args(argv)
    all_()


if __name__ == "__main__":
    main()
