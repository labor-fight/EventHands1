#!/usr/bin/env python3
"""PMR written screen (docs/EXPERIMENT_RECORDS.md#pmr-prereg-20261109 and the design's section), computed once after
every reading of the design's three arms; CPU only, read-only on the readings (this study's and PFO2's B2).

    python -m tools.pmr.analyze screen --design d1
        -> analysis/screen_<design>.md, decisions/screen_<design>.json, main rows outputs/semkine/pmr_<arm>_s3407_main_row.json

Rows (tools/evf/mech.py `row`: E1 five, E3, amplification, zgz_local retention R15 / F15 / T10 / T30, C): B2 (seed
3407), R, R-noprop, R-mlp, and R with every message zeroed at test time (frozen, descriptive). Training summary from the
TensorBoard logs (mean train loss over the last 10 logged steps, val joint error at step 6000; descriptive).
Written rules (seed 3407; RA = overall recursive MPJPE-RA of E1, mm; E3 = overall single-step MPJPE-RA):
  (a) necessary   RA(noprop) - RA(R) >= 1.0, and the zgz_local and zgz_global MPJPE-RA differences noprop - R are
                  both > 0 (same direction)
  (b) sufficient  RA(mlp) >= RA(R) - 0.3  (the concat + MLP control is not better than R by more than 0.3)
  (c) no regress  RA(R) - RA(B2) <= +0.3
  health          |E3(noprop) - E3(R)| <= 1.0 and |E3(mlp) - E3(R)| <= 1.0 (a gap from a collapsed control does not pass)
  SCREEN_PASS = (a) and (b) and (c) and health; else SCREEN_FAIL (the failure attribution is written by hand from the
  rows, the training summary and the mechanism reading, in the design's results section)
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.pmr import arms as A  # noqa: E402
from tools.pmr.common import B2_TAG, NOISE_RA, OUT, PFO2_OUT, S43_ENGINE, SEED, now, read, require, save  # noqa: E402

SEMKINE = ROOT / "outputs/semkine"
KEYS = ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm")
GAP_A, TOL_B, TOL_C, HEALTH = 1.0, 0.3, 0.3, 1.0
LABELS = {"r": "R：部位图传播 P + 逐关节局部读出", "noprop": "R-noprop：去掉 P 的消息（逐节点更新与局部读出保留）",
          "mlp": "R-mlp：同样的节点输入 + 拼接 + MLP（旁路对照）"}


def mech():
    from tools.evf import mech as M
    M.ROOTS.update(pmr=OUT, pfo2=PFO2_OUT)
    return M


def rules(b2, r, n, m):
    """The written verdicts (pure; tested on synthetic numbers). Each argument: {"five": [L, G, VL, VG, RA], "e3": E3}."""
    ra = {k: v["five"][4] for k, v in (("b2", b2), ("r", r), ("n", n), ("m", m))}
    gap_n = ra["n"] - ra["r"]
    dl, dg = n["five"][0] - r["five"][0], n["five"][1] - r["five"][1]
    a = bool(gap_n >= GAP_A and dl > 0 and dg > 0)
    b = bool(ra["m"] >= ra["r"] - TOL_B)
    c = bool(ra["r"] - ra["b2"] <= TOL_C)
    health = bool(abs(n["e3"] - r["e3"]) <= HEALTH and abs(m["e3"] - r["e3"]) <= HEALTH)
    return {"a_necessary": a, "b_sufficient": b, "c_no_regress": c, "health": health,
            "verdict": "SCREEN_PASS" if (a and b and c and health) else "SCREEN_FAIL",
            "numbers": {"ra": ra, "gap_noprop_minus_r": gap_n, "local_diff": dl, "global_diff": dg,
                        "mlp_minus_r": ra["m"] - ra["r"], "r_minus_b2": ra["r"] - ra["b2"],
                        "e3_noprop_minus_r": n["e3"] - r["e3"], "e3_mlp_minus_r": m["e3"] - r["e3"]}}


def train_summary(run_dir):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return {}
    logs = list((Path(run_dir) / "logs").glob("*"))
    if not logs:
        return {}
    ea = EventAccumulator(str(logs[0]), size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    tags = ea.Tags()["scalars"]
    for t in ("train_loss", "train_joint_error_mm", "train_rotation_error_deg"):
        if t in tags:
            out[t + "_last10"] = float(np.mean([s.value for s in ea.Scalars(t)[-10:]]))
    for t in ("val_loss", "val_joint_error_mm", "val_rotation_error_deg"):
        if t in tags:
            s = ea.Scalars(t)
            out[t + "_final"] = float(s[-1].value)
    return out


def fmt(v, d=2):
    return "–" if v is None else f"{v:.{d}f}"


def cost_of(tag):
    p = OUT / "costs" / f"{tag}__fast__default" / "report.json"
    return read(p) if p.exists() else None


def main_row(arm, tag, label):
    c = cost_of(tag)
    require(c is not None, "the screen's main rows need the arm's cost reading (default flags): " + tag)
    r = read(OUT / "readings" / tag / "e1" / "report.json")
    s = r["per_sequence"]
    per = {"0": {"reading": f"pmr:{tag}", "checkpoint": r["identity"]["checkpoint"],
                 "local": {k: s["zgz_local"][k] for k in KEYS}, "global": {k: s["zgz_global"][k] for k in KEYS},
                 "overall": {k: r["overall"][k] for k in KEYS}}}
    m = {b: dict(per["0"][b]) for b in ("local", "global", "overall")}
    out = {"arm": f"pmr_{arm}", "label": label, "split": "val_core", "n_seeds": 1, "two_seed_mean": m, "per_seed": per,
           "latency_ms_raw": c["latency_row"]["raw_ms"], "latency_anchor_ms": c["latency_row"]["anchor_ms"],
           "latency_ms_scaled_full1p75": c["latency_row"]["scaled_ms"], "macs_forward_packet": c["macs_layers"]["median"],
           "params_total": c["params_total"], "latency_e2e_ms": c["latency_e2e_ms"],
           "cost_reading": str(OUT / "costs" / f"{tag}__fast__default"), "written_utc": now(),
           "note": "PMR screen row (seed 3407, PFO's engine E1, deterministic flags); cost = this arm's own reading "
                   "(PFO's harness, PyTorch default flags, GPU 4 exclusive)"}
    p = SEMKINE / f"pmr_{arm}_s{SEED}_main_row.json"
    save(p, out)
    return p


def screen(design):
    require(design in A.DESIGNS, "unregistered design")
    dst = OUT / "decisions" / f"screen_{design}.json"
    require(not dst.exists(), "written once: " + str(dst))
    M = mech()
    tags = {v: A.run_name(f"{design}{c}", SEED) for c, v in A.VARIANTS.items()}
    rows = {"b2": M.row("pfo2", B2_TAG), **{v: M.row("pmr", t) for v, t in tags.items()},
            "r_nomsg": M.row("pmr", tags["r"] + "__nomsg")}
    verdict = rules(rows["b2"], rows["r"], rows["noprop"], rows["mlp"])
    train = {v: train_summary(A.run_dir(f"{design}{c}", SEED)) for c, v in A.VARIANTS.items()}
    train["b2"] = train_summary(PFO2_OUT / "runs" / B2_TAG)
    names = {"b2": "B2（PFO2 fgs，参照）", "r": LABELS["r"], "noprop": LABELS["noprop"], "mlp": LABELS["mlp"],
             "r_nomsg": "R 测试时消息置零（冻结，描述）"}
    L = [f"## PMR {design.upper()} 筛选（种子 3407，从头 6000 步，无 ROL；PFO 引擎；写定规则）", "",
         f"S43（引擎三种子）{' / '.join(fmt(x) for x in S43_ENGINE)}；B2 3407 RA {fmt(rows['b2']['five'][4])}；"
         f"单种子噪声底约 {NOISE_RA} mm RA。", "",
         "| 行 | E1 五项 | E3 | 放大 | local R15 / F15 / T10 / T30 | C |", "|---|---|---|---|---|---|"]
    for k, r in rows.items():
        ret = " / ".join(fmt(r["retention"][q]) for q in ("R15", "F15", "T10", "T30")) if "retention" in r else "–"
        five = " / ".join(fmt(x) for x in r["five"]) if "five" in r else "–"
        L.append(f"| {names[k]} | {five} | {fmt(r.get('e3'))} | {fmt(r['amp'][0]) if 'amp' in r else '–'} | {ret} | "
                 f"{fmt(r.get('C'), 3)} |")
    n = verdict["numbers"]
    L += ["", "| 判据 | 数值 | 结果 |", "|---|---|---|",
          f"| (a) 必要：RA(noprop) − RA(R) ≥ {GAP_A}，local / global 同向 | {fmt(n['gap_noprop_minus_r'])}（local "
          f"{fmt(n['local_diff'])} / global {fmt(n['global_diff'])}） | {verdict['a_necessary']} |",
          f"| (b) 充分：RA(mlp) ≥ RA(R) − {TOL_B} | mlp − R = {fmt(n['mlp_minus_r'])} | {verdict['b_sufficient']} |",
          f"| (c) 不退步：RA(R) − RA(B2) ≤ +{TOL_C} | {fmt(n['r_minus_b2'])} | {verdict['c_no_regress']} |",
          f"| 健全性：两对照 E3 与 R 差 ≤ {HEALTH} | noprop {fmt(n['e3_noprop_minus_r'])} / mlp {fmt(n['e3_mlp_minus_r'])} | "
          f"{verdict['health']} |", "", f"**{verdict['verdict']}**", "",
          "训练摘要（末 10 次 train loss / train 关节误差 mm / 第 6000 步 val 关节误差 mm，描述）：", ""]
    for k, t in train.items():
        L.append(f"- {names.get(k, k)}：{fmt(t.get('train_loss_last10'), 4)} / {fmt(t.get('train_joint_error_mm_last10'))} / "
                 f"{fmt(t.get('val_joint_error_mm_final'))}")
    text = "\n".join(L) + "\n"
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    (OUT / "analysis" / f"screen_{design}.md").write_text(text)
    main_rows = [str(main_row(f"{design}{c}", tags[v], f"PMR {design.upper()} {LABELS[v]}（种子 3407）"))
                 for c, v in A.VARIANTS.items() if cost_of(tags[v]) is not None]
    save(dst, {"design": design, "rows": rows, "verdict": verdict, "train": train, "main_rows": main_rows,
               "thresholds": {"a": GAP_A, "b": TOL_B, "c": TOL_C, "health": HEALTH}, "written_at_utc": now()},
         exclusive=True)
    print(text)
    print(verdict)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("screen")
    s.add_argument("--design", required=True)
    a = p.parse_args(argv)
    screen(a.design)


if __name__ == "__main__":
    main()
