#!/usr/bin/env python3
"""S45-PX decision and report (docs/EXPERIMENT_RECORDS.md#s45px-prereg-20261030); CPU only, run once after every
registered reading. Writes outputs/s45px/20261030/summary.json and report.md. Single seed 3407, descriptive.

Each arm against S45:
  zgz (XCL single-seed rule), per domain: better if S45 - arm >= 5 mm (local) / 8 mm (global) in E1 at step 6000,
      E1 / E2 / E3 all favour the arm and >= 2 of the checkpoints 5000 / 5500 / 6000 clear the same margin; worse is
      the mirror image; divergence = E1 global > S45 + 8 mm
  in-sample (nine training local recordings, own loop, median RA): better if S45 - arm >= 2.5 mm at each of 5000 /
      5500 / 6000 and >= 7 of 9 recordings favour the arm at 6000; worse is the mirror image
  class: reverse (any worse or divergence) > better (zgz better in a domain and in-sample better) > sample_only >
      zgz_only > noise
Study decision: REVERSE > A_MANO (pxm better) > A_CAMERA (pxc better) > B_SAMPLE_ONLY > E_ZGZ_ONLY > C_NULL.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.s45px.common import ARMS, OUT, S45TR, now, read, require, run_name, save, verify_protocol  # noqa: E402

TH = {"zgz_local_mm": 5.0, "zgz_global_mm": 8.0, "late_needed": 2, "divergence_mm": 8.0,
      "insample_mm": 2.5, "recordings_needed": 7}
STEPS = (5000, 5500, 6000)
S45_EVAL = ROOT / "outputs/c1sf/20261029/evaluations/so3"
S45CT_READOUTS = ROOT / "outputs/s45ct/20261030/readouts"
S45_ROW = ROOT / "outputs/semkine/s45_main_row.json"
LABELS = {"pxc": "S45-PX pxc：记录 + 关联顶点的前态相机 XYZ",
          "pxm": "S45-PX pxm：记录 + 关联顶点相对前态腕点的位置（MANO 系）"}
NEXT = {
    "A_MANO": "the hand-frame vertex position is a missing record input: a second seed (3408) of pxm and of S45 under "
              "the same rules (new pre-registration) before any adoption; the correspondence line is unaffected",
    "A_CAMERA": "the gain needs only the camera position: zero-training S45-LOC-style translation / depth-shift readout "
                "on pxc before anything else; no adoption",
    "B_SAMPLE_ONLY": "the vertex position helps fit the training subjects but does not transfer to zgz: stop this line",
    "E_ZGZ_ONLY": "zgz moves without an in-sample change: descriptive only, no follow-up training",
    "C_NULL": "the missing vertex position is not the bottleneck (consistent with S45-DEEP): stop this line",
    "REVERSE": "the added previous-mesh position makes the network worse: stop this line and record it",
}


def trace(directory):
    m = read(Path(directory) / "report.json")["metrics"]
    return {d: m[d]["mpjpe_ra_mm"] for d in ("local", "global")}


def zgz(net):
    if net == "s45":
        e1 = {s: trace(S45_EVAL / f"e1_step{s}_repeat1_none") for s in STEPS}
        # E2 = fed S45's own E1 states, so S45's E2 is its E1 (as in S45-TR)
        return {"e1": e1, "e2": e1[6000], "e3": trace(S45_EVAL / "e3_step6000_repeat1_none")}
    ev = OUT / "evaluations" / net
    return {"e1": {s: trace(ev / f"e1_step{s}") for s in STEPS}, "e2": trace(ev / "e2_step6000"),
            "e3": trace(ev / "e3_step6000")}


def insample(net, step):
    if net == "s45":
        base = S45TR / "readouts/s45" if step == 6000 else S45CT_READOUTS / f"s45_{step}"
    else:
        base = OUT / "readouts" / (net if step == 6000 else f"{net}_{step}")
    z = np.load(base / "tl_e1" / "states.npz", allow_pickle=False)
    recs = sorted({k.split("/")[0] for k in z.files})
    return {r: float(z[f"{r}/mpjpe_ra_mm"].mean()) for r in recs}


def classify(arm, ref):
    z, zs = zgz(arm), ref["zgz"]
    out = {"zgz": {}, "insample": {}}
    for dom, th in (("local", TH["zgz_local_mm"]), ("global", TH["zgz_global_mm"])):
        gain = {s: zs["e1"][s][dom] - z["e1"][s][dom] for s in STEPS}
        g2, g3 = zs["e2"][dom] - z["e2"][dom], zs["e3"][dom] - z["e3"][dom]
        late_up = sum(gain[s] >= th for s in STEPS)
        late_down = sum(gain[s] <= -th for s in STEPS)
        out["zgz"][dom] = {"e1_gain_by_step": {str(s): v for s, v in gain.items()}, "e2_gain": g2, "e3_gain": g3,
                           "better": gain[6000] >= th and g2 > 0 and g3 > 0 and late_up >= TH["late_needed"],
                           "worse": gain[6000] <= -th and g2 < 0 and g3 < 0 and late_down >= TH["late_needed"]}
    divergence = z["e1"][6000]["global"] > zs["e1"][6000]["global"] + TH["divergence_mm"]
    per = {s: insample(arm, s) for s in STEPS}
    med = {s: float(np.median(list(per[s].values()))) for s in STEPS}
    gain = {s: ref["insample_median"][s] - med[s] for s in STEPS}
    favour = sum(per[6000][r] < ref["insample"][6000][r] for r in per[6000])
    against = sum(per[6000][r] > ref["insample"][6000][r] for r in per[6000])
    require(set(per[6000]) == set(ref["insample"][6000]) and len(per[6000]) == 9, "in-sample recordings differ")
    ins_better = all(gain[s] >= TH["insample_mm"] for s in STEPS) and favour >= TH["recordings_needed"]
    ins_worse = all(gain[s] <= -TH["insample_mm"] for s in STEPS) and against >= TH["recordings_needed"]
    out["insample"] = {"median_by_step": {str(s): v for s, v in med.items()}, "gain_by_step": {str(s): v for s, v in gain.items()},
                       "recordings_favouring_arm_6000": favour, "better": ins_better, "worse": ins_worse}
    zb = any(out["zgz"][d]["better"] for d in ("local", "global"))
    zw = any(out["zgz"][d]["worse"] for d in ("local", "global"))
    if zw or divergence or ins_worse:
        c = "reverse"
    elif zb and ins_better:
        c = "better"
    elif ins_better:
        c = "sample_only"
    elif zb:
        c = "zgz_only"
    else:
        c = "noise"
    out.update(zgz_three=z, global_divergence=divergence, cls=c)
    return out


def decide(cls):
    if "reverse" in cls.values():
        return "REVERSE"
    if cls["pxm"] == "better":
        return "A_MANO"
    if cls["pxc"] == "better":
        return "A_CAMERA"
    if "sample_only" in cls.values():
        return "B_SAMPLE_ONLY"
    if "zgz_only" in cls.values():
        return "E_ZGZ_ONLY"
    return "C_NULL"


def main_table():
    from tools import report_table as original
    paths = {"dt3_dz_l3_w200cfg": ROOT / "outputs/semkine/dt3_dz_l3_w200cfg_main_row.json", "s45": S45_ROW}
    labels = {"s45": "S45：XYZ-C1 + SO(3)+平移+FK 损失（seed3407）"}
    for arm in ARMS:
        prefix = run_name(arm).removesuffix("_s3407")
        f = OUT / "evaluations" / arm / "row_step6000" / f"{prefix}_main_row_step6000.json"
        require(f.is_file(), "main row missing: " + str(f))
        paths[prefix], labels[prefix] = f, LABELS[arm]

    class Rows:
        def __truediv__(self, filename):
            return paths[str(filename).removesuffix("_main_row.json")]
    old = original.OUT
    original.OUT = Rows()
    try:
        lines = [original.HEADER, original.RULE, original.BASELINE,
                 original.row("dt3_dz_l3_w200cfg", "S39：dt_dz_l3，W=200 + 自适应滤波 + 证据门 ramp1000 *")]
        lines += [original.row(p, labels[p]) for p in paths if p != "dt3_dz_l3_w200cfg"]
    finally:
        original.OUT = old
    return "\n".join(lines)


def fmt(v, d=2):
    return "—" if v is None else f"{v:.{d}f}"


def main():
    verify_protocol()
    target = OUT / "summary.json"
    require(not target.exists(), "the S45-PX decision is computed once")
    ins = {s: insample("s45", s) for s in STEPS}
    ref = {"zgz": zgz("s45"), "insample": ins, "insample_median": {s: float(np.median(list(ins[s].values()))) for s in STEPS}}
    arms = {arm: classify(arm, ref) for arm in ARMS}
    cls = {arm: r["cls"] for arm, r in arms.items()}
    decision = decide(cls)
    mech = {"s45": read(S45TR / "readouts/s45/summary.json"), "t1d": read(S45TR / "readouts/t1d/summary.json")}
    mech.update({arm: read(OUT / "readouts" / arm / "summary.json") for arm in ARMS})
    lines = ["# S45-PX 结果（自动生成；单 seed 3407，描述性）", "", main_table(), "",
             f"**判读：{decision}** —— " + NEXT[decision], "",
             "各臂分类：" + "；".join(f"{a} = {c}" for a, c in cls.items()), "",
             "## zgz E1 / E2 / E3（RA，mm；括号内为 S45 − 臂，正 = 臂更好）", "",
             "| 网络 | E1 local | E1 global | E2 local | E2 global | E3 local | E3 global | E1 local @5000 / 5500 | E1 global @5000 / 5500 |",
             "|---|---|---|---|---|---|---|---|---|"]
    rz = ref["zgz"]
    lines.append(f"| s45 | {fmt(rz['e1'][6000]['local'])} | {fmt(rz['e1'][6000]['global'])} | {fmt(rz['e2']['local'])} | "
                 f"{fmt(rz['e2']['global'])} | {fmt(rz['e3']['local'])} | {fmt(rz['e3']['global'])} | "
                 f"{fmt(rz['e1'][5000]['local'])} / {fmt(rz['e1'][5500]['local'])} | {fmt(rz['e1'][5000]['global'])} / {fmt(rz['e1'][5500]['global'])} |")
    for arm, r in arms.items():
        z = r["zgz_three"]
        cell = lambda v, refv: f"{fmt(v)}（{refv - v:+.2f}）"  # noqa: E731
        lines.append(f"| {arm} | {cell(z['e1'][6000]['local'], rz['e1'][6000]['local'])} | {cell(z['e1'][6000]['global'], rz['e1'][6000]['global'])} | "
                     f"{cell(z['e2']['local'], rz['e2']['local'])} | {cell(z['e2']['global'], rz['e2']['global'])} | "
                     f"{cell(z['e3']['local'], rz['e3']['local'])} | {cell(z['e3']['global'], rz['e3']['global'])} | "
                     f"{fmt(z['e1'][5000]['local'])} / {fmt(z['e1'][5500]['local'])} | {fmt(z['e1'][5000]['global'])} / {fmt(z['e1'][5500]['global'])} |")
    lines += ["", "## 训练受试者自身闭环（9 条 local 录制，RA 中位，mm）", "",
              "| 网络 | 5000 | 5500 | 6000 | S45 − 臂 @5000 / 5500 / 6000 | 6000 步优于 S45 的录制数 |", "|---|---|---|---|---|---|"]
    m = ref["insample_median"]
    lines.append(f"| s45 | {fmt(m[5000])} | {fmt(m[5500])} | {fmt(m[6000])} | — | — |")
    for arm, r in arms.items():
        i = r["insample"]
        lines.append(f"| {arm} | {fmt(i['median_by_step']['5000'])} | {fmt(i['median_by_step']['5500'])} | {fmt(i['median_by_step']['6000'])} | "
                     f"{i['gain_by_step']['5000']:+.2f} / {i['gain_by_step']['5500']:+.2f} / {i['gain_by_step']['6000']:+.2f} | "
                     f"{i['recordings_favouring_arm_6000']}/9 |")
    lines += ["", "## 机制读数（zgz_local / 训练 local 中位；定义同 S45-TR）", "",
              "| 网络 | 保留率 T10 / R5 / Fs | 流形内 / 外 | 收缩 k（全部 / 低运动） | 反复送入 GT 起点 RA / k @20 | 均值起点 k @20 | 闭环起点 RA / k @20 | 钳位份额 姿态 / 平移 |",
              "|---|---|---|---|---|---|---|---|"]
    for net, s in mech.items():
        for dom in ("zgz_local", "train_local"):
            r1, r2, r3, r4 = s["r1_retention"][dom], s["r2_shrinkage"][dom], s["r3_fixed_point"][dom], s["r4_clamp"][dom]
            lines.append(
                f"| {net} {dom} | {fmt(r1['T10'])} / {fmt(r1['R5'])} / {fmt(r1['Fs'])} | {fmt(r1['top'])} / {fmt(r1['bottom'])} | "
                f"{fmt(r2['k_all'])} / {fmt(r2['k_low_motion'])} | {fmt(r3['gt']['20']['ra_mm_median'], 1)} / {fmt(r3['gt']['20']['shrink_k_median'])} | "
                f"{fmt(r3['mean']['20']['shrink_k_median'])} | {fmt(r3['loop']['20']['ra_mm_median'], 1)} / {fmt(r3['loop']['20']['shrink_k_median'])} | "
                f"{fmt(r4['pose_clamp_share'])} / {fmt(r4['transl_clamp_share'])} |")
    text = "\n".join(lines) + "\n"
    save(target, {"computed_utc": now(), "thresholds": TH, "decision": decision, "next": NEXT[decision], "classes": cls,
                  "arms": {a: {k: v for k, v in r.items() if k != "zgz_three"} for a, r in arms.items()},
                  "zgz_three": {a: r["zgz_three"] for a, r in arms.items()},
                  "s45_reference": {"zgz": ref["zgz"], "insample_median": ref["insample_median"]},
                  "mechanism": {k: {kk: v[kk] for kk in ("r1_retention", "r2_shrinkage", "r3_fixed_point", "r4_clamp")}
                                for k, v in mech.items()},
                  "single_seed_descriptive_only": True, "s39_s43_c1_s45_unchanged": True}, exclusive=True)
    (OUT / "report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
