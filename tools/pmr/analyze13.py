#!/usr/bin/env python3
"""PMR design D2 single-seed screen, the written reading (docs/EXPERIMENT_RECORDS.md#pmr-d2-screen-prereg-20261110).
CPU only; computed once.

    python -m tools.pmr.analyze13 screen     decisions/s9x_d2_screen.json, analysis/s9x_d2_screen.md

Seed 3407. D2 + rol5 against D1 R + rol5 (EvHandTrack's seed 3407), both in the streaming form, PFO's engine E1:
  five    MPJPE-L, MPJPE-G, MPVPE-L, MPVPE-G, RA of the reading's report
  yb      zgz_global mean of the camera-y component (deg) of rotvec(R_pred R_gt^T) over the closed loop
  rootG   zgz_global RA (mm) of the prediction with its fingers replaced by the ground truth (root only); rootL on local
Written rule (thresholds fixed before any D2 reading):
  MECH    |yb(D2)| <= |yb(D1)| - YB_DROP and rootG(D2) <= rootG(D1) - ROOTG_DROP
  METRIC  RA(D2) <= RA(D1) - RA_DROP and MPJPE-G(D2) <= MPJPE-G(D1) - G_DROP and MPJPE-L(D2) <= MPJPE-L(D1) + L_TOL
  D2_SCREEN_PASS = MECH and METRIC; D2_MECH_ONLY; D2_METRIC_ONLY; D2_NULL.
Validity: the per-endpoint RA recomputed here from the saved states must equal the engine's stored values (<= 1e-3 mm)
for every reading used; otherwise INVALID and no verdict.
Descriptive beside it: the packet form, B2 + rol5 (streaming, TPD), S9's registered S1 rules on the from-scratch models
(tools/pmr/analyze.py `rules`, controls D1's R-noprop / R-mlp), retention and repeated-packet rows.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

import numpy as np  # noqa: E402

from tools.pmr.common import NOISE_RA, OUT, S43_ENGINE, ZGZ, now, read, require, save  # noqa: E402

SEED = 3407
YB_DROP, ROOTG_DROP = 1.0, 0.4
RA_DROP, G_DROP, L_TOL = 0.3, 0.5, 0.5
VALID_MM = 1e-3
DATA = Path("/data1/lyq/code/mesh/EventHands/data/hand_data51/val")
TPD_READINGS = ROOT / "outputs/tpd/20261105/readings"
READINGS = OUT / "readings"
TAGS = {"d2_str": f"pmr_d1hubfull_rol5_s{SEED}__str", "d1_str": f"pmr_d1r_rol5_s{SEED}__str",
        "d2_pkt": f"pmr_d1hubfull_rol5_s{SEED}", "d1_pkt": f"pmr_d1r_rol5_s{SEED}",
        "d2_base": f"pmr_d1hubfull_s{SEED}", "d1_base": f"pmr_d1r_s{SEED}"}
NAMES = {"d2_str": "D2 + rol5（流式）", "d1_str": "D1 R + rol5（流式，EvHandTrack 种子 3407）",
         "d2_pkt": "D2 + rol5（分包）", "d1_pkt": "D1 R + rol5（分包）", "d2_base": "D2 从头（分包）",
         "d1_base": "D1 R 从头（分包）", "b2r_str": "B2 + rol5（流式，TPD）"}
VERDICTS = {(True, True): "D2_SCREEN_PASS", (True, False): "D2_MECH_ONLY", (False, True): "D2_METRIC_ONLY",
            (False, False): "D2_NULL"}


def rule(d1, d2):
    """The written verdict (pure). d1 / d2: {"five": [L, G, VL, VG, RA], "yb": deg, "rootG": mm}."""
    mech = bool(abs(d2["yb"]) <= abs(d1["yb"]) - YB_DROP and d2["rootG"] <= d1["rootG"] - ROOTG_DROP)
    metric = bool(d2["five"][4] <= d1["five"][4] - RA_DROP and d2["five"][1] <= d1["five"][1] - G_DROP
                  and d2["five"][0] <= d1["five"][0] + L_TOL)
    return {"mech": mech, "metric": metric, "verdict": VERDICTS[(mech, metric)],
            "numbers": {"yb_d1": d1["yb"], "yb_d2": d2["yb"], "rootG_d1": d1["rootG"], "rootG_d2": d2["rootG"],
                        "five_d2_minus_d1": [b - a for a, b in zip(d1["five"], d2["five"])]}}


def five(report):
    s = report["per_sequence"]
    return [s["zgz_local"]["mpjpe_ra_mm"], s["zgz_global"]["mpjpe_ra_mm"], s["zgz_local"]["mpvpe_ra_mm"],
            s["zgz_global"]["mpvpe_ra_mm"], report["overall"]["mpjpe_ra_mm"]]


class Kin:
    """MANO forward kinematics on the CPU for the two zgz recordings."""

    def __init__(self):
        import torch
        from mano_layer import ManoLayer
        from semkine import eval_track as ET
        torch.set_num_threads(16)
        self.torch, self.ET, self.dev = torch, ET, torch.device("cpu")
        self.mano = ManoLayer(str(ROOT / "assets/mano_right.npz"), add_mean=False).to(self.dev).eval()
        self.betas = {s: torch.tensor(np.asarray(np.load(DATA / f"{s}_aux.npz", allow_pickle=True)["betas"],
                                                 np.float32)).view(1, -1) for s in ZGZ}

    def ra(self, seq, pred, target):
        """per-endpoint MPJPE-RA (mm)."""
        with self.torch.no_grad():
            pj, _ = self.ET.mano_fk(self.mano, np.ascontiguousarray(pred, np.float32), self.betas[seq], self.dev)
            gj, _ = self.ET.mano_fk(self.mano, np.ascontiguousarray(target, np.float32), self.betas[seq], self.dev)
        return np.linalg.norm(self.ET.root_align(pj) - self.ET.root_align(gj), axis=-1).mean(-1) * 1000


def root_error_deg(pred, target):
    """rotvec (deg) of R_pred R_gt^T, camera axes; state = [transl 3 | root rotvec 3 | fingers 45]."""
    from scipy.spatial.transform import Rotation as Rot
    p, t = np.asarray(pred, np.float64), np.asarray(target, np.float64)
    return np.degrees((Rot.from_rotvec(p[:, 3:6]) * Rot.from_rotvec(t[:, 3:6]).inv()).as_rotvec())


def mechanism(kin, directory):
    """yb, rootG, rootL and the validity deviation of one e1 reading."""
    z = np.load(Path(directory) / "e1" / "states.npz", allow_pickle=False)
    out, dev = {}, 0.0
    for s in ZGZ:
        pr, t = z[f"{s}/prediction"], z[f"{s}/target"]
        dev = max(dev, float(np.abs(kin.ra(s, pr, t) - z[f"{s}/mpjpe_ra_mm"]).max()))
        a = pr.copy()
        a[:, 6:] = t[:, 6:]
        e = root_error_deg(pr, t)
        out[s] = {"root_only_ra": float(kin.ra(s, a, t).mean()), "signed_xyz_deg": [float(v) for v in e.mean(0)],
                  "abs_xyz_deg": [float(v) for v in np.abs(e).mean(0)]}
    return {"yb": out["zgz_global"]["signed_xyz_deg"][1], "rootG": out["zgz_global"]["root_only_ra"],
            "rootL": out["zgz_local"]["root_only_ra"], "per_sequence": out, "max_abs_dev_mm": dev}


def fmt(v, d=2):
    return "–" if v is None else f"{v:.{d}f}"


def screen():
    dst = OUT / "decisions" / "s9x_d2_screen.json"
    require(not dst.exists(), "written once: " + str(dst))
    from tools.pmr import analyze as A0
    kin = Kin()
    dirs = {k: READINGS / t for k, t in TAGS.items()}
    dirs["b2r_str"] = TPD_READINGS / f"tpd_rol5_s{SEED}__str"
    rows = {}
    for k, d in dirs.items():
        rows[k] = {"reading": str(d), "five": five(read(d / "e1" / "report.json")), **mechanism(kin, d)}
    valid = all(r["max_abs_dev_mm"] <= VALID_MM for r in rows.values())
    verdict = rule(rows["d1_str"], rows["d2_str"]) if valid else {"verdict": "INVALID"}
    M = A0.mech()
    s1_rows = {"b2": M.row("pfo2", A0.B2_TAG), "r": M.row("pmr", TAGS["d2_base"]), "n": M.row("pmr", f"pmr_d1n_s{SEED}"),
               "m": M.row("pmr", f"pmr_d1m_s{SEED}"), "d1r": M.row("pmr", TAGS["d1_base"]),
               "d2_rol5": M.row("pmr", TAGS["d2_pkt"]), "d1_rol5": M.row("pmr", TAGS["d1_pkt"])}
    s1 = A0.rules(s1_rows["b2"], s1_rows["r"], s1_rows["n"], s1_rows["m"])
    L = ["## PMR D2（d1hubfull）单种子筛选（种子 3407；PFO 引擎 E1；写定规则 `#pmr-d2-screen-prereg-20261110`）", "",
         f"S43（引擎三种子）{' / '.join(fmt(x) for x in S43_ENGINE)}；单种子噪声底约 {NOISE_RA} mm RA。", "",
         "| 行 | E1 五项（MPJPE-L / G / MPVPE-L / G / RA） | global 绕相机 y 根偏置（°） | 只算根 RA local / global |",
         "|---|---|---|---|"]
    for k in ("d2_str", "d1_str", "b2r_str", "d2_pkt", "d1_pkt", "d2_base", "d1_base"):
        r = rows[k]
        L.append(f"| {NAMES[k]} | {' / '.join(fmt(x) for x in r['five'])} | {r['yb']:+.2f} | {fmt(r['rootL'])} / "
                 f"{fmt(r['rootG'])} |")
    L += ["", f"有效性（由状态重算的逐端点 RA 与引擎存值之差 ≤ {VALID_MM} mm）：{valid}"
              f"（最大 {max(r['max_abs_dev_mm'] for r in rows.values()):.2e} mm）。", ""]
    if valid:
        n = verdict["numbers"]
        d = n["five_d2_minus_d1"]
        L += ["| 判据 | 数值 | 结果 |", "|---|---|---|",
              f"| MECH：|yb| 降 ≥ {YB_DROP}° 且只算根 global 降 ≥ {ROOTG_DROP} mm | yb {n['yb_d1']:+.2f} → {n['yb_d2']:+.2f}；"
              f"只算根 {fmt(n['rootG_d1'])} → {fmt(n['rootG_d2'])} | {verdict['mech']} |",
              f"| METRIC：RA 降 ≥ {RA_DROP}、MPJPE-G 降 ≥ {G_DROP}、MPJPE-L 不差超过 {L_TOL} | D2 − D1："
              f"{' / '.join(f'{x:+.2f}' for x in d)} | {verdict['metric']} |", ""]
    L += [f"**{verdict['verdict']}**", "",
          "S9 登记的 S1 规则（从头训练的模型，分包形式；对照 D1 的 R-noprop / R-mlp；描述）：",
          f"(a) {s1['a_necessary']}（noprop − D2 = {fmt(s1['numbers']['gap_noprop_minus_r'])}），"
          f"(b) {s1['b_sufficient']}（mlp − D2 = {fmt(s1['numbers']['mlp_minus_r'])}），"
          f"(c) {s1['c_no_regress']}（D2 − B2 = {fmt(s1['numbers']['r_minus_b2'])}），健全性 {s1['health']} → {s1['verdict']}。",
          "", "| 行（分包） | E3 | 放大 | local R15 / F15 / T10 / T30 | C |", "|---|---|---|---|---|"]
    for k, name in (("r", "D2 从头"), ("d1r", "D1 R 从头"), ("d2_rol5", "D2 + rol5"), ("d1_rol5", "D1 R + rol5")):
        r = s1_rows[k]
        ret = " / ".join(fmt(r["retention"][q]) for q in ("R15", "F15", "T10", "T30")) if "retention" in r else "–"
        L.append(f"| {name} | {fmt(r.get('e3'))} | {fmt(r['amp'][0]) if 'amp' in r else '–'} | {ret} | "
                 f"{fmt(r.get('C'), 3)} |")
    text = "\n".join(L) + "\n"
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    (OUT / "analysis" / "s9x_d2_screen.md").write_text(text)
    save(dst, {"seed": SEED, "rows": rows, "valid": valid, "verdict": verdict, "s1_registered": s1, "s1_rows": s1_rows,
               "thresholds": {"yb_drop_deg": YB_DROP, "rootG_drop_mm": ROOTG_DROP, "ra_drop_mm": RA_DROP,
                              "g_drop_mm": G_DROP, "l_tol_mm": L_TOL, "valid_mm": VALID_MM},
               "written_at_utc": now()}, exclusive=True)
    print(text)
    print(verdict)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("verb", choices=("screen",))
    p.parse_args(argv)
    screen()


if __name__ == "__main__":
    main()
