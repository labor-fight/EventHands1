#!/usr/bin/env python3
"""ABL ablation table and its two descriptive readouts (docs/EXPERIMENT_RECORDS.md#abl-prereg-20261111). CPU only.

    python -m tools.pmr.abltable table [--path 1|2] [--out outputs/pmr/20261109/analysis/abl_table.md]

  table   zgz_local / zgz_global MPJPE and MPVPE (closed loop E1, streaming form; mm, root-aligned), the full model and the
          six variants, mean over the seeds read so far (per-seed values in parentheses)
  under   zgz_local joint error grouped by whether the joint's part had fewer than 10 labelled nodes in the packet
          (labels = XAG pseudo-labels at the nodes, identical for every variant: router_capture/r_s2d.npz)
  curve   error against the time since the last initialisation (packet form), full model vs (f) vs no fine-tune
The path defaults to the one the gates opened (decisions/abl_gate_*.json).
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

from tools.pmr.common import OUT, read  # noqa: E402

READ = OUT / "readings"
CAP = ROOT / "outputs/evhandtrack/analysis/posthoc_20261110/router_capture/r_s2d.npz"
DATA = Path("/data1/lyq/code/mesh/EventHands/data/hand_data51/val")
SEEDS = (3407, 3408, 3409)
UNDER = 10
#: OpenPose joint at the distal end of every MANO part (palm: the four finger MCPs)
PART_JOINTS = {0: (5, 9, 13, 17), 1: (6,), 2: (7,), 3: (8,), 4: (10,), 5: (11,), 6: (12,), 7: (18,), 8: (19,), 9: (20,),
               10: (14,), 11: (15,), 12: (16,), 13: (2,), 14: (3,), 15: (4,)}
BINS = (0, 250, 550, 1000, 2000, 4000, 8000, 10 ** 9)
ROWS = {2: [("完整 EvHandTrack（D1 R + r_s2d + rol5）", "d1r_rol5"),
            ("(a) 几何邻近关联替代 ESMNet 表面对应", "abgeo_rol5"),
            ("(b) 去除事件—前态几何残差", "abnr_rol5"),
            ("(c) 部位特征拼接 MLP 替代图传播与局部读出", "d1m_rol5"),
            ("(d) 前态几何门控 → 静态边类型门控", "d1g0_rol5"),
            ("(e) 取消整手节点与部位节点的通信", "d1h0_rol5"),
            ("(f) 单步训练替代多步闭环展开训练", "d1r_rol1")],
        1: [("完整模型（D1 R + RDR v3 + rol5）", "d1rv3_rol5"),
            ("(a) 几何邻近关联替代 ESMNet 表面对应", "abgeo_rol5"),
            ("(b) 去除事件—前态几何残差", "abnrv3_rol5"),
            ("(c) 部位特征拼接 MLP 替代图传播与局部读出", "abmv3_rol5"),
            ("(d) 前态几何门控 → 静态边类型门控", "abg0v3_rol5"),
            ("(e) 取消整手节点与部位节点的通信", "abh0v3_rol5"),
            ("(f) 单步训练替代多步闭环展开训练", "d1rv3_rol1")]}
BASE = {2: "d1r", 1: "d1rv3"}


def path_opened():
    d = OUT / "decisions"
    if (d / "abl_gate_p1_v3.json").exists() and read(d / "abl_gate_p1_v3.json").get("open"):
        return 1
    if (d / "abl_gate_p2.json").exists() and read(d / "abl_gate_p2.json").get("open"):
        return 2
    return None


def report(tag):
    p = READ / tag / "e1" / "report.json"
    return read(p) if p.exists() else None


def cells(name):
    out = {}
    for s in SEEDS:
        r = report(f"pmr_{name}_s{s}__str")
        if r:
            q = r["per_sequence"]
            out[s] = (q["zgz_local"]["mpjpe_ra_mm"], q["zgz_local"]["mpvpe_ra_mm"], q["zgz_global"]["mpjpe_ra_mm"],
                      q["zgz_global"]["mpvpe_ra_mm"])
    return out


def fmt(vals, k):
    if not vals:
        return "–"
    m = np.mean([v[k] for v in vals.values()])
    return f"{m:.2f}" if len(vals) == 1 else f"{m:.2f}（{' / '.join(f'{vals[s][k]:.2f}' for s in sorted(vals))}）"


class Joints:
    def __init__(self):
        import torch
        from mano_layer import ManoLayer
        from semkine import eval_track as ET
        torch.set_num_threads(16)
        self.torch, self.ET = torch, ET
        self.mano = ManoLayer(str(ROOT / "assets/mano_right.npz"), add_mean=False).eval()
        self.betas = {s: torch.tensor(np.asarray(np.load(DATA / f"{s}_aux.npz", allow_pickle=True)["betas"], np.float32))
                      .view(1, -1) for s in ("zgz_local", "zgz_global")}

    def errors(self, seq, pred, target):
        """(n, 21) root-aligned and (n, 21) palm-aligned joint errors in mm. Palm-aligned: the prediction rigidly fitted
        (Kabsch) on the five palm joints 0, 5, 9, 13, 17, so the root rotation error is removed and what is left is the
        articulation of the fingers."""
        with self.torch.no_grad():
            pj, _ = self.ET.mano_fk(self.mano, np.ascontiguousarray(pred, np.float32), self.betas[seq], "cpu")
            gj, _ = self.ET.mano_fk(self.mano, np.ascontiguousarray(target, np.float32), self.betas[seq], "cpu")
        pj, gj = np.asarray(pj, np.float64), np.asarray(gj, np.float64)
        ra = np.linalg.norm((pj - pj[:, :1]) - (gj - gj[:, :1]), axis=-1) * 1000
        idx = [0, 5, 9, 13, 17]
        a, b = pj[:, idx], gj[:, idx]
        ca, cb = a.mean(1, keepdims=True), b.mean(1, keepdims=True)
        H = np.einsum("nki,nkj->nij", a - ca, b - cb)
        U, _, Vt = np.linalg.svd(H)
        dt = np.sign(np.linalg.det(np.einsum("nij,njk->nik", U, Vt)))
        D = np.repeat(np.eye(3)[None], len(pj), 0)
        D[:, 2, 2] = dt
        Rm = np.einsum("nij,njk,nkl->nil", U, D, Vt)                       # a @ Rm ~ b
        pa = np.einsum("nkj,nji->nki", pj - ca, Rm) + cb
        return ra, np.linalg.norm(pa - gj, axis=-1) * 1000


def label_counts(seq="zgz_local"):
    z = np.load(CAP)
    si = [str(s) for s in z["seqs"]].index(seq)
    m = z["seq"] == si
    pkt, ok, part = z["pkt"][m], z["l_ok"][m], z["lpart"][m].astype(int)
    n = int(pkt.max()) + 1
    c = np.zeros((n, 16), np.int64)
    np.add.at(c, (pkt[ok], part[ok]), 1)
    return c


def under(rows, J):
    counts = label_counts("zgz_local")
    lines = ["| 变体 | 观测不足 / 充分部位的关节误差（去根对齐） | 观测不足 / 充分部位的手指关节误差（手掌对齐） | 观测不足的（包, 部位）占比 |",
             "|---|---|---|---|"]
    for label, name in rows:
        vals = []
        for s in SEEDS:
            p = READ / f"pmr_{name}_s{s}__str" / "e1" / "states.npz"
            if not p.exists():
                continue
            z = np.load(p)
            e, ep = J.errors("zgz_local", z["zgz_local/prediction"], z["zgz_local/target"])
            n = min(len(e), len(counts))
            lo, hi, plo, phi = [], [], [], []
            for part, joints in PART_JOINTS.items():
                few = counts[:n, part] < UNDER
                ej, pj = e[:n][:, list(joints)].mean(1), ep[:n][:, list(joints)].mean(1)
                lo.append(ej[few]); hi.append(ej[~few])
                if part:
                    plo.append(pj[few]); phi.append(pj[~few])
            lo, hi, plo, phi = (np.concatenate(x) for x in (lo, hi, plo, phi))
            vals.append((lo.mean(), hi.mean(), plo.mean(), phi.mean(), len(lo) / (len(lo) + len(hi))))
        if vals:
            v = np.mean(vals, 0)
            lines.append(f"| {label} | {v[0]:.2f} / {v[1]:.2f} | {v[2]:.2f} / {v[3]:.2f} | {v[4]:.2f} |")
        else:
            lines.append(f"| {label} | – | – | – |")
    return lines


def curve(path):
    rows = [("完整模型", ROWS[path][0][1]), ("(f) 单步训练", ROWS[path][-1][1]), ("不做微调的基座", BASE[path])]
    head = " | ".join(f"{a / 1000:g}–{b / 1000:g} s" if b < 10 ** 9 else f"> {a / 1000:g} s" for a, b in zip(BINS[:-1], BINS[1:]))
    lines = [f"| 变体（分包形式，种子 3407） | 录制 | {head} |", "|---|---|" + "---|" * (len(BINS) - 1)]
    for label, name in rows:
        p = READ / f"pmr_{name}_s3407" / "e1" / "states.npz"
        if not p.exists():
            lines.append(f"| {label} | – |" + " – |" * (len(BINS) - 1))
            continue
        z = np.load(p)
        for seq in ("zgz_local", "zgz_global"):
            el, ra = z[f"{seq}/elapsed_ms"], z[f"{seq}/mpjpe_ra_mm"]
            v = [ra[(el >= a) & (el < b)].mean() if ((el >= a) & (el < b)).any() else float("nan")
                 for a, b in zip(BINS[:-1], BINS[1:])]
            lines.append(f"| {label} | {seq[4:]} | " + " | ".join(f"{x:.2f}" for x in v) + " |")
    return lines


def table(path, out):
    rows = ROWS[path]
    L = [f"## 消融表（路线 {path}；闭环 E1，流式形式；zgz，去根对齐，mm）", "",
         "| 变体 | MPJPE（Local） | MPVPE（Local） | MPJPE（Global） | MPVPE（Global） | 种子 |", "|---|---|---|---|---|---|"]
    for label, name in rows:
        v = cells(name)
        L.append(f"| {label} | {fmt(v, 0)} | {fmt(v, 1)} | {fmt(v, 2)} | {fmt(v, 3)} | {len(v)} |")
    J = Joints()
    L += ["", f"### 局部运动下观测不足部位的重建（zgz_local；每包该部位标注节点 < {UNDER} 记为观测不足；关节误差 mm）", ""]
    L += under(rows, J)
    L += ["", "### 长时误差曲线（MPJPE-RA，按距上次初始化的时间分箱，mm）", ""]
    L += curve(path)
    text = "\n".join(L) + "\n"
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text)
    print(text)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("verb", choices=("table",))
    p.add_argument("--path", type=int, choices=(1, 2))
    p.add_argument("--out")
    a = p.parse_args(argv)
    path = a.path or path_opened()
    if path is None:
        sys.exit("no path opened yet; pass --path to preview")
    table(path, a.out)


if __name__ == "__main__":
    main()
