#!/usr/bin/env python3
"""Router-level comparison, r_s2dx2 (part loss) vs r_s2dx2np (no part loss) (docs/EXPERIMENT_RECORDS.md#nopart-prereg-20261113).
Zero training, descriptive. Both routers see the same nodes: EvHandTrack seed 3407's packet nodes with the GT prior on
zgz_local, zgz_global, ylf_local (held out of both routers' training) and lyq_local (a training subject), XAG labels.

    CUDA_VISIBLE_DEVICES=k python tools/nopart/router_compare.py all        captures (GPU) + table
    python tools/nopart/router_compare.py table                             table from existing captures (CPU)

Per recording, on the labelled nodes (both routers on the same nodes):
  c_i error         mean |c_i - label template point| (mm), the raw coordinate head
  decoded error     the deployed decode: r_s2dx2 = part argmax then the nearest centroid inside that part; r_s2dx2np =
                    the nearest centroid over all faces; mean |decoded template point - label point| (mm)
  part / finger     16-part and finger-level accuracy of the deployed decode
  coherent offset   per (packet, labelled part) with >= 10 nodes, |mean decoded error vector| (mm), averaged
On all nodes: on-hand recall P(selected | labelled) and precision P(labelled | selected).
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/nopart/20261113/router_compare"
PY = "/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python"
ROUTERS = {"r_s2dx2": ROOT / "outputs/rfsu/20261113/sim/routers/r_s2dx2/r_s2dx2_final.ckpt",
           "r_s2dx2np": ROOT / "outputs/nopart/20261113/sim/routers/r_s2dx2np/r_s2dx2np_final.ckpt"}
LABEL = {"r_s2dx2": "有部位损失（论文的 r_s2dx2）", "r_s2dx2np": "无部位损失（r_s2dx2np）"}


def finger(p):
    import numpy as np
    p = p.astype(int)
    return np.where(p == 0, 0, (p - 1) // 3 + 1)


def coherent(err, pkt, lpart, min_n=10):
    import numpy as np
    key = pkt.astype(np.int64) * 16 + lpart
    order = np.argsort(key, kind="stable")
    k, e = key[order], err[order]
    starts = np.r_[0, np.flatnonzero(np.diff(k)) + 1]
    counts = np.diff(np.r_[starts, len(k)])
    sums = np.add.reduceat(e, starts, axis=0)
    keep = counts >= min_n
    return float(np.mean(np.linalg.norm(sums[keep] / counts[keep, None], axis=1)) * 1000)


def metrics(z, si):
    import numpy as np
    m = z["seq"] == si
    l_ok, r_sel = z["l_ok"][m], z["r_sel"][m]
    lab = l_ok
    lp, rp, cp, lpart, rpart = z["lp"][m][lab], z["rp"][m][lab], z["cp"][m][lab], z["lpart"][m][lab], z["rpart"][m][lab]
    pkt = z["pkt"][m][lab]
    return {"n_labelled": int(lab.sum()),
            "c_err": float(np.mean(np.linalg.norm(cp - lp, axis=1)) * 1000),
            "dec_err": float(np.mean(np.linalg.norm(rp - lp, axis=1)) * 1000),
            "part_acc": float(np.mean(rpart == lpart)),
            "finger_acc": float(np.mean(finger(rpart) == finger(lpart))),
            "coherent": coherent(rp - lp, pkt, lpart),
            "recall": float(r_sel[l_ok].mean()),
            "precision": float(l_ok[r_sel].mean()) if r_sel.any() else float("nan")}


def capture(name):
    out = OUT / "capture" / f"{name}.npz"
    if out.exists():
        return
    env = dict(os.environ)
    r = subprocess.run([PY, str(ROOT / "tools/nopart/router_capture.py"), "--name", name, "--rkind", "pfr", "--ckpt",
                        str(ROUTERS[name])], cwd=str(ROOT), env=env)
    if r.returncode != 0 or not out.exists():
        raise SystemExit(f"capture {name} failed rc={r.returncode}")


def table():
    import numpy as np
    res, seqs = {}, None
    for name in ROUTERS:
        z = np.load(OUT / "capture" / f"{name}.npz")
        seqs = [str(s) for s in z["seqs"]]
        res[name] = {s: metrics(z, i) for i, s in enumerate(seqs)}
    cols = [("c_err", "c_i 误差 mm", "{:.2f}"), ("dec_err", "解码后表面点误差 mm", "{:.2f}"),
            ("part_acc", "16 部位准确率", "{:.3f}"), ("finger_acc", "手指级准确率", "{:.3f}"),
            ("coherent", "相干偏移 mm", "{:.2f}"), ("recall", "手上召回", "{:.3f}"), ("precision", "手上精确率", "{:.3f}")]
    L = ["## 路由器层面：有 / 无部位损失（零训练读数；同一批节点；XAG 标签）", "",
         "每格为 有部位损失 / 无部位损失；误差类越低越好，准确率类越高越好。ylf 是两个路由器训练时都留出的受试者，lyq 在训练集内。", "",
         "| 录制 | " + " | ".join(c[1] for c in cols) + " |", "|---|" + "---|" * len(cols)]
    a, b = res["r_s2dx2"], res["r_s2dx2np"]
    for s in seqs:
        L.append(f"| {s} | " + " | ".join(f"{c[2].format(a[s][c[0]])} / {c[2].format(b[s][c[0]])}" for c in cols) + " |")
    text = "\n".join(L) + "\n"
    (OUT / "table.md").write_text(text)
    import json
    (OUT / "metrics.json").write_text(json.dumps(res, indent=1) + "\n")
    print(text)


def main():
    verb = sys.argv[1] if len(sys.argv) > 1 else "table"
    OUT.mkdir(parents=True, exist_ok=True)
    if verb == "all":
        for name in ROUTERS:
            capture(name)
        table()
        (OUT / "DONE").write_text("ok\n")
    else:
        table()


if __name__ == "__main__":
    main()
