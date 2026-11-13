#!/usr/bin/env python3
"""EvRealHands result tables (docs/EXPERIMENT_RECORDS.md#erh-prereg-20261111), in the user's nine columns.

Scored at the annotation times only (EvRealHands labels are 15 fps; the E1 loop runs on the 50 ms grid with the
interpolated labels only as its start states, and every run is aligned so that every fourth endpoint lands exactly on a
label, tools/erh/convert.py): each reading's per-endpoint root-aligned errors (states.npz, the engine's schema; FastMETRO's
own reading writes the same keys) are kept where the endpoint is a screened label time (aux gt_ms_idx[gt_valid]).
EvRealHands has no local / global sequences; the local / global columns use the per-label motion groups of the official
val8 tiers recomputed for every label (<seq>_motion.npz: 3 = local, ratio >= 1.25; 4 = global, ratio <= 0.8), RA =
every scored label of the group of sequences (quiet / mixed / unscoreable labels included). Tables: test (subjects 1, 3)
per scene (normal, highlight, flash) and all scenes, dev (subject 7, normal only) descriptive. Three-seed means, the RA
cell with the seeds. Cost cells: the same networks on the same tensors as the real study (346 x 260 / 128 x 128 crop /
256 x 192), so the real study's cost readings (tools/real/table.py `cost`, EvHandTrack: tools/real/table2.py).

    python tools/erh/run.py tools.erh.table
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.erh import ns as E  # noqa: E402

R = E.install()
from tools.real import evht as X  # noqa: E402
from tools.real import table as TB  # noqa: E402
from tools.real import table2 as T2  # noqa: E402

ARMS = ("eho", "epx", "ehpe", "fme", "evht")
LABELS = {**{a: TB.LABELS[a] for a in ("eho", "epx", "ehpe", "fme")},
          "evht": "EvHandTrack（运动学树 D1 + rol5，整幅 346×260，流式形式；路由器在 EvRealHands 训练集上训练）"}
FIELDS = ("mpjpe_ra_mm", "mpvpe_ra_mm")
SCENES = ("normal", "highlight", "flash")
LOCAL, GLOBAL = 3, 4
REAL_OUT = ROOT / "outputs/real/20261110"


def reading(arm, seed):
    if arm == "evht":
        return X.report("d1r_rol5", seed).parent
    return R.OUT / "readings" / f"cmp_{arm}_s{seed}" / "e1"


def seq_info(seq):
    root = R.CANVAS["n346"]["root"]
    m = json.loads(R.splits("n346").read_text())
    d = {**m["val"]["legacy_dir"], **m["test"]["legacy_dir"]}[seq]
    aux = np.load(root / d / f"{seq}_aux.npz", allow_pickle=True)
    mot = np.load(root / d / f"{seq}_motion.npz", allow_pickle=True)
    labels = np.asarray(aux["gt_ms_idx"])[np.asarray(aux["gt_valid"], bool)]
    group = dict(zip(np.asarray(mot["label_ms"]).tolist(), np.asarray(mot["motion_group"]).tolist()))
    return {"labels": set(labels.tolist()), "group": group, "scene": str(aux["scene"])}


_INFO = {}


def info(seq):
    if seq not in _INFO:
        _INFO[seq] = seq_info(seq)
    return _INFO[seq]


def cell(states, seqs):
    acc = {k: {f: [] for f in FIELDS} for k in ("local", "global", "overall")}
    for seq in seqs:
        inf = info(seq)
        end = states[f"{seq}/end_ms"]
        keep = np.array([int(e) in inf["labels"] for e in end])
        grp = np.array([inf["group"].get(int(e), -1) for e in end])
        for f in FIELDS:
            v = np.asarray(states[f"{seq}/{f}"], np.float64)
            acc["overall"][f].append(v[keep])
            acc["local"][f].append(v[keep & (grp == LOCAL)])
            acc["global"][f].append(v[keep & (grp == GLOBAL)])
    out = {}
    for k, d in acc.items():
        out[k] = {f: float(np.concatenate(d[f]).mean()) if sum(len(x) for x in d[f]) else float("nan") for f in FIELDS}
        out[k]["n"] = int(sum(len(x) for x in d["mpjpe_ra_mm"]))
    return out


def per_seed(arm, seqs):
    out = []
    for seed in R.SEEDS:
        d = reading(arm, seed)
        rep = d / "report.json"
        if not rep.exists() or json.loads(rep.read_text()).get("status") != "COMPLETE":
            continue
        with np.load(d / "states.npz", allow_pickle=False) as z:
            states = {k: z[k] for k in z.files if k.endswith(("/end_ms", "/mpjpe_ra_mm", "/mpvpe_ra_mm"))}
        out.append((seed, cell(states, seqs)))
    return out


def cost(arm):
    if arm == "evht":
        p = REAL_OUT / "costs" / T2.EVHT_COST / "report.json"
    else:
        p = REAL_OUT / "costs" / TB.OWN_COST[arm] / "report.json"
        if not p.exists() and arm in TB.CMP_COST:
            p = ROOT / "outputs/cmp/20261110/costs" / TB.CMP_COST[arm] / "report.json"
    if not p.exists():
        return "- | - | -"
    c = json.loads(p.read_text())
    if c.get("status") != "COMPLETE":
        return "- | - | -"
    return (f"{c['latency_row']['scaled_ms']:.2f} ms | {c['macs_layers']['median'] / 1e9:.3f} G "
            f"| {c['params_total'] / 1e6:.2f} M")


def main():
    rows, lines = {}, []
    groups = [("test", f"测试（受试者 1、3）· {s}", [q for q in R.TEST if info(q)["scene"] == s]) for s in SCENES]
    groups += [("test", "测试（受试者 1、3）· 全部场景", list(R.TEST)), ("dev", "调参（受试者 7，均为 normal；描述）", list(R.DEV))]
    for group, title, seqs in groups:
        lines += ["", f"**{title}**（{len(seqs)} 条序列；只在标注时刻计分）", "", TB.HEADER]
        for arm in ARMS:
            per = per_seed(arm, seqs)
            if not per:
                continue
            m = {k: {f: float(np.mean([c[k][f] for _, c in per])) for f in FIELDS} for k in ("local", "global", "overall")}
            rows[f"{title}/{arm}"] = {"seeds": [s for s, _ in per], "mean": m, "per_seed": {str(s): c for s, c in per}}
            ra = f"{m['overall']['mpjpe_ra_mm']:.2f}" + ("（" + " / ".join(
                f"{c['overall']['mpjpe_ra_mm']:.2f}" for _, c in per) + "）" if len(per) > 1 else "")
            tag = "" if len(per) == 3 else f"（{len(per)} 种子）"
            lines.append(f"| {LABELS[arm]}{tag} | {m['local']['mpjpe_ra_mm']:.2f} | {m['global']['mpjpe_ra_mm']:.2f} "
                         f"| {m['local']['mpvpe_ra_mm']:.2f} | {m['global']['mpvpe_ra_mm']:.2f} | {ra} | {cost(arm)} |")
    (R.OUT / "rows.json").write_text(json.dumps(rows, indent=1) + "\n")
    text = "\n".join(lines).strip() + "\n"
    (R.OUT / "table.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
