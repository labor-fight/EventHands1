#!/usr/bin/env python3
"""Real-study result tables (docs/EXPERIMENT_RECORDS.md#real-prereg-20261110).

From the E1 readings outputs/real/20261110/readings/cmp_<arm>_s<seed>/e1/report.json: per light (normal / dark), the
user's columns -- MPJPE-local / -global, MPVPE-local / -global (root-aligned, pooled over the test subjects' local resp.
global sequences of that light, weighted by endpoints), RA = root-aligned MPJPE pooled over all four sequences of that
light (the CMP rows' "overall") -- three-seed means, the RA cell with the seeds. Latency / FLOPs / Params: the real
study's own cost reading (outputs/real/20261110/costs/<tag>/report.json) when it exists, else for EvHandPose (128 x 128
crop) and EventHPE (256 x 192 frame; the CMP port ran 240 x 180 padded to the same 256 x 192 tensor) the CMP reading of
the same network and tensor size; EventHands runs on 346 x 260 here, so only its own reading counts; "-" until measured. The dev subject (wh) gets
the same table, descriptive. Writes table.md and rows.json next to the readings.

    python tools/real/table.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]

from tools.real import ns  # noqa: E402

ARMS = ("eho", "epx", "ehpe", "fme")
LABELS = {"eho": "EventHands（ICCV 2021 原版配方，整幅 346×260）", "epx": "EvHandPose（346×260 上裁 128×128，在线训练，CMP 修订 6）",
          "ehpe": "EventHPE（ICCV 2021，官方 256 帧 = 256×192，MANO，光流取冻结的 EvHandPose FlowNet）",
          "fme": "FastMETRO-Event（EvRGBHand 官方事件版，整幅 346×260，逐帧，MPJPE 取关节 token）"}
CMP_COST = {"epx": "cmp_epx_s3407__default", "ehpe": "cmp_ehpe_s3407__default"}
OWN_COST = {"eho": "cmp_eho_s3407__default", "epx": "cmp_epx_s3407__default", "ehpe": "cmp_ehpe_s3407__default",
            "fme": "cmp_fme_s3407__default"}
FIELDS = ("mpjpe_ra_mm", "mpvpe_ra_mm")
HEADER = ("| 网络结构 | MPJPE-local | MPJPE-global | MPVPE-local | MPVPE-global | RA-MPJPE(递推) | Latency | FLOPs/step "
          "| Params |\n|---|---|---|---|---|---|---|---|---|")


def pooled(blocks, seqs):
    n = sum(blocks[s]["n"] for s in seqs)
    return {f: sum(blocks[s][f] * blocks[s]["n"] for s in seqs) / n for f in FIELDS}


def cell(blocks, subjects, light):
    suf = "_dark" if light == "dark" else ""
    loc = [f"{s}_local{suf}" for s in subjects]
    glo = [f"{s}_global{suf}" for s in subjects]
    return {"local": pooled(blocks, loc), "global": pooled(blocks, glo), "overall": pooled(blocks, loc + glo)}


def cost(arm):
    p = ns.OUT / "costs" / OWN_COST[arm] / "report.json"
    if not p.exists() and arm in CMP_COST:
        p = ROOT / "outputs/cmp/20261110/costs" / CMP_COST[arm] / "report.json"
    if not p.exists():
        return "- | - | -"
    c = json.loads(p.read_text())
    if c.get("status") != "COMPLETE":
        return "- | - | -"
    return (f"{c['latency_row']['scaled_ms']:.2f} ms | {c['macs_layers']['median'] / 1e9:.3f} G "
            f"| {c['params_total'] / 1e6:.2f} M")


def main():
    rows, lines = {}, []
    for group, subjects in (("test", ("kqj", "lqh")), ("dev", ("wh",))):
        for light in ("normal", "dark"):
            lines += ["", f"**{'测试 kqj、lqh' if group == 'test' else '调参 wh（描述）'} · "
                          f"{'正常光' if light == 'normal' else '暗光'}**", "", HEADER]
            for arm in ARMS:
                per = []
                for seed in ns.SEEDS:
                    p = ns.OUT / "readings" / f"cmp_{arm}_s{seed}" / "e1" / "report.json"
                    if not p.exists():
                        continue
                    r = json.loads(p.read_text())
                    if r.get("status") == "COMPLETE":
                        per.append((seed, cell(r["per_sequence"], subjects, light)))
                if not per:
                    continue
                m = {k: {f: sum(c[k][f] for _, c in per) / len(per) for f in FIELDS} for k in ("local", "global", "overall")}
                rows[f"{group}/{light}/{arm}"] = {"seeds": [s for s, _ in per], "mean": m, "per_seed": dict(per)}
                ra = f"{m['overall']['mpjpe_ra_mm']:.2f}" + ("（" + " / ".join(
                    f"{c['overall']['mpjpe_ra_mm']:.2f}" for _, c in per) + "）" if len(per) > 1 else "")
                tag = "" if len(per) == 3 else f"（{len(per)} 种子）"
                lines.append(f"| {LABELS[arm]}{tag} | {m['local']['mpjpe_ra_mm']:.2f} | {m['global']['mpjpe_ra_mm']:.2f} "
                             f"| {m['local']['mpvpe_ra_mm']:.2f} | {m['global']['mpvpe_ra_mm']:.2f} | {ra} | {cost(arm)} |")
    (ns.OUT / "rows.json").write_text(json.dumps(rows, indent=1) + "\n")
    text = "\n".join(lines).strip() + "\n"
    (ns.OUT / "table.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
