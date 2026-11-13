#!/usr/bin/env python3
"""CMP main rows (docs/EXPERIMENT_RECORDS.md#cmp-prereg-20261110): outputs/semkine/cmp_<method>[_<form>]_main_row.json in
the schema `tools/report_table.py` reads (three-seed means under the historical key `two_seed_mean`, per-seed readings,
and the same-block cost of seed 3407), then the comparison table.

    python -m tools.cmp.rows write            # every row whose readings and cost reading exist
    python -m tools.cmp.rows table            # the five-method table (tools/report_table.py)

Rows: eho (EventHands as published, amendment 4), epc (EvHandPose with its online hand crop, amendment 1), emrol5 (event encoder + previous-state MLP after rol5), eg (event
encoder + GRU, sliding form), evhandtrack (the identity's streaming readings, re-costed in this block; amendments 5 / 5b: the
kinematic-tree identity, PMR D1 + rol5); descriptive: b2rol5 (B2 + rol5, the EvHandTrack row before amendment 5),
eh (the EventHands network on B2's training contract, the pre-registered port; main row until amendment 4), eho__kf /
eho__kff (the published pipeline's Kalman filters), eh__w50, em (first stage), eg__rolling, ep (EvHandPose on the full
frame, the pre-registered port), and s43 (S43 engine three-seed accuracy with its executor in this block).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.cmp.common import EVHANDTRACK, OUT, SEEDS, now, read, require, save  # noqa: E402

ROWS = ROOT / "outputs/semkine"
#: amendment 5 (#cmp-amend5-20261110): B2 + rol5, the identity the EvHandTrack row read before the rename
B2ROL5 = ROOT / "outputs/evhandtrack/b2rol5/identity.json"
SEQ = {"local": "zgz_local", "global": "zgz_global"}
LABELS = {
    "eh": "EventHands 网络 + B2 训练合同（全 45 维姿态、域随机化等，三种子；描述）",
    # amendment 4 (#cmp-amend4-20261110): EventHands as published replaces eh in the main table
    "eho": "EventHands（ICCV 2021 原版配方：PCA6 + 原损失 + 原训练，三种子）",
    "eho__kf": "EventHands 原版 + 官方卡尔曼滤波 filter.py（1 kHz 预测，成本列为单次前向；描述）",
    "eho__kff": "EventHands 原版 + 官方卡尔曼滤波 filter_fast.py（1 kHz 预测，成本列为单次前向；描述）",
    "epc": "EvHandPose 在线裁剪，修订 6 之前的训练（框取 GT、固定 12 包历史；在线发散，描述）",
    "epcl": "EvHandPose 在线裁剪，修订 6 之前的训练，22500 步（描述）",
    # amendment 6 (#cmp-amend6-20261110): the cropped EvHandPose trained for online use replaces epc in the main table
    "epx": "EvHandPose（官方网络 + 在线手部裁剪，训练框取带噪跟踪状态，三种子）",
    "epx__warm": "EvHandPose 在线裁剪（修订 6），段首前历史补满（描述）",
    "epx__gtcrop": "EvHandPose（修订 6），GT 裁剪框（论文评测方式，不可部署；描述）",
    # amendment 7 (#cmp-amend7-20261110): EventHPE, the event-based human-mesh tracker, with MANO
    "ehpe": "EventHPE（ICCV 2021 官方网络与损失，换成 MANO，三种子）",
    "ehpe__carry": "EventHPE，循环状态在段内不重置（描述）",
    # amendment 9 (#cmp-amend9-20261110, descriptive): the official evaluation, 8 steps from the GT pose
    "ehpe__gtinit": "EventHPE，每 8 个包从 GT 姿态重新初始化（论文评测方式，不可部署；描述）",
    # amendment 8 (#cmp-amend8-20261110): FastMETRO with event input (the event-only baseline of EvRGBHand)
    "fme": "FastMETRO-Event（ECCV 2022；EvRGBHand 官方事件配置，全帧，三种子）",
    "ep": "EvHandPose，全帧无裁剪（描述）",
    "emrol5": "Event encoder + previous-state MLP（+ rol5，三种子）",
    "eg": "Event encoder + GRU（三种子）",
    # amendments 5 / 5b (#cmp-amend5-20261110, #cmp-amend5b-20261110): EvHandTrack is the kinematic-tree method PMR D1 +
    # rol5; B2 + rol5 moves to the descriptive table
    "evhandtrack": "EvHandTrack（运动学树 D1 + rol5，三种子，流式形式）",
    "b2rol5": "B2 + rol5（三种子，流式形式；修订 5 前的 EvHandTrack 行，描述）",
    "eh__w50": "EventHands，50 ms 窗口（描述）",
    "em": "Event encoder + previous-state MLP，无 rol5（描述）",
    "eg__rolling": "Event encoder + GRU，段首起滚动隐状态（描述）",
    "s43": "S43（W50 原始，引擎三种子，同块成本；参照）",
    # amendment 3 (#cmp-amend3-20261110): descriptive cold-start diagnostics
    "ep__warm": "EvHandPose，全帧，段首前历史补满（描述）",
    "epc__warm": "EvHandPose，在线裁剪，段首前历史补满（描述）",
    "epc__gtcrop": "EvHandPose，GT 裁剪框（论文评测方式，不可部署；描述）",
    "epcl__warm": "EvHandPose 22500 步，在线裁剪，历史补满（描述）",
    "epcl__gtcrop": "EvHandPose 22500 步，GT 裁剪框（不可部署；描述）",
    "eh__warm": "EventHands，100 ms 窗口不在段首截断（描述）",
    "eg__warm": "Event encoder + GRU，段首前历史补满（描述）",
}
MAIN = ("eho", "epx", "ehpe", "fme", "emrol5", "eg", "evhandtrack")
#: cost reading (outputs/cmp/20261110/costs/<tag>) of every row
#: (descriptive rows carry the cost of the main arm with the same network: em = emrol5's network, the rolling GRU form
#: runs the sliding form's network with one recurrent step instead of 16)
COSTS = {"eh": "cmp_eh_s3407__default", "eh__w50": "cmp_eh_s3407__default", "em": "cmp_emrol5_s3407__default",
         "emrol5": "cmp_emrol5_s3407__default", "eg": "cmp_eg_s3407__default", "eg__rolling": "cmp_eg_s3407__default",
         "ep": "cmp_ep_s3407__default", "epc": "cmp_epc_s3407__default", "epcl": "cmp_epc_s3407__default",
         "evhandtrack": "evhandtrack_s3407__stream__pmrfast__default", "b2rol5": "evhandtrack_s3407__stream__default",
         "s43": "s43_s3407__exec__default", "ep__warm": "cmp_ep_s3407__default", "epc__warm": "cmp_epc_s3407__default",
         "epc__gtcrop": "cmp_epc_s3407__default", "epcl__warm": "cmp_epc_s3407__default",
         "epcl__gtcrop": "cmp_epc_s3407__default", "eh__warm": "cmp_eh_s3407__default", "eg__warm": "cmp_eg_s3407__default",
         "eho": "cmp_eho_s3407__default", "eho__kf": "cmp_eho_s3407__default", "eho__kff": "cmp_eho_s3407__default",
         "epx": "cmp_epx_s3407__default", "epx__warm": "cmp_epx_s3407__default", "epx__gtcrop": "cmp_epx_s3407__default",
         "ehpe": "cmp_ehpe_s3407__default", "ehpe__carry": "cmp_ehpe_s3407__default", "fme": "cmp_fme_s3407__default",
         "ehpe__gtinit": "cmp_ehpe_s3407__default"}
FIELDS = ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm")


def from_report(path):
    r = read(path)
    require(r["status"] == "COMPLETE", "incomplete reading " + str(path))
    out = {k: {f: r["per_sequence"][s][f] for f in FIELDS} for k, s in SEQ.items()}
    out["overall"] = {f: r["overall"][f] for f in FIELDS}
    return out


def mean(per):
    return {k: {f: sum(p[k][f] for p in per) / len(per) for f in FIELDS} for k in ("local", "global", "overall")}


def cost_fields(tag):
    c = read(OUT / "costs" / tag / "report.json")
    require(c["status"] == "COMPLETE", "incomplete cost reading " + tag)
    return {"latency_ms_raw": c["latency_row"]["raw_ms"], "latency_anchor_ms": c["latency_row"]["anchor_ms"],
            "latency_ms_scaled_full1p75": c["latency_row"]["scaled_ms"], "macs_forward_packet": c["macs_layers"]["median"],
            "params_total": c["params_total"], "latency_e2e_ms": c["latency_e2e_ms"],
            "cost_reading": str((OUT / "costs" / tag).relative_to(ROOT))}


def readings(name):
    if name in ("evhandtrack", "b2rol5"):
        ident = read(EVHANDTRACK if name == "evhandtrack" else B2ROL5)
        return [({"reading": s["forms"]["streaming"]["reading"], "checkpoint": s["checkpoint"]},
                 {k: s["forms"]["streaming"][k] for k in ("local", "global", "overall")}) for s in ident["seeds"]]
    if name == "s43":
        row = read(ROWS / "evf_s43_main_row.json")
        return [({"reading": v.get("reading"), "checkpoint": v.get("checkpoint")},
                 {k: v[k] for k in ("local", "global", "overall")}) for _, v in sorted(row["per_seed"].items())]
    arm, _, form = name.partition("__")
    out = []
    for seed in SEEDS:
        d = OUT / "readings" / (f"cmp_{arm}_s{seed}" + (f"__{form}" if form else "")) / "e1" / "report.json"
        out.append(({"reading": str(d.relative_to(ROOT)), "checkpoint": read(OUT / "receipts" / f"cmp_{arm}_s{seed}.json")["checkpoint"]},
                    from_report(d)))
    return out


def write(names=None):
    done = []
    for name in names or LABELS:
        try:
            per = readings(name)
            cost = cost_fields(COSTS[name])
        except (FileNotFoundError, KeyError, RuntimeError) as e:
            print("skip", name, type(e).__name__, e)
            continue
        row = {"arm": f"cmp_{name}", "label": LABELS[name], "split": "val_core", "n_seeds": len(per),
               "form": name.partition("__")[2] or "main", "two_seed_mean": mean([p for _, p in per]),
               "per_seed": {str(i): {**ident, **p} for i, (ident, p) in enumerate(per)}, **cost,
               "written_utc": now(),
               "note": "CMP (#cmp-prereg-20261110): PFO engine E1 on zgz, three seeds; 'two_seed_mean' is the three-seed "
                       "mean (historical key); cost = seed 3407 in the CMP same-block cost reading (GPU 4, default flags)"}
        save(ROWS / f"cmp_{name}_main_row.json", row)
        done.append(name)
        print("row", name, {k: round(row["two_seed_mean"][k]["mpjpe_ra_mm"], 2) for k in ("local", "global", "overall")})
    return done


def table(names=MAIN):
    args = [f"cmp_{n}" for n in names] + sum((["--label", f"cmp_{n}={LABELS[n]}"] for n in names), [])
    subprocess.run([sys.executable, str(ROOT / "tools/report_table.py"), *args], check=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("verb", choices=("write", "table", "table_all"))
    a = p.parse_args(argv)
    if a.verb == "write":
        write()
    elif a.verb == "table":
        table()
    else:
        table(tuple(LABELS))


if __name__ == "__main__":
    main()
