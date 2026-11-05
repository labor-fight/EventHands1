#!/usr/bin/env python3
"""ACU D1b (docs/EXPERIMENT_RECORDS.md#acu-diag-20261104): are fingers without evidence still updated? Zero training,
CPU, read-only on existing PFO-engine readings plus the XAG pseudo-labels of each packet.

    python -m tools.acu.diag_support counts      per-endpoint label-valid node counts per hand group (zgz, all 2590
                                                 endpoints; S45 stride-sampled nodes, the ones every routed net reads)
    python -m tools.acu.diag_support analyse     E3 update / true motion / injection and f1 drift per finger, binned by
                                                 that finger's node count in the packet

Groups: palm (MANO part 0) and the five fingers (index 1-3, middle 4-6, pinky 7-9, ring 10-12, thumb 13-15); finger
joints in OpenPose order (thumb 1-4, index 5-8, middle 9-12, ring 13-16, pinky 17-20), hand frame (translation and root
rotation zeroed), mm per joint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.acu.diag_update import NETS, READ_ROOTS, ZGZ, hand_joints, states  # noqa: E402

OUT = ROOT / "outputs/acu/20261104/diag"
GROUP_OF_PART = np.array([0, 2, 2, 2, 3, 3, 3, 5, 5, 5, 4, 4, 4, 1, 1, 1])     # 0 palm, 1 thumb, 2 index, 3 middle, 4 ring, 5 pinky
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
JOINTS = {f: list(range(1 + 4 * i, 5 + 4 * i)) for i, f in enumerate(FINGERS)}   # OpenPose
BINS = ((0, 0), (1, 5), (6, 20), (21, 80), (81, 10 ** 9))
ANALYSE = ("B2 3407", "B2 3407 +label test", "B2-gt label", "S43 3407")


def counts():
    import torch
    import yaml
    from semkine import eval_track as ET
    from semkine import xag as XAG
    from semkine.xyz_mesh import stride_sample
    from tools.pfo.common import XAG_SOURCE_ROOT
    from tools.pfo.engine import endpoint_rows, sequences
    from tools.pfo.nets import sequence
    torch.set_num_threads(16)
    cfg = yaml.safe_load((ROOT / "configs/pfo2_20261103/eval/pfo2_fgs_s3407.yaml").read_text())
    cfg["DATA"]["XAG_SOURCE_ROOT"] = str(XAG_SOURCE_ROOT)
    root, seqs = sequences(cfg, ZGZ)
    dev = torch.device("cpu")
    out = {}
    with torch.no_grad():
        for seq, directory in seqs:
            ctx = sequence(root, directory, seq, dev)
            rows = endpoint_rows(ctx.aux)
            c = np.zeros((len(rows), 6), np.int64)
            n_all = np.zeros(len(rows), np.int64)
            for i, (_, end, _, _) in enumerate(rows):
                ev5 = ET._window_events(ctx.events, ctx.offsets, ctx.tsub, int(end), 50)
                n_all[i] = len(ev5)
                if len(ev5) == 0:
                    continue
                b = ET.make_eval_packet(ev5, torch.zeros(1, 51), ctx.betas, ctx.K, 50, dev)
                b = ET.attach_xag_sources(b, cfg, seq, ctx.offsets, int(end), 50, dev)
                src, mask = stride_sample(b.ptr, 2048)
                lab = XAG.sample_sources(b, src, mask)
                ok = (lab["xag_valid"] & mask & (lab["xag_part"] >= 0)).reshape(-1)
                parts = lab["xag_part"].reshape(-1)[ok].numpy()
                c[i] = np.bincount(GROUP_OF_PART[parts], minlength=6)
            out[f"{seq}/group_counts"] = c
            out[f"{seq}/n_events"] = n_all
            print(seq, len(rows), "mean counts", c.mean(0).round(1), flush=True)
    np.savez_compressed(OUT / "d1b_counts.npz", **out)


def rms(v):
    return float(np.sqrt((v * v).sum(-1).mean())) if len(v) else float("nan")


def analyse():
    cnt = np.load(OUT / "d1b_counts.npz")
    res = {}
    for name in ANALYSE:
        root, tag = NETS[name]
        e3, f1 = states(root, tag, "e3"), states(root, tag, "f1")
        res[name] = {}
        for seq in ZGZ:
            c = cnt[f"{seq}/group_counts"]
            prev, pred, tgt = e3[f"{seq}/gt_previous"], e3[f"{seq}/prediction"], e3[f"{seq}/target"]
            jp, jq, jt = hand_joints(prev, seq), hand_joints(pred, seq), hand_joints(tgt, seq)
            blk = {}
            for gi, f in enumerate(FINGERS, start=1):
                J = JOINTS[f]
                D, M = (jq - jp)[:, J], (jt - jp)[:, J]
                E = (jq - jt)[:, J]
                rows = {}
                for lo, hi in BINS:
                    sel = (c[:, gi] >= lo) & (c[:, gi] <= hi)
                    if sel.sum() < 15:
                        continue
                    d2, m2 = D[sel].reshape(sel.sum(), -1), M[sel].reshape(sel.sum(), -1)
                    g = float((d2 * m2).sum() / max((m2 * m2).sum(), 1e-12))
                    nrm = len(J)
                    rows[f"{lo}-{hi}"] = {"n": int(sel.sum()), "G": g,
                                          "rms_D": float(np.sqrt((d2 ** 2).sum(1).mean() / nrm)),
                                          "rms_m": float(np.sqrt((m2 ** 2).sum(1).mean() / nrm)),
                                          "rms_inj": float(np.sqrt(((d2 - g * m2) ** 2).sum(1).mean() / nrm)),
                                          "rms_err": float(np.sqrt((E[sel].reshape(sel.sum(), -1) ** 2).sum(1).mean() / nrm))}
                blk[f] = rows
            fd = {}
            if f1 is not None and f"{seq}/gt_iter1" in f1.files:
                idx = f1[f"{seq}/index"].astype(np.int64)
                j1, j20, jt1 = hand_joints(f1[f"{seq}/gt_iter1"], seq), hand_joints(f1[f"{seq}/gt_iter20"], seq), \
                    hand_joints(f1[f"{seq}/target"], seq)
                jm = hand_joints(f1[f"{seq}/mean_iter20"], seq)
                cc = c[idx]
                for gi, f in enumerate(FINGERS, start=1):
                    J = JOINTS[f]
                    rows = {}
                    for lo, hi in BINS:
                        sel = (cc[:, gi] >= lo) & (cc[:, gi] <= hi)
                        if sel.sum() < 8:
                            continue
                        nrm = len(J)
                        dr = (j20 - j1)[sel][:, J].reshape(sel.sum(), -1)
                        rows[f"{lo}-{hi}"] = {"n": int(sel.sum()),
                                              "drift_1_20": float(np.sqrt((dr ** 2).sum(1).mean() / nrm)),
                                              "err_iter1": float(np.sqrt((((j1 - jt1)[sel][:, J]) ** 2).sum(-1).mean())),
                                              "err_iter20": float(np.sqrt((((j20 - jt1)[sel][:, J]) ** 2).sum(-1).mean())),
                                              "err_mean20": float(np.sqrt((((jm - jt1)[sel][:, J]) ** 2).sum(-1).mean()))}
                    fd[f] = rows
            res[name][seq] = {"e3": blk, "f1": fd}
    (OUT / "d1b.json").write_text(json.dumps(res, indent=1))
    L = ["# ACU D1b: finger updates binned by that finger's label-valid node count in the packet", "",
         "E3 per finger: n / RMS update |D| / RMS true motion |m| / gain G / injection / error (mm per joint); "
         "f1: drift iter1->20 / error iter20 (mm per joint)", ""]
    for seq in ZGZ:
        L += [f"## {seq}", ""]
        for name in ANALYSE:
            L += [f"### {name}", "", "| finger | bin | n | RMS D | RMS m | G | inj | err | f1 n | drift 1->20 | err 20 | err mean-start 20 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
            r = res[name][seq]
            for f in FINGERS:
                for b, v in r["e3"].get(f, {}).items():
                    w = r["f1"].get(f, {}).get(b, {})
                    L.append(f"| {f} | {b} | {v['n']} | {v['rms_D']:.2f} | {v['rms_m']:.2f} | {v['G']:.2f} | {v['rms_inj']:.2f} | "
                             f"{v['rms_err']:.2f} | {w.get('n', '')} | {w.get('drift_1_20', float('nan')):.2f} | "
                             f"{w.get('err_iter20', float('nan')):.2f} | {w.get('err_mean20', float('nan')):.2f} |")
            L.append("")
    (OUT / "d1b.md").write_text("\n".join(L))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("verb", choices=("counts", "analyse"))
    a = p.parse_args(argv)
    counts() if a.verb == "counts" else analyse()


if __name__ == "__main__":
    main()
