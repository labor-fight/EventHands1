#!/usr/bin/env python3
"""PFO router quality (docs/EXPERIMENT_RECORDS.md#pfo-prereg-20261103, router sections): one router per process, one GPU.

    python -m tools.pfo.router_read --router NAME            NAME: sr1 | t0p | a PFO router (tools/pfo/router_train.py)

On S45's stride-sampled nodes of every 4th 50 ms endpoint of zgz_local, zgz_global, ylf_local (sr1's held-out subject)
and lyq_local (a training subject), against the XAG per-event labels (diagnostic only):
  on-hand precision / recall at the deployed threshold 0.5, and the share of routed nodes without a label ("off-hand in
  the mesh"); part accuracy and point error (routed point on the rest template vs the label point, mm, median) on the
  routed labelled nodes; mean routed nodes per packet. Writes routers_read/<NAME>.json (exclusive).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.pfo.common import OUT, SR1, T0P, XAG_SOURCE_ROOT, now, require, require_one_gpu, save, sha  # noqa: E402

SEQS = ("zgz_local", "zgz_global", "ylf_local", "lyq_local")
STRIDE = 4


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--router", required=True)
    a = p.parse_args()
    import torch
    from config import load_config
    from semkine import eval_track as ET
    from semkine import s45sm as SM
    from semkine import s45sm_fast as FAST
    from semkine import xag as XAG
    from semkine.encoder import event_tokens
    from semkine.xyz_mesh import stride_sample
    from tools.pfo.engine import sequences
    from tools.pfo.nets import sequence
    require_one_gpu()
    out = OUT / "routers_read" / f"{a.router}.json"
    require(not out.exists(), "read once: " + str(out))
    if a.router == "sr1":
        kind, ckpt = "sr", SR1
    elif a.router == "t0p":
        kind, ckpt = "t0p", T0P
    else:
        from semkine import pfo_router as PR
        from tools.pfo.router_train import checkpoint
        PR.register()
        kind, ckpt = "pfr", checkpoint(a.router)
    device = torch.device("cuda")
    SM.install(kind, str(ckpt), "pfa", 0.5)
    net = SM._net(device)
    cfg = load_config(str(ROOT / "configs/s45sm_20261031/s45sm_pfasrdz_s3407.yaml"))
    cfg["DATA"]["XAG_SOURCE_ROOT"] = str(XAG_SOURCE_ROOT)
    root, seqs = sequences(cfg, SEQS)
    res = {"router": a.router, "kind": kind, "checkpoint_sha256": sha(ckpt), "params": SM.n_params(net),
           "reader_sha256": sha(__file__), "started_utc": now(), "per_sequence": {}}
    native = SM._NATIVE["sample_sources"]
    with torch.no_grad():
        for seq, directory in seqs:
            ctx = sequence(root, directory, seq, device)
            from tools.pfo.engine import endpoint_rows
            rows = endpoint_rows(ctx.aux)
            acc = {k: 0.0 for k in ("routed", "lab", "tp", "routed_lab", "part_ok", "packets")}
            errs = []
            for i in range(0, len(rows), STRIDE):
                end = rows[i][1]
                ev5 = ET._window_events(ctx.events, ctx.offsets, ctx.tsub, int(end), 50)
                if len(ev5) == 0:
                    continue
                prev = torch.zeros(1, 51, device=device)
                b = ET.make_eval_packet(ev5, prev, ctx.betas, ctx.K, 50, device)
                b = ET.attach_xag_sources(b, cfg, seq, ctx.offsets, int(end), 50, device)
                src, mask = stride_sample(b.ptr, SM.MAX_NODES)
                outr, routed, face, bary, _ = SM.route_nodes(net, SM._STATE["router"], b.events, b.ptr, b.delta_t_s,
                                                             0.5)
                lab = native(b, src, mask)
                F_ = net.faces.shape[0]
                lb = lab["xag_bary"].float()
                lv = lab["xag_valid"] & mask & (lab["xag_face"] >= 0) & (lab["xag_face"] < F_)
                lv = lv & torch.isfinite(lb).all(-1) & (lb >= 0).all(-1) & (lb.sum(-1) > 0)
                lf = lab["xag_face"].clamp(0, F_ - 1)
                lpt = ((lb / lb.sum(-1, keepdim=True).clamp_min(1e-12)).unsqueeze(-1) * net.tmpl[net.faces[lf]]).sum(-2)
                rpt = (bary.float().unsqueeze(-1) * net.tmpl[net.faces[face]]).sum(-2)
                both = routed & lv
                acc["routed"] += float(routed.sum())
                acc["lab"] += float(lv.sum())
                acc["tp"] += float(both.sum())
                acc["routed_lab"] += float(both.sum())
                acc["part_ok"] += float((net.face_part[face] == net.face_part[lf])[both].sum())
                acc["packets"] += 1
                errs.append(((rpt - lpt).norm(dim=-1)[both] * 1000).cpu().numpy())
            e = np.concatenate(errs) if errs else np.zeros(0)
            res["per_sequence"][seq] = {
                "onhand_precision": acc["tp"] / max(acc["routed"], 1), "onhand_recall": acc["tp"] / max(acc["lab"], 1),
                "offhand_in_mesh": 1 - acc["tp"] / max(acc["routed"], 1),
                "part_acc": acc["part_ok"] / max(acc["routed_lab"], 1),
                "point_err_mm_median": float(np.median(e)) if len(e) else None,
                "point_err_mm_mean": float(np.mean(e)) if len(e) else None,
                "routed_per_packet": acc["routed"] / max(acc["packets"], 1), "packets": int(acc["packets"])}
            print(seq, {k: round(v, 3) if isinstance(v, float) else v for k, v in res["per_sequence"][seq].items()},
                  flush=True)
    res["finished_utc"] = now()
    save(out, res, exclusive=True)


if __name__ == "__main__":
    main()
