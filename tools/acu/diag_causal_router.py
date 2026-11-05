#!/usr/bin/env python3
"""ACU D3 (docs/EXPERIMENT_RECORDS.md#acu-diag-20261104): how much r_s2d's routing accuracy depends on seeing events that
arrive after the routed one. Zero training (the deployed r_s2d weights), CPU, descriptive.

    python -m tools.acu.diag_causal_router run --seq zgz_local
    python -m tools.acu.diag_causal_router table

Nodes = S45's stride-sampled events of every 4th zgz packet (the routed set of B2), scored against XAG pseudo-labels.
Feature map variants (r_s2d's CNN / FPN, its per-node head on the node's own token):
  packet     the whole 50 ms packet's event image (as deployed; not causal: an early node sees later events)
  prev50     the image of the previous 50 ms window, ending where the packet starts (causal; 0-50 ms stale)
  slide25    a 50 ms window ending at the start of the node's 25 ms sub-slot, refreshed every 25 ms (causal)
  slide10    the same refreshed every 10 ms (causal)
Scores on nodes with a valid label: part accuracy, finger-group accuracy (palm + 5 fingers), on-hand recall
(router p >= 0.5 among labelled nodes), template-point error median (mm), and the same for the router's on-hand set.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

OUT = ROOT / "outputs/acu/20261104/diag/d3"
STRIDE = 4
VARIANTS = ("packet", "prev50", "slide25", "slide10")
GROUP_OF_PART = np.array([0, 2, 2, 2, 3, 3, 3, 5, 5, 5, 4, 4, 4, 1, 1, 1])


def run(seq):
    import torch
    import yaml
    from semkine import eval_track as ET
    from semkine import pfo_router as PR, s45sm as SM
    from semkine.encoder import EV_X, EV_Y, event_tokens
    from semkine.s45rt_router import event_image
    from semkine.xyz_mesh import stride_sample
    from tools.pfo.common import XAG_SOURCE_ROOT
    from tools.pfo.engine import endpoint_rows, sequences
    from tools.pfo.nets import sequence
    from tools.pfo.router_train import checkpoint
    torch.set_num_threads(16)
    cfg = yaml.safe_load((ROOT / "configs/pfo2_20261103/eval/pfo2_fgs_s3407.yaml").read_text())
    cfg["DATA"]["XAG_SOURCE_ROOT"] = str(XAG_SOURCE_ROOT)
    PR.register()
    SM.install("pfr", str(checkpoint("r_s2d")), "pfa", 0.5)
    dev = torch.device("cpu")
    net = SM._net(dev)
    native = SM._NATIVE["sample_sources"]
    root, seqs = sequences(cfg, (seq,))
    ctx = sequence(root, seqs[0][1], seq, dev)
    rows = endpoint_rows(ctx.aux)
    H, W = 180, 240

    def fmap(end_ms, window):
        ev5 = ET._window_events(ctx.events, ctx.offsets, ctx.tsub, int(end_ms), int(window))
        if len(ev5) == 0:
            return None
        b = ET.make_eval_packet(ev5, torch.zeros(1, 51), ctx.betas, ctx.K, int(window), dev)
        return net.trunk(event_image(b.events, b.ptr, b.delta_t_s).to(net.stem[0].weight.dtype))

    acc = {v: {"part": [], "group": [], "onhand": [], "err": [], "err_routed": [], "part_routed": []} for v in VARIANTS}
    with torch.no_grad():
        for i in range(0, len(rows), STRIDE):
            _, end, _, first = rows[i]
            if int(end) - 99 < 0:
                continue
            ev5 = ET._window_events(ctx.events, ctx.offsets, ctx.tsub, int(end), 50)
            if len(ev5) < 20:
                continue
            b = ET.make_eval_packet(ev5, torch.zeros(1, 51), ctx.betas, ctx.K, 50, dev)
            b = ET.attach_xag_sources(b, cfg, seq, ctx.offsets, int(end), 50, dev)
            src, mask = stride_sample(b.ptr, 2048)
            m = mask[0]
            nodes = src[0][m]
            tok = event_tokens(b.events, b.ptr, b.delta_t_s, H, W)[nodes]
            ev = b.events[nodes]
            lab = native(b, src, mask)
            F_ = net.faces.shape[0]
            lv = lab["xag_valid"][0][m] & (lab["xag_face"][0][m] >= 0) & (lab["xag_face"][0][m] < F_)
            lb = lab["xag_bary"][0][m].float()
            lv = lv & torch.isfinite(lb).all(-1) & (lb >= 0).all(-1) & (lb.sum(-1) > 0)
            lf = lab["xag_face"][0][m].clamp(0, F_ - 1)
            lpt = ((lb / lb.sum(-1, keepdim=True).clamp_min(1e-12)).unsqueeze(-1) * net.tmpl[net.faces[lf]]).sum(-2)
            lpart = net.face_part[lf]
            ms_rel = (ev[:, 3] * 1000.0).floor().long().clamp(0, 49)
            maps = {"packet": [(torch.ones_like(ms_rel, dtype=torch.bool), fmap(end, 50))],
                    "prev50": [(torch.ones_like(ms_rel, dtype=torch.bool), fmap(end - 50, 50))]}
            for v, P in (("slide25", 25), ("slide10", 10)):
                maps[v] = [((ms_rel >= j) & (ms_rel < j + P), fmap(end - 50 + j, 50)) for j in range(0, 50, P)]
            for v in VARIANTS:
                part = torch.zeros(len(nodes), dtype=torch.long)
                coord = torch.zeros(len(nodes), 3)
                onh = torch.zeros(len(nodes))
                for sel, fm in maps[v]:
                    if not sel.any():
                        continue
                    if fm is None:
                        onh[sel] = 0.0
                        continue
                    h4, w4 = fm.shape[-2:]
                    yy = (ev[sel, EV_Y].long() * h4 // H).clamp(0, h4 - 1)
                    xx = (ev[sel, EV_X].long() * w4 // W).clamp(0, w4 - 1)
                    feat = fm[0, :, yy, xx].t()
                    z = net.head(torch.cat([feat.to(tok.dtype), tok[sel]], -1))
                    part[sel] = net.part(z).argmax(-1)
                    coord[sel] = net.coord(z).float() * 0.01
                    onh[sel] = torch.sigmoid(net.onhand(z).squeeze(-1).float())
                # router point -> its face as deployed (part, then nearest centroid), projected barycentrics
                cand = net.part_faces[part]
                ok = net.part_faces_valid[part]
                d = (net.centroid[cand] - coord.unsqueeze(1)).norm(dim=-1).masked_fill(~ok, float("inf"))
                face = cand.gather(1, d.argmin(1, keepdim=True)).squeeze(1)
                from semkine.s45tr_t0 import project_bary
                bary = project_bary(coord, net.tmpl[net.faces[face]])
                rpt = (bary.unsqueeze(-1) * net.tmpl[net.faces[face]]).sum(-2)
                rpart = net.face_part[face]
                L = lv
                acc[v]["part"].append((rpart[L] == lpart[L]).numpy())
                acc[v]["group"].append((GROUP_OF_PART[rpart[L].numpy()] == GROUP_OF_PART[lpart[L].numpy()]))
                acc[v]["onhand"].append((onh[L] >= 0.5).numpy())
                acc[v]["err"].append(((rpt[L] - lpt[L]).norm(dim=-1) * 1000).numpy())
                R = L & (onh >= 0.5)
                acc[v]["err_routed"].append(((rpt[R] - lpt[R]).norm(dim=-1) * 1000).numpy())
                acc[v]["part_routed"].append((rpart[R] == lpart[R]).numpy())
    res = {}
    for v in VARIANTS:
        c = {k: np.concatenate(x) if x else np.zeros(0) for k, x in acc[v].items()}
        res[v] = {"n_labelled": int(len(c["part"])), "part_acc": float(c["part"].mean()), "group_acc": float(c["group"].mean()),
                  "onhand_recall": float(c["onhand"].mean()), "err_med_mm": float(np.median(c["err"])),
                  "routed_part_acc": float(c["part_routed"].mean()), "routed_err_med_mm": float(np.median(c["err_routed"]))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{seq}.json").write_text(json.dumps(res, indent=1))
    print(seq, json.dumps(res), flush=True)


def table():
    L = ["# ACU D3: r_s2d routing from causal maps (zero training)", "",
         "| recording | map | labelled nodes | part acc | finger-group acc | on-hand recall | point err med mm | routed: part acc / err mm |",
         "|---|---|---|---|---|---|---|---|"]
    for f in sorted(OUT.glob("*.json")):
        r = json.loads(f.read_text())
        for v in VARIANTS:
            x = r[v]
            L.append(f"| {f.stem} | {v} | {x['n_labelled']} | {x['part_acc']:.3f} | {x['group_acc']:.3f} | {x['onhand_recall']:.3f} | "
                     f"{x['err_med_mm']:.2f} | {x['routed_part_acc']:.3f} / {x['routed_err_med_mm']:.2f} |")
    (OUT.parent / "d3.md").write_text("\n".join(L))
    print("\n".join(L))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    r = sub.add_parser("run")
    r.add_argument("--seq", required=True)
    sub.add_parser("table")
    a = p.parse_args(argv)
    if a.verb == "run":
        if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
            raise SystemExit("CPU only: set CUDA_VISIBLE_DEVICES=''")
        run(a.seq)
    else:
        table()


if __name__ == "__main__":
    main()
