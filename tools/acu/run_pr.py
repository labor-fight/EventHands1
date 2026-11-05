#!/usr/bin/env python3
"""ACU s1c readings of the PR router (docs/EXPERIMENT_RECORDS.md#acu-s1c-prereg-20261104); one GPU among 4-7 (or CPU).

    python -m tools.acu.run_pr read --seq S      identity accuracy on every 4th endpoint of S (zgz_local, zgz_global,
                                                 ylf_local = the routers' held-out subject): PR on its causal window nodes
                                                 (50 ms history, causal stride), and r_s2d on the same nodes with its
                                                 whole-packet image (the non-causal reference); against the XAG labels
    python -m tools.acu.run_pr table             analysis/s1c_router.md
    python -m tools.acu.run_pr equiv             streaming equivalence on real packets: every window node's outputs from the
                                                 whole window equal those from the prefix ending at that node (sampled
                                                 nodes, float64); receipts/pr_equiv.json
Scores as D3 (#acu-diag-20261104): part / finger-group accuracy, on-hand recall (p >= 0.5 among labelled nodes),
rest-template point error median (mm), and on the router's own on-hand set.
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

from tools.acu.common import OUT, now, read, require, save, sha  # noqa: E402

STRIDE = 4
SEQS = ("zgz_local", "zgz_global", "ylf_local")
GROUP_OF_PART = np.array([0, 2, 2, 2, 3, 3, 3, 5, 5, 5, 4, 4, 4, 1, 1, 1])


def _setup(device):
    import torch
    import yaml
    from semkine import acu_pr as PR
    from semkine import pfo_router as PRR
    from semkine import s45sm as SM
    from tools.acu.router_pr import checkpoint
    from tools.pfo.common import XAG_SOURCE_ROOT
    from tools.pfo.router_train import checkpoint as pfo_ck
    torch.set_num_threads(16)
    cfg = yaml.safe_load((ROOT / "configs/pfo2_20261103/eval/pfo2_fgs_s3407.yaml").read_text())
    cfg["DATA"]["XAG_SOURCE_ROOT"] = str(XAG_SOURCE_ROOT)
    receipt = read(checkpoint().with_suffix(".json"))
    require(receipt["status"] == "COMPLETE" and sha(checkpoint()) == receipt["checkpoint_sha256"], "PR router not complete")
    pr, _ = PR.load(checkpoint(), map_location=device)
    PRR.register()
    SM.install("pfr", str(pfo_ck("r_s2d")), "pfa", 0.5)
    rs = SM._net(device)
    return cfg, pr.to(device).eval(), rs.eval(), SM


def _packet(ctx, cfg, seq, end, run_a, device):
    import torch
    from semkine import acu_hist as AH
    from semkine import eval_track as ET
    ev5 = ET._window_events(ctx.events, ctx.offsets, ctx.tsub, int(end), 50)
    b = ET.make_eval_packet(ev5, torch.zeros(1, 51), ctx.betas, ctx.K, 50, device)
    b = ET.attach_xag_sources(b, cfg, seq, ctx.offsets, int(end), 50, device)
    hist, n_hist, hist_len_s, k_hist = AH.eval_history(ctx.events, ctx.offsets, ctx.tsub, int(end), int(run_a))
    hev = torch.zeros(len(hist), 5, device=device)
    if len(hist):
        hev[:, 1:] = torch.from_numpy(hist).to(device)
    hptr = torch.tensor([0, len(hist)], device=device)
    from semkine.acu_pr import stride_k
    k_win = stride_k(n_hist, hist_len_s, 0.05).view(1).to(device)
    return b, hev, hptr, k_win, torch.tensor([k_hist], device=device)


def read_seq(seq):
    import torch
    from semkine import xag as XAG
    from semkine.s45tr_t0 import project_bary
    from tools.pfo.engine import endpoint_rows, sequences
    from tools.pfo.nets import sequence
    dev = torch.device("cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("",) else "cpu")
    cfg, pr, rs, SM = _setup(dev)
    native = SM._NATIVE["sample_sources"]
    root, seqs = sequences(cfg, (seq,))
    ctx = sequence(root, seqs[0][1], seq, dev)
    rows = endpoint_rows(ctx.aux)
    runs = np.asarray(ctx.aux["valid_runs_ms"], np.int64).reshape(-1, 2)
    acc = {r: {"part": [], "group": [], "onhand": [], "err": [], "part_routed": [], "err_routed": []} for r in ("pr", "r_s2d")}
    nodes = []
    with torch.no_grad():
        for i in range(0, len(rows), STRIDE):
            rid, end, _, _ = rows[i]
            b, hev, hptr, k_win, k_hist = _packet(ctx, cfg, seq, end, runs[rid][0], dev)
            if int(b.counts.sum()) < 20:
                continue
            o = pr(b.events, b.ptr, b.delta_t_s, hev, hptr, k_win, k_hist)
            src, mask = o["src"], o["mask"]
            nodes.append(int(mask.sum()))
            ro = rs(b.events, b.ptr, b.delta_t_s)                    # r_s2d on its own stride nodes: re-query below
            # r_s2d on PR's nodes: its feature map and per-node head at PR's node events
            from semkine.s45rt_router import event_image
            from semkine.encoder import event_tokens
            from semkine.events import EV_X, EV_Y
            fmap = rs.trunk(event_image(b.events, b.ptr, b.delta_t_s).to(rs.stem[0].weight.dtype))
            tok = event_tokens(b.events, b.ptr, b.delta_t_s, 180, 240)
            flat = src[mask]
            ne = b.events[flat]
            h4, w4 = fmap.shape[-2:]
            yy = (ne[:, EV_Y].long() * h4 // 180).clamp(0, h4 - 1)
            xx = (ne[:, EV_X].long() * w4 // 240).clamp(0, w4 - 1)
            z = rs.head(torch.cat([fmap[0, :, yy, xx].t().to(tok.dtype), tok[flat]], -1))
            outs = {"pr": (o["part"][mask], o["coord"][mask], o["onhand"][mask], pr),
                    "r_s2d": (rs.part(z), rs.coord(z), rs.onhand(z).squeeze(-1), rs)}
            del ro
            lab = native(b, src, mask)
            F_ = pr.faces.shape[0]
            lv = (lab["xag_valid"] & (lab["xag_face"] >= 0) & (lab["xag_face"] < F_))[mask]
            lb = lab["xag_bary"][mask].float()
            lv = lv & torch.isfinite(lb).all(-1) & (lb >= 0).all(-1) & (lb.sum(-1) > 0)
            lf = lab["xag_face"][mask].clamp(0, F_ - 1)
            lpt = ((lb / lb.sum(-1, keepdim=True).clamp_min(1e-12)).unsqueeze(-1) * pr.tmpl[pr.faces[lf]]).sum(-2)
            lpart = pr.face_part[lf]
            for name, (part, coord, onh, net) in outs.items():
                pt = coord.float() * 0.01
                pp = part.float().argmax(-1)
                cand, ok = net.part_faces[pp], net.part_faces_valid[pp]
                dd = (net.centroid[cand] - pt.unsqueeze(1)).norm(dim=-1).masked_fill(~ok, float("inf"))
                face = cand.gather(1, dd.argmin(1, keepdim=True)).squeeze(1)
                bary = project_bary(pt, net.tmpl[net.faces[face]])
                rpt = (bary.unsqueeze(-1) * net.tmpl[net.faces[face]]).sum(-2)
                rpart = net.face_part[face]
                on = torch.sigmoid(onh.float()) >= 0.5
                L = lv
                a = acc[name]
                a["part"].append((rpart[L] == lpart[L]).cpu().numpy())
                a["group"].append(GROUP_OF_PART[rpart[L].cpu().numpy()] == GROUP_OF_PART[lpart[L].cpu().numpy()])
                a["onhand"].append(on[L].cpu().numpy())
                a["err"].append(((rpt[L] - lpt[L]).norm(dim=-1) * 1000).cpu().numpy())
                R = L & on
                a["part_routed"].append((rpart[R] == lpart[R]).cpu().numpy())
                a["err_routed"].append(((rpt[R] - lpt[R]).norm(dim=-1) * 1000).cpu().numpy())
    res = {"seq": seq, "packets": len(nodes), "nodes_mean": float(np.mean(nodes)), "read_at_utc": now()}
    for name, a in acc.items():
        c = {k: np.concatenate(v) if v else np.zeros(0) for k, v in a.items()}
        res[name] = {"n_labelled": int(len(c["part"])), "part_acc": float(c["part"].mean()),
                     "group_acc": float(c["group"].mean()), "onhand_recall": float(c["onhand"].mean()),
                     "err_med_mm": float(np.median(c["err"])), "routed_part_acc": float(c["part_routed"].mean()),
                     "routed_err_med_mm": float(np.median(c["err_routed"]))}
    out = OUT / "readings" / "router" / f"pr_{seq}.json"
    require(not out.exists(), "preserve existing reading: " + str(out))
    save(out, res, exclusive=True)
    print(json.dumps(res), flush=True)


def table():
    L = ["# ACU s1c: PR router identity accuracy (causal window nodes; r_s2d on the same nodes with its whole-packet image)",
         "", "| recording | router | packets | nodes / packet | part acc | finger-group acc | on-hand recall | point err med mm |"
         " routed: part acc / err mm |", "|---|---|---|---|---|---|---|---|---|"]
    for seq in SEQS:
        p = OUT / "readings" / "router" / f"pr_{seq}.json"
        if not p.exists():
            continue
        r = read(p)
        for name in ("pr", "r_s2d"):
            x = r[name]
            L.append(f"| {seq} | {name} | {r['packets']} | {r['nodes_mean']:.0f} | {x['part_acc']:.3f} | {x['group_acc']:.3f} | "
                     f"{x['onhand_recall']:.3f} | {x['err_med_mm']:.2f} | {x['routed_part_acc']:.3f} / {x['routed_err_med_mm']:.2f} |")
    out = OUT / "analysis" / "s1c_router.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


def equiv():
    """Every sampled window node: outputs from the whole window == outputs from the window cut right after that node."""
    import torch
    from tools.pfo.engine import endpoint_rows, sequences
    from tools.pfo.nets import sequence
    dev = torch.device("cpu")
    cfg, pr, _, _ = _setup(dev)
    pr = pr.double()
    rows_out, worst = [], 0.0
    for seq in ("zgz_local", "zgz_global"):
        root, seqs = sequences(cfg, (seq,))
        ctx = sequence(root, seqs[0][1], seq, dev)
        rows = endpoint_rows(ctx.aux)
        runs = np.asarray(ctx.aux["valid_runs_ms"], np.int64).reshape(-1, 2)
        for i in range(1, len(rows), max(1, len(rows) // 12)):
            rid, end, _, _ = rows[i]
            b, hev, hptr, k_win, k_hist = _packet(ctx, cfg, seq, end, runs[rid][0], dev)
            if int(b.counts.sum()) < 20:
                continue
            full = pr(b.events.double(), b.ptr, b.delta_t_s, hev.double(), hptr, k_win, k_hist)
            n = int(full["mask"].sum())
            for m in np.unique(np.linspace(0, n - 1, 6).astype(int)):
                cut = int(full["src"][0, m]) + 1
                ptr = torch.tensor([0, cut])
                part = pr(b.events[:cut].double(), ptr, b.delta_t_s, hev.double(), hptr, k_win, k_hist)
                d = max(float((full[k][0, m] - part[k][0, m]).abs().max()) for k in ("part", "coord", "onhand"))
                worst = max(worst, d)
                rows_out.append({"seq": seq, "end": int(end), "node": int(m), "nodes": n, "max_abs_diff": d})
    receipt = {"check": "pr_prefix_equivalence", "rows": rows_out, "max_abs_diff": worst,
               "status": "PASS" if worst <= 1e-9 else "FAIL", "checked_at_utc": now()}
    save(OUT / "receipts" / "pr_equiv.json", receipt, exclusive=True)
    print(receipt["status"], worst, len(rows_out), flush=True)
    require(receipt["status"] == "PASS", "PR is not prefix-stable")


def check(phase):
    """The router training completed and its checkpoint is the one its completion record names."""
    from tools.acu.router_pr import checkpoint
    rec = read(checkpoint(phase).with_suffix(".json"))
    ok = rec["status"] == "COMPLETE" and sha(checkpoint(phase)) == rec["checkpoint_sha256"] and \
        rec["step"] == (40 if phase == "preflight" else 6000)
    save(OUT / "receipts" / f"pr_{phase}.json", {"record": rec, "status": "PASS" if ok else "FAIL",
                                                 "checked_at_utc": now()}, exclusive=True)
    require(ok, "PR router training incomplete")


#: the written s1c rule (#acu-s1c-prereg-20261104)
RULE = {"group_margin": 0.03, "part_margin": 0.05, "recordings": ("ylf_local", "zgz_local")}


def decide():
    out = OUT / "decisions" / "s1c.json"
    require(not out.exists(), "decided once")
    eq = read(OUT / "receipts" / "pr_equiv.json")
    rows = {seq: read(OUT / "readings" / "router" / f"pr_{seq}.json") for seq in SEQS}
    ok = {seq: rows[seq]["pr"]["group_acc"] >= rows[seq]["r_s2d"]["group_acc"] - RULE["group_margin"]
          and rows[seq]["pr"]["part_acc"] >= rows[seq]["r_s2d"]["part_acc"] - RULE["part_margin"] for seq in RULE["recordings"]}
    verdict = "PR_OK" if eq["status"] == "PASS" and all(ok.values()) else "PR_WEAK"
    save(out, {"rule": RULE, "equivalence": eq["status"], "per_recording": ok, "decision": verdict,
               "decided_at_utc": now()}, exclusive=True)
    table()
    print(verdict, ok, flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    r = sub.add_parser("read")
    r.add_argument("--seq", required=True, choices=SEQS)
    sub.add_parser("table")
    sub.add_parser("equiv")
    c = sub.add_parser("check")
    c.add_argument("--phase", required=True, choices=("preflight", "formal"))
    sub.add_parser("decide")
    a = p.parse_args(argv)
    {"read": lambda: read_seq(a.seq), "table": table, "equiv": equiv, "check": lambda: check(a.phase),
     "decide": decide}[a.verb]()


if __name__ == "__main__":
    main()
