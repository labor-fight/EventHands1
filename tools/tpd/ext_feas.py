#!/usr/bin/env python3
"""TPD problem 2, zero-training feasibility of the EXT finger-extent innovation (docs/EXPERIMENT_RECORDS.md#tpd-ext0-prereg-20261105).

Does the extent of a finger's routed event cloud measure the error of the prior's finger length in the image, in the
loop that matters (L1 read with the deployed router r_s2d: IDP reading idp_l1_s3407/e1, its incoming states)?

    python -m tools.tpd.ext_feas run       one GPU among 2-7: zgz packets (the e1 endpoints, 50 ms) -> r_s2d routes
                                           (deployed decoding, on-hand tau 0.5) -> per (packet, finger) quantities,
                                           outputs/tpd/20261105/ext0/pairs.npz
    python -m tools.tpd.ext_feas decide    CPU: the written rule -> ext0/decision.json (once)
    TPD_EXT0_DIR=<scratch> python -m tools.tpd.ext_feas run --limit N
                                           engineering smoke only (first N endpoints per sequence, scratch root)

Per (packet, finger) (fingers in MANO's order index / middle / pinky / ring / thumb; parts 1-15 -> finger (p - 1) // 3):
  n, tip   routed nodes whose argmax part is one of the finger's three parts; of them, those on its distal part
  axis     principal axis (2-D PCA) of those nodes' pixel coordinates (240 x 180 frame)
  m_ext    q95 - q05 of the nodes' projections on the axis                                   (measurement, events only)
  p_ext    max - min of the prior's four chain joints (MCP, PIP, DIP, TIP; FK of `incoming`) on the same axis (prediction)
  g_ext    the same with the GT at the packet end (`target`)
  p_axis   the prior's MCP -> TIP direction; m_ext_p / g_ext_p / p_ext_p on that axis (descriptive)
Kept pairs: n >= 20 and tip >= 5 ("packets with fingertip events").
Rule: r = Pearson(m_ext - p_ext, g_ext - p_ext) pooled over zgz_local and zgz_global; null r0 = the same with m_ext
replaced by the finger's mean m_ext in that sequence (a measurement without packet information). TRAIN if r >= 0.5 and
r - r0 >= 0.10; KILL if r < 0.3; otherwise WEAK (not trained).
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

from tools.tpd.common import IDP_OUT, OUT, ZGZ, now, read, require, save, sha  # noqa: E402

E0 = OUT / "ext0"
SOURCE = IDP_OUT / "readings" / "idp_l1_s3407" / "e1"
L1_SPEC = IDP_OUT / "specs" / "idp_l1_s3407.json"
MCP = (5, 9, 17, 13, 1)                     # OpenPose MCP of MANO's finger order index, middle, pinky, ring, thumb
N_MIN, TIP_MIN = 20, 5
RENDER_SCALE = 0.375


def finger_of_part(part):
    part = np.asarray(part)
    return np.where(part >= 1, (part - 1) // 3, -1)


def extent(xy, axis, lo=5.0, hi=95.0):
    t = xy @ axis
    return float(np.percentile(t, hi) - np.percentile(t, lo))


def span(points, axis):
    t = points @ axis
    return float(t.max() - t.min())


def pca_axis(xy):
    c = xy - xy.mean(0)
    w, v = np.linalg.eigh(c.T @ c)
    return v[:, -1]


def pearson(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def rule(r, r0):
    if r < 0.3:
        return "KILL"
    if r >= 0.5 and r - r0 >= 0.10:
        return "TRAIN"
    return "WEAK"


def project(J, K):
    fx, fy, cx, cy = K[0, 0] * RENDER_SCALE, K[1, 1] * RENDER_SCALE, K[0, 2] * RENDER_SCALE, K[1, 2] * RENDER_SCALE
    z = np.clip(J[..., 2], 1e-6, None)
    return np.stack([fx * J[..., 0] / z + cx, fy * J[..., 1] / z + cy], -1)


def run(limit=None):
    import torch
    import yaml
    from tools.pfo.common import require_one_gpu
    from tools.pfo.engine import sequences
    from tools.pfo import nets as N
    from semkine import eval_track as ET
    from semkine import pfo_router as PR
    from semkine import s45sm as SM
    from semkine import s45sm_fast as FAST
    from semkine.events import EV_X, EV_Y
    from semkine.xyz_mesh import stride_sample
    from mano_layer import ManoLayer
    from pose_repr import decode_to_mano_inputs
    root_out = E0 if limit is None else Path(os.environ["TPD_EXT0_DIR"])
    if limit is None:
        require_one_gpu()
    out = root_out / "pairs.npz"
    require(not out.exists(), "run once: " + str(out))
    root_out.mkdir(parents=True, exist_ok=True)
    spec = read(L1_SPEC)
    cfg = yaml.safe_load(Path(spec["config"]).read_text())
    s = cfg["S45SM"]
    require(s["ROUTER_KIND"] == "pfr", "L1's deployed router is r_s2d")
    dev = torch.device("cuda")
    PR.register()
    FAST.install(s["ROUTER_KIND"], s["ROUTER_CHECKPOINT"], "pfa", float(s["ONHAND_TAU"]))
    net = SM._net(dev)
    tau = float(s["ONHAND_TAU"])
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(dev).eval()          # as MNISTModel._build_mano / _fk
    z = np.load(SOURCE / "states.npz", allow_pickle=False)
    root, seqs = sequences(cfg, ZGZ)
    cols = {k: [] for k in ("seq", "end_ms", "finger", "n", "tip", "m_ext", "p_ext", "g_ext", "m_ext_p", "p_ext_p",
                            "g_ext_p")}
    counts = {}
    with torch.no_grad():
        for seq, directory in seqs:
            ctx = N.sequence(root, directory, seq, dev)
            K = np.asarray(ctx.aux["camera_K"], np.float64).reshape(3, 3)
            inc, tgt, ends = z[f"{seq}/incoming"][:limit], z[f"{seq}/target"][:limit], z[f"{seq}/end_ms"][:limit]
            beta = ctx.betas.expand(len(inc), -1)

            def joints(params):
                dec = decode_to_mano_inputs(torch.tensor(params, device=dev), "mano_full_axis_angle",
                                            mano.hands_components, mano.hands_mean)
                return mano(beta, dec["global_orient"], dec["local_full_aa"], dec["transl"])[1].double().cpu().numpy()
            Jp, Jg = project(joints(inc), K), project(joints(tgt), K)
            kept = 0
            for i, end in enumerate(ends):
                ev5 = ET._window_events(ctx.events, ctx.offsets, ctx.tsub, int(end), 50)
                pk = ET.make_eval_packet(ev5, torch.zeros(1, 51, device=dev), ctx.betas, ctx.K, 50, dev)
                src, mask = stride_sample(pk.ptr, 2048)
                o = FAST.router_forward(net, pk.events, pk.ptr, pk.delta_t_s, src, mask)
                routed = (mask & (torch.sigmoid(o["onhand"].float()) >= tau))[0].cpu().numpy()
                part = o["part"].float().argmax(-1)[0].cpu().numpy()
                node = pk.events[src[0]].cpu().numpy()
                xy = np.stack([node[:, EV_X], node[:, EV_Y]], -1).astype(np.float64)
                fing = finger_of_part(part)
                for f in range(5):
                    sel = routed & (fing == f)
                    n = int(sel.sum())
                    tip = int((routed & (part == 3 * f + 3)).sum())
                    if n < N_MIN or tip < TIP_MIN:
                        continue
                    pts = xy[sel]
                    ax = pca_axis(pts)
                    chain_p = Jp[i, MCP[f]:MCP[f] + 4]
                    chain_g = Jg[i, MCP[f]:MCP[f] + 4]
                    pa = chain_p[3] - chain_p[0]
                    pa = pa / max(np.linalg.norm(pa), 1e-9)
                    for k, v in (("seq", seq), ("end_ms", int(end)), ("finger", f), ("n", n), ("tip", tip),
                                 ("m_ext", extent(pts, ax)), ("p_ext", span(chain_p, ax)), ("g_ext", span(chain_g, ax)),
                                 ("m_ext_p", extent(pts, pa)), ("p_ext_p", span(chain_p, pa)),
                                 ("g_ext_p", span(chain_g, pa))):
                        cols[k].append(v)
                    kept += 1
            counts[seq] = {"packets": int(len(ends)), "pairs_kept": kept}
    arrays = {k: np.asarray(v) for k, v in cols.items()}
    np.savez_compressed(out, **arrays)
    save(root_out / "run.json", {"source": str(SOURCE), "source_states_sha256": sha(SOURCE / "states.npz"),
                           "l1_spec": str(L1_SPEC), "router_checkpoint": s["ROUTER_CHECKPOINT"], "tau": tau,
                           "counts": counts, "pairs_sha256": sha(out), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                           "finished_utc": now()}, exclusive=True)
    print(json.dumps(counts))


def stats(a, sel):
    dm, dg = a["m_ext"][sel] - a["p_ext"][sel], a["g_ext"][sel] - a["p_ext"][sel]
    mean_f = np.zeros(int(sel.sum()))
    seqs, fing, m = a["seq"][sel], a["finger"][sel], a["m_ext"][sel]
    for s in np.unique(seqs):
        for f in range(5):
            k = (seqs == s) & (fing == f)
            if k.any():
                mean_f[k] = m[k].mean()
    r, r0 = pearson(dm, dg), pearson(mean_f - a["p_ext"][sel], dg)
    rp = pearson(a["m_ext_p"][sel] - a["p_ext_p"][sel], a["g_ext_p"][sel] - a["p_ext_p"][sel])
    return {"pairs": int(sel.sum()), "r": r, "r_null": r0, "r_prior_axis": rp,
            "sd_true_px": float(dg.std()), "sd_meas_px": float(dm.std()),
            "slope_meas_on_true": float(np.polyfit(dg, dm, 1)[0])}


def decide():
    out = E0 / "decision.json"
    require(not out.exists(), "decided once")
    a = dict(np.load(E0 / "pairs.npz", allow_pickle=False))
    every = np.ones(len(a["seq"]), bool)
    pooled = stats(a, every)
    per_seq = {s: stats(a, a["seq"] == s) for s in ZGZ}
    per_finger = {str(f): stats(a, a["finger"] == f) for f in range(5) if (a["finger"] == f).sum() > 30}
    verdict = rule(pooled["r"], pooled["r_null"])
    save(out, {"rule": "TRAIN if r >= 0.5 and r - r_null >= 0.10; KILL if r < 0.3; else WEAK", "verdict": verdict,
               "pooled": pooled, "per_sequence": per_seq, "per_finger": per_finger,
               "pairs_sha256": sha(E0 / "pairs.npz"), "decided_utc": now()}, exclusive=True)
    print(verdict, json.dumps(pooled))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("verb", choices=("run", "decide"))
    p.add_argument("--limit", type=int)
    a = p.parse_args(argv)
    if a.verb == "run":
        run(a.limit)
    else:
        require(a.limit is None, "decide reads the full run")
        decide()


if __name__ == "__main__":
    main()
