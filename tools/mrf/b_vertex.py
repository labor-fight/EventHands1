#!/usr/bin/env python3
"""MRF Stage B-vertex (docs/EXPERIMENT_RECORDS.md#mrf-prereg-20261024): what each round of the frozen module G adds
per vertex. One process = one condition (A0: frozen C1; A3: frozen A3F0), E3 anchors (GT at the previous endpoint).

Feature sets (all from the frozen module on the same packets):

    H0 H1 H2 H3   the module's vertex features before / after each round (support O, S1, S2, S3)
    H0near        parameter-free inpainting of H0: three frontier-mean rounds (`semkine.mrf.diffuse_round`), i.e. the
                  copy of the nearest supported vertices; support S3
    H0smooth      three rounds of the mean over the vertex itself and its supported 1-ring, support growing as S

Probes (linear and 2-layer MLP, 3 seeds each per feature set) on [feature, |n.d|, O, own support] -> the vertex-frame 2-D
displacement / rho (Gaussian NLL, visible supported vertices; AGG778 / P4 convention, rho = f 2 cm / z) and the 3-D
displacement / L_REL (MSE, all supported vertices). 3000 steps x 64, Adam 1e-3, warmup 300, cosine to 2 %, clip 10.

Readouts per block (sufficient statistics, bootstrapped in `summarize.py`): normal-displacement correlation on visible
contour vertices that own a GT event (the AGG778 primary), on O and on S3 \\ O; 3-D displacement R^2 on O, on visible
S3 \\ O and on hidden S3 \\ O.

    CUDA_VISIBLE_DEVICES=6 CUDA_DEVICE_ORDER=PCI_BUS_ID python tools/mrf/b_vertex.py --cond A0 --out outputs/mrf/b_vertex/A0
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.agg778 import common as C  # noqa: E402
from tools.agg778.b_vertex import pack, sample_stats  # noqa: E402
from tools.mrf import common as MC  # noqa: E402
from semkine.mrf import diffuse_round  # noqa: E402

FEATS = ("H0", "H1", "H2", "H3", "H0near", "H0smooth")
READOUTS = ("lin", "mlp")
SEEDS = (0, 1, 2)
STEPS, BATCH, LR, WARMUP, LR_FLOOR, CLIP, LAMBDA_3D = 3000, 64, 1e-3, 300, 0.02, 10.0, 1.0
KEYS = ("own|con|normal", "O|con|normal", "SO|con|normal", "O|3d", "SOvis|3d", "SOhid|3d")


def lr_lambda(step, steps):
    if step < WARMUP:
        return float(step + 1) / WARMUP
    p = min(max((step - WARMUP) / float(steps - WARMUP), 0.0), 1.0)
    return LR_FLOOR + (1.0 - LR_FLOOR) * 0.5 * (1.0 + math.cos(math.pi * p))


def smooth_round(H, S, idx, emask):
    """Mean over the vertex itself (if supported) and its supported 1-ring; S' = S or any(S_l)."""
    B, V, Cc = H.shape
    K = idx.shape[1]
    flat = idx.reshape(-1)
    Sn = S[:, flat].view(B, V, K) * emask.unsqueeze(0).to(S.dtype)
    num = (H[:, flat].view(B, V, K, Cc) * Sn.unsqueeze(-1)).sum(2) + H * S.unsqueeze(-1)
    den = Sn.sum(2) + S
    S_next = torch.maximum(S, Sn.amax(2))
    return num / den.clamp_min(1.0).unsqueeze(-1) * S_next.unsqueeze(-1), S_next


@torch.no_grad()
def batch_inputs(frozen, items, anchors, dev):
    batch = C.make_batch(items, anchors, dev)
    cap = C.forward_capture(frozen, batch)
    xyz = frozen.xyz
    H0, O = cap["H0"], cap["O"]
    gemb = torch.relu(xyz.edge(xyz.edge_geometry(cap["verts"])))
    feats, sup = {"H0": H0}, {"H0": O}
    H, S = H0, O
    for i, layer in enumerate(xyz.layers):
        H, S = layer(H, S, gemb, xyz.ring_idx, xyz.ring_mask)
        feats[f"H{i + 1}"], sup[f"H{i + 1}"] = H, S
    Hn, Sn, Hs, Ss = H0, O, H0, O
    for _ in range(3):
        Hn, Sn = diffuse_round(Hn, Sn, xyz.ring_idx, xyz.ring_mask)
        Hs, Ss = smooth_round(Hs, Ss, xyz.ring_idx, xyz.ring_mask)
    feats["H0near"], sup["H0near"] = Hn, Sn
    feats["H0smooth"], sup["H0smooth"] = Hs, Ss
    geo = C.vertex_geometry(frozen, batch, cap["verts"])
    vgt = C.fk_verts(frozen, batch.target, batch)
    uv_gt = C.project(vgt, geo["fx"], geo["fy"], geo["cx"], geo["cy"])
    t2 = uv_gt - geo["uv"]
    tf = (geo["R"].transpose(-1, -2) @ t2.unsqueeze(-1)).squeeze(-1) / geo["rho"].unsqueeze(-1)
    return {"batch": batch, "cap": cap, "geo": geo, "feats": feats, "sup": sup, "S3": sup["H3"], "t2": t2, "tf": tf,
            "t3": (vgt - cap["verts"]) / 0.01, "own": C.gt_owned(items, dev)}


class ProbeHead(nn.Module):
    def __init__(self, readout="mlp", f_in=128, hidden=128):
        super().__init__()
        if readout == "lin":
            self.net = nn.Linear(f_in + 3, 8)
        else:
            self.net = nn.Sequential(nn.Linear(f_in + 3, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden),
                                     nn.ReLU(inplace=True), nn.Linear(hidden, 8))

    def forward(self, feat, cos, O, S):
        o = self.net(torch.cat([feat, cos.unsqueeze(-1), (O > 0).unsqueeze(-1).to(feat.dtype),
                                (S > 0).unsqueeze(-1).to(feat.dtype)], -1))
        m = o[..., :2]
        L = torch.zeros(*o.shape[:-1], 2, 2, device=o.device, dtype=o.dtype)
        L[..., 0, 0] = F.softplus(o[..., 2]) + 1e-3
        L[..., 1, 1] = F.softplus(o[..., 3]) + 1e-3
        L[..., 1, 0] = o[..., 4]
        return m, L, o[..., 5:8]


def head_out(head, name, inp):
    return head(inp["feats"][name].float(), inp["geo"]["cos"].float(), inp["cap"]["O"], inp["sup"][name])


def loss_of(out, inp):
    from semkine import xvm as XV
    m, L, d3 = out
    S3 = (inp["S3"] > 0).float()
    vis = inp["geo"]["vis"].float() * S3
    nll = XV.gaussian_nll(m.float(), L.float(), inp["tf"], vis)
    mse = (((d3.float() - inp["t3"]) ** 2).sum(-1) * S3).sum() / (3.0 * S3.sum().clamp(min=1.0))
    return nll + LAMBDA_3D * mse


@torch.no_grad()
def accumulate(acc, name, out, inp, blocks):
    m, L, d3 = (o.float() for o in out)
    geo = inp["geo"]
    e1 = geo["R"][..., :, 0]
    pn = m[..., 0] * geo["rho"]
    tn = (e1 * inp["t2"]).sum(-1)
    con = geo["cos"] < C.CONTOUR_COS
    vis = geo["vis"]
    O = inp["cap"]["O"] > 0
    SO = (inp["S3"] > 0) & ~O
    rows = {"own|con|normal": sample_stats(pn, tn, inp["own"] & vis & con),
            "O|con|normal": sample_stats(pn, tn, O & vis & con),
            "SO|con|normal": sample_stats(pn, tn, SO & vis & con),
            "O|3d": sample_stats(d3, inp["t3"], O), "SOvis|3d": sample_stats(d3, inp["t3"], SO & vis),
            "SOhid|3d": sample_stats(d3, inp["t3"], SO & ~vis)}
    for key, st in rows.items():
        st = st.cpu().numpy()
        d = acc.setdefault(name, {}).setdefault(key, {})
        for b, row in zip(blocks, st):
            d[int(b)] = d.get(int(b), 0.0) + row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", choices=("A0", "A3"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    if args.cpu and not args.smoke:
        raise SystemExit("--cpu is for the engineering smoke only")
    dev = torch.device("cpu") if args.cpu else MC.require_gpu()
    if not args.smoke:
        MC.verify_protocol("B")
    C.set_policy()
    out = Path(args.out)
    if (out / "result.json").exists():
        raise SystemExit(f"{out} holds a result; preserve it")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(3407)
    frozen, cfg = C.load_frozen(args.cond, dev)
    steps, batch = (3, 4 if args.cpu else BATCH) if args.smoke else (STEPS, BATCH)
    heads, names = [], []
    for f in FEATS:
        for ro in READOUTS:
            for s in SEEDS:
                with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):
                    torch.manual_seed(7000 + 10 * READOUTS.index(ro) + s)    # one head init per readout and seed
                    heads.append(ProbeHead(ro).to(dev))
                names.append((f, ro, s))
    opts = [torch.optim.Adam(h.parameters(), lr=LR) for h in heads]
    scheds = [torch.optim.lr_scheduler.LambdaLR(o, lambda i, n=steps: lr_lambda(i, n)) for o in opts]
    fit = C.Endpoints(C.subject_seqs(C.FIT_SUBJ))
    if args.smoke:
        fit.index = fit.index[:4 * batch]
    ld = C.loader(fit, batch, True, 3407, workers=args.workers, drop_last=True)
    run = np.zeros(len(heads))
    t0, it, curve = time.time(), 0, []
    while it < steps:
        for items in ld:
            inp = batch_inputs(frozen, items, np.stack([x["gt_prev"] for x in items]), dev)
            for k, (h, (f, ro, s)) in enumerate(zip(heads, names)):
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):     # CPU smoke: fp32
                    o = head_out(h, f, inp)
                loss = loss_of(o, inp)
                opts[k].zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(h.parameters(), CLIP)
                opts[k].step()
                scheds[k].step()
                run[k] += float(loss)
            it += 1
            if it % 100 == 0 or it == steps:
                n = 100 if it % 100 == 0 else it % 100
                ent = {"/".join(map(str, nm)): round(run[k] / n, 4) for k, nm in enumerate(names)}
                curve.append({"it": it, **ent})
                print(it, json.dumps(ent), f"{time.time() - t0:.0f} s", flush=True)
                run[:] = 0
            if it >= steps:
                break
    torch.save({"heads": [h.state_dict() for h in heads], "names": names}, out / "heads.pt")
    for h in heads:
        h.eval()
    for split, ds in (("zgz", C.Endpoints(C.zgz_seqs())), ("sel", C.Endpoints(C.subject_seqs(C.SELECT_SUBJ), stride=4))):
        if args.smoke:
            ds.index = ds.index[:8]
        acc = {}
        with torch.no_grad():
            for items in C.loader(ds, 32, False, 0, workers=0 if args.cpu else args.workers):
                inp = batch_inputs(frozen, items, np.stack([x["gt_prev"] for x in items]), dev)
                blocks = C.block_ids(items)
                for h, (f, ro, s) in zip(heads, names):
                    accumulate(acc, f"{f}/{ro}/{s}", head_out(h, f, inp), inp, blocks)
                # support sizes per block (descriptive)
                for key, msk in (("n_O", inp["cap"]["O"] > 0), ("n_SO", (inp["S3"] > 0) & (inp["cap"]["O"] == 0))):
                    d = acc.setdefault("support", {}).setdefault(key, {})
                    for b, v in zip(blocks, msk.float().sum(1).cpu().numpy()):
                        d[int(b)] = d.get(int(b), 0.0) + np.array([1.0, v, 0, 0, 0, 0])
        np.savez_compressed(out / f"stats_{split}.npz", **pack(acc))
    C.save_json(out / "result.json", {"cond": args.cond, "feats": FEATS, "readouts": READOUTS, "seeds": SEEDS,
                                      "keys": KEYS, "names": ["/".join(map(str, n)) for n in names],
                                      "loss_curve": curve,
                                      "train_wall_s": time.time() - t0, "smoke": bool(args.smoke),
                                      "budget": {"steps": steps, "batch": batch, "lr": LR}}, exclusive=False)
    MC.receipt(out, {"stage": "b_vertex", "cond": args.cond, "smoke": bool(args.smoke)}, scripts=(__file__,))
    print("COMPLETE", out, flush=True)


if __name__ == "__main__":
    main()
