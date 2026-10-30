#!/usr/bin/env python3
"""MRF Stage C (docs/EXPERIMENT_RECORDS.md#mrf-prereg-20261024), descriptive: zero-training interventions on module G
of the frozen C1 (A0) and A3F0 (A3), and the sparsity robustness of the Stage-B probes.

    C1  truncation: from round k+1 on the messages are zero (the per-vertex update kept), k = 0, 1, 2, with the support
        growing as usual (`k?e`) or held at O (`k?f`); `k3f` = all three rounds with the support held (noprop on the
        frozen weights). Out-of-distribution interventions on a trained module: descriptive only.
    C2  pooling: the LBS slots pool over O instead of S (`poolO`; G itself unchanged)
        -> E3 (GT previous endpoint), E2 (C1 strict-trace incoming), E1 (own closed loop, the XCL E1 contract) on all
           2590 zgz endpoints
    C3  sparsity: B's M.N0d / M.L3 / M.L3np probes (E3) on zgz with every packet uniformly thinned to keep 1, 1/1.5,
        1/2.4 (the zgz_local event rate relative to training), 1/4
    C4  per packet: the propagated-only share of the LBS slot weight vs the frozen module's E2 / E3 error (Spearman)

    CUDA_VISIBLE_DEVICES=4 CUDA_DEVICE_ORDER=PCI_BUS_ID python tools/mrf/c_interv.py --cond A0 --part single \
        --out outputs/mrf/c/A0_single
"""
from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.agg778 import common as C  # noqa: E402
from tools.agg778 import b_task as BT  # noqa: E402
from tools.agg778.c_robust import thin  # noqa: E402
from tools.mrf import common as MC  # noqa: E402

VARIANTS = ("orig", "k0e", "k1e", "k2e", "k0f", "k1f", "k2f", "k3f", "poolO")
KEEP = (1.0, 1 / 1.5, 1 / 2.4, 1 / 4)
THIN_ARMS = ("M.N0d", "M.L3", "M.L3np")
THIN_SEED = 5353


def self_grow(layer, H, S, idx, emask, grow: bool):
    """A round without messages: S' = S or any(S_l) when `grow`, else S; H' = H + S' relu(W_s H)."""
    if grow:
        B, V = S.shape
        K = idx.shape[1]
        Sn = S[:, idx.reshape(-1)].view(B, V, K) * emask.unsqueeze(0).to(S.dtype)
        S = torch.maximum(S, Sn.amax(2))
    return H + S.unsqueeze(-1).to(H.dtype) * torch.relu(layer.slf(H)), S


@contextmanager
def variant(model, name):
    import semkine.xyz_mesh as XM
    from semkine.mrf import fixed_round
    if name not in VARIANTS:
        raise ValueError(name)
    layers = list(model.xyz.layers)
    native_pool = XM.lbs_slot_pool
    try:
        if name == "poolO":
            XM.lbs_slot_pool = lambda H, S, O, W, hard: native_pool(H, O, O, W, hard)
        elif name != "orig":
            k, grow = int(name[1]), name[2] == "e"
            for i, layer in enumerate(layers):
                if i < k and not grow:
                    layer.forward = (lambda H, S, gemb, idx, emask, l=layer: fixed_round(l, H, S, gemb, idx, emask))
                elif i >= k:
                    layer.forward = (lambda H, S, gemb, idx, emask, l=layer, g=grow: self_grow(l, H, S, idx, emask, g))
        yield
    finally:
        for layer in layers:
            layer.__dict__.pop("forward", None)
        XM.lbs_slot_pool = native_pool


def prop_share(O, S, W):
    """Per packet: mean over live LBS slots of the slot weight on propagated-only vertices."""
    a = W.unsqueeze(0) * S.unsqueeze(-1)
    live = a.sum(1) > 0
    p = (a * (1 - O).unsqueeze(-1)).sum(1) / a.sum(1).clamp_min(1e-9)
    return ((p * live).sum(1) / live.sum(1).clamp_min(1)).cpu().numpy()


@torch.no_grad()
def single_step(frozen, ds, afn, dev, mano, workers):
    from tools.xcl.evaluate import score
    preds, tgt, meta, share = [], [], [], []
    for items in C.loader(ds, 64, False, 0, workers=workers):
        anchors = np.stack([afn(x) for x in items])
        cap = C.forward_capture(frozen, C.make_batch(items, anchors, dev))
        preds.append(cap["out"].cpu().numpy())
        tgt.append(np.stack([x["gt_end"] for x in items]))
        meta += [(x["seq"], x["si"], x["k"], x["rid"], x["blk"]) for x in items]
        share.append(prop_share(cap["O"], cap["S"], frozen.xyz.lbs_w.float()))
    out = BT.finish([preds], None, tgt, meta, ds, mano, dev, score)
    out["prop_share"] = np.concatenate(share)
    return out


def means(res):
    o = {}
    for dom in ("local", "global"):
        sel = np.array([("_" + dom) in s for s in res["seq"]])
        o[dom] = {m: float(res[m][0, sel].mean()) for m in ("mpjpe_ra_mm", "mpvpe_ra_mm", "wrist_mm")}
    return o


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def part_single(args, frozen, cfg, dev, mano, out):
    trace = C.strict_trace()
    ds = C.Endpoints(C.zgz_seqs())
    if args.smoke:
        ds.index = ds.index[:64]
    res = {}
    for v in VARIANTS:
        with variant(frozen, v):
            for regime, afn in (("E3", lambda it: it["gt_prev"]), ("E2", lambda it: trace[it["seq"]][it["k"]])):
                r = single_step(frozen, ds, afn, dev, mano, args.workers)
                res[f"{v}/{regime}"] = means(r)
                np.savez_compressed(out / f"{v}_{regime}.npz", **{k: r[k] for k in ("seq", "si", "k", "rid", "blk",
                                                                                     "mpjpe_ra_mm", "prop_share")})
                if v == "orig":
                    c4 = {}
                    for dom in ("local", "global"):
                        sel = np.array([("_" + dom) in s for s in r["seq"]])
                        c4[dom] = {"spearman_share_vs_ra": spearman(r["prop_share"][sel], r["mpjpe_ra_mm"][0, sel]),
                                   "share_mean": float(r["prop_share"][sel].mean())}
                    res[f"C4/{regime}"] = c4
                print(v, regime, json.dumps(res[f"{v}/{regime}"]), flush=True)
    return res


def part_loop(args, frozen, cfg, dev, mano, out):
    from tools.agg778.a_audit import own_loop
    res = {}
    for v in args.variants.split(","):
        with variant(frozen, v):
            r, preds = own_loop(frozen, cfg, mano, dev, False)
        res[f"{v}/E1"] = r
        np.savez_compressed(out / f"{v}_E1_states.npz", **{s + "/prediction": p for s, p in preds.items()})
        print(v, "E1", json.dumps({s: r[s]["mpjpe_ra_mm"] for s in r}), flush=True)
    return res


@torch.no_grad()
def part_thin(args, frozen, cfg, dev, mano, out):
    from tools.mrf import b_task as MB
    from tools.xcl.evaluate import score
    st = torch.load(MC.OUT / "b" / "probe" / f"{args.cond}_e3_m1" / "arms.pt", map_location=dev)
    faces = frozen.mano.f.detach().long().cpu()
    arms, names = [], []
    for sd, (a, s) in zip(st["arms"], st["names"]):
        if a in THIN_ARMS:
            m = MB.Arm(a, frozen, faces).to(dev).eval()
            m.load_state_dict(sd)
            arms.append(m)
            names.append((a, s))
    need = MB.needs({a: 1 for a, _ in names})
    ds = C.Endpoints(C.zgz_seqs())
    if args.smoke:
        ds.index = ds.index[:64]
    res = {}
    for keep in KEEP:
        preds = [[] for _ in arms]
        base, tgt, meta = [], [], []
        for items in C.loader(ds, 64, False, 0, workers=args.workers):
            anchors = np.stack([it["gt_prev"] for it in items])
            if keep < 1.0:
                evs, srcs = thin(items, keep, THIN_SEED + items[0]["i"])
                items = [dict(it, ev5=e, src=s) for it, e, s in zip(items, evs, srcs)]
            inp = MB.batch_inputs(frozen, items, anchors, dev, MB.PERTURB_SEED + 2 * 10 ** 6 + items[0]["i"], need)
            for k, m in enumerate(arms):
                preds[k].append((inp["batch"].prev_state + m(inp)).float().cpu().numpy())
            base.append(inp["own"].cpu().numpy())
            tgt.append(np.stack([it["gt_end"] for it in items]))
            meta += [(it["seq"], it["si"], it["k"], it["rid"], it["blk"]) for it in items]
        r = BT.finish(preds, base, tgt, meta, ds, mano, dev, score)
        key = f"keep_{keep:.4f}"
        res[key] = {}
        for dom in ("local", "global"):
            sel = np.array([("_" + dom) in s for s in r["seq"]])
            res[key][dom] = {f"{a}/{s}": float(r["mpjpe_ra_mm"][j, sel].mean()) for j, (a, s) in enumerate(names)}
            res[key][dom]["frozen_own"] = float(r["mpjpe_ra_mm"][-1, sel].mean())
        np.savez_compressed(out / f"thin_{key}.npz", **{k: r[k] for k in ("seq", "si", "k", "rid", "blk", "mpjpe_ra_mm")})
        print(key, json.dumps(res[key]), flush=True)
    res["names"] = [f"{a}/{s}" for a, s in names]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", choices=("A0", "A3"), required=True)
    ap.add_argument("--part", choices=("single", "loop", "thin"), required=True)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = MC.require_gpu()
    if not args.smoke:
        MC.verify_protocol("C")
    C.set_policy()
    out = Path(args.out)
    if (out / "result.json").exists():
        raise SystemExit(f"{out} holds a result; preserve it")
    out.mkdir(parents=True, exist_ok=True)
    frozen, cfg = C.load_frozen(args.cond, dev)
    mano = C.mano32(cfg, dev)
    fn = {"single": part_single, "loop": part_loop, "thin": part_thin}[args.part]
    res = {"cond": args.cond, "part": args.part, "smoke": bool(args.smoke), **fn(args, frozen, cfg, dev, mano, out)}
    C.save_json(out / "result.json", res, exclusive=False)
    MC.receipt(out, {"stage": "c_interv", "cond": args.cond, "part": args.part, "smoke": bool(args.smoke)},
               scripts=(__file__,))
    print("COMPLETE", out, flush=True)


if __name__ == "__main__":
    main()
