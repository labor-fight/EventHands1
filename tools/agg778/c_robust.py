#!/usr/bin/env python3
"""AGG778 Stage C (docs/EXPERIMENT_RECORDS.md#agg778-prereg-20261024): robustness of the 778-vertex aggregation, zero
training of any model (the Stage-B tails / heads and the frozen modules are only evaluated), zgz E3 anchors.

    C1  event rate: every packet uniformly thinned to keep 1, 1/1.5, 1/2.4 (zgz_local vs training rate), 1/4
        -> H0 drift (log mass, peak, support) and the E3 RA of the task tails (T1, T2, T3, T6) and the frozen module;
        pre-registered flag: thinning 1/2.4 makes T2 worse by >= 1.5 mm (local) while T6 moves < 0.5 mm
    C2  association noise (A3): a fraction p in {0, 0.1, 0.25, 0.5} of the known-source events re-routed to a random
        1-ring neighbour (ring) or a random vertex of another LBS part (part) -> E3 RA of T2 / T4 / T5 / T6
    C3  mass semantics (A0, frozen C1): per-event q x U(0.5, 1.5) (support unchanged) and dropping every record of
        weight < 0.05 -> change of x, support and of the single-step output (RA mm between the two predictions)

    CUDA_VISIBLE_DEVICES=4 CUDA_DEVICE_ORDER=PCI_BUS_ID python tools/agg778/c_robust.py --out outputs/agg778/c
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

KEEP = (1.0, 1 / 1.5, 1 / 2.4, 1 / 4)
P_NOISE = (0.0, 0.1, 0.25, 0.5)
TAIL_ARMS = {"A0": ("T1", "T2", "T3", "T6"), "A3": ("T1", "T2", "T3", "T4", "T5", "T6")}


def load_tails(cond, frozen, dev, root=None):
    from tools.agg778.heads import Tail
    st = torch.load(Path(root or C.OUT / "b_task") / cond / "tails.pt", map_location=dev)
    tails, names = [], []
    for sd, (arm, s) in zip(st["tails"], st["names"]):
        if arm in TAIL_ARMS[cond]:
            t = Tail(BT.F_IN[arm], frozen.xyz).to(dev).eval()
            t.load_state_dict(sd)
            tails.append(t)
            names.append((arm, s))
    return tails, names


def thin(items, keep, seed):
    rng = np.random.default_rng(seed)
    evs, srcs = [], []
    for it in items:
        m = rng.random(len(it["ev5"])) < keep
        evs.append(it["ev5"][m])
        srcs.append({k: np.asarray(v)[m] for k, v in it["src"].items()})
    return evs, srcs


@contextmanager
def route_noise(model, p, mode, seed):
    """Re-route a fraction p of the known-source (q > 0) events after the A3 full-oracle routing."""
    import semkine.xag as X
    native = X.route_association
    ring = model.xyz.ring_idx
    rmask = model.xyz.ring_mask
    hard = model.xyz.lbs_hard
    g = torch.Generator(device=ring.device).manual_seed(seed)

    def patched(*args, **kwargs):
        assoc, audit = native(*args, **kwargs)
        if p <= 0:
            return assoc, audit
        out = dict(assoc)
        vid = assoc["vid"].clone()
        pick = (torch.rand(assoc["q"].shape, generator=g, device=vid.device) < p) & (assoc["q"] > 0)
        if mode == "ring":
            k = torch.randint(0, ring.shape[1], vid.shape, generator=g, device=vid.device)
            nb = ring[vid, k]
            ok = rmask[vid, k] > 0
            new = torch.where(ok, nb, vid)
        else:
            cand = torch.randint(0, C.V, vid.shape, generator=g, device=vid.device)
            for _ in range(8):                                                   # resample until another part
                bad = hard[cand] == hard[vid]
                cand = torch.where(bad, torch.randint(0, C.V, vid.shape, generator=g, device=vid.device), cand)
            new = cand
        out["vid"] = torch.where(pick.unsqueeze(-1), new, vid)
        return out, audit
    X.route_association = patched
    try:
        yield
    finally:
        X.route_association = native


@torch.no_grad()
def tails_e3(frozen, cond, tails, names, ds, dev, mano, workers, mutate=None):
    """E3 single step of every tail (and the frozen module) on zgz, optionally with mutated packets."""
    from tools.xcl.evaluate import score
    preds = [[] for _ in tails]
    base, tgt, meta, h0 = [], [], [], []
    for items in C.loader(ds, 64, False, 0, workers=workers):
        anchors = np.stack([it["gt_prev"] for it in items])
        if mutate is not None:
            evs, srcs = mutate(items)
            items = [dict(it, ev5=e, src=s) for it, e, s in zip(items, evs, srcs)]
        batch, cap, feats, edge = BT.batch_inputs(frozen, cond, items, anchors, dev, BT.PERTURB_SEED + 3 * 10 ** 6 +
                                                  items[0]["i"])
        for k, ((arm, _), tail) in enumerate(zip(names, tails)):
            preds[k].append((batch.prev_state + BT.tail_delta(tail, arm, cap, feats, edge)).cpu().numpy())
        base.append(cap["out"].cpu().numpy())
        tgt.append(np.stack([it["gt_end"] for it in items]))
        meta += [(it["seq"], it["si"], it["k"], it["rid"], it["blk"]) for it in items]
        O = cap["O"] > 0
        h0.append(np.stack([O.float().sum(1).cpu().numpy(),
                            (cap["x257"][..., -1] * O).sum(1).cpu().numpy() / O.sum(1).clamp(min=1).cpu().numpy(),
                            (cap["x257"][..., 128:256].abs().mean(-1) * O).sum(1).cpu().numpy()
                            / O.sum(1).clamp(min=1).cpu().numpy()], 1))
    res = BT.finish(preds, base, tgt, meta, ds, mano, dev, score)
    h0 = np.concatenate(h0)
    out = {}
    for dom in ("local", "global"):
        sel = np.array([("_" + dom) in s for s in res["seq"]])
        if not sel.any():
            continue
        out[dom] = {f"{a}/{s}": float(res["mpjpe_ra_mm"][j, sel].mean()) for j, (a, s) in enumerate(names)}
        out[dom]["frozen_own"] = float(res["mpjpe_ra_mm"][-1, sel].mean())
        out[dom]["h0"] = {"support": float(h0[sel, 0].mean()), "logmass": float(h0[sel, 1].mean()),
                          "peak_abs": float(h0[sel, 2].mean())}
    return out


def arm_avg(r, arm):
    return float(np.mean([v for k, v in r.items() if k.startswith(arm + "/")]))


@torch.no_grad()
def c3(frozen, ds, dev, mano, workers):
    """Frozen C1: relative-q perturbation and dropping low-weight records; change of x, support and output."""
    from tools.xcl.evaluate import score
    import semkine.xyz_mesh as XM
    rows = {"q_rel": [], "drop_low": []}
    native = frozen.xyz.forward
    for items in C.loader(ds, 32, False, 0, workers=workers):
        anchors = np.stack([it["gt_prev"] for it in items])
        batch = C.make_batch(items, anchors, dev)
        cap0 = C.forward_capture(frozen, batch)
        for name in rows:
            g = torch.Generator(device=dev).manual_seed(11)

            def patched(tok, d, mask, verts, assoc, prev48, **kw):
                a = dict(assoc)
                if name == "q_rel":
                    f = 0.5 + torch.rand(a["q"].shape, generator=g, device=dev)
                    a["q"] = (a["q"] * f).clamp(max=1.0) * (a["q"] > 0)
                else:
                    w = a["q"].unsqueeze(-1) * a["A"]
                    a["A"] = torch.where(w < 0.05, torch.zeros_like(a["A"]), a["A"])
                return native(tok, d, mask, verts, a, prev48, **kw)
            frozen.xyz.forward = patched
            try:
                cap1 = C.forward_capture(frozen, batch)
            finally:
                frozen.xyz.forward = native
            O0, O1 = cap0["O"] > 0, cap1["O"] > 0
            for b, it in enumerate(items):
                beta = torch.tensor(it["betas"], device=dev).view(1, -1)
                sc = score(mano, cap1["out"][b:b + 1].cpu().numpy(), cap0["out"][b:b + 1].cpu().numpy(), beta, dev)
                rows[name].append({"seq": it["seq"], "out_ra_mm": float(sc["mpjpe_ra_mm"][0]),
                                   "support_changed": float((O0[b] != O1[b]).float().sum()),
                                   "x_rel_change": float((cap1["x257"][b] - cap0["x257"][b]).norm()
                                                         / cap0["x257"][b].norm().clamp(min=1e-12))})
    out = {}
    for name, rs in rows.items():
        for dom in ("local", "global"):
            sel = [r for r in rs if ("_" + dom) in r["seq"]]
            if not sel:
                continue
            out[f"{name}/{dom}"] = {k: float(np.mean([r[k] for r in sel])) for k in ("out_ra_mm", "support_changed",
                                                                                     "x_rel_change")}
            out[f"{name}/{dom}"]["out_ra_mm_p90"] = float(np.percentile([r["out_ra_mm"] for r in sel], 90))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--b-task-dir", default=None, help="smoke only: tails of a smoke B-task run")
    args = ap.parse_args()
    if args.b_task_dir and not args.smoke:
        raise SystemExit("formal Stage C reads the formal Stage-B tails")
    dev = C.require_gpu()
    if not args.smoke:
        C.verify_protocol()
    C.set_policy()
    out = Path(args.out)
    if (out / "result.json").exists():
        raise SystemExit(f"{out} holds a result; preserve it")
    out.mkdir(parents=True, exist_ok=True)
    res = {"C1": {}, "C2": {}}
    for cond in ("A0", "A3"):
        frozen, cfg = C.load_frozen(cond, dev)
        mano = C.mano32(cfg, dev)
        tails, names = load_tails(cond, frozen, dev, args.b_task_dir)
        ds = C.Endpoints(C.zgz_seqs())
        if args.smoke:
            ds.index = ds.index[:64]
        for keep in KEEP:
            mut = None if keep == 1.0 else (lambda items, k=keep: thin(items, k, 20261024 + items[0]["i"]))
            res["C1"][f"{cond}/keep{keep:.3f}"] = tails_e3(frozen, cond, tails, names, ds, dev, mano, args.workers, mut)
        if cond == "A3":
            for mode in ("ring", "part"):
                for p in P_NOISE:
                    with route_noise(frozen, p, mode, 7):
                        res["C2"][f"{mode}/p{p}"] = tails_e3(frozen, cond, tails, names, ds, dev, mano, args.workers)
        if cond == "A0":
            res["C3"] = c3(frozen, ds, dev, mano, args.workers)
        del frozen
        torch.cuda.empty_cache()
    flags = {}
    for cond in ("A0", "A3"):
        if "local" not in res["C1"][f"{cond}/keep1.000"]:
            continue
        r1, r24 = res["C1"][f"{cond}/keep1.000"]["local"], res["C1"][f"{cond}/keep0.417"]["local"]
        d_t2, d_t6 = arm_avg(r24, "T2") - arm_avg(r1, "T2"), arm_avg(r24, "T6") - arm_avg(r1, "T6")
        flags[cond] = {"T2_local_delta": d_t2, "T6_local_delta": d_t6, "rate_flag": bool(d_t2 >= 1.5 and d_t6 < 0.5)}
    res["flags"] = flags
    res["rate_flag"] = any(f["rate_flag"] for f in flags.values())
    C.save_json(out / "result.json", res, exclusive=False)
    C.receipt(out, {"stage": "c_robust", "smoke": bool(args.smoke)}, scripts=(__file__, Path(__file__).with_name("b_task.py")))
    print(json.dumps(flags, indent=1), flush=True)


if __name__ == "__main__":
    main()
