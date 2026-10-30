#!/usr/bin/env python3
"""AGG778 Stage B-task (docs/EXPERIMENT_RECORDS.md#agg778-prereg-20261024): task-level isolation of the 778-vertex
aggregation. One process = one association condition:

    A0  C1 routing, learned module = frozen C1 (step 6000)
    A3  full_oracle routing, learned module = frozen XCL A3F0 (step 6000; oracle, not deployable)

Every arm feeds a fresh instance of C1's own downstream (`heads.Tail`) with a different per-vertex content on the same
support O and the same residual slot; all tails train side by side on the same batches (7 fit subjects, 50 ms windows,
anchors = GT at the previous endpoint + C1's mixed prev noise), each with its own Adam and clipping, 3 seeds per arm.

    T0 state only (no per-vertex content, no support, no residual slot)   T1 log1p(mass), has
    T2 frozen H0          T3 frozen pre-vin x (257)
    T4 R1: per-vertex weighted mean of the record dXYZ (A3)             T5 R2: material-point displacement mean (A3)
    T6 R0: GT vertex displacement on O    T6sX: R0 + N(0, (X mm)^2) per axis (resolution dose-response)

Readouts: selection subjects (ycy, ylf; every 4th endpoint) E3 and R-noise; zgz E3, R-noise, E2 (C1 strict-trace
incoming) and the closed loop E1 (all tails in lockstep, XCL initial noise). Writes per-endpoint metrics only.

    CUDA_VISIBLE_DEVICES=4 CUDA_DEVICE_ORDER=PCI_BUS_ID python tools/agg778/b_task.py --cond A3 --out outputs/agg778/b_task/A3
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.agg778 import common as C  # noqa: E402
from tools.agg778.heads import Tail  # noqa: E402

ARMS = {"A0": ("T0", "T1", "T2", "T3", "T6", "T6s3", "T6s6", "T6s12"),
        "A3": ("T0", "T1", "T2", "T3", "T4", "T5", "T6", "T6s3", "T6s6", "T6s12")}
F_IN = {"T0": 1, "T1": 2, "T2": 128, "T3": 257, "T4": 3, "T5": 3, "T6": 3, "T6s3": 3, "T6s6": 3, "T6s12": 3}
SEEDS = (0, 1, 2)
STEPS, BATCH, LR, WARMUP, LR_FLOOR, CLIP = 4000, 128, 2e-3, 300, 0.02, 10.0
NOISE_SEED_TRAIN, NOISE_SEED_EVAL, PERTURB_SEED = 3407, 777, 4242
METRICS = ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm", "wrist_mm")


def lr_lambda(step, steps):
    if step < WARMUP:
        return float(step + 1) / WARMUP
    p = min(max((step - WARMUP) / float(steps - WARMUP), 0.0), 1.0)
    return LR_FLOOR + (1.0 - LR_FLOOR) * 0.5 * (1.0 + math.cos(math.pi * p))


@torch.no_grad()
def batch_inputs(frozen, cond, items, anchors, dev, step_seed: int):
    """Frozen forward + every arm's per-vertex content for one packet batch."""
    from xvm.omega_r import record_offsets
    batch = C.make_batch(items, anchors, dev)
    cap = C.forward_capture(frozen, batch)
    vgt = C.fk_verts(frozen, batch.target, batch)
    R0 = (vgt - cap["verts"]) / 0.01                                                  # L_REL units
    feats = {"R0": R0}
    if cond == "A3":
        r1, _ = record_offsets(cap["d"], cap["mask"], cap["verts"], cap["assoc"], "R1")
        r2, _ = record_offsets(cap["d"], cap["mask"], cap["verts"], cap["assoc"], "R2")
        feats["R1"], feats["R2"] = r1 / 0.01, r2 / 0.01
    g = torch.Generator(device=dev).manual_seed(int(step_seed))
    for s in (3, 6, 12):
        feats[f"R0s{s}"] = R0 + torch.randn(R0.shape, generator=g, device=dev) * (s * 1e-3 / 0.01)
    edge = frozen.xyz.edge_geometry(cap["verts"])
    return batch, cap, feats, edge


def arm_features(arm, cap, feats):
    O = cap["O"]
    B = O.shape[0]
    if arm == "T0":
        return O.new_zeros(B, C.V, 1), torch.zeros_like(O), torch.zeros_like(cap["bslot"])
    if arm == "T1":
        f = torch.stack([cap["x257"][..., -1], O], -1)
    elif arm == "T2":
        f = cap["H0"]
    elif arm == "T3":
        f = cap["x257"]
    elif arm == "T4":
        f = feats["R1"]
    elif arm == "T5":
        f = feats["R2"]
    elif arm == "T6":
        f = feats["R0"]
    else:
        f = feats["R0s" + arm[3:]]
    return f * O.unsqueeze(-1), O, cap["bslot"]


def tail_delta(tail, arm, cap, feats, edge):
    f, O, bslot = arm_features(arm, cap, feats)
    delta = tail(f, O, edge, bslot, cap["prev48"])
    return torch.where((cap["counts"] <= 0).unsqueeze(-1), torch.zeros_like(delta), delta)


def build_tails(cond, frozen, dev):
    tails, opts, scheds, names = [], [], [], []
    for ai, arm in enumerate(ARMS[cond]):
        for s in SEEDS:
            with torch.random.fork_rng(devices=[dev]):
                torch.manual_seed(1000 * (ai + 1) + s)
                t = Tail(F_IN[arm], frozen.xyz).to(dev)
            o = torch.optim.Adam(t.parameters(), lr=LR)
            tails.append(t)
            opts.append(o)
            scheds.append(torch.optim.lr_scheduler.LambdaLR(o, lambda it: lr_lambda(it, STEPS)))
            names.append((arm, s))
    return tails, opts, scheds, names


def train(args, cond, frozen, cfg, dev, log):
    fit = C.Endpoints(C.subject_seqs(C.FIT_SUBJ))
    ld = C.loader(fit, BATCH, True, 3407, workers=args.workers, drop_last=True)
    draw = C.noise_sampler(cfg)
    rng = np.random.default_rng(NOISE_SEED_TRAIN)
    tails, opts, scheds, names = build_tails(cond, frozen, dev)
    log.update(n_fit=len(fit), tails=[f"{a}/{s}" for a, s in names], precision="bf16-mixed train, fp32 eval",
               params={a: sum(p.numel() for p in tails[i].parameters()) for i, (a, s) in enumerate(names) if s == 0})
    run = np.zeros(len(tails))
    t0, it, curve = time.time(), 0, []
    while it < args.steps:
        for items in ld:
            noise = np.stack([draw(rng)[0] for _ in items])
            anchors = np.stack([it_["gt_prev"] for it_ in items]) + noise
            batch, cap, feats, edge = batch_inputs(frozen, cond, items, anchors, dev, PERTURB_SEED + it)
            for k, ((arm, _), tail, opt, sch) in enumerate(zip(names, tails, opts, scheds)):
                with torch.autocast("cuda", dtype=torch.bfloat16):                    # C1's own bf16-mixed training
                    delta = tail_delta(tail, arm, cap, feats, edge)
                pred = batch.prev_state + delta.float()
                loss, _ = frozen._compute_loss(pred, batch.target)
                loss = loss.float().log10()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(tail.parameters(), CLIP)
                opt.step()
                sch.step()
                run[k] += float(loss)
            it += 1
            if it % 100 == 0 or it == args.steps:
                n = 100 if it % 100 == 0 else it % 100
                ent = {f"{a}/{s}": round(run[k] / n, 4) for k, (a, s) in enumerate(names)}
                curve.append({"it": it, **ent})
                print(it, json.dumps(ent), f"{time.time() - t0:.0f} s", flush=True)
                run[:] = 0
            if it >= args.steps:
                break
    log["loss_curve"] = curve
    log["train_wall_s"] = time.time() - t0
    torch.save({"tails": [t.state_dict() for t in tails], "names": names}, args.out / "tails.pt")
    return tails, names


@torch.no_grad()
def single_step(frozen, cond, tails, names, ds, anchor_fn, dev, mano, workers):
    """Every tail's single-step prediction from the given anchors on every endpoint of `ds`."""
    from tools.xcl.evaluate import score
    preds = [[] for _ in tails]
    tgt, meta, base = [], [], []
    for items in C.loader(ds, 64, False, 0, workers=workers):
        anchors = np.stack([anchor_fn(it) for it in items])
        batch, cap, feats, edge = batch_inputs(frozen, cond, items, anchors, dev, PERTURB_SEED + 10 ** 6 + items[0]["i"])
        for k, ((arm, _), tail) in enumerate(zip(names, tails)):
            preds[k].append((batch.prev_state + tail_delta(tail, arm, cap, feats, edge)).cpu().numpy())
        base.append(cap["out"].cpu().numpy())
        tgt.append(np.stack([it["gt_end"] for it in items]))
        meta += [(it["seq"], it["si"], it["k"], it["rid"], it["blk"]) for it in items]
    return finish(preds, base, tgt, meta, ds, mano, dev, score)


def finish(preds, base, tgt, meta, ds, mano, dev, score):
    """Per-endpoint metrics of every tail (rows 0..K-1) and, when `base` is given, of the frozen model's own
    single-step output from the same anchors (row K)."""
    tgt = np.concatenate(tgt)
    P = np.stack([np.concatenate(p) for p in preds])                                   # (tails, n, 51)
    Pb = np.concatenate(base) if base is not None else None
    seqs = np.array([m[0] for m in meta])
    out = {"seq": seqs, "si": np.array([m[1] for m in meta]), "k": np.array([m[2] for m in meta]),
           "rid": np.array([m[3] for m in meta]), "blk": np.array([m[4] for m in meta]),
           "pred": P.astype(np.float32), "target": tgt.astype(np.float32)}
    rows = len(preds) + (Pb is not None)
    mets = {m: np.zeros((rows, len(tgt)), np.float32) for m in METRICS}
    for s in np.unique(seqs):
        sel = seqs == s
        beta = torch.tensor(ds.seqs[[q.name for q in ds.seqs].index(s)].betas, device=dev).view(1, -1)
        for k in range(rows):
            pr = Pb[sel] if k == len(preds) else P[k][sel]
            sc = score(mano, np.ascontiguousarray(pr, np.float32), np.ascontiguousarray(tgt[sel], np.float32), beta, dev)
            for m in METRICS:
                mets[m][k, sel] = sc[m]
    out.update({m: v for m, v in mets.items()})
    return out


@torch.no_grad()
def closed_loop(frozen, cond, cfg, tails, names, dev, mano):
    """E1: all tails track zgz in lockstep (same packets, own previous predictions), XCL initial noise (rng 0)."""
    from semkine import eval_track as ET
    from tools.xcl.evaluate import score
    ds = C.Endpoints(C.zgz_seqs())
    rng = np.random.default_rng(0)
    K = len(tails)
    preds = [[] for _ in tails]
    tgt, meta = [], []
    cur = None
    for i in range(len(ds)):
        it = ds[i]
        if it["first"]:
            a = int(ds.seqs[it["si"]].runs[it["rid"]][0])
            init = ds.seqs[it["si"]].pos51[a] + ET.sample_init_noise(cfg, rng, 1.0)
            cur = np.repeat(init[None].astype(np.float32), K, 0)
        batch, cap, feats, edge = batch_inputs(frozen, cond, [it] * K, cur, dev, PERTURB_SEED + 2 * 10 ** 6 + i)
        nxt = np.zeros_like(cur)
        for k, ((arm, _), tail) in enumerate(zip(names, tails)):
            d = tail_delta(tail, arm, cap, feats, edge)
            nxt[k] = (batch.prev_state[k] + d[k]).cpu().numpy()
        cur = nxt
        for k in range(K):
            preds[k].append(cur[k:k + 1])
        tgt.append(it["gt_end"][None])
        meta.append((it["seq"], it["si"], it["k"], it["rid"], it["blk"]))
    n_expected = sum(C.ZGZ_COUNTS.values())
    if len(meta) != n_expected:
        raise RuntimeError("incomplete E1 denominator")
    return finish(preds, None, tgt, meta, ds, mano, dev, score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", choices=("A0", "A3"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--smoke", action="store_true", help="tiny run, not a result")
    args = ap.parse_args()
    dev = C.require_gpu()
    if not args.smoke:
        C.verify_protocol()
        if args.steps != STEPS:
            raise SystemExit("formal runs use the pre-registered step budget")
    C.set_policy()
    args.out = Path(args.out)
    if (args.out / "result.json").exists():
        raise SystemExit(f"{args.out} holds a result; preserve it")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(3407)
    frozen, cfg = C.load_frozen(args.cond, dev)
    mano = C.mano32(cfg, dev)
    log = {"cond": args.cond, "args": {k: str(v) for k, v in vars(args).items()},
           "budget": {"steps": args.steps, "batch": BATCH, "lr": LR, "warmup": WARMUP, "clip": CLIP}}
    tails, names = train(args, args.cond, frozen, cfg, dev, log)
    for t in tails:
        t.eval()
    draw = C.noise_sampler(cfg)
    trace = C.strict_trace()
    readouts = {}
    sel = C.Endpoints(C.subject_seqs(C.SELECT_SUBJ), stride=4)
    zg = C.Endpoints(C.zgz_seqs())
    if args.smoke:
        sel.index, zg.index = sel.index[:64], zg.index[:64]
    for name, ds in (("sel", sel), ("zgz", zg)):
        rng = np.random.default_rng(NOISE_SEED_EVAL)
        noise = {i: draw(rng) for i in range(len(ds))}
        readouts[f"{name}_e3"] = single_step(frozen, args.cond, tails, names, ds, lambda it: it["gt_prev"], dev, mano,
                                             args.workers)
        readouts[f"{name}_noise"] = single_step(frozen, args.cond, tails, names, ds,
                                                lambda it: it["gt_prev"] + noise[it["i"]][0], dev, mano, args.workers)
        readouts[f"{name}_noise"]["noise_kind"] = np.array([noise[i][1] for i in range(len(ds))])
        if name == "zgz":
            readouts["zgz_e2"] = single_step(frozen, args.cond, tails, names, ds,
                                             lambda it: trace[it["seq"]][it["k"]], dev, mano, args.workers)
    if not args.smoke:
        readouts["zgz_e1"] = closed_loop(frozen, args.cond, cfg, tails, names, dev, mano)
    for k, v in readouts.items():
        np.savez_compressed(args.out / f"eval_{k}.npz", **v)
    summary = {}
    for k, v in readouts.items():
        for s in np.unique(v["seq"]):
            sel_ = v["seq"] == s
            summary[f"{k}/{s}"] = {f"{a}/{sd}": float(v["mpjpe_ra_mm"][j, sel_].mean()) for j, (a, sd) in enumerate(names)}
            if v["mpjpe_ra_mm"].shape[0] > len(names):
                summary[f"{k}/{s}"]["frozen_own"] = float(v["mpjpe_ra_mm"][-1, sel_].mean())
    log["summary_mpjpe_ra"] = summary
    log["names"] = [f"{a}/{s}" for a, s in names]
    C.save_json(args.out / "result.json", log, exclusive=False)
    C.receipt(args.out, {"stage": "b_task", "cond": args.cond, "smoke": bool(args.smoke)},
              scripts=(__file__, Path(__file__).with_name("heads.py")))
    print("COMPLETE", args.out, flush=True)


if __name__ == "__main__":
    main()
