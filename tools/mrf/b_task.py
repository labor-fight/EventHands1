#!/usr/bin/env python3
"""MRF Stage B task-level probes (docs/EXPERIMENT_RECORDS.md#mrf-prereg-20261024): module G isolated under the AGG778
R2 instrument (the state-blind tail: prev48 = 0, residual slot 0, the anchor's edge geometry kept; bf16-mixed training,
fp32 evaluation, deterministic algorithms; 4000 steps x 128, Adam 2e-3, warmup 300, cosine to 2 %, clip 10; fit = the
7 fit subjects; evaluation on the selection subjects ycy / ylf every 4th endpoint and on all 2590 zgz endpoints).

Arm = <input>.<G>. G is the receptive-field variant of the tail (`tools/mrf/heads.G_OF`); the input family decides
what reaches G on the direct support O of the frozen module (C1 for A0, A3F0 for A3):

    M       fresh phi + C1 pooling, trained with G (the AGG778 R2 M13 form)
    H       the frozen module's own H0
    T0s     nothing (no events, no state): calibration
    T6      GT vertex displacement / L_REL;   T6s12: + N(0, 12 mm) per axis
    T6out   T6 with 10 % of O carrying the GT displacement of a vertex of another hard part (outliers)
    T6ap    aperture: on O's contour vertices (|n.d| < 0.4) only the normal component of the GT image displacement
            (pixels / rho, rho = f 1 cm / z) and the image normal direction

Under one seed index every G variant of one input family starts from the same values for the common parameters (one
seed per family, the extra blocks built last); all arms of a process train on the same batches.

    CUDA_VISIBLE_DEVICES=4 CUDA_DEVICE_ORDER=PCI_BUS_ID python tools/mrf/b_task.py --cond A3 --regime e3 --group m1 \
        --out outputs/mrf/b/probe/A3_e3_m1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.agg778 import common as C  # noqa: E402
from tools.agg778 import b_task as BT  # noqa: E402
from tools.agg778.heads import RecordOp, c1_records  # noqa: E402
from tools.mrf import common as MC  # noqa: E402
from tools.mrf.heads import G_OF, MRFTail  # noqa: E402

E3_T1 = {"T0s.L3": 3, "T6.L3": 3, "T6s12.L3": 3, "T6.N0d": 1, "T6.L3np": 1, "T6.K12np": 1, "T6.Gg": 1, "T6.Gl": 1}
GROUPS = {
    ("A3", "e3", "m1"): {"M.N0d": 3, "M.L3": 3, "M.L3np": 3},
    ("A3", "e3", "m2"): {"M.K12np": 3, "M.K12": 1},
    ("A3", "e3", "m3"): {"M.Gg": 3, "M.Gl": 3},
    ("A3", "e3", "m4"): {"M.N0": 1, "M.Dfix": 1, "M.Lg0": 1, "H.N0d": 1, "H.L3": 1, "H.L3np": 1},
    ("A3", "e3", "t1"): dict(E3_T1),
    ("A3", "e3", "c1"): {"T6s12.N0d": 3, "T6s12.K12np": 3, "T6s12.Gg": 3,
                         "T6out.N0d": 3, "T6out.L3": 3, "T6out.K12np": 3, "T6out.Gg": 3},
    ("A3", "e3", "c2"): {"T6ap.N0d": 3, "T6ap.L3": 3, "T6ap.K12np": 3, "T6ap.Gg": 3},
    ("A3", "noise", "n1"): {"T0s.L3": 3, "M.N0d": 3, "M.L3": 3, "M.L3np": 3, "M.Gg": 3},
    ("A0", "e3", "m1"): {"M.N0d": 3, "M.L3": 3, "M.L3np": 3},
    ("A0", "e3", "m2"): {"M.K12np": 3, "M.K12": 1},
    ("A0", "e3", "m3"): {"M.Gg": 3, "M.Gl": 3},
    ("A0", "e3", "m4"): {"M.N0": 1, "M.Dfix": 1, "M.Lg0": 1, "H.N0d": 1, "H.L3": 1, "H.L3np": 1},
    ("A0", "e3", "t1"): dict(E3_T1),
    ("A0", "noise", "n1"): {"T0s.L3": 3, "M.N0d": 3, "M.L3": 3, "M.L3np": 3, "M.Gg": 3},
}
FAMILY_SEED = {"M": 30000, "H": 30100, "T0s": 30200, "T6": 30300, "T6s12": 30400, "T6out": 30500, "T6ap": 30600}
STEPS, BATCH, LR, WARMUP, LR_FLOOR, CLIP = 4000, 128, 2e-3, 300, 0.02, 10.0
NOISE_SEED_TRAIN, NOISE_SEED_EVAL, PERTURB_SEED = 3407, 777, 4242
OUTLIER_FRAC, NOISE_MM, CONTOUR_COS = 0.10, 12.0, 0.4


def lr_lambda(step, steps):
    if step < WARMUP:
        return float(step + 1) / WARMUP
    p = min(max((step - WARMUP) / float(steps - WARMUP), 0.0), 1.0)
    return LR_FLOOR + (1.0 - LR_FLOOR) * 0.5 * (1.0 + np.cos(np.pi * p))


def needs(spec):
    fams = {a.split(".")[0] for a in spec}
    return {"rec": "M" in fams, "R0s12": "T6s12" in fams, "R0out": "T6out" in fams, "ap": "T6ap" in fams}


@torch.no_grad()
def outliers(R0, O, hard, seed):
    """10 % of every packet's O (rounded) take the GT displacement of a random O vertex of another hard part (any
    vertex of another part when O has none)."""
    rng = np.random.default_rng(seed)
    On, hd = (O > 0).cpu().numpy(), hard.cpu().numpy()
    out = R0.clone()
    for b in range(On.shape[0]):
        ob = np.nonzero(On[b])[0]
        n = int(round(OUTLIER_FRAC * len(ob)))
        if n == 0:
            continue
        pick = rng.choice(ob, size=n, replace=False)
        src = np.empty(n, np.int64)
        for i, v in enumerate(pick):
            pool = ob[hd[ob] != hd[v]]
            if len(pool) == 0:
                pool = np.nonzero(hd != hd[v])[0]
            src[i] = pool[rng.integers(len(pool))]
        out[b, torch.from_numpy(pick).to(R0.device)] = R0[b, torch.from_numpy(src).to(R0.device)]
    return out


@torch.no_grad()
def aperture(frozen, batch, verts, vgt, O):
    """(B, V, 3) [normal component of the GT image displacement / rho, image normal direction] and the contour support
    O & (|n.d| < 0.4); rho = f 1 cm / z at the anchor vertex."""
    from semkine.xyz_mesh import vertex_normals
    fx, fy, cx, cy = C.intrinsics(frozen, batch)
    duv = C.project(vgt, fx, fy, cx, cy) - C.project(verts, fx, fy, cx, cy)
    n = vertex_normals(verts.float(), frozen.mano.f)
    X, Y, Z = verts[..., 0], verts[..., 1], verts[..., 2].clamp_min(1e-6)
    ex = fx[:, None] * (n[..., 0] / Z - n[..., 2] * X / Z ** 2)
    ey = fy[:, None] * (n[..., 1] / Z - n[..., 2] * Y / Z ** 2)
    e = torch.stack([ex, ey], -1)
    e1 = e / e.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    dhat = verts / verts.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    contour = (n * dhat).sum(-1).abs() < CONTOUR_COS
    rho = torch.sqrt(fx * fy)[:, None] * 0.01 / Z
    s = (e1 * duv).sum(-1) / rho
    Oap = O * contour.float()
    return torch.cat([s.unsqueeze(-1), e1], -1), Oap


@torch.no_grad()
def batch_inputs(frozen, items, anchors, dev, seed, need):
    batch = C.make_batch(items, anchors, dev)
    cap = C.forward_capture(frozen, batch)
    B = cap["O"].shape[0]
    vgt = C.fk_verts(frozen, batch.target, batch)
    R0 = (vgt - cap["verts"]) / 0.01
    inp = {"batch": batch, "cap": cap, "B": B, "R0": R0, "edge": frozen.xyz.edge_geometry(cap["verts"]),
           "own": cap["out"]}
    if need["rec"]:
        inp["rec"] = c1_records(cap)
    if need["R0s12"]:
        g = torch.Generator(device=dev).manual_seed(int(seed))
        inp["R0s12"] = R0 + torch.randn(R0.shape, generator=g, device=dev) * (NOISE_MM * 1e-3 / 0.01)
    if need["R0out"]:
        inp["R0out"] = outliers(R0, cap["O"], frozen.xyz.lbs_hard, int(seed))
    if need["ap"]:
        inp["AP"], inp["Oap"] = aperture(frozen, batch, cap["verts"], vgt, cap["O"])
    return inp


class Arm(nn.Module):
    def __init__(self, name, frozen, faces):
        super().__init__()
        self.name = name
        self.family, g = name.split(".")
        self.op = None
        if self.family == "M":
            self.op = RecordOp(extra=False, pool="c1")
            f_in = self.op.out_dim
        else:
            f_in = {"H": 128, "T0s": 1}.get(self.family, 3)
        self.tail = MRFTail(f_in, frozen.xyz, G_OF[g], faces=faces)

    def features(self, inp):
        cap, B = inp["cap"], inp["B"]
        O = cap["O"]
        k = self.family
        if k == "T0s":
            return O.new_zeros(B, C.V, 1), torch.zeros_like(O)
        if k == "M":
            return self.op(inp["rec"], B)
        if k == "H":
            return cap["H0"], O
        if k == "T6ap":
            return inp["AP"], inp["Oap"]
        return inp[{"T6": "R0", "T6s12": "R0s12", "T6out": "R0out"}[k]], O

    def forward(self, inp):
        f, O = self.features(inp)
        f = f * (O > 0).unsqueeze(-1).to(f.dtype)
        B = inp["B"]
        delta = self.tail(f, O, inp["edge"], f.new_zeros(B, 2 * 128 + 2), f.new_zeros(B, 48), inp["cap"]["verts"])
        return torch.where((inp["cap"]["counts"] <= 0).unsqueeze(-1), torch.zeros_like(delta), delta)


def build(spec, frozen, dev, steps):
    faces = frozen.mano.f.detach().long().cpu()
    arms, names, opts, scheds = [], [], [], []
    for a, n in spec.items():
        for s in range(n):
            with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):
                torch.manual_seed(FAMILY_SEED[a.split(".")[0]] + s)
                m = Arm(a, frozen, faces).to(dev)
            o = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=LR)
            arms.append(m)
            names.append((a, s))
            opts.append(o)
            scheds.append(torch.optim.lr_scheduler.LambdaLR(o, lambda it, n_=steps: lr_lambda(it, n_)))
    return arms, names, opts, scheds


def train(args, frozen, cfg, dev, spec, need, log):
    fit = C.Endpoints(C.subject_seqs(C.FIT_SUBJ))
    if args.smoke:
        fit.index = fit.index[:4 * args.batch]
    ld = C.loader(fit, args.batch, True, 3407, workers=args.workers, drop_last=True)
    draw = C.noise_sampler(cfg)
    rng = np.random.default_rng(NOISE_SEED_TRAIN)
    arms, names, opts, scheds = build(spec, frozen, dev, args.steps)
    log.update(n_fit=len(fit), arms=[f"{a}/{s}" for a, s in names],
               params={f"{a}/{s}": sum(p.numel() for p in m.parameters() if p.requires_grad)
                       for m, (a, s) in zip(arms, names) if s == 0})
    run = np.zeros(len(arms))
    t0, it, curve = time.time(), 0, []
    peak = 0
    while it < args.steps:
        for items in ld:
            if args.regime == "e3":
                anchors = np.stack([x["gt_prev"] for x in items])
            else:
                anchors = np.stack([x["gt_prev"] + draw(rng)[0] for x in items])
            inp = batch_inputs(frozen, items, anchors, dev, PERTURB_SEED + it, need)
            for k, (m, o, sc) in enumerate(zip(arms, opts, scheds)):
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):     # CPU smoke: fp32
                    delta = m(inp)
                loss, _ = frozen._compute_loss(inp["batch"].prev_state + delta.float(), inp["batch"].target)
                loss = loss.float().log10()
                o.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), CLIP)
                o.step()
                sc.step()
                run[k] += float(loss)
            if dev.type == "cuda":
                peak = max(peak, torch.cuda.max_memory_allocated() / 2 ** 30)
            it += 1
            if it % 100 == 0 or it == args.steps:
                n = 100 if it % 100 == 0 else it % 100
                ent = {f"{a}/{s}": round(run[k] / n, 4) for k, (a, s) in enumerate(names)}
                curve.append({"it": it, **ent})
                print(it, json.dumps(ent), f"{time.time() - t0:.0f} s", f"peak {peak:.1f} GiB", flush=True)
                run[:] = 0
            if it >= args.steps:
                break
    log["loss_curve"] = curve
    log["train_wall_s"] = time.time() - t0
    log["peak_alloc_gib"] = peak
    torch.save({"arms": [m.state_dict() for m in arms], "names": names}, args.out / "arms.pt")
    return arms, names


@torch.no_grad()
def evaluate(frozen, arms, names, ds, afn, need, dev, mano, workers):
    from tools.xcl.evaluate import score
    preds = [[] for _ in arms]
    base, tgt, meta, anc = [], [], [], []
    for items in C.loader(ds, 64, False, 0, workers=workers):
        anchors = np.stack([afn(x) for x in items])
        inp = batch_inputs(frozen, items, anchors, dev, PERTURB_SEED + 10 ** 6 + items[0]["i"], need)
        for k, m in enumerate(arms):
            preds[k].append((inp["batch"].prev_state + m(inp)).float().cpu().numpy())
        base.append(inp["own"].cpu().numpy())
        tgt.append(np.stack([x["gt_end"] for x in items]))
        anc.append(anchors.astype(np.float32))
        meta += [(x["seq"], x["si"], x["k"], x["rid"], x["blk"]) for x in items]
    out = BT.finish(preds, base, tgt, meta, ds, mano, dev, score)
    out["anchor"] = np.concatenate(anc)
    return out


def anchor_fn(regime, noise):
    if regime == "e3":
        return lambda it: it["gt_prev"]
    return lambda it: it["gt_prev"] + noise[it["i"]][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", choices=("A0", "A3"), required=True)
    ap.add_argument("--regime", choices=("e3", "noise"), default="e3")
    ap.add_argument("--group", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--smoke", action="store_true", help="tiny run (3 steps, 64 endpoints), not a result")
    ap.add_argument("--cpu", action="store_true", help="engineering smoke on the CPU (with --smoke only)")
    args = ap.parse_args()
    if args.cpu and not args.smoke:
        raise SystemExit("--cpu is for the engineering smoke only")
    dev = torch.device("cpu") if args.cpu else MC.require_gpu()
    spec = GROUPS[(args.cond, args.regime, args.group)]
    args.steps, args.batch = (3, 8 if args.cpu else BATCH) if args.smoke else (STEPS, BATCH)
    if not args.smoke:
        MC.verify_protocol("B")
    C.set_policy()
    args.out = Path(args.out)
    if (args.out / "result.json").exists():
        raise SystemExit(f"{args.out} holds a result; preserve it")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(3407)
    frozen, cfg = C.load_frozen(args.cond, dev)
    mano = C.mano32(cfg, dev)
    need = needs(spec)
    log = {"cond": args.cond, "regime": args.regime, "group": args.group, "spec": spec,
           "budget": {"steps": args.steps, "batch": args.batch, "lr": LR, "warmup": WARMUP, "clip": CLIP},
           "instrument": "AGG778 R2: state-blind tail (prev48 = 0), residual slot 0, bf16-mixed train, fp32 eval"}
    arms, names = train(args, frozen, cfg, dev, spec, need, log)
    for m in arms:
        m.eval()
    draw = C.noise_sampler(cfg)
    summary = {}
    for name, ds in (("sel", C.Endpoints(C.subject_seqs(C.SELECT_SUBJ), stride=4)), ("zgz", C.Endpoints(C.zgz_seqs()))):
        if args.smoke:
            ds.index = ds.index[:16 if args.cpu else 64]
        rng = np.random.default_rng(NOISE_SEED_EVAL)
        noise = {i: draw(rng) for i in range(len(ds))}
        res = evaluate(frozen, arms, names, ds, anchor_fn(args.regime, noise), need, dev, mano,
                       0 if args.cpu else args.workers)
        if args.regime == "noise":
            res["noise_kind"] = np.array([noise[i][1] for i in range(len(ds))])
        np.savez_compressed(args.out / f"eval_{name}.npz", **res)
        for dom in ("local", "global"):
            sel_ = np.array([("_" + dom) in s for s in res["seq"]])
            if sel_.any():
                summary[f"{name}/{dom}"] = {f"{a}/{s}": float(res["mpjpe_ra_mm"][j, sel_].mean())
                                            for j, (a, s) in enumerate(names)}
                summary[f"{name}/{dom}"]["frozen_own"] = float(res["mpjpe_ra_mm"][-1, sel_].mean())
    log["summary_mpjpe_ra"] = summary
    log["names"] = [f"{a}/{s}" for a, s in names]
    C.save_json(args.out / "result.json", log, exclusive=False)
    MC.receipt(args.out, {"stage": "b_task", "cond": args.cond, "regime": args.regime, "group": args.group,
                          "smoke": bool(args.smoke)},
               scripts=(__file__, Path(__file__).with_name("heads.py"), C.REPO / "semkine" / "mrf.py"))
    print("COMPLETE", args.out, flush=True)


if __name__ == "__main__":
    main()
