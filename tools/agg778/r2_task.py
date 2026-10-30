#!/usr/bin/env python3
"""AGG778 R2 task-level probes (docs/EXPERIMENT_RECORDS.md#agg778-r2-prereg-20261024).

Corrected instrument (against R1): every arm feeds a fresh C1 downstream (`heads.Tail`) that does NOT read prev48 (the
state shortcut is zero; the anchor mesh's edge geometry stays), with the residual slot zero except the bridge arms
(M13R / MabsR: a residual slot rebuilt by the arm's own phi), trained and evaluated in one anchor regime:

    e3     anchor = GT at the previous endpoint (clean), train and evaluate
    noise  anchor = that GT + C1's mixed prev noise (train rng 3407, evaluation rng 777), train and evaluate

One process = one (condition, regime, group) of the pre-registered arm table `GROUPS`; `--pilot` runs T0s and T6 (one
seed, 2000 steps, selection subjects only) to check that the ceiling can beat identity before the freeze.

    CUDA_VISIBLE_DEVICES=4 CUDA_DEVICE_ORDER=PCI_BUS_ID python tools/agg778/r2_task.py --cond A3 --regime e3 --group g1 \
        --out outputs/agg778/r2/probe/A3_e3_g1
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
from tools.agg778.heads import KernelOp, RecordOp, Tail, c1_records, fanout_records, pool_records  # noqa: E402

R2_PROTOCOL = C.OUT / "r2" / "protocol.json"
#: record columns: tok7 [0 x/W, 1 y/H, 2 polarity, 3 t/dt, 4 log1p(dt_ie), 5 SAE same, 6 SAE opposite] | dXYZ 7-9 |
#: 10 zn | 11 valid | 12 hit
ABS = (0, 1, 10)
MASKS = {"Mabs": ABS, "MabsR": ABS, "Mabsfan": ABS, "Mcore": (0, 1, 2, 3, 4, 5, 6, 10, 12), "Mpol": (2,),
         "Mrate": (4, 5, 6), "Mmass": (12,)}
RECORD_SOURCE = {"DN": "rec_donor", "M13fan": "fan8", "M13fan4": "fan4", "Mabsfan": "fan8"}
FRESH = ("DN", "M13", "M13i", "Mabs", "Mcore", "Mpol", "Mrate", "Mmass", "M13R", "MabsR", "M13fan", "M13fan4",
         "Mabsfan", "K")
GROUPS = {
    ("A3", "e3", "g1"): {"T0s": 3, "T6": 3, "T6s12": 3, "T2": 3, "T2abs": 3, "T4": 3, "T5": 3},
    ("A3", "e3", "g2"): {"DN": 3, "M13": 3, "M13i": 3, "Mabs": 3, "Mcore": 3},
    ("A3", "e3", "g3"): {"Mpol": 1, "Mrate": 1, "Mmass": 1, "M13R": 3, "MabsR": 3},
    ("A3", "e3", "gb"): {"M13": 1},
    ("A3", "noise", "g1"): {"T0s": 3, "T2": 3, "T4": 3, "T5": 3, "M13": 3, "Mabs": 3},
    ("A0", "e3", "g1"): {"T0s": 3, "T6": 3, "T6s12": 3, "T6fan": 3, "T6K": 3, "T2": 3, "T2abs": 3},
    ("A0", "e3", "g2"): {"DN": 3, "M13": 3, "M13i": 3, "Mabs": 3},
    ("A0", "e3", "g3"): {"M13fan": 3, "M13fan4": 3, "Mabsfan": 3, "K": 3},
}
STEPS = {"gb": 8000}
DEFAULT_STEPS, PILOT_STEPS, BATCH, LR, WARMUP, LR_FLOOR, CLIP = 4000, 2000, 128, 2e-3, 300, 0.02, 10.0
NOISE_SEED_TRAIN, NOISE_SEED_EVAL, PERTURB_SEED = 3407, 777, 4242


def lr_lambda(step, steps):
    if step < WARMUP:
        return float(step + 1) / WARMUP
    p = min(max((step - WARMUP) / float(steps - WARMUP), 0.0), 1.0)
    return LR_FLOOR + (1.0 - LR_FLOOR) * 0.5 * (1.0 + np.cos(np.pi * p))


def needs(arms):
    n = {"rec"}
    for a in arms:
        if a == "T6s12":
            n.add("R0s12")
        if a in ("T4", "T5"):
            n.add("R1R2")
        if a == "T2abs":
            n.add("T2abs")
        if a == "DN":
            n.add("donor")
        if a in ("T6fan", "M13fan", "Mabsfan"):
            n |= {"geo", "fan8"}
        if a == "M13fan4":
            n |= {"geo", "fan4"}
        if a in ("T6K", "K"):
            n |= {"geo", "kev"}
    return n


def fan_support(rec, B):
    key = rec["rb"] * C.V + rec["vrec"]
    mass = torch.zeros(B * C.V, device=key.device).index_add(0, key, rec["wrec"].float())
    return (mass > 0).float().view(B, C.V)


@torch.no_grad()
def batch_inputs(frozen, cond, items, anchors, dev, seed, need):
    from xvm.omega_r import record_offsets
    batch = C.make_batch(items, anchors, dev)
    cap = C.forward_capture(frozen, batch)
    B = cap["O"].shape[0]
    vgt = C.fk_verts(frozen, batch.target, batch)
    inp = {"batch": batch, "cap": cap, "B": B, "R0": (vgt - cap["verts"]) / 0.01,
           "edge": frozen.xyz.edge_geometry(cap["verts"])}
    if "rec" in need:
        inp["rec"] = c1_records(cap)
    if "R0s12" in need:
        g = torch.Generator(device=dev).manual_seed(int(seed))
        inp["R0s12"] = inp["R0"] + torch.randn(inp["R0"].shape, generator=g, device=dev) * (12e-3 / 0.01)
    if "R1R2" in need and cond == "A3":
        r1, _ = record_offsets(cap["d"], cap["mask"], cap["verts"], cap["assoc"], "R1")
        r2, _ = record_offsets(cap["d"], cap["mask"], cap["verts"], cap["assoc"], "R2")
        inp["R1"], inp["R2"] = r1 / 0.01, r2 / 0.01
    if "T2abs" in need:
        rec = inp["rec"]
        f = rec["feat"].clone()
        f[:, list(ABS)] = 0.0
        x, O = pool_records(frozen.xyz.rel(f), rec["rb"], rec["vrec"], rec["wrec"], B, "c1")
        inp["T2abs"] = (frozen.xyz.vin(x).float(), O)
    if "donor" in need:
        dcap = C.forward_capture(frozen, C.make_batch(items, anchors, dev, events=[x["donor"]["ev5"] for x in items],
                                                      srcs=[x["donor"]["src"] for x in items]))
        inp["rec_donor"] = c1_records(dcap)
    if "geo" in need:
        inp["geo"] = C.vertex_geometry(frozen, batch, cap["verts"])
    if "fan8" in need:
        inp["fan8"] = fanout_records(cap, inp["geo"], m=8)
        inp["Ofan"] = fan_support(inp["fan8"], B)
    if "fan4" in need:
        inp["fan4"] = fanout_records(cap, inp["geo"], m=4)
    if "kev" in need:
        from semkine import xvm as XV
        from tools.agg778.b_vertex import kernel_events
        inp["kev"] = kernel_events(batch, cap, inp["geo"], "2048")
        geo = inp["geo"]
        rho = (geo["f"][:, None] * 0.01 / geo["z"].clamp(min=1e-3)).float()
        _, ok, _ = XV.gather(geo["uv"], geo["vis"], rho, inp["kev"]["xy"], inp["kev"]["mask"], 64)
        inp["OK"] = ok.any(-1).float()
    # the frozen module's own single-step output from the same anchors (validity V1 under E3)
    inp["own"] = cap["out"]
    return inp


def masked(rec, cols):
    if not cols:
        return rec
    r = dict(rec)
    r["feat"] = rec["feat"].clone()
    r["feat"][:, list(cols)] = 0.0
    return r


def residual_slot(phi_net, cap, cols):
    """C1's residual summary (r-weighted mean, max, share, has) rebuilt with the arm's own phi; token columns `cols`
    zeroed; geometry, valid and hit are zero for residual records as in `XYZMeshNet.forward`."""
    r, tok, mask = cap["r"], cap["tok"], cap["mask"]
    B = r.shape[0]
    eb, en = (r > 0).nonzero(as_tuple=True)
    tr = tok[eb, en].float().clone()
    if cols:
        tr[:, [c for c in cols if c < 7]] = 0.0
    feat = torch.cat([tr, tr.new_zeros(tr.shape[0], 6)], -1)
    phi = phi_net(feat).float()
    Cc = phi.shape[-1]
    rw = r[eb, en].float()
    rs = torch.zeros(B, device=r.device).index_add(0, eb, rw)
    rnum = torch.zeros(B, Cc, device=r.device).index_add(0, eb, rw.unsqueeze(-1) * phi)
    rhas = (rs > 0).unsqueeze(-1)
    rmean = rnum / rs.clamp_min(1e-12).unsqueeze(-1) * rhas
    rpeak = torch.full((B, Cc), -1e4, device=r.device).scatter_reduce(
        0, eb.unsqueeze(-1).expand(-1, Cc), phi, reduce="amax") * rhas
    live = mask.float().sum(1).clamp_min(1.0)
    return torch.cat([rmean, rpeak, (rs / live).unsqueeze(-1), rhas.float()], -1)


class Arm(nn.Module):
    def __init__(self, name, frozen):
        super().__init__()
        self.name = name
        self.cols = MASKS.get(name, ())
        self.op = None
        if name == "K":
            self.op = KernelOp(0.01)
            f_in = self.op.out_dim
        elif name in FRESH:
            self.op = RecordOp(extra=False, pool="c1")
            f_in = self.op.out_dim
        else:
            f_in = {"T0s": 1, "T2": 128, "T2abs": 128}.get(name, 3)
        self.tail = Tail(f_in, frozen.xyz)
        if name == "M13i":
            # the frozen module's own phi and vin as the starting point (the tail's input projection plays vin)
            self.op.rel.load_state_dict(frozen.xyz.rel.state_dict())
            with torch.no_grad():
                self.tail.inp.weight.copy_(frozen.xyz.vin.weight)

    def features(self, inp):
        cap, B = inp["cap"], inp["B"]
        O = cap["O"]
        zeros_b = O.new_zeros(B, 2 * 128 + 2)
        a = self.name
        if a == "T0s":
            return O.new_zeros(B, C.V, 1), torch.zeros_like(O), zeros_b
        if a in ("T6", "T6s12"):
            return inp["R0" if a == "T6" else "R0s12"], O, zeros_b
        if a == "T6fan":
            return inp["R0"], inp["Ofan"], zeros_b
        if a == "T6K":
            return inp["R0"], inp["OK"], zeros_b
        if a == "T2":
            return cap["H0"], O, zeros_b
        if a == "T2abs":
            h, Oa = inp["T2abs"]
            return h, Oa, zeros_b
        if a in ("T4", "T5"):
            return inp["R1" if a == "T4" else "R2"], O, zeros_b
        if a == "K":
            f, Ok = self.op(inp["geo"], inp["kev"])
            return f, Ok, zeros_b
        rec = masked(inp[RECORD_SOURCE.get(a, "rec")], self.cols)
        x, Ox = self.op(rec, B)
        if a == "Mmass":
            x = torch.cat([x[..., :-1], torch.zeros_like(x[..., -1:])], -1)
        bslot = residual_slot(self.op.rel, cap, self.cols) if a in ("M13R", "MabsR") else zeros_b
        return x, Ox, bslot

    def forward(self, inp):
        f, O, bslot = self.features(inp)
        f = f * (O > 0).unsqueeze(-1).to(f.dtype)
        delta = self.tail(f, O, inp["edge"], bslot, f.new_zeros(inp["B"], 48))        # state-blind: no prev48
        return torch.where((inp["cap"]["counts"] <= 0).unsqueeze(-1), torch.zeros_like(delta), delta)


def build(spec, frozen, dev, steps):
    arms, names, opts, scheds = [], [], [], []
    order = list(spec)
    for ai, a in enumerate(order):
        for s in range(spec[a]):
            with torch.random.fork_rng(devices=[dev]):
                torch.manual_seed(20000 + 1000 * (["T0s", "T6", "T6s12", "T6fan", "T6K", "T2", "T2abs", "T4", "T5",
                                                    *FRESH].index(a)) + s)
                m = Arm(a, frozen).to(dev)
            o = torch.optim.Adam(m.parameters(), lr=LR)
            arms.append(m)
            names.append((a, s))
            opts.append(o)
            scheds.append(torch.optim.lr_scheduler.LambdaLR(o, lambda it, n=steps: lr_lambda(it, n)))
    return arms, names, opts, scheds


def anchor_fn(regime, noise):
    if regime == "e3":
        return lambda it: it["gt_prev"]
    return lambda it: it["gt_prev"] + noise[it["i"]][0]


def train(args, frozen, cfg, dev, spec, need, log):
    fit = C.Endpoints(C.subject_seqs(C.FIT_SUBJ), with_donor="donor" in need)
    ld = C.loader(fit, BATCH, True, 3407, workers=args.workers, drop_last=True)
    draw = C.noise_sampler(cfg)
    rng = np.random.default_rng(NOISE_SEED_TRAIN)
    arms, names, opts, scheds = build(spec, frozen, dev, args.steps)
    log.update(n_fit=len(fit), arms=[f"{a}/{s}" for a, s in names],
               params={f"{a}/{s}": sum(p.numel() for p in m.parameters()) for m, (a, s) in zip(arms, names) if s == 0})
    run = np.zeros(len(arms))
    t0, it, curve = time.time(), 0, []
    while it < args.steps:
        for items in ld:
            if args.regime == "e3":
                anchors = np.stack([x["gt_prev"] for x in items])
            else:
                anchors = np.stack([x["gt_prev"] + draw(rng)[0] for x in items])
            inp = batch_inputs(frozen, args.cond, items, anchors, dev, PERTURB_SEED + it, need)
            for k, (m, o, sc) in enumerate(zip(arms, opts, scheds)):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    delta = m(inp)
                loss, _ = frozen._compute_loss(inp["batch"].prev_state + delta.float(), inp["batch"].target)
                loss = loss.float().log10()
                o.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), CLIP)
                o.step()
                sc.step()
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
    torch.save({"arms": [m.state_dict() for m in arms], "names": names}, args.out / "arms.pt")
    return arms, names


@torch.no_grad()
def evaluate(frozen, cond, arms, names, ds, afn, need, dev, mano, workers):
    from tools.xcl.evaluate import score
    preds = [[] for _ in arms]
    base, tgt, meta, anc = [], [], [], []
    for items in C.loader(ds, 64, False, 0, workers=workers):
        anchors = np.stack([afn(x) for x in items])
        inp = batch_inputs(frozen, cond, items, anchors, dev, PERTURB_SEED + 10 ** 6 + items[0]["i"], need)
        for k, m in enumerate(arms):
            preds[k].append((inp["batch"].prev_state + m(inp)).cpu().numpy())
        base.append(inp["own"].cpu().numpy())
        tgt.append(np.stack([x["gt_end"] for x in items]))
        anc.append(anchors.astype(np.float32))
        meta += [(x["seq"], x["si"], x["k"], x["rid"], x["blk"]) for x in items]
    out = BT.finish(preds, base, tgt, meta, ds, mano, dev, score)
    out["anchor"] = np.concatenate(anc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", choices=("A0", "A3"), required=True)
    ap.add_argument("--regime", choices=("e3", "noise"), default="e3")
    ap.add_argument("--group", default=None)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--smoke", action="store_true", help="tiny run, not a result")
    args = ap.parse_args()
    dev = C.require_gpu()
    if args.pilot:
        spec, args.steps = {"T0s": 1, "T6": 1}, PILOT_STEPS
        if args.regime != "e3":
            raise SystemExit("the pilot calibrates the e3 ceiling")
    else:
        spec = GROUPS[(args.cond, args.regime, args.group)]
        args.steps = STEPS.get(args.group, DEFAULT_STEPS)
        if not args.smoke:
            p = json.loads(R2_PROTOCOL.read_text())
            bad = [k for k, v in p["files"].items() if C.sha(C.REPO / k) != v]
            if bad:
                raise SystemExit(f"AGG778 R2 frozen sources changed: {bad}")
    if args.smoke:
        args.steps = 3
    C.set_policy()
    args.out = Path(args.out)
    if (args.out / "result.json").exists():
        raise SystemExit(f"{args.out} holds a result; preserve it")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(3407)
    frozen, cfg = C.load_frozen(args.cond, dev)
    mano = C.mano32(cfg, dev)
    need = needs(spec)
    log = {"cond": args.cond, "regime": args.regime, "group": args.group, "pilot": args.pilot, "spec": spec,
           "budget": {"steps": args.steps, "batch": BATCH, "lr": LR, "warmup": WARMUP, "clip": CLIP},
           "instrument": "state-blind tail (prev48 = 0), residual slot 0 except M13R / MabsR, bf16-mixed train, fp32 eval"}
    arms, names = train(args, frozen, cfg, dev, spec, need, log)
    for m in arms:
        m.eval()
    draw = C.noise_sampler(cfg)
    sel = C.Endpoints(C.subject_seqs(C.SELECT_SUBJ), stride=4, with_donor="donor" in need)
    sets = [("sel", sel)]
    if not args.pilot:
        sets.append(("zgz", C.Endpoints(C.zgz_seqs(), with_donor="donor" in need)))
    summary = {}
    for name, ds in sets:
        if args.smoke:
            ds.index = ds.index[:64]
        rng = np.random.default_rng(NOISE_SEED_EVAL)
        noise = {i: draw(rng) for i in range(len(ds))}
        res = evaluate(frozen, args.cond, arms, names, ds, anchor_fn(args.regime, noise), need, dev, mano,
                       args.workers)
        if args.regime == "noise":
            res["noise_kind"] = np.array([noise[i][1] for i in range(len(ds))])
        np.savez_compressed(args.out / f"eval_{name}.npz", **res)
        for dom in ("local", "global"):
            sel_ = np.array([("_" + dom) in s for s in res["seq"]])
            if not sel_.any():
                continue
            summary[f"{name}/{dom}"] = {f"{a}/{s}": float(res["mpjpe_ra_mm"][j, sel_].mean())
                                        for j, (a, s) in enumerate(names)}
            summary[f"{name}/{dom}"]["frozen_own"] = float(res["mpjpe_ra_mm"][-1, sel_].mean())
    log["summary_mpjpe_ra"] = summary
    log["names"] = [f"{a}/{s}" for a, s in names]
    C.save_json(args.out / "result.json", log, exclusive=False)
    C.receipt(args.out, {"stage": "r2_task", "cond": args.cond, "regime": args.regime, "group": args.group,
                         "pilot": args.pilot, "smoke": bool(args.smoke)},
              scripts=(__file__, Path(__file__).with_name("heads.py"), Path(__file__).with_name("b_task.py")))
    print("COMPLETE", args.out, flush=True)


if __name__ == "__main__":
    main()
