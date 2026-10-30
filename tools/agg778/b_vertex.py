#!/usr/bin/env python3
"""AGG778 Stage B-vertex (docs/EXPERIMENT_RECORDS.md#agg778-prereg-20261024): operator-level isolation of the
778-vertex aggregation. One process = one association condition (A0: frozen C1 / C1 routing; A3: frozen A3F0 /
full_oracle routing). Same events as C1 (its 50 ms window, <= 2048 by its own stride), same anchors as B-task
(GT previous endpoint + C1's mixed prev noise), per-vertex targets in the anchor vertex frame (P4 convention:
2-D displacement / rho, rho = f 0.02 / z, Gaussian NLL on visible vertices) plus the 3-D displacement / L_REL.

    V0  counts only (log1p mass, has)                       null
    V1  positions only: (x, y) permuted inside the packet, time / polarity kept per row    decomposition (descriptive)
    V2  donor events (same sequence, half a recording away)  null
    V3  frozen H0                 V4  frozen pre-vin x (257)              V5  C1 form, fresh phi
    V6  V5 + A_k + material-point in-plane offset (representation)        V7  V5 mass-consistent pooling
    V8  V5 on fan-out binning (m = 8 nearest projected vertices in the band)
    V9a XVM kernel r = 1 cm on C1's 2048 events             V9b the same on <= 8192 events (event budget)
    V10a / V10b  fixed R1 / R2 features (A3 only)

Two readouts per arm (per-vertex MLP; per-vertex head after 3 fresh mesh rounds), 3 seeds; nulls, V1 and V9b MLP
only. `--p4repro` instead re-scores the trained P4 kernel head (outputs/xvm/s2/p4/A_m50, xvm_r1) through this harness
under P4's own setting (smoothed GT anchor / target, <= 8192 events, first endpoints of a run excluded).

    CUDA_VISIBLE_DEVICES=6 CUDA_DEVICE_ORDER=PCI_BUS_ID python tools/agg778/b_vertex.py --cond A0 --out outputs/agg778/b_vertex/A0
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
from tools.agg778.heads import KernelOp, RecordOp, VertexHead, c1_records, fanout_records  # noqa: E402

ARMS = {"A0": ("V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9a", "V9b"),
        "A3": ("V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9a", "V9b", "V10a", "V10b")}
MLP_ONLY = ("V0", "V1", "V2", "V9b")
SEEDS = (0, 1, 2)
STEPS, BATCH, LR, WARMUP, LR_FLOOR, CLIP, LAMBDA_3D = 3000, 64, 1e-3, 300, 0.02, 10.0, 1.0
NOISE_SEED_TRAIN, NOISE_SEED_EVAL, PERM_SEED = 3407, 777, 5151
H, W = 180, 240
N_KERNEL = 8192


def lr_lambda(step, steps):
    if step < WARMUP:
        return float(step + 1) / WARMUP
    p = min(max((step - WARMUP) / float(steps - WARMUP), 0.0), 1.0)
    return LR_FLOOR + (1.0 - LR_FLOOR) * 0.5 * (1.0 + math.cos(math.pi * p))


# ------------------------------------------------------------------------------------------------ inputs
@torch.no_grad()
def kernel_events(batch, cap, geo, which: str):
    """Event lists for the kernel: C1's own <= 2048 selection (`src`, `mask`) or <= 8192 by uniform stride."""
    from semkine import xvm as XV
    ev = batch.events
    if which == "2048":
        src, mask = cap["src"], cap["mask"]
        e = ev[src.reshape(-1)].view(*src.shape, ev.shape[1])
    else:
        ptr = batch.ptr
        B = int(ptr.numel() - 1)
        cnt = (ptr[1:] - ptr[:-1])
        n = cnt.clamp(max=N_KERNEL)
        Nm = max(int(n.max()), 1)
        pos = torch.arange(Nm, device=ev.device)
        mask = pos[None] < n[:, None]
        step = cnt.float() / n.clamp(min=1).float()
        src = ptr[:-1, None] + (pos[None].double() * step[:, None].double()).floor().long()
        src = torch.minimum(src, (ptr[1:] - 1).clamp(min=0)[:, None]).clamp(min=0)
        e = ev[src.reshape(-1)].view(B, Nm, ev.shape[1])
    xy = e[..., 1:3].float()
    sil = XV.splat_mask(geo["uv"], geo["z"] > 0, H, W, r=1)
    xi = xy[..., 0].long().clamp(0, W - 1)
    yi = xy[..., 1].long().clamp(0, H - 1)
    in_sil = sil[torch.arange(xy.shape[0], device=xy.device)[:, None], yi, xi].float()
    return {"xy": xy, "age_ms": (50.0 - e[..., 3].float() * 1000.0).clamp(min=0.0), "pol": 2.0 * e[..., 4].float() - 1.0,
            "mask": mask, "in_sil": in_sil}


@torch.no_grad()
def batch_inputs(frozen, cond, items, anchors, dev, it: int, need=("perm", "donor")):
    from xvm.omega_r import record_offsets
    batch = C.make_batch(items, anchors, dev)
    cap = C.forward_capture(frozen, batch)
    geo = C.vertex_geometry(frozen, batch, cap["verts"])
    fx, fy, cx, cy = geo["fx"], geo["fy"], geo["cx"], geo["cy"]
    vgt = C.fk_verts(frozen, batch.target, batch)
    gprev = torch.as_tensor(np.stack([x["gt_prev"] for x in items]), device=dev, dtype=torch.float32)
    vprev = C.fk_verts(frozen, gprev, batch)
    uv_gt = C.project(vgt, fx, fy, cx, cy)
    t2 = uv_gt - geo["uv"]                                                             # px, anchor -> GT end
    t2m = uv_gt - C.project(vprev, fx, fy, cx, cy)                                     # px, motion only
    tf = (geo["R"].transpose(-1, -2) @ t2.unsqueeze(-1)).squeeze(-1) / geo["rho"].unsqueeze(-1)
    inp = {"batch": batch, "cap": cap, "geo": geo, "t2": t2, "t2m": t2m, "tf": tf,
           "t3": (vgt - cap["verts"]) / 0.01, "edge": frozen.xyz.edge_geometry(cap["verts"]),
           "own": C.gt_owned(items, dev), "rec": c1_records(cap), "rec_fan": fanout_records(cap, geo)}
    if "perm" in need:
        evs, srcs = C.permute_positions(items, PERM_SEED + it)
        inp["rec_perm"] = c1_records(C.forward_capture(frozen, C.make_batch(items, anchors, dev, events=evs, srcs=srcs)))
    if "donor" in need:
        dcap = C.forward_capture(frozen, C.make_batch(items, anchors, dev, events=[x["donor"]["ev5"] for x in items],
                                                      srcs=[x["donor"]["src"] for x in items]))
        inp["rec_donor"] = c1_records(dcap)
    if cond == "A3":
        r1, _ = record_offsets(cap["d"], cap["mask"], cap["verts"], cap["assoc"], "R1")
        r2, _ = record_offsets(cap["d"], cap["mask"], cap["verts"], cap["assoc"], "R2")
        inp["R1"], inp["R2"] = r1 / 0.01, r2 / 0.01
    inp["kev2048"] = kernel_events(batch, cap, geo, "2048")
    inp["kev8192"] = kernel_events(batch, cap, geo, "8192")
    return inp


# ------------------------------------------------------------------------------------------------ arms
class Arm(torch.nn.Module):
    def __init__(self, arm: str, readout: str, xyz):
        super().__init__()
        self.arm = arm
        self.op = None
        if arm in ("V1", "V2", "V5", "V8"):
            self.op = RecordOp(extra=False, pool="c1")
        elif arm == "V6":
            self.op = RecordOp(extra=True, pool="c1")
        elif arm == "V7":
            self.op = RecordOp(extra=False, pool="mass")
        elif arm in ("V9a", "V9b"):
            self.op = KernelOp(0.01)
        f_in = {"V0": 2, "V3": 128, "V4": 257, "V10a": 3, "V10b": 3}.get(arm, None)
        if f_in is None:
            f_in = self.op.out_dim
        self.head = VertexHead(f_in, readout, xyz)

    def features(self, inp):
        cap, B = inp["cap"], inp["cap"]["O"].shape[0]
        a = self.arm
        if a == "V0":
            return torch.stack([cap["x257"][..., -1], cap["O"]], -1), cap["O"]
        if a == "V3":
            return cap["H0"], cap["O"]
        if a == "V4":
            return cap["x257"], cap["O"]
        if a in ("V10a", "V10b"):
            return inp["R1" if a == "V10a" else "R2"] * cap["O"].unsqueeze(-1), cap["O"]
        if a in ("V9a", "V9b"):
            return self.op(inp["geo"], inp["kev2048" if a == "V9a" else "kev8192"])
        rec = {"V1": "rec_perm", "V2": "rec_donor", "V8": "rec_fan"}.get(a, "rec")
        return self.op(inp[rec], B)

    def forward(self, inp):
        f, O = self.features(inp)
        return self.head(f * (O > 0).unsqueeze(-1).to(f.dtype), O, inp["geo"]["cos"], inp["edge"])


def build(cond, frozen, dev):
    arms, names = [], []
    for ai, a in enumerate(ARMS[cond]):
        for ro in (("mlp",) if a in MLP_ONLY else ("mlp", "gnn")):
            for s in SEEDS:
                with torch.random.fork_rng(devices=[dev]):
                    torch.manual_seed(5000 + 100 * ai + 10 * (ro == "gnn") + s)
                    arms.append(Arm(a, ro, frozen.xyz).to(dev))
                names.append((a, ro, s))
    return arms, names


def loss_of(out, inp):
    from semkine import xvm as XV
    m, L, d3 = out
    vis = inp["geo"]["vis"].float()
    nll = XV.gaussian_nll(m.float(), L.float(), inp["tf"], vis)
    mse = (((d3.float() - inp["t3"]) ** 2).sum(-1) * vis).sum() / (3.0 * vis.sum().clamp(min=1.0))
    return nll + LAMBDA_3D * mse


# ------------------------------------------------------------------------------------------------ metrics
KEYS = ("own|con|normal", "own|int|normal", "own|all|2d", "own|all|3d", "own|con|normal_motion",
        "has|con|normal", "vis|con|normal", "own|con|normal|k0", "own|con|normal|k1", "own|con|normal|k2")


def sample_stats(x, y, m):
    """Per-sample [n, sx, sy, sxy, sxx, syy] over masked elements; x, y (B, V[, c]) and m (B, V)."""
    if x.dim() == 3:
        m = m.unsqueeze(-1).expand_as(x)
    mf = m.double()
    x, y = x.double() * mf, y.double() * mf
    dims = tuple(range(1, x.dim()))
    return torch.stack([mf.sum(dims), x.sum(dims), y.sum(dims), (x * y).sum(dims), (x * x).sum(dims),
                        (y * y).sum(dims)], -1)


@torch.no_grad()
def accumulate(acc, name, out, inp, blocks, kinds=None):
    from semkine import xvm as XV
    m, L, d3 = (o.float() for o in out)
    geo = inp["geo"]
    rho = geo["rho"]
    e1 = geo["R"][..., :, 0]
    pn = m[..., 0] * rho
    tn = (e1 * inp["t2"]).sum(-1)
    tnm = (e1 * inp["t2m"]).sum(-1)
    mt, _ = XV.to_image(m, L, geo["R"], rho)
    con = geo["cos"] < C.CONTOUR_COS
    own, has, vis = inp["own"] & geo["vis"], inp["cap"]["O"] > 0, geo["vis"]
    rows = {"own|con|normal": sample_stats(pn, tn, own & con), "own|int|normal": sample_stats(pn, tn, own & ~con),
            "own|all|2d": sample_stats(mt, inp["t2"], own), "own|all|3d": sample_stats(d3, inp["t3"], own),
            "own|con|normal_motion": sample_stats(pn, tnm, own & con),
            "has|con|normal": sample_stats(pn, tn, has & con & vis), "vis|con|normal": sample_stats(pn, tn, vis & con)}
    if kinds is not None:
        base = rows["own|con|normal"]
        for kk in range(3):
            rows[f"own|con|normal|k{kk}"] = base * torch.as_tensor(kinds == kk, device=base.device).double().unsqueeze(-1)
    for key, st in rows.items():
        st = st.cpu().numpy()
        d = acc.setdefault(name, {}).setdefault(key, {})
        for b, row in zip(blocks, st):
            d[int(b)] = d.get(int(b), 0.0) + row


def pack(acc):
    """{name: {key: (n_blocks, 6)}} on one common sorted block list."""
    blocks = sorted({b for nm in acc.values() for kv in nm.values() for b in kv})
    out = {"blocks": np.array(blocks, np.int64)}
    for nm, kv in acc.items():
        for key, d in kv.items():
            out[f"{nm}#{key}"] = np.stack([np.asarray(d.get(b, np.zeros(6))) for b in blocks])
    return out


# ------------------------------------------------------------------------------------------------ main
def train(args, cond, frozen, cfg, dev, log):
    fit = C.Endpoints(C.subject_seqs(C.FIT_SUBJ), with_donor=True)
    ld = C.loader(fit, BATCH, True, 3407, workers=args.workers, drop_last=True)
    draw = C.noise_sampler(cfg)
    rng = np.random.default_rng(NOISE_SEED_TRAIN)
    arms, names = build(cond, frozen, dev)
    opts = [torch.optim.Adam(a.parameters(), lr=LR) for a in arms]
    scheds = [torch.optim.lr_scheduler.LambdaLR(o, lambda i, s=args.steps: lr_lambda(i, s)) for o in opts]
    log.update(n_fit=len(fit), heads=["/".join(map(str, n)) for n in names],
               params={"/".join(map(str, n)): sum(p.numel() for p in a.parameters()) for a, n in zip(arms, names)
                       if n[2] == 0})
    run = np.zeros(len(arms))
    t0, it, curve = time.time(), 0, []
    while it < args.steps:
        for items in ld:
            noise = np.stack([draw(rng)[0] for _ in items])
            anchors = np.stack([x["gt_prev"] for x in items]) + noise
            inp = batch_inputs(frozen, cond, items, anchors, dev, it)
            for k, (arm, opt, sch) in enumerate(zip(arms, opts, scheds)):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = arm(inp)
                loss = loss_of(out, inp)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(arm.parameters(), CLIP)
                opt.step()
                sch.step()
                run[k] += float(loss)
            it += 1
            if it % 100 == 0 or it == args.steps:
                n = 100 if it % 100 == 0 else it % 100
                ent = {"/".join(map(str, nm)): round(run[k] / n, 4) for k, nm in enumerate(names)}
                curve.append({"it": it, **ent})
                print(it, json.dumps(ent), f"{time.time() - t0:.0f} s", flush=True)
                run[:] = 0
            if it >= args.steps:
                break
    log["loss_curve"] = curve
    log["train_wall_s"] = time.time() - t0
    torch.save({"arms": [a.state_dict() for a in arms], "names": names}, args.out / "heads.pt")
    return arms, names


@torch.no_grad()
def evaluate(frozen, cond, cfg, arms, names, ds, anchor_fn, dev, workers, kinds_fn=None):
    acc = {}
    for items in C.loader(ds, 32, False, 0, workers=workers):
        anchors = np.stack([anchor_fn(x) for x in items])
        inp = batch_inputs(frozen, cond, items, anchors, dev, 10 ** 6 + items[0]["i"])
        blocks = C.block_ids(items)
        kinds = np.array([kinds_fn(x) for x in items]) if kinds_fn is not None else None
        for arm, nm in zip(arms, names):
            accumulate(acc, "/".join(map(str, nm)), arm(inp), inp, blocks, kinds)
    return pack(acc)


@torch.no_grad()
def p4repro(args, dev):
    """Re-score the trained P4 kernel head (r = 1 cm) through this harness under P4's own setting."""
    from semkine import xvm as XV
    from xvm.common import smoothed_gt
    frozen, cfg = C.load_frozen("A0", dev)
    st = torch.load(C.REPO / "outputs/xvm/s2/p4/A_m50/heads.pt", map_location="cpu")["heads"]
    net = XV.XVMKernel(64).to(dev).eval()
    net.load_state_dict({k[len("xvm_r1.net."):]: v for k, v in st.items() if k.startswith("xvm_r1.net.")})
    res = {}
    for s, d in C.zgz_seqs():
        ds = C.Endpoints([(s, d)])
        ds.index = [x for x in ds.index if not x[5]]                                  # P4 skips first endpoints
        if args.smoke:
            ds.index = ds.index[:16]
        acc = np.zeros(6)
        n_ep = 0
        for items in C.loader(ds, 16, False, 0, workers=args.workers):
            sq = ds.seqs[0]
            anc, tgt = [], []
            for x in items:
                a, b = (int(v) for v in sq.runs[x["rid"]])
                anc.append(smoothed_gt(sq.pos51, x["end"] - 50, a, b - 1).astype(np.float32))
                tgt.append(smoothed_gt(sq.pos51, x["end"], a, b - 1).astype(np.float32))
                x["gt_end"] = tgt[-1]
            batch = C.make_batch(items, np.stack(anc), dev)
            cap = C.forward_capture(frozen, batch)
            geo = C.vertex_geometry(frozen, batch, cap["verts"])
            vgt = C.fk_verts(frozen, batch.target, batch)
            t = C.project(vgt, geo["fx"], geo["fy"], geo["cx"], geo["cy"]) - geo["uv"]
            ev = kernel_events(batch, cap, geo, "8192")
            rho2 = (geo["f"][:, None] * 0.02 / geo["z"].clamp(min=1e-3)).float()
            _, ok2, _ = XV.gather(geo["uv"], geo["vis"], rho2, ev["xy"], ev["mask"] & (ev["age_ms"] < 50.0), 64)
            rho = (geo["f"][:, None] * 0.01 / geo["z"].clamp(min=1e-3)).float()
            idx, ok, _ = XV.gather(geo["uv"], geo["vis"], rho, ev["xy"], ev["mask"], 64)
            feats = XV.pair_features(geo["uv"], geo["R"], rho, geo["cos"], ev, idx, ok, 50.0, ev["in_sil"])
            m, L = net(feats, ok, geo["cos"])
            pn = m[..., 0] * rho
            tn = (geo["R"][..., :, 0] * t).sum(-1)
            mk = geo["vis"] & ok2.any(-1) & (geo["cos"] < C.CONTOUR_COS)
            acc += sample_stats(pn, tn, mk).sum(0).cpu().numpy()
            n_ep += len(items)
        res[s] = {"corr": float(C.corr_from_stats(acc)), "n_vertices": int(acc[0]), "n_endpoints": n_ep}
    p4 = json.loads((C.REPO / "outputs/xvm/s2/p4/A_m50/result.json").read_text())
    for s in C.ZGZ:
        ref = p4["zgz"][s]["xvm_r1"]["obs_contour|normal"]
        res[s].update(p4_corr=ref["corr"], p4_n=ref["n"], diff=res[s]["corr"] - ref["corr"],
                      pass_=abs(res[s]["corr"] - ref["corr"]) <= 0.02)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", choices=("A0", "A3"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--smoke", action="store_true", help="tiny run, not a result")
    ap.add_argument("--p4repro", action="store_true")
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
    if args.p4repro:
        res = p4repro(args, dev)
        C.save_json(args.out / "result.json", {"p4repro": res}, exclusive=False)
        C.receipt(args.out, {"stage": "b_vertex_p4repro"}, scripts=(__file__, Path(__file__).with_name("heads.py")))
        print("P4REPRO", json.dumps(res), flush=True)
        return
    frozen, cfg = C.load_frozen(args.cond, dev)
    log = {"cond": args.cond, "budget": {"steps": args.steps, "batch": BATCH, "lr": LR, "warmup": WARMUP, "clip": CLIP,
                                          "lambda_3d": LAMBDA_3D}, "precision": "bf16-mixed train, fp32 eval"}
    arms, names = train(args, args.cond, frozen, cfg, dev, log)
    for a in arms:
        a.eval()
    draw = C.noise_sampler(cfg)
    trace = C.strict_trace()
    sel = C.Endpoints(C.subject_seqs(C.SELECT_SUBJ), stride=4, with_donor=True)
    zg = C.Endpoints(C.zgz_seqs(), with_donor=True)
    if args.smoke:
        sel.index, zg.index = sel.index[:32], zg.index[:32]
    for name, ds in (("sel", sel), ("zgz", zg)):
        rng = np.random.default_rng(NOISE_SEED_EVAL)
        noise = {i: draw(rng) for i in range(len(ds))}
        packs = {"e3": evaluate(frozen, args.cond, cfg, arms, names, ds, lambda x: x["gt_prev"], dev, args.workers),
                 "noise": evaluate(frozen, args.cond, cfg, arms, names, ds, lambda x: x["gt_prev"] + noise[x["i"]][0],
                                   dev, args.workers, kinds_fn=lambda x: noise[x["i"]][1])}
        if name == "zgz":
            packs["loop"] = evaluate(frozen, args.cond, cfg, arms, names, ds, lambda x: trace[x["seq"]][x["k"]], dev,
                                     args.workers)
        for reg, pk in packs.items():
            np.savez_compressed(args.out / f"stats_{name}_{reg}.npz", **pk)
            log[f"{name}_{reg}"] = {k.split("#")[0] + "#" + k.split("#")[1]: float(C.corr_from_stats(v.sum(0)))
                                    for k, v in pk.items() if k != "blocks" and k.endswith("normal")}
    C.save_json(args.out / "result.json", log, exclusive=False)
    C.receipt(args.out, {"stage": "b_vertex", "cond": args.cond, "smoke": bool(args.smoke)},
              scripts=(__file__, Path(__file__).with_name("heads.py")))
    print("COMPLETE", args.out, flush=True)


if __name__ == "__main__":
    main()
