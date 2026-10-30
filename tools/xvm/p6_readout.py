#!/usr/bin/env python3
"""XVM P6: exact kinematic transport (b) vs a learned LBS-slot readout (a) on the same frozen P4 measurements
(docs/EXPERIMENT_RECORDS.md#xvm-prereg-20261021 §3).

    train  (a): per-vertex [m_t / rho, rho^2 Lambda_t (3), observed, |n.d|] -> MLP -> LBS-weighted mean into 16 joint
           slots -> MLP(16*64 + prev48 -> 256 -> 51, last layer zero) -> Delta; x = x0 + Delta. Training subjects only,
           anchors = smoothed GT(end - 50) + small noise, target = smoothed GT(end), loss = mean 21-joint distance (mm).
    eval   (a) or (b) at E3 / E1 on zgz, and the recovery test: 20 evenly spaced start endpoints per zgz sequence,
           GT state perturbed by 50 mm translation (random direction), 0.1 rad root (random axis) and N(0, 0.1 rad)
           fingers (rng 0), 20 closed-loop steps; error per step.

    python tools/xvm/p6_readout.py train --run outputs/xvm/s2/p4/A_m600 --head patch_r2 --out outputs/xvm/s3/p6/readout_a
    python tools/xvm/p6_readout.py eval --run ... --head patch_r2 --path a --readout outputs/xvm/s3/p6/readout_a --out ...
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
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xvm import common as C  # noqa: E402
from xvm.common import EF  # noqa: E402
from xvm.p4_data import P4Data, collate  # noqa: E402
from xvm import p4_train as T  # noqa: E402
from xvm.p5_solve import load_head, solve, E1_TRACE  # noqa: E402
from semkine import xvm as XV  # noqa: E402
from pose_repr import decode_to_mano_inputs  # noqa: E402

REC_STARTS, REC_STEPS = 20, 20


class Readout(nn.Module):
    def __init__(self, lbs_w):
        super().__init__()
        self.register_buffer("W", lbs_w.float())
        self.vert = nn.Sequential(nn.Linear(8, 64), nn.ReLU(inplace=True), nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.Linear(16 * 64 + 48, 256), nn.ReLU(inplace=True), nn.Linear(256, 51))
        nn.init.zeros_(self.head[2].weight)
        nn.init.zeros_(self.head[2].bias)

    def forward(self, g, mt, Lt, rho):
        r = rho.unsqueeze(-1)
        f = torch.cat([mt / r, (Lt[..., 0, 0] * rho ** 2).unsqueeze(-1), (Lt[..., 0, 1] * rho ** 2).unsqueeze(-1),
                       (Lt[..., 1, 1] * rho ** 2).unsqueeze(-1), g["obs2"].float().unsqueeze(-1),
                       g["cos"].float().unsqueeze(-1), g["vis"].float().unsqueeze(-1)], -1)
        f = torch.nan_to_num(f.clamp(-1e3, 1e3))
        h = self.vert(f) * g["vis"].float().unsqueeze(-1)
        a = self.W.unsqueeze(0) * g["vis"].float().unsqueeze(-1)                     # (B, V, 16)
        slots = torch.einsum("bvj,bvc->bjc", a, h) / a.sum(1).clamp(min=1e-6).unsqueeze(-1)
        x0 = g["x0"].float()
        return self.head(torch.cat([slots.flatten(1), x0[:, 3:51]], -1))


def joints(mano32, x, betas):
    dec = decode_to_mano_inputs(x.float(), "mano_full_axis_angle", mano32.hands_components, mano32.hands_mean)
    _, j = mano32(betas.float(), dec["global_orient"], dec["local_full_aa"], dec["transl"])
    return j


def measure(head, g, batch):
    m, L, rho, _ = head(g) if isinstance(head, T.KernelHead) else head(g, batch)
    mt, Lt = XV.to_image(m.float(), L.float(), g["R"].float(), rho)
    return mt, Lt, rho


def train(a, dev):
    head, mem = load_head(Path(a.run), a.head, dev)
    for p in head.parameters():
        p.requires_grad_(False)
    geo = T.Geo(dev)
    m32 = C.mano32(dev)
    ro = Readout(geo.mano.m.weights).to(dev)
    ds = P4Data(C.subject_seqs(C.TRAIN_SUBJ), mem, with_sae=a.head.startswith("patch"))
    gen = torch.Generator().manual_seed(3407)
    torch.manual_seed(3407)
    loader = DataLoader(ds, batch_size=16, shuffle=True, generator=gen, num_workers=12, collate_fn=collate,
                        drop_last=True, persistent_workers=True, prefetch_factor=2)
    opt = torch.optim.Adam(ro.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: 0.5 * (1 + math.cos(math.pi * min(it, a.iters) / a.iters)))
    rng = np.random.default_rng(3407)
    it, run, t0, log = 0, 0.0, time.time(), []
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    while it < a.iters:
        for batch in loader:
            g = geo(batch, "B", rng)
            with torch.no_grad():
                mt, Lt, rho = measure(head, g, batch)
            delta = ro(g, mt, Lt, rho)
            x = g["x0"].float() + delta
            betas = batch["betas"].to(dev)
            jx = joints(m32, x, betas)
            jt = joints(m32, batch["xt"].to(dev), betas)
            loss = (jx - jt).norm(dim=-1).mean() * 1000
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += float(loss)
            it += 1
            if it % 200 == 0:
                log.append({"it": it, "loss_mm": run / 200})
                print(it, round(run / 200, 3), f"{time.time() - t0:.0f} s", flush=True)
                run = 0.0
            if it >= a.iters:
                break
    torch.save({"readout": ro.state_dict(), "run": a.run, "head": a.head}, out / "readout.pt")
    (out / "train_log.json").write_text(json.dumps(log, indent=1))
    C.receipt(out, {"probe": "p6_train"}, scripts=(__file__,))


@torch.no_grad()
def step(path, head, ro, geo, g, batch, betas, intr, P):
    mt, Lt, rho = measure(head, g, batch)
    if path == "a":
        return (g["x0"].float() + ro(g, mt, Lt, rho)).double()
    return torch.stack([solve(geo.mano, P, g, b, mt[b], Lt[b], g["vis"][b], betas, intr) for b in range(len(mt))])


@torch.no_grad()
def evaluate(a, dev):
    head, mem = load_head(Path(a.run), a.head, dev)
    geo = T.Geo(dev)
    P = C.increment_precision(dev)
    m32 = C.mano32(dev)
    ro = None
    if a.path == "a":
        ro = Readout(geo.mano.m.weights).to(dev).eval()
        ro.load_state_dict(torch.load(Path(a.readout) / "readout.pt", map_location=dev)["readout"])
    rng = np.random.default_rng(0)
    sae = a.head.startswith("patch")
    res = {"run": a.run, "head": a.head, "path": a.path, "mode": a.mode, "per_seq": {}}
    t0 = time.time()
    trace = np.load(E1_TRACE)
    for s, d in C.zgz_seqs():
        ds = P4Data([(s, d)], mem, e3=True, with_sae=sae)
        sq = ds.seqs[0]
        intr = sq.intrinsics()
        betas = torch.tensor(sq.betas, dtype=torch.float64, device=dev)[None]
        rows = C.endpoint_grid(sq.runs)
        if a.mode in ("e3", "e1"):
            X, X0, GE = [], [], []
            if a.mode == "e3":
                for batch in DataLoader(ds, batch_size=16, shuffle=False, num_workers=12, collate_fn=collate):
                    g = geo(batch, "A", rng)
                    X.append(step(a.path, head, ro, geo, g, batch, betas, intr, P))
                    X0.append(g["x0"]); GE.append(batch["gt"].to(dev))
                X, X0, GE = torch.cat(X), torch.cat(X0), torch.cat(GE)
            else:
                inc = trace[f"{s}/incoming"].astype(np.float64)
                x = None
                for i in range(len(ds)):
                    item = collate([ds[i]])
                    if rows[i][3]:
                        x = torch.as_tensor(inc[i], dtype=torch.float64, device=dev)
                    item["x0"] = x[None].cpu()
                    g = geo(item, "A", rng)
                    X0.append(x.clone())
                    x = step(a.path, head, ro, geo, g, item, betas, intr, P)[0]
                    X.append(x.clone()); GE.append(item["gt"][0].to(dev))
                X, X0, GE = torch.stack(X), torch.stack(X0), torch.stack(GE)
            sc = C.score_ra(m32, X.cpu().numpy(), GE.cpu().numpy(), sq.betas, dev)
            ea, ez, _, _ = C.r_art_terms(geo.mano, X0, X, GE, betas)
            res["per_seq"][s] = {"n": int(len(X)), "mpjpe_ra_mm": float(sc["mpjpe_ra_mm"].mean()),
                                 "wrist_mm": float(sc["wrist_mm"].mean()), "R_art": float(ea.sum() / ez.sum())}
        else:  # recovery
            prng = np.random.default_rng(0)
            cand = [i for i, r in enumerate(rows) if not r[3] and i + REC_STEPS < len(rows)
                    and all(rows[i + k][0] == rows[i][0] for k in range(REC_STEPS + 1))]
            starts = [cand[int(k)] for k in np.linspace(0, len(cand) - 1, REC_STARTS)]
            err = np.zeros((len(starts), REC_STEPS + 1))
            wr = np.zeros((len(starts), REC_STEPS + 1))
            for si, i0 in enumerate(starts):
                x = torch.as_tensor(sq.pos51[rows[i0][2]], dtype=torch.float64, device=dev).clone()
                u = prng.standard_normal(3); u /= np.linalg.norm(u)
                ax = prng.standard_normal(3); ax /= np.linalg.norm(ax)
                x[:3] += torch.as_tensor(0.05 * u, device=dev)
                x[3:6] += torch.as_tensor(0.1 * ax, device=dev)
                x[6:] += torch.as_tensor(prng.standard_normal(45) * 0.1, device=dev)
                gt_prev = sq.pos51[rows[i0][2]]
                sc0 = C.score_ra(m32, x[None].cpu().numpy(), np.asarray(gt_prev)[None], sq.betas, dev)
                err[si, 0], wr[si, 0] = sc0["mpjpe_ra_mm"][0], sc0["wrist_mm"][0]
                for k in range(REC_STEPS):
                    item = collate([ds[i0 + k]])
                    item["x0"] = x[None].cpu()
                    g = geo(item, "A", rng)
                    x = step(a.path, head, ro, geo, g, item, betas, intr, P)[0]
                    sc = C.score_ra(m32, x[None].cpu().numpy(), item["gt"].numpy(), sq.betas, dev)
                    err[si, k + 1], wr[si, k + 1] = sc["mpjpe_ra_mm"][0], sc["wrist_mm"][0]
            res["per_seq"][s] = {"mpjpe_ra_by_step": err.mean(0).tolist(), "wrist_by_step": wr.mean(0).tolist(),
                                 "n_starts": len(starts)}
        print(s, json.dumps(res["per_seq"][s])[:300], f"{time.time() - t0:.0f} s", flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    C.receipt(out.parent, {"probe": "p6_eval"}, scripts=(__file__,))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("train", "eval"))
    ap.add_argument("--run", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--path", choices=("a", "b"), default="b")
    ap.add_argument("--readout", default="")
    ap.add_argument("--mode", choices=("e3", "e1", "recovery"), default="e3")
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = C.require_gpu()
    if Path(a.out).exists() and a.cmd == "eval":
        raise SystemExit(f"{a.out} exists; preserve it")
    torch.backends.cuda.matmul.allow_tf32 = False
    if a.cmd == "train":
        train(a, dev)
    else:
        evaluate(a, dev)


if __name__ == "__main__":
    main()
