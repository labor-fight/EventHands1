#!/usr/bin/env python3
"""XVM P4: learnability of the per-vertex event motion measurement (docs/EXPERIMENT_RECORDS.md#xvm-prereg-20261021 §3).

One process = one memory length W_mem; it trains the XVM kernel at the three radii r = 1 / 2 / 4 cm side by side on
the same batches (plus, at W_mem = 50, the C1 phi + mean/max record path, and at W_mem = 600, the vertex-anchored patch
CNN at r = 2 and 4 cm), then evaluates every head on the selection subjects (ycy, ylf; every 4th endpoint) and on
zgz (every endpoint). Training subjects: ch, lfz, lpc, lr, ly, lyh, lyq (56 sequences). Seed 3407, no augmentation.

    python tools/xvm/p4_train.py --mem 200 --arm A --out outputs/xvm/s2/p4/A_m200
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xvm import common as C  # noqa: E402
from xvm.common import EF  # noqa: E402
from xvm.p4_data import P4Data, collate  # noqa: E402
from semkine import xvm as XV  # noqa: E402
from semkine.xyz_mesh import vertex_normals  # noqa: E402

RADII_CM = (1.0, 2.0, 4.0)
K_NEAR = 64
NOISE = (0.005, 0.05, 0.05)                     # arm B anchor noise: t (m), root (rad), pose (rad)
TAUS = (10.0, 50.0, 200.0)
PATCH = 16


# ------------------------------------------------------------------------------------------------ geometry
class Geo:
    def __init__(self, dev):
        self.mano = EF.Mano64(dev)
        self.faces = self.mano.faces
        self.dev = dev

    @torch.no_grad()
    def __call__(self, batch, arm, rng):
        dev = self.dev
        x0 = batch["x0"].to(dev)
        if arm == "B":
            n = torch.zeros_like(x0)
            g = torch.as_tensor(rng.standard_normal((x0.shape[0], 51)), dtype=torch.float64, device=dev)
            n[:, :3] = g[:, :3] * NOISE[0]
            n[:, 3:6] = g[:, 3:6] * NOISE[1]
            n[:, 6:] = g[:, 6:] * NOISE[2]
            x0 = x0 + n
        xt, gt = batch["xt"].to(dev), batch["gt"].to(dev)
        betas = batch["betas"].to(dev)
        K = batch["K"].to(dev)
        fx, fy, cx, cy = K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]
        B = x0.shape[0]
        V0, Vt, Vg = [], [], []
        for b in range(B):
            v, _ = self.mano.fk(torch.stack([x0[b], xt[b], gt[b]]), betas[b:b + 1])
            V0.append(v[0]); Vt.append(v[1]); Vg.append(v[2])
        V0, Vt, Vg = torch.stack(V0), torch.stack(Vt), torch.stack(Vg)

        def proj(v):
            z = v[..., 2].clamp(min=1e-6)
            return torch.stack([fx[:, None] * v[..., 0] / z + cx[:, None], fy[:, None] * v[..., 1] / z + cy[:, None]], -1)
        uv0, uvt, uvg = proj(V0), proj(Vt), proj(Vg)
        z0 = V0[..., 2]
        normals = vertex_normals(V0.float(), self.faces).double()
        Jp = XV.projection_jacobian(V0, fx, fy)
        R, cos = XV.vertex_frames(V0, normals, Jp)
        vis = torch.stack([EF.visible(uv0[b], z0[b]) for b in range(B)])
        sil = XV.splat_mask(uv0, z0 > 0, EF.H, EF.W, r=1)
        xy = batch["xy"].to(dev)
        xi_, yi_ = xy[..., 0].long().clamp(0, EF.W - 1), xy[..., 1].long().clamp(0, EF.H - 1)
        in_sil = sil[torch.arange(B, device=dev)[:, None], yi_, xi_]
        f = torch.sqrt(fx * fy)
        ev_mask = batch["ev_mask"].to(dev)
        age = batch["age_ms"].to(dev)
        # common "observed" category for every head: >= 1 event of the last 50 ms within the 2 cm radius
        rho2 = (f[:, None] * 0.02 / z0.clamp(min=1e-3)).float()
        idx2, ok2, _ = XV.gather(uv0, vis, rho2, xy, ev_mask & (age < 50.0), K_NEAR)
        obs2 = ok2.any(-1)
        return {"obs2": obs2, "x0": x0, "V0": V0, "uv0": uv0, "z0": z0, "t": uvt - uv0, "t_raw": uvg - uv0, "R": R, "cos": cos,
                "vis": vis, "f": f, "fx": fx, "fy": fy, "cx": cx, "cy": cy, "in_sil": in_sil,
                "ev": {"xy": xy, "age_ms": batch["age_ms"].to(dev), "pol": batch["pol"].to(dev)},
                "ev_mask": batch["ev_mask"].to(dev), "Jp": Jp, "normals": normals}


# ------------------------------------------------------------------------------------------------ heads
class KernelHead(nn.Module):
    def __init__(self, r_cm, mem):
        super().__init__()
        self.r, self.mem = r_cm / 100.0, mem
        self.net = XV.XVMKernel(64)

    def rho(self, g):
        return (g["f"][:, None] * self.r / g["z0"].clamp(min=1e-3)).float()

    def forward(self, g):
        rho = self.rho(g)
        idx, ok, _ = XV.gather(g["uv0"], g["vis"], rho, g["ev"]["xy"], g["ev_mask"], K_NEAR)
        feats = XV.pair_features(g["uv0"], g["R"], rho, g["cos"], g["ev"], idx, ok, self.mem, g["in_sil"])
        m, L = self.net(feats, ok, g["cos"])
        nmeas = (ok & (torch.gather(g["ev"]["age_ms"], 1, idx.reshape(idx.shape[0], -1)).view_as(idx) < 50)).sum(-1)
        return m, L, rho, nmeas


class PhiHead(nn.Module):
    """C1's record path: hit_first association on the anchor mesh, phi(token7, dXYZ, z_prior, valid, hit),
    per-vertex weighted mean / unweighted max / log mass, then the same (m, L) head."""

    def __init__(self, geo):
        super().__init__()
        from semkine.xyz_mesh import REL_IN, vertex_face_table
        self.rel = nn.Sequential(nn.Linear(REL_IN, 64), nn.ReLU(inplace=True), nn.Linear(64, 128))
        self.head = nn.Sequential(nn.Linear(257 + 1, 64), nn.ReLU(inplace=True), nn.Linear(64, 5))
        self.register_buffer("vf", vertex_face_table(geo.faces.cpu(), 778), persistent=False)
        self.faces = geo.faces
        self.r = 0.02

    def forward(self, g, batch):
        from semkine.encoder import event_tokens
        from semkine.xyz_mesh import associate_rays, L_REL, Z_REF, Z_SCALE
        dev = g["uv0"].device
        ev5 = batch["ev5"].to(dev)
        m5 = batch["ev5_mask"].to(dev)
        B, N, _ = ev5.shape
        counts = m5.sum(1)
        rows = torch.cat([ev5[b, :int(counts[b])] for b in range(B)])
        rows[:, 0] = torch.repeat_interleave(torch.arange(B, device=dev, dtype=rows.dtype), counts)
        ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), counts.cumsum(0)])
        tok_flat = event_tokens(rows, ptr, torch.full((B,), 0.05, device=dev), EF.H, EF.W)
        tok = torch.zeros(B, N, 7, device=dev)
        tok[m5] = tok_flat
        with torch.no_grad():
            d = torch.stack([(ev5[..., 1] - g["cx"][:, None]) / g["fx"][:, None],
                             (ev5[..., 2] - g["cy"][:, None]) / g["fy"][:, None], torch.ones_like(ev5[..., 1])], -1)
            f = g["f"]
            assoc = associate_rays(d.float(), m5, g["V0"].float(), self.faces, self.vf, band=16.0 / f,
                                   front_tol=3.0 / f, z_tol=0.01, k=16)
        w = assoc["q"].unsqueeze(-1) * assoc["A"] * m5.float().unsqueeze(-1)
        rb, rn, rr = (w > 0).nonzero(as_tuple=True)
        vrec = assoc["vid"][rb, rn, rr]
        wrec = w[rb, rn, rr]
        drec = d[rb, rn].float()
        zrec = assoc["z"][rb, rn]
        X = zrec.unsqueeze(-1) * drec
        dxyz = (X - g["V0"].float()[rb, vrec]) / L_REL
        zn = ((zrec - Z_REF) / Z_SCALE).unsqueeze(-1)
        feat = torch.cat([tok[rb, rn], dxyz, zn, torch.ones_like(zn), assoc["hit"][rb, rn].float().unsqueeze(-1)], -1)
        phi = self.rel(feat)
        key = rb * 778 + vrec
        mass = torch.zeros(B * 778, device=dev).index_add(0, key, wrec)
        num = torch.zeros(B * 778, 128, device=dev).index_add(0, key, wrec.unsqueeze(-1) * phi)
        has = mass > 0
        mean = num / mass.clamp_min(1e-12).unsqueeze(-1) * has.unsqueeze(-1)
        peak = torch.full((B * 778, 128), -1e4, device=dev).scatter_reduce(0, key.unsqueeze(-1).expand(-1, 128), phi,
                                                                            reduce="amax") * has.unsqueeze(-1)
        x = torch.cat([mean, peak, torch.log1p(mass).unsqueeze(-1)], -1).view(B, 778, 257)
        o = self.head(torch.cat([x, g["cos"].float().unsqueeze(-1)], -1))
        m, L = _ml(o)
        rho = (g["f"][:, None] * self.r / g["z0"].clamp(min=1e-3)).float()
        return m, L, rho, has.view(B, 778).long()


class PatchHead(nn.Module):
    """Vertex-anchored patch CNN on polarity time surfaces exp(-age / tau), tau 10 / 50 / 200 ms (reference arm)."""

    def __init__(self, r_cm):
        super().__init__()
        self.r = r_cm / 100.0
        self.cnn = nn.Sequential(nn.Conv2d(6, 32, 3, padding=1), nn.ReLU(inplace=True),
                                 nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
                                 nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(64 + 1, 64), nn.ReLU(inplace=True), nn.Linear(64, 5))

    def forward(self, g, batch):
        dev = g["uv0"].device
        age = batch["sae_age"].to(dev)                                       # (B, 2, H, W)
        ts = torch.cat([torch.exp(-age / t) for t in TAUS], 1)                 # (B, 6, H, W)
        B = age.shape[0]
        rho = (g["f"][:, None] * self.r / g["z0"].clamp(min=1e-3)).float()
        vis = g["vis"]
        o = torch.linspace(-1, 1, PATCH, device=dev)
        gy, gx = torch.meshgrid(o, o, indexing="ij")
        loc = torch.stack([gx, gy], -1).view(1, 1, PATCH * PATCH, 2)            # vertex-frame offsets / rho
        R = g["R"].float()
        off = (R.unsqueeze(2) @ loc.unsqueeze(-1).expand(B, 778, -1, 2, 1)).squeeze(-1) * rho[..., None, None]
        pix = g["uv0"].float().unsqueeze(2) + off                              # (B, V, P, 2) px
        norm = torch.stack([pix[..., 0] / (EF.W - 1) * 2 - 1, pix[..., 1] / (EF.H - 1) * 2 - 1], -1)
        m_all = torch.zeros(B, 778, 2, device=dev)
        L_all = torch.eye(2, device=dev).expand(B, 778, 2, 2).clone() * 1e-3
        for b in range(B):
            ids = vis[b].nonzero().flatten()
            if not len(ids):
                continue
            smp = F.grid_sample(ts[b:b + 1], norm[b:b + 1, ids].view(1, len(ids) * PATCH, PATCH, 2),
                                align_corners=True)                           # (1, 6, n*P, P)
            patches = smp.view(6, len(ids), PATCH, PATCH).permute(1, 0, 2, 3)
            h = self.cnn(patches)
            oo = self.head(torch.cat([h, g["cos"][b, ids].float().unsqueeze(-1)], -1))
            mb, Lb = _ml(oo)
            m_all[b, ids] = mb
            L_all[b, ids] = Lb
        return m_all, L_all, rho, vis.long()


def _ml(o):
    m = o[..., :2]
    L = torch.zeros(*o.shape[:-1], 2, 2, device=o.device, dtype=o.dtype)
    L[..., 0, 0] = F.softplus(o[..., 2]) + 1e-3
    L[..., 1, 1] = F.softplus(o[..., 3]) + 1e-3
    L[..., 1, 0] = o[..., 4]
    return m, L


def target_frame(g, rho, key="t"):
    """Target displacement in the vertex frame, units of rho."""
    return (g["R"].float().transpose(-1, -2) @ g[key].float().unsqueeze(-1)).squeeze(-1) / rho.unsqueeze(-1)


# ------------------------------------------------------------------------------------------------ metrics
CATS = ("visible", "observed", "obs_contour", "obs_interior")


class Corr:
    def __init__(self):
        self.s = {}

    def add(self, key, x, y):
        x, y = x.double().flatten(), y.double().flatten()
        a = self.s.setdefault(key, np.zeros(6))
        a += np.array([len(x), float(x.sum()), float(y.sum()), float((x * y).sum()), float((x * x).sum()),
                       float((y * y).sum())])

    def result(self):
        out = {}
        for k, (n, sx, sy, sxy, sxx, syy) in self.s.items():
            if n < 2:
                continue
            cov = sxy / n - sx * sy / n ** 2
            vx, vy = sxx / n - (sx / n) ** 2, syy / n - (sy / n) ** 2
            corr = cov / math.sqrt(max(vx * vy, 1e-30))
            sse = syy - 2 * sxy + sxx                                           # sum (y - x)^2 with x = prediction
            sst = syy - sy ** 2 / n
            out[k] = {"n": int(n), "corr": float(corr), "R2": float(1 - sse / max(sst, 1e-30))}
        return out


def accumulate(acc, g, m, L, rho):
    """Prediction vs target (px) per category, normal (e1) component and full 2-D (both image axes)."""
    mt, Lt = XV.to_image(m.float(), L.float(), g["R"].float(), rho)
    tnorm = (g["R"].float()[..., :, 0] * g["t"].float()).sum(-1)
    pnorm = m[..., 0].float() * rho
    tnorm_raw = (g["R"].float()[..., :, 0] * g["t_raw"].float()).sum(-1)
    vis = g["vis"]
    obs = vis & g["obs2"]
    con = g["cos"] < XV.CONTOUR_COS
    masks = {"visible": vis, "observed": obs, "obs_contour": obs & con, "obs_interior": obs & ~con}
    for c, mk in masks.items():
        if mk.any():
            acc.add(f"{c}|normal", pnorm[mk], tnorm[mk])
            acc.add(f"{c}|normal_vs_raw", pnorm[mk], tnorm_raw[mk])
            acc.add(f"{c}|2d", mt[mk], g["t"].float()[mk])


# ------------------------------------------------------------------------------------------------ main
def build_heads(mem, geo, which):
    heads = nn.ModuleDict()
    for r in RADII_CM:
        heads[f"xvm_r{int(r)}"] = KernelHead(r, mem)
    if which.get("phi"):
        heads["phi"] = PhiHead(geo)
    if which.get("patch"):
        for r in (2.0, 4.0):
            heads[f"patch_r{int(r)}"] = PatchHead(r)
    return heads


def run_heads(heads, g, batch):
    outs = {}
    for k, h in heads.items():
        outs[k] = h(g) if isinstance(h, KernelHead) else h(g, batch)
    return outs


def evaluate(heads, geo, loader, arm, dev):
    acc = {k: Corr() for k in heads}
    rng = np.random.default_rng(12345)
    heads.eval()
    with torch.no_grad():
        for batch in loader:
            g = geo(batch, arm, rng)
            for k, (m, L, rho, nmeas) in run_heads(heads, g, batch).items():
                accumulate(acc[k], g, m, L, rho)
    heads.train()
    return {k: v.result() for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mem", type=int, required=True, choices=(50, 200, 600))
    ap.add_argument("--arm", choices=("A", "B"), default="A")
    ap.add_argument("--radii", default="")
    ap.add_argument("--iters", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    dev = C.require_gpu()
    out = Path(a.out)
    if (out / "result.json").exists():
        raise SystemExit(f"{out} holds a result; preserve it")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(3407)
    np.random.seed(3407)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    global RADII_CM
    if a.radii:
        RADII_CM = tuple(float(v) for v in a.radii.split(","))
    which = {"phi": a.mem == 50 and a.arm == "A" and not a.radii, "patch": a.mem == 600 and a.arm == "A" and not a.radii}
    train_seqs = C.subject_seqs(C.TRAIN_SUBJ)
    sel_seqs = C.subject_seqs(C.SELECT_SUBJ)
    ds_kw = {"with_sae": which["patch"], "raw_window50": which["phi"]}
    train = P4Data(train_seqs, a.mem, **ds_kw)
    sel = P4Data(sel_seqs, a.mem, stride=4, **ds_kw)
    zg = P4Data(list(C.zgz_seqs()), a.mem, **ds_kw)
    if a.debug:
        a.iters = 40
        sel.index, zg.index = sel.index[:64], zg.index[:64]
    gen = torch.Generator().manual_seed(3407)
    loader = DataLoader(train, batch_size=a.batch, shuffle=True, generator=gen, num_workers=a.workers,
                        collate_fn=collate, drop_last=True, persistent_workers=True, prefetch_factor=2)
    ekw = dict(batch_size=a.batch, shuffle=False, num_workers=a.workers, collate_fn=collate)
    geo = Geo(dev)
    heads = build_heads(a.mem, geo, which).to(dev)
    opt = torch.optim.Adam(heads.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: 0.5 * (1 + math.cos(math.pi * min(it, a.iters) / a.iters)))
    rng = np.random.default_rng(3407)
    log = {"args": vars(a), "n_train": len(train), "n_sel": len(sel), "n_zgz": len(zg),
           "params": {k: sum(p.numel() for p in h.parameters()) for k, h in heads.items()}, "loss": []}
    print(json.dumps({k: log[k] for k in ("n_train", "n_sel", "n_zgz", "params")}), flush=True)
    t0 = time.time()
    it = 0
    run = {k: 0.0 for k in heads}
    while it < a.iters:
        for batch in loader:
            g = geo(batch, a.arm, rng)
            opt.zero_grad(set_to_none=True)
            total = 0.0
            for k, (m, L, rho, nmeas) in run_heads(heads, g, batch).items():
                loss = XV.gaussian_nll(m, L, target_frame(g, rho), g["vis"].float())
                total = total + loss
                run[k] += float(loss)
            total.backward()
            torch.nn.utils.clip_grad_norm_(heads.parameters(), 10.0)
            opt.step()
            sched.step()
            it += 1
            if it % 200 == 0 or it == a.iters:
                ent = {k: v / 200 for k, v in run.items()}
                log["loss"].append({"it": it, **ent})
                print(it, json.dumps({k: round(v, 4) for k, v in ent.items()}), f"{time.time() - t0:.0f} s", flush=True)
                run = {k: 0.0 for k in heads}
            if it >= a.iters:
                break
    torch.save({"heads": heads.state_dict(), "args": vars(a), "radii": RADII_CM}, out / "heads.pt")
    log["train_wall_s"] = time.time() - t0
    sel_res = evaluate(heads, geo, DataLoader(sel, **ekw), a.arm, dev)
    zgz_res = {}
    for s in C.ZGZ:
        sub = P4Data([x for x in C.zgz_seqs() if x[0] == s], a.mem, **ds_kw)
        if a.debug:
            sub.index = sub.index[:32]
        zgz_res[s] = evaluate(heads, geo, DataLoader(sub, **ekw), a.arm, dev)
    log["selection"] = sel_res
    log["zgz"] = zgz_res
    log["wall_s"] = time.time() - t0
    (out / "result.json").write_text(json.dumps(log, indent=1))
    C.receipt(out, {"probe": "p4", "mem": a.mem, "arm": a.arm}, scripts=(__file__, Path(__file__).with_name("p4_data.py"),
                                                                          C.REPO / "semkine" / "xvm.py"))
    for k in heads:
        sc = sel_res[k].get("obs_contour|normal", {})
        zl = zgz_res["zgz_local"][k].get("obs_contour|normal", {})
        print("RESULT", k, "sel", sc, "zgz_local", zl, flush=True)


if __name__ == "__main__":
    main()
