#!/usr/bin/env python3
"""S44 P0.4: per-finger pooled readout on the frozen S43 trunk (docs/EXPERIMENT_RECORDS.md#s44-prereg-20261019).

1. Training packets (the 72 training sequences, 12000 random protocol-like steps, rng 0): W200 LNES clipped to the
   segment start, prev = GT(end - 50) + the training "mixed" noise (50% small / 30% large / 20% directed); the frozen
   S43 trunk gives the level-4 map y4 (256 x 12 x 15) and its active mask. Features:
     global  the trunk's own pooled vector [mean over active cells | sum / 180]  (512, the fc input)
     part    + for palm and each finger (dominant-joint groups of the PREV mesh's vertices): mean of y4 over the
             active level-4 cells that hold a projected vertex of the group (6 x 256), and log1p(cells) (6)
   Target: the finger rows the trunk should output, (GT_end - prev)_fingers - prev_mlp(prev)_fingers.
2. Ridge (standardized features, intercept), lambda by leave-one-subject-out CV over the nine training subjects.
3. zgz closed loop, S39 stack (ramp1000, W200 clip): the trunk's finger rows replaced by the ridge readout (global or
   part); root / translation rows, prev_mlp, zero-event gate and the filter unchanged. Mode `fc` (the trunk's own fc)
   must reproduce the S43 control.

    python tools/s44/p04_partpool.py --run outputs/semkine/s43n_sp_s3407 --out outputs/s44/p04/s3407
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tools" / "dt"))
from s44 import evflow as EF  # noqa: E402
import filter_eval as FE  # noqa: E402
from mano_layer import ManoLayer  # noqa: E402
from semkine import eval_track as ET  # noqa: E402
from semkine import events as EV  # noqa: E402
from semkine.anchored import AdaptiveFilter  # noqa: E402
from semkine.async_tracker import _Fixed  # noqa: E402
from sparse_trunk import input_mask  # noqa: E402

EX = FE.EX
SPEC = REPO / "outputs" / "dt3" / "gate_specs" / "ramp1000.json"
N_TRAIN, WIN, CELL = 12000, 200, 16
LAMS = np.logspace(-1, 5, 13)


def mixed_noise(rng):
    def fill(a, b, c):
        n = np.empty(51, np.float32)
        n[0:3], n[3:6], n[6:] = rng.standard_normal(3) * a, rng.standard_normal(3) * b, rng.standard_normal(45) * c
        return n
    u = rng.random()
    if u < 0.5:
        return fill(0.005, 0.05, 0.05)
    if u < 0.8:
        return fill(0.05, 0.3, 0.3)
    def directed(n, mag):
        v = rng.standard_normal(n).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-8) * np.float32(rng.uniform(0.0, mag))
    n = np.empty(51, np.float32)
    n[0:3], n[3:6], n[6:] = directed(3, 0.05), directed(3, 0.3), directed(45, 0.3)
    return n


class Trunk:
    """The S43 forward with access to the level-4 map (bit-identical to `MNISTModel.forward` for the fc readout)."""

    def __init__(self, m):
        self.m = m
        dom = m.mano.weights.argmax(dim=1)
        self.group = torch.where(dom == 0, torch.zeros_like(dom), (dom - 1) // 3 + 1)     # 0 palm, 1..5 fingers

    def maps(self, lnes, prev):
        m = self.m
        betas_f, k_f = m._resolve_betas_K(prev, None, None)
        rend = m._render_prev(prev.float(), betas_f, k_f).to(dtype=lnes.dtype)
        x = torch.cat([lnes, rend], dim=-1).permute(0, 3, 1, 2).contiguous()
        m0 = input_mask(x)
        xx = m.conv1(x) * m0
        rn = m.rn
        ms = rn.masks(m0.to(xx.dtype))
        if rn.coord:
            cp = rn.coord_planes(xx.shape[2], xx.shape[3], xx.device, xx.dtype)
            xx = torch.cat([xx, cp.expand(xx.shape[0], -1, -1, -1) * ms[0]], dim=1)
        y = F.relu(rn.bn1(rn.conv1(xx), ms[1]))
        y = F.max_pool2d(y, 3, 2, 1) * ms[2]
        for lvl, stage in zip((2, 3, 4), (rn.layer1, rn.layer2, rn.layer3)):
            for blk in stage:
                y = blk(y, ms[lvl])
        m4 = ms[4]
        s = y.sum(dim=(2, 3), dtype=torch.float32)
        n = m4.sum(dim=(2, 3), dtype=torch.float32).clamp(min=1.0)
        feats = torch.cat([s / n, s / float(m4.shape[2] * m4.shape[3])], dim=1)
        return feats, y, m4, betas_f, k_f

    def part_feats(self, y4, m4, prev, betas_f, k_f):
        m = self.m
        verts, _ = m._fk(prev.float(), betas_f)
        fx, fy, cx, cy = m._intrinsics(k_f)
        z = verts[..., 2].clamp(min=1e-6)
        u = (fx[:, None] * verts[..., 0] / z + cx[:, None]).round().long() // CELL
        v = (fy[:, None] * verts[..., 1] / z + cy[:, None]).round().long() // CELL
        B, C, h4, w4 = y4.shape
        inb = (u >= 0) & (u < w4) & (v >= 0) & (v < h4)
        out, cnts = [], []
        for g in range(6):
            sel = (self.group == g)[None] & inb                                       # (B, V)
            own = torch.zeros(B, h4 * w4, device=y4.device)
            idx = (v.clamp(0, h4 - 1) * w4 + u.clamp(0, w4 - 1))
            own.scatter_reduce_(1, idx, sel.float(), reduce="amax", include_self=True)
            own = own.view(B, 1, h4, w4) * m4
            c = own.sum(dim=(2, 3))
            out.append((y4 * own).sum(dim=(2, 3)) / c.clamp(min=1.0))
            cnts.append(torch.log1p(c))
        return torch.cat(out + cnts, dim=1)

    def pm(self, prev):
        return self.m.prev_mlp(prev)


def build_training(tr, cfg, dev, rng):
    seqs = EF.split_seqs("train")
    ev_ch = EV.event_channels(cfg)
    per = int(np.ceil(N_TRAIN / len(seqs)))
    X_g, X_p, Y, subj = [], [], [], []
    for si, (s, d) in enumerate(seqs):
        events, offsets, aux, pos51 = ET.load_sequence(EF.DATA_ROOT, d, s)
        betas = torch.tensor(aux["betas"], dtype=torch.float32, device=dev).view(1, -1)
        K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=dev).view(1, 3, 3)
        tr.m.set_hand_context(betas, K)
        runs = np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)
        cand = [(int(a), int(e)) for a, b in runs for e in np.arange(a + EF.STEP - 1, b, EF.STEP) if e - EF.STEP >= a]
        pick = rng.choice(len(cand), size=min(per, len(cand)), replace=False)
        items = [cand[i] for i in pick]
        for i0 in range(0, len(items), 64):
            ch = items[i0:i0 + 64]
            lnes = torch.from_numpy(np.stack([ET.build_lnes(events, offsets, e, EX.window_of(offsets, e, "fixed", WIN, 0,
                                                                                            300, floor=a), ev_ch)
                                              for a, e in ch])).to(dev)
            prev = torch.from_numpy(np.stack([pos51[e - EF.STEP] + mixed_noise(rng) for _, e in ch])).float().to(dev)
            ge = torch.from_numpy(np.stack([pos51[e] for _, e in ch])).float().to(dev)
            feats, y4, m4, bf, kf = tr.maps(lnes, prev)
            pf = tr.part_feats(y4, m4, prev, bf, kf)
            pm = tr.pm(prev)
            X_g.append(feats.double().cpu())
            X_p.append(pf.double().cpu())
            Y.append(((ge - prev)[:, EF.FINGER] - pm[:, EF.FINGER]).double().cpu())
            subj += [s.split("_")[0]] * len(ch)
        print(f"  train features {s}: {len(items)}", flush=True)
    return torch.cat(X_g), torch.cat(X_p), torch.cat(Y), np.array(subj)


def ridge_fit(X, Y, lam):
    mu, sd = X.mean(0), X.std(0).clamp(min=1e-6)
    Z = (X - mu) / sd
    ym = Y.mean(0)
    A = Z.T @ Z + lam * torch.eye(Z.shape[1], dtype=Z.dtype)
    Wt = torch.linalg.solve(A, Z.T @ (Y - ym))
    return {"mu": mu, "sd": sd, "W": Wt, "b": ym}


def ridge_apply(r, X):
    return ((X - r["mu"]) / r["sd"]) @ r["W"] + r["b"]


def loso_select(X, Y, subj):
    errs = np.zeros(len(LAMS))
    for s in np.unique(subj):
        te = torch.from_numpy(subj == s)
        for k, lam in enumerate(LAMS):
            r = ridge_fit(X[~te], Y[~te], lam)
            errs[k] += float(((ridge_apply(r, X[te]) - Y[te]) ** 2).sum())
    k = int(np.argmin(errs))
    return float(LAMS[k]), (errs / len(subj)).tolist()


@torch.no_grad()
def closed_loop(tr, cfg, dev, readout, mano32):
    spec = json.loads(SPEC.read_text())
    filt = AdaptiveFilter.from_spec(_Fixed(), spec).to(dev).eval()
    ev_ch = EV.event_channels(cfg)
    rng = np.random.default_rng(0)
    m = tr.m
    summ = {}
    preds_all = {}
    for s, d in EF.split_seqs("val_core"):
        events, offsets, aux, pos51 = ET.load_sequence(EF.DATA_ROOT, d, s)
        betas = torch.tensor(aux["betas"], dtype=torch.float32, device=dev).view(1, -1)
        K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=dev).view(1, 3, 3)
        m.set_hand_context(betas, K)
        runs = np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)
        P, G = [], []
        for ra, rb in runs:
            ends = np.arange(ra + EF.STEP - 1, rb, EF.STEP, dtype=np.int64)
            if not len(ends):
                continue
            prev = torch.from_numpy(pos51[ra].copy() + ET.sample_init_noise(cfg, rng, 1.0)).view(1, -1).to(dev)
            filt.reset_state(prev)
            for end in ends:
                end = int(end)
                n_ev = int(offsets[end + 1] - offsets[end - EF.STEP + 1])
                w = EX.window_of(offsets, end, "fixed", WIN, 0, 300, floor=int(ra))
                x = torch.from_numpy(ET.build_lnes(events, offsets, end, w, ev_ch)).unsqueeze(0).to(dev)
                if readout is None:
                    z = m(x, prev)
                else:
                    feats, y4, m4, bf, kf = tr.maps(x, prev)
                    out = m.rn.fc(feats)
                    if len(P) < 20:                       # the re-implemented trunk is the model's own trunk
                        chk = out + m.prev_mlp(prev.to(out.dtype)) + prev.to(out.dtype)
                        summ.setdefault("trunk_check_max_abs", 0.0)
                        summ["trunk_check_max_abs"] = max(summ["trunk_check_max_abs"],
                                                          float((chk - m(x, prev)).abs().max()) if n_ev > 0 else 0.0)
                    Xf = feats.double().cpu() if readout["kind"] == "global" else \
                        torch.cat([feats.double().cpu(), tr.part_feats(y4, m4, prev, bf, kf).double().cpu()], 1)
                    out = out.clone()
                    out[:, EF.FINGER] = ridge_apply(readout["ridge"], Xf).to(out.dtype).to(dev)
                    out = out + m.prev_mlp(prev.to(out.dtype))
                    empty = x.reshape(1, -1).abs().sum(dim=1, keepdim=True) <= 0
                    delta = torch.where(empty, torch.zeros_like(out), out)
                    z = delta + prev.to(out.dtype)
                filt.trk_model.z = z
                prev = filt(x, prev, n_events=n_ev)
                P.append(prev.cpu().numpy()[0])
                G.append(pos51[end])
        r = {"pred": np.stack(P).astype(np.float32), "gt": np.stack(G).astype(np.float32), "betas": aux["betas"]}
        mm = EX.per_step_metrics(mano32, r, dev)
        summ[s] = {k: float(v.mean()) for k, v in mm.items()}
        summ[s]["n"] = int(len(P))
        preds_all[s] = r["pred"]
    n = sum(summ[s]["n"] for s in ("zgz_global", "zgz_local"))
    summ["overall"] = {"mpjpe_ra_mm": sum(summ[s]["mpjpe_ra_mm"] * summ[s]["n"] for s in ("zgz_global", "zgz_local")) / n}
    return summ, preds_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cfg, m, ckpt, step, _ = FE.build_run(Path(a.run), "last", dev)
    tr = Trunk(m)
    mano32 = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(dev).eval()
    with torch.no_grad():
        Xg, Xp, Y, subj = build_training(tr, cfg, dev, np.random.default_rng(0))
    Xgp = torch.cat([Xg, Xp], 1)
    lam_g, cv_g = loso_select(Xg, Y, subj)
    lam_p, cv_p = loso_select(Xgp, Y, subj)
    rg, rp = ridge_fit(Xg, Y, lam_g), ridge_fit(Xgp, Y, lam_p)
    # the trunk's own fc finger rows on the same training targets (reference)
    fc_err = float(((Xg.float().to(dev) @ m.rn.fc.weight[EF.FINGER].T + m.rn.fc.bias[EF.FINGER]).double().cpu() - Y)
                   .pow(2).mean())
    res = {"run": a.run, "ckpt": str(ckpt), "n_train": int(len(Y)), "lambda_global": lam_g, "lambda_part": lam_p,
           "cv_mse_global": cv_g, "cv_mse_part": cv_p, "fc_train_mse": fc_err, "loops": {}}
    print(f"ridge: lam_g {lam_g} cv {min(cv_g):.5f}  lam_p {lam_p} cv {min(cv_p):.5f}  fc {fc_err:.5f}", flush=True)
    for name, ro in (("fc", None), ("global", {"kind": "global", "ridge": rg}), ("part", {"kind": "part", "ridge": rp})):
        summ, preds = closed_loop(tr, cfg, dev, ro, mano32)
        res["loops"][name] = summ
        np.savez_compressed(out / f"loop_{name}.npz", **preds)
        print(name, json.dumps({s: (round(v["mpjpe_ra_mm"], 4) if isinstance(v, dict) else v) for s, v in summ.items()}),
              f"({time.time() - t0:.0f} s)", flush=True)
    torch.save({"global": rg, "part": rp}, out / "ridge.pt")
    (out / "result.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
