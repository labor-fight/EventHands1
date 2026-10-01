#!/usr/bin/env python3
"""x1001 Phase 1 analysis: E1a / H2 absolute decodability, E4 / H5 event-rate sweep, effective
temporal receptive field, spatial coverage -- all from the one export of `diag_export.py`.

Probes are fitted on the training subjects only (64 sequences, every 50 ms packet). Their
hyper-parameters (ridge lambda; MLP weight decay and epoch) are chosen on the development subject
zgz by two-fold cross-fitting over alternating 10 s time blocks: chosen on fold A and scored on
fold B, and the reverse, so every reported dev number is out-of-selection. GT-routed features
(oracle) and GT-initialised closed loops (protocol) are reported in their own table and never
enter the normal-inference verdict, which uses only GT-free inputs.

    python tools/x1001/diag_probe.py            # writes diag/e1a_e4/report.{json,md}
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))
from mano_layer import ManoLayer                                      # noqa: E402
from pose_repr import decode_to_mano_inputs                           # noqa: E402

PROG = Path("/data1/lyq/code/mesh/EventHands1_x1001")
D = PROG / "diag" / "e1a_e4"
DATA = Path("/data1/lyq/code/mesh/EventHands/data/hand_data51")
MANO_NPZ = REPO / "assets" / "mano_right.npz"
GROUPS = {"g1": ("s37a_sel", "cnna_sel"), "g2": ("s37a_last",), "g3": ("s37b_sel", "cnnb_sel"), "g4": ("s37b_last",)}
BLOCK_MS = 10_000
SEQS = ("zgz_global", "zgz_local")
KEEPS = {"zgz_global": (1.0, 0.5, 0.2, 0.1, 0.05, 0.03), "zgz_local": (1.0, 0.5, 0.25, 0.13)}
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------------------------ geometry
def aa_to_R(aa):
    th = aa.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    k = aa / th
    K = torch.zeros(*aa.shape[:-1], 3, 3, dtype=aa.dtype, device=aa.device)
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    I = torch.eye(3, dtype=aa.dtype, device=aa.device).expand_as(K)
    s, c = th.sin()[..., None], th.cos()[..., None]
    return I + s * K + (1 - c) * (K @ K)


def R_to_aa(R):
    tr = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    th = ((tr - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7).acos()
    w = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    return w / (2 * th.sin()[..., None]).clamp_min(1e-9) * th[..., None]


def project_so3(M):
    U, _, Vh = torch.linalg.svd(M)
    d = torch.det(U @ Vh)
    Dg = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    return U @ Dg @ Vh


def geodesic_deg(Ra, Rb):
    tr = (Ra.transpose(-1, -2) @ Rb).diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.rad2deg(((tr - 1) / 2).clamp(-1, 1).acos())


HANDS_MEAN = torch.from_numpy(np.load(MANO_NPZ)["hands_mean"].astype(np.float64).reshape(-1))


def finger_err_deg(pred45, gt45):
    """Mean over the 15 joints of the geodesic angle between the local rotations (mean added)."""
    p = aa_to_R((pred45.double() + HANDS_MEAN).reshape(-1, 15, 3))
    g = aa_to_R((gt45.double() + HANDS_MEAN).reshape(-1, 15, 3))
    return geodesic_deg(p, g).mean(-1)


class FK:
    def __init__(self):
        self.mano = ManoLayer(str(MANO_NPZ), add_mean=False).eval()
        self.betas = {}
        for s in SEQS:
            aux = np.load(DATA / "val" / f"{s}_aux.npz", allow_pickle=True)
            self.betas[s] = torch.tensor(np.asarray(aux["betas"], np.float32)).view(1, -1)

    @torch.no_grad()
    def joints(self, p51, seq):
        out = []
        for i in range(0, len(p51), 2048):
            c = torch.as_tensor(p51[i:i + 2048], dtype=torch.float32)
            dec = decode_to_mano_inputs(c, "mano_full_axis_angle", self.mano.hands_components, self.mano.hands_mean)
            v, j = self.mano(self.betas[seq].expand(len(c), -1), dec["global_orient"], dec["local_full_aa"], dec["transl"])
            out.append(j)
        j = torch.cat(out).numpy()
        return j - j[:, :1]

    def ra(self, pred51, gt51, seq):
        return np.linalg.norm(self.joints(pred51, seq) - self.joints(gt51, seq), axis=-1).mean(-1) * 1000


# ------------------------------------------------------------------------------------ data
def load(group, sub, key):
    return np.load(D / group / sub / f"{key}.npy", mmap_mode="r")


def dev_rows(seq, keep=1.0):
    sub = f"dev_{seq}_k{keep:g}"
    return {"gt": np.asarray(load("g1", sub, "gt")), "end": np.asarray(load("g1", sub, "end")),
            "run": np.asarray(load("g1", sub, "run")), "count": np.asarray(load("g1", sub, "count")),
            "gstat": np.asarray(load("g1", sub, "gstat")), "lnes_active": np.asarray(load("g1", sub, "lnes_active"))}


def feature(name, split, seq=None, keep=1.0):
    """name = 'hist' | '<model>:pf' | '<model>:pen' | '<model>:r_<state>'; split = train | dev | occl."""
    if name == "hist":
        g = "g1"
        key = "hist"
    else:
        model, kind = name.split(":")
        g = next(k for k, v in GROUPS.items() if model in v)
        key = f"{kind}_{model}"
    if split == "train":
        if name.split(":")[-1] in ("r_gtinit", "r_cold"):
            return np.asarray(load(g, f"train_loop_{name.split('_')[-1]}", key), np.float32)
        return np.asarray(load(g, "train", key), np.float32)
    if split == "occl":
        return np.asarray(load(g, f"occl_{seq}", key), np.float32)
    if name.split(":")[-1] in ("r_gtinit", "r_cold"):
        state = name.split("_")[-1]
        sub = "devloop_k1_gtinit" if state == "gtinit" else "devloop_k1_cold"
        arr = np.asarray(load(g, sub, key), np.float32)
        seqi = np.asarray(load(g, sub, "seq"))
        return arr[seqi == SEQS.index(seq)]
    return np.asarray(load(g, f"dev_{seq}_k{keep:g}", key), np.float32)


def train_rows_aligned(name):
    """GT and keys of the training rows a feature was exported on (loops are segment-ordered)."""
    if name.split(":")[-1] in ("r_gtinit", "r_cold"):
        model = name.split(":")[0]
        g = next(k for k, v in GROUPS.items() if model in v)
        sub = f"train_loop_{name.split('_')[-1]}"
        return np.asarray(load(g, sub, "gt"))
    return np.asarray(load("g1", "train", "gt"))


def targets(gt51):
    R = aa_to_R(torch.as_tensor(gt51[:, 3:6], dtype=torch.float64))
    return {"root": R.reshape(-1, 9).numpy(), "fing": np.asarray(gt51[:, 6:51], np.float64)}


def folds(end):
    return ((end // BLOCK_MS) % 2).astype(np.int64)


# ------------------------------------------------------------------------------------ probes
def ridge_family(Xtr, Ytr, Xdevs, lams=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)):
    """Closed-form ridge on train-standardised features; dev predictions for every lambda."""
    X = torch.as_tensor(Xtr, dtype=torch.float64, device=DEV)
    Y = torch.as_tensor(Ytr, dtype=torch.float64, device=DEV)
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
    A = (X - mu) / sd
    ym = Y.mean(0)
    evals, V = torch.linalg.eigh(A.T @ A)
    B = V.T @ (A.T @ (Y - ym))
    n = len(A)
    outs = []
    for Xd in Xdevs:
        Ad = ((torch.as_tensor(Xd, dtype=torch.float64, device=DEV) - mu) / sd) @ V
        outs.append({lam: (Ad @ (B / (evals + lam * n).unsqueeze(1)) + ym).cpu().numpy() for lam in lams})
    trfit = {lam: float((((A @ V) @ (B / (evals + lam * n).unsqueeze(1)) + ym - Y) ** 2).sum(1).mean()) for lam in lams}
    return outs, trfit


def mlp_family(Xtr, Ytr, Xdevs, wds=(1e-4, 1e-2), epochs=30, seed=0, hidden=256):
    """2 x 256 ReLU MLP (dropout 0.1), AdamW; dev predictions after every epoch for every wd,
    plus proof of training (step-0 gradient, parameter change, loss curves)."""
    X = torch.as_tensor(Xtr, dtype=torch.float32)
    Y = torch.as_tensor(Ytr, dtype=torch.float32)
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
    ym, ys = Y.mean(0), Y.std(0).clamp_min(1e-6)
    A = ((X - mu) / sd).to(DEV)
    y = ((Y - ym) / ys).to(DEV)
    Ads = [((torch.as_tensor(Xd, dtype=torch.float32) - mu) / sd).to(DEV) for Xd in Xdevs]
    res = {}
    for wd in wds:
        torch.manual_seed(seed)
        net = torch.nn.Sequential(torch.nn.Linear(A.shape[1], hidden), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                                  torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                                  torch.nn.Linear(hidden, y.shape[1])).to(DEV)
        p0 = [p.detach().clone() for p in net.parameters()]
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        g = torch.Generator(device="cpu").manual_seed(seed)
        preds, curve, proof = [], [], {}
        n = len(A)
        for ep in range(epochs):
            net.train()
            perm = torch.randperm(n, generator=g).to(DEV)
            tl = []
            with torch.enable_grad():
                for i in range(0, n, 1024):
                    idx = perm[i:i + 1024]
                    loss = ((net(A[idx]) - y[idx]) ** 2).mean()
                    opt.zero_grad()
                    loss.backward()
                    if ep == 0 and i == 0:
                        gr = [p.grad for p in net.parameters()]
                        proof = {"grad_enabled": torch.is_grad_enabled(), "loss_step0": float(loss),
                                 "grad_norm_step0": float(torch.sqrt(sum((q.float() ** 2).sum() for q in gr))),
                                 "params_with_grad": sum(int(q is not None and bool((q != 0).any())) for q in gr),
                                 "n_params": len(gr)}
                    opt.step()
                    tl.append(float(loss))
            sched.step()
            net.eval()
            with torch.no_grad():
                preds.append([(net(Ad) * ys.to(DEV) + ym.to(DEV)).cpu().numpy().astype(np.float64) for Ad in Ads])
            curve.append(float(np.mean(tl)))
        num = sum(float(((p.detach() - q) ** 2).sum()) for p, q in zip(net.parameters(), p0))
        den = sum(float((q ** 2).sum()) for q in p0)
        proof.update({"rel_param_change": math.sqrt(num / max(den, 1e-12)), "train_loss_curve": curve})
        res[wd] = (preds, proof)
    return res


def crossfit(pred_by_hp, y_dev, fold, err_fn):
    """pred_by_hp: {hp: (n, d)}; choose hp on fold A, score on B and vice versa.
    Returns (out-of-selection predictions, chosen hp per fold)."""
    out = np.zeros_like(next(iter(pred_by_hp.values())))
    chosen = {}
    for f in (0, 1):
        sel = fold == f
        best = min(pred_by_hp, key=lambda hp: float(err_fn(pred_by_hp[hp][sel], y_dev[sel]).mean()))
        chosen[int(1 - f)] = best
        out[~sel] = pred_by_hp[best][~sel]
    return out, chosen


def root_err(pred9, gt9):
    R = project_so3(torch.as_tensor(pred9, dtype=torch.float64).reshape(-1, 3, 3))
    return geodesic_deg(R, torch.as_tensor(gt9, dtype=torch.float64).reshape(-1, 3, 3)).numpy()


def fing_err(pred45, gt45):
    return finger_err_deg(torch.as_tensor(pred45), torch.as_tensor(gt45)).numpy()


def block_ci(vals, end, n_boot=2000, seed=0):
    """Mean and 95% CI by bootstrap over 10 s blocks (paired when vals is a difference)."""
    blocks = end // BLOCK_MS
    ub = np.unique(blocks)
    sums = np.array([vals[blocks == b].sum() for b in ub])
    cnts = np.array([(blocks == b).sum() for b in ub])
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(ub), len(ub))
        bs.append(sums[i].sum() / cnts[i].sum())
    return float(vals.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


# ------------------------------------------------------------------------------------ main
def main():
    t0 = time.time()
    fk = FK()
    rep = {"device": str(DEV), "block_ms": BLOCK_MS}
    gt_tr = np.asarray(load("g1", "train", "gt"))
    mean_pose = gt_tr.mean(0)
    dev = {s: dev_rows(s) for s in SEQS}
    dev_gt = np.concatenate([dev[s]["gt"] for s in SEQS])
    dev_end = np.concatenate([dev[s]["end"] + (10**8 if s == "zgz_local" else 0) for s in SEQS])
    dev_seq = np.concatenate([[i] * len(dev[s]["gt"]) for i, s in enumerate(SEQS)])
    fold = folds(dev_end)
    T_dev = targets(dev_gt)

    normal = ["hist", "s37a_sel:pf", "s37b_sel:pf", "s37a_last:pf", "s37b_last:pf", "cnna_sel:pen", "cnnb_sel:pen"]
    routed = ["s37a_sel:r_cold", "s37b_sel:r_cold", "s37a_sel:r_gtinit", "s37b_sel:r_gtinit",
              "s37a_sel:r_oracle", "s37b_sel:r_oracle"]
    results, probes_kept = {}, {}
    # train-mean reference
    for tgt, ef in (("root", root_err), ("fing", fing_err)):
        mpred = np.repeat(targets(mean_pose[None])[tgt], len(dev_gt), 0)
        results[f"mean|{tgt}|const"] = ef(mpred, T_dev[tgt])
    for name in normal + routed:
        Xtr = feature(name, "train")
        gtr = train_rows_aligned(name)
        T_tr = targets(gtr)
        Xdev = np.concatenate([feature(name, "dev", s) for s in SEQS])
        assert len(Xdev) == len(dev_gt), (name, Xdev.shape, dev_gt.shape)
        if name.split(":")[-1] in ("r_gtinit", "r_cold"):
            model = name.split(":")[0]
            g = next(k for k, v in GROUPS.items() if model in v)
            sub = "devloop_k1_gtinit" if name.endswith("gtinit") else "devloop_k1_cold"
            lgt = np.asarray(load(g, sub, "gt"))
            assert np.array_equal(lgt, dev_gt), f"{name}: closed-loop rows not aligned with dev rows"
        # extra dev inputs for the E4 sweep and the occlusion RF (normal-inference features only)
        extras, extra_keys = [], []
        if name in normal:
            for s in SEQS:
                for k in KEEPS[s][1:]:
                    extras.append(feature(name, "dev", s, k))
                    extra_keys.append(("thin", s, k))
                extras.append(feature(name, "occl", s))
                extra_keys.append(("occl", s, None))
        for tgt, ef in (("root", root_err), ("fing", fing_err)):
            rfam, trfit = ridge_family(Xtr, T_tr[tgt], [Xdev] + extras)
            pr, ch = crossfit(rfam[0], T_dev[tgt], fold, ef)
            results[f"{name}|{tgt}|ridge"] = ef(pr, T_dev[tgt])
            rep.setdefault("ridge_lambda", {})[f"{name}|{tgt}"] = {str(k): v for k, v in ch.items()}
            mfam = mlp_family(Xtr, T_tr[tgt], [Xdev] + extras)
            hp_preds = {(wd, ep): mfam[wd][0][ep][0] for wd in mfam for ep in range(len(mfam[wd][0]))}
            pm, chm = crossfit(hp_preds, T_dev[tgt], fold, ef)
            results[f"{name}|{tgt}|mlp"] = ef(pm, T_dev[tgt])
            rep.setdefault("mlp_choice", {})[f"{name}|{tgt}"] = {str(k): list(v) for k, v in chm.items()}
            rep.setdefault("mlp_proof", {})[f"{name}|{tgt}"] = {
                str(wd): {k: v for k, v in mfam[wd][1].items() if k != "train_loss_curve"}
                | {"train_loss_first_last": [mfam[wd][1]["train_loss_curve"][0], mfam[wd][1]["train_loss_curve"][-1]]}
                for wd in mfam}
            # for the sweep / occlusion: the hyper-parameters best on the whole dev set at keep 1
            if extras:
                best = min(hp_preds, key=lambda hp: float(ef(hp_preds[hp], T_dev[tgt]).mean()))
                rbest = min(rfam[0], key=lambda l: float(ef(rfam[0][l], T_dev[tgt]).mean()))
                probes_kept[(name, tgt)] = {"mlp": [mfam[best[0]][0][best[1]][1 + i] for i in range(len(extras))],
                                            "ridge": [rfam[1 + i][rbest] for i in range(len(extras))],
                                            "keys": extra_keys, "full_mlp": hp_preds[best], "full_ridge": rfam[0][rbest]}
            # RA contribution of the probe's own prediction (GT translation)
            if tgt == "root":
                results[f"{name}|root|_pred9_mlp"] = pm
            else:
                results[f"{name}|fing|_pred45_mlp"] = pm
        print(f"[{time.time() - t0:.0f}s] {name} done", flush=True)

    # ---------------- summaries
    def stat(arr, mask):
        return block_ci(arr[mask], dev_end[mask])
    summ = {}
    for key, arr in results.items():
        if "|_pred" in key:
            continue
        summ[key] = {s: stat(arr, dev_seq == i) for i, s in enumerate(SEQS)}
        summ[key]["all"] = stat(arr, np.ones_like(dev_seq, bool))
    rep["decode"] = summ
    # paired differences (per seed): S37 pf vs CNN pen, S37 pf vs hist
    pairs = {}
    for tag in ("a", "b"):
        for ck in ("sel", "last"):
            for tgt in ("root", "fing"):
                for pt in ("ridge", "mlp"):
                    s37 = results[f"s37{tag}_{ck}:pf|{tgt}|{pt}"]
                    cnn = results[f"cnn{tag}_sel:pen|{tgt}|{pt}"]
                    hist = results[f"hist|{tgt}|{pt}"]
                    for i, s in enumerate(SEQS + ("all",)):
                        m = (dev_seq == i) if s != "all" else np.ones_like(dev_seq, bool)
                        pairs[f"s37{tag}_{ck}-cnn{tag}|{tgt}|{pt}|{s}"] = block_ci((s37 - cnn)[m], dev_end[m])
                        pairs[f"s37{tag}_{ck}-hist|{tgt}|{pt}|{s}"] = block_ci((s37 - hist)[m], dev_end[m])
    rep["paired"] = pairs

    # RA contributions of the probe predictions (MLP, GT translation): rotation-only, finger-only, both
    ra = {}
    for name in normal + routed:
        p9 = results[f"{name}|root|_pred9_mlp"]
        p45 = results[f"{name}|fing|_pred45_mlp"]
        Rp = project_so3(torch.as_tensor(p9).reshape(-1, 3, 3))
        aa = R_to_aa(Rp).numpy()
        for i, s in enumerate(SEQS):
            m = dev_seq == i
            g = dev_gt[m]
            rot_only, fing_only, both = g.copy(), g.copy(), g.copy()
            rot_only[:, 3:6] = aa[m]
            fing_only[:, 6:51] = p45[m]
            both[:, 3:6], both[:, 6:51] = aa[m], p45[m]
            ra[f"{name}|{s}"] = {k: block_ci(fk.ra(v, g, s), dev_end[m])
                                 for k, v in (("rot_only", rot_only), ("fing_only", fing_only), ("both", both))}
    rep["ra_contrib_mlp"] = ra

    # ---------------- E4: closed-loop / per-frame RA and decodability versus keep
    e4 = {}
    for s in SEQS:
        for k in KEEPS[s]:
            rows = dev_rows(s, k)
            ent = {"events_median": float(np.median(rows["count"])),
                   "gstat_median": dict(zip(("events", "nodes", "retention", "span1_ms", "span3_ms", "rad3_px",
                                             "edge_px", "cell_cov"), np.median(rows["gstat"], 0).round(4).tolist())),
                   "lnes_active_median": float(np.median(rows["lnes_active"]))}
            for model in ("cnna_sel", "cnnb_sel"):
                out = np.asarray(load("g1" if model == "cnna_sel" else "g3", f"dev_{s}_k{k:g}", f"out_{model}"))
                ent[f"ra_{model}"] = float(fk.ra(out, rows["gt"], s).mean())
            for g, models in GROUPS.items():
                for model in models:
                    if not model.startswith("s37"):
                        continue
                    sub = f"devloop_k{k:g}_gtinit"
                    lo = np.asarray(load(g, sub, f"loop_out_{model}"))
                    lseq = np.asarray(load(g, sub, "seq"))
                    seqs_in = json.loads((D / g / sub / "meta.json").read_text())["seqs"]
                    lo = lo[lseq == seqs_in.index(s)]
                    lgt = np.asarray(load(g, sub, "gt"))[lseq == seqs_in.index(s)]
                    ent[f"ra_loop_gtinit_{model}"] = float(fk.ra(lo, lgt, s).mean())
                    if k == 1.0:
                        sub = "devloop_k1_cold"
                        lo = np.asarray(load(g, sub, f"loop_out_{model}"))
                        lseq = np.asarray(load(g, sub, "seq"))
                        seqs_in = json.loads((D / g / sub / "meta.json").read_text())["seqs"]
                        lo = lo[lseq == seqs_in.index(s)]
                        lgt = np.asarray(load(g, sub, "gt"))[lseq == seqs_in.index(s)]
                        ent[f"ra_loop_cold_{model}"] = float(fk.ra(lo, lgt, s).mean())
                        ent[f"rot_loop_cold_{model}"] = float(root_err(
                            aa_to_R(torch.as_tensor(lo[:, 3:6], dtype=torch.float64)).reshape(-1, 9).numpy(),
                            targets(lgt)["root"]).mean())
            # probe-decoded root rotation at this keep (probes fitted at keep 1 on clean train features)
            for (name, tgt), pk in probes_kept.items():
                if k == 1.0:
                    pred = pk["full_mlp"][dev_seq == SEQS.index(s)]
                else:
                    j = pk["keys"].index(("thin", s, k))
                    pred = pk["mlp"][j]
                ef = root_err if tgt == "root" else fing_err
                ent[f"probe_{tgt}_{name}"] = float(ef(pred, targets(rows["gt"])[tgt]).mean())
            e4[f"{s}|k{k:g}"] = ent
    rep["e4"] = e4

    # ---------------- effective temporal RF by 5 ms occlusion (every 4th dev packet)
    rf = {}
    for s in SEQS:
        full_idx = np.arange(len(dev[s]["gt"]))[::4]
        n_full = len(full_idx)
        for name in normal:
            f_full = feature(name, "dev", s)[full_idx]
            f_occ = feature(name, "occl", s).reshape(n_full, 10, -1)
            rel = np.linalg.norm(f_occ - f_full[:, None], axis=-1) / np.maximum(np.linalg.norm(f_full, axis=-1)[:, None], 1e-6)
            prof = rel.mean(0)
            pk = probes_kept[(name, "root")]
            j = pk["keys"].index(("occl", s, None))
            p_occ = pk["mlp"][j].reshape(n_full, 10, 9)
            p_full = pk["full_mlp"][dev_seq == SEQS.index(s)][full_idx]
            dang = np.stack([root_err(p_occ[:, b], project_so3(torch.as_tensor(p_full).reshape(-1, 3, 3)).reshape(-1, 9).numpy())
                             for b in range(10)], 1).mean(0)

            def span90(p):
                # bins are 5 ms slices from the window start; age = time before the window end
                w = p[::-1] / max(p.sum(), 1e-12)                 # youngest slice first
                c = np.cumsum(w)
                return float((np.searchsorted(c, 0.9) + 1) * 5.0)
            rf[f"{name}|{s}"] = {"feat_rel_change_by_slice": prof.round(5).tolist(),
                                 "root_deg_change_by_slice": dang.round(4).tolist(),
                                 "feat_span90_ms": span90(prof), "root_span90_ms": span90(dang)}
    rep["temporal_rf"] = rf
    rep["seconds"] = round(time.time() - t0, 1)
    (D / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(f"wrote {D / 'report.json'} in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
