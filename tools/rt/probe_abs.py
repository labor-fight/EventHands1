#!/usr/bin/env python3
"""Root-tracking round, H2 per encoder: how much absolute root rotation a trained encoder's
state-free packet feature carries.

The feature is what the encoder produces before the state enters anywhere: the pooled vector `f` of
an event-graph / LNES-CNN frontend (S37, G3, E8, C37 read `prev` only at the routed readout), or the
penultimate 512-d vector of a dense ResNet18 arm. A ridge map (lambda by 5-fold CV on the training
packets) from the standardised feature to the 3x3 root rotation matrix is fitted on the training
subjects' 50 ms packets, projected to SO(3) (SVD), and scored on zgz as a geodesic angle. Same
absolute-rotation quantity, using no MLP, one export.

    python tools/rt/probe_abs.py --run-dir outputs/semkine/rt_c37_2k_s3407 [--ckpt last|selected|step=N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "model"), str(REPO / "tools" / "tracking")]
from config import load_config                                   # noqa: E402
from model import MNISTModel                                     # noqa: E402
from semkine import eval_track as ET                             # noqa: E402
from semkine.dataset import sequences_for_split, splits_manifest  # noqa: E402
import evalx as EX                                               # noqa: E402

STEP = 50


def windows(root, d, s, every_ms):
    events, offsets, aux, pos51 = ET.load_sequence(root, d, s)
    tsub_p = root / d / f"{s}_tsub.npy"
    tsub = np.load(tsub_p, mmap_mode="r") if tsub_p.exists() else None
    out = []
    for a, b in np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2):
        for end in np.arange(a + STEP - 1, b, every_ms, dtype=np.int64):
            out.append((events, offsets, tsub, int(end), pos51[int(end)]))
    return out


@torch.no_grad()
def features(model, cfg, items, dev, bs=256):
    raw = bool(model.encoder_name)
    feats = []
    for i0 in range(0, len(items), bs):
        chunk = items[i0:i0 + bs]
        if raw:
            evs, ptr = [], [0]
            for b, (events, offsets, tsub, end, _) in enumerate(chunk):
                e = ET._window_events(events, offsets, tsub, end, STEP)
                e[:, 0] = b
                evs.append(e)
                ptr.append(ptr[-1] + len(e))
            ev = torch.from_numpy(np.concatenate(evs)).to(dev)
            f = model.event_encoder(ev, torch.tensor(ptr, device=dev),
                                    torch.full((len(chunk),), STEP * 1e-3, device=dev))
        else:
            x = torch.from_numpy(np.stack([ET.build_lnes(e, o, end, STEP) for e, o, _, end, _ in chunk])).to(dev)
            rn = model.rn
            h = model.conv1(x.permute(0, 3, 1, 2).contiguous())
            h = rn.maxpool(rn.relu(rn.bn1(rn.conv1(h))))
            h = rn.layer4(rn.layer3(rn.layer2(rn.layer1(h))))
            f = torch.flatten(rn.avgpool(h), 1)
        feats.append(f.float().cpu().numpy())
    return np.concatenate(feats)


def rotmats(aa):
    return EX.aa_to_R(torch.from_numpy(aa).double()).numpy().reshape(len(aa), 9)


def ridge_fit(X, Y, lam):
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ Y)


def so3_err_deg(P9, G9):
    P = P9.reshape(-1, 3, 3)
    U, _, Vt = np.linalg.svd(P)
    d = np.sign(np.linalg.det(U @ Vt))
    U[:, :, -1] *= d[:, None]
    R = U @ Vt
    G = G9.reshape(-1, 3, 3)
    tr = np.einsum("nij,nij->n", R, G)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", default="last")
    ap.add_argument("--config", default=None)
    ap.add_argument("--every-ms", type=int, default=250, help="training packet spacing")
    a = ap.parse_args()
    run = Path(a.run_dir)
    cfg = load_config(a.config or next(iter(sorted(run.glob("*.yaml")))))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt, step, _ = EX.find_ckpt(run, a.ckpt)
    model = MNISTModel.load_from_checkpoint(str(ckpt), cfg=cfg, map_location=dev).to(dev).eval()
    root = Path(cfg["DATA"]["ROOT"])
    mani = splits_manifest(cfg)
    tr_items = [w for s, d in sequences_for_split(root, "train", mani) for w in windows(root, d, s, a.every_ms)]
    te = {s: windows(root, d, s, STEP) for s, d in sequences_for_split(root, "val_core", mani)}
    Xtr = features(model, cfg, tr_items, dev)
    Ytr = rotmats(np.stack([w[4][3:6] for w in tr_items]))
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xs = np.hstack([(Xtr - mu) / sd, np.ones((len(Xtr), 1))])
    folds = np.arange(len(Xs)) % 5
    cv = {}
    for lam in (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4):
        errs = []
        for k in range(5):
            W = ridge_fit(Xs[folds != k], Ytr[folds != k], lam)
            errs.append(so3_err_deg(Xs[folds == k] @ W, Ytr[folds == k]).mean())
        cv[lam] = float(np.mean(errs))
    lam = min(cv, key=cv.get)
    W = ridge_fit(Xs, Ytr, lam)
    res = {"run": run.name, "ckpt": str(ckpt), "step": step, "n_train": len(Xtr), "lambda": lam,
           "cv_train_deg": cv[lam]}
    allerr = []
    for s, items in te.items():
        X = np.hstack([(features(model, cfg, items, dev) - mu) / sd, np.ones((len(items), 1))])
        e = so3_err_deg(X @ W, rotmats(np.stack([w[4][3:6] for w in items])))
        res[s] = float(e.mean())
        allerr.append(e)
    res["zgz_all"] = float(np.concatenate(allerr).mean())
    out = run / f"probe_abs_{a.ckpt.replace('=', '')}.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
