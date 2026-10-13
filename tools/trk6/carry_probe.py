#!/usr/bin/env python3
"""trk6 / S42: one-step carry-over of a previous-state error (no filter, no training).

On zgz steps (every `--every`-th step of every segment, W200 clipped LNES, the evaluation's own windows), the bare
network is run twice: prev = GT(t-1) (clean) and prev = GT(t-1) + d, d one block's perturbation (root: rotation by
`--root-deg` about a random axis; fingers: N(0, sigma) on the 45 angles; transl: a random direction x `--t-mm`).
carry = <J(out_d) - J(out_0), J(prev_d) - J(prev_0)> / |J(prev_d) - J(prev_0)|^2 on root-aligned joints (root, fingers)
or on the translation (transl): 1 = the perturbation passes through untouched, 0 = fully corrected in one step.
Also the clean one-step RA (TF). Seeded, identical draws for every model.
    python tools/trk6/carry_probe.py --run-dir R [--ckpt last] [--every 3]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "model")); sys.path.insert(0, str(REPO / "tools"))
from config import load_config                                        # noqa: E402
from mano_layer import ManoLayer                                      # noqa: E402
from model import MNISTModel                                          # noqa: E402
from semkine import eval_track as ET                                  # noqa: E402
from semkine import events as EV                                      # noqa: E402
from semkine.dataset import sequences_for_split                       # noqa: E402
from semkine.eval_track import mano_fk                                # noqa: E402
from tracking.evalx import find_ckpt                                  # noqa: E402

STEP = 50


def aa_compose(rot_aa, aa):
    from scipy.spatial.transform import Rotation as Rr
    return (Rr.from_rotvec(rot_aa) * Rr.from_rotvec(aa)).as_rotvec().astype(np.float32)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", default="last")
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--window-ms", type=int, default=200)
    ap.add_argument("--root-deg", type=float, default=8.0)
    ap.add_argument("--finger-sigma", type=float, default=0.10)
    ap.add_argument("--t-mm", type=float, default=15.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = torch.device("cuda")
    run = Path(a.run_dir)
    cfg = load_config(next(iter(sorted(run.glob("*.yaml")))))
    ck, step, _ = find_ckpt(run, a.ckpt)
    model = MNISTModel.load_from_checkpoint(str(ck), cfg=cfg, map_location=dev).to(dev).eval()
    mano = ManoLayer(str(REPO / "assets/mano_right.npz"), add_mean=False).to(dev).eval()
    root = Path(cfg["DATA"]["ROOT"])
    ev_ch = EV.event_channels(cfg)
    res = {}
    for seq, d in sequences_for_split(root, "val_core"):
        events, offsets, aux, pos51 = ET.load_sequence(root, d, seq)
        betas = torch.tensor(aux["betas"], dtype=torch.float32, device=dev).view(1, -1)
        K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=dev).view(1, 3, 3)
        model.set_hand_context(betas, K)
        rng = np.random.default_rng(1234)
        xs, prevs, gts = [], [], []
        for ra, rb in np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2):
            ends = np.arange(ra + STEP - 1, rb, STEP, dtype=np.int64)
            for k, end in enumerate(ends):
                if k == 0 or k % a.every:
                    continue
                w = min(a.window_ms, int(end + 1 - ra))
                xs.append(ET.build_lnes(events, offsets, int(end), w, ev_ch))
                prevs.append(np.asarray(pos51[int(end) - STEP], np.float32))
                gts.append(np.asarray(pos51[int(end)], np.float32))
        X = torch.from_numpy(np.stack(xs)).to(dev)
        P0 = np.stack(prevs)
        G = np.stack(gts)
        n = len(P0)
        pert = {}
        ax = rng.standard_normal((n, 3)); ax /= np.linalg.norm(ax, axis=1, keepdims=True)
        pr = P0.copy()
        for i in range(n):
            pr[i, 3:6] = aa_compose(ax[i] * np.deg2rad(a.root_deg), P0[i, 3:6])
        pert["root"] = pr
        pf = P0.copy(); pf[:, 6:] += rng.standard_normal((n, 45)).astype(np.float32) * a.finger_sigma
        pert["fingers"] = pf
        dt = rng.standard_normal((n, 3)); dt /= np.linalg.norm(dt, axis=1, keepdims=True)
        pt = P0.copy(); pt[:, :3] += (dt * a.t_mm / 1000).astype(np.float32)
        pert["transl"] = pt

        def run_net(P):
            outs = []
            for i0 in range(0, n, 64):
                outs.append(model(X[i0:i0 + 64], torch.from_numpy(P[i0:i0 + 64]).to(dev)).float().cpu().numpy())
            return np.concatenate(outs)

        def raj(p):
            j = mano_fk(mano, p.astype(np.float32), betas, dev)[0]
            return j - j[:, :1]

        O0 = run_net(P0)
        J0, JG, JP0 = raj(O0), raj(G), raj(P0)
        r = {"n": n, "tf_ra_mm": float(np.linalg.norm(J0 - JG, axis=-1).mean() * 1000),
             "hold_ra_mm": float(np.linalg.norm(JP0 - JG, axis=-1).mean() * 1000)}
        for name, P in pert.items():
            O = run_net(P)
            if name == "transl":
                din, dout = (P - P0)[:, :3], (O - O0)[:, :3]
            else:
                din = (raj(P) - JP0).reshape(n, -1)
                dout = (raj(O) - J0).reshape(n, -1)
            r[f"carry_{name}"] = float((din * dout).sum() / (din * din).sum())
            r[f"carry_{name}_resid"] = float(np.linalg.norm(dout - din * r[f"carry_{name}"], axis=1).mean() /
                                             np.linalg.norm(din, axis=1).mean())
        res[seq] = r
        print(f"{run.name} {seq:10s} n={n} TF {r['tf_ra_mm']:.2f} (hold {r['hold_ra_mm']:.2f}) carry root "
              f"{r['carry_root']:.3f} fingers {r['carry_fingers']:.3f} transl {r['carry_transl']:.3f}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"run": run.name, "step": step, "args": vars(a), "res": res}, indent=1))


if __name__ == "__main__":
    main()
