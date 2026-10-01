#!/usr/bin/env python3
"""Root-tracking round: can tracking keep its single-step advantage if every step is re-anchored on an
absolute measurement? (the "absolute measurement + residual tracking" structure, evaluated without training)

Two trained arms run together in one protocol loop: a state-free absolute arm A (x_abs = A(E_t)) and a
tracker T (x_trk = T(E_t, x_{t-1})). Each step the fed-back state is the blend
    root rotation:  R = R_trk Exp(a_r Log(R_trk^T R_abs))      fingers / translation: (1 - a_f) x_trk + a_f x_abs
so a = 0 is the tracker's own loop and a = 1 the absolute arm alone. Every (a_r, a_f) on the grid is a full
protocol loop (init GT + noise, rng 0); the gains are chosen by two-fold cross-fitting over alternating 10 s
blocks (chosen on one fold, scored on the other), as in tools/x1001/filter_screen.py.

    python tools/rt/fuse_eval.py --abs outputs/semkine/rt_cnn_2k_s3407 --trk outputs/semkine/rt_cnndelta_2k_s3407
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "model"), str(REPO / "tools" / "x1001")]
from config import load_config                                   # noqa: E402
from mano_layer import ManoLayer                                 # noqa: E402
from model import MNISTModel                                     # noqa: E402
from semkine import eval_track as ET                             # noqa: E402
from semkine.dataset import sequences_for_split, splits_manifest  # noqa: E402
import evalx as EX                                               # noqa: E402

STEP = 50
GAINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def load(run: Path, ckpt: str, dev):
    cfg = load_config(next(iter(sorted(run.glob("*.yaml")))))
    path, step, _ = EX.find_ckpt(run, ckpt)
    return cfg, MNISTModel.load_from_checkpoint(str(path), cfg=cfg, map_location=dev).to(dev).eval(), step


def blend_root(r_trk, r_abs, a):
    from scipy.spatial.transform import Rotation as Rot
    Rt, Ra = Rot.from_rotvec(r_trk), Rot.from_rotvec(r_abs)
    return (Rt * Rot.from_rotvec(a * (Rt.inv() * Ra).as_rotvec())).as_rotvec()


@torch.no_grad()
def run_loop(m_abs, m_trk, cfg, root, d, s, dev, a_r, a_f):
    events, offsets, aux, pos51 = ET.load_sequence(root, d, s)
    tsub_p = root / d / f"{s}_tsub.npy"
    tsub = np.load(tsub_p, mmap_mode="r") if tsub_p.exists() else None
    betas = torch.tensor(aux["betas"], dtype=torch.float32, device=dev).view(1, -1)
    K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=dev).view(1, 3, 3)
    for m in (m_abs, m_trk):
        if hasattr(m, "set_hand_context"):
            m.set_hand_context(betas, K)

    def call(m, end, prev):
        """one protocol step of either kind of arm: raw events (event graph / LNES-CNN frontends) or LNES"""
        if m.encoder_name:
            ev5 = ET._window_events(events, offsets, tsub, int(end), STEP)
            return m.forward_packet(ET.make_eval_packet(ev5, prev, betas, K, STEP, dev))
        return m(torch.from_numpy(ET.build_lnes(events, offsets, int(end), STEP)).unsqueeze(0).to(dev), prev)
    rng = np.random.default_rng(0)
    preds, gts, ends = [], [], []
    for a, b in np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2):
        es = np.arange(a + STEP - 1, b, STEP, dtype=np.int64)
        if not len(es):
            continue
        prev = torch.from_numpy(pos51[a].copy() + ET.sample_init_noise(cfg, rng, 1.0)).view(1, -1).to(dev)
        for end in es:
            if m_abs is m_trk:
                # --single: both heads of one ABS_TRACK network, blended here instead of in the model
                call(m_trk, end, prev)
                xa, xt = (t.cpu().numpy()[0] for t in m_trk._abs_track_parts)
            else:
                xa = call(m_abs, end, prev).cpu().numpy()[0]
                xt = call(m_trk, end, prev).cpu().numpy()[0]
            out = (1 - a_f) * xt + a_f * xa
            out[3:6] = blend_root(xt[3:6], xa[3:6], a_r)
            prev = torch.from_numpy(out.astype(np.float32)).view(1, -1).to(dev)
            preds.append(out.astype(np.float32))
            gts.append(pos51[int(end)])
            ends.append(int(end))
    return {"pred": np.stack(preds), "gt": np.stack(gts).astype(np.float32), "end": np.asarray(ends),
            "betas": aux["betas"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--abs", default=None)
    ap.add_argument("--trk", default=None)
    ap.add_argument("--single", default=None, help="an ABS_TRACK run: blend its own two heads")
    ap.add_argument("--ckpt", default="last")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda")
    if a.single:
        cfg_t, m_trk, st_t = load(Path(a.single), a.ckpt, dev)
        cfg_a, m_abs, st_a = cfg_t, m_trk, st_t
        a.abs = a.trk = a.single
    else:
        cfg_a, m_abs, st_a = load(Path(a.abs), a.ckpt, dev)
        cfg_t, m_trk, st_t = load(Path(a.trk), a.ckpt, dev)
    mano = ManoLayer(cfg_t["MANO"]["NPZ"], add_mean=False).to(dev).eval()
    root = Path(cfg_t["DATA"]["ROOT"])
    seqs = sequences_for_split(root, "val_core", splits_manifest(cfg_t))
    grid, ends = {}, None
    for ar, af in itertools.product(GAINS, GAINS):
        errs, rots, es = [], [], []
        for s, d in seqs:
            r = run_loop(m_abs, m_trk, cfg_t, root, d, s, dev, ar, af)
            m = EX.per_step_metrics(mano, r, dev)
            errs.append(m["mpjpe_ra_mm"])
            rots.append(m["root_rot_deg"])
            es.append(r["end"] + (10 ** 8 if "local" in s else 0))
        grid[(ar, af)] = (np.concatenate(errs), np.concatenate(rots))
        ends = np.concatenate(es)
        print(f"a_root={ar:.1f} a_fing={af:.1f}: RA {grid[(ar, af)][0].mean():.2f}  root {grid[(ar, af)][1].mean():.2f}", flush=True)
    fold = (ends // 10_000) % 2
    oos, oos_rot, chosen = np.zeros_like(ends, dtype=np.float64), np.zeros_like(ends, dtype=np.float64), {}
    for f in (0, 1):
        m = fold == f
        best = min(grid, key=lambda k: float(grid[k][0][m].mean()))
        chosen[int(1 - f)] = list(best)
        oos[~m] = grid[best][0][~m]
        oos_rot[~m] = grid[best][1][~m]
    base_abs = grid[(1.0, 1.0)][0]
    rep = {"abs_run": a.abs, "abs_step": st_a, "trk_run": a.trk, "trk_step": st_t,
           "tracker_alone": {"ra": float(grid[(0.0, 0.0)][0].mean()), "root": float(grid[(0.0, 0.0)][1].mean())},
           "absolute_alone": {"ra": float(base_abs.mean()), "root": float(grid[(1.0, 1.0)][1].mean())},
           "fused_out_of_selection": {"ra": float(oos.mean()), "root": float(oos_rot.mean())},
           "paired_vs_absolute": EX.block_ci(oos - base_abs, ends),
           "chosen_gains_by_fold": chosen,
           "grid": {f"{k[0]:.1f},{k[1]:.1f}": [float(v[0].mean()), float(v[1].mean())] for k, v in grid.items()}}
    print(json.dumps({k: v for k, v in rep.items() if k != "grid"}, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
