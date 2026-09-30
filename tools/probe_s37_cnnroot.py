#!/usr/bin/env python3
"""How much is a better absolute root worth to S37? (docs/S37_ROTW_CNNROOT_PREREG.md, part B.) Zero training.

S37's closed loop on zgz (val_core, 50 ms, the make_s36_row rng protocol), with the absolute CNN
(ResNet18 on LNES, per-frame, no prev) run on the window ending at every S37 output. Variants:

  s37       S37 alone (must reproduce its main row)
  cnn       the CNN's own absolute output (reference)
  fuse_R    S37 output with its global rotation replaced by the CNN's, fed back as the next prev
  fuse_RT   the same with translation and global rotation (the whole root) replaced

Per sequence and pooled: RA, root rotation error, articulation-only RA (GT root substituted).

    CUDA_VISIBLE_DEVICES=6 python tools/probe_s37_cnnroot.py \\
        --pair outputs/semkine/s37_routed_s3407=outputs/semkine/s37diag_cnnabs_s3407 \\
        --pair outputs/semkine/s37_routed_s3408=outputs/semkine/s37diag_cnnabs_s3408 \\
        --out outputs/semkine/probe_s37_cnnroot.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))
sys.path.insert(0, str(REPO / "tools"))

from probe_s37_rootinnov import rot_deg                           # noqa: E402
from probe_s37_route import load_run                              # noqa: E402
from semkine import eval_track as ET                              # noqa: E402
from semkine import events as EV                                  # noqa: E402
from semkine.dataset import sequences_for_split, splits_manifest  # noqa: E402

STEP = 50
VARIANTS = ("s37", "cnn", "fuse_R", "fuse_RT")
SMOKE = False


@torch.no_grad()
def run_seq(s37, cnn, cnn_cfg, cfg, root, d, seq, device, rng, variant):
    events, offsets, aux, pos51 = ET.load_sequence(root, d, seq)
    p = root / d / f"{seq}_tsub.npy"
    tsub = np.load(p, mmap_mode="r") if p.exists() else None
    betas = torch.tensor(aux["betas"], dtype=torch.float32, device=device).view(1, -1)
    K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=device).view(1, 3, 3)
    s37.set_hand_context(betas, K)
    win = int(cnn_cfg.get("EVAL", {}).get("WINDOW_MS", STEP))
    ch = EV.event_channels(cnn_cfg)
    preds, gts = [], []
    for a, b in np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2):
        ends = np.arange(a + STEP - 1, b, STEP, dtype=np.int64)
        if not len(ends):
            continue
        if SMOKE and preds:
            break
        if SMOKE:
            ends = ends[:40]
        prev = torch.from_numpy(pos51[a].copy() + ET.sample_init_noise(cfg, rng, 1.0)).view(1, -1).to(device)
        for e in ends:
            w = min(win, int(e) - int(a) + 1)
            x = torch.from_numpy(ET.build_lnes(events, offsets, int(e), w, ch)).unsqueeze(0).to(device)
            c = cnn(x, prev).float()
            if variant == "cnn":
                pred = c
            else:
                ev5 = ET._window_events(events, offsets, tsub, int(e), STEP)
                pred = s37.forward_packet(ET.make_eval_packet(ev5, prev, betas, K, STEP, device)).float()
                if variant == "fuse_R":
                    pred[:, 3:6] = c[:, 3:6]
                elif variant == "fuse_RT":
                    pred[:, 0:6] = c[:, 0:6]
            prev = pred
            preds.append(pred[0].cpu().numpy())
            gts.append(pos51[int(e)])
    P = torch.from_numpy(np.stack(preds)).to(device)
    G = torch.from_numpy(np.stack(gts).astype(np.float32)).to(device)

    def fk(x):
        return torch.cat([s37._fk(x[i:i + 2048], betas.expand(len(x[i:i + 2048]), -1))[1]
                          for i in range(0, len(x), 2048)])

    def ra(jp, jg):
        return ((jp - jp[:, :1]) - (jg - jg[:, :1])).norm(dim=-1).mean(-1) * 1000

    jg = fk(G)
    Part = P.clone(); Part[:, 0:6] = G[:, 0:6]
    return {"n": int(len(P)), "ra": float(ra(fk(P), jg).mean()), "ra_art": float(ra(fk(Part), jg).mean()),
            "rot_deg": float(rot_deg(P[:, 3:6], G[:, 3:6]).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", required=True, help="S37_RUN_DIR=CNN_RUN_DIR (selected checkpoints)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="first valid run of each sequence only")
    a = ap.parse_args()
    global SMOKE
    SMOKE = a.smoke
    device = torch.device("cuda")
    res = {}
    for pair in a.pair:
        s37_dir, cnn_dir = pair.split("=", 1)
        s37, cfg, sel = load_run(s37_dir, device)
        cnn, cnn_cfg, csel = load_run(cnn_dir, device)
        root = Path(cfg["DATA"]["ROOT"])
        seqs = sequences_for_split(root, "val_core", splits_manifest(cfg))
        r = {"s37_ckpt": sel["ckpt"], "cnn_ckpt": csel["ckpt"]}
        for v in VARIANTS:
            rng = np.random.default_rng(0)
            rv = {s: run_seq(s37, cnn, cnn_cfg, cfg, root, d, s, device, rng, v) for s, d in seqs}
            n = sum(x["n"] for x in rv.values())
            rv["overall_ra"] = sum(x["ra"] * x["n"] for x in rv.values()) / n
            r[v] = rv
            print(Path(s37_dir).name, v, json.dumps(rv), flush=True)
        res[Path(s37_dir).name] = r
        Path(a.out).write_text(json.dumps(res, indent=1))
    means = {v: float(np.mean([r[v]["overall_ra"] for r in res.values()])) for v in VARIANTS}
    res["two_seed_mean_ra"] = means
    print("two-seed mean RA", json.dumps(means), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
