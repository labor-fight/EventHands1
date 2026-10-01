#!/usr/bin/env python3
"""x1001 effective temporal receptive field of a trained arm, by 5 ms occlusion.

For every 4th protocol step of the development set (excluding the first step of each segment), the
step is re-run ten times, each time with one 5 ms slice of its 50 ms window removed. The state fed
in is the arm's own previous output from its recorded closed loop (`evalx_val_core_selected.npz`;
no GT), so the measurement is the normal-inference sensitivity of the output to each slice:
root-rotation change (deg) and root-aligned joint displacement (mm). span90 = the youngest span that
holds 90% of the summed sensitivity. Also reports the structural 3-hop temporal span of the graph
for event-graph arms.

    python tools/x1001/rf_occlusion.py --run-dir RUNS/x1001_e8_s3407
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))
sys.path.insert(0, str(REPO / "tools" / "x1001"))
from config import load_config                                        # noqa: E402
from mano_layer import ManoLayer                                      # noqa: E402
from model import MNISTModel                                          # noqa: E402
from semkine import eval_track as ET                                  # noqa: E402
from semkine import events as EV                                      # noqa: E402
import evalx as EX                                                    # noqa: E402

DATA = Path("/data1/lyq/code/mesh/EventHands/data/hand_data51")
STEP, NB = 50, 10


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--every", type=int, default=4)
    a = ap.parse_args()
    run = Path(a.run_dir)
    cfg = load_config(next(iter(sorted(run.glob("*.yaml")))))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck, step, _ = EX.find_ckpt(run, "selected")
    m = MNISTModel.load_from_checkpoint(str(ck), cfg=cfg, map_location=dev).to(dev).eval()
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(dev).eval()
    z = np.load(run / "evalx_val_core_selected.npz")
    raw = bool(getattr(m, "encoder_name", ""))
    ev_ch = EV.event_channels(cfg)
    out = {"run": run.name, "ckpt": str(ck), "step": step}
    for seq in ("zgz_global", "zgz_local"):
        events, offsets, aux, pos51 = ET.load_sequence(DATA, "val", seq)
        tsub = np.load(DATA / "val" / f"{seq}_tsub.npy", mmap_mode="r")
        betas = torch.tensor(aux["betas"], dtype=torch.float32, device=dev).view(1, -1)
        K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=dev).view(1, 3, 3)
        if hasattr(m, "set_hand_context"):
            m.set_hand_context(betas, K)
        pred, end, rr = z[f"model|{seq}|pred"], z[f"model|{seq}|end"], z[f"model|{seq}|run"]
        idx = [i for i in range(1, len(end)) if rr[i] == rr[i - 1]][::a.every]
        rot = np.zeros((len(idx), NB))
        mm = np.zeros((len(idx), NB))
        for n, i in enumerate(idx):
            prev = torch.from_numpy(pred[i - 1]).view(1, -1).to(dev)
            ev = ET._window_events(events, offsets, tsub, int(end[i]), STEP)
            outs = []
            for b in [-1] + list(range(NB)):
                e = ev
                if b >= 0 and len(ev):
                    ms = np.floor(ev[:, 3] * 1000.0 + 5e-4)
                    e = ev[(ms // (STEP // NB)) != b]
                if raw:
                    o = m.forward_packet(ET.make_eval_packet(e, prev, betas, K, STEP, dev))
                else:
                    if len(e):
                        msr = np.floor(e[:, 3] * 1000.0 + 5e-4).astype(np.float32)
                        img = EV.splat_event_image(e[:, 1].astype(np.uint8), e[:, 2].astype(np.uint8),
                                                   e[:, 4].astype(np.uint8), msr, STEP, 180, 240, ev_ch)
                    else:
                        img = np.zeros((180, 240, 2 * len(ev_ch)), np.float32)
                    o = m(torch.from_numpy(img).unsqueeze(0).to(dev), prev)
                outs.append(o.float().cpu().numpy()[0])
            outs = np.stack(outs)
            j, _ = EX.mano_fk(mano, outs.astype(np.float32), betas, dev)
            j = j - j[:, :1]
            mm[n] = np.linalg.norm(j[1:] - j[0], axis=-1).mean(-1) * 1000
            rot[n] = EX.rot_err_deg(outs[1:], np.repeat(outs[:1], NB, 0))
        def span90(p):
            w = p[::-1] / max(p.sum(), 1e-12)
            return float((np.searchsorted(np.cumsum(w), 0.9) + 1) * (STEP // NB))
        prof_r, prof_m = rot.mean(0), mm.mean(0)
        out[seq] = {"n": len(idx), "root_deg_by_slice": prof_r.round(4).tolist(), "joint_mm_by_slice": prof_m.round(4).tolist(),
                    "root_span90_ms": span90(prof_r), "joint_span90_ms": span90(prof_m),
                    "root_deg_total": float(prof_r.sum()), "joint_mm_total": float(prof_m.sum())}
        print(seq, json.dumps(out[seq]), flush=True)
    (run / "rf_occlusion.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
