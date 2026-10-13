#!/usr/bin/env python3
"""(Server side, not shipped.) Full-zgz checks of the deployment package against the official S39 evaluation:
events read from the raw AEDAT4 through eh346.sources.Aedat4File (dv-processing), official segments and initial states,
compared step by step with outputs/trk6/s39_repro (seed runs) / the seed-ensemble result, optionally another step.
    PYTHONPATH=/tmp/dvp python server_test_full.py --weights weights/s39_s3407.pt [--step-ms 10] [--cuda-graph]
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import torch

PKG = Path("/data1/lyq/code/mesh/EventHands1/deploy/EventHands_DAVIS346")
REPO = Path("/data1/lyq/code/mesh/EventHands1")
sys.path.insert(0, str(PKG))
from eh346.events import EventBuffer, SensorMap  # noqa: E402
from eh346.sources import Aedat4File  # noqa: E402
from eh346.tracker import S39Tracker  # noqa: E402

RAW = Path("/data1/lyq/data/hand_data/hand_data/val")
DATA = Path("/data1/lyq/code/mesh/EventHands/data/hand_data51")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--step-ms", type=int, default=50)
    ap.add_argument("--cuda-graph", action="store_true")
    ap.add_argument("--ref", default="", help="npz glob of the reference (default: s39_repro of the seed)")
    a = ap.parse_args()
    clip = np.load(PKG / "tests" / "zgz_clip.npz")
    ws = a.weights.split(",")
    trk = S39Tracker(ws, device=a.device, step_ms=a.step_ms, cuda_graph=a.cuda_graph)
    seed = trk.nets[0].meta["seed"]
    smap = SensorMap(640, 480, clip["K"])
    trk.set_camera(smap.K_model())
    betas = clip["betas"]
    ref = np.load(glob.glob(a.ref)[0] if a.ref else glob.glob(str(REPO / f"outputs/trk6/s39_repro/dt_dz_l3_s{seed}/*.npz"))[0])
    sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "model"))
    from semkine.eval_track import load_sequence
    from semkine.dataset import sequences_for_split
    legacy = dict(sequences_for_split(DATA, "val_core"))
    res = {}
    for seq in ("zgz_global", "zgz_local"):
        off = 0 if seq == "zgz_global" else 100_000
        src = Aedat4File(str(RAW / seq / "event" / f"{seq}_events.aedat4"))
        buf = EventBuffer(t0_us=0)
        t0 = time.perf_counter()
        while True:
            try:
                b = src.read()
            except StopIteration:
                break
            if b is None:
                continue
            t, x, y, p = b
            mx, my, mp, ok = smap.map(x, y, p)
            buf.push(t[ok], mx, my, mp)
        print(f"{seq}: read {buf.hi} events via dv-processing in {time.perf_counter() - t0:.1f} s")
        _, _, aux, pos = load_sequence(DATA, legacy[seq], seq)
        valid = np.asarray(aux["valid_runs_ms"], np.int64).reshape(-1, 2)
        seg = [k for k in range(len(clip["seg_a"])) if (clip["seq_name"][clip["seg_id"] == k][0] == seq)]
        out_end, out_state = [], []
        for k in seg:
            a0 = int(clip["seg_a"][k]) - off
            b0 = int(clip["seg_b"][k]) - off
            trk.reset(clip["seg_init"][k], a0, betas)
            for end in range(trk.first_end(), b0, trk.step_ms):
                r = trk.step(buf, end)
                out_end.append(end); out_state.append(r["state"])
        out_end, out_state = np.asarray(out_end), np.asarray(out_state)
        rend, rpred = ref[f"model|{seq}|end"], ref[f"model|{seq}|pred"]
        pos_idx = {int(e): i for i, e in enumerate(out_end)}
        sel = np.asarray([pos_idx[int(e)] for e in rend])          # the official 50 ms grid within ours
        st = out_state[sel]
        gt = np.stack([np.asarray(pos[int(e)], np.float32) for e in rend])
        bt = torch.as_tensor(betas, dtype=torch.float32, device=trk.device).view(1, -1)

        def ra(s):
            s_ = torch.as_tensor(s, device=trk.device)
            _, js = trk.mano.from_state(s_, bt.expand(len(s), -1))
            _, jg = trk.mano.from_state(torch.as_tensor(gt, device=trk.device), bt.expand(len(s), -1))
            return float((((js - js[:, :1]) - (jg - jg[:, :1])).norm(dim=-1).mean()) * 1000)
        res[seq] = (ra(st), ra(rpred), float(np.abs(st - rpred).max()), len(rend), len(out_end))
        print(f"{seq}: package RA {res[seq][0]:.4f} | reference RA {res[seq][1]:.4f} | max |diff| {res[seq][2]:.2e} "
              f"| scored {res[seq][3]} of {res[seq][4]} steps")
    n_g, n_l = res["zgz_global"][3], res["zgz_local"][3]
    print(f"overall RA package {(res['zgz_global'][0] * n_g + res['zgz_local'][0] * n_l) / (n_g + n_l):.4f} | "
          f"reference {(res['zgz_global'][1] * n_g + res['zgz_local'][1] * n_l) / (n_g + n_l):.4f}")


if __name__ == "__main__":
    main()
