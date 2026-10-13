#!/usr/bin/env python3
"""Check this installation against the training server's official S39 evaluation.

tests/zgz_clip.npz holds raw sensor events of the zgz test subject (640 x 480 training camera), the protocol's initial
states, the ground truth, and the server's official S39 outputs for every step (three seeds). This script runs the
deployment code path (SensorMap -> EventBuffer -> LNES -> network -> filter, the same calls run_davis346.py makes) on
those events and compares.

On the training server (same GPU type and torch) the states match bit for bit. On another GPU / CPU / torch version,
floating-point differences (cuDNN kernels) are expected and the closed loop amplifies them a little; the check is then
on the accuracy: |RA-MPJPE(this machine) - RA-MPJPE(server)| <= 0.15 mm.
    python verify_install.py [--weights weights/s39_s3408.pt] [--device cuda|cpu]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "weights" / "s39_s3408.pt"))
    ap.add_argument("--device", default="")
    ap.add_argument("--data", default=str(ROOT / "tests" / "zgz_clip.npz"))
    ap.add_argument("--cuda-graph", action="store_true")
    a = ap.parse_args()
    import torch
    from eh346.events import EventBuffer, SensorMap
    from eh346.tracker import S39Tracker
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    z = np.load(a.data)
    trk = S39Tracker([a.weights], device=device, cuda_graph=a.cuda_graph)
    seed = str(trk.nets[0].meta.get("seed", ""))
    ref_key = f"ref_state_s{seed}"
    if ref_key not in z.files:
        sys.exit(f"no server reference for weights of seed {seed!r} in {a.data}")
    smap = SensorMap(int(z["width"]), int(z["height"]), z["K"])
    trk.set_camera(smap.K_model())
    betas = z["betas"]
    buf = EventBuffer(t0_us=0)
    x, y, p, ok = smap.map(z["x"], z["y"], z["p"])
    buf.push(z["t_us"].astype(np.int64)[ok], x, y, p)
    print(f"device {device} | torch {torch.__version__} | weights {Path(a.weights).name} (seed {seed}) | "
          f"{int(ok.sum())} events, {len(z['end_ms'])} steps in {len(z['seg_a'])} segments")
    out, comp = [], []
    for k, (sa, sb) in enumerate(zip(z["seg_a"], z["seg_b"])):
        trk.reset(z["seg_init"][k], int(sa), betas)
        for end in z["end_ms"][z["seg_id"] == k]:
            r = trk.step(buf, int(end))
            out.append(r["state"])
            comp.append(r["compute_ms"])
    out = np.asarray(out)
    ref = z[ref_key]
    gt = z["gt"]

    def ra_mpjpe(states, seq_mask):
        s = torch.as_tensor(states[seq_mask], device=trk.device)
        g = torch.as_tensor(gt[seq_mask], device=trk.device)
        b = torch.as_tensor(betas, dtype=torch.float32, device=trk.device).view(1, -1).expand(len(s), -1)
        _, js = trk.mano.from_state(s, b)
        _, jg = trk.mano.from_state(g, b)
        e = (js - js[:, :1]) - (jg - jg[:, :1])
        return float(e.norm(dim=-1).mean() * 1000)

    ok_all = True
    for name in np.unique(z["seq_name"]):
        m = z["seq_name"] == name
        r_pkg, r_ref = ra_mpjpe(out, m), ra_mpjpe(ref, m)
        d = float(np.abs(out[m] - ref[m]).max())
        good = abs(r_pkg - r_ref) <= 0.15
        ok_all &= good
        print(f"  {name:11s} steps {int(m.sum()):5d}  RA-MPJPE this machine {r_pkg:.3f} mm | server {r_ref:.3f} mm | "
              f"max |state diff| {d:.2e} {'(bit-identical)' if d == 0 else ''}  {'OK' if good else 'MISMATCH'}")
    c = np.asarray(comp[5:])
    print(f"  compute per step (LNES + render + network + filter, batch 1): p50 {np.median(c):.2f} ms, "
          f"p99 {np.percentile(c, 99):.2f} ms")
    print("PASS" if ok_all else "FAIL: the outputs differ from the server beyond floating-point noise")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
