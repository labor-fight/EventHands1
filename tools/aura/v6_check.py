#!/usr/bin/env python3
"""AURA V6 (docs/EXPERIMENT_RECORDS.md#aura-prereg-20261024): on one physical GPU of 4-7, under the deterministic
evaluation policy, on 64 zgz packets (every 40th endpoint, E3 anchors):

  * `semkine.aura.associate("c1")` equals the frozen module's own association bit for bit on the GPU (gate);
  * every variant, the shuffle and the visibility run on CUDA under `torch.use_deterministic_algorithms(True)` and give
    identical results twice (gate);
  * the CPU-vs-CUDA disagreement rate of the C1 association (descriptive: float32 ties).

    CUDA_VISIBLE_DEVICES=4 CUDA_DEVICE_ORDER=PCI_BUS_ID python tools/aura/v6_check.py --out outputs/aura/c/v6.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.agg778 import common as C  # noqa: E402
from tools.aura import common as AC  # noqa: E402
from semkine import aura as AU  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = AC.require_gpu()
    AC.verify_protocol("C")
    C.set_policy()
    frozen, cfg = C.load_frozen("A0", dev)
    ds = C.Endpoints(C.zgz_seqs(), stride=40)
    items = [ds[i] for i in range(min(64, len(ds)))]
    res = {"packets": len(items), "c1_equal": True, "repeat_equal": True, "cpu_cuda_mismatch_events": 0, "events": 0}
    for i0 in range(0, len(items), 16):
        it = items[i0:i0 + 16]
        batch = C.make_batch(it, np.stack([x["gt_prev"] for x in it]), dev)
        cap = C.forward_capture(frozen, batch)
        d, mask, verts = cap["d"], cap["mask"], cap["verts"]
        intr = C.intrinsics(frozen, batch)
        faces, vf = frozen.mano.f.long(), frozen.xyz.vert_faces
        mine = AU.associate("c1", d, mask, verts, faces, vf, intr)
        res["c1_equal"] &= all(torch.equal(mine[k], cap["assoc"][k]) for k in mine)
        for rule in AU.RULES:
            x1 = AU.associate(rule, d, mask, verts, faces, vf, intr, c1=mine)
            x2 = AU.associate(rule, d, mask, verts, faces, vf, intr, c1=mine)
            res["repeat_equal"] &= all(torch.equal(x1[k], x2[k]) for k in x1)
        seeds = torch.arange(len(it), device=dev) * 7 + 1
        for part in (None, frozen.xyz.lbs_hard):
            s1 = AU.shuffle_assoc(mine, d, mask, verts, seeds, part)[0]
            s2 = AU.shuffle_assoc(mine, d, mask, verts, seeds, part)[0]
            res["repeat_equal"] &= all(torch.equal(s1[k], s2[k]) for k in s1)
        cpu = AU.associate("c1", d.cpu(), mask.cpu(), verts.cpu(), faces.cpu(), vf.cpu(), tuple(x.cpu() for x in intr))
        bad = torch.zeros_like(mask.cpu())
        for k in ("vid", "valid", "hit"):
            x, y = cpu[k], mine[k].cpu()
            bad |= (x != y).reshape(mask.shape[0], mask.shape[1], -1).any(-1)
        res["cpu_cuda_mismatch_events"] += int((bad & mask.cpu()).sum())
        res["events"] += int(mask.sum())
    res["cpu_cuda_mismatch_rate"] = res["cpu_cuda_mismatch_events"] / max(res["events"], 1)
    res["pass"] = bool(res["c1_equal"] and res["repeat_equal"])
    C.save_json(Path(a.out), res, exclusive=False)
    print("V6", res, flush=True)
    if not res["pass"]:
        raise SystemExit(1)
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
