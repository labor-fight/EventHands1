#!/usr/bin/env python3
"""S45-CT step-0 corruption audit (one GPU among 2-7): on 512 fixed training packets of T1d's training dataset
(training mode: S45's augmentation and prior-noise contract, XAG sources), run frozen T1d with the arm's route
corruption installed and capture every routed packet. Registered checks:
  sigma16  >= 99 % of the replaced faces at template geodesic distance in [8, 24] mm, median in [13, 19] mm,
           barycentrics 1/3, and the same corruption when recomputed from the captured inputs
  shuffle  every packet's routed (face, barycentrics) multiset unchanged, >= 90 % of the routed events reassigned,
           and the same corruption when recomputed
Writes outputs/s45ct/20261030/acceptance/audit_<mode>.json (exclusive); exit code 0 iff PASS.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.s45ct.common import OUT, PHYSICAL_GPUS, S45TR, T1D_CONFIG, T1D_RUN, now, read, require, save, sha, verify_protocol  # noqa: E402

N_PACKETS, BATCH = 512, 64


def main(argv=None):
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--mode", required=True, choices=("sigma16", "shuffle"))
    a = p.parse_args(argv)
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in tuple(map(str, PHYSICAL_GPUS)), "one GPU among 2-7")
    from tools.xcl import evaluate as XE
    XE.configure_environment()
    XE.set_policy()
    verify_protocol()
    target = OUT / "acceptance" / f"audit_{a.mode}.json"
    require(not target.exists(), "audit is read once: " + str(target))
    import torch
    from config import load_config
    from model import MNISTModel
    from semkine import s45ct_route as H
    from semkine.dataset import build_dataset
    from semkine.events import collate_packets
    cfg = load_config(str(T1D_CONFIG))
    checkpoint = T1D_RUN / "s45tr_t1d_s3407-step=6000.ckpt"
    require(sha(checkpoint) == read(S45TR / "receipts/formal_t1d.json")["checkpoint_sha256_by_step"]["6000"], "T1d changed")
    H.install(a.mode, cfg["MANO"]["NPZ"])
    dev = torch.device("cuda")
    model = MNISTModel.load_from_checkpoint(str(checkpoint), cfg=cfg, map_location=dev, strict=True).to(dev).eval()
    components = np.load(cfg["MANO"]["NPZ"])["hands_components"].astype(np.float32)
    ds = build_dataset(cfg, "train", components, train=True)
    pick = np.sort(np.random.default_rng(3407).choice(len(ds), N_PACKETS, replace=False))
    H.capture(True)
    with torch.no_grad():
        for start in range(0, N_PACKETS, BATCH):
            batch = collate_packets([ds[int(i)] for i in pick[start:start + BATCH]]).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model.forward_packet(batch)
    cap = H.captured()
    require(len(cap) == N_PACKETS // BATCH, "every audited batch must be routed exactly once")
    with np.load(cfg["MANO"]["NPZ"], allow_pickle=False) as z:
        dist = H.face_geodesic(z["v_template"], z["f"])
    same, n_events, packets, stats = True, 0, 0, {}
    disp, moved, multiset_ok = [], [], []
    for c in cap:
        e = c["eligible"]
        nf, nb = H.corrupt(a.mode, c["face"], c["bary"], e, c["rows"], cfg["MANO"]["NPZ"])
        same = same and torch.equal(nf, c["new_face"]) and torch.equal(nb, c["new_bary"])
        n_events += int(e.sum())
        if a.mode == "sigma16":
            disp.append(dist[c["face"][e].numpy(), c["new_face"][e].numpy()])
            same = same and bool(torch.allclose(c["new_bary"][e], torch.full_like(c["new_bary"][e], 1 / 3)))
        else:
            for b in range(e.shape[0]):
                if not bool(e[b].any()):
                    continue
                packets += 1
                f0, f1 = c["face"][b][e[b]], c["new_face"][b][e[b]]
                b0, b1 = c["bary"][b][e[b]], c["new_bary"][b][e[b]]
                multiset_ok.append(sorted(zip(f0.tolist(), map(tuple, b0.tolist()))) ==
                                   sorted(zip(f1.tolist(), map(tuple, b1.tolist()))))
                moved.append(((f0 != f1) | (b0 != b1).any(-1)).numpy())
    if a.mode == "sigma16":
        d = np.concatenate(disp) * 1000
        inside, median = float(np.mean((d >= 8) & (d <= 24))), float(np.median(d))
        stats = {"replaced_events": int(len(d)), "share_in_8_24_mm": inside, "median_mm": median,
                 "p05_mm": float(np.percentile(d, 5)), "p95_mm": float(np.percentile(d, 95))}
        ok = inside >= 0.99 and 13 <= median <= 19
    else:
        share = float(np.concatenate(moved).mean())
        stats = {"packets": packets, "multiset_unchanged_share": float(np.mean(multiset_ok)), "reassigned_share": share}
        ok = all(multiset_ok) and share >= 0.90
    result = {"mode": a.mode, "status": "PASS" if ok and same else "FAIL", "recomputed_identically": bool(same),
              "routed_events": n_events, "packets_audited": N_PACKETS, **stats, "audited_utc": now(),
              "hook_sha256": sha(ROOT / "semkine/s45ct_route.py")}
    save(target, result, exclusive=True)
    print(result, flush=True)
    require(result["status"] == "PASS", "corruption audit failed")


if __name__ == "__main__":
    main()
