#!/usr/bin/env python3
"""XVM checkpoint evaluation on zgz with the XCL harness (E1 own closed loop / E3 GT previous endpoint).

Same loop as tools/xvm/omega_r.py R0_base, which reproduces the XCL A0FOmega-o E1 numbers bit for bit: strict
deterministic policy, `ET.track_sequence` with one rng seeded 0 over val_core in order, XCL `score`. Writes the per-step
predictions too, so a second fresh run can be compared array by array.

    python tools/xvm/eval_ckpt.py --config configs/xvm/xvm_c1sign_s3407.yaml --ckpt <...>/last.ckpt --mode e1 \
        --out outputs/xvm/s3/c1sign/e1_repeat1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "model"), str(REPO / "tools")]
from xvm import common as C  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", choices=("e1", "e3", "gt_pose", "gt_transl"), required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = C.require_gpu()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; preserve it")
    from tools.xcl.evaluate import set_policy, score, endpoint_grid, select_previous
    set_policy()
    from config import load_config
    from mano_layer import ManoLayer
    from model import MNISTModel
    from semkine import eval_track as ET
    from semkine.dataset import sequences_for_split, splits_manifest
    cfg = load_config(a.config)
    model = MNISTModel.load_from_checkpoint(a.ckpt, cfg=cfg, map_location=dev, strict=True).to(dev).eval()
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(dev).eval()
    seqs = sequences_for_split(Path(cfg["DATA"]["ROOT"]), "val_core", splits_manifest(cfg))
    native = model.forward_packet
    rng = np.random.default_rng(0)
    res = {"config": a.config, "config_sha256": C.sha(a.config), "ckpt": a.ckpt, "ckpt_sha256": C.sha(a.ckpt),
           "mode": a.mode, "per_seq": {}}
    arrays = {}
    t0 = time.time()
    for s, d in seqs:
        _, _, aux, gt = ET.load_sequence(Path(cfg["DATA"]["ROOT"]), d, s)
        rows = endpoint_grid(aux)
        preds, targets = [], []

        def capture(packet):
            i = len(preds)
            rid, end, previous_ms, first = rows[i]
            raw = packet.prev_state.detach().cpu().numpy()[0].copy()
            prev = select_previous(a.mode if a.mode != "e1" else "e1", raw, np.asarray(gt[previous_ms], np.float32))
            packet = replace(packet, prev_state=torch.from_numpy(prev).view(1, -1).to(dev))
            pred = native(packet)
            preds.append(pred.detach().cpu().numpy()[0].copy())
            targets.append(np.asarray(gt[end], np.float32).copy())
            return pred

        model.forward_packet = capture
        try:
            r = ET.track_sequence(model, mano, cfg, Path(cfg["DATA"]["ROOT"]), d, s, 50, dev, rng, 1.0, 1.0, None)
        finally:
            model.forward_packet = native
        assert r["n_frames"] == len(preds) == len(rows)
        beta = torch.tensor(np.array(aux["betas"]), device=dev, dtype=torch.float32).view(1, -1)
        P, T = np.stack(preds), np.stack(targets)
        m = score(mano, P, T, beta, dev)
        res["per_seq"][s] = {k: float(v.mean()) for k, v in m.items()}
        res["per_seq"][s]["n"] = int(len(P))
        arrays[f"{s}/prediction"] = P
        print(a.mode, s, json.dumps(res["per_seq"][s]), f"{time.time() - t0:.0f} s", flush=True)
    n = sum(v["n"] for v in res["per_seq"].values())
    res["overall"] = {k: sum(res["per_seq"][s][k] * res["per_seq"][s]["n"] for s in res["per_seq"]) / n
                      for k in ("mpjpe_ra_mm", "mpvpe_ra_mm", "wrist_mm")}
    res["wall_s"] = time.time() - t0
    out.mkdir(parents=True)
    np.savez_compressed(out / "states.npz", **arrays)
    res["states_sha256"] = C.sha(out / "states.npz")
    (out / "report.json").write_text(json.dumps(res, indent=1))
    C.receipt(out, {"eval": a.mode}, scripts=(__file__,))


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
