#!/usr/bin/env python3
"""S45-TR N4 state bank: the frozen S45's own W50 closed loop on every training sequence (one GPU among 2-7 per shard).

  n4_bank.py run --shard i --shards n   loop over the i-th share of the 72 training sequences (sorted by name)
  n4_bank.py verify                     every sequence present; the 18 base recordings equal S45-LOC's own-loop
                                        predictions (`outputs/s45loc/20261029/tasks/c1_train_e1`) bit for bit

The loop is the evaluator's (`semkine.eval_track.track_sequence`, 50 ms packets on the endpoint grid, initial-state
noise `sample_init_noise` from a fresh `default_rng(0)` per recording, as S45-LOC's training-domain loops). Each
`<bank>/<seq>.npz` holds `end_ms (K,)` and `state (K, 51)`: the state the loop held after the update at each endpoint.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.s45tr.common import (N4_BANK, S45_CHECKPOINT, S45_CHECKPOINT_SHA256, S45_CONFIG, now, read, require,  # noqa: E402
                                save, sha, verify_protocol)

S45LOC_E1 = ROOT / "outputs/s45loc/20261029/tasks/c1_train_e1/states.npz"


def sequences(cfg):
    from semkine.dataset import sequences_for_split, splits_manifest
    seqs = sorted(sequences_for_split(Path(cfg["DATA"]["ROOT"]), "train", splits_manifest(cfg)))
    require(len(seqs) == 72, "training split changed")
    return seqs


def run(shard, shards):
    import torch
    from dataclasses import replace  # noqa: F401
    from config import load_config
    from mano_layer import ManoLayer
    from model import MNISTModel
    from semkine import eval_track as ET
    from tools.xcl import evaluate as XE
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in tuple(map(str, range(2, 8))), "one GPU among 2-7")
    XE.configure_environment()
    XE.set_policy()
    verify_protocol()
    require(sha(S45_CHECKPOINT) == S45_CHECKPOINT_SHA256, "S45 checkpoint changed")
    device = torch.device("cuda")
    cfg = load_config(str(S45_CONFIG))
    model = MNISTModel.load_from_checkpoint(str(S45_CHECKPOINT), cfg=cfg, map_location=device, strict=True).to(device).eval()
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    root = Path(cfg["DATA"]["ROOT"])
    N4_BANK.mkdir(parents=True, exist_ok=True)
    native = model.forward_packet
    for seq, directory in sequences(cfg)[shard::shards]:
        target = N4_BANK / f"{seq}.npz"
        require(not target.exists(), "bank entry exists: " + str(target))
        _, _, aux, _ = ET.load_sequence(root, directory, seq)
        rows = XE.endpoint_grid(aux)
        states = []

        def capture(packet):
            require(len(states) < len(rows), "too many packets")
            pred = native(packet)
            value = pred.detach().cpu().numpy()[0].copy()
            require(value.shape == (51,) and np.isfinite(value).all(), "nonfinite loop state")
            states.append(value)
            return pred
        model.forward_packet = capture
        try:
            with torch.no_grad():
                result = ET.track_sequence(model, mano, cfg, root, directory, seq, 50, device,
                                           np.random.default_rng(0), 1.0, 1.0, None)
        finally:
            model.forward_packet = native
        require(result is not None and len(states) == len(rows) == result["n_frames"], "incomplete loop: " + seq)
        np.savez_compressed(target, end_ms=np.array([r[1] for r in rows], np.int64), state=np.stack(states))
        print(seq, len(rows), flush=True)


def verify():
    from config import load_config
    cfg = load_config(str(S45_CONFIG))
    names = [s for s, _ in sequences(cfg)]
    ref = np.load(S45LOC_E1, allow_pickle=False)
    base = sorted({k.split("/")[0] for k in ref.files})
    require(len(base) == 18, "S45-LOC base loop recordings")
    checks = {}
    for seq in names:
        with np.load(N4_BANK / f"{seq}.npz", allow_pickle=False) as z:
            e, s = z["end_ms"], z["state"]
        require(s.shape == (len(e), 51) and np.isfinite(s).all() and np.all(np.diff(e) > 0), "malformed " + seq)
        if seq in base:
            checks[seq] = bool(np.array_equal(e, ref[f"{seq}/end_ms"]) and np.array_equal(s, ref[f"{seq}/prediction"]))
    require(len(checks) == 18 and all(checks.values()), f"bank differs from S45-LOC own loop: {checks}")
    save(N4_BANK.parent / "n4_bank_receipt.json", {
        "status": "PASS", "sequences": len(names), "base_equal_s45loc": checks, "verified_at_utc": now(),
        "files_sha256": {s: sha(N4_BANK / f"{s}.npz") for s in names}}, exclusive=True)
    print("PASS", len(names))


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("action", choices=("run", "verify"))
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    a = p.parse_args()
    if a.action == "run":
        run(a.shard, a.shards)
    else:
        verify()


if __name__ == "__main__":
    main()
