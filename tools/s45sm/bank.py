"""S45-SM own-loop state bank of one arm (as S45-TR N4's tools/s45tr/n4_bank.py, for an S45-SM arm).

  bank run --arm A --shard i --shards n   the arm's W50 closed loop (step 6000, its routing via the latency path, bit-identical
                                          routes) over the i-th share of the 72 training sequences
  bank verify --arm A                     every sequence present and well formed; the nine training local recordings equal
                                          the arm's own `tl_e1` readout predictions bit for bit

The loop is the evaluator's (`semkine.eval_track.track_sequence`, 50 ms packets on the endpoint grid, initial-state noise from
a fresh `default_rng(0)` per recording, as N4's bank). Each `banks/<arm>/<seq>.npz` holds `end_ms (K,)` and `state (K, 51)`: the
state the loop held after the update at each endpoint.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .paths import OUT, PHYSICAL_GPUS, now, read, require, save, sha
from .protocol import verify_protocol
from .registry import bank_dir, registry
from .routing import register_for_net, resolve


def sequences(cfg):
    from semkine.dataset import sequences_for_split, splits_manifest
    seqs = sorted(sequences_for_split(Path(cfg["DATA"]["ROOT"]), "train", splits_manifest(cfg)))
    require(len(seqs) == 72, "training split changed")
    return seqs


def run(arm, shard, shards):
    import torch
    from config import load_config
    from mano_layer import ManoLayer
    from model import MNISTModel
    from semkine import eval_track as ET
    from semkine import s45sm_fast as FAST
    from tools.xcl import evaluate as XE
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in tuple(map(str, PHYSICAL_GPUS)), "one GPU among 2-7")
    XE.configure_environment()
    XE.set_policy()
    verify_protocol()
    register_for_net(arm)
    checkpoint, config = resolve(arm, 6000)
    cfg = load_config(str(config))
    s = cfg["S45SM"]
    FAST.install(s["ROUTER_KIND"], s["ROUTER_CHECKPOINT"], s["ASSOC"], float(s["ONHAND_TAU"]))
    ET.attach_xag_sources = lambda batch, *args, **kwargs: batch
    device = torch.device("cuda")
    model = MNISTModel.load_from_checkpoint(str(checkpoint), cfg=cfg, map_location=device, strict=True).to(device).eval()
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    root = Path(cfg["DATA"]["ROOT"])
    target_dir = bank_dir(arm)
    target_dir.mkdir(parents=True, exist_ok=True)
    native = model.forward_packet
    for seq, directory in sequences(cfg)[shard::shards]:
        target = target_dir / f"{seq}.npz"
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


def verify(arm):
    from config import load_config
    verify_protocol()
    checkpoint, config = resolve(arm, 6000)
    cfg = load_config(str(config))
    names = [s for s, _ in sequences(cfg)]
    ref = np.load(OUT / "readouts" / arm / "tl_e1" / "states.npz", allow_pickle=False)
    base = sorted({k.split("/")[0] for k in ref.files})
    require(len(base) == 9, "nine training local recordings in the arm's tl_e1")
    checks = {}
    for seq in names:
        with np.load(bank_dir(arm) / f"{seq}.npz", allow_pickle=False) as z:
            e, s = z["end_ms"], z["state"]
        require(s.shape == (len(e), 51) and np.isfinite(s).all() and np.all(np.diff(e) > 0), "malformed " + seq)
        if seq in base:
            checks[seq] = bool(np.array_equal(e, ref[f"{seq}/end_ms"]) and np.array_equal(s, ref[f"{seq}/prediction"]))
    require(len(checks) == 9 and all(checks.values()), f"bank differs from the arm's own tl_e1 loop: {checks}")
    save(bank_dir(arm).parent / f"{arm}_receipt.json", {
        "status": "PASS", "arm": arm, "checkpoint_sha256": sha(checkpoint), "sequences": len(names),
        "local_equal_tl_e1": checks, "verified_at_utc": now(),
        "files_sha256": {f"{s}.npz": sha(bank_dir(arm) / f"{s}.npz") for s in names}}, exclusive=True)
    print("PASS", arm, len(names))
