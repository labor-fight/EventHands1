#!/usr/bin/env python3
"""Preserve all strict-runtime recursive predictions from one frozen weight."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

# This import configures the process environment before torch is imported.
from evaluate import ROOT, set_policy
import numpy as np
import torch
from config import load_config
from mano_layer import ManoLayer
from model import MNISTModel
from semkine import eval_track as ET
from semkine.dataset import sequences_for_split, splits_manifest


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError("Preserve earlier traces")
    args.output_dir.mkdir(parents=True)
    set_policy()
    cfg = load_config(args.config)
    device = torch.device("cuda")
    model = MNISTModel.load_from_checkpoint(args.checkpoint, cfg=cfg, map_location=device,
                                          strict=True).to(device).eval()
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    rng, arrays, results = np.random.default_rng(0), {}, []
    for sequence, directory in sequences_for_split(Path(cfg["DATA"]["ROOT"]), "val_core", splits_manifest(cfg)):
        incoming, predictions = [], []
        def capture(stage, previous, prediction):
            if stage == "step":
                incoming.append(previous.detach().cpu().numpy()[0].copy())
                predictions.append(prediction.detach().cpu().numpy()[0].copy())
            return prediction
        result = ET.track_sequence(model, mano, cfg, Path(cfg["DATA"]["ROOT"]), directory,
                                   sequence, 50, device, rng, 1.0, 1.0, None, state_hook=capture)
        expected = {"zgz_global": 1386, "zgz_local": 1204}[sequence]
        assert result["n_frames"] == len(predictions) == expected
        arrays[sequence + "/incoming"] = np.stack(incoming)
        arrays[sequence + "/prediction"] = np.stack(predictions)
        assert np.isfinite(arrays[sequence + "/prediction"]).all()
        results.append(result)
    assert sum(r["n_frames"] for r in results) == 2590
    np.savez_compressed(args.output_dir / "states.npz", **arrays)
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    receipt = {"status": "COMPLETE", "n_endpoints": 2590, "per_sequence": results,
               "checkpoint_sha256": sha(args.checkpoint), "config_sha256": sha(args.config),
               "states_sha256": sha(args.output_dir / "states.npz"),
               "source_sha256": sha(__file__), "runtime_policy": "XYZ_C1_REPRO_20261018_STRICT"}
    (args.output_dir / "report.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
