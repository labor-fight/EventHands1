#!/usr/bin/env python3
"""Isolated XC trainer: four geometry/time arms with identical causal-history inputs.

The canonical trainer and all default models stay untouched. This entry substitutes
only the study model, pins the data RNG and records the actual study provenance.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time
import datetime as dt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))

import torch
from semkine import train as canonical


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class StudyProvenanceCallback(canonical.pl.Callback):
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.started = None
        self.reported = -1

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
        self.started = time.monotonic()
        path = self.output_dir / "training_metadata.json"
        metadata = json.loads(path.read_text())
        cfg = pl_module.cfg
        model_cfg = cfg.get("MODEL", {})
        sources = ("semkine/xyz_time_study.py", "tools/xc/train.py",
                   "model/model.py", "semkine/xyz_mesh.py", "semkine/train.py",
                   "semkine/dataset.py", "semkine/events.py", "model/mano_layer.py")
        frozen = REPO / "outputs/xc/frozen_manifest.json"
        metadata.update({
            "selection_policy": "XC exact step6000 only; no intermediate selection",
            "data_generator_seed": int(cfg["SEED"]),
            "study": "XYZ-GEOMETRY-CAUSAL-TIME-20261012",
            "xc_study": True,
            "model_class": type(pl_module).__name__,
            "xyz_geometry": bool(model_cfg["XYZ_GEOM"]),
            "xyz_time_mode": model_cfg["XYZ_CAUSAL_TIME_MODE"],
            "motion_delta_dim": 51,
            "precision": str(cfg["TRAIN"]["PRECISION"]),
            "requested_data_unroll_pair": bool(cfg.get("DATA", {}).get("UNROLL_PAIR")),
            "effective_track_unroll_pair": bool(cfg.get("TRACK", {}).get("UNROLL_PAIR")),
            "history_protocol": "past lead prediction minus past lead conditioning on SO(3); main noisy prior retained; lead.target excluded",
            "deployment_history_protocol": "past incoming states and predictions only; reset each valid run; real 49/50ms intervals and 0/1ms query offsets",
            "source_sha256": {name: sha(REPO / name) for name in sources},
            "frozen_manifest_sha256": sha(frozen) if frozen.exists() else None,
        })
        path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if trainer.global_rank != 0 or step < 1 or step % 100 or step == self.reported:
            return
        self.reported = step
        elapsed = time.monotonic() - self.started
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        finite = bool(torch.isfinite(loss).all()) if torch.is_tensor(loss) else None
        value = float(loss.detach().float().cpu()) if finite else None
        progress = {"time": dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
                    "global_step": step, "max_steps": int(trainer.max_steps), "elapsed_s": elapsed,
                    "optimizer_steps_per_second": step / elapsed, "latest_batch_loss": value,
                    "latest_batch_loss_finite": finite}
        path = self.output_dir / "progress.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(progress, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
        print(f"XC progress {self.output_dir.name}: {step}/{trainer.max_steps}, "
              f"elapsed={elapsed:.1f}s, loss={value}, finite={finite}", flush=True)


def install():
    from semkine.xyz_time_study import TimeXYZModel
    original_loader = canonical.DataLoader
    original_trainer = canonical.pl.Trainer
    original_config = canonical.load_config

    def study_config(path):
        cfg = original_config(path)
        if not cfg.get("MODEL", {}).get("XYZ_CAUSAL_STUDY", False):
            raise RuntimeError("XC entry requires MODEL.XYZ_CAUSAL_STUDY=true")
        if cfg.get("MODEL", {}).get("XYZ_RESIDUAL_MODE", "legacy") != "legacy":
            raise RuntimeError("XC does not mix the ABCD residual intervention")
        cfg.setdefault("TRACK", {})["UNROLL_PAIR"] = True
        cfg["TRACK"]["UNROLL_P"] = 0.0
        return cfg

    def matched_loader(dataset, *args, **kwargs):
        if kwargs.get("shuffle", False):
            kwargs["generator"] = torch.Generator().manual_seed(int(dataset.seed))
        return original_loader(dataset, *args, **kwargs)

    class StudyTrainer(original_trainer):
        def __init__(self, *args, **kwargs):
            directory = kwargs["default_root_dir"]
            callbacks = list(kwargs.get("callbacks", []))
            callbacks += [canonical.XYZPairStreamCallback(directory),
                          StudyProvenanceCallback(directory)]
            kwargs["callbacks"] = callbacks
            kwargs["enable_progress_bar"] = False
            super().__init__(*args, **kwargs)

    canonical.MNISTModel = TimeXYZModel
    canonical.load_config = study_config
    canonical.DataLoader = matched_loader
    canonical.pl.Trainer = StudyTrainer


def main():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible not in ("4", "5", "6", "7"):
        raise RuntimeError(f"XC training requires one authorized physical GPU4-7, got {visible!r}")
    install()
    canonical.main()


if __name__ == "__main__":
    main()
