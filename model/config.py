#!/usr/bin/env python3
"""YAML config loader for absolute-pose baselines."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path.resolve())
    # Resolve relative paths against repo root (parent of model/)
    repo = Path(__file__).resolve().parents[1]
    root = Path(cfg["DATA"]["ROOT"])
    if not root.is_absolute():
        cfg["DATA"]["ROOT"] = str((repo / root).resolve())
    mano = Path(cfg["MANO"]["NPZ"])
    if not mano.is_absolute():
        cfg["MANO"]["NPZ"] = str((repo / mano).resolve())
    for key in ("OUTPUT_DIR",):
        if key in cfg.get("TRAIN", {}):
            p = Path(cfg["TRAIN"][key])
            if not p.is_absolute():
                cfg["TRAIN"][key] = str((repo / p).resolve())
        if key in cfg.get("EVAL", {}):
            p = Path(cfg["EVAL"][key])
            if not p.is_absolute():
                cfg["EVAL"][key] = str((repo / p).resolve())
    return cfg


def pose_repr(cfg: Dict[str, Any]) -> str:
    return cfg["MODEL"]["POSE_REPR"]


def output_dim(cfg: Dict[str, Any]) -> int:
    return int(cfg["MODEL"]["OUTPUT_DIM"])
