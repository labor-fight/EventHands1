"""PFO2 race-safe write-once files (docs/EXPERIMENT_RECORDS.md#pfo2-amend2-20261103): reading specs and evaluation
configs that several concurrent jobs of the same checkpoint create. The content goes to a per-process temporary file
and is linked into place only if the target does not exist (atomic); every caller then checks the target equals its
own content. tools/pfo2/run.py's writers shared one temporary name, so two jobs starting together could fail.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from tools.pfo2.common import OUT, read, require


def write_once_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(text)
        try:
            os.link(tmp, path)
        except FileExistsError:
            pass
        finally:
            tmp.unlink()
    require(path.read_text() == text, "write-once file differs: " + str(path))
    return path


def spec(tag, sp):
    path = OUT / "specs" / f"{tag}.json"
    write_once_text(path, json.dumps(sp, indent=2, allow_nan=False) + "\n")
    require(read(path) == sp, "spec differs: " + str(path))
    return path


def config(path, cfg):
    import yaml
    return write_once_text(path, yaml.safe_dump(cfg, sort_keys=False))


__all__ = ["config", "spec", "write_once_text"]
