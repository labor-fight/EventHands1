#!/usr/bin/env python3
"""ABL rollout training, amendment 1 (docs/EXPERIMENT_RECORDS.md#abl-amend1-20261111): tools/pmr/train_rol6.py's patches
unchanged (imported, not run), plus the rollout model class of the ray-association arm (semkine/abl_rol.py) when the config's
router block is "none"; every other arm gets semkine/tpd_rol.py's ROLModel as before. Run with NCCL_P2P_DISABLE=1.

    python tools/pmr/train_rol7.py --config configs/pmr_20261109/pmr_abgeo_rol5_s<seed>[_preflight].yaml
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "model"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.pmr import train_rol6 as W  # noqa: E402  (its module-level patches)
from semkine import abl_rol as AR  # noqa: E402  (captures ROLModel's factory first)
from semkine import tpd_rol as TR  # noqa: E402

_native = TR.rol_model_class


def rol_model_class():
    import yaml
    p = yaml.safe_load(Path(sys.argv[2]).read_text())["PFO"]
    return AR.ray_rol_model_class() if p["ROUTER_KIND"] == "none" else _native()


TR.rol_model_class = rol_model_class
W.T.CODE = W.T.CODE + ("tools/pmr/train_rol7.py", "semkine/abl_rol.py")

if __name__ == "__main__":
    W.T.main()
