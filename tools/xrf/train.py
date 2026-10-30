#!/usr/bin/env python3
"""XRF trainer: `tools/xdl/train.py` with the long-window dataset (main packet only) and `semkine.xrf.XRFModel`.
The XDL trainer is not edited; its module-level names are rebound here before `main()` runs. Debug rounds use
`--preflight --max-steps N`; formal runs need `--frozen-manifest` and exact step 6000."""
from __future__ import annotations

from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'model'))

import tools.xdl.train as T  # noqa: E402
import semkine.xyz_recovery_study as S  # noqa: E402
from semkine.xrf import XRFModel  # noqa: E402
from tools.xkf.data import build_train_dataset  # noqa: E402

T.build_train_dataset = build_train_dataset
T.SOURCE_FILES = tuple(T.SOURCE_FILES) + ('semkine/xkf.py', 'semkine/xkf_direct.py', 'semkine/xrf.py',
                                          'tools/xkf/data.py', 'tools/xrf/train.py', 'model/backbones.py',
                                          'semkine/lie.py')
S.RecoveryXYZModel = XRFModel

if __name__ == '__main__':
    T.main()
