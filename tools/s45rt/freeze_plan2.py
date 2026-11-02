#!/usr/bin/env python3
"""Freeze S45-RT plan 2 before T0P's preflight: outputs/s45rt/20261031/manifests/plan2.json (exclusive) with the SHA
of every plan-2 file and config and of the surface-patch table (docs/EXPERIMENT_RECORDS.md#s45rt-prereg-20261031).
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.s45rt.common import MANIFESTS, now, require, save, sha, verify_protocol  # noqa: E402
from tools.s45rt.plan2 import ARMS, CONFIGS, FILES, ROUTER_DIR, config_path  # noqa: E402


def main():
    verify_protocol()
    target = MANIFESTS / "plan2.json"
    require(not target.exists(), "plan 2 is frozen once")
    require(not ROUTER_DIR.exists(), "plan 2 must be frozen before any T0P output exists")
    from config import load_config
    from semkine import s45rt_router as RT
    from tools.s45rt.common import S45_CONFIG
    npz = load_config(str(S45_CONFIG))["MANO"]["NPZ"]
    patch, part, centres = RT.patch_table(npz)
    configs = [str(config_path(arm, phase).relative_to(ROOT)) for arm in ARMS for phase in ("preflight", "formal")]
    require(all((ROOT / c).is_file() for c in configs), "write the configs first (tools/s45rt/make_configs.py)")
    save(target, {"plan": 2, "frozen_utc": now(), "prereg": "docs/EXPERIMENT_RECORDS.md#s45rt-prereg-20261031",
                  "file_sha256": {name: sha(ROOT / name) for name in (*FILES, *configs)},
                  "patch_table_sha256": RT.table_sha256(npz), "n_patches": int(patch.max() + 1),
                  "patch_radius_m": RT.PATCH_RADIUS_M, "configs_dir": str(CONFIGS.relative_to(ROOT))}, exclusive=True)
    print(target)


if __name__ == "__main__":
    main()
