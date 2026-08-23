#!/usr/bin/env python3
"""Convert MANO_RIGHT.pkl (chumpy) into a clean numpy npz for training envs."""
from __future__ import annotations

import argparse
import hashlib
import pickle
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _to_dense(x) -> np.ndarray:
    if hasattr(x, "todense"):
        return np.asarray(x.todense())
    if hasattr(x, "toarray"):
        return np.asarray(x.toarray())
    return np.asarray(x)


def convert(pkl_path: Path, out_path: Path) -> None:
    with pkl_path.open("rb") as f:
        data = pickle.load(f, encoding="latin1")

    J_regressor = _to_dense(data["J_regressor"]).astype(np.float64)
    payload = {
        "v_template": np.asarray(data["v_template"], dtype=np.float64),
        "shapedirs": np.asarray(data["shapedirs"], dtype=np.float64),
        "posedirs": np.asarray(data["posedirs"], dtype=np.float64),
        "J_regressor": J_regressor,
        "weights": np.asarray(data["weights"], dtype=np.float64),
        "f": np.asarray(data["f"], dtype=np.int32),
        "kintree_table": np.asarray(data["kintree_table"], dtype=np.int64),
        "hands_mean": np.asarray(data["hands_mean"], dtype=np.float64).reshape(-1),
        "hands_components": np.asarray(data["hands_components"], dtype=np.float64),
        "J": np.asarray(data["J"], dtype=np.float64),
        "source_pkl": np.array(str(pkl_path.resolve())),
        "source_sha256": np.array(_sha256(pkl_path)),
        # Right-hand fingertip vertex ids used by smplx / WiLoR OpenPose-21 mapping.
        "fingertip_vertex_ids": np.asarray([744, 320, 443, 554, 671], dtype=np.int64),
        # MANO-16 + tips -> OpenPose-21 reorder used by WiLoR / hand_data joints_cam_m.
        "mano16_tips_to_openpose21": np.asarray(
            [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20],
            dtype=np.int64,
        ),
    }

    assert payload["v_template"].shape == (778, 3)
    assert payload["shapedirs"].shape == (778, 3, 10)
    assert payload["posedirs"].shape == (778, 3, 135)
    assert payload["J_regressor"].shape == (16, 778)
    assert payload["weights"].shape == (778, 16)
    assert payload["hands_mean"].shape == (45,)
    assert payload["hands_components"].shape == (45, 45)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    print(f"wrote {out_path}")
    print(f"source_sha256={payload['source_sha256']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pkl",
        type=Path,
        default=Path("/data1/lyq/data/hand_data/WiLoR/mano_data/MANO_RIGHT.pkl"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "assets" / "mano_right.npz",
    )
    args = ap.parse_args()
    convert(args.pkl, args.out)


if __name__ == "__main__":
    main()
