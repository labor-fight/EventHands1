#!/usr/bin/env python3
"""Build motion buckets from the fixed, read-only splits_semkine.json.

    python tools/make_bucket_manifest.py --step-ms 50 --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from semkine import buckets as BK  # noqa: E402
from semkine.dataset import sequences_for_split  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--step-ms", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    if args.step_ms <= 0:
        ap.error("--step-ms must be positive")

    root = args.root
    if root is None:
        sys.path.insert(0, str(REPO / "model"))
        from config import load_config
        root = Path(load_config(REPO / "configs/semkine/s1_track_domrand.yaml")["DATA"]["ROOT"])

    manifest = root / "splits_semkine.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"fixed split manifest required: {manifest}")
    splits = {split: sequences_for_split(root, split, manifest)
              for split in ("train", "val", "test")}
    out_dir = args.out_dir or root / "buckets"
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = None
    totals = {}
    for split, sequences in splits.items():
        table = BK.build_bucket_manifest(
            root, sequences, str(REPO / "assets/mano_right.npz"),
            step_ms=args.step_ms, thresholds=thresholds, device=args.device,
        )
        if split == "train":
            thresholds = dict(table["thresholds"])
        path = out_dir / f"{split}_step{args.step_ms}.json"
        path.write_text(json.dumps(table, indent=2))
        totals[split] = table["totals"]
        print(f"wrote buckets: {path}")
    (out_dir / f"summary_step{args.step_ms}.json").write_text(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
