#!/usr/bin/env python3
"""Build the S1 subject-disjoint split manifest and the bucket manifest.

Writes into the dataset root so the loaders find it next to `splits.json`:

  splits_semkine.json     train/val/test by subject, with the leakage report
  sequence_manifest.csv   per-sequence statistics behind the split decision
  buckets/<split>.json    per-step motion-activity buckets (S5, needed by the evaluator)

Usage:
  python tools/make_semkine_manifest.py                # splits + buckets
  python tools/make_semkine_manifest.py --splits-only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from semkine import buckets as BK      # noqa: E402
from semkine import protocol as PR     # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--step-ms", type=int, default=50)
    ap.add_argument("--splits-only", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    root = args.root
    if root is None:
        sys.path.insert(0, str(REPO / "model"))
        from config import load_config
        root = Path(load_config(REPO / "configs" / "eventhands_track_render51.yaml")["DATA"]["ROOT"])

    print(f"scanning {root} ...", flush=True)
    stats = PR.scan_sequences(root)
    manifest = PR.build_manifest(root, stats)
    paths = PR.write_manifest(manifest, root)
    print(PR.summarise(manifest))
    for k, v in paths.items():
        print(f"  wrote {k}: {v}")

    if args.splits_only:
        return

    mano = REPO / "assets" / "mano_right.npz"
    out_dir = root / "buckets"
    out_dir.mkdir(parents=True, exist_ok=True)
    totals = {}
    # Thresholds are quantiles of the training split and are then frozen, so a val/test bucket
    # boundary is never fitted to the split it is used to report on.
    train_th = None
    for split in ("train", "val", "test"):
        seqs = [(s, manifest.legacy_dir[s]) for s in manifest.splits[split]]
        table = BK.build_bucket_manifest(root, seqs, str(mano), step_ms=args.step_ms,
                                         thresholds=train_th, device=args.device)
        if split == "train":
            train_th = {k: v for k, v in table["thresholds"].items()}
        p = out_dir / f"{split}_step{args.step_ms}.json"
        p.write_text(json.dumps(table, indent=2))
        totals[split] = table["totals"]
        print(f"  wrote buckets: {p}")
        for name, n in sorted(table["totals"].items(), key=lambda kv: -kv[1]):
            print(f"    {name:22s} {n:7d}")
    (out_dir / f"summary_step{args.step_ms}.json").write_text(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
