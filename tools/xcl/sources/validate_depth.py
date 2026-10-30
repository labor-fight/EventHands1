#!/usr/bin/env python3
"""Publish all-74 complete XCL depth sidecar SHA/stat identity after CPU build."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))
from semkine.dataset import sequences_for_split
from semkine.xag_sources import depth_manifest, verify_depth_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO / "data" / "hand_data51")
    parser.add_argument("--source-root", type=Path, default=REPO / "outputs" / "xag" / "sources")
    parser.add_argument("--output", type=Path, default=REPO / "outputs" / "xcl" / "20261019" / "depth_manifest.json")
    parser.add_argument("--check", action="store_true", help="fast check an already SHA-verified manifest")
    args = parser.parse_args()
    sequences = [seq for seq, _ in sequences_for_split(args.data_root, "train", None) +
                 sequences_for_split(args.data_root, "val", None)]
    if len(sequences) != 74:
        raise ValueError(f"fixed train+zgz split must have 74 sequences, got {len(sequences)}")
    if args.check:
        verify_depth_manifest(json.loads(args.output.read_text()), sequences)
    else:
        manifest = depth_manifest(args.source_root, sequences)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_suffix(args.output.suffix + ".tmp")
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        tmp.replace(args.output)
    print(json.dumps({"complete": True, "sequences": len(sequences), "manifest": str(args.output)}))


if __name__ == "__main__":
    main()
