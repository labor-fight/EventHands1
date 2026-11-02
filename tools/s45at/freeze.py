#!/usr/bin/env python3
"""Freeze S45-AT (docs/EXPERIMENT_RECORDS.md#s45at-prereg-20261031): outputs/s45at/20261031/protocol.json with the
pre-registration text and the SHA of every source, config and dependency; refuses to overwrite."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.s45at.common import (ARMS, CONFIGS, DEPENDENCIES, OUT, PROTOCOL, ROOT, SOURCES, TAG, now, read, require,  # noqa: E402
                                save, sha)


def main():
    require(not PROTOCOL.exists(), "S45-AT protocol already frozen")
    text = (ROOT / "docs/EXPERIMENT_RECORDS.md").read_text()
    begin, end = f"<!-- EXPERIMENT_RECORD:{TAG}:BEGIN -->", f"<!-- EXPERIMENT_RECORD:{TAG}:END -->"
    require(text.count(begin) == 1 and text.count(end) == 1, "write the pre-registration section first")
    OUT.mkdir(parents=True, exist_ok=True)
    section = OUT / "prereg_section.md"
    with section.open("x") as stream:
        stream.write(text[text.index(begin):text.index(end) + len(end)])
    configs = sorted(str(p.relative_to(ROOT)) for p in CONFIGS.glob("*.yaml"))
    require(len(configs) == 2 * len(ARMS), "write the configs first (tools/s45at/make_configs.py)")
    names = (*SOURCES, "tools/s45at/freeze.py", *DEPENDENCIES, *configs)
    save(PROTOCOL, {
        "study": "S45AT_20261031", "status": "FROZEN", "frozen_utc": now(),
        "prereg": "docs/EXPERIMENT_RECORDS.md#s45at-prereg-20261031", "prereg_section_sha256": sha(section),
        "arms": ARMS, "references": {
            "s45_formal_receipt": read(ROOT / "outputs/c1sf/20261029/receipts/formal_so3.json")["checkpoint_sha256_by_step"],
            "t1d_formal_receipt": read(ROOT / "outputs/s45tr/20261030/receipts/formal_t1d.json")["checkpoint_sha256_by_step"]},
        "file_sha256": {name: sha(ROOT / name) for name in names}, "single_seed_descriptive": True}, exclusive=True)
    print(PROTOCOL)


if __name__ == "__main__":
    main()
