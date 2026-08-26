#!/usr/bin/env python3
"""Build the canonical `splits_semkine.json`: the capture-time split, nine subjects train, zgz held out.

This is the project's only subject split. It restates `splits.json` in the manifest format the loader
reads, so `sequences_for_split` resolves `val_core` and `test` without falling back to the legacy
file, which defines neither.

History worth keeping, because it silently cost half the training data. A manifest written on
2026-08-22 re-partitioned the same 74 sequences into 5 train / 2 val / 3 test so that checkpoint
selection and the reported figure would sit on disjoint unseen subjects. Because the loader prefers
this filename over `splits.json`, every run after that date trained on five subjects while the
configs and the operator's intent still said nine, and nothing in the logs said so. §12 then measured
the price: 6.34 mm single-step bias on training subjects against 15.58 mm on unseen ones, making
subject count the dominant error term while four subjects sat unused in the evaluation splits.

What the nine-subject split costs in exchange, and what any run under it must therefore report:

* One held-out subject means `val`, `val_core` and `test` are the same two sequences, so a step
  chosen by recursive accuracy is chosen on the reported set. Quote the a-priori fixed step and the
  grid spread next to any selected step.
* `zgz_local` carries roughly a twentieth of the event rate of the same subject's global sequence, so
  it is half of the evaluation set and a documented stress case. Report per-sequence numbers; the
  pooled mean over these two mixes a normal and an extreme regime in equal parts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RETIRED = "_retired_splits_semkine_5v2v3.json"

# Carried over from the retired manifest: this sequence is half the held-out set under this split.
STRESS = {"zgz_local": "low event rate (1490 ev/50ms, ~1/20 of same-subject global)"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data1/lyq/code/mesh/EventHands/data/hand_data51")
    ap.add_argument("--keep-retired", action="store_true",
                    help="keep the 5/2/3 manifest under its retired name as an audit trail for "
                         "numbers already written into docs")
    a = ap.parse_args()

    root = Path(a.root)
    dst = root / "splits_semkine.json"
    legacy = json.loads((root / "splits.json").read_text())

    # The physical directory each sequence lives in. `splits.json` does not record it and the nine
    # training subjects span both directories, so recover the mapping from whichever manifest is
    # present, preferring one already in canonical form.
    src = dst if dst.exists() else root / RETIRED
    legacy_dir = {}
    for v in json.loads(src.read_text()).values():
        if isinstance(v, dict) and "legacy_dir" in v:
            legacy_dir.update(v["legacy_dir"])

    train = sorted(legacy["train"]["trials"])
    held = sorted(legacy["val"]["trials"])
    missing = [s for s in train + held if s not in legacy_dir]
    assert not missing, f"no legacy_dir for {missing}"

    def block(trials):
        return {"trials": trials,
                "subjects": sorted({s.split("_")[0] for s in trials}),
                "legacy_dir": {s: legacy_dir[s] for s in trials}}

    out = {"train": block(train), "val": block(held),
           "val_core": block(held), "test": block(held)}

    ts, hs = set(out["train"]["subjects"]), set(out["val"]["subjects"])
    out["_leakage"] = {"train|val": {"subject_overlap": sorted(ts & hs),
                                     "sequence_overlap": sorted(set(train) & set(held))},
                       "clean": not (ts & hs) and not (set(train) & set(held))}
    out["_stress_sequences"] = STRESS
    out["_provenance"] = {
        "source": "splits.json (capture-time split)",
        "held_out_subjects": sorted(hs),
        "note": "val/val_core/test are the same two sequences; quote the a-priori fixed step and "
                "the grid spread beside any selected step, and report per-sequence numbers",
    }

    if dst.exists():
        prev = json.loads(dst.read_text())["train"]["subjects"]
        if prev != out["train"]["subjects"]:
            print(f"replacing a {len(prev)}-subject manifest {prev}")
            if a.keep_retired:
                (root / RETIRED).write_text(dst.read_text())
                print(f"  audit trail kept at {RETIRED}")

    dst.write_text(json.dumps(out, indent=2))
    print(f"wrote {dst}")
    for k in ("train", "val", "val_core", "test"):
        b = out[k]
        print(f"  {k:<9} {len(b['trials']):>3} seqs  {len(b['subjects'])} subjects  {b['subjects']}")
    print(f"  leakage clean = {out['_leakage']['clean']}")
    print(f"  stress in held-out set: {sorted(set(STRESS) & set(held))}")


if __name__ == "__main__":
    main()
