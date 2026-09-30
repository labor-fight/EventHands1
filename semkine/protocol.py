#!/usr/bin/env python3
"""S1 protocol reform: subject-disjoint splits and the statistics that motivate them.

The legacy split puts all 9 of one group of subjects in train and the single subject `zgz` in
val. That has two measured consequences:

* `val/zgz_local` fires 1490 events per 50 ms against 29 255 for `val/zgz_global` on the same
  subject and camera, yet supplies 1204 of 2590 val frames. The overall metric is a weighted
  average dominated by one anomalous acquisition.
* With one val subject there is no way to separate "this method generalises" from "this method
  fits zgz".

This module defines a subject-disjoint train/val/test partition. Recording variants (`_v2`,
`_v3`, `_v4`) are separate sessions of the same subject, so grouping by subject keeps every
augmented view of a session on one side of the partition, which is the requirement S5 states.

`zgz` is deliberately placed in test: it carries the low-event-rate sequence, which becomes a
named stress bucket instead of silently dominating the headline number.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

#: Subject-disjoint partition, registered before any model was trained on it.
#: 10 subjects -> 5 train / 2 val / 3 test. Test holds `zgz` (the low-event-rate stress
#: acquisition) plus two full 8-sequence subjects, so no single sequence can dominate.
SUBJECT_SPLITS: Dict[str, List[str]] = {
    "train": ["ch", "lfz", "lpc", "ly", "lyh"],
    "val": ["lr", "lyq"],
    "test": ["ycy", "ylf", "zgz"],
}

#: Sequences singled out for their acquisition conditions rather than their motion content.
#: Reported separately; never folded into a headline average.
STRESS_SEQUENCES = {
    "zgz_local": "low event rate (1490 ev/50ms, ~1/20 of same-subject global)",
}

#: Checkpoint selection runs on this subset of `val`, everything else on full `val`.
#: Rationale: scoring a 12-point checkpoint grid for 8 training arms on all 16 val sequences
#: costs about 3 GPU-hours of pure evaluation. The subset keeps both val subjects and both
#: categories, and it is registered here *before* any arm is trained so it cannot be reshaped
#: after seeing results. Reported numbers always come from full `val` or `test`.
VAL_CORE = [
    "lr_global", "lr_global_v3", "lr_local", "lr_local_v3",
    "lyq_global", "lyq_global_v3", "lyq_local", "lyq_local_v3",
]


def subject_of(seq: str) -> str:
    """`ch_global_v2` -> `ch`. Subject is everything before the first underscore."""
    return seq.split("_")[0]


def session_of(seq: str) -> str:
    """`ch_global_v2` -> `ch_global_v2`; a session is one recording.

    Kept as a separate function from :func:`subject_of` so the leakage check can state which
    granularity it is enforcing.
    """
    return seq


def category_of(seq: str) -> str:
    if "global" in seq:
        return "global"
    if "local" in seq:
        return "local"
    return "other"


@dataclass
class SequenceStats:
    seq: str
    legacy_split: str
    subject: str
    category: str
    n_ms: int
    n_valid_ms: int
    n_runs: int
    n_steps_50ms: int
    events_total: int
    events_per_50ms: float
    has_subms: bool
    lnes_drop_frac: float = float("nan")


def scan_sequences(root: Path, sample_windows: int = 200) -> List[SequenceStats]:
    """Measure every sequence in `root` (both legacy splits) without loading a model."""
    legacy = json.loads((root / "splits.json").read_text())
    out: List[SequenceStats] = []
    for legacy_split in ("train", "val"):
        for seq in legacy.get(legacy_split, {}).get("trials", []):
            base = root / legacy_split / seq
            offsets = np.load(str(base) + "_offsets.npy")
            aux = np.load(str(base) + "_aux.npz", allow_pickle=True)
            runs = np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)
            valid = np.asarray(aux["valid_ms"], dtype=bool)
            n_steps = int(sum(len(np.arange(a + 49, b, 50)) for a, b in runs))
            events = np.load(str(base) + "_events.npy", mmap_mode="r")

            # LNES drop = fraction of events the last-writer-wins surface overwrites.
            drops = []
            per_win = []
            if len(runs):
                a, b = runs[0]
                ends = list(range(a + 49, min(a + 49 + 50 * sample_windows, b), 50))
                for end in ends:
                    a0, a1 = int(offsets[end - 49]), int(offsets[end + 1])
                    e = np.asarray(events[a0:a1])
                    per_win.append(len(e))
                    if len(e):
                        key = (e[:, 1].astype(np.int64) * 240 + e[:, 0]) * 2 + np.clip(e[:, 2], 0, 1)
                        drops.append(1.0 - len(np.unique(key)) / len(e))
            out.append(
                SequenceStats(
                    seq=seq,
                    legacy_split=legacy_split,
                    subject=subject_of(seq),
                    category=category_of(seq),
                    n_ms=int(len(offsets) - 1),
                    n_valid_ms=int(valid.sum()),
                    n_runs=int(len(runs)),
                    n_steps_50ms=n_steps,
                    events_total=int(offsets[-1]),
                    events_per_50ms=float(np.mean(per_win)) if per_win else 0.0,
                    has_subms=(Path(str(base) + "_tsub.npy")).exists(),
                    lnes_drop_frac=float(np.mean(drops)) if drops else float("nan"),
                )
            )
    return out


@dataclass
class Manifest:
    """The `train`/`val`/`test` assignment plus everything needed to audit it."""

    splits: Dict[str, List[str]] = field(default_factory=dict)
    legacy_dir: Dict[str, str] = field(default_factory=dict)
    subjects: Dict[str, List[str]] = field(default_factory=dict)
    stats: Dict[str, dict] = field(default_factory=dict)

    def leakage_report(self) -> dict:
        """Subject and session overlap between every pair of splits. All must be empty."""
        subj = {k: {subject_of(s) for s in v} for k, v in self.splits.items()}
        sess = {k: {session_of(s) for s in v} for k, v in self.splits.items()}
        rep = {}
        keys = list(self.splits)
        for i, a in enumerate(keys):
            for b in keys[i + 1 :]:
                rep[f"{a}|{b}"] = {
                    "subject_overlap": sorted(subj[a] & subj[b]),
                    "sequence_overlap": sorted(sess[a] & sess[b]),
                }
        rep["clean"] = all(
            not v["subject_overlap"] and not v["sequence_overlap"]
            for k, v in rep.items() if k != "clean"
        )
        return rep


def build_manifest(root: Path, stats: Sequence[SequenceStats] | None = None) -> Manifest:
    if stats is None:
        stats = scan_sequences(root)
    by_subject: Dict[str, List[str]] = {}
    legacy_dir: Dict[str, str] = {}
    stat_map: Dict[str, dict] = {}
    for s in stats:
        by_subject.setdefault(s.subject, []).append(s.seq)
        legacy_dir[s.seq] = s.legacy_split
        stat_map[s.seq] = s.__dict__.copy()

    known = set(by_subject)
    assigned = {s for v in SUBJECT_SPLITS.values() for s in v}
    missing = sorted(known - assigned)
    extra = sorted(assigned - known)
    if missing or extra:
        raise ValueError(
            f"SUBJECT_SPLITS does not cover the data: unassigned subjects={missing}, "
            f"assigned but absent={extra}"
        )

    splits = {
        name: sorted(seq for subj in subjects for seq in by_subject[subj])
        for name, subjects in SUBJECT_SPLITS.items()
    }
    return Manifest(
        splits=splits,
        legacy_dir=legacy_dir,
        subjects={k: sorted(v) for k, v in SUBJECT_SPLITS.items()},
        stats=stat_map,
    )


def write_manifest(manifest: Manifest, out_dir: Path) -> Dict[str, Path]:
    """Write the manifest as a split json (loader-facing) plus CSV/markdown (human-facing).

    The split json mirrors the legacy `splits.json` schema so the existing loaders can read it,
    and adds `legacy_dir` because a sequence's files still live in its original directory.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    split_json = {
        name: {
            "trials": seqs,
            "subjects": manifest.subjects[name],
            "legacy_dir": {s: manifest.legacy_dir[s] for s in seqs},
        }
        for name, seqs in manifest.splits.items()
    }
    core = [s for s in manifest.splits["val"] if s in VAL_CORE]
    missing = sorted(set(VAL_CORE) - set(core))
    if missing:
        raise ValueError(f"VAL_CORE names sequences that are not in val: {missing}")
    split_json["val_core"] = {
        "trials": core,
        "subjects": manifest.subjects["val"],
        "legacy_dir": {s: manifest.legacy_dir[s] for s in core},
        "purpose": "checkpoint selection only; never used for a reported number",
    }
    split_json["_leakage"] = manifest.leakage_report()
    split_json["_stress_sequences"] = STRESS_SEQUENCES
    p_json = out_dir / "splits_semkine.json"
    p_json.write_text(json.dumps(split_json, indent=2))

    cols = ["split", "seq", "subject", "category", "legacy_dir", "n_ms", "n_valid_ms",
            "n_runs", "n_steps_50ms", "events_total", "events_per_50ms",
            "lnes_drop_frac", "has_subms", "is_stress"]
    lines = [",".join(cols)]
    for name, seqs in manifest.splits.items():
        for s in seqs:
            st = manifest.stats[s]
            lines.append(",".join(str(v) for v in [
                name, s, st["subject"], st["category"], st["legacy_split"], st["n_ms"],
                st["n_valid_ms"], st["n_runs"], st["n_steps_50ms"], st["events_total"],
                f"{st['events_per_50ms']:.1f}", f"{st['lnes_drop_frac']:.4f}",
                int(st["has_subms"]), int(s in STRESS_SEQUENCES),
            ]))
    p_csv = out_dir / "sequence_manifest.csv"
    p_csv.write_text("\n".join(lines) + "\n")
    return {"json": p_json, "csv": p_csv}


def load_manifest(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def summarise(manifest: Manifest) -> str:
    """Human-readable split summary for the stage report."""
    rows = []
    for name, seqs in manifest.splits.items():
        n_steps = sum(manifest.stats[s]["n_steps_50ms"] for s in seqs)
        ev = [manifest.stats[s]["events_per_50ms"] for s in seqs]
        rows.append(
            f"  {name:5s} subjects={','.join(manifest.subjects[name]):24s} "
            f"seqs={len(seqs):3d} steps@50ms={n_steps:6d} "
            f"ev/50ms min={min(ev):8.0f} med={np.median(ev):8.0f} max={max(ev):8.0f}"
        )
    leak = manifest.leakage_report()
    return "\n".join(rows) + f"\n  leakage clean: {leak['clean']}"
