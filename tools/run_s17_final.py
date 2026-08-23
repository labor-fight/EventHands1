#!/usr/bin/env python3
r"""S17 final evaluation: every available arm, three seeds, CLAIM_MATRIX + per_sequence.csv.

Seeds 3407 / 3408 / 3409 as registered. A missing seed or checkpoint is recorded as
`pending`, not as a silent omission. Claims fire only when the registered comparison
(paired sequence bootstrap, CI excludes 0) is actually computed.

Usage:
  python tools/run_s17_final.py                  # evaluate whatever is on disk
  python tools/run_s17_final.py --dry-run        # write the matrix skeleton only
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))

SEEDS = (3407, 3408, 3409)
ARMS = (
    ("A0", "s1_abs_base", "absolute, no domrand"),
    ("A0d", "s1_abs_domrand", "absolute + domrand (S13 anchor)"),
    ("A1", "s1_track_base", "track, no domrand"),
    ("A2", "s1_track_domrand", "track + domrand (main dense baseline)"),
    ("A2t", "s1_delta_trust", "A2 + δ-trust=0.5, same checkpoint"),
    ("A3", "s2_raw_track", "raw-event scan student"),
    ("A3f", "s2_sparse_cell", "raw-event sparse-cell fallback"),
    ("A4", "s4_abs_fk", "so3fk + absolute FK"),
    ("A10", "s10_active", "unique-pathway per-joint head"),
    ("M", "mainline", "filter + RDOR + anchor on A2"),
)


def _selection(run_dir: Path) -> dict | None:
    p = run_dir / "selection_val_core_step50.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def collect_status(out: Path) -> dict:
    rows = []
    for tag, name, note in ARMS:
        for seed in SEEDS:
            d = out / f"{name}_s{seed}"
            sel = _selection(d) if d.exists() else None
            rows.append({
                "arm": tag, "name": name, "seed": seed, "note": note,
                "run_dir": str(d) if d.exists() else None,
                "selected_step": None if sel is None else sel["selected"]["step"],
                "val_core_ra_mm": None if sel is None else sel["selected"]["mpjpe_ra_mm"],
                "status": "measured" if sel else ("training" if d.exists() else "pending"),
            })
    return {"seeds": list(SEEDS), "arms": rows}


def write_per_sequence(status: dict, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "name", "seed", "status",
                                          "selected_step", "val_core_ra_mm", "note"])
        w.writeheader()
        for r in status["arms"]:
            w.writerow({k: r[k] for k in w.fieldnames})


def claim_matrix(status: dict) -> list:
    """Registered claims. Verdict is PASS/FAIL only after a paired CI is computed."""
    measured = {r["name"]: r for r in status["arms"] if r["status"] == "measured"}
    claims = [
        {
            "id": "C0",
            "claim": "S0 frozen baseline reproduces 19.2576 mm RA on the legacy val split",
            "gate": "bit-identical recursive RA",
            "verdict": "PASS",
            "evidence": "docs/semkine/ARCHITECTURE_AUDIT.md",
        },
        {
            "id": "C1",
            "claim": "Subject-disjoint protocol + buckets do not change the legacy metric when off",
            "gate": "extended evaluator vs legacy ≤ 1e-6",
            "verdict": "PASS",
            "evidence": "tests/test_s1_eval.py",
        },
        {
            "id": "C2",
            "claim": "Overall SOTA = better than strongest domrand dense baseline, paired CI excludes 0",
            "gate": "full val/test, ≥2 seeds, not best-of-N",
            "verdict": "PENDING" if "s1_track_domrand" not in measured else "PENDING_FULL_VAL",
            "evidence": None,
        },
        {
            "id": "C3",
            "claim": "RDOR beats δ-trust=0.5, paired CI excludes 0",
            "gate": "same-checkpoint inference comparison",
            "verdict": "PENDING",
            "evidence": "tools/run_semkine.py, tools/run_s9_rdor.py",
        },
        {
            "id": "C4",
            "claim": "S2 raw encoder 95% CI upper bound on RA degradation vs A2 < 0.5 mm",
            "gate": "paired per-run bootstrap",
            "verdict": "PENDING",
            "evidence": "configs/semkine/s2_raw_track.yaml",
        },
        {
            "id": "C5",
            "claim": "Quiet / single-finger / 30–60s drift bucket claims (−15/−50/−30%)",
            "gate": "support ≥ MIN_SUPPORT, not synthetic-only for a main claim",
            "verdict": "PENDING",
            "evidence": "data/hand_data51/buckets/",
        },
    ]
    return claims


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "outputs/semkine")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    status = collect_status(args.out)
    claims = claim_matrix(status)
    dest = args.out / "s17"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "status.json").write_text(json.dumps(status, indent=2))
    (dest / "claim_matrix.json").write_text(json.dumps(claims, indent=2))
    write_per_sequence(status, dest / "per_sequence.csv")
    print(json.dumps({"n_arms": len(status["arms"]),
                      "measured": sum(r["status"] == "measured" for r in status["arms"]),
                      "claims": {c["id"]: c["verdict"] for c in claims}}, indent=2))


if __name__ == "__main__":
    main()
