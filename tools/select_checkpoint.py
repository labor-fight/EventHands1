#!/usr/bin/env python3
r"""Pick a run's checkpoint by closed-loop recursive accuracy on the fixed step grid.

Not by `val_loss`. The S0 audit found `val_loss` to be *anti*-correlated with recursive tracking
error on this data, which is not surprising: the loss is a single-step teacher-forced quantity and
the metric that matters is what happens after a hundred steps of feeding predictions back in. Every
checkpoint on the pre-registered step grid is evaluated with the recursive protocol on `val_core`
and the best is reported, so selection uses the same quantity the paper will claim.

Selection runs on `val_core`. Under the canonical split that is the whole held-out subject, so the
step is chosen on the set it is reported on; the grid median and spread are printed beside the
selected value to bound how much of the gain is selection noise.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))

from config import load_config                          # noqa: E402
from mano_layer import ManoLayer                         # noqa: E402
from model import MNISTModel                            # noqa: E402

from semkine import eval_track as ET                    # noqa: E402
from semkine.dataset import sequences_for_split, splits_manifest  # noqa: E402

KEY = "mpjpe_ra_mm"


def grid_checkpoints(run_dir: Path):
    out = []
    for p in sorted(run_dir.glob("*step=*.ckpt")):
        m = re.search(r"step=(\d+)", p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", default=None, help="defaults to the config copied into the run")
    ap.add_argument("--split", default="val_core")
    ap.add_argument("--manifest", default=None,
                    help="override the config's split manifest; needed to score a run trained "
                         "under one subject split against another protocol's held-out subject")
    ap.add_argument("--step-ms", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delta-trust", type=float, default=1.0)
    a = ap.parse_args()

    run = Path(a.run_dir)
    cfg_path = a.config or str(next(iter(sorted(run.glob("*.yaml"))), ""))
    if not cfg_path:
        # The run records the config it was launched with, which is the one to reuse: a tracking
        # arm's yaml is generated per replicate and carries its own warm-start provenance.
        cfg_path = json.loads((run / "training_metadata.json").read_text())["config_path"]
    cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    root = Path(cfg["DATA"]["ROOT"])
    mani = Path(a.manifest) if a.manifest else splits_manifest(cfg)
    if mani and not mani.is_absolute() and not mani.exists():
        mani = root / mani
    seqs = sequences_for_split(root, a.split, mani)
    # Scoring a run on subjects it trained on would report a fit, not a generalisation number.
    trained = json.loads((run / "training_metadata.json").read_text()).get("train_sequences", [])
    leak = {s.split("_")[0] for s, _ in seqs} & {s.split("_")[0] for s in trained}
    assert not leak, f"{a.split} subjects {sorted(leak)} are in {run.name}'s training set"
    grid = grid_checkpoints(run)
    assert grid, f"no step-grid checkpoints in {run}"
    print(f"{run.name}: {len(grid)} checkpoints, {len(seqs)} sequences in {a.split}"
          f" [{mani.name if mani else 'splits_semkine.json'}]")

    rows = []
    for step, ckpt in grid:
        model = MNISTModel.load_from_checkpoint(str(ckpt), cfg=cfg,
                                               map_location=device).to(device).eval()
        rng = np.random.default_rng(a.seed)
        res = [ET.track_sequence(model, mano, cfg, root, d, s, a.step_ms, device, rng,
                                 1.0, a.delta_trust, None)
               for s, d in seqs]
        res = [r for r in res if r]
        n = sum(r["n_frames"] for r in res)
        row = {"step": step, "ckpt": str(ckpt), "n_frames": n,
               **{k: float(sum(r[k] * r["n_frames"] for r in res) / max(n, 1))
                  for k in ("mpjpe_ra_mm", "mpjpe_abs_mm", "mpvpe_ra_mm")},
               "per_seq": {r["seq"]: r[KEY] for r in res}}
        rows.append(row)
        print(f"  step={step:<6d} RA={row[KEY]:8.4f}  abs={row['mpjpe_abs_mm']:8.3f}", flush=True)
        del model
        torch.cuda.empty_cache()

    # A diverged checkpoint scores NaN, and NaN compares False against everything, so a bare
    # min() would return it whenever it sorts first and silently select a broken model.
    finite = [r for r in rows if math.isfinite(r[KEY])]
    if not finite:
        raise SystemExit(f"every checkpoint in {run.name} scored non-finite {KEY}")
    if len(finite) < len(rows):
        skipped = [r["step"] for r in rows if not math.isfinite(r[KEY])]
        print(f"  (skipped {len(skipped)} non-finite checkpoint(s): steps {skipped})")
    best = min(finite, key=lambda r: r[KEY])
    print(f"\nselected step={best['step']}  RA={best[KEY]:.4f}  {best['ckpt']}")
    # With one held-out subject there is no set left to select on that is not also the reported set,
    # so a selected step carries an optimistic bias. Quote the grid's spread against it: if selection
    # barely beats the grid median the bias is immaterial, and if it does not, it is selection noise.
    vals = sorted(r[KEY] for r in finite)
    med = vals[len(vals) // 2]
    print(f"grid median={med:.4f}  spread={vals[-1] - vals[0]:.4f}  "
          f"selection gain over median={med - best[KEY]:.4f} mm")
    tag = f"_{mani.stem}" if mani else ""
    out = run / f"selection_{a.split}_step{a.step_ms}{tag}.json"
    out.write_text(json.dumps({"run": run.name, "split": a.split, "step_ms": a.step_ms,
                               "manifest": mani.name if mani else "splits_semkine.json",
                               "grid_median": med, "selection_gain_over_median": med - best[KEY],
                               "selected": best, "grid": rows}, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
