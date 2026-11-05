#!/usr/bin/env python3
"""PFO2 amendment 1 (docs/EXPERIMENT_RECORDS.md#pfo2-amend1-20261103): is PFO2's entry (tools/pfo2/train.py) the same
procedure as PFO's tools/pfo/train4.py on F's recipe, given that this procedure is not bitwise reproducible?

Runs at step 40 of F's preflight config: PFO's F preflight, PFO2's f1rep preflight, and two diagnostic runs of the same
entry logic (diagnostics/amend1/dA, dB). d(X, Y) = ||theta_X - theta_Y|| / ||theta_Y|| over the trunk's trainable
parameters. PASS when
  (a) the first-batch data-stream hashes of both ranks are identical in all four runs,
  (b) the logged routing statistics (train_xyz_verts_direct / _prop / _hit / _valid at the logged steps) are identical,
  (c) max d(F, {f1rep, dA, dB}) <= 1.5 x max d among {f1rep, dA, dB} (F is no further from the PFO2-entry runs than
      those runs are from each other).
CPU only; receipt outputs/pfo2/20261103/receipts/equiv_f1rep.json (written once); exit non-zero on FAIL.
"""
from __future__ import annotations

import glob
from itertools import combinations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pfo2.common import OUT, now, pfo_run, read, require, save, sha  # noqa: E402

FACTOR = 1.5
STATS = ("train_xyz_verts_direct", "train_xyz_verts_prop", "train_xyz_hit", "train_xyz_valid")


def runs():
    pd = pfo_run("f1", 3407, "preflight")
    rd = OUT / "runs" / "pfo2_f1rep_s3407_preflight"
    dd = OUT / "diagnostics" / "amend1"
    return {"F": (pd, pd / f"{pd.name}-step=40.ckpt"), "f1rep": (rd, rd / f"{rd.name}-step=40.ckpt"),
            "dA": (dd / "dA", dd / "dA" / "det-step=40.ckpt"), "dB": (dd / "dB", dd / "dB" / "det-step=40.ckpt")}


def params(ck):
    import torch
    sd = torch.load(str(ck), map_location="cpu", weights_only=False)["state_dict"]
    return {k: v.double() for k, v in sd.items() if k.startswith("xyz.") and v.is_floating_point()}


def dist(a, b):
    num = sum(float(((a[k] - b[k]) ** 2).sum()) for k in b)
    den = sum(float((b[k] ** 2).sum()) for k in b)
    return (num / den) ** 0.5


def scalars(d):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    f = glob.glob(str(Path(d) / "logs" / "*" / "events.out.tfevents.*"))
    require(len(f) == 1, "one event file per run")
    ea = EventAccumulator(f[0])
    ea.Reload()
    tags = ea.Tags()["scalars"]
    return {t: [(e.step, e.value) for e in ea.Scalars(t)] for t in STATS if t in tags}, \
           {t: [(e.step, e.value) for e in ea.Scalars(t)] for t in ("train_loss",) if t in tags}


def main():
    dest = OUT / "receipts" / "equiv_f1rep.json"
    require(not dest.exists(), "checked once")
    R = runs()
    streams = {n: [read(d / f"pair_stream_rank{r}.json") for r in (0, 1)] for n, (d, _) in R.items()}
    a_ok = all(streams[n] == streams["F"] for n in R)
    sc = {n: scalars(d) for n, (d, _) in R.items()}
    b_ok = all(sc[n][0] == sc["F"][0] for n in R) and bool(sc["F"][0])
    P = {n: params(ck) for n, (_, ck) in R.items()}
    D = {f"{x}-{y}": dist(P[x], P[y]) for x, y in combinations(R, 2)}
    same = max(D[k] for k in ("f1rep-dA", "f1rep-dB", "dA-dB"))
    to_f = max(D[k] for k in ("F-f1rep", "F-dA", "F-dB"))
    c_ok = to_f <= FACTOR * same
    receipt = {"criterion": "streams equal, routing stats equal, max d(F, entry runs) <= 1.5 x max d(entry runs)",
               "streams_equal": a_ok, "routing_stats_equal": b_ok, "distances": D, "max_same_entry": same,
               "max_to_F": to_f, "ratio": to_f / same, "train_loss": {n: sc[n][1] for n in R},
               "checkpoints": {n: str(ck) for n, (_, ck) in R.items()},
               "checkpoint_sha256": {n: sha(ck) for n, (_, ck) in R.items()},
               "status": "PASS" if (a_ok and b_ok and c_ok) else "FAIL", "checked_at_utc": now()}
    save(dest, receipt, exclusive=True)
    print(receipt["status"], {k: round(v, 4) for k, v in D.items()}, "ratio", round(receipt["ratio"], 3), flush=True)
    require(receipt["status"] == "PASS", "procedure equivalence failed")


if __name__ == "__main__":
    main()
