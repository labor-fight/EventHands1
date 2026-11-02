#!/usr/bin/env python3
"""Freeze the S45-LOC protocol (docs/EXPERIMENT_RECORDS.md#s45loc-prereg-20261029); CPU only, no model.

Constants come from labels and event counts alone: the median root-joint camera depth p0_z = j0_z + t_z and the
median raw event count per W50 endpoint window of every recording of the domains. zgz_local's own label depth and
event count are used for the transplant / density targets (diagnostic interventions, disclosed in the
pre-registration). Writes protocol.json once; an existing file is never overwritten.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.frozen import accepts  # noqa: E402
from tools.s45loc import common as C  # noqa: E402

#: files whose bytes define a reading; run.py refuses to start when any of them changed
FROZEN_SOURCES = (
    "tools/s45loc/common.py", "tools/s45loc/run.py", "tools/s45loc/chain.py", "tools/s45loc/analyze.py",
    "tools/s45loc/freeze.py", "tools/xcl/evaluate.py", "semkine/xyz_mesh.py", "model/model.py",
    "semkine/eval_track.py", "semkine/xag_sources.py", "semkine/encoder.py", "semkine/dataset.py",
    "configs/c1sf_20261029/r2/c1sf_so3_s3407.yaml",
)


def recording_constants(cfg):
    from semkine import eval_track as ET
    from semkine.dataset import mano_root_joint, sequences_for_split, splits_manifest
    root = Path(cfg["DATA"]["ROOT"])
    found = {}
    for split in ("val_core", "train"):
        found.update(dict(sequences_for_split(root, split, splits_manifest(cfg))))
    out = {}
    for seq in C.TRAIN_BASE + C.ZGZ:
        _, offsets, aux, gt = ET.load_sequence(root, found[seq], seq)
        j0_z = float(mano_root_joint(np.asarray(aux["betas"], np.float32), cfg["MANO"]["NPZ"])[2])
        ends = [end for a, b in np.asarray(aux["valid_runs_ms"], np.int64).reshape(-1, 2)
                for end in range(int(a) + 49, int(b), 50)]
        p0z = np.array([j0_z + float(gt[e][2]) for e in ends])
        ev50 = np.array([int(offsets[e + 1] - offsets[e - 49]) for e in ends])
        out[seq] = {"n_endpoints": len(ends), "j0_z": j0_z, "p0z_median": float(np.median(p0z)),
                    "p0z_p10": float(np.percentile(p0z, 10)), "p0z_p90": float(np.percentile(p0z, 90)),
                    "ev50_median": float(np.median(ev50)), "ev50_p10": float(np.percentile(ev50, 10)),
                    "ev50_p90": float(np.percentile(ev50, 90))}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--smoke", action="store_true", help="write a SMOKE protocol (S45LOC_OUT must point elsewhere)")
    args = p.parse_args(argv)
    C.require(args.smoke == C.SMOKE, "--smoke iff S45LOC_OUT overrides the reading directory")
    from config import load_config
    C.require(C.sha(C.CHECKPOINT) == C.CHECKPOINT_SHA256, "S45 checkpoint changed")
    C.require(accepts(C.CONFIG, C.CONFIG_SHA256), "S45 config changed")
    cfg = load_config(str(C.CONFIG))
    rec = recording_constants(cfg)
    z_ref = float(np.median([rec[s]["p0z_median"] for s in C.TRAIN_LOCAL]))
    zl = rec["zgz_local"]
    #: event-side sigma* (SBA P0 visibility ledger, grid argmax median, +-0.01; Stage 0 cites the same file)
    sbar = C.read(C.SIGMA_STAR)["recordings"]
    sig = {s: float(sbar[s]["sigma_star_q0_median"]) for s in C.TRAIN_LOCAL + ("zgz_local",)}
    sig_ref = float(np.median([sig[s] for s in C.TRAIN_LOCAL]))
    constants = {
        "z_ref_train_local_median_m": z_ref,
        "k_ref": {"zgz_local": z_ref / zl["p0z_median"]},
        "zn_shift_ref": {"zgz_local": (z_ref - zl["p0z_median"]) / 0.1},
        "zn_shift_zgz": {s: (zl["p0z_median"] - rec[s]["p0z_median"]) / 0.1 for s in C.TRAIN_LOCAL},
        "p_sub": {s: min(1.0, zl["ev50_median"] / rec[s]["ev50_median"]) for s in C.TRAIN_LOCAL},
        "sigma_star": sig, "sigma_star_train_local_median": sig_ref,
        "sigma_star_source_sha256": C.sha(C.SIGMA_STAR),
        "sigma_factor_ref": {"zgz_local": sig_ref / sig["zgz_local"]},
        "sigma_factor_zgz": {s: sig["zgz_local"] / sig[s] for s in C.TRAIN_LOCAL},
        "recordings": rec,
    }
    section = C.OUT / "prereg_section.md"
    C.require(C.SMOKE or section.exists(), "write the pre-registration section before freezing")
    protocol = {
        "study": "S45-LOC", "status": "SMOKE" if C.SMOKE else "FROZEN", "frozen_utc": C.now(),
        "prereg": "docs/EXPERIMENT_RECORDS.md#s45loc-prereg-20261029",
        "prereg_section_sha256": C.sha(section) if section.exists() else None,
        "s45": {"baseline": str(C.BASELINE.relative_to(C.ROOT)), "checkpoint": str(C.CHECKPOINT.relative_to(C.ROOT)),
                "checkpoint_sha256": C.CHECKPOINT_SHA256, "config_sha256": C.CONFIG_SHA256},
        "physical_gpus": list(C.PHYSICAL_GPUS), "processes_per_gpu": 2, "seed": C.SEED,
        "tasks": {k: list(v) for k, v in C.TASKS.items()}, "stage1": list(C.STAGE1), "stage2": list(C.STAGE2),
        "perturb": C.PERTURB, "constants": constants,
        "sources_sha256": {name: C.sha(C.ROOT / name) for name in FROZEN_SOURCES},
        "no_training_or_selection": True,
    }
    C.save(C.PROTOCOL, protocol, exclusive=True)
    print(C.PROTOCOL)
    for key in ("z_ref_train_local_median_m", "k_ref", "zn_shift_ref", "zn_shift_zgz", "p_sub", "sigma_factor_ref",
                "sigma_factor_zgz"):
        print(key, constants[key])


if __name__ == "__main__":
    main()
