"""Permanent guard of the core training path: initial weights, loss, gradients and one optimiser step are bit-identical
to the reference of 2026-11-01 for nine configs (S45, C1, C1SF dz, S39, S43, two EventHands baselines, S45-TR t1d,
S45-AT t1ddz).

`tests/data/core_fingerprint_20261101.json` was computed from the core code BEFORE the 2026-11-01 cleanup that removed
the dead opt-in features of `model/model.py`, `semkine/train.py`, `semkine/dataset.py`, `semkine/xyz_mesh.py`, ...: it is
the numerical contract that cleanup had to keep, and the guard against any later edit that moves a bit of the S39 / S43 /
S45 training path. The fingerprints are recomputed by `tests/core_fingerprint_runner.py` in a fresh process (the bits of
float32 results depend on the thread configuration of the process, which other tests change); a deliberate numerical
change of the core must come with a new reference file.

Skipped when the hand dataset (`data/hand_data51`) is not on this machine.

    CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_core_fingerprint.py -q
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = Path(__file__).resolve().parent / "data" / "core_fingerprint_20261101.json"
RUNNER = Path(__file__).resolve().parent / "core_fingerprint_runner.py"
PREFIXES = ("init", "loss", "grads", "after", "params_with")
CONFIGS = ("S45", "C1", "C1SF_dz", "S39_dt_dz_l3", "S43_s43n_sp", "EH_abs_full51", "EH_track_delta51",
           "S45TR_t1d", "S45AT_t1ddz")

pytestmark = pytest.mark.skipif(not (ROOT / "data" / "hand_data51").exists(), reason="hand_data51 not available")


@pytest.fixture(scope="module")
def fresh(tmp_path_factory):
    out = tmp_path_factory.mktemp("fingerprint") / "now.json"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    env.pop("OMP_NUM_THREADS", None)               # the reference process ran with the default pool
    done = subprocess.run([sys.executable, str(RUNNER), str(out)], cwd=str(ROOT), env=env,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(out.read_text())


def test_reference_covers_the_nine_configs():
    ref = json.loads(REFERENCE.read_text())
    assert tuple(ref) == CONFIGS and all(v["status"] == "ok" for v in ref.values())


@pytest.mark.parametrize("name", CONFIGS)
def test_core_training_path_is_bit_identical_to_the_reference(fresh, name):
    ref = json.loads(REFERENCE.read_text())[name]
    now = fresh[name]
    assert now["status"] == "ok", now.get("error")
    keys = [k for k in ref if k.startswith(PREFIXES)] + ["n_params"]
    assert keys, "the reference has no fingerprint keys"
    bad = [k for k in keys if now.get(k) != ref[k]]
    assert not bad, f"{name}: {bad}"
