"""Shared test fixtures.

* `OMP_NUM_THREADS=2` is set before torch is imported (the invocation the CPU tests document): several tests compare the
  raw bytes of float32 results across processes (`tests/test_dt_render.py::test_fastmano_equals_layer_cpu` hashes MANO
  outputs of this process against a freshly spawned reference), and those bits depend on the process's thread
  configuration. A test that changes the thread count in-process (a `torch.set_num_threads` fixture) would otherwise
  make that comparison order dependent.
* The S45-SM / S45-RT / S45-TR evaluation hooks rebind `semkine.eval_track.attach_xag_sources` for the rest of their
  process on purpose (their arms read no ground-truth source). Tests that exercise those hooks must not leak the
  rebinding into later tests, so it is restored after every test.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import pytest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _restore_eval_track_hooks():
    sys.path[:0] = [p for p in (str(ROOT), str(ROOT / "model")) if p not in sys.path]
    from semkine import eval_track as ET
    saved = ET.attach_xag_sources
    yield
    ET.attach_xag_sources = saved
