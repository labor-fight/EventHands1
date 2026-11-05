"""PFO2 amendment 2 (docs/EXPERIMENT_RECORDS.md#pfo2-amend2-20261103): race-safe write-once files, CPU tests."""
from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]


def _writer(args):
    path, text = args
    from tools.pfo2.specs import write_once_text
    write_once_text(path, text)
    return Path(path).read_text()


def test_concurrent_writers_agree(tmp_path):
    target = tmp_path / "spec.json"
    text = '{"kind": "routed"}\n' * 200
    with mp.get_context("fork").Pool(16) as pool:
        out = pool.map(_writer, [(str(target), text)] * 64)
    assert all(o == text for o in out)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["spec.json"]       # no temporary file left


def test_different_content_is_refused(tmp_path):
    import pytest
    from tools.pfo2.specs import write_once_text
    write_once_text(tmp_path / "a", "x")
    with pytest.raises(RuntimeError):
        write_once_text(tmp_path / "a", "y")


def test_amend2_jobs():
    from tools.pfo2 import jobs_a2 as JB
    jobs = JB.amend2()
    by = {j["key"]: j for j in jobs}
    assert by["a2:decide:mp"]["deps"] == ["a1:eval:f1mp:3407:e1@6000", "a2:eval:f1mp:3408:e1@6000"]
    assert "tools.pfo2.run2" in by["a2:eval:fl1:3407:e3@6000"]["command"]
    assert by["p0d2:run:fl1:3409"]["deps"] == ["a2:eval:fl1:3409:e1@6000", "a2:eval:fl1:3409:tl_e1@6000"]
    assert "tools/pfo2/train_det.py" in by["a2:train:fgs:3408"]["command"][3]
