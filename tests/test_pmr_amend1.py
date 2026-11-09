"""PMR engineering amendment 1 (docs/EXPERIMENT_RECORDS.md#pmr-amend1-20261109), CPU: the PMR stage breakdown of
tools/pmr/cost2.py times the PMR trunk's own stages for every variant (PFO's frozen breakdown names B2's submodules and
raises on the PMR trunk) and leaves the outputs unchanged; a failed attempt's directory is archived, a complete reading
never. Run with CUDA_VISIBLE_DEVICES="".
"""
from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tests.test_pmr import _inputs, _net  # noqa: E402


@pytest.mark.parametrize("variant", ("r", "noprop", "mlp"))
def test_pmr_breakdown_times_the_pmr_stages(variant, monkeypatch):
    from tools.pfo import cost as C
    from tools.pmr import cost2 as C2
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    net = _net(variant).eval()
    for name in ("head_j2", "head_h2", "fc2"):
        mod = getattr(net, name, None)
        if mod is not None:
            for p in mod.parameters():
                torch.nn.init.normal_(p, std=0.05)
    _, _, args = _inputs()
    with torch.no_grad():
        ref = net(*args)[0]
    with pytest.raises(AttributeError):
        C.stage_breakdown(None, types.SimpleNamespace(xyz=net, _fk=lambda *a: None), "routed", "fast", [], C.Stages(),
                          None, None, None)
    model = types.SimpleNamespace(xyz=net, _fk=lambda *a: None)
    outs = []
    with torch.no_grad():
        bd = C2.pmr_stage_breakdown(None, model, "routed", "fast", [args, args], C.Stages(),
                                    lambda it: outs.append(net(*it)[0]), None, None)
    assert torch.equal(outs[0], ref) and torch.equal(outs[1], ref)
    expect = {"step_total", "xyz_total", "node_inputs", "phi", "encode"}
    expect |= {"propagate", "readout"} if variant in ("r", "noprop") else {"readout"}
    if variant == "r":
        expect.add("gates")
    assert expect <= set(bd), (variant, sorted(bd))


def test_archive_moves_only_a_failed_attempt(tmp_path, monkeypatch):
    from tools.pmr import cost2 as C2
    monkeypatch.setattr(C2, "OUT", tmp_path)
    failed = tmp_path / "costs" / "x__fast__default"
    failed.mkdir(parents=True)
    (failed / "launch.json").write_text("{}")
    C2.archive_failed("x__fast__default")
    assert not failed.exists() and len(list((tmp_path / "costs" / "attempt_failed").iterdir())) == 1
    done = tmp_path / "costs" / "y__fast__default"
    done.mkdir(parents=True)
    (done / "report.json").write_text("{}")
    C2.archive_failed("y__fast__default")
    assert done.exists()
