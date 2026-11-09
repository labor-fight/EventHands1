"""TPD cost block wrapper (tools/tpd/cost.py): the causal form's per-packet k inside PFO's cost harness. CPU only."""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pathlib import Path  # noqa: E402
import sys  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from semkine import eval_track as ET  # noqa: E402
from semkine import tpd_causal as TC  # noqa: E402
from tools.pfo import nets as NM  # noqa: E402
from tools.tpd import cost as TCOST  # noqa: E402


class _Net:
    def __init__(self, seen):
        class M:
            def forward_packet(self, batch):
                seen.append(int(TC._STATE["k"][0]))
                return batch.prev_state
        self.model = M()


@pytest.fixture
def hooked(monkeypatch):
    seen = []
    for mod, name in ((ET, "_window_events"), (ET, "make_eval_packet")):
        monkeypatch.setattr(mod, name, getattr(mod, name))
    monkeypatch.setattr(NM, "load", lambda *a, **k: _Net(seen))
    TCOST._KS.clear()
    TCOST.causal_cost_hooks()
    yield seen
    TC.set_k(None)


def _recording(per_ms=100, ms=400):
    offsets = np.arange(ms + 2, dtype=np.int64) * per_ms
    rng = np.random.default_rng(0)
    events = np.stack([rng.integers(0, 346, offsets[-1]), rng.integers(0, 260, offsets[-1]),
                       rng.integers(0, 2, offsets[-1])], 1).astype(np.float32)
    return events, offsets


def _packet(events, offsets, end):
    """The harness's own builder (tools/pfo/cost.py `build`, non-S43 branch)."""
    prev = torch.zeros(1, 51)
    ev5 = ET._window_events(events, offsets, None, end, 50)
    return ET.make_eval_packet(ev5, prev, torch.zeros(1, 10), torch.eye(3)[None], 50, torch.device("cpu"))


def test_each_packet_carries_its_own_k_and_sets_it_before_the_trunk(hooked):
    events, offsets = _recording()
    ends = (49, 99, 149, 349)
    staged = [_packet(events, offsets, e) for e in ends]          # row form: all packets built first
    expect = [int(TC.k_from_prev(TC.n_prev_events(offsets, e))) for e in ends]
    assert expect == [1, 3, 3, 3]                                 # 0 events before the first packet; 5000 after
    assert [b.causal_k for b in staged] == expect and TCOST._KS == expect
    net = NM.load("any")
    for b in reversed(staged):                                    # forward order independent of build order
        net.model.forward_packet(b)
    assert hooked == expect[::-1]


def test_rejects_other_windows_and_packets_without_a_cut(hooked):
    events, offsets = _recording()
    with pytest.raises(RuntimeError):
        ET._window_events(events, offsets, None, 199, 200)
    with pytest.raises(RuntimeError):
        ET.make_eval_packet(np.zeros((0, 5), np.float32), torch.zeros(1, 51), torch.zeros(1, 10), torch.eye(3)[None], 50,
                            torch.device("cpu"))


def test_block_order_and_exclusivity():
    J = TCOST.jobs("tpdcost", ["a3c:syn:final"], ["c5:3407"])
    keys = [j["key"] for j in J]
    assert keys[0] == "tpdcost:cost:s43:exec:default" and keys[5] == "tpdcost:cost:s43:exec:det" and len(J) == 10
    assert all(j["kind"] == "exclusive" for j in J)
    assert J[0]["deps"] == ["a3c:syn:final"] and all(J[i]["deps"] == [keys[i - 1]] for i in range(1, len(J)))
    assert J[1]["command"][3:] == ["run", "--net", "arm", "--arm", "c5", "--seed", "3407", "--policy", "default"]
