"""AGG778 R2 probe instrument (docs/EXPERIMENT_RECORDS.md#agg778-r2-prereg-20261024): masks touch only their columns,
the downstream is state-blind, M13i starts from the frozen module, the rebuilt residual slot is zero without residual
events, fan-out weights sum to q."""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "tests"))

from test_xcl_model import inputs, net  # noqa: E402
from tools.agg778 import common as C  # noqa: E402
from tools.agg778 import heads as H  # noqa: E402
from tools.agg778 import r2_task as R  # noqa: E402


@pytest.fixture(autouse=True)
def small_mesh(monkeypatch):
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    monkeypatch.setattr(C, "V", 6)
    monkeypatch.setattr(H, "V", 6)
    yield
    torch.set_num_threads(old)


def capture():
    m = net().eval()
    tok, d, mask, verts, assoc, prev48 = inputs()
    with torch.no_grad():
        m(tok, d, mask, verts, assoc, prev48, capture=True)
    w = assoc["q"].unsqueeze(-1) * assoc["A"] * mask.float().unsqueeze(-1)
    cap = {"assoc": assoc, "d": d, "mask": mask, "verts": verts, "tok": tok, "O": m.xag_last_capture["O"],
           "H0": m.xag_last_capture["H0"], "r": (1 - assoc["q"]) * mask.float(), "w": w,
           "counts": torch.tensor([int(mask.sum())])}
    return m, cap


def test_masks_zero_only_their_columns():
    _, cap = capture()
    rec = H.c1_records(cap)
    for name, cols in R.MASKS.items():
        out = R.masked(rec, cols)
        keep = [c for c in range(13) if c not in cols]
        assert torch.equal(out["feat"][:, keep], rec["feat"][:, keep]), name
        assert (out["feat"][:, list(cols)] == 0).all(), name
    assert R.masked(rec, ())["feat"] is rec["feat"]


def test_downstream_is_state_blind(monkeypatch):
    m, cap = capture()
    frozen = SimpleNamespace(xyz=m)
    seen = []
    native = H.Tail.forward

    def spy(self, feat, O, edge_g, bslot, prev48):
        seen.append(prev48.clone())
        return native(self, feat, O, edge_g, bslot, prev48)
    monkeypatch.setattr(H.Tail, "forward", spy)
    arm = R.Arm("T0s", frozen)
    inp = {"cap": cap, "B": 1, "edge": m.edge_geometry(cap["verts"])}
    with torch.no_grad():
        out = arm(inp)
    assert seen and (seen[0] == 0).all() and out.shape == (1, 51)


def test_m13i_starts_from_the_frozen_module():
    m, _ = capture()
    arm = R.Arm("M13i", SimpleNamespace(xyz=m))
    for a, b in zip(arm.op.rel.state_dict().values(), m.rel.state_dict().values()):
        assert torch.equal(a, b)
    assert torch.equal(arm.tail.inp.weight, m.vin.weight)
    fresh = R.Arm("M13", SimpleNamespace(xyz=m))
    assert not torch.equal(fresh.op.rel[0].weight, m.rel[0].weight)


def test_rebuilt_residual_slot_is_zero_without_residual_events():
    _, cap = capture()
    phi = torch.nn.Sequential(torch.nn.Linear(13, 8), torch.nn.ReLU(), torch.nn.Linear(8, 128))
    cap0 = dict(cap, r=torch.zeros_like(cap["r"]))
    out = R.residual_slot(phi, cap0, ())
    assert out.shape == (1, 258) and (out == 0).all()
    out_abs = R.residual_slot(phi, cap, R.ABS)
    assert out_abs.shape == (1, 258) and torch.isfinite(out_abs).all()


@pytest.mark.parametrize("m", [4, 6])  # the scene has 6 vertices; the probes use m = 4 and 8 on 778
def test_fanout_weights_sum_to_q(m):
    _, cap = capture()
    B, V = 1, 6
    verts = cap["verts"]
    f = torch.ones(B)
    geo = {"fx": f * 100, "fy": f * 100, "cx": f * 0, "cy": f * 0,
           "uv": C.project(verts, f * 100, f * 100, f * 0, f * 0)}
    rec = H.fanout_records(cap, geo, band_px=1e4, m=m)
    q = cap["assoc"]["q"]
    live = (q > 0) & cap["mask"]
    total = torch.zeros(B, cap["d"].shape[1]).index_put_((rec["rb"], rec["rn"]), rec["wrec"], accumulate=True)
    assert torch.allclose(total[live], q[live].float(), atol=1e-5)
    assert int((rec["rb"] == 0).sum()) <= int(live.sum()) * min(m, V)


def test_groups_and_needs_are_consistent():
    for key, spec in R.GROUPS.items():
        cond, regime, group = key
        assert regime in ("e3", "noise")
        for arm in spec:
            assert arm in R.FRESH or arm in ("T0s", "T6", "T6s12", "T6fan", "T6K", "T2", "T2abs", "T4", "T5")
            if arm in ("T4", "T5"):
                assert cond == "A3"
        n = R.needs(spec)
        assert "rec" in n
