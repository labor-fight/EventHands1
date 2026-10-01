"""Root tracking R2-A: with ROOT_ABS the global rotation is the state-free head's output (prev's on an
empty packet) and everything else is the tracking arm's; without it nothing changes."""
import copy
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "model"))
from config import load_config                     # noqa: E402
from model import MNISTModel                       # noqa: E402
from semkine import eval_track as ET               # noqa: E402


def _batch(n_events, prev):
    g = np.random.default_rng(0)
    ev = np.zeros((n_events, 5), np.float32)
    ev[:, 1] = g.integers(60, 180, n_events)
    ev[:, 2] = g.integers(40, 140, n_events)
    ev[:, 3] = np.sort(g.random(n_events) * 0.05)
    ev[:, 4] = g.integers(0, 2, n_events)
    return ET.make_eval_packet(ev, prev, torch.zeros(1, 10), torch.tensor([[[200., 0, 120], [0, 200., 90], [0, 0, 1]]]),
                               50, "cpu")


def _model(root_abs):
    cfg = load_config(REPO / "configs/rt/rt_s37_2k.yaml")
    cfg = copy.deepcopy(cfg)
    if root_abs:
        cfg["MODEL"]["ROOT_ABS"] = True
    torch.manual_seed(0)
    return MNISTModel(cfg).eval()


def test_root_abs_replaces_rotation_only():
    prev = torch.zeros(1, 51)
    prev[0, 2] = 0.4
    prev[0, 3:6] = torch.tensor([0.3, -0.2, 0.1])
    base, ra = _model(False), _model(True)
    sd = base.state_dict()
    missing = ra.load_state_dict(sd, strict=False)
    assert set(missing.missing_keys) == {k for k in ra.state_dict() if k.startswith("root_abs_head")}
    b = _batch(500, prev)
    with torch.no_grad():
        o0, o1 = base.forward_packet(b), ra.forward_packet(b)
        f = ra.event_encoder(b.events, b.ptr, b.delta_t_s, None)
        rot = ra.root_abs_head(f)
    assert torch.equal(o0[:, :3], o1[:, :3]) and torch.equal(o0[:, 6:], o1[:, 6:])
    assert torch.allclose(o1[:, 3:6], rot, atol=1e-6)
    with torch.no_grad():
        e = ra.forward_packet(_batch(0, prev))
    assert torch.equal(e, prev)


def test_root_abs_head_trains():
    ra = _model(True).train()
    prev = torch.zeros(1, 51)
    prev[0, 2] = 0.4
    out = ra.forward_packet(_batch(400, prev))
    out[:, 3:6].pow(2).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in ra.root_abs_head.parameters())
