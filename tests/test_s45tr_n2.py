"""S45-TR N2 (docs/EXPERIMENT_RECORDS.md#s45tr-prereg-20261030): the second pass sees the same packet with the first
output (detached) as its prior, the objective is L1 + lambda L2, gradients reach the parameters through both passes,
and the journalled prediction / route statistics stay the first pass's."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

from semkine import s45tr_n2 as N2  # noqa: E402
from tools.s45tr import common as C  # noqa: E402


def small_batch():
    from config import load_config
    import semkine.train as T
    from semkine.dataset import build_dataset
    from semkine.events import collate_packets
    cfg = load_config(str(C.S45_CONFIG))
    components = np.load(cfg["MANO"]["NPZ"])["hands_components"].astype(np.float32)
    ds = build_dataset(cfg, "train", components, train=True)
    batch = collate_packets([ds[i] for i in (11, 222222, 3333333)])
    torch.manual_seed(3407)
    model = T.MNISTModel(cfg)
    with torch.no_grad():   # a non-zero head so the second pass differs from the first
        model.xyz.fc2.weight.normal_(0, 0.01)
    return model.train(), batch


def test_second_pass_uses_detached_first_output_and_same_packet():
    model, batch = small_batch()
    calls = []
    native = type(model).forward_packet

    def spy(self, packet):
        out = native(self, packet)
        calls.append((packet, out))
        return out
    logged = {}
    try:
        type(model).forward_packet = spy
        model.log = lambda name, value, **kw: logged.__setitem__(name, value)
        N2.install(1.0)
        out = model.training_step(batch, 0)
    finally:
        N2.uninstall()
        type(model).forward_packet = native
    assert len(calls) == 2
    (p1, o1), (p2, o2) = calls
    assert torch.equal(p2.prev_state, o1.detach().to(p1.prev_state.dtype)) and not p2.prev_state.requires_grad
    for name in ("events", "ptr", "target", "betas", "camera_K", "delta_t_s"):
        assert getattr(p2, name) is getattr(p1, name)
    l1 = model._compute_loss(o1, p1.target, p1.betas)[0]
    l2 = model._compute_loss(o2, p1.target, p1.betas)[0]
    assert torch.allclose(logged["train_loss"], l1 + l2) and torch.allclose(out, (l1 + l2).float().log10())
    assert torch.allclose(logged["train_loss_first"], l1) and torch.allclose(logged["train_loss_second"], l2)
    out.backward()
    assert model.xyz.fc2.weight.grad is not None and torch.count_nonzero(model.xyz.fc2.weight.grad) > 0


def test_uninstall_restores_s45_training_step():
    import semkine.train as T
    original = T.MNISTModel.training_step
    N2.install(1.0)
    assert N2.installed() == 1.0 and T.MNISTModel.training_step is N2.training_step
    N2.uninstall()
    assert T.MNISTModel.training_step is original and N2.installed() is None
