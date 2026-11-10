"""CMP amendment 4 (docs/EXPERIMENT_RECORDS.md#cmp-amend4-20261110): EventHands as published, against the official code.

CPU only (CUDA_VISIBLE_DEVICES=""). The official definitions (github.com/r00tman/EventHands, model/model.py at abfca72)
are restated here verbatim where they are compared.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.cmp import arms as A  # noqa: E402
from tools.cmp.nets import KF, KF_DT, kf_update  # noqa: E402


class OfficialNet(nn.Module):
    """model/model.py `MNISTModel` of the official code (network part)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 3, kernel_size=3, padding=1)
        self.rn = models.resnet18(num_classes=12)

    def forward(self, x, prevpos):
        x = x.permute(0, 3, 1, 2)
        return self.rn(self.conv1(x))


def official_loss(pred, y):
    """model/model.py `BaseModel.training_step` of the official code, before its log10."""
    mano_loss = F.mse_loss(pred[:, :6], y[:, :6])
    pos_loss = F.mse_loss(pred[:, 6:9], y[:, 6:9])
    rot_loss = F.mse_loss(pred[:, 9:], y[:, 9:])
    return (mano_loss * 6 / 0.1 + pos_loss * 3 / 0.0001 + rot_loss * 3 / 0.05) / 12


@pytest.fixture(scope="module")
def model():
    from semkine.cmp_eho import EHOModel
    torch.manual_seed(0)
    return EHOModel(A.build_config("eho", 3407)).eval()


def test_config_is_the_official_recipe():
    c = A.build_config("eho", 3407)
    assert c["TRAIN"]["BATCH_SIZE_PER_GPU"] * c["TRAIN"]["ACCUMULATE_GRAD_BATCHES"] == 64
    assert c["TRAIN"]["LR"] == 1e-3 and c["TRAIN"]["WARMUP_STEPS"] == 0 and c["TRAIN"]["LR_SCHEDULE"] == "constant"
    assert c["TRAIN"]["PRECISION"] == 32
    assert c["TRAIN"]["MAX_STEPS"] * 64 == A.ARMS["eh"]["steps"] * A.ARMS["eh"]["batch"] * A.ARMS["eh"]["accumulate"]
    assert c["DATA"]["WINDOW_MIN"] == c["DATA"]["WINDOW_MAX"] == 100 and c["CMP"]["EVAL"]["WINDOW_MS"] == 100
    assert not c["AUG"]["DOMRAND"]["ENABLED"] and not c["AUG"]["SPEED_AUG"]
    assert not c["AUG"]["POLARITY_FLIP"] and not c["AUG"]["PIXEL_POLARITY_SWAP"]
    b2 = A.build_config("eh", 3407)
    assert c["TRACK"] == b2["TRACK"] and c["MANO"] == b2["MANO"]          # the protocol's segment-start noise
    assert {k: v for k, v in c["DATA"].items() if not k.startswith("WINDOW")} == \
        {k: v for k, v in b2["DATA"].items() if not k.startswith("WINDOW")}


def test_network_is_the_official_network(model):
    off = OfficialNet()
    ours = {k: v.shape for k, v in model.state_dict().items()}
    assert ours == {k: v.shape for k, v in off.state_dict().items()}
    off.load_state_dict(model.state_dict())
    off.eval()
    x = torch.rand(3, 180, 240, 2)
    with torch.no_grad():
        # this repository's forward makes the permuted input contiguous; same network, the convolutions' summation
        # order follows the memory layout, so the outputs agree to float32 rounding (bitwise on a contiguous input)
        assert torch.allclose(model(x, None), off(x, None), rtol=1e-5, atol=1e-6)
        xc = x.permute(0, 3, 1, 2).contiguous().permute(0, 2, 3, 1)
        assert torch.equal(model(x, None), off.rn(off.conv1(xc.permute(0, 3, 1, 2).contiguous())))


def test_loss_is_the_official_loss(model):
    g = torch.Generator().manual_seed(1)
    y51 = torch.randn(16, 51, generator=g) * 0.3
    pred = torch.randn(16, 12, generator=g) * 0.3
    loss, _ = model._compute_loss(pred, y51)
    ref = official_loss(pred, model.to_target(y51))
    assert torch.allclose(loss, ref, rtol=1e-6, atol=0)
    assert model.log10_loss


def test_target_and_state(model):
    g = torch.Generator().manual_seed(2)
    y51 = torch.randn(8, 51, generator=g) * 0.3
    t12 = model.to_target(y51)
    s = model.state(t12)
    assert torch.equal(s[:, :6], y51[:, :6])                                   # t and R pass through
    c6 = torch.tensor(np.load(model.cfg["MANO"]["NPZ"])["hands_components"][:6], dtype=torch.float64)
    proj = y51[:, 6:].double() @ torch.linalg.pinv(c6) @ c6                    # PCA6 projection of the residual
    assert torch.allclose(s[:, 6:].double(), proj, atol=1e-5)
    out = torch.randn(8, 12, generator=g)
    assert torch.allclose(model.to_target(model.state(out)), out, atol=1e-5)  # 12 -> 51 -> 12


def dense_reference(zs, r, q):
    """filter/filter.py as one 24-D filterpy filter (predict; Joseph-form update), with position = z, velocity 0 first."""
    n = zs.shape[1]
    Fm, Hm = np.zeros((2 * n, 2 * n)), np.zeros((n, 2 * n))
    for i in range(n):
        Fm[2 * i, 2 * i] = Fm[2 * i, 2 * i + 1] = Fm[2 * i + 1, 2 * i + 1] = 1.0
        Hm[i, 2 * i] = 1.0
    qb = q * np.array([[KF_DT ** 4 / 4, KF_DT ** 3 / 2], [KF_DT ** 3 / 2, KF_DT ** 2]])
    Qm = np.kron(np.eye(n), qb)
    Rm, P = r * np.eye(n), np.eye(2 * n)
    x = np.zeros(2 * n)
    x[0::2] = zs[0]
    out = [zs[0].copy()]
    for z in zs[1:]:
        x = Fm @ x
        P = Fm @ P @ Fm.T + Qm
        PHT = P @ Hm.T
        K = PHT @ np.linalg.inv(Hm @ PHT + Rm)
        x = x + K @ (z - Hm @ x)
        I_KH = np.eye(2 * n) - K @ Hm
        P = I_KH @ P @ I_KH.T + K @ Rm @ K.T
        out.append(x[0::2].copy())
    return np.array(out)


@pytest.mark.parametrize("form", sorted(KF))
def test_kalman_matches_the_dense_filter(form):
    r, q = KF[form]
    rng = np.random.default_rng(3)
    zs = np.cumsum(rng.standard_normal((400, 12)) * 0.01, 0) + rng.standard_normal((400, 12)) * 0.05
    state, ours = None, []
    for z in zs:
        state = kf_update(state, z, r, q)
        ours.append(state[0][:, 0].copy())
    np.testing.assert_allclose(np.array(ours), dense_reference(zs, r, q), rtol=1e-10, atol=1e-12)


def test_existing_configs_unchanged():
    import yaml
    for arm in A.ARMS:
        if arm == "eho":
            continue
        for seed in A.SEEDS:
            p = A.config_path(arm, seed)
            assert p.read_text() == yaml.safe_dump(A.build_config(arm, seed), sort_keys=False), p
