"""CMP amendment 6 (docs/EXPERIMENT_RECORDS.md#cmp-amend7-20261110): EventHPE on the CMP protocol, against the official code.

CPU only (CUDA_VISIBLE_DEVICES=""). The official definitions (github.com/JimmyZou/EventHPE at 2af451f) are restated here
where they are compared.
"""
from __future__ import annotations

from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.cmp import arms as A  # noqa: E402
from tools.cmp import ext_ehpe as X  # noqa: E402
from semkine import cmp_ehpe as M  # noqa: E402
from semkine.cmp_seq_ehpe import event_frames  # noqa: E402
from semkine.eventhpe_official import geometry as G  # noqa: E402

H, W = 180, 240


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return M.EHPEModel(A.build_config("ehpe", 3407)).eval()


def random_states(n, seed=0, scale=0.4):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(n, 51, generator=g) * scale
    s[:, :3] = torch.tensor([0.0, 0.0, 0.55]) + 0.05 * torch.randn(n, 3, generator=g)
    return s


def random_inputs(b, t, seed=0):
    g = torch.Generator().manual_seed(seed)
    ev = (torch.rand(b, t, 8, H, W, generator=g) < 0.02).to(torch.uint8)
    vox = torch.randn(b, t, 5, H, W, generator=g) * (torch.rand(b, t, 5, H, W, generator=g) < 0.02)
    return ev, vox


# ---------------------------------------------------------------------------------------------- registry and config
def test_config_is_the_official_recipe():
    cfg = A.build_config("ehpe", 3407)
    c, t = cfg["CMP"], cfg["TRAIN"]
    assert c["MODEL"] == "ehpe" and c["STAGE"] == "seq" and c["INIT"] == {"ARM": "epf", "STEP": 3000}
    assert c["SEQ"] == {"PACKET_MS": 50, "K": 8, "VOXEL_BINS": 5, "EVENT_BINS": 8}          # official num_steps 8, 8 channels
    assert c["LOSS"] == {"DELTA_TRAN": 0.0, "TRAN": 1.0, "THETA": 10.0, "JOINTS3D": 1.0, "JOINTS2D": 10.0, "FLOW": 0.1,
                         "GEODESIC": True, "FLOW_VALID_PX": 4.0}                              # train.py get_args defaults
    assert t["LR"] == 1.0e-3 and t["BATCH_SIZE_PER_GPU"] == 8 and t["ACCUMULATE_GRAD_BATCHES"] == 1
    assert t["OPTIMIZER"] == "adam" and "WEIGHT_DECAY_L2" not in t and t["MAX_STEPS"] == 45000
    assert t["MAX_STEPS"] * t["BATCH_SIZE_PER_GPU"] * c["SEQ"]["K"] == 7500 * 32 * 12         # ep's packet-samples
    assert t["MAX_STEPS"] % t["SAVE_EVERY_N_STEPS"] == 0
    assert cfg["AUG"] == {"DOMRAND": {"ENABLED": False}, "SPEED_AUG": False, "POLARITY_FLIP": False,
                          "PIXEL_POLARITY_SWAP": False}
    assert cfg["MODEL"]["INIT_FROM"].endswith("runs/cmp_epf_s3407/cmp_epf_s3407-step=3000.ckpt")
    assert cfg["TRACK"]["PREV_NOISE_MODE"] == "mixed"                                         # the tracker's initial state
    assert A.EVAL["ehpe"] == {"WINDOW_MS": 50, "HISTORY": 1}
    assert A.EXT["ehpe"] is X


def test_registered_configs_unchanged():
    A.write_configs("formal", [a for a in A.ARMS if a not in A.EXT])


# ---------------------------------------------------------------------------------------------- event frame
def official_events_frame(events, num_partitions, h, w):
    """flow_net/prepare_data.py `generate_events_frame` without the sensor flip: events [N, 3] = v, u, time."""
    start_time, end_time = events[0, 2], events[-1, 2]
    time_window = (end_time - start_time) / num_partitions
    frames = []
    for i in range(num_partitions):
        frame = np.zeros([h, w])
        idx = (events[:, 2] >= (start_time + i * time_window)) & (events[:, 2] < (start_time + (i + 1) * time_window))
        frame[events[idx, 0], events[idx, 1]] = 1
        frames.append(frame)
    return np.stack(frames, axis=0)


def test_event_frames_are_the_official_partitions():
    rng = np.random.default_rng(0)
    n = 5000
    us = np.sort(rng.integers(0, 50000, n))
    us[0] = 0
    xs, ys = rng.integers(0, W, n), rng.integers(0, H, n)
    ours = event_frames(xs, ys, us, 50, 8, H, W)
    assert ours.shape == (8, H, W) and ours.dtype == np.uint8 and set(np.unique(ours)) <= {0, 1}
    # the official partitions run from the first to the last event; a sentinel at the packet end makes them the packet's
    ev = np.concatenate([np.stack([ys, xs, us], 1), [[0, 0, 50000]]])
    assert np.array_equal(ours, official_events_frame(ev, 8, H, W).astype(np.uint8))
    # polarity is not an input, the last microsecond falls in the last partition, an empty packet is all zero
    assert event_frames(np.array([3]), np.array([2]), np.array([49999]), 50, 8, H, W)[7, 2, 3] == 1
    assert event_frames(np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64), 50, 8, H, W).sum() == 0


# ---------------------------------------------------------------------------------------------- MANO and rotations
def test_mano_from_rotation_matrices_equals_the_layer(model):
    s = random_states(6, 1)
    betas = 0.5 * torch.randn(6, 10, generator=torch.Generator().manual_seed(2))
    v0, j0 = model._fk(s, betas)
    v1, j1 = M.mano_rotmats(model.mano, betas, model.state_rotmats(s), s[:, :3])
    assert v1.shape == (6, 778, 3) and j1.shape == (6, 21, 3)
    assert (v0 - v1).abs().max() < 1e-5 and (j0 - j1).abs().max() < 1e-5


def test_state_rotation_round_trip(model):
    s = random_states(64, 3)
    back = model.rotmats_state(s[:, :3], model.state_rotmats(s))
    assert back.shape == (64, 51) and (back - s).abs().max() < 1e-4
    zero = torch.zeros(2, 51)
    assert (model.rotmats_state(zero[:, :3], model.state_rotmats(zero)) - zero).abs().max() < 1e-5


def test_compose_is_the_official_accumulation(model):
    B, T = 3, 8
    g = torch.Generator().manual_seed(4)
    init = random_states(B, 5)
    delta_pose, delta_tran = torch.randn(B, T, 96, generator=g), 0.01 * torch.randn(B, T, 3, generator=g)
    rot, trans = model.compose(init, delta_pose, delta_tran)
    # event_pose_estimation/model.py `EventTrackNet.forward`, restated
    off_trans = init[:, None, :3] + torch.cumsum(delta_tran, dim=1)
    delta = G.rot6d_to_rotmat(delta_pose).view(B, T, 16, 3, 3)
    init_rot = G.batch_rodrigues(torch.cat([init[:, 3:6], init[:, 6:] + model.mano.hands_mean.view(1, -1)], 1)
                                 .reshape(-1, 3)).view(B, 16, 3, 3)
    off_rot = init_rot
    for k in range(T):
        off_rot = torch.matmul(delta[:, k], off_rot)
        assert (rot[:, k] - off_rot).abs().max() < 1e-5
    assert (trans - off_trans).abs().max() < 1e-6
    eye = torch.eye(3).expand(B, T, 16, 3, 3)
    assert (torch.matmul(delta, delta.transpose(-1, -2)) - eye).abs().max() < 1e-5          # 6D -> rotations
    # step by step through the 51-D state (the deployed form) = the snippet accumulation
    state = init
    for k in range(T):
        r, t = model.compose(state, delta_pose[:, k:k + 1], delta_tran[:, k:k + 1])
        state = model.rotmats_state(t[:, 0], r[:, 0])
        assert (r[:, 0] - rot[:, k]).abs().max() < 1e-4 and (t[:, 0] - trans[:, k]).abs().max() < 1e-6


# ---------------------------------------------------------------------------------------------- network
def test_network_is_the_official_design(model):
    assert model.img_encoder.conv1.in_channels == 10 and model.img_encoder.conv1.out_channels == 64    # 8 events + 2 flow
    assert type(model.img_encoder).__name__ == "ResNet" and len(model.img_encoder.layer3) == 6         # ResNet-50
    gru = model.tmp_encoder.gru
    assert (gru.input_size, gru.hidden_size, gru.num_layers, gru.bidirectional) == (2048, 2048, 1, False)
    assert model.tmp_encoder.use_residual and model.tmp_encoder.linear is None
    assert model.regressor.decpose.out_features == 16 * 6 and model.regressor.dectrans.out_features == 3
    assert all(not p.requires_grad for p in model.flow_encoder.parameters())
    assert all(p.requires_grad for n, p in model.named_parameters() if not n.startswith("flow_encoder."))


def test_deployed_step_equals_the_snippet_forward(model):
    B, T = 2, 8
    ev, vox = random_inputs(B, T, 6)
    init = random_states(B, 7, 0.2)
    with torch.no_grad():
        out = model.snippet(ev, vox, init)
        states = model.snippet_states(out)
        assert states.shape == (B, T, 51) and out["flow"].shape == (B * T, 2, 192, 256)
        prev, hidden = init, None
        for k in range(T):
            prev, hidden = model.track_step(ev[:, k], vox[:, k], prev, hidden)
            assert (prev - states[:, k]).abs().max() < 2e-4, k
        # a recurrent state reset changes the output: the chunking of the deployed form matters
        other, _ = model.track_step(ev[:, T - 1], vox[:, T - 1], states[:, T - 2], None)
        assert (other - states[:, T - 1]).abs().max() > 1e-6


def test_loss_is_the_official_sum(model):
    B, T = 2, 8
    ev, vox = random_inputs(B, T, 8)
    init = random_states(B, 9, 0.2)
    target = init[:, None] + 0.05 * torch.randn(B, T, 51, generator=torch.Generator().manual_seed(10))
    betas = torch.zeros(B, 10)
    K = torch.tensor([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]).expand(B, 3, 3)
    out = model.snippet(ev, vox, init)
    loss, logs = model.losses(out, init, target, betas, K)
    pred, tgt = model.loss_inputs(out, init, target, betas, K)
    # event_pose_estimation/loss_funcs.py `compute_losses`, restated term by term
    tran = torch.mean((pred["tran"] - tgt["tran"]) ** 2)
    trace = torch.sum(pred["pred_rotmats"] * model.state_rotmats(target.reshape(-1, 51)).view(B, T, 16, 3, 3), dim=(-2, -1))
    theta = torch.mean(torch.acos(torch.clamp(0.5 * (trace - 1), -1 + 1e-6, 1 - 1e-6)))
    j3d = torch.mean((pred["joints3d"] - tgt["joints3d"]) ** 2)
    j2d = torch.mean((torch.clamp(pred["joints2d"], 0, 1) - tgt["joints2d"]) ** 2)
    close = lambda a, b: abs(float(a) - float(b)) <= 1e-5 * max(1.0, abs(float(b)))  # noqa: E731
    assert close(logs["loss_tran"], tran) and close(logs["loss_theta"], 10.0 * theta)
    assert close(logs["loss_joints3d"], j3d) and close(logs["loss_joints2d"], 10.0 * j2d)
    assert 0.0 <= float(logs["loss_flow"]) <= 0.1 * 2.0 and float(logs["loss_delta_tran"]) == 0.0
    total = sum(float(logs[f"loss_{k}"]) for k in ("tran", "theta", "joints3d", "joints2d", "flow", "delta_tran"))
    assert close(loss, total)
    assert pred["verts"].shape == (B, T + 1, 778, 3) and tgt["flows"].shape == (B, T, 2, H, W)
    # the translation is detached in the joints and vertices (official forward): only the tran term reaches the head
    g_all = torch.autograd.grad(loss, out["delta_tran"], retain_graph=True)[0]
    g_tran = torch.autograd.grad(tran, out["delta_tran"], retain_graph=True)[0]
    assert g_tran.abs().max() > 0 and (g_all - g_tran).abs().max() <= 1e-6 * g_tran.abs().max()
    loss.backward()
    assert all(p.grad is None for p in model.flow_encoder.parameters())
    assert model.regressor.decpose.weight.grad.abs().sum() > 0 and model.img_encoder.conv1.weight.grad.abs().sum() > 0
    model.zero_grad()


# ---------------------------------------------------------------------------------------------- deployed form
class _StubModel:
    voxel_bins = 5

    def __init__(self):
        self.fresh = []

    def track_step(self, ev, vox, prev, hidden):
        self.fresh.append(hidden is None)
        return prev + 1.0, (0 if hidden is None else hidden) + 1


def test_adapter_chains_snippets_as_the_official_test():
    runs = np.array([[0, 1000], [2000, 2700]])

    def net(form):
        n = types.SimpleNamespace(cfg={"CMP": {"SEQ": {"K": 8, "EVENT_BINS": 8}}, "DATA": {"HEIGHT": H, "WIDTH": W}},
                                  form=form, device=torch.device("cpu"), model=_StubModel())
        n.run_start = lambda end: int(runs[np.searchsorted(runs[:, 0], end, side="right") - 1, 0])
        return n

    for form, want in (("main", [k % 8 == 0 for k in range(20)] + [k % 8 == 0 for k in range(14)]),
                       ("carry", [k == 0 for k in range(20)] + [k == 0 for k in range(14)])):
        n = net(form)
        ad = X.Adapter(n)
        ad.inputs = lambda ctx, end: (None, None)
        ad.bind(None)
        prev = np.zeros(51, np.float32)
        for a, b in runs:
            for end in range(a + 49, b, 50):
                prev = ad.step(None, end, prev)
                assert prev.shape == (51,) and prev.dtype == np.float32
        assert n.model.fresh == want, form
    # a rebind and a gap in the packets both start a snippet
    n = net("carry")
    ad = X.Adapter(n)
    ad.inputs = lambda ctx, end: (None, None)
    ad.bind(None)
    for end in (49, 99, 249):
        ad.step(None, end, np.zeros(51, np.float32))
    ad.bind(None)
    ad.step(None, 299, np.zeros(51, np.float32))
    assert n.model.fresh == [True, False, True, True]
    assert X.Adapter.FORMS == ("main", "carry", "gtinit") and X.Adapter.history == 1


def test_gtinit_starts_every_snippet_from_the_gt_state():
    """Amendment 9: the official evaluation's form -- the state fed to the first packet of every snippet is GT at
    that packet's start, every other packet gets the tracker's own state."""
    runs = np.array([[0, 1000], [2000, 2700]])
    gt = np.arange(3000, dtype=np.float32)[:, None].repeat(51, 1)          # GT state at ms t = t
    ctx = types.SimpleNamespace(gt=gt)
    seen = []

    class Stub(_StubModel):
        def track_step(self, ev, vox, prev, hidden):
            seen.append(float(prev[0, 0]))
            return super().track_step(ev, vox, prev, hidden)
    n = types.SimpleNamespace(cfg={"CMP": {"SEQ": {"K": 8, "EVENT_BINS": 8}}, "DATA": {"HEIGHT": H, "WIDTH": W}},
                              form="gtinit", device=torch.device("cpu"), model=Stub())
    n.run_start = lambda end: int(runs[np.searchsorted(runs[:, 0], end, side="right") - 1, 0])
    ad = X.Adapter(n)
    ad.inputs = lambda ctx, end: (None, None)
    ad.bind(ctx)
    prev = np.full(51, -5.0, np.float32)                                 # the engine's state is ignored at snippet starts
    want = []
    for a, b in runs:
        for k, end in enumerate(range(a + 49, b, 50)):
            start = float(max(end - 50, a))
            want.append(start if k % 8 == 0 else want[-1] + 1.0)
            prev = ad.step(ctx, end, prev)
    assert seen == want
