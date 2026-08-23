#!/usr/bin/env python3
"""Sanity checks for absolute-pose baselines (PCA12 vs Full51)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

from config import load_config
from mano_layer import ManoLayer
from model import MNISTModel
from pose_repr import axis_angle_to_quaternion, decode_to_mano_inputs, meta51_to_target


@pytest.fixture(scope="module")
def mano():
    return ManoLayer(ROOT / "assets" / "mano_right.npz", add_mean=False)


@pytest.fixture(scope="module")
def cfg_pca():
    return load_config(ROOT / "configs" / "eventhands_abs_pca12.yaml")


@pytest.fixture(scope="module")
def cfg_full():
    return load_config(ROOT / "configs" / "eventhands_abs_full51.yaml")


def test_output_shape_pca12(cfg_pca):
    m = MNISTModel(cfg_pca)
    x = torch.randn(4, 180, 240, 2)
    prev = torch.zeros(4, 12)
    y = m(x, prev)
    assert y.shape == (4, 12)


def test_output_shape_full51(cfg_full):
    m = MNISTModel(cfg_full)
    x = torch.randn(4, 180, 240, 2)
    prev = torch.zeros(4, 51)
    y = m(x, prev)
    assert y.shape == (4, 51)


def test_mano_shapes_consistent(mano, cfg_pca, cfg_full):
    B = 2
    betas = torch.zeros(B, 10)
    go = torch.randn(B, 3) * 0.1
    transl = torch.randn(B, 3) * 0.05
    # Use a residual exactly in the PCA-6 subspace so both reprs decode identically.
    alpha = torch.randn(B, 6) * 0.1
    residual = alpha @ mano.hands_components[:6]

    pred12 = torch.cat([alpha, transl, go], dim=-1)
    dec12 = decode_to_mano_inputs(
        pred12, "mano_pca6", mano.hands_components, mano.hands_mean
    )
    v12, j12 = mano(betas, dec12["global_orient"], dec12["local_full_aa"], dec12["transl"])

    pred51 = torch.cat([transl, go, residual], dim=-1)
    dec51 = decode_to_mano_inputs(
        pred51, "mano_full_axis_angle", mano.hands_components, mano.hands_mean
    )
    v51, j51 = mano(betas, dec51["global_orient"], dec51["local_full_aa"], dec51["transl"])

    assert v12.shape == v51.shape == (B, 778, 3)
    assert j12.shape == j51.shape == (B, 21, 3)
    assert torch.allclose(v12, v51, atol=1e-5)
    assert torch.allclose(j12, j51, atol=1e-5)


def test_gt_mano_matches_dataset(mano):
    mesh_path = Path(
        "/data1/lyq/data/hand_data/hand_data/train/ch_global/mesh/ch_global_mesh.npz"
    )
    if not mesh_path.exists():
        pytest.skip("hand_data not available")
    mesh = np.load(mesh_path)
    idx = np.array([0, 50, 200])
    betas = torch.tensor(mesh["betas"][idx])
    go = torch.tensor(mesh["global_orient_rotvec"][idx])
    hp = torch.tensor(mesh["hand_pose_residual_aa"][idx])
    tr = torch.tensor(mesh["camera_transl_right_canonical_m"][idx])
    local = hp + mano.hands_mean.cpu()
    verts, joints = mano(betas, go, local, tr)
    gt_v = torch.tensor(mesh["verts_right_canonical_cam_m"][idx])
    gt_j = torch.tensor(mesh["joints_right_canonical_cam_m"][idx])
    v_err = (verts - gt_v).norm(dim=-1).mean().item() * 1000
    j_err = (joints - gt_j).norm(dim=-1).mean().item() * 1000
    assert v_err < 0.5, v_err
    assert j_err < 0.5, j_err


def test_betas_constant_in_sequence():
    mesh_path = Path(
        "/data1/lyq/data/hand_data/hand_data/train/ch_global/mesh/ch_global_mesh.npz"
    )
    if not mesh_path.exists():
        pytest.skip("hand_data not available")
    mesh = np.load(mesh_path)
    b = mesh["betas"]
    assert float((b.max(0) - b.min(0)).max()) < 1e-4


def test_pca12_checkpoint_compatible(cfg_pca):
    ckpt = ROOT / "outputs/eventhands12/eventhands12_pchip_gpu4/last.ckpt"
    if not ckpt.exists():
        pytest.skip("legacy 12D checkpoint not found")
    m = MNISTModel(cfg_pca)
    state = torch.load(ckpt, map_location="cpu")
    sd = state["state_dict"] if "state_dict" in state else state
    # Filter to matching keys
    model_sd = m.state_dict()
    filtered = {k: v for k, v in sd.items() if k in model_sd and model_sd[k].shape == v.shape}
    missing, unexpected = m.load_state_dict(filtered, strict=False)
    # Critical: rn.fc and conv1 must load
    assert "conv1.weight" in filtered
    assert "rn.fc.weight" in filtered
    assert filtered["rn.fc.weight"].shape == (12, 512)


def test_meta51_adapter_roundtrip(mano):
    residual = torch.randn(3, 45)
    t = torch.randn(3, 3)
    R = torch.randn(3, 3)
    p51 = torch.cat([t, R, residual], dim=-1)
    t12 = meta51_to_target(p51, "mano_pca6", mano.hands_components)
    assert t12.shape == (3, 12)
    t51 = meta51_to_target(p51, "mano_full_axis_angle", mano.hands_components)
    assert torch.allclose(t51, p51)


def test_config_diff_only_model_loss():
    import subprocess

    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "diff_configs.py"),
            str(ROOT / "configs" / "eventhands_abs_pca12.yaml"),
            str(ROOT / "configs" / "eventhands_abs_full51.yaml"),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr


# ---------------- tracking (delta) mode ----------------


@pytest.fixture(scope="module")
def cfg_track():
    return load_config(ROOT / "configs" / "eventhands_track_delta51.yaml")


def test_track_forward_is_prevpos_plus_delta(cfg_track):
    torch.manual_seed(0)
    m = MNISTModel(cfg_track).eval()
    assert m.predict_delta and m.prevpos_embed
    x = torch.randn(4, 180, 240, 2)
    prev = torch.randn(4, 51)
    with torch.no_grad():
        out = m(x, prev)
        m.predict_delta = False
        delta = m(x, prev)
        m.predict_delta = True
    assert out.shape == (4, 51)
    assert torch.allclose(out, prev + delta, atol=1e-6)


def test_track_prev_mlp_zero_init(cfg_track):
    torch.manual_seed(0)
    m = MNISTModel(cfg_track).eval()
    x = torch.randn(2, 180, 240, 2)
    p1 = torch.randn(2, 51)
    p2 = torch.randn(2, 51)
    with torch.no_grad():
        # At init the prev MLP contributes zero, so delta must not depend on prevpos.
        d1 = m(x, p1) - p1
        d2 = m(x, p2) - p2
    assert torch.allclose(d1, d2, atol=1e-6)


def test_abs_mode_ignores_prevpos(cfg_full):
    torch.manual_seed(0)
    m = MNISTModel(cfg_full).eval()
    assert not m.predict_delta and not m.prevpos_embed
    x = torch.randn(2, 180, 240, 2)
    with torch.no_grad():
        y1 = m(x, torch.zeros(2, 51))
        y2 = m(x, torch.randn(2, 51))
    assert torch.allclose(y1, y2)


def test_prev_noise_sampling(cfg_track):
    from fastevc import HandData51Dataset

    root = Path(cfg_track["DATA"]["ROOT"])
    if not (root / "splits.json").exists():
        pytest.skip("hand_data51 not available")
    import json as _json

    seq = _json.loads((root / "splits.json").read_text())["val"]["trials"][0]
    mano_npz = np.load(cfg_track["MANO"]["NPZ"])
    comps = mano_npz["hands_components"].astype(np.float32)
    kwargs = dict(
        root=str(root), seq=seq, split="val", pose_repr="mano_full_axis_angle",
        components=comps, window_min=30, window_max=300, speed_aug=False,
        polarity_flip=False, pixel_polarity_swap=False, fixed_window=50,
    )
    ds_noisy = HandData51Dataset(train=True, prev_noise=(0.005, 0.05, 0.05), **kwargs)
    ds_clean = HandData51Dataset(train=True, prev_noise=(0.0, 0.0, 0.0), **kwargs)
    ds_eval = HandData51Dataset(train=False, prev_noise=(0.005, 0.05, 0.05), **kwargs)

    np.random.seed(0)
    _, prev_noisy, y_noisy, _, _ = ds_noisy[100]
    _, prev_clean, y_clean, _, _ = ds_clean[100]
    _, prev_eval, _, _, _ = ds_eval[100]
    # Same GT sample; noise only on prevpos, only in train mode
    assert torch.allclose(y_noisy, y_clean)
    assert not torch.allclose(prev_noisy, prev_clean)
    assert torch.allclose(prev_eval, prev_clean)
    # Noise magnitude roughly matches configured sigmas
    diff = (prev_noisy - prev_clean).numpy()
    assert 0 < np.abs(diff[0:3]).max() < 0.005 * 5
    assert 0 < np.abs(diff[6:51]).max() < 0.05 * 5


def test_track_config_diff_only_model_track():
    import subprocess

    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "diff_configs.py"),
            str(ROOT / "configs" / "eventhands_abs_full51.yaml"),
            str(ROOT / "configs" / "eventhands_track_delta51.yaml"),
            "--allow", "MODEL", "LOSS", "TRACK",
            "TRAIN.OUTPUT_DIR", "TRAIN.RUN_NAME", "EVAL.OUTPUT_DIR",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr

# ---------------- render-and-compare (4 channel) ----------------


@pytest.fixture(scope="module")
def cfg_render():
    return load_config(ROOT / "configs" / "eventhands_track_render51.yaml")


def test_render_input_is_four_channel(cfg_render):
    """PREV_RENDER concatenates silhouette + inverse depth onto the LNES pair."""
    m = MNISTModel(cfg_render).eval()
    assert m.conv1.weight.shape[1] == 4
    prev = torch.randn(2, 51) * 0.02
    prev[:, 2] += 0.4
    betas, K = m._resolve_betas_K(prev, None, None)
    with torch.no_grad():
        rend = m._render_prev(prev, betas, K)
    assert rend.shape == (2, 180, 240, 2)
    # Silhouette is a 0/1 mask and inverse depth is only defined inside it.
    sil, inv = rend[..., 0], rend[..., 1]
    assert torch.equal(sil, sil.clamp(0, 1))
    assert (inv[sil == 0] == 0).all()
    assert sil.sum() > 0


def test_fk_matches_direct_mano_call(cfg_render, mano):
    """The FK helper is the same computation the renderer performs inline."""
    m = MNISTModel(cfg_render).eval()
    params = torch.randn(3, 51) * 0.05
    params[:, 2] += 0.4
    betas = torch.zeros(3, 10)
    with torch.no_grad():
        dec = decode_to_mano_inputs(
            params, "mano_full_axis_angle", m.mano.hands_components, m.mano.hands_mean
        )
        v_ref, j_ref = m.mano(betas, dec["global_orient"], dec["local_full_aa"], dec["transl"])
        v, j = m._fk(params, betas)
    assert torch.equal(v, v_ref) and torch.equal(j, j_ref)


def test_zero_event_gate_holds_pose_on_empty_window(cfg_render):
    """With no events the delta is forced to zero, so the state must not move."""
    m = MNISTModel(cfg_render).eval()
    assert m.zero_event_gate
    prev = torch.randn(4, 51) * 0.02
    prev[:, 2] += 0.4
    with torch.no_grad():
        out = m(torch.zeros(4, 180, 240, 2), prev)
    assert torch.allclose(out, prev, atol=1e-6)


def test_warm_start_from_abs_checkpoint_fills_extra_conv1_channels(cfg_full, cfg_render, tmp_path):
    """A 2ch absolute ckpt warm-starts the 4ch trunk; the render channels stay zero."""
    torch.manual_seed(0)
    base = MNISTModel(cfg_full).eval()
    ckpt = tmp_path / "abs.ckpt"
    torch.save({"state_dict": base.state_dict()}, ckpt)

    torch.manual_seed(1)
    m = MNISTModel(cfg_render).eval()
    loaded = m.init_from_abs_checkpoint(str(ckpt))
    assert loaded > 0
    assert torch.equal(m.conv1.weight[:, :2], base.conv1.weight)
    assert torch.count_nonzero(m.conv1.weight[:, 2:]) == 0


# ---------------- render-channel selection ----------------


@pytest.fixture(scope="module")
def cfg_so3fk():
    return load_config(ROOT / "configs" / "eventhands_track_render51_so3fk.yaml")


def _render_inputs(n=2):
    prev = torch.randn(n, 51) * 0.02
    prev[:, 2] += 0.4
    return torch.rand(n, 180, 240, 2), prev


def _render(m, prev):
    betas, K = m._resolve_betas_K(prev, None, None)
    with torch.no_grad():
        return m._render_prev(prev, betas, K)


def test_render_channels_default_is_the_published_pair(cfg_render):
    """Omitting RENDER_CHANNELS must keep the published 4-channel input."""
    m = MNISTModel(cfg_render).eval()
    assert "RENDER_CHANNELS" not in cfg_render["MODEL"]
    assert m.render_channels == ("sil", "inv")
    assert m.conv1.weight.shape[1] == 4


def test_render_channels_width_and_order(cfg_so3fk):
    """Channel widths add up and the stack follows the configured order."""
    m = MNISTModel(cfg_so3fk).eval()
    assert m.render_channels == ("sil", "inv")
    assert m.conv1.weight.shape[1] == 2 + 1 + 1
    _, prev = _render_inputs(2)
    rend = _render(m, prev)
    assert rend.shape == (2, 180, 240, 2)


def test_render_channels_are_composable(cfg_so3fk):
    """A face is bit-identical whether rendered alone or inside a larger stack."""
    m = MNISTModel(cfg_so3fk).eval()
    _, prev = _render_inputs(2)
    full = _render(m, prev)
    m.render_channels = ("inv",)
    alone = _render(m, prev)
    assert torch.equal(full[..., 1:2], alone)


def test_render_channels_reject_unknown_names(cfg_so3fk):
    cfg = {**cfg_so3fk, "MODEL": {**cfg_so3fk["MODEL"], "RENDER_CHANNELS": ["sil", "nope"]}}
    with pytest.raises(ValueError):
        MNISTModel(cfg)
    dup = {**cfg_so3fk, "MODEL": {**cfg_so3fk["MODEL"], "RENDER_CHANNELS": ["sil", "sil"]}}
    with pytest.raises(ValueError):
        MNISTModel(dup)


def test_unknown_model_key_is_rejected(cfg_render):
    """A silently ignored MODEL key would be a config-only ablation; refuse to build."""
    cfg = {**cfg_render, "MODEL": {**cfg_render["MODEL"], "PART_HEAD": True}}
    with pytest.raises(ValueError, match="unknown MODEL keys"):
        MNISTModel(cfg)


# ---------------- semsil: kinematic semantics inside the mask value ----------------


@pytest.fixture(scope="module")
def cfg_sem():
    return load_config(ROOT / "configs" / "eventhands_track_render51_sem.yaml")


def test_semcode_is_the_radial_skeleton_coordinate(cfg_sem, mano):
    """The per-vertex code is the wrist distance of W @ J_rest, normalized to [0,1]."""
    m = MNISTModel(cfg_sem).eval()
    j_rest = mano.J_regressor @ mano.v_template
    radial = ((mano.weights @ j_rest) - j_rest[0:1]).norm(dim=-1, keepdim=True)
    ref = (radial - radial.amin()) / (radial.amax() - radial.amin())
    assert m.vert_semcode.shape == (778, 1)
    assert torch.allclose(m.vert_semcode, ref, atol=1e-6)
    # Continuity is the point of using the weights rather than an argmax part ID:
    # vertices sharing a part must not collapse onto one value.
    part = mano.weights.argmax(dim=1)
    inside = m.vert_semcode[part == int(part.mode().values)]
    assert float(inside.std()) > 1e-3


def test_semsil_keeps_the_silhouette_support_exactly(cfg_sem, cfg_so3fk):
    """semsil must stay a drop-in for sil: same occupancy, same channel count."""
    m = MNISTModel(cfg_sem).eval()
    assert m.render_channels == ("semsil", "inv")
    assert m.conv1.weight.shape[1] == 4
    ref = MNISTModel(cfg_so3fk).eval()
    _, prev = _render_inputs(3)
    sem, base = _render(m, prev), _render(ref, prev)
    assert base[..., 0].sum() > 0
    assert torch.equal((sem[..., 0] > 0).float(), base[..., 0])
    # The depth channel must be untouched: semantics go in the mask only, because
    # inv has ~37x bf16 step of dynamic range and cannot survive modulation.
    assert torch.equal(sem[..., 1], base[..., 1])


def test_semsil_is_floored_and_bounded(cfg_sem):
    """Inside the mask the value stays in [floor, 1], so occupancy is still readable."""
    m = MNISTModel(cfg_sem).eval()
    _, prev = _render_inputs(3)
    sem = _render(m, prev)[..., 0]
    inside = sem[sem > 0]
    assert float(inside.min()) >= MNISTModel.SEMSIL_FLOOR - 1e-6
    assert float(inside.max()) <= 1.0 + 1e-6
    # The floor has to dominate the bf16 spacing at that magnitude, otherwise
    # "masked" and "background" are not separable after autocast.
    spacing = MNISTModel.SEMSIL_FLOOR * torch.finfo(torch.bfloat16).eps
    assert MNISTModel.SEMSIL_FLOOR > 100 * spacing
    # And the value must actually vary, else it is just a rescaled sil.
    assert float(inside.std()) > 1e-2


def test_semsil_follows_the_pose(cfg_sem):
    """Bending a finger moves its code, so the field is pose-driven not static."""
    m = MNISTModel(cfg_sem).eval()
    _, prev = _render_inputs(1)
    a = _render(m, prev)
    moved = prev.clone()
    moved[:, 6:9] += 0.6  # first local joint of the 51D layout
    assert not torch.allclose(a[..., 0], _render(m, moved)[..., 0])


def test_semsil_arm_is_iso_parameter(cfg_so3fk, cfg_sem):
    """The arm must be free: same conv1 width means identical params and FLOPs."""
    base = sum(p.numel() for p in MNISTModel(cfg_so3fk).parameters())
    sem = sum(p.numel() for p in MNISTModel(cfg_sem).parameters())
    assert base == sem


def test_semsil_arms_differ_from_their_control_in_one_key_only():
    """Each semantic arm must be a single-key diff off its own control.

    There are two, one per loss: `ch_semsil` sits on the published 51D MSE recipe and
    `sem` on the SO(3)+FK objective, so each has to be checked against the control it
    is actually claimed against.
    """
    import subprocess

    allowed = ["TRAIN.OUTPUT_DIR", "TRAIN.RUN_NAME", "EVAL.OUTPUT_DIR"]
    arms = {"ch_semsil": "ch_both", "sem": "so3fk"}
    for arm, control in arms.items():
        r = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "diff_configs.py"),
             str(ROOT / "configs" / f"eventhands_track_render51_{control}.yaml"),
             str(ROOT / "configs" / f"eventhands_track_render51_{arm}.yaml"),
             "--allow", "MODEL.RENDER_CHANNELS", *allowed],
            capture_output=True, text=True)
        assert r.returncode == 0, f"{arm} vs {control}: {r.stdout}{r.stderr}"
        r2 = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "diff_configs.py"),
             str(ROOT / "configs" / f"eventhands_track_render51_{arm}.yaml"),
             str(ROOT / "configs" / f"eventhands_track_render51_{arm}_rep2.yaml"),
             "--allow", "SEED", *allowed],
            capture_output=True, text=True)
        assert r2.returncode == 0, f"{arm}_rep2: {r2.stdout}{r2.stderr}"


# ---------------- SO(3) + FK training objective ----------------


def _so3fk_batch(n=4, noise=0.02, seed=0):
    """A GT 51D row and a perturbed prediction of it, at a plausible hand depth."""
    torch.manual_seed(seed)
    y = torch.randn(n, 51) * 0.05
    y[:, 2] += 0.4
    pred = (y + torch.randn(n, 51) * noise).requires_grad_(True)
    return pred, y


def test_axis_angle_to_quaternion_matches_rodrigues():
    """Agrees with the closed form on generic rotations, and stays unit norm."""
    torch.manual_seed(0)
    aa = torch.randn(64, 3) * 1.5
    q = axis_angle_to_quaternion(aa)
    theta = aa.norm(dim=-1, keepdim=True)
    ref = torch.cat([torch.cos(theta / 2), aa / theta * torch.sin(theta / 2)], dim=-1)
    assert torch.allclose(q, ref, atol=1e-6)
    assert torch.allclose(q.norm(dim=-1), torch.ones(64), atol=1e-6)


def test_axis_angle_to_quaternion_is_finite_at_identity():
    """r = 0 is where an untrained head sits; a plain norm would give NaN grads."""
    aa = torch.zeros(3, 3, requires_grad=True)
    q = axis_angle_to_quaternion(aa)
    assert torch.allclose(q, torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(3, 4))
    q.sum().backward()
    assert torch.isfinite(aa.grad).all()


def test_rotation_term_is_antipodally_invariant(cfg_so3fk):
    """q and -q are the same rotation, so 1 - cos^2 must not tell them apart."""
    m = MNISTModel(cfg_so3fk).eval()
    pred, y = _so3fk_batch()
    base = m._compute_loss(pred, y, None)[1]["loss_rot"]

    def full_turn(aa):
        """Same rotation, angle + 2pi, which flips the sign of the quaternion."""
        theta = aa.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return aa * (1.0 + 2.0 * torch.pi / theta)

    # The turn has to be applied to the decoded rotations: the 45 local dims are a
    # residual to hands_mean, so turning them in place would not be the same rotation.
    dec = decode_to_mano_inputs(
        pred.detach(), m.pose_repr, m.mano.hands_components, m.mano.hands_mean
    )
    flipped = pred.detach().clone()
    flipped[:, m.slices.root] = full_turn(dec["global_orient"])
    flipped[:, m.slices.local] = (
        full_turn(dec["local_full_aa"].view(-1, 15, 3)).view(-1, 45) - m.mano.hands_mean
    )
    got = m._compute_loss(flipped.requires_grad_(True), y, None)[1]["loss_rot"]
    assert torch.allclose(base, got, atol=1e-4)


def test_so3fk_loss_equals_the_specified_sum(cfg_so3fk):
    """Total is exactly 1*L_rot + 1*L_trans + 2*L_FK, with the weights from the yaml."""
    m = MNISTModel(cfg_so3fk).eval()
    assert (m.loss_type, m.rot_weight, m.trans_weight, m.fk_weight) == \
        ("so3_trans_fk", 1.0, 1.0, 2.0)
    pred, y = _so3fk_batch()
    loss, parts = m._compute_loss(pred, y, None)
    want = parts["loss_rot"] + parts["loss_trans"] + 2.0 * parts["loss_fk"]
    assert torch.allclose(loss, want, atol=1e-7)
    # joint_error_mm is the same distance in millimetres, so MANO joints are metres.
    assert torch.allclose(parts["joint_error_mm"], parts["loss_fk"].detach() * 1000.0)


def test_so3fk_loss_is_zero_at_the_ground_truth(cfg_so3fk):
    """GT decodes through the same MANO, so a perfect prediction costs nothing."""
    m = MNISTModel(cfg_so3fk).eval()
    _, y = _so3fk_batch()
    loss, parts = m._compute_loss(y.clone().requires_grad_(True), y, None)
    assert float(loss) >= 0.0 and float(loss) < 1e-6
    for k in ("loss_rot", "loss_trans", "loss_fk"):
        assert float(parts[k]) < 1e-6


def test_so3fk_fk_term_is_root_relative(cfg_so3fk):
    """Shifting the prediction bodily must move L_trans only, never L_FK."""
    m = MNISTModel(cfg_so3fk).eval()
    pred, y = _so3fk_batch()
    a = m._compute_loss(pred, y, None)[1]
    shifted = pred.detach().clone()
    shifted[:, m.slices.transl] += 0.05
    b = m._compute_loss(shifted.requires_grad_(True), y, None)[1]
    assert torch.allclose(a["loss_fk"], b["loss_fk"], atol=1e-6)
    assert torch.allclose(a["loss_rot"], b["loss_rot"], atol=1e-6)
    assert float(b["loss_trans"]) > float(a["loss_trans"])


def test_so3fk_gradient_reaches_the_whole_trunk(cfg_so3fk):
    """No detach on the prediction branch, and each term drives the right outputs."""
    m = MNISTModel(cfg_so3fk).eval()
    pred, y = _so3fk_batch()
    _, parts = m._compute_loss(pred, y, None)
    for term, want in (("loss_trans", 3), ("loss_rot", 48), ("loss_fk", 51)):
        g = torch.autograd.grad(parts[term], pred, retain_graph=True)[0]
        assert torch.isfinite(g).all()
        assert int((g != 0).any(dim=0).sum()) == want


def test_legacy_loss_is_logged_but_not_trained(cfg_so3fk):
    """The old 51D MSE stays available for comparison, detached from the objective."""
    m = MNISTModel(cfg_so3fk).eval()
    pred, y = _so3fk_batch()
    loss, parts = m._compute_loss(pred, y, None)
    assert not parts["legacy_loss_51d"].requires_grad
    assert float(parts["legacy_loss_51d"]) > 0.0
    # The total must be reproducible from the three terms alone.
    assert torch.allclose(
        loss, parts["loss_rot"] + parts["loss_trans"] + 2.0 * parts["loss_fk"], atol=1e-7
    )


def test_so3fk_rejects_unknown_loss_type(cfg_so3fk):
    cfg = {**cfg_so3fk, "LOSS": {**cfg_so3fk["LOSS"], "TYPE": "nope"}}
    with pytest.raises(ValueError, match="LOSS.TYPE"):
        MNISTModel(cfg)


def test_so3fk_config_diff_is_loss_only():
    """The arm must differ from its control in the LOSS block and nowhere else."""
    import subprocess

    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "diff_configs.py"),
            str(ROOT / "configs" / "eventhands_track_render51_ch_both.yaml"),
            str(ROOT / "configs" / "eventhands_track_render51_so3fk.yaml"),
            "--allow", "LOSS.TYPE", "LOSS.ROT_WEIGHT", "LOSS.TRANS_WEIGHT",
            "LOSS.FK_WEIGHT",
            "TRAIN.OUTPUT_DIR", "TRAIN.RUN_NAME", "EVAL.OUTPUT_DIR",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
