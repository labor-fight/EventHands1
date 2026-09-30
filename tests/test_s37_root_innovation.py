#!/usr/bin/env python3
"""S37 root innovation (docs/S37_ROOT_INNOVATION_PREREG.md): the contract tests.

What must hold:

  * the innovation head is exactly zero when the residual features are zero, whatever the geometry
    (prev's lever arms can only multiply a measurement, never move the root on their own);
  * the chamfer SDF agrees with scipy's exact EDT inside the 16 px band, and the residual features
    point the right way;
  * at initialisation the arm reproduces the warm-start S37 output bitwise (zero-init geometry map);
  * `PREV_MLP_ROOT: false` removes exactly `prev_mlp(prev)[:, :6]` and nothing else;
  * `TRAIN.TRAINABLE_PREFIXES` leaves only the root heads trainable and in the optimiser;
  * an event-free packet still returns `prev` bitwise;
  * the two stage-1 configs differ from S37 only in the arm keys.
"""
from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "tests"))

from model import MNISTModel  # noqa: E402
from semkine.root_innovation import (RootInnovationHead, innovation_features,  # noqa: E402
                                     silhouette_sdf, splat_silhouette)
from test_s37_routed_readout import _cfg, _events, _packet, _projected, _test_prev  # noqa: E402


def _innov_cfg(prev_mlp_root=True, trainable=None):
    cfg = _cfg()
    cfg["MODEL"].update({"ROOT_INNOVATION": True, "ROOT_INNOV_HIDDEN": 8, "ROOT_INNOV_WIDTH": 4,
                         "PREV_MLP_ROOT": prev_mlp_root})
    if trainable is not None:
        cfg["TRAIN"]["TRAINABLE_PREFIXES"] = trainable
    return cfg


def _on_hand_packet(model, n=24):
    prev = _test_prev()
    u, v, _ = _projected(model, prev)
    ok = (u >= 0) & (u < 240) & (v >= 0) & (v < 180)
    idx = ok.nonzero().squeeze(1)[:n]
    rows = [(0, float(u[i].round()) + 3.0, float(v[i].round()), 0.001 * k, k % 2) for k, i in enumerate(idx)]
    rows += [(0, 2.0, 2.0, 0.03, 1)]
    pk = _packet(_events(rows))
    pk.prev_state = prev
    return pk


# ------------------------------------------------------------------ head
def test_zero_residual_gives_exactly_zero_whatever_the_geometry():
    torch.manual_seed(0)
    head = RootInnovationHead(hidden=8, width=4)
    torch.nn.init.normal_(head.geo.weight)
    q = torch.zeros(3, 17, 7)
    g = torch.randn(3, 17, 7) * 10
    assert torch.equal(head(q, g), torch.zeros(3, 6))
    q[1, 4, 0] = 0.5
    out = head(q, g)
    assert torch.equal(out[0], torch.zeros(6)) and torch.equal(out[2], torch.zeros(6))


def test_geometry_map_starts_at_zero():
    head = RootInnovationHead()
    assert torch.equal(head.geo.weight, torch.zeros_like(head.geo.weight))
    assert head.psi1.bias is None and head.psi2.bias is None and head.geo.bias is None


# ------------------------------------------------------------------ features
def test_chamfer_sdf_matches_the_exact_edt_inside_the_band():
    from scipy import ndimage
    H, W = 180, 240
    yy, xx = np.mgrid[:H, :W]
    m = (xx - 120) ** 2 + (yy - 90) ** 2 <= 30 ** 2
    sdf, g = silhouette_sdf(torch.from_numpy(m)[None])
    ref = ndimage.distance_transform_edt(~m) - ndimage.distance_transform_edt(m)
    band = (np.abs(ref) >= 2) & (np.abs(ref) <= 16)
    got = sdf[0].numpy()
    assert np.all(np.sign(got[band]) == np.sign(ref[band]))
    assert np.max(np.abs(got[band] - ref[band]) / np.abs(ref[band])) < 0.10
    # outside the disc on its right the gradient points right (+x), on its left it points left
    assert float(g[0, 0, 90, 160]) > 0.9 and float(g[0, 0, 90, 80]) < -0.9


def test_empty_frame_saturates_positive():
    sdf, g = silhouette_sdf(torch.zeros(1, 180, 240, dtype=torch.bool))
    assert torch.all(sdf > 0) and torch.all(g == 0)


def test_events_right_of_the_silhouette_give_a_rightward_residual():
    B, V = 1, 64
    ang = torch.linspace(0, 2 * np.pi, V)
    r = torch.linspace(0, 20, 8)
    uv = torch.stack([120 + r[:, None] * torch.cos(ang[None]), 90 + r[:, None] * torch.sin(ang[None])], -1)
    uv = uv.reshape(1, -1, 2)
    N = 4
    px = torch.tensor([[146.0, 147.0, 148.0, 0.0]])
    py = torch.tensor([[90.0, 91.0, 89.0, 0.0]])
    mask = torch.tensor([[True, True, True, False]])
    a = torch.zeros(B, N, 16)
    a[0, :3, 5] = 1.0
    vid = torch.zeros(B, N, dtype=torch.long)
    q = innovation_features(px, py, mask, a, vid, uv, 180, 240)
    assert q.shape == (1, 17, 7)
    assert float(q[0, 5, 0]) > 0.2 and abs(float(q[0, 5, 1])) < 0.1     # s*n points +x
    assert float(q[0, 5, 2]) > 0                                         # outside: s > 0
    assert torch.all(q[0, [j for j in range(16) if j != 5]] == 0)        # unreached parts are zero
    assert float(q[0, 16, 6]) == 1.0 and float(q[0, 5, 6]) == pytest.approx(1.0)


def test_splat_ignores_vertices_outside_the_frame():
    uv = torch.tensor([[[-5.0, 10.0], [300.0, 10.0], [50.0, 60.0]]])
    m = splat_silhouette(uv, 180, 240)
    assert bool(m[0, 60, 50]) and int(m.sum()) < 30


# ------------------------------------------------------------------ model
def test_arm_at_init_reproduces_the_warm_start_bitwise():
    torch.manual_seed(0)
    base = MNISTModel(_cfg()).eval()
    arm = MNISTModel(_innov_cfg()).eval()
    missing, unexpected = arm.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("root_innov_head.") for k in missing)
    pk = _on_hand_packet(base)
    with torch.no_grad():
        assert torch.equal(base.forward_packet(pk), arm.forward_packet(pk))


def test_innovation_reaches_the_root_only_and_its_ablation_silences_it():
    torch.manual_seed(0)
    arm = MNISTModel(_innov_cfg()).eval()
    torch.nn.init.normal_(arm.root_innov_head.geo.weight)
    pk = _on_hand_packet(arm)
    with torch.no_grad():
        out = arm.forward_packet(pk)
        arm.ablate_innovation = True
        try:
            abl = arm.forward_packet(pk)
        finally:
            arm.ablate_innovation = False
        zero = copy.deepcopy(arm)
        torch.nn.init.zeros_(zero.root_innov_head.geo.weight)
        ref = zero.forward_packet(pk)
    assert torch.equal(abl, ref), "ablation must equal a zero innovation path"
    assert not torch.equal(out[:, :6], abl[:, :6]), "the innovation must reach the root"
    assert torch.equal(out[:, 6:], abl[:, 6:]), "the innovation must not reach the fingers"


def test_prev_mlp_root_off_removes_exactly_the_root_rows():
    torch.manual_seed(0)
    on = MNISTModel(_innov_cfg(prev_mlp_root=True)).eval()
    torch.nn.init.normal_(on.prev_mlp[2].weight, std=0.1)
    off = MNISTModel(_innov_cfg(prev_mlp_root=False)).eval()
    off.load_state_dict(on.state_dict())
    pk = _on_hand_packet(on)
    with torch.no_grad():
        a, b = on.forward_packet(pk), off.forward_packet(pk)
        pm = on.prev_mlp(pk.prev_state)
    assert torch.allclose(a[:, :6] - b[:, :6], pm[:, :6], atol=1e-6)
    assert torch.equal(a[:, 6:], b[:, 6:])


def test_only_the_root_heads_train():
    m = MNISTModel(_innov_cfg(trainable=["root_head.", "root_innov_head."]))
    train = {n for n, p in m.named_parameters() if p.requires_grad}
    assert train and all(n.startswith(("root_head.", "root_innov_head.")) for n in train)
    assert any(n.startswith("root_innov_head.") for n in train) and any(n.startswith("root_head.") for n in train)
    opt = m.configure_optimizers()
    opt = opt["optimizer"] if isinstance(opt, dict) else opt
    n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert n_opt == sum(p.numel() for n, p in m.named_parameters() if n in train)


def test_empty_packet_returns_prev_bitwise():
    m = MNISTModel(_innov_cfg(prev_mlp_root=False)).eval()
    torch.nn.init.normal_(m.root_innov_head.geo.weight)
    pk = _packet(_events([]), n_packets=2)
    pk.prev_state = torch.randn(2, 51)
    assert torch.equal(m.forward_packet(pk), pk.prev_state)


# ------------------------------------------------------------------ configs
@pytest.mark.parametrize("seed", [3407, 3408])
def test_config_differs_from_s37_only_in_the_arm_keys(seed):
    allow = ["MODEL.ROOT_INNOVATION", "MODEL.ROOT_INNOV_HIDDEN", "MODEL.ROOT_INNOV_WIDTH",
             "MODEL.PREV_MLP_ROOT", "MODEL.INIT_FROM", "TRAIN.MAX_STEPS", "TRAIN.VAL_CHECK_INTERVAL",
             "TRAIN.SAVE_EVERY_N_STEPS", "TRAIN.TRAINABLE_PREFIXES", "TRAIN.OUTPUT_DIR",
             "TRAIN.RUN_NAME", "EVAL.OUTPUT_DIR"] + (["SEED"] if seed != 3407 else [])
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "diff_configs.py"),
         str(ROOT / "configs" / "semkine" / "s37_routed_s3407.yaml"),
         str(ROOT / "configs" / "semkine" / f"s37_rootinnov_s{seed}.yaml"), "--allow", *allow],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_stage1_model_keys_are_s37_plus_the_innovation_head():
    import json
    from config import load_config
    cfg = load_config(str(ROOT / "configs/semkine/s37_rootinnov_s3407.yaml"))
    m = MNISTModel(cfg)
    sel = ROOT / "outputs/semkine/s37_routed_s3407/selection_val_core_step50.json"
    if not sel.exists():
        pytest.skip("S37 zgz run not present on this machine")
    ckpt = ROOT / json.loads(sel.read_text())["selected"]["ckpt"]
    assert ckpt == ROOT / cfg["MODEL"]["INIT_FROM"], "stage 1 must warm-start from the selected S37 step"
    sd = torch.load(ckpt, map_location="cpu")["state_dict"]
    extra = set(m.state_dict()) - set(sd)
    assert set(sd) <= set(m.state_dict()) and extra and all(k.startswith("root_innov_head.") for k in extra)
