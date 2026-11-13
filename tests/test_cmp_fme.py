"""CMP extension arm `fme` (docs/EXPERIMENT_RECORDS.md#cmp-amend8-20261110): FastMETRO with event input, against the
official EvRGBHand code (src/configs/config/fastmetro_ev.yaml, src/modeling/Loss.py, train.py at bc6cc3b).

CPU only (CUDA_VISIBLE_DEVICES=""). Official definitions are restated here where they are compared.
"""
from __future__ import annotations

import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.cmp import arms as A  # noqa: E402
from tools.cmp import ext_fme as X  # noqa: E402


def config(pretrained=False):
    cfg = copy.deepcopy(A.build_config("fme", 3407))
    cfg["CMP"]["FME"]["NET"]["PRETRAINED"] = pretrained            # the tests need no ImageNet weights
    return cfg


@pytest.fixture(scope="module")
def model():
    from semkine.cmp_models import model_class
    torch.manual_seed(0)
    return model_class("fme")(config()).eval()


def states(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = torch.randn(n, 51, generator=g) * 0.3
    y[:, :3] = torch.tensor([0.0, 0.0, 0.55]) + torch.randn(n, 3, generator=g) * 0.05
    return y, torch.randn(n, 10, generator=g) * 0.5


def test_registered_configs_unchanged():
    assert len(A.write_configs("formal", [a for a in A.ARMS if a not in A.EXT])) == 30
    assert A.EXT["fme"] is X and A.ARMS["fme"] is X.ARMS["fme"]


def test_config_is_the_official_recipe():
    c = A.build_config("fme", 3407)
    t, f = c["TRAIN"], c["CMP"]["FME"]
    # fastmetro_ev.yaml: per_gpu_batch_size 32, lr 1e-4, lr_backbone 1e-5 for 'backbone.layer', amp false
    assert t["BATCH_SIZE_PER_GPU"] * t["ACCUMULATE_GRAD_BATCHES"] == 32 and t["LR"] == 1e-4 and t["PRECISION"] == 32
    assert f["OPT"] == {"LR_BACKBONE": 1e-5, "BACKBONE_NAMES": ["backbone.layer"], "WEIGHT_DECAY": 1e-4,
                        "BETAS": [0.9, 0.999], "CLIP_NORM": 1.0, "ETA_MIN": 1e-6}
    assert f["NET"] == {"BACKBONE": "resnet34", "PRETRAINED": True, "HIDDEN_DIM_1": 512, "DIM_FEEDFORWARD_1": 2048,
                        "HIDDEN_DIM_2": 128, "DIM_FEEDFORWARD_2": 512, "ENC_LAYERS": 3, "DEC_LAYERS": 3, "NHEAD": 8,
                        "DROPOUT": 0.1}
    assert f["LOSS"] == {"VERTICES": 100.0, "VERTICES_SUB": 100.0, "JOINTS_3D": 500.0, "JOINTS_3D_FROM_MESH": 500.0,
                         "BETA": 0.01}
    assert f["PHOTOMETRY"] == {"GAUSS_P": 0.1, "GAUSS_VAR": [0.05, 0.1], "SALT_PEPPER_P": 0.1, "SALT_PEPPER_RATE": 0.05}
    # the frame arms' sample budget, a fixed 100 ms window, the official geometric ranges and nothing of B2's contract
    assert t["MAX_STEPS"] * 32 == A.ARMS["eh"]["steps"] * A.ARMS["eh"]["batch"] * A.ARMS["eh"]["accumulate"]
    assert t["MAX_STEPS"] % t["SAVE_EVERY_N_STEPS"] == 0
    assert c["DATA"]["WINDOW_MIN"] == c["DATA"]["WINDOW_MAX"] == c["CMP"]["EVAL"]["WINDOW_MS"] == 100
    d = c["AUG"]["DOMRAND"]
    assert abs(np.deg2rad(d["ROLL_DEG"]) - 0.4) < 1e-4 and (d["SCALE_MIN"], d["SCALE_MAX"]) == (0.9, 1.1)
    assert d["SHIFT_PX"] == 0.0 and d["KEEP_MIN"] == d["KEEP_MAX"] == 1.0 and d["HOT_PIXEL_RATE"] == 0.0
    assert not (c["AUG"]["SPEED_AUG"] or c["AUG"]["POLARITY_FLIP"] or c["AUG"]["PIXEL_POLARITY_SWAP"])
    s = A.build_config("fme", 3407, "smoke")
    assert s["TRAIN"]["MAX_STEPS"] == A.SMOKE_STEPS and s["CMP"]["FME"] == f


def test_network_and_frame(model):
    x = torch.rand(2, 180, 240, 2)
    fr = model.frame(x)
    assert fr.shape == (2, 192, 256, 3)                              # official layout: polarity 0, polarity 1, zeros
    assert torch.equal(fr[:, 6:186, 8:248, :2], x) and float(fr[..., 2].abs().max()) == 0.0
    assert float(fr[:, :6].abs().max()) == float(fr[:, 186:].abs().max()) == float(fr[:, :, :8].abs().max()) == 0.0
    with torch.no_grad():
        out = model.predict(x)
    assert out["pred_3d_joints"].shape == (2, 21, 3) and out["pred_vertices_sub"].shape == (2, 195, 3)
    assert out["pred_vertices"].shape == (2, 778, 3) and all(torch.isfinite(v).all() for v in out.values())
    net = model.net
    assert type(net.backbone.layer1[0][0].bn1).__name__ == "FrozenBatchNorm2d"      # the official event backbone
    assert net.attention_mask.shape == (216, 216) and not net.attention_mask[:21].any()
    assert sum(p.numel() for p in model.parameters()) == 45385451


def test_joint_regressor_and_downsampling(model):
    y, betas = states(64)
    verts, sub, joints = model.targets(y, betas)
    reg = model.joints_from_mesh(verts)
    tips = [4, 8, 12, 16, 20]                                        # the five tips (OpenPose order) are vertices
    assert float((reg[:, tips] - joints[:, tips]).abs().max()) < 1e-6
    # the wrist and the fifteen others are FK joints, which the regressor only approximates on a posed mesh (hence the
    # main field of the reading scores the joint tokens; tools/cmp/ext_fme.py)
    assert float((reg[:, 0] - joints[:, 0]).norm(dim=-1).max()) < 5e-3
    dev = (reg - joints).norm(dim=-1) * 1000
    assert 0.1 < float(dev.mean()) < 6.0
    down = model.down
    assert down.shape == (195, 778) and bool(((down == 1).sum(1) == 1).all()) and float(down.sum()) == 195.0
    assert torch.equal(sub, verts[:, down.argmax(1)])


def test_metric_definitions_equal_the_engine_score(model):
    from mano_layer import ManoLayer
    from semkine import eval_track as ET
    from tools.xcl.evaluate import score
    cfg = model.cfg
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).eval()
    pred, _ = states(32, 1)
    target, betas = states(32, 2)
    beta = betas[:1]
    dev = torch.device("cpu")
    ref = score(mano, pred.numpy(), target.numpy(), beta, dev)
    pj, pv = ET.mano_fk(mano, pred.numpy(), beta, dev)
    gj, gv = ET.mano_fk(mano, target.numpy(), beta, dev)
    m = X.score_mesh(pj, pv, gj, gv, model.jreg.numpy())
    for key in ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm"):
        assert np.array_equal(m[key], ref[key]), key
    assert set(m) == set(X.METRICS) and all(v.shape == (32,) for v in m.values())
    same = X.score_mesh(gj, gv, gj, gv, model.jreg.numpy())         # the label's own mesh
    assert all(float(v.max()) == 0.0 for k, v in same.items() if k != "mpjpe_ra_mesh_mm")
    assert 0.1 < float(same["mpjpe_ra_mesh_mm"].mean()) < 6.0       # the regressor's distance from the FK joints


def official_loss(out, gt_vertices, gt_sub, gt_joints, jreg, w):
    """src/modeling/Loss.py `get_loss_mano` for an output without 'pred_cam' / 'joints_hm', every sample valid."""
    crit = torch.nn.SmoothL1Loss(reduction="none", beta=0.01)
    gt_root = gt_joints[:, 0, :]
    gt_3d_joints_aligned = gt_joints - gt_root[:, None, :]
    gt_vertices_aligned = gt_vertices - gt_root[:, None, :]
    gt_vertices_sub_aligned = gt_sub - gt_root[:, None, :]
    pred_3d_joints_from_mesh = torch.einsum("bik,ji->bjk", [out["pred_vertices"], jreg])
    pred_3d_joints_root = out["pred_3d_joints"][:, :1, :]
    pred_3d_joints_from_mesh_root = pred_3d_joints_from_mesh[:, :1, :]
    pred_vertices_coarse_aligned = out["pred_vertices_sub"] - pred_3d_joints_from_mesh_root
    pred_vertices_fine_aligned = out["pred_vertices"] - pred_3d_joints_from_mesh_root
    pred_3d_joints_aligned = out["pred_3d_joints"] - pred_3d_joints_root
    pred_3d_joints_from_mesh_aligned = pred_3d_joints_from_mesh - pred_3d_joints_from_mesh_root
    loss_3d_joints = crit(gt_3d_joints_aligned, pred_3d_joints_aligned).mean()
    loss_3d_joints_reg = crit(gt_3d_joints_aligned, pred_3d_joints_from_mesh_aligned).mean()
    loss_vertices = crit(gt_vertices_aligned, pred_vertices_fine_aligned).mean()
    loss_vertices_sub = crit(gt_vertices_sub_aligned, pred_vertices_coarse_aligned).mean()
    return (w["vertices"] * loss_vertices + w["vertices_sub"] * loss_vertices_sub + w["3d_joints"] * loss_3d_joints
            + w["3d_joints_from_mesh"] * loss_3d_joints_reg)


def test_loss_is_the_official_expression(model):
    y, betas = states(8, 3)
    gv, gs, gj = model.targets(y, betas)
    g = torch.Generator().manual_seed(4)
    out = {"pred_3d_joints": gj + torch.randn(8, 21, 3, generator=g) * 0.02,
           "pred_vertices_sub": gs + torch.randn(8, 195, 3, generator=g) * 0.02,
           "pred_vertices": gv + torch.randn(8, 778, 3, generator=g) * 0.02}
    loss, parts = model.mesh_loss(out, y, betas)
    ref = official_loss(out, gv, gs, gj, model.jreg, {"vertices": 100.0, "vertices_sub": 100.0, "3d_joints": 500.0,
                                                      "3d_joints_from_mesh": 500.0})
    assert torch.isfinite(loss) and abs(float(loss) - float(ref)) < 1e-5 * float(ref)
    # every term is aligned to a wrist, so the place of a perfect prediction costs nothing: the joint tokens exactly,
    # the mesh up to the regressed wrist's distance from the FK wrist (the official alignment of the two sides)
    shift = torch.tensor([0.3, -0.2, 0.1])
    perfect = {"pred_3d_joints": gj + shift, "pred_vertices_sub": gs - shift, "pred_vertices": gv - shift}
    _, p = model.mesh_loss(perfect, y, betas)
    assert float(p["loss_3d_joints"]) < 1e-9 and float(p["joint_error_mm"]) < 1e-3
    wrist = (model.joints_from_mesh(gv)[:, 0] - gj[:, 0]).norm(dim=-1) * 1000
    assert abs(float(p["vertex_error_mm"]) - float(wrist.mean())) < 1e-3 and float(wrist.max()) < 5.0
    assert float(p["loss_vertices"]) < 1e-4 and float(p["loss_vertices_sub"]) < 1e-4


def test_event_photometry(model):
    torch.manual_seed(5)
    x = (torch.rand(6, 180, 240, 2) > 0.9).float() * torch.rand(6, 180, 240, 2)
    off = dict(model.photo)
    try:
        model.photo = {**off, "GAUSS_P": 0.0, "SALT_PEPPER_P": 0.0}
        assert torch.equal(model.degrade(x.clone()), x)
        model.photo = {**off, "GAUSS_P": 1.0, "SALT_PEPPER_P": 0.0}
        g = model.degrade(x.clone())
        assert float(g.min()) >= 0.0 and float(g.max()) <= 1.0 and not torch.equal(g, x)
        assert float((g - x).abs().mean()) < 0.1                    # std U(0, 0.1), clipped to [0, 1]
        model.photo = {**off, "GAUSS_P": 0.0, "SALT_PEPPER_P": 1.0}
        s = model.degrade(x.clone())
        changed = s != x
        assert bool((s[changed] == 1).all())                        # only ones are written
        rate = float(changed.any(-1).float().mean())                # ~ the rate (a slot already at 1 does not change)
        assert 0.03 < rate < 0.06 and not bool((changed[..., 0] & changed[..., 1]).any())
    finally:
        model.photo = off


def test_optimizer_is_the_official_one(model):
    o = model.configure_optimizers()
    opt, sched = o["optimizer"], o["lr_scheduler"]["scheduler"]
    assert type(opt).__name__ == "AdamW" and o["lr_scheduler"]["interval"] == "step"
    main, back = opt.param_groups
    assert main["initial_lr"] == 1e-4 and back["initial_lr"] == 1e-5
    assert main["weight_decay"] == back["weight_decay"] == 1e-4 and main["betas"] == (0.9, 0.999)
    names = {id(p): n for n, p in model.named_parameters()}
    assert all("backbone.layer" in names[id(p)] for p in back["params"]) and len(back["params"]) == 36
    assert not any("backbone.layer" in names[id(p)] for p in main["params"])
    assert sum(p.numel() for g in opt.param_groups for p in g["params"]) == sum(p.numel() for p in model.parameters())
    t = model.cfg["TRAIN"]
    last = [f(int(t["MAX_STEPS"])) for f in sched.lr_lambdas]
    assert abs(last[0] * 1e-4 - 1e-6) < 1e-12 and abs(last[1] * 1e-5 - 1e-6) < 1e-12      # the official eta_min
    assert sched.lr_lambdas[0](int(t["WARMUP_STEPS"])) == 1.0 and sched.lr_lambdas[0](0) == 1.0 / t["WARMUP_STEPS"]


def test_adapter(model):
    class Net:
        pass
    net = Net()
    net.model = model
    ad = X.Adapter(net)
    with pytest.raises(RuntimeError):
        ad.step(None, 49, None)
    with torch.no_grad():
        y = ad.cost_step({"x": torch.rand(1, 180, 240, 2)}, None, {})
    assert y.shape == (1, 51)
    # 48 image tokens (6 x 8), 216 joint / vertex tokens; per attention (2 Lq + 2 Lk) d^2 + 2 Lq Lk d
    att = lambda d, lq, lk: (2 * lq + 2 * lk) * d * d + 2 * lq * lk * d  # noqa: E731
    want = sum(3 * att(d, 48, 48) + 3 * (att(d, 216, 216) + att(d, 216, 48)) for d in (512, 128))
    assert ad.extra_macs({}) == want
    assert X.Adapter.FORMS == ("main",) and X.EVAL["fme"]["WINDOW_MS"] == 100
