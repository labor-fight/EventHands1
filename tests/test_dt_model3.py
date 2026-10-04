"""DT2 round, package C: five default-off features of the dense tracker (docs/DT2_PREREG.md section 2) and their arms.

  LOSS.ROOT_ROT_WEIGHT  w: so3_trans_fk's L_rot = (w d_root + sum of the 15 finger d) / (w + 15); w == 1 is the original path
  TRAIN.WEIGHT_DECAY    wd > 0: torch.optim.AdamW, Conv / Linear weights decay, BatchNorm parameters and every bias do not
  TRAIN.EMA_DECAY       d > 0: EMA of parameters + BatchNorm running statistics; every saved checkpoint's state_dict is the EMA
                        model, the raw weights are in raw_state_dict; --resume continues from the raw weights with the EMA
                        restored; validation runs on the EMA weights
  MODEL.ROOT_HEAD       spatial | anchor: a zero-initialised second root-rotation readout added to the fc's root row
  MODEL.RENDER_FP32     the previous-state render in fp32 under bf16 autocast

Pinned here (letters follow the task text):

  (a) defaults: with every new key at its default (absent or written out) the model, its optimiser, its loss and a real
      Lightning `fit` are the pre-edit model's (outputs/dt/ref/model_orig2.py, sha256 pinned) bit for bit; the checkpoints
      carry no EMA entries; the guards raise ValueError instead of silently doing nothing.
  (b) ROOT_ROT_WEIGHT: equals an independent scipy computation, w -> 0 / w -> inf limits, gradient of the root columns.
  (c) WEIGHT_DECAY: parameter groups (every trainable parameter exactly once; decay = Conv / Linear weights), the decoupled
      decay arithmetic of one AdamW step, frozen parameters stay out, Adam when 0.
  (d) EMA: the update arithmetic and its warm-up, exactness for constant weights, checkpoint layout (also from inside the
      validation loop, where the live tensors hold the EMA), a real Lightning `fit` on CPU: saved state_dict = the EMA,
      load_from_checkpoint returns the EMA weights (evaluation) while `fit(ckpt_path=...)` returns the raw weights with the
      EMA continuing -- a run split by a resume equals the uninterrupted run bit for bit --, `validate(ckpt_path=...)` keeps
      the EMA, validation uses the EMA weights and restores the raw ones.
  (e) ROOT_HEAD: initial model == baseline (same seed, bit for bit, eval / train / bf16), exact parameter counts, additive
      only on the root columns, spatial layout is read (the GAP row is permutation invariant), anchor geometry (stride-8 grid
      derived from the stem, per-sample K', out-of-bounds and behind-camera joints), gradients, guards.
  (f) RENDER_FP32: under bf16 autocast the render equals the plain fp32 render bit for bit (CPU and, when DT3_DEVICE=cuda,
      GPU), where the default render differs on GPU.
  (g) the 14 generated configs: arm = parent + exactly its factor, names, parameter counts.

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python -m pytest tests/test_dt_model3.py -q
    DT3_DEVICE=cuda CUDA_VISIBLE_DEVICES=7 python -m pytest tests/test_dt_model3.py -q -k "render_fp32 or anchor"
"""
from __future__ import annotations

import copy
import functools
import hashlib
import importlib.util
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from scipy.spatial.transform import Rotation as Rot
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "model")]
import backbones as BB                                                       # noqa: E402
from config import load_config                                               # noqa: E402
from model import AnchorRootHead, MNISTModel, SpatialRootHead                # noqa: E402

REF = REPO / "outputs" / "dt" / "ref"
ORIG_MODEL = REF / "model_orig2.py"
ORIG_MODEL_SHA256 = "758785e65c5af032c262001807477c457efca66226f4f75f79364fe74fd56c89"
DT = REPO / "configs" / "dt"
BASE = DT / "dt_dz_l3.yaml"
ARMS7 = ("ema", "wd", "reg", "rootw4", "rooth", "rootanc", "r32")
H, W = 180, 240
BASE_K = torch.tensor([[603.4507, 0.0, 325.09183], [0.0, 602.95654, 242.09796], [0.0, 0.0, 1.0]])
DEV = torch.device("cuda" if os.environ.get("DT3_DEVICE") == "cuda" and torch.cuda.is_available() else "cpu")
BASE_PARAMS = 2_802_645
SPATIAL_PARAMS, ANCHOR_PARAMS = 183_395, 705_283


# ----------------------------------------------------------------------------------------------------- helpers
@functools.lru_cache(maxsize=None)
def _load(path):
    return load_config(Path(path))


def _cfg(path=BASE, model=None, loss=None, train=None):
    cfg = copy.deepcopy(_load(path))
    cfg["MODEL"].update(model or {})
    cfg["LOSS"].update(loss or {})
    cfg["TRAIN"].update(train or {})
    return cfg


def _build(cls, cfg, seed=1234):
    torch.manual_seed(seed)
    return cls(copy.deepcopy(cfg))


def _randomize(m, seed=7):
    """What a trained model has and a fresh one does not: prev_mlp's zero-initialised last layer, BatchNorm statistics that
    are not the identity (the heads' zero last layers stay zero: that is what the initial-equivalence tests check)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        if hasattr(m, "prev_mlp"):
            for p in m.prev_mlp.parameters():
                p.copy_(0.05 * torch.randn(p.shape, generator=g))
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.running_mean.copy_(0.1 * torch.randn(mod.running_mean.shape, generator=g))
                mod.running_var.copy_(1.0 + 0.2 * torch.rand(mod.running_var.shape, generator=g))
    return m


def _nudge_head(m, seed=11, scale=0.02):
    """Give the zero-initialised last layer of the root head non-zero weights (what training does in its first steps)."""
    g = torch.Generator().manual_seed(seed)
    last = m.root_readout.fc if m.root_head_kind == "spatial" else m.root_readout.fc2
    with torch.no_grad():
        last.weight.copy_(scale * torch.randn(last.weight.shape, generator=g))
        last.bias.copy_(scale * torch.randn(last.bias.shape, generator=g))
    return m


def _lnes(B, seed=0, empty=()):
    g = torch.Generator().manual_seed(seed)
    x = (torch.rand(B, H, W, 2, generator=g) < 0.12).float() * torch.rand(B, H, W, 2, generator=g)
    for i in empty:
        x[i] = 0.0
    return x


def _prev(B, seed=0, norm=2.3):
    g = torch.Generator().manual_seed(seed)
    p = torch.zeros(B, 51)
    p[:, :2] = 0.05 * torch.randn(B, 2, generator=g)
    p[:, 2] = 0.55 + 0.03 * torch.randn(B, generator=g)
    axis = torch.randn(B, 3, generator=g)
    p[:, 3:6] = axis / axis.norm(dim=-1, keepdim=True) * norm
    p[:, 6:] = 0.3 * torch.randn(B, 45, generator=g)
    return p


def _betas(B, seed=3):
    return 0.3 * torch.randn(B, 10, generator=torch.Generator().manual_seed(seed))


def _fwd(m, x, p, bf16=False, **kw):
    dev = next(m.parameters()).device
    with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=bf16):
        return m(x.to(dev), p.to(dev), **{k: v.to(dev) for k, v in kw.items()})


def _assert_same_state(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb)
    for k in sa:
        assert sa[k].shape == sb[k].shape and sa[k].dtype == sb[k].dtype, k
        assert torch.equal(sa[k], sb[k]), k


def _n_params(m):
    return sum(p.numel() for p in m.parameters())


class _Synth(Dataset):
    """Deterministic legacy-tuple samples (lnes, prev, target, betas, K): the target is the prev state plus a small step."""

    def __init__(self, n, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.x = [_lnes(1, seed=100 + i)[0] for i in range(n)]
        self.prev = [_prev(1, seed=200 + i)[0] for i in range(n)]
        self.y = [p + 0.02 * torch.randn(51, generator=g) for p in self.prev]
        self.betas = [_betas(1, seed=300 + i)[0] for i in range(n)]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.prev[i], self.y[i], self.betas[i], BASE_K.clone() * torch.tensor([1.0, 1.0, 1.0])


def _fit_cfg(**kw):
    """dt_dz_l3's architecture with a 8-step schedule (the LR schedule reads TRAIN.MAX_STEPS)."""
    train = {"LR": 4e-3, "WARMUP_STEPS": 2, "MAX_STEPS": 8}
    train.update(kw.pop("train", {}))
    return _cfg(train=train, **kw)


def _trainer(tmp, max_steps, every=2, callbacks=()):
    ck = ModelCheckpoint(dirpath=str(tmp), filename="c-{step}", monitor=None, save_last=True, save_top_k=-1,
                         every_n_train_steps=every)
    return pl.Trainer(accelerator="cpu", devices=1, max_steps=max_steps, precision=32, logger=False,
                      enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0,
                      accumulate_grad_batches=2, val_check_interval=4, check_val_every_n_epoch=None,
                      callbacks=[ck, *callbacks], default_root_dir=str(tmp)), ck


def _loaders():
    return (DataLoader(_Synth(8), batch_size=2, shuffle=False), DataLoader(_Synth(2, seed=9), batch_size=2, shuffle=False))


@pytest.fixture(scope="module")
def orig():
    """The pre-edit model.py (before the DT2 edit and before the model-acc edit's ACCEL / DISTILL), another module name."""
    if not ORIG_MODEL.exists():
        pytest.skip(f"{ORIG_MODEL} not available")
    assert hashlib.sha256(ORIG_MODEL.read_bytes()).hexdigest() == ORIG_MODEL_SHA256, "the reference copy was modified"
    spec = importlib.util.spec_from_file_location("model_orig2_t3", ORIG_MODEL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["model_orig2_t3"] = mod
    spec.loader.exec_module(mod)
    return mod


# ================================================================================================== (a) defaults
EXPLICIT = [{}, {"ROOT_HEAD": "none"}, {"RENDER_FP32": False}, {"ROOT_HEAD": "none", "RENDER_FP32": False}]


@pytest.mark.parametrize("explicit", EXPLICIT, ids=str)
def test_a_default_model_equals_the_original_bitwise(orig, explicit):
    """State dict, module tree, random draws, eval / train / bf16 forwards of the dense dt_dz_l3 tracker, with the new MODEL
    keys absent or written out at their defaults, the new LOSS / TRAIN keys at theirs."""
    a = _build(orig.MNISTModel, _cfg())
    rng_a = torch.get_rng_state()
    b = _build(MNISTModel, _cfg(model=explicit, loss={"ROOT_ROT_WEIGHT": 1.0}, train={"WEIGHT_DECAY": 0.0, "EMA_DECAY": 0.0}))
    assert torch.equal(rng_a, torch.get_rng_state())
    _assert_same_state(a, b)
    assert [n for n, _ in a.named_modules()] == [n for n, _ in b.named_modules()]
    assert (b.root_head_kind, b.render_fp32, b.root_rot_weight, b.weight_decay, b.ema_decay) == ("none", False, 1.0, 0.0, 0.0)
    a, b = _randomize(a).eval(), _randomize(b).eval()
    x, prev = _lnes(3, empty=(1,)), _prev(3)
    for B in (1, 3):
        for bf16 in (False, True):
            oa, ob = _fwd(a, x[:B], prev[:B], bf16), _fwd(b, x[:B], prev[:B], bf16)
            assert oa.dtype == ob.dtype and torch.equal(oa, ob), (B, bf16)


def test_a_the_module_overrides_no_lightning_optimization_hook():
    """No `optimizer_step` / `optimizer_zero_grad` override (Lightning would warn about gradient accumulation in every run),
    and an EMA-free optimiser carries no step hook."""
    for name in ("optimizer_step", "optimizer_zero_grad", "on_before_zero_grad"):
        assert getattr(MNISTModel, name) is getattr(pl.LightningModule, name), name
    opt = _build(MNISTModel, _fit_cfg()).configure_optimizers()["optimizer"]
    assert len(opt._optimizer_step_post_hooks) == 0
    ema = _build(MNISTModel, _fit_cfg(train={"EMA_DECAY": 0.9}))
    opt = ema.configure_optimizers()["optimizer"]
    assert len(opt._optimizer_step_post_hooks) == 1
    for p in ema.parameters():
        p.grad = torch.zeros_like(p)
    opt.step()
    opt.step()
    assert ema._ema_steps == 2                                         # one EMA update per `opt.step()`


def test_a_default_optimizer_is_plain_adam_and_the_schedule_is_the_originals(orig):
    a = _build(orig.MNISTModel, _fit_cfg()).configure_optimizers()
    b = _build(MNISTModel, _fit_cfg()).configure_optimizers()
    assert type(b["optimizer"]) is torch.optim.Adam and type(a["optimizer"]) is torch.optim.Adam
    assert len(b["optimizer"].param_groups) == 1 and b["optimizer"].param_groups[0]["weight_decay"] == 0
    for step in (0, 1, 2, 5, 8):
        assert a["lr_scheduler"]["scheduler"].lr_lambdas[0](step) == b["lr_scheduler"]["scheduler"].lr_lambdas[0](step)


def test_a_default_loss_and_gradients_equal_the_original():
    """w == 1 takes the original expression: written out or absent, the loss parts are the same tensors."""
    a, b = _build(MNISTModel, _cfg()).train(), _build(MNISTModel, _cfg(loss={"ROOT_ROT_WEIGHT": 1.0})).train()
    pred, y = _prev(4, seed=1) + 0.03 * torch.randn(4, 51, generator=torch.Generator().manual_seed(2)), _prev(4, seed=1)
    la, pa = a._compute_loss(pred, y, _betas(4))
    lb, pb = b._compute_loss(pred, y, _betas(4))
    assert torch.equal(la, lb) and all(torch.equal(pa[k], pb[k]) for k in pa)


def test_a_default_lightning_fit_equals_the_original_bitwise(orig, tmp_path):
    """A real CPU `fit` (accumulate 2, validation, checkpoints) of the pre-edit model and of the edited one with every new
    key at its default: the same weights after 4 optimiser steps (the `optimizer_step` override is transparent)."""
    finals, ckpts = [], []
    for i, cls in enumerate((orig.MNISTModel, MNISTModel)):
        torch.manual_seed(0)
        m = _randomize(cls(copy.deepcopy(_fit_cfg())))
        tr, ck = _trainer(tmp_path / f"d{i}", 4)
        tr.fit(m, *_loaders())
        finals.append({k: v.clone() for k, v in m.state_dict().items()})
        ckpts.append(torch.load(ck.last_model_path, map_location="cpu"))
    assert list(finals[0]) == list(finals[1]) and all(torch.equal(finals[0][k], finals[1][k]) for k in finals[0])
    for c in ckpts:
        assert "raw_state_dict" not in c and "ema" not in c
    assert all(torch.equal(ckpts[0]["state_dict"][k], ckpts[1]["state_dict"][k]) for k in ckpts[0]["state_dict"])


@pytest.mark.parametrize("model, loss, train, match", [
    ({}, {"ROOT_ROT_WEIGHT": 0.0}, {}, "ROOT_ROT_WEIGHT"),
    ({}, {"ROOT_ROT_WEIGHT": -2.0}, {}, "ROOT_ROT_WEIGHT"),
    ({}, {"ROOT_ROT_WEIGHT": float("nan")}, {}, "ROOT_ROT_WEIGHT"),
    ({}, {"ROOT_ROT_WEIGHT": True}, {}, "ROOT_ROT_WEIGHT"),
    ({}, {"ROOT_ROT_WEIGHT": "4"}, {}, "ROOT_ROT_WEIGHT"),
    ({}, {"ROOT_ROT_WEIGHT": 4.0, "TYPE": "mse_51d"}, {}, "ROOT_ROT_WEIGHT"),
    ({}, {}, {"WEIGHT_DECAY": -0.01}, "WEIGHT_DECAY"),
    ({}, {}, {"WEIGHT_DECAY": "0.01"}, "WEIGHT_DECAY"),
    ({}, {}, {"EMA_DECAY": 1.0}, "EMA_DECAY"),
    ({}, {}, {"EMA_DECAY": -0.1}, "EMA_DECAY"),
    ({}, {}, {"EMA_DECAY": None}, "EMA_DECAY"),
    ({"ROOT_HEAD": "temporal"}, {}, {}, "ROOT_HEAD"),
    ({"ROOT_HEAD": True}, {}, {}, "ROOT_HEAD"),
    ({"ROOT_HEAD": "spatial", "PREDICT_DELTA": False}, {}, {}, "ROOT_HEAD"),
    ({"ROOT_HEAD": "anchor", "PREV_RENDER": False}, {}, {}, "ROOT_HEAD"),
    ({"ROOT_HEAD": "spatial", "CNN_BACKBONE": "mobilenet_v3_small"}, {}, {}, "ROOT_HEAD"),
    ({"RENDER_FP32": 1}, {}, {}, "RENDER_FP32"),
    ({"RENDER_FP32": "true"}, {}, {}, "RENDER_FP32"),
    ({"RENDER_FP32": True, "PREV_RENDER": False}, {}, {}, "RENDER_FP32"),
])
def test_a_guards_raise_instead_of_silently_doing_nothing(model, loss, train, match):
    with pytest.raises(ValueError, match=match):
        _build(MNISTModel, _cfg(model=model, loss=loss, train=train))


def test_a_unknown_model_keys_still_raise_and_the_two_new_ones_are_accepted():
    with pytest.raises(ValueError, match="unknown MODEL keys"):
        _build(MNISTModel, _cfg(model={"ROOT_HEADS": "spatial"}))
    assert {"ROOT_HEAD", "RENDER_FP32"} <= set(MNISTModel.MODEL_KEYS)


def test_a_root_head_needs_a_layer_the_trunk_has():
    BB.BACKBONES["_t3_l2"] = lambda out_dim: BB.ScaledResNet(out_dim, layers=(2, 2, 0, 0))     # no layer3
    try:
        with pytest.raises(ValueError, match="layer3"):
            _build(MNISTModel, _cfg(model={"ROOT_HEAD": "spatial", "CNN_BACKBONE": "_t3_l2"}))
        _build(MNISTModel, _cfg(model={"ROOT_HEAD": "anchor", "CNN_BACKBONE": "_t3_l2"}))       # layer2 is there
    finally:
        BB.BACKBONES.pop("_t3_l2", None)


def test_a_root_head_refuses_the_raw_event_and_active_head_models():
    for extra in ({"ACTIVE_HEAD": True}, {"ENCODER": "event_gnn"}):
        cfg = _cfg(model={"ROOT_HEAD": "spatial", **extra})
        cfg["MODEL"].pop("CNN_BACKBONE")
        with pytest.raises(ValueError, match="ROOT_HEAD"):
            _build(MNISTModel, cfg)


# ================================================================================================ (b) ROOT_ROT_WEIGHT
def _root_and_finger_d(m, pred, y):
    from pose_repr import decode_to_mano_inputs
    ds = []
    for t in (pred, y):
        dec = decode_to_mano_inputs(t, m.pose_repr, m.mano.hands_components, m.mano.hands_mean)
        aa = torch.cat([dec["global_orient"], dec["local_full_aa"]], dim=-1).view(-1, 16, 3).double().numpy()
        ds.append(aa)
    d = np.zeros((pred.shape[0], 16))
    for j in range(16):
        rel = Rot.from_rotvec(ds[0][:, j]) * Rot.from_rotvec(ds[1][:, j]).inv()
        d[:, j] = np.sin(rel.magnitude() / 2.0) ** 2
    return d


@pytest.mark.parametrize("w", [4.0, 0.25, 1.5])
def test_b_loss_rot_equals_an_independent_scipy_computation(w):
    m = _build(MNISTModel, _cfg(loss={"ROOT_ROT_WEIGHT": w})).train()
    y = _prev(6, seed=1)
    pred = y + 0.15 * torch.randn(6, 51, generator=torch.Generator().manual_seed(2))
    loss, parts = m._compute_loss(pred, y, _betas(6))
    d = _root_and_finger_d(m, pred, y)
    want = ((w * d[:, 0] + d[:, 1:].sum(1)) / (w + 15.0)).mean()
    assert abs(float(parts["loss_rot"]) - want) < 1e-5 * max(1.0, want) + 1e-7, (float(parts["loss_rot"]), want)
    # the other terms are untouched by the weight
    ref = _build(MNISTModel, _cfg()).train()
    _, pr = ref._compute_loss(pred, y, _betas(6))
    for k in ("loss_trans", "loss_fk", "loss_abs_fk"):
        assert torch.equal(parts[k], pr[k]), k
    assert abs(float(pr["loss_rot"]) - d.mean()) < 1e-5 * max(1.0, d.mean()) + 1e-7


def test_b_limits_root_only_and_fingers_only():
    y = _prev(6, seed=1)
    pred = y + 0.2 * torch.randn(6, 51, generator=torch.Generator().manual_seed(4))
    d = _root_and_finger_d(_build(MNISTModel, _cfg()), pred, y)
    big = _build(MNISTModel, _cfg(loss={"ROOT_ROT_WEIGHT": 1e7})).train()._compute_loss(pred, y, _betas(6))[1]["loss_rot"]
    small = _build(MNISTModel, _cfg(loss={"ROOT_ROT_WEIGHT": 1e-7})).train()._compute_loss(pred, y, _betas(6))[1]["loss_rot"]
    assert abs(float(big) - d[:, 0].mean()) < 1e-4 * d[:, 0].mean() + 1e-7
    assert abs(float(small) - d[:, 1:].sum(1).mean() / 15.0) < 1e-4 * d[:, 1:].sum(1).mean() / 15.0 + 1e-7


def test_b_gradient_on_the_root_columns_scales_with_the_weight():
    """With the weight the root columns of the prediction get a larger share of the gradient of L_rot alone."""
    y = _prev(5, seed=1)
    pred0 = y + 0.2 * torch.randn(5, 51, generator=torch.Generator().manual_seed(4))
    share = {}
    for w in (1.0, 4.0):
        m = _build(MNISTModel, _cfg(loss={"ROOT_ROT_WEIGHT": w, "TRANS_WEIGHT": 0.0, "FK_WEIGHT": 0.0})).train()
        pred = pred0.clone().requires_grad_(True)
        m._compute_loss(pred, y, _betas(5))[0].backward()
        g = pred.grad
        assert torch.isfinite(g).all()
        share[w] = float(g[:, 3:6].norm() / g.norm())
    assert share[4.0] > share[1.0] > 0.0


# ================================================================================================ (c) WEIGHT_DECAY
def test_c_param_groups_cover_every_trainable_parameter_once_and_decay_conv_linear_weights_only():
    m = _build(MNISTModel, _cfg(train={"WEIGHT_DECAY": 0.01}), 5)
    groups = m._param_groups(0.01)
    assert [g["weight_decay"] for g in groups] == [0.01, 0.0]
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == sum(1 for p in m.parameters() if p.requires_grad)
    name_of = {id(p): n for n, p in m.named_parameters()}
    decay = {name_of[id(p)] for p in groups[0]["params"]}
    plain = {name_of[id(p)] for p in groups[1]["params"]}
    mods = dict(m.named_modules())
    want_decay = {n + ".weight" for n, mod in mods.items() if isinstance(mod, (nn.Conv2d, nn.Linear))}
    assert decay == want_decay and len(decay) > 10
    for n in plain:
        mod = mods[n.rsplit(".", 1)[0]]
        assert n.endswith(".bias") or isinstance(mod, nn.BatchNorm2d), n
    assert {"rn.fc.weight", "conv1.weight", "prev_mlp.0.weight"} <= decay
    assert {"rn.fc.bias", "conv1.bias", "rn.bn1.weight", "rn.bn1.bias", "rn.layer1.0.bn1.weight"} <= plain


def test_c_adamw_with_zero_gradient_applies_only_the_decoupled_decay():
    m = _build(MNISTModel, _fit_cfg(train={"WEIGHT_DECAY": 0.05}), 6)
    opt = m.configure_optimizers()["optimizer"]
    assert type(opt) is torch.optim.AdamW
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    for p in m.parameters():
        p.grad = torch.zeros_like(p)
    lr = opt.param_groups[0]["lr"]
    opt.step()
    groups = m._param_groups(0.05)
    decayed = {n for n, p in m.named_parameters() if any(p is q for q in groups[0]["params"])}
    for n, p in m.named_parameters():
        if n in decayed:
            assert torch.allclose(p, before[n] * (1.0 - lr * 0.05), rtol=1e-6, atol=0), n
            if float(before[n].abs().max()) > 0:
                assert not torch.equal(p, before[n]), n
        else:
            assert torch.equal(p, before[n]), n


def test_c_frozen_parameters_stay_out_of_the_groups_and_zero_decay_is_adam():
    cfg = _fit_cfg(train={"WEIGHT_DECAY": 0.01, "TRAINABLE_PREFIXES": ["rn.fc", "conv1"]})
    m = _build(MNISTModel, cfg)
    groups = m._param_groups(0.01)
    n_in = sum(len(g["params"]) for g in groups)
    assert n_in == sum(1 for p in m.parameters() if p.requires_grad) and 0 < n_in < len(list(m.parameters()))
    assert type(_build(MNISTModel, _fit_cfg(train={"WEIGHT_DECAY": 0.0})).configure_optimizers()["optimizer"]) is torch.optim.Adam
    sched = _build(MNISTModel, _fit_cfg(train={"WEIGHT_DECAY": 0.01})).configure_optimizers()
    ref = _build(MNISTModel, _fit_cfg()).configure_optimizers()
    assert all(sched["lr_scheduler"]["scheduler"].lr_lambdas[0](s) == ref["lr_scheduler"]["scheduler"].lr_lambdas[0](s)
               for s in range(10))


# ====================================================================================================== (d) EMA
def _perturb(m, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.01 * torch.randn(p.shape, generator=g))
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.running_mean.add_(0.01 * torch.randn(mod.running_mean.shape, generator=g))
                mod.running_var.mul_(1.0 + 0.01 * torch.rand(mod.running_var.shape, generator=g))


def test_d_ema_update_arithmetic_warmup_and_tracked_entries():
    m = _build(MNISTModel, _cfg(train={"EMA_DECAY": 0.999}))
    live0 = m._ema_live()
    names = set(live0)
    sd_keys = set(m.state_dict())
    assert names <= sd_keys
    assert not any("num_batches_tracked" in k for k in names)
    assert {k for k in sd_keys if k.endswith(("running_mean", "running_var"))} <= names
    assert {n for n, _ in m.named_parameters()} <= names
    ref = {k: v.detach().clone().double() for k, v in live0.items()}
    m._ema_ready(live0)
    for step in range(30):
        _perturb(m, 100 + step)
        m.ema_update()
        d = min(0.999, (1.0 + step) / (10.0 + step))
        live = m._ema_live()
        for k in ref:
            ref[k] = ref[k] + (1.0 - d) * (live[k].double() - ref[k])
    assert m._ema_steps == 30
    worst = max(float((m._ema[k].double() - ref[k]).abs().max()) for k in ref)
    assert worst < 1e-5, worst
    assert min(0.999, (1.0 + 0) / (10.0 + 0)) == 0.1 and min(0.999, (1.0 + 20000) / (10.0 + 20000)) == 0.999


def test_d_ema_is_exactly_the_weights_when_they_do_not_change():
    m = _build(MNISTModel, _cfg(train={"EMA_DECAY": 0.999}))
    live = m._ema_live()
    m._ema_ready(live)
    for _ in range(40):
        m.ema_update()
    assert all(torch.equal(m._ema[k], v) for k, v in live.items())


def test_d_off_means_off():
    m = _build(MNISTModel, _cfg())
    m.ema_update()
    assert m._ema is None and m._ema_steps == 0
    ck = {"state_dict": m.state_dict()}
    m.on_save_checkpoint(ck)
    assert set(ck) == {"state_dict"}
    with pytest.raises(RuntimeError, match="EMA_DECAY"):
        m.ema_state_dict()


def test_d_checkpoint_layout_and_the_raw_weights_survive_a_save_from_inside_validation():
    m = _randomize(_build(MNISTModel, _cfg(train={"EMA_DECAY": 0.9})))
    m._ema_ready(m._ema_live())
    for s in range(5):
        _perturb(m, 40 + s)
        m.ema_update()
    raw = {k: v.clone() for k, v in m.state_dict().items()}
    ema = m.ema_state_dict()
    assert any(not torch.equal(raw[k], ema[k]) for k in raw)
    ck = {"state_dict": m.state_dict()}
    m.on_save_checkpoint(ck)
    assert list(ck["state_dict"]) == list(raw) == list(ck["raw_state_dict"])
    assert all(torch.equal(ck["state_dict"][k], ema[k]) for k in raw)
    assert all(torch.equal(ck["raw_state_dict"][k], raw[k]) for k in raw)
    assert ck["ema"] == {"decay": 0.9, "steps": 5}
    # the tensors are copies: later EMA steps do not change the written dicts
    snap = {k: v.clone() for k, v in ck["state_dict"].items()}
    _perturb(m, 77)
    m.ema_update()
    assert all(torch.equal(ck["state_dict"][k], snap[k]) for k in snap)
    # inside the validation loop the live tensors hold the EMA; the raw weights are still what `raw_state_dict` gets
    raw2 = {k: v.clone() for k, v in m.state_dict().items()}
    ema2 = m.ema_state_dict()
    m.on_validation_start()
    assert all(torch.equal(m.state_dict()[k], ema2[k]) for k in ema2 if k in m._ema)
    ck2 = {"state_dict": m.state_dict()}
    m.on_save_checkpoint(ck2)
    assert all(torch.equal(ck2["raw_state_dict"][k], raw2[k]) for k in raw2)
    assert all(torch.equal(ck2["state_dict"][k], ema2[k]) for k in ema2)
    m.on_validation_end()
    assert all(torch.equal(m.state_dict()[k], raw2[k]) for k in raw2) and m._ema_backup is None


class _Probe(Callback):
    """Records, around the validation loop, whether the model held the EMA or the raw weights."""

    def __init__(self):
        self.raw_before, self.during, self.after = None, [], []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.raw_before = {k: v.clone() for k, v in pl_module._ema_live().items()}

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        live = pl_module._ema_live()
        self.during.append((all(torch.equal(live[k], pl_module._ema[k]) for k in live),
                            all(torch.equal(live[k], self.raw_before[k]) for k in live)))

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.during:
            live = pl_module._ema_live()
            self.after.append(all(torch.equal(live[k], self.raw_before[k]) for k in live))


@pytest.fixture(scope="module")
def ema_runs(tmp_path_factory):
    """Real CPU Lightning runs of the EMA + AdamW arm (dt_dz_l3_reg's keys), 8 optimiser steps (accumulate 2, 4 epochs of 4
    micro-batches, validation every epoch): (A) uninterrupted; (B) 4 steps, then `fit(ckpt_path=last.ckpt)` to 8."""
    root = tmp_path_factory.mktemp("ema_runs")
    cfg = _fit_cfg(train={"EMA_DECAY": 0.9, "WEIGHT_DECAY": 0.01})
    out = {"cfg": cfg}

    torch.manual_seed(0)
    a = _randomize(MNISTModel(copy.deepcopy(cfg)))
    probe = _Probe()
    tr, ck = _trainer(root / "A", 8, callbacks=[probe])
    tr.fit(a, *_loaders())
    out.update(A=a, A_ck=ck, probe=probe, A_dir=root / "A")

    torch.manual_seed(0)
    b = _randomize(MNISTModel(copy.deepcopy(cfg)))
    tr, ck1 = _trainer(root / "B1", 4)
    tr.fit(b, *_loaders())
    shutil.copy(ck1.last_model_path, root / "B1_step4.ckpt")
    out["B1_last"] = str(root / "B1_step4.ckpt")
    b1_ckpt = torch.load(out["B1_last"], map_location="cpu")
    out["B1_ckpt"] = b1_ckpt
    torch.manual_seed(123)                                   # a resumed run must not depend on the construction seed
    b2 = MNISTModel(copy.deepcopy(cfg))
    seen = {}

    class _AtStart(Callback):
        def on_train_start(self, trainer, pl_module):
            seen["live"] = {k: v.clone() for k, v in pl_module.state_dict().items()}
            seen["ema_steps"] = pl_module._ema_steps
            seen["pending"] = None if pl_module._ema_pending is None else {k: v.clone() for k, v in pl_module._ema_pending.items()}
            seen["global_step"] = int(trainer.global_step)

    tr, ck2 = _trainer(root / "B2", 8, callbacks=[_AtStart()])
    tr.fit(b2, *_loaders(), ckpt_path=out["B1_last"])
    out.update(B=b2, B_ck=ck2, at_start=seen)
    return out


def test_d_real_fit_validation_runs_on_the_ema_weights_and_restores_the_raw_ones(ema_runs):
    p = ema_runs["probe"]
    assert len(p.during) >= 2
    assert all(ema and not raw for ema, raw in p.during), p.during          # validation saw the EMA, not the raw weights
    assert p.after and all(p.after), p.after                                # and the raw weights were back for training


def test_d_real_fit_saved_state_dict_is_the_ema_and_raw_state_dict_the_trained_weights(ema_runs):
    a, ck = ema_runs["A"], ema_runs["A_ck"]
    c = torch.load(ck.last_model_path, map_location="cpu")
    assert c["global_step"] == 8 and c["ema"] == {"decay": 0.9, "steps": 8}
    final_raw, final_ema = a.state_dict(), a.ema_state_dict()
    assert list(c["state_dict"]) == list(c["raw_state_dict"]) == list(final_raw)
    assert all(torch.equal(c["raw_state_dict"][k].cpu(), final_raw[k].cpu()) for k in final_raw)
    assert all(torch.equal(c["state_dict"][k].cpu(), final_ema[k].cpu()) for k in final_ema)
    assert any(not torch.equal(c["state_dict"][k], c["raw_state_dict"][k]) for k in final_ema)     # the EMA lags the weights
    grid = sorted(Path(ema_runs["A_dir"]).glob("c-step=*.ckpt"))
    assert len(grid) == 4                                                                           # every 2nd step: 2, 4, 6, 8
    steps = [torch.load(g, map_location="cpu")["ema"]["steps"] for g in grid]
    assert steps == [2, 4, 6, 8]                                         # a checkpoint at step N holds the EMA after N updates


def test_d_load_from_checkpoint_returns_the_ema_weights(ema_runs):
    """Evaluation (evalx, select_checkpoint, the distillation teacher): no Trainer, so no raw weights are swapped in."""
    c = torch.load(ema_runs["A_ck"].last_model_path, map_location="cpu")
    m = MNISTModel.load_from_checkpoint(ema_runs["A_ck"].last_model_path, cfg=copy.deepcopy(ema_runs["cfg"]), map_location="cpu")
    sd = m.state_dict()
    assert all(torch.equal(sd[k], c["state_dict"][k]) for k in sd)
    assert any(not torch.equal(sd[k], c["raw_state_dict"][k]) for k in sd)


def test_d_resume_restores_the_raw_weights_and_the_ema_continues_without_a_seam(ema_runs):
    c1, at = ema_runs["B1_ckpt"], ema_runs["at_start"]
    assert at["global_step"] == 4 and at["ema_steps"] == 4
    # at the start of training the model holds the RAW weights of the checkpoint, not the EMA ones
    assert all(torch.equal(at["live"][k], c1["raw_state_dict"][k]) for k in at["live"])
    assert any(not torch.equal(at["live"][k], c1["state_dict"][k]) for k in at["live"])
    # and the EMA was restored from the checkpoint's state_dict
    assert at["pending"] is not None and all(torch.equal(at["pending"][k], c1["state_dict"][k]) for k in at["pending"])
    # the split run equals the uninterrupted run bit for bit: raw weights, EMA, checkpoints
    a, b = ema_runs["A"], ema_runs["B"]
    _assert_same_state(a, b)
    ea, eb = a.ema_state_dict(), b.ema_state_dict()
    assert all(torch.equal(ea[k], eb[k]) for k in ea) and a._ema_steps == b._ema_steps == 8
    ca = torch.load(ema_runs["A_ck"].last_model_path, map_location="cpu")
    cb = torch.load(ema_runs["B_ck"].last_model_path, map_location="cpu")
    for key in ("state_dict", "raw_state_dict"):
        assert all(torch.equal(ca[key][k], cb[key][k]) for k in ca[key]), key
    assert ca["ema"] == cb["ema"]


def test_d_checkpoint_written_at_the_end_of_validation_still_holds_the_raw_weights(tmp_path):
    """ModelCheckpoint's `on_validation_end` (a callback hook, run BEFORE the module's) writes `last.ckpt` while the live
    tensors hold the EMA: its raw_state_dict must still be the trained weights, its state_dict the EMA."""
    cfg = _fit_cfg(train={"EMA_DECAY": 0.9})
    torch.manual_seed(0)
    m = _randomize(MNISTModel(copy.deepcopy(cfg)))
    probe = _Probe()
    tr, ck = _trainer(tmp_path / "v", 2, every=1000, callbacks=[probe])         # one validation, after the 2nd optimiser step
    tr.fit(m, *_loaders())
    assert len(probe.during) == 1 and probe.during[0] == (True, False)
    c = torch.load(ck.last_model_path, map_location="cpu")
    assert c["global_step"] == 2 and c["ema"]["steps"] == 2
    live = m._ema_live()
    ema = m.ema_state_dict()
    assert all(torch.equal(c["raw_state_dict"][k], v) for k, v in live.items())
    assert all(torch.equal(c["raw_state_dict"][k], probe.raw_before[k]) for k in probe.raw_before)
    assert all(torch.equal(c["state_dict"][k], ema[k]) for k in ema)
    assert any(not torch.equal(c["state_dict"][k], c["raw_state_dict"][k]) for k in live)


def test_d_validate_with_a_checkpoint_keeps_the_ema_weights(ema_runs, tmp_path):
    """`trainer.validate(ckpt_path=...)` is not FITTING: the checkpoint's state_dict (the EMA) is what gets loaded."""
    c = torch.load(ema_runs["A_ck"].last_model_path, map_location="cpu")
    m = MNISTModel(copy.deepcopy(ema_runs["cfg"]))
    tr = pl.Trainer(accelerator="cpu", devices=1, logger=False, enable_progress_bar=False, enable_model_summary=False,
                    enable_checkpointing=False, default_root_dir=str(tmp_path))
    tr.validate(m, _loaders()[1], ckpt_path=ema_runs["A_ck"].last_model_path)
    sd = m.state_dict()
    assert all(torch.equal(sd[k], c["state_dict"][k]) for k in sd)


def test_d_resuming_a_non_ema_checkpoint_starts_the_ema_at_its_weights(tmp_path):
    """A checkpoint without raw_state_dict (any earlier run): its weights are the raw weights and the EMA's start."""
    cfg0 = _fit_cfg()
    torch.manual_seed(0)
    m0 = _randomize(MNISTModel(copy.deepcopy(cfg0)))
    tr, ck = _trainer(tmp_path / "p", 4)
    tr.fit(m0, *_loaders())
    c0 = torch.load(ck.last_model_path, map_location="cpu")
    assert "raw_state_dict" not in c0
    m1 = MNISTModel(copy.deepcopy(_fit_cfg(train={"EMA_DECAY": 0.9})))
    seen = {}

    class _At(Callback):
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            if "live" not in seen:                      # after the module's on_train_start built the EMA
                seen["live"] = {k: v.clone() for k, v in pl_module.state_dict().items()}
                seen["ema"] = {k: v.clone() for k, v in pl_module._ema.items()}
                seen["steps"] = pl_module._ema_steps

    tr, _ = _trainer(tmp_path / "q", 8, callbacks=[_At()])
    tr.fit(m1, *_loaders(), ckpt_path=ck.last_model_path)
    assert seen["steps"] == 0
    assert all(torch.equal(seen["live"][k], c0["state_dict"][k]) for k in seen["live"])
    assert all(torch.equal(seen["ema"][k], c0["state_dict"][k]) for k in seen["ema"])
    c1 = torch.load(tmp_path / "q" / "last.ckpt", map_location="cpu")
    assert c1["ema"]["steps"] == 4 and "raw_state_dict" in c1


def test_d_an_ema_free_run_that_resumes_an_ema_checkpoint_trains_from_the_raw_weights(ema_runs, tmp_path):
    at = {}

    class _At(Callback):
        def on_train_start(self, trainer, pl_module):
            at["live"] = {k: v.clone() for k, v in pl_module.state_dict().items()}

    m = MNISTModel(copy.deepcopy(_fit_cfg(train={"WEIGHT_DECAY": 0.01})))             # same optimiser, no EMA
    tr, _ = _trainer(tmp_path / "r", 6, callbacks=[_At()])
    tr.fit(m, *_loaders(), ckpt_path=ema_runs["B1_last"])
    c1 = ema_runs["B1_ckpt"]
    assert all(torch.equal(at["live"][k], c1["raw_state_dict"][k]) for k in at["live"])


# ================================================================================================= (e) ROOT_HEAD
def _head_model(kind, seed=1234, **kw):
    return _randomize(_build(MNISTModel, _cfg(model={"ROOT_HEAD": kind}, **kw), seed))


@pytest.mark.parametrize("kind", ["spatial", "anchor"])
def test_e_initial_model_equals_the_baseline_bit_for_bit(kind):
    base = _randomize(_build(MNISTModel, _cfg()))
    m = _head_model(kind)
    sb, sm = base.state_dict(), m.state_dict()
    extra = [k for k in sm if k not in sb]
    assert extra and all(k.startswith("root_readout.") for k in extra) and all(k in sm for k in sb)
    assert all(torch.equal(sb[k], sm[k]) for k in sb)                           # same seed: the rest has the baseline's draws
    assert _n_params(m) - _n_params(base) == (SPATIAL_PARAMS if kind == "spatial" else ANCHOR_PARAMS)
    last = m.root_readout.fc if kind == "spatial" else m.root_readout.fc2
    assert float(last.weight.abs().max()) == 0.0 and float(last.bias.abs().max()) == 0.0
    x, prev = _lnes(3, empty=(1,)), _prev(3)
    base.eval(), m.eval()
    for B in (1, 3):
        for bf16 in (False, True):
            ob, om = _fwd(base, x[:B], prev[:B], bf16), _fwd(m, x[:B], prev[:B], bf16)
            assert ob.dtype == om.dtype and torch.equal(ob, om), (kind, B, bf16)
    # train mode, a batch with explicit betas / K (the training call)
    betas, K = _betas(3), BASE_K.expand(3, 3, 3) * torch.tensor([1.1, 0.9, 1.0]).view(1, 1, 3)
    base.train(), m.train()
    assert torch.equal(base(x, prev, betas=betas, camera_K=K), m(x, prev, betas=betas, camera_K=K))


def test_e_exact_parameter_counts_of_the_arms():
    assert _n_params(_build(MNISTModel, _cfg())) == BASE_PARAMS
    assert _n_params(_build(MNISTModel, _cfg(DT / "dt_dz_l3_rooth.yaml"))) == BASE_PARAMS + SPATIAL_PARAMS == 2_986_040
    assert _n_params(_build(MNISTModel, _cfg(DT / "dt_dz_l3_rootanc.yaml"))) == BASE_PARAMS + ANCHOR_PARAMS == 3_507_928
    sp = SpatialRootHead(256, 12, 15)
    assert _n_params(sp) == 147_520 + 128 + 18_464 + 17_283
    an = AnchorRootHead(128, 21)
    assert _n_params(an) == 2751 * 256 + 256 + 256 * 3 + 3


def test_e_resnet_forward_taps_is_the_trunk_forward_bitwise():
    m = _head_model("spatial").eval()
    z = torch.randn(2, 3, H, W)
    for rn in (m.rn, BB.build_backbone("resnet18", 51).eval()):
        want = rn(z)
        got, taps = BB.resnet_forward_taps(rn, z, ("layer1", "layer2", "layer3", "layer4"))
        assert torch.equal(want, got)
        assert taps["layer2"].shape == (2, 128, 23, 30) and taps["layer3"].shape == (2, 256, 12, 15)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.equal(m.rn(z), BB.resnet_forward_taps(m.rn, z)[0])
    with pytest.raises(ValueError, match="stages"):
        BB.resnet_forward_taps(m.rn, z, ("layer5",))


@pytest.mark.parametrize("kind", ["spatial", "anchor"])
def test_e_the_head_adds_to_the_root_columns_only(kind):
    base = _randomize(_build(MNISTModel, _cfg())).eval()
    m = _nudge_head(_head_model(kind)).eval()
    x, prev = _lnes(2, seed=4), _prev(2, seed=5)
    ob, om = _fwd(base, x, prev), _fwd(m, x, prev)
    keep = [i for i in range(51) if i not in (3, 4, 5)]
    assert torch.equal(ob[:, keep], om[:, keep])                                 # fc, prev_mlp, gate: untouched elsewhere
    assert not torch.equal(ob[:, 3:6], om[:, 3:6]) and torch.isfinite(om).all()
    # the difference is the head's own output: recompute it from the trunk's maps
    z = m.conv1(torch.cat([x, m._render_prev(prev, *m._resolve_betas_K(prev, None, None))], -1).permute(0, 3, 1, 2))
    with torch.no_grad():
        _, taps = BB.resnet_forward_taps(m.rn, z, ("layer3",) if kind == "spatial" else ("layer2",))
        betas_f, k_f = m._resolve_betas_K(prev, None, None)
        r = (m.root_readout(taps["layer3"]) if kind == "spatial"
             else m.root_readout(m._anchor_features(taps["layer2"], prev, betas_f, k_f)))
    assert torch.allclose((om - ob)[:, 3:6], r, atol=5e-6)


@pytest.mark.parametrize("kind", ["spatial", "anchor"])
def test_e_event_free_packet_returns_prev_exactly_and_gradients_flow(kind):
    m = _nudge_head(_head_model(kind)).eval()
    x, prev = _lnes(3, seed=4, empty=(1,)), _prev(3, seed=5)
    out = _fwd(m, x, prev)
    assert torch.equal(out[1], prev[1])
    m.train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        o = m(x, prev)
    o.float().pow(2).sum().backward()
    for n, p in m.named_parameters():
        if n.startswith("root_readout."):
            assert p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs().max()) > 0, n
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


@pytest.mark.parametrize("kind", ["spatial", "anchor"])
def test_e_first_step_of_a_zero_initialised_head_reaches_only_its_last_layer(kind):
    m = _head_model(kind).train()
    x, prev = _lnes(2, seed=4), _prev(2, seed=5)
    m(x, prev)[:, 3:6].float().pow(2).sum().backward()
    last = {"spatial": "root_readout.fc.", "anchor": "root_readout.fc2."}[kind]
    for n, p in m.named_parameters():
        if n.startswith("root_readout."):
            assert p.grad is not None
            assert (float(p.grad.abs().max()) > 0) == n.startswith(last), n


def test_e_spatial_head_reads_the_spatial_layout():
    h = SpatialRootHead(256, 12, 15)
    g = torch.Generator().manual_seed(2)
    for p in h.parameters():
        nn.init.normal_(p, std=0.05)
    h.eval()
    f = torch.randn(2, 256, 12, 15, generator=g)
    flipped = f.flip(-1)
    assert torch.allclose(f.mean((2, 3)), flipped.mean((2, 3)), atol=1e-6)          # the GAP row cannot tell them apart
    assert not torch.allclose(h(f), h(flipped), atol=1e-4)
    with pytest.raises(ValueError, match="feature map"):
        h(torch.randn(1, 256, 11, 15))


def test_e_stem_geometry_layer2_cell_m_sits_at_input_pixel_8m():
    """The claim `_anchor_features` rests on, shown on the stem's own hyper-parameters (7x7 s2 p3, max-pool 3x3 s2 p1,
    3x3 s2 p1) with centre-delta kernels: an impulse at pixel (8 i, 8 j) lands in cell (i, j) of the 23 x 30 map, nowhere else."""
    conv1 = nn.Conv2d(1, 1, 7, stride=2, padding=3, bias=False)
    pool = nn.MaxPool2d(3, 2, 1)
    conv2 = nn.Conv2d(1, 1, 3, stride=2, padding=1, bias=False)
    with torch.no_grad():
        conv1.weight.zero_(); conv1.weight[0, 0, 3, 3] = 1.0
        conv2.weight.zero_(); conv2.weight[0, 0, 1, 1] = 1.0
    for (i, j) in [(0, 0), (3, 5), (11, 14), (22, 29)]:
        x = torch.zeros(1, 1, H, W)
        x[0, 0, 8 * i, 8 * j] = 1.0
        y = conv2(pool(conv1(x)))
        assert y.shape[-2:] == (23, 30)
        nz = (y[0, 0] != 0).nonzero().tolist()
        assert nz == [[i, j]], (i, j, nz)


def _anchor_model():
    return _randomize(_build(MNISTModel, _cfg(DT / "dt_dz_l3_rootanc.yaml"))).eval()


def _project(m, prev, betas, K):
    """The 21 joints of `prev` in render pixels, by the model's own FK and intrinsics (independent of the sampling code)."""
    with torch.no_grad():
        _, j = m._fk(prev.float(), betas)
        fx, fy, cx, cy = m._intrinsics(K)
    x, y, z = j.unbind(-1)
    return fx[:, None] * x / z + cx[:, None], fy[:, None] * y / z + cy[:, None], z


def test_e_anchor_samples_the_feature_map_at_the_projected_joints_with_stride_8():
    m = _anchor_model()
    B = 3
    prev = _prev(B, seed=8)
    betas = _betas(B)
    K = BASE_K.expand(B, 3, 3).clone() * torch.tensor([1.0, 1.0, 1.0]).view(1, 1, 3)
    K[1] = K[1] * torch.tensor([1.15, 1.15, 1.0]).view(1, 3)          # per-sample K' (DomRand), here a different focal length
    K[2, 0, 2] += 30.0                                                # and a shifted principal point
    f2 = torch.zeros(B, 128, 23, 30)
    f2[:, 0] = torch.arange(30.0).view(1, 1, 30).expand(B, 23, 30)    # channel 0 = column index of the cell
    f2[:, 1] = torch.arange(23.0).view(1, 23, 1).expand(B, 23, 30)    # channel 1 = row index
    feats = m._anchor_features(f2, prev, betas, K)
    assert feats.shape == (B, 21, 131) and feats.dtype == torch.float32
    u, v, z = _project(m, prev, betas, K)
    inb = (z > 1e-6) & (u >= -0.5) & (u < W - 0.5) & (v >= -0.5) & (v < H - 0.5)
    assert inb.any() and bool(inb.all())                              # these hands are inside the image
    # a linear ramp is reproduced exactly by bilinear sampling: channel 0 / 1 at a joint = u / 8, v / 8 (inside the grid)
    cu, cv = (u / 8.0).clamp(0, 29), (v / 8.0).clamp(0, 22)
    assert torch.allclose(feats[..., 0], cu, atol=1e-4) and torch.allclose(feats[..., 1], cv, atol=1e-4)
    assert (feats[..., 0] != 0).any() and float(feats[..., 2:128].abs().max()) == 0.0
    # position features: u, v normalised to [-1, 1] over the image's pixel centres; the flag
    assert torch.allclose(feats[..., 128], 2.0 * u / (W - 1) - 1.0, atol=1e-5)
    assert torch.allclose(feats[..., 129], 2.0 * v / (H - 1) - 1.0, atol=1e-5)
    assert bool((feats[..., 130] == 1.0).all())
    # per-sample intrinsics matter: the same hand, a different K', other sample positions
    feats2 = m._anchor_features(f2, prev, betas, K[:1].expand(B, 3, 3).contiguous())
    assert not torch.allclose(feats[1, :, 0], feats2[1, :, 0], atol=1e-3)


def test_e_anchor_out_of_bounds_and_behind_camera_joints_are_zeroed_and_flagged():
    m = _anchor_model()
    f2 = torch.randn(3, 128, 23, 30)
    betas = _betas(3)
    K = BASE_K.expand(3, 3, 3).contiguous()
    prev = _prev(3, seed=8)
    # sample 0: hand straddling the right image border; sample 1: far outside; sample 2: behind the camera
    u0, v0, z0 = _project(m, prev, betas, K)
    # sample 0: the centroid of the 21 joints moved onto the right border, so some joints are in and some are out
    prev[0, 0] += ((W - 0.5) - float(u0[0].mean())) * float(z0[0].mean()) / float(m._intrinsics(K)[0][0])
    prev[1, 0] += 0.6
    prev[2, 2] = -0.5
    feats = m._anchor_features(f2, prev, betas, K)
    u, v, z = _project(m, prev, betas, K)
    inb = (z > 1e-6) & (u >= -0.5) & (u < W - 0.5) & (v >= -0.5) & (v < H - 0.5)
    assert 0 < int(inb[0].sum()) < 21 and int(inb[1].sum()) == 0 and int(inb[2].sum()) == 0
    assert torch.equal(feats[..., 130], inb.float())
    assert float(feats[..., :128][~inb].abs().max()) == 0.0                    # features zeroed where the joint is out
    assert float(feats[..., :128][inb].abs().sum()) > 0
    assert torch.isfinite(feats).all() and float(feats[..., 128:130].abs().max()) <= 2.0 + 1e-6


def test_e_anchor_reads_only_the_anchors_and_is_differentiable_in_the_feature_map():
    m = _nudge_head(_head_model("anchor"), scale=0.05).eval()
    B = 2
    prev, betas = _prev(B, seed=8), _betas(B)
    K = BASE_K.expand(B, 3, 3).contiguous()
    f2 = torch.randn(B, 128, 23, 30)
    base = m.root_readout(m._anchor_features(f2, prev, betas, K))
    u, v, _ = _project(m, prev, betas, K)
    cells = torch.zeros(B, 23, 30, dtype=torch.bool)
    for b in range(B):
        for j in range(21):
            cu, cv = float(u[b, j]) / 8.0, float(v[b, j]) / 8.0
            for i in (math.floor(cv), math.floor(cv) + 1):
                for k in (math.floor(cu), math.floor(cu) + 1):
                    if 0 <= i < 23 and 0 <= k < 30:
                        cells[b, i, k] = True
    assert 0 < int(cells.sum()) < B * 23 * 30
    far = f2.clone()
    far.permute(0, 2, 3, 1)[~cells] += 5.0                                    # change every cell no joint's bilinear stencil touches
    assert torch.equal(base, m.root_readout(m._anchor_features(far, prev, betas, K)))
    near = f2.clone()
    near.permute(0, 2, 3, 1)[cells] += 0.5
    assert not torch.allclose(base, m.root_readout(m._anchor_features(near, prev, betas, K)), atol=1e-5)
    # gradient: reaches exactly the stencil cells; none to the previous state (it is an input, not a variable)
    f = f2.clone().requires_grad_(True)
    p = prev.clone().requires_grad_(True)
    m.root_readout(m._anchor_features(f, p, betas, K)).sum().backward()
    gmag = f.grad.abs().sum(1) > 0
    assert int((gmag & ~cells).sum()) == 0 and int(gmag.sum()) > 0
    assert p.grad is None


def test_e_anchor_features_under_bf16_autocast_are_float32_and_equal_fp32_geometry():
    m = _anchor_model()
    B = 2
    prev, betas = _prev(B, seed=8), _betas(B)
    K = BASE_K.expand(B, 3, 3).contiguous()
    f2 = torch.randn(B, 128, 23, 30)
    a = m._anchor_features(f2, prev, betas, K)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        b = m._anchor_features(f2.bfloat16(), prev, betas, K)
    assert b.dtype == torch.float32 and torch.equal(a[..., 128:], b[..., 128:])        # geometry: fp32 either way


def test_e_anchor_with_a_wrong_sized_map_raises_and_works_with_the_full_resnet18():
    m = _anchor_model()
    with pytest.raises(ValueError, match="layer2 map"):
        m._anchor_features(torch.randn(1, 128, 22, 30), _prev(1), _betas(1), BASE_K.expand(1, 3, 3))
    full = _randomize(_build(MNISTModel, _cfg(model={"ROOT_HEAD": "spatial", "CNN_BACKBONE": "resnet18"}))).eval()
    assert full.root_readout.in_hw == (12, 15) and _fwd(full, _lnes(1), _prev(1)).shape == (1, 51)
    base = _randomize(_build(MNISTModel, _cfg(model={"CNN_BACKBONE": "resnet18"}))).eval()
    assert torch.equal(_fwd(full, _lnes(1), _prev(1)), _fwd(base, _lnes(1), _prev(1)))
    half = _randomize(_build(MNISTModel, _cfg(model={"ROOT_HEAD": "anchor", "CNN_BACKBONE": "resnet18_w0.5"}))).eval()
    assert half.root_readout.c_in == 64 and _fwd(half, _lnes(1), _prev(1)).shape == (1, 51)


def test_e_closed_loop_context_path_works_with_both_heads():
    for kind in ("spatial", "anchor"):
        m = _nudge_head(_head_model(kind)).eval()
        m.set_hand_context(_betas(1)[0], BASE_K * 1.05)
        out = _fwd(m, _lnes(1), _prev(1))
        assert out.shape == (1, 51) and torch.isfinite(out).all()


# ============================================================================================== (f) RENDER_FP32
def _render_models(**extra):
    a = _build(MNISTModel, _cfg(model=extra)).to(DEV).eval()
    return a


def test_f_render_fp32_equals_the_plain_fp32_render_under_bf16_autocast():
    on = _render_models(RENDER_FP32=True)
    off = _render_models()
    B = 24
    prev = _prev(B, seed=21, norm=2.0).to(DEV)
    betas = _betas(B, seed=22).to(DEV)
    K = (BASE_K.expand(B, 3, 3).clone() * torch.tensor([1.0, 1.0, 1.0]).view(1, 1, 3)).to(DEV)
    K[:, 0, 0] *= torch.linspace(0.85, 1.2, B, device=DEV)
    K[:, 1, 1] = K[:, 0, 0]
    with torch.no_grad():
        plain = off._render_prev(prev, betas, K)                                      # no autocast: the evaluation render
        with torch.autocast(DEV.type, dtype=torch.bfloat16):
            guarded = on._render_prev(prev, betas, K)
            unguarded = off._render_prev(prev, betas, K)
    assert guarded.dtype == plain.dtype == torch.float32
    assert torch.equal(guarded, plain)
    assert float(plain[..., 0].sum()) > 0
    if DEV.type == "cuda":
        assert not torch.equal(unguarded, plain)                                     # without the guard, bf16 FK moves pixels
    print(f"\nrender differs without the guard on {DEV.type}: {not torch.equal(unguarded, plain)}; "
          f"pixels differing {int((unguarded != plain).sum())} of {plain.numel()}")


def test_f_forward_in_fp32_is_the_same_with_the_flag_and_changes_only_under_autocast():
    on = _randomize(_build(MNISTModel, _cfg(model={"RENDER_FP32": True}))).to(DEV).eval()
    off = _randomize(_build(MNISTModel, _cfg())).to(DEV).eval()
    x, prev = _lnes(3, seed=4), _prev(3, seed=5)
    a, b = _fwd(on, x, prev), _fwd(off, x, prev)
    assert torch.equal(a, b)
    with torch.no_grad(), torch.autocast(DEV.type, dtype=torch.bfloat16):
        c = on(x.to(DEV), prev.to(DEV))
    assert torch.isfinite(c).all()


def test_f_flag_state_dict_is_the_baselines():
    _assert_same_state(_build(MNISTModel, _cfg()), _build(MNISTModel, _cfg(model={"RENDER_FP32": True})))


# ================================================================================================== (g) configs
def _make_configs():
    spec = importlib.util.spec_from_file_location("dt_make_configs", REPO / "tools" / "dt" / "make_configs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, prefix + k + "."))
        else:
            out[prefix + k] = v
    return out


FACTORS = {"ema": {"TRAIN.EMA_DECAY": 0.999}, "wd": {"TRAIN.WEIGHT_DECAY": 0.01},
           "reg": {"TRAIN.EMA_DECAY": 0.999, "TRAIN.WEIGHT_DECAY": 0.01}, "rootw4": {"LOSS.ROOT_ROT_WEIGHT": 4.0},
           "rooth": {"MODEL.ROOT_HEAD": "spatial"}, "rootanc": {"MODEL.ROOT_HEAD": "anchor"},
           "r32": {"MODEL.RENDER_FP32": True}}


def test_g_arms_follow_the_dt2_arms_in_the_registry_and_are_single_factor():
    mc = _make_configs()
    names = list(mc.ARMS)
    assert names[-7:] == [f"dt_{a}" for a in ARMS7] and names[-8] == "dt_nosw"
    for a in ARMS7:
        assert _flat(mc.ARMS[f"dt_{a}"][0]) == FACTORS[a]


@pytest.mark.parametrize("arm", ARMS7)
def test_g_generated_configs_are_the_parent_plus_exactly_the_factor(arm):
    parent, child = _flat(_load(BASE)), _flat(_load(DT / f"dt_dz_l3_{arm}.yaml"))
    parent.pop("_config_path"), child.pop("_config_path")                   # added by load_config
    diff = {k: v for k, v in child.items() if parent.get(k, "<absent>") != v}
    ident = {"TRAIN.OUTPUT_DIR", "TRAIN.RUN_NAME", "EVAL.OUTPUT_DIR"}
    assert {k for k in diff if k not in ident} == set(FACTORS[arm]), diff
    assert all(diff[k] == v for k, v in FACTORS[arm].items())
    assert diff["TRAIN.RUN_NAME"] == f"dt_dz_l3_{arm}" and not set(parent) - set(child)
    assert (DT / f"dt_dz_l3_{arm}_2k.yaml").exists() and (DT / f"dt_{arm}.yaml").exists()
    m = _build(MNISTModel, _cfg(DT / f"dt_dz_l3_{arm}.yaml"))
    want = BASE_PARAMS + {"rooth": SPATIAL_PARAMS, "rootanc": ANCHOR_PARAMS}.get(arm, 0)
    assert _n_params(m) == want
    assert m.ema_decay == FACTORS[arm].get("TRAIN.EMA_DECAY", 0.0) and m.weight_decay == FACTORS[arm].get("TRAIN.WEIGHT_DECAY", 0.0)
    assert m.root_rot_weight == FACTORS[arm].get("LOSS.ROOT_ROT_WEIGHT", 1.0)
    assert m.root_head_kind == FACTORS[arm].get("MODEL.ROOT_HEAD", "none")
    assert m.render_fp32 is FACTORS[arm].get("MODEL.RENDER_FP32", False)
    # the depth-scale augmentation of dt_dz and the l3 trunk are in every arm (the dt_dz* naming rule of test_dt_domrand)
    assert child["AUG.DOMRAND.SCALE_MODE"] == "depth" and child["MODEL.CNN_BACKBONE"] == "resnet18_l3"
