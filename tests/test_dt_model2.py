"""Task `model-acc` of the DT round: two default-off features of the dense tracker `MNISTModel` and of
`semkine/train.py` (docs/DT_RENDER_TRACK_PREREG.md sections 2 and 3):

  LOSS.ACCEL_WEIGHT                  arm dt_acc: on a triplet batch (DATA.TRIPLET, `{"w0", "w1", "w2", "valid"}`) the
                                     acceleration error of the absolute FK joints over three consecutive windows is added
                                     to the loss before its log10 (`BaseModel._accel_loss`, logged `train_loss_accel`)
  MODEL.DISTILL_WEIGHT / DISTILL_CKPT  arm dt_w05_kd: dense distillation. The frozen teacher's output is the target of the
                                     student's output under the training objective (`_distill_dense`, logged
                                     `train_loss_distill`); `semkine/train.py` builds the teacher from the student's config
                                     minus the student-only keys, `CNN_BACKBONE` now among them.

Pinned here (the letters follow the task text):

  (a) backward compatibility: with both features off, the edited model is the pre-edit one (outputs/dt/ref/model_orig2.py,
      sha256 pinned) bit for bit: state_dict / modules / random draws of every distinct config block in configs/, eval
      forwards, and the loss, its parts and every gradient of a train-mode `training_step` on a real batch, for
      rt_cnntrack (mse_51d), dt_base (so3_trans_fk) and the packed S37 path (rt_s37_2k: forward_packet and
      training_step on a real two-packet batch);
  (b) the acceleration term: exactly 0 for predictions equal to the targets, ~0 for targets + a common translation, equal
      to a hand-computed second difference and to a row-by-row reference, rows with valid false take no part (also with
      NaN in them), the main term is the plain loss of the w2 rows to 1e-6, the gradient is main + w * accel and finite for
      every parameter, the BatchNorm running statistics are those of the plain step, the guards raise ValueError, a plain
      batch with ACCEL_WEIGHT 0 is unchanged, and the cost of one step against the plain step (CPU, B = 16);
  (c) the distillation term: finite, non-zero, equal to the objective applied to (student, teacher), gradients reach only
      the student, the teacher stays frozen / eval / unchanged and out of the student's checkpoint, the teacher config
      drops CNN_BACKBONE and keeps the architecture keys, the packed-batch behaviour is the original's;
  (d) the consumers of a dict batch: `ThroughputCallback`, a real 2-worker DataLoader, `pin_memory` recursion and a real
      Lightning `fit` on CPU (train and validation loops).

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python -m pytest tests/test_dt_model2.py -q
"""
from __future__ import annotations

import copy
import functools
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import types
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "model")]
from config import load_config                                               # noqa: E402
from model import MNISTModel                                                 # noqa: E402

REF = REPO / "outputs" / "dt" / "ref"
ORIG_MODEL = REF / "model_orig2.py"
ORIG_TRAIN = REF / "train_orig2.py"
#: sha256 of model/model.py and semkine/train.py as found in STAGE (= MAIN) before the model-acc edit (2026-10-03)
ORIG_MODEL_SHA256 = "758785e65c5af032c262001807477c457efca66226f4f75f79364fe74fd56c89"
ORIG_TRAIN_SHA256 = "f7eefea11f5c2250fc4103bed823d8b27187856d1f94887c23d03632e198c786"
RT = REPO / "configs" / "rt"
DT = REPO / "configs" / "dt"
RT_CFG, BASE_CFG, ACC_CFG, W05_CFG = RT / "rt_cnntrack.yaml", DT / "dt_base.yaml", DT / "dt_acc.yaml", DT / "dt_w05.yaml"
S37_CFG = RT / "rt_s37_2k.yaml"
#: DT2 round (tests/test_dt_model3.py): the MODEL keys the pre-edit reference does not know, and the LOSS / TRAIN keys the
#: edited model reads and the reference ignores. Configs that carry any of them are not "existing blocks".
DT2_MODEL_KEYS = frozenset({"ROOT_HEAD", "RENDER_FP32"})
DT2_OTHER_KEYS = {"LOSS": ("ROOT_ROT_WEIGHT",), "TRAIN": ("WEIGHT_DECAY", "EMA_DECAY"),
                  # DT2 package E (tests/test_dt_model4.py): the rollout keys of the TRACK block
                  "TRACK": ("ROLLOUT_P", "ROLLOUT_RAMP")}
NEW_TRACK_KEYS = frozenset(DT2_OTHER_KEYS["TRACK"])


def _uses_dt2_keys(cfg):
    return bool(DT2_MODEL_KEYS & set(cfg.get("MODEL", {}))) or any(
        k in (cfg.get(sec) or {}) for sec, keys in DT2_OTHER_KEYS.items() for k in keys)
DATA_ROOT = REPO / "data" / "hand_data51"
REAL_TEACHER = REPO / "outputs" / "semkine" / "rt_cnntrack_s3407" / "last.ckpt"
H, W = 180, 240
K_RENDER_INPUT = torch.tensor([[225.0, 0.0, 122.0], [0.0, 226.0, 90.0], [0.0, 0.0, 1.0]])

pytestmark = pytest.mark.skipif(not DATA_ROOT.exists(), reason="hand_data51 not available")


# ----------------------------------------------------------------------------------------------------- helpers
@functools.lru_cache(maxsize=None)
def _load(path):
    return load_config(path)


def _cfg(path, model=None, loss=None, data=None):
    cfg = copy.deepcopy(_load(Path(path)))
    cfg["MODEL"].update(model or {})
    cfg["LOSS"].update(loss or {})
    cfg["DATA"].update(data or {})
    return cfg


def _build(cls, cfg, seed=1234):
    torch.manual_seed(seed)
    return cls(copy.deepcopy(cfg))


def _randomize(m, seed=7):
    """What a trained model has and a fresh one does not: prev_mlp's zero-initialised last layer and BatchNorm
    statistics that are not the identity."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        if hasattr(m, "prev_mlp"):
            for p in m.prev_mlp.parameters():
                p.copy_(0.05 * torch.randn(p.shape, generator=g))
        for mod in m.modules():
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.running_mean.copy_(0.1 * torch.randn(mod.running_mean.shape, generator=g))
                mod.running_var.copy_(1.0 + 0.2 * torch.rand(mod.running_var.shape, generator=g))
    return m


def _prev(B, seed=0, norm=2.3):
    """A realistic state: translation near (0, 0, 0.55), root axis-angle of norm `norm`, finger angles ~0.3."""
    g = torch.Generator().manual_seed(seed)
    p = torch.zeros(B, 51)
    p[:, :2] = 0.05 * torch.randn(B, 2, generator=g)
    p[:, 2] = 0.55 + 0.03 * torch.randn(B, generator=g)
    axis = torch.randn(B, 3, generator=g)
    p[:, 3:6] = axis / axis.norm(dim=-1, keepdim=True) * norm
    p[:, 6:] = 0.3 * torch.randn(B, 45, generator=g)
    return p


def _lnes(B, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(B, H, W, 2, generator=g) < 0.12).float() * torch.rand(B, H, W, 2, generator=g)


def _assert_same_state(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb)
    for k in sa:
        assert sa[k].shape == sb[k].shape and sa[k].dtype == sb[k].dtype, k
        assert torch.equal(sa[k], sb[k]), k
    assert [n for n, _ in a.named_parameters()] == [n for n, _ in b.named_parameters()]
    assert [n for n, _ in a.named_buffers()] == [n for n, _ in b.named_buffers()]
    assert [(n, type(mod).__name__) for n, mod in a.named_modules()] == \
           [(n, type(mod).__name__) for n, mod in b.named_modules()]


def _capture_logs(m):
    """Replace `self.log` (a no-op with a warning without a Trainer) by a recorder."""
    logged = {}
    m.log = lambda name, value, **kw: logged.__setitem__(name, value.detach().clone() if torch.is_tensor(value) else value)
    return logged


def _ts(model, batch, bf16=False):
    """One `training_step` plus backward: (returned loss, logged values, every parameter gradient)."""
    logged = _capture_logs(model)
    model.zero_grad(set_to_none=True)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        out = model.training_step(batch, 0)
    out.backward()
    grads = {n: (None if p.grad is None else p.grad.clone()) for n, p in model.named_parameters()}
    return out.detach(), logged, grads


def _assert_same_step(ref, got, label=""):
    assert ref[0].dtype == got[0].dtype and torch.equal(ref[0], got[0]), label
    assert list(ref[1]) == list(got[1]), label
    for k in ref[1]:
        assert torch.equal(torch.as_tensor(ref[1][k]), torch.as_tensor(got[1][k])), (label, k)
    assert list(ref[2]) == list(got[2])
    for n in ref[2]:
        assert (ref[2][n] is None) == (got[2][n] is None), (label, n)
        if ref[2][n] is not None:
            assert torch.equal(ref[2][n], got[2][n]), (label, n)


def _bn_buffers(m):
    return {n: b.clone() for n, b in m.named_buffers() if "running" in n or "num_batches" in n}


def _train_ds(cfg, split="train", train=True):
    from semkine.dataset import build_dataset
    comps = np.load(cfg["MANO"]["NPZ"])["hands_components"].astype(np.float32)
    return build_dataset(cfg, split, comps, train=train)


# ---------------------------------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def orig():
    """The pre-edit model.py, imported under another module name (same sys.path as the live import)."""
    if not ORIG_MODEL.exists():
        pytest.skip(f"{ORIG_MODEL} not available")
    assert hashlib.sha256(ORIG_MODEL.read_bytes()).hexdigest() == ORIG_MODEL_SHA256, "the reference copy was modified"
    spec = importlib.util.spec_from_file_location("model_orig2", ORIG_MODEL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["model_orig2"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def train_mod():
    import semkine.train as T
    return T


@pytest.fixture(scope="module")
def real_batch():
    """Six plain training samples (DomRand on, so K is the augmented K') as the dataset returns them."""
    ds = _train_ds(_cfg(BASE_CFG))
    return default_collate([ds[i] for i in (0, 9000, 21000, 36000, 52000, 70000)])


@pytest.fixture(scope="module")
def trip():
    """`(cfg, dataset, batch)`: dt_acc's training split with the coin always passing (TRIPLET_FRAC 1), so a sample is valid
    iff it has 3 * w ms of history in its run. The batch: 4 samples deep inside long runs (>= 900 ms of history: valid for
    every window up to 300 ms) and 2 right at a run start (< 90 ms: invalid for every window >= 30 ms)."""
    cfg = _cfg(ACC_CFG, data={"TRIPLET_FRAC": 1.0})
    ds = _train_ds(cfg)
    assert ds.triplet and ds.triplet_frac == 1.0
    hist = ds.index[:, 1] - ds.index[:, 2] + 1
    deep, shallow = np.nonzero(hist >= 900)[0], np.nonzero(hist < 90)[0]
    sel = [int(deep[i]) for i in (0, 5000, 20000, 40000)] + [int(shallow[i]) for i in (3, 700)]
    batch = default_collate([ds[i] for i in sel])
    assert batch["valid"].dtype == torch.bool and batch["valid"].tolist() == [True] * 4 + [False] * 2
    return cfg, ds, batch


def _acc_model(trip_cfg, seed=1234, **loss):
    cfg = copy.deepcopy(trip_cfg)
    cfg["LOSS"].update(loss)
    return _randomize(_build(MNISTModel, cfg, seed)).train()


@pytest.fixture(scope="module")
def stub():
    """A dt_acc model whose network is replaced per test (`m.forward = ...`): only its FK and `_accel_loss` are exercised."""
    return _build(MNISTModel, _cfg(ACC_CFG)).train()


def _synthetic_triplet(valid, g, betas=None, garbage=None):
    """A triplet dict whose targets are `g` (three (B, 51) tensors). The inputs the stub forward ignores are zeros; `betas`
    is one row per sample (default: the same 10 numbers for all); `garbage` fills every tensor of w0 / w1 of the invalid rows."""
    valid = torch.as_tensor(valid)
    B = len(valid)
    if betas is None:
        betas = (0.3 * torch.randn(1, 10, generator=torch.Generator().manual_seed(5))).expand(B, 10).clone()
    gen = torch.Generator().manual_seed(11)
    # every window has its own input, prev state and camera, all different from the others' (a forward that read the wrong
    # window's tensor would be seen by `test_b_extra_forward_...`)
    ws = [[torch.randn(B, 2, generator=gen), _prev(B, seed=100 + k), g[k].clone(), betas.clone(),
           K_RENDER_INPUT.expand(B, 3, 3).clone() * (1.0 + 0.1 * k)] for k in range(3)]
    if garbage is not None:
        for k in (0, 1):
            for t in ws[k]:
                t[~valid] = garbage
    return {"w0": ws[0], "w1": ws[1], "w2": ws[2], "valid": valid}


def _stub_forward(m, out01):
    m.forward = lambda x, prevpos, betas=None, camera_K=None: out01


# ======================================================================================== (a) backward compatibility
def test_a_pristine_copies_are_the_files_before_the_edit(orig):
    assert hashlib.sha256(ORIG_TRAIN.read_bytes()).hexdigest() == ORIG_TRAIN_SHA256
    assert set(orig.MNISTModel.MODEL_KEYS) == set(MNISTModel.MODEL_KEYS) - DT2_MODEL_KEYS   # nothing else new: unknown ones still raise
    assert DT2_MODEL_KEYS <= set(MNISTModel.MODEL_KEYS) and not DT2_MODEL_KEYS & set(orig.MNISTModel.MODEL_KEYS)
    assert set(orig.MNISTModel.TRACK_KEYS) == set(MNISTModel.TRACK_KEYS) - NEW_TRACK_KEYS
    assert not hasattr(orig.MNISTModel, "_accel_loss") and hasattr(MNISTModel, "_accel_loss")
    with pytest.raises(ValueError, match="unknown MODEL keys"):
        MNISTModel(_cfg(BASE_CFG, model={"NOT_A_KEY": 1}))


@pytest.mark.parametrize("path", [RT_CFG, BASE_CFG], ids=["rt_cnntrack", "dt_base"])
@pytest.mark.parametrize("explicit", [{}, {"ACCEL_WEIGHT": 0.0}, {"ACCEL_WEIGHT": 0}], ids=["plain", "0.0", "0"])
def test_a_default_state_dict_modules_and_rng_equal_the_original(orig, path, explicit):
    a = _build(orig.MNISTModel, _cfg(path))
    rng_a = torch.get_rng_state()
    b = _build(MNISTModel, _cfg(path, loss=explicit))
    assert torch.equal(rng_a, torch.get_rng_state())
    _assert_same_state(a, b)
    assert b.accel_weight == 0.0 and b.teacher is None and b.distill_weight == 0.0


def _model_blocks():
    """One config per distinct (MODEL, TRACK, LOSS, DATA, MANO) block of every existing config file."""
    seen, out = set(), []
    for p in sorted((REPO / "configs").rglob("*.yaml")):
        try:
            cfg = load_config(p)
        except Exception:                                         # not a model config
            continue
        if "MODEL" not in cfg or _uses_dt2_keys(cfg):
            continue
        key = json.dumps({k: cfg.get(k) for k in ("MODEL", "TRACK", "LOSS", "DATA", "MANO")}, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def test_a_every_existing_model_block_builds_identically(orig):
    """The guards sit in `__init__` (DISTILL_WEIGHT, ACCEL_WEIGHT), so the claim is checked for every distinct block that
    exists, the raw-event arms with a DISTILL_WEIGHT included (configs/semkine/s2_*.yaml carry one and no DISTILL_CKPT):
    same state_dict, same random draws; a block the original refuses is refused with the same exception type."""
    paths = _model_blocks()
    assert len(paths) >= 20, len(paths)
    built, distill_raw = 0, []
    for p in paths:
        cfg = _load(p)
        try:
            a = _build(orig.MNISTModel, cfg)
        except Exception as e:
            with pytest.raises(type(e)):
                _build(MNISTModel, cfg)
            continue
        rng_a = torch.get_rng_state()
        b = _build(MNISTModel, cfg)
        assert torch.equal(rng_a, torch.get_rng_state()), p
        _assert_same_state(a, b)
        built += 1
        if float(cfg["MODEL"].get("DISTILL_WEIGHT", 0.0)) > 0 and cfg["MODEL"].get("ENCODER"):
            distill_raw.append(p.name)
    assert built >= 20, built
    print("\nraw-event configs with DISTILL_WEIGHT > 0 and no DISTILL_CKPT that still build:", distill_raw)


@pytest.mark.parametrize("path", [RT_CFG, BASE_CFG], ids=["rt_cnntrack", "dt_base"])
def test_a_default_forward_equals_the_original_bitwise(orig, path):
    a = _randomize(_build(orig.MNISTModel, _cfg(path))).eval()
    b = _randomize(_build(MNISTModel, _cfg(path))).eval()
    x, prev = _lnes(5), _prev(5)
    for B in (1, 5):
        for bf16 in (False, True):
            with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
                oa, ob = a(x[:B], prev[:B]), b(x[:B], prev[:B])
            assert oa.dtype == ob.dtype and torch.equal(oa, ob), (B, bf16)


@pytest.mark.parametrize("path", [RT_CFG, BASE_CFG], ids=["rt_cnntrack (mse_51d)", "dt_base (so3_trans_fk)"])
@pytest.mark.parametrize("bf16", [False, True], ids=["fp32", "bf16"])
def test_a_plain_training_step_equals_the_original_on_a_real_batch(orig, real_batch, path, bf16):
    """A plain tuple batch with ACCEL_WEIGHT 0: the returned (log10) loss, every logged value and every gradient, and the
    BatchNorm statistics the step wrote, bit for bit (the reference is checked against itself first)."""
    mk = lambda cls: _randomize(_build(cls, _cfg(path))).train()                       # noqa: E731
    a, a2, b = mk(orig.MNISTModel), mk(orig.MNISTModel), mk(MNISTModel)
    ra = _ts(a, copy.deepcopy(real_batch), bf16)
    _assert_same_step(ra, _ts(a2, copy.deepcopy(real_batch), bf16), "reference vs itself")
    rb = _ts(b, copy.deepcopy(real_batch), bf16)
    _assert_same_step(ra, rb, "edited vs original")
    _assert_same_state(a, b)
    assert not any(k.endswith(("_accel", "_distill")) for k in rb[1])


@pytest.fixture(scope="module")
def s37_batch():
    from semkine.events import collate_packets
    ds = _train_ds(_cfg(S37_CFG))
    assert ds.input_mode == "raw_packed"
    return collate_packets([ds[i] for i in (3000, 700000)])


def test_a_packed_s37_path_is_the_originals(orig, s37_batch):
    """configs/rt/rt_s37_2k.yaml (S37 routed readout, the packed path): same state_dict and random draws, the same
    `forward_packet` output in eval mode, and the same `training_step` (loss, logs, gradients) on a real two-packet batch."""
    cfg = _cfg(S37_CFG)
    a = _build(orig.MNISTModel, cfg)
    rng_a = torch.get_rng_state()
    b = _build(MNISTModel, cfg)
    assert torch.equal(rng_a, torch.get_rng_state())
    _assert_same_state(a, b)
    a, b = _randomize(a).eval(), _randomize(b).eval()
    outs = []
    for m in (a, b):
        torch.manual_seed(5)
        with torch.no_grad():
            outs.append(m.forward_packet(s37_batch))
    assert outs[0].dtype == outs[1].dtype and torch.equal(outs[0], outs[1])
    a, b = a.train(), b.train()
    steps = []
    for m in (a, b):
        torch.manual_seed(5)
        steps.append(_ts(m, s37_batch))
    _assert_same_step(steps[0], steps[1], "S37 training_step")


# ============================================================================================== (b) acceleration loss
def test_b_dict_batch_is_read_through_its_main_window(orig, trip):
    """The reviewer's finding: the original `_unpack_batch` has no dict branch (KeyError: 0). The edit reads `w2`, and the
    forward / loss of the dict batch are those of the plain w2 tuple, bit for bit."""
    cfg, _, batch = trip
    with pytest.raises(KeyError):
        _build(orig.MNISTModel, _cfg(BASE_CFG))._predict_batch(batch)
    m = _acc_model(cfg)
    assert [torch.equal(a, b) for a, b in zip(m._unpack_batch(batch), batch["w2"])] == [True] * 5
    m2 = _acc_model(cfg)
    pa, ya, ba, _ = m._predict_batch(batch)
    pb, yb, bb, _ = m2._predict_batch(tuple(batch["w2"]))
    assert torch.equal(pa, pb) and torch.equal(ya, yb) and torch.equal(ba, bb)


def test_b_accel_is_exactly_zero_when_predictions_equal_targets(stub):
    B, valid = 5, [True, False, True, True, False]
    g = [_prev(B, seed=k + 1) for k in range(3)]
    v = torch.tensor(valid)
    _stub_forward(stub, torch.cat([g[0][v], g[1][v]]))
    l_acc = stub._accel_loss(_synthetic_triplet(valid, g), g[2].clone())
    assert l_acc.dtype == torch.float32 and float(l_acc) == 0.0


def test_b_accel_is_zero_for_a_common_translation_offset_on_a_constant_velocity_trajectory(stub):
    """Constant velocity in translation (g_k = a + k v) and predictions = targets + one common translation offset: the
    second difference of the joints is 0 on both sides, and so is the term (float32 rounding of the FK sums: < 1e-6)."""
    B, valid = 4, [True, True, True, True]
    base = _prev(B, seed=3)
    vel = torch.tensor([0.004, -0.002, 0.003])
    g = []
    for k in range(3):
        s = base.clone()
        s[:, :3] += k * vel
        g.append(s)
    off = torch.tensor([0.05, 0.01, -0.03])
    p = []
    for k in range(3):
        s = g[k].clone()
        s[:, :3] += off
        p.append(s)
    batch = _synthetic_triplet(valid, g)
    _stub_forward(stub, torch.cat([p[0], p[1]]))
    l_acc = stub._accel_loss(batch, p[2].clone())
    assert float(l_acc) < 1e-6, float(l_acc)
    # and the term is not blind: the same trajectory with the middle prediction moved by 1 cm is 5 mm / axis off
    p1 = p[1].clone()
    p1[:, 0] += 0.01
    _stub_forward(stub, torch.cat([p[0], p1]))
    bumped = stub._accel_loss(batch, p[2].clone())
    assert abs(float(bumped) - 0.02 / 3.0) < 2e-6, float(bumped)       # |-2 * 0.01| on one of the three axes, mean over axes


def test_b_accel_equals_the_hand_computed_second_difference(stub):
    """Only the translation differs between windows, so every joint moves with it and the term is
    mean over valid rows and axes of |(t2 - 2 t1 + t0)_pred - (t2 - 2 t1 + t0)_gt|, computed here by hand."""
    B, valid = 4, [True, True, False, True]
    base = _prev(B, seed=4)
    row = torch.tensor([1.0, 1.5, 2.0, 2.5]).view(1, B, 1)
    tp = torch.tensor([[[0.010, -0.020, 0.030]] * B, [[0.015, -0.010, 0.020]] * B, [[0.030, 0.000, 0.010]] * B]) * row
    tg = torch.tensor([[[0.000, 0.000, 0.000]] * B, [[0.004, 0.001, -0.002]] * B, [[0.010, 0.003, 0.000]] * B])
    p, g = [], []
    for k in range(3):
        s = base.clone()
        s[:, :3] += tp[k]
        p.append(s)
        s = base.clone()
        s[:, :3] += tg[k]
        g.append(s)
    v = torch.tensor(valid)
    _stub_forward(stub, torch.cat([p[0][v], p[1][v]]))
    l_acc = stub._accel_loss(_synthetic_triplet(valid, g), p[2].clone())
    dp = (tp[2] - 2 * tp[1] + tp[0]).double().numpy()[valid]
    dg = (tg[2] - 2 * tg[1] + tg[0]).double().numpy()[valid]
    hand = np.abs(dp - dg).mean()
    assert abs(float(l_acc) - hand) < 1e-6 and hand > 1e-3, (float(l_acc), hand)


def _reference_accel(m, p, g, betas, valid):
    """Row by row, one MANO call per window and float64 differences: the definition, without any batching."""
    vals = []
    for i in torch.nonzero(torch.as_tensor(valid)).flatten().tolist():
        J = lambda s: m._fk(s[i:i + 1], betas[i:i + 1])[1][0].detach().double()                 # noqa: E731
        d = (J(p[2]) - 2 * J(p[1]) + J(p[0])) - (J(g[2]) - 2 * J(g[1]) + J(g[0]))
        vals.append(d.abs().mean())
    return float(torch.stack(vals).mean())


def test_b_accel_equals_a_row_by_row_reference_and_the_window_roles_matter(stub):
    B, valid = 6, [True, False, True, True, False, True]
    p = [_prev(B, seed=10 + k, norm=1.5 + 0.4 * k) for k in range(3)]
    g = [_prev(B, seed=20 + k, norm=1.6 + 0.3 * k) for k in range(3)]
    betas = 0.3 * torch.randn(B, 10, generator=torch.Generator().manual_seed(8))               # one shape per sample
    batch = _synthetic_triplet(valid, g, betas=betas)
    v = torch.tensor(valid)
    _stub_forward(stub, torch.cat([p[0][v], p[1][v]]))
    got = float(stub._accel_loss(batch, p[2].clone()))
    want = _reference_accel(stub, p, g, betas, valid)
    assert abs(got - want) < max(2e-6, 1e-4 * want) and want > 1e-3, (got, want)
    # window roles: the middle prediction is p1 (swapping it with p0 changes the term), p0 is the first half of the forward
    _stub_forward(stub, torch.cat([p[1][v], p[0][v]]))
    swapped = float(stub._accel_loss(batch, p[2].clone()))
    assert abs(swapped - want) > 1e-3
    assert abs(swapped - _reference_accel(stub, [p[1], p[0], p[2]], g, betas, valid)) < max(2e-6, 1e-4 * want)
    # the targets of w0 / w1 are the first two targets
    b2 = {**batch, "w0": batch["w1"], "w1": batch["w0"]}
    _stub_forward(stub, torch.cat([p[0][v], p[1][v]]))
    assert abs(float(stub._accel_loss(b2, p[2].clone())) - want) > 1e-3
    # the betas of the w2 rows are used: other shapes, other joints
    b3 = _synthetic_triplet(valid, g, betas=betas.flip(0))
    assert abs(float(stub._accel_loss(b3, p[2].clone())) - want) > 1e-7


def test_b_extra_forward_gets_the_valid_rows_of_w0_and_w1_with_their_own_inputs(stub):
    """Teacher forcing: one extra forward on [valid rows of w0; valid rows of w1], each row with its own lnes, its own prev
    state, its own betas and camera, with the BatchNorm running statistics frozen during it and restored after."""
    B, valid = 6, [True, False, True, True, False, True]
    v = torch.tensor(valid)
    p = [_prev(B, seed=50 + k) for k in range(3)]
    g = [_prev(B, seed=60 + k) for k in range(3)]
    betas = 0.3 * torch.randn(B, 10, generator=torch.Generator().manual_seed(12))
    batch = _synthetic_triplet(valid, g, betas=betas)
    for k, w in enumerate((batch["w0"], batch["w1"])):                    # make betas differ between windows too
        w[3] = w[3] + 0.01 * (k + 1)
    calls, flags = [], []
    out01 = torch.cat([p[0][v], p[1][v]])

    def fwd(x, prevpos, betas=None, camera_K=None):
        calls.append((x, prevpos, betas, camera_K))
        flags.append([m.track_running_stats for m in stub.modules() if isinstance(m, torch.nn.BatchNorm2d)])
        return out01

    stub.forward = fwd
    stub._accel_loss(batch, p[2].clone())
    assert len(calls) == 1
    x, prevpos, b, K = calls[0]
    w0, w1 = batch["w0"], batch["w1"]
    assert torch.equal(x, torch.cat([w0[0][v], w1[0][v]])) and torch.equal(prevpos, torch.cat([w0[1][v], w1[1][v]]))
    assert torch.equal(b, torch.cat([w0[3][v], w1[3][v]])) and torch.equal(K, torch.cat([w0[4][v], w1[4][v]]))
    assert not torch.equal(w0[1], w1[1]) and not torch.equal(w0[1], batch["w2"][1])     # the windows really differ
    assert flags[0] and not any(flags[0])                                  # frozen during the forward ...
    assert all(m.track_running_stats for m in stub.modules() if isinstance(m, torch.nn.BatchNorm2d))   # ... restored after


def test_b_invalid_rows_take_no_part_and_get_no_gradient(stub):
    B, valid = 6, [True, False, True, True, False, True]
    v = torch.tensor(valid)
    p = [_prev(B, seed=30 + k) for k in range(3)]
    g = [_prev(B, seed=40 + k) for k in range(3)]
    betas = 0.3 * torch.randn(B, 10, generator=torch.Generator().manual_seed(9))
    res = []
    for poison in (False, True):
        batch = _synthetic_triplet(valid, g, betas=betas, garbage=float("nan") if poison else None)
        p2 = p[2].clone()
        if poison:
            batch["w2"][2][~v] = 1e3                                          # targets and predictions of the invalid rows
            p2[~v] = 1e3
        p2.requires_grad_(True)
        _stub_forward(stub, torch.cat([p[0][v], p[1][v]]))
        l_acc = stub._accel_loss(batch, p2)
        l_acc.backward()
        res.append((float(l_acc), p2.grad.clone()))
    assert res[0][0] == res[1][0] and np.isfinite(res[1][0]) and res[0][0] > 1e-3
    for _, gr in res:
        assert torch.isfinite(gr).all() and torch.equal(gr[~v], torch.zeros_like(gr[~v])) and gr[v].abs().sum() > 0
    assert torch.equal(res[0][1], res[1][1])
    # the term over a batch made of the valid rows alone is the same
    sub = _synthetic_triplet([True] * 4, [x[v] for x in g], betas=betas[v])
    _stub_forward(stub, torch.cat([p[0][v], p[1][v]]))
    assert abs(float(stub._accel_loss(sub, p[2][v].clone())) - res[0][0]) < 1e-7


def test_b_real_model_ignores_the_content_of_invalid_rows(trip):
    cfg, _, batch = trip
    out = []
    for poison in (False, True):
        b = copy.deepcopy(batch)
        if poison:
            inv = ~b["valid"]
            for k in ("w0", "w1"):
                for t in b[k]:
                    t[inv] = float("nan")
        m = _acc_model(cfg)
        pred, _, _, _ = m._predict_batch(b)
        out.append(m._accel_loss(b, pred))
    assert torch.isfinite(out[1]) and torch.equal(out[0], out[1]) and float(out[0]) > 1e-3


def test_b_no_valid_row_gives_a_constant_zero_and_the_plain_step(trip):
    cfg, _, batch = trip
    b = copy.deepcopy(batch)
    b["valid"] = torch.zeros_like(b["valid"])
    m = _acc_model(cfg)
    pred, _, _, _ = m._predict_batch(b)
    l_acc = m._accel_loss(b, pred)
    assert l_acc.shape == () and float(l_acc) == 0.0 and not l_acc.requires_grad
    got = _ts(_acc_model(cfg), b)
    want = _ts(_acc_model(cfg, ACCEL_WEIGHT=0.0), tuple(b["w2"]))
    assert float(got[1]["train_loss_accel"]) == 0.0
    assert torch.equal(got[0], want[0])                                       # log10 of the main term alone, bit for bit
    for n in want[2]:
        assert torch.equal(got[2][n], want[2][n]), n


def test_b_main_term_is_the_plain_loss_of_the_w2_rows_and_the_total_is_main_plus_w_accel(trip):
    cfg, _, batch = trip
    w = 1.7
    m, ref = _acc_model(cfg, ACCEL_WEIGHT=w), _acc_model(cfg, ACCEL_WEIGHT=0.0)
    out, logged, _ = _ts(m, batch)
    _, ref_logged, _ = _ts(ref, tuple(batch["w2"]))            # the normal-batch loss on the same w2 rows (all six rows)
    l_acc = float(logged["train_loss_accel"])
    assert l_acc > 1e-3 and "train_loss_accel" not in ref_logged
    main = float(ref_logged["train_loss"])
    assert abs(float(logged["train_loss"]) - (main + w * l_acc)) < 1e-6
    assert abs(float(out) - math.log10(main + w * l_acc)) < 2e-6          # the log10 is applied to the sum
    for k in ("train_mano_loss", "train_pos_loss", "train_rot_loss", "train_loss_rot", "train_loss_trans", "train_loss_fk",
              "train_legacy_loss_51d", "train_rotation_error_deg", "train_joint_error_mm"):
        assert torch.equal(logged[k], ref_logged[k]), k                         # the main term's parts: all rows, bitwise
    assert set(logged) - set(ref_logged) == {"train_loss_accel"}


def test_b_gradient_is_main_plus_w_times_accel_and_finite_for_every_parameter(trip):
    """d log10(L_main + w L_acc) = (dL_main + w dL_acc) / (L ln 10): the step's gradient, undone, equals the gradients of
    the two terms taken separately on identical models, and every trainable parameter has a finite gradient."""
    cfg, _, batch = trip
    w = 2.0
    _, _, g_tot = _ts(_acc_model(cfg, ACCEL_WEIGHT=w), batch)
    mm = _acc_model(cfg, ACCEL_WEIGHT=0.0)
    mm.zero_grad(set_to_none=True)
    pred, y, betas, _ = mm._predict_batch(tuple(batch["w2"]))
    loss, _ = mm._compute_loss(pred, y, betas)
    loss.backward()
    g_main = {n: p.grad.clone() for n, p in mm.named_parameters()}
    ma = _acc_model(cfg, ACCEL_WEIGHT=w)
    ma.zero_grad(set_to_none=True)
    pred, _, _, _ = ma._predict_batch(batch)
    l_acc = ma._accel_loss(batch, pred)
    (w * l_acc).backward()
    g_acc = {n: p.grad.clone() for n, p in ma.named_parameters()}
    total = float(loss) + w * float(l_acc)
    worst, n_checked = 0.0, 0
    for n, p in ma.named_parameters():
        if not p.requires_grad:
            continue
        assert g_tot[n] is not None and torch.isfinite(g_tot[n]).all(), n
        want = (g_main[n] + g_acc[n]).double()
        got = g_tot[n].double() * total * math.log(10.0)
        worst = max(worst, float((got - want).abs().max()) / max(float(want.abs().max()), 1e-12))
        n_checked += 1
    assert n_checked > 60 and worst < 2e-3, (n_checked, worst)
    assert sum(float(g_acc[n].abs().sum()) for n in g_acc) > 0                  # the term does move the network


def test_b_batchnorm_running_statistics_are_those_of_the_plain_step(trip):
    """The extra forward normalises with its own batch but must not write the running statistics (the S22 lesson,
    `_frozen_bn_stats`): after a triplet step they equal those of the plain step on the same w2 rows, bit for bit."""
    cfg, _, batch = trip
    a, b = _acc_model(cfg), _acc_model(cfg, ACCEL_WEIGHT=0.0)
    _ts(a, batch)
    _ts(b, tuple(batch["w2"]))
    sa, sb = _bn_buffers(a), _bn_buffers(b)
    assert list(sa) == list(sb) and len(sa) >= 60
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert all(m.track_running_stats for m in a.modules() if isinstance(m, torch.nn.BatchNorm2d))   # restored


def test_b_guards_raise_value_error():
    # (1) a positive weight without triplet data would never be computed
    for data in ({}, {"TRIPLET": False}, {"TRIPLET": "true"}):
        cfg = _cfg(ACC_CFG)
        cfg["DATA"].pop("TRIPLET")
        cfg["DATA"].update(data)
        with pytest.raises(ValueError, match="TRIPLET"):
            MNISTModel(cfg)
    # (2) a raw-event (EventGNN / routed) model has no (lnes, prev, target, betas, K) windows
    with pytest.raises(ValueError, match="raw-event"):
        MNISTModel(_cfg(S37_CFG, loss={"ACCEL_WEIGHT": 1.0}, data={"TRIPLET": True}))
    # (3) the term needs the model's MANO layer (PREV_RENDER) and a sane weight
    with pytest.raises(ValueError, match="PREV_RENDER"):
        MNISTModel(_cfg(ACC_CFG, model={"PREV_RENDER": False, "ZERO_EVENT_GATE": False}))
    for bad in (-1.0, float("nan"), float("inf"), "abc", None):
        with pytest.raises(ValueError, match="ACCEL_WEIGHT"):
            MNISTModel(_cfg(ACC_CFG, loss={"ACCEL_WEIGHT": bad}))
    # the same configs with the weight off build (TRIPLET alone is not an error: the dict batch is read through w2)
    assert MNISTModel(_cfg(ACC_CFG, loss={"ACCEL_WEIGHT": 0.0})).accel_weight == 0.0
    assert MNISTModel(_cfg(ACC_CFG)).accel_weight == 1.0


def test_b_plain_batches_keep_working(trip):
    cfg, _, batch = trip
    # weight 0 with a triplet dict (DATA.TRIPLET without the term): the main window only, bit for bit the plain step
    got = _ts(_acc_model(cfg, ACCEL_WEIGHT=0.0), batch)
    want = _ts(_acc_model(cfg, ACCEL_WEIGHT=0.0), tuple(batch["w2"]))
    _assert_same_step(want, got, "dict batch, weight 0")
    # the debug harness and validation feed plain tuples: forward and loss work with the dt_acc model; `training_step`
    # refuses a plain tuple only because it would silently train without the term
    m = _acc_model(cfg)
    pred, y, betas, _ = m._predict_batch(tuple(batch["w2"]))
    loss, parts = m._compute_loss(pred, y, betas)
    assert torch.isfinite(loss)
    logged = _capture_logs(m)
    m.validation_step(tuple(batch["w2"]), 0)
    assert "val_loss" in logged and torch.isfinite(logged["val_loss"])
    with pytest.raises(ValueError, match="triplet"):
        m.training_step(tuple(batch["w2"]), 0)


def test_b_step_time_with_and_without_the_triplet(trip):
    """CPU, B = 16: one training step (forward, loss, backward) of the plain batch against the triplet batch, at the
    dataset's natural valid fraction (0.331; here 5 of 16 rows) and with every row valid. Written to
    $DT_MODEL2_TIMING_JSON if set."""
    cfg, ds, _ = trip
    hist = ds.index[:, 1] - ds.index[:, 2] + 1
    deep = np.nonzero(hist >= 900)[0]
    rng = np.random.default_rng(0)
    batch = default_collate([ds[int(i)] for i in rng.choice(deep, 16, replace=False)])
    assert batch["valid"].all()
    natural = copy.deepcopy(batch)
    natural["valid"] = torch.arange(16) < 5

    def step_time(model, b, reps=3):
        _ts(model, b)                                                    # warm-up
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            _ts(model, b)
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts)), ts

    m = _acc_model(cfg)
    t_plain, ts_plain = step_time(_acc_model(cfg, ACCEL_WEIGHT=0.0), tuple(batch["w2"]))
    t_nat, ts_nat = step_time(m, natural)
    t_all, ts_all = step_time(m, batch)
    res = {"batch": 16, "threads": torch.get_num_threads(), "plain_s": t_plain, "triplet_natural_s": t_nat,
           "triplet_all_valid_s": t_all, "valid_rows_natural": 5, "ratio_natural": t_nat / t_plain,
           "ratio_all_valid": t_all / t_plain, "reps_s": {"plain": ts_plain, "natural": ts_nat, "all_valid": ts_all}}
    print("\nstep time (CPU, B=16):", json.dumps(res))
    if os.environ.get("DT_MODEL2_TIMING_JSON"):
        Path(os.environ["DT_MODEL2_TIMING_JSON"]).write_text(json.dumps(res, indent=1))
    assert 1.05 < res["ratio_natural"] < 3.0 and res["ratio_all_valid"] < 5.0, res


# ============================================================================================= (c) dense distillation
def _student_cfg(ckpt, model=None, loss=None):
    return _cfg(W05_CFG, model={"DISTILL_WEIGHT": 1.0, "DISTILL_CKPT": str(ckpt), **(model or {})}, loss=loss)


def _save_lightning_ckpt(m, path):
    import pytorch_lightning as pl
    ck = {"state_dict": m.state_dict(), "hyper_parameters": {}, "pytorch-lightning_version": pl.__version__,
          "epoch": 0, "global_step": 0}
    m.on_save_checkpoint(ck)
    torch.save(ck, path)


@pytest.fixture(scope="module")
def teacher_ckpt(tmp_path_factory):
    """A dense default-trunk model made on the fly (randomised so that it does something), saved as a Lightning-style
    checkpoint dict, the way `train.py` finds a teacher on disk."""
    path = tmp_path_factory.mktemp("kd") / "teacher.ckpt"
    t = _randomize(_build(MNISTModel, _cfg(BASE_CFG), seed=99), seed=99).eval()
    _save_lightning_ckpt(t, path)
    return path, t


def _student(teacher_ckpt, train_mod, seed=1234, model=None, loss=None):
    """A w0.5 student with the teacher attached the way `train.py` does, in train mode (what `Trainer.fit` leaves)."""
    path, _ = teacher_ckpt
    cfg = _student_cfg(path, model=model, loss=loss)
    m = _build(MNISTModel, cfg, seed)
    m.teacher = train_mod.load_distill_teacher(str(path), cfg)
    return m.train()


def test_c_teacher_config_drops_cnn_backbone_and_keeps_the_architecture_keys(train_mod):
    old_drop = ("ENCODER", "DISTILL_", "ACTIVE_")                      # semkine/train.py before the edit
    assert train_mod.TEACHER_DROP == old_drop + ("CNN_BACKBONE",)
    cfg = _cfg(W05_CFG, model={"DISTILL_WEIGHT": 1.0, "DISTILL_CKPT": "x.ckpt", "CAM_PLANES": True, "ROOT_COMPOSE": "so3",
                               "PREV_MLP_TRANSL": False})
    before = copy.deepcopy(cfg)
    t = train_mod.teacher_config(cfg)
    assert cfg == before                                                # the student's config is not touched
    for k in ("CNN_BACKBONE", "DISTILL_WEIGHT", "DISTILL_CKPT"):
        assert k in cfg["MODEL"] and k not in t["MODEL"]
    for k in ("CAM_PLANES", "ROOT_COMPOSE", "PREV_MLP_TRANSL", "PREV_RENDER", "BACKBONE", "OUTPUT_DIM"):
        assert t["MODEL"][k] == cfg["MODEL"][k]
    assert {k for k in t if k != "MODEL"} == {k for k in cfg if k != "MODEL"}
    # for every student without CNN_BACKBONE the result is exactly the inline expression of the original train.py
    for p in (BASE_CFG, RT_CFG, S37_CFG, REPO / "configs" / "semkine" / "s2_sparse_cell.yaml"):
        c = copy.deepcopy(_load(p))
        want = dict(c)
        want["MODEL"] = {k: v for k, v in c["MODEL"].items()
                         if not any(k == d.rstrip("_") or k.startswith(d) for d in old_drop)}
        assert train_mod.teacher_config(c) == want, p


def test_c_the_old_drop_tuple_could_not_load_its_own_teacher(teacher_ckpt, train_mod):
    """What the edit fixes: with CNN_BACKBONE left in, the teacher of a w0.5 student is a w0.5 trunk and refuses the
    default-trunk teacher checkpoint (the model-core reviewer's check 6f)."""
    path, _ = teacher_ckpt
    cfg = _student_cfg(path)
    old = dict(cfg)
    old["MODEL"] = {k: v for k, v in cfg["MODEL"].items()
                    if not any(k == d.rstrip("_") or k.startswith(d) for d in ("ENCODER", "DISTILL_", "ACTIVE_"))}
    assert old["MODEL"]["CNN_BACKBONE"] == "resnet18_w0.5"
    with pytest.raises(RuntimeError, match="size mismatch"):
        MNISTModel.load_from_checkpoint(str(path), cfg=old, map_location="cpu")
    t = train_mod.load_distill_teacher(str(path), cfg)
    assert sum(p.numel() for p in t.parameters()) == 11_209_429


def test_c_teacher_is_loaded_frozen_and_equal_to_the_saved_model(teacher_ckpt, train_mod):
    path, saved = teacher_ckpt
    t = train_mod.load_distill_teacher(str(path), _student_cfg(path))
    assert not t.training and all(not p.requires_grad for p in t.parameters())
    x, prev = _lnes(3, seed=1), _prev(3, seed=2)
    with torch.no_grad():
        assert torch.equal(t(x, prev), saved(x, prev))


def test_c_real_rt_cnntrack_checkpoint_loads_as_the_teacher_of_a_w05_student(train_mod):
    if not REAL_TEACHER.exists():
        pytest.skip(f"{REAL_TEACHER} not available")
    t = train_mod.load_distill_teacher(str(REAL_TEACHER), _student_cfg(REAL_TEACHER))
    assert not t.training and sum(p.numel() for p in t.parameters()) == 11_209_429
    assert t.loss_type == "so3_trans_fk"            # the student's loss block, not rt_cnntrack's mse_51d (the teacher only forwards)


def test_c_distill_term_is_the_training_objective_between_student_and_teacher(teacher_ckpt, train_mod, real_batch):
    w = 0.7
    path, _ = teacher_ckpt
    m = _student(teacher_ckpt, train_mod, model={"DISTILL_WEIGHT": w})
    ref = _build(MNISTModel, _student_cfg(path, model={"DISTILL_WEIGHT": 0.0}), 1234).train()
    _, logged, _ = _ts(m, copy.deepcopy(real_batch))
    _, ref_logged, _ = _ts(ref, copy.deepcopy(real_batch))
    d = float(logged["train_loss_distill"])
    assert np.isfinite(d) and d > 1e-3 and "train_loss_distill" not in ref_logged
    assert abs(float(logged["train_loss"]) - (float(ref_logged["train_loss"]) + w * d)) < 1e-6
    # by hand: the teacher's eval-mode output on the same input, the main objective applied to (student, teacher)
    h = _student(teacher_ckpt, train_mod)
    pred, y, betas, _ = h._predict_batch(copy.deepcopy(real_batch))
    with torch.no_grad():
        t_out = h.teacher.eval()(real_batch[0], real_batch[1], betas=real_batch[3], camera_K=real_batch[4])
    hand = float(h._compute_loss(pred, t_out, betas)[0])
    assert abs(hand - d) < 1e-6
    assert abs(hand - float(h._compute_loss(pred, y, betas)[0])) > 1e-3        # it is not the loss against the ground truth
    assert m.loss_type == "so3_trans_fk"


def test_c_other_loss_types_use_the_mse(teacher_ckpt, train_mod, real_batch):
    m = _student(teacher_ckpt, train_mod, loss={"TYPE": "mse_51d"})
    pred, _, betas, _ = m._predict_batch(copy.deepcopy(real_batch))
    loss, parts = m._compute_loss(pred, real_batch[2], betas)
    with torch.no_grad():
        t_out = m.teacher.eval()(real_batch[0], real_batch[1], betas=real_batch[3], camera_K=real_batch[4])
    out_loss, out_parts = m._maybe_distill(pred, copy.deepcopy(real_batch), loss, dict(parts))
    assert torch.equal(out_parts["loss_distill"], F.mse_loss(pred, t_out))
    assert torch.equal(out_loss, loss + 1.0 * out_parts["loss_distill"])


def test_c_gradients_reach_only_the_student_and_the_teacher_stays_frozen(teacher_ckpt, train_mod, real_batch):
    m = _student(teacher_ckpt, train_mod)
    t = m.teacher
    assert t.training is True                                             # Lightning's model.train() flips the submodule ...
    snap = {k: v.clone() for k, v in t.state_dict().items()}
    _, logged, grads = _ts(m, copy.deepcopy(real_batch))
    assert t.training is False                                            # ... and the distillation puts it back to eval
    assert all(torch.equal(snap[k], v) for k, v in t.state_dict().items())  # parameters and BatchNorm statistics untouched
    assert all(p.grad is None and not p.requires_grad for p in t.parameters())
    student_params = [(n, p) for n, p in m.named_parameters() if not n.startswith("teacher.")]
    assert len(student_params) > 50 and all(p.grad is not None and torch.isfinite(p.grad).all() for _, p in student_params)
    # the distillation term alone moves the student: its gradient is non-zero on the trunk
    m.zero_grad(set_to_none=True)
    pred, _, _, _ = m._predict_batch(copy.deepcopy(real_batch))
    _, parts = m._maybe_distill(pred, copy.deepcopy(real_batch), torch.zeros(()), {})
    parts["loss_distill"].backward()
    assert float(m.rn.fc.weight.grad.abs().sum()) > 0 and all(p.grad is None for p in t.parameters())
    # and the optimiser sees the student only
    opt_params = m.configure_optimizers()
    opt_params = opt_params["optimizer"] if isinstance(opt_params, dict) else opt_params
    ids = {id(p) for g in opt_params.param_groups for p in g["params"]}
    assert not ids & {id(p) for p in t.parameters()} and len(ids) == len(student_params)


def test_c_teacher_is_not_in_the_students_checkpoint(teacher_ckpt, train_mod):
    m = _student(teacher_ckpt, train_mod)
    bare = _build(MNISTModel, _student_cfg(teacher_ckpt[0]), 1234)
    sd = m.state_dict()
    assert any(k.startswith("teacher.") for k in sd)                       # the submodule is in state_dict ...
    ck = {"state_dict": dict(sd)}
    m.on_save_checkpoint(ck)                                               # ... and left out of the saved file
    assert not any(k.startswith("teacher.") for k in ck["state_dict"])
    assert list(ck["state_dict"]) == list(bare.state_dict())
    bare.load_state_dict(ck["state_dict"], strict=True)
    ck2 = {"state_dict": dict(sd)}                                         # a file that does have the teacher loads too
    bare.on_load_checkpoint(ck2)
    assert not any(k.startswith("teacher.") for k in ck2["state_dict"])


def test_c_a_resumed_student_gets_its_teacher_back(teacher_ckpt, train_mod):
    """`train.py --resume`: the student, with its teacher already attached, loads a file that has no teacher in it. Lightning
    loads strictly, so `on_load_checkpoint` hands the teacher its own tensors back (and without a teacher nothing is added)."""
    m = _student(teacher_ckpt, train_mod)
    saved = {"state_dict": dict(m.state_dict())}
    m.on_save_checkpoint(saved)                                          # what the file holds
    assert not any(k.startswith("teacher.") for k in saved["state_dict"])
    m2 = _student(teacher_ckpt, train_mod, seed=77)
    teacher_before = {k: v.clone() for k, v in m2.teacher.state_dict().items()}
    ck = {"state_dict": dict(saved["state_dict"])}
    m2.on_load_checkpoint(ck)
    assert sum(k.startswith("teacher.") for k in ck["state_dict"]) == len(teacher_before) > 100
    m2.load_state_dict(ck["state_dict"], strict=True)                    # the strict load Lightning does
    assert all(torch.equal(teacher_before[k], v) for k, v in m2.teacher.state_dict().items())
    ref = m.state_dict()
    assert all(torch.equal(v, ref[k]) for k, v in m2.state_dict().items() if not k.startswith("teacher."))
    bare = _build(MNISTModel, _student_cfg(teacher_ckpt[0]), 5)           # no teacher: the file is loaded as before
    ck = {"state_dict": dict(saved["state_dict"])}
    bare.on_load_checkpoint(ck)
    assert list(ck["state_dict"]) == list(saved["state_dict"])
    bare.load_state_dict(ck["state_dict"], strict=True)


def test_c_guards(teacher_ckpt):
    path, _ = teacher_ckpt
    with pytest.raises(ValueError, match="DISTILL_CKPT"):
        MNISTModel(_cfg(W05_CFG, model={"DISTILL_WEIGHT": 1.0}))
    with pytest.raises(ValueError, match="DISTILL_CKPT"):
        MNISTModel(_cfg(W05_CFG, model={"DISTILL_WEIGHT": 0.5, "DISTILL_CKPT": ""}))
    m = MNISTModel(_student_cfg(path))                                       # key present: builds, the teacher comes later
    assert m.teacher is None and m.distill_weight == 1.0
    batch = (_lnes(2), _prev(2), _prev(2, seed=1), torch.zeros(2, 10), K_RENDER_INPUT.expand(2, 3, 3).clone())
    with pytest.raises(RuntimeError, match="no teacher"):
        m._maybe_distill(torch.zeros(2, 51), batch, torch.zeros(()), {})    # a skipped term would be silent
    assert MNISTModel(_cfg(W05_CFG)).distill_weight == 0.0                   # weight 0 needs no checkpoint


def test_c_packed_path_is_the_originals(orig, teacher_ckpt, train_mod):
    """The EventPacketBatch path of `_maybe_distill` is untouched: no lnes -> unchanged; lnes -> MSE to the teacher; a pair
    distils on its main window; no teacher -> unchanged. Compared with the pristine implementation."""
    m = _student(teacher_ckpt, train_mod)
    o = orig.MNISTModel(copy.deepcopy(m.cfg))
    o.teacher, o.distill_weight = m.teacher, m.distill_weight
    pred = torch.randn(3, 51, requires_grad=True)
    mk = lambda lnes: types.SimpleNamespace(events=torch.zeros(1, 5), lnes=lnes, prev_state=_prev(3),            # noqa: E731
                                            betas=torch.zeros(3, 10), camera_K=K_RENDER_INPUT.expand(3, 3, 3))
    loss = torch.tensor(0.5)
    for lnes in (None, _lnes(3)):
        packed = mk(lnes)
        a = o._maybe_distill(pred, packed, loss, {})
        b = m._maybe_distill(pred, packed, loss, {})
        assert torch.equal(a[0], b[0]) and list(a[1]) == list(b[1])
        assert all(torch.equal(a[1][k], b[1][k]) for k in a[1])
        assert ("loss_distill" in b[1]) == (lnes is not None)
    pair = (mk(None), mk(_lnes(3)))
    assert torch.equal(o._maybe_distill(pred, pair, loss, {})[1]["loss_distill"],
                       m._maybe_distill(pred, pair, loss, {})[1]["loss_distill"])
    m.teacher = None
    parts = {}
    out = m._maybe_distill(pred, mk(_lnes(3)), loss, parts)
    assert out[0] is loss and out[1] is parts and parts == {}


# ======================================================================================= (d) consumers of a dict batch
def _load_orig_train():
    spec = importlib.util.spec_from_file_location("train_orig2", ORIG_TRAIN)
    mod = importlib.util.module_from_spec(spec)
    saved = list(sys.path)
    try:
        spec.loader.exec_module(mod)                 # it inserts its own REPO (outputs/dt here) into sys.path
    finally:
        sys.path[:] = saved
    return mod


def test_d_throughput_callback_counts_a_dict_batch(trip, train_mod):
    """The reviewer's finding: `ThroughputCallback` did `head[0].shape[0]` on the batch, which is a KeyError for a dict."""
    _, _, batch = trip
    t_orig = _load_orig_train()

    class Trainer:
        global_rank, world_size = 0, 2

    with pytest.raises(KeyError):
        t_orig.ThroughputCallback().on_train_batch_end(Trainer(), mock.MagicMock(), None, batch, 0)
    for b in (batch, list(batch["w2"]), tuple(batch["w2"])):
        cb = train_mod.ThroughputCallback()
        cb.on_train_batch_end(Trainer(), mock.MagicMock(), None, b, 0)
        assert cb.n == 6 * 2, (type(b), cb.n)
        if not isinstance(b, dict):                                       # the plain batch is counted as before
            cb_o = t_orig.ThroughputCallback()
            cb_o.on_train_batch_end(Trainer(), mock.MagicMock(), None, b, 0)
            assert cb_o.n == cb.n
    packed = types.SimpleNamespace(batch_size=4)                          # a packet batch / an unroll pair keep their rule
    for b in (packed, (packed, packed), [packed, packed]):
        cb, cb_o = train_mod.ThroughputCallback(), t_orig.ThroughputCallback()
        cb.on_train_batch_end(Trainer(), mock.MagicMock(), None, b, 0)
        cb_o.on_train_batch_end(Trainer(), mock.MagicMock(), None, b, 0)
        assert cb.n == cb_o.n == 8


def test_d_real_two_worker_loader_pin_memory_and_lightning_fit_on_cpu(trip, train_mod):
    """The train loader as `train.py` builds it (shuffle, drop_last, persistent workers, prefetch 2, default collate) with
    2 workers; torch's pin_memory recursion over the real batch; a real `Trainer.fit` of 2 optimizer steps plus a
    validation loop of plain tuples on CPU: transfer to device, logging, callbacks."""
    import pytorch_lightning as pl
    from torch.utils.data._utils import pin_memory as pm
    cfg, ds, _ = trip
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2, persistent_workers=True, prefetch_factor=2,
                        drop_last=True, generator=torch.Generator().manual_seed(0))
    it = iter(loader)
    b = next(it)
    assert sorted(b) == ["valid", "w0", "w1", "w2"] and b["valid"].dtype == torch.bool and b["valid"].shape == (4,)
    assert all(isinstance(b[k], list) and len(b[k]) == 5 and b[k][0].shape == (4, H, W, 2) for k in ("w0", "w1", "w2"))
    with mock.patch.object(torch.Tensor, "pin_memory", lambda self, *a, **k: self):
        pinned = pm.pin_memory(b)
    assert sorted(pinned) == ["valid", "w0", "w1", "w2"] and isinstance(pinned["w2"], list) and len(pinned["w2"]) == 5
    del it

    val_loader = DataLoader(_train_ds(cfg, "val_core", train=False), batch_size=4, shuffle=False, num_workers=0)
    model = _acc_model(cfg)
    cb = train_mod.ThroughputCallback()
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_steps=2, logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, callbacks=[cb], num_sanity_val_steps=0, val_check_interval=2,
                         limit_val_batches=1, log_every_n_steps=1, check_val_every_n_epoch=None, enable_model_summary=False)
    trainer.fit(model, loader, val_loader)
    m = trainer.callback_metrics
    assert trainer.global_step == 2 and cb.n >= 8
    assert "train_loss_accel" in m and "val_loss" in m and torch.isfinite(m["train_loss"])
    assert torch.isfinite(m["train_loss_accel"]) and float(m["train_loss_accel"]) > 0
