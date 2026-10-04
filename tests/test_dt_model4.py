"""DT2 package E: three default-off arms of the dense tracker `MNISTModel` and of `semkine/train.py`.

  MODEL.CNN_BACKBONE resnet18_l3w128     arm dt_dz_l3w128: the l3 trunk with a 128-wide layer3 (model/backbones.py)
  MODEL.DISTILL_WEIGHT / DISTILL_CKPT    arm dt_dz_l3w128_kd: the frozen dt_dz_l3 of the same seed teaches the student;
                                         `{SEED}` in the path is replaced by the run's seed (`train.resolve_distill_ckpt`)
                                         and the teacher is built with the trunk its own checkpoint records
                                         (`train.load_distill_teacher`; its resnet18_l3 is not the student's trunk)
  TRACK.ROLLOUT_P / ROLLOUT_RAMP         arm dt_dz_l3_roll: on a triplet batch (DATA.TRIPLET) a fraction p(step) of the
                                         rows with a real w0 / w1 gets, as the previous state of window 2, the tracker's own
                                         two-step bare feed-back (`_maybe_rollout`): no gradient, fp32, BatchNorm running
                                         statistics untouched

Pinned here:

  (a) rollout off: the dict batch, the draws, the loss, every gradient and the BatchNorm statistics are those of the model
      before the edit (outputs/dt2/ref/model_pre_e.py, sha256 pinned) bit for bit, with the new keys absent, 0 or written
      out; no extra log; the new TRACK keys are the only difference between the key sets;
  (b) rollout on: p(step) is p * clip((step - start) / (full - start), 0, 1); rows with valid false and rows that were not
      drawn keep their prev bit for bit; the replaced rows' prev equals an independently computed two-step bare feed-back
      (o0 = f(w0, prev0), o1 = f(w1, o0), one batched forward over all valid rows each); nothing but w2's prev changes (and
      the batch dict is updated in place); the BatchNorm running statistics do not move; the rollout forward runs in fp32
      under an outer bf16 autocast; no gradient flows through the rollout (the step's gradient equals that of the plain
      step on the hand-replaced batch); p and the replaced rows are logged; a real `Trainer.fit` ramps p with the optimiser
      step; every impossible configuration raises ValueError;
  (c) the l3w128 trunk inside MNISTModel (parameter counts, shapes, default names untouched);
  (d) distillation with a differently-built teacher: `{SEED}`, the teacher's trunk from its checkpoint, the term on the dense
      path (finite, non-zero, the objective between student and teacher, teacher frozen), rollout and distillation together.

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python -m pytest tests/test_dt_model4.py -q
"""
from __future__ import annotations

import copy
import functools
import hashlib
import importlib.util
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
from torch.utils.data.dataloader import default_collate

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "model")]
from config import load_config                                               # noqa: E402
from model import MNISTModel                                                 # noqa: E402

PRE_E = REPO / "outputs" / "dt2" / "ref" / "model_pre_e.py"
#: sha256 of model/model.py as found in MAIN and STAGE before the package-E edit (2026-10-04, md5 9c2021c0bea4)
PRE_E_SHA256 = "34d9c978328603c632dbdd7159608a7e6bc555dfc43354e83015b18623f95765"
DT = REPO / "configs" / "dt"
DT2 = REPO / "configs" / "dt2"
L3_CFG, ROLL_CFG, KD_CFG, W128_CFG = (DT / "dt_dz_l3.yaml", DT2 / "dt_dz_l3_roll.yaml", DT2 / "dt_dz_l3w128_kd.yaml",
                                      DT2 / "dt_dz_l3w128.yaml")
DATA_ROOT = REPO / "data" / "hand_data51"
H, W = 180, 240
NEW_TRACK_KEYS = frozenset({"ROLLOUT_P", "ROLLOUT_RAMP"})

pytestmark = pytest.mark.skipif(not DATA_ROOT.exists(), reason="hand_data51 not available")


# ----------------------------------------------------------------------------------------------------- helpers
@functools.lru_cache(maxsize=None)
def _load(path):
    return load_config(path)


def _cfg(path, model=None, track=None, data=None, loss=None):
    cfg = copy.deepcopy(_load(Path(path)))
    cfg["MODEL"].update(model or {})
    cfg["TRACK"].update(track or {})
    cfg["DATA"].update(data or {})
    cfg["LOSS"].update(loss or {})
    return cfg


def _build(cls, cfg, seed=1234):
    torch.manual_seed(seed)
    return cls(copy.deepcopy(cfg))


def _randomize(m, seed=7):
    """What a trained model has and a fresh one does not: prev_mlp's zero-initialised last layer and BatchNorm statistics
    that are not the identity."""
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


def _assert_same_state(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb)
    for k in sa:
        assert sa[k].shape == sb[k].shape and sa[k].dtype == sb[k].dtype, k
        assert torch.equal(sa[k], sb[k]), k


def _capture_logs(m):
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


def _assert_same_step(ref, got, label="", skip_logs=()):
    assert ref[0].dtype == got[0].dtype and torch.equal(ref[0], got[0]), label
    keys = [k for k in got[1] if k not in skip_logs]
    assert list(ref[1]) == keys, label
    for k in keys:
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


def _clone_batch(batch):
    """A deep copy that keeps the container types (dict of lists / tuples of tensors)."""
    out = {}
    for k, v in batch.items():
        out[k] = v.clone() if torch.is_tensor(v) else type(v)(t.clone() for t in v)
    return out


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


BASE_K = torch.tensor([[603.4507, 0.0, 325.09183], [0.0, 602.95654, 242.09796], [0.0, 0.0, 1.0]])


def _step_patch(step):
    return mock.patch.object(MNISTModel, "global_step", new_callable=mock.PropertyMock, return_value=step)


# ---------------------------------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def pre_e():
    """model/model.py before the package-E edit, imported under another module name."""
    if not PRE_E.exists():
        pytest.skip(f"{PRE_E} not available")
    assert hashlib.sha256(PRE_E.read_bytes()).hexdigest() == PRE_E_SHA256, "the reference copy was modified"
    spec = importlib.util.spec_from_file_location("model_pre_e", PRE_E)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["model_pre_e"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def train_mod():
    import semkine.train as T
    return T


@pytest.fixture(scope="module")
def trip():
    """`(cfg, batch)`: the roll arm's training split with the coin always passing (TRIPLET_FRAC 1). The batch: 6 samples
    deep inside long runs (valid for every window up to 300 ms) and 2 right at a run start (invalid): valid = [T] * 6 + [F] * 2.
    Real windows, real DomRand / noise draws, real betas and camera K'."""
    cfg = _cfg(ROLL_CFG, data={"TRIPLET_FRAC": 1.0})
    ds = _train_ds(cfg)
    assert ds.triplet and ds.triplet_frac == 1.0
    hist = ds.index[:, 1] - ds.index[:, 2] + 1
    deep, shallow = np.nonzero(hist >= 900)[0], np.nonzero(hist < 90)[0]
    sel = [int(deep[i]) for i in (0, 3000, 9000, 20000, 33000, 40000)] + [int(shallow[i]) for i in (3, 700)]
    batch = default_collate([ds[i] for i in sel])
    assert batch["valid"].dtype == torch.bool and batch["valid"].tolist() == [True] * 6 + [False] * 2
    return cfg, batch


def _roll_model(trip_cfg, seed=1234, p=0.5, ramp=(500, 2000), **model):
    cfg = copy.deepcopy(trip_cfg)
    cfg["TRACK"].update(ROLLOUT_P=p, ROLLOUT_RAMP=list(ramp))
    cfg["MODEL"].update(model)
    return _randomize(_build(MNISTModel, cfg, seed)).train()


def _expected_o01(model, batch):
    """The two-step bare feed-back, computed by hand on a COPY of the model (so the copy's own running statistics may move
    freely: a train-mode forward normalises with the batch's statistics): o0 = f(w0, prev0), o1 = f(w1, o0), each as one
    batched forward over all valid rows."""
    m = copy.deepcopy(model).train()
    v = batch["valid"].bool()
    w0, w1 = batch["w0"], batch["w1"]
    with torch.no_grad():
        o0 = m(w0[0][v], w0[1][v], betas=w0[3][v], camera_K=w0[4][v])
        o1 = m(w1[0][v], o0, betas=w1[3][v], camera_K=w1[4][v])
    return o0, o1


def _expected_o1(model, batch):
    return _expected_o01(model, batch)[1]


def _find_seed(valid, p, B):
    """A seed whose draws pick a proper subset of the valid rows, at least one of them, and also fall below p on at least
    one invalid row (which must still be left alone)."""
    for seed in range(200):
        torch.manual_seed(seed)
        u = torch.rand(B)
        take = valid & (u < p)
        if 0 < int(take.sum()) < int(valid.sum()) and bool(((u < p) & ~valid).any()):
            return seed, take
    raise AssertionError("no suitable seed")


# =============================================================================================== (a) rollout off
def test_a_pristine_reference_is_the_file_before_the_edit(pre_e):
    assert set(pre_e.MNISTModel.TRACK_KEYS) == set(MNISTModel.TRACK_KEYS) - NEW_TRACK_KEYS
    assert NEW_TRACK_KEYS <= set(MNISTModel.TRACK_KEYS) and not NEW_TRACK_KEYS & set(pre_e.MNISTModel.TRACK_KEYS)
    assert set(pre_e.MNISTModel.MODEL_KEYS) == set(MNISTModel.MODEL_KEYS)
    assert not hasattr(pre_e.MNISTModel, "_maybe_rollout") and hasattr(MNISTModel, "_maybe_rollout")
    with pytest.raises(ValueError, match="unknown TRACK keys"):
        _build(pre_e.MNISTModel, _cfg(L3_CFG, track={"ROLLOUT_P": 0.5}))
    with pytest.raises(ValueError, match="unknown TRACK keys"):
        MNISTModel(_cfg(L3_CFG, track={"ROLLOUT_PP": 0.5}))


@pytest.mark.parametrize("track", [{}, {"ROLLOUT_P": 0.0}, {"ROLLOUT_P": 0}, {"ROLLOUT_P": 0.0, "ROLLOUT_RAMP": [500, 2000]},
                                   {"ROLLOUT_RAMP": [5, 20]}], ids=str)
def test_a_default_state_dict_modules_and_rng_equal_the_pre_edit_model(pre_e, track):
    cfg_old = _cfg(L3_CFG)
    a = _build(pre_e.MNISTModel, cfg_old)
    rng_a = torch.get_rng_state()
    b = _build(MNISTModel, _cfg(L3_CFG, track=track))
    assert torch.equal(rng_a, torch.get_rng_state())
    _assert_same_state(a, b)
    assert [n for n, _ in a.named_modules()] == [n for n, _ in b.named_modules()]
    assert b.rollout_p == 0.0 and b.rollout_p_now() == 0.0


@pytest.mark.parametrize("track", [{}, {"ROLLOUT_P": 0.0, "ROLLOUT_RAMP": [5, 20]}], ids=["absent", "written-out"])
@pytest.mark.parametrize("bf16", [False, True], ids=["fp32", "bf16"])
def test_a_off_training_step_on_a_dict_batch_equals_the_pre_edit_model_bitwise(pre_e, trip, track, bf16):
    """The real triplet batch (valid and invalid rows) through `training_step`: loss, logs, gradients, BatchNorm statistics."""
    cfg, batch = trip
    cfg_off = _cfg(L3_CFG, data={"TRIPLET": True, "TRIPLET_FRAC": 1.0}, track=track)
    a = _randomize(_build(pre_e.MNISTModel, _cfg(L3_CFG, data={"TRIPLET": True}))).train()
    b = _randomize(_build(MNISTModel, cfg_off)).train()
    with _step_patch(5000):
        ra = _ts(a, _clone_batch(batch), bf16)
        rb = _ts(b, _clone_batch(batch), bf16)
    _assert_same_step(ra, rb, "off vs pre-edit")
    _assert_same_state(a, b)
    assert not any("rollout" in k for k in rb[1])
    assert b._rollout_stats is None


def test_a_off_makes_no_draw_no_forward_and_returns_the_same_object(trip):
    cfg, batch = trip
    b = _randomize(_build(MNISTModel, _cfg(L3_CFG, data={"TRIPLET": True}))).train()
    calls = []

    def no_forward(*a, **k):
        calls.append(1)
        raise AssertionError("forward must not run")

    b.forward = no_forward
    torch.manual_seed(3)
    before = torch.get_rng_state()
    out = b._maybe_rollout(batch)
    assert out is batch and torch.equal(before, torch.get_rng_state()) and not calls
    # also a plain tuple batch, an eval-mode model, and p > 0 with the model in eval mode
    assert b._maybe_rollout(tuple(batch["w2"])) is not None
    m = _roll_model(cfg).eval()
    with _step_patch(2000):
        assert m._maybe_rollout(batch) is batch                       # evaluation never rolls out (and never raises)
        assert m._maybe_rollout(tuple(batch["w2"]))[0] is batch["w2"][0]


# ============================================================================================= (b) rollout on: p(step)
@pytest.mark.parametrize("p,ramp,steps,want", [
    (0.5, (500, 2000), (0, 1, 499, 500, 501, 1250, 1999, 2000, 2001, 6000), (0, 0, 0, 0, 0.5 / 1500, 0.25, 0.5 * 1499 / 1500, 0.5, 0.5, 0.5)),
    (1.0, (5, 20), (0, 5, 6, 12, 20, 100), (0, 0, 1 / 15, 7 / 15, 1.0, 1.0)),
    (0.3, (0, 1), (0, 1, 2), (0.0, 0.3, 0.3)),
])
def test_b_p_of_step_is_the_clipped_linear_ramp(trip, p, ramp, steps, want):
    m = _roll_model(trip[0], p=p, ramp=ramp)
    for s, w in zip(steps, want):
        with _step_patch(s):
            assert m.rollout_p_now() == pytest.approx(w, abs=1e-12), (s, w)
    off = _roll_model(trip[0], p=0.0, ramp=ramp)
    with _step_patch(10_000):
        assert off.rollout_p_now() == 0.0


def test_b_default_ramp_is_500_2000_and_keys_are_whitelisted():
    cfg = _cfg(L3_CFG, data={"TRIPLET": True}, track={"ROLLOUT_P": 0.4})
    m = _build(MNISTModel, cfg)
    assert m.rollout_p == 0.4 and m.rollout_ramp == (500, 2000)
    with _step_patch(1250):
        assert m.rollout_p_now() == pytest.approx(0.2)
    r = _build(MNISTModel, _cfg(ROLL_CFG))
    assert (r.rollout_p, r.rollout_ramp) == (0.5, (500, 2000))


def test_b_no_rollout_before_the_ramp_starts_draws_nothing_and_runs_nothing(trip):
    cfg, batch = trip
    m = _roll_model(cfg)
    m.forward = lambda *a, **k: (_ for _ in ()).throw(AssertionError("forward must not run"))
    torch.manual_seed(9)
    before = torch.get_rng_state()
    with _step_patch(400):
        out = m._maybe_rollout(batch)
    assert out is batch and torch.equal(before, torch.get_rng_state())
    assert m._rollout_stats == {"p": 0.0, "rows": 0, "valid": 0}


# =============================================================================================== (b) rollout on: the rows
def test_b_replaced_rows_are_the_independent_two_step_feedback_and_everything_else_is_untouched(trip):
    cfg, batch0 = trip
    m = _roll_model(cfg)
    valid = batch0["valid"].bool()
    B = valid.shape[0]
    seed, take = _find_seed(valid, 0.5, B)
    want_o1 = _expected_o1(m, batch0)                               # on a deep copy: m itself is untouched
    batch = _clone_batch(batch0)
    w2_type = type(batch["w2"])
    bn_before = _bn_buffers(m)
    torch.manual_seed(seed)
    with _step_patch(2000):                                          # p = 0.5
        out = m._maybe_rollout(batch)
    assert out is batch and m._rollout_stats == {"p": 0.5, "rows": int(take.sum()), "valid": 6}
    assert isinstance(batch["w2"], w2_type) and len(batch["w2"]) == 5
    new_prev, old_prev = batch["w2"][1], batch0["w2"][1]
    assert new_prev.dtype == torch.float32 and not new_prev.requires_grad
    # replaced rows: exactly the independently computed o1; the others (not drawn, and the invalid ones): untouched
    vidx = valid.nonzero().reshape(-1)
    for j, row in enumerate(vidx.tolist()):
        if take[row]:
            assert torch.equal(new_prev[row], want_o1[j]), row
            assert not torch.equal(new_prev[row], old_prev[row])      # a genuinely different state
        else:
            assert torch.equal(new_prev[row], old_prev[row]), row
    for row in (~valid).nonzero().reshape(-1).tolist():
        assert torch.equal(new_prev[row], old_prev[row])
    # nothing else of the batch changed
    assert torch.equal(batch["valid"], batch0["valid"])
    for k in ("w0", "w1"):
        assert all(torch.equal(a, b) for a, b in zip(batch[k], batch0[k])), k
    for i in (0, 2, 3, 4):
        assert torch.equal(batch["w2"][i], batch0["w2"][i]), i
    # the BatchNorm running statistics and counters did not move, and the flags are back
    bn_after = _bn_buffers(m)
    assert list(bn_before) == list(bn_after) and all(torch.equal(bn_before[k], bn_after[k]) for k in bn_before)
    assert all(mod.track_running_stats for mod in m.modules() if isinstance(mod, torch.nn.BatchNorm2d))
    assert m.training and all(p.grad is None for p in m.parameters())


def test_b_w2_container_type_is_kept(trip):
    cfg, batch0 = trip
    m = _roll_model(cfg)
    for wrap in (list, tuple):
        batch = _clone_batch(batch0)
        batch["w2"] = wrap(batch["w2"])
        with _step_patch(2000):
            torch.manual_seed(0)
            m._maybe_rollout(batch)
        assert type(batch["w2"]) is wrap and len(batch["w2"]) == 5


def test_b_one_batched_forward_over_all_valid_rows_per_step(trip):
    """o0 and o1 are each ONE forward of all valid rows (not only the drawn ones: BatchNorm batch statistics), fed with
    window 0's, then window 1's tensors; w1's prev input is o0, not its own prev."""
    cfg, batch0 = trip
    m = _roll_model(cfg)
    valid = batch0["valid"].bool()
    seed, take = _find_seed(valid, 0.5, valid.shape[0])
    seen = []
    orig_forward = m.forward

    def spy(x, prev, betas=None, camera_K=None):
        seen.append((x.clone(), prev.clone(), betas.clone(), camera_K.clone()))
        return orig_forward(x, prev, betas=betas, camera_K=camera_K)

    m.forward = spy
    batch = _clone_batch(batch0)
    torch.manual_seed(seed)
    with _step_patch(2000):
        m._maybe_rollout(batch)
    assert len(seen) == 2 and all(s[0].shape[0] == 6 for s in seen)
    for call, w in zip(seen, ("w0", "w1")):
        assert torch.equal(call[0], batch0[w][0][valid]) and torch.equal(call[2], batch0[w][3][valid])
        assert torch.equal(call[3], batch0[w][4][valid])
    assert torch.equal(seen[0][1], batch0["w0"][1][valid])                       # w0 keeps its own (noisy GT) prev
    assert not torch.equal(seen[1][1], batch0["w1"][1][valid])                   # w1 reads o0 instead of its own prev
    assert torch.equal(seen[1][1], _expected_o01(m, batch0)[0])                  # ... which is the independently computed o0
    assert seen[0][0].dtype == seen[1][1].dtype == torch.float32


def test_b_rollout_forward_is_fp32_and_no_grad_under_an_outer_bf16_autocast_and_the_main_forward_is_not(trip):
    cfg, batch0 = trip
    m = _roll_model(cfg)
    valid = batch0["valid"].bool()
    seed, _ = _find_seed(valid, 0.5, valid.shape[0])
    seen = []
    h = m.rn.register_forward_hook(lambda mod, inp, out: seen.append((out.dtype, torch.is_grad_enabled(), out.requires_grad)))
    batch = _clone_batch(batch0)
    torch.manual_seed(seed)
    with _step_patch(2000), torch.autocast("cpu", dtype=torch.bfloat16):
        m._predict_batch(batch)                                                   # the rollout, then the main forward
    h.remove()
    # o0, o1: float32, grad mode off, no graph; the main forward: bf16 autocast, grad mode on, with a graph
    assert seen == [(torch.float32, False, False)] * 2 + [(torch.bfloat16, True, True)], seen
    assert batch["w2"][1].dtype == torch.float32


def test_b_rollout_on_equals_the_plain_step_on_the_hand_replaced_batch_so_no_gradient_flows_through_it(trip):
    """`training_step` with the rollout is the plain step on the batch whose w2 prev was replaced by hand: same loss, same
    logs (plus the rollout's two), every gradient, and the BatchNorm statistics the main forward wrote -- nothing more."""
    cfg, batch0 = trip
    m = _roll_model(cfg)
    ref = _roll_model(cfg, p=0.0)
    ref.load_state_dict(m.state_dict())
    valid = batch0["valid"].bool()
    seed, take = _find_seed(valid, 0.5, valid.shape[0])
    o1 = _expected_o1(m, batch0)
    hand = _clone_batch(batch0)
    prev = hand["w2"][1].clone()
    vidx = valid.nonzero().reshape(-1)
    for j, row in enumerate(vidx.tolist()):
        if take[row]:
            prev[row] = o1[j]
    hand["w2"] = list(hand["w2"])
    hand["w2"][1] = prev
    with _step_patch(2000):
        torch.manual_seed(seed)
        rm = _ts(m, _clone_batch(batch0))
    rr = _ts(ref, hand)
    assert set(rm[1]) - set(rr[1]) == {"train_rollout_p", "train_rollout_rows"}
    assert rm[1]["train_rollout_p"] == 0.5 and rm[1]["train_rollout_rows"] == float(take.sum())
    _assert_same_step(rr, rm, "rollout vs hand-replaced", skip_logs=("train_rollout_p", "train_rollout_rows"))
    _assert_same_state(ref, m)


def test_b_gradients_never_reach_the_rollout_graph(trip):
    """The replaced prev is a leaf without history: `.backward()` through the main loss must not touch a graph made by the two
    extra forwards (no_grad), so a second backward on the same batch's tensors does not raise and the rollout leaves no
    `grad_fn` on anything it hands over."""
    cfg, batch0 = trip
    m = _roll_model(cfg)
    batch = _clone_batch(batch0)
    torch.manual_seed(1)
    with _step_patch(2000):
        m._maybe_rollout(batch)
    assert batch["w2"][1].grad_fn is None and not batch["w2"][1].requires_grad
    assert all(p.grad is None for p in m.parameters())


def test_b_training_step_logs_p_and_the_replaced_rows(trip):
    cfg, batch0 = trip
    m = _roll_model(cfg)
    valid = batch0["valid"].bool()
    seed, take = _find_seed(valid, 0.5, valid.shape[0])
    for step, p in ((600, 0.5 * 100 / 1500), (2500, 0.5)):
        logged = _capture_logs(m)
        batch = _clone_batch(batch0)
        torch.manual_seed(seed)
        with _step_patch(step):
            m.training_step(batch, 0)
        assert logged["train_rollout_p"] == pytest.approx(p) and isinstance(logged["train_rollout_rows"], float)
        assert 0 <= logged["train_rollout_rows"] <= 6
    # before the ramp: p = 0 and no row
    logged = _capture_logs(m)
    with _step_patch(100):
        m.training_step(_clone_batch(batch0), 0)
    assert logged["train_rollout_p"] == 0.0 and logged["train_rollout_rows"] == 0.0


def test_b_p_one_replaces_every_valid_row_and_only_those(trip):
    cfg, batch0 = trip
    m = _roll_model(cfg, p=1.0, ramp=(0, 1))
    batch = _clone_batch(batch0)
    with _step_patch(5):
        m._maybe_rollout(batch)
    valid = batch0["valid"].bool()
    assert m._rollout_stats["rows"] == 6
    changed = (batch["w2"][1] != batch0["w2"][1]).any(dim=1)
    assert torch.equal(changed, valid)
    assert torch.equal(batch["w2"][1][valid], _expected_o1(m, batch0))


def test_b_no_valid_row_is_a_no_op(trip):
    cfg, batch0 = trip
    m = _roll_model(cfg, p=1.0, ramp=(0, 1))
    batch = _clone_batch(batch0)
    batch["valid"] = torch.zeros(8, dtype=torch.bool)
    m.forward = lambda *a, **k: (_ for _ in ()).throw(AssertionError("forward must not run"))
    with _step_patch(5):
        m._maybe_rollout(batch)
    assert m._rollout_stats == {"p": 1.0, "rows": 0, "valid": 0}
    assert torch.equal(batch["w2"][1], batch0["w2"][1])


def test_b_invalid_rows_with_garbage_windows_take_no_part(trip):
    """valid=False rows alias w2 in the real data; whatever their w0 / w1 hold they must not influence the valid rows'
    result (NaN included) and are never replaced."""
    cfg, batch0 = trip
    m = _roll_model(cfg, p=1.0, ramp=(0, 1))
    clean = _clone_batch(batch0)
    dirty = _clone_batch(batch0)
    inv = ~batch0["valid"].bool()
    for k in ("w0", "w1"):
        for t in dirty[k]:
            t[inv] = float("nan")
    with _step_patch(5):
        m._maybe_rollout(clean)
        m._maybe_rollout(dirty)
    assert torch.equal(clean["w2"][1], dirty["w2"][1]) and torch.isfinite(dirty["w2"][1]).all()
    assert torch.equal(dirty["w2"][1][inv], batch0["w2"][1][inv])


def test_b_validation_batches_of_plain_tuples_are_untouched_but_a_training_plain_batch_raises(trip):
    cfg, batch0 = trip
    m = _roll_model(cfg)
    plain = tuple(batch0["w2"])
    with _step_patch(2500):
        with pytest.raises(ValueError, match="triplet batches"):
            m._maybe_rollout(plain)
        with pytest.raises(ValueError, match="triplet batches"):
            m._maybe_rollout({"w2": plain})
        m.eval()
        assert m._maybe_rollout(plain) is plain


# ============================================================================================= (b) guards
@pytest.mark.parametrize("track,data,loss,msg", [
    ({"ROLLOUT_P": 0.5}, {"TRIPLET": False}, {}, "needs DATA.TRIPLET"),
    ({"ROLLOUT_P": 0.5}, {"TRIPLET": "false"}, {}, "needs DATA.TRIPLET"),
    ({"ROLLOUT_P": 0.5}, {"TRIPLET": True}, {"ACCEL_WEIGHT": 0.1}, "ACCEL_WEIGHT"),
    ({"ROLLOUT_P": -0.1}, {"TRIPLET": True}, {}, "probability"),
    ({"ROLLOUT_P": 1.5}, {"TRIPLET": True}, {}, "probability"),
    ({"ROLLOUT_P": True}, {"TRIPLET": True}, {}, "finite number"),
    ({"ROLLOUT_P": "0.5"}, {"TRIPLET": True}, {}, "finite number"),
    ({"ROLLOUT_P": float("nan")}, {"TRIPLET": True}, {}, "finite number"),
    ({"ROLLOUT_P": 0.5, "ROLLOUT_RAMP": [2000, 500]}, {"TRIPLET": True}, {}, "ROLLOUT_RAMP"),
    ({"ROLLOUT_P": 0.5, "ROLLOUT_RAMP": [500, 500]}, {"TRIPLET": True}, {}, "ROLLOUT_RAMP"),
    ({"ROLLOUT_P": 0.5, "ROLLOUT_RAMP": [-1, 5]}, {"TRIPLET": True}, {}, "ROLLOUT_RAMP"),
    ({"ROLLOUT_P": 0.5, "ROLLOUT_RAMP": [500]}, {"TRIPLET": True}, {}, "ROLLOUT_RAMP"),
    ({"ROLLOUT_P": 0.5, "ROLLOUT_RAMP": 500}, {"TRIPLET": True}, {}, "ROLLOUT_RAMP"),
    ({"ROLLOUT_P": 0.5, "ROLLOUT_RAMP": [5.0, 20.0]}, {"TRIPLET": True}, {}, "ROLLOUT_RAMP"),
    ({"ROLLOUT_RAMP": [20, 5]}, {}, {}, "ROLLOUT_RAMP"),                      # a bad ramp is refused even with p = 0
])
def test_b_impossible_configurations_raise(track, data, loss, msg):
    with pytest.raises(ValueError, match=msg):
        MNISTModel(_cfg(L3_CFG, track=track, data=data, loss=loss))


def test_b_raw_event_models_are_refused():
    S37 = REPO / "configs" / "rt" / "rt_s37_2k.yaml"
    cfg = copy.deepcopy(_load(S37))
    cfg["TRACK"] = dict(cfg.get("TRACK") or {}, ROLLOUT_P=0.5)
    cfg["DATA"]["TRIPLET"] = True
    with pytest.raises(ValueError, match="raw-event"):
        MNISTModel(cfg)


# ============================================================================================= (b) a real Lightning fit
def test_b_real_trainer_fit_ramps_p_with_the_optimiser_step(trip):
    """A real `Trainer.fit` on CPU, 4 optimiser steps, ramp [1, 3], P = 1: the p the model uses at the start of step s is
    0, 0, 0.5, 1 (global_step 0..3), and from step 2 on the valid rows of the batch are drawn at that probability."""
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader
    cfg, _ = trip
    cfg = copy.deepcopy(cfg)
    cfg["MODEL"]["CNN_BACKBONE"] = "resnet18_l3w128"                             # a small trunk keeps the test short
    ds = _train_ds(cfg)
    loader = DataLoader(ds, batch_size=6, shuffle=True, num_workers=0, drop_last=True,
                        generator=torch.Generator().manual_seed(0))
    m = _roll_model(cfg, p=1.0, ramp=(1, 3), CNN_BACKBONE="resnet18_l3w128")
    seen = []

    class Rec(pl.Callback):
        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            seen.append((int(trainer.global_step), dict(module._rollout_stats)))

    trainer = pl.Trainer(accelerator="cpu", devices=1, max_steps=4, logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, callbacks=[Rec()], num_sanity_val_steps=0, val_check_interval=100,
                         limit_val_batches=0, log_every_n_steps=1, enable_model_summary=False)
    trainer.fit(m, loader)
    assert [s[1]["p"] for s in seen] == pytest.approx([0.0, 0.0, 0.5, 1.0]), seen
    assert [s[1]["rows"] for s in seen][:2] == [0, 0]
    assert seen[3][1]["rows"] == seen[3][1]["valid"]                              # p = 1: every valid row
    cm = trainer.callback_metrics
    assert float(cm["train_rollout_p"]) == 1.0 and torch.isfinite(cm["train_loss"])


# ============================================================================================= (c) the l3w128 trunk
def test_c_w128_inside_the_model_has_the_expected_parameters_and_shapes():
    m = _build(MNISTModel, _cfg(W128_CFG))
    assert m.cnn_backbone == "resnet18_l3w128"
    n_trunk = sum(p.numel() for p in m.rn.parameters())
    n_all = sum(p.numel() for p in m.parameters())
    assert n_trunk == 1_297_139 and n_all == 1_303_893                           # + conv1 111 + prev_mlp 6,643
    ref = _build(MNISTModel, _cfg(L3_CFG))
    assert sum(p.numel() for p in ref.parameters()) == 2_802_645
    with torch.no_grad():
        out = _randomize(m).eval()(_lnes(3), _prev(3), betas=torch.zeros(3, 10), camera_K=BASE_K.expand(3, 3, 3))
    assert out.shape == (3, 51) and torch.isfinite(out).all()
    assert m.rn.fc.in_features == 128 and m.conv1.out_channels == 3


def test_c_default_names_are_unchanged_by_the_new_backbone(pre_e):
    """resnet18_l3 and the default resnet18: the model built by the edited code equals the pre-edit one bit for bit."""
    for path in (L3_CFG, DT / "dt_base.yaml"):
        a = _build(pre_e.MNISTModel, _cfg(path))
        rng_a = torch.get_rng_state()
        b = _build(MNISTModel, _cfg(path))
        assert torch.equal(rng_a, torch.get_rng_state())
        _assert_same_state(a, b)


def test_c_w128_supports_the_root_head_taps():
    """`resnet_forward_taps` and the DT2 root heads read the trunk's stages generically: the spatial head builds on the 128
    channel layer3 map."""
    m = _build(MNISTModel, _cfg(W128_CFG, model={"ROOT_HEAD": "spatial"}))
    with torch.no_grad():
        out = _randomize(m).eval()(_lnes(2), _prev(2))
    assert out.shape == (2, 51) and torch.isfinite(out).all()


# ============================================================================================= (d) {SEED} and the teacher
def _save_lightning_ckpt(m, path, hparams):
    import pytorch_lightning as pl
    ck = {"state_dict": m.state_dict(), "hyper_parameters": hparams, "pytorch-lightning_version": pl.__version__,
          "epoch": 0, "global_step": 0}
    m.on_save_checkpoint(ck)
    torch.save(ck, path)


@pytest.fixture(scope="module")
def l3_teacher(tmp_path_factory):
    """A dt_dz_l3-like teacher (resnet18_l3) saved the way Lightning saves it, hyper-parameters included (top-level
    sections of the config, as the real dt_dz_l3_s3407 checkpoint stores them)."""
    path = tmp_path_factory.mktemp("kd") / "dt_dz_l3_s3407" / "last.ckpt"
    path.parent.mkdir()
    cfg = _cfg(L3_CFG)
    t = _randomize(_build(MNISTModel, cfg, seed=99), seed=99).eval()
    _save_lightning_ckpt(t, path, {k: v for k, v in cfg.items()})
    return path, t


def test_d_seed_placeholder_is_replaced_and_written_back(train_mod, tmp_path):
    for seed in (3407, 3408):
        d = tmp_path / f"dt_dz_l3_s{seed}"
        d.mkdir()
        (d / "last.ckpt").write_bytes(b"x")
    cfg = {"SEED": 3408, "MODEL": {"DISTILL_CKPT": str(tmp_path / "dt_dz_l3_s{SEED}" / "last.ckpt")}}
    got = train_mod.resolve_distill_ckpt(cfg)
    assert got == str(tmp_path / "dt_dz_l3_s3408" / "last.ckpt") and cfg["MODEL"]["DISTILL_CKPT"] == got
    cfg = {"SEED": 3407, "MODEL": {"DISTILL_CKPT": str(tmp_path / "dt_dz_l3_s{SEED}" / "last.ckpt")}}
    assert train_mod.resolve_distill_ckpt(cfg).endswith("dt_dz_l3_s3407/last.ckpt")
    # every occurrence, no placeholder, no key, a seed without a teacher
    cfg = {"SEED": 3407, "MODEL": {"DISTILL_CKPT": str(tmp_path / "dt_dz_l3_s{SEED}" / "last.ckpt")}}
    assert train_mod.resolve_distill_ckpt(cfg) == str(tmp_path / "dt_dz_l3_s3407" / "last.ckpt")
    plain = str(tmp_path / "dt_dz_l3_s3407" / "last.ckpt")
    cfg = {"SEED": 99, "MODEL": {"DISTILL_CKPT": plain}}
    assert train_mod.resolve_distill_ckpt(cfg) == plain and cfg["MODEL"]["DISTILL_CKPT"] == plain
    assert train_mod.resolve_distill_ckpt({"SEED": 1, "MODEL": {}}) is None
    assert train_mod.resolve_distill_ckpt({"SEED": 1, "MODEL": {"DISTILL_CKPT": None}}) is None
    cfg = {"SEED": 5, "MODEL": {"DISTILL_CKPT": str(tmp_path / "dt_dz_l3_s{SEED}" / "last.ckpt")}}
    with pytest.raises(FileNotFoundError, match="dt_dz_l3_s5"):
        train_mod.resolve_distill_ckpt(cfg)
    assert "{SEED}" in cfg["MODEL"]["DISTILL_CKPT"]                              # untouched on failure


def test_d_the_kd_config_resolves_to_the_same_seed_teacher_on_the_real_files(train_mod):
    for seed in (3407, 3408):
        cfg = copy.deepcopy(_load(KD_CFG))
        cfg["SEED"] = seed
        want = REPO.parent / "EventHands1" / "outputs" / "semkine" / f"dt_dz_l3_s{seed}" / "last.ckpt"
        if not want.exists():
            pytest.skip(f"{want} not available")
        assert train_mod.resolve_distill_ckpt(cfg) == str(want)


def test_d_teacher_is_built_with_the_trunk_its_checkpoint_records(train_mod, l3_teacher):
    path, saved = l3_teacher
    cfg = _cfg(KD_CFG, model={"DISTILL_CKPT": str(path)})
    assert cfg["MODEL"]["CNN_BACKBONE"] == "resnet18_l3w128"
    assert train_mod.teacher_backbone(str(path)) == "resnet18_l3"
    # without the fix: teacher_config drops CNN_BACKBONE, the teacher is a default resnet18 and refuses its own weights
    with pytest.raises(RuntimeError, match="size mismatch|Missing key|Unexpected key"):
        MNISTModel.load_from_checkpoint(str(path), cfg=train_mod.teacher_config(cfg), map_location="cpu")
    t = train_mod.load_distill_teacher(str(path), cfg)
    assert t.cnn_backbone == "resnet18_l3" and sum(p.numel() for p in t.parameters()) == 2_802_645
    assert not t.training and all(not p.requires_grad for p in t.parameters())
    x, prev = _lnes(2, seed=2), _prev(2, seed=3)
    with torch.no_grad():
        assert torch.equal(t(x, prev), saved(x, prev))
    # the student's own config is not touched, and the teacher keeps the student's non-backbone architecture keys
    assert cfg["MODEL"]["CNN_BACKBONE"] == "resnet18_l3w128"


def test_d_teacher_without_a_recorded_trunk_is_still_the_default_resnet18(train_mod, tmp_path):
    """Hyper-parameters without CNN_BACKBONE (rt_cnntrack-style), empty hyper-parameters, the `cfg` layout, none at all."""
    t0 = _randomize(_build(MNISTModel, _cfg(DT / "dt_base.yaml"), seed=99), seed=99).eval()
    for i, hp in enumerate([{}, {"MODEL": {"POSE_REPR": "mano_full_axis_angle"}}, {"cfg": {"MODEL": {}}}]):
        path = tmp_path / f"t{i}.ckpt"
        _save_lightning_ckpt(t0, path, hp)
        assert train_mod.teacher_backbone(str(path)) is None
        t = train_mod.load_distill_teacher(str(path), _cfg(KD_CFG, model={"DISTILL_CKPT": str(path)}))
        assert t.cnn_backbone == "resnet18" and sum(p.numel() for p in t.parameters()) == 11_209_429
    path = tmp_path / "cfgkey.ckpt"
    _save_lightning_ckpt(t0, path, {"cfg": {"MODEL": {"CNN_BACKBONE": "resnet18_l3"}}})
    assert train_mod.teacher_backbone(str(path)) == "resnet18_l3"


def _kd_student(train_mod, l3_teacher, seed=1234, **model):
    path, _ = l3_teacher
    cfg = _cfg(KD_CFG, model={"DISTILL_CKPT": str(path), **model})
    m = _randomize(_build(MNISTModel, cfg, seed))                                # before the teacher is attached
    m.teacher = train_mod.load_distill_teacher(str(path), cfg)
    return m.train()


@pytest.fixture(scope="module")
def plain_batch(trip):
    """The six valid rows' w2 windows as a plain legacy tuple, as the dense dt path takes them."""
    _, batch = trip
    return [t[:6].clone() for t in batch["w2"]]


def test_d_dense_distill_term_is_finite_nonzero_and_the_objective_between_student_and_teacher(train_mod, l3_teacher, plain_batch):
    w = 1.0
    m = _kd_student(train_mod, l3_teacher)
    ref = _build(MNISTModel, _cfg(KD_CFG, model={"DISTILL_WEIGHT": 0.0, "DISTILL_CKPT": str(l3_teacher[0])}), 1234)
    ref = _randomize(ref).train()
    ref.load_state_dict({k: v for k, v in m.state_dict().items() if not k.startswith("teacher.")})
    _, lg, _ = _ts(m, [t.clone() for t in plain_batch])
    _, lr_, _ = _ts(ref, [t.clone() for t in plain_batch])
    d = float(lg["train_loss_distill"])
    assert np.isfinite(d) and d > 1e-4 and "train_loss_distill" not in lr_
    assert abs(float(lg["train_loss"]) - (float(lr_["train_loss"]) + w * d)) < 1e-6
    # by hand: the teacher's eval-mode output on the same input, the main objective applied to (student, teacher)
    h = _kd_student(train_mod, l3_teacher)
    x, prev, _, betas, K = plain_batch
    with torch.no_grad():
        t_out = h.teacher.eval()(x, prev, betas=betas, camera_K=K)
        pred, _, _, _ = h._predict_batch([t.clone() for t in plain_batch])
        want = h._compute_loss(pred, t_out, betas)[0]
    assert abs(float(want) - d) < 1e-5
    assert m.teacher.training is False and all(not p.requires_grad for p in m.teacher.parameters())


def test_d_distillation_gradients_reach_only_the_student_and_move_it_toward_the_teacher(train_mod, l3_teacher, plain_batch):
    m = _kd_student(train_mod, l3_teacher)
    t_before = {k: v.clone() for k, v in m.teacher.state_dict().items()}
    _, _, grads = _ts(m, [t.clone() for t in plain_batch])
    assert all(g is None for n, g in grads.items() if n.startswith("teacher."))
    sg = [g for n, g in grads.items() if not n.startswith("teacher.")]
    assert sg and all(g is not None and torch.isfinite(g).all() for g in sg)
    assert all(torch.equal(t_before[k], v) for k, v in m.teacher.state_dict().items())
    # a few Adam steps on the distillation term alone reduce it
    m2 = _kd_student(train_mod, l3_teacher)
    opt = torch.optim.Adam([p for p in m2.parameters() if p.requires_grad], lr=1e-3)
    terms = []
    for _ in range(6):
        opt.zero_grad()
        pred, y, betas, _ = m2._predict_batch([t.clone() for t in plain_batch])
        _, parts = m2._maybe_distill(pred, plain_batch, torch.zeros(()), {})
        parts["loss_distill"].backward()
        opt.step()
        terms.append(float(parts["loss_distill"]))
    assert terms[-1] < terms[0], terms


def test_d_distillation_reads_the_replaced_prev_of_a_rollout_batch(train_mod, l3_teacher, trip):
    """Rollout and distillation together: `_maybe_distill` takes window 2 from the batch dict, so the teacher sees the same
    (replaced) prev as the student."""
    cfg, batch0 = trip
    path, _ = l3_teacher
    kcfg = copy.deepcopy(cfg)
    kcfg["MODEL"].update(DISTILL_WEIGHT=1.0, DISTILL_CKPT=str(path))
    m = _randomize(_build(MNISTModel, kcfg)).train()
    m.teacher = train_mod.load_distill_teacher(str(path), kcfg)
    m.rollout_p, m.rollout_ramp = 1.0, (0, 1)
    seen = []
    orig_t = m.teacher.forward
    m.teacher.forward = lambda x, prev, betas=None, camera_K=None: (seen.append(prev.clone()),
                                                                    orig_t(x, prev, betas=betas, camera_K=camera_K))[1]
    batch = _clone_batch(batch0)
    logged = _capture_logs(m)
    with _step_patch(5):
        m.training_step(batch, 0)
    assert len(seen) == 1 and torch.equal(seen[0], batch["w2"][1])
    assert not torch.equal(seen[0], batch0["w2"][1]) and np.isfinite(float(logged["train_loss_distill"]))
