"""Task `model-core` of the DT round: four default-off features of the dense tracker `MNISTModel`
(docs/DT_RENDER_TRACK_PREREG.md sections 2 and 3):

  MODEL.CNN_BACKBONE     the trunk `rn` by name (model/backbones.py); default `resnet18` = torchvision's resnet18
  MODEL.CAM_PLANES       two input channels, the camera-ray slopes of every render pixel under the sample's own K
  MODEL.ROOT_COMPOSE     `so3`: root rotation R = Exp(delta_root) R_prev (left), returned as principal axis-angle
  MODEL.PREV_MLP_TRANSL  false: prev_mlp's translation outputs are masked to 0

Pinned here (the letters follow the task text):

  (a) every default configuration is the ORIGINAL model bit for bit: modules, state_dict keys / shapes / values under
      the same seed, the random draws consumed while building (checked for every distinct MODEL block in configs/),
      eval forward outputs (B in {1, 5}, fp32 and bf16 autocast on CPU), and the loss, its parts and every gradient of
      a train-mode step on a real batch of the dataset. The original is outputs/dt/ref/model_orig.py, the file as it
      was before the edit (sha256 pinned below), imported under another module name.
  (b) ROOT_COMPOSE so3: delta 0 returns the previous root, an event-free packet returns prevpos bit for bit, a 10 deg
      rotation of prev about the camera z axis moves the output by 10 deg, the composition equals scipy's
      Rotation.from_rotvec(delta) * Rotation.from_rotvec(prev) on 1000 random pairs (|prev| up to pi, tiny deltas),
      gradients are finite and correct, the root norm never exceeds pi, translation and fingers are the additive ones.
  (c) CAM_PLANES: conv1 has two more input channels appended last, the planes are the hand formula, K changes the output
      and the same K leaves it bitwise unchanged, and the planes are the ONLY K dependence when the render is empty.
  (d) PREV_MLP_TRANSL false: the output equals the default model whose prev_mlp translation rows were zeroed by hand,
      rotation / fingers are the default model's, state_dicts are identical, the masked outputs get zero gradient.
  (e) CNN_BACKBONE: every name of backbones.BACKBONES builds inside MNISTModel, gives (B, 51) and finite gradients;
      `resnet18` is the default bit for bit; unknown names and illegal combinations raise ValueError.
  (f) unknown MODEL keys still raise; the four new keys are accepted; impossible combinations raise.

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python -m pytest tests/test_dt_model.py -q
"""
from __future__ import annotations

import copy
import functools
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as Rot
from torch.utils.data.dataloader import default_collate

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "model")]
import backbones as BB                                                       # noqa: E402
from config import load_config                                               # noqa: E402
from model import MNISTModel                                                 # noqa: E402

ORIG = REPO / "outputs" / "dt" / "ref" / "model_orig.py"
#: sha256 of model/model.py as found in MAIN and STAGE before the edit (2026-10-03)
ORIG_SHA256 = "0521bd0ae87839a3529cf7a3b3ee05ad20b47a63d6595008fc3e4ac6fd70a7c2"
RT_CFG = REPO / "configs" / "rt" / "rt_cnntrack.yaml"
DT = REPO / "configs" / "dt"
DATA_ROOT = REPO / "data" / "hand_data51"
#: MODEL keys the pristine reference (outputs/dt/ref/model_orig.py) does not know. The DT2 round added the last two
#: (ROOT_HEAD, RENDER_FP32; tests/test_dt_model3.py pins them): configs that carry any of them are not "existing blocks".
NEW_KEYS = ("CNN_BACKBONE", "CAM_PLANES", "ROOT_COMPOSE", "PREV_MLP_TRANSL", "ROOT_HEAD", "RENDER_FP32")
#: TRACK keys the pristine reference does not know (DT2 package E, tests/test_dt_model4.py pins them); configs that carry
#: any of them are not "existing blocks" either (the reference refuses them as unknown TRACK keys)
NEW_TRACK_KEYS = ("ROLLOUT_P", "ROLLOUT_RAMP")
H, W = 180, 240
BASE_K = torch.tensor([[603.4507, 0.0, 325.09183], [0.0, 602.95654, 242.09796], [0.0, 0.0, 1.0]])


# ----------------------------------------------------------------------------------------------------- helpers
@functools.lru_cache(maxsize=None)
def _load(path):
    return load_config(path)


def _cfg(path, **model):
    cfg = copy.deepcopy(_load(Path(path)))
    cfg["MODEL"].update(model)
    return cfg


def _mini_cfg(**model):
    """A small dense config without render (test_s10_active_head's)."""
    return {
        "MODEL": {"POSE_REPR": "mano_full_axis_angle", "OUTPUT_DIM": 51, "PREDICT_DELTA": True, **model},
        "MANO": {"NPZ": str(REPO / "assets/mano_right.npz")},
        "LOSS": {"TYPE": "mse_51d", "LAMBDA_POSE": 60.0, "LAMBDA_T": 30000.0, "LAMBDA_R": 60.0,
                 "NORMALIZER": 12, "LOG10": True},
        "DATA": {"HEIGHT": H, "WIDTH": W},
    }


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


def _lnes(B, seed=0, empty=()):
    """A sparse random LNES (B, H, W, 2); rows in `empty` carry no event at all."""
    g = torch.Generator().manual_seed(seed)
    x = (torch.rand(B, H, W, 2, generator=g) < 0.12).float() * torch.rand(B, H, W, 2, generator=g)
    for i in empty:
        x[i] = 0.0
    return x


def _prev(B, seed=0, norm=2.3):
    """A realistic previous state: translation near (0, 0, 0.55), root axis-angle of norm `norm`."""
    g = torch.Generator().manual_seed(seed)
    p = torch.zeros(B, 51)
    p[:, :2] = 0.05 * torch.randn(B, 2, generator=g)
    p[:, 2] = 0.55 + 0.03 * torch.randn(B, generator=g)
    axis = torch.randn(B, 3, generator=g)
    p[:, 3:6] = axis / axis.norm(dim=-1, keepdim=True) * norm
    p[:, 6:] = 0.3 * torch.randn(B, 45, generator=g)
    return p


def _K(fx, fy, cx, cy):
    return torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def _fwd(m, x, p, bf16=False, **kw):
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        return m(x, p, **kw)


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


def _geo(a, b):
    """Geodesic angle in rad between the rotations of two rotation-vector arrays (N, 3), float64."""
    return (Rot.from_rotvec(np.asarray(a, np.float64)) * Rot.from_rotvec(np.asarray(b, np.float64)).inv()).magnitude()


def _rand_rotvec(n, lo, hi, rng):
    axis = rng.standard_normal((n, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    return axis * rng.uniform(lo, hi, size=(n, 1))


def _conv1_input(m, x, p, **kw):
    """The tensor conv1 receives (NCHW) in an eval forward."""
    got = []
    h = m.conv1.register_forward_hook(lambda mod, inp, out: got.append(inp[0].detach().clone()))
    try:
        _fwd(m, x, p, **kw)
    finally:
        h.remove()
    return got[0]


def _step(model, batch, bf16):
    """One training_step-style loss computation (without the logging): returns prediction, the loss as
    training_step returns it (log10), the loss parts and every parameter gradient."""
    model.zero_grad(set_to_none=True)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        pred, y, betas, _ = model._predict_batch(batch)
        loss, parts = model._compute_loss(pred, y, betas)
    out = loss.float().log10() if model.log10_loss else loss.float()
    out.backward()
    grads = {n: (None if p.grad is None else p.grad.clone()) for n, p in model.named_parameters()}
    return pred.detach(), out.detach(), {k: v.detach() for k, v in parts.items()}, grads


# ---------------------------------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def orig():
    """The pre-edit model.py, imported under another module name (same sys.path as the live import)."""
    if not ORIG.exists():
        pytest.skip(f"{ORIG} not available")
    assert hashlib.sha256(ORIG.read_bytes()).hexdigest() == ORIG_SHA256, "the reference copy was modified"
    spec = importlib.util.spec_from_file_location("model_orig", ORIG)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["model_orig"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def real_batch():
    """Six training samples (DomRand on, so K is the augmented K') as the dataset returns them."""
    if not DATA_ROOT.exists():
        pytest.skip("hand_data51 not available")
    from semkine.dataset import build_dataset
    cfg = _load(DT / "dt_base.yaml")
    comps = np.load(cfg["MANO"]["NPZ"])["hands_components"].astype(np.float32)
    ds = build_dataset(cfg, "train", comps, train=True)
    return default_collate([ds[i] for i in (0, 9000, 21000, 36000, 52000, 70000)])


@pytest.fixture(scope="module")
def so3_model():
    """dt_so3c with a non-zero prev_mlp and BN statistics, eval mode (shared, never mutated by the tests)."""
    return _randomize(_build(MNISTModel, _cfg(DT / "dt_so3c.yaml"))).eval()


# =========================================================================================== (a) backward compatibility
@pytest.mark.parametrize("path", [RT_CFG, DT / "dt_base.yaml"], ids=["rt_cnntrack", "dt_base"])
@pytest.mark.parametrize("explicit", [{}, {"CNN_BACKBONE": "resnet18"}, {"CAM_PLANES": False},
                                      {"ROOT_COMPOSE": "add"}, {"PREV_MLP_TRANSL": True},
                                      {"ROOT_HEAD": "none"}, {"RENDER_FP32": False}], ids=str)
def test_default_state_dict_modules_and_rng_equal_the_original(orig, path, explicit):
    """Same keys, shapes, values, module tree and the same number of random draws, also with every new key written
    out at its default value."""
    a = _build(orig.MNISTModel, _cfg(path))
    rng_a = torch.get_rng_state()
    b = _build(MNISTModel, _cfg(path, **explicit))
    assert torch.equal(rng_a, torch.get_rng_state())
    _assert_same_state(a, b)
    assert (b.cnn_backbone, b.cam_planes, b.root_compose, b.prev_mlp_transl) == ("resnet18", False, "add", True)


def _model_blocks():
    """One config per distinct (MODEL, TRACK, LOSS, DATA, MANO) block of every existing config file."""
    seen, out = set(), []
    for p in sorted((REPO / "configs").rglob("*.yaml")):
        try:
            cfg = load_config(p)
        except Exception:                                         # not a model config
            continue
        if "MODEL" not in cfg or any(k in cfg["MODEL"] for k in NEW_KEYS) \
                or any(k in (cfg.get("TRACK") or {}) for k in NEW_TRACK_KEYS):
            continue
        key = json.dumps({k: cfg.get(k) for k in ("MODEL", "TRACK", "LOSS", "DATA", "MANO")}, sort_keys=True,
                         default=str)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def test_every_existing_model_block_builds_identically(orig):
    """The edit touches `__init__` before the trunk is chosen, so the claim is checked for every distinct MODEL block
    that exists (dense, absolute, active-head, S37 / S38 / U1a encoders ...): same state_dict, same random draws; a
    block the original refuses is refused by the edited model with the same exception type."""
    paths = _model_blocks()
    assert len(paths) >= 20, len(paths)
    built = 0
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
    assert built >= 20, built


@pytest.mark.parametrize("path", [RT_CFG, DT / "dt_base.yaml"], ids=["rt_cnntrack", "dt_base"])
@pytest.mark.parametrize("explicit", [{}, {"CNN_BACKBONE": "resnet18", "CAM_PLANES": False, "ROOT_COMPOSE": "add",
                                           "PREV_MLP_TRANSL": True, "ROOT_HEAD": "none", "RENDER_FP32": False}],
                         ids=["plain", "defaults-written-out"])
def test_default_forward_equals_the_original_bitwise(orig, path, explicit):
    a = _randomize(_build(orig.MNISTModel, _cfg(path))).eval()
    b = _randomize(_build(MNISTModel, _cfg(path, **explicit))).eval()
    x, prev = _lnes(5, empty=(2,)), _prev(5)
    for B in (1, 5):
        for bf16 in (False, True):
            oa, ob = _fwd(a, x[:B], prev[:B], bf16), _fwd(b, x[:B], prev[:B], bf16)
            assert oa.dtype == ob.dtype and torch.equal(oa, ob), (B, bf16)
    # an explicit betas / K, as the training step passes them
    betas = 0.3 * torch.randn(5, 10, generator=torch.Generator().manual_seed(3))
    K = BASE_K.expand(5, 3, 3) * torch.tensor([1.1, 0.9, 1.0]).view(1, 1, 3)
    assert torch.equal(_fwd(a, x, prev, False, betas=betas, camera_K=K), _fwd(b, x, prev, False, betas=betas, camera_K=K))
    # the context path of the closed loop
    a.set_hand_context(betas[0], BASE_K * 1.05)
    b.set_hand_context(betas[0], BASE_K * 1.05)
    assert torch.equal(_fwd(a, x[:1], prev[:1]), _fwd(b, x[:1], prev[:1]))
    # an event-free packet: the original and the edited model agree on what they return
    assert torch.equal(_fwd(a, x[2:3] * 0, prev[2:3]), _fwd(b, x[2:3] * 0, prev[2:3]))


@pytest.mark.parametrize("path", [RT_CFG, DT / "dt_base.yaml"], ids=["rt_cnntrack (mse_51d)", "dt_base (so3_trans_fk)"])
@pytest.mark.parametrize("bf16", [False, True], ids=["fp32", "bf16"])
def test_default_loss_and_gradients_equal_the_original_on_a_real_batch(orig, real_batch, path, bf16):
    a = _randomize(_build(orig.MNISTModel, _cfg(path))).train()
    b = _randomize(_build(MNISTModel, _cfg(path))).train()
    ra = _step(a, copy.deepcopy(real_batch), bf16)
    # the reference must be reproducible with itself, otherwise "equal" would mean nothing
    a2 = _randomize(_build(orig.MNISTModel, _cfg(path))).train()
    rb = _step(b, copy.deepcopy(real_batch), bf16)
    rr = _step(a2, copy.deepcopy(real_batch), bf16)
    for ref, got in ((ra, rr), (ra, rb)):
        assert ref[0].dtype == got[0].dtype and torch.equal(ref[0], got[0])
        assert torch.equal(ref[1], got[1])
        assert list(ref[2]) == list(got[2]) and all(torch.equal(ref[2][k], got[2][k]) for k in ref[2])
        assert list(ref[3]) == list(got[3])
        for n in ref[3]:
            assert (ref[3][n] is None) == (got[3][n] is None), n
            if ref[3][n] is not None:
                assert torch.equal(ref[3][n], got[3][n]), n
    # and the BatchNorm running statistics the step wrote
    _assert_same_state(a, b)


def test_pristine_reference_is_the_file_before_the_edit(orig):
    """The reference's MODEL_KEYS does not know the new keys, the edited one does."""
    assert not set(NEW_KEYS) & set(orig.MNISTModel.MODEL_KEYS)
    assert set(NEW_KEYS) <= set(MNISTModel.MODEL_KEYS)
    assert set(orig.MNISTModel.MODEL_KEYS) <= set(MNISTModel.MODEL_KEYS)
    assert set(orig.MNISTModel.TRACK_KEYS) == set(MNISTModel.TRACK_KEYS) - set(NEW_TRACK_KEYS)
    assert set(NEW_TRACK_KEYS) <= set(MNISTModel.TRACK_KEYS) and not set(NEW_TRACK_KEYS) & set(orig.MNISTModel.TRACK_KEYS)


# ================================================================================================== (b) ROOT_COMPOSE so3
def test_so3_config_state_dict_equals_the_add_model(so3_model):
    """ROOT_COMPOSE adds no parameter: same keys / shapes / values as the model without it."""
    _assert_same_state(_randomize(_build(MNISTModel, _cfg(DT / "dt_base.yaml"))), so3_model)
    assert so3_model.root_compose == "so3"


def _zero_delta_model(**model):
    """so3 model whose network output is exactly 0: fc zeroed, prev_mlp at its zero init (and no BN statistics)."""
    m = _build(MNISTModel, _cfg(DT / "dt_so3c.yaml", **model)).eval()
    with torch.no_grad():
        m.rn.fc.weight.zero_()
        m.rn.fc.bias.zero_()
    return m


@pytest.mark.parametrize("norm", [0.05, 1.0, 2.3, 3.0, 3.1, 3.14159])
def test_so3_zero_delta_returns_the_previous_root(norm):
    m = _zero_delta_model()
    x, prev = _lnes(4, seed=1), _prev(4, seed=2, norm=norm)
    for bf16 in (False, True):
        out = _fwd(m, x, prev, bf16)
        assert out.dtype == torch.float32
        assert _geo(out[:, 3:6], prev[:, 3:6]).max() < 1e-6
        assert (out[:, 3:6] - prev[:, 3:6]).abs().max() < 1e-6              # |prev| < pi: the same vector
        ref = prev.bfloat16().float() if bf16 else prev                      # the additive columns round in bf16
        assert torch.equal(out[:, :3], ref[:, :3]) and torch.equal(out[:, 6:], ref[:, 6:])


def test_so3_event_free_packet_returns_prevpos_bit_for_bit(so3_model):
    x, prev = _lnes(5, seed=3, empty=(1, 3)), _prev(5, seed=4, norm=3.1)
    for bf16 in (False, True):
        out = _fwd(so3_model, x, prev, bf16)
        for i in (1, 3):
            assert torch.equal(out[i], prev[i]), (i, bf16)
        for i in (0, 2, 4):
            assert not torch.equal(out[i], prev[i])
    # train mode (BatchNorm on batch statistics), where the training loss sees it
    m = copy.deepcopy(so3_model).train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = m(x, prev)
    assert torch.equal(out[1], prev[1]) and torch.equal(out[3], prev[3])
    # a prev whose root norm lies above pi (noise on the state can do that) is held exactly too, not wrapped
    over = prev.clone()
    over[:, 3:6] *= 3.3 / over[:, 3:6].norm(dim=-1, keepdim=True)
    assert torch.equal(_fwd(so3_model, x, over)[1], over[1])


def test_so3_rotating_prev_about_camera_z_rotates_the_output_by_that_angle():
    m = _zero_delta_model()
    x = _lnes(4, seed=5)
    prev = _prev(4, seed=6, norm=2.3)
    rz = Rot.from_rotvec([0.0, 0.0, np.deg2rad(10.0)])
    prev2 = prev.clone()
    prev2[:, 3:6] = torch.from_numpy((rz * Rot.from_rotvec(prev[:, 3:6].double().numpy())).as_rotvec()).float()
    o1, o2 = _fwd(m, x, prev)[:, 3:6].double().numpy(), _fwd(m, x, prev2)[:, 3:6].double().numpy()
    rel = Rot.from_rotvec(o2) * Rot.from_rotvec(o1).inv()                    # left multiplication: o2 = Rz o1
    assert np.abs(rel.magnitude() - np.deg2rad(10.0)).max() < 1e-5
    assert np.abs(rel.as_rotvec() - np.array([0.0, 0.0, np.deg2rad(10.0)])).max() < 1e-5
    wrong = Rot.from_rotvec(o1).inv() * Rot.from_rotvec(o2)                  # a right multiplication would read this
    assert np.abs(wrong.as_rotvec() - np.array([0.0, 0.0, np.deg2rad(10.0)])).max() > 1e-2


def _pairs(n=1000, seed=0):
    """(prev, delta) rotation vectors: 400 prev with |prev| in [3.0, 3.14159], 600 in [0, 3]; 100 deltas exactly
    0, 300 tiny (1e-9 .. 1e-4), 600 in [1e-3, 1]."""
    rng = np.random.default_rng(seed)
    prev = np.concatenate([_rand_rotvec(400, 3.0, 3.14159, rng), _rand_rotvec(600, 0.0, 3.0, rng)])
    mag = np.concatenate([np.zeros(100), 10 ** rng.uniform(-9, -4, 300), rng.uniform(1e-3, 1.0, 600)])
    axis = rng.standard_normal((n, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    perm = rng.permutation(n)
    return prev[perm], (axis * mag[:, None])[perm]


def test_so3_composition_equals_scipy_on_1000_random_pairs(so3_model):
    prev, delta = _pairs()
    g = torch.Generator().manual_seed(11)
    d51, p51 = 0.1 * torch.randn(1000, 51, generator=g), 0.1 * torch.randn(1000, 51, generator=g)
    d51[:, 3:6] = torch.from_numpy(delta).float()
    p51[:, 3:6] = torch.from_numpy(prev).float()
    added = d51 + p51
    out = so3_model._compose_root_so3(added, d51, p51)
    assert out.dtype == torch.float32 and out.shape == (1000, 51)
    pd, dd = p51[:, 3:6].double().numpy(), d51[:, 3:6].double().numpy()      # the float32 values the model saw
    want = (Rot.from_rotvec(dd) * Rot.from_rotvec(pd)).as_rotvec()
    got = out[:, 3:6].double().numpy()
    assert _geo(got, want).max() < 1e-5
    safe = np.linalg.norm(want, axis=1) < np.pi - 1e-3                       # away from the seam the vectors agree
    assert safe.sum() > 800 and np.abs(got - want)[safe].max() < 1e-5
    assert np.linalg.norm(got, axis=1).max() <= np.pi + 1e-5
    # the other columns are the additive update, untouched
    assert torch.equal(out[:, :3], added[:, :3]) and torch.equal(out[:, 6:], added[:, 6:])
    # delta = 0 (the first 100 sorted by magnitude): the previous rotation up to float error
    zero = np.linalg.norm(dd, axis=1) == 0
    assert zero.sum() == 100 and _geo(got[zero], pd[zero]).max() < 1e-6


def test_so3_output_norm_never_exceeds_pi_with_large_random_deltas(so3_model):
    rng = np.random.default_rng(5)
    prev = _rand_rotvec(2000, 0.0, np.pi, rng)
    delta = _rand_rotvec(2000, 0.0, 4.0, rng)                                # up to > 2 rad steps: wraps round pi often
    d51, p51 = torch.zeros(2000, 51), torch.zeros(2000, 51)
    d51[:, 3:6], p51[:, 3:6] = torch.from_numpy(delta).float(), torch.from_numpy(prev).float()
    out = so3_model._compose_root_so3(d51 + p51, d51, p51)[:, 3:6]
    assert torch.isfinite(out).all() and out.norm(dim=-1).max() <= np.pi + 1e-5
    want = (Rot.from_rotvec(d51[:, 3:6].double().numpy()) * Rot.from_rotvec(p51[:, 3:6].double().numpy())).as_rotvec()
    assert _geo(out.double().numpy(), want).max() < 1e-5
    # the whole model: a random network and a prev at the seam
    x, prev_s = _lnes(4, seed=8), _prev(4, seed=9, norm=3.14159)
    for bf16 in (False, True):
        assert _fwd(so3_model, x, prev_s, bf16)[:, 3:6].norm(dim=-1).max() <= np.pi + 1e-5


def test_so3_gradient_is_the_gradient_of_the_composition(so3_model):
    """Autograd through so3_exp / so3_log against a float64 finite difference of scipy's composition, away from the seam."""
    rng = np.random.default_rng(3)
    prev = _rand_rotvec(20, 0.5, 2.6, rng)
    delta = _rand_rotvec(20, 0.01, 0.4, rng)
    v = rng.standard_normal((20, 3))
    w = rng.standard_normal((20, 3))
    d51, p51 = torch.zeros(20, 51), torch.zeros(20, 51)
    d51[:, 3:6], p51[:, 3:6] = torch.from_numpy(delta).float(), torch.from_numpy(prev).float()
    d51.requires_grad_(True)
    out = so3_model._compose_root_so3(d51 + p51, d51, p51)
    (out[:, 3:6] * torch.from_numpy(w).float()).sum().backward()
    auto = (d51.grad[:, 3:6].double().numpy() * v).sum(axis=1)               # directional derivative along v, per row
    h = 1e-6
    f = lambda d: (Rot.from_rotvec(d) * Rot.from_rotvec(prev.astype(np.float32).astype(np.float64))).as_rotvec()  # noqa: E731
    d0 = delta.astype(np.float32).astype(np.float64)
    fd = ((f(d0 + h * v) - f(d0 - h * v)) / (2 * h) * w).sum(axis=1)
    assert np.isfinite(auto).all() and np.abs(auto - fd).max() < 2e-3 * (1 + np.abs(fd).max())
    # no gradient reaches prev's translation / finger columns through the root block
    assert d51.grad[:, :3].abs().max() == 0 and d51.grad[:, 6:].abs().max() == 0


def test_so3_gradients_are_finite_for_every_parameter_in_train_mode_bf16(real_batch):
    m = _build(MNISTModel, _cfg(DT / "dt_so3c.yaml")).train()
    _randomize(m)
    pred, loss, parts, grads = _step(m, copy.deepcopy(real_batch), bf16=True)
    assert pred.dtype == torch.float32 and torch.isfinite(loss) and all(torch.isfinite(v) for v in parts.values())
    assert all(g is not None and torch.isfinite(g).all() for g in grads.values()), \
        [n for n, g in grads.items() if g is None or not torch.isfinite(g).all()]
    assert grads["conv1.weight"].abs().sum() > 0 and grads["rn.fc.weight"].abs().sum() > 0
    assert grads["prev_mlp.2.weight"].abs().sum() > 0
    # the rotation loss at the target is zero (up to the float rounding of q . q), root included
    y = real_batch[2]
    assert float(m._compute_loss(y.clone(), y, real_batch[3])[1]["loss_rot"]) < 1e-6


def test_so3_train_and_eval_paths_are_the_same_code(so3_model):
    """There is one composition: with BatchNorm held at its running statistics, a train-mode forward equals the eval one
    bit for bit (the gate "training and evaluation paths agree" of the pre-registration)."""
    m = copy.deepcopy(so3_model)
    x, prev = _lnes(4, seed=16, empty=(2,)), _prev(4, seed=17, norm=3.0)
    ev = _fwd(m.eval(), x, prev)
    m.train()
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.eval()
    with torch.no_grad():
        tr = m(x, prev)
    assert torch.equal(ev, tr)


def test_so3_changes_the_root_columns_only(so3_model):
    """Same weights, `add` vs `so3`: translation and finger columns are bit for bit the additive ones, in fp32 and bf16."""
    add = _randomize(_build(MNISTModel, _cfg(DT / "dt_base.yaml"))).eval()
    x, prev = _lnes(4, seed=12, empty=(3,)), _prev(4, seed=13)
    for bf16 in (False, True):
        oa, os_ = _fwd(add, x, prev, bf16), _fwd(so3_model, x, prev, bf16)
        cols = [0, 1, 2] + list(range(6, 51))
        assert torch.equal(os_[:3, cols], oa[:3, cols].float())
        assert not torch.equal(os_[:3, 3:6], oa[:3, 3:6].float())
        assert torch.equal(os_[3], prev[3])                                  # the empty row is held in so3 mode


def test_so3_and_pmt_work_with_the_dense_active_head():
    """The dense active head (S10) shares `forward`; ROOT_COMPOSE / PREV_MLP_TRANSL apply to its 51-D delta too."""
    add = _build(MNISTModel, _mini_cfg(ACTIVE_HEAD=True)).eval()
    so3 = _build(MNISTModel, _mini_cfg(ACTIVE_HEAD=True, ROOT_COMPOSE="so3")).eval()
    x, p = torch.randn(2, H, W, 2), _prev(2, seed=18)
    oa, os_ = _fwd(add, x, p), _fwd(so3, x, p)
    cols = [0, 1, 2] + list(range(6, 51))
    assert torch.isfinite(os_).all() and torch.equal(os_[:, cols], oa[:, cols]) and not torch.equal(os_[:, 3:6], oa[:, 3:6])
    pmt = _randomize(_build(MNISTModel, _mini_cfg(ACTIVE_HEAD=True, PREVPOS_EMBED=True, PREV_MLP_TRANSL=False))).eval()
    ref = _randomize(_build(MNISTModel, _mini_cfg(ACTIVE_HEAD=True, PREVPOS_EMBED=True))).eval()
    op, orf = _fwd(pmt, x, p), _fwd(ref, x, p)
    assert torch.equal(op[:, 3:], orf[:, 3:]) and not torch.equal(op[:, :3], orf[:, :3])


def test_so3_works_without_the_event_gate_and_with_the_12d_layout():
    # no gate: nothing holds an event-free packet, the composition still runs
    m = _randomize(_build(MNISTModel, _cfg(DT / "dt_so3c.yaml", ZERO_EVENT_GATE=False))).eval()
    x, prev = _lnes(2, seed=14, empty=(1,)), _prev(2, seed=15)
    out = _fwd(m, x, prev)
    assert torch.isfinite(out).all() and not torch.equal(out[1], prev[1])
    # the PCA layout [alpha6, t3, R3]: the root sits in columns 9:12
    m12 = _build(MNISTModel, _mini_cfg(POSE_REPR="mano_pca6", OUTPUT_DIM=12, ROOT_COMPOSE="so3")).eval()
    m12b = _build(MNISTModel, _mini_cfg(POSE_REPR="mano_pca6", OUTPUT_DIM=12)).eval()
    x2 = torch.randn(3, H, W, 2)
    p2 = torch.randn(3, 12) * 0.2
    o, ob = _fwd(m12, x2, p2), _fwd(m12b, x2, p2)
    assert o.shape == (3, 12) and torch.equal(o[:, :9], ob[:, :9])
    delta = ob[:, 9:12] - p2[:, 9:12]
    want = (Rot.from_rotvec(delta.double().numpy()) * Rot.from_rotvec(p2[:, 9:12].double().numpy())).as_rotvec()
    assert _geo(o[:, 9:12].double().numpy(), want).max() < 1e-5


# ====================================================================================================== (c) CAM_PLANES
@pytest.fixture(scope="module")
def cam_model():
    return _randomize(_build(MNISTModel, _cfg(DT / "dt_cam.yaml"))).eval()


def test_cam_conv1_has_two_more_input_channels_appended_last():
    base = _build(MNISTModel, _cfg(DT / "dt_base.yaml"))
    cam = _build(MNISTModel, _cfg(DT / "dt_cam.yaml"))
    assert base.conv1.in_channels == 4 and cam.conv1.in_channels == 6
    assert tuple(base.conv1.weight.shape) == (3, 4, 3, 3) and tuple(cam.conv1.weight.shape) == (3, 6, 3, 3)
    assert sum(p.numel() for p in cam.parameters()) - sum(p.numel() for p in base.parameters()) == 3 * 2 * 9
    assert list(cam.state_dict()) == list(base.state_dict())


def test_cam_initial_weights_equal_the_model_without_planes_except_the_new_slices():
    """What is and is not the case under the same seed: conv1's first four input channels, its bias, the trunk and
    prev_mlp are the SAME tensors (no random draw is spent on the planes, the new slices start at zero), the two new
    input slices are zero, and so the models compute the same function at initialisation."""
    base = _build(MNISTModel, _cfg(DT / "dt_base.yaml"))
    rng_base = torch.get_rng_state()
    cam = _build(MNISTModel, _cfg(DT / "dt_cam.yaml"))
    assert torch.equal(rng_base, torch.get_rng_state())                      # the same number of draws
    sb, sc = base.state_dict(), cam.state_dict()
    for k in sb:
        if k == "conv1.weight":
            assert torch.equal(sc[k][:, :4], sb[k]) and sc[k][:, 4:].abs().max() == 0
        else:
            assert torch.equal(sc[k], sb[k]), k
    base.eval(), cam.eval()
    x, prev = _lnes(3, seed=20), _prev(3, seed=21)
    ob, oc = _fwd(base, x, prev), _fwd(cam, x, prev)
    assert (ob - oc).abs().max() < 1e-5                                      # summation order may differ in the last bits
    # the dense model with the same planes but default init would NOT have these properties
    torch.manual_seed(1234)
    plain = torch.nn.Conv2d(6, 3, kernel_size=3, padding=1)
    assert not torch.equal(plain.weight[:, :4], base.conv1.weight)


def test_cam_planes_are_the_hand_formula():
    """Per sample, at the corners and the centre, for fx != fy and different principal points; render intrinsics = K * 0.375."""
    cam = _build(MNISTModel, _cfg(DT / "dt_cam.yaml"))
    Ks = torch.stack([_K(610.0, 580.0, 330.0, 250.0), _K(480.0, 700.0, 300.0, 230.0), BASE_K])
    planes = cam._cam_planes(Ks)
    assert planes.shape == (3, H, W, 2) and planes.dtype == torch.float32
    s = 0.375
    for b in range(3):
        K = Ks[b].double()
        fx, fy, cx, cy = K[0, 0] * s, K[1, 1] * s, K[0, 2] * s, K[1, 2] * s
        for r, c in [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1), (H // 2, W // 2), (45, 200), (150, 17)]:
            assert abs(float(planes[b, r, c, 0]) - float((c - cx) / fx)) < 1e-6, (b, r, c)
            assert abs(float(planes[b, r, c, 1]) - float((r - cy) / fy)) < 1e-6, (b, r, c)
    # the plane value is the slope x / z of the ray the render's rounding assigns to that pixel
    for b in range(3):
        K = Ks[b]
        x, z = 0.04, 0.55
        u = float(K[0, 0] * s * x / z + K[0, 2] * s)
        c = int(round(u))
        assert abs(float(planes[b, 0, c, 0]) - x / z) < 0.5 / float(K[0, 0] * s) + 1e-6


def test_cam_planes_are_the_slopes_of_the_pixels_the_render_projects_vertices_to():
    """Ties the pixel convention to the render itself: every vertex lands on the integer pixel (round(u), round(v)) of
    `_render_chunk` (whose silhouette is set there), and the planes at that pixel are the vertex's x / z and y / z to
    within half a pixel of slope, per sample, for different K."""
    m = _build(MNISTModel, _cfg(DT / "dt_cam.yaml")).eval()
    prev = _prev(2, seed=70)
    Ks = torch.stack([_K(610.0, 580.0, 330.0, 250.0), _K(480.0, 700.0, 300.0, 230.0)])
    betas, k_f = m._resolve_betas_K(prev, None, Ks)
    with torch.no_grad():
        verts, _ = m._fk(prev, betas)
        fx, fy, cx, cy = m._intrinsics(k_f)
        x, y, z = verts.unbind(-1)
        ui = (fx[:, None] * x / z + cx[:, None]).round().long()
        vi = (fy[:, None] * y / z + cy[:, None]).round().long()
        valid = (z > 1e-6) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        planes, rend = m._cam_planes(k_f), m._render_prev(prev, betas, k_f)
    assert int(valid.sum()) > 800
    for b in range(2):
        s = valid[b]
        pu, pv = planes[b, vi[b][s], ui[b][s], 0], planes[b, vi[b][s], ui[b][s], 1]
        assert bool(((pu - (x / z)[b][s]).abs() <= 0.5 / fx[b] + 1e-6).all())
        assert bool(((pv - (y / z)[b][s]).abs() <= 0.5 / fy[b] + 1e-6).all())
        assert bool((rend[b, vi[b][s], ui[b][s], 0] == 1.0).all())


def test_cam_planes_reach_conv1_as_the_last_two_channels(cam_model):
    """Through the forward: the conv1 input is [lnes(2), sil, inv, plane_u, plane_v]; the first four channels are
    exactly what the model without planes feeds, in the same order."""
    base = _randomize(_build(MNISTModel, _cfg(DT / "dt_base.yaml"))).eval()
    x, prev = _lnes(2, seed=22), _prev(2, seed=23)
    Ks = torch.stack([_K(610.0, 580.0, 330.0, 250.0), _K(480.0, 700.0, 300.0, 230.0)])
    cin = _conv1_input(cam_model, x, prev, camera_K=Ks)
    cb = _conv1_input(base, x, prev, camera_K=Ks)
    assert cin.shape == (2, 6, H, W) and cb.shape == (2, 4, H, W) and cin.dtype == torch.float32
    assert torch.equal(cin[:, :4], cb)
    assert torch.equal(cin[:, :2], x.permute(0, 3, 1, 2))
    want = cam_model._cam_planes(Ks).permute(0, 3, 1, 2)
    assert torch.equal(cin[:, 4:], want)
    assert (cin[:, 4] - cin[:, 4].roll(1, dims=0)).abs().max() > 0          # the two samples have different K
    # an explicit K and the sequence context give the same planes
    cam_model.set_hand_context(torch.zeros(10), Ks[1])
    ctx = _conv1_input(cam_model, x[1:], prev[1:])
    cam_model.set_hand_context(None, None)
    assert torch.equal(ctx[:, 4:], cin[1:, 4:])
    # a batch is the stack of its samples
    one = _conv1_input(cam_model, x[:1], prev[:1], camera_K=Ks[:1])
    assert torch.equal(one[:, 4:], cin[:1, 4:])


def test_cam_output_changes_with_k_and_is_bitwise_unchanged_for_the_same_k(cam_model):
    x, prev = _lnes(2, seed=24), _prev(2, seed=25)
    K1 = torch.stack([_K(610.0, 580.0, 330.0, 250.0)] * 2)
    K2 = torch.stack([_K(560.0, 640.0, 310.0, 260.0)] * 2)
    for bf16 in (False, True):
        a, b = _fwd(cam_model, x, prev, bf16, camera_K=K1), _fwd(cam_model, x, prev, bf16, camera_K=K1)
        assert torch.equal(a, b)
        assert not torch.equal(a, _fwd(cam_model, x, prev, bf16, camera_K=K2))


def test_cam_planes_are_the_only_k_dependence_when_the_render_is_empty():
    """A hand behind the camera renders nothing for any K, so K can reach the output only through the planes: with
    the plane weights at their zero initialisation the output does not depend on K at all, with non-zero ones it does."""
    m = _randomize(_build(MNISTModel, _cfg(DT / "dt_cam.yaml"))).eval()
    x, prev = _lnes(2, seed=26), _prev(2, seed=27)
    prev[:, 2] = -1.0
    K1 = torch.stack([_K(610.0, 580.0, 330.0, 250.0)] * 2)
    K2 = torch.stack([_K(560.0, 640.0, 310.0, 260.0)] * 2)
    assert torch.equal(_fwd(m, x, prev, camera_K=K1), _fwd(m, x, prev, camera_K=K2))
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        m.conv1.weight[:, 4:] = 0.2 * torch.randn(3, 2, 3, 3, generator=g)
    o1, o2 = _fwd(m, x, prev, camera_K=K1), _fwd(m, x, prev, camera_K=K2)
    assert not torch.equal(o1, o2)
    # and the same K again is bitwise the same
    assert torch.equal(o1, _fwd(m, x, prev, camera_K=K1))


def test_cam_planes_get_gradient_and_have_no_gradient_path_to_k():
    m = _build(MNISTModel, _cfg(DT / "dt_cam.yaml")).train()
    x, prev = _lnes(2, seed=28), _prev(2, seed=29)
    K = torch.stack([_K(610.0, 580.0, 330.0, 250.0)] * 2).requires_grad_(True)
    out = m(x, prev, camera_K=K)
    out.sum().backward()
    assert K.grad is None or K.grad.abs().max() == 0                        # planes are built under no_grad
    assert m.conv1.weight.grad[:, 4:].abs().max() > 0                        # the zero-initialised slices do learn


def test_cam_warm_start_from_a_four_channel_checkpoint():
    """`init_from_abs_checkpoint_state` already pads conv1's extra channels with zeros; it loads the default model into
    the planes model, which is then the same function."""
    base = _randomize(_build(MNISTModel, _cfg(DT / "dt_base.yaml")))
    cam = _build(MNISTModel, _cfg(DT / "dt_cam.yaml"), seed=99)
    loaded = cam.init_from_abs_checkpoint_state(base.state_dict())
    assert loaded == len(base.state_dict())
    sb, sc = base.state_dict(), cam.state_dict()
    assert torch.equal(sc["conv1.weight"][:, :4], sb["conv1.weight"]) and sc["conv1.weight"][:, 4:].abs().max() == 0
    assert all(torch.equal(sc[k], sb[k]) for k in sb if k != "conv1.weight")


@pytest.mark.parametrize("event_channels, render", [
    (["last"], ["sil", "inv"]), (["last", "count", "first"], ["sil", "inv"]),
    (["last"], ["sil", "inv", "semsil"]), (["last"], ["inv"])], ids=str)
def test_cam_planes_stay_last_for_any_event_and_render_channel_set(event_channels, render):
    cfg = _cfg(DT / "dt_cam.yaml", RENDER_CHANNELS=render)
    cfg["DATA"]["EVENT_CHANNELS"] = event_channels
    m = _build(MNISTModel, cfg).eval()
    n_ev, n_in = 2 * len(event_channels), 2 * len(event_channels) + len(render)
    assert m.conv1.in_channels == n_in + 2 and tuple(m.conv1.weight.shape) == (3, n_in + 2, 3, 3)
    x, prev = torch.rand(2, H, W, n_ev), _prev(2, seed=60)
    Ks = torch.stack([_K(610.0, 580.0, 330.0, 250.0), _K(480.0, 700.0, 300.0, 230.0)])
    cin = _conv1_input(m, x, prev, camera_K=Ks)
    assert cin.shape == (2, n_in + 2, H, W)
    assert torch.equal(cin[:, -2:], m._cam_planes(Ks).permute(0, 3, 1, 2))
    assert torch.equal(cin[:, :n_ev], x.permute(0, 3, 1, 2))
    assert float(m.conv1.weight[:, n_in:].abs().max()) == 0.0


# ================================================================================================ (d) PREV_MLP_TRANSL false
def test_pmt_state_dict_is_identical_in_keys_and_shapes_and_loads_both_ways():
    base = _randomize(_build(MNISTModel, _cfg(DT / "dt_base.yaml")))
    pmt = _build(MNISTModel, _cfg(DT / "dt_pmt.yaml"))
    assert pmt.prev_mlp_transl is False
    sa, sb = base.state_dict(), pmt.state_dict()
    assert list(sa) == list(sb) and all(sa[k].shape == sb[k].shape and sa[k].dtype == sb[k].dtype for k in sa)
    pmt.load_state_dict(sa)                                                  # strict
    base.load_state_dict(pmt.state_dict())
    assert sum(p.numel() for p in base.parameters()) == sum(p.numel() for p in pmt.parameters())


@pytest.mark.parametrize("bf16", [False, True], ids=["fp32", "bf16"])
def test_pmt_output_equals_the_model_with_the_translation_rows_zeroed_by_hand(bf16):
    default = _randomize(_build(MNISTModel, _cfg(DT / "dt_base.yaml"))).eval()
    pmt = _randomize(_build(MNISTModel, _cfg(DT / "dt_pmt.yaml"))).eval()
    hand = copy.deepcopy(default)
    with torch.no_grad():
        hand.prev_mlp[2].weight[:3] = 0.0
        hand.prev_mlp[2].bias[:3] = 0.0
    x, prev = _lnes(5, seed=30, empty=(4,)), _prev(5, seed=31)
    for B in (1, 5):
        od, op, oh = (_fwd(m, x[:B], prev[:B], bf16) for m in (default, pmt, hand))
        assert torch.equal(op, oh), B                                          # translation, rotation, fingers
        # rotation and fingers are the default model's, only the translation differs
        assert torch.equal(op[:, 3:], od[:, 3:])
        assert not torch.equal(op[:, :3], od[:, :3])
    # the translation difference is exactly prev_mlp's translation output
    pm = default.prev_mlp(prev)[:, :3]
    assert pm.abs().max() > 1e-3
    od, op = _fwd(default, x[:4], prev[:4]), _fwd(pmt, x[:4], prev[:4])
    assert torch.allclose(od[:, :3] - op[:, :3], pm[:4], atol=1e-6)
    # an event-free packet still returns prev
    assert torch.equal(_fwd(pmt, x[4:5], prev[4:5]), prev[4:5])


def test_pmt_masked_outputs_get_zero_gradient_and_the_rest_learns():
    m = _randomize(_build(MNISTModel, _cfg(DT / "dt_pmt.yaml"))).train()
    out = m(_lnes(3, seed=32), _prev(3, seed=33))
    out.pow(2).sum().backward()
    w, b = m.prev_mlp[2].weight.grad, m.prev_mlp[2].bias.grad
    assert w[:3].abs().max() == 0 and b[:3].abs().max() == 0
    assert w[3:].abs().max() > 0 and b[3:].abs().max() > 0
    # without the mask the translation rows do get gradient
    d = _randomize(_build(MNISTModel, _cfg(DT / "dt_base.yaml"))).train()
    d(_lnes(3, seed=32), _prev(3, seed=33)).pow(2).sum().backward()
    assert d.prev_mlp[2].weight.grad[:3].abs().max() > 0
    # the translation columns of the first prev_mlp layer's gradient come from the rotation / finger outputs only
    assert m.prev_mlp[0].weight.grad.abs().max() > 0


def test_pmt_with_the_12d_layout_masks_the_translation_columns():
    a = _randomize(_build(MNISTModel, _mini_cfg(POSE_REPR="mano_pca6", OUTPUT_DIM=12, PREVPOS_EMBED=True))).eval()
    b = _randomize(_build(MNISTModel, _mini_cfg(POSE_REPR="mano_pca6", OUTPUT_DIM=12, PREVPOS_EMBED=True,
                                                PREV_MLP_TRANSL=False))).eval()
    x, p = torch.randn(2, H, W, 2), torch.randn(2, 12) * 0.2
    oa, ob = _fwd(a, x, p), _fwd(b, x, p)
    assert torch.equal(oa[:, :6], ob[:, :6]) and torch.equal(oa[:, 9:], ob[:, 9:])
    assert torch.allclose(oa[:, 6:9] - ob[:, 6:9], a.prev_mlp(p)[:, 6:9], atol=1e-6)


# ===================================================================================================== (e) CNN_BACKBONE
@pytest.mark.parametrize("name", list(BB.BACKBONES))
def test_every_backbone_builds_in_the_dense_model_and_trains(name):
    m = _build(MNISTModel, _cfg(DT / "dt_base.yaml", CNN_BACKBONE=name)).train()
    ref = BB.build_backbone(name, 51)
    assert type(m.rn) is type(ref) and m.cnn_backbone == name
    assert sum(p.numel() for p in m.rn.parameters()) == sum(p.numel() for p in ref.parameters())
    x, prev = _lnes(2, seed=40), _prev(2, seed=41)
    out = m(x, prev)
    assert out.shape == (2, 51) and torch.isfinite(out).all()
    out.pow(2).sum().backward()
    bad = [n for n, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not bad, bad


def test_resnet18_backbone_is_the_default_bit_for_bit(orig):
    a = _randomize(_build(orig.MNISTModel, _cfg(DT / "dt_base.yaml"))).eval()
    b = _randomize(_build(MNISTModel, _cfg(DT / "dt_base.yaml", CNN_BACKBONE="resnet18"))).eval()
    _assert_same_state(a, b)
    x, prev = _lnes(3, seed=42), _prev(3, seed=43)
    assert torch.equal(_fwd(a, x, prev), _fwd(b, x, prev))


def test_backbone_widths_and_depths_change_only_the_trunk():
    base = _build(MNISTModel, _cfg(DT / "dt_base.yaml"))
    for name in ("resnet18_w0.5", "resnet18_l3"):
        m = _build(MNISTModel, _cfg(DT / "dt_base.yaml", CNN_BACKBONE=name))
        assert [k for k in m.state_dict() if not k.startswith("rn.")] == [k for k in base.state_dict() if not k.startswith("rn.")]
        assert m.rn.fc.out_features == 51 and m.prev_mlp[2].out_features == 51 and m.conv1.out_channels == 3


@pytest.mark.parametrize("bad", ["resnet50", "", "ResNet18", None, 18, ["resnet18"]])
def test_unknown_backbone_name_raises(bad):
    with pytest.raises(ValueError):
        _build(MNISTModel, _cfg(DT / "dt_base.yaml", CNN_BACKBONE=bad))


@pytest.mark.parametrize("name", ["resnet18", "resnet18_w0.5"])
def test_backbone_with_an_encoder_or_active_head_raises(name):
    with pytest.raises(ValueError, match="CNN_BACKBONE"):
        _build(MNISTModel, _cfg(REPO / "configs/semkine_recipes/s37.yaml", CNN_BACKBONE=name))      # ENCODER + ACTIVE_HEAD
    with pytest.raises(ValueError, match="CNN_BACKBONE"):
        _build(MNISTModel, _mini_cfg(ACTIVE_HEAD=True, CNN_BACKBONE=name))                         # dense active head
    with pytest.raises(ValueError, match="CNN_BACKBONE"):
        _build(MNISTModel, _mini_cfg(ENCODER="event_gnn", CNN_BACKBONE=name))
    # the legacy dead key stays accepted and ignored
    _build(MNISTModel, _cfg(RT_CFG))
    assert _load(RT_CFG)["MODEL"]["BACKBONE"] == "resnet18"


# ======================================================================================== (f) keys, validation, combinations
def test_unknown_model_keys_still_raise_and_the_new_keys_are_accepted():
    with pytest.raises(ValueError, match="unknown MODEL keys"):
        _build(MNISTModel, _cfg(DT / "dt_base.yaml", BACKBONES="resnet18"))
    with pytest.raises(ValueError, match="unknown MODEL keys"):
        _build(MNISTModel, _cfg(DT / "dt_base.yaml", ROOT_COMPOSITION="so3"))
    with pytest.raises(ValueError, match="unknown TRACK keys"):
        cfg = _cfg(DT / "dt_base.yaml")
        cfg["TRACK"]["NOT_A_KEY"] = 1
        _build(MNISTModel, cfg)
    for k in NEW_KEYS:
        assert k in MNISTModel.MODEL_KEYS
    m = _build(MNISTModel, _cfg(DT / "dt_base.yaml", CNN_BACKBONE="resnet18_w0.5", CAM_PLANES=True, ROOT_COMPOSE="so3",
                                PREV_MLP_TRANSL=False))
    assert (m.cnn_backbone, m.cam_planes, m.root_compose, m.prev_mlp_transl) == ("resnet18_w0.5", True, "so3", False)


@pytest.mark.parametrize("model, match", [
    ({"CAM_PLANES": True, "PREV_RENDER": False}, "CAM_PLANES"),
    ({"CAM_PLANES": "true"}, "CAM_PLANES"),
    ({"CAM_PLANES": 1}, "CAM_PLANES"),
    ({"ROOT_COMPOSE": "quat"}, "ROOT_COMPOSE"),
    ({"ROOT_COMPOSE": True}, "ROOT_COMPOSE"),
    ({"ROOT_COMPOSE": "so3", "PREDICT_DELTA": False}, "ROOT_COMPOSE"),
    ({"PREV_MLP_TRANSL": False, "PREVPOS_EMBED": False}, "PREV_MLP_TRANSL"),
    ({"PREV_MLP_TRANSL": "false"}, "PREV_MLP_TRANSL"),
    ({"PREV_MLP_TRANSL": 0}, "PREV_MLP_TRANSL"),
])
def test_impossible_values_and_combinations_raise(model, match):
    with pytest.raises(ValueError, match=match):
        _build(MNISTModel, _cfg(DT / "dt_base.yaml", **model))


@pytest.mark.parametrize("key, value", [("CAM_PLANES", True), ("ROOT_COMPOSE", "so3"), ("PREV_MLP_TRANSL", False)])
def test_dense_only_keys_raise_on_the_raw_event_arms(key, value):
    for base in (REPO / "configs/semkine_recipes/s37.yaml", REPO / "configs/s38/s38_spabs_2k.yaml"):
        with pytest.raises(ValueError, match=key):
            _build(MNISTModel, _cfg(base, **{key: value}))
    # explicit defaults are harmless anywhere
    _build(MNISTModel, _cfg(REPO / "configs/semkine_recipes/s37.yaml", **{"CAM_PLANES": False, "ROOT_COMPOSE": "add",
                                                                         "PREV_MLP_TRANSL": True}))


def test_cam_planes_without_prev_render_raises_even_on_the_plain_dense_model():
    with pytest.raises(ValueError, match="CAM_PLANES"):
        _build(MNISTModel, _mini_cfg(CAM_PLANES=True))
    _build(MNISTModel, _mini_cfg(CAM_PLANES=False))


def test_the_four_features_work_together_and_train():
    """CAM_PLANES + ROOT_COMPOSE so3 + PREV_MLP_TRANSL false + a small trunk: builds, forward (B = 1, 3), eval / train,
    fp32 / bf16, finite gradients on every parameter, event-free packets held exactly."""
    cfg = _cfg(DT / "dt_base.yaml", CAM_PLANES=True, ROOT_COMPOSE="so3", PREV_MLP_TRANSL=False,
               CNN_BACKBONE="resnet18_w0.5")
    m = _randomize(_build(MNISTModel, cfg))
    assert m.conv1.in_channels == 6 and m.rn.fc.out_features == 51
    x, prev = _lnes(3, seed=50, empty=(1,)), _prev(3, seed=51, norm=3.0)
    m.eval()
    for B in (1, 3):
        for bf16 in (False, True):
            out = _fwd(m, x[:B], prev[:B], bf16)
            assert out.shape == (B, 51) and torch.isfinite(out).all() and out[:, 3:6].norm(dim=-1).max() <= np.pi + 1e-5
    assert torch.equal(_fwd(m, x, prev)[1], prev[1])
    m.train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = m(x, prev)
    out.float().pow(2).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    assert m.prev_mlp[2].weight.grad[:3].abs().max() == 0


def test_every_new_dt_config_builds_with_the_expected_parameter_count():
    """The generated arms of the pre-registration that this model implements (counts measured at the time of writing)."""
    want = {"dt_base": 11_209_429, "dt_cam": 11_209_483, "dt_so3c": 11_209_429, "dt_w05": 2_818_741,
            "dt_l3": 2_802_645, "dt_pmt": 11_209_429}
    for name, n in want.items():
        m = _build(MNISTModel, _cfg(DT / f"{name}.yaml"))
        assert sum(p.numel() for p in m.parameters()) == n, name
    m = _build(MNISTModel, _cfg(DT / "dt_cam.yaml"))
    assert m.cam_planes and m.conv1.in_channels == 6
    assert _build(MNISTModel, _cfg(DT / "dt_so3c.yaml")).root_compose == "so3"
    assert _build(MNISTModel, _cfg(DT / "dt_pmt.yaml")).prev_mlp_transl is False
    assert _build(MNISTModel, _cfg(DT / "dt_w05.yaml")).cnn_backbone == "resnet18_w0.5"
    assert _build(MNISTModel, _cfg(DT / "dt_l3.yaml")).cnn_backbone == "resnet18_l3"
