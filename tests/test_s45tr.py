"""S45-TR (docs/EXPERIMENT_RECORDS.md#s45tr-prereg-20261030): every arm config is S45 plus exactly its registered
change; N1 only zeroes the readout's previous-pose input and keeps S45's state dict and random stream; the N3 / N4
prior hooks change nothing but the prior of the registered samples, draw the offset inside the valid run, and leave
validation and S45's own per-sample stream untouched."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

from semkine import s45tr_n1 as N1  # noqa: E402
from semkine import s45tr_prior as P  # noqa: E402
from semkine import xyz_mesh as X  # noqa: E402
from semkine.xyz_mesh import associate_rays, vertex_face_table  # noqa: E402
from tools.s45tr import common as C  # noqa: E402

NPZ = ROOT / "assets" / "mano_right.npz"


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(old)


def scene(B=2, N=192, seed=0):
    with np.load(NPZ, allow_pickle=False) as z:
        faces = torch.tensor(np.asarray(z["f"], np.int64))
        lbs = torch.tensor(np.asarray(z["weights"], np.float32))
        v = torch.tensor(np.asarray(z["v_template"], np.float32))
    g = torch.Generator().manual_seed(seed)
    verts = torch.stack([v + torch.tensor([0.0, 0.0, 0.5 + 0.2 * b]) + 0.01 * torch.randn(1, 3, generator=g)
                         for b in range(B)])
    pick = torch.randint(0, 778, (B, N), generator=g)
    target = verts[torch.arange(B)[:, None], pick] + 0.002 * torch.randn(B, N, 3, generator=g)
    d = target / target[..., 2:3]
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[:, -16:] = False
    assoc = associate_rays(d, mask, verts, faces, vertex_face_table(faces, 778), torch.full((B,), 16 / 300.0),
                           torch.full((B,), 3 / 300.0), 0.01)
    tok = torch.randn(B, N, 7, generator=g)
    prev48 = torch.randn(B, 48, generator=g) / 10
    return faces, lbs, (tok, d, mask, verts, assoc, prev48)


def n1_pair():
    faces, lbs, x = scene()
    torch.manual_seed(3407)
    a = X.XYZMeshNet(lbs, faces)
    after_a = torch.randn(4)
    torch.manual_seed(3407)
    b = N1.N1XYZMeshNet(lbs, faces)
    after_b = torch.randn(4)
    with torch.no_grad():
        w = torch.randn(a.fc2.weight.shape, generator=torch.Generator().manual_seed(6)) * .02
        a.fc2.weight.copy_(w)
        b.fc2.weight.copy_(w)
    return a.eval(), b.eval(), x, torch.equal(after_a, after_b)


def test_n1_keeps_s45_state_dict_and_random_stream():
    a, b, _, same_stream = n1_pair()
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)
    assert same_stream


def test_n1_is_s45_with_zero_prev48_and_ignores_prev48():
    a, b, x, _ = n1_pair()
    tok, d, mask, verts, assoc, prev48 = x
    with torch.no_grad():
        out_b = b(*x)[0]
        ref = a(tok, d, mask, verts, assoc, torch.zeros_like(prev48))[0]
        other = b(tok, d, mask, verts, assoc, prev48 * 7 + 1)[0]
        s45 = a(*x)[0]
    assert torch.equal(out_b, ref) and torch.equal(out_b, other)
    assert not torch.equal(out_b, s45)


def test_n1_prev48_columns_get_zero_gradient():
    _, b, x, _ = n1_pair()
    b.train()
    b(*x)[0].square().sum().backward()
    cols = b.fc1.weight.grad[:, X.XYZMeshNet.N_SLOTS * 32:X.XYZMeshNet.N_SLOTS * 32 + 48]
    assert torch.count_nonzero(cols) == 0 and torch.count_nonzero(b.fc1.weight.grad) > 0


def test_n1_install_routes_constructor():
    saved = X.XYZMeshNet
    try:
        N1.install()
        assert N1.installed() and X.XYZMeshNet is N1.N1XYZMeshNet
    finally:
        X.XYZMeshNet = saved


def test_offset_draw_stays_in_run_and_in_range():
    rng = np.random.default_rng(0)
    for _ in range(20000):
        run_a = int(rng.integers(0, 1000))
        run_b = run_a + int(rng.integers(1, 5000))
        start = int(rng.integers(run_a, run_b))
        ms, sign, mag = P.offset_ms(rng, start, run_a, run_b, 100, 1000)
        assert run_a <= ms <= run_b - 1 and 100 <= mag <= 1000 and sign in (-1, 1)
        if run_a <= start + sign * mag <= run_b - 1:
            assert ms == start + sign * mag
        elif run_a <= start - sign * mag <= run_b - 1:
            assert ms == start - sign * mag
    signs = [P.offset_ms(np.random.default_rng(i), 5000, 0, 10000, 100, 1000)[1] for i in range(4000)]
    assert abs(np.mean(signs)) < 0.05


def test_arm_configs_are_s45_plus_one_registered_change():
    s45 = yaml.safe_load(C.S45_CONFIG.read_text())
    assert C.sha(C.S45_CONFIG) == C.S45_CONFIG_SHA256
    for arm, change in C.ARMS.items():
        for phase in ("preflight", "formal"):
            if phase == "formal" and arm not in C.FORMAL_ARMS:
                continue
            path = C.config_path(arm, phase)
            if not path.exists():
                pytest.skip("configs not generated yet")
            cfg = yaml.safe_load(path.read_text())
            assert cfg == C.expected_raw(arm, phase, s45)
            base = {k: v for k, v in s45.items() if k != "C1SF"}
            flat_cfg, flat_base = flatten(cfg), flatten(base)
            differ = {k for k in set(flat_cfg) | set(flat_base) if flat_cfg.get(k) != flat_base.get(k)}
            allowed = set(change) | {"S45TR.ARM", "TRAIN.RUN_NAME", "TRAIN.OUTPUT_DIR", "EVAL.OUTPUT_DIR"}
            if phase == "preflight":
                allowed |= {"TRAIN.MAX_STEPS", "TRAIN.VAL_CHECK_INTERVAL"}
            assert differ <= allowed, (arm, phase, differ - allowed)


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and not (prefix + k == "S45TR.T1A_GNN"):
            out.update(flatten(v, prefix + k + "."))
        else:
            out[prefix + k] = v
    return out


# ---------------------------------------------------------------- the prior hooks on the real training set
def dataset(split, train):
    from config import load_config
    from semkine.dataset import build_dataset
    cfg = load_config(str(C.S45_CONFIG))
    components = np.load(cfg["MANO"]["NPZ"])["hands_components"].astype(np.float32)
    return cfg, build_dataset(cfg, split, components, train=train)


def training_set():
    return dataset("train", True)


def fields(packet):
    return {k: getattr(packet, k) for k in ("events", "target", "prev_state", "betas", "camera_K", "t_start_us",
                                             "t_end_us", "sequence_id")}


def same(a, b):
    return all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in a)


def test_n3_changes_only_the_large_and_directed_priors():
    _, ds = training_set()
    rng = np.random.default_rng(1)
    idx = rng.choice(len(ds), 48, replace=False)
    base = [fields(ds[int(i)]) for i in idx]
    try:
        P.install_n3(100, 1000)
        arm = [fields(ds[int(i)]) for i in idx]
    finally:
        P.uninstall()
    changed = 0
    for i, a, b in zip(idx, base, arm):
        u = s45_branch_draw(ds, int(i))
        rest = [k for k in a if k != "prev_state"]
        assert all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in rest)
        if u < ds.prev_noise_mix[0]:
            assert np.array_equal(a["prev_state"], b["prev_state"])
        else:
            changed += 1
            assert not np.array_equal(a["prev_state"], b["prev_state"])
    assert changed > 10


def s45_branch_draw(ds, idx):
    """S45's first noise draw `u` for sample idx, replayed through the original code path."""
    seen = {}
    original = type(ds)._sample_prev_noise

    def spy(self, rng):
        state = rng.bit_generator.state
        seen["u"] = float(rng.random())
        rng.bit_generator.state = state
        return original(self, rng)
    type(ds)._sample_prev_noise = spy
    try:
        ds[idx]
    finally:
        type(ds)._sample_prev_noise = original
    return seen["u"]


def test_n3_offset_prior_is_the_runs_ground_truth_under_the_same_augmentation():
    from semkine import domrand as DR
    cfg, ds = training_set()
    rng = np.random.default_rng(2)
    checked = 0
    for i in rng.choice(len(ds), 64, replace=False):
        i = int(i)
        if s45_branch_draw(ds, i) < ds.prev_noise_mix[0]:
            continue
        si, end, run_a, run_b = (int(v) for v in ds.index[i])
        window = ds._window(i, np.random.default_rng((ds.seed * 1_000_003 + i) & 0x7FFFFFFF))[2]
        start = end - window + 1
        ms, _, _ = P.offset_ms(np.random.default_rng([ds.seed, i, P.TAG_N3]), start, run_a, run_b, 100, 1000)
        h = ds._handle(si)
        try:
            P.install_n3(100, 1000)
            pkt = ds[i]
        finally:
            P.uninstall()
        base = ds[i]
        # the S45 sample's target went through the same realisation: recover it via the ground-truth row
        captured = {}
        orig = DR.transform_labels

        def spy(pos, dr, K, j0, scale):
            captured.setdefault("dr", dr)
            return orig(pos, dr, K, j0, scale)
        DR.transform_labels = spy
        try:
            ds[i]
        finally:
            DR.transform_labels = orig
        expect = orig(np.stack([h.pos51[ms], h.pos51[end]]).astype(np.float32), captured["dr"], h.camera_K.copy(),
                      h.j0, ds.render_scale)[0][0]
        assert np.array_equal(pkt.prev_state, expect) and np.array_equal(pkt.target, base.target)
        checked += 1
    assert checked >= 10


def test_validation_set_untouched():
    _, ds = dataset("val_core", False)
    idx = [0, len(ds) // 2, len(ds) - 1]
    base = [fields(ds[i]) for i in idx]
    try:
        P.install_n3(100, 1000)
        arm = [fields(ds[i]) for i in idx]
    finally:
        P.uninstall()
    assert all(same(a, b) for a, b in zip(base, arm))
