"""AGG778 probe tooling (docs/EXPERIMENT_RECORDS.md#agg778-prereg-20261024): the probes rebuild C1's records and
pooling exactly, the nulls keep what they claim to keep, the anchors are C1's own noise, the statistics are right."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "tests"))

from test_xcl_model import inputs, net  # noqa: E402
from tools.agg778 import common as C  # noqa: E402
from tools.agg778.heads import c1_records, pool_records  # noqa: E402


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def scene_capture():
    m = net().eval()
    tok, d, mask, verts, assoc, prev48 = inputs()
    store = {}
    m.vin.register_forward_pre_hook(lambda mod, i: store.__setitem__("x", i[0].detach()))
    rel = []
    m.rel.register_forward_hook(lambda mod, i, o: rel.append((i[0].detach(), o.detach())))
    with torch.no_grad():
        m(tok, d, mask, verts, assoc, prev48, capture=True)
    cap = {"assoc": assoc, "d": d, "mask": mask, "verts": verts, "tok": tok, "O": m.xag_last_capture["O"]}
    return m, cap, store["x"], rel[0]


def test_probe_records_are_the_modules_records():
    m, cap, x, (feat, phi) = scene_capture()
    rec = c1_records(cap)
    assert rec["feat"].shape == feat.shape and torch.equal(rec["feat"], feat.float())


def test_probe_pooling_equals_module_pooling_and_reference():
    from tools.agg778.a_audit import reference_pool
    m, cap, x, (feat, phi) = scene_capture()
    rec = c1_records(cap)
    V = cap["verts"].shape[1]
    old = C.V
    try:
        import tools.agg778.heads as H
        H.V, C.V = V, V
        xp, O = pool_records(phi.float(), rec["rb"], rec["vrec"], rec["wrec"], 1, "c1")
        xr, has = reference_pool(phi.numpy(), rec["rb"].numpy(), rec["vrec"].numpy(), rec["wrec"].numpy(), 1)
    finally:
        H.V, C.V = 778, old
    assert torch.allclose(xp, x.float(), atol=1e-6)
    assert np.allclose(xr, x.double().numpy(), atol=1e-6)
    assert torch.equal(O > 0, cap["O"] > 0)


def test_mass_pooling_is_weight_consistent():
    import tools.agg778.heads as H
    phi = torch.tensor([[1.0, -2.0], [3.0, 0.5]])
    rb = torch.tensor([0, 0])
    v = torch.tensor([1, 1])
    w = torch.tensor([0.99, 0.01])
    H.V = 3
    try:
        x, O = pool_records(phi, rb, v, w, 1, "mass")
        xc, Oc = pool_records(phi, rb, v, w, 1, "c1")
    finally:
        H.V = 778
    # the unweighted max lets the 1 %-weight record set channel 0; the weighted RMS barely moves
    assert xc[0, 1, 2] == 3.0
    assert abs(float(x[0, 1, 2]) - float(np.sqrt(0.99 * 1 + 0.01 * 9))) < 1e-6
    assert float(O[0, 1]) == pytest.approx(1.0) and float(O[0, 0]) == 0.0


def test_noise_sampler_is_the_datasets_own_draw():
    from semkine.dataset import SemKineDataset
    from config import load_config
    from types import SimpleNamespace
    cfg = load_config(str(ROOT / "configs/xyz_c1_repro_20261018/xyz_c1_stable_s3407.yaml"))
    draw = C.noise_sampler(cfg)
    t = cfg["TRACK"]
    ns = SimpleNamespace(prev_noise=(t["PREV_NOISE_T"], t["PREV_NOISE_R"], t["PREV_NOISE_POSE"]),
                         prev_noise_mode="mixed",
                         prev_noise_large=(t["PREV_NOISE_LARGE_T"], t["PREV_NOISE_LARGE_R"], t["PREV_NOISE_LARGE_POSE"]),
                         prev_noise_mix=(0.5, 0.3, 0.2))
    a, b = np.random.default_rng(5), np.random.default_rng(5)
    kinds = []
    for _ in range(200):
        n1, k = draw(a)
        n2 = SemKineDataset._sample_prev_noise(ns, b)
        assert np.array_equal(n1, n2)
        kinds.append(k)
    assert set(kinds) == {0, 1, 2}


def _items(n=3, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        m = 50 + 10 * i
        ev = np.zeros((m, 5), np.float32)
        ev[:, 1] = rng.integers(0, 240, m)
        ev[:, 2] = rng.integers(0, 180, m)
        ev[:, 3] = np.sort(rng.random(m)) * 0.05
        ev[:, 4] = rng.integers(0, 2, m)
        src = {k: np.arange(m) for k in ("xag_source_index", "xag_source_ms", "xag_part", "xag_face")}
        src.update(xag_bary=rng.random((m, 3)).astype(np.float32), xag_valid=np.ones(m, bool),
                   xag_depth=rng.random(m).astype(np.float32))
        out.append({"ev5": ev, "src": src, "i": i})
    return out


def test_position_permutation_keeps_coverage_time_and_counts():
    items = _items()
    evs, srcs = C.permute_positions(items, 3)
    for it, e, s in zip(items, evs, srcs):
        assert len(e) == len(it["ev5"])
        assert np.array_equal(e[:, 3:], it["ev5"][:, 3:]), "time and polarity stay per row (time ordered)"
        a = sorted(map(tuple, it["ev5"][:, 1:3].tolist()))
        b = sorted(map(tuple, e[:, 1:3].tolist()))
        assert a == b, "the same set of positions"
        assert not np.array_equal(e[:, 1:3], it["ev5"][:, 1:3])
        pi = s["xag_source_index"]
        assert np.array_equal(e[:, 1:3], it["ev5"][pi, 1:3]), "source labels follow the positions"


def test_thinning_is_deterministic_and_row_consistent():
    from tools.agg778.c_robust import thin
    items = _items()
    a, sa = thin(items, 1 / 2.4, 9)
    b, sb = thin(items, 1 / 2.4, 9)
    for x, y, s in zip(a, b, sa):
        assert np.array_equal(x, y) and len(s["xag_face"]) == len(x)


def test_statistics():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    y = 0.5 * x + rng.normal(size=500)
    st = np.array([len(x), x.sum(), y.sum(), (x * y).sum(), (x * x).sum(), (y * y).sum()])
    assert C.corr_from_stats(st) == pytest.approx(np.corrcoef(x, y)[0, 1], abs=1e-12)
    r2 = 1 - ((y - x) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    assert C.r2_from_stats(st) == pytest.approx(r2, abs=1e-12)
    blocks = np.repeat(np.arange(50), 10)
    ci = C.block_bootstrap({"v": x}, blocks)["v"]
    assert ci["lo"] < x.mean() < ci["hi"] and ci["blocks"] == 50


def test_arm_configs_differ_from_their_controls_only_where_registered():
    from tools.agg778 import d_common as D
    from config import load_config
    for arm, a in D.ARMS.items():
        ctrl = load_config(str(ROOT / D.CONTROLS[a["cond"]]["config"]))
        ctrl.pop("_config_path")
        for phase in ("formal", "preflight"):
            cfg = D.expected_config(arm, phase)
            diff = set()
            for s in set(cfg) | set(ctrl):
                a_, b_ = cfg.get(s), ctrl.get(s)
                if isinstance(a_, dict) or isinstance(b_, dict):
                    a_, b_ = a_ or {}, b_ or {}
                    diff |= {(s, k) for k in set(a_) | set(b_) if a_.get(k) != b_.get(k)}
                elif a_ != b_:
                    diff.add((s, None))
            allowed = {("MODEL", "XYZ_AGG"), ("TRAIN", "RUN_NAME"), ("TRAIN", "OUTPUT_DIR"), ("EVAL", "OUTPUT_DIR"),
                       ("XCL", "CELL")}
            if phase == "preflight":
                allowed |= {("TRAIN", "MAX_STEPS"), ("TRAIN", "VAL_CHECK_INTERVAL")}
            assert diff <= allowed, (arm, phase, diff - allowed)
            assert cfg["MODEL"]["XYZ_AGG"] == a["variant"]
