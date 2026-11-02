"""S45-RT plan 2 (docs/EXPERIMENT_RECORDS.md#s45rt-prereg-20261031): surface patches, event image, T0P routing, the
T1p hook and the T1p config; CPU only."""
from __future__ import annotations

from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from semkine import s45rt_router as RT  # noqa: E402
from semkine import xag as XAG  # noqa: E402
from semkine.events import EV_BATCH, EV_P, EV_T, EV_X, EV_Y  # noqa: E402
from semkine.s45ct_route import face_geodesic  # noqa: E402
from semkine.xyz_mesh import stride_sample  # noqa: E402

NPZ = ROOT / "assets/mano_right.npz"


@pytest.fixture(scope="module")
def table():
    return RT.patch_table(NPZ)


def test_patch_table_partition(table):
    patch, part, centres = table
    with np.load(NPZ) as z:
        dist = face_geodesic(z["v_template"], z["f"])
        assert len(patch) == len(z["f"])
    assert (patch >= 0).all() and patch.max() + 1 == len(centres)
    for k in range(len(centres)):
        assert len(set(part[patch == k])) == 1, "a patch crosses a part"
        assert patch[centres[k]] == k
    reach = dist[centres[patch], np.arange(len(patch))]
    assert reach.max() <= RT.PATCH_RADIUS_M + 1e-6
    assert RT.table_sha256(NPZ) == RT.table_sha256(NPZ)


def packets(rng, sizes, dt=0.05):
    rows, ptr = [], [0]
    for b, n in enumerate(sizes):
        t = np.sort(rng.uniform(0, dt, n))
        ev = np.zeros((n, 5), np.float32)
        ev[:, EV_BATCH], ev[:, EV_X], ev[:, EV_Y] = b, rng.integers(0, 240, n), rng.integers(0, 180, n)
        ev[:, EV_T], ev[:, EV_P] = t, rng.integers(0, 2, n)
        rows.append(ev)
        ptr.append(ptr[-1] + n)
    events = torch.from_numpy(np.concatenate(rows))
    return types.SimpleNamespace(events=events, ptr=torch.tensor(ptr), delta_t_s=torch.full((len(sizes),), dt))


def test_event_image_last_time_and_counts():
    rng = np.random.default_rng(0)
    batch = packets(rng, [300, 50])
    img = RT.event_image(batch.events, batch.ptr, batch.delta_t_s).numpy()
    ev = batch.events.numpy()
    ref = np.zeros_like(img)
    count = np.zeros_like(img)
    for e in ev:                                                     # time-sorted within each packet
        b, x, y, t, p = int(e[EV_BATCH]), int(e[EV_X]), int(e[EV_Y]), e[EV_T], int(e[EV_P] > 0.5)
        ref[b, p, y, x] = min(max(t / 0.05, 0.0), 1.0)
        count[b, 2 + p, y, x] += 1
    ref[:, 2:] = np.log1p(count[:, 2:])
    np.testing.assert_allclose(img, ref, rtol=0, atol=1e-6)


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return RT.build(NPZ).eval()


def test_forward_and_route(net):
    rng = np.random.default_rng(1)
    batch = packets(rng, [3000, 10, 0, 2500])
    with torch.no_grad():
        out = net(batch.events, batch.ptr, batch.delta_t_s)
        src, mask = stride_sample(batch.ptr, RT.MAX_NODES)
        assert torch.equal(out["src"][mask], src[mask]) and torch.equal(out["mask"], mask)
        assert out["patch"].shape == (*mask.shape, net.n_patches) and out["coord"].shape == (*mask.shape, 3)
        face, bary, pprob = net.route(out, mask)
    patch = out["patch"].argmax(-1)
    assert torch.equal(net.patch_of_face[face][mask], patch[mask]), "routed face outside the predicted patch"
    assert torch.allclose(bary[mask].sum(-1), torch.ones(int(mask.sum()))) and (bary[mask] >= 0).all()
    assert ((pprob > 0) & (pprob <= 1)).all()


def test_hook_install_routes_and_uninstall(net, tmp_path):
    ckpt = tmp_path / "t0p.ckpt"
    RT.save_checkpoint(net, ckpt, step=0, phase="test", heldout="ylf")
    original = XAG.sample_sources
    RT.install(ckpt, 0.5, 0.0)
    try:
        assert XAG.sample_sources is RT.predicted_sources
        assert RT.installed() == {"checkpoint": str(ckpt), "onhand_tau": 0.5, "patch_tau": 0.0}
        batch = packets(np.random.default_rng(2), [800, 1200])
        src, mask = stride_sample(batch.ptr, RT.MAX_NODES)
        s = XAG.sample_sources(batch, src, mask)
        assert set(s) == {"xag_source_index", "xag_source_ms", "xag_part", "xag_face", "xag_bary", "xag_valid"}
        assert s["xag_face"].shape == mask.shape and s["xag_bary"].shape == (*mask.shape, 3)
        assert not (s["xag_valid"] & ~mask).any()
        with pytest.raises(RuntimeError):
            XAG.sample_sources(batch, src.flip(-1), mask)
        assert all(not p.requires_grad for p in RT._STATE["net"].parameters())
    finally:
        RT.uninstall()
    assert XAG.sample_sources is original and RT.installed() is None


def test_t1p_config_is_one_change():
    from tools.s45rt.common import S45_CONFIG
    from tools.s45rt.plan2 import ROUTER_CHECKPOINT, TAUS, expected_raw
    s45 = yaml.safe_load(S45_CONFIG.read_text())
    for phase in ("preflight", "formal"):
        cfg = expected_raw("t1p", phase, s45)
        assert cfg["MODEL"]["XAG_ASSOC"] == "face_route" and cfg["MODEL"]["XAG_GNN"] == "legacy"
        assert cfg["S45RT"] == {"ARM": "t1p", "ROUTER": str(ROUTER_CHECKPOINT), **TAUS}
        assert "C1SF" not in cfg and "XAG_SOURCE_ROOT" not in cfg["DATA"]
        rest = {k: v for k, v in cfg.items() if k not in ("S45RT", "MODEL", "TRAIN", "EVAL")}
        assert rest == {k: v for k, v in s45.items() if k not in ("C1SF", "MODEL", "TRAIN", "EVAL")}
        model = {k: v for k, v in cfg["MODEL"].items() if k not in ("XAG_ASSOC", "XAG_GNN")}
        assert model == s45["MODEL"]
        changed = {k for k in set(cfg["TRAIN"]) | set(s45["TRAIN"]) if cfg["TRAIN"].get(k) != s45["TRAIN"].get(k)}
        assert changed <= {"RUN_NAME", "OUTPUT_DIR", "MAX_STEPS", "VAL_CHECK_INTERVAL"}
        assert (phase == "preflight") == (cfg["TRAIN"]["MAX_STEPS"] == 50)
