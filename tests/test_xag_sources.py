"""XAG source provenance, old-label parity, and unchanged augmentation RNG tests."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

from semkine import domrand as DR
from semkine.dataset import SemKineDataset
from semkine.events import EventPacket, collate_packets
from semkine.xag_sources import (SOURCE_FIELDS, SOURCE_VERSION, load_sources,
                                 source_maps, source_barycentrics)


def xa_label_reference(h, w):
    # Read the real legacy implementation without its process-wide CUDA env setup.
    source = ROOT / "tools" / "xcl" / "sources" / "step1_source_labels.py"
    tree = ast.parse(source.read_text())
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("_shift", "label_maps")]
    namespace = {"np": np, "H": h, "W": w, "DEPTH_JUMP": 0.01,
                 "_OFFS": sorted([(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3)
                                  if dx * dx + dy * dy <= 4], key=lambda o: (o[0] ** 2 + o[1] ** 2, o))}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["label_maps"]


@pytest.mark.parametrize("seed", [3, 18, 91])
def test_extended_sources_keep_exact_xa_part_kind_depth(seed):
    rng = np.random.default_rng(seed)
    Z = rng.uniform(0.4, 0.8, (19, 23))
    Z[rng.random(Z.shape) < 0.25] = np.inf
    P = rng.integers(0, 16, Z.shape, dtype=np.int8)
    F = rng.integers(0, 1538, Z.shape, dtype=np.int16)
    P[~np.isfinite(Z)], F[~np.isfinite(Z)] = -1, -1
    ref = xa_label_reference(*Z.shape)(Z, P)
    got = source_maps(Z, P, F)
    for name, expected in zip(("part", "kind", "depth"), ref[:3]):
        assert np.array_equal(got[name], expected, equal_nan=True), name
    valid = got["face"] >= 0
    src = got["pixel"][valid]
    assert np.array_equal(got["face"][valid], F.reshape(-1)[src])
    assert np.array_equal(got["part"][valid], P.reshape(-1)[src])
    assert np.array_equal(got["depth"][valid], Z.reshape(-1)[src].astype(np.float32))


def test_occluding_contour_uses_foreground_face_and_pixel():
    Z = np.full((13, 15), np.inf)
    P = np.full(Z.shape, -1, np.int8)
    F = np.full(Z.shape, -1, np.int16)
    Z[2:11, 2:13], P[2:11, 2:13], F[2:11, 2:13] = 0.7, 0, 10
    Z[4:9, 6:9], P[4:9, 6:9], F[4:9, 6:9] = 0.5, 7, 20
    got = source_maps(Z, P, F)
    assert got["part"][6, 5] == 7 and got["face"][6, 5] == 20
    assert got["pixel"][6, 5] == 6 * Z.shape[1] + 6
    assert got["kind"][6, 5] == 2
    assert got["face"][0, 14] == -1


def test_perspective_bary_reconstructs_source_ray_not_event_ray():
    verts = np.array([[0., 0., 1.], [2., 0., 2.], [0., 3., 3.]])
    faces = np.array([[0, 1, 2]])
    # Projection vertices (0,0),(4,0),(0,4); source pixel (1,1).
    bary = source_barycentrics(verts, faces, (4, 4, 0, 0), np.array([0, -1]), np.array([11, -1]), 10)
    point = bary[0] @ verts
    np.testing.assert_allclose(point[:2] / point[2], [0.25, 0.25], atol=1e-7)
    np.testing.assert_allclose(bary[0].sum(), 1.0)
    assert np.array_equal(bary[1], np.zeros(3))
    assert not np.allclose(bary[0], [0.5, 0.25, 0.25])


def make_cache(root):
    folder = root / "seq"
    folder.mkdir()
    (folder / "metadata.json").write_text(json.dumps({"version": SOURCE_VERSION, "seq": "seq", "n_events": 4}))
    for name, array in {"part": np.array([2, -1, 3, 4], np.int8),
                        "face": np.array([10, -1, 11, 12], np.int16),
                        "bary": np.array([[.2, .3, .5], [0, 0, 0], [.1, .7, .2], [.8, .1, .1]], np.float32),
                        "offsets": np.array([0, 2, 2, 4], np.int64),
                        "done_ms": np.array([True, True, False])}.items():
        np.save(folder / (name + ".npy"), array)
    return folder


def test_lookup_preserves_order_repeats_hot_and_unknown(tmp_path):
    make_cache(tmp_path)
    out = load_sources(tmp_path, "seq", [1, -1, 0, 0], [0, 2, 0, 0])
    assert out["xag_part"].tolist() == [-1, -1, 2, 2]
    assert out["xag_valid"].tolist() == [False, False, True, True]
    assert out["xag_source_index"].tolist() == [1, -1, 0, 0]
    assert out["xag_source_ms"].tolist() == [0, 2, 0, 0]
    assert out["xag_bary"].dtype == np.float32


def test_lookup_refuses_wrong_time_and_incomplete_cache(tmp_path):
    make_cache(tmp_path)
    with pytest.raises(ValueError, match="mismatch"):
        load_sources(tmp_path, "seq", [0], [2])
    with pytest.raises(RuntimeError, match="incomplete"):
        load_sources(tmp_path, "seq", [2], [2])
    with pytest.raises(IndexError):
        load_sources(tmp_path, "seq", [4], [0])
    with pytest.raises(ValueError):
        load_sources(tmp_path, "seq", [-2], [0])


def test_empty_and_hot_only_lookup_needs_no_cache(tmp_path):
    assert load_sources(tmp_path, "absent", [], [])["xag_bary"].shape == (0, 3)
    out = load_sources(tmp_path, "absent", [-1, -1], [10, 11])
    assert not out["xag_valid"].any()


def packet(n, with_sources=True):
    fields = load_sources("/not/accessed", "empty", np.full(n, -1), np.zeros(n, np.int64)) if with_sources else {}
    return EventPacket(events=np.zeros((n, 4), np.float32), sequence_id=0, t_start_us=0, t_end_us=50000,
                       is_sequence_start=False, is_sequence_end=False, target=np.zeros(51, np.float32),
                       prev_state=np.zeros(51, np.float32), betas=np.zeros(10, np.float32),
                       camera_K=np.eye(3, dtype=np.float32), **fields)


def test_source_collation_device_move_empty_packet_and_xkf_field():
    items = [packet(3), packet(0), packet(2)]
    for item in items:
        item.lnes_window_ms = 200
    batch = collate_packets(items).to("cpu")
    batch.validate()
    assert batch.counts.tolist() == [3, 0, 2]
    assert batch.xag_bary.shape == (5, 3)
    assert batch.xag_source_index.tolist() == [-1] * 5
    assert batch.lnes_window_ms.tolist() == [200] * 3
    assert all(getattr(collate_packets([packet(1, False)]), key) is None for key in SOURCE_FIELDS)
    with pytest.raises(ValueError, match="missing or partial"):
        collate_packets([packet(1), packet(1, False)])


@pytest.mark.parametrize("n,hot,keep", [(200, 30, .4), (0, 5, 1.), (200, 0, 0.), (200, 0, 1.)])
def test_domrand_source_tracking_preserves_events_and_rng(n, hot, keep):
    ds = SemKineDataset.__new__(SemKineDataset)
    ds.render_scale, ds.width, ds.height = .375, 240, 180
    rng = np.random.default_rng(23)
    original = [rng.integers(0, 240, n), rng.integers(0, 180, n), rng.integers(0, 2, n),
                np.sort(rng.integers(0, 50000, n))]
    dr = DR.DomRandSample(.1, 1.1, (13., -7.), keep, hot)
    K = np.array([[600., 0, 320], [0, 600, 240], [0, 0, 1]], np.float32)
    a, b = np.random.default_rng(91), np.random.default_rng(91)
    ref = ds._domrand_events(*(x.copy() for x in original), 50, dr, K, a)
    out = ds._domrand_events(*(x.copy() for x in original), 50, dr, K, b, source_rows=np.arange(n))
    for x, y in zip(ref, out[:4]):
        assert np.array_equal(x, y)
    assert a.bit_generator.state == b.bit_generator.state
    rows = out[4]
    real = rows >= 0
    assert int((~real).sum()) == hot
    assert np.array_equal(out[3][real], original[3][rows[real]])


@pytest.mark.parametrize("unroll", [False, True])
def test_real_training_samples_sources_on_off_are_bitwise_equal(monkeypatch, unroll):
    root, mano = ROOT / "data" / "hand_data51", ROOT / "assets" / "mano_right.npz"
    if not mano.exists() or not (root / "val" / "zgz_local_events.npy").exists():
        pytest.skip("requires repository MANO and sequence fixture")
    offsets = np.load(root / "val" / "zgz_local_offsets.npy")
    def fake_sources(cache, seq, rows, ms):
        real = rows >= 0
        assert np.all(rows[real] >= offsets[ms[real]])
        assert np.all(rows[real] < offsets[ms[real] + 1])
        result = load_sources("unused", "unused", np.full(len(rows), -1), ms)
        result["xag_source_index"] = rows.copy()
        result["xag_part"][real], result["xag_face"][real] = rows[real] % 16, rows[real] % 1538
        result["xag_bary"][real] = [.25, .25, .5]
        result["xag_valid"][real] = True
        return result
    monkeypatch.setattr("semkine.xag_sources.load_sources", fake_sources)
    common = dict(root=root, sequences=[("zgz_local", "val")], components=np.load(mano)["hands_components"],
                  mano_npz=str(mano), input_mode="raw_packed", train=True, seed=3407, speed_aug=True,
                  polarity_flip=True, pixel_polarity_swap=True, prev_noise=(.005, .05, .05), prev_noise_mode="mixed",
                  domrand=DR.DomRandConfig(enabled=True), unroll_pair=unroll)
    baseline, oracle = SemKineDataset(**common), SemKineDataset(**common, xag_source_root="unused")
    for idx in [0, 13, 143, len(baseline) // 2, len(baseline) - 1]:
        ref, got = baseline[idx], oracle[idx]
        pairs = [(ref, got)] if not unroll else list(zip(ref, got))
        for a, b in pairs:
            for name in ("events", "prev_state", "target", "betas", "camera_K"):
                assert np.array_equal(getattr(a, name), getattr(b, name)), (idx, name)
            assert (a.t_start_us, a.t_end_us) == (b.t_start_us, b.t_end_us)
            assert len(b.xag_source_index) == b.n_events
            collate_packets([b]).validate()
