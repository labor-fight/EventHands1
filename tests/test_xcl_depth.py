"""The A3 depth sidecar follows immutable event rows and fails closed."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from semkine.events import EventPacket, collate_packets
from semkine.xag_sources import (DEPTH_VERSION, SOURCE_FIELDS, SOURCE_VERSION, depth_manifest,
                                 verify_depth_manifest, load_sources)


def cache(root):
    base = root / "seq"
    base.mkdir()
    source = {"version": SOURCE_VERSION, "seq": "seq", "n_events": 4, "n_ms": 3, "complete": True}
    (base / "metadata.json").write_text(json.dumps(source))
    arrays = {"part": np.array([2, -1, 3, 4], np.int8), "face": np.array([10, -1, 11, 12], np.int16),
              "bary": np.array([[.2, .3, .5], [0, 0, 0], [.1, .7, .2], [.8, .1, .1]], np.float32),
              "offsets": np.array([0, 2, 2, 4], np.int64), "done_ms": np.ones(3, bool),
              "depth": np.array([.5, np.nan, .7, .8], np.float32), "depth_done_ms": np.ones(3, bool)}
    for name, array in arrays.items():
        np.save(base / (name + ".npy"), array)
    meta = {"version": DEPTH_VERSION, "seq": "seq", "source_version": SOURCE_VERSION,
            "n_events": 4, "n_ms": 3, "complete": True,
            "outputs_sha256": {name: hashlib.sha256((base / name).read_bytes()).hexdigest()
                               for name in ("depth.npy", "depth_done_ms.npy")},
            "source_metadata_sha256": hashlib.sha256((base / "metadata.json").read_bytes()).hexdigest()}
    (base / "depth_metadata.json").write_text(json.dumps(meta))
    return base, meta


def test_optional_depth_keeps_legacy_output_and_source_identity(tmp_path):
    cache(tmp_path)
    args = (tmp_path, "seq", [3, -1, 1, 0, 3], [2, 99, 0, 0, 2])
    before = load_sources(*args)
    after = load_sources(*args, require_depth=True)
    assert set(before) == set(SOURCE_FIELDS)
    assert set(after) == set(SOURCE_FIELDS) | {"xag_depth"}
    for key in SOURCE_FIELDS:
        np.testing.assert_array_equal(before[key], after[key])
    np.testing.assert_array_equal(after["xag_depth"], np.array([.8, np.nan, np.nan, .5, .8], np.float32))


@pytest.mark.parametrize("damage", ["missing", "version", "source_identity", "partial", "done", "shape", "dtype", "nonpositive", "unknown", "positive_corruption"])
def test_sidecar_refuses_missing_stale_unfinished_and_corrupt(tmp_path, damage):
    base, meta = cache(tmp_path)
    if damage == "missing":
        (base / "depth.npy").unlink()
    elif damage in ("version", "source_identity", "partial"):
        if damage == "version":
            meta["version"] = "old"
        elif damage == "source_identity":
            meta["source_metadata_sha256"] = "wrong"
        else:
            meta["complete"] = False
        (base / "depth_metadata.json").write_text(json.dumps(meta))
    elif damage == "done":
        np.save(base / "depth_done_ms.npy", np.array([False, True, True]))
    elif damage == "shape":
        np.save(base / "depth.npy", np.ones(3, np.float32))
    elif damage == "dtype":
        np.save(base / "depth.npy", np.ones(4, np.float64))
    else:
        depth = np.array([.5, np.nan, .7, .8], np.float32)
        depth[1 if damage == "unknown" else 0] = .123 if damage == "positive_corruption" else 0.0
        np.save(base / "depth.npy", depth)
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        load_sources(tmp_path, "seq", [0, 1], [0, 0], require_depth=True)
    # Old A0/A1/A2 source readers never touch the optional sidecar.
    assert "xag_depth" not in load_sources(tmp_path, "seq", [0], [0])


def packet(depth):
    fields = load_sources("/absent", "none", [-1, -1], [0, 0], require_depth=depth)
    return EventPacket(events=np.zeros((2, 4), np.float32), sequence_id=0, t_start_us=0, t_end_us=1000,
                       is_sequence_start=False, is_sequence_end=False, target=np.zeros(51, np.float32),
                       prev_state=np.zeros(51, np.float32), betas=np.zeros(10, np.float32),
                       camera_K=np.eye(3, dtype=np.float32), **fields)


def test_depth_collates_and_moves_with_event_rows():
    batch = collate_packets([packet(True), packet(True)]).to("cpu")
    batch.validate()
    assert batch.xag_depth.shape == (4,) and batch.xag_depth.dtype == torch.float32
    assert torch.isnan(batch.xag_depth).all()
    assert collate_packets([packet(False)]).xag_depth is None
    with pytest.raises(ValueError, match="partial XCL"):
        collate_packets([packet(True), packet(False)])


def test_full_oracle_dataset_requires_source_root():
    from semkine.dataset import SemKineDataset
    with pytest.raises(ValueError, match="full_oracle requires"):
        SemKineDataset(root="/unused", sequences=[], components=np.zeros((45, 45)),
                       mano_npz="unused", input_mode="raw_packed", xag_require_depth=True)


def test_acceptance_manifest_binds_all_requested_sidecar_files(tmp_path):
    base, _ = cache(tmp_path)
    manifest = depth_manifest(tmp_path, ["seq"])
    assert verify_depth_manifest(manifest, ["seq"])
    with pytest.raises(ValueError, match="fixed split"):
        verify_depth_manifest(manifest, ["seq", "other"])
    with (base / "depth.npy").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="array changed"):
        verify_depth_manifest(manifest, ["seq"])


@pytest.mark.parametrize("unroll", [False, True])
def test_depth_follows_real_augmentation_and_sort_without_changing_inputs(monkeypatch, unroll):
    from semkine import domrand as DR
    from semkine.dataset import SemKineDataset
    root, mano = ROOT / "data" / "hand_data51", ROOT / "assets" / "mano_right.npz"
    if not mano.exists() or not (root / "val" / "zgz_local_events.npy").exists():
        pytest.skip("requires real MANO and sequence fixture")
    offsets = np.load(root / "val" / "zgz_local_offsets.npy")

    def fake_sources(source_root, seq, rows, ms, *, require_depth=False):
        real = rows >= 0
        assert np.all((rows[real] >= offsets[ms[real]]) & (rows[real] < offsets[ms[real] + 1]))
        result = load_sources("unused", "unused", np.full(len(rows), -1), ms, require_depth=require_depth)
        result["xag_source_index"] = rows.copy()
        result["xag_part"][real], result["xag_face"][real] = 2, 10
        result["xag_bary"][real] = [.2, .3, .5]
        result["xag_valid"][real] = True
        if require_depth:
            result["xag_depth"][real] = .5 + (rows[real] % 100).astype(np.float32) / 1000
        return result

    monkeypatch.setattr("semkine.xag_sources.load_sources", fake_sources)
    common = dict(root=root, sequences=[("zgz_local", "val")], components=np.load(mano)["hands_components"],
                  mano_npz=str(mano), input_mode="raw_packed", train=True, seed=3407,
                  speed_aug=True, polarity_flip=True, pixel_polarity_swap=True,
                  prev_noise=(.005, .05, .05), prev_noise_mode="mixed",
                  domrand=DR.DomRandConfig(enabled=True), unroll_pair=unroll)
    reference = SemKineDataset(**common)
    oracle = SemKineDataset(**common, xag_source_root="unused", xag_require_depth=True)
    for index in [0, 143, len(reference) // 2, len(reference) - 1]:
        a, b = reference[index], oracle[index]
        for ref, got in zip(a, b) if unroll else [(a, b)]:
            for key in ("events", "target", "prev_state", "betas", "camera_K"):
                np.testing.assert_array_equal(getattr(ref, key), getattr(got, key))
            real = got.xag_source_index >= 0
            np.testing.assert_array_equal(got.xag_depth[real],
                                          .5 + (got.xag_source_index[real] % 100).astype(np.float32) / 1000)
            assert np.isnan(got.xag_depth[~real]).all()
            collate_packets([got]).validate()


def test_eval_attaches_depth_only_for_full_oracle(monkeypatch):
    from semkine.eval_track import attach_xag_sources
    seen = []

    def fake_sources(root, seq, rows, ms, **kwargs):
        seen.append(kwargs)
        return load_sources("unused", "unused", [-1] * len(rows), ms, **kwargs)

    monkeypatch.setattr("semkine.xag_sources.load_sources", fake_sources)
    batch = collate_packets([packet(False)])
    offsets = np.array([0, 2], np.int64)
    cfg = {"DATA": {"XAG_SOURCE_ROOT": "unused"}, "MODEL": {"XAG_ASSOC": "full_oracle"}}
    attach_xag_sources(batch, cfg, "seq", offsets, 0, 1, "cpu")
    assert seen == [{"require_depth": True}]
    assert batch.xag_depth.shape == (2,)
    cfg["MODEL"]["XAG_ASSOC"] = "surface_oracle"
    attach_xag_sources(batch, cfg, "seq", offsets, 0, 1, "cpu")
    assert seen[-1] == {}
