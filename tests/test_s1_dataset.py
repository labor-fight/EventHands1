#!/usr/bin/env python3
"""S1 gates: legacy parity, raw-packed contract, domrand label mirroring, protocol hygiene.

The parity tests are the reason this stage is safe to build on: they establish that turning the
new dataset on with `INPUT_MODE: legacy_lnes` and `AUG.DOMRAND.ENABLED: false` cannot move the
frozen 19.2576 mm baseline, so any later difference is attributable to the change under test.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

from config import load_config                       # noqa: E402
from fastevc import HandData51Dataset                # noqa: E402
from mano_layer import ManoLayer                     # noqa: E402
from pose_repr import decode_to_mano_inputs          # noqa: E402
from semkine import domrand as DR                    # noqa: E402
from semkine import events as EV                     # noqa: E402
from semkine import protocol as PR                   # noqa: E402
from semkine.dataset import SemKineDataset, sequences_for_split, mano_root_joint  # noqa: E402

CFG = ROOT / "configs" / "eventhands_track_render51.yaml"
MANO_NPZ = str(ROOT / "assets" / "mano_right.npz")
PROBE = ("zgz_local", "val")   # small, and the low-event-rate stress case


@pytest.fixture(scope="module")
def cfg():
    return load_config(CFG)


@pytest.fixture(scope="module")
def root(cfg):
    return Path(cfg["DATA"]["ROOT"])


@pytest.fixture(scope="module")
def components():
    return np.load(MANO_NPZ)["hands_components"].astype(np.float32)


def _make(root, components, **kw):
    kw.setdefault("fixed_window", 50)
    kw.setdefault("train", False)
    kw.setdefault("speed_aug", False)
    kw.setdefault("polarity_flip", False)
    kw.setdefault("pixel_polarity_swap", False)
    return SemKineDataset(
        root=root, sequences=[PROBE], components=components, mano_npz=MANO_NPZ, **kw
    )


# ------------------------------------------------------------------ legacy parity
def test_legacy_lnes_is_bitwise_identical(root, components):
    """The whole S1 safety argument: same window, same LNES, bit for bit."""
    legacy = HandData51Dataset(
        root=str(root), seq=PROBE[0], split=PROBE[1], pose_repr="mano_full_axis_angle",
        components=components, fixed_window=50, train=False,
        speed_aug=False, polarity_flip=False, pixel_polarity_swap=False,
    )
    new = _make(root, components, input_mode="legacy_lnes")
    assert len(new) == len(legacy), (len(new), len(legacy))

    idxs = np.linspace(0, len(legacy) - 1, 40).astype(int)
    for i in idxs:
        xl, pl, yl, bl, kl = legacy[int(i)]
        xn, pn, yn, bn, kn = new[int(i)]
        assert torch.equal(xn, xl), f"LNES differs at {i}: max {(xn - xl).abs().max()}"
        assert torch.equal(pn, pl) and torch.equal(yn, yl)
        assert torch.equal(bn, bl) and torch.equal(kn, kl)


def test_sample_index_matches_legacy_runs(root, components):
    new = _make(root, components, input_mode="legacy_lnes")
    aux = np.load(str(root / PROBE[1] / PROBE[0]) + "_aux.npz", allow_pickle=True)
    runs = np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)
    for si, end, a, b in new.index:
        assert a <= end < b, "sample end outside its run"
        assert end - 50 + 1 >= a, "window reaches outside its run"
    assert set(new.index[:, 2].tolist()) <= set(runs[:, 0].tolist())


# ------------------------------------------------------------------ raw contract
def test_raw_packed_reproduces_lnes(root, components):
    """Raw events must carry everything LNES has; identical after re-splatting."""
    ds = _make(root, components, input_mode="both")
    for i in np.linspace(0, len(ds) - 1, 25).astype(int):
        pkt = ds[int(i)]
        rebuilt = EV.lnes_from_packet(pkt.events, pkt.delta_t_s, pkt.meta["window_ms"],
                                     ds.height, ds.width)
        assert np.array_equal(rebuilt, pkt.lnes), f"LNES mismatch at {i}"


def test_raw_packed_timestamps_are_microsecond_and_ordered(root, components):
    ds = _make(root, components, input_mode="raw_packed")
    seen_sub_ms = False
    for i in np.linspace(0, len(ds) - 1, 25).astype(int):
        pkt = ds[int(i)]
        if pkt.n_events == 0:
            continue
        t = pkt.events[:, 2]
        assert np.all(np.diff(t) >= -1e-9), "timestamps not non-decreasing"
        assert t.min() >= 0.0 and t.max() < pkt.delta_t_s + 1e-9
        us = np.rint(t * 1e6).astype(np.int64)
        if np.any(us % 1000 != 0):
            seen_sub_ms = True
        assert np.all(pkt.events[:, 0] < ds.width) and np.all(pkt.events[:, 0] >= 0)
        assert np.all(pkt.events[:, 1] < ds.height) and np.all(pkt.events[:, 1] >= 0)
        assert set(np.unique(pkt.events[:, 3]).tolist()) <= {0.0, 1.0}
    assert seen_sub_ms, "no sub-millisecond timestamps: tsub extraction did not take effect"


def test_raw_packed_keeps_every_event(root, components):
    """No event may be dropped on the raw path; that is the point of it."""
    ds = _make(root, components, input_mode="raw_packed")
    offsets = np.load(str(root / PROBE[1] / PROBE[0]) + "_offsets.npy")
    for i in np.linspace(0, len(ds) - 1, 25).astype(int):
        pkt = ds[int(i)]
        end = pkt.meta["end_idx"]
        w = pkt.meta["window_ms"]
        expect = int(offsets[end + 1]) - int(offsets[end - w + 1])
        assert pkt.n_events == expect, f"{pkt.n_events} != {expect} at {i}"


def test_collate_is_ragged_and_valid(root, components):
    ds = _make(root, components, input_mode="both")
    items = [ds[int(i)] for i in np.linspace(0, len(ds) - 1, 8).astype(int)]
    batch = EV.collate_packets(items)
    batch.validate()
    assert batch.batch_size == len(items)
    assert int(batch.events.shape[0]) == sum(it.n_events for it in items)
    # ragged, not padded
    assert int(batch.events.shape[0]) < batch.batch_size * max(it.n_events for it in items) + 1
    for b, it in enumerate(items):
        got = batch.packet(b).numpy()
        assert np.array_equal(got, it.events), f"packet {b} corrupted by collate"
    chk = EV.concat_check(batch)
    assert chk["n_nonmonotonic"] == 0 and chk["n_out_of_bounds"] == 0


def test_zero_event_packet_keeps_its_slot():
    """A silent window is information for S12, so it must survive collation."""
    def mk(n):
        return EV.EventPacket(
            events=np.zeros((n, 4), np.float32), sequence_id=0, t_start_us=0,
            t_end_us=50_000, is_sequence_start=False, is_sequence_end=False,
            target=np.zeros(51, np.float32), prev_state=np.zeros(51, np.float32),
            betas=np.zeros(10, np.float32), camera_K=np.eye(3, dtype=np.float32),
        )
    batch = EV.collate_packets([mk(3), mk(0), mk(5)])
    batch.validate()
    assert batch.counts.tolist() == [3, 0, 5]
    assert batch.batch_size == 3
    assert torch.allclose(batch.delta_t_s, torch.full((3,), 0.05))


# ------------------------------------------------------------------ domrand
def test_domrand_off_is_the_identity(root, components):
    off = _make(root, components, input_mode="legacy_lnes",
                domrand=DR.DomRandConfig(enabled=False))
    explicit = _make(root, components, input_mode="legacy_lnes", domrand=None)
    for i in np.linspace(0, len(off) - 1, 10).astype(int):
        a, b = off[int(i)], explicit[int(i)]
        for ta, tb in zip(a, b):
            assert torch.equal(ta, tb)


def test_domrand_disabled_for_eval_even_if_configured(root, components):
    ds = _make(root, components, input_mode="legacy_lnes", train=False,
               domrand=DR.DomRandConfig(enabled=True))
    assert not ds.domrand.enabled, "randomising the eval input makes the metric a moving target"


def test_domrand_label_mirroring_is_exact(root, components):
    """The core correctness claim: transformed labels + K' reproject onto moved events.

    Compares projections of the GT joints, since the events and the hand see the same map.
    """
    mano = ManoLayer(MANO_NPZ, add_mean=False)
    ds = _make(root, components, input_mode="legacy_lnes")
    h = ds._handle(0)
    j0 = mano_root_joint(h.betas, MANO_NPZ)
    pos = h.pos51[ds.index[len(ds) // 2][1]][None]
    s = ds.render_scale

    def project(p51, K):
        t = torch.from_numpy(np.asarray(p51, np.float32))
        dec = decode_to_mano_inputs(t, "mano_full_axis_angle", mano.hands_components,
                                    mano.hands_mean)
        _, j = mano(torch.from_numpy(h.betas)[None].expand(len(t), -1),
                    dec["global_orient"], dec["local_full_aa"], dec["transl"])
        j = j[0].detach().numpy().astype(np.float64)
        fx, fy = K[0, 0] * s, K[1, 1] * s
        cx, cy = K[0, 2] * s, K[1, 2] * s
        return np.stack([fx * j[:, 0] / j[:, 2] + cx, fy * j[:, 1] / j[:, 2] + cy], -1)

    K0 = h.camera_K.astype(np.float64)
    uv0 = project(pos, K0)
    rng = np.random.default_rng(0)
    worst_scale, worst_roll = 0.0, 0.0
    for _ in range(12):
        dr = DR.sample_params(DR.DomRandConfig(enabled=True), rng, 180 * 240 * 2 * 50)
        p_new, K_new = DR.transform_labels(pos, dr, h.camera_K, j0, s)
        uv_new = project(p_new, np.asarray(K_new, np.float64))

        # Where the pixel map says those points should land.
        cxs, cys = K0[0, 2] * s, K0[1, 2] * s
        c, sn = np.cos(dr.roll_rad), np.sin(dr.roll_rad)
        u, v = uv0[:, 0] - cxs, uv0[:, 1] - cys
        ru, rv = c * u - sn * v, sn * u + c * v
        exp = np.stack([dr.scale * ru + cxs + dr.shift_px[0],
                        dr.scale * rv + cys + dr.shift_px[1]], -1)

        err = np.max(np.linalg.norm(uv_new - exp, axis=-1))
        resid = DR.roll_pixel_residual_px(K0, dr.roll_rad, ds.width, ds.height, s)
        # Scale/shift is exact algebra; roll is exact only for fx == fy, and the residual
        # bound is computed from this camera's own fx/fy.
        assert err <= max(resid * dr.scale, 1e-3) + 1e-6, (err, resid, dr)
        worst_scale = max(worst_scale, abs(dr.scale - 1))
        worst_roll = max(worst_roll, abs(dr.roll_rad))
    assert worst_scale > 0.02 and worst_roll > 0.02, "sampler produced only near-identities"


def test_roll_residual_is_subpixel(root, components):
    ds = _make(root, components, input_mode="legacy_lnes")
    K = ds._handle(0).camera_K
    worst = max(
        DR.roll_pixel_residual_px(K, np.deg2rad(d), ds.width, ds.height, ds.render_scale)
        for d in (-15, -7.5, 7.5, 15)
    )
    assert worst < 0.5, f"fx/fy mismatch exceeds half a pixel: {worst}"


def test_domrand_changes_events_and_labels_together(root, components):
    on = _make(root, components, input_mode="legacy_lnes", train=True,
               domrand=DR.DomRandConfig(enabled=True), prev_noise=(0.0, 0.0, 0.0))
    off = _make(root, components, input_mode="legacy_lnes", train=True,
                domrand=DR.DomRandConfig(enabled=False), prev_noise=(0.0, 0.0, 0.0))
    n_diff_img = n_diff_lab = 0
    for i in np.linspace(0, len(on) - 1, 20).astype(int):
        a, b = on[int(i)], off[int(i)]
        n_diff_img += int(not torch.equal(a[0], b[0]))
        n_diff_lab += int(not torch.equal(a[2], b[2]))
    assert n_diff_img >= 18, "geometric/statistical augmentation did not reach the events"
    assert n_diff_lab >= 18, "labels were not mirrored"


def test_domrand_keep_fraction_reduces_events(root, components):
    cfg = DR.DomRandConfig(enabled=True, roll_deg=0.0, scale_min=1.0, scale_max=1.0,
                           shift_px=0.0, keep_min=0.25, keep_max=0.26, hot_pixel_rate=0.0)
    on = _make(root, components, input_mode="raw_packed", train=True, domrand=cfg,
               prev_noise=(0.0, 0.0, 0.0))
    off = _make(root, components, input_mode="raw_packed", train=True,
                domrand=DR.DomRandConfig(enabled=False), prev_noise=(0.0, 0.0, 0.0))
    ratios = []
    for i in np.linspace(0, len(on) - 1, 20).astype(int):
        n_off = off[int(i)].n_events
        if n_off > 200:
            ratios.append(on[int(i)].n_events / n_off)
    assert ratios, "probe sequence too sparse for this test"
    assert 0.20 < float(np.mean(ratios)) < 0.31, np.mean(ratios)


def test_domrand_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown AUG.DOMRAND"):
        DR.DomRandConfig.from_cfg({"AUG": {"DOMRAND": {"ENABLED": True, "ROLL_DEGREES": 5}}})


def test_domrand_is_reproducible_from_index(root, components):
    kw = dict(input_mode="legacy_lnes", train=True,
              domrand=DR.DomRandConfig(enabled=True), prev_noise=(0.005, 0.05, 0.05))
    a = _make(root, components, seed=7, **kw)
    b = _make(root, components, seed=7, **kw)
    c = _make(root, components, seed=8, **kw)
    i = len(a) // 3
    assert torch.equal(a[i][0], b[i][0]) and torch.equal(a[i][2], b[i][2])
    assert not torch.equal(a[i][0], c[i][0]), "seed does not change the stream"


# ------------------------------------------------------------------ protocol
def test_protocol_splits_are_subject_disjoint(root):
    m = PR.build_manifest(root)
    rep = m.leakage_report()
    assert rep["clean"], rep
    assert len(m.subjects["test"]) >= 3, "S1 requires at least 3 test subjects"
    assert len(m.subjects["val"]) >= 2, "S1 requires at least 2 val subjects"
    assert sum(len(v) for v in m.splits.values()) == 74


def test_protocol_keeps_recording_variants_together(root):
    m = PR.build_manifest(root)
    where = {s: k for k, v in m.splits.items() for s in v}
    for seq in where:
        base = seq.replace("_v2", "").replace("_v3", "").replace("_v4", "")
        for other, split in where.items():
            if other.startswith(base):
                assert split == where[seq], f"{seq} and {other} straddle splits"


def test_stress_sequence_is_in_test_and_flagged(root):
    m = PR.build_manifest(root)
    assert "zgz_local" in m.splits["test"]
    assert "zgz_local" in PR.STRESS_SEQUENCES
    ev = {s: m.stats[s]["events_per_50ms"] for s in m.splits["test"]}
    assert ev["zgz_local"] == min(ev.values()), "stress label should mark the sparsest sequence"


def test_all_sequences_have_microsecond_timestamps(root):
    m = PR.build_manifest(root)
    missing = [s for s, st in m.stats.items() if not st["has_subms"]]
    assert not missing, f"run tools/extract_subms.py; missing for {missing[:5]}"


def test_sequences_for_split_falls_back_to_legacy(root, tmp_path):
    got = sequences_for_split(root, "train", manifest_path=tmp_path / "nope.json")
    legacy = json.loads((root / "splits.json").read_text())["train"]["trials"]
    assert [s for s, _ in got] == legacy


# ------------------------------------------------------------------ protocol since 2026-09-05: zgz only

def test_default_split_validates_selects_and_reports_on_zgz_only(root):
    """Nine subjects train; val = val_core = test = the held-out subject zgz."""
    train = [s for s, _ in sequences_for_split(root, "train")]
    assert sorted({s.split("_")[0] for s in train}) == ["ch", "lfz", "lpc", "lr", "ly", "lyh", "lyq", "ycy", "ylf"]
    assert len(train) == 72
    for split in ("val", "val_core", "test"):
        assert sorted(s for s, _ in sequences_for_split(root, split)) == ["zgz_global", "zgz_local"], split


def test_retired_split_manifests_are_refused(root, monkeypatch):
    from semkine.dataset import ALLOW_RETIRED_SPLITS_ENV
    retired = root / "_retired_splits_semkine_5v2v3.json"
    monkeypatch.delenv(ALLOW_RETIRED_SPLITS_ENV, raising=False)
    with pytest.raises(ValueError, match="retired"):
        sequences_for_split(root, "val_core", manifest_path=retired)
    # the configs of every arm must not pin it either
    for p in (ROOT / "configs").rglob("*.yaml"):
        assert "_retired_splits" not in p.read_text(), p
    # re-scoring an archived artefact for the record stays possible, explicitly
    monkeypatch.setenv(ALLOW_RETIRED_SPLITS_ENV, "1")
    if retired.exists():
        assert [s for s, _ in sequences_for_split(root, "val_core", manifest_path=retired)]
