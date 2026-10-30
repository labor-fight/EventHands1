"""XKF evidence identity, causal boundaries, and incremental evaluation raster contracts."""
from pathlib import Path
import struct
import sys

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'model')]

from semkine import eval_track as ET
from semkine.domrand import DomRandConfig
from semkine.events import splat_event_image
from tools.xdl.data import RawRecoveryDataset, collate_recovery, packet_hash
from tools.xkf.data import LongWindowRecoveryDataset
from tools.xkf.runtime import LongWindowEval


@pytest.fixture
def stream(tmp_path):
    n = 700
    base = tmp_path / 'train' / 'fixture_global'
    base.parent.mkdir()
    ms = np.arange(n)
    events = np.column_stack([ms % 37, ms % 29, ms % 2]).astype(np.uint8)
    np.save(str(base) + '_events.npy', events)
    np.save(str(base) + '_offsets.npy', np.arange(n + 1, dtype=np.int64))
    # Exact millisecond boundaries catch float32 seconds -> integer-ms flooring errors.
    np.save(str(base) + '_tsub.npy', np.zeros(n, dtype=np.uint16))
    runs = np.array([[10, 360], [400, 700]])
    np.savez(str(base) + '_aux.npz', valid_runs_ms=runs, category='global',
             betas=np.zeros(10, np.float32),
             camera_K=np.array([[400., 0, 120.], [0, 400., 90.], [0, 0, 1]], np.float32))
    dtype = np.dtype([('data', np.float64, 51), ('m0', np.uint8), ('m1', np.uint8)])
    labels = np.zeros(n, dtype=dtype)
    labels['data'][:, 2] = .5
    labels['m0'], labels['m1'] = 4, 13
    with open(str(base) + '.meta', 'wb') as f:
        f.write(struct.pack('<i', 51))
        labels.tofile(f)
    return tmp_path, events, np.arange(n + 1, dtype=np.int64), runs


def dataset(root, cls=LongWindowRecoveryDataset, **kwargs):
    return cls(root=root, sequences=[('fixture_global', 'train')],
               components=np.eye(45, dtype=np.float32), mano_npz=str(REPO / 'assets/mano_right.npz'),
               seed=3407, max_steps=2, **kwargs)


def decode(p):
    code = p.lnes.astype(np.float32)
    return np.where(code > 0, (code - 1) / float(p.lnes_window_ms), 0)


@pytest.mark.parametrize('window', [30, 50, 200])
def test_long_window_reuses_augmented_packets_and_keeps_hashes(stream, window):
    root, _, _, _ = stream
    kw = dict(domrand=DomRandConfig(enabled=True, keep_min=.3, keep_max=.3),
              polarity_flip=True, pixel_polarity_swap=True, prev_noise=(.005, .05, .05))
    base = dataset(root, RawRecoveryDataset, **kw)
    long = dataset(root, window_min=window, window_max=window, **kw)
    for index in (0, 7, 1025):
        a, b = base[index], long[index]
        ba, bb = collate_recovery([a]), collate_recovery([b])
        assert packet_hash(ba['main']) == packet_hash(bb['main'])
        assert all(packet_hash(p) == packet_hash(q) for p, q in zip(ba['prefix'], bb['prefix']))
        # Main windows <= 200 ms consist entirely of the exact augmented K4 clip.
        packets = b['prefix'] + [b['main']]
        ev = np.concatenate([p.events for p in packets])
        ms = np.concatenate([p.t_start_us // 1000
                             + np.rint(p.events[:, 2].astype(np.float64) * 1e6).astype(np.int64) // 1000
                             for p in packets])
        main = b['main']
        start = main.meta['end_idx'] - window + 1
        keep = ms >= start
        ref = splat_event_image(ev[keep, 0], ev[keep, 1], ev[keep, 3], ms[keep] - start,
                                window, 180, 240)
        assert np.array_equal(decode(main), ref)


@pytest.mark.parametrize('prefix_lnes', [False, True])
def test_long_window_history_is_causal_and_matches_raw_builder(stream, prefix_lnes):
    root, events, offsets, _ = stream
    ds = dataset(root, window_min=300, window_max=300, prefix_lnes=prefix_lnes)
    for index in (0, 2, 7, 1001):
        item = ds[index]
        packets = item['prefix'] + [item['main']] if prefix_lnes else [item['main']]
        for p in packets:
            end = p.meta['end_idx']
            w = min(300, end - p.meta['run_start_ms'] + 1)
            assert p.lnes_window_ms == w
            assert np.array_equal(decode(p), ET.build_lnes(events, offsets, end, w))
        if not prefix_lnes:
            assert all(p.lnes is None for p in item['prefix'])


def test_only_preclip_history_gets_additional_augmentation(stream, monkeypatch):
    root, _, _, _ = stream
    ds = dataset(root, window_min=300, window_max=300,
                 domrand=DomRandConfig(enabled=True))
    calls = []
    original = ds._raw_events

    def record(handle, end, window):
        calls.append((end, window))
        return original(handle, end, window)

    monkeypatch.setattr(ds, '_raw_events', record)
    item = ds[0]
    first = item['prefix'][0]
    assert len(calls) == 5
    assert all(window == 50 for _, window in calls[:4])
    history_end, history_span = calls[-1]
    assert history_end + 1 == first.t_start_us // 1000
    assert history_end - history_span + 1 >= first.meta['run_start_ms']


@pytest.mark.parametrize('channels', [('last',), ('last', 'count', 'first')])
def test_eval_surface_matches_reference_across_boundaries_and_seeks(stream, channels):
    _, events, offsets, runs = stream
    runtime = LongWindowEval(runs, channels=channels)
    for end in (10, 59, 109, 159, 209, 299, 359, 400, 449, 499, 699, 449, 449):
        got, w = runtime.lnes(events, offsets, end)
        assert w == min(200, end - runtime.floor(end) + 1)
        assert np.array_equal(got, ET.build_lnes(events, offsets, end, w, channels))
    changed = events.copy()
    changed[:, 0] = (changed[:, 0] + 1) % 240
    got, w = runtime.lnes(changed, offsets, 499)
    assert np.array_equal(got, ET.build_lnes(changed, offsets, 499, w, channels))


def test_eval_context_preserves_packet_events_and_restores_on_exception(stream):
    _, events, offsets, runs = stream
    original = ET._window_events, ET.make_eval_packet
    ev = ET._window_events(events, offsets, None, 59, 50)
    with pytest.raises(RuntimeError, match='test interruption'):
        with LongWindowEval(runs):
            got_ev = ET._window_events(events, offsets, None, 59, 50)
            p = ET.make_eval_packet(got_ev, torch.zeros(1, 51), torch.zeros(1, 10),
                                    torch.eye(3).unsqueeze(0), 50, 'cpu')
            assert np.array_equal(p.events.numpy(), ev)
            assert np.array_equal(p.lnes.numpy()[0], ET.build_lnes(events, offsets, 59, 50))
            assert p.lnes_window_ms.tolist() == [50]
            raise RuntimeError('test interruption')
    assert (ET._window_events, ET.make_eval_packet) == original
