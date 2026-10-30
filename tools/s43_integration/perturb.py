"""Fixed SELECT prior-perturbation and online/batch consistency diagnostics.

The first four SELECT lead windows are fixed by cache order. Ground truth is
used only after representations/predictions exist, by the original MANO FK
scorer. No perturbation, window, or source is selected by its result.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .data import load_clips, DEFAULT_CLIPS
from .memory import SequenceBank, batch_from_windows
from .model import load_integrated


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/s43_integration_20261027'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _window(clip, history):
    value = dict(clip['windows'][0])
    value['history'] = np.asarray(history, np.float32).copy()
    return value


def _construct(clip, history, mode):
    t0 = time.perf_counter()
    rep = batch_from_windows([_window(clip, history)], mode=mode)[0]
    return rep, time.perf_counter() - t0


def _predict(model, rep, history, beta, camera):
    before = time.perf_counter()
    value = model([rep], torch.from_numpy(np.asarray(history, np.float32)[None]),
                  torch.from_numpy(np.asarray(beta, np.float32)[None]),
                  torch.from_numpy(np.asarray(camera, np.float32)[None]))
    if not torch.isfinite(value).all():
        raise ArithmeticError('nonfinite perturbation prediction')
    return value, time.perf_counter() - before


def _score(model, prediction, target, beta):
    betas = torch.from_numpy(np.asarray(beta, np.float32)[None])
    target = torch.from_numpy(np.asarray(target, np.float32)[None])
    _, joint = model.base._fk(prediction.float(), betas)
    _, truth = model.base._fk(target.float(), betas)
    absolute = torch.linalg.vector_norm(joint - truth, dim=-1).mean() * 1000.
    aligned = torch.linalg.vector_norm((joint - joint[:, :1]) - (truth - truth[:, :1]), dim=-1).mean() * 1000.
    return {'absolute_joint_mm': float(absolute), 'root_aligned_joint_mm': float(aligned)}


def _delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return {'max_abs': float(np.max(np.abs(a - b))),
            'l2': float(np.linalg.norm((a - b).ravel())),
            'array_equal': bool(np.array_equal(a, b))}


def _render(model, history, beta, camera):
    return model.base._render_prev(torch.from_numpy(np.asarray(history, np.float32)[None]),
                torch.from_numpy(np.asarray(beta, np.float32)[None]),
                torch.from_numpy(np.asarray(camera, np.float32)[None])).cpu().numpy()


def _prefix_parity(clip, shift_mm):
    history = np.asarray(clip['history'], np.float32).copy()
    history[0] += shift_mm / 1000.
    window = _window(clip, history)
    sid = window['seq_id']
    start = int(window['start_us'])
    bank = SequenceBank(mode='br')
    bank.reset(sid, start, history, window['betas'], window['K_cam'],
               segment_id=window['segment_id'], history_time_us=window['history_time_us'])
    event = window['events']
    source = window['source_ids']
    bins = (event[:, 0] - start) // 1000
    rows = []
    started = time.perf_counter()
    for b in range(50):
        cutoff = start + (b + 1) * 1000
        take = bins == b
        bank.ingest(sid, event[take], source[take], cutoff)
        online = bank.readout(sid, cutoff)
        prefix = dict(window)
        prefix['events'] = event[bins <= b]
        prefix['source_ids'] = source[bins <= b]
        prefix['cutoff_us'] = cutoff
        prefix['watermark_us'] = cutoff
        batched = batch_from_windows([prefix], mode='br')[0]
        values = {key: _delta(online[key], batched[key]) for key in
                  ('R', 'templates', 'lnes', 'enhanced')}
        if not all(value['array_equal'] for value in values.values()):
            raise AssertionError(f'online/batch mismatch {clip["clip_id"]} shift={shift_mm} prefix={b+1}: {values}')
        if online['has_events'] != batched['has_events']:
            raise AssertionError('empty flag differs across constructors')
        if online['meta']['native_source_count'] != int(np.count_nonzero(bins <= b)):
            raise AssertionError('online source consumption not a complete prefix')
        rows.append({'prefix_ms': b + 1, 'source_count': online['meta']['native_source_count'],
                     'max_abs': {key: value['max_abs'] for key, value in values.items()},
                     'all_arrays_exact': True})
    return {'clip_id': clip['clip_id'], 'shift_mm': shift_mm,
            'prefixes': rows, 'all_50_prefixes_exact': True,
            'seconds': time.perf_counter() - started, 'memory': bank.memory_bytes(sid)}


def run(config, checkpoint, out):
    config, checkpoint, out = Path(config), Path(checkpoint), Path(out)
    if out.exists():
        raise FileExistsError(out)
    if not checkpoint.exists():
        raise FileNotFoundError('trained checkpoint must be complete before this diagnostic')
    dependency_paths = [Path(__file__), ROOT / 'tools/s43_integration/data.py',
          ROOT / 'tools/s43_integration/memory.py', ROOT / 'tools/s43_integration/model.py',
          ROOT / 'tools/s43_integration/fast_geometry.py', ROOT / 'tools/s43_integration/raster.cpp',
          ROOT / 'tools/s43_br/carrier.py', ROOT / 'model/model.py', ROOT / 'model/mano_layer.py',
          ROOT / 'assets/mano_right.npz', config, checkpoint,
          DEFAULT_CLIPS, DEFAULT_CLIPS.with_name('metadata.json'), OUT / 'protocol.json']
    before = {str(p): _sha(p) for p in dependency_paths}
    source_path = out.with_name(out.stem + '_sources.json')
    if source_path.exists():
        raise FileExistsError(source_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(json.dumps({'at': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
              'source_sha256': before, 'selection': 'first four SELECT clips, fixed before outcomes',
              'perturbations_mm': [0, 5, 15], 'parity_subset': 'first two of these clips, shifts 0 and 15mm, every 1ms'}, indent=2) + '\n')
    torch.set_num_threads(2)
    model, _ = load_integrated(config, checkpoint=checkpoint)
    model.eval()
    if model.mode != 'br':
        raise ValueError('this diagnostic expects the locked B-R integration config')
    selected = [clip for clip in load_clips() if clip['partition'] == 'validation'][:4]
    if len(selected) != 4:
        raise ValueError('registered four SELECT clips unavailable')
    results = []
    start = time.perf_counter()
    with torch.inference_mode():
        for clip in selected:
            history = np.asarray(clip['history'], np.float32)
            original, original_seconds = _construct(clip, history, 'br')
            original_render = _render(model, history, clip['betas'], clip['K_cam'])
            native_controls = {mode: _construct(clip, history, mode)[0] for mode in ('legacy', 'enhanced')}
            for shift in (0, 5, 15):
                changed = history.copy(); changed[0] += shift / 1000.
                try:
                    fresh, fresh_seconds = _construct(clip, changed, 'br')
                    changed_render = _render(model, changed, clip['betas'], clip['K_cam'])
                    controls = {mode: _construct(clip, changed, mode)[0] for mode in ('legacy', 'enhanced')}
                    invariants = {mode: _delta(native_controls[mode][key], controls[mode][key])
                                  for mode, key in [('legacy', 'lnes'), ('enhanced', 'enhanced')]}
                    if not all(v['array_equal'] for v in invariants.values()):
                        raise AssertionError('state-independent event representation changed with hand prior')
                    fresh_output, fresh_network_seconds = _predict(model, fresh, changed, clip['betas'], clip['K_cam'])
                    # Deliberate stale geometry: only the external base hand
                    # prior/render changes, while original R/templates persist.
                    stale_output, stale_network_seconds = _predict(model, original, changed, clip['betas'], clip['K_cam'])
                    fresh_score = _score(model, fresh_output, clip['targets'][0], clip['betas'])
                    stale_score = _score(model, stale_output, clip['targets'][0], clip['betas'])
                    item = {'clip_id': clip['clip_id'], 'recording': clip['recording'],
                            'shift_mm': shift, 'status': fresh['meta']['status'],
                            'source_count': fresh['meta']['native_source_count'],
                            'native_invariants': invariants,
                            'R_delta': _delta(original['R'], fresh['R']),
                            'template_delta': _delta(original['templates'], fresh['templates']),
                            'base_prev_render_delta': _delta(original_render, changed_render),
                            'fresh_vs_stale_prediction_max_abs': float((fresh_output - stale_output).abs().max()),
                            'fresh': fresh_score, 'stale': stale_score,
                            'construct_seconds': fresh_seconds, 'original_construct_seconds': original_seconds,
                            'fresh_network_seconds': fresh_network_seconds, 'stale_network_seconds': stale_network_seconds,
                            'construction_components': {key: fresh['meta'][key] for key in
                                 ('reset_seconds', 'geometry_seconds', 'template_seconds',
                                  'local_ingest_seconds', 'phase_advance_seconds', 'readout_seconds')}}
                except (ValueError, ArithmeticError) as error:
                    item = {'clip_id': clip['clip_id'], 'recording': clip['recording'],
                            'shift_mm': shift, 'status': 'unsupported', 'reason': str(error)}
                results.append(item)
                print(json.dumps({'clip': clip['clip_id'], 'shift_mm': shift, 'status': item['status']}, ensure_ascii=False), flush=True)
        consistency = []
        for clip in selected[:2]:
            for shift in (0, 15):
                consistency.append(_prefix_parity(clip, shift))
                print(json.dumps({'prefix_case': clip['clip_id'], 'shift_mm': shift,
                                  'all_50_exact': True}), flush=True)
    summary = {}
    for shift in (0, 5, 15):
        rows = [row for row in results if row['shift_mm'] == shift and row['status'] == 'ok']
        summary[str(shift)] = {'supported': len(rows), 'total': 4,
            'fresh_mean_absolute_joint_mm': float(np.mean([r['fresh']['absolute_joint_mm'] for r in rows])) if rows else None,
            'stale_mean_absolute_joint_mm': float(np.mean([r['stale']['absolute_joint_mm'] for r in rows])) if rows else None,
            'fresh_mean_root_aligned_joint_mm': float(np.mean([r['fresh']['root_aligned_joint_mm'] for r in rows])) if rows else None,
            'stale_mean_root_aligned_joint_mm': float(np.mean([r['stale']['root_aligned_joint_mm'] for r in rows])) if rows else None,
            'mean_construct_seconds': float(np.mean([r['construct_seconds'] for r in rows])) if rows else None}
    source_unchanged = {path: _sha(path) == value for path, value in before.items()}
    if not all(source_unchanged.values()):
        raise AssertionError(f'diagnostic dependency changed: {source_unchanged}')
    receipt = {'kind': 'fixed SELECT integration prior perturbation, not training or selection',
          'config': str(config), 'checkpoint': str(checkpoint), 'selected_clip_ids': [c['clip_id'] for c in selected],
          'perturbations_mm': [0, 5, 15], 'GT_role': 'only final original _fk scorer, never representation/perturbation selection',
          'history_source': 'old public-initial frozen S43 causal warmup, no GT initialization',
          'rows': results, 'summary': summary, 'prefix_consistency': consistency,
          'all_200_prefix_checks_exact': all(row['all_50_prefixes_exact'] for row in consistency),
          'source_sha256_unchanged': source_unchanged, 'seconds': time.perf_counter() - start,
          'interpretation': ['Event memory remains state independent; BR evidence/templates depend on frozen geometry condition.',
              'The same shifted prior also changes unchanged S43 prev-render/MLP; the stale control separates BR rebuilding.',
              'Refresh changes the physical c0 and conditionally restarts the finite phase model; it is not a free chart transformation.',
              'These four fixed SELECT windows do not establish a universal recovery basin or independent cross-person performance.']}
    out.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, default=_json) + '\n')
    print(out, flush=True)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); run(a.config, a.checkpoint, a.out)


if __name__ == '__main__':
    main()
