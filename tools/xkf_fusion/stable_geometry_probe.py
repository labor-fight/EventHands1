#!/usr/bin/env python3
"""Known-failure GPU validation of true per-sample geometry, without padding fixes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]
import numpy as np
import torch
from semkine.xkf_stable_geometry import XKFStableGeometryModel
from tools.xkf_fusion import batch_probe as B


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def trace_for_rows(model, count, copies):
    if model.stability_mode in ('per_sample_i', 'per_sample_model') and count > 1:
        traces = model.stability_last_rows[::copies]
    else:
        aux = model.xyz_last
        traces = [{key: (value[index:index + 1] if key != 'assoc' else
                   {field: tensor[index:index + 1] for field, tensor in value.items()})
                   for key, value in aux.items()} for index in range(0, count, copies)]
    return {key: torch.cat([t['verts'] if key == 'verts' else t['assoc'][key] for t in traces]).cpu().numpy()
            for key in ('verts', 'vid', 'A', 'q', 'z', 'hit', 'valid')}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', required=True, type=int, choices=(3407, 3408))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--limit-loop-steps', type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    device = torch.device(args.device)
    B.Q.XKFFusionModel = XKFStableGeometryModel
    model, cfg, identity = B.load(args.seed, {}, device)
    model.xyz_keep_aux = True
    bank, banksha = B.Q.R.D.load_bank()
    index, decisive = (0, 4) if args.seed == 3408 else (86, 1)
    clip = B.Q.R.schedule(bank, 10)[0][index]
    meta = bank['sequences'][clip['sequence']]
    data = B.Q.R.D.load_data(cfg, clip['sequence'], meta)
    cache = B.Q.PacketCache(model, data, device)
    parent = ROOT / f'outputs/xkf/fusion_20261015_r3/bn/batch_gpu_failure_s{args.seed}/rejected_clip_raw.npz'
    with np.load(parent, allow_pickle=False) as value:
        old = {key: value[key].copy() for key in value.files}
    source = B.Q.sources() | {name: B.Q.E.sha(ROOT / name) for name in (
        'semkine/xkf_stable_geometry.py', 'tools/xkf_fusion/stable_geometry_probe.py',
        'tools/xkf_fusion/batch_probe.py', 'semkine/xkf_centroid.py')}
    parameter_hash = B.parameter_digest(model)
    protocol = {'formal': False, 'automatic_full_training': False, 'identity': identity,
        'clip': clip, 'fixed_equal_input_step': decisive, 'bank_sha256': banksha,
        'reference_raw_sha256': B.Q.E.sha(parent), 'source_sha256': source,
        'layouts': [1, 9, 18, 27], 'mode_scope': 'per_sample_geometry changes only I geometry; per_sample_i changes whole I; per_sample_model also uses uncached single-row D/fusion',
        'thresholds_unchanged': True, 'rounding': False, 'partial_loop_smoke': bool(args.limit_loop_steps)}
    B.write(args.output / 'protocol.json', protocol)
    row = meta['rows'][clip['row_indices'][decisive]]
    incoming = torch.from_numpy(old['predictions'][decisive - 1, :, 2]).to(device)
    static = {}
    for mode in ('off', 'per_sample_geometry', 'per_sample_i'):
        model.configure_stability(mode)
        values, traces, times = {}, {}, {}
        for layout in (1, 9, 18, 27):
            sync(device); started = time.perf_counter()
            if layout == 1:
                predictions, parts = [], []
                for ci in range(9):
                    packet, _ = cache.packet(row, incoming[ci:ci + 1], [1.])
                    predictions.append(model.incremental_prediction(packet))
                    parts.append(trace_for_rows(model, 1, 1))
                values[layout] = torch.cat(predictions).cpu().numpy()
                traces[layout] = {key: np.concatenate([value[key] for value in parts]) for key in parts[0]}
            else:
                copies = layout // 9
                prev = incoming[:, None].expand(-1, copies, -1).reshape(-1, 51)
                packet, _ = cache.packet(row, prev, [1.])
                values[layout] = model.incremental_prediction(packet).reshape(9, copies, 51)[:, 0].cpu().numpy()
                traces[layout] = trace_for_rows(model, layout, copies)
            sync(device); times[layout] = time.perf_counter() - started
        comparisons = {}
        for layout in (9, 18, 27):
            a, b = traces[1], traces[layout]
            comparisons[str(layout)] = {**B.difference(values[1], values[layout]),
                'incremental_prediction_bitwise_equal': bool(np.array_equal(values[1], values[layout])),
                'all_geometry_bitwise_equal': all(np.array_equal(a[key], b[key]) for key in a),
                'vertex_max_mm': float(np.abs(a['verts'] - b['verts']).max() * 1000),
                'association_vid_disagreements': int(np.any(a['vid'] != b['vid'], -1).sum()),
                'association_hit_disagreements': int((a['hit'] != b['hit']).sum())}
        static[mode] = {'versus_actual_batch1': comparisons, 'elapsed_seconds_by_layout': times}
    model.xyz_keep_aux = False
    initial = torch.from_numpy(old['initial']).to(device)
    rows = [meta['rows'][i] for i in clip['row_indices']]
    if args.limit_loop_steps:
        rows = rows[:args.limit_loop_steps]
    loops = {}
    saved = {}
    for mode in ('per_sample_i', 'per_sample_model'):
        model.configure_stability(mode).configure_fusion(None)
        model.kmode = 'fused'
        predictions, elapsed = {}, {}
        for layout in (1, 9):
            previous = initial.clone()
            trajectory, duration = [], 0.
            B.Q.R.D.reset(model)
            for row in rows:
                if layout == 1:
                    current = []
                    for ci in range(9):
                        packet, _ = cache.packet(row, previous[ci:ci + 1], [1.])
                        sync(device); start = time.perf_counter()
                        current.append(model.forward_packet(packet))
                        sync(device); duration += time.perf_counter() - start
                    previous = torch.cat(current)
                else:
                    packet, _ = cache.packet(row, previous, [1.])
                    sync(device); start = time.perf_counter()
                    previous = model.forward_packet(packet)
                    sync(device); duration += time.perf_counter() - start
                trajectory.append(previous.cpu().numpy())
            predictions[layout] = np.stack(trajectory)
            elapsed[str(layout)] = duration
            saved[f'{mode}_batch{layout}'] = predictions[layout]
        loops[mode] = {**B.difference(predictions[1], predictions[9]),
            'bitwise_equal_all_steps_conditions': bool(np.array_equal(predictions[1], predictions[9])),
            'steps': len(rows), 'conditions': 9, 'actual_forward_seconds': elapsed,
            'batch1_mean_ms_per_state': elapsed['1'] / (len(rows) * 9) * 1000,
            'actual_direct_network_every_forward': True, 'no_direct_prediction_cache_used_by_forward': True}
    if B.parameter_digest(model) != parameter_hash:
        raise ValueError('stable geometry probe changed parameters')
    for name, digest in source.items():
        if B.Q.E.sha(ROOT / name) != digest:
            raise ValueError('source changed during probe')
    np.savez_compressed(args.output / 'actual_forward_trajectories.npz', **saved)
    B.write(args.output / 'result.json', {'formal': False, 'automatic_full_training': False,
        'identity': identity, 'clip': clip, 'equal_input_static': static, 'own_loop': loops,
        'parameter_sha256_unchanged': parameter_hash, 'partial_loop_smoke': bool(args.limit_loop_steps),
        'protocol_sha256': B.Q.E.sha(args.output / 'protocol.json'),
        'scope': 'known-failure numerical validation only, not112-clip or formal recovery acceptance'})


if __name__ == '__main__':
    main()
