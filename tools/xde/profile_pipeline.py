#!/usr/bin/env python3
"""Frozen Q1 own-loop software pipeline, including host-ready mesh output.

This is diagnostic timing, not a replacement for the historical main row.
No sensor acquisition, real-time queue, or display is simulated.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'model'))
import numpy as np
import torch
from mano_layer import ManoLayer
from tools.xdl import diagnostics as D


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def stats(a):
    a = np.asarray(a, np.float64)
    assert len(a) and np.isfinite(a).all()
    return {'n': len(a), 'mean': float(a.mean()),
            **{f'p{q}': float(np.percentile(a, q)) for q in (50, 95, 99)},
            'max': float(a.max())}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--output', default='outputs/xde/pipeline')
    args = ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '7':
        raise RuntimeError('This root-owned job is allocated physical GPU7 only')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device('cuda:0')
    out = Path(args.output)
    bank, banksha = D.load_bank()
    names = ['zgz_global', 'zgz_local']
    script_sha = sha(__file__)
    protocol = {'study': 'XDE-PIPELINE-20261014', 'gpu_physical': 7,
        'run': 'xdl_q1_s3407', 'names': names, 'batch_size': 1,
        'initial': 'GT at start of every valid run; then own prediction',
        'window_ms': 50, 'query_clock': 'unchanged packet_bank rows',
        'limit_per_sequence': args.limit, 'warmup_queries': 20,
        'scope': 'software events-in-host-memory to host-ready state/joints/mesh',
        'exclusions': ['sequence file opening', 'sensor acquisition', 'real-time queue', 'display'],
        'timing': 'wall clock with CUDA synchronization at stage boundaries; includes sync overhead',
        'stages': ['slice_events', 'packet_H2D', 'full_forward', 'output_MANO', 'host_outputs'],
        'bank_sha256': banksha, 'script_sha256': script_sha,
        'torch': torch.__version__, 'numpy': np.__version__, 'python': sys.executable,
        'readout_and_model': 'unchanged; fast_chain as historical scientific diagnostics',
        'budget_gpu_hours': 0.5}
    protocol_path = out / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise RuntimeError('Existing protocol differs; use a new output directory')
    write(protocol_path, protocol)
    start_job = time.perf_counter()
    model, cfg, identity = D.load_model('xdl_q1_s3407', device)
    mano = ManoLayer(cfg['MANO']['NPZ'], add_mean=False).to(device).eval()
    mano.fast_chain = True
    raw = []
    sequence_results = {}
    for name in names:
        meta = bank['sequences'][name]
        data = D.load_data(cfg, name, meta)
        beta, camera = D.context(data, device)
        model.set_hand_context(beta, camera)
        rows = meta['rows'][:args.limit or None]
        for row in rows[:20]:
            ev = D.raw_events(data, row)
            p = D.make_packet(ev, data['gt'][row['previous_end_ms']][None], beta, camera, device, row)
            y = model.forward_packet(p)
            dec = D.ET.decode_to_mano_inputs(y, 'mano_full_axis_angle', mano.hands_components, mano.hands_mean)
            mano(beta, dec['global_orient'], dec['local_full_aa'], dec['transl'])
        torch.cuda.synchronize(device)
        D.reset(model)
        torch.cuda.reset_peak_memory_stats(device)
        prev_run = None
        records = []
        predictions = []
        truths = []
        incoming = []
        for i, row in enumerate(rows):
            if row['run_id'] != prev_run:
                D.reset(model)
                prev = torch.tensor(data['gt'][row['previous_end_ms']].copy(), device=device).view(1, 51)
                prev_run = row['run_id']
            incoming.append(prev.detach().cpu().numpy()[0].copy())
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            ev = D.raw_events(data, row)
            t1 = time.perf_counter()
            packet = D.make_packet(ev, np.zeros((1, 51), np.float32), beta, camera, device, row)
            packet.prev_state = prev
            torch.cuda.synchronize(device)
            t2 = time.perf_counter()
            pred = model.forward_packet(packet)
            torch.cuda.synchronize(device)
            t3 = time.perf_counter()
            dec = D.ET.decode_to_mano_inputs(pred, 'mano_full_axis_angle', mano.hands_components, mano.hands_mean)
            vertices, joints = mano(beta, dec['global_orient'], dec['local_full_aa'], dec['transl'])
            torch.cuda.synchronize(device)
            t4 = time.perf_counter()
            host_state = pred.detach().cpu().numpy().copy()
            host_vertices = vertices.detach().cpu().numpy().copy()
            host_joints = joints.detach().cpu().numpy().copy()
            prev = pred.detach().clone()
            torch.cuda.synchronize(device)
            t5 = time.perf_counter()
            if not (np.isfinite(host_state).all() and np.isfinite(host_vertices).all() and np.isfinite(host_joints).all()):
                write(out / 'failure.json', {'sequence': name, 'index': i, 'row': row, 'nonfinite': True})
                raise FloatingPointError('No finite subset timing or scoring is allowed')
            timestamps = [t0, t1, t2, t3, t4, t5]
            item = {'index': i, 'run_id': int(row['run_id']), 'end_ms': int(row['end_ms']),
                    'n_events': len(ev), 'total_ms': (t5-t0)*1000,
                    **{key+'_ms': (timestamps[k+1]-timestamps[k])*1000
                       for k, key in enumerate(protocol['stages'])}}
            records.append(item)
            predictions.append(host_state[0])
            truths.append(data['gt'][row['end_ms']].copy())
            if i == 99 or (i+1) % 500 == 0:
                print(json.dumps({'sequence': name, 'done': i+1, 'total': len(rows),
                                  'mean_total_ms': np.mean([r['total_ms'] for r in records])}), flush=True)
            if (time.perf_counter()-start_job)/3600 > protocol['budget_gpu_hours']:
                raise TimeoutError('Allocated single-GPU budget exceeded')
        seq = {'n': len(records), 'counts': stats([r['n_events'] for r in records]),
               **{key: stats([r[key] for r in records]) for key in records[0] if key.endswith('_ms') and key != 'end_ms'},
               'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
               'queries_per_compute_second': 1000/np.mean([r['total_ms'] for r in records]),
               'events_per_compute_second': sum(r['n_events'] for r in records)/(sum(r['total_ms'] for r in records)/1000)}
        mean = mano.hands_mean.cpu().numpy()
        ep, _, _ = D.endpoint(mano, np.stack(predictions), np.stack(truths), beta, device, mean)
        seq['clean_start_own_loop_metrics'] = {k: stats(v) for k, v in ep.items()}
        sequence_results[name] = seq
        raw.append({'sequence': name, 'records': records, 'initial': np.stack(incoming),
                    'prediction': np.stack(predictions), 'target': np.stack(truths), 'endpoint': ep})
        print(json.dumps({'sequence_complete': name, 'n': seq['n'], 'total_ms': seq['total_ms']}), flush=True)
    torch.save({'protocol': protocol, 'identity': identity, 'sequences': raw}, out / 'raw.pt')
    result = {'status': 'complete', 'protocol_sha256': sha(protocol_path), 'identity': identity,
              'results': sequence_results, 'raw_sha256': sha(out / 'raw.pt'),
              'gpu_hours_allocated': (time.perf_counter()-start_job)/3600,
              'full_denominator': not args.limit,
              'expected_queries': sum(len(bank['sequences'][n]['rows']) for n in names),
              'actual_queries': sum(v['n'] for v in sequence_results.values()),
              'timing_limits': protocol['exclusions'],
              'environment': {'gpu': torch.cuda.get_device_name(device), 'torch_threads': torch.get_num_threads(),
                              'tf32_matmul': torch.backends.cuda.matmul.allow_tf32}}
    write(out / 'result.json', result)
    print(json.dumps({'status': 'complete', 'output': str(out / 'result.json'),
                      'gpu_hours': result['gpu_hours_allocated']}), flush=True)


if __name__ == '__main__':
    main()
