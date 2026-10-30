#!/usr/bin/env python3
"""Frozen-weight heatmap temperature readout candidates, all W200 D queries."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]
import numpy as np
import torch
from mano_layer import ManoLayer
from tools.xkf_repair.evaluate import setup
from tools.xkf import debug_eval as E
from tools.xkf import direct_probe as P


def write(path, value):
    Path(path).write_text(json.dumps(E.safe_json(value), indent=2, allow_nan=False) + '\n')


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, choices=[3407, 3408, 3409], required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    setup()
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    cfgpath = ROOT / f'configs/xkf_repair/repair_s{args.seed}.yaml'
    ckpt = ROOT / f'outputs/xkf/repair_20261015_r2/repair_s{args.seed}/xkf_repair_s{args.seed}-step=2200.ckpt'
    frozen = json.loads((ROOT / 'outputs/xkf/repair_20261015_r2/frozen_manifest.json').read_text())
    for name, digest in frozen['source_sha256'].items():
        assert E.sha(ROOT / name) == digest, name
    model, cfg, identity = E.load_model(str(cfgpath), str(ckpt), device)
    for module in model.modules():
        if isinstance(module, ManoLayer):
            module.fast_chain = True
    bank, banksha = P.D.load_bank()
    declaration = {'kind': 'R3_D_TEMPERATURE_READOUT_PROBE', 'formal': False,
                   'identity': identity, 'bank_sha256': banksha,
                   'multipliers': [1., 1.25, 1.5, 2.], 'windows_ms': [200],
                   'scope': 'all1204local+1386global+128eachtrainingcontrols; no optimizer',
                   'selection': 'fixed four multipliers, everyseed; then actual own-loop if viable',
                   'source_sha256': {**frozen['source_sha256'], str(Path(__file__).relative_to(ROOT)): E.sha(__file__)},
                   'inference': 'zero target, no use of previous state by D; TF32off'}
    write(out / 'protocol.json', declaration)
    P.WINDOWS = (200,)
    options = argparse.Namespace(batch_size=32, control_rows=128, limit_rows=0)
    original = model.direct.log_beta.detach().clone()
    results = {}
    started = time.time()
    for multiplier in declaration['multipliers']:
        tag = str(multiplier)
        directory = out / f'temp_{tag}'
        directory.mkdir()
        model.direct.log_beta.copy_(original + math.log(multiplier))
        value = {'multiplier': multiplier, 'sequences': {}, 'failures': [], 'raw_files': []}
        for name in ('zgz_local', 'zgz_global', 'lyq_local', 'ch_global'):
            arrays, summaries, failures = P.sequence(model, cfg, name, bank['sequences'][name], device, options)
            for label, raw in arrays.items():
                path = directory / f'{label}.npz'
                np.savez_compressed(path, **raw)
                value['raw_files'].append({'path': str(path), 'sha256': E.sha(path)})
            value['sequences'][name] = summaries
            value['failures'].extend(failures)
        value['complete'] = not value['failures']
        write(directory / 'result.json', value)
        results[tag] = value
        print(json.dumps({'seed': args.seed, 'multiplier': multiplier, 'failures': len(value['failures']),
                          'local_t_rmse_mm': value['sequences']['zgz_local']['200']['all']['variants']['learned__predpose_pred2d']['t_rmse_mm']}), flush=True)
    assert E.sha(__file__) == declaration['source_sha256'][str(Path(__file__).relative_to(ROOT))]
    write(out / 'result.json', {'identity': identity, 'candidates': results,
                              'elapsed_seconds': time.time() - started, 'formal': False})


if __name__ == '__main__':
    main()
