#!/usr/bin/env python3
"""Full2590 own-loop debug row for an explicit fixed XKF fusion policy.

Optional leave-run calibration is assigned once at evaluator reset. It is an
evaluation fold assignment, not a sequence-conditioned deployed policy. Latency
uses the separately declared common development refit. All input hashes and
actual predictions are saved, and old experiment sources are never modified.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]
import numpy as np
import torch
from mano_layer import ManoLayer
from semkine import eval_track as ET
from semkine.dataset import sequences_for_split, splits_manifest
from semkine.xkf_fusion import XKFFusionModel, FusionPolicy
from tools import make_s36_row as C
from tools.xdl import make_row as X
from tools.xkf import debug_eval as E, make_row as F
from tools.xkf.runtime import LongWindowEval
from tools.xkf_repair.evaluate import setup
from tools.xkf_fusion.evaluate import sources


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed', type=int, required=True, choices=[3407, 3408, 3409])
    ap.add_argument('--policy', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--run-corrections', type=Path)
    ap.add_argument('--bn-buffers', type=Path)
    ap.add_argument('--checkpoint', type=Path)
    ap.add_argument('--config', type=Path)
    ap.add_argument('--calibration-provenance', type=Path)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    setup()
    torch.set_num_threads(4)
    E.XKFModel = F.XKFModel = XKFFusionModel
    inputs = [args.policy, *([args.run_corrections] if args.run_corrections else []),
              *([args.bn_buffers] if args.bn_buffers else []),
              *([args.calibration_provenance] if args.calibration_provenance else [])]
    hashes = {str(p.resolve()): E.sha(p) for p in inputs}
    source = {**sources(), 'tools/xkf_fusion/make_row.py': E.sha(__file__)}
    if args.bn_buffers:
        source['tools/xkf_fusion/bn_probe.py'] = E.sha(ROOT / 'tools/xkf_fusion/bn_probe.py')
    policy = json.loads(args.policy.read_text())
    FusionPolicy(policy)
    calibration = json.loads(args.calibration_provenance.read_text()) if args.calibration_provenance else None
    if calibration and (calibration['seed'] != args.seed or calibration['spec'] != policy.get('direct_correction')):
        raise ValueError('calibration provenance differs from the actual policy')
    corrections = json.loads(args.run_corrections.read_text()) if args.run_corrections else None
    if corrections and corrections['seed'] != args.seed:
        raise ValueError('calibration seed mismatch')
    config = args.config or ROOT / f'configs/xkf_repair/repair_s{args.seed}.yaml'
    checkpoint = args.checkpoint or ROOT / f'outputs/xkf/repair_20261015_r2/repair_s{args.seed}/xkf_repair_s{args.seed}-step=2200.ckpt'
    device = torch.device(args.device)
    model, cfg, identity = E.load_model(config, checkpoint, device)
    step = identity['global_step']
    if identity['seed'] != args.seed or not 2200 <= step < 6000:
        raise ValueError('expected matching seed and explicit nonformal step2200..5999')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if 'xkf_direct_continuation' in saved:
        clock = saved['xkf_direct_continuation']
        if clock['i_adam_step'] != 2200 or clock['d_adam_step'] != step:
            raise ValueError('D-only continuation parameter clocks differ')
        identity['training_clocks'] = dict(clock, all_parameters_at_global_step=False)
    elif step != 2200:
        raise ValueError('this row adapter requires explicit D-only clock metadata after2200')
    del saved
    if calibration and calibration['checkpoint_sha256'] != identity['checkpoint_sha256']:
        raise ValueError('calibration checkpoint differs from evaluated model')
    if args.bn_buffers:
        from tools.xkf_fusion.bn_probe import bn_modules, apply_buffers
        saved = torch.load(args.bn_buffers, map_location='cpu', weights_only=False)
        if saved['base_checkpoint_sha256'] != identity['checkpoint_sha256']:
            raise ValueError('BN checkpoint hash mismatch')
        apply_buffers(bn_modules(model), saved['bn_buffers'])
    model.configure_fusion(policy)
    for module in model.modules():
        if isinstance(module, ManoLayer):
            module.fast_chain = True
    mano = ManoLayer(cfg['MANO']['NPZ'], add_mean=False).to(device).eval()
    root_data = Path(cfg['DATA']['ROOT'])
    protocol = {'formal': False, 'scope': 'complete2590 own-loop debug; fixed external fusion policy',
                'seeds': [args.seed], 'n_seeds': 1, 'exact_optimizer_step': step,
                'parameter_training_clocks': identity.get('training_clocks'),
                'split': 'val_core', 'step_ms': 50, 'window_ms': 50, 'direct_window_ms': 200,
                'noise_rng_seed': 0, 'noise_scale': 1., 'tf32': False,
                'policy': policy, 'input_sha256': hashes, 'source_sha256': source,
                'calibration_scope': 'leave-whole-valid-run-out' if corrections else 'fixed declared policy',
                'calibration_fit_provenance': calibration,
                'finite_subset_reduction': False}
    X.write_json(str(args.output) + '.protocol.json', protocol)
    rng, raw = np.random.default_rng(0), []
    for seq, directory in sequences_for_split(root_data, 'val_core', splits_manifest(cfg)):
        runs = F.runs_of(root_data, directory, seq)
        scored_runs = [(i, a, b) for i, (a, b) in enumerate(runs) if b - a >= 50]
        index, records, run_ids = -1, [], []
        def hook(event, prev, pred):
            nonlocal index
            if event == 'run_start':
                index += 1
                run_id = scored_runs[index][0]
                if corrections:
                    spec = copy.deepcopy(policy)
                    spec['direct_correction'] = corrections['run_specs'][seq][str(run_id)]
                    model.configure_fusion(spec)
                return None
            if not bool(torch.isfinite(pred).all()):
                raise FloatingPointError('nonfinite full-sequence prediction')
            records.append(pred.detach().cpu().numpy()[0])
            run_ids.append(scored_runs[index][0])
            return pred
        with LongWindowEval(runs, 200):
            result = ET.track_sequence(model, mano, cfg, root_data, directory, seq, 50,
                                       device, rng, 1., 1., None, window_ms=50, state_hook=hook)
        expected = 1204 if seq == 'zgz_local' else 1386
        if result is None or result['n_frames'] != expected or len(records) != expected:
            raise ValueError('incomplete full sequence')
        if index + 1 != len(scored_runs):
            raise ValueError('run assignment mismatch')
        np.savez_compressed(str(args.output) + f'.{seq}.npz', predictions=np.stack(records),
                            run_ids=np.array(run_ids))
        raw.append(result)
        X.write_json(str(args.output) + '.raw_accuracy.json', {'identity': identity,
                     'protocol': protocol, 'raw_sequences': raw, 'complete': len(raw) == 2,
                     'finite_subset_reduction': False})
    accuracy = X.accuracy_aggregate(raw)
    accuracy.update(step=step, ckpt=str(checkpoint), checkpoint_sha256=identity['checkpoint_sha256'])
    if corrections:
        latency_policy = copy.deepcopy(policy)
        latency_policy['direct_correction'] = corrections['full_development_refit_spec']
        model.configure_fusion(latency_policy)
    latency, audit = F.latency(model, cfg, device, 200)
    macs, params = F.macs(cfg, model)
    if model.fusion_policy.correction and model.fusion_policy.correction['kind'] == 'translation_affine':
        macs += 9
    audit['fixed_policy'] = model.fusion_policy.spec
    del model, mano
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    anchor = C.latency_anchor(device)
    assert source == {name: E.sha(ROOT / name) for name in source}
    assert hashes == {str(p.resolve()): E.sha(p) for p in inputs}
    X.write_json(args.output, {'per_seed': {str(args.seed): accuracy},
                 'two_seed_mean': {k: accuracy[k] for k in ('overall', 'local', 'global')},
                 'latency_ms_scaled_full1p75': latency / anchor * C.FULL_PUBLISHED_MS,
                 'latency_ms_raw': latency, 'anchor_full_raw_ms': anchor, 'latency_context_audit': audit,
                 'macs_forward_packet': macs, 'params_total': params, 'protocol': protocol,
                 'numeric_status': 'PASS', 'finite_subset_reduction': False,
                 'provenance': dict(identity, source_sha256=source)})
    print(json.dumps({'output': str(args.output), 'formal': False}), flush=True)


if __name__ == '__main__':
    main()
