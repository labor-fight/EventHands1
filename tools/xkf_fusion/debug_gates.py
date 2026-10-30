#!/usr/bin/env python3
"""Apply unchanged G1--G4 formulas to an explicit debug checkpoint and policy.

This never produces a formal verdict or training authorization. The frozen
step6000 tool remains untouched; this adapter records the actual parameter
clocks, all2590 one-step packets and all65 recovery starts, including short runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]
import numpy as np
import torch
from mano_layer import ManoLayer
from semkine.dataset import splits_manifest
from semkine.xkf_centroid import XKFCentroidModel, CentroidFusionPolicy
from semkine.xkf_fusion import correct_direct
from tools.xkf import debug_eval as E, diagnostics as G
from tools.xkf_fusion.evaluate import sources
from tools.xkf_repair.evaluate import setup
from tools.xkf_fusion.train_calibration_tuned import validate_clocks
from tools.xdl.make_row import write_json, accuracy_aggregate


def read_debug_row(path, identity, policy):
    result = {'values': {}, 'issues': [], 'identities': []}
    if path is None:
        result['issues'].append({'reason': 'no complete debug main row supplied'})
        return result
    row = json.loads(path.read_text())
    p = row['protocol']
    seed = str(identity['seed'])
    accuracy = row['per_seed'][seed]
    if (p['formal'] is not False or p['exact_optimizer_step'] != identity['global_step']
            or accuracy['checkpoint_sha256'] != identity['checkpoint_sha256']
            or p['policy'] != policy or row['numeric_status'] != 'PASS'
            or row['finite_subset_reduction'] is not False or accuracy['step'] != 2400
            or accuracy['n_frames'] != 2590 or p['step_ms'] != 50
            or p['window_ms'] != 50 or p['direct_window_ms'] != 200):
        raise ValueError('debug row does not match this exact checkpoint/policy')
    rawpath = Path(str(path) + '.raw_accuracy.json')
    raw = json.loads(rawpath.read_text())
    if (not raw['complete'] or raw['identity']['checkpoint_sha256'] != identity['checkpoint_sha256']
            or raw['protocol'] != p or len(raw['raw_sequences']) != 2):
        raise ValueError('incomplete debug accuracy or wrong checkpoint')
    byname = {item['seq']: item for item in raw['raw_sequences']}
    if set(byname) != set(G.NAMES) or byname['zgz_local']['n_frames'] != 1204 or byname['zgz_global']['n_frames'] != 1386:
        raise ValueError('debug row must contain all2590 frames')
    aggregated = accuracy_aggregate(raw['raw_sequences'])
    for cohort in ('overall', 'local', 'global'):
        for metric, value in aggregated[cohort].items():
            if not np.isfinite(value) or abs(value - accuracy[cohort][metric]) > 1e-8:
                raise ValueError('main row does not match raw accuracy aggregation')
    local = byname['zgz_local']
    result['values'] = {key: local[key] for key in ('jitter_all_mm_per_step', 'gt_move_mm_per_step')}
    result['values']['fused'] = accuracy['overall']['mpjpe_ra_mm']
    result['identities'] = [{'path': str(f), 'sha256': E.sha(f)} for f in (path, rawpath)]
    result['issues'].append({'reason': 'G5 requires separate matched I/D/zero-motion full rows; not assessed here'})
    return result


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('config', 'checkpoint', 'policy', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--calibration-provenance', type=Path)
    parser.add_argument('--main-row', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=1)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.npz').exists():
        raise FileExistsError(args.output)
    if args.batch_size < 1:
        raise ValueError('batch size must be positive')
    setup()
    torch.set_num_threads(4)
    E.XKFModel = XKFCentroidModel
    policy = json.loads(args.policy.read_text())
    CentroidFusionPolicy(policy)
    device = torch.device(args.device)
    model, cfg, identity = E.load_model(args.config, args.checkpoint, device)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    clock = saved.get('xkf_direct_continuation')
    if not clock or identity['global_step'] != 2400 or clock['i_adam_step'] != 2200 or clock['d_adam_step'] != 2400:
        raise ValueError('expected exact2400 D-only debug checkpoint with I at2200')
    identity['parameter_training_clocks'] = dict(clock, all_parameters_at_global_step=False)
    identity['independently_verified_parameter_clocks'] = validate_clocks(model, saved, False)
    del saved
    calibration = None
    if policy.get('direct_correction'):
        if args.calibration_provenance is None:
            raise ValueError('corrected D requires hashed calibration provenance')
        calibration = json.loads(args.calibration_provenance.read_text())
        if (calibration['checkpoint_sha256'] != identity['checkpoint_sha256']
                or calibration['seed'] != identity['seed'] or calibration['spec'] != policy['direct_correction']
                or calibration['partial_smoke'] is not False or calibration['fit_rows'] != 2304
                or calibration['identity']['config_sha256'] != identity['config_sha256']):
            raise ValueError('calibration does not match evaluated D/policy')
        fit_root = args.calibration_provenance.parent
        if (E.sha(fit_root / 'protocol.json') != calibration['protocol_sha256']
                or E.sha(fit_root / 'train_predictions.npz') != calibration['fit_source_sha256']):
            raise ValueError('calibration protocol/predictions changed')
        declaration = json.loads((fit_root / 'protocol.json').read_text())
        for name, digest in declaration['source_sha256'].items():
            if E.sha(ROOT / name) != digest:
                raise ValueError('calibration source changed: ' + name)
        if declaration['partial_smoke'] is not False or declaration['fit_rows'] != 2304:
            raise ValueError('partial calibration declaration')
    model.configure_fusion(policy)
    model.kmode = 'fused'
    if getattr(model, 'xyz_assoc_prev', 'self') != 'self':
        raise ValueError('GT association forbidden')
    for module in model.modules():
        if isinstance(module, ManoLayer):
            module.fast_chain = True
    bank, banksha = G.D.load_bank()
    if E.sha(splits_manifest(cfg)) != bank['split']['sha256']:
        raise ValueError('canonical split mismatch')
    if sum(len(bank['sequences'][name]['rows']) for name in G.NAMES) != 2590:
        raise ValueError('all2590 packets required')
    source = {**sources(), **{name: E.sha(ROOT / name) for name in (
        'semkine/xkf_centroid.py', 'tools/xkf/diagnostics.py', 'tools/xkf_fusion/debug_gates.py',
        'tools/xkf_fusion/train_calibration_tuned.py')}}
    inputs = [args.config, args.checkpoint, args.policy]
    if args.calibration_provenance:
        inputs.append(args.calibration_provenance)
        inputs.extend(args.calibration_provenance.parent / name for name in ('protocol.json', 'train_predictions.npz'))
    if args.main_row:
        inputs.extend([args.main_row, Path(str(args.main_row) + '.raw_accuracy.json')])
    hashes = {str(path.resolve()): E.sha(path) for path in inputs}
    rows = read_debug_row(args.main_row, identity, policy)
    original_predict = G.predict
    def predict(current, packet, *, direct=False, batch_size=7):
        if not direct:
            return original_predict(current, packet, batch_size=batch_size)
        # The G1 measurement is the very same fixed corrected D used by fusion.
        # Its input still has target=0 and its output does not use incoming state.
        if bool(packet.target.any()):
            raise ValueError('nonzero direct inference target')
        z, aux = current.measure_direct(packet, current.fusion_policy.temperature_multiplier)
        z = correct_direct(z, aux, current.fusion_policy.correction)
        value = z.detach().float().cpu().numpy()
        errors = [{'row': int(i), 'reason': 'nonfinite_prediction'} for i in np.flatnonzero(~np.isfinite(value).all(-1))]
        return value, errors
    G.predict = predict
    args.output.parent.mkdir(parents=True, exist_ok=True)
    single, raw, failures = G.single_step(model, cfg, bank, device, args.batch_size)
    recovered, raw_rec, rec_failures = G.recovery(model, cfg, bank, device, args.batch_size)
    raw.update(raw_rec)
    failures += rec_failures
    rawpath = args.output.with_suffix('.npz')
    np.savez_compressed(rawpath, **raw)
    gates = G.assess_gates(single, raw, recovered, rows)
    assert source == {name: E.sha(ROOT / name) for name in source}
    assert hashes == {str(path.resolve()): E.sha(path) for path in inputs}
    result = dict(kind='XKF_DEBUG_ORIGINAL_GATE_FORMULAS', formal=False,
        identity=identity, policy=policy, calibration_provenance=calibration,
        sources=source, inputs=hashes, bank_sha256=banksha, batch_size=args.batch_size,
        gates=gates, single_step=single, recovery=recovered, row_artifacts=rows,
        failures=failures, numeric_status='PASS' if not failures else 'FAIL',
        finite_subset_reduction=False, raw={'path': str(rawpath), 'sha256': E.sha(rawpath)},
        claim={'problem_solved': False, 'ready_for_full_training': False,
               'reason': 'Debug parameter clocks; unchanged formulas only. Formal requires exact6000 and matched original readiness.'})
    write_json(args.output, result)
    print(json.dumps({'output': str(args.output), 'formal': False, 'gates': {k: v['status'] for k, v in gates.items()}}), flush=True)


if __name__ == '__main__':
    main()
