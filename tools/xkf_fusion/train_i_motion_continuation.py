#!/usr/bin/env python3
"""Canonical2304x10 I-motion diagnosis for fixed I-only2600 continuation.

Same raw events, declared initial perturbations and independent target audit as
the parent2304 probe. This is one-step training-domain diagnosis, not own-loop.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from mano_layer import ManoLayer
from semkine.xkf_stable_geometry import XKFStableGeometryModel
from tools.xkf import recovery_probe as R
from tools.xkf_fusion import bn_probe as B
from tools.xkf_fusion import train_calibration_tuned as T
from tools.xkf_repair.evaluate import setup
from tools.xkf import debug_eval as E
from tools.xkf_fusion.incremental_checkpoint import validate_incremental_endpoint

STUDY = ROOT / 'outputs/xkf/fusion_20261015_r3/direct_finetune'
GAUSSIAN_SEED = 2026101501
SMALL_STD = np.array([.005] * 3 + [.05] * 48, np.float32)
CONDITIONS = tuple(value[0] for value in R.CONDITIONS) + ('small_gaussian',)
COUNT_BINS = ((0, 1), (1, 50), (50, 200), (200, 1000), (1000, 5000), (5000, None))
SPEED_BINS = ((0., 20.), (20., 100.), (100., 300.), (300., None))


def sources():
    return T.source_hashes() | {name: E.sha(ROOT / name) for name in (
        'tools/xkf_fusion/train_i_motion_probe.py', 'semkine/xkf_stable_geometry.py',
        'semkine/xkf_centroid.py', 'tools/xkf/recovery_probe.py',
        'tools/xdl/diagnostics.py', 'semkine/eval_track.py',
        'tools/xkf_fusion/train_i_motion_continuation.py','tools/xkf_fusion/incremental_checkpoint.py')}


def digest_arrays(values):
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def rotations(raw):
    """World-left relative rotations; independent of additive AA target convention."""
    incoming = raw['incoming'][:, :, 3:6]
    prediction = raw['prediction'][:, :, 3:6]
    shape = incoming.shape[:-1]
    measured = (Rotation.from_rotvec(prediction.reshape(-1, 3))
                * Rotation.from_rotvec(incoming.reshape(-1, 3)).inv())
    truth = (Rotation.from_rotvec(raw['truth'][:, 3:6])
             * Rotation.from_rotvec(raw['previous_truth'][:, 3:6]).inv())
    target = Rotation.from_rotvec(np.repeat(truth.as_rotvec()[:, None], len(CONDITIONS), 1).reshape(-1, 3))
    return (measured.as_rotvec().reshape(*shape, 3).astype(np.float32),
            truth.as_rotvec().astype(np.float32),
            np.rad2deg((measured * target.inv()).magnitude()).reshape(shape).astype(np.float32))


def reduce(raw):
    categories = raw['category']
    speed = np.linalg.norm(raw['motion_delta'][:, :3], axis=-1) * 1000 / raw['actual_label_delta_t_s']
    rows = []
    for cohort in ('all', 'local', 'global'):
        cohort_mask = np.ones(len(speed), bool) if cohort == 'all' else categories == cohort
        groups = [('all', 'all', cohort_mask)]
        for kind, values, bins in (('event_count', raw['short_count'], COUNT_BINS), ('true_t_speed_mm_s', speed, SPEED_BINS)):
            for low, high in bins:
                mask = cohort_mask & (values >= low)
                if high is not None:
                    mask &= values < high
                groups.append((kind, f'[{low},{high if high is not None else "inf"})', mask))
        for kind, label, mask in groups:
            for ci, condition in enumerate(CONDITIONS):
                row = {'cohort': cohort, 'stratum_kind': kind, 'stratum': label, 'condition': condition, 'n': int(mask.sum())}
                if not mask.any():
                    rows.append(dict(row, measured=None, reason='empty fixed bin; no invented denominator'))
                    continue
                delta = raw['predicted_motion'][mask, ci].astype(np.float64)
                target = raw['motion_delta'][mask].astype(np.float64)
                error = delta - target
                t_error, t_target = error[:, :3] * 1000, target[:, :3] * 1000
                root_world = raw['root_motion_geodesic_error_deg'][mask, ci].astype(np.float64)
                hold_root = np.rad2deg(np.linalg.norm(raw['true_root_world_delta'][mask].astype(np.float64), axis=-1))
                extra = (delta[:, :3] - raw['predicted_motion'][mask, 0, :3]) * 1000
                row['measured'] = {
                    'motion_t_rmse_mm': float(np.sqrt(np.mean(np.sum(t_error ** 2, axis=-1)))),
                    'zero_motion_hold_t_rmse_mm': float(np.sqrt(np.mean(np.sum(t_target ** 2, axis=-1)))),
                    'motion_t_axis_bias_mm': t_error.mean(0).tolist(),
                    'motion_t_axis_rmse_mm': np.sqrt(np.mean(t_error ** 2, 0)).tolist(),
                    'predicted_motion_xyz_mean_mm': (delta[:, :3].mean(0) * 1000).tolist(),
                    'true_motion_xyz_mean_mm': t_target.mean(0).tolist(),
                    'motion_t_better_than_zero_fraction': float((np.sum(t_error ** 2, -1) < np.sum(t_target ** 2, -1)).mean()),
                    'root_world_motion_rmse_deg': float(np.sqrt(np.mean(root_world ** 2))),
                    'zero_motion_hold_root_rmse_deg': float(np.sqrt(np.mean(hold_root ** 2))),
                    'root_additive_AA_motion_rmse_rad': float(np.sqrt(np.mean(error[:, 3:6] ** 2))),
                    'finger_additive_AA_motion_rmse_rad': float(np.sqrt(np.mean(error[:, 6:] ** 2))),
                    'zero_motion_hold_finger_rmse_rad': float(np.sqrt(np.mean(target[:, 6:] ** 2))),
                    'extra_I_translation_vs_clean_rmse_mm': float(np.sqrt(np.mean(np.sum(extra ** 2, -1)))),
                    'true_t_speed_mean_mm_s': float(speed[mask].mean()),
                    'event_count_mean': float(raw['short_count'][mask].mean()),
                    'endpoint': {name: float(raw['endpoint_metrics'][mask, ci, mi].mean()) for mi, name in enumerate(R.METRICS)},
                    'held_initial_endpoint': {name: float(raw['hold_endpoint_metrics'][mask, ci, mi].mean()) for mi, name in enumerate(R.METRICS)}}
                rows.append(row)
    return rows


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', required=True, type=int, choices=(3407, 3408, 3409))
    parser.add_argument('--arm',required=True,choices=('parent','raw','aug'))
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--declaration',required=True,type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--limit-rows', type=int, default=0)
    args = parser.parse_args()
    if args.limit_rows < 0:
        parser.error('limit-rows must be nonnegative')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    setup(); E.XKFModel = XKFStableGeometryModel
    torch.set_num_threads(4)
    device = torch.device(args.device)
    config, checkpoint = args.config, args.checkpoint
    model, cfg, identity = E.load_model(config, checkpoint, device)
    if identity['seed'] != args.seed:raise ValueError('explicit seed differs from model')
    declaration=json.loads(args.declaration.read_text())
    if declaration['kind']!='CANONICAL_I_CONTINUATION_MOTION_DECLARATION' or declaration['rows']!=2304 or declaration['conditions']!=list(CONDITIONS):
        raise ValueError('canonical motion predeclaration differs')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if args.arm=='parent':
        identity['training_clocks'] = T.validate_clocks(model, saved, False)
        parent_path = ROOT / f'outputs/xkf/repair_20261015_r2/repair_s{args.seed}/xkf_repair_s{args.seed}-step=2200.ckpt'
        parent = torch.load(parent_path, map_location='cpu', weights_only=False)['state_dict']
        for name, value in saved['state_dict'].items():
            if not name.startswith('direct.') or name.endswith(('running_mean', 'running_var', 'num_batches_tracked')):
                if not torch.equal(value, parent[name]):raise ValueError('parent I/BN/EMA differs from2200: '+name)
        if checkpoint.name!=f'xkf_donly_raw_s{args.seed}-step=2400.ckpt':raise ValueError('wrong parent checkpoint')
        parent_contract='I/BN/EMA bitwise2200; D2400 irrelevant to I-only forward'
        del parent
    else:
        identity['training_clocks']=validate_incremental_endpoint(model,saved,checkpoint)
        if checkpoint.name!=f'xkf_ionly_{args.arm}_s{args.seed}-step=2600.ckpt':raise ValueError('wrong I-only checkpoint arm')
        metadata=json.loads((checkpoint.parent/'training_metadata.json').read_text())
        if metadata['arm']!=args.arm:raise ValueError('training arm metadata differs')
        parent_path=Path(identity['training_clocks']['parent_checkpoint'])
        parent_contract='I updated2200to2400; D2400/allBN/EMA/DAdam bitwise raw2400 parent; global2600'
    del saved
    model.configure_stability('per_sample_i')
    model.xyz_keep_aux = False
    for module in model.modules():
        if isinstance(module, ManoLayer):
            module.fast_chain = True
    dataset = B.calibration_dataset(cfg)
    all_rows = B.calibration_rows(dataset)
    frozen_rows_path = ROOT / f'outputs/xkf/fusion_20261015_r3/bn/s{args.seed}/calibration_rows.json'
    if all_rows != json.loads(frozen_rows_path.read_text()) or len(all_rows) != 2304:
        raise ValueError('requires exact canonical2304 frozen endpoints')
    if len(dataset.sequences) != 72 or any(row['sequence'].startswith('zgz_') for row in all_rows):
        raise ValueError('noncanonical training input')
    rows = all_rows[:args.limit_rows] if args.limit_rows else all_rows
    small = (np.random.default_rng(GAUSSIAN_SEED).normal(size=(len(all_rows), 51)) * SMALL_STD).astype(np.float32)
    reference_path = STUDY / 'trajectory1049_analysis/canonical2304_motion_gt.npz'
    source = sources()
    source_inputs = {str(frozen_rows_path): E.sha(frozen_rows_path), str(reference_path): E.sha(reference_path),
                     str(Path(cfg['DATA']['ROOT']) / 'splits_semkine.json'): E.sha(Path(cfg['DATA']['ROOT']) / 'splits_semkine.json')}
    source_inputs[str(args.declaration.resolve())]=E.sha(args.declaration)
    for filename in ('training_metadata.json','train_finish.json','freeze_audit_rank0.json','freeze_audit_rank1.json','actual_exposure_rank0.json','actual_exposure_rank1.json'):
        audit_path=checkpoint.parent/filename
        if audit_path.is_file():source_inputs[str(audit_path.resolve())]=E.sha(audit_path)
    protocol = {'kind': 'CANONICAL_I_CONTINUATION_MOTION_DIAGNOSIS', 'formal': False, 'automatic_full_training': False,
        'identity': identity, 'arm':args.arm,'declaration_sha256':E.sha(args.declaration), 'source_sha256': source, 'input_sha256': source_inputs,
        'parent_checkpoint': str(parent_path), 'parent_checkpoint_sha256': E.sha(parent_path), 'parent_contract':parent_contract,
        'partial_smoke': bool(args.limit_rows), 'rows': rows, 'conditions': CONDITIONS, 'metrics': R.METRICS,
        'sample_count': len(rows), 'canonical_full_sample_count': 2304, 'canonical_train_sequences': [x[0] for x in dataset.sequences],
        'small_gaussian': {'seed': GAUSSIAN_SEED, 'translation_std_m': .005, 'root_AA_std_rad': .05, 'finger_AA_std_rad': .05,
                           'additive_same_as_training': True, 'all2304_draws_sha256': digest_arrays([small])},
        'sampling': 'exact previously frozen2304 training endpoints; raw W50 only; no DR/hot/polarity rewrites; no D',
        'previous_label_ms': 'max(run_start_ms,end_ms-50); retain first49ms label gap explicitly',
        'model_delta_t_s': .05, 'actual_inference_batch': 1, 'stability_mode': 'per_sample_i',
        'current_GT': 'used only after I forward for scoring and independent label audit; packet target always zero',
        'motion_scoring': 'primary translation/additive51: (Iout-incoming)-(GTcurrent-GTprevious), no initial perturbation in target; root world-left SO3 scored separately',
        'endpoint_scoring': 'Iout vs currentGT, compared separately with incoming vs currentGT',
        'count_bins_left_inclusive_right_exclusive': COUNT_BINS, 'true_t_speed_bins_mm_s': SPEED_BINS,
        'no_fitting_or_gate_selection': True, 'finite_subset_reduction': False,
        'domain_limit': 'canonical2304 has only8 samples below50 events; training diagnosis cannot certify sparse dev recovery'}
    B.write(args.output / 'protocol.json', protocol)
    parameters_before = B.parameter_digest(model)
    buffers_before = {name: value.cpu().clone() for name, value in model.named_buffers()}
    forbidden_calls = {'D': 0}
    def forbid_D(*unused):
        forbidden_calls['D'] += 1
        raise AssertionError('I-only diagnosis must never run D')
    hook = model.direct.register_forward_pre_hook(forbid_D)
    collected = {key: [] for key in ('incoming', 'prediction', 'previous_truth', 'truth', 'motion_delta',
        'endpoint_metrics', 'hold_endpoint_metrics', 'beta', 'camera_K', 'short_count', 'actual_label_delta_t_s', 'event_sha256')}
    elapsed_forward = 0.
    started = time.perf_counter()
    for index, row in enumerate(rows):
        handle = dataset._handle(row['sequence_id'])
        end = row['end_ms']; previous_ms = max(row['run_start_ms'], end - 50)
        previous_truth = handle.pos51[previous_ms].copy()
        raw = R.D.ET._window_events(handle.events, handle.offsets, handle.tsub, end, 50)
        beta = torch.tensor(handle.betas.copy(), device=device).float().unsqueeze(0)
        camera = torch.tensor(handle.camera_K.copy(), device=device).float().unsqueeze(0)
        clean = torch.tensor(previous_truth, device=device).float()
        incoming = torch.cat([R.initial_states(clean), (clean + torch.from_numpy(small[index]).to(device))[None]])
        packet_row = {'end_ms': end, 'event_start_ms': end - 49}
        predictions = []
        R.D.reset(model)
        packet = R.D.make_packet(raw, incoming[:1], beta, camera, device, packet_row)
        for condition_index in range(len(CONDITIONS)):
            packet.prev_state = incoming[condition_index:condition_index + 1]
            if bool(packet.target.any()) or packet.lnes is not None:
                raise AssertionError('I packet must carry zero target and no D LNES')
            R.D.reset(model)
            if device.type == 'cuda': torch.cuda.synchronize(device)
            start = time.perf_counter()
            predictions.append(model.incremental_prediction(packet))
            if device.type == 'cuda': torch.cuda.synchronize(device)
            elapsed_forward += time.perf_counter() - start
        prediction = torch.cat(predictions)
        R.D.reset(model)
        # Current GT is first used here, after all ten inference calls.
        truth = handle.pos51[end].copy()
        target = torch.from_numpy(truth).to(device).float().unsqueeze(0)
        endpoint_metrics, _, _ = R.frame_metrics(model, prediction, target, beta)
        hold_metrics, _, _ = R.frame_metrics(model, incoming, target, beta)
        values = {'incoming': incoming.cpu().numpy(), 'prediction': prediction.cpu().numpy(),
            'previous_truth': previous_truth, 'truth': truth, 'motion_delta': truth - previous_truth,
            'endpoint_metrics': endpoint_metrics.cpu().numpy(), 'hold_endpoint_metrics': hold_metrics.cpu().numpy(),
            'beta': handle.betas.copy(), 'camera_K': handle.camera_K.copy(), 'short_count': len(raw),
            'actual_label_delta_t_s': (end - previous_ms) / 1000., 'event_sha256': digest_arrays([raw])}
        for key, value in values.items():
            if np.asarray(value).dtype.kind in 'fc' and not np.isfinite(value).all():
                raise FloatingPointError('nonfinite row: ' + key)
            collected[key].append(value)
        if (index + 1) % 64 == 0 or index + 1 == len(rows):
            B.write(args.output / 'progress.json', {'completed_rows': index + 1, 'total_rows': len(rows),
                                                  'elapsed_seconds': time.perf_counter() - started})
            print(json.dumps({'seed': args.seed, 'completed_rows': index + 1}), flush=True)
    hook.remove()
    raw = {key: np.stack(value) for key, value in collected.items()}
    raw.update(sequence=np.array([r['sequence'] for r in rows]), end_ms=np.array([r['end_ms'] for r in rows]),
               previous_ms=np.array([max(r['run_start_ms'], r['end_ms'] - 50) for r in rows]),
               category=np.array(['local' if '_local' in r['sequence'] else 'global' for r in rows]),
               small_gaussian_draw=small[:len(rows)],
               predicted_motion=raw['prediction'] - raw['incoming'])
    raw['motion_error'] = raw['predicted_motion'] - raw['motion_delta'][:, None]
    (raw['predicted_root_world_delta'], raw['true_root_world_delta'], raw['root_motion_geodesic_error_deg']) = rotations(raw)
    with np.load(reference_path, allow_pickle=False) as reference:
        for key, name in (('previous_truth', 'previous_truth'), ('truth', 'truth'), ('motion_delta', 'motion_delta'),
                          ('short_count', 'short_count'), ('actual_label_delta_t_s', 'delta_t_s'), ('sequence', 'sequence'), ('end_ms', 'end_ms')):
            if not np.array_equal(raw[key], reference[name][:len(rows)]):
                raise ValueError('independent canonical label/domain reference differs: ' + key)
    if B.parameter_digest(model) != parameters_before or sources() != source:
        raise ValueError('model parameters/source changed')
    for name, value in model.named_buffers():
        if not torch.equal(value.cpu(), buffers_before[name]):
            raise ValueError('buffer changed: ' + name)
    for path, digest in source_inputs.items():
        if E.sha(path) != digest:
            raise ValueError('input changed: ' + path)
    if E.sha(checkpoint) != identity['checkpoint_sha256'] or E.sha(config) != identity['config_sha256']:
        raise ValueError('checkpoint/config changed')
    if forbidden_calls['D']:
        raise AssertionError('D was invoked')
    raw_path = args.output / 'predictions.npz'
    np.savez_compressed(raw_path, **raw)
    result = {'kind': protocol['kind'], 'status': 'COMPLETE', 'formal': False, 'automatic_full_training': False,
        'identity': identity, 'arm':args.arm, 'sample_count': len(rows), 'condition_count': len(CONDITIONS), 'partial_smoke': bool(args.limit_rows),
        'protocol_sha256': E.sha(args.output / 'protocol.json'), 'raw_sha256': E.sha(raw_path),
        'input_feature_sha256': digest_arrays([raw[k] for k in ('event_sha256', 'beta', 'camera_K', 'previous_truth', 'small_gaussian_draw')]),
        'independent_canonical_GT_count_clock_byte_equal': True,
        'all_parameters_unchanged_sha256': parameters_before, 'all_named_buffers_unchanged': True,
        'D_forward_calls': forbidden_calls['D'], 'actual_I_forward_calls': len(rows) * len(CONDITIONS),
        'actual_I_forward_seconds': elapsed_forward, 'elapsed_seconds': time.perf_counter() - started,
        'conditions': CONDITIONS, 'metrics': R.METRICS, 'summaries': reduce(raw), 'finite_subset_reduction': False,
        'scope': 'train-only motion diagnosis; no fitted coefficient, threshold, optimizer or readiness certification'}
    B.write(args.output / 'result.json', result)
    print(json.dumps({'seed': args.seed, 'status': 'COMPLETE', 'rows': len(rows), 'seconds': result['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
