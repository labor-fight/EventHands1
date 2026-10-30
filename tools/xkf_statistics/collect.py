#!/usr/bin/env python3
"""Collect matched train-only I/D residuals after actual baseline self-burn.

This exports observations, never fits or installs coefficients. Only the first
declared GT state initializes inference; all packet targets are zero. Each of
2016 fixed legal training anchors owns three prefix forwards and one main.
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
from semkine.xkf_calibrated_motion_gate import XKFCalibratedMotionModel
from tools.xkf import debug_eval as E
from tools.xkf import recovery_probe as R
from tools.xkf.runtime import LongWindowEval
from tools.xkf_repair.evaluate import setup
from tools.xkf_fusion import bn_probe as B
from tools.xkf_fusion.evaluate_i_continuation import sources as baseline_sources
from tools.xkf_fusion.incremental_checkpoint import validate_incremental_endpoint

SEEDS = (3407, 3408, 3409)
BASELINE = dict(id='fixed1000', gain_scale=[.5, .25, .5], count_gate='ramp1000',
                motion_count_gate='ramp1000', translation_reference='joint_centroid')
reset = R.D.reset


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return E.sha(Path(path))


def write(path, value):
    Path(path).write_text(json.dumps(E.safe_json(value), indent=2, allow_nan=False) + '\n')


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def array_sha(*values):
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def sources():
    return baseline_sources() | {name: sha(ROOT / name) for name in (
        'tools/xkf_statistics/collect.py', 'tools/xkf/runtime.py',
        'tools/xkf/data.py', 'tools/xdl/data.py', 'tools/xdl/diagnostics.py',
        'semkine/dataset.py', 'semkine/eval_track.py', 'semkine/events.py',
        'model/mano_layer.py', 'model/pose_repr.py')}


def verify_bound_files(declaration):
    for field in ('source_sha256', 'input_sha256'):
        for name, digest in declaration.get(field, {}).items():
            require(sha(resolve(name)) == digest, 'declared dependency changed: ' + name)
    for key in ('rows', 'gaussian_draws', 'validation_schedule'):
        require(sha(resolve(declaration[key + '_path'])) == declaration[key + '_sha256'],
                'declared input changed: ' + key)
    for model in declaration['models'].values():
        for key in ('config', 'checkpoint'):
            require(sha(resolve(model[key])) == model[key + '_sha256'], 'bound model changed: ' + key)


def load_declaration(path):
    value = json.loads(Path(path).read_text())
    require(value['kind'] == 'TRAIN_ONLY_PAIRED_STATISTICS_DECLARATION'
            and value['status'] == 'ACTIVE_TRAIN_ONLY_POLICY_TRIAL', 'inactive/wrong statistics declaration')
    require(value['seeds'] == list(SEEDS) and set(value['models']) == {str(s) for s in SEEDS},
            'three same-rule endpoints required')
    require(value['read_development_data'] is False and value['training_weights_update'] is False
            and value['automatic_full_training'] is False, 'wrong trial scope')
    require(value['baseline_policy'] == BASELINE, 'baseline policy changed')
    require(value.get('crossfit', {}).get('secondary') == 'motion_family'
            and value['crossfit']['identity_folds'] == 9
            and value['crossfit']['motion_family_folds'] == 18,
            'superseded v1/72-sequence leakage protocol cannot be executed')
    require(value['gaussian_seed'] == 2026101502 and value['gaussian_std'] == [.005]*3 + [.05]*48,
            'fixed independent Gaussian changed')
    verify_bound_files(value)
    return value


def load_bound_model(declaration, seed, device):
    """Shared collector/validation loader; no dataset or development bank reads."""
    require(seed in SEEDS, 'unknown seed')
    bound = declaration['models'][str(seed)]
    config, checkpoint = resolve(bound['config']), resolve(bound['checkpoint'])
    require(sha(config) == bound['config_sha256'] and sha(checkpoint) == bound['checkpoint_sha256'],
            'bound endpoint/config changed')
    setup()
    E.XKFModel = XKFCalibratedMotionModel
    model, cfg, identity = E.load_model(config, checkpoint, torch.device(device))
    require(identity['seed'] == seed and checkpoint.name == f'xkf_ionly_raw_s{seed}-step=2600.ckpt',
            'statistics candidate must use the same raw2600 arm for every seed')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    identity['training_clocks'] = validate_incremental_endpoint(model, saved, checkpoint)
    metadata = json.loads((checkpoint.parent / 'training_metadata.json').read_text())
    require(metadata['arm'] == 'raw', 'wrong training arm')
    del saved
    model.configure_stability('per_sample_model')
    model.configure_fusion(declaration['baseline_policy'])
    model.kmode = 'fused'
    model.eval()
    for module in model.modules():
        if isinstance(module, ManoLayer):
            module.fast_chain = True
    return model, cfg, identity


build_model = load_bound_model


def training_inputs(cfg, declaration):
    dataset = B.calibration_dataset(cfg)
    all_rows = json.loads(resolve(declaration['rows_path']).read_text())
    require(all_rows == B.calibration_rows(dataset) and len(all_rows) == 2304,
            'canonical raw training anchors changed')
    names = [name for name, _ in dataset.sequences]
    require(len(names) == 72 and len({name.split('_')[0] for name in names}) == 9
            and not any(name.startswith('zgz_') for name in names), 'noncanonical training population')
    indices = declaration['fit_row_indices']
    require(indices == [i for i, row in enumerate(all_rows) if row['stratum'] == 'full200']
            and len(indices) == len(set(indices)) == 2016, '2016 fit row membership/order changed')
    # Independently prove every selected row belongs to the original K4 main index.
    # Search one sequence at a time, without a multi-million-tuple Python set.
    for si, name in enumerate(names):
        legal = dataset.legal_index[dataset.legal_index[:, 0] == si]
        order = np.argsort(legal[:, 1], kind='stable')
        legal = legal[order]
        selected = [i for i in indices if all_rows[i]['sequence_id'] == si]
        require(len(selected) == 28, 'must retain 28 legal anchors per sequence')
        for index in selected:
            row = all_rows[index]
            end = row['end_ms']
            loc = int(np.searchsorted(legal[:, 1], end))
            expected = np.array([si, end, row['run_start_ms'], row['run_stop_ms']])
            require(row['sequence'] == name and row['window_ms'] == 200
                    and end >= row['run_start_ms'] + 199 and loc < len(legal)
                    and np.array_equal(legal[loc], expected), 'anchor not in original main legal_index')
    with np.load(resolve(declaration['gaussian_draws_path']), allow_pickle=False) as raw:
        draws = raw['draws'].copy()
    expected = (np.random.default_rng(declaration['gaussian_seed']).normal(size=(2304, 51))
                * np.array(declaration['gaussian_std'], dtype=np.float32)).astype(np.float32)
    require(draws.shape == (2304, 51) and draws.dtype == np.float32 and np.isfinite(draws).all()
            and np.array_equal(draws, expected), 'Gaussian draw array differs from frozen generator')
    return dataset, all_rows, indices, draws


def packet_for_end(handle, end, run_start, prev, beta, camera):
    """Raw W50/W200 input only. Never touches handle.pos51 or any GT state."""
    require(prev.shape == (1, 51), 'actual full-forward batch must be one')
    end, run_start = int(end), int(run_start)
    runtime = LongWindowEval(handle.valid_runs, 200)
    require(runtime.floor(end) == run_start and end - 49 >= run_start,
            'W50 packet crosses the declared valid run')
    raw = R.D.ET._window_events(handle.events, handle.offsets, handle.tsub, end, 50)
    lnes, window = runtime.lnes(handle.events, handle.offsets, end)
    packet = R.D.make_packet(raw, prev, beta, camera, prev.device,
                             {'end_ms': end, 'event_start_ms': end - 49})
    packet.lnes = torch.from_numpy(lnes).unsqueeze(0).to(prev.device)
    packet.lnes_window_ms = torch.tensor([window], device=prev.device, dtype=torch.long)
    require(not bool(packet.target.any()), 'inference target must stay zero')
    return packet, dict(end_ms=end, event_start_ms=end-49, short_count=len(raw),
                        lnes_window_ms=window, event_sha256=array_sha(raw), lnes_sha256=array_sha(lnes))


def snapshot_model(model):
    return dict(parameters=B.parameter_digest(model),
                buffers={name: value.detach().cpu().clone() for name, value in model.named_buffers()})


def verify_model(model, before):
    require(B.parameter_digest(model) == before['parameters'], 'model parameters changed')
    after = dict(model.named_buffers())
    require(set(after) == set(before['buffers']), 'model buffer structure changed')
    hashes = {}
    for name, value in before['buffers'].items():
        require(torch.equal(value, after[name].detach().cpu()), 'model buffer changed: ' + name)
        hashes[name] = array_sha(value.numpy())
    return dict(all_parameters_unchanged_sha256=before['parameters'],
                all_named_buffers_bitwise_unchanged=True, named_buffer_sha256=hashes,
                runtime_buffer_evidence='before/after torch.equal and per-buffer typed content hashes')


@torch.no_grad()
def joints_single(model, state, beta):
    require(state.shape == (1, 51) and beta.shape == (1, 10), 'geometry must use matching B1 beta')
    return model._fk(state, beta)[1][0].detach().cpu().numpy().copy()


def paired_residuals(gated_i, direct, truth, joints_i, joints_d, joints_gt):
    """Physical centroid, GT right tangent, and local-AA residuals, in m/rad."""
    ei = np.asarray(gated_i, np.float64) - np.asarray(truth, np.float64)
    ed = np.asarray(direct, np.float64) - np.asarray(truth, np.float64)
    ci, cd, cg = [np.asarray(j, np.float64).mean(0) for j in (joints_i, joints_d, joints_gt)]
    ei[:3], ed[:3] = ci-cg, cd-cg
    ri, rd, rg = [Rotation.from_rotvec(x[3:6]) for x in (gated_i, direct, truth)]
    ei[3:6], ed[3:6] = (rg.inv()*ri).as_rotvec(), (rg.inv()*rd).as_rotvec()
    innovation = (ri.inv()*rd).as_rotvec()
    clipped = innovation * min(1., np.deg2rad(60.) / max(np.linalg.norm(innovation), 1e-12))
    return ei, ed, innovation, clipped


def data_file_signature(handle):
    """Selected features carry content hashes; large source files carry stat identity."""
    result = {}
    for suffix in ('_events.npy', '_offsets.npy', '_tsub.npy', '_aux.npz', '.meta'):
        path = Path(handle.base + suffix)
        if path.exists():
            stat = path.stat()
            result[str(path)] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, inode=stat.st_ino)
    return result


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declaration', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--limit-rows', type=int, default=0)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    require(args.limit_rows >= 0, 'limit-rows must be nonnegative')
    require(not args.output.exists(), 'output already exists; never overwrite a frozen run')
    declaration = load_declaration(args.declaration)
    torch.set_num_threads(4)
    source_before = sources()
    model, cfg, identity = load_bound_model(declaration, args.seed, torch.device(args.device))
    dataset, rows, indices, draws = training_inputs(cfg, declaration)
    chosen = indices[:args.limit_rows] if args.limit_rows else indices
    input_hashes = {str(args.declaration.resolve()): sha(args.declaration),
                    str(resolve(declaration['rows_path'])): declaration['rows_sha256'],
                    str(resolve(declaration['gaussian_draws_path'])): declaration['gaussian_draws_sha256'],
                    str(resolve(declaration['validation_schedule_path'])): declaration['validation_schedule_sha256'],
                    str(Path(cfg['DATA']['ROOT'])/'splits_semkine.json'): sha(Path(cfg['DATA']['ROOT'])/'splits_semkine.json'),
                    str(resolve(cfg['MANO']['NPZ'])): sha(resolve(cfg['MANO']['NPZ']))}
    args.output.mkdir(parents=True)
    baseline_effective = model.fusion_policy.gains(model.gains(), torch.tensor([1000.], device=args.device))[0].cpu().tolist()
    protocol = dict(kind='PAIRED_STATISTICS_TRAIN_COLLECTION', formal=False, automatic_full_training=False,
        identity=identity, declaration_sha256=sha(args.declaration), source_sha256=source_before,
        input_sha256=input_hashes, baseline_policy=BASELINE, partial_smoke=bool(args.limit_rows),
        row_indices=chosen, rows=[rows[i] for i in chosen], original_main_legal_index_verified=True,
        legal_index_sha256=dataset.coverage['eligible_main_index_sha256'], fit_population=2016,
        excluded_cold_rows=288, actual_inference_batch=1, stability_mode='per_sample_model',
        prefix_count=3, total_forward_steps_per_anchor=4, initial_gt_ms='main_end-199',
        end_offsets_ms=[-150,-100,-50,0], GT_path='declared initial GT only before inference; every step target0; current/previous GT only after forward',
        e_i_e_d='Nx51: physical all21 centroid metres, Log(R_GT.T R_pred) radians, local15x3AA radians; I residual uses gated I',
        high_motion_speed='mean_j norm((J_GTnow-J_GTprevious)/.05)*1000 mm/s for main',
        input_hash_scope='exact selected W50 event and W200 LNES hashes in NPZ; full GT/beta/K/valid-run typed hashes per visited training sequence; data file stat identity before/after, no claim of full event-file SHA',
        no_development_data_loaded=True, no_coefficient_fitting=True, finite_subset_reduction=False)
    write(args.output/'protocol.json', protocol)
    if args.prepare_only:
        return
    before = snapshot_model(model)
    collected, handles, data_signatures, domain_hashes = [], {}, {}, {}
    actual_calls = 0
    started = time.perf_counter()
    for position, row_index in enumerate(chosen):
        row = rows[row_index]
        handle = dataset._handle(row['sequence_id'])
        require(not handle.seq.startswith('zgz_'), 'development handle prohibited')
        if handle.seq not in handles:
            handles[handle.seq] = handle
            data_signatures[handle.seq] = data_file_signature(handle)
            domain_hashes[handle.seq] = array_sha(handle.pos51, handle.betas, handle.camera_K, handle.valid_runs)
        beta = torch.from_numpy(handle.betas.copy()).float().to(args.device)[None]
        camera = torch.from_numpy(handle.camera_K.copy()).float().to(args.device)[None]
        main_end = row['end_ms']
        initial_truth = handle.pos51[main_end-199].copy()
        initial = initial_truth + draws[row_index]
        previous = torch.from_numpy(initial.copy()).float().to(args.device)[None]
        reset(model)
        model.configure_fusion(BASELINE)
        model.fusion_policy.capture = True
        trajectory = []
        for step, end in enumerate(range(main_end-150, main_end+1, 50)):
            packet, obs = packet_for_end(handle, end, row['run_start_ms'], previous, beta, camera)
            # This reproduces the baseline's packet-local cache contract; previous
            # remains the actual previous fused state, never a GT replacement.
            reset(model)
            fused = model.forward_packet(packet)
            actual_calls += 1
            trace = model.fusion_policy.trace
            require(len(trace) == step+1 and all(v['motion'].shape == (1,51) for v in trace), 'not actual uncached B1')
            observation = trace[-1]
            require(torch.equal(observation['incoming'], previous), 'unexpected incoming state')
            # GT is first consulted here after this packet's complete forward.
            truth = handle.pos51[end].copy()
            previous_ms = main_end-199 if step == 0 else end-50
            previous_truth = handle.pos51[previous_ms].copy()
            truth_state = torch.from_numpy(truth).float().to(args.device)[None]
            old_truth_state = torch.from_numpy(previous_truth).float().to(args.device)[None]
            values = dict(incoming=previous[0].cpu().numpy().copy(),
                motion=observation['motion'][0].cpu().numpy().copy(),
                gated_motion=observation['gated_motion'][0].cpu().numpy().copy(),
                direct=observation['direct'][0].cpu().numpy().copy(), fused=fused[0].cpu().numpy().copy(),
                effective_gains=observation['effective_gains'][0].cpu().numpy().copy(),
                motion_gate=float(observation['motion_gate'][0]), truth=truth, previous_truth=previous_truth,
                truth_joints=joints_single(model,truth_state,beta),
                previous_truth_joints=joints_single(model,old_truth_state,beta),
                previous_gt_ms=previous_ms, actual_label_dt_s=(end-previous_ms)/1000., **obs)
            require(all(np.isfinite(v).all() for v in values.values() if np.asarray(v).dtype.kind in 'fc'),
                    'nonfinite trajectory; no row may be dropped')
            trajectory.append(values)
            previous = fused
        main = trajectory[-1]
        ji = joints_single(model, torch.from_numpy(main['gated_motion']).to(args.device)[None], beta)
        jd = joints_single(model, torch.from_numpy(main['direct']).to(args.device)[None], beta)
        ei, ed, innovation, clipped = paired_residuals(main['gated_motion'], main['direct'], main['truth'], ji, jd, main['truth_joints'])
        speed = float(np.linalg.norm((main['truth_joints'].astype(float)-main['previous_truth_joints'].astype(float))/.05, axis=1).mean()*1000)
        item = dict(row_indices=row_index, sequence=row['sequence'], training_identity=row['sequence'].split('_')[0],
            motion_family=row['sequence'].split('_v')[0], end_ms=main_end, run_start_ms=row['run_start_ms'], run_stop_ms=row['run_stop_ms'],
            initial_gt_ms=main_end-199, initial_truth=initial_truth, initial=initial, gaussian_draw=draws[row_index],
            beta=handle.betas.copy(), camera_K=handle.camera_K.copy(), e_i=ei, e_d=ed,
            gates=np.float32(main['motion_gate']), gate=np.float32(main['motion_gate']), gt_joint_speed_mm_s=speed,
            truth=main['truth'], previous_truth=main['previous_truth'], truth_joints=main['truth_joints'],
            previous_truth_joints=main['previous_truth_joints'], gated_i_joints=ji, direct_joints=jd,
            root_i_to_d_innovation=innovation, root_i_to_d_clipped60=clipped)
        item.update({'trajectory_'+key:np.stack([r[key] for r in trajectory]) for key in trajectory[0]})
        require(np.isfinite(ei).all() and np.isfinite(ed).all(), 'nonfinite paired main residual')
        collected.append(item)
        reset(model)
        if (position+1)%64 == 0 or position+1 == len(chosen):
            progress=dict(seed=args.seed, completed_rows=position+1, total_rows=len(chosen), elapsed_seconds=time.perf_counter()-started)
            write(args.output/'progress.json', progress)
            print(json.dumps(progress), flush=True)
    raw = {key:np.stack([r[key] for r in collected]) for key in collected[0]}
    require(actual_calls == len(chosen)*4, 'missing actual self-burn/main forward')
    for name, handle in handles.items():
        require(data_file_signature(handle) == data_signatures[name], 'training source file changed: '+name)
        require(array_sha(handle.pos51,handle.betas,handle.camera_K,handle.valid_runs) == domain_hashes[name], 'training GT/context changed')
    audit = verify_model(model, before)
    require(sources() == source_before, 'collector/runtime source changed')
    verify_bound_files(declaration)
    for path, digest in input_hashes.items():
        require(sha(path) == digest, 'input changed during collection: '+path)
    path = args.output/'residuals.npz'
    np.savez_compressed(path, **raw)
    write(args.output/'model_state_audit.json', audit)
    write(args.output/'result.json', dict(kind=protocol['kind'], status='COMPLETE', formal=False,
        identity=identity, partial_smoke=bool(args.limit_rows), rows=len(chosen), canonical_fit_rows=2016,
        actual_uncached_B1_forward_calls=actual_calls, original_main_legal_index_verified=True,
        protocol_sha256=sha(args.output/'protocol.json'), raw_sha256=sha(path),
        source_sha256=source_before,input_sha256=input_hashes,
        model_state_audit_sha256=sha(args.output/'model_state_audit.json'), **audit,
        training_domain_sha256=domain_hashes, data_file_stat_identity=data_signatures,
        all_training_data_file_stats_unchanged=True, no_development_data_loaded=True,
        no_coefficient_fitting=True, finite_subset_reduction=False, elapsed_seconds=time.perf_counter()-started))
    write(args.output/'summary.json', dict(kind=protocol['kind'], status='COMPLETE', seed=args.seed,
        declaration_path=str(args.declaration.resolve()), declaration_sha256=sha(args.declaration),
        raw_path=str(path.resolve()), raw_sha256=sha(path), baseline_effective_gains=baseline_effective,
        rows=len(chosen), actual_forward_calls=actual_calls, partial_smoke=bool(args.limit_rows),
        source_sha256=source_before, input_sha256=input_hashes, identity=identity,
        all_parameters_unchanged_sha256=audit['all_parameters_unchanged_sha256'],
        all_buffers_unchanged=True, result_path=str((args.output/'result.json').resolve()),
        result_sha256=sha(args.output/'result.json')))
    print(json.dumps(dict(status='COMPLETE', seed=args.seed, rows=len(chosen))), flush=True)


if __name__ == '__main__':
    main()
