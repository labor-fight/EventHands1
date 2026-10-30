"""FIT7-only, label-free GPU parity for the evaluation geometry backend.

The scope and tolerances are fixed here.  This runner must finish before any
SELECT sampling.  It never creates a supervision reader, target geometry,
optimizer or task-error metric, and never modifies the frozen training source.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'outputs/xvm_joint/20261029_070707'
GPU_UUID = 'GPU-d1abbac0-1c62-62cd-4803-15a5a172398e'
FLOAT_ATOL = 1e-5
DELTA_ATOL = 1e-6


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def record_identity(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def tensor_identity(value):
    array = value.detach().cpu().contiguous().numpy()
    return dict(shape=list(array.shape), dtype=str(array.dtype),
                sha256=hashlib.sha256(array.tobytes()).hexdigest())


def compare_tensor(reference, candidate, *, atol):
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise AssertionError('Shape or dtype differs')
    if reference.is_floating_point():
        if not bool(torch.isfinite(reference).all() & torch.isfinite(candidate).all()):
            raise AssertionError('Nonfinite parity tensor')
        difference = float((reference-candidate).abs().max()) if reference.numel() else 0.
        if difference > atol:
            raise AssertionError(f'Absolute difference {difference} exceeds fixed tolerance {atol}')
        return dict(dtype=str(reference.dtype), shape=list(reference.shape),
                    exact=bool(torch.equal(reference, candidate)), max_abs=difference, atol=atol)
    if not torch.equal(reference, candidate):
        raise AssertionError('Integer or boolean parity mismatch')
    return dict(dtype=str(reference.dtype), shape=list(reference.shape), exact=True)


def compare_prepared(reference, candidate):
    comparisons = {}
    for field in fields(reference):
        a, b = getattr(reference, field.name), getattr(candidate, field.name)
        try:
            if isinstance(a, torch.Tensor):
                comparisons[field.name] = compare_tensor(a, b, atol=FLOAT_ATOL)
            else:
                if a != b:
                    raise AssertionError('Prepared metadata differs')
                comparisons[field.name] = dict(exact=True)
        except Exception as error:
            raise AssertionError('PreparedGeometry.' + field.name + ': ' + str(error)) from error
    return comparisons


def compare_aux(reference, candidate):
    if reference.keys() != candidate.keys():
        raise AssertionError('Local auxiliary keys differ')
    comparisons = {}
    for name in reference:
        a, b = reference[name], candidate[name]
        try:
            if isinstance(a, torch.Tensor):
                comparisons[name] = compare_tensor(a, b, atol=FLOAT_ATOL)
            else:
                if a != b:
                    raise AssertionError('Local auxiliary value differs')
                comparisons[name] = dict(exact=True)
        except Exception as error:
            raise AssertionError('Local aux.' + name + ': ' + str(error)) from error
    return comparisons


def make_sourcefreeze(evaldir):
    # This operation precedes model loading, input-reader construction and every
    # real sample.  Actual event bytes are bound later by immutable batch tokens.
    from .control import resources, verify
    config_path = RUN / 'config.json'
    training_freeze = RUN / 'sourcefreeze_v2.json'
    checked = verify(training_freeze, config_path)
    training = json.loads(training_freeze.read_text())
    config = json.loads(config_path.read_text())
    source_paths = {ROOT / name for name in training['files']}
    source_paths.update((ROOT / 'tools/s43_rel').glob('*.py'))
    source_paths.update([Path(__file__), ROOT / 'tools/xvm_joint/evaluation_geometry.py'])
    files = {str(path.relative_to(ROOT)): sha(path) for path in sorted(source_paths)}
    complete = json.loads((RUN / 'train/complete.json').read_text())
    checkpoint = record_identity(complete['checkpoint']['path'])
    if checkpoint['sha256'] != complete['checkpoint']['sha256']:
        raise RuntimeError('Final checkpoint differs from completion receipt')
    selected = resources(3)
    if selected['uuid'] != GPU_UUID or os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID:
        raise RuntimeError('Only the fixed physical GPU3 UUID is authorized here')
    if os.environ.get('XJ_GPU_MEMORY_LIMIT_GIB') != '16':
        raise RuntimeError('The fixed 16GiB process cap is required')
    frozen = dict(
        schema='xj_eval_fit7_engineering_sourcefreeze_v1',
        created_utc=datetime.now(timezone.utc).isoformat(),
        scope=dict(recording='ch_global', packets=3, warmup_packets=2,
                   beta_input='zero10', previous='publicinitial then own frozen-path feedback',
                   canonical='S43 W50 original no filter', data_root=config['data_root'],
                   supervision_reads=0, optimizer_steps=0, SELECT_reads=0, zgz_reads=0,
                   prepared_float_atol=FLOAT_ATOL, physical_delta_atol=DELTA_ATOL),
        training_source_verification=checked, files=files,
        inputs=dict(config=record_identity(config_path),
                    training_sourcefreeze=record_identity(training_freeze),
                    base_checkpoint=record_identity(config['base_checkpoint']),
                    base_config=record_identity(config['base_config']),
                    split=record_identity(Path(config['data_root']) / 'splits_semkine.json'),
                    checkpoint=checkpoint,
                    evaluation_protocol=record_identity(evaldir / 'protocol.json')),
        resources_before_init=selected,
        source_bytes_bound_by='XJInputBatch token includes native rows, F1 and source K; no GT bytes',
    )
    destination = evaldir / 'engineering_sourcefreeze.json'
    write_new(destination, frozen)
    return frozen, destination, config, complete


def verify_unchanged(frozen):
    for name, digest in frozen['files'].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError('Engineering source changed: ' + name)
    for name, entry in frozen['inputs'].items():
        if sha(entry['path']) != entry['sha256']:
            raise RuntimeError('Engineering input changed: ' + name)


def run(evaldir):
    frozen, freeze_path, config, complete = make_sourcefreeze(evaldir)
    report = dict(
        schema='xj_eval_fit7_geometry_gpu_parity_v1', status='RUNNING',
        started_utc=datetime.now(timezone.utc).isoformat(),
        sourcefreeze=record_identity(freeze_path),
        frozen_geometry_sha256=frozen['files']['tools/xvm_joint/geometry.py'],
        fast_geometry_sha256=frozen['files']['tools/xvm_joint/evaluation_geometry.py'],
        runner_sha256=frozen['files']['tools/xvm_joint/evaluation_engineering.py'],
        local_model_sha256=frozen['files']['tools/xvm_joint/model.py'],
        checkpoint_sha256=frozen['inputs']['checkpoint']['sha256'],
        scope=frozen['scope'], optimizer_steps=0, supervision_reads=0,
        target_calls=0, SELECT_reads=0, zgz_reads=0, packets=[],
        note='Numerical implementation parity only; no task errors or efficacy inference.',
    )
    started = time.perf_counter()
    try:
        from tools.s43_integration.model import load_legacy
        from tools.s43_rel.mean_shape_replay import InferenceAccessGuard
        from .data import XJRunLanes, stack_inputs
        from .geometry import GeometryConfig, JointGeometry
        from .evaluation_geometry import FastJointGeometry
        from .model import JointLocalOperator
        from .train import camera_valid, configure_device, native_lnes_ms, sync, tensor_digest

        random.seed(3407)
        np.random.seed(3407)
        torch.manual_seed(3407)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        device, report['device'] = configure_device('cuda:0', config, engineering=True)
        base, _ = load_legacy(config['base_checkpoint'], config['base_config'], device)
        base = base.float().eval()
        base.render_fp32 = True
        if getattr(base, 'root_compose', 'add') != 'add':
            raise RuntimeError('Physical additive Jacobians require canonical additive root composition')
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        local = JointLocalOperator().to(device).float().eval()
        checkpoint = torch.load(frozen['inputs']['checkpoint']['path'], map_location='cpu', weights_only=False)
        if checkpoint['updates'] != 12000:
            raise RuntimeError('Parity requires the fixed final12000 checkpoint')
        local.load_state_dict(checkpoint['local'], strict=True)
        local_before = tensor_digest(local.state_dict())
        base_before = tensor_digest(base.state_dict())
        if local_before != complete['local_digest']:
            raise RuntimeError('Loaded final local tensor digest differs')
        report['loaded_local_digest'] = local_before
        del checkpoint
        gc = GeometryConfig(max_supports=int(config['num_supports']),
                            candidates_per_support=int(config['candidates_per_support']),
                            radius_m=float(config['radius_m']))
        reference_geometry = JointGeometry(base, gc)
        fast_geometry = FastJointGeometry(base, gc)
        previous = None
        with InferenceAccessGuard() as guard, torch.no_grad():
            lanes = XJRunLanes(config['data_root'], subjects=('ch',), recordings=('ch_global',),
                               batch_size=1, seed=3407, warmup_steps=2, finite=True, log_counts=False)
            report['input_identity'] = lanes.identity()
            for ordinal in range(3):
                batch = lanes.peek_batch()
                arrays = stack_inputs(batch)
                item = batch[0]
                if (len(batch) != 1 or not arrays['valid'].all()
                        or item['recording'] != 'ch_global' or item['run_id'] != 0
                        or item['packet_ordinal'] != ordinal or item['start_ms'] != 50*ordinal):
                    raise RuntimeError('First-three-packet scope differs from declared ch_global run0')
                f0 = torch.from_numpy(arrays['field0']).to(device)
                f1 = torch.from_numpy(arrays['field1']).to(device)
                K = torch.from_numpy(arrays['K_source']).to(device)
                beta0 = torch.from_numpy(arrays['beta_input']).to(device)
                initial = torch.from_numpy(arrays['initial_state']).to(device)
                exposure = torch.from_numpy(arrays['both_snapshots_valid']).to(device)
                if bool(beta0.any()):
                    raise RuntimeError('Nonzero input beta is forbidden')
                if previous is None or bool(arrays['reset'][0]):
                    previous = initial.clone()
                lnes = torch.from_numpy(np.stack([native_lnes_ms(
                    item['events'], item['start_ms'], item['cutoff_ms'])])).to(device)
                absolute_raw = base(lnes, previous, betas=beta0, camera_K=K).float()
                absolute_valid = camera_valid(base, absolute_raw, beta0)
                absolute = torch.where(absolute_valid[:, None], absolute_raw, previous)
                sync(device)
                stamp = time.perf_counter()
                reference = reference_geometry.prepare(f0, f1, previous, absolute, K,
                                                        both_exposed=exposure)
                sync(device)
                reference_s = time.perf_counter()-stamp
                stamp = time.perf_counter()
                candidate = fast_geometry.prepare(f0, f1, previous, absolute, K,
                                                  both_exposed=exposure)
                sync(device)
                fast_s = time.perf_counter()-stamp
                prepared_comparison = compare_prepared(reference, candidate)
                delta0, delta1, aux = local(**reference.model_kwargs)
                fast_delta0, fast_delta1, fast_aux = local(**candidate.model_kwargs)
                delta_comparison = dict(
                    delta0=compare_tensor(delta0, fast_delta0, atol=DELTA_ATOL),
                    delta1=compare_tensor(delta1, fast_delta1, atol=DELTA_ATOL))
                aux_comparison = compare_aux(aux, fast_aux)
                proposed = absolute+delta1
                proposed_valid = camera_valid(base, proposed, beta0)
                empty = len(item['events']) == 0
                previous = torch.where((proposed_valid & ~torch.tensor([empty], device=device))[:, None],
                                       proposed, previous).clone()
                if not bool(torch.isfinite(previous).all()):
                    raise ArithmeticError('Nonfinite shared own feedback')
                report['packets'].append(dict(
                    input_token=batch.token, batch_id=batch.batch_id,
                    recording=item['recording'], run_id=item['run_id'],
                    packet_ordinal=ordinal, start_ms=item['start_ms'], cutoff_ms=item['cutoff_ms'],
                    source_range=list(item['source_range']), event_count=len(item['events']),
                    both_exposed=bool(exposure[0]), active_records=int(reference.observed.sum()),
                    absolute_camera_valid=bool(absolute_valid[0]), local_camera_valid=bool(proposed_valid[0]),
                    input_tensors={name: tensor_identity(value) for name, value in
                                   dict(field0=f0, field1=f1, q0=reference.q0_reference,
                                        q1=reference.q1_reference, K_source=K).items()},
                    prepared=prepared_comparison, delta=delta_comparison, aux=aux_comparison,
                    descriptive_timing_s=dict(frozen=reference_s, fast=fast_s),
                ))
                lanes.commit_batch(batch)
        report['input_audit'] = guard.receipt()
        if guard.blocked_aux_beta_attempts or guard.blocked_pose_meta_attempts:
            raise RuntimeError('Inference firewall recorded a forbidden target attempt')
        if not report['packets'][2]['active_records']:
            raise RuntimeError('Third FIT7 packet did not exercise any active local record')
        if tensor_digest(local.state_dict()) != local_before or tensor_digest(base.state_dict()) != base_before:
            raise RuntimeError('Parity changed model weights')
        verify_unchanged(frozen)
        report.update(status='PASS', weights_unchanged=True, source_files_unchanged=True,
                      compared_prepared_fields=len(fields(reference)),
                      compared_aux_keys=sorted(aux), cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    except BaseException as error:
        report.update(status='FAIL', error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        raise
    finally:
        report['completed_utc'] = datetime.now(timezone.utc).isoformat()
        report['elapsed_s'] = time.perf_counter()-started
        write_new(evaldir / 'geometry_gpu_parity.json', report)
        print(json.dumps({key: report[key] for key in
                          ('status', 'runner_sha256', 'fast_geometry_sha256',
                           'frozen_geometry_sha256', 'checkpoint_sha256', 'elapsed_s')}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-dir', type=Path)
    args = parser.parse_args()
    evaldir = args.eval_dir or Path((RUN / 'active_evaluation.txt').read_text().strip())
    run(evaldir.resolve())


if __name__ == '__main__':
    main()
