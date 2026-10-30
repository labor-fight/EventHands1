#!/usr/bin/env python3
"""Assess the fixed r2 paired debug endpoint. Never starts or authorizes training.

This reader is deliberately outside the frozen training dependency manifest. It
records its own hash and every input hash. Missing evidence is INCOMPLETE; invalid
evidence is INVALID. A debug safeguard PASS is never a formal or full-training PASS.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ASSESSOR_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
SEEDS = (3407, 3408, 3409)
ARMS = ('control', 'repair')
STEP = 2200
LR = .0004
ACCURACY = ('mpjpe_ra_mm', 'mpvpe_ra_mm', 'mpjpe_abs_mm', 'mpvpe_abs_mm')
NROWS = {'zgz_local': 1204, 'zgz_global': 1386, 'lyq_local': 128, 'ch_global': 128}
GUARDS = {
    'allthree_original_debug_readiness': True,
    'eachseed_localbank_clean_fused_abs_not_worse_than_hold': True,
    'eachseed_localcold_fullhorizon_clean_fused_abs_not_worse_than_hold': True,
    'eachseed_globalbank_clean_fused_abs_le_control_times': 1.1,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def finite(value):
    """Required numeric fields may not silently become JSON null or booleans."""
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def assert_close(actual, expected, where):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f'{where}: key mismatch')
        for key in expected:
            assert_close(actual[key], expected[key], f'{where}.{key}')
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), f'{where}: length mismatch')
        for i, (a, b) in enumerate(zip(actual, expected)):
            assert_close(a, b, f'{where}[{i}]')
    elif isinstance(expected, float):
        require(finite(actual) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-8), f'{where}: numeric mismatch')
    else:
        require(actual == expected, f'{where}: value mismatch')


def extracted_functions(path, names, namespace=None):
    """Execute only named pure reducers from hash-checked frozen source, no model imports."""
    tree = ast.parse(Path(path).read_text())
    selected = [node for node in tree.body if
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
                or isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)]
    env = dict(namespace or {})
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), env)
    require(all(name in env for name in names), 'frozen reducer not found')
    return env


def validate_lr(record, rank):
    require(record['global_step'] == 2000 and record['rank'] == rank, 'LR restore clock/rank mismatch')
    require(record['phase'] == 'after_restore_before_first_update', 'LR not measured after restore')
    require(record['configured_lr'] == LR and record['actual_after'] == [[LR]], 'actual restored LR is not .0004')
    require(record['adam_state_and_scheduler_clock_preserved'] is True, 'optimizer restore contract absent')
    require(len(record['schedulers']) == 1, 'expected one scheduler')
    scheduler = record['schedulers'][0]
    require(scheduler['last_epoch'] == 2000 and scheduler['base_lrs'] == [LR]
            and scheduler['last_lr'] == [LR], 'restored scheduler clock/rate mismatch')


def validate_stream(stream, rank):
    require(len(stream) == 3, 'need all three recorded paired batches')
    for offset, row in enumerate(stream):
        require(row['optimizer_step'] == 2000 + offset and row['rank'] == rank, 'data optimizer clock/rank mismatch')
        require(row['global_rows'] == list(range(rank, 1024, 2)) and row['main_rows'] == 512, 'rank-stride batch mismatch')
        require(len(row['prefix_sha256']) == 3 and row['main_events'] > 0, 'missing prefix/event evidence')
        for digest in [row['main_sha256'], row['sample_index_sha256'], *row['prefix_sha256']]:
            require(isinstance(digest, str) and len(digest) == 64, 'invalid packet hash')


def validate_main_row(row, seed, step=STEP):
    protocol = row['protocol']
    require(protocol['formal'] is False and protocol['exact_optimizer_step'] == step, 'row is formal or wrong optimizer step')
    require(protocol['seeds'] == [seed] and protocol['n_seeds'] == 1, 'row seed identity mismatch')
    for key, value in {'split': 'val_core', 'step_ms': 50, 'window_ms': 50, 'direct_window_ms': 200,
                       'noise_rng_seed': 0, 'noise_scale': 1., 'tf32': False}.items():
        require(protocol[key] == value, f'row protocol mismatch: {key}')
    require(row['finite_subset_reduction'] is False and row['numeric_status'] == 'PASS'
            and row['numeric_valid'] is True, 'row contains invalid/subset accuracy')
    require(set(row['per_seed']) == {str(seed)}, 'row contains extra/missing seeds')
    accuracy = row['per_seed'][str(seed)]
    require(accuracy['n_frames'] == 2590 and accuracy['step'] == step, 'row denominator/step mismatch')
    require(set(accuracy['sequences']) == {'zgz_local', 'zgz_global'}, 'row sequences incomplete')
    for group in ('overall', 'local', 'global'):
        require(all(finite(accuracy[group].get(k)) for k in ACCURACY), f'nonfinite row {group}')
        assert_close(row['two_seed_mean'][group], accuracy[group], f'row.{group}')
    for key in ('latency_ms_scaled_full1p75', 'macs_forward_packet', 'params_total'):
        require(finite(row[key]) and row[key] > 0, f'invalid {key}')
    return accuracy


def merge_main_rows(rows, arm, provenance):
    require(set(rows) == set(SEEDS), 'all three fixed seeds are required for a merged row')
    for seed in SEEDS:
        validate_main_row(rows[seed], seed)
    first = rows[SEEDS[0]]
    for seed in SEEDS[1:]:
        for key in ('macs_forward_packet', 'params_total'):
            require(rows[seed][key] == first[key], f'cross-seed {key} mismatch')
    merged = {key: first[key] for key in ('macs_forward_packet', 'params_total')}
    merged.update(per_seed={str(seed): rows[seed]['per_seed'][str(seed)] for seed in SEEDS},
                  two_seed_mean={group: {key: float(np.mean([rows[s]['per_seed'][str(s)][group][key] for s in SEEDS]))
                                          for key in ACCURACY} for group in ('overall', 'local', 'global')},
                  latency_ms_scaled_full1p75=float(np.mean([rows[s]['latency_ms_scaled_full1p75'] for s in SEEDS])),
                  latency_context_audit={str(s): rows[s]['latency_context_audit'] for s in SEEDS},
                  protocol=dict(first['protocol'], seeds=list(SEEDS), n_seeds=3,
                                arm=arm, formal=False, averaging='equal seed mean; legacy two_seed_mean key contains all three seeds'),
                  provenance=provenance, finite_subset_reduction=False, numeric_status='PASS',
                  numeric_valid=True, invalid_numeric_paths=[], automatic_full_training=False)
    return merged


def expanded_guards(rollouts, readiness):
    """Every comparison is per seed; a good average cannot hide a failed seed."""
    per_seed = {}
    for seed in SEEDS:
        control, repair = rollouts['control'][seed], rollouts['repair'][seed]
        def value(result, cohort, mode, scope='all_observed'):
            return result[(cohort, 'clean', mode, scope)]['pooled_frames']['abs_mpjpe_mm']
        entries = {}
        for name, cohort, scope in (
            ('localbank_clean', 'local_bank', 'all_observed'),
            ('localcold_fullhorizon_clean', 'local_cold', 'full_horizon'),
        ):
            observed, limit = value(repair, cohort, 'fused', scope), value(repair, cohort, 'hold', scope)
            require(finite(observed) and finite(limit), 'nonfinite guard value')
            entries[name] = {'repair_fused': observed, 'hold': limit, 'pass': observed <= limit}
        observed, reference = value(repair, 'global_bank', 'fused'), value(control, 'global_bank', 'fused')
        require(finite(observed) and finite(reference), 'nonfinite global guard value')
        entries['globalbank_clean'] = {'repair_fused': observed, 'control_fused': reference,
                                       'control_multiplier': 1.1, 'limit': reference * 1.1, 'pass': observed <= reference * 1.1}
        entries['original_debug_readiness'] = readiness['repair'][seed]['status'] == 'PASS'
        per_seed[str(seed)] = entries
    checks = {
        'allthree_original_debug_readiness': all(v['original_debug_readiness'] for v in per_seed.values()),
        'eachseed_localbank_clean_fused_abs_not_worse_than_hold': all(v['localbank_clean']['pass'] for v in per_seed.values()),
        'eachseed_localcold_fullhorizon_clean_fused_abs_not_worse_than_hold': all(v['localcold_fullhorizon_clean']['pass'] for v in per_seed.values()),
        'eachseed_globalbank_clean_fused_abs_le_control_times': all(v['globalbank_clean']['pass'] for v in per_seed.values()),
    }
    return {'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks, 'per_seed': per_seed,
            'scope': 'r2 additional debug safeguards only; does not assess formal G gates or authorize full training'}


class Assessment:
    def __init__(self, study, row_dir):
        self.study, self.row_dir = Path(study), Path(row_dir)
        self.missing, self.errors, self.inputs = [], [], {}
        self.hashes = {}

    def path(self, path):
        path = Path(path)
        return path if path.is_absolute() else ROOT / path

    def digest(self, path):
        path = self.path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        signature = (path.stat().st_size, path.stat().st_mtime_ns)
        key = str(path)
        if key not in self.hashes or self.hashes[key][0] != signature:
            self.hashes[key] = (signature, sha(path))
        self.inputs[key] = self.hashes[key][1]
        return self.hashes[key][1]

    def read(self, path):
        path = self.path(path)
        self.digest(path)
        return json.loads(path.read_text())

    def attempt(self, label, fn):
        try:
            return fn()
        except FileNotFoundError as exc:
            self.missing.append({'scope': label, 'path': str(exc)})
        except (ValueError, KeyError, TypeError, IndexError, AssertionError, OSError) as exc:
            self.errors.append({'scope': label, 'error': type(exc).__name__, 'message': str(exc)})
        return None

    def sources(self, mapping, required=()):
        require(isinstance(mapping, dict) and mapping, 'missing source hashes')
        require(set(required) <= set(mapping), 'missing required producer/dependency hashes')
        for name, digest in mapping.items():
            require(self.digest(name) == digest, f'source changed: {name}')
            if name in self.protocol['source_sha256']:
                require(digest == self.protocol['source_sha256'][name], f'source differs from r2 freeze: {name}')

    def identity(self, identity, arm, seed, checkpoint_sha=None, step=STEP):
        require(identity['seed'] == seed and identity['global_step'] == step, 'evaluation seed/step mismatch')
        expected_config = self.protocol['source_sha256'][f'configs/xkf_repair/{arm}_s{seed}.yaml']
        require(identity['config_sha256'] == expected_config, 'evaluation config mismatch')
        require(self.digest(identity['config']) == expected_config, 'evaluation config changed')
        require(identity['checkpoint_sha256'] == self.digest(identity['checkpoint']), 'evaluation checkpoint changed')
        if checkpoint_sha is not None:
            require(identity['checkpoint_sha256'] == checkpoint_sha, 'evaluation/checkpoint lineage mismatch')

    def frozen(self):
        self.protocol = self.read(self.study / 'protocol.json')
        p = self.protocol
        require(p['study'] == 'XKF_REPAIR_R2_PAIRED_DEBUG' and p['first_endpoint'] == STEP, 'not the fixed r2 endpoint')
        require(p['formal'] is False and p['automatic_full_training'] is False, 'formal/automatic scope changed')
        require(p['additional_full_readiness_safeguards'] == GUARDS, 'r2 safeguards changed')
        require(set(p['lineage']) == set(map(str, SEEDS)) and set(p['paired_arms']) == set(ARMS), 'paired seed/arm selection changed')
        manifest = self.read(self.study / 'frozen_manifest.json')
        require(manifest['protocol_sha256'] == self.digest(self.study / 'protocol.json'), 'protocol changed after freeze')
        require(manifest['source_sha256'] == p['source_sha256'], 'source manifest/protocol disagree')
        self.sources(p['source_sha256'])
        for seed in SEEDS:
            for kind in ('checkpoint', 'actual_random_initial'):
                identity = p['lineage'][str(seed)][kind]
                require(self.digest(identity['path']) == identity['sha256'], f'original {kind} changed for {seed}')
        self.reducers = extracted_functions(ROOT / 'tools/xkf/recovery_probe.py',
            {'METRICS', 'MODES', 'CONDITIONS', 'summarize_values', 'aggregate'}, {'np': np, 'math': math, 'defaultdict': defaultdict})
        self.readiness_fn = extracted_functions(ROOT / 'tools/xkf/debug_eval.py', {'DEBUG_CRITERIA', 'readiness'})['readiness']
        self.accuracy_fn = extracted_functions(ROOT / 'tools/xdl/make_row.py', {'accuracy_aggregate'},
                                              {'math': math, 'METRICS': ACCURACY})['accuracy_aggregate']
        self.old_schedule = self.read(ROOT / 'outputs/xkf/local_recovery_20261015/rollout/protocol.json')
        return {'protocol_sha256': manifest['protocol_sha256'], 'source_file_count': len(p['source_sha256']), 'verified': True}

    def training(self, arm, seed):
        import torch
        torch.set_num_threads(4)
        directory = self.study / f'{arm}_s{seed}'
        metadata = self.read(directory / 'training_metadata.json')
        finish = self.read(directory / 'train_finish.json')
        parent = self.protocol['lineage'][str(seed)]['checkpoint']
        require(metadata['seed'] == seed and metadata['start_step'] == 2000 and metadata['max_steps'] == STEP, 'training clock/seed mismatch')
        require(self.path(metadata['resumed_from']).resolve() == self.path(parent['path']).resolve(), 'wrong original seed2000 parent')
        require(self.digest(metadata['resumed_from']) == parent['sha256'], 'parent SHA mismatch')
        require(metadata['config_sha256'] == self.protocol['source_sha256'][f'configs/xkf_repair/{arm}_s{seed}.yaml'], 'training config mismatch')
        require(metadata['freeze_manifest_sha256'] == self.digest(self.study / 'frozen_manifest.json'), 'training used a different freeze')
        require(metadata['preflight_not_formal'] is True and metadata['formal_training_prohibited'] is True, 'formal training unexpectedly enabled')
        require(metadata['lr'] == LR and metadata['precision'] == 'bf16', 'training LR/precision mismatch')
        require(metadata['effective_batch'] == 1024 and metadata['devices'] == 2
                and metadata['batch_size_per_gpu'] == 512 and metadata['accumulate_grad_batches'] == 1, 'training batch contract mismatch')
        require(len(metadata['train_subjects']) == 9 and len(metadata['train_sequences']) == 72, 'canonical split incomplete')
        require(metadata['split_sha256'] == self.protocol['source_sha256']['data/hand_data51/splits_semkine.json'], 'split identity mismatch')
        require(metadata['learning_rate_override_pending'] is False, 'LR restore was not completed')
        self.sources(metadata['source_sha256'], ('tools/xkf_repair/train.py', 'semkine/xkf_repair.py', 'tools/xkf_repair/data.py'))
        require(finish['status'] == 'preflight_complete' and finish['preflight_not_formal'] is True
                and finish['global_step'] == STEP, 'training endpoint incomplete/wrong')
        require(self.digest(finish['checkpoint']) == finish['checkpoint_sha256'], 'saved checkpoint SHA mismatch')
        require(self.digest(directory / 'config_resolved.yaml') == finish['config_sha256'], 'resolved config changed')
        streams, exposures = {}, {}
        for rank in range(2):
            validate_lr(self.read(directory / f'learning_rate_override_rank{rank}.json'), rank)
            streams[rank] = self.read(directory / f'pair_stream_rank{rank}.json')
            validate_stream(streams[rank], rank)
            exposure = self.read(directory / f'actual_exposure_rank{rank}.json')
            require(exposure['final_global_step'] == STEP and exposure['main_supervised_rows'] == 200 * 512,
                    'actual optimizer exposure denominator/step mismatch')
            require(exposure['self_main_rows'] > 0 and exposure['burn_model_rows'] > 0
                    and exposure['burn_count_attribute_present'] is True, 'missing actual self/burn exposure')
            exposures[rank] = exposure
        checkpoint = torch.load(self.path(finish['checkpoint']), map_location='cpu', weights_only=False)
        require(checkpoint['global_step'] == STEP and checkpoint['hyper_parameters']['SEED'] == seed, 'saved optimizer clock/seed mismatch')
        require(checkpoint['hyper_parameters']['TRAIN']['LR'] == LR, 'saved configured LR mismatch')
        require(all(bool(torch.isfinite(t).all()) for t in checkpoint['state_dict'].values() if torch.is_tensor(t)), 'nonfinite saved model state')
        require(len(checkpoint['optimizer_states']) == 1, 'missing/multiple optimizer states')
        optimizer = checkpoint['optimizer_states'][0]
        require([g['lr'] for g in optimizer['param_groups']] == [LR], 'actual saved optimizer LR mismatch')
        require(len(optimizer['state']) == 96, 'Adam parameter-state coverage changed')
        for state in optimizer['state'].values():
            require(float(state['step']) == STEP, 'Adam moment clock differs from endpoint')
            require(all(bool(torch.isfinite(t).all()) for t in state.values() if torch.is_tensor(t)), 'nonfinite Adam state')
        require(len(checkpoint['lr_schedulers']) == 1, 'missing scheduler state')
        scheduler = checkpoint['lr_schedulers'][0]
        require(scheduler['last_epoch'] == STEP and scheduler['_step_count'] == STEP + 1
                and scheduler['base_lrs'] == [LR] and scheduler['_last_lr'] == [LR], 'saved scheduler clock/LR mismatch')
        del checkpoint
        return {'parent_checkpoint_sha256': parent['sha256'], 'checkpoint_sha256': finish['checkpoint_sha256'],
                'global_step': STEP, 'actual_lr': LR, 'adam_states': 96, 'streams': streams, 'exposure': exposures,
                'verified': True}

    def debug(self, arm, seed, training):
        base = self.study / 'evaluations'
        result = self.read(base / f'{arm}_s{seed}_step{STEP}.json')
        initial = self.read(base / f'{arm}_s{seed}_initial.json')
        self.identity(result['identity'], arm, seed, training['checkpoint_sha256'])
        self.identity(initial['identity'], arm, seed, self.protocol['lineage'][str(seed)]['actual_random_initial']['sha256'], step=0)
        for value in (result, initial):
            self.sources(value['repair_adapter_sha256'], ('tools/xkf_repair/evaluate.py',))
            require(value['formal_gate_assessed'] is False and value['finite_subset_reduction'] is False, 'debug scope/subset changed')
            require(value.get('numeric_valid') is True, 'debug numeric validation failed')
            require(value['bank_sha256'] == self.old_schedule['bank_sha256'], 'debug packet bank changed')
            require(set(value['cohorts']) == {'train', 'check', 'zgz'} and set(value['snippets']) == {'zgz_local', 'zgz_global'}, 'debug cohorts missing')
            for name, count in (('train', 8), ('check', 16), ('zgz', 16)):
                cohort = value['cohorts'][name]
                require(cohort['count'] == count and cohort['finite'] is True, 'debug rows missing/nonfinite')
                require(all(finite(v) for v in cohort['loss'].values()), 'debug loss is nonfinite')
                require(all(m['finite'] is True and m['count'] == count for m in cohort['metrics'].values()), 'debug metric subset/nonfinite')
            for key, expected in {'train_batches': 2, 'batch_size': 4, 'train_start_index': 0, 'check_rows': 8, 'clips': 1, 'steps': 10}.items():
                require(value['sampling'][key] == expected, f'debug sampling changed: {key}')
        expected = self.readiness_fn(result, initial)
        assert_close(result['readiness'], expected, 'original readiness recomputation')
        require(result['training_identity']['initial_checkpoint']['sha256'] == initial['identity']['checkpoint_sha256'], 'debug initial identity mismatch')
        return result

    def rollout(self, arm, seed, training):
        directory = self.study / 'evaluations' / f'{arm}_s{seed}_step{STEP}_rollout'
        result, protocol = self.read(directory / 'summary.json'), self.read(directory / 'protocol.json')
        self.identity(result['identity'], arm, seed, training['checkpoint_sha256'])
        self.sources(result['source_sha256'], ('tools/xkf_repair/evaluate.py',))
        require(result['protocol_sha256'] == self.digest(directory / 'protocol.json'), 'rollout protocol changed')
        require(protocol['clips'] == self.old_schedule['clips'] and protocol['excluded'] == self.old_schedule['excluded'], 'rollout sampling/censoring changed')
        require(protocol['conditions'] == self.old_schedule['conditions'] and protocol['modes'] == self.old_schedule['modes'], 'rollout conditions/modes changed')
        require(protocol['bank_sha256'] == self.old_schedule['bank_sha256'] and protocol['source_sha256'] == result['source_sha256'], 'rollout source/bank mismatch')
        require(protocol['checkpoint_sha256'] == training['checkpoint_sha256'] and protocol['config_sha256'] == result['identity']['config_sha256'], 'rollout declared identity mismatch')
        require(protocol['inference_targets'] == 'zero' and protocol['tf32'] is False
                and protocol['finite_subset_reduction'] is False, 'rollout inference contract changed')
        require(result['partial'] is False and result['clip_count'] == 112 and result['formal_gate_assessed'] is False, 'rollout incomplete/formal')
        require(result['metrics'] == list(self.reducers['METRICS']), 'rollout metric definition changed')
        require(set(result['parity']) == {'zgz_local', 'zgz_global'}, 'missing rollout parity checks')
        for value in result['parity'].values():
            require(value['translation_max_mm'] <= .2 and value['pose_max_rad'] <= .002, 'rollout serial parity failed')
        rawfiles = sorted((directory / 'raw').glob('*.npz'))
        require(len(rawfiles) == 112, 'raw rollout missing/extra clips')
        records, packet_hashes = [], []
        for index, clip in enumerate(protocol['clips']):
            path = directory / 'raw' / f'{index:03d}_{clip["cohort"]}_run{clip["run_id"]}.npz'
            self.digest(path)
            with np.load(path, allow_pickle=False) as raw:
                n = clip['observed_steps']
                require(raw['predictions'].shape == (n, 9, 6, 51) and raw['metrics'].shape == (n, 9, 6, len(result['metrics'])), 'raw rollout dimensions differ')
                require(all(np.isfinite(raw[k]).all() for k in raw.files if raw[k].dtype.kind in 'fc'), 'nonfinite raw rollout; no subset reduction allowed')
                packet_hashes.append(self.packet_fingerprint(raw))
                values = raw['metrics']
                for ci, condition in enumerate(protocol['conditions']):
                    for mi, mode in enumerate(protocol['modes']):
                        summarize = self.reducers['summarize_values']
                        v = values[:, ci, mi]
                        records.append(dict(clip, condition=condition[0], mode=mode,
                                            mean=summarize(v), first=summarize(v[:1]), endpoint=summarize(v[-1:]),
                                            lnes_first_ms=int(raw['lnes_window_ms'][0])))
        perclip_path = directory / 'per_clip.jsonl'
        self.digest(perclip_path)
        perclip = [json.loads(line) for line in perclip_path.read_text().splitlines()]
        assert_close(perclip, records, 'rollout perclip/raw')
        recomputed = self.reducers['aggregate'](records)
        assert_close(result['aggregate'], recomputed, 'rollout aggregate/raw')
        require(len(recomputed) == 432, 'rollout full/all cohorts incomplete')
        return {'result': result, 'index': self.rollout_index(result), 'packet_hashes': packet_hashes,
                'finite_raw_frames': 48708, 'trajectories': len(records), 'complete': True}

    @staticmethod
    def packet_fingerprint(raw):
        h = hashlib.sha256()
        for key in ('truth', 'clean', 'end_ms', 'packet_ids', 'lnes_window_ms', 'n_raw_events'):
            x = np.ascontiguousarray(raw[key])
            h.update(key.encode()); h.update(str((x.dtype, x.shape)).encode()); h.update(x.tobytes())
        return h.hexdigest()

    @staticmethod
    def rollout_index(result):
        return {(x['cohort'], x['condition'], x['mode'], x['scope']): x for x in result['aggregate']}

    def direct(self, arm, seed, training):
        directory = self.study / 'evaluations' / f'{arm}_s{seed}_step{STEP}_direct'
        result = self.read(directory / 'result.json')
        self.identity(result['identity'], arm, seed, training['checkpoint_sha256'])
        self.sources(result['repair_adapter_sha256'], ('tools/xkf_repair/evaluate.py',))
        self.sources(result['source_sha256'])
        require(result['complete'] is True and result['partial_smoke'] is False
                and result['formal_gate_assessed'] is False and result['finite_subset_reduction'] is False, 'direct probe incomplete/formal/subset')
        require(not result['failures'] and result['bank_sha256'] == self.old_schedule['bank_sha256'], 'direct failed rows/bank mismatch')
        require(result['preregistration_sha256'] == self.digest(directory / 'preregistration.json'), 'direct preregistration changed')
        require(set(result['sequences']) == set(NROWS) and len(result['raw_files']) == 16, 'direct cohorts/windows incomplete')
        files = {self.path(x['path']).name: x for x in result['raw_files']}
        for sequence, count in NROWS.items():
            require(set(result['sequences'][sequence]) == {'50', '100', '150', '200'}, 'direct windows missing')
            for window in (50, 100, 150, 200):
                descriptor = files[f'{sequence}_W{window}.npz']
                require(self.digest(descriptor['path']) == descriptor['sha256'], 'direct raw SHA mismatch')
                with np.load(self.path(descriptor['path']), allow_pickle=False) as raw:
                    require(raw['success'].shape == (count,) and raw['success'].all(), 'direct failed/missing raw rows')
                    require(raw['success'].dtype.kind == 'b', 'direct success mask is not boolean')
                    require(all(raw[k].ndim >= 1 and len(raw[k]) == count for k in raw.files
                                if k not in ('camera_K', 'beta')), 'direct array has a truncated/mismatched row denominator')
                    require(raw['direct_state'].shape == (count, 51) and raw['truth'].shape == (count, 51), 'direct state shape mismatch')
                    require(all(np.isfinite(raw[k]).all() for k in raw.files if raw[k].dtype.kind in 'fc'), 'nonfinite direct raw arrays')
                    require(len(set(raw['packet_id'])) == count, 'duplicate direct packet rows')
                    summary = result['sequences'][sequence][str(window)]['all']
                    require(summary['n'] == count and summary['failed_rows'] == 0, 'direct summary denominator mismatch')
                    expected_variants = {f'{weight}__{variant}' for weight in ('learned', 'equal')
                                         for variant in ('predpose_pred2d', 'GTpose_pred2d', 'predpose_GT2d', 'GTpose_GT2d')}
                    require(set(summary['variants']) == expected_variants, 'direct decomposition variant missing')
                    for variant, metrics in summary['variants'].items():
                        error = raw[f'{variant}__t_error_mm']
                        assert_close(metrics['t_rmse_mm'], float(np.sqrt(np.mean(np.sum(error ** 2, -1)))), 'direct translation RMSE')
                        wrist = raw[f'{variant}__wrist_error_mm']
                        assert_close(metrics['physical_wrist_rmse_mm'], float(np.sqrt(np.mean(np.sum(wrist ** 2, -1)))), 'direct wrist RMSE')
                        assert_close(metrics['t_depth_bias_mm'], float(error[:, 2].mean()), 'direct translation depth bias')
                        assert_close(metrics['physical_wrist_depth_bias_mm'], float(wrist[:, 2].mean()), 'direct wrist depth bias')
                        for metric in ('abs_mpjpe_mm', 'ra_mpjpe_mm', 'root_deg'):
                            require(metrics[metric]['count'] == count and metrics[metric]['nonfinite_count'] == 0, 'direct finite subset in summary')
                            assert_close(metrics[metric]['mean'], float(raw[f'{variant}__{metric}'].mean()), 'direct metric mean')
        return result

    def main_row(self, arm, seed, training):
        path = self.row_dir / f'xkf_r2_{arm}_s{seed}_step{STEP}_main_row.json'
        row = self.read(path)
        validate_main_row(row, seed)
        self.identity(row['provenance'], arm, seed, training['checkpoint_sha256'])
        self.sources(row['provenance']['source_sha256'], ('tools/xkf_repair/evaluate.py',))
        raw = self.read(str(path) + '.raw_accuracy.json')
        require(raw['complete'] is True and raw['numeric_valid'] is True and raw['finite_subset_reduction'] is False, 'row raw sequences incomplete/nonfinite')
        self.identity(raw['identity'], arm, seed, training['checkpoint_sha256'])
        require(raw['protocol'] == row['protocol'], 'row/raw protocol mismatch')
        require(len(raw['raw_sequences']) == 2, 'missing raw sequence')
        require({s['seq'] for s in raw['raw_sequences']} == {'zgz_local', 'zgz_global'}, 'duplicate/missing raw sequence')
        for sequence in raw['raw_sequences']:
            name = sequence['seq']
            require(name in ('zgz_local', 'zgz_global') and sequence['n_frames'] == NROWS[name], 'wrong full sequence denominator')
            require(all(finite(sequence[k]) for k in ACCURACY), 'nonfinite complete sequence')
            assert_close({k: sequence[k] for k in ACCURACY}, row['per_seed'][str(seed)]['per_seq'][name], 'row/raw accuracy')
        recomputed = self.accuracy_fn(raw['raw_sequences'])
        for group in ('overall', 'local', 'global'):
            assert_close(row['per_seed'][str(seed)][group], recomputed[group], f'row/recomputed {group}')
        return row

    def baseline(self, seed):
        parent = self.protocol['lineage'][str(seed)]['checkpoint']['sha256']
        debug = self.read(ROOT / f'outputs/xkf/debug_20261015_r1/pipeline/evaluations/xkf_s{seed}_step2000.json')
        directory = ROOT / f'outputs/xkf/local_recovery_20261015/rollout/s{seed}'
        rollout = self.read(directory / 'summary.json')
        direct = self.read(ROOT / f'outputs/xkf/local_recovery_20261015/direct_s{seed}/result.json')
        row = self.read(self.row_dir / f'xkf_debug2000_s{seed}_main_row.json')
        for item in (debug, rollout, direct):
            require(item['identity']['seed'] == seed and item['identity']['global_step'] == 2000
                    and item['identity']['checkpoint_sha256'] == parent, 'retained old2000 identity mismatch')
        validate_main_row(row, seed, step=2000)
        require(row['per_seed'][str(seed)]['checkpoint_sha256'] == parent, 'retained old row lineage mismatch')
        require(rollout['partial'] is False and rollout['clip_count'] == 112 and len(rollout['aggregate']) == 432, 'retained old rollout incomplete')
        packets = []
        for index, clip in enumerate(self.old_schedule['clips']):
            path = directory / 'raw' / f'{index:03d}_{clip["cohort"]}_run{clip["run_id"]}.npz'
            self.digest(path)
            with np.load(path, allow_pickle=False) as raw:
                packets.append(self.packet_fingerprint(raw))
        return {'debug': debug, 'rollout': rollout, 'direct': direct, 'row': row, 'packet_hashes': packets}

    def intervention_baseline(self, arm, seed):
        """Retain the zero-update effect of each configured inference contract."""
        result = self.read(self.study / 'evaluations' / f'{arm}_s{seed}_step2000.json')
        initial = self.read(self.study / 'evaluations' / f'{arm}_s{seed}_initial.json')
        self.identity(result['identity'], arm, seed, self.protocol['lineage'][str(seed)]['checkpoint']['sha256'], step=2000)
        self.identity(initial['identity'], arm, seed, self.protocol['lineage'][str(seed)]['actual_random_initial']['sha256'], step=0)
        self.sources(result['repair_adapter_sha256'], ('tools/xkf_repair/evaluate.py',))
        require(result['formal_gate_assessed'] is False and result['finite_subset_reduction'] is False
                and result['numeric_valid'] is True, 'step2000 intervention baseline invalid')
        assert_close(result['readiness'], self.readiness_fn(result, initial), 'step2000 readiness recomputation')
        return {'identity': result['identity'], 'readiness': result['readiness'], 'cohorts': result['cohorts'],
                'snippets': {name: {k: v for k, v in value.items() if k != 'traces'} for name, value in result['snippets'].items()},
                'scope': 'configured inference at original step2000; no new optimizer update'}

    def run(self):
        frozen = self.attempt('frozen_protocol', self.frozen)
        output = {'kind': 'XKF_R2_PAIRED_DEBUG_ASSESSMENT', 'created_utc': datetime.now(timezone.utc).isoformat(),
                  'formal_gate_assessed': False, 'automatic_full_training': False,
                  'full_training_authorized_by_this_assessment': False, 'exact_optimizer_step': STEP,
                  'assessor_sha256': ASSESSOR_SHA256, 'frozen': frozen, 'arms': {}, 'old2000': {},
                  'paired_data': {}, 'merged_main_rows': {}, 'intervention_before_continuation': {},
                  'known_limits': getattr(self, 'protocol', {}).get('known_limits', [])}
        training, debug, rollout, direct, rows = ({arm: {} for arm in ARMS} for _ in range(5))
        old = {}
        if frozen:
            for seed in SEEDS:
                old[seed] = self.attempt(f'old2000/{seed}', lambda s=seed: self.baseline(s))
            for arm in ARMS:
                output['arms'][arm] = {}
                output['intervention_before_continuation'][arm] = {}
                for seed in SEEDS:
                    label = f'{arm}/{seed}'
                    output['intervention_before_continuation'][arm][str(seed)] = self.attempt(
                        f'{label}/step2000_configured', lambda a=arm, s=seed: self.intervention_baseline(a, s))
                    training[arm][seed] = self.attempt(f'{label}/training', lambda a=arm, s=seed: self.training(a, s))
                    t = training[arm][seed]
                    entry = {'training': None if t is None else {k: v for k, v in t.items() if k != 'streams'}}
                    output['arms'][arm][str(seed)] = entry
                    if t is None:
                        continue
                    debug[arm][seed] = self.attempt(f'{label}/debug', lambda a=arm, s=seed, tr=t: self.debug(a, s, tr))
                    rollout[arm][seed] = self.attempt(f'{label}/rollout', lambda a=arm, s=seed, tr=t: self.rollout(a, s, tr))
                    direct[arm][seed] = self.attempt(f'{label}/direct', lambda a=arm, s=seed, tr=t: self.direct(a, s, tr))
                    rows[arm][seed] = self.attempt(f'{label}/main_row', lambda a=arm, s=seed, tr=t: self.main_row(a, s, tr))
                    entry['readiness'] = debug[arm][seed]['readiness'] if debug[arm][seed] else None
                    entry['rollout'] = ({k: v for k, v in rollout[arm][seed].items() if k not in ('result', 'index', 'packet_hashes')}
                                        if rollout[arm][seed] else None)
                    entry['direct_complete_all_rows'] = direct[arm][seed] is not None
                    entry['main_row_complete_2590'] = rows[arm][seed] is not None
            for seed in SEEDS:
                if all(training[a].get(seed) for a in ARMS):
                    def paired(s=seed):
                        a, b = (training[arm][s] for arm in ARMS)
                        require(a['parent_checkpoint_sha256'] == b['parent_checkpoint_sha256'], 'paired parent checkpoints differ')
                        require(a['streams'] == b['streams'], 'paired raw main/prefix/sample hashes differ')
                        return {'identical_parent': True, 'ranks': 2, 'recorded_batches_per_rank': 3,
                                'main_prefix_sample_hashes_identical': True,
                                'scope': 'first three batches per rank audited; later counter stream guaranteed by frozen data contract'}
                    output['paired_data'][str(seed)] = self.attempt(f'paired/{seed}', paired)
                if old.get(seed):
                    output['old2000'][str(seed)] = {'checkpoint_sha256': self.protocol['lineage'][str(seed)]['checkpoint']['sha256'],
                                                   'readiness': old[seed]['debug']['readiness'], 'retained': True}
                    for arm in ARMS:
                        if rollout[arm].get(seed):
                            self.attempt(f'{arm}/{seed}/rollout_packet_match_old2000', lambda a=arm, s=seed:
                                require(rollout[a][s]['packet_hashes'] == old[s]['packet_hashes'], 'rollout input clocks/GT/event counts changed'))
            ready_inputs = all(debug[a].get(s) and rollout[a].get(s) for a in ARMS for s in SEEDS)
            if ready_inputs:
                output['debug_readiness_safeguards'] = self.attempt('expanded_guards', lambda: expanded_guards(
                    {a: {s: rollout[a][s]['index'] for s in SEEDS} for a in ARMS},
                    {a: {s: debug[a][s]['readiness'] for s in SEEDS} for a in ARMS}))
            if all(old.get(s) and all(rollout[a].get(s) and direct[a].get(s) and rows[a].get(s) for a in ARMS) for s in SEEDS):
                output['mechanisms'] = self.comparisons(old, rollout, direct, rows)
            self.attempt('frozen_sources_at_finalization', lambda: self.sources(self.protocol['source_sha256']))
            for path, digest in list(self.inputs.items()):
                self.attempt(f'input_immutable/{path}', lambda p=path, d=digest:
                             require(self.digest(p) == d, 'input changed during assessment'))
            self.attempt('assessor_source_at_finalization', lambda:
                         require(sha(__file__) == ASSESSOR_SHA256, 'assessor source changed while running'))
            # Refuse row merging after a failed provenance/numeric check. Missing
            # another evaluation is not a reason to discard a complete arm's row.
            if not self.errors:
                for arm in ARMS:
                    if all(rows[arm].get(s) for s in SEEDS) and all(output['paired_data'].get(str(s)) for s in SEEDS):
                        source_rows = [self.row_dir / f'xkf_r2_{arm}_s{s}_step{STEP}_main_row.json' for s in SEEDS]
                        provenance = {'source_rows': [{'path': str(p), 'sha256': self.digest(p)} for p in source_rows],
                                      'assessor_sha256': output['assessor_sha256'], 'protocol_sha256': frozen['protocol_sha256']}
                        merged = self.attempt(f'merge/{arm}', lambda a=arm, p=provenance: merge_main_rows(rows[a], a, p))
                        if merged:
                            destination = self.row_dir / f'xkf_r2_{arm}_step{STEP}_main_row.json'
                            write_json(destination, merged)
                            output['merged_main_rows'][arm] = {'path': str(destination), 'sha256': sha(destination)}
        output.setdefault('debug_readiness_safeguards', {'status': 'INCOMPLETE'})
        output.update(status='INVALID' if self.errors else 'INCOMPLETE' if self.missing else 'ASSESSED',
                      validation_errors=self.errors, missing=self.missing, input_sha256=self.inputs)
        if output['status'] != 'ASSESSED':
            output['debug_readiness_safeguards']['eligible_for_decision'] = False
        else:
            output['debug_readiness_safeguards']['eligible_for_decision'] = True
        return output

    def comparisons(self, old, rollouts, direct, rows):
        labels = ('old2000', 'control', 'repair')
        result = {'comparison': 'equal seed mean of each complete cohort; signed deltas are repair-control and each continuation-old2000',
                  'rollout': [], 'direct': [], 'main_row': []}
        indexes = {'old2000': {s: self.rollout_index(old[s]['rollout']) for s in SEEDS},
                   **{a: {s: rollouts[a][s]['index'] for s in SEEDS} for a in ARMS}}
        def paired(values):
            return {'per_seed': {str(s): {**{label: values[label][s] for label in labels},
                                         'repair_minus_control': values['repair'][s] - values['control'][s],
                                         'repair_minus_old2000': values['repair'][s] - values['old2000'][s],
                                         'control_minus_old2000': values['control'][s] - values['old2000'][s]} for s in SEEDS},
                    'equal_seed_mean': {label: float(np.mean([values[label][s] for s in SEEDS])) for label in labels}}
        for key in sorted(indexes['old2000'][SEEDS[0]]):
            group = {'cohort': key[0], 'condition': key[1], 'mode': key[2], 'scope': key[3], 'metrics': {}}
            for metric in ('abs_mpjpe_mm', 'ra_mpjpe_mm', 'abs_mpvpe_mm', 'ra_mpvpe_mm',
                           'state_t_rmse_mm', 'physical_wrist_rmse_mm', 'state_depth_error_mm', 'root_error_deg'):
                group['metrics'][metric] = paired({label: {s: indexes[label][s][key]['pooled_frames'][metric] for s in SEEDS} for label in labels})
            result['rollout'].append(group)
        sources = {'old2000': {s: old[s]['direct'] for s in SEEDS}, **direct}
        for sequence in NROWS:
            for window in ('50', '100', '150', '200'):
                variants = sources['old2000'][SEEDS[0]]['sequences'][sequence][window]['all']['variants']
                for variant in variants:
                    group = {'sequence': sequence, 'window_ms': int(window), 'variant': variant, 'n_rows': NROWS[sequence], 'metrics': {}}
                    for metric in ('t_rmse_mm', 'physical_wrist_rmse_mm', 't_depth_bias_mm', 'abs_mpjpe_mm', 'ra_mpjpe_mm', 'root_deg'):
                        values = {label: {} for label in labels}
                        for label in labels:
                            for seed in SEEDS:
                                v = sources[label][seed]['sequences'][sequence][window]['all']['variants'][variant][metric]
                                values[label][seed] = v['mean'] if isinstance(v, dict) else v
                        group['metrics'][metric] = paired(values)
                    result['direct'].append(group)
        for group in ('overall', 'local', 'global'):
            for metric in ACCURACY:
                values = {'old2000': {s: old[s]['row']['per_seed'][str(s)][group][metric] for s in SEEDS},
                          **{a: {s: rows[a][s]['per_seed'][str(s)][group][metric] for s in SEEDS} for a in ARMS}}
                result['main_row'].append({'cohort': group, 'metric': metric, **paired(values)})
        return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=ROOT / 'outputs/xkf/repair_20261015_r2')
    parser.add_argument('--row-dir', type=Path, default=ROOT / 'outputs/semkine')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.study / 'assessment.json'
    assessment = Assessment(args.study.resolve(), args.row_dir.resolve()).run()
    write_json(output, assessment)
    print(json.dumps({'output': str(output), 'status': assessment['status'],
                      'debug_safeguards': assessment['debug_readiness_safeguards']['status'],
                      'missing': len(assessment['missing']), 'errors': len(assessment['validation_errors']),
                      'automatic_full_training': False}))
    return 2 if assessment['status'] == 'INVALID' else 0


if __name__ == '__main__':
    raise SystemExit(main())
