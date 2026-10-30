#!/usr/bin/env python3
"""Independent six-endpoint / complete paired-stream audit of I-only debug.

Read-only on all training/evaluation artifacts. Reconstructs the sampling clock,
canonical legal index and original GT hashes without forwarding/training a model.
No accuracy gate is asserted here. Missing endpoints are explicitly incomplete.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]
import numpy as np
import torch
from config import load_config
from semkine.dataset import sequences_for_split, splits_manifest
from semkine.xkf_repair import XKFRepairModel

BASE = ROOT / 'outputs/xkf/fusion_20261015_r3/direct_finetune/incremental_finetune'
START, END, UPDATES, RATE = 2400, 2600, 200, .0004


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2 ** 20), b''):
            h.update(block)
    return h.hexdigest()


def tensor_sha(value):
    value = value.detach().cpu().contiguous()
    h = hashlib.sha256(str((tuple(value.shape), str(value.dtype))).encode())
    h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def rng(seed, step, row, stream):
    # Independent copy of the frozen public counter formula, not its sampler.
    key = int.from_bytes(hashlib.sha256(f'XDL-v2|{seed}|{step}|{row}|{stream}'.encode()).digest()[:16], 'little')
    return np.random.default_rng(key)


class Audit:
    def __init__(self, root):
        self.root = root.resolve()
        self.inputs = {}
        self.protocol = self.read(self.root / 'protocol.json')
        self.frozen = self.read(self.root / 'frozen_manifest.json')
        require(self.protocol['kind'] == 'XKF_R3_I_ONLY_PAIRED200_PREREG', 'wrong protocol')
        require(self.protocol['global_clock'] == [START, END] and self.protocol['fixed_updates_per_arm'] == UPDATES, 'wrong protocol endpoint')
        require(self.protocol['I_Adam_clock'] == [2200, 2400] and self.protocol['D_Adam_clock'] == [2400, 2400], 'wrong protocol parameter clocks')
        require(self.protocol['learning_rate'] == RATE and self.protocol['effective_batch'] == 1024, 'wrong protocol LR/batch')
        require(self.protocol['source_sha256'] == self.frozen['source_sha256'], 'protocol/frozen sources differ')
        for name, digest in self.frozen['source_sha256'].items():
            require(self.hash(ROOT / name) == digest, 'frozen source changed: ' + name)
            require(self.hash(self.root / 'source_snapshot' / name) == digest, 'source snapshot differs: ' + name)
        declaration = self.root / 'evaluation_declaration.json'
        require(self.hash(declaration) == self.protocol['evaluation_declaration_sha256'], 'evaluation prereg changed')
        self.index = None
        self.labels = {}

    def hash(self, path):
        path = Path(path).resolve()
        value = sha(path)
        if str(path) in self.inputs:
            require(self.inputs[str(path)] == value, 'input changed between reads: ' + str(path))
        self.inputs[str(path)] = value
        return value

    def read(self, path):
        self.hash(path)
        return json.loads(Path(path).read_text())

    def load(self, path):
        self.hash(path)
        return torch.load(path, map_location='cpu', weights_only=False)

    def data_index(self, cfg):
        sequences = sequences_for_split(Path(cfg['DATA']['ROOT']), 'train', splits_manifest(cfg))
        require(len(sequences) == 72 and {s.split('_')[0] for s, _ in sequences} == {'ch','lfz','lpc','lr','ly','lyh','lyq','ycy','ylf'}, 'canonical9/72 differs')
        self.hash(splits_manifest(cfg))
        if self.index is None:
            chunks = []
            for si, (seq, legacy) in enumerate(sequences):
                base = Path(cfg['DATA']['ROOT']) / legacy / seq
                self.hash(str(base) + '_aux.npz')
                with np.load(str(base) + '_aux.npz', allow_pickle=True) as aux:
                    runs = np.asarray(aux['valid_runs_ms'], dtype=np.int64).reshape(-1, 2)
                for a, b in runs:
                    # Original K4 needs four complete W50 packets (200ms).
                    ends = np.arange(int(a) + 199, int(b), dtype=np.int64)
                    if len(ends):
                        chunks.append(np.column_stack([np.full_like(ends, si), ends, np.full_like(ends, a), np.full_like(ends, b)]))
            self.index = np.concatenate(chunks)
            self.sequences, self.data_root = sequences, Path(cfg['DATA']['ROOT'])
        require(sequences == self.sequences, 'sequence order changed across arms/seeds')
        return sequences

    def label(self, si):
        if si not in self.labels:
            seq, legacy = self.sequences[si]
            path = self.data_root / legacy / (seq + '.meta')
            self.hash(path)
            with path.open('rb') as f:
                require(int(np.fromfile(f, dtype='<i4', count=1)[0]) == 51, 'GT dimension differs')
                raw = np.fromfile(f, dtype=np.dtype([('data', '<f8', 51), ('m0', 'u1'), ('m1', 'u1')]))
            require(bool((raw['m0'] == 4).all() and (raw['m1'] == 13).all()), 'GT delimiters differ')
            self.labels[si] = raw['data'].astype(np.float32)
        return self.labels[si]

    def expected_sample(self, seed, step, row):
        legal = int(rng(seed, step, row, 'clip_selection').integers(len(self.index)))
        si, end, a, b = map(int, self.index[legal])
        width = round(float(np.exp(rng(seed, step, row, 'xkf_window').uniform(np.log(30), np.log(300)))))
        gt = self.label(si)
        truth_hash = hashlib.sha256(np.stack([gt[end - 50], gt[end]]).astype(np.float32).tobytes()).hexdigest()
        return dict(sample_index=step * 1024 + row, optimizer_step=step, global_row=row,
                    sequence_id=si, sequence=self.sequences[si][0], end_ms=end, incoming_ms=end - 50,
                    run_start_ms=a, run_end_ms=b, lnes_window_ms=min(width, end - a + 1),
                    original_GT_endpoints_sha256=truth_hash)

    def arm(self, seed, arm):
        directory = self.root / f'{arm}_s{seed}'
        meta, finish = self.read(directory / 'training_metadata.json'), self.read(directory / 'train_finish.json')
        config = directory / 'config_resolved.yaml'
        self.hash(config)
        cfg = load_config(str(config))
        declaration = self.protocol['declared_configs'][f'{arm}_s{seed}']
        require(self.hash(declaration['path']) == declaration['sha256'] == self.hash(config) == meta['config_sha256'], 'resolved config differs from predeclared treatment')
        original = ROOT / f'configs/xkf_repair/repair_s{seed}.yaml'
        require(self.hash(original) == meta['source_config_sha256'], 'original input config changed')
        expected = load_config(str(original))
        expected['XKF_DATA']['PREFIX_LNES'] = True
        if arm == 'raw':
            expected['AUG']['DOMRAND']['ENABLED'] = False
            expected['AUG']['POLARITY_FLIP'] = False
            expected['AUG']['PIXEL_POLARITY_SWAP'] = False
        expected['TRAIN'].update(MAX_STEPS=END, LR=RATE, OUTPUT_DIR=str(directory), NUM_WORKERS=12, RUN_NAME=f'xkf_ionly_{arm}_s{seed}')
        normalized = lambda value: {k: v for k, v in value.items() if k != '_config_path'}
        require(normalized(expected) == normalized(cfg), 'effective treatment differs from original+declared modifications')
        sequences = self.data_index(cfg)
        require(meta['train_sequences'] == [s for s, _ in sequences], 'training canonical sequence list differs')
        require(meta['coverage']['eligible_main_index_sha256'] == hashlib.sha256(self.index.tobytes()).hexdigest(), 'independent legal index differs')
        require(meta['seed'] == seed and meta['arm'] == arm and not meta['formal'] and not meta['smoke_only'], 'metadata seed/arm/scope differs')
        require(meta['start_step'] == START and meta['max_steps'] == END and meta['updates'] == UPDATES, 'metadata endpoint differs')
        require(meta['lr'] == RATE and meta['effective_batch'] == 1024 and meta['batch_size_per_gpu'] == 512 and meta['devices'] == 2 and meta['precision'] == 'bf16', 'training optimizer/batch/precision differs')
        require(meta['source_sha256'] == finish['source_sha256'] == self.frozen['source_sha256'], 'training source declaration differs')
        parent_declared = self.protocol['parent_checkpoints'][str(seed)]
        parent_path = Path(parent_declared['path']).resolve()
        require(Path(meta['resumed_from']).resolve() == parent_path and self.hash(parent_path) == parent_declared['sha256'] == meta['parent_checkpoint_sha256'] == finish['parent_checkpoint_sha256'], 'parent differs from prereg')
        parent = self.load(parent_path)
        path = directory / f'xkf_ionly_{arm}_s{seed}-step=2600.ckpt'
        saved = self.load(path)
        require(finish['status'] == 'DEBUG200_COMPLETE' and finish['global_step'] == END and finish['checkpoint_sha256'] == self.hash(path), 'missing exact completed checkpoint')
        require(parent['global_step'] == START and saved['global_step'] == END and saved['hyper_parameters']['SEED'] == seed, 'saved step/seed differs')
        require(not saved.get('ema') and 'raw_state_dict' not in saved, 'unexpected EMA weights')
        require(normalized(saved['hyper_parameters']) == normalized(cfg), 'saved effective config differs')
        marker = saved['xkf_incremental_continuation']
        require(marker['start_step'] == START and marker['i_adam_step'] == marker['d_adam_step'] == 2400 and marker['all_bn_and_xkf_statistics_frozen'] and marker['parent_direct_continuation'] == parent['xkf_direct_continuation'], 'parameter-clock lineage marker differs')
        model = XKFRepairModel(cfg=cfg)
        names = [name for name, _ in model.named_parameters()]
        direct = {name for name in names if name.startswith('direct.')}
        buffers = {name for name, _ in model.named_buffers()}
        require(len(names) == 96 and len(direct) == 60, 'parameter identity changed')
        require(set(saved['state_dict']) == set(parent['state_dict']) == set(model.state_dict()), 'state tensor structure changed')
        changed = []
        for name, value in saved['state_dict'].items():
            require(value.shape == parent['state_dict'][name].shape and (not value.is_floating_point() or bool(torch.isfinite(value).all())), 'saved tensor shape/numeric: ' + name)
            if name in direct | buffers:
                require(torch.equal(value, parent['state_dict'][name]), 'frozen D/buffer changed: ' + name)
            elif not torch.equal(value, parent['state_dict'][name]):
                changed.append(name)
        require(bool(changed) and set(changed) <= set(names) - direct, 'updates are not restricted to I parameters')
        require(len(saved['optimizer_states']) == len(parent['optimizer_states']) == 1, 'optimizer count changed')
        old, new = parent['optimizer_states'][0], saved['optimizer_states'][0]
        require(len(old['param_groups']) == len(new['param_groups']) == 1, 'Adam groups changed')
        ids = new['param_groups'][0]['params']
        require(ids == old['param_groups'][0]['params'] and len(ids) == len(new['state']) == len(old['state']) == 96, 'Adam original slots/order changed')
        require(new['param_groups'][0]['lr'] == RATE and new['param_groups'][0]['initial_lr'] == RATE, 'saved actual/base LR differs')
        for name, key in zip(names, ids):
            previous, current = old['state'][key], new['state'][key]
            require(set(previous) == set(current) and int(previous['step']) == (2400 if name in direct else 2200) and int(current['step']) == 2400, 'Adam fields/clock: ' + name)
            for field in ('exp_avg', 'exp_avg_sq'):
                require(current[field].shape == saved['state_dict'][name].shape and bool(torch.isfinite(current[field]).all()), 'Adam moment shape/numeric: ' + name)
            if name in direct:
                for field, value in current.items():
                    require(torch.equal(value, previous[field]) if torch.is_tensor(value) else value == previous[field], 'frozen D Adam changed: ' + name + '/' + field)
        require(len(saved['lr_schedulers']) == 1, 'scheduler count differs')
        scheduler = saved['lr_schedulers'][0]
        require(scheduler['last_epoch'] == END and scheduler['base_lrs'] == scheduler['_last_lr'] == [RATE], 'actual saved scheduler clock/LR differs')
        initial = directory / 'initial_state.ckpt'
        require(self.hash(initial) == meta['initial_checkpoint_sha256'] == self.hash(meta['original_initial_checkpoint']), 'true initial lineage changed')
        initial_saved = self.load(initial)
        require(initial_saved['global_step'] == 0 and initial_saved['hyper_parameters']['SEED'] == seed, 'copied initial is not this seed step0')
        rank_evidence = []
        for rank in (0, 1):
            resume = self.read(directory / f'resume_audit_rank{rank}.json')
            require(resume['parameter_order'] == names and resume['actual_restored_state_matches_parent'], 'actual resume named order differs')
            lr = resume['learning_rate']
            require(lr['configured_lr'] == RATE and lr['actual_after'] == [[RATE]] and lr['adam_state_and_scheduler_clock_preserved'], 'actual override differs')
            require(all(s['last_epoch'] == START and s['base_lrs'] == s['last_lr'] == [RATE] for s in lr['schedulers']), 'resume scheduler differs')
            expected_opt = {name: {field: tensor_sha(value) if torch.is_tensor(value) else value for field, value in old['state'][key].items()} for name, key in zip(names, ids)}
            require(resume['optimizer_state'] == expected_opt, 'actual runtime Adam restore differs from parent')
            expected_frozen = {name: tensor_sha(parent['state_dict'][name]) for name in direct | buffers
                               if name in parent['state_dict']}
            require(resume['frozen_state_sha256'] == expected_frozen, 'runtime start frozen state differs')
            audit = self.read(directory / f'freeze_audit_rank{rank}.json')
            exposure = self.read(directory / f'actual_exposure_rank{rank}.json')
            require(audit['status'] == 'PASS' and audit['global_step'] == END and audit['i_adam_step'] == audit['d_adam_step'] == 2400 and set(audit['changed_parameters']) == set(changed), 'final rank audit differs')
            require(audit['source_sha256'] == self.frozen['source_sha256'], 'rank source differs')
            require(exposure['final_global_step'] == END and exposure['main_supervised_rows'] == UPDATES * 512 and exposure['burn_model_rows'] == UPDATES * 512 * 3 and 0 < exposure['self_main_rows'] <= UPDATES * 512 and exposure['burn_count_attribute_present'], 'actual main/self/burn exposure differs')
            require(audit['main_supervised_rows'] == UPDATES * 512 and audit['burn_model_rows'] == UPDATES * 512 * 3 and audit['i_forward_rows'] == UPDATES * 512 * 4, 'rank forward exposure differs')
            losses = self.read(directory / f'losses_rank{rank}.json')
            require([r['global_step'] for r in losses] == list(range(START + 1, END + 1)) and all(np.isfinite(r['rank_local_log10_I_motion_loss']) for r in losses), 'loss clock/numeric differs')
            rank_evidence.append(dict(rank=rank, learning_rate=lr, exposure=exposure, all_named_parent_Adam_hashes_match=True))
        del saved, parent, initial_saved, model
        return dict(status='PASS', checkpoint=str(path), checkpoint_sha256=self.hash(path), global_step=END,
                    I_Adam_step=2400, D_Adam_step=2400, changed_I_parameters=changed, frozen_D_parameters=60,
                    frozen_saved_buffer_count=len(buffers & set(expected_frozen)),
                    nonpersistent_buffers_not_checkpoint_serialized=sorted(buffers - set(expected_frozen)),
                    named_Adam_slots=96, parent_declared=parent_declared,
                    source_count=len(self.frozen['source_sha256']), rank_evidence=rank_evidence)

    def pair(self, seed):
        dirs = {arm: self.root / f'{arm}_s{seed}' for arm in ('raw', 'aug')}
        counts = 0
        different_lnes = 0
        seq_counts = Counter()
        windows = Counter()
        expected_self = {}
        config = load_config(str(dirs['raw'] / 'config_resolved.yaml'))
        track = config['TRACK']
        pmax = float(track.get('XDL_SELF_P', .5))
        warmup, ramp_end = int(track.get('XDL_WARMUP_START', 500)), int(track.get('XDL_RAMP_END', 2000))
        for rank in (0, 1):
            self_rows = 0
            streams = {arm: self.read(d / f'pair_stream_rank{rank}.json') for arm, d in dirs.items()}
            for arm, stream in streams.items():
                require(len(stream) == UPDATES and [r['optimizer_step'] for r in stream] == list(range(START, END)), arm + ' step stream incomplete')
                require(all(r['rank'] == rank and r['rows'] == 512 for r in stream), 'rank stream shape differs')
            files = {arm: d / f'sample_identities_rank{rank}.jsonl' for arm, d in dirs.items()}
            for path in files.values():
                self.hash(path)
            with files['raw'].open() as rf, files['aug'].open() as af:
                for offset, (rawline, augline) in enumerate(itertools.zip_longest(rf, af)):
                    require(rawline is not None and augline is not None and offset < UPDATES, 'paired complete stream length differs')
                    raw, aug = json.loads(rawline), json.loads(augline)
                    require(raw == aug and raw['optimizer_step'] == START + offset, 'paired sample identities differ')
                    samples = raw['samples']
                    require(len(samples) == 512, 'sample count differs')
                    digest = hashlib.sha256(json.dumps(samples, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                    sr, sa = streams['raw'][offset], streams['aug'][offset]
                    require(digest == sr['sampling_identity_sha256'] == sa['sampling_identity_sha256'], 'batch identity hash differs from complete samples')
                    require(sr['lnes_window_sha256'] == sa['lnes_window_sha256'], 'window treatment differs')
                    different_lnes += sr['main_lnes_sha256'] != sa['main_lnes_sha256']
                    for row, sample in zip(range(rank, 1024, 2), samples):
                        require(sample == self.expected_sample(seed, START + offset, row), 'independent RNG/window/originalGT reconstruction differs at ' + str((seed, START + offset, row)))
                        probability = pmax * max(0., min(1., (START + offset - warmup) / float(ramp_end - warmup)))
                        self_rows += int(rng(seed, START + offset, row, 'self').random() < probability)
                        seq_counts[sample['sequence']] += 1
                        windows[sample['lnes_window_ms']] += 1
                    counts += len(samples)
            expected_self[str(rank)] = self_rows
            for arm, directory in dirs.items():
                exposure = self.read(directory / f'actual_exposure_rank{rank}.json')
                require(exposure['self_main_rows'] == self_rows, arm + ' actual self rows differ from independent counter RNG')
        require(counts == UPDATES * 1024 and different_lnes == UPDATES * 2, 'incomplete paired exposure or missing augmentation difference')
        require(len(seq_counts) == 72, 'some canonical sequences not exposed')
        return dict(status='PASS', paired_rows=counts, independently_reconstructed_rows=counts,
                    original_GT_endpoint_hashes_verified=counts, different_LNES_batches=different_lnes,
                    actual_self_rows_equal_independent_RNG_both_arms=expected_self,
                    sequence_counts=dict(sorted(seq_counts.items())), window_histogram=dict(sorted(windows.items())),
                    scope='Original identities and original GT equal; transformed augmented targets and own self-burn states may differ.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=BASE)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seeds', nargs='+', type=int, choices=(3407, 3408, 3409), default=[3407, 3408, 3409])
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = dict(kind='INDEPENDENT_IONLY_PAIRED200_TRAINING_AUDIT', formal=False, automatic_full_training=False,
                  scope='Implementation/lineage/complete sampling audit only; no effect or readiness pass.',
                  validator_sha256=sha(__file__), seeds={}, errors=[], missing=[])
    audit = None
    try:
        audit = Audit(args.root)
        for seed in args.seeds:
            result['seeds'][str(seed)] = {}
            for arm in ('raw', 'aug'):
                try:
                    result['seeds'][str(seed)][arm] = audit.arm(seed, arm)
                except FileNotFoundError as error:
                    result['missing'].append(str(error))
                except Exception as error:
                    result['errors'].append(dict(seed=seed, arm=arm, error=str(error)))
            if len(result['seeds'][str(seed)]) == 2:
                try:
                    result['seeds'][str(seed)]['pair'] = audit.pair(seed)
                except Exception as error:
                    result['errors'].append(dict(seed=seed, scope='complete_pair', error=str(error)))
            print(json.dumps(dict(seed=seed, completed=list(result['seeds'][str(seed)]), errors=result['errors'], missing=result['missing'])), flush=True)
    except FileNotFoundError as error:
        result['missing'].append(str(error))
    except Exception as error:
        result['errors'].append(dict(scope='setup', error=str(error)))
    result['input_sha256'] = audit.inputs if audit is not None else {}
    for path, digest in result['input_sha256'].items():
        if sha(path) != digest:
            result['errors'].append(dict(scope='final_input_hash', path=path))
    full = args.seeds == [3407, 3408, 3409] and all(set(result['seeds'].get(str(s), {})) == {'raw', 'aug', 'pair'} for s in args.seeds)
    result['status'] = 'INVALID' if result['errors'] else 'INCOMPLETE' if result['missing'] or not full else 'PASS'
    output = args.output or args.root / 'independent_training_audit.json'
    require(not output.exists(), 'refusing to overwrite existing audit')
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(output), status=result['status'], errors=result['errors'], missing=result['missing'])), flush=True)
    return 2 if result['status'] == 'INVALID' else 0


if __name__ == '__main__':
    raise SystemExit(main())
