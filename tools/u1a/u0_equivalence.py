#!/usr/bin/env python3
"""U0 algebraic interface equivalence diagnostic; no training or accuracy row.

Fold a trained S38 LN576->64->3 root head and LN576->256->45 finger head
into LN576(no affine)->Linear576,320->GELU->Linear320,48. The final layer
is block diagonal; this preserves the two learned functions, rather than
claiming that all 48 outputs share a newly trained unrestricted hidden state.
The original S38 translation, prev_mlp, encoder, root reference and filters
remain unchanged. Default device is CPU; CUDA must be explicitly requested.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'model')]
from config import load_config
from model import MNISTModel
from semkine import eval_track as ET
from semkine.dataset import sequences_for_split, splits_manifest
from semkine.events import EventPacket, collate_packets
from semkine.lie import so3_exp, so3_log
from tools.x1001.evalx import find_ckpt


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def state_hash(module):
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str((value.dtype, tuple(value.shape))).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


class U0Measurement48(nn.Module):
    """Single folded forward, retaining the original 64+256 hidden subspaces."""
    def __init__(self, root, finger):
        super().__init__()
        for head in (root, finger):
            if len(head) != 4 or not isinstance(head[0], nn.LayerNorm) or not isinstance(head[1], nn.Linear) or not isinstance(head[2], nn.GELU) or not isinstance(head[3], nn.Linear):
                raise ValueError('expected LayerNorm -> Linear -> GELU -> Linear')
        dim = root[1].in_features
        if dim != 576 or finger[1].in_features != dim or root[1].out_features != 64 or finger[1].out_features != 256 or root[3].out_features != 3 or finger[3].out_features != 45:
            raise ValueError('expected the trained S38 576/64/256/3/45 contract')
        if root[0].normalized_shape != (dim,) or finger[0].normalized_shape != (dim,) or root[0].eps != finger[0].eps or root[2].approximate != finger[2].approximate:
            raise ValueError('LayerNorm domains/epsilon and GELU must agree')
        self.norm = nn.LayerNorm(dim, eps=root[0].eps, elementwise_affine=False)
        self.first = nn.Linear(dim, 320)
        self.act = nn.GELU(approximate=root[2].approximate)
        self.last = nn.Linear(320, 48)
        self.to(device=root[1].weight.device, dtype=root[1].weight.dtype)
        with torch.no_grad():
            first_w, first_b = [], []
            for head in (root, finger):
                ln, fc = head[0], head[1]
                if not ln.elementwise_affine or fc.bias is None or head[3].bias is None:
                    raise ValueError('S38 affine LayerNorm and both linear biases are required')
                # LN(x)=z*gamma+beta; fc(LN(x))=(W*gamma)z+(b+W beta).
                first_w.append(fc.weight * ln.weight.unsqueeze(0))
                first_b.append(fc.bias + fc.weight @ ln.bias)
            self.first.weight.copy_(torch.cat(first_w))
            self.first.bias.copy_(torch.cat(first_b))
            self.last.weight.zero_()
            self.last.weight[:3, :64].copy_(root[3].weight)
            self.last.weight[3:, 64:].copy_(finger[3].weight)
            self.last.bias.copy_(torch.cat([root[3].bias, finger[3].bias]))

    def forward(self, pooled):
        return self.last(self.act(self.first(self.norm(pooled))))


class _CallCache:
    def __init__(self, fused):
        self.fused = fused
        self.reset()

    def reset(self):
        self.raw = None
        self.input = None
        self.calls = 0


class _RootAdapter(nn.Module):
    def __init__(self, cache):
        super().__init__()
        self.fused = cache.fused  # register the single 48D measurement module once
        self.cache = cache

    def forward(self, pooled):
        if self.cache.raw is not None:
            raise RuntimeError('reset the U0 cache before each complete forward_packet call')
        self.cache.input = pooled
        self.cache.raw = self.fused(pooled)
        self.cache.calls += 1
        return self.cache.raw[:, :3]


class _FingerAdapter(nn.Module):
    def __init__(self, cache):
        super().__init__()
        self.cache = cache

    def forward(self, pooled):
        if self.cache.raw is None or self.cache.input is not pooled:
            raise RuntimeError('root/finger must consume the same pooled tensor in one forward')
        return self.cache.raw[:, 3:]


def maxabs(a, b):
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('non-finite equivalence result')
    return float((a.double() - b.double()).abs().max()) if a.numel() else 0.0


def rotation_difference_deg(a, b):
    relative = so3_exp(a.double()) @ so3_exp(b.double()).transpose(-1, -2)
    return float(torch.rad2deg(so3_log(relative).norm(dim=-1)).max())


def raw_rotation_difference_deg(a, b, reference):
    ar = so3_exp(a.double()) @ reference.double()
    br = so3_exp(b.double()) @ reference.double()
    return float(torch.rad2deg(so3_log(ar @ br.transpose(-1, -2)).norm(dim=-1)).max())


def update_max(target, values):
    for name, value in values.items():
        target[name] = max(target.get(name, 0.0), value)


class Pair:
    def __init__(self, original):
        self.original = original
        self.folded = copy.deepcopy(original)
        self.fused = U0Measurement48(original.root_abs_head, original.finger_abs_head)
        self.cache = _CallCache(self.fused)
        self.folded.root_abs_head = _RootAdapter(self.cache)
        self.folded.finger_abs_head = _FingerAdapter(self.cache)
        self.raw = {}
        self.handles = []
        for name, key in [('root_abs_head', 'root'), ('finger_abs_head', 'finger')]:
            def hook(_module, _inputs, output, key=key):
                self.raw[key] = output.detach().clone()
            self.handles.append(getattr(original, name).register_forward_hook(hook))
        for model in (self.original, self.folded):
            model.requires_grad_(False)
            model.eval()
            model.event_encoder.eval()

    def mode(self, training):
        for model in (self.original, self.folded):
            model.train(training)
            model.event_encoder.eval()  # explicitly freeze S38 BN even during train-path checks

    @torch.inference_mode()
    def forward(self, batch):
        self.raw.clear()
        self.cache.reset()
        original = self.original.forward_packet(batch)
        original_pool = self.original.event_encoder.pooled.detach().clone()
        folded = self.folded.forward_packet(batch)
        if self.cache.calls != 1 or set(self.raw) != {'root', 'finger'}:
            raise RuntimeError('one fused measurement call / both original hooks required')
        original_raw = torch.cat([self.raw['root'], self.raw['finger']], dim=-1)
        raw = self.cache.raw.detach().clone()
        values = {'raw48_maxabs_rad': maxabs(original_raw, raw),
                  'raw_root_maxabs_rad': maxabs(original_raw[:, :3], raw[:, :3]),
                  'raw_finger_maxabs_rad': maxabs(original_raw[:, 3:], raw[:, 3:]),
                  'raw_root_rotation_maxdiff_deg': raw_rotation_difference_deg(original_raw[:, :3], raw[:, :3], self.original.root_ref_R),
                  'full51_maxabs': maxabs(original, folded),
                  'translation_maxabs_m': maxabs(original[:, :3], folded[:, :3]),
                  'filtered_root_maxabs_rad': maxabs(original[:, 3:6], folded[:, 3:6]),
                  'filtered_root_rotation_maxdiff_deg': rotation_difference_deg(original[:, 3:6], folded[:, 3:6]),
                  'filtered_finger_maxabs_rad': maxabs(original[:, 6:], folded[:, 6:]),
                  'encoder_pool_maxabs': maxabs(original_pool, self.folded.event_encoder.pooled)}
        empty = batch.counts <= 0
        if empty.any():
            if not torch.equal(original[empty], batch.prev_state[empty]) or not torch.equal(folded[empty], batch.prev_state[empty]):
                raise AssertionError('empty full51 must hold previous state bitwise')
        self.last_original = original.detach().clone()
        return values, original_raw, original_pool

    def close(self):
        for handle in self.handles:
            handle.remove()


@torch.inference_mode()
def random_pool_check(pair, device):
    generator = torch.Generator(device='cpu').manual_seed(1701)
    pooled = torch.cat([torch.randn(128, 576, generator=generator) * scale + 0.37 for scale in (0.1, 1.0, 10.0)]).to(device)
    old = torch.cat([pair.original.root_abs_head(pooled), pair.original.finger_abs_head(pooled)], dim=-1)
    new = pair.fused(pooled)
    # Independent high precision folding checks the algebra separately from CUDA accumulation.
    r64 = copy.deepcopy(pair.original.root_abs_head).cpu().double()
    f64 = copy.deepcopy(pair.original.finger_abs_head).cpu().double()
    fused64 = U0Measurement48(r64, f64)
    p64 = pooled.cpu().double()
    exact = maxabs(torch.cat([r64(p64), f64(p64)], -1), fused64(p64))
    blockzero = bool((pair.fused.last.weight[:3, 64:] == 0).all() and (pair.fused.last.weight[3:, :64] == 0).all())
    if exact > 1e-11 or not blockzero:
        raise AssertionError('FP64 fold algebra/block diagonal failed')
    return {'samples': len(pooled), 'nonzero_pool': True, 'fp32_raw48_maxabs_rad': maxabs(old, new),
            'fp64_raw48_maxabs_rad': exact, 'trained_original_raw_rms': float(old.double().square().mean().sqrt()),
            'block_diagonal_cross_weights_exact_zero': blockzero}


def packets_for_sequence(cfg, seq_id, seq, directory, max_windows):
    root = Path(cfg['DATA']['ROOT'])
    events, offsets, aux, pos51 = ET.load_sequence(root, directory, seq)
    tsub = np.load(root / directory / f'{seq}_tsub.npy', mmap_mode='r')
    betas = np.asarray(aux['betas'], dtype=np.float32).copy()
    camera = np.asarray(aux['camera_K'], dtype=np.float32).reshape(3, 3).copy()
    records = [(run, int(end)) for run, (a, b) in enumerate(np.asarray(aux['valid_runs_ms'], dtype=np.int64).reshape(-1, 2)) for end in np.arange(a + 49, b, 50, dtype=np.int64)]
    total = len(records)
    if max_windows:
        records = records[:max_windows]
    packets, frames = [], []
    for index, (run, end) in enumerate(records):
        start = end - 49
        ev5 = ET._window_events(events, offsets, tsub, end, 50)
        packets.append(EventPacket(events=ev5[:, 1:].copy(), sequence_id=seq_id, t_start_us=start * 1000,
            t_end_us=(end + 1) * 1000, is_sequence_start=False, is_sequence_end=False,
            target=pos51[end].copy(), prev_state=pos51[start].copy(), betas=betas, camera_K=camera))
        frames.append([seq, run, end, index, len(ev5)])
    aux.close()
    return packets, frames, total


@torch.inference_mode()
def window_checks(pair, cfg, seqs, device, batch_size, max_windows):
    overall, by_context = {}, {'GT_start': {}, 'noised_start': {}}
    seqrows, allframes, protocol_total = [], [], 0
    contexts_different_T = 0.0
    context_raw_diff = 0.0
    empty_checks = {}
    sigma = np.array([0.05] * 3 + [0.3] * 48, dtype=np.float32)
    for sid, (seq, directory) in enumerate(seqs):
        packets, frames, total = packets_for_sequence(cfg, sid, seq, directory, max_windows)
        protocol_total += total
        allframes.extend(frames)
        rng = np.random.default_rng(1701 + sid)
        noise = rng.normal(size=(len(packets), 51)).astype(np.float32) * sigma
        seqmax = {}
        for offset in range(0, len(packets), batch_size):
            items = packets[offset:offset + batch_size]
            gt_batch = collate_packets(items).to(device)
            gtvalues, gt_raw, gt_pool = pair.forward(gt_batch)
            update_max(seqmax, gtvalues)
            update_max(by_context['GT_start'], gtvalues)
            gt_T = pair.last_original[:, :3].detach().clone()
            noised = [copy.copy(p) for p in items]
            for k, p in enumerate(noised):
                p.prev_state = p.prev_state + noise[offset + k]
            nv, nraw, npool = pair.forward(collate_packets(noised).to(device))
            update_max(seqmax, nv)
            update_max(by_context['noised_start'], nv)
            context_raw_diff = max(context_raw_diff, maxabs(gt_raw, nraw))
            if not torch.equal(gt_pool, npool):
                raise AssertionError('encoder pooled evidence changed with previous state')
            contexts_different_T = max(contexts_different_T, maxabs(gt_T, pair.last_original[:, :3]))
        if packets and sid == 0:
            live = copy.copy(packets[0])
            empty = copy.copy(live)
            empty.events = np.zeros((0, 4), dtype=np.float32)
            empty.prev_state = live.prev_state + noise[0]
            for training in (False, True):
                pair.mode(training)
                for name, items in [('empty_B1', [empty]), ('empty_B2', [empty, empty]), ('mixed_B2', [empty, live]), ('live_B1', [live])]:
                    vals, _, _ = pair.forward(collate_packets(items).to(device))
                    empty_checks[f'{"train" if training else "eval"}_{name}'] = vals
            pair.mode(False)
        update_max(overall, seqmax)
        seqrows.append({'sequence': seq, 'protocol_windows': total, 'sampled_windows': len(packets),
                        'empty_windows': sum(len(p.events) == 0 for p in packets), 'maxima': seqmax})
        print(f'U0 {seq}: {len(packets)}/{total} windows, GT+noise contexts complete', file=sys.stderr, flush=True)
    if protocol_total != 2590:
        raise AssertionError(f'expected 2590 fixed protocol windows, got {protocol_total}')
    if context_raw_diff != 0:
        raise AssertionError('S38 raw measurement depends on the previous-state context')
    canonical = json.dumps(allframes, separators=(',', ':')).encode()
    return {'protocol_windows_total': protocol_total, 'sampled_windows_total': len(allframes),
            'full_2590_coverage': max_windows == 0, 'window_ms': 50, 'step_ms': 50, 'batch_size': batch_size,
            'frame_count_index_sha256': hashlib.sha256(canonical).hexdigest(),
            'contexts': {'GT_start': 'GT state at end_ms-49; same state supplied to both models',
                         'noised_start': {'seed_per_sequence': [1701, 1702], 'independent_gaussian_std': {'T_m': 0.05, 'root_rad': 0.3, 'finger_rad': 0.3}, 'method': 'add noise to full51 GT start-state; identical between original and U0'}},
            'per_sequence': seqrows, 'maxima': overall, 'context_maxima': by_context,
            'context_raw48_maxabs_rad': context_raw_diff,
            'context_translation_response_maxabs_m': contexts_different_T,
            'empty_and_training_contracts': empty_checks}


def contracts(pair, cfg, before):
    for model in (pair.original, pair.folded):
        if model.u1a_mode != 'off' or not model.prevpos_embed or not model.zero_event_gate or not model.predict_delta:
            raise AssertionError('original S38 T/prev/gate contract required')
        if any(p.requires_grad for p in model.parameters()):
            raise AssertionError('diagnostic parameters must all be frozen')
        if model.event_encoder.training or any(m.training for m in model.event_encoder.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)):
            raise AssertionError('encoder BatchNorm must remain frozen in eval')
    current = {name: state_hash(getattr(pair.original, name)) for name in before}
    equal = {name: before[name] == current[name] == state_hash(getattr(pair.folded, name)) for name in before}
    if not all(equal.values()):
        raise AssertionError('preserved components changed during equivalence diagnostic')
    if not torch.equal(pair.original.root_ref_R, pair.folded.root_ref_R) or pair.original.root_ref_q != pair.folded.root_ref_q:
        raise AssertionError('root reference changed')
    return {'unchanged_components_sha256': current, 'unchanged_components_equal_before_after_and_between_models': equal,
            'encoder_parameters_frozen': True, 'encoder_BN_eval_and_statistics_unchanged': True,
            'T_original_routed_root_head_and_prev_mlp_retained': True,
            'root_reference_retained': cfg['MODEL']['ROOT_REF'],
            'root_filter_gain': pair.original.root_filter_gain, 'finger_filter_gain': pair.original.finger_filter_gain,
            'full51_packing': 'T_m[0:3], global_axis_angle_rad[3:6], MANO_finger_residual_rad[6:51]',
            'raw48_packing': 'root_reference_residual_rad[0:3], finger_residual_rad[3:48]',
            'MANO_mean_contract': 'unchanged original S38 model._fk; no additional mean or FK implementation',
            'optimizer_or_training_steps': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', nargs='+', required=True, type=Path)
    parser.add_argument('--ckpt', choices=['last'], required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-windows', type=int, default=0, help='per-sequence smoke cap; 0 means all 2590')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if min(args.batch_size, args.threads) < 1 or args.max_windows < 0:
        parser.error('positive batch-size/threads and nonnegative max-windows required')
    torch.set_num_threads(args.threads)
    # Avoid TF32 conflating algebraic equivalence with lower precision GEMM differences.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    start = time.monotonic()
    protected = [REPO / 'model/model.py', REPO / 'semkine/u1a_readout.py', REPO / 'docs/U1A_SHARED_MEASUREMENT_PREREG.md']
    protected = [p for p in protected if p.is_file()]
    protected_hashes = {str(p): sha256(p) for p in protected}
    result = {'kind': 'U0_folded_S38_interface_equivalence_no_training', 'device': str(device),
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'precision': 'fp32 full model, independent fp64 random algebra; TF32 disabled',
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
              'script_sha256': sha256(__file__), 'source_files_before_sha256': protected_hashes,
              'not_an_accuracy_arm': True, 'runs': [],
              'limitations': ['U0 preserves the original S38 two hidden subspaces with a block-diagonal last layer; it is not a learned fully shared 17-slot head.',
                  'Equivalent fixed-window outputs are compared with identical GT/noised previous-state contexts; this is not a recursive accuracy evaluation.',
                  'Passing U0 cannot identify padding/LayerNorm as causal; 17-slot normalization, capacity, translation reparameterization and training/generalization still require controlled tests.',
                  'Original S38 finger hidden256 versus U1a hidden64 changes readout expression; the encoder stays fixed.',
                  'zgz is development and test under the fixed protocol; no independent test or final local mesh architecture claim.']}
    for run in args.run_dir:
        run = run.resolve()
        checkpoint, step, _ = find_ckpt(run, 'last')
        config = run / 'config_resolved.yaml'
        cfg = load_config(config)
        state = torch.load(str(checkpoint), map_location='cpu')
        if int(state['global_step']) != step:
            raise AssertionError('numbered explicit-last step and checkpoint metadata disagree')
        del state
        manifest = splits_manifest(cfg) or Path(cfg['DATA']['ROOT']) / 'splits_semkine.json'
        if Path(manifest).name != 'splits_semkine.json':
            raise AssertionError('fixed splits_semkine.json required')
        seqs = sequences_for_split(Path(cfg['DATA']['ROOT']), 'val_core', manifest)
        if [seq for seq, _ in seqs] != ['zgz_global', 'zgz_local']:
            raise AssertionError('fixed zgz val_core ordering required')
        original = MNISTModel.load_from_checkpoint(str(checkpoint), cfg=cfg, map_location='cpu').to(device).eval()
        if original.pose_repr != 'mano_full_axis_angle' or original.root_meas != 'abs' or original.finger_meas != 'abs' or original.u1a_mode != 'off':
            raise AssertionError('trained original S38 full51 absolute measurements required')
        pair = Pair(original)
        before = {name: state_hash(getattr(original, name)) for name in ['event_encoder', 'root_head', 'prev_mlp', 'mano']}
        try:
            random = random_pool_check(pair, device)
            windows = window_checks(pair, cfg, seqs, device, args.batch_size, args.max_windows)
            check = contracts(pair, cfg, before)
            limits = {'fp32_raw48_maxabs_rad': 2e-5, 'full51_maxabs': 5e-5, 'root_rotation_maxdiff_deg': 0.003}
            candidates = [windows['maxima']] + list(windows['empty_and_training_contracts'].values())
            passed = random['fp32_raw48_maxabs_rad'] <= limits['fp32_raw48_maxabs_rad'] and all(v['raw48_maxabs_rad'] <= limits['fp32_raw48_maxabs_rad'] and v['full51_maxabs'] <= limits['full51_maxabs'] and v['raw_root_rotation_maxdiff_deg'] <= limits['root_rotation_maxdiff_deg'] and v['filtered_root_rotation_maxdiff_deg'] <= limits['root_rotation_maxdiff_deg'] and v['translation_maxabs_m'] == 0 and v['encoder_pool_maxabs'] == 0 for v in candidates)
            row = {'run': run.name, 'seed': int(cfg['SEED']), 'step': step, 'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint),
                   'config': str(config), 'config_sha256': sha256(config), 'manifest': str(manifest), 'manifest_sha256': sha256(manifest),
                   'fold': {'input': 576, 'root_hidden': 64, 'finger_hidden': 256, 'fused_hidden': 320, 'output': 48, 'LN_affine': False, 'epsilon': pair.fused.norm.eps,
                            'formula': 'W_first=[W_root*gamma_root;W_finger*gamma_finger]; b_first=[b_root+W_root beta_root;b_finger+W_finger beta_finger]; W_last=blockdiag(W_root_last,W_finger_last)'},
                   'random_pool': random, 'fixed_windows': windows, 'contracts': check, 'tolerances': limits, 'interface_equivalence_pass': passed}
            result['runs'].append(row)
            print(f'U0 {run.name}: pass={passed}, raw48_max={windows["maxima"]["raw48_maxabs_rad"]:.3g}, full51_max={windows["maxima"]["full51_maxabs"]:.3g}', file=sys.stderr, flush=True)
        finally:
            pair.close()
        del pair, original
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    result['source_files_after_sha256'] = {str(p): sha256(p) for p in protected}
    if result['source_files_after_sha256'] != protected_hashes:
        raise AssertionError('protected source changed during diagnostic')
    result['all_interface_checks_pass'] = all(r['interface_equivalence_pass'] for r in result['runs'])
    result['elapsed_s'] = time.monotonic() - start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'wrote {args.output}', file=sys.stderr, flush=True)
    if not result['all_interface_checks_pass']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()