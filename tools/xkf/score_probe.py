#!/usr/bin/env python3
"""CPU-only FK audit of XKF debug scoring; no model forward or changed gate.

The oracle substitutions here only rescore already saved states. They never
produce a deployable result, update a trajectory, or qualify an arm for training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'model')]

from config import load_config
from mano_layer import ManoLayer
from pose_repr import decode_to_mano_inputs
from semkine.lie import so3_exp, so3_log
from tools.xdl import diagnostics as D
from tools.xkf.diagnostics import recovery_clips


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FK:
    def __init__(self, cfg, beta):
        self.mano = ManoLayer(cfg['MANO']['NPZ'], add_mean=False).cpu().eval()
        self.mano.fast_chain = True
        self.beta = beta.cpu().float()

    @torch.no_grad()
    def __call__(self, states):
        x = np.asarray(states, np.float32)
        shape = x.shape[:-1]
        flat = torch.from_numpy(x.reshape(-1, 51).copy())
        result = []
        for part in flat.split(128):
            dec = decode_to_mano_inputs(part, 'mano_full_axis_angle',
                                        self.mano.hands_components, self.mano.hands_mean)
            _, joints = self.mano(self.beta.expand(len(part), -1), dec['global_orient'],
                                  dec['local_full_aa'], dec['transl'])
            result.append(joints.numpy())
        return np.concatenate(result).reshape(*shape, 21, 3).astype(np.float64)


def measures(pred, truth, pj, gj):
    """Every vector is in mm; squared terms are in mm^2, not MPJPE."""
    pred, truth = np.asarray(pred, np.float64), np.asarray(truth, np.float64)
    error = (pj - gj) * 1000.
    wrist = error[..., 0, :]
    relative = error - wrist[..., None, :]
    state_t = (pred[..., :3] - truth[..., :3]) * 1000.
    total = np.sum(error ** 2, axis=-1).mean(axis=-1)
    translation = np.sum(wrist ** 2, axis=-1)
    shape = np.sum(relative ** 2, axis=-1).mean(axis=-1)
    cross = 2. * np.sum(wrist[..., None, :] * relative, axis=-1).mean(axis=-1)
    state_shape = error - state_t[..., None, :]
    state_cross = 2. * np.sum(state_t[..., None, :] * state_shape, axis=-1).mean(axis=-1)
    closure = np.abs(total - translation - shape - cross)
    assert np.isfinite(error).all(), 'Never reduce to a finite subset'
    assert closure.max() < 1e-7, 'Physical wrist decomposition does not close'
    return dict(
        abs_mpjpe_mm=np.linalg.norm(error, axis=-1).mean(axis=-1),
        ra_mpjpe_mm=np.linalg.norm(relative, axis=-1).mean(axis=-1),
        physical_wrist_mm=np.linalg.norm(wrist, axis=-1),
        state_t_mm=np.linalg.norm(state_t, axis=-1),
        joint_mse_mm2=total,
        wrist_translation_mse_mm2=translation,
        wrist_relative_mse_mm2=shape,
        wrist_cross_mm2=cross,
        state_t_translation_mse_mm2=np.sum(state_t ** 2, axis=-1),
        state_t_relative_mse_mm2=np.sum(state_shape ** 2, axis=-1).mean(axis=-1),
        state_t_cross_mm2=state_cross,
        physical_wrist_minus_state_t_max_abs_mm=np.max(np.abs(wrist - state_t), axis=-1),
        decomposition_closure_abs_mm2=closure,
    )


def summarize(arrays):
    summary = {k: float(np.asarray(v).mean()) for k, v in arrays.items()}
    summary['count'] = int(np.asarray(arrays['abs_mpjpe_mm']).size)
    summary['physical_wrist_rmse_mm'] = float(np.sqrt(np.mean(arrays['physical_wrist_mm'] ** 2)))
    summary['state_t_rmse_mm'] = float(np.sqrt(np.mean(arrays['state_t_mm'] ** 2)))
    summary['joint_rms_mm'] = float(np.sqrt(summary['joint_mse_mm2']))
    summary['negative_cross_fraction'] = float(np.mean(arrays['wrist_cross_mm2'] < 0))
    denominator = summary['wrist_translation_mse_mm2'] + summary['wrist_relative_mse_mm2']
    summary['cross_fraction_of_uncoupled_mse'] = summary['wrist_cross_mm2'] / denominator if denominator else 0.
    summary['physical_wrist_minus_state_t_max_abs_mm'] = float(np.max(arrays['physical_wrist_minus_state_t_max_abs_mm']))
    summary['decomposition_closure_abs_mm2'] = float(np.max(arrays['decomposition_closure_abs_mm2']))
    return summary


def concatenate(parts):
    return {key: np.concatenate([np.atleast_1d(part[key]) for part in parts]) for key in parts[0]}


def perturb(initial):
    initial = np.asarray(initial, np.float32)
    out = {name: initial.copy() for name in ('clean', 'z+50mm', 'rx+20deg', 'combined')}
    rotated = so3_log(so3_exp(torch.tensor([[np.pi / 9, 0., 0.]], dtype=torch.float32))
                     @ so3_exp(torch.tensor(initial[None, 3:6])))[0].numpy()
    out['z+50mm'][2] += .05
    out['rx+20deg'][3:6] = rotated
    out['combined'][2] += .05
    out['combined'][3:6] = rotated
    return out


def all_local_starts(cfg, bank):
    meta = bank['sequences']['zgz_local']
    data = D.load_data(cfg, 'zgz_local', meta)
    beta, _ = D.context(data, torch.device('cpu'))
    fk = FK(cfg, beta)
    clips = [v for v in recovery_clips(bank) if v[0] == 'zgz_local']
    assert len(clips) == 41
    per_run, hold_parts, initial_parts, raw = [], {}, {}, {}
    all_initials, all_variant_states = [], []
    for _, name, rows in clips:
        clean = np.array(data['gt'][rows[0]['previous_end_ms']], np.float32)
        truth = np.stack([data['gt'][r['end_ms']] for r in rows])
        gt_joints = fk(truth)
        initial_joints = fk(clean[None])
        variants = perturb(clean)
        entry = dict(run=name, run_id=rows[0]['run_id'], packet_count=len(rows),
                     initial_ms=rows[0]['previous_end_ms'], end_ms=[r['end_ms'] for r in rows],
                     initial_same_time={}, hold_over_available_packets={})
        all_initials.append(clean)
        all_variant_states.extend(variants.values())
        raw[f'{name}_truth'] = truth
        for mode, state in variants.items():
            pred = np.repeat(state[None], len(rows), axis=0)
            joints = np.repeat(fk(state[None]), len(rows), axis=0)
            held = measures(pred, truth, joints, gt_joints)
            at_initial = measures(state[None], clean[None], joints[:1], initial_joints)
            entry['initial_same_time'][mode] = summarize(at_initial)
            entry['hold_over_available_packets'][mode] = summarize(held)
            initial_parts.setdefault(mode, []).append(at_initial)
            hold_parts.setdefault(mode, []).append(held)
            raw[f'{name}_{mode}_state'] = state
        per_run.append(entry)
    starts = np.stack(all_initials)
    variants = np.stack(all_variant_states)
    j = fk(variants)
    zeros = variants.copy()
    zeros[:, :3] = 0
    j0 = fk(zeros)[:, 0]
    shaped = fk.mano.v_template + torch.einsum('bl,mkl->bmk', beta, fk.mano.shapedirs)
    rest_j0 = torch.einsum('bik,ji->bjk', shaped, fk.mano.J_regressor)[0, 0].numpy()
    residual = (j[:, 0] - variants[:, :3] - rest_j0) * 1000.
    zero_residual = (j0 - rest_j0) * 1000.
    assert np.max(np.abs(residual)) < .001
    assert np.max(np.abs(zero_residual)) < .001
    initial_summary = {k: summarize(concatenate(v)) for k, v in initial_parts.items()}
    hold_summary = {k: summarize(concatenate(v)) for k, v in hold_parts.items()}
    # Explicit equal-run averages coexist with pooled packet averages because short
    # runs have fewer observations. Neither changes the existing readiness gate.
    equal_run = {
        mode: {metric: float(np.mean([r['hold_over_available_packets'][mode][metric] for r in per_run]))
               for metric in ('abs_mpjpe_mm', 'ra_mpjpe_mm', 'joint_mse_mm2',
                              'wrist_translation_mse_mm2', 'wrist_relative_mse_mm2', 'wrist_cross_mm2')}
        for mode in hold_parts
    }
    combined_vs_isolated = {}
    for scope in ('initial_same_time', 'hold_over_available_packets'):
        combined_vs_isolated[scope] = {
            'combined_abs_below_z_only_run_count': sum(r[scope]['combined']['abs_mpjpe_mm'] < r[scope]['z+50mm']['abs_mpjpe_mm'] for r in per_run),
            'combined_abs_below_root_only_run_count': sum(r[scope]['combined']['abs_mpjpe_mm'] < r[scope]['rx+20deg']['abs_mpjpe_mm'] for r in per_run),
            'combined_negative_cross_run_count': sum(r[scope]['combined']['wrist_cross_mm2'] < 0 for r in per_run),
        }
    return dict(
        run_count=len(clips), packet_count=sum(len(rows) for _, _, rows in clips),
        short_runs_under_10_packets=sum(len(rows) < 10 for _, _, rows in clips),
        max_packets_per_run=12, same_time_initial_summary=initial_summary,
        held_packet_pooled_summary=hold_summary, held_equal_run_summary=equal_run,
        comparisons=combined_vs_isolated, per_run=per_run,
        coordinate_contract=dict(
            wrist_equation='physical_joint0 = J0(beta) + state_t; same beta in prediction and truth',
            rest_J0_m=rest_j0.tolist(),
            root_or_finger_changes_do_not_move_physical_wrist=True,
            checked_states=len(variants),
            physical_wrist_minus_J0_plus_t_max_abs_mm=float(np.max(np.abs(residual))),
            zero_translation_wrist_minus_J0_max_abs_mm=float(np.max(np.abs(zero_residual))),
            conclusion='debug wrist_rmse uses state_t error, numerically the physical wrist error under the fixed same-beta contract',
        )), raw


def replay(cfg, bank, evaluation_root):
    results, identities, raw = {}, [], {}
    for seed in (3407, 3408, 3409):
        path = evaluation_root / f'xkf_s{seed}_step2000.json'
        saved = json.loads(path.read_text())
        identities.append(dict(path=str(path), sha256=sha(path), identity=saved['identity']))
        per_sequence = {}
        for seq, snippet in saved['snippets'].items():
            assert len(snippet['traces']) == 1
            data = D.load_data(cfg, seq, bank['sequences'][seq])
            beta, _ = D.context(data, torch.device('cpu'))
            fk = FK(cfg, beta)
            trace = snippet['traces'][0]
            truth = np.asarray([s['truth'] for s in trace['steps']], np.float32)
            gj = fk(truth)
            raw[f's{seed}_{seq}_truth'] = truth
            record = dict(clip_id=trace['clip_id'], packet_count=len(truth), modes={})
            for mode in ('i', 'd', 'fused', 'd_zero', 'hold'):
                state = np.asarray([s['predictions'][mode] for s in trace['steps']], np.float32)
                pj = fk(state)
                metrics = measures(state, truth, pj, gj)
                summary = summarize(metrics)
                # Reproduce original unsquared metrics before decomposing.
                original = snippet['metrics'][mode]
                drift = {
                    'abs_mpjpe_mm': abs(summary['abs_mpjpe_mm'] - original['abs_mpjpe_mm']),
                    'ra_mpjpe_mm': abs(summary['ra_mpjpe_mm'] - original['ra_mpjpe_mm']),
                    'wrist_rmse_mm': abs(summary['state_t_rmse_mm'] - original['wrist_rmse_mm']),
                }
                assert max(drift.values()) < .001, (seed, seq, mode, drift)
                original_summary = dict(summary=summary, endpoint=summarize({k: v[-1:] for k, v in metrics.items()}),
                                        original_metric_reproduction_abs_difference=drift,
                                        per_packet={k: v.tolist() for k, v in metrics.items()}, oracle_substitution={})
                for label, pieces in [('t_from_GT', (slice(0, 3),)),
                                      ('root_from_GT', (slice(3, 6),)),
                                      ('t_and_root_from_GT', (slice(0, 6),))]:
                    oracle = state.copy()
                    for piece in pieces:
                        oracle[:, piece] = truth[:, piece]
                    oracle_metrics = measures(oracle, truth, fk(oracle), gj)
                    original_summary['oracle_substitution'][label] = summarize(oracle_metrics)
                    raw[f's{seed}_{seq}_{mode}_{label}'] = oracle
                record['modes'][mode] = original_summary
                raw[f's{seed}_{seq}_{mode}'] = state
            per_sequence[seq] = record
        results[str(seed)] = per_sequence
    return dict(source_evaluations=identities, results=results,
                oracle_scope='Offline GT block substitutions into recorded states only; no inference or recursive replay. Not deployable, not a new training/evaluation result.'), raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/xkf/xkf_s3407.yaml')
    parser.add_argument('--evaluations', default='outputs/xkf/debug_20261015_r1/pipeline/evaluations')
    parser.add_argument('--output', default='outputs/xkf/local_recovery_20261015/scoring')
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    cfg = load_config(args.config)
    bank, banksha = D.load_bank()
    output = Path(args.output)
    if (output / 'score_audit.json').exists():
        raise FileExistsError('Refuse to overwrite a completed scoring audit')
    output.mkdir(parents=True, exist_ok=True)
    starts, raw_starts = all_local_starts(cfg, bank)
    traces, raw_traces = replay(cfg, bank, Path(args.evaluations))
    result = dict(kind='XKF_FK_SCORING_DIAGNOSTIC_ONLY', device='cpu',
                  formal_gate_assessed=False, original_metrics_and_gates_unchanged=True,
                  model_forward_calls=0, config_sha256=sha(args.config),
                  producer_sha256=sha(__file__), packet_bank_sha256=banksha,
                  mano_sha256=sha(cfg['MANO']['NPZ']),
                  squared_error_identity='mean_j ||e_j||^2 = ||delta_wrist||^2 + mean_j ||delta_RA_j||^2 + 2 mean_j delta_wrist dot delta_RA_j',
                  squared_units='mm^2; decomposition is of squared error, never an additive decomposition of MPJPE',
                  state_t_pivot_note='For fixed beta, physical wrist delta = state_t delta. No rotation-dependent wrist offset is present in this MANO implementation.',
                  local_run_starts=starts, saved_trace_replay=traces)
    path = output / 'score_audit.json'
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(output / 'scored_states.npz', **raw_starts, **raw_traces)
    print(json.dumps(dict(output=str(path), local_runs=starts['run_count'],
                          local_packets=starts['packet_count'],
                          coordinate_contract=starts['coordinate_contract'])), flush=True)


if __name__ == '__main__':
    main()
