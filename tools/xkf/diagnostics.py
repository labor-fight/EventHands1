#!/usr/bin/env python3
"""Exact-step6000 XKF G1--G5 audit against the frozen protocol, CPU by default.

Targets are zero at every model call. GT is used only for declared incoming
states and scoring. All packets/trials remain in denominators, including short
runs and numerical failures. Raw predictions accompany the gate report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'model')]

from config import load_config
from mano_layer import ManoLayer
from semkine.dataset import splits_manifest
from semkine.xkf import XKFModel
from tools.xdl import diagnostics as D
from tools.xdl import make_row as ROW
from tools.xdl.prepare import packet_rows
from tools.xkf.runtime import LongWindowEval

NAMES = ('zgz_global', 'zgz_local')
PERTURBATIONS = ('clean', 'z-50mm', 'z+50mm', 'rx-20deg', 'rx+20deg', 'ry-20deg', 'ry+20deg')


def gate(passed, **details):
    return dict(status='PASS' if passed else 'FAIL', passed=bool(passed), **details)


def finite_mean(values):
    values = np.asarray(values, np.float64)
    return float(values.mean()) if values.size and np.isfinite(values).all() else None


def ols_slope(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if x.shape != y.shape or not x.size or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    x, y = x.ravel() - x.mean(), y.ravel() - y.mean()
    denominator = float(x @ x)
    return float(x @ y / denominator) if denominator > 1e-18 else None


def root_error_deg(pred, truth):
    pred, truth = np.broadcast_arrays(np.asarray(pred)[..., 3:6], np.asarray(truth)[..., 3:6])
    shape = pred.shape[:-1]
    a, b = pred.reshape(-1, 3), truth.reshape(-1, 3)
    live = np.isfinite(a).all(-1) & np.isfinite(b).all(-1)
    out = np.full(len(a), np.nan)
    if live.any():
        out[live] = np.degrees((Rotation.from_rotvec(b[live]).inv() * Rotation.from_rotvec(a[live])).magnitude())
    return out.reshape(shape)


def initial_states(state):
    """Clean + six literal frozen perturbations; roots rotate about camera axes."""
    result = np.repeat(np.asarray(state, np.float32)[None], len(PERTURBATIONS), axis=0)
    result[1, 2] -= .05
    result[2, 2] += .05
    base = Rotation.from_rotvec(np.asarray(state[3:6], np.float64))
    for i, (axis, sign) in enumerate(((0, -1), (0, 1), (1, -1), (1, 1)), start=3):
        delta = np.zeros(3)
        delta[axis] = sign * np.deg2rad(20.)
        result[i, 3:6] = (Rotation.from_rotvec(delta) * base).as_rotvec()
    return result


def wrong_time_indices(rows, minimum_ms=1000):
    """Nearest later >= 1 s packet, otherwise nearest earlier >= 1 s, same sequence."""
    ends = np.array([r['end_ms'] for r in rows], np.int64)
    if np.any(np.diff(ends) <= 0):
        raise ValueError('wrong-time controls require chronological unique endpoints')
    result = []
    for end in ends:
        later = int(np.searchsorted(ends, end + minimum_ms))
        earlier = int(np.searchsorted(ends, end - minimum_ms, side='right')) - 1
        chosen = later if later < len(ends) else earlier
        if chosen < 0 or abs(int(ends[chosen]) - int(end)) < minimum_ms:
            raise ValueError(f'no same-sequence packet >= {minimum_ms} ms from {end}')
        result.append(chosen)
    return np.array(result, np.int64)


def validate_bank_sequence(data, meta):
    actual_runs = np.asarray(data['aux']['valid_runs_ms'], np.int64).reshape(-1, 2)
    if not np.array_equal(actual_runs, np.asarray(meta['valid_runs_ms'])):
        raise ValueError('frozen bank valid-run metadata differs from current data')
    expected = packet_rows(actual_runs)
    if len(expected) != len(meta['rows']):
        raise ValueError('frozen bank omits evaluation packets')
    for got, want in zip(meta['rows'], expected):
        if any(int(got[k]) != int(want[k]) for k in want):
            raise ValueError('frozen packet clock mismatch')
        n = int(data['offsets'][got['end_ms'] + 1] - data['offsets'][got['event_start_ms']])
        if n != int(got['n_raw_events']):
            raise ValueError('frozen packet event count mismatch')


def make_packet(data, row, states, beta, camera, device, runtime):
    packet = D.make_packet(D.raw_events(data, row), states, beta, camera, device, row)
    image, window = runtime.lnes(data['events'], data['offsets'], row['end_ms'])
    packet.lnes = torch.from_numpy(image).to(device).unsqueeze(0).expand(len(states), -1, -1, -1)
    packet.lnes_window_ms = torch.full((len(states),), window, dtype=torch.long, device=device)
    if bool(packet.target.any()):
        raise AssertionError('XKF diagnostics must never expose inference targets')
    return packet


@torch.no_grad()
def predict(model, packet, *, direct=False, batch_size=7):
    from semkine.xyz_recovery_study import select_packet_rows
    results = np.full((packet.batch_size, 51), np.nan, np.float32)
    failures = []
    for start in range(0, packet.batch_size, batch_size):
        stop = min(start + batch_size, packet.batch_size)
        part = select_packet_rows(packet, torch.arange(start, stop, device=packet.prev_state.device))
        if bool(part.target.any()):
            raise AssertionError('nonzero inference target')
        try:
            value = model.direct_measurement(part)[0] if direct else model.forward_packet(part)
            results[start:stop] = value.detach().float().cpu().numpy()
            for offset in np.flatnonzero(~np.isfinite(results[start:stop]).all(-1)):
                failures.append({'row': int(start + offset), 'reason': 'nonfinite_prediction'})
        except Exception as error:
            failures.extend({'row': i, 'reason': type(error).__name__, 'message': str(error)} for i in range(start, stop))
    return results, failures


@torch.no_grad()
def wrists(model, states, beta, device):
    states = np.asarray(states)
    flat = states.reshape(-1, 51)
    output = np.full((len(flat), 3), np.nan)
    valid = np.flatnonzero(np.isfinite(flat).all(-1))
    for start in range(0, len(valid), 128):
        indices = valid[start:start + 128]
        x = torch.as_tensor(flat[indices], dtype=torch.float32, device=device)
        _, j = model._fk(x, beta.expand(len(indices), -1))
        output[indices] = j[:, model.FK_ROOT_JOINT].detach().cpu().numpy()
    return output.reshape(*states.shape[:-1], 3)


@torch.no_grad()
def single_step(model, cfg, bank, device, batch_size):
    raw, summaries, failures = {}, {}, []
    for seq in NAMES:
        meta = bank['sequences'][seq]
        data = D.load_data(cfg, seq, meta)
        validate_bank_sequence(data, meta)
        beta, camera = D.context(data, device)
        model.set_hand_context(beta, camera)
        runtime = LongWindowEval(meta['valid_runs_ms'], 200)
        rows = meta['rows']
        direct, fused, incoming = [], [], []
        empty_ok = True
        for i, row in enumerate(rows):
            D.reset(model)
            states = initial_states(data['gt'][row['previous_end_ms']])
            # Pure D bypasses both the I path and the SO(3) innovation clip.
            p = make_packet(data, row, states[:1], beta, camera, device, runtime)
            z, err = predict(model, p, direct=True, batch_size=batch_size)
            direct.append(z[0])
            failures.extend(dict(seq=seq, packet_id=row['packet_id'], section='direct', **e) for e in err)
            if seq == 'zgz_local':
                p = make_packet(data, row, states, beta, camera, device, runtime)
                pred, err = predict(model, p, batch_size=batch_size)
                fused.append(pred)
                incoming.append(states)
                failures.extend(dict(seq=seq, packet_id=row['packet_id'], section='one_step', **e) for e in err)
                if i == 0:
                    # Preserve long evidence when clearing the new packet: the empty gate must still hold.
                    p.events = p.events[:0]
                    p.ptr = torch.zeros_like(p.ptr)
                    held, err = predict(model, p, batch_size=batch_size)
                    empty_ok &= not err and np.array_equal(held, states)
                    failures.extend(dict(seq=seq, packet_id=row['packet_id'], section='empty', **e) for e in err)
            if (i + 1) % 100 == 0:
                print(f'single {seq}: {i + 1}/{len(rows)}', flush=True)
        direct = np.stack(direct)
        truth = np.stack([data['gt'][r['end_ms']] for r in rows])
        actual_wrist, true_wrist = wrists(model, direct, beta, device), wrists(model, truth, beta, device)
        wrong = wrong_time_indices(rows)
        errors = np.linalg.norm(actual_wrist - true_wrist, axis=-1) * 1000
        wrong_errors = np.linalg.norm(actual_wrist[wrong] - true_wrist, axis=-1) * 1000
        root = root_error_deg(direct, truth)
        summaries[seq] = dict(n_packets=len(rows), slope=ols_slope(true_wrist[:, 2], actual_wrist[:, 2]),
                             z_bias_mm=finite_mean((actual_wrist[:, 2] - true_wrist[:, 2]) * 1000),
                             root_mean_deg=finite_mean(root), wrist_mean_mm=finite_mean(errors),
                             wrong_time_wrist_mean_mm=finite_mean(wrong_errors),
                             wrong_time_min_gap_ms=int(min(abs(rows[j]['end_ms'] - rows[i]['end_ms']) for i, j in enumerate(wrong))),
                             empty_packets=sum(r['n_raw_events'] == 0 for r in rows))
        raw.update({f'{seq}_direct': direct, f'{seq}_truth': truth, f'{seq}_direct_wrist': actual_wrist,
                    f'{seq}_true_wrist': true_wrist, f'{seq}_wrong_time_indices': wrong,
                    f'{seq}_end_ms': np.array([r['end_ms'] for r in rows])})
        if seq == 'zgz_local':
            fused, incoming = np.stack(fused), np.stack(incoming)
            out_wrist = wrists(model, fused, beta, device)
            in_wrist = wrists(model, incoming, beta, device)
            root = root_error_deg(fused, truth[:, None])
            slopes = [ols_slope(x[:3, 2], y[:3, 2]) for x, y in zip(in_wrist, out_wrist)]
            raw.update(zgz_local_fused=fused, zgz_local_incoming=incoming,
                       zgz_local_fused_wrist=out_wrist, zgz_local_incoming_wrist=in_wrist)
            summaries['one_step'] = dict(depth_slope=ols_slope(in_wrist[:, :3, 2], out_wrist[:, :3, 2]),
                within_packet_depth_slope_mean=finite_mean([np.nan if v is None else v for v in slopes]),
                root_ratio_median=float(np.median(root[:, 3:] / 20.)) if np.isfinite(root[:, 3:]).all() else None,
                clean_root_mean_deg=finite_mean(root[:, 0]), empty_packet_holds=bool(empty_ok),
                empty_packet_initial_states_checked=len(PERTURBATIONS),
                depth_regression='pooled OLS over literal +/-50 mm and clean input wrist depths; per-packet OLS also retained')
    return summaries, raw, failures


def recovery_clips(bank):
    result = []
    for seq in NAMES:
        meta = bank['sequences'][seq]
        rows = meta['rows']
        if seq == 'zgz_global':
            if len(meta['clips']) != 24:
                raise ValueError('G3 requires exactly 24 frozen global clip starts')
            chosen = [(c['clip_id'], c['end_indices'][:12]) for c in meta['clips']]
        else:
            chosen = [(f'{seq}:run{rid}', [i for i, r in enumerate(rows) if r['run_id'] == rid][:12])
                      for rid in sorted({r['run_id'] for r in rows})]
            if len(chosen) != 41:
                raise ValueError('G3 requires all 41 queryable local run starts')
        for name, indices in chosen:
            segment = [rows[i] for i in indices]
            if any(b['previous_end_ms'] != a['end_ms'] for a, b in zip(segment, segment[1:])):
                raise ValueError('recovery clip crosses a gap or run boundary')
            result.append((seq, name, segment))
    return result


def sustained_recovery(wrist_mm, root_deg, clean_wrist_mm, clean_root_deg):
    """Recovery starts at step <=10, confirmed for that and the next two steps.

    Up to 12 actual packets are replayed; never cross a run boundary to obtain
    confirmation. This matches the existing XDL sustained-recovery hit convention.
    """
    w, r, cw, cr = map(lambda v: np.asarray(v, np.float64), (wrist_mm, root_deg, clean_wrist_mm, clean_root_deg))
    if not (w.shape == r.shape == cw.shape == cr.shape) or w.ndim != 1:
        raise ValueError('recovery error arrays must be paired one-dimensional trajectories')
    good = np.isfinite(w) & np.isfinite(r) & np.isfinite(cw) & np.isfinite(cr) & (w <= 2 * cw + 1) & (r <= 2 * cr + 1)
    hit = next((i + 1 for i in range(min(10, max(0, len(good) - 2))) if good[i:i + 3].all()), None)
    return dict(recovered=hit is not None, hit_step=hit, confirmation_step=None if hit is None else hit + 2,
                observed_steps=len(w), insufficient_for_any_confirmation=len(w) < 3,
                right_censored=hit is None and len(w) < 12,
                trajectory_finite=bool(np.isfinite(w).all() and np.isfinite(r).all()
                                       and np.isfinite(cw).all() and np.isfinite(cr).all()))


def summarize_recovery(trials):
    formal = [t for t in trials if t['spec'] != 'c0_natural']
    if not formal:
        raise ValueError('no formal recovery trials')
    possible = [t for t in formal if not t['insufficient_for_any_confirmation']]
    passed = sum(t['recovered'] for t in formal)
    ceiling = len(possible) / len(formal)
    result = gate(passed / len(formal) >= .9, trials=len(formal), recovered=passed,
                  recovery_fraction=passed / len(formal), right_censored=sum(t['right_censored'] for t in formal),
                  impossible_short_trials=len(formal) - len(possible), mathematical_ceiling=ceiling,
                  protocol_feasibility=ceiling >= .9,
                  eligible_only_descriptive_fraction=(sum(t['recovered'] for t in possible) / len(possible)) if possible else None,
                  eligible_only_denominator=len(possible), required_fraction=.9,
                  recovery_convention='first good step <=10, confirmed by 3 consecutive steps; replay up to12 without crossing runs')
    if ceiling < .9:
        result['status'] = 'UNASSESSABLE_PROTOCOL'
        result['limitation'] = 'Literal denominator includes runs with fewer than3 packets; even perfect predictions cannot reach90%. No trials were removed.'
    natural = [t for t in trials if t['spec'] == 'c0_natural']
    result['c0_natural_supplement'] = dict(included=bool(natural), part_of_frozen_gate=False, trials=len(natural),
                                         recovery_fraction=sum(t['recovered'] for t in natural) / len(natural) if natural else None)
    return result


@torch.no_grad()
def recovery(model, cfg, bank, device, batch_size, reference=None):
    trials, raw, failures = [], {}, []
    loaded = {}
    for ci, (seq, name, rows) in enumerate(recovery_clips(bank)):
        if seq not in loaded:
            data = D.load_data(cfg, seq, bank['sequences'][seq])
            validate_bank_sequence(data, bank['sequences'][seq])
            loaded[seq] = (data, *D.context(data, device))
        data, beta, camera = loaded[seq]
        model.set_hand_context(beta, camera)
        D.reset(model)
        states = initial_states(data['gt'][rows[0]['previous_end_ms']])
        specs = list(PERTURBATIONS)
        if reference is not None:
            natural = D.reference_lookup(reference, seq, rows[0])
            states = np.concatenate([states, natural[None]], axis=0)
            specs.append('c0_natural')
        predictions = [states.copy()]
        runtime = LongWindowEval(bank['sequences'][seq]['valid_runs_ms'], 200)
        for step, row in enumerate(rows, start=1):
            live = np.flatnonzero(np.isfinite(states).all(-1))
            next_states = np.full_like(states, np.nan)
            if len(live):
                p = make_packet(data, row, states[live], beta, camera, device, runtime)
                values, errors = predict(model, p, batch_size=batch_size)
                next_states[live] = values
                failures.extend(dict(clip_id=name, step=step, trial=int(live[e['row']]),
                                     **{k: v for k, v in e.items() if k != 'row'}) for e in errors)
            states = next_states
            predictions.append(states.copy())
        predictions = np.stack(predictions, axis=1)
        truth = np.stack([data['gt'][r['end_ms']] for r in rows])
        pred_wrist = wrists(model, predictions[:, 1:], beta, device)
        true_wrist = wrists(model, truth, beta, device)
        errors = np.linalg.norm(pred_wrist - true_wrist[None], axis=-1) * 1000
        roots = root_error_deg(predictions[:, 1:], truth[None])
        for i, spec in enumerate(specs[1:], start=1):
            trials.append(dict(seq=seq, clip_id=name, spec=spec, end_ms=[r['end_ms'] for r in rows],
                wrist_mm=errors[i].tolist(), root_deg=roots[i].tolist(),
                clean_wrist_mm=errors[0].tolist(), clean_root_deg=roots[0].tolist(),
                **sustained_recovery(errors[i], roots[i], errors[0], roots[0])))
        raw[f'recovery_{ci:03d}_prediction'] = predictions
        raw[f'recovery_{ci:03d}_truth'] = truth
        print(f'recovery {ci + 1}/65 {name}: {len(rows)} steps', flush=True)
    return dict(summary=summarize_recovery(trials), trials=trials), raw, failures


def read_main_rows(run, seed, checkpoint_sha256, row_dir):
    """Only complete, checkpoint-matched exact6000 artifacts may decide G4/G5."""
    values, issues, identities = {}, [], []
    for mode in ('fused', 'i', 'd', 'd_zero'):
        suffix = '' if mode == 'fused' else f'_k{mode}'
        path = Path(row_dir) / f'{run}{suffix}_main_row.json'
        if not path.exists():
            issues.append(dict(mode=mode, reason='missing_main_row', path=str(path)))
            continue
        row = json.loads(path.read_text())
        protocol = row['protocol']
        per_seed = row['per_seed'][str(seed)]
        if (protocol.get('exact_optimizer_step') != 6000 or per_seed.get('step') != 6000
                or per_seed.get('checkpoint_sha256') != checkpoint_sha256 or row['xkf']['kmode'] != mode
                or protocol.get('step_ms') != 50 or protocol.get('window_ms') != 50
                or protocol.get('direct_window_ms') != 200 or per_seed.get('n_frames') != 2590
                or row.get('finite_subset_reduction') is not False or row.get('numeric_status') != 'PASS'):
            raise ValueError(f'formal row identity/protocol mismatch: {path}')
        value = per_seed['overall']['mpjpe_ra_mm']
        if value is None or not np.isfinite(value):
            raise ValueError(f'nonfinite formal row: {path}')
        values[mode] = float(value)
        identities.append(dict(path=str(path), sha256=ROW.sha(path)))
        if mode == 'fused':
            rawpath = Path(str(path) + '.raw_accuracy.json')
            if not rawpath.exists():
                issues.append(dict(mode=mode, reason='missing_raw_accuracy', path=str(rawpath)))
                continue
            raw = json.loads(rawpath.read_text())
            if not raw.get('complete') or raw.get('checkpoint_sha256') != checkpoint_sha256:
                raise ValueError(f'incomplete or wrong-checkpoint raw accuracy: {rawpath}')
            ROW.accuracy_aggregate(raw['raw_sequences'])
            local = next(s for s in raw['raw_sequences'] if s['seq'] == 'zgz_local')
            values['jitter_all_mm_per_step'] = local['jitter_all_mm_per_step']
            values['gt_move_mm_per_step'] = local['gt_move_mm_per_step']
            identities.append(dict(path=str(rawpath), sha256=ROW.sha(rawpath)))
    return dict(values=values, issues=issues, identities=identities)


def assess_gates(single, raw, recovery_result, rows):
    true_z = np.concatenate([raw[f'{n}_true_wrist'][:, 2] for n in NAMES])
    pred_z = np.concatenate([raw[f'{n}_direct_wrist'][:, 2] for n in NAMES])
    slope = ols_slope(true_z, pred_z)
    n = sum(single[s]['n_packets'] for s in NAMES)
    true_error = finite_mean(np.concatenate([np.linalg.norm(raw[f'{s}_direct_wrist'] - raw[f'{s}_true_wrist'], axis=-1) * 1000 for s in NAMES]))
    wrong_error = finite_mean(np.concatenate([np.linalg.norm(raw[f'{s}_direct_wrist'][raw[f'{s}_wrong_time_indices']] - raw[f'{s}_true_wrist'], axis=-1) * 1000 for s in NAMES]))
    local, one = single['zgz_local'], single['one_step']
    tests = dict(depth_slope=slope is not None and .8 <= slope <= 1.2,
                 local_bias=local['z_bias_mm'] is not None and abs(local['z_bias_mm']) <= 15,
                 local_root=local['root_mean_deg'] is not None and local['root_mean_deg'] <= 10,
                 wrong_time=true_error is not None and wrong_error is not None and wrong_error >= 2 * true_error)
    result = {'G1': gate(all(tests.values()), checks=tests, n_packets=n, pooled_depth_slope=slope,
                        wrist_mean_mm=true_error, wrong_time_wrist_mean_mm=wrong_error,
                        wrong_time_ratio=wrong_error / true_error if true_error and wrong_error is not None else None,
                        scope='pooled slope and wrong-time mean across all zgz packets; local bias/root per frozen protocol'),
              'G2': gate(one['depth_slope'] is not None and one['depth_slope'] <= .6
                         and one['root_ratio_median'] is not None and one['root_ratio_median'] <= .5
                         and one.get('empty_packet_holds') is True, **one),
              'G3': recovery_result['summary']}
    v = rows['values']
    jitter, motion = v.get('jitter_all_mm_per_step'), v.get('gt_move_mm_per_step')
    complete_g4 = jitter is not None and motion is not None and np.isfinite([jitter, motion]).all()
    result['G4'] = gate(complete_g4 and one['clean_root_mean_deg'] is not None
                        and one['clean_root_mean_deg'] <= 4 and jitter <= 2 * motion,
                        clean_root_mean_deg=one['clean_root_mean_deg'], jitter_mm_per_step=jitter, gt_move_mm_per_step=motion)
    if not complete_g4:
        result['G4']['status'] = 'MISSING_ARTIFACT'
    complete_g5 = all(k in v for k in ('fused', 'i', 'd', 'd_zero'))
    improvements = {k: v[k] - v['fused'] for k in ('i', 'd', 'd_zero')} if complete_g5 else {}
    result['G5'] = gate(complete_g5 and all(x >= 2 for x in improvements.values()), improvement_mm=improvements,
                        overall_ra_mm={k: v[k] for k in ('fused', 'i', 'd', 'd_zero') if k in v})
    if not complete_g5:
        result['G5']['status'] = 'MISSING_ARTIFACT'
    return result


def formal_identity(directory, checkpoint, cfg):
    finish_path, metadata_path = directory / 'train_finish.json', directory / 'training_metadata.json'
    if not finish_path.is_file() or not metadata_path.is_file():
        raise ValueError('formal diagnostics require train_finish.json and training_metadata.json')
    finish, metadata = json.loads(finish_path.read_text()), json.loads(metadata_path.read_text())
    checkpoint_sha = ROW.sha(checkpoint)
    if (finish.get('status') != 'completed_exact_step6000' or finish.get('global_step') != 6000
            or finish.get('preflight_not_formal') is not False
            or finish.get('checkpoint_sha256') != checkpoint_sha):
        raise ValueError('formal train_finish status/preflight/checkpoint identity mismatch')
    if (metadata.get('preflight_not_formal') is not False or metadata.get('max_steps') != 6000
            or metadata.get('seed') != int(cfg['SEED']) or int(cfg['TRAIN']['MAX_STEPS']) != 6000):
        raise ValueError('formal training_metadata preflight/max_steps/seed mismatch')
    return dict(checkpoint_sha256=checkpoint_sha,
                train_finish=dict(path=str(finish_path), sha256=ROW.sha(finish_path)),
                training_metadata=dict(path=str(metadata_path), sha256=ROW.sha(metadata_path)))


def load_model(run, config, device):
    directory = Path(run)
    if not directory.is_dir():
        directory = REPO / 'outputs/semkine' / run
    checkpoint = ROW.exact_checkpoint(directory)
    cfgpath = Path(config) if config else directory / 'config_resolved.yaml'
    cfg = load_config(str(cfgpath))
    provenance = formal_identity(directory, checkpoint, cfg)
    cfg['MODEL']['XKF_KMODE'] = 'fused'
    if int(cfg.get('XKF_DATA', {}).get('EVAL_WINDOW_MS', 200)) != 200:
        raise ValueError('frozen XKF gates require W200')
    model = XKFModel.load_from_checkpoint(str(checkpoint), cfg=cfg, map_location='cpu', strict=True).to(device).float().eval()
    if getattr(model, 'xyz_assoc_prev', 'self') != 'self':
        raise ValueError('GT association is forbidden')
    model.kmode = 'fused'
    for module in model.modules():
        if isinstance(module, ManoLayer):
            module.fast_chain = True
    return model, cfg, dict(run=directory.name, checkpoint=str(checkpoint), **provenance,
                            config=str(cfgpath), config_sha256=ROW.sha(cfgpath), global_step=6000)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', required=True)
    ap.add_argument('--config')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--batch-size', type=int, default=7)
    ap.add_argument('--bank-dir', default=str(REPO / 'outputs/xdl'))
    ap.add_argument('--row-dir', default=str(REPO / 'outputs/semkine'))
    ap.add_argument('--protocol', default=str(REPO / 'outputs/xkf/protocol.json'))
    ap.add_argument('--output')
    args = ap.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        ap.error('batch-size and threads must be positive')
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    model, cfg, identity = load_model(args.run, args.config, device)
    bank, bank_sha = D.load_bank(args.bank_dir)
    if ROW.sha(splits_manifest(cfg)) != bank['split']['sha256']:
        raise ValueError('checkpoint split differs from frozen bank')
    if sum(len(bank['sequences'][s]['rows']) for s in NAMES) != 2590:
        raise ValueError('full 2590 zgz packet denominator required')
    protocol = json.loads(Path(args.protocol).read_text())
    rows = read_main_rows(identity['run'], cfg['SEED'], identity['checkpoint_sha256'], args.row_dir)
    output = Path(args.output or REPO / 'outputs/xkf/diagnostics' / f'{identity["run"]}.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    single, raw, failures = single_step(model, cfg, bank, device, args.batch_size)
    rawpath = output.with_suffix('.npz')
    np.savez_compressed(rawpath, **raw)
    reference, reference_identity = None, []
    if list(Path(args.bank_dir).glob('reference*.pt')):
        reference, reference_identity = D.load_reference(args.bank_dir)
    recovered, raw_rec, rec_failures = recovery(model, cfg, bank, device, args.batch_size, reference)
    raw.update(raw_rec)
    np.savez_compressed(rawpath, **raw)
    failures += rec_failures
    verdict = assess_gates(single, raw, recovered, rows)
    result = dict(study=protocol['study'], identity=identity, seed=int(cfg['SEED']), device=str(device),
        protocol=dict(path=str(args.protocol), sha256=ROW.sha(args.protocol), frozen_gates=protocol['gates']),
        bank=dict(path=str(Path(args.bank_dir) / 'packet_bank.json'), sha256=bank_sha),
        implementation_sha256=ROW.sha(__file__), gates=verdict, single_step=single,
        recovery=recovered, row_artifacts=rows, failures=failures, finite_subset_reduction=False,
        numeric_status='PASS' if not failures else 'FAIL',
        raw=dict(path=str(rawpath), sha256=ROW.sha(rawpath)), c0_reference=reference_identity,
        c0_note='Natural C0 starts are supplementary only; frozen G3 contains six GT perturbations and no C0 natural requirement.',
        scoring='physical MANO wrist position; root geodesic; all inference targets zero; pure D cached once per packet and reused for wrong-time evidence from the same sequence >=1s away',
        claim=dict(single_seed_g1_g4_pass=all(verdict[k]['status'] == 'PASS' for k in ('G1', 'G2', 'G3', 'G4')),
                   problem_solved=False, reason='Two independently assessed seeds must pass G1-G4; this is one checkpoint audit.'))
    ROW.write_json(output, result)
    print(json.dumps(dict(output=str(output), gates={k: v['status'] for k, v in verdict.items()})), flush=True)


if __name__ == '__main__':
    main()
