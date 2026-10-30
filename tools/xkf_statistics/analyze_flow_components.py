#!/usr/bin/env python3
"""Post-decision, training-only algebraic flow diagnosis; no policy fitting."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'model')]
SEEDS = (3407, 3408, 3409)
MODES = ('baseline', 'identity_cv', 'family_cv')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def energy(x):
    return np.sum(x*x, axis=-1).mean()


def dot(x, y):
    return np.sum(x*y, axis=-1).mean()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    require(not args.output.exists(), 'refuse overwrite')
    args.output.mkdir(parents=True)
    import torch
    import yaml
    from mano_layer import ManoLayer
    from pose_repr import decode_to_mano_inputs
    torch.set_num_threads(4)
    inputs = {}

    def read(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        require(expected is None or digest == expected, 'input hash differs '+str(path))
        inputs[str(path)] = digest
        return json.loads(path.read_text())

    assessment = read(args.root/'assessment/result.json')
    require(assessment['status'] == 'REJECTED_TRAIN_VALIDATION'
            and not assessment['accepted_for_dev'], 'this diagnostic must preserve rejection')
    decl = read(args.root/'declaration_v2.json', assessment['declaration_sha256'])
    cfgpath = decl['models']['3407']['config']
    inputs[str(Path(cfgpath).resolve())] = sha(cfgpath)
    cfg = yaml.safe_load(Path(cfgpath).read_text())
    inputs[str(Path(cfg['MANO']['NPZ']).resolve())] = sha(cfg['MANO']['NPZ'])
    mano = ManoLayer(cfg['MANO']['NPZ'], add_mean=False).float().eval()
    mano.fast_chain = True

    @torch.no_grad()
    def fk(states, beta):
        states = np.asarray(states)
        shape = states.shape[:-1]
        flat = states.reshape(-1, 51)
        b = np.broadcast_to(np.asarray(beta), (len(flat), 10)).copy()
        out = []
        for start in range(0, len(flat), 256):
            x = torch.from_numpy(np.asarray(flat[start:start+256], np.float32))
            d = decode_to_mano_inputs(x, cfg['MODEL']['POSE_REPR'], mano.hands_components, mano.hands_mean)
            out.append(mano(torch.from_numpy(b[start:start+256]).float(), d['global_orient'],
                            d['local_full_aa'], d['transl'])[1].numpy())
        return np.concatenate(out).astype(float).reshape(*shape, 21, 3)*1000.

    def geometric_change(a, b, ja, jb, beta):
        """Symmetric two-order root/finger intervention, after centering joints."""
        root_only = a.copy(); root_only[..., 3:6] = b[..., 3:6]
        finger_only = a.copy(); finger_only[..., 6:] = b[..., 6:]
        jr, jf = fk(np.stack([root_only, finger_only]), beta)
        center = lambda j: j-j.mean(axis=-2, keepdims=True)
        sa, sb, sr, sf = [center(j) for j in (ja, jb, jr, jf)]
        centroid = np.broadcast_to(jb.mean(-2, keepdims=True)-ja.mean(-2, keepdims=True), ja.shape)
        root = .5*((sr-sa)+(sb-sf))
        fingers = .5*((sf-sa)+(sb-sr))
        require(np.allclose(centroid+root+fingers, jb-ja, atol=1e-10, rtol=1e-10), 'geometry identity')
        return np.stack([centroid, root, fingers], axis=-3)

    prereg = dict(kind='POST_REJECTION_TRAIN_FLOW_DIAGNOSTIC', no_policy_fitting=True,
        no_neural_forward=True, no_development_data=True, accepted_for_dev=False,
        decisions_unchanged=True, conditions=['clean'], seeds=list(SEEDS), modes=list(MODES),
        phases=['all', 'step0', 'step1', 'step2', 'prefix_0to2', 'main_3plus'],
        domains=['all', 'local', 'global'], motion=['all', 'high', 'low'],
        high_motion_mask='unchanged fold-fit q75 from original declaration, same per-policy mask',
        vector_identity='fused_flow_error = raw_I_flow_error + I_gate_change + fusion_correction',
        geometry='centroid + symmetric two-order root/finger centered FK contributions; exact sum, diagnostic hybrid states only',
        centroid_temporal_identity='t>=1: fused_flow_error_centroid=(1-K)*gated_I_flow_error_centroid + K*(D_error_t-D_error_tminus1) + K*(D_error_tminus1-previous_fused_error)',
        limitations=['D error changes include deterministic event/pose/lag effects and are not identified stochastic noise.',
                     'Each policy uses its own actual incoming state; differences include prior closed-loop effects.',
                     'Branch algebra and symmetric geometric attribution are not new counterfactual closed-loop trials.',
                     'Current clean initial truth makes step0 unbiased initial state, later steps have prior prediction error.'])
    (args.output/'declaration.json').write_text(json.dumps(prereg, indent=2)+'\n')
    cells, checks, seed_outputs = [], [], {}
    for seed in SEEDS:
        base = args.root/f'validation_s{seed}_shard0'
        protocol = read(base/'protocol.json')
        summary = read(base/'summary.json')
        folds = read(args.root/f'folds_s{seed}.json')
        require(protocol['condition_indices'] == list(range(5)) and protocol['conditions'][0] == 'clean'
                and len(protocol['clips']) == 72, 'clean schedule changed')
        accum = {}
        def append(k, value):
            accum.setdefault(k, []).append(np.asarray(value))
        max_fk_flow_difference = 0.
        for ci, clip in enumerate(protocol['clips']):
            path = base/'raw'/f'{ci:03d}_{clip["sequence"]}.npz'
            digest = sha(path)
            require(digest == summary['raw_sha256'][path.name]
                    and assessment['input_sha256'][str(path.resolve())] == digest, 'unassessed raw')
            inputs[str(path.resolve())] = digest
            with np.load(path, allow_pickle=False) as loaded:
                raw = {k: loaded[k] for k in loaded.files}
            n = len(raw['end_ms'])
            require(n == len(clip['planned_end_ms']) and np.array_equal(raw['incoming'][1:], raw['predictions'][:-1]), 'own-loop mismatch')
            states = [raw[k][:, 0] for k in ('incoming', 'motion_predictions', 'gated_motion_predictions', 'predictions', 'direct_predictions')]
            ji, jr, jg, jf, jd = fk(np.stack(states), raw['beta'])
            jt0, jt = fk(np.stack([raw['previous_truth'], raw['truth']]), raw['beta'])
            true_flow = jt-jt0
            gt_geom = geometric_change(raw['previous_truth'], raw['truth'], jt0, jt, raw['beta'])
            prediction_flow = jf-ji
            stored_flow = raw['joint_motion_mm'][:, 0].astype(float)
            max_fk_flow_difference = max(max_fk_flow_difference, float(np.max(np.abs(prediction_flow-stored_flow))))
            require(np.allclose(prediction_flow, stored_flow, atol=.002, rtol=2e-4), 'CPU FK flow differs')
            raw_error = jr-ji-true_flow[:, None]
            gated_error = jg-ji-true_flow[:, None]
            fused_error = jf-ji-true_flow[:, None]
            gate_change = jg-jr
            correction = jf-jg
            correction_geom = geometric_change(states[2], states[3], jg, jf, raw['beta'])
            raw_geom = geometric_change(states[0], states[1], ji, jr, raw['beta'])-gt_geom[:, None]
            gated_geom = geometric_change(states[0], states[2], ji, jg, raw['beta'])-gt_geom[:, None]
            fused_geom = geometric_change(states[0], states[3], ji, jf, raw['beta'])-gt_geom[:, None]
            require(np.allclose(fused_error, raw_error+gate_change+correction, atol=1e-10), 'branch identity')
            derror = jd-jt[:, None]
            prev_error = ji-jt0[:, None]
            ddelta_error = np.zeros_like(derror)
            ddelta_error[1:] = derror[1:]-derror[:-1]
            k = raw['effective_gains'][:, 0, :, :3].astype(float)
            egc = gated_error.mean(-2)
            epc = prev_error.mean(-2)
            edc = derror.mean(-2)
            edprev = np.zeros_like(edc); edprev[1:] = edc[:-1]
            centroid_terms = np.stack([(1-k)*egc, k*(edc-edprev), k*(edprev-epc)], -2)
            require(np.allclose(centroid_terms.sum(-2)[1:], fused_error.mean(-2)[1:], atol=.002, rtol=1e-4), 'centroid temporal identity')
            # At step0 the D current absolute error term is separated; temporal D difference is undefined.
            for key, value in dict(raw_i_error=raw_error, gated_i_error=gated_error, fused_error=fused_error,
                    gate_change=gate_change, correction=correction, correction_geometry=correction_geom,
                    raw_i_geometry_error=raw_geom, gated_i_geometry_error=gated_geom, fused_geometry_error=fused_geom,
                    direct_absolute_error=derror, previous_fused_error=prev_error, direct_error_change=ddelta_error,
                    centroid_temporal_terms=centroid_terms, true_flow=true_flow, stored_fused_error=raw['joint_motion_error_mm'][:, 0],
                    stored_prediction_flow=stored_flow, stored_true_flow=raw['true_joint_motion_mm'],
                    gains=raw['effective_gains'][:, 0], motion_gate=raw['motion_gate'][:, 0]).items():
                append(key, value)
            append('sequence', np.full(n, clip['sequence']))
            append('step', np.arange(n))
            append('count', raw['n_raw_events'])
            append('speed', raw['true_joint_speed_mm_s'])
            append('end_ms', raw['end_ms'])
            append('is_local', np.full(n, '_local' in clip['sequence']))
            append('thresholds', np.tile([folds['identity'][clip['training_identity']]['high_motion_threshold_mm_s'],
                        folds['trial'][clip['sequence']]['high_motion_threshold_mm_s']], (n, 1)))
        a = {k: np.concatenate(v) for k, v in accum.items()}
        require(len(a['step']) == 690, 'all clean frames required')
        out = args.output/f'derived_s{seed}.npz'
        np.savez_compressed(out, **a)
        seed_outputs[str(seed)] = dict(path=str(out.resolve()), sha256=sha(out), rows=690, clips=72,
                                      max_cpu_vs_gpu_flow_component_difference_mm=max_fk_flow_difference)
        phase_masks = dict(all=np.ones(690, bool), step0=a['step']==0, step1=a['step']==1,
                           step2=a['step']==2, prefix_0to2=a['step']<3, main_3plus=a['step']>=3)
        for si, scheme in enumerate(('identity_cv', 'family_cv')):
            high = a['speed'] >= a['thresholds'][:, si]
            for mi in (0, si+1):
                for phase, pmask in phase_masks.items():
                    for domain, dmask in dict(all=np.ones(690, bool), local=a['is_local'], global_=~a['is_local']).items():
                        for motion, mmask in dict(all=np.ones(690, bool), high=high, low=~high).items():
                            mask = pmask & dmask & mmask
                            if not mask.any():
                                continue
                            rr, gg, ff, corr = [a[k][mask, mi] for k in ('raw_i_error', 'gated_i_error', 'fused_error', 'correction')]
                            gt = a['true_flow'][mask]
                            value = dict(seed=seed, scheme=scheme, mode=MODES[mi], phase=phase,
                                domain=domain.replace('global_', 'global'), motion=motion, frames=int(mask.sum()), clips=len(np.unique(a['sequence'][mask])),
                                raw_i_flow_rmse_mm=float(np.sqrt(energy(rr))), gated_i_flow_rmse_mm=float(np.sqrt(energy(gg))),
                                fused_flow_rmse_mm=float(np.sqrt(energy(ff))), correction_rms_mm=float(np.sqrt(energy(corr))),
                                raw_to_gated_delta_mse_mm2=float(energy(gg)-energy(rr)),
                                gated_to_fused_delta_mse_mm2=float(energy(ff)-energy(gg)),
                                correction_energy_mm2=float(energy(corr)), gated_error_correction_cross_mm2=float(2*dot(gg, corr)),
                                direct_absolute_rmse_mm=float(np.sqrt(energy(a['direct_absolute_error'][mask, mi]))),
                                fused_absolute_rmse_mm=float(np.sqrt(energy(a['previous_fused_error'][mask, mi]+ff))),
                                true_flow_rms_mm=float(np.sqrt(energy(gt))),
                                projection=float(dot(ff+gt, gt)/energy(gt)) if energy(gt)>0 else None,
                                mean_motion_gate=float(a['motion_gate'][mask, mi].mean()),
                                mean_gains=a['gains'][mask, mi].mean(0).tolist())
                            for bi, name in enumerate(('centroid', 'root', 'fingers')):
                                cc = a['correction_geometry'][mask, mi, bi]
                                value[name+'_correction_rms_mm'] = float(np.sqrt(energy(cc)))
                                value[name+'_correction_delta_mse_allocation_mm2'] = float(dot(cc, corr)+2*dot(gg, cc))
                                for prefix, key in (('raw_i', 'raw_i_geometry_error'), ('gated_i', 'gated_i_geometry_error'), ('fused', 'fused_geometry_error')):
                                    component = a[key][mask, mi, bi]
                                    value[prefix+'_'+name+'_flow_component_rmse_mm'] = float(np.sqrt(energy(component)))
                            later = mask & (a['step']>0)
                            if later.any():
                                value['later_frames'] = int(later.sum())
                                value['direct_error_change_rmse_mm'] = float(np.sqrt(energy(a['direct_error_change'][later, mi])))
                                terms = a['centroid_temporal_terms'][later, mi]
                                total = terms.sum(1)
                                value['centroid_flow_error_mse_mm2'] = float(energy(total))
                                for ti, name in enumerate(('weighted_gated_I', 'weighted_D_error_change', 'weighted_previous_D_fused_gap')):
                                    value[name+'_rms_mm'] = float(np.sqrt(energy(terms[:, ti])))
                                    value[name+'_signed_mse_allocation_mm2'] = float(dot(terms[:, ti], total))
                            cells.append(value)
            for failure in assessment['failures']:
                if failure['seed'] == seed and failure['scheme'] == scheme:
                    for mi, name in ((0, 'baseline'), (si+1, 'candidate')):
                        pred = a['stored_prediction_flow'][high, mi].astype(float)
                        gt = a['stored_true_flow'][high].astype(float)
                        err = pred-gt
                        result = dict(frames=int(high.sum()), joint_motion_rmse_mm=float(np.sqrt(energy(err))), projection=float(dot(pred, gt)/energy(gt)))
                        require(result['frames'] == failure[name]['frames']
                                and abs(result['joint_motion_rmse_mm']-failure[name]['joint_motion_rmse_mm'])<1e-10
                                and abs(result['projection']-failure[name]['projection'])<1e-10, 'independent high-motion guard differs')
                        checks.append(dict(seed=seed, scheme=scheme, mode=name, **result))
        print(json.dumps(dict(seed=seed, rows=690, independent_guard_match=True)), flush=True)
    for name, digest in inputs.items():
        require(sha(name)==digest, 'input changed during diagnostic')
    result = dict(kind=prereg['kind'], status='COMPLETE', accepted_for_dev=False, source_sha256={str(Path(__file__).resolve()):sha(__file__)},
        declaration_sha256=sha(args.output/'declaration.json'), input_sha256=inputs, derived=seed_outputs,
        independent_guard_checks=checks, cells=cells, limitations=prereg['limitations'])
    (args.output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status='COMPLETE', cells=len(cells))), flush=True)


if __name__ == '__main__':
    main()
