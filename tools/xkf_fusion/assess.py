#!/usr/bin/env python3
"""Independent CPU audit of r3 own-loop raw evidence and unchanged debug gates.

Never changes a candidate, threshold, input artifact or training state. Incomplete
evidence is not a pass. Original checkpoint readiness is retained separately from
candidate loop checks; crossfit calibration is not a single deployed model.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]
from tools.xkf_repair.assess import extracted_functions, assert_close, require

SEEDS = (3407, 3408, 3409)
R2 = ROOT / 'outputs/xkf/repair_20261015_r2'
CONDITIONS = ('clean', 'z_plus50', 'z_minus50', 'rx_plus20', 'rx_minus20', 'ry_plus20', 'ry_minus20',
              'z_plus50_rx_plus20', 'z_minus50_rx_minus20')
DIRECTIONAL = CONDITIONS[1:7]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def near(actual, expected, label, atol=1e-6, rtol=1e-6):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape, f'{label}: dimensions differ: {a.shape}/{b.shape}')
    require(np.isfinite(a).all() and np.isfinite(b).all(), f'{label}: nonfinite data')
    require(np.allclose(a, b, atol=atol, rtol=rtol), f'{label}: mismatch; max={np.max(np.abs(a-b))}')


def angle(a, b):
    return np.degrees((Rotation.from_rotvec(np.asarray(a)) .inv() * Rotation.from_rotvec(np.asarray(b))).magnitude())


def directional_reduce(records, cohort, condition, mode, scope, target_fraction, residue_fraction):
    rows = [r for r in records if r['cohort'] == cohort and r['condition'] == condition and r['mode'] == mode
            and (scope == 'all_observed' or (scope == 'full_horizon' and not r['censored'])
                 or (scope == 'short_only' and r['censored']))]
    if not rows:
        return None
    def rmse(key):
        return float(np.sqrt(np.mean([r[key] ** 2 for r in rows])))
    target, hold, residue, initial = (rmse(k) for k in ('target_error', 'hold_target_error', 'perturbation_residue', 'initial_perturbation'))
    require(hold > 0 and initial > 0, 'directional denominator must be positive')
    return {'cohort': cohort, 'condition': condition, 'mode': mode, 'scope': scope, 'n_clips': len(rows),
            'n_observed_frames': sum(r['observed_steps'] for r in rows), 'endpoint_target_rmse': target,
            'hold_endpoint_target_rmse': hold, 'endpoint_perturbation_residue_rmse': residue,
            'initial_perturbation_rmse': initial, 'target_fraction_of_hold': target / hold,
            'residue_fraction_of_initial': residue / initial,
            'target_pass': target <= target_fraction * hold, 'residue_pass': residue <= residue_fraction * initial,
            'pass': target <= target_fraction * hold and residue <= residue_fraction * initial,
            'used_for_gate': scope == 'full_horizon', 'unit': 'mm' if condition.startswith('z_') else 'degrees'}


class Assessor:
    def __init__(self, study, rollout, fk=True):
        self.study, self.rollout_root, self.fk = study, rollout, fk
        self.inputs, self.cache, self.errors, self.missing = {}, {}, [], []
        self.calibrations = {}

    def path(self, path):
        path = Path(path)
        return path if path.is_absolute() else ROOT / path

    def digest(self, path):
        path = self.path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        key = str(path.resolve())
        if key not in self.inputs:
            self.inputs[key] = sha(path)
        return self.inputs[key]

    def read(self, path):
        path = self.path(path)
        self.digest(path)
        return json.loads(path.read_text())

    def attempt(self, label, fn):
        try:
            return fn()
        except FileNotFoundError as exc:
            self.missing.append({'scope': label, 'path': str(exc)})
        except (ValueError, KeyError, TypeError, IndexError, AssertionError, OSError, RuntimeError, np.linalg.LinAlgError) as exc:
            self.errors.append({'scope': label, 'error': type(exc).__name__, 'message': str(exc)})
        return None

    def setup(self):
        self.protocol = self.read(self.study / 'protocol.json')
        require(self.protocol['study'] == 'XKF_LOCAL_FUSION_R3', 'wrong study protocol')
        sanity = self.protocol['directional_debug_sanity']
        require(sanity['conditions'] == list(DIRECTIONAL), 'directional conditions changed')
        require(sanity['cohorts'] == ['local_bank', 'local_cold', 'local_warm3'], 'directional cohorts changed')
        require(sanity['endpoint_target_rmse_max_fraction_of_hold'] == .9
                and sanity['endpoint_perturbation_residue_max_fraction_of_initial'] == .9, 'directional thresholds changed')
        require(sanity['per_seed_per_condition_per_cohort'] is True, 'directional seed/condition contract changed')
        require(self.protocol['clean_guards'] == 'r2 localbank clean fused<=hold; localcold full10 clean fused<=hold; globalbank clean fused<=sameLRcontrol2200*1.1, everyseed',
                'clean thresholds changed')
        self.parent = self.read(R2 / 'assessment.json')
        require(self.parent['status'] == 'ASSESSED' and self.parent['validation_errors'] == [], 'r2 parent evidence is not valid/complete')
        self.frozen = self.read(R2 / 'frozen_manifest.json')['source_sha256']
        for path, digest in self.frozen.items():
            require(self.digest(path) == digest, f'frozen source changed: {path}')
        self.old_schedule = self.read(ROOT / 'outputs/xkf/local_recovery_20261015/rollout/protocol.json')
        self.reducers = extracted_functions(ROOT / 'tools/xkf/recovery_probe.py',
            {'METRICS', 'MODES', 'CONDITIONS', 'summarize_values', 'aggregate'}, {'np': np, 'math': math, 'defaultdict': defaultdict})
        self.metrics = tuple(self.reducers['METRICS'])
        import torch
        torch.set_num_threads(4)
        self.torch = torch
        from semkine.xkf import fuse_states
        self.fuse_states = fuse_states
        if self.fk:
            import yaml
            from mano_layer import ManoLayer
            from pose_repr import decode_to_mano_inputs
            from semkine.lie import so3_exp, so3_log
            config = self.read_yaml(ROOT / 'configs/xkf_repair/repair_s3407.yaml')
            self.digest(config['MANO']['NPZ'])
            mano = ManoLayer(config['MANO']['NPZ'], add_mean=False).float().eval()
            mano.fast_chain = True
            class FKOnly:
                def _fk(self, state, betas):
                    decoded = decode_to_mano_inputs(state, config['MODEL']['POSE_REPR'], mano.hands_components, mano.hands_mean)
                    return mano(betas, decoded['global_orient'], decoded['local_full_aa'], decoded['transl'])
            self.fk_model = FKOnly()
            self.frame_metrics = extracted_functions(ROOT / 'tools/xkf/recovery_probe.py', {'frame_metrics'},
                {'torch': torch, 'math': math, 'so3_exp': so3_exp, 'so3_log': so3_log})['frame_metrics']
        return {'verified': True, 'frozen_sources': len(self.frozen), 'full_cpu_fk_replay': self.fk}

    def read_yaml(self, path):
        import yaml
        self.digest(path)
        return yaml.safe_load(self.path(path).read_text())

    def calibration(self, descriptor, seed, clips):
        if descriptor is None:
            return None
        require(self.digest(descriptor['path']) == descriptor['sha256'], 'calibration source hash mismatch')
        key = descriptor['sha256']
        if key in self.calibrations:
            return self.calibrations[key]
        doc = self.read(descriptor['path'])
        require(doc['seed'] == seed and set(doc['clip_specs']) == {c['clip_id'] for c in clips}, 'calibration seed/clip set mismatch')
        require(self.digest(doc['source_spec_file']) == doc['source_spec_sha256'], 'calibration fit export changed')
        raw_data = []
        for item in doc['inputs']:
            require(self.digest(item['path']) == item['sha256'], 'calibration input hash changed')
            with np.load(self.path(item['path']), allow_pickle=False) as raw:
                raw_data.append({k: raw[k].copy() for k in ('packet_id', 'end_ms', 'direct_state', 'truth')})
        raw = {k: np.concatenate([r[k] for r in raw_data]) for k in raw_data[0]}
        ids = raw['packet_id']; seq = np.array([x.split(':')[0] for x in ids]); end = raw['end_ms']; block = end // 5000
        require(len(ids) == 2590 and len(set(ids)) == 2590, 'calibration did not use complete unique zgz pool')
        byid = {v: i for i, v in enumerate(ids)}
        fitted = {}
        for digest, part in doc['fit_partitions'].items():
            train_ids = part['train_packet_ids']
            require(len(set(train_ids)) == len(train_ids) == part['train_rows'], 'calibration fit IDs duplicated/missing')
            train = np.array([byid[x] for x in train_ids], dtype=int)
            mask = np.zeros(len(ids), dtype=bool); mask[train] = True
            expected_sha = hashlib.sha256(str((mask.dtype, mask.shape)).encode() + mask.tobytes()).hexdigest()
            require(expected_sha == digest, 'calibration fit mask hash mismatch')
            x, y = raw['direct_state'][train, :3], raw['truth'][train, :3]
            mean, scale = x.mean(0), np.maximum(x.std(0), 1e-8)
            z, residual = (x - mean) / scale, y - x
            bias = residual.mean(0)
            weights = np.linalg.solve(z.T @ z + len(x) * .001 * np.eye(3), z.T @ (residual - bias))
            delta = (weights / scale[:, None]).T
            fitted[digest] = {'matrix': np.eye(3) + delta, 'bias': bias - delta @ mean, 'mask': mask}
        for clip in clips:
            cid = clip['clip_id']; audit = doc['clip_fit_audit'][cid]; spec = doc['clip_specs'][cid]
            require(spec['kind'] == 'translation_affine', 'assessor currently requires declared affine clip calibration')
            partition = fitted[audit['train_mask_sha256']]
            near(spec['matrix'], partition['matrix'], cid+' refit matrix', atol=1e-10, rtol=1e-10)
            near(spec['bias'], partition['bias'], cid+' refit bias', atol=1e-10, rtol=1e-10)
            test = np.array([byid[x] for x in audit['scored_packet_ids']], dtype=int)
            require(len(test) == clip['observed_steps'] and not partition['mask'][test].any(), cid+': scored GT leaked into calibration')
            require(np.array_equal(end[test], audit['scored_end_ms']), 'calibration scored clocks changed')
            touched = sorted(set(block[test].tolist()))
            require(touched == audit['excluded_time_blocks_in_clip_sequence'], 'calibration exclusion blocks wrong')
            same = seq == clip['sequence']
            require(not (partition['mask'] & same & np.isin(block, touched)).any(), 'crossing block calibration leakage')
            for train_time in end[partition['mask'] & same]:
                require(np.min(np.abs(end[test] - train_time)) >= 200, 'calibration W200 evidence overlaps scored clip')
        self.calibrations[key] = doc
        return doc

    def raw_verify(self, raw, old, protocol, clip, clip_index, beta):
        torch = self.torch
        modes, motion_modes = protocol['modes'], protocol['motion_modes']
        n, m, c = clip['observed_steps'], len(modes), len(modes)-6
        shapes = {'predictions': (n,9,m,51), 'truth': (n,51), 'metrics': (n,9,m,len(self.metrics)),
                  'physical_wrist_error_xyz_mm': (n,9,m,3), 'state_t_error_xyz_mm': (n,9,m,3),
                  'motion_predictions': (n,9,3+c,51), 'corrected_direct': (n,c,51), 'effective_gains': (n,c,19),
                  'initial': (9,51), 'clean': (51,), 'end_ms': (n,), 'packet_ids': (n,),
                  'lnes_window_ms': (n,), 'n_raw_events': (n,)}
        require(set(raw) == set(shapes), 'raw field set changed or missing')
        for key, shape in shapes.items():
            require(raw[key].shape == shape, f'{clip_index} {key}: dimensions wrong')
            if raw[key].dtype.kind in 'fc':
                require(np.isfinite(raw[key]).all(), f'{clip_index} {key}: nonfinite; no row filtering allowed')
        for key in ('truth','clean','end_ms','packet_ids','lnes_window_ms','n_raw_events'):
            require(np.array_equal(raw[key], old[key]), f'{clip_index}: changed source packet {key}')
        near(raw['predictions'][:,:,:6,:3], old['predictions'][...,:3], 'original translation parity', atol=.0002, rtol=0)
        near(raw['predictions'][:,:,:6,3:], old['predictions'][...,3:], 'original pose parity', atol=.002, rtol=0)
        initial = raw['initial']; clean = raw['clean']
        near(initial[0], clean, 'clean initialization', atol=0, rtol=0)
        for ci, (_,z,rx,ry) in enumerate(self.reducers['CONDITIONS']):
            expected = clean.copy(); expected[2] += z
            if rx or ry:
                expected[3:6] = (Rotation.from_rotvec(np.radians([rx,ry,0.])) * Rotation.from_rotvec(clean[3:6])).as_rotvec()
                require(float(angle(expected[3:6], initial[ci,3:6])) < .001, 'initial SO3 perturbation changed')
                expected[3:6] = initial[ci,3:6]
            near(initial[ci], expected, 'initial conditions', atol=1e-6, rtol=0)
        predictions = raw['predictions']
        previous = np.concatenate([np.broadcast_to(initial[:,None], (9,m,51))[None], predictions[:-1]], axis=0)
        near(predictions[:,:,modes.index('hold')], previous[:,:,modes.index('hold')], 'hold changed state', atol=0,rtol=0)
        error = (predictions[...,:3] - raw['truth'][:,None,None,:3]) * 1000
        near(raw['state_t_error_xyz_mm'], error, 'state translation error', atol=.0002,rtol=1e-6)
        policies = protocol['resolved_clip_policies'][clip['clip_id']]
        for ci, policy in enumerate(policies):
            require(policy.get('temperature_multiplier',1.) == 1., 'raw D correction replay supports declared temperature1 only')
            zd = predictions[:,0,modes.index('d')].copy()
            correction = policy.get('direct_correction')
            if correction is not None:
                require(correction['kind']=='translation_affine' and 'root_left_rotation' not in correction, 'unsupported raw correction replay')
                zd[:,:3] = zd[:,:3] @ np.array(correction['matrix']).T + correction['bias']
            live = raw['n_raw_events'] > 0
            near(raw['corrected_direct'][live,ci], zd[live], 'corrected D replay', atol=1e-6,rtol=1e-6)
            scales = policy.get('gain_scale',[1.,1.,1.]); k = np.clip(np.asarray(protocol['base_gains']) * ([scales[0]]*3+[scales[1]]+[scales[2]]*15),0,1)
            k = np.broadcast_to(k,(n,19)).copy()
            if policy.get('count_gate')=='ramp1000':
                k *= np.minimum(1.,raw['n_raw_events']/1000)[:,None]
            near(raw['effective_gains'][:,ci],k,'gain replay',atol=1e-7,rtol=1e-6)
            mi = modes.index(policy['id']); ii = motion_modes.index(policy['id'])
            a = torch.from_numpy(raw['motion_predictions'][:,:,ii].reshape(-1,51)).float()
            b = torch.from_numpy(np.repeat(raw['corrected_direct'][:,ci,None],9,axis=1).reshape(-1,51)).float()
            gains = torch.from_numpy(np.repeat(raw['effective_gains'][:,ci,None],9,axis=1).reshape(-1,19)).float()
            replay = self.fuse_states(a,b,gains).numpy().reshape(n,9,51)
            replay[~live] = previous[~live,:,mi]
            near(predictions[:,:,mi],replay,'candidate fusion algebra',atol=2e-6,rtol=2e-6)
        for step in np.flatnonzero(raw['n_raw_events']==0):
            near(predictions[step],previous[step],'empty holds all modes',atol=0,rtol=0)
        fkmax = 0.
        if self.fk:
            pred = predictions.reshape(-1,51)
            truth = np.broadcast_to(raw['truth'][:,None,None],(n,9,m,51)).reshape(-1,51).copy()
            values, wrists, states = [], [], []
            with torch.no_grad():
                for begin in range(0,len(pred),256):
                    p=torch.from_numpy(pred[begin:begin+256]).float(); t=torch.from_numpy(truth[begin:begin+256]).float()
                    v,w,s=self.frame_metrics(self.fk_model,p,t,torch.from_numpy(beta).float())
                    values.append(v.numpy());wrists.append(w.numpy());states.append(s.numpy())
            values=np.concatenate(values).reshape(raw['metrics'].shape)
            near(raw['metrics'],values,'all MANO/metric replay',atol=.01,rtol=5e-5)
            near(raw['physical_wrist_error_xyz_mm'],np.concatenate(wrists).reshape(raw['physical_wrist_error_xyz_mm'].shape),'physical wrist replay',atol=.002,rtol=2e-5)
            fkmax=float(np.abs(raw['metrics']-values).max())
        return fkmax

    def seed(self, seed):
        directory=self.rollout_root/f's{seed}'
        summary=self.read(directory/'summary.json'); protocol=self.read(directory/'protocol.json')
        require(summary['protocol_sha256']==self.digest(directory/'protocol.json'),'run protocol hash mismatch')
        for key in ('source_sha256','spec_sha256','modes','metrics'):
            require(summary[key]==protocol[key],f'summary/protocol {key} mismatch')
        require(summary['kind']=='XKF_FUSION_R3_EXPLORATORY' and summary['partial'] is False and summary['clip_count']==112,'partial/wrong rollout')
        require(summary['formal_gate_assessed'] is False and summary['automatic_full_training'] is False
                and summary['finite_subset_reduction'] is False and summary['temperature_parameter_restored'] is True,'scope or restored parameter contract changed')
        require(protocol['clips']==self.old_schedule['clips'] and protocol['excluded']==self.old_schedule['excluded'],'112-clip schedule changed')
        require(protocol['conditions']==self.old_schedule['conditions'] and protocol['metrics']==list(self.metrics),'conditions/metrics changed')
        require(protocol['bank_sha256']==self.old_schedule['bank_sha256'],'bank changed')
        require(self.digest(protocol['spec_path'])==protocol['spec_sha256'],'candidate specification changed')
        spec=self.read(protocol['spec_path']);require(spec==protocol['spec'],'candidate spec payload differs')
        names=[x['id'] for x in spec['candidates']];modes=protocol['modes']
        require(modes==list(self.reducers['MODES'])+names and len(set(modes))==len(modes),'candidate/original modes incomplete')
        require(protocol['motion_modes']==['i','fused','fused_keep_t']+names,'motion mode order mismatch')
        require(protocol['seed']==seed and protocol['checkpoint_step']==2200,'seed/step mismatch')
        for name,digest in protocol['source_sha256'].items():
            require(self.digest(name)==digest,f'run source changed: {name}')
        identity=summary['identity'];parent=self.parent['arms']['repair'][str(seed)]['training']
        require(identity['seed']==seed and identity['global_step']==2200 and identity['checkpoint_sha256']==parent['checkpoint_sha256'],'wrong repair2200 identity')
        require(self.digest(identity['checkpoint'])==identity['checkpoint_sha256'] and self.digest(identity['config'])==identity['config_sha256'],'checkpoint/config changed')
        require(protocol['checkpoint_sha256']==identity['checkpoint_sha256'] and protocol['config_sha256']==identity['config_sha256'],'declared identity changed')
        require(len(protocol['calibration_sources'])==len(names),'calibration source count mismatch')
        for i,candidate in enumerate(spec['candidates']):
            doc=self.calibration(protocol['calibration_sources'][i],seed,protocol['clips'])
            for clip in protocol['clips']:
                resolved=protocol['resolved_clip_policies'][clip['clip_id']][i]
                expected={k:v for k,v in candidate.items() if k not in ('direct_correction_by_seed','direct_correction_file')}
                if doc is not None:expected['direct_correction']=doc['clip_specs'][clip['clip_id']]
                require(resolved==expected,'resolved clip calibration/policy changed')
        protocol=dict(protocol,base_gains=summary['gains_fused'])
        require(len(summary['parity'])==112 and [r['clip_id'] for r in summary['parity']]==[c['clip_id'] for c in protocol['clips']],'per-clip parity incomplete')
        rawfiles=list((directory/'raw').glob('*.npz')); require(len(rawfiles)==112,'missing/extra raw clips')
        records,directional,original_loops=[],[],{}
        fkmax=0.;counts=defaultdict(lambda:{'clips':0,'frames':0})
        for index,clip in enumerate(protocol['clips']):
            filename=f'{index:03d}_{clip["cohort"]}_run{clip["run_id"]}.npz'
            path=directory/'raw'/filename;self.digest(path)
            oldpath=R2/f'evaluations/repair_s{seed}_step2200_rollout/raw'/filename
            require(self.digest(oldpath)==self.parent['input_sha256'][str(oldpath.resolve())],'parent raw changed since assessment')
            with np.load(path,allow_pickle=False) as source:raw={k:source[k] for k in source.files}
            with np.load(oldpath,allow_pickle=False) as source:old={k:source[k] for k in source.files}
            directpath=R2/f'evaluations/repair_s{seed}_step2200_direct/{clip["sequence"]}_W200.npz'
            if clip['sequence'] not in self.cache:
                self.digest(directpath)
                with np.load(directpath,allow_pickle=False) as source:self.cache[clip['sequence']]=source['beta'].copy()
            fkmax=max(fkmax,self.raw_verify(raw,old,protocol,clip,index,self.cache[clip['sequence']]))
            for scope in ('all_observed','short_only' if clip['censored'] else 'full_horizon'):
                counts[(clip['cohort'],scope)]['clips']+=1;counts[(clip['cohort'],scope)]['frames']+=clip['observed_steps']
            for ci,condition in enumerate(CONDITIONS):
                for mi,mode in enumerate(modes):
                    values=raw['metrics'][:,ci,mi];summarize=self.reducers['summarize_values']
                    records.append(dict(clip,condition=condition,mode=mode,mean=summarize(values),first=summarize(values[:1]),endpoint=summarize(values[-1:]),lnes_first_ms=int(raw['lnes_window_ms'][0])))
                    if condition in DIRECTIONAL and mode in names:
                        hold=modes.index('hold');pred=raw['predictions'][-1]
                        if condition.startswith('z_'):
                            target=float(raw['physical_wrist_error_xyz_mm'][-1,ci,mi,2]);held=float(raw['physical_wrist_error_xyz_mm'][-1,ci,hold,2])
                            residue=float(raw['physical_wrist_error_xyz_mm'][-1,ci,mi,2]-raw['physical_wrist_error_xyz_mm'][-1,0,mi,2])
                            initial=float((raw['initial'][ci,2]-raw['initial'][0,2])*1000)
                        else:
                            target=float(angle(raw['truth'][-1,3:6],pred[ci,mi,3:6]));held=float(angle(raw['truth'][-1,3:6],pred[ci,hold,3:6]))
                            residue=float(angle(pred[0,mi,3:6],pred[ci,mi,3:6]));initial=float(angle(raw['initial'][0,3:6],raw['initial'][ci,3:6]))
                        directional.append(dict(clip,condition=condition,mode=mode,target_error=target,hold_target_error=held,perturbation_residue=residue,initial_perturbation=initial))
            if clip['cohort'].endswith('_bank') and clip['sequence'] not in original_loops:
                original_loops[clip['sequence']]={}
                for mode in names:
                    mi=modes.index(mode);hold=modes.index('hold');ci=7
                    candidate_abs=float(raw['metrics'][:,ci,mi,0].mean());hold_abs=float(raw['metrics'][:,ci,hold,0].mean())
                    wrist=float(np.linalg.norm(raw['physical_wrist_error_xyz_mm'][-1,ci,mi]));hold_wrist=float(np.linalg.norm(raw['physical_wrist_error_xyz_mm'][-1,ci,hold]))
                    original_loops[clip['sequence']][mode]={'clip_id':clip['clip_id'],'condition':CONDITIONS[ci],
                        'candidate_abs':candidate_abs,'hold_abs':hold_abs,'candidate_endpoint_wrist':wrist,'hold_endpoint_wrist':hold_wrist,
                        'mean_abs_pass':candidate_abs<=hold_abs,'endpoint_wrist_pass':wrist<hold_wrist}
        perclip=directory/'per_clip.jsonl';self.digest(perclip)
        assert_close([json.loads(x) for x in perclip.read_text().splitlines()],records,'perclip/raw reducers')
        aggregate=self.reducers['aggregate'](records);assert_close(summary['aggregate'],aggregate,'aggregate/raw reducers')
        short=self.reducers['aggregate']([r for r in records if r['censored']])
        for row in short:row['scope']='short_only'
        index={(r['cohort'],r['condition'],r['mode'],r['scope']):r for r in aggregate}
        control=self.read(R2/f'evaluations/control_s{seed}_step2200_rollout/summary.json')
        ctl={(r['cohort'],r['condition'],r['mode'],r['scope']):r for r in control['aggregate']}
        original=self.read(R2/f'evaluations/repair_s{seed}_step2200.json')
        candidates={}
        for name in names:
            guards={}
            for cohort,scope in (('local_bank','all_observed'),('local_cold','full_horizon')):
                value=index[(cohort,'clean',name,scope)]['pooled_frames']['abs_mpjpe_mm'];limit=index[(cohort,'clean','hold',scope)]['pooled_frames']['abs_mpjpe_mm']
                guards[cohort]={'value':value,'hold':limit,'pass':value<=limit}
            value=index[('global_bank','clean',name,'all_observed')]['pooled_frames']['abs_mpjpe_mm'];reference=ctl[('global_bank','clean','fused','all_observed')]['pooled_frames']['abs_mpjpe_mm']
            guards['global_bank']={'value':value,'control':reference,'limit':reference*1.1,'pass':value<=reference*1.1}
            directions=[directional_reduce(directional,cohort,condition,name,scope,.9,.9)
                        for cohort in ('local_bank','local_cold','local_warm3') for condition in DIRECTIONAL
                        for scope in ('all_observed','full_horizon','short_only')]
            directions=[v for v in directions if v is not None]
            full=[v for v in directions if v['used_for_gate']];require(len(full)==18,'directional gate cells missing')
            candidates[name]={'clean_guards':guards,'clean_pass':all(x['pass'] for x in guards.values()),
                'directional':directions,'directional_pass':all(x['pass'] for x in full),
                'directional_failed_cells':[{'cohort':x['cohort'],'condition':x['condition'],'target_fraction_of_hold':x['target_fraction_of_hold'],
                                            'residue_fraction_of_initial':x['residue_fraction_of_initial']} for x in full if not x['pass']],
                'original_candidate_loop_checks':{seq:v[name] for seq,v in original_loops.items()},
                'candidate_original_readiness':{'status':'INCOMPLETE','reason':'retained parent criteria are unchanged; no matching candidate initial/check-cohort evidence supplied',
                                               'criteria':original['readiness']['criteria']}}
            candidates[name]['additional_debug_status']='PASS' if candidates[name]['clean_pass'] and candidates[name]['directional_pass'] else 'FAIL'
        return {'identity':identity,'candidates':candidates,'original_parent_readiness':original['readiness'],
                'raw_verification':{'complete':True,'clips':112,'conditions':9,'modes':modes,'all_metrics_cpu_fk_replayed':self.fk,
                                    'metric_max_cpu_gpu_difference':fkmax,'fusion_algebra_replayed':True,
                                    'neural_i_inference_replayed':False,'own_loop_input_contract':'source audit plus existing nonlinear recurrence unit test; each candidate uses own previous state'},
                'counts':[{'cohort':k[0],'scope':k[1],**v} for k,v in sorted(counts.items())],
                'aggregate':aggregate,'short_only_aggregate':short}

    def run(self):
        setup=self.attempt('setup',self.setup)
        result={'kind':'XKF_FUSION_R3_RAW_ASSESSMENT','formal_gate_assessed':False,'automatic_full_training':False,
                'assessor_sha256':sha(__file__),'setup':setup,'seeds':{},'all_three_seed_candidates':{},
                'formal_recovery_gates':{'status':'NOT_ASSESSED','original_constraints_preserved':[
                    'one-step depth response slope <=0.6', 'one-step root +/-20deg error at least halved',
                    'clean root <=4deg', 'original formal 90%-within-2x-clean recovery gate'],
                    'scope':'the r3 0.9 endpoint checks are additional minimal debug checks and do not replace formal G1-G4 or original readiness'}}
        if setup:
            for seed in SEEDS:
                result['seeds'][str(seed)]=self.attempt(f'seed{seed}',lambda s=seed:self.seed(s))
            if all(result['seeds'].values()):
                names=set(result['seeds']['3407']['candidates'])
                if not all(set(x['candidates'])==names for x in result['seeds'].values()):
                    self.errors.append({'scope':'cross_seed','error':'ValueError','message':'cross-seed candidate sets differ'})
                    names=set()
                for name in sorted(names):
                    cells={s:r['candidates'][name] for s,r in result['seeds'].items()}
                    result['all_three_seed_candidates'][name]={'additional_debug_status':'PASS' if all(x['additional_debug_status']=='PASS' for x in cells.values()) else 'FAIL',
                        'clean_all_seed_pass':all(x['clean_pass'] for x in cells.values()),'directional_all_seed_pass':all(x['directional_pass'] for x in cells.values()),
                        'original_readiness_status':'INCOMPLETE','ready_for_full_training':False}
        result.update(status='INVALID' if self.errors else 'INCOMPLETE' if self.missing or not self.fk else 'ASSESSED',
                      validation_errors=self.errors,missing=self.missing,input_sha256=self.inputs)
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=ROOT/'outputs/xkf/fusion_20261015_r3')
    parser.add_argument('--rollout-dir',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--skip-fk',action='store_true',help='explicit incomplete reducer smoke; never ASSESSED')
    args=parser.parse_args()
    directory=args.rollout_dir or args.study/'rollout_six'
    output=args.output or args.study/'assessment.json'
    result=Assessor(args.study.resolve(),directory.resolve(),not args.skip_fk).run()
    write(output,result)
    print(json.dumps({'output':str(output),'status':result['status'],'errors':result['validation_errors'],
                      'missing':len(result['missing']),'candidates':result['all_three_seed_candidates']}),flush=True)
    return 2 if result['status']=='INVALID' else 0


if __name__=='__main__':
    raise SystemExit(main())
