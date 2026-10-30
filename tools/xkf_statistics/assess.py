#!/usr/bin/env python3
"""Independent CPU replay and frozen-declaration reduction of all six shards.

Reads training identities only. Never forwards a neural network, fits a new
policy, selects an endpoint, updates a checkpoint, or releases full training.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tools.xkf_statistics import collect as C

SEEDS = (3407, 3408, 3409)
DECLARATION_SHA256 = '64ecaaa1a11bbebb1b01a17e837fcc38ced207a22ffde090f89851cda8a8e9f7'
MODES = ('baseline', 'identity_cv', 'family_cv')
METRICS = ('abs_mpjpe_mm', 'ra_mpjpe_mm', 'abs_mpvpe_mm', 'ra_mpvpe_mm',
           'state_t_error_mm', 'state_t_squared_mm2', 'state_depth_error_mm', 'state_depth_squared_mm2',
           'physical_wrist_error_mm', 'physical_wrist_squared_mm2', 'physical_wrist_depth_error_mm',
           'physical_wrist_depth_squared_mm2', 'root_error_deg', 'fingers_rmse_rad',
           'absolute_joint_mse_mm2', 'relative_joint_mse_mm2', 'wrist_relative_cross_mm2')
GROUPS = [(0, 1), (1, 2), (2, 3), (3, 6)] + [(i, i+3) for i in range(6, 51, 3)]


def require(value, message):
    if not value:
        raise ValueError(message)


def close(a, b, message, atol=1e-8, rtol=1e-7):
    a, b = np.asarray(a), np.asarray(b)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            and np.allclose(a, b, atol=atol, rtol=rtol), message)
    return float(np.max(np.abs(a-b))) if a.size else 0.


def coefficient_fit(ei, ed, gates, fallback, floor):
    """Independent NumPy constrained scalar quadratic solution."""
    answer = []
    for i, (a, b) in enumerate(GROUPS):
        e, difference = ei[:, a:b], ei[:, a:b]-ed[:, a:b]
        numerator = np.mean(gates*np.sum(e*difference, axis=1))
        denominator = np.mean(gates*gates*np.sum(difference*difference, axis=1))
        answer.append(float(fallback[i]) if denominator <= floor else float(np.clip(numerator/denominator, 0., 1.)))
    return answer


def initial_conditions(clean, draw):
    values = np.repeat(clean[None], 10, axis=0)
    values[1, 2] += .05; values[2, 2] -= .05
    for ci, axis, sign in ((3, 0, 1), (4, 0, -1), (5, 1, 1), (6, 1, -1), (7, 0, 1), (8, 0, -1)):
        delta = np.zeros(3); delta[axis] = sign*np.deg2rad(20.)
        values[ci, 3:6] = (Rotation.from_rotvec(delta)*Rotation.from_rotvec(clean[3:6])).as_rotvec()
    values[7, 2] += .05; values[8, 2] -= .05
    values[9] = clean+draw
    return values


def root_branches(root, reference):
    norm = np.linalg.norm(root, axis=-1, keepdims=True)
    alternate = root*(1.-2*np.pi/np.maximum(norm, 1e-9))
    choose = (np.linalg.norm(alternate-reference, axis=-1, keepdims=True)
              < np.linalg.norm(root-reference, axis=-1, keepdims=True)) & (norm > 1e-6)
    return np.where(choose, alternate, root)


def interpolated_motion(previous, proposal, gate):
    shape = previous.shape
    a, b, g = previous.reshape(-1, 51).astype(float), proposal.reshape(-1, 51).astype(float), gate.reshape(-1)
    out = a+g[:, None]*(b-a)
    ra, rb = Rotation.from_rotvec(a[:, 3:6]), Rotation.from_rotvec(b[:, 3:6])
    root = (ra*Rotation.from_rotvec(g[:, None]*(ra.inv()*rb).as_rotvec())).as_rotvec()
    out[:, 3:6] = root_branches(root, a[:, 3:6])
    out[g == 0] = a[g == 0]; out[g == 1] = b[g == 1]
    return out.reshape(shape)


def primary_reduction(records):
    """One independent paired unit is an identity after averaging its seeds."""
    chosen = [r for r in records if r['condition'] == 'small_gaussian' and r['scope'] == 'main_onward']
    cells = defaultdict(dict)
    for r in chosen:
        cells[(r['seed'], r['identity'], r['clip'])][r['mode']] = r['joint_mse_mm2']
    per_seed_identity = defaultdict(list)
    for (seed, identity, _), value in cells.items():
        require(set(value) == set(MODES), 'primary missing mode')
        per_seed_identity[(seed, identity)].append(value['baseline']-value['identity_cv'])
    ids = sorted({key[1] for key in per_seed_identity})
    require(len(ids) == 9 and len(per_seed_identity) == 27, 'primary requires nine identities for all seeds')
    values = {(seed, identity): float(np.mean(per_seed_identity[(seed, identity)])) for seed in SEEDS for identity in ids}
    paired = [float(np.mean([values[(seed, identity)] for seed in SEEDS])) for identity in ids]
    mean, se = float(np.mean(paired)), float(np.std(paired, ddof=1)/np.sqrt(9.))
    seed_means = {str(seed): float(np.mean([values[(seed, identity)] for identity in ids])) for seed in SEEDS}
    return dict(mean_improvement_mm2=mean, paired_SE_mm2=se, independent_units=9,
                mean_gt_one_paired_SE=mean > se, per_seed_mean_improvement_mm2=seed_means,
                per_seed_noninferiority={seed: v >= 0 for seed, v in seed_means.items()},
                identity_improvements_mm2=dict(zip(ids, paired)),
                per_seed_identity_improvements_mm2={f'{s}/{i}':v for (s, i), v in values.items()})


def balanced_absolute(records, seed, mode, scope):
    cells = defaultdict(list)
    for r in records:
        if r['seed'] == seed and r['mode'] == mode and r['condition'] == 'clean' and r['scope'] == scope:
            cells[r['identity']].append(r['abs_mpjpe_mm'])
    require(len(cells) == 9 and all(cells.values()), 'clean scope missing a whole identity')
    return float(np.mean([np.mean(v) for v in cells.values()]))


def flow_reduction(cells):
    require(cells and sum(c['n'] for c in cells) > 0, 'high-motion selection empty')
    n = sum(c['n'] for c in cells)
    denominator = sum(c['true_energy'] for c in cells)
    require(denominator > 0, 'high-motion projection denominator zero')
    return dict(frames=n, joint_motion_rmse_mm=float(np.sqrt(sum(c['error_energy'] for c in cells)/(n*21))),
                projection=float(sum(c['dot'] for c in cells)/denominator), true_energy_mm2=denominator)


class Audit:
    def __init__(self, declaration, directory):
        self.inputs, self.root = {}, directory.resolve()
        self.declaration_path = declaration.resolve()
        self.decl = self.read(self.declaration_path,DECLARATION_SHA256)
        C.load_declaration(self.declaration_path)
        require(self.decl['crossfit']['secondary'] == 'motion_family'
                and self.decl['crossfit']['motion_family_folds'] == 18, 'old leaking sequence CV is forbidden')
        self.dependencies(self.decl)
        self.rows = self.read(self.decl['rows_path'])
        self.schedule = self.read(self.decl['validation_schedule_path'])['clips']
        require(len(self.rows) == 2304 and len(self.schedule) == 72
                and sum(len(c['planned_end_ms']) for c in self.schedule) == 690
                and sum(len(c['planned_end_ms']) < 10 for c in self.schedule) == 5, 'schedule denominators differ')
        self.indices = np.asarray(self.decl['fit_row_indices'])
        require(np.array_equal(self.indices, [i for i, r in enumerate(self.rows) if r['stratum'] == 'full200'])
                and len(self.indices) == 2016, 'full200 fixed fit membership differs')
        self.draws = self.npz(self.decl['gaussian_draws_path'])['draws']
        self.records, self.high, self.validation = [], defaultdict(list), {}
        self.shared = {}

    def path(self, value):
        path = Path(value)
        return path.resolve() if path.is_absolute() else (ROOT/path).resolve()

    def digest(self, value, expected=None):
        path = self.path(value); digest = C.sha(path)
        require(expected is None or digest == expected, 'hash mismatch: '+str(path))
        require(str(path) not in self.inputs or self.inputs[str(path)] == digest, 'input changed during audit')
        self.inputs[str(path)] = digest
        return digest

    def read(self, path, expected=None):
        self.digest(path, expected)
        return json.loads(self.path(path).read_text())

    def npz(self, path, expected=None):
        self.digest(path, expected)
        with np.load(self.path(path), allow_pickle=False) as loaded:
            result = {k: loaded[k].copy() for k in loaded.files}
        for key, value in result.items():
            if value.dtype.kind in 'fc':
                require(np.isfinite(value).all(), 'nonfinite raw, no subset allowed: '+str(path)+'/'+key)
        return result

    def dependencies(self, value):
        for field in ('source_sha256', 'input_sha256'):
            for path, digest in value.get(field, {}).items():
                self.digest(path, digest)

    @torch.no_grad()
    def fk(self, states, beta):
        flat = np.asarray(states).reshape(-1, 51)
        betas = np.asarray(beta)
        if betas.shape == (10,):
            betas = np.broadcast_to(betas, (len(flat), 10))
        require(betas.shape == (len(flat), 10), 'geometry beta rows mismatch')
        vs, js = [], []
        for begin in range(0, len(flat), 256):
            v, j = self.model._fk(torch.tensor(flat[begin:begin+256], dtype=torch.float32),
                                  torch.tensor(betas[begin:begin+256], dtype=torch.float32))
            vs.append(v.numpy()); js.append(j.numpy())
        return np.concatenate(vs).astype(float), np.concatenate(js).astype(float)

    def collection(self, seed, folds):
        summary = self.read(folds['collection_summary_path'], folds['collection_summary_sha256'])
        self.dependencies(summary)
        require(summary['seed'] == seed and summary['status'] == 'COMPLETE' and not summary['partial_smoke']
                and summary['rows'] == 2016 and summary['actual_forward_calls'] == 8064
                and summary['all_buffers_unchanged'] is True
                and summary['declaration_sha256'] == self.digest(self.declaration_path), 'incomplete collection')
        require(summary['identity'] == self.identity, 'collection model differs from actual bound endpoint')
        result = self.read(summary['result_path'], summary['result_sha256'])
        self.dependencies(result)
        require(result['all_named_buffers_bitwise_unchanged'] is True
                and result['all_training_data_file_stats_unchanged'] is True, 'collection mutation')
        raw = self.npz(summary['raw_path'], summary['raw_sha256'])
        require(folds['collection_raw_sha256'] == summary['raw_sha256'], 'fit collection raw differs')
        close(raw['row_indices'], self.indices, 'collection row order', atol=0, rtol=0)
        sequences = np.array([self.rows[i]['sequence'] for i in self.indices])
        require(np.array_equal(raw['sequence'], sequences), 'collection sequence order differs')
        require(raw['e_i'].shape == raw['e_d'].shape == (2016, 51)
                and raw['trajectory_incoming'].shape == (2016, 4, 51), 'collection raw shapes differ')
        require(np.array_equal(raw['gate'], raw['gates']), 'gate aliases differ')
        main_i, direct = raw['trajectory_gated_motion'][:, -1], raw['trajectory_direct'][:, -1]
        _, ji = self.fk(main_i, raw['beta']); _, jd = self.fk(direct, raw['beta'])
        _, jg = self.fk(raw['truth'], raw['beta']); _, jp = self.fk(raw['previous_truth'], raw['beta'])
        ei, ed = main_i.astype(float)-raw['truth'], direct.astype(float)-raw['truth']
        ei[:, :3], ed[:, :3] = (ji-jg).mean(1), (jd-jg).mean(1)
        rg = Rotation.from_rotvec(raw['truth'][:, 3:6])
        ei[:, 3:6] = (rg.inv()*Rotation.from_rotvec(main_i[:, 3:6])).as_rotvec()
        ed[:, 3:6] = (rg.inv()*Rotation.from_rotvec(direct[:, 3:6])).as_rotvec()
        close(raw['e_i'], ei, 'actual physical I residual', atol=8e-7, rtol=2e-5)
        close(raw['e_d'], ed, 'actual physical D residual', atol=8e-7, rtol=2e-5)
        speed = np.linalg.norm((jg-jp)/.05, axis=-1).mean(1)*1000
        close(raw['gt_joint_speed_mm_s'], speed, 'fit-side physical joint speed', atol=.02, rtol=5e-5)
        gate = np.minimum(raw['trajectory_short_count'][:, -1].astype(np.float32)/1000., 1.)
        close(raw['gate'], gate, 'fit D gate from actual counts', atol=2e-8, rtol=1e-7)
        close(raw['gate'], raw['trajectory_motion_gate'][:,-1], 'main gate and actual trajectory differ', atol=0, rtol=0)
        for ri, index in enumerate(self.indices):
            row = self.rows[index]; handle = self.dataset._handle(row['sequence_id']); end = row['end_ms']
            require(np.array_equal(raw['beta'][ri], handle.betas) and np.array_equal(raw['camera_K'][ri], handle.camera_K), 'collection context differs')
            require(np.array_equal(raw['initial_truth'][ri], handle.pos51[end-199])
                    and np.array_equal(raw['initial'][ri], handle.pos51[end-199]+self.draws[index]), 'collection initial is not declared GT+draw')
            require(np.array_equal(raw['truth'][ri], handle.pos51[end])
                    and np.array_equal(raw['previous_truth'][ri], handle.pos51[end-50]), 'collection true motion clock differs')
            for step in range(4):
                require(np.array_equal(raw['trajectory_truth'][ri, step], handle.pos51[end-150+50*step]), 'collection prefix GT differs')
                expected = raw['initial'][ri] if step == 0 else raw['trajectory_fused'][ri, step-1]
                require(np.array_equal(raw['trajectory_incoming'][ri, step], expected), 'collection does not own its incoming state')
        fallback = np.asarray(summary['baseline_effective_gains'])
        expected = (self.model.gains().float()*torch.tensor([.5]*3+[.25]+[.5]*15)).numpy()
        close(fallback, expected, 'fallback is not original effective gain', atol=0, rtol=0)
        identities = np.array([s.split('_')[0] for s in sequences]); families = np.array([s.split('_v')[0] for s in sequences])
        require(set(folds['identity']) == set(identities) and set(folds['motion_family']) == set(families)
                and set(folds['trial']) == set(sequences), 'crossfit coverage differs')
        entries = [('all_train', None, folds['all_train'])]
        entries += [('identity', name, f) for name, f in folds['identity'].items()]
        entries += [('motion_family', name, f) for name, f in folds['motion_family'].items()]
        for scheme, name, value in entries:
            mask = np.ones(2016, bool) if name is None else (identities != name if scheme == 'identity' else families != name)
            require(value['train_indices'] == self.indices[mask].tolist()
                    and value['excluded_indices'] == self.indices[~mask].tolist(), 'held-out fitting labels leaked')
            coefficients = coefficient_fit(raw['e_i'][mask], raw['e_d'][mask], raw['gate'][mask].astype(float),
                                            fallback, self.decl['fitting']['denominator_floor'])
            close(value['coefficients'], coefficients, 'independent constrained coefficient fit', atol=1e-9, rtol=1e-9)
            require(value['high_motion_threshold_mm_s'] == float(np.quantile(raw['gt_joint_speed_mm_s'][mask], .75)),
                    'high-motion threshold uses held-out or different labels')
        for name, value in folds['trial'].items():
            require(value == folds['motion_family'][name.split('_v')[0]], 'four motion versions do not share one whole-family fit')
        return dict(rows=2016, physical_residuals_recomputed=True, independent_folds=28,
                    identity_folds=9, motion_family_folds=18, sequence_aliases=72)

    def metric_values(self, pred, truth, beta):
        shape = pred.shape[:-1]; flat = pred.reshape(-1, 51)
        gt = np.broadcast_to(truth[:, None, None], pred.shape).reshape(-1, 51)
        vp, jp = self.fk(flat, beta); vg, jg = self.fk(gt, beta)
        dj, dv = (jp-jg)*1000, (vp-vg)*1000
        wrist = dj[:, 0]; relative = dj-wrist[:, None]; rv = dv-wrist[:, None]
        dt = (flat[:, :3].astype(float)-gt[:, :3])*1000
        root = np.degrees((Rotation.from_rotvec(gt[:, 3:6]).inv()*Rotation.from_rotvec(flat[:, 3:6])).magnitude())
        values = np.stack([np.linalg.norm(dj, axis=-1).mean(1), np.linalg.norm(relative, axis=-1).mean(1),
            np.linalg.norm(dv, axis=-1).mean(1), np.linalg.norm(rv, axis=-1).mean(1),
            np.linalg.norm(dt, axis=-1), np.sum(dt*dt, axis=-1), dt[:, 2], dt[:, 2]**2,
            np.linalg.norm(wrist, axis=-1), np.sum(wrist*wrist, axis=-1), wrist[:, 2], wrist[:, 2]**2,
            root, np.sqrt(np.mean((flat[:, 6:].astype(float)-gt[:, 6:])**2, axis=1)),
            np.mean(np.sum(dj*dj, axis=-1), axis=1), np.mean(np.sum(relative*relative, axis=-1), axis=1),
            2*np.sum(wrist*relative.mean(1), axis=-1)], axis=-1)
        return values.reshape(*shape, 17), jp.reshape(*shape, 21, 3)

    def clip(self, seed, shard, clip, raw, folds):
        n, count = len(clip['planned_end_ms']), 5
        shapes = {'incoming':(n,count,3,51), 'predictions':(n,count,3,51), 'metrics':(n,count,3,17),
            'truth':(n,51), 'previous_truth':(n,51), 'initial':(count,51), 'clean':(51,), 'beta':(10,), 'camera_K':(3,3),
            'condition_indices':(5,), 'coefficients':(3,19), 'gaussian_draw':(51,),
            'motion_predictions':(n,count,3,51), 'gated_motion_predictions':(n,count,3,51), 'direct_predictions':(n,count,3,51),
            'effective_gains':(n,count,3,19), 'motion_gate':(n,count,3), 'joint_motion_mm':(n,count,3,21,3),
            'joint_motion_error_mm':(n,count,3,21,3), 'true_joint_motion_mm':(n,21,3), 'true_joint_speed_mm_s':(n,),
            'n_raw_events':(n,), 'lnes_window_ms':(n,), 'end_ms':(n,), 'previous_gt_ms':(n,), 'input_sha256':(n,),
            'raw_event_sha256':(n,), 'lnes_sha256':(n,)}
        require(set(raw) == set(shapes) and all(raw[k].shape == v for k, v in shapes.items()), 'validation raw dimension/field mismatch')
        conditions = list(range(shard*5, shard*5+5)); require(raw['condition_indices'].tolist() == conditions, 'condition shard changed')
        index = clip['canonical2304_row_index']; row = self.rows[index]
        require(row == clip['canonical_anchor'] and index in self.indices, 'validation canonical anchor differs')
        handle = self.dataset._handle(row['sequence_id']); beta = raw['beta']
        require(np.array_equal(beta, handle.betas) and np.array_equal(raw['camera_K'], handle.camera_K), 'validation ordinary context differs')
        ends = np.asarray(clip['planned_end_ms']); prevms = np.r_[clip['declared_initial_gt_ms'], ends[:-1]]
        require(np.array_equal(raw['end_ms'], ends) and np.array_equal(raw['previous_gt_ms'], prevms), 'validation clock/censoring differs')
        require(np.array_equal(raw['truth'], handle.pos51[ends]) and np.array_equal(raw['previous_truth'], handle.pos51[prevms]), 'validation GT clock differs')
        require(np.array_equal(raw['clean'], handle.pos51[prevms[0]]) and np.array_equal(raw['gaussian_draw'], self.draws[index]), 'validation initialization labels/draw differs')
        close(raw['initial'], initial_conditions(raw['clean'], self.draws[index])[conditions], 'declared ten initial conditions', atol=2e-6, rtol=2e-6)
        incoming = np.concatenate([np.broadcast_to(raw['initial'][:, None], (5,3,51))[None], raw['predictions'][:-1]], 0)
        require(np.array_equal(raw['incoming'], incoming), 'candidate does not own its incoming state')
        foldrows = [None, folds['identity'][clip['training_identity']], folds['trial'][clip['sequence']]]
        coefficients = np.asarray([folds['baseline_effective_gains']]+[f['coefficients'] for f in foldrows[1:]])
        close(raw['coefficients'], coefficients, 'whole-clip fixed coefficient vector', atol=0, rtol=0)
        gate = np.clip(raw['n_raw_events'].astype(np.float32)/1000., 0., 1.)
        expected_gate = np.broadcast_to(gate[:, None, None], (n,5,3))
        close(raw['motion_gate'], expected_gate, 'actual I evidence gate', atol=2e-8, rtol=1e-7)
        expected_k = np.broadcast_to(gate[:,None,None,None]*coefficients[None,None].astype(np.float32), (n,5,3,19))
        close(raw['effective_gains'], expected_k, 'effective D coefficient applied exactly once', atol=1e-7, rtol=1e-6)
        gated = interpolated_motion(incoming, raw['motion_predictions'], expected_gate)
        close(raw['gated_motion_predictions'], gated, 'independent I root interpolation', atol=3e-6, rtol=3e-6)
        require(np.array_equal(raw['direct_predictions'], np.broadcast_to(raw['direct_predictions'][:,:1,:1], (n,5,3,51))), 'D output depends on prior or policy')
        values, joints = self.metric_values(raw['predictions'], raw['truth'], beta)
        difference = close(raw['metrics'], values, 'all17 metrics from actual state FK', atol=.01, rtol=5e-5)
        _, initial_j = self.fk(raw['initial'], beta)
        old_j = np.concatenate([np.broadcast_to(initial_j[:,None], (5,3,21,3))[None], joints[:-1]], 0)
        flow = (joints-old_j)*1000
        _, jt = self.fk(raw['truth'], beta); _, jp = self.fk(raw['previous_truth'], beta)
        true_flow = (jt-jp)*1000
        close(raw['joint_motion_mm'], flow, 'physical own-state joint flow', atol=.003, rtol=5e-5)
        close(raw['true_joint_motion_mm'], true_flow, 'true physical joint flow', atol=.003, rtol=5e-5)
        close(raw['joint_motion_error_mm'], flow-true_flow[:,None,None], 'physical flow residual', atol=.005, rtol=5e-5)
        speed = np.linalg.norm(true_flow, axis=-1).mean(1)/((ends-prevms)/1000.)
        close(raw['true_joint_speed_mm_s'], speed, 'actual49/50ms physical speed', atol=.02, rtol=5e-5)
        # Independently verify root/fingers and centroid mixing, not just metrics.
        a, b, k = gated.reshape(-1,51), raw['direct_predictions'].reshape(-1,51).astype(float), raw['effective_gains'].reshape(-1,19).astype(float)
        ra = Rotation.from_rotvec(a[:,3:6]); innovation = (ra.inv()*Rotation.from_rotvec(b[:,3:6])).as_rotvec()
        innovation *= np.minimum(1., np.deg2rad(60.)/np.maximum(np.linalg.norm(innovation,axis=1,keepdims=True),1e-9))
        root = root_branches((ra*Rotation.from_rotvec(k[:,3:4]*innovation)).as_rotvec(), a[:,3:6])
        root[k[:,3] == 0] = a[k[:,3] == 0,3:6]
        fingers = a[:,6:]+np.repeat(k[:,4:],3,axis=1)*(b[:,6:]-a[:,6:])
        live = np.repeat(raw['n_raw_events'] > 0, 15)
        pred = raw['predictions'].reshape(-1,51)
        close(pred[live,3:6], root[live], 'actual root innovation/clip', atol=6e-6, rtol=4e-6)
        close(pred[live,6:], fingers[live], 'actual finger interpolation', atol=3e-6, rtol=3e-6)
        _, ja = self.fk(raw['gated_motion_predictions'], beta); _, jb = self.fk(raw['direct_predictions'][:,:1,:1], beta)
        cb = np.broadcast_to(jb.reshape(n,1,1,21,3).mean(3), (n,5,3,3)).reshape(-1,3)
        centroid = (1-k[:,:3])*ja.mean(1)+k[:,:3]*cb
        close(joints.reshape(-1,21,3).mean(1)[live], centroid[live], 'actual coupled centroid fusion', atol=2e-6, rtol=3e-5)
        require(np.array_equal(raw['predictions'][gate == 0], incoming[gate == 0]), 'empty packet must hold exactly')
        shared_fields = ('truth','previous_truth','clean','beta','camera_K','gaussian_draw','true_joint_motion_mm',
                         'true_joint_speed_mm_s','n_raw_events','lnes_window_ms','end_ms','previous_gt_ms','input_sha256',
                         'raw_event_sha256','lnes_sha256')
        sharedkey = (seed, clip['sequence'])
        if shard == 0:
            self.shared[sharedkey] = {key: raw[key].copy() for key in shared_fields}
            self.shared[sharedkey]['direct'] = raw['direct_predictions'][:,0,0].copy()
        else:
            require(all(np.array_equal(raw[key], self.shared[sharedkey][key]) for key in shared_fields), 'paired shards use different inputs')
            require(np.array_equal(raw['direct_predictions'][:,0,0], self.shared[sharedkey]['direct']), 'paired shards D differs')
        # Reduce raw data only; producer per-clip reductions are not trusted.
        for ci, condition in enumerate(conditions):
            for mi, mode in enumerate(MODES):
                scopes = [('all_observed',np.ones(n,bool)), ('main_onward',np.arange(n)>=3)]
                if n == 10: scopes.append(('full10_main',np.arange(n)>=3))
                for scope, mask in scopes:
                    metric = raw['metrics'][mask,ci,mi].astype(float)
                    self.records.append(dict(seed=seed, identity=clip['training_identity'], clip=clip['sequence'],
                        family=clip['sequence'].split('_v')[0], condition=self.decl['conditions'][condition], mode=mode,
                        scope=scope, frames=int(mask.sum()), full10=n==10, abs_mpjpe_mm=float(metric[:,0].mean()),
                        joint_mse_mm2=float(metric[:,14].mean()), root_mean_deg=float(metric[:,12].mean()),
                        physical_wrist_mean_mm=float(metric[:,8].mean()),
                        joint_jitter_mm_per_step=float(np.linalg.norm(raw['joint_motion_mm'][mask,ci,mi],axis=-1).mean())))
                if condition == 0:
                    for scheme, fi in (('identity_cv',1), ('family_cv',2)):
                        mask = raw['true_joint_speed_mm_s'] >= foldrows[fi]['high_motion_threshold_mm_s']
                        for main_only in (False,True):
                            selected = mask & (np.arange(n)>=3 if main_only else np.ones(n,bool))
                            if selected.any():
                                predicted = raw['joint_motion_mm'][selected,ci,mi].astype(float)
                                target = raw['true_joint_motion_mm'][selected].astype(float)
                                self.high[(seed,scheme,mode,main_only)].append(dict(n=int(selected.sum()), clip=clip['sequence'],
                                    error_energy=float(np.sum((predicted-target)**2)), dot=float(np.sum(predicted*target)),
                                    true_energy=float(np.sum(target*target))))
        return difference

    def seed(self, seed):
        self.model, cfg, self.identity = C.load_bound_model(self.decl, seed, torch.device('cpu'))
        model_before = C.snapshot_model(self.model)
        self.digest(self.identity['checkpoint'], self.identity['checkpoint_sha256'])
        self.digest(self.identity['config'], self.identity['config_sha256'])
        self.dataset, rows, _, _ = C.training_inputs(cfg, self.decl)
        require(rows == self.rows, 'canonical dataset differs')
        folds = self.read(self.root/f'folds_s{seed}.json'); self.dependencies(folds)
        require(folds['status'] == 'FITTED_NOT_ACCEPTED' and folds['accepted_for_dev'] is False
                and folds['seed'] == seed and folds['declaration_sha256'] == self.digest(self.declaration_path), 'wrong unaccepted fold candidates')
        collection = self.collection(seed, folds)
        fkmax = 0.
        for shard in (0,1):
            directory = self.root/f'validation_s{seed}_shard{shard}'
            summary = self.read(directory/'summary.json'); self.dependencies(summary)
            protocol = self.read(directory/'protocol.json', summary['protocol_sha256']); self.dependencies(protocol)
            require(summary['status'] == 'COMPLETE_CONDITION_SHARD' and summary['partial_shard'] is True
                    and not summary['partial_smoke'] and not protocol['partial_smoke']
                    and summary['clips'] == 72 and summary['observed_steps_per_condition'] == 690
                    and summary['actual_full_B1_calls'] == summary['actual_D_forward_calls'] == 690*5*3,
                    'partial/cached/missing validation shard')
            require(summary['identity'] == protocol['identity'] == self.identity
                    and summary['declaration_sha256'] == protocol['declaration_sha256'] == self.digest(self.declaration_path)
                    and summary['fold_specs_sha256'] == protocol['fold_specs_sha256'] == self.digest(self.root/f'folds_s{seed}.json'),
                    'validation model/declaration/fold binding mismatch')
            require(protocol['clips'] == self.schedule and protocol['conditions'] == self.decl['conditions']
                    and protocol['condition_indices'] == summary['condition_indices'] == list(range(shard*5,shard*5+5))
                    and protocol['modes'] == summary['modes'] == list(MODES)
                    and protocol['metrics'] == list(METRICS), 'shard condition/mode/schedule mismatch')
            require(summary['baseline_real_B1_parity']['bitwise_equal'] is True
                    and summary['all65_named_buffers_unchanged'] is True
                    and summary['all_parameters_unchanged_sha256'] == model_before['parameters'], 'model mutation or baseline parity failure')
            require(summary['named_buffer_sha256']=={name:C.array_sha(value.numpy()) for name,value in model_before['buffers'].items()},
                    'runtime buffers differ from frozen checkpoint')
            require(summary['all_training_data_file_stats_unchanged'] is True and summary['no_development_data_loaded'] is True
                    and summary['finite_subset_reduction'] is False, 'training scope or complete-data invariant differs')
            for handle in self.dataset._handles.values():
                require(summary['data_file_stat_identity'][handle.seq]==C.data_file_signature(handle)
                        and summary['training_domain_sha256'][handle.seq]==C.array_sha(handle.pos51,handle.betas,handle.camera_K,handle.valid_runs),
                        'canonical source files or physical label/context content changed')
            self.digest(directory/'per_clip.jsonl', summary['per_clip_sha256'])
            require(len(summary['raw_sha256']) == len(list((directory/'raw').glob('*.npz'))) == 72, 'raw clip file denominator mismatch')
            for ci, clip in enumerate(self.schedule):
                filename=f'{ci:03d}_{clip["sequence"]}.npz'
                raw=self.npz(directory/'raw'/filename,summary['raw_sha256'][filename])
                fkmax=max(fkmax,self.clip(seed,shard,clip,raw,folds))
            print(json.dumps(dict(seed=seed,shard=shard,raw_clips=72,CPU_FK_max_metric_difference=fkmax)),flush=True)
        # Bind independent true labels/context used for all collection and scoring.
        for handle in self.dataset._handles.values():
            for suffix in ('.meta','_aux.npz','_offsets.npy'):
                self.digest(handle.base+suffix)
        C.verify_model(self.model,model_before)
        self.validation[str(seed)] = dict(collection=collection, shards=2, clips_per_shard=72,
            conditions=10, observed_steps_per_condition=690, modes=3, actual_B1_forward_calls=20700,
            CPU_FK_max_metric_difference=fkmax, complete_raw_replay=True)

    def reduce(self):
        primary=primary_reduction(self.records); failures=[]
        if not primary['mean_gt_one_paired_SE']: failures.append(dict(check='primary_identity_CV_improvement_gt_one_SE',details=primary))
        for seed,passed in primary['per_seed_noninferiority'].items():
            if not passed: failures.append(dict(check='primary_per_seed_noninferiority',seed=int(seed),improvement_mm2=primary['per_seed_mean_improvement_mm2'][seed]))
        clean=[]; high=[]
        for seed in SEEDS:
            for scheme in MODES[1:]:
                for scope in ('all_observed','main_onward','full10_main'):
                    baseline=balanced_absolute(self.records,seed,'baseline',scope);candidate=balanced_absolute(self.records,seed,scheme,scope)
                    cell=dict(seed=seed,scheme=scheme,scope=scope,baseline_mm=baseline,candidate_mm=candidate,passed=candidate<=baseline)
                    clean.append(cell)
                    if not cell['passed']: failures.append(dict(check='clean_noninferiority',**cell))
                baseline=flow_reduction(self.high[(seed,scheme,'baseline',False)]);candidate=flow_reduction(self.high[(seed,scheme,scheme,False)])
                require(baseline['frames']==candidate['frames'] and baseline['true_energy_mm2']==candidate['true_energy_mm2'], 'high motion paired denominators differ')
                assessable=baseline['projection']>0
                rmse=candidate['joint_motion_rmse_mm']<=1.05*baseline['joint_motion_rmse_mm']
                projection=assessable and candidate['projection']>=.95*min(baseline['projection'],1.)
                cell=dict(seed=seed,scheme=scheme,baseline=baseline,candidate=candidate,baseline_projection_positive=assessable,
                          rmse_pass=rmse,projection_pass=projection,passed=rmse and projection,
                          main_onward_descriptive={mode:(flow_reduction(self.high[(seed,scheme,mode,True)])
                                                       if self.high[(seed,scheme,mode,True)] else dict(frames=0))
                                                   for mode in ('baseline',scheme)})
                high.append(cell)
                if not cell['passed']:failures.append(dict(check='clean_high_motion_guard',**cell))
        return dict(primary=primary,clean=clean,high_motion=high),failures


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('declaration','root','output'):parser.add_argument('--'+key,required=True,type=Path)
    args=parser.parse_args();require(not args.output.exists(),'refusing to overwrite independent assessment')
    args.output.mkdir(parents=True);torch.set_num_threads(4)
    source={**C.sources(),str(Path(__file__).relative_to(ROOT)):C.sha(__file__),
            'tests/test_xkf_statistics_assess.py':C.sha(ROOT/'tests/test_xkf_statistics_assess.py')}
    audit=None;failures=[];gates={}
    try:
        audit=Audit(args.declaration,args.root)
        for seed in SEEDS:audit.seed(seed)
        gates,failures=audit.reduce()
        for path,digest in audit.inputs.items():require(C.sha(path)==digest,'final input hash changed: '+path)
        require(source=={name:C.sha(ROOT/name) for name in source},'assessment source changed')
    except Exception as error:
        failures.append(dict(check='completeness_or_independent_raw_validation',error=type(error).__name__,message=str(error)))
    accepted=not failures and audit is not None and len(audit.validation)==3
    result=dict(kind='INDEPENDENT_TRAIN_ONLY_PAIRED_STATISTICS_ASSESSMENT',
        status='ACCEPTED_FOR_DEV_EVALUATION' if accepted else 'REJECTED_TRAIN_VALIDATION', accepted_for_dev=accepted,
        automatic_full_training=False, formal=False, declaration_path=str(args.declaration.resolve()),declaration_sha256=C.sha(args.declaration),
        source_sha256=source,input_sha256={} if audit is None else audit.inputs,failures=failures,
        verification={} if audit is None else audit.validation,gates=gates,
        no_neural_forward=True,no_development_data_loaded=True,no_policy_or_checkpoint_change=True,
        scope='Frozen declaration_v2 reducers only. Acceptance authorizes a separately declared development evaluation, never full training. '
              'Original G1-G4/readiness are unchanged; all failed gates and all eight large perturbation conditions remain reported.')
    if audit is not None:
        records=args.output/'raw_reduced_cells.json'
        records.write_text(json.dumps(audit.records,indent=2,allow_nan=False)+'\n')
        result['all_condition_raw_reductions']=dict(path=str(records.resolve()),sha256=C.sha(records))
    (args.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],failures=len(failures),output=str(args.output/'result.json'))),flush=True)
    return 0 if accepted or gates else 2


if __name__=='__main__':raise SystemExit(main())
