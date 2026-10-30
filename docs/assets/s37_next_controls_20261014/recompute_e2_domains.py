#!/usr/bin/env python3
"""Read existing E2 arrays on CPU; never load a model or run inference/training."""
from pathlib import Path
import hashlib
import json
import os
import sys
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
E2 = ROOT / 'outputs/xde/e2'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Explicit CUDA_VISIBLE_DEVICES="" required; CPU artifact audit only')
    sources = {}

    def source(path):
        path = Path(path)
        if not path.is_absolute(): path = ROOT / path
        key = os.path.relpath(path, ROOT)
        sources[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def read_json(path):
        return json.loads(source(path).read_text())

    source(__file__)
    protocol = read_json(E2/'protocol.json')
    bank = read_json('outputs/xdl/packet_bank.json')
    mano = np.load(source('assets/mano_right.npz'),allow_pickle=False)
    j0s = {}
    arrays, meta = {}, []
    names = sorted({r['sequence'] for r in protocol['selected_packets']})
    for name in names:
        path = source(E2/'cache'/f'{name}.npz')
        audit = read_json(path.with_suffix('.json'))
        assert sources[os.path.relpath(path,ROOT)] == audit['cache_sha256']
        with np.load(path,allow_pickle=False) as value:
            for k in value.files: arrays.setdefault(k,[]).append(value[k])
        meta.extend(audit['rows'])
        aux_path = next(s['path'] for s in bank['sequences'][name]['sources'] if s['path'].endswith('_aux.npz'))
        beta = np.load(source(aux_path),allow_pickle=False)['betas'].astype(np.float64)
        shaped = mano['v_template'].astype(np.float64) + np.einsum('l,mkl->mk',beta,mano['shapedirs'].astype(np.float64))
        j0s[name] = (mano['J_regressor'].astype(np.float64) @ shaped)[0] * 1000.
    data = {k:np.concatenate(v) for k,v in arrays.items()}
    assert len(meta)==len(data['target51'])==7680
    assert all(np.isfinite(v).all() for v in data.values())
    target = data['target_xyz_mm'].astype(np.float64)
    # Match the actual float32 subtraction in feature extraction, then audit the mm target.
    assert np.array_equal(data['target_xyz_mm'],(data['target51'][:,:3]-data['incoming51'][:,:3])*1000.)
    split = np.array([r['split'] for r in meta])
    sequence = np.array([r['sequence'] for r in meta])
    dz = np.array([r['dz_mm'] for r in meta])
    j0 = np.stack([j0s[name] for name in sequence])
    gt_t = data['target51'][:,:3].astype(np.float64)*1000.
    prev_t = data['incoming51'][:,:3].astype(np.float64)*1000.
    gt_wrist = gt_t+j0
    prev_wrist = prev_t+j0
    base_groups = {'fit':split=='fit','check':split=='check',
                   'check_local':(split=='check')&np.char.find(sequence.astype(str),'_local').__ge__(0),
                   'check_global':(split=='check')&np.char.find(sequence.astype(str),'_global').__ge__(0),
                   'zgz_local':sequence=='zgz_local','zgz_global':sequence=='zgz_global'}
    groups = {}
    for key,mask in base_groups.items():
        groups[key+'/mixed'] = mask
        for v in (0.,-50.,50.): groups[key+f'/dz{int(v):+d}'] = mask&(dz==v)

    def summary(value):
        value = np.asarray(value,np.float64)
        return {'n':len(value),'min':float(value.min()),'max':float(value.max()),
                'mean':float(value.mean()),'mean_absolute':float(np.abs(value).mean()),
                'p05':float(np.quantile(value,.05)),'p50':float(np.quantile(value,.5)),
                'p95':float(np.quantile(value,.95))}

    fit = split=='fit'
    fit_t_bounds = [float(gt_t[fit,2].min()),float(gt_t[fit,2].max())]
    fit_w_bounds = [float(gt_wrist[fit,2].min()),float(gt_wrist[fit,2].max())]
    domains = {}
    for key,mask in groups.items():
        domains[key] = {'rows':int(mask.sum()),
                       'target_transl_z_mm':summary(gt_t[mask,2]),
                       'incoming_transl_z_mm':summary(prev_t[mask,2]),
                       'target_actual_wrist_z_mm':summary(gt_wrist[mask,2]),
                       'incoming_actual_wrist_z_mm':summary(prev_wrist[mask,2]),
                       'required_delta_z_mm':summary(target[mask,2]),
                       'hold_wrist_error_mm':float(np.linalg.norm(target[mask],axis=-1).mean()),
                       'target_transl_z_above_fit_target_max':int((gt_t[mask,2]>fit_t_bounds[1]).sum()),
                       'target_actual_wrist_z_above_fit_target_max':int((gt_wrist[mask,2]>fit_w_bounds[1]).sum())}

    def score(prediction,mask):
        error = prediction[mask]-target[mask]
        hold = target[mask]
        return {'n':int(mask.sum()),'z_mae_mm':float(np.abs(error[:,2]).mean()),
                'z_signed_bias_mm':float(error[:,2].mean()),
                'z_rmse_mm':float(np.sqrt(np.square(error[:,2]).mean())),
                'z_absolute_error_p95_mm':float(np.quantile(np.abs(error[:,2]),.95)),
                'wrist_error_mm':float(np.linalg.norm(error,axis=-1).mean()),
                'hold_z_mae_mm':float(np.abs(hold[:,2]).mean()),
                'hold_wrist_error_mm':float(np.linalg.norm(hold,axis=-1).mean()),
                'z_better_than_hold_count':int((np.abs(error[:,2])<np.abs(hold[:,2])).sum()),
                'wrist_better_than_hold_count':int((np.linalg.norm(error,axis=-1)<np.linalg.norm(hold,axis=-1)).sum()),
                'negative_z_error_count':int((error[:,2]<0).sum())}

    heads = {}
    for width in protocol['head']['widths']:
        for interface in protocol['interfaces']:
            for seed in protocol['head']['seeds']:
                key = f'{interface}_w{width}_s{seed}'
                path = source(E2/'heads'/f'{key}.predictions.npz')
                with np.load(path,allow_pickle=False) as value:
                    assert np.array_equal(value['target_xyz_mm'],data['target_xyz_mm'])
                    predicted = value['delta_xyz_mm'].astype(np.float64)
                details = read_json(E2/'heads'/f'{key}.json')
                heads[key] = {'interface':interface,'width':width,'head_seed':seed,
                              'best_epoch':details['best_epoch'],
                              'nominal_parameters':details['nominal_parameters'],
                              'variable_input_dimensions':details['variable_input_dimensions'],
                              'metrics':{group:score(predicted,mask) for group,mask in groups.items()}}
    assert len(heads)==20
    q1_delta=(data['prediction51'][:,:3]-data['incoming51'][:,:3]).astype(np.float64)*1000.
    q1={group:score(q1_delta,mask) for group,mask in groups.items()}
    rollouts={}
    for seq in ('zgz_local','zgz_global'):
        old = read_json(E2/f'rollout_{seq}.pt.json')
        rollouts[seq] = {key:{k:v for k,v in value.items() if k.startswith('first_') or k.startswith('final_') or k in ('n_clips','n_queries','new_sustained_recovery','initially_satisfied')}
                         for key,value in old['metrics'].items()}
    supplemental=read_json(E2/'supplemental_audit.json')
    cfg=yaml.safe_load(source('configs/xdl/xdl_q1_s3407.yaml').read_text())
    for name in ('tools/xde/e2_probes.py','semkine/xyz_recovery_study.py','semkine/xyz_observation.py','semkine/xyz_mesh.py','model/model.py','model/pose_repr.py'):
        source(name)
    value={'study':'E2 existing-cache domain and clean-protection audit 20261014',
           'scope':'CPU-only descriptive recomputation of existing arrays. No inference, training, GPU use, new checkpoint selection, or change to E2 registered gates.',
           'command':'CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /data1/lyq/miniconda3/envs/EventHandsTrain/bin/python docs/assets/s37_next_controls_20261014/recompute_e2_domains.py',
           'row_contract':{'n':7680,'sequence_order':names,'within_sequence_order':'cache JSON rows; clean,-50,+50 for each selected packet','groups':'fit/check include only training identities; zgz_local/global are evaluation sequences; mixed includes all three dz values','depth_coordinate':'transl_z and camera-frame actual wrist_z are reported separately. actual_wrist_z=transl_z+J0(beta)_z; beta is sequence-specific. The fitted target depth support below is ONLY these E2 probe samples, not all original Q1 training exposure.','target':'(GT endpoint transl-incoming transl)*1000; fixed-beta endpoint wrist difference is identical algebraically','precision':'Input arrays float32; target subtraction verified against actual float32 extraction; descriptive moments computed float64.'},
           'fit_target_transl_z_range_mm':fit_t_bounds,'fit_target_actual_wrist_z_range_mm':fit_w_bounds,
           'sequence_J0_xyz_mm':{k:v.tolist() for k,v in j0s.items()},'domains':domains,'heads':heads,'original_Q1_matched_cache':q1,
           'rollout_first_and_final_metrics':rollouts,'rollout_first_angle_equality_audit':supplemental['audit'],
           'source_contracts':{
               'original_Q1_final_input':'legacy XYZ544 + observation bank64 + state99 =707 (semkine/xyz_recovery_study.py:61,75-85).',
               'probe_input':'state99 + observation bank16x34; post uses4 channels and pads30 zeros per slot. No legacy XYZ544 (tools/xde/e2_probes.py:228-235,352-357).',
               'probe_state_distribution':'GT preceding state with z only0/-50/+50mm; remaining48 rotations and xy unchanged GT (tools/xde/e2_probes.py:161-163).',
               'root_loss':cfg['LOSS'].get('ROOT_LOSS','mse_aa'),
               'loss_type':cfg['LOSS'].get('TYPE','mse_51d'),
               'original_loss_config':cfg['LOSS'],
               'original_loss_formula':'log10((450*MSE(fingerAA45)+60*MSE(rootAA3)+30000*MSE(endpoint_translation3))/51); means are within each block and rank-local minibatch before log10. No FK or SO3 geodesic root loss in this Q1 configuration.',
               'translation_equivalence':'For a fixed incoming t, ||delta-(t_GTnext-t_in)||^2 = ||(t_in+delta)-t_GTnext||^2. With fixed beta this is also endpoint wrist squared error. Therefore merely renaming the translation term to wrist endpoint MSE does not change its mathematical target. Joint training weights/logloss/shared features and access-state distribution remain different between Q1 and probes.',
               'feedback':'Each branch feeds its full own state. Only translation output is replaced; Q1 roots/fingers recompute from changed FK/association/UV next step (tools/xde/e2_probes.py:343-360).',
               'causal_limit':'Generalization gap, domain support gap, incomplete probe interface, and multiblock rollout shift are simultaneous observations, not identified contributions.'},
           'source_sha256':sources}
    path=OUT/'e2_domain_clean_audit.json'
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False,sort_keys=True)+'\n')
    print(json.dumps({'output':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                      'fit_target_transl_z_range_mm':fit_t_bounds,'fit_target_actual_wrist_z_range_mm':fit_w_bounds,
                      'local_actual_wrist_range':domains['zgz_local/dz+0']['target_actual_wrist_z_mm'],
                      'heads':len(heads),'rows':len(meta)}))


if __name__=='__main__': main()
