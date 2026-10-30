#!/usr/bin/env python3
"""Frozen, zero-training focal/depth geometry preflight for the XDE study.

Only outputs/xde/e3_preflight is written. CUDA, if requested, is restricted to
physical GPU 6. Candidate identities use the production top16/incident-face
semantics, never a single nearest vertex. No training is launched here.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/xde/e3_preflight'
CONFIG = ROOT / 'configs/xdl/xdl_q1_s3407.yaml'
DEPENDENCIES = [
    'tools/xde/e3_geometry.py', 'tools/xdl/data.py', 'semkine/domrand.py',
    'semkine/dataset.py', 'semkine/xyz_mesh.py', 'semkine/events.py',
    'model/mano_layer.py', 'model/pose_repr.py', 'model/config.py', 'assets/mano_right.npz',
    'tools/xcl/sources/step1_source_labels.py', 'configs/xdl/xdl_q1_s3407.yaml',
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def save(name, value):
    path = OUT / name
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        'study': 'XDE-E3-GEOMETRY-PREFLIGHT-20261014',
        'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'authorization': 'Parent delegated zero-training preflight only; physical GPU6 only; no formal training.',
        'n_states': 1000,
        'sample_rule': 'np.linspace(0,6000*1024-1,1000,dtype=int64); XDL seed3407 counter clip_selection and shared_domrand streams; main target state.',
        'augmentation_rule': 'Exact Q1 DomRand parameter distribution and named RNG; same draw changes only scale_mode focal to depth.',
        'projection_rule': 'All 778 vertices per state; exact floating image affine of original projections is reference; compare transformed-label FK projection under transformed K. Pool all finite vertices without selecting by residual.',
        'projection_gate': {'metric': 'pooled_vertex_residual_event_px_p95', 'maximum': 3.0},
        'candidate_rule': 'Visible reference surface = original triangles carried by the image affine (same per-vertex depths), z-buffer all faces at integer event pixels; up to128 visible pixels per state at deterministic evenly spaced raster indices. On same reference rays compare image-exact mesh and depth-label mesh: production top16 angular-nearest vertices -> union of their incident faces. A query changes identity if these face sets differ.',
        'candidate_gate': {'metric': 'fraction_queries_with_any_incident_face_candidate_set_change', 'maximum': .05},
        'candidate_secondary': ['fraction_top16_sets_changed', 'mean_top16_members_replaced_fraction', 'mean_face_symmetric_difference_over_union', 'production_associate_rays_hit_near_none_change', 'positive_weight_vertex_support_set_change'],
        'early_stop': 'If projection p95 exceeds3px, skip full1000-state candidate audit, retain first32-state fixed candidate sample as descriptive only. Never declare full candidate gate passed from this subset.',
        'scope': 'Conservative implementation admissibility gate, not a test that depth coverage is scientifically useless. No outcome-driven scale tuning.',
        'events': 'All events in each sampled main W50 window; record geometric OOB, keep-mask removal, hot additions, and exact focal/depth transformed event equality. State sample is exposure-weighted, not sequence-balanced.',
        'source_sha256': {p: sha(ROOT / p) for p in DEPENDENCIES},
        'split_sha256': sha(ROOT / 'data/hand_data51/splits_semkine.json'),
        'command': 'CUDA_VISIBLE_DEVICES=6 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 /data1/lyq/miniconda3/envs/EventHandsTrain/bin/python tools/xde/e3_geometry.py --run --device cuda',
    }
    if (OUT / 'protocol.json').exists():
        raise RuntimeError('Protocol already frozen; refusing overwrite')
    save('protocol.json', protocol)
    print(json.dumps({'status': 'FROZEN', 'protocol_sha256': sha(OUT / 'protocol.json')}), flush=True)


def identity_audit(torch, yaml):
    from config import load_config
    meta_path = ROOT / 'outputs/semkine/xdl_q1_s3407/training_metadata.json'
    old_freeze = ROOT / 'outputs/xdl/frozen_manifest.json'
    meta = json.loads(meta_path.read_text())
    frozen = json.loads(old_freeze.read_text())
    entries = frozen['source_sha256']
    # Restrict drift decisions to training/evaluation code, config and model asset.
    relevant = {p: h for p, h in entries.items()
                if p.endswith('.py') or p == 'assets/mano_right.npz' or p == 'configs/xdl/xdl_q1_s3407.yaml'}
    checks = {p: {'expected': h, 'actual': sha(ROOT / p), 'match': sha(ROOT / p) == h}
              for p, h in relevant.items() if (ROOT / p).exists()}
    missing = [p for p in relevant if not (ROOT / p).exists()]
    ckpts = list((ROOT / 'outputs/semkine/xdl_q1_s3407').glob('*step=6000.ckpt'))
    ckpt = torch.load(ckpts[0], map_location='cpu', weights_only=False) if len(ckpts) == 1 else {}
    cfg = load_config(CONFIG)
    cfg.pop('_config_path')
    resolved = yaml.safe_load((meta_path.parent / 'config_resolved.yaml').read_text())
    gates = {
        'unique_exact6000': len(ckpts) == 1 and ckpt.get('global_step') == 6000,
        'metadata_config_sha_matches': sha(CONFIG) == meta['config_sha256'],
        'resolved_config_matches': resolved == cfg,
        'old_freeze_sha_matches': sha(old_freeze) == meta['freeze_manifest_sha256'],
        'source_sha_matches': not missing and all(x['match'] for x in checks.values()),
        'split_sha_matches': sha(ROOT / 'data/hand_data51/splits_semkine.json') == meta['split_sha256'],
        'focal_default': cfg['AUG']['DOMRAND'].get('SCALE_MODE', 'focal') == 'focal',
        'formal_fresh_two_rank512': not meta['preflight_not_formal'] and meta['resumed_from'] is None and meta['start_step'] == 0 and meta['devices'] == 2 and meta['batch_size_per_gpu'] == 512 and meta['accumulate_grad_batches'] == 1 and meta['max_steps'] == 6000,
        'rawW50_fullburn_self': cfg['DATA']['WINDOW_MIN'] == cfg['DATA']['WINDOW_MAX'] == 50 and not cfg['AUG']['SPEED_AUG'] and cfg['MODEL']['XDL_BURN_MODE'] == 'full' and cfg['MODEL']['XDL_SELF'] and cfg['MODEL']['XDL_QUERY'] == 'mano',
    }
    value = {'status': 'PASS' if all(gates.values()) else 'FAIL', 'gates': gates,
             'source_checks': checks, 'missing': missing,
             'checkpoint': str(ckpts[0]) if len(ckpts) == 1 else None,
             'checkpoint_sha256': sha(ckpts[0]) if len(ckpts) == 1 else None,
             'old_freeze_python': frozen['python'], 'current_python': sys.executable,
             'current_torch': torch.__version__,
             'metadata_sha256': sha(meta_path), 'scope': 'Identity eligibility only. Does not authorize training or establish new repeat comparability.'}
    save('q1_reuse_identity.json', value)
    print('Q1 identity', value['status'], flush=True)
    return value


def run(device):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '6':
        raise RuntimeError('Must explicitly set CUDA_VISIBLE_DEVICES=6, even for CPU phase')
    sys.path[:0] = [str(ROOT), str(ROOT / 'model')]
    import numpy as np
    import torch
    import yaml
    from tools.xdl.data import build_train_dataset, counter_rng
    from semkine import domrand as DR
    from semkine.xyz_mesh import associate_rays, vertex_face_table
    from mano_layer import ManoLayer
    from pose_repr import decode_to_mano_inputs

    torch.set_num_threads(8)
    started = time.monotonic()
    protocol = json.loads((OUT / 'protocol.json').read_text())
    for p, expected in protocol['source_sha256'].items():
        if sha(ROOT / p) != expected:
            raise RuntimeError('Frozen code changed: ' + p)
    if sha(ROOT / 'data/hand_data51/splits_semkine.json') != protocol['split_sha256']:
        raise RuntimeError('Frozen split changed')
    identity = identity_audit(torch, yaml)
    cfg = yaml.safe_load(CONFIG.read_text())
    ds = build_train_dataset(cfg)
    sample_indices = np.linspace(0, 6000 * 1024 - 1, 1000, dtype=np.int64)
    states, focal_states, depth_states, betas, cams, focal_cams, depth_cams = [], [], [], [], [], [], []
    manifests, augments, counts = [], [], []
    event_equal = True
    for ix, sample_index in enumerate(sample_indices):
        step, row = divmod(int(sample_index), 1024)
        legal_row = int(counter_rng(3407, step, row, 'clip_selection').integers(len(ds.legal_index)))
        si, end, run_a, run_b = map(int, ds.legal_index[legal_row])
        h = ds._handle(si)
        sample = DR.sample_params(ds.domrand, counter_rng(3407, step, row, 'shared_domrand'), 180*240*2*50)
        other = dataclasses.replace(sample, scale_mode='depth')
        state = h.pos51[end].copy()
        fstate, fcam = DR.transform_labels(state, sample, h.camera_K, h.j0, .375)
        dstate, dcam = DR.transform_labels(state, other, h.camera_K, h.j0, .375)
        raw = ds._raw_events(h, end, 50)
        x, y = raw[0].copy(), raw[1].copy()
        inbounds = DR.transform_events(x, y, sample, h.camera_K, .375, 240, 180)
        keep = DR.keep_mask(len(x), sample.keep, counter_rng(3407, step, row, 'events_3'))
        combined = inbounds if keep is None else inbounds & keep
        fev = ds._domrand_events(*[a.copy() for a in raw], 50, sample, h.camera_K, counter_rng(3407, step, row, 'events_3'))
        dev = ds._domrand_events(*[a.copy() for a in raw], 50, other, h.camera_K, counter_rng(3407, step, row, 'events_3'))
        equal = all(np.array_equal(a, b) for a, b in zip(fev, dev))
        event_equal &= equal
        counts.append([len(x), int(inbounds.sum()), int(combined.sum()), sample.n_hot, len(fev[0])])
        assert len(fev[0]) == int(combined.sum()) + sample.n_hot
        states.append(state); focal_states.append(fstate); depth_states.append(dstate)
        betas.append(h.betas); cams.append(h.camera_K); focal_cams.append(fcam); depth_cams.append(dcam)
        augments.append(dataclasses.asdict(sample))
        manifests.append({'sample_index': int(sample_index), 'optimizer_step': step, 'row': row,
                          'legal_row': legal_row, 'seq': h.seq, 'end_ms': end, 'run': [run_a, run_b],
                          'j0_z_m': float(h.j0[2]), 'event_modes_equal': equal})
        if (ix+1) % 100 == 0:
            print('sampled', ix+1, 'elapsed_s', round(time.monotonic()-started, 2), flush=True)
    save('samples.json', {'identities': manifests, 'augmentations': augments,
                         'index_sha256': hashlib.sha256(sample_indices.tobytes()).hexdigest(),
                         'legal_coverage': ds.coverage})
    del ds
    states, focal_states, depth_states, betas, cams, focal_cams, depth_cams = [np.asarray(x, np.float32) for x in (states, focal_states, depth_states, betas, cams, focal_cams, depth_cams)]
    mano = ManoLayer(ROOT / 'assets/mano_right.npz', add_mean=False).eval()
    @torch.no_grad()
    def fk(params):
        arrays=[]
        for start in range(0,len(params),100):
            p=torch.from_numpy(params[start:start+100]); b=torch.from_numpy(betas[start:start+100])
            dec=decode_to_mano_inputs(p,'mano_full_axis_angle',mano.hands_components,mano.hands_mean)
            v,_=mano(b,dec['global_orient'],dec['local_full_aa'],dec['transl'])
            arrays.append(v.numpy())
        return np.concatenate(arrays)
    def project(v,k):
        return v[...,:2]/v[...,2:3] * (.375*np.stack([k[:,0,0],k[:,1,1]],-1))[:,None,:] + (.375*k[:,:2,2])[:,None,:]
    v0,vf,vd=[fk(p) for p in (states,focal_states,depth_states)]
    uv0=project(v0,cams)
    ref=uv0.astype(np.float64).copy()
    for i,sm in enumerate(augments):
        center=.375*cams[i,:2,2]; u=uv0[i]-center
        c,s=np.cos(sm['roll_rad']),np.sin(sm['roll_rad'])
        ref[i]=sm['scale']*(u@np.array([[c,s],[-s,c]]))+center+sm['shift_px']
    residual=np.linalg.norm(project(vd,depth_cams)-ref,axis=-1)
    focal_residual=np.linalg.norm(project(vf,focal_cams)-ref,axis=-1)
    def summary(a):
        a=np.asarray(a).reshape(-1)
        return {'n':len(a),'mean':float(a.mean()),'p50':float(np.quantile(a,.5)),
                'p95':float(np.quantile(a,.95)),'p99':float(np.quantile(a,.99)),'max':float(a.max())}
    projection_pass=bool(np.quantile(residual,.95)<=3)
    j0z=np.asarray([m['j0_z_m'] for m in manifests])
    zf=(focal_states[:,2]+j0z)*1000; zd=(depth_states[:,2]+j0z)*1000
    ec=np.asarray(counts,np.int64); totals=ec.sum(0)
    projection={'depth_residual_event_px':summary(residual),'focal_residual_event_px':summary(focal_residual),
                'projection_gate': 'PASS' if projection_pass else 'FAIL',
                'per_state_p95':summary(np.quantile(residual,.95,axis=1)),
                'n_states_per_state_p95_gt3':int((np.quantile(residual,.95,axis=1)>3).sum()),
                'focal_root_depth_mm':summary(zf),'depth_root_depth_mm':summary(zd),
                'n_focal_in_local699_742mm':int(((zf>=699)&(zf<=742)).sum()),
                'n_depth_in_local699_742mm':int(((zd>=699)&(zd<=742)).sum()),
                'n_focal_above710p2mm':int((zf>710.2).sum()),'n_depth_above710p2mm':int((zd>710.2).sum()),
                'sampled_sequences':len({m['seq'] for m in manifests}),
                'event_counts_columns':['raw','geometric_inbounds','after_keep','hot_added','final'],
                'event_counts_total':totals.tolist(),
                'geometry_oob_fraction':float(1-totals[1]/totals[0]),
                'keep_removed_fraction_of_inbounds':float(1-totals[2]/totals[1]),
                'focal_depth_event_arrays_exact_equal_all1000':bool(event_equal)}
    np.savez_compressed(OUT/'geometry_arrays.npz',states=states,focal_states=focal_states,depth_states=depth_states,
                        root_depth_focal_mm=zf,root_depth_depth_mm=zd,vertex_residual_px=residual,
                        focal_vertex_residual_px=focal_residual,event_counts=ec,sample_indices=sample_indices)
    save('projection.json',projection)
    print('PROJECTION',json.dumps(projection),flush=True)

    # Load only the authoritative rasterize function, not the module which forces
    # CUDA_VISIBLE_DEVICES="" at import. Its AST is compiled unchanged, source hashed.
    tree=ast.parse((ROOT/'tools/xcl/sources/step1_source_labels.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='rasterize')
    namespace={'np':np,'H':180,'W':240,'BARY_EPS':1e-5}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'XA:rasterize','exec'),namespace)
    rasterize=namespace['rasterize']
    dev=torch.device(device)
    if dev.type=='cuda':
        assert torch.cuda.device_count()==1
        torch.cuda.set_device(0)
    faces=mano.f.to(dev); table=vertex_face_table(faces,778).to(dev)
    n_candidate_states=1000 if projection_pass else 32
    candidate_rows=[]
    gpu_started=time.monotonic()
    for i in range(n_candidate_states):
        k=depth_cams[i]; fx,fy,cx,cy=.375*np.array([k[0,0],k[1,1],k[0,2],k[1,2]])
        # Anisotropic depth remap exactly represents the reference image. Its
        # visibility ordering stays original; it is a diagnostic reference only.
        vr=np.c_[((ref[i,:,0]-cx)/fx)*v0[i,:,2],((ref[i,:,1]-cy)/fy)*v0[i,:,2],v0[i,:,2]].astype(np.float32)
        Z,_,_=rasterize(vr,mano.f.numpy(),np.zeros(len(mano.f),np.int64),(fx,fy,cx,cy))
        pixels=np.flatnonzero(np.isfinite(Z))
        chosen=pixels[np.linspace(0,len(pixels)-1,min(128,len(pixels)),dtype=np.int64)] if len(pixels) else pixels
        if not len(chosen):
            candidate_rows.append({'sample':i,'n':0,'visible_pixels':0});continue
        py,px=np.divmod(chosen,240)
        d=torch.tensor(np.c_[(px-cx)/fx,(py-cy)/fy,np.ones(len(px))],dtype=torch.float32,device=dev)[None]
        mask=torch.ones((1,len(px)),dtype=torch.bool,device=dev)
        verts=[torch.tensor(v[None],dtype=torch.float32,device=dev) for v in (vr,vd[i])]
        sets=[];assocs=[]
        for v in verts:
            uv=v[...,:2]/v[...,2:3].clamp_min(1e-6)
            uv=uv.masked_fill((v[...,2:3]<=1e-6),1e6)
            delta=uv[:,None,:,:]-d[:,:,:2][:,:,None,:]
            ids=delta.square().sum(-1).topk(16,dim=-1,largest=False).indices[0]
            fids=table[ids].reshape(len(px),-1)
            sets.append((ids.cpu().numpy(),fids.cpu().numpy()))
            assocs.append(associate_rays(d,mask,v,faces,table,
                band=torch.tensor([16/np.sqrt(fx*fy)],device=dev),
                front_tol=torch.tensor([3/np.sqrt(fx*fy)],device=dev),z_tol=.01,k=16))
        fchange=vchange=typechange=supportchange=0; vrepl=[];fjacc=[]
        av=[{key:value.cpu().numpy()[0] for key,value in a.items()} for a in assocs]
        for j in range(len(px)):
            vs=[set(x[0][j].tolist()) for x in sets]
            fs=[set(x[1][j][x[1][j]>=0].tolist()) for x in sets]
            vchange+=vs[0]!=vs[1];fchange+=fs[0]!=fs[1]
            vrepl.append(len(vs[0]-vs[1])/16);fjacc.append(len(fs[0]^fs[1])/max(1,len(fs[0]|fs[1])))
            kinds=[int(a['hit'][j])*2+int(a['valid'][j]) for a in av]
            typechange+=kinds[0]!=kinds[1]
            supports=[set(a['vid'][j][a['q'][j]*a['A'][j]>0].tolist()) for a in av]
            supportchange+=supports[0]!=supports[1]
        candidate_rows.append({'sample':i,'n':len(px),'visible_pixels':len(pixels),
                               'face_set_changed':int(fchange),'top16_set_changed':int(vchange),
                               'top16_replaced_fraction_sum':float(sum(vrepl)),
                               'face_symmetric_over_union_sum':float(sum(fjacc)),
                               'hit_near_none_changed':int(typechange),'positive_vertex_support_changed':int(supportchange)})
        if (i+1)%100==0:print('candidates',i+1,'elapsed_s',round(time.monotonic()-gpu_started,2),flush=True)
    fields=['face_set_changed','top16_set_changed','top16_replaced_fraction_sum','face_symmetric_over_union_sum','hit_near_none_changed','positive_vertex_support_changed']
    den=sum(r['n'] for r in candidate_rows)
    candidate={'n_states':n_candidate_states,'n_queries':den,'full_registered1000':projection_pass,
               'fractions':{key:sum(r.get(key,0) for r in candidate_rows)/den for key in fields},
               'candidate_seconds':time.monotonic()-gpu_started,'device':str(dev),
               'candidate_gate':'NOT_RUN_FULL_PROJECTION_FAILED' if not projection_pass else ('PASS' if sum(r.get('face_set_changed',0) for r in candidate_rows)/den<=.05 else 'FAIL'),
               'rows':candidate_rows}
    save('candidates.json',candidate)
    overall=projection_pass and candidate['candidate_gate']=='PASS' and identity['status']=='PASS'
    result={'status':'PASS' if overall else 'FAIL','projection_gate':projection['projection_gate'],
            'candidate_gate':candidate['candidate_gate'],'q1_reuse_identity':identity['status'],
            'projection_p95_event_px':projection['depth_residual_event_px']['p95'],
            'candidate_identity_changed_fraction':candidate['fractions']['face_set_changed'],
            'total_seconds':time.monotonic()-started,'candidate_seconds':candidate['candidate_seconds'],
            'physical_gpu':6 if dev.type=='cuda' else None,'training_launched':False,
            'protocol_sha256':sha(OUT/'protocol.json'),
            'interpretation':'Admissibility only; a failed discrete candidate identity gate does not refute depth-support hypothesis. Formal training requires merged E1/E2 and parent decision.',
            'artifacts_sha256':{name:sha(OUT/name) for name in ['q1_reuse_identity.json','samples.json','projection.json','geometry_arrays.npz','candidates.json']}}
    save('result.json',result)
    print('RESULT',json.dumps(result),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');ap.add_argument('--run',action='store_true');ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');args=ap.parse_args()
    os.chdir(ROOT)
    if args.freeze:freeze()
    elif args.run:run(args.device)
    else:ap.error('Choose --freeze or --run')
