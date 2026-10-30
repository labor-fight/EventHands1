"""Independent old-evaluator reproduction check; GT-init is diagnostic only.

This module does not edit or monkeypatch any replay/model/memory dependency.
Run the new standard-initialization replay in one process, ``old`` below in a
separate process, then ``compare``. The old path executes the unchanged
semkine.eval_track.track_sequence and captures predictions with its state_hook.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/s43_integration_20261027'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda:file.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()


def _json(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,(np.integer,np.floating)):return value.item()
    if isinstance(value,Path):return str(value)
    raise TypeError(type(value).__name__)


def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=_json)+'\n')


def old_replay(seq,out,threads=1):
    os.environ['CUDA_VISIBLE_DEVICES']='';torch.set_num_threads(threads)
    # Fresh process and the original evaluator's own public imports, not the
    # integrated wrapper or its private metric-only function extraction.
    for p in (str(ROOT),str(ROOT/'model')):
        if p not in sys.path:sys.path.insert(0,p)
    from semkine import eval_track as ET
    from config import load_config
    config=ROOT/'outputs/semkine/s43n_sp_s3407/config_resolved.yaml'
    checkpoint=ROOT/'outputs/semkine/s43n_sp_s3407/s43n_sp_s3407-step=6000.ckpt'
    cfg=load_config(config)
    model=ET.MNISTModel.load_from_checkpoint(str(checkpoint),cfg=cfg,map_location='cpu').cpu().eval()
    mano=ET.ManoLayer(cfg['MANO']['NPZ'],add_mean=False).cpu().eval()
    data_root=Path(cfg['DATA']['ROOT'])
    split=json.loads((data_root/'splits_semkine.json').read_text())
    legacy=None
    for part in ('train','val_core'):
        if seq in split[part]['trials']:legacy=split[part]['legacy_dir'][seq]
    if legacy is None:raise ValueError('sequence absent from the unchanged split')
    base=data_root/legacy/seq
    dependencies=[Path(__file__),ROOT/'semkine/eval_track.py',ROOT/'model/model.py',
                  ROOT/'model/mano_layer.py',ROOT/'model/pose_repr.py',ROOT/'semkine/events.py',
                  config,checkpoint,data_root/'splits_semkine.json',ROOT/'assets/mano_right.npz']
    dependencies += [Path(str(base)+suffix) for suffix in ('_events.npy','_offsets.npy','_aux.npz','.meta','_tsub.npy')]
    sources={str(p.resolve()):sha(p) for p in dependencies}
    prefix=Path(out);prefix.parent.mkdir(parents=True,exist_ok=True)
    if Path(str(prefix)+'.json').exists() or Path(str(prefix)+'.npz').exists():raise FileExistsError(prefix)
    preds=[];initial=[];started=time.perf_counter()
    def hook(kind,prev,pred):
        if kind=='run_start':initial.append(prev.detach().cpu().numpy()[0].copy());return None
        if kind!='step':raise ValueError(kind)
        preds.append(pred.detach().cpu().numpy()[0].copy())
        if len(preds)%100==0:print(json.dumps({'old_sequence':seq,'packets':len(preds),
                                              'seconds':time.perf_counter()-started}),flush=True)
        return pred
    result=ET.track_sequence(model,mano,cfg,data_root,legacy,seq,50,torch.device('cpu'),
                             np.random.default_rng(0),noise_scale=1.,window_ms=50,state_hook=hook)
    if result is None:raise ValueError('no old evaluator predictions')
    _,_,aux,targets=ET.load_sequence(data_root,legacy,seq)
    ends=[];run_ids=[]
    for run_id,(a,b) in enumerate(np.asarray(aux['valid_runs_ms'],np.int64).reshape(-1,2)):
        row=np.arange(int(a)+49,int(b),50,dtype=np.int64)
        ends.extend(row.tolist());run_ids.extend([run_id]*len(row))
    aux.close()
    pred=np.asarray(preds,np.float32)
    if pred.shape!=(len(ends),51):raise AssertionError('old hook sample count differs from evaluator clock')
    arrays=Path(str(prefix)+'.npz')
    np.savez_compressed(arrays,pred=pred,gt=targets[np.asarray(ends)],end_ms=np.asarray(ends,np.int64),
                        run_id=np.asarray(run_ids,np.int64),initial=np.asarray(initial,np.float32))
    unchanged={path:sha(path)==value for path,value in sources.items()}
    if not all(unchanged.values()):raise RuntimeError('old evaluator dependency changed while running')
    payload={'sequence':seq,'path':'original semkine.eval_track.track_sequence, unmodified',
             'GT_initialization':True,'role':'explicit standard-protocol reproduction diagnostic, not deployment',
             'checkpoint':str(checkpoint),'config':str(config),'rng':'default_rng(0)','noise_scale':1.,
             'step_ms':50,'window_ms':50,'metrics':result,'predictions':len(pred),
             'seconds':time.perf_counter()-started,'source_sha256_before':sources,
             'source_sha256_unchanged':unchanged,'arrays':str(arrays),'arrays_sha256':sha(arrays),
             'python':sys.executable,'numpy':np.__version__,'torch':torch.__version__,
             'threads':threads,'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES')}
    write(Path(str(prefix)+'.json'),payload)
    print(json.dumps({'old_report':str(prefix)+'.json','predictions':len(pred),'seconds':payload['seconds']}),flush=True)


def compare(new,old,out):
    new_json=json.loads(Path(str(new)+'.json').read_text())
    old_json=json.loads(Path(str(old)+'.json').read_text())
    with np.load(str(new)+'.npz',allow_pickle=False) as z:new_arrays={k:z[k] for k in ('pred','gt','end_ms','run_id')}
    with np.load(str(old)+'.npz',allow_pickle=False) as z:old_arrays={k:z[k] for k in ('pred','gt','end_ms','run_id')}
    exact={key:bool(np.array_equal(new_arrays[key],old_arrays[key])) for key in new_arrays}
    if new_arrays['pred'].shape!=old_arrays['pred'].shape:raise AssertionError('new/old shape mismatch')
    maximum=float(np.max(np.abs(new_arrays['pred']-old_arrays['pred'])))
    keys=('mpjpe_abs_mm','mpvpe_abs_mm','mpjpe_ra_mm','mpvpe_ra_mm')
    metric_errors={key:abs(new_json['metrics'][key]-old_json['metrics'][key]) for key in keys}
    metrics_match=all(value<=1e-5 for value in metric_errors.values())
    prediction_match=maximum<=1e-6
    passed=prediction_match and metrics_match and exact['gt'] and exact['end_ms'] and exact['run_id']
    result={'at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
            'PASS':bool(passed),'sequence':new_json['sequence'],'GT_initialization':True,
            'scope':'old standard W50/raw evaluator reproduction, not deployment or new accuracy evidence',
            'new':str(new),'old':str(old),'predictions':len(new_arrays['pred']),
            'exact_arrays':exact,'max_abs_pred_difference':maximum,
            'prediction_atol':1e-6,'metric_abs_errors_mm':metric_errors,'metric_atol_mm':1e-5,
            'new_metrics':new_json['metrics'],'old_metrics':{key:old_json['metrics'][key] for key in keys},
            'new_all_sources_unchanged':all(new_json['source_files_unchanged'].values()),
            'old_all_sources_unchanged':all(old_json['source_sha256_unchanged'].values()),
            'artifact_sha256':{str(Path(str(p)+suffix)):sha(str(p)+suffix)
                               for p in (new,old) for suffix in ('.json','.npz')},
            'executed_sources':{'new':new_json['source_files_before'],'old':old_json['source_sha256_before']}}
    write(out,result)
    print(json.dumps({key:result[key] for key in ['PASS','sequence','predictions','exact_arrays',
                        'max_abs_pred_difference','metric_abs_errors_mm']}),flush=True)
    if not passed:raise SystemExit('legacy reproduction tolerance failed; result preserved')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='stage',required=True)
    p=sub.add_parser('old');p.add_argument('--seq',default='ycy_global');p.add_argument('--out',required=True,type=Path)
    p.add_argument('--threads',type=int,default=1)
    p=sub.add_parser('compare');p.add_argument('--new',required=True,type=Path);p.add_argument('--old',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    a=parser.parse_args()
    if a.stage=='old':old_replay(a.seq,a.out,a.threads)
    else:compare(a.new,a.old,a.out)


if __name__=='__main__':main()
