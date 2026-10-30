"""Post-completion state51/FK export audit, without labels or task metrics.

Fixed 0/mid/last rows per MANO51 fit. No training, targets, risk scores,
checkpoint selection or tolerance changes. Wait for all74 predictions first.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib, json, os
from pathlib import Path
import numpy as np
import torch

from . import newfield_mano_probe as P
from .field_mano_decoder import FieldManoDecoder


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for part in iter(lambda:stream.read(1<<20),b''):h.update(part)
    return h.hexdigest()


def audit(root,freeze,out):
    root,freeze,out=Path(root),Path(freeze),Path(out)
    if out.exists():raise FileExistsError(out)
    marker=root/'all_predictions_complete.json'
    if not marker.exists():raise RuntimeError('wait for all74 fixed predictions before export audit')
    done=json.loads(marker.read_text(encoding='utf-8'))
    if done.get('fits')!=74 or done.get('optimizer_steps')!=148000:
        raise RuntimeError('all74 fixed-fit completion required')
    f=json.loads(freeze.read_text(encoding='utf-8'))
    if f['dependencies']!=P.dependencies() or f['protocol_sha256']!=P.PROTOCOL_SHA:
        raise RuntimeError('frozen source changed')
    torch.set_num_threads(2)
    decoder=FieldManoDecoder().eval()
    jobs=[];meta={};feature_rows={}
    for domain in ('synthetic','existing'):
        feature=root/'features'/domain
        complete=json.loads((feature/'complete.json').read_text(encoding='utf-8'))
        if complete['rows_sha256']!=sha(feature/'rows.json') or complete['source_freeze_sha256']!=sha(freeze):
            raise RuntimeError('prepared observation metadata identity changed')
        # labels.json is deliberately never opened, even to recover anchor/beta.
        rows=json.loads((feature/'rows.json').read_text(encoding='utf-8'))
        feature_rows[domain]=rows
        meta[domain]=dict(rows_sha256=sha(feature/'rows.json'),complete_sha256=sha(feature/'complete.json'))
        for arm in P.ARMS:
            for reader in P.READERS:
                jobs.append((domain,arm,reader,'all3',root/'readers'/domain/arm/reader,feature))
    for arm in ('hand_box','static'):
        for reader in P.READERS:
            jobs.append(('existing',arm,reader,'nominal_guard',root/'readers_nominal_guard/existing'/arm/reader,
                         root/'features/existing'))
    # Full all74 gate (including joint63 controls) before inspecting learned poses.
    for domain,arm,reader,regime,folder,feature in jobs:
        P.check_predictions(folder,feature,domain,reader,freeze,regime)
        c=json.loads((folder/'predictions_complete.json').read_text(encoding='utf-8'))
        if c['domain']!=domain or c['arm']!=arm or c['reader']!=reader:
            raise RuntimeError('prediction identity mismatch')
        if any(a['steps']!=2000 or a['batch']!=32 for a in c['accounts'].values()):
            raise RuntimeError('incomplete fixed fit')
    records=[];states=[];anchors=[];betas=[];raws=[];references=[]
    for domain,arm,reader,regime,folder,feature in jobs:
        if reader!='mano51':continue
        rows=[r for r in feature_rows[domain] if r['arm']==arm]
        for fold in (range(4) if domain=='synthetic' else [-1]):
            path=folder/f'fold{fold}/predictions.npz'
            with np.load(path,allow_pickle=False) as z:
                n=len(z['state51']);indices=[0,n//2,n-1]
                for i in indices:
                    context=np.asarray(rows[i]['context'],np.float32)
                    states.append(z['state51'][i]);anchors.append(context[:51]);betas.append(context[51:61])
                    raws.append(z['raw_output'][i]);references.append(z['prediction_joints_m'][i])
                    records.append(dict(domain=domain,arm=arm,regime=regime,fold=fold,row=i,
                        case_id=rows[i]['case_id'],condition=rows[i]['condition'],prediction_file=str(path),
                        prediction_sha256=sha(path)))
    assert len(records)==111
    state=torch.tensor(np.asarray(states));anchor=torch.tensor(np.asarray(anchors));beta=torch.tensor(np.asarray(betas))
    raw=torch.tensor(np.asarray(raws));ref=torch.tensor(np.asarray(references))
    with torch.inference_mode():
        from_export=decoder(state,beta,torch.zeros_like(state)).joints
        from_action=decoder(anchor,beta,raw).joints
    export_error=(from_export-ref).abs().flatten(1).amax(1).numpy()
    action_error=(from_action-ref).abs().flatten(1).amax(1).numpy()
    finite=bool(torch.isfinite(from_export).all() and torch.isfinite(from_action).all())
    for r,x,y in zip(records,export_error,action_error):
        r.update(export_FK_max_abs_m=float(x),raw_action_FK_max_abs_m=float(y))
    passed=finite and bool(np.max(export_error)<=1e-6 and np.max(action_error)<=1e-6)
    result=dict(schema='s43_stage4_mano_export_independent_audit_v1',status='PASS' if passed else 'FAIL',
        created_utc=datetime.now(timezone.utc).isoformat(),reviewer='/root/rel_geometry',
        source_sha256=sha(__file__),decoder_sha256=sha(Path(P.ROOT)/'tools/s43_revisit/field_mano_decoder.py'),
        sourcefreeze_sha256=sha(freeze),completion_marker_sha256=sha(marker),prepared_observations=meta,
        all74_fits_validated_before_export=True,mano_fits=37,fixed_samples=111,indices='0, floor(N/2), N-1 per fit',
        tolerance_m=1e-6,finite=finite,export_FK_max_abs_m=float(np.max(export_error)),
        raw_action_FK_max_abs_m=float(np.max(action_error)),records=records,
        GT_labels_opened=False,task_metrics_computed=False,training=False,device='cpu',
        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
        caveat='fixed-row physical export equivalence only; not exhaustive every-frame verification or task performance')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','mano_fits','fixed_samples','export_FK_max_abs_m','raw_action_FK_max_abs_m')}))
    if not passed:raise RuntimeError('learned MANO state export exceeds frozen1e-6m tolerance; preserve FAIL receipt')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=P.OUT/'field_mano_atomic_v1')
    p.add_argument('--source-freeze',type=Path,default=P.OUT/'field_mano_engineering/actual_sourcefreeze.json')
    p.add_argument('--out',type=Path,default=P.OUT/'newfield_mano_export_audit.json')
    a=p.parse_args();audit(a.root,a.source_freeze,a.out)


if __name__=='__main__':main()
