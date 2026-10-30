"""Add explicit legal beta/K access to all cached arm contexts, without GT.

This engineering fairness repair happened before inspecting any outcome.
Evidence/pixel features/counts/labels/split are unchanged. Old files preserved.
"""
from pathlib import Path
import argparse
import numpy as np
from .cache_data import load_cache
from .synthetic import load_synthetic
from .representations import geometry_conditions
from .experiment import load_features,write_json,sha,freeze_stage


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--kind',choices=['synthetic','existing'],required=True)
    p.add_argument('--samples',type=Path,required=True); p.add_argument('--axis',required=True)
    a=p.parse_args(); freeze_stage(a.out.stem)
    data=load_features(a.source)
    samples=load_synthetic(a.samples,axis=a.axis) if a.kind=='synthetic' else load_cache(a.samples)
    if data['context'].shape[1]!=972 or len(samples)!=len(data['label']):
        raise ValueError('unexpected context layout, cannot upgrade twice')
    extra=[]
    for i,s in enumerate(samples):
        group=s['context_id'] if a.kind=='synthetic' else s['recording']
        if group!=data['group'][i]: raise ValueError('sample order mismatch')
        camera=s['K_cam'].copy()
        if a.kind=='existing': camera[:2]*=.375
        extra.append(geometry_conditions(s['betas'],camera))
    data['context']=np.concatenate([data['context'],np.stack(extra)],axis=1)
    if a.out.exists(): raise FileExistsError(a.out)
    np.savez_compressed(a.out,**data)
    write_json(a.out.with_suffix('.json'),{'source':str(a.source),'source_sha':sha(a.source),
               'sha256':sha(a.out),'added_context':'explicit beta[10] and normalized K_event[9], all arms',
               'no_target_in_constructor':True,'features_labels_splits_unchanged':True,
               'polarity': 'physical ON/OFF, controlled generator' if a.kind=='synthetic' else
                           'stored 0/1, unverified 0-to-ON assumption, model-conditional scores',
               'aged_prior':None if a.kind=='synthetic' else 'same held-state approximation, prior timestamp t0-1ms'})
    print(a.out)


if __name__=='__main__': main()
