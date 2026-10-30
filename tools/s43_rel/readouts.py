"""Postprocess frozen E12 heads; never refit or select hyperparameters."""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from . import mechanisms as M


def load_head(path, device='cpu'):
    import torch
    state = torch.load(path, map_location='cpu', weights_only=False)
    head = M.RegressionHead(state['kind'], device)
    normal = M.Normalize()
    normal.__dict__.update(state.pop('normal'))
    model = state.pop('model', None)
    head.__dict__.update(state)
    head.normal = normal
    head.device = device
    if head.kind == 'mlp':
        dims = int(model['0.weight'].shape[1])
        head.model = torch.nn.Sequential(torch.nn.Linear(dims,128),torch.nn.SiLU(),
            torch.nn.Linear(128,128),torch.nn.SiLU(),torch.nn.Linear(128,63)).to(device)
        head.model.load_state_dict(model)
        head.model.eval()
    return head


def summarize(out=M.OUT):
    import torch
    torch.set_num_threads(1)
    final = dict(protocol_sha256=M.sha(M.PROTOCOL),source_sha256=M.sha(M.__file__),
        postprocess_source_sha256=M.sha(__file__),complete=True,seed=M.SEED,domains={},
        qualification=['Local50ms equal-boundary resets are mechanism snippets, not continuous deployment.',
            'Existing labels are hand_data51 pseudo-GT; no zgz or real-sensor task readout was used.',
            'Time destruction keeps the target fixed and is nonphysical; no different-target matched physical-pair proof.',
            'MSE is proper conditional-mean regression, without predictive variance or calibration claims.',
            'SELECT has only two existing subjects; windows do not enlarge the independence count.',
            'Initial implementation_manifest was written before construction and scores by main(); a subsequent heads-only call rewrote the identical configuration/source identity and its timestamp. No parameters changed.'])
    for domain in ('existing','synthetic'):
        directory=out/domain
        with np.load(out/f'{domain}_features.npz',allow_pickle=False) as z: values={k:z[k] for k in z.files}
        report=json.loads((directory/'metrics.json').read_text())
        manifest=json.loads((out/f'{domain}_features.manifest.json').read_text())
        select=values['partition']=='SELECT'
        predictions={};by_domain={};rows={}
        for kind in ('rbf','mlp'):
            rows[kind]={};by_domain[kind]={}
            for variant in ('original','destroy'):
                by_domain[kind][variant]={}
                for arm in M.ARMS:
                    head=load_head(directory/f'{kind}_{variant}_{arm}.pt')
                    name=arm if variant=='original' else arm+'_destroy'
                    pred=head.predict(values[name],values['context'])
                    predictions[f'{kind}__{variant}__{arm}']=pred.astype(np.float32)
                    by_domain[kind][variant][arm]={}
                    for condition in np.unique(values['domain'][select]):
                        mask=select&(values['domain']==condition)
                        by_domain[kind][variant][arm][str(condition)]=M.risks(pred[mask],
                            values['prior_joints'][mask],values['target_joints'][mask],values['recording'][mask])
                    if variant=='original':
                        q=report['E1'][kind]['original'][arm]['SELECT']
                        rows[kind][arm]={k:q[k] for k in ('abs_mm','ra_mm','mse_mm2')}
        np.savez_compressed(directory/'predictions.npz',**predictions)
        e2_support={}
        for condition,result in report['E2'].items():
            e2_support[condition]={kind:result[kind]['fresh']['abs_mm']<=.98*result[kind]['stale']['abs_mm']
                                   for kind in ('rbf','mlp')}
        matching={}
        if domain=='synthetic':
            # Equality of full protected raw marginals necessarily implies
            # equality of this fixed pooled transform. A zero candidate count
            # therefore certifies that these generated contexts supplied NO
            # exact matched pair; it never certifies a match after compression.
            cheap=values['enhanced'].reshape(len(select),48,20)[:,:,:14]
            tested=candidates=0
            for recording in np.unique(values['recording']):
                ids=np.flatnonzero(values['recording']==recording)
                for ii,a in enumerate(ids):
                    for b in ids[ii+1:]:
                        if np.max(np.abs(values['target_joints'][a]-values['target_joints'][b]))<1e-6:continue
                        tested+=1
                        candidates+=int(np.array_equal(cheap[a],cheap[b]))
            matching=dict(different_target_same_public_context_pairs_tested=tested,
                exact_protected_marginal_matching_candidates_after_necessary_pool_filter=candidates,
                certified_exact_physical_matched_pairs=0,
                outcome='not found' if not candidates else 'compressed candidates require raw verification; no match claimed',
                tolerance=0.,independent_contexts_per_axis=30,heldout_contexts_per_axis=8)
        final['domains'][domain]=dict(n=len(select),SELECT_n=int(select.sum()),E1_task_support=report['E1_support'],
            E1_rows=rows,by_domain=by_domain,E2_support_per_head=e2_support,E2=report['E2'],
            raw_same_core_parity=manifest['raw_parity'],raw_same_core_max_error=manifest['raw_parity_max'],
            protected_statistics_exact=manifest['protected_invariants'],
            changed_interior_timestamps=manifest['total_changed_timestamps'],physical_pair_matching=matching,
            files=dict(features=str(out/f'{domain}_features.npz'),metrics=str(directory/'metrics.json'),
                       predictions=str(directory/'predictions.npz'),source_manifest=str(out/f'{domain}_features.manifest.json')))
    final['interpretation']=dict(
        existing_E1='Unsupported at the preregistered two-reader 2% gate: full fails to beat unordered-Z sufficiently and MLP is worse than the simple control.',
        synthetic_E1='Controlled task support under known synthetic public geometry; TORE4 same-kernel is comparable or better. No independent new-statistic claim.',
        synthetic_time_destroy='Zero timestamps can change under the protected first/last/bin contract for these silhouette prefixes; this intervention supplies no evidence here.',
        E2='Exact raw-rebuild/requery parity passes. Task benefit is condition/head dependent, not uniformly >=2% in both readers.',
        correction_recommendation='If the one authorized correction is used, target readout/geometry conditioning after checking FIT optimization and SELECT capture range; do not tune tau, reset history, or claim the two-point statistic has passed.')
    M.dump(out/'summary.json',final)
    return final


if __name__=='__main__':
    summarize()
