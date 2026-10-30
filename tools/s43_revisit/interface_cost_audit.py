"""Read-only O0/C0 attribution of frozen training and measured whole-chain cost."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from .interface_audit import ROOT, OUT, OLD, sha, write, guards


def summarize(x):
    x=np.asarray(x,dtype=np.float64)
    return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),p95=float(np.quantile(x,.95)))


def main():
    torch.set_num_threads(1)
    protocol=OUT/'stage1_protocol.json';guards(protocol)
    bc=ROOT/'outputs/semkine/s43n_sp_s3407/config_resolved.yaml'
    basecfg=yaml.safe_load(bc.read_text())
    basepath=ROOT/'outputs/semkine/s43n_sp_s3407/s43n_sp_s3407-step=6000.ckpt'
    initial=torch.load(basepath,map_location='cpu',weights_only=False)['state_dict']
    reports={};logs={}
    for arm in ('s43','enhanced','no_order','full'):
        path=OLD/f'train/full_r1/{arm}/curve.jsonl'
        rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        logs[arm]=rows
        ckpt=OLD/f'train/full_r1/{arm}/step=6000.ckpt'
        payload=torch.load(ckpt,map_location='cpu',weights_only=False)
        state=payload['model_state_dict'];groups={}
        # Frozen-BN proof compares actual buffers, not only the configuration.
        bn=[k for k in initial if k.endswith(('running_mean','running_var','num_batches_tracked'))]
        for group,prefix in (('conv1','conv1.'),('trunk','rn.'),('previous_MLP','prev_mlp.')):
            names=[k for k,v in initial.items() if k.startswith(prefix) and v.dtype.is_floating_point and k not in bn and 'base.'+k in state]
            old=torch.cat([initial[k].reshape(-1).double() for k in names]);new=torch.cat([state['base.'+k].reshape(-1).double() for k in names])
            groups[group]=dict(numel=old.numel(),initial_norm=float(old.norm()),change_norm=float((new-old).norm()),relative_change=float((new-old).norm()/old.norm()))
        report=dict(curve_sha256=sha(path),checkpoint_sha256=sha(ckpt),steps=len(rows),outputs=rows[-1]['outputs'],
            loss_first100=summarize([r['loss'] for r in rows[:100]]),loss_last100=summarize([r['loss'] for r in rows[-100:]]),
            preclip_grad_norm=summarize([r['grad_norm'] for r in rows]),fraction_global_norm_above10=float(np.mean([r['grad_norm']>10 for r in rows])),
            actual_first_update_lr=.0001/500,logged_after_first_update_next_lr=rows[0]['lr'],last_logged_lr=rows[-1]['lr'],
            base_parameter_shifts=groups,bn_buffer_count=len(bn),bn_buffers_exactly_equal_to_initial=all(torch.equal(initial[k],state['base.'+k]) for k in bn))
        reports[arm]=report
    paired=all([r['step_source_digest'] for r in logs['s43']]==[r['step_source_digest'] for r in logs[a]] for a in reports)
    original=basecfg['TRAIN']
    O=dict(schema='s43_revisit_O0_frozen_training_audit_v1',protocol_sha256=sha(protocol),source_sha256=sha(__file__),
        base_checkpoint_sha256=sha(basepath),base_config_sha256=sha(bc),all_step_input_digests_paired=paired,
        canonical=dict(optimizer=original['OPTIMIZER'],LR=original['LR'],precision=original['PRECISION'],steps=original['MAX_STEPS'],effective_batch=original['BATCH_SIZE_PER_GPU']*original['ACCUMULATE_GRAD_BATCHES'],warmup=original['WARMUP_STEPS'],output_pairs=6000*512*2),
        paired_finetune=dict(optimizer='Adam',LR=.0001,precision='bf16',steps=6000,lane_batch=8,consecutive_windows_per_lane=4,effective_output_pairs_per_update=32,TBPTT_steps=1,warmup=500,cosine_floor_fraction=.02,global_grad_clip=10,BN_running_stats='frozen; affine parameters trainable',outputs=192000,original_output_pair_fraction=1/32,LR_ratio_to_original=1/40,LR_per_output_pair_ratio=.8),
        arms=reports,interpretation=[
            'All base parameters remain trainable and actually change. BN running buffers remain exactly initialized; this is a protocol difference from canonical training, not evidence it caused the result.',
            'BF16 and Adam are shared. Smaller LR, shorter new-data output budget, frozen BN statistics, consecutive correlated windows, own-feedback and gradient clipping are distinct factors; no one-factor causal experiment was run.',
            'Curve losses follow changing input windows and own-prediction histories. First/last loss summaries are not a matched-input optimization curve; the multiobjective log10 SO3/trans/FK loss is not identical to J0 risk.',
            'Nonzero gradients and parameter movement refute a completely disconnected branch or completely frozen optimizer; they do not establish adequate training. A failed gain alone does not diagnose optimization failure.',
            'The O1 4 recording prefixes were all in canonical/full-finetune training subjects. SELECT denotes diagnostic-head holdout only. These prefixes cannot identify unseen-domain overfit or select a checkpoint.'])
    write(OUT/'interface_training/results.json',O)
    measured=OLD/'mean_shape/resources/full_h50.json';d=json.loads(measured.read_text());records=d['records']
    components=('ingest_seconds','geometry_seconds','readout_seconds','representation_seconds','context_transfer_seconds','decode_including_observation_transfer_seconds','prediction_transfer_seconds','commit_seconds','output_mesh_total_seconds')
    account={k:summarize([r[k]*1000 for r in records]) for k in components}
    account['unattributed_overhead_ms']=summarize([(r['end_to_end_seconds']-sum(r[k] for k in components))*1000 for r in records])
    total=summarize([r['end_to_end_seconds']*1000 for r in records]);fastpath=OLD/'engineering/geometry_fast.json';f=json.loads(fastpath.read_text())
    C=dict(schema='s43_revisit_C0_existing_service_cost_v1',protocol_sha256=sha(protocol),source_sha256=sha(__file__),
        measured_receipt_sha256=sha(measured),measured_scope=d['scope'],input_kind=d['input_kind'],recording=d['recording_geometry_context'],queries=d['queries'],device=d['device'],beta=d['inference_beta'],
        total_ms=total,disjoint_components_ms=account,resources=d['resources'],
        fast_receipt_sha256=sha(fastpath),fast_backend=dict(costs=f['costs'],speedup=f['speedup'],exclusions=f['exclusions'],equivalence_cases=len(f['cases']),all_vertices_bitwise=all(x['forward_bitwise'] for x in f['cases']),maximum_action_J_difference_m=max(x['jacobian_reference_max_m'] for x in f['cases']),full_chain_acceleration_measured=False),
        source_complexity=[
            'C++ seal: for each timestamp and active destination-cell, visit d<=16 incoming edges; each edge updates 4*R*L=24 channels. A and Z are lazy-decayed at touched cells/edges; no whole Z scan per event.',
            'Python ingest first parses every event, groups timestamps and uses bincount(minlength=2*C) per timestamp. Seal active-cell extraction scans n.any(axis=1) over C. Thus the current wrapper has O(events + timestamps*C) allocation/scan overhead even though the mathematical update is local.',
            'Readout: selected Z edges O(E_query*4*R*L); include_A materializes C*2*R. Full production sensor representation and old LNES are dense grids; no per-query full Z copy is needed.',
            'Each new previous pose globally rebinds MANO FK/action Jacobian, visibility/silhouette, Ns128 sampling and nearby edges. Late binding is query-time geometry work, not event-local maintenance. Dense C is 51x51 and construction contracts Ns*51*51 plus channels as defined by the reader.',
            'The existing CNN is masked-dense periodic GPU execution; sparse mathematical support does not imply sparse device execution or asynchronous network updates. All masks derive from old input.',
            'Rolling raw-W50 has a separate configured cap8MiB; resident sensor arrays18,616,960B plus shared topology5,510,720B are independent of duration. Recording prediction/log artifacts grow with duration.'],
        exact_cache_boundaries=[
            'Static shape-dependent MANO terms may cache on exact beta plus model/source identity. Full binding requires previous51, beta, event-grid K, geometry config, topology and MANO asset/source versions; cached evidence/readout must additionally bind memory version/cutoff.',
            'Near-pi geometry uses actual Exp matrices and right-local action derivatives, without log branch selection. Equivalent matrix poses may differ as chart values; exact chart cache misses are safe. Beta/K/source change must invalidate relevant bindings.',
            'At query25ms with commit50ms, unchanged previous pose may permit repeated binding reuse. At commit=query50ms generally every previous pose changes; repeated hot-binding timings overstate attainable deployment cache hits.',
            'Legacy previous-pose render and geometry repeat FK of the same state, but use different numeric implementations; sharing requires projected/render/action parity. Output mesh FK evaluates the new predicted state and cannot be silently reused from the previous-state binding.',
            'A valid incremental CNN must invalidate old pixels on temporal renormalization, render/context changes, mask insertions/deletions and downstream nonlinear/pooling dependencies. Hot FK acceleration alone is not whole-service acceleration.'],
        interpretation='Existing measured receipt only, no new GPU/task run. Component sums are disjoint; do not add output_mesh_total to its children. The exact sparse-J optimization is local arithmetic evidence, not measured full-chain speed. A zero-geometry counterfactual is an arithmetic upper bound, not an achievable latency claim.')
    write(OUT/'cost_audit/results.json',C);guards(protocol)
    print(json.dumps({'training':reports,'total_ms':total,'components_ms':account,'sources_unchanged':True}))


if __name__=='__main__':main()
