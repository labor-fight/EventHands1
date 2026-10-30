"""Preregistered E2/E3 controls for the finite S43 B-R CPU prototype.

Only the first sample of the first three TRAIN contexts of each synthetic
axis is used. Targets are consumed exclusively by the scoring function.
Physical history perturbations explicitly restart their boundary prior; they
are not represented as exact coordinate rebasings or valid old-memory reuse.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import tracemalloc

import numpy as np

from .carrier import Event, EvidenceMemory, GridKernel, encode_batch, logsumexp
from .geometry import candidate_paths, fk_joints, render_paths
from .synthetic import CONFIG, fully_occluded_paths, load_synthetic, sample_counts

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/s43_br_20261027'
SENSOR = dict(ap=2, am=2, alpha=.9, nu=.02, cap=8)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (np.integer, np.floating)): return value.item()
    if isinstance(value, np.bool_): return bool(value)
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def write_exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, default=jsonable, allow_nan=False)
        f.write('\n')


def construct_memory(paths, counts, context_id):
    t0 = time.perf_counter()
    kernel = GridKernel(paths, context_id=context_id, **SENSOR)
    memory = EvidenceMemory(kernel)
    memory.counts[:] = counts
    built = time.perf_counter()
    memory.advance(len(counts) * 1000, watermark_us=len(counts) * 1000)
    advanced = time.perf_counter()
    result = memory.query()
    queried = time.perf_counter()
    if result['status'] != 'ok':
        raise ArithmeticError('unsupported controlled observation')
    return memory, result, {'kernel_and_memory_seconds': built-t0,
                            'advance_seconds': advanced-built,
                            'query_with_diagnostic_copies_seconds': queried-advanced,
                            'total_seconds': queried-t0,
                            'resident': memory.memory_bytes()}


def evaluate_only(result, candidate_states, target_joints, target_class, betas):
    """GT enters here only, after construction/refresh has already completed."""
    joints = fk_joints(candidate_states[:, -1], betas)
    all_errors = np.linalg.norm(joints-target_joints, axis=-1).mean(-1)*1000.
    choice = int(np.argmax(result['posterior']))
    p = np.asarray(result['posterior'])
    return {'choice': choice, 'correct_generating_class': choice == target_class,
            'posterior': p, 'absolute_joint_mm': float(all_errors[choice]),
            'oracle_candidate_mm': float(all_errors.min()),
            'nearest_candidate_index':int(all_errors.argmin()),
            'nll_nearest_candidate':float(-np.log(max(p[int(all_errors.argmin())],1e-300))),
            'nll_generating_class': float(-np.log(max(p[target_class], 1e-300))),
            'entropy': float(-(p*np.log(np.maximum(p, 1e-300))).sum())}


def geometry_controls(sample):
    axis = sample['axis']
    amplitudes = np.array([-.015, 0., .015]) if axis == 'root_x' else np.array([-.25, 0., .25])
    axis_name = 'tx' if axis == 'root_x' else 'index_y'
    context_id = 'diagnostic:' + sample['context_id']
    counts = np.asarray(sample['counts'])
    # Reconstruct all three perturbation groups through the same constructor.
    # Cached float32 histories are shared, no true candidate drives a trajectory.
    rows = []
    base_memory = base_result = base_states = base_paths = None
    for millimetres in (0., 5., 15.):
        history = np.asarray(sample['history'], dtype=np.float64).copy()
        history[0] += millimetres/1000.
        t0 = time.perf_counter()
        states = candidate_paths(history, amplitudes=amplitudes, axis=axis_name)
        paths = render_paths(states, sample['betas'], sample['K_cam'],
                             image_size=(24,32), texture_seed=7)
        render_seconds = time.perf_counter()-t0
        memory, fresh, timing = construct_memory(paths, counts, context_id)
        if millimetres == 0:
            base_memory, base_result, base_states, base_paths = memory, fresh, states, paths
        row = {'history_x_perturbation_mm': millimetres,
               'boundary_handling': 'explicit uniform-phase conditional restart; physical c0 changed',
               'coverage_warning':'generating index is only a response tag after physical history perturbation; true trajectory may lie outside bank',
               'fresh': evaluate_only(fresh, states, sample['target_joints'],
                                      sample['target_class'], sample['betas']),
               'stale_readout': evaluate_only(base_result, states, sample['target_joints'],
                                              sample['target_class'], sample['betas']),
               'max_log_likelihood_change_from_stale':
                   float(np.max(np.abs(fresh['log_likelihood']-base_result['log_likelihood']))),
               'render_candidate_seconds': render_seconds, 'construction': timing}
        rows.append(row)

    # Changing only numerical coordinates cannot change physical evidence.
    before = base_memory.digest()
    q1 = base_memory.query(np.zeros((3,51)))
    q2 = base_memory.query(np.ones((3,51)))
    assert before == base_memory.digest()
    assert np.array_equal(q1['log_likelihood'], q2['log_likelihood'])
    assert np.array_equal(q1['phi'], q2['phi'])

    # Consistent candidate re-numbering is an equivariance, not a geometry error.
    perm = np.array([2,0,1])
    permuted = encode_batch(base_paths[perm], counts, context_id=context_id, **SENSOR)
    permutation_error = float(np.max(np.abs(permuted['log_likelihood']-
                                            base_result['log_likelihood'][perm])))
    assert permutation_error < 1e-10

    # Only the window path changes; initial physical radiance is identical.
    t0 = time.perf_counter()
    refreshed_states = candidate_paths(sample['history'], amplitudes=amplitudes*.8,
                                       axis=axis_name)
    refreshed_paths = render_paths(refreshed_states, sample['betas'], sample['K_cam'],
                                   image_size=(24,32), texture_seed=7)
    render_refresh_seconds = time.perf_counter()-t0
    assert np.array_equal(refreshed_paths[...,0], base_paths[...,0])
    refreshed_kernel = GridKernel(refreshed_paths, context_id=context_id, **SENSOR)
    t0 = time.perf_counter()
    replayed_memory = base_memory.replay(refreshed_kernel)
    replayed = replayed_memory.query()
    replay_seconds = time.perf_counter()-t0
    t0 = time.perf_counter()
    full = encode_batch(refreshed_paths, counts, context_id=context_id, **SENSOR)
    full_seconds = time.perf_counter()-t0
    replay_error = float(np.max(np.abs(replayed['log_likelihood']-full['log_likelihood'])))
    phase_error = float(np.max(np.abs(replayed['phi']-full['phi'])))
    assert replay_error < 1e-10 and phase_error < 1e-12
    assert np.array_equal(replayed_memory.counts, base_memory.counts)
    return {'context': sample['context_id'], 'axis':axis,
            'sample':sample['sample_id'], 'split':'train', 'history_perturbations':rows,
            'pure_chart_evidence_invariant':True,
            'consistent_candidate_permutation_max_error':permutation_error,
            'refresh':{'boundary':'same physical c0 and initial radiance; window amplitudes multiplied by .8',
                       'max_log_likelihood_error_replay_vs_full':replay_error,
                       'max_phase_error_replay_vs_full':phase_error,
                       'render_seconds':render_refresh_seconds,'replay_seconds':replay_seconds,
                       'full_batch_seconds':full_seconds,
                       'memory':replayed_memory.memory_bytes(),
                       'refresh_count':replayed_memory.refresh_count,
                       'evaluation':evaluate_only(replayed,refreshed_states,sample['target_joints'],
                                                   sample['target_class'],sample['betas'])}}


def ambiguity_control(sample):
    # Same actual observation for all candidates; only path identity is removed.
    identical = np.broadcast_to(sample['paths'][0:1], sample['paths'].shape).copy()
    _, result, timing = construct_memory(identical, sample['counts'], 'identical-path-negative')
    error = float(np.max(np.abs(result['posterior']-1/3)))
    assert error < 1e-12
    # A separate fully occluded, physically supported finite-generator control.
    occluded = fully_occluded_paths(sample['paths'])
    counts = sample_counts(occluded, 0, 1, np.random.default_rng(20261027))
    _, occluded_result, occluded_timing = construct_memory(occluded, counts, 'full-occlusion-negative')
    occluded_error = float(np.max(np.abs(occluded_result['posterior']-1/3)))
    assert occluded_error < 1e-12
    return {'same_observation_identical_candidates':{'posterior':result['posterior'],
                'max_deviation_from_uniform':error,'construction':timing},
            'fully_occluded_generator':{'posterior':occluded_result['posterior'],
                'max_deviation_from_uniform':occluded_error,
                'construction':occluded_timing},
            'interpretation':'equal observation support preserves declared uniform finite prior; not global uncertainty calibration'}


def lifecycle_control(epochs=512):
    # A tiny fixed physical constant-radiance system isolates persistent lifecycle.
    paths = np.full((3,3,4,3), -16, dtype=np.int16)
    kernel = GridKernel(paths, context_id='bounded-lifecycle', **SENSOR)
    memory = EvidenceMemory(kernel, max_sources=8)
    resident=[]; arrays=[]
    timings={'ingest_seconds':0.,'advance_seconds':0.,'query_seconds':0.,'handoff_seconds':0.}
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    for epoch in range(epochs):
        start=epoch*2000
        events=[Event(epoch*2,start,0,0), Event(epoch*2+1,start+1000,1,1)]
        t=time.perf_counter(); memory.ingest(events); timings['ingest_seconds']+=time.perf_counter()-t
        # Retransmission of the same source is idempotent within the epoch.
        assert memory.ingest(events) == 0
        t=time.perf_counter(); memory.advance(start+2000,watermark_us=start+2000)
        timings['advance_seconds']+=time.perf_counter()-t
        digest=memory.digest()
        t=time.perf_counter()
        for _ in range(3): memory.query()
        timings['query_seconds']+=time.perf_counter()-t
        assert memory.digest()==digest
        info=memory.memory_bytes()
        resident.append(info['total_accounted_bytes'])
        arrays.append(info['array_bytes'])
        assert info['source_count']==2
        if epoch+1 < epochs:
            t=time.perf_counter()
            boundary=memory.branch_handoff(0)
            memory=EvidenceMemory.from_handoff(kernel,boundary,max_sources=8)
            timings['handoff_seconds']+=time.perf_counter()-t
            assert len(memory.seen)==0
    elapsed=time.perf_counter()-t0
    traced_current,traced_peak=tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # CPython's zero integer can occupy fewer bytes than a positive integer.
    # Epoch zero has exactly two otherwise absent zeros: uid and timestamp.
    # Account for this known object-size effect, then require exact stationarity.
    cold_object_size_difference=2*max(0,sys.getsizeof(1)-sys.getsizeof(0))
    assert len(set(arrays))==1
    assert len(set(resident[1:]))==1
    assert max(resident)-min(resident)<=cold_object_size_difference
    # A fixed observation with different ingest chunks/query density has same state.
    first=EvidenceMemory(kernel,max_sources=8); second=EvidenceMemory(kernel,max_sources=8)
    events=[Event(1,0,0,0),Event(2,1000,1,1)]
    first.ingest(events); first.advance(2000,watermark_us=2000)
    second.ingest(events[:1]); second.advance(1000,watermark_us=1000); second.query()
    second.ingest(events[1:]); second.advance(2000,watermark_us=2000)
    assert first.digest()==second.digest()
    return {'epochs':epochs,'epoch_ms':2,'source_cap':8,'max_sources_per_epoch':2,
            'carrier_accounted_bytes_min':min(resident),'carrier_accounted_bytes_max':max(resident),
            'carrier_array_bytes_all_epochs':arrays[0],
            'cold_zero_integer_object_size_bound_bytes':cold_object_size_difference,
            'warm_epoch_resident_bytes_exactly_constant':True,
            'carrier_array_and_source_memory_bounded':True,'query_read_only':True,
            'partition_and_query_density_digest_equal':True,'elapsed_seconds':elapsed,
            'timing_components':timings,'tracemalloc_current_bytes':traced_current,
            'tracemalloc_peak_bytes':traced_peak,
            'memory_scope':'carrier resident arrays/source account; traced process allocations include audit list, not a total RSS bound',
            'query_scope':'actual diagnostic snapshot query, including phase/weights copies',
            'timing_scope':'CPU replay; no sensor arrival latency claim',
            'handoff_scope':'conditional on fixed hard-selected branch; not full-history Bayes'}


def wrong_dof_control(sample):
    """One fixed wrong MANO degree of freedom, with output identities unchanged."""
    if sample['axis']!='index_y' or sample['context_id']!='index_y/context_00':
        raise ValueError('this control is fixed to index_y TRAIN context zero')
    # Original output candidates remain axis7. Only the assumed scoring paths
    # change to axis16. No anatomical name or searched-for worst axis is used.
    direction=np.zeros(51); direction[16]=1.
    t0=time.perf_counter()
    wrong_states=candidate_paths(sample['history'],amplitudes=(-.25,0.,.25),axis=direction)
    wrong_paths=render_paths(wrong_states,sample['betas'],sample['K_cam'],
                             image_size=(24,32),texture_seed=7)
    render_seconds=time.perf_counter()-t0
    _,correct,correct_timing=construct_memory(sample['paths'],sample['counts'],'correct-dof')
    _,wrong,wrong_timing=construct_memory(wrong_paths,sample['counts'],'wrong-dof')
    return {'sample':sample['sample_id'],'split':'train','correct_scoring_residual_axis':7,
            'wrong_scoring_residual_axis':16,'anatomical_identity_claim':False,
            'intervention':'change scoring geometry only; output candidate coordinates/identities remain original axis7',
            'fixed':{'counts_sha256':hashlib.sha256(sample['counts'].tobytes()).hexdigest(),
                     'output_candidate_states_sha256':hashlib.sha256(sample['candidate_states'].tobytes()).hexdigest(),
                     'history_sha256':hashlib.sha256(sample['history'].tobytes()).hexdigest(),
                     'amplitudes_rad':[-.25,0.,.25],'eta_scales':[.8,1.,1.2],
                     'texture_seed':7,'sensor':SENSOR},
            'correct':evaluate_only(correct,sample['candidate_states'],sample['target_joints'],
                                    sample['target_class'],sample['betas']),
            'wrong':evaluate_only(wrong,sample['candidate_states'],sample['target_joints'],
                                  sample['target_class'],sample['betas']),
            'max_log_likelihood_change':float(np.max(np.abs(correct['log_likelihood']-wrong['log_likelihood']))),
            'changed_path_entries':int(np.count_nonzero(wrong_paths-sample['paths'])),
            'wrong_geometry_render_seconds':render_seconds,
            'correct_construction':correct_timing,'wrong_construction':wrong_timing,
            'GT_role':'evaluation only; fixed wrong axis chosen before this readout, no axis search'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--synthetic',type=Path,default=OUT/'synthetic')
    p.add_argument('--out',type=Path,default=OUT/'diagnostics.json')
    p.add_argument('--wrong-dof-only',action='store_true')
    args=p.parse_args()
    if args.out.exists(): raise SystemExit(f'preserve existing {args.out}')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    torch.set_num_threads(1)
    sources={str(Path(__file__).relative_to(ROOT)):sha(__file__)}
    for name in ['carrier.py','geometry.py','synthetic.py','preregistration.md']:
        path=Path(__file__).with_name(name)
        sources[str(path.relative_to(ROOT))]=sha(path)
    receipt={'command':sys.argv,'python':sys.executable,'device':'cpu',
             'before_diagnostics_source_hashes':sources,
             'sample_selection':'first sample of TRAIN contexts 0,1,2 for each axis',
             'GT_role':'target score only, never constructor or perturbation selection',
             'protocol_sha256':sha(OUT/'protocol.json')}
    write_exclusive(args.out.with_name(args.out.stem+'_sources.json'),receipt)
    try:
        if args.wrong_dof_only:
            sample=load_synthetic(args.synthetic,axis='index_y',split='train')[0]
            result=wrong_dof_control(sample)
            write_exclusive(args.out,result)
            print(json.dumps({'out':str(args.out),'wrong_dof_control_complete':True}),flush=True)
            return
        samples=[]
        for axis in ('root_x','index_y'):
            rows=load_synthetic(args.synthetic,axis=axis,split='train')
            groups=[]
            for row in rows:
                if row['context_id'] not in groups:
                    groups.append(row['context_id']); samples.append(row)
                    if len(groups)==3: break
        results=[]
        started=time.perf_counter()
        for sample in samples:
            result=geometry_controls(sample)
            result['ambiguity']=ambiguity_control(sample)
            results.append(result)
            print(json.dumps({'completed':sample['context_id'],'elapsed_seconds':time.perf_counter()-started}),flush=True)
        output={'identity':'S43-BR-E2-E3-small-controls','scope':'synthetic train controls only; no existing/zgz outcomes',
                'samples':results,'lifecycle':lifecycle_control(),
                'elapsed_seconds':time.perf_counter()-started,
                'all_assertions_passed':True,'source_receipt':str(args.out.with_name(args.out.stem+'_sources.json'))}
        write_exclusive(args.out,output)
        print(json.dumps({'out':str(args.out),'all_assertions_passed':True,
                          'elapsed_seconds':output['elapsed_seconds']}),flush=True)
    except Exception as exc:
        write_exclusive(args.out.with_name(args.out.stem+'_failure.json'),
                        {'exception':repr(exc),'sources':sources})
        raise


if __name__=='__main__': main()
