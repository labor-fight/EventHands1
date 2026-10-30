"""E3 finite support contracts, actual prefix parity and CPU service accounting.

Formal moment examples, synthetic event grammars and existing hand_data51
prefixes are kept separate.  No factor inverse is interpreted as a posterior.
The real-prefix checks hold the deployed public query context fixed and are
not a full neural closed-loop service or new accuracy evaluation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import resource
import time
from types import SimpleNamespace

import numpy as np

from tools.s43_br.data import DATA_ROOT, PUBLIC_INITIAL_STATE, SequenceReader
from tools.s43_rel.core import RelConfig, RelationMemory
from tools.s43_rel.data import ContinuousRunLanes
from tools.s43_rel.geometry import GeometryConfig, ManoGeometry, provider_metadata

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_OUT=ROOT/'outputs/s43_rel_20261027/mechanisms/e3'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def digest(*arrays):
    h=hashlib.sha256()
    for a in arrays:
        a=np.ascontiguousarray(a)
        h.update(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes())
    return h.hexdigest()


def costs(values):
    a=np.asarray(values,dtype=np.float64)
    if not len(a):return {'n':0,'total':0.}
    return {'n':len(a),'total':float(a.sum()),'p50':float(np.median(a)),
            'p95':float(np.percentile(a,95)),'p99':float(np.percentile(a,99))}


def context():
    reader=SequenceReader('ch_local',DATA_ROOT)
    memory=RelationMemory(RelConfig(),seq_id='e3-public')
    geometry=ManoGeometry()
    binding=geometry.bind_source(PUBLIC_INITIAL_STATE,reader.betas,reader.K_cam,memory.topology)
    return reader,memory,geometry,binding


def formal_support(binding):
    """Analytic moment counterexamples, not realizable-raw equivalence claims."""
    i=int(np.argmax(np.linalg.norm(binding.a,axis=1)*binding.branch_regular))
    row=binding.a[i]
    if np.linalg.norm(row)<1e-10:raise RuntimeError('nonzero MANO action row required')
    rng=np.random.default_rng(3407)
    v=rng.normal(size=51)
    null=v-row*(row@v)/(row@row)
    null/=np.linalg.norm(null)
    xi=rng.normal(size=51)*.2
    k=np.array([-1.,1.]);mass=np.array([.5,.5])
    psi=lambda q:float(np.sum(mass*np.tanh(k*(row@q))))
    lam=np.zeros((len(binding.a),24));lam[i]=1.
    action_error=float(abs(row@null))
    gram_error=float(np.linalg.norm(binding.a.T@(lam[:,0]*(binding.a@null))))
    psi_error=abs(psi(xi+null)-psi(xi))
    # Use nonsymmetric evidence as well; a symmetric +/- example alone would
    # make Psi identically zero and would be a vacuous nullspace check.
    mass2=np.array([.2,.8])
    psi2=lambda q:float(np.sum(mass2*np.tanh(k*(row@q))))
    psi_error=max(psi_error,abs(psi2(xi+null)-psi2(xi)))
    # Same total relation mass, same deployed action row and zero first moment:
    # ideal tangent mass has no normal support, balanced opposite has support.
    collision={'total_mass_each':1.,'geometry_row_index':i,
               'ideal_tangent':{'kappa':[0.],'mass':[1.],'first':0.,'second':0.},
               'balanced_opposite':{'kappa':[-1.,1.],'mass':[.5,.5],'first':0.,'second':1.},
               'f_only_collision':True,'lambda_separates_this_example':True,
               'boundary':'ideal relation-moment construction, not matched physical raw streams'}
    ka=np.array([-1.,0.,1.]);ma=np.array([.3,.4,.3])
    kb=np.array([-.6,1.]);mb=np.array([.625,.375])
    moment={'A':{'kappa':ka.tolist(),'mass':ma.tolist()},
            'B':{'kappa':kb.tolist(),'mass':mb.tolist()},
            'first_difference':float(ma@ka-mb@kb),
            'second_difference':float(ma@(ka**2)-mb@(kb**2)),
            'Psi_difference_at_action_one':float(ma@np.tanh(ka)-mb@np.tanh(kb)),
            'interpretation':'f and lambda also collide; they do not preserve complete multimodality'}
    return {'evidence_kind':'formal finite-dimensional factor identities',
            'geometry_row_index':i,'action_row_norm':float(np.linalg.norm(row)),
            'null_action_residual':action_error,'factor_Gram_null_residual':gram_error,
            'Psi_null_invariance_residual':psi_error,
            'pass':bool(max(action_error,gram_error,psi_error)<1e-9),
            'f_only_collision':collision,'higher_moment_collision':moment,
            'physical_observability_claim':False,'calibrated_uncertainty_claim':False}


def _pair_events(topology,edge,kind,multiplicity=32):
    a,b=topology.centers_px[[topology.src[edge],topology.dst[edge]]].astype(np.int64)
    if kind=='forward':groups=[(5,a,multiplicity),(7,b,multiplicity)]
    elif kind=='reverse':groups=[(5,b,multiplicity),(7,a,multiplicity)]
    elif kind in ('balanced','static_flicker'):
        groups=[(5,a,multiplicity//2),(5,b,multiplicity//2),
                (7,a,multiplicity//2),(7,b,multiplicity//2)]
    else:raise ValueError(kind)
    return np.concatenate([np.tile([*xy,t,0],(n,1)) for t,xy,n in groups]).astype(np.int64)


def raw_support_cases(memory,binding):
    # Select the regular point whose normal best aligns to this *fixed* sensor
    # direction dictionary.  The measured tangent cosine is reported, not set
    # artificially to zero after quantization.
    best=None
    for point in np.flatnonzero(binding.branch_regular):
        links=np.flatnonzero(binding.point_index==point)
        if not len(links):continue
        nd=binding.normal_dot_delta[links]
        score=float(np.max(np.abs(nd)))
        if best is None or score>best[0]:best=(score,int(point),links)
    if best is None:raise RuntimeError('no regular supported contour point')
    _,point,links=best
    nlink=links[np.argmax(binding.normal_dot_delta[links])]
    tlink=links[np.argmin(np.abs(binding.normal_dot_delta[links]))]
    normal_edge=int(binding.edge_ids[binding.edge_slot[nlink]])
    tangent_edge=int(binding.edge_ids[binding.edge_slot[tlink]])
    rows=[]
    packs={}
    for name,edge,kind in [('normal_forward',normal_edge,'forward'),
                           ('normal_reverse',normal_edge,'reverse'),
                           ('tangent_grammar',tangent_edge,'forward'),
                           ('balanced_opposite',normal_edge,'balanced'),
                           ('static_flicker',normal_edge,'static_flicker')]:
        memory.reset('e3-synthetic',boundary_ms=0)
        ev=_pair_events(memory.topology,edge,kind)
        memory.ingest(ev,watermark_ms=10,packet_ordinal=0)
        snapshot=memory.readout(10,edge_ids=binding.edge_ids)
        pack=binding.read(snapshot)
        packs[name]=pack
        nd=(memory.topology.delta_px[edge]/np.linalg.norm(memory.topology.delta_px[edge]))@binding.n[point]
        rows.append({'name':name,'events':len(ev),'raw_sha256':digest(ev),
                     'total_polarity_counts':np.bincount(ev[:,3],minlength=2).tolist(),
                     'target_point_normal_dot_edge':float(nd),'f_norm':float(np.linalg.norm(pack['f'])),
                     'lambda_sum':float(pack['lam'].sum()),'support_points':int(pack['support'].sum()),
                     'target_point_lambda_sum':float(pack['lam'][point].sum())})
    forward,reverse=packs['normal_forward'],packs['normal_reverse']
    balanced,flicker=packs['balanced_opposite'],packs['static_flicker']
    return {'evidence_kind':'synthetic raw event grammar through real sensor core and MANO binding',
            'same_count_each':64,'same_query_geometry':True,'target_point':point,
            'normal_cosine':float(binding.normal_dot_delta[nlink]),
            'tangent_cosine':float(binding.normal_dot_delta[tlink]),'rows':rows,
            'reverse_f_antisymmetry_residual':float(np.max(np.abs(forward['f']+reverse['f']))),
            'reverse_lambda_symmetry_residual':float(np.max(np.abs(forward['lam']-reverse['lam']))),
            'balanced_f_norm':float(np.linalg.norm(balanced['f'])),
            'balanced_support_positive':bool(balanced['lam'].sum()>0),
            'flicker_balanced_identical_raw_and_package':bool(np.array_equal(balanced['f'],flicker['f']) and np.array_equal(balanced['lam'],flicker['lam'])),
            'limitations':['The grammars do not simulate hand radiance or verify a physical normal velocity.',
                           'Grid direction quantization makes tangent only approximate; its measured cosine is retained.',
                           'Balanced opposing propagation and static flicker can produce exactly these same tuples; lambda is not motion confidence.',
                           'All source counts and geometric routing are shared; event first/last marginal matching is not claimed.']}


def _packet(reader,a,b):
    return ContinuousRunLanes._packet(reader,int(a),int(b))


def actual_prefixes(reader,geometry,*,duration_ms=1000):
    a,b=next((int(a),int(b)) for a,b in reader.runs if b-a>=duration_ms)
    stop=a+duration_ms
    reference={};rows=[]
    final_digest=None
    for chunk in (1,7,50):
        for interval in (25,50,100):
            mem=RelationMemory(RelConfig(),seq_id='actual-prefix')
            mem.reset('actual-prefix',boundary_ms=a)
            cursor=a;ordinal=0;next_query=a+interval
            ingest_times=[];read_times=[];bind_times=[];contract_times=[]
            errors={'A':0.,'Z':0.,'f':0.,'lam':0.};equal=True
            while cursor<stop:
                end=min(cursor+chunk,next_query,stop)
                packet=_packet(reader,cursor,end)
                started=time.perf_counter()
                mem.ingest(packet['events'],watermark_ms=end,packet_ordinal=ordinal,
                           source_range=packet['source_range'])
                ingest_times.append(time.perf_counter()-started)
                cursor=end;ordinal+=1
                if cursor==next_query:
                    binding=geometry.bind_source(PUBLIC_INITIAL_STATE,reader.betas,reader.K_cam,mem.topology)
                    bind_times.append(binding.metadata['seconds']['total_bind'])
                    started=time.perf_counter();snapshot=mem.readout(cursor,edge_ids=binding.edge_ids)
                    read_times.append(time.perf_counter()-started)
                    started=time.perf_counter();pack=binding.read(snapshot)
                    contract_times.append(time.perf_counter()-started)
                    if (cursor-a)%100==0:
                        value={'A':snapshot.A,'Z':snapshot.Z,'f':pack['f'],'lam':pack['lam']}
                        if chunk==1 and interval==25:
                            reference[cursor]={k:v.copy() for k,v in value.items()}
                        for k,v in value.items():
                            error=float(np.max(np.abs(v-reference[cursor][k]),initial=0.))
                            errors[k]=max(errors[k],error)
                            equal &= bool(np.array_equal(v,reference[cursor][k]))
                    next_query+=interval
            full=mem.readout(stop)
            dh=digest(full.A,full.Z)
            if final_digest is None:final_digest=dh
            if dh!=final_digest:equal=False
            # Rebinding uses the same evidence prefix. A query change is not a
            # rewrite, reset, replay or a new feedback opportunity.
            before=dh
            perturbed=PUBLIC_INITIAL_STATE.copy();perturbed[0]+=.015
            moved=geometry.bind_source(perturbed,reader.betas,reader.K_cam,mem.topology)
            moved_pack=moved.read(mem.readout(stop,edge_ids=moved.edge_ids))
            after=mem.readout(stop)
            unchanged=before==digest(after.A,after.Z)
            rows.append({'chunk_ms':chunk,'query_interval_ms':interval,
                         'all_common_100ms_prefixes_bitwise_equal':equal,
                         'max_absolute_difference':errors,'final_full_A_Z_sha256':dh,
                         'prior_15mm_rebind_sensor_A_Z_unchanged':unchanged,
                         'rebound_geometry_version':moved.metadata['geometry_version'],
                         'memory':mem.stats(),
                         'seconds':{'ingest':costs(ingest_times),'global_geometry_rebind':costs(bind_times),
                                    'sparse_sensor_read':costs(read_times),'typed_contraction':costs(contract_times)}})
            if not equal or not unchanged:raise RuntimeError('actual prefix or geometry-rebind parity failed')
    return {'evidence_kind':'existing hand_data51 recording, native ms source bins',
            'recording':reader.recording,'start_ms':a,'stop_ms':stop,
            'events':int(reader.offsets[stop])-int(reader.offsets[a]),
            'rows':rows,'pass':True,'query_context':'public deployed initial, fixed; no labels/history teacher forcing',
            'scope':'common sensor/geometry core parity, not complete neural prediction parity',
            'source':{'base':str(reader.base),'raw_slice_source_range':[int(reader.offsets[a]),int(reader.offsets[stop])],
                      'raw_slice_sha256':digest(reader.events[int(reader.offsets[a]):int(reader.offsets[stop])]),
                      'offset_slice_sha256':digest(reader.offsets[a:stop+1])},
            'cost_exclusions':['network','commit','GPU transfer','complete causal predicted-state feedback'],
            'data_warning':'Existing hand_data51 streams; real-source file access does not establish real physical sensor provenance. No 1ms labels are used as high-frequency truth.'}


def synthetic_resources(reader,geometry):
    rows=[]
    rng=np.random.default_rng(3407)
    for rate in (10000,50000,150000):
        n=2*rate
        ev=np.column_stack((rng.integers(0,240,n),rng.integers(0,180,n),
                            np.sort(rng.integers(0,100,n)),rng.integers(0,2,n))).astype(np.int64)
        for interval in (25,50,100):
            mem=RelationMemory(RelConfig(),seq_id='resource')
            mem.reset('resource',boundary_ms=0)
            cursor=0;ordinal=0;ingest=[];bind=[];read=[];contract=[];service=[]
            rss_before=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
            started_all=time.perf_counter()
            for cutoff in range(interval,101,interval):
                lo=np.searchsorted(ev[:,2],cursor);hi=np.searchsorted(ev[:,2],cutoff)
                started=time.perf_counter()
                t=time.perf_counter();mem.ingest(ev[lo:hi],watermark_ms=cutoff,packet_ordinal=ordinal)
                ingest.append(time.perf_counter()-t)
                b=geometry.bind_source(PUBLIC_INITIAL_STATE,reader.betas,reader.K_cam,mem.topology)
                bind.append(b.metadata['seconds']['total_bind'])
                t=time.perf_counter();snapshot=mem.readout(cutoff,edge_ids=b.edge_ids)
                read.append(time.perf_counter()-t)
                t=time.perf_counter();b.read(snapshot);contract.append(time.perf_counter()-t)
                service.append(time.perf_counter()-started)
                cursor=cutoff;ordinal+=1
            total=time.perf_counter()-started_all
            rows.append({'events_per_50ms':rate,'horizon_ms':100,'events':n,'query_interval_ms':interval,
                         'seconds':{'ingest':costs(ingest),'global_geometry':costs(bind),
                                    'sparse_sensor_read':costs(read),'typed_contraction':costs(contract),
                                    'complete_representation_service':costs(service),'total':total},
                         'events_per_wall_second':n/total,'queries_per_wall_second':len(service)/total,
                         'memory':mem.stats(),'geometry_binding_array_bytes':b.metadata['binding_array_bytes'],
                         'spatial_index_data_bytes':b.metadata['static_spatial_index_data_bytes'],
                         'process_historical_peak_RSS_before_bytes':rss_before,
                         'process_historical_peak_RSS_after_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})
    return {'evidence_kind':'uniform synthetic workload, not actual recording-rate claims',
            'rows':rows,'query_context':'uncached global rebind from fixed public state every query',
            'cold_policy':'compiler/static constants/index warmed by previous formal and real-prefix cases',
            'cost_scope':'CPU memory+geometry representation service only; no neural network/commit/transfer/watermark waiting',
            'memory_limit':'NumPy resident arrays measured; process historical RSS is not per-case workspace peak; native tree allocator overhead not fully enumerated',
            'real_time_conclusion':'No real-time claim; compare each measured service duration to its declared query interval explicitly.'}


def existing_heads(cache,out,*,target_unit='mm'):
    """Optional fixed-budget task proper-head check on the E1 source cache.

    This measures task prediction under an explicit constant-variance Gaussian
    readout.  Existing data lack independently labelled physical ambiguity, so
    even a favourable NLL does not validate Gram calibration or raw observability.
    """
    plan={'schema':'s43_rel_e3_existing_heads_v1','source_sha256':sha(__file__),
          'cache_expected_schema':'E1 full[288,11112],context[288,72],delta[288,63] in mm,partition',
          'seed':3407,'feature_projection_dimension':64,'ridge_alpha':1.,
          'common_context':True,'standardization':'FIT only, per feature and per context coordinate',
          'variance':'FIT leave-one-out linear-hat residual scalar variance, lower bound .01mm2',
          'SELECT_tuning':False,'ambiguity_truth_available':False,
          'readout_limitation':'One fixed linear projected readout; failure does not rule out all factor adapters.',
          'written_before_cache_read':True}
    write(Path(out)/'existing_head_manifest.json',plan)
    archive=np.load(cache,allow_pickle=False)
    # The producer writes a schema manifest; fail instead of guessing channels
    # or learning partitions/units from SELECT values.
    required={'full','delta','context','partition'}
    if not required.issubset(archive.files):
        raise ValueError('existing cache needs explicit full,delta,context,partition fields; producer schema required')
    X=np.asarray(archive['full'],np.float64)
    Y=np.asarray(archive['delta'],np.float64).reshape(len(X),-1)
    C=np.asarray(archive['context'],np.float64)
    partition=archive['partition'].astype(str)
    fit=np.isin(partition,['FIT7','train']);select=np.isin(partition,['SELECT','validation'])
    if not fit.any() or not select.any() or X.shape[1]!=11112 or C.shape!=(len(X),72):
        raise ValueError('fixed E1 typed feature/partition schema mismatch')
    if target_unit not in ('m','mm'):raise ValueError('target units must be declared')
    Y=Y*(1000. if target_unit=='m' else 1.)
    groups={'sensor_only':np.arange(288),'f_only':np.arange(288,1512),
            'f_support':np.arange(288,11112),'full_typed':np.arange(11112)}
    results={};D=64;alpha=1.
    cm=C[fit].mean(0);cs=C[fit].std(0);cs=np.where(cs>1e-8,cs,1.)
    C=(C-cm)/cs
    for name,indices in groups.items():
        values=X[:,indices]
        mean=values[fit].mean(0);scale=values[fit].std(0)
        scale=np.where(scale>1e-8,scale,1.)
        values=(values-mean)/scale
        # Same learned coefficient count; fixed projection is chosen without Y.
        rng=np.random.default_rng(3407)
        projection=rng.choice([-1.,1.],size=(len(indices),D))/np.sqrt(len(indices))
        values=values@projection
        vm=values[fit].mean(0);vs=values[fit].std(0);vs=np.where(vs>1e-8,vs,1.)
        design=np.column_stack(((values-vm)/vs,C,np.ones(len(X))))
        F=design[fit];T=Y[fit]
        penalty=np.eye(design.shape[1])*alpha;penalty[-1,-1]=0.
        inverse=np.linalg.inv(F.T@F+penalty)
        weight=inverse@F.T@T
        pred=design@weight
        hat=np.einsum('ni,ij,nj->n',F,inverse,F)
        loo=(T-pred[fit])/np.maximum(1-hat,.01)[:,None]
        variance=max(float(np.mean(loo**2)),.01)
        err=pred[select]-Y[select]
        nll=.5*(np.log(2*np.pi*variance)+np.mean(err**2,axis=1)/variance)
        results[name]={'n_fit':int(fit.sum()),'n_SELECT':int(select.sum()),
                       'SELECT_mean_joint_delta_error_mm':float(np.linalg.norm(err.reshape(-1,21,3),axis=2).mean()),
                       'SELECT_Gaussian_NLL_per_coordinate':float(nll.mean()),
                       'fit_LOO_variance_mm2':variance,
                       'SELECT_coordinate_95coverage':float(np.mean(np.abs(err)<=1.95996398454*np.sqrt(variance))),
                       'learned_coefficients':int(weight.size),'FIT_effective_design_rank':int(np.linalg.matrix_rank(F))}
    receipt={'evidence_kind':'existing task-target readout, not independent ambiguity ground truth',
             'cache':str(cache),'sha256':sha(cache),'units_declared':target_unit,
             'head':'fixed 64D Rademacher feature projection + common 72D context, FIT standardization, ridge alpha1 + intercept; same coefficient count; isotropic Gaussian variance fitted by FIT LOO residuals',
             'SELECT_used_for_fitting_or_tuning':False,'results':results,
             'ambiguity_calibration':'unidentified: no independent physical ambiguity labels in this cache',
             'Gram_inverse_used':False,'one_seed_descriptive':True}
    write(Path(out)/'existing_heads.json',receipt)
    return receipt


def run(out=DEFAULT_OUT,*,resources=True,duration_ms=1000):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    reader,memory,geometry,binding=context()
    manifest={'source_sha256':sha(__file__),'core_sha256':sha(Path(__file__).with_name('core.py')),
              'geometry':provider_metadata(),'geometry_config':asdict(GeometryConfig()),
              'sensor_config':asdict(RelConfig()),'seed':3407,
              'selection':'fixed ch_local public context; no target or zgz used',
              'duration_ms':duration_ms,'resource_grid':{'events_per_50ms':[10000,50000,150000],
                                                        'query_ms':[25,50,100]},
              'write_before_readouts':True}
    write(out/'execution_manifest.json',manifest)
    formal=formal_support(binding);write(out/'formal_support.json',formal)
    synthetic=raw_support_cases(memory,binding);write(out/'synthetic_raw_support.json',synthetic)
    actual=actual_prefixes(reader,geometry,duration_ms=duration_ms);write(out/'actual_prefixes.json',actual)
    resource_report=synthetic_resources(reader,geometry) if resources else {'status':'skipped by explicit CLI'}
    write(out/'synthetic_resources.json',resource_report)
    summary={'formal_factor_contract_pass':formal['pass'],'actual_prefix_parity_pass':actual['pass'],
             'raw_balanced_keeps_nonzero_support':synthetic['balanced_support_positive'],
             'raw_static_flicker_collision':synthetic['flicker_balanced_identical_raw_and_package'],
             'claim':'Implementation/factor support contracts only. No task/ambiguity innovation is concluded.',
             'existing_proper_head':{'status':'pending independent E1 cache, optional --existing-cache',
                                    'ambiguity_truth_available':False},
             'resources':'CPU representation only, actual stream and synthetic grid reported separately',
             'files':['formal_support.json','synthetic_raw_support.json','actual_prefixes.json','synthetic_resources.json']}
    write(out/'summary.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    p.add_argument('--skip-resources',action='store_true');p.add_argument('--duration-ms',type=int,default=1000)
    p.add_argument('--existing-cache',type=Path);p.add_argument('--target-unit',choices=['m','mm'],default='mm')
    a=p.parse_args()
    result=existing_heads(a.existing_cache,a.out,target_unit=a.target_unit) if a.existing_cache else run(a.out,resources=not a.skip_resources,duration_ms=a.duration_ms)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
