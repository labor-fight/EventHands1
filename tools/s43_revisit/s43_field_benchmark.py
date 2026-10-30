"""Target-free Stage5 sensor-to-full-MANO resources on fixed native/noisy input.

No accuracy scores, labels or shape estimates enter this module. Formal GPU
measurements require finished training and exclusive service-device occupancy.
"""
from __future__ import annotations
import argparse,contextlib,functools,json,os,subprocess,time
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_info
from . import s43_field_replay as R
from tools.s43_rel.mean_shape_replay import MeanShapeInputReader,InferenceAccessGuard


class ResourceInput:
    """Mmap precomputed Nx4 source, bounded slices, no timestamp-column copy."""
    def __init__(self,prefix):
        self.prefix=Path(prefix);self.manifest=R.read(self.prefix.with_suffix('.json'))
        self.events=np.load(self.prefix.with_suffix('.events.npy'),mmap_mode='r',allow_pickle=False)
        self.offsets=np.load(self.prefix.with_suffix('.offsets.npy'),mmap_mode='r',allow_pickle=False)
        self.K_cam=np.asarray(self.manifest['K_source'],np.float32)
        self.runs=np.asarray([[0,self.manifest['stop_ms']]],np.int64)
        if self.events.shape!=(int(self.offsets[-1]),4) or self.events.dtype!=np.int64:raise ValueError('fixed resource input identity')
    def packet(self,a,b):
        lo,hi=int(self.offsets[a]),int(self.offsets[b])
        return dict(events=np.asarray(self.events[lo:hi]),source_range=(lo,hi))
    def input_paths(self):return [self.prefix.with_suffix(s) for s in ('.json','.events.npy','.offsets.npy')]


def prepare_resource_inputs(out,root=R.DATA_ROOT):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'native.json').exists() or (out/'noise30.json').exists():raise FileExistsError('precompute resource inputs once under a fresh output identity')
    with InferenceAccessGuard() as guard:
        reader=MeanShapeInputReader('ch_local',root);events=reader.packet(0,3300)['events']
        if len(events)!=247738:raise AssertionError('frozen ch3300 native count must be247738')
        source={str(p):R.sha(p) for p in reader.input_paths()}
        rng=np.random.default_rng(3407);n=int(np.floor(.3*len(events)))
        noise=np.column_stack((rng.integers(0,240,n),rng.integers(0,180,n),rng.integers(0,3300,n),rng.integers(0,2,n))).astype(np.int64)
        for name,data in (('native',events),('noise30',np.concatenate((events,noise)))):
            order=np.argsort(data[:,2],kind='stable');data=data[order]
            offsets=np.r_[0,np.cumsum(np.bincount(data[:,2],minlength=3300))].astype(np.int64)
            prefix=out/name;np.save(prefix.with_suffix('.events.npy'),data);np.save(prefix.with_suffix('.offsets.npy'),offsets)
            R.dump(prefix.with_suffix('.json'),dict(schema='s43_stage5_fixed_resource_stream_v1',stop_ms=3300,
                K_source=reader.K_cam.tolist(),native_count=len(events),noise_count=0 if name=='native' else n,total_events=len(data),seed=3407,
                noise='none' if name=='native' else 'floor(.30*native) independentuniformintegerxy/t/p sensor-noise; not physicalbackground',
                source_files=source,events_sha256=R.sha(prefix.with_suffix('.events.npy')),offsets_sha256=R.sha(prefix.with_suffix('.offsets.npy')),
                time_unit='ms',boundary_ms=0,input_shape='noGTbeta/publicbeta0',input_guard=guard.receipt()))
    return [str(out/n) for n in ('native','noise30')]


class PhaseProfiler:
    """Restored instance/module wrappers; identical kernels/values, sync wall."""
    def __init__(self,runner):self.runner=runner;self.records={};self.restores=[]
    def add(self,name,ms):
        category=self.runner.phase_category;self.records.setdefault(category+'/'+name,[]).append(float(ms))
    def wrap(self,obj,name,label,*,gpu_sync=True):
        old=getattr(obj,name)
        @functools.wraps(old)
        def wrapped(*a,**kw):
            if gpu_sync:R.sync(self.runner.runtime.device)
            start=time.perf_counter()
            try:return old(*a,**kw)
            finally:
                if gpu_sync:R.sync(self.runner.runtime.device)
                self.add(label,(time.perf_counter()-start)*1000)
        setattr(obj,name,wrapped);self.restores.append((obj,name,old))
    def __enter__(self):
        runtime=self.runner.runtime;model=self.runner.model;base=getattr(model,'base',model)
        original_as_tensor=torch.as_tensor;profiler=self
        def as_tensor(*a,**kw):
            target=kw.get('device');source=a[0] if a else kw.get('data')
            transfer=target is not None and torch.device(target).type=='cuda' and (not torch.is_tensor(source) or source.device.type=='cpu')
            if not transfer:return original_as_tensor(*a,**kw)
            R.sync(runtime.device);start=time.perf_counter()
            try:return original_as_tensor(*a,**kw)
            finally:
                R.sync(runtime.device);profiler.add('explicit_as_tensor_H2D_ms',(time.perf_counter()-start)*1000)
        torch.as_tensor=as_tensor;self.restores.append((torch,'as_tensor',original_as_tensor))
        if runtime.__class__.__name__ in ('SharedS43RuntimeR1','SharedS43RuntimeFastReadout'):
            from .s43_field_runtime import SharedS43Runtime
            self.wrap(SharedS43Runtime,'_ingest','sensor_maintenance_ms',gpu_sync=False)
        else:self.wrap(runtime,'_ingest','sensor_maintenance_ms',gpu_sync=False)
        self.wrap(base,'_render_prev','prior_render_full_FK_ms')
        self.wrap(base.conv1,'forward','stem_conv_ms');self.wrap(base.rn,'pooled','global_sparse_trunk_ms');self.wrap(base.rn.fc,'forward','global_head_ms')
        if getattr(base,'prevpos_embed',False):self.wrap(base.prev_mlp,'forward','previous_context_MLP_ms')
        if runtime.arm!='legacy_s43':
            if runtime.__class__.__name__=='SharedS43RuntimeFastReadout':
                from . import field_tensor_fast as T
                self.wrap(T,'tensorize_memories','integer_readout_H2D_log_ms')
                self.wrap(T,'aggregate_memories','integer_timebin_readout_CPU_ms',gpu_sync=False)
            else:
                from . import field_tensor as T
                self.wrap(T,'tensorize','integer_readout_H2D_log_ms')
                self.wrap(T.TimeFieldSnapshot,'rasterize','integer_timebin_readout_CPU_ms',gpu_sync=False)
            self.wrap(model.local_encoder,'forward','local_encoder_ms');self.wrap(model.local_head,'forward','local_head_ms')
            if hasattr(model,'prepare_queries_cpu'):
                from . import s43_field_model_reacquire as M
                self.wrap(model,'prepare_queries_cpu','query_geometry_and_conditions_D2H_ms')
                self.wrap(M,'sample_query_patches','local_sampling_ms')
                return self
            from . import s43_field_model as M
            self.wrap(M,'sample_query_patches','local_sampling_ms')
            original=M.FieldQueryBuilder;profiler=self
            class TimedBuilder:
                def __init__(self,q):self.q=q
                def build(self,*a,**kw):
                    R.sync(runtime.device);start=time.perf_counter()
                    try:return self.q.build(*a,**kw)
                    finally:
                        R.sync(runtime.device);profiler.add('query_geometry_build_ms',(time.perf_counter()-start)*1000)
            def builder(*a,**kw):
                R.sync(runtime.device);start=time.perf_counter();q=original(*a,**kw)
                R.sync(runtime.device);profiler.add('query_geometry_init_ms',(time.perf_counter()-start)*1000)
                # No per-query object retained in the restoration registry.
                return TimedBuilder(q)
            M.FieldQueryBuilder=builder;self.restores.append((M,'FieldQueryBuilder',original))
        return self
    def __exit__(self,*exc):
        for obj,name,old in reversed(self.restores):setattr(obj,name,old)
        return False
    def report(self):
        result={}
        for name,values in self.records.items():
            a=np.asarray(values,np.float64);result[name]=dict(n=len(a),total_ms=float(a.sum()),mean_ms=float(a.mean()),
                p50_ms=float(np.percentile(a,50)),p95_ms=float(np.percentile(a,95)),p99_ms=float(np.percentile(a,99)),max_ms=float(a.max()))
        return result


def assert_service_exclusive(device):
    if torch.device(device).type!='cuda':return dict(scope='CPUengineering/reference; no GPU service device')
    physical=int(os.environ['CUDA_VISIBLE_DEVICES'].split(',')[torch.device(device).index])
    uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader','-i',str(physical)],text=True).strip()
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    occupants=[int(row.split(',')[1].strip()) for row in rows.splitlines() if row.split(',')[0].strip()==uuid]
    foreign=[pid for pid in occupants if pid!=os.getpid()]
    if foreign:raise RuntimeError('formal service GPU is occupied; queue instead: '+repr(foreign))
    return dict(physical_gpu=physical,gpu_uuid=uuid,other_compute_processes_before=[])


def benchmark(arm,source,out,*,checkpoint=None,device='cpu',threads=2,hop=50,commit_ms=None,native_s43=False,
              engineering=False,train_root=R.TRAIN):
    out=Path(out)
    if out.exists():raise FileExistsError('fresh resource result path required')
    R.configure(device,threads)
    if threads!=2 and not engineering:raise ValueError('formal CPU2 contract')
    gate=None if engineering else R.all_weights_gate(train_root)
    if not engineering and arm!='canonical':
        final=Path(gate['checkpoints'][arm])
        if checkpoint is None:checkpoint=final
        if Path(checkpoint).resolve()!=final.resolve():raise ValueError('resource uses fixedlast complete weight')
    isolation=assert_service_exclusive(device) if not engineering else dict(engineering_not_formal_isolation=True)
    with InferenceAccessGuard() as guard:
        reader=ResourceInput(source)
        if not engineering and (reader.manifest.get('schema')!='s43_stage5_fixed_resource_stream_v1' or
                reader.manifest['stop_ms']!=3300 or reader.manifest['native_count']!=247738 or
                reader.manifest['noise_count'] not in (0,74321)):
            raise ValueError('formalresources use frozen native3300ms or fixed30percent stream only')
        for suffix,key in (('.events.npy','events_sha256'),('.offsets.npy','offsets_sha256')):
            if R.sha(reader.prefix.with_suffix(suffix))!=reader.manifest[key]:raise RuntimeError('precomputed resource input changed')
        model,payload=R.load_model(arm,checkpoint,device)
        files=R.source_files(reader,checkpoint,payload)+[Path(__file__).resolve(),Path(__file__).with_name('s43_field_operations.py')]
        before={str(p):R.sha(p) for p in files}
        R.dump(out.with_suffix('.sourcefreeze.json'),dict(schema='s43_stage5_resource_execution_source_v1',created_ns=time.time_ns(),
            engineering=engineering,allfive_weights_gate=gate,sources=before,command=dict(arm=arm,source=str(source),out=str(out),checkpoint=str(checkpoint),
                device=device,threads=threads,hop=hop,commit_ms=commit_ms or hop,native_s43=native_s43),isolation=isolation))
        if torch.device(device).type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        runner=R.FullStreamRunner(model,payload['runtime_contract'],native_s43=native_s43,hop=hop,commit_ms=commit_ms,audit_prefix=False)
        # First pass is the actual service path. No internal RN/geometry
        # wrappers or nested synchronization may remove CPU/GPU overlap.
        primary_timer=PhaseProfiler(runner);runner.phase=primary_timer.add
        surface_count=0;surface_h=__import__('hashlib').sha256()
        def observe(row,warm):
            nonlocal surface_count
            surface_count+=1;surface_h.update(row['vertices_m'].tobytes());surface_h.update(row['joints_m'].tobytes())
        try:rows,warm,summary=runner.run(reader,'ch_local_resource',on_output=observe)
        finally:runner.close()
        primary_surface_digest=surface_h.hexdigest();primary_surface_count=surface_count
        primary_peak_allocated=torch.cuda.max_memory_allocated(device) if torch.device(device).type=='cuda' else 0
        primary_peak_reserved=torch.cuda.max_memory_reserved(device) if torch.device(device).type=='cuda' else 0
        # A second independently reset run classifies components. These synced
        # phase costs are never substituted for first-pass service latency.
        detailed=R.FullStreamRunner(model,payload['runtime_contract'],native_s43=native_s43,hop=hop,commit_ms=commit_ms,audit_prefix=False)
        profiler=PhaseProfiler(detailed);detailed.phase=profiler.add
        surface_h=__import__('hashlib').sha256();surface_count=0
        try:
            with profiler:drows,dwarm,dsummary=detailed.run(reader,'ch_local_resource',on_output=observe)
        finally:detailed.close()
        all_a=warm+rows;all_b=dwarm+drows
        parity=(len(all_a)==len(all_b) and all(x['end_ms']==y['end_ms'] and np.array_equal(x['pred'],y['pred']) for x,y in zip(all_a,all_b))
            and primary_surface_digest==surface_h.hexdigest() and summary['native_input_sha256']==dsummary['native_input_sha256'])
        if not parity:raise AssertionError('component profiling changed actual51/fullsurface predictions or source')
        # A third independently reset pass counts executed Conv/Linear shapes.
        # Occupancy reductions/hook overhead do not enter either timing pass.
        from .s43_field_operations import NNOperations
        counted=R.FullStreamRunner(model,payload['runtime_contract'],native_s43=native_s43,hop=hop,commit_ms=commit_ms,audit_prefix=False)
        surface_h=__import__('hashlib').sha256();surface_count=0
        try:
            with NNOperations(counted) as operations:orows,owarm,osummary=counted.run(reader,'ch_local_resource',on_output=observe)
        finally:counted.close()
        ops_rows=owarm+orows
        ops_parity=(len(all_a)==len(ops_rows) and all(x['end_ms']==y['end_ms'] and np.array_equal(x['pred'],y['pred']) for x,y in zip(all_a,ops_rows))
            and primary_surface_digest==surface_h.hexdigest() and summary['native_input_sha256']==osummary['native_input_sha256'])
        if not ops_parity:raise AssertionError('operation counting changed51/fullsurface/source')
        operation_report=operations.report(dict(warmup=len(owarm),steady=len(orows)),complete=osummary['status']=='COMPLETE')
        operation_report.update(prediction_and_surface_and_source_exact=ops_parity,separate_third_pass=True,native_input_count=osummary['native_input_count'])
        after={str(p):R.sha(p) for p in files}
        if before!=after:raise RuntimeError('resource source changed during execution')
        values=np.asarray([r['cycle_ms'] for r in rows]);seconds=float(values.sum()/1000)
        warm_events=warm[-1]['native_prefix_count'] if warm else 0
        # A failed query may drain the rest of the stream without prediction.
        # Its tail events are never divided by successful-prefix service time.
        steady_events=rows[-1]['native_prefix_count']-warm_events if rows else 0
        params=sum(p.numel() for p in model.parameters());buffers=sum(p.numel()*p.element_size() for p in model.buffers())
        result=dict(schema='s43_stage5_sensor_to_fullmesh_resource_v1',**summary,arm=arm,engineering=engineering,device=device,threads=threads,
            precision='FP32eval/TF32off/renderFP32/stateFP32',checkpoint=str(checkpoint),source=str(source),source_files_unchanged=before==after,input_guard=guard.receipt(),
            artifact_status=payload.get('serving_status'),
            environment=dict(torch=torch.__version__,numpy=np.__version__,cuda_runtime=torch.version.cuda,
                CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),torch_threads=torch.get_num_threads(),
                actual_threadpools=threadpool_info(),thread_note='actual loadedBLAS/OpenMP pools; OPENBLAS env doesnot imply MKL1, torch.set_num_threads also affects sharedMKL',
                device_name=torch.cuda.get_device_name(device) if torch.device(device).type=='cuda' else 'CPU',
                OPENBLAS_NUM_THREADS=os.environ.get('OPENBLAS_NUM_THREADS'),OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS')),
            native_s43_without_unused_field=native_s43,shared_audit_only_field_cost=bool(payload['runtime_contract']['runtime_arm']=='legacy_s43' and not native_s43),
            phases=primary_timer.report(),phase_semantics='main E2E firstpass has no internal RN/query/sampling profiling wrappers. Cycle includes sourcepack+digest+contracthash+ingest/readout/H2D+geometry/render+NN+fullMANO778/21+D2H+commit; only external stage timers/sync. Output digest excluded.',
            component_profile=dict(separate_second_pass=True,prediction_and_surface_and_source_exact=parity,status=dsummary['status'],
                native_input_count=dsummary['native_input_count'],native_input_sha256=dsummary['native_input_sha256'],phases=profiler.report(),
                scope='inclusive nested synced phase timings for classification only; do notsum nested rows or substitute this instrumentedpass for mainlatency. CPU/GPU overlap may be reduced here. explicit_as_tensor_H2D covers those transfers only; small query-grid/flag.to transfers remain within local_sampling/net residual.'),
            throughput=dict(steady_service_seconds=seconds,steady_native_events=steady_events,events_per_second=steady_events/seconds if seconds else None,
                outputs_per_second=len(rows)/seconds if seconds else None,deadline_miss_fraction=float(np.mean(values>hop)) if len(values) else None,
                steady_cycle_mean_ms=float(values.mean()) if len(values) else None,
                steady_cycle_p50_ms=float(np.percentile(values,50)) if len(values) else None,steady_cycle_p95_ms=float(np.percentile(values,95)) if len(values) else None,
                steady_cycle_p99_ms=float(np.percentile(values,99)) if len(values) else None),
            memory=dict(model_parameter_count=params,model_parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),model_buffer_bytes=buffers,
                peak_resident_numeric_state_bytes=summary['peak_resident_numeric_state_bytes'],peak_RSS_bytes=summary['peak_RSS_bytes'],
                RSS_source='Linux getrusage(RUSAGE_SELF).ru_maxrss*1024 process highwater; includes priorload/firstpass state, not an exactisolated instantaneous allocatorpeak',
                torch_peak_allocated_bytes=primary_peak_allocated,torch_peak_reserved_bytes=primary_peak_reserved,
                output_surface_payload_bytes=9588,output_state_payload_bytes=204,
                input_mmap_and_temporary_packet_and_readout_and_Python_overhead='covered byprocessRSS andCUDAallocator; not asserted equal to persistentnumericstate'),
            full_surface_outputs_including_warmup=primary_surface_count,surface_output_sha256=primary_surface_digest,
            timing_exclusions='cold model/source loading, pre/post streaminghash, output JSON/NPZ serialization and output digest excluded fromcycle; input mmap/packing included. Prehash warms OS cache, not a coldI/O benchmark.',
            FLOPs=operation_report,
            isolation=isolation,isolation_after=assert_service_exclusive(device) if not engineering else None,
            sourcefreeze_sha256=R.sha(out.with_suffix('.sourcefreeze.json')))
        if summary['status']!='COMPLETE':
            result['partial_successful_prefix_costs_only']=result['throughput'];result['throughput']=None
            result['resource_failure_policy']='failedworkload; partialsuccessfulprefixcosts are diagnostic only; no completeflowlatency/throughput claimed and no hold/clamp'
        R.dump(out,result)
    return result


def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('prepare');a.add_argument('--out',required=True);a.add_argument('--root',default=str(R.DATA_ROOT))
    a=s.add_parser('run');a.add_argument('--arm',choices=R.MODELS,required=True);a.add_argument('--source',required=True);a.add_argument('--out',required=True)
    a.add_argument('--checkpoint');a.add_argument('--device',default='cpu');a.add_argument('--threads',type=int,default=2);a.add_argument('--hop',type=int,default=50);a.add_argument('--commit-ms',type=int)
    a.add_argument('--native-s43',action='store_true');a.add_argument('--engineering',action='store_true');a.add_argument('--train-root',default=str(R.TRAIN))
    args=vars(p.parse_args());c=args.pop('command')
    if c=='prepare':prepare_resource_inputs(**args)
    else:benchmark(**args)


if __name__=='__main__':main()
