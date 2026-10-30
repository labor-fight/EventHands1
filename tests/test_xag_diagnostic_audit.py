import copy,hashlib,json
from pathlib import Path
import numpy as np
import pytest
from tools.xag.assess_diagnostics import Audit,digest,summary,weighted_error,ROUTE_MODES,rollout_specs,ENDPOINT_METRICS


def make_audit(tmp_path):
 a=Audit.__new__(Audit);a.out=tmp_path;a.ph='p';a.code={'f':'h'};a.names=['s_global'];a.feature={};a.node_caps={3407:2048}
 a.p={'loss':{},'checkpoints':{'3407':{'checkpoint':'c','checkpoint_sha256':'ch','config':'cfg','config_sha256':'cfgh'}},'selected_packets':[]}
 a.expected={'s_global':{}}
 for i,split in enumerate(['fit','check']):
  row={'index':i,'segment':f's:{i}','end_ms':49+50*i,'previous_end_ms':50*i}
  item={'sequence':'s_global','split':split,'row':row,'donor_row':{'end_ms':49+50*(1-i)}}
  a.p['selected_packets'].append(item);a.expected['s_global'][('s_global',i,'clean')]=item
 return a


def identity(a,phase):
 return {**a.p['checkpoints']['3407'],'global_step':6000,'source_hashes':a.code,'protocol_sha256':a.ph,'phase':phase,'sequence':'s_global','limit':None}


def write(path,arrays,meta):
 np.savez(path,**arrays);meta['cache_sha256']=digest(path);path.with_suffix('.json').write_text(json.dumps(meta))


def feature(a,tmp_path):
 n=2;shapes={'H0':(n,778,128),'H3':(n,778,128),'O':(n,778),'S':(n,778),'slots':(n,17,258),'zs':(n,544),'bslot':(n,258),'prev48':(n,48),'incoming51':(n,51),'target51':(n,51),'prediction51':(n,51)}
 shapes.update({'wrong_'+k:shapes[k] for k in ['H0','H3','slots','zs']})
 arrays={k:np.zeros(s,np.float32) for k,s in shapes.items()};arrays['O'][:,0]=1;arrays['S'][:,0]=1
 rows=[]
 for i in range(n):
  item=a.expected['s_global'][('s_global',i,'clean')]
  rows.append({'sequence':'s_global','index':i,'split':item['split'],'segment':item['row']['segment'],'perturbation':'clean','end_ms':item['row']['end_ms'],'previous_end_ms':item['row']['previous_end_ms'],'donor_end_ms':item['donor_row']['end_ms'],'event_sha256':'0'*64})
 meta={'identity':identity(a,'extract'),'rows':rows,'failed_rows':0};p=tmp_path/'feature.npz';write(p,arrays,meta)
 return p,arrays,meta


def test_denominator_and_group_weights():
 assert summary([1,np.nan])=={'n':2,'n_failed':1,'mean':None}
 target=np.zeros((1,51));target[:,:3]=.01;target[:,3:6]=.2;target[:,6:]=.3
 assert np.allclose(weighted_error(np.zeros_like(target),target,{}),(30000*.01**2+60*.2**2+450*.3**2)/51)


def test_feature_identity_and_contract(tmp_path):
 a=make_audit(tmp_path);p,arrays,meta=feature(a,tmp_path)
 key,small=a.feature_one(3407,'s_global',p);assert key==(3407,'s_global') and small['std_count']==1
 arrays['prev48'][0,0]=1;write(p,arrays,meta)
 with pytest.raises(ValueError,match='incoming/prev48'):a.feature_one(3407,'s_global',p)
 arrays['prev48'][0,0]=0;arrays['H0'][0,0,0]=np.nan;write(p,arrays,meta)
 with pytest.raises(ValueError,match='nonfinite'):a.feature_one(3407,'s_global',p)


def test_reject_limit_and_hash_drift(tmp_path):
 a=make_audit(tmp_path);p,arrays,meta=feature(a,tmp_path)
 meta['identity']['limit']=1;write(p,arrays,meta)
 with pytest.raises(ValueError,match='partial'):a.feature_one(3407,'s_global',p)
 meta['identity']['limit']=None;write(p,arrays,meta);p.write_bytes(p.read_bytes()+b'changed')
 with pytest.raises(ValueError,match='SHA'):a.feature_one(3407,'s_global',p)


def fstar(a,tmp_path):
 fp,fa,fm=feature(a,tmp_path);key,f=a.feature_one(3407,'s_global',fp);a.feature[key]=f
 rows=[{**r,'steps':200,'radius':3.,'max_bound_violation':0.,'initial_loss':[0.],'best_loss':[0.]} for r in fm['rows']]
 std=np.zeros(128,np.float32)
 ident={**identity(a,'fstar'),'radius':3.,'steps':200,'std_sha256':hashlib.sha256(std.tobytes()).hexdigest()}
 meta={'identity':ident,'rows':rows,'failed_rows':0,'readout_replay_max_abs':0.,'train_std':{'std':std.tolist(),'n_supported_vertices':1,'sources':[{'path':str(fp),'sha256':digest(fp)}]}}
 arrays={k:np.zeros((2,51),np.float32) for k in ['incoming51','target51','original51','optimized51']};p=tmp_path/'fstar.npz';write(p,arrays,meta)
 return p,arrays,meta


def test_fstar_rejects_bound_and_candidate_regressions(tmp_path):
 a=make_audit(tmp_path);p,arrays,meta=fstar(a,tmp_path);a.fstar_one(3407,'s_global',p)
 meta['rows'][0]['max_bound_violation']=1e-8;write(p,arrays,meta);a.fstar_one(3407,'s_global',p)
 meta['rows'][0]['max_bound_violation']=1e-4;write(p,arrays,meta)
 with pytest.raises(ValueError,match='bound'):a.fstar_one(3407,'s_global',p)
 meta['rows'][0]['max_bound_violation']=0;meta['rows'][0]['best_loss']=[.1];write(p,arrays,meta)
 with pytest.raises(ValueError,match='candidate'):a.fstar_one(3407,'s_global',p)


def test_fstar_rejects_wrong_motion_target_and_support_std(tmp_path):
 a=make_audit(tmp_path);p,arrays,meta=fstar(a,tmp_path)
 arrays['target51'][0,0]=.01;write(p,arrays,meta)
 with pytest.raises(ValueError,match='target differs'):a.fstar_one(3407,'s_global',p)
 arrays['target51'][0,0]=0;meta['train_std']['n_supported_vertices']=2;write(p,arrays,meta)
 with pytest.raises(ValueError,match='support denominator'):a.fstar_one(3407,'s_global',p)


def rollout(a,tmp_path):
 name='zgz_local';rid=0;rows=[{'end_ms':49,'n_raw_events':10}];specs=rollout_specs(name,rid)
 a.bank={'sequences':{name:{'valid_runs_ms':[[0,100]]}}};a.source_manifest_sha='source'
 ident={**identity(a,'rollout'),'sequence':name,'run_id':rid,'source_manifest_sha256':'source'}
 arrays={'truth51':np.zeros((2,51),np.float32),'initial51':np.zeros((14,51),np.float32),'times_ms':np.array([0,49])}
 for m in ROUTE_MODES:
  for k,shape in [('own',(14,2,51)),('common',(14,1,51)),('gt',(1,51))]:arrays[f'{m}_{k}51']=np.zeros(shape,np.float32)
 physical={'passed':True,'records':[{'spec_id':s['id'],'checks':{'test':True}} for s in specs if s['block'] not in ['clean','natural']]}
 meta={'identity':ident,'specs':specs,'physical_checks':physical,'expected_own_endpoints':70,'expected_common_endpoints':70,'expected_gt_endpoints':5,'empty_packets':0,'failures':[],'route_audits':[{'mode':m,'step':0,**({} if m=='base' else {'xag_known':10,'xag_eligible':10,'xag_replaced':10,'xag_fallback':0})} for m in ROUTE_MODES],
 'scores':{m:{'own':[{'spec':s,'finite':True,'auc':{k:0. for k in ENDPOINT_METRICS}} for s in specs], 'common':{k:np.zeros((14,1)).tolist() for k in ENDPOINT_METRICS}, 'gt':{k:[0.] for k in ENDPOINT_METRICS}} for m in ROUTE_MODES}}
 p=tmp_path/'rollout.npz';write(p,arrays,meta);return name,rid,rows,p,arrays,meta


def test_rollout_retains_numerical_failure(tmp_path):
 a=make_audit(tmp_path);name,rid,rows,p,arrays,meta=rollout(a,tmp_path);a.rollout_one(3407,name,rid,rows,p)
 arrays['part_route_own51'][0,1]=np.nan;meta['scores']['part_route']['own'][0]['finite']=False;meta['scores']['part_route']['own'][0]['auc']={k:None for k in ENDPOINT_METRICS}
 meta['failures']=[{'mode':'part_route','kind':'own','trajectory':0,'step':0,'reason':'FloatingPointError'}];write(p,arrays,meta)
 _,value=a.rollout_one(3407,name,rid,rows,p);assert len(value['failures'])==1
 bad=next(g for g in value['groups'] if g['mode']=='part_route' and g['kind']=='own')
 assert summary(bad['values'])['n_failed']==1 and summary(bad['values'])['mean'] is None


def test_rollout_rejects_dropped_failure_and_program_error(tmp_path):
 a=make_audit(tmp_path);name,rid,rows,p,arrays,meta=rollout(a,tmp_path)
 arrays['part_route_own51'][0,1]=np.nan;meta['scores']['part_route']['own'][0]['finite']=False;meta['scores']['part_route']['own'][0]['auc']={k:None for k in ENDPOINT_METRICS};write(p,arrays,meta)
 with pytest.raises(ValueError,match='retained failures'):a.rollout_one(3407,name,rid,rows,p)
 meta['failures']=[{'mode':'part_route','kind':'own','trajectory':0,'step':0,'reason':'IndexError'}];write(p,arrays,meta)
 with pytest.raises(ValueError,match='implementation exception'):a.rollout_one(3407,name,rid,rows,p)


def test_route_counts_use_all_sampled_events_and_keep_raw_separate(tmp_path):
 a=make_audit(tmp_path);name,rid,rows,p,arrays,meta=rollout(a,tmp_path)
 rows[0]['n_raw_events']=6000
 for entry in meta['route_audits']:
  if entry['mode']=='base':continue
  entry.update(xag_known=1200,xag_eligible=800,xag_replaced=500 if 'part' in entry['mode'] else 800,xag_fallback=1548 if 'part' in entry['mode'] else 1248)
 write(p,arrays,meta);_,value=a.rollout_one(3407,name,rid,rows,p)
 rec=next(x for x in value['route_counts'] if x['mode']=='part_route')
 assert rec['raw_events']==6000 and rec['sampled_events']==2048
 assert rec['observed_counts']=={'known':1200,'eligible':800,'replaced':500,'fallback':1548,'unknown':848,'known_original_none':400,'candidate_missing':300}
 meta['route_audits'][1]['xag_fallback']=1;write(p,arrays,meta)
 with pytest.raises(ValueError,match='fallback'):a.rollout_one(3407,name,rid,rows,p)


def test_complete_probe_and_reject_wrong_initialization(tmp_path):
 import torch
 from tools.xag.assess_diagnostics import groups
 from tools.xag.probes import LayerProbe,tensor_hash
 a=make_audit(tmp_path);fp,_,_=feature(a,tmp_path);key,f=a.feature_one(3407,'s_global',fp);a.feature[key]=f
 torch.manual_seed(1701);head=LayerProbe('zs');initial=tensor_hash(head.state_dict());a.initial_hashes={(1701,'zs'):initial}
 orders=np.zeros((300,1),np.int64)
 ident={'source_hashes':a.code,'protocol_sha256':a.ph,'model_seed':3407,'head_seed':1701,'interface':'zs','control':'real','epoch':300,
 'extraction_sources':a.code,'feature_sources':[{'path':str(fp),'sha256':digest(fp)}],'orders_sha256':hashlib.sha256(orders.tobytes()).hexdigest()}
 with torch.no_grad():delta=head(torch.zeros(2,17,258),torch.zeros(2,258),torch.zeros(2,48)).numpy()
 error=weighted_error(delta,np.zeros_like(delta),{})
 metrics={g:{'n':int(m.sum()),'n_failed':0,'weighted51_mse':float(error[m].mean())} for g,m in groups(f['rows'])}
 curve=[{'epoch':i,'fit_weighted51_mse':0.} for i in range(25,301,25)]
 path=tmp_path/'probe.npz';statepath=path.with_suffix('.pt')
 checkpoint={'identity':ident,'completed_epoch':300,'initial_sha256':initial,'curve':curve,'state_dict':head.state_dict()};torch.save(checkpoint,statepath)
 meta={'identity':ident,'selection':'fixed epoch300','rows':f['rows'],'head_sha256':digest(statepath),'initial_sha256':initial,'curve':curve,'parameters':sum(v.numel() for v in head.parameters()),'metrics':metrics}
 arrays={'delta51':delta,'incoming51':f['incoming51'],'target51':f['target51'],'target_delta51':f['target51']-f['incoming51']};write(path,arrays,meta)
 a.probe_one(3407,1701,'zs','real',path)
 a.initial_hashes[1701,'zs']='wrong'
 with pytest.raises(ValueError,match='registered head seed'):a.probe_one(3407,1701,'zs','real',path)


def failure_safe_fstar(a,tmp_path):
 p,arrays,meta=fstar(a,tmp_path)
 a.execution_contract=lambda:{'execution_sha256':'ep','implementation_sha256':'impl','allowed_batch_sizes':[1]}
 a.failure_contract=lambda:{'policy_sha256':'pol','driver_sha256':'drv'}
 meta['identity'].update(optimization_execution_protocol_sha256='ep',optimization_implementation_sha256='impl',optimization_batch_size=1,optimization_failure_policy_sha256='pol',optimization_driver_sha256='drv')
 for row in meta['rows']:
  row.update(completed_steps=200,optimization_failed=False,failure_reason=None,fallback_original=False,numerical_batch_retry=False,actual_optimization_batch_size=1)
 meta.update(optimization_failures=[],optimization_failed_rows=0,readout_replay_checked_rows=2,readout_replay_nonfinite_rows=0)
 write(p,arrays,meta)
 return p,arrays,meta


def test_fstar_finite_fallback_is_failure_not_reachability(tmp_path):
 a=make_audit(tmp_path);p,arrays,meta=failure_safe_fstar(a,tmp_path);a.fstar_one(3407,'s_global',p)
 meta['rows'][0].update(optimization_failed=True,failure_reason='FloatingPointError',fallback_original=True,numerical_batch_retry=True)
 meta.update(optimization_failed_rows=1,optimization_failures=[{'row':0,'reason':'FloatingPointError','completed_steps':200,'finite_original':True}]);write(p,arrays,meta)
 _,value=a.fstar_one(3407,'s_global',p)
 assert value['failed'].tolist()==[True,False] and value['final_nonfinite'].tolist()==[False,False]
 assert value['numerical_batch_retries']==1 and summary(value['after'])=={'n':2,'n_failed':1,'mean':None}
 assert value['failures'][0]['finite_fallback'] is True
 meta['optimization_failed_rows']=0;write(p,arrays,meta)
 with pytest.raises(ValueError,match='failure denominator'):a.fstar_one(3407,'s_global',p)


def test_fstar_capture_failure_retains_nonfinite_endpoint(tmp_path):
 a=make_audit(tmp_path);p,arrays,meta=failure_safe_fstar(a,tmp_path)
 arrays['original51'][0]=np.nan;arrays['optimized51'][0]=np.nan
 meta['rows'][0].update(completed_steps=0,optimization_failed=True,failure_reason='FloatingPointError',fallback_original=True,actual_optimization_batch_size=0,max_bound_violation=None,initial_loss=[None],best_loss=[None])
 meta.update(failed_rows=1,optimization_failed_rows=1,optimization_failures=[{'row':0,'reason':'FloatingPointError','completed_steps':0,'finite_original':False}],readout_replay_checked_rows=1);write(p,arrays,meta)
 _,value=a.fstar_one(3407,'s_global',p)
 assert value['final_nonfinite'].tolist()==[True,False] and value['failures'][0]['finite_fallback'] is False
 meta['rows'][0]['failure_reason']='IndexError';write(p,arrays,meta)
 with pytest.raises(ValueError,match='program error'):a.fstar_one(3407,'s_global',p)


def test_fstar_rejects_fabricated_fallback_and_policy(tmp_path):
 a=make_audit(tmp_path);p,arrays,meta=failure_safe_fstar(a,tmp_path)
 meta['rows'][0].update(optimization_failed=True,failure_reason='FloatingPointError',fallback_original=True)
 meta.update(optimization_failed_rows=1,optimization_failures=[{'row':0,'reason':'FloatingPointError','completed_steps':200,'finite_original':True}]);arrays['optimized51'][0,0]=.001;write(p,arrays,meta)
 with pytest.raises(ValueError,match='exact original'):a.fstar_one(3407,'s_global',p)
 arrays['optimized51'][0,0]=0;meta['identity']['optimization_driver_sha256']='changed';write(p,arrays,meta)
 with pytest.raises(ValueError,match='driver identity'):a.fstar_one(3407,'s_global',p)
